//! Tiled grouped-query attention with online softmax and deterministic split-KV.
//! Each key/value row is read once for a tile of query positions and GQA heads.
use half::f16;

#[derive(Clone)]
struct Tile {
    first: usize,
    rows: usize,
    length: usize,
}
struct Partial {
    max: Vec<f32>,
    sum: Vec<f32>,
    acc: Vec<f32>,
}
#[repr(C, align(32))]
#[derive(Clone, Default)]
pub(crate) struct QueryBlock {
    pub q: [i8; 32],
    pub d: f32,
    pub init: i32,
}

#[derive(Default)]
struct Q8AttentionBuffers {
    scores: Vec<f32>,
    values: Vec<f32>,
}

thread_local! {
    static Q8_ATTENTION_BUFFERS: std::cell::RefCell<Q8AttentionBuffers> = const {
        std::cell::RefCell::new(Q8AttentionBuffers {
            scores: Vec::new(),
            values: Vec::new(),
        })
    };
}

/// Owns the per-thread scratch while one Q8 attention work item is active.
/// Taking the vectors leaves TLS unborrowed and lets nested calls own distinct
/// buffers; Drop returns the largest buffers to the current thread.
struct Q8AttentionScratch(Q8AttentionBuffers);
impl Q8AttentionScratch {
    fn take() -> Self {
        let buffers = Q8_ATTENTION_BUFFERS.with(|slot| {
            let mut retained = slot.borrow_mut();
            Q8AttentionBuffers {
                scores: std::mem::take(&mut retained.scores),
                values: std::mem::take(&mut retained.values),
            }
        });
        Self(buffers)
    }
}
impl Drop for Q8AttentionScratch {
    fn drop(&mut self) {
        let mut buffers = std::mem::take(&mut self.0);
        let _ = Q8_ATTENTION_BUFFERS.try_with(|slot| {
            let mut retained = slot.borrow_mut();
            if buffers.scores.capacity() > retained.scores.capacity() {
                retained.scores = std::mem::take(&mut buffers.scores);
            }
            if buffers.values.capacity() > retained.values.capacity() {
                retained.values = std::mem::take(&mut buffers.values);
            }
        });
    }
}

#[derive(Clone, Copy)]
pub struct Cache<'a> {
    pub data: &'a [u16],
    pub slots: usize,
    pub kv_heads: usize,
    pub dim: usize,
    pub q8: bool,
}
impl<'a> Cache<'a> {
    fn half_row(&self, slot: usize, h: usize) -> &'a [u16] {
        let start = (slot * self.kv_heads + h) * self.dim;
        &self.data[start..start + self.dim]
    }
    fn quant_row(&self, slot: usize, h: usize) -> (&'a [i8], &'a [u16]) {
        let count = self.slots * self.kv_heads * self.dim;
        // SAFETY: the public entry validates cache length. u16 storage is contiguous
        // and x86 is little endian; byte views neither alias mutable storage
        // nor outlive its borrow. Generic/non-x86 score code reads these too.
        let bytes = unsafe { std::slice::from_raw_parts(self.data.as_ptr().cast::<i8>(), count) };
        let start = (slot * self.kv_heads + h) * self.dim;
        let groups = self.dim / 32;
        let scale_start = count.div_ceil(2) + (slot * self.kv_heads + h) * groups;
        (
            &bytes[start..start + self.dim],
            &self.data[scale_start..scale_start + groups],
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn attention(
    q: &[f32],
    key: Cache<'_>,
    value: Cache<'_>,
    tables: &[i64],
    lengths: &[i64],
    heads: usize,
    block_size: usize,
    stride: usize,
) -> Option<Vec<f32>> {
    attention_with(q, key, value, tables, lengths, heads, block_size, stride, &Options::default())
}

/// What a layer changes in [`attention`]: the score scale (default `1 / sqrt(dim)`) and a sliding window (row `b` sees the last `window`
/// positions of its `lengths[b]`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    pub scale: Option<f32>,
    pub window: Option<usize>,
}

/// [`attention`] with a score scale and a sliding window.
#[allow(clippy::too_many_arguments)]
pub fn attention_with(
    q: &[f32],
    key: Cache<'_>,
    value: Cache<'_>,
    tables: &[i64],
    lengths: &[i64],
    heads: usize,
    block_size: usize,
    stride: usize,
    options: &Options,
) -> Option<Vec<f32>> {
    let batch = lengths.len();
    let dim = key.dim;
    let kv_heads = key.kv_heads;
    if cfg!(target_endian = "big")
        || dim == 0
        || !dim.is_multiple_of(32)
        || dim > 256
        || kv_heads == 0
        || !heads.is_multiple_of(kv_heads)
        || heads / kv_heads > 16
        || options.window == Some(0)
    {
        return None;
    }
    // first position row b attends to
    let start_of = |b: usize| options.window.map_or(0, |w| (lengths[b] as usize).saturating_sub(w));
    assert_eq!(q.len(), batch * heads * dim);
    assert_eq!(tables.len(), batch * stride);
    assert!(block_size > 0 && stride > 0);
    assert!(
        value.dim == dim
            && value.kv_heads == kv_heads
            && value.slots == key.slots
            && value.q8 == key.q8
    );
    let words = key.slots * kv_heads * dim;
    let words = if key.q8 {
        words.div_ceil(2) + key.slots * kv_heads * (dim / 32)
    } else {
        words
    };
    assert!(key.data.len() >= words && value.data.len() >= words);
    for (r, &length) in lengths.iter().enumerate() {
        assert!(length > 0 && length as usize <= stride * block_size);
        for &b in &tables[r * stride..r * stride + (length as usize).div_ceil(block_size)] {
            assert!(b >= 0 && (b as usize + 1) * block_size <= key.slots);
        }
    }
    let g = heads / kv_heads;
    let mut tiles = Vec::new();
    let mut row_tiles = vec![0; batch];
    let mut first = 0;
    while first < batch {
        let mut rows = 1;
        while rows < 16 / g
            && first + rows < batch
            && lengths[first + rows] == lengths[first + rows - 1] + 1
        {
            let n = (lengths[first + rows] as usize).div_ceil(block_size);
            if tables[first * stride..first * stride + n]
                != tables[(first + rows) * stride..(first + rows) * stride + n]
            {
                break;
            }
            rows += 1;
        }
        row_tiles[first..first + rows].fill(tiles.len());
        tiles.push(Tile {
            first,
            rows,
            length: lengths[first + rows - 1] as usize,
        });
        first += rows;
    }
    let blocks = dim / 32;
    let path = Path::detect();
    #[cfg(target_arch = "x86_64")]
    let fast = crate::cpu_kernels::isa() != "scalar";
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    let fast = crate::cpu_kernels::neon();
    let packed = if key.q8 {
        q.as_chunks::<32>()
            .0
            .iter()
            .map(|x| {
                #[cfg(target_arch = "x86_64")]
                if fast {
                    let mut query = QueryBlock::default();
                    // SAFETY: AVX2/F16C confirmed (`fast`); `x` is one 32-value block.
                    let (d, _, sums) =
                        unsafe { crate::cpu_kernels::x86::pack_q8_block(x, &mut query.q, true) };
                    query.d = d;
                    query.init = -16 * (sums[0] + sums[1]);
                    return query;
                }
                #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
                if fast {
                    let mut query = QueryBlock::default();
                    // SAFETY: NEON detected (`fast`); `x` is one 32-value block.
                    let (d, _, sums) =
                        unsafe { crate::cpu_kernels::neon::pack_q8_block(x, &mut query.q, true) };
                    query.d = d;
                    query.init = -16 * (sums[0] + sums[1]);
                    return query;
                }
                let amax = x.iter().map(|v| v.abs()).fold(0.0, f32::max);
                let d = amax / 127.0;
                let inv = if d == 0.0 { 0.0 } else { d.recip() };
                let mut qb = QueryBlock {
                    d: f16::from_f32(d).to_f32(),
                    ..Default::default()
                };
                for (a, &b) in qb.q.iter_mut().zip(x) {
                    *a = (b * inv).round_ties_even() as i8;
                }
                qb.init = -16 * qb.q.iter().map(|&v| v as i32).sum::<i32>();
                qb
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let base = tiles.len() * kv_heads;
    let max_len = tiles.iter().map(|t| t.length).max().unwrap_or(0);
    let segments = (rayon::current_num_threads() * 4)
        .div_ceil(base.max(1))
        .max(1)
        .min((max_len / 256).max(1));
    let scale = options.scale.unwrap_or_else(|| (dim as f32).sqrt().recip());
    let parts = crate::cpu_spin::collect(base * segments, |it| {
        let ti = tiles.len() - 1 - it / (kv_heads * segments);
        let kh = (it / segments) % kv_heads;
        let seg = it % segments;
        let tile = &tiles[ti];
        // the rows of a tile are consecutive positions: the first one has the earliest window start
        let lo = start_of(tile.first) / 64 * 64;
        let span = (tile.length - lo).div_ceil(segments).div_ceil(64) * 64;
        let end = tile.length.min(lo + (seg + 1) * span);
        let nr = tile.rows * g;
        let mut p = Partial {
            max: vec![f32::NEG_INFINITY; nr],
            sum: vec![0.0; nr],
            acc: vec![0.0; nr * dim],
        };
        let mut q8_scratch = key.q8.then(Q8AttentionScratch::take);
        let mut scores_local = if key.q8 {
            Vec::new()
        } else {
            vec![0.0; nr * 64]
        };
        let mut values_local = if key.q8 {
            Vec::new()
        } else {
            vec![0.0; 64 * dim]
        };
        let (scores, values) = if let Some(scratch) = q8_scratch.as_mut() {
            scratch.0.scores.resize(nr * 64, 0.0);
            scratch.0.values.resize(64 * dim, 0.0);
            (&mut scratch.0.scores, &mut scratch.0.values)
        } else {
            (&mut scores_local, &mut values_local)
        };
        let mut query_rows = [&[][..]; 16];
        if key.q8 {
            debug_assert!(nr <= query_rows.len());
            for (r, query_row) in query_rows.iter_mut().take(nr).enumerate() {
                let b = tile.first + r / g;
                let h = kh * g + r % g;
                *query_row = &packed[(b * heads + h) * blocks..][..blocks];
            }
        }
        for kb in (lo + seg * span..end).step_by(64) {
            let nk = (end - kb).min(64);
            for j in 0..nk {
                let t = kb + j;
                let slot = tables[tile.first * stride + t / block_size] as usize * block_size
                    + t % block_size;
                let (kq, ks) = if key.q8 {
                    key.quant_row(slot, kh)
                } else {
                    (&[][..], &[][..])
                };
                if key.q8 {
                    score_q8_tile(&query_rows[..nr], kq, ks, &mut scores[j..], scale, path);
                }
                for r in 0..nr {
                    let b = tile.first + r / g;
                    let h = kh * g + r % g;
                    scores[r * 64 + j] = if t >= lengths[b] as usize || t < start_of(b) {
                        f32::NEG_INFINITY
                    } else if key.q8 {
                        scores[r * 64 + j]
                    } else {
                        score_half(
                            &q[(b * heads + h) * dim..][..dim],
                            key.half_row(slot, kh),
                            path,
                        ) * scale
                    };
                }
                let dst = &mut values[j * dim..(j + 1) * dim];
                if value.q8 {
                    let (v, s) = value.quant_row(slot, kh);
                    decode_q8(v, s, dst, path);
                } else {
                    decode_half(value.half_row(slot, kh), dst, path);
                }
            }
            for r in 0..nr {
                let row = &mut scores[r * 64..r * 64 + nk];
                let bmax = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                if bmax == f32::NEG_INFINITY {
                    row.fill(0.0);
                    continue;
                }
                let m = p.max[r].max(bmax);
                let correction = (p.max[r] - m).exp();
                // Masked positions must stay zero even after the exponent clamp.
                let valid = (lengths[tile.first + r / g] as usize)
                    .saturating_sub(kb)
                    .min(nk);
                let denominator = crate::cpu_elementwise::exp_shifted(&mut row[..valid], m);
                row[valid..].fill(0.0);
                p.sum[r] = p.sum[r] * correction + denominator;
                p.max[r] = m;
                let dst = &mut p.acc[r * dim..(r + 1) * dim];
                for a in dst.iter_mut() {
                    *a *= correction;
                }
                accumulate(dst, row, &values[..nk * dim], path);
            }
        }
        p
    });
    let mut output = vec![0.0; q.len()];
    crate::cpu_spin::chunks_mut(&mut output, dim, |bh, out| {
        let b = bh / heads;
        let h = bh % heads;
        let ti = row_tiles[b];
        let tile = &tiles[ti];
        let r = (b - tile.first) * g + h % g;
        let base = ((tiles.len() - 1 - ti) * kv_heads + h / g) * segments;
        let max = parts[base..base + segments]
            .iter()
            .map(|p| p.max[r])
            .fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0.0;
        for p in &parts[base..base + segments] {
            if p.sum[r] == 0.0 {
                continue;
            }
            let w = (p.max[r] - max).exp();
            total += w * p.sum[r];
            for (o, &a) in out.iter_mut().zip(&p.acc[r * dim..(r + 1) * dim]) {
                *o += a * w;
            }
        }
        if total > 0.0 {
            for x in out {
                *x /= total;
            }
        }
    });
    Some(output)
}

/// Kernel path of the attention inner loops, detected once per call (the per-key helpers used
/// to probe CPU features on every invocation).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Path {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    AvxVnni,
    #[cfg(target_arch = "x86_64")]
    Avx512Bw,
    #[cfg(target_arch = "x86_64")]
    Avx512Vnni,
    /// NEON with dotprod.
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    Neon,
}
impl Path {
    fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            match crate::cpu_kernels::avx512() {
                Some(crate::cpu_kernels::Avx512::Vnni) => return Self::Avx512Vnni,
                Some(crate::cpu_kernels::Avx512::Bw) => return Self::Avx512Bw,
                None => {}
            }
            match crate::cpu_kernels::isa() {
                "avx-vnni" => return Self::AvxVnni,
                "avx2" => return Self::Avx2,
                _ => {}
            }
        }
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        if crate::cpu_kernels::neon_dotprod() {
            return Self::Neon;
        }
        Self::Scalar
    }
}

// SAFETY (all dispatchers below): `Path::detect` only returns a SIMD path whose features the
// CPU reported; the slices satisfy the length contracts `attention` validated (`dim` is a
// multiple of 32 and at most 256, keys and values hold `dim` values and `dim / 32` scales).
fn score_q8_tile(
    q: &[&[QueryBlock]],
    k: &[i8],
    scales: &[u16],
    out: &mut [f32],
    scale: f32,
    path: Path,
) {
    match path {
        #[cfg(target_arch = "x86_64")]
        Path::Avx512Vnni => unsafe { x86::score_tile512_vnni(q, k, scales, out, scale) },
        #[cfg(target_arch = "x86_64")]
        Path::Avx512Bw => unsafe { x86::score_tile512_bw(q, k, scales, out, scale) },
        #[cfg(target_arch = "x86_64")]
        Path::AvxVnni => unsafe { x86::score_tile_vnni(q, k, scales, out, scale) },
        #[cfg(target_arch = "x86_64")]
        Path::Avx2 => unsafe { x86::score_tile_avx2(q, k, scales, out, scale) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        Path::Neon => unsafe { neon::score_tile(q, k, scales, out, scale) },
        Path::Scalar => {
            for (r, query) in q.iter().enumerate() {
                out[r * 64] = score_q8(query, k, scales) * scale;
            }
        }
    }
}
fn score_q8(q: &[QueryBlock], k: &[i8], scales: &[u16]) -> f32 {
    q.iter()
        .zip(k.as_chunks::<32>().0.iter())
        .zip(scales)
        .map(|((q, k), &d)| {
            let sum =
                q.q.iter()
                    .zip(k)
                    .map(|(&a, &b)| a as i32 * b as i32)
                    .sum::<i32>();
            sum as f32 * q.d * f16::from_bits(d).to_f32()
        })
        .sum()
}
fn score_half(q: &[f32], k: &[u16], path: Path) -> f32 {
    match path {
        #[cfg(target_arch = "x86_64")]
        Path::Avx512Bw | Path::Avx512Vnni => unsafe { x86::score_half512(q, k) },
        #[cfg(target_arch = "x86_64")]
        Path::Avx2 | Path::AvxVnni => unsafe { x86::score_half(q, k) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        Path::Neon => unsafe { neon::score_half(q, k) },
        Path::Scalar => q
            .iter()
            .zip(k)
            .map(|(&a, &b)| a * f16::from_bits(b).to_f32())
            .sum(),
    }
}
fn decode_q8(v: &[i8], scales: &[u16], out: &mut [f32], path: Path) {
    match path {
        #[cfg(target_arch = "x86_64")]
        Path::Avx512Bw | Path::Avx512Vnni => unsafe { x86::decode_q8_512(v, scales, out) },
        #[cfg(target_arch = "x86_64")]
        Path::Avx2 | Path::AvxVnni => unsafe { x86::decode_q8(v, scales, out) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        Path::Neon => unsafe { neon::decode_q8(v, scales, out) },
        Path::Scalar => {
            for ((v, out), &d) in v
                .as_chunks::<32>()
                .0
                .iter()
                .zip(out.as_chunks_mut::<32>().0.iter_mut())
                .zip(scales)
            {
                let d = f16::from_bits(d).to_f32();
                for (x, &q) in out.iter_mut().zip(v) {
                    *x = q as f32 * d;
                }
            }
        }
    }
}
fn decode_half(v: &[u16], out: &mut [f32], path: Path) {
    match path {
        #[cfg(target_arch = "x86_64")]
        Path::Avx512Bw | Path::Avx512Vnni => unsafe { x86::decode_half512(v, out) },
        #[cfg(target_arch = "x86_64")]
        Path::Avx2 | Path::AvxVnni => unsafe { x86::decode_half(v, out) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        Path::Neon => unsafe { neon::decode_half(v, out) },
        Path::Scalar => {
            for (x, &h) in out.iter_mut().zip(v) {
                *x = f16::from_bits(h).to_f32();
            }
        }
    }
}
fn accumulate(out: &mut [f32], scores: &[f32], values: &[f32], path: Path) {
    match path {
        #[cfg(target_arch = "x86_64")]
        Path::Avx512Bw | Path::Avx512Vnni => unsafe { x86::accumulate512(out, scores, values) },
        #[cfg(target_arch = "x86_64")]
        Path::Avx2 | Path::AvxVnni => unsafe { x86::accumulate(out, scores, values) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        Path::Neon => unsafe { neon::accumulate(out, scores, values) },
        Path::Scalar => {
            for (&p, v) in scores.iter().zip(values.chunks_exact(out.len())) {
                for (a, &v) in out.iter_mut().zip(v) {
                    *a = p.mul_add(v, *a);
                }
            }
        }
    }
}
#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::QueryBlock;
    use std::arch::x86_64::*;
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum(v: __m256) -> f32 {
        let t = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let t = _mm_add_ps(t, _mm_movehl_ps(t, t));
        _mm_cvtss_f32(_mm_add_ss(t, _mm_shuffle_ps(t, t, 0x55)))
    }
    // Load each key block and decode its scale once for the entire GQA tile.
    // Individual queries retain the original FMA and horizontal reduction order.
    macro_rules! score_tile {
        ($name:ident, $feature:literal, $vnni:literal) => {
            #[target_feature(enable = $feature)]
            pub(super) unsafe fn $name(
                q: &[&[QueryBlock]],
                k: &[i8],
                scales: &[u16],
                out: &mut [f32],
                scale: f32,
            ) {
                let mut keys = [_mm256_setzero_si256(); 8];
                let mut absolute = [_mm256_setzero_si256(); 8];
                let mut ds = [0.0; 8];
                for (g, &d) in scales.iter().enumerate() {
                    let key = _mm256_loadu_si256(k.as_ptr().add(g * 32).cast());
                    keys[g] = if $vnni {
                        _mm256_xor_si256(key, _mm256_set1_epi8(-128))
                    } else {
                        key
                    };
                    absolute[g] = _mm256_abs_epi8(key);
                    ds[g] = crate::cpu_kernels::x86::half_to_float(d);
                }
                for (r, query) in q.iter().enumerate() {
                    let mut acc = _mm256_setzero_ps();
                    for (g, qb) in query.iter().enumerate() {
                        let qv = _mm256_load_si256(qb.q.as_ptr().cast());
                        let dot = if $vnni {
                            crate::cpu_kernels::x86::dot_vnni(
                                _mm256_set1_epi32(qb.init),
                                keys[g],
                                qv,
                            )
                        } else {
                            _mm256_madd_epi16(
                                _mm256_maddubs_epi16(absolute[g], _mm256_sign_epi8(qv, keys[g])),
                                _mm256_set1_epi16(1),
                            )
                        };
                        acc = _mm256_fmadd_ps(
                            _mm256_cvtepi32_ps(dot),
                            _mm256_set1_ps(qb.d * ds[g]),
                            acc,
                        );
                    }
                    out[r * 64] = hsum(acc) * scale;
                }
            }
        };
    }
    score_tile!(score_tile_avx2, "avx2,fma,f16c", false);
    score_tile!(score_tile_vnni, "avx2,fma,f16c,avxvnni", true);

    // 512-bit tile: two key blocks per vector; an odd last block has a zero upper half and a
    // zero upper scale. Each query keeps one accumulator, as in the AVX2 tile.
    macro_rules! score_tile512 {
        ($name:ident, $feature:literal, $dot:ident) => {
            #[target_feature(enable = $feature)]
            pub(super) unsafe fn $name(
                q: &[&[QueryBlock]],
                k: &[i8],
                scales: &[u16],
                out: &mut [f32],
                scale: f32,
            ) {
                let groups = scales.len();
                debug_assert!(groups <= 8);
                let pairs = groups.div_ceil(2);
                let flip = _mm512_set1_epi8(-128);
                let mut keys = [_mm512_setzero_si512(); 4];
                let mut ds = [(0.0, 0.0); 4];
                for p in 0..pairs {
                    let full = 2 * p + 1 < groups;
                    let low = _mm256_loadu_si256(k.as_ptr().add(64 * p).cast());
                    let high = if full {
                        _mm256_loadu_si256(k.as_ptr().add(64 * p + 32).cast())
                    } else {
                        _mm256_setzero_si256()
                    };
                    // Unsigned (signed + 128) keys, the operand form of `$dot`.
                    keys[p] = _mm512_xor_si512(crate::cpu_kernels::x86::join256(low, high), flip);
                    ds[p] = (
                        crate::cpu_kernels::x86::half_to_float(scales[2 * p]),
                        if full {
                            crate::cpu_kernels::x86::half_to_float(scales[2 * p + 1])
                        } else {
                            0.0
                        },
                    );
                }
                for (r, query) in q.iter().enumerate() {
                    let mut acc = _mm512_setzero_ps();
                    for p in 0..pairs {
                        let qa = &query[2 * p];
                        let qb = query.get(2 * p + 1);
                        let low = _mm256_load_si256(qa.q.as_ptr().cast());
                        let (qv, init, db) = match qb {
                            Some(qb) => (
                                crate::cpu_kernels::x86::join256(
                                    low,
                                    _mm256_load_si256(qb.q.as_ptr().cast()),
                                ),
                                _mm512_mask_blend_epi32(
                                    0xff00,
                                    _mm512_set1_epi32(qa.init),
                                    _mm512_set1_epi32(qb.init),
                                ),
                                qb.d * ds[p].1,
                            ),
                            None => (
                                _mm512_zextsi256_si512(low),
                                _mm512_mask_blend_epi32(
                                    0xff00,
                                    _mm512_set1_epi32(qa.init),
                                    _mm512_setzero_si512(),
                                ),
                                0.0,
                            ),
                        };
                        let dot = crate::cpu_kernels::x86::$dot(init, keys[p], qv);
                        acc = _mm512_fmadd_ps(
                            _mm512_cvtepi32_ps(dot),
                            crate::cpu_kernels::x86::halves_ps(qa.d * ds[p].0, db),
                            acc,
                        );
                    }
                    out[r * 64] = _mm512_reduce_add_ps(acc) * scale;
                }
            }
        };
    }
    score_tile512!(
        score_tile512_bw,
        "avx2,fma,f16c,avx512f,avx512bw,avx512vl",
        dot_biased512_bw
    );
    score_tile512!(
        score_tile512_vnni,
        "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni",
        dot_biased512_vnni
    );
    #[target_feature(enable = "avx512f,avx2,f16c")]
    pub(super) unsafe fn decode_half512(v: &[u16], out: &mut [f32]) {
        for j in (0..v.len()).step_by(16) {
            _mm512_storeu_ps(
                out.as_mut_ptr().add(j),
                _mm512_cvtph_ps(_mm256_loadu_si256(v.as_ptr().add(j).cast())),
            );
        }
    }
    #[target_feature(enable = "avx512f,avx2,f16c")]
    pub(super) unsafe fn decode_q8_512(v: &[i8], scales: &[u16], out: &mut [f32]) {
        for (g, &d) in scales.iter().enumerate() {
            let d = _mm512_set1_ps(crate::cpu_kernels::x86::half_to_float(d));
            for j in [0, 16] {
                let q = _mm_loadu_si128(v.as_ptr().add(g * 32 + j).cast());
                _mm512_storeu_ps(
                    out.as_mut_ptr().add(g * 32 + j),
                    _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q)), d),
                );
            }
        }
    }
    #[target_feature(enable = "avx512f,avx2,fma,f16c")]
    pub(super) unsafe fn score_half512(q: &[f32], k: &[u16]) -> f32 {
        let mut acc = _mm512_setzero_ps();
        for j in (0..q.len()).step_by(16) {
            let v = _mm512_cvtph_ps(_mm256_loadu_si256(k.as_ptr().add(j).cast()));
            acc = _mm512_fmadd_ps(_mm512_loadu_ps(q.as_ptr().add(j)), v, acc);
        }
        _mm512_reduce_add_ps(acc)
    }
    /// Same fused multiply-add per element as `accumulate`, 32 dimensions per pass.
    #[target_feature(enable = "avx512f,avx2,fma")]
    pub(super) unsafe fn accumulate512(out: &mut [f32], scores: &[f32], values: &[f32]) {
        let dim = out.len();
        for d in (0..dim).step_by(32) {
            let mut a0 = _mm512_loadu_ps(out.as_ptr().add(d));
            let mut a1 = _mm512_loadu_ps(out.as_ptr().add(d + 16));
            for (j, &p) in scores.iter().enumerate() {
                let v = values.as_ptr().add(j * dim + d);
                let p = _mm512_set1_ps(p);
                a0 = _mm512_fmadd_ps(p, _mm512_loadu_ps(v), a0);
                a1 = _mm512_fmadd_ps(p, _mm512_loadu_ps(v.add(16)), a1);
            }
            _mm512_storeu_ps(out.as_mut_ptr().add(d), a0);
            _mm512_storeu_ps(out.as_mut_ptr().add(d + 16), a1);
        }
    }
    #[target_feature(enable = "avx2,f16c")]
    pub(super) unsafe fn decode_half(v: &[u16], out: &mut [f32]) {
        for j in (0..v.len()).step_by(8) {
            _mm256_storeu_ps(
                out.as_mut_ptr().add(j),
                _mm256_cvtph_ps(_mm_loadu_si128(v.as_ptr().add(j).cast())),
            );
        }
    }
    #[target_feature(enable = "avx2,f16c")]
    pub(super) unsafe fn decode_q8(v: &[i8], scales: &[u16], out: &mut [f32]) {
        for (g, &d) in scales.iter().enumerate() {
            let d = _mm256_set1_ps(crate::cpu_kernels::x86::half_to_float(d));
            for j in [0, 8, 16, 24] {
                let q = _mm_loadl_epi64(v.as_ptr().add(g * 32 + j).cast());
                _mm256_storeu_ps(
                    out.as_mut_ptr().add(g * 32 + j),
                    _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q)), d),
                );
            }
        }
    }
    #[target_feature(enable = "avx2,fma,f16c")]
    pub(super) unsafe fn score_half(q: &[f32], k: &[u16]) -> f32 {
        let mut acc = _mm256_setzero_ps();
        for j in (0..q.len()).step_by(8) {
            let v = _mm256_cvtph_ps(_mm_loadu_si128(k.as_ptr().add(j).cast()));
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(q.as_ptr().add(j)), v, acc);
        }
        hsum(acc)
    }
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn accumulate(out: &mut [f32], scores: &[f32], values: &[f32]) {
        let dim = out.len();
        for d in (0..dim).step_by(32) {
            let mut a0 = _mm256_loadu_ps(out.as_ptr().add(d));
            let mut a1 = _mm256_loadu_ps(out.as_ptr().add(d + 8));
            let mut a2 = _mm256_loadu_ps(out.as_ptr().add(d + 16));
            let mut a3 = _mm256_loadu_ps(out.as_ptr().add(d + 24));
            for (j, &p) in scores.iter().enumerate() {
                let v = values.as_ptr().add(j * dim + d);
                let p = _mm256_set1_ps(p);
                a0 = _mm256_fmadd_ps(p, _mm256_loadu_ps(v), a0);
                a1 = _mm256_fmadd_ps(p, _mm256_loadu_ps(v.add(8)), a1);
                a2 = _mm256_fmadd_ps(p, _mm256_loadu_ps(v.add(16)), a2);
                a3 = _mm256_fmadd_ps(p, _mm256_loadu_ps(v.add(24)), a3);
            }
            _mm256_storeu_ps(out.as_mut_ptr().add(d), a0);
            _mm256_storeu_ps(out.as_mut_ptr().add(d + 8), a1);
            _mm256_storeu_ps(out.as_mut_ptr().add(d + 16), a2);
            _mm256_storeu_ps(out.as_mut_ptr().add(d + 24), a3);
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
mod neon {
    use super::QueryBlock;
    use crate::cpu_kernels::neon::half_to_float;
    use std::arch::aarch64::*;

    /// Load each key block and decode its scale once for the whole GQA tile; signed keys by
    /// signed queries with `sdot`, one accumulator per query.
    #[target_feature(enable = "neon,dotprod")]
    pub(super) unsafe fn score_tile(
        q: &[&[QueryBlock]],
        k: &[i8],
        scales: &[u16],
        out: &mut [f32],
        scale: f32,
    ) {
        debug_assert!(scales.len() <= 8);
        let mut keys = [[vdupq_n_s8(0); 2]; 8];
        let mut ds = [0.0; 8];
        for (g, &d) in scales.iter().enumerate() {
            let p = k.as_ptr().add(g * 32);
            keys[g] = [vld1q_s8(p), vld1q_s8(p.add(16))];
            ds[g] = half_to_float(d);
        }
        let zero = vdupq_n_s32(0);
        for (r, query) in q.iter().enumerate() {
            let mut acc = vdupq_n_f32(0.0);
            for (g, qb) in query.iter().enumerate() {
                let q0 = vld1q_s8(qb.q.as_ptr());
                let q1 = vld1q_s8(qb.q.as_ptr().add(16));
                let dot = vdotq_s32(vdotq_s32(zero, keys[g][0], q0), keys[g][1], q1);
                acc = vfmaq_f32(acc, vcvtq_f32_s32(dot), vdupq_n_f32(qb.d * ds[g]));
            }
            out[r * 64] = vaddvq_f32(acc) * scale;
        }
    }
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn score_half(q: &[f32], k: &[u16]) -> f32 {
        let mut acc = [vdupq_n_f32(0.0); 2];
        for j in (0..q.len()).step_by(8) {
            for (i, acc) in acc.iter_mut().enumerate() {
                let keys = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(k.as_ptr().add(j + 4 * i))));
                *acc = vfmaq_f32(*acc, vld1q_f32(q.as_ptr().add(j + 4 * i)), keys);
            }
        }
        vaddvq_f32(vaddq_f32(acc[0], acc[1]))
    }
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn decode_half(v: &[u16], out: &mut [f32]) {
        for j in (0..v.len()).step_by(4) {
            vst1q_f32(
                out.as_mut_ptr().add(j),
                vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(v.as_ptr().add(j)))),
            );
        }
    }
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn decode_q8(v: &[i8], scales: &[u16], out: &mut [f32]) {
        for (g, &d) in scales.iter().enumerate() {
            let d = vdupq_n_f32(half_to_float(d));
            for j in [0, 8, 16, 24] {
                let q = vmovl_s8(vld1_s8(v.as_ptr().add(g * 32 + j)));
                let dst = out.as_mut_ptr().add(g * 32 + j);
                vst1q_f32(dst, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q))), d));
                vst1q_f32(dst.add(4), vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(q)), d));
            }
        }
    }
    /// Same fused multiply-add per element as the scalar `accumulate`.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn accumulate(out: &mut [f32], scores: &[f32], values: &[f32]) {
        let dim = out.len();
        for d in (0..dim).step_by(16) {
            let o = out.as_mut_ptr().add(d);
            let mut a = [
                vld1q_f32(o),
                vld1q_f32(o.add(4)),
                vld1q_f32(o.add(8)),
                vld1q_f32(o.add(12)),
            ];
            for (j, &p) in scores.iter().enumerate() {
                let v = values.as_ptr().add(j * dim + d);
                let p = vdupq_n_f32(p);
                for (i, a) in a.iter_mut().enumerate() {
                    *a = vfmaq_f32(*a, vld1q_f32(v.add(4 * i)), p);
                }
            }
            for (i, a) in a.into_iter().enumerate() {
                vst1q_f32(o.add(4 * i), a);
            }
        }
    }
}

// The AVX-512 attention kernels against scalar references: every head size from one to eight
// 32-value groups (odd counts end on a half-empty key pair). Decode and accumulation perform the
// same per-element operations as the scalar code, so those compare bit for bit.
#[cfg(all(test, target_arch = "x86_64"))]
mod avx512_tests {
    use super::{x86, QueryBlock};
    use crate::cpu_selftest::Lcg;
    use half::f16;

    #[test]
    fn avx512_attention_kernels_match_scalar_references() {
        if !(std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vl")
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c"))
        {
            return;
        }
        let vnni = std::arch::is_x86_feature_detected!("avx512vnni");
        let mut rng = Lcg(77);
        for groups in 1..=8 {
            let dim = groups * 32;
            let keys = (0..dim)
                .map(|_| ((rng.next() % 255) as i32 - 127) as i8)
                .collect::<Vec<_>>();
            let scales = (0..groups)
                .map(|_| f16::from_f32(0.001 + (rng.next() % 100) as f32 / 1000.0).to_bits())
                .collect::<Vec<_>>();
            let queries = (0..3)
                .map(|_| {
                    (0..groups)
                        .map(|_| {
                            let x = rng.vector(32, 2.0);
                            let mut block = QueryBlock::default();
                            // SAFETY: AVX2/F16C detected above; exactly 32 values.
                            let (d, _, sums) = unsafe {
                                crate::cpu_kernels::x86::pack_q8_block(&x, &mut block.q, true)
                            };
                            block.d = d;
                            block.init = -16 * (sums[0] + sums[1]);
                            block
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let rows = queries.iter().map(Vec::as_slice).collect::<Vec<_>>();
            for use_vnni in [false, true] {
                if use_vnni && !vnni {
                    continue;
                }
                let mut out = vec![0.0f32; rows.len() * 64];
                // SAFETY: features detected above; `keys` holds `groups` blocks with scales.
                unsafe {
                    if use_vnni {
                        x86::score_tile512_vnni(&rows, &keys, &scales, &mut out, 0.125);
                    } else {
                        x86::score_tile512_bw(&rows, &keys, &scales, &mut out, 0.125);
                    }
                }
                for (r, query) in rows.iter().enumerate() {
                    let (mut want, mut magnitude) = (0.0f64, 0.0f64);
                    for (g, block) in query.iter().enumerate() {
                        let d = f64::from(block.d) * f64::from(f16::from_bits(scales[g]).to_f32());
                        for (&q, &k) in block.q.iter().zip(&keys[g * 32..g * 32 + 32]) {
                            let term = f64::from(q) * f64::from(k) * d * 0.125;
                            want += term;
                            magnitude += term.abs();
                        }
                    }
                    let error = (f64::from(out[r * 64]) - want).abs();
                    assert!(
                        error <= 1e-4 * magnitude + 1e-6,
                        "score tile vnni={use_vnni} groups {groups} row {r}: {error:e}"
                    );
                }
            }

            let query = rng.vector(dim, 1.0);
            let half_keys = rng
                .vector(dim, 1.0)
                .iter()
                .map(|&v| f16::from_f32(v).to_bits())
                .collect::<Vec<_>>();
            // SAFETY: features detected above; `dim` is a multiple of 32.
            let have = unsafe { x86::score_half512(&query, &half_keys) };
            let (mut want, mut magnitude) = (0.0f64, 0.0f64);
            for (&q, &k) in query.iter().zip(&half_keys) {
                let term = f64::from(q) * f64::from(f16::from_bits(k).to_f32());
                want += term;
                magnitude += term.abs();
            }
            assert!((f64::from(have) - want).abs() <= 1e-5 * magnitude + 1e-6);

            let mut have = vec![0.0f32; dim];
            // SAFETY: as above.
            unsafe { x86::decode_q8_512(&keys, &scales, &mut have) };
            for (i, &v) in have.iter().enumerate() {
                let want = keys[i] as f32 * f16::from_bits(scales[i / 32]).to_f32();
                assert_eq!(
                    v.to_bits(),
                    want.to_bits(),
                    "decode q8 groups {groups} index {i}"
                );
            }
            // SAFETY: as above.
            unsafe { x86::decode_half512(&half_keys, &mut have) };
            for (&v, &h) in have.iter().zip(&half_keys) {
                assert_eq!(v.to_bits(), f16::from_bits(h).to_f32().to_bits());
            }

            let positions = 13;
            let probabilities = rng.vector(positions, 1.0);
            let values = rng.vector(positions * dim, 1.0);
            let start = rng.vector(dim, 1.0);
            let mut have = start.clone();
            // SAFETY: as above; `values` holds `positions` rows of `dim`.
            unsafe { x86::accumulate512(&mut have, &probabilities, &values) };
            let mut want = start;
            for (&p, row) in probabilities.iter().zip(values.chunks_exact(dim)) {
                for (a, &v) in want.iter_mut().zip(row) {
                    *a = p.mul_add(v, *a);
                }
            }
            assert_eq!(
                have.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "accumulate groups {groups}"
            );
        }
    }
}

#[cfg(test)]
mod q8_attention_scratch_tests {
    use super::{Q8AttentionBuffers, Q8AttentionScratch, Q8_ATTENTION_BUFFERS};

    fn reset_tls() {
        Q8_ATTENTION_BUFFERS.with(|slot| *slot.borrow_mut() = Q8AttentionBuffers::default());
    }

    fn allocate(scratch: &mut Q8AttentionScratch) {
        scratch.0.scores.resize(128, 0.0);
        scratch.0.values.resize(2048, 0.0);
    }

    #[test]
    fn tls_scratch_is_reused_after_owner_drop() {
        reset_tls();
        let first_ptrs;
        let first_capacities;
        {
            let mut first = Q8AttentionScratch::take();
            allocate(&mut first);
            first_ptrs = (first.0.scores.as_ptr(), first.0.values.as_ptr());
            first_capacities = (first.0.scores.capacity(), first.0.values.capacity());
        }
        {
            let mut next = Q8AttentionScratch::take();
            allocate(&mut next);
            assert_eq!((next.0.scores.as_ptr(), next.0.values.as_ptr()), first_ptrs);
            assert_eq!(
                (next.0.scores.capacity(), next.0.values.capacity()),
                first_capacities
            );
        }
        reset_tls();
    }

    #[test]
    fn nested_scratch_owns_distinct_buffers_and_keeps_larger_capacity() {
        reset_tls();
        let mut outer = Q8AttentionScratch::take();
        outer.0.scores.resize(512, 0.0);
        outer.0.values.resize(8192, 0.0);
        let outer_ptrs = (outer.0.scores.as_ptr(), outer.0.values.as_ptr());
        let nested_ptrs;
        {
            let mut nested = Q8AttentionScratch::take();
            nested.0.scores.resize(128, 0.0);
            nested.0.values.resize(2048, 0.0);
            nested_ptrs = (nested.0.scores.as_ptr(), nested.0.values.as_ptr());
            assert_ne!(nested_ptrs.0, outer_ptrs.0);
            assert_ne!(nested_ptrs.1, outer_ptrs.1);
        }
        drop(outer);
        let mut retained = Q8AttentionScratch::take();
        retained.0.scores.resize(512, 0.0);
        retained.0.values.resize(8192, 0.0);
        assert_eq!(
            (retained.0.scores.as_ptr(), retained.0.values.as_ptr()),
            outer_ptrs
        );
        drop(retained);
        reset_tls();
    }
}

#[cfg(test)]
mod window_tests {
    use super::{attention_with, Cache, Options};
    use half::f16;

    /// Softmax attention over f32 values, row b seeing positions start..lengths[b].
    #[allow(clippy::too_many_arguments)]
    fn reference(q: &[f32], keys: &[f32], values: &[f32], lengths: &[i64], heads: usize, kv_heads: usize, dim: usize, scale: f32, window: Option<usize>) -> Vec<f32> {
        let g = heads / kv_heads;
        let mut out = vec![0.0f32; q.len()];
        for (b, &n) in lengths.iter().enumerate() {
            let n = n as usize;
            let start = window.map_or(0, |w| n.saturating_sub(w));
            for h in 0..heads {
                let kh = h / g;
                let qv = &q[(b * heads + h) * dim..][..dim];
                let scores: Vec<f32> = (start..n)
                    .map(|t| keys[(t * kv_heads + kh) * dim..][..dim].iter().zip(qv).map(|(k, x)| k * x).sum::<f32>() * scale)
                    .collect();
                let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
                let sum: f32 = e.iter().sum();
                for (i, t) in (start..n).enumerate() {
                    for d in 0..dim {
                        out[(b * heads + h) * dim + d] += e[i] / sum * values[(t * kv_heads + kh) * dim + d];
                    }
                }
            }
        }
        out
    }

    #[test]
    fn sliding_window_and_scale_match_a_direct_computation() {
        let (heads, kv_heads, dim, block_size) = (4, 2, 64, 16);
        let slots = 512;
        let mut state = 7u32;
        let mut rnd = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 9) as f32 / (1u32 << 23) as f32) - 1.0
        };
        let keys: Vec<f32> = (0..slots * kv_heads * dim).map(|_| f16::from_f32(rnd()).to_f32()).collect();
        let values: Vec<f32> = (0..slots * kv_heads * dim).map(|_| f16::from_f32(rnd()).to_f32()).collect();
        let key_words: Vec<u16> = keys.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
        let value_words: Vec<u16> = values.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
        // a decode row and a run of prompt rows, all of one sequence laid out in slot order
        let lengths: Vec<i64> = vec![300, 200, 201, 202, 203, 204];
        let stride = slots / block_size;
        let tables: Vec<i64> = lengths.iter().flat_map(|_| (0..stride as i64).collect::<Vec<_>>()).collect();
        let q: Vec<f32> = (0..lengths.len() * heads * dim).map(|_| rnd()).collect();
        for (scale, window) in [(None, None), (Some(1.0), None), (Some(0.2), Some(100)), (None, Some(7)), (Some(1.0), Some(1000))] {
            let key = Cache { data: &key_words, slots, kv_heads, dim, q8: false };
            let value = Cache { data: &value_words, ..key };
            let have = attention_with(&q, key, value, &tables, &lengths, heads, block_size, stride, &Options { scale, window })
                .expect("the tiled path takes this shape");
            let want = reference(&q, &keys, &values, &lengths, heads, kv_heads, dim, scale.unwrap_or(1.0 / (dim as f32).sqrt()), window);
            let worst = have.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(worst < 2e-3, "scale {scale:?} window {window:?}: max difference {worst}");
        }
    }
}
