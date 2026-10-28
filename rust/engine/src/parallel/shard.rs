//! Slicing of weight tensors for tensor parallelism.
//!
//! A projection `y = W x` with `W` of `rows x cols` (GGUF dimensions `[cols, rows]`, one output feature per row) is split either by
//! rows (column-parallel in the Megatron/vLLM vocabulary: every rank produces a slice of the outputs) or by columns (row-parallel:
//! every rank multiplies a slice of the inputs and the partial results are summed). The code here is the only place that knows how
//! each stored layout is cut; the cuts respect the block / group sizes of the quantized formats. A row cut of a plain block format
//! is a contiguous byte range and is never copied; a column cut has to gather one range per row and copies only the bytes the rank
//! keeps. Nothing here reads more than the shard it returns.
use crate::compute::{quant_layout, Linear, Tensor};
use fllm_gguf_probe::TensorBytes;
use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShardDim {
    /// Output features (`rows`): column-parallel. The result is a slice of the output vector.
    Rows,
    /// Input features (`cols`): row-parallel. The result is a partial sum.
    Cols,
}

/// One rank's part of a tensor: `len` entries from `start` along `dim`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShardSpec {
    pub dim: ShardDim,
    pub start: usize,
    pub len: usize,
}

impl ShardSpec {
    pub fn rows(start: usize, len: usize) -> Self {
        Self { dim: ShardDim::Rows, start, len }
    }
    pub fn cols(start: usize, len: usize) -> Self {
        Self { dim: ShardDim::Cols, start, len }
    }
    pub fn from_range(dim: ShardDim, range: Range<usize>) -> Self {
        Self { dim, start: range.start, len: range.len() }
    }
}

/// Data stored after the block payload of some formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tail {
    None,
    /// FP8M: one binary16 scale per row.
    RowScales,
    /// Q4ZP (GPTQ with activation order): one `u32` source column per stored column.
    ColPerm,
}

/// How a tensor of one GGUF type is stored: `block` values take `block_bytes` bytes; `tail` follows the rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedLayout {
    pub kind: u32,
    pub block: usize,
    pub block_bytes: usize,
    pub tail: Tail,
}

pub fn layout_for(kind: u32) -> Result<PackedLayout, String> {
    if fllm_gguf_probe::fp8b_type(kind) {
        // the tile scales and the trailer follow the codes of the whole matrix: a row or column range is not a byte range
        return Err("tensor-parallel splitting of block-wise FP8 (FP8B) weights is not implemented (use pipeline parallelism or one device)".into());
    }
    let (block, block_bytes) = match kind {
        0 => (1, 4),
        other => quant_layout(other).ok_or_else(|| format!("tensor-parallel splitting does not know GGUF type {other}"))?,
    };
    let tail = if fllm_gguf_probe::fp8m_type(kind) {
        Tail::RowScales
    } else if fllm_gguf_probe::q4zp_group(kind).is_some() {
        Tail::ColPerm
    } else {
        Tail::None
    };
    Ok(PackedLayout { kind, block, block_bytes, tail })
}

impl PackedLayout {
    pub fn row_bytes(&self, cols: usize) -> Result<usize, String> {
        if !cols.is_multiple_of(self.block) {
            return Err(format!("{cols} columns are not a multiple of the {}-value block", self.block));
        }
        Ok(cols / self.block * self.block_bytes)
    }

    pub fn total_bytes(&self, rows: usize, cols: usize) -> Result<usize, String> {
        let data = rows.checked_mul(self.row_bytes(cols)?).ok_or("tensor size overflow")?;
        Ok(data
            + match self.tail {
                Tail::None => 0,
                Tail::RowScales => rows * 2,
                Tail::ColPerm => cols * 4,
            })
    }

    /// Smallest piece a split along the input dimension may have: a whole block (and a whole group, for the group formats).
    pub fn k_granule(&self) -> usize {
        self.block
    }
}

/// Alignment a split has to keep, in values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Align {
    /// Pieces of the input dimension (row-parallel split): block / group size, and the Marlin tile height on the GPU.
    pub k: usize,
    /// Pieces of the output dimension (column-parallel split): the Marlin tile width on the GPU.
    pub n: usize,
}

/// Formats that the CUDA backend runs through the Marlin kernels (tiles of 16 x 64): Q4_K and the 4-bit group formats.
fn marlin_kind(kind: u32) -> bool {
    kind == 12 || (0x8000_0010..=0x8000_0016).contains(&kind)
}

pub fn alignment(kind: u32, gpu: bool) -> Result<Align, String> {
    let layout = layout_for(kind)?;
    let mut align = Align { k: layout.k_granule(), n: 1 };
    if gpu && marlin_kind(kind) {
        align.k = lcm(align.k, 16);
        align.n = 64;
    }
    Ok(align)
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

fn lcm(a: usize, b: usize) -> usize {
    a / gcd(a, b) * b
}

/// Every number of equal parts (up to `max`) `total` can be cut into with each part a multiple of `align`.
pub fn valid_parts(total: usize, align: usize, max: usize) -> Vec<usize> {
    (1..=max).filter(|&p| total.is_multiple_of(p) && (total / p).is_multiple_of(align.max(1))).collect()
}

/// Part `index` of `parts` equal parts of `total`; the error names the tensor and the sizes that would work.
pub fn even_split(what: &str, total: usize, align: usize, parts: usize, index: usize) -> Result<Range<usize>, String> {
    if parts == 0 || index >= parts {
        return Err(format!("{what}: part {index} of {parts} does not exist"));
    }
    let ok = total.is_multiple_of(parts) && (total / parts).is_multiple_of(align.max(1));
    if !ok {
        let working = valid_parts(total, align, 64);
        let list = if working.is_empty() { "none".to_owned() } else { working.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ") };
        let reason = if total.is_multiple_of(parts) {
            format!("each part would hold {} values, which is not a multiple of {align}", total / parts)
        } else {
            format!("{total} is not divisible by {parts}")
        };
        return Err(format!(
            "{what}: cannot split {total} values {parts} ways ({reason}). Tensor-parallel sizes that work for this tensor: {list}"
        ));
    }
    let size = total / parts;
    Ok(index * size..(index + 1) * size)
}

/// The bytes of a shard: a range of the source (no copy) or bytes gathered from strided pieces.
#[derive(Debug, PartialEq, Eq)]
pub enum Piece {
    Range { start: usize, len: usize },
    Owned(Vec<u8>),
}

impl Piece {
    pub fn bytes<'a>(&'a self, source: &'a [u8]) -> &'a [u8] {
        match self {
            Piece::Range { start, len } => &source[*start..*start + *len],
            Piece::Owned(v) => v,
        }
    }
}

fn perm_at(perm: &[u8], c: usize) -> u32 {
    u32::from_le_bytes([perm[c * 4], perm[c * 4 + 1], perm[c * 4 + 2], perm[c * 4 + 3]])
}

/// Cut `spec` out of the packed tensor `bytes` (`rows x cols` of `layout`, tail included).
pub fn slice_packed(layout: &PackedLayout, bytes: &[u8], rows: usize, cols: usize, spec: ShardSpec) -> Result<Piece, String> {
    let total = layout.total_bytes(rows, cols)?;
    if bytes.len() != total {
        return Err(format!("packed tensor has {} bytes, a {rows} x {cols} matrix of this format has {total}", bytes.len()));
    }
    let row_bytes = layout.row_bytes(cols)?;
    let data_end = rows * row_bytes;
    let tail = &bytes[data_end..];
    match spec.dim {
        ShardDim::Rows => {
            if spec.len == 0 || spec.start + spec.len > rows {
                return Err(format!("rows {}..{} are outside the {rows} rows", spec.start, spec.start + spec.len));
            }
            if spec.len == rows {
                return Ok(Piece::Range { start: 0, len: total });
            }
            let first = spec.start * row_bytes;
            let data = &bytes[first..(spec.start + spec.len) * row_bytes];
            match layout.tail {
                Tail::None => Ok(Piece::Range { start: first, len: data.len() }),
                Tail::RowScales => {
                    let mut out = data.to_vec();
                    out.extend_from_slice(&tail[spec.start * 2..(spec.start + spec.len) * 2]);
                    Ok(Piece::Owned(out))
                }
                Tail::ColPerm => {
                    let mut out = data.to_vec();
                    out.extend_from_slice(tail);
                    Ok(Piece::Owned(out))
                }
            }
        }
        ShardDim::Cols => {
            if spec.len == 0 || spec.start + spec.len > cols {
                return Err(format!("columns {}..{} are outside the {cols} columns", spec.start, spec.start + spec.len));
            }
            if spec.len == cols {
                return Ok(Piece::Range { start: 0, len: total });
            }
            if !spec.start.is_multiple_of(layout.block) || !spec.len.is_multiple_of(layout.block) {
                return Err(format!(
                    "columns {}..{} do not start and end on a {}-value block of this format",
                    spec.start,
                    spec.start + spec.len,
                    layout.block
                ));
            }
            let at = spec.start / layout.block * layout.block_bytes;
            let width = spec.len / layout.block * layout.block_bytes;
            let mut out = Vec::with_capacity(rows * width + tail.len());
            for r in 0..rows {
                out.extend_from_slice(&bytes[r * row_bytes + at..r * row_bytes + at + width]);
            }
            match layout.tail {
                Tail::None => {}
                Tail::RowScales => out.extend_from_slice(tail),
                Tail::ColPerm => {
                    // stored column c reads input column perm[c]; the shard keeps stored columns start..start+len and sees only the
                    // input slice start..start+len, so every one of those columns has to read from inside that slice
                    for c in spec.start..spec.start + spec.len {
                        let source = perm_at(tail, c) as usize;
                        if source < spec.start || source >= spec.start + spec.len {
                            return Err(format!(
                                "the activation-order permutation of this GPTQ tensor maps stored column {c} to input column {source}, outside the \
                                 shard {}..{}; use a model quantized without activation order (desc_act=false), a smaller tensor-parallel size, or pipeline parallelism",
                                spec.start,
                                spec.start + spec.len
                            ));
                        }
                        out.extend_from_slice(&((source - spec.start) as u32).to_le_bytes());
                    }
                }
            }
            Ok(Piece::Owned(out))
        }
    }
}

fn into_storage(piece: Piece, source: &TensorBytes) -> Result<TensorBytes, String> {
    match piece {
        Piece::Range { start, len } => source.slice(start, len),
        Piece::Owned(v) => Ok(TensorBytes::from_vec(v)),
    }
}

/// The shard of a loaded projection. `keep_bias` decides whether the bias travels with a column cut (only one rank may add it).
pub fn shard_linear(l: &Linear, spec: ShardSpec, keep_bias: bool, what: &str) -> Result<Linear, String> {
    let (rows, cols) = match spec.dim {
        ShardDim::Rows => (spec.len, l.cols),
        ShardDim::Cols => (l.rows, spec.len),
    };
    let bias = match (&l.bias, spec.dim) {
        (Some(b), ShardDim::Rows) => Some(b[spec.start..spec.start + spec.len].to_vec()),
        (Some(b), ShardDim::Cols) if keep_bias => Some(b.clone()),
        _ => None,
    };
    let mut out = if let Some((kind, storage)) = l.packed_storage() {
        let layout = layout_for(kind).map_err(|e| format!("{what}: {e}"))?;
        let piece = slice_packed(&layout, storage, l.rows, l.cols, spec).map_err(|e| format!("{what}: {e}"))?;
        Linear::from_packed(kind, into_storage(piece, storage)?, rows, cols, bias, l.scale, l.input_scale)?
    } else {
        let data = &l.weight.data;
        if data.len() != l.rows * l.cols {
            return Err(format!("{what}: the weights are not resident in a form that can be split"));
        }
        let sliced: Vec<f32> = match spec.dim {
            ShardDim::Rows => data[spec.start * l.cols..(spec.start + spec.len) * l.cols].to_vec(),
            ShardDim::Cols => (0..l.rows).flat_map(|r| data[r * l.cols + spec.start..r * l.cols + spec.start + spec.len].iter().copied()).collect(),
        };
        Linear::from_parts(Tensor { shape: vec![cols, rows], data: sliced }, rows, cols, bias)?
    };
    out.scale = l.scale;
    out.input_scale = l.input_scale;
    out.enable_fast_cpu();
    Ok(out)
}

/// Several row ranges of one projection joined in order (a query projection whose output gate is stored after all queries needs
/// both ranges on one rank).
pub fn shard_linear_rows_multi(l: &Linear, ranges: &[Range<usize>], what: &str) -> Result<Linear, String> {
    let pieces: Vec<Linear> = ranges.iter().map(|r| shard_linear(l, ShardSpec::from_range(ShardDim::Rows, r.clone()), true, what)).collect::<Result<_, _>>()?;
    if pieces.len() == 1 {
        return pieces.into_iter().next().ok_or_else(|| "no rows".to_owned());
    }
    let rows: usize = pieces.iter().map(|p| p.rows).sum();
    let bias = if pieces.iter().all(|p| p.bias.is_some()) { Some(pieces.iter().flat_map(|p| p.bias.iter().flatten().copied()).collect()) } else { None };
    let mut out = if let Some((kind, _)) = l.packed_storage() {
        let layout = layout_for(kind)?;
        if layout.tail != Tail::None {
            return Err(format!("{what}: a split gate layout is not supported for this weight format"));
        }
        let mut joined = Vec::new();
        for p in &pieces {
            if let Some((_, b)) = p.packed_storage() {
                joined.extend_from_slice(b);
            }
        }
        Linear::from_packed(kind, TensorBytes::from_vec(joined), rows, l.cols, bias, l.scale, l.input_scale)?
    } else {
        let data: Vec<f32> = pieces.iter().flat_map(|p| p.weight.data.iter().copied()).collect();
        Linear::from_parts(Tensor { shape: vec![l.cols, rows], data }, rows, l.cols, bias)?
    };
    out.scale = l.scale;
    out.input_scale = l.input_scale;
    out.enable_fast_cpu();
    Ok(out)
}

/// A per-expert vector of NVFP4 tensor scales (`<name>.scale` of a stacked expert tensor): the entries of the experts a rank holds.
pub fn shard_expert_scales(scales: &[f32], experts: Range<usize>) -> Result<Vec<f32>, String> {
    scales
        .get(experts.clone())
        .map(<[f32]>::to_vec)
        .ok_or_else(|| format!("experts {}..{} are outside the {} per-expert scales", experts.start, experts.end, scales.len()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_selftest::{random_weights, Lcg};

    const ROWS: usize = 12;
    const COLS: usize = 1024;

    fn random_float_bytes(kind: u32, count: usize, rng: &mut Lcg) -> Vec<u8> {
        let mut out = Vec::new();
        for _ in 0..count {
            let v = rng.unit();
            match kind {
                0 => out.extend_from_slice(&v.to_le_bytes()),
                1 => out.extend_from_slice(&half::f16::from_f32(v).to_le_bytes()),
                _ => out.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes()),
            }
        }
        out
    }

    /// Q4ZP permutations that stay inside blocks of 256 columns, so that every split into 1, 2 or 4 parts is legal.
    fn local_permutation(bytes: &mut [u8], rng: &mut Lcg) {
        let n = bytes.len() - COLS * 4;
        for chunk in 0..COLS / 256 {
            let mut perm: Vec<u32> = (0..256).map(|i| (chunk * 256 + i) as u32).collect();
            for i in (1..256).rev() {
                perm.swap(i, (rng.next() as usize) % (i + 1));
            }
            for (i, p) in perm.iter().enumerate() {
                let at = n + (chunk * 256 + i) * 4;
                bytes[at..at + 4].copy_from_slice(&p.to_le_bytes());
            }
        }
    }

    fn tensor_bytes(kind: u32, rng: &mut Lcg) -> Vec<u8> {
        match kind {
            0 | 1 | 30 => random_float_bytes(kind, ROWS * COLS, rng),
            _ => {
                let mut bytes = random_weights(kind, ROWS, COLS, rng);
                if matches!(kind, 0x8000_0014..=0x8000_0016) {
                    local_permutation(&mut bytes, rng);
                }
                bytes
            }
        }
    }

    const KINDS: &[u32] = &[
        0, 1, 30, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 39, 40, 0x8000_0010, 0x8000_0011, 0x8000_0012,
        0x8000_0014, 0x8000_0015, 0x8000_0016, 0x8000_0020,
    ];

    fn same_bits(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    #[test]
    fn row_and_column_shards_reassemble_to_the_dequantized_tensor_for_every_format() {
        let mut rng = Lcg(0x5eed);
        let mut checked = 0;
        for &kind in KINDS {
            let bytes = tensor_bytes(kind, &mut rng);
            let layout = layout_for(kind).unwrap();
            assert_eq!(bytes.len(), layout.total_bytes(ROWS, COLS).unwrap(), "kind {kind:#x}");
            let Ok(full) = crate::compute::dequantize(&bytes, kind, ROWS * COLS) else { continue };
            checked += 1;
            for parts in [1usize, 2, 3, 4] {
                if !ROWS.is_multiple_of(parts) {
                    continue;
                }
                // rows
                let mut joined = Vec::new();
                for i in 0..parts {
                    let r = even_split("t", ROWS, 1, parts, i).unwrap();
                    let piece = slice_packed(&layout, &bytes, ROWS, COLS, ShardSpec::from_range(ShardDim::Rows, r.clone())).unwrap();
                    if layout.tail == Tail::None && parts > 1 {
                        assert!(matches!(piece, Piece::Range { .. }), "a row cut of a plain format must not copy (kind {kind:#x})");
                    }
                    let shard = piece.bytes(&bytes);
                    let values = crate::compute::dequantize(shard, kind, r.len() * COLS).unwrap();
                    assert!(same_bits(&values, &full[r.start * COLS..r.end * COLS]), "rows kind {kind:#x} parts {parts} index {i}");
                    joined.extend_from_slice(&shard[..r.len() * layout.row_bytes(COLS).unwrap()]);
                }
                assert_eq!(joined, bytes[..ROWS * layout.row_bytes(COLS).unwrap()], "kind {kind:#x}");
            }
            for parts in [1usize, 2, 4] {
                for i in 0..parts {
                    let c = even_split("t", COLS, layout.k_granule(), parts, i).unwrap();
                    let piece = slice_packed(&layout, &bytes, ROWS, COLS, ShardSpec::from_range(ShardDim::Cols, c.clone())).unwrap();
                    let values = crate::compute::dequantize(piece.bytes(&bytes), kind, ROWS * c.len()).unwrap();
                    let mut expected = Vec::new();
                    for r in 0..ROWS {
                        expected.extend_from_slice(&full[r * COLS + c.start..r * COLS + c.end]);
                    }
                    assert!(same_bits(&values, &expected), "cols kind {kind:#x} parts {parts} index {i}");
                }
            }
        }
        assert!(checked >= 20, "only {checked} formats could be dequantized");
    }

    #[test]
    fn a_column_cut_inside_a_block_is_refused() {
        let mut rng = Lcg(1);
        let bytes = tensor_bytes(12, &mut rng);
        let layout = layout_for(12).unwrap();
        let error = slice_packed(&layout, &bytes, ROWS, COLS, ShardSpec::cols(128, 256)).unwrap_err();
        assert!(error.contains("block"), "{error}");
    }

    #[test]
    fn an_activation_order_permutation_that_crosses_the_cut_is_refused_with_advice() {
        let mut rng = Lcg(9);
        let layout = layout_for(0x8000_0016).unwrap();
        let bytes = random_weights(0x8000_0016, ROWS, COLS, &mut rng);
        let error = slice_packed(&layout, &bytes, ROWS, COLS, ShardSpec::cols(0, 512)).unwrap_err();
        assert!(error.contains("activation-order") && error.contains("desc_act=false"), "{error}");
        // a column cut of the whole tensor is not a cut
        assert!(slice_packed(&layout, &bytes, ROWS, COLS, ShardSpec::cols(0, COLS)).is_ok());
    }

    #[test]
    fn the_split_error_names_the_tensor_and_the_sizes_that_work() {
        let error = even_split("blk.3.ffn_down.weight", 11008, 256, 4, 0).unwrap_err();
        assert!(error.contains("blk.3.ffn_down.weight") && error.contains("11008") && error.contains("Tensor-parallel sizes"), "{error}");
        assert_eq!(valid_parts(8192, 256, 16), vec![1, 2, 4, 8, 16, 32].into_iter().filter(|&p| p <= 16).collect::<Vec<_>>());
        assert_eq!(valid_parts(11008, 256, 8), vec![1]);
        assert_eq!(valid_parts(11000, 256, 8), Vec::<usize>::new());
        assert_eq!(even_split("t", 4096, 256, 8, 3).unwrap(), 1536..2048);
        assert!(even_split("t", 4096, 1, 3, 0).unwrap_err().contains("not divisible"));
    }

    #[test]
    fn marlin_tiles_tighten_the_alignment_on_the_gpu_only() {
        assert_eq!(alignment(12, false).unwrap(), Align { k: 256, n: 1 });
        assert_eq!(alignment(12, true).unwrap(), Align { k: 256, n: 64 });
        assert_eq!(alignment(0x8000_0010, true).unwrap(), Align { k: 32, n: 64 });
        assert_eq!(alignment(8, true).unwrap(), Align { k: 32, n: 1 });
        assert!(alignment(0x7777_7777, false).is_err());
    }

    #[test]
    fn linear_shards_add_up_to_the_projection() {
        // f32 weights: partial sums over column shards equal the full product, row shards concatenate
        let (rows, cols) = (8usize, 64usize);
        let mut rng = Lcg(77);
        let data = rng.vector(rows * cols, 0.5);
        let bias = rng.vector(rows, 0.5);
        let full = Linear::from_parts(Tensor { shape: vec![cols, rows], data }, rows, cols, Some(bias.clone())).unwrap();
        let x = rng.vector(cols, 1.0);
        let want = full.forward(&x).unwrap();
        let mut by_rows = Vec::new();
        for i in 0..2 {
            let part = shard_linear(&full, ShardSpec::rows(i * 4, 4), true, "t").unwrap();
            by_rows.extend(part.forward(&x).unwrap());
        }
        for (a, b) in by_rows.iter().zip(&want) {
            assert!((a - b).abs() <= 1e-5 * (1.0 + b.abs()), "{a} vs {b}");
        }
        let mut sum = vec![0.0f32; rows];
        for i in 0..4 {
            let part = shard_linear(&full, ShardSpec::cols(i * 16, 16), i == 0, "t").unwrap();
            for (s, v) in sum.iter_mut().zip(part.forward(&x[i * 16..(i + 1) * 16]).unwrap()) {
                *s += v;
            }
        }
        for (a, b) in sum.iter().zip(&want) {
            assert!((a - b).abs() <= 1e-5 * (1.0 + b.abs()), "{a} vs {b}");
        }
    }

    fn l2(a: &[f32], b: &[f32]) -> f64 {
        let n: f64 = a.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt().max(1e-30);
        a.iter().zip(b).map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2)).sum::<f64>().sqrt() / n
    }

    /// The kernels themselves: the forward pass of row shards concatenates to, and of column shards adds up to, the forward pass of the
    /// whole projection, for every format (the quantized kernels quantize the activations per 32 values, which a block-aligned cut keeps).
    #[test]
    fn shards_of_a_projection_agree_with_the_whole_projection_in_every_format() {
        let mut rng = Lcg(0xabc);
        let mut worst: Vec<(u32, f64, f64)> = Vec::new();
        for &kind in KINDS {
            let bytes = tensor_bytes(kind, &mut rng);
            let bias = rng.vector(ROWS, 0.5);
            let Ok(full) = Linear::from_packed(kind, TensorBytes::from_vec(bytes), ROWS, COLS, Some(bias), 1.0, 0.0) else { continue };
            let mut full = full;
            full.enable_fast_cpu();
            let x = rng.vector(COLS, 1.0);
            let Ok(want) = full.forward(&x) else { continue };
            let mut rows_error = 0.0f64;
            let mut cols_error = 0.0f64;
            for parts in [2usize, 4] {
                let mut joined = Vec::new();
                for i in 0..parts {
                    let part = shard_linear(&full, ShardSpec::rows(i * ROWS / parts, ROWS / parts), true, "t").unwrap();
                    joined.extend(part.forward(&x).unwrap());
                }
                rows_error = rows_error.max(l2(&want, &joined));
                let mut sum = vec![0.0f32; ROWS];
                for i in 0..parts {
                    let c = even_split("t", COLS, layout_for(kind).unwrap().k_granule(), parts, i).unwrap();
                    let part = shard_linear(&full, ShardSpec::from_range(ShardDim::Cols, c.clone()), i == 0, "t").unwrap();
                    for (s, v) in sum.iter_mut().zip(part.forward(&x[c]).unwrap()) {
                        *s += v;
                    }
                }
                cols_error = cols_error.max(l2(&want, &sum));
            }
            worst.push((kind, rows_error, cols_error));
        }
        let bad: Vec<String> = worst
            .iter()
            .filter(|(_, r, c)| *r > 1e-4 || *c > 3e-2)
            .map(|(k, r, c)| format!("{} (type {k:#x}): row shards {r:.2e}, column shards {c:.2e}", crate::cpu_selftest::format_name(*k)))
            .collect();
        for (k, r, c) in &worst {
            eprintln!("{:<8} rows {r:.2e} cols {c:.2e}", crate::cpu_selftest::format_name(*k));
        }
        assert!(worst.len() >= 20 && bad.is_empty(), "{}", bad.join("
"));
    }

    #[test]
    fn expert_scale_vectors_follow_the_expert_range() {
        let scales = [0.5f32, 1.0, 2.0, 4.0];
        assert_eq!(shard_expert_scales(&scales, 1..3).unwrap(), vec![1.0, 2.0]);
        assert!(shard_expert_scales(&scales, 3..5).is_err());
    }
}
