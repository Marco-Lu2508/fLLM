//! Latent-cache multi-head latent attention (`FLLM_MLA=latent`): the host side.
//!
//! The expanded path (default) caches the rebuilt per-head keys and values of every token: `heads * (nope + rope)` key values and
//! `heads * value` value values per layer (DeepSeek-V2-Lite: 16 * 192 + 16 * 128 = 5120, Kimi K2 / DeepSeek-V3: 128 * 192 + 128 * 128 = 40960).
//! The latent path caches only what MLA actually compresses to: the normalized latent (`kv_lora`) and the rotated shared rope key (`rope`)
//! of each token, 576 values for those models, and attends in the latent space (CUDA only, `cuda/mla_latent.cu`, `cuda/mla_decode.cu`).
//!
//! The cache of a latent model reuses the paged pools of the engine with another geometry: one KV head of width `kv_lora + rope` for the
//! keys (the row `[latent | rope key]`; the values are the first `kv_lora` values of the same row) and a placeholder value pool of
//! [`LATENT_VALUE_DIM`] values per token (the pools always come in pairs), pages of [`LATENT_BLOCK_SIZE`] tokens.
use crate::compute::{dequantize, Linear};
use crate::config::MlaConfig;
use crate::kv_format::KvFormat;
use half::f16;

/// Tokens per page of the latent cache.
pub const LATENT_BLOCK_SIZE: usize = 64;
/// Width of the placeholder value pool of a latent cache (the latent values are read from the key pool).
pub const LATENT_VALUE_DIM: usize = 2;

/// `FLLM_MLA`: which MLA implementation the CUDA backend uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlaMode {
    /// Per-head keys and values in the cache (default).
    Expanded,
    /// Latent cache with absorbed up-projections.
    Latent,
}

/// `FLLM_MLA` value to mode; anything but `expanded` / `latent` (or unset / empty) is an error.
pub fn parse_mode(value: Option<&str>) -> Result<MlaMode, String> {
    match value.map(str::trim) {
        None | Some("") | Some("expanded") | Some("0") => Ok(MlaMode::Expanded),
        Some("latent") | Some("1") => Ok(MlaMode::Latent),
        Some(other) => Err(format!("FLLM_MLA={other} is not expanded or latent")),
    }
}

/// The mode requested through the environment; an invalid value is reported and counts as `expanded`.
pub fn mode_from_env() -> MlaMode {
    match parse_mode(std::env::var("FLLM_MLA").ok().as_deref()) {
        Ok(mode) => mode,
        Err(message) => {
            eprintln!("[fllm-rs] warning: {message}; using the expanded MLA cache");
            MlaMode::Expanded
        }
    }
}

/// Cache geometry `(kv_heads, head_dim, value_dim)` of a latent model.
pub fn latent_geometry(m: &MlaConfig) -> (usize, usize, usize) {
    (1, m.kv_lora + m.rope, LATENT_VALUE_DIM)
}

/// Cache formats the latent kernels read: f16 and the 8-bit formats with a scale per token (q8_0, int8, fp8_e4m3).
pub fn format_supported(format: KvFormat) -> bool {
    matches!(format, KvFormat::F16 | KvFormat::Q8_0 | KvFormat::Int8 | KvFormat::Fp8E4m3)
}

/// Values the expanded cache holds per token and layer (keys + values of every head).
pub fn expanded_values_per_token(heads: usize, m: &MlaConfig) -> usize {
    heads * (m.nope + m.rope) + heads * m.value
}

/// Values the latent cache holds per token and layer (key row + the placeholder value pool).
pub fn latent_values_per_token(m: &MlaConfig) -> usize {
    let (_, head_dim, value_dim) = latent_geometry(m);
    head_dim + value_dim
}

/// Cache bytes per token and layer of the expanded cache in `format` (keys and values in the same format).
pub fn expanded_bytes_per_token(heads: usize, m: &MlaConfig, format: KvFormat) -> u64 {
    format.bytes_per_slot(heads, m.nope + m.rope) + format.bytes_per_slot(heads, m.value)
}

/// Cache bytes per token and layer of the latent cache in `format`.
pub fn latent_bytes_per_token(m: &MlaConfig, format: KvFormat) -> u64 {
    let (kv_heads, head_dim, value_dim) = latent_geometry(m);
    format.bytes_per_slot(kv_heads, head_dim) + format.bytes_per_slot(kv_heads, value_dim)
}

/// The kv_b projection (`heads * (nope + value)` rows of `kv_lora` values) as dense FP16 bits: the matrices W_UK (rows `h * (nope + value) + j`) and
/// W_UV (rows `h * (nope + value) + nope + v`) that the latent kernels absorb into the queries and apply to the attention output.
pub fn dense_kv_b_f16(kv_b: &Linear) -> Result<Vec<u16>, String> {
    let count = kv_b.rows.checked_mul(kv_b.cols).ok_or("kv_b size overflow")?;
    let mut values = match kv_b.packed_weights() {
        Some((kind, bytes)) => dequantize(bytes, kind, count).map_err(|e| format!("kv_b: {e}"))?,
        None => kv_b.weight.data.clone(),
    };
    if values.len() != count {
        return Err(format!("kv_b holds {} values, expected {count}", values.len()));
    }
    if kv_b.scale != 1.0 {
        values.iter_mut().for_each(|v| *v *= kv_b.scale);
    }
    if values.iter().any(|v| !v.is_finite() || v.abs() > 65000.0) {
        return Err("kv_b has values that do not fit FP16".into());
    }
    Ok(values.iter().map(|&v| f16::from_f32(v).to_bits()).collect())
}

/// How many query rows one chunk of the latent attention processes so that its FP16 temporaries (absorbed queries and attention outputs,
/// `heads * (2 * kv_lora + rope)` halves per row) stay below `budget_bytes`.
pub fn rows_per_chunk(heads: usize, m: &MlaConfig, budget_bytes: usize) -> usize {
    let per_row = heads * (2 * m.kv_lora + m.rope) * 2;
    (budget_bytes / per_row.max(1)).clamp(1, 4096)
}

/// f64 references of the two formulations of MLA attention for one query row: the expanded one (what the CPU decoder and the default CUDA path
/// do: per-head keys and values rebuilt from the latents, ordinary attention) and the absorbed one (what the latent kernels do). They agree
/// mathematically; the self test compares the kernels with them and the unit tests below prove that the two agree.
pub mod reference {
    /// Dimensions and constants of one MLA layer.
    #[derive(Clone, Copy, Debug)]
    pub struct Layer {
        pub heads: usize,
        pub nope: usize,
        pub rope: usize,
        pub value: usize,
        pub kv_lora: usize,
        pub neox: bool,
        /// Softmax scale.
        pub scale: f64,
    }

    impl Layer {
        pub fn width(&self) -> usize {
            self.kv_lora + self.rope
        }
        fn per_head(&self) -> usize {
            self.nope + self.value
        }
    }

    /// cos / sin tables `[position][pair]` of a rotary embedding over `rope` values (the layout of the rotary tables of the engine).
    pub fn rope_tables(max_pos: usize, rope: usize, base: f64) -> (Vec<f32>, Vec<f32>) {
        let half = rope / 2;
        let mut cos = Vec::with_capacity(max_pos * half);
        let mut sin = Vec::with_capacity(max_pos * half);
        for p in 0..max_pos {
            for i in 0..half {
                let angle = p as f64 * base.powf(-2.0 * i as f64 / rope as f64);
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
        (cos, sin)
    }

    /// The rotary part `x` (`rope` values) rotated for `pos`.
    pub fn rotate(x: &[f32], pos: usize, cos: &[f32], sin: &[f32], neox: bool) -> Vec<f64> {
        let half = x.len() / 2;
        let mut out = vec![0.0f64; x.len()];
        for p in 0..half {
            let (c, s) = (f64::from(cos[pos * half + p]), f64::from(sin[pos * half + p]));
            let (ia, ib) = if neox { (p, p + half) } else { (2 * p, 2 * p + 1) };
            let (a, b) = (f64::from(x[ia]), f64::from(x[ib]));
            out[ia] = a * c - b * s;
            out[ib] = b * c + a * s;
        }
        out
    }

    /// The row the latent cache stores for a token: `[latent | rotated rope key]`.
    pub fn cache_row(l: &Layer, latent: &[f32], pe: &[f32], pos: usize, cos: &[f32], sin: &[f32]) -> Vec<f32> {
        let mut row = latent.to_vec();
        row.extend(rotate(pe, pos, cos, sin, l.neox).iter().map(|&v| v as f32));
        row
    }

    /// Absorbed query of one row: per head `[q_nope W_UK | rotated q_pe]`. `q_full`: `[heads][nope + rope]` as projected (content, then rotary).
    pub fn absorbed_query(l: &Layer, w: &[f32], q_full: &[f32], pos: usize, cos: &[f32], sin: &[f32]) -> Vec<f64> {
        let dk = l.nope + l.rope;
        let mut out = vec![0.0f64; l.heads * l.width()];
        for h in 0..l.heads {
            let q = &q_full[h * dk..(h + 1) * dk];
            let o = &mut out[h * l.width()..(h + 1) * l.width()];
            for j in 0..l.nope {
                let row = &w[(h * l.per_head() + j) * l.kv_lora..(h * l.per_head() + j + 1) * l.kv_lora];
                for (c, &wv) in row.iter().enumerate() {
                    o[c] += f64::from(q[j]) * f64::from(wv);
                }
            }
            o[l.kv_lora..].copy_from_slice(&rotate(&q[l.nope..], pos, cos, sin, l.neox));
        }
        out
    }

    /// Absorbed attention of one row over the cache rows (`len` of them are visible): `(o_lat [heads][kv_lora], out [heads][value])`.
    pub fn attend_latent(l: &Layer, w: &[f32], q_lat: &[f64], rows: &[Vec<f32>], len: usize) -> (Vec<f64>, Vec<f64>) {
        let mut o_lat = vec![0.0f64; l.heads * l.kv_lora];
        let mut out = vec![0.0f64; l.heads * l.value];
        for h in 0..l.heads {
            let q = &q_lat[h * l.width()..(h + 1) * l.width()];
            let scores: Vec<f64> =
                rows[..len].iter().map(|r| l.scale * r.iter().zip(q).map(|(&a, &b)| f64::from(a) * b).sum::<f64>()).collect();
            let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let p: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
            let total: f64 = p.iter().sum();
            let o = &mut o_lat[h * l.kv_lora..(h + 1) * l.kv_lora];
            for (t, pt) in p.iter().enumerate() {
                for (c, ov) in o.iter_mut().enumerate() {
                    *ov += pt / total * f64::from(rows[t][c]);
                }
            }
            for v in 0..l.value {
                let row = &w[(h * l.per_head() + l.nope + v) * l.kv_lora..(h * l.per_head() + l.nope + v + 1) * l.kv_lora];
                out[h * l.value + v] = row.iter().zip(o.iter()).map(|(&a, &b)| f64::from(a) * b).sum();
            }
        }
        (o_lat, out)
    }

    /// The expanded keys (content part) and values of every cache row: `[token][head][nope]`, `[token][head][value]`.
    pub fn expand(l: &Layer, w: &[f32], rows: &[Vec<f32>]) -> (Vec<f64>, Vec<f64>) {
        let mut k = vec![0.0f64; rows.len() * l.heads * l.nope];
        let mut v = vec![0.0f64; rows.len() * l.heads * l.value];
        for (t, row) in rows.iter().enumerate() {
            for h in 0..l.heads {
                for j in 0..l.per_head() {
                    let wr = &w[(h * l.per_head() + j) * l.kv_lora..(h * l.per_head() + j + 1) * l.kv_lora];
                    let x: f64 = wr.iter().zip(row).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum();
                    if j < l.nope {
                        k[(t * l.heads + h) * l.nope + j] = x;
                    } else {
                        v[(t * l.heads + h) * l.value + j - l.nope] = x;
                    }
                }
            }
        }
        (k, v)
    }

    /// Expanded attention of one row (`mla_qkv` of the decoder + ordinary attention): per head the query `[rotated q_pe | q_nope]` against the keys
    /// `[shared rotated rope key | K_nope]`, values `V`. `rows` supplies the rope keys, `k` / `v` come from [`expand`].
    #[allow(clippy::too_many_arguments)]
    pub fn attend_expanded(
        l: &Layer,
        q_full: &[f32],
        pos: usize,
        cos: &[f32],
        sin: &[f32],
        rows: &[Vec<f32>],
        k: &[f64],
        v: &[f64],
        len: usize,
    ) -> Vec<f64> {
        let dk = l.nope + l.rope;
        let mut out = vec![0.0f64; l.heads * l.value];
        for h in 0..l.heads {
            let q = &q_full[h * dk..(h + 1) * dk];
            let q_pe = rotate(&q[l.nope..], pos, cos, sin, l.neox);
            let scores: Vec<f64> = (0..len)
                .map(|t| {
                    let pe = &rows[t][l.kv_lora..];
                    let rope: f64 = q_pe.iter().zip(pe).map(|(&a, &b)| a * f64::from(b)).sum();
                    let content: f64 = (0..l.nope).map(|j| f64::from(q[j]) * k[(t * l.heads + h) * l.nope + j]).sum();
                    l.scale * (rope + content)
                })
                .collect();
            let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let p: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
            let total: f64 = p.iter().sum();
            for (t, pt) in p.iter().enumerate() {
                for e in 0..l.value {
                    out[h * l.value + e] += pt / total * v[(t * l.heads + h) * l.value + e];
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lite() -> MlaConfig {
        MlaConfig { q_lora: None, kv_lora: 512, nope: 128, rope: 64, value: 128, rotary: true }
    }

    #[test]
    fn mode_parsing() {
        assert_eq!(parse_mode(None), Ok(MlaMode::Expanded));
        assert_eq!(parse_mode(Some("")), Ok(MlaMode::Expanded));
        assert_eq!(parse_mode(Some("expanded")), Ok(MlaMode::Expanded));
        assert_eq!(parse_mode(Some("latent")), Ok(MlaMode::Latent));
        assert!(parse_mode(Some("absorbed")).is_err());
    }

    #[test]
    fn cache_memory_per_token() {
        let m = lite();
        // DeepSeek-V2-Lite: 16 heads
        assert_eq!(expanded_values_per_token(16, &m), 5120);
        assert_eq!(expanded_bytes_per_token(16, &m, KvFormat::F16), 10240);
        assert_eq!(latent_geometry(&m), (1, 576, LATENT_VALUE_DIM));
        assert_eq!(latent_bytes_per_token(&m, KvFormat::F16), 576 * 2 + 4);
        // Kimi K2 / DeepSeek-V3: 128 heads
        assert_eq!(expanded_values_per_token(128, &m), 40960);
        assert_eq!(expanded_bytes_per_token(128, &m, KvFormat::F16), 81920);
        let ratio = expanded_bytes_per_token(128, &m, KvFormat::F16) as f64 / latent_bytes_per_token(&m, KvFormat::F16) as f64;
        assert!(ratio > 70.0 && ratio < 71.5, "{ratio}");
        // fp8 and q8_0 shrink both further; the fp8 latent row is 576 codes plus one f16 scale
        assert_eq!(latent_bytes_per_token(&m, KvFormat::Fp8E4m3), 576 + 2 + 2 + 2);
        assert_eq!(latent_bytes_per_token(&m, KvFormat::Q8_0), 576 + 36 + 2 + 2);
        assert!(expanded_bytes_per_token(16, &m, KvFormat::F16) > 8 * latent_bytes_per_token(&m, KvFormat::F16));
    }

    #[test]
    fn chunking_bounds_the_temporaries() {
        let m = lite();
        for heads in [16usize, 64, 128] {
            let rows = rows_per_chunk(heads, &m, 48 << 20);
            assert!(rows >= 1);
            if rows > 1 && rows < 4096 {
                assert!(rows * heads * (2 * m.kv_lora + m.rope) * 2 <= 48 << 20);
            }
        }
        assert_eq!(rows_per_chunk(100_000, &m, 1 << 20), 1);
    }

    #[test]
    fn kv_b_becomes_fp16_rows() {
        use crate::compute::Tensor;
        let (rows, cols) = (6usize, 8usize);
        let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32 - 20.0) * 0.125).collect();
        let linear = Linear::from_parts(Tensor { shape: vec![cols, rows], data: data.clone() }, rows, cols, None).unwrap();
        let bits = dense_kv_b_f16(&linear).unwrap();
        assert_eq!(bits.len(), rows * cols);
        for (b, v) in bits.iter().zip(&data) {
            assert_eq!(f16::from_bits(*b).to_f32(), *v); // multiples of 1/8 below 8 are exact in FP16
        }
        let big = Linear::from_parts(Tensor { shape: vec![2, 1], data: vec![1.0e6, 1.0] }, 1, 2, None).unwrap();
        assert!(dense_kv_b_f16(&big).is_err());
    }

    #[test]
    fn absorbed_and_expanded_attention_agree() {
        use super::reference::*;
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut unit = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) % 20_001) as f32 / 10_000.0 - 1.0
        };
        for (heads, nope, rope, value, kv_lora, neox) in [(3usize, 8usize, 4usize, 6usize, 16usize, false), (4, 16, 8, 8, 32, true), (1, 4, 2, 4, 8, false)] {
            let l = Layer { heads, nope, rope, value, kv_lora, neox, scale: 1.0 / ((nope + rope) as f64).sqrt() };
            let w: Vec<f32> = (0..heads * (nope + value) * kv_lora).map(|_| unit() * 0.3).collect();
            let (cos, sin) = rope_tables(64, rope, 10_000.0);
            let len = 37;
            let rows: Vec<Vec<f32>> = (0..len)
                .map(|t| {
                    let latent: Vec<f32> = (0..kv_lora).map(|_| unit()).collect();
                    let pe: Vec<f32> = (0..rope).map(|_| unit()).collect();
                    cache_row(&l, &latent, &pe, t, &cos, &sin)
                })
                .collect();
            let q_full: Vec<f32> = (0..heads * (nope + rope)).map(|_| unit()).collect();
            let (k, v) = expand(&l, &w, &rows);
            let mut previous: Option<Vec<f64>> = None;
            for visible in [len, len / 2, 1] {
                let pos = visible - 1;
                let q_lat = absorbed_query(&l, &w, &q_full, pos, &cos, &sin);
                let (o_lat, absorbed) = attend_latent(&l, &w, &q_lat, &rows, visible);
                assert_eq!(o_lat.len(), heads * kv_lora);
                let expanded = attend_expanded(&l, &q_full, pos, &cos, &sin, &rows, &k, &v, visible);
                let scale = expanded.iter().fold(0.0f64, |m, x| m.max(x.abs()));
                for (i, (a, b)) in absorbed.iter().zip(&expanded).enumerate() {
                    assert!((a - b).abs() <= 1e-9 * (1.0 + scale), "heads {heads} neox {neox} visible {visible} element {i}: absorbed {a} expanded {b}");
                }
                // the visible length matters (a defect that ignores it would not change the result)
                if let Some(p) = &previous {
                    assert!(p.iter().zip(&absorbed).any(|(a, b)| (a - b).abs() > 1e-6), "visible {visible} gives the result of the previous length");
                }
                previous = Some(absorbed);
            }
        }
    }

    #[test]
    fn rotation_keeps_the_norm_and_neox_pairs_differ() {
        use super::reference::*;
        let (cos, sin) = rope_tables(16, 8, 10_000.0);
        let x = [0.5f32, -1.0, 2.0, 0.25, 1.5, -0.75, 0.1, 0.9];
        for neox in [false, true] {
            let r = rotate(&x, 9, &cos, &sin, neox);
            let before: f64 = x.iter().map(|&v| f64::from(v).powi(2)).sum();
            let after: f64 = r.iter().map(|v| v * v).sum();
            assert!((before - after).abs() < 1e-5, "{before} {after}");
        }
        assert_ne!(rotate(&x, 9, &cos, &sin, false), rotate(&x, 9, &cos, &sin, true));
        // position 0 is the identity
        let r = rotate(&x, 0, &cos, &sin, true);
        assert!(r.iter().zip(&x).all(|(a, b)| (a - f64::from(*b)).abs() < 1e-12));
    }

    #[test]
    fn supported_cache_formats() {
        assert!(format_supported(KvFormat::F16) && format_supported(KvFormat::Fp8E4m3) && format_supported(KvFormat::Q8_0));
        assert!(!format_supported(KvFormat::Q4_0) && !format_supported(KvFormat::Bf16));
        // the placeholder value pool must be valid in every supported format
        for f in [KvFormat::F16, KvFormat::Q8_0, KvFormat::Int8, KvFormat::Fp8E4m3] {
            assert!(f.check_dim(LATENT_VALUE_DIM).is_ok() && f.check_dim(576).is_ok(), "{}", f.name());
        }
    }
}
