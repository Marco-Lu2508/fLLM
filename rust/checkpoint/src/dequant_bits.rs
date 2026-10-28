//! Bit-packed integer weights that the converter expands: GPTQ with 2, 3 or 8 bits (and the `gptq_v2` zero-point
//! convention) and MLX affine quantization (2 to 8 bits, any group size). The 4-bit GPTQ/AWQ paths with native
//! `Q4Z` output live in `packed_quant.rs`.
use crate::dequant::Dequant;
use crate::safetensors::{Payload, Tensor};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// `bits` bits at bit offset `position` of a little-endian bit stream.
fn read_bits(bytes: &[u8], position: usize, bits: usize) -> u32 {
    let first = position / 8;
    let mut window = 0u64;
    for i in 0..5 {
        window |= u64::from(bytes.get(first + i).copied().unwrap_or(0)) << (8 * i);
    }
    ((window >> (position % 8)) & ((1u64 << bits) - 1)) as u32
}

fn floats(t: &Tensor) -> Result<Vec<f32>, String> {
    let b = t.payload.as_slice();
    match t.dtype.as_str() {
        "F32" => Ok(b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()),
        "F16" => Ok(b.as_chunks::<2>().0.iter().map(|c| half::f16::from_bits(u16::from_le_bytes(*c)).to_f32()).collect()),
        "BF16" => Ok(b.as_chunks::<2>().0.iter().map(|c| f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16)).collect()),
        other => Err(format!("{}: expected a float tensor, found {other}", t.name)),
    }
}

// ---------------------------------------------------------------- GPTQ with 2, 3 or 8 bits

/// AutoGPTQ/GPTQModel packing: `qweight` `[in * bits / 32, out]` holds the codes of one output column as a
/// little-endian bit stream down the input dimension; `qzeros` `[groups, out * bits / 32]` the zero points of one group
/// as a bit stream along the outputs. Version 1 stores `zero - 1`, version 2 (`gptq_v2`) stores `zero`.
#[derive(Debug)]
pub(crate) struct GptqBits {
    qweight: Payload,
    qzeros: Payload,
    scales: Vec<f32>,
    g_idx: Option<Vec<u32>>,
    bits: usize,
    group: usize,
    in_features: usize,
    out_features: usize,
    stored_minus_one: bool,
}

impl GptqBits {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        qweight: Payload,
        qzeros: Payload,
        scales: Vec<f32>,
        g_idx: Option<Vec<u32>>,
        bits: usize,
        group: usize,
        in_features: usize,
        out_features: usize,
        v2: bool,
    ) -> Result<Self, String> {
        if !matches!(bits, 2 | 3 | 4 | 8) || group == 0 || in_features == 0 || out_features == 0 {
            return Err(format!("unsupported GPTQ layout: {bits} bits, group {group}, {in_features} x {out_features}"));
        }
        let groups = in_features.div_ceil(group);
        let weight_rows = (in_features * bits).div_ceil(32);
        let zero_words = (out_features * bits).div_ceil(32);
        if qweight.as_slice().len() != weight_rows * out_features * 4 {
            return Err(format!("qweight has {} bytes, expected {}", qweight.as_slice().len(), weight_rows * out_features * 4));
        }
        if qzeros.as_slice().len() != groups * zero_words * 4 {
            return Err(format!("qzeros has {} bytes, expected {}", qzeros.as_slice().len(), groups * zero_words * 4));
        }
        if scales.len() != groups * out_features {
            return Err(format!("scales has {} entries, expected {}", scales.len(), groups * out_features));
        }
        if let Some(g) = &g_idx {
            if g.len() != in_features || g.iter().any(|&x| x as usize >= groups) {
                return Err("g_idx does not match the input dimension and group count".into());
            }
        }
        Ok(Self { qweight, qzeros, scales, g_idx, bits, group, in_features, out_features, stored_minus_one: !v2 })
    }
}

impl Dequant for GptqBits {
    fn element_count(&self) -> usize {
        self.in_features * self.out_features
    }

    fn decode_chunk(&self, start: usize, dst: &mut [f32]) -> Result<(), String> {
        let end = start.checked_add(dst.len()).filter(|&e| e <= self.element_count()).ok_or("GPTQ range exceeds the tensor")?;
        let weight_rows = (self.in_features * self.bits).div_ceil(32);
        let zero_words = (self.out_features * self.bits).div_ceil(32);
        let (qw, qz) = (self.qweight.as_slice(), self.qzeros.as_slice());
        let mask = (1u32 << self.bits) - 1;
        // gather one column of codes at a time: the bit stream of output column `o` runs down the rows
        let mut column: Vec<u8> = Vec::new();
        let mut loaded = usize::MAX;
        for (slot, e) in dst.iter_mut().zip(start..end) {
            let (o, i) = (e / self.in_features, e % self.in_features);
            if loaded != o {
                column.clear();
                for row in 0..weight_rows {
                    let at = (row * self.out_features + o) * 4;
                    column.extend_from_slice(&qw[at..at + 4]);
                }
                loaded = o;
            }
            let q = read_bits(&column, i * self.bits, self.bits);
            let g = self.g_idx.as_ref().map_or(i / self.group, |t| t[i] as usize);
            let zero_row = &qz[g * zero_words * 4..(g + 1) * zero_words * 4];
            let stored = read_bits(zero_row, o * self.bits, self.bits);
            let zero = if self.stored_minus_one { (stored + 1) & mask } else { stored };
            *slot = (q as f32 - zero as f32) * self.scales[g * self.out_features + o];
        }
        Ok(())
    }
}

// ---------------------------------------------------------------- MLX affine quantization

/// MLX `mx.quantize` (affine): codes of `bits` bits in a little-endian bit stream per row (`uint32` words), one
/// `scale` and `bias` per `group` inputs; `w = q * scale + bias`.
///
/// Besides the dequantizing reader this is what the converter stores natively (`MLXA` tensor types, see
/// `fllm_gguf_probe::MlxAffine`): the packed words are copied as they are, the scales and biases as 16-bit floats. That is
/// exact whenever every scale and bias is representable as an IEEE half or a bfloat16 ([`MlxAffine::native`]).
#[derive(Debug)]
pub struct MlxAffine {
    words: Payload,
    scales: Vec<f32>,
    biases: Vec<f32>,
    bits: usize,
    group: usize,
    rows: usize,
    cols: usize,
    /// The 16-bit float format that holds every scale and bias without a change (`Some(true)`: bfloat16), `None` if neither does.
    scale_format: Option<bool>,
}

/// Whether every value survives the round trip through a 16-bit float format, bit for bit.
fn exact_in(values: &[f32], to: impl Fn(f32) -> f32) -> bool {
    values.iter().all(|&v| v.is_finite() && to(v).to_bits() == v.to_bits())
}

fn bf16_round_trip(v: f32) -> f32 {
    half::bf16::from_f32(v).to_f32()
}

fn f16_round_trip(v: f32) -> f32 {
    half::f16::from_f32(v).to_f32()
}

impl MlxAffine {
    pub(crate) fn new(words: Payload, scales: Vec<f32>, biases: Vec<f32>, bits: usize, group: usize, rows: usize, cols: usize) -> Result<Self, String> {
        if !(2..=8).contains(&bits) || group == 0 || cols == 0 || !cols.is_multiple_of(group) || !(cols * bits).is_multiple_of(32) {
            return Err(format!("unsupported MLX layout: {bits} bits, group {group}, {rows} x {cols}"));
        }
        if words.as_slice().len() != rows * cols * bits / 8 {
            return Err(format!("MLX weight has {} bytes, expected {}", words.as_slice().len(), rows * cols * bits / 8));
        }
        let groups = rows * (cols / group);
        if scales.len() != groups || biases.len() != groups {
            return Err(format!("MLX scales/biases have {}/{} entries, expected {groups}", scales.len(), biases.len()));
        }
        // half first: its 11 bit mantissa holds everything bfloat16's 8 bits do, as long as the exponents fit
        let scale_format = if exact_in(&scales, f16_round_trip) && exact_in(&biases, f16_round_trip) {
            Some(false)
        } else if exact_in(&scales, bf16_round_trip) && exact_in(&biases, bf16_round_trip) {
            Some(true)
        } else {
            None
        };
        Ok(Self { words, scales, biases, bits, group, rows, cols, scale_format })
    }

    /// The native tensor type of this weight, or `None` when it cannot be stored without changing a value (a bit width or group
    /// size the `MLXA` types do not have, or a scale or bias that neither half nor bfloat16 holds exactly).
    pub fn native(&self) -> Option<fllm_gguf_probe::MlxAffine> {
        let (bits, group) = (self.bits as u32, self.group as u32);
        if !fllm_gguf_probe::mlx_affine_supported(bits, group) {
            return None;
        }
        Some(fllm_gguf_probe::MlxAffine { bits, group, bf16: self.scale_format? })
    }

    /// Why the weight cannot be stored natively (for the warning of the converter).
    pub fn native_refusal(&self) -> String {
        if !fllm_gguf_probe::mlx_affine_supported(self.bits as u32, self.group as u32) {
            format!("{} bits with group size {} have no native type (groups are multiples of 32, bits 2 to 6 or 8)", self.bits, self.group)
        } else {
            "a scale or bias is not exactly representable as an f16 or a bfloat16".to_owned()
        }
    }

    /// Append the `MLXA` blocks of output row `row`: per group the scale and the bias as 16-bit floats, then the codes as they are
    /// in the MLX words (`group * bits / 8` bytes).
    pub fn encode_gguf_row(&self, row: usize, dst: &mut Vec<u8>) -> Result<(), String> {
        let native = self.native().ok_or_else(|| self.native_refusal())?;
        if row >= self.rows {
            return Err(format!("MLX row {row} out of range ({} rows)", self.rows));
        }
        let per_row = self.cols / self.group;
        let group_bytes = self.group * self.bits / 8;
        let row_bytes = self.cols * self.bits / 8;
        let codes = &self.words.as_slice()[row * row_bytes..(row + 1) * row_bytes];
        dst.reserve(per_row * (group_bytes + 4));
        for g in 0..per_row {
            for v in [self.scales[row * per_row + g], self.biases[row * per_row + g]] {
                let bits = if native.bf16 { half::bf16::from_f32(v).to_bits() } else { half::f16::from_f32(v).to_bits() };
                dst.extend_from_slice(&bits.to_le_bytes());
            }
            dst.extend_from_slice(&codes[g * group_bytes..(g + 1) * group_bytes]);
        }
        Ok(())
    }
}

impl Dequant for MlxAffine {
    fn element_count(&self) -> usize {
        self.rows * self.cols
    }

    fn decode_chunk(&self, start: usize, dst: &mut [f32]) -> Result<(), String> {
        let end = start.checked_add(dst.len()).filter(|&e| e <= self.element_count()).ok_or("MLX range exceeds the tensor")?;
        let bytes = self.words.as_slice();
        let row_bytes = self.cols * self.bits / 8;
        let per_row = self.cols / self.group;
        for (slot, e) in dst.iter_mut().zip(start..end) {
            let (r, c) = (e / self.cols, e % self.cols);
            let q = read_bits(&bytes[r * row_bytes..(r + 1) * row_bytes], c * self.bits, self.bits);
            let g = r * per_row + c / self.group;
            *slot = q as f32 * self.scales[g] + self.biases[g];
        }
        Ok(())
    }

    fn mlx(&self) -> Option<&MlxAffine> {
        Some(self)
    }
}

/// Name of the native tensor type in the converter's output tensors: `MLXA<bits>G<group>`, with a `B` when the scales and biases are bfloat16.
pub(crate) fn mlx_dtype_name(m: fllm_gguf_probe::MlxAffine) -> String {
    format!("MLXA{}G{}{}", m.bits, m.group, if m.bf16 { "B" } else { "" })
}

/// GGUF type id of a name made by [`mlx_dtype_name`].
pub(crate) fn mlx_dtype_id(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("MLXA")?;
    let (bits, rest) = rest.split_once('G')?;
    let (group, bf16) = match rest.strip_suffix('B') {
        Some(g) => (g, true),
        None => (rest, false),
    };
    let m = fllm_gguf_probe::MlxAffine { bits: bits.parse().ok()?, group: group.parse().ok()?, bf16 };
    fllm_gguf_probe::mlx_affine_supported(m.bits, m.group).then(|| m.type_id())
}
/// Bits and group size of one MLX module: the global `quantization` entry, overridden per module path.
fn mlx_params(config: &Value, base: &str) -> Option<(usize, usize)> {
    let get = |v: &Value| -> Option<(usize, usize)> { Some((v.get("bits")?.as_u64()? as usize, v.get("group_size")?.as_u64()? as usize)) };
    let stripped = base.strip_prefix("language_model.").unwrap_or(base);
    for key in [base, stripped, &format!("model.{stripped}")] {
        if let Some(v) = config.get(key).filter(|v| v.is_object()) {
            if let Some(p) = get(v) {
                return Some(p);
            }
        }
    }
    get(config)
}

/// MLX checkpoints: `<name>.weight` (U32) next to `<name>.scales` and `<name>.biases`; every other tensor is dense.
pub(crate) fn split_mlx(
    all: Vec<Tensor>,
    config: &Value,
    map: &dyn Fn(&str) -> Result<Option<String>, String>,
) -> Result<(Vec<Tensor>, Vec<crate::convert::PackedGroup>), String> {
    let mut dense = Vec::new();
    let mut weights = Vec::new();
    let mut scales: HashMap<String, Tensor> = HashMap::new();
    let mut biases: HashMap<String, Tensor> = HashMap::new();
    for t in all {
        if let Some(base) = t.name.strip_suffix(".scales") {
            scales.insert(base.to_owned(), t);
        } else if let Some(base) = t.name.strip_suffix(".biases") {
            // dense `.biases` do not exist in Hugging Face layouts; MLX quantization biases sit next to `.scales`
            biases.insert(base.to_owned(), t);
        } else if t.dtype == "U32" && t.name.ends_with(".weight") {
            weights.push(t);
        } else {
            dense.push(t);
        }
    }
    let mut packed = Vec::new();
    for w in weights {
        let base = w.name.strip_suffix(".weight").unwrap_or(&w.name).to_owned();
        let s = scales.remove(&base).ok_or_else(|| format!("MLX weight {} has no {base}.scales", w.name))?;
        let b = biases.remove(&base).ok_or_else(|| format!("MLX weight {} has no {base}.biases", w.name))?;
        if w.shape.len() != 2 {
            return Err(format!("{}: MLX weight must be a matrix, found shape {:?} (stacked expert weights are not supported)", w.name, w.shape));
        }
        let (bits, group) = mlx_params(config, &base).ok_or_else(|| format!("{}: no bits/group_size in the quantization config", w.name))?;
        let rows = w.shape[0] as usize;
        let cols = w.shape[1] as usize * 32 / bits;
        let m = MlxAffine::new(w.payload.clone(), floats(&s)?, floats(&b)?, bits, group, rows, cols).map_err(|e| format!("{}: {e}", w.name))?;
        let mapped = map(&w.name)?.ok_or_else(|| format!("packed tensor {:?} maps to a skipped tensor", w.name))?;
        packed.push((mapped, crate::packed_quant::PackedData::Dyn(Arc::new(m)), vec![rows as u64, cols as u64], w.payload));
    }
    if let Some(left) = scales.keys().next() {
        return Err(format!("{left}.scales has no packed weight"));
    }
    if packed.is_empty() {
        return Err("the config declares MLX quantization but no U32 weights with scales and biases were found".into());
    }
    Ok((dense, packed))
}

/// GPTQ tensors (`qweight`, `qzeros`, `scales`, optional `g_idx`) with 2, 3 or 8 bits, or `gptq_v2` zero points.
pub(crate) fn split_gptq_bits(
    all: Vec<Tensor>,
    bits: usize,
    group: usize,
    v2: bool,
    map: &dyn Fn(&str) -> Result<Option<String>, String>,
) -> Result<(Vec<Tensor>, Vec<crate::convert::PackedGroup>), String> {
    let mut dense = Vec::new();
    let mut groups: HashMap<String, HashMap<&'static str, Tensor>> = HashMap::new();
    for t in all {
        let suffix = ["qweight", "qzeros", "scales", "g_idx"].into_iter().find(|s| t.name.strip_suffix(s).is_some_and(|b| b.ends_with('.')));
        match suffix {
            Some(s) => {
                let base = t.name[..t.name.len() - s.len() - 1].to_owned();
                if groups.entry(base).or_default().insert(s, t).is_some() {
                    return Err("duplicate GPTQ component".into());
                }
            }
            None => dense.push(t),
        }
    }
    let mut packed = Vec::new();
    let mut names: Vec<_> = groups.keys().cloned().collect();
    names.sort();
    for base in names {
        let mut g = groups.remove(&base).unwrap();
        let mut take = |k: &'static str| g.remove(k).ok_or_else(|| format!("GPTQ tensor group {base:?} is missing {k}"));
        let (qw, qz, sc) = (take("qweight")?, take("qzeros")?, take("scales")?);
        let gi = g.remove("g_idx");
        if qw.dtype != "I32" || qz.dtype != "I32" || qw.shape.len() != 2 || qz.shape.len() != 2 {
            return Err(format!("GPTQ components of {base:?} must be rank-2 I32 tensors"));
        }
        let out_features = qw.shape[1] as usize;
        let in_features = (qw.shape[0] as usize * 32) / bits;
        let g_idx = match gi {
            Some(t) => {
                if t.dtype != "I32" {
                    return Err(format!("{}: g_idx must be I32", t.name));
                }
                Some(t.payload.as_slice().as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c) as u32).collect::<Vec<_>>())
            }
            None => None,
        };
        let in_features = g_idx.as_ref().map_or(in_features, Vec::len);
        let d = GptqBits::new(qw.payload.clone(), qz.payload, floats(&sc)?, g_idx, bits, group, in_features, out_features, v2)
            .map_err(|e| format!("{base}: {e}"))?;
        let mapped = map(&format!("{base}.weight"))?.ok_or_else(|| format!("packed tensor {base:?} maps to a skipped tensor"))?;
        packed.push((mapped, crate::packed_quant::PackedData::Dyn(Arc::new(d)), vec![out_features as u64, in_features as u64], qw.payload));
    }
    if packed.is_empty() {
        return Err("quantization_config declares GPTQ but no qweight/qzeros/scales groups were found".into());
    }
    Ok((dense, packed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::PackedGroup;

    #[test]
    fn bit_streams_cross_word_boundaries() {
        // 3-bit codes 0..9 packed little-endian: code k at bit 3k
        let mut bytes = vec![0u8; 8];
        for k in 0..10usize {
            let v = (k % 8) as u64;
            let pos = 3 * k;
            let mut word = u64::from_le_bytes(bytes[..8].try_into().unwrap());
            word |= v << pos;
            bytes.copy_from_slice(&word.to_le_bytes());
        }
        for k in 0..10usize {
            assert_eq!(read_bits(&bytes, 3 * k, 3), (k % 8) as u32);
        }
    }

    fn fixture(case: &str) -> (Vec<Tensor>, Vec<f32>) {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata").join("quant").join(format!("{case}.safetensors"));
        let mut tensors = crate::safetensors::read_file(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let at = tensors.iter().position(|t| t.name == "expected").expect("expected tensor");
        let expected = floats(&tensors.swap_remove(at)).unwrap();
        (tensors, expected)
    }

    fn identity(n: &str) -> Result<Option<String>, String> {
        Ok(Some(n.to_owned()))
    }

    fn decode_all(group: &PackedGroup) -> Vec<f32> {
        let mut out = vec![0f32; group.1.element_count()];
        for (i, chunk) in out.chunks_mut(29).enumerate() {
            group.1.decode_f32_chunk(i * 29, chunk).unwrap();
        }
        out
    }

    fn assert_close(case: &str, got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len(), "{case}");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() <= tol * w.abs().max(1.0), "{case}: element {i}: got {g}, want {w}");
        }
    }

    #[test]
    fn gptq_2_3_4_8_bit_match_gptqmodel() {
        let cases: Value = serde_json::from_slice(&std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/quant/cases.json")).unwrap()).unwrap();
        let mut seen = 0;
        for (name, meta) in cases.as_object().unwrap() {
            if meta["format"] != "gptq" {
                continue;
            }
            let (tensors, expected) = fixture(name);
            let (bits, group, v2) = (meta["bits"].as_u64().unwrap() as usize, meta["group_size"].as_u64().unwrap() as usize, meta["v2"].as_bool().unwrap());
            let (dense, packed) = split_gptq_bits(tensors, bits, group, v2, &identity).unwrap();
            assert!(dense.is_empty(), "{name}");
            // the library's dequantization is [in, out]; the converter produces [out, in]
            let (rows, cols) = (meta["out_features"].as_u64().unwrap() as usize, meta["in_features"].as_u64().unwrap() as usize);
            let transposed: Vec<f32> = (0..rows * cols).map(|e| expected[(e % cols) * rows + e / cols]).collect();
            assert_close(name, &decode_all(&packed[0]), &transposed, 1e-6);
            seen += 1;
        }
        assert!(seen >= 8, "only {seen} GPTQ fixtures");
    }

    #[test]
    fn mlx_affine_matches_mlx() {
        let cases: Value = serde_json::from_slice(&std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/quant/cases.json")).unwrap()).unwrap();
        let mut seen = 0;
        for (name, meta) in cases.as_object().unwrap() {
            if meta["format"] != "mlx" {
                continue;
            }
            let (tensors, expected) = fixture(name);
            let config = serde_json::json!({"bits": meta["bits"], "group_size": meta["group_size"]});
            let (dense, packed) = split_mlx(tensors, &config, &identity).unwrap();
            assert!(dense.is_empty());
            assert_close(name, &decode_all(&packed[0]), &expected, 1e-6);
            seen += 1;
        }
        assert!(seen >= 8, "only {seen} MLX fixtures");
        // per-module overrides win over the global setting
        let config = serde_json::json!({"bits": 4, "group_size": 64, "model.layers.0.mlp": {"bits": 8, "group_size": 32}});
        assert_eq!(mlx_params(&config, "model.layers.0.mlp"), Some((8, 32)));
        assert_eq!(mlx_params(&config, "model.layers.1.mlp"), Some((4, 64)));
    }

    /// Reference decoder of `MLXA` blocks, written the way the kernels read them: 32-bit words, shifts, and a second word where a
    /// code straddles a word boundary (independent of the encoder and of `read_bits`).
    pub(crate) fn decode_mlxa(bytes: &[u8], m: fllm_gguf_probe::MlxAffine, cols: usize) -> Vec<f32> {
        let (bits, group) = (m.bits as usize, m.group as usize);
        let block = group * bits / 8 + 4;
        assert_eq!(bytes.len(), cols / group * block);
        let half16 = |b: &[u8]| {
            let raw = u16::from_le_bytes([b[0], b[1]]);
            if m.bf16 {
                half::bf16::from_bits(raw).to_f32()
            } else {
                half::f16::from_bits(raw).to_f32()
            }
        };
        let mut out = Vec::with_capacity(cols);
        for b in bytes.chunks_exact(block) {
            let (scale, bias) = (half16(&b[0..2]), half16(&b[2..4]));
            let words: Vec<u32> = b[4..].as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w)).collect();
            for i in 0..group {
                let (word, shift) = (i * bits / 32, (i * bits) % 32);
                let mut v = words[word] >> shift;
                if shift + bits > 32 {
                    v |= words[word + 1] << (32 - shift);
                }
                out.push((v & ((1 << bits) - 1)) as f32 * scale + bias);
            }
        }
        out
    }

    #[test]
    fn mlx_fixtures_become_native_blocks_that_decode_identically() {
        let cases: Value = serde_json::from_slice(&std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/quant/cases.json")).unwrap()).unwrap();
        let (mut seen, mut refused) = (0, 0);
        for (name, meta) in cases.as_object().unwrap() {
            if meta["format"] != "mlx" {
                continue;
            }
            let (tensors, _) = fixture(name);
            let config = serde_json::json!({"bits": meta["bits"], "group_size": meta["group_size"]});
            let (bits, group) = (meta["bits"].as_u64().unwrap() as usize, meta["group_size"].as_u64().unwrap() as usize);
            let (_, packed) = split_mlx(tensors.clone(), &config, &identity).unwrap();
            let (rows, cols) = (packed[0].2[0] as usize, packed[0].2[1] as usize);
            // the fixtures carry float32 scales from mx.quantize: native only if they happen to be exact in 16 bits ...
            let original = packed[0].1.mlx().expect("MLX tensor");
            if original.native().is_none() {
                assert!(original.encode_gguf_row(0, &mut Vec::new()).is_err(), "{name}");
                refused += 1;
            }
            // ... so also run them with the scales and biases rounded to half or bfloat16 (what mlx-community checkpoints store)
            let tensor = |suffix: &str| floats(tensors.iter().find(|t| t.name.ends_with(suffix)).unwrap()).unwrap();
            let words = tensors.iter().find(|t| t.name.ends_with(".weight")).unwrap().payload.clone();
            for bf16 in [false, true] {
                let round = |v: &f32| if bf16 { half::bf16::from_f32(*v).to_f32() } else { half::f16::from_f32(*v).to_f32() };
                let m = MlxAffine::new(words.clone(), tensor(".scales").iter().map(round).collect(), tensor(".biases").iter().map(round).collect(), bits, group, rows, cols).unwrap();
                let native = m.native().unwrap_or_else(|| panic!("{name}: {}", m.native_refusal()));
                assert_eq!((native.bits as usize, native.group as usize), (bits, group));
                let mut native_values = Vec::new();
                for r in 0..rows {
                    let mut row = Vec::new();
                    m.encode_gguf_row(r, &mut row).unwrap();
                    assert_eq!(row.len(), cols / group * native.block_bytes() as usize, "{name}");
                    native_values.extend(decode_mlxa(&row, native, cols));
                }
                let mut expanded = vec![0f32; rows * cols];
                m.decode_chunk(0, &mut expanded).unwrap();
                assert!(native_values.iter().zip(&expanded).all(|(a, b)| a.to_bits() == b.to_bits()), "{name}: native blocks decode differently");
            }
            seen += 1;
        }
        assert!(seen >= 12, "only {seen} MLX fixtures");
        assert!(refused > 0, "the float32 fixtures should include scales without an exact 16-bit form");
    }

    /// MLX packing of `codes` (`bits` wide, one row): code `i` at bit `i * bits` of a little-endian bit stream of 32-bit words.
    fn pack_words(codes: &[u32], bits: usize) -> Vec<u8> {
        let mut words = vec![0u32; (codes.len() * bits).div_ceil(32)];
        for (i, &c) in codes.iter().enumerate() {
            let (w, s) = (i * bits / 32, (i * bits) % 32);
            words[w] |= c << s;
            if s + bits > 32 {
                words[w + 1] |= c >> (32 - s);
            }
        }
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    fn synthetic(bits: usize, group: usize, rows: usize, cols: usize, seed: u32, scale_of: impl Fn(u32) -> f32) -> (MlxAffine, Vec<u32>) {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(12345);
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let codes: Vec<u32> = (0..rows * cols).map(|_| next() & ((1 << bits) - 1)).collect();
        let bytes: Vec<u8> = (0..rows).flat_map(|r| pack_words(&codes[r * cols..(r + 1) * cols], bits)).collect();
        let groups = rows * cols / group;
        let scales: Vec<f32> = (0..groups).map(|_| scale_of(next())).collect();
        let biases: Vec<f32> = (0..groups).map(|_| scale_of(next()) * -8.0).collect();
        let len = bytes.len();
        let m = MlxAffine::new(Payload::Owned(Arc::new(bytes), 0..len), scales, biases, bits, group, rows, cols).unwrap();
        (m, codes)
    }

    #[test]
    fn native_blocks_unpack_every_bit_width_including_codes_that_straddle_words() {
        for bits in [2usize, 3, 4, 5, 6, 8] {
            for group in [32usize, 64, 128] {
                // cols a multiple of the group and of 32 / gcd: 3, 5 and 6 bit rows end inside a word unless cols % 32 == 0
                let (rows, cols) = (3, group * 3);
                let (m, codes) = synthetic(bits, group, rows, cols, (bits * 1000 + group) as u32, |r| half::f16::from_f32(((r % 1000) as f32 + 1.0) / 4096.0).to_f32());
                let native = m.native().expect("f16 scales are exact");
                assert!(!native.bf16);
                for r in 0..rows {
                    let mut row = Vec::new();
                    m.encode_gguf_row(r, &mut row).unwrap();
                    let got = decode_mlxa(&row, native, cols);
                    let mut want = vec![0f32; cols];
                    m.decode_chunk(r * cols, &mut want).unwrap();
                    assert_eq!(got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "{bits} bits, group {group}, row {r}");
                    // and the codes are the ones that were packed
                    let per_row = cols / group;
                    for g in 0..per_row {
                        let (scale, bias) = (m.scales[r * per_row + g], m.biases[r * per_row + g]);
                        for i in 0..group {
                            let q = codes[r * cols + g * group + i];
                            assert_eq!(got[g * group + i].to_bits(), (q as f32 * scale + bias).to_bits());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn scales_pick_half_then_bfloat16_and_anything_else_is_not_native() {
        // exact in f16
        let (m, _) = synthetic(4, 64, 2, 128, 1, |r| half::f16::from_f32((r % 97) as f32 / 64.0 + 0.25).to_f32());
        assert_eq!(m.native().map(|n| n.bf16), Some(false));
        // a bfloat16 value below the smallest subnormal of half (2^-24): only bfloat16 holds it
        let (m, _) = synthetic(4, 64, 2, 128, 2, |r| half::bf16::from_f32(1.0 / (1u32 << 30) as f32 * (1.0 + (r % 8) as f32 / 8.0)).to_f32());
        assert_eq!(m.native().map(|n| n.bf16), Some(true));
        // f32 scales with a full mantissa: neither format is exact, the weight is not stored natively
        let (m, _) = synthetic(4, 64, 2, 128, 3, |r| 0.1 + (r % 1000) as f32 * 1e-5);
        assert_eq!(m.native(), None);
        assert!(m.native_refusal().contains("exactly"));
        assert!(m.encode_gguf_row(0, &mut Vec::new()).is_err());
        // 7 bits and groups that are no multiple of 32 expand
        let (m, _) = synthetic(7, 32, 1, 32, 4, |r| half::f16::from_f32((r % 8) as f32).to_f32());
        assert_eq!(m.native(), None);
        let (m, _) = synthetic(4, 16, 1, 64, 5, |r| half::f16::from_f32((r % 8) as f32).to_f32());
        assert_eq!(m.native(), None);
        // non-finite values are never exact
        let (m, _) = synthetic(4, 32, 1, 32, 6, |_| f32::INFINITY);
        assert_eq!(m.native(), None);
    }

    #[test]
    fn mlx_dtype_names_round_trip_to_type_ids() {
        for (bits, group, bf16) in [(4u32, 64u32, false), (3, 32, true), (8, 128, false), (6, 96, true)] {
            let m = fllm_gguf_probe::MlxAffine { bits, group, bf16 };
            let name = mlx_dtype_name(m);
            assert_eq!(mlx_dtype_id(&name), Some(m.type_id()), "{name}");
        }
        assert_eq!(mlx_dtype_name(fllm_gguf_probe::MlxAffine { bits: 4, group: 64, bf16: true }), "MLXA4G64B");
        for bad in ["MLXA", "MLXA7G64", "MLXA4G0", "MLXA4G48", "MLXAxG64", "Q4Z32"] {
            assert_eq!(mlx_dtype_id(bad), None, "{bad}");
        }
    }
}