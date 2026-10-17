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
#[derive(Debug)]
pub(crate) struct MlxAffine {
    words: Payload,
    scales: Vec<f32>,
    biases: Vec<f32>,
    bits: usize,
    group: usize,
    rows: usize,
    cols: usize,
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
        Ok(Self { words, scales, biases, bits, group, rows, cols })
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
}
