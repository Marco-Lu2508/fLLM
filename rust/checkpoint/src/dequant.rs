//! Weight formats the converter expands to F16/F32 (or maps onto an existing compact type) because the engine has no
//! kernel of their own: bitsandbytes 4-bit (NF4, FP4, double quantization) and 8-bit (LLM.int8), OCP MXFP4 blocks of
//! compressed-tensors checkpoints and the other formats listed in `docs/changes/formats.md`.
//!
//! Every decoder implements [`Dequant`]: it produces row-major `[out_features, in_features]` f32 values on demand, in
//! bounded chunks, exactly as the reference library computes them (checked against fixtures written by the real
//! libraries, see `scripts/make_quant_fixtures.py`).
use crate::safetensors::{Payload, Tensor};
use serde_json::Value;
use std::collections::HashMap;

/// A packed weight that can be expanded to f32 values chunk by chunk.
pub trait Dequant: Send + Sync + std::fmt::Debug {
    /// Number of weight elements (`rows * cols`).
    fn element_count(&self) -> usize;
    /// Fill `dst` with the elements `start..start + dst.len()` in row-major order.
    fn decode_chunk(&self, start: usize, dst: &mut [f32]) -> Result<(), String>;
    /// The MLX affine tensor behind this reader, which the converter can store natively (`MLXA` tensor types).
    fn mlx(&self) -> Option<&crate::dequant_bits::MlxAffine> {
        None
    }
}

fn range(start: usize, len: usize, total: usize) -> Result<std::ops::Range<usize>, String> {
    let end = start.checked_add(len).filter(|&e| e <= total).ok_or_else(|| format!("range {start}+{len} exceeds {total} elements"))?;
    Ok(start..end)
}

fn f32_values(t: &Tensor) -> Result<Vec<f32>, String> {
    let b = t.payload.as_slice();
    match t.dtype.as_str() {
        "F32" => Ok(b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()),
        "F16" => Ok(b.as_chunks::<2>().0.iter().map(|c| half::f16::from_bits(u16::from_le_bytes(*c)).to_f32()).collect()),
        "BF16" => Ok(b.as_chunks::<2>().0.iter().map(|c| f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16)).collect()),
        other => Err(format!("{}: expected a float tensor, found {other}", t.name)),
    }
}

// ---------------------------------------------------------------- bitsandbytes

/// bitsandbytes 4-bit: two codes per byte (the first element in the high nibble), a 16-entry code book and one absolute
/// maximum per `blocksize` elements, optionally stored as 8-bit codes themselves (double quantization).
#[derive(Debug)]
pub(crate) struct Bnb4 {
    packed: Payload,
    code: [f32; 16],
    absmax: Vec<f32>,
    blocksize: usize,
    elements: usize,
}

impl Bnb4 {
    pub(crate) fn new(packed: Payload, code: [f32; 16], absmax: Vec<f32>, blocksize: usize, elements: usize) -> Result<Self, String> {
        if blocksize == 0 || elements == 0 {
            return Err("bitsandbytes block size and element count must be nonzero".into());
        }
        if packed.as_slice().len() != elements.div_ceil(2) {
            return Err(format!("bitsandbytes 4-bit weight has {} bytes, expected {}", packed.as_slice().len(), elements.div_ceil(2)));
        }
        if absmax.len() != elements.div_ceil(blocksize) {
            return Err(format!("bitsandbytes absmax has {} entries, expected {}", absmax.len(), elements.div_ceil(blocksize)));
        }
        Ok(Self { packed, code, absmax, blocksize, elements })
    }
}

impl Dequant for Bnb4 {
    fn element_count(&self) -> usize {
        self.elements
    }

    fn decode_chunk(&self, start: usize, dst: &mut [f32]) -> Result<(), String> {
        let r = range(start, dst.len(), self.elements)?;
        let bytes = self.packed.as_slice();
        for (slot, e) in dst.iter_mut().zip(r) {
            let byte = bytes[e / 2];
            let nibble = if e % 2 == 0 { byte >> 4 } else { byte & 0x0f };
            *slot = self.code[nibble as usize] * self.absmax[e / self.blocksize];
        }
        Ok(())
    }
}

/// LLM.int8(): int8 weights, one absmax statistic (`SCB`) per output row; `w = q * SCB / 127`.
#[derive(Debug)]
pub(crate) struct Bnb8 {
    weight: Payload,
    scb: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl Dequant for Bnb8 {
    fn element_count(&self) -> usize {
        self.rows * self.cols
    }

    fn decode_chunk(&self, start: usize, dst: &mut [f32]) -> Result<(), String> {
        let r = range(start, dst.len(), self.rows * self.cols)?;
        let bytes = self.weight.as_slice();
        for (slot, e) in dst.iter_mut().zip(r) {
            *slot = f32::from(bytes[e] as i8) * self.scb[e / self.cols] * (1.0 / 127.0);
        }
        Ok(())
    }
}

/// Parses the JSON blob of `<name>.weight.quant_state.bitsandbytes__{nf4,fp4}`.
fn quant_state(t: &Tensor) -> Result<Value, String> {
    serde_json::from_slice(t.payload.as_slice()).map_err(|e| format!("{}: invalid bitsandbytes quant_state JSON: {e}", t.name))
}

/// Splits bitsandbytes tensors out of a checkpoint: returns the remaining dense tensors and the expandable weights.
pub(crate) fn split_bnb(
    all: Vec<Tensor>,
    map: &dyn Fn(&str) -> Result<Option<String>, String>,
) -> Result<(Vec<Tensor>, Vec<crate::convert::PackedGroup>), String> {
    const COMPANIONS: [&str; 4] = ["absmax", "quant_map", "nested_absmax", "nested_quant_map"];
    let mut dense = Vec::new();
    let mut weights: Vec<Tensor> = Vec::new();
    let mut companions: HashMap<String, HashMap<String, Tensor>> = HashMap::new();
    let mut scb: HashMap<String, Tensor> = HashMap::new();
    for t in all {
        let name = t.name.clone();
        if let Some(base) = name.strip_suffix(".SCB") {
            scb.insert(base.to_owned(), t);
        } else if name.ends_with(".CB") || name.ends_with(".weight_format") {
            // older LLM.int8 checkpoints also carry the unscaled copy and a format tag
        } else if let Some((base, suffix)) = name.split_once(".weight.") {
            let key = if COMPANIONS.contains(&suffix) {
                suffix.to_owned()
            } else if suffix.starts_with("quant_state.bitsandbytes__") {
                "quant_state".to_owned()
            } else {
                return Err(format!("unsupported bitsandbytes tensor {name}"));
            };
            companions.entry(format!("{base}.weight")).or_default().insert(key, t);
        } else if name.ends_with(".weight") && t.shape.len() == 2 && ((t.dtype == "U8" && t.shape[1] == 1) || t.dtype == "I8") {
            // 4-bit codes are stored as a [bytes, 1] column; int8 weights keep their [out, in] shape
            weights.push(t);
        } else {
            dense.push(t);
        }
    }
    let mut packed = Vec::new();
    for w in weights {
        let base = w.name.strip_suffix(".weight").unwrap_or(&w.name).to_owned();
        let mapped = map(&w.name)?.ok_or_else(|| format!("packed tensor {:?} maps to a skipped tensor", w.name))?;
        if w.dtype == "I8" {
            let s = scb.remove(&base).ok_or_else(|| format!("int8 weight {} has no {base}.SCB", w.name))?;
            let (rows, cols) = (w.shape[0] as usize, w.shape[1] as usize);
            let scb_values = f32_values(&s)?;
            if scb_values.len() != rows {
                return Err(format!("{} has {} entries, expected one per output row ({rows})", s.name, scb_values.len()));
            }
            let d = Bnb8 { weight: w.payload.clone(), scb: scb_values, rows, cols };
            packed.push((mapped, crate::packed_quant::PackedData::Dyn(std::sync::Arc::new(d)), vec![rows as u64, cols as u64], w.payload));
            continue;
        }
        let mut c = companions.remove(&w.name).ok_or_else(|| format!("4-bit weight {} has no bitsandbytes quant_state tensors", w.name))?;
        let mut take = |k: &str| c.remove(k).ok_or_else(|| format!("4-bit weight {} has no {k} tensor", w.name));
        let state = quant_state(&take("quant_state")?)?;
        let shape: Vec<u64> = state
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{}: quant_state has no shape", w.name))?
            .iter()
            .filter_map(Value::as_u64)
            .collect();
        if shape.len() != 2 {
            return Err(format!("{}: bitsandbytes weight shape {shape:?} is not a matrix", w.name));
        }
        let blocksize = state.get("blocksize").and_then(Value::as_u64).ok_or_else(|| format!("{}: quant_state has no blocksize", w.name))? as usize;
        let code_values = f32_values(&take("quant_map")?)?;
        let code: [f32; 16] = code_values.try_into().map_err(|v: Vec<f32>| format!("{}: quant_map has {} entries, expected 16", w.name, v.len()))?;
        let absmax_tensor = take("absmax")?;
        let absmax = if absmax_tensor.dtype == "U8" {
            // double quantization: the absmax values are 8-bit codes of a second, blockwise quantizer
            let nested_blocksize = state.get("nested_blocksize").and_then(Value::as_u64).ok_or_else(|| format!("{}: nested quant_state has no nested_blocksize", w.name))? as usize;
            let offset = state.get("nested_offset").and_then(Value::as_f64).ok_or_else(|| format!("{}: nested quant_state has no nested_offset", w.name))? as f32;
            let nested_code = f32_values(&take("nested_quant_map")?)?;
            let nested_absmax = f32_values(&take("nested_absmax")?)?;
            if nested_code.len() != 256 || nested_blocksize == 0 || nested_absmax.len() != absmax_tensor.payload.as_slice().len().div_ceil(nested_blocksize) {
                return Err(format!("{}: inconsistent bitsandbytes nested statistics", w.name));
            }
            absmax_tensor.payload.as_slice().iter().enumerate().map(|(i, &q)| nested_code[q as usize] * nested_absmax[i / nested_blocksize] + offset).collect()
        } else {
            f32_values(&absmax_tensor)?
        };
        if let Some(extra) = c.keys().next() {
            return Err(format!("{}: unexpected bitsandbytes tensor {extra}", w.name));
        }
        let (rows, cols) = (shape[0] as usize, shape[1] as usize);
        let d = Bnb4::new(w.payload.clone(), code, absmax, blocksize, rows * cols).map_err(|e| format!("{}: {e}", w.name))?;
        packed.push((mapped, crate::packed_quant::PackedData::Dyn(std::sync::Arc::new(d)), shape, w.payload));
    }
    if let Some(left) = scb.keys().next() {
        return Err(format!("{left}.SCB has no int8 weight"));
    }
    if let Some(left) = companions.keys().next() {
        return Err(format!("bitsandbytes tensors of {left} have no packed weight"));
    }
    if packed.is_empty() {
        return Err("quantization_config declares bitsandbytes but no 4-bit or int8 weights were found".into());
    }
    Ok((dense, packed))
}

// ---------------------------------------------------------------- OCP MXFP4

/// OCP MXFP4 weights of compressed-tensors (`mxfp4-pack-quantized`): two E2M1 codes per byte (even element in the low
/// nibble) and one E8M0 exponent per 32 inputs. Written as GGUF type MXFP4 (`[e8m0][16 bytes]` per block).
#[derive(Debug, Clone)]
pub struct Mxfp4 {
    weight: Payload,
    scales: Payload,
    rows: usize,
    cols: usize,
}

/// Value of an E8M0 scale byte: `2^(e - 127)`; the lowest exponent is the f32 subnormal `2^-127`.
fn e8m0(e: u8) -> f32 {
    if e == 0 {
        f32::from_bits(1 << 22)
    } else {
        f32::from_bits(u32::from(e) << 23)
    }
}

const E2M1: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

impl Mxfp4 {
    pub(crate) fn new(weight: Payload, scales: Payload, rows: usize, cols: usize) -> Result<Self, String> {
        if cols == 0 || !cols.is_multiple_of(32) {
            return Err(format!("MXFP4 needs a multiple of 32 inputs per row, found {cols}"));
        }
        if weight.as_slice().len() != rows * cols / 2 {
            return Err(format!("MXFP4 weight has {} bytes, expected {}", weight.as_slice().len(), rows * cols / 2));
        }
        if scales.as_slice().len() != rows * cols / 32 {
            return Err(format!("MXFP4 scale tensor has {} entries, expected {}", scales.as_slice().len(), rows * cols / 32));
        }
        if scales.as_slice().contains(&0xff) {
            return Err("MXFP4 scale tensor contains the E8M0 NaN code 0xff".into());
        }
        Ok(Self { weight, scales, rows, cols })
    }

    pub fn element_count(&self) -> usize {
        self.rows * self.cols
    }

    pub fn decode_chunk(&self, start: usize, dst: &mut [f32]) -> Result<(), String> {
        let r = range(start, dst.len(), self.element_count())?;
        let (w, s) = (self.weight.as_slice(), self.scales.as_slice());
        for (slot, e) in dst.iter_mut().zip(r) {
            let byte = w[e / 2];
            let code = if e % 2 == 0 { byte & 0x0f } else { byte >> 4 };
            *slot = E2M1[code as usize] * e8m0(s[e / 32]);
        }
        Ok(())
    }

    /// Bytes of one GGUF row: per block the E8M0 scale, then byte j holds elements j (low nibble) and j + 16 (high).
    #[cfg(test)]
    pub fn gguf_row_bytes(&self) -> usize {
        self.cols / 32 * 17
    }

    pub fn encode_gguf_row(&self, row: usize, dst: &mut Vec<u8>) -> Result<(), String> {
        if row >= self.rows {
            return Err(format!("row {row} is outside the {} MXFP4 rows", self.rows));
        }
        let (w, s) = (self.weight.as_slice(), self.scales.as_slice());
        let blocks = self.cols / 32;
        for b in 0..blocks {
            let at = row * blocks + b;
            let bytes = &w[at * 16..(at + 1) * 16];
            let element = |i: usize| if i.is_multiple_of(2) { bytes[i / 2] & 0xf } else { bytes[i / 2] >> 4 };
            dst.push(s[at]);
            dst.extend((0..16).map(|j| element(j) | (element(j + 16) << 4)));
        }
        Ok(())
    }
}

/// compressed-tensors `mxfp4-pack-quantized`: `weight_packed` (U8) and `weight_scale` (U8 E8M0) per linear.
pub(crate) fn split_ct_mxfp4(
    all: Vec<Tensor>,
    map: &dyn Fn(&str) -> Result<Option<String>, String>,
) -> Result<(Vec<Tensor>, Vec<crate::convert::PackedGroup>), String> {
    let mut dense = Vec::new();
    let mut weights = Vec::new();
    let mut scales: HashMap<String, Tensor> = HashMap::new();
    for t in all {
        if let Some(base) = t.name.strip_suffix(".weight_scale") {
            scales.insert(base.to_owned(), t);
        } else if t.name.ends_with(".input_global_scale") {
            // activations are not quantized by the engine
        } else if t.name.ends_with(".weight_packed") {
            weights.push(t);
        } else {
            dense.push(t);
        }
    }
    let mut packed = Vec::new();
    for w in weights {
        let base = w.name.strip_suffix(".weight_packed").unwrap_or(&w.name).to_owned();
        let s = scales.remove(&base).ok_or_else(|| format!("MXFP4 weight {} has no {base}.weight_scale", w.name))?;
        if w.dtype != "U8" || w.shape.len() != 2 {
            return Err(format!("{} must be a rank-2 U8 tensor, found {} {:?}", w.name, w.dtype, w.shape));
        }
        if !matches!(s.dtype.as_str(), "U8" | "F8_E8M0") {
            return Err(format!("{} must be U8 (E8M0 exponents), found {}", s.name, s.dtype));
        }
        let (rows, cols) = (w.shape[0] as usize, w.shape[1] as usize * 2);
        let m = Mxfp4::new(w.payload.clone(), s.payload, rows, cols).map_err(|e| format!("{}: {e}", w.name))?;
        let mapped = map(&format!("{base}.weight"))?.ok_or_else(|| format!("packed tensor {:?} maps to a skipped tensor", w.name))?;
        packed.push((mapped, crate::packed_quant::PackedData::Mxfp4(m), vec![rows as u64, cols as u64], w.payload));
    }
    if let Some(left) = scales.keys().next() {
        return Err(format!("{left}.weight_scale has no packed weight"));
    }
    if packed.is_empty() {
        return Err("quantization_config declares MXFP4 but no weight_packed tensors were found".into());
    }
    Ok((dense, packed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::PackedGroup;
    use std::path::PathBuf;

    pub(crate) fn fixture(case: &str) -> (Vec<Tensor>, Vec<f32>) {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata").join("quant").join(format!("{case}.safetensors"));
        let mut tensors = crate::safetensors::read_file(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let at = tensors.iter().position(|t| t.name == "expected").expect("fixture has an expected tensor");
        let expected = tensors.swap_remove(at);
        let values = f32_values(&expected).unwrap();
        (tensors, values)
    }

    fn decode_all(group: &PackedGroup) -> Vec<f32> {
        let mut out = vec![0f32; group.1.element_count()];
        for (i, chunk) in out.chunks_mut(37).enumerate() {
            group.1.decode_f32_chunk(i * 37, chunk).unwrap();
        }
        out
    }

    fn identity(n: &str) -> Result<Option<String>, String> {
        Ok(Some(n.to_owned()))
    }

    fn assert_close(case: &str, got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len(), "{case}");
        let mut worst = (0usize, 0f32);
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let d = (g - w).abs();
            if d > worst.1 {
                worst = (i, d);
            }
        }
        assert!(worst.1 <= tol, "{case}: max error {} at element {} (got {}, want {}), tolerance {tol}", worst.1, worst.0, got[worst.0], want[worst.0]);
    }

    #[test]
    fn bitsandbytes_4bit_matches_the_library() {
        for case in ["nf4", "fp4", "nf4_dq", "fp4_dq"] {
            for bs in [64, 128] {
                let name = format!("bnb_{case}_bs{bs}");
                let (tensors, expected) = fixture(&name);
                let (dense, packed) = split_bnb(tensors, &identity).unwrap();
                assert!(dense.is_empty(), "{name}");
                assert_eq!(packed.len(), 1);
                assert_close(&name, &decode_all(&packed[0]), &expected, 1e-6);
                assert_eq!(packed[0].2, vec![16, 256]);
            }
        }
    }

    #[test]
    fn bitsandbytes_int8_matches_the_library() {
        let (tensors, expected) = fixture("bnb_int8");
        let (dense, packed) = split_bnb(tensors, &identity).unwrap();
        assert!(dense.is_empty());
        assert_close("bnb_int8", &decode_all(&packed[0]), &expected, 1e-6);
    }

    #[test]
    fn bitsandbytes_errors_name_the_missing_piece() {
        let (mut tensors, _) = fixture("bnb_nf4_bs64");
        tensors.retain(|t| !t.name.ends_with(".absmax"));
        assert!(split_bnb(tensors, &identity).unwrap_err().contains("absmax"));
        let (mut tensors, _) = fixture("bnb_int8");
        tensors.retain(|t| !t.name.ends_with(".SCB"));
        assert!(split_bnb(tensors, &identity).unwrap_err().contains("SCB"));
    }

    /// Reads `n` MXFP4 blocks of a GGUF tensor as f32 (scale `2^(e-127)`, byte j = elements j and j + 16).
    fn decode_mxfp4(bytes: &[u8]) -> Vec<f32> {
        let mut out = Vec::new();
        for block in bytes.chunks(17) {
            let scale = e8m0(block[0]);
            out.extend((0..16).map(|j| E2M1[(block[1 + j] & 0xf) as usize] * scale));
            out.extend((0..16).map(|j| E2M1[(block[1 + j] >> 4) as usize] * scale));
        }
        out
    }

    /// A gpt-oss checkpoint in the published `quant_method: mxfp4` layout converts (it was refused before the `mxfp4`
    /// method was recognized) and the experts decode to what `transformers` dequantizes.
    #[test]
    fn gpt_oss_mxfp4_checkpoint_converts_and_decodes_like_transformers() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata").join("gptoss_mxfp4");
        let out = std::env::temp_dir().join(format!("fllm-gptoss-mxfp4-{}.gguf", std::process::id()));
        let _ = std::fs::remove_file(&out);
        let report = crate::convert_checkpoint(&dir, &out, crate::ConvertOptions::default()).unwrap();
        assert_eq!(report.architecture, "gpt-oss");
        let gguf = fllm_gguf_probe::MappedGguf::open_single(&out).unwrap();
        let expected = crate::safetensors::read_file(&dir.join("expected.safetensors")).unwrap();
        let bf16 = |name: &str| -> Vec<f32> {
            let t = expected.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("{name}"));
            f32_values(t).unwrap()
        };
        let (e, h, f) = (2usize, 64usize, 64usize);
        for layer in 0..2 {
            let gate_up = bf16(&format!("model.layers.{layer}.mlp.experts.gate_up_proj")); // [E, H, 2F], gate/up interleaved
            let down = bf16(&format!("model.layers.{layer}.mlp.experts.down_proj")); // [E, F, H]
            for (k, part) in ["ffn_gate_exps", "ffn_up_exps"].iter().enumerate() {
                let got = decode_mxfp4(gguf.tensor_bytes(&format!("blk.{layer}.{part}.weight")).unwrap());
                assert_eq!(got.len(), e * f * h);
                for ex in 0..e {
                    for out_row in 0..f {
                        for inp in 0..h {
                            let want = gate_up[(ex * h + inp) * 2 * f + 2 * out_row + k];
                            assert_eq!(got[(ex * f + out_row) * h + inp], want, "layer {layer} {part} expert {ex} row {out_row} col {inp}");
                        }
                    }
                }
            }
            let got = decode_mxfp4(gguf.tensor_bytes(&format!("blk.{layer}.ffn_down_exps.weight")).unwrap());
            for ex in 0..e {
                for out_row in 0..h {
                    for inp in 0..f {
                        let want = down[(ex * f + inp) * h + out_row];
                        assert_eq!(got[(ex * h + out_row) * f + inp], want, "layer {layer} down expert {ex} row {out_row} col {inp}");
                    }
                }
            }
        }
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn compressed_tensors_nvfp4_matches_the_library() {
        let (tensors, expected) = fixture("ct_nvfp4");
        let (dense, packed) = crate::convert::split_packed(tensors, crate::convert::Quantization::Nvfp4, None, &identity).unwrap();
        assert!(dense.is_empty());
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0].0, "layer.weight");
        // the library divides the block scale by weight_global_scale; the converter multiplies by its reciprocal
        assert_close("ct_nvfp4", &decode_all(&packed[0]), &expected, 1e-5);
        let crate::packed_quant::PackedData::Nvfp4(t) = &packed[0].1 else { panic!("not NVFP4") };
        assert!(t.input_scale().is_some_and(|v| (v - 3.1 / (448.0 * 6.0)).abs() < 1e-6), "{:?}", t.input_scale());
    }

    #[test]
    fn compressed_tensors_mxfp4_matches_the_library() {
        let (tensors, expected) = fixture("ct_mxfp4");
        let (dense, packed) = split_ct_mxfp4(tensors, &identity).unwrap();
        assert!(dense.is_empty());
        assert_eq!(packed[0].0, "layer.weight");
        assert_close("ct_mxfp4", &decode_all(&packed[0]), &expected, 1e-6);
        // the GGUF rows decode to the same values: scale * code table, byte j = elements j and j + 16
        let crate::packed_quant::PackedData::Mxfp4(m) = &packed[0].1 else { panic!("not MXFP4") };
        for row in 0..m.rows {
            let mut bytes = Vec::new();
            m.encode_gguf_row(row, &mut bytes).unwrap();
            assert_eq!(bytes.len(), m.gguf_row_bytes());
            for (b, block) in bytes.chunks(17).enumerate() {
                let scale = e8m0(block[0]);
                for j in 0..16 {
                    for (half, shift) in [(0usize, 0), (16, 4)] {
                        let got = E2M1[((block[1 + j] >> shift) & 0xf) as usize] * scale;
                        assert_eq!(got, expected[row * m.cols + b * 32 + j + half], "row {row} block {b} element {}", j + half);
                    }
                }
            }
        }
    }
}
