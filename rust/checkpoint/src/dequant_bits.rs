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
/// MLX `mxfp4` / `nvfp4` / `mxfp8` modes of `mx.quantize`: `weight` `[out, in * bits / 32]` uint32 holds the codes of one row
/// (E2M1 nibbles, the even element in the low nibble, or E4M3 bytes), `scales` `[out, in / group]` uint8 one block scale:
/// E8M0 (`2^(e-127)`) for `mxfp4` / `mxfp8` (group 32), E4M3 for `nvfp4` (group 16). There are no biases and no per-tensor factor.
#[derive(Debug)]
pub(crate) struct MlxFp {
    weight: Payload,
    scales: Payload,
    mode: MlxFpMode,
    rows: usize,
    cols: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MlxFpMode {
    Mxfp4,
    Nvfp4,
    Mxfp8,
}

impl MlxFpMode {
    pub(crate) fn parse(name: &str) -> Option<Self> {
        match name {
            "mxfp4" => Some(Self::Mxfp4),
            "nvfp4" => Some(Self::Nvfp4),
            "mxfp8" => Some(Self::Mxfp8),
            _ => None,
        }
    }
    fn group(self) -> usize {
        if self == Self::Nvfp4 { 16 } else { 32 }
    }
    fn code_bits(self) -> usize {
        if self == Self::Mxfp8 { 8 } else { 4 }
    }
}

impl MlxFp {
    fn new(weight: Payload, scales: Payload, mode: MlxFpMode, rows: usize, cols: usize) -> Result<Self, String> {
        let group = mode.group();
        if cols == 0 || !cols.is_multiple_of(group) || !(cols * mode.code_bits()).is_multiple_of(32) {
            return Err(format!("{mode:?} needs a multiple of {group} inputs per row, found {cols}"));
        }
        if weight.as_slice().len() != rows * cols * mode.code_bits() / 8 {
            return Err(format!("{mode:?} weight has {} bytes, expected {}", weight.as_slice().len(), rows * cols * mode.code_bits() / 8));
        }
        if scales.as_slice().len() != rows * cols / group {
            return Err(format!("{mode:?} scales have {} entries, expected {}", scales.as_slice().len(), rows * cols / group));
        }
        Ok(Self { weight, scales, mode, rows, cols })
    }
}

impl Dequant for MlxFp {
    fn element_count(&self) -> usize {
        self.rows * self.cols
    }

    fn decode_chunk(&self, start: usize, dst: &mut [f32]) -> Result<(), String> {
        let end = start.checked_add(dst.len()).filter(|&e| e <= self.element_count()).ok_or("MLX range exceeds the tensor")?;
        let (w, s) = (self.weight.as_slice(), self.scales.as_slice());
        for (slot, e) in dst.iter_mut().zip(start..end) {
            let block = s[e / self.mode.group()];
            *slot = match self.mode {
                MlxFpMode::Mxfp4 | MlxFpMode::Nvfp4 => {
                    let byte = w[e / 2];
                    let code = if e % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                    let scale = if self.mode == MlxFpMode::Mxfp4 { crate::dequant::e8m0(block) } else { crate::nvfp4::e4m3_value(block) };
                    crate::dequant::E2M1[code as usize] * scale
                }
                MlxFpMode::Mxfp8 => {
                    let code = w[e];
                    let magnitude = crate::nvfp4::e4m3_value(code & 0x7f);
                    (if code & 0x80 != 0 { -magnitude } else { magnitude }) * crate::dequant::e8m0(block)
                }
            };
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
    let fp_mode = config.get("mode").and_then(Value::as_str).and_then(MlxFpMode::parse);
    for w in weights {
        let base = w.name.strip_suffix(".weight").unwrap_or(&w.name).to_owned();
        let s = scales.remove(&base).ok_or_else(|| format!("MLX weight {} has no {base}.scales", w.name))?;
        if let Some(mode) = fp_mode {
            if w.shape.len() != 2 || s.dtype != "U8" {
                return Err(format!("{}: {mode:?} needs a matrix with U8 block scales (found {:?}, scales {})", w.name, w.shape, s.dtype));
            }
            let rows = w.shape[0] as usize;
            let cols = w.shape[1] as usize * 32 / mode.code_bits();
            let m = MlxFp::new(w.payload.clone(), s.payload, mode, rows, cols).map_err(|e| format!("{}: {e}", w.name))?;
            let mapped = map(&w.name)?.ok_or_else(|| format!("packed tensor {:?} maps to a skipped tensor", w.name))?;
            packed.push((mapped, crate::packed_quant::PackedData::Dyn(Arc::new(m)), vec![rows as u64, cols as u64], w.payload));
            continue;
        }
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

// ---------------------------------------------------------------- GPTQ-Marlin repacked checkpoints

/// Symmetric 4- or 8-bit weights in the Marlin layout (IST-DASLab Marlin `B` / `s` tensors, GPTQModel / vLLM
/// `checkpoint_format: marlin`), read back to plain `[out, in]` values.
///
/// The permutation is the one of vLLM's `marlin_weights` / `marlin_permute_scales` (Apache-2.0): the `[in, out]` code matrix is cut
/// into 16 x 16 tiles stored tile after tile, every run of 1024 codes is permuted (`perm`, with the interleave 0 2 4 6 1 3 5 7 for
/// 4 bits, 0 2 1 3 for 8 bits) and `32 / bits` consecutive codes go into one int32, the first in the low bits. `qweight` is
/// `[in / 16, out * 16 / (32 / bits)]`. Scales `[groups, out]` are permuted in runs of 64 values (grouped) or 32 values (one
/// group, channel-wise). A code stands for `(code - 2^(bits-1)) * scale`.
#[derive(Debug)]
pub(crate) struct GptqMarlin {
    qweight: Payload,
    scales: Vec<f32>,
    inverse_perm: Vec<usize>,
    inverse_scale_perm: Vec<usize>,
    bits: usize,
    group: usize,
    in_features: usize,
    out_features: usize,
}

/// `perm[j]`: position in the tile-ordered run of 1024 codes that the j-th stored code comes from.
fn marlin_weight_perm(bits: usize) -> Vec<usize> {
    let mut perm = Vec::with_capacity(1024);
    for i in 0..32usize {
        let col = i / 4;
        let mut one = Vec::with_capacity(8);
        for block in 0..2 {
            for row in [2 * (i % 4), 2 * (i % 4) + 1, 2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1] {
                one.push(16 * row + col + 8 * block);
            }
        }
        for j in 0..4 {
            perm.extend(one.iter().map(|p| p + 256 * j));
        }
    }
    let interleave: &[usize] = if bits == 4 { &[0, 2, 4, 6, 1, 3, 5, 7] } else { &[0, 2, 1, 3] };
    let mut out = vec![0; perm.len()];
    for (chunk, dst) in perm.chunks(interleave.len()).zip(out.chunks_mut(interleave.len())) {
        for (d, &k) in dst.iter_mut().zip(interleave) {
            *d = chunk[k];
        }
    }
    out
}

fn marlin_scale_perm(grouped: bool) -> Vec<usize> {
    if grouped {
        (0..8).flat_map(|i| (0..8).map(move |j| i + 8 * j)).collect()
    } else {
        (0..4).flat_map(|i| [0, 1, 8, 9, 16, 17, 24, 25].map(|j| 2 * i + j)).collect()
    }
}

fn invert(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0; perm.len()];
    for (j, &p) in perm.iter().enumerate() {
        inv[p] = j;
    }
    inv
}

impl GptqMarlin {
    pub(crate) fn new(qweight: Payload, scales: Vec<f32>, bits: usize, group: usize, in_features: usize, out_features: usize) -> Result<Self, String> {
        if !matches!(bits, 4 | 8) || !in_features.is_multiple_of(16) || !out_features.is_multiple_of(64) || group == 0 || !in_features.is_multiple_of(group) {
            return Err(format!("unsupported Marlin layout: {bits} bits, group {group}, {in_features} x {out_features} (needs inputs % 16 == 0, outputs % 64 == 0)"));
        }
        let words = in_features / 16 * out_features * 16 / (32 / bits);
        if qweight.as_slice().len() != words * 4 {
            return Err(format!("Marlin weight has {} bytes, expected {}", qweight.as_slice().len(), words * 4));
        }
        let groups = in_features / group;
        if scales.len() != groups * out_features {
            return Err(format!("Marlin scales have {} values, expected {}", scales.len(), groups * out_features));
        }
        let grouped = group < in_features;
        Ok(Self {
            qweight,
            scales,
            inverse_perm: invert(&marlin_weight_perm(bits)),
            inverse_scale_perm: invert(&marlin_scale_perm(grouped)),
            bits,
            group,
            in_features,
            out_features,
        })
    }
}

impl Dequant for GptqMarlin {
    fn element_count(&self) -> usize {
        self.in_features * self.out_features
    }

    fn decode_chunk(&self, start: usize, dst: &mut [f32]) -> Result<(), String> {
        let end = start.checked_add(dst.len()).filter(|&e| e <= self.element_count()).ok_or("Marlin range exceeds the tensor")?;
        let pack = 32 / self.bits;
        let mask = (1u32 << self.bits) - 1;
        let bytes = self.qweight.as_slice();
        let words_per_row = self.out_features * 16 / pack;
        let period = self.inverse_scale_perm.len();
        for (slot, e) in dst.iter_mut().zip(start..end) {
            let (o, i) = (e / self.in_features, e % self.in_features);
            // position in the tile-ordered row, then in the permuted one
            let column = (o / 16) * 256 + (i % 16) * 16 + o % 16;
            let stored = column / 1024 * 1024 + self.inverse_perm[column % 1024];
            let at = ((i / 16) * words_per_row + stored / pack) * 4;
            let word = u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
            let code = (word >> (self.bits * (stored % pack))) & mask;
            let flat = (i / self.group) * self.out_features + o;
            let scale = self.scales[flat / period * period + self.inverse_scale_perm[flat % period]];
            *slot = (code as i32 - (1 << (self.bits - 1))) as f32 * scale;
        }
        Ok(())
    }
}

/// Marlin-format GPTQ tensors: `<name>.B` / `<name>.s` (IST-DASLab Marlin) or `<name>.qweight` / `<name>.scales` (GPTQModel); the
/// `workspace`, empty `g_idx` and `qzeros` tensors of the second layout are dropped.
pub(crate) fn split_gptq_marlin(
    all: Vec<Tensor>,
    bits: usize,
    group: i64,
    map: &dyn Fn(&str) -> Result<Option<String>, String>,
) -> Result<(Vec<Tensor>, Vec<crate::convert::PackedGroup>), String> {
    let mut dense = Vec::new();
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    let mut scales: HashMap<String, Tensor> = HashMap::new();
    for t in all {
        let n = t.name.as_str();
        if let Some(base) = n.strip_suffix(".B").or_else(|| n.strip_suffix(".qweight")) {
            weights.insert(base.to_owned(), t);
        } else if let Some(base) = n.strip_suffix(".s").or_else(|| n.strip_suffix(".scales")) {
            scales.insert(base.to_owned(), t);
        } else if n.ends_with(".workspace") || n.ends_with(".qzeros") || n.ends_with(".g_idx") {
            if n.ends_with(".g_idx") && t.shape.iter().product::<u64>() > 0 {
                return Err(format!("{n}: Marlin checkpoints with activation ordering (desc_act) are not supported"));
            }
        } else {
            dense.push(t);
        }
    }
    let mut packed = Vec::new();
    let mut bases: Vec<_> = weights.keys().cloned().collect();
    bases.sort();
    for base in bases {
        let w = weights.remove(&base).unwrap();
        let s = scales.remove(&base).ok_or_else(|| format!("Marlin weight {} has no scales", w.name))?;
        if w.dtype != "I32" || w.shape.len() != 2 || s.shape.len() != 2 {
            return Err(format!("{}: Marlin weights must be rank-2 I32 tensors with rank-2 scales", w.name));
        }
        let out_features = s.shape[1] as usize;
        let in_features = w.shape[0] as usize * 16;
        let group_size = if group <= 0 { in_features } else { group as usize };
        let d = GptqMarlin::new(w.payload.clone(), floats(&s)?, bits, group_size, in_features, out_features).map_err(|e| format!("{base}: {e}"))?;
        let mapped = map(&format!("{base}.weight"))?.ok_or_else(|| format!("packed tensor {base:?} maps to a skipped tensor"))?;
        packed.push((mapped, crate::packed_quant::PackedData::Dyn(Arc::new(d)), vec![out_features as u64, in_features as u64], w.payload));
    }
    if let Some(left) = scales.keys().next() {
        return Err(format!("{left}: Marlin scales without a packed weight"));
    }
    if packed.is_empty() {
        return Err("quantization_config declares a Marlin checkpoint but no packed weights were found".into());
    }
    Ok((dense, packed))
}

// ---------------------------------------------------------------- compressed-tensors with zero points / activation order

/// `compressed-tensors` integer weights that the symmetric reader (`symint.rs`) does not cover: asymmetric zero points
/// (`weight_zero_point`) and activation ordering (`weight_g_idx`, the quantization group of every input column).
///
/// * `pack-quantized`: `weight_packed` `[out, ceil(in * bits / 32)]` int32, the codes of one row as a dense little-endian
///   bit stream along the inputs, each stored as `code + 2^(bits-1)`. The zero points of the group / channel strategies are
///   packed the same way along the *output* dimension: `weight_zero_point` `[ceil(out * bits / 32), groups]`.
/// * `int-quantized` int8: `weight` `[out, in]` signed codes, `weight_zero_point` unpacked `[out, groups]` (or one value).
///
/// `value = (code - zero_point) * scale`.
#[derive(Debug)]
pub(crate) struct CtAffine {
    codes: Payload,
    /// Bits per packed code; `None` for plain int8 codes.
    bits: Option<usize>,
    scales: Vec<f32>,
    /// One zero point per scale (empty: all zero).
    zeros: Vec<i32>,
    g_idx: Option<Vec<u32>>,
    rows: usize,
    cols: usize,
    group: usize,
    per_tensor: bool,
}

impl CtAffine {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        codes: Payload,
        bits: Option<usize>,
        scales: Vec<f32>,
        zeros: Vec<i32>,
        g_idx: Option<Vec<u32>>,
        rows: usize,
        cols: usize,
        group: usize,
    ) -> Result<Self, String> {
        if rows == 0 || cols == 0 || group == 0 || !cols.is_multiple_of(group) || bits.is_some_and(|b| !(1..=8).contains(&b)) {
            return Err(format!("unsupported integer layout: {rows} x {cols}, group {group}, bits {bits:?}"));
        }
        let groups = cols / group;
        let row_bytes = bits.map_or(cols, |b| (cols * b).div_ceil(32) * 4);
        if codes.as_slice().len() != rows * row_bytes {
            return Err(format!("weight has {} bytes, expected {}", codes.as_slice().len(), rows * row_bytes));
        }
        let per_tensor = scales.len() == 1 && rows * groups != 1;
        if scales.len() != rows * groups && !per_tensor {
            return Err(format!("weight_scale has {} values, expected {} ([{rows}, {groups}]) or 1", scales.len(), rows * groups));
        }
        if !zeros.is_empty() && zeros.len() != scales.len() {
            return Err(format!("weight_zero_point has {} values, expected {}", zeros.len(), scales.len()));
        }
        if let Some(g) = &g_idx {
            if g.len() != cols || g.iter().any(|&x| x as usize >= groups) {
                return Err("weight_g_idx does not match the input dimension and group count".into());
            }
        }
        Ok(Self { codes, bits, scales, zeros, g_idx, rows, cols, group, per_tensor })
    }
}

impl Dequant for CtAffine {
    fn element_count(&self) -> usize {
        self.rows * self.cols
    }

    fn decode_chunk(&self, start: usize, dst: &mut [f32]) -> Result<(), String> {
        let end = start.checked_add(dst.len()).filter(|&e| e <= self.element_count()).ok_or("integer weight range exceeds the tensor")?;
        let groups = self.cols / self.group;
        let bytes = self.codes.as_slice();
        let row_bytes = self.bits.map_or(self.cols, |b| (self.cols * b).div_ceil(32) * 4);
        for (slot, e) in dst.iter_mut().zip(start..end) {
            let (r, c) = (e / self.cols, e % self.cols);
            let q = match self.bits {
                Some(b) => read_bits(&bytes[r * row_bytes..(r + 1) * row_bytes], c * b, b) as i32 - (1 << (b - 1)),
                None => i32::from(bytes[r * row_bytes + c] as i8),
            };
            let g = self.g_idx.as_ref().map_or(c / self.group, |t| t[c] as usize);
            let at = if self.per_tensor { 0 } else { r * groups + g };
            let zero = self.zeros.get(at).copied().unwrap_or(0);
            *slot = (q - zero) as f32 * self.scales[at];
        }
        Ok(())
    }
}

/// Integer value `index` of a zero-point tensor that is stored plain (not bit-packed).
fn plain_ints(t: &Tensor) -> Result<Vec<i32>, String> {
    let b = t.payload.as_slice();
    match t.dtype.as_str() {
        "I8" => Ok(b.iter().map(|&v| i32::from(v as i8)).collect()),
        "U8" => Ok(b.iter().map(|&v| i32::from(v)).collect()),
        "I16" => Ok(b.as_chunks::<2>().0.iter().map(|c| i32::from(i16::from_le_bytes(*c))).collect()),
        "I32" => Ok(b.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c)).collect()),
        "I64" => Ok(b.as_chunks::<8>().0.iter().map(|c| i64::from_le_bytes(*c) as i32).collect()),
        other => Err(format!("{}: zero points must be integers, found {other}", t.name)),
    }
}

/// `compressed-tensors` integer weights with zero points and/or `weight_g_idx`: `pack-quantized` (`packed`: `weight_packed`
/// with `bits` bits) or int8 `weight`. `zero_packed`: the zero points are bit-packed along the output dimension.
pub(crate) fn split_ct_affine(
    all: Vec<Tensor>,
    bits: usize,
    group: usize,
    packed: bool,
    zero_packed: bool,
    map: &dyn Fn(&str) -> Result<Option<String>, String>,
) -> Result<(Vec<Tensor>, Vec<crate::convert::PackedGroup>), String> {
    let mut dense = Vec::new();
    let mut parts: HashMap<String, HashMap<&'static str, Tensor>> = HashMap::new();
    for t in all {
        let n = t.name.as_str();
        let weight_name = if packed { ".weight_packed" } else { ".weight" };
        let suffix = [".weight_scale", ".weight_zero_point", ".weight_g_idx", ".weight_shape", weight_name]
            .into_iter()
            .find(|s| n.ends_with(s) && (*s != ".weight" || t.dtype == "I8"));
        match suffix {
            Some(s) => {
                let base = n[..n.len() - s.len()].to_owned();
                let key = match s {
                    ".weight_scale" => "scale",
                    ".weight_zero_point" => "zero",
                    ".weight_g_idx" => "g_idx",
                    ".weight_shape" => "shape",
                    _ => "weight",
                };
                if parts.entry(base).or_default().insert(key, t).is_some() {
                    return Err("duplicate compressed-tensors component".into());
                }
            }
            // activation quantization parameters are not stored: activations are quantized dynamically by the engine
            None if n.ends_with(".input_scale") || n.ends_with(".input_zero_point") => {}
            None => dense.push(t),
        }
    }
    let mut out = Vec::new();
    let mut bases: Vec<_> = parts.keys().cloned().collect();
    bases.sort();
    for base in bases {
        let mut p = parts.remove(&base).unwrap();
        let w = p.remove("weight").ok_or_else(|| format!("{base}: scale or zero point without an integer weight"))?;
        let scale = p.remove("scale").ok_or_else(|| format!("integer weight {} has no {base}.weight_scale", w.name))?;
        if w.shape.len() != 2 {
            return Err(format!("{}: expected a rank-2 weight", w.name));
        }
        let rows = w.shape[0] as usize;
        let cols = if packed {
            if w.dtype != "I32" {
                return Err(format!("{} must be I32, found {}", w.name, w.dtype));
            }
            match p.remove("shape") {
                Some(s) => {
                    let dims: Vec<u64> = s.payload.as_slice().as_chunks::<8>().0.iter().map(|c| u64::from_le_bytes(*c)).collect();
                    if dims.len() != 2 || dims[0] as usize != rows {
                        return Err(format!("{base}.weight_shape {dims:?} does not match the packed weight"));
                    }
                    dims[1] as usize
                }
                None => w.shape[1] as usize * 32 / bits,
            }
        } else {
            w.shape[1] as usize
        };
        let scales = floats(&scale)?;
        let group_size = if group == 0 { cols } else { group };
        let groups = cols / group_size.max(1);
        let zeros = match p.remove("zero") {
            None => Vec::new(),
            Some(z) if packed && zero_packed => {
                if z.dtype != "I32" || z.shape.len() != 2 || z.shape[0] as usize != (rows * bits).div_ceil(32) || z.shape[1] as usize != groups {
                    return Err(format!("{}: expected an I32 zero-point tensor [{}, {groups}]", z.name, (rows * bits).div_ceil(32)));
                }
                // the bit stream of one group runs down the output rows (columns of the stored matrix)
                let words = z.shape[0] as usize;
                let zb = z.payload.as_slice();
                let mut v = vec![0i32; rows * groups];
                let mut column = Vec::with_capacity(words * 4);
                for g in 0..groups {
                    column.clear();
                    for wd in 0..words {
                        let at = (wd * groups + g) * 4;
                        column.extend_from_slice(&zb[at..at + 4]);
                    }
                    for r in 0..rows {
                        v[r * groups + g] = read_bits(&column, r * bits, bits) as i32 - (1 << (bits - 1));
                    }
                }
                v
            }
            Some(z) => {
                plain_ints(&z)?
            }
        };
        let g_idx = match p.remove("g_idx") {
            None => None,
            Some(t) if t.dtype == "I32" => Some(t.payload.as_slice().as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c) as u32).collect()),
            Some(t) => return Err(format!("{}: weight_g_idx must be I32", t.name)),
        };
        let d = CtAffine::new(w.payload.clone(), packed.then_some(bits), scales, zeros, g_idx, rows, cols, group_size)
            .map_err(|e| format!("{}: {e}", w.name))?;
        let mapped = map(&format!("{base}.weight"))?.ok_or_else(|| format!("packed tensor {:?} maps to a skipped tensor", w.name))?;
        out.push((mapped, crate::packed_quant::PackedData::Dyn(Arc::new(d)), vec![rows as u64, cols as u64], w.payload));
    }
    if out.is_empty() {
        return Err("quantization_config declares compressed-tensors integer weights but none were found".into());
    }
    Ok((dense, out))
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
    fn compressed_tensors_zero_points_match_the_library() {
        let cases: Value = serde_json::from_slice(&std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/quant/cases.json")).unwrap()).unwrap();
        let mut seen = 0;
        for (name, meta) in cases.as_object().unwrap() {
            if meta["format"] != "ct_asym" {
                continue;
            }
            let (tensors, expected) = fixture(name);
            let (bits, strategy) = (meta["bits"].as_u64().unwrap() as usize, meta["strategy"].as_str().unwrap());
            let group = if strategy == "group" { meta["group_size"].as_u64().unwrap() as usize } else { 0 };
            let (dense, packed) = split_ct_affine(tensors, bits, group, true, true, &identity).unwrap();
            assert!(dense.is_empty(), "{name}");
            assert_close(name, &decode_all(&packed[0]), &expected, 1e-6);
            seen += 1;
        }
        assert_eq!(seen, 4, "int4 and int8, group and channel");
    }

    #[test]
    fn activation_order_indexes_the_scales_by_the_group_of_each_column() {
        // 2 rows x 8 columns, 4 bits, two groups of 4; columns 0, 2, 4, 6 belong to group 0 and the others to group 1
        let codes: Vec<u8> = vec![8, 9, 10, 11, 12, 13, 14, 15, 7, 6, 5, 4, 3, 2, 1, 0]; // code - 8 = -8..7
        let mut bytes = Vec::new();
        for row in codes.chunks(8) {
            let word = row.iter().enumerate().fold(0u32, |w, (i, &c)| w | (u32::from(c) << (4 * i)));
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        let g_idx = vec![0u32, 1, 0, 1, 0, 1, 0, 1];
        let len = bytes.len();
        let d = CtAffine::new(Payload::Owned(Arc::new(bytes), 0..len), Some(4), vec![0.5, 2.0, 1.0, 4.0], vec![1, 0, 0, -1], Some(g_idx.clone()), 2, 8, 4).unwrap();
        let mut got = vec![0f32; 16];
        d.decode_chunk(0, &mut got).unwrap();
        for e in 0..16 {
            let (r, c) = (e / 8, e % 8);
            let q = i32::from(codes[e]) - 8;
            let g = g_idx[c] as usize;
            let (scale, zero) = ([0.5f32, 2.0, 1.0, 4.0][r * 2 + g], [1, 0, 0, -1][r * 2 + g]);
            assert_eq!(got[e], (q - zero) as f32 * scale, "element {e}");
        }
    }

    #[test]
    fn marlin_checkpoints_read_back_to_plain_values() {
        let cases: Value = serde_json::from_slice(&std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/quant/cases.json")).unwrap()).unwrap();
        let mut seen = 0;
        for (name, meta) in cases.as_object().unwrap() {
            if meta["format"] != "gptq_marlin" {
                continue;
            }
            let (tensors, expected) = fixture(name);
            let (bits, group) = (meta["bits"].as_u64().unwrap() as usize, meta["group_size"].as_i64().unwrap());
            let (dense, packed) = split_gptq_marlin(tensors, bits, group, &identity).unwrap();
            assert!(dense.is_empty(), "{name}");
            assert_close(name, &decode_all(&packed[0]), &expected, 1e-6);
            seen += 1;
        }
        assert_eq!(seen, 3, "4-bit grouped, 4-bit channel-wise, 8-bit grouped");
    }

    #[test]
    fn mlx_floating_point_modes_match_mlx() {
        let cases: Value = serde_json::from_slice(&std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/quant/cases.json")).unwrap()).unwrap();
        let mut seen = 0;
        for (name, meta) in cases.as_object().unwrap() {
            if meta["format"] != "mlx_fp" {
                continue;
            }
            let (tensors, expected) = fixture(name);
            let config = serde_json::json!({"mode": meta["mode"], "bits": meta["bits"], "group_size": meta["group_size"]});
            let (dense, packed) = split_mlx(tensors, &config, &identity).unwrap();
            assert!(dense.is_empty(), "{name}");
            assert_close(name, &decode_all(&packed[0]), &expected, 1e-6);
            seen += 1;
        }
        assert_eq!(seen, 3, "mxfp4, nvfp4 and mxfp8");
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