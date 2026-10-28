//! Independent Rust tensor operations. No GGML or inference-runtime calls.
use fllm_gguf_probe::{MappedGguf, TensorBytes};
use half::f16;
use rayon::prelude::*;

pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn load(model: &MappedGguf, name: &str) -> Result<Self, String> {
        let info = model
            .index()
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        let shape = info
            .dimensions
            .iter()
            .map(|&n| usize::try_from(n).map_err(|_| "tensor dimension overflow"))
            .collect::<Result<Vec<_>, _>>()?;
        let count = shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or("tensor size overflow")?;
        let data = dequantize(model.tensor_bytes(name)?, info.type_id, count)?;
        if data.iter().any(|x| !x.is_finite()) {
            return Err(format!("non-finite weights: {name}"));
        }
        Ok(Self { shape, data })
    }

    pub fn matrix(&self, rows: usize, cols: usize, name: &str) -> Result<(), String> {
        if self.shape != [cols, rows] {
            return Err(format!(
                "invalid dimensions for {name}: {:?}, expected [{cols}, {rows}]",
                self.shape
            ));
        }
        Ok(())
    }
}

fn read_half(bytes: &[u8]) -> f32 {
    f16::from_le_bytes([bytes[0], bytes[1]]).to_f32()
}
pub(crate) fn quant_layout(kind: u32) -> Option<(usize, usize)> {
    match kind {
        0 => None,
        1 | 30 => Some((1, 2)),
        2 => Some((32, 18)),
        3 => Some((32, 20)),
        6 => Some((32, 22)),
        7 => Some((32, 24)),
        8 => Some((32, 34)),
        12 => Some((256, 144)),
        13 => Some((256, 176)),
        14 => Some((256, 210)),
        0x8000_0010 | 0x8000_0014 => Some((32, 19)),
        0x8000_0011 | 0x8000_0015 => Some((64, 35)),
        0x8000_0012 | 0x8000_0016 => Some((128, 67)),
        0x8000_0020 => Some((1, 1)),
        k if fllm_gguf_probe::fp8b_type(k) => Some((1, 1)),
        _ => fllm_gguf_probe::mlx_affine(kind)
            .map(|m| (m.group as usize, m.block_bytes() as usize))
            .or_else(|| fllm_gguf_probe::quant_types::by_id(kind).map(|ty| (ty.block_elements, ty.bytes_per_block))),
    }
}

/// Q4Z blocks (see `QuantType::Q4Z32`) to f32.
fn dequantize_q4z(bytes: &[u8], group: usize, count: usize) -> Result<Vec<f32>, String> {
    let width = group / 2 + 3;
    if !count.is_multiple_of(group) || count / group * width != bytes.len() {
        return Err("invalid Q4Z tensor size".into());
    }
    let mut out = Vec::with_capacity(count);
    for b in bytes.chunks_exact(width) {
        let scale = read_half(&b[group / 2..]);
        let zero = (b[group / 2 + 2] & 15) as f32;
        for &q in &b[..group / 2] {
            out.push(((q & 15) as f32 - zero) * scale);
            out.push(((q >> 4) as f32 - zero) * scale);
        }
    }
    Ok(out)
}

fn dequantize_q4zp(bytes: &[u8], group: usize, count: usize) -> Result<Vec<f32>, String> {
    let width = group / 2 + 3;
    let q4z_len = (count / group) * width;
    if bytes.len() <= q4z_len {
        return Err("missing perm table in Q4ZP tensor".into());
    }
    let perm_bytes = &bytes[q4z_len..];
    let cols = perm_bytes.len() / 4;
    if cols == 0 || !count.is_multiple_of(cols) || perm_bytes.len() != cols * 4 {
        return Err("invalid perm table length in Q4ZP tensor".into());
    }
    let rows = count / cols;
    let perm: Vec<usize> = (0..cols)
        .map(|c| u32::from_le_bytes(perm_bytes[c * 4..(c + 1) * 4].try_into().unwrap()) as usize)
        .collect();
    let permuted = dequantize_q4z(&bytes[..q4z_len], group, count)?;
    let mut out = vec![0.0f32; count];
    for r in 0..rows {
        let r_perm = &permuted[r * cols..(r + 1) * cols];
        let r_out = &mut out[r * cols..(r + 1) * cols];
        for c in 0..cols {
            let orig_k = perm[c];
            if orig_k >= cols {
                return Err("perm index out of bounds".into());
            }
            r_out[orig_k] = r_perm[c];
        }
    }
    Ok(out)
}

pub(crate) fn e4m3fn_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exp = (bits >> 3) & 0x0f;
    let mant = bits & 0x07;
    if exp == 0 {
        sign * (mant as f32) * 2f32.powi(-9)
    } else if exp == 0x0f && mant == 0x07 {
        f32::NAN
    } else {
        sign * (1.0 + mant as f32 / 8.0) * 2f32.powi(exp as i32 - 7)
    }
}

fn dequantize_fp8m(bytes: &[u8], count: usize) -> Result<Vec<f32>, String> {
    if bytes.len() <= count {
        return Err("missing scales in FP8M tensor".into());
    }
    let scale_bytes = bytes.len() - count;
    if !scale_bytes.is_multiple_of(2) {
        return Err("invalid scale byte length in FP8M tensor".into());
    }
    let rows = scale_bytes / 2;
    if rows == 0 || !count.is_multiple_of(rows) {
        return Err("invalid dimensions in FP8M tensor".into());
    }
    let cols = count / rows;
    let weight_bytes = &bytes[..count];
    let scales_raw = &bytes[count..];
    let mut out = Vec::with_capacity(count);
    for r in 0..rows {
        let s_bytes = [scales_raw[r * 2], scales_raw[r * 2 + 1]];
        let scale = f16::from_le_bytes(s_bytes).to_f32();
        let row_w = &weight_bytes[r * cols..(r + 1) * cols];
        for &b in row_w {
            out.push(e4m3fn_to_f32(b) * scale);
        }
    }
    Ok(out)
}

/// `FP8B` (block-wise FP8) to f32: `e4m3(code) * scale of the tile`; the geometry comes from the trailer of the tensor.
fn dequantize_fp8b(bytes: &[u8], kind: u32, count: usize) -> Result<Vec<f32>, String> {
    let layout = fllm_gguf_probe::Fp8bLayout::parse(kind, bytes)?;
    if layout.elements() != count {
        return Err(format!("FP8B tensor holds {} values, {count} expected", layout.elements()));
    }
    let mut out = Vec::with_capacity(count);
    for slab in 0..layout.slabs {
        for row in 0..layout.rows {
            let base = (slab * layout.rows + row) * layout.cols;
            for (tile, codes) in bytes[base..base + layout.cols].chunks(layout.block_cols).enumerate() {
                let scale = layout.scale(bytes, layout.tile_index(slab, row, tile * layout.block_cols));
                out.extend(codes.iter().map(|&b| e4m3fn_to_f32(b) * scale));
            }
        }
    }
    Ok(out)
}

pub fn dequantize(bytes: &[u8], kind: u32, count: usize) -> Result<Vec<f32>, String> {
    if fllm_gguf_probe::exl::is_exl_type(kind) {
        // native EXL2 / EXL3 blobs hold a whole matrix: this is the expand-at-upload fallback of backends without a kernel
        return fllm_gguf_probe::exl::dequantize(bytes, kind, count);
    }
    if fllm_gguf_probe::fp8m_type(kind) {
        return dequantize_fp8m(bytes, count);
    }
    if fllm_gguf_probe::fp8b_type(kind) {
        return dequantize_fp8b(bytes, kind, count);
    }
    if let Some(group) = fllm_gguf_probe::q4zp_group(kind) {
        return dequantize_q4zp(bytes, group as usize, count);
    }
    if let Some(group) = fllm_gguf_probe::q4z_group(kind) {
        return dequantize_q4z(bytes, group as usize, count);
    }
    if crate::cpu_mlx::handles(kind) {
        return crate::cpu_mlx::dequantize(bytes, kind, count);
    }
    if matches!(kind, 16..=23 | 29) {
        return crate::quant_iq::dequantize_iq(bytes, kind, count);
    }
    if matches!(kind, 9..=11 | 15 | 24..=28 | 34 | 35 | 39..=42) {
        return crate::quant_extra::dequantize_extra(bytes, kind, count);
    }
    let (block, width) = match kind {
        0 => (1, 4),
        1 | 30 => (1, 2),
        _ => quant_layout(kind).ok_or_else(|| {
            format!("independent Rust tensor reader does not support GGUF type {kind}")
        })?,
    };
    if !count.is_multiple_of(block) || (count / block).checked_mul(width) != Some(bytes.len()) {
        return Err("invalid quantized tensor size".into());
    }
    let mut out = Vec::with_capacity(count);
    for b in bytes.chunks_exact(width) {
        match kind {
            0 => out.push(f32::from_le_bytes(b.try_into().unwrap())),
            1 => out.push(read_half(b)),
            30 => out.push(f32::from_bits(
                (u16::from_le_bytes(b.try_into().unwrap()) as u32) << 16,
            )),
            8 => {
                let d = read_half(b);
                out.extend(b[2..].iter().map(|&q| d * (q as i8) as f32));
            }
            2 | 3 | 6 | 7 => {
                out.extend((0..32).map(|j| classic_value(b, kind, j)));
            }
            12 | 13 => {
                let d = read_half(b);
                let min = read_half(&b[2..]);
                let scales = &b[4..16];
                let qs = &b[if kind == 13 { 48 } else { 16 }..];
                for group in 0..8 {
                    let (s, m) = if group < 4 {
                        (scales[group] & 63, scales[group + 4] & 63)
                    } else {
                        (
                            (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4),
                            (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4),
                        )
                    };
                    let chunk = &qs[group / 2 * 32..group / 2 * 32 + 32];
                    out.extend(chunk.iter().enumerate().map(|(j, &q)| {
                        let mut q = if group % 2 == 0 { q & 15 } else { q >> 4 };
                        if kind == 13 {
                            q |= ((b[16 + j] >> group) & 1) << 4;
                        }
                        d * s as f32 * q as f32 - min * m as f32
                    }));
                }
            }
            14 => {
                let d = read_half(&b[208..]);
                let mut block = [0.0; 256];
                for n in 0..2 {
                    let ql = &b[n * 64..];
                    let qh = &b[128 + n * 32..];
                    let scales = &b[192 + n * 8..];
                    for j in 0..32 {
                        let scale = j / 16;
                        let values = [
                            (ql[j] & 15) | ((qh[j] & 3) << 4),
                            (ql[j + 32] & 15) | (((qh[j] >> 2) & 3) << 4),
                            (ql[j] >> 4) | (((qh[j] >> 4) & 3) << 4),
                            (ql[j + 32] >> 4) | (((qh[j] >> 6) & 3) << 4),
                        ];
                        for group in 0..4 {
                            block[n * 128 + group * 32 + j] = d
                                * (scales[scale + group * 2] as i8) as f32
                                * (values[group] as f32 - 32.0);
                        }
                    }
                }
                out.extend_from_slice(&block);
            }
            _ => unreachable!(),
        }
    }
    Ok(out)
}

pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Vec<f32>>,
    pub rows: usize,
    pub cols: usize,
    packed: Option<(u32, TensorBytes)>,
    /// Per-tensor output factor of NVFP4 weights (1.0 for every other format).
    pub scale: f32,
    /// Static activation scale of an NVFP4 weight (ModelOpt `input_scale`, the F32 tensor `x.input_scale` of the GGUF); 0 = the checkpoint has none.
    pub input_scale: f32,
    fast_cpu: bool,
    #[cfg(feature = "rust-gpu")]
    gpu: Option<Vec<crate::gpu::GpuLinear>>,
}

/// Retained scratch for sibling Q8 projections within one decoder invocation.
/// The buffers are reused across Q/K/V and gate/up groups and across layers.
#[derive(Default)]
pub(crate) struct ProjectionGroupWorkspace {
    q8_input: crate::cpu_kernels::Q8Input,
    q8_output: Vec<Vec<f32>>,
    /// Buffers of the fused MoE expert pass (`Linear::forward_experts`).
    experts: crate::cpu_quantized::ExpertScratch,
}

/// The value of a per-tensor scale for the rows `first..` of a tensor of `total_rows` rows: the one value of a plain tensor, or, for `n`
/// values, entry `first / (total_rows / n)` (one value per expert in a stacked expert tensor; a fused gate/up tensor has the two halves of
/// an expert in one slice).
fn scale_for_rows(values: &[f32], first: usize, total_rows: usize) -> Option<f32> {
    match values {
        [v] => Some(*v),
        v if !v.is_empty() && total_rows.is_multiple_of(v.len()) => v.get(first / (total_rows / v.len())).copied(),
        _ => None,
    }
}

/// The per-tensor scale of an NVFP4 weight `x.weight` is stored as the F32 tensor `x.scale` (llama.cpp convention): one value, or one per
/// expert for stacked experts (`first` and `total_rows` pick the expert, see `scale_for_rows`).
fn nvfp4_tensor_scale(model: &MappedGguf, name: &str, first: usize, total_rows: usize) -> Result<f32, String> {
    let scale_name = match name.strip_suffix(".weight") {
        Some(stem) => format!("{stem}.scale"),
        None => return Ok(1.0),
    };
    if !model.index().tensors.iter().any(|t| t.name == scale_name) {
        return Ok(1.0);
    }
    let t = Tensor::load(model, &scale_name)?;
    match scale_for_rows(&t.data, first, total_rows) {
        Some(v) if v.is_finite() => Ok(v),
        _ => Err(format!("{scale_name} must hold one finite value, or one per expert for stacked experts")),
    }
}

/// The static activation scale of an NVFP4 weight `x.weight`: the F32 tensor `x.input_scale` (written by llama.cpp's converter for ModelOpt
/// checkpoints unless it is 1; for stacked experts one value per expert, picked like `nvfp4_tensor_scale`). 0 when the tensor is absent or
/// does not hold a positive finite value for these rows; the GPU kernels then quantize activations without a per-tensor factor.
fn nvfp4_input_scale(model: &MappedGguf, name: &str, first: usize, total_rows: usize) -> f32 {
    let Some(stem) = name.strip_suffix(".weight") else { return 0.0 };
    let scale_name = format!("{stem}.input_scale");
    if !model.index().tensors.iter().any(|t| t.name == scale_name) {
        return 0.0;
    }
    match Tensor::load(model, &scale_name) {
        Ok(t) => match scale_for_rows(&t.data, first, total_rows) {
            Some(v) if v.is_finite() && v > 0.0 => v,
            _ => 0.0,
        },
        Err(_) => 0.0,
    }
}

/// Rows `first..first + rows` (counted over all rows of all matrices) of the `FP8B` tensor `raw` of GGUF dimensions `dims`, as a tensor of
/// its own. The whole tensor of a single matrix is a view; a part is copied (codes and the scales of its tile rows are not contiguous).
/// The part has to lie in one matrix and to start at a tile row.
pub(crate) fn fp8b_rows(raw: &TensorBytes, kind: u32, dims: &[u64], first: usize, rows: usize, name: &str) -> Result<TensorBytes, String> {
    let all = fllm_gguf_probe::Fp8bLayout::from_dims(kind, dims).ok_or_else(|| format!("invalid FP8B dimensions of {name}: {dims:?}"))?;
    if fllm_gguf_probe::Fp8bLayout::parse(kind, raw).map_err(|e| format!("{name}: {e}"))? != all {
        return Err(format!("{name}: the FP8B trailer does not match the tensor dimensions {dims:?}"));
    }
    if first == 0 && rows == all.rows && all.slabs == 1 {
        return raw.slice(0, raw.len());
    }
    let (slab, within) = (first / all.rows, first % all.rows);
    if within + rows > all.rows {
        return Err(format!("{name}: rows {first}..{} cross the boundary of a {}-row matrix", first + rows, all.rows));
    }
    if !within.is_multiple_of(all.block_rows) {
        return Err(format!(
            "{name}: block-wise FP8 rows {first}.. do not start at a {}-row tile (a fused tensor cut inside a tile cannot keep its scales)",
            all.block_rows
        ));
    }
    let part = fllm_gguf_probe::Fp8bLayout { rows, slabs: 1, ..all };
    let mut buf = Vec::with_capacity(part.total_len());
    buf.extend_from_slice(&raw[first * all.cols..(first + rows) * all.cols]);
    let first_tile = slab * all.tiles_per_slab() + within / all.block_rows * all.tile_cols();
    let tiles = rows.div_ceil(all.block_rows) * all.tile_cols();
    buf.extend_from_slice(&raw[all.scale_offset() + first_tile * 4..all.scale_offset() + (first_tile + tiles) * 4]);
    buf.extend_from_slice(&part.trailer());
    debug_assert_eq!(buf.len(), part.total_len());
    Ok(TensorBytes::from_vec(buf))
}

impl Linear {
    pub fn packed_weights(&self) -> Option<(u32, &[u8])> {
        self.packed
            .as_ref()
            .map(|(kind, bytes)| (*kind, bytes.as_ref()))
    }

    /// The packed storage itself (shared, so that a row range of it can be viewed without a copy).
    pub fn packed_storage(&self) -> Option<(u32, &TensorBytes)> {
        self.packed.as_ref().map(|(kind, bytes)| (*kind, bytes))
    }

    /// A projection over packed bytes of GGUF type `kind` (`rows x cols`, the layout `Linear::load` expects, extras included).
    /// Used for the shards of a tensor-parallel layer.
    pub fn from_packed(
        kind: u32,
        bytes: TensorBytes,
        rows: usize,
        cols: usize,
        bias: Option<Vec<f32>>,
        scale: f32,
        input_scale: f32,
    ) -> Result<Self, String> {
        if rows == 0 || cols == 0 || rows.checked_mul(cols).is_none() {
            return Err("invalid projection dimensions".into());
        }
        if bias.as_ref().is_some_and(|b| b.len() != rows) {
            return Err("invalid projection bias".into());
        }
        Ok(Self {
            weight: Tensor { shape: vec![cols, rows], data: Vec::new() },
            bias,
            rows,
            cols,
            packed: Some((kind, bytes)),
            scale,
            input_scale,
            fast_cpu: false,
            #[cfg(feature = "rust-gpu")]
            gpu: None,
        })
    }

    pub fn from_parts(
        weight: Tensor,
        rows: usize,
        cols: usize,
        bias: Option<Vec<f32>>,
    ) -> Result<Self, String> {
        if rows == 0 || cols == 0 || rows.checked_mul(cols).is_none() {
            return Err("invalid projection dimensions".into());
        }
        weight.matrix(rows, cols, "projection")?;
        if bias.as_ref().is_some_and(|b| b.len() != rows) {
            return Err("invalid projection bias".into());
        }
        Ok(Self {
            weight,
            bias,
            rows,
            cols,
            packed: None,
            scale: 1.0,
            input_scale: 0.0,
            fast_cpu: false,
            #[cfg(feature = "rust-gpu")]
            gpu: None,
        })
    }
    pub fn load(model: &MappedGguf, base: &str, rows: usize, cols: usize) -> Result<Self, String> {
        let count = rows
            .checked_mul(cols)
            .filter(|_| rows > 0 && cols > 0)
            .ok_or("invalid projection dimensions")?;
        let name = format!("{base}.weight");
        let info = model
            .index()
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        let packed = if let Some((block, bytes)) = quant_layout(info.type_id) {
            let raw = model.tensor_bytes(&name)?;
            let is_q4zp = fllm_gguf_probe::q4zp_group(info.type_id).is_some();
            let perm_bytes = if is_q4zp { cols.checked_mul(4).ok_or("cols overflow")? } else { 0 };
            let extra_bytes = if fllm_gguf_probe::fp8m_type(info.type_id) {
                rows.checked_mul(2).ok_or("rows overflow")?
            } else if fllm_gguf_probe::fp8b_type(info.type_id) {
                // scales and trailer (the geometry is checked against the trailer below)
                fllm_gguf_probe::Fp8bLayout::new(info.type_id, cols, rows, 1).map_or(0, |l| l.total_len() - count)
            } else {
                perm_bytes
            };
            let expected_bytes = (count / block).checked_mul(bytes).and_then(|n| n.checked_add(extra_bytes));
            if !cols.is_multiple_of(block) || expected_bytes != Some(raw.len())
            {
                return Err(format!("invalid packed projection {name}"));
            }
            Some((info.type_id, model.owned_tensor_bytes(&name)?))
        } else if fllm_gguf_probe::exl::is_exl_type(info.type_id) {
            // native EXL2 / EXL3 blob: self-describing, validated once here
            crate::cpu_exl::validate(model.tensor_bytes(&name)?, info.type_id, rows, cols).map_err(|e| format!("invalid packed projection {name}: {e}"))?;
            Some((info.type_id, model.owned_tensor_bytes(&name)?))
        } else {
            None
        };
        let weight = if packed.is_some() {
            Tensor {
                shape: info.dimensions.iter().map(|&n| n as usize).collect(),
                data: Vec::new(),
            }
        } else {
            Tensor::load(model, &name)?
        };
        weight.matrix(rows, cols, base)?;
        let bias_name = format!("{base}.bias");
        let bias = if model.index().tensors.iter().any(|t| t.name == bias_name) {
            let tensor = Tensor::load(model, &bias_name)?;
            if tensor.shape != [rows] {
                return Err(format!("invalid bias shape for {base}"));
            }
            Some(tensor.data)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            rows,
            cols,
            scale: match &packed {
                Some((40, _)) => nvfp4_tensor_scale(model, &name, 0, rows)?,
                _ => 1.0,
            },
            input_scale: match &packed {
                Some((40, _)) => nvfp4_input_scale(model, &name, 0, rows),
                _ => 0.0,
            },
            packed,
            fast_cpu: false,
            #[cfg(feature = "rust-gpu")]
            gpu: None,
        })
    }
    /// A projection made of `rows` rows starting at `first` of a fused or stacked tensor: the K part
    /// of a fused `attn_qkv`, the gate half of a fused `ffn_up`, or one expert of `ffn_up_exps`.
    /// Quantized and half-precision weights stay memory-mapped; no bytes are copied.
    pub fn load_rows(
        model: &MappedGguf,
        name: &str,
        bias: Option<&str>,
        first: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Self, String> {
        let info = model
            .index()
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        if info.dimensions.first().copied() != Some(cols as u64) {
            return Err(format!("invalid width for {name}: {:?}", info.dimensions));
        }
        let total_rows = info.dimensions[1..]
            .iter()
            .try_fold(1usize, |n, &d| usize::try_from(d).ok()?.checked_mul(n))
            .ok_or("tensor size overflow")?;
        if rows == 0 || cols == 0 || first.checked_add(rows).is_none_or(|end| end > total_rows) {
            return Err(format!(
                "rows {first}..{} are outside {name} ({total_rows} rows)",
                first.saturating_add(rows)
            ));
        }
        let bias = match bias {
            Some(bias_name) if model.index().tensors.iter().any(|t| t.name == bias_name) => {
                let all = Tensor::load(model, bias_name)?.data;
                if all.len() < first + rows {
                    return Err(format!("bias {bias_name} is shorter than its projection"));
                }
                Some(all[first..first + rows].to_vec())
            }
            _ => None,
        };
        if fllm_gguf_probe::exl::is_exl_type(info.type_id) {
            // native EXL2 rows are independent: a slice is a smaller blob; EXL3 mixes the outputs of 128-blocks and is only used whole
            let whole = first == 0 && rows == total_rows;
            let raw = model.tensor_bytes(name)?;
            let part = if whole {
                model.owned_tensor_bytes(name)?
            } else if info.type_id == fllm_gguf_probe::exl::EXL2_TYPE {
                TensorBytes::from_vec(fllm_gguf_probe::exl::Exl2View::parse(raw)?.slice_rows(first, rows)?)
            } else {
                return Err(format!("{name}: an EXL3 tensor cannot be split into rows"));
            };
            crate::cpu_exl::validate(&part, info.type_id, rows, cols).map_err(|e| format!("invalid packed projection {name}: {e}"))?;
            return Ok(Self {
                weight: Tensor { shape: vec![cols, rows], data: Vec::new() },
                bias,
                rows,
                cols,
                packed: Some((info.type_id, part)),
                scale: 1.0,
                input_scale: 0.0,
                fast_cpu: false,
                #[cfg(feature = "rust-gpu")]
                gpu: None,
            });
        }
        if let Some((block, bytes)) = quant_layout(info.type_id) {
            if !cols.is_multiple_of(block) {
                return Err(format!("{name} width is not a multiple of its block size"));
            }
            let row_bytes = cols / block * bytes;
            let raw = model.owned_tensor_bytes(name)?;
            let is_q4zp = fllm_gguf_probe::q4zp_group(info.type_id).is_some();
            let is_fp8m = fllm_gguf_probe::fp8m_type(info.type_id);
            let is_fp8b = fllm_gguf_probe::fp8b_type(info.type_id);
            let perm_bytes = if is_q4zp { cols.checked_mul(4).ok_or("cols overflow")? } else { 0 };
            let extra_bytes = if is_fp8m {
                total_rows.checked_mul(2).ok_or("total_rows overflow")?
            } else if is_fp8b {
                // scales and trailer; `fp8b_rows` checks the geometry against the trailer
                raw.len().saturating_sub(row_bytes * total_rows)
            } else {
                perm_bytes
            };
            if raw.len() != row_bytes * total_rows + extra_bytes {
                return Err(format!("invalid packed tensor {name}"));
            }
            let part = if is_q4zp && (first > 0 || rows < total_rows) {
                let mut buf = Vec::with_capacity(rows * row_bytes + perm_bytes);
                buf.extend_from_slice(&raw[first * row_bytes..(first + rows) * row_bytes]);
                buf.extend_from_slice(&raw[total_rows * row_bytes..total_rows * row_bytes + perm_bytes]);
                TensorBytes::from_vec(buf)
            } else if is_fp8m && (first > 0 || rows < total_rows) {
                let mut buf = Vec::with_capacity(rows * row_bytes + rows * 2);
                buf.extend_from_slice(&raw[first * row_bytes..(first + rows) * row_bytes]);
                buf.extend_from_slice(&raw[total_rows * row_bytes + first * 2..total_rows * row_bytes + (first + rows) * 2]);
                TensorBytes::from_vec(buf)
            } else if is_fp8b {
                fp8b_rows(&raw, info.type_id, &info.dimensions, first, rows, name)?
            } else {
                raw.slice(first * row_bytes, rows * row_bytes + extra_bytes)?
            };
            Ok(Self {
                weight: Tensor {
                    shape: vec![cols, rows],
                    data: Vec::new(),
                },
                bias,
                rows,
                cols,
                packed: Some((info.type_id, part)),
                scale: if info.type_id == 40 { nvfp4_tensor_scale(model, name, first, total_rows)? } else { 1.0 },
                input_scale: if info.type_id == 40 { nvfp4_input_scale(model, name, first, total_rows) } else { 0.0 },
                fast_cpu: false,
                #[cfg(feature = "rust-gpu")]
                gpu: None,
            })
        } else {
            let bytes = model.tensor_bytes(name)?;
            let row_bytes = cols * 4;
            let part = bytes
                .get(first * row_bytes..(first + rows) * row_bytes)
                .ok_or_else(|| format!("invalid F32 tensor {name}"))?;
            let data: Vec<f32> = part
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            if data.iter().any(|x| !x.is_finite()) {
                return Err(format!("non-finite weights: {name}"));
            }
            Self::from_parts(
                Tensor {
                    shape: vec![cols, rows],
                    data,
                },
                rows,
                cols,
                bias,
            )
        }
    }
    #[cfg(feature = "rust-gpu")]
    pub fn enable_gpu(
        &mut self,
        context: &std::sync::Arc<crate::gpu::GpuContext>,
    ) -> Result<(), String> {
        self.gpu = Some(if let Some((kind, bytes)) = &self.packed {
            if !matches!(kind, 2 | 3 | 6 | 7 | 8 | 12 | 13 | 14) {
                context.upload_chunks(
                    &dequantize(bytes, *kind, self.rows * self.cols)?,
                    self.rows,
                    self.cols,
                )?
            } else {
                context.upload_quantized_chunks(bytes, *kind, self.rows, self.cols)?
            }
        } else {
            context.upload_chunks(&self.weight.data, self.rows, self.cols)?
        });
        Ok(())
    }
    /// Prepare one shared input for sibling projections (Q/K/V or gate/up): Q8_0, the packed
    /// integer formats (K formats may mix) and float formats each quantize or round the input
    /// once. Other mixes, adapted weights and GPU projections use their usual path.
    pub(crate) fn forward_group_with_workspace<'a>(
        projections: &[&Self],
        input: &[f32],
        workspace: &'a mut ProjectionGroupWorkspace,
    ) -> Result<&'a mut [Vec<f32>], String> {
        let Some(first) = projections.first() else {
            workspace.q8_output.clear();
            return Ok(&mut workspace.q8_output);
        };
        #[cfg(feature = "npu-openvino")]
        if first.npu_host_wants(input) {
            workspace.q8_output = projections.iter().map(|p| p.forward(input)).collect::<Result<_, _>>()?;
            return Ok(&mut workspace.q8_output);
        }
        let cols = first.cols;
        let reusable_q8 = cols > 0
            && crate::cpu_kernels::packed_simd()
            && projections.iter().all(|p| {
                #[cfg(feature = "rust-gpu")]
                if p.gpu.is_some() {
                    return false;
                }
                p.fast_cpu && p.cols == cols && matches!(p.packed, Some((8, _)))
            });
        if reusable_q8 {
            if !input.len().is_multiple_of(cols) {
                return Err("invalid shared projection input".into());
            }
            workspace.q8_input.prepare(input, cols);
            let members = projections
                .iter()
                .map(|p| {
                    let (_, weights) = p.packed.as_ref().unwrap();
                    crate::cpu_kernels::Q8Projection {
                        weights,
                        rows: p.rows,
                        bias: p.bias.as_deref(),
                    }
                })
                .collect::<Vec<_>>();
            workspace.q8_output.resize_with(projections.len(), Vec::new);
            crate::cpu_kernels::q8_linear_group_into(
                &members,
                &workspace.q8_input,
                &mut workspace.q8_output,
            )
            .ok_or_else(|| "shared Q8 projection unavailable".to_owned())?;
        } else {
            workspace.q8_output = Self::forward_group(projections, input)?;
        }
        Ok(&mut workspace.q8_output)
    }

    /// The packed K-format weights of a projection that may join the fused expert pass:
    /// plain CPU weights with the standard output scale and no adapter or GPU copy.
    fn expert_member(&self) -> Option<(crate::cpu_quantized::Projection<'_>, u32)> {
        #[cfg(feature = "rust-gpu")]
        if self.gpu.is_some() {
            return None;
        }
        if !self.fast_cpu || self.scale != 1.0 || !crate::cpu_kernels::packed_simd() {
            return None;
        }
        let (kind, weights) = self.packed.as_ref()?;
        Some((
            crate::cpu_quantized::Projection {
                weights,
                rows: self.rows,
                bias: self.bias.as_deref(),
            },
            *kind,
        ))
    }

    /// The gated FFNs of every expert that received rows of a prompt batch (`input` rows per
    /// expert) in two joined work queues over all experts. `None` (nothing computed) when an
    /// expert does not qualify; see `cpu_quantized::linear_experts_prompt`.
    pub(crate) fn forward_experts_prompt(
        experts: &[(&Self, &Self, &Self, &[f32])],
        activate: &(dyn Fn(&mut [f32], &[f32]) + Sync),
    ) -> Option<Vec<Vec<f32>>> {
        let cols = experts.first()?.0.cols;
        let rows = experts
            .iter()
            .map(|(gate, up, down, input)| {
                let (gate_projection, gate_kind) = gate.expert_member()?;
                let (up_projection, up_kind) = up.expert_member()?;
                let (down_projection, down_kind) = down.expert_member()?;
                (gate.cols == cols
                    && up.cols == cols
                    && up.rows == gate.rows
                    && down.cols == gate.rows)
                    .then_some(crate::cpu_quantized::ExpertRows {
                        experts: crate::cpu_quantized::ExpertProjections {
                            gate: gate_projection,
                            up: up_projection,
                            down: down_projection,
                            kinds: [gate_kind, up_kind, down_kind],
                        },
                        input,
                    })
            })
            .collect::<Option<Vec<_>>>()?;
        crate::cpu_quantized::linear_experts_prompt(&rows, cols, activate)
    }

    /// The gated FFNs `down(act(gate(x), up(x)))` of one token's active experts in two joined
    /// work queues (all gate/up rows, then all down rows) instead of a fork/join per
    /// projection. The results equal those of running the experts one by one bit for bit.
    ///
    /// `None` when any member does not qualify (not K-format packed, GPU or adapted weights,
    /// scaled output, inconsistent shapes, no kernel on this CPU); the caller then uses
    /// the per-expert path. `input` is one token.
    pub(crate) fn forward_experts<'w>(
        experts: &[(&Self, &Self, &Self)],
        input: &[f32],
        activate: &dyn Fn(&mut [f32], &[f32]),
        workspace: &'w mut ProjectionGroupWorkspace,
    ) -> Option<&'w [Vec<f32>]> {
        let cols = input.len();
        let members = experts
            .iter()
            .map(|(gate, up, down)| {
                let (gate_projection, gate_kind) = gate.expert_member()?;
                let (up_projection, up_kind) = up.expert_member()?;
                let (down_projection, down_kind) = down.expert_member()?;
                (gate.cols == cols
                    && up.cols == cols
                    && up.rows == gate.rows
                    && down.cols == gate.rows)
                    .then_some(crate::cpu_quantized::ExpertProjections {
                        gate: gate_projection,
                        up: up_projection,
                        down: down_projection,
                        kinds: [gate_kind, up_kind, down_kind],
                    })
            })
            .collect::<Option<Vec<_>>>()?;
        crate::cpu_quantized::linear_experts(
            &members,
            input,
            cols,
            activate,
            &mut workspace.experts,
        )
    }

    /// `FLLM_NPU_HOST=1` (feature `npu-openvino`): the host decoder hands the large prompt projections to the NPU engine. True when a
    /// prompt of this many rows is big enough for it; the sibling projections of a group are then computed one by one.
    #[cfg(feature = "npu-openvino")]
    fn npu_host_wants(&self, input: &[f32]) -> bool {
        self.cols != 0
            && self.packed.is_some()
            && crate::backend::npu::openvino::host_engine().is_some_and(|e| input.len() / self.cols >= e.plan.min_batch)
    }

    /// The first rows of `input` computed by the NPU engine (`FLLM_NPU_HOST=1`): `(rows done, values without bias)`.
    #[cfg(feature = "npu-openvino")]
    fn npu_rows(&self, input: &[f32]) -> Option<(usize, Vec<f32>)> {
        if self.scale != 1.0 || !self.npu_host_wants(input) {
            return None;
        }
        let engine = crate::backend::npu::openvino::host_engine()?;
        let (kind, packed) = self.packed.as_ref()?;
        let quant = crate::backend::QuantType::from_gguf_type_id(*kind);
        let batch = input.len() / self.cols;
        let done = engine.offload_host(quant, packed, self.rows, self.cols, input, batch)?;
        (done.rows_done > 0).then_some((done.rows_done, done.values))
    }

    pub fn forward_group(projections: &[&Self], input: &[f32]) -> Result<Vec<Vec<f32>>, String> {
        let Some(first) = projections.first() else {
            return Ok(Vec::new());
        };
        #[cfg(feature = "npu-openvino")]
        if first.npu_host_wants(input) {
            return projections.iter().map(|p| p.forward(input)).collect();
        }
        let cols = first.cols;
        let reusable = crate::cpu_kernels::packed_simd()
            && projections.iter().all(|p| {
                #[cfg(feature = "rust-gpu")]
                if p.gpu.is_some() {
                    return false;
                }
                p.fast_cpu && p.cols == cols && matches!(p.packed, Some((8, _)))
            });
        if reusable {
            if !input.len().is_multiple_of(cols) {
                return Err("invalid shared projection input".into());
            }
            let packed_input = crate::cpu_kernels::Q8Input::new(input, cols);
            let members = projections
                .iter()
                .map(|p| {
                    let (_, weights) = p.packed.as_ref().unwrap();
                    crate::cpu_kernels::Q8Projection {
                        weights,
                        rows: p.rows,
                        bias: p.bias.as_deref(),
                    }
                })
                .collect::<Vec<_>>();
            crate::cpu_kernels::q8_linear_group(&members, &packed_input)
                .ok_or_else(|| "shared Q8 projection unavailable".into())
        } else {
            // Every packed format with an integer kernel shares one activation quantization
            // across its siblings; the K formats (e.g. the Q4_K and Q6_K members of a Q4_K_M
            // Q/K/V group) share it even when their weight formats differ.
            let kinds = projections
                .iter()
                .map(|p| {
                    #[cfg(feature = "rust-gpu")]
                    if p.gpu.is_some() {
                        return None;
                    }
                    if !p.fast_cpu || p.cols != cols {
                        return None;
                    }
                    p.packed.as_ref().map(|(kind, _)| *kind)
                })
                .collect::<Option<Vec<_>>>();
            if let Some(kinds) = kinds.filter(|kinds| {
                crate::cpu_kernels::packed_simd() && crate::cpu_quantized::groupable(kinds)
            }) {
                if !input.len().is_multiple_of(cols) {
                    return Err("invalid shared projection input".into());
                }
                let members = projections
                    .iter()
                    .map(|p| {
                        let (_, weights) = p.packed.as_ref().unwrap();
                        crate::cpu_quantized::Projection {
                            weights,
                            rows: p.rows,
                            bias: p.bias.as_deref(),
                        }
                    })
                    .collect::<Vec<_>>();
                if let Some(output) =
                    crate::cpu_quantized::linear_mixed(&members, &kinds, input, cols)
                {
                    return Ok(output);
                }
            }
            let float_kind = match &first.packed {
                None => Some(0),
                Some((kind @ (1 | 30), _)) => Some(*kind),
                _ => None,
            };
            if let Some(kind) = float_kind {
                let reusable = projections.iter().all(|p| {
                    #[cfg(feature = "rust-gpu")]
                    if p.gpu.is_some() {
                        return false;
                    }
                    p.fast_cpu
                        && p.cols == cols
                        && match &p.packed {
                            None => kind == 0,
                            Some((packed_kind, _)) => *packed_kind == kind,
                        }
                });
                if reusable {
                    if !input.len().is_multiple_of(cols) {
                        return Err("invalid shared projection input".into());
                    }
                    let members = projections
                        .iter()
                        .map(|p| {
                            let weights = if let Some((_, packed)) = &p.packed {
                                packed.as_ref()
                            } else {
                                // Immutable bytes of the borrowed F32 tensor; the tensor
                                // outlives the joined work queue and is never resized here.
                                unsafe {
                                    std::slice::from_raw_parts(
                                        p.weight.data.as_ptr().cast::<u8>(),
                                        p.weight.data.len() * 4,
                                    )
                                }
                            };
                            crate::cpu_float::Projection {
                                weights,
                                rows: p.rows,
                                bias: p.bias.as_deref(),
                            }
                        })
                        .collect::<Vec<_>>();
                    if let Some(output) =
                        crate::cpu_float::linear_group(&members, kind, input, cols)
                    {
                        return Ok(output);
                    }
                }
            }
            projections.iter().map(|p| p.forward(input)).collect()
        }
    }
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        #[cfg(feature = "rust-gpu")]
        if let Some(gpu) = &self.gpu {
            let batch = input.len() / self.cols;
            let mut out = vec![0.0; batch * self.rows];
            let mut offset = 0;
            for chunk in gpu {
                let values = chunk.forward(input)?;
                let width = values.len() / batch;
                for b in 0..batch {
                    out[b * self.rows + offset..b * self.rows + offset + width]
                        .copy_from_slice(&values[b * width..(b + 1) * width]);
                }
                offset += width;
            }
            if let Some(bias) = &self.bias {
                for row in out.chunks_exact_mut(self.rows) {
                    add(row, bias);
                }
            }
            return Ok(out);
        }
        if self.cols == 0 || !input.len().is_multiple_of(self.cols) {
            return Err("projection input dimensions do not match".into());
        }
        if input.is_empty() {
            return Ok(Vec::new());
        }
        if let Some((kind, packed)) = &self.packed {
            if fllm_gguf_probe::fp8b_type(*kind) {
                // block-wise FP8 stays 8-bit in memory: the tile scales are applied while the codes are decoded
                return crate::cpu_fp8::linear(*kind, packed, input, self.rows, self.cols, self.bias.as_deref());
            }
        }
        #[cfg(feature = "npu-openvino")]
        if let Some((done, mut out)) = self.npu_rows(input) {
            if let Some(bias) = &self.bias {
                for row in out.chunks_exact_mut(self.rows) {
                    add(row, bias);
                }
            }
            // the rows the NPU did not take (fewer than a full block) run on the CPU kernels
            out.extend(self.forward(&input[done * self.cols..])?);
            return Ok(out);
        }
        if let Some((kind, packed)) = &self.packed {
            if crate::cpu_exl::handles(*kind) {
                return crate::cpu_exl::linear(packed, *kind, input, self.rows, self.cols, self.bias.as_deref());
            }
        }
        // MLX affine weights: f32 activations against the packed codes (also the reference of the device kernels)
        if let Some((kind, packed)) = &self.packed {
            if crate::cpu_mlx::handles(*kind) {
                if let Some(out) = crate::cpu_mlx::linear(packed, *kind, input, self.rows, self.cols, self.bias.as_deref()) {
                    return Ok(out);
                }
            }
        }
        if self.fast_cpu {
            let float = if let Some((kind @ (1 | 30), packed)) = &self.packed {
                Some(crate::cpu_float::linear(
                    packed,
                    *kind,
                    input,
                    self.rows,
                    self.cols,
                    self.bias.as_deref(),
                ))
            } else if self.packed.is_none() {
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        self.weight.data.as_ptr().cast::<u8>(),
                        self.weight.data.len() * 4,
                    )
                };
                Some(crate::cpu_float::linear(
                    bytes,
                    0,
                    input,
                    self.rows,
                    self.cols,
                    self.bias.as_deref(),
                ))
            } else {
                None
            };
            if let Some(Some(out)) = float {
                return Ok(out);
            }
            #[cfg(target_arch = "x86_64")]
            if let Some((kind, packed)) = &self.packed {
                if crate::cpu_q4z::handles(*kind) {
                    if let Some(out) = crate::cpu_q4z::linear(
                        packed,
                        *kind,
                        input,
                        self.rows,
                        self.cols,
                        self.bias.as_deref(),
                    ) {
                        return Ok(out);
                    }
                }
            }
            if let Some((kind, packed)) = &self.packed {
                if !matches!(kind, 1 | 8 | 30) {
                    if let Some(out) = crate::cpu_quantized::linear(
                        packed,
                        *kind,
                        input,
                        self.rows,
                        self.cols,
                        self.bias.as_deref(),
                    ) {
                        return Ok(out);
                    }
                }
            }
            if let Some((8, packed)) = &self.packed {
                if let Some(out) = crate::cpu_kernels::q8_linear(
                    packed,
                    input,
                    self.rows,
                    self.cols,
                    self.bias.as_deref(),
                ) {
                    return Ok(out);
                }
            }
        }
        let mut out = vec![0.0; input.len() / self.cols * self.rows];
        out.par_chunks_mut(64).enumerate().try_for_each(
            |(chunk, output)| -> Result<(), String> {
                for (j, y) in output.iter_mut().enumerate() {
                    let index = chunk * 64 + j;
                    let b = index / self.rows;
                    let i = index % self.rows;
                    let x = &input[b * self.cols..(b + 1) * self.cols];
                    let sum = if let Some((kind, packed)) = &self.packed {
                        if fllm_gguf_probe::fp8m_type(*kind) {
                            let s_bytes = [
                                packed[self.rows * self.cols + i * 2],
                                packed[self.rows * self.cols + i * 2 + 1],
                            ];
                            let scale = half::f16::from_le_bytes(s_bytes).to_f32();
                            let row_w = &packed[i * self.cols..(i + 1) * self.cols];
                            let mut row_sum = 0.0f32;
                            for (&xv, &b) in x.iter().zip(row_w) {
                                row_sum += xv * (e4m3fn_to_f32(b) * scale);
                            }
                            row_sum
                        } else if let Some(group) = fllm_gguf_probe::q4zp_group(*kind) {
                            // GPTQ with act-order: Q4Z rows over inputs sorted by group, then the permutation (u32 per input)
                            let group = group as usize;
                            let row_bytes = self.cols / group * (group / 2 + 3);
                            let row = &packed[i * row_bytes..(i + 1) * row_bytes];
                            let perm = &packed[self.rows * row_bytes..];
                            let values = dequantize(row, 0x8000_0010 + (group.trailing_zeros() - 5), self.cols)?;
                            let mut row_sum = 0.0f32;
                            for (j, &w) in values.iter().enumerate() {
                                let source = u32::from_le_bytes(perm[j * 4..j * 4 + 4].try_into().unwrap()) as usize;
                                row_sum += x.get(source).copied().ok_or("perm index out of bounds")? * w;
                            }
                            row_sum
                        } else {
                            let width = packed.len() / self.rows;
                            let row = &packed[i * width..(i + 1) * width];
                            if matches!(kind, 1 | 2 | 3 | 6 | 7 | 8 | 12 | 13 | 14 | 30) {
                                dot_quantized(x, row, *kind)
                            } else {
                                // Keep the model packed: only one row of temporary f32
                                // values is needed per worker, not a full-model sidecar.
                                let values = dequantize(row, *kind, self.cols)?;
                                if values.iter().any(|v| !v.is_finite()) {
                                    return Err("non-finite packed projection weights".into());
                                }
                                dot(x, &values)
                            }
                        }
                    } else {
                        dot(x, &self.weight.data[i * self.cols..(i + 1) * self.cols])
                    };
                    *y = sum * self.scale + self.bias.as_ref().map_or(0.0, |bias| bias[i]);
                }
                Ok(())
            },
        )?;
        Ok(out)
    }
    pub fn replace_weights(&mut self, weights: Option<Vec<f32>>) -> Result<(), String> {
        if let Some(weights) = weights {
            if weights.len() != self.rows * self.cols {
                return Err("adapted projection shape mismatch".into());
            }
            self.weight.data = weights;
            self.packed = None;
        }
        Ok(())
    }
    /// Quantized activations match native decoder kernels. Encoders/vision
    /// retain their full-precision activation reference unless explicitly enabled.
    pub fn enable_fast_cpu(&mut self) {
        self.fast_cpu = true;
    }
}
fn dot_quantized(x: &[f32], packed: &[u8], kind: u32) -> f32 {
    if matches!(kind, 1 | 30) {
        return x
            .iter()
            .zip(packed.as_chunks::<2>().0.iter())
            .map(|(&x, b)| {
                x * if kind == 1 {
                    read_half(b)
                } else {
                    f32::from_bits((u16::from_le_bytes(*b) as u32) << 16)
                }
            })
            .sum();
    }
    #[cfg(target_arch = "x86_64")]
    if kind == 8
        && std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
    {
        // Layout and slice lengths are checked when the projection is loaded.
        return unsafe { dot_q8_avx2(x, packed) };
    }
    let (block, width) = quant_layout(kind).expect("validated quantization");
    let mut sum = 0.0;
    for (x, b) in x.chunks_exact(block).zip(packed.chunks_exact(width)) {
        if kind < 8 {
            for (j, &v) in x.iter().enumerate() {
                sum += v * classic_value(b, kind, j);
            }
        } else if kind == 8 {
            let d = read_half(b);
            let mut lanes = [0.0; 8];
            for j in 0..32 {
                lanes[j % 8] += x[j] * (b[j + 2] as i8) as f32;
            }
            sum += d * lanes.iter().sum::<f32>();
        } else if kind == 12 || kind == 13 {
            let d = read_half(b);
            let min = read_half(&b[2..]);
            let scales = &b[4..16];
            let qs = &b[if kind == 13 { 48 } else { 16 }..];
            for g in 0..8 {
                let (s, m) = if g < 4 {
                    (scales[g] & 63, scales[g + 4] & 63)
                } else {
                    (
                        (scales[g + 4] & 15) | ((scales[g - 4] >> 6) << 4),
                        (scales[g + 4] >> 4) | ((scales[g] >> 6) << 4),
                    )
                };
                for j in 0..32 {
                    let byte = qs[g / 2 * 32 + j];
                    let mut q = if g % 2 == 0 { byte & 15 } else { byte >> 4 };
                    if kind == 13 {
                        q |= ((b[16 + j] >> g) & 1) << 4;
                    }
                    sum += x[g * 32 + j] * (d * s as f32 * q as f32 - min * m as f32);
                }
            }
        } else {
            let d = read_half(&b[208..]);
            for n in 0..2 {
                let ql = &b[n * 64..];
                let qh = &b[128 + n * 32..];
                let scales = &b[192 + n * 8..];
                for j in 0..32 {
                    let values = [
                        (ql[j] & 15) | ((qh[j] & 3) << 4),
                        (ql[j + 32] & 15) | (((qh[j] >> 2) & 3) << 4),
                        (ql[j] >> 4) | (((qh[j] >> 4) & 3) << 4),
                        (ql[j + 32] >> 4) | (((qh[j] >> 6) & 3) << 4),
                    ];
                    for g in 0..4 {
                        sum += x[n * 128 + g * 32 + j]
                            * (d * (scales[j / 16 + g * 2] as i8) as f32
                                * (values[g] as f32 - 32.0));
                    }
                }
            }
        }
    }
    sum
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q8_avx2(x: &[f32], packed: &[u8]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = 0.0;
    for (values, block) in x.as_chunks::<32>().0.iter().zip(packed.as_chunks::<34>().0) {
        let mut accumulator = _mm256_setzero_ps();
        for j in [0, 8, 16, 24] {
            // Each load stays inside a validated 32-value block; loadl reads 8 bytes.
            let bytes = unsafe { _mm_loadl_epi64(block.as_ptr().add(2 + j).cast()) };
            let weights = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
            let input = unsafe { _mm256_loadu_ps(values.as_ptr().add(j)) };
            accumulator = _mm256_fmadd_ps(input, weights, accumulator);
        }
        let mut lanes = [0.0; 8];
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), accumulator) };
        sum += read_half(block) * lanes.iter().sum::<f32>();
    }
    sum
}
fn classic_value(b: &[u8], kind: u32, j: usize) -> f32 {
    let d = read_half(b);
    let affine = matches!(kind, 3 | 7);
    let five = matches!(kind, 6 | 7);
    let offset = if affine { 4 } else { 2 };
    let qoffset = offset + if five { 4 } else { 0 };
    let byte = b[qoffset + j % 16];
    let mut q = if j < 16 { byte & 15 } else { byte >> 4 };
    if five {
        let high = u32::from_le_bytes(b[offset..offset + 4].try_into().unwrap());
        q |= (((high >> j) & 1) as u8) << 4;
    }
    if affine {
        d * q as f32 + read_half(&b[2..])
    } else {
        d * (q as f32 - if five { 16.0 } else { 8.0 })
    }
}
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sums = [0.0; 8];
    let end = a.len() / 8 * 8;
    for (x, y) in a[..end]
        .as_chunks::<8>()
        .0
        .iter()
        .zip(b[..end].as_chunks::<8>().0)
    {
        for j in 0..8 {
            sums[j] += x[j] * y[j];
        }
    }
    sums.iter().sum::<f32>()
        + a[end..]
            .iter()
            .zip(&b[end..])
            .map(|(x, y)| x * y)
            .sum::<f32>()
}
pub fn add(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

pub struct LayerNorm {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub eps: f32,
}
impl LayerNorm {
    pub fn from_parts(weight: Vec<f32>, bias: Vec<f32>, eps: f32) -> Self {
        Self { weight, bias, eps }
    }
    pub fn load(model: &MappedGguf, base: &str, hidden: usize, eps: f32) -> Result<Self, String> {
        let weight = Tensor::load(model, &format!("{base}.weight"))?;
        let bias = Tensor::load(model, &format!("{base}.bias"))?;
        if weight.shape != [hidden] || bias.shape != [hidden] {
            return Err(format!("invalid norm dimensions: {base}"));
        }
        Ok(Self {
            weight: weight.data,
            bias: bias.data,
            eps,
        })
    }
    pub fn apply(&self, x: &mut [f32]) {
        let hidden = self.weight.len();
        x.par_chunks_mut(hidden).for_each(|row| {
            let mean = row.iter().sum::<f32>() / hidden as f32;
            let variance = row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / hidden as f32;
            let inv = (variance + self.eps).sqrt().recip();
            for (j, x) in row.iter_mut().enumerate() {
                *x = (*x - mean) * inv * self.weight[j] + self.bias[j];
            }
        });
    }
}
/// Bidirectional scaled dot-product attention, independently over every head.
pub fn attention(q: &[f32], k: &[f32], v: &[f32], hidden: usize, heads: usize) -> Vec<f32> {
    let n = q.len() / hidden;
    let dim = hidden / heads;
    let scale = (dim as f32).sqrt().recip();
    let mut out = vec![0.0; q.len()];
    out.par_chunks_mut(hidden).enumerate().for_each(|(i, row)| {
        let mut scores = vec![0.0; n];
        for h in 0..heads {
            let offset = h * dim;
            for (j, s) in scores.iter_mut().enumerate() {
                *s = dot(
                    &q[i * hidden + offset..i * hidden + offset + dim],
                    &k[j * hidden + offset..j * hidden + offset + dim],
                ) * scale;
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            for s in &mut scores {
                *s = (*s - max).exp();
            }
            let sum = scores.iter().sum::<f32>();
            for (j, s) in scores.iter().enumerate() {
                for d in 0..dim {
                    row[offset + d] += *s / sum * v[j * hidden + offset + d];
                }
            }
        }
    });
    out
}
/// GELU uses the error function approximation (not the tanh GELU variant).
pub fn gelu(x: f32) -> f32 {
    let z = x * std::f32::consts::FRAC_1_SQRT_2;
    let a = z.abs();
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let erf = 1.0
        - (((((1.0614054 * t - 1.4531521) * t) + 1.4214138) * t - 0.28449672) * t + 0.2548296)
            * t
            * (-a * a).exp();
    0.5 * x * (1.0 + erf.copysign(z))
}
pub fn normalize(x: &mut [f32]) -> Result<(), String> {
    if x.iter().any(|v| !v.is_finite()) {
        return Err("non-finite encoder result".into());
    }
    let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm == 0.0 {
        return Err("zero encoder result".into());
    }
    for v in x {
        *v /= norm;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn per_expert_scales_are_picked_by_the_rows_they_cover() {
        // one value: every row; four experts of 8 rows (gate and up of an expert share a slice when the tensor is fused)
        assert_eq!(scale_for_rows(&[0.5], 24, 32), Some(0.5));
        let per_expert = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(scale_for_rows(&per_expert, 0, 32), Some(1.0));
        assert_eq!(scale_for_rows(&per_expert, 8, 32), Some(2.0));
        assert_eq!(scale_for_rows(&per_expert, 31, 32), Some(4.0));
        assert_eq!(scale_for_rows(&per_expert, 16, 32), Some(3.0));
        assert_eq!(scale_for_rows(&per_expert, 4, 30), None, "30 rows do not split into 4 slices");
        assert_eq!(scale_for_rows(&[], 0, 8), None);
    }
    #[test]
    fn signed_quantization_and_validation() {
        let mut block = vec![0; 34];
        block[..2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        block[2] = 254;
        block[3] = 3;
        let x = dequantize(&block, 8, 32).unwrap();
        assert_eq!(&x[..2], &[-1.0, 1.5]);
        assert!(dequantize(&block, 8, 31).is_err());
        assert!(dequantize(&block[..33], 8, 32).is_err());
    }
    #[test]
    fn bidirectional_attention_sees_future_tokens() {
        let x = attention(&[1.0, 0.0], &[1.0, 0.0], &[2.0, 6.0], 1, 1);
        assert!((x[0] - 3.0757656).abs() < 1e-5);
        assert!((x[1] - 4.0).abs() < 1e-5);
    }
    #[test]
    fn gelu_matches_known_values() {
        assert!(gelu(0.0).abs() < 1e-6);
        assert!((gelu(1.0) - 0.8413447).abs() < 1e-6);
        assert!((gelu(-1.0) + 0.1586553).abs() < 1e-6);
    }
}
