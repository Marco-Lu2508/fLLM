//! `fllm-rs quantize`: rewrite the float tensors of a GGUF file in a block-quantized format.
//! The metadata section is copied byte for byte; only tensor infos and data are rewritten.
use crate::quantize::{quantize_row, QType};
use crate::CheckpointError;
use half::{bf16, f16};
use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Named mixes, modelled on llama.cpp's `llama-quantize` types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2K,
    Q3K,
    Q4KS,
    Q4KM,
    Q5KS,
    Q5KM,
    Q6K,
    Mxfp4,
    Nvfp4,
}

impl Preset {
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "q4_0" => Self::Q4_0,
            "q4_1" => Self::Q4_1,
            "q5_0" => Self::Q5_0,
            "q5_1" => Self::Q5_1,
            "q8_0" => Self::Q8_0,
            "q2_k" | "q2k" => Self::Q2K,
            "q3_k" | "q3k" => Self::Q3K,
            "mxfp4" => Self::Mxfp4,
            "nvfp4" => Self::Nvfp4,
            "q4_k_s" => Self::Q4KS,
            "q4_k_m" | "q4_k" => Self::Q4KM,
            "q5_k_s" => Self::Q5KS,
            "q5_k_m" | "q5_k" => Self::Q5KM,
            "q6_k" => Self::Q6K,
            _ => return None,
        })
    }
    pub const NAMES: &'static str =
        "q4_0, q4_1, q5_0, q5_1, q8_0, q2_k, q3_k, q4_k_s, q4_k_m, q5_k_s, q5_k_m, q6_k, mxfp4, nvfp4";

    fn base(self) -> QType {
        match self {
            Self::Q4_0 => QType::Q4_0,
            Self::Q4_1 => QType::Q4_1,
            Self::Q5_0 => QType::Q5_0,
            Self::Q5_1 => QType::Q5_1,
            Self::Q8_0 => QType::Q8_0,
            Self::Q2K => QType::Q2K,
            Self::Q3K => QType::Q3K,
            Self::Mxfp4 => QType::Mxfp4,
            Self::Nvfp4 => QType::Nvfp4,
            Self::Q4KS | Self::Q4KM => QType::Q4K,
            Self::Q5KS | Self::Q5KM => QType::Q5K,
            Self::Q6K => QType::Q6K,
        }
    }
    fn is_k(self) -> bool {
        matches!(
            self,
            Self::Q2K | Self::Q3K | Self::Q4KS | Self::Q4KM | Self::Q5KS | Self::Q5KM | Self::Q6K
        )
    }
}

fn more_bits(i: usize, n: usize) -> bool {
    i < n / 8 || i >= 7 * n / 8 || (i - n / 8) % 3 == 2
}

/// Target type for one tensor, or `None` to keep it unchanged.
fn choose(preset: Preset, name: &str, dims: &[u64], layers: usize) -> Option<QType> {
    let quantizable = name.ends_with(".weight")
        && dims.len() >= 2
        && !name.contains("norm")
        && !name.contains("ffn_gate_inp")
        && !name.contains("rope_")
        && !name.starts_with("v.")
        && !name.contains("ssm_")
        && !name.contains("pos_embd");
    if !quantizable {
        return None;
    }
    let layer = name
        .strip_prefix("blk.")
        .and_then(|r| r.split('.').next())
        .and_then(|i| i.parse::<usize>().ok());
    let mut kind = preset.base();
    if name == "output.weight" && preset != Preset::Q8_0 {
        kind = QType::Q6K;
    } else if let (true, Some(i)) = (preset.is_k(), layer) {
        let attn_v = name.contains("attn_v.weight");
        let ffn_down = name.contains("ffn_down");
        kind = match preset {
            Preset::Q4KM | Preset::Q5KM if (attn_v || ffn_down) && more_bits(i, layers) => {
                QType::Q6K
            }
            Preset::Q4KS if attn_v && i < 4 => QType::Q5K,
            Preset::Q4KS if ffn_down && i < layers / 8 => QType::Q5K,
            _ => kind,
        };
    }
    let cols = dims[0] as usize;
    if !cols.is_multiple_of(kind.block_elements()) {
        kind = match kind {
            QType::Q2K | QType::Q3K | QType::Q4K => QType::Q5_0,
            QType::Q5K => QType::Q5_1,
            QType::Q6K => QType::Q8_0,
            other => other,
        };
        if !cols.is_multiple_of(kind.block_elements()) {
            return None;
        }
    }
    Some(kind)
}

struct Info {
    name: String,
    dims: Vec<u64>,
    type_id: u32,
    offset: u64,
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], String> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.b.len());
        let end = end.ok_or("truncated GGUF file")?;
        let s = &self.b[self.at..end];
        self.at = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, String> {
        let n = self.u64()? as usize;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| "invalid UTF-8 in GGUF".to_string())
    }
    /// Skip one metadata value of type `t`; returns the value when it is an unsigned integer.
    fn skip_value(&mut self, t: u32) -> Result<Option<u64>, String> {
        Ok(match t {
            0 | 1 | 7 => {
                self.take(1)?;
                None
            }
            2 | 3 => {
                self.take(2)?;
                None
            }
            4 => Some(u64::from(self.u32()?)),
            5 | 6 => {
                self.take(4)?;
                None
            }
            8 => {
                self.string()?;
                None
            }
            9 => {
                let inner = self.u32()?;
                let n = self.u64()?;
                for _ in 0..n {
                    self.skip_value(inner)?;
                }
                None
            }
            10..=12 => {
                self.take(8)?;
                None
            }
            other => return Err(format!("unknown GGUF metadata type {other}")),
        })
    }
}

pub struct QuantizeReport {
    pub tensors: usize,
    pub quantized: usize,
    pub input_bytes: u64,
    pub output_bytes: u64,
}

fn element_size(type_id: u32) -> Option<usize> {
    match type_id {
        0 => Some(4),
        1 | 30 => Some(2),
        _ => None,
    }
}

/// Byte size of a tensor of any supported type (needed to copy unconverted tensors).
fn stored_bytes(type_id: u32, elements: u64) -> Result<u64, String> {
    let (block, bytes) = fllm_gguf_probe::quant_types::GGML_TYPES
        .iter()
        .find(|t| t.id == type_id)
        .map(|t| (t.block_elements as u64, t.bytes_per_block as u64))
        .ok_or_else(|| format!("unknown tensor type {type_id}"))?;
    Ok(elements / block * bytes)
}

fn row_to_f32(type_id: u32, raw: &[u8], out: &mut Vec<f32>) {
    out.clear();
    match type_id {
        0 => out.extend(
            raw.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c)),
        ),
        1 => out.extend(
            raw.as_chunks::<2>()
                .0
                .iter()
                .map(|c| f16::from_bits(u16::from_le_bytes(*c)).to_f32()),
        ),
        _ => out.extend(
            raw.as_chunks::<2>()
                .0
                .iter()
                .map(|c| bf16::from_bits(u16::from_le_bytes(*c)).to_f32()),
        ),
    }
}

pub fn quantize_gguf(
    input: &Path,
    output: &Path,
    preset: Preset,
) -> Result<QuantizeReport, CheckpointError> {
    let file = File::open(input)?;
    // SAFETY: the file is only read and not modified while the mapping is alive.
    let map = unsafe { Mmap::map(&file)? };
    let mut r = Reader { b: &map, at: 0 };
    if r.take(4)? != b"GGUF" {
        return Err("not a GGUF file".into());
    }
    let version = r.u32()?;
    if version != 3 {
        return Err(format!("unsupported GGUF version {version}").into());
    }
    let tensor_count = r.u64()? as usize;
    let kv_count = r.u64()?;
    let mut alignment = 32u64;
    for _ in 0..kv_count {
        let key = r.string()?;
        let t = r.u32()?;
        let v = r.skip_value(t)?;
        if key == "general.alignment" {
            alignment = v
                .filter(|a| a.is_power_of_two())
                .ok_or("invalid general.alignment")?;
        }
    }
    let prefix_end = r.at;
    let mut infos = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = r.string()?;
        let n = r.u32()? as usize;
        let dims = (0..n).map(|_| r.u64()).collect::<Result<Vec<_>, _>>()?;
        let type_id = r.u32()?;
        let offset = r.u64()?;
        infos.push(Info {
            name,
            dims,
            type_id,
            offset,
        });
    }
    let data_start = (r.at as u64).div_ceil(alignment) * alignment;
    let layers = infos
        .iter()
        .filter_map(|t| {
            t.name
                .strip_prefix("blk.")?
                .split('.')
                .next()?
                .parse::<usize>()
                .ok()
        })
        .max()
        .map_or(0, |m| m + 1);

    // plan the output layout
    struct Out {
        kind: Option<QType>,
        type_id: u32,
        bytes: u64,
    }
    let mut plan = Vec::with_capacity(infos.len());
    for t in &infos {
        let elements: u64 = t.dims.iter().product();
        let kind = if element_size(t.type_id).is_some() {
            choose(preset, &t.name, &t.dims, layers)
        } else {
            None
        };
        plan.push(match kind {
            Some(k) => Out {
                kind,
                type_id: k.ggml_id(),
                bytes: k.row_bytes(elements as usize) as u64,
            },
            None => Out {
                kind: None,
                type_id: t.type_id,
                bytes: stored_bytes(t.type_id, elements)?,
            },
        });
    }

    let mut w = BufWriter::with_capacity(1 << 20, File::create(output)?);
    w.write_all(&map[..prefix_end])?;
    let mut offset = 0u64;
    let mut head = Vec::new();
    for (t, p) in infos.iter().zip(&plan) {
        head.extend_from_slice(&(t.name.len() as u64).to_le_bytes());
        head.extend_from_slice(t.name.as_bytes());
        head.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            head.extend_from_slice(&d.to_le_bytes());
        }
        head.extend_from_slice(&p.type_id.to_le_bytes());
        head.extend_from_slice(&offset.to_le_bytes());
        offset += p.bytes.div_ceil(alignment) * alignment;
    }
    w.write_all(&head)?;
    let written = prefix_end as u64 + head.len() as u64;
    let pad = |w: &mut BufWriter<File>, at: u64| -> std::io::Result<u64> {
        let target = at.div_ceil(alignment) * alignment;
        w.write_all(&vec![0u8; (target - at) as usize])?;
        Ok(target)
    };
    let mut at = pad(&mut w, written)?;

    let mut quantized = 0;
    for (t, p) in infos.iter().zip(&plan) {
        let src_len = stored_bytes(t.type_id, t.dims.iter().product())? as usize;
        let start = data_start as usize + t.offset as usize;
        let src = map
            .get(start..start + src_len)
            .ok_or_else(|| format!("tensor {} lies outside the file", t.name))?;
        match p.kind {
            None => w.write_all(src)?,
            Some(kind) => {
                quantized += 1;
                let cols = t.dims[0] as usize;
                let width = element_size(t.type_id).unwrap();
                let rows = src.len() / (cols * width);
                // quantize in parallel batches of rows, then write in order
                let batch = 256usize;
                for first in (0..rows).step_by(batch) {
                    let last = (first + batch).min(rows);
                    let blocks: Vec<Vec<u8>> = (first..last)
                        .into_par_iter()
                        .map(|row| {
                            let mut values = Vec::with_capacity(cols);
                            row_to_f32(
                                t.type_id,
                                &src[row * cols * width..(row + 1) * cols * width],
                                &mut values,
                            );
                            let mut bytes = Vec::with_capacity(kind.row_bytes(cols));
                            quantize_row(kind, &values, &mut bytes);
                            bytes
                        })
                        .collect();
                    for b in blocks {
                        w.write_all(&b)?;
                    }
                }
            }
        }
        at += p.bytes;
        at = pad(&mut w, at)?;
    }
    w.flush()?;
    Ok(QuantizeReport {
        tensors: infos.len(),
        quantized,
        input_bytes: map.len() as u64,
        output_bytes: at,
    })
}
