pub mod bnb;
pub mod exl;
pub mod quant_types;
use memmap2::{Mmap, MmapOptions};
use serde::Serialize;
use serde_json::{Map, Number, Value};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io;
use std::path::Path;
use std::sync::Arc;
#[derive(Debug)]
struct Cursor<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], String> {
        let end = self.offset.checked_add(count).ok_or("offset overflow")?;
        let bytes = self.data.get(self.offset..end).ok_or_else(|| {
            format!(
                "truncated GGUF at byte {} (need {count} bytes)",
                self.offset
            )
        })?;
        self.offset = end;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn i8(&mut self) -> Result<i8, String> {
        Ok(self.u8()? as i8)
    }

    fn i16(&mut self) -> Result<i16, String> {
        Ok(i16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn i32(&mut self) -> Result<i32, String> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn i64(&mut self) -> Result<i64, String> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String, String> {
        let len = usize::try_from(self.u64()?).map_err(|_| "string length does not fit usize")?;
        let bytes = self.take(len)?;
        let value = std::str::from_utf8(bytes).map_err(|e| format!("invalid GGUF UTF-8: {e}"))?;
        Ok(value.to_owned())
    }
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub type_id: u32,
    pub data_offset: u64,
    pub payload_len: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct GgufIndex {
    pub version: u32,
    pub metadata: Map<String, Value>,
    pub tensors: Vec<TensorInfo>,
    pub data_offset: u64,
    pub file_len: u64,
}

fn json_float(v: f64) -> Result<Value, String> {
    Number::from_f64(v)
        .map(Value::Number)
        .ok_or_else(|| format!("non-finite GGUF metadata float {v}"))
}

fn scalar(c: &mut Cursor<'_>, kind: u32) -> Result<Value, String> {
    match kind {
        0 => Ok(Value::from(c.u8()?)),
        1 => Ok(Value::from(c.i8()?)),
        2 => Ok(Value::from(u16::from_le_bytes(
            c.take(2)?.try_into().unwrap(),
        ))),
        3 => Ok(Value::from(c.i16()?)),
        4 => Ok(Value::from(c.u32()?)),
        5 => Ok(Value::from(c.i32()?)),
        6 => json_float(c.f32()? as f64),
        7 => match c.u8()? {
            0 => Ok(Value::Bool(false)),
            1 => Ok(Value::Bool(true)),
            other => Err(format!("invalid GGUF boolean byte {other}")),
        },
        8 => Ok(Value::String(c.string()?)),
        10 => Ok(Value::from(c.u64()?)),
        11 => Ok(Value::from(c.i64()?)),
        12 => json_float(c.f64()?),
        _ => Err(format!("unsupported GGUF metadata scalar type {kind}")),
    }
}

fn min_encoded_width(kind: u32) -> Result<usize, String> {
    match kind {
        0 | 1 | 7 => Ok(1),
        2 | 3 => Ok(2),
        4..=6 => Ok(4),
        8 | 10 | 11 | 12 => Ok(8),
        _ => Err(format!("unsupported GGUF array element type {kind}")),
    }
}

fn value(c: &mut Cursor<'_>, kind: u32) -> Result<Value, String> {
    if kind != 9 {
        return scalar(c, kind);
    }
    let element_type = c.u32()?;
    if element_type == 9 {
        return Err("nested GGUF arrays are invalid".to_owned());
    }
    let count = usize::try_from(c.u64()?).map_err(|_| "array count does not fit usize")?;
    let min_width = min_encoded_width(element_type)?;
    if count > c.data.len().saturating_sub(c.offset) / min_width {
        return Err(format!(
            "GGUF array count {count} exceeds remaining file data"
        ));
    }
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(scalar(c, element_type)?);
    }
    Ok(Value::Array(values))
}

fn checked_count(count: u64, max: u64, label: &str) -> Result<usize, String> {
    if count > max {
        return Err(format!("unreasonable GGUF {label}: {count}"));
    }
    usize::try_from(count).map_err(|_| format!("GGUF {label} does not fit usize"))
}

fn align_up(value: usize, alignment: usize) -> Result<usize, String> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(format!("invalid GGUF alignment {alignment}"));
    }
    value
        .checked_add(alignment - 1)
        .map(|v| v & !(alignment - 1))
        .ok_or_else(|| "GGUF alignment overflow".to_owned())
}

/// Group size of the fLLM-specific `Q4Z` tensor types (4-bit AWQ / GPTQ weights): ids 0x8000_0010 + log2(group / 32).
pub fn q4z_group(type_id: u32) -> Option<u64> {
    match type_id {
        0x8000_0010 | 0x8000_0014 => Some(32),
        0x8000_0011 | 0x8000_0015 => Some(64),
        0x8000_0012 | 0x8000_0016 => Some(128),
        _ => None,
    }
}

/// Group size of the fLLM-specific `Q4ZP` tensor types (4-bit GPTQ weights with column permutation table): ids 0x8000_0014 + log2(group / 32).
pub fn q4zp_group(type_id: u32) -> Option<u64> {
    match type_id {
        0x8000_0014 => Some(32),
        0x8000_0015 => Some(64),
        0x8000_0016 => Some(128),
        _ => None,
    }
}

/// Geometry of the fLLM-specific `MLXA` tensor types: MLX affine group quantization kept native.
///
/// Type id `0x8001_0000 | bf16 << 12 | (group / 32) << 4 | bits`. A row is `cols / group` blocks; a block holds the values of one group:
/// the scale and the bias (`w = scale * code + bias`) as two 16-bit floats (f16 or bf16, as the checkpoint stores them), then
/// `group * bits / 8` bytes of codes in the MLX bit order (a little-endian bit stream, code `i` at bit `i * bits`; the 3, 5 and 6 bit
/// codes straddle the 32-bit words).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MlxAffine {
    pub bits: u32,
    pub group: u32,
    /// Scales and biases are bfloat16 (otherwise IEEE half).
    pub bf16: bool,
}

impl MlxAffine {
    /// Bytes of one block (one group).
    pub fn block_bytes(&self) -> u64 {
        u64::from(self.group) * u64::from(self.bits) / 8 + 4
    }

    pub fn type_id(&self) -> u32 {
        MLX_AFFINE_BASE | (u32::from(self.bf16) << 12) | ((self.group / 32) << 4) | self.bits
    }
}

/// First id of the `MLXA` tensor types.
pub const MLX_AFFINE_BASE: u32 = 0x8001_0000;

/// Whether `bits` / `group` can be stored as an `MLXA` type: 2, 3, 4, 5, 6 or 8 bits and a group that is a multiple of 32 (up to 8160).
pub fn mlx_affine_supported(bits: u32, group: u32) -> bool {
    matches!(bits, 2..=6 | 8) && group != 0 && group.is_multiple_of(32) && group / 32 <= 255
}

pub fn mlx_affine(type_id: u32) -> Option<MlxAffine> {
    if type_id & !0x1fff != MLX_AFFINE_BASE {
        return None;
    }
    let found = MlxAffine { bits: type_id & 15, group: ((type_id >> 4) & 255) * 32, bf16: type_id & 0x1000 != 0 };
    mlx_affine_supported(found.bits, found.group).then_some(found)
}

/// fLLM-specific `FP8M` tensor type (FP8 E4M3 weights with per-channel FP16 scales for Marlin W8A16): id 0x8000_0020.
pub fn fp8m_type(type_id: u32) -> bool {
    type_id == 0x8000_0020
}

/// Tile shape `(rows, columns)` of the fLLM-specific `FP8B` tensor types (FP8 E4M3 weights with one scale per tile, the
/// `weight_scale_inv` of block-wise FP8 checkpoints). The shape is part of the type id: `0x8000_0021` is the usual 128 x 128 tile,
/// `0x8000_0100 | (log2(rows) << 4) | log2(columns)` any other power-of-two tile up to 32768.
pub fn fp8b_shape(type_id: u32) -> Option<(usize, usize)> {
    match type_id {
        0x8000_0021 => Some((128, 128)),
        0x8000_0100..=0x8000_01ff => {
            let code = type_id & 0xff;
            Some((1usize << (code >> 4), 1usize << (code & 15)))
        }
        _ => None,
    }
}

/// Whether `type_id` is one of the `FP8B` types (see [`fp8b_shape`]).
pub fn fp8b_type(type_id: u32) -> bool {
    fp8b_shape(type_id).is_some()
}

/// The `FP8B` type id of a tile shape; `None` unless both sides are powers of two up to 32768.
pub fn fp8b_type_id(block_rows: usize, block_cols: usize) -> Option<u32> {
    if (block_rows, block_cols) == (128, 128) {
        return Some(0x8000_0021);
    }
    if !block_rows.is_power_of_two() || !block_cols.is_power_of_two() {
        return None;
    }
    let (lr, lc) = (block_rows.trailing_zeros(), block_cols.trailing_zeros());
    (lr <= 15 && lc <= 15).then_some(0x8000_0100 | (lr << 4) | lc)
}

/// Size of the trailer behind the scales of an `FP8B` tensor.
pub const FP8B_TRAILER: usize = 16;
const FP8B_MAGIC: u32 = u32::from_le_bytes(*b"F8B1");

/// Byte layout of an `FP8B` tensor, a stack of `slabs` weight matrices of `rows` x `cols` (the experts of a mixture-of-experts layer;
/// 1 for a plain matrix):
///
/// 1. `slabs * rows * cols` E4M3 bytes, row-major, exactly as in the checkpoint;
/// 2. one little-endian `f32` scale per tile, slab after slab, tiles row-major (`ceil(rows / tile_rows) * ceil(cols / tile_cols)`
///    per slab); a weight is `e4m3(code) * scale`, the convention of `weight_scale_inv`;
/// 3. a 16-byte trailer `"F8B1", cols, rows, slabs` (little-endian `u32`), so that a reader that only holds the bytes (a
///    generic dequantizer, a backend that expands the weights at upload) knows the geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp8bLayout {
    pub block_rows: usize,
    pub block_cols: usize,
    pub rows: usize,
    pub cols: usize,
    pub slabs: usize,
}

impl Fp8bLayout {
    /// The layout of a tensor of GGUF type `type_id` and `slabs` matrices of `rows` x `cols`.
    pub fn new(type_id: u32, cols: usize, rows: usize, slabs: usize) -> Option<Self> {
        let (block_rows, block_cols) = fp8b_shape(type_id)?;
        (cols > 0 && rows > 0 && slabs > 0).then_some(Self { block_rows, block_cols, rows, cols, slabs })
    }

    /// The layout of a tensor with GGUF dimensions `[cols, rows, slabs...]` (a rank-1 tensor is one row).
    pub fn from_dims(type_id: u32, dims: &[u64]) -> Option<Self> {
        let cols = usize::try_from(*dims.first()?).ok()?;
        let rows = usize::try_from(dims.get(1).copied().unwrap_or(1)).ok()?;
        let slabs = dims.iter().skip(2).try_fold(1usize, |n, &d| n.checked_mul(usize::try_from(d).ok()?))?;
        Self::new(type_id, cols, rows, slabs)
    }

    pub fn elements(&self) -> usize {
        self.slabs.saturating_mul(self.rows).saturating_mul(self.cols)
    }

    /// Tiles per row of tiles.
    pub fn tile_cols(&self) -> usize {
        self.cols.div_ceil(self.block_cols)
    }

    pub fn tiles_per_slab(&self) -> usize {
        self.rows.div_ceil(self.block_rows).saturating_mul(self.tile_cols())
    }

    pub fn tiles(&self) -> usize {
        self.slabs.saturating_mul(self.tiles_per_slab())
    }

    /// Offset of the scale table.
    pub fn scale_offset(&self) -> usize {
        self.elements()
    }

    /// Total bytes: codes, scales, trailer.
    pub fn total_len(&self) -> usize {
        self.elements().saturating_add(self.tiles().saturating_mul(4)).saturating_add(FP8B_TRAILER)
    }

    /// Index into the scale table of the tile that holds element (`row`, `col`) of matrix `slab`.
    pub fn tile_index(&self, slab: usize, row: usize, col: usize) -> usize {
        slab * self.tiles_per_slab() + (row / self.block_rows) * self.tile_cols() + col / self.block_cols
    }

    /// The trailer bytes.
    pub fn trailer(&self) -> [u8; FP8B_TRAILER] {
        let mut t = [0u8; FP8B_TRAILER];
        t[..4].copy_from_slice(&FP8B_MAGIC.to_le_bytes());
        t[4..8].copy_from_slice(&(self.cols as u32).to_le_bytes());
        t[8..12].copy_from_slice(&(self.rows as u32).to_le_bytes());
        t[12..].copy_from_slice(&(self.slabs as u32).to_le_bytes());
        t
    }

    /// Read the geometry from the trailer of `bytes` (a whole `FP8B` tensor) and check the length.
    pub fn parse(type_id: u32, bytes: &[u8]) -> Result<Self, String> {
        let (block_rows, block_cols) = fp8b_shape(type_id).ok_or_else(|| format!("GGUF type {type_id:#x} is not an FP8B type"))?;
        let tail = bytes
            .len()
            .checked_sub(FP8B_TRAILER)
            .map(|at| &bytes[at..])
            .ok_or("FP8B tensor shorter than its trailer")?;
        let word = |i: usize| u32::from_le_bytes([tail[4 * i], tail[4 * i + 1], tail[4 * i + 2], tail[4 * i + 3]]);
        if word(0) != FP8B_MAGIC {
            return Err("FP8B tensor without the F8B1 trailer (written by an older version?)".into());
        }
        let layout = Self { block_rows, block_cols, cols: word(1) as usize, rows: word(2) as usize, slabs: word(3) as usize };
        if layout.cols == 0 || layout.rows == 0 || layout.slabs == 0 || layout.total_len() != bytes.len() {
            return Err(format!("FP8B trailer ({} x {} x {}) does not match the {} bytes of the tensor", layout.slabs, layout.rows, layout.cols, bytes.len()));
        }
        Ok(layout)
    }

    /// The scale of tile `index` of the table in `bytes`.
    pub fn scale(&self, bytes: &[u8], index: usize) -> f32 {
        let at = self.scale_offset() + index * 4;
        f32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
    }
}

fn ggml_quant_size(type_id: u32) -> Option<(u64, u64)> {
    if let Some(group) = q4z_group(type_id) {
        // per group: group / 2 code bytes, an fp16 scale and a zero-point byte
        return Some((group, group / 2 + 3));
    }
    if fp8m_type(type_id) || fp8b_type(type_id) {
        return Some((1, 1));
    }
    if let Some(l) = bnb::layout(type_id) {
        return Some((l.block as u64, l.bytes as u64));
    }
    if let Some(mlx) = mlx_affine(type_id) {
        return Some((u64::from(mlx.group), mlx.block_bytes()));
    }
    quant_types::by_id(type_id).map(|ty| (ty.block_elements as u64, ty.bytes_per_block as u64))
}

fn tensor_payload_len(type_id: u32, dimensions: &[u64]) -> Result<u64, String> {
    let (block_size, type_size) = ggml_quant_size(type_id)
        .ok_or_else(|| format!("unsupported GGML tensor type {type_id}"))?;
    if dimensions.is_empty() || dimensions.contains(&0) {
        return Err("tensor dimensions must be nonempty and positive".into());
    }
    if !dimensions[0].is_multiple_of(block_size) {
        return Err(format!(
            "tensor row length {} is not divisible by GGML block size {block_size}",
            dimensions[0]
        ));
    }
    if fp8b_type(type_id) {
        // codes, f32 scales per tile and the trailer: the length depends on the tile shape of the type and the dimensions
        let layout = Fp8bLayout::from_dims(type_id, dimensions).ok_or("FP8B tensor dimensions do not fit this platform")?;
        let tiles = layout.tiles() as u64;
        return (layout.elements() as u64)
            .checked_add(tiles.checked_mul(4).ok_or("FP8B scale table size overflow")?)
            .and_then(|n| n.checked_add(FP8B_TRAILER as u64))
            .ok_or_else(|| "tensor payload length overflow".to_owned());
    }
    let elements = dimensions.iter().try_fold(1u64, |total, dim| {
        total
            .checked_mul(*dim)
            .ok_or_else(|| "tensor element count overflow".to_owned())
    })?;
    if elements % block_size != 0 {
        return Err(format!(
            "tensor has {elements} elements, not divisible by GGML block size {block_size}"
        ));
    }
    let q4z_len = (elements / block_size)
        .checked_mul(type_size)
        .ok_or_else(|| "tensor payload length overflow".to_owned())?;
    if q4zp_group(type_id).is_some() {
        let perm_len = dimensions[0]
            .checked_mul(4)
            .ok_or_else(|| "tensor payload length overflow".to_owned())?;
        q4z_len
            .checked_add(perm_len)
            .ok_or_else(|| "tensor payload length overflow".to_owned())
    } else if fp8m_type(type_id) {
        let scale_len = dimensions
            .get(1)
            .copied()
            .unwrap_or(1)
            .checked_mul(2)
            .ok_or_else(|| "tensor payload length overflow".to_owned())?;
        q4z_len
            .checked_add(scale_len)
            .ok_or_else(|| "tensor payload length overflow".to_owned())
    } else {
        Ok(q4z_len)
    }
}
fn parse(data: &[u8]) -> Result<GgufIndex, String> {
    let mut c = Cursor::new(data);
    if c.take(4)? != b"GGUF" {
        return Err("GGUF magic invalid or byte order unsupported".to_owned());
    }
    let version = c.u32()?;
    if !(2..=3).contains(&version) {
        return Err(format!("unsupported GGUF version {version}"));
    }
    let tensor_count_raw = c.u64()?;
    let kv_count_raw = c.u64()?;
    let tensor_count = checked_count(tensor_count_raw, 1_000_000, "tensor count")?;
    let kv_count = checked_count(kv_count_raw, 1_000_000, "metadata count")?;

    let mut metadata = Map::new();
    metadata.insert("GGUF.version".to_owned(), Value::from(version));
    metadata.insert(
        "GGUF.tensor_count".to_owned(),
        Value::from(tensor_count_raw),
    );
    metadata.insert("GGUF.kv_count".to_owned(), Value::from(kv_count_raw));
    for _ in 0..kv_count {
        let key = c.string()?;
        let kind = c.u32()?;
        let item = value(&mut c, kind).map_err(|e| format!("metadata {key:?}: {e}"))?;
        if metadata.insert(key.clone(), item).is_some() {
            return Err(format!("duplicate GGUF metadata key {key:?}"));
        }
    }

    let mut tensor_names = HashSet::with_capacity(tensor_count);
    let mut tensors = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = c.string()?;
        if !tensor_names.insert(name.clone()) {
            return Err(format!("duplicate GGUF tensor {name:?}"));
        }
        let dimensions_count = c.u32()?;
        if dimensions_count > 4 {
            return Err(format!(
                "tensor {name:?} has unsupported {dimensions_count} dimensions"
            ));
        }
        let mut dimensions = Vec::with_capacity(dimensions_count as usize);
        for _ in 0..dimensions_count {
            dimensions.push(c.u64()?);
        }
        let type_id = c.u32()?;
        let relative_offset = c.u64()?;
        tensors.push((name, dimensions, type_id, relative_offset));
    }

    let alignment = match metadata.get("general.alignment") {
        Some(Value::Number(n)) => n
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| "general.alignment must be an unsigned integer".to_owned())?,
        Some(_) => return Err("general.alignment must be an unsigned integer".to_owned()),
        None => 32,
    };
    let data_offset = align_up(c.offset, alignment)?;
    if data_offset > data.len() {
        return Err("GGUF tensor data starts past end of file".to_owned());
    }
    let data_offset_u64 = data_offset as u64;
    let mut result_tensors = Vec::with_capacity(tensors.len());
    for (name, dimensions, type_id, relative_offset) in tensors {
        let absolute = data_offset_u64
            .checked_add(relative_offset)
            .ok_or_else(|| format!("tensor {name:?} data offset overflow"))?;
        // the native EXL2 / EXL3 blobs carry their own length in the header (it depends on the bit widths of the matrix)
        let payload_len = if exl::is_exl_type(type_id) {
            let at = usize::try_from(absolute).map_err(|_| format!("tensor {name:?} offset does not fit usize"))?;
            exl::payload_len(data.get(at..).unwrap_or(&[]), type_id).map_err(|e| format!("tensor {name:?}: {e}"))?
        } else {
            tensor_payload_len(type_id, &dimensions)?
        };
        let end = absolute
            .checked_add(payload_len)
            .ok_or_else(|| format!("tensor {name:?} payload range overflow"))?;
        if end > data.len() as u64 {
            return Err(format!(
                "tensor {name:?} payload range {absolute}..{end} is outside the file"
            ));
        }
        result_tensors.push(TensorInfo {
            name,
            dimensions,
            type_id,
            data_offset: absolute,
            payload_len,
        });
    }

    Ok(GgufIndex {
        version,
        metadata,
        tensors: result_tensors,
        data_offset: data_offset_u64,
        file_len: data.len() as u64,
    })
}

fn tensor_slice<'a>(data: &'a [u8], tensor: &TensorInfo) -> Result<&'a [u8], String> {
    let start = usize::try_from(tensor.data_offset)
        .map_err(|_| format!("tensor {:?} offset does not fit usize", tensor.name))?;
    let len = usize::try_from(tensor.payload_len)
        .map_err(|_| format!("tensor {:?} length does not fit usize", tensor.name))?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| format!("tensor {:?} byte range overflow", tensor.name))?;
    data.get(start..end)
        .ok_or_else(|| format!("tensor {:?} byte range is outside the mapping", tensor.name))
}

/// Read-only, zero-copy GGUF mapping with validated tensor byte ranges.
pub struct MappedGguf {
    mapping: Arc<Mmap>,
    index: GgufIndex,
    extra_mappings: HashMap<String, Arc<Mmap>>,
}

impl MappedGguf {
    /// Make a private tensor-name view sharing every mapped payload, including
    /// split-file mappings. Entries are `(alias, existing_name)`; collisions and
    /// missing sources are errors. The original on-disk index remains unchanged.
    pub fn with_tensor_aliases(&self, aliases: &[(String, String)]) -> Result<Self, String> {
        let mut index = self.index.clone();
        let mut extra_mappings = self.extra_mappings.clone();
        let mut names: HashSet<_> = index.tensors.iter().map(|t| t.name.clone()).collect();
        for (alias, source) in aliases {
            if !names.insert(alias.clone()) {
                return Err(format!("duplicate GGUF tensor alias {alias}"));
            }
            let mut info = self
                .index
                .tensors
                .iter()
                .find(|t| &t.name == source)
                .cloned()
                .ok_or_else(|| format!("missing GGUF alias source {source}"))?;
            info.name.clone_from(alias);
            index.tensors.push(info);
            extra_mappings.insert(
                alias.clone(),
                Arc::clone(self.extra_mappings.get(source).unwrap_or(&self.mapping)),
            );
        }
        Ok(Self {
            mapping: Arc::clone(&self.mapping),
            index,
            extra_mappings,
        })
    }
    /// A view of a file written for architecture `from` as one of architecture `to`: `general.architecture` and
    /// every `<from>.` metadata key are renamed, tensors and payloads are shared. For llama.cpp architecture names
    /// whose layout equals one this program implements under another name.
    pub fn with_architecture_alias(&self, from: &str, to: &str) -> Self {
        let mut index = self.index.clone();
        let prefix = format!("{from}.");
        let renamed: Vec<(String, Value)> = index
            .metadata
            .iter()
            .filter_map(|(k, v)| k.strip_prefix(&prefix).map(|rest| (format!("{to}.{rest}"), v.clone())))
            .collect();
        index.metadata.retain(|k, _| !k.starts_with(&prefix));
        index.metadata.extend(renamed);
        index.metadata.insert("general.architecture".into(), Value::String(to.into()));
        Self {
            mapping: Arc::clone(&self.mapping),
            index,
            extra_mappings: self.extra_mappings.clone(),
        }
    }

    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        let mut model = Self::open_single(path)?;
        fn invalid(e: impl Into<String>) -> io::Error {
            io::Error::new(io::ErrorKind::InvalidData, e.into())
        }
        let meta = &model.index.metadata;
        if !meta.contains_key("split.count") {
            if meta.contains_key("split.no") || meta.contains_key("split.tensors.count") {
                return Err(invalid("incomplete GGUF split metadata"));
            }
            return Ok(model);
        }
        let count = meta
            .get("split.count")
            .and_then(Value::as_u64)
            .filter(|&n| n > 0 && n <= 99999)
            .ok_or_else(|| invalid("invalid split.count"))?;
        if meta.get("split.no").and_then(Value::as_u64) != Some(0) {
            return Err(invalid("open the first GGUF shard (split.no = 0)"));
        }
        let total = meta
            .get("split.tensors.count")
            .and_then(Value::as_u64)
            .filter(|&n| n <= 1_000_000)
            .ok_or_else(|| invalid("invalid split.tensors.count"))?;
        let filename = path
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| invalid("invalid GGUF shard filename"))?;
        let suffix = format!("-00001-of-{count:05}.gguf");
        let prefix = filename
            .strip_suffix(&suffix)
            .ok_or_else(|| invalid("GGUF shard filename does not match split.count"))?;
        let mut names: HashSet<String> =
            model.index.tensors.iter().map(|t| t.name.clone()).collect();
        for no in 1..count {
            let shard_path =
                path.with_file_name(format!("{prefix}-{:05}-of-{count:05}.gguf", no + 1));
            let shard = Self::open_single(&shard_path)?;
            if shard.index.version != model.index.version
                || shard.index.metadata.get("split.no").and_then(Value::as_u64) != Some(no)
                || shard
                    .index
                    .metadata
                    .get("split.count")
                    .and_then(Value::as_u64)
                    != Some(count)
                || shard
                    .index
                    .metadata
                    .get("split.tensors.count")
                    .and_then(Value::as_u64)
                    != Some(total)
            {
                return Err(invalid(format!(
                    "inconsistent split metadata in {}",
                    shard_path.display()
                )));
            }
            // llama.cpp may put tokenizer metadata in only the first shard.
            for (key, value) in &shard.index.metadata {
                if key.starts_with("split.") || key.starts_with("GGUF.") {
                    continue;
                }
                if model.index.metadata.get(key).is_some_and(|v| v != value) {
                    return Err(invalid(format!("conflicting shard metadata {key}")));
                }
                model
                    .index
                    .metadata
                    .entry(key.clone())
                    .or_insert_with(|| value.clone());
            }
            model.index.file_len = model
                .index
                .file_len
                .checked_add(shard.index.file_len)
                .ok_or_else(|| invalid("GGUF shard file size overflow"))?;
            for tensor in shard.index.tensors {
                if !names.insert(tensor.name.clone()) {
                    return Err(invalid(format!(
                        "duplicate tensor across shards: {}",
                        tensor.name
                    )));
                }
                model
                    .extra_mappings
                    .insert(tensor.name.clone(), Arc::clone(&shard.mapping));
                model.index.tensors.push(tensor);
            }
        }
        if model.index.tensors.len() as u64 != total {
            return Err(invalid(
                "GGUF split tensor count does not match split.tensors.count",
            ));
        }
        model
            .index
            .metadata
            .insert("GGUF.tensor_count".into(), Value::from(total));
        Ok(model)
    }

    /// Map one physical GGUF file without following its split metadata.
    /// Create a MappedGguf directly from an in-memory index and per-tensor mappings.
    pub fn from_index_and_mappings(
        index: GgufIndex,
        extra_mappings: HashMap<String, Arc<Mmap>>,
    ) -> io::Result<Self> {
        let mapping = if let Some(first) = extra_mappings.values().next() {
            Arc::clone(first)
        } else {
            let anon = MmapOptions::new().len(1).map_anon()?;
            Arc::new(anon.make_read_only()?)
        };
        Ok(Self {
            mapping,
            index,
            extra_mappings,
        })
    }

    pub fn open_single(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::open(path)?;
        // SAFETY: the mapping is read-only and owned for as long as tensor slices are returned.
        let mapping = unsafe { MmapOptions::new().map(&file)? };
        let index = parse(&mapping).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(Self {
            mapping: Arc::new(mapping),
            index,
            extra_mappings: HashMap::new(),
        })
    }

    pub fn index(&self) -> &GgufIndex {
        &self.index
    }

    pub fn extra_mappings(&self) -> &HashMap<String, Arc<Mmap>> {
        &self.extra_mappings
    }

    pub fn mapping(&self) -> &Arc<Mmap> {
        &self.mapping
    }

    /// Return the on-disk bytes for a named tensor without copying them.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], String> {
        let tensor = self
            .index
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .ok_or_else(|| format!("tensor {name:?} not found"))?;
        let mapping = self.extra_mappings.get(name).unwrap_or(&self.mapping);
        tensor_slice(mapping, tensor)
    }
    /// Shared, zero-copy tensor storage that keeps the read-only map alive.
    pub fn owned_tensor_bytes(&self, name: &str) -> Result<TensorBytes, String> {
        let tensor = self
            .index
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("tensor {name:?} not found"))?;
        let mapping = self.extra_mappings.get(name).unwrap_or(&self.mapping);
        tensor_slice(mapping, tensor)?;
        let start = tensor.data_offset as usize;
        Ok(TensorBytes {
            inner: TensorBytesInner::Mmap(Arc::clone(mapping), start..start + tensor.payload_len as usize),
        })
    }
}

#[derive(Clone)]
enum TensorBytesInner {
    Mmap(Arc<Mmap>, std::ops::Range<usize>),
    Owned(Arc<Vec<u8>>, std::ops::Range<usize>),
}

#[derive(Clone)]
pub struct TensorBytes {
    inner: TensorBytesInner,
}
impl TensorBytes {
    /// The bytes are a view of a memory-mapped file (or other mapping), not a heap copy: reading them may fault pages in from storage, and
    /// the operating system can drop them again under memory pressure.
    pub fn is_mapped(&self) -> bool {
        matches!(self.inner, TensorBytesInner::Mmap(..))
    }

    pub fn from_vec(v: Vec<u8>) -> Self {
        let len = v.len();
        Self {
            inner: TensorBytesInner::Owned(Arc::new(v), 0..len),
        }
    }

    /// A zero-copy view of `len` bytes starting `start` bytes into this tensor.
    pub fn slice(&self, start: usize, len: usize) -> Result<TensorBytes, String> {
        let cur_len = self.len();
        let end = start
            .checked_add(len)
            .filter(|&end| end <= cur_len)
            .ok_or("tensor slice exceeds the tensor")?;
        let inner = match &self.inner {
            TensorBytesInner::Mmap(m, r) => {
                TensorBytesInner::Mmap(Arc::clone(m), r.start + start..r.start + end)
            }
            TensorBytesInner::Owned(o, r) => {
                TensorBytesInner::Owned(Arc::clone(o), r.start + start..r.start + end)
            }
        };
        Ok(TensorBytes { inner })
    }
}
impl std::ops::Deref for TensorBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match &self.inner {
            TensorBytesInner::Mmap(m, r) => &m[r.clone()],
            TensorBytesInner::Owned(o, r) => &o[r.clone()],
        }
    }
}
impl AsRef<[u8]> for TensorBytes {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8b_type_ids_round_trip_and_describe_the_tile() {
        assert_eq!(fp8b_shape(0x8000_0021), Some((128, 128)));
        assert_eq!(fp8b_type_id(128, 128), Some(0x8000_0021));
        for (rows, cols) in [(1, 1), (1, 128), (16, 16), (32, 32), (64, 8), (128, 32), (256, 64), (32768, 32768)] {
            let id = fp8b_type_id(rows, cols).unwrap();
            assert_eq!(fp8b_shape(id), Some((rows, cols)), "{rows}x{cols}");
            assert!(fp8b_type(id));
        }
        assert_eq!(fp8b_type_id(48, 48), None, "not a power of two");
        assert_eq!(fp8b_type_id(65536, 1), None, "tile above 32768");
        assert_eq!(fp8b_type_id(0, 1), None);
        // neighbours in the id space are not FP8B types
        for id in [0x8000_0020, 0x8000_0022, 0x8000_00ff, 0x8000_0200, 0x8000_0030, 0x8001_0000, 40] {
            assert!(!fp8b_type(id), "{id:#x}");
        }
    }

    #[test]
    fn fp8b_payload_length_counts_codes_scales_and_trailer() {
        // GGUF dimensions are [columns, rows, matrices...]
        let id = fp8b_type_id(128, 128).unwrap();
        assert_eq!(tensor_payload_len(id, &[130, 129]).unwrap(), 130 * 129 + 2 * 2 * 4 + 16);
        assert_eq!(tensor_payload_len(id, &[64, 128, 6]).unwrap(), 6 * 64 * 128 + 6 * 4 + 16, "six experts of one partial tile each");
        let small = fp8b_type_id(32, 16).unwrap();
        assert_eq!(tensor_payload_len(small, &[40, 70, 3]).unwrap(), 3 * 40 * 70 + 3 * 3 * 3 * 4 + 16);
        let layout = Fp8bLayout::from_dims(small, &[40, 70, 3]).unwrap();
        assert_eq!((layout.tile_cols(), layout.tiles_per_slab(), layout.tiles(), layout.scale_offset()), (3, 9, 27, 8400));
    }

    #[test]
    fn fp8b_trailer_round_trips_and_rejects_other_bytes() {
        let id = fp8b_type_id(8, 4).unwrap();
        let layout = Fp8bLayout::new(id, 10, 9, 2).unwrap();
        let mut bytes = vec![0u8; layout.total_len() - FP8B_TRAILER];
        bytes.extend_from_slice(&layout.trailer());
        assert_eq!(Fp8bLayout::parse(id, &bytes).unwrap(), layout);
        // the length has to agree with the geometry in the trailer, the type has to be an FP8B type
        assert!(Fp8bLayout::parse(id, &bytes[1..]).is_err());
        assert!(Fp8bLayout::parse(8, &bytes).is_err());
        let mut other = bytes.clone();
        let at = other.len() - FP8B_TRAILER;
        other[at] = b'X';
        assert!(Fp8bLayout::parse(id, &other).is_err());
        let mut wrong_shape = bytes;
        let n = wrong_shape.len();
        wrong_shape[n - 4..].copy_from_slice(&3u32.to_le_bytes());
        assert!(Fp8bLayout::parse(id, &wrong_shape).is_err(), "matrices 3 instead of 2");
    }

    fn put_u32(out: &mut Vec<u8>, v: u32) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    fn put_u64(out: &mut Vec<u8>, v: u64) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    fn put_str(out: &mut Vec<u8>, v: &str) {
        put_u64(out, v.len() as u64);
        out.extend_from_slice(v.as_bytes());
    }

    #[test]
    fn parses_metadata_arrays_and_validates_tensor_range() {
        let mut bytes = b"GGUF".to_vec();
        put_u32(&mut bytes, 3);
        put_u64(&mut bytes, 1);
        put_u64(&mut bytes, 2);
        put_str(&mut bytes, "general.alignment");
        put_u32(&mut bytes, 4);
        put_u32(&mut bytes, 32);
        put_str(&mut bytes, "tokenizer.ggml.tokens");
        put_u32(&mut bytes, 9);
        put_u32(&mut bytes, 8);
        put_u64(&mut bytes, 2);
        put_str(&mut bytes, "a");
        put_str(&mut bytes, "b");
        put_str(&mut bytes, "weight");
        put_u32(&mut bytes, 2);
        put_u64(&mut bytes, 2);
        put_u64(&mut bytes, 3);
        put_u32(&mut bytes, 1);
        put_u64(&mut bytes, 0);
        let data_offset = (bytes.len() + 31) & !31;
        bytes.resize(data_offset, 0);
        bytes.extend_from_slice(&[42; 12]);

        let parsed = parse(&bytes).unwrap();
        assert_eq!(
            parsed.metadata["tokenizer.ggml.tokens"],
            serde_json::json!(["a", "b"])
        );
        assert_eq!(parsed.tensors[0].payload_len, 12);
        assert_eq!(tensor_slice(&bytes, &parsed.tensors[0]).unwrap(), &[42; 12]);
        let mut anonymous = MmapOptions::new().len(bytes.len()).map_anon().unwrap();
        anonymous.copy_from_slice(&bytes);
        let mapped = MappedGguf {
            mapping: Arc::new(anonymous.make_read_only().unwrap()),
            index: parsed,
            extra_mappings: HashMap::new(),
        };
        let owned = mapped.owned_tensor_bytes("weight").unwrap();
        assert_eq!(
            owned.as_ptr(),
            mapped.tensor_bytes("weight").unwrap().as_ptr()
        );
        let aliases = [("nextn.weight".into(), "weight".into())];
        let view = mapped.with_tensor_aliases(&aliases).unwrap();
        assert_eq!(
            view.tensor_bytes("nextn.weight").unwrap().as_ptr(),
            owned.as_ptr()
        );
        assert!(view.with_tensor_aliases(&aliases).is_err());
        assert!(mapped
            .with_tensor_aliases(&[("missing_alias".into(), "missing".into())])
            .is_err());
        drop(mapped);
        assert_eq!(&*owned, &[42; 12]);
        let mut truncated = bytes.clone();
        truncated.pop();
        assert!(parse(&truncated).unwrap_err().contains("payload range"));
    }

    #[test]
    fn architecture_alias_renames_the_key_prefix() {
        let mut bytes = b"GGUF".to_vec();
        put_u32(&mut bytes, 3);
        put_u64(&mut bytes, 0);
        put_u64(&mut bytes, 3);
        put_str(&mut bytes, "general.architecture");
        put_u32(&mut bytes, 8);
        put_str(&mut bytes, "exaone");
        put_str(&mut bytes, "exaone.block_count");
        put_u32(&mut bytes, 4);
        put_u32(&mut bytes, 7);
        put_str(&mut bytes, "tokenizer.ggml.model");
        put_u32(&mut bytes, 8);
        put_str(&mut bytes, "gpt2");
        bytes.resize((bytes.len() + 31) & !31, 0);
        let parsed = parse(&bytes).unwrap();
        let mut anonymous = MmapOptions::new().len(bytes.len()).map_anon().unwrap();
        anonymous.copy_from_slice(&bytes);
        let mapped = MappedGguf { mapping: Arc::new(anonymous.make_read_only().unwrap()), index: parsed, extra_mappings: HashMap::new() };
        let view = mapped.with_architecture_alias("exaone", "llama");
        let meta = &view.index().metadata;
        assert_eq!(meta["general.architecture"], "llama");
        assert_eq!(meta["llama.block_count"], 7);
        assert!(!meta.contains_key("exaone.block_count"));
        assert_eq!(meta["tokenizer.ggml.model"], "gpt2");
        // the original view is untouched
        assert_eq!(mapped.index().metadata["general.architecture"], "exaone");
    }

    #[test]
    fn mlx_affine_type_ids_round_trip_and_size_the_payload() {
        for bits in [2u32, 3, 4, 5, 6, 8] {
            for group in [32u32, 64, 96, 128, 256, 8160] {
                for bf16 in [false, true] {
                    let m = MlxAffine { bits, group, bf16 };
                    assert_eq!(mlx_affine(m.type_id()), Some(m));
                    assert!(mlx_affine_supported(bits, group));
                    // a row of two groups
                    let cols = u64::from(group) * 2;
                    assert_eq!(tensor_payload_len(m.type_id(), &[cols, 3]).unwrap(), 3 * 2 * (u64::from(group) * u64::from(bits) / 8 + 4));
                }
            }
        }
        for (bits, group) in [(1u32, 64u32), (7, 64), (9, 64), (4, 0), (4, 48), (4, 32 * 256)] {
            assert!(!mlx_affine_supported(bits, group), "{bits} bits, group {group}");
        }
        // ids outside the family and ids with unsupported fields are not MLX types
        for id in [0x8000_0010u32, 0x8001_2000, 0x8001_0000, 0x8001_0017, 0x8001_0004 + 0x20_0000] {
            assert_eq!(mlx_affine(id), None, "{id:#x}");
        }
        assert!(tensor_payload_len(0x8001_0024, &[96, 1]).is_err(), "row length 96 is not a multiple of the 64 group");
    }

    #[test]
    fn rejects_truncated_and_nested_array_metadata() {
        assert!(parse(b"GGUF").unwrap_err().contains("truncated"));
        let mut bytes = b"GGUF".to_vec();
        put_u32(&mut bytes, 3);
        put_u64(&mut bytes, 0);
        put_u64(&mut bytes, 1);
        put_str(&mut bytes, "bad");
        put_u32(&mut bytes, 9);
        put_u32(&mut bytes, 9);
        put_u64(&mut bytes, 0);
        assert!(parse(&bytes).unwrap_err().contains("nested"));
    }
}
