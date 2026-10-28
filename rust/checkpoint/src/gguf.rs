use crate::{packed_quant::PackedData, safetensors::Payload};
use std::io::{self, Write};
use std::sync::Arc;

#[derive(Clone)]
pub enum Meta {
    U32(u32),
    F32(f32),
    Bool(bool),
    Str(String),
    Strs(Vec<String>),
    U32s(Vec<u32>),
    F32s(Vec<f32>),
    Bools(Vec<bool>),
}
pub struct OutTensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub source_dtype: String,
    pub dtype: String,
    pub data: Payload,
    pub packed: Option<Arc<PackedData>>,
    pub data_len: u64,
    pub source_for_output: Option<Vec<usize>>,
    /// Constant added to every element while converting (Gemma stores `1 + weight` for its norms).
    pub add: f32,
    /// Tensor data built in memory (little-endian, already in `dtype`); `data` is ignored.
    pub owned: Option<Vec<u8>>,
    /// Further source tensors appended after `data` (stacked experts); same dtype and layout.
    pub parts: Vec<Payload>,
}
fn put_str(w: &mut impl Write, s: &str) -> io::Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())
}
fn write_meta(w: &mut impl Write, key: &str, v: &Meta) -> io::Result<()> {
    put_str(w, key)?;
    match v {
        Meta::U32(x) => {
            w.write_all(&4u32.to_le_bytes())?;
            w.write_all(&x.to_le_bytes())?;
        }
        Meta::F32(x) => {
            w.write_all(&6u32.to_le_bytes())?;
            w.write_all(&x.to_le_bytes())?;
        }
        Meta::Bool(x) => {
            w.write_all(&7u32.to_le_bytes())?;
            w.write_all(&[*x as u8])?;
        }
        Meta::Str(x) => {
            w.write_all(&8u32.to_le_bytes())?;
            put_str(w, x)?;
        }
        Meta::Strs(xs) => {
            w.write_all(&9u32.to_le_bytes())?;
            w.write_all(&8u32.to_le_bytes())?;
            w.write_all(&(xs.len() as u64).to_le_bytes())?;
            for x in xs {
                put_str(w, x)?;
            }
        }
        Meta::U32s(xs) => {
            w.write_all(&9u32.to_le_bytes())?;
            w.write_all(&4u32.to_le_bytes())?;
            w.write_all(&(xs.len() as u64).to_le_bytes())?;
            for x in xs {
                w.write_all(&x.to_le_bytes())?;
            }
        }
        Meta::F32s(xs) => {
            w.write_all(&9u32.to_le_bytes())?;
            w.write_all(&6u32.to_le_bytes())?;
            w.write_all(&(xs.len() as u64).to_le_bytes())?;
            for x in xs {
                w.write_all(&x.to_le_bytes())?;
            }
        }
        Meta::Bools(xs) => {
            w.write_all(&9u32.to_le_bytes())?;
            w.write_all(&7u32.to_le_bytes())?;
            w.write_all(&(xs.len() as u64).to_le_bytes())?;
            for x in xs {
                w.write_all(&[*x as u8])?;
            }
        }
    }
    Ok(())
}
pub fn write_to(
    mut w: impl Write,
    metadata: &[(String, Meta)],
    tensors: &[OutTensor],
) -> io::Result<u64> {
    let mut header = Vec::new();
    header.write_all(b"GGUF")?;
    header.write_all(&3u32.to_le_bytes())?;
    header.write_all(&(tensors.len() as u64).to_le_bytes())?;
    header.write_all(&(metadata.len() as u64).to_le_bytes())?;
    for (k, v) in metadata {
        write_meta(&mut header, k, v)?;
    }
    let mut rel = 0u64;
    for t in tensors {
        put_str(&mut header, &t.name)?;
        header.write_all(&(t.dims.len() as u32).to_le_bytes())?;
        for d in &t.dims {
            header.write_all(&d.to_le_bytes())?;
        }
        let ty = match t.dtype.as_str() {
            "F32" => 0u32,
            "F16" => 1,
            "BF16" => 30,
            "MXFP4" => 39,
            "NVFP4" => 40,
            "Q8_0" => 8,
            "Q4Z32" => 0x8000_0010,
            "Q4Z64" => 0x8000_0011,
            "Q4Z128" => 0x8000_0012,
            "Q4ZP32" => 0x8000_0014,
            "Q4ZP64" => 0x8000_0015,
            "Q4ZP128" => 0x8000_0016,
            "FP8M" | "F8M" => 0x8000_0020,
            other => match crate::packed_quant::fp8b_dtype_id(other) {
                Some(id) => id,
                None => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "unsupported GGUF tensor dtype",
                    ))
                }
            },
        };
        header.write_all(&ty.to_le_bytes())?;
        header.write_all(&rel.to_le_bytes())?;
        rel = rel.checked_add(t.data_len).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "GGUF tensor offset overflow")
        })?;
        rel = rel.checked_add(31).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "GGUF tensor alignment overflow",
            )
        })? & !31;
    }
    while header.len() % 32 != 0 {
        header.push(0);
    }
    w.write_all(&header)?;
    let mut written = header.len() as u64;
    for t in tensors {
        let pad = (32 - (written % 32)) % 32;
        if pad > 0 {
            w.write_all(&vec![0; pad as usize])?;
            written += pad;
        }
        let emitted = write_tensor(&mut w, t)?;
        if emitted != t.data_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "tensor {} wrote {emitted} bytes but expected {}",
                    t.name, t.data_len
                ),
            ));
        }
        written = written.checked_add(emitted).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "GGUF file length overflow")
        })?;
    }
    Ok(written)
}
pub(crate) fn type_width(t: &str) -> io::Result<usize> {
    match t {
        "F32" => Ok(4),
        "F16" | "BF16" => Ok(2),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported source tensor dtype",
        )),
    }
}
pub(crate) fn write_tensor(w: &mut impl Write, t: &OutTensor) -> io::Result<u64> {
    if let Some(packed) = &t.packed {
        return write_packed(w, t, packed);
    }
    if let Some(bytes) = &t.owned {
        w.write_all(bytes)?;
        return Ok(bytes.len() as u64);
    }
    let mut total = write_bytes(w, t, t.data.as_slice(), t.source_for_output.as_deref())?;
    for part in &t.parts {
        total += write_bytes(w, t, part.as_slice(), None)?;
    }
    Ok(total)
}
fn write_bytes(
    w: &mut impl Write,
    t: &OutTensor,
    src: &[u8],
    order: Option<&[usize]>,
) -> io::Result<u64> {
    let sw = type_width(&t.source_dtype)?;
    if !src.len().is_multiple_of(sw) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "source tensor has incomplete scalar",
        ));
    }
    let convert = t.source_dtype != t.dtype || t.add != 0.0;
    if let Some(order) = order {
        let rows = order.len();
        if rows == 0 || !src.len().is_multiple_of(rows) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid tensor row permutation",
            ));
        }
        let row_src = src.len() / rows;
        if !row_src.is_multiple_of(sw) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "permuted tensor row has incomplete scalar",
            ));
        }
        let out_row = row_src * type_width(&t.dtype)? / sw;
        let mut batch = Vec::with_capacity(64 * 1024 + out_row);
        for &source_row in order {
            let start = source_row
                .checked_mul(row_src)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "row offset overflow"))?;
            let end = start
                .checked_add(row_src)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "row range overflow"))?;
            let row = src.get(start..end).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "row source index outside tensor",
                )
            })?;
            if !convert {
                batch.extend_from_slice(row);
            } else {
                for scalar in row.chunks_exact(sw) {
                    let v = to_f32(&t.source_dtype, scalar)? + t.add;
                    append_converted(&mut batch, &t.dtype, v)?;
                }
            }
            if batch.len() >= 64 * 1024 {
                w.write_all(&batch)?;
                batch.clear();
            }
        }
        if !batch.is_empty() {
            w.write_all(&batch)?;
        }
        return Ok((rows * out_row) as u64);
    }
    if !convert {
        w.write_all(src)?;
        return Ok(src.len() as u64);
    }
    let outw = type_width(&t.dtype)?;
    let elems = src.len() / sw;
    let mut output = Vec::with_capacity(16 * 1024 * outw);
    for chunk in src.chunks((16 * 1024 * sw).max(sw)) {
        output.clear();
        for scalar in chunk.chunks_exact(sw) {
            let v = to_f32(&t.source_dtype, scalar)? + t.add;
            append_converted(&mut output, &t.dtype, v)?;
        }
        w.write_all(&output)?;
    }
    Ok((elems * outw) as u64)
}
fn write_packed_fp8m(w: &mut impl Write, t: &OutTensor, p: &PackedData) -> io::Result<u64> {
    let (weight, scales, rows, cols) = p.fp8m_info().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "tensor cannot be stored as FP8M")
    })?;
    let bytes = weight.as_slice();
    if bytes.len() != rows * cols || scales.len() != rows {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "FP8M dimensions mismatch"));
    }
    if let Some(order) = &t.source_for_output {
        if order.len() != rows {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid packed row permutation"));
        }
        for &src in order {
            let row = &bytes[src * cols..(src + 1) * cols];
            w.write_all(row)?;
        }
        for &src in order {
            w.write_all(&scales[src].to_le_bytes())?;
        }
    } else {
        w.write_all(bytes)?;
        for &s in scales {
            w.write_all(&s.to_le_bytes())?;
        }
    }
    let total = bytes.len() + scales.len() * 2;
    if total as u64 != t.data_len {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "FP8M length mismatch"));
    }
    Ok(total as u64)
}

/// Block-wise FP8 (`FP8B`): the E4M3 codes of every matrix, then the f32 tile scales, then the trailer (layout: `fllm_gguf_probe::Fp8bLayout`).
/// A row permutation is applied to the code rows and, tile row by tile row, to the scales (`fp8b_tile_row_map`).
fn write_packed_fp8b(w: &mut impl Write, t: &OutTensor, p: &PackedData) -> io::Result<u64> {
    let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());
    let id = crate::packed_quant::fp8b_dtype_id(&t.dtype).ok_or_else(|| bad("unknown FP8B dtype"))?;
    let layout = fllm_gguf_probe::Fp8bLayout::from_dims(id, &t.dims).ok_or_else(|| bad("invalid FP8B dimensions"))?;
    if layout.total_len() as u64 != t.data_len {
        return Err(bad("FP8B length mismatch"));
    }
    let blocks: Vec<&crate::symint::Fp8Block> = match p {
        PackedData::Fp8Block(b) => vec![b],
        PackedData::Concat { parts, .. } => parts
            .iter()
            .map(|part| part.fp8_block().ok_or_else(|| bad("non-FP8B part in stacked FP8B tensor")))
            .collect::<io::Result<_>>()?,
        _ => return Err(bad("tensor cannot be stored as FP8B")),
    };
    if blocks.len() != layout.slabs
        || blocks.iter().any(|b| {
            (b.rows(), b.cols(), b.block_rows(), b.block_cols()) != (layout.rows, layout.cols, layout.block_rows, layout.block_cols)
        })
    {
        return Err(bad("FP8B matrices do not match the tensor dimensions"));
    }
    let tile_map = match t.source_for_output.as_deref() {
        Some(order) if layout.slabs == 1 => Some(
            crate::packed_quant::fp8b_tile_row_map(order, layout.rows, layout.block_rows)
                .ok_or_else(|| bad("row permutation of an FP8B tensor cuts through tile rows"))?,
        ),
        Some(_) => return Err(bad("row permutation of stacked FP8B tensors")),
        None => None,
    };
    for b in &blocks {
        match t.source_for_output.as_deref() {
            Some(order) => {
                for &src in order {
                    w.write_all(&b.weight_bytes()[src * layout.cols..(src + 1) * layout.cols])?;
                }
            }
            None => w.write_all(b.weight_bytes())?,
        }
    }
    let tile_cols = layout.tile_cols();
    let mut scales = Vec::with_capacity(layout.tiles_per_slab() * 4);
    for b in &blocks {
        scales.clear();
        match &tile_map {
            Some(map) => {
                for &src in map {
                    for &s in &b.scales()[src * tile_cols..(src + 1) * tile_cols] {
                        scales.extend_from_slice(&s.to_le_bytes());
                    }
                }
            }
            None => {
                for &s in b.scales() {
                    scales.extend_from_slice(&s.to_le_bytes());
                }
            }
        }
        w.write_all(&scales)?;
    }
    w.write_all(&layout.trailer())?;
    Ok(layout.total_len() as u64)
}

fn write_packed(w: &mut impl Write, t: &OutTensor, p: &PackedData) -> io::Result<u64> {
    if t.dtype == "FP8M" || t.dtype == "F8M" {
        return write_packed_fp8m(w, t, p);
    }
    if crate::packed_quant::fp8b_dtype_id(&t.dtype).is_some() {
        return write_packed_fp8b(w, t, p);
    }
    if t.dtype == "NVFP4" {
        return write_packed_rows(w, t, |row, buf| p.encode_nvfp4_row(row, buf));
    }
    if t.dtype == "MXFP4" {
        return write_packed_rows(w, t, |row, buf| {
            p.mxfp4()
                .ok_or_else(|| "tensor cannot be stored as MXFP4".to_owned())?
                .encode_gguf_row(row, buf)
        });
    }
    if t.dtype == "Q8_0" {
        return write_packed_rows(w, t, |row, buf| {
            p.sym_int()
                .ok_or_else(|| "tensor cannot be stored as Q8_0".to_owned())?
                .encode_q8_0_row(row, buf)
        });
    }
    if t.dtype.starts_with("Q4ZP") {
        return write_packed_q4zp(w, t, p);
    }
    if t.dtype.starts_with("Q4Z") {
        return write_packed_q4z(w, t, p);
    }
    let count = p.element_count();
    let width = type_width(&t.dtype)?;
    let expected = (count as u64).checked_mul(width as u64).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "packed output byte size overflow",
        )
    })?;
    if expected != t.data_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "packed tensor output length mismatch",
        ));
    }
    let out_row = if let Some(order) = &t.source_for_output {
        let n = order.len();
        if n == 0 || !count.is_multiple_of(n) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid packed row permutation",
            ));
        }
        count / n
    } else {
        0
    };
    if let Some(order) = &t.source_for_output {
        let mut values = vec![0f32; out_row];
        let mut bytes = Vec::with_capacity(out_row * width);
        for &row in order {
            let start = row.checked_mul(out_row).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "packed row offset overflow")
            })?;
            p.decode_f32_chunk(start, &mut values)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            bytes.clear();
            for &v in &values {
                append_converted(&mut bytes, &t.dtype, v)?;
            }
            w.write_all(&bytes)?;
        }
    } else {
        let chunk = 16 * 1024;
        let mut values = vec![0f32; chunk.min(count.max(1))];
        let mut bytes = Vec::with_capacity(values.len() * width);
        for start in (0..count).step_by(chunk) {
            let n = chunk.min(count - start);
            let values = &mut values[..n];
            p.decode_f32_chunk(start, values)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            bytes.clear();
            for &v in values.iter() {
                append_converted(&mut bytes, &t.dtype, v)?;
            }
            w.write_all(&bytes)?;
        }
    }
    Ok(expected)
}
/// Row-wise encoders (`NVFP4`, `Q8_0`): `encode(source_row, buf)` appends one output row, rows are reordered like the dense path.
fn write_packed_rows(
    w: &mut impl Write,
    t: &OutTensor,
    encode: impl Fn(usize, &mut Vec<u8>) -> Result<(), String>,
) -> io::Result<u64> {
    let bad = |m: String| io::Error::new(io::ErrorKind::InvalidData, m);
    // (stacked experts have a third dimension: all rows of all experts are written one after the other)
    let rows = t.dims.get(1..).filter(|d| !d.is_empty()).ok_or_else(|| bad(format!("{} tensor needs two dimensions", t.dtype)))?.iter().product::<u64>() as usize;
    let mut buf = Vec::new();
    let mut total = 0u64;
    for i in 0..rows {
        let src = match &t.source_for_output {
            Some(order) => *order.get(i).ok_or_else(|| bad("invalid packed row permutation".into()))?,
            None => i,
        };
        buf.clear();
        encode(src, &mut buf).map_err(bad)?;
        w.write_all(&buf)?;
        total += buf.len() as u64;
    }
    if total != t.data_len {
        return Err(bad(format!("{} output length mismatch: wrote {total}, expected {}", t.dtype, t.data_len)));
    }
    Ok(total)
}

/// 4-bit GPTQ weights with act-order as native `Q4ZP` rows followed by column permutation indices.
fn write_packed_q4zp(w: &mut impl Write, t: &OutTensor, p: &PackedData) -> io::Result<u64> {
    let bad = |m: String| io::Error::new(io::ErrorKind::InvalidData, m);
    let group = p.q4zp_group().ok_or_else(|| bad("tensor cannot be stored as Q4ZP".into()))?;
    let perm = p.q4zp_perm().ok_or_else(|| bad("missing permutation table for Q4ZP".into()))?;
    let rows = *t.dims.get(1).ok_or_else(|| bad("Q4ZP tensor needs two dimensions".into()))? as usize;
    let row_bytes = p.element_count() / rows / group * (group / 2 + 3);
    let perm_bytes = perm.len() * 4;
    if (row_bytes * rows + perm_bytes) as u64 != t.data_len {
        return Err(bad("Q4ZP output length mismatch".into()));
    }
    let mut buf = Vec::with_capacity(row_bytes);
    for i in 0..rows {
        let src = match &t.source_for_output {
            Some(order) => *order.get(i).ok_or_else(|| bad("invalid packed row permutation".into()))?,
            None => i,
        };
        buf.clear();
        p.encode_q4zp_row(src, &perm, &mut buf).map_err(bad)?;
        w.write_all(&buf)?;
    }
    for &idx in &perm {
        w.write_all(&idx.to_le_bytes())?;
    }
    Ok(t.data_len)
}

/// 4-bit AWQ / GPTQ weights as native `Q4Z` rows, one output row after the other (rows reordered like the dense path).
fn write_packed_q4z(w: &mut impl Write, t: &OutTensor, p: &PackedData) -> io::Result<u64> {
    let bad = |m: String| io::Error::new(io::ErrorKind::InvalidData, m);
    let group = p.q4z_group().ok_or_else(|| bad("tensor cannot be stored as Q4Z".into()))?;
    let rows = t.dims.get(1..).filter(|d| !d.is_empty()).ok_or_else(|| bad("Q4Z tensor needs two dimensions".into()))?.iter().product::<u64>() as usize;
    let row_bytes = p.element_count() / rows / group * (group / 2 + 3);
    if (row_bytes * rows) as u64 != t.data_len {
        return Err(bad("Q4Z output length mismatch".into()));
    }
    let mut buf = Vec::with_capacity(row_bytes);
    for i in 0..rows {
        let src = match &t.source_for_output {
            Some(order) => *order.get(i).ok_or_else(|| bad("invalid packed row permutation".into()))?,
            None => i,
        };
        buf.clear();
        p.encode_q4z_row(src, &mut buf).map_err(bad)?;
        w.write_all(&buf)?;
    }
    Ok(t.data_len)
}
pub(crate) fn to_f32(dtype: &str, scalar: &[u8]) -> io::Result<f32> {
    Ok(match dtype {
        "F32" => f32::from_le_bytes(scalar.try_into().unwrap()),
        "F16" => half::f16::from_bits(u16::from_le_bytes(scalar.try_into().unwrap())).to_f32(),
        "BF16" => f32::from_bits((u16::from_le_bytes(scalar.try_into().unwrap()) as u32) << 16),
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported source dtype",
            ))
        }
    })
}
pub(crate) fn append_converted(out: &mut Vec<u8>, dtype: &str, v: f32) -> io::Result<()> {
    match dtype {
        "F32" => out.extend_from_slice(&v.to_le_bytes()),
        "F16" => out.extend_from_slice(&half::f16::from_f32(v).to_bits().to_le_bytes()),
        "BF16" => {
            // round to nearest even
            let bits = v.to_bits();
            let rounded = if v.is_nan() {
                bits | 0x0040_0000
            } else {
                bits.wrapping_add(0x7fff + ((bits >> 16) & 1))
            };
            out.extend_from_slice(&((rounded >> 16) as u16).to_le_bytes());
        }
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported conversion output dtype",
            ))
        }
    }
    Ok(())
}
