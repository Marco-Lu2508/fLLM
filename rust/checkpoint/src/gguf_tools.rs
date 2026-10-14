//! GGUF rewrites preserve encoded metadata types and copy untouched tensor bytes.
use crate::{safetensors, CheckpointError, Preset};
use fllm_gguf_probe::MappedGguf;
use memmap2::MmapOptions;
use serde_json::Value;
use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

type Metadata = Vec<(String, Vec<u8>)>;

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], CheckpointError> {
        let end = self
            .at
            .checked_add(n)
            .ok_or("GGUF header offset overflow")?;
        let bytes = self
            .bytes
            .get(self.at..end)
            .ok_or("truncated GGUF metadata")?;
        self.at = end;
        Ok(bytes)
    }
    fn u32(&mut self) -> Result<u32, CheckpointError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().map_err(|_| "invalid u32")?,
        ))
    }
    fn u64(&mut self) -> Result<u64, CheckpointError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().map_err(|_| "invalid u64")?,
        ))
    }
    fn string(&mut self) -> Result<&'a str, CheckpointError> {
        let len = usize::try_from(self.u64()?).map_err(|_| "GGUF string length overflow")?;
        std::str::from_utf8(self.take(len)?).map_err(|e| format!("invalid GGUF string: {e}").into())
    }
    fn scalar(&mut self, kind: u32) -> Result<(), CheckpointError> {
        let width = match kind {
            0 | 1 | 7 => 1,
            2 | 3 => 2,
            4..=6 => 4,
            10..=12 => 8,
            8 => {
                self.string()?;
                return Ok(());
            }
            _ => return Err(format!("invalid GGUF metadata type {kind}").into()),
        };
        self.take(width)?;
        Ok(())
    }
}

fn metadata(path: &Path) -> Result<Metadata, CheckpointError> {
    let file = File::open(path)?;
    // SAFETY: read-only mapping is retained for the whole header read.
    let map = unsafe { MmapOptions::new().map(&file)? };
    let mut r = Reader { bytes: &map, at: 0 };
    if r.take(4)? != b"GGUF" {
        return Err("invalid GGUF magic".into());
    }
    if !(2..=3).contains(&r.u32()?) {
        return Err("unsupported GGUF version".into());
    }
    r.u64()?;
    let count = r.u64()?;
    if count > 1_000_000 {
        return Err("unreasonable GGUF metadata count".into());
    }
    let mut meta = Vec::new();
    for _ in 0..count {
        let key = r.string()?.to_owned();
        let start = r.at;
        let kind = r.u32()?;
        if kind == 9 {
            let element = r.u32()?;
            let n = r.u64()?;
            if n > map.len().saturating_sub(r.at) as u64 {
                return Err("GGUF array exceeds remaining header".into());
            }
            for _ in 0..n {
                r.scalar(element)?;
            }
        } else {
            r.scalar(kind)?;
        }
        meta.push((key, map[start..r.at].to_vec()));
    }
    Ok(meta)
}

fn put_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}
fn replace(meta: &mut Metadata, key: &str, value: Vec<u8>) {
    if let Some((_, bytes)) = meta.iter_mut().find(|(k, _)| k == key) {
        *bytes = value;
    } else {
        meta.push((key.into(), value));
    }
}
fn unsigned(kind: u32, v: u64) -> Result<Vec<u8>, CheckpointError> {
    let mut bytes = kind.to_le_bytes().to_vec();
    match kind {
        2 => bytes.extend_from_slice(
            &u16::try_from(v)
                .map_err(|_| "split count exceeds u16")?
                .to_le_bytes(),
        ),
        4 => bytes.extend_from_slice(
            &u32::try_from(v)
                .map_err(|_| "metadata integer exceeds u32")?
                .to_le_bytes(),
        ),
        5 => bytes.extend_from_slice(
            &i32::try_from(v)
                .map_err(|_| "split tensor count exceeds i32")?
                .to_le_bytes(),
        ),
        10 => bytes.extend_from_slice(&v.to_le_bytes()),
        _ => return Err("unsupported unsigned metadata type".into()),
    }
    Ok(bytes)
}
fn remove_split(meta: &mut Metadata) {
    meta.retain(|(k, _)| !k.starts_with("split."));
}

fn complete_metadata(input: &Path, model: &MappedGguf) -> Result<Metadata, CheckpointError> {
    let mut meta = metadata(input)?;
    if let Some(count) = model
        .index()
        .metadata
        .get("split.count")
        .and_then(Value::as_u64)
    {
        let suffix = format!("-00001-of-{count:05}.gguf");
        let filename = input
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or("invalid shard filename")?;
        let prefix = filename
            .strip_suffix(&suffix)
            .ok_or("invalid first shard filename")?;
        for no in 1..count {
            for (key, bytes) in metadata(
                &input.with_file_name(format!("{prefix}-{:05}-of-{count:05}.gguf", no + 1)),
            )? {
                if !meta.iter().any(|(k, _)| k == &key) {
                    meta.push((key, bytes));
                }
            }
        }
    }
    remove_split(&mut meta);
    Ok(meta)
}

struct Tensor<'a> {
    name: &'a str,
    dims: &'a [u64],
    kind: u32,
    bytes: Cow<'a, [u8]>,
}
fn tensors(model: &MappedGguf) -> Result<Vec<Tensor<'_>>, CheckpointError> {
    model
        .index()
        .tensors
        .iter()
        .map(|t| {
            Ok(Tensor {
                name: &t.name,
                dims: &t.dimensions,
                kind: t.type_id,
                bytes: Cow::Borrowed(model.tensor_bytes(&t.name)?),
            })
        })
        .collect()
}
fn alignment(model: &MappedGguf) -> Result<u64, CheckpointError> {
    let n = model
        .index()
        .metadata
        .get("general.alignment")
        .and_then(Value::as_u64)
        .unwrap_or(32);
    if !n.is_power_of_two() || n > 1024 * 1024 {
        return Err("GGUF alignment must be a power of two at most 1 MiB".into());
    }
    Ok(n)
}
fn align(n: u64, a: u64) -> Result<u64, CheckpointError> {
    n.checked_add(a - 1)
        .map(|n| n & !(a - 1))
        .ok_or_else(|| "GGUF size overflow".into())
}

fn header(
    meta: &Metadata,
    tensors: &[Tensor<'_>],
    a: u64,
    version: u32,
) -> Result<Vec<u8>, CheckpointError> {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&version.to_le_bytes());
    bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(meta.len() as u64).to_le_bytes());
    for (key, value) in meta {
        put_string(&mut bytes, key);
        bytes.extend_from_slice(value);
    }
    let mut offset = 0u64;
    for t in tensors {
        put_string(&mut bytes, t.name);
        bytes.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for n in t.dims {
            bytes.extend_from_slice(&n.to_le_bytes());
        }
        bytes.extend_from_slice(&t.kind.to_le_bytes());
        bytes.extend_from_slice(&offset.to_le_bytes());
        offset = align(
            offset
                .checked_add(t.bytes.len() as u64)
                .ok_or("GGUF payload size overflow")?,
            a,
        )?;
    }
    let len =
        usize::try_from(align(bytes.len() as u64, a)?).map_err(|_| "GGUF header size overflow")?;
    bytes.resize(len, 0);
    Ok(bytes)
}
fn output_size(
    meta: &Metadata,
    ts: &[Tensor<'_>],
    a: u64,
    version: u32,
) -> Result<u64, CheckpointError> {
    let mut n = header(meta, ts, a, version)?.len() as u64;
    for t in ts {
        n = align(n, a)?
            .checked_add(t.bytes.len() as u64)
            .ok_or("GGUF file size overflow")?;
    }
    Ok(n)
}
fn temporary(out: &Path) -> PathBuf {
    let mut p = out.as_os_str().to_owned();
    p.push(format!(
        ".{}.{}.tmp",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    PathBuf::from(p)
}
fn write(
    out: &Path,
    meta: &Metadata,
    ts: &[Tensor<'_>],
    a: u64,
    version: u32,
) -> Result<(), CheckpointError> {
    if out.exists() {
        return Err(format!("output already exists: {}", out.display()).into());
    }
    let tmp = temporary(out);
    let result = (|| -> Result<(), CheckpointError> {
        let mut w = BufWriter::new(OpenOptions::new().write(true).create_new(true).open(&tmp)?);
        let header = header(meta, ts, a, version)?;
        w.write_all(&header)?;
        let mut at = header.len() as u64;
        for t in ts {
            let pad = align(at, a)? - at;
            w.write_all(&vec![0; pad as usize])?;
            for chunk in t.bytes.chunks(1024 * 1024) {
                w.write_all(chunk)?;
            }
            at = at
                .checked_add(pad)
                .and_then(|n| n.checked_add(t.bytes.len() as u64))
                .ok_or("GGUF file size overflow")?;
        }
        w.flush()?;
        w.get_ref().sync_all()?;
        drop(w);
        fs::hard_link(&tmp, out)?;
        fs::remove_file(&tmp)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

/// Rewrite scalar metadata; unmodified records retain their original GGUF types.
pub fn set_metadata(
    input: &Path,
    output: &Path,
    assignments: &[String],
) -> Result<(), CheckpointError> {
    let model = MappedGguf::open_single(input)?;
    if model.index().metadata.contains_key("split.count") {
        return Err("merge GGUF shards before editing metadata".into());
    }
    let mut meta = metadata(input)?;
    let mut keys = HashSet::new();
    let mut output_alignment = alignment(&model)?;
    for item in assignments {
        let (key, text) = item
            .split_once('=')
            .ok_or("metadata assignment must be KEY=VALUE")?;
        if key.is_empty() || key.starts_with("GGUF.") || key.starts_with("split.") {
            return Err(
                format!("metadata key {key:?} is empty or controls the GGUF layout").into(),
            );
        }
        if !keys.insert(key) {
            return Err(format!("duplicate metadata assignment {key}").into());
        }
        let value =
            serde_json::from_str::<Value>(text).unwrap_or_else(|_| Value::String(text.into()));
        if key == "general.alignment" {
            output_alignment = value
                .as_u64()
                .filter(|n| n.is_power_of_two() && *n <= 1024 * 1024)
                .ok_or("general.alignment must be a power of two at most 1 MiB")?;
        }
        let mut bytes = Vec::new();
        match value {
            Value::String(s) => {
                bytes.extend_from_slice(&8u32.to_le_bytes());
                put_string(&mut bytes, &s);
            }
            Value::Bool(b) => {
                bytes.extend_from_slice(&7u32.to_le_bytes());
                bytes.push(u8::from(b));
            }
            Value::Number(n) => {
                if let Some(n) = n.as_u64() {
                    bytes = unsigned(if u32::try_from(n).is_ok() { 4 } else { 10 }, n)?;
                } else if let Some(n) = n.as_i64() {
                    if let Ok(n) = i32::try_from(n) {
                        bytes.extend_from_slice(&5u32.to_le_bytes());
                        bytes.extend_from_slice(&n.to_le_bytes());
                    } else {
                        bytes.extend_from_slice(&11u32.to_le_bytes());
                        bytes.extend_from_slice(&n.to_le_bytes());
                    }
                } else {
                    let n = n
                        .as_f64()
                        .filter(|n| n.is_finite())
                        .ok_or("metadata number must be finite")?;
                    bytes.extend_from_slice(&12u32.to_le_bytes());
                    bytes.extend_from_slice(&n.to_le_bytes());
                }
            }
            _ => return Err("metadata values must be strings, numbers or booleans".into()),
        }
        replace(&mut meta, key, bytes);
    }
    write(
        output,
        &meta,
        &tensors(&model)?,
        output_alignment,
        model.index().version,
    )
}

/// Split at tensor boundaries; the size cap includes each shard's complete header.
pub fn split(input: &Path, prefix: &Path, max_size: u64) -> Result<Vec<PathBuf>, CheckpointError> {
    if max_size == 0 {
        return Err("--max-size must be positive".into());
    }
    let model = MappedGguf::open(input)?;
    let ts = tensors(&model)?;
    if ts.is_empty() {
        return Err("cannot split a GGUF without tensors".into());
    }
    let mut meta = complete_metadata(input, &model)?;
    replace(&mut meta, "split.no", unsigned(2, 0)?);
    replace(&mut meta, "split.count", unsigned(2, 1)?);
    replace(
        &mut meta,
        "split.tensors.count",
        unsigned(5, ts.len() as u64)?,
    );
    let a = alignment(&model)?;
    let mut groups = Vec::new();
    let mut first = 0;
    for end in 1..=ts.len() {
        if output_size(&meta, &ts[first..end], a, model.index().version)? > max_size {
            if end - first == 1 {
                return Err(format!(
                    "tensor {} plus GGUF header exceeds --max-size",
                    ts[first].name
                )
                .into());
            }
            groups.push(first..end - 1);
            first = end - 1;
            if output_size(&meta, &ts[first..end], a, model.index().version)? > max_size {
                return Err(format!(
                    "tensor {} plus GGUF header exceeds --max-size",
                    ts[first].name
                )
                .into());
            }
        }
    }
    groups.push(first..ts.len());
    replace(&mut meta, "split.count", unsigned(2, groups.len() as u64)?);
    let outputs = (1..=groups.len())
        .map(|n| {
            let mut path = prefix.as_os_str().to_owned();
            path.push(format!("-{n:05}-of-{:05}.gguf", groups.len()));
            PathBuf::from(path)
        })
        .collect::<Vec<_>>();
    for out in &outputs {
        if out.exists() {
            return Err(format!("output already exists: {}", out.display()).into());
        }
    }
    let mut written: Vec<PathBuf> = Vec::new();
    for (no, (range, out)) in groups.iter().zip(&outputs).enumerate() {
        replace(&mut meta, "split.no", unsigned(2, no as u64)?);
        if let Err(e) = write(out, &meta, &ts[range.clone()], a, model.index().version) {
            for path in written {
                let _ = fs::remove_file(path);
            }
            return Err(e);
        }
        written.push(out.clone());
    }
    Ok(outputs)
}

/// Join validated, consecutively named shards in their original tensor order.
pub fn merge(input: &Path, output: &Path) -> Result<(), CheckpointError> {
    let model = MappedGguf::open(input)?;
    if !model.index().metadata.contains_key("split.count") {
        return Err("input is not a GGUF shard set".into());
    }
    let meta = complete_metadata(input, &model)?;
    write(
        output,
        &meta,
        &tensors(&model)?,
        alignment(&model)?,
        model.index().version,
    )
}

fn projection(target: &str, model: &MappedGguf) -> Result<String, CheckpointError> {
    let target = target.strip_prefix("base_model.model.").unwrap_or(target);
    if target == "lm_head" || target == "model.lm_head" {
        if model
            .index()
            .tensors
            .iter()
            .any(|t| t.name == "output.weight")
        {
            return Ok("output.weight".into());
        }
        return Err("LoRA lm_head requires an explicit output.weight tensor; tied embeddings are unsupported".into());
    }
    let parts: Vec<_> = target.split('.').collect();
    let pos = parts
        .iter()
        .position(|&s| s == "layers")
        .ok_or_else(|| format!("unsupported PEFT LoRA target {target}"))?;
    let layer = parts
        .get(pos + 1)
        .ok_or("missing LoRA layer number")?
        .parse::<u32>()
        .map_err(|_| "invalid LoRA layer number")?;
    let tail = parts
        .get(pos + 2..)
        .ok_or("missing LoRA projection")?
        .join(".");
    let part = match tail.as_str() {
        "self_attn.q_proj" => "attn_q",
        "self_attn.k_proj" => "attn_k",
        "self_attn.v_proj" => "attn_v",
        "self_attn.o_proj" => "attn_output",
        "mlp.gate_proj" => "ffn_gate",
        "mlp.up_proj" => "ffn_up",
        "mlp.down_proj" => "ffn_down",
        _ => return Err(format!("unsupported PEFT LoRA projection {tail}").into()),
    };
    Ok(format!("blk.{layer}.{part}.weight"))
}
fn values(t: &safetensors::Tensor) -> Result<Vec<f32>, CheckpointError> {
    let width = match t.dtype.as_str() {
        "F32" => 4,
        "F16" | "BF16" => 2,
        _ => return Err(format!("unsupported LoRA dtype {}", t.dtype).into()),
    };
    let mut values = Vec::new();
    for scalar in t.payload.as_slice().chunks_exact(width) {
        values.push(crate::gguf::to_f32(&t.dtype, scalar)?);
    }
    if values.iter().any(|n| !n.is_finite()) {
        return Err(format!("non-finite LoRA tensor {}", t.name).into());
    }
    Ok(values)
}
fn pattern_number(
    cfg: &Value,
    field: &str,
    target: &str,
    fallback: f64,
) -> Result<f64, CheckpointError> {
    let Some(map) = cfg[field].as_object() else {
        return Ok(fallback);
    };
    if map
        .keys()
        .any(|key| key.chars().any(|c| "^$[]()*+?|\\".contains(c)))
    {
        return Err(format!("LoRA {field} supports literal target suffixes only").into());
    }
    let matches = map
        .iter()
        .filter(|(key, _)| {
            target == key.as_str()
                || target
                    .strip_suffix(key.as_str())
                    .is_some_and(|prefix| prefix.ends_with('.'))
        })
        .collect::<Vec<_>>();
    if matches.len() > 1 {
        return Err(format!("ambiguous LoRA {field} for {target}").into());
    }
    matches
        .first()
        .map(|(_, v)| {
            v.as_f64()
                .filter(|n| n.is_finite())
                .ok_or_else(|| format!("invalid LoRA {field}").into())
        })
        .unwrap_or(Ok(fallback))
}

/// Merge PEFT linear LoRA using the caller's native GGUF dequantizer.
pub fn merge_lora(
    base: &Path,
    adapter: &Path,
    output: &Path,
    scale: f32,
    outtype: &str,
    decode: impl Fn(&[u8], u32, usize) -> Result<Vec<f32>, String>,
) -> Result<usize, CheckpointError> {
    if !scale.is_finite() {
        return Err("LoRA scale must be finite".into());
    }
    let outtype = outtype.to_ascii_lowercase();
    let preset = Preset::parse(&outtype);
    if !matches!(outtype.as_str(), "f16" | "f32") && preset.is_none() {
        return Err(format!(
            "unsupported --outtype {outtype}; use f16, f32 or {}",
            Preset::NAMES
        )
        .into());
    }
    if output.exists() {
        return Err(format!("output already exists: {}", output.display()).into());
    }
    let root = if adapter.is_dir() {
        adapter
    } else {
        adapter.parent().ok_or("adapter has no parent directory")?
    };
    let cfg: Value = serde_json::from_slice(&fs::read(root.join("adapter_config.json"))?)
        .map_err(|e| format!("invalid adapter_config.json: {e}"))?;
    if cfg["peft_type"].as_str().is_some_and(|s| s != "LORA")
        || cfg["bias"].as_str().is_some_and(|s| s != "none")
        || cfg["use_dora"].as_bool() == Some(true)
        || cfg["fan_in_fan_out"].as_bool() == Some(true)
        || !cfg["alora_invocation_tokens"].is_null()
        || cfg["modules_to_save"]
            .as_array()
            .is_some_and(|a| !a.is_empty())
    {
        return Err(
            "only linear PEFT LoRA without DoRA, aLoRA, extra biases or saved modules is supported"
                .into(),
        );
    }
    let rank = cfg["r"]
        .as_u64()
        .filter(|&n| n > 0)
        .ok_or("adapter_config.json requires positive r")?;
    let alpha = cfg["lora_alpha"]
        .as_f64()
        .filter(|n| n.is_finite() && *n >= 0.0)
        .ok_or("adapter_config.json requires non-negative lora_alpha")?;
    let model = MappedGguf::open(base)?;
    let source = if adapter.is_dir() {
        root.join("adapter_model.safetensors")
    } else {
        adapter.to_owned()
    };
    if source.extension().and_then(|s| s.to_str()) != Some("safetensors") {
        return Err("LoRA adapter must be a safetensors file or PEFT directory".into());
    }
    let mut pairs: HashMap<String, (Option<safetensors::Tensor>, Option<safetensors::Tensor>)> =
        HashMap::new();
    for t in safetensors::read_file(&source)? {
        let (target, side, suffix) = if let Some((name, suffix)) = t.name.split_once(".lora_A.") {
            (name.to_owned(), true, suffix)
        } else if let Some((name, suffix)) = t.name.split_once(".lora_B.") {
            (name.to_owned(), false, suffix)
        } else {
            return Err(format!("unsupported PEFT adapter tensor {}", t.name).into());
        };
        if suffix != "weight" && suffix != "default.weight" {
            return Err(format!("unsupported PEFT adapter suffix {suffix}").into());
        }
        let pair = pairs.entry(target).or_default();
        let slot = if side { &mut pair.0 } else { &mut pair.1 };
        if slot.is_some() {
            return Err(format!("duplicate LoRA tensor {}", t.name).into());
        }
        *slot = Some(t);
    }
    if pairs.is_empty() {
        return Err("LoRA adapter contains no projections".into());
    }
    let mut replacements = HashMap::new();
    for (target, (a, b)) in pairs {
        let a = a.ok_or_else(|| format!("missing lora_A for {target}"))?;
        let b = b.ok_or_else(|| format!("missing lora_B for {target}"))?;
        let name = projection(&target, &model)?;
        if replacements.contains_key(&name) {
            return Err(format!("multiple LoRA targets map to {name}").into());
        }
        let info = model
            .index()
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("base tensor {name} is missing"))?;
        let [cols, rows] = info.dimensions.as_slice() else {
            return Err(format!("LoRA target {name} must be a matrix").into());
        };
        let expected_rank = pattern_number(&cfg, "rank_pattern", &target, rank as f64)?;
        let r = *a.shape.first().ok_or("empty LoRA shape")?;
        if r == 0 || r as f64 != expected_rank || a.shape != [r, *cols] || b.shape != [*rows, r] {
            return Err(format!("incompatible LoRA rank or shape for {name}").into());
        }
        let alpha = pattern_number(&cfg, "alpha_pattern", &target, alpha)?;
        if alpha < 0.0 {
            return Err("LoRA alpha must be non-negative".into());
        }
        let factor = scale as f64 * alpha
            / if cfg["use_rslora"].as_bool() == Some(true) {
                (r as f64).sqrt()
            } else {
                r as f64
            };
        if !factor.is_finite() || !(factor as f32).is_finite() {
            return Err("LoRA effective scale is not finite".into());
        }
        let count = usize::try_from(cols.checked_mul(*rows).ok_or("LoRA matrix size overflow")?)
            .map_err(|_| "LoRA matrix size exceeds usize")?;
        let mut weights = decode(model.tensor_bytes(&name)?, info.type_id, count)?;
        if weights.len() != count {
            return Err("GGUF dequantizer returned the wrong tensor length".into());
        }
        let av = values(&a)?;
        let bv = values(&b)?;
        let (cols, rows, r) = (*cols as usize, *rows as usize, r as usize);
        let metadata = &model.index().metadata;
        let arch = metadata
            .get("general.architecture")
            .and_then(Value::as_str)
            .ok_or("missing base general.architecture")?;
        let order = if matches!(arch, "llama" | "mistral" | "granite" | "qwen3next")
            && (name.ends_with("attn_q.weight") || name.ends_with("attn_k.weight"))
        {
            let heads = metadata
                .get(&format!("{arch}.attention.head_count"))
                .and_then(Value::as_u64)
                .filter(|&n| n > 0)
                .ok_or("missing base attention head count")?;
            let kv_heads = metadata
                .get(&format!("{arch}.attention.head_count_kv"))
                .and_then(Value::as_u64)
                .unwrap_or(heads);
            let hidden = metadata
                .get(&format!("{arch}.embedding_length"))
                .and_then(Value::as_u64)
                .ok_or("missing base embedding length")?;
            let head_dim = metadata
                .get(&format!("{arch}.attention.key_length"))
                .and_then(Value::as_u64)
                .unwrap_or(hidden / heads);
            if heads > rows as u64
                || kv_heads > rows as u64
                || head_dim == 0
                || head_dim > rows as u64
            {
                return Err("invalid base RoPE dimensions for LoRA".into());
            }
            let order = crate::convert::row_permutation(arch, &name, heads, kv_heads, head_dim)?;
            if order.as_ref().is_some_and(|order| order.len() != rows) {
                return Err(format!("base RoPE dimensions do not match {name}").into());
            }
            order
        } else {
            None
        };
        for row in 0..rows {
            let source_row = order.as_ref().map_or(row, |order| order[row]);
            for col in 0..cols {
                let delta = (0..r)
                    .map(|k| bv[source_row * r + k] * av[k * cols + col])
                    .sum::<f32>();
                weights[row * cols + col] += factor as f32 * delta;
            }
        }
        if weights.iter().any(|n| !n.is_finite()) {
            return Err(format!("non-finite merged weights for {name}").into());
        }
        let mut bytes = Vec::new();
        let dtype = if outtype == "f16" { "F16" } else { "F32" };
        for v in weights {
            if dtype == "F16" && !half::f16::from_f32(v).is_finite() {
                return Err(
                    format!("merged tensor {name} exceeds f16 range; use --outtype f32").into(),
                );
            }
            crate::gguf::append_converted(&mut bytes, dtype, v)?;
        }
        replacements.insert(name, bytes);
    }
    let changed = replacements.len();
    let mut ts = tensors(&model)?;
    for t in &mut ts {
        if let Some(bytes) = replacements.remove(t.name) {
            t.bytes = Cow::Owned(bytes);
            t.kind = if outtype == "f16" { 1 } else { 0 };
        }
    }
    let mut meta = complete_metadata(base, &model)?;
    let a = alignment(&model)?;
    if let Some(preset) = preset {
        // The existing quantize writer uses 32-byte alignment.
        replace(&mut meta, "general.alignment", unsigned(4, 32)?);
        let tmp = temporary(output);
        let quantized = temporary(output);
        let result = (|| -> Result<(), CheckpointError> {
            write(&tmp, &meta, &ts, 32, model.index().version)?;
            crate::quantize_gguf(&tmp, &quantized, preset)?;
            fs::hard_link(&quantized, output)?;
            Ok(())
        })();
        let _ = fs::remove_file(tmp);
        let _ = fs::remove_file(quantized);
        result?;
    } else {
        write(output, &meta, &ts, a, model.index().version)?;
    }
    Ok(changed)
}
