use memmap2::{Mmap, MmapOptions};
use serde_json::Value;
use std::{collections::HashMap, fs::File, ops::Range, path::Path, sync::Arc};

#[derive(Clone)]
pub enum Payload {
    Mapped(Arc<Mmap>, Range<usize>),
    Owned(Arc<Vec<u8>>, Range<usize>),
}
impl Payload {
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Self::Mapped(m, r) => &m[r.clone()],
            Self::Owned(b, r) => &b[r.clone()],
        }
    }
    pub fn slice(&self, range: Range<usize>) -> Result<Self, String> {
        if range.start > range.end || range.end > self.as_slice().len() {
            return Err("payload subrange is out of bounds".into());
        }
        Ok(match self {
            Self::Mapped(m, r) => {
                Self::Mapped(Arc::clone(m), r.start + range.start..r.start + range.end)
            }
            Self::Owned(b, r) => {
                Self::Owned(Arc::clone(b), r.start + range.start..r.start + range.end)
            }
        })
    }
}
impl std::fmt::Debug for Payload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Payload")
            .field("len", &self.as_slice().len())
            .finish()
    }
}
#[derive(Clone, Debug)]
pub struct Tensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub payload: Payload,
}

pub fn read_file(path: &Path) -> Result<Vec<Tensor>, String> {
    let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mmap = Arc::new(
        unsafe { MmapOptions::new().map(&file) }
            .map_err(|e| format!("mmap {}: {e}", path.display()))?,
    );
    if mmap.len() < 8 {
        return Err(format!("{}: truncated safetensors header", path.display()));
    }
    let hlen = u64::from_le_bytes(mmap[..8].try_into().unwrap());
    let hlen = usize::try_from(hlen).map_err(|_| "safetensors header length does not fit usize")?;
    let hend = 8usize
        .checked_add(hlen)
        .ok_or("safetensors header overflow")?;
    if hend > mmap.len() {
        return Err("safetensors header extends past file".into());
    }
    let header: Value = serde_json::from_slice(&mmap[8..hend])
        .map_err(|e| format!("invalid safetensors JSON header: {e}"))?;
    let obj = header
        .as_object()
        .ok_or("safetensors header must be an object")?;
    let mut tensors = Vec::new();
    let mut ranges = Vec::new();
    for (name, meta) in obj {
        if name == "__metadata__" {
            continue;
        }
        let m = meta
            .as_object()
            .ok_or_else(|| format!("tensor {name}: metadata must be an object"))?;
        let dtype = m
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("tensor {name}: missing dtype"))?
            .to_owned();
        let shape = m
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("tensor {name}: missing shape"))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .ok_or_else(|| format!("tensor {name}: invalid shape dimension"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let offs = m
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("tensor {name}: missing data_offsets"))?;
        if offs.len() != 2 {
            return Err(format!("tensor {name}: data_offsets must have two entries"));
        }
        let start = usize::try_from(offs[0].as_u64().ok_or("invalid data offset")?)
            .map_err(|_| "data offset does not fit usize")?;
        let end = usize::try_from(offs[1].as_u64().ok_or("invalid data offset")?)
            .map_err(|_| "data offset does not fit usize")?;
        if start > end {
            return Err(format!("tensor {name}: reversed data_offsets"));
        }
        let count = shape.iter().try_fold(1u64, |a, &b| {
            a.checked_mul(b)
                .ok_or_else(|| format!("tensor {name}: shape size overflow"))
        })?;
        let width = match dtype.as_str() {
            "F64" | "I64" | "U64" => 8,
            "F32" | "I32" | "U32" => 4,
            "F16" | "BF16" | "I16" | "U16" => 2,
            "F8_E4M3" | "F8_E5M2" | "F8_E8M0" | "BOOL" | "U8" | "I8" => 1,
            _ => return Err(format!("tensor {name}: unsupported dtype {dtype}")),
        };
        let expected = count
            .checked_mul(width)
            .ok_or_else(|| format!("tensor {name}: byte size overflow"))?;
        if expected != (end - start) as u64 {
            return Err(format!(
                "tensor {name}: expected {expected} bytes from shape/dtype, found {}",
                end - start
            ));
        }
        let abs_start = hend.checked_add(start).ok_or("tensor offset overflow")?;
        let abs_end = hend.checked_add(end).ok_or("tensor offset overflow")?;
        if abs_end > mmap.len() {
            return Err(format!("tensor {name}: data extends past file"));
        }
        ranges.push((start, end, name.clone()));
        tensors.push(Tensor {
            name: name.clone(),
            dtype,
            shape,
            payload: Payload::Mapped(Arc::clone(&mmap), abs_start..abs_end),
        });
    }
    ranges.sort_by_key(|r| r.0);
    let mut cursor = 0usize;
    for (s, e, n) in ranges {
        if s < cursor {
            return Err(format!("overlapping tensor data at {n}"));
        }
        if s != cursor {
            return Err(format!("gap in safetensors data before {n}"));
        }
        cursor = e;
    }
    if hend.checked_add(cursor).ok_or("tensor data end overflow")? != mmap.len() {
        return Err("unreferenced or trailing safetensors bytes".into());
    }
    Ok(tensors)
}

pub fn read_checkpoint(dir: &Path) -> Result<Vec<Tensor>, String> {
    if dir.is_file() {
        return match dir.extension().and_then(|x| x.to_str()) {
            Some("safetensors") => read_file(dir),
            Some("bin" | "pt" | "pth") => crate::torch_zip::read(dir),
            _ => Err(format!(
                "unsupported checkpoint file type: {}",
                dir.display()
            )),
        };
    }
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.is_file() {
        let v: Value = serde_json::from_slice(
            &std::fs::read(&index_path).map_err(|e| format!("read index: {e}"))?,
        )
        .map_err(|e| format!("invalid shard index: {e}"))?;
        let wm = v
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or("index missing weight_map object")?;
        let mut by_shard: HashMap<String, Vec<String>> = HashMap::new();
        for (tensor, shard) in wm {
            let shard = shard
                .as_str()
                .ok_or("index shard filename must be a string")?;
            validate_shard_name(shard)?;
            if !Path::new(shard)
                .extension()
                .is_some_and(|x| x == "safetensors")
            {
                return Err(format!("index shard {shard:?} is not a safetensors file"));
            }
            by_shard
                .entry(shard.into())
                .or_default()
                .push(tensor.clone());
        }
        let mut all = Vec::new();
        for (shard, expected) in by_shard {
            let tensors = read_file(&resolve_shard(dir, &shard)?)?;
            let names: std::collections::HashSet<_> =
                tensors.iter().map(|t| t.name.as_str()).collect();
            for n in &expected {
                if !names.contains(n.as_str()) {
                    return Err(format!("index references missing tensor {n}"));
                }
            }
            for t in tensors {
                if !expected.iter().any(|n| n == &t.name) {
                    return Err(format!("shard contains unindexed tensor {}", t.name));
                }
                all.push(t);
            }
        }
        if all.len() != wm.len() {
            return Err("duplicate tensor across indexed shards".into());
        }
        Ok(all)
    } else {
        let p = dir.join("model.safetensors");
        if p.is_file() {
            return read_file(&p);
        }
        let mut shards: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| format!("read checkpoint directory: {e}"))?
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        shards.sort();
        if shards.is_empty() {
            let pt_index = dir.join("pytorch_model.bin.index.json");
            let model_index = dir.join("model.safetensors.index.json");
            if pt_index.is_file() {
                return read_torch_index(dir, &pt_index);
            }
            if model_index.is_file() {
                return Err(
                    "model.safetensors.index.json is present but no safetensors tensors were read"
                        .into(),
                );
            }
            let mut files: Vec<_> = std::fs::read_dir(dir)
                .map_err(|e| format!("read checkpoint directory: {e}"))?
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| {
                    p.extension()
                        .is_some_and(|x| matches!(x.to_str(), Some("bin" | "pt" | "pth")))
                })
                .collect();
            files.sort();
            if !files.is_empty() {
                let mut all = Vec::new();
                for p in files {
                    all.extend(crate::torch_zip::read(&p)?);
                }
                return Ok(all);
            }
            return Err("no model.safetensors or indexed safetensors shards found".into());
        }
        let mut all = Vec::new();
        for p in shards {
            all.extend(read_file(&p)?);
        }
        Ok(all)
    }
}
fn read_torch_index(dir: &Path, index: &Path) -> Result<Vec<Tensor>, String> {
    let v: Value = serde_json::from_slice(
        &std::fs::read(index).map_err(|e| format!("read PyTorch shard index: {e}"))?,
    )
    .map_err(|e| format!("invalid PyTorch shard index: {e}"))?;
    let wm = v
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or("PyTorch index missing weight_map object")?;
    let mut by_file: HashMap<String, Vec<String>> = HashMap::new();
    for (name, file) in wm {
        let file = file
            .as_str()
            .ok_or("PyTorch index filename must be a string")?;
        validate_shard_name(file)?;
        if !matches!(
            Path::new(file).extension().and_then(|s| s.to_str()),
            Some("bin" | "pt" | "pth")
        ) {
            return Err(format!("unsupported PyTorch shard extension in {file:?}"));
        }
        by_file.entry(file.into()).or_default().push(name.clone());
    }
    let mut all = Vec::new();
    for (file, expected) in by_file {
        let shard = resolve_shard(dir, &file)?;
        let tensors = crate::torch_zip::read(&shard)?;
        let found: std::collections::HashSet<_> = tensors.iter().map(|t| t.name.as_str()).collect();
        for name in &expected {
            if !found.contains(name.as_str()) {
                return Err(format!("PyTorch index references missing tensor {name}"));
            }
        }
        for t in tensors {
            if !expected.iter().any(|n| n == &t.name) {
                return Err(format!(
                    "PyTorch shard contains unindexed tensor {}",
                    t.name
                ));
            }
            all.push(t);
        }
    }
    if all.len() != wm.len() {
        return Err("duplicate tensor across indexed PyTorch shards".into());
    }
    Ok(all)
}
fn validate_shard_name(s: &str) -> Result<(), String> {
    let p = Path::new(s);
    if p.is_absolute()
        || p.components().count() != 1
        || s == "."
        || s == ".."
        || s.contains('/')
        || s.contains('\\')
    {
        return Err(format!("unsafe shard path {s:?}"));
    }
    Ok(())
}
fn resolve_shard(dir: &Path, name: &str) -> Result<std::path::PathBuf, String> {
    let path = dir.join(name);
    let metadata = std::fs::symlink_metadata(&path)
        .map_err(|e| format!("open indexed shard {name:?}: {e}"))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!(
            "indexed shard {name:?} must be a regular file, not a symlink"
        ));
    }
    let root =
        std::fs::canonicalize(dir).map_err(|e| format!("resolve checkpoint directory: {e}"))?;
    let resolved =
        std::fs::canonicalize(&path).map_err(|e| format!("resolve shard {name:?}: {e}"))?;
    if resolved.parent() != Some(root.as_path()) {
        return Err(format!(
            "indexed shard {name:?} resolves outside checkpoint directory"
        ));
    }
    Ok(resolved)
}
