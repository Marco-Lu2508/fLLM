//! Restricted reader for ordinary `torch.save(state_dict)` ZIP checkpoints.
//! This is deliberately a tiny pickle VM: it has no generic object constructor
//! and recognizes only primitive containers, storage IDs, and tensor rebuilds.
use crate::safetensors::{Payload, Tensor};
use memmap2::{Mmap, MmapOptions};
use std::{collections::HashMap, fs::File, io::Read, path::Path, sync::Arc};
use zip::ZipArchive;

#[derive(Clone, Debug)]
enum Value {
    Mark,
    None,
    Bool,
    Int(i64),
    Float,
    Str(String),
    Tuple(Vec<Value>),
    List(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    Global(String),
    Storage(Storage),
    Tensor(Tensor),
}
#[derive(Clone, Debug)]
struct Storage {
    dtype: String,
    payload: Payload,
}

pub fn read(path: &Path) -> Result<Vec<Tensor>, String> {
    let map_file =
        File::open(path).map_err(|e| format!("open PyTorch archive {}: {e}", path.display()))?;
    let mapping = Arc::new(
        unsafe { MmapOptions::new().map(&map_file) }
            .map_err(|e| format!("mmap PyTorch archive {}: {e}", path.display()))?,
    );
    if mapping.get(..4).is_none_or(|m| m != &b"PK\x03\x04"[..]) {
        return Err("legacy non-ZIP PyTorch pickle detected; this importer only accepts restricted ZIP state_dict archives and never executes legacy pickle".into());
    }
    let file =
        File::open(path).map_err(|e| format!("open PyTorch archive {}: {e}", path.display()))?;
    let mut zip = ZipArchive::new(file).map_err(|e| {
        format!(
            "{} is not a supported PyTorch ZIP archive: {e}",
            path.display()
        )
    })?;
    let mut names = HashMap::<String, usize>::new();
    for i in 0..zip.len() {
        let f = zip
            .by_index(i)
            .map_err(|e| format!("invalid ZIP entry: {e}"))?;
        let name = f.name().to_owned();
        if names.insert(name.clone(), i).is_some() {
            return Err(format!("duplicate PyTorch ZIP member {name:?}"));
        }
    }
    let data_name = names
        .keys()
        .find(|n| n.as_str() == "data.pkl" || n.ends_with("/data.pkl"))
        .cloned()
        .ok_or("PyTorch ZIP has no data.pkl (TorchScript and custom archives are unsupported)")?;
    let prefix = data_name.strip_suffix("data.pkl").unwrap().to_owned();
    let normalized = prefix.trim_end_matches('/');
    if normalized.contains('/')
        || normalized == "."
        || normalized == ".."
        || normalized.contains('\\')
    {
        return Err("unsafe PyTorch ZIP root directory".into());
    }
    let index = names[&data_name];
    let mut data_file = zip
        .by_index(index)
        .map_err(|e| format!("open data.pkl: {e}"))?;
    let size =
        usize::try_from(data_file.size()).map_err(|_| "data.pkl length does not fit usize")?;
    if size > 512 * 1024 * 1024 {
        return Err("PyTorch data.pkl exceeds the 512 MiB safety limit".into());
    }
    let mut pickle = Vec::with_capacity(size);
    data_file
        .read_to_end(&mut pickle)
        .map_err(|e| format!("read data.pkl: {e}"))?;
    drop(data_file);
    let byteorder = if let Some(i) = names.get(&format!("{prefix}byteorder")) {
        let mut f = zip
            .by_index(*i)
            .map_err(|e| format!("read byteorder: {e}"))?;
        let mut s = String::new();
        f.read_to_string(&mut s)
            .map_err(|e| format!("read byteorder: {e}"))?;
        match s.trim() {
            "little" => true,
            "big" => false,
            x => return Err(format!("invalid PyTorch byteorder {x:?}")),
        }
    } else {
        cfg!(target_endian = "little")
    };
    let mut machine = Pickle::new(&pickle, &mut zip, &names, &prefix, byteorder, mapping);
    let root = machine.run()?;
    let entries = match root {
        Value::Dict(items) => {
            if let Some((_, Value::Dict(nested))) = items
                .iter()
                .find(|(k, _)| matches!(k,Value::Str(s) if s=="state_dict"))
            {
                nested.clone()
            } else {
                items
            }
        }
        _ => return Err("PyTorch data.pkl root must be a state_dict mapping".into()),
    };
    let mut out = Vec::with_capacity(entries.len());
    let mut names_seen = std::collections::HashSet::new();
    for (k, v) in entries {
        let name = match k {
            Value::Str(s) => s,
            _ => return Err("PyTorch state_dict keys must be strings".into()),
        };
        let t = match v {
            Value::Tensor(t) => t,
            _ => return Err(format!("PyTorch state_dict item {name:?} is not a tensor")),
        };
        if !names_seen.insert(name.clone()) {
            return Err(format!("duplicate PyTorch state_dict tensor {name:?}"));
        }
        out.push(Tensor {
            name,
            dtype: t.dtype,
            shape: t.shape,
            payload: t.payload,
        });
    }
    Ok(out)
}

struct Pickle<'a, 'z> {
    b: &'a [u8],
    p: usize,
    stack: Vec<Value>,
    memo: Vec<Value>,
    zip: &'z mut ZipArchive<std::fs::File>,
    members: &'z HashMap<String, usize>,
    prefix: &'z str,
    little: bool,
    mapping: Arc<Mmap>,
}
impl<'a, 'z> Pickle<'a, 'z> {
    fn new(
        b: &'a [u8],
        zip: &'z mut ZipArchive<std::fs::File>,
        members: &'z HashMap<String, usize>,
        prefix: &'z str,
        little: bool,
        mapping: Arc<Mmap>,
    ) -> Self {
        Self {
            b,
            p: 0,
            stack: Vec::new(),
            memo: Vec::new(),
            zip,
            members,
            prefix,
            little,
            mapping,
        }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let e = self.p.checked_add(n).ok_or("pickle offset overflow")?;
        let x = self.b.get(self.p..e).ok_or("truncated PyTorch pickle")?;
        self.p = e;
        Ok(x)
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
    fn line(&mut self) -> Result<String, String> {
        let rest = self.b.get(self.p..).ok_or("pickle offset outside data")?;
        let n = rest
            .iter()
            .position(|&x| x == b'\n')
            .ok_or("unterminated pickle line")?;
        let s = std::str::from_utf8(&rest[..n])
            .map_err(|_| "pickle string is not UTF-8")?
            .to_owned();
        self.p += n + 1;
        Ok(s)
    }
    fn utf8(&mut self, n: usize) -> Result<String, String> {
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| "pickle string is not UTF-8".into())
    }
    fn pop(&mut self) -> Result<Value, String> {
        self.stack
            .pop()
            .ok_or_else(|| "pickle stack underflow".into())
    }
    fn mark(&mut self) -> Result<Vec<Value>, String> {
        let i = self
            .stack
            .iter()
            .rposition(|v| matches!(v, Value::Mark))
            .ok_or("pickle MARK missing")?;
        let rest = self.stack.split_off(i + 1);
        self.stack.pop();
        Ok(rest)
    }
    fn memo_put(&mut self, i: usize) -> Result<(), String> {
        if i > 1_000_000 {
            return Err("pickle memo index exceeds the safety limit".into());
        }
        if self.memo.len() <= i {
            self.memo.resize(i + 1, Value::None);
        }
        self.memo[i] = self
            .stack
            .last()
            .ok_or("memoize on empty pickle stack")?
            .clone();
        Ok(())
    }
    fn pop_int(v: Value) -> Result<i64, String> {
        if let Value::Int(i) = v {
            Ok(i)
        } else {
            Err("pickle expected integer".into())
        }
    }
    fn pop_str(v: Value) -> Result<String, String> {
        if let Value::Str(s) = v {
            Ok(s)
        } else {
            Err("pickle expected string".into())
        }
    }
    fn run(&mut self) -> Result<Value, String> {
        loop {
            let op = self.u8()?;
            match op {
                b'.' => return self.pop(),
                0x80 => {
                    let v = self.u8()?;
                    if v > 5 {
                        return Err(format!("unsupported pickle protocol {v}"));
                    }
                }
                0x95 => {
                    let n =
                        usize::try_from(self.u64()?).map_err(|_| "pickle FRAME length overflow")?;
                    if n > self.b.len().saturating_sub(self.p) {
                        return Err("pickle FRAME extends past data.pkl".into());
                    }
                }
                b'(' => self.stack.push(Value::Mark),
                b'N' => self.stack.push(Value::None),
                0x88 | 0x89 => self.stack.push(Value::Bool),
                b'}' => self.stack.push(Value::Dict(Vec::new())),
                b']' => self.stack.push(Value::List(Vec::new())),
                b')' => self.stack.push(Value::Tuple(Vec::new())),
                b'K' => {
                    let n = self.u8()?;
                    self.stack.push(Value::Int(n as i64));
                }
                b'M' => {
                    let n = u16::from_le_bytes(self.take(2)?.try_into().unwrap());
                    self.stack.push(Value::Int(n as i64));
                }
                b'J' => {
                    let n = i32::from_le_bytes(self.take(4)?.try_into().unwrap());
                    self.stack.push(Value::Int(n as i64));
                }
                0x8a => {
                    let n = self.u8()? as usize;
                    let raw = self.take(n)?;
                    let mut b = [if raw.last().is_some_and(|x| x & 0x80 != 0) {
                        0xff
                    } else {
                        0
                    }; 8];
                    let take = n.min(8);
                    b[..take].copy_from_slice(&raw[..take]);
                    self.stack.push(Value::Int(i64::from_le_bytes(b)));
                }
                0x8b => {
                    let n =
                        usize::try_from(self.u32()?).map_err(|_| "pickle LONG4 length overflow")?;
                    if n > 8 {
                        return Err("pickle integer exceeds i64 range".into());
                    }
                    let raw = self.take(n)?;
                    let mut b = [if raw.last().is_some_and(|x| x & 0x80 != 0) {
                        0xff
                    } else {
                        0
                    }; 8];
                    b[..n].copy_from_slice(raw);
                    self.stack.push(Value::Int(i64::from_le_bytes(b)));
                }
                b'F' => {
                    self.line()?
                        .parse::<f64>()
                        .map_err(|_| "invalid pickle FLOAT")?;
                    self.stack.push(Value::Float);
                }
                b'G' => {
                    let _ = f64::from_be_bytes(self.take(8)?.try_into().unwrap());
                    self.stack.push(Value::Float);
                }
                b'X' => {
                    let n = usize::try_from(self.u32()?).map_err(|_| "pickle string too long")?;
                    let s = self.utf8(n)?;
                    self.stack.push(Value::Str(s));
                }
                0x8c => {
                    let n = self.u8()? as usize;
                    let s = self.utf8(n)?;
                    self.stack.push(Value::Str(s));
                }
                0x8d => {
                    let n = usize::try_from(self.u64()?).map_err(|_| "pickle string too long")?;
                    let s = self.utf8(n)?;
                    self.stack.push(Value::Str(s));
                }
                b'U' => {
                    let n = self.u8()? as usize;
                    let s = self.utf8(n)?;
                    self.stack.push(Value::Str(s));
                }
                b'c' => {
                    let m = self.line()?;
                    let n = self.line()?;
                    self.stack.push(Value::Global(format!("{m}.{n}")));
                }
                0x93 => {
                    let name = Self::pop_str(self.pop()?)?;
                    let module = Self::pop_str(self.pop()?)?;
                    self.stack.push(Value::Global(format!("{module}.{name}")));
                }
                b'q' => {
                    let i = self.u8()? as usize;
                    self.memo_put(i)?;
                }
                b'r' => {
                    let i =
                        usize::try_from(self.u32()?).map_err(|_| "pickle memo index overflow")?;
                    self.memo_put(i)?;
                }
                0x94 => {
                    let i = self.memo.len();
                    self.memo_put(i)?;
                }
                b'h' => {
                    let i = self.u8()? as usize;
                    self.stack.push(
                        self.memo
                            .get(i)
                            .ok_or("invalid pickle memo reference")?
                            .clone(),
                    );
                }
                b'j' => {
                    let i =
                        usize::try_from(self.u32()?).map_err(|_| "pickle memo index overflow")?;
                    self.stack.push(
                        self.memo
                            .get(i)
                            .ok_or("invalid pickle memo reference")?
                            .clone(),
                    );
                }
                b't' => {
                    let v = self.mark()?;
                    self.stack.push(Value::Tuple(v));
                }
                0x85 => {
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(vec![a]));
                }
                0x86 => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(vec![a, b]));
                }
                0x87 => {
                    let c = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(vec![a, b, c]));
                }
                b'l' => {
                    let v = self.mark()?;
                    self.stack.push(Value::List(v));
                }
                b'd' => {
                    let v = self.mark()?;
                    self.stack.push(make_dict(v)?);
                }
                b'a' => {
                    let item = self.pop()?;
                    match self.stack.last_mut() {
                        Some(Value::List(v)) => v.push(item),
                        _ => return Err("APPEND target is not a list".into()),
                    }
                }
                b'e' => {
                    let vals = self.mark()?;
                    match self.stack.last_mut() {
                        Some(Value::List(v)) => v.extend(vals),
                        _ => return Err("APPENDS target is not a list".into()),
                    }
                }
                b's' => {
                    let value = self.pop()?;
                    let key = self.pop()?;
                    match self.stack.last_mut() {
                        Some(Value::Dict(v)) => v.push((key, value)),
                        _ => return Err("SETITEM target is not a dict".into()),
                    }
                }
                b'u' => {
                    let vals = self.mark()?;
                    let (pairs, remainder) = vals.as_chunks::<2>();
                    if !remainder.is_empty() {
                        return Err("SETITEMS has odd number of values".into());
                    }
                    match self.stack.last_mut() {
                        Some(Value::Dict(v)) => {
                            for pair in pairs {
                                let [key, value] = pair;
                                v.push((key.clone(), value.clone()));
                            }
                        }
                        _ => return Err("SETITEMS target is not a dict".into()),
                    }
                }
                b'Q' => {
                    let pid = self.pop()?;
                    let storage = self.storage(pid)?;
                    self.stack.push(Value::Storage(storage));
                }
                b'R' => {
                    let args = self.pop()?;
                    let callable = self.pop()?;
                    let result = self.reduce(callable, args)?;
                    self.stack.push(result);
                }
                b'b' => {
                    let state = self.pop()?;
                    let obj = self.pop()?;
                    match (obj, state) {
                        (Value::Dict(d), Value::Dict(s))
                            if s.len() == 1
                                && s.first().is_some_and(|(k, v)| {
                                    matches!(k,Value::Str(k) if k=="_metadata")
                                        && matches!(v, Value::Dict(_))
                                }) =>
                        {
                            self.stack.push(Value::Dict(d))
                        }
                        _ => {
                            return Err(
                                "pickle BUILD is not allowed for this checkpoint object".into()
                            )
                        }
                    }
                }
                _ => return Err(format!("unsupported or unsafe pickle opcode 0x{op:02x}")),
            }
        }
    }
    fn storage(&mut self, pid: Value) -> Result<Storage, String> {
        let v = match pid {
            Value::Tuple(v) => v,
            _ => return Err("PyTorch persistent ID is not a tuple".into()),
        };
        if v.len() < 5
            || Self::pop_str(v.first().ok_or("PyTorch persistent ID is empty")?.clone())?
                != "storage"
        {
            return Err("unsupported PyTorch persistent ID".into());
        }
        let storage_type = match v
            .get(1)
            .ok_or("PyTorch persistent ID is missing storage type")?
        {
            Value::Global(g) => g.rsplit('.').next().unwrap_or(g),
            _ => return Err("storage type is not a known Torch global".into()),
        };
        let dtype = match storage_type {
            "FloatStorage" => "F32",
            "HalfStorage" => "F16",
            "BFloat16Storage" => "BF16",
            "UntypedStorage" => "UNTYPED",
            _ => return Err(format!("unsupported Torch storage type {storage_type}")),
        }
        .to_owned();
        let key = Self::pop_str(
            v.get(2)
                .ok_or("PyTorch persistent ID is missing storage key")?
                .clone(),
        )?;
        if key.is_empty() || !key.bytes().all(|b| b.is_ascii_digit()) {
            return Err("unsafe Torch storage key".into());
        }
        let elements = u64::try_from(Self::pop_int(
            v.get(4)
                .ok_or("PyTorch persistent ID is missing storage size")?
                .clone(),
        )?)
        .map_err(|_| "negative Torch storage size")?;
        let width = width(&dtype);
        let wanted = elements
            .checked_mul(width)
            .ok_or("Torch storage byte count overflow")?;
        let member = format!("{}data/{key}", self.prefix);
        let index = *self
            .members
            .get(&member)
            .ok_or_else(|| format!("missing PyTorch storage {member:?}"))?;
        let mut f = self
            .zip
            .by_index(index)
            .map_err(|e| format!("open storage {key}: {e}"))?;
        if f.size() != wanted {
            return Err(format!(
                "Torch storage {key} has {} bytes but dtype/size requires {wanted}",
                f.size()
            ));
        }
        let payload = if f.compression() == zip::CompressionMethod::Stored
            && f.compressed_size() == f.size()
            && self.little
        {
            let start =
                usize::try_from(f.data_start()).map_err(|_| "ZIP storage offset overflow")?;
            let end = start
                .checked_add(usize::try_from(wanted).map_err(|_| "Torch storage too large")?)
                .ok_or("ZIP storage range overflow")?;
            if end > self.mapping.len() {
                return Err("ZIP storage payload extends past archive".into());
            }
            Payload::Mapped(Arc::clone(&self.mapping), start..end)
        } else {
            let n =
                usize::try_from(wanted).map_err(|_| "Torch storage does not fit address space")?;
            let mut bytes = Vec::with_capacity(n);
            f.read_to_end(&mut bytes)
                .map_err(|e| format!("read Torch storage {key}: {e}"))?;
            if !self.little && width > 1 {
                for c in bytes.chunks_exact_mut(width as usize) {
                    c.reverse();
                }
            }
            Payload::Owned(Arc::new(bytes), 0..n)
        };
        Ok(Storage { dtype, payload })
    }
    fn reduce(&self, call: Value, args: Value) -> Result<Value, String> {
        let name = match call {
            Value::Global(n) => n,
            other => {
                return Err(format!(
                    "pickle REDUCE callable is not an allowlisted global: {other:?}"
                ))
            }
        };
        let args = match args {
            Value::Tuple(v) => v,
            _ => return Err("pickle REDUCE arguments are not a tuple".into()),
        };
        if name == "collections.OrderedDict" {
            if !args.is_empty() {
                return Err("nonempty OrderedDict constructor arguments unsupported".into());
            }
            return Ok(Value::Dict(Vec::new()));
        }
        if name == "torch._utils._rebuild_tensor_v2"
            || name == "torch._utils._rebuild_tensor"
            || name == "torch._utils._rebuild_tensor_v3"
        {
            return self.rebuild(&name, &args).map(Value::Tensor);
        }
        Err(format!("pickle REDUCE global {name:?} is not allowlisted"))
    }
    fn rebuild(&self, name: &str, a: &[Value]) -> Result<Tensor, String> {
        if a.len() < 4 {
            return Err("tensor rebuild has too few arguments".into());
        }
        if name.ends_with("_rebuild_tensor_v2")
            && (a.len() != 6
                || !matches!(a.get(4), Some(Value::Bool))
                || !matches!(a.get(5), Some(Value::None | Value::Dict(_))))
        {
            return Err("unsupported _rebuild_tensor_v2 requires_grad/hooks arguments".into());
        }
        if name.ends_with("_rebuild_tensor") && a.len() != 4 {
            return Err("unsupported _rebuild_tensor argument count".into());
        }
        if name.ends_with("_rebuild_tensor_v3")
            && (a.len() < 7
                || !matches!(a.get(4), Some(Value::Bool))
                || !matches!(a.get(5), Some(Value::None | Value::Dict(_))))
        {
            return Err("unsupported _rebuild_tensor_v3 requires_grad/hooks arguments".into());
        }
        let storage = match a.first().ok_or("tensor rebuild is missing storage")? {
            Value::Storage(s) => s,
            _ => return Err("tensor rebuild does not reference a persistent storage".into()),
        };
        let offset = u64::try_from(Self::pop_int(
            a.get(1).ok_or("tensor rebuild is missing offset")?.clone(),
        )?)
        .map_err(|_| "negative tensor storage offset")?;
        let shape = ints(a.get(2).ok_or("tensor rebuild is missing shape")?)?;
        let stride = ints(a.get(3).ok_or("tensor rebuild is missing stride")?)?;
        if shape.len() != stride.len() || shape.len() > 4 {
            return Err("unsupported tensor rank or invalid stride rank".into());
        }
        let dtype = if name.ends_with("_v3") {
            dtype_from_value(a.get(6).ok_or("v3 tensor missing dtype")?)?
        } else {
            storage.dtype.clone()
        };
        if storage.dtype != "UNTYPED" && dtype != storage.dtype {
            return Err("tensor dtype does not match typed storage dtype".into());
        }
        let width = width(&dtype);
        let count = shape.iter().try_fold(1u64, |n, &d| {
            n.checked_mul(d).ok_or("tensor shape size overflow")
        })?;
        let out_bytes = count
            .checked_mul(width)
            .ok_or("tensor output byte count overflow")?;
        let storage_bytes = storage.payload.as_slice().len() as u64;
        if out_bytes > storage_bytes {
            return Err("tensor output exceeds storage byte length".into());
        }
        let source_start = offset
            .checked_mul(width)
            .ok_or("tensor source offset overflow")?;
        let is_contiguous = contiguous(&shape, &stride);
        if count == 0 {
            if source_start > storage_bytes {
                return Err("empty tensor offset exceeds storage".into());
            }
            let start = usize::try_from(source_start).map_err(|_| "tensor offset overflow")?;
            return Ok(Tensor {
                name: String::new(),
                dtype,
                shape,
                payload: storage.payload.slice(start..start)?,
            });
        }
        let mut max_elem = offset;
        for i in 0..shape.len() {
            max_elem = max_elem
                .checked_add(
                    (shape[i] - 1)
                        .checked_mul(stride[i])
                        .ok_or("tensor stride overflow")?,
                )
                .ok_or("tensor offset overflow")?;
        }
        let source_end = max_elem
            .checked_add(1)
            .and_then(|x| x.checked_mul(width))
            .ok_or("tensor source range overflow")?;
        if source_end > storage_bytes {
            return Err("tensor view exceeds its storage".into());
        }
        if is_contiguous && !(storage.dtype == "UNTYPED" && !self.little) {
            let end = source_start
                .checked_add(out_bytes)
                .ok_or("tensor byte range overflow")?;
            let range_start =
                usize::try_from(source_start).map_err(|_| "tensor offset overflow")?;
            let range_end = usize::try_from(end).map_err(|_| "tensor end overflow")?;
            return Ok(Tensor {
                name: String::new(),
                dtype,
                shape,
                payload: storage.payload.slice(range_start..range_end)?,
            });
        }
        let mut bytes =
            Vec::with_capacity(usize::try_from(out_bytes).map_err(|_| "tensor output too large")?);
        let mut indices = vec![0u64; shape.len()];
        for _ in 0..count {
            let mut elem = offset;
            for i in 0..shape.len() {
                elem = elem
                    .checked_add(
                        indices[i]
                            .checked_mul(stride[i])
                            .ok_or("tensor stride overflow")?,
                    )
                    .ok_or("tensor offset overflow")?;
            }
            let start = usize::try_from(
                elem.checked_mul(width)
                    .ok_or("tensor byte offset overflow")?,
            )
            .map_err(|_| "tensor byte offset overflow")?;
            let end = start
                .checked_add(width as usize)
                .ok_or("tensor byte range overflow")?;
            let src = storage
                .payload
                .as_slice()
                .get(start..end)
                .ok_or("tensor view exceeds its storage")?;
            if storage.dtype == "UNTYPED" && !self.little {
                bytes.extend(src.iter().rev());
            } else {
                bytes.extend_from_slice(src);
            }
            for d in (0..indices.len()).rev() {
                indices[d] += 1;
                if indices[d] < shape[d] {
                    break;
                }
                indices[d] = 0;
            }
        }
        let len = bytes.len();
        Ok(Tensor {
            name: String::new(),
            dtype,
            shape,
            payload: Payload::Owned(Arc::new(bytes), 0..len),
        })
    }
}
fn make_dict(v: Vec<Value>) -> Result<Value, String> {
    let (pairs, remainder) = v.as_chunks::<2>();
    if !remainder.is_empty() {
        return Err("pickle DICT has odd number of values".into());
    }
    Ok(Value::Dict(
        pairs
            .iter()
            .map(|[key, value]| (key.clone(), value.clone()))
            .collect(),
    ))
}
fn ints(v: &Value) -> Result<Vec<u64>, String> {
    let seq = match v {
        Value::Tuple(v) | Value::List(v) => v,
        _ => return Err("tensor shape/stride is not a tuple".into()),
    };
    seq.iter()
        .map(|x| {
            u64::try_from(Pickle::pop_int(x.clone())?)
                .map_err(|_| "negative tensor shape or stride".into())
        })
        .collect()
}
fn width(dtype: &str) -> u64 {
    match dtype {
        "F32" => 4,
        "UNTYPED" => 1,
        _ => 2,
    }
}
fn contiguous(shape: &[u64], stride: &[u64]) -> bool {
    let mut expected = 1u64;
    for i in (0..shape.len()).rev() {
        if stride[i] != expected {
            return false;
        }
        let Some(next) = expected.checked_mul(shape[i]) else {
            return false;
        };
        expected = next;
    }
    true
}
fn dtype_from_value(v: &Value) -> Result<String, String> {
    let s = match v {
        Value::Global(s) => s.rsplit('.').next().unwrap_or(s),
        Value::Str(s) => s.as_str(),
        _ => return Err("unsupported v3 tensor dtype encoding".into()),
    };
    match s {
        "float32" | "Float" => Ok("F32".into()),
        "float16" | "Half" => Ok("F16".into()),
        "bfloat16" | "BFloat16" => Ok("BF16".into()),
        _ => Err(format!("unsupported v3 tensor dtype {s}")),
    }
}
