//! Models that are not GGUF files: `run`, `serve` and `pull` accept a Hugging Face repository or a local directory that
//! holds a safetensors / PyTorch checkpoint (plain, AWQ, GPTQ, FP8, compressed-tensors, ModelOpt NVFP4), convert it to
//! GGUF once and keep the result in the model cache.
//!
//! Cache layout: a converted model of a Hub repository lives in the snapshot directory of the commit it was made from,
//! `models--owner--repo/snapshots/<commit>/<repo>-<label>-fllm<key>.gguf`, with `<name>.json` beside it describing the
//! source. The key is a hash of the converter version, repository, commit, output type and the weight files, so a
//! second start finds the file, and a new commit, another `--quant` or a newer converter makes a new one. Local
//! directories are converted into `<cache>/fllm-local/`, keyed by the canonical path and the size and modification time
//! of the files.

use crate::hub::{self, CacheEntry, Reference, RepoSession};
use fllm_checkpoint::{ConvertOptions, OutputDType, Preset};
use fllm_gguf_probe::MappedGguf;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex, OnceLock,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

/// Bump when the converter changes in a way that changes the written file: older conversions are then not reused.
pub const IMPORT_VERSION: u32 = 1;

const ARCHITECTURES_URL: &str = "https://github.com/Marco-Lu2508/fLLM/blob/main/docs/architectures.md";

/// How a checkpoint is converted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OutType {
    /// Keep the weights as stored (bf16/f16/f32, or the checkpoint's own quantization).
    Preserve,
    F16,
    F32,
    /// Re-encode the 16-bit weights with an existing `quantize` preset (`q4_k_m`, `q8_0`, ...).
    Quant(String),
}

impl OutType {
    pub fn label(&self) -> String {
        match self {
            Self::Preserve => "preserve".into(),
            Self::F16 => "f16".into(),
            Self::F32 => "f32".into(),
            Self::Quant(q) => q.clone(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ImportOptions {
    pub revision: Option<String>,
    pub offline: bool,
    /// `--quant TYPE`: quantize an unquantized checkpoint while converting.
    pub quant: Option<String>,
    /// `--outtype preserve|f16|f32`.
    pub outtype: Option<String>,
    /// `--direct` / `--zero-copy`: load the checkpoint directly, without a GGUF file, and refuse instead of falling back to
    /// the conversion when that is not possible.
    pub direct: bool,
    /// `--no-direct-load`: always convert to a cached GGUF file first (the behaviour before direct loading existed).
    pub no_direct: bool,
}

impl ImportOptions {
    /// Take one of the flags `--quant`, `--outtype`, `--revision`, `--local-files-only` at `args[*i]`; returns whether it was one.
    pub fn parse_flag(&mut self, args: &[String], i: &mut usize) -> Result<bool, String> {
        let flag = args[*i].as_str();
        let value = |i: &mut usize| -> Result<String, String> {
            *i += 1;
            args.get(*i).cloned().ok_or_else(|| format!("missing value for {flag}"))
        };
        match flag {
            "--quant" | "--quantize" => self.quant = Some(value(i)?),
            "--outtype" => self.outtype = Some(value(i)?),
            "--revision" => self.revision = Some(value(i)?),
            "--local-files-only" | "--offline" => self.offline = true,
            "--direct" | "--zero-copy" => self.direct = true,
            "--no-direct-load" | "--no-direct" => self.no_direct = true,
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// The flags that make a child process (`run` starts `serve`) decide the same way about the same checkpoint.
    pub fn to_args(&self) -> Vec<String> {
        let mut args = Vec::new();
        for (flag, value) in [("--quant", &self.quant), ("--outtype", &self.outtype)] {
            if let Some(v) = value {
                args.push(flag.to_owned());
                args.push(v.clone());
            }
        }
        if self.direct {
            args.push("--direct".into());
        }
        if self.no_direct {
            args.push("--no-direct-load".into());
        }
        args
    }

    pub fn validate(&self) -> Result<(), String> {
        if let Some(q) = &self.quant {
            if Preset::parse(q).is_none() {
                return Err(format!("--quant must be one of {}, not {q:?}", Preset::NAMES));
            }
        }
        if let Some(o) = &self.outtype {
            if !matches!(o.as_str(), "preserve" | "f16" | "f32") {
                return Err(format!("--outtype must be preserve, f16 or f32, not {o:?}"));
            }
        }
        if self.quant.is_some() && self.outtype.is_some() {
            return Err("--quant and --outtype cannot be combined".into());
        }
        if self.direct && self.no_direct {
            return Err("--direct and --no-direct-load cannot be combined".into());
        }
        Ok(())
    }
}

/// What a repository offers.
#[derive(Debug, PartialEq, Eq)]
pub enum RepoKind {
    /// At least one GGUF file (a checkpoint next to it is ignored).
    Gguf,
    /// No GGUF but `config.json` and loadable weights.
    Checkpoint,
    Neither(String),
}

/// Classify a repository by its file names.
pub fn classify_repo(names: &[&str]) -> RepoKind {
    if names.iter().any(|n| n.to_ascii_lowercase().ends_with(".gguf")) {
        return RepoKind::Gguf;
    }
    let has = |n: &str| names.contains(&n);
    let weights = has("model.safetensors")
        || has("model.safetensors.index.json")
        || has("pytorch_model.bin")
        || has("pytorch_model.bin.index.json");
    if has("config.json") && weights {
        return RepoKind::Checkpoint;
    }
    // Mistral's own layout: params.json and consolidated*.safetensors
    if !has("config.json") && has("params.json") && names.iter().any(|n| n.starts_with("consolidated") && n.ends_with(".safetensors")) {
        return RepoKind::Checkpoint;
    }
    let reason = if has("adapter_config.json") {
        "the repository holds a LoRA adapter, not a model: merge it into its base model first (fllm-rs merge-lora)".to_owned()
    } else if has("config.json") {
        "the repository has a config.json but no model.safetensors, model.safetensors.index.json or pytorch_model.bin".to_owned()
    } else {
        "the repository has no GGUF file and no config.json with weights".to_owned()
    };
    RepoKind::Neither(reason)
}

/// Decide the output type from what the user asked for and what the checkpoint is.
pub fn choose_outtype(quantization: &str, quant: Option<&str>, outtype: Option<&str>) -> Result<OutType, String> {
    if let Some(q) = quant {
        Preset::parse(q).ok_or_else(|| format!("--quant must be one of {}, not {q:?}", Preset::NAMES))?;
        if outtype.is_some() {
            return Err("--quant and --outtype cannot be combined".into());
        }
        if quantization != "none" {
            return Err(format!(
                "this checkpoint is already quantized ({quantization}); --quant re-encodes unquantized bf16/f16/f32 checkpoints. Leave --quant out to keep the checkpoint's own quantization"
            ));
        }
        return Ok(OutType::Quant(q.to_ascii_lowercase()));
    }
    Ok(match outtype {
        None | Some("preserve") => OutType::Preserve,
        Some("f16") => OutType::F16,
        Some("f32") => OutType::F32,
        Some(other) => return Err(format!("--outtype must be preserve, f16 or f32, not {other:?}")),
    })
}

/// A stable 12-digit hex key over the given strings.
pub fn cache_key(parts: &[&str]) -> String {
    let mut hash = Sha256::new();
    for part in parts {
        hash.update((part.len() as u64).to_le_bytes());
        hash.update(part.as_bytes());
    }
    let digest = hash.finalize();
    digest.iter().take(6).map(|b| format!("{b:02x}")).collect()
}

/// `<stem>-<label>-fllm<key>.gguf`, the name of a converted model in the cache.
pub fn converted_name(stem: &str, label: &str, key: &str) -> String {
    let clean = |s: &str| {
        s.chars()
            .map(|c| if c.is_ascii_alphanumeric() || "-_.".contains(c) { c } else { '_' })
            .collect::<String>()
    };
    format!("{}-{}-fllm{key}.gguf", clean(stem), clean(label))
}

/// The model name without the cache key: `Llama-3-8B-preserve-fllm89abcdef0123` becomes `Llama-3-8B-preserve`.
pub fn display_name(stem: &str) -> &str {
    if hub::is_converted_name(&format!("{stem}.gguf")) {
        &stem[..stem.len() - 17]
    } else {
        stem
    }
}

fn gb(bytes: u64) -> String {
    format!("{:.2} GB", bytes as f64 / 1e9)
}

/// Disk space needed to download `download` bytes and write the converted model made from `weights` bytes of checkpoint.
pub fn disk_needed(download: u64, weights: u64, outtype: &OutType) -> u64 {
    let converted = match outtype {
        OutType::Preserve | OutType::F16 => weights + weights / 10,
        OutType::F32 => weights * 2 + weights / 5,
        // the 16-bit file is staged next to the quantized result
        OutType::Quant(_) => weights + weights / 2,
    };
    download + converted + (256 << 20)
}

/// Free bytes on the filesystem that holds `dir` (or its nearest existing parent); `None` when unknown.
pub fn free_space(dir: &Path) -> Option<u64> {
    let mut at = dir;
    while !at.exists() {
        at = at.parent()?;
    }
    free_space_of(at)
}

#[cfg(unix)]
fn free_space_of(dir: &Path) -> Option<u64> {
    let out = std::process::Command::new("df").arg("-Pk").arg(dir).output().ok()?;
    parse_df(&String::from_utf8_lossy(&out.stdout))
}

/// Available kilobytes of the last line of `df -Pk`, in bytes.
pub fn parse_df(output: &str) -> Option<u64> {
    let line = output.lines().rfind(|l| !l.trim().is_empty())?;
    let fields: Vec<&str> = line.split_whitespace().collect();
    // Filesystem, 1024-blocks, Used, Available, Capacity, Mounted on
    fields.get(3)?.parse::<u64>().ok()?.checked_mul(1024)
}

#[cfg(windows)]
fn free_space_of(dir: &Path) -> Option<u64> {
    use std::os::windows::ffi::OsStrExt;
    extern "system" {
        fn GetDiskFreeSpaceExW(dir: *const u16, available: *mut u64, total: *mut u64, free: *mut u64) -> i32;
    }
    let wide: Vec<u16> = dir.as_os_str().encode_wide().chain(Some(0)).collect();
    let (mut available, mut total, mut free) = (0u64, 0u64, 0u64);
    // SAFETY: `wide` is a NUL-terminated path and the three out pointers are valid for the call.
    let ok = unsafe { GetDiskFreeSpaceExW(wide.as_ptr(), &mut available, &mut total, &mut free) };
    (ok != 0).then_some(available)
}

#[cfg(not(any(unix, windows)))]
fn free_space_of(_dir: &Path) -> Option<u64> {
    None
}

fn check_disk(dir: &Path, needed: u64, download: u64) -> Result<(), String> {
    match free_space(dir) {
        Some(free) if free < needed => Err(format!(
            "not enough free disk space for the model cache ({}): about {} needed ({} to download, the rest for the converted model), {} free. Free some space or set FLLM_CACHE to a larger disk",
            dir.display(),
            gb(needed),
            gb(download),
            gb(free)
        )),
        _ => Ok(()),
    }
}

/// A hint when a 16-bit model of 7B parameters or more will not fit into the memory of the device.
pub fn memory_hint(model_bytes: u64, free_bytes: u64, outtype: &OutType, quantization: &str) -> Option<String> {
    let sixteen_bit = matches!(outtype, OutType::Preserve | OutType::F16 | OutType::F32) && quantization == "none";
    // 7B parameters in 16 bits are about 13 GB
    if !sixteen_bit || model_bytes < 13_000_000_000 || free_bytes == 0 || model_bytes + model_bytes / 10 <= free_bytes {
        return None;
    }
    Some(format!(
        "this 16-bit model is {} but the device reports {} free: it will not fit or will run mostly on the CPU. Quantize while converting, e.g. --quant q4_k_m (about {}) or --quant q8_0 (about {})",
        gb(model_bytes),
        gb(free_bytes),
        gb(model_bytes * 32 / 100),
        gb(model_bytes * 53 / 100)
    ))
}

fn print_memory_hint(model_bytes: u64, outtype: &OutType, quantization: &str) {
    let free = crate::catalog::available_memory_bytes();
    if let Some(hint) = memory_hint(model_bytes, free, outtype, quantization) {
        eprintln!("[fllm-rs] hint: {hint}");
    }
}

fn read_config(root: &Path) -> Result<Value, String> {
    fllm_checkpoint::read_checkpoint_config(root)
}

/// Check architecture and quantization from `config.json` alone; the error says what to do.
fn check_architecture(root: &Path, what: &str) -> Result<fllm_checkpoint::ConfigCheck, String> {
    fllm_checkpoint::check_config(root).map_err(|e| {
        let text = e.to_string();
        if text.contains("unsupported model_type") {
            let model_type = read_config(root)
                .ok()
                .and_then(|c| c.get("model_type").and_then(Value::as_str).map(str::to_owned))
                .unwrap_or_else(|| "unknown".into());
            format!(
                "{what} cannot be run: its architecture (model_type {model_type:?}) is not supported by fLLM, so nothing else was downloaded or converted. The supported architectures are listed in docs/architectures.md ({ARCHITECTURES_URL}); a GGUF made by llama.cpp is another option if the architecture is supported there"
            )
        } else {
            format!("{what} cannot be converted: {text}")
        }
    })
}

/// Tokenizer files the conversion can build the GGUF vocabulary from.
pub(crate) fn has_tokenizer_source(names: &[&str]) -> bool {
    names.iter().any(|n| {
        matches!(
            *n,
            "tokenizer.json" | "tekken.json" | "tiktoken.model" | "qwen.tiktoken" | "tokenizer.model" | "spiece.model" | "sentencepiece.bpe.model"
        ) || n.ends_with(".tiktoken")
    })
}

fn require_tokenizer(has_tokenizer: bool, what: &str) -> Result<(), String> {
    if has_tokenizer {
        Ok(())
    } else {
        Err(format!(
            "{what} has no tokenizer the conversion can use to build the GGUF vocabulary: expected tokenizer.json, tekken.json, a tiktoken rank file (tiktoken.model, qwen.tiktoken, *.tiktoken) or a SentencePiece model (tokenizer.model, spiece.model, sentencepiece.bpe.model)"
        ))
    }
}

/// `bf16`, `f16`, `f32` from the `torch_dtype` / `dtype` of config.json, else `native`.
fn source_dtype_label(cfg: &Value) -> &'static str {
    let dtype = cfg
        .get("torch_dtype")
        .or_else(|| cfg.get("dtype"))
        .or_else(|| cfg.get("text_config").and_then(|t| t.get("torch_dtype").or_else(|| t.get("dtype"))))
        .and_then(Value::as_str)
        .unwrap_or("");
    match dtype {
        "bfloat16" => "bf16",
        "float16" => "f16",
        "float32" => "f32",
        _ => "native",
    }
}

/// The part of the file name that says what is inside: the quantization of a quantized checkpoint, else the dtype.
fn label_for(outtype: &OutType, check: &fllm_checkpoint::ConfigCheck, cfg: &Value) -> String {
    match outtype {
        OutType::Preserve if check.quantization != "none" => check.quantization.clone(),
        OutType::Preserve => source_dtype_label(cfg).to_owned(),
        other => other.label(),
    }
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn sidecar_path(gguf: &Path) -> PathBuf {
    let mut name = gguf.as_os_str().to_owned();
    name.push(".json");
    PathBuf::from(name)
}

fn lock_path(gguf: &Path) -> PathBuf {
    let mut name = gguf.as_os_str().to_owned();
    name.push(".fllm-lock");
    PathBuf::from(name)
}

fn part_path(gguf: &Path, stage: &str) -> PathBuf {
    let mut name = gguf.as_os_str().to_owned();
    name.push(format!("{stage}.fllm-part"));
    PathBuf::from(name)
}

/// Remove what an interrupted conversion of `final_path` left behind. The caller holds the conversion lock.
pub fn remove_partial_files(final_path: &Path) {
    let (Some(dir), Some(name)) = (final_path.parent(), final_path.file_name().and_then(|n| n.to_str())) else {
        return;
    };
    let prefix = format!("{name}.");
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let file = entry.file_name().to_string_lossy().into_owned();
        if file.starts_with(&prefix) && file.contains(".fllm-part") {
            let _ = fs::remove_file(entry.path());
        }
    }
}

struct Heartbeat {
    stop: std::sync::Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Heartbeat {
    fn start(message: String) -> Self {
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let thread = std::thread::spawn(move || {
            let started = Instant::now();
            let mut last = Instant::now();
            while !flag.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(200));
                if last.elapsed() >= Duration::from_secs(15) {
                    eprintln!("[fllm-rs] {message} ({} s so far)", started.elapsed().as_secs());
                    last = Instant::now();
                }
            }
        });
        Self { stop, thread: Some(thread) }
    }
}

impl Drop for Heartbeat {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// What is written into `<name>.json` beside a converted model.
struct Origin<'a> {
    source: &'a str,
    revision: &'a str,
    outtype: &'a OutType,
    check: &'a fllm_checkpoint::ConfigCheck,
}

/// Convert `input` (checkpoint directory or weights file) into `final_path`, atomically: nothing but the finished file
/// ever carries the final name, a partial file is removed on failure, and partial files of an interrupted run are removed
/// by the next one. Concurrent starts of the same conversion wait for each other.
fn convert_into(input: &Path, final_path: &Path, origin: &Origin<'_>) -> Result<(), String> {
    let dir = final_path.parent().ok_or("cache path has no parent")?;
    fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let lock = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(lock_path(final_path))
        .map_err(|e| format!("cannot lock the conversion: {e}"))?;
    if lock.try_lock().is_err() {
        eprintln!("[fllm-rs] another fllm-rs process is converting this model; waiting for it");
        lock.lock().map_err(|e| format!("cannot lock the conversion: {e}"))?;
    }
    if final_path.is_file() {
        return Ok(()); // another process finished it while we waited
    }
    remove_partial_files(final_path);
    let partial = part_path(final_path, "");
    let result = (|| -> Result<(), String> {
        let (dtype, preset) = dtype_and_preset(origin.outtype);
        let started = Instant::now();
        let _beat = Heartbeat::start(format!("still converting {}", origin.source));
        let stage = if preset.is_some() { part_path(final_path, ".f16") } else { partial.clone() };
        let report = fllm_checkpoint::convert_checkpoint(input, &stage, ConvertOptions { output_dtype: dtype, overwrite: true, spec_head: None })
            .map_err(|e| format!("checkpoint conversion failed: {e}"))?;
        if let Some(preset) = preset {
            eprintln!("[fllm-rs] quantizing the converted model to {}", origin.outtype.label());
            fllm_checkpoint::quantize_gguf(&stage, &partial, preset).map_err(|e| format!("quantization failed: {e}"))?;
            let _ = fs::remove_file(&stage);
        }
        fs::rename(&partial, final_path).map_err(|e| format!("cannot publish the converted model: {e}"))?;
        let meta = json!({
            "source": origin.source,
            "revision": origin.revision,
            "outtype": origin.outtype.label(),
            "architecture": report.architecture,
            "source_quantization": origin.check.quantization,
            "import_version": IMPORT_VERSION,
            "converter": crate::version::long_version(),
            "created": now_secs(),
        });
        let _ = fs::write(sidecar_path(final_path), meta.to_string());
        eprintln!(
            "[fllm-rs] converted {} tensors ({}) in {:.0} s",
            report.tensors_written,
            gb(report.bytes_written),
            started.elapsed().as_secs_f64()
        );
        Ok(())
    })();
    if result.is_err() {
        remove_partial_files(final_path);
        let _ = fs::remove_file(&partial);
    }
    drop(lock);
    result
}

/// The tensor type the converter writes for `outtype`, and the preset that re-encodes it afterwards.
fn dtype_and_preset(outtype: &OutType) -> (OutputDType, Option<Preset>) {
    match outtype {
        OutType::Preserve => (OutputDType::Preserve, None),
        OutType::F16 => (OutputDType::F16, None),
        OutType::F32 => (OutputDType::F32, None),
        OutType::Quant(q) => (OutputDType::F16, Preset::parse(q)),
    }
}

// ---- loading a checkpoint without writing a GGUF file ----

/// Source quantizations whose direct load is checked, tensor by tensor, against the conversion by tests (see
/// `docs/rust-formats-and-import.md`). Other checkpoints are converted unless `--direct` asks for the direct load anyway.
const VERIFIED_DIRECT: &[&str] = &["none", "awq", "gptq", "nvfp4", "fp8", "int4", "int8", "mxfp4", "bitsandbytes", "mlx", "exl2", "exl3"];

fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| !v.is_empty() && v != "0")
}

/// How the checkpoints that `prepare` handed out for direct loading were asked to be loaded (`--outtype`, `--quant`).
fn direct_registry() -> &'static Mutex<HashMap<PathBuf, OutType>> {
    static REGISTRY: OnceLock<Mutex<HashMap<PathBuf, OutType>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn hand_out_direct(input: PathBuf, outtype: &OutType) -> PathBuf {
    direct_registry().lock().unwrap_or_else(|e| e.into_inner()).insert(input.clone(), outtype.clone());
    input
}

/// Whether `prepare` returned this path for direct loading: a checkpoint directory (which may hold converted GGUF files
/// of earlier runs) or a weights file, to be opened with `open_model`.
pub fn is_direct(path: &Path) -> bool {
    direct_registry().lock().unwrap_or_else(|e| e.into_inner()).contains_key(path)
}

/// A name for a model that is loaded directly: the repository of a cached Hub snapshot, else the directory or file stem.
pub fn direct_model_name(path: &Path) -> String {
    let repo = path
        .parent()
        .and_then(Path::parent)
        .and_then(Path::file_name)
        .and_then(|n| n.to_str())
        .and_then(|n| n.strip_prefix("models--"))
        .and_then(|n| n.rsplit("--").next())
        .filter(|n| !n.is_empty());
    if let (Some(repo), true) = (repo, path.is_dir()) {
        return repo.to_owned();
    }
    let name = if path.is_dir() { path.file_name() } else { path.file_stem() };
    name.and_then(|n| n.to_str()).filter(|n| !n.is_empty()).unwrap_or("fllm-model").to_owned()
}

/// Whether `open_model` reads `path` as a checkpoint: a `.safetensors` file or a directory with `config.json` / `params.json`.
pub fn is_checkpoint_input(path: &Path) -> bool {
    (path.is_file() && path.extension().is_some_and(|e| e.eq_ignore_ascii_case("safetensors")))
        || (path.is_dir() && (path.join("config.json").is_file() || path.join("params.json").is_file()))
}

/// Open a model for the engine: a GGUF file is memory-mapped, a checkpoint (`prepare` hands one out when it loads
/// directly) is presented as the GGUF model the converter would have written, without writing it; see
/// `fllm_checkpoint::load_checkpoint_quantized`.
pub fn open_model(path: &Path) -> Result<MappedGguf, String> {
    if is_checkpoint_input(path) {
        return load_direct(path);
    }
    match MappedGguf::open(path) {
        Ok(mapping) => Ok(mapping),
        Err(e) if path.exists() => load_direct(path).map_err(|_| e.to_string()),
        Err(e) => Err(e.to_string()),
    }
}

fn load_direct(path: &Path) -> Result<MappedGguf, String> {
    let outtype = direct_registry().lock().unwrap_or_else(|e| e.into_inner()).get(path).cloned().unwrap_or(OutType::Preserve);
    let (output_dtype, preset) = dtype_and_preset(&outtype);
    // tensors that cannot stay slices of the checkpoint are built in memory that the system cannot take back under pressure
    // (a GGUF file would sit in the evictable page cache), so refuse early instead of driving the machine into swap
    let free = crate::devices::host_free_memory_bytes();
    let limit = (free > 0).then(|| free / 4 * 3);
    fllm_checkpoint::load_checkpoint_quantized(path, ConvertOptions { output_dtype, overwrite: false, spec_head: None }, preset, limit).map_err(|e| match e {
        fllm_checkpoint::CheckpointError::OverBudget { .. } => format!("{e}; --no-direct-load converts to a cached GGUF file instead"),
        other => other.to_string(),
    })
}

/// Why this checkpoint is not loaded directly by default, or `None` when it is.
fn direct_unsupported(check: &fllm_checkpoint::ConfigCheck, weight_bytes: u64, outtype: &OutType) -> Option<String> {
    if !VERIFIED_DIRECT.contains(&check.quantization.as_str()) {
        return Some(format!("the direct load of {} checkpoints is not verified yet", check.quantization));
    }
    // PyTorch ZIP weights (.bin/.pt/.pth) are read by the restricted reader (it never runs pickle code) and load directly as
    // well: stored storages are slices of the mapped archive, compressed ones are built
    // the converted tensors live in memory, the converted file would live in the page cache; formats the converter expands
    // to 16-bit weights need several times the size of the files
    let free = crate::devices::host_free_memory_bytes();
    let (expansion, share) = match check.quantization.as_str() {
        "none" => (1, free / 4 * 3),
        // (bitsandbytes NF4 / FP4 / int8 stay packed in their own types; a tensor whose rows are not a multiple of 64 values is expanded, and the
        // memory limit of the load stops that before it is built)
        "mlx" => (4, free / 4 * 3),
        "exl2" | "exl3" => (5, free / 4 * 3),
        _ => (1, free / 2),
    };
    let needed = weight_bytes.saturating_mul(expansion);
    if free > 0 && !matches!(outtype, OutType::Quant(_)) && needed > share {
        return Some(format!("the weights ({}{}) are large for the free memory ({})", gb(weight_bytes), if expansion > 1 { format!(", about {} expanded", gb(needed)) } else { String::new() }, gb(free)));
    }
    None
}

/// What to do with a checkpoint: `Some(forced)` loads it directly (`forced` when `--direct` asked for it), `None` converts it.
fn direct_mode(opts: &ImportOptions, unsupported: Option<String>, what: &str) -> Option<bool> {
    if opts.no_direct || env_flag("FLLM_NO_DIRECT_LOAD") {
        return None;
    }
    if opts.direct || env_flag("FLLM_DIRECT_SAFETENSORS") {
        if let Some(reason) = unsupported {
            eprintln!("[fllm-rs] loading {what} directly although {reason}");
        }
        return Some(true);
    }
    match unsupported {
        None => Some(false),
        Some(reason) => {
            eprintln!("[fllm-rs] {what} is converted to a GGUF file: {reason} (--direct loads it anyway)");
            None
        }
    }
}

/// Resolve what `run`, `serve` and `pull` were given to a GGUF file (or, for a local directory with GGUF files, that
/// directory): downloads and converts a checkpoint when there is no GGUF.
pub fn prepare(reference: &str, opts: &ImportOptions) -> Result<PathBuf, String> {
    opts.validate()?;
    let normalized = hub::normalize_reference(reference);
    if normalized.starts_with("hf://") {
        let parsed = hub::parse_reference(&normalized)?;
        if parsed.file.is_some() || parsed.quant.is_some() {
            if opts.quant.is_some() || opts.outtype.is_some() {
                return Err("--quant and --outtype apply to checkpoint repositories; this reference already selects a GGUF file".into());
            }
            return hub::resolve(&normalized, opts.revision.as_deref(), opts.offline);
        }
        return prepare_hub(&parsed, opts);
    }
    prepare_local(Path::new(&normalized), opts)
}

fn prepare_hub(reference: &Reference, opts: &ImportOptions) -> Result<PathBuf, String> {
    let revision = opts.revision.as_deref().unwrap_or("main");
    let repo_name = format!("{}/{}", reference.owner, reference.repo);
    let session = RepoSession::open(reference, revision, opts.offline)?;
    let names: Vec<String> = session.names().into_iter().map(str::to_owned).collect();
    let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
    match classify_repo(&name_refs) {
        RepoKind::Gguf => {
            if opts.quant.is_some() || opts.outtype.is_some() {
                return Err(format!("{repo_name} holds GGUF files; --quant and --outtype apply to checkpoint repositories. Select a file or :QUANT instead"));
            }
            return hub::resolve_with_session(&session, reference, opts.offline, true);
        }
        RepoKind::Neither(reason) => return Err(format!("{repo_name} cannot be run: {reason}")),
        RepoKind::Checkpoint => {}
    }
    // 1. architecture and quantization from config.json, before any weights are fetched
    session.download(if name_refs.contains(&"config.json") { "config.json" } else { "params.json" })?;
    if name_refs.contains(&"hf_quant_config.json") {
        session.download("hf_quant_config.json")?;
    }
    let check = check_architecture(&session.snapshot, &repo_name)?;
    require_tokenizer(has_tokenizer_source(&name_refs), &repo_name)?;
    let outtype = choose_outtype(&check.quantization, opts.quant.as_deref(), opts.outtype.as_deref())?;
    let cfg = read_config(&session.snapshot)?;
    // 2. the files that make up the checkpoint
    let weights = hub::checkpoint_weight_files(&session)?;
    let weight_bytes: u64 = weights.iter().filter_map(|w| session.size_of(w)).sum();
    let listing: Vec<String> = weights.iter().map(|w| format!("{w}:{}", session.size_of(w).unwrap_or(0))).collect();
    let version = env!("CARGO_PKG_VERSION");
    let import_version = IMPORT_VERSION.to_string();
    let label = outtype.label();
    let key = cache_key(&[
        "fllm-import",
        &import_version,
        version,
        &repo_name,
        &session.commit,
        &label,
        &listing.join(","),
    ]);
    let final_path = session
        .snapshot
        .join(converted_name(&reference.repo, &label_for(&outtype, &check, &cfg), &key));
    // a conversion that is already in the cache starts faster than building the tensors again, so it is used unless --direct
    let direct = direct_mode(opts, direct_unsupported(&check, weight_bytes, &outtype), &repo_name);
    if direct != Some(true) && final_path.is_file() {
        session.record_revision()?;
        eprintln!("[fllm-rs] using the converted model from the cache: {}", final_path.display());
        print_memory_hint(fs::metadata(&final_path).map_or(0, |m| m.len()), &outtype, &check.quantization);
        return Ok(final_path);
    }
    // 3. disk space and a plan the user can read
    let missing: u64 = weights
        .iter()
        .map(String::as_str)
        .chain(hub::CHECKPOINT_SUPPORT_FILES.iter().copied().filter(|f| name_refs.contains(f)))
        .filter(|f| !session.is_cached(f))
        .filter_map(|f| session.size_of(f))
        .sum();
    if direct.is_some() {
        check_disk(&session.snapshot, missing, missing)?;
        eprintln!(
            "[fllm-rs] {repo_name} is a {} checkpoint without GGUF files ({}, {} of weights); it is loaded directly, without writing a GGUF file",
            if check.quantization == "none" { "plain".to_owned() } else { check.quantization.clone() },
            check.architecture,
            gb(weight_bytes)
        );
        if missing > 0 {
            eprintln!("[fllm-rs] downloading {} ...", gb(missing));
        }
        for file in hub::CHECKPOINT_SUPPORT_FILES.iter().filter(|f| name_refs.contains(f)) {
            session.download(file)?;
        }
        for file in &weights {
            session.download(file)?;
        }
        session.record_revision()?;
        return Ok(hand_out_direct(session.snapshot.clone(), &outtype));
    }
    check_disk(&session.snapshot, disk_needed(missing, weight_bytes, &outtype), missing)?;
    eprintln!(
        "[fllm-rs] {repo_name} is a {} checkpoint without GGUF files ({}, {} of weights); it is converted to GGUF ({}) once and the result is cached under {}",
        if check.quantization == "none" { "plain".to_owned() } else { check.quantization.clone() },
        check.architecture,
        gb(weight_bytes),
        label_for(&outtype, &check, &cfg),
        session.snapshot.display()
    );
    if missing > 0 {
        eprintln!("[fllm-rs] downloading {} ...", gb(missing));
    }
    for file in hub::CHECKPOINT_SUPPORT_FILES.iter().filter(|f| name_refs.contains(f)) {
        session.download(file)?;
    }
    for file in &weights {
        session.download(file)?;
    }
    session.record_revision()?;
    print_memory_hint(weight_bytes, &outtype, &check.quantization);
    eprintln!("[fllm-rs] converting {repo_name} to GGUF ... this happens once");
    convert_into(
        &session.snapshot,
        &final_path,
        &Origin { source: &format!("hf://{repo_name}"), revision: &session.commit, outtype: &outtype, check: &check },
    )?;
    eprintln!("[fllm-rs] ready: {}", final_path.display());
    Ok(final_path)
}

/// Directory for conversions of local checkpoint directories.
pub fn local_cache_dir() -> PathBuf {
    hub::cache_root().join("fllm-local")
}

fn is_weight_name(name: &str) -> bool {
    name.ends_with(".safetensors") || name.ends_with(".bin") || name.ends_with(".pt") || name.ends_with(".pth")
}

fn is_checkpoint_file(path: &Path) -> bool {
    matches!(path.extension().and_then(|e| e.to_str()), Some("safetensors" | "bin" | "pt" | "pth"))
}

/// Whether `dir` holds a checkpoint (config.json and weights) and no GGUF.
fn local_checkpoint_root(path: &Path) -> Option<(PathBuf, PathBuf)> {
    if path.is_file() && is_checkpoint_file(path) {
        let root = path.parent()?.to_path_buf();
        return (root.join("config.json").is_file() || root.join("params.json").is_file()).then(|| (root, path.to_path_buf()));
    }
    if !path.is_dir() {
        return None;
    }
    let entries: Vec<_> = fs::read_dir(path).ok()?.flatten().collect();
    let has_gguf = entries.iter().any(|e| {
        e.path().extension().is_some_and(|x| x.eq_ignore_ascii_case("gguf")) && !hub::is_converted_name(&e.file_name().to_string_lossy())
    });
    let has_weights = entries.iter().any(|e| {
        let n = e.file_name().to_string_lossy().into_owned();
        n.ends_with(".safetensors") || n.ends_with(".safetensors.index.json") || n == "pytorch_model.bin" || n == "pytorch_model.bin.index.json"
    });
    (!has_gguf && has_weights && (path.join("config.json").is_file() || path.join("params.json").is_file())).then(|| (path.to_path_buf(), path.to_path_buf()))
}

/// Size and modification time of the files that decide what a local conversion contains.
fn local_fingerprint(root: &Path) -> Result<(Vec<String>, u64), String> {
    let mut parts = Vec::new();
    let mut weight_bytes = 0u64;
    let mut names: Vec<_> = fs::read_dir(root)
        .map_err(|e| format!("cannot read {}: {e}", root.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .collect();
    names.sort();
    for path in names {
        let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
        let weight = is_weight_name(&name);
        let relevant = weight || matches!(name.as_str(), "config.json" | "hf_quant_config.json" | "tokenizer.json" | "tokenizer_config.json" | "generation_config.json" | "chat_template.jinja" | "chat_template.json" | "params.json" | "tekken.json" | "tokenizer.model" | "tiktoken.model" | "qwen.tiktoken" | "spiece.model" | "sentencepiece.bpe.model")
            || name.ends_with(".tiktoken")
            || name.ends_with(".index.json");
        if !relevant {
            continue;
        }
        let meta = fs::metadata(&path).map_err(|e| e.to_string())?;
        let modified = meta.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs());
        if weight {
            weight_bytes += meta.len();
        }
        parts.push(format!("{name}:{}:{modified}", meta.len()));
    }
    Ok((parts, weight_bytes))
}

fn prepare_local(path: &Path, opts: &ImportOptions) -> Result<PathBuf, String> {
    let Some((root, input)) = local_checkpoint_root(path) else {
        if opts.quant.is_some() && path.exists() {
            return Err(format!("--quant applies to checkpoint directories; {} is not one", path.display()));
        }
        return Ok(path.to_path_buf());
    };
    let what = root.display().to_string();
    let check = check_architecture(&root, &what)?;
    let local_names: Vec<String> = fs::read_dir(&root)
        .map(|entries| entries.flatten().filter_map(|e| e.file_name().into_string().ok()).collect())
        .unwrap_or_default();
    let local_refs: Vec<&str> = local_names.iter().map(String::as_str).collect();
    require_tokenizer(has_tokenizer_source(&local_refs), &what)?;
    let outtype = choose_outtype(&check.quantization, opts.quant.as_deref(), opts.outtype.as_deref())?;
    let (files, weight_bytes) = local_fingerprint(&root)?;
    let direct = direct_mode(opts, direct_unsupported(&check, weight_bytes, &outtype), &what);
    let load_directly = |input: PathBuf| {
        eprintln!("[fllm-rs] loading {what} directly, without writing a GGUF file");
        hand_out_direct(input, &outtype)
    };
    if direct == Some(true) {
        return Ok(load_directly(input));
    }
    let cfg = read_config(&root)?;
    let canonical = fs::canonicalize(&input).unwrap_or_else(|_| input.clone());
    let label = outtype.label();
    let canonical_text = canonical.to_string_lossy().into_owned();
    let import_version = IMPORT_VERSION.to_string();
    let key = cache_key(&[
        "fllm-import-local",
        &import_version,
        env!("CARGO_PKG_VERSION"),
        &canonical_text,
        &label,
        &files.join(","),
    ]);
    let stem = root.file_name().and_then(|n| n.to_str()).unwrap_or("model");
    let stem = if stem.is_empty() || stem == "." { "model" } else { stem };
    let final_path = local_cache_dir().join(converted_name(stem, &label_for(&outtype, &check, &cfg), &key));
    if final_path.is_file() {
        eprintln!("[fllm-rs] using the converted model from the cache: {}", final_path.display());
        print_memory_hint(fs::metadata(&final_path).map_or(0, |m| m.len()), &outtype, &check.quantization);
        return Ok(final_path);
    }
    if direct.is_some() {
        return Ok(load_directly(input));
    }
    check_disk(&local_cache_dir(), disk_needed(0, weight_bytes, &outtype), 0)?;
    eprintln!(
        "[fllm-rs] {what} is a {} checkpoint without GGUF files ({}, {} of weights); converting to GGUF ({}), cached under {}",
        if check.quantization == "none" { "plain".to_owned() } else { check.quantization.clone() },
        check.architecture,
        gb(weight_bytes),
        label_for(&outtype, &check, &cfg),
        local_cache_dir().display()
    );
    print_memory_hint(weight_bytes, &outtype, &check.quantization);
    eprintln!("[fllm-rs] converting {what} to GGUF ... this happens once");
    convert_into(
        &input,
        &final_path,
        &Origin { source: &canonical_text, revision: "-", outtype: &outtype, check: &check },
    )?;
    eprintln!("[fllm-rs] ready: {}", final_path.display());
    Ok(final_path)
}

/// Details of a converted model read from the metadata file beside it.
pub fn converted_note(gguf: &Path) -> Option<String> {
    let meta: Value = serde_json::from_slice(&fs::read(sidecar_path(gguf)).ok()?).ok()?;
    Some(format!(
        "converted from {} ({}, source quantization {})",
        meta.get("source")?.as_str()?,
        meta.get("outtype")?.as_str()?,
        meta.get("source_quantization").and_then(Value::as_str).unwrap_or("none")
    ))
}

/// Converted models of local directories as cache entries.
pub fn list_local_conversions() -> Vec<CacheEntry> {
    let Ok(entries) = fs::read_dir(local_cache_dir()) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if !hub::is_converted_name(&name) || !path.is_file() {
            continue;
        }
        let meta: Option<Value> = fs::read(sidecar_path(&path)).ok().and_then(|b| serde_json::from_slice(&b).ok());
        let source = meta.as_ref().and_then(|m| m.get("source")).and_then(Value::as_str).unwrap_or("unknown source").to_owned();
        out.push(CacheEntry {
            reference: source,
            revision: "-".into(),
            bytes: fs::metadata(&path).map_or(0, |m| m.len()),
            note: converted_note(&path).unwrap_or_else(|| "converted from a local checkpoint".into()),
            path,
        });
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

/// Remove the conversions made from the local directory `source`; returns the number of files removed.
pub fn remove_local_conversions(source: &Path) -> Result<usize, String> {
    let wanted = fs::canonicalize(source).unwrap_or_else(|_| source.to_path_buf());
    let wanted = wanted.to_string_lossy().into_owned();
    let mut removed = 0;
    for entry in list_local_conversions() {
        let matches = entry.reference == wanted
            || Path::new(&entry.reference).parent().is_some_and(|p| p.to_string_lossy() == wanted);
        if !matches {
            continue;
        }
        let lock = fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(lock_path(&entry.path)).map_err(|e| e.to_string())?;
        lock.try_lock().map_err(|e| format!("the conversion is in use: {e}"))?;
        for path in [entry.path.clone(), sidecar_path(&entry.path)] {
            if path.exists() {
                fs::remove_file(&path).map_err(|e| format!("cannot remove {}: {e}", path.display()))?;
                removed += 1;
            }
        }
        remove_partial_files(&entry.path);
        drop(lock);
        let _ = fs::remove_file(lock_path(&entry.path));
    }
    if removed == 0 {
        return Err(format!("no converted model of {} in the cache", source.display()));
    }
    Ok(removed)
}

#[cfg(test)]
mod tests;
