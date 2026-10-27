//! Model management of the Studio: load options, unload (also after idle time), the native file picker, the download queue,
//! the models folder, the log view and model cards.
use super::*;
use axum::extract::Query;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::time::Duration;

// ------------------------------------------------------------------------------------------------------------------------------
// Load options
// ------------------------------------------------------------------------------------------------------------------------------

/// How a model is loaded. `None` means "automatic": the flag is not passed and the server decides.
#[derive(Clone, Debug, Default, PartialEq)]
pub(in crate::server) struct LoadOptions {
    pub context: Option<usize>,
    /// `all` or a number
    pub gpu_layers: Option<String>,
    pub threads: Option<usize>,
    pub kv_cache_type: Option<String>,
    pub max_seqs: Option<usize>,
    /// A `--device` value (`auto`, `cpu`, `vulkan:1`, `npu`); `None` keeps the start option of the server.
    pub device: Option<String>,
    /// `all`: the NPU computes every matrix product it can, also the single rows of the answer (default: only the long prompt blocks).
    pub npu_mode: Option<String>,
    /// Speculative decoding, a start option of the server: `off`, `lookup` (n-gram prompt lookup), `draft` (a small model of the
    /// same family) or `mtp` (the multi-token prediction heads of the model). `None` keeps the start options of the server.
    pub speculative: Option<String>,
    /// The draft model (a GGUF path) for `draft`.
    pub draft_model: Option<String>,
    /// Tokens proposed per step for `lookup` and `draft`.
    pub speculative_tokens: Option<usize>,
}

/// Architectures with recurrent layers: rejected draft tokens would advance the recurrent state, so they cannot speculate.
const STATEFUL_ARCHITECTURES: &[&str] = &[
    "qwen3next", "qwen35", "qwen35moe", "olmo_hybrid", "mamba", "mamba2", "jamba", "granitehybrid", "bamba", "nemotron_h",
    "nemotron_h_moe", "falcon-h1", "zamba2", "lfm2", "lfm2moe", "rwkv6", "rwkv7",
];

const CONTEXT_FLAGS: &[&str] = &["--ctx", "--ctx-size", "--max-model-len"];
const GPU_LAYER_FLAGS: &[&str] = &["--gpu-layers", "-ngl", "--n-gpu-layers"];
const THREAD_FLAGS: &[&str] = &["--threads"];
const KV_FLAGS: &[&str] = &["--kv-cache-type", "--kv-cache-dtype", "--cache-type-k", "-ctk", "--cache-type-v", "-ctv"];
const SEQ_FLAGS: &[&str] = &["--max-seqs"];
const DEVICE_FLAGS: &[&str] = &["--device"];
const NPU_MODE_FLAGS: &[&str] = &["--npu-mode"];
const SPECULATIVE_FLAGS: &[&str] = &[
    "--speculative", "-sp", "--speculative-lookup", "--speculative-model", "--draft-model", "--model-draft",
    "--num-speculative-tokens", "--draft-tokens",
];

static NULL: Value = Value::Null;

fn is_auto(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::String(text) => text.trim().is_empty() || text.trim().eq_ignore_ascii_case("auto"),
        _ => false,
    }
}

fn whole_number(value: &Value, what: &str, min: u64, max: u64) -> Result<usize, String> {
    let number = match value {
        Value::Number(n) => n.as_u64(),
        Value::String(text) => text.trim().parse::<u64>().ok(),
        _ => None,
    };
    match number {
        Some(n) if (min..=max).contains(&n) => Ok(n as usize),
        _ => Err(format!("{what} must be a whole number from {min} to {max}, or auto.")),
    }
}

impl LoadOptions {
    /// Reads the options of a request; unknown or impossible values are turned down with a plain message.
    pub fn from_json(value: &Value) -> Result<Self, String> {
        let get = |key: &str| value.get(key).unwrap_or(&NULL);
        let mut options = LoadOptions::default();
        if !is_auto(get("context")) {
            options.context = Some(whole_number(get("context"), "The context length", 128, 4_194_304)?);
        }
        if !is_auto(get("gpu_layers")) {
            let layers = get("gpu_layers");
            options.gpu_layers = Some(match layers.as_str() {
                Some(text) if text.trim().eq_ignore_ascii_case("all") => "all".to_owned(),
                _ => whole_number(layers, "The number of GPU layers", 0, 1024)?.to_string(),
            });
        }
        if !is_auto(get("threads")) {
            options.threads = Some(whole_number(get("threads"), "The number of threads", 1, 1024)?);
        }
        if !is_auto(get("kv_cache_type")) {
            let name = get("kv_cache_type").as_str().unwrap_or_default().trim().to_ascii_lowercase();
            if crate::kv_format::KvFormat::parse(&name).is_none() {
                return Err(format!("The memory type \"{name}\" is not known. Choose one of: auto, {}.", crate::kv_format::KvFormat::NAMES));
            }
            options.kv_cache_type = Some(name);
        }
        if !is_auto(get("max_seqs")) {
            options.max_seqs = Some(whole_number(get("max_seqs"), "The number of parallel requests", 1, 256)?);
        }
        // unlike the other fields `auto` is kept here: it removes a `--device` the server was started with
        if let Some(name) = get("device").as_str().map(|s| s.trim().to_ascii_lowercase()).filter(|s| !s.is_empty()) {
            let plain = name.starts_with(|c: char| c.is_ascii_alphabetic()) && name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, ':' | '-' | '_'));
            match crate::devices::parse_spec(&name) {
                Ok(_) if plain => options.device = Some(name),
                _ => return Err(format!("The device \"{name}\" is not known. Use auto, cpu, or a device from the Hardware page.")),
            }
        }
        match get("npu_mode").as_str().map(|s| s.trim().to_ascii_lowercase()).as_deref() {
            None | Some("") | Some("auto") | Some("prompt") => {}
            Some("all") => options.npu_mode = Some("all".to_owned()),
            Some(other) => return Err(format!("The NPU mode \"{other}\" is not known. Use prompt or all.")),
        }
        options.read_speculative(value)?;
        Ok(options)
    }

    fn read_speculative(&mut self, value: &Value) -> Result<(), String> {
        let get = |key: &str| value.get(key).unwrap_or(&NULL);
        let mode = get("speculative").as_str().map(|s| s.trim().to_ascii_lowercase());
        let mode = match mode.as_deref() {
            None | Some("") | Some("auto") => return Ok(()),
            Some("off") => "off",
            Some("lookup") | Some("ngram") => "lookup",
            Some("suffix") => "suffix",
            Some("draft") => "draft",
            Some("mtp") => "mtp",
            Some(other) => return Err(format!("The speculative decoding mode \"{other}\" is not known. Use off, lookup, suffix, draft or mtp.")),
        };
        self.speculative = Some(mode.to_owned());
        if mode == "draft" {
            let model = get("draft_model").as_str().map(str::trim).unwrap_or_default();
            if model.is_empty() {
                return Err("Choose the small model that drafts for speculative decoding, or pick another mode.".into());
            }
            self.draft_model = Some(model.to_owned());
        }
        if matches!(mode, "lookup" | "suffix" | "draft") && !is_auto(get("speculative_tokens")) {
            self.speculative_tokens = Some(whole_number(get("speculative_tokens"), "The number of tokens proposed per step", 1, 32)?);
        }
        Ok(())
    }

    /// Whether the options ask for an NPU.
    pub fn wants_npu(&self) -> bool {
        self.device.as_deref().is_some_and(|d| d == "npu" || d.starts_with("npu:") || d == "openvino")
    }

    /// `null` stands for automatic.
    pub fn to_json(&self) -> Value {
        json!({
            "context": self.context,
            "gpu_layers": self.gpu_layers,
            "threads": self.threads,
            "kv_cache_type": self.kv_cache_type,
            "max_seqs": self.max_seqs,
            "device": self.device,
            "npu_mode": self.npu_mode,
            "speculative": self.speculative,
            "draft_model": self.draft_model,
            "speculative_tokens": self.speculative_tokens,
        })
    }

    pub fn is_automatic(&self) -> bool {
        *self == LoadOptions::default()
    }

    /// Refuses combinations that cannot work on `device` (the `--device` value the server runs with).
    pub fn check_with_device(&self, device: &str) -> Result<(), String> {
        // a device chosen in the options replaces the one the server started with; `vulkan:1` names the second device of the Vulkan backend
        let device = self.device.as_deref().unwrap_or(device);
        let device = device.split(':').next().unwrap_or_default().to_ascii_lowercase();
        if let Some(format) = self.kv_cache_type.as_deref().and_then(crate::kv_format::KvFormat::parse) {
            if matches!(device.as_str(), "vulkan" | "dx12" | "metal") && !format.is_legacy() {
                return Err(format!(
                    "The {device} graphics backend can only keep the conversation memory as f16 or q8_0. Choose one of those, or run on the processor."
                ));
            }
        }
        if self.speculative.as_deref() == Some("mtp") && !matches!(device.as_str(), "auto" | "cpu" | "cuda") {
            return Err(format!("Multi-token prediction runs on the processor or on a CUDA graphics card, not on the {device} backend. Pick another mode or another device."));
        }
        if self.gpu_layers.as_deref().is_some_and(|n| n != "0") && device == "cpu" {
            return Err("GPU layers have no effect while the model runs on the processor. Set GPU layers to automatic.".into());
        }
        Ok(())
    }

    /// The server arguments with these options in place of the ones given before.
    pub fn apply(&self, args: &[String]) -> Vec<String> {
        let mut all: Vec<&str> = [CONTEXT_FLAGS, GPU_LAYER_FLAGS, THREAD_FLAGS, KV_FLAGS, SEQ_FLAGS, NPU_MODE_FLAGS].concat();
        if self.speculative.is_some() {
            all.extend(SPECULATIVE_FLAGS);
        }
        // the device is only replaced when the options name one
        let all: Vec<&str> = if self.device.is_some() { [all, DEVICE_FLAGS.to_vec()].concat() } else { all };
        let mut out = remove_flags(args, &all);
        if let Some(n) = self.context {
            out.extend(["--ctx".to_owned(), n.to_string()]);
        }
        if let Some(layers) = &self.gpu_layers {
            // "all" is a number larger than any model has layers
            out.extend(["--gpu-layers".to_owned(), if layers == "all" { "9999".to_owned() } else { layers.clone() }]);
        }
        if let Some(n) = self.threads {
            out.extend(["--threads".to_owned(), n.to_string()]);
        }
        if let Some(kind) = &self.kv_cache_type {
            out.extend(["--kv-cache-type".to_owned(), kind.clone()]);
        }
        if let Some(n) = self.max_seqs {
            out.extend(["--max-seqs".to_owned(), n.to_string()]);
        }
        if let Some(device) = self.device.as_ref().filter(|d| d.as_str() != "auto") {
            out.extend(["--device".to_owned(), device.clone()]);
        }
        // only meaningful on an NPU; the server ignores it otherwise
        if let (Some(mode), true) = (&self.npu_mode, self.wants_npu()) {
            out.extend(["--npu-mode".to_owned(), mode.clone()]);
        }
        out.extend(self.speculative_args());
        out
    }

    /// The server flags of the speculative decoding choice.
    fn speculative_args(&self) -> Vec<String> {
        let tokens = |default: usize| self.speculative_tokens.unwrap_or(default).to_string();
        match self.speculative.as_deref() {
            Some("off") => vec!["--speculative".into(), "off".into()],
            Some("lookup") => vec!["--speculative".into(), "ngram".into(), "--num-speculative-tokens".into(), tokens(5)],
            Some("suffix") => vec!["--speculative".into(), "suffix".into(), "--num-speculative-tokens".into(), tokens(6)],
            Some("draft") => {
                let model = self.draft_model.clone().unwrap_or_default();
                vec!["--speculative".into(), "draft".into(), "--speculative-model".into(), model, "--num-speculative-tokens".into(), tokens(5)]
            }
            Some("mtp") => vec!["--speculative".into(), "mtp".into()],
            _ => Vec::new(),
        }
    }

    /// Refuses a speculative decoding choice that cannot work with the model file `target` (checked before the server restarts).
    pub fn check_with_model(&self, target: &Path) -> Result<(), String> {
        let Some(mode) = self.speculative.as_deref().filter(|m| *m != "off") else { return Ok(()) };
        let probe = fllm_gguf_probe::MappedGguf::open(target).map_err(|e| e.to_string())?;
        let index = probe.index();
        let arch = index.metadata.get("general.architecture").and_then(Value::as_str).unwrap_or_default();
        if STATEFUL_ARCHITECTURES.contains(&arch) {
            return Err(format!("Speculative decoding does not work with this model: its architecture ({arch}) keeps a running state that cannot be rolled back when a guess is wrong. Choose off."));
        }
        match mode {
            "mtp" if !index.tensors.iter().any(|t| t.name.contains(".nextn.")) => {
                Err("This model file has no multi-token prediction heads. Choose prompt lookup or a draft model instead.".into())
            }
            "draft" => {
                let draft = self.draft_model.as_deref().unwrap_or_default();
                let path = PathBuf::from(draft);
                if !path.is_file() {
                    return Err(format!("The draft model \"{draft}\" was not found. Pick one of the downloaded models."));
                }
                let small = fllm_gguf_probe::MappedGguf::open(&path).map_err(|e| format!("The draft model cannot be read: {e}"))?;
                if small.index().file_len >= index.file_len {
                    return Err("The draft model must be smaller than the model it drafts for, or speculation only slows it down.".into());
                }
                for key in ["tokenizer.ggml.tokens", "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id"] {
                    if index.metadata.get(key) != small.index().metadata.get(key) {
                        return Err("The draft model does not use the same vocabulary as the model. Pick a draft model of the same family.".into());
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

/// `args` without the given flags and their values (`--flag value` and `--flag=value`).
fn remove_flags(args: &[String], names: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if names.contains(&arg) {
            i += 2;
        } else if names.iter().any(|name| arg.strip_prefix(name).is_some_and(|rest| rest.starts_with('='))) {
            i += 1;
        } else {
            out.push(args[i].clone());
            i += 1;
        }
    }
    out
}

/// The value of the last occurrence of one of the flags.
fn flag_value(args: &[String], names: &[&str]) -> Option<String> {
    let mut found = None;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if names.contains(&arg) {
            found = args.get(i + 1).cloned();
            i += 2;
            continue;
        }
        if let Some(value) = names.iter().find_map(|name| arg.strip_prefix(name).and_then(|rest| rest.strip_prefix('='))) {
            found = Some(value.to_owned());
        }
        i += 1;
    }
    found
}

/// The device the server was started with (`auto` when none was named).
pub(in crate::server) fn device_of(args: &[String]) -> String {
    flag_value(args, &["--device"]).unwrap_or_else(|| "auto".to_owned()).to_ascii_lowercase()
}

/// The options this server process runs with, as the loaded-model card shows them (`auto` for what was not given).
pub(in crate::server) fn active_options_json(args: &[String]) -> Value {
    let given = |names: &[&str]| flag_value(args, names).unwrap_or_else(|| "auto".to_owned());
    json!({
        "context": given(CONTEXT_FLAGS),
        "gpu_layers": given(GPU_LAYER_FLAGS),
        "threads": given(THREAD_FLAGS),
        "kv_cache_type": flag_value(args, &["--kv-cache-type", "--kv-cache-dtype"]).or_else(|| flag_value(args, &["--cache-type-k", "-ctk"])).unwrap_or_else(|| "auto".to_owned()),
        "max_seqs": given(SEQ_FLAGS),
        "device": device_of(args),
        "speculative": active_speculative(args),
        "draft_model": flag_value(args, &["--draft-model", "--model-draft", "--speculative-model"]),
    })
}

/// The speculative decoding mode the server runs with, in the words of the load options (`auto`: no flag given).
fn active_speculative(args: &[String]) -> String {
    let lookup = flag_value(args, &["--speculative-lookup", "-sp"]).and_then(|v| v.parse::<usize>().ok()).unwrap_or(0) > 0;
    let draft = flag_value(args, &["--draft-model", "--model-draft", "--speculative-model"]).is_some();
    match flag_value(args, &["--speculative"]).map(|v| v.to_ascii_lowercase()).as_deref() {
        Some("off") => "off".into(),
        Some("ngram") => "lookup".into(),
        Some("draft") => "draft".into(),
        Some("mtp") => "mtp".into(),
        Some(number) if number.parse::<usize>().is_ok_and(|n| n > 0) => "lookup".into(),
        Some(other) => other.to_owned(),
        None if draft => "draft".into(),
        None if lookup => "lookup".into(),
        None => "auto".into(),
    }
}

const OPTIONS_FILE: &str = "model-options.json";

/// Options are kept per model file name.
fn options_key(model: &str) -> String {
    Path::new(model).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| model.to_owned())
}

fn read_saved() -> serde_json::Map<String, Value> {
    std::fs::read_to_string(crate::hub::studio_data_file(OPTIONS_FILE))
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_default()
}

/// The saved options of all models, by file name.
pub(in crate::server) fn saved_options_json() -> Value {
    Value::Object(read_saved())
}

fn saved_options(model: &str) -> Option<LoadOptions> {
    read_saved().get(&options_key(model)).and_then(|v| LoadOptions::from_json(v).ok())
}

fn save_options(model: &str, options: &LoadOptions) {
    let mut saved = read_saved();
    if options.is_automatic() {
        saved.remove(&options_key(model));
    } else {
        saved.insert(options_key(model), options.to_json());
    }
    let file = crate::hub::studio_data_file(OPTIONS_FILE);
    if let Some(dir) = file.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(file, Value::Object(saved).to_string());
}

/// The options a load request asks for: the ones in the request, otherwise the ones saved for this model; `None` leaves the start
/// options of the server as they are.
pub(in crate::server) fn requested_options(body: &Value) -> Result<Option<LoadOptions>, String> {
    match body.get("options").filter(|v| v.is_object()) {
        Some(given) => LoadOptions::from_json(given).map(Some),
        None => Ok(None),
    }
}

/// Refuses a speculative decoding choice (given or saved for the model) that cannot work with the model file.
pub(in crate::server) fn check_speculative(requested: Option<&LoadOptions>, model: &str, path: &Path) -> Result<(), String> {
    match requested.cloned().or_else(|| saved_options(model)) {
        Some(options) => options.check_with_model(path),
        None => Ok(()),
    }
}

/// The arguments for the new server: options of the request (saved for next time), or the ones saved for this model.
pub(in crate::server) fn args_with_options(args: Vec<String>, model: &str, requested: Option<LoadOptions>) -> Vec<String> {
    let options = match requested {
        Some(options) => {
            save_options(model, &options);
            Some(options)
        }
        None => saved_options(model),
    };
    // all automatic for a model that was given options means the start flags of the server do not apply either
    match options {
        Some(options) => options.apply(&args),
        None => args,
    }
}

// ------------------------------------------------------------------------------------------------------------------------------
// Starting the next server process: model change and unload
// ------------------------------------------------------------------------------------------------------------------------------

/// Starts a server with `args` that takes over the port from this one. `setup`: it starts without a model.
pub(in crate::server) fn spawn_replacement(args: Vec<String>, setup: bool) -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| format!("cannot find the server program: {e}"))?;
    let mut command = std::process::Command::new(exe);
    command
        .args(args)
        .env("FLLM_REPLACE_SERVER", "1")
        // the desktop app watches its server through this pipe, so the new server keeps it
        .stdin(if std::env::var_os("FLLM_EXIT_ON_STDIN_EOF").is_some() { std::process::Stdio::inherit() } else { std::process::Stdio::null() });
    if setup {
        command.env("FLLM_STUDIO_SETUP", "1");
    }
    if let Some(dir) = crate::hub::saved_models_dir() {
        command.env("FLLM_MODELS_DIR", dir);
    }
    command.spawn().map(|_| ()).map_err(|e| format!("cannot start the new server: {e}"))
}

/// Ends this process shortly after the reply has gone out.
pub(in crate::server) fn exit_after_reply() {
    let _ = std::thread::Builder::new().name("fllm-exit".into()).spawn(|| {
        std::thread::sleep(Duration::from_millis(700));
        std::process::exit(0);
    });
}

/// The start arguments of a server without a model: no model flags, and `serve` in place of the browser launcher `studio`.
pub(in crate::server) fn unload_args(given: &[String]) -> Vec<String> {
    let mut out = super::studio::strip_model_args(given);
    out.retain(|a| a != "--no-open");
    if out.first().map(String::as_str) == Some("studio") {
        out[0] = "serve".to_owned();
    }
    out
}

/// `POST /api/studio/models/unload`: the server restarts without a model on the same port, with all its other options.
pub(in crate::server) async fn unload_model() -> Response {
    let args = unload_args(&std::env::args().skip(1).collect::<Vec<_>>());
    if let Err(error) = spawn_replacement(args, true) {
        return api_error(StatusCode::INTERNAL_SERVER_ERROR, &error);
    }
    exit_after_reply();
    Json(json!({"status": "restarting"})).into_response()
}

/// Whether a model that was idle for `idle` is to be unloaded after `minutes` (0: never).
fn idle_due(minutes: u64, idle: Duration) -> bool {
    minutes > 0 && idle >= Duration::from_secs(minutes.saturating_mul(60))
}

/// Unloads the model after the idle time set in the Studio. Started once with the server; reads the setting again at every check.
pub(in crate::server) fn spawn_idle_unload(metrics: Arc<Metrics>) {
    let _ = std::thread::Builder::new().name("fllm-idle-unload".into()).spawn(move || {
        let activity = || metrics.requests.load(Ordering::Relaxed).wrapping_add(metrics.forward_calls.load(Ordering::Relaxed));
        let mut last = activity();
        let mut since = std::time::Instant::now();
        loop {
            std::thread::sleep(Duration::from_secs(15));
            let now = activity();
            let busy = metrics.slots_running.load(Ordering::Relaxed) + metrics.slots_waiting.load(Ordering::Relaxed) > 0;
            if busy || now != last {
                last = now;
                since = std::time::Instant::now();
                continue;
            }
            if idle_due(crate::hub::idle_unload_minutes(), since.elapsed()) {
                tracing::info!("unloading the model after {} idle minutes", crate::hub::idle_unload_minutes());
                let args = unload_args(&std::env::args().skip(1).collect::<Vec<_>>());
                match spawn_replacement(args, true) {
                    Ok(()) => {
                        std::thread::sleep(Duration::from_millis(300));
                        std::process::exit(0);
                    }
                    Err(error) => {
                        tracing::warn!("cannot unload the idle model: {error}");
                        since = std::time::Instant::now();
                    }
                }
            }
        }
    });
}

// ------------------------------------------------------------------------------------------------------------------------------
// Settings: idle unload and the models folder
// ------------------------------------------------------------------------------------------------------------------------------

fn settings_json() -> Value {
    let default_dir = crate::hub::default_models_dir();
    json!({
        "idle_unload_minutes": crate::hub::idle_unload_minutes(),
        "load_on_start": crate::hub::load_on_start(),
        "models_dir": crate::hub::saved_models_dir(),
        "models_dir_active": crate::hub::cache_root().display().to_string(),
        "models_dir_default": default_dir.display().to_string(),
    })
}

/// A models folder typed or picked by the user: must be an absolute path to a folder that exists or can be created and written to.
fn validate_models_dir(text: &str) -> Result<PathBuf, String> {
    let cleaned = text.trim().trim_matches(|c| c == '"' || c == '\'').trim();
    let path = PathBuf::from(cleaned);
    if !path.is_absolute() {
        return Err("Type the full path of the folder, for example C:\\Models or /home/me/models.".into());
    }
    if path.is_file() {
        return Err("This path is a file, not a folder.".into());
    }
    std::fs::create_dir_all(&path).map_err(|e| format!("The folder cannot be created: {e}"))?;
    let probe = path.join(format!(".fllm-write-test-{}", std::process::id()));
    std::fs::write(&probe, b"").map_err(|e| format!("fLLM cannot write into this folder: {e}"))?;
    let _ = std::fs::remove_file(probe);
    Ok(path)
}

// ------------------------------------------------------------------------------------------------------------------------------
// Engine options: caches, preemption, telemetry and the admin key (saved here, applied when the server starts again)
// ------------------------------------------------------------------------------------------------------------------------------

/// The value an option has when nothing is saved (what the server uses by default).
fn engine_defaults() -> Value {
    json!({
        "output_cache": "exact", "output_cache_mb": 64, "output_cache_ttl": 3600, "cache_scope": "global",
        "prefix_cache": true, "prefix_salt": "", "preemption": "recompute", "otel_endpoint": "", "otel_service": "fllm",
    })
}

fn engine_json() -> Value {
    let mut values = engine_defaults();
    let saved = crate::hub::engine_settings();
    for (key, value) in &saved {
        if key != "admin_key" {
            values[key.as_str()] = value.clone();
        }
    }
    let overridden: Vec<&str> = crate::hub::ENGINE_OPTIONS
        .iter()
        .filter(|(key, name)| !saved.contains_key(*key) && std::env::var_os(name).is_some_and(|v| !v.is_empty()))
        .map(|(key, _)| *key)
        .collect();
    json!({
        "values": values,
        "defaults": engine_defaults(),
        "admin_key_set": saved.get("admin_key").and_then(Value::as_str).is_some_and(|k| !k.is_empty()) || std::env::var_os("FLLM_ADMIN_KEY").is_some_and(|v| !v.is_empty()),
        // options that the environment of the server sets and the page cannot change
        "set_by_environment": overridden,
    })
}

/// `GET /api/studio/engine`
pub(in crate::server) async fn get_engine() -> Response {
    Json(engine_json()).into_response()
}

fn engine_choice(body: &Value, key: &str, allowed: &[&str]) -> Result<Option<Value>, String> {
    match body.get(key) {
        None => Ok(None),
        Some(Value::String(s)) if allowed.contains(&s.as_str()) => Ok(Some(json!(s))),
        Some(_) => Err(format!("`{key}` must be one of: {}.", allowed.join(", "))),
    }
}

fn engine_number(body: &Value, key: &str, min: u64, max: u64) -> Result<Option<Value>, String> {
    match body.get(key) {
        None => Ok(None),
        Some(v) => match v.as_u64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok())) {
            Some(n) if (min..=max).contains(&n) => Ok(Some(json!(n))),
            _ => Err(format!("`{key}` must be a whole number from {min} to {max}.")),
        },
    }
}

/// The changes a `POST /api/studio/engine` body asks for: a value sets an option, an empty text takes it back to its default.
fn engine_changes(body: &Value) -> Result<Vec<(&'static str, Option<Value>)>, String> {
    let mut changes: Vec<(&'static str, Option<Value>)> = Vec::new();
    for (key, value) in [
        ("output_cache", engine_choice(body, "output_cache", &["exact", "off"])),
        ("output_cache_mb", engine_number(body, "output_cache_mb", 1, 65_536)),
        ("output_cache_ttl", engine_number(body, "output_cache_ttl", 1, 31_536_000)),
        ("cache_scope", engine_choice(body, "cache_scope", &["global", "tenant", "off"])),
        ("preemption", engine_choice(body, "preemption", &["recompute", "swap"])),
    ] {
        if let Some(v) = value? {
            changes.push((key, Some(v)));
        }
    }
    if let Some(on) = body.get("prefix_cache").and_then(Value::as_bool) {
        changes.push(("prefix_cache", Some(json!(on))));
    }
    for key in ["prefix_salt", "otel_service", "admin_key"] {
        if let Some(text) = body.get(key).and_then(Value::as_str) {
            let text = text.trim();
            if text.len() > 200 || text.chars().any(char::is_control) {
                return Err(format!("`{key}` is too long or has control characters."));
            }
            changes.push((key, (!text.is_empty()).then(|| json!(text))));
        }
    }
    if let Some(text) = body.get("otel_endpoint").and_then(Value::as_str) {
        let text = text.trim();
        if !text.is_empty() && !(text.starts_with("http://") || text.starts_with("https://")) {
            return Err("The telemetry address must start with http:// or https://.".into());
        }
        changes.push(("otel_endpoint", (!text.is_empty()).then(|| json!(text))));
    }
    Ok(changes)
}

/// `POST /api/studio/engine` with any of the options. Takes effect when the server starts again.
pub(in crate::server) async fn update_engine(Json(body): Json<Value>) -> Response {
    let changes = match engine_changes(&body) {
        Ok(changes) => changes,
        Err(message) => return api_error(StatusCode::BAD_REQUEST, &message),
    };
    let mut saved = crate::hub::engine_settings();
    for (key, value) in changes {
        match value {
            Some(v) => saved.insert(key.to_owned(), v),
            None => saved.remove(key),
        };
    }
    crate::hub::set_studio_setting("engine", if saved.is_empty() { Value::Null } else { Value::Object(saved) });
    Json(engine_json()).into_response()
}

/// `POST /api/studio/restart`: the server starts again with the same model and options (the engine options take effect).
pub(in crate::server) async fn restart_server() -> Response {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    let setup = std::env::var_os("FLLM_STUDIO_SETUP").is_some();
    let has_model = args.iter().any(|a| a == "--model" || a == "-m" || a.starts_with("--model="));
    if !setup && !has_model {
        if let Some(model) = crate::hub::last_studio_model() {
            args.push("--model".into());
            args.push(model);
        }
    }
    args.retain(|a| a != "--no-open");
    if let Err(error) = spawn_replacement(args, setup) {
        return api_error(StatusCode::INTERNAL_SERVER_ERROR, &error);
    }
    exit_after_reply();
    Json(json!({"status": "restarting"})).into_response()
}

/// `GET /api/studio/settings`
pub(in crate::server) async fn get_settings() -> Response {
    Json(settings_json()).into_response()
}

/// `POST /api/studio/settings` with `{idle_unload_minutes?, models_dir?}` (an empty `models_dir` goes back to the default folder).
pub(in crate::server) async fn update_settings(Json(body): Json<Value>) -> Response {
    if let Some(value) = body.get("idle_unload_minutes") {
        let minutes = value.as_u64().or_else(|| value.as_str().and_then(|s| s.trim().parse().ok()));
        match minutes {
            Some(m) if m <= 10_080 => crate::hub::set_studio_setting("idle_unload_minutes", json!(m)),
            _ => return api_error(StatusCode::BAD_REQUEST, "The idle time must be a whole number of minutes, from 0 (never) to 10080."),
        }
    }
    if let Some(on) = body.get("load_on_start").and_then(Value::as_bool) {
        crate::hub::set_studio_setting("load_on_start", json!(on));
    }
    if let Some(text) = body.get("models_dir").and_then(Value::as_str) {
        if text.trim().is_empty() {
            crate::hub::set_studio_setting("models_dir", Value::Null);
            crate::hub::set_models_dir(Some(crate::hub::default_models_dir()));
        } else {
            match tokio::task::spawn_blocking({
                let text = text.to_owned();
                move || validate_models_dir(&text)
            })
            .await
            {
                Ok(Ok(path)) => {
                    crate::hub::set_studio_setting("models_dir", json!(path.display().to_string()));
                    crate::hub::set_models_dir(Some(path));
                }
                Ok(Err(error)) => return api_error(StatusCode::BAD_REQUEST, &error),
                Err(e) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
            }
        }
    }
    Json(settings_json()).into_response()
}

// ------------------------------------------------------------------------------------------------------------------------------
// Native file picker
// ------------------------------------------------------------------------------------------------------------------------------

#[derive(Debug, PartialEq)]
struct PickCommand {
    program: &'static str,
    args: Vec<String>,
}

fn command(program: &'static str, args: &[&str]) -> PickCommand {
    PickCommand { program, args: args.iter().map(|a| (*a).to_owned()).collect() }
}

/// The programs that can show a file or folder dialog on `os`, best first.
fn pick_commands(kind: &str, os: &str) -> Vec<PickCommand> {
    let folder = kind == "folder";
    match os {
        "windows" => {
            let dialog = if folder {
                "$d = New-Object System.Windows.Forms.FolderBrowserDialog; $d.Description = 'Choose a folder'; $r = $d.ShowDialog($o); $p = $d.SelectedPath"
            } else {
                "$d = New-Object System.Windows.Forms.OpenFileDialog; $d.Title = 'Choose a model file'; $d.Filter = 'GGUF models (*.gguf)|*.gguf|All files (*.*)|*.*'; $r = $d.ShowDialog($o); $p = $d.FileName"
            };
            let script = format!(
                "Add-Type -AssemblyName System.Windows.Forms; $o = New-Object System.Windows.Forms.Form; $o.TopMost = $true; {dialog}; if ($r -eq 'OK') {{ [Console]::Out.Write($p) }}"
            );
            vec![PickCommand { program: "powershell", args: vec!["-NoProfile".into(), "-STA".into(), "-Command".into(), script] }]
        }
        "macos" => {
            let script = if folder {
                "POSIX path of (choose folder with prompt \"Choose a folder\")"
            } else {
                "POSIX path of (choose file with prompt \"Choose a model file\" of type {\"gguf\"})"
            };
            vec![command("osascript", &["-e", script])]
        }
        _ => {
            if folder {
                vec![
                    command("zenity", &["--file-selection", "--directory", "--title=Choose a folder"]),
                    command("kdialog", &["--getexistingdirectory", ".", "--title", "Choose a folder"]),
                ]
            } else {
                vec![
                    command("zenity", &["--file-selection", "--title=Choose a model file", "--file-filter=GGUF models | *.gguf", "--file-filter=All files | *"]),
                    command("kdialog", &["--getopenfilename", ".", "*.gguf|GGUF models", "--title", "Choose a model file"]),
                ]
            }
        }
    }
}

/// What a dialog program answered: the chosen path, `None` when the user closed the dialog, or an error text.
fn pick_outcome(success: bool, stdout: &str, stderr: &str) -> Result<Option<String>, String> {
    let chosen = stdout.trim_end_matches(['\r', '\n']).to_owned();
    if success {
        return Ok((!chosen.is_empty()).then_some(chosen));
    }
    let problem = stderr.trim();
    if problem.is_empty() || problem.to_ascii_lowercase().contains("cancel") {
        Ok(None)
    } else {
        Err(problem.chars().take(300).collect())
    }
}

enum PickError {
    Unavailable,
    Failed(String),
}

fn run_picker(kind: &str) -> Result<Option<String>, PickError> {
    for candidate in pick_commands(kind, std::env::consts::OS) {
        let mut process = std::process::Command::new(candidate.program);
        process.args(&candidate.args).stdin(std::process::Stdio::null());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            process.creation_flags(0x0800_0000); // no console window
        }
        match process.output() {
            Ok(output) => {
                return pick_outcome(output.status.success(), &String::from_utf8_lossy(&output.stdout), &String::from_utf8_lossy(&output.stderr))
                    .map_err(PickError::Failed);
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(PickError::Failed(e.to_string())),
        }
    }
    Err(PickError::Unavailable)
}

/// `POST /api/studio/pick-file` with `{kind: "gguf" | "folder"}`: opens the dialog of the operating system on this machine and answers
/// with `{path}` (`null` when the dialog was closed).
pub(in crate::server) async fn pick_file(Json(body): Json<Value>) -> Response {
    let kind = body.get("kind").and_then(Value::as_str).unwrap_or("gguf").to_owned();
    if kind != "gguf" && kind != "folder" {
        return api_error(StatusCode::BAD_REQUEST, "kind must be \"gguf\" or \"folder\"");
    }
    match tokio::task::spawn_blocking(move || run_picker(&kind)).await {
        Ok(Ok(path)) => Json(json!({"path": path})).into_response(),
        Ok(Err(PickError::Unavailable)) => api_error(
            StatusCode::NOT_IMPLEMENTED,
            "This computer has no program that can open a file dialog (on Linux install zenity or kdialog). Type the path instead.",
        ),
        Ok(Err(PickError::Failed(error))) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("The file dialog failed: {error}")),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

// ------------------------------------------------------------------------------------------------------------------------------
// Download queue: one download runs, the others wait; pause, resume and cancel
// ------------------------------------------------------------------------------------------------------------------------------

#[derive(Clone)]
struct Download {
    id: u64,
    model: String,
    quant: Option<String>,
    /// `queued`, `running`, `paused`, `done` or `failed`
    state: &'static str,
    attempt: u32,
    message: String,
    path: String,
    reference: String,
}

static QUEUE: std::sync::Mutex<Vec<Download>> = std::sync::Mutex::new(Vec::new());
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static WORKER_RUNNING: AtomicBool = AtomicBool::new(false);

const PULL_ATTEMPTS: u32 = 8;
/// Finished and failed entries stay in the list for a while; the oldest go first.
const KEPT_FINISHED: usize = 12;

fn queue() -> std::sync::MutexGuard<'static, Vec<Download>> {
    QUEUE.lock().unwrap_or_else(|e| e.into_inner())
}

fn update(id: u64, change: impl FnOnce(&mut Download)) {
    if let Some(entry) = queue().iter_mut().find(|d| d.id == id) {
        change(entry);
    }
}

/// Whether another try can help: the connection broke or timed out (the partial file stays and the next try continues it).
fn worth_retrying(error: &str) -> bool {
    let lower = error.to_ascii_lowercase();
    ["interrupted", "connection", "timed out", "timeout", "truncated", "error sending", "dns", "reset", "broken pipe", "unexpected eof", "hub request"]
        .iter()
        .any(|word| lower.contains(word))
}

fn run_job(id: u64, spec: String, quant: Option<String>) {
    for attempt in 1..=PULL_ATTEMPTS {
        update(id, |d| d.attempt = attempt);
        let outcome = super::studio::resolve_model(&spec, quant.as_deref());
        let error = match outcome {
            Ok((path, reference)) => {
                update(id, |d| {
                    d.state = "done";
                    d.path = path.display().to_string();
                    d.reference = reference;
                    d.message.clear();
                });
                return;
            }
            Err(error) => error,
        };
        if error.contains(crate::hub::DOWNLOAD_PAUSED) {
            update(id, |d| d.state = "paused");
            return;
        }
        if error.contains(crate::hub::DOWNLOAD_CANCELLED) {
            queue().retain(|d| d.id != id);
            return;
        }
        update(id, |d| d.message = error.clone());
        if attempt == PULL_ATTEMPTS || !worth_retrying(&error) {
            break;
        }
        // wait before the next try, but react to pause and cancel
        for _ in 0..(attempt.min(5) * 3 * 4) {
            if crate::hub::download_stop() != crate::hub::STOP_NONE {
                break;
            }
            std::thread::sleep(Duration::from_millis(250));
        }
        match crate::hub::download_stop() {
            crate::hub::STOP_PAUSE => {
                update(id, |d| d.state = "paused");
                return;
            }
            crate::hub::STOP_CANCEL => {
                queue().retain(|d| d.id != id);
                return;
            }
            _ => {}
        }
    }
    update(id, |d| d.state = "failed");
}

fn worker() {
    loop {
        let next = {
            let mut list = queue();
            match list.iter_mut().find(|d| d.state == "queued") {
                Some(entry) => {
                    entry.state = "running";
                    entry.message.clear();
                    Some((entry.id, entry.model.clone(), entry.quant.clone()))
                }
                None => {
                    // decided under the lock: a download added after this starts a new worker
                    WORKER_RUNNING.store(false, Ordering::SeqCst);
                    None
                }
            }
        };
        let Some((id, spec, quant)) = next else { return };
        crate::hub::set_download_stop(crate::hub::STOP_NONE);
        run_job(id, spec, quant);
        crate::hub::set_download_stop(crate::hub::STOP_NONE);
    }
}

fn ensure_worker() -> Result<(), String> {
    if WORKER_RUNNING.swap(true, Ordering::SeqCst) {
        return Ok(());
    }
    std::thread::Builder::new().name("fllm-model-pull".into()).spawn(worker).map(|_| ()).map_err(|e| {
        WORKER_RUNNING.store(false, Ordering::SeqCst);
        format!("cannot start the download: {e}")
    })
}

fn trim_finished(list: &mut Vec<Download>) {
    let finished = list.iter().filter(|d| matches!(d.state, "done" | "failed")).count();
    for _ in KEPT_FINISHED..finished {
        if let Some(position) = list.iter().position(|d| matches!(d.state, "done" | "failed")) {
            list.remove(position);
        }
    }
}

/// Adds a download to the queue (the same model and quantization twice gives the existing entry).
fn enqueue(model: &str, quant: Option<String>) -> Result<u64, String> {
    let id = {
        let mut list = queue();
        if let Some(existing) = list.iter().find(|d| d.model == model && d.quant == quant && matches!(d.state, "queued" | "running" | "paused")) {
            return Ok(existing.id);
        }
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        list.push(Download { id, model: model.to_owned(), quant, state: "queued", attempt: 0, message: String::new(), path: String::new(), reference: String::new() });
        trim_finished(&mut list);
        id
    };
    ensure_worker()?;
    Ok(id)
}

/// `done`, `failed` or a stopped entry cannot be asked to stop again; this applies one user action. Returns whether the id exists.
fn apply_action(id: u64, action: &str) -> Result<bool, String> {
    let mut list = queue();
    let Some(position) = list.iter().position(|d| d.id == id) else { return Ok(false) };
    let state = list[position].state;
    match action {
        "pause" => match state {
            "queued" => list[position].state = "paused",
            "running" => crate::hub::set_download_stop(crate::hub::STOP_PAUSE),
            _ => {}
        },
        "resume" => {
            if matches!(state, "paused" | "failed") {
                list[position].state = "queued";
                list[position].message.clear();
                drop(list);
                ensure_worker()?;
            }
        }
        "cancel" => {
            if state == "running" {
                crate::hub::set_download_stop(crate::hub::STOP_CANCEL);
            } else {
                list.remove(position);
            }
        }
        "clear" => {
            if matches!(state, "done" | "failed") {
                list.remove(position);
            }
        }
        other => return Err(format!("unknown action \"{other}\" (pause, resume, cancel or clear)")),
    }
    Ok(true)
}

/// Seconds left at the current speed (`None` when it cannot be told).
fn eta_seconds(written: u64, total: u64, megabytes_per_second: f64) -> Option<u64> {
    if total <= written || megabytes_per_second < 0.01 {
        return None;
    }
    Some(((total - written) as f64 / (megabytes_per_second * 1_000_000.0)).ceil() as u64)
}

fn downloads_json() -> Vec<Value> {
    let progress = crate::hub::download_progress();
    let list = queue().clone();
    list.into_iter()
        .map(|d| {
            let mut entry = json!({
                "id": d.id, "model": d.model, "quant": d.quant, "state": d.state, "attempt": d.attempt, "max_attempts": PULL_ATTEMPTS,
                "error": d.message, "model_path": d.path, "reference": d.reference,
            });
            if let (true, Some(p)) = (d.state == "running", progress.as_ref().filter(|p| !p.finished)) {
                entry["file"] = json!(p.file);
                entry["written"] = json!(p.written);
                entry["total"] = json!(p.total);
                entry["mb_per_s"] = json!(p.mb_per_s);
                entry["eta_seconds"] = json!(eta_seconds(p.written, p.total.unwrap_or(0), p.mb_per_s));
            }
            entry
        })
        .collect()
}

/// `POST /api/studio/models/pull` with `{model, quant?}`: adds the download to the queue and answers at once.
pub(in crate::server) async fn pull_model(Json(body): Json<Value>) -> Response {
    let Some(spec) = body.get("model").and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty()) else {
        return api_error(StatusCode::BAD_REQUEST, "model specification is required");
    };
    let quant = body.get("quant").and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty()).map(str::to_owned);
    match enqueue(spec, quant) {
        Ok(id) => Json(json!({"status": "queued", "id": id})).into_response(),
        Err(error) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &error),
    }
}

/// `GET /api/studio/models/progress`: every download with its state, speed and time left.
pub(in crate::server) async fn model_progress() -> Response {
    let downloads = downloads_json();
    let active = downloads.iter().any(|d| d["state"] == "running" || d["state"] == "queued");
    Json(json!({"active": active, "downloads": downloads})).into_response()
}

/// `POST /api/studio/models/downloads` with `{id, action: "pause" | "resume" | "cancel" | "clear"}`.
pub(in crate::server) async fn download_action(Json(body): Json<Value>) -> Response {
    let id = body.get("id").and_then(Value::as_u64).unwrap_or(0);
    let action = body.get("action").and_then(Value::as_str).unwrap_or_default().to_owned();
    match apply_action(id, &action) {
        Ok(true) => Json(json!({"downloads": downloads_json()})).into_response(),
        Ok(false) => api_error(StatusCode::NOT_FOUND, "this download is not in the list any more"),
        Err(error) => api_error(StatusCode::BAD_REQUEST, &error),
    }
}

// ------------------------------------------------------------------------------------------------------------------------------
// Logs and model cards
// ------------------------------------------------------------------------------------------------------------------------------

/// `GET /api/studio/logs?lines=N`: the most recent log lines of the server, oldest first.
pub(in crate::server) async fn logs(Query(query): Query<HashMap<String, String>>) -> Response {
    let count = query.get("lines").and_then(|v| v.parse::<usize>().ok()).unwrap_or(200).clamp(1, 2000);
    Json(json!({"lines": super::super::logging::recent_lines(count)})).into_response()
}

/// `GET /api/studio/hf-readme?repo=owner/name`: the start of the model card as plain text.
pub(in crate::server) async fn hf_readme(Query(query): Query<HashMap<String, String>>) -> Response {
    let Some(repo) = query.get("repo").map(|s| s.trim().to_owned()).filter(|s| !s.is_empty()) else {
        return api_error(StatusCode::BAD_REQUEST, "missing ?repo=owner/name query parameter");
    };
    match tokio::task::spawn_blocking(move || crate::hub::repository_readme(&repo)).await {
        Ok(Ok(text)) => Json(json!({"summary": crate::hub::readme_summary(&text, 600)})).into_response(),
        Ok(Err(error)) => api_error(StatusCode::BAD_GATEWAY, &error),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_options_are_checked_before_they_are_saved() {
        let ok = engine_changes(&json!({"output_cache": "off", "output_cache_mb": "128", "cache_scope": "tenant", "prefix_cache": false, "prefix_salt": " a ", "otel_endpoint": "http://127.0.0.1:4318"})).expect("valid");
        assert!(ok.contains(&("output_cache", Some(json!("off")))));
        assert!(ok.contains(&("output_cache_mb", Some(json!(128)))));
        assert!(ok.contains(&("prefix_cache", Some(json!(false)))));
        assert!(ok.contains(&("prefix_salt", Some(json!("a")))));
        // an empty text takes an option back to its default
        assert_eq!(engine_changes(&json!({"otel_endpoint": "", "admin_key": ""})).unwrap(), vec![("admin_key", None), ("otel_endpoint", None)]);
        for bad in [json!({"output_cache": "maybe"}), json!({"output_cache_mb": 0}), json!({"cache_scope": "x"}), json!({"otel_endpoint": "ftp://x"}), json!({"prefix_salt": "a
b"}), json!({"preemption": 3})] {
            assert!(engine_changes(&bad).is_err(), "{bad}");
        }
    }


    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn options_are_read_and_checked() {
        let ok = LoadOptions::from_json(&json!({"context": "8192", "gpu_layers": "all", "threads": 6, "kv_cache_type": "Q8_0", "max_seqs": "auto"})).unwrap();
        assert_eq!(ok, LoadOptions { context: Some(8192), gpu_layers: Some("all".into()), threads: Some(6), kv_cache_type: Some("q8_0".into()), max_seqs: None, device: None, npu_mode: None, ..LoadOptions::default() });
        assert!(LoadOptions::from_json(&json!({})).unwrap().is_automatic());
        assert!(LoadOptions::from_json(&json!({"context": "auto", "gpu_layers": "auto"})).unwrap().is_automatic());
        for bad in [json!({"context": 3}), json!({"context": "big"}), json!({"gpu_layers": -1}), json!({"threads": 0}), json!({"kv_cache_type": "q3_9"}), json!({"max_seqs": 1000})] {
            let error = LoadOptions::from_json(&bad).unwrap_err();
            assert!(error.ends_with('.'), "{error}");
        }
    }

    #[test]
    fn speculative_choices_become_server_flags() {
        let given = args(&["serve", "--model", "m.gguf", "--speculative-lookup", "4", "--draft-tokens", "3", "--port", "9000"]);
        let off = LoadOptions::from_json(&json!({"speculative": "off"})).unwrap();
        assert_eq!(off.apply(&given), args(&["serve", "--model", "m.gguf", "--port", "9000", "--speculative", "off"]));
        let lookup = LoadOptions::from_json(&json!({"speculative": "Lookup", "speculative_tokens": "8"})).unwrap();
        assert_eq!(lookup.apply(&args(&["serve"])), args(&["serve", "--speculative", "ngram", "--num-speculative-tokens", "8"]));
        let suffix = LoadOptions::from_json(&json!({"speculative": "suffix"})).unwrap();
        assert_eq!(suffix.apply(&args(&["serve"])), args(&["serve", "--speculative", "suffix", "--num-speculative-tokens", "6"]));
        let suffix_tokens = LoadOptions::from_json(&json!({"speculative": "Suffix", "speculative_tokens": 3})).unwrap();
        assert!(suffix_tokens.apply(&args(&["serve"])).ends_with(&args(&["--num-speculative-tokens", "3"])));
        let default_tokens = LoadOptions::from_json(&json!({"speculative": "lookup"})).unwrap();
        assert!(default_tokens.apply(&args(&["serve"])).ends_with(&args(&["--num-speculative-tokens", "5"])));
        let draft = LoadOptions::from_json(&json!({"speculative": "draft", "draft_model": "/models/small.gguf", "speculative_tokens": 4})).unwrap();
        assert_eq!(draft.apply(&given), args(&["serve", "--model", "m.gguf", "--port", "9000", "--speculative", "draft", "--speculative-model", "/models/small.gguf", "--num-speculative-tokens", "4"]));
        let mtp = LoadOptions::from_json(&json!({"speculative": "mtp", "speculative_tokens": 4, "draft_model": "x.gguf"})).unwrap();
        assert_eq!((mtp.speculative_tokens, mtp.draft_model.as_deref()), (None, None));
        assert_eq!(mtp.apply(&args(&["serve"])), args(&["serve", "--speculative", "mtp"]));
        // not chosen: the start flags of the server stay
        let none = LoadOptions::from_json(&json!({"speculative": "auto", "context": 4096})).unwrap();
        assert_eq!(none.apply(&given)[..given.len()], given[..]);
        // saved choices come back as they were
        let again = LoadOptions::from_json(&draft.to_json()).unwrap();
        assert_eq!(again, draft);
        assert_eq!(draft.to_json()["speculative"], "draft");
    }

    #[test]
    fn impossible_speculative_choices_are_turned_down() {
        for bad in [json!({"speculative": "turbo"}), json!({"speculative": "draft"}), json!({"speculative": "draft", "draft_model": "  "}), json!({"speculative": "lookup", "speculative_tokens": 99})] {
            let error = LoadOptions::from_json(&bad).unwrap_err();
            assert!(error.ends_with('.'), "{error}");
        }
        let mtp = LoadOptions::from_json(&json!({"speculative": "mtp"})).unwrap();
        assert!(mtp.check_with_device("auto").is_ok() && mtp.check_with_device("cpu").is_ok() && mtp.check_with_device("cuda").is_ok());
        assert!(mtp.check_with_device("vulkan").unwrap_err().contains("CUDA"));
        let on_vulkan = LoadOptions { device: Some("vulkan:1".into()), ..mtp.clone() };
        assert!(on_vulkan.check_with_device("cpu").is_err());
        let lookup = LoadOptions::from_json(&json!({"speculative": "lookup"})).unwrap();
        assert!(lookup.check_with_device("vulkan").is_ok());
    }

    #[test]
    fn the_active_speculative_mode_is_read_from_the_start_flags() {
        let mode = |list: &[&str]| active_options_json(&args(list))["speculative"].clone();
        assert_eq!(mode(&["serve"]), "auto");
        assert_eq!(mode(&["serve", "--speculative", "ngram"]), "lookup");
        assert_eq!(mode(&["serve", "--speculative=5"]), "lookup");
        assert_eq!(mode(&["serve", "--speculative-lookup", "4"]), "lookup");
        assert_eq!(mode(&["serve", "--draft-model", "d.gguf"]), "draft");
        assert_eq!(mode(&["serve", "--speculative", "mtp"]), "mtp");
        assert_eq!(mode(&["serve", "--speculative", "off"]), "off");
        assert_eq!(mode(&["serve", "--speculative", "suffix"]), "suffix");
    }

    #[test]
    fn the_npu_mode_is_passed_only_to_an_npu() {
        let all = LoadOptions::from_json(&json!({"device": "npu", "npu_mode": "all"})).unwrap();
        assert_eq!(all.npu_mode.as_deref(), Some("all"));
        assert_eq!(all.apply(&args(&["serve", "--model", "m.gguf"])), args(&["serve", "--model", "m.gguf", "--device", "npu", "--npu-mode", "all"]));
        // a stale mode does not travel to another device, and the default is no flag
        let other = LoadOptions::from_json(&json!({"device": "cpu", "npu_mode": "all"})).unwrap();
        assert!(!other.apply(&args(&["serve"])).contains(&"--npu-mode".to_owned()));
        let prompt = LoadOptions::from_json(&json!({"device": "npu", "npu_mode": "prompt"})).unwrap();
        assert_eq!(prompt.npu_mode, None);
        assert!(LoadOptions::from_json(&json!({"npu_mode": "everything"})).is_err());
        assert_eq!(all.to_json()["npu_mode"], "all");
    }

    #[test]
    fn the_device_option_replaces_the_start_flag() {
        let npu = LoadOptions::from_json(&json!({"device": "NPU"})).unwrap();
        assert_eq!(npu.device.as_deref(), Some("npu"));
        assert!(npu.wants_npu() && !npu.is_automatic());
        assert!(LoadOptions::from_json(&json!({"device": "npu:0"})).unwrap().wants_npu());
        assert!(!LoadOptions::from_json(&json!({"device": "vulkan:1"})).unwrap().wants_npu());
        assert!(!LoadOptions::from_json(&json!({"device": ""})).unwrap().wants_npu());
        for bad in [json!({"device": "npu:x"}), json!({"device": "cpu; rm -rf"}), json!({"device": "--ctx"})] {
            assert!(LoadOptions::from_json(&bad).is_err(), "{bad}");
        }
        let given = args(&["serve", "--device", "cuda:0", "--model", "m.gguf"]);
        assert_eq!(npu.apply(&given), args(&["serve", "--model", "m.gguf", "--device", "npu"]));
        // `auto` removes the flag, no device in the options keeps it
        let auto = LoadOptions::from_json(&json!({"device": "auto"})).unwrap();
        assert_eq!(auto.apply(&given), args(&["serve", "--model", "m.gguf"]));
        let keep = LoadOptions { context: Some(4096), ..LoadOptions::default() };
        assert_eq!(keep.apply(&given), args(&["serve", "--device", "cuda:0", "--model", "m.gguf", "--ctx", "4096"]));
        // the device of the options decides which combinations are refused
        let layers = LoadOptions { gpu_layers: Some("20".into()), device: Some("cpu".into()), ..LoadOptions::default() };
        assert!(layers.check_with_device("vulkan").is_err());
        assert_eq!(npu.to_json()["device"], "npu");
    }

    #[test]
    fn impossible_combinations_are_refused() {
        let q4 = LoadOptions { kv_cache_type: Some("q4_0".into()), ..LoadOptions::default() };
        assert!(q4.check_with_device("vulkan").is_err());
        assert!(q4.check_with_device("auto").is_ok());
        assert!(q4.check_with_device("cuda").is_ok());
        let q8 = LoadOptions { kv_cache_type: Some("q8_0".into()), ..LoadOptions::default() };
        assert!(q8.check_with_device("metal").is_ok());
        let layers = LoadOptions { gpu_layers: Some("20".into()), ..LoadOptions::default() };
        assert!(layers.check_with_device("cpu").is_err());
        assert!(layers.check_with_device("vulkan").is_ok());
        assert!(LoadOptions { gpu_layers: Some("0".into()), ..LoadOptions::default() }.check_with_device("cpu").is_ok());
    }

    #[test]
    fn options_replace_the_start_flags_of_the_server() {
        let given = args(&["serve", "--port", "9000", "--ctx", "4096", "--threads=3", "-ctk", "q8_0", "--max-seqs", "4", "--model", "m.gguf"]);
        let options = LoadOptions { context: Some(16384), gpu_layers: Some("all".into()), kv_cache_type: Some("f16".into()), ..LoadOptions::default() };
        assert_eq!(
            options.apply(&given),
            args(&["serve", "--port", "9000", "--model", "m.gguf", "--ctx", "16384", "--gpu-layers", "9999", "--kv-cache-type", "f16"])
        );
        // all automatic: the flags are gone and the server decides
        assert_eq!(LoadOptions::default().apply(&given), args(&["serve", "--port", "9000", "--model", "m.gguf"]));
    }

    #[test]
    fn the_loaded_model_card_shows_what_the_server_runs_with() {
        let active = active_options_json(&args(&["serve", "--ctx=8192", "-ngl", "20", "--cache-type-k", "q8_0", "--device", "VULKAN:1"]));
        assert_eq!(active["context"], "8192");
        assert_eq!(active["gpu_layers"], "20");
        assert_eq!(active["threads"], "auto");
        assert_eq!(active["kv_cache_type"], "q8_0");
        assert_eq!(active["max_seqs"], "auto");
        assert_eq!(active["device"], "vulkan:1");
        assert_eq!(device_of(&args(&["serve"])), "auto");
    }

    #[test]
    fn unloading_keeps_the_options_but_not_the_model() {
        let out = unload_args(&args(&["studio", "--no-open", "--model", "m.gguf", "--port", "9000", "--model-name=x", "--ctx", "8192"]));
        assert_eq!(out, args(&["serve", "--port", "9000", "--ctx", "8192"]));
        assert_eq!(unload_args(&args(&["serve", "-m", "a.gguf"])), args(&["serve"]));
    }

    #[test]
    fn idle_time_decides_the_unload() {
        assert!(!idle_due(0, Duration::from_secs(100_000)));
        assert!(!idle_due(10, Duration::from_secs(599)));
        assert!(idle_due(10, Duration::from_secs(600)));
    }

    #[test]
    fn the_file_dialog_commands_fit_each_system() {
        let windows = pick_commands("gguf", "windows");
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].program, "powershell");
        assert!(windows[0].args.contains(&"-STA".to_owned()));
        assert!(windows[0].args.last().unwrap().contains("OpenFileDialog") && windows[0].args.last().unwrap().contains("*.gguf"));
        assert!(pick_commands("folder", "windows")[0].args.last().unwrap().contains("FolderBrowserDialog"));
        let mac = pick_commands("gguf", "macos");
        assert_eq!((mac[0].program, mac[0].args[0].as_str()), ("osascript", "-e"));
        assert!(mac[0].args[1].contains("choose file") && pick_commands("folder", "macos")[0].args[1].contains("choose folder"));
        let linux: Vec<&str> = pick_commands("folder", "linux").iter().map(|c| c.program).collect();
        assert_eq!(linux, ["zenity", "kdialog"]);
        assert!(pick_commands("folder", "linux")[0].args.contains(&"--directory".to_owned()));
        assert!(pick_commands("gguf", "linux")[0].args.iter().any(|a| a.contains("*.gguf")));
    }

    #[test]
    fn the_answer_of_a_dialog_is_read() {
        assert_eq!(pick_outcome(true, "/home/me/m.gguf\n", ""), Ok(Some("/home/me/m.gguf".into())));
        assert_eq!(pick_outcome(true, "C:\\Models\\m.gguf", ""), Ok(Some("C:\\Models\\m.gguf".into())));
        assert_eq!(pick_outcome(true, "", ""), Ok(None));
        assert_eq!(pick_outcome(false, "", ""), Ok(None), "zenity closed");
        assert_eq!(pick_outcome(false, "", "execution error: User canceled. (-128)"), Ok(None));
        assert!(pick_outcome(false, "", "no display").is_err());
    }

    #[test]
    fn time_left_follows_the_speed() {
        assert_eq!(eta_seconds(0, 100_000_000, 10.0), Some(10));
        assert_eq!(eta_seconds(100, 100, 10.0), None);
        assert_eq!(eta_seconds(0, 100, 0.0), None);
    }

    #[test]
    fn the_models_folder_must_be_a_writable_absolute_folder() {
        assert!(validate_models_dir("relative/dir").is_err());
        let base = std::env::temp_dir().join(format!("fllm-models-dir-{}", std::process::id()));
        let wanted = base.join("nested");
        let path = validate_models_dir(&format!("\"{}\"", wanted.display())).unwrap();
        assert!(path.is_dir());
        let file = base.join("file.txt");
        std::fs::write(&file, b"x").unwrap();
        assert!(validate_models_dir(&file.display().to_string()).is_err());
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn the_queue_runs_one_download_and_keeps_the_others_waiting() {
        // entries are added directly: a real download needs the network
        let entry = |id: u64, state: &'static str| Download { id, model: format!("m{id}"), quant: None, state, attempt: 0, message: String::new(), path: String::new(), reference: String::new() };
        {
            let mut list = queue();
            list.clear();
            list.extend([entry(9001, "running"), entry(9002, "queued"), entry(9003, "queued")]);
        }
        assert_eq!(apply_action(9002, "pause"), Ok(true));
        assert_eq!(queue().iter().find(|d| d.id == 9002).unwrap().state, "paused");
        assert_eq!(apply_action(9003, "cancel"), Ok(true));
        assert!(queue().iter().all(|d| d.id != 9003));
        assert_eq!(apply_action(9001, "pause"), Ok(true));
        assert_eq!(crate::hub::download_stop(), crate::hub::STOP_PAUSE);
        assert_eq!(apply_action(9001, "cancel"), Ok(true));
        assert_eq!(crate::hub::download_stop(), crate::hub::STOP_CANCEL);
        crate::hub::set_download_stop(crate::hub::STOP_NONE);
        assert_eq!(apply_action(404, "pause"), Ok(false));
        assert!(apply_action(9001, "explode").is_err());
        queue().clear();
    }
}
