//! Tells a person what a path, a folder or a repository name is before it is loaded: the format, the size, the architecture, the
//! quantization, whether it fits into the memory, whether it carries multi-token prediction heads, and, when it is not a model, why
//! not and what to do about it.
//!
//! Nothing here loads weights. A GGUF file is only mapped to read its header; a checkpoint folder is judged by `config.json`, the
//! file names and the shard index. The same answer drives the Studio (model dialogs) and `POST /api/studio/models/inspect`.
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};

/// What was found. `path` is what the server is started with (`--model`): a GGUF file (the first shard of a split file), a checkpoint
/// folder, or a `owner/repo[:quant]` reference.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Inspection {
    /// `gguf`, `gguf_split`, `checkpoint`, `repo`, `folder_of_models` or `not_a_model`.
    pub kind: &'static str,
    /// Whether the model can be loaded as it is.
    pub ok: bool,
    pub path: String,
    /// A short name for a person.
    pub title: String,
    /// In words: "GGUF file", "Hugging Face checkpoint (safetensors)", ...
    pub format: String,
    pub size_bytes: u64,
    pub files: usize,
    pub architecture: Option<String>,
    pub quantization: Option<String>,
    pub context_length: Option<u64>,
    /// `Some(true)`: the model has multi-token prediction heads; `Some(false)`: it has none; `None`: not known without downloading.
    pub mtp: Option<bool>,
    /// Why speculative decoding with the heads of the model or a draft model cannot work for this architecture.
    pub speculation_blocked: Option<String>,
    pub fits: Option<bool>,
    pub notes: Vec<String>,
    pub problem: Option<String>,
    pub fix: Option<String>,
    /// For a folder with several models (or a folder that contains model folders): paths the user can choose from.
    pub candidates: Vec<(String, String)>,
}

impl Inspection {
    pub fn to_json(&self) -> Value {
        json!({
            "kind": self.kind, "ok": self.ok, "path": self.path, "title": self.title, "format": self.format,
            "size_bytes": self.size_bytes, "files": self.files, "architecture": self.architecture,
            "quantization": self.quantization, "context_length": self.context_length, "mtp": self.mtp,
            "speculation_blocked": self.speculation_blocked, "fits": self.fits, "notes": self.notes,
            "problem": self.problem, "fix": self.fix,
            "candidates": self.candidates.iter().map(|(path, label)| json!({"path": path, "label": label})).collect::<Vec<_>>(),
        })
    }

    fn broken(kind: &'static str, path: &str, title: &str, problem: impl Into<String>, fix: impl Into<String>) -> Self {
        Self { kind, path: path.to_owned(), title: title.to_owned(), problem: Some(problem.into()), fix: Some(fix.into()), ..Self::default() }
    }
}

/// Architectures with recurrent layers: rejected draft tokens would advance the recurrent state, so they cannot speculate.
pub const STATEFUL_ARCHITECTURES: &[&str] = &[
    "qwen3next", "qwen35", "qwen35moe", "olmo_hybrid", "mamba", "mamba2", "jamba", "granitehybrid", "bamba", "nemotron_h",
    "nemotron_h_moe", "falcon-h1", "zamba2", "lfm2", "lfm2moe", "rwkv6", "rwkv7",
];

fn blocked_text(architecture: &str) -> Option<String> {
    STATEFUL_ARCHITECTURES.contains(&architecture).then(|| {
        format!(
            "Speculative decoding does not work with this model: its architecture ({architecture}) keeps a running state that cannot be rolled back when a guess is wrong."
        )
    })
}

fn fits(size: u64) -> Option<bool> {
    let free = crate::catalog::available_memory_bytes();
    (free > 0 && size > 0).then_some(size as f64 * 1.15 <= free as f64)
}

fn name_of(path: &Path) -> String {
    path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| path.display().to_string())
}

fn shown(path: &Path) -> String {
    let text = path.display().to_string();
    text.strip_prefix(r"\\?\").map(str::to_owned).unwrap_or(text)
}

/// `(prefix, number, total)` of `name-00002-of-00005.gguf`.
fn shard_of(name: &str) -> Option<(String, u32, u32)> {
    let stem = name.strip_suffix(".gguf").or_else(|| name.strip_suffix(".GGUF"))?;
    let (rest, total) = stem.rsplit_once("-of-")?;
    let (prefix, number) = rest.rsplit_once('-')?;
    if number.len() != 5 || total.len() != 5 {
        return None;
    }
    Some((prefix.to_owned(), number.parse().ok()?, total.parse().ok()?))
}

fn shard_name(prefix: &str, number: u32, total: u32) -> String {
    format!("{prefix}-{number:05}-of-{total:05}.gguf")
}

fn has_gguf_magic(path: &Path) -> bool {
    use std::io::Read;
    let mut magic = [0u8; 4];
    fs::File::open(path).and_then(|mut f| f.read_exact(&mut magic)).is_ok() && &magic == b"GGUF"
}

/// The quantization of a GGUF file: the `general.file_type` of the header, else the name of the file.
fn gguf_quantization(file_type: Option<u64>, file_name: &str) -> Option<String> {
    let named = match file_type? {
        0 => "F32", 1 => "F16", 2 => "Q4_0", 3 => "Q4_1", 7 => "Q8_0", 8 => "Q5_0", 9 => "Q5_1", 10 => "Q2_K", 11 => "Q3_K_S",
        12 => "Q3_K_M", 13 => "Q3_K_L", 14 => "Q4_K_S", 15 => "Q4_K_M", 16 => "Q5_K_S", 17 => "Q5_K_M", 18 => "Q6_K",
        19 => "IQ2_XXS", 20 => "IQ2_XS", 21 => "Q2_K_S", 22 => "IQ3_XS", 23 => "IQ3_XXS", 24 => "IQ1_S", 25 => "IQ4_NL",
        26 => "IQ3_S", 27 => "IQ3_M", 28 => "IQ2_S", 29 => "IQ2_M", 30 => "IQ4_XS", 31 => "IQ1_M", 32 => "BF16",
        _ => "",
    };
    if named.is_empty() { crate::hub::quant_of_filename(file_name) } else { Some(named.to_owned()) }
}

fn inspect_gguf(path: &Path) -> Inspection {
    let file = name_of(path);
    let here = shown(path);
    if file.to_ascii_lowercase().contains("mmproj") {
        return Inspection::broken(
            "not_a_model",
            &here,
            &file,
            "This is a vision projector (mmproj), the add-on of a model that reads images, not a model that can answer.",
            "Choose the main model file of the same download; fLLM finds the projector next to it by itself.",
        );
    }
    // a later shard of a split file: the set is loaded from the first one
    let mut notes = Vec::new();
    let mut first = path.to_path_buf();
    let mut shards = vec![path.to_path_buf()];
    if let Some((prefix, number, total)) = shard_of(&file) {
        let dir = path.parent().unwrap_or(Path::new("."));
        shards = (1..=total).map(|n| dir.join(shard_name(&prefix, n, total))).collect();
        if let Some(missing) = shards.iter().find(|s| !s.is_file()) {
            return Inspection::broken(
                "gguf_split",
                &here,
                &file,
                format!("This is part {number} of {total} of a split model, but {} is missing.", name_of(missing)),
                "Put all the parts of the model in one folder (the download is not complete), then choose the first part.",
            );
        }
        first = shards[0].clone();
        if number != 1 {
            notes.push(format!("You chose part {number} of {total}. The model is loaded from part 1; all {total} parts are used."));
        }
    }
    if !has_gguf_magic(&first) {
        return Inspection::broken(
            "not_a_model",
            &here,
            &file,
            "This file does not start with the GGUF header, so it is not a GGUF model (it may be damaged or cut off).",
            "Download the file again, or choose a .safetensors folder (a Hugging Face model) instead.",
        );
    }
    let mapping = match fllm_gguf_probe::MappedGguf::open(&first) {
        Ok(mapping) => mapping,
        Err(e) => {
            return Inspection::broken(
                "not_a_model",
                &here,
                &file,
                format!("The GGUF file cannot be read: {e}."),
                "The download is probably incomplete. Download the file again.",
            );
        }
    };
    let size: u64 = shards.iter().filter_map(|s| fs::metadata(s).ok()).map(|m| m.len()).sum();
    let index = mapping.index();
    let text = |key: &str| index.metadata.get(key).and_then(Value::as_str).map(str::to_owned);
    let architecture = text("general.architecture");
    let context_length = architecture.as_deref().and_then(|a| index.metadata.get(&format!("{a}.context_length"))).and_then(Value::as_u64);
    let mtp = index.tensors.iter().any(|t| t.name.contains(".nextn."));
    let usable = crate::tokenizer::TokenizerBundle::from_gguf(&mapping).map(|_| ());
    let mut out = Inspection {
        kind: if shards.len() > 1 { "gguf_split" } else { "gguf" },
        ok: true,
        path: shown(&first),
        title: text("general.name").filter(|n| !n.trim().is_empty()).unwrap_or_else(|| file.clone()),
        format: if shards.len() > 1 { format!("GGUF file, split into {} parts", shards.len()) } else { "GGUF file".into() },
        size_bytes: size,
        files: shards.len(),
        quantization: gguf_quantization(index.metadata.get("general.file_type").and_then(Value::as_u64), &file),
        context_length,
        mtp: Some(mtp),
        speculation_blocked: architecture.as_deref().and_then(blocked_text),
        fits: fits(size),
        notes,
        architecture,
        ..Inspection::default()
    };
    if let Err(reason) = usable {
        out.ok = false;
        out.problem = Some(format!("fLLM cannot use this model yet: {reason}"));
        out.fix = Some("Try another quantization or another build of the same model; the supported architectures are listed in docs/architectures.md.".into());
    }
    out
}

fn is_weight_file(name: &str) -> bool {
    name.ends_with(".safetensors") || (name.starts_with("pytorch_model") && name.ends_with(".bin"))
}

const INDEXES: &[&str] = &["model.safetensors.index.json", "pytorch_model.bin.index.json"];

fn mtp_config(config: &Value) -> bool {
    ["num_nextn_predict_layers", "num_mtp_layers", "mtp_num_hidden_layers"]
        .iter()
        .any(|key| config.get(key).and_then(Value::as_u64).or_else(|| config["text_config"].get(key).and_then(Value::as_u64)).is_some_and(|n| n > 0))
}

fn inspect_checkpoint(root: &Path, picked: Option<&Path>) -> Inspection {
    let here = shown(root);
    let title = name_of(root);
    let names: Vec<String> = fs::read_dir(root)
        .map(|entries| entries.flatten().filter(|e| e.path().is_file()).filter_map(|e| e.file_name().into_string().ok()).collect())
        .unwrap_or_default();
    let has = |n: &str| names.iter().any(|x| x == n);
    let weights: Vec<&String> = names.iter().filter(|n| is_weight_file(n)).collect();
    if !has("config.json") && !has("params.json") {
        return Inspection::broken(
            "not_a_model",
            &here,
            &title,
            "This folder has weight files but no config.json, so fLLM cannot tell what architecture they belong to.",
            "Download the whole repository, not only the .safetensors files: config.json and the tokenizer files are needed next to them.",
        );
    }
    if weights.is_empty() {
        return Inspection::broken(
            "not_a_model",
            &here,
            &title,
            "This folder has a config.json but no weights (no .safetensors files).",
            "The download is not finished or the weights are in a sub-folder. Choose the folder that contains the .safetensors files, or download the model again.",
        );
    }
    let check = match fllm_checkpoint::check_config(root) {
        Ok(check) => check,
        Err(e) => {
            let text = e.to_string();
            let (problem, fix) = if text.contains("unsupported model_type") {
                (
                    format!("fLLM does not support this architecture yet ({text})."),
                    "Pick a model of a supported architecture (docs/architectures.md lists them) or a GGUF made by llama.cpp if the architecture is supported there.".to_owned(),
                )
            } else {
                (format!("This checkpoint cannot be used: {text}."), "Check that the download is complete and that config.json belongs to these weights.".to_owned())
            };
            return Inspection::broken("not_a_model", &here, &title, problem, fix);
        }
    };
    let tokenizer_names: Vec<&str> = names.iter().map(String::as_str).collect();
    let tokenizer = crate::import::has_tokenizer_source(&tokenizer_names);
    // shards named by the index must all be there
    let mut shard_files: Vec<String> = Vec::new();
    let mut mtp_in_index = false;
    for index in INDEXES.iter().filter(|i| has(i)) {
        if let Ok(value) = fs::read(root.join(index)).map_err(|e| e.to_string()).and_then(|b| serde_json::from_slice::<Value>(&b).map_err(|e| e.to_string())) {
            if let Some(map) = value["weight_map"].as_object() {
                for (tensor, file) in map {
                    if let Some(file) = file.as_str() {
                        if !shard_files.iter().any(|s| s == file) {
                            shard_files.push(file.to_owned());
                        }
                    }
                    mtp_in_index |= tensor.starts_with("mtp.") || tensor.contains(".nextn.") || tensor.contains("mtp_layers");
                }
            }
        }
    }
    if let Some(missing) = shard_files.iter().find(|s| !root.join(s).is_file()) {
        return Inspection::broken(
            "checkpoint",
            &here,
            &title,
            format!("The weights are split into {} files, but {missing} is missing.", shard_files.len()),
            "The download is not complete. Download the missing file into this folder (or the whole model again).",
        );
    }
    let counted: Vec<&String> = if shard_files.is_empty() { weights.clone() } else { weights.iter().copied().filter(|w| shard_files.contains(w)).collect() };
    let size: u64 = counted.iter().filter_map(|w| fs::metadata(root.join(w)).ok()).map(|m| m.len()).sum();
    let config = fllm_checkpoint::read_checkpoint_config(root).unwrap_or(Value::Null);
    let mut notes = Vec::new();
    if let Some(file) = picked.and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned()).filter(|n| n.ends_with(".safetensors") || n.ends_with(".json")) {
        notes.push(format!("You chose {file}. A model is a whole folder: all of its files (config.json, tokenizer, every weight part) are loaded together."));
    }
    let format = match check.quantization.as_str() {
        "none" => "Hugging Face checkpoint (safetensors, unquantized)".to_owned(),
        q => format!("Hugging Face checkpoint (safetensors, {q})"),
    };
    let mut out = Inspection {
        kind: "checkpoint",
        ok: true,
        path: here.clone(),
        title,
        format,
        size_bytes: size,
        files: counted.len(),
        quantization: Some(if check.quantization == "none" { "unquantized".into() } else { check.quantization.clone() }),
        context_length: ["max_position_embeddings", "n_positions", "seq_length"]
            .iter()
            .find_map(|k| config.get(k).and_then(Value::as_u64).or_else(|| config["text_config"].get(k).and_then(Value::as_u64))),
        mtp: Some(mtp_config(&config) || mtp_in_index),
        speculation_blocked: blocked_text(&check.architecture),
        fits: fits(size),
        notes,
        architecture: Some(check.architecture),
        ..Inspection::default()
    };
    if !tokenizer {
        out.ok = false;
        out.problem = Some("The tokenizer files are missing (no tokenizer.json, tokenizer.model or similar), so the text cannot be turned into tokens.".into());
        out.fix = Some("Download tokenizer.json and tokenizer_config.json of the same model into this folder.".into());
    } else if out.quantization.as_deref() == Some("unquantized") && size > 0 {
        out.notes.push("The weights are not quantized, so the model needs about as much memory as its size. Start fllm-rs with --quant (for example --quant q4_k_m) to quantize while loading, or use a GGUF file.".into());
    }
    out
}

fn gguf_in(dir: &Path) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = fs::read_dir(dir)
        .map(|entries| entries.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();
    found.retain(|p| {
        let name = name_of(p);
        let lower = name.to_ascii_lowercase();
        p.is_file()
            && lower.ends_with(".gguf")
            && !lower.contains("mmproj")
            && !shard_of(&name).is_some_and(|(_, n, _)| n != 1)
    });
    found.sort();
    found
}

fn inspect_folder(dir: &Path) -> Inspection {
    let here = shown(dir);
    let title = name_of(dir);
    let ggufs = gguf_in(dir);
    let is_checkpoint = (dir.join("config.json").is_file() || dir.join("params.json").is_file())
        || fs::read_dir(dir).is_ok_and(|e| e.flatten().any(|f| is_weight_file(&f.file_name().to_string_lossy())));
    if ggufs.len() == 1 && !is_checkpoint {
        return inspect_gguf(&ggufs[0]);
    }
    if is_checkpoint && ggufs.is_empty() {
        return inspect_checkpoint(dir, None);
    }
    if is_checkpoint && ggufs.len() == 1 {
        let mut out = inspect_gguf(&ggufs[0]);
        out.notes.push("The folder also holds checkpoint files; the GGUF file is used.".into());
        return out;
    }
    if ggufs.len() > 1 {
        return Inspection {
            kind: "folder_of_models",
            ok: false,
            path: here,
            title,
            format: format!("Folder with {} GGUF models", ggufs.len()),
            problem: Some(format!("This folder holds {} GGUF models; a server loads one at a time.", ggufs.len())),
            fix: Some("Choose one of the models, or pick a single .gguf file.".into()),
            candidates: ggufs.iter().map(|p| (shown(p), name_of(p))).collect(),
            ..Inspection::default()
        };
    }
    // nothing here: look one level down for models, so the answer can point at them
    let mut below = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten().take(400) {
            let path = entry.path();
            if path.is_dir() {
                let sub = gguf_in(&path);
                if path.join("config.json").is_file() {
                    below.push((shown(&path), format!("{} (Hugging Face folder)", name_of(&path))));
                } else if let Some(first) = sub.first() {
                    below.push((shown(first), format!("{}/{}", name_of(&path), name_of(first))));
                }
            }
            if below.len() >= 20 {
                break;
            }
        }
    }
    let mut out = Inspection::broken(
        "not_a_model",
        &here,
        &title,
        "This folder is not a model: it holds no .gguf file and no Hugging Face checkpoint (config.json together with .safetensors files).",
        if below.is_empty() {
            "Choose a .gguf file, or the folder that directly contains config.json and the .safetensors files. For a model on Hugging Face type owner/name."
        } else {
            "The folder has models in sub-folders; choose one of those."
        },
    );
    if !below.is_empty() {
        out.kind = "folder_of_models";
        out.format = format!("Folder with {} models in sub-folders", below.len());
        out.candidates = below;
    }
    out
}

fn inspect_repo(reference: &str) -> Inspection {
    let normalized = crate::hub::normalize_reference(reference);
    let parsed = match crate::hub::parse_reference(&normalized) {
        Ok(parsed) => parsed,
        Err(e) => {
            return Inspection::broken("not_a_model", reference, reference, e, "Type owner/name, for example Qwen/Qwen2.5-0.5B-Instruct, or add :Q4_K_M to pick a quantization.");
        }
    };
    let repo = format!("{}/{}", parsed.owner, parsed.repo);
    let files = match crate::hub::repository_files(&repo) {
        Ok(files) => files,
        Err(e) => {
            return Inspection::broken(
                "repo",
                reference,
                &repo,
                format!("Hugging Face did not list {repo}: {e}"),
                "Check the spelling (owner/name), your internet connection, or sign in under Settings, Models for private and gated models.",
            );
        }
    };
    let listed: Vec<&str> = files.iter().map(|(n, _)| n.as_str()).collect();
    let ggufs: Vec<&(String, u64)> = files.iter().filter(|(n, _)| n.to_ascii_lowercase().ends_with(".gguf") && !n.to_ascii_lowercase().contains("mmproj")).collect();
    let checkpoint = listed.contains(&"config.json") && files.iter().any(|(n, _)| is_weight_file(n));
    let mut out = Inspection { kind: "repo", ok: true, path: normalized.clone(), title: repo.clone(), ..Inspection::default() };
    if let Some(quant) = &parsed.quant {
        let wanted = quant.to_ascii_lowercase();
        let matching: Vec<&&(String, u64)> = ggufs.iter().filter(|(n, _)| n.to_ascii_lowercase().contains(&wanted)).collect();
        if matching.is_empty() {
            let available: Vec<String> = ggufs.iter().filter_map(|(n, _)| crate::hub::quant_of_filename(n)).collect();
            out.ok = false;
            out.problem = Some(format!("{repo} has no GGUF file for {quant}."));
            out.fix = Some(if available.is_empty() { "Leave out :quantization to load the checkpoint itself.".into() } else { format!("Available: {}.", available.join(", ")) });
            return out;
        }
        out.size_bytes = matching.iter().map(|(_, s)| *s).sum();
        out.files = matching.len();
        out.format = "GGUF file from Hugging Face (downloaded first)".into();
        out.quantization = Some(quant.to_ascii_uppercase());
    } else if checkpoint {
        out.size_bytes = files.iter().filter(|(n, _)| is_weight_file(n)).map(|(_, s)| *s).sum();
        out.files = files.iter().filter(|(n, _)| is_weight_file(n)).count();
        out.format = "Hugging Face checkpoint (safetensors, downloaded first)".into();
    } else if !ggufs.is_empty() {
        out.format = format!("Hugging Face repository with {} GGUF files (the best one for your memory is chosen)", ggufs.len());
        out.size_bytes = ggufs.iter().map(|(_, s)| *s).min().unwrap_or(0);
        out.notes.push("Add :Q4_K_M (or another quantization of the list) to the name to choose one yourself.".into());
    } else {
        out.ok = false;
        out.problem = Some(format!("{repo} holds neither GGUF files nor a checkpoint with config.json and .safetensors weights."));
        out.fix = Some("Choose another repository. A tool-specific format (ONNX, MLX-only, TensorRT) cannot be loaded.".into());
        return out;
    }
    out.fits = fits(out.size_bytes);
    out.notes.push("The files are downloaded when you load it. Whether it has multi-token prediction heads is known after the download.".into());
    out
}

/// Whether `text` should be treated as a Hugging Face reference rather than a path of this computer.
fn is_reference(text: &str) -> bool {
    !Path::new(text).exists() && crate::hub::normalize_reference(text).starts_with("hf://")
}

/// What `input` (a path or a Hugging Face reference) is. Blocks for a reference (it asks Hugging Face for the file list).
pub fn inspect(input: &str) -> Inspection {
    let cleaned = input.trim().trim_matches(|c| c == '"' || c == '\'').trim();
    if cleaned.is_empty() {
        return Inspection::broken("not_a_model", "", "", "Nothing was entered.", "Type or paste a path, or choose a file or folder with Browse.");
    }
    if is_reference(cleaned) {
        return inspect_repo(cleaned);
    }
    let path = PathBuf::from(cleaned);
    if !path.exists() {
        return Inspection::broken(
            "not_a_model",
            cleaned,
            cleaned,
            "There is no file or folder with this name on this computer, and it is not a Hugging Face name (owner/name).",
            "Check the path for typing errors (the Browse button lists what is there), or type a Hugging Face name like Qwen/Qwen2.5-0.5B-Instruct.",
        );
    }
    if path.is_dir() {
        return inspect_folder(&path);
    }
    let lower = name_of(&path).to_ascii_lowercase();
    if lower.ends_with(".gguf") {
        return inspect_gguf(&path);
    }
    let root = path.parent().unwrap_or(Path::new(".")).to_path_buf();
    if lower.ends_with(".safetensors") || lower.ends_with(".json") || lower.ends_with(".bin") || lower.ends_with(".pt") || lower.ends_with(".pth") {
        if lower.ends_with(".bin") && !root.join("config.json").is_file() {
            return Inspection::broken(
                "not_a_model",
                cleaned,
                &name_of(&path),
                "This .bin file has no config.json beside it, so it is not a model fLLM can identify (older llama.cpp .bin formats are not supported).",
                "Use a .gguf file, or a Hugging Face folder with config.json.",
            );
        }
        return inspect_checkpoint(&root, Some(&path));
    }
    Inspection::broken(
        "not_a_model",
        cleaned,
        &name_of(&path),
        format!("fLLM cannot read a {} file as a model.", path.extension().map(|e| format!(".{}", e.to_string_lossy())).unwrap_or_else(|| "this".into())),
        "Choose a .gguf file, a .safetensors file or the model's folder (the one with config.json), or type a Hugging Face name.",
    )
}

/// Entries of a folder for the browser in the Studio: folders first, then the files that can be a model.
pub fn browse(requested: &str) -> Result<Value, String> {
    let cleaned = requested.trim().trim_matches(|c| c == '"' || c == '\'').trim();
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).filter(|v| !v.is_empty()).map(PathBuf::from);
    let mut roots: Vec<Value> = Vec::new();
    if let Some(home) = &home {
        roots.push(json!({"label": "Home", "path": shown(home)}));
        for sub in ["Downloads", "Documents", "Desktop", "models"] {
            if home.join(sub).is_dir() {
                roots.push(json!({"label": sub, "path": shown(&home.join(sub))}));
            }
        }
    }
    let cache = crate::hub::cache_root();
    if cache.is_dir() {
        roots.push(json!({"label": "Models folder of fLLM", "path": shown(&cache)}));
    }
    if cfg!(windows) {
        for letter in b'A'..=b'Z' {
            let drive = format!("{}:\\", letter as char);
            if Path::new(&drive).exists() {
                roots.push(json!({"label": format!("Drive {}:", letter as char), "path": drive}));
            }
        }
    } else {
        roots.push(json!({"label": "Computer (/)", "path": "/"}));
    }
    let dir = if cleaned.is_empty() {
        home.clone().unwrap_or_else(|| PathBuf::from(if cfg!(windows) { "C:\\" } else { "/" }))
    } else {
        PathBuf::from(cleaned)
    };
    let dir = if dir.is_file() { dir.parent().map(Path::to_path_buf).unwrap_or(dir) } else { dir };
    if !dir.is_dir() {
        return Err(format!("{} is not a folder on this computer.", dir.display()));
    }
    let read = fs::read_dir(&dir).map_err(|e| format!("The folder cannot be opened: {e}"))?;
    let mut folders = Vec::new();
    let mut files = Vec::new();
    let mut truncated = false;
    for (n, entry) in read.flatten().enumerate() {
        if n >= 3000 {
            truncated = true;
            break;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        let Ok(kind) = entry.file_type() else { continue };
        let path = entry.path();
        if kind.is_dir() {
            let model = path.join("config.json").is_file() || path.join("params.json").is_file();
            folders.push(json!({"name": name, "path": shown(&path), "kind": "folder", "model_folder": model}));
        } else if kind.is_file() {
            let lower = name.to_ascii_lowercase();
            let (is_model, label) = if lower.ends_with(".gguf") {
                (!lower.contains("mmproj"), "GGUF")
            } else if lower.ends_with(".safetensors") {
                (true, "safetensors")
            } else {
                (false, "")
            };
            if is_model {
                let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                files.push(json!({"name": name, "path": shown(&path), "kind": "file", "format": label, "size_bytes": size}));
            }
        }
    }
    let sort = |list: &mut Vec<Value>| list.sort_by_key(|v| v["name"].as_str().unwrap_or_default().to_ascii_lowercase());
    sort(&mut folders);
    sort(&mut files);
    let mut entries = folders;
    entries.extend(files);
    Ok(json!({
        "path": shown(&dir),
        "parent": dir.parent().map(shown),
        "roots": roots,
        "entries": entries,
        "truncated": truncated,
        "separator": std::path::MAIN_SEPARATOR.to_string(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fllm-inspect-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn shard_names_are_read_and_rebuilt() {
        assert_eq!(shard_of("m-00002-of-00005.gguf"), Some(("m".into(), 2, 5)));
        assert_eq!(shard_name("m", 1, 5), "m-00001-of-00005.gguf");
        assert_eq!(shard_of("model-Q4_K_M.gguf"), None);
    }

    #[test]
    fn a_missing_path_and_an_empty_folder_get_a_fix() {
        let dir = temp("empty");
        let none = inspect(dir.join("nope.gguf").to_str().unwrap());
        assert!(!none.ok && none.problem.unwrap().contains("no file or folder") && none.fix.is_some());
        let empty = inspect(dir.to_str().unwrap());
        assert!(!empty.ok && empty.kind == "not_a_model" && empty.problem.unwrap().contains("not a model"));
        fs::write(dir.join("model.safetensors"), b"x").unwrap();
        let no_config = inspect(dir.to_str().unwrap());
        assert!(no_config.problem.unwrap().contains("no config.json"));
        fs::write(dir.join("config.json"), b"{}").unwrap();
        fs::remove_file(dir.join("model.safetensors")).unwrap();
        let no_weights = inspect(dir.to_str().unwrap());
        assert!(no_weights.problem.unwrap().contains("no weights"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_fake_gguf_is_refused_and_a_projector_is_named() {
        let dir = temp("fake");
        fs::write(dir.join("fake.gguf"), vec![0u8; 100]).unwrap();
        let fake = inspect(dir.join("fake.gguf").to_str().unwrap());
        assert!(!fake.ok && fake.problem.unwrap().contains("GGUF header"));
        fs::write(dir.join("mmproj-f16.gguf"), b"GGUF").unwrap();
        assert!(inspect(dir.join("mmproj-f16.gguf").to_str().unwrap()).problem.unwrap().contains("projector"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_shard_is_reported() {
        let dir = temp("split");
        fs::write(dir.join("m-00001-of-00002.gguf"), b"GGUF").unwrap();
        let out = inspect(dir.join("m-00001-of-00002.gguf").to_str().unwrap());
        assert!(!out.ok && out.problem.unwrap().contains("m-00002-of-00002.gguf is missing"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_folder_with_several_ggufs_lists_them_and_the_browser_flags_model_folders() {
        let dir = temp("several");
        fs::write(dir.join("a.gguf"), b"GGUF").unwrap();
        fs::write(dir.join("b.gguf"), b"GGUF").unwrap();
        fs::write(dir.join("mmproj-x.gguf"), b"GGUF").unwrap();
        let out = inspect(dir.to_str().unwrap());
        assert_eq!((out.kind, out.candidates.len()), ("folder_of_models", 2));
        let sub = dir.join("hf");
        fs::create_dir_all(&sub).unwrap();
        fs::write(sub.join("config.json"), b"{}").unwrap();
        let listing = browse(dir.to_str().unwrap()).unwrap();
        let entries = listing["entries"].as_array().unwrap();
        assert_eq!(entries[0]["name"], "hf");
        assert_eq!(entries[0]["model_folder"], true);
        assert_eq!(entries.len(), 3, "the projector is not offered: {entries:?}");
        assert!(browse(dir.join("a.gguf").to_str().unwrap()).is_ok(), "a file shows its folder");
        assert!(browse(dir.join("gone").to_str().unwrap()).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn multi_token_prediction_is_read_from_the_config() {
        assert!(mtp_config(&json!({"num_nextn_predict_layers": 1})));
        assert!(mtp_config(&json!({"text_config": {"num_mtp_layers": 2}})));
        assert!(!mtp_config(&json!({"num_nextn_predict_layers": 0})));
        assert!(blocked_text("qwen35").is_some() && blocked_text("llama").is_none());
    }
}
