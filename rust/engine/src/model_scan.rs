//! Finds model files that are already on this computer, so the Studio can offer them instead of a download.
//!
//! The Studio's own folder (and the Hugging Face cache) hold models in the layout of the Hugging Face tools. A model the user got
//! another way is not there: a GGUF file in the Downloads folder or the models folder itself, a model of LM Studio, GPT4All or
//! llama.cpp, or one pulled by Ollama (a blob next to a manifest). Nothing is copied or changed: the files stay where they are and
//! are only listed. The search is shallow and bounded, so it is cheap enough to run whenever the model page is opened.
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

/// A model file found outside the Studio's own model list.
#[derive(Debug, Clone, PartialEq)]
pub struct Found {
    pub path: PathBuf,
    /// The file name, or `name:tag` for an Ollama model.
    pub name: String,
    pub size: u64,
    /// Where it was found, in words for a person ("Downloads", "LM Studio", "Ollama").
    pub source: &'static str,
}

/// Files smaller than this are not language models (test fixtures, projectors).
const MIN_BYTES: u64 = 8 << 20;
const MAX_FOUND: usize = 200;
const MAX_DIRS: usize = 400;

fn home() -> PathBuf {
    std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).filter(|v| !v.is_empty()).map(PathBuf::from).unwrap_or_default()
}

fn has_gguf_magic(path: &Path) -> bool {
    let mut magic = [0u8; 4];
    fs::File::open(path).and_then(|mut f| f.read_exact(&mut magic)).is_ok() && &magic == b"GGUF"
}

/// Whether a file name is the first (or only) part of a model: not a vision projector, not the second shard of a split file.
fn is_model_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    if !lower.ends_with(".gguf") || lower.contains("mmproj") {
        return false;
    }
    match lower.rfind("-of-") {
        Some(at) => {
            let before = &lower[..at];
            let part = before.rsplit('-').next().unwrap_or("");
            !(part.len() == 5 && part.chars().all(|c| c.is_ascii_digit())) || part == "00001"
        }
        None => true,
    }
}

fn collect(dir: &Path, depth: usize, budget: &mut usize, out: &mut Vec<(PathBuf, u64)>) {
    if *budget == 0 || out.len() >= MAX_FOUND {
        return;
    }
    *budget -= 1;
    let Ok(entries) = fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let Ok(kind) = entry.file_type() else { continue };
        let path = entry.path();
        if kind.is_symlink() {
            continue;
        }
        if kind.is_dir() {
            if depth > 0 {
                collect(&path, depth - 1, budget, out);
            }
        } else if kind.is_file() && is_model_name(&entry.file_name().to_string_lossy()) {
            let Ok(meta) = entry.metadata() else { continue };
            if meta.len() >= MIN_BYTES && has_gguf_magic(&path) {
                out.push((path, meta.len()));
            }
        }
    }
}

/// Models pulled by Ollama: `<dir>/manifests/<host>/<namespace>/<model>/<tag>` names the blob (a GGUF file without extension).
pub fn ollama_models(dir: &Path) -> Vec<Found> {
    let manifests = dir.join("manifests");
    let mut files = Vec::new();
    let mut stack = vec![(manifests.clone(), 0usize)];
    while let Some((folder, depth)) = stack.pop() {
        let Ok(entries) = fs::read_dir(&folder) else { continue };
        for entry in entries.flatten() {
            let Ok(kind) = entry.file_type() else { continue };
            if kind.is_dir() && depth < 5 {
                stack.push((entry.path(), depth + 1));
            } else if kind.is_file() {
                files.push(entry.path());
            }
            if files.len() >= MAX_FOUND {
                break;
            }
        }
    }
    let mut out = Vec::new();
    for manifest in files {
        let Ok(text) = fs::read_to_string(&manifest) else { continue };
        let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else { continue };
        let Some(digest) = json["layers"].as_array().and_then(|layers| {
            layers.iter().find(|l| l["mediaType"].as_str() == Some("application/vnd.ollama.image.model")).and_then(|l| l["digest"].as_str())
        }) else {
            continue;
        };
        let blob = dir.join("blobs").join(digest.replace(':', "-"));
        let Ok(meta) = fs::metadata(&blob) else { continue };
        if meta.len() < MIN_BYTES || !has_gguf_magic(&blob) {
            continue;
        }
        // library/qwen3/8b -> qwen3:8b ; someone/model/tag -> someone/model:tag
        let rel: Vec<String> = manifest.strip_prefix(&manifests).unwrap_or(&manifest).components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
        let parts: Vec<&String> = rel.iter().skip(1).collect(); // the first is the registry host
        let Some((tag, path)) = parts.split_last() else { continue };
        let path: Vec<&str> = path.iter().map(|s| s.as_str()).skip_while(|s| *s == "library").collect();
        out.push(Found { path: blob, name: format!("{}:{}", path.join("/"), tag), size: meta.len(), source: "Ollama" });
    }
    out
}

/// Searches the given places: `(folder, how many levels below it, label)`.
pub fn scan_in(places: &[(PathBuf, usize, &'static str)], ollama: &[PathBuf]) -> Vec<Found> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    let mut budget = MAX_DIRS;
    for (dir, depth, source) in places {
        let mut files = Vec::new();
        collect(dir, *depth, &mut budget, &mut files);
        for (path, size) in files {
            let key = fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
            if seen.insert(key) {
                out.push(Found { name: path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(), path, size, source });
            }
        }
    }
    for dir in ollama {
        for found in ollama_models(dir) {
            let key = fs::canonicalize(&found.path).unwrap_or_else(|_| found.path.clone());
            if seen.insert(key) {
                out.push(found);
            }
        }
    }
    out.truncate(MAX_FOUND);
    out
}

/// The usual places of this computer, plus the Studio's models folder itself (a file put straight into it is not in the cache layout).
pub fn scan() -> Vec<Found> {
    let home = home();
    let var = |name: &str| std::env::var_os(name).filter(|v| !v.is_empty()).map(PathBuf::from);
    let mut places: Vec<(PathBuf, usize, &'static str)> = vec![
        (crate::hub::cache_root(), 1, "models folder"),
        (home.join("Downloads"), 0, "Downloads"),
        (home.join("models"), 2, "models folder in your home"),
        (home.join(".lmstudio").join("models"), 3, "LM Studio"),
        (home.join(".cache").join("lm-studio").join("models"), 3, "LM Studio"),
        (home.join(".cache").join("llama.cpp"), 0, "llama.cpp"),
        (home.join(".local").join("share").join("nomic.ai").join("GPT4All"), 0, "GPT4All"),
    ];
    if let Some(dir) = var("LLAMA_CACHE") {
        places.push((dir, 0, "llama.cpp"));
    }
    if let Some(local) = var("LOCALAPPDATA") {
        places.push((local.join("llama.cpp"), 0, "llama.cpp"));
        places.push((local.join("nomic.ai").join("GPT4All"), 0, "GPT4All"));
    }
    let mut ollama = vec![home.join(".ollama").join("models")];
    if let Some(dir) = var("OLLAMA_MODELS") {
        ollama.insert(0, dir);
    }
    scan_in(&places, &ollama)
}

fn squash(text: &str) -> String {
    text.chars().filter(char::is_ascii_alphanumeric).map(|c| c.to_ascii_lowercase()).collect()
}

/// Whether a model file is the catalog model `entry` (for example `Qwen3.5-9B-Q4_K_M.gguf` is `qwen3.5:9b`). The name has to appear with
/// no further digit after it, so `qwen3:8b` does not match `Qwen3-80B`.
pub fn file_is_entry(file_name: &str, entry: &str) -> bool {
    let name = squash(file_name.strip_suffix(".gguf").or_else(|| file_name.strip_suffix(".GGUF")).unwrap_or(file_name));
    let wanted = squash(entry.strip_suffix(":latest").unwrap_or(entry));
    if wanted.len() < 4 {
        return false;
    }
    let mut from = 0;
    while let Some(at) = name[from..].find(&wanted) {
        let end = from + at + wanted.len();
        if !name[end..].chars().next().is_some_and(|c| c.is_ascii_digit()) {
            return true;
        }
        from += at + 1;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fllm-scan-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn gguf(path: &Path, bytes: usize) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut data = b"GGUF".to_vec();
        data.resize(bytes, 0);
        fs::write(path, data).unwrap();
    }

    #[test]
    fn catalog_names_match_file_names() {
        assert!(file_is_entry("Qwen3.5-9B-Q4_K_M.gguf", "qwen3.5:9b"));
        assert!(file_is_entry("qwen3.5-9b.Q8_0.gguf", "qwen3.5:9b"));
        assert!(file_is_entry("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "llama3.1:8b"));
        assert!(file_is_entry("gemma-3-4b-it-Q4_K_M.gguf", "gemma3:4b"));
        assert!(!file_is_entry("Qwen3.5-90B-Q4_K_M.gguf", "qwen3.5:9b"), "a longer number is another model");
        assert!(!file_is_entry("Qwen3.5-4B-Q4_K_M.gguf", "qwen3.5:9b"));
        assert!(!file_is_entry("anything.gguf", "x"), "very short names never match");
    }

    #[test]
    fn model_files_are_found_and_projectors_and_later_shards_are_not() {
        assert!(is_model_name("model.gguf") && is_model_name("m-00001-of-00003.gguf"));
        assert!(!is_model_name("m-00002-of-00003.gguf") && !is_model_name("mmproj-f16.gguf") && !is_model_name("notes.txt"));
        let dir = temp("find");
        gguf(&dir.join("Downloads").join("Qwen3.5-9B-Q4_K_M.gguf"), 9 << 20);
        gguf(&dir.join("Downloads").join("tiny.gguf"), 1 << 20);
        fs::write(dir.join("Downloads").join("fake.gguf"), vec![0u8; 9 << 20]).unwrap();
        gguf(&dir.join("lm").join("pub").join("repo").join("m.gguf"), 9 << 20);
        gguf(&dir.join("lm").join("a").join("b").join("c").join("d").join("deep.gguf"), 9 << 20);
        let found = scan_in(&[(dir.join("Downloads"), 0, "Downloads"), (dir.join("lm"), 3, "LM Studio")], &[]);
        let mut names: Vec<_> = found.iter().map(|f| f.name.as_str()).collect();
        names.sort();
        assert_eq!(names, ["Qwen3.5-9B-Q4_K_M.gguf", "m.gguf"], "small files, files without the GGUF header and deep ones are left out");
        assert_eq!(found.iter().find(|f| f.name == "m.gguf").unwrap().source, "LM Studio");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn ollama_blobs_are_named_after_their_manifest() {
        let dir = temp("ollama");
        gguf(&dir.join("blobs").join("sha256-abc"), 9 << 20);
        let manifest = dir.join("manifests").join("registry.ollama.ai").join("library").join("qwen3").join("8b");
        fs::create_dir_all(manifest.parent().unwrap()).unwrap();
        fs::write(&manifest, r#"{"layers":[{"mediaType":"application/vnd.ollama.image.template","digest":"sha256:no"},{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:abc"}]}"#).unwrap();
        let found = ollama_models(&dir);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, "qwen3:8b");
        assert!(found[0].path.ends_with("sha256-abc"));
        assert!(file_is_entry(&found[0].name, "qwen3:8b"));
        let _ = fs::remove_dir_all(&dir);
    }
}
