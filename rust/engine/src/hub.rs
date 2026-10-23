//! Explicit Hugging Face references and immutable local snapshots, in Rust.
use reqwest::blocking::Client;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Component, Path, PathBuf},
    time::{Duration, Instant},
};

#[derive(Debug, PartialEq, Eq)]
pub struct Reference {
    pub owner: String,
    pub repo: String,
    pub file: Option<String>,
    pub quant: Option<String>,
}

/// Turn what people paste into a terminal into an `hf://owner/repo[:QUANT][/file]` reference: the address bar URL of a model page
/// or file (`https://huggingface.co/owner/repo`, `.../blob/main/file.gguf`, `.../resolve/main/file.gguf?download=true`), the
/// `hf.co/owner/repo:Q4_K_M` form and a bare `owner/repo[:QUANT]`. Anything that exists as a local path, starts like a path, or is
/// already an `hf://` reference is returned unchanged.
pub fn normalize_reference(input: &str) -> String {
    let input = input.trim();
    if input.starts_with("hf://") {
        return input.to_owned();
    }
    let mut rest = input;
    let mut web = false;
    for prefix in ["https://", "http://"] {
        if let Some(r) = rest.strip_prefix(prefix) {
            rest = r;
            web = true;
        }
    }
    for host in ["www.huggingface.co/", "huggingface.co/", "hf.co/"] {
        if let Some(r) = rest.strip_prefix(host) {
            rest = r;
            web = true;
            break;
        }
    }
    if web {
        rest = rest.split(['?', '#']).next().unwrap_or(rest);
    } else if Path::new(input).exists() || input.starts_with(['.', '/', '~', '\\']) || input.contains(':') && input.as_bytes().get(1) == Some(&b':') {
        return input.to_owned();
    }
    let parts: Vec<&str> = rest.trim_matches('/').split('/').collect();
    let plain = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b));
    let (head, quant) = parts.get(1).map_or(("", None), |r| r.split_once(':').map_or((*r, None), |(a, q)| (a, Some(q))));
    // "dir/file.gguf" is a relative path, not a repository
    let file_like = [".gguf", ".safetensors", ".bin", ".pt"].iter().any(|e| head.to_ascii_lowercase().ends_with(e));
    if parts.len() < 2 || !plain(parts[0]) || !plain(head) || (!web && (parts.len() != 2 || file_like)) {
        return input.to_owned();
    }
    let mut out = format!("hf://{}/{head}", parts[0]);
    if let Some(q) = quant {
        out.push(':');
        out.push_str(q);
    }
    // .../blob/REV/file or .../resolve/REV/file: the file is whatever follows the revision
    if parts.len() > 4 && matches!(parts[2], "blob" | "resolve") && quant.is_none() {
        out.push('/');
        out.push_str(&parts[4..].join("/"));
    }
    out
}

pub fn parse_reference(reference: &str) -> Result<Reference, String> {
    let reference = normalize_reference(reference);
    let raw = reference
        .strip_prefix("hf://")
        .ok_or("expected hf://owner/repository[/file]")?;
    let mut parts = raw.splitn(3, '/');
    let owner = parts.next().unwrap_or_default();
    let repository = parts.next().unwrap_or_default();
    let (repo, quant) = repository
        .split_once(':')
        .map_or((repository, None), |(repo, quant)| {
            (repo, Some(quant.to_owned()))
        });
    if quant.as_ref().is_some_and(|q| {
        q.is_empty()
            || !q
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
    }) {
        return Err("invalid GGUF quantization selector".into());
    }
    for name in [owner, repo] {
        if name.is_empty()
            || name == "."
            || name == ".."
            || !name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
        {
            return Err("invalid Hugging Face repository name".into());
        }
    }
    let file = parts.next().map(str::to_owned);
    if let Some(file) = &file {
        safe_relative(file)?;
        if quant.is_some() {
            return Err("use either a GGUF filename or :QUANT, not both".into());
        }
    }
    Ok(Reference {
        owner: owner.into(),
        repo: repo.into(),
        file,
        quant,
    })
}

fn safe_relative(name: &str) -> Result<(), String> {
    if name.is_empty()
        || name.contains('\\')
        || name.contains(':')
        || name
            .split('/')
            .any(|p| p.is_empty() || p == "." || p == "..")
        || Path::new(name)
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(format!("invalid relative model filename: {name}"));
    }
    Ok(())
}

fn hub_home() -> PathBuf {
    env::var_os("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            env::var_os("XDG_CACHE_HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    PathBuf::from(
                        env::var_os("HOME")
                            .or_else(|| env::var_os("USERPROFILE"))
                            .unwrap_or_default(),
                    )
                    .join(".cache")
                })
                .join("huggingface")
        })
}
/// The models folder chosen in the Studio (it wins over the environment, and applies at once).
static MODELS_DIR_OVERRIDE: std::sync::RwLock<Option<PathBuf>> = std::sync::RwLock::new(None);

/// Where downloaded models are kept: the folder chosen in the Studio, `FLLM_MODELS_DIR`, `FLLM_CACHE`, `HF_HUB_CACHE`, or the
/// Hugging Face cache of the user.
pub fn cache_root() -> PathBuf {
    if let Some(dir) = MODELS_DIR_OVERRIDE.read().unwrap_or_else(|e| e.into_inner()).clone() {
        return dir;
    }
    env::var_os("FLLM_MODELS_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(default_cache_root)
}

/// The cache folder without the Studio's models-folder setting; the Studio's own small files live next to it.
fn default_cache_root() -> PathBuf {
    env::var_os("FLLM_CACHE")
        .or_else(|| env::var_os("HF_HUB_CACHE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| hub_home().join("hub"))
}

/// The models folder when nothing is chosen in the Studio.
pub fn default_models_dir() -> PathBuf {
    default_cache_root()
}

/// Switches the models folder of this process (`None`: back to the default).
pub fn set_models_dir(dir: Option<PathBuf>) {
    *MODELS_DIR_OVERRIDE.write().unwrap_or_else(|e| e.into_inner()) = dir;
}

/// The models folder saved by the Studio, if any.
pub fn saved_models_dir() -> Option<String> {
    studio_setting("models_dir").and_then(|v| v.as_str().map(str::to_owned)).filter(|s| !s.is_empty())
}

/// Applies the saved models folder to this process (called when the Studio starts).
pub fn apply_saved_models_dir() {
    if let Some(dir) = saved_models_dir() {
        set_models_dir(Some(PathBuf::from(dir)));
    }
}
fn token() -> Option<String> {
    env::var("HF_TOKEN")
        .ok()
        .or_else(|| env::var("HUGGING_FACE_HUB_TOKEN").ok())
        .or_else(|| fs::read_to_string(hub_home().join("token")).ok())
        .map(|t| t.trim().to_owned())
        .filter(|t| !t.is_empty())
}
/// Where the access token is kept (the file the Hugging Face tools use too, so one login serves both).
fn token_file() -> PathBuf {
    hub_home().join("token")
}

/// The Hugging Face account of the token in use: `(name, where the token comes from)`; `None` without a token.
pub fn account() -> Result<Option<(String, &'static str)>, String> {
    let Some(token) = token() else {
        return Ok(None);
    };
    let source = if env::var("HF_TOKEN").is_ok() || env::var("HUGGING_FACE_HUB_TOKEN").is_ok() { "environment" } else { "saved" };
    whoami(&token).map(|name| Some((name, source)))
}

fn whoami(token: &str) -> Result<String, String> {
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(20))
        .timeout(Duration::from_secs(30))
        .user_agent("fllm-rust/0.1")
        .build()
        .map_err(|e| e.to_string())?;
    let response = client
        .get(url(&["api", "whoami-v2"])?)
        .bearer_auth(token)
        .send()
        .map_err(|e| format!("cannot reach Hugging Face: {e}"))?;
    if response.status() == reqwest::StatusCode::UNAUTHORIZED {
        return Err("Hugging Face does not accept this token".into());
    }
    if !response.status().is_success() {
        return Err(format!("Hugging Face answered {}", response.status()));
    }
    let body: Value = response.json().map_err(|e| e.to_string())?;
    Ok(body.get("name").and_then(Value::as_str).unwrap_or("unknown").to_owned())
}

/// Checks `token` with the Hub and keeps it for later downloads. An empty token signs out.
pub fn save_token(token: &str) -> Result<Option<String>, String> {
    let token = token.trim();
    let file = token_file();
    if token.is_empty() {
        let _ = fs::remove_file(&file);
        return Ok(None);
    }
    let name = whoami(token)?;
    if let Some(dir) = file.parent() {
        fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    fs::write(&file, token).map_err(|e| format!("cannot save the token: {e}"))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&file, fs::Permissions::from_mode(0o600));
    }
    Ok(Some(name))
}

fn is_commit(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|b| b.is_ascii_hexdigit())
}
/// Base URL of the Hub: `HF_ENDPOINT` (the variable the Hugging Face tools use for mirrors) or huggingface.co.
fn endpoint() -> String {
    env::var("HF_ENDPOINT")
        .ok()
        .map(|e| e.trim().trim_end_matches('/').to_owned())
        .filter(|e| e.starts_with("https://") || e.starts_with("http://"))
        .unwrap_or_else(|| "https://huggingface.co".to_owned())
}
fn url(parts: &[&str]) -> Result<reqwest::Url, String> {
    let mut url = reqwest::Url::parse(&endpoint()).map_err(|e| e.to_string())?;
    url.path_segments_mut()
        .map_err(|_| "invalid Hub URL")?
        .extend(parts);
    Ok(url)
}
/// What to tell the user about an HTTP error status of the Hub.
pub(crate) fn status_hint(status: reqwest::StatusCode, has_token: bool, repo: &str) -> String {
    match status.as_u16() {
        401 if !has_token => format!(
            "{repo} is private or gated (HTTP 401): create an access token at https://huggingface.co/settings/tokens, set the environment variable HF_TOKEN to it and run again"
        ),
        401 => format!(
            "Hugging Face rejected the token in HF_TOKEN for {repo} (HTTP 401): check that the token is valid and may read this repository"
        ),
        403 if !has_token => format!(
            "{repo} is gated (HTTP 403): accept its license on https://huggingface.co/{repo} while logged in, create an access token at https://huggingface.co/settings/tokens, set the environment variable HF_TOKEN to it and run again"
        ),
        403 => format!(
            "access to {repo} is denied (HTTP 403): accept the model license on https://huggingface.co/{repo} with the account that owns the token in HF_TOKEN, or use a token that may read this repository"
        ),
        404 => format!(
            "{repo} or the requested revision/file was not found (HTTP 404): check the spelling; private repositories need HF_TOKEN"
        ),
        429 => format!("Hugging Face is rate limiting requests for {repo} (HTTP 429): wait a moment, or set HF_TOKEN for higher limits"),
        code => format!("Hub request for {repo} failed with HTTP {code} (check connectivity and HF_TOKEN)"),
    }
}
fn get(
    client: &Client,
    url: reqwest::Url,
    auth: &Option<String>,
) -> Result<reqwest::blocking::Response, String> {
    let repo = url
        .path_segments()
        .map(|s| s.skip_while(|p| *p != "models").skip(1).take(2).collect::<Vec<_>>().join("/"))
        .filter(|r| !r.is_empty())
        .unwrap_or_else(|| "the repository".to_owned());
    let mut request = client.get(url);
    if let Some(token) = auth {
        request = request.bearer_auth(token);
    }
    let response = request.send().map_err(|e| {
        format!(
            "Hub request failed (check connectivity, repository/revision and HF_TOKEN): {}",
            e.without_url()
        )
    })?;
    if !response.status().is_success() {
        return Err(status_hint(response.status(), auth.is_some(), &repo));
    }
    Ok(response)
}

/// A Hub repository pinned to one commit together with its file list: online (the inventory comes from the Hub API and
/// files are downloaded on demand) or offline (the inventory is the local snapshot, nothing is downloaded).
pub(crate) struct RepoSession {
    pub owner: String,
    pub repo: String,
    pub commit: String,
    pub revision: String,
    pub repository: PathBuf,
    pub snapshot: PathBuf,
    files: Vec<(String, FileInfo)>,
    network: Option<(Client, Option<String>)>,
    _guard: Option<fs::File>,
}

impl RepoSession {
    pub(crate) fn open(reference: &Reference, revision: &str, offline: bool) -> Result<Self, String> {
        safe_relative(revision)?;
        let repository = cache_root().join(format!("models--{}--{}", reference.owner, reference.repo));
        let cached_revision = if is_commit(revision) {
            Some(revision.to_owned())
        } else {
            fs::read_to_string(repository.join("refs").join(revision))
                .ok()
                .map(|s| s.trim().to_owned())
                .filter(|s| is_commit(s))
        };
        if offline {
            let commit = cached_revision.ok_or("model revision is not in the local Hub cache")?;
            let snapshot = repository.join("snapshots").join(&commit);
            let files = inventory(&snapshot)?
                .into_iter()
                .filter(|name| !name.ends_with(".fllm-part") && !name.ends_with(".fllm-lock"))
                .map(|name| {
                    let size = fs::metadata(snapshot.join(&name)).ok().map(|m| m.len());
                    (name, FileInfo { size, sha256: None })
                })
                .collect();
            return Ok(Self {
                owner: reference.owner.clone(),
                repo: reference.repo.clone(),
                commit,
                revision: revision.to_owned(),
                repository,
                snapshot,
                files,
                network: None,
                _guard: None,
            });
        }
        let guard = repository_lock(&repository, true)?;
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(20))
            // a big model on a slow line takes hours: the limit only ends a connection that hangs for good
            .timeout(Duration::from_secs(6 * 3600))
            .user_agent("fllm-rust/0.1")
            .build()
            .map_err(|e| e.to_string())?;
        let auth = token();
        let mut info_url = url(&[
            "api",
            "models",
            &reference.owner,
            &reference.repo,
            "revision",
            revision,
        ])?;
        info_url.query_pairs_mut().append_pair("blobs", "true");
        let metadata: Value = get(&client, info_url, &auth)?
            .json()
            .map_err(|e| e.to_string())?;
        let commit = metadata
            .get("sha")
            .and_then(Value::as_str)
            .filter(|s| is_commit(s))
            .ok_or("Hub response has no valid immutable commit")?
            .to_owned();
        let snapshot = repository.join("snapshots").join(&commit);
        fs::create_dir_all(&snapshot).map_err(|e| e.to_string())?;
        let mut files = Vec::new();
        for sibling in metadata
            .get("siblings")
            .and_then(Value::as_array)
            .ok_or("Hub response has no file inventory")?
        {
            if let Some(name) = sibling.get("rfilename").and_then(Value::as_str) {
                files.push((name.to_owned(), FileInfo::from_hub(sibling)?));
            }
        }
        Ok(Self {
            owner: reference.owner.clone(),
            repo: reference.repo.clone(),
            commit,
            revision: revision.to_owned(),
            repository,
            snapshot,
            files,
            network: Some((client, auth)),
            _guard: Some(guard),
        })
    }

    /// File names of the revision. Offline, files this program wrote itself (converted GGUFs, their metadata) are left out.
    pub(crate) fn names(&self) -> Vec<&str> {
        self.files
            .iter()
            .map(|(n, _)| n.as_str())
            .filter(|n| self.network.is_some() || !(is_converted_name(n) || n.ends_with(CONVERTED_META_SUFFIX)))
            .collect()
    }

    pub(crate) fn size_of(&self, name: &str) -> Option<u64> {
        self.files.iter().find(|(n, _)| n == name).and_then(|(_, i)| i.size)
    }

    pub(crate) fn is_cached(&self, name: &str) -> bool {
        self.snapshot.join(name).is_file()
    }

    /// Make `name` available in the snapshot (download or verify the cached copy).
    pub(crate) fn download(&self, name: &str) -> Result<(), String> {
        safe_relative(name)?;
        let destination = self.snapshot.join(name);
        let Some((client, auth)) = &self.network else {
            return if destination.is_file() {
                Ok(())
            } else {
                Err(format!("{name} is not in the local Hub cache (offline mode)"))
            };
        };
        let parent = destination.parent().ok_or("model filename has no parent")?;
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        // Existing cache directories must not redirect a write outside this snapshot.
        if !parent
            .canonicalize()
            .map_err(|e| e.to_string())?
            .starts_with(self.snapshot.canonicalize().map_err(|e| e.to_string())?)
        {
            return Err("model cache directory escapes its snapshot".into());
        }
        let mut parts = vec![
            self.owner.as_str(),
            self.repo.as_str(),
            "resolve",
            self.commit.as_str(),
        ];
        parts.extend(name.split('/'));
        let info = self
            .files
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, i)| i.clone())
            .ok_or("missing Hub file metadata")?;
        download_file(
            client,
            url(&parts)?,
            auth,
            &destination,
            info,
            name.to_ascii_lowercase().ends_with(".gguf"),
            &format!("{}/{}", self.owner, self.repo),
        )
    }

    /// Remember which commit `revision` meant, so that offline runs find the snapshot.
    pub(crate) fn record_revision(&self) -> Result<(), String> {
        if self.network.is_none() {
            return Ok(());
        }
        let refs = self.repository.join("refs").join(&self.revision);
        if let Some(parent) = refs.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        fs::write(refs, &self.commit).map_err(|e| e.to_string())
    }
}

/// Resolve local paths without networking; HF references use commit-pinned cache directories.
pub fn resolve(reference: &str, revision: Option<&str>, offline: bool) -> Result<PathBuf, String> {
    resolve_impl(reference, revision, offline, false)
}

fn resolve_impl(reference: &str, revision: Option<&str>, offline: bool, prefer_gguf: bool) -> Result<PathBuf, String> {
    let reference = normalize_reference(reference);
    if !reference.starts_with("hf://") {
        return Ok(PathBuf::from(reference));
    }
    resolve_reference(&parse_reference(&reference)?, revision, offline, prefer_gguf)
}

fn resolve_reference(reference: &Reference, revision: Option<&str>, offline: bool, prefer_gguf: bool) -> Result<PathBuf, String> {
    let session = RepoSession::open(reference, revision.unwrap_or("main"), offline)?;
    resolve_with_session(&session, reference, offline, prefer_gguf)
}

pub(crate) fn resolve_with_session(
    session: &RepoSession,
    reference: &Reference,
    offline: bool,
    prefer_gguf: bool,
) -> Result<PathBuf, String> {
    let siblings = session.names();
    let snapshot = session.snapshot.clone();
    let has_gguf = siblings.iter().any(|s| s.to_ascii_lowercase().ends_with(".gguf"));
    let selected = if reference.quant.is_some()
        || reference.file.is_none()
            && (!siblings.contains(&"config.json") || prefer_gguf && has_gguf)
    {
        Some(select_gguf(&siblings, reference.quant.as_deref())?.to_owned())
    } else {
        reference.file.clone()
    };
    let selected = first_shard(selected, &siblings)?;
    if offline {
        if let Some(file) = selected
            .as_ref()
            .filter(|f| f.to_ascii_lowercase().ends_with(".gguf"))
        {
            for part in gguf_shards(file, &siblings)? {
                if !snapshot.join(part).is_file() {
                    return Err("GGUF shard set is not complete in the local cache".into());
                }
            }
        }
        return cached_result(&snapshot, selected.as_deref());
    }
    if let Some(file) = &selected {
        if !siblings.contains(&file.as_str()) {
            return Err(format!("model repository contains no file {file}"));
        }
        if file.to_ascii_lowercase().ends_with(".gguf") {
            for part in gguf_shards(file, &siblings)? {
                session.download(&part)?;
            }
        } else {
            session.download(file)?;
            for name in [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "generation_config.json",
                "chat_template.jinja",
                "chat_template.json",
            ] {
                if siblings.contains(&name) {
                    session.download(name)?;
                }
            }
        }
    } else {
        if !siblings.contains(&"config.json") {
            return Err("checkpoint repository has no config.json; for GGUF select hf://owner/repo/file.gguf".into());
        }
        for name in CHECKPOINT_SUPPORT_FILES {
            if siblings.contains(&name) {
                session.download(name)?;
            }
        }
        for name in &siblings {
            if name.starts_with("additional_chat_templates/") && name.ends_with(".jinja") {
                session.download(name)?;
            }
        }
        for file in checkpoint_weight_files(session)? {
            session.download(&file)?;
        }
    }
    session.record_revision()?;
    cached_result(&snapshot, selected.as_deref())
}

/// Config, tokenizer and template files that accompany a checkpoint.
pub(crate) const CHECKPOINT_SUPPORT_FILES: [&str; 20] = [
    "config.json",
    "hf_quant_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "generation_config.json",
    "chat_template.jinja",
    "chat_template.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "tiktoken.model",
    "qwen.tiktoken",
    "tekken.json",
    "params.json",
    "1_Pooling/config.json",
];

/// Weight files of a checkpoint repository: the shards listed by the index (the index is downloaded to read it), or the
/// single unindexed weights file.
pub(crate) fn checkpoint_weight_files(session: &RepoSession) -> Result<Vec<String>, String> {
    let siblings = session.names();
    if !siblings.contains(&"config.json") && siblings.contains(&"params.json") {
        // Mistral's own layout: params.json and consolidated*.safetensors
        let mut files: Vec<String> = siblings
            .iter()
            .filter(|n| n.starts_with("consolidated") && n.ends_with(".safetensors"))
            .map(|n| (*n).to_owned())
            .collect();
        files.sort();
        if files.is_empty() {
            return Err("repository has params.json but no consolidated*.safetensors weights".into());
        }
        return Ok(files);
    }
    let indexes = CHECKPOINT_INDEXES
        .into_iter()
        .filter(|name| siblings.contains(name))
        .collect::<Vec<_>>();
    if indexes.len() > 1 {
        return Err("repository has conflicting SafeTensors and PyTorch checkpoint indexes".into());
    }
    if let Some(index) = indexes.first() {
        session.download(index)?;
        let mut files = vec![(*index).to_owned()];
        for file in manifest_shards(&session.snapshot, index)? {
            if !siblings.contains(&file.as_str()) {
                return Err(format!(
                    "checkpoint index references missing repository file {file}"
                ));
            }
            files.push(file);
        }
        Ok(files)
    } else {
        Ok(vec![select_unindexed_weight(&siblings)?.to_owned()])
    }
}
const CHECKPOINT_INDEXES: [&str; 2] = [
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
];

fn shard_name(name: &str) -> Option<(&str, usize, usize)> {
    let stem = name.strip_suffix(".gguf")?;
    let (left, total) = stem.rsplit_once("-of-")?;
    let (prefix, number) = left.rsplit_once('-')?;
    if number.len() != 5
        || total.len() != 5
        || !number
            .bytes()
            .chain(total.bytes())
            .all(|b| b.is_ascii_digit())
    {
        return None;
    }
    Some((prefix, number.parse().ok()?, total.parse().ok()?))
}

fn select_gguf<'a>(files: &[&'a str], quant: Option<&str>) -> Result<&'a str, String> {
    let selector = quant.map(str::to_ascii_lowercase);
    let mut candidates = files
        .iter()
        .copied()
        .filter(|name| {
            name.to_ascii_lowercase().ends_with(".gguf")
                && selector
                    .as_ref()
                    .is_none_or(|q| name.to_ascii_lowercase().contains(q))
                && shard_name(name).is_none_or(|(_, number, _)| number == 1)
        })
        .collect::<Vec<_>>();
    candidates.sort();
    candidates.dedup();
    match candidates.as_slice() {
        [file] => Ok(file),
        [] => Err(format!("no GGUF matches {}; available GGUF files:\n{}", quant.unwrap_or("the requested model"), files.iter().filter(|s| s.to_ascii_lowercase().ends_with(".gguf")).copied().collect::<Vec<_>>().join("\n"))),
        _ => Err(format!("GGUF selection is ambiguous; choose a filename or a more specific :QUANT. Candidates:\n{}", candidates.join("\n"))),
    }
}

fn gguf_shards(file: &str, inventory: &[&str]) -> Result<Vec<String>, String> {
    let Some((prefix, number, total)) = shard_name(file) else {
        return Ok(vec![file.into()]);
    };
    if number == 0 || number > total || total == 0 || total > 65535 {
        return Err(format!("invalid GGUF shard filename {file}"));
    }
    let mut files = Vec::new();
    for n in 1..=total {
        let part = format!("{prefix}-{n:05}-of-{total:05}.gguf");
        if !inventory.contains(&part.as_str()) {
            return Err(format!("GGUF shard set is incomplete: missing {part}"));
        }
        files.push(part);
    }
    Ok(files)
}

/// Marker of files this program converted from a checkpoint: `<name>-<label>-fllm<12 hex digits>.gguf`.
pub(crate) fn is_converted_name(name: &str) -> bool {
    let Some(stem) = name.strip_suffix(".gguf") else {
        return false;
    };
    stem.len() > 17
        && stem
            .get(stem.len() - 17..)
            .is_some_and(|tail| tail.starts_with("-fllm") && tail[5..].bytes().all(|b| b.is_ascii_hexdigit()))
}
/// Metadata written beside a converted GGUF.
pub(crate) const CONVERTED_META_SUFFIX: &str = ".gguf.json";

#[derive(Clone, Default)]
struct FileInfo {
    size: Option<u64>,
    sha256: Option<String>,
}

fn first_shard(selected: Option<String>, names: &[&str]) -> Result<Option<String>, String> {
    selected
        .map(|file| {
            if file.to_ascii_lowercase().ends_with(".gguf") {
                gguf_shards(&file, names)?
                    .into_iter()
                    .next()
                    .ok_or_else(|| "empty GGUF shard set".into())
            } else {
                Ok(file)
            }
        })
        .transpose()
}

fn repository_lock(repository: &Path, shared: bool) -> Result<fs::File, String> {
    let parent = repository
        .parent()
        .ok_or("cache repository has no parent")?;
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let path = sidecar(repository, ".fllm-lock");
    regular_or_absent(&path)?;
    let lock = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(|e| e.to_string())?;
    if shared {
        lock.try_lock_shared()
    } else {
        lock.try_lock()
    }
    .map_err(|e| format!("cache repository is in use by a download or removal: {e}"))?;
    Ok(lock)
}
impl FileInfo {
    fn from_hub(value: &Value) -> Result<Self, String> {
        let sha256 = value["lfs"]["sha256"].as_str().map(str::to_owned);
        if sha256.as_ref().is_some_and(|s| !is_sha256(s)) {
            return Err("Hub returned an invalid SHA256 digest".into());
        }
        Ok(Self {
            size: value["lfs"]["size"]
                .as_u64()
                .or_else(|| value["size"].as_u64()),
            sha256,
        })
    }
}
fn is_sha256(hash: &str) -> bool {
    hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit())
}
fn checked_file(path: &Path, info: &FileInfo, gguf: bool) -> Result<(), String> {
    let mut file = fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let len = file.metadata().map_err(|e| e.to_string())?.len();
    if info.size.is_some_and(|n| n != len) {
        return Err(format!(
            "file size mismatch for {}: expected {:?} bytes, found {len}",
            path.display(),
            info.size
        ));
    }
    if gguf {
        let mut magic = [0; 4];
        file.read_exact(&mut magic)
            .map_err(|e| format!("truncated GGUF download: {e}"))?;
        if &magic != b"GGUF" {
            return Err("download is not a GGUF file".into());
        }
        file.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    }
    if let Some(expected) = &info.sha256 {
        let mut hash = Sha256::new();
        let mut buffer = vec![0; 1024 * 1024];
        loop {
            let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            hash.update(&buffer[..n]);
        }
        let actual = format!("{:x}", hash.finalize());
        if !actual.eq_ignore_ascii_case(expected) {
            return Err(format!(
                "SHA256 mismatch for {}: expected {expected}, found {actual}",
                path.display()
            ));
        }
    }
    Ok(())
}
fn sidecar(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}
fn regular_or_absent(path: &Path) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Ok(meta) if !meta.file_type().is_file() => Err(format!(
            "cache download path is not a regular file: {}",
            path.display()
        )),
        Ok(_) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.to_string()),
    }
}

fn download_file(
    client: &Client,
    url: reqwest::Url,
    auth: &Option<String>,
    destination: &Path,
    mut info: FileInfo,
    gguf: bool,
    repo: &str,
) -> Result<(), String> {
    if destination.is_file() {
        return checked_file(destination, &info, gguf);
    }
    let partial = sidecar(destination, ".fllm-part");
    let lock_path = sidecar(destination, ".fllm-lock");
    regular_or_absent(&partial)?;
    regular_or_absent(&lock_path)?;
    let lock = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .map_err(|e| e.to_string())?;
    lock.try_lock().map_err(|e| {
        format!(
            "another download or cache removal holds {}: {e}",
            destination.display()
        )
    })?;
    if destination.is_file() {
        return checked_file(destination, &info, gguf);
    }
    let mut output = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&partial)
        .map_err(|e| e.to_string())?;
    let mut offset = output.metadata().map_err(|e| e.to_string())?.len();
    if info.size.is_some_and(|n| offset > n) {
        output.set_len(0).map_err(|e| e.to_string())?;
        offset = 0;
    }
    if info.size != Some(offset) {
        let mut request = client
            .get(url)
            .header(reqwest::header::ACCEPT_ENCODING, "identity");
        if let Some(token) = auth {
            request = request.bearer_auth(token);
        }
        if offset > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={offset}-"));
        }
        let mut response = request.send().map_err(|e| {
            format!(
                "download connection failed; partial file retained for resume: {}",
                e.without_url()
            )
        })?;
        if response.status() == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
            let total = response
                .headers()
                .get(reqwest::header::CONTENT_RANGE)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.strip_prefix("bytes */"))
                .and_then(|s| s.parse::<u64>().ok());
            if total != Some(offset) || info.size.is_some_and(|n| n != offset) {
                return Err(
                    "Hub rejected resume range; remove the cached partial download and retry"
                        .into(),
                );
            }
            info.size = total;
        } else {
            if !response.status().is_success() {
                return Err(format!(
                    "{}; partial file retained",
                    status_hint(response.status(), auth.is_some(), repo)
                ));
            }
            let mut expected_body = response.content_length();
            if response.status() == reqwest::StatusCode::PARTIAL_CONTENT {
                let range = response
                    .headers()
                    .get(reqwest::header::CONTENT_RANGE)
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.strip_prefix("bytes "))
                    .ok_or("resume response has no valid Content-Range")?;
                let (range, total) = range
                    .split_once('/')
                    .ok_or("invalid resume Content-Range")?;
                let (start, end) = range.split_once('-').ok_or("invalid resume byte range")?;
                let start = start.parse::<u64>().map_err(|_| "invalid resume start")?;
                let end = end.parse::<u64>().map_err(|_| "invalid resume end")?;
                let total = total
                    .parse::<u64>()
                    .map_err(|_| "resume response has no total size")?;
                if start != offset
                    || end < start
                    || end >= total
                    || info.size.is_some_and(|n| n != total)
                {
                    return Err(
                        "resume response does not match the requested offset or Hub file size"
                            .into(),
                    );
                }
                let length = end
                    .checked_sub(start)
                    .and_then(|n| n.checked_add(1))
                    .ok_or("resume size overflow")?;
                if expected_body.is_some_and(|n| n != length) {
                    return Err("resume Content-Length does not match Content-Range".into());
                }
                expected_body = Some(length);
                info.size = Some(total);
            } else if response.status() == reqwest::StatusCode::OK {
                // A server ignoring Range returns the whole file: restart, never append it.
                output.set_len(0).map_err(|e| e.to_string())?;
                offset = 0;
                if let (Some(expected), Some(body)) = (info.size, expected_body) {
                    if expected != body {
                        return Err("download Content-Length does not match Hub file size".into());
                    }
                }
                info.size = info.size.or(expected_body);
            } else {
                return Err(format!(
                    "unexpected download HTTP status {}",
                    response.status()
                ));
            }
            if info.sha256.is_none() {
                info.sha256 = ["x-linked-etag", "etag"]
                    .iter()
                    .filter_map(|key| response.headers().get(*key).and_then(|v| v.to_str().ok()))
                    .map(|s| s.trim_matches('"'))
                    .find(|s| is_sha256(s))
                    .map(str::to_owned);
            }
            output
                .seek(SeekFrom::Start(offset))
                .map_err(|e| e.to_string())?;
            let started = Instant::now();
            let mut last = Instant::now();
            let mut received = 0u64;
            let mut buffer = vec![0; 1024 * 1024];
            loop {
                let n = response
                    .read(&mut buffer)
                    .map_err(|e| format!("download interrupted; resume by repeating pull: {e}"))?;
                if n == 0 {
                    break;
                }
                output
                    .write_all(&buffer[..n])
                    .map_err(|e| format!("cannot write download: {e}"))?;
                match DOWNLOAD_STOP.load(std::sync::atomic::Ordering::Relaxed) {
                    STOP_PAUSE => return Err(DOWNLOAD_PAUSED.into()),
                    STOP_CANCEL => return Err(DOWNLOAD_CANCELLED.into()),
                    _ => {}
                }
                received = received
                    .checked_add(n as u64)
                    .ok_or("download byte count overflow")?;
                let written = offset
                    .checked_add(received)
                    .ok_or("download size overflow")?;
                if info.size.is_some_and(|n| written > n) {
                    return Err("download exceeds expected file size".into());
                }
                if last.elapsed() >= Duration::from_millis(500) {
                    progress(
                        destination,
                        written,
                        info.size,
                        received,
                        started.elapsed().as_secs_f64(),
                        false,
                    );
                    last = Instant::now();
                }
            }
            output.sync_all().map_err(|e| e.to_string())?;
            progress(
                destination,
                offset + received,
                info.size,
                received,
                started.elapsed().as_secs_f64(),
                true,
            );
            if expected_body.is_some_and(|n| received != n) {
                return Err("truncated HTTP response; partial file retained for resume".into());
            }
        }
    }
    output.sync_all().map_err(|e| e.to_string())?;
    drop(output);
    if let Err(error) = checked_file(&partial, &info, gguf) {
        if info
            .size
            .is_some_and(|n| fs::metadata(&partial).is_ok_and(|m| m.len() == n))
        {
            let _ = fs::OpenOptions::new()
                .write(true)
                .truncate(true)
                .open(&partial);
        }
        return Err(error);
    }
    match fs::hard_link(&partial, destination) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
            checked_file(destination, &info, gguf)?
        }
        Err(e) => return Err(format!("cannot publish download: {e}")),
    }
    fs::remove_file(partial).map_err(|e| e.to_string())?;
    Ok(())
}

/// Asks the download that is running to stop (checked after every block it writes; the partial file stays for resuming).
pub const STOP_NONE: u8 = 0;
pub const STOP_PAUSE: u8 = 1;
pub const STOP_CANCEL: u8 = 2;
pub const DOWNLOAD_PAUSED: &str = "download paused; the finished part is kept";
pub const DOWNLOAD_CANCELLED: &str = "download cancelled; the finished part is kept";
static DOWNLOAD_STOP: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(STOP_NONE);

/// Sets what the running download should do next: `STOP_NONE` to go on, `STOP_PAUSE` or `STOP_CANCEL` to end it.
pub fn set_download_stop(kind: u8) {
    DOWNLOAD_STOP.store(kind, std::sync::atomic::Ordering::Relaxed);
}

/// What the running download has been asked to do (`STOP_NONE`, `STOP_PAUSE` or `STOP_CANCEL`).
pub fn download_stop() -> u8 {
    DOWNLOAD_STOP.load(std::sync::atomic::Ordering::Relaxed)
}

/// The file a download is writing right now, for a UI that polls (the Studio shows it as a progress bar).
#[derive(Clone, Debug)]
pub struct DownloadProgress {
    pub file: String,
    pub written: u64,
    pub total: Option<u64>,
    pub mb_per_s: f64,
    pub finished: bool,
}

static DOWNLOAD: std::sync::Mutex<Option<DownloadProgress>> = std::sync::Mutex::new(None);

/// The latest progress report of a running or just finished download.
pub fn download_progress() -> Option<DownloadProgress> {
    DOWNLOAD.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

pub fn clear_download_progress() {
    *DOWNLOAD.lock().unwrap_or_else(|e| e.into_inner()) = None;
}

fn progress(
    path: &Path,
    written: u64,
    total: Option<u64>,
    received: u64,
    seconds: f64,
    finished: bool,
) {
    *DOWNLOAD.lock().unwrap_or_else(|e| e.into_inner()) = Some(DownloadProgress {
        file: path.file_name().unwrap_or_default().to_string_lossy().into_owned(),
        written,
        total,
        mb_per_s: received as f64 / 1_000_000.0 / seconds.max(0.001),
        finished,
    });
    let percent = total.filter(|&n| n > 0).map_or_else(
        || "size unknown".into(),
        |n| format!("{:.1}%", written as f64 * 100.0 / n as f64),
    );
    let speed = received as f64 / 1_000_000.0 / seconds.max(0.001);
    eprint!(
        "\r{}: {percent}, {:.1} MB, {speed:.1} MB/s",
        path.file_name().unwrap_or_default().to_string_lossy(),
        written as f64 / 1_000_000.0
    );
    if finished {
        eprintln!();
    }
}

fn inventory(root: &Path) -> Result<Vec<String>, String> {
    fn visit(root: &Path, dir: &Path, files: &mut Vec<String>) -> Result<(), String> {
        for entry in fs::read_dir(dir).map_err(|e| format!("read model cache: {e}"))? {
            let entry = entry.map_err(|e| e.to_string())?;
            let ty = entry.file_type().map_err(|e| e.to_string())?;
            let path = entry.path();
            if ty.is_dir() {
                visit(root, &path, files)?;
            } else if path.is_file() {
                files.push(
                    path.strip_prefix(root)
                        .map_err(|e| e.to_string())?
                        .to_string_lossy()
                        .replace('\\', "/"),
                );
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    visit(root, root, &mut files)?;
    files.sort();
    Ok(files)
}

pub struct CacheEntry {
    pub reference: String,
    pub revision: String,
    pub bytes: u64,
    pub path: PathBuf,
    /// Extra information, e.g. that a GGUF was converted from a checkpoint by this program.
    pub note: String,
}
/// Enumerate complete cached GGUF variants and checkpoint snapshots without networking.
pub fn list_cache() -> Result<Vec<CacheEntry>, String> {
    let root = cache_root();
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut entries = Vec::new();
    for entry in fs::read_dir(&root).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        if !entry.file_type().map_err(|e| e.to_string())?.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(rest) = name.strip_prefix("models--") else {
            continue;
        };
        let Some((owner, repo)) = rest.split_once("--") else {
            continue;
        };
        let snapshots = entry.path().join("snapshots");
        if !snapshots.is_dir() {
            continue;
        }
        for snapshot in fs::read_dir(snapshots).map_err(|e| e.to_string())? {
            let snapshot = snapshot.map_err(|e| e.to_string())?;
            if !snapshot.file_type().map_err(|e| e.to_string())?.is_dir() {
                continue;
            }
            let path = snapshot.path();
            let files = inventory(&path)?;
            let names = files.iter().map(String::as_str).collect::<Vec<_>>();
            let revision = snapshot.file_name().to_string_lossy().into_owned();
            for name in names.iter().copied().filter(|s| {
                s.to_ascii_lowercase().ends_with(".gguf")
                    && shard_name(s).is_none_or(|(_, n, _)| n == 1)
            }) {
                let Ok(shards) = gguf_shards(name, &names) else {
                    continue;
                };
                let bytes = shards.iter().try_fold(0u64, |n, name| {
                    n.checked_add(
                        fs::metadata(path.join(name))
                            .map_err(|e| e.to_string())?
                            .len(),
                    )
                    .ok_or_else(|| "cache size overflow".to_owned())
                })?;
                let note = if is_converted_name(name) {
                    crate::import::converted_note(&path.join(name)).unwrap_or_else(|| "converted from a checkpoint".into())
                } else {
                    String::new()
                };
                entries.push(CacheEntry {
                    reference: format!("hf://{owner}/{repo}/{name}"),
                    revision: revision.clone(),
                    bytes,
                    path: path.join(name),
                    note,
                });
            }
            if path.join("config.json").is_file() && cached_result(&path, None).is_ok() {
                let bytes = files
                    .iter()
                    .filter(|s| {
                        !s.contains(".fllm-part")
                            && !s.ends_with(".fllm-lock")
                            && !s.ends_with(CONVERTED_META_SUFFIX)
                            && !s.to_ascii_lowercase().ends_with(".gguf")
                    })
                    .try_fold(0u64, |n, name| {
                        n.checked_add(
                            fs::metadata(path.join(name))
                                .map_err(|e| e.to_string())?
                                .len(),
                        )
                        .ok_or_else(|| "cache size overflow".to_owned())
                    })?;
                entries.push(CacheEntry {
                    reference: format!("hf://{owner}/{repo}"),
                    revision,
                    bytes,
                    path,
                    note: String::new(),
                });
            }
        }
    }
    entries.extend(crate::import::list_local_conversions());
    entries.sort_by(|a, b| (&a.reference, &a.revision).cmp(&(&b.reference, &b.revision)));
    Ok(entries)
}

/// Remove only the selected cache repository or matching GGUF shard sets.
pub fn remove_cached(reference: &str) -> Result<usize, String> {
    let reference = parse_reference(reference)?;
    let root = cache_root()
        .canonicalize()
        .map_err(|e| format!("model cache unavailable: {e}"))?;
    let repository = root.join(format!("models--{}--{}", reference.owner, reference.repo));
    if fs::symlink_metadata(&repository)
        .map_err(|e| format!("cached model not found: {e}"))?
        .file_type()
        .is_symlink()
        || repository
            .canonicalize()
            .map_err(|e| e.to_string())?
            .parent()
            != Some(root.as_path())
    {
        return Err("cache repository escapes the configured cache root".into());
    }
    let _repository_guard = repository_lock(&repository, false)?;
    if reference.quant.is_none() && reference.file.is_none() {
        // Refuse removal while any fLLM downloader owns a per-file lock.
        let files = inventory(&repository)?;
        let mut locks = Vec::new();
        for name in files.iter().filter(|s| s.ends_with(".fllm-lock")) {
            let lock = fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(repository.join(name))
                .map_err(|e| e.to_string())?;
            lock.try_lock()
                .map_err(|e| format!("model download is active: {e}"))?;
            locks.push(lock);
        }
        let count = files.iter().filter(|s| !s.ends_with(".fllm-lock")).count();
        // The repository lock stays held outside the tree while Windows file
        // handles inside it are closed for deletion.
        drop(locks);
        fs::remove_dir_all(&repository).map_err(|e| format!("cannot remove model cache: {e}"))?;
        return Ok(count);
    }
    let mut targets = Vec::new();
    for snapshot in fs::read_dir(repository.join("snapshots")).map_err(|e| e.to_string())? {
        let snapshot = snapshot.map_err(|e| e.to_string())?;
        if !snapshot.file_type().map_err(|e| e.to_string())?.is_dir() {
            continue;
        }
        let path = snapshot.path();
        let files = inventory(&path)?;
        let mut names = files
            .iter()
            .map(|s| s.strip_suffix(".fllm-part").unwrap_or(s))
            .filter(|s| !s.ends_with(".fllm-lock"))
            .map(str::to_owned)
            .collect::<Vec<_>>();
        names.sort();
        names.dedup();
        let mut groups = names
            .iter()
            .map(|s| {
                shard_name(s).map_or_else(
                    || s.clone(),
                    |(prefix, _, total)| format!("{prefix}-00001-of-{total:05}.gguf"),
                )
            })
            .collect::<Vec<_>>();
        groups.sort();
        groups.dedup();
        let selected = if let Some(file) = &reference.file {
            let file = shard_name(file).map_or_else(
                || file.clone(),
                |(prefix, _, total)| format!("{prefix}-00001-of-{total:05}.gguf"),
            );
            if !groups.contains(&file) {
                continue;
            }
            file
        } else {
            let q = reference.quant.as_deref().ok_or("missing cache selector")?;
            if !groups.iter().any(|s| {
                s.to_ascii_lowercase().ends_with(".gguf")
                    && s.to_ascii_lowercase().contains(&q.to_ascii_lowercase())
            }) {
                continue;
            }
            select_gguf(
                &groups.iter().map(String::as_str).collect::<Vec<_>>(),
                Some(q),
            )?
            .to_owned()
        };
        for file in names {
            let same_group = match (shard_name(&selected), shard_name(&file)) {
                (Some((a, _, at)), Some((b, _, bt))) => a == b && at == bt,
                _ => selected == file,
            };
            if same_group {
                targets.push(path.join(file));
            }
        }
    }
    if targets.is_empty() {
        return Err("no cached model matches the requested reference".into());
    }
    let mut locks = Vec::new();
    for path in &targets {
        let lock_path = sidecar(path, ".fllm-lock");
        regular_or_absent(&lock_path)?;
        let lock = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(lock_path)
            .map_err(|e| e.to_string())?;
        lock.try_lock()
            .map_err(|e| format!("model download is active: {e}"))?;
        locks.push(lock);
    }
    for path in &targets {
        if fs::symlink_metadata(path).is_ok() {
            fs::remove_file(path).map_err(|e| format!("cannot remove {}: {e}", path.display()))?;
        }
        let partial = sidecar(path, ".fllm-part");
        if partial.exists() {
            fs::remove_file(partial).map_err(|e| e.to_string())?;
        }
        let meta = sidecar(path, ".json");
        if is_converted_name(&path.file_name().unwrap_or_default().to_string_lossy()) && meta.exists() {
            fs::remove_file(meta).map_err(|e| e.to_string())?;
        }
        crate::import::remove_partial_files(path);
    }
    Ok(targets.len())
}

/// Remove only the GGUF files this program converted from the checkpoint of `reference`, keeping the downloaded
/// checkpoint. Returns the number of files removed.
pub fn remove_converted(reference: &str) -> Result<usize, String> {
    let reference = parse_reference(reference)?;
    let root = cache_root()
        .canonicalize()
        .map_err(|e| format!("model cache unavailable: {e}"))?;
    let repository = root.join(format!("models--{}--{}", reference.owner, reference.repo));
    let _guard = repository_lock(&repository, false)?;
    let mut removed = 0;
    for snapshot in fs::read_dir(repository.join("snapshots")).map_err(|e| format!("cached model not found: {e}"))? {
        let path = snapshot.map_err(|e| e.to_string())?.path();
        for name in inventory(&path)? {
            if !is_converted_name(&name) {
                continue;
            }
            let target = path.join(&name);
            let lock = fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(sidecar(&target, ".fllm-lock"))
                .map_err(|e| e.to_string())?;
            lock.try_lock().map_err(|e| format!("a conversion is running: {e}"))?;
            for file in [target.clone(), sidecar(&target, ".json")] {
                if file.exists() {
                    fs::remove_file(&file).map_err(|e| format!("cannot remove {}: {e}", file.display()))?;
                    removed += 1;
                }
            }
            crate::import::remove_partial_files(&target);
            drop(lock);
            let _ = fs::remove_file(sidecar(&target, ".fllm-lock"));
        }
    }
    if removed == 0 {
        return Err("no converted model in the cache for this repository".into());
    }
    Ok(removed)
}

/// Shard file names listed by a checkpoint index, validated, deduplicated and sorted.
fn manifest_shards(snapshot: &Path, index: &str) -> Result<Vec<String>, String> {
    let text = fs::read_to_string(snapshot.join(index))
        .map_err(|e| format!("cannot read {index}: {e}"))?;
    let manifest: Value =
        serde_json::from_str(&text).map_err(|e| format!("invalid {index}: {e}"))?;
    let map = manifest
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{index} has no weight_map"))?;
    let mut shards = Vec::new();
    for value in map.values() {
        let name = value
            .as_str()
            .ok_or_else(|| format!("{index} has a non-string shard name"))?;
        safe_relative(name)?;
        if !shards.iter().any(|known: &String| known == name) {
            shards.push(name.to_owned());
        }
    }
    if shards.is_empty() {
        return Err(format!("{index} lists no shards"));
    }
    shards.sort();
    Ok(shards)
}

/// Single weights file of a repository without an index; adapters and unindexed shards are not
/// guessed.
fn select_unindexed_weight<'a>(siblings: &[&'a str]) -> Result<&'a str, String> {
    for name in ["model.safetensors", "pytorch_model.bin"] {
        if let Some(found) = siblings.iter().copied().find(|s| *s == name) {
            return Ok(found);
        }
    }
    Err(
        "checkpoint repository has no model.safetensors, pytorch_model.bin or weight index; \
         for GGUF select hf://owner/repo/file.gguf"
            .into(),
    )
}

fn cached_result(snapshot: &Path, file: Option<&str>) -> Result<PathBuf, String> {
    const UNAVAILABLE: &str = "requested model is not fully available in the local Hub cache";
    let path = file.map_or_else(|| snapshot.to_owned(), |file| snapshot.join(file));
    if file.is_some() {
        return if path.is_file() {
            Ok(path)
        } else {
            Err(UNAVAILABLE.into())
        };
    }
    if !path.join("config.json").is_file() {
        return Err(UNAVAILABLE.into());
    }
    let indexes = CHECKPOINT_INDEXES
        .into_iter()
        .filter(|name| path.join(name).is_file())
        .collect::<Vec<_>>();
    match indexes.as_slice() {
        [] => {
            if path.join("model.safetensors").is_file() || path.join("pytorch_model.bin").is_file()
            {
                Ok(path)
            } else {
                Err(UNAVAILABLE.into())
            }
        }
        [index] => {
            for shard in manifest_shards(&path, index)? {
                if !path.join(&shard).is_file() {
                    return Err(format!("missing shard {shard}"));
                }
            }
            Ok(path)
        }
        _ => Err("cache holds conflicting SafeTensors and PyTorch checkpoint indexes".into()),
    }
}

/// Directory used by running servers to publish their status for .
pub fn status_dir() -> std::path::PathBuf {
    let base = cache_root();
    let parent = base.parent().unwrap_or(&base);
    parent.join("fllm").join("servers")
}


/// Return (file name, byte size) for every entry in a Hub repository revision.
pub fn repository_files(repository: &str) -> Result<Vec<(String, u64)>, String> {
    let parts: Vec<&str> = repository.split('/').collect();
    if parts.len() != 2 {
        return Err("repository must be owner/repo".into());
    }
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(20))
        .timeout(std::time::Duration::from_secs(120))
        .user_agent("fllm-rust/0.1")
        .build()
        .map_err(|e| e.to_string())?;
    let auth = token();
    let mut url = reqwest::Url::parse("https://huggingface.co").map_err(|e| e.to_string())?;
    {
        let mut seg = url.path_segments_mut().map_err(|_| "invalid Hub URL")?;
        seg.extend(["api", "models", parts[0], parts[1], "revision", "main"]);
    }
    url.query_pairs_mut().append_pair("blobs", "true");
    let metadata: serde_json::Value = get(&client, url, &auth)?.json().map_err(|e| e.to_string())?;
    let siblings = metadata
        .get("siblings")
        .and_then(serde_json::Value::as_array)
        .ok_or("Hub response has no file inventory")?;
    let mut files = Vec::new();
    for s in siblings {
        let name = match s.get("rfilename").and_then(serde_json::Value::as_str) {
            Some(n) => n,
            None => continue,
        };
        let size = s.get("size").and_then(serde_json::Value::as_u64).unwrap_or(0);
        files.push((name.to_owned(), size));
    }
    Ok(files)
}

/// The model card (README.md) of a repository, as text.
pub fn repository_readme(repository: &str) -> Result<String, String> {
    let parts: Vec<&str> = repository.split('/').collect();
    if parts.len() != 2 || parts.iter().any(|p| p.is_empty() || p.contains(['?', '#', '\\'])) {
        return Err("repository must be owner/repo".into());
    }
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(20))
        .timeout(std::time::Duration::from_secs(30))
        .user_agent("fllm-rust/0.1")
        .build()
        .map_err(|e| e.to_string())?;
    let mut url = reqwest::Url::parse("https://huggingface.co").map_err(|e| e.to_string())?;
    {
        let mut seg = url.path_segments_mut().map_err(|_| "invalid Hub URL")?;
        seg.extend([parts[0], parts[1], "raw", "main", "README.md"]);
    }
    let response = client.get(url).header(reqwest::header::RANGE, "bytes=0-65535");
    let response = match token() {
        Some(t) => response.bearer_auth(t),
        None => response,
    }
    .send()
    .map_err(|e| format!("cannot read the model card: {}", e.without_url()))?;
    if !response.status().is_success() {
        return Err(format!("this repository has no model card (HTTP {})", response.status().as_u16()));
    }
    let bytes = response.bytes().map_err(|e| e.to_string())?;
    Ok(String::from_utf8_lossy(&bytes[..bytes.len().min(65536)]).into_owned())
}

/// The start of a model card as plain text: no front matter, no images, no HTML tags, no headings marks; at most `max` characters.
pub fn readme_summary(text: &str, max: usize) -> String {
    let mut body = text.trim_start_matches('\u{feff}').trim_start();
    if let Some(rest) = body.strip_prefix("---") {
        if let Some(end) = rest.find("\n---") {
            body = rest[end + 4..].trim_start();
        }
    }
    let mut kept = Vec::new();
    let mut in_code = false;
    for line in body.lines() {
        let line = line.trim();
        if line.starts_with("```") {
            in_code = !in_code;
            continue;
        }
        if in_code || line.is_empty() || line.starts_with("![") || line.starts_with('|') || line.starts_with("<!--") {
            if line.is_empty() && !kept.is_empty() && kept.last() != Some(&String::new()) {
                kept.push(String::new());
            }
            continue;
        }
        // drop HTML tags such as <p align="center"> and <img ...>
        let mut clean = String::new();
        let mut in_tag = false;
        for c in line.trim_start_matches('#').trim().chars() {
            match c {
                '<' => in_tag = true,
                '>' if in_tag => in_tag = false,
                _ if !in_tag => clean.push(c),
                _ => {}
            }
        }
        let clean = clean.trim().to_owned();
        if !clean.is_empty() {
            kept.push(clean);
        }
    }
    let joined = kept.join("\n");
    let joined = joined.trim();
    if joined.chars().count() <= max {
        return joined.to_owned();
    }
    let cut: String = joined.chars().take(max).collect();
    format!("{}...", cut.trim_end())
}

/// The quantization named in a GGUF file name (`model-Q4_K_M.gguf` gives `Q4_K_M`), upper case; `None` when it names none.
pub fn quant_of_filename(name: &str) -> Option<String> {
    let stem = name.rsplit('/').next().unwrap_or(name);
    let stem = stem.strip_suffix(".gguf").or_else(|| stem.strip_suffix(".GGUF")).unwrap_or(stem);
    stem.split(['-', '.']).map(str::to_ascii_uppercase).find(|token| {
        let t = token.strip_prefix('I').unwrap_or(token);
        let quant_like = t.starts_with('Q') && t.chars().nth(1).is_some_and(|c| c.is_ascii_digit());
        let plain = matches!(token.as_str(), "F16" | "F32" | "BF16" | "MXFP4" | "FP16");
        (quant_like && token.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')) || plain
    })
}

/// The folder of the Studio's small files (`studio.json`, the lists of added models and per-model options). It does not move with the
/// models folder setting.
fn studio_dir() -> PathBuf {
    let base = default_cache_root();
    base.parent().unwrap_or(&base).join("fllm")
}

fn studio_file() -> PathBuf {
    studio_dir().join("studio.json")
}

/// A small data file of the Studio, next to `studio.json`.
pub fn studio_data_file(name: &str) -> PathBuf {
    studio_dir().join(name)
}

fn read_studio() -> serde_json::Map<String, Value> {
    fs::read_to_string(studio_file())
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_default()
}

fn write_studio(settings: &serde_json::Map<String, Value>) {
    let file = studio_file();
    if let Some(dir) = file.parent() {
        let _ = fs::create_dir_all(dir);
    }
    let _ = fs::write(file, Value::Object(settings.clone()).to_string());
}

/// One value of `studio.json`.
pub fn studio_setting(key: &str) -> Option<Value> {
    read_studio().get(key).cloned()
}

/// Sets (or, with `Null`, removes) one value of `studio.json` and keeps the others.
pub fn set_studio_setting(key: &str, value: Value) {
    let mut settings = read_studio();
    if value.is_null() {
        settings.remove(key);
    } else {
        settings.insert(key.to_owned(), value);
    }
    write_studio(&settings);
}

/// Minutes of inactivity after which the Studio unloads the model (0: never).
pub fn idle_unload_minutes() -> u64 {
    studio_setting("idle_unload_minutes").and_then(|v| v.as_u64()).unwrap_or(0)
}

/// The MCP server list (`mcp.json`, the common `{"mcpServers": {...}}` format) next to `studio.json`.
pub fn mcp_file() -> PathBuf {
    studio_file().with_file_name("mcp.json")
}

/// The model the Studio was last started or switched to (`fllm-rs studio` starts with it when no `--model` is given).
pub fn last_studio_model() -> Option<String> {
    studio_setting("last_model")?.as_str().map(str::to_owned)
}

/// GGUF files the user added from anywhere on the computer (they stay where they are). The list is a small file next to `studio.json`.
fn local_models_file() -> PathBuf {
    studio_data_file("local-models.json")
}

fn read_local_models() -> Vec<String> {
    fs::read_to_string(local_models_file())
        .ok()
        .and_then(|text| serde_json::from_str::<Vec<String>>(&text).ok())
        .unwrap_or_default()
}

fn write_local_models(list: &[String]) {
    let file = local_models_file();
    if let Some(dir) = file.parent() {
        let _ = fs::create_dir_all(dir);
    }
    let _ = fs::write(file, serde_json::json!(list).to_string());
}

fn is_gguf_file(path: &Path) -> bool {
    use std::io::Read;
    let mut magic = [0u8; 4];
    path.is_file() && fs::File::open(path).and_then(|mut f| f.read_exact(&mut magic)).is_ok() && &magic == b"GGUF"
}

/// The added files that still exist: `(path, size in bytes)`.
pub fn local_models() -> Vec<(String, u64)> {
    read_local_models()
        .into_iter()
        .filter_map(|p| fs::metadata(&p).ok().filter(|m| m.is_file()).map(|m| (p, m.len())))
        .collect()
}

/// Adds a GGUF file, or every GGUF file in a folder (one level of sub-folders is searched too). Returns the files that were added.
pub fn add_local_model(input: &str) -> Result<Vec<String>, String> {
    let cleaned = input.trim().trim_matches(|c| c == '"' || c == '\'').trim();
    if cleaned.is_empty() {
        return Err("type or paste the path of a .gguf file".into());
    }
    let path = PathBuf::from(cleaned);
    let mut found = Vec::new();
    if path.is_dir() {
        let mut stack = vec![(path.clone(), 0)];
        while let Some((dir, depth)) = stack.pop() {
            for entry in fs::read_dir(&dir).map_err(|e| e.to_string())?.flatten() {
                let p = entry.path();
                if p.is_dir() && depth < 1 {
                    stack.push((p, depth + 1));
                } else if p.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf")) && is_gguf_file(&p) {
                    found.push(p);
                }
            }
        }
        if found.is_empty() {
            return Err("this folder has no .gguf files".into());
        }
    } else if path.is_file() {
        if !is_gguf_file(&path) {
            return Err("this is not a GGUF file (it does not start with the GGUF header)".into());
        }
        found.push(path);
    } else {
        return Err("no such file or folder".into());
    }
    let mut list = read_local_models();
    let mut added = Vec::new();
    for p in found {
        let text = fs::canonicalize(&p).unwrap_or(p).display().to_string();
        let text = text.strip_prefix(r"\\?\").unwrap_or(&text).to_owned();
        if !list.contains(&text) {
            list.push(text.clone());
        }
        added.push(text);
    }
    write_local_models(&list);
    Ok(added)
}

/// Takes a file off the list of added models (the file itself is not touched).
pub fn remove_local_model(path: &str) {
    let list: Vec<String> = read_local_models().into_iter().filter(|p| p != path).collect();
    write_local_models(&list);
}

pub fn forget_studio_model() {
    set_studio_setting("last_model", Value::Null);
}

pub fn remember_studio_model(model: &str) {
    set_studio_setting("last_model", Value::String(model.to_owned()));
}

/// Search the Hub for repositories that contain GGUF files, most downloaded first.
pub fn search_models(query: &str, limit: usize) -> Result<Vec<Value>, String> {
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(20))
        .timeout(std::time::Duration::from_secs(60))
        .user_agent("fllm-rust/0.1")
        .build()
        .map_err(|e| e.to_string())?;
    let mut url = reqwest::Url::parse("https://huggingface.co/api/models").map_err(|e| e.to_string())?;
    url.query_pairs_mut()
        .append_pair("search", query)
        .append_pair("filter", "gguf")
        .append_pair("sort", "downloads")
        .append_pair("direction", "-1")
        .append_pair("limit", &limit.clamp(1, 50).to_string());
    let found: Vec<Value> = get(&client, url, &token())?.json().map_err(|e| e.to_string())?;
    Ok(found
        .into_iter()
        .filter_map(|m| {
            let id = m.get("id").or_else(|| m.get("modelId"))?.as_str()?.to_owned();
            Some(serde_json::json!({
                "id": id,
                "downloads": m.get("downloads").and_then(Value::as_u64).unwrap_or(0),
                "likes": m.get("likes").and_then(Value::as_u64).unwrap_or(0),
                "updated": m.get("lastModified").and_then(Value::as_str).unwrap_or(""),
            }))
        })
        .collect())
}

#[cfg(test)]
mod tests {
    #[test]
    fn pasted_links_become_hub_references() {
        for (input, expected) in [
            ("https://huggingface.co/owner/repo", "hf://owner/repo"),
            ("https://huggingface.co/owner/repo/", "hf://owner/repo"),
            ("https://huggingface.co/owner/repo/blob/main/model-Q4_K_M.gguf", "hf://owner/repo/model-Q4_K_M.gguf"),
            ("https://huggingface.co/owner/repo/resolve/main/sub/m.gguf?download=true", "hf://owner/repo/sub/m.gguf"),
            ("hf.co/owner/repo:Q4_K_M", "hf://owner/repo:Q4_K_M"),
            ("huggingface.co/owner/repo", "hf://owner/repo"),
            ("owner/repo", "hf://owner/repo"),
            ("owner/repo:Q8_0", "hf://owner/repo:Q8_0"),
            ("hf://owner/repo/file.gguf", "hf://owner/repo/file.gguf"),
        ] {
            assert_eq!(normalize_reference(input), expected, "{input}");
        }
        for local in ["./model.gguf", "/models/m.gguf", "model.gguf", "C:/models/m.gguf", "~/m.gguf", "a/b/c.gguf", "missing/local.gguf"] {
            assert_eq!(normalize_reference(local), local);
        }
    }

    use super::*;
    #[test]
    fn references_reject_traversal_and_ambiguous_paths() {
        for value in [
            "hf://a",
            "hf://../repo/a.gguf",
            "hf://a/b/../secret",
            "hf://a/b/a\\b",
            "hf://a/b//x",
            "hf://a/b/C:x",
        ] {
            assert!(parse_reference(value).is_err(), "{value}");
        }
        assert_eq!(
            parse_reference("hf://org/model/sub/model.gguf")
                .unwrap()
                .file
                .as_deref(),
            Some("sub/model.gguf")
        );
    }
    #[test]
    fn local_paths_are_network_free() {
        assert_eq!(
            resolve("missing/local.gguf", None, true).unwrap(),
            PathBuf::from("missing/local.gguf")
        );
    }
    #[test]
    fn revisions_are_single_encoded_url_segments() {
        assert!(url(&["a", "b", "resolve", "feature/test"])
            .unwrap()
            .as_str()
            .contains("feature%2Ftest"));
        assert!(!is_commit("../../"));
        assert!(is_commit("0123456789012345678901234567890123456789"));
    }
    fn checkpoint_fixture() -> PathBuf {
        let path = env::temp_dir().join(format!(
            "fllm-hub-cache-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&path).unwrap();
        fs::write(path.join("config.json"), "{}").unwrap();
        path
    }

    #[test]
    fn offline_cache_requires_every_indexed_shard_but_concrete_file_is_not_full_checkpoint() {
        let snapshot = checkpoint_fixture();
        let first = "model-00001-of-00002.safetensors";
        let second = "model-00002-of-00002.safetensors";
        fs::write(
            snapshot.join("model.safetensors.index.json"),
            format!(r#"{{"weight_map":{{"a":"{first}","b":"{second}"}}}}"#),
        )
        .unwrap();
        fs::write(snapshot.join(first), b"first").unwrap();
        assert!(cached_result(&snapshot, None)
            .unwrap_err()
            .contains("missing shard"));
        assert_eq!(
            cached_result(&snapshot, Some(first)).unwrap(),
            snapshot.join(first)
        );
        fs::write(snapshot.join(second), b"second").unwrap();
        assert_eq!(cached_result(&snapshot, None).unwrap(), snapshot);
        fs::remove_dir_all(snapshot).unwrap();
    }

    #[test]
    fn offline_cache_rejects_unsafe_or_missing_manifest_shards() {
        let snapshot = checkpoint_fixture();
        fs::write(
            snapshot.join("model.safetensors.index.json"),
            r#"{"weight_map":{"x":"../escape.safetensors"}}"#,
        )
        .unwrap();
        assert!(cached_result(&snapshot, None)
            .unwrap_err()
            .contains("invalid relative model filename"));
        fs::remove_dir_all(snapshot).unwrap();
    }

    #[test]
    fn unindexed_online_selection_prefers_single_model_file_and_rejects_ambiguity() {
        assert_eq!(
            select_unindexed_weight(&["adapter_model.safetensors", "model.safetensors"]).unwrap(),
            "model.safetensors"
        );
        assert!(
            select_unindexed_weight(&["model-00001.safetensors", "adapter_model.safetensors"])
                .is_err()
        );
        assert!(select_unindexed_weight(&["adapter_model.safetensors"]).is_err());
        assert_eq!(
            select_unindexed_weight(&["pytorch_model.bin"]).unwrap(),
            "pytorch_model.bin"
        );
    }
}

#[cfg(test)]
mod studio_tests {
    use super::*;

    #[test]
    fn quantization_is_read_from_the_file_name() {
        assert_eq!(quant_of_filename("Qwen3-8B-Q4_K_M.gguf").as_deref(), Some("Q4_K_M"));
        assert_eq!(quant_of_filename("sub/dir/model.IQ3_XS.gguf").as_deref(), Some("IQ3_XS"));
        assert_eq!(quant_of_filename("gemma-3-12b-it-UD-Q8_0-00001-of-00002.gguf").as_deref(), Some("Q8_0"));
        assert_eq!(quant_of_filename("model-bf16.gguf").as_deref(), Some("BF16"));
        assert_eq!(quant_of_filename("Qwen3-8B.gguf"), None);
    }

    #[test]
    fn the_model_card_summary_skips_front_matter_and_markup() {
        let card = "---\nlicense: mit\ntags:\n- gguf\n---\n# My Model\n\n![logo](x.png)\n<p align=\"center\"><img src=\"a\"></p>\nA small model for chat.\n\n```sh\nrun it\n```\n| a | b |\nSecond paragraph.";
        let text = readme_summary(card, 600);
        assert!(!text.contains("license") && !text.contains("logo") && !text.contains("run it") && !text.contains('<'), "{text}");
        assert!(text.starts_with("My Model") && text.contains("A small model for chat.") && text.contains("Second paragraph."), "{text}");
        let long = readme_summary(&"word ".repeat(400), 100);
        assert!(long.chars().count() <= 104 && long.ends_with("..."));
    }
}
