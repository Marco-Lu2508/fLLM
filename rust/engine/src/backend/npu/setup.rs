//! Installing the OpenVINO runtime for the Intel NPU (`fllm-rs npu-setup`, and `POST /api/studio/npu/setup` of the Studio).
//!
//! What is installed: the Python wheel `openvino` of one pinned release. The wheel is the official binary package of the C runtime
//! (`libopenvino_c`, `libopenvino`, the NPU plug-in and the libraries they need); only `openvino/libs/` is unpacked, into
//! `~/.fllm/openvino` (`$FLLM_HOME/openvino`), where the backend looks for it (`openvino/ffi.rs`, same layout as `pip install --target`).
//! No Python is needed: the file is downloaded, checked and unpacked here.
//!
//! Where the pinned values come from: the file list of the release on PyPI (`https://pypi.org/pypi/openvino/<version>/json`: file name,
//! URL, size, SHA-256) read on 2026-10-25. The download is refused unless it comes from `https://files.pythonhosted.org/` and its
//! size and SHA-256 equal the pinned ones; an interrupted download is resumed. OpenVINO is licensed under Apache-2.0; nothing of it is
//! redistributed by fLLM (the user's machine downloads it from PyPI).
//!
//! Verification level: L0 for the download and the unpacking of the real wheel (the code ran against hand-made archives only; the
//! pinned values were copied from the PyPI page and have not been used against the real file by the authors).

use sha2::{Digest, Sha256};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// The pinned release.
pub const OPENVINO_VERSION: &str = "2026.4.1";
/// Where the pinned values come from (shown to the user).
pub const PIN_SOURCE: &str = "pypi.org/project/openvino/2026.4.1 (file list with SHA-256 digests)";
/// The only host a download may come from.
pub const OFFICIAL_HOST: &str = "files.pythonhosted.org";
/// Name of the file that records what was installed.
pub const MARKER: &str = "fllm-runtime.json";

/// One downloadable file of the pinned release.
#[derive(Debug, PartialEq, Eq)]
pub struct Artifact {
    /// `os-arch` as in `std::env::consts` (`linux-x86_64`, `linux-aarch64`, `windows-x86_64`).
    pub platform: &'static str,
    pub file_name: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub size: u64,
}

// The wheels for CPython 3.12 are used on every Python-less machine: the files under `openvino/libs/` are the same in all tags of one
// platform; only the Python bindings differ, and they are not unpacked.
pub const ARTIFACTS: [Artifact; 3] = [
    Artifact {
        platform: "linux-x86_64",
        file_name: "openvino-2026.4.1-22982-cp312-cp312-manylinux_2_28_x86_64.whl",
        url: "https://files.pythonhosted.org/packages/ec/3a/2a173ac1ad749ff0b041788eefc1ade0d410231fedfc43f77474f3b806cc/openvino-2026.4.1-22982-cp312-cp312-manylinux_2_28_x86_64.whl",
        sha256: "6b4375c17ddcac83a5180349e2e2bb811185c261066e2a920659892d58ef0e3b",
        size: 59_144_434,
    },
    Artifact {
        platform: "linux-aarch64",
        file_name: "openvino-2026.4.1-22982-cp312-cp312-manylinux_2_35_aarch64.whl",
        url: "https://files.pythonhosted.org/packages/b2/d7/390c0ec5b81b6e089b012aaba6a2dc14f3ac7c52bfd78d10f074e72616ab/openvino-2026.4.1-22982-cp312-cp312-manylinux_2_35_aarch64.whl",
        sha256: "82efccb2f9f1bdc7e5a1996e05a3b719ebff9232dd54b44150d6d2e983a86b7d",
        size: 30_332_437,
    },
    Artifact {
        platform: "windows-x86_64",
        file_name: "openvino-2026.4.1-22982-cp312-cp312-win_amd64.whl",
        url: "https://files.pythonhosted.org/packages/d0/44/66a61b7cfccea1dfa20e95a04b4157f07a0e4dc3f7e894b22a92abb8822b/openvino-2026.4.1-22982-cp312-cp312-win_amd64.whl",
        sha256: "4e04316abff1b99e29b8cbd38deaef9bde4739eba216d982d4b3981e456ecd87",
        size: 84_964_415,
    },
];

/// The file for a platform (`os` and `arch` as in `std::env::consts`).
pub fn artifact_for(os: &str, arch: &str) -> Option<&'static Artifact> {
    let key = format!("{os}-{arch}");
    ARTIFACTS.iter().find(|a| a.platform == key)
}

/// The file for this machine, `None` where no runtime is pinned.
pub fn host_artifact() -> Option<&'static Artifact> {
    artifact_for(std::env::consts::OS, std::env::consts::ARCH)
}

/// Whether `url` is an https address on the official host (the host name is compared exactly).
pub fn url_is_official(url: &str) -> bool {
    reqwest::Url::parse(url).is_ok_and(|u| u.scheme() == "https" && u.host_str() == Some(OFFICIAL_HOST) && u.port().is_none() && u.username().is_empty() && u.password().is_none())
}

/// Whether `digest` (any case) is a SHA-256 in hex.
pub fn is_sha256(digest: &str) -> bool {
    digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit())
}

// ------------------------------------------------------------------------------------------------------------------------------
// Where things are
// ------------------------------------------------------------------------------------------------------------------------------

/// `$FLLM_HOME`, else `~/.fllm`.
pub fn fllm_home() -> Option<PathBuf> {
    std::env::var_os("FLLM_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).filter(|v| !v.is_empty()).map(|h| PathBuf::from(h).join(".fllm")))
}

/// The directory the runtime is installed in.
pub fn install_dir() -> Option<PathBuf> {
    fllm_home().map(|h| h.join("openvino"))
}

/// What the marker file of the installation says.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Installed {
    pub version: String,
    pub dir: PathBuf,
}

/// The installation made by this module (or `npu-setup` of an earlier version, which wrote no marker: version unknown).
pub fn installed() -> Option<Installed> {
    let dir = install_dir()?;
    if !dir.is_dir() {
        return None;
    }
    let version = std::fs::read_to_string(dir.join(MARKER))
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|v| v["version"].as_str().map(str::to_owned))
        .unwrap_or_else(|| "unknown".to_owned());
    Some(Installed { version, dir })
}

// ------------------------------------------------------------------------------------------------------------------------------
// The state machine of one setup run
// ------------------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Idle,
    Downloading,
    Verifying,
    Unpacking,
    Done,
    Failed,
    Cancelled,
}

impl Phase {
    pub fn as_str(self) -> &'static str {
        match self {
            Phase::Idle => "idle",
            Phase::Downloading => "downloading",
            Phase::Verifying => "verifying",
            Phase::Unpacking => "unpacking",
            Phase::Done => "done",
            Phase::Failed => "failed",
            Phase::Cancelled => "cancelled",
        }
    }

    /// A run is going on.
    pub fn busy(self) -> bool {
        matches!(self, Phase::Downloading | Phase::Verifying | Phase::Unpacking)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    Start,
    Downloaded,
    Verified,
    Unpacked,
    Cancel,
    Fail,
}

/// The phase after `event`; `Err` for an event that does not belong to the phase (for example a second start while a run goes on).
pub fn transition(phase: Phase, event: &Event) -> Result<Phase, String> {
    match (phase, event) {
        (Phase::Idle | Phase::Done | Phase::Failed | Phase::Cancelled, Event::Start) => Ok(Phase::Downloading),
        (p, Event::Start) if p.busy() => Err("the setup is already running".to_owned()),
        (Phase::Downloading, Event::Downloaded) => Ok(Phase::Verifying),
        (Phase::Verifying, Event::Verified) => Ok(Phase::Unpacking),
        (Phase::Unpacking, Event::Unpacked) => Ok(Phase::Done),
        (p, Event::Cancel) if p.busy() => Ok(Phase::Cancelled),
        (p, Event::Fail) if p.busy() => Ok(Phase::Failed),
        (p, e) => Err(format!("event {e:?} does not fit the phase {}", p.as_str())),
    }
}

/// What the page shows about a run.
#[derive(Clone, Debug)]
pub struct Progress {
    pub phase: Phase,
    pub done: u64,
    pub total: u64,
    pub message: String,
    pub error: Option<String>,
}

impl Progress {
    const fn idle() -> Progress {
        Progress { phase: Phase::Idle, done: 0, total: 0, message: String::new(), error: None }
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "phase": self.phase.as_str(),
            "busy": self.phase.busy(),
            "done_bytes": self.done,
            "total_bytes": self.total,
            "message": self.message,
            "error": self.error,
        })
    }
}

static JOB: Mutex<Progress> = Mutex::new(Progress::idle());
static CANCEL: AtomicBool = AtomicBool::new(false);

fn job() -> std::sync::MutexGuard<'static, Progress> {
    JOB.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The state of the current or last run.
pub fn progress() -> Progress {
    job().clone()
}

fn advance(event: &Event, message: &str) {
    let mut state = job();
    if let Ok(next) = transition(state.phase, event) {
        state.phase = next;
    }
    state.message = message.to_owned();
}

fn report(done: u64, total: u64) {
    let mut state = job();
    state.done = done;
    state.total = total;
}

/// Asks the running setup to stop (the finished part of the download is kept for the next try).
pub fn cancel() -> bool {
    let running = job().phase.busy();
    if running {
        CANCEL.store(true, Ordering::Relaxed);
    }
    running
}

/// Starts the setup in a background thread. `Err` when one is running, or when this machine has no pinned runtime.
pub fn start() -> Result<(), String> {
    let artifact = host_artifact().ok_or_else(|| {
        format!("no OpenVINO runtime is pinned for this system ({} {}); install OpenVINO yourself and set FLLM_OPENVINO_DIR", std::env::consts::OS, std::env::consts::ARCH)
    })?;
    let home = fllm_home().ok_or("neither FLLM_HOME nor HOME is set")?;
    {
        let mut state = job();
        let next = transition(state.phase, &Event::Start)?;
        *state = Progress { phase: next, done: 0, total: artifact.size, message: "Downloading the OpenVINO runtime".to_owned(), error: None };
    }
    CANCEL.store(false, Ordering::Relaxed);
    std::thread::Builder::new()
        .name("fllm-npu-setup".into())
        .spawn(move || match run(artifact, &home) {
            Ok(()) => advance(&Event::Unpacked, "The OpenVINO runtime is installed"),
            Err(e) if e == CANCELLED => advance(&Event::Cancel, "The setup was cancelled; the finished part of the download is kept"),
            Err(e) => {
                let mut state = job();
                if let Ok(next) = transition(state.phase, &Event::Fail) {
                    state.phase = next;
                }
                state.message = "The setup failed".to_owned();
                state.error = Some(e);
            }
        })
        .map(|_| ())
        .map_err(|e| {
            advance(&Event::Fail, "The setup could not start");
            format!("cannot start the setup thread: {e}")
        })
}

const CANCELLED: &str = "cancelled";

fn cancelled() -> bool {
    CANCEL.load(Ordering::Relaxed)
}

// ------------------------------------------------------------------------------------------------------------------------------
// The work
// ------------------------------------------------------------------------------------------------------------------------------

fn sha256_of(path: &Path, step: &mut dyn FnMut(u64)) -> Result<String, String> {
    let mut file = std::fs::File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    let mut done = 0u64;
    loop {
        let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
        done += n as u64;
        step(done);
        if cancelled() {
            return Err(CANCELLED.to_owned());
        }
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn client() -> Result<reqwest::blocking::Client, String> {
    reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(20))
        .timeout(Duration::from_secs(3 * 3600))
        .user_agent("fllm-rust/0.1")
        // a redirect may only stay on the official host
        .redirect(reqwest::redirect::Policy::custom(|attempt| {
            if url_is_official(attempt.url().as_str()) && attempt.previous().len() < 5 {
                attempt.follow()
            } else {
                attempt.stop()
            }
        }))
        .build()
        .map_err(|e| e.to_string())
}

/// Downloads `artifact` to `target` (resuming a `.part` file); the result has the pinned size but is not yet checked against the digest.
fn download(artifact: &Artifact, target: &Path, step: &mut dyn FnMut(u64)) -> Result<(), String> {
    if !url_is_official(artifact.url) {
        return Err("the pinned address is not on the official download host".to_owned());
    }
    let part = PathBuf::from(format!("{}.part", target.display()));
    let mut file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&part).map_err(|e| format!("cannot write {}: {e}", part.display()))?;
    let mut offset = file.metadata().map_err(|e| e.to_string())?.len();
    if offset > artifact.size {
        file.set_len(0).map_err(|e| e.to_string())?;
        offset = 0;
    }
    if offset < artifact.size {
        let mut request = client()?.get(artifact.url).header(reqwest::header::ACCEPT_ENCODING, "identity");
        if offset > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={offset}-"));
        }
        let mut response = request.send().map_err(|e| format!("the download connection failed: {}", e.without_url()))?;
        match response.status() {
            reqwest::StatusCode::PARTIAL_CONTENT => {
                let start = response
                    .headers()
                    .get(reqwest::header::CONTENT_RANGE)
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.strip_prefix("bytes "))
                    .and_then(|s| s.split('-').next())
                    .and_then(|s| s.parse::<u64>().ok());
                if start != Some(offset) {
                    return Err("the server answered a resume request with another range; remove the partial download and retry".to_owned());
                }
            }
            reqwest::StatusCode::OK => {
                // the server ignored the range: start over, never append the whole file to the part
                file.set_len(0).map_err(|e| e.to_string())?;
                offset = 0;
            }
            other => return Err(format!("the download server answered HTTP {other}")),
        }
        if response.content_length().is_some_and(|n| offset + n != artifact.size) {
            return Err("the size announced by the server is not the pinned size".to_owned());
        }
        file.seek(SeekFrom::Start(offset)).map_err(|e| e.to_string())?;
        let mut buffer = vec![0u8; 1 << 20];
        let mut last = Instant::now();
        loop {
            let n = response.read(&mut buffer).map_err(|e| format!("the download was interrupted (run the setup again to resume): {e}"))?;
            if n == 0 {
                break;
            }
            file.write_all(&buffer[..n]).map_err(|e| format!("cannot write the download: {e}"))?;
            offset += n as u64;
            if offset > artifact.size {
                return Err("the download is larger than the pinned size".to_owned());
            }
            if cancelled() {
                return Err(CANCELLED.to_owned());
            }
            if last.elapsed() >= Duration::from_millis(300) {
                step(offset);
                last = Instant::now();
            }
        }
        step(offset);
    }
    file.sync_all().map_err(|e| e.to_string())?;
    drop(file);
    let size = std::fs::metadata(&part).map_err(|e| e.to_string())?.len();
    if size != artifact.size {
        return Err(format!("the download stopped at {size} of {} bytes (run the setup again to resume)", artifact.size));
    }
    std::fs::rename(&part, target).map_err(|e| format!("cannot finish the download file: {e}"))
}

/// Whether `dir` (the unpacked `openvino/libs`) holds the C runtime.
fn has_c_library(dir: &Path) -> bool {
    std::fs::read_dir(dir).is_ok_and(|entries| {
        entries.filter_map(Result::ok).any(|e| {
            let name = e.file_name().to_string_lossy().to_ascii_lowercase();
            name == "openvino_c.dll" || name.starts_with("libopenvino_c.so") || (name.starts_with("libopenvino_c.") && name.ends_with(".dylib"))
        })
    })
}

fn run(artifact: &Artifact, home: &Path) -> Result<(), String> {
    if !is_sha256(artifact.sha256) {
        return Err("the pinned checksum is invalid".to_owned());
    }
    let downloads = home.join("downloads").join("openvino");
    std::fs::create_dir_all(&downloads).map_err(|e| format!("cannot create {}: {e}", downloads.display()))?;
    let wheel = downloads.join(artifact.file_name);
    if !wheel.is_file() {
        download(artifact, &wheel, &mut |done| report(done, artifact.size))?;
    }
    advance(&Event::Downloaded, "Checking the download");
    report(0, artifact.size);
    let digest = sha256_of(&wheel, &mut |done| report(done, artifact.size))?;
    if !digest.eq_ignore_ascii_case(artifact.sha256) {
        let _ = std::fs::remove_file(&wheel);
        return Err(format!("the checksum of the download is wrong (expected {}, found {digest}); the file was deleted", artifact.sha256));
    }
    advance(&Event::Verified, "Unpacking the runtime");
    report(0, 0);
    let staging = home.join(format!("openvino.new-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&staging);
    let limits = super::zip::Limits { max_entry: 1 << 30, max_total: 3 << 30, max_entries: 20_000 };
    let unpacked = super::zip::extract(&wheel, &staging, &|name| name.starts_with("openvino/libs/"), &limits, &cancelled, &mut |done, total| report(done, total));
    let finish = unpacked.and_then(|_| {
        if !has_c_library(&staging.join("openvino").join("libs")) {
            return Err("the package does not contain the OpenVINO C runtime".to_owned());
        }
        let marker = serde_json::json!({"runtime": "openvino", "version": OPENVINO_VERSION, "file": artifact.file_name, "sha256": artifact.sha256, "source": PIN_SOURCE});
        std::fs::write(staging.join(MARKER), marker.to_string()).map_err(|e| e.to_string())?;
        let target = home.join("openvino");
        if target.exists() {
            std::fs::remove_dir_all(&target).map_err(|e| format!("cannot replace the old installation in {} (a running fLLM may hold its files; restart the app and retry): {e}", target.display()))?;
        }
        std::fs::rename(&staging, &target).map_err(|e| format!("cannot move the runtime into place: {e}"))
    });
    if finish.is_err() {
        let _ = std::fs::remove_dir_all(&staging);
    }
    finish?;
    let _ = std::fs::remove_file(&wheel);
    Ok(())
}

/// Deletes the installed runtime (and downloaded packages). `Err` while a setup runs or when files are in use.
pub fn remove() -> Result<(), String> {
    if job().phase.busy() {
        return Err("the setup is running; cancel it first".to_owned());
    }
    let home = fllm_home().ok_or("neither FLLM_HOME nor HOME is set")?;
    let target = home.join("openvino");
    if target.exists() {
        std::fs::remove_dir_all(&target).map_err(|e| {
            format!("cannot delete {}: {e}. If the NPU was in use, stop fLLM (or restart the app) and try again", target.display())
        })?;
    }
    let _ = std::fs::remove_dir_all(home.join("downloads").join("openvino"));
    *job() = Progress::idle();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_artifacts_are_consistent() {
        for artifact in &ARTIFACTS {
            assert!(url_is_official(artifact.url), "{}", artifact.platform);
            assert!(is_sha256(artifact.sha256), "{}", artifact.platform);
            assert!(artifact.url.ends_with(artifact.file_name));
            assert!(artifact.file_name.contains(OPENVINO_VERSION));
            assert!(artifact.size > 10_000_000);
        }
        let mut platforms: Vec<_> = ARTIFACTS.iter().map(|a| a.platform).collect();
        platforms.sort_unstable();
        platforms.dedup();
        assert_eq!(platforms.len(), ARTIFACTS.len());
    }

    #[test]
    fn the_artifact_follows_the_platform() {
        assert!(artifact_for("linux", "x86_64").unwrap().file_name.contains("manylinux_2_28_x86_64"));
        assert!(artifact_for("linux", "aarch64").unwrap().file_name.contains("aarch64"));
        assert!(artifact_for("windows", "x86_64").unwrap().file_name.ends_with("win_amd64.whl"));
        assert!(artifact_for("macos", "aarch64").is_none());
        assert!(artifact_for("windows", "aarch64").is_none());
    }

    #[test]
    fn only_the_official_host_is_accepted() {
        assert!(url_is_official("https://files.pythonhosted.org/packages/x/y.whl"));
        assert!(!url_is_official("http://files.pythonhosted.org/packages/x/y.whl"));
        assert!(!url_is_official("https://files.pythonhosted.org.evil.example/x"));
        assert!(!url_is_official("https://evil.example/files.pythonhosted.org/x"));
        assert!(!url_is_official("https://user@evil.example@files.pythonhosted.org/x"));
        assert!(!url_is_official("https://files.pythonhosted.org:8443/x"));
        assert!(!url_is_official("not a url"));
    }

    #[test]
    fn checksum_text_is_validated() {
        assert!(is_sha256(&"a".repeat(64)));
        assert!(is_sha256(&"A1".repeat(32)));
        assert!(!is_sha256(&"a".repeat(63)));
        assert!(!is_sha256(&"g".repeat(64)));
    }

    #[test]
    fn a_run_goes_through_the_phases_in_order() {
        let mut phase = Phase::Idle;
        for (event, expected) in [
            (Event::Start, Phase::Downloading),
            (Event::Downloaded, Phase::Verifying),
            (Event::Verified, Phase::Unpacking),
            (Event::Unpacked, Phase::Done),
        ] {
            phase = transition(phase, &event).unwrap();
            assert_eq!(phase, expected);
        }
        assert!(!phase.busy());
        // another run may follow a finished, failed or cancelled one
        for finished in [Phase::Done, Phase::Failed, Phase::Cancelled] {
            assert_eq!(transition(finished, &Event::Start), Ok(Phase::Downloading));
        }
    }

    #[test]
    fn a_second_start_and_wrong_events_are_refused() {
        for busy in [Phase::Downloading, Phase::Verifying, Phase::Unpacking] {
            assert!(busy.busy());
            assert!(transition(busy, &Event::Start).unwrap_err().contains("already running"));
            assert_eq!(transition(busy, &Event::Cancel), Ok(Phase::Cancelled));
            assert_eq!(transition(busy, &Event::Fail), Ok(Phase::Failed));
        }
        assert!(transition(Phase::Idle, &Event::Verified).is_err());
        assert!(transition(Phase::Downloading, &Event::Unpacked).is_err());
        assert!(transition(Phase::Done, &Event::Cancel).is_err());
        assert!(transition(Phase::Failed, &Event::Fail).is_err());
    }

    #[test]
    fn progress_is_reported_as_json() {
        let p = Progress { phase: Phase::Downloading, done: 5, total: 10, message: "m".into(), error: None };
        let json = p.to_json();
        assert_eq!(json["phase"], "downloading");
        assert_eq!(json["busy"], true);
        assert_eq!(json["done_bytes"], 5);
        assert!(json["error"].is_null());
        assert_eq!(Progress::idle().to_json()["busy"], false);
    }

    #[test]
    fn the_c_library_is_recognised_by_name() {
        let dir = std::env::temp_dir().join(format!("fllm-ov-setup-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!has_c_library(&dir));
        std::fs::write(dir.join("libopenvino.so.2641"), b"").unwrap();
        assert!(!has_c_library(&dir));
        let c = if cfg!(windows) { "openvino_c.dll" } else { "libopenvino_c.so.2641" };
        std::fs::write(dir.join(c), b"").unwrap();
        assert!(has_c_library(&dir));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
