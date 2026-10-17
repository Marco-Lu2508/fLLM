//! Tests of the checkpoint import in `run` / `serve` / `pull`: classification, cache keys, cleanup of interrupted
//! conversions, the architecture and tokenizer checks that happen before weights are fetched, and the whole Hub flow
//! against a local HTTP stub (nothing here touches the network).

use super::*;
use crate::import_tests::{build_pair, scratch, tensor_types, Kind};
use std::{
    collections::HashMap,
    ffi::OsString,
    io::{Read, Write},
    net::TcpListener,
    sync::{Arc, Mutex, MutexGuard},
    thread::JoinHandle,
};

static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Sets environment variables for one test and restores them afterwards; tests that use it run one at a time.
struct EnvGuard {
    saved: Vec<(String, Option<OsString>)>,
    _lock: MutexGuard<'static, ()>,
}

impl EnvGuard {
    fn set(vars: &[(&str, Option<&str>)]) -> Self {
        let lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut saved = Vec::new();
        for (key, value) in vars {
            saved.push(((*key).to_owned(), std::env::var_os(key)));
            match value {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
        Self { saved, _lock: lock }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, value) in &self.saved {
            match value {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }
}

type Routes = HashMap<String, (u16, Vec<u8>)>;

/// A tiny HTTP server that answers fixed routes and records every request path (and whether it carried a bearer token).
struct Stub {
    base: String,
    hits: Arc<Mutex<Vec<(String, bool)>>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Stub {
    fn start(routes: Routes) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let hits = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let routes = Arc::new(routes);
        let (h, s) = (hits.clone(), stop.clone());
        let thread = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((mut conn, _)) => {
                        let (routes, hits) = (routes.clone(), h.clone());
                        std::thread::spawn(move || {
                            conn.set_nonblocking(false).unwrap();
                            let mut request = Vec::new();
                            let mut buf = [0u8; 1024];
                            while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                                match conn.read(&mut buf) {
                                    Ok(0) | Err(_) => return,
                                    Ok(n) => request.extend_from_slice(&buf[..n]),
                                }
                            }
                            let text = String::from_utf8_lossy(&request).into_owned();
                            let target = text.split_whitespace().nth(1).unwrap_or("/");
                            let path = target.split('?').next().unwrap_or("/").to_owned();
                            let bearer = text.to_ascii_lowercase().contains("authorization: bearer");
                            hits.lock().unwrap().push((path.clone(), bearer));
                            let (status, body) = routes.get(&path).cloned().unwrap_or((404, b"not found".to_vec()));
                            let head = format!(
                                "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\nContent-Type: application/octet-stream\r\n\r\n",
                                body.len()
                            );
                            let _ = conn.write_all(head.as_bytes());
                            let _ = conn.write_all(&body);
                        });
                    }
                    Err(_) => std::thread::sleep(Duration::from_millis(5)),
                }
            }
        });
        Self { base, hits, stop, thread: Some(thread) }
    }

    fn paths(&self) -> Vec<String> {
        self.hits.lock().unwrap().iter().map(|(p, _)| p.clone()).collect()
    }
}

impl Drop for Stub {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

/// The routes of a Hub repository with these files: the revision listing and the file downloads.
fn repo_routes(owner: &str, repo: &str, files: &[(String, Vec<u8>)]) -> Routes {
    let mut routes = Routes::new();
    let siblings: Vec<Value> = files.iter().map(|(n, b)| json!({"rfilename": n, "size": b.len()})).collect();
    routes.insert(
        format!("/api/models/{owner}/{repo}/revision/main"),
        (200, json!({"sha": COMMIT, "siblings": siblings}).to_string().into_bytes()),
    );
    for (name, bytes) in files {
        routes.insert(format!("/{owner}/{repo}/resolve/{COMMIT}/{name}"), (200, bytes.clone()));
    }
    routes
}

fn dir_files(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut files: Vec<_> = fs::read_dir(dir)
        .unwrap()
        .flatten()
        .filter(|e| e.path().is_file())
        .map(|e| (e.file_name().to_string_lossy().into_owned(), fs::read(e.path()).unwrap()))
        .filter(|(n, _)| !n.ends_with(".gguf"))
        .collect();
    files.sort();
    files
}

fn hub_env(stub: &Stub, cache: &Path, home: &Path) -> EnvGuard {
    EnvGuard::set(&[
        ("HF_ENDPOINT", Some(&stub.base)),
        ("FLLM_CACHE", cache.to_str()),
        ("HF_HOME", home.to_str()),
        ("HF_TOKEN", None),
        ("HUGGING_FACE_HUB_TOKEN", None),
    ])
}

fn position(paths: &[String], suffix: &str) -> Option<usize> {
    paths.iter().position(|p| p.ends_with(suffix))
}

#[test]
fn repositories_are_classified_by_their_files() {
    assert_eq!(classify_repo(&["config.json", "model.safetensors", "tokenizer.json"]), RepoKind::Checkpoint);
    assert_eq!(classify_repo(&["config.json", "model.safetensors.index.json", "model-00001-of-00002.safetensors"]), RepoKind::Checkpoint);
    assert_eq!(classify_repo(&["config.json", "pytorch_model.bin"]), RepoKind::Checkpoint);
    assert_eq!(classify_repo(&["model-Q4_K_M.gguf"]), RepoKind::Gguf);
    // a repository with both prefers the GGUF
    assert_eq!(classify_repo(&["config.json", "model.safetensors", "model.Q8_0.GGUF"]), RepoKind::Gguf);
    assert!(matches!(classify_repo(&["README.md"]), RepoKind::Neither(r) if r.contains("no GGUF")));
    assert!(matches!(classify_repo(&["config.json", "model-00001-of-00002.safetensors"]), RepoKind::Neither(r) if r.contains("no model.safetensors")));
    assert!(matches!(classify_repo(&["adapter_config.json", "adapter_model.safetensors"]), RepoKind::Neither(r) if r.contains("LoRA")));
}

#[test]
fn output_type_follows_the_checkpoint_and_the_flags() {
    assert_eq!(choose_outtype("none", None, None).unwrap(), OutType::Preserve);
    assert_eq!(choose_outtype("nvfp4", None, None).unwrap(), OutType::Preserve);
    assert_eq!(choose_outtype("none", None, Some("f16")).unwrap(), OutType::F16);
    assert_eq!(choose_outtype("none", Some("Q4_K_M"), None).unwrap(), OutType::Quant("q4_k_m".into()));
    let already = choose_outtype("awq", Some("q4_k_m"), None).unwrap_err();
    assert!(already.contains("already quantized (awq)"), "{already}");
    assert!(choose_outtype("none", Some("q3_banana"), None).unwrap_err().contains("--quant must be one of"));
    assert!(choose_outtype("none", Some("q8_0"), Some("f16")).is_err());
    assert!(choose_outtype("none", None, Some("bf16")).is_err());
}

#[test]
fn import_flags_are_parsed_and_validated() {
    let args = |v: &[&str]| v.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
    let mut options = ImportOptions::default();
    let a = args(&["--quant", "q4_k_m", "--revision", "v1", "--local-files-only", "--ctx", "8192"]);
    let mut i = 0;
    assert!(options.parse_flag(&a, &mut i).unwrap());
    assert_eq!(i, 1);
    i += 1;
    assert!(options.parse_flag(&a, &mut i).unwrap());
    i += 1;
    assert!(options.parse_flag(&a, &mut i).unwrap());
    i += 1;
    assert!(!options.parse_flag(&a, &mut i).unwrap(), "--ctx belongs to the server");
    assert_eq!(options, ImportOptions { revision: Some("v1".into()), offline: true, quant: Some("q4_k_m".into()), outtype: None, direct: false });
    let mut direct_opts = ImportOptions::default();
    assert!(direct_opts.parse_flag(&args(&["--direct"]), &mut 0).unwrap());
    assert!(direct_opts.direct);
    let mut zc_opts = ImportOptions::default();
    assert!(zc_opts.parse_flag(&args(&["--zero-copy"]), &mut 0).unwrap());
    assert!(zc_opts.direct);
    assert!(ImportOptions { direct: true, quant: Some("q4_0".into()), ..Default::default() }.validate().is_err());
    assert!(ImportOptions { direct: true, outtype: Some("f16".into()), ..Default::default() }.validate().is_err());
    assert!(options.validate().is_ok());
    assert!(ImportOptions { quant: Some("q9".into()), ..Default::default() }.validate().is_err());
    assert!(ImportOptions { quant: Some("q8_0".into()), outtype: Some("f16".into()), ..Default::default() }.validate().is_err());
    assert!(options.parse_flag(&args(&["--quant"]), &mut 0).unwrap_err().contains("missing value"));
}

#[test]
fn cache_keys_are_stable_and_depend_on_every_part() {
    assert_eq!(cache_key(&["a", "b"]), "cf6ab613e394");
    assert_ne!(cache_key(&["a", "b"]), cache_key(&["ab"]));
    assert_ne!(cache_key(&["a", "b"]), cache_key(&["b", "a"]));
    assert_ne!(cache_key(&["x", "q4_k_m"]), cache_key(&["x", "q8_0"]));
    let name = converted_name("Llama-3.1-8B", "nvfp4", &cache_key(&["a", "b"]));
    assert_eq!(name, "Llama-3.1-8B-nvfp4-fllmcf6ab613e394.gguf");
    assert!(hub::is_converted_name(&name));
    assert_eq!(display_name(name.strip_suffix(".gguf").unwrap()), "Llama-3.1-8B-nvfp4");
    assert_eq!(display_name("plain-model-Q4_K_M"), "plain-model-Q4_K_M");
    assert!(!hub::is_converted_name("model-Q4_K_M.gguf"));
    assert!(!hub::is_converted_name("x-fllmZZZZZZZZZZZZ.gguf"));
    assert_eq!(converted_name("a b/c", "q4_k_m", "000000000000"), "a_b_c-q4_k_m-fllm000000000000.gguf");
}

#[test]
fn disk_and_memory_estimates() {
    let gib = 1u64 << 30;
    assert!(disk_needed(10 * gib, 10 * gib, &OutType::Preserve) > 20 * gib);
    assert!(disk_needed(0, 10 * gib, &OutType::Quant("q4_k_m".into())) >= 15 * gib);
    assert!(disk_needed(0, 10 * gib, &OutType::F32) > disk_needed(0, 10 * gib, &OutType::F16));
    let df = "Filesystem     1024-blocks      Used Available Capacity Mounted on\n/dev/sda1        100000000  60000000  40000000      61% /\n";
    assert_eq!(parse_df(df), Some(40_000_000 * 1024));
    assert_eq!(parse_df(""), None);
    assert_eq!(parse_df("garbage line"), None);
    // too little room is reported with the numbers and the way out
    if let Some(free) = free_space(&std::env::temp_dir()) {
        let err = check_disk(&std::env::temp_dir(), free.saturating_add(1 << 40), 5 << 30).unwrap_err();
        assert!(err.contains("not enough free disk space") && err.contains("FLLM_CACHE"), "{err}");
        assert!(check_disk(&std::env::temp_dir(), 1, 0).is_ok());
    }
    // 7B+ in 16 bit on a small device: hint; quantized or small models: none
    assert!(memory_hint(15_000_000_000, 8_000_000_000, &OutType::Preserve, "none").unwrap().contains("--quant q4_k_m"));
    assert!(memory_hint(15_000_000_000, 80_000_000_000, &OutType::Preserve, "none").is_none());
    assert!(memory_hint(15_000_000_000, 8_000_000_000, &OutType::Preserve, "nvfp4").is_none());
    assert!(memory_hint(15_000_000_000, 8_000_000_000, &OutType::Quant("q4_k_m".into()), "none").is_none());
    assert!(memory_hint(3_000_000_000, 1_000_000_000, &OutType::Preserve, "none").is_none());
    assert!(memory_hint(15_000_000_000, 0, &OutType::Preserve, "none").is_none(), "unknown memory gives no hint");
}

fn origin_for<'a>(outtype: &'a OutType, check: &'a fllm_checkpoint::ConfigCheck) -> Origin<'a> {
    Origin { source: "test", revision: "-", outtype, check }
}

#[test]
fn interrupted_conversions_are_cleaned_up_and_failures_leave_nothing() {
    let quant = scratch("clean-q");
    let plain = scratch("clean-r");
    build_pair(Kind::Nvfp4, 21, &quant, &plain);
    let out = scratch("clean-out");
    let final_path = out.join("m-nvfp4-fllm000000000001.gguf");
    let check = fllm_checkpoint::check_config(&quant).unwrap();
    // leftovers of an earlier run that was killed
    let stale = [
        part_path(&final_path, ""),
        part_path(&final_path, ".f16"),
        PathBuf::from(format!("{}.12.345.tmp", part_path(&final_path, "").display())),
    ];
    for s in &stale {
        fs::write(s, b"half a model").unwrap();
    }
    let unrelated = out.join("other-model.gguf.fllm-part");
    fs::write(&unrelated, b"someone else's download").unwrap();
    let preserve = OutType::Preserve;
    convert_into(&quant, &final_path, &origin_for(&preserve, &check)).unwrap();
    assert!(final_path.is_file());
    assert!(stale.iter().all(|s| !s.exists()), "stale partial files must be removed");
    assert!(unrelated.exists(), "files of other models are not touched");
    assert!(converted_note(&final_path).unwrap().contains("converted from test"));
    assert_eq!(tensor_types(&final_path).get("blk.0.attn_q.weight"), Some(&40));

    // a conversion that fails leaves no model and no partial file behind
    let broken = scratch("clean-broken");
    fs::copy(quant.join("config.json"), broken.join("config.json")).unwrap();
    fs::copy(quant.join("tokenizer.json"), broken.join("tokenizer.json")).unwrap();
    let weights = fs::read(quant.join("model.safetensors")).unwrap();
    fs::write(broken.join("model.safetensors"), &weights[..weights.len() / 2]).unwrap();
    let failed = out.join("broken-nvfp4-fllm000000000002.gguf");
    fs::write(part_path(&failed, ".f16"), b"stale").unwrap();
    let err = convert_into(&broken, &failed, &origin_for(&preserve, &check)).unwrap_err();
    assert!(err.contains("conversion failed"), "{err}");
    let left: Vec<String> = fs::read_dir(&out)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("broken"))
        .collect();
    assert!(left.iter().all(|n| n.ends_with(".fllm-lock")), "{left:?}");
    for d in [quant, plain, out, broken] {
        let _ = fs::remove_dir_all(d);
    }
}

#[test]
fn two_starts_of_the_same_conversion_produce_one_file() {
    let quant = scratch("race-q");
    let plain = scratch("race-r");
    build_pair(Kind::Nvfp4, 71, &quant, &plain);
    let out = scratch("race-out");
    let final_path = out.join("m-nvfp4-fllm000000000003.gguf");
    let check = fllm_checkpoint::check_config(&quant).unwrap();
    let results: Vec<Result<(), String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..3)
            .map(|_| scope.spawn(|| convert_into(&quant, &final_path, &origin_for(&OutType::Preserve, &check))))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert!(results.iter().all(Result::is_ok), "{results:?}");
    assert_eq!(tensor_types(&final_path).get("blk.0.attn_q.weight"), Some(&40));
    let files: Vec<String> = fs::read_dir(&out).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    assert!(files.iter().all(|f| !f.contains(".fllm-part")), "{files:?}");
    for d in [quant, plain, out] {
        let _ = fs::remove_dir_all(d);
    }
}

#[test]
fn local_directories_convert_once_and_follow_their_files() {
    let quant = scratch("local-q");
    let plain = scratch("local-r");
    build_pair(Kind::Nvfp4, 31, &quant, &plain);
    let cache = scratch("local-cache");
    let _env = EnvGuard::set(&[("FLLM_CACHE", cache.to_str()), ("HF_HOME", cache.to_str())]);
    let dir = quant.to_str().unwrap();
    let options = ImportOptions::default();
    let first = prepare(dir, &options).unwrap();
    assert!(first.starts_with(local_cache_dir()), "{}", first.display());
    assert_eq!(tensor_types(&first).get("blk.1.ffn_up.weight"), Some(&40));
    let modified = fs::metadata(&first).unwrap().modified().unwrap();
    let second = prepare(dir, &options).unwrap();
    assert_eq!(first, second);
    assert_eq!(fs::metadata(&second).unwrap().modified().unwrap(), modified, "second start must reuse the file");
    // list shows it with its origin
    let listed = hub::list_cache().unwrap();
    let entry = listed.iter().find(|e| e.path == first).expect("converted model listed");
    assert!(entry.note.contains("converted from") && entry.note.contains("nvfp4"), "{}", entry.note);
    // other parameters make another file
    let f16 = prepare(dir, &ImportOptions { outtype: Some("f16".into()), ..Default::default() }).unwrap();
    assert_ne!(f16, first);
    assert_eq!(tensor_types(&f16).get("blk.0.attn_q.weight"), Some(&1));
    // changing the checkpoint changes the key
    let mut config = fs::read_to_string(quant.join("config.json")).unwrap();
    config.push('\n');
    fs::write(quant.join("config.json"), config).unwrap();
    let third = prepare(dir, &options).unwrap();
    assert_ne!(third, first);
    // removal by the directory
    assert_eq!(remove_local_conversions(&quant).unwrap(), 6);
    assert!(!first.exists() && !third.exists() && !sidecar_path(&first).exists());
    assert!(remove_local_conversions(&quant).unwrap_err().contains("no converted model"));
    for d in [quant, plain, cache] {
        let _ = fs::remove_dir_all(d);
    }
}

#[test]
fn quant_re_encodes_an_unquantized_directory_and_is_refused_for_quantized_ones() {
    let quant = scratch("q8-q");
    let plain = scratch("q8-r");
    build_pair(Kind::Nvfp4, 41, &quant, &plain);
    let cache = scratch("q8-cache");
    let _env = EnvGuard::set(&[("FLLM_CACHE", cache.to_str()), ("HF_HOME", cache.to_str())]);
    let path = prepare(plain.to_str().unwrap(), &ImportOptions { quant: Some("q8_0".into()), ..Default::default() }).unwrap();
    let types = tensor_types(&path);
    assert_eq!(types.get("blk.0.attn_q.weight"), Some(&8), "linear weights become Q8_0");
    assert_eq!(types.get("blk.0.attn_norm.weight"), Some(&1), "norms stay F16");
    let leftovers: Vec<_> = fs::read_dir(local_cache_dir())
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains(".fllm-part"))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
    let err = prepare(quant.to_str().unwrap(), &ImportOptions { quant: Some("q4_k_m".into()), ..Default::default() }).unwrap_err();
    assert!(err.contains("already quantized (nvfp4)"), "{err}");
    let err = prepare(plain.to_str().unwrap(), &ImportOptions { quant: Some("nope".into()), ..Default::default() }).unwrap_err();
    assert!(err.contains("--quant must be one of"), "{err}");
    for d in [quant, plain, cache] {
        let _ = fs::remove_dir_all(d);
    }
}

#[test]
fn directories_with_a_gguf_or_without_a_checkpoint_are_left_alone() {
    let dir = scratch("plain-dir");
    fs::write(dir.join("model.gguf"), b"GGUF").unwrap();
    assert_eq!(prepare(dir.to_str().unwrap(), &ImportOptions::default()).unwrap(), dir);
    assert_eq!(prepare("missing/local.gguf", &ImportOptions::default()).unwrap(), PathBuf::from("missing/local.gguf"));
    let file = dir.join("model.gguf");
    assert_eq!(prepare(file.to_str().unwrap(), &ImportOptions::default()).unwrap(), file);
    assert!(prepare(file.to_str().unwrap(), &ImportOptions { quant: Some("q8_0".into()), ..Default::default() })
        .unwrap_err()
        .contains("not one"));
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn hub_checkpoint_is_downloaded_converted_cached_and_removable() {
    let quant = scratch("hub-q");
    let plain = scratch("hub-r");
    build_pair(Kind::Nvfp4, 51, &quant, &plain);
    let stub = Stub::start(repo_routes("acme", "tiny-nvfp4", &dir_files(&quant)));
    let (cache, home) = (scratch("hub-cache"), scratch("hub-home"));
    let _env = hub_env(&stub, &cache, &home);

    let path = prepare("acme/tiny-nvfp4", &ImportOptions::default()).unwrap();
    assert!(path.starts_with(cache.join("models--acme--tiny-nvfp4").join("snapshots").join(COMMIT)), "{}", path.display());
    assert!(hub::is_converted_name(&path.file_name().unwrap().to_string_lossy()));
    assert_eq!(tensor_types(&path).get("blk.0.attn_q.weight"), Some(&40));
    let paths = stub.paths();
    let config = position(&paths, "/config.json").expect("config.json fetched");
    let weights = position(&paths, "/model.safetensors").expect("weights fetched");
    assert!(config < weights, "config.json must be fetched before the weights: {paths:?}");

    // a second start: only the revision lookup, no downloads, same file
    let before = stub.paths().len();
    let modified = fs::metadata(&path).unwrap().modified().unwrap();
    assert_eq!(prepare("acme/tiny-nvfp4", &ImportOptions::default()).unwrap(), path);
    let new_hits = stub.paths()[before..].to_vec();
    assert!(new_hits.iter().all(|p| p.starts_with("/api/")), "{new_hits:?}");
    assert_eq!(fs::metadata(&path).unwrap().modified().unwrap(), modified);

    // offline works from the cache and does not touch the network
    let before = stub.paths().len();
    let offline = ImportOptions { offline: true, ..Default::default() };
    assert_eq!(prepare("https://huggingface.co/acme/tiny-nvfp4", &offline).unwrap(), path);
    assert_eq!(stub.paths().len(), before);

    // list shows the checkpoint and the converted model
    let listed = hub::list_cache().unwrap();
    assert!(listed.iter().any(|e| e.reference == "hf://acme/tiny-nvfp4"));
    let converted = listed.iter().find(|e| e.path == path).expect("converted model listed");
    assert!(converted.reference.starts_with("hf://acme/tiny-nvfp4/") && converted.note.contains("converted from"));

    // a different output type is a different file; --quant on a quantized checkpoint is an error
    let f16 = prepare("acme/tiny-nvfp4", &ImportOptions { outtype: Some("f16".into()), ..Default::default() }).unwrap();
    assert_ne!(f16, path);
    assert!(prepare("acme/tiny-nvfp4", &ImportOptions { quant: Some("q8_0".into()), ..Default::default() })
        .unwrap_err()
        .contains("already quantized"));

    // remove only the converted files, then convert again offline from the downloaded checkpoint
    assert_eq!(hub::remove_converted("hf://acme/tiny-nvfp4").unwrap(), 4);
    assert!(!path.exists() && !f16.exists() && !sidecar_path(&path).exists());
    assert!(cache.join("models--acme--tiny-nvfp4/snapshots").join(COMMIT).join("model.safetensors").is_file());
    let again = prepare("acme/tiny-nvfp4", &offline).unwrap();
    assert_eq!(again, path);
    // removing the repository removes everything
    assert!(hub::remove_cached("hf://acme/tiny-nvfp4").unwrap() >= 4);
    assert!(hub::list_cache().unwrap().is_empty());
    for d in [quant, plain, cache, home] {
        let _ = fs::remove_dir_all(d);
    }
}

#[test]
fn unsupported_architecture_is_reported_before_the_weights_are_downloaded() {
    let dir = scratch("arch");
    fs::write(dir.join("config.json"), json!({"model_type": "totally_new", "hidden_size": 8}).to_string()).unwrap();
    fs::write(dir.join("tokenizer.json"), "{}").unwrap();
    fs::write(dir.join("model.safetensors"), vec![0u8; 1 << 20]).unwrap();
    let stub = Stub::start(repo_routes("acme", "novel", &dir_files(&dir)));
    let (cache, home) = (scratch("arch-cache"), scratch("arch-home"));
    let _env = hub_env(&stub, &cache, &home);
    let err = prepare("acme/novel", &ImportOptions::default()).unwrap_err();
    assert!(err.contains("totally_new") && err.contains("not supported") && err.contains("docs/architectures.md"), "{err}");
    assert!(err.contains("nothing else was downloaded"), "{err}");
    let paths = stub.paths();
    assert!(position(&paths, "/config.json").is_some());
    assert!(position(&paths, "/model.safetensors").is_none(), "weights must not be fetched: {paths:?}");
    for d in [dir, cache, home] {
        let _ = fs::remove_dir_all(d);
    }
}

#[test]
fn missing_tokenizer_is_reported_before_the_weights_are_downloaded() {
    let quant = scratch("tok-q");
    let plain = scratch("tok-r");
    build_pair(Kind::Nvfp4, 61, &quant, &plain);
    let files: Vec<_> = dir_files(&quant).into_iter().filter(|(n, _)| n != "tokenizer.json").collect();
    let stub = Stub::start(repo_routes("acme", "no-tokenizer", &files));
    let (cache, home) = (scratch("tok-cache"), scratch("tok-home"));
    let _env = hub_env(&stub, &cache, &home);
    let err = prepare("acme/no-tokenizer", &ImportOptions::default()).unwrap_err();
    assert!(err.contains("has no tokenizer the conversion can use"), "{err}");
    assert!(position(&stub.paths(), "/model.safetensors").is_none(), "{:?}", stub.paths());
    for d in [quant, plain, cache, home] {
        let _ = fs::remove_dir_all(d);
    }
}

#[test]
fn gated_and_private_repositories_name_the_token_variable() {
    let mut routes = Routes::new();
    routes.insert("/api/models/acme/secret/revision/main".into(), (401, b"{}".to_vec()));
    routes.insert("/api/models/acme/gated/revision/main".into(), (403, b"{}".to_vec()));
    let stub = Stub::start(routes);
    let (cache, home) = (scratch("gate-cache"), scratch("gate-home"));
    {
        let _env = hub_env(&stub, &cache, &home);
        let err = prepare("acme/secret", &ImportOptions::default()).unwrap_err();
        assert!(err.contains("HF_TOKEN") && err.contains("private or gated"), "{err}");
        let err = prepare("acme/gated", &ImportOptions::default()).unwrap_err();
        assert!(err.contains("HF_TOKEN") && err.contains("license"), "{err}");
        let err = prepare("acme/missing", &ImportOptions::default()).unwrap_err();
        assert!(err.contains("not found") && err.contains("404"), "{err}");
    }
    {
        let _env = hub_env(&stub, &cache, &home);
        std::env::set_var("HF_TOKEN", "test-token");
        let err = prepare("acme/gated", &ImportOptions::default()).unwrap_err();
        assert!(err.contains("accept the model license") && err.contains("HF_TOKEN"), "{err}");
        assert!(stub.hits.lock().unwrap().iter().any(|(_, bearer)| *bearer), "the token must be sent");
    }
    for d in [cache, home] {
        let _ = fs::remove_dir_all(d);
    }
}

#[test]
fn repositories_with_ggufs_keep_resolving_to_the_gguf() {
    let mut files: Vec<(String, Vec<u8>)> = vec![
        ("config.json".into(), b"{}".to_vec()),
        ("model.safetensors".into(), vec![0u8; 4096]),
        ("model-Q4_K_M.gguf".into(), [b"GGUF".as_slice(), &[0u8; 60]].concat()),
    ];
    files.sort();
    let stub = Stub::start(repo_routes("acme", "both", &files));
    let (cache, home) = (scratch("both-cache"), scratch("both-home"));
    let _env = hub_env(&stub, &cache, &home);
    let path = prepare("acme/both", &ImportOptions::default()).unwrap();
    assert!(path.ends_with("model-Q4_K_M.gguf"), "{}", path.display());
    assert!(position(&stub.paths(), "/model.safetensors").is_none());
    let err = prepare("acme/both", &ImportOptions { quant: Some("q8_0".into()), ..Default::default() }).unwrap_err();
    assert!(err.contains("holds GGUF files"), "{err}");
    // selecting a quantization or a file is the old behaviour and rejects conversion flags
    let selected = prepare("acme/both:Q4_K_M", &ImportOptions::default()).unwrap();
    assert_eq!(selected, path);
    assert!(prepare("acme/both:Q4_K_M", &ImportOptions { outtype: Some("f16".into()), ..Default::default() }).is_err());
    for d in [cache, home] {
        let _ = fs::remove_dir_all(d);
    }
}

#[test]
fn prepare_local_with_direct_flag_bypasses_conversion() {
    let quant = scratch("direct-q");
    let plain = scratch("direct-p");
    build_pair(Kind::Nvfp4, 88, &quant, &plain);
    let cache = scratch("direct-cache");
    let _env = EnvGuard::set(&[("FLLM_CACHE", cache.to_str()), ("HF_HOME", cache.to_str())]);
    let dir = plain.to_str().unwrap();
    let path = prepare(dir, &ImportOptions { direct: true, ..Default::default() }).unwrap();
    assert_eq!(path, plain, "direct mode should return input directory directly");
    assert!(fs::read_dir(&cache).unwrap().next().is_none(), "cache must remain empty in direct mode");
    for d in [quant, plain, cache] {
        let _ = fs::remove_dir_all(d);
    }
}
