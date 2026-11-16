//! Row 47 of PARITY.md: the app fetches its own new version. The bundle of the release that fits
//! this installation is downloaded, checked against the `.sha256` file of the release, unpacked
//! next to the data, and put in place when you press *Restart*.
//!
//! **The same build:** an installation with the CUDA engine gets the CUDA bundle, one with the AMD
//! engine the AMD bundle, a standard one the standard bundle. A release that has no bundle of that
//! build is not replaced by another one: you are told, and the page of the release is offered.
//!
//! **What this does not do:** the check is the checksum that the release publishes next to the
//! file, which finds a damaged or cut-off download. It is not a signature of the maker: the
//! bundles carry a Sigstore signature that this app does not verify.

use futures::StreamExt;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// The graphics build of an installation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Variant {
    #[default]
    Base,
    Cuda,
    Amd,
}

impl Variant {
    pub fn label(self) -> &'static str {
        match self {
            Variant::Base => "standard",
            Variant::Cuda => "CUDA",
            Variant::Amd => "AMD (HIP)",
        }
    }

    /// The word in `variant.txt` of a bundle.
    pub fn from_marker(text: &str) -> Option<Variant> {
        match text.trim().to_ascii_lowercase().as_str() {
            "base" | "standard" => Some(Variant::Base),
            "cuda" | "nvidia" => Some(Variant::Cuda),
            "amd" | "hip" | "rocm" => Some(Variant::Amd),
            _ => None,
        }
    }

    /// The backends the engine says it was built with (`... backends: cpu,vulkan,cuda`).
    pub fn from_build_info(info: &str) -> Variant {
        let backends = info.rsplit_once("backends:").map_or("", |(_, b)| b).to_ascii_lowercase();
        if backends.split([',', ' ']).any(|b| b == "cuda") {
            Variant::Cuda
        } else if backends.split([',', ' ']).any(|b| b == "hip" || b == "rocm") {
            Variant::Amd
        } else {
            Variant::Base
        }
    }

    /// What this installation is: the marker of the bundle beside the program first, then what the engine says.
    pub fn detect(beside: Option<&Path>, build_info: &str) -> Variant {
        beside
            .and_then(|d| std::fs::read_to_string(d.join("variant.txt")).ok())
            .and_then(|t| Variant::from_marker(&t))
            .unwrap_or_else(|| Variant::from_build_info(build_info))
    }
}

/// The file of the bundle for a system and a build, or `None` when there is none.
pub fn asset_name(os: &str, arch: &str, variant: Variant) -> Option<String> {
    let (system, ext, suffix) = match (os, arch) {
        ("linux", "x86_64") => ("linux-x86_64", "tar.gz", [("", Variant::Base), ("-cuda", Variant::Cuda), ("-rocm", Variant::Amd)]),
        ("windows", "x86_64") => ("windows-x86_64", "zip", [("", Variant::Base), ("-cuda", Variant::Cuda), ("-hip", Variant::Amd)]),
        // one build only
        ("macos", "aarch64") => return (variant == Variant::Base).then(|| "fllm-desktop-macos-aarch64.tar.gz".to_string()),
        _ => return None,
    };
    let suffix = suffix.iter().find(|(_, v)| *v == variant)?.0;
    Some(format!("fllm-desktop-{system}{suffix}.{ext}"))
}

/// What is to be fetched.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    pub name: String,
    pub url: String,
    pub sum_url: String,
    pub size: u64,
}

/// The bundle of `release` for this system and build, with its checksum file.
pub fn plan(release: &crate::updates::Release, os: &str, arch: &str, variant: Variant) -> Result<Plan, String> {
    let Some(name) = asset_name(os, arch, variant) else {
        return Err(format!("There is no ready-made {} bundle of the app for this system. Open the download page.", variant.label()));
    };
    let Some(file) = release.asset(&name) else {
        let again = if variant == Variant::Base { String::new() } else { format!(" Your {} app is not replaced by a different build.", variant.label()) };
        return Err(format!("Version {} has no {} bundle of the app ({name}).{again} Open the download page.", release.tag.trim_start_matches('v'), variant.label()));
    };
    let Some(sum) = release.asset(&format!("{name}.sha256")) else {
        return Err(format!("Version {} has no checksum for {name}, so the download could not be checked. Open the download page.", release.tag.trim_start_matches('v')));
    };
    Ok(Plan { name, url: file.url.clone(), sum_url: sum.url.clone(), size: file.size })
}

/// Where in the fetching things are.
#[derive(Debug, Clone, PartialEq)]
pub enum Phase {
    Download { written: u64, total: u64 },
    Verify,
    Unpack,
}

/// The first word of a `.sha256` file: 64 hexadecimal digits.
pub fn parse_sum(text: &str) -> Option<String> {
    let word = text.split_whitespace().next()?;
    (word.len() == 64 && word.bytes().all(|b| b.is_ascii_hexdigit())).then(|| word.to_ascii_lowercase())
}

/// Fetches, checks and unpacks `plan` into `root/<version>`; the folder holds a file `.complete` when it is whole.
pub async fn stage(plan: &Plan, root: &Path, version: &str, mut progress: impl FnMut(Phase)) -> Result<PathBuf, String> {
    let dest = root.join(version);
    let archive = root.join(&plan.name);
    tokio::fs::create_dir_all(root).await.map_err(|e| format!("The folder for the update could not be made: {e}"))?;
    let client = reqwest::Client::builder().user_agent(concat!("fllm-app/", env!("CARGO_PKG_VERSION"))).connect_timeout(std::time::Duration::from_secs(20)).build().map_err(|e| e.to_string())?;

    let want = {
        let resp = client.get(&plan.sum_url).send().await.and_then(|r| r.error_for_status()).map_err(|e| format!("The checksum could not be fetched: {e}"))?;
        parse_sum(&resp.text().await.map_err(|e| e.to_string())?).ok_or("The checksum file of the release is not readable.")?
    };

    let resp = client.get(&plan.url).send().await.and_then(|r| r.error_for_status()).map_err(|e| format!("The download failed: {e}"))?;
    let total = resp.content_length().unwrap_or(plan.size);
    let mut file = tokio::fs::File::create(&archive).await.map_err(|e| format!("The download could not be saved: {e}"))?;
    let mut hash = Sha256::new();
    let mut written = 0u64;
    let mut last = 0u64;
    let mut body = resp.bytes_stream();
    use tokio::io::AsyncWriteExt;
    while let Some(chunk) = body.next().await {
        let chunk = chunk.map_err(|e| format!("The download broke off: {e}"))?;
        hash.update(&chunk);
        file.write_all(&chunk).await.map_err(|e| format!("The download could not be saved: {e}"))?;
        written += chunk.len() as u64;
        // about a hundred steps, not one for every piece
        if written - last >= (total / 100).max(65_536) {
            last = written;
            progress(Phase::Download { written, total });
        }
    }
    file.flush().await.map_err(|e| e.to_string())?;
    drop(file);
    progress(Phase::Download { written, total: total.max(written) });

    progress(Phase::Verify);
    let got = format!("{:x}", hash.finalize());
    if got != want {
        let _ = tokio::fs::remove_file(&archive).await;
        return Err("The download does not match the checksum of the release, so it was thrown away. Try again.".into());
    }

    progress(Phase::Unpack);
    let (a, d) = (archive.clone(), dest.clone());
    let unpacked = tokio::task::spawn_blocking(move || {
        if d.exists() {
            std::fs::remove_dir_all(&d).map_err(|e| e.to_string())?;
        }
        std::fs::create_dir_all(&d).map_err(|e| e.to_string())?;
        unpack(&a, &d)?;
        std::fs::write(d.join(".complete"), b"").map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?;
    let _ = tokio::fs::remove_file(&archive).await;
    unpacked.map_err(|e| {
        let _ = std::fs::remove_dir_all(&dest);
        format!("The download could not be unpacked: {e}")
    })?;
    Ok(dest)
}

/// Unpacks a `.zip` or a `.tar.gz` into `dest`; a name that leads out of `dest` is refused.
pub fn unpack(archive: &Path, dest: &Path) -> Result<(), String> {
    let name = archive.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if name.ends_with(".zip") {
        let mut zip = zip::ZipArchive::new(std::fs::File::open(archive).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        for i in 0..zip.len() {
            let mut entry = zip.by_index(i).map_err(|e| e.to_string())?;
            let Some(rel) = entry.enclosed_name() else { return Err(format!("the archive has a file outside its folder: {}", entry.name().map(|n| n.to_string()).unwrap_or_default())) };
            let out = dest.join(rel);
            if entry.is_dir() {
                std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
                continue;
            }
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            let mut file = std::fs::File::create(&out).map_err(|e| e.to_string())?;
            std::io::copy(&mut entry, &mut file).map_err(|e| e.to_string())?;
            #[cfg(unix)]
            if let Some(mode) = entry.unix_mode() {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&out, std::fs::Permissions::from_mode(mode)).map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    } else if name.ends_with(".tar.gz") {
        let gz = flate2::read::GzDecoder::new(std::fs::File::open(archive).map_err(|e| e.to_string())?);
        let mut tar = tar::Archive::new(gz);
        tar.set_preserve_permissions(true);
        for entry in tar.entries().map_err(|e| e.to_string())? {
            let mut entry = entry.map_err(|e| e.to_string())?;
            // `unpack_in` leaves out what would land outside `dest` and says so with `false`
            if !entry.unpack_in(dest).map_err(|e| e.to_string())? {
                return Err(format!("the archive has a file outside its folder: {}", entry.path().map(|p| p.display().to_string()).unwrap_or_default()));
            }
        }
        Ok(())
    } else {
        Err(format!("{name} is neither a .zip nor a .tar.gz"))
    }
}

/// The ending of a file that was replaced and waits to be deleted.
pub const OLD: &str = ".fllm-old";

/// Puts the files of `stage` over those of `install`. A file in use cannot be written but can be
/// renamed, so every file that is there is renamed to `<name>.fllm-old` first and the new one is
/// copied in its place; when something fails everything is put back.
pub fn apply(stage: &Path, install: &Path, exe_name: &str) -> Result<usize, String> {
    if !stage.join(".complete").is_file() {
        return Err("The new version is not whole (it was not unpacked to the end).".into());
    }
    if !stage.join(exe_name).is_file() {
        return Err(format!("The new version has no {exe_name}."));
    }
    // the folder must be one that may be changed
    let probe = install.join(".fllm-write-test");
    std::fs::write(&probe, b"").map_err(|e| format!("fLLM is installed where it cannot be changed ({}): {e}. Download the new version from its page.", install.display()))?;
    let _ = std::fs::remove_file(&probe);
    let mut files = Vec::new();
    collect(stage, stage, &mut files)?;
    let mut moved: Vec<(PathBuf, PathBuf)> = Vec::new();
    let mut made: Vec<PathBuf> = Vec::new();
    let result = (|| -> Result<(), String> {
        for rel in &files {
            let target = install.join(rel);
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            if target.exists() {
                let old = PathBuf::from(format!("{}{OLD}", target.display()));
                let _ = std::fs::remove_file(&old);
                std::fs::rename(&target, &old).map_err(|e| format!("{} could not be moved aside: {e}", rel.display()))?;
                moved.push((target.clone(), old));
            }
            std::fs::copy(stage.join(rel), &target).map_err(|e| format!("{} could not be written: {e}", rel.display()))?;
            made.push(target);
        }
        Ok(())
    })();
    if let Err(e) = result {
        for t in &made {
            let _ = std::fs::remove_file(t);
        }
        for (target, old) in moved.iter().rev() {
            let _ = std::fs::remove_file(target);
            let _ = std::fs::rename(old, target);
        }
        return Err(format!("The update was not installed, nothing was changed: {e}"));
    }
    let _ = std::fs::remove_dir_all(stage);
    Ok(files.len())
}

fn collect(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in std::fs::read_dir(dir).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.is_dir() {
            collect(root, &path, out)?;
        } else if path.file_name().is_some_and(|n| n != ".complete") {
            out.push(path.strip_prefix(root).map_err(|e| e.to_string())?.to_path_buf());
        }
    }
    Ok(())
}

/// At the start: the files an earlier update moved aside are deleted, and so are the downloads of
/// versions that are not newer than this one.
pub fn clean_up(install: &Path, stage_root: &Path, current: &str) {
    fn old_files(dir: &Path, depth: u32) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() && depth < 4 {
                old_files(&p, depth + 1);
            } else if p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.ends_with(OLD)) {
                let _ = std::fs::remove_file(&p);
            }
        }
    }
    old_files(install, 0);
    let Ok(entries) = std::fs::read_dir(stage_root) else { return };
    let now = crate::updates::parse_version(current);
    for e in entries.flatten() {
        let p = e.path();
        let name = e.file_name().to_string_lossy().to_string();
        let keep = p.is_dir() && crate::updates::parse_version(&name).zip(now).is_some_and(|(v, c)| v > c);
        if !keep {
            let _ = if p.is_dir() { std::fs::remove_dir_all(&p) } else { std::fs::remove_file(&p) };
        }
    }
}

/// The unpacked version of a release that is already waiting, if it is whole.
pub fn waiting(stage_root: &Path, version: &str) -> Option<PathBuf> {
    let dir = stage_root.join(version);
    dir.join(".complete").is_file().then_some(dir)
}

/// The name of the program file as it is on disk (a running program that was replaced is named `... (deleted)` by Linux).
pub fn exe_name(current: &Path) -> String {
    let name = current.file_name().and_then(|n| n.to_str()).unwrap_or("fllm-app");
    name.trim_end_matches(" (deleted)").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::updates::{Asset, Release};

    fn release(names: &[&str]) -> Release {
        Release {
            tag: "v0.3.0".into(),
            url: "page".into(),
            version: (0, 3, 0),
            assets: names.iter().map(|n| Asset { name: n.to_string(), url: format!("https://x/{n}"), size: 10 }).collect(),
        }
    }

    #[test]
    fn the_bundle_is_the_one_of_the_same_build() {
        let name = |os, arch, v| asset_name(os, arch, v);
        assert_eq!(name("windows", "x86_64", Variant::Base).unwrap(), "fllm-desktop-windows-x86_64.zip");
        assert_eq!(name("windows", "x86_64", Variant::Cuda).unwrap(), "fllm-desktop-windows-x86_64-cuda.zip");
        assert_eq!(name("windows", "x86_64", Variant::Amd).unwrap(), "fllm-desktop-windows-x86_64-hip.zip");
        assert_eq!(name("linux", "x86_64", Variant::Cuda).unwrap(), "fllm-desktop-linux-x86_64-cuda.tar.gz");
        assert_eq!(name("linux", "x86_64", Variant::Amd).unwrap(), "fllm-desktop-linux-x86_64-rocm.tar.gz");
        assert_eq!(name("linux", "x86_64", Variant::Base).unwrap(), "fllm-desktop-linux-x86_64.tar.gz");
        assert_eq!(name("macos", "aarch64", Variant::Base).unwrap(), "fllm-desktop-macos-aarch64.tar.gz");
        assert_eq!(name("macos", "aarch64", Variant::Cuda), None, "no CUDA on a Mac");
        assert_eq!(name("macos", "x86_64", Variant::Base), None);
        assert_eq!(name("linux", "aarch64", Variant::Base), None);
    }

    #[test]
    fn a_cuda_app_never_gets_the_standard_bundle() {
        let only_standard = release(&["fllm-desktop-windows-x86_64.zip", "fllm-desktop-windows-x86_64.zip.sha256"]);
        let err = plan(&only_standard, "windows", "x86_64", Variant::Cuda).unwrap_err();
        assert!(err.contains("has no CUDA bundle") && err.contains("not replaced by a different build"), "{err}");
        let both = release(&["fllm-desktop-windows-x86_64.zip", "fllm-desktop-windows-x86_64.zip.sha256", "fllm-desktop-windows-x86_64-cuda.zip", "fllm-desktop-windows-x86_64-cuda.zip.sha256"]);
        assert_eq!(plan(&both, "windows", "x86_64", Variant::Cuda).unwrap().name, "fllm-desktop-windows-x86_64-cuda.zip");
        assert_eq!(plan(&both, "windows", "x86_64", Variant::Base).unwrap().name, "fllm-desktop-windows-x86_64.zip", "and a standard app never the CUDA one");
        let no_sum = release(&["fllm-desktop-windows-x86_64.zip"]);
        assert!(plan(&no_sum, "windows", "x86_64", Variant::Base).unwrap_err().contains("no checksum"));
        assert!(plan(&both, "freebsd", "x86_64", Variant::Base).unwrap_err().contains("no ready-made"));
    }

    #[test]
    fn the_build_is_read_from_the_marker_or_from_the_engine() {
        assert_eq!(Variant::from_build_info("fllm-rs 0.2.2 (git abc, 2026-11-01) features: avx512 backends: cpu,vulkan,cuda"), Variant::Cuda);
        assert_eq!(Variant::from_build_info("fllm-rs 0.2.2 (git abc, 2026-11-01) features: none backends: cpu,vulkan,hip"), Variant::Amd);
        assert_eq!(Variant::from_build_info("fllm-rs 0.2.2 (git abc, 2026-11-01) features: none backends: cpu,vulkan"), Variant::Base);
        assert_eq!(Variant::from_build_info(""), Variant::Base);
        // a feature that only has the letters in it is not the backend
        assert_eq!(Variant::from_build_info("fllm-rs 0.2.2 (git cuda1234, 2026-11-01) features: cudarc backends: cpu"), Variant::Base);
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(Variant::detect(Some(dir.path()), "backends: cpu,cuda"), Variant::Cuda, "no marker: the engine decides");
        std::fs::write(dir.path().join("variant.txt"), "amd\n").unwrap();
        assert_eq!(Variant::detect(Some(dir.path()), "backends: cpu,cuda"), Variant::Amd, "the marker of the bundle decides first");
        std::fs::write(dir.path().join("variant.txt"), "nonsense").unwrap();
        assert_eq!(Variant::detect(Some(dir.path()), "backends: cpu"), Variant::Base, "an unreadable marker is ignored");
    }

    #[test]
    fn a_checksum_file_is_read_like_sha256sum_writes_it() {
        let sum = "A".repeat(64);
        assert_eq!(parse_sum(&format!("{sum}  fllm-desktop.zip\n")), Some("a".repeat(64)));
        assert_eq!(parse_sum("short  file"), None);
        assert_eq!(parse_sum(""), None);
        assert_eq!(parse_sum(&format!("{}g", "a".repeat(63))), None);
    }

    fn tar_gz(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut out = Vec::new();
        {
            let gz = flate2::write::GzEncoder::new(&mut out, flate2::Compression::fast());
            let mut tar = tar::Builder::new(gz);
            for (name, data) in files {
                let mut header = tar::Header::new_gnu();
                header.set_size(data.len() as u64);
                header.set_mode(0o755);
                header.set_cksum();
                tar.append_data(&mut header, name, *data).unwrap();
            }
            tar.into_inner().unwrap().finish().unwrap();
        }
        out
    }

    fn zip_of(files: &[(&str, &[u8])]) -> Vec<u8> {
        use std::io::Write;
        let mut out = std::io::Cursor::new(Vec::new());
        {
            let mut zip = zip::ZipWriter::new(&mut out);
            let options = zip::write::SimpleFileOptions::default().unix_permissions(0o755);
            for (name, data) in files {
                zip.start_file(*name, options).unwrap();
                zip.write_all(data).unwrap();
            }
            zip.finish().unwrap();
        }
        out.into_inner()
    }

    #[test]
    fn both_kinds_of_archive_are_unpacked_with_their_folders() {
        let dir = tempfile::tempdir().unwrap();
        let files: [(&str, &[u8]); 3] = [("fllm-app", b"app"), ("fllm-rs", b"engine"), ("sub/lib.dll", b"dll")];
        std::fs::write(dir.path().join("a.tar.gz"), tar_gz(&files)).unwrap();
        std::fs::write(dir.path().join("a.zip"), zip_of(&files)).unwrap();
        for (archive, out) in [("a.tar.gz", "t"), ("a.zip", "z")] {
            let dest = dir.path().join(out);
            std::fs::create_dir_all(&dest).unwrap();
            unpack(&dir.path().join(archive), &dest).unwrap();
            assert_eq!(std::fs::read(dest.join("fllm-rs")).unwrap(), b"engine", "{archive}");
            assert_eq!(std::fs::read(dest.join("sub/lib.dll")).unwrap(), b"dll", "{archive}");
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                assert_eq!(std::fs::metadata(dest.join("fllm-app")).unwrap().permissions().mode() & 0o111, 0o111, "{archive}: still runnable");
            }
        }
        std::fs::write(dir.path().join("a.rar"), b"x").unwrap();
        assert!(unpack(&dir.path().join("a.rar"), dir.path()).unwrap_err().contains("neither"));
    }

    #[test]
    fn a_zip_that_leads_out_of_its_folder_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("evil.zip"), zip_of(&[("../outside.txt", b"x")])).unwrap();
        let dest = dir.path().join("d");
        std::fs::create_dir_all(&dest).unwrap();
        assert!(unpack(&dir.path().join("evil.zip"), &dest).unwrap_err().contains("outside"));
        assert!(!dir.path().join("outside.txt").exists());
    }

    fn staged(files: &[(&str, &str)], complete: bool) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for (name, data) in files {
            let p = dir.path().join(name);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, data).unwrap();
        }
        if complete {
            std::fs::write(dir.path().join(".complete"), b"").unwrap();
        }
        dir
    }

    #[test]
    fn the_new_files_replace_the_old_ones_and_the_old_ones_wait_to_be_deleted() {
        let install = staged(&[("fllm-app", "old app"), ("fllm-rs", "old engine"), ("keep.txt", "mine")], false);
        let stage = staged(&[("fllm-app", "new app"), ("fllm-rs", "new engine"), ("sub/new.dll", "dll")], true);
        let n = apply(stage.path(), install.path(), "fllm-app").unwrap();
        assert_eq!(n, 3);
        let read = |p: &str| std::fs::read_to_string(install.path().join(p)).unwrap();
        assert_eq!((read("fllm-app").as_str(), read("fllm-rs").as_str(), read("sub/new.dll").as_str(), read("keep.txt").as_str()), ("new app", "new engine", "dll", "mine"));
        assert_eq!(read("fllm-app.fllm-old"), "old app", "the old one is still there, as a file that can be deleted");
        assert!(!stage.path().exists(), "the unpacked copy is gone");
        clean_up(install.path(), &install.path().join("none"), "0.2.2");
        assert!(!install.path().join("fllm-app.fllm-old").exists() && !install.path().join("fllm-rs.fllm-old").exists(), "deleted at the next start");
        assert_eq!(read("fllm-app"), "new app");
    }

    #[test]
    fn an_incomplete_or_wrong_stage_changes_nothing() {
        let install = staged(&[("fllm-app", "old")], false);
        let half = staged(&[("fllm-app", "new")], false);
        assert!(apply(half.path(), install.path(), "fllm-app").unwrap_err().contains("not whole"));
        let wrong = staged(&[("other", "new")], true);
        assert!(apply(wrong.path(), install.path(), "fllm-app").unwrap_err().contains("has no fllm-app"));
        assert_eq!(std::fs::read_to_string(install.path().join("fllm-app")).unwrap(), "old");
        assert!(half.path().exists(), "what was not installed is not deleted");
    }

    #[cfg(unix)]
    #[test]
    fn a_failure_in_the_middle_puts_everything_back() {
        use std::os::unix::fs::PermissionsExt;
        let install = staged(&[("fllm-app", "old app"), ("fllm-rs", "old engine")], false);
        let stage = staged(&[("fllm-app", "new app"), ("fllm-rs", "new engine"), ("zlocked/x.txt", "x")], true);
        // a folder of the install that exists and cannot be written to: the last file fails
        std::fs::create_dir_all(install.path().join("zlocked")).unwrap();
        std::fs::set_permissions(install.path().join("zlocked"), std::fs::Permissions::from_mode(0o500)).unwrap();
        let root = std::process::Command::new("id").arg("-u").output().map(|o| String::from_utf8_lossy(&o.stdout).trim() == "0").unwrap_or(false);
        let result = apply(stage.path(), install.path(), "fllm-app");
        std::fs::set_permissions(install.path().join("zlocked"), std::fs::Permissions::from_mode(0o700)).unwrap();
        if root {
            return; // root writes anywhere: nothing to see
        }
        assert!(result.unwrap_err().contains("nothing was changed"));
        assert_eq!(std::fs::read_to_string(install.path().join("fllm-app")).unwrap(), "old app");
        assert_eq!(std::fs::read_to_string(install.path().join("fllm-rs")).unwrap(), "old engine");
        assert!(!install.path().join("fllm-app.fllm-old").exists());
    }

    #[test]
    fn the_start_deletes_downloads_that_are_not_newer() {
        let root = tempfile::tempdir().unwrap();
        for v in ["0.2.1", "0.2.2", "0.3.0"] {
            std::fs::create_dir_all(root.path().join(v)).unwrap();
        }
        std::fs::write(root.path().join("fllm-desktop-linux-x86_64.tar.gz"), b"half").unwrap();
        let install = tempfile::tempdir().unwrap();
        clean_up(install.path(), root.path(), "0.2.2");
        assert!(!root.path().join("0.2.1").exists() && !root.path().join("0.2.2").exists());
        assert!(root.path().join("0.3.0").exists(), "a newer one that waits stays");
        assert!(!root.path().join("fllm-desktop-linux-x86_64.tar.gz").exists(), "a cut-off download goes");
        assert!(waiting(root.path(), "0.3.0").is_none(), "and it only counts when it is whole");
        std::fs::write(root.path().join("0.3.0/.complete"), b"").unwrap();
        assert!(waiting(root.path(), "0.3.0").is_some());
        assert_eq!(exe_name(Path::new("/opt/fllm/fllm-app (deleted)")), "fllm-app");
    }
}
