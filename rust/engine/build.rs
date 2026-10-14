//! Embed build provenance; no runtime Git installation is required.
use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

fn git(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}
fn provenance() {
    println!("cargo:rerun-if-env-changed=FLLM_GIT_COMMIT");
    let supplied = env::var("FLLM_GIT_COMMIT").ok();
    let commit = supplied
        .or_else(|| git(&["rev-parse", "HEAD"]))
        .unwrap_or_else(|| "unknown".into());
    let commit = if commit.len() >= 7 && commit.bytes().all(|b| b.is_ascii_hexdigit()) {
        commit
    } else {
        "unknown".into()
    };
    println!("cargo:rustc-env=FLLM_GIT_COMMIT={commit}");
    let dirty =
        git(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty());
    println!("cargo:rustc-env=FLLM_GIT_DIRTY={dirty}");
    for name in ["HEAD", "refs/heads/main", "packed-refs", "index"] {
        if let Some(path) = git(&["rev-parse", "--git-path", name]) {
            let path = PathBuf::from(path);
            let path = if path.is_absolute() {
                path
            } else {
                PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap_or_default()).join(path)
            };
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=build.rs");
}

/// What differs between the platforms the GPU kernels are built for. The build script runs on the host, the kernels are built for the
/// cargo target (`CARGO_CFG_TARGET_*`); CUDA and ROCm build on the host they run on, so the two are the same machine in practice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Target {
    windows: bool,
    /// `x86_64-pc-windows-msvc` and friends: `cl.exe` as host compiler, `.obj` objects, `lib.exe` archives.
    msvc: bool,
    /// `-C target-feature=+crt-static`: the kernels must use the static C runtime too (`/MT`), or the link fails with a CRT mismatch.
    crt_static: bool,
}

impl Target {
    fn from_env() -> Self {
        Self::from_values(
            &env::var("CARGO_CFG_TARGET_OS").unwrap_or_default(),
            &env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default(),
            &env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default(),
        )
    }

    fn from_values(os: &str, target_env: &str, features: &str) -> Self {
        Target { windows: os == "windows", msvc: target_env == "msvc", crt_static: features.split(',').any(|f| f == "crt-static") }
    }

    /// File name of the object compiled from `source` (`api.cu` -> `api.o` / `api.obj`).
    fn object_name(&self, source: &str) -> String {
        format!("{}.{}", source.strip_suffix(".cu").unwrap_or(source), if self.msvc { "obj" } else { "o" })
    }

    /// File name of the static library `name` (`libname.a` / `name.lib`).
    fn library_name(&self, name: &str) -> String {
        if self.msvc {
            format!("{name}.lib")
        } else {
            format!("lib{name}.a")
        }
    }

    /// Options of hipcc that depend on the platform: `-fPIC` (an error for the Windows targets), or the C runtime rustc links.
    fn hipcc_platform_flags(&self) -> Vec<&'static str> {
        if self.msvc {
            vec![if self.crt_static { "-fms-runtime-lib=static" } else { "-fms-runtime-lib=dll" }]
        } else if self.windows {
            Vec::new()
        } else {
            vec!["-fPIC"]
        }
    }

    /// The host-compiler options nvcc passes on with `-Xcompiler`: position-independent code elsewhere; with MSVC the C runtime matching
    /// the one rustc links (`/MD`, or `/MT` for a `crt-static` build), exception handling, and big object files (the Marlin instances).
    fn nvcc_host_flags(&self) -> String {
        if self.msvc {
            format!("{},/EHsc,/bigobj,/Zc:__cplusplus", if self.crt_static { "/MT" } else { "/MD" })
        } else {
            "-fPIC".to_owned()
        }
    }
}

/// Name of a host tool: `nvcc` / `nvcc.exe`. (The build script's own platform decides, not the target's.)
fn host_exe(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_owned()
    }
}

/// First `name` found on `PATH`.
fn which(name: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    env::split_paths(&path).map(|dir| dir.join(name)).find(|candidate| candidate.is_file())
}

/// Sort key of a version directory such as `v12.8` or `14.40.33807`: its numbers, so that `v12.10` follows `v12.9`.
fn version_key(name: &str) -> Vec<u32> {
    name.split(|c: char| !c.is_ascii_digit()).filter(|part| !part.is_empty()).filter_map(|part| part.parse().ok()).collect()
}

/// The sub-directory of `parent` with the highest version number whose name starts with `prefix`.
fn newest_version_dir(parent: &Path, prefix: &str) -> Option<PathBuf> {
    let mut dirs: Vec<(Vec<u32>, PathBuf)> = std::fs::read_dir(parent)
        .ok()?
        .flatten()
        .filter(|entry| entry.file_name().to_string_lossy().starts_with(prefix) && entry.path().is_dir())
        .map(|entry| (version_key(&entry.file_name().to_string_lossy()), entry.path()))
        .collect();
    dirs.sort();
    dirs.pop().map(|(_, path)| path)
}

/// The CUDA toolkit: `CUDA_HOME`, `CUDA_PATH` (set by the Windows installer), the newest `NVIDIA GPU Computing Toolkit\CUDA\vX.Y` on Windows,
/// else `/usr/local/cuda`.
fn cuda_home(target: &Target) -> PathBuf {
    if let Some(dir) = env::var_os("CUDA_HOME").or_else(|| env::var_os("CUDA_PATH")) {
        return PathBuf::from(dir);
    }
    if target.windows {
        let program_files = env::var_os("ProgramFiles").map_or_else(|| PathBuf::from(r"C:\Program Files"), PathBuf::from);
        if let Some(dir) = newest_version_dir(&program_files.join("NVIDIA GPU Computing Toolkit").join("CUDA"), "v") {
            return dir;
        }
    }
    PathBuf::from("/usr/local/cuda")
}

/// The ROCm / HIP install: `ROCM_PATH`, `HIP_PATH` (set by the Windows HIP SDK installer), the newest `Program Files\AMD\ROCm\X.Y` on
/// Windows, else `/opt/rocm`.
fn rocm_home(target: &Target) -> PathBuf {
    if let Some(dir) = env::var_os("ROCM_PATH").or_else(|| env::var_os("HIP_PATH")) {
        return PathBuf::from(dir);
    }
    if target.windows {
        let program_files = env::var_os("ProgramFiles").map_or_else(|| PathBuf::from(r"C:\Program Files"), PathBuf::from);
        if let Some(dir) = newest_version_dir(&program_files.join("AMD").join("ROCm"), "") {
            return dir;
        }
    }
    PathBuf::from("/opt/rocm")
}

/// Directory of the MSVC build tools (`cl.exe`, `lib.exe`): `FLLM_MSVC_BIN`, else the directory of `cl.exe` on `PATH` (a Developer Command
/// Prompt), else the newest toolset of the Visual Studio that `vswhere` reports.
fn msvc_bin_dir() -> Option<PathBuf> {
    if let Some(dir) = env::var_os("FLLM_MSVC_BIN") {
        return Some(PathBuf::from(dir));
    }
    if let Some(cl) = which("cl.exe") {
        return cl.parent().map(Path::to_path_buf);
    }
    let program_files = env::var_os("ProgramFiles(x86)").or_else(|| env::var_os("ProgramFiles"))?;
    let vswhere = Path::new(&program_files).join("Microsoft Visual Studio").join("Installer").join("vswhere.exe");
    let output = Command::new(vswhere)
        .args(["-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"])
        .output()
        .ok()?;
    let installation = String::from_utf8_lossy(&output.stdout).lines().next()?.trim().to_owned();
    if installation.is_empty() {
        return None;
    }
    let toolset = newest_version_dir(&Path::new(&installation).join("VC").join("Tools").join("MSVC"), "")?;
    let (host, target) = match (env::consts::ARCH, env::var("CARGO_CFG_TARGET_ARCH").as_deref()) {
        ("aarch64", Ok("aarch64")) => ("Hostarm64", "arm64"),
        (_, Ok("aarch64")) => ("Hostx64", "arm64"),
        ("aarch64", _) => ("Hostarm64", "x64"),
        _ => ("Hostx64", "x64"),
    };
    Some(toolset.join("bin").join(host).join(target))
}

/// `lib.exe` (or the file named by `FLLM_LIB_EXE`).
fn find_lib_exe() -> PathBuf {
    if let Some(path) = env::var_os("FLLM_LIB_EXE") {
        return PathBuf::from(path);
    }
    which("lib.exe")
        .or_else(|| msvc_bin_dir().map(|dir| dir.join("lib.exe")).filter(|p| p.is_file()))
        .unwrap_or_else(|| panic!("lib.exe not found: build from a Visual Studio Developer Command Prompt, or set FLLM_MSVC_BIN (the directory of cl.exe and lib.exe) or FLLM_LIB_EXE"))
}

/// Arguments that point nvcc at the MSVC host compiler when `cl.exe` is not on `PATH` (`FLLM_CUDA_CCBIN` overrides; empty elsewhere).
fn nvcc_ccbin(target: &Target) -> Vec<String> {
    if let Some(dir) = env::var_os("FLLM_CUDA_CCBIN") {
        return vec!["-ccbin".into(), dir.to_string_lossy().into_owned()];
    }
    if target.msvc && which("cl.exe").is_none() {
        if let Some(dir) = msvc_bin_dir() {
            return vec!["-ccbin".into(), dir.to_string_lossy().into_owned()];
        }
    }
    Vec::new()
}

/// The text of the response file for `lib.exe`: the output and one quoted object per line (a long command line would hit the 32 k limit).
fn lib_response_file(library: &Path, objects: &[PathBuf]) -> String {
    let mut text = format!("/OUT:\"{}\"\n", library.display());
    for object in objects {
        text.push_str(&format!("\"{}\"\n", object.display()));
    }
    text
}

/// Pack `objects` into the static library `library`: `ar crs` on Unix, `lib.exe` for the MSVC target.
fn archive(target: &Target, library: &Path, objects: &[PathBuf]) {
    let _ = std::fs::remove_file(library);
    if target.msvc {
        let lib = find_lib_exe();
        let response = library.with_extension("rsp");
        std::fs::write(&response, lib_response_file(library, objects)).expect("write the lib.exe response file");
        let status = Command::new(&lib)
            .arg("/NOLOGO")
            .arg(format!("@{}", response.display()))
            .status()
            .unwrap_or_else(|e| panic!("cannot run {}: {e}", lib.display()));
        assert!(status.success(), "lib.exe failed");
    } else {
        let status = Command::new("ar").arg("crs").arg(library).args(objects).status().expect("ar");
        assert!(status.success(), "ar failed");
    }
}

/// Builds the CUDA kernels into a static library when the `cuda` feature is on.
///
/// `FLLM_CUDA_ARCHS` selects the GPU architectures (default: Ampere, Ada, Hopper, Blackwell);
/// e.g. `FLLM_CUDA_ARCHS=120` for a fast development build. `CUDA_HOME` / `CUDA_PATH` point at
/// the toolkit (default `/usr/local/cuda`, on Windows the newest `Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y`).
/// The CUDA runtime is linked statically, so the result needs only the NVIDIA driver at run time.
///
/// Windows (target `x86_64-pc-windows-msvc`): nvcc needs the MSVC host compiler. Build from a Developer Command Prompt, or let the
/// script find it (`FLLM_MSVC_BIN` = directory of `cl.exe`, or the Visual Studio that `vswhere` reports; `FLLM_CUDA_CCBIN` hands
/// nvcc another `-ccbin`). The objects are `.obj`, packed with `lib.exe` into `fllmcuda.lib`; the C runtime is `/MD`, or `/MT` when the
/// crate is built with `-C target-feature=+crt-static`. A Visual Studio newer than the toolkit supports needs
/// `FLLM_CUDA_EXTRA_FLAGS=-allow-unsupported-compiler`.
///
/// `FLLM_CUDA_EXTRA_FLAGS` adds space separated flags to every nvcc call, e.g.
/// `-DQG_VARIANT=4` to build the Q4_K GEMM variant 4 (docs/cuda-qgemm4k.md).
fn cuda_kernels() {
    println!("cargo:rerun-if-changed=cuda");
    println!("cargo:rerun-if-env-changed=FLLM_CUDA_ARCHS");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=FLLM_CUDA_EXTRA_FLAGS");
    println!("cargo:rerun-if-env-changed=FLLM_CUDA_CCBIN");
    println!("cargo:rerun-if-env-changed=FLLM_MSVC_BIN");
    println!("cargo:rerun-if-env-changed=FLLM_LIB_EXE");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let target = Target::from_env();
    if target.windows && !target.msvc {
        panic!("feature cuda on Windows needs the MSVC target (x86_64-pc-windows-msvc) and the CUDA Toolkit: nvcc cannot use the MinGW compiler");
    }
    let cuda_home = cuda_home(&target);
    let nvcc = cuda_home.join("bin").join(host_exe("nvcc"));
    let ccbin = nvcc_ccbin(&target);
    let host_flags = target.nvcc_host_flags();
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let archs = std::env::var("FLLM_CUDA_ARCHS").unwrap_or_else(|_| "61;70;75;80;86;89;90;120".into());
    let archs: Vec<&str> = archs.split(';').map(str::trim).filter(|a| !a.is_empty()).collect();
    let mut flags: Vec<String> = Vec::new();
    for arch in &archs {
        flags.push("-gencode".into());
        flags.push(format!("arch=compute_{arch},code=sm_{arch}"));
    }
    // PTX of the newest architecture keeps the binary usable on GPUs released later.
    if let Some(newest) = archs.last() {
        flags.push("-gencode".into());
        flags.push(format!("arch=compute_{newest},code=compute_{newest}"));
    }
    // Optional: CUTLASS headers (BSD-3-Clause) enable the large-batch NVFP4 GEMM for Blackwell GeForce. Found through
    // FLLM_CUTLASS_DIR or third_party/cutlass; without them the built-in NVFP4 kernel is used for every batch size.
    println!("cargo:rerun-if-env-changed=FLLM_CUTLASS_DIR");
    let cutlass = std::env::var_os("FLLM_CUTLASS_DIR")
        .map(PathBuf::from)
        .or_else(|| Some(PathBuf::from("../../third_party/cutlass")))
        .filter(|dir| dir.join("include/cutlass/cutlass.h").exists() && archs.contains(&"120"));
    if cutlass.is_none() && archs.contains(&"120") {
        println!("cargo:warning=CUTLASS headers not found (third_party/cutlass or FLLM_CUTLASS_DIR): NVFP4 prompts use the built-in kernel, see THIRD_PARTY.md");
    }
    // Optional: FlashInfer headers (Apache-2.0) give the prompt attention a faster kernel on Turing and newer GPUs. Found
    // through FLLM_FLASHINFER_DIR or third_party/flashinfer.
    println!("cargo:rerun-if-env-changed=FLLM_FLASHINFER_DIR");
    let flashinfer_archs: Vec<&str> = archs.iter().copied().filter(|a| a.parse::<u32>().is_ok_and(|v| v >= 75)).collect();
    let flashinfer = std::env::var_os("FLLM_FLASHINFER_DIR")
        .map(PathBuf::from)
        .or_else(|| Some(PathBuf::from("../../third_party/flashinfer")))
        .filter(|dir| dir.join("include/flashinfer/attention/prefill.cuh").exists() && !flashinfer_archs.is_empty());
    // Extra nvcc flags, space separated, e.g. FLLM_CUDA_EXTRA_FLAGS="-DQG_VARIANT=4".
    let mut extra: Vec<String> = std::env::var("FLLM_CUDA_EXTRA_FLAGS")
        .unwrap_or_default()
        .split_whitespace()
        .map(str::to_string)
        .collect();
    // CUDA 13 gives the host stubs of __global__ function templates internal linkage by default: a kernel instantiated explicitly in
    // one file (the generated Marlin instances) is then invisible to the file that launches it and the link fails.
    let nvcc_major = Command::new(&nvcc)
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8_lossy(&o.stdout).split("release ").nth(1).map(str::to_owned))
        .and_then(|r| r.split('.').next().and_then(|m| m.trim().parse::<u32>().ok()))
        .unwrap_or(0);
    if nvcc_major >= 13 {
        extra.push("-static-global-template-stub=false".into());
    }
    let sources = ["api.cu", "linear.cu", "gemv2.cu", "elementwise.cu", "attention.cu", "attention2.cu", "tc_gemm.cu", "qgemm4k.cu", "attention3.cu", "qgemm6k.cu", "qgemm5k.cu", "qgemm8.cu", "qgemm_leg.cu", "attention4.cu", "mmvq.cu", "nvfp4.cu", "nvfp4_cutlass.cu", "flashinfer_prefill.cu", "marlin_q4k.cu", "deltanet.cu", "kv_quant.cu", "arch_ops.cu", "mla.cu", "mamba2.cu", "shortconv.cu", "mamba1.cu", "fused.cu"];
    let mut objects = Vec::new();
    for source in sources {
        if source == "nvfp4_cutlass.cu" && cutlass.is_none() {
            continue;
        }
        let object = out.join(target.object_name(source));
        // The FP4 tensor-core instructions exist only in the architecture-specific sm_120a target.
        let fp4 = source == "nvfp4.cu" || source == "nvfp4_cutlass.cu";
        let fp4_flags: Vec<String> = if archs.contains(&"120") {
            vec!["-gencode".into(), "arch=compute_120a,code=sm_120a".into()]
        } else {
            // no FP4 target: the file still has to compile for the other architectures (stub that declines every call)
            let mut f = flags.clone();
            f.push("-DFLLM_NO_FP4".into());
            f
        };
        let mut command = Command::new(&nvcc);
        let flashinfer_build = source == "flashinfer_prefill.cu" && flashinfer.is_some();
        command.args(["-c", "-O3", "--std=c++17", "-lineinfo", "-Xcompiler", &host_flags]).args(&ccbin);
        // kv_quant.cu needs IEEE division and denormals: its encoders must match the Rust ones bit for bit
        if !flashinfer_build && source != "kv_quant.cu" {
            command.arg("--use_fast_math");
        }
        if let Some(dir) = &cutlass {
            if source == "nvfp4.cu" {
                command.arg("-DFLLM_HAVE_CUTLASS");
            } else if source == "nvfp4_cutlass.cu" {
                command
                    .arg("--expt-relaxed-constexpr")
                    .arg(format!("-I{}", dir.join("include").display()))
                    .arg(format!("-I{}", dir.join("tools/util/include").display()));
            }
        }
        let mut fi_flags: Vec<String> = Vec::new();
        if flashinfer_build {
            let dir = flashinfer.as_ref().expect("checked above");
            for arch in &flashinfer_archs {
                fi_flags.push("-gencode".into());
                fi_flags.push(format!("arch=compute_{arch},code=sm_{arch}"));
            }
            if let Some(newest) = flashinfer_archs.last() {
                fi_flags.push("-gencode".into());
                fi_flags.push(format!("arch=compute_{newest},code=compute_{newest}"));
            }
            fi_flags.push("-DFLLM_HAVE_FLASHINFER".into());
            fi_flags.push("--expt-relaxed-constexpr".into());
            fi_flags.push(format!("-I{}", dir.join("include").display()));
        }
        let status = command
            .args(if flashinfer_build { &fi_flags } else if fp4 { &fp4_flags } else { &flags })
            .args(&extra)
            .arg("-o")
            .arg(&object)
            .arg(PathBuf::from("cuda").join(source))
            .status()
            .unwrap_or_else(|e| panic!("cannot run {}: {e} (set CUDA_HOME)", nvcc.display()));
        assert!(status.success(), "nvcc failed on {source}");
        objects.push(object);
    }
    // vLLM's Marlin GEMM for 4-bit AWQ / GPTQ weights (cuda/marlin_vllm): many template instantiations, so compiled in parallel,
    // for Turing and newer only (older targets get the stub branch of the kernels and are never dispatched to)
    let marlin_dir = PathBuf::from("cuda").join("marlin_vllm");
    let mut marlin_sources: Vec<PathBuf> = std::fs::read_dir(&marlin_dir)
        .expect("cuda/marlin_vllm")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "cu"))
        .collect();
    marlin_sources.sort();
    marlin_sources.push(PathBuf::from("cuda").join("marlin_vllm_glue.cu"));
    let marlin_archs: Vec<&str> = archs.iter().copied().filter(|a| a.parse::<u32>().is_ok_and(|v| v >= 75)).collect();
    let mut marlin_flags: Vec<String> = Vec::new();
    for arch in &marlin_archs {
        marlin_flags.push("-gencode".into());
        marlin_flags.push(format!("arch=compute_{arch},code=sm_{arch}"));
    }
    if let Some(newest) = marlin_archs.last() {
        marlin_flags.push("-gencode".into());
        marlin_flags.push(format!("arch=compute_{newest},code=compute_{newest}"));
    }
    if marlin_archs.is_empty() {
        marlin_flags = flags.clone();
    }
    let jobs = std::thread::available_parallelism().map_or(4, |n| n.get()).min(12);
    let queue = std::sync::Mutex::new(marlin_sources.clone());
    let marlin_objects: Vec<PathBuf> = marlin_sources
        .iter()
        .map(|s| out.join(target.object_name(&format!("marlin_{}.cu", s.file_stem().expect("file name").to_string_lossy()))))
        .collect();
    std::thread::scope(|scope| {
        for _ in 0..jobs {
            scope.spawn(|| loop {
                let Some(source) = queue.lock().expect("queue").pop() else { return };
                let object = out.join(target.object_name(&format!("marlin_{}.cu", source.file_stem().expect("file name").to_string_lossy())));
                let status = Command::new(&nvcc)
                    .args(["-c", "-O3", "--std=c++17", "-lineinfo", "--use_fast_math", "--expt-relaxed-constexpr", "-Xcompiler", &host_flags])
                    .args(&ccbin)
                    .arg(format!("-I{}", marlin_dir.display()))
                    .args(&marlin_flags)
                    .args(&extra)
                    .arg("-o")
                    .arg(&object)
                    .arg(&source)
                    .status()
                    .unwrap_or_else(|e| panic!("cannot run {}: {e} (set CUDA_HOME)", nvcc.display()));
                assert!(status.success(), "nvcc failed on {}", source.display());
            });
        }
    });
    objects.extend(marlin_objects);
    let library = out.join(target.library_name("fllmcuda"));
    archive(&target, &library, &objects);
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=fllmcuda");
    // the toolkit's library directories: Linux (x86_64, and the Arm server / Jetson layouts) and Windows (lib\x64)
    for dir in ["lib64", "lib64/stubs", "targets/x86_64-linux/lib", "targets/sbsa-linux/lib", "targets/aarch64-linux/lib", "lib/x64"] {
        let path = cuda_home.join(dir);
        if path.is_dir() || dir == "lib64" {
            println!("cargo:rustc-link-search=native={}", path.display());
        }
    }
    println!("cargo:rustc-link-lib=static=cudart_static");
    if !target.windows {
        for lib in ["dl", "rt", "pthread", "stdc++"] {
            println!("cargo:rustc-link-lib={lib}");
        }
    }
}

/// AMD GPUs: the same kernel sources compiled with hipcc (`FLLM_HIP=1`, `--features cuda`). The tensor-core kernels are
/// compiled out (cuda/hip_compat.h), the HIP runtime is linked dynamically (libamdhip64 from the ROCm install).
/// `FLLM_HIP_ARCHS` lists the offload architectures (default: RDNA2-4 consumer cards and gfx90a/gfx942), `ROCM_PATH` the install
/// (on Windows `HIP_PATH`, which the HIP SDK installer sets, or the newest `Program Files\AMD\ROCm\X.Y`; `amdhip64.lib` from its `lib`).
/// With the MSVC target the objects are `.obj`, packed with `lib.exe`, and compiled for the C runtime rustc links (`-fms-runtime-lib`).
fn hip_kernels() {
    println!("cargo:rerun-if-changed=cuda");
    println!("cargo:rerun-if-changed=hip");
    println!("cargo:rerun-if-env-changed=FLLM_HIP_ARCHS");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    let target = Target::from_env();
    let rocm = rocm_home(&target);
    let hipcc = rocm.join("bin").join(host_exe("hipcc"));
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let archs = std::env::var("FLLM_HIP_ARCHS")
        .unwrap_or_else(|_| "gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1200;gfx1201".into());
    let sources = ["api.cu", "linear.cu", "gemv2.cu", "elementwise.cu", "attention.cu", "attention2.cu", "attention4.cu", "mmvq.cu", "tc_gemm.cu", "fused.cu", "hip_stubs.cu"];
    // kernels that exist only for AMD GPUs (matrix-core GEMM, skinny GEMM, Q4Z; they include ../cuda/*.cuh)
    let hip_sources = ["hip_device.cu", "hip_gemm.cu", "hip_skinny.cu", "hip_q4z.cu"];
    let mut objects = Vec::new();
    for (dir, source) in sources.iter().map(|s| ("cuda", *s)).chain(hip_sources.iter().map(|s| ("hip", *s))) {
        let object = out.join(target.object_name(source));
        let mut command = Command::new(&hipcc);
        command.args(["-x", "hip", "-c", "-O3", "-std=c++17", "-ffast-math", "-DFLLM_NO_FP4", "-Wno-unused-value", "-Wno-unused-result"]);
        command.args(target.hipcc_platform_flags());
        for arch in archs.split(';').map(str::trim).filter(|a| !a.is_empty()) {
            command.arg(format!("--offload-arch={arch}"));
        }
        let status = command
            .arg("-o")
            .arg(&object)
            .arg(PathBuf::from(dir).join(source))
            .status()
            .unwrap_or_else(|e| panic!("cannot run {}: {e} (set ROCM_PATH)", hipcc.display()));
        assert!(status.success(), "hipcc failed on {source}");
        objects.push(object);
    }
    let library = out.join(target.library_name("fllmcuda"));
    archive(&target, &library, &objects);
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=fllmcuda");
    println!("cargo:rustc-link-search=native={}", rocm.join("lib").display());
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    if !target.windows {
        for lib in ["dl", "rt", "pthread", "stdc++"] {
            println!("cargo:rustc-link-lib={lib}");
        }
    }
}

/// The shader compiler: `GLSLC`, else `glslc` in `Bin` / `bin` of the Vulkan SDK (`VULKAN_SDK`: the Windows installer sets it, but a shell opened
/// before the installation does not have the SDK on `PATH`), else `glslc` from `PATH`.
fn glslc_command() -> String {
    if let Ok(path) = env::var("GLSLC") {
        return path;
    }
    if let Some(sdk) = env::var_os("VULKAN_SDK") {
        for dir in ["Bin", "bin"] {
            let candidate = Path::new(&sdk).join(dir).join(host_exe("glslc"));
            if candidate.is_file() {
                return candidate.to_string_lossy().into_owned();
            }
        }
    }
    "glslc".into()
}

/// Feature `vulkan`: compile `shaders/vulkan/*.comp` with glslc into SPIR-V (OUT_DIR/vulkan/NAME.spv) and write `vulkan_shaders.rs`.
///
/// Every `*.comp` is compiled once (output name = file stem) unless a `variants*.list` names it; a line of such a list is
/// `<output> <source.comp> [-DNAME=VALUE ...]`. Files starting with `_` and `*.glsl` / `*.inc` are includes. `GLSLC` overrides the compiler.
fn vulkan_shaders() {
    use std::collections::BTreeSet;
    println!("cargo:rerun-if-changed=shaders/vulkan");
    println!("cargo:rerun-if-env-changed=GLSLC");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    let dir = PathBuf::from("shaders/vulkan");
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR")).join("vulkan");
    std::fs::create_dir_all(&out).expect("create OUT_DIR/vulkan");
    let glslc = glslc_command();
    let mut jobs: Vec<(String, String, Vec<String>)> = Vec::new();
    let mut listed: BTreeSet<String> = BTreeSet::new();
    let mut names: Vec<String> = std::fs::read_dir(&dir).map(|d| d.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect()).unwrap_or_default();
    names.sort();
    for name in &names {
        if name.starts_with("variants") && name.ends_with(".list") {
            for line in std::fs::read_to_string(dir.join(name)).unwrap_or_default().lines() {
                let line = line.split('#').next().unwrap_or("").trim();
                let mut parts = line.split_whitespace();
                if let (Some(output), Some(source)) = (parts.next(), parts.next()) {
                    listed.insert(source.to_owned());
                    jobs.push((output.to_owned(), source.to_owned(), parts.map(str::to_owned).collect()));
                }
            }
        }
    }
    for name in &names {
        if name.ends_with(".comp") && !name.starts_with('_') && !listed.contains(name) {
            jobs.push((name.trim_end_matches(".comp").to_owned(), name.clone(), Vec::new()));
        }
    }
    let mut table = String::from("/// (name, SPIR-V bytes) of every compiled shader\npub static SHADERS: &[(&str, &[u8])] = &[\n");
    for (output, source, defines) in &jobs {
        let spv = out.join(format!("{output}.spv"));
        let status = Command::new(&glslc)
            .args(["--target-env=vulkan1.2", "-O", "-I"])
            .arg(&dir)
            .args(defines)
            .arg("-o")
            .arg(&spv)
            .arg(dir.join(source))
            .status()
            .unwrap_or_else(|e| panic!("cannot run {glslc}: {e} (install the Vulkan SDK / glslc, set VULKAN_SDK or GLSLC)"));
        assert!(status.success(), "glslc failed on {source} ({output})");
        table.push_str(&format!("    ({output:?}, include_bytes!({:?})),\n", spv.to_string_lossy().replace('\\', "/")));
    }
    table.push_str("];\n");
    std::fs::write(PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR")).join("vulkan_shaders.rs"), table).expect("write vulkan_shaders.rs");
}

/// Feature `metal` (any host): embed `metal/*.metal` into `metal_sources.rs` (`SOURCES: &[(&str, &str)]`, `common.metal` first).
fn metal_sources() {
    println!("cargo:rerun-if-changed=metal");
    let dir = std::fs::canonicalize("metal").unwrap_or_else(|_| PathBuf::from("metal"));
    let mut names: Vec<String> = std::fs::read_dir(&dir).map(|d| d.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect()).unwrap_or_default();
    names.retain(|n| n.ends_with(".metal"));
    names.sort_by_key(|n| (n != "common.metal", n.clone()));
    let mut table = String::from("/// (file name, MSL source)\npub static SOURCES: &[(&str, &str)] = &[\n");
    for name in &names {
        table.push_str(&format!("    ({name:?}, include_str!({:?})),\n", dir.join(name).to_string_lossy().replace('\\', "/")));
    }
    table.push_str("];\n");
    std::fs::write(PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR")).join("metal_sources.rs"), table).expect("write metal_sources.rs");
}

/// Feature `sycl` (Intel GPUs through SYCL / oneAPI): compiles `sycl/*.cpp` with the DPC++ compiler into `libfllmsycl.a` and links it
/// together with the SYCL runtime. Without a compiler this only warns: no cfg is set, `src/backend/sycl/ffi.rs` then uses Rust stubs
/// (the crate builds, `--device sycl` reports that the runtime is missing), so `cargo check --features sycl` works everywhere.
///
/// The compiler is `FLLM_SYCL_CXX` (explicit path), `$ONEAPI_ROOT/compiler/latest/bin/icpx` (`icx.exe` on Windows), `icpx` / `icx` in
/// PATH, or `/opt/intel/oneapi/compiler/latest/bin/icpx`. `FLLM_SYCL_AOT=acm-g10,pvc,bmg-g21` compiles ahead of time for these GPUs
/// (needs ocloc; default: SPIR-V, compiled by the driver on first use), `FLLM_SYCL_EXTRA_FLAGS` adds flags to every compiler call.
/// `FLLM_SKIP_GPU_KERNELS=1` skips the build like it does for CUDA / HIP (cargo check on a machine with oneAPI).
/// The steps are those of `sycl/build_sycl.sh`: compile every file, pre-link the device images with `-fsycl-link` (so that the final
/// link by the system linker needs no SYCL driver), archive; the archive is linked as a whole (the device object has no symbol that
/// anything references, a normal static link would drop it).
fn sycl_kernels() {
    println!("cargo:rustc-check-cfg=cfg(fllm_sycl_native)");
    println!("cargo:rerun-if-changed=sycl");
    for var in ["FLLM_SYCL_CXX", "ONEAPI_ROOT", "FLLM_SYCL_AOT", "FLLM_SYCL_EXTRA_FLAGS", "FLLM_SKIP_GPU_KERNELS"] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    if env::var_os("FLLM_SKIP_GPU_KERNELS").is_some() {
        return;
    }
    let windows = env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "windows");
    let exe = if windows { "icx.exe" } else { "icpx" };
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(path) = env::var_os("FLLM_SYCL_CXX") {
        candidates.push(PathBuf::from(path));
    }
    if let Some(root) = env::var_os("ONEAPI_ROOT") {
        candidates.push(PathBuf::from(root).join("compiler").join("latest").join("bin").join(exe));
    }
    if let Some(paths) = env::var_os("PATH") {
        candidates.extend(env::split_paths(&paths).map(|dir| dir.join(exe)));
    }
    candidates.push(PathBuf::from(if windows { r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin\icx.exe" } else { "/opt/intel/oneapi/compiler/latest/bin/icpx" }));
    let Some(cxx) = candidates.into_iter().find(|c| c.is_file()) else {
        println!("cargo:warning=feature sycl: no DPC++ compiler found (icpx; set ONEAPI_ROOT, source setvars.sh or set FLLM_SYCL_CXX): the SYCL kernels are not built, --device sycl will report a missing runtime, see docs/gpu-backends.md");
        return;
    };
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let src = PathBuf::from("sycl");
    let mut sources: Vec<PathBuf> = std::fs::read_dir(&src)
        .expect("sycl/")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "cpp"))
        .collect();
    sources.sort();
    let obj_ext = if windows { "obj" } else { "o" };
    let mut flags: Vec<String> = if windows {
        ["-fsycl", "/std:c++17", "/O2", "-ffp-model=precise", "/EHsc"].map(str::to_owned).to_vec()
    } else {
        ["-fsycl", "-std=c++17", "-O3", "-ffp-model=precise", "-fPIC"].map(str::to_owned).to_vec()
    };
    flags.push(format!("-I{}", src.display()));
    // one device image per kernel: the driver then compiles only the kernels that are launched (about 170 instances exist)
    flags.push("-fsycl-device-code-split=per_kernel".to_owned());
    if let Ok(aot) = env::var("FLLM_SYCL_AOT") {
        if !aot.trim().is_empty() {
            flags.extend(["-fsycl-targets=spir64_gen".to_owned(), "-Xsycl-target-backend=spir64_gen".to_owned(), format!("-device {}", aot.trim())]);
        }
    }
    flags.extend(env::var("FLLM_SYCL_EXTRA_FLAGS").unwrap_or_default().split_whitespace().map(str::to_owned));
    let objects: Vec<PathBuf> = sources.iter().map(|s| out.join(format!("{}.{obj_ext}", s.file_stem().expect("file name").to_string_lossy()))).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let jobs = std::thread::available_parallelism().map_or(4, |n| n.get()).min(8);
    std::thread::scope(|scope| {
        for _ in 0..jobs {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let Some(source) = sources.get(i) else { return };
                let status = Command::new(&cxx)
                    .args(&flags)
                    .arg("-c")
                    .arg(source)
                    .arg("-o")
                    .arg(&objects[i])
                    .status()
                    .unwrap_or_else(|e| panic!("cannot run {}: {e}", cxx.display()));
                assert!(status.success(), "{} failed on {}", cxx.display(), source.display());
            });
        }
    });
    let device_object = out.join(format!("fllm_sycl_dev.{obj_ext}"));
    let mut link = Command::new(&cxx);
    link.args(["-fsycl", "-fsycl-link"]);
    if !windows {
        link.arg("-fPIC");
    }
    let status = link.args(flags.iter().filter(|f| f.starts_with("-fsycl-targets") || f.starts_with("-Xsycl-target-backend") || f.starts_with("-fsycl-device-code-split"))).args(&objects).arg("-o").arg(&device_object).status().unwrap_or_else(|e| panic!("cannot run {}: {e}", cxx.display()));
    assert!(status.success(), "device link failed");
    let library = out.join(if windows { "fllmsycl.lib" } else { "libfllmsycl.a" });
    let _ = std::fs::remove_file(&library);
    let status = if windows {
        Command::new(cxx.with_file_name("llvm-lib.exe")).arg(format!("/OUT:{}", library.display())).args(&objects).arg(&device_object).status()
    } else {
        Command::new("ar").arg("crs").arg(&library).args(&objects).arg(&device_object).status()
    }
    .expect("archiver");
    assert!(status.success(), "archiving libfllmsycl failed");
    println!("cargo:rustc-cfg=fllm_sycl_native");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static:+whole-archive=fllmsycl");
    // the SYCL runtime sits next to the compiler: <root>/lib (Linux), <root>/lib (sycl*.lib, Windows)
    let root = cxx.parent().and_then(|bin| bin.parent()).map(PathBuf::from).unwrap_or_default();
    let lib_dir = root.join("lib");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    if windows {
        let import = std::fs::read_dir(&lib_dir).ok().and_then(|d| {
            d.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| n.starts_with("sycl") && n.ends_with(".lib")).min()
        });
        println!("cargo:rustc-link-lib=dylib={}", import.map_or("sycl8".to_owned(), |n| n.trim_end_matches(".lib").to_owned()));
    } else {
        println!("cargo:rustc-link-lib=dylib=sycl");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
        // Intel math libraries that icpx-built objects may reference
        for lib in ["svml", "imf", "irc", "intlc"] {
            if lib_dir.join(format!("lib{lib}.so")).exists() || lib_dir.join(format!("lib{lib}.a")).exists() {
                println!("cargo:rustc-link-lib={lib}");
            }
        }
        for lib in ["dl", "pthread", "stdc++"] {
            println!("cargo:rustc-link-lib={lib}");
        }
    }
}

fn main() {
    provenance();
    if env::var_os("CARGO_FEATURE_VULKAN").is_some() {
        vulkan_shaders();
    }
    if env::var_os("CARGO_FEATURE_METAL").is_some() {
        metal_sources();
    }
    if env::var_os("CARGO_FEATURE_SYCL").is_some() {
        sycl_kernels();
    }
    // `cargo check` of a GPU feature on a machine without the toolchain (no nvcc, no hipcc): FLLM_SKIP_GPU_KERNELS=1 skips the kernel libraries
    if env::var_os("FLLM_SKIP_GPU_KERNELS").is_some() {
        println!("cargo:rerun-if-env-changed=FLLM_SKIP_GPU_KERNELS");
        return;
    }
    println!("cargo:rerun-if-env-changed=FLLM_HIP");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_some() && std::env::var_os("FLLM_HIP").is_some() {
        hip_kernels();
    } else {
        cuda_kernels();
    }
}

// The helpers above are plain functions: `rustc --edition 2021 --test build.rs -o /tmp/build-test && /tmp/build-test` (in rust/engine) runs these.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_names_and_flags() {
        let linux = Target::from_values("linux", "gnu", "fxsr,sse,sse2");
        assert!(!linux.windows && !linux.msvc && !linux.crt_static);
        assert_eq!(linux.object_name("api.cu"), "api.o");
        assert_eq!(linux.library_name("fllmcuda"), "libfllmcuda.a");
        assert_eq!(linux.nvcc_host_flags(), "-fPIC");
        assert_eq!(linux.hipcc_platform_flags(), ["-fPIC"]);

        let msvc = Target::from_values("windows", "msvc", "fxsr,sse,sse2");
        assert!(msvc.windows && msvc.msvc && !msvc.crt_static);
        assert_eq!(msvc.object_name("api.cu"), "api.obj");
        assert_eq!(msvc.object_name("marlin_x.cu"), "marlin_x.obj");
        assert_eq!(msvc.library_name("fllmcuda"), "fllmcuda.lib");
        assert_eq!(msvc.nvcc_host_flags(), "/MD,/EHsc,/bigobj,/Zc:__cplusplus");
        assert_eq!(msvc.hipcc_platform_flags(), ["-fms-runtime-lib=dll"]);
        // no -fPIC for a Windows target: neither cl.exe nor clang for msvc accept it

        let static_crt = Target::from_values("windows", "msvc", "fxsr,crt-static,sse");
        assert!(static_crt.crt_static);
        assert!(static_crt.nvcc_host_flags().starts_with("/MT,"));
        assert_eq!(static_crt.hipcc_platform_flags(), ["-fms-runtime-lib=static"]);

        let mingw = Target::from_values("windows", "gnu", "");
        assert!(mingw.windows && !mingw.msvc);
        assert_eq!(mingw.object_name("api.cu"), "api.o");
        assert!(mingw.hipcc_platform_flags().is_empty());
    }

    #[test]
    fn versions_sort_by_number() {
        assert_eq!(version_key("v12.8"), [12, 8]);
        assert_eq!(version_key("14.40.33807"), [14, 40, 33807]);
        assert_eq!(version_key("other"), Vec::<u32>::new());
        let mut names = vec!["v12.10", "v12.9", "v11.8", "v9.0"];
        names.sort_by_key(|n| version_key(n));
        assert_eq!(names, ["v9.0", "v11.8", "v12.9", "v12.10"]);
    }

    #[test]
    fn the_newest_version_directory_wins() {
        let root = env::temp_dir().join(format!("fllm-build-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for name in ["v11.8", "v12.9", "v12.10", "notes"] {
            std::fs::create_dir_all(root.join(name)).unwrap();
        }
        std::fs::write(root.join("v99.0"), "a file, not a directory").unwrap();
        assert_eq!(newest_version_dir(&root, "v"), Some(root.join("v12.10")));
        assert_eq!(newest_version_dir(&root, "nothing"), None);
        assert_eq!(newest_version_dir(&root.join("missing"), "v"), None);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_lib_response_file_quotes_every_path() {
        let text = lib_response_file(Path::new(r"C:\out dir\fllmcuda.lib"), &[PathBuf::from(r"C:\out dir\api.obj"), PathBuf::from(r"C:\out dir\marlin_a.obj")]);
        assert_eq!(text, "/OUT:\"C:\\out dir\\fllmcuda.lib\"\n\"C:\\out dir\\api.obj\"\n\"C:\\out dir\\marlin_a.obj\"\n");
    }
}
