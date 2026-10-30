#[cfg(target_os = "linux")]
fn pin_threads_to_cores(spec: &str) {
    extern "C" {
        fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u64) -> i32;
    }
    let mut mask = [0u64; 16];
    let spec_trimmed = spec.trim();
    if spec_trimmed.starts_with("0x") || spec_trimmed.starts_with("0X") {
        if let Ok(val) = u64::from_str_radix(&spec_trimmed[2..], 16) {
            mask[0] = val;
        }
    } else if spec_trimmed.contains(',') {
        for part in spec_trimmed.split(',') {
            if let Ok(c) = part.trim().parse::<usize>() {
                if c < 1024 {
                    mask[c / 64] |= 1u64 << (c % 64);
                }
            }
        }
    } else if let Ok(num_cores) = spec_trimmed.parse::<usize>() {
        for i in 0..num_cores.min(1024) {
            mask[i / 64] |= 1u64 << (i % 64);
        }
    }

    let rc = unsafe { sched_setaffinity(0, std::mem::size_of_val(&mask), mask.as_ptr()) };
    if rc == 0 {
        tracing::info!("Pinned inference process to CPU affinity mask: {spec_trimmed}");
    }
}
#[cfg(windows)]
fn pin_threads_to_cores(spec: &str) {
    // a count: the first N logical processors in the order the thread policy fills them (fast cores first)
    match spec.trim().parse::<usize>() {
        Ok(n) if n > 0 => {
            let topology = fllm_engine::cpu_topology::detect();
            if fllm_engine::cpu_topology::pin_first(topology, n) {
                tracing::info!("Restricted the process to {n} logical processors (fast cores first)");
            }
        }
        _ => tracing::warn!("--pin-cores on Windows takes a number of logical processors, not a list or a mask: {spec}"),
    }
}
#[cfg(not(any(target_os = "linux", windows)))]
fn pin_threads_to_cores(_spec: &str) {}
use fllm_engine::{
    compiled_backends, kv_format::KvFormat,
    server::{serve, ServerConfig},
    tokenizer::TokenizerBundle,
    DeviceType, GgufModel, InferenceRuntime,
};
use std::{env, process};

/// A draft model given as a file, a Hugging Face reference or a directory: resolved to one GGUF file (checkpoints are converted).
fn resolve_gguf(spec: &str, offline: bool) -> Result<std::path::PathBuf, String> {
    let path = fllm_engine::import::prepare(
        spec,
        &fllm_engine::import::ImportOptions { revision: None, offline, quant: None, outtype: None, direct: false, no_direct: true },
    )?;
    if !path.is_dir() {
        return Ok(path);
    }
    let mut ggufs: Vec<_> = std::fs::read_dir(&path)
        .map_err(|e| e.to_string())?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()).is_some_and(|s| s.eq_ignore_ascii_case("gguf")))
        .collect();
    ggufs.sort();
    ggufs
        .iter()
        .find(|p| p.to_string_lossy().to_lowercase().contains("q4_k_m"))
        .or_else(|| ggufs.first())
        .cloned()
        .ok_or_else(|| format!("directory '{}' contains no .gguf file", path.display()))
}

fn value(args: &[String], index: &mut usize, flag: &str) -> String {
    *index += 1;
    args.get(*index)
        .cloned()
        .unwrap_or_else(|| usage(&format!("missing value for {flag}")))
}

fn usage(error: &str) -> ! {
    if !error.is_empty() {
        eprintln!("{error}");
    }
    eprintln!("usage: fllm-rs [serve] --model MODEL [--quant TYPE] [--outtype preserve|f16|f32] [--device auto|cpu|cuda|hip|metal|vulkan|dx12|sycl|npu|tpu|BACKEND:N] [--gpu-memory-utilization 0.90] [--gpu-layers auto|N] [--kv-cache-type TYPE | --kv-cache-dtype TYPE] [--cache-type-k TYPE] [--cache-type-v TYPE] [--speculative off|ngram|draft|mtp|suffix|N] [--speculative-lookup N] [--speculative-window W] [--pin-cores N] [--speculative-model PATH_OR_HF] [--num-speculative-tokens N] [--host 127.0.0.1] [--port 8000] [--ctx auto|N] [--max-seqs 16] [--max-batched-tokens 512] [--threads auto|N] [--tensor-parallel-size N | -tp N] [--pipeline-parallel-size N | -pp N] [--expert-parallel] [--devices 0,1,..] [--distributed-backend nccl|p2p|local] [--nnodes N --node-rank R --master-addr HOST --master-port P] [--api-key KEY] [--model-name NAME[,ALIAS...]] [--tool-parser auto|ID] [--reasoning-parser auto|ID] [--no-repack] [--no-jump-forward] [--temp T] [--top-p P] [--top-k K]");
    eprintln!("       --multi-step N (alias --num-scheduler-steps): up to N decode rounds per scheduler round while the batch does not change; off by default, CPU backend only until verified elsewhere");
    eprintln!("       --device: BACKEND:N counts the devices of that backend from 0 (vulkan:1 = second Vulkan device; `fllm-rs devices` prints the value to use); auto = best GPU (CUDA > ROCm > Metal > Vulkan > DX12 > SYCL), else CPU");
    eprintln!("       MODEL: a .gguf file, a Hugging Face link or owner/repo[:QUANT], or a checkpoint (safetensors / PyTorch: unquantized, AWQ, GPTQ, FP8, ModelOpt NVFP4, MXFP4, bitsandbytes, MLX, EXL2, EXL3) as repository or directory: a verified safetensors checkpoint is loaded directly without writing a GGUF file (--direct forces that, --no-direct-load converts to a cached GGUF file instead); --quant ({}) re-encodes an unquantized checkpoint while loading", fllm_checkpoint::Preset::NAMES);
    eprintln!("       KV cache TYPE (keys and values separately with --cache-type-k/-v): auto|{}", KvFormat::NAMES);
    eprintln!("       (wgpu devices: f16 and q8_0 only; nvfp4 is experimental, keys below 8 bit can break some models, see docs/gpu-backends.md)");
    eprintln!("       fllm-rs --version | -V | version");
    eprintln!("       fllm-rs studio [--model MODEL] [--no-open] [serve options]   (the browser app; reuses the last model)");
    eprintln!("       fllm-rs devices [--json] [--measure] | formats | inspect PATH | pull hf://org/repo/file.gguf");
    eprintln!("       fllm-rs npu-setup [--remove]   (downloads the OpenVINO runtime for the Intel NPU into ~/.fllm/openvino; --remove deletes it)");
    eprintln!("       fllm-rs npu-bench [--n 2048] [--k 2048] [--batches 32,64,128] [--variants q4_0:g128,...] [--cache DIR]   (times the Intel NPU against the CPU on one projection; --help)");
    eprintln!("       fllm-rs selftest-cpu [--tier TIER] | selftest-gpu [--device auto|vulkan|metal|hip|...] [--only GROUPS] [--quants LIST] [--json FILE] [--list]");
    eprintln!("       fllm-rs bench-kernel --op marlin_gemv [--m 1] [--n 2048] [--k 2048] [--iters 200] (CUDA builds; --help)");
    eprintln!("       fllm-rs convert CHECKPOINT OUTPUT.gguf [--outtype preserve|f16|f32]");
    eprintln!("       fllm-rs chat | bench --url http://127.0.0.1:8000 [--model MODEL_ID]");
    eprintln!("       fllm-rs gateway telegram --chat ID [--allow-edit] [--yes] [--once]   (token in FLLM_TELEGRAM_TOKEN)");
    eprintln!("       fllm-rs amd-npu status | list | pull MODEL | serve MODEL [--register]   (the NPU of an AMD Ryzen AI through FastFlowLM, which you install yourself)");
    eprintln!("       fllm-rs agent -p \"TASK\" [--folder DIR] [--allow-edit] [--allow-command] [--yes] [--session NAME] [--json]");
    eprintln!(
        "       sampling: --min-p P --typical-p P --mirostat 0|1|2 --dry-multiplier M --seed S"
    );
    eprintln!("       --ctx auto (default) = the trained context of the model: at most 32768 tokens on a GPU (the memory planner lowers it to what fits), at most 8192 on the CPU");
    eprintln!("       --generation-config auto|vllm|PATH (default auto): sampling defaults shipped with the model (GGUF metadata, generation_config.json) for requests that omit them; vllm or --no-generation-config uses none");
    eprintln!("{}", fllm_engine::server::options::USAGE);
    eprintln!("       adapters/draft: --lora FILE --lora-scaled FILE SCALE --draft-model FILE --draft-tokens N");
    eprintln!("       Rust encoder: --model BERT.gguf --embeddings [--pooling cls|mean|last]");
    eprintln!("       vision: --mmproj FILE | --no-vision | --max-image-pixels N | --limit-mm-per-prompt N | --vision-concurrency N");
    process::exit(if error.is_empty() { 0 } else { 2 })
}

fn print_devices_help() {
    println!("usage: fllm-rs devices [--json] [--measure]");
    println!("Lists every compute device of every backend compiled into this binary, with memory and capabilities");
    println!("(FP16, BF16, int8 dot product, matrix cores, unified memory, bandwidth) and the `--device` value that selects it.");
    println!("  --json      machine-readable output");
    println!("  --measure   measure the host memory bandwidth (about a second); GPU bandwidth is shown only where the backend reports it");
    println!("Device numbers: the [n] in front of a device is its position in this list. `--device <backend>:<N>` counts the devices of that");
    println!("backend from 0 (vulkan:1 is the second Vulkan device); see the `Select` line of each device.");
    println!("`--device auto` takes the first of CUDA > ROCm > Metal > Vulkan > DX12 > CPU that has a GPU (FLLM_AUTO_ORDER changes the order,");
    println!("FLLM_AUTO_IGPU=0 leaves integrated GPUs other than Apple's out).");
}

fn print_devices(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    use fllm_engine::devices::{auto_chain, describe_choice, inventory, measure_cpu_bandwidth_gbs, Api, AutoPolicy};
    let (mut json, mut measure) = (false, false);
    for arg in args {
        match arg.as_str() {
            "--json" => json = true,
            "--measure" => measure = true,
            "--help" | "-h" => {
                print_devices_help();
                return Ok(());
            }
            other => return Err(format!("unknown option {other} for `fllm-rs devices` (see `fllm-rs devices --help`)").into()),
        }
    }
    let entries = inventory();
    let policy = AutoPolicy::from_env();
    let chain = auto_chain(&entries, &policy);
    let cpu_bandwidth = measure.then(measure_cpu_bandwidth_gbs);
    let caps_of = |entry: &fllm_engine::devices::InventoryEntry| {
        let mut caps = entry.caps();
        if let (Some(gbs), Api::Host) = (cpu_bandwidth, entry.api) {
            caps.bandwidth_gbs = Some(gbs);
            caps.bandwidth_measured = true;
        }
        caps
    };
    if json {
        let devices: Vec<serde_json::Value> = entries
            .iter()
            .map(|entry| {
                serde_json::json!({
                    "inventory_index": entry.info.index,
                    "name": entry.info.name,
                    "description": entry.info.description,
                    "type": entry.info.dev_type.as_str(),
                    "backend": entry.backend,
                    "api": entry.api.as_str(),
                    "backend_index": entry.local_index,
                    "device_spec": entry.spec,
                    "free_memory_bytes": entry.info.free_memory_bytes,
                    "total_memory_bytes": entry.info.total_memory_bytes,
                    "capabilities": caps_of(entry).to_json(),
                })
            })
            .collect();
        let document = serde_json::json!({
            "tool": "fllm-rs devices",
            "version": env!("CARGO_PKG_VERSION"),
            "version_long": fllm_engine::version::long_version(),
            "compiled_backends": compiled_backends(),
            "devices": devices,
            "cpu_topology": fllm_engine::cpu_topology::detect().to_json(),
            "threads": fllm_engine::cpu_topology::detect().default_threads(),
            "unusable_npus": fllm_engine::backend::npu::status::notices().iter().map(|n| serde_json::json!({"name": n.name, "vendor": n.vendor, "status": n.status, "text": n.text})).collect::<Vec<_>>(),
            "auto_order": policy.order,
            "auto_integrated_gpus": policy.integrated,
            "auto_chain": chain.iter().map(|(backend, index)| format!("{backend}:{index}")).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&document)?);
        return Ok(());
    }
    println!("fLLM Accelerator & Compute Device Inventory\n");
    let backends = compiled_backends();
    println!("Compiled backends: {}", backends.join(", "));

    println!("Detected devices:  {}", entries.len());
    if entries.is_empty() {
        println!("  (no compute devices discovered)");
    }
    for entry in &entries {
        let dev = &entry.info;
        let free_mb = dev.free_memory_bytes / (1024 * 1024);
        let total_mb = dev.total_memory_bytes / (1024 * 1024);
        let (capabilities, role) = match dev.dev_type {
            DeviceType::Cpu => (
                fllm_engine::devices::cpu_capabilities(),
                "Host Fallback / CPU Inference / Reference",
            ),
            DeviceType::Gpu | DeviceType::IntegratedGpu => {
                let capabilities = if entry.api == Api::Wgpu {
                    fllm_engine::devices::gpu_capabilities()
                } else {
                    format!("native {} backend; check with `fllm-rs selftest-gpu --device {}`", entry.backend, entry.spec)
                };
                let role = if dev.dev_type == DeviceType::IntegratedGpu {
                    "Partial Offload (Layers, Shared Memory)"
                } else if total_mb >= 16384 {
                    "Primary Prefill & Decode (Full Offload)"
                } else {
                    "Partial Offload (Layers + KV Cache)"
                };
                (capabilities, role)
            }
            DeviceType::Accelerator => (
                format!("native {} backend (experimental); check with `fllm-rs selftest-gpu --device {}`", entry.backend, entry.spec),
                "Only with an explicit --device; never chosen by auto",
            ),
        };
        let caps = caps_of(entry);
        // CUDA: the compute capability and what this binary has for it (cuda/arch_report.cu)
        let mut cuda_lines = String::new();
        if let Some(cc) = &caps.compute_capability {
            cuda_lines.push_str(&format!("\n      Compute:      capability {cc}"));
            if let Some(kernels) = &caps.kernels {
                cuda_lines.push_str(&format!("; kernels of this binary: {kernels}"));
            }
        }
        if !caps.fast_paths.is_empty() {
            cuda_lines.push_str(&format!("\n      Fast paths:   {}", caps.fast_paths.join("; ")));
        }
        println!(
            "  [{}] {} ({})\n      Backend:      {}\n      Select:       --device {}\n      Type:         {}\n      Memory:       {} MB free / {} MB total\n      Features:     {}\n      Capabilities: {}{}\n      Recommended:  {}",
            dev.index,
            dev.name,
            dev.description,
            if entry.api == Api::Host { "cpu".to_owned() } else { format!("{} ({})", entry.backend, entry.api.as_str()) },
            entry.spec,
            dev.dev_type.as_str(),
            free_mb,
            total_mb,
            caps.summary(),
            capabilities,
            cuda_lines,
            role,
        );
    }
    {
        let topology = fllm_engine::cpu_topology::detect();
        println!(
            "CPU threads (--threads auto starts from): {}; a short measurement at model load may adjust it (FLLM_NO_AUTOTUNE=1 skips it)",
            fllm_engine::cpu_threads::summary(topology.default_threads(), topology)
        );
    }
    // NPUs the operating system lists but that cannot be used (yet): shown so that the user knows why `--device npu` does not work
    let notices = fllm_engine::backend::npu::status::notices();
    for notice in &notices {
        println!("  [-] {} ({})
      Status:       {}", notice.name, notice.vendor, notice.text);
    }
    println!();
    let described: Vec<String> = chain.iter().map(|(backend, index)| describe_choice(backend, *index)).collect();
    println!("Auto selection (--device auto): {}", described.join(" -> "));
    if chain.len() == 1 {
        println!("Active default device (auto): CPU");
    }
    println!("Device numbers: [n] is the position in this list; `--device <backend>:<N>` counts the devices of one backend from 0 (see `Select`).");
    #[cfg(feature = "rust-gpu")]
    println!("The wgpu adapters are selected with --device vulkan, metal or dx12 (or wgpu-vulkan, ... where a native backend of the same API exists).");
    if !measure {
        println!("Bandwidth is shown only where the backend reports it; `fllm-rs devices --measure` measures the host memory.");
    }
    Ok(())
}

struct StatusGuard {
    path: std::path::PathBuf,
}
impl Drop for StatusGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// The device whose memory the planner budgets for `--device`: `None` when the model runs on the CPU.
/// Why the latent MLA cache cannot be used with this backend, device and cache format (`None`: it can).
fn latent_mla_unavailable(model: &GgufModel, backend: &str, index: usize, key: KvFormat, value: KvFormat) -> Option<String> {
    if backend != "cuda" {
        return Some(format!("the {backend} backend has no latent MLA kernels"));
    }
    if !fllm_engine::mla_latent::format_supported(key) {
        return Some(format!("the {} key cache format is not covered", key.name()));
    }
    if value.check_dim(fllm_engine::mla_latent::LATENT_VALUE_DIM).is_err() {
        return Some(format!("the {} value cache format cannot hold the placeholder value pool", value.name()));
    }
    #[cfg(feature = "cuda")]
    {
        let m = model.config().mla.as_ref()?;
        if !fllm_engine::backend::cuda::mla_latent_supported(index, m, key) {
            return Some("the CUDA kernels do not cover this model's dimensions on this GPU".to_owned());
        }
        None
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (model, index);
        Some("this build has no CUDA backend".to_owned())
    }
}

fn plan_target(selected_backend: &str, selected_index: usize) -> Option<fllm_engine::DeviceInfo> {
    // the NPU backend is a hybrid: the host's memory and the CPU stream do the work, so the plan is the CPU's
    if selected_backend == "cpu" || selected_backend == "npu" {
        return None;
    }
    fllm_engine::devices::find_entry(selected_backend, selected_index).map(|entry| entry.info)
}

/// `fllm-rs studio [--model M] [serve options]`: the server for the browser app. Starts with the last model when none is given
/// and opens the page in the default browser (`--no-open` keeps it closed).
fn studio_args(rest: &[String]) -> Vec<String> {
    let mut args = vec!["serve".to_owned()];
    let mut open = true;
    for a in rest {
        if a == "--no-open" {
            open = false;
        } else {
            args.push(a.clone());
        }
    }
    let given = args.iter().position(|a| a == "--model" || a == "-m").and_then(|i| args.get(i + 1).cloned()).or_else(|| {
        args.iter().find_map(|a| a.strip_prefix("--model=").map(str::to_owned))
    });
    match given {
        Some(model) => fllm_engine::hub::remember_studio_model(&model),
        None => match fllm_engine::hub::last_studio_model() {
            Some(model) if fllm_engine::hub::load_on_start() => {
                args.push("--model".into());
                args.push(model);
            }
            // the last model is known but loads with the first message (Settings > Models > Load the last model when the app starts)
            Some(_) => env::set_var("FLLM_STUDIO_SETUP", "1"),
            // no model chosen yet: the server starts without one and the page offers to pick one
            None => env::set_var("FLLM_STUDIO_SETUP", "1"),
        },
    }
    env::set_var("FLLM_STUDIO", "1");
    // the models folder chosen in the Studio
    fllm_engine::hub::apply_saved_models_dir();
    if open {
        env::set_var("FLLM_OPEN_BROWSER", "1");
    }
    args
}

/// The Studio must stay usable when its model cannot be loaded (unsupported architecture, broken file, not enough memory): it then
/// serves the model page with the reason instead of ending, so the app never sits in front of a window that stays empty.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let result = run();
    if let Err(error) = &result {
        if env::var_os("FLLM_STUDIO").is_some() {
            let args = env::args().collect::<Vec<_>>();
            let after = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned();
            let host = after("--host").unwrap_or_else(|| "127.0.0.1".to_owned());
            let port = after("--port").and_then(|p| p.parse().ok()).unwrap_or(8000);
            let message = error.to_string();
            eprintln!("error: {message}");
            fllm_engine::hub::forget_studio_model();
            if fllm_engine::server::serve_setup(&host, port, Some(message)).is_ok() {
                return Ok(());
            }
        }
    }
    result
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if matches!(args.first().map(String::as_str), Some("--version" | "-V" | "version")) {
        println!("{}", fllm_engine::version::long_version());
        return Ok(());
    }
    // hidden: the launcher of the Windows command sandbox; first of all, because it hands its standard handles to the command
    // (nothing may read stdin or start a thread before it)
    if args.first().map(String::as_str) == Some("sandbox-exec") {
        std::process::exit(fllm_engine::server::run_sandbox_launcher(&args[1..]));
    }
    // hidden: writes a tiny random model for the HTTP API tests (scripts/api-tests, .github/workflows/api.yml)
    if args.first().map(String::as_str) == Some("make-tiny-model") {
        return fllm_engine::tiny_model::run(&args[1..]).map_err(Into::into);
    }
    // a program that starts the server (the desktop app) closes the pipe on stdin when it ends, which stops the server too
    if env::var_os("FLLM_EXIT_ON_STDIN_EOF").is_some() {
        std::thread::spawn(|| {
            let mut sink = Vec::new();
            let _ = std::io::Read::read_to_end(&mut std::io::stdin(), &mut sink);
            std::process::exit(0);
        });
    }
    let args = if args.first().map(String::as_str) == Some("studio") { studio_args(&args[1..]) } else { args };
    // the engine options chosen on the Settings page (caches, telemetry); in `serve` too, which is what a container runs
    fllm_engine::hub::apply_saved_engine_settings();
    if fllm_engine::cli::dispatch(&args)? {
        return Ok(());
    }
    if args.is_empty() {
        usage("");
    }
    if args.first().map(String::as_str) == Some("npu-setup") {
        return npu_setup(&args[1..]);
    }
    #[cfg(feature = "npu-openvino")]
    if args.first().map(String::as_str) == Some("npu-bench") {
        return fllm_engine::backend::npu::openvino::bench::run(&args[1..]).map_err(Into::into);
    }
    if args.first().map(String::as_str) == Some("amd-npu") {
        return fllm_engine::backend::npu::flm::cli(
            &args[1..],
            &|tag, base| fllm_engine::server::amd_npu_register(tag, base),
            &fllm_engine::server::amd_npu_unregister,
        )
        .map_err(Into::into);
    }
    if args.first().map(String::as_str) == Some("devices") {
        return print_devices(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("selftest-gpu") {
        return Ok(fllm_engine::backend::selftest::run(&args[1..])?);
    }
    if args.first().map(String::as_str) == Some("bench-kernel") {
        return Ok(fllm_engine::bench_kernel::run(&args[1..])?);
    }
    let mut i = if args.first().map(String::as_str) == Some("serve") {
        1usize
    } else if args.first().map(|s| s.starts_with('-')).unwrap_or(false) {
        0usize
    } else {
        usage(&format!("unknown subcommand: {}", args[0]));
    };

    let mut model_path = None;
    let mut hub_revision = None;
    let mut hub_offline = false;
    let mut import_quant: Option<String> = None;
    let mut import_outtype: Option<String> = None;
    let mut import_direct = false;
    let mut import_no_direct = false;
    let mut host = "127.0.0.1".to_owned();
    let mut port = 8000u16;
    // None: --ctx auto
    let mut context_request: Option<usize> = None;
    let mut max_sequences = 16usize;
    let mut max_seqs_auto = false;
    let mut scheduling_policy = fllm_engine::scheduler::SchedulingPolicy::Fcfs;
    let mut sched_watermark: Option<usize> = None;
    let mut max_partial_prefills = 0usize;
    let mut multi_step = 1usize;
    let mut long_prefill_threshold = 0usize;
    let mut max_long_partial_prefills = 1usize;
    let mut decode_first = true;
    let mut preemption_swap = false;
    let mut swap_space_gib = 0.0f64;
    let mut max_batched_tokens = 512usize;
    let mut batch_explicit = false;
    // --threads auto (the default): the processor's own default now, refined by a short measurement once the model is loaded
    let mut threads = fllm_engine::cpu_topology::detect().default_threads() as i32;
    let mut threads_auto = true;
    let mut http = fllm_engine::server::options::HttpOptions::default();
    let mut generation_source = fllm_engine::server::gencfg::Source::Auto;
    let mut jump_forward = fllm_engine::structured::jump_forward_default();
    let mut default_temperature: Option<f64> = None;
    let mut default_top_p: Option<f64> = None;
    let mut default_top_k: Option<usize> = None;
    let mut model_name = None;
    let mut tool_parser: Option<String> = None;
    let mut reasoning_parser: Option<String> = None;
    let mut device = "auto".to_owned();
    let mut gpu_memory_utilization = 0.90f64;
    let mut gpu_layers = "auto".to_owned();
    let mut kv_cache_type: Option<String> = None;
    let mut kv_cache_type_k: Option<String> = None;
    let mut kv_cache_type_v: Option<String> = None;
    let mut speculative_lookup = 0usize;
    let mut speculative_mtp = false;
    let mut speculative_window = 32usize;
    let mut draft_model = None;
    let mut draft_tokens = 5usize;
    let mut speculative_method: Option<fllm_engine::speculative::SpeculativeMethod> = None;
    let mut num_speculative_tokens: Option<usize> = None;
    let mut speculative_suffix = false;
    let mut speculative_min_match = 3usize;
    let mut speculative_cache_tokens = 1usize << 20;
    let mut lora_paths = Vec::new();
    let mut sampling_options = serde_json::Map::new();
    let mut pin_cores = None;
    let mut embeddings = false;
    let mut pooling = None;
    let mut mmproj = None;
    let mut parallel = fllm_engine::parallel::ParallelConfig::default();
    let mut no_vision = false;
    let mut media_limits = fllm_engine::vision::preprocess::MediaLimits::get();

    while i < args.len() {
        match args[i].as_str() {
            flag if fllm_engine::parallel::ParallelConfig::is_flag(flag) => {
                let flag = flag.to_owned();
                let given = fllm_engine::parallel::ParallelConfig::takes_value(&flag).then(|| value(&args, &mut i, &flag));
                parallel.apply_flag(&flag, given.as_deref()).unwrap_or_else(|e| usage(&e));
            }
            "--min-p"
            | "--typical-p"
            | "--presence-penalty"
            | "--frequency-penalty"
            | "--repeat-penalty"
            | "--repeat-last-n"
            | "--dry-multiplier"
            | "--dry-base"
            | "--dry-allowed-length"
            | "--dry-penalty-last-n"
            | "--mirostat"
            | "--mirostat-tau"
            | "--mirostat-eta"
            | "--seed" => {
                let flag = args[i].clone();
                let key = flag.trim_start_matches('-').replace('-', "_");
                let raw = value(&args, &mut i, &flag);
                let number: serde_json::Value = serde_json::from_str(&raw)
                    .unwrap_or_else(|_| usage(&format!("invalid number for {flag}")));
                sampling_options.insert(key, number);
            }
            "--draft-model" | "--model-draft" => {
                let flag = args[i].clone();
                draft_model = Some(value(&args, &mut i, &flag));
            }
            "--draft-tokens" => {
                draft_tokens = value(&args, &mut i, "--draft-tokens")
                    .parse()
                    .unwrap_or_else(|_| usage("invalid draft token count"));
            }
            "--lora" => {
                lora_paths.push((value(&args, &mut i, "--lora"), 1.0f32));
            }
            "--lora-scaled" => {
                let path = value(&args, &mut i, "--lora-scaled");
                let scale = value(&args, &mut i, "--lora-scaled scale")
                    .parse()
                    .unwrap_or_else(|_| usage("invalid LoRA scale"));
                lora_paths.push((path, scale));
            }
            "--quant" | "--quantize" => import_quant = Some(value(&args, &mut i, "--quant")),
            "--outtype" => import_outtype = Some(value(&args, &mut i, "--outtype")),
            "--revision" => hub_revision = Some(value(&args, &mut i, "--revision")),
            "--local-files-only" | "--offline" => hub_offline = true,
            "--direct" | "--zero-copy" => import_direct = true,
            "--no-direct-load" | "--no-direct" => import_no_direct = true,
            "--model" => model_path = Some(value(&args, &mut i, "--model")),
            "--engine" => {
                let engine = value(&args, &mut i, "--engine");
                if !matches!(engine.as_str(), "auto" | "rust") {
                    usage("only the independent Rust engine is supported");
                }
            }
            "--embeddings" => embeddings = true,
            "--mmproj" => mmproj = Some(value(&args, &mut i, "--mmproj")),
            "--no-vision" => no_vision = true,
            "--max-image-pixels" => {
                media_limits.max_image_pixels = value(&args, &mut i, "--max-image-pixels")
                    .parse()
                    .ok()
                    .filter(|&n| n >= 784)
                    .unwrap_or_else(|| usage("--max-image-pixels needs a pixel count of at least 784"));
            }
            "--limit-mm-per-prompt" => {
                media_limits.max_media_per_prompt = value(&args, &mut i, "--limit-mm-per-prompt")
                    .parse()
                    .ok()
                    .filter(|&n| n > 0)
                    .unwrap_or_else(|| usage("--limit-mm-per-prompt needs a positive count"));
            }
            "--vision-concurrency" => {
                media_limits.concurrency = value(&args, &mut i, "--vision-concurrency")
                    .parse()
                    .ok()
                    .filter(|n| (1..=16).contains(n))
                    .unwrap_or_else(|| usage("--vision-concurrency needs a count from 1 to 16"));
            }
            "--compute" => {
                // Kept for older command lines: Rust is the only compute path.
                let compute = value(&args, &mut i, "--compute");
                if !matches!(compute.as_str(), "rust" | "auto") {
                    usage("--compute native was removed; fLLM computes in Rust only");
                }
            }
            "--pooling" => pooling = Some(value(&args, &mut i, "--pooling")),
            "--host" => host = value(&args, &mut i, "--host"),
            "--port" => {
                port = value(&args, &mut i, "--port")
                    .parse()
                    .unwrap_or_else(|_| usage("invalid --port"))
            }
            "--ctx" | "--ctx-size" | "--max-model-len" => {
                let flag = args[i].clone();
                let text = value(&args, &mut i, &flag);
                context_request = if text.eq_ignore_ascii_case("auto") {
                    None
                } else {
                    Some(text.parse().ok().filter(|n| *n > 0).unwrap_or_else(|| usage("invalid context length (a positive number, or auto)")))
                }
            }
            "--preemption-mode" => match value(&args, &mut i, "--preemption-mode").as_str() {
                "recompute" => preemption_swap = false,
                "swap" => preemption_swap = true,
                _ => usage("--preemption-mode expects recompute or swap"),
            },
            "--swap-space" => {
                swap_space_gib = value(&args, &mut i, "--swap-space")
                    .parse()
                    .ok()
                    .filter(|g: &f64| g.is_finite() && *g >= 0.0)
                    .unwrap_or_else(|| usage("--swap-space expects a number of GiB"));
            }
            "--scheduling-policy" => {
                scheduling_policy = fllm_engine::scheduler::SchedulingPolicy::parse(&value(&args, &mut i, "--scheduling-policy"))
                    .unwrap_or_else(|| usage("--scheduling-policy expects fcfs or priority"));
            }
            "--kv-watermark-blocks" => {
                sched_watermark = Some(
                    value(&args, &mut i, "--kv-watermark-blocks")
                        .parse()
                        .unwrap_or_else(|_| usage("invalid --kv-watermark-blocks")),
                );
            }
            "--multi-step" | "--num-scheduler-steps" => {
                multi_step = value(&args, &mut i, "--multi-step")
                    .parse()
                    .ok()
                    .filter(|n| (1..=16).contains(n))
                    .unwrap_or_else(|| usage("--multi-step expects a number from 1 to 16"));
            }
            "--max-num-partial-prefills" => {
                max_partial_prefills = value(&args, &mut i, "--max-num-partial-prefills")
                    .parse()
                    .unwrap_or_else(|_| usage("invalid --max-num-partial-prefills"));
            }
            "--long-prefill-token-threshold" => {
                long_prefill_threshold = value(&args, &mut i, "--long-prefill-token-threshold")
                    .parse()
                    .unwrap_or_else(|_| usage("invalid --long-prefill-token-threshold"));
            }
            "--max-long-partial-prefills" => {
                max_long_partial_prefills = value(&args, &mut i, "--max-long-partial-prefills")
                    .parse()
                    .ok()
                    .filter(|n| *n >= 1)
                    .unwrap_or_else(|| usage("invalid --max-long-partial-prefills"));
            }
            "--chunked-prefill-policy" => match value(&args, &mut i, "--chunked-prefill-policy").as_str() {
                "prefill-first" => decode_first = false,
                "decode-first" => decode_first = true,
                _ => usage("--chunked-prefill-policy expects prefill-first or decode-first"),
            },
            "--max-seqs" if args.get(i + 1).is_some_and(|v| v == "auto") => {
                i += 1;
                max_seqs_auto = true;
            }
            "--max-seqs" => {
                max_sequences = value(&args, &mut i, "--max-seqs")
                    .parse()
                    .unwrap_or_else(|_| usage("invalid --max-seqs"))
            }
            "--max-batched-tokens" => {
                batch_explicit = true;
                max_batched_tokens = value(&args, &mut i, "--max-batched-tokens")
                    .parse()
                    .unwrap_or_else(|_| usage("invalid --max-batched-tokens"))
            }
            "--threads" => {
                let given = value(&args, &mut i, "--threads");
                if given.eq_ignore_ascii_case("auto") {
                    threads_auto = true;
                } else {
                    threads = given.parse().ok().filter(|n| *n > 0).unwrap_or_else(|| usage("invalid --threads"));
                    threads_auto = false;
                }
            }
            "--device" => device = value(&args, &mut i, "--device").to_lowercase(),
            // `all`: the NPU computes every matrix product it can (the defaults send only long prompt blocks to it; the native AMD XDNA engine takes its three FLLM_XDNA_MIN_* settings)
            "--npu-mode" => {
                if value(&args, &mut i, "--npu-mode").eq_ignore_ascii_case("all") {
                    // every product through the NPU needs the hybrid backend, not the host decoder
                    if env::var_os("FLLM_NPU_HYBRID").is_none() {
                        env::set_var("FLLM_NPU_HYBRID", "1");
                    }
                    for (name, setting) in [("FLLM_NPU_MIN_BATCH", "1"), ("FLLM_NPU_CHUNK", "16"), ("FLLM_NPU_MIN_ELEMS", "1"), ("FLLM_NPU_CPU_GOPS", "0.001"), ("FLLM_NPU_OVERHEAD_US", "0"), ("FLLM_XDNA_MIN_ROWS", "1"), ("FLLM_XDNA_MIN_ELEMS", "1"), ("FLLM_XDNA_MIN_EFF", "0")] {
                        if env::var_os(name).is_none() {
                            env::set_var(name, setting);
                        }
                    }
                }
            }
            "--gpu-memory-utilization" => {
                gpu_memory_utilization = value(&args, &mut i, "--gpu-memory-utilization")
                    .parse()
                    .unwrap_or_else(|_| usage("invalid --gpu-memory-utilization"));
            }
            "--gpu-layers" | "-ngl" | "--n-gpu-layers" => {
                let flag = args[i].clone();
                gpu_layers = value(&args, &mut i, &flag);
            }
            "--generation-config" => {
                generation_source = fllm_engine::server::gencfg::Source::parse(&value(&args, &mut i, "--generation-config"))
            }
            "--no-generation-config" => generation_source = fllm_engine::server::gencfg::Source::Vllm,
            "--no-repack" => fllm_engine::cpu_kernels::set_repack(false),
            "--no-jump-forward" => jump_forward = false,
            "--jump-forward" => jump_forward = true,
            "--tool-parser" => {
                let name = value(&args, &mut i, "--tool-parser");
                if !fllm_engine::tool_parsers::is_known(&name) {
                    usage(&format!(
                        "unknown --tool-parser {name} (available: auto, {})",
                        fllm_engine::tool_parsers::parser_ids().join(", ")
                    ));
                }
                tool_parser = Some(name);
            }
            "--reasoning-parser" => {
                let name = value(&args, &mut i, "--reasoning-parser");
                if !fllm_engine::reasoning::Family::is_known(&name) {
                    usage(&format!(
                        "unknown --reasoning-parser {name} (available: auto, {})",
                        fllm_engine::reasoning::family_ids().join(", ")
                    ));
                }
                reasoning_parser = Some(name);
            }
            "--temp" | "--temperature" => {
                let flag = args[i].clone();
                default_temperature = Some(
                    value(&args, &mut i, &flag)
                        .parse()
                        .unwrap_or_else(|_| usage("invalid --temp")),
                );
            }
            "--top-p" => {
                default_top_p = Some(
                    value(&args, &mut i, "--top-p")
                        .parse()
                        .unwrap_or_else(|_| usage("invalid --top-p")),
                );
            }
            "--top-k" => {
                default_top_k = Some(
                    value(&args, &mut i, "--top-k")
                        .parse()
                        .unwrap_or_else(|_| usage("invalid --top-k")),
                );
            }
            "--model-name" | "--served-name" => {
                let flag = args[i].clone();
                model_name = Some(value(&args, &mut i, &flag))
            }
            "--kv-cache-type" | "--kv-cache-dtype" => {
                let flag = args[i].clone();
                kv_cache_type = Some(value(&args, &mut i, &flag).to_lowercase());
            }
            "--cache-type-k" | "-ctk" => {
                let flag = args[i].clone();
                kv_cache_type_k = Some(value(&args, &mut i, &flag).to_lowercase());
            }
            "--cache-type-v" | "-ctv" => {
                let flag = args[i].clone();
                kv_cache_type_v = Some(value(&args, &mut i, &flag).to_lowercase());
            }
            "--speculative" => {
                let source = value(&args, &mut i, "--speculative");
                match source.as_str() {
                    "off" => {
                        speculative_mtp = false;
                        speculative_lookup = 0;
                        speculative_method = Some(fllm_engine::speculative::SpeculativeMethod::Off);
                    }
                    _ => {
                        if let Some(method) = fllm_engine::speculative::SpeculativeMethod::parse(&source) {
                            if let Some(reason) = method.unsupported_reason() {
                                usage(reason);
                            }
                            speculative_method = Some(method);
                        } else {
                            // a bare number is the lookup length of the older syntax
                            speculative_lookup = source.parse().unwrap_or_else(|_| {
                                usage("--speculative expects off, ngram, draft, mtp, suffix or a lookup length")
                            });
                        }
                    }
                }
            }
            "--speculative-model" => {
                draft_model = Some(value(&args, &mut i, "--speculative-model"));
            }
            "--num-speculative-tokens" => {
                num_speculative_tokens = Some(
                    value(&args, &mut i, "--num-speculative-tokens")
                        .parse()
                        .ok()
                        .filter(|n| (1..=32).contains(n))
                        .unwrap_or_else(|| usage("--num-speculative-tokens expects 1 to 32")),
                );
            }
            "--speculative-tree" => {
                let shape = value(&args, &mut i, "--speculative-tree");
                if !matches!(shape.as_str(), "off" | "none" | "0") {
                    usage("--speculative-tree: tree verification needs a tree-attention pass that this engine does not have yet; draft chains are verified (see docs/speculative-decoding.md)");
                }
            }
            "--speculative-min-match" => {
                speculative_min_match = value(&args, &mut i, "--speculative-min-match")
                    .parse()
                    .ok()
                    .filter(|n| (1..=32).contains(n))
                    .unwrap_or_else(|| usage("--speculative-min-match expects 1 to 32"));
            }
            "--speculative-cache-tokens" => {
                speculative_cache_tokens = value(&args, &mut i, "--speculative-cache-tokens")
                    .parse()
                    .ok()
                    .filter(|n| *n >= 1024)
                    .unwrap_or_else(|| usage("--speculative-cache-tokens expects at least 1024"));
            }
            "--speculative-lookup" | "-sp" => {
                let flag = args[i].clone();
                speculative_lookup = value(&args, &mut i, &flag)
                    .parse()
                    .unwrap_or_else(|_| usage("invalid --speculative-lookup"));
            }
            "--speculative-window" | "-sw" => {
                let flag = args[i].clone();
                speculative_window = value(&args, &mut i, &flag)
                    .parse()
                    .unwrap_or_else(|_| usage("invalid --speculative-window"));
            }
            "-C" | "--pin-cores" | "--p-cores" | "--pin-mask" => {
                let flag = args[i].clone();
                pin_cores = Some(value(&args, &mut i, &flag));
            }
            "--help" | "-h" => usage(""),
            other => match http.parse_flag(&args, &mut i) {
                Ok(true) => {}
                Ok(false) => usage(&format!("unknown option: {other}")),
                Err(message) => usage(&message),
            },
        }
        i += 1;
    }
    http.apply_environment();
    if let Some(list) = model_name.take() {
        // --model-name NAME[,ALIAS...]: the first is the served name, the others are accepted in the `model` field
        let mut names = list.split(',').map(str::trim).filter(|n| !n.is_empty()).map(str::to_owned);
        model_name = names.next();
        http.model_aliases.extend(names);
    }
    http.validate()?;
    fllm_engine::server::logging::init(&http);
    {
        use fllm_engine::speculative::SpeculativeMethod as Method;
        match speculative_method {
            Some(Method::Ngram) => {
                if speculative_lookup == 0 {
                    speculative_lookup = num_speculative_tokens.unwrap_or(5);
                }
            }
            Some(Method::Draft) => {
                if draft_model.is_none() {
                    usage("--speculative draft needs --speculative-model (or --draft-model)");
                }
            }
            Some(Method::Mtp) => speculative_mtp = true,
            Some(Method::Suffix) => speculative_suffix = true,
            Some(Method::Off) | None => {}
            Some(other) => usage(other.unsupported_reason().unwrap_or("unsupported speculation method")),
        }
        if let Some(n) = num_speculative_tokens {
            draft_tokens = n;
        }
        if speculative_method == Some(Method::Off) {
            draft_model = None;
        }
    }
    if model_path.is_none() && env::var_os("FLLM_STUDIO_SETUP").is_some() {
        if let Err(e) = fllm_engine::server::serve_setup(&host, port, None) {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
        return Ok(());
    }
    let model_path = model_path.unwrap_or_else(|| usage("--model is required"));
    // `--device cuda:0,vulkan:1,cpu`: the model is split over these devices; `device` names the first one for everything that is not split
    let multi_devices: Option<Vec<(String, usize)>> = if fllm_engine::devices::is_list(&device) {
        let list = fllm_engine::devices::parse_device_list(&device).unwrap_or_else(|e| usage(&e));
        parallel.apply_device_list(&list).unwrap_or_else(|e| usage(&e));
        device = if list[0].0 == "cpu" { "cpu".to_owned() } else { format!("{}:{}", list[0].0, list[0].1) };
        if draft_model.is_some() || speculative_mtp {
            usage("draft-model and MTP speculation are not available with several devices (--device a,b); lookup speculation is");
        }
        Some(list)
    } else {
        None
    };
    if let Err(e) = parallel.validate() {
        usage(&e);
    }
    if parallel.is_parallel() {
        fllm_engine::parallel::config::set_global(parallel.clone());
    }
    let mut path = fllm_engine::import::prepare(
        &model_path,
        &fllm_engine::import::ImportOptions {
            revision: hub_revision.clone(),
            offline: hub_offline,
            quant: import_quant,
            outtype: import_outtype,
            direct: import_direct,
            no_direct: import_no_direct,
        },
    )?;
    if path.is_dir() && !fllm_engine::import::is_direct(&path) {
        let mut ggufs = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&path) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_file()
                    && p.extension()
                        .and_then(|s| s.to_str())
                        .map(|s| s.eq_ignore_ascii_case("gguf"))
                        .unwrap_or(false)
                {
                    ggufs.push(p);
                }
            }
        }
        if let Some(chosen) = ggufs
            .iter()
            .find(|p| p.to_string_lossy().to_lowercase().contains("q4_k_m"))
            .or_else(|| ggufs.first())
        {
            tracing::info!(
                "Resolved checkpoint directory '{}' to GGUF model '{}'",
                path.display(),
                chosen.display()
            );
            path = chosen.clone();
        } else {
            return Err(format!(
                "directory '{}' contains neither a .gguf model nor a checkpoint (config.json with model.safetensors or pytorch_model.bin)",
                path.display()
            )
            .into());
        }
    }
    let direct_load = fllm_engine::import::is_direct(&path);
    let inferred_name = if direct_load {
        fllm_engine::import::direct_model_name(&path)
    } else {
        path.file_stem()
            .and_then(|x| x.to_str())
            .map(fllm_engine::import::display_name)
            .unwrap_or("fllm-model")
            .to_owned()
    };

    // a checkpoint loaded directly is built once here and shared with the model (a GGUF file is cheap to map again)
    let mapping = fllm_engine::import::open_model(&path)?;
    let facts = fllm_engine::server::ModelFacts::from_gguf(&path, &mapping);
    let is_encoder = mapping
        .index()
        .metadata
        .get("general.architecture")
        .and_then(serde_json::Value::as_str)
        == Some("bert");
    // `--device npu` with an Intel NPU: the host decoder runs the model and sends the long prompt projections to the NPU
    #[cfg(feature = "npu-openvino")]
    if let Some(rerouted) = fllm_engine::backend::npu::host::route_device(&device) {
        device = rerouted;
    }
    // fail early on a malformed or impossible --device, before the model is read; `vulkan:1` counts the devices of that backend
    let device_chain = fllm_engine::devices::device_candidates(&device)?;
    if device == "cpu" {
        tracing::info!("Using explicit CPU runtime.");
    }

    if is_encoder {
        if draft_model.is_some()
            || !lora_paths.is_empty()
            || speculative_lookup > 0
            || speculative_mtp
        {
            return Err("draft models, LoRA and speculative decoding apply to decoder models, not BERT encoders".into());
        }
        if let Some(ref cores) = pin_cores {
            pin_threads_to_cores(cores);
        }
        let mut encoder = if direct_load {
            fllm_engine::encoder::BertEncoder::from_mapping(mapping.with_tensor_aliases(&[])?, threads.max(1) as usize, pooling)?
        } else {
            fllm_engine::encoder::BertEncoder::load(path.to_str().ok_or("invalid model path")?, threads.max(1) as usize, pooling)?
        };
        encoder.set_device(&device)?;
        if batch_explicit {
            fllm_engine::encoder_server::set_max_batched_tokens(max_batched_tokens);
        }
        tracing::info!(
            "Independent Rust BERT encoder: {} dimensions, {} token context, pooling={}",
            encoder.hidden, encoder.context, encoder.default_pooling
        );
        return fllm_engine::encoder_server::serve(
            encoder,
            ServerConfig {
                host,
                port,
                model_name: model_name.unwrap_or(inferred_name),
                http,
                facts,
                default_temperature,
                default_top_p,
                default_top_k,
                sampling_defaults: Default::default(),
                tool_parser: None,
                reasoning_parser: None,
                jump_forward: false,
            },
        )
        .map_err(Into::into);
    }
    if embeddings || pooling.is_some() {
        return Err("--embeddings/--pooling selects a dedicated Rust BERT encoder; decoder models already expose contextual embedding endpoints".into());
    }
    let (selected_backend, selected_index) = device_chain[0].clone();
    if fllm_engine::devices::is_auto(&device) {
        tracing::info!("--device auto: {}", fllm_engine::devices::describe_chain(&device_chain));
    }
    let plan_device = plan_target(&selected_backend, selected_index);
    let tokenizer = TokenizerBundle::from_gguf(&mapping)?;
    // sampling defaults of the model (--generation-config) sit below the command line: --temp, --top-p, --top-k, --min-p
    // and --repeat-penalty win over them, a request field wins over everything
    let (model_defaults, model_defaults_origin) = fllm_engine::server::gencfg::GenerationDefaults::resolve(
        &generation_source,
        &mapping.index().metadata,
        &path,
    )?;
    if model_defaults.is_empty() {
        tracing::info!("sampling defaults of the model: {model_defaults_origin}; requests without temperature use the OpenAI default (temperature 1.0)");
    } else {
        tracing::info!("sampling defaults of the model ({model_defaults_origin}): {}", model_defaults.describe());
    }
    default_temperature = default_temperature.or(model_defaults.temperature);
    default_top_p = default_top_p.or(model_defaults.top_p);
    default_top_k = default_top_k.or(model_defaults.top_k);
    if let Some(v) = model_defaults.min_p {
        sampling_options.entry("min_p".to_owned()).or_insert(serde_json::json!(v));
    }
    if let Some(v) = model_defaults.repetition_penalty {
        if !sampling_options.contains_key("repeat_penalty") {
            sampling_options.insert("repeat_penalty".to_owned(), serde_json::json!(v));
        }
    }
    let sampling_defaults = fllm_engine::parameters::parse_sampling(
        &serde_json::Value::Object(sampling_options),
        Default::default(),
        tokenizer.vocab_size(),
    )?;
    let eos_ids = tokenizer
        .eos_ids()
        .iter()
        .map(|id| i64::from(*id))
        .collect();
    if let Some(ref cores) = pin_cores {
        pin_threads_to_cores(cores);
    }
    // --ctx auto: the model's trained context, capped where it cannot be planned. A GPU run starts from the trained
    // context (at most 32768; the block tables grow with it) and the memory planner below lowers it to what fits; on the
    // CPU nothing plans the KV cache, so the cap is 8192.
    let on_gpu = plan_device.as_ref().is_some_and(|d| d.dev_type != DeviceType::Cpu)
        || multi_devices.as_ref().is_some_and(|list| list.iter().any(|(backend, _)| backend != "cpu"));
    // every backend that takes part (the split runs on several)
    let run_backends: Vec<String> = multi_devices.as_ref().map_or_else(|| vec![selected_backend.clone()], |list| list.iter().map(|(b, _)| b.clone()).collect());
    let (mut context, context_origin) = match context_request {
        Some(n) => (n, "--ctx".to_owned()),
        None => {
            let cap = if on_gpu { 32768 } else { 8192 };
            let trained = facts.trained_context;
            (if trained == 0 { cap.min(8192) } else { trained.min(cap) }, format!("auto: trained context {}, cap {cap}", if trained == 0 { "unknown".to_owned() } else { trained.to_string() }))
        }
    };
    let requested_context = context;
    if max_seqs_auto {
        // GPU pools are cut to the device's memory below; the CPU keeps the default pool
        max_sequences = if matches!(selected_backend.as_str(), "cpu" | "npu") { 16 } else { 256 };
    }
    let mut model = if direct_load {
        GgufModel::from_mapping(mapping.with_tensor_aliases(&[])?, context, max_sequences, threads)?
    } else {
        GgufModel::load(path.to_str().unwrap(), context, max_sequences, threads)?
    };
    {
        let topology = fllm_engine::cpu_topology::detect();
        let choice = if !threads_auto {
            if selected_backend == "cpu" && pin_cores.is_none() {
                fllm_engine::cpu_threads::place_explicit(threads as usize);
            }
            fllm_engine::cpu_threads::ThreadChoice { threads: threads as usize, prefill: threads as usize, policy: topology.default_threads(), source: "flag", detail: "--threads".into() }
        } else if selected_backend == "cpu" && pin_cores.is_none() {
            fllm_engine::cpu_threads::tune(model.mapping(), model.hidden_size())
        } else {
            fllm_engine::cpu_threads::ThreadChoice { threads: threads as usize, prefill: threads as usize, policy: threads as usize, source: "policy", detail: "not running on the CPU, or --pin-cores given".into() }
        };
        threads = choice.threads as i32;
        model.set_threads(threads);
        model.set_prefill_threads(choice.prefill as i32);
        let prompts = if choice.prefill > choice.threads { format!("; {} for prompts", choice.prefill) } else { String::new() };
        tracing::info!("threads: {}{prompts} [{}: {}]", fllm_engine::cpu_threads::summary(choice.threads, topology), choice.source, choice.detail);
        fllm_engine::cpu_threads::record(choice);
    }
    if context_request.is_none() {
        // --ctx auto lets a single sequence grow to the trained length; it does not enlarge the memory: the KV pool keeps the
        // size of the former default (max-seqs x 4096 tokens), never below one full-length sequence
        let former_default = (max_sequences * 4096).div_ceil(model.block_size());
        if model.pool_blocks() > former_default {
            model.set_pool_blocks(former_default);
        }
    }
    for (adapter, scale) in lora_paths {
        let count = model.apply_lora(&adapter, scale)?;
        tracing::info!("Merged LoRA {adapter} at scale {scale}: {count} F32 projections");
    }
    let parse_format = |flag: &str, name: &Option<String>| -> Result<KvFormat, String> {
        match name {
            None => Ok(KvFormat::F16),
            Some(n) => KvFormat::parse(n)
                .ok_or_else(|| format!("unknown KV cache type '{n}' for {flag} (supported: auto, {})", KvFormat::NAMES)),
        }
    };
    let kv_both = kv_cache_type.clone();
    let key_format = parse_format("--cache-type-k", &kv_cache_type_k.clone().or_else(|| kv_both.clone()))?;
    let value_format = parse_format("--cache-type-v", &kv_cache_type_v.clone().or(kv_both))?;
    let kv_description = if key_format == value_format {
        key_format.name().to_owned()
    } else {
        format!("keys {} / values {}", key_format.name(), value_format.name())
    };
    if run_backends.iter().any(|b| matches!(b.as_str(), "vulkan" | "dx12" | "metal" | "opengl" | "gl")) && !(key_format == value_format && key_format.is_legacy()) {
        return Err(format!(
            "the {device} backend only supports the f16 and q8_0 KV cache (requested {kv_description}); use --device cuda or --device cpu for the other formats"
        )
        .into());
    }
    if key_format != value_format || !key_format.is_legacy() {
        let (kh, hd, vd) = (model.config().kv_heads, model.config().head_dim, model.config().value_dim);
        let ratio = (key_format.bytes_per_slot(kh, hd) + value_format.bytes_per_slot(kh, vd)) as f64
            / (KvFormat::F16.bytes_per_slot(kh, hd) + KvFormat::F16.bytes_per_slot(kh, vd)) as f64;
        tracing::info!("Activated {kv_description} KV cache ({:.0}% of the FP16 KV memory)", ratio * 100.0);
        for format in [key_format, value_format] {
            if let Some(note) = format.experimental_note() {
                tracing::warn!("{} KV cache is experimental: {note}", format.name());
            }
        }
        if let Some(note) = key_format.key_accuracy_note() {
            tracing::warn!("{} keys: {note}", key_format.name());
        }
    } else if key_format == KvFormat::F16 {
        tracing::info!("Using standard FP16 KV cache");
    } else {
        tracing::info!("Activated quantized Q8_0 KV cache (47% memory reduction)");
    }
    model.set_kv_formats(key_format, value_format);
    // FLLM_MLA=latent: cache the compressed latents of multi-head latent attention instead of the rebuilt per-head keys and values (CUDA only)
    if model.config().mla.is_some() && fllm_engine::mla_latent::mode_from_env() == fllm_engine::mla_latent::MlaMode::Latent && multi_devices.is_some() {
        eprintln!("[fllm-rs] warning: FLLM_MLA=latent is not used with several devices; keeping the expanded MLA cache");
    } else if model.config().mla.is_some() && fllm_engine::mla_latent::mode_from_env() == fllm_engine::mla_latent::MlaMode::Latent {
        match latent_mla_unavailable(&model, &selected_backend, selected_index, key_format, value_format) {
            None => {
                model.enable_mla_latent()?;
                eprintln!("[fllm-rs] Activated the latent MLA cache (FLLM_MLA=latent): {} values per token and layer, pages of {} tokens", model.config().head_dim, model.block_size());
            }
            Some(why) => eprintln!("[fllm-rs] warning: FLLM_MLA=latent is not used ({why}); keeping the expanded MLA cache"),
        }
    }
    if speculative_lookup > 0 {
        tracing::info!(
            "Activated speculative lookahead (k={}, window={}; stochastic and grammar-constrained requests speculate too)",
            speculative_lookup, speculative_window
        );
    }
    // A draft model lives on the same device as the target and has its own weights and KV pool: take its memory out of the budget
    // the target's plan may use.
    let draft_path = match &draft_model {
        Some(spec) => Some(resolve_gguf(spec, hub_offline)?),
        None => None,
    };
    let draft_reserve: u64 = if device == "cpu" {
        0
    } else {
        draft_path
            .as_ref()
            .and_then(|p| std::fs::metadata(p).ok())
            .map_or(0, |m| m.len() + m.len() / 16 + (256 << 20))
    };
    if let Some(list) = &multi_devices {
        // Split the layers over the listed devices in proportion to their memory and speed; the KV pool is one block table that every
        // stage holds for its own layers, so it has as many blocks as the smallest stage can hold.
        let mut planning = Vec::new();
        for (n, (backend, index)) in list.iter().enumerate() {
            let mut d = fllm_engine::devices::planning_device(backend, *index, gpu_memory_utilization)?;
            if n == 0 {
                d.budget_bytes = d.budget_bytes.saturating_sub(draft_reserve);
            }
            planning.push(d);
        }
        // a tensor-parallel group of `tp` devices is one pipeline stage: equal shards, so it holds `tp` times the smallest budget and reads
        // `tp` times as fast as its slowest member
        let tp = parallel.tp.max(1);
        let devices: Vec<fllm_engine::backend::memory::PipelineDevice> = planning
            .chunks(tp)
            .enumerate()
            .map(|(stage, group)| {
                let speed_of = |k: usize, d: &fllm_engine::devices::PlanningDevice| parallel.split_ratio.get(stage * tp + k).copied().unwrap_or(d.speed_gbs);
                fllm_engine::backend::memory::PipelineDevice {
                    label: group.iter().map(|d| d.label.clone()).collect::<Vec<_>>().join(" + "),
                    budget_bytes: group.iter().map(|d| d.budget_bytes).min().unwrap_or(0) * group.len() as u64,
                    speed: group.iter().enumerate().map(|(k, d)| speed_of(k, d)).fold(f64::MAX, f64::min) * group.len() as f64,
                }
            })
            .collect();
        for d in &planning {
            tracing::info!(
                "  device {}: {:.2} GB of {:.2} GB usable | read speed {:.0} GB/s ({})",
                d.label,
                d.budget_bytes as f64 / 1e9,
                d.total_bytes as f64 / 1e9,
                d.speed_gbs,
                if d.speed_known { "reported or measured" } else { "estimate for this kind of device" }
            );
        }
        let plan = fllm_engine::backend::memory::plan_pipeline(&model, &devices, max_batched_tokens)?;
        if plan.context_blocks > 0 {
            let fit = plan.context_blocks * model.block_size();
            tracing::warn!(
                "a context of {} tokens needs more KV cache than the devices have left; using {fit} tokens (--ctx; --kv-cache-type q8_0 or fp8_e4m3 roughly doubles it)",
                model.max_context()
            );
            model.limit_context(fit);
            context = model.max_context();
        }
        if plan.kv_blocks < model.pool_blocks() {
            tracing::info!("KV pool limited to {} of {} blocks to fit the devices ({} tokens)", plan.kv_blocks, model.pool_blocks(), plan.kv_blocks * model.block_size());
            model.set_pool_blocks(plan.kv_blocks);
        }
        tracing::info!(
            "Layer split over {} devices: shared KV pool of {} blocks ({} tokens); each stage holds the cache of its own layers",
            plan.stages.len(),
            plan.kv_blocks,
            plan.kv_blocks * model.block_size()
        );
        for (n, stage) in plan.stages.iter().enumerate() {
            tracing::info!(
                "  stage {n}: {} layers {}..{} of {} | weights {:.2} GB{} | KV cache {:.2} GB | scratch {:.0} MB | budget {:.2} GB | weight on this stage / speed {:.0}{}",
                devices[stage.device].label,
                stage.layers.start,
                stage.layers.end,
                plan.total_layers,
                (stage.weight_bytes + stage.head_bytes) as f64 / 1e9,
                if stage.head_bytes > 0 { format!(" (output head {:.2} GB)", stage.head_bytes as f64 / 1e9) } else { String::new() },
                stage.kv_cache_bytes as f64 / 1e9,
                stage.scratchpad_bytes as f64 / 1e6,
                stage.budget_bytes as f64 / 1e9,
                devices[stage.device].speed,
                if parallel.split_ratio.is_empty() { " GB/s" } else { " (--tensor-split)" }
            );
        }
        parallel.stage_ranges = plan.ranges();
        parallel.validate().map_err(|e| format!("layer split: {e}"))?;
        fllm_engine::parallel::config::set_global(parallel.clone());
        #[cfg(feature = "cuda")]
        fllm_engine::backend::cuda::set_marlin_q4k_enabled(false);
    } else if device != "cpu" {
        // Size the KV pool to the device: every sequence gets its full context only if it fits
        let target_dev = plan_device.as_ref();
        if let Some(d) = target_dev {
            if d.dev_type != DeviceType::Cpu {
                // an integrated GPU has no memory of its own: what it may use is what the system has left, not what the driver reports
                let unified = d.dev_type == DeviceType::IntegratedGpu;
                // (FLLM_NO_HOST_MEMORY_CLAMP=1 trusts the device's own figure: on a machine whose BIOS or driver reserves a large block of the RAM for the GPU,
                // that block is not part of the memory the system still has free, and clamping to the system's figure would hide it)
                let host_clamp = unified && !std::env::var("FLLM_NO_HOST_MEMORY_CLAMP").is_ok_and(|v| !v.is_empty() && v != "0");
                let device_free = if host_clamp { d.free_memory_bytes.min(fllm_engine::devices::host_free_memory_bytes()) } else { d.free_memory_bytes };
                let plan_for = |m: &GgufModel, tokens: usize| {
                    fllm_engine::backend::memory::MemoryPlan::plan_for_backend(
                        m,
                        device_free.saturating_sub(draft_reserve),
                        d.total_memory_bytes,
                        gpu_memory_utilization,
                        Some(&gpu_layers),
                        tokens,
                        selected_backend == "cuda",
                    )
                };
                let mut planned = plan_for(&model, max_batched_tokens);
                if let Err(e) = &planned {
                    tracing::warn!("memory plan for the device failed: {e}");
                }
                if let Ok(p) = &planned {
                    // the pool must hold one sequence of the full context: otherwise lower the context instead of failing
                    // (and with an automatic context, also when the full context would only fit next to some layers whose feed-forward weights
                    // then stream from the host: a window of 4096 tokens or more with every layer on the device decodes two to three times faster;
                    // `--ctx N` keeps the longer window and the offload)
                    let speed_first = context_request.is_none() && p.offload_layers < p.total_layers && p.all_on_device_kv_blocks >= 256;
                    if p.kv_block_count < model.table_stride() || speed_first {
                        // keep every layer on the device (decode speed) when that still leaves a usable window; when not even that fits, a context
                        // of 8192 tokens at most (512 blocks of 16): every block above it costs feed-forward weights that then stream from the host,
                        // and a model that is too big for the device is slow enough already (--ctx asks for more)
                        let blocks = if p.all_on_device_kv_blocks >= 256 { p.all_on_device_kv_blocks } else { p.kv_block_count.min(512) };
                        let fit = blocks * model.block_size();
                        if speed_first && p.kv_block_count >= model.table_stride() {
                            tracing::info!(
                                "context lowered from {} to {} tokens so that every layer stays on the device (decode speed); --ctx {} keeps the long window and moves the feed-forward weights of some layers to host memory",
                                model.max_context(),
                                fit,
                                model.max_context()
                            );
                        } else {
                            tracing::warn!(
                                "a context of {} tokens needs more KV cache than the device has left; using {} tokens (--ctx; --kv-cache-type q8_0 or fp8_e4m3 roughly doubles it)",
                                model.max_context(),
                                fit
                            );
                        }
                        model.limit_context(fit);
                        context = model.max_context();
                        planned = plan_for(&model, max_batched_tokens);
                    }
                }
                if let Ok(first) = planned {
                    // weights in host memory are streamed to the device per batch: bigger batches amortise the transfer
                    // (the activations of such a batch are part of the plan)
                    // (an integrated GPU reads host memory directly: nothing is streamed, so the large batches would only cost scratch memory)
                    let plan = if first.offload_layers < first.total_layers && !batch_explicit && !unified && max_batched_tokens < 2048 {
                        max_batched_tokens = 2048;
                        plan_for(&model, max_batched_tokens).unwrap_or(first)
                    } else if !batch_explicit && max_batched_tokens < 1024 {
                        // prompt chunks of 1024 tokens fill the GEMM tiles better (NVFP4 prompts +8 %); a discrete card that serves
                        // several sequences takes 2048 (the prompts of several users need half the steps, as with vLLM's default
                        // budget); either only if the plan stays the same apart from a little less KV cache
                        let keeps = |bigger: &fllm_engine::backend::memory::MemoryPlan| {
                            bigger.offload_layers == first.offload_layers && bigger.kv_block_count * 10 >= first.kv_block_count * 9
                        };
                        let wide = (!unified && max_sequences > 1).then(|| plan_for(&model, 2048).ok()).flatten().filter(|b| keeps(b));
                        match wide {
                            Some(bigger) => {
                                max_batched_tokens = 2048;
                                bigger
                            }
                            None => match plan_for(&model, 1024) {
                                Ok(bigger) if keeps(&bigger) => {
                                    max_batched_tokens = 1024;
                                    bigger
                                }
                                _ => first,
                            },
                        }
                    } else {
                        first
                    };
                    if plan.kv_block_count < model.pool_blocks() {
                        tracing::info!(
                            "KV pool limited to {} of {} blocks to fit the device ({} tokens)",
                            plan.kv_block_count,
                            model.pool_blocks(),
                            plan.kv_block_count * model.block_size()
                        );
                        model.set_pool_blocks(plan.kv_block_count);
                    }
                    if plan.offload_layers < plan.total_layers {
                        tracing::info!(
                            "{} of {} layers fit on the device completely; the feed-forward weights of the other {} stay in host memory",
                            plan.offload_layers,
                            plan.total_layers,
                            plan.total_layers - plan.offload_layers
                        );
                        fllm_engine::rust_decoder::set_ffn_host_from(plan.total_layers, plan.offload_layers);
                    }
                }
            }
        }
    }
    if max_seqs_auto {
        // as many sequences as the KV pool can hold at a typical 512 tokens each, at most 256
        let pool_tokens = model.pool_blocks() * model.block_size();
        max_sequences = (pool_tokens / 512).clamp(1, model.max_sequences().min(256));
        eprintln!("[fllm-rs] --max-seqs auto: {max_sequences} sequences ({pool_tokens} KV tokens in the pool)");
    }
    let runtime_pool_blocks = model.pool_blocks();
    let mut runtime = InferenceRuntime::with_speculative(
        model,
        max_sequences,
        max_batched_tokens,
        eos_ids,
        speculative_lookup,
        speculative_window,
    )?;
    {
        let cfg = runtime.scheduler_config_mut();
        cfg.policy = scheduling_policy;
        cfg.max_partial_prefills = max_partial_prefills;
        cfg.long_prefill_threshold = long_prefill_threshold;
        cfg.max_long_partial_prefills = max_long_partial_prefills;
        cfg.decode_first = decode_first;
        // keep about 1 % of the pool free for the running sequences to grow into (vLLM's default watermark)
        cfg.watermark_blocks = sched_watermark.unwrap_or_else(|| (runtime_pool_blocks / 100).min(64));
        if preemption_swap {
            cfg.preemption = "swap".into();
        }
        eprintln!(
            "[fllm-rs] Scheduler: policy {:?}, prefill {}, partial prefills {}, watermark {} blocks",
            cfg.policy,
            if cfg.decode_first { "decode-first" } else { "prefill-first" },
            if cfg.max_partial_prefills == 0 { "unlimited".to_owned() } else { cfg.max_partial_prefills.to_string() },
            cfg.watermark_blocks
        );
    }
    if speculative_mtp {
        if draft_model.is_some() || speculative_lookup > 0 {
            return Err(
                "--speculative mtp cannot be combined with a draft model or lookup speculation"
                    .into(),
            );
        }
        runtime.enable_mtp()?;
        tracing::info!("MTP enabled: first NextN block, one target-verified candidate; prefix reuse disabled");
    }
    if let Some(path) = draft_path {
        let draft = GgufModel::load(path.to_str().ok_or("invalid draft model path")?, context, max_sequences, threads)?;
        runtime.enable_draft(draft, draft_tokens)?;
        eprintln!(
            "[fllm-rs] Draft-model speculation enabled: {} ({draft_tokens} candidate tokens per step, adaptive length unless FLLM_SPEC_ADAPTIVE=0, same device as the target)",
            path.display()
        );
    }
    // Conversation lookup is on unless the user chose something else: it costs nothing when it finds no match and about doubles the speed on
    // text that repeats (code edits, summaries, quotes, agent loops), with the same text as without. Only where it has been run: the
    // processor and the Vulkan / DirectX chips (the CUDA, HIP and Metal verification kernels have not been run on hardware yet), and not
    // for models with a recurrent state. `--speculative off` or FLLM_SPECULATIVE_DEFAULT=off keeps it away.
    if speculative_method.is_none()
        && speculative_lookup == 0
        && !speculative_mtp
        && draft_model.is_none()
        && !parallel.is_parallel()
        && !speculative_suffix
        && !runtime.model.is_stateful()
        && matches!(selected_backend.as_str(), "cpu" | "vulkan" | "dx12" | "wgpu-vulkan" | "wgpu-dx12" | "wgpu")
        && !env::var("FLLM_SPECULATIVE_DEFAULT").is_ok_and(|v| matches!(v.trim(), "off" | "0" | "false"))
    {
        speculative_suffix = true;
    }
    if speculative_suffix {
        let tokens = num_speculative_tokens.unwrap_or(6);
        runtime.enable_suffix(
            tokens,
            fllm_engine::speculative::SuffixConfig { min_match: speculative_min_match, token_budget: speculative_cache_tokens },
        )?;
        eprintln!("[fllm-rs] Suffix speculation enabled: up to {tokens} tokens, matches of at least {speculative_min_match} tokens across the conversation and recent generations (cache {speculative_cache_tokens} tokens)");
    }
    if device != "cpu" && multi_devices.is_none() {
        let target_dev = plan_device.as_ref();
        if let Some(d) = target_dev {
            if let Ok(plan) = fllm_engine::backend::memory::MemoryPlan::plan_for_backend(
                &runtime.model,
                d.free_memory_bytes,
                d.total_memory_bytes,
                gpu_memory_utilization,
                Some(&gpu_layers),
                max_batched_tokens,
                selected_backend == "cuda",
            ) {
                #[cfg(feature = "cuda")]
                if selected_backend == "cuda" {
                    fllm_engine::backend::cuda::set_marlin_q4k_enabled(plan.marlin_q4k_dual);
                }
                tracing::info!(
                    "Memory plan: target device [{}] {} (budget: {:.2} GB / {:.2} GB total, util: {:.0}%)",
                    d.index,
                    d.name,
                    plan.budget_bytes as f64 / 1e9,
                    plan.total_vram_bytes as f64 / 1e9,
                    plan.utilization * 100.0
                );
                tracing::info!(
                    "Memory split: {}/{} layers offloaded | Model weights: {:.2} GB{} | KV Cache: {:.2} GB ({} blocks) | Scratchpad: {:.2} MB",
                    plan.offload_layers,
                    plan.total_layers,
                    plan.estimated_model_bytes as f64 / 1e9,
                    if plan.marlin_q4k_dual { " (Q4_K dual Marlin enabled)" } else { "" },
                    plan.kv_cache_bytes as f64 / 1e9,
                    plan.kv_block_count,
                    plan.scratchpad_bytes as f64 / 1e6
                );
            }
        }
    }
    runtime.attach_compute(&device)?;
    if multi_step > 1 {
        let effective = runtime.set_multi_step(multi_step)?;
        if effective == multi_step {
            eprintln!("[fllm-rs] Multi-step scheduling: up to {effective} decode rounds per scheduler round (plain decoding only; speculation, grammars, media and beam search use single steps)");
        } else {
            eprintln!("[fllm-rs] --multi-step {multi_step} is ignored on this backend: it has only been verified on the CPU backend (set FLLM_MULTI_STEP_UNVERIFIED=1 to use it anyway)");
        }
    }
    if preemption_swap {
        let c = runtime.model.config();
        let (key_fmt, value_fmt) = runtime.model.kv_formats();
        let block_bytes = c.layers * runtime.model.block_size() * (key_fmt.bytes_per_slot(c.kv_heads, c.head_dim) + value_fmt.bytes_per_slot(c.kv_heads, c.value_dim)) as usize;
        let blocks = ((swap_space_gib * (1u64 << 30) as f64) / block_bytes.max(1) as f64) as usize;
        if !runtime.kv_block_ops_available() {
            eprintln!("[fllm-rs] --preemption-mode swap: this backend cannot copy KV blocks (the device self check failed or the operation is missing); preempted sequences are recomputed");
            runtime.scheduler_config_mut().preemption = "recompute".into();
        } else if blocks == 0 {
            eprintln!("[fllm-rs] --preemption-mode swap needs --swap-space GiB; preempted sequences are recomputed");
            runtime.scheduler_config_mut().preemption = "recompute".into();
        } else {
            runtime.scheduler_config_mut().swap_blocks = blocks;
            eprintln!("[fllm-rs] Preemption by swap: {blocks} blocks ({swap_space_gib} GiB) of host memory for preempted sequences");
        }
    }
    #[cfg(feature = "cuda")]
    if selected_backend == "cuda" {
        let ms = fllm_engine::backend::cuda::total_repack_ms();
        if ms > 0.0 {
            tracing::info!("Q4_K Marlin CPU repacking total time: {:.1} ms ({:.3} s)", ms, ms / 1000.0);
        }
    }
    tracing::info!("Rust decoder forward pass enabled (packed quantized weights, paged {kv_description} KV)");
    fllm_engine::vision::preprocess::MediaLimits::set(media_limits);
    if mmproj.is_none() && !no_vision {
        // a GGUF written by `fllm-rs convert` carries the vision tower of a multimodal checkpoint
        // the tower runs on the CPU unless a wgpu build can take its projections (CUDA builds keep it on the host)
        let tower = match fllm_engine::vision::load_from_model(runtime.model.mapping(), threads.max(1) as usize, &device) {
            Err(error) if !matches!(device.as_str(), "cpu" | "auto") => {
                eprintln!("[fllm-rs] vision tower on {device} unavailable ({error}); running it on the CPU");
                fllm_engine::vision::load_from_model(runtime.model.mapping(), threads.max(1) as usize, "cpu")?
            }
            other => other?,
        };
        if let Some(vision) = tower {
            if vision.output_hidden() != runtime.model.hidden_size() {
                return Err("the vision tower in the model file does not match its language model".into());
            }
            eprintln!("[fllm-rs] Own Rust {} vision encoder enabled (tower from the model file; --no-vision turns it off)", vision.family());
            runtime.vision = Some(vision);
        }
    }
    if let Some(projector) = mmproj {
        let vision =
            fllm_engine::vision::VisionEncoder::load(&projector, threads.max(1) as usize, &device)?;
        if vision.output_hidden != runtime.model.hidden_size()
            || !runtime.model.chat_template().contains("<image>")
        {
            return Err("projector does not match this multimodal decoder/template".into());
        }
        tracing::info!(
            "Own Rust SigLIP/Idefics3 vision encoder enabled: {} image tokens",
            vision.tokens
        );
        runtime.vision = Some(std::sync::Arc::new(vision));
    }
    let context_note = if context != requested_context { format!("{context_origin}; lowered from {requested_context} to fit the device memory") } else { context_origin };
    tracing::info!(
        "model={} context={} ({}) max_sequences={} max_batched_tokens={} threads={}",
        inferred_name, context, context_note, max_sequences, max_batched_tokens, threads
    );
    let status_model = model_name.clone().unwrap_or_else(|| inferred_name.clone());
    let status_dir = fllm_engine::hub::status_dir();
    let _ = std::fs::create_dir_all(&status_dir);
    let status_path = status_dir.join(format!("{}.json", std::process::id()));
    let _ = std::fs::write(
        &status_path,
        format!(
            "{{\"pid\":{},\"port\":{},\"host\":\"{}\",\"model\":\"{}\"}}",
            std::process::id(),
            port,
            host,
            status_model.replace('\\', "\\\\").replace('"', "\\\"")
        ),
    );
    let _status_guard = StatusGuard { path: status_path };
    serve(
        runtime,
        tokenizer,
        ServerConfig {
            host,
            port,
            model_name: model_name.unwrap_or(inferred_name),
            http,
            facts,
            default_temperature,
            default_top_p,
            default_top_k,
            sampling_defaults,
            tool_parser,
            reasoning_parser,
            jump_forward,
        },
    )?;
    Ok(())
}

/// `fllm-rs npu-setup`: downloads the pinned OpenVINO runtime (checked against its SHA-256) and unpacks it into `~/.fllm/openvino`
/// (`$FLLM_HOME/openvino`), where the NPU backend looks for it. `fllm-rs npu-setup --remove` deletes it again. The same code runs
/// behind `POST /api/studio/npu/setup`.
fn npu_setup(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    use fllm_engine::backend::npu::setup;
    match args {
        [] => {}
        [flag] if flag == "--remove" => {
            setup::remove()?;
            println!("The OpenVINO runtime was removed.");
            return Ok(());
        }
        _ => return Err("usage: fllm-rs npu-setup [--remove]".into()),
    }
    setup::start()?;
    let mut last = String::new();
    loop {
        std::thread::sleep(std::time::Duration::from_millis(400));
        let p = setup::progress();
        let line = if p.total > 0 {
            format!("{}: {:.1} of {:.1} MB", p.message, p.done as f64 / 1e6, p.total as f64 / 1e6)
        } else {
            p.message.clone()
        };
        if line != last {
            eprint!("\r{line}   ");
            last = line;
        }
        if !p.phase.busy() {
            eprintln!();
            return match p.phase {
                setup::Phase::Done => {
                    println!("OpenVINO {} is installed in {}; `fllm-rs devices` lists the NPU when the driver provides one.", setup::OPENVINO_VERSION, setup::install_dir().map(|d| d.display().to_string()).unwrap_or_default());
                    if cfg!(target_os = "linux") {
                        println!("The NPU driver (intel_vpu kernel module and its user-space part) comes from your distribution; the user needs access to /dev/accel (group render).");
                    }
                    Ok(())
                }
                setup::Phase::Cancelled => Err("cancelled".into()),
                _ => Err(p.error.unwrap_or_else(|| "the setup failed".to_owned()).into()),
            };
        }
    }
}
