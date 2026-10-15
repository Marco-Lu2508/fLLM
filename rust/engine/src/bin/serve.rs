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
        eprintln!("[fllm-rs] Pinned inference process to CPU affinity mask: {spec_trimmed}");
    }
}
#[cfg(not(target_os = "linux"))]
fn pin_threads_to_cores(_spec: &str) {}
use fllm_engine::{
    compiled_backends, kv_format::KvFormat,
    server::{serve, ServerConfig},
    tokenizer::TokenizerBundle,
    DeviceType, GgufModel, InferenceRuntime,
};
use fllm_gguf_probe::MappedGguf;
use std::{env, process, thread};

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
    eprintln!("usage: fllm-rs [serve] --model MODEL [--quant TYPE] [--outtype preserve|f16|f32] [--device auto|cpu|cuda|hip|metal|vulkan|dx12|sycl|npu|tpu|BACKEND:N] [--gpu-memory-utilization 0.90] [--gpu-layers auto|N] [--kv-cache-type TYPE | --kv-cache-dtype TYPE] [--cache-type-k TYPE] [--cache-type-v TYPE] [--speculative mtp|off] [--speculative-lookup N] [--speculative-window W] [--pin-cores N] [--host 127.0.0.1] [--port 8000] [--ctx 4096] [--max-seqs 16] [--max-batched-tokens 1024] [--threads N] [--api-key KEY] [--model-name NAME] [--tool-parser auto|ID] [--reasoning-parser auto|ID] [--no-repack] [--temp T] [--top-p P] [--top-k K]");
    eprintln!("       --device: BACKEND:N counts the devices of that backend from 0 (vulkan:1 = second Vulkan device; `fllm-rs devices` prints the value to use); auto = best GPU (CUDA > ROCm > Metal > Vulkan > DX12 > SYCL), else CPU");
    eprintln!("       MODEL: a .gguf file, a Hugging Face link or owner/repo[:QUANT], or a checkpoint (safetensors / PyTorch / AWQ / GPTQ / FP8 / ModelOpt NVFP4) as repository or directory: a checkpoint is converted to GGUF once and cached; --quant ({}) re-encodes an unquantized checkpoint while converting", fllm_checkpoint::Preset::NAMES);
    eprintln!("       KV cache TYPE (keys and values separately with --cache-type-k/-v): auto|{}", KvFormat::NAMES);
    eprintln!("       (wgpu devices: f16 and q8_0 only; nvfp4 is experimental, keys below 8 bit can break some models, see docs/gpu-backends.md)");
    eprintln!("       fllm-rs devices [--json] [--measure] | formats | inspect PATH | pull hf://org/repo/file.gguf");
    eprintln!("       fllm-rs selftest-cpu [--tier TIER] | selftest-gpu [--device auto|vulkan|metal|hip|...] [--only GROUPS] [--quants LIST] [--json FILE] [--list]");
    eprintln!("       fllm-rs convert CHECKPOINT OUTPUT.gguf [--outtype preserve|f16|f32]");
    eprintln!("       fllm-rs chat | bench --url http://127.0.0.1:8000 [--model MODEL_ID]");
    eprintln!(
        "       sampling: --min-p P --typical-p P --mirostat 0|1|2 --dry-multiplier M --seed S"
    );
    eprintln!("       adapters/draft: --lora FILE --lora-scaled FILE SCALE --draft-model FILE --draft-tokens N");
    eprintln!("       Rust encoder: --model BERT.gguf --embeddings [--pooling cls|mean|last]");
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
    println!("`--device auto` takes the first of CUDA > ROCm > Metal > Vulkan > DX12 > SYCL > CPU that has a GPU (FLLM_AUTO_ORDER changes the order,");
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
            "compiled_backends": compiled_backends(),
            "devices": devices,
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
        println!(
            "  [{}] {} ({})\n      Backend:      {}\n      Select:       --device {}\n      Type:         {}\n      Memory:       {} MB free / {} MB total\n      Features:     {}\n      Capabilities: {}\n      Recommended:  {}",
            dev.index,
            dev.name,
            dev.description,
            if entry.api == Api::Host { "cpu".to_owned() } else { format!("{} ({})", entry.backend, entry.api.as_str()) },
            entry.spec,
            dev.dev_type.as_str(),
            free_mb,
            total_mb,
            caps_of(entry).summary(),
            capabilities,
            role,
        );
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
fn plan_target(selected_backend: &str, selected_index: usize) -> Option<fllm_engine::DeviceInfo> {
    if selected_backend == "cpu" {
        return None;
    }
    fllm_engine::devices::find_entry(selected_backend, selected_index).map(|entry| entry.info)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if fllm_engine::cli::dispatch(&args)? {
        return Ok(());
    }
    if args.is_empty() {
        usage("");
    }
    if args.first().map(String::as_str) == Some("devices") {
        return print_devices(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("selftest-gpu") {
        return Ok(fllm_engine::backend::selftest::run(&args[1..])?);
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
    let mut host = "127.0.0.1".to_owned();
    let mut port = 8000u16;
    let mut context = 4096usize;
    let mut max_sequences = 16usize;
    let mut max_batched_tokens = 512usize;
    let mut batch_explicit = false;
    let mut threads = thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(1);
    let mut api_key = env::var("FLLM_API_KEY").ok();
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
    let mut lora_paths = Vec::new();
    let mut sampling_options = serde_json::Map::new();
    let mut pin_cores = None;
    let mut embeddings = false;
    let mut pooling = None;
    let mut mmproj = None;

    while i < args.len() {
        match args[i].as_str() {
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
            "--model" => model_path = Some(value(&args, &mut i, "--model")),
            "--engine" => {
                let engine = value(&args, &mut i, "--engine");
                if !matches!(engine.as_str(), "auto" | "rust") {
                    usage("only the independent Rust engine is supported");
                }
            }
            "--embeddings" => embeddings = true,
            "--mmproj" => mmproj = Some(value(&args, &mut i, "--mmproj")),
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
                context = value(&args, &mut i, &flag)
                    .parse()
                    .unwrap_or_else(|_| usage("invalid context length"))
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
                threads = value(&args, &mut i, "--threads")
                    .parse()
                    .unwrap_or_else(|_| usage("invalid --threads"))
            }
            "--device" => device = value(&args, &mut i, "--device").to_lowercase(),
            "--gpu-memory-utilization" => {
                gpu_memory_utilization = value(&args, &mut i, "--gpu-memory-utilization")
                    .parse()
                    .unwrap_or_else(|_| usage("invalid --gpu-memory-utilization"));
            }
            "--gpu-layers" | "-ngl" | "--n-gpu-layers" => {
                let flag = args[i].clone();
                gpu_layers = value(&args, &mut i, &flag);
            }
            "--api-key" => api_key = Some(value(&args, &mut i, "--api-key")),
            "--no-repack" => fllm_engine::cpu_kernels::set_repack(false),
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
                    "mtp" => speculative_mtp = true,
                    "off" => {
                        speculative_mtp = false;
                        speculative_lookup = 0;
                    }
                    _ => {
                        speculative_lookup = source.parse().unwrap_or_else(|_| {
                            usage("--speculative expects mtp, off or a lookup length")
                        })
                    }
                }
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
            other => usage(&format!("unknown option: {other}")),
        }
        i += 1;
    }
    let model_path = model_path.unwrap_or_else(|| usage("--model is required"));
    let mut path = fllm_engine::import::prepare(
        &model_path,
        &fllm_engine::import::ImportOptions {
            revision: hub_revision.clone(),
            offline: hub_offline,
            quant: import_quant,
            outtype: import_outtype,
        },
    )?;
    if path.is_dir() {
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
            eprintln!(
                "[fllm-rs] Resolved checkpoint directory '{}' to GGUF model '{}'",
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
    let inferred_name = path
        .file_stem()
        .and_then(|x| x.to_str())
        .map(fllm_engine::import::display_name)
        .unwrap_or("fllm-model")
        .to_owned();

    let mapping = MappedGguf::open(&path)?;
    let is_encoder = mapping
        .index()
        .metadata
        .get("general.architecture")
        .and_then(serde_json::Value::as_str)
        == Some("bert");
    // fail early on a malformed or impossible --device, before the model is read; `vulkan:1` counts the devices of that backend
    let device_chain = fllm_engine::devices::device_candidates(&device)?;
    if device == "cpu" {
        eprintln!("[fllm-rs] Using explicit CPU runtime.");
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
        let mut encoder = fllm_engine::encoder::BertEncoder::load(
            path.to_str().ok_or("invalid model path")?,
            threads.max(1) as usize,
            pooling,
        )?;
        encoder.set_device(&device)?;
        eprintln!(
            "[fllm-rs] Independent Rust BERT encoder: {} dimensions, {} token context, pooling={}",
            encoder.hidden, encoder.context, encoder.default_pooling
        );
        return fllm_engine::encoder_server::serve(
            encoder,
            ServerConfig {
                host,
                port,
                model_name: model_name.unwrap_or(inferred_name),
                api_key,
                default_temperature,
                default_top_p,
                default_top_k,
                sampling_defaults: Default::default(),
                tool_parser: None,
                reasoning_parser: None,
            },
        )
        .map_err(Into::into);
    }
    if embeddings || pooling.is_some() {
        return Err("--embeddings/--pooling selects a dedicated Rust BERT encoder; decoder models already expose contextual embedding endpoints".into());
    }
    let (selected_backend, selected_index) = device_chain[0].clone();
    if fllm_engine::devices::is_auto(&device) {
        eprintln!("[fllm-rs] --device auto: {}", fllm_engine::devices::describe_chain(&device_chain));
    }
    let plan_device = plan_target(&selected_backend, selected_index);
    let tokenizer = TokenizerBundle::from_gguf(&mapping)?;
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
    let mut model = GgufModel::load(path.to_str().unwrap(), context, max_sequences, threads)?;
    for (adapter, scale) in lora_paths {
        let count = model.apply_lora(&adapter, scale)?;
        eprintln!("[fllm-rs] Merged LoRA {adapter} at scale {scale}: {count} F32 projections");
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
    if matches!(device.as_str(), "vulkan" | "dx12" | "metal") && !(key_format == value_format && key_format.is_legacy()) {
        return Err(format!(
            "the {device} backend only supports the f16 and q8_0 KV cache (requested {kv_description}); use --device cuda or --device cpu for the other formats"
        )
        .into());
    }
    if key_format != value_format || !key_format.is_legacy() {
        let (kh, hd, vd) = (model.config().kv_heads, model.config().head_dim, model.config().value_dim);
        let ratio = (key_format.bytes_per_slot(kh, hd) + value_format.bytes_per_slot(kh, vd)) as f64
            / (KvFormat::F16.bytes_per_slot(kh, hd) + KvFormat::F16.bytes_per_slot(kh, vd)) as f64;
        eprintln!("[fllm-rs] Activated {kv_description} KV cache ({:.0}% of the FP16 KV memory)", ratio * 100.0);
        for format in [key_format, value_format] {
            if let Some(note) = format.experimental_note() {
                eprintln!("[fllm-rs] WARNING: {} KV cache is experimental: {note}", format.name());
            }
        }
        if let Some(note) = key_format.key_accuracy_note() {
            eprintln!("[fllm-rs] WARNING: {} keys: {note}", key_format.name());
        }
    } else if key_format == KvFormat::F16 {
        eprintln!("[fllm-rs] Using standard FP16 KV cache");
    } else {
        eprintln!("[fllm-rs] Activated quantized Q8_0 KV cache (47% memory reduction)");
    }
    model.set_kv_formats(key_format, value_format);
    if speculative_lookup > 0 {
        eprintln!(
            "[fllm-rs] Activated speculative lookahead (k={}, window={})",
            speculative_lookup, speculative_window
        );
    }
    if device != "cpu" {
        // Size the KV pool to the device: every sequence gets its full context only if it fits
        let target_dev = plan_device.as_ref();
        if let Some(d) = target_dev {
            if d.dev_type != DeviceType::Cpu {
                let plan_for = |m: &GgufModel, tokens: usize| {
                    fllm_engine::backend::memory::MemoryPlan::plan(
                        m,
                        d.free_memory_bytes,
                        d.total_memory_bytes,
                        gpu_memory_utilization,
                        Some(&gpu_layers),
                        tokens,
                    )
                };
                let mut planned = plan_for(&model, max_batched_tokens);
                if let Ok(p) = &planned {
                    // the pool must hold one sequence of the full context: otherwise lower the context instead of failing
                    if p.kv_block_count < model.table_stride() {
                        // keep every layer on the device (decode speed) when that still leaves a usable window
                        let blocks = if p.all_on_device_kv_blocks >= 256 { p.all_on_device_kv_blocks } else { p.kv_block_count };
                        let fit = blocks * model.block_size();
                        eprintln!(
                            "[fllm-rs] warning: a context of {} tokens needs more KV cache than the device has left; using {} tokens (--ctx; --kv-cache-type q8_0 or fp8_e4m3 roughly doubles it)",
                            model.max_context(),
                            fit
                        );
                        model.limit_context(fit);
                        context = model.max_context();
                        planned = plan_for(&model, max_batched_tokens);
                    }
                }
                if let Ok(first) = planned {
                    // weights in host memory are streamed to the device per batch: bigger batches amortise the transfer
                    // (the activations of such a batch are part of the plan)
                    let plan = if first.offload_layers < first.total_layers && !batch_explicit && max_batched_tokens < 2048 {
                        max_batched_tokens = 2048;
                        plan_for(&model, max_batched_tokens).unwrap_or(first)
                    } else if !batch_explicit && max_batched_tokens < 1024 {
                        // prompt chunks of 1024 tokens fill the GEMM tiles better (NVFP4 prompts +8 %); keep them only if the
                        // plan stays the same apart from a little less KV cache
                        match plan_for(&model, 1024) {
                            Ok(bigger) if bigger.offload_layers == first.offload_layers && bigger.kv_block_count * 10 >= first.kv_block_count * 9 => {
                                max_batched_tokens = 1024;
                                bigger
                            }
                            _ => first,
                        }
                    } else {
                        first
                    };
                    if plan.kv_block_count < model.pool_blocks() {
                        eprintln!(
                            "[fllm-rs] KV pool limited to {} of {} blocks to fit the device ({} tokens)",
                            plan.kv_block_count,
                            model.pool_blocks(),
                            plan.kv_block_count * model.block_size()
                        );
                        model.set_pool_blocks(plan.kv_block_count);
                    }
                    if plan.offload_layers < plan.total_layers {
                        eprintln!(
                            "[fllm-rs] {} of {} layers fit on the device completely; the feed-forward weights of the other {} stay in host memory",
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
    let mut runtime = InferenceRuntime::with_speculative(
        model,
        max_sequences,
        max_batched_tokens,
        eos_ids,
        speculative_lookup,
        speculative_window,
    )?;
    if speculative_mtp {
        if draft_model.is_some() || speculative_lookup > 0 {
            return Err(
                "--speculative mtp cannot be combined with a draft model or lookup speculation"
                    .into(),
            );
        }
        runtime.enable_mtp()?;
        eprintln!("[fllm-rs] MTP enabled: first NextN block, one target-verified candidate; prefix reuse disabled");
    }
    if let Some(path) = draft_model {
        let draft = GgufModel::load(&path, context, max_sequences, threads)?;
        runtime.enable_draft(draft, draft_tokens)?;
        eprintln!("[fllm-rs] Draft-model verification enabled ({draft_tokens} candidate tokens)");
    }
    if device != "cpu" {
        let target_dev = plan_device.as_ref();
        if let Some(d) = target_dev {
            if let Ok(plan) = fllm_engine::backend::memory::MemoryPlan::plan(
                &runtime.model,
                d.free_memory_bytes,
                d.total_memory_bytes,
                gpu_memory_utilization,
                Some(&gpu_layers),
                max_batched_tokens,
            ) {
                #[cfg(feature = "cuda")]
                if selected_backend == "cuda" {
                    fllm_engine::backend::cuda::set_marlin_q4k_enabled(plan.marlin_q4k_dual);
                }
                eprintln!(
                    "[fllm-rs] Memory plan: target device [{}] {} (budget: {:.2} GB / {:.2} GB total, util: {:.0}%)",
                    d.index,
                    d.name,
                    plan.budget_bytes as f64 / 1e9,
                    plan.total_vram_bytes as f64 / 1e9,
                    plan.utilization * 100.0
                );
                eprintln!(
                    "[fllm-rs] Memory split: {}/{} layers offloaded | Model weights: {:.2} GB{} | KV Cache: {:.2} GB ({} blocks) | Scratchpad: {:.2} MB",
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
    #[cfg(feature = "cuda")]
    if selected_backend == "cuda" {
        let ms = fllm_engine::backend::cuda::total_repack_ms();
        if ms > 0.0 {
            eprintln!("[fllm-rs] Q4_K Marlin CPU repacking total time: {:.1} ms ({:.3} s)", ms, ms / 1000.0);
        }
    }
    eprintln!("[fllm-rs] Rust decoder forward pass enabled (packed quantized weights, paged {kv_description} KV)");
    if let Some(projector) = mmproj {
        let vision =
            fllm_engine::vision::VisionEncoder::load(&projector, threads.max(1) as usize, &device)?;
        if vision.output_hidden != runtime.model.hidden_size()
            || !runtime.model.chat_template().contains("<image>")
        {
            return Err("projector does not match this multimodal decoder/template".into());
        }
        eprintln!(
            "[fllm-rs] Own Rust SigLIP/Idefics3 vision encoder enabled: {} image tokens",
            vision.tokens
        );
        runtime.vision = Some(std::sync::Arc::new(vision));
    }
    eprintln!(
        "model={} context={} max_sequences={} max_batched_tokens={} threads={}",
        inferred_name, context, max_sequences, max_batched_tokens, threads
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
            api_key,
            default_temperature,
            default_top_p,
            default_top_k,
            sampling_defaults,
            tool_parser,
            reasoning_parser,
        },
    )?;
    Ok(())
}
