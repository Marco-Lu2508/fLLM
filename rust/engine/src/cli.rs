//! User-facing non-serving commands for the Rust fLLM executable.
//!
//! Network-backed commands use the Rust Hugging Face resolver or an explicit
//! OpenAI-compatible server URL. Model conversion and inspection never load a
//! model into the inference runtime.

use std::{
    collections::BTreeMap,
    io::{self, Write},
    path::{Path, PathBuf},
    time::Duration,
};

/// Dispatch a non-serving subcommand. `args` includes the command name; `false`
/// means that the caller should continue with its existing serve/device parser.
pub fn dispatch(args: &[String]) -> Result<bool, String> {
    let Some(command) = args.first().map(String::as_str) else {
        return Ok(false);
    };
    match command {
        "merge-lora" | "gguf-set" | "gguf-split" | "gguf-merge" => {
            gguf_tool(command, &args[1..])?;
            Ok(true)
        }
        "pull" => {
            pull(&args[1..])?;
            Ok(true)
        }
        "run" => {
            run(&args[1..])?;
            Ok(true)
        }
        "list" => {
            list(&args[1..])?;
            Ok(true)
        }
        "rm" => {
            rm(&args[1..])?;
            Ok(true)
        }
        "ps" => {
            ps(&args[1..])?;
            Ok(true)
        }
        "show" => {
            show(&args[1..])?;
            Ok(true)
        }
        "search" => {
            search(&args[1..])?;
            Ok(true)
        }
        "completions" => {
            completions(&args[1..])?;
            Ok(true)
        }
        "convert" => {
            convert(&args[1..])?;
            Ok(true)
        }
        "quantize" => {
            quantize(&args[1..])?;
            Ok(true)
        }
        "inspect" => {
            inspect(&args[1..])?;
            Ok(true)
        }
        "ppl" => {
            crate::ppl::run(&args[1..])?;
            Ok(true)
        }
        "formats" => {
            formats(&args[1..])?;
            Ok(true)
        }
        "chat" => {
            chat(&args[1..])?;
            Ok(true)
        }
        "bench-report" => {
            crate::bench::report(&args[1..])?;
            Ok(true)
        }
        "bench" => {
            crate::bench::run(&args[1..])?;
            Ok(true)
        }
        "selftest-cpu" => {
            crate::cpu_selftest::run(&args[1..])?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn help(command: &str) -> &'static str {
    match command {
        "merge-lora" => "usage: fllm-rs merge-lora BASE.gguf ADAPTER OUT.gguf [--scale S] [--outtype f16|f32|TYPE]\nADAPTER is a PEFT directory or safetensors file beside adapter_config.json. TYPE accepts quantize presets. Default: scale 1, outtype f16. Existing outputs are refused.",
        "gguf-set" => "usage: fllm-rs gguf-set KEY=VALUE ... IN.gguf OUT.gguf\nValues are strings, numbers or booleans; quote strings containing spaces. Tensor payloads are copied unchanged. Merge shards first.",
        "gguf-split" => "usage: fllm-rs gguf-split IN.gguf --max-size 4G OUT-PREFIX\nSplit at tensor boundaries. K/M/G/T use powers of 1024; the cap includes metadata and padding. A tensor larger than the cap is refused.",
        "gguf-merge" => "usage: fllm-rs gguf-merge FIRST-SHARD.gguf OUT.gguf\nMerge consecutive PREFIX-00001-of-0000N.gguf shards and remove split metadata.",
        "pull" => "usage: fllm-rs pull hf://owner/repo[:QUANT] [--revision REV] [--local-files-only]\n       fllm-rs pull hf://owner/repo/file.gguf [--revision REV] [--local-files-only]\n       fllm-rs pull --list\n       fllm-rs pull --remove hf://owner/repo[:QUANT]\nSelect a quantization or explicit filename; all GGUF shards are downloaded. Interrupted downloads resume. FLLM_CACHE overrides the usual Hub cache.",
        "run" => "usage: fllm-rs run MODEL [serve options]
MODEL: a Hugging Face link as pasted from the browser (https://huggingface.co/owner/repo, .../blob/main/file.gguf), hf.co/owner/repo:Q4_K_M, owner/repo, hf://owner/repo[:QUANT] or a local .gguf. Downloads the model on first use, starts a server on a free local port, opens a chat in this terminal and stops the server when you quit (Ctrl-D). Extra options go to serve, e.g. --ctx 8192 or --device cpu.",
        "convert" => "usage: fllm-rs convert CHECKPOINT_PATH OUTPUT.gguf [--outtype preserve|f16|f32|TYPE] [--overwrite] [--revision REV] [--local-files-only]\nCHECKPOINT_PATH may be a supported checkpoint directory or a .safetensors/.bin/.pt/.pth file with config.json and tokenizer files beside it.",
        "quantize" => "usage: fllm-rs quantize INPUT.gguf OUTPUT.gguf TYPE [--overwrite]
TYPE: q4_0, q4_1, q5_0, q5_1, q8_0, q4_k_s, q4_k_m, q5_k_s, q5_k_m, q6_k. Float tensors are re-encoded; norms, routers and already quantized tensors are kept.",
        "inspect" => "usage: fllm-rs inspect PATH [--revision REV] [--local-files-only]\nPATH may be a GGUF file or a supported dense checkpoint directory.",
        "formats" => "usage: fllm-rs formats\nLists registered GGML storage types and the Rust decoder class; registration does not promise inference support.",
        "chat" => "usage: fllm-rs chat [--url http://127.0.0.1:8000] [--model MODEL_ID] [--api-key KEY] [--max-new N] [--temperature T] [--top-p P]\nConnects to an already running OpenAI-compatible chat-completions server.",
        "bench" => crate::bench::HELP,
        "selftest-cpu" => crate::cpu_selftest::HELP,
        _ => "",
    }
}

fn option_value(args: &[String], index: &mut usize, name: &str) -> Result<String, String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| format!("missing value for {name}"))
}

fn parse_size(text: &str) -> Result<u64, String> {
    let text = text.to_ascii_uppercase();
    let text = text
        .strip_suffix("IB")
        .or_else(|| text.strip_suffix('B'))
        .unwrap_or(&text);
    let (digits, factor) = match text.as_bytes().last() {
        Some(b'K') => (&text[..text.len() - 1], 1024u64),
        Some(b'M') => (&text[..text.len() - 1], 1024u64.pow(2)),
        Some(b'G') => (&text[..text.len() - 1], 1024u64.pow(3)),
        Some(b'T') => (&text[..text.len() - 1], 1024u64.pow(4)),
        _ => (text, 1),
    };
    digits
        .parse::<u64>()
        .ok()
        .and_then(|n| n.checked_mul(factor))
        .filter(|&n| n > 0)
        .ok_or_else(|| {
            "--max-size requires a positive integer with optional K, M, G or T suffix".into()
        })
}

fn gguf_tool(command: &str, args: &[String]) -> Result<(), String> {
    use fllm_checkpoint::gguf_tools;
    if args.iter().any(|s| s == "--help" || s == "-h") {
        println!("{}", help(command));
        return Ok(());
    }
    let fail = || help(command).to_owned();
    match command {
        "merge-lora" => {
            let mut paths = Vec::new();
            let mut scale = 1.0f32;
            let mut outtype = "f16".to_owned();
            let mut i = 0;
            while i < args.len() {
                match args[i].as_str() {
                    "--scale" => {
                        scale =
                            parse_float(&option_value(args, &mut i, "--scale")?, "--scale")? as f32;
                        if !scale.is_finite() {
                            return Err("--scale exceeds f32 range".into());
                        }
                    }
                    "--outtype" => outtype = option_value(args, &mut i, "--outtype")?,
                    s if s.starts_with('-') => {
                        return Err(format!("unknown merge-lora option: {s}"))
                    }
                    s => paths.push(PathBuf::from(s)),
                }
                i += 1;
            }
            if paths.len() != 3 {
                return Err(fail());
            }
            let count = gguf_tools::merge_lora(
                &paths[0],
                &paths[1],
                &paths[2],
                scale,
                &outtype,
                crate::compute::dequantize,
            )
            .map_err(|e| format!("LoRA merge failed: {e}"))?;
            println!(
                "Merged {count} LoRA projections into {}",
                paths[2].display()
            );
        }
        "gguf-set" => {
            if args.len() < 3 {
                return Err(fail());
            }
            let end = args.len();
            gguf_tools::set_metadata(
                Path::new(&args[end - 2]),
                Path::new(&args[end - 1]),
                &args[..end - 2],
            )
            .map_err(|e| format!("GGUF metadata rewrite failed: {e}"))?;
        }
        "gguf-split" => {
            let mut paths = Vec::new();
            let mut size = None;
            let mut i = 0;
            while i < args.len() {
                match args[i].as_str() {
                    "--max-size" => {
                        if size.is_some() {
                            return Err("duplicate --max-size".into());
                        }
                        size = Some(parse_size(&option_value(args, &mut i, "--max-size")?)?);
                    }
                    s if s.starts_with('-') => {
                        return Err(format!("unknown gguf-split option: {s}"))
                    }
                    s => paths.push(PathBuf::from(s)),
                }
                i += 1;
            }
            if paths.len() != 2 {
                return Err(fail());
            }
            let outputs = gguf_tools::split(&paths[0], &paths[1], size.ok_or_else(fail)?)
                .map_err(|e| format!("GGUF split failed: {e}"))?;
            for path in outputs {
                println!("{}", path.display());
            }
        }
        "gguf-merge" => {
            if args.len() != 2 {
                return Err(fail());
            }
            gguf_tools::merge(Path::new(&args[0]), Path::new(&args[1]))
                .map_err(|e| format!("GGUF merge failed: {e}"))?;
        }
        _ => return Err(format!("unknown GGUF command: {command}")),
    }
    Ok(())
}

#[derive(Default, Debug, PartialEq, Eq)]
struct ResolveFlags {
    revision: Option<String>,
    offline: bool,
}

fn parse_resolve_flags(
    args: &[String],
    index: &mut usize,
    flags: &mut ResolveFlags,
) -> Result<(), String> {
    match args[*index].as_str() {
        "--revision" => flags.revision = Some(option_value(args, index, "--revision")?),
        "--local-files-only" | "--offline" => flags.offline = true,
        other => return Err(format!("unknown option: {other}")),
    }
    Ok(())
}

fn resolve_input(input: &str, flags: &ResolveFlags) -> Result<PathBuf, String> {
    if input.starts_with("hf://") {
        crate::hub::resolve(input, flags.revision.as_deref(), flags.offline)
    } else {
        Ok(PathBuf::from(input))
    }
}

/// `fllm-rs run MODEL [serve options]`: download the model if needed, start the server on a free local port, chat with it in the
/// terminal and stop the server again. MODEL is anything `pull` and `serve` understand plus what people paste from a browser.
fn run(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{}", help("run"));
        return Ok(());
    }
    let model_arg = args[0].clone();
    let memory = crate::catalog::available_memory_bytes();
    let normalized = match crate::catalog::expand_reference(&model_arg, Some(memory))? {
        Some(reference) => {
            eprintln!("[fllm-rs] resolved catalog alias {model_arg} to {reference}");
            reference
        }
        None => crate::hub::normalize_reference(&model_arg),
    };
    let path = crate::hub::resolve(&normalized, None, false)?;
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map_err(|e| format!("no free local port: {e}"))?
        .port();
    let log_path = std::env::temp_dir().join("fllm-run.log");
    let log = std::fs::File::create(&log_path).map_err(|e| format!("cannot write {}: {e}", log_path.display()))?;
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let mut child = std::process::Command::new(exe)
        .arg("serve")
        .args(["--model"])
        .arg(&path)
        .args(["--host", "127.0.0.1", "--port", &port.to_string()])
        .args(&args[1..])
        .stdout(log.try_clone().map_err(|e| e.to_string())?)
        .stderr(log)
        .spawn()
        .map_err(|e| format!("cannot start the server: {e}"))?;
    println!("Loading {} ...", path.display());
    let client = http_client()?;
    let health = format!("http://127.0.0.1:{port}/health");
    let started = std::time::Instant::now();
    let ready = loop {
        if let Ok(Some(status)) = child.try_wait() {
            let tail = std::fs::read_to_string(&log_path).unwrap_or_default();
            let tail: Vec<&str> = tail.lines().rev().take(15).collect();
            return Err(format!(
                "the server stopped ({status}); last lines of {}:\n{}",
                log_path.display(),
                tail.into_iter().rev().collect::<Vec<_>>().join("\n")
            ));
        }
        if client.get(&health).send().is_ok_and(|r| r.status().is_success()) {
            break true;
        }
        if started.elapsed() > Duration::from_secs(900) {
            break false;
        }
        std::thread::sleep(Duration::from_millis(500));
    };
    let result = if ready {
        let mut chat_args = vec!["--url".to_owned(), format!("http://127.0.0.1:{port}")];
        chat_args.extend(std::env::var("FLLM_RUN_MODEL").ok().map(|m| ["--model".to_owned(), m]).into_iter().flatten());
        chat(&chat_args)
    } else {
        Err(format!("the server did not become ready in 15 minutes; see {}", log_path.display()))
    };
    let _ = child.kill();
    let _ = child.wait();
    result
}

fn pull(args: &[String]) -> Result<(), String> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{}", help("pull"));
        return Ok(());
    }
    if args.first().is_some_and(|s| s == "--list") {
        if args.len() != 1 {
            return Err("pull --list takes no further arguments".into());
        }
        let entries = crate::hub::list_cache()?;
        if entries.is_empty() {
            println!("No complete models in the local cache.");
        }
        for entry in entries {
            println!(
                "{}\t{} bytes ({:.2} GB)\trevision {}\t{}",
                entry.reference,
                entry.bytes,
                entry.bytes as f64 / 1_000_000_000.0,
                entry.revision,
                entry.path.display()
            );
        }
        return Ok(());
    }
    if args.first().is_some_and(|s| s == "--remove") {
        if args.len() != 2 {
            return Err("usage: fllm-rs pull --remove hf://owner/repo[:QUANT][/file.gguf]".into());
        }
        let count = crate::hub::remove_cached(&args[1])?;
        println!("Removed {count} cached files for {}", args[1]);
        return Ok(());
    }
    let reference = args.first().ok_or_else(|| help("pull").to_owned())?;
    if !reference.starts_with("hf://") {
        return Err("fllm-rs pull expects an hf://owner/repo reference".into());
    }
    let mut flags = ResolveFlags::default();
    let mut i = 1;
    while i < args.len() {
        parse_resolve_flags(args, &mut i, &mut flags)?;
        i += 1;
    }
    let path = crate::hub::resolve(reference, flags.revision.as_deref(), flags.offline)?;
    println!("{}", path.display());
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
struct ConvertArgs {
    input: String,
    output: String,
    outtype: String,
    overwrite: bool,
    resolve: ResolveFlags,
}

fn parse_convert_args(args: &[String]) -> Result<ConvertArgs, String> {
    let mut positional = Vec::new();
    let mut outtype = "preserve".to_owned();
    let mut overwrite = false;
    let mut resolve = ResolveFlags::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--outtype" => outtype = option_value(args, &mut i, "--outtype")?,
            "--overwrite" => overwrite = true,
            "--revision" | "--local-files-only" | "--offline" => {
                parse_resolve_flags(args, &mut i, &mut resolve)?
            }
            "--help" | "-h" => return Err(help("convert").to_owned()),
            option if option.starts_with('-') => {
                return Err(format!("unknown convert option: {option}"))
            }
            value => positional.push(value.to_owned()),
        }
        i += 1;
    }
    if positional.len() != 2 {
        return Err(help("convert").to_owned());
    }
    if !matches!(outtype.as_str(), "preserve" | "f16" | "f32")
        && fllm_checkpoint::Preset::parse(&outtype).is_none()
    {
        return Err(format!(
            "--outtype must be preserve, f16, f32 or one of {}",
            fllm_checkpoint::Preset::NAMES
        ));
    }
    Ok(ConvertArgs {
        input: positional.remove(0),
        output: positional.remove(0),
        outtype,
        overwrite,
        resolve,
    })
}

fn convert(args: &[String]) -> Result<(), String> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{}", help("convert"));
        return Ok(());
    }
    let args = parse_convert_args(args)?;
    let input = resolve_input(&args.input, &args.resolve)?;
    if !input.is_dir() && !is_supported_checkpoint_file(&input) {
        return Err(format!(
            "unsupported conversion input {}; use a checkpoint directory or a .safetensors/.bin/.pt/.pth file",
            input.display()
        ));
    }
    let preset = fllm_checkpoint::Preset::parse(&args.outtype);
    let dtype = match args.outtype.as_str() {
        "preserve" => fllm_checkpoint::OutputDType::Preserve,
        "f32" => fllm_checkpoint::OutputDType::F32,
        _ => fllm_checkpoint::OutputDType::F16,
    };
    let final_output = Path::new(&args.output);
    let staging = preset.map(|_| final_output.with_extension("f16.tmp"));
    let output = staging.as_deref().unwrap_or(final_output);
    let report = fllm_checkpoint::convert_checkpoint(
        &input,
        output,
        fllm_checkpoint::ConvertOptions {
            output_dtype: dtype,
            overwrite: args.overwrite,
        },
    )
    .map_err(|e| format!("checkpoint conversion failed: {e}"))?;
    if let Some(preset) = preset {
        if final_output.exists() && !args.overwrite {
            let _ = std::fs::remove_file(output);
            return Err(format!(
                "{} exists; pass --overwrite",
                final_output.display()
            ));
        }
        let result = fllm_checkpoint::quantize_gguf(output, final_output, preset);
        let _ = std::fs::remove_file(output);
        let q = result.map_err(|e| format!("quantization failed: {e}"))?;
        println!(
            "Converted {} tensors for {} ({} quantized): {} bytes -> {}",
            q.tensors,
            report.architecture,
            q.quantized,
            q.output_bytes,
            final_output.display()
        );
        return Ok(());
    }
    println!(
        "Converted {} tensors for {}: {} bytes -> {}",
        report.tensors_written,
        report.architecture,
        report.bytes_written,
        report.output.display()
    );
    if !report.skipped_tensors.is_empty() {
        println!(
            "Skipped {} unsupported/non-model tensors.",
            report.skipped_tensors.len()
        );
    }
    Ok(())
}

fn quantize(args: &[String]) -> Result<(), String> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{}", help("quantize"));
        return Ok(());
    }
    let overwrite = args.iter().any(|a| a == "--overwrite");
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let [input, output, kind] = positional[..] else {
        return Err(help("quantize").to_owned());
    };
    let preset = fllm_checkpoint::Preset::parse(kind).ok_or_else(|| {
        format!(
            "unknown type {kind}; use one of {}",
            fllm_checkpoint::Preset::NAMES
        )
    })?;
    if Path::new(output).exists() && !overwrite {
        return Err(format!("{output} exists; pass --overwrite"));
    }
    let q = fllm_checkpoint::quantize_gguf(Path::new(input), Path::new(output), preset)
        .map_err(|e| format!("quantization failed: {e}"))?;
    println!(
        "Quantized {} of {} tensors: {} -> {} bytes ({:.2}x smaller)",
        q.quantized,
        q.tensors,
        q.input_bytes,
        q.output_bytes,
        q.input_bytes as f64 / q.output_bytes as f64
    );
    Ok(())
}

fn inspect(args: &[String]) -> Result<(), String> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{}", help("inspect"));
        return Ok(());
    }
    let input = args.first().ok_or_else(|| help("inspect").to_owned())?;
    let mut flags = ResolveFlags::default();
    let mut i = 1;
    while i < args.len() {
        parse_resolve_flags(args, &mut i, &mut flags)?;
        i += 1;
    }
    let path = resolve_input(input, &flags)?;
    if inspect_kind(&path)? == InspectKind::Checkpoint {
        let inventory = fllm_checkpoint::inspect_checkpoint(&path)
            .map_err(|e| format!("checkpoint inspection failed: {e}"))?;
        println!("Checkpoint: {}", path.display());
        println!("Format: {}", inventory.formats.join(", "));
        println!("Tensors: {}", inventory.tensors.len());
        let mut dtypes = BTreeMap::<String, usize>::new();
        for tensor in &inventory.tensors {
            *dtypes.entry(tensor.dtype.clone()).or_default() += 1;
        }
        println!(
            "Dtypes: {}",
            dtypes
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    } else {
        let mapping = fllm_gguf_probe::MappedGguf::open(&path)
            .map_err(|e| format!("GGUF inspection failed: {e}"))?;
        let index = mapping.index();
        println!("GGUF: {}", path.display());
        println!("Version: {}", index.version);
        println!("Metadata fields: {}", index.metadata.len());
        println!("Tensors: {}", index.tensors.len());
        let mut types = BTreeMap::<String, usize>::new();
        for tensor in &index.tensors {
            let name = fllm_gguf_probe::quant_types::by_id(tensor.type_id)
                .map(|q| q.name)
                .unwrap_or("unknown");
            *types
                .entry(format!("{name} (id {})", tensor.type_id))
                .or_default() += 1;
        }
        println!(
            "Tensor types: {}",
            types
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        if let Some(arch) = index
            .metadata
            .get("general.architecture")
            .and_then(serde_json::Value::as_str)
        {
            println!("Architecture: {arch}");
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InspectKind {
    Gguf,
    Checkpoint,
}

fn is_supported_checkpoint_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some("safetensors" | "bin" | "pt" | "pth")
    )
}

fn inspect_kind(path: &Path) -> Result<InspectKind, String> {
    if path.is_dir() {
        return Ok(InspectKind::Checkpoint);
    }
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("gguf") => Ok(InspectKind::Gguf),
        Some("safetensors" | "bin" | "pt" | "pth") => Ok(InspectKind::Checkpoint),
        _ => Err(format!(
            "unsupported inspect input {}; use a .gguf file, .safetensors/.bin/.pt/.pth file, or checkpoint directory",
            path.display()
        )),
    }
}

fn formats(args: &[String]) -> Result<(), String> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{}", help("formats"));
        return Ok(());
    }
    if !args.is_empty() {
        return Err(format!("{}\nformats takes no options", help("formats")));
    }
    use fllm_gguf_probe::quant_types::{DecoderClass, GGML_TYPES};
    println!(
        "Registered GGML tensor types: {} (registration/layout support is not inference support)",
        GGML_TYPES.len()
    );
    for ty in GGML_TYPES {
        let class = match ty.decoder {
            DecoderClass::Scalar => "scalar decode",
            DecoderClass::ExistingRust => "Rust compute decoder",
            DecoderClass::ExtraRust => "additional Rust decoder",
            DecoderClass::SeparateIq => "IQ Rust decoder",
            DecoderClass::RuntimeLayoutOnly => "layout only; no Rust decoder",
        };
        println!(
            "{:>2} {:<10} block={:<3} bytes={:<3} {class}",
            ty.id, ty.name, ty.block_elements, ty.bytes_per_block
        );
    }
    println!("Decoder labels describe tensor decoding only; architecture, kernel, and device support are separate.");
    Ok(())
}

#[derive(Clone, Debug)]
struct HttpOptions {
    base_url: String,
    model: String,
    api_key: Option<String>,
    max_new: usize,
    temperature: f64,
    top_p: f64,
}

fn endpoint(base: &str) -> Result<String, String> {
    let base = base.trim_end_matches('/');
    if !(base.starts_with("http://") || base.starts_with("https://")) {
        return Err("--url must begin with http:// or https://".into());
    }
    Ok(if base.ends_with("/v1/chat/completions") {
        base.to_owned()
    } else if base.ends_with("/v1") {
        format!("{base}/chat/completions")
    } else {
        format!("{base}/v1/chat/completions")
    })
}

fn http_client() -> Result<reqwest::blocking::Client, String> {
    reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(3600))
        .user_agent("fllm-rust-cli/0.1")
        .build()
        .map_err(|e| format!("cannot initialize HTTP client: {e}"))
}

fn parse_chat_args(args: &[String]) -> Result<HttpOptions, String> {
    let mut options = HttpOptions {
        base_url: "http://127.0.0.1:8000".into(),
        model: "fllm-model".into(),
        api_key: std::env::var("FLLM_API_KEY").ok(),
        max_new: 512,
        temperature: 0.7,
        top_p: 0.9,
    };
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        match flag {
            "--url" => options.base_url = option_value(args, &mut i, flag)?,
            "--model" => options.model = option_value(args, &mut i, flag)?,
            "--api-key" => options.api_key = Some(option_value(args, &mut i, flag)?),
            "--max-new" => {
                options.max_new = parse_positive(&option_value(args, &mut i, flag)?, flag)?
            }
            "--temperature" => {
                options.temperature = parse_float(&option_value(args, &mut i, flag)?, flag)?
            }
            "--top-p" => options.top_p = parse_float(&option_value(args, &mut i, flag)?, flag)?,
            "--help" | "-h" => return Err(help("chat").to_owned()),
            other => return Err(format!("unknown chat option: {other}")),
        }
        i += 1;
    }
    if options.temperature < 0.0 {
        return Err("--temperature must be non-negative".into());
    }
    if !(0.0..=1.0).contains(&options.top_p) {
        return Err("--top-p must be between 0 and 1".into());
    }
    endpoint(&options.base_url)?;
    Ok(options)
}

fn parse_positive(value: &str, flag: &str) -> Result<usize, String> {
    let number = value
        .parse::<usize>()
        .map_err(|_| format!("{flag} requires a positive integer"))?;
    if number == 0 {
        return Err(format!("{flag} must be greater than zero"));
    }
    Ok(number)
}

fn parse_float(value: &str, flag: &str) -> Result<f64, String> {
    let number = value
        .parse::<f64>()
        .map_err(|_| format!("{flag} requires a finite number"))?;
    if !number.is_finite() {
        return Err(format!("{flag} requires a finite number"));
    }
    Ok(number)
}

fn list(args: &[String]) -> Result<(), String> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{}", help("list"));
        return Ok(());
    }
    let entries = crate::hub::list_cache()?;
    if entries.is_empty() {
        println!("No complete models in the local cache.");
        return Ok(());
    }
    for entry in entries {
        println!(
            "{}\t{} ({:.2} GB)\t{}",
            entry.reference,
            entry.bytes,
            entry.bytes as f64 / 1_000_000_000.0,
            entry.path.display()
        );
    }
    Ok(())
}

fn rm(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        return Err("usage: fllm-rs rm MODEL".into());
    }
    let input = &args[0];
    let base = repository_of_reference(input);
    let count = crate::hub::remove_cached(&base)?;
    println!("Removed {count} cached entries for {base}");
    Ok(())
}

fn ps(_args: &[String]) -> Result<(), String> {
    let dir = crate::hub::status_dir();
    if !dir.is_dir() {
        println!("No running fllm-rs servers recorded.");
        return Ok(());
    }
    let mut found = false;
    for entry in std::fs::read_dir(&dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap_or_default();
        let v: serde_json::Value = serde_json::from_str(&text).unwrap_or_default();
        println!(
            "pid={} port={} model={} host={}",
            v.get("pid").and_then(serde_json::Value::as_i64).unwrap_or_default(),
            v.get("port").and_then(serde_json::Value::as_i64).unwrap_or_default(),
            v.get("model").and_then(serde_json::Value::as_str).unwrap_or("-"),
            v.get("host").and_then(serde_json::Value::as_str).unwrap_or("-")
        );
        found = true;
    }
    if !found {
        println!("No running fllm-rs servers recorded.");
    }
    Ok(())
}

fn show(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        return Err("usage: fllm-rs show MODEL".into());
    }
    let mut flags = ResolveFlags::default();
    let mut i = 1;
    while i < args.len() {
        parse_resolve_flags(args, &mut i, &mut flags)?;
        i += 1;
    }
    let input = args[0].as_str();
    let resolved = if let Some(expanded) = crate::catalog::expand_reference(input, None)? {
        expanded
    } else {
        crate::hub::normalize_reference(input)
    };
    let path = if resolved.starts_with("hf://") {
        crate::hub::resolve(&resolved, flags.revision.as_deref(), flags.offline)?
    } else {
        std::path::PathBuf::from(&resolved)
    };
    let metadata = std::fs::metadata(&path).map_err(|e| format!("{}", e))?;
    println!("Path: {}", path.display());
    println!("Size: {} bytes ({:.2} GB)", metadata.len(), metadata.len() as f64 / 1_000_000_000.0);
    if path.extension().and_then(|e| e.to_str()).map(|e| e.eq_ignore_ascii_case("gguf")).unwrap_or(false) {
        let mapping = fllm_gguf_probe::MappedGguf::open(&path)
            .map_err(|e| format!("GGUF inspection failed: {e}"))?;
        let index = mapping.index();
        if let Some(arch) = index.metadata.get("general.architecture").and_then(serde_json::Value::as_str) {
            println!("Architecture: {arch}");
        }
        for key in ["general.name", "tokenizer.ggml.model", "llm.context_length", "llama.context_length", "qwen2.context_length"] {
            if let Some(value) = index.metadata.get(key) {
                println!("{key}: {value}");
            }
        }
    }
    Ok(())
}

fn search(args: &[String]) -> Result<(), String> {
    let query = args.join(" ").to_lowercase();
    for entry in crate::catalog::catalog() {
        if query.is_empty()
            || entry.name.to_lowercase().contains(&query)
            || entry.repo.to_lowercase().contains(&query)
        {
            println!("{}\t{}\t(default {})", entry.name, entry.repo, entry.default_quant);
        }
    }
    Ok(())
}

fn completions(args: &[String]) -> Result<(), String> {
    let shell = args.first().map(String::as_str).unwrap_or("bash");
    let commands = [
        "serve", "run", "pull", "list", "rm", "ps", "show", "search", "convert", "quantize",
        "inspect", "formats", "chat", "bench", "bench-report", "devices", "completions",
        "merge-lora", "gguf-set", "gguf-split", "gguf-merge", "selftest-cpu", "selftest-gpu",
    ];
    match shell {
        "bash" | "zsh" => {
            println!("_fllm_rs() {{");
            println!("  local cur=\"${{COMP_WORDS[COMP_CWORD]}}\"");
            println!("  COMPREPLY=( $(compgen -W \"{}\" -- \"$cur\") )", commands.join(" "));
            println!("}}");
            println!("complete -F _fllm_rs fllm-rs");
        }
        "fish" => {
            for c in commands {
                println!("complete -c fllm-rs -a {}", c);
            }
        }
        "powershell" => {
            println!("Register-ArgumentCompleter -Native -CommandName fllm-rs -ScriptBlock {{");
            println!("  param($wordToComplete)");
            println!("  @('{}') | Where-Object {{ $_ -like \"$wordToComplete*\" }}", commands.join("', '"));
            println!("}}");
        }
        _ => return Err(format!("unsupported shell: {shell}")),
    }
    Ok(())
}

fn repository_of_reference(input: &str) -> String {
    let normalized = crate::hub::normalize_reference(input);
    if let Some(raw) = normalized.strip_prefix("hf://") {
        let mut parts = raw.splitn(3, '/');
        let owner = parts.next().unwrap_or("");
        let repo_with_quant = parts.next().unwrap_or("");
        let repo = repo_with_quant.split(':').next().unwrap_or(repo_with_quant);
        if !owner.is_empty() && !repo.is_empty() {
            return format!("hf://{}/{}", owner, repo);
        }
    }
    normalized
}


fn chat(args: &[String]) -> Result<(), String> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{}", help("chat"));
        return Ok(());
    }
    let options = parse_chat_args(args)?;
    let url = endpoint(&options.base_url)?;
    let client = http_client()?;
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let mut history = Vec::<serde_json::Value>::new();
    println!("fLLM chat (Ctrl-D to exit; connects to {url})");
    let mut system: Option<String> = None;
    loop {
        print!("
> ");
        io::stdout().flush().map_err(|e| e.to_string())?;
        let mut line = String::new();
        if input.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
            break;
        }
        let message = line.trim();
        if message.is_empty() {
            continue;
        }
        if message == "/exit" || message == "/quit" {
            break;
        }
        if message == "/clear" {
            history.clear();
            println!("[history cleared]");
            continue;
        }
        if let Some(rest) = message.strip_prefix("/system ") {
            system = Some(rest.to_owned());
            history.retain(|m| m["role"] != "system");
            history.insert(0, serde_json::json!({"role":"system","content":rest}));
            println!("[system set]");
            continue;
        }
        if let Some(rest) = message.strip_prefix("/set temperature ") {
            // accepted for UX parity; the value is not re-applied mid-session beyond a note.
            println!("[temperature {} acknowledged]", rest);
            continue;
        }
        history.push(serde_json::json!({"role":"user","content":message}));
        let mut body = serde_json::json!({
            "model": options.model,
            "messages": history,
            "max_tokens": options.max_new,
            "temperature": options.temperature,
            "top_p": options.top_p,
            "stream": true
        });
        let mut msgs = history.clone();
        if let Some(sys) = &system {
            if !msgs.iter().any(|m| m["role"] == "system") {
                msgs.insert(0, serde_json::json!({"role":"system","content":sys}));
            }
        }
        body["messages"] = serde_json::json!(msgs);

        let mut request = client.post(&url).json(&body);
        if let Some(key) = options.api_key.as_deref() {
            request = request.bearer_auth(key);
        }
        let response = match request.send() {
            Ok(r) => r,
            Err(e) => { history.pop(); return Err(format!("chat request failed: {e}")); }
        };
        if !response.status().is_success() {
            let text = response.text().unwrap_or_default();
            history.pop();
            return Err(format!("server returned HTTP {}: {text}", text));
        }
        use std::io::BufRead;
        let reader = std::io::BufReader::new(response);
        let mut assistant = String::new();
        for line in reader.lines() {
            let line = match line { Ok(l) => l, Err(_) => break };
            let line = line.trim();
            if line.is_empty() { continue; }
            let Some(rest) = line.strip_prefix("data:") else { continue; };
            let rest = rest.trim();
            if rest == "[DONE]" { break; }
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(rest) {
                if let Some(content) = value.pointer("/choices/0/delta/content").and_then(serde_json::Value::as_str) {
                    print!("{}", content);
                    io::stdout().flush().ok();
                    assistant.push_str(content);
                }
            }
        }
        println!();
        history.push(serde_json::json!({"role":"assistant","content":assistant}));
    }
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    fn args(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn dispatch_leaves_serve_and_devices_for_existing_parser() {
        assert!(!dispatch(&args(&["serve"])).unwrap());
        assert!(!dispatch(&args(&["devices"])).unwrap());
        assert!(dispatch(&args(&["formats"])).unwrap());
    }

    #[test]
    fn convert_options_default_to_preserve_and_require_distinct_paths() {
        let parsed = parse_convert_args(&args(&["model", "out.gguf"])).unwrap();
        assert_eq!(parsed.outtype, "preserve");
        assert!(!parsed.overwrite);
        let parsed = parse_convert_args(&args(&[
            "model",
            "out.gguf",
            "--outtype",
            "f32",
            "--overwrite",
            "--revision",
            "v1",
            "--offline",
        ]))
        .unwrap();
        assert_eq!(parsed.outtype, "f32");
        assert!(parsed.overwrite);
        assert!(parsed.resolve.offline);
        assert!(parse_convert_args(&args(&["model"])).is_err());
        assert!(parse_convert_args(&args(&["model", "out", "--outtype", "bf16"])).is_err());
    }

    #[test]
    fn checkpoint_paths_are_distinguished_from_gguf_and_other_files() {
        assert!(is_supported_checkpoint_file(Path::new(
            "weights.safetensors"
        )));
        assert!(is_supported_checkpoint_file(Path::new("weights.bin")));
        assert!(is_supported_checkpoint_file(Path::new("weights.pt")));
        assert!(is_supported_checkpoint_file(Path::new("weights.pth")));
        assert!(!is_supported_checkpoint_file(Path::new("model.gguf")));
        assert_eq!(
            inspect_kind(Path::new("model.GGUF")).unwrap(),
            InspectKind::Gguf
        );
        assert_eq!(
            inspect_kind(Path::new("weights.safetensors")).unwrap(),
            InspectKind::Checkpoint
        );
        assert!(inspect_kind(Path::new("notes.txt"))
            .unwrap_err()
            .contains("unsupported inspect input"));
    }

    #[test]
    fn http_cli_rejects_invalid_ranges_and_normalizes_endpoint() {
        let parsed = parse_chat_args(&args(&[
            "--url",
            "http://localhost:9000/v1",
            "--top-p",
            "0.5",
        ]))
        .unwrap();
        assert_eq!(
            endpoint(&parsed.base_url).unwrap(),
            "http://localhost:9000/v1/chat/completions"
        );
        assert!(parse_chat_args(&args(&["--max-new", "0"])).is_err());
        assert!(parse_chat_args(&args(&["--top-p", "1.5"])).is_err());
        assert!(parse_chat_args(&args(&["--url", "ftp://localhost"])).is_err());
    }
}
