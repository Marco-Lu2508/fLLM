//! Studio control plane routes (`/api/studio/*`) for the fLLM WebUI.
//! Exposes live Rust engine telemetry, hardware inventory, CPU SIMD kernel self-testing,
//! runtime sampling & engine configuration, API key management, and GGUF model hub/cache management.
use super::*;
use crate::cpu_kernels::{self, CpuTier};
use axum::extract::{Path as AxumPath, Query};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub(in crate::server) struct StudioRuntimeConfig {
    pub sampling: crate::sampling::SamplingParams,
    pub override_temperature: Option<f64>,
    pub override_top_p: Option<f64>,
    pub override_top_k: Option<usize>,
    pub system_prompt: String,
    pub cpu_tier_cap: Option<String>,
}

fn token_text(state: &AppState, ids: &[u32]) -> String {
    ids.first()
        .and_then(|id| state.tokenizer.decode(&[*id], false).ok())
        .unwrap_or_default()
}

pub(in crate::server) fn build_status_json(state: &AppState) -> Value {
    let cfg = state.studio_config.read().unwrap().clone();
    let aliases = state.aliases.read().unwrap().clone();
    let vision = state
        .vision
        .as_ref()
        .is_some_and(|v| v.marker(crate::vision::MediaKind::Image).is_some());
    let video = state
        .vision
        .as_ref()
        .is_some_and(|v| v.marker(crate::vision::MediaKind::Video).is_some());
    let audio = state.vision.as_ref().is_some_and(|v| v.family() == "whisper");
    let thinking = state.chat_template.contains("hinking")
        || !matches!(state.reasoning_family, crate::reasoning::Family::None);
    let rate_limit = state.http.security.rate_limit_spec();
    let counters = &state.http.counters;
    let m = &state.metrics;

    let running = m.slots_running.load(Ordering::Relaxed);
    let waiting = m.slots_waiting.load(Ordering::Relaxed);
    let max_seqs = state.max_sequences;

    json!({
        "status": "ok",
        "ready": state.ready.load(Ordering::Relaxed),
        "version": env!("CARGO_PKG_VERSION"),
        "build_info": crate::version::long_version(),
        "model": {
            "name": state.model_name,
            "arch": state.model_arch,
            "hidden_size": state.hidden_size,
            "max_context": state.max_context,
            "trained_context": state.facts.trained_context,
            "max_sequences": max_seqs,
            "vocab_size": state.tokenizer.vocab_size(),
            "size_bytes": state.facts.size_bytes,
            "parameters": state.facts.parameters,
            "parameter_size": state.facts.parameter_size(),
            "quantization": state.facts.quantization,
            "digest": state.facts.digest,
            "chat_template": state.chat_template,
            "tool_chat_template": state.tool_chat_template,
            "tool_parser": state.tool_parser.id(),
            "reasoning_family": format!("{:?}", state.reasoning_family),
            "bos_token": token_text(state, state.tokenizer.bos_ids()),
            "eos_token": token_text(state, state.tokenizer.eos_ids()),
            "bos_ids": state.tokenizer.bos_ids(),
            "eos_ids": state.tokenizer.eos_ids(),
            "shared_choices": state.shared_choices,
            "aliases": aliases,
            "capabilities": {
                "completion": true,
                "chat": true,
                "tools": true,
                "fim_infill": true,
                "embeddings": true,
                "vision": vision,
                "video": video,
                "audio": audio,
                "thinking": thinking,
            }
        },
        "config": {
            "temperature": cfg.override_temperature.unwrap_or(cfg.sampling.temperature),
            "override_temperature": cfg.override_temperature,
            "top_p": cfg.override_top_p.unwrap_or(cfg.sampling.top_p),
            "override_top_p": cfg.override_top_p,
            "top_k": cfg.override_top_k.unwrap_or(cfg.sampling.top_k),
            "override_top_k": cfg.override_top_k,
            "min_p": cfg.sampling.min_p,
            "typical_p": cfg.sampling.typical_p,
            "repetition_penalty": cfg.sampling.repetition_penalty,
            "repeat_last_n": cfg.sampling.repeat_last_n,
            "presence_penalty": cfg.sampling.presence_penalty,
            "frequency_penalty": cfg.sampling.frequency_penalty,
            "dry_multiplier": cfg.sampling.dry_multiplier,
            "dry_base": cfg.sampling.dry_base,
            "dry_allowed_length": cfg.sampling.dry_allowed_length,
            "dry_penalty_last_n": cfg.sampling.dry_penalty_last_n,
            "mirostat": cfg.sampling.mirostat,
            "mirostat_tau": cfg.sampling.mirostat_tau,
            "mirostat_eta": cfg.sampling.mirostat_eta,
            "seed": cfg.sampling.seed,
            "min_tokens": cfg.sampling.min_tokens,
            "ignore_eos": cfg.sampling.ignore_eos,
            "speculative": cfg.sampling.speculative,
            "priority": cfg.sampling.priority,
            "system_prompt": cfg.system_prompt,
            "cpu_repack": cpu_kernels::repack_enabled(),
            "cpu_tier_cap": cfg.cpu_tier_cap,
            "active_cpu_tier": cpu_kernels::capped_tiers().last().map(|t| t.name()).unwrap_or("scalar"),
        },
        "http": {
            "auth_required": state.http.security.auth_required(),
            "keys_count": state.http.security.list_keys().len(),
            "rate_limit": {
                "requests_per_minute": rate_limit.requests_per_minute,
                "tokens_per_minute": rate_limit.tokens_per_minute,
            },
            "cors_origins": state.http.options.cors_origins,
            "public_routes": state.http.options.public_routes,
            "max_connections": state.http.options.max_connections,
            "max_body_bytes": state.http.options.max_body_bytes,
            "request_timeout_secs": state.http.options.request_timeout.as_secs_f64(),
            "header_read_timeout_secs": state.http.options.header_read_timeout.as_secs_f64(),
            "keep_alive_timeout_secs": state.http.options.keep_alive_timeout.as_secs_f64(),
            "stream_idle_timeout_secs": state.http.options.stream_idle_timeout.as_secs_f64(),
            "shutdown_grace_secs": state.http.options.shutdown_grace.as_secs_f64(),
            "allow_remote_images": state.http.options.allow_remote_images,
            "access_log": state.http.options.access_log,
            "metrics_style": format!("{:?}", state.http.options.metrics_style),
        },
        "metrics": {
            "uptime_secs": m.started.elapsed().as_secs_f64(),
            "heartbeat_ms": m.heartbeat_ms.load(Ordering::Relaxed),
            "requests_total": m.requests.load(Ordering::Relaxed),
            "generated_tokens_total": m.generated_tokens.load(Ordering::Relaxed),
            "forward_calls_total": m.forward_calls.load(Ordering::Relaxed),
            "rows_total": m.rows_total.load(Ordering::Relaxed),
            "multi_row_calls": m.multi_row_calls.load(Ordering::Relaxed),
            "max_rows": m.max_rows.load(Ordering::Relaxed),
            "prefill_rows_total": m.prefill_rows.load(Ordering::Relaxed),
            "decode_rows_total": m.decode_rows.load(Ordering::Relaxed),
            "draft_proposed": m.draft_proposed.load(Ordering::Relaxed),
            "draft_accepted": m.draft_accepted.load(Ordering::Relaxed),
            "slots_running": running,
            "slots_waiting": waiting,
            "slots_total": max_seqs,
            "queue_rejections": m.queue_rejections.load(Ordering::Relaxed),
            "worker_panics": m.worker_panics.load(Ordering::Relaxed),
            "connections_open": counters.connections_open.load(Ordering::Relaxed),
            "connections_rejected": counters.connections_rejected.load(Ordering::Relaxed),
            "request_timeouts": counters.request_timeouts.load(Ordering::Relaxed),
            "rate_limited_total": state.http.security.rejected_total(),
            "scheduler_ms": m.scheduler_ns.load(Ordering::Relaxed) as f64 / 1e6,
            "batch_build_ms": m.batch_build_ns.load(Ordering::Relaxed) as f64 / 1e6,
            "model_forward_ms": m.model_forward_ns.load(Ordering::Relaxed) as f64 / 1e6,
            "postprocess_ms": m.postprocess_ns.load(Ordering::Relaxed) as f64 / 1e6,
        }
    })
}

pub(in crate::server) async fn status(State(state): State<Arc<AppState>>) -> Response {
    Json(build_status_json(&state)).into_response()
}

pub(in crate::server) fn apply_config_update(state: &AppState, body: &Value) -> Result<(), String> {
    let src = body.get("default_generation_settings").unwrap_or(body);
    let mut cfg = state.studio_config.write().unwrap();

    if let Some(v) = src.get("temperature").and_then(Value::as_f64) {
        if !v.is_finite() || v < 0.0 {
            return Err("temperature must be a non-negative finite number".into());
        }
        cfg.sampling.temperature = v;
        cfg.override_temperature = Some(v);
    }
    if let Some(v) = src.get("top_p").and_then(Value::as_f64) {
        if !v.is_finite() || !(0.0..=1.0).contains(&v) {
            return Err("top_p must be between 0.0 and 1.0".into());
        }
        cfg.sampling.top_p = v;
        cfg.override_top_p = Some(v);
    }
    if let Some(v) = src.get("top_k").and_then(Value::as_u64) {
        cfg.sampling.top_k = v as usize;
        cfg.override_top_k = Some(v as usize);
    }
    if let Some(v) = src.get("min_p").and_then(Value::as_f64) {
        if (0.0..=1.0).contains(&v) {
            cfg.sampling.min_p = v;
        }
    }
    if let Some(v) = src.get("typical_p").and_then(Value::as_f64) {
        if (0.0..=1.0).contains(&v) {
            cfg.sampling.typical_p = v;
        }
    }
    if let Some(v) = src
        .get("repetition_penalty")
        .or_else(|| src.get("repeat_penalty"))
        .and_then(Value::as_f64)
    {
        if v > 0.0 {
            cfg.sampling.repetition_penalty = v as f32;
        }
    }
    if let Some(v) = src.get("repeat_last_n").and_then(Value::as_i64) {
        cfg.sampling.repeat_last_n = v;
    }
    if let Some(v) = src.get("presence_penalty").and_then(Value::as_f64) {
        cfg.sampling.presence_penalty = v as f32;
    }
    if let Some(v) = src.get("frequency_penalty").and_then(Value::as_f64) {
        cfg.sampling.frequency_penalty = v as f32;
    }
    if let Some(v) = src.get("dry_multiplier").and_then(Value::as_f64) {
        if v >= 0.0 {
            cfg.sampling.dry_multiplier = v as f32;
        }
    }
    if let Some(v) = src.get("dry_base").and_then(Value::as_f64) {
        if v >= 1.0 {
            cfg.sampling.dry_base = v as f32;
        }
    }
    if let Some(v) = src.get("dry_allowed_length").and_then(Value::as_u64) {
        cfg.sampling.dry_allowed_length = v as usize;
    }
    if let Some(v) = src.get("dry_penalty_last_n").and_then(Value::as_i64) {
        cfg.sampling.dry_penalty_last_n = v;
    }
    if let Some(v) = src.get("mirostat").and_then(Value::as_u64) {
        if v <= 2 {
            cfg.sampling.mirostat = v as u8;
        }
    }
    if let Some(v) = src.get("mirostat_tau").and_then(Value::as_f64) {
        if v > 0.0 {
            cfg.sampling.mirostat_tau = v as f32;
        }
    }
    if let Some(v) = src.get("mirostat_eta").and_then(Value::as_f64) {
        if v > 0.0 {
            cfg.sampling.mirostat_eta = v as f32;
        }
    }
    if let Some(seed_val) = src.get("seed") {
        if seed_val.is_null() {
            cfg.sampling.seed = None;
        } else if let Some(s) = seed_val.as_i64() {
            cfg.sampling.seed = if s < 0 { None } else { Some(s as u64) };
        } else if let Some(s) = seed_val.as_u64() {
            cfg.sampling.seed = Some(s);
        }
    }
    if let Some(v) = src.get("min_tokens").and_then(Value::as_u64) {
        cfg.sampling.min_tokens = v as usize;
    }
    if let Some(v) = src.get("ignore_eos").and_then(Value::as_bool) {
        cfg.sampling.ignore_eos = v;
    }
    if let Some(v) = src.get("speculative").and_then(Value::as_bool) {
        cfg.sampling.speculative = v;
    }
    if let Some(v) = src.get("priority").and_then(Value::as_i64) {
        cfg.sampling.priority = v.clamp(-1000, 1000) as i32;
    }
    if let Some(v) = body.get("system_prompt").and_then(Value::as_str) {
        cfg.system_prompt = v.to_owned();
    }
    if let Some(repack) = body.get("cpu_repack").and_then(Value::as_bool) {
        cpu_kernels::set_repack(repack);
    }
    if let Some(tier_val) = body.get("cpu_tier_cap") {
        if tier_val.is_null() || tier_val.as_str() == Some("") || tier_val.as_str() == Some("auto") {
            cpu_kernels::set_tier_cap(None);
            cfg.cpu_tier_cap = None;
        } else if let Some(name) = tier_val.as_str() {
            let tier = CpuTier::parse(name.trim()).ok_or_else(|| format!("unknown CPU tier: {name}"))?;
            cpu_kernels::set_tier_cap(Some(tier));
            cfg.cpu_tier_cap = Some(tier.name().to_owned());
        }
    }
    drop(cfg);

    if let Some(aliases) = body.get("aliases").and_then(Value::as_array) {
        let mut guard = state.aliases.write().unwrap();
        *guard = aliases
            .iter()
            .filter_map(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect();
    }

    if let Some(rl) = body.get("rate_limit").and_then(Value::as_object) {
        let rpm = rl.get("requests_per_minute").and_then(Value::as_f64).filter(|&v| v > 0.0);
        let tpm = rl.get("tokens_per_minute").and_then(Value::as_f64).filter(|&v| v > 0.0);
        state.http.security.set_rate_limit(options::RateLimitSpec {
            requests_per_minute: rpm,
            tokens_per_minute: tpm,
        });
    }

    Ok(())
}

pub(in crate::server) async fn update_config(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Response {
    if let Err(msg) = apply_config_update(&state, &body) {
        return api_error(StatusCode::BAD_REQUEST, &msg);
    }
    Json(build_status_json(&state)).into_response()
}

pub(in crate::server) async fn hardware(
    Query(query): Query<HashMap<String, String>>,
) -> Response {
    let run_bench = query.get("benchmark").is_some_and(|v| v == "true" || v == "1");
    let probe = tokio::task::spawn_blocking(move || {
        let devices: Vec<Value> = crate::devices::inventory()
            .into_iter()
            .map(|d| {
                let caps = d.caps();
                json!({
                    "id": d.spec,
                    "backend": d.backend,
                    "api": d.api.as_str(),
                    "index": d.local_index,
                    "name": d.info.name,
                    "description": d.info.description,
                    "dev_type": d.info.dev_type.as_str(),
                    "available": true,
                    "free_memory_bytes": d.info.free_memory_bytes,
                    "total_memory_bytes": d.info.total_memory_bytes,
                    "compute_units": d.info.description,
                    "bandwidth_gbs": Value::Null,
                    "unified_memory": caps.unified_memory,
                    "notes": if caps.notes.is_empty() { d.info.description.clone() } else { caps.notes.join(", ") },
                })
            })
            .collect();
        let measured_bw = if run_bench {
            Some(crate::devices::measure_cpu_bandwidth_gbs())
        } else {
            None
        };
        let supported_tiers: Vec<&'static str> = cpu_kernels::supported_tiers()
            .into_iter()
            .map(|t| t.name())
            .collect();
        let capped_tiers: Vec<&'static str> = cpu_kernels::capped_tiers()
            .into_iter()
            .map(|t| t.name())
            .collect();
        json!({
            "arch": std::env::consts::ARCH,
            "os": std::env::consts::OS,
            "threads": rayon::current_num_threads(),
            "host_free_memory_bytes": crate::devices::host_free_memory_bytes(),
            "available_memory_bytes": crate::catalog::available_memory_bytes(),
            "compiled_backends": crate::devices::compiled_backends(),
            "cpu_capabilities": crate::devices::cpu_capabilities(),
            "cpu_features": crate::devices::cpu_features(),
            "supported_cpu_tiers": supported_tiers,
            "capped_cpu_tiers": capped_tiers,
            "active_cpu_tier": capped_tiers.last().copied().unwrap_or("scalar"),
            "cpu_repack": cpu_kernels::repack_enabled(),
            "auto_order": crate::devices::auto_order(),
            "auto_choice": crate::devices::device_candidates("auto").ok().and_then(|chain| chain.first().map(|(backend, index)| if backend == "cpu" { "cpu".to_owned() } else { format!("{backend}:{index}") })),
            "measured_cpu_bandwidth_gbs": measured_bw,
            "devices": devices,
        })
    });
    // a graphics driver that hangs while it is asked must not leave the page waiting for ever
    match tokio::time::timeout(std::time::Duration::from_secs(60), probe).await {
        Ok(Ok(val)) => Json(val).into_response(),
        Ok(Err(e)) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("the hardware check failed: {e}")),
        Err(_) => api_error(StatusCode::GATEWAY_TIMEOUT, "the hardware check did not finish within 60 seconds (a graphics driver may not answer)"),
    }
}

pub(in crate::server) async fn selftest_cpu(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Response {
    let tier = body.get("tier").and_then(Value::as_str).map(str::to_owned);
    let saved_cap = state.studio_config.read().unwrap().cpu_tier_cap.clone();
    let result = tokio::task::spawn_blocking(move || {
        let out = crate::cpu_selftest::run_json(tier.as_deref());
        if let Some(cap_name) = saved_cap {
            if let Some(t) = CpuTier::parse(&cap_name) {
                cpu_kernels::set_tier_cap(Some(t));
            }
        }
        out
    })
    .await;

    match result {
        Ok(Ok(val)) => Json(val).into_response(),
        Ok(Err(msg)) => api_error(StatusCode::BAD_REQUEST, &msg),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

pub(in crate::server) async fn models_hub(State(state): State<Arc<AppState>>) -> Response {
    let aliases = state.aliases.read().unwrap().clone();
    models_hub_with(json!({
        "name": state.model_name,
        "arch": state.model_arch,
        "size_bytes": state.facts.size_bytes,
        "parameters": state.facts.parameters,
        "parameter_size": state.facts.parameter_size(),
        "quantization": state.facts.quantization,
        "max_context": state.max_context,
        "trained_context": state.facts.trained_context,
        "digest": state.facts.digest,
        "aliases": aliases,
    }))
    .await
}

/// `models_hub_json` off the request thread (looking at the graphics cards can take a moment, or hang on a bad driver).
pub(in crate::server) async fn models_hub_with(loaded: Value) -> Response {
    let work = tokio::task::spawn_blocking(move || models_hub_json(loaded));
    match tokio::time::timeout(std::time::Duration::from_secs(60), work).await {
        Ok(Ok(value)) => Json(value).into_response(),
        Ok(Err(e)) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
        Err(_) => api_error(StatusCode::GATEWAY_TIMEOUT, "the hardware check did not finish within 60 seconds (a graphics driver may not answer)"),
    }
}

/// General-purpose chat models that are known to run well, best first: what the Studio suggests when they fit into the memory.
const STARTER_MODELS: &[&str] = &[
    "qwen3.5:9b", "qwen3:14b", "gpt-oss:20b", "gemma3:12b", "qwen3:8b", "llama3.1:8b", "qwen3.5:4b", "gemma3:4b", "phi4-mini:3.8b", "llama3.2:3b", "qwen3.5:0.8b",
];

/// The model page data: the loaded model (`Null` when the server runs without one), downloads and suggestions.
pub(in crate::server) fn models_hub_json(loaded: Value) -> Value {
    let avail_mem = crate::catalog::available_memory_bytes();
    let has_discrete_gpu = crate::devices::list_devices().iter().any(|d| d.dev_type == crate::devices::DeviceType::Gpu);
    let cached: Vec<Value> = crate::hub::list_cache()
        .unwrap_or_default()
        .into_iter()
        .map(|c| {
            json!({
                "name": c.reference,
                "revision": c.revision,
                "path": c.path.display().to_string(),
                "size_bytes": c.bytes,
                "note": c.note,
            })
        })
        .collect();

    let catalog: Vec<Value> = crate::catalog::catalog()
        .iter()
        .map(|entry| {
            let params_b = crate::catalog::infer_parameters_billion(&entry.name).unwrap_or(7.0);
            let rec_quant = crate::catalog::recommend_quant(entry, avail_mem);
            let est_bytes = crate::catalog::estimate_quant_bytes(params_b, &rec_quant);
            // 2-bit and 1-bit quantizations run, but answer badly: a model that fits only that way counts as too large
            let very_low_quality = ["Q2", "IQ2", "IQ1"].iter().any(|p| rec_quant.to_ascii_uppercase().starts_with(p));
            let quants_list = if entry.quants.is_empty() {
                crate::catalog::DEFAULT_QUANTS.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>()
            } else {
                entry.quants.clone()
            };
            let variants: Vec<Value> = quants_list
                .iter()
                .map(|q| {
                    let b = crate::catalog::estimate_quant_bytes(params_b, q);
                    json!({
                        "quant": q,
                        "estimated_bytes": b,
                        "fits_memory": avail_mem == 0 || (b as f64) <= (avail_mem as f64) * 0.85,
                    })
                })
                .collect();
            let arch = entry.name.split([':', '-']).next().unwrap_or("llm").to_owned();
            let is_vision = entry.name.contains("vl") || entry.name.contains("vision") || entry.name.contains("gemma3");
            json!({
                "id": entry.name,
                "display_name": entry.name,
                "architecture": arch,
                "default_repo": entry.repo,
                "gguf_repo": entry.repo,
                "default_quant": entry.default_quant,
                "recommended_quant": rec_quant,
                "estimated_bytes": est_bytes,
                "parameters_billion": params_b,
                "context_length": 32768,
                "vision": is_vision,
                "fits": (avail_mem > 0).then_some(est_bytes as f64 * 1.15 <= avail_mem as f64 && !very_low_quality),
                "tags": quants_list,
                "variants": variants,
            })
        })
        .collect();

    // the largest models that run comfortably (at 4 bit they take at most 40 % of the memory, so they also answer quickly), one per family
    let mut picks: Vec<&Value> = Vec::new();
    for wanted in STARTER_MODELS {
        let Some(entry) = catalog.iter().find(|e| e["id"] == *wanted) else { continue };
        let params = entry["parameters_billion"].as_f64().unwrap_or(1000.0);
        let q4 = crate::catalog::estimate_quant_bytes(params, "Q4_K_M");
        // without a discrete graphics card the processor does the work: a model above 10 billion parameters would answer slowly
        let fast_enough = has_discrete_gpu || params <= 10.0;
        if avail_mem > 0 && fast_enough && q4 as f64 * 2.5 <= avail_mem as f64 {
            picks.push(entry);
        }
    }
    let mut families = Vec::new();
    let recommended: Vec<Value> = picks
        .into_iter()
        .filter(|e| {
            let family = e["architecture"].as_str().unwrap_or_default().to_owned();
            !families.contains(&family) && {
                families.push(family);
                true
            }
        })
        .take(3)
        .map(|e| e["id"].clone())
        .collect();    json!({
        "available_memory_bytes": avail_mem,
        "recommended": recommended,
        "loaded": loaded,
        "cached": cached,
        "local": crate::hub::local_models()
            .into_iter()
            .map(|(path, size)| json!({"path": path, "name": std::path::Path::new(&path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(), "size_bytes": size}))
            .collect::<Vec<_>>(),
        "catalog": catalog,
    })
}

pub(in crate::server) async fn hf_files(
    Query(query): Query<HashMap<String, String>>,
) -> Response {
    let Some(repo) = query.get("repo").map(|s| s.trim().to_owned()).filter(|s| !s.is_empty()) else {
        return api_error(StatusCode::BAD_REQUEST, "missing ?repo=owner/name query parameter");
    };
    let repo_clone = repo.clone();
    let res = tokio::task::spawn_blocking(move || crate::hub::repository_files(&repo_clone)).await;
    match res {
        Ok(Ok(files)) => {
            let gguf_files: Vec<Value> = files
                .iter()
                .filter(|(name, _)| name.to_ascii_lowercase().ends_with(".gguf"))
                .map(|(name, size)| json!({"filename": name, "size_bytes": size}))
                .collect();
            let all_files: Vec<Value> = files
                .into_iter()
                .map(|(name, size)| json!({"filename": name, "size_bytes": size}))
                .collect();
            Json(json!({
                "repo": repo,
                "gguf_files": gguf_files,
                "all_files": all_files,
            }))
            .into_response()
        }
        Ok(Err(err)) => api_error(StatusCode::BAD_GATEWAY, &err),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

/// Downloads in progress (a pull or the first half of a model change); `model_progress` reports them.
static ACTIVE_DOWNLOADS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// A catalog name, a Hugging Face reference (`owner/repo[:QUANT]`, `hf://owner/repo/file.gguf`) or a local file, resolved to a file
/// on disk; downloads what is missing.
fn resolve_model(spec: &str, quant: Option<&str>) -> Result<(std::path::PathBuf, String), String> {
    if std::path::Path::new(spec).is_file() {
        return Ok((std::path::PathBuf::from(spec), spec.to_owned()));
    }
    let reference = if let Some(entry) = crate::catalog::find(spec) {
        let q = quant
            .map(str::to_owned)
            .unwrap_or_else(|| crate::catalog::recommend_quant(&entry, crate::catalog::available_memory_bytes()));
        format!("hf://{}:{}", entry.repo, q)
    } else if let Some(q) = quant.filter(|_| !spec.contains(':')) {
        format!("{spec}:{q}")
    } else {
        spec.to_owned()
    };
    ACTIVE_DOWNLOADS.fetch_add(1, Ordering::SeqCst);
    crate::hub::clear_download_progress();
    let path = crate::hub::resolve(&reference, None, false);
    ACTIVE_DOWNLOADS.fetch_sub(1, Ordering::SeqCst);
    Ok((path?, reference))
}

/// `GET /api/studio/models/progress`: the file that is being downloaded, if any.
pub(in crate::server) async fn model_progress() -> Response {
    let active = ACTIVE_DOWNLOADS.load(Ordering::SeqCst) > 0;
    let job = pull_job().map(|j| {
        json!({"model": j.model, "state": j.state, "attempt": j.attempt, "max_attempts": PULL_ATTEMPTS, "error": j.message, "model_path": j.path, "reference": j.reference})
    });
    match crate::hub::download_progress() {
        Some(p) if active || !p.finished => Json(json!({
            "active": active,
            "file": p.file,
            "written": p.written,
            "total": p.total,
            "mb_per_s": p.mb_per_s,
            "job": job,
        }))
        .into_response(),
        _ => Json(json!({"active": active, "job": job})).into_response(),
    }
}

/// `POST /api/studio/models/local` with `{path}`: adds a GGUF file (or the GGUF files of a folder) from this computer to the models.
pub(in crate::server) async fn add_local_model(Json(body): Json<Value>) -> Response {
    let path = body.get("path").and_then(Value::as_str).unwrap_or_default().to_owned();
    match tokio::task::spawn_blocking(move || crate::hub::add_local_model(&path)).await {
        Ok(Ok(added)) => Json(json!({"added": added})).into_response(),
        Ok(Err(error)) => api_error(StatusCode::BAD_REQUEST, &error),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

/// `DELETE /api/studio/models/local` with `{path}`: takes the file off the list (the file stays on disk).
pub(in crate::server) async fn remove_local_model(Json(body): Json<Value>) -> Response {
    let path = body.get("path").and_then(Value::as_str).unwrap_or_default();
    crate::hub::remove_local_model(path);
    Json(json!({"status": "removed"})).into_response()
}

/// `GET /api/studio/hf-account`: whose Hugging Face token is in use (private and gated models need one).
pub(in crate::server) async fn hf_account() -> Response {
    match tokio::task::spawn_blocking(crate::hub::account).await {
        Ok(Ok(Some((name, source)))) => Json(json!({"signed_in": true, "name": name, "source": source})).into_response(),
        Ok(Ok(None)) => Json(json!({"signed_in": false})).into_response(),
        Ok(Err(error)) => Json(json!({"signed_in": false, "error": error})).into_response(),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

/// `POST /api/studio/hf-account` with `{token}`: checks the token with Hugging Face and keeps it; an empty token signs out.
pub(in crate::server) async fn hf_login(Json(body): Json<Value>) -> Response {
    let token = body.get("token").and_then(Value::as_str).unwrap_or_default().to_owned();
    match tokio::task::spawn_blocking(move || crate::hub::save_token(&token)).await {
        Ok(Ok(Some(name))) => Json(json!({"signed_in": true, "name": name, "source": "saved"})).into_response(),
        Ok(Ok(None)) => Json(json!({"signed_in": false})).into_response(),
        Ok(Err(error)) => api_error(StatusCode::BAD_REQUEST, &error),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

/// `POST /api/studio/open-url` with `{url}`: opens a page of Hugging Face or of the fLLM project in the browser of the computer (the
/// window of the desktop app cannot open links itself).
pub(in crate::server) async fn open_url(Json(body): Json<Value>) -> Response {
    let url = body.get("url").and_then(Value::as_str).unwrap_or_default();
    let allowed = ["https://huggingface.co/", "https://github.com/Marco-Lu2508/fLLM"];
    if !allowed.iter().any(|prefix| url.starts_with(prefix)) {
        return api_error(StatusCode::BAD_REQUEST, "only pages of Hugging Face and of the fLLM project can be opened");
    }
    super::super::open_browser(url);
    Json(json!({"status": "opened"})).into_response()
}

/// `GET /api/studio/models/search?q=...`: Hugging Face repositories with GGUF files, most downloaded first.
pub(in crate::server) async fn search_models(Query(query): Query<HashMap<String, String>>) -> Response {
    let text = query.get("q").map(|s| s.trim().to_owned()).unwrap_or_default();
    let res = tokio::task::spawn_blocking(move || crate::hub::search_models(&text, 20)).await;
    match res {
        Ok(Ok(found)) => Json(json!({"results": found})).into_response(),
        Ok(Err(err)) => api_error(StatusCode::BAD_GATEWAY, &err),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

/// The arguments this server was started with, but with `model` as its model (and without a fixed `--model-name`).
fn relaunch_args(model: &str) -> Vec<String> {
    relaunch_args_from(&std::env::args().skip(1).collect::<Vec<_>>(), model)
}

fn relaunch_args_from(given: &[String], model: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < given.len() {
        match given[i].as_str() {
            "--model" | "-m" | "--model-name" => i += 2,
            a if a.starts_with("--model=") || a.starts_with("--model-name=") => i += 1,
            other => {
                out.push(other.to_owned());
                i += 1;
            }
        }
    }
    out.push("--model".into());
    out.push(model.to_owned());
    out
}

/// `POST /api/studio/models/load` with `{model, quant?}`: gets the model (downloading it if needed), starts a new server process
/// with it and ends this one. The new process waits for the port, so the page only has to wait until the server answers again.
pub(in crate::server) async fn load_model(Json(body): Json<Value>) -> Response {
    let Some(spec) = body.get("model").and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty()).map(str::to_owned) else {
        return api_error(StatusCode::BAD_REQUEST, "model specification is required");
    };
    let quant = body.get("quant").and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty()).map(str::to_owned);
    let resolved = tokio::task::spawn_blocking(move || resolve_model(&spec, quant.as_deref())).await;
    let path = match resolved {
        Ok(Ok((path, _))) => path,
        Ok(Err(err)) => return api_error(StatusCode::BAD_REQUEST, &err),
        Err(e) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };
    // a model the engine cannot use is turned down here, while the current one is still running
    let checked = path.clone();
    let usable = tokio::task::spawn_blocking(move || -> Result<(), String> {
        let mapping = fllm_gguf_probe::MappedGguf::open(&checked).map_err(|e| e.to_string())?;
        crate::tokenizer::TokenizerBundle::from_gguf(&mapping).map(|_| ())
    })
    .await;
    match usable {
        Ok(Ok(())) => {}
        Ok(Err(reason)) => return api_error(StatusCode::BAD_REQUEST, &format!("fLLM cannot use this model yet: {reason}")),
        Err(e) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
    let model = path.display().to_string();
    let exe = match std::env::current_exe() {
        Ok(exe) => exe,
        Err(e) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("cannot find the server program: {e}")),
    };
    let started = std::process::Command::new(exe)
        .args(relaunch_args(&model))
        .env("FLLM_REPLACE_SERVER", "1")
        // the desktop app watches its server through this pipe, so the new server keeps it
        .stdin(if std::env::var_os("FLLM_EXIT_ON_STDIN_EOF").is_some() { std::process::Stdio::inherit() } else { std::process::Stdio::null() })
        .spawn();
    if let Err(e) = started {
        return api_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("cannot start the new server: {e}"));
    }
    crate::hub::remember_studio_model(&model);
    tokio::spawn(async {
        tokio::time::sleep(std::time::Duration::from_millis(700)).await;
        std::process::exit(0);
    });
    Json(json!({"status": "restarting", "model_path": model})).into_response()
}

/// The model download that runs in the background (one at a time). The page starts it and polls `model_progress`, so a long download is
/// not tied to one HTTP request, and a dropped connection is resumed by the next attempt.
#[derive(Clone, Default)]
struct PullJob {
    model: String,
    /// `running`, `done` or `failed`
    state: &'static str,
    attempt: u32,
    message: String,
    path: String,
    reference: String,
}

static PULL_JOB: std::sync::Mutex<Option<PullJob>> = std::sync::Mutex::new(None);

fn pull_job() -> Option<PullJob> {
    PULL_JOB.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

fn set_pull_job(job: PullJob) {
    *PULL_JOB.lock().unwrap_or_else(|e| e.into_inner()) = Some(job);
}

/// Whether another try can help: the connection broke or timed out (the partial file stays and the next try continues it).
fn worth_retrying(error: &str) -> bool {
    let lower = error.to_ascii_lowercase();
    ["interrupted", "connection", "timed out", "timeout", "truncated", "error sending", "dns", "reset", "broken pipe", "unexpected eof", "hub request"]
        .iter()
        .any(|word| lower.contains(word))
}

const PULL_ATTEMPTS: u32 = 8;

fn run_pull(spec: String, quant: Option<String>) {
    let mut job = PullJob { model: spec.clone(), state: "running", ..PullJob::default() };
    for attempt in 1..=PULL_ATTEMPTS {
        job.attempt = attempt;
        set_pull_job(job.clone());
        match resolve_model(&spec, quant.as_deref()) {
            Ok((path, reference)) => {
                job.state = "done";
                job.path = path.display().to_string();
                job.reference = reference;
                job.message.clear();
                set_pull_job(job);
                return;
            }
            Err(error) => {
                job.message = error.clone();
                if attempt == PULL_ATTEMPTS || !worth_retrying(&error) {
                    break;
                }
                set_pull_job(job.clone());
                std::thread::sleep(std::time::Duration::from_secs(u64::from(attempt.min(5)) * 3));
            }
        }
    }
    job.state = "failed";
    set_pull_job(job);
}

/// `POST /api/studio/models/pull` with `{model, quant?}`: starts the download and answers at once; `model_progress` tells how it goes.
pub(in crate::server) async fn pull_model(Json(body): Json<Value>) -> Response {
    let Some(spec) = body.get("model").and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty()) else {
        return api_error(StatusCode::BAD_REQUEST, "model specification is required");
    };
    let quant = body.get("quant").and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty()).map(str::to_owned);
    if pull_job().is_some_and(|job| job.state == "running") {
        return api_error(StatusCode::CONFLICT, "a download is already running");
    }
    set_pull_job(PullJob { model: spec.to_owned(), state: "running", attempt: 1, ..PullJob::default() });
    let spec = spec.to_owned();
    if let Err(e) = std::thread::Builder::new().name("fllm-model-pull".into()).spawn(move || run_pull(spec, quant)) {
        set_pull_job(PullJob { state: "failed", message: format!("cannot start the download: {e}"), ..PullJob::default() });
        return api_error(StatusCode::INTERNAL_SERVER_ERROR, "cannot start the download");
    }
    Json(json!({"status": "started"})).into_response()
}

pub(in crate::server) async fn delete_cached_model(Json(body): Json<Value>) -> Response {
    let Some(name) = body.get("name").and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty()) else {
        return api_error(StatusCode::BAD_REQUEST, "cached reference name is required");
    };
    match crate::hub::remove_cached(name) {
        Ok(removed_count) => Json(json!({
            "status": "deleted",
            "removed": removed_count,
        }))
        .into_response(),
        Err(err) => api_error(StatusCode::BAD_REQUEST, &err),
    }
}

pub(in crate::server) async fn list_keys(State(state): State<Arc<AppState>>) -> Response {
    let rl = state.http.security.rate_limit_spec();
    Json(json!({
        "auth_required": state.http.security.auth_required(),
        "rate_limit": {
            "requests_per_minute": rl.requests_per_minute,
            "tokens_per_minute": rl.tokens_per_minute,
        },
        "keys": state.http.security.list_keys(),
    }))
    .into_response()
}

/// A new secret from the operating system's random source (24 bytes, 192 bits).
fn generate_random_api_key() -> Result<String, String> {
    let mut bytes = [0u8; 24];
    getrandom::fill(&mut bytes).map_err(|e| format!("the system random source is not available: {e}"))?;
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    Ok(format!("fllm-sk-{hex}"))
}

pub(in crate::server) async fn create_key(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Response {
    let raw_key = match body
        .get("key")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
    {
        Some(key) => key,
        None => match generate_random_api_key() {
            Ok(key) => key,
            Err(err) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, &err),
        },
    };
    let label = body
        .get("label")
        .and_then(Value::as_str)
        .unwrap_or("Studio Key");
    let rpm = body.get("rpm").and_then(Value::as_f64).filter(|&v| v > 0.0);
    let tpm = body.get("tpm").and_then(Value::as_f64).filter(|&v| v > 0.0);

    match state.http.security.add_key(&raw_key, label, rpm, tpm) {
        Ok(key_info) => Json(json!({
            "status": "created",
            "secret": raw_key,
            "key": key_info,
            "auth_required": state.http.security.auth_required(),
            "keys": state.http.security.list_keys(),
        }))
        .into_response(),
        Err(err) => api_error(StatusCode::BAD_REQUEST, &err),
    }
}

pub(in crate::server) async fn delete_key(
    State(state): State<Arc<AppState>>,
    AxumPath(id): AxumPath<String>,
) -> Response {
    if state.http.security.remove_key(&id) {
        Json(json!({
            "status": "deleted",
            "id": id,
            "auth_required": state.http.security.auth_required(),
            "keys": state.http.security.list_keys(),
        }))
        .into_response()
    } else {
        api_error(StatusCode::NOT_FOUND, &format!("API key '{id}' not found"))
    }
}

#[cfg(test)]
mod tests {
    use super::relaunch_args_from;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn a_model_change_keeps_every_other_option() {
        let out = relaunch_args_from(&args(&["serve", "--model", "old.gguf", "--port", "9000", "--model-name", "old", "--ctx", "8192"]), "new.gguf");
        assert_eq!(out, args(&["serve", "--port", "9000", "--ctx", "8192", "--model", "new.gguf"]));
        let out = relaunch_args_from(&args(&["studio", "--model=old.gguf", "--model-name=x"]), "new.gguf");
        assert_eq!(out, args(&["studio", "--model", "new.gguf"]));
        let out = relaunch_args_from(&args(&["serve"]), "m.gguf");
        assert_eq!(out, args(&["serve", "--model", "m.gguf"]));
    }
}

// ------------------------------------------------------------------------------------------------------------------------------
// Updates: the app asks GitHub for a newer release and can fetch and start its installer
// ------------------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Default)]
struct UpdateInstall {
    /// `idle`, `downloading`, `started` or `failed`
    state: &'static str,
    written: u64,
    total: u64,
    message: String,
}

static UPDATE_INSTALL: std::sync::Mutex<UpdateInstall> = std::sync::Mutex::new(UpdateInstall { state: "idle", written: 0, total: 0, message: String::new() });

fn version_parts(version: &str) -> Vec<u64> {
    version.trim_start_matches('v').split(['.', '-']).map_while(|part| part.parse().ok()).collect()
}

/// The installer of this system in the assets of a release: `(file name, download URL)`.
fn installer_asset(assets: &[Value]) -> Option<(String, String)> {
    let wanted: &[&str] = if cfg!(windows) {
        &["-setup.exe"]
    } else if cfg!(target_os = "linux") {
        &[".AppImage", ".deb"]
    } else {
        &[]
    };
    wanted.iter().find_map(|suffix| {
        assets.iter().find_map(|asset| {
            let name = asset.get("name")?.as_str()?;
            let url = asset.get("browser_download_url")?.as_str()?;
            name.ends_with(suffix).then(|| (name.to_owned(), url.to_owned()))
        })
    })
}

fn check_update() -> Result<Value, String> {
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(15))
        .timeout(std::time::Duration::from_secs(30))
        .user_agent("fllm-rust/0.1")
        .build()
        .map_err(|e| e.to_string())?;
    let releases: Vec<Value> = client
        .get("https://api.github.com/repos/Marco-Lu2508/fLLM/releases?per_page=10")
        .send()
        .and_then(|r| r.error_for_status())
        .map_err(|e| format!("cannot ask GitHub for updates: {e}"))?
        .json()
        .map_err(|e| e.to_string())?;
    let current = env!("CARGO_PKG_VERSION");
    let newest = releases
        .iter()
        .filter(|r| !r.get("draft").and_then(Value::as_bool).unwrap_or(false))
        .max_by_key(|r| version_parts(r.get("tag_name").and_then(Value::as_str).unwrap_or_default()));
    let Some(release) = newest else {
        return Ok(json!({"current": current, "newer": false}));
    };
    let tag = release.get("tag_name").and_then(Value::as_str).unwrap_or_default();
    let newer = version_parts(tag) > version_parts(current);
    let assets = release.get("assets").and_then(Value::as_array).cloned().unwrap_or_default();
    let installer = installer_asset(&assets);
    Ok(json!({
        "current": current,
        "latest": tag.trim_start_matches('v'),
        "newer": newer,
        "page": release.get("html_url"),
        "installer": installer.map(|(name, _)| name),
    }))
}

/// `GET /api/studio/update`: is there a newer release, and how far is the installer download.
pub(in crate::server) async fn update_status() -> Response {
    match tokio::task::spawn_blocking(check_update).await {
        Ok(Ok(mut value)) => {
            let install = UPDATE_INSTALL.lock().unwrap_or_else(|e| e.into_inner()).clone();
            value["install"] = json!({"state": if install.state.is_empty() { "idle" } else { install.state }, "written": install.written, "total": install.total, "error": install.message});
            Json(value).into_response()
        }
        Ok(Err(error)) => Json(json!({"current": env!("CARGO_PKG_VERSION"), "newer": false, "error": error})).into_response(),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

fn set_install(state: &'static str, written: u64, total: u64, message: &str) {
    *UPDATE_INSTALL.lock().unwrap_or_else(|e| e.into_inner()) = UpdateInstall { state, written, total, message: message.to_owned() };
}

fn fetch_installer() -> Result<(), String> {
    let releases: Vec<Value> = reqwest::blocking::Client::builder()
        .user_agent("fllm-rust/0.1")
        .build()
        .map_err(|e| e.to_string())?
        .get("https://api.github.com/repos/Marco-Lu2508/fLLM/releases?per_page=10")
        .send()
        .and_then(|r| r.error_for_status())
        .map_err(|e| format!("cannot ask GitHub for updates: {e}"))?
        .json()
        .map_err(|e| e.to_string())?;
    let release = releases
        .iter()
        .filter(|r| !r.get("draft").and_then(Value::as_bool).unwrap_or(false))
        .max_by_key(|r| version_parts(r.get("tag_name").and_then(Value::as_str).unwrap_or_default()))
        .ok_or("no release found")?;
    let assets = release.get("assets").and_then(Value::as_array).cloned().unwrap_or_default();
    let (name, url) = installer_asset(&assets).ok_or("this release has no installer for this system")?;
    let target = std::env::temp_dir().join(&name);
    let mut response = reqwest::blocking::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(20))
        .timeout(std::time::Duration::from_secs(3600))
        .user_agent("fllm-rust/0.1")
        .build()
        .map_err(|e| e.to_string())?
        .get(&url)
        .send()
        .and_then(|r| r.error_for_status())
        .map_err(|e| format!("cannot download the installer: {e}"))?;
    let total = response.content_length().unwrap_or(0);
    let mut file = std::fs::File::create(&target).map_err(|e| format!("cannot write {}: {e}", target.display()))?;
    let mut written = 0u64;
    let mut buffer = vec![0u8; 256 * 1024];
    loop {
        let n = std::io::Read::read(&mut response, &mut buffer).map_err(|e| format!("download interrupted: {e}"))?;
        if n == 0 {
            break;
        }
        std::io::Write::write_all(&mut file, &buffer[..n]).map_err(|e| e.to_string())?;
        written += n as u64;
        set_install("downloading", written, total, "");
    }
    drop(file);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o755));
    }
    // the installer asks to close the app before it replaces it
    let started = if cfg!(windows) {
        std::process::Command::new(&target).spawn().map(|_| ())
    } else if name.ends_with(".AppImage") {
        std::process::Command::new(&target).spawn().map(|_| ())
    } else {
        std::process::Command::new("xdg-open").arg(&target).spawn().map(|_| ())
    };
    started.map_err(|e| format!("cannot start the installer {}: {e}", target.display()))?;
    set_install("started", written, total, &target.display().to_string());
    Ok(())
}

/// `POST /api/studio/update/install`: downloads the installer of the newest release in the background and starts it.
pub(in crate::server) async fn update_install() -> Response {
    {
        let mut state = UPDATE_INSTALL.lock().unwrap_or_else(|e| e.into_inner());
        if state.state == "downloading" {
            return api_error(StatusCode::CONFLICT, "the installer is already downloading");
        }
        *state = UpdateInstall { state: "downloading", ..UpdateInstall::default() };
    }
    let spawned = std::thread::Builder::new().name("fllm-update".into()).spawn(|| {
        if let Err(error) = fetch_installer() {
            set_install("failed", 0, 0, &error);
        }
    });
    if let Err(e) = spawned {
        set_install("failed", 0, 0, &e.to_string());
        return api_error(StatusCode::INTERNAL_SERVER_ERROR, "cannot start the download");
    }
    Json(json!({"status": "started"})).into_response()
}