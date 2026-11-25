//! DSpine (`DSpineDraftModel`) and DFlash (`DFlashDraftModel` / `DFlash2DraftModel`) parallel
//! speculative decoding draft models.
//!
//! These checkpoints share `embed_tokens` and `lm_head` with the target model (and ship without
//! `tokenizer.json`), project concatenated hidden states from `target_layer_ids` through
//! `fc` + `hidden_norm`, run Qwen3-style decoder layers with block-causal attention over
//! `[context_kv, block_kv]`, and (for DSpine) apply adjacent causal injection via
//! `selected_prefix_bridge` plus a serial `RefinementTail` over the top-K candidates.
use crate::gguf::Meta;
use crate::plan::{rope_params, Family, Mapped, Plan};
use serde_json::Value;

fn num(v: &Value, key: &str) -> Option<u64> {
    v.get(key).and_then(Value::as_u64)
}

fn flt(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(Value::as_f64)
}

/// True when `config.json` describes a DSpine or DFlash draft checkpoint.
pub(crate) fn is_dspine_or_dflash(cfg: &Value) -> bool {
    if cfg.get("dflash_config").is_some_and(Value::is_object) {
        return true;
    }
    cfg.get("architectures")
        .and_then(Value::as_array)
        .is_some_and(|list| {
            list.iter().filter_map(Value::as_str).any(|a| {
                matches!(
                    a,
                    "DSpineDraftModel" | "DFlashDraftModel" | "DFlash2DraftModel"
                )
            })
        })
}

pub(crate) fn is_dspine_arch(arch: &str) -> bool {
    matches!(arch, "dspine" | "dflash")
}

fn resolve_head_topk(cfg: &Value, dflash: &Value, num_layers: u64) -> Result<u32, String> {
    let mut found: Vec<u64> = Vec::new();
    if let Some(second) = cfg.get("code_bridge_second_pass").filter(|v| !v.is_null()) {
        let obj = second
            .as_object()
            .ok_or("code_bridge_second_pass must be an object")?;
        for k in obj.keys() {
            if !matches!(k.as_str(), "mode" | "first_exact" | "head_topk") {
                return Err(format!("unknown code_bridge_second_pass key {k:?}"));
            }
        }
        if obj.get("mode").and_then(Value::as_str) != Some("serial_table") {
            return Err("DSpine requires code_bridge_second_pass.mode = \"serial_table\"".into());
        }
        if obj.get("first_exact").and_then(Value::as_u64) != Some(num_layers.saturating_sub(1)) {
            return Err(format!(
                "code_bridge_second_pass.first_exact must be {}",
                num_layers.saturating_sub(1)
            ));
        }
        let k = obj
            .get("head_topk")
            .and_then(Value::as_u64)
            .ok_or("code_bridge_second_pass requires head_topk")?;
        found.push(k);
    }
    if let Some(refine) = dflash
        .get("selected_prefix_last_write_refine")
        .filter(|v| !v.is_null())
    {
        let obj = refine
            .as_object()
            .ok_or("selected_prefix_last_write_refine must be an object")?;
        if obj.get("kind").and_then(Value::as_str) != Some("exact_sender") {
            return Err(
                "dflash_config.selected_prefix_last_write_refine.kind must be \"exact_sender\""
                    .into(),
            );
        }
        let k = obj
            .get("head_topk")
            .and_then(Value::as_u64)
            .ok_or("selected_prefix_last_write_refine requires head_topk")?;
        found.push(k);
    }
    if found.is_empty() {
        return Err("DSpine checkpoint config names no refinement width (head_topk)".into());
    }
    let first = found[0];
    if found.iter().any(|&v| v != first) {
        return Err("DSpine refinement head_topk values disagree".into());
    }
    if !first.is_power_of_two() || !(2..=64).contains(&first) {
        return Err(format!(
            "DSpine head_topk must be a power of two in 2..=64, got {first}"
        ));
    }
    Ok(first as u32)
}

fn parse_sliding_window(cfg: &Value, layers: u64) -> Option<u32> {
    if cfg.get("use_sliding_window").and_then(Value::as_bool) == Some(false) {
        return None;
    }
    let w = num(cfg, "sliding_window").filter(|&v| v > 0)?;
    if let Some(types) = cfg.get("layer_types").and_then(Value::as_array) {
        let has_swa = types
            .iter()
            .take(layers as usize)
            .any(|v| v.as_str() == Some("sliding_attention"));
        if !has_swa {
            return None;
        }
    }
    Some(w as u32)
}

pub(crate) fn plan(cfg: &Value) -> Result<Plan, String> {
    let dflash = cfg
        .get("dflash_config")
        .and_then(Value::as_object)
        .ok_or("DSpine/DFlash config.json requires dflash_config object")?;
    let dflash_val = Value::Object(dflash.clone());

    let hidden = num(cfg, "hidden_size").ok_or("config.json requires hidden_size")?;
    let layers = num(cfg, "num_hidden_layers").ok_or("config.json requires num_hidden_layers")?;
    let inter = num(cfg, "intermediate_size").ok_or("config.json requires intermediate_size")?;
    let heads =
        num(cfg, "num_attention_heads").ok_or("config.json requires num_attention_heads")?;
    let kv_heads = num(cfg, "num_key_value_heads").unwrap_or(heads);
    let head_dim = num(cfg, "head_dim").unwrap_or(hidden / heads);
    let vocab = num(cfg, "vocab_size")
        .or_else(|| num(cfg, "draft_vocab_size"))
        .ok_or("config.json requires vocab_size")?;
    let ctx = num(cfg, "max_position_embeddings").unwrap_or(32768);
    let eps = flt(cfg, "rms_norm_eps").unwrap_or(1e-6);
    let rp = rope_params(cfg, None);
    let rope_theta = rp.theta.unwrap_or(10_000_000.0);

    let block_size = num(cfg, "block_size")
        .or_else(|| num(&dflash_val, "block_size"))
        .unwrap_or(16);
    if !(2..=32).contains(&block_size) {
        return Err(format!(
            "DSpine/DFlash block_size must be in 2..=32, got {block_size}"
        ));
    }
    let mask_token_id = num(&dflash_val, "mask_token_id")
        .ok_or("dflash_config requires mask_token_id")?;
    let target_layer_ids: Vec<u32> = dflash_val
        .get("target_layer_ids")
        .and_then(Value::as_array)
        .ok_or("dflash_config requires target_layer_ids")?
        .iter()
        .map(|v| {
            v.as_u64()
                .map(|x| x as u32)
                .ok_or_else(|| "target_layer_ids must be non-negative integers".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if target_layer_ids.is_empty() {
        return Err("dflash_config.target_layer_ids must not be empty".into());
    }
    let num_target_layers = num(cfg, "num_target_layers")
        .or_else(|| num(&dflash_val, "num_target_layers"))
        .unwrap_or_else(|| target_layer_ids.iter().copied().max().unwrap_or(0) as u64 + 1);
    let block_causal = dflash_val
        .get("draft_block_causal")
        .and_then(Value::as_bool)
        .unwrap_or(true);

    let bridge_obj = dflash_val
        .get("selected_prefix_code_bridge")
        .and_then(Value::as_object);
    let has_bridge = bridge_obj.is_some();
    let arch = if has_bridge { "dspine" } else { "dflash" };
    let name = cfg
        .get("_name_or_path")
        .and_then(Value::as_str)
        .unwrap_or(arch);

    let mut meta = vec![
        ("general.architecture".into(), Meta::Str(arch.into())),
        ("general.name".into(), Meta::Str(name.into())),
        (format!("{arch}.block_count"), Meta::U32(layers as u32)),
        (
            format!("{arch}.embedding_length"),
            Meta::U32(hidden as u32),
        ),
        (
            format!("{arch}.feed_forward_length"),
            Meta::U32(inter as u32),
        ),
        (
            format!("{arch}.attention.head_count"),
            Meta::U32(heads as u32),
        ),
        (
            format!("{arch}.attention.head_count_kv"),
            Meta::U32(kv_heads as u32),
        ),
        (
            format!("{arch}.attention.key_length"),
            Meta::U32(head_dim as u32),
        ),
        (
            format!("{arch}.attention.value_length"),
            Meta::U32(head_dim as u32),
        ),
        (
            format!("{arch}.attention.layer_norm_rms_epsilon"),
            Meta::F32(eps as f32),
        ),
        (format!("{arch}.rope.freq_base"), Meta::F32(rope_theta as f32)),
        (
            format!("{arch}.rope.dimension_count"),
            Meta::U32(head_dim as u32),
        ),
        (format!("{arch}.rope.neox"), Meta::Bool(true)),
        (format!("{arch}.context_length"), Meta::U32(ctx as u32)),
        (format!("{arch}.vocab_size"), Meta::U32(vocab as u32)),
        (format!("{arch}.block_size"), Meta::U32(block_size as u32)),
        (
            format!("{arch}.mask_token_id"),
            Meta::U32(mask_token_id as u32),
        ),
        (
            format!("{arch}.target_layer_ids"),
            Meta::U32s(target_layer_ids),
        ),
        (
            format!("{arch}.num_target_layers"),
            Meta::U32(num_target_layers as u32),
        ),
        (format!("{arch}.block_causal"), Meta::Bool(block_causal)),
        (format!("{arch}.has_bridge"), Meta::Bool(has_bridge)),
    ];
    if let Some(w) = parse_sliding_window(cfg, layers) {
        meta.push((format!("{arch}.sliding_window"), Meta::U32(w)));
    }
    if let Some(bridge) = bridge_obj {
        if dflash_val
            .get("selected_prefix_terminal")
            .and_then(Value::as_bool)
            != Some(false)
        {
            return Err("DSpine requires dflash_config.selected_prefix_terminal = false".into());
        }
        if dflash_val
            .get("selected_prefix_feature_writes")
            .and_then(Value::as_bool)
            != Some(true)
        {
            return Err("DSpine requires dflash_config.selected_prefix_feature_writes = true".into());
        }
        if dflash_val
            .get("selected_prefix_feature_write_mode")
            .and_then(Value::as_str)
            != Some("independent")
        {
            return Err(
                "DSpine requires dflash_config.selected_prefix_feature_write_mode = \"independent\""
                    .into(),
            );
        }
        if !block_causal {
            return Err("DSpine requires dflash_config.draft_block_causal = true".into());
        }
        let code_size = bridge
            .get("code_size")
            .and_then(Value::as_u64)
            .filter(|&v| v > 0)
            .ok_or("selected_prefix_code_bridge requires positive code_size")?;
        let rank = bridge
            .get("rank")
            .and_then(Value::as_u64)
            .filter(|&v| v > 0)
            .ok_or("selected_prefix_code_bridge requires positive rank")?;
        let bridge_eps = bridge
            .get("eps")
            .and_then(Value::as_f64)
            .filter(|&v| v > 0.0)
            .unwrap_or(1e-6);
        let feature_layers: Vec<u32> = dflash_val
            .get("selected_prefix_feature_layers")
            .and_then(Value::as_array)
            .ok_or("dflash_config requires selected_prefix_feature_layers")?
            .iter()
            .map(|v| {
                let one_based = v
                    .as_u64()
                    .filter(|&x| x >= 1 && x <= layers)
                    .ok_or_else(|| "selected_prefix_feature_layers out of range".to_owned())?;
                Ok((one_based - 1) as u32)
            })
            .collect::<Result<Vec<_>, String>>()?;
        if !feature_layers.contains(&((layers - 1) as u32)) {
            return Err("DSpine requires the last draft layer in selected_prefix_feature_layers".into());
        }
        let head_topk = resolve_head_topk(cfg, &dflash_val, layers)?;
        let terminal_residual_rank = dflash_val
            .get("selected_prefix_terminal_residual_rank")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;
        meta.extend([
            (format!("{arch}.code_size"), Meta::U32(code_size as u32)),
            (format!("{arch}.rank"), Meta::U32(rank as u32)),
            (format!("{arch}.bridge_eps"), Meta::F32(bridge_eps as f32)),
            (format!("{arch}.head_topk"), Meta::U32(head_topk)),
            (
                format!("{arch}.feature_layers"),
                Meta::U32s(feature_layers),
            ),
            (
                format!("{arch}.terminal_residual_rank"),
                Meta::U32(terminal_residual_rank),
            ),
        ]);
    }

    Ok(Plan {
        arch: arch.into(),
        family: Family::Qwen,
        meta,
        norm_add: 0.0,
        layers,
        moe: false,
        gated: true,
        moe_layers: vec![false; layers as usize],
        q_lora: false,
        mixtral_names: false,
        synthesized: Vec::new(),
        linear: None,
        kinds: Vec::new(),
        classic: None,
    })
}

pub(crate) fn map(name: &str) -> Result<Mapped, String> {
    let short = name.strip_prefix("model.").unwrap_or(name);
    let plain = |out: String| Ok(Mapped::Tensor { name: out, add: 0.0 });
    match short {
        "fc.weight" => return plain("dspine.fc.weight".into()),
        "hidden_norm.weight" => return plain("dspine.hidden_norm.weight".into()),
        "norm.weight" => return plain("output_norm.weight".into()),
        "draft_block_causal_version"
        | "last_write_refine_version"
        | "selected_prefix_bridge.independent_write_version"
        | "selected_prefix_bridge.terminal_disabled_version" => return Ok(Mapped::Skip),
        "selected_prefix_bridge.codebook" => return plain("dspine.bridge.codebook".into()),
        "selected_prefix_bridge.read_weight" => return plain("dspine.bridge.read_weight".into()),
        "selected_prefix_bridge.early_read_weights" => {
            return plain("dspine.bridge.early_read_weights".into())
        }
        "selected_prefix_bridge.feature_write_weights" => {
            return plain("dspine.bridge.feature_write_weights".into())
        }
        "selected_prefix_bridge.gate.weight" => return plain("dspine.bridge.gate.weight".into()),
        "selected_prefix_bridge.gate.bias" => return plain("dspine.bridge.gate.bias".into()),
        "selected_prefix_bridge.predecessor_gate.weight" => {
            return plain("dspine.bridge.predecessor_gate.weight".into())
        }
        "selected_prefix_bridge.prev.weight" => return plain("dspine.bridge.prev.weight".into()),
        "selected_prefix_bridge.out.weight" => return plain("dspine.bridge.out.weight".into()),
        "selected_prefix_bridge.terminal_residual_embedding" => {
            return plain("dspine.bridge.terminal_residual_embedding".into())
        }
        "selected_prefix_bridge.terminal_residual_projection" => {
            return plain("dspine.bridge.terminal_residual_projection".into())
        }
        _ => {}
    }
    if let Some(rest) = short.strip_prefix("layers.") {
        let (idx_str, suf) = rest
            .split_once('.')
            .ok_or_else(|| format!("unrecognized DSpine layer tensor {name}"))?;
        let idx: u64 = idx_str
            .parse()
            .map_err(|_| format!("bad layer index in {name}"))?;
        let at = |s: &str| format!("blk.{idx}.{s}");
        return match suf {
            "input_layernorm.weight" => plain(at("attn_norm.weight")),
            "post_attention_layernorm.weight" => plain(at("ffn_norm.weight")),
            "self_attn.q_proj.weight" => plain(at("attn_q.weight")),
            "self_attn.k_proj.weight" => plain(at("attn_k.weight")),
            "self_attn.v_proj.weight" => plain(at("attn_v.weight")),
            "self_attn.o_proj.weight" => plain(at("attn_output.weight")),
            "self_attn.q_norm.weight" => plain(at("attn_q_norm.weight")),
            "self_attn.k_norm.weight" => plain(at("attn_k_norm.weight")),
            "mlp.gate_proj.weight" => plain(at("ffn_gate.weight")),
            "mlp.up_proj.weight" => plain(at("ffn_up.weight")),
            "mlp.down_proj.weight" => plain(at("ffn_down.weight")),
            "self_attn.rotary_emb.inv_freq" => Ok(Mapped::Skip),
            _ => Err(format!("unsupported DSpine/DFlash tensor {name}")),
        };
    }
    Err(format!("unsupported DSpine/DFlash tensor {name}"))
}

pub(crate) fn required_top_tensors(has_bridge: bool, terminal_residual_rank: u32) -> Vec<&'static str> {
    let mut req = vec!["output_norm.weight", "dspine.fc.weight", "dspine.hidden_norm.weight"];
    if has_bridge {
        req.extend([
            "dspine.bridge.codebook",
            "dspine.bridge.read_weight",
            "dspine.bridge.early_read_weights",
            "dspine.bridge.feature_write_weights",
            "dspine.bridge.gate.weight",
            "dspine.bridge.gate.bias",
            "dspine.bridge.predecessor_gate.weight",
            "dspine.bridge.prev.weight",
        ]);
        if terminal_residual_rank > 0 {
            req.extend([
                "dspine.bridge.terminal_residual_embedding",
                "dspine.bridge.terminal_residual_projection",
            ]);
        }
    }
    req
}
