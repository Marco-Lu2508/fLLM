//! xLSTM (`xLSTMForCausalLM`, NX-AI/xLSTM-7b): blocks of an mLSTM layer (matrix memory with exponential input gate and sigmoid forget gate, per-head layer norm, output gate)
//! and a gated SiLU feed-forward, each behind an RMS norm. There is no attention and no cache of keys; a sequence keeps one matrix, one vector and one maximum per head and
//! layer. The plan is the Llama plan of a normalised configuration renamed to the GGUF architecture `xlstm`.
use crate::gguf::Meta;
use crate::plan::{Mapped, Plan};
use serde_json::{json, Value};

fn num(cfg: &Value, key: &str) -> Option<u64> {
    cfg.get(key).and_then(Value::as_u64)
}

fn flt(cfg: &Value, key: &str) -> Option<f64> {
    cfg.get(key).and_then(Value::as_f64)
}

pub(crate) fn config(cfg: &Value) -> Result<Value, String> {
    let hidden = num(cfg, "hidden_size").ok_or("config.json requires hidden_size")?;
    let heads = num(cfg, "num_heads").ok_or("config.json requires num_heads")?;
    let layers = num(cfg, "num_blocks").or_else(|| num(cfg, "num_hidden_layers")).ok_or("config.json requires num_blocks")?;
    if num(cfg, "embedding_dim").is_some_and(|e| e != hidden) {
        return Err("xLSTM with embedding_dim different from hidden_size is not supported".into());
    }
    if flt(cfg, "v_dim_factor").is_some_and(|f| f != 1.0) || cfg.get("use_bias").and_then(Value::as_bool).unwrap_or(false) || cfg.get("weight_mode").and_then(Value::as_str).is_some_and(|m| m != "single") {
        return Err("xLSTM is supported with v_dim_factor 1, no biases and unfused weights".into());
    }
    if heads == 0 || !hidden.is_multiple_of(heads) {
        return Err("xLSTM hidden_size must be a multiple of num_heads".into());
    }
    let factor = flt(cfg, "ffn_proj_factor").unwrap_or(2.667);
    let round = num(cfg, "ffn_round_up_to_multiple_of").unwrap_or(64).max(1);
    let up = ((hidden as f64 * factor + round as f64 - 1.0) / round as f64).floor() as u64 * round;
    let mut out = json!({
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "num_key_value_heads": heads,
        "head_dim": hidden / heads,
        "num_hidden_layers": layers,
        "intermediate_size": up,
        "vocab_size": num(cfg, "vocab_size").ok_or("config.json requires vocab_size")?,
        "rms_norm_eps": flt(cfg, "norm_eps").unwrap_or(1e-6),
        "max_position_embeddings": num(cfg, "max_position_embeddings").unwrap_or(131072),
        "hidden_act": "silu",
        "tie_word_embeddings": cfg.get("tie_word_embeddings").and_then(Value::as_bool).unwrap_or(false),
    });
    for key in ["bos_token_id", "eos_token_id", "pad_token_id", "_name_or_path"] {
        if let Some(v) = cfg.get(key) {
            out[key] = v.clone();
        }
    }
    Ok(out)
}

pub(crate) fn plan(cfg: &Value) -> Result<Plan, String> {
    let normalized = config(cfg)?;
    let mut plan = Plan::from_config(&normalized)?;
    for (key, value) in &mut plan.meta {
        *key = key.replace("llama.", "xlstm.");
        if key == "general.architecture" || (key == "general.name" && cfg.get("_name_or_path").is_none()) {
            *value = Meta::Str("xlstm".into());
        }
    }
    plan.arch = "xlstm".into();
    plan.meta.retain(|(k, _)| !matches!(k.as_str(), "xlstm.rope.freq_base" | "xlstm.rope.dimension_count"));
    plan.meta.push(("xlstm.gate_soft_cap".into(), Meta::F32(flt(cfg, "gate_soft_cap").unwrap_or(15.0) as f32)));
    plan.meta.push(("xlstm.cell_epsilon".into(), Meta::F32(flt(cfg, "eps").unwrap_or(1e-6) as f32)));
    if let Some(cap) = flt(cfg, "output_logit_soft_cap").filter(|c| *c > 0.0) {
        plan.meta.retain(|(k, _)| k != "xlstm.final_logit_softcapping");
        plan.meta.push(("xlstm.final_logit_softcapping".into(), Meta::F32(cap as f32)));
    }
    Ok(plan)
}

/// GGUF name of any tensor of an xLSTM checkpoint (`None`: not an xLSTM tensor).
pub(crate) fn map_any(n: &str) -> Option<Mapped> {
    let plain = |name: String| Some(Mapped::Tensor { name, add: 0.0 });
    match n {
        "backbone.embeddings.weight" => return plain("token_embd.weight".into()),
        "backbone.out_norm.weight" => return plain("output_norm.weight".into()),
        "lm_head.weight" => return plain("output.weight".into()),
        _ => {}
    }
    let rest = n.strip_prefix("backbone.blocks.")?;
    let (i, suf) = rest.split_once('.')?;
    let i: u64 = i.parse().ok()?;
    let name = match suf {
        "norm_mlstm.weight" => "attn_norm.weight",
        "norm_ffn.weight" => "ffn_norm.weight",
        "mlstm_layer.q.weight" => "xlstm_q.weight",
        "mlstm_layer.k.weight" => "xlstm_k.weight",
        "mlstm_layer.v.weight" => "xlstm_v.weight",
        "mlstm_layer.ogate_preact.weight" => "xlstm_o.weight",
        "mlstm_layer.igate_preact.weight" => "xlstm_i.weight",
        "mlstm_layer.igate_preact.bias" => "xlstm_i.bias",
        "mlstm_layer.fgate_preact.weight" => "xlstm_f.weight",
        "mlstm_layer.fgate_preact.bias" => "xlstm_f.bias",
        "mlstm_layer.multihead_norm.weight" => "xlstm_norm.weight",
        "mlstm_layer.out_proj.weight" => "attn_output.weight",
        "ffn.proj_up_gate.weight" => "ffn_gate.weight",
        "ffn.proj_up.weight" => "ffn_up.weight",
        "ffn.proj_down.weight" => "ffn_down.weight",
        _ => return None,
    };
    plain(format!("blk.{i}.{name}"))
}

pub(crate) fn layer_requirements() -> [&'static str; 15] {
    [
        "attn_norm.weight", "ffn_norm.weight", "xlstm_q.weight", "xlstm_k.weight", "xlstm_v.weight", "xlstm_o.weight", "xlstm_i.weight", "xlstm_i.bias", "xlstm_f.weight",
        "xlstm_f.bias", "xlstm_norm.weight", "attn_output.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
    ]
}
