//! JetMoE (`JetMoeForCausalLM`): every layer has a mixture of attention heads and a mixture of feed-forward experts. In the attention the router picks `top_k`
//! experts per token, each expert projects the token to a query of `kv_heads` heads (`input_linear`, the heads of one expert side by side, experts after each
//! other), all of them attend over the one shared key / value of the token (`kv_proj`), and the per-expert outputs go through `output_linear`, are weighted by the
//! (softmax over the selected logits) gates and summed, plus a bias. The feed-forward experts are the usual gated experts with fused `input_linear` (gate, up) and a
//! bias behind the sum. The plan is the Mixtral plan of a normalised configuration renamed to the GGUF architecture `jetmoe`; the attention experts keep their
//! stacked tensors (`attn_moa_*`, read by `rust_decoder/jetmoe.rs`).
use crate::gguf::Meta;
use crate::plan::{Mapped, Plan};
use serde_json::{json, Value};

fn num(cfg: &Value, key: &str) -> Option<u64> {
    cfg.get(key).and_then(Value::as_u64)
}

pub(crate) fn config(cfg: &Value) -> Result<Value, String> {
    let hidden = num(cfg, "hidden_size").ok_or("config.json requires hidden_size")?;
    let kv_heads = num(cfg, "num_key_value_heads").ok_or("config.json requires num_key_value_heads")?;
    let top_k = num(cfg, "num_experts_per_tok").ok_or("config.json requires num_experts_per_tok")?;
    let head_dim = num(cfg, "kv_channels").ok_or("config.json requires kv_channels")?;
    let experts = num(cfg, "num_local_experts").ok_or("config.json requires num_local_experts")?;
    let mut out = json!({
        "model_type": "mixtral",
        "architectures": ["MixtralForCausalLM"],
        "hidden_size": hidden,
        "num_attention_heads": kv_heads * top_k,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "num_hidden_layers": num(cfg, "num_hidden_layers").ok_or("config.json requires num_hidden_layers")?,
        "intermediate_size": num(cfg, "intermediate_size").ok_or("config.json requires intermediate_size")?,
        "vocab_size": num(cfg, "vocab_size").ok_or("config.json requires vocab_size")?,
        "rms_norm_eps": cfg.get("rms_norm_eps").cloned().unwrap_or(1e-6.into()),
        "rope_theta": cfg.get("rope_theta").cloned().or_else(|| cfg.get("rope_parameters").and_then(|r| r.get("rope_theta")).cloned()).unwrap_or(10000.0.into()),
        "max_position_embeddings": num(cfg, "max_position_embeddings").unwrap_or(4096),
        "hidden_act": cfg.get("activation_function").cloned().unwrap_or("silu".into()),
        "num_local_experts": experts,
        "num_experts_per_tok": top_k,
        "tie_word_embeddings": cfg.get("tie_word_embeddings").and_then(Value::as_bool).unwrap_or(true),
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
    let top_k = num(cfg, "num_experts_per_tok").unwrap_or(2);
    let experts = num(cfg, "num_local_experts").unwrap_or(0);
    let mut plan = Plan::from_config(&normalized)?;
    for (key, value) in &mut plan.meta {
        *key = key.replace("llama.", "jetmoe.");
        if key == "general.architecture" || (key == "general.name" && cfg.get("_name_or_path").is_none()) {
            *value = Meta::Str("jetmoe".into());
        }
    }
    plan.arch = "jetmoe".into();
    plan.meta.push(("jetmoe.attention.expert_count".into(), Meta::U32(experts as u32)));
    plan.meta.push(("jetmoe.attention.expert_used_count".into(), Meta::U32(top_k as u32)));
    Ok(plan)
}

/// GGUF name of a per-layer tensor of layer `i` with the checkpoint suffix `suf`.
pub(crate) fn map(i: u64, suf: &str) -> Option<Mapped> {
    let plain = |name: &str| Some(Mapped::Tensor { name: format!("blk.{i}.{name}"), add: 0.0 });
    let fused = |part: &str| Some(Mapped::Fused { layer: i, part: part.to_owned() });
    match suf {
        "input_layernorm.weight" => plain("attn_norm.weight"),
        "post_attention_layernorm.weight" => plain("ffn_norm.weight"),
        "self_attention.kv_proj.weight" => plain("attn_kv.weight"),
        "self_attention.experts.input_linear.weight" => plain("attn_moa_in.weight"),
        "self_attention.experts.output_linear.weight" => plain("attn_moa_out.weight"),
        "self_attention.experts.router.layer.weight" => plain("attn_moa_router.weight"),
        "self_attention.experts.bias" => plain("attn_moa_bias.weight"),
        "mlp.router.layer.weight" => plain("ffn_gate_inp.weight"),
        "mlp.input_linear.weight" => fused("experts_input"),
        "mlp.output_linear.weight" => fused("experts_output"),
        "mlp.bias" => plain("ffn_moe_bias.weight"),
        _ => None,
    }
}

pub(crate) fn layer_requirements() -> [&'static str; 11] {
    [
        "attn_norm.weight",
        "ffn_norm.weight",
        "attn_kv.weight",
        "attn_moa_in.weight",
        "attn_moa_out.weight",
        "attn_moa_router.weight",
        "attn_moa_bias.weight",
        "ffn_gate_inp.weight",
        "ffn_gate_exps.weight",
        "ffn_up_exps.weight",
        "ffn_down_exps.weight",
    ]
}
