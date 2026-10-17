//! Zamba2 (`Zamba2ForCausalLM`): a Mamba-2 backbone and shared transformer blocks that run in front of some of its layers.
//!
//! GGUF layout (architecture `zamba2`): the Mamba-2 mixer and its input norm of layer `i` are the usual `blk.i.ssm_*` and
//! `blk.i.attn_norm.weight`. A layer that starts with a call of a shared block also has `blk.i.zamba_linear.weight` and the
//! adapters of that call (`adapter_{q,k,v}_{a,b}`, `adapter_ffn_{a,b}`). The weights of the shared blocks exist once per block as
//! `shared.<b>.{attn_norm,attn_q,attn_k,attn_v,attn_output,ffn_norm,ffn_gate_up,ffn_down}.weight`; call `n` uses block
//! `n % shared_blocks`. Hugging Face checkpoints keep a single copy of the tied weights (in the first layer that uses a block);
//! copies of later layers are dropped.
use crate::gguf::Meta;
use crate::plan::{Mapped, Plan};
use serde_json::Value;

fn num(cfg: &Value, key: &str) -> Option<u64> {
    cfg.get(key).and_then(Value::as_u64)
}

/// Kinds per layer: `H` starts with a shared-block call, `M` is a plain Mamba-2 layer.
fn kinds(cfg: &Value) -> Result<Vec<u8>, String> {
    let layers = num(cfg, "num_hidden_layers").ok_or("config.json requires num_hidden_layers")? as usize;
    let list = cfg.get("layers_block_type").and_then(Value::as_array);
    let kinds: Vec<u8> = match list {
        Some(list) => list
            .iter()
            .map(|v| match v.as_str() {
                Some("hybrid") => Ok(b'H'),
                Some("mamba" | "linear_attention") => Ok(b'M'),
                other => Err(format!("unsupported Zamba2 layer type {other:?}")),
            })
            .collect::<Result<_, _>>()?,
        // the default pattern of the configuration class
        None => {
            let mut k = vec![b'M'];
            for _ in 0..7 {
                k.extend(b"MMMMMH");
            }
            k.extend(b"MMMMHMMMHMM");
            k
        }
    };
    if kinds.len() != layers {
        return Err(format!("layers_block_type has {} entries for {layers} layers", kinds.len()));
    }
    Ok(kinds)
}

/// The configuration with the keys the generic shape checks and the plan expect.
pub(crate) fn config(cfg: &Value) -> Result<Value, String> {
    let get = |k: &str| num(cfg, k).ok_or_else(|| format!("config.json requires {k}"));
    let (hidden, heads) = (get("hidden_size")?, get("num_attention_heads")?);
    if heads == 0 || !(2 * hidden).is_multiple_of(heads) {
        return Err("Zamba2: 2 * hidden_size must be a multiple of num_attention_heads".into());
    }
    if cfg.get("add_bias_linear").and_then(Value::as_bool) == Some(true) {
        return Err("Zamba2 with add_bias_linear is unsupported".into());
    }
    if cfg.get("hidden_act").and_then(Value::as_str).is_some_and(|a| a != "gelu") {
        return Err("Zamba2 supports hidden_act gelu only".into());
    }
    let mut out = cfg.clone();
    out["model_type"] = "llama".into();
    out["hidden_act"] = "silu".into();
    out["head_dim"] = (2 * hidden / heads).into();
    out["intermediate_size"] = num(cfg, "intermediate_size").unwrap_or(4 * hidden).into();
    out["num_key_value_heads"] = num(cfg, "num_key_value_heads").unwrap_or(heads).into();
    out["num_hidden_layers"] = (kinds(cfg)?.len() as u64).into();
    if cfg.get("tie_word_embeddings").is_none() {
        out["tie_word_embeddings"] = true.into();
    }
    Ok(out)
}

pub(crate) fn plan(cfg: &Value) -> Result<Plan, String> {
    let normalized = config(cfg)?;
    let kinds = kinds(cfg)?;
    let layers = kinds.len() as u64;
    let mut plan = Plan::from_config(&normalized)?;
    let (family_arch, arch) = ("llama", "zamba2");
    plan.arch = arch.into();
    for (key, value) in &mut plan.meta {
        *key = key.replace(&format!("{family_arch}."), &format!("{arch}."));
        if key == "general.architecture" || (key == "general.name" && cfg.get("_name_or_path").is_none()) {
            *value = Meta::Str(arch.into());
        }
    }
    let hidden = num(cfg, "hidden_size").unwrap_or(0);
    let heads = num(cfg, "num_attention_heads").unwrap_or(1);
    let head_dim = 2 * hidden / heads;
    let groups = num(cfg, "mamba_ngroups").unwrap_or(1);
    let inner = num(cfg, "mamba_expand").unwrap_or(2) * hidden;
    let ssm_heads = num(cfg, "n_mamba_heads").unwrap_or(8);
    if ssm_heads == 0 || !inner.is_multiple_of(ssm_heads) || !ssm_heads.is_multiple_of(groups.max(1)) {
        return Err("inconsistent Zamba2 Mamba-2 dimensions".into());
    }
    let u = |k: &str, v: u64| (format!("{arch}.{k}"), Meta::U32(v as u32));
    let f = |k: &str, v: f64| (format!("{arch}.{k}"), Meta::F32(v as f32));
    let flags: Vec<bool> = kinds.iter().map(|&k| k == b'H').collect();
    let mem_rope = cfg.get("use_mem_rope").and_then(Value::as_bool).unwrap_or(false);
    plan.meta.retain(|(k, _)| !k.ends_with(".rope.layer_flags"));
    plan.meta.extend([
        u("ssm.conv_kernel", num(cfg, "mamba_d_conv").unwrap_or(4)),
        u("ssm.inner_size", inner),
        u("ssm.state_size", num(cfg, "mamba_d_state").unwrap_or(64)),
        u("ssm.time_step_rank", ssm_heads),
        u("ssm.group_count", groups),
        u("ssm.norm_groups", groups),
        f("ssm.time_step_min", cfg.get("time_step_min").and_then(Value::as_f64).unwrap_or(0.001)),
        f("ssm.norm_eps", 1e-5),
        f("attention.scale", (head_dim as f64 / 2.0).powf(-0.5)),
        u("shared_blocks", num(cfg, "num_mem_blocks").unwrap_or(1)),
        u("adapter_rank", num(cfg, "adapter_rank").unwrap_or(128)),
        (
            format!("{arch}.shared_attention_adapter"),
            Meta::Bool(cfg.get("use_shared_attention_adapter").and_then(Value::as_bool).unwrap_or(false)),
        ),
        (format!("{arch}.shared_layer_flags"), Meta::Bools(flags.clone())),
        (
            format!("{arch}.rope.layer_flags"),
            Meta::Bools(flags.iter().map(|&h| h && mem_rope).collect()),
        ),
    ]);
    plan.kinds = kinds;
    plan.layers = layers;
    Ok(plan)
}

/// Layer index of the `j`-th shared-block call and the shared block of layer `i`.
fn hybrid_layers(kinds: &[u8]) -> Vec<usize> {
    kinds.iter().enumerate().filter(|(_, &k)| k == b'H').map(|(i, _)| i).collect()
}

pub(crate) fn map(plan: &Plan, n: &str) -> Result<Mapped, String> {
    let blocks = plan
        .meta
        .iter()
        .find_map(|(k, v)| match (k.as_str(), v) {
            ("zamba2.shared_blocks", Meta::U32(b)) => Some(*b as usize),
            _ => None,
        })
        .unwrap_or(1);
    let tensor = |name: String| Ok(Mapped::Tensor { name, add: 0.0 });
    match n {
        "model.embed_tokens.weight" => return tensor("token_embd.weight".into()),
        "model.final_layernorm.weight" => return tensor("output_norm.weight".into()),
        "lm_head.weight" => return tensor("output.weight".into()),
        _ => {}
    }
    let rest = n.strip_prefix("model.layers.").ok_or_else(|| format!("unrecognized checkpoint tensor {n:?}"))?;
    let (i, suf) = rest.split_once('.').ok_or_else(|| format!("malformed layer tensor name {n:?}"))?;
    let layer: usize = i.parse().map_err(|_| format!("invalid layer index in tensor {n:?}"))?;
    let kind = *plan.kinds.get(layer).ok_or_else(|| format!("tensor {n:?} refers to an invalid layer"))?;
    let hybrid = kind == b'H';
    let ids = hybrid_layers(&plan.kinds);
    let mamba = |suf: &str| -> Option<Mapped> {
        let name = match suf {
            "in_proj.weight" => "ssm_in.weight",
            "conv1d.weight" => "ssm_conv1d.weight",
            "conv1d.bias" => "ssm_conv1d.bias",
            "dt_bias" => "ssm_dt.bias",
            "D" => "ssm_d",
            "norm.weight" => "ssm_norm.weight",
            "out_proj.weight" => "ssm_out.weight",
            "A_log" => return Some(Mapped::Fused { layer: layer as u64, part: "A_log".into() }),
            _ => return None,
        };
        Some(Mapped::Tensor { name: format!("blk.{layer}.{name}"), add: 0.0 })
    };
    let (mamba_part, norm_name) = if hybrid { ("mamba_decoder.mamba.", "mamba_decoder.input_layernorm.weight") } else { ("mamba.", "input_layernorm.weight") };
    if suf == norm_name {
        return tensor(format!("blk.{layer}.attn_norm.weight"));
    }
    if let Some(m) = suf.strip_prefix(mamba_part).and_then(mamba) {
        return Ok(m);
    }
    if hybrid && (suf == "linear.weight" || suf == "linear.bias") {
        let ext = if suf.ends_with(".bias") { "bias" } else { "weight" };
        return tensor(format!("blk.{layer}.zamba_linear.{ext}"));
    }
    if let (true, Some(shared)) = (hybrid, suf.strip_prefix("shared_transformer.")) {
        let ordinal = ids.iter().position(|&l| l == layer).unwrap_or(0);
        // adapters belong to one call each, wherever the checkpoint stores them
        for (prefix, stem) in [
            ("self_attn.linear_q_adapter_list.", "adapter_q"),
            ("self_attn.linear_k_adapter_list.", "adapter_k"),
            ("self_attn.linear_v_adapter_list.", "adapter_v"),
            ("feed_forward.gate_up_proj_adapter_list.", "adapter_ffn"),
        ] {
            if let Some(tail) = shared.strip_prefix(prefix) {
                let parts: Vec<&str> = tail.split('.').collect();
                let call: usize = parts.first().and_then(|c| c.parse().ok()).ok_or_else(|| format!("malformed adapter tensor {n:?}"))?;
                let target = *ids.get(call).ok_or_else(|| format!("adapter {call} of {n:?} has no shared-block call"))?;
                let side = match parts.get(1..) {
                    Some(["0", "weight"]) => "a",
                    Some(["1", "weight"]) => "b",
                    _ => return Err(format!("unsupported adapter tensor {n:?}")),
                };
                return tensor(format!("blk.{target}.{stem}_{side}.weight"));
            }
        }
        let (stem, is_bias) = if let Some(base) = shared.strip_suffix(".bias") {
            (base, true)
        } else if let Some(base) = shared.strip_suffix(".weight") {
            (base, false)
        } else {
            (shared, false)
        };
        let ext = if is_bias { "bias" } else { "weight" };
        let name_owned: String;
        let name = match stem {
            "input_layernorm" => "attn_norm.weight",
            "pre_ff_layernorm" => "ffn_norm.weight",
            "self_attn.q_proj" => { name_owned = format!("attn_q.{ext}"); &name_owned },
            "self_attn.k_proj" => { name_owned = format!("attn_k.{ext}"); &name_owned },
            "self_attn.v_proj" => { name_owned = format!("attn_v.{ext}"); &name_owned },
            "self_attn.o_proj" => { name_owned = format!("attn_output.{ext}"); &name_owned },
            "feed_forward.gate_up_proj" => { name_owned = format!("ffn_gate_up.{ext}"); &name_owned },
            "feed_forward.down_proj" => { name_owned = format!("ffn_down.{ext}"); &name_owned },
            _ => return Err(format!("unsupported forward-affecting layer tensor {n:?}")),
        };
        // later calls of a block repeat the (tied) weights of its first call
        if ordinal >= blocks {
            return Ok(Mapped::Skip);
        }
        return tensor(format!("shared.{}.{name}", ordinal % blocks));
    }
    Err(format!("unsupported forward-affecting layer tensor {n:?}"))
}

/// GGUF tensors layer `i` must have after conversion.
pub(crate) fn requirements(cfg: &Value, i: u64) -> Option<Vec<String>> {
    let kinds = kinds(cfg).ok()?;
    let blocks = num(cfg, "num_mem_blocks").unwrap_or(1) as usize;
    let mut need: Vec<String> = ["attn_norm.weight", "ssm_in.weight", "ssm_conv1d.weight", "ssm_dt.bias", "ssm_a", "ssm_d", "ssm_norm.weight", "ssm_out.weight"]
        .iter()
        .map(|n| format!("blk.{i}.{n}"))
        .collect();
    if kinds[i as usize] == b'H' {
        need.push(format!("blk.{i}.zamba_linear.weight"));
        need.extend(["adapter_ffn_a.weight", "adapter_ffn_b.weight"].iter().map(|n| format!("blk.{i}.{n}")));
        if cfg.get("use_shared_attention_adapter").and_then(Value::as_bool) == Some(true) {
            for s in ["q", "k", "v"] {
                need.extend([format!("blk.{i}.adapter_{s}_a.weight"), format!("blk.{i}.adapter_{s}_b.weight")]);
            }
        }
        let ordinal = hybrid_layers(&kinds).iter().position(|&l| l as u64 == i).unwrap_or(0);
        if ordinal < blocks {
            for n in ["attn_norm", "attn_q", "attn_k", "attn_v", "attn_output", "ffn_norm", "ffn_gate_up", "ffn_down"] {
                need.push(format!("shared.{ordinal}.{n}.weight"));
            }
        }
    }
    Some(need)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn cfg() -> Value {
        json!({
            "model_type": "zamba2", "hidden_size": 16, "num_hidden_layers": 6, "num_attention_heads": 4,
            "layers_block_type": ["mamba", "hybrid", "mamba", "hybrid", "mamba", "hybrid"], "num_mem_blocks": 2,
            "mamba_d_state": 8, "mamba_expand": 2, "n_mamba_heads": 4, "mamba_ngroups": 2, "adapter_rank": 3,
            "use_shared_attention_adapter": true, "max_position_embeddings": 64, "rope_theta": 10000.0
        })
    }

    fn name(m: Mapped) -> String {
        match m {
            Mapped::Tensor { name, .. } => name,
            Mapped::Skip => "skip".into(),
            Mapped::Fused { layer, part } => format!("fused:{layer}:{part}"),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn plan_describes_the_shared_blocks() {
        let plan = plan(&cfg()).unwrap();
        assert_eq!(plan.arch, "zamba2");
        assert_eq!(plan.kinds, b"MHMHMH");
        let get = |k: &str| plan.meta.iter().find(|(n, _)| n == &format!("zamba2.{k}")).map(|(_, v)| v.clone());
        assert!(matches!(get("shared_blocks"), Some(Meta::U32(2))));
        assert!(matches!(get("shared_layer_flags"), Some(Meta::Bools(f)) if f == [false, true, false, true, false, true]));
        // head_dim = 2 * 16 / 4 = 8, scale 1 / sqrt(4)
        assert!(matches!(get("attention.scale"), Some(Meta::F32(s)) if (s - 0.5).abs() < 1e-6));
        assert!(matches!(get("attention.key_length"), Some(Meta::U32(8))));
        assert!(matches!(get("rope.layer_flags"), Some(Meta::Bools(f)) if f.iter().all(|&b| !b)));
        assert!(plan.meta.iter().all(|(k, _)| !k.starts_with("llama.")));
    }

    #[test]
    fn tied_blocks_keep_their_first_copy_and_adapters_follow_their_call() {
        let plan = plan(&cfg()).unwrap();
        let m = |n: &str| name(map(&plan, n).unwrap());
        assert_eq!(m("model.layers.0.mamba.in_proj.weight"), "blk.0.ssm_in.weight");
        assert_eq!(m("model.layers.0.input_layernorm.weight"), "blk.0.attn_norm.weight");
        assert_eq!(m("model.layers.1.mamba_decoder.mamba.A_log"), "fused:1:A_log");
        assert_eq!(m("model.layers.1.linear.weight"), "blk.1.zamba_linear.weight");
        // ordinal 0 -> block 0, ordinal 1 (layer 3) -> block 1, ordinal 2 (layer 5) repeats block 0
        assert_eq!(m("model.layers.1.shared_transformer.self_attn.q_proj.weight"), "shared.0.attn_q.weight");
        assert_eq!(m("model.layers.3.shared_transformer.feed_forward.gate_up_proj.weight"), "shared.1.ffn_gate_up.weight");
        assert_eq!(m("model.layers.5.shared_transformer.self_attn.q_proj.weight"), "skip");
        // the adapter of the third call sits in the module of block 0 (layer 1) and belongs to layer 5
        assert_eq!(
            m("model.layers.1.shared_transformer.self_attn.linear_k_adapter_list.2.1.weight"),
            "blk.5.adapter_k_b.weight"
        );
        assert_eq!(
            m("model.layers.3.shared_transformer.feed_forward.gate_up_proj_adapter_list.1.0.weight"),
            "blk.3.adapter_ffn_a.weight"
        );
        assert!(map(&plan, "model.layers.1.shared_transformer.self_attn.unknown.weight").is_err());
    }

    #[test]
    fn requirements_cover_shared_blocks_once() {
        let c = cfg();
        let first = requirements(&c, 1).unwrap();
        assert!(first.contains(&"shared.0.ffn_gate_up.weight".to_owned()) && first.contains(&"blk.1.adapter_v_b.weight".to_owned()));
        let third = requirements(&c, 5).unwrap();
        assert!(!third.iter().any(|n| n.starts_with("shared.")) && third.contains(&"blk.5.zamba_linear.weight".to_owned()));
        let plain = requirements(&c, 0).unwrap();
        assert!(!plain.iter().any(|n| n.contains("adapter")));
    }

    #[test]
    fn unsupported_variants_are_refused() {
        let mut c = cfg();
        c["add_bias_linear"] = true.into();
        assert!(config(&c).is_err());
        let mut c = cfg();
        c["hidden_act"] = "silu".into();
        assert!(config(&c).is_err());
    }
}
