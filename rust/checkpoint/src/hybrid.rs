//! Conversion helpers of the hybrid state-space / attention families: GGUF metadata that the generic plan does not
//! derive, and the folding of mu-P multipliers into weights (Falcon-H1).
use crate::gguf::Meta;
use crate::safetensors::{Payload, Tensor};
use serde_json::Value;
use std::sync::Arc;

fn num(cfg: &Value, key: &str) -> Option<u64> {
    cfg.get(key).and_then(Value::as_u64)
}

fn flt(cfg: &Value, key: &str) -> Option<f64> {
    cfg.get(key).and_then(Value::as_f64)
}

fn flag(cfg: &Value, key: &str, default: bool) -> bool {
    cfg.get(key).and_then(Value::as_bool).unwrap_or(default)
}

/// A multiplier for a whole tensor or for ranges of its rows (`[start, end)` -> factor).
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Scale {
    All(f32),
    Rows(Vec<(usize, usize, f32)>),
}

/// Weight tensors that are multiplied by constants while converting, selected by the tail of the Hugging Face name.
///
/// Falcon-H1 is trained with mu-P multipliers that sit between its matrix products. All of them are linear, so
/// they fold into the neighbouring weights (and biases) once; the decoder then needs no per-model code and the
/// multiplied weights quantise as a normal matrix. Only the embedding and head multipliers stay as metadata
/// (`embedding_scale`, `logit_scale`) because the head can share the embedding matrix.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct ScaleSet {
    rules: Vec<(String, Scale)>,
}

impl ScaleSet {
    fn push(&mut self, suffix: &str, scale: Scale) {
        let trivial = matches!(&scale, Scale::All(f) if *f == 1.0) || matches!(&scale, Scale::Rows(r) if r.iter().all(|x| x.2 == 1.0));
        if !trivial {
            self.rules.push((suffix.to_owned(), scale));
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// The factor rule for `name`, if any.
    pub(crate) fn rule(&self, name: &str) -> Option<&Scale> {
        self.rules.iter().find(|(suffix, _)| name.ends_with(suffix.as_str())).map(|(_, s)| s)
    }

    /// Multiplies `t` in place when a rule matches (the dtype is kept: F32, F16 or BF16).
    pub(crate) fn apply(&self, t: &mut Tensor) -> Result<(), String> {
        let Some(scale) = self.rule(&t.name) else { return Ok(()) };
        let width = match t.dtype.as_str() {
            "F32" => 4,
            "F16" | "BF16" => 2,
            other => return Err(format!("cannot fold a multiplier into {} of dtype {other}", t.name)),
        };
        let rows = *t.shape.first().ok_or_else(|| format!("{} has no shape", t.name))? as usize;
        let count = t.shape.iter().product::<u64>() as usize;
        let per_row = count / rows.max(1);
        let src = t.payload.as_slice();
        if src.len() != count * width {
            return Err(format!("tensor {} payload length does not match its shape", t.name));
        }
        let factor = |row: usize| -> f32 {
            match scale {
                Scale::All(f) => *f,
                Scale::Rows(ranges) => ranges.iter().find(|(a, b, _)| row >= *a && row < *b).map_or(1.0, |r| r.2),
            }
        };
        if let Scale::Rows(ranges) = scale {
            if ranges.iter().any(|r| r.1 > rows) {
                return Err(format!("row ranges of the multipliers exceed the {rows} rows of {}", t.name));
            }
        }
        let mut out = Vec::with_capacity(src.len());
        for row in 0..rows {
            let f = factor(row);
            for chunk in src[row * per_row * width..(row + 1) * per_row * width].chunks_exact(width) {
                let v = crate::gguf::to_f32(&t.dtype, chunk).map_err(|e| e.to_string())?;
                crate::gguf::append_converted(&mut out, &t.dtype, v * f).map_err(|e| e.to_string())?;
            }
        }
        let len = out.len();
        t.payload = Payload::Owned(Arc::new(out), 0..len);
        Ok(())
    }
}

/// Hidden size of the Mamba-2 part (`mamba_d_ssm`, or `mamba_expand * hidden_size`).
fn falcon_h1_inner(cfg: &Value, hidden: u64) -> u64 {
    num(cfg, "mamba_d_ssm").unwrap_or(num(cfg, "mamba_expand").unwrap_or(2) * hidden)
}

/// Multipliers of a `falcon_h1` configuration as weight rules; `None` for other model types.
pub(crate) fn weight_scales(cfg: &Value) -> Result<Option<ScaleSet>, String> {
    if cfg.get("model_type").and_then(Value::as_str) != Some("falcon_h1") {
        return Ok(None);
    }
    let hidden = num(cfg, "hidden_size").ok_or("config.json requires hidden_size")?;
    let list = |key: &str, n: usize| -> Result<Vec<f32>, String> {
        match cfg.get(key).filter(|v| !v.is_null()) {
            None => Ok(vec![1.0; n]),
            Some(Value::Array(a)) if a.len() == n => a
                .iter()
                .map(|v| v.as_f64().map(|f| f as f32).ok_or_else(|| format!("{key} must hold numbers")))
                .collect(),
            Some(_) => Err(format!("{key} must be a list of {n} numbers")),
        }
    };
    let one = |key: &str| flt(cfg, key).unwrap_or(1.0) as f32;
    let (ssm_in, ssm_out) = (one("ssm_in_multiplier"), one("ssm_out_multiplier"));
    let (attn_in, attn_out, key_mul) = (one("attention_in_multiplier"), one("attention_out_multiplier"), one("key_multiplier"));
    let mlp = list("mlp_multipliers", 2)?;
    let zxbcdt = list("ssm_multipliers", 5)?;
    let inner = falcon_h1_inner(cfg, hidden) as usize;
    let groups_state = (num(cfg, "mamba_n_groups").unwrap_or(1) * num(cfg, "mamba_d_state").unwrap_or(256)) as usize;
    let heads = num(cfg, "mamba_n_heads").unwrap_or(128) as usize;
    // mup vector of the in_proj output: [z | x | B | C | dt]
    let edges = [0, inner, 2 * inner, 2 * inner + groups_state, 2 * inner + 2 * groups_state, 2 * inner + 2 * groups_state + heads];
    let mup: Vec<(usize, usize, f32)> = (0..5).map(|i| (edges[i], edges[i + 1], zxbcdt[i])).collect();
    let mut set = ScaleSet::default();
    // the bias is added before the multiplier of the in projection, so only the weight carries `ssm_in_multiplier`
    set.push(".mamba.in_proj.weight", Scale::Rows(mup.iter().map(|&(a, b, f)| (a, b, f * ssm_in)).collect()));
    set.push(".mamba.in_proj.bias", Scale::Rows(mup));
    set.push(".mamba.out_proj.weight", Scale::All(ssm_out));
    set.push(".mamba.out_proj.bias", Scale::All(ssm_out));
    for part in ["q_proj", "v_proj"] {
        set.push(&format!(".self_attn.{part}.weight"), Scale::All(attn_in));
    }
    set.push(".self_attn.k_proj.weight", Scale::All(attn_in * key_mul));
    set.push(".self_attn.k_proj.bias", Scale::All(key_mul));
    set.push(".self_attn.o_proj.weight", Scale::All(attn_out));
    set.push(".self_attn.o_proj.bias", Scale::All(attn_out));
    set.push(".feed_forward.gate_proj.weight", Scale::All(mlp[0]));
    set.push(".feed_forward.gate_proj.bias", Scale::All(mlp[0]));
    set.push(".feed_forward.down_proj.weight", Scale::All(mlp[1]));
    set.push(".feed_forward.down_proj.bias", Scale::All(mlp[1]));
    if set.is_empty() {
        return Ok(None);
    }
    if cfg.get("quantization_config").is_some_and(|q| !q.is_null()) {
        return Err("Falcon-H1 checkpoints with multipliers cannot be converted from a quantised source".into());
    }
    Ok(Some(set))
}

/// SSM, embedding and head metadata of Falcon-H1 (`falcon-h1.*`).
pub(crate) fn falcon_h1_meta(cfg: &Value, hidden: u64, meta: &mut Vec<(String, Meta)>) -> Result<(), String> {
    let arch = "falcon-h1";
    let inner = falcon_h1_inner(cfg, hidden);
    let heads = num(cfg, "mamba_n_heads").unwrap_or(128);
    let groups = num(cfg, "mamba_n_groups").unwrap_or(1);
    let head_dim = match cfg.get("mamba_d_head") {
        Some(Value::Number(n)) => n.as_u64().unwrap_or(0),
        _ => inner / heads.max(1),
    };
    if heads == 0 || head_dim * heads != inner || !heads.is_multiple_of(groups.max(1)) {
        return Err("inconsistent Falcon-H1 Mamba-2 dimensions".into());
    }
    let u = |k: &str, v: u64| (format!("{arch}.{k}"), Meta::U32(v as u32));
    let rms_norm = flag(cfg, "mamba_rms_norm", false);
    let before_gate = flag(cfg, "mamba_norm_before_gate", true);
    // 0: norm(y * silu(z)), 1: y * silu(z), 2: norm(y) * silu(z)
    let mode = match (rms_norm, before_gate) {
        (false, _) => 1,
        (true, true) => 2,
        (true, false) => 0,
    };
    meta.extend([
        u("ssm.conv_kernel", num(cfg, "mamba_d_conv").unwrap_or(4)),
        u("ssm.state_size", num(cfg, "mamba_d_state").unwrap_or(256)),
        u("ssm.group_count", groups),
        u("ssm.time_step_rank", heads),
        u("ssm.inner_size", inner),
        u("ssm.norm_groups", groups),
        u("ssm.norm_mode", mode),
    ]);
    if let Some([lo, hi]) = cfg.get("time_step_limit").and_then(Value::as_array).map(|a| [a.first(), a.get(1)]).map(|[a, b]| [a.and_then(Value::as_f64), b.and_then(Value::as_f64)]).map(|[a, b]| [a.unwrap_or(0.0), b.unwrap_or(f64::INFINITY)]) {
        if lo > 0.0 {
            meta.push((format!("{arch}.ssm.time_step_min"), Meta::F32(lo as f32)));
        }
        if hi.is_finite() {
            meta.push((format!("{arch}.ssm.time_step_max"), Meta::F32(hi as f32)));
        }
    }
    for (key, name) in [("embedding_multiplier", "embedding_scale"), ("lm_head_multiplier", "logit_scale")] {
        if let Some(v) = flt(cfg, key).filter(|v| *v != 1.0) {
            meta.push((format!("{arch}.{name}"), Meta::F32(v as f32)));
        }
    }
    Ok(())
}

/// Jamba: Mamba-1 layers with learned dt/B/C norms, attention without rotary embeddings every `attn_layer_period` layers.
pub(crate) fn jamba_meta(cfg: &Value, hidden: u64, layers: u64, meta: &mut Vec<(String, Meta)>) -> Result<(), String> {
    let arch = "jamba";
    let u = |k: &str, v: u64| (format!("{arch}.{k}"), Meta::U32(v as u32));
    let rank = match cfg.get("mamba_dt_rank") {
        Some(Value::Number(n)) => n.as_u64().ok_or("invalid mamba_dt_rank")?,
        _ => hidden.div_ceil(16),
    };
    let period = num(cfg, "attn_layer_period").unwrap_or(8);
    let offset = num(cfg, "attn_layer_offset").unwrap_or(4);
    if period == 0 || offset >= period {
        return Err("Jamba attn_layer_offset must be smaller than attn_layer_period".into());
    }
    if num(cfg, "expert_layer_period").unwrap_or(2) == 0 {
        return Err("Jamba expert_layer_period must be positive".into());
    }
    meta.extend([
        u("ssm.conv_kernel", num(cfg, "mamba_d_conv").unwrap_or(4)),
        u("ssm.inner_size", num(cfg, "mamba_expand").unwrap_or(2) * hidden),
        u("ssm.state_size", num(cfg, "mamba_d_state").unwrap_or(16)),
        u("ssm.time_step_rank", rank),
        (format!("{arch}.ssm.dt_b_c_rms_eps"), Meta::F32(flt(cfg, "rms_norm_eps").unwrap_or(1e-6) as f32)),
        // Jamba has no positional encoding at all
        (format!("{arch}.rope.layer_flags"), Meta::Bools(vec![false; layers as usize])),
    ]);
    Ok(())
}

/// Whether Jamba layer `i` uses experts (`layers_num_experts[i] > 1`).
pub(crate) fn jamba_expert_layer(cfg: &Value, i: u64) -> bool {
    i % num(cfg, "expert_layer_period").unwrap_or(2).max(1) == num(cfg, "expert_layer_offset").unwrap_or(1)
}

/// Whether Jamba layer `i` is an attention layer (the others are Mamba).
pub(crate) fn jamba_attention_layer(cfg: &Value, i: u64) -> bool {
    i % num(cfg, "attn_layer_period").unwrap_or(8).max(1) == num(cfg, "attn_layer_offset").unwrap_or(4)
}

/// Tensor names of a Jamba layer that the generic plan does not know: the Mamba-1 mixer with its norms, and the expert
/// tensors in both checkpoint layouts (one tensor per expert in the Hub files, stacked `gate_up_proj` / `down_proj`
/// in the transformers 5 format).
pub(crate) fn map_jamba(layer: &str, suf: &str) -> Option<crate::plan::Mapped> {
    use crate::plan::Mapped;
    let plain = |name: &str| Some(Mapped::Tensor { name: format!("blk.{layer}.{name}"), add: 0.0 });
    let fused = |part: &str| Some(Mapped::Fused { layer: layer.parse().unwrap_or(0), part: part.to_owned() });
    match suf {
        "mamba.in_proj.weight" => plain("ssm_in.weight"),
        "mamba.in_proj.bias" => plain("ssm_in.bias"),
        "mamba.conv1d.weight" => plain("ssm_conv1d.weight"),
        "mamba.conv1d.bias" => plain("ssm_conv1d.bias"),
        "mamba.x_proj.weight" => plain("ssm_x.weight"),
        "mamba.dt_proj.weight" => plain("ssm_dt.weight"),
        "mamba.dt_proj.bias" => plain("ssm_dt.bias"),
        "mamba.D" => plain("ssm_d"),
        "mamba.out_proj.weight" => plain("ssm_out.weight"),
        "mamba.out_proj.bias" => plain("ssm_out.bias"),
        "mamba.dt_layernorm.weight" => plain("ssm_dt_norm.weight"),
        "mamba.b_layernorm.weight" => plain("ssm_b_norm.weight"),
        "mamba.c_layernorm.weight" => plain("ssm_c_norm.weight"),
        "mamba.A_log" => fused("A_log"),
        "feed_forward.experts.gate_up_proj" => fused("experts_input"),
        "feed_forward.experts.down_proj" => fused("experts_output"),
        _ => {
            let (index, kind) = suf.strip_prefix("feed_forward.experts.")?.split_once('.')?;
            let kind = match kind {
                "gate_proj.weight" => "gate",
                "up_proj.weight" => "up",
                "down_proj.weight" => "down",
                _ => return None,
            };
            Some(Mapped::Expert { name: format!("blk.{layer}.ffn_{kind}_exps.weight"), index: index.parse().ok()? })
        }
    }
}

/// GGUF tensors that layer `i` of a hybrid model must have after conversion; `None` for architectures with their own check.
pub(crate) fn layer_requirements(arch: &str, i: u64, cfg: &Value) -> Option<Vec<String>> {
    let mut need: Vec<&str> = vec!["attn_norm.weight", "ffn_norm.weight"];
    match arch {
        "falcon-h1" => {
            need.extend(["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"]);
            need.extend(["ssm_in.weight", "ssm_conv1d.weight", "ssm_dt.bias", "ssm_a", "ssm_d", "ssm_out.weight"]);
            if flag(cfg, "mamba_rms_norm", false) {
                need.push("ssm_norm.weight");
            }
            need.extend(["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]);
        }
        "jamba" => {
            if jamba_attention_layer(cfg, i) {
                need.extend(["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"]);
            } else {
                need.extend([
                    "ssm_in.weight",
                    "ssm_conv1d.weight",
                    "ssm_x.weight",
                    "ssm_dt.weight",
                    "ssm_dt.bias",
                    "ssm_a",
                    "ssm_d",
                    "ssm_out.weight",
                    "ssm_dt_norm.weight",
                    "ssm_b_norm.weight",
                    "ssm_c_norm.weight",
                ]);
            }
            if jamba_expert_layer(cfg, i) {
                need.extend(["ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"]);
            } else {
                need.extend(["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]);
            }
        }
        "zamba2" => return crate::zamba2::requirements(cfg, i),
        "kimi-linear" => return crate::kimi_linear::requirements(cfg, i),
        "deepseek4" => return crate::deepseek4::requirements(cfg, i),
        "longcat-flash" => return crate::longcat::requirements(cfg, i),
        _ => return None,
    }
    Some(need.into_iter().map(|n| format!("blk.{i}.{n}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tensor(name: &str, shape: Vec<u64>, values: &[f32]) -> Tensor {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let len = bytes.len();
        Tensor { name: name.into(), dtype: "F32".into(), shape, payload: Payload::Owned(Arc::new(bytes), 0..len) }
    }

    fn falcon() -> Value {
        json!({
            "model_type": "falcon_h1", "hidden_size": 8, "mamba_d_ssm": 4, "mamba_n_heads": 2, "mamba_n_groups": 1, "mamba_d_state": 2,
            "ssm_in_multiplier": 2.0, "ssm_multipliers": [1.0, 3.0, 5.0, 7.0, 11.0], "ssm_out_multiplier": 0.5,
            "attention_in_multiplier": 4.0, "key_multiplier": 0.25, "mlp_multipliers": [2.0, 0.5]
        })
    }

    fn values(t: &Tensor) -> Vec<f32> {
        t.payload.as_slice().as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
    }

    #[test]
    fn falcon_h1_in_projection_gets_the_mup_vector_by_row() {
        let set = weight_scales(&falcon()).unwrap().unwrap();
        // rows: z(4) x(4) B(2) C(2) dt(2) = 14
        let mut w = tensor("model.layers.0.mamba.in_proj.weight", vec![14, 1], &[1.0; 14]);
        set.apply(&mut w).unwrap();
        let got = values(&w);
        assert_eq!(&got[..4], &[2.0; 4]);
        assert_eq!(&got[4..8], &[6.0; 4]);
        assert_eq!(&got[8..10], &[10.0; 2]);
        assert_eq!(&got[10..12], &[14.0; 2]);
        assert_eq!(&got[12..], &[22.0; 2]);
        // the bias only meets the mup vector
        let mut b = tensor("model.layers.0.mamba.in_proj.bias", vec![14], &[1.0; 14]);
        set.apply(&mut b).unwrap();
        assert_eq!(&values(&b)[4..8], &[3.0; 4]);
    }

    #[test]
    fn falcon_h1_attention_and_mlp_multipliers_fold_into_weights() {
        let set = weight_scales(&falcon()).unwrap().unwrap();
        let mut k = tensor("model.layers.1.self_attn.k_proj.weight", vec![1, 2], &[1.0, 2.0]);
        set.apply(&mut k).unwrap();
        assert_eq!(values(&k), vec![1.0, 2.0]);
        let mut q = tensor("model.layers.1.self_attn.q_proj.weight", vec![1, 2], &[1.0, 2.0]);
        set.apply(&mut q).unwrap();
        assert_eq!(values(&q), vec![4.0, 8.0]);
        let mut gate = tensor("model.layers.1.feed_forward.gate_proj.weight", vec![1, 2], &[1.0, 2.0]);
        set.apply(&mut gate).unwrap();
        assert_eq!(values(&gate), vec![2.0, 4.0]);
        let mut norm = tensor("model.layers.1.input_layernorm.weight", vec![2], &[1.0, 2.0]);
        set.apply(&mut norm).unwrap();
        assert_eq!(values(&norm), vec![1.0, 2.0]);
    }

    #[test]
    fn jamba_names_cover_both_expert_layouts() {
        use crate::plan::Mapped;
        let name = |m: Option<Mapped>| match m {
            Some(Mapped::Tensor { name, .. }) => name,
            Some(Mapped::Expert { name, index }) => format!("{name}#{index}"),
            Some(Mapped::Fused { layer, part }) => format!("fused:{layer}:{part}"),
            other => panic!("unexpected mapping {other:?}"),
        };
        assert_eq!(name(map_jamba("3", "mamba.dt_layernorm.weight")), "blk.3.ssm_dt_norm.weight");
        assert_eq!(name(map_jamba("3", "mamba.A_log")), "fused:3:A_log");
        assert_eq!(name(map_jamba("1", "feed_forward.experts.2.up_proj.weight")), "blk.1.ffn_up_exps.weight#2");
        assert_eq!(name(map_jamba("1", "feed_forward.experts.gate_up_proj")), "fused:1:experts_input");
        assert_eq!(name(map_jamba("1", "feed_forward.experts.down_proj")), "fused:1:experts_output");
        assert!(map_jamba("1", "feed_forward.gate_proj.weight").is_none());
        assert!(map_jamba("1", "feed_forward.experts.x.gate_proj.weight").is_none());
    }

    #[test]
    fn jamba_layer_kinds_follow_the_period_and_offset() {
        let cfg = json!({"attn_layer_period": 4, "attn_layer_offset": 2, "expert_layer_period": 2, "expert_layer_offset": 1});
        let attn: Vec<bool> = (0..8).map(|i| jamba_attention_layer(&cfg, i)).collect();
        assert_eq!(attn, [false, false, true, false, false, false, true, false]);
        let moe: Vec<bool> = (0..4).map(|i| jamba_expert_layer(&cfg, i)).collect();
        assert_eq!(moe, [false, true, false, true]);
        let need = layer_requirements("jamba", 1, &cfg).unwrap();
        assert!(need.contains(&"blk.1.ssm_b_norm.weight".to_owned()) && need.contains(&"blk.1.ffn_down_exps.weight".to_owned()));
        let need = layer_requirements("jamba", 2, &cfg).unwrap();
        assert!(need.contains(&"blk.2.attn_q.weight".to_owned()) && need.contains(&"blk.2.ffn_gate.weight".to_owned()));
        assert!(layer_requirements("llama", 0, &cfg).is_none());
    }

    #[test]
    fn jamba_metadata_has_no_rope_and_the_auto_time_step_rank() {
        let mut meta = Vec::new();
        let cfg = json!({"attn_layer_period": 4, "attn_layer_offset": 2, "rms_norm_eps": 1e-6, "mamba_dt_rank": "auto"});
        jamba_meta(&cfg, 64, 4, &mut meta).unwrap();
        let get = |k: &str| meta.iter().find(|(n, _)| n == &format!("jamba.{k}")).map(|(_, v)| v.clone());
        assert!(matches!(get("ssm.time_step_rank"), Some(Meta::U32(4))));
        assert!(matches!(get("ssm.inner_size"), Some(Meta::U32(128))));
        assert!(matches!(get("rope.layer_flags"), Some(Meta::Bools(f)) if f == vec![false; 4]));
        let bad = json!({"attn_layer_period": 2, "attn_layer_offset": 2});
        assert!(jamba_meta(&bad, 64, 4, &mut Vec::new()).is_err());
    }

    #[test]
    fn falcon_h1_metadata_selects_the_norm_order() {
        let mut meta = Vec::new();
        let mut cfg = falcon();
        cfg["mamba_rms_norm"] = true.into();
        cfg["mamba_norm_before_gate"] = false.into();
        cfg["lm_head_multiplier"] = 0.5.into();
        falcon_h1_meta(&cfg, 8, &mut meta).unwrap();
        let get = |k: &str| meta.iter().find(|(n, _)| n == &format!("falcon-h1.{k}")).map(|(_, v)| v.clone());
        assert!(matches!(get("ssm.norm_mode"), Some(Meta::U32(0))));
        assert!(matches!(get("logit_scale"), Some(Meta::F32(v)) if v == 0.5));
        assert!(get("embedding_scale").is_none());
        let mut bad = falcon();
        bad["mamba_n_heads"] = 3.into();
        assert!(falcon_h1_meta(&bad, 8, &mut Vec::new()).is_err());
    }
}
