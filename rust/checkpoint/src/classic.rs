//! Older decoder families whose checkpoints use their own tensor names and layouts: GPT-2,
//! GPT-BigCode, GPT-Neo, GPT-J, OPT, Falcon, BLOOM and MPT.
use crate::gguf::Meta;
use crate::plan::{Family, LinearDims, Mapped, Plan};
use serde_json::{json, Value};

/// How the rows of a fused query/key/value projection are ordered in the checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QkvLayout {
    /// `[Q; K; V]` already.
    Contiguous,
    /// `[head][q, k, v][head_dim]` (GPT-NeoX, BLOOM, old Falcon).
    PerHead,
    /// `[kv_group][q_per_kv + 2][head_dim]` (new-architecture Falcon).
    KvGroups,
}

pub(crate) struct Classic {
    pub qkv: QkvLayout,
    /// Standard-key copy of the dimensions, so the shape checks of the generic importer work.
    pub normalized: Value,
    /// Conv1D-style weights (`[in, out]`) that must be transposed.
    pub transposed: bool,
}

fn num(v: &Value, k: &str) -> Option<u64> {
    v.get(k).and_then(Value::as_u64)
}
fn flt(v: &Value, k: &str) -> Option<f64> {
    v.get(k).and_then(Value::as_f64)
}
fn first(v: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|k| num(v, k))
}

/// ALiBi slopes for `heads` heads (the BLOOM / MPT recipe with a maximum bias of 8).
fn alibi_slopes(heads: usize) -> Vec<f32> {
    let closest = 1usize << (usize::BITS - 1 - heads.leading_zeros());
    let base = 2f64.powf(-(8.0 / closest as f64));
    let mut slopes: Vec<f32> = (1..=closest).map(|i| base.powi(i as i32) as f32).collect();
    if closest != heads {
        let extra = 2f64.powf(-(8.0 / (2 * closest) as f64));
        let remaining = (heads - closest).min(closest);
        slopes.extend((0..remaining).map(|i| extra.powi(2 * i as i32 + 1) as f32));
    }
    slopes
}

/// Metadata for the families handled here; `None` when `raw` is not one of them.
pub(crate) fn plan_for(raw: &str, cfg: &Value) -> Result<Option<Plan>, String> {
    let arch = match raw {
        "gpt2" => "gpt2",
        "gpt_bigcode" => "starcoder",
        "gpt_neo" => "gpt_neo",
        "gptj" => "gptj",
        "opt" => "opt",
        "falcon" => "falcon",
        "bloom" => "bloom",
        "mpt" => "mpt",
        _ => return Ok(None),
    };
    let missing = |k: &str| format!("config.json requires {k}");
    let attn = cfg.get("attn_config").cloned().unwrap_or(Value::Null);
    let hidden =
        first(cfg, &["n_embd", "hidden_size", "d_model"]).ok_or_else(|| missing("hidden_size"))?;
    let layers = first(
        cfg,
        &["n_layer", "num_hidden_layers", "num_layers", "n_layers"],
    )
    .ok_or_else(|| missing("num_hidden_layers"))?;
    let heads = first(
        cfg,
        &["n_head", "num_attention_heads", "num_heads", "n_heads"],
    )
    .ok_or_else(|| missing("num_attention_heads"))?;
    let head_dim = hidden / heads;
    let vocab = num(cfg, "vocab_size").ok_or_else(|| missing("vocab_size"))?;
    let ctx = first(
        cfg,
        &[
            "n_positions",
            "max_position_embeddings",
            "max_seq_len",
            "seq_length",
        ],
    )
    .unwrap_or(2048);
    let eps = flt(cfg, "layer_norm_epsilon")
        .or_else(|| flt(cfg, "layer_norm_eps"))
        .unwrap_or(1e-5);
    let mut kv_heads = heads;
    let mut ff = first(
        cfg,
        &["n_inner", "ffn_dim", "ffn_hidden_size", "intermediate_size"],
    )
    .unwrap_or(4 * hidden);
    let mut qkv = QkvLayout::Contiguous;
    let mut transposed = false;
    let mut tie_default = true;
    let mut parallel = false;
    let mut alibi = false;
    let mut rope: Option<(u64, f64)> = None;
    let mut scale: Option<f64> = None;
    let mut clip: Option<f64> = None;
    let mut pos_offset = 0u64;
    let mut window: Option<(u64, Vec<bool>)> = None;
    let activation = cfg
        .get("activation_function")
        .or_else(|| cfg.get("hidden_act"))
        .or_else(|| cfg.get("activation"))
        .and_then(Value::as_str)
        .unwrap_or("gelu");
    match raw {
        "gpt2" => {
            transposed = true;
            if cfg.get("scale_attn_weights").and_then(Value::as_bool) == Some(false) {
                scale = Some(1.0);
            }
        }
        "gpt_bigcode" => {
            if cfg.get("multi_query").and_then(Value::as_bool) != Some(false) {
                kv_heads = 1;
            } else {
                qkv = QkvLayout::PerHead;
            }
        }
        "gpt_neo" => {
            scale = Some(1.0);
            // attention_types: [[["global", "local"], 12]] expanded per layer
            let mut flags = Vec::new();
            if let Some(list) = cfg.get("attention_types").and_then(Value::as_array) {
                for entry in list {
                    let kinds = entry.get(0).and_then(Value::as_array);
                    let repeat = entry.get(1).and_then(Value::as_u64).unwrap_or(0);
                    for _ in 0..repeat {
                        for kind in kinds.into_iter().flatten() {
                            flags.push(kind.as_str() == Some("local"));
                        }
                    }
                }
            }
            if flags.len() != layers as usize {
                flags = vec![false; layers as usize];
            }
            if flags.iter().any(|&f| f) {
                window = Some((num(cfg, "window_size").unwrap_or(256), flags));
            }
            ff = num(cfg, "intermediate_size").unwrap_or(4 * hidden);
        }
        "gptj" => {
            parallel = true;
            rope = Some((num(cfg, "rotary_dim").unwrap_or(head_dim), 10_000.0));
        }
        "opt" => {
            if num(cfg, "word_embed_proj_dim").is_some_and(|d| d != hidden) {
                return Err("OPT models with a projected embedding width are unsupported".into());
            }
            if cfg.get("do_layer_norm_before").and_then(Value::as_bool) == Some(false) {
                return Err("OPT models with post-layer-norm blocks are unsupported".into());
            }
            pos_offset = 2;
            ff = num(cfg, "ffn_dim").unwrap_or(4 * hidden);
        }
        "falcon" => {
            let falcon_new =
                cfg.get("new_decoder_architecture").and_then(Value::as_bool) == Some(true);
            let multi_query = cfg.get("multi_query").and_then(Value::as_bool) != Some(false);
            parallel =
                falcon_new || cfg.get("parallel_attn").and_then(Value::as_bool) != Some(false);
            alibi = cfg.get("alibi").and_then(Value::as_bool) == Some(true);
            if falcon_new {
                kv_heads = first(cfg, &["num_kv_heads", "n_head_kv"]).unwrap_or(heads);
                qkv = QkvLayout::KvGroups;
            } else if multi_query {
                kv_heads = 1;
            } else {
                qkv = QkvLayout::PerHead;
            }
            if !alibi {
                rope = Some((head_dim, flt(cfg, "rope_theta").unwrap_or(10_000.0)));
            }
            ff = num(cfg, "ffn_hidden_size").unwrap_or(4 * hidden);
        }
        "bloom" => {
            alibi = true;
            qkv = QkvLayout::PerHead;
        }
        "mpt" => {
            alibi = attn.get("alibi").and_then(Value::as_bool) == Some(true);
            if attn.get("qk_ln").and_then(Value::as_bool) == Some(true) {
                return Err("MPT models with qk_ln are unsupported".into());
            }
            kv_heads = num(&attn, "kv_n_heads").unwrap_or(heads);
            clip = flt(&attn, "clip_qkv");
            // transformers fixes the MLP width at four times the model width
            ff = 4 * hidden;
            if !alibi {
                rope = Some((head_dim, 10_000.0));
            }
        }
        _ => unreachable!(),
    }
    if raw == "opt" {
        tie_default = true;
    }
    let tie = cfg
        .get("tie_word_embeddings")
        .and_then(Value::as_bool)
        .unwrap_or(tie_default);
    let _ = tie;
    match (raw, activation) {
        ("opt", "relu") => {}
        ("opt", other) => return Err(format!("unsupported OPT activation {other:?}")),
        (_, "gelu_new" | "gelu_pytorch_tanh" | "gelu_fast" | "gelu" | "gelu_tanh") => {}
        (_, other) => return Err(format!("unsupported activation {other:?}")),
    }

    let mut meta: Vec<(String, Meta)> = vec![
        ("general.architecture".into(), Meta::Str(arch.into())),
        ("general.name".into(), Meta::Str(arch.into())),
        ("general.alignment".into(), Meta::U32(32)),
    ];
    let mut u = |k: &str, v: u64| meta.push((format!("{arch}.{k}"), Meta::U32(v as u32)));
    u("context_length", ctx);
    u("embedding_length", hidden);
    u("block_count", layers);
    u("feed_forward_length", ff);
    u("attention.head_count", heads);
    u("attention.head_count_kv", kv_heads);
    u("attention.key_length", head_dim);
    u("attention.value_length", head_dim);
    if pos_offset > 0 {
        u("position_embedding_offset", pos_offset);
    }
    meta.push((
        format!("{arch}.attention.layer_norm_epsilon"),
        Meta::F32(eps as f32),
    ));
    if let Some((dims, theta)) = rope {
        meta.push((format!("{arch}.rope.freq_base"), Meta::F32(theta as f32)));
        meta.push((
            format!("{arch}.rope.dimension_count"),
            Meta::U32(dims as u32),
        ));
    } else {
        meta.push((
            format!("{arch}.rope.layer_flags"),
            Meta::Bools(vec![false; layers as usize]),
        ));
    }
    if let Some(s) = scale {
        meta.push((format!("{arch}.attention.scale"), Meta::F32(s as f32)));
    }
    if let Some(c) = clip {
        meta.push((format!("{arch}.attention.clip_qkv"), Meta::F32(c as f32)));
    }
    if alibi {
        meta.push((
            format!("{arch}.attention.alibi_slopes"),
            Meta::F32s({
                let mut slopes = alibi_slopes(heads as usize);
                if raw == "falcon" {
                    // Falcon adds the bias before the 1/sqrt(d) scaling of the scores
                    let inv = 1.0 / (head_dim as f32).sqrt();
                    slopes.iter_mut().for_each(|s| *s *= inv);
                }
                slopes
            }),
        ));
    }
    if let Some((size, flags)) = window {
        meta.push((
            format!("{arch}.attention.sliding_window"),
            Meta::U32(size as u32),
        ));
        meta.push((
            format!("{arch}.attention.sliding_window_pattern"),
            Meta::Bools(flags),
        ));
    }
    if parallel {
        meta.push((format!("{arch}.use_parallel_residual"), Meta::Bool(true)));
    }

    let normalized = json!({
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "num_hidden_layers": layers,
        "intermediate_size": ff,
        "vocab_size": vocab,
        "head_dim": head_dim,
        "tie_word_embeddings": tie,
    });
    Ok(Some(Plan {
        arch: arch.to_owned(),
        family: Family::Classic,
        meta,
        norm_add: 0.0,
        layers,
        moe: false,
        gated: false,
        moe_layers: vec![false; layers as usize],
        q_lora: false,
        mixtral_names: false,
        synthesized: Vec::new(),
        linear: Some(LinearDims {
            k_heads: 1,
            v_heads: 1,
            k_dim: 1,
            v_dim: 1,
        }),
        kinds: Vec::new(),
        classic: Some(Classic {
            qkv,
            normalized,
            transposed,
        }),
    }))
}

/// `(layer, rest)` of a per-layer tensor name under one of the usual block prefixes.
fn layer_of<'a>(n: &'a str, prefixes: &[&str]) -> Option<(u64, &'a str)> {
    for prefix in prefixes {
        if let Some(rest) = n.strip_prefix(prefix) {
            let (i, suffix) = rest.split_once('.')?;
            return Some((i.parse().ok()?, suffix));
        }
    }
    None
}

pub(crate) fn map(plan: &Plan, n: &str) -> Result<Mapped, String> {
    let classic = plan.classic.as_ref().ok_or("not a classic plan")?;
    let t = |name: String| Ok(Mapped::Tensor { name, add: 0.0 });
    let tr = |name: String| {
        Ok(if classic.transposed {
            Mapped::Transposed { name }
        } else {
            Mapped::Tensor { name, add: 0.0 }
        })
    };
    let arch = plan.arch.as_str();
    // top-level tensors
    let top = n
        .strip_prefix("transformer.")
        .or_else(|| n.strip_prefix("model.decoder."))
        .or_else(|| n.strip_prefix("model."))
        .unwrap_or(n);
    match top {
        "wte.weight" | "word_embeddings.weight" | "embed_tokens.weight" => {
            return t("token_embd.weight".into())
        }
        "wpe.weight" | "embed_positions.weight" => return t("position_embd.weight".into()),
        "ln_f.weight" | "norm_f.weight" | "final_layer_norm.weight" => {
            return t("output_norm.weight".into())
        }
        "ln_f.bias" | "norm_f.bias" | "final_layer_norm.bias" => {
            return t("output_norm.bias".into())
        }
        "word_embeddings_layernorm.weight" => return t("token_embd_norm.weight".into()),
        "word_embeddings_layernorm.bias" => return t("token_embd_norm.bias".into()),
        "lm_head.weight" => return t("output.weight".into()),
        "lm_head.bias" => return t("output.bias".into()),
        _ => {}
    }
    let prefixes = [
        "transformer.h.",
        "h.",
        "transformer.blocks.",
        "model.decoder.layers.",
        "model.layers.",
    ];
    let (i, suffix) =
        layer_of(n, &prefixes).ok_or_else(|| format!("unrecognized checkpoint tensor {n:?}"))?;
    if i >= plan.layers {
        return Err(format!("tensor {n:?} refers to out-of-range layer {i}"));
    }
    let blk = |name: &str| format!("blk.{i}.{name}");
    let mapped: &str = match (arch, suffix) {
        (_, s) if s.ends_with("attn.bias") || s.ends_with("masked_bias") => {
            return Ok(Mapped::Skip)
        }
        (_, s) if s.ends_with("rotary_emb.inv_freq") => return Ok(Mapped::Skip),
        // GPT-2 / BigCode / Neo / J
        ("gpt2" | "starcoder" | "gpt_neo" | "gptj", "ln_1.weight") => "attn_norm.weight",
        ("gpt2" | "starcoder" | "gpt_neo" | "gptj", "ln_1.bias") => "attn_norm.bias",
        ("gpt2" | "starcoder" | "gpt_neo", "ln_2.weight") => "ffn_norm.weight",
        ("gpt2" | "starcoder" | "gpt_neo", "ln_2.bias") => "ffn_norm.bias",
        ("gpt2" | "starcoder", "attn.c_attn.weight") => return tr(blk("attn_qkv.weight")),
        ("gpt2" | "starcoder", "attn.c_attn.bias") => "attn_qkv.bias",
        ("gpt2" | "starcoder", "attn.c_proj.weight") => return tr(blk("attn_output.weight")),
        ("gpt2" | "starcoder", "attn.c_proj.bias") => "attn_output.bias",
        ("gpt2" | "starcoder" | "gpt_neo", "mlp.c_fc.weight") => return tr(blk("ffn_up.weight")),
        ("gpt2" | "starcoder" | "gpt_neo", "mlp.c_fc.bias") => "ffn_up.bias",
        ("gpt2" | "starcoder" | "gpt_neo", "mlp.c_proj.weight") => {
            return tr(blk("ffn_down.weight"))
        }
        ("gpt2" | "starcoder" | "gpt_neo", "mlp.c_proj.bias") => "ffn_down.bias",
        ("gpt_neo", "attn.attention.q_proj.weight") => "attn_q.weight",
        ("gpt_neo", "attn.attention.k_proj.weight") => "attn_k.weight",
        ("gpt_neo", "attn.attention.v_proj.weight") => "attn_v.weight",
        ("gpt_neo", "attn.attention.out_proj.weight") => "attn_output.weight",
        ("gpt_neo", "attn.attention.out_proj.bias") => "attn_output.bias",
        ("gptj", "attn.q_proj.weight") => "attn_q.weight",
        ("gptj", "attn.k_proj.weight") => "attn_k.weight",
        ("gptj", "attn.v_proj.weight") => "attn_v.weight",
        ("gptj", "attn.out_proj.weight") => "attn_output.weight",
        ("gptj", "mlp.fc_in.weight") => "ffn_up.weight",
        ("gptj", "mlp.fc_in.bias") => "ffn_up.bias",
        ("gptj", "mlp.fc_out.weight") => "ffn_down.weight",
        ("gptj", "mlp.fc_out.bias") => "ffn_down.bias",
        // OPT
        ("opt", "self_attn.q_proj.weight") => "attn_q.weight",
        ("opt", "self_attn.q_proj.bias") => "attn_q.bias",
        ("opt", "self_attn.k_proj.weight") => "attn_k.weight",
        ("opt", "self_attn.k_proj.bias") => "attn_k.bias",
        ("opt", "self_attn.v_proj.weight") => "attn_v.weight",
        ("opt", "self_attn.v_proj.bias") => "attn_v.bias",
        ("opt", "self_attn.out_proj.weight") => "attn_output.weight",
        ("opt", "self_attn.out_proj.bias") => "attn_output.bias",
        ("opt", "self_attn_layer_norm.weight") => "attn_norm.weight",
        ("opt", "self_attn_layer_norm.bias") => "attn_norm.bias",
        ("opt", "final_layer_norm.weight") => "ffn_norm.weight",
        ("opt", "final_layer_norm.bias") => "ffn_norm.bias",
        ("opt", "fc1.weight") => "ffn_up.weight",
        ("opt", "fc1.bias") => "ffn_up.bias",
        ("opt", "fc2.weight") => "ffn_down.weight",
        ("opt", "fc2.bias") => "ffn_down.bias",
        // Falcon / BLOOM
        ("falcon" | "bloom", "self_attention.query_key_value.weight") => "attn_qkv.weight",
        ("falcon" | "bloom", "self_attention.query_key_value.bias") => "attn_qkv.bias",
        ("falcon" | "bloom", "self_attention.dense.weight") => "attn_output.weight",
        ("falcon" | "bloom", "self_attention.dense.bias") => "attn_output.bias",
        ("falcon" | "bloom", "mlp.dense_h_to_4h.weight") => "ffn_up.weight",
        ("falcon" | "bloom", "mlp.dense_h_to_4h.bias") => "ffn_up.bias",
        ("falcon" | "bloom", "mlp.dense_4h_to_h.weight") => "ffn_down.weight",
        ("falcon" | "bloom", "mlp.dense_4h_to_h.bias") => "ffn_down.bias",
        ("falcon", "ln_attn.weight") => "attn_norm.weight",
        ("falcon", "ln_attn.bias") => "attn_norm.bias",
        ("falcon", "ln_mlp.weight") => "ffn_norm.weight",
        ("falcon", "ln_mlp.bias") => "ffn_norm.bias",
        ("falcon" | "bloom", "input_layernorm.weight") => "attn_norm.weight",
        ("falcon" | "bloom", "input_layernorm.bias") => "attn_norm.bias",
        ("falcon" | "bloom", "post_attention_layernorm.weight") => "ffn_norm.weight",
        ("falcon" | "bloom", "post_attention_layernorm.bias") => "ffn_norm.bias",
        // MPT
        ("mpt", "norm_1.weight") => "attn_norm.weight",
        ("mpt", "norm_1.bias") => "attn_norm.bias",
        ("mpt", "norm_2.weight") => "ffn_norm.weight",
        ("mpt", "norm_2.bias") => "ffn_norm.bias",
        ("mpt", "attn.Wqkv.weight") => "attn_qkv.weight",
        ("mpt", "attn.Wqkv.bias") => "attn_qkv.bias",
        ("mpt", "attn.out_proj.weight") => "attn_output.weight",
        ("mpt", "attn.out_proj.bias") => "attn_output.bias",
        ("mpt", "ffn.up_proj.weight") => "ffn_up.weight",
        ("mpt", "ffn.up_proj.bias") => "ffn_up.bias",
        ("mpt", "ffn.down_proj.weight") => "ffn_down.weight",
        ("mpt", "ffn.down_proj.bias") => "ffn_down.bias",
        _ => return Err(format!("unsupported forward-affecting layer tensor {n:?}")),
    };
    t(blk(mapped))
}

/// Row order that turns the fused projection of `layout` into `[Q; K; V]`.
pub(crate) fn qkv_order(
    layout: QkvLayout,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Option<Vec<usize>> {
    match layout {
        QkvLayout::Contiguous => None,
        QkvLayout::PerHead => {
            let mut order = vec![0usize; 3 * heads * head_dim];
            for head in 0..heads {
                for part in 0..3 {
                    for i in 0..head_dim {
                        order[part * heads * head_dim + head * head_dim + i] =
                            head * 3 * head_dim + part * head_dim + i;
                    }
                }
            }
            Some(order)
        }
        QkvLayout::KvGroups => {
            let per = heads / kv_heads;
            let mut order = Vec::with_capacity((heads + 2 * kv_heads) * head_dim);
            for g in 0..kv_heads {
                for j in 0..per {
                    order.extend((0..head_dim).map(|i| (g * (per + 2) + j) * head_dim + i));
                }
            }
            for g in 0..kv_heads {
                order.extend((0..head_dim).map(|i| (g * (per + 2) + per) * head_dim + i));
            }
            for g in 0..kv_heads {
                order.extend((0..head_dim).map(|i| (g * (per + 2) + per + 1) * head_dim + i));
            }
            Some(order)
        }
    }
}
