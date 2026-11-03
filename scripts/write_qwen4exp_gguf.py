#!/usr/bin/env python3
"""Write a Qwen3.8-Flash-Next checkpoint (Hugging Face layout, e.g. the tiny `qwen4_exp_gguf` fixture of hf_golden.py) as the
GGUF that llama.cpp's converter writes for `general.architecture = qwen4exp`, in f32.

This is a port of the tensor rules of llama.cpp `conversion/qwen4exp.py` and `conversion/qwen.py` (`Qwen3NextModel`,
`_LinearAttentionVReorderBase`, commit 0ef27309b78d054832e44a95a6fb9314d4b7363c) and of the key and tensor names of
`gguf-py/gguf/constants.py` at that commit, so that the engine's reader of that layout can be tested without a 180 GB checkpoint:

* value heads of the gated delta layers are reordered from grouped (by key head) to tiled order: rows of `in_proj_qkv` (value part),
  `in_proj_z`, `in_proj_a`, `in_proj_b`, `A_log`, `dt_bias`, the value channels of `conv1d`, the columns of `out_proj`;
* `A_log` is stored as `-exp(A_log)` (`ssm_a`), the 1-D convolution kernels lose their unit dimension;
* every `...norm.weight` except the gated delta norm (`linear_attn.norm.weight`) is stored as `1 + w`, and so are the norms of the
  indexer and of the n-gram layer; the per-head `[q, gate]` rows of `q_proj` stay interleaved as they are;
* the indexer projection `index_qk_proj` is split into `indexer.q_proj` (first `heads * dim` rows) and `indexer.k_proj`;
* the shards of the n-gram table are one tensor `per_layer_token_embd`; the hash constants go to the metadata (uint64 arrays);
* the experts are stacked per layer; the shared-expert gate loses its unit row.

With `--merge` the pairs the converter fuses for checkpoints that are not pre-quantized are written fused (`hc_*_down_inject`,
`attn_qkv_gate`, `ssm_beta_alpha`), as current llama.cpp does by default.

    python write_qwen4exp_gguf.py CHECKPOINT_DIR OUT.gguf [--merge]

Needs numpy, torch, safetensors and the `gguf` package.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
import gguf
from gguf import GGUFValueType as VT


def reorder_v(t: torch.Tensor, dim: int, nk: int, vpk: int, hd: int) -> torch.Tensor:
    """grouped (key head major) -> tiled (value-per-key major) value heads along `dim`."""
    shape = list(t.shape)
    new = shape[:dim] + [nk, vpk, hd] + shape[dim + 1:]
    t = t.reshape(*new)
    perm = list(range(len(new)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return t.permute(*perm).contiguous().reshape(*shape)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("output")
    ap.add_argument("--merge", action="store_true", help="write the fused shared-input weights")
    args = ap.parse_args()
    src = Path(args.checkpoint)
    cfg = json.loads((src / "config.json").read_text())
    cfg = cfg.get("text_config", cfg)
    f = safe_open(str(src / "model.safetensors"), "pt")
    get = lambda name: f.get_tensor(name).float() if f.get_slice(name).get_dtype() != "I64" else f.get_tensor(name)

    n_layer = cfg["num_hidden_layers"]
    hidden = cfg["hidden_size"]
    nk, nv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
    hk, hv = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
    assert hk == hv, "llama.cpp has one head width (ssm.state_size) for keys and values"
    vpk = nv // nk
    layer_types = cfg["layer_types"]
    rope = cfg.get("rope_parameters", {})
    ratio = cfg["indexer_compress_ratio"]
    ple_layers = [i - 1 for i in cfg.get("ple_layer_ids", [])]
    assert len(ple_layers) <= 1, "llama.cpp supports one n-gram layer"
    eos = cfg["eos_token_id"]
    eos = int(eos[-1] if isinstance(eos, list) else eos)

    w = gguf.GGUFWriter(args.output, "qwen4exp")
    a = "qwen4exp"
    u32 = lambda k, v: w.add_uint32(f"{a}.{k}", int(v))
    w.add_string("general.name", "tiny qwen4exp fixture")
    u32("block_count", n_layer)
    u32("context_length", cfg["max_position_embeddings"])
    u32("embedding_length", hidden)
    u32("attention.head_count", cfg["num_attention_heads"])
    u32("attention.head_count_kv", cfg["num_key_value_heads"])
    w.add_key_value(f"{a}.rope.dimension_sections", list(rope.get("mrope_section", [11, 11, 10])) + [0], VT.ARRAY, VT.INT32)
    w.add_float32(f"{a}.rope.freq_base", float(rope.get("rope_theta", 10000.0)))
    w.add_float32(f"{a}.attention.layer_norm_rms_epsilon", float(cfg["rms_norm_eps"]))
    u32("expert_count", cfg["num_experts"])
    u32("expert_used_count", cfg["num_experts_per_tok"])
    u32("attention.key_length", cfg["head_dim"])
    u32("attention.value_length", cfg["head_dim"])
    u32("expert_feed_forward_length", cfg["moe_intermediate_size"])
    u32("expert_shared_feed_forward_length", cfg["shared_expert_intermediate_size"])
    u32("ssm.conv_kernel", cfg["linear_conv_kernel_dim"])
    u32("ssm.state_size", hk)
    u32("ssm.group_count", nk)
    u32("ssm.time_step_rank", nv)
    u32("ssm.inner_size", hv * nv)
    u32("full_attention_interval", cfg.get("full_attention_interval", 4))
    u32("rope.dimension_count", int(cfg["head_dim"] * rope.get("partial_rotary_factor", cfg.get("partial_rotary_factor", 0.25))))
    u32("hyper_connection.count", cfg["hc_count"])
    u32("hyper_connection.low_rank", cfg["hc_lowrank"])
    u32("attention.indexer.head_count", cfg["indexer_n_heads"])
    u32("attention.indexer.key_length", cfg["indexer_head_dim"])
    u32("attention.indexer.top_k", cfg["indexer_budget"])
    w.add_key_value(f"{a}.attention.compress_ratios", [0 if t == "linear_attention" else ratio for t in layer_types], VT.ARRAY, VT.INT32)
    if ple_layers:
        w.add_key_value(f"{a}.ple.layers", ple_layers, VT.ARRAY, VT.INT32)
        u32("ple.ngram_size", cfg["ngram_size"])
        u32("ple.heads_per_ngram", cfg["heads_per_ngram"])
        u32("ple.conv_kernel", cfg["ple_conv_kernel_size"])
        u32("ple.eos_token_id", eos)
        layer = ple_layers[0]
        # the row width of one n-gram head (the last dimension of a table shard)
        u32("embedding_length_per_layer_input", f.get_slice(f"model.layers.{layer}.ple.ple_embedding.ngram_embedding.shard_0.weight").get_shape()[-1])
        mult = get(f"model.layers.{layer}.ple.ple_embedding.layer_multipliers").tolist()
        offsets = get(f"model.layers.{layer}.ple.ple_embedding.ngram_heads_offsets").tolist()
        sizes = get(f"model.layers.{layer}.ple.ple_embedding.ngram_heads_vocab_sizes").tolist()
        w.add_key_value(f"{a}.ple.layer_multipliers", [int(x) for x in mult], VT.ARRAY, VT.UINT64)
        w.add_key_value(f"{a}.ple.head_offsets", [int(x) for x in offsets], VT.ARRAY, VT.UINT64)
        w.add_key_value(f"{a}.ple.head_vocab_sizes", [int(x) for x in sizes], VT.ARRAY, VT.UINT64)

    # vocabulary: the byte-level BPE of the checkpoint (llama.cpp only needs it to load the file; the tests feed token ids)
    tok = json.loads((src / "tokenizer.json").read_text())
    vocab = tok["model"]["vocab"]
    tokens = [""] * (max(vocab.values()) + 1)
    for t, i in vocab.items():
        tokens[i] = t
    specials = {t["content"] for t in tok.get("added_tokens", [])}
    merges = [m if isinstance(m, str) else " ".join(m) for m in tok["model"]["merges"]]
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_string("tokenizer.ggml.pre", "qwen35")
    w.add_array("tokenizer.ggml.tokens", tokens)
    w.add_array("tokenizer.ggml.token_type", [3 if t in specials else 1 for t in tokens])
    w.add_array("tokenizer.ggml.merges", merges)
    w.add_uint32("tokenizer.ggml.eos_token_id", eos)
    w.add_uint32("tokenizer.ggml.bos_token_id", eos)
    w.add_uint32("tokenizer.ggml.padding_token_id", eos)
    w.add_bool("tokenizer.ggml.add_bos_token", False)

    out: dict[str, np.ndarray] = {}

    def put(name: str, t: torch.Tensor) -> None:
        assert name not in out, name
        out[name] = t.contiguous().float().numpy()

    plus1 = lambda t: t + 1
    put("token_embd.weight", get("model.embed_tokens.weight"))
    put("output.weight", get("lm_head.weight"))
    put("output_hc_norm.weight", plus1(get("model.hyper_connection_mixer.hc_norm.weight")))
    put("output_hc_down.weight", get("model.hyper_connection_mixer.input_mix_weight_down.weight"))
    put("output_hc_up.weight", get("model.hyper_connection_mixer.input_mix_weight_up.weight"))

    for i in range(n_layer):
        p, b = f"model.layers.{i}", f"blk.{i}"
        for which, ours in (("attn", "attn"), ("mlp", "ffn")):
            h = f"{p}.{which}_hyper_connection"
            put(f"{b}.hc_{ours}_norm.weight", plus1(get(f"{h}.hc_norm.weight")))
            put(f"{b}.hc_{ours}_up.weight", get(f"{h}.input_mix_weight_up.weight"))
            down, inject = get(f"{h}.input_mix_weight_down.weight"), get(f"{h}.block_inject_weight.weight")
            if args.merge:
                put(f"{b}.hc_{ours}_down_inject.weight", torch.cat([down, inject], dim=0))
            else:
                put(f"{b}.hc_{ours}_down.weight", down)
                put(f"{b}.hc_{ours}_inject.weight", inject)
        if layer_types[i] == "linear_attention":
            la = f"{p}.linear_attn"
            qkv = get(f"{la}.in_proj_qkv.weight")
            kd = hk * nk
            qkv = torch.cat([qkv[:kd], qkv[kd:2 * kd], reorder_v(qkv[2 * kd:], 0, nk, vpk, hv)], dim=0)
            z = reorder_v(get(f"{la}.in_proj_z.weight"), 0, nk, vpk, hv)
            alpha = reorder_v(get(f"{la}.in_proj_a.weight"), 0, nk, vpk, 1)
            beta = reorder_v(get(f"{la}.in_proj_b.weight"), 0, nk, vpk, 1)
            if args.merge:
                put(f"{b}.attn_qkv_gate.weight", torch.cat([qkv, z], dim=0))
                put(f"{b}.ssm_beta_alpha.weight", torch.cat([beta, alpha], dim=0))
            else:
                put(f"{b}.attn_qkv.weight", qkv)
                put(f"{b}.attn_gate.weight", z)
                put(f"{b}.ssm_beta.weight", beta)
                put(f"{b}.ssm_alpha.weight", alpha)
            a_log = -torch.exp(get(f"{la}.A_log"))
            put(f"{b}.ssm_a", reorder_v(a_log.unsqueeze(-1), 0, nk, vpk, 1).squeeze(-1))
            put(f"{b}.ssm_dt.bias", reorder_v(get(f"{la}.dt_bias").unsqueeze(-1), 0, nk, vpk, 1).squeeze(-1))
            conv = get(f"{la}.conv1d.weight").squeeze()
            put(f"{b}.ssm_conv1d.weight", torch.cat([conv[:2 * kd], reorder_v(conv[2 * kd:], 0, nk, vpk, hv)], dim=0))
            put(f"{b}.ssm_norm.weight", get(f"{la}.norm.weight"))
            put(f"{b}.ssm_out.weight", reorder_v(get(f"{la}.out_proj.weight"), 1, nk, vpk, hv))
        else:
            sa = f"{p}.self_attn"
            put(f"{b}.attn_q.weight", get(f"{sa}.q_proj.weight"))
            put(f"{b}.attn_k.weight", get(f"{sa}.k_proj.weight"))
            put(f"{b}.attn_v.weight", get(f"{sa}.v_proj.weight"))
            put(f"{b}.attn_output.weight", get(f"{sa}.o_proj.weight"))
            put(f"{b}.attn_q_norm.weight", plus1(get(f"{sa}.q_norm.weight")))
            put(f"{b}.attn_k_norm.weight", plus1(get(f"{sa}.k_norm.weight")))
            qk = get(f"{sa}.indexer.index_qk_proj.weight")
            n_q = cfg["indexer_n_heads"] * cfg["indexer_head_dim"]
            put(f"{b}.indexer.q_proj.weight", qk[:n_q])
            put(f"{b}.indexer.k_proj.weight", qk[n_q:])
            put(f"{b}.indexer.q_norm.weight", plus1(get(f"{sa}.indexer.q_layernorm.weight")))
            put(f"{b}.indexer.k_norm.weight", plus1(get(f"{sa}.indexer.k_layernorm.weight")))
        if i in ple_layers:
            pl = f"{p}.ple"
            put(f"{b}.ple_key.weight", get(f"{pl}.key_proj.weight"))
            put(f"{b}.ple_value.weight", get(f"{pl}.value_proj.weight"))
            for n in ("key", "query", "conv"):
                put(f"{b}.ple_norm_{n}.weight", plus1(get(f"{pl}.norm_{n}.weight")))
            put(f"{b}.ple_conv1d.weight", get(f"{pl}.conv1d.weight").squeeze())
            shards = sorted((k for k in f.keys() if k.startswith(f"{pl}.ple_embedding.ngram_embedding.shard_")), key=lambda k: int(re.search(r"shard_(\d+)", k).group(1)))
            put("per_layer_token_embd.weight", torch.cat([get(k) for k in shards], dim=0))
        put(f"{b}.ffn_gate_inp.weight", get(f"{p}.mlp.gate.weight"))
        put(f"{b}.ffn_gate_inp_shexp.weight", get(f"{p}.mlp.shared_expert_gate.weight").squeeze())
        for ours, theirs in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
            put(f"{b}.ffn_{ours}_shexp.weight", get(f"{p}.mlp.shared_expert.{theirs}.weight"))
            put(f"{b}.ffn_{ours}_exps.weight", torch.stack([get(f"{p}.mlp.experts.{e}.{theirs}.weight") for e in range(cfg["num_experts"])]))

    for name, array in out.items():
        w.add_tensor(name, array)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"{args.output}: {len(out)} tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
