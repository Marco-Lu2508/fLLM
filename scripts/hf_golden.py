#!/usr/bin/env python3
"""Tiny random Hugging Face checkpoints plus transformers reference logits, per architecture.

Each fixture is a complete checkpoint directory (config.json, model.safetensors, tokenizer.json,
tokenizer_config.json) and a golden.json with the transformers logits for a fixed token sequence.
`verify_architectures.py` converts the directory with `fllm-rs convert`, runs fLLM on the same
tokens and compares.

Requires torch, transformers and tokenizers (see docs/architectures.md).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors, normalizers

import types
if "bitsandbytes" not in sys.modules:
    sys.modules["bitsandbytes"] = types.ModuleType("bitsandbytes")

if "transformers_stream_generator" not in sys.modules:
    _d_tsg = types.ModuleType("transformers_stream_generator")
    _d_tsg_main = types.ModuleType("transformers_stream_generator.main")
    _d_tsg_main.NewGenerationMixin = type("NewGenerationMixin", (), {})
    _d_tsg_main.StreamGenerationConfig = type("StreamGenerationConfig", (), {})
    _d_tsg.main = _d_tsg_main
    sys.modules["transformers_stream_generator"] = _d_tsg
    sys.modules["transformers_stream_generator.main"] = _d_tsg_main

from transformers import DynamicCache
if not hasattr(DynamicCache, "from_legacy_cache"):
    DynamicCache.from_legacy_cache = staticmethod(lambda p=None: DynamicCache() if p is None else p)
if not hasattr(DynamicCache, "get_usable_length"):
    DynamicCache.get_usable_length = lambda self, seq_length=0, layer_idx=0: self.get_seq_length(layer_idx) if hasattr(self, "get_seq_length") else 0
if not hasattr(DynamicCache, "to_legacy_cache"):
    DynamicCache.to_legacy_cache = lambda self: ()


TOKENS = int(__import__("os").environ.get("TOKENS", "12"))
SPECIALS = ["<unk>", "<s>", "</s>", "<pad>", "<|im_start|>", "<|im_end|>"]
# Fixtures for aliased architectures are built with the closest native config; the
# saved config.json is then rewritten to the real model_type / config keys.
CONFIG_PATCH = {
    "internlm3": {"model_type": "internlm3"},
    "mimo": {"model_type": "mimo", "set": {"num_nextn_predict_layers": 1}},
    "solar": {"model_type": "solar", "set": {"sliding_window": 2047}},
    "orion": {"model_type": "orion"},
    "iquestcoder": {"model_type": "iquestcoder"},
    "xverse": {"model_type": "xverse", "pop": ["rope_parameters"]},
    "xverse_moe": {
        "model_type": "xverse_moe",
        "set": {
            "num_experts": 4,
            "moe_top_k": 2,
            "num_shared_experts": 1,
            "intermediate_size": 128,
        },
        "pop": ["rope_parameters"],
    },
    "minicpm": {
        "model_type": "minicpm",
        "set": {
            "scale_emb": 12.0,
            "scale_depth": 1.4,
            "dim_model_base": 256.0,
        },
        "pop": ["rope_parameters"],
    },
}
CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Die Hauptstadt von Frankreich ist Paris, und 17 mal 23 ergibt 391.",
    "def add(a, b):\n    return a + b\n",
    "fLLM runs language models in pure Rust, on CPU and GPU.",
    "Zahlen: 0123456789 und Sonderzeichen: !?;:,.-_()[]{}",
] * 8


def make_tokenizer(out: Path, vocab: int) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(CORPUS, trainer)
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    tok.save(str(out / "tokenizer.json"))
    (out / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
                "unk_token": "<unk>",
                "add_bos_token": False,
                "add_eos_token": False,
            }
        )
    )
    return tok


def _k2_config(vocab, mova, **extra):
    import importlib
    sys.path.insert(0, os.path.expanduser(os.environ.get("K2HZ_DIR", "~/k2pkg")))
    mod = importlib.import_module("k2hz.configuration_k2_horizon")
    return mod.K2HorizonConfig(
        vocab_size=vocab, hidden_size=64, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=128, rms_norm_eps=1e-6,
        max_position_embeddings=256, initializer_range=0.08, query_key_norm=False,
        rope_parameters={"rope_type": "default", "rope_theta": 100000.0},
        mova_num_experts=mova, mova_num_experts_per_tok=3 if mova else 0,
        bos_token_id=1, eos_token_id=2, pad_token_id=3, tie_word_embeddings=False, **extra,
    )


def _il2_config(vocab):
    """InternLM2 has no transformers class: its remote code (internlm/internlm2_5-7b-chat,
    configuration_internlm2.py + modeling_internlm2.py) is expected as package `il2` under $IL2_DIR.
    The code needs `einops` (a two-pattern stand-in is enough) and, for transformers >= 5, a one-line
    patch in `_init_rope` (`rope_scaling` is always a dict there; treat a dict without "type" as no scaling)."""
    import importlib
    sys.path.insert(0, os.path.expanduser(os.environ.get("IL2_DIR", "~/il2pkg")))
    mod = importlib.import_module("il2.configuration_internlm2")
    return mod.InternLM2Config(
        vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        rope_theta=10000.0, rms_norm_eps=1e-5, bias=False, rope_scaling=None, initializer_range=0.08, use_cache=False,
        bos_token_id=1, eos_token_id=2, pad_token_id=3, tie_word_embeddings=False,
    )


def _ds_config(vocab):
    import importlib
    for p in [os.environ.get("DSPKG_DIR"), "/home/marco/dspkg", os.path.expanduser("~/dspkg")]:
        if p and p not in sys.path:
            sys.path.insert(0, p)
    mod = importlib.import_module("deepseek.configuration_deepseek")
    return mod.DeepseekConfig(
        vocab_size=vocab, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1,
        first_k_dense_replace=1, scoring_func="softmax", norm_topk_prob=False,
        max_position_embeddings=256, rope_theta=10000.0, rms_norm_eps=1e-6,
        initializer_range=0.08, bos_token_id=1, eos_token_id=2, pad_token_id=3,
        tie_word_embeddings=False,
    )



def _chatglm_config(vocab):
    import importlib
    for p in [os.environ.get("CHATGLM_DIR"), "/home/marco/chatglmpkg", os.path.expanduser("~/chatglmpkg"), "/chatglmpkg"]:
        if p and p not in sys.path:
            sys.path.insert(0, p)
    for parent in ["/home/marco", os.path.expanduser("~"), "/"]:
        if parent and parent not in sys.path:
            sys.path.insert(0, parent)
    try:
        mod = importlib.import_module("chatglmpkg.configuration_chatglm")
    except ImportError:
        mod = importlib.import_module("configuration_chatglm")
    return mod.ChatGLMConfig(
        num_layers=3, padded_vocab_size=vocab, hidden_size=64, ffn_hidden_size=128,
        num_attention_heads=4, multi_query_attention=True, multi_query_group_num=2,
        kv_channels=16, add_qkv_bias=True, rmsnorm=True, seq_length=2048,
        layernorm_epsilon=1e-5, rope_ratio=1, original_rope=True,
    )

def _exaone_config(vocab):
    import importlib
    for p in [os.environ.get("EXAONE_DIR"), "/home/marco/expkg", os.path.expanduser("~/expkg")]:
        if p and p not in sys.path:
            sys.path.insert(0, p)
    mod = importlib.import_module("exaone.configuration_exaone")
    return mod.ExaoneConfig(
        vocab_size=vocab, hidden_size=64, intermediate_size=128,
        num_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, rope_theta=10000.0,
        rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 128},
        layer_norm_epsilon=1e-5, initializer_range=0.02,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
        tie_word_embeddings=False,
    )


def _g9_config(vocab):
    import importlib
    sys.path.insert(0, os.path.expanduser(os.environ.get("G9V3_DIR", "~/g9pkg")))
    mod = importlib.import_module("g9v3.configuration_g9v3")
    return mod.G9v3Config(
        vocab_size=vocab, hidden_size=64, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=128, moe_intermediate_size=32,
        n_routed_experts=6, n_shared_experts=1, num_experts_per_tok=3, first_k_dense_replace=1,
        routed_scaling_factor=2.5, norm_topk_prob=True, use_gated_attention=True,
        rms_norm_eps=1e-6, rope_theta=100000.0, max_position_embeddings=256, initializer_range=0.08,
        bos_token_id=1, eos_token_id=2, pad_token_id=3, tie_word_embeddings=False,
    )


def _baichuan_config(vocab: int):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained("baichuan-inc/baichuan-7B", trust_remote_code=True)
    cfg.vocab_size = vocab
    cfg.hidden_size = 64
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 4
    cfg.intermediate_size = 128
    cfg.max_position_embeddings = 256
    return cfg


def _qwen_config(vocab: int):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained("Qwen/Qwen-1_8B", trust_remote_code=True)
    cfg.vocab_size = vocab
    cfg.hidden_size = 64
    cfg.num_hidden_layers = 3
    cfg.num_attention_heads = 4
    cfg.kv_channels = 16
    cfg.intermediate_size = 128
    cfg.seq_length = 256
    cfg.max_position_embeddings = 256
    cfg.use_dynamic_ntk = False
    cfg.use_logn_attn = False
    cfg.use_flash_attn = False
    cfg.rotary_pct = 1.0
    cfg.rotary_emb_base = 10000.0
    cfg.no_bias = True
    cfg.tie_word_embeddings = False
    return cfg


def config_for(arch: str, vocab: int):
    """Smallest configuration that still exercises every code path of the architecture."""
    import transformers as T

    common = dict(
        vocab_size=vocab,
        hidden_size=64,
        num_hidden_layers=3,
        initializer_range=0.08,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    dense = dict(num_attention_heads=4, num_key_value_heads=2, intermediate_size=128)
    def dbrx_config():
        # the DBRX sub-configs are not exported at the top level of transformers 5.19
        from transformers.models.dbrx.configuration_dbrx import DbrxAttentionConfig, DbrxFFNConfig

        attention = DbrxAttentionConfig(kv_n_heads=2, clip_qkv=0.25)
        # Transformers 5.19 still reads this removed field in DbrxAttention.__init__.
        attention.rope_theta = 10000.0
        return T.DbrxConfig(
            d_model=64, n_heads=4, n_layers=3, max_seq_len=256, vocab_size=vocab,
            initializer_range=0.08, bos_token_id=1, eos_token_id=2, pad_token_id=3,
            attn_config=attention,
            ffn_config=DbrxFFNConfig(ffn_hidden_size=48, moe_num_experts=4,
                moe_top_k=2, moe_normalize_expert_weights=2.0),
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        )

    cfg = {
        "dbrx": dbrx_config,
        "lfm2_moe": lambda: T.Lfm2MoeConfig(
            **{**common, "num_hidden_layers": 4}, **dense, norm_eps=1e-5,
            layer_types=["conv", "full_attention", "conv", "full_attention"],
            conv_L_cache=4, conv_bias=True, num_dense_layers=1,
            num_experts=4, num_experts_per_tok=2, moe_intermediate_size=48,
            use_expert_bias=True, norm_topk_prob=True, routed_scaling_factor=1.7,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        ),
        "lfm2": lambda: T.Lfm2Config(
            **{**common, "num_hidden_layers": 4}, **dense, norm_eps=1e-5,
            layer_types=["conv", "full_attention", "conv", "full_attention"],
            conv_L_cache=3, conv_bias=True, block_auto_adjust_ff_dim=True,
            block_multiple_of=16, block_ffn_dim_multiplier=1.25,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        ),
        "llama": lambda: T.LlamaConfig(**common, **dense, rms_norm_eps=1e-5, rope_theta=10000.0),
        "llama_wide": lambda: T.LlamaConfig(
            **{**common, "hidden_size": 512, "num_hidden_layers": 8}, num_attention_heads=8,
            num_key_value_heads=4, intermediate_size=1024, rms_norm_eps=1e-5, rope_theta=10000.0,
        ),
        "mistral": lambda: T.MistralConfig(**common, **dense, rms_norm_eps=1e-5, rope_theta=10000.0),
        # Package A aliases: the model is built with the closest native config and the
        # saved config.json is rewritten to the real model_type afterwards (CONFIG_PATCH).
        "internlm3": lambda: T.LlamaConfig(
            **common, **dense, rms_norm_eps=1e-5, rope_theta=50_000_000.0,
            rope_parameters={"rope_type": "dynamic", "factor": 6.0, "rope_theta": 50_000_000.0},
        ),
        "mimo": lambda: T.Qwen2Config(
            **common, **dense, rms_norm_eps=1e-5, rope_theta=640000.0, attention_bias=True
        ),
        "solar": lambda: T.LlamaConfig(**common, **dense, rms_norm_eps=1e-5, rope_theta=10000.0),
        "orion": lambda: T.LlamaConfig(**common, **dense, rms_norm_eps=1e-5, rope_theta=10000.0),
        "iquestcoder": lambda: T.LlamaConfig(**common, **dense, rms_norm_eps=1e-5, rope_theta=500_000.0),
        "xverse": lambda: T.LlamaConfig(**common, **dense, rms_norm_eps=1e-6),
        "xverse_moe": lambda: T.LlamaConfig(**common, **dense, rms_norm_eps=1e-6),
        "minicpm": lambda: T.LlamaConfig(**common, **dense, rms_norm_eps=1e-5),
        "persimmon": lambda: T.PersimmonConfig(
            **common,
            **{**dense, "num_key_value_heads": 4},
            layer_norm_eps=1e-5,
            rope_theta=10000.0,
        ),
        "modernbert": lambda: T.ModernBertConfig(
            vocab_size=vocab, hidden_size=64, num_hidden_layers=3, num_attention_heads=4,
            intermediate_size=128, max_position_embeddings=256, pad_token_id=3, bos_token_id=1,
            eos_token_id=2, cls_token_id=1, sep_token_id=2,
            layer_types=["full_attention", "sliding_attention", "sliding_attention"],
            tie_word_embeddings=False,
        ),
        "nomic_bert": lambda: T.NomicBertConfig(
            vocab_size=vocab, hidden_size=64, num_hidden_layers=3, num_attention_heads=4,
            intermediate_size=128, max_position_embeddings=256, type_vocab_size=2,
            hidden_act="silu", layer_norm_eps=1e-12, pad_token_id=3,
            rope_parameters={"rope_type": "default", "rope_theta": 1000.0},
            tie_word_embeddings=False,
        ),
        "xlm_roberta": lambda: T.XLMRobertaConfig(
            vocab_size=vocab, hidden_size=64, num_hidden_layers=3, num_attention_heads=4,
            intermediate_size=128, max_position_embeddings=8194, type_vocab_size=1,
            hidden_act="gelu", layer_norm_eps=1e-5, pad_token_id=1, bos_token_id=0,
            eos_token_id=2, tie_word_embeddings=False,
        ),
        "roberta": lambda: T.RobertaConfig(
            vocab_size=vocab, hidden_size=64, num_hidden_layers=3, num_attention_heads=4,
            intermediate_size=128, max_position_embeddings=514, type_vocab_size=1,
            hidden_act="gelu", layer_norm_eps=1e-5, pad_token_id=1, bos_token_id=0,
            eos_token_id=2, tie_word_embeddings=False,
        ),
        "qwen2": lambda: T.Qwen2Config(**common, **dense, rms_norm_eps=1e-6, rope_theta=1000000.0),
        "qwen3": lambda: T.Qwen3Config(
            **common, **dense, head_dim=32, rms_norm_eps=1e-6, rope_theta=1000000.0
        ),
        "mistral_swa": lambda: T.MistralConfig(
            **common, **dense, rms_norm_eps=1e-5, rope_theta=10000.0, sliding_window=5
        ),
        "gemma": lambda: T.GemmaConfig(
            **common, **dense, head_dim=16, rms_norm_eps=1e-6, hidden_activation="gelu_pytorch_tanh"
        ),
        "gemma2": lambda: T.Gemma2Config(
            **common, **dense, head_dim=16, rms_norm_eps=1e-6, sliding_window=5,
            query_pre_attn_scalar=24, attn_logit_softcapping=20.0, final_logit_softcapping=15.0,
            hidden_activation="gelu_pytorch_tanh",
        ),
        "muse_glimmer": lambda: T.MuseGlimmerConfig(
            text_config=dict(
                vocab_size=vocab, hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
                num_key_value_heads=2, intermediate_size=128, head_dim=16, rms_norm_eps=1e-5,
                post_norm_eps=1e-8, sliding_window=5, qk_scale_factor=3.87,
                output_multiplier=0.35, final_logit_softcapping=20.0, hidden_activation="silu",
                max_position_embeddings=256, initializer_range=0.08, tie_word_embeddings=False,
                bos_token_id=1, eos_token_id=2, pad_token_id=3,
                layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
                layer_rope_theta=[500000.0, 500000.0, 500000.0, 0],
                rope_parameters={"rope_type": "default", "rope_theta": 500000.0},
            ),
            vision_config=dict(
                hidden_size=32, intermediate_size=64, num_attention_heads=2, num_hidden_layers=2,
                layer_types=["window_attention", "full_attention"], patch_size=14,
            ),
            out_hidden_size=64, projector_hidden_size=32, tie_word_embeddings=False,
        ),
        "gemma3": lambda: T.Gemma3TextConfig(
            **common, **dense, head_dim=16, rms_norm_eps=1e-6, sliding_window=5,
            query_pre_attn_scalar=24, rope_local_base_freq=10000.0, rope_theta=1000000.0,
            layer_types=["sliding_attention", "sliding_attention", "full_attention"],
            hidden_activation="gelu_pytorch_tanh",
        ),
        "mixtral": lambda: T.MixtralConfig(
            **common, **dense, rms_norm_eps=1e-5, rope_theta=1000000.0,
            num_local_experts=4, num_experts_per_tok=2,
        ),
        "qwen2_moe": lambda: T.Qwen2MoeConfig(
            **common, **dense, rms_norm_eps=1e-6, rope_theta=1000000.0, num_experts=4,
            num_experts_per_tok=2, moe_intermediate_size=48, shared_expert_intermediate_size=96,
            norm_topk_prob=False,
        ),
        "qwen3_moe": lambda: T.Qwen3MoeConfig(
            **common, **dense, head_dim=16, rms_norm_eps=1e-6, rope_theta=1000000.0,
            num_experts=4, num_experts_per_tok=2, moe_intermediate_size=48, norm_topk_prob=True,
        ),
        "olmoe": lambda: T.OlmoeConfig(
            **common, **dense, rms_norm_eps=1e-5, rope_theta=10000.0, num_experts=4,
            num_experts_per_tok=2, norm_topk_prob=False,
        ),
        "phi3": lambda: T.Phi3Config(**common, **dense, rms_norm_eps=1e-5, rope_theta=10000.0),
        "phi3_longrope": lambda: T.Phi3Config(
            **common, **dense, rms_norm_eps=1e-5, rope_theta=10000.0,
            original_max_position_embeddings=8,
            rope_scaling={
                "rope_type": "longrope",
                "short_factor": [1.0 + 0.1 * i for i in range(8)],
                "long_factor": [2.0 + 0.5 * i for i in range(8)],
            },
        ),
        "deepseek_v3": lambda: T.DeepseekV3Config(
            **common, num_attention_heads=4, num_key_value_heads=4, intermediate_size=128,
            moe_intermediate_size=32, n_routed_experts=8, num_experts_per_tok=2, n_group=2,
            topk_group=1, first_k_dense_replace=1, n_shared_experts=1, kv_lora_rank=16,
            q_lora_rank=24, qk_rope_head_dim=8, qk_nope_head_dim=16, v_head_dim=16,
            routed_scaling_factor=2.5, norm_topk_prob=True, rms_norm_eps=1e-6,
            rope_scaling={
                "type": "yarn", "rope_type": "yarn", "factor": 4.0, "beta_fast": 32.0,
                "beta_slow": 1.0, "mscale": 1.0, "mscale_all_dim": 1.0,
                "original_max_position_embeddings": 64,
            },
        ),
        "deepseek_v2": lambda: T.DeepseekV2Config(
            **common, num_attention_heads=4, num_key_value_heads=4, intermediate_size=128,
            moe_intermediate_size=32, n_routed_experts=4, num_experts_per_tok=2, n_group=2,
            topk_group=1, topk_method="group_limited_greedy", first_k_dense_replace=1,
            n_shared_experts=2, kv_lora_rank=16, q_lora_rank=None, qk_rope_head_dim=8,
            qk_nope_head_dim=16, v_head_dim=16, routed_scaling_factor=1.5, norm_topk_prob=False,
            rms_norm_eps=1e-6,
        ),
        "cohere": lambda: T.CohereConfig(
            **common, **dense, layer_norm_eps=1e-5, rope_theta=10000.0, logit_scale=0.0625,
        ),
        "starcoder2": lambda: T.Starcoder2Config(
            **common, **dense, norm_epsilon=1e-5, rope_theta=100000.0, sliding_window=5,
            hidden_act="gelu_pytorch_tanh",
        ),
        "gpt_neox": lambda: T.GPTNeoXConfig(
            **common, num_attention_heads=4, intermediate_size=128, layer_norm_eps=1e-5,
            rotary_pct=0.25, rotary_emb_base=10000, hidden_act="gelu", use_parallel_residual=True,
        ),
        "gpt_neox_serial": lambda: T.GPTNeoXConfig(
            **common, num_attention_heads=4, intermediate_size=128, layer_norm_eps=1e-5,
            rotary_pct=0.5, rotary_emb_base=10000, hidden_act="gelu", use_parallel_residual=False,
        ),
        "glm": lambda: T.GlmConfig(
            **common, **dense, head_dim=16, rms_norm_eps=1e-5, partial_rotary_factor=0.5,
        ),
        "glm4": lambda: T.Glm4Config(
            **common, **dense, head_dim=16, rms_norm_eps=1e-5, partial_rotary_factor=0.5,
        ),
        "seed_oss": lambda: T.SeedOssConfig(
            **common, **dense, head_dim=16, rms_norm_eps=1e-6, attention_bias=True,
            attention_out_bias=False,
        ),
        "smollm3": lambda: T.SmolLM3Config(
            **common, **dense, rms_norm_eps=1e-6, no_rope_layers=[1, 0, 1],
        ),
        "arcee": lambda: T.ArceeConfig(**common, **dense, rms_norm_eps=1e-5),
        "ernie4_5": lambda: T.Ernie4_5Config(**common, **dense, head_dim=16, rms_norm_eps=1e-5),
        "hunyuan_v1_dense": lambda: T.HunYuanDenseV1Config(
            **common, **dense, head_dim=16, rms_norm_eps=1e-5,
        ),
        "exaone4": lambda: T.Exaone4Config(
            **common, **dense, head_dim=16, rms_norm_eps=1e-5, sliding_window=5,
            layer_types=["sliding_attention", "sliding_attention", "full_attention"],
        ),
        "olmo3": lambda: T.Olmo3Config(
            **common, **dense, rms_norm_eps=1e-6, sliding_window=5,
            layer_types=["sliding_attention", "sliding_attention", "full_attention"],
            rope_parameters={
                "rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 64,
                "rope_theta": 500000.0, "beta_fast": 32.0, "beta_slow": 1.0,
            },
        ),
        "ministral3": lambda: T.Ministral3Config(
            **common, **dense, head_dim=16, rms_norm_eps=1e-5,
            rope_parameters={
                "rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 4,
                "rope_theta": 1000000.0, "beta_fast": 32.0, "beta_slow": 1.0,
                "mscale": 1.0, "mscale_all_dim": 1.0, "llama_4_scaling_beta": 0.3,
            },
        ),
        "glm4_moe": lambda: T.Glm4MoeConfig(
            **common, num_attention_heads=4, num_key_value_heads=2, intermediate_size=128,
            head_dim=16, moe_intermediate_size=32, n_routed_experts=8, num_experts_per_tok=2,
            n_group=2, topk_group=1, first_k_dense_replace=1, n_shared_experts=1,
            routed_scaling_factor=1.8, norm_topk_prob=True, rms_norm_eps=1e-5,
            partial_rotary_factor=0.5, use_qk_norm=True, attention_bias=True,
        ),
        "dots1": lambda: T.Dots1Config(
            **common, num_attention_heads=4, num_key_value_heads=2, intermediate_size=128,
            head_dim=16, moe_intermediate_size=32, n_routed_experts=8, num_experts_per_tok=2,
            n_group=2, topk_group=1, first_k_dense_replace=1, n_shared_experts=1,
            routed_scaling_factor=1.8, norm_topk_prob=True, rms_norm_eps=1e-6, sliding_window=5,
            layer_types=["full_attention", "sliding_attention", "full_attention"],
        ),
        "minimax_m2": lambda: T.MiniMaxM2Config(
            **common, num_attention_heads=4, num_key_value_heads=2, intermediate_size=64,
            head_dim=16, num_local_experts=8, num_experts_per_tok=2, rms_norm_eps=1e-6,
            rotary_dim=8,
        ),
        "ernie4_5_moe": lambda: T.Ernie4_5_MoeConfig(
            **common, num_attention_heads=4, num_key_value_heads=2, intermediate_size=128,
            head_dim=16, moe_intermediate_size=32, num_experts=8, num_experts_per_tok=2,
            moe_num_shared_experts=1, moe_layer_start_index=1, moe_layer_interval=1,
            rms_norm_eps=1e-5,
        ),
        "hunyuan_v1_moe": lambda: T.HunYuanMoEV1Config(
            **common, num_attention_heads=4, num_key_value_heads=2, intermediate_size=64,
            head_dim=16, num_experts=8, moe_topk=2, num_shared_expert=1, rms_norm_eps=1e-5,
        ),
        "gpt_oss": lambda: T.GptOssConfig(
            **common, num_attention_heads=4, num_key_value_heads=2, intermediate_size=32,
            head_dim=16, num_local_experts=8, num_experts_per_tok=2, sliding_window=5,
            rms_norm_eps=1e-5, swiglu_limit=2.0,
            rope_parameters={
                "rope_type": "yarn", "factor": 4.0, "beta_fast": 32.0, "beta_slow": 1.0,
                "truncate": False, "original_max_position_embeddings": 64, "rope_theta": 150000.0,
            },
        ),
        "qwen3_next": lambda: T.Qwen3NextConfig(
            **common, num_attention_heads=4, num_key_value_heads=2, intermediate_size=128,
            head_dim=16, partial_rotary_factor=0.5, rms_norm_eps=1e-6,
            layer_types=["linear_attention", "linear_attention", "full_attention"],
            linear_conv_kernel_dim=3, linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=4,
            num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
            shared_expert_intermediate_size=48, norm_topk_prob=True, decoder_sparse_step=1,
            mlp_only_layers=[],
        ),
        "qwen3_5": lambda: T.Qwen3_5TextConfig(
            **common, num_attention_heads=4, num_key_value_heads=2, intermediate_size=128,
            head_dim=16, rms_norm_eps=1e-6,
            layer_types=["linear_attention", "full_attention", "linear_attention"],
            linear_conv_kernel_dim=3, linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=4,
            rope_parameters={"rope_type": "default", "rope_theta": 100000.0,
                             "partial_rotary_factor": 0.5, "mrope_section": [2, 1, 1]},
        ),
        "mamba2": lambda: T.Mamba2Config(
            vocab_size=vocab, hidden_size=64, num_hidden_layers=3, num_heads=8, head_dim=16,
            state_size=16, expand=2, n_groups=2, conv_kernel=4, layer_norm_epsilon=1e-5,
            bos_token_id=1, eos_token_id=2, pad_token_id=3, tie_word_embeddings=False,
            initializer_range=0.08,
        ),
        "granitemoehybrid": lambda: T.GraniteMoeHybridConfig(
            **common, num_attention_heads=4, num_key_value_heads=2, intermediate_size=32,
            rms_norm_eps=1e-5, num_local_experts=4, num_experts_per_tok=2,
            shared_intermediate_size=48, embedding_multiplier=2.0, logits_scaling=3.0,
            residual_multiplier=0.5, attention_multiplier=0.1,
            layer_types=["linear_attention", "full_attention", "linear_attention"],
            mamba_n_heads=8, mamba_d_head=16, mamba_n_groups=2, mamba_d_state=16, mamba_d_conv=4,
            mamba_expand=2, position_embedding_type=None,
        ),
        "granitehybrid_dense": lambda: T.GraniteMoeHybridConfig(
            **common, num_attention_heads=4, num_key_value_heads=2, intermediate_size=32,
            rms_norm_eps=1e-5, num_local_experts=0, num_experts_per_tok=0,
            shared_intermediate_size=48, embedding_multiplier=2.0, logits_scaling=3.0,
            residual_multiplier=0.5, attention_multiplier=0.1,
            layer_types=["linear_attention", "full_attention", "linear_attention"],
            mamba_n_heads=8, mamba_d_head=16, mamba_n_groups=2, mamba_d_state=16, mamba_d_conv=4,
            mamba_expand=2, position_embedding_type="rope",
        ),
        "bamba": lambda: T.BambaConfig(
            **common, num_attention_heads=4, num_key_value_heads=2, intermediate_size=96,
            rms_norm_eps=1e-5, attn_layer_indices=[1], partial_rotary_factor=0.5,
            mamba_n_heads=8, mamba_d_head=16, mamba_n_groups=2, mamba_d_state=16, mamba_d_conv=4,
            mamba_expand=2, rope_theta=10000.0,
        ),
        "nemotron_h": lambda: T.NemotronHConfig(
            vocab_size=vocab, hidden_size=64, layers_block_type=[
                "linear_attention", "full_attention", "mlp", "moe", "linear_attention"],
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, intermediate_size=96,
            mamba_num_heads=8, mamba_head_dim=16, ssm_state_size=16, conv_kernel=4, n_groups=2,
            n_routed_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
            moe_shared_expert_intermediate_size=48, routed_scaling_factor=1.5, n_group=1,
            topk_group=1, norm_topk_prob=True, layer_norm_epsilon=1e-5, mlp_hidden_act="relu2",
            bos_token_id=1, eos_token_id=2, pad_token_id=3, tie_word_embeddings=False,
            max_position_embeddings=256, initializer_range=0.08,
        ),
        # K2 Horizon ships as remote code; put its configuration_/modeling_ files in a package named k2hz
        # (K2HZ_DIR, default ~/k2pkg) to build this fixture
        "k2_horizon_moe": lambda: _k2_config(
            vocab, mova=0, mlp_only_layers=[0], num_experts=4, num_experts_per_tok=2,
            moe_intermediate_size=48, num_shared_experts=1, moe_gate_bias=True,
            router_score_func="sigmoid", router_scaling_factor=2.5, norm_topk_prob=True,
            attention_gate_func="softplus", layernorm_num_groups=2,
        ),
        "k2_horizon_mova": lambda: _k2_config(
            vocab, mova=6, mlp_only_layers=[0], num_experts=4, num_experts_per_tok=2,
            moe_intermediate_size=48, num_shared_experts=1, moe_gate_bias=True,
            router_score_func="sigmoid", router_scaling_factor=2.5, norm_topk_prob=True,
            attention_gate_func="softplus", layernorm_num_groups=2,
        ),
        # G9v3 is remote code too: put it in a package named g9v3 (G9V3_DIR, default ~/g9pkg)
        "deepseek": lambda: _ds_config(vocab),
        "exaone": lambda: _exaone_config(vocab),
        "chatglm": lambda: _chatglm_config(vocab),
        "zamba2": lambda: T.Zamba2Config(
            **{**common, "num_hidden_layers": 6, "tie_word_embeddings": True},
            intermediate_size=96,
            num_attention_heads=4,
            num_key_value_heads=4,
            rms_norm_eps=1e-5,
            layers_block_type=["mamba", "hybrid", "mamba", "hybrid", "mamba", "hybrid"],
            mamba_d_state=16,
            mamba_d_conv=4,
            mamba_expand=2,
            mamba_ngroups=2,
            n_mamba_heads=8,
            chunk_size=8,
            num_mem_blocks=2,
            use_shared_attention_adapter=True,
            adapter_rank=8,
            use_mem_rope=False,
            time_step_min=0.001,
            hidden_act="gelu",
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        ),
        "jamba": lambda: T.JambaConfig(
            **{**common, "num_hidden_layers": 4},
            intermediate_size=96,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-6,
            num_experts=4,
            num_experts_per_tok=2,
            expert_layer_period=2,
            expert_layer_offset=1,
            attn_layer_period=4,
            attn_layer_offset=2,
            mamba_d_state=8,
            mamba_d_conv=4,
            mamba_expand=2,
            mamba_dt_rank=8,
            mamba_conv_bias=True,
            mamba_proj_bias=False,
        ),
        "falcon_h1": lambda: T.FalconH1Config(
            **common,
            intermediate_size=96,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            rms_norm_eps=1e-5,
            mamba_d_ssm=64,
            mamba_n_heads=8,
            mamba_d_head=8,
            mamba_n_groups=2,
            mamba_d_state=16,
            mamba_d_conv=4,
            mamba_chunk_size=8,
            mamba_conv_bias=True,
            mamba_proj_bias=False,
            mamba_norm_before_gate=True,
            mamba_rms_norm=False,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
            lm_head_multiplier=0.7,
            embedding_multiplier=1.5,
            mlp_multipliers=[0.8, 1.3],
            key_multiplier=0.9,
            attention_out_multiplier=1.2,
            attention_in_multiplier=0.6,
            ssm_multipliers=[1.1, 0.9, 1.3, 0.7, 1.2],
            ssm_in_multiplier=0.8,
            ssm_out_multiplier=1.4,
        ),
        "g9v3": lambda: _g9_config(vocab),
        "internlm2": lambda: _il2_config(vocab),
        "nemotron_h_latent": lambda: T.NemotronHConfig(
            vocab_size=vocab, hidden_size=64, layers_block_type=[
                "linear_attention", "full_attention", "moe", "moe", "linear_attention"],
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, intermediate_size=96,
            mamba_num_heads=8, mamba_head_dim=16, ssm_state_size=16, conv_kernel=4, n_groups=2,
            n_routed_experts=4, num_experts_per_tok=2, moe_intermediate_size=32, moe_latent_size=24,
            moe_shared_expert_intermediate_size=48, routed_scaling_factor=1.5, n_group=1,
            topk_group=1, norm_topk_prob=True, layer_norm_epsilon=1e-5, mlp_hidden_act="relu2",
            bos_token_id=1, eos_token_id=2, pad_token_id=3, tie_word_embeddings=False,
            max_position_embeddings=256, initializer_range=0.08,
        ),
        "jais2": lambda: T.Jais2Config(**common, **dense, head_dim=16, layer_norm_eps=1e-5),
        "nemotron": lambda: T.NemotronConfig(**common, **dense, head_dim=16, norm_eps=1e-5),
        "youtu": lambda: T.YoutuConfig(
            **common, num_attention_heads=4, num_key_value_heads=4, intermediate_size=96,
            kv_lora_rank=16, q_lora_rank=24, qk_rope_head_dim=8, qk_nope_head_dim=16,
            v_head_dim=16, rms_norm_eps=1e-6,
        ),
        "mistral4": lambda: T.Mistral4Config(
            **common, num_attention_heads=4, num_key_value_heads=4, intermediate_size=128,
            moe_intermediate_size=32, num_local_experts=8, num_experts_per_tok=2, n_group=2,
            topk_group=1, first_k_dense_replace=1, n_shared_experts=1, kv_lora_rank=16,
            q_lora_rank=24, qk_rope_head_dim=8, qk_nope_head_dim=16, v_head_dim=16,
            routed_scaling_factor=1.5, norm_topk_prob=True, rms_norm_eps=1e-6,
        ),
        "solar_open": lambda: T.SolarOpenConfig(
            **common, num_attention_heads=4, num_key_value_heads=2, intermediate_size=128,
            head_dim=16, moe_intermediate_size=32, n_routed_experts=8, num_experts_per_tok=2,
            n_group=2, topk_group=1, first_k_dense_replace=1, n_shared_experts=1,
            routed_scaling_factor=1.8, norm_topk_prob=True, rms_norm_eps=1e-5,
        ),
        "cwm": lambda: T.CwmConfig(
            **common, **dense, head_dim=16, rms_norm_eps=1e-5, sliding_window=5,
            layer_types=["sliding_attention", "full_attention", "sliding_attention"],
        ),
        "hyperclovax": lambda: T.HyperCLOVAXConfig(
            **common, **dense, rms_norm_eps=1e-5, use_post_norm=True, embedding_multiplier=2.0,
            logits_scaling=0.5, residual_multiplier=0.7, attention_multiplier=0.09,
        ),
        "flex_olmo": lambda: T.FlexOlmoConfig(
            **common, num_attention_heads=4, num_key_value_heads=2, intermediate_size=48,
            rms_norm_eps=1e-6, num_experts=4, num_experts_per_tok=2,
        ),
        "nanochat": lambda: T.NanoChatConfig(
            **common, **dense, rms_norm_eps=1e-6, final_logit_softcapping=15.0,
        ),
        "hy_v3": lambda: T.HYV3Config(
            **common, num_attention_heads=4, num_key_value_heads=2, intermediate_size=96,
            head_dim=16, num_experts=8, num_experts_per_tok=2, num_shared_experts=1,
            moe_intermediate_size=32, router_scaling_factor=1.7, rms_norm_eps=1e-5,
            mlp_layer_types=["dense", "sparse", "sparse"],
        ),
        "exaone_moe": lambda: T.ExaoneMoeConfig(
            **common, num_attention_heads=4, num_key_value_heads=2, intermediate_size=96,
            moe_intermediate_size=32, num_experts=8, num_experts_per_tok=2, num_shared_experts=1,
            norm_topk_prob=True, routed_scaling_factor=1.6, n_group=2, topk_group=1,
            first_k_dense_replace=1, sliding_window=5, rms_norm_eps=1e-5,
            layer_types=["sliding_attention", "sliding_attention", "full_attention"],
            mlp_layer_types=["dense", "sparse", "sparse"],
        ),
        "afmoe": lambda: T.AfmoeConfig(
            **common, num_attention_heads=4, num_key_value_heads=2, intermediate_size=96,
            moe_intermediate_size=32, num_dense_layers=1, head_dim=16, num_experts=8,
            num_experts_per_tok=2, num_shared_experts=1, route_scale=1.4, sliding_window=5,
            layer_types=["sliding_attention", "sliding_attention", "full_attention"],
            mup_enabled=True, rms_norm_eps=1e-5,
        ),
        "apertus": lambda: T.ApertusConfig(
            **common, **dense, head_dim=16, rms_norm_eps=1e-5, hidden_act="xielu",
        ),
        "minicpm3": lambda: T.MiniCPM3Config(
            **common, num_attention_heads=4, num_key_value_heads=4, intermediate_size=96,
            kv_lora_rank=16, q_lora_rank=24, qk_rope_head_dim=8, qk_nope_head_dim=16,
            rms_norm_eps=1e-5, scale_emb=3.0, scale_depth=1.4, dim_model_base=32,
        ),
        "llama4": lambda: T.Llama4TextConfig(
            **common, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            intermediate_size=32, intermediate_size_mlp=96, num_local_experts=4,
            num_experts_per_tok=1, moe_layers=[1, 2], no_rope_layer_interval=2,
            attention_chunk_size=5, attn_temperature_tuning=True, floor_scale=4, attn_scale=0.3,
            use_qk_norm=True, rms_norm_eps=1e-5, rope_parameters={"rope_type": "default", "rope_theta": 50000.0},
        ),
        "olmo_hybrid": lambda: T.OlmoHybridConfig(
            **common, num_attention_heads=4, num_key_value_heads=2, intermediate_size=96,
            head_dim=16, rms_norm_eps=1e-6,
            layer_types=["linear_attention", "full_attention", "linear_attention"],
            linear_conv_kernel_dim=3, linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=4, linear_allow_neg_eigval=True,
        ),
        "gpt2": lambda: T.GPT2Config(
            vocab_size=vocab, n_embd=64, n_layer=3, n_head=4, n_positions=256,
            bos_token_id=1, eos_token_id=2, pad_token_id=3, tie_word_embeddings=False,
            activation_function="gelu_new", initializer_range=0.08,
        ),
        "gpt_bigcode": lambda: T.GPTBigCodeConfig(
            vocab_size=vocab, n_embd=64, n_layer=3, n_head=4, n_positions=256, multi_query=True,
            bos_token_id=1, eos_token_id=2, tie_word_embeddings=False,
            activation_function="gelu_pytorch_tanh", initializer_range=0.08,
        ),
        "gpt_neo": lambda: T.GPTNeoConfig(
            vocab_size=vocab, hidden_size=64, num_layers=4, num_heads=4,
            max_position_embeddings=256, window_size=5, intermediate_size=128,
            attention_types=[[["global", "local"], 2]], bos_token_id=1, eos_token_id=2,
            tie_word_embeddings=False, initializer_range=0.08,
        ),
        "gptj": lambda: T.GPTJConfig(
            vocab_size=vocab, n_embd=64, n_layer=3, n_head=4, n_positions=256, rotary_dim=8,
            n_inner=128, bos_token_id=1, eos_token_id=2, tie_word_embeddings=False,
            initializer_range=0.08,
        ),
        "opt": lambda: T.OPTConfig(
            vocab_size=vocab, hidden_size=64, num_hidden_layers=3, num_attention_heads=4,
            ffn_dim=128, max_position_embeddings=256, word_embed_proj_dim=64, bos_token_id=1,
            eos_token_id=2, pad_token_id=3, tie_word_embeddings=False, init_std=0.08,
        ),
        "falcon": lambda: T.FalconConfig(
            vocab_size=vocab, hidden_size=64, num_hidden_layers=3, num_attention_heads=4,
            multi_query=True, parallel_attn=True, new_decoder_architecture=False, bias=False,
            alibi=False, bos_token_id=1, eos_token_id=2, tie_word_embeddings=False,
            initializer_range=0.08, rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        ),
        "falcon_new": lambda: T.FalconConfig(
            vocab_size=vocab, hidden_size=64, num_hidden_layers=3, num_attention_heads=4,
            num_kv_heads=2, new_decoder_architecture=True, bias=True, alibi=False,
            bos_token_id=1, eos_token_id=2, tie_word_embeddings=False, initializer_range=0.08,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        ),
        "bloom": lambda: T.BloomConfig(
            vocab_size=vocab, hidden_size=64, n_layer=3, n_head=4, bos_token_id=1,
            eos_token_id=2, pad_token_id=3, tie_word_embeddings=False, initializer_range=0.08,
        ),
        "mpt": lambda: T.MptConfig(
            vocab_size=vocab, d_model=64, n_layers=3, n_heads=4, expansion_ratio=2,
            max_seq_len=256, no_bias=True, tie_word_embeddings=False, initializer_range=0.08,
            attn_config={"alibi": True, "clip_qkv": 3.0, "alibi_bias_max": 8, "attn_impl": "torch"},
        ),
        "phi": lambda: T.PhiConfig(
            **common, **dense, rms_norm_eps=1e-5, layer_norm_eps=1e-5, partial_rotary_factor=0.5,
            qk_layernorm=False, hidden_act="gelu_new",
        ),
        "stablelm": lambda: T.StableLmConfig(
            **common, **dense, layer_norm_eps=1e-5, partial_rotary_factor=0.25, use_qkv_bias=True,
            use_parallel_residual=False, qk_layernorm=False,
        ),
        "stablelm_parallel": lambda: T.StableLmConfig(
            **common, **dense, layer_norm_eps=1e-5, partial_rotary_factor=0.5, use_qkv_bias=False,
            use_parallel_residual=True, qk_layernorm=False,
        ),
        "olmo": lambda: T.OlmoConfig(**common, **dense, clip_qkv=0.3),
        "granitemoe": lambda: T.GraniteMoeConfig(
            **common, num_attention_heads=4, num_key_value_heads=2, intermediate_size=32,
            rms_norm_eps=1e-5, num_local_experts=4, num_experts_per_tok=2, embedding_multiplier=2.0,
            logits_scaling=3.0, residual_multiplier=0.5, attention_multiplier=0.1,
        ),
        "granitemoeshared": lambda: T.GraniteMoeSharedConfig(
            **common, num_attention_heads=4, num_key_value_heads=2, intermediate_size=32,
            rms_norm_eps=1e-5, num_local_experts=4, num_experts_per_tok=2, shared_intermediate_size=48,
            embedding_multiplier=2.0, logits_scaling=3.0, residual_multiplier=0.5,
            attention_multiplier=0.1,
        ),
        "vaultgemma": lambda: T.VaultGemmaConfig(
            vocab_size=vocab, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            sliding_window=4, query_pre_attn_scalar=16,
            attn_logit_softcapping=50.0, final_logit_softcapping=30.0,
            max_position_embeddings=256, initializer_range=0.1,
            layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
            pad_token_id=0, eos_token_id=1, bos_token_id=2, tie_word_embeddings=True,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        ),
        "mellum": lambda: T.MellumConfig(
            vocab_size=vocab, hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, sliding_window=4, num_experts=6, num_experts_per_tok=2,
            moe_intermediate_size=32, norm_topk_prob=True, max_position_embeddings=256, initializer_range=0.1,
            layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
            mlp_layer_types=["dense", "sparse", "sparse", "dense"], pad_token_id=0, eos_token_id=1, bos_token_id=2,
            tie_word_embeddings=False,
            rope_parameters={
                "full_attention": {"rope_type": "default", "rope_theta": 500000.0},
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            },
        ),
        "laguna": lambda: T.LagunaConfig(
            vocab_size=vocab, hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
            num_attention_heads_per_layer=[4, 6, 2, 4], num_key_value_heads=2, head_dim=16, sliding_window=4,
            num_experts=6, num_experts_per_tok=3, moe_intermediate_size=32, shared_expert_intermediate_size=48,
            moe_routed_scaling_factor=1.7, moe_router_logit_softcapping=5.0, max_position_embeddings=256,
            initializer_range=0.1, gating=True,
            layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
            mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
            pad_token_id=0, eos_token_id=1, bos_token_id=2, tie_word_embeddings=False,
            rope_parameters={
                "full_attention": {"rope_type": "default", "rope_theta": 500000.0, "partial_rotary_factor": 0.5},
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
            },
        ),
        "cohere2_moe": lambda: T.Cohere2MoeConfig(
            **common, num_attention_heads=4, num_key_value_heads=2,
            intermediate_size=96, num_experts=4, num_experts_per_tok=2,
            layer_norm_eps=1e-5, logit_scale=0.0625, sliding_window=5, sliding_window_pattern=3,
            layer_types=["sliding_attention", "sliding_attention", "full_attention"],
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        ),
        "cohere2": lambda: T.Cohere2Config(
            **common, **dense, layer_norm_eps=1e-5, logit_scale=0.0625, sliding_window=5,
            layer_types=["sliding_attention", "sliding_attention", "full_attention"],
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        ),
        "mamba": lambda: T.MambaConfig(
            vocab_size=vocab, hidden_size=64, state_size=8, num_hidden_layers=3, expand=2,
            conv_kernel=4, time_step_rank=4, layer_norm_epsilon=1e-5, bos_token_id=1,
            eos_token_id=2, pad_token_id=3, tie_word_embeddings=False, initializer_range=0.08,
        ),
        "falcon_mamba": lambda: T.FalconMambaConfig(
            vocab_size=vocab, hidden_size=64, state_size=8, num_hidden_layers=3, expand=2,
            conv_kernel=4, time_step_rank=4, layer_norm_epsilon=1e-5, mixer_rms_eps=1e-6,
            bos_token_id=1, eos_token_id=2, pad_token_id=3, tie_word_embeddings=False,
            initializer_range=0.08,
        ),
        "qwen3_5_moe": lambda: T.Qwen3_5MoeTextConfig(
            **common, num_attention_heads=4, num_key_value_heads=2, intermediate_size=128,
            head_dim=16, rms_norm_eps=1e-6,
            layer_types=["linear_attention", "full_attention", "linear_attention"],
            linear_conv_kernel_dim=3, linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=4,
            num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
            shared_expert_intermediate_size=48,
            rope_parameters={"rope_type": "default", "rope_theta": 100000.0,
                             "partial_rotary_factor": 0.5, "mrope_section": [2, 1, 1]},
        ),
        "phimoe": lambda: T.PhimoeConfig(
            **common, **dense, head_dim=16, rms_norm_eps=1e-5, num_local_experts=4,
            num_experts_per_tok=2, router_jitter_noise=0.01,
        ),
        "glm4_moe_lite": lambda: T.Glm4MoeLiteConfig(
            **common, num_attention_heads=4, num_key_value_heads=4, intermediate_size=128,
            moe_intermediate_size=32, n_routed_experts=8, num_experts_per_tok=2, n_group=2,
            topk_group=1, n_shared_experts=1, kv_lora_rank=16, q_lora_rank=24,
            qk_rope_head_dim=8, qk_nope_head_dim=16, v_head_dim=16, routed_scaling_factor=1.5,
            norm_topk_prob=True, rms_norm_eps=1e-6,
            mlp_layer_types=["dense", "sparse", "sparse"],
        ),
        "deepseek_v32": lambda: T.DeepseekV32Config(
            **common, num_attention_heads=4, num_key_value_heads=4, intermediate_size=128,
            moe_intermediate_size=32, n_routed_experts=8, num_experts_per_tok=2, n_group=2,
            topk_group=1, first_k_dense_replace=1, n_shared_experts=1, kv_lora_rank=16,
            q_lora_rank=24, qk_rope_head_dim=8, qk_nope_head_dim=16, v_head_dim=16,
            routed_scaling_factor=2.5, rms_norm_eps=1e-6,
            index_n_heads=4, index_head_dim=16, index_topk=64,  # topk >= sequence length: the indexer keeps every key
        ),
        "glm_moe_dsa": lambda: T.GlmMoeDsaConfig(
            **common, num_attention_heads=4, num_key_value_heads=4, intermediate_size=128,
            moe_intermediate_size=32, n_routed_experts=8, num_experts_per_tok=2, n_group=1,
            topk_group=1, first_k_dense_replace=1, n_shared_experts=1, kv_lora_rank=16,
            q_lora_rank=24, qk_rope_head_dim=8, qk_nope_head_dim=16, v_head_dim=16,
            routed_scaling_factor=2.5, rms_norm_eps=1e-5,
            index_n_heads=4, index_head_dim=16, index_topk=64,
            indexer_types=["full", "shared", "full"],
        ),
        "mimo_v2_flash": lambda: T.MiMoV2FlashConfig(
            **{**common, "num_hidden_layers": 4}, num_attention_heads=4, num_key_value_heads=2, intermediate_size=128,
            moe_intermediate_size=32, n_routed_experts=8, num_experts_per_tok=2, n_group=2, topk_group=1,
            head_dim=16, v_head_dim=8, sliding_window=6, attention_value_scale=0.75, rms_norm_eps=1e-5,
            routed_scaling_factor=1.5, norm_topk_prob=True,
            layer_types=["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
            mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
            rope_parameters={
                "full_attention": {"rope_type": "default", "rope_theta": 500000.0, "partial_rotary_factor": 0.5},
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
            },
        ),
        "helium": lambda: T.HeliumConfig(
            **common, **dense, head_dim=16, rms_norm_eps=1e-8, rope_theta=20000.0,
        ),
        "olmo2": lambda: T.Olmo2Config(**common, **dense, rms_norm_eps=1e-6, rope_theta=500000.0),
        "qwen": lambda: _qwen_config(vocab),
        "baichuan": lambda: _baichuan_config(vocab),
        "granite": lambda: T.GraniteConfig(
            **common, **dense, rms_norm_eps=1e-5, rope_theta=10000.0, embedding_multiplier=3.0,
            attention_multiplier=0.07, residual_multiplier=0.4, logits_scaling=4.0,
        ),
    }
    if arch not in cfg:
        raise SystemExit(f"unknown architecture {arch!r}; known: {', '.join(sorted(cfg))}")
    return cfg[arch]()



def build_classifier(arch: str, out: Path, vocab: int, seed: int) -> dict:
    """Sequence-classification fixtures for verify_encoder.py, never causal-LM logits."""
    import transformers as T

    out.mkdir(parents=True, exist_ok=True)
    roberta = arch == "xlm_roberta_classifier_1"
    if roberta:
        tok = make_tokenizer(out, vocab)
        tok.post_processor = processors.TemplateProcessing(
            single="<s> $A </s>", pair="<s> $A </s> </s> $B </s>",
            special_tokens=[("<s>", tok.token_to_id("<s>")), ("</s>", tok.token_to_id("</s>"))],
        )
    else:
        tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        tok.normalizer = normalizers.BertNormalizer()
        tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tok.decoder = decoders.WordPiece(prefix="##", cleanup=True)
        tok.train_from_iterator(CORPUS, trainers.WordPieceTrainer(
            vocab_size=vocab, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], show_progress=False,
        ))
        tok.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))],
        )
        (out / "tokenizer_config.json").write_text(json.dumps({
            "tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]", "cls_token": "[CLS]",
            "sep_token": "[SEP]", "pad_token": "[PAD]", "mask_token": "[MASK]", "do_lower_case": True,
        }))
    tok.save(str(out / "tokenizer.json"))
    labels = 3 if arch == "bert_classifier_3" else 1
    common = dict(vocab_size=tok.get_vocab_size(), hidden_size=64, num_hidden_layers=3,
                  num_attention_heads=4, intermediate_size=128, max_position_embeddings=256,
                  pad_token_id=3, num_labels=labels, hidden_dropout_prob=0.0,
                  attention_probs_dropout_prob=0.0, initializer_range=0.08,
                  id2label={i: f"CLASS_{i}" for i in range(labels)},
                  label2id={f"CLASS_{i}": i for i in range(labels)})
    cfg = T.XLMRobertaConfig(**common, type_vocab_size=1, layer_norm_eps=1e-5) if roberta else T.BertConfig(**common, type_vocab_size=2, layer_norm_eps=1e-12)
    torch.manual_seed(seed)
    model = T.AutoModelForSequenceClassification.from_config(cfg, attn_implementation="eager")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("LayerNorm.weight"):
                parameter.copy_(0.8 + 0.4 * torch.rand_like(parameter))
    model.eval()
    model.save_pretrained(out, safe_serialization=True)
    examples = []
    for query, document in [("The cat", "The cat sits on the mat."), ("Eine Straße", "Grüße aus Paris!"), ("The cat", "")]:
        encoding = tok.encode(query, document, add_special_tokens=True)
        ids = torch.tensor([encoding.ids], dtype=torch.long)
        types = torch.tensor([encoding.type_ids], dtype=torch.long)
        with torch.no_grad():
            logits = model(input_ids=ids, token_type_ids=types, attention_mask=torch.ones_like(ids)).logits[0].float()
            probs = torch.sigmoid(logits) if labels == 1 else torch.softmax(logits, dim=-1)
        examples.append(dict(query=query, document=document, ids=encoding.ids, token_type_ids=encoding.type_ids,
                             logits=logits.tolist(), probs=probs.tolist()))
    golden = dict(arch=arch, task="sequence_classification", num_labels=labels, vocab=tok.get_vocab_size(),
                  ids=examples[0]["ids"], logits=examples[0]["logits"], shape=[labels], examples=examples)
    (out / "golden.json").write_text(json.dumps(golden))
    return golden


def build_modernbert(out: Path, vocab: int, seed: int) -> dict:
    """ModernBERT embedding fixture: last-hidden pooled vectors (cls/mean, L2)."""
    import transformers as T

    out.mkdir(parents=True, exist_ok=True)
    tok = make_tokenizer(out, vocab)
    vocab = tok.get_vocab_size()
    cfg = config_for("modernbert", vocab)
    torch.manual_seed(seed)
    model = T.AutoModel.from_config(cfg, attn_implementation="eager")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("norm.weight") or name.endswith("ln_1.weight") or name.endswith("ln_2.weight") or name.endswith("ln_f.weight"):
                parameter.copy_(0.8 + 0.4 * torch.rand_like(parameter))
    model.eval()
    model.save_pretrained(out, safe_serialization=True)
    generator = torch.Generator().manual_seed(seed + 1)
    examples = []
    for length in (12, 60, 200):
        ids = torch.randint(4, vocab, (1, length), generator=generator)
        mask = torch.ones_like(ids)
        with torch.no_grad():
            hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state[0].float()
        for pooling in ("cls", "mean"):
            vector = hidden[0] if pooling == "cls" else hidden.mean(dim=0)
            vector = vector / vector.norm().item()
            examples.append(dict(ids=ids[0].tolist(), pooling=pooling,
                                 embedding=vector.tolist()))
    golden = dict(arch="modernbert", task="embedding", hidden=64, layers=3,
                  shape=[64], examples=examples)
    (out / "golden.json").write_text(json.dumps(golden))
    return golden


def build_nomicbert(out: Path, vocab: int, seed: int) -> dict:
    """NomicBERT embedding fixture: WordPiece vocab, RoPE+SwiGLU, mean-pooled L2 vectors."""
    import transformers as T

    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer()
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.decoder = decoders.WordPiece(prefix="##", cleanup=True)
    tok.train_from_iterator(CORPUS, trainers.WordPieceTrainer(
        vocab_size=vocab, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], show_progress=False,
    ))
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))],
    )
    tok.save(str(out / "tokenizer.json"))
    (out / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]", "cls_token": "[CLS]",
        "sep_token": "[SEP]", "pad_token": "[PAD]", "mask_token": "[MASK]", "do_lower_case": True,
    }))
    vocab = tok.get_vocab_size()
    cfg = config_for("nomic_bert", vocab)
    torch.manual_seed(seed)
    model = T.AutoModel.from_config(cfg, attn_implementation="eager")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("norm.weight") or name.endswith("ln_1.weight") or name.endswith("ln_2.weight") or name.endswith("ln_f.weight"):
                parameter.copy_(0.8 + 0.4 * torch.rand_like(parameter))
    model.eval()
    model.save_pretrained(out, safe_serialization=True)
    generator = torch.Generator().manual_seed(seed + 1)
    examples = []
    for length in (12, 60, 200):
        ids = torch.randint(4, vocab, (1, length), generator=generator)
        mask = torch.ones_like(ids)
        with torch.no_grad():
            hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state[0].float()
        for pooling in ("cls", "mean"):
            vector = hidden[0] if pooling == "cls" else hidden.mean(dim=0)
            vector = vector / vector.norm().item()
            examples.append(dict(ids=ids[0].tolist(), pooling=pooling,
                                 embedding=vector.tolist()))
    golden = dict(arch="nomic_bert", task="embedding", hidden=64, layers=3,
                  shape=[64], examples=examples)
    (out / "golden.json").write_text(json.dumps(golden))
    return golden


def build_xlm_roberta(out: Path, vocab: int, seed: int, arch: str = "xlm_roberta") -> dict:
    """XLM-R/RoBERTa embedding fixture: BPE vocab, absolute positions (xlm_roberta
    uses 8194 rows and exercises the bge-m3 context-gate fix), CLS/mean-pooled
    L2 vectors."""
    import transformers as T

    out.mkdir(parents=True, exist_ok=True)
    tok = make_tokenizer(out, vocab)
    vocab = tok.get_vocab_size()
    cfg = config_for(arch, vocab)
    torch.manual_seed(seed)
    model = T.AutoModel.from_config(cfg, attn_implementation="eager")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("LayerNorm.weight"):
                parameter.copy_(0.8 + 0.4 * torch.rand_like(parameter))
    model.eval()
    model.save_pretrained(out, safe_serialization=True)
    generator = torch.Generator().manual_seed(seed + 1)
    examples = []
    for length in (12, 60, 200):
        ids = torch.randint(4, vocab, (1, length), generator=generator)
        mask = torch.ones_like(ids)
        with torch.no_grad():
            hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state[0].float()
        for pooling in ("cls", "mean"):
            vector = hidden[0] if pooling == "cls" else hidden.mean(dim=0)
            vector = vector / vector.norm().item()
            examples.append(dict(ids=ids[0].tolist(), pooling=pooling,
                                 embedding=vector.tolist()))
    golden = dict(arch=arch, task="embedding", hidden=64, layers=3,
                  shape=[64], examples=examples)
    (out / "golden.json").write_text(json.dumps(golden))
    return golden


def build(arch: str, out: Path, vocab: int = 320, seed: int = 1234) -> dict:
    import transformers as T

    if arch == "modernbert":
        return build_modernbert(out, vocab, seed)
    if arch == "nomic_bert":
        return build_nomicbert(out, vocab, seed)
    if arch == "xlm_roberta":
        return build_xlm_roberta(out, vocab, seed)
    if arch == "roberta":
        return build_xlm_roberta(out, vocab, seed, arch="roberta")
    if arch in {"bert_classifier_1", "bert_classifier_3", "xlm_roberta_classifier_1"}:
        return build_classifier(arch, out, vocab, seed)
    out.mkdir(parents=True, exist_ok=True)
    tok = make_tokenizer(out, vocab)
    vocab = tok.get_vocab_size()
    cfg = config_for(arch, vocab)
    torch.manual_seed(seed)
    try:
        model = T.AutoModelForCausalLM.from_config(cfg, dtype=torch.float32, attn_implementation="eager", trust_remote_code=True)
    except (ValueError, TypeError):
        # newer architectures that the Auto mapping does not know yet
        base = type(cfg).__name__.replace("Config", "")
        cls = (
            getattr(T, base + "ForCausalLM", None)
            or getattr(T, base + "ForConditionalGeneration", None)
            or getattr(T, base + "LMHeadModel", None)
        )
        if cls is None:  # remote-code model that lives next to its configuration module
            import importlib
            mod = importlib.import_module(type(cfg).__module__.replace("configuration_", "modeling_"))
            cls = (
                getattr(mod, base + "ForCausalLM", None)
                or getattr(mod, base + "ForConditionalGeneration", None)
                or getattr(mod, base + "LMHeadModel", None)
                or getattr(mod, "BaiChuanForCausalLM", None)
                or getattr(mod, "BaichuanForCausalLM", None)
            )
        model = cls._from_config(cfg, dtype=torch.float32, attn_implementation="eager")
    if arch == "qwen":
        import inspect
        mod = inspect.getmodule(type(model))
        mod.QWenPreTrainedModel.get_head_mask = (
            lambda self, head_mask, n, is_attention_chunked=False: [None] * n
            if head_mask is None
            else head_mask
        )
    if arch == "orion":
        # Orion uses biased LayerNorms (input/post-attention/final) on top of Llama.
        import torch.nn as nn
        for layer in model.model.layers:
            layer.input_layernorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            layer.post_attention_layernorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        model.model.norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    if arch == "xverse_moe":
        import torch.nn as nn
        import torch.nn.functional as F

        class XverseMLP(nn.Module):
            def __init__(self, hidden_size, intermediate_size):
                super().__init__()
                self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
                self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
                self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
                self.act_fn = nn.SiLU()

            def forward(self, x):
                return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        class XverseMoEMLP(nn.Module):
            def __init__(self, hidden_size, intermediate_size, num_experts=4, top_k=2, num_shared_experts=1):
                super().__init__()
                self.hidden_size = hidden_size
                self.top_k = top_k
                self.num_experts = num_experts
                self.router = nn.Linear(hidden_size, num_experts, bias=False)
                self.experts = nn.ModuleList([XverseMLP(hidden_size, intermediate_size) for _ in range(num_experts)])
                if num_shared_experts:
                    self.shared_experts = XverseMLP(hidden_size, num_shared_experts * intermediate_size)
                else:
                    self.shared_experts = None

            def forward(self, hidden_states):
                batch_size, seq_len, hidden_dim = hidden_states.shape
                flat = hidden_states.view(-1, hidden_dim)
                router_logits = self.router(flat)
                weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
                routing_weights, selected = torch.topk(weights, self.top_k, dim=-1)
                routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-6)
                routing_weights = routing_weights.to(hidden_states.dtype)

                out = torch.zeros_like(flat)
                for k in range(self.top_k):
                    exp_idx = selected[:, k]
                    w = routing_weights[:, k : k + 1]
                    for e in range(self.num_experts):
                        mask = exp_idx == e
                        if mask.any():
                            out[mask] += w[mask] * self.experts[e](flat[mask])
                if self.shared_experts is not None:
                    out += self.shared_experts(flat)
                return out.view(batch_size, seq_len, hidden_dim)

        for layer in model.model.layers:
            layer.mlp = XverseMoEMLP(cfg.hidden_size, cfg.intermediate_size, num_experts=4, top_k=2, num_shared_experts=1)
    if arch == "minicpm":
        import torch.nn as nn
        import torch.nn.functional as F
        import math

        class ScaledEmbedding(nn.Module):
            def __init__(self, orig, scale):
                super().__init__()
                self.weight = orig.weight
                self.scale = scale

            def forward(self, x):
                return F.embedding(x, self.weight) * self.scale

        class ScaledHead(nn.Module):
            def __init__(self, orig, scale):
                super().__init__()
                self.weight = orig.weight
                self.bias = orig.bias
                self.scale = scale

            def forward(self, x):
                return F.linear(x / self.scale, self.weight, self.bias)

        class MiniCPMDecoderLayer(nn.Module):
            def __init__(self, layer, scale_depth, num_layers):
                super().__init__()
                self.self_attn = layer.self_attn
                self.mlp = layer.mlp
                self.input_layernorm = layer.input_layernorm
                self.post_attention_layernorm = layer.post_attention_layernorm
                self.residual_scale = scale_depth / math.sqrt(num_layers)

            def forward(self, hidden_states, **kwargs):
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
                hidden_states = self.self_attn(hidden_states, **kwargs)
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
                hidden_states = residual + hidden_states * self.residual_scale
                residual = hidden_states
                hidden_states = self.post_attention_layernorm(hidden_states)
                hidden_states = self.mlp(hidden_states)
                hidden_states = residual + hidden_states * self.residual_scale
                return hidden_states

        model.model.embed_tokens = ScaledEmbedding(model.model.embed_tokens, 12.0)
        model.lm_head = ScaledHead(model.lm_head, cfg.hidden_size / 256.0)
        for i in range(len(model.model.layers)):
            model.model.layers[i] = MiniCPMDecoderLayer(model.model.layers[i], 1.4, len(model.model.layers))
    # keep the norm weights away from the trivial 1.0 so that missing norms are visible
    with torch.no_grad():
        if arch == "lfm2_moe":
            for name, buffer in model.named_buffers():
                if name.endswith("expert_bias"):
                    buffer.copy_(torch.linspace(-0.2, 0.2, buffer.numel()).reshape_as(buffer))
        for name, p in model.named_parameters():
            if name.endswith("norm.weight") or name.endswith("ln_1.weight") or name.endswith("ln_2.weight") or name.endswith("ln_f.weight"):
                p.copy_(0.8 + 0.4 * torch.rand_like(p))
            if not bool(torch.isfinite(p).all()):
                p.normal_(std=0.2)
            if name.endswith("A_log"):
                p.copy_(torch.log(torch.empty_like(p).uniform_(0.3, 3.0)))
            if name.endswith(".D"):
                p.normal_(std=1.0)
            if name.endswith("dt_bias"):
                p.normal_(std=0.5)
            if "adapter_list" in name:
                p.normal_(std=0.25)
            if arch in ("nemotron", "orion", "persimmon") and name.endswith("norm.bias"):
                p.normal_(std=0.2)  # LayerNorm biases must matter
            if arch == "chatglm" and p.dim() >= 2:
                p.normal_(std=0.05)  # the remote-code model leaves its weights uninitialised (NaN or garbage)
            if arch == "chatglm" and p.dim() == 1 and name.endswith(".bias"):
                p.normal_(std=0.01)
            if name.endswith("alpha_p") or name.endswith("alpha_n"):
                p.normal_(std=0.5)
            if name.endswith("sinks"):
                p.normal_(std=1.0)
            if p.dim() == 2 and float(p.abs().sum()) == 0.0:
                p.normal_(std=0.3)  # DeepSeek-V3 initialises its routers with zeros (all ties)
            if name.endswith("e_score_correction_bias"):
                p.copy_(0.2 * torch.randn_like(p))
            if name.endswith(".gate.bias") or name.endswith("v_router.bias"):
                p.normal_(std=0.3)  # K2 Horizon selection biases
        for name, b in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                b.copy_(0.2 * torch.randn_like(b))
    model.eval()
    model.save_pretrained(out, safe_serialization=True)
    patch = CONFIG_PATCH.get(arch)
    if patch:
        cfg_path = out / "config.json"
        data = json.loads(cfg_path.read_text())
        data["model_type"] = patch["model_type"]
        for key in patch.get("pop", ()):
            data.pop(key, None)
        data.update(patch.get("set", {}))
        cfg_path.write_text(json.dumps(data, indent=2) + "\n")
    g = torch.Generator().manual_seed(seed + 1)
    ids = torch.randint(4, vocab, (1, TOKENS), generator=g)
    with torch.no_grad():
        logits = model(ids).logits[0].float()
    golden = {
        "arch": arch,
        "ids": ids[0].tolist(),
        "vocab": vocab,
        "logits": logits.reshape(-1).tolist(),
        "shape": list(logits.shape),
    }
    (out / "golden.json").write_text(json.dumps(golden))
    return golden


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arch")
    parser.add_argument("output")
    parser.add_argument("--vocab", type=int, default=320)
    args = parser.parse_args()
    golden = build(args.arch, Path(args.output), args.vocab)
    print(f"{args.arch}: {golden['shape']} logits written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

