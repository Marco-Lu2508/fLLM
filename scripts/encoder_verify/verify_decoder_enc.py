#!/usr/bin/env python3
"""Decoder-architecture encoders against transformers: Qwen3 embedder (last token + EOS suffix),
EmbeddingGemma style Gemma 3 (bidirectional sliding window, mean pooling, two Dense modules,
Normalize), Llama bidirectional with llama3 rotary scaling, sequence-classification reward models and
the Qwen3-Reranker yes/no scoring.

usage: verify_decoder_enc.py [workdir]
"""
import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import common  # noqa: E402

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import (Gemma3TextConfig, Gemma3TextModel, LlamaConfig, LlamaModel,  # noqa: E402
                          Qwen2Config, Qwen2ForSequenceClassification, Qwen3Config, Qwen3ForCausalLM,
                          Qwen3ForSequenceClassification, Qwen3Model)

work = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="dec_")
TEXTS = ["the cat sat on the mat", "hello world", "paris is the capital of france and berlin is the capital of germany"]


def randomize(model, std=0.25):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.dim() > 1:
                p.normal_(0, std)
            else:
                p.normal_(0, 0.2)
    model.eval()


def tok_ids(tok, text, special=True):
    return tok(text, add_special_tokens=special)["input_ids"]


def forward(model, ids):
    with torch.no_grad():
        return model(input_ids=torch.tensor([ids])).last_hidden_state[0]


def normalized(v):
    return torch.nn.functional.normalize(v, dim=-1).tolist()


# ------------------------------------------------------------------ Qwen3 embedder
def qwen3_embedder():
    d = os.path.join(work, "qwen3_emb")
    common.seed(11)
    n = common.write_bpe_tokenizer(d, specials=("<|endoftext|>",), single_template="$A <|endoftext|>")
    tok = common.load_fast_tokenizer(d)
    cfg = Qwen3Config(vocab_size=n, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=128,
                      rope_theta=1_000_000.0, tie_word_embeddings=False)
    cfg.architectures = ["Qwen3Model"]
    model = Qwen3Model(cfg)
    randomize(model)
    model.save_pretrained(d)
    with common.Server(convert_dir(d, "qwen3_emb")) as srv:
        code, out = srv.post("/v1/embeddings", {"model": "m", "input": TEXTS})
        assert code == 200, out
        for i, t in enumerate(TEXTS):
            ids = tok_ids(tok, t)
            assert ids[-1] == tok.convert_tokens_to_ids("<|endoftext|>")
            common.check(f"qwen3 embedder[{i}]", out["data"][i]["embedding"], normalized(forward(model, ids)[-1]))


def convert_dir(d, tag):
    gguf = os.path.join(work, tag + ".gguf")
    print(common.convert(d, gguf))
    return gguf


# ------------------------------------------------------------------ Gemma 3 bidirectional (EmbeddingGemma style)
def gemma3_embedding():
    d = os.path.join(work, "embeddinggemma_tiny")
    common.seed(12)
    n = common.write_bpe_tokenizer(d, specials=("<bos>", "<eos>", "<pad>"), single_template="<bos> $A <eos>")
    tok = common.load_fast_tokenizer(d, pad="<pad>")
    cfg = Gemma3TextConfig(vocab_size=n, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=8, query_pre_attn_scalar=12,
                           sliding_window=6, layer_types=["sliding_attention", "sliding_attention", "full_attention"],
                           max_position_embeddings=128, use_bidirectional_attention=True,
                           rms_norm_eps=1e-6, hidden_activation="gelu_pytorch_tanh")
    cfg.architectures = ["Gemma3TextModel"]
    model = Gemma3TextModel(cfg)
    randomize(model)
    model.save_pretrained(d)
    # hub-style rope keys, as in the EmbeddingGemma config.json
    c = json.load(open(os.path.join(d, "config.json")))
    c.pop("rope_parameters", None)
    c["rope_theta"], c["rope_local_base_freq"] = 1000000.0, 10000.0
    json.dump(c, open(os.path.join(d, "config.json"), "w"))
    model = Gemma3TextModel.from_pretrained(d)
    model.eval()
    # sentence-transformers layout: Transformer, mean Pooling, Dense (no bias), Dense (no bias), Normalize
    dense = []
    for i, (a, b) in enumerate([(32, 48), (48, 32)]):
        lin = torch.nn.Linear(a, b, bias=False)
        with torch.no_grad():
            lin.weight.normal_(0, 0.4)
        dense.append(lin)
        os.makedirs(os.path.join(d, f"{i + 2}_Dense"), exist_ok=True)
        json.dump({"in_features": a, "out_features": b, "bias": False,
                   "activation_function": "torch.nn.modules.linear.Identity"},
                  open(os.path.join(d, f"{i + 2}_Dense", "config.json"), "w"))
        save_file({"linear.weight": lin.weight.detach().contiguous()}, os.path.join(d, f"{i + 2}_Dense", "model.safetensors"))
    os.makedirs(os.path.join(d, "1_Pooling"), exist_ok=True)
    json.dump({"word_embedding_dimension": 32, "pooling_mode_mean_tokens": True},
              open(os.path.join(d, "1_Pooling", "config.json"), "w"))
    json.dump([{"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
               {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
               {"idx": 2, "name": "2", "path": "2_Dense", "type": "sentence_transformers.models.Dense"},
               {"idx": 3, "name": "3", "path": "3_Dense", "type": "sentence_transformers.models.Dense"},
               {"idx": 4, "name": "4", "path": "4_Normalize", "type": "sentence_transformers.models.Normalize"}],
              open(os.path.join(d, "modules.json"), "w"))
    with common.Server(convert_dir(d, "embeddinggemma")) as srv:
        code, out = srv.post("/v1/embeddings", {"model": "m", "input": TEXTS})
        assert code == 200, out
        for i, t in enumerate(TEXTS):
            ids = tok_ids(tok, t)
            assert ids[0] == tok.convert_tokens_to_ids("<bos>") and ids[-1] == tok.convert_tokens_to_ids("<eos>")
            h = forward(model, ids).mean(0)
            with torch.no_grad():
                for lin in dense:
                    h = lin(h)
            common.check(f"gemma3 bidirectional [{i}] ({len(ids)} tokens)", out["data"][i]["embedding"], normalized(h))
        assert len(tok_ids(tok, TEXTS[2])) > 8, "the sliding window must be exercised"


# ------------------------------------------------------------------ Llama bidirectional with llama3 rope scaling
def llama_bidirectional():
    import transformers.models.llama.modeling_llama as ml

    d = os.path.join(work, "llama_bidir")
    common.seed(13)
    n = common.write_bpe_tokenizer(d, specials=("<|begin_of_text|>", "<|end_of_text|>"),
                                   single_template="<|begin_of_text|> $A")
    tok = common.load_fast_tokenizer(d, pad="<|end_of_text|>")
    cfg = LlamaConfig(vocab_size=n, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      rope_theta=500000.0, tie_word_embeddings=False,
                      rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                    "high_freq_factor": 4.0, "original_max_position_embeddings": 64})
    cfg.architectures = ["LlamaBidirectionalModel"]
    cfg.pooling = "avg"
    model = LlamaModel(cfg)
    randomize(model)
    model.save_pretrained(d)
    c = json.load(open(os.path.join(d, "config.json")))
    c["pooling"] = "avg"
    c["architectures"] = ["LlamaBidirectionalModel"]
    json.dump(c, open(os.path.join(d, "config.json"), "w"))
    model = LlamaModel.from_pretrained(d, attn_implementation="eager")
    model.eval()
    causal = {i: normalized(forward(model, tok_ids(tok, t)).mean(0)) for i, t in enumerate(TEXTS)}
    original = ml.create_causal_mask
    ml.create_causal_mask = lambda **kwargs: None  # no mask: bidirectional eager attention
    try:
        want = {i: normalized(forward(model, tok_ids(tok, t)).mean(0)) for i, t in enumerate(TEXTS)}
    finally:
        ml.create_causal_mask = original
    assert common.max_rel(causal[2], want[2]) > 1e-2, "bidirectional reference must differ from causal"
    with common.Server(convert_dir(d, "llama_bidir")) as srv:
        code, out = srv.post("/v1/embeddings", {"model": "m", "input": TEXTS})
        assert code == 200, out
        for i in range(len(TEXTS)):
            common.check(f"llama bidirectional[{i}]", out["data"][i]["embedding"], want[i])


# ------------------------------------------------------------------ sequence classification (reward model)
def sequence_classification():
    d = os.path.join(work, "reward3")
    common.seed(14)
    n = common.write_bpe_tokenizer(d, specials=("<|endoftext|>",))
    tok = common.load_fast_tokenizer(d)
    labels = ["bad", "ok", "good"]
    cfg = Qwen2Config(vocab_size=n, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      num_labels=3, id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)},
                      pad_token_id=n - 1)
    model = Qwen2ForSequenceClassification(cfg)
    randomize(model)
    model.save_pretrained(d)
    with common.Server(convert_dir(d, "reward3")) as srv:
        code, out = srv.post("/classify", {"model": "m", "input": TEXTS})
        assert code == 200, out
        for i, t in enumerate(TEXTS):
            ids = tok_ids(tok, t, special=False)
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([ids])).logits[0]
            common.check(f"qwen2 seq-cls probs[{i}]", out["data"][i]["probs"], torch.softmax(logits, -1).tolist())
            common.check(f"qwen2 seq-cls logits[{i}]", out["data"][i]["logits"], logits.tolist())
            assert out["data"][i]["label"] == labels[int(logits.argmax())]
    d = os.path.join(work, "reward1")
    n = common.write_bpe_tokenizer(d, specials=("<|endoftext|>",))
    cfg = Qwen3Config(vocab_size=n, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=128,
                      num_labels=1, pad_token_id=n - 1)
    model = Qwen3ForSequenceClassification(cfg)
    randomize(model)
    model.save_pretrained(d)
    with common.Server(convert_dir(d, "reward1")) as srv:
        code, out = srv.post("/classify", {"model": "m", "input": TEXTS[:2]})
        assert code == 200, out
        for i, t in enumerate(TEXTS[:2]):
            with torch.no_grad():
                logit = model(input_ids=torch.tensor([tok_ids(tok, t, special=False)])).logits[0]
            common.check_scalar(f"qwen3 reward logit[{i}]", out["data"][i]["logits"][0], float(logit[0]))


# ------------------------------------------------------------------ Qwen3-Reranker
def qwen3_reranker():
    d = os.path.join(work, "Qwen3-Reranker-tiny")
    common.seed(15)
    specials = ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>")
    n = common.write_bpe_tokenizer(d, specials=specials, extra_tokens=("yes", "no"))
    tok = common.load_fast_tokenizer(d)
    cfg = Qwen3Config(vocab_size=n + 8, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
                      tie_word_embeddings=False)
    model = Qwen3ForCausalLM(cfg)
    randomize(model)
    model.save_pretrained(d)
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    prefix = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
              "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def reference(query, doc, instruction="Given a web search query, retrieve relevant passages that answer the query"):
        text = f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}{suffix}"
        ids = tok(text, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids])).logits[0, -1]
        return float(torch.softmax(torch.stack([logits[no], logits[yes]]), 0)[1])

    docs = ["paris is the capital of france", "the cat sat on the mat", "berlin is the capital of germany"]
    query = "what is the capital of france ?"
    with common.Server(convert_dir(d, "reranker")) as srv:
        code, out = srv.post("/v1/rerank", {"model": "m", "query": query, "documents": docs})
        assert code == 200, out
        by_index = {r["index"]: r["relevance_score"] for r in out["results"]}
        for i, doc in enumerate(docs):
            common.check_scalar(f"reranker score[{i}]", by_index[i], reference(query, doc))
        ranked = [r["index"] for r in out["results"]]
        assert ranked == sorted(range(3), key=lambda i: -by_index[i]), ranked
        code, out = srv.post("/v1/rerank", {"query": query, "documents": docs[:1], "instruction": "Judge relevance"})
        assert code == 200, out
        common.check_scalar("reranker custom instruction", out["results"][0]["relevance_score"],
                            reference(query, docs[0], "Judge relevance"))
        code, out = srv.post("/v1/score", {"text_1": query, "text_2": docs})
        assert code == 200, out
        common.check_scalar("reranker /v1/score", out["data"][1]["score"], reference(query, docs[1]))


qwen3_embedder()
gemma3_embedding()
llama_bidirectional()
sequence_classification()
qwen3_reranker()
common.finish()
