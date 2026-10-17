#!/usr/bin/env python3
"""Sparse (SPLADE, BGE-M3 lexical), multi-vector (BGE-M3 and original ColBERT) and token
classification heads against Python references built from `transformers` + torch.

usage: verify_heads.py [workdir]
"""
import json
import os
import string
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import common  # noqa: E402

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import (BertConfig, BertForMaskedLM, BertForTokenClassification,  # noqa: E402
                          BertModel)

work = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="heads_")
TEXTS = ["the cat sat on the mat", "hello world, paris is the capital of france", "dog"]


def small_config(nvocab, **extra):
    return BertConfig(vocab_size=common.pad_vocab(nvocab), hidden_size=32, num_hidden_layers=2,
                      num_attention_heads=4, intermediate_size=64, max_position_embeddings=128, **extra)


def randomize(model, std=0.3):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "LayerNorm.weight" in n:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            elif p.dim() > 1:
                p.normal_(0, std)
            else:
                p.normal_(0, 0.1)
    model.eval()


def hidden(model, tok, ids):
    with torch.no_grad():
        return model(input_ids=torch.tensor([ids])).last_hidden_state[0]


def prepare(tag):
    directory = os.path.join(work, tag)
    n = common.write_wordpiece_tokenizer(directory)
    return directory, n, common.load_tokenizer(directory)


def serve(directory, tag):
    gguf = os.path.join(work, tag + ".gguf")
    print(common.convert(directory, gguf))
    return common.Server(gguf)


# ---------------------------------------------------------------- SPLADE
def splade():
    common.seed(5)
    d, n, tok = prepare("splade_tiny")
    cfg = small_config(n)
    model = BertForMaskedLM(cfg)
    randomize(model)
    model.save_pretrained(d)
    want = []
    for t in TEXTS:
        ids = tok(t)["input_ids"]
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids])).logits[0]
        w = torch.log1p(torch.relu(logits)).max(0).values
        want.append({i: float(v) for i, v in enumerate(w) if v > 0})
    with serve(d, "splade") as srv:
        code, out = srv.post("/v1/embeddings", {"model": "m", "input": TEXTS})
        assert code == 200, out
        assert out["data"][0]["embedding"] == [], "sparse-only model must not return a dense vector"
        for i, row in enumerate(out["data"]):
            common.check_sparse(f"splade[{i}]", row["sparse"], want[i])
        code, out = srv.post("/pooling", {"input": TEXTS[:1], "task": "sparse"})
        assert code == 200, out
        common.check_sparse("splade /pooling", out["data"][0]["data"], want[0])
        code, out = srv.post("/v1/score", {"text_1": TEXTS[0], "text_2": TEXTS[:2]})
        assert code == 200, out
        for j in range(2):
            ref = sum(want[0][k] * want[j].get(k, 0.0) for k in want[0])
            common.check_scalar(f"splade score[{j}]", out["data"][j]["score"], ref)
        code, out = srv.post("/v1/embeddings", {"input": "x", "multi_vector": True})
        assert code == 400, (code, out)


# ---------------------------------------------------------------- token classification
def token_classification():
    common.seed(6)
    d, n, tok = prepare("ner_tiny")
    labels = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]
    cfg = small_config(n, num_labels=5, id2label=dict(enumerate(labels)),
                       label2id={l: i for i, l in enumerate(labels)})
    model = BertForTokenClassification(cfg)
    randomize(model)
    model.save_pretrained(d)
    with serve(d, "ner") as srv:
        code, out = srv.post("/classify", {"model": "m", "input": TEXTS})
        assert code == 200, out
        for i, t in enumerate(TEXTS):
            ids = tok(t)["input_ids"]
            with torch.no_grad():
                probs = torch.softmax(model(input_ids=torch.tensor([ids])).logits[0], -1)
            row = out["data"][i]
            assert row["tokens"] == ids, (row["tokens"], ids)
            common.check_matrix(f"ner[{i}]", row["probs"], probs.tolist())
            assert row["labels"] == [labels[j] for j in probs.argmax(-1).tolist()], row["labels"]
        code, out = srv.post("/pooling", {"input": TEXTS[:1], "task": "token_classify"})
        assert code == 200 and len(out["data"][0]["data"]) == len(tok(TEXTS[0])["input_ids"]), out


# ---------------------------------------------------------------- BGE-M3 style heads
def bge_m3_heads():
    common.seed(7)
    d, n, tok = prepare("bge_m3_tiny")
    cfg = small_config(n)
    model = BertModel(cfg, add_pooling_layer=False)
    randomize(model)
    model.save_pretrained(d)
    colbert = torch.nn.Linear(32, 24)
    sparse = torch.nn.Linear(32, 1)
    with torch.no_grad():
        colbert.weight.normal_(0, 0.5)
        colbert.bias.normal_(0, 0.2)
        sparse.weight.normal_(0, 1.0)
        sparse.bias.zero_()
        # centre the logits so that roughly half of the tokens get a positive weight
        logits = torch.cat([sparse(hidden(model, tok, tok(t)["input_ids"])).flatten() for t in TEXTS])
        sparse.bias.fill_(-float(logits.median()))
    torch.save(colbert.state_dict(), os.path.join(d, "colbert_linear.pt"))
    torch.save(sparse.state_dict(), os.path.join(d, "sparse_linear.pt"))
    unused = {tok.cls_token_id, tok.sep_token_id, tok.pad_token_id, tok.unk_token_id}
    dense_ref, sparse_ref, multi_ref = [], [], []
    for t in TEXTS:
        ids = tok(t)["input_ids"]
        h = hidden(model, tok, ids)
        dense_ref.append(torch.nn.functional.normalize(h[0], dim=-1).tolist())
        with torch.no_grad():
            tw = torch.relu(sparse(h)).squeeze(-1)
            cv = torch.nn.functional.normalize(colbert(h[1:]), dim=-1)
        best = {}
        for i, w in zip(ids, tw.tolist()):
            if i not in unused and w > 0:
                best[i] = max(best.get(i, 0.0), w)
        sparse_ref.append(best)
        multi_ref.append(cv.tolist())
    with serve(d, "bgem3") as srv:
        code, out = srv.post("/v1/embeddings", {"model": "m", "input": TEXTS, "sparse": True, "multi_vector": True})
        assert code == 200, out
        for i, row in enumerate(out["data"]):
            common.check(f"bge-m3 dense[{i}]", row["embedding"], dense_ref[i])
            common.check_sparse(f"bge-m3 sparse[{i}]", row["sparse"], sparse_ref[i])
            common.check_matrix(f"bge-m3 colbert[{i}]", row["multi_vector"], multi_ref[i])
        # late-interaction score, FlagEmbedding style: sum of MaxSim / number of query vectors
        for q, doc in [(0, 1), (1, 1)]:
            qv, dv = torch.tensor(multi_ref[q]), torch.tensor(multi_ref[doc])
            ref_multi = float((qv @ dv.T).max(1).values.sum() / qv.shape[0])
            ref_sparse = sum(w * sparse_ref[doc].get(k, 0.0) for k, w in sparse_ref[q].items())
            ref_dense = float(torch.tensor(dense_ref[q]) @ torch.tensor(dense_ref[doc]))
            for mode, ref in [("multi_vector", ref_multi), ("sparse", ref_sparse), ("dense", ref_dense),
                              ("combined", 0.4 * ref_dense + 0.2 * ref_sparse + 0.4 * ref_multi)]:
                code, out = srv.post("/v1/score", {"text_1": TEXTS[q], "text_2": TEXTS[doc], "score_type": mode})
                assert code == 200, out
                common.check_scalar(f"bge-m3 score {mode} ({q},{doc})", out["data"][0]["score"], ref)
        q = 0
        code, out = srv.post("/v1/rerank", {"query": TEXTS[q], "documents": TEXTS, "score_type": "multi_vector"})
        assert code == 200, out
        assert out["results"][0]["index"] == q, out
        code, out = srv.post("/pooling", {"input": TEXTS[:1], "task": "token_embed"})
        common.check_matrix("bge-m3 /pooling token_embed", out["data"][0]["data"], multi_ref[0])
        code, out = srv.post("/v1/embeddings", {"input": TEXTS[:1], "multi_vector": True, "encoding_format": "base64"})
        assert out["data"][0]["multi_vector_shape"] == [len(multi_ref[0]), 24], out["data"][0]


# ---------------------------------------------------------------- original ColBERT (HF_ColBERT)
def colbert():
    common.seed(8)
    d, n, tok = prepare("colbert_tiny")
    cfg = small_config(n)
    cfg.architectures = ["HF_ColBERT"]
    model = BertModel(cfg)
    randomize(model)
    linear = torch.nn.Linear(32, 16, bias=False)
    with torch.no_grad():
        linear.weight.normal_(0, 0.5)
    state = {"bert." + k: v.contiguous() for k, v in model.state_dict().items()}
    state["linear.weight"] = linear.weight.detach().contiguous()
    save_file(state, os.path.join(d, "model.safetensors"))
    cfg.save_pretrained(d)
    vocab = tok.get_vocab()
    q_marker, d_marker, mask = vocab["[unused0]"], vocab["[unused1]"], tok.mask_token_id
    skip = {vocab[c] for c in string.punctuation if c in vocab}

    def reference(text, query):
        ids = tok(text)["input_ids"]
        body = ids[1:-1]
        limit = 32 if query else 180
        body = body[:limit - 3]
        full = [ids[0], q_marker if query else d_marker] + body + [ids[-1]]
        if query:
            full += [mask] * (limit - len(full))
        h = hidden(model, tok, full)
        with torch.no_grad():
            v = torch.nn.functional.normalize(linear(h), dim=-1)
        keep = [i for i, t in enumerate(full) if query or t not in skip]
        return v[keep].tolist()

    with serve(d, "colbert") as srv:
        for kind in ("query", "document"):
            code, out = srv.post("/v1/embeddings", {"model": "m", "input": TEXTS, "input_type": kind})
            assert code == 200, out
            for i, row in enumerate(out["data"]):
                common.check_matrix(f"colbert {kind}[{i}]", row["multi_vector"], reference(TEXTS[i], kind == "query"))
        q = reference(TEXTS[0], True)
        for j in range(2):
            dv = reference(TEXTS[j], False)
            ref = float((torch.tensor(q) @ torch.tensor(dv).T).max(1).values.sum())
            code, out = srv.post("/v1/score", {"text_1": TEXTS[0], "text_2": TEXTS[j]})
            assert code == 200, out
            common.check_scalar(f"colbert score[{j}]", out["data"][0]["score"], ref)


splade()
token_classification()
bge_m3_heads()
colbert()
common.finish()
