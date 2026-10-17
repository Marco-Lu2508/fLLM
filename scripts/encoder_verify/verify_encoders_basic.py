#!/usr/bin/env python3
"""Regression check of the established encoder families against transformers: BERT (CLS / mean /
last pooling, matryoshka), a BERT cross-encoder (`/v1/rerank`, `/v1/score`, `/classify`), RoBERTa
(BPE tokenizer, position offset) and ModernBERT (sliding-window local layers).

usage: verify_encoders_basic.py [workdir]
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import common  # noqa: E402

import torch  # noqa: E402
from transformers import (BertConfig, BertForSequenceClassification, BertModel, ModernBertConfig,  # noqa: E402
                          ModernBertModel, RobertaConfig, RobertaModel)

work = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="basic_")
TEXTS = ["the cat sat on the mat", "hello world", "paris is the capital of france and berlin is the capital of germany"]


def randomize(model, std=0.3):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "LayerNorm.weight" in n or n.endswith("norm.weight"):
                p.copy_(1 + 0.1 * torch.randn_like(p))
            elif p.dim() > 1:
                p.normal_(0, std)
            else:
                p.normal_(0, 0.1)
    model.eval()


def unit(v):
    return torch.nn.functional.normalize(v, dim=-1).tolist()


def hidden(model, ids):
    with torch.no_grad():
        return model(input_ids=torch.tensor([ids])).last_hidden_state[0]


def serve(d, tag):
    gguf = os.path.join(work, tag + ".gguf")
    print(common.convert(d, gguf))
    return common.Server(gguf)


def bert():
    d = os.path.join(work, "bert_tiny")
    common.seed(21)
    n = common.write_wordpiece_tokenizer(d)
    tok = common.load_tokenizer(d)
    cfg = BertConfig(vocab_size=common.pad_vocab(n), hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                     intermediate_size=64, max_position_embeddings=64)
    model = BertModel(cfg)
    randomize(model)
    model.save_pretrained(d)
    with serve(d, "bert") as srv:
        for pooling in ("cls", "mean", "last"):
            code, out = srv.post("/v1/embeddings", {"model": "m", "input": TEXTS, "pooling": pooling})
            assert code == 200, out
            for i, t in enumerate(TEXTS):
                h = hidden(model, tok(t)["input_ids"])
                ref = {"cls": h[0], "mean": h.mean(0), "last": h[-1]}[pooling]
                common.check(f"bert {pooling}[{i}]", out["data"][i]["embedding"], unit(ref))
        code, out = srv.post("/v1/embeddings", {"input": TEXTS[0], "pooling": "cls", "dimensions": 8})
        ref = unit(hidden(model, tok(TEXTS[0])["input_ids"])[0][:8])
        common.check("bert dimensions=8", out["data"][0]["embedding"], ref)
        ids = tok(TEXTS[1])["input_ids"]
        code, out = srv.post("/v1/embeddings", {"input": [ids, TEXTS[1]], "pooling": "cls"})
        common.check("bert token-id input equals text input", out["data"][0]["embedding"], out["data"][1]["embedding"])
        code, out = srv.post("/v1/embeddings", {"input": TEXTS[2], "truncate_prompt_tokens": 6, "pooling": "cls"})
        short = tok(TEXTS[2])["input_ids"][:5] + [tok.sep_token_id]
        common.check("bert truncate_prompt_tokens=6", out["data"][0]["embedding"], unit(hidden(model, short)[0]))
        code, out = srv.post("/v1/embeddings", {"input": "x " * 100})
        assert code == 400, (code, out)
        code, out = srv.post("/v1/embeddings", {"input": "x " * 100, "truncate_prompt_tokens": -1})
        assert code == 200, (code, out)


def cross_encoder():
    d = os.path.join(work, "cross_tiny")
    common.seed(22)
    n = common.write_wordpiece_tokenizer(d)
    tok = common.load_tokenizer(d)
    cfg = BertConfig(vocab_size=common.pad_vocab(n), hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                     intermediate_size=64, max_position_embeddings=64, num_labels=1)
    model = BertForSequenceClassification(cfg)
    randomize(model)
    model.save_pretrained(d)
    query = "capital of france"
    docs = [TEXTS[2], TEXTS[0], TEXTS[1]]
    want = []
    for doc in docs:
        enc = tok(query, doc, return_tensors="pt")
        with torch.no_grad():
            want.append(float(torch.sigmoid(model(input_ids=enc["input_ids"],
                                                  token_type_ids=enc["token_type_ids"]).logits[0, 0])))
    with serve(d, "cross") as srv:
        code, out = srv.post("/v1/rerank", {"model": "m", "query": query, "documents": docs})
        assert code == 200, out
        got = {r["index"]: r["relevance_score"] for r in out["results"]}
        for i in range(3):
            common.check_scalar(f"cross-encoder rerank[{i}]", got[i], want[i])
        code, out = srv.post("/v1/score", {"text_1": query, "text_2": docs})
        for i in range(3):
            common.check_scalar(f"cross-encoder score[{i}]", out["data"][i]["score"], want[i])


def roberta():
    d = os.path.join(work, "roberta_tiny")
    common.seed(23)
    n = common.write_bpe_tokenizer(d, specials=("<s>", "<pad>", "</s>", "<unk>", "<mask>"),
                                   single_template="<s> $A </s>")
    tok = common.load_fast_tokenizer(d, pad="<pad>")
    import json
    cfgj = {"tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>",
            "unk_token": "<unk>", "cls_token": "<s>", "sep_token": "</s>", "mask_token": "<mask>"}
    json.dump(cfgj, open(os.path.join(d, "tokenizer_config.json"), "w"))
    cfg = RobertaConfig(vocab_size=n, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                        intermediate_size=64, max_position_embeddings=130, pad_token_id=1, bos_token_id=0,
                        eos_token_id=2, type_vocab_size=1)
    model = RobertaModel(cfg, add_pooling_layer=False)
    randomize(model)
    model.save_pretrained(d)
    with serve(d, "roberta") as srv:
        code, out = srv.post("/v1/embeddings", {"model": "m", "input": TEXTS, "pooling": "mean"})
        assert code == 200, out
        for i, t in enumerate(TEXTS):
            ids = tok(t)["input_ids"]
            with torch.no_grad():
                h = model(input_ids=torch.tensor([ids])).last_hidden_state[0]
            common.check(f"roberta mean[{i}]", out["data"][i]["embedding"], unit(h.mean(0)))


def modernbert():
    d = os.path.join(work, "modernbert_tiny")
    common.seed(24)
    specials = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")
    n = common.write_bpe_tokenizer(d, specials=specials, single_template="[CLS] $A [SEP]")
    tok = common.load_fast_tokenizer(d, pad="[PAD]")
    import json
    json.dump({"tokenizer_class": "PreTrainedTokenizerFast", "cls_token": "[CLS]", "sep_token": "[SEP]",
               "pad_token": "[PAD]", "unk_token": "[UNK]", "mask_token": "[MASK]"},
              open(os.path.join(d, "tokenizer_config.json"), "w"))
    cfg = ModernBertConfig(vocab_size=n, hidden_size=32, intermediate_size=48, num_hidden_layers=3,
                           num_attention_heads=4, max_position_embeddings=128, local_attention=4,
                           global_attn_every_n_layers=2, pad_token_id=0, cls_token_id=2, sep_token_id=3,
                           bos_token_id=2, eos_token_id=3, attn_implementation="eager")
    model = ModernBertModel(cfg)
    randomize(model, 0.2)
    model.save_pretrained(d)
    with serve(d, "modernbert") as srv:
        for pooling in ("cls", "mean"):
            code, out = srv.post("/v1/embeddings", {"model": "m", "input": TEXTS, "pooling": pooling})
            assert code == 200, out
            for i, t in enumerate(TEXTS):
                ids = tok(t)["input_ids"]
                h = hidden(model, ids)
                common.check(f"modernbert {pooling}[{i}] ({len(ids)} tokens)", out["data"][i]["embedding"],
                             unit(h[0] if pooling == "cls" else h.mean(0)))


bert()
cross_encoder()
roberta()
modernbert()
common.finish()
