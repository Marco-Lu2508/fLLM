#!/usr/bin/env python3
"""BERT + sentence-transformers pipeline (every pooling mode, two Dense layers, Normalize)
against sentence-transformers itself.

usage: verify_st_pipeline.py [workdir]
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import common  # noqa: E402

import torch  # noqa: E402
from sentence_transformers import SentenceTransformer, models  # noqa: E402
from transformers import BertConfig, BertModel  # noqa: E402

work = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="st_pipe_")
hf = os.path.join(work, "hf")
common.seed(1)
nvocab = common.write_wordpiece_tokenizer(hf)
cfg = BertConfig(vocab_size=common.pad_vocab(nvocab), hidden_size=32, num_hidden_layers=2,
                 num_attention_heads=4, intermediate_size=64, max_position_embeddings=64)
BertModel(cfg).save_pretrained(hf)

word = models.Transformer(hf)
dense1 = models.Dense(32, 48, activation_function=torch.nn.Tanh())
dense2 = models.Dense(48, 16, activation_function=torch.nn.Identity())
norm = models.Normalize()
texts = ["the cat sat on the mat", "hello world", "paris is the capital of france . berlin is the capital of germany"]

refs = {}
for mode in ["mean", "cls", "max", "lasttoken", "weightedmean", "mean_sqrt_len_tokens"]:
    pool = models.Pooling(32, pooling_mode=mode)
    st = SentenceTransformer(modules=[word, pool, dense1, dense2, norm])
    st.eval()
    refs[mode] = st.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
    if mode == "mean":
        st.save(os.path.join(work, "st"))
        # the first saved model carries mean pooling; other modes use the request override

print(common.convert(os.path.join(work, "st"), os.path.join(work, "m.gguf")))
names = {"mean": "mean", "cls": "cls", "max": "max", "lasttoken": "last",
         "weightedmean": "weightedmean", "mean_sqrt_len_tokens": "mean_sqrt_len"}
with common.Server(os.path.join(work, "m.gguf")) as srv:
    for mode, ref in refs.items():
        code, out = srv.post("/v1/embeddings", {"model": "m", "input": texts, "pooling": names[mode]})
        assert code == 200, out
        for i, row in enumerate(out["data"]):
            common.check(f"{mode}[{i}]", row["embedding"], ref[i].tolist())
    # matryoshka truncation: first 8 components, renormalised
    code, out = srv.post("/v1/embeddings", {"model": "m", "input": texts[:1], "dimensions": 8})
    want = torch.tensor(refs["mean"][0][:8])
    want = (want / want.norm()).tolist()
    common.check("dimensions=8", out["data"][0]["embedding"], want)
    # normalize=false leaves the Dense output as is
    code, out = srv.post("/v1/embeddings", {"model": "m", "input": texts[:1], "normalize": False})
    st = SentenceTransformer(modules=[word, models.Pooling(32, pooling_mode="mean"), dense1, dense2])
    common.check("normalize=false", out["data"][0]["embedding"], st.encode(texts[:1])[0].tolist())
    # bad pooling is rejected
    code, out = srv.post("/v1/embeddings", {"input": "x", "pooling": "nope"})
    assert code == 400, (code, out)
common.finish()
