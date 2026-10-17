#!/usr/bin/env python3
"""GTE "new" (NewModel remote code) tiny random model against the remote-code reference.

The reference modeling code is not stored in this repository. Fetch `modeling.py` and
`configuration.py` of the Hugging Face repository Alibaba-NLP/new-impl (Apache-2.0) into a
directory and pass it as GTE_REMOTE_CODE_DIR.

usage: GTE_REMOTE_CODE_DIR=/path/to/new-impl verify_gte.py [workdir]
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import common  # noqa: E402

import torch  # noqa: E402
from transformers import AutoConfig, AutoModel  # noqa: E402

code = os.environ.get("GTE_REMOTE_CODE_DIR")
if not code or not os.path.isfile(os.path.join(code, "modeling.py")):
    sys.exit("set GTE_REMOTE_CODE_DIR to a directory holding modeling.py and configuration.py")
work = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="gte_")
texts = ["the cat sat on the mat", "hello world", "paris is the capital of france . berlin is the capital of germany"]


def run(tag, scaling):
    hf = os.path.join(work, tag)
    common.seed(3)
    nvocab = common.write_wordpiece_tokenizer(hf)
    for f in ("modeling.py", "configuration.py"):
        shutil.copy(os.path.join(code, f), hf)
    cfg = {
        "model_type": "new", "architectures": ["NewModel"],
        "auto_map": {"AutoConfig": "configuration.NewConfig", "AutoModel": "modeling.NewModel"},
        "vocab_size": common.pad_vocab(nvocab), "hidden_size": 32, "num_hidden_layers": 2,
        "num_attention_heads": 4, "intermediate_size": 48, "hidden_act": "gelu",
        "max_position_embeddings": 128, "type_vocab_size": 2, "layer_norm_eps": 1e-12,
        "position_embedding_type": "rope", "rope_theta": 500.0, "rope_scaling": scaling,
        "pack_qkv": True, "unpad_inputs": False, "use_memory_efficient_attention": False,
        "pad_token_id": 0, "logn_attention_scale": False, "logn_attention_clip1": False, "layer_norm_type": "layer_norm",
    }
    json.dump(cfg, open(os.path.join(hf, "config.json"), "w"))
    config = AutoConfig.from_pretrained(hf, trust_remote_code=True)
    model = AutoModel.from_config(config, trust_remote_code=True)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.dim() > 1:
                p.normal_(0, 0.25)
            elif n.endswith("LayerNorm.weight") or n.endswith("_ln.weight"):
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(0, 0.1)
    model.eval()
    model.save_pretrained(hf)
    # sentence-transformers layout: CLS pooling then Normalize
    os.makedirs(os.path.join(hf, "1_Pooling"), exist_ok=True)
    json.dump({"word_embedding_dimension": 32, "pooling_mode_cls_token": True,
               "pooling_mode_mean_tokens": False}, open(os.path.join(hf, "1_Pooling", "config.json"), "w"))
    json.dump([{"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
               {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
               {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"}],
              open(os.path.join(hf, "modules.json"), "w"))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf)
    want = []
    for t in texts:
        enc = tok(t, return_tensors="pt")
        with torch.no_grad():
            h = model(**enc).last_hidden_state[0, 0]
        want.append((h / h.norm()).tolist())
    gguf = os.path.join(work, tag + ".gguf")
    print(common.convert(hf, gguf))
    with common.Server(gguf) as srv:
        code_, out = srv.post("/v1/embeddings", {"model": "m", "input": texts})
        assert code_ == 200, out
        for i, row in enumerate(out["data"]):
            common.check(f"{tag}[{i}]", row["embedding"], want[i])


run("gte_plain", None)
run("gte_ntk", {"type": "ntk", "factor": 2.0})
common.finish()
