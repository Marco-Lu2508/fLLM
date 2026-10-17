#!/usr/bin/env python3
"""Jina embeddings v3 (XLM-RoBERTa flash implementation with rotary positions and per-task LoRA
adapters) against the remote-code reference.

The reference code is not stored in this repository. Fetch the files of the Hugging Face repository
jinaai/xlm-roberta-flash-implementation (modeling_lora.py, modeling_xlm_roberta.py, rotary.py, mha.py,
mlp.py, block.py, embedding.py, configuration_xlm_roberta.py, stochastic_depth.py, xlm_padding.py) into
a directory and pass it as JINA_REMOTE_CODE_DIR. Needs transformers 4.x (see docs).

usage: JINA_REMOTE_CODE_DIR=/path verify_jina.py [workdir]
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import common  # noqa: E402

import torch  # noqa: E402

code = os.environ.get("JINA_REMOTE_CODE_DIR")
if not code or not os.path.isfile(os.path.join(code, "modeling_lora.py")):
    sys.exit("set JINA_REMOTE_CODE_DIR to a directory with the xlm-roberta-flash-implementation files")
work = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="jina_")
TEXTS = ["the cat sat on the mat", "hello world", "paris is the capital of france and berlin is the capital of germany"]
TASKS = ["retrieval.query", "retrieval.passage", "separation"]
INSTRUCTIONS = {"retrieval.query": "query: ", "retrieval.passage": "passage: ", "separation": ""}

d = os.path.join(work, "jina_v3_tiny")
common.seed(31)
n = common.write_bpe_tokenizer(d, specials=("<s>", "<pad>", "</s>", "<unk>", "<mask>"), single_template="<s> $A </s>")
json.dump({"tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>",
           "unk_token": "<unk>", "cls_token": "<s>", "sep_token": "</s>", "mask_token": "<mask>"},
          open(os.path.join(d, "tokenizer_config.json"), "w"))
tok = common.load_fast_tokenizer(d, pad="<pad>")
for f in os.listdir(code):
    if f.endswith(".py"):
        shutil.copy(os.path.join(code, f), d)
cfg = {
    "architectures": ["XLMRobertaModel"],
    "auto_map": {"AutoConfig": "configuration_xlm_roberta.XLMRobertaFlashConfig",
                 "AutoModel": "modeling_lora.XLMRobertaLoRA"},
    "vocab_size": n, "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4, "intermediate_size": 64,
    "hidden_act": "gelu", "max_position_embeddings": 130, "type_vocab_size": 1, "layer_norm_eps": 1e-5,
    "pad_token_id": 1, "bos_token_id": 0, "eos_token_id": 2, "position_embedding_type": "rotary",
    "rotary_emb_base": 20000.0, "use_flash_attn": False, "load_trained_adapters": True,
    "lora_adaptations": TASKS, "lora_rank": 4, "lora_alpha": 1, "lora_dropout_p": 0.0,
    "lora_main_params_trainable": False, "task_instructions": INSTRUCTIONS, "matryoshka_dimensions": [8, 16, 32],
}
json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
# import the reference as a package instead of going through trust_remote_code
open(os.path.join(d, "__init__.py"), "w").close()
sys.path.insert(0, work)
import importlib  # noqa: E402

pkg = os.path.basename(d)
configuration = importlib.import_module(pkg + ".configuration_xlm_roberta")
modeling = importlib.import_module(pkg + ".modeling_lora")
config = configuration.XLMRobertaFlashConfig(**{k: v for k, v in cfg.items() if k not in ("architectures", "auto_map")})
config._name_or_path = d  # the reference loads a tokenizer from here
model = modeling.XLMRobertaLoRA(config)
with torch.no_grad():
    for name, p in model.state_dict().items():
        if "lora" in name:
            p.normal_(0, 0.6)
        elif "norm" in name or "emb_ln" in name:
            if name.endswith("weight"):
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(0, 0.1)
        elif p.dim() > 1:
            p.normal_(0, 0.25)
        else:
            p.normal_(0, 0.1)
model.eval()
model.save_pretrained(d)
print(sorted(k for k in model.state_dict() if "layers.0.mixer.Wqkv" in k))

gguf = os.path.join(work, "jina_v3.gguf")
print(common.convert(d, gguf))


def reference(text, task):
    ids = tok((INSTRUCTIONS[task] if task else "") + text)["input_ids"]
    kwargs = {}
    if task:
        kwargs["adapter_mask"] = torch.tensor([TASKS.index(task)], dtype=torch.int32)
    with torch.no_grad():
        h = model.roberta(input_ids=torch.tensor([ids]), **kwargs).last_hidden_state[0]
    return torch.nn.functional.normalize(h.mean(0), dim=-1).tolist()


with common.Server(gguf) as srv:
    for task in [None] + TASKS:
        body = {"model": "m", "input": TEXTS}
        if task:
            body["task"] = task
        code_, out = srv.post("/v1/embeddings", body)
        assert code_ == 200, out
        for i, t in enumerate(TEXTS):
            common.check(f"jina v3 task={task} [{i}]", out["data"][i]["embedding"], reference(t, task))
    base = reference(TEXTS[0], None)
    lora = reference(TEXTS[0], "separation")
    assert common.max_rel(base, lora) > 1e-2, "the adapter must change the output"
    code_, out = srv.post("/v1/embeddings", {"input": TEXTS[0], "task": "nope"})
    assert code_ == 400 and "retrieval.query" in out["error"]["message"], out
    code_, out = srv.post("/v1/embeddings", {"input": TEXTS[0], "task": "separation", "dimensions": 8})
    want = torch.tensor(reference(TEXTS[0], "separation")[:8])
    common.check("jina v3 matryoshka dimensions=8", out["data"][0]["embedding"], (want / want.norm()).tolist())
common.finish()
