"""Shared helpers for the encoder verification scripts.

Each script builds a tiny random checkpoint with `transformers` (CPU torch),
converts it with `fllm-rs convert`, serves it with `fllm-rs serve` and compares
the HTTP results with the Python reference. Nothing is downloaded.

Environment:
  FLLM_BIN  path to the fllm-rs binary (default: target-encoders/release/fllm-rs
            next to the repository root)
"""
import base64
import json
import os
import random
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BIN = os.environ.get("FLLM_BIN", os.path.join(ROOT, "target-encoders", "release", "fllm-rs"))


def seed(n=0):
    random.seed(n)
    import numpy as np
    import torch

    np.random.seed(n)
    torch.manual_seed(n)


def write_wordpiece_tokenizer(directory, extra_words=()):
    """A tiny WordPiece tokenizer (tokenizer.json, vocab.txt, tokenizer_config.json)."""
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors

    os.makedirs(directory, exist_ok=True)
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[unused0]", "[unused1]"]
    letters = list("abcdefghijklmnopqrstuvwxyz")
    words = ["the", "cat", "sat", "on", "mat", "dog", "a", "is", "red", "hello", "world",
             "query", "document", "paris", "capital", "france", "berlin", "germany", "of"]
    words += list(extra_words)
    vocab = specials + letters + ["##" + c for c in letters] + list(".,?!;:") + words
    seen, uniq = set(), []
    for v in vocab:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    with open(os.path.join(directory, "vocab.txt"), "w") as f:
        f.write("\n".join(uniq) + "\n")
    tok = Tokenizer(models.WordPiece({v: i for i, v in enumerate(uniq)}, unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.decoder = decoders.WordPiece()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", uniq.index("[CLS]")), ("[SEP]", uniq.index("[SEP]"))])
    tok.add_special_tokens(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    tok.save(os.path.join(directory, "tokenizer.json"))
    with open(os.path.join(directory, "tokenizer_config.json"), "w") as f:
        json.dump({"tokenizer_class": "BertTokenizer", "do_lower_case": True, "cls_token": "[CLS]",
                   "sep_token": "[SEP]", "pad_token": "[PAD]", "unk_token": "[UNK]", "mask_token": "[MASK]"}, f)
    return len(uniq)


def pad_vocab(n, multiple=8):
    return (n + multiple - 1) // multiple * multiple


def convert(ckpt_dir, gguf, outtype="f32"):
    r = subprocess.run([BIN, "convert", ckpt_dir, gguf, "--outtype", outtype, "--overwrite"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("convert failed: " + r.stdout + r.stderr)
    return r.stdout + r.stderr


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Server:
    def __init__(self, gguf, extra=(), name="m"):
        self.port = free_port()
        self.proc = subprocess.Popen(
            [BIN, "serve", "--model", gguf, "--embeddings", "--device", "cpu", "--port", str(self.port),
             "--model-name", name, "--threads", "2", *extra],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.name = name
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1).read()
                return
            except Exception:
                if self.proc.poll() is not None:
                    raise RuntimeError("server exited: " + self.proc.stdout.read())
                time.sleep(0.2)
        raise RuntimeError("server did not start")

    def post(self, path, body, expect=200):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", json.dumps(body).encode(),
                                     {"content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(5)
        except Exception:
            self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def cosine(a, b):
    import math
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def max_rel(a, b, floor=1e-3):
    """Largest |a-b| / max(|b|, floor-scale) with scale = max |b| so tiny components do not dominate."""
    scale = max(max(abs(x) for x in b), floor)
    return max(abs(x - y) for x, y in zip(a, b)) / scale


def check(name, got, want, cos_min=0.9999, rel_max=1e-3):
    c = cosine(got, want)
    r = max_rel(got, want)
    ok = c >= cos_min and r <= rel_max
    print(f"{'PASS' if ok else 'FAIL'} {name}: cosine={c:.8f} max_err/max|ref|={r:.2e}")
    if not ok:
        worst = max(range(len(want)), key=lambda i: abs(got[i] - want[i]))
        print(f"   worst element {worst}: got {got[worst]} want {want[worst]}")
        global FAILED
        FAILED = True
    return ok


FAILED = False


def finish():
    print("RESULT:", "FAIL" if FAILED else "PASS")
    sys.exit(1 if FAILED else 0)


def check_sparse(name, got, want, rel=1e-3):
    """got: {"id": weight} from the server; want: {int id: weight} from the reference."""
    got = {int(k): v for k, v in got.items()}
    ok = True
    for k, w in want.items():
        if w > 1e-6 and (k not in got or abs(got[k] - w) > rel * max(abs(w), 1.0)):
            ok = False
            print(f"   token {k}: got {got.get(k)} want {w}")
    for k, g in got.items():
        if k not in want or want[k] <= 1e-6:
            if g > 1e-6:
                ok = False
                print(f"   unexpected token {k}: {g}")
    print(f"{'PASS' if ok else 'FAIL'} {name}: {len(want)} reference entries, {len(got)} returned")
    if not ok:
        global FAILED
        FAILED = True
    return ok


def check_matrix(name, got, want, cos_min=0.9999, rel_max=1e-3):
    """Row-wise comparison of two matrices; prints one summary line."""
    global FAILED
    if len(got) != len(want):
        print(f"FAIL {name}: {len(got)} rows returned, {len(want)} expected")
        FAILED = True
        return False
    worst_cos, worst_rel = 1.0, 0.0
    for g, w in zip(got, want):
        worst_cos = min(worst_cos, cosine(g, w))
        worst_rel = max(worst_rel, max_rel(g, w))
    ok = worst_cos >= cos_min and worst_rel <= rel_max
    print(f"{'PASS' if ok else 'FAIL'} {name}: {len(want)} rows, min cosine={worst_cos:.8f} "
          f"max err/max|ref|={worst_rel:.2e}")
    FAILED |= not ok
    return ok


def check_scalar(name, got, want, rel=1e-4):
    ok = abs(got - want) <= rel * max(abs(want), 1e-3)
    print(f"{'PASS' if ok else 'FAIL'} {name}: got {got:.7f} want {want:.7f}")
    if not ok:
        global FAILED
        FAILED = True
    return ok


def load_tokenizer(directory):
    """The tokenizer.json written by write_wordpiece_tokenizer, loaded without any model config."""
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(directory, "tokenizer.json"), unk_token="[UNK]", sep_token="[SEP]",
        pad_token="[PAD]", cls_token="[CLS]", mask_token="[MASK]",
        model_input_names=["input_ids", "token_type_ids", "attention_mask"])


CORPUS = [
    "the cat sat on the mat", "hello world", "paris is the capital of france", "berlin is the capital of germany",
    "a dog is red", "what is the capital of france ?", "query document relevance yes no", "judge the document",
    "embedding models map text to vectors", "the quick brown fox jumps over the lazy dog",
] * 4


def write_bpe_tokenizer(directory, specials=("<|endoftext|>",), single_template=None, extra_tokens=(), vocab_size=320):
    """A tiny byte-level BPE tokenizer.json (+ tokenizer_config.json) for decoder-style models.

    single_template: e.g. "$A <|endoftext|>" to add a post-processor that appends or prepends special tokens.
    Returns the number of tokens.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

    os.makedirs(directory, exist_ok=True)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=list(specials),
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(CORPUS, trainer)
    if extra_tokens:
        tok.add_tokens(list(extra_tokens))
    if single_template:
        ids = [(s, tok.token_to_id(s)) for s in specials if s in single_template]
        tok.post_processor = processors.TemplateProcessing(single=single_template, special_tokens=ids)
    tok.save(os.path.join(directory, "tokenizer.json"))
    with open(os.path.join(directory, "tokenizer_config.json"), "w") as f:
        json.dump({"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": specials[0], "pad_token": specials[0]}, f)
    return tok.get_vocab_size()


def load_fast_tokenizer(directory, pad="<|endoftext|>"):
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(tokenizer_file=os.path.join(directory, "tokenizer.json"), pad_token=pad)
