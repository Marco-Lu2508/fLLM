#!/usr/bin/env python3
"""Builds tiny tokenizers and their reference encodings under rust/engine/testdata/tokenizers/:

  spm_unigram_nfkc/     SentencePiece unigram, nmt_nfkc normalization (precompiled charsmap)
  spm_unigram_bytefb/   unigram with byte fallback, no dummy prefix, identity normalization
  spm_bpe_llama/        BPE with byte fallback, dummy prefix, user-defined symbols (Llama style)
  spm_bpe_plain/        BPE without byte fallback, remove_extra_whitespaces
  tiktoken_small/       tiktoken ranks (cl100k pattern) + special tokens, reference ids from `tiktoken`
  tekken_small/         tekken.json (v3, 20 named + filler special tokens), reference ids from `mistral_common`

Each directory has `golden.json` {"strings": [...], "ids": [[...]]}. Requires sentencepiece, tiktoken, mistral_common.
Usage: make_tokenizer_fixtures.py <output dir>"""
import base64
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer_corpus import TRAIN_TEXT, corpus  # noqa: E402

out = sys.argv[1]
strings = corpus(fuzz=120)


def write_golden(d, ids):
    json.dump({"strings": strings, "ids": ids}, open(os.path.join(d, "golden.json"), "w", encoding="utf-8"), ensure_ascii=False)


def train_spm(name, **kw):
    import sentencepiece as sp

    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    corpus_file = os.path.join(d, "train.txt")
    open(corpus_file, "w", encoding="utf-8").write("\n".join(TRAIN_TEXT))
    sp.SentencePieceTrainer.train(input=corpus_file, model_prefix=os.path.join(d, "spm"), minloglevel=2, **kw)
    os.remove(corpus_file)
    os.remove(os.path.join(d, "spm.vocab"))
    os.replace(os.path.join(d, "spm.model"), os.path.join(d, "tokenizer.model"))
    proc = sp.SentencePieceProcessor(model_file=os.path.join(d, "tokenizer.model"))
    control = [proc.id_to_piece(i) for i in range(proc.get_piece_size()) if proc.is_control(i) or proc.is_unknown(i)]
    global strings
    keep = [s for s in strings if not any(c in s for c in control)]
    ids = [proc.encode(s) for s in keep]
    json.dump({"strings": keep, "ids": ids}, open(os.path.join(d, "golden.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print(name, proc.get_piece_size(), "pieces,", len(keep), "strings", os.path.getsize(os.path.join(d, "tokenizer.model")), "bytes")


train_spm("spm_unigram_nfkc", model_type="unigram", vocab_size=400, normalization_rule_name="nmt_nfkc", character_coverage=1.0, hard_vocab_limit=False)
train_spm("spm_unigram_bytefb", model_type="unigram", vocab_size=500, byte_fallback=True, add_dummy_prefix=False, normalization_rule_name="identity",
          character_coverage=0.9995, hard_vocab_limit=False, remove_extra_whitespaces=False)
train_spm("spm_bpe_llama", model_type="bpe", vocab_size=500, byte_fallback=True, add_dummy_prefix=True, normalization_rule_name="identity",
          remove_extra_whitespaces=False, user_defined_symbols=["<|user|>", "<|assistant|>"], character_coverage=0.9995, split_digits=True, hard_vocab_limit=False)
train_spm("spm_bpe_plain", model_type="bpe", vocab_size=400, byte_fallback=False, add_dummy_prefix=True, normalization_rule_name="identity",
          remove_extra_whitespaces=True, character_coverage=1.0, hard_vocab_limit=False)

# ---- tiktoken
import tiktoken  # noqa: E402
from tiktoken._educational import SimpleBytePairEncoding  # noqa: E402

PAT = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
simple = SimpleBytePairEncoding.train("\n".join(TRAIN_TEXT), vocab_size=520, pat_str=PAT)
ranks = simple.mergeable_ranks
d = os.path.join(out, "tiktoken_small")
os.makedirs(d, exist_ok=True)
with open(os.path.join(d, "tiktoken.model"), "wb") as f:
    for tok, r in sorted(ranks.items(), key=lambda kv: kv[1]):
        f.write(base64.b64encode(tok) + b" " + str(r).encode() + b"\n")
base = len(ranks)
specials = {"<|endoftext|>": base, "<|im_start|>": base + 1, "<|im_end|>": base + 2}
json.dump({"added_tokens_decoder": {str(i): {"content": t, "special": True} for t, i in specials.items()}, "eos_token": "<|im_end|>", "tokenizer_class": "PreTrainedTokenizerFast"},
          open(os.path.join(d, "tokenizer_config.json"), "w"))
json.dump({"model_type": "llama"}, open(os.path.join(d, "config.json"), "w"))
open(os.path.join(d, "pat_str.txt"), "w").write(PAT)
enc = tiktoken.Encoding("small", pat_str=PAT, mergeable_ranks=ranks, special_tokens=specials)
write_golden(d, [enc.encode(s, allowed_special="all") for s in strings])
print("tiktoken_small", base, "ranks")

# ---- tekken
d = os.path.join(out, "tekken_small")
os.makedirs(d, exist_ok=True)
n_special = 24
vocab = [{"rank": r, "token_bytes": base64.b64encode(t).decode(), "token_str": t.decode("utf-8", "replace")} for t, r in sorted(ranks.items(), key=lambda kv: kv[1])]
doc = {"config": {"pattern": PAT, "num_vocab_tokens": len(vocab), "default_vocab_size": len(vocab) + n_special, "default_num_special_tokens": n_special, "version": "v3"}, "vocab": vocab}
json.dump(doc, open(os.path.join(d, "tekken.json"), "w"))
from mistral_common.tokens.tokenizers.tekken import Tekkenizer  # noqa: E402

t = Tekkenizer.from_file(os.path.join(d, "tekken.json"))
skip = ("[INST]", "[/INST]", "<s>", "</s>", "<unk>", "<pad>", "[IMG]", "[PREFIX]", "[MIDDLE]", "[SUFFIX]", "[TOOL_CALLS]", "[SYSTEM_PROMPT]", "[TOOL_CONTENT]", "[IMG_END]", "[IMG_BREAK]", "[AVAILABLE_TOOLS]", "[TOOL_RESULTS]", "[/AVAILABLE_TOOLS]", "[/TOOL_RESULTS]", "[/SYSTEM_PROMPT]")
keep = [s for s in strings if not any(x in s for x in skip)]
json.dump({"strings": keep, "ids": [t.encode(s, bos=False, eos=False) for s in keep],
           "specials": {t.id_to_piece(i): i for i in range(n_special)}}, open(os.path.join(d, "golden.json"), "w", encoding="utf-8"), ensure_ascii=False)
print("tekken_small", len(vocab), "tokens", n_special, "special")
