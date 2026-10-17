#!/usr/bin/env python3
"""Compare tokenizers converted by fLLM (tokenizer.json derived from SentencePiece / tiktoken / tekken files) with the
reference implementations on a corpus of tricky strings.

  verify_tokenizers.py spm      <model file> <converted tokenizer.json> [--no-special]
  verify_tokenizers.py tiktoken <rank file> <kimi|glm|qwen1|cl100k|regex> <converted tokenizer.json>
  verify_tokenizers.py tekken   <tekken.json> <converted tokenizer.json>

`converted tokenizer.json` is the output of `cargo run -p fllm-checkpoint --example tokenizer_json -- <dir>`.
Exit status 0 when every string encodes to the same ids and decodes to the same text."""
import base64
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer_corpus import corpus  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402


def report(name, bad, total):
    print(f"{name}: {total - len(bad)}/{total} strings identical")
    for s, ref, got in bad[:8]:
        print("  MISMATCH", repr(s)[:100])
        print("    reference:", ref[:40])
        print("    converted:", got[:40])
    return 0 if not bad else 1


def spm(model, converted, skip_control=True):
    import sentencepiece as sp

    proc = sp.SentencePieceProcessor(model_file=model)
    tok = Tokenizer.from_str(open(converted, encoding="utf-8").read())
    control = [proc.id_to_piece(i) for i in range(proc.get_piece_size()) if proc.is_control(i) or proc.is_unknown(i)]
    bad = []
    items = [s for s in corpus() if not (skip_control and any(c in s for c in control))]
    for s in items:
        ref = proc.encode(s)
        got = tok.encode(s, add_special_tokens=False).ids
        if ref != got:
            bad.append((s, ref, got))
            continue
        # decoding of the reference ids
        if proc.decode(ref) != tok.decode(ref, skip_special_tokens=False) and not proc.is_unknown(ref[0] if ref else 0):
            bad.append((s + "  [decode]", [proc.decode(ref)], [tok.decode(ref, skip_special_tokens=False)]))
    return report("sentencepiece", bad, len(items))


def tiktoken_check(rank_file, pattern_name, converted, specials_path):
    import tiktoken

    ranks = {}
    for line in open(rank_file, "rb").read().splitlines():
        if line:
            t, r = line.split()
            ranks[base64.b64decode(t)] = int(r)
    doc = json.load(open(converted, encoding="utf-8"))
    # the added tokens of the converted tokenizer are the reference's special tokens
    specials = {t["content"]: t["id"] for t in doc["added_tokens"]}
    # patterns copied from the repositories' own tokenizer code (tokenization_kimi.py, tokenization_chatglm.py,
    # tokenization_qwen.py) and from tiktoken's o200k_base definition
    patterns = {
        "cl100k": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        "qwen1": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        "kimi": "|".join([
            r"""[\p{Han}]+""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
        ]),
    }
    patterns["glm"] = patterns["cl100k"]
    pat = patterns.get(pattern_name, pattern_name)
    enc = tiktoken.Encoding("ref", pat_str=pat, mergeable_ranks=ranks, special_tokens=specials)
    tok = Tokenizer.from_str(open(converted, encoding="utf-8").read())
    bad = []
    items = corpus()
    for s in items:
        ref = enc.encode(s, allowed_special="all")
        got = tok.encode(s, add_special_tokens=False).ids
        if ref != got:
            bad.append((s, ref, got))
        elif enc.decode(ref) != tok.decode(ref, skip_special_tokens=False):
            bad.append((s + "  [decode]", [enc.decode(ref)], [tok.decode(ref, skip_special_tokens=False)]))
    return report("tiktoken", bad, len(items))


def tekken(path, converted):
    from mistral_common.tokens.tokenizers.tekken import Tekkenizer

    t = Tekkenizer.from_file(path)
    tok = Tokenizer.from_str(open(converted, encoding="utf-8").read())
    bad = []
    items = [s for s in corpus() if not any(x in s for x in ("[INST]", "[/INST]", "<s>", "</s>", "<unk>", "<pad>"))]
    for s in items:
        ref = t.encode(s, bos=False, eos=False)
        got = tok.encode(s, add_special_tokens=False).ids
        if ref != got:
            bad.append((s, ref, got))
    # special tokens map to the same ids
    for i in range(min(t.num_special_tokens, 40)):
        piece = t.id_to_piece(i)
        ids = tok.encode(piece, add_special_tokens=False).ids
        if ids != [i]:
            bad.append((f"special {piece}", [i], ids))
    return report("tekken", bad, len(items) + 40)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "spm":
        sys.exit(spm(sys.argv[2], sys.argv[3], "--no-special" not in sys.argv))
    if mode == "tiktoken":
        sys.exit(tiktoken_check(sys.argv[2], sys.argv[3], sys.argv[4], "-"))
    if mode == "tekken":
        sys.exit(tekken(sys.argv[2], sys.argv[3]))
    sys.exit(__doc__)
