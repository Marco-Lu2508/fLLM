#!/usr/bin/env python3
"""Compare fLLM's tokenization of a GGUF with the Hugging Face tokenizer of the same model.

    python scripts/verify_tokenizer.py MODEL.gguf TOKENIZER_DIR target/release/examples/tokenize
"""
import json
import subprocess
import sys

from tokenizers import Tokenizer

LINES = [
    "Hello, world!",
    "Die Hauptstadt von Frankreich ist Paris, 17 mal 23 ergibt 391.",
    "12345 + 678900 = 691245",
    "def add(a, b):\n    return a + b\n",
    "  leading and   multiple    spaces\n\nnew paragraphs",
    "naïve café – Übergrößen, 日本語のテキスト, русский текст, 😀 emoji",
    "I'm sure it's fine; they've said we'll go, he'd agree.",
    "URL https://example.com/a/b?x=1&y=2 and email a.b@c.de",
]

gguf, directory, tool = sys.argv[1:4]
reference = Tokenizer.from_file(f"{directory}/tokenizer.json")
stdin = "".join(json.dumps(line) + "\n" for line in LINES)
out = subprocess.run([tool, gguf], input=stdin, capture_output=True, text=True, check=True)
bad = 0
for line, mine in zip(LINES, out.stdout.strip().splitlines()):
    want = reference.encode(line, add_special_tokens=False).ids
    if json.loads(mine) != want:
        bad += 1
        print("MISMATCH", repr(line), "\n  fllm", mine[:200], "\n  hf  ", want[:60])
print(f"{len(LINES) - bad}/{len(LINES)} lines identical")
sys.exit(1 if bad else 0)
