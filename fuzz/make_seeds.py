#!/usr/bin/env python3
"""Writes small valid inputs into fuzz/corpus/<target>/ so that the fuzzers start from files that get past the first checks.

    python3 fuzz/make_seeds.py            # from the repository root
"""
import json
import os
import struct

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")


def put(target, name, data):
    directory = os.path.join(root, target)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, name), "wb") as f:
        f.write(data)


def gguf_string(text):
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def gguf_kv(key, kind, payload):
    return gguf_string(key) + struct.pack("<I", kind) + payload


def gguf(tensors, kvs, version=3):
    out = b"GGUF" + struct.pack("<IQQ", version, len(tensors), len(kvs)) + b"".join(kvs)
    offset = 0
    data = b""
    for name, dims, kind, nbytes in tensors:
        out += gguf_string(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        out += struct.pack("<IQ", kind, offset)
        offset += (nbytes + 31) // 32 * 32
        data += bytes(nbytes) + bytes((nbytes + 31) // 32 * 32 - nbytes)
    out += bytes((32 - len(out) % 32) % 32)
    return out + data


kvs = [
    gguf_kv("general.architecture", 8, gguf_string("llama")),
    gguf_kv("llama.context_length", 4, struct.pack("<I", 512)),
    gguf_kv("general.sampling.temp", 6, struct.pack("<f", 0.7)),
    gguf_kv("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 3) + gguf_string("a") + gguf_string("b") + gguf_string("c")),
    gguf_kv("tokenizer.ggml.token_type", 9, struct.pack("<IQ", 4, 3) + struct.pack("<3I", 1, 1, 3)),
]
put("gguf", "empty", gguf([], []))
put("gguf", "metadata", gguf([], kvs))
put("gguf", "one_tensor", gguf([("token_embd.weight", [4, 2], 0, 32)], kvs))
put("gguf", "two_tensors", gguf([("a", [8], 0, 32), ("blk.0.attn_q.weight", [2, 2], 0, 16)], kvs[:2]))

header = {"w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}, "__metadata__": {"format": "pt"}}
raw = json.dumps(header).encode()
put("safetensors", "one", struct.pack("<Q", len(raw)) + raw + bytes(16))
header = {"a": {"dtype": "F16", "shape": [4], "data_offsets": [0, 8]}, "b": {"dtype": "BF16", "shape": [2], "data_offsets": [8, 12]}}
raw = json.dumps(header).encode()
put("safetensors", "two", struct.pack("<Q", len(raw)) + raw + bytes(12))

# data.pkl of a state_dict with one 4-element float tensor (the fuzz target wraps it in a stored ZIP with a storage "0" of 16 bytes);
# the first byte selects how the target reads the input: even = pickle inside a generated archive
pickle = (
    b"\x80\x02ccollections\nOrderedDict\n)R(X\x01\x00\x00\x00w"
    b"ctorch._utils\n_rebuild_tensor_v2\n"
    b"((X\x07\x00\x00\x00storagectorch\nFloatStorage\nX\x01\x00\x00\x000X\x03\x00\x00\x00cpuK\x04tQ"
    b"K\x00K\x04\x85K\x01\x85\x89ccollections\nOrderedDict\n)Rt"
    b"Ru."
)
put("pytorch", "state_dict", b"\x00" + pickle)
put("pytorch", "state_dict_nested", b"\x00" + b"\x80\x02}(X\x0a\x00\x00\x00state_dict}u.")

# chat templates: the first line is the template, the rest the JSON message list
messages = b'\n[{"role":"system","content":"s"},{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}]'
templates = {
    "chatml": rb"{% for m in messages %}<|im_start|>{{ m['role'] }}{{ '\n' }}{{ m['content'] }}<|im_end|>{{ '\n' }}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant{{ '\n' }}{% endif %}",
    "llama": rb"{{ bos_token }}{% for m in messages %}{% if m['role'] == 'system' %}[SYS]{{ m['content']|trim }}[/SYS]{% else %}{{ m['role']|upper }}: {{ m['content']|tojson }}{% endif %}{% endfor %}",
    "namespace": rb"{% set ns = namespace(n=0) %}{% for m in messages %}{% set ns.n = ns.n + loop.index %}{{ m.content.split()|join('-') }}{% endfor %}{{ ns.n }}{{ raise_exception('x') if ns.n > 100 }}",
    "loop": rb"{% for i in range(100000000000) %}x{% endfor %}",
}
for name, template in templates.items():
    put("chat_template", name, template + messages)
print("seeds written to", root)
