"""Verify independent encoder inference against PyTorch using the same GGUF weights.

Supports the BERT layout (`bert`, `roberta`, `xlm-roberta`, `distilbert`,
`electra`) with absolute positions, `modernbert` (fused QKV/GeGLU,
dual-theta RoPE, local/global sliding attention, bias-free norms) as well as
`nomic_bert` (single-theta RoPE, separate bias-free Q/K/V/O, SwiGLU, biased
post norms). Dimensions, layer count, position handling, pooling and
normalization are read from the GGUF metadata instead of being hard-coded.

Optional test dependencies: torch, numpy, gguf, tokenizers. No llama runtime.
"""
import argparse
import base64
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


def request(url, payload=None, key="encoder-test", expected=200):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data, headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            status, data = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, data = error.code, error.read()
    assert status == expected, (status, data[:500])
    return json.loads(data) if data else None


def _scalar(field):
    return field.parts[field.data[0]][0]


def _numbers(field):
    try:
        return list(field.contents())
    except Exception:
        return list(field.parts[field.data[0]])


def _text(field):
    try:
        contents = field.contents()
        if isinstance(contents, str):
            return contents
        return contents[0]
    except Exception:
        return _scalar(field)


def gguf_meta(model):
    from gguf import GGUFReader

    reader = GGUFReader(model)
    fields = reader.fields

    def opt(key, default=None):
        field = fields.get(key)
        return default if field is None else _scalar(field)

    def opt_text(key, default=None):
        field = fields.get(key)
        return default if field is None else _text(field)

    meta = {
        "hidden": int(opt("bert.embedding_length")),
        "layers": int(opt("bert.block_count")),
        "heads": int(opt("bert.attention.head_count")),
        "intermediate": int(opt("bert.feed_forward_length")),
        "context": int(opt("bert.context_length")),
        "eps": float(opt("bert.attention.layer_norm_epsilon", 1e-12)),
        "activation": opt_text("bert.activation", "gelu"),
        "model_type": opt_text("bert.model_type", "bert"),
        "offset": int(opt("bert.position_offset", 0)),
        "tokenizer_model": opt_text("tokenizer.ggml.model", "bert"),
        "theta_global": float(opt("bert.rope_theta_global", 160000.0)),
        "theta_local": float(opt("bert.rope_theta_local", 10000.0)),
        "theta": float(opt("bert.rope_theta", 1000.0)),
        "local_attention": int(opt("bert.local_attention", 128)),
        "vocab": len(fields["tokenizer.ggml.tokens"].contents()),
        "has_position": any(t.name == "position_embd.weight" for t in reader.tensors),
        "has_types": any(t.name == "token_types.weight" for t in reader.tensors),
    }
    layer_field = fields.get("bert.layer_global")
    if layer_field is not None:
        meta["layer_global"] = [bool(v) for v in _numbers(layer_field)]
    else:
        meta["layer_global"] = None
    cls_field = fields.get("tokenizer.ggml.cls_token_id")
    sep_field = fields.get("tokenizer.ggml.separator_token_id")
    meta["cls_id"] = int(_scalar(cls_field)) if cls_field is not None else None
    meta["sep_id"] = int(_scalar(sep_field)) if sep_field is not None else None
    return meta


def _torch_activation(name):
    import torch
    from torch.nn import functional as f

    if name == "relu":
        return lambda x: torch.relu(x)
    if name in ("silu", "swish"):
        return lambda x: f.silu(x)
    if name in ("gelu_new", "gelu_fast", "geglu"):
        return lambda x: f.gelu(x, approximate="tanh")
    return lambda x: f.gelu(x, approximate="none")


def reference_bert(weights, meta, ids, pooling):
    import torch
    from torch.nn import functional as f

    hidden, heads = meta["hidden"], meta["heads"]
    head_dim = hidden // heads
    n = len(ids)

    def linear(x, name):
        return f.linear(x, weights[name + ".weight"], weights.get(name + ".bias"))

    def norm(x, name):
        return f.layer_norm(x, (hidden,), weights[name + ".weight"], weights.get(name + ".bias"), meta["eps"])

    act = _torch_activation(meta["activation"])
    with torch.inference_mode():
        x = weights["token_embd.weight"][ids]
        if meta["has_position"]:
            positions = torch.arange(meta["offset"], meta["offset"] + n)
            x = x + weights["position_embd.weight"][positions]
        if meta["has_types"]:
            x = x + weights["token_types.weight"][0]
        x = norm(x, "token_embd_norm")
        for i in range(meta["layers"]):
            prefix = f"blk.{i}"
            q, k, v = [linear(x, prefix + ".attn_" + name).view(n, heads, head_dim).transpose(0, 1).unsqueeze(0) for name in ["q", "k", "v"]]
            a = f.scaled_dot_product_attention(q, k, v).squeeze(0).transpose(0, 1).reshape(n, hidden)
            x = norm(x + linear(a, prefix + ".attn_output"), prefix + ".attn_output_norm")
            x = norm(x + linear(act(linear(x, prefix + ".ffn_up")), prefix + ".ffn_down"), prefix + ".layer_output_norm")
        vector = x[0] if pooling == "cls" else x[-1] if pooling == "last" else x.mean(dim=0)
        return f.normalize(vector, dim=0).numpy()


def reference_modernbert(weights, meta, ids, pooling):
    import torch
    from torch.nn import functional as f

    hidden, heads = meta["hidden"], meta["heads"]
    head_dim = hidden // heads
    n = len(ids)
    half = meta["local_attention"] // 2
    layer_global = meta["layer_global"]
    if layer_global is None:
        layer_global = [(i % 3 == 0) for i in range(meta["layers"])]
    pos = torch.arange(n)

    def plain_norm(x, name):
        return f.layer_norm(x, (hidden,), weights[name + ".weight"], None, meta["eps"])

    with torch.inference_mode():
        x = plain_norm(weights["token_embd.weight"][ids], "token_embd_norm")
        for i in range(meta["layers"]):
            prefix = f"blk.{i}"
            h = x if f"{prefix}.attn_output_norm.weight" not in weights else plain_norm(x, prefix + ".attn_output_norm")
            qkv = f.linear(h, weights[prefix + ".attn_qkv.weight"])
            q, k, v = [t.view(n, heads, head_dim).transpose(0, 1) for t in qkv.chunk(3, dim=-1)]
            theta = meta["theta_global"] if layer_global[i] else meta["theta_local"]
            inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
            emb = torch.cat([torch.outer(pos.float(), inv)] * 2, dim=-1)
            cos, sin = emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)

            def rotate(t):
                a, b = t.chunk(2, dim=-1)
                return torch.cat([-b, a], dim=-1)

            q = q * cos + rotate(q) * sin
            k = k * cos + rotate(k) * sin
            scores = q @ k.transpose(-1, -2) / (head_dim ** 0.5)
            if not layer_global[i]:
                dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
                scores = scores.masked_fill(dist.unsqueeze(0) > half, float("-inf"))
            a = (torch.softmax(scores, dim=-1) @ v).transpose(0, 1).reshape(n, hidden)
            x = x + f.linear(a, weights[prefix + ".attn_output.weight"])
            h = plain_norm(x, prefix + ".ffn_norm")
            up = f.linear(h, weights[prefix + ".ffn_up.weight"])
            first, gate = up.chunk(2, dim=-1)
            x = x + f.linear(f.gelu(first, approximate="tanh") * gate, weights[prefix + ".ffn_down.weight"])
        x = plain_norm(x, "output_norm")
        vector = x[0] if pooling == "cls" else x[-1] if pooling == "last" else x.mean(dim=0)
        return f.normalize(vector, dim=0).numpy()


def reference_nomicbert(weights, meta, ids, pooling):
    import torch
    from torch.nn import functional as f

    hidden, heads = meta["hidden"], meta["heads"]
    head_dim = hidden // heads
    n = len(ids)
    pos = torch.arange(n)

    def linear(x, name):
        return f.linear(x, weights[name + ".weight"], weights.get(name + ".bias"))

    def norm(x, name):
        return f.layer_norm(x, (hidden,), weights[name + ".weight"], weights.get(name + ".bias"), meta["eps"])

    with torch.inference_mode():
        x = weights["token_embd.weight"][ids]
        if meta["has_types"]:
            x = x + weights["token_types.weight"][0]
        x = norm(x, "token_embd_norm")
        theta = meta["theta"]
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        emb = torch.cat([torch.outer(pos.float(), inv)] * 2, dim=-1)
        cos, sin = emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)

        def rotate(t):
            a, b = t.chunk(2, dim=-1)
            return torch.cat([-b, a], dim=-1)

        for i in range(meta["layers"]):
            prefix = f"blk.{i}"
            if prefix + ".attn_qkv.weight" in weights:
                qkv = f.linear(x, weights[prefix + ".attn_qkv.weight"])
                q, k, v = [t.view(n, heads, head_dim).transpose(0, 1).unsqueeze(0) for t in qkv.chunk(3, dim=-1)]
            else:
                q, k, v = [linear(x, prefix + ".attn_" + name).view(n, heads, head_dim).transpose(0, 1).unsqueeze(0) for name in ["q", "k", "v"]]
            q = q * cos + rotate(q) * sin
            k = k * cos + rotate(k) * sin
            a = f.scaled_dot_product_attention(q, k, v).squeeze(0).transpose(0, 1).reshape(n, hidden)
            x = norm(x + linear(a, prefix + ".attn_output"), prefix + ".attn_output_norm")
            x = norm(x + linear(f.silu(linear(x, prefix + ".ffn_gate")) * linear(x, prefix + ".ffn_up"), prefix + ".ffn_down"), prefix + ".layer_output_norm")
        vector = x[0] if pooling == "cls" else x[-1] if pooling == "last" else x.mean(dim=0)
        return f.normalize(vector, dim=0).numpy()


def reference(model, ids, pooling):
    import numpy as np
    import torch
    from gguf import GGUFReader, dequantize

    torch.set_num_threads(3)
    meta = gguf_meta(model)
    reader = GGUFReader(model)
    weights = {tensor.name: torch.from_numpy(np.array(dequantize(tensor.data, tensor.tensor_type), copy=True)) for tensor in reader.tensors}
    if meta["model_type"] == "modernbert":
        return reference_modernbert(weights, meta, ids, pooling)
    if meta["model_type"] == "nomic_bert":
        return reference_nomicbert(weights, meta, ids, pooling)
    return reference_bert(weights, meta, ids, pooling)



def verify_classifier(url, source, num_labels, skip_reference):
    """Compare HTTP probabilities to the installed transformers implementation."""
    import numpy as np
    import torch
    import transformers as T

    texts = ["The cat sits on the mat.", "Eine schöne Straße, Grüße!", ""]
    query = "The cat"
    model = tokenizer = None
    if not skip_reference:
        model = T.AutoModelForSequenceClassification.from_pretrained(source, local_files_only=True, attn_implementation="eager").float().eval()
        tokenizer = T.PreTrainedTokenizerFast(tokenizer_file=str(Path(source) / "tokenizer.json"))
        assert model.config.num_labels == num_labels

    def expected(text, pair=None):
        encoding = tokenizer.backend_tokenizer.encode(text, pair, add_special_tokens=True)
        ids = torch.tensor([encoding.ids], dtype=torch.long)
        types = torch.tensor([encoding.type_ids], dtype=torch.long)
        with torch.inference_mode():
            logits = model(input_ids=ids, token_type_ids=types, attention_mask=torch.ones_like(ids)).logits[0]
            return (torch.sigmoid(logits) if num_labels == 1 else torch.softmax(logits, dim=-1)).numpy()

    def compare(actual, target):
        actual = np.asarray(actual, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        assert actual.shape == target.shape and np.isfinite(actual).all()
        relative = float(np.max(np.abs(actual - target) / np.maximum(np.abs(target), 1e-6)))
        assert relative < 2e-3, (actual.tolist(), target.tolist(), relative)
        assert int(actual.argmax()) == int(target.argmax())
        print(json.dumps({"task": "classification", "labels": num_labels, "max_relative_error": relative}), flush=True)

    request(url + "/classify", {"input": texts}, key="wrong", expected=401)
    result = request(url + "/classify", {"model": "bge", "input": texts})
    assert len(result["data"]) == len(texts)
    for index, item in enumerate(result["data"]):
        assert item["index"] == index and item["num_classes"] == num_labels
        probs = np.asarray(item["probs"], dtype=np.float32)
        assert probs.shape == (num_labels,) and ((probs >= 0) & (probs <= 1)).all()
        if num_labels > 1:
            assert abs(float(probs.sum()) - 1.0) < 1e-5
        if not skip_reference:
            target = expected(texts[index])
            compare(probs, target)
            assert item["label"] == model.config.id2label[int(target.argmax())]
    pooled = request(url + "/pooling", {"input": texts})
    assert len(pooled["data"]) == len(texts)
    if model is not None:
        assert all(len(item["data"]) == model.config.hidden_size for item in pooled["data"])
    if num_labels == 1:
        scores = request(url + "/v1/score", {"text_1": query, "text_2": texts})["data"]
        for index, item in enumerate(scores):
            assert item["index"] == index
            if not skip_reference:
                compare([item["score"]], expected(query, texts[index]))
        docs = [texts[0], {"text": texts[1], "title": "preserved"}, texts[2]]
        reranked = request(url + "/v1/rerank", {"query": query, "documents": docs, "top_n": 2, "return_documents": True})["results"]
        order = sorted(range(len(scores)), key=lambda i: (-scores[i]["score"], i))[:2]
        assert [item["index"] for item in reranked] == order
        for item in reranked:
            assert item["document"]["text"] == texts[item["index"]]
            assert item["relevance_score"] == scores[item["index"]]["score"]
        alias = request(url + "/rerank", {"query": query, "documents": texts, "top_n": 2})["results"]
        assert [item["index"] for item in alias] == order and all("document" not in item for item in alias)
    else:
        request(url + "/v1/score", {"text_1": query, "text_2": texts}, expected=400)
        request(url + "/v1/rerank", {"query": query, "documents": texts}, expected=400)
    for payload in [{"input": []}, {"input": [1]}, {"input": "x" * 65537}, {"input": "x", "model": "wrong"}]:
        request(url + "/classify", payload, expected=400)
    print("Encoder classification/parity checks passed", flush=True)


def main():
    import numpy as np
    from gguf import GGUFReader
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--url", help="Verify an already running encoder, without starting/stopping it")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--hf-reference", help="Local source checkpoint for transformers classification parity")
    args = parser.parse_args()
    meta = gguf_meta(args.model)
    hidden, context = meta["hidden"], meta["context"]
    wordpiece = meta["tokenizer_model"] == "bert"
    reader = GGUFReader(args.model)
    label_field = reader.fields.get("bert.classifier.label_count")
    num_labels = int(label_field.parts[label_field.data[0]][0]) if label_field is not None else 0
    if num_labels and not args.skip_reference and not args.hf_reference:
        parser.error("classification parity requires --hf-reference pointing to the local source checkpoint")
    vocab = {token: i for i, token in enumerate(reader.fields["tokenizer.ggml.tokens"].contents())}
    tokenizer = None
    if wordpiece and not num_labels:
        tokenizer = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
        tokenizer.normalizer = normalizers.BertNormalizer()
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = args.url or f"http://127.0.0.1:{port}"
    with tempfile.TemporaryFile() as log:
        process = None if args.url else subprocess.Popen([args.binary, "serve", "--model", args.model, "--embeddings", "--threads", "3", "--device", args.device, "--port", str(port), "--model-name", "bge", "--api-key", "encoder-test"], stdout=log, stderr=log)
        try:
            for _ in range(120):
                if process is not None and process.poll() is not None:
                    log.seek(0)
                    raise RuntimeError(log.read().decode())
                try:
                    health = request(url + "/health")
                    assert health["engine"] == "fllm-rust"
                    break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.25)
            else:
                raise RuntimeError("encoder did not start")
            request(url + "/v1/models", key="wrong", expected=401)
            if num_labels:
                verify_classifier(url, args.hf_reference, num_labels, args.skip_reference)
                return
            texts = ["The cat sits on the mat.", "Eine schöne Straße, Grüße!", "Hello world, this is a test."]
            for text in texts:
                ids = request(url + "/tokenize", {"content": text})["tokens"]
                if tokenizer is not None:
                    expected = [vocab["[CLS]"], *tokenizer.encode(text).ids, vocab["[SEP]"]]
                    assert ids == expected, (ids, expected)
                for pooling in ["cls", "mean", "last"]:
                    payload = {"model": "bge", "input": text, "pooling": pooling}
                    result = request(url + "/v1/embeddings", payload)
                    vector = np.array(result["data"][0]["embedding"], dtype=np.float32)
                    assert vector.shape == (hidden,) and np.isfinite(vector).all()
                    assert abs(np.linalg.norm(vector) - 1) < 1e-5
                    if not args.skip_reference:
                        target = reference(args.model, ids, pooling)
                        maximum = float(np.max(np.abs(vector - target)))
                        cosine = float(vector @ target)
                        assert maximum < 3e-5 and cosine > 0.99999, (maximum, cosine)
                        print(json.dumps({"model_type": meta["model_type"], "device": args.device, "pooling": pooling, "max_abs_error": maximum, "cosine": cosine}), flush=True)
            batch = request(url + "/api/embed", {"model": "bge", "input": ["The quick brown fox jumps over the lazy dog.", "The quick brown fox jumps over the lazy dog.", "Quantum mechanics describes nature at the smallest scales with uncertainty."]})["embeddings"]
            assert np.array_equal(batch[0], batch[1])
            assert float(np.linalg.norm(np.array(batch[0]) - np.array(batch[2]))) > 1e-4
            mid = min(10, meta["vocab"] - 1)
            probe = [meta["cls_id"] or 0, mid, meta["sep_id"] or 0]
            result = request(url + "/v1/embeddings", {"input": probe, "encoding_format": "base64"})
            assert len(base64.b64decode(result["data"][0]["embedding"])) == hidden * 4
            # Matryoshka truncation: dimensions < hidden is accepted, truncated and renormalized.
            half = max(1, hidden // 2)
            result = request(url + "/v1/embeddings", {"input": "test", "dimensions": half})
            truncated = np.array(result["data"][0]["embedding"], dtype=np.float32)
            assert truncated.shape == (half,) and abs(np.linalg.norm(truncated) - 1) < 1e-5
            print(json.dumps({"matryoshka": half, "norm": float(np.linalg.norm(truncated))}), flush=True)
            for payload in [{"input": []}, {"input": [meta["vocab"] + 999]}, {"input": "test", "dimensions": 0}, {"input": "test", "dimensions": hidden + 1}, {"input": "test", "pooling": "bad"}, {"input": [[probe[0]] * (context + 1)]}, {"input": "test", "model": "wrong"}]:
                request(url + "/v1/embeddings", payload, expected=400)
            request(url + "/v1/chat/completions", {"messages": []}, expected=404)
            print("Independent Rust encoder HTTP/parity checks passed", flush=True)
        finally:
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()

