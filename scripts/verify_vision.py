"""Native SigLIP/Idefics3 parity and multimodal chat verification against PyTorch."""
import argparse
import base64
import binascii
import json
import socket
import struct
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import zlib


def png_url(rgb):
    def chunk(name, data):
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", binascii.crc32(name + data) & 0xFFFFFFFF)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 512, 512, 8, 2, 0, 0, 0))
    data += chunk(b"IDAT", zlib.compress((b"\0" + bytes(rgb) * 512) * 512)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(data).decode()


def request(base, path, body, expected=200):
    req = urllib.request.Request(base + path, None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            status, data = response.status, response.read()
    except urllib.error.HTTPError as exc:
        status, data = exc.code, exc.read()
    assert status == expected, (status, data[:1000])
    return json.loads(data)


def reference(projector, rgb):
    import numpy as np
    import torch
    from gguf import GGUFReader, dequantize
    from torch.nn import functional as f

    torch.set_num_threads(3)
    reader = GGUFReader(projector)
    weights = {t.name: torch.from_numpy(np.array(dequantize(t.data, t.tensor_type), copy=True)) for t in reader.tensors}
    def linear(x, name):
        return f.linear(x, weights[name + ".weight"], weights.get(name + ".bias"))
    def norm(x, name):
        return f.layer_norm(x, (768,), weights[name + ".weight"], weights[name + ".bias"], 1e-6)
    with torch.inference_mode():
        pixels = torch.tensor(rgb, dtype=torch.float32).reshape(1, 3, 1, 1).expand(1, 3, 512, 512) / 127.5 - 1
        x = f.conv2d(pixels, weights["v.patch_embd.weight"], weights["v.patch_embd.bias"], stride=16).flatten(2).transpose(1, 2)
        x += weights["v.position_embd.weight"]
        for i in range(12):
            p = f"v.blk.{i}"
            n = norm(x, p + ".ln1")
            q, k, v = [linear(n, p + ".attn_" + name).reshape(1, 1024, 12, 64).transpose(1, 2) for name in ["q", "k", "v"]]
            a = f.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, 1024, 768)
            x += linear(a, p + ".attn_out")
            n = norm(x, p + ".ln2")
            x += linear(f.gelu(linear(n, p + ".ffn_down"), approximate="none"), p + ".ffn_up")
        x = norm(x, "v.post_ln").reshape(1, 32, 8, 3072).permute(0, 2, 1, 3).reshape(1, 8, 8, 12288).permute(0, 2, 1, 3).reshape(64, 12288)
        return linear(x, "mm.model.fc").numpy()


def main():
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mmproj", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--url")
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = args.url or f"http://127.0.0.1:{port}"
    with tempfile.TemporaryFile(mode="w+") as log:
        proc = None if args.url else subprocess.Popen([args.binary, "serve", "--model", args.model, "--mmproj", args.mmproj, "--ctx", "512", "--max-seqs", "2", "--threads", "3", "--port", str(port), "--device", args.device], stdout=log, stderr=log)
        try:
            for _ in range(240):
                if proc is not None and proc.poll() is not None:
                    log.seek(0)
                    raise RuntimeError(log.read())
                try:
                    request(base, "/health", None)
                    break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.25)
            else:
                raise RuntimeError("vision server did not start")
            red = png_url([255, 0, 0])
            result = request(base, "/v1/images/embeddings", {"image_url": red})
            assert result["engine"] == "fllm-rust" and result["tokens"] == 64 and result["dimensions"] == 576
            features = np.array(result["data"], dtype=np.float32)
            target = reference(args.mmproj, [255, 0, 0])
            max_error = float(np.max(np.abs(features - target)))
            cosine = float(np.vdot(features, target) / (np.linalg.norm(features) * np.linalg.norm(target)))
            assert max_error < 0.003 and cosine > 0.99999, (max_error, cosine)
            print(json.dumps({"device": args.device, "vision_max_abs_error": max_error, "vision_cosine": cosine}), flush=True)
            results = []
            for rgb in [[255, 0, 0], [0, 0, 255], [255, 0, 0]]:
                content = [{"type": "image_url", "image_url": {"url": png_url(rgb)}}, {"type": "text", "text": "What color is this image? Answer with one color word."}]
                result = request(base, "/v1/chat/completions", {"messages": [{"role": "user", "content": content}], "max_tokens": 8, "temperature": 0})
                results.append(result["choices"][0]["message"]["content"].strip().lower())
            assert "red" in results[0] and "blue" in results[1] and results[0] == results[2], results
            request(base, "/v1/images/embeddings", {"image_url": "data:image/png;base64,broken"}, expected=400)
            request(base, "/v1/chat/completions", {"messages": [{"role": "user", "content": "<image>"}]}, expected=400)
            print(json.dumps({"vision_chat": results, "cache_isolation": "passed"}), flush=True)
        finally:
            if proc is not None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


if __name__ == "__main__":
    main()
