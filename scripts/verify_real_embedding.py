#!/usr/bin/env python3
"""Cosine check of fllm encoder HTTP embeddings against transformers.

Starts `fllm-rs serve --embeddings` with a converted GGUF, embeds a few
sentences via `/v1/embeddings` and compares against the transformers
reference (CLS or mean pooling over the last hidden state).

Usage:
    verify_real_embedding.py --binary ./target/release/fllm-rs \\
        --gguf /tmp/nomic15.gguf --hf-dir /tmp/nomic15 --pooling mean
"""
import argparse
import json
import socket
import subprocess
import tempfile
import time
import urllib.request


def serve_request(url, payload, key="real-test"):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.loads(response.read().decode())


def transformers_embeddings(hf_dir, sentences, pooling):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=False)
    model = AutoModel.from_pretrained(hf_dir, trust_remote_code=False)
    model.eval()
    out = []
    with torch.inference_mode():
        for text in sentences:
            batch = tok(text, return_tensors="pt")
            hidden = model(**batch).last_hidden_state[0]
            if pooling == "cls":
                vector = hidden[0]
            elif pooling == "last":
                last = int(batch["attention_mask"].sum()) - 1
                vector = hidden[last]
            else:
                mask = batch["attention_mask"][0].unsqueeze(-1).float()
                vector = (hidden * mask).sum(0) / mask.sum()
            out.append(vector.numpy())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--hf-dir", required=True)
    parser.add_argument("--pooling", default="mean", choices=["cls", "mean", "last"])
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--sentences", nargs="+", default=[
        "Hello world, this is an embedding test.",
        "Rust systems programming for inference engines.",
        "Mean pooling with L2 normalization over tokens.",
    ])
    args = parser.parse_args()

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = args.port or sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(
            [args.binary, "serve", "--model", args.gguf, "--embeddings",
             "--threads", "3", "--device", "cpu", "--port", str(port),
             "--model-name", "real", "--api-key", "real-test"],
            stdout=log, stderr=log,
        )
        try:
            body = None
            for _ in range(120):
                if process.poll() is not None:
                    log.seek(0)
                    raise RuntimeError("server exited: " + log.read().decode()[-2000:])
                try:
                    body = serve_request(
                        url + "/v1/embeddings",
                        {"model": "real", "input": args.sentences, "pooling": args.pooling},
                    )
                    break
                except Exception:
                    time.sleep(1)
            if body is None:
                raise RuntimeError("server did not come up")
            server_vectors = [item["embedding"] for item in body["data"]]
            assert len(server_vectors) == len(args.sentences)
        finally:
            process.terminate()

    import numpy as np

    reference = transformers_embeddings(args.hf_dir, args.sentences, args.pooling)
    cosines = []
    for got, want in zip(server_vectors, reference):
        got = np.asarray(got, dtype=np.float64)
        want = np.asarray(want, dtype=np.float64)
        cosines.append(float(got @ want / (np.linalg.norm(got) * np.linalg.norm(want))))
    print(json.dumps({"pooling": args.pooling, "cosines": cosines,
                      "min_cosine": min(cosines)}, indent=2), flush=True)
    assert min(cosines) > 0.9999, f"cosine too low: {cosines}"


if __name__ == "__main__":
    main()
