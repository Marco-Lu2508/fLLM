#!/usr/bin/env python3
"""Real-model API checks for fLLM's independent runtime. No third-party test runner.

Run: python3 scripts/verify_features.py --binary /path/to/fllm-rs --model MODEL.gguf
Use --draft-model MODEL.gguf to check target-vs-draft greedy output equivalence.
"""

import argparse
import base64
import concurrent.futures
import json
import math
import os
import socket
import struct
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model")
    parser.add_argument("--lora")
    parser.add_argument("--url", help="Check an already running isolated test server")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--kv-cache-type", default="f16")
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    if args.url and args.draft_model:
        parser.error("draft parity requires two locally launched test servers")
    base = args.url or f"http://127.0.0.1:{port}"
    key = "local-feature-verification"

    def request(path, body=None, auth=True, stream=False):
        headers = {"Content-Type": "application/json"}
        if auth:
            headers["Authorization"] = "Bearer " + key
        req = urllib.request.Request(
            base + path, None if body is None else json.dumps(body).encode(), headers
        )
        try:
            response = urllib.request.urlopen(req, timeout=180)
        except urllib.error.HTTPError as exc:
            if exc.code >= 500:
                raise RuntimeError(f"{path}: {exc.code}: {exc.read().decode()}") from exc
            raise
        with response:
            if stream:
                return [
                    json.loads(line[6:])
                    for line in response.read().decode().splitlines()
                    if line.startswith("data: ") and line != "data: [DONE]"
                ]
            return json.load(response)

    command = [
        args.binary,
        "serve",
        "--model",
        args.model,
        "--port",
        str(port),
        "--ctx",
        "1024",
        "--max-seqs",
        "4",
        "--max-batched-tokens",
        "128",
        "--threads",
        "3",
        "--model-name",
        "test-model",
        "--api-key",
        key,
    ]
    if args.lora:
        command += ["--lora", args.lora]
    command += ["--device", args.device, "--kv-cache-type", args.kv_cache_type]
    with tempfile.TemporaryFile(mode="w+") as log:
        proc = None if args.url else subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy()
        )
        try:
            deadline = time.monotonic() + 60
            while True:
                if proc is not None and proc.poll() is not None:
                    log.seek(0)
                    raise RuntimeError(log.read())
                try:
                    request("/health")
                    break
                except (OSError, urllib.error.URLError):
                    if time.monotonic() > deadline:
                        raise TimeoutError("server did not become ready")
                    time.sleep(0.2)
            passed = 0

            def check(name, callback):
                nonlocal passed
                callback()
                passed += 1
                print(f"PASS {name}", flush=True)

            def authentication():
                try:
                    request("/v1/models", auth=False)
                except urllib.error.HTTPError as exc:
                    assert exc.code == 401
                else:
                    raise AssertionError("unauthenticated model listing succeeded")

            check("authentication", authentication)

            def round_trip():
                ids = request("/tokenize", {"content": "Grüße aus Berlin!", "add_special": False})[
                    "tokens"
                ]
                assert request("/detokenize", {"tokens": ids})["content"] == "Grüße aus Berlin!"
                for text in ["中文 日本語 😀", "مرحبا", "Привет", "\n\téèß"]:
                    ids = request("/tokenize", {"content": text, "add_special": False})["tokens"]
                    assert request("/detokenize", {"tokens": ids})["content"] == text

            check("UTF-8 tokenize/detokenize", round_trip)

            schema = {
                "type": "object",
                "properties": {"answer": {"const": "Berlin"}},
                "required": ["answer"],
                "additionalProperties": False,
            }
            completion = {
                "model": "test-model",
                "prompt": "Answer with a JSON object naming Germany's capital:",
                "max_tokens": 64,
                "temperature": 0,
            }

            def structured_output():
                body = {
                    **completion,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "answer", "schema": schema},
                    },
                }
                result = request("/v1/completions", body)
                assert json.loads(result["choices"][0]["text"]) == {"answer": "Berlin"}
                pieces = request("/v1/completions", {**body, "stream": True}, stream=True)
                assert json.loads("".join(p["choices"][0].get("text", "") for p in pieces)) == {
                    "answer": "Berlin"
                }

            check("JSON schema in normal and streaming responses", structured_output)

            def grammar():
                result = request("/v1/completions", {**completion, "grammar": 'root ::= "yes"'})
                assert result["choices"][0]["text"] == "yes"

            check("GBNF constrained generation", grammar)

            def embeddings():
                result = request(
                    "/v1/embeddings", {"model": "test-model", "input": ["Berlin", "The ocean"]}
                )
                a, b = [item["embedding"] for item in result["data"]]
                assert len(a) == len(b) and len(a) > 64
                assert all(math.isfinite(x) for x in a + b)
                assert abs(math.sqrt(sum(x * x for x in a)) - 1) < 1e-5
                assert sum(abs(x - y) for x, y in zip(a, b)) > 1e-3
                again = request("/api/embed", {"model": "test-model", "input": "Berlin"})[
                    "embeddings"
                ][0]
                assert max(abs(x - y) for x, y in zip(a, again)) < 1e-5
                encoded = request(
                    "/v1/embeddings",
                    {"model": "test-model", "input": "Berlin", "encoding_format": "base64"},
                )["data"][0]["embedding"]
                decoded = struct.unpack("<" + "f" * len(a), base64.b64decode(encoded))
                assert max(abs(x - y) for x, y in zip(a, decoded)) < 1e-5

            check("contextual embeddings and clean KV isolation", embeddings)

            tool_request = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Call weather for Berlin."}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"const": "Berlin"}},
                                "required": ["city"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ],
                "tool_choice": "required",
                "temperature": 0,
                "max_tokens": 128,
            }

            def tools():
                choice = request("/v1/chat/completions", tool_request)["choices"][0]
                assert choice["finish_reason"] == "tool_calls", choice
                call = choice["message"]["tool_calls"][0]
                assert call["function"]["name"] == "weather"
                assert json.loads(call["function"]["arguments"]) == {"city": "Berlin"}
                pieces = request(
                    "/v1/chat/completions", {**tool_request, "stream": True}, stream=True
                )
                calls = [
                    call for p in pieces for call in p["choices"][0]["delta"].get("tool_calls", [])
                ]
                assert len(calls) == 1
                assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Berlin"}
                assert pieces[-1]["choices"][0]["finish_reason"] == "tool_calls"

            check("required tool calls and streaming protocol", tools)

            def seeded_requests():
                body = {
                    **completion,
                    "prompt": "One day in a small village",
                    "temperature": 0.8,
                    "seed": 17,
                    "min_p": 0.05,
                    "max_tokens": 16,
                }
                first = request("/v1/completions", body)["choices"][0]["text"]
                second = request("/v1/completions", body)["choices"][0]["text"]
                assert first == second, (first, second)
                with concurrent.futures.ThreadPoolExecutor(3) as pool:
                    results = list(
                        pool.map(
                            lambda _: request("/v1/completions", body)["choices"][0]["text"],
                            range(3),
                        )
                    )
                # Quantized single-row and multi-row kernels are not bitwise
                # equivalent. RNG isolation itself is checked in Rust tests.
                assert all(isinstance(result, str) and result for result in results), results

            check("seeded concurrent requests", seeded_requests)

            def validation():
                for body in [
                    {**completion, "min_p": 1.1},
                    {**completion, "prompt": [-1]},
                    {**completion, "grammar": "bad syntax"},
                    {**completion, "temperature": "bad"},
                ]:
                    try:
                        request("/v1/completions", body)
                    except urllib.error.HTTPError as exc:
                        assert exc.code == 400, exc.code
                    else:
                        raise AssertionError(f"invalid request accepted: {body}")
                assert request("/health")["status"] == "ok"

            check("invalid parameters cannot crash the worker", validation)

            def ollama():
                result = request(
                    "/api/generate",
                    {
                        "model": "test-model",
                        "prompt": completion["prompt"],
                        "format": schema,
                        "stream": False,
                        "options": {"num_predict": 64, "min_p": 0.1},
                    },
                )
                assert json.loads(result["response"]) == {"answer": "Berlin"}
                result = request(
                    "/api/chat",
                    {
                        **tool_request,
                        "stream": False,
                        "options": {"num_predict": 128, "temperature": 0},
                    },
                )
                assert result["message"]["tool_calls"][0]["function"]["arguments"] == {
                    "city": "Berlin"
                }, result

            check("Ollama schema and sampling options", ollama)

            prompts = ["The capital of France is", "1 2 3 4 5", "A red square is"]
            references = {
                prompt: request(
                    "/v1/completions", {**completion, "prompt": prompt, "max_tokens": 24}
                )["choices"][0]["text"]
                for prompt in prompts
            }
            if proc is not None:
                proc.terminate()
                proc.wait(timeout=10)
            if args.draft_model:
                proc = subprocess.Popen(
                    command + ["--draft-model", args.draft_model, "--draft-tokens", "3"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                deadline = time.monotonic() + 60
                while True:
                    if proc.poll() is not None:
                        log.seek(0)
                        raise RuntimeError(log.read())
                    try:
                        request("/health")
                        break
                    except OSError:
                        if time.monotonic() > deadline:
                            raise TimeoutError("draft server not ready")
                        time.sleep(0.2)

                def draft_parity():
                    for prompt, reference in references.items():
                        output = request(
                            "/v1/completions", {**completion, "prompt": prompt, "max_tokens": 24}
                        )["choices"][0]["text"]
                        assert output == reference, (prompt, reference, output)
                    for limit in [1, 2, 7]:
                        result = request(
                            "/v1/completions",
                            {**completion, "prompt": prompts[0], "max_tokens": limit},
                        )
                        assert result["usage"]["completion_tokens"] <= limit, result
                    with urllib.request.urlopen(base + "/metrics", timeout=10) as response:
                        metrics = response.read().decode()
                    counters = {
                        line.split()[0]: int(line.split()[1])
                        for line in metrics.splitlines()
                        if line.startswith("fllm_draft_")
                    }
                    assert counters["fllm_draft_proposed_tokens_total"] > 0, counters
                    print(f"Draft verification counters: {counters}", flush=True)

                check("draft model preserves target greedy output", draft_parity)
            print(f"Verified {passed} real-model feature checks", flush=True)
        except BaseException:
            log.flush()
            log.seek(0)
            print(log.read()[-16000:], flush=True)
            print(f"Server return code: {proc.poll() if proc else 'external'}", flush=True)
            raise
        finally:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


if __name__ == "__main__":
    main()
