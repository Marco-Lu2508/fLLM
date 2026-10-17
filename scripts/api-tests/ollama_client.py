"""Ollama client compat test against fllm-rs. Exits nonzero on failure."""
import json
import os
import sys

import ollama

HOST = os.environ.get("FLLM_URL", "http://127.0.0.1:8990")
MODEL = os.environ.get("FLLM_MODEL", "test-model")

client = ollama.Client(host=HOST)
fails = []


def check(name, fn):
    try:
        fn()
    except Exception as e:
        fails.append(name)
        print(f"FAIL {name}: {type(e).__name__}: {e}")
    else:
        print(f"ok {name}")


def t_list():
    r = client.list()
    names = []
    for m in r.models:
        names.append(getattr(m, "model", "") or getattr(m, "name", ""))
    assert MODEL in names or any(MODEL in n for n in names), f"{MODEL} not in {names}"
    print("  list:", names)


def t_chat():
    r = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        stream=False,
        options={"temperature": 0, "num_predict": 16},
        keep_alive=0,
    )
    assert r.message and r.message.content and r.message.content.strip(), "empty chat"
    assert r.done is True, f"done flag: {r.done}"
    print("  chat:", repr(r.message.content[:80]))


def t_chat_stream():
    parts = []
    last = None
    for chunk in client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "Count: 1 2"}],
        stream=True,
        options={"temperature": 0, "num_predict": 16},
    ):
        last = chunk
        if chunk.message and chunk.message.content:
            parts.append(chunk.message.content)
    assert parts, "stream produced no text"
    assert last is not None and last.done is True, "final chunk not done"
    print("  chat-stream:", repr("".join(parts)[:80]))


def t_generate():
    r = client.generate(
        model=MODEL,
        prompt="The capital of France is",
        stream=False,
        options={"temperature": 0, "num_predict": 8},
        keep_alive=0,
    )
    assert r.response and r.response.strip(), "empty generate"
    assert r.done is True
    print("  generate:", repr(r.response[:80]))


def t_generate_stream():
    parts = []
    last = None
    for chunk in client.generate(
        model=MODEL,
        prompt="1, 2, 3,",
        stream=True,
        options={"temperature": 0, "num_predict": 8},
    ):
        last = chunk
        if chunk.response:
            parts.append(chunk.response)
    assert parts, "generate stream empty"
    assert last is not None and last.done is True
    print("  generate-stream:", repr("".join(parts)[:80]))


def t_embed():
    r = client.embed(model=MODEL, input=["hello", "world"])
    assert len(r.embeddings) == 2, f"expected 2, got {len(r.embeddings)}"
    assert len(r.embeddings[0]) > 16
    print("  embed dim:", len(r.embeddings[0]))


def t_show():
    r = client.show(model=MODEL)
    assert r.modelfile and "PARAMETER" in r.modelfile, "show: no modelfile"
    assert r.details, "show: no details"
    assert r.modelinfo or getattr(r, "model_info", None), "show: no model info"
    print("  show modelfile bytes:", len(r.modelfile))


def t_ps():
    r = client.ps()
    assert r.models, "ps: empty"
    print("  ps models:", len(r.models))


def t_format_json():
    r = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "Give me a person."}],
        stream=False,
        format={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
        options={"temperature": 0, "num_predict": 64},
    )
    content = r.message.content
    obj = json.loads(content)
    assert "name" in obj, f"format=json did not constrain: {content!r}"
    print("  format-json:", repr(content[:100]))


def t_tools_shape():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    r = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "Weather in Berlin?"}],
        stream=False,
        tools=tools,
        options={"temperature": 0, "num_predict": 64},
    )
    calls = r.message.tool_calls or []
    if calls:
        for tc in calls:
            assert tc.function.name == "get_weather", tc.function.name
            assert isinstance(tc.function.arguments, dict), f"args not object: {tc.function.arguments!r}"
        print("  tools called:", len(calls))
    else:
        assert r.message.content, "neither content nor tool_calls"
        print("  tools not called, content ok")


def t_pull_delete_unsupported():
    # fLLM serves one model per process; pull/delete must fail loudly, not hang.
    for op in ("pull", "delete"):
        try:
            if op == "pull":
                client.pull(model=MODEL, stream=False)
            else:
                client.delete(model=MODEL)
        except Exception as e:
            print(f"  {op} correctly rejected: {type(e).__name__}")
        else:
            raise AssertionError(f"{op} unexpectedly succeeded")


def t_error_shapes():
    import httpx

    cases = [
        ("POST", "/api/generate", {}, 400),
        ("POST", "/api/chat", {"model": MODEL, "stream": False, "messages": []}, 400),
        ("POST", "/api/show", {"model": "does-not-exist"}, 404),
        ("POST", "/api/embed", {}, 400),
    ]
    for method, path, payload, status in cases:
        r = httpx.post(HOST + path, json=payload, timeout=30.0)
        assert r.status_code == status, f"{path}: {r.status_code} != {status}: {r.text[:160]}"
        body = r.json()
        assert isinstance(body.get("error"), str), (
            f"{path}: Ollama error must be a string, got {r.text[:160]}"
        )
        print(f"  {path} -> {status} {body['error'][:60]!r}")


for name, fn in [
    ("list", t_list),
    ("chat", t_chat),
    ("chat-stream", t_chat_stream),
    ("generate", t_generate),
    ("generate-stream", t_generate_stream),
    ("embed", t_embed),
    ("show", t_show),
    ("ps", t_ps),
    ("format-json", t_format_json),
    ("tools-shape", t_tools_shape),
    ("pull-delete-unsupported", t_pull_delete_unsupported),
    ("error-shapes", t_error_shapes),
]:
    check(name, fn)

if fails:
    print(f"{len(fails)} FAILURES: {fails}")
    sys.exit(1)
print("ollama_client: all ok")
