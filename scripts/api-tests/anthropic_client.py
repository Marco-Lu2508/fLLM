"""Anthropic client compat test against fllm-rs. Exits nonzero on failure."""
import json
import os
import sys

import anthropic

BASE = os.environ.get("FLLM_URL", "http://127.0.0.1:8990")
MODEL = os.environ.get("FLLM_MODEL", "test-model")

client = anthropic.Anthropic(
    api_key=os.environ.get("FLLM_API_KEY", "sk-test"), base_url=BASE
)

fails = []


def check(name, fn):
    try:
        fn()
    except Exception as e:
        fails.append(name)
        print(f"FAIL {name}: {type(e).__name__}: {e}")
    else:
        print(f"ok {name}")


def t_message():
    m = client.messages.create(
        model=MODEL,
        max_tokens=16,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
    )
    assert m.content, "empty content blocks"
    texts = [b.text for b in m.content if getattr(b, "type", "") == "text"]
    assert "".join(texts).strip(), "no text in message"
    assert m.usage.input_tokens > 0
    print("  message:", repr("".join(texts)[:80]))


def t_stream():
    texts = []
    with client.messages.stream(
        model=MODEL,
        max_tokens=16,
        messages=[{"role": "user", "content": "Count: 1 2"}],
    ) as stream:
        for event in stream:
            if getattr(event, "type", "") == "content_block_delta":
                d = event.delta
                if getattr(d, "text", None):
                    texts.append(d.text)
        final = stream.get_final_message()
    assert texts, "stream produced no text"
    assert final.content, "no final message"
    print("  stream:", repr("".join(texts)[:80]))


def t_tools_shape():
    tools = [
        {
            "name": "get_weather",
            "description": "weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]
    m = client.messages.create(
        model=MODEL,
        max_tokens=64,
        messages=[{"role": "user", "content": "Weather in Berlin?"}],
        tools=tools,
        tool_choice={"type": "auto"},
    )
    kinds = [getattr(b, "type", "") for b in m.content]
    assert "text" in kinds or "tool_use" in kinds, f"unexpected blocks: {kinds}"
    for b in m.content:
        if getattr(b, "type", "") == "tool_use":
            assert b.name == "get_weather", b.name
            assert isinstance(b.input, dict), f"tool input not object: {b.input!r}"
    print("  blocks:", kinds)


def t_count_tokens():
    r = client.messages.count_tokens(
        model=MODEL, messages=[{"role": "user", "content": "Hello world"}]
    )
    assert r.input_tokens > 0, f"bad count: {r}"
    print("  input_tokens:", r.input_tokens)


def t_models_anthropic_shape():
    import httpx

    r = httpx.get(
        BASE + "/v1/models",
        headers={"anthropic-version": "2023-06-01"},
        timeout=10.0,
    )
    assert r.status_code == 200, r.text[:200]
    body = r.json()
    assert "data" in body and body["data"], body
    print("  anthropic models:", body["data"][0].get("id"))


def t_temperature_raw():
    # SDK 1.13 has no top-level temperature param; the REST API does.
    import httpx

    for temp in (0, 1):
        r = httpx.post(
            BASE + "/v1/messages",
            headers={"anthropic-version": "2023-06-01"},
            json={
                "model": MODEL,
                "max_tokens": 8,
                "temperature": temp,
                "messages": [{"role": "user", "content": "Hi"}],
            },
            timeout=60.0,
        )
        assert r.status_code == 200, f"temperature={temp}: {r.status_code} {r.text[:200]}"
    r = httpx.post(
        BASE + "/v1/messages",
        headers={"anthropic-version": "2023-06-01"},
        json={
            "model": MODEL,
            "max_tokens": 8,
            "temperature": -1,
            "messages": [{"role": "user", "content": "Hi"}],
        },
        timeout=30.0,
    )
    assert r.status_code == 400, f"bad temperature accepted: {r.status_code}"
    print("  temperature honored, invalid rejected")


for name, fn in [
    ("message", t_message),
    ("stream", t_stream),
    ("tools-shape", t_tools_shape),
    ("count-tokens", t_count_tokens),
    ("models-anthropic-shape", t_models_anthropic_shape),
    ("temperature-raw", t_temperature_raw),
]:
    check(name, fn)

if fails:
    print(f"{len(fails)} FAILURES: {fails}")
    sys.exit(1)
print("anthropic_client: all ok")
