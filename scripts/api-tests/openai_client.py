"""OpenAI client compat test against fllm-rs. Exits nonzero on failure."""
import json
import os
import sys

from openai import OpenAI

BASE = os.environ.get("FLLM_URL", "http://127.0.0.1:8990") + "/v1"
MODEL = os.environ.get("FLLM_MODEL", "test-model")

client = OpenAI(api_key=os.environ.get("FLLM_API_KEY", "sk-test"), base_url=BASE)

fails = []


def check(name, fn):
    try:
        fn()
    except Exception as e:
        fails.append(name)
        print(f"FAIL {name}: {type(e).__name__}: {e}")
    else:
        print(f"ok {name}")


def t_models():
    ms = client.models.list()
    ids = [m.id for m in ms.data]
    assert MODEL in ids, f"model {MODEL} not in {ids}"


def t_chat():
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=16,
        temperature=0,
    )
    assert r.choices, "no choices"
    c = r.choices[0]
    assert c.message.content is not None and len(c.message.content) > 0, "empty content"
    assert r.usage and r.usage.prompt_tokens > 0
    print("  chat content:", repr(c.message.content[:80]))


def t_chat_stream():
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Count: 1 2"}],
        max_tokens=16,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
    )
    texts = []
    saw_usage = False
    for chunk in stream:
        if chunk.usage:
            saw_usage = True
        for ch in chunk.choices:
            if ch.delta and ch.delta.content:
                texts.append(ch.delta.content)
    assert texts, "stream produced no text"
    assert saw_usage, "stream missing usage chunk"
    print("  stream text:", repr("".join(texts)[:80]))


def t_completions():
    r = client.completions.create(
        model=MODEL, prompt="The capital of France is", max_tokens=8, temperature=0
    )
    assert r.choices and len(r.choices[0].text) > 0, "empty completion"
    print("  completion:", repr(r.choices[0].text[:80]))


def t_embeddings():
    r = client.embeddings.create(model=MODEL, input=["hello", "world"])
    assert len(r.data) == 2, f"expected 2 embeddings, got {len(r.data)}"
    assert len(r.data[0].embedding) > 16, "embedding suspiciously short"
    print("  embedding dim:", len(r.data[0].embedding))


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
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Weather in Berlin?"}],
        max_tokens=64,
        temperature=0,
        tools=tools,
        tool_choice="auto",
    )
    m = r.choices[0].message
    if m.tool_calls:
        for tc in m.tool_calls:
            assert tc.function.name == "get_weather", tc.function.name
            json.loads(tc.function.arguments)  # must be JSON string
        print("  tool called:", len(m.tool_calls))
    else:
        assert m.content, "neither content nor tool_calls"
        print("  tool not called, content ok")


def t_json_schema():
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
        "additionalProperties": False,
    }
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Give me a person."}],
        max_tokens=64,
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "person", "schema": schema, "strict": True},
        },
    )
    content = r.choices[0].message.content
    obj = json.loads(content)
    assert "name" in obj and "age" in obj, f"schema keys missing: {content!r}"
    print("  json_schema:", repr(content[:100]))


def t_logprobs():
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=8,
        temperature=0,
        logprobs=True,
        top_logprobs=2,
    )
    assert r.choices[0].logprobs is not None, "logprobs missing"
    print("  logprobs ok")


def t_responses():
    r = client.responses.create(
        model=MODEL, input="Reply with exactly: OK", max_output_tokens=16, temperature=0
    )
    assert r.output, "empty response output"
    texts = []
    for item in r.output:
        if getattr(item, "type", "") == "message":
            for part in item.content:
                if getattr(part, "text", None):
                    texts.append(part.text)
    assert "".join(texts).strip(), "responses output has no text"
    print("  responses:", repr("".join(texts)[:80]))


for name, fn in [
    ("models", t_models),
    ("chat", t_chat),
    ("chat-stream", t_chat_stream),
    ("completions", t_completions),
    ("embeddings", t_embeddings),
    ("tools-shape", t_tools_shape),
    ("json-schema", t_json_schema),
    ("logprobs", t_logprobs),
    ("responses", t_responses),
]:
    check(name, fn)

if fails:
    print(f"{len(fails)} FAILURES: {fails}")
    sys.exit(1)
print("openai_client: all ok")
