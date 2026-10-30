"""What coding agents (Claude Code, Codex, OpenCode, Aider, Continue ...) depend on, checked with the official SDKs.

Runs against a live server like the other files here. The tests that need a model to call a tool are skipped unless
FLLM_TOOL_MODEL=1 (the served model must follow tool calls, e.g. a Qwen2.5-Coder or Qwen3 instruct model); everything
else works on the tiny test model.
"""
import json
import os

import anthropic
import openai
import pytest
import requests

TOOL_MODEL = os.environ.get("FLLM_TOOL_MODEL") == "1"
needs_tools = pytest.mark.skipif(not TOOL_MODEL, reason="set FLLM_TOOL_MODEL=1 with a tool-capable model")

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather of a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


@pytest.fixture(scope="module")
def oa(base_url, api_key):
    return openai.OpenAI(base_url=base_url + "/v1", api_key=api_key or "unused", max_retries=0, timeout=600)


@pytest.fixture(scope="module")
def an(base_url, api_key):
    return anthropic.Anthropic(base_url=base_url, api_key=api_key or "unused", max_retries=0, timeout=600)


def test_streaming_usage_chunk_comes_last_with_no_choices(oa, model):
    chunks = list(oa.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "hi"}], max_tokens=4, stream=True,
        stream_options={"include_usage": True}))
    assert chunks[-1].choices == [] and chunks[-1].usage.completion_tokens > 0
    assert chunks[-2].choices[0].finish_reason in ("stop", "length")
    assert all(c.usage is None for c in chunks[:-1])


def test_context_overflow_is_the_openai_error(oa, model):
    with pytest.raises(openai.BadRequestError) as caught:
        oa.chat.completions.create(model=model, messages=[{"role": "user", "content": "hi"}], max_tokens=10_000_000)
    assert caught.value.code == "context_length_exceeded"


def test_agent_dialects_cut_a_big_output_ceiling_to_the_context(an, oa, model):
    # Claude Code asks for 32000 tokens, Codex and OpenCode for 32000 or more: a small server must still answer
    message = an.messages.create(model=model, max_tokens=1_000_000, extra_body={"temperature": 0},
                                 messages=[{"role": "user", "content": "hi"}])
    assert message.stop_reason in ("end_turn", "max_tokens")
    reply = oa.responses.create(model=model, input="hi", max_output_tokens=1_000_000)
    assert reply.status in ("completed", "incomplete")


def test_claude_model_names_are_answered_by_the_loaded_model(an):
    message = an.messages.create(model="claude-haiku-4-5", max_tokens=4, messages=[{"role": "user", "content": "hi"}])
    assert message.type == "message"


def test_claude_code_shaped_request(base_url, api_key, model):
    """System entries inside `messages`, a billing block in `system` and adaptive thinking, as Claude Code 2.1 sends them."""
    headers = {"x-api-key": api_key or "unused", "anthropic-version": "2023-06-01"}
    body = {
        "model": model, "max_tokens": 8, "thinking": {"type": "adaptive", "display": "omitted"},
        "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1; cch=0;"},
                   {"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": "hi"},
                     {"role": "system", "content": [{"type": "text", "text": "Platform: linux"}]}],
        "context_management": {"edits": []}, "output_config": {"effort": "high"},
    }
    reply = requests.post(base_url + "/v1/messages?beta=true", json=body, headers=headers, timeout=300)
    assert reply.status_code == 200, reply.text
    assert reply.json()["type"] == "message"


def test_anthropic_errors_and_key_header(base_url, api_key):
    reply = requests.post(base_url + "/v1/messages", json={"model": "x", "max_tokens": 1, "messages": []},
                          headers={"x-api-key": api_key or "unused", "anthropic-version": "2023-06-01"}, timeout=60)
    assert reply.status_code == 400
    assert reply.json()["type"] == "error" and reply.json()["error"]["type"] == "invalid_request_error"
    if api_key:
        denied = requests.post(base_url + "/v1/messages", json={}, headers={"x-api-key": "wrong"}, timeout=60)
        assert denied.status_code == 401 and denied.json()["error"]["type"] == "authentication_error"


def test_cors_preflight(base_url):
    reply = requests.options(base_url + "/v1/chat/completions", headers={
        "Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization,content-type,x-api-key,anthropic-version"}, timeout=60)
    assert reply.status_code in (200, 204, 403)


@needs_tools
def test_forced_tool_call_roundtrip_openai(oa, model):
    first = oa.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "Weather in Paris?"}], tools=[WEATHER],
        tool_choice={"type": "function", "function": {"name": "get_weather"}}, max_tokens=120, temperature=0)
    choice = first.choices[0]
    assert choice.finish_reason == "tool_calls"
    call = choice.message.tool_calls[0]
    assert call.type == "function" and call.id and json.loads(call.function.arguments)["city"]
    second = oa.chat.completions.create(model=model, tools=[WEATHER], max_tokens=40, temperature=0, messages=[
        {"role": "user", "content": "Weather in Paris?"}, choice.message.model_dump(exclude_none=True),
        {"role": "tool", "tool_call_id": call.id, "content": "sunny, 21 C"}])
    assert second.choices[0].message.content


@needs_tools
def test_streamed_tool_call_assembles_like_openai(oa, model):
    stream = oa.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "Weather in Paris?"}], tools=[WEATHER],
        tool_choice={"type": "function", "function": {"name": "get_weather"}}, max_tokens=120, temperature=0, stream=True)
    name, args, ident, finish = "", "", None, None
    for chunk in stream:
        for choice in chunk.choices:
            finish = choice.finish_reason or finish
            for call in choice.delta.tool_calls or []:
                assert call.index == 0
                ident = call.id or ident
                name += call.function.name or ""
                args += call.function.arguments or ""
    assert (name, finish) == ("get_weather", "tool_calls") and ident
    assert "city" in json.loads(args)


@needs_tools
def test_anthropic_tool_use_streaming_and_result_turn(an, model):
    tools = [{"name": "get_weather", "description": "Weather of a city",
              "input_schema": WEATHER["function"]["parameters"]}]
    with an.messages.stream(model=model, max_tokens=120, extra_body={"temperature": 0}, tools=tools,
                            tool_choice={"type": "tool", "name": "get_weather"},
                            messages=[{"role": "user", "content": "Weather in Paris?"}]) as stream:
        message = stream.get_final_message()
    assert message.stop_reason == "tool_use"
    use = next(b for b in message.content if b.type == "tool_use")
    assert use.name == "get_weather" and use.input.get("city")
    follow = an.messages.create(model=model, max_tokens=40, tools=tools, extra_body={"temperature": 0}, messages=[
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in message.content]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": use.id, "content": "sunny, 21 C"}]}])
    assert follow.content and follow.usage.input_tokens > 0


@needs_tools
def test_responses_function_call_item_and_stream_events(oa, model):
    tools = [{"type": "function", "name": "get_weather", "description": "w",
              "parameters": WEATHER["function"]["parameters"]}]
    reply = oa.responses.create(model=model, input="Weather in Paris?", tools=tools,
                                tool_choice={"type": "function", "name": "get_weather"}, max_output_tokens=120, temperature=0)
    item = next(i for i in reply.output if i.type == "function_call")
    assert item.name == "get_weather" and json.loads(item.arguments)["city"] and item.call_id
    kinds = [e.type for e in oa.responses.create(model=model, input="Weather in Paris?", tools=tools, stream=True,
                                                tool_choice={"type": "function", "name": "get_weather"},
                                                max_output_tokens=120, temperature=0)]
    assert kinds[0] == "response.created" and kinds[-1] == "response.completed"
    assert "response.function_call_arguments.done" in kinds
