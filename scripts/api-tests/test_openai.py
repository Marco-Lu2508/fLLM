"""The official OpenAI Python SDK against fllm-rs."""
import json
import time

import openai
import pytest
from openai import OpenAI


@pytest.fixture(scope="module")
def client(base_url, api_key):
    return OpenAI(base_url=f"{base_url}/v1", api_key=api_key or "unused", max_retries=0, timeout=60)


def test_models(client, model):
    ids = [m.id for m in client.models.list().data]
    assert model in ids
    assert client.models.retrieve(model).id == model


def test_unknown_model_is_a_404_with_the_openai_error_object(client):
    with pytest.raises(openai.NotFoundError) as caught:
        client.chat.completions.create(model="no-such-model", messages=[{"role": "user", "content": "hi"}])
    body = caught.value.body
    assert body["code"] == "model_not_found"
    assert body["type"] == "invalid_request_error"
    assert caught.value.response.headers["x-request-id"]


def test_chat_completion(client, model):
    reply = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello"}],
        max_tokens=8,
        temperature=0,
        user="sdk-test",
    )
    assert reply.object == "chat.completion"
    assert reply.choices[0].message.role == "assistant"
    assert reply.choices[0].finish_reason in ("stop", "length")
    assert reply.usage.total_tokens == reply.usage.prompt_tokens + reply.usage.completion_tokens
    again = client.chat.completions.create(
        model=model, messages=[{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello"}],
        max_tokens=8, temperature=0,
    )
    assert again.choices[0].message.content == reply.choices[0].message.content


def test_chat_streaming_with_usage(client, model):
    stream = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "hello"}], max_tokens=6, temperature=0,
        stream=True, stream_options={"include_usage": True}, extra_body={"ignore_eos": True},
    )
    chunks = list(stream)
    assert chunks[-1].usage is not None and chunks[-1].usage.completion_tokens == 6
    assert any(c.choices and c.choices[0].finish_reason == "length" for c in chunks)
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert len(text) > 0


def test_default_max_tokens_fills_the_context(client, model):
    reply = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "hello"}], temperature=0, extra_body={"ignore_eos": True},
    )
    assert reply.choices[0].finish_reason == "length"
    assert reply.usage.completion_tokens > 128


def test_text_completion(client, model):
    reply = client.completions.create(model=model, prompt="abc", max_tokens=4, temperature=0)
    assert reply.object == "text_completion"
    assert isinstance(reply.choices[0].text, str)


def test_embeddings(client, model):
    reply = client.embeddings.create(model=model, input=["a", "bc"])
    assert len(reply.data) == 2 and len(reply.data[0].embedding) == 64
    assert reply.usage.prompt_tokens == 3


def test_structured_output(client, model):
    schema = {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "boolean"}},
              "required": ["a", "b"], "additionalProperties": False}
    reply = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "x"}], max_tokens=60,
        response_format={"type": "json_schema", "json_schema": {"name": "t", "schema": schema, "strict": True}},
    )
    value = json.loads(reply.choices[0].message.content)
    assert isinstance(value["a"], int) and isinstance(value["b"], bool)


def test_tool_call_round_trip(client, model):
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "weather",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    # a grammar makes the random model emit a Hermes-format call, which the server must parse
    call = r'start: "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Berlin\"}}\n</tool_call>"'
    reply = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "weather in Berlin?"}], tools=tools, max_tokens=120,
        extra_body={"grammar": call, "grammar_type": "lark"},
    )
    choice = reply.choices[0]
    assert choice.finish_reason == "tool_calls"
    tool_call = choice.message.tool_calls[0]
    assert tool_call.function.name == "get_weather"
    assert json.loads(tool_call.function.arguments) == {"city": "Berlin"}
    followup = client.chat.completions.create(
        model=model, tools=tools, max_tokens=4,
        messages=[
            {"role": "user", "content": "weather in Berlin?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": tool_call.id, "type": "function", "function": tool_call.function.model_dump()}]},
            {"role": "tool", "tool_call_id": tool_call.id, "content": "sunny"},
        ],
    )
    assert followup.choices[0].message.role == "assistant"


def test_responses_api(client, model):
    reply = client.responses.create(model=model, input="hello", max_output_tokens=4, temperature=0)
    assert reply.status in ("completed", "incomplete")  # incomplete: the 4-token limit was reached
    assert client.responses.retrieve(reply.id).id == reply.id


def test_files_and_batches(client, model):
    line = {"custom_id": "one", "method": "POST", "url": "/v1/chat/completions",
            "body": {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3, "temperature": 0}}
    uploaded = client.files.create(file=("batch.jsonl", (json.dumps(line) + "\n").encode()), purpose="batch")
    batch = client.batches.create(input_file_id=uploaded.id, endpoint="/v1/chat/completions", completion_window="24h")
    for _ in range(200):
        batch = client.batches.retrieve(batch.id)
        if batch.status in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert batch.status == "completed"
    assert batch.request_counts.completed == 1
    content = client.files.content(batch.output_file_id).text
    assert json.loads(content.splitlines()[0])["response"]["status_code"] == 200


def test_wrong_key_is_rejected(base_url, api_key):
    if not api_key:
        pytest.skip("the server under test does not require a key")
    wrong = OpenAI(base_url=f"{base_url}/v1", api_key="definitely-wrong", max_retries=0)
    with pytest.raises(openai.AuthenticationError) as caught:
        wrong.models.list()
    assert caught.value.body["code"] == "invalid_api_key"
