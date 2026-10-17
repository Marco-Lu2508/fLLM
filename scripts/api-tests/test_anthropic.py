"""The official Anthropic Python SDK against fllm-rs (Messages API)."""
import anthropic
import pytest


@pytest.fixture(scope="module")
def client(base_url, api_key):
    return anthropic.Anthropic(base_url=base_url, api_key=api_key or "unused", max_retries=0, timeout=60)


def test_message(client, model):
    message = client.messages.create(
        model=model,
        max_tokens=8,
        extra_body={"temperature": 0},
        system=[{"type": "text", "text": "be brief", "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": [{"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}]}],
    )
    assert message.type == "message" and message.role == "assistant"
    assert message.usage.input_tokens > 0
    assert message.stop_reason in ("end_turn", "max_tokens")


def test_streaming(client, model):
    with client.messages.stream(model=model, max_tokens=6, extra_body={"temperature": 0},
                                messages=[{"role": "user", "content": "hello"}]) as stream:
        text = "".join(stream.text_stream)
        final = stream.get_final_message()
    assert len(text) > 0
    assert final.usage.input_tokens > 0, "message_start reports the real prompt size"
    assert final.usage.output_tokens > 0


def test_count_tokens(client, model):
    counted = client.messages.count_tokens(model=model, messages=[{"role": "user", "content": "hello"}])
    assert counted.input_tokens > 0


def test_unknown_model_is_not_found(client):
    with pytest.raises(anthropic.NotFoundError) as caught:
        client.messages.create(model="claude-nope", max_tokens=4, messages=[{"role": "user", "content": "hi"}])
    assert caught.value.body["error"]["type"] == "not_found_error"


def test_models(client, model):
    assert model in [m.id for m in client.models.list().data]


def test_x_api_key_is_enforced(base_url, api_key):
    if not api_key:
        pytest.skip("the server under test does not require a key")
    wrong = anthropic.Anthropic(base_url=base_url, api_key="definitely-wrong", max_retries=0)
    with pytest.raises(anthropic.AuthenticationError):
        wrong.messages.create(model="tiny", max_tokens=1, messages=[{"role": "user", "content": "hi"}])
