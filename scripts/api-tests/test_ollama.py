"""The official Ollama Python client against fllm-rs."""
import ollama
import pytest


@pytest.fixture(scope="module")
def client(base_url, auth_headers):
    return ollama.Client(host=base_url, headers=auth_headers, timeout=60)


def test_list_reports_size_digest_and_quantization(client, model):
    entry = next(m for m in client.list().models if m.model == model)
    assert entry.size > 0
    assert len(entry.digest) == 64
    assert entry.details.quantization_level
    assert entry.details.parameter_size


def test_chat(client, model):
    reply = client.chat(model=model, messages=[{"role": "user", "content": "hi"}],
                        options={"temperature": 0, "num_predict": 4})
    assert reply.done and reply.message.role == "assistant"


def test_chat_streaming(client, model):
    parts = list(client.chat(model=model, messages=[{"role": "user", "content": "hi"}], stream=True,
                             options={"temperature": 0, "num_predict": 4}))
    assert parts[-1].done and not any(p.done for p in parts[:-1])


def test_generate_and_keep_alive(client, model):
    reply = client.generate(model=model, prompt="abc", options={"temperature": 0, "num_predict": 3}, keep_alive="10m")
    assert reply.done and isinstance(reply.response, str)
    assert client.ps().models[0].model == model


def test_embed_truncates(client, model):
    reply = client.embed(model=model, input=["y" * 400, "short"])
    assert len(reply.embeddings) == 2 and len(reply.embeddings[0]) == 64


def test_show(client, model):
    info = client.show(model)
    assert info.modelinfo["general.architecture"] == "llama"


def test_unknown_model_is_a_response_error(client):
    with pytest.raises(ollama.ResponseError) as caught:
        client.chat(model="no-such-model", messages=[{"role": "user", "content": "hi"}])
    assert caught.value.status_code == 404
