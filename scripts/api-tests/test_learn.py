"""The learning routes of the Studio (`/api/studio/learn/...`): skill revisions, memory with sources, the profile of the user.

They work on files of the Studio folder of the server under test and put back what they change, so run them against a server
with its own folder (FLLM_HOME=<temp dir>) rather than against your everyday one. The routes answer only to this machine.
"""
import pytest
import requests


@pytest.fixture(scope="module")
def studio(base_url, auth_headers):
    def call(method, path, **kw):
        r = requests.request(method, base_url + "/api/studio/learn" + path, headers=auth_headers, timeout=60, **kw)
        return r

    if call("GET", "").status_code == 404:
        pytest.skip("this server has no learning routes")
    return call


def test_overview_counts_what_waits(studio):
    r = studio("GET", "")
    assert r.status_code == 200
    body = r.json()
    for key in ("skills", "memory", "profile", "pending", "profile_in_prompt", "running"):
        assert key in body
    assert body["pending"] == len(body["skills"]) + len(body["memory"]) + len(body["profile"])


def test_profile_round_trip_refuses_secrets_and_is_put_back(studio):
    before = studio("GET", "/profile").json()
    try:
        assert studio("POST", "/profile", json={"action": "save", "text": "## Languages\n- Writes Rust\n"}).status_code == 200
        got = studio("GET", "/profile").json()
        assert "Writes Rust" in got["text"] and got["max_chars"] == 3000
        refused = studio("POST", "/profile", json={"action": "save", "text": "- password = hunter2hunter2"})
        assert refused.status_code == 400
        assert "secret" in refused.json()["error"]["message"]
        assert studio("POST", "/profile", json={"action": "save", "text": "x" * 3001}).status_code == 400
        assert studio("POST", "/profile", json={"action": "bogus"}).status_code == 400
    finally:
        if before["text"].strip():
            studio("POST", "/profile", json={"action": "save", "text": before["text"]})
        else:
            studio("POST", "/profile", json={"action": "delete"})


def test_the_profile_goes_into_the_instructions_of_the_agent(studio, base_url, auth_headers):
    before = studio("GET", "/profile").json()
    try:
        studio("POST", "/profile", json={"action": "save", "text": "## Preferences\n- Short answers\n"})
        studio("POST", "/profile", json={"action": "prompt", "on": True})
        r = requests.post(base_url + "/api/studio/agent/instructions", headers=auth_headers, json={}, timeout=60)
        if r.status_code == 403:
            pytest.skip("the agent is switched off on this server")
        assert r.status_code == 200
        assert "Short answers" in r.json()["profile"]
        studio("POST", "/profile", json={"action": "prompt", "on": False})
        assert requests.post(base_url + "/api/studio/agent/instructions", headers=auth_headers, json={}, timeout=60).json()["profile"] == ""
    finally:
        studio("POST", "/profile", json={"action": "prompt", "on": before["in_prompt"]})
        if before["text"].strip():
            studio("POST", "/profile", json={"action": "save", "text": before["text"]})
        else:
            studio("POST", "/profile", json={"action": "delete"})


def test_memory_entries_are_added_listed_edited_and_deleted(studio):
    text = "Zebra stripes are checked by the api test suite only"
    assert studio("POST", "/memory", json={"action": "add", "scope": "global", "text": text}).status_code == 200
    try:
        listed = studio("GET", "/memory").json()
        rows = [e for e in listed["global"] if e["text"] == text]
        assert rows and rows[0]["source"] is None, "an entry written by hand has no source session"
        assert studio("POST", "/memory", json={"action": "add", "scope": "global", "text": text}).status_code == 400, "a repeat is refused"
        i = rows[0]["index"]
        assert studio("POST", "/memory", json={"action": "edit", "scope": "global", "index": i, "text": text + " (edited)"}).status_code == 200
        assert any(e["text"].endswith("(edited)") for e in studio("GET", "/memory").json()["global"])
    finally:
        for e in studio("GET", "/memory").json()["global"]:
            if e["text"].startswith(text):
                studio("POST", "/memory", json={"action": "delete", "scope": "global", "index": e["index"]})
    assert not any(e["text"].startswith(text) for e in studio("GET", "/memory").json()["global"])


def test_memory_with_a_secret_is_refused(studio):
    r = studio("POST", "/memory", json={"action": "add", "scope": "global", "text": "The staging key is sk-abcdefghijklmnopqrstuvwx1234"})
    assert r.status_code == 400


def test_bad_requests_get_clear_errors(studio):
    assert studio("POST", "/skills", json={"action": "approve", "name": "../x"}).status_code == 400
    assert studio("POST", "/skills", json={"action": "approve", "name": "no-such-skill-here"}).status_code == 400
    assert studio("POST", "/feedback", json={"value": "meh", "skills": []}).status_code == 400
    assert studio("POST", "/after-run", json={"messages": []}).status_code == 400
    assert studio("POST", "/memory", json={"action": "accept", "id": "mem-00000000"}).status_code == 400
    assert studio("GET", "/skills").status_code == 200


def test_a_session_is_learned_from_in_the_background(studio):
    messages = [
        {"role": "user", "content": "run the tests and tell me the result"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "bash", "arguments": "{\"command\":\"cargo test\"}"}},
                                                         {"id": "2", "function": {"name": "bash", "arguments": "{\"command\":\"cargo clippy\"}"}}]},
        {"role": "tool", "content": "ok"},
        {"role": "tool", "content": "ok"},
        {"role": "assistant", "content": "All tests pass and clippy is clean."},
    ]
    r = studio("POST", "/after-run", json={"session": "api-test", "outcome": "done", "messages": messages})
    assert r.status_code == 202
    assert "queued" in r.json()


def test_learning_is_on_by_default_and_the_switches_work(studio):
    before = studio("GET", "").json()
    assert before["enabled"] in (True, False) and "ask_first" in before and before["notice"]
    try:
        assert studio("POST", "/settings", json={"enabled": False}).status_code == 200
        assert studio("GET", "").json()["enabled"] is False
        r = studio("POST", "/after-run", json={"session": "off", "messages": [{"role": "user", "content": "hello there"}]})
        assert r.status_code == 202 and r.json()["queued"] is False and "off" in r.json()["reason"]
        assert studio("POST", "/settings", json={"enabled": True, "ask_first": True}).status_code == 200
        o = studio("GET", "").json()
        assert o["enabled"] is True and o["ask_first"] is True
    finally:
        studio("POST", "/settings", json={"enabled": before["enabled_global"], "ask_first": before["ask_first"]})


def test_the_log_of_automatic_changes_and_undo(studio):
    assert studio("GET", "/log?limit=5").status_code == 200
    assert isinstance(studio("GET", "/log").json()["log"], list)
    r = studio("POST", "/undo", json={"id": "chg-00000000"})
    assert r.status_code == 400 and "no change" in r.json()["error"]["message"]
