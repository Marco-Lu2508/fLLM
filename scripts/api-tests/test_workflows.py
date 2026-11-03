"""The routes behind specs, steering files and the extra hook events (docs/user/agent-workflows.md).

Runs against a live server like the other files here; no model is needed (the routes only read and write files of a temporary
workspace). The server must run on this machine, because the agent routes refuse other machines. Start it with its own FLLM_HOME,
for example `FLLM_HOME=$(mktemp -d) fllm-rs --model tiny.gguf --device cpu`, so that the hook list of your own account is not used.
"""
import pytest
import requests


@pytest.fixture(scope="module")
def api(base_url, auth_headers):
    session = requests.Session()
    session.headers.update(auth_headers)

    def call(path, body=None):
        if body is None:
            return session.get(base_url + path, timeout=60)
        return session.post(base_url + path, json=body, timeout=60)

    return call


@pytest.fixture(scope="module")
def workspace(api, tmp_path_factory):
    root = tmp_path_factory.mktemp("workflows")
    (root / ".fllm" / "steering").mkdir(parents=True)
    (root / ".fllm" / "specs" / "export").mkdir(parents=True)
    (root / ".fllm" / "steering" / "style.md").write_text("Always use tabs.")
    (root / ".fllm" / "steering" / "ts.md").write_text('---\ninclusion: fileMatch\nfileMatchPattern: "**/*.ts"\n---\nUse strict mode.')
    (root / ".fllm" / "steering" / "release.md").write_text("---\ninclusion: manual\ndescription: release steps\n---\nTag first.")
    (root / ".fllm" / "specs" / "export" / "tasks.md").write_text("# Tasks\n- [ ] 1. Parser\n- [ ] 2. CLI\n")
    (root / "a.ts").write_text("let x = 1;\n")
    response = api("/api/studio/agent/workspace", {"path": str(root)})
    assert response.status_code == 200, response.text
    return root


def test_steering_is_part_of_the_instructions_and_a_matching_file_brings_its_own(api, workspace):
    files = api("/api/studio/agent/instructions", {}).json()["files"]
    assert any(f["text"] == "Always use tabs." for f in files)
    index = next(f for f in files if f["file"] == "steering index")
    assert "release.md" in index["text"] and "release steps" in index["text"]
    assert not any(f["text"] == "Use strict mode." for f in files), "a file-matching rule is not in the prompt"
    first = api("/api/studio/agent/tool", {"name": "read_file", "args": {"path": "a.ts"}}).json()
    assert first["ok"] and "Use strict mode." in first["output"]
    again = api("/api/studio/agent/tool", {"name": "read_file", "args": {"path": "a.ts"}}).json()
    assert "[steering]" not in again["output"], "shown once per session"


def test_specs_are_listed_ticked_and_turned_into_prompts(api, workspace):
    spec = api("/api/studio/agent/specs").json()["specs"][0]
    assert (spec["name"], spec["total"], spec["done"]) == ("export", 2, 0)
    ticked = api("/api/studio/agent/specs", {"action": "tick", "name": "export", "index": 2}).json()
    assert ticked["specs"][0]["done"] == 1
    assert "- [x] 2. CLI" in (workspace / ".fllm" / "specs" / "export" / "tasks.md").read_text()
    assert api("/api/studio/agent/specs", {"action": "untick", "name": "export", "index": 2}).json()["specs"][0]["done"] == 0
    assert api("/api/studio/agent/specs", {"action": "tick", "name": "export", "index": 9}).status_code == 400
    prompt = api("/api/studio/agent/specs", {"action": "task_prompt", "name": "export", "index": 1}).json()["prompt"]
    assert "Parser" in prompt and "tick its box" in prompt
    new = api("/api/studio/agent/specs", {"action": "new_prompt", "name": "CSV Export", "description": "export as csv"}).json()
    assert new["name"] == "csv-export" and "requirements.md" in new["prompt"]
    assert api("/api/studio/agent/specs", {"action": "new_prompt", "name": "x"}).status_code == 400


def test_the_hook_route_knows_the_new_events(api, workspace):
    stop = api("/api/studio/agent/hook", {"event": "Stop"})
    assert stop.status_code == 200 and isinstance(stop.json()["blocked"], bool)
    changed = api("/api/studio/agent/hook", {"event": "file_changed", "tool": "docs/a.md", "arguments": {"kind": "modified"}})
    assert changed.status_code == 200
    assert api("/api/studio/agent/hook", {"event": "whatever"}).status_code == 400
    events = api("/api/studio/hooks").json()["events"]
    for name in ("user_prompt_submit", "stop", "subagent_stop", "notification", "pre_compact", "file_changed"):
        assert name in events
