"""Python client for `fllm-rs agent --rpc`: drive the fLLM coding agent from your own program.

No dependencies beyond the standard library. The agent runs as a child process and talks JSON-RPC 2.0, one message per line, over its
standard input and output (protocol: docs/user/agent-cli.md, "Embedding the agent"). A server with a model that can call tools must be
running (`fllm-rs serve ...`).

    from fllm_agent import Agent

    with Agent(url="http://127.0.0.1:8000") as agent:
        session = agent.new_session("/path/to/project", allow=["edit"])
        result = session.prompt("Fix the failing test in tests/test_parse.py",
                                on_event=lambda e: print(e["type"], e.get("name") or e.get("text", "")[:60]),
                                approve=lambda call: call["tool"] == "bash" and call["args"]["command"].startswith("pytest"))
        print(result["status"], result["answer"])

`approve(call)` is asked for every action the session's `allow` list does not cover; `call` has `tool`, `category`, `args`. Return True to
allow it once, or a dict `{"allow": True, "always": True}` to allow that category for the rest of the session. Without `approve`, such
actions are refused (the model is told so).
"""
import json
import subprocess
import threading


class AgentError(Exception):
    """The agent answered a request with a JSON-RPC error."""

    def __init__(self, code, message):
        super().__init__(f"{message} (code {code})")
        self.code = code
        self.message = message


class Agent:
    def __init__(self, exe="fllm-rs", url=None, api_key=None, model=None, extra_args=()):
        args = [exe, "agent", "--rpc"]
        if url:
            args += ["--url", url]
        if api_key:
            args += ["--api-key", api_key]
        if model:
            args += ["--model", model]
        args += list(extra_args)
        self._proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
                                      text=True, encoding="utf-8", bufsize=1)
        self._next_id = 1
        self._write_lock = threading.Lock()
        self.info = self.request("initialize")

    # -- plumbing ---------------------------------------------------------------------------------------
    def _send(self, message):
        with self._write_lock:
            self._proc.stdin.write(json.dumps(message) + "\n")
            self._proc.stdin.flush()

    def request(self, method, params=None, on_message=None):
        """Sends a request and returns its result; `on_message(msg)` sees every other message that arrives meanwhile."""
        rid = self._next_id
        self._next_id += 1
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise AgentError(-32000, "the agent process ended")
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            if msg.get("id") == rid and "method" not in msg:
                if "error" in msg:
                    raise AgentError(msg["error"].get("code", -32603), msg["error"].get("message", "error"))
                return msg.get("result")
            if on_message:
                on_message(msg)

    def notify(self, method, params=None):
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def new_session(self, folder, allow=(), plan=False, model=None, system=None, max_steps=None,
                    allow_rules=(), deny_rules=(), ask_rules=()):
        """`folder` must be an absolute path. `allow`: categories beyond reading: edit, command, network, memory, or "all"."""
        params = {"folder": folder, "allow": list(allow), "plan": plan,
                  "allow_rules": list(allow_rules), "deny_rules": list(deny_rules), "ask_rules": list(ask_rules)}
        if model:
            params["model"] = model
        if system:
            params["system"] = system
        if max_steps:
            params["max_steps"] = max_steps
        result = self.request("session/new", params)
        return Session(self, result["session"])

    def tools(self):
        return self.request("tools/list")["tools"]

    def close(self):
        if self._proc.poll() is None:
            try:
                self.request("shutdown")
            except (AgentError, OSError, ValueError):
                pass
            try:
                self._proc.stdin.close()
            except OSError:
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Session:
    def __init__(self, agent, session_id):
        self.agent = agent
        self.id = session_id

    def prompt(self, text, on_event=None, approve=None):
        """Runs one task. Returns {answer, status, steps, edited, tool_calls}."""
        def handle(msg):
            method = msg.get("method")
            if method == "session/event" and on_event:
                on_event(msg["params"])
            elif method == "session/approve":
                verdict = approve(msg["params"]) if approve else False
                if isinstance(verdict, dict):
                    result = {"allow": bool(verdict.get("allow", False)), "always": bool(verdict.get("always", False))}
                else:
                    result = {"allow": bool(verdict)}
                self.agent._send({"jsonrpc": "2.0", "id": msg["id"], "result": result})
        return self.agent.request("session/prompt", {"session": self.id, "text": text}, on_message=handle)

    def cancel(self):
        """Ends the running prompt (call from another thread)."""
        self.agent.notify("session/cancel", {"session": self.id})

    def history(self):
        return self.agent.request("session/history", {"session": self.id})["messages"]

    def close(self):
        self.agent.request("session/close", {"session": self.id})
