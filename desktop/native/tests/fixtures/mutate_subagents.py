#!/usr/bin/env python3
"""One word at a time: the tests of row 77 must go red.

Covers the decisions before a sub-agent starts (the type, the tools, the words, the limits),
what the loop's answer makes of the card, the limits of the loop (four at a time, the last step),
the refusals of a sub-agent's calls, the cards, the agents route and the kept session. A run is red only when a test
reports `FAILED`; one that does not compile or reports nothing is broken and its tail is printed.
"""
import os
import pathlib
import signal
import subprocess
import sys

# a kill must not leave a mutation in the source: the `finally` below puts every file back
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

ROOT = pathlib.Path(__file__).resolve().parents[2]  # desktop/native
SRC = ROOT / "src"
# the whole environment is kept: rustup finds its toolchain only over RUSTUP_HOME/CARGO_HOME
CARGO_ENV = dict(os.environ)
CARGO_ENV["PATH"] = "/opt/data/home/.cargo/bin:" + CARGO_ENV.get("PATH", "")
# no debug info and its own target directory: the build is what each mutation waits for
CARGO_ENV["CARGO_TARGET_DIR"] = "/opt/data/cache/scratch/fllm-mutation-target"
CARGO_ENV["CARGO_PROFILE_DEV_DEBUG"] = "0"
CARGO_ENV["FLLM_SNAPSHOT_DIR"] = str(ROOT / "ui-snapshots")

# (file, old text, new text, which tests must go red)
MUTATIONS = [
    ('subagents.rs', 'if prompt.trim().is_empty() {', 'if prompt.is_empty() {', 'a blank prompt is no prompt'),
    ('subagents.rs', 'if depth > 0 {', 'if depth > 1 {', 'a sub-agent starts none'),
    ('subagents.rs', '"Error: task needs a prompt."', '"Error: no prompt."', 'the reply to a missing prompt'),
    ('subagents.rs', '"Sub-agents cannot start sub-agents."', '"No."', 'the card of a nested task'),
    ('subagents.rs', '.chars().take(80).collect()', '.chars().take(120).collect()', 'a description is cut at eighty'),
    ('subagents.rs', 'js_text(args.get("agent"), "").trim().to_string()', 'js_text(args.get("agent"), "")', 'the name of the agent is trimmed'),
    ('subagents.rs', '&& wanted != "explore" && wanted != "general" {', '&& wanted != "explore" {', 'general is a known agent'),
    ('subagents.rs', '"There is no sub-agent \\"{wanted}\\". Known: {names}."', '"There is no sub-agent \\"{wanted}\\"."', 'the card of an unknown agent'),
    ('subagents.rs', '.filter(|a| !a.tools.is_empty()).map(|a| a.tools.clone())', '.map(|a| a.tools.clone())', 'an empty tool list is all tools'),
    ('subagents.rs', 'let wants_general = raw_agent == "general" ||', 'let wants_general = raw_agent.trim() == "general" ||', 'a padded general is only explore'),
    ('subagents.rs', 'Some("full");', 'Some("fullest");', 'mode full is general'),
    ('subagents.rs', 'None if wants_general && !plan_mode =>', 'None if wants_general =>', 'plan mode has no general agent'),
    ('subagents.rs', '.any(|t| WRITE_TOOLS.contains(&t.as_str()))', '.any(|t| t == "write_file")', 'every write tool counts'),
    ('subagents.rs', '.is_none_or(|tools| !tools', '.is_some_and(|tools| !tools', 'no tool list is read-only'),
    ('subagents.rs', 'None => agent_type == "explore",', 'None => false,', 'an explore agent is read-only'),
    ('subagents.rs', 'format!(" — {}", a.description)', 'format!(" - {}", a.description)', 'the first message of a custom agent'),
    ('subagents.rs', 'format!("\\n\\n[sub-agent \\"{}\\"]\\n{}", a.name, a.body)', 'format!("\\n\\n{}", a.body)', 'the system text of a custom agent'),
    ('subagents.rs', '.filter(|n| *n >= 1.0).map_or(CHILD_MAX_STEPS,', '.filter(|n| *n >= 1.0).map_or(8,', 'sixteen steps by default'),
    ('subagents.rs', '.filter(|m| !m.is_empty())', '.filter(|_| false)', "a custom agent's model"),
    ('subagents.rs', 'pub const CHILD_MAX_STEPS: usize = 16;', 'pub const CHILD_MAX_STEPS: usize = 15;', 'sixteen steps by default'),
    ('subagents.rs', 'pub const MAX_PARALLEL_CHILDREN: usize = 4;', 'pub const MAX_PARALLEL_CHILDREN: usize = 5;', 'four work at a time'),
    ('subagents.rs', 'ctx_max * 3 / 4', 'ctx_max / 2', 'the context sets the budget'),
    ('subagents.rs', 'if stopped { "denied" } else { "ok" }', '"ok"', 'a stopped loop is denied'),
    ('subagents.rs', '"(the sub-agent wrote no report)"', '"(no report)"', 'an empty report'),
    ('subagents.rs', 'Err(ChildFailure::Stopped) => ("denied", "Stopped.".to_string()),', 'Err(ChildFailure::Stopped) => ("error", "Stopped.".to_string()),', 'the loop is stopped'),
    ('subagents.rs', 'format!("Sub-agent report ({description}):\\n{output}")', 'format!("{output}")', 'the model reads the report'),
    ('subagents.rs', 'format!("Sub-agent failed: {output}")', 'format!("{output}")', 'the loop fails'),
    ('subagents.rs', '(status, output, crate::review::clip_output(&reply, ctx_max))', '(status, output, reply)', 'a long report is clipped'),
    ('subagents.rs', 'step + 1 == max_steps || fixed_tokens + message_tokens > budget', 'step + 2 == max_steps || fixed_tokens + message_tokens > budget', 'the last step'),
    ('subagents.rs', 'fixed_tokens + message_tokens > budget', 'fixed_tokens + message_tokens >= budget', 'exactly the budget is still room'),
    ('subagents.rs', '/ 3.6).ceil()', '/ 3.6).floor()', 'estimateTokens'),
    ('subagents.rs', '+ estimate_tokens(&calls) + 6', '+ estimate_tokens(&calls)', 'estimateMessages'),
    ('subagents.rs', '"Stop investigating now. Write your final report with what you found so far, without calling tools."', '"Stop."', 'the last step asks for the report'),
    ('subagents.rs', 'if read_only && !plain_read {', 'if read_only {', 'an explore agent reads'),
    ('subagents.rs', 'if client {', 'if false {', 'the tools of the window are not for sub-agents'),
    ('subagents.rs', 'if computer {', 'if false {', 'the screen is not for sub-agents'),
    ('subagents.rs', '"This sub-agent is read-only."', '"Read-only."', 'the card of a refusal'),
    ('subagents.rs', 'agent["enum"] = json!(agent_names(custom));', 'agent["enum"] = json!(["explore", "general"]);', "the page's list for a folder with agents"),
    ('subagents.rs', 'format!("Sub-agent ({kind})")', 'format!("Sub-agent {kind}")', 'the verb of the card'),
    ('code.rs', 'let run_now = state == CallState::Running && kind != ToolKind::Task;', 'let run_now = state == CallState::Running;', 'a task runs at once'),
    ('code.rs', 'if parsed.get("scope").and_then(Value::as_str) == Some("user") {', 'if false {', 'for every folder it always asks'),
    ('code.rs', 'if working >= crate::subagents::MAX_PARALLEL_CHILDREN {', 'if working >= 100 {', 'four work at a time'),
    ('code.rs', 'let partial = if kid.wrap_up { Vec::new() } else { partial };', 'let partial = partial;', 'the last step ignores calls'),
    ('code.rs', 'return self.kid_finish(call_id, Ok(content.trim().to_string()), env);', 'return self.kid_finish(call_id, Ok(content), env);', 'the report is trimmed'),
    ('code.rs', 'format!("{call_id}~{step}.{n}")', 'format!("{call_id}~{n}")', 'its calls have ids of their own'),
    ('code.rs', 'let read_only = kid.plan.read_only;', 'let read_only = false;', 'an explore agent is refused every change'),
    ('code.rs', 'kid.step += 1;\n            if kid.step >= kid.plan.max_steps {', 'kid.step += 2;\n            if kid.step >= kid.plan.max_steps {', 'the next step started'),
    ('code.rs', 'm.tool_call_id = Some(call_id);\n                kid.messages.push(m);', 'kid.messages.push(m);', 'the answers carry the ids'),
    ('code.rs', 'self.kids.clear();\n        self.each_call_mut(', 'self.each_call_mut(', 'stopping ends the sub-agents too'),
    ('code.rs', 'self.kid_finish(call_id, Err(crate::subagents::ChildFailure::Error(e)), env),', 'self.kid_finish(call_id, Ok(e), env),', 'a failed request is the failure of the task'),
    ('code.rs', 'self.each_call_mut(|c| {\n                        if c.state == CallState::Pending && c.kind != ToolKind::Ask {', 'self.each_call_mut(|c| {\n                        if false {', 'plan mode refuses what a sub-agent waits to do'),
    ('code.rs', 'if ok && name == "create_agent" {', 'if false {', 'a new agent is offered at once'),
    ('app.rs', 'Task::batch([self.agents_refresh(), self.code.reload_agents(&self.base)])\n            }\n            Message::AgentSaved(_, _, Err(e)) => {', 'self.agents_refresh()\n            }\n            Message::AgentSaved(_, _, Err(e)) => {', 'a new agent is offered at once'),
    ('code.rs', 'CodeMsg::AgentsLoaded(Some(agents)) => {\n                self.agents = agents;', 'CodeMsg::AgentsLoaded(Some(agents)) => {\n                let _ = agents;', 'the task tool names the agents'),
    ('code.rs', 'if let (Some(allowed), Some(list)) = (&plan.allowed, tools.as_array_mut()) {', 'if let (Some(allowed), Some(list)) = (&None::<Vec<String>>, tools.as_array_mut()) {', 'only the tools the agent was given'),
    ('code.rs', 'kid.messages.push(WireMessage::new("user", crate::subagents::WRAP_UP_MESSAGE));', 'let _ = &kid;', 'the last step is the report'),
    ('code.rs', 'if kid.wrap_up { ", writing report" } else { "" }', '""', 'the last step is the report'),
    ('code.rs', 'let system = format!("{}{}", self.system_prompt(env.extra), plan.system_suffix);', 'let system = self.system_prompt(env.extra);', 'its own instructions follow'),
    ('code.rs', 'children,\n        agent: c.kid.as_ref().map(|k| k.agent_type.clone()),', 'children: Vec::new(),\n        agent: c.kid.as_ref().map(|k| k.agent_type.clone()),', 'a session keeps the conversation'),
    ('code.rs', 'agent: c.kid.as_ref().map(|k| k.agent_type.clone()),', 'agent: None,', 'a session keeps the type of the agent'),
    ('code.rs', 'text("Report").size(11)', 'text("Result").size(11)', 'the report has its heading'),
    ('code.rs', 'text("Starting ...")', 'text("...")', 'the one that waits has not begun'),
    ('loopguard.rs', 'is_repeating(history, &signature, 3)', 'is_repeating(history, &signature, 4)', 'the same call three times in a row'),
    ('loopguard.rs', '!NO_LOOP_GUARD.contains(&name) && ', '', 'polling and bookkeeping may repeat'),
    ('loopguard.rs', 'pub const HISTORY: usize = 8;', 'pub const HISTORY: usize = 1;', 'the history keeps eight'),
    ('loopguard.rs', 'keys.sort_by_key(|k| k.encode_utf16().collect::<Vec<u16>>());', 'keys.sort();', 'the keys are sorted by UTF-16 units'),
    ('loopguard.rs', 'indexed.sort_by_key(|k| index_key(k));', '', 'array indices come first'),
    ('loopguard.rs', '} else if -6 < n && n <= 0 {', '} else if -7 < n && n <= 0 {', 'numbers as JavaScript writes them'),
    ('loopguard.rs', 'if k <= n && n <= 21 {', 'if k <= n && n <= 20 {', 'numbers as JavaScript writes them (1e21)'),
    ('loopguard.rs', 'if history.len() < times.saturating_sub(1) {', 'if history.len() < times {', 'a history of two is enough'),
    ('code.rs', 'if crate::loopguard::check(&mut self.history, &p.name, &args) {', 'if false && crate::loopguard::check(&mut self.history, &p.name, &args) {', 'the same call three times in a row is refused'),
    ('code.rs', 'if crate::loopguard::check(&mut self.kids[i].history, &p.name, &args) {', 'if false && crate::loopguard::check(&mut self.kids[i].history, &p.name, &args) {', 'a sub-agent has a history of its own'),
    ('code.rs', 'if crate::loopguard::check(&mut self.kids[i].history, &p.name, &args) {', 'if crate::loopguard::check(&mut self.history, &p.name, &args) {', 'a sub-agent has a history of its own (shared)'),
    ('code.rs', 'self.error = None;\n        self.steps = 0;\n        self.history.clear();', 'self.error = None;\n        self.steps = 0;', 'a new run forgets what was called'),
]

TEST = ["cargo", "test", "--locked", "--no-fail-fast", "--lib", "--test", "subagent_fixtures", "--test", "subagents_window", "--test", "subagents_server"]


def run() -> tuple[bool, list[str], str]:
    out = subprocess.run(TEST, cwd=ROOT, env=CARGO_ENV, capture_output=True, text=True)
    text = out.stdout + out.stderr
    failed = [line.split()[1] for line in text.splitlines() if line.startswith("test ") and line.endswith("FAILED")]
    ran = "test result:" in text
    return (ran and bool(failed)) or ("test result: FAILED" in text), failed, text


if len(sys.argv) > 1 and sys.argv[1] == "--check":
    bad = 0
    for file, old, new, _ in MUTATIONS:
        count = (SRC / file).read_text().count(old)
        if count != 1:
            bad += 1
            print(f"{file}: {count} matches for {old[:70]!r}")
    print("anchors ok" if not bad else f"{bad} bad anchors")
    sys.exit(1 if bad else 0)

# `ONLY=words python3 mutate_specs.py` runs the entries whose test words contain `words`
ONLY = os.environ.get("ONLY")
if ONLY:
    MUTATIONS[:] = [m for m in MUTATIONS if ONLY in m[3]]

original: dict[str, str] = {}
red: list[str] = []
problems: list[str] = []
try:
    for file, old, new, must in MUTATIONS:
        path = SRC / file
        text = path.read_text()
        count = text.count(old)
        if count != 1:
            problems.append(f"{file}: {count} matches for {old[:60]!r} — fix the script (check the FILE named first)")
            continue
        original[file] = text
        path.write_text(text.replace(old, new, 1))
        try:
            is_red, failed, tail = run()
        finally:
            path.write_text(original[file])
        if is_red:
            red.append(must)
            print(f"RED   {must}: {sorted(set(failed))[:3]}", flush=True)
        elif "test result:" not in tail:
            problems.append(f"BROKEN {must}")
            print(f"BROKEN {must} — it did not compile or ran nothing; tail:\n" + "\n".join(tail.splitlines()[-12:]), flush=True)
        else:
            problems.append(f"GREEN {must}")
            print(f"GREEN {must}", flush=True)
finally:
    for file, text in original.items():
        (SRC / file).write_text(text)

print(f"{len(red)}/{len(MUTATIONS)} mutations red", flush=True)
if problems:
    print("PROBLEMS:", problems, flush=True)
    sys.exit(1)
print("ALL MUTATIONS BIT", flush=True)
