//! Row 77 of PARITY.md: *Sub-agents* — what the page's `runSubAgent` decides before a
//! sub-agent's loop starts (which agent, what it may use, what it is told, how long it may
//! work), what it makes of the loop's answer, the two tool definitions (`task` and
//! `create_agent`) and the token estimates the loop's budget is read from.
//!
//! The loop itself — a second conversation with its own stream — lives in `code.rs`; this file
//! is the part that can be compared with the page's own code case by case
//! (`tests/subagent_fixtures.rs`).

use serde::Deserialize;
use serde_json::{json, Value};

/// `CHILD_MAX_STEPS` of the page: the model calls a sub-agent may take.
pub const CHILD_MAX_STEPS: usize = 16;
/// `MAX_PARALLEL_CHILDREN`: how many `task` calls of one turn run at the same time.
pub const MAX_PARALLEL_CHILDREN: usize = 4;
/// The tools that change something (`WRITE_TOOLS` in `runSubAgent`): a custom agent that is
/// given none of them is read-only.
pub const WRITE_TOOLS: [&str; 7] = ["write_file", "edit_file", "multi_edit", "patch", "bash", "bash_kill", "memory"];

/// One custom agent of `GET /api/studio/agents` (`agents/*.md` of the folder and of the user).
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct CustomAgent {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub tools: Vec<String>,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub body: String,
    #[serde(default)]
    pub max_steps: Option<f64>,
    #[serde(default)]
    pub scope: String,
}

/// `code.agents = data.agents` of `loadAgents`.
pub fn parse_agents(v: &Value) -> Vec<CustomAgent> {
    v.get("agents")
        .and_then(Value::as_array)
        .map(|rows| rows.iter().filter_map(|row| serde_json::from_value(row.clone()).ok()).collect())
        .unwrap_or_default()
}

/// `agentNames()`: the two built-in types, then the custom ones.
pub fn agent_names(custom: &[CustomAgent]) -> Vec<String> {
    ["explore".to_string(), "general".to_string()].into_iter().chain(custom.iter().map(|a| a.name.clone())).collect()
}

const SUB_COMMON: [&str; 3] = [
    "You are a sub-agent. The main agent gave you the job below and cannot see this conversation, only your final reply.",
    "You cannot ask the user questions or start other agents. Work on your own with as few tool calls as it takes.",
    "Your last message is the report: findings first, with file paths and line numbers, no filler, no repetition of the job.",
];
const SUB_EXPLORE: [&str; 2] = [
    "Type: explore. You may only read and search (glob, grep, list_dir, read_file); changes and commands are refused.",
    "Search wide first (several greps in different words), then read the few files that matter. Stop as soon as you can answer. If the answer is not in the code, say so plainly.",
];
const SUB_GENERAL: [&str; 2] = [
    "Type: general. You may also change files and run commands; the user is asked for permission as usual.",
    "Do exactly the job and no more. Check your work (build, tests or diagnostics) and report what you changed (paths) and the result of the check.",
];

/// `subAgentPrompt(args, type)` of `codeprompts.js`.
pub fn sub_agent_prompt(description: &str, prompt: &str, general: bool) -> String {
    let kind: &[&str] = if general { &SUB_GENERAL } else { &SUB_EXPLORE };
    let mut parts: Vec<&str> = SUB_COMMON.iter().chain(kind.iter()).copied().collect();
    let job = format!("Job: {description}");
    parts.extend(["", job.as_str(), "", prompt]);
    parts.join("\n")
}

/// `String(value || fallback)`: what the page reads of an argument the model may have written
/// as anything (a number for a text, nothing at all).
pub(crate) fn js_text(v: Option<&Value>, fallback: &str) -> String {
    match v {
        Some(Value::String(s)) if !s.is_empty() => s.clone(),
        Some(Value::Number(n)) if n.as_f64().is_some_and(|f| f != 0.0 && !f.is_nan()) => crate::specs::index_text(n.as_f64().unwrap_or_default()),
        Some(Value::Bool(true)) => "true".to_string(),
        Some(Value::Array(items)) => items.iter().map(|i| js_text(Some(i), "")).collect::<Vec<_>>().join(","),
        Some(Value::Object(_)) => "[object Object]".to_string(),
        _ => fallback.to_string(),
    }
}

/// What the page decided before the loop starts.
#[derive(Debug, Clone, PartialEq)]
pub struct ChildPlan {
    /// `item.agentType`: `explore`, `general` or the custom agent's name.
    pub agent_type: String,
    /// The loop refuses every tool that is not a plain read (`ctx.readOnly`).
    pub read_only: bool,
    /// The tools a custom agent was given (`None`: all of the parent's).
    pub allowed: Option<Vec<String>>,
    /// What follows the parent's system text (`\n\n[sub-agent "name"]\nbody` of a custom agent).
    pub system_suffix: String,
    /// The first message of the child's conversation.
    pub first_message: String,
    pub max_steps: usize,
    /// Another model for this agent (`custom.model`), when it names one.
    pub model: Option<String>,
    /// The words of the card and of the reply (cut at eighty).
    pub description: String,
}

/// A task that is refused before it starts: the status and text of the card and what the model
/// is told.
#[derive(Debug, Clone, PartialEq)]
pub struct Refusal {
    /// `error` or `denied` (the colour of the card).
    pub status: &'static str,
    pub output: String,
    pub reply: String,
}

/// The first half of `runSubAgent`: from the arguments of the call to the plan of the loop.
/// `plan_mode` is the parent's mode (`code.perm === 'plan'`), `depth` the depth of the agent
/// that calls (0 for the conversation).
pub fn plan_child(args: &Value, custom: &[CustomAgent], plan_mode: bool, depth: u32) -> Result<ChildPlan, Refusal> {
    let prompt = js_text(args.get("prompt"), "");
    if prompt.trim().is_empty() {
        return Err(Refusal { status: "error", output: "No prompt.".into(), reply: "Error: task needs a prompt.".into() });
    }
    if depth > 0 {
        return Err(Refusal {
            status: "denied",
            output: "Sub-agents cannot start sub-agents.".into(),
            reply: "Denied: sub-agents cannot start sub-agents.".into(),
        });
    }
    let description: String = js_text(args.get("description"), "Sub-task").chars().take(80).collect();
    let wanted = js_text(args.get("agent"), "").trim().to_string();
    let found = if wanted.is_empty() { None } else { custom.iter().find(|a| a.name == wanted) };
    if !wanted.is_empty() && found.is_none() && wanted != "explore" && wanted != "general" {
        let names = agent_names(custom).join(", ");
        return Err(Refusal {
            status: "error",
            output: format!("There is no sub-agent \"{wanted}\". Known: {names}."),
            reply: format!("Error: there is no sub-agent \"{wanted}\". Known: {names}."),
        });
    }
    let allowed = found.filter(|a| !a.tools.is_empty()).map(|a| a.tools.clone());
    let raw_agent = args.get("agent").and_then(Value::as_str).unwrap_or_default();
    let wants_general = raw_agent == "general" || args.get("mode").and_then(Value::as_str) == Some("full");
    let agent_type = match found {
        Some(a) => a.name.clone(),
        None if wants_general && !plan_mode => "general".to_string(),
        None => "explore".to_string(),
    };
    let read_only = match found {
        Some(_) => allowed.as_ref().is_none_or(|tools| !tools.iter().any(|t| WRITE_TOOLS.contains(&t.as_str()))),
        None => agent_type == "explore",
    };
    let first_message = match found {
        Some(a) => {
            let about = if a.description.is_empty() { String::new() } else { format!(" — {}", a.description) };
            format!("Sub-agent \"{}\"{about}\n\n{prompt}", a.name)
        }
        None => sub_agent_prompt(&description, &prompt, agent_type == "general"),
    };
    Ok(ChildPlan {
        system_suffix: found.map(|a| format!("\n\n[sub-agent \"{}\"]\n{}", a.name, a.body)).unwrap_or_default(),
        max_steps: found.and_then(|a| a.max_steps).filter(|n| *n >= 1.0).map_or(CHILD_MAX_STEPS, |n| n as usize),
        model: found.map(|a| a.model.clone()).filter(|m| !m.is_empty()),
        agent_type,
        read_only,
        allowed,
        first_message,
        description,
    })
}

/// What the card and the model get when the loop is over (`runSubAgent`, its second half): the
/// status of the card (`ok`, `denied` when the conversation was stopped, `error`), its output,
/// and the reply — clipped like every tool answer.
pub fn child_outcome(result: Result<String, ChildFailure>, stopped: bool, description: &str, ctx_max: usize) -> (&'static str, String, String) {
    let (status, output) = match result {
        Ok(report) => (
            if stopped { "denied" } else { "ok" },
            if report.is_empty() { "(the sub-agent wrote no report)".to_string() } else { report },
        ),
        Err(ChildFailure::Stopped) => ("denied", "Stopped.".to_string()),
        Err(ChildFailure::Error(message)) => ("error", message),
    };
    let reply = if status == "ok" { format!("Sub-agent report ({description}):\n{output}") } else { format!("Sub-agent failed: {output}") };
    (status, output, crate::review::clip_output(&reply, ctx_max))
}

/// Why a sub-agent's loop ended without a report.
#[derive(Debug, Clone, PartialEq)]
pub enum ChildFailure {
    /// The conversation was stopped while it worked (`AbortError`).
    Stopped,
    /// The model could not be asked (`e.message`).
    Error(String),
}

/// `Math.ceil(String(text || '').length / 3.6)`: the page counts UTF-16 units.
pub fn estimate_tokens(text: &str) -> usize {
    (text.encode_utf16().count() as f64 / 3.6).ceil() as usize
}

/// `estimateMessages` of the page for text messages: the content, the calls as JSON (an
/// absent list counts as the two characters of `""`) and six for the frame.
pub fn estimate_messages(messages: &[crate::api::WireMessage]) -> usize {
    messages
        .iter()
        .map(|m| {
            let calls = m.tool_calls.as_ref().map_or_else(|| "\"\"".to_string(), Value::to_string);
            estimate_tokens(&m.content) + estimate_tokens(&calls) + 6
        })
        .sum()
}

/// The condition of the child loop's wrap-up (`runLoop`): the last step, or a conversation that
/// no longer fits the budget — the model is then asked for its report and its calls are ignored.
pub fn needs_wrap_up(step: usize, max_steps: usize, fixed_tokens: usize, message_tokens: usize, budget: usize) -> bool {
    step + 1 == max_steps || fixed_tokens + message_tokens > budget
}

/// The words the child is told when it must stop.
pub const WRAP_UP_MESSAGE: &str = "Stop investigating now. Write your final report with what you found so far, without calling tools.";

/// `ctx.budget` of a child: three quarters of the model's context.
pub fn child_budget(ctx_max: usize) -> usize {
    ctx_max * 3 / 4
}

/// What the page's `runToolCall` refuses a sub-agent before it asks about anything: a tool of
/// the page itself, the screen tools, and (for a read-only agent) every tool that is not a plain
/// read. `(card text, reply to the model)`.
pub fn child_denial(name: &str, client: bool, computer: bool, plain_read: bool, read_only: bool) -> Option<(&'static str, String)> {
    if client {
        return Some((
            "Not available to sub-agents.",
            format!("Denied: {name} is not available to a sub-agent. Do the work yourself and put the answer in your report."),
        ));
    }
    if computer {
        return Some(("Not available to sub-agents.", "Denied: a sub-agent may not use the screen, mouse or keyboard. Do that work yourself.".to_string()));
    }
    if read_only && !plain_read {
        return Some((
            "This sub-agent is read-only.",
            "Denied: this sub-agent is read-only. Describe the change in your report instead.".to_string(),
        ));
    }
    None
}

/// `TOOL_LABEL.task` of the page: the verb of the card (`Sub-agent (explore)`) and the words
/// of the job.
pub fn task_label(args: &Value) -> (String, String) {
    let agent = js_text(args.get("agent"), "");
    let kind = if !agent.is_empty() {
        agent
    } else if args.get("mode").and_then(Value::as_str) == Some("full") {
        "general".to_string()
    } else {
        "explore".to_string()
    };
    (format!("Sub-agent ({kind})"), js_text(args.get("description"), ""))
}

/// The `task` tool as the page lists it for a folder with custom agents (`toolListFor`): the
/// agent argument names them all and says what each is for.
pub fn task_with_agents(def: &Value, custom: &[CustomAgent]) -> Value {
    if custom.is_empty() {
        return def.clone();
    }
    let extra = custom
        .iter()
        .map(|a| if a.description.is_empty() { a.name.clone() } else { format!("{} ({})", a.name, a.description) })
        .collect::<Vec<_>>()
        .join("; ");
    let mut out = def.clone();
    if let Some(agent) = out.pointer_mut("/function/parameters/properties/agent") {
        agent["enum"] = json!(agent_names(custom));
        agent["description"] = json!(format!(
            "explore (default): read and search only. general: may also edit and run commands. Custom agents of this folder: {extra}."
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The words of the prompt of a built-in type, joined as the page joins them.
    #[test]
    fn the_prompt_of_a_type_is_the_pages_text() {
        let text = sub_agent_prompt("Find it", "Where?", false);
        assert!(text.starts_with("You are a sub-agent."));
        assert!(text.contains("Type: explore."));
        assert!(text.ends_with("\nJob: Find it\n\nWhere?"));
        assert!(sub_agent_prompt("x", "y", true).contains("Type: general."));
    }

    /// The budget, the wrap-up and the counts of tokens.
    #[test]
    fn the_budget_and_the_wrap_up() {
        assert_eq!(child_budget(8192), 6144);
        assert!(needs_wrap_up(15, 16, 0, 0, 100), "the last step");
        assert!(!needs_wrap_up(14, 16, 0, 0, 100));
        assert!(needs_wrap_up(0, 16, 60, 41, 100), "over the budget");
        assert!(!needs_wrap_up(0, 16, 60, 40, 100), "exactly the budget is still room");
        assert!(needs_wrap_up(0, 1, 0, 0, 100), "one step is the last one at once");
    }
}
