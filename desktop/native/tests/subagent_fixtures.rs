//! Row 77: the decisions of the page's `runSubAgent` against the port. The fixtures are the
//! page's own function run in a vm over thirty cases (`tests/fixtures/subagent_fixtures.js`); a
//! case that the page refuses must be refused here with the same card text and the same reply,
//! one it starts must give the loop the same agent, tools, text, limits and model, and what the
//! loop's answer makes of the card and of the reply must match, too.

use fllm_app::api::WireMessage;
use fllm_app::subagents::*;
use serde_json::{json, Value};

fn fixtures() -> Value {
    let raw = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/subagent_fixtures.json")).expect("the fixtures of the page");
    serde_json::from_str(&raw).expect("fixtures as json")
}

fn agents(v: &Value) -> Vec<CustomAgent> {
    parse_agents(&json!({ "agents": v }))
}

#[test]
fn every_task_is_decided_the_way_the_page_decides_it() {
    let data = fixtures();
    let cases = data["cases"].as_array().expect("cases");
    assert!(cases.len() >= 30, "the generator found the function: {} cases", cases.len());
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let custom = agents(&case["agents"]);
        let plan_mode = case["perm"] == "plan";
        let depth = case["depth"].as_u64().unwrap() as u32;
        let planned = plan_child(&case["args"], &custom, plan_mode, depth);
        if case["ran"].is_null() {
            let refusal = planned.expect_err(&format!("{name}: the page refuses it"));
            assert_eq!(refusal.status, case["item"]["status"].as_str().unwrap(), "{name}: the card's colour");
            assert_eq!(refusal.output, case["item"]["output"].as_str().unwrap(), "{name}: the card's text");
            assert_eq!(refusal.reply, case["reply"].as_str().unwrap(), "{name}: the model's reply");
            assert!(case["item"]["agentType"].is_null(), "{name}: a refused task has no type");
            continue;
        }
        let plan = planned.unwrap_or_else(|r| panic!("{name}: the page starts it, the port refuses it: {r:?}"));
        let ran = &case["ran"];
        assert_eq!(Some(plan.agent_type.as_str()), case["item"]["agentType"].as_str(), "{name}: the type");
        assert_eq!(plan.read_only, ran["readOnly"].as_bool().unwrap(), "{name}: read-only");
        let parent: Vec<&str> = case["parentTools"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
        let tools: Vec<&str> = parent.iter().copied().filter(|t| plan.allowed.as_ref().is_none_or(|a| a.iter().any(|x| x == t))).collect();
        let page_tools: Vec<&str> = ran["tools"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
        assert_eq!(tools, page_tools, "{name}: the tools of the loop");
        assert_eq!(format!("SYSTEM{}", plan.system_suffix), ran["system"].as_str().unwrap(), "{name}: the system text");
        assert_eq!(plan.first_message, ran["messages"][0]["content"].as_str().unwrap(), "{name}: the first message");
        assert_eq!(ran["messages"].as_array().unwrap().len(), 1, "{name}: one message to start with");
        assert_eq!(plan.max_steps as u64, ran["maxSteps"].as_u64().unwrap(), "{name}: the steps");
        assert_eq!(plan.model.as_deref(), ran["model"].as_str(), "{name}: the model");
        let ctx = case["ctx"].as_u64().unwrap() as usize;
        assert_eq!(child_budget(ctx) as u64, ran["budget"].as_u64().unwrap(), "{name}: the budget");
        assert_eq!(ran["depth"], 1, "{name}: a sub-agent is one level down");
        assert_eq!(ran["child"], true);

        // what the loop's answer makes of the card and the reply
        let result = match case["loop"]["throws"]["name"].as_str() {
            Some("AbortError") => Err(ChildFailure::Stopped),
            Some(_) => Err(ChildFailure::Error(case["loop"]["throws"]["message"].as_str().unwrap().to_string())),
            None => Ok(case["loop"]["report"].as_str().unwrap_or("REPORT").to_string()),
        };
        let stopped = case["stopped"].as_bool().unwrap_or(false);
        let (status, output, reply) = child_outcome(result, stopped, &plan.description, ctx);
        assert_eq!(status, case["item"]["status"].as_str().unwrap(), "{name}: the status after the loop");
        assert_eq!(output, case["item"]["output"].as_str().unwrap(), "{name}: the output after the loop");
        assert_eq!(reply, case["reply"].as_str().unwrap(), "{name}: the reply after the loop");
    }
}

#[test]
fn the_helpers_count_and_word_like_the_page() {
    let data = fixtures();
    for row in data["estimates"].as_array().unwrap() {
        let text = row["text"].as_str().unwrap();
        assert_eq!(estimate_tokens(text) as u64, row["tokens"].as_u64().unwrap(), "estimateTokens({text:?})");
    }
    for row in data["messages"].as_array().unwrap() {
        let messages: Vec<WireMessage> = row["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| {
                let mut w = WireMessage::new(m["role"].as_str().unwrap(), m["content"].as_str().unwrap());
                w.tool_calls = m.get("tool_calls").cloned();
                w
            })
            .collect();
        assert_eq!(estimate_messages(&messages) as u64, row["tokens"].as_u64().unwrap(), "estimateMessages of {:?}", row["messages"]);
    }
    for row in data["prompts"].as_array().unwrap() {
        let general = row["type"] == "general";
        assert_eq!(sub_agent_prompt(row["description"].as_str().unwrap(), row["prompt"].as_str().unwrap(), general), row["text"].as_str().unwrap());
    }
}

#[test]
fn the_two_tools_are_defined_with_the_pages_words() {
    let data = fixtures();
    let (tools, defs) = fllm_app::code::client_tools();
    for name in ["task", "create_agent"] {
        let tool = tools.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("{name} is offered"));
        let page = &data["toolDefs"][name];
        let want_permission = page["permission"].as_str().unwrap_or("read");
        assert_eq!(fllm_app::code::Permission::parse(want_permission), tool.permission, "{name}: the permission");
        let def = defs.iter().find(|d| d["function"]["name"] == name).unwrap_or_else(|| panic!("{name} has a definition"));
        assert_eq!(def["function"]["description"], page["description"], "{name}: the description");
        assert_eq!(def["function"]["parameters"]["properties"], page["properties"], "{name}: the properties");
        assert_eq!(def["function"]["parameters"]["required"], page["required"], "{name}: what is required");
    }
    // the task tool of a folder with custom agents
    let custom = agents(&data["agents"]);
    let (_, defs) = fllm_app::code::client_tools();
    let task = defs.iter().find(|d| d["function"]["name"] == "task").unwrap();
    assert_eq!(task_with_agents(task, &custom), data["taskWithAgents"], "the page's list for a folder with agents");
    assert_eq!(&task_with_agents(task, &[]), task, "no agents, no change");
}

#[test]
fn the_guard_against_a_stuck_model_counts_like_the_page() {
    use fllm_app::loopguard::*;
    let data = fixtures();
    for row in data["signatures"].as_array().unwrap() {
        assert_eq!(call_signature(row["name"].as_str().unwrap(), &row["args"]), row["signature"].as_str().unwrap(), "the signature of {}", row["args"]);
    }
    for row in data["repeats"].as_array().unwrap() {
        let history: Vec<String> = row["history"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect();
        let times = row["times"].as_u64().unwrap() as usize;
        assert_eq!(is_repeating(&history, row["signature"].as_str().unwrap(), times), row["repeating"].as_bool().unwrap(), "{row}");
    }
    let page: Vec<&str> = data["guardless"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
    assert_eq!(NO_LOOP_GUARD.to_vec(), page, "the tools that are never counted");
}
