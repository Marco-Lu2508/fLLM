//! The specs of the open workspace (`.fllm/specs/<name>/`) for the Code side panel: the list with the tasks of each, ticking a
//! task by hand, and the prompts that start a spec or one of its tasks.
use super::agent::{agent_disabled, disabled_reply};
use super::*;

fn no_workspace() -> Response {
    api_error(StatusCode::BAD_REQUEST, "no workspace folder is chosen yet; choose one first")
}

/// `GET /api/studio/agent/specs`: every spec with its files and tasks.
pub(in crate::server) async fn list(State(state): State<Arc<AppState>>) -> Response {
    if agent_disabled() {
        return disabled_reply();
    }
    let Some(root) = state.agent.current_root() else { return Json(json!({"specs": []})).into_response() };
    Json(crate::spec::list_json(&root)).into_response()
}

/// `POST /api/studio/agent/specs`: `{action: "tick" | "untick", name, index}` changes a box; `{action: "task_prompt", name, index}` and
/// `{action: "new_prompt", name, description}` return the prompt the page sends to the agent.
pub(in crate::server) async fn act(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    if agent_disabled() {
        return disabled_reply();
    }
    let Some(root) = state.agent.current_root() else { return no_workspace() };
    let text = |key: &str| body.get(key).and_then(Value::as_str).unwrap_or("").to_owned();
    let index = body.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
    let name = match crate::spec::slug(&text("name")) {
        Ok(n) => n,
        Err(e) => return api_error(StatusCode::BAD_REQUEST, &e),
    };
    match text("action").as_str() {
        "tick" | "untick" => match crate::spec::tick(&root, &name, index, text("action") == "tick") {
            Ok(()) => Json(crate::spec::list_json(&root)).into_response(),
            Err(e) => api_error(StatusCode::BAD_REQUEST, &e),
        },
        "task_prompt" => match crate::spec::studio_task_prompt(&root, &name, index) {
            Ok(prompt) => Json(json!({"prompt": prompt})).into_response(),
            Err(e) => api_error(StatusCode::BAD_REQUEST, &e),
        },
        "new_prompt" => {
            let description = text("description");
            if description.trim().is_empty() {
                return api_error(StatusCode::BAD_REQUEST, "describe what you want: /spec NAME what you want");
            }
            Json(json!({"prompt": crate::spec::studio_new_prompt(&name, &description), "name": name})).into_response()
        }
        other => api_error(StatusCode::BAD_REQUEST, &format!("action must be tick, untick, task_prompt or new_prompt, not '{other}'")),
    }
}
