//! A mailbox between sessions of one workspace. Chats live in the browser, so one session cannot talk to another one that is
//! closed — but it can leave a handover note: what was tried, what is left, which file matters. The note waits in
//! `messages.json` of the workspace, the Studio shows it as soon as that chat is opened, and the user decides whether it goes
//! into the composer. That is the honest version of "cross-session messaging": no background loop pretends to work on it.
use super::agent::{agent_disabled, disabled_reply, AgentSession};
use super::*;
use serde_json::{json, Value};
use axum::extract::Query;
use std::collections::HashMap;
use std::path::PathBuf;

const MAX_TEXT_CHARS: usize = 4000;
const MAX_MESSAGES: usize = 200;

fn file(root: &std::path::Path) -> PathBuf {
    root.join(".fllm").join("messages.json")
}

fn read_all(root: &std::path::Path) -> Vec<Value> {
    let Ok(text) = std::fs::read_to_string(file(root)) else { return Vec::new() };
    let Ok(value) = serde_json::from_str::<Value>(&text) else { return Vec::new() };
    value.get("messages").and_then(Value::as_array).cloned().unwrap_or_default()
}

fn write_all(root: &std::path::Path, messages: &[Value]) -> Result<(), String> {
    let path = file(root);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }
    let text = serde_json::to_string_pretty(&json!({"messages": messages})).map_err(|e| e.to_string())?;
    std::fs::write(&path, text).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

fn next_id(messages: &[Value]) -> u64 {
    messages.iter().filter_map(|m| m.get("id").and_then(Value::as_u64)).max().unwrap_or(0) + 1
}

/// Adds a note; older notes beyond the limit are dropped first (the oldest read ones).
pub(super) fn add(root: &std::path::Path, from: &str, to: &str, text: &str) -> Result<Value, String> {
    let text = text.trim();
    if text.is_empty() {
        return Err("the note is empty".to_owned());
    }
    let mut messages = read_all(root);
    let id = next_id(&messages);
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    messages.push(json!({
        "id": id,
        "from": from.trim().chars().take(80).collect::<String>(),
        "to": to.trim().chars().take(80).collect::<String>(),
        "text": text.chars().take(MAX_TEXT_CHARS).collect::<String>(),
        "at": now,
    }));
    while messages.len() > MAX_MESSAGES {
        messages.remove(0);
    }
    write_all(root, &messages)?;
    Ok(json!({"id": id, "count": messages.len()}))
}

/// The notes that belong to one chat: addressed to it, to "all" or to nobody in particular.
pub(super) fn for_chat(root: &std::path::Path, chat: &str) -> Vec<Value> {
    read_all(root)
        .into_iter()
        .filter(|m| {
            let to = m.get("to").and_then(Value::as_str).unwrap_or("");
            to.is_empty() || to == "all" || to == chat
        })
        .collect()
}

/// Marks notes as done; unknown ids are ignored.
pub(super) fn dismiss(root: &std::path::Path, ids: &[u64]) -> Result<Value, String> {
    let mut messages = read_all(root);
    let before = messages.len();
    messages.retain(|m| !m.get("id").and_then(Value::as_u64).is_some_and(|id| ids.contains(&id)));
    write_all(root, &messages)?;
    Ok(json!({"removed": before - messages.len(), "count": messages.len()}))
}

/// The tool: `{text, to?}` writes a note, `{list: true}` shows what is waiting.
pub(super) fn tool(session: &AgentSession, args: &Value) -> Result<String, String> {
    let root = session.root_dir()?;
    if args.get("list").and_then(Value::as_bool).unwrap_or(false) {
        let messages = read_all(&root);
        if messages.is_empty() {
            return Ok("There are no notes from other sessions.".to_owned());
        }
        let mut out = format!("{} note(s) in this workspace:\n", messages.len());
        for item in messages.iter().take(50) {
            out.push_str(&format!(
                "- #{} from {}: {}\n",
                item.get("id").and_then(Value::as_u64).unwrap_or(0),
                item.get("from").and_then(Value::as_str).unwrap_or(""),
                item.get("text").and_then(Value::as_str).unwrap_or("")
            ));
        }
        return Ok(out);
    }
    let text = args.get("text").and_then(Value::as_str).unwrap_or("");
    let to = args.get("to").and_then(Value::as_str).unwrap_or("");
    let from = root.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "a session".to_owned());
    let value = add(&root, &from, to, text)?;
    Ok(format!("Left note #{} for the next session. It shows up in the Studio as soon as a chat is opened.", value["id"]))
}

/// `GET /api/studio/agent/messages?chat=`: the notes that wait for this chat.
pub(in crate::server) async fn route_list(State(state): State<Arc<AppState>>, Query(query): Query<HashMap<String, String>>) -> Response {
    if agent_disabled() {
        return disabled_reply();
    }
    let chat = query.get("chat").cloned().unwrap_or_default();
    let worker = state.clone();
    match tokio::task::spawn_blocking(move || -> Result<Value, String> {
        // without a chosen folder there are no notes (an empty list, not an error: the page may ask before a folder is chosen)
        let Ok(root) = worker.agent.root_dir() else { return Ok(json!({"messages": []})) };
        Ok(json!({"messages": for_chat(&root, &chat)}))
    })
    .await
    {
        Ok(Ok(value)) => Json(value).into_response(),
        Ok(Err(message)) => api_error(StatusCode::BAD_REQUEST, &message),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

/// `POST /api/studio/agent/messages` `{chat, from, to, text}`.
pub(in crate::server) async fn route_add(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    if agent_disabled() {
        return disabled_reply();
    }
    let from = body.get("from").and_then(Value::as_str).unwrap_or("").to_owned();
    let to = body.get("to").and_then(Value::as_str).unwrap_or("").to_owned();
    let text = body.get("text").and_then(Value::as_str).unwrap_or("").to_owned();
    let worker = state.clone();
    match tokio::task::spawn_blocking(move || -> Result<Value, String> {
        let root = worker.agent.root_dir()?;
        add(&root, &from, &to, &text)
    })
    .await
    {
        Ok(Ok(value)) => Json(value).into_response(),
        Ok(Err(message)) => api_error(StatusCode::BAD_REQUEST, &message),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

/// `DELETE /api/studio/agent/messages` `{ids: [1, 2]}`.
pub(in crate::server) async fn route_dismiss(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    if agent_disabled() {
        return disabled_reply();
    }
    let ids: Vec<u64> = body
        .get("ids")
        .and_then(Value::as_array)
        .map(|items| items.iter().filter_map(Value::as_u64).collect())
        .unwrap_or_default();
    if ids.is_empty() {
        return api_error(StatusCode::BAD_REQUEST, "give the ids of the notes to remove");
    }
    let worker = state.clone();
    match tokio::task::spawn_blocking(move || -> Result<Value, String> {
        let root = worker.agent.root_dir()?;
        dismiss(&root, &ids)
    })
    .await
    {
        Ok(Ok(value)) => Json(value).into_response(),
        Ok(Err(message)) => api_error(StatusCode::BAD_REQUEST, &message),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fllm-msg-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_note_is_written_and_read_back() {
        let root = temp_root("add");
        let value = add(&root, "parser-chat", "all", "  the fix waits in worktree fix-parser  ").unwrap();
        assert_eq!(value["id"], 1);
        let messages = for_chat(&root, "any-chat");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["text"], "the fix waits in worktree fix-parser");
        assert_eq!(messages[0]["from"], "parser-chat");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_note_for_one_chat_stays_out_of_the_others() {
        let root = temp_root("scope");
        add(&root, "a", "chat-1", "nur für chat 1").unwrap();
        assert_eq!(for_chat(&root, "chat-1").len(), 1);
        assert_eq!(for_chat(&root, "chat-2").len(), 0);
        assert_eq!(for_chat(&root, "chat-2").len(), 0);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn an_empty_note_is_refused() {
        let root = temp_root("empty");
        assert!(add(&root, "a", "", "   ").unwrap_err().contains("empty"));
        assert_eq!(for_chat(&root, "x").len(), 0);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn dismissing_removes_only_the_given_notes() {
        let root = temp_root("dismiss");
        add(&root, "a", "", "eins").unwrap();
        add(&root, "a", "", "zwei").unwrap();
        let value = dismiss(&root, &[1, 99]).unwrap();
        assert_eq!(value["removed"], 1);
        let rest = for_chat(&root, "x");
        assert_eq!(rest.len(), 1);
        assert_eq!(rest[0]["text"], "zwei");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_oldest_notes_fall_out_of_a_full_mailbox() {
        let root = temp_root("full");
        for i in 0..MAX_MESSAGES + 5 {
            add(&root, "a", "", &format!("note {i}")).unwrap();
        }
        let messages = for_chat(&root, "x");
        assert_eq!(messages.len(), MAX_MESSAGES);
        assert_eq!(messages[0]["text"], "note 5", "the oldest ones are gone");
        let _ = std::fs::remove_dir_all(&root);
    }
}
