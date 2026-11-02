//! The routes behind the learning pages of the Studio: what the agent learned from sessions and what still waits for a decision.
//! Skill revisions (diff, approve, reject, rollback, usage), memory entries with their source session, the profile of the user,
//! and the reflective step that runs in the background after a Code session. The store and the steps are in `learn` and `learn_flow`
//! (shared with `fllm-rs learn`); this file only carries requests to them. All of it stays on this machine (see `http::is_personal_path`).
use super::*;
use axum::extract::Query;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Longest a reflective model call may take in the background.
const JOB_SECONDS: u64 = 240;
/// A session sent for reflection is cut to this many messages.
const MAX_MESSAGES: usize = 300;

static BUSY: AtomicBool = AtomicBool::new(false);

fn studio() -> PathBuf {
    crate::agent_cli::studio_folder()
}

fn workspace_of(text: Option<&str>) -> Option<PathBuf> {
    text.map(str::trim).filter(|w| !w.is_empty()).map(PathBuf::from).filter(|p| p.is_absolute() && p.is_dir())
}

fn bad(message: &str) -> Response {
    api_error(StatusCode::BAD_REQUEST, message)
}

async fn blocking<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> Result<T, Box<Response>> {
    tokio::task::spawn_blocking(f).await.map_err(|e| Box::new(api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string())))
}

fn skill_of(body: &Value) -> Result<crate::learn::SkillRef, String> {
    crate::learn::SkillRef::from_parts(body["name"].as_str().unwrap_or_default(), body["scope"].as_str().unwrap_or("user"), body["workspace"].as_str().unwrap_or_default())
}

/// `GET /api/studio/learn`: what waits for a decision, the last background run, and whether one is running now.
pub(in crate::server) async fn overview() -> Response {
    match blocking(|| crate::learn::overview(&studio())).await {
        Ok(mut o) => {
            o["running"] = json!(BUSY.load(Ordering::SeqCst));
            Json(o).into_response()
        }
        Err(r) => *r,
    }
}

/// `GET /api/studio/learn/skills?workspace=`: every skill with its usage numbers, versions and the revision that waits (with diff).
pub(in crate::server) async fn skills(Query(query): Query<HashMap<String, String>>) -> Response {
    let ws = workspace_of(query.get("workspace").map(String::as_str));
    match blocking(move || {
        let studio = studio();
        let mut list = crate::learn_flow::skill_overview(&studio, ws.as_deref());
        for s in &mut list {
            let Ok(r) = skill_of(s) else { continue };
            if let Some(rev) = crate::learn::pending_revision(&studio, &r) {
                s["revision"] = json!({"reason": rev.reason, "trigger": rev.trigger, "created": rev.created, "diff": crate::learn::diff(&rev.old, &rev.new), "new": rev.new});
            }
            let usage: Vec<Value> = crate::learn::read_usage(&studio, &r)
                .iter()
                .rev()
                .take(8)
                .map(|u| json!({"t": u.t, "session": u.session, "outcome": u.outcome, "verify": u.verify, "retries": u.retries, "feedback": u.feedback, "note": u.note}))
                .collect();
            s["recent"] = json!(usage);
        }
        list
    })
    .await
    {
        Ok(list) => Json(json!({"skills": list})).into_response(),
        Err(r) => *r,
    }
}

/// `POST /api/studio/learn/skills` `{action: approve | reject | rollback | improve, name, scope?, workspace?, version?, force?}`.
pub(in crate::server) async fn skills_action(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    let skill = match skill_of(&body) {
        Ok(s) => s,
        Err(e) => return bad(&e),
    };
    let action = body["action"].as_str().unwrap_or_default().to_owned();
    if action == "improve" {
        return match start_job(state, Job::Improve { skill }) {
            Ok(()) => (StatusCode::ACCEPTED, Json(json!({"queued": true}))).into_response(),
            Err(e) => api_error(StatusCode::CONFLICT, &e),
        };
    }
    let (version, force) = (body["version"].as_u64().map(|v| v as u32), body["force"].as_bool().unwrap_or(false));
    match blocking(move || {
        let studio = studio();
        match action.as_str() {
            "approve" => crate::learn::approve_revision(&studio, &skill, force).map(|v| json!({"ok": true, "version": v})),
            "reject" => crate::learn::reject_revision(&studio, &skill).map(|()| json!({"ok": true})),
            "rollback" => crate::learn::rollback(&studio, &skill, version).map(|v| json!({"ok": true, "version": v})),
            other => Err(format!("unknown action '{other}'")),
        }
    })
    .await
    {
        Ok(Ok(v)) => Json(v).into_response(),
        Ok(Err(e)) => bad(&e),
        Err(r) => *r,
    }
}

/// `POST /api/studio/learn/feedback` `{value: up | down, skills: [name], session?, workspace?}`: a thumb for the skills of a session.
pub(in crate::server) async fn feedback(Json(body): Json<Value>) -> Response {
    let names: Vec<String> = body["skills"].as_array().into_iter().flatten().filter_map(|n| n.as_str().map(str::to_owned)).take(20).collect();
    let value = body["value"].as_str().unwrap_or_default().to_owned();
    let session = body["session"].as_str().unwrap_or("studio").chars().take(80).collect::<String>();
    let ws = workspace_of(body["workspace"].as_str());
    match blocking(move || crate::learn_flow::record_feedback(&studio(), ws.as_deref(), &session, &names, &value)).await {
        Ok(Ok(n)) => Json(json!({"ok": true, "noted": n})).into_response(),
        Ok(Err(e)) => bad(&e),
        Err(r) => *r,
    }
}

/// `GET /api/studio/learn/memory?workspace=`: the entries of both memory files with their source session, and the proposals.
pub(in crate::server) async fn memory(Query(query): Query<HashMap<String, String>>) -> Response {
    let ws = workspace_of(query.get("workspace").map(String::as_str));
    match blocking(move || {
        let studio = studio();
        let sources = crate::learn::memory_sources(&studio);
        let rows = |scope: &str| -> Vec<Value> {
            crate::learn::read_memory(&studio, scope, ws.as_deref())
                .iter()
                .enumerate()
                .map(|(i, e)| json!({"index": i, "text": e, "source": sources.get(&crate::learn::normalize(e))}))
                .collect()
        };
        let proposals: Vec<Value> = crate::learn::memory_proposals(&studio)
            .iter()
            .map(|p| json!({"id": p.id, "action": p.action, "scope": p.scope, "workspace": p.workspace, "text": p.text, "kind": p.kind, "session": p.session, "created": p.created, "reason": p.reason}))
            .collect();
        json!({
            "global": rows("global"),
            "workspace": if ws.is_some() { rows("workspace") } else { Vec::new() },
            "proposals": proposals,
            "limits": {"entries": crate::learn::MEMORY_MAX_ENTRIES, "entry_chars": crate::learn::MEMORY_ENTRY_CHARS},
        })
    })
    .await
    {
        Ok(v) => Json(v).into_response(),
        Err(r) => *r,
    }
}

/// `POST /api/studio/learn/memory` `{action: accept | reject | edit | delete | add | forget, id?, scope?, workspace?, index?, text?}`.
pub(in crate::server) async fn memory_action(Json(body): Json<Value>) -> Response {
    let action = body["action"].as_str().unwrap_or_default().to_owned();
    let id = body["id"].as_str().unwrap_or_default().to_owned();
    let scope = body["scope"].as_str().unwrap_or("global").to_owned();
    let text = body["text"].as_str().unwrap_or_default().to_owned();
    let index = body["index"].as_u64().map(|i| i as usize);
    let ws = workspace_of(body["workspace"].as_str());
    match blocking(move || {
        let studio = studio();
        match action.as_str() {
            "accept" => crate::learn::decide_memory(&studio, &id, true),
            "reject" => crate::learn::decide_memory(&studio, &id, false),
            "edit" | "delete" => {
                let i = index.ok_or("'index' is missing")?;
                let new = if action == "delete" { "" } else { text.as_str() };
                crate::learn::edit_memory(&studio, &scope, ws.as_deref(), i, new).map(|()| "saved".to_owned())
            }
            "add" => crate::learn::add_memory(&studio, &scope, ws.as_deref(), &text).map(|()| "saved".to_owned()),
            "forget" => crate::learn::forget(&studio, ws.as_deref(), &text, body["all_learned"].as_bool().unwrap_or(false)).map(|g| format!("forgot {} item(s)", g.len())),
            other => Err(format!("unknown action '{other}'")),
        }
    })
    .await
    {
        Ok(Ok(m)) => Json(json!({"ok": true, "message": m})).into_response(),
        Ok(Err(e)) => bad(&e),
        Err(r) => *r,
    }
}

/// `GET /api/studio/learn/profile`: the profile text, whether it goes into the prompt, the lines waiting to join it.
pub(in crate::server) async fn profile() -> Response {
    match blocking(|| {
        let studio = studio();
        let proposals: Vec<Value> = crate::learn::profile_proposals(&studio).iter().map(|p| json!({"id": p.id, "section": p.section, "text": p.text, "session": p.session, "created": p.created})).collect();
        json!({
            "text": crate::learn::read_profile(&studio),
            "in_prompt": crate::learn::profile_in_prompt(&studio),
            "proposals": proposals,
            "max_chars": crate::learn::PROFILE_FILE_CHARS,
            "sections": crate::learn::PROFILE_SECTIONS.iter().map(|(k, t)| json!({"key": k, "title": t})).collect::<Vec<_>>(),
        })
    })
    .await
    {
        Ok(v) => Json(v).into_response(),
        Err(r) => *r,
    }
}

/// `POST /api/studio/learn/profile` `{action: save | delete | accept | reject | prompt, text?, id?, on?}`.
pub(in crate::server) async fn profile_action(Json(body): Json<Value>) -> Response {
    let action = body["action"].as_str().unwrap_or_default().to_owned();
    let text = body["text"].as_str().unwrap_or_default().to_owned();
    let id = body["id"].as_str().unwrap_or_default().to_owned();
    let on = body["on"].as_bool().unwrap_or(true);
    match blocking(move || {
        let studio = studio();
        match action.as_str() {
            "save" => crate::learn::save_profile_text(&studio, &text).map(|()| "saved".to_owned()),
            "delete" => crate::learn::delete_profile(&studio).map(|()| "deleted".to_owned()),
            "accept" => crate::learn::decide_profile(&studio, &id, true),
            "reject" => crate::learn::decide_profile(&studio, &id, false),
            "prompt" => crate::learn::set_profile_in_prompt(&studio, on).map(|()| "saved".to_owned()),
            other => Err(format!("unknown action '{other}'")),
        }
    })
    .await
    {
        Ok(Ok(m)) => Json(json!({"ok": true, "message": m})).into_response(),
        Ok(Err(e)) => bad(&e),
        Err(r) => *r,
    }
}

// ---------------------------------------------------------------------------------------------------------------
// background jobs: the reflective steps run in the server, after the answer is delivered
// ---------------------------------------------------------------------------------------------------------------

enum Job {
    AfterRun { info: crate::learn_flow::RunInfo, messages: Vec<Value>, auto_memory: bool },
    Improve { skill: crate::learn::SkillRef },
    Consolidate { workspace: Option<PathBuf> },
}

struct Release;
impl Drop for Release {
    fn drop(&mut self) {
        BUSY.store(false, Ordering::SeqCst);
    }
}

/// Starts `job` on its own thread; one job at a time. The model is the one loaded in this server.
fn start_job(state: Arc<AppState>, job: Job) -> Result<(), String> {
    if state.model_name.is_empty() {
        return Err("no model is loaded; the learning steps need one".into());
    }
    if BUSY.swap(true, Ordering::SeqCst) {
        return Err("a learning step is running already".into());
    }
    let handle = tokio::runtime::Handle::current();
    let spawned = std::thread::Builder::new().name("fllm-learn".into()).spawn(move || {
        let _release = Release;
        let mut complete = |mut body: Value| -> Result<Value, String> {
            body["model"] = json!(state.model_name);
            body["stream"] = json!(false);
            handle.block_on(async {
                match tokio::time::timeout(Duration::from_secs(JOB_SECONDS), async {
                    let response = chat_completions(State(state.clone()), Json(body)).await;
                    read_json(response).await.map_err(|(_, message)| message)
                })
                .await
                {
                    Ok(r) => r,
                    Err(_) => Err(format!("the model needed more than {JOB_SECONDS} seconds")),
                }
            })
        };
        let studio = studio();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match job {
            Job::AfterRun { info, messages, auto_memory } => {
                let opts = crate::learn_flow::Options { auto_memory, ..crate::learn_flow::Options::default() };
                let lines = crate::learn_flow::after_run(&studio, &info, &messages, &opts, &mut complete);
                crate::learn_flow::save_last_run(&studio, &info.session, &lines);
            }
            Job::Improve { skill } => {
                let trigger = crate::learn::improve_due(&studio, &skill).unwrap_or_else(|| "requested by hand".to_owned());
                let line = match crate::learn_flow::improve_skill(&studio, &skill, &trigger, "", &mut complete) {
                    Ok(true) => format!("drafted a revision of skill '{}' ({trigger})", skill.name),
                    Ok(false) => format!("skill '{}' reviewed: the model found nothing to change", skill.name),
                    Err(e) => format!("skill '{}': no revision ({e})", skill.name),
                };
                crate::learn_flow::save_last_run(&studio, "improve", &[line]);
            }
            Job::Consolidate { workspace } => {
                crate::learn_flow::consolidate(&studio, workspace.as_deref(), &mut complete);
            }
        }));
        if result.is_err() {
            crate::learn_flow::save_last_run(&studio, "error", &["the learning step stopped unexpectedly".to_owned()]);
        }
    });
    if spawned.is_err() {
        BUSY.store(false, Ordering::SeqCst);
        return Err("cannot start a thread".into());
    }
    Ok(())
}

fn clean_messages(raw: &Value) -> Vec<Value> {
    let mut out = Vec::new();
    for m in raw.as_array().into_iter().flatten().rev().take(MAX_MESSAGES).collect::<Vec<_>>().into_iter().rev() {
        let role = m["role"].as_str().unwrap_or_default();
        if !matches!(role, "user" | "assistant" | "tool") {
            continue;
        }
        let mut c = json!({"role": role});
        c["content"] = match &m["content"] {
            Value::String(s) => json!(s.chars().take(4000).collect::<String>()),
            Value::Array(parts) => json!(parts.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join("").chars().take(4000).collect::<String>()),
            _ => json!(""),
        };
        if let Some(calls) = m["tool_calls"].as_array() {
            c["tool_calls"] = json!(calls
                .iter()
                .map(|call| json!({"id": call["id"], "function": {"name": call["function"]["name"], "arguments": match &call["function"]["arguments"] {
                    Value::String(s) => json!(s.chars().take(800).collect::<String>()),
                    other => json!(other.to_string().chars().take(800).collect::<String>()),
                }}}))
                .collect::<Vec<_>>());
        }
        out.push(c);
    }
    out
}

/// `POST /api/studio/learn/after-run` `{session, workspace?, outcome, messages, skills?, retries?, verify?, note?, auto_memory?}`:
/// returns at once (202); the skill notes, revision drafts and the memory and profile reflection run in the background.
pub(in crate::server) async fn after_run(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    let messages = clean_messages(&body["messages"]);
    if messages.is_empty() {
        return bad("there are no messages to learn from");
    }
    let skills: Vec<String> = body["skills"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|n| n.as_str())
        .filter(|n| crate::agent_cli::valid_session_name(n))
        .map(str::to_owned)
        .take(20)
        .collect();
    let info = crate::learn_flow::RunInfo {
        session: body["session"].as_str().unwrap_or("studio").chars().take(80).collect(),
        workspace: workspace_of(body["workspace"].as_str()),
        outcome: if body["outcome"].as_str() == Some("failed") { "failed" } else { "done" }.to_owned(),
        verify: body["verify"].as_bool(),
        retries: body["retries"].as_u64().unwrap_or(0).min(1000) as u32,
        note: body["note"].as_str().unwrap_or_default().chars().take(400).collect(),
        skills,
        matched: Vec::new(),
        feedback: body["feedback"].as_str().filter(|f| matches!(*f, "up" | "down")).map(str::to_owned),
    };
    let auto_memory = body["auto_memory"].as_bool().unwrap_or(false);
    // the usage of skills is written even when no model is loaded: that part needs none
    if state.model_name.is_empty() {
        let used = blocking({
            let info = info.clone();
            move || crate::learn_flow::record_skill_use(&studio(), &info).len()
        })
        .await
        .unwrap_or(0);
        return (StatusCode::ACCEPTED, Json(json!({"queued": false, "reason": "no model is loaded", "skills_noted": used}))).into_response();
    }
    match start_job(state, Job::AfterRun { info, messages, auto_memory }) {
        Ok(()) => (StatusCode::ACCEPTED, Json(json!({"queued": true}))).into_response(),
        Err(e) => (StatusCode::ACCEPTED, Json(json!({"queued": false, "reason": e}))).into_response(),
    }
}

/// `POST /api/studio/learn/consolidate` `{workspace?}`: the nightly job, started now.
pub(in crate::server) async fn consolidate(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    match start_job(state, Job::Consolidate { workspace: workspace_of(body["workspace"].as_str()) }) {
        Ok(()) => (StatusCode::ACCEPTED, Json(json!({"queued": true}))).into_response(),
        Err(e) => api_error(StatusCode::CONFLICT, &e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_from_the_page_are_reduced_to_text_and_capped() {
        let raw = json!([
            {"role": "system", "content": "x"},
            {"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": {"url": "data:..."}}]},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "bash", "arguments": {"command": "ls"}}}]},
            {"role": "tool", "content": "y".repeat(9000)},
        ]);
        let out = clean_messages(&raw);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0]["content"], "hello");
        assert!(out[1]["tool_calls"][0]["function"]["arguments"].as_str().unwrap().contains("ls"));
        assert_eq!(out[2]["content"].as_str().unwrap().chars().count(), 4000);
        let many: Vec<Value> = (0..400).map(|i| json!({"role": "user", "content": format!("m{i}")})).collect();
        let kept = clean_messages(&Value::Array(many));
        assert_eq!(kept.len(), MAX_MESSAGES);
        assert_eq!(kept.last().unwrap()["content"], "m399");
    }

    #[test]
    fn workspaces_must_be_existing_absolute_folders_and_skills_need_valid_names() {
        assert!(workspace_of(Some("relative/dir")).is_none());
        assert!(workspace_of(Some("")).is_none());
        assert!(workspace_of(Some(&std::env::temp_dir().display().to_string())).is_some());
        assert!(skill_of(&json!({"name": "../x"})).is_err());
        assert!(skill_of(&json!({"name": "ok-name", "scope": "workspace", "workspace": "rel"})).is_err());
        assert!(skill_of(&json!({"name": "ok-name"})).is_ok());
    }
}
