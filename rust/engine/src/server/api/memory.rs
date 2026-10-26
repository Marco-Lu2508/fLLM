//! The chat memory of the Studio: short notes about the user that the page keeps in the browser and that the
//! assistant may add to with its `memory` tool. A copy is kept in `memory.json` next to `studio.json`, so the notes
//! survive a cleared browser. The page imports the copy when its own storage is empty.
//!
//! `GET /api/studio/memory` returns `{"memory": <stored object or null>}`; `PUT /api/studio/memory` replaces it with
//! `{"memories": [...], ...}`. The notes are personal, so both routes stay on this machine (see `http::studio_admin_blocked`).
use super::*;
use std::path::Path;
use std::sync::Mutex;

/// Largest stored memory file and request body.
pub(in crate::server) const MAX_BODY: usize = 512 << 10;
const MAX_ENTRIES: usize = 500;

static WRITE_LOCK: Mutex<()> = Mutex::new(());

fn file() -> std::path::PathBuf {
    crate::hub::studio_data_file("memory.json")
}

/// Checks the shape and size of a memory document sent by the page.
fn validate(doc: &Value) -> Result<(), String> {
    let obj = doc.as_object().ok_or("the memory must be a JSON object")?;
    let list = obj
        .get("memories")
        .and_then(Value::as_array)
        .ok_or("the memory needs a \"memories\" list")?;
    if list.len() > MAX_ENTRIES {
        return Err(format!("the memory holds at most {MAX_ENTRIES} entries"));
    }
    if list.iter().any(|m| !m.is_object()) {
        return Err("every memory entry must be an object".into());
    }
    if doc.to_string().len() > MAX_BODY {
        return Err("the memory is too large".into());
    }
    Ok(())
}

fn read_at(path: &Path) -> Option<Value> {
    let text = std::fs::read_to_string(path).ok()?;
    let doc: Value = serde_json::from_str(&text).ok()?;
    validate(&doc).ok().map(|()| doc)
}

fn write_at(path: &Path, doc: &Value) -> std::io::Result<()> {
    let _guard = WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, doc.to_string())?;
    std::fs::rename(&tmp, path)
}

/// `GET /api/studio/memory`
pub(in crate::server) async fn get_memory() -> Response {
    match tokio::task::spawn_blocking(|| read_at(&file())).await {
        Ok(memory) => Json(json!({ "memory": memory })).into_response(),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

/// `PUT /api/studio/memory` with `{"memories": [...]}`
pub(in crate::server) async fn put_memory(Json(body): Json<Value>) -> Response {
    if let Err(e) = validate(&body) {
        return api_error(StatusCode::BAD_REQUEST, &e);
    }
    match tokio::task::spawn_blocking(move || write_at(&file(), &body)).await {
        Ok(Ok(())) => Json(json!({"status": "saved"})).into_response(),
        Ok(Err(e)) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("the memory could not be saved: {e}")),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_checks_shape_and_size() {
        assert!(validate(&json!({"memories": []})).is_ok());
        assert!(validate(&json!({"memories": [{"id": "a", "text": "likes tea"}], "version": 2})).is_ok());
        assert!(validate(&json!([])).is_err());
        assert!(validate(&json!({"memories": "x"})).is_err());
        assert!(validate(&json!({"memories": ["plain string"]})).is_err());
        assert!(validate(&json!({"memories": vec![json!({}); MAX_ENTRIES + 1]})).is_err());
        let big = "x".repeat(MAX_BODY);
        assert!(validate(&json!({"memories": [{"text": big}]})).is_err());
    }

    #[test]
    fn the_file_round_trips_and_ignores_garbage() {
        let dir = std::env::temp_dir().join(format!("fllm-memory-test-{}", std::process::id()));
        let path = dir.join("memory.json");
        assert!(read_at(&path).is_none(), "no file yet");
        let doc = json!({"memories": [{"id": "m1", "text": "name is Sam", "createdAt": 1}]});
        write_at(&path, &doc).unwrap();
        assert_eq!(read_at(&path), Some(doc));
        let newer = json!({"memories": []});
        write_at(&path, &newer).unwrap();
        assert_eq!(read_at(&path), Some(newer));
        std::fs::write(&path, "not json").unwrap();
        assert!(read_at(&path).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
