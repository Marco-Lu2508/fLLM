//! Prices for the token meter: a small table of model patterns and dollars per million tokens, in `prices.json`
//! next to `studio.json`.
//!
//! Local models cost nothing and need no entry. The table exists for models reached through a provider (OpenAI,
//! Anthropic, OpenRouter, ...), whose prices move and differ per plan: nothing is prefilled, so the page never shows a
//! number nobody entered. A pattern is matched against the model name, the longest match wins, `*` matches any tail.
use super::*;
use std::fs;
use std::path::PathBuf;

const MAX_ROWS: usize = 100;

pub(super) fn file() -> PathBuf {
    crate::hub::studio_data_file("prices.json")
}

fn read_rows() -> Vec<Value> {
    let Ok(text) = fs::read_to_string(file()) else { return Vec::new() };
    let Ok(value) = serde_json::from_str::<Value>(&text) else { return Vec::new() };
    value.get("prices").and_then(Value::as_array).cloned().unwrap_or_default()
}

/// The rows with only the fields the page uses.
fn clean(rows: &[Value]) -> Result<Vec<Value>, String> {
    if rows.len() > MAX_ROWS {
        return Err(format!("at most {MAX_ROWS} price rows are allowed"));
    }
    let mut out = Vec::new();
    for row in rows {
        let pattern = row.get("pattern").and_then(Value::as_str).unwrap_or("").trim();
        if pattern.is_empty() {
            continue;
        }
        if pattern.chars().count() > 120 {
            return Err("a model pattern is limited to 120 characters".to_owned());
        }
        let number = |key: &str| -> Result<f64, String> {
            match row.get(key) {
                None | Some(Value::Null) => Ok(0.0),
                Some(value) => value.as_f64().filter(|n| n.is_finite() && *n >= 0.0 && *n < 1_000_000.0).ok_or_else(|| format!("'{key}' must be a number of dollars, not negative")),
            }
        };
        out.push(json!({
            "pattern": pattern,
            "input_per_million": number("input_per_million")?,
            "output_per_million": number("output_per_million")?,
            "note": row.get("note").and_then(Value::as_str).unwrap_or("").chars().take(120).collect::<String>(),
        }));
    }
    Ok(out)
}

pub(super) fn list_json() -> Value {
    json!({ "file": file().display().to_string(), "prices": read_rows() })
}

pub(super) fn save(body: &Value) -> Result<Value, String> {
    let rows = match body.get("prices").cloned().unwrap_or_else(|| body.clone()) {
        Value::Array(rows) => rows,
        _ => return Err("the body must be {\"prices\": [...]}".to_owned()),
    };
    let rows = clean(&rows)?;
    let text = serde_json::to_string_pretty(&json!({ "prices": rows })).map_err(|e| e.to_string())?;
    if let Some(dir) = file().parent() {
        fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    fs::write(file(), text + "\n").map_err(|e| format!("cannot write {}: {e}", file().display()))?;
    Ok(list_json())
}

/// `GET /api/studio/prices`: the table as the page shows it.
pub(in crate::server) async fn list() -> Response {
    Json(list_json()).into_response()
}

/// `POST /api/studio/prices`: replaces the table.
pub(in crate::server) async fn save_handler(Json(body): Json<Value>) -> Response {
    match save(&body) {
        Ok(saved) => Json(saved).into_response(),
        Err(message) => api_error(StatusCode::BAD_REQUEST, &message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_cleaned_and_empty_patterns_dropped() {
        let rows = clean(&[json!({"pattern": "gpt-6*", "input_per_million": 2.5, "output_per_million": 10}), json!({"pattern": "  ", "input_per_million": 1})]).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["pattern"], "gpt-6*");
        assert_eq!(rows[0]["output_per_million"], 10.0);
        assert_eq!(rows[0]["note"], "");
    }

    #[test]
    fn a_negative_price_is_refused() {
        let err = clean(&[json!({"pattern": "x", "input_per_million": -1})]).unwrap_err();
        assert!(err.contains("not negative"), "{err}");
    }

    #[test]
    fn missing_prices_count_as_zero_and_too_many_rows_are_refused() {
        let rows = clean(&[json!({"pattern": "local"})]).unwrap();
        assert_eq!(rows[0]["input_per_million"], 0.0);
        let many: Vec<Value> = (0..MAX_ROWS + 1).map(|i| json!({"pattern": format!("m{i}")})).collect();
        assert!(clean(&many).is_err());
    }
}
