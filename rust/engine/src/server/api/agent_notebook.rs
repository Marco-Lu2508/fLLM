//! Editing Jupyter notebooks cell by cell. `read_file` already shows a notebook as text cells, but a change needs the JSON
//! to stay intact — rewriting a `.ipynb` as text would break it, so this tool edits the document itself: replace the source
//! of a cell, insert a cell or delete one. Outputs of a changed code cell are cleared, because they no longer match the code.
use super::agent::AgentSession;
use serde_json::{json, Value};

/// A notebook bigger than this is not edited in place (it is a data file at that point).
const MAX_NOTEBOOK_BYTES: u64 = 20 * 1024 * 1024;
const MAX_SOURCE_CHARS: usize = 200_000;

/// The `source` of a cell is a list of lines, each ending in a newline.
fn lines(text: &str) -> Value {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(index) = rest.find('\n') {
        out.push(Value::String(format!("{}\n", &rest[..index])));
        rest = &rest[index + 1..];
    }
    if !rest.is_empty() {
        out.push(Value::String(rest.to_owned()));
    }
    Value::Array(out)
}

fn cell_type(value: &Value) -> &str {
    value.get("cell_type").and_then(Value::as_str).unwrap_or("code")
}

/// One edit on the document; the document is left unchanged when the edit is refused.
pub(super) fn edit_document(document: &mut Value, action: &str, cell: usize, source: &str, kind: &str) -> Result<String, String> {
    let cells = document
        .get_mut("cells")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| "this is not a notebook: it has no cells array".to_owned())?;
    let count = cells.len();
    if source.chars().count() > MAX_SOURCE_CHARS {
        return Err(format!("the source is longer than {MAX_SOURCE_CHARS} characters"));
    }
    match action {
        "replace" => {
            if cell < 1 || cell > count {
                return Err(format!("cell {cell} does not exist (the notebook has {count})"));
            }
            let target = &mut cells[cell - 1];
            let was = cell_type(target).to_owned();
            target["source"] = lines(source);
            if was == "code" {
                // the old output belongs to the old code
                target["outputs"] = json!([]);
                target["execution_count"] = Value::Null;
            }
            Ok(format!("Replaced the source of cell {cell} ({was})."))
        }
        "insert" => {
            if cell < 1 || cell > count + 1 {
                return Err(format!("cell {cell} is not a position in this notebook (1 to {})", count + 1));
            }
            let kind = match kind {
                "" | "code" => "code",
                "markdown" => "markdown",
                "raw" => "raw",
                other => return Err(format!("cell_type has to be code, markdown or raw, not '{other}'")),
            };
            let mut new_cell = json!({"cell_type": kind, "metadata": {}, "source": lines(source)});
            if kind == "code" {
                new_cell["outputs"] = json!([]);
                new_cell["execution_count"] = Value::Null;
            }
            cells.insert(cell - 1, new_cell);
            Ok(format!("Inserted a {kind} cell at position {cell}."))
        }
        "delete" => {
            if cell < 1 || cell > count {
                return Err(format!("cell {cell} does not exist (the notebook has {count})"));
            }
            if count == 1 {
                return Err("refusing to delete the only cell of the notebook".to_owned());
            }
            let removed = cells.remove(cell - 1);
            Ok(format!("Deleted cell {cell} ({}).", cell_type(&removed)))
        }
        other => Err(format!("unknown action '{other}'; use replace, insert or delete")),
    }
}

/// `{path, action, cell, source, cell_type}` — one cell, 1-based.
pub(super) fn run(session: &AgentSession, args: &Value) -> Result<String, String> {
    let path_arg = args.get("path").and_then(Value::as_str).unwrap_or("");
    if path_arg.trim().is_empty() {
        return Err("path is missing".to_owned());
    }
    let action = args.get("action").and_then(Value::as_str).unwrap_or("");
    let cell = args.get("cell").and_then(Value::as_u64).unwrap_or(0) as usize;
    let source = args.get("source").and_then(Value::as_str).unwrap_or("");
    let kind = args.get("cell_type").and_then(Value::as_str).unwrap_or("");
    let path = session.resolve_path(path_arg)?;
    if !path.extension().is_some_and(|e| e.eq_ignore_ascii_case("ipynb")) {
        return Err("notebook_edit only edits .ipynb files; change other files with edit_file".to_owned());
    }
    let meta = std::fs::metadata(&path).map_err(|e| format!("cannot read {path_arg}: {e}"))?;
    if meta.len() > MAX_NOTEBOOK_BYTES {
        return Err(format!("{path_arg} is bigger than {} MB and is not edited in place", MAX_NOTEBOOK_BYTES / 1024 / 1024));
    }
    let text = std::fs::read_to_string(&path).map_err(|e| format!("cannot read {path_arg}: {e}"))?;
    let mut document: Value = serde_json::from_str(&text).map_err(|e| format!("{path_arg} is not valid JSON: {e}"))?;
    let summary = edit_document(&mut document, action, cell, source, kind)?;
    let pretty = serde_json::to_string_pretty(&document).map_err(|e| e.to_string())?;
    std::fs::write(&path, format!("{pretty}\n")).map_err(|e| format!("cannot write {path_arg}: {e}"))?;
    // the model should see what the notebook looks like now, not the JSON
    let cells = document.get("cells").and_then(Value::as_array).cloned().unwrap_or_default();
    let mut out = format!("{summary}\n\nThe notebook now has {} cell(s):\n", cells.len());
    for (index, item) in cells.iter().enumerate().take(40) {
        let source: String = match item.get("source") {
            Some(Value::Array(lines)) => lines.iter().filter_map(Value::as_str).collect(),
            Some(Value::String(text)) => text.clone(),
            _ => String::new(),
        };
        let first = source.lines().next().unwrap_or("").chars().take(100).collect::<String>();
        out.push_str(&format!("- {} [{}] {}\n", index + 1, cell_type(item), first));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn notebook(cells: &[(&str, &str)]) -> Value {
        let mut items = Vec::new();
        for (kind, source) in cells {
            let mut cell = json!({"cell_type": kind, "metadata": {}, "source": lines(source)});
            if *kind == "code" {
                cell["outputs"] = json!([{"output_type": "stream", "text": ["alt\n"]}]);
                cell["execution_count"] = json!(3);
            }
            items.push(cell);
        }
        json!({"cells": items, "metadata": {}, "nbformat": 4, "nbformat_minor": 5})
    }

    fn source_of(document: &Value, index: usize) -> String {
        document["cells"][index]["source"].as_array().unwrap().iter().filter_map(Value::as_str).collect()
    }

    #[test]
    fn replacing_source_clears_stale_output() {
        let mut doc = notebook(&[("code", "x = 1\n")]);
        let message = edit_document(&mut doc, "replace", 1, "x = 2\ny = 3\n", "").unwrap();
        assert!(message.contains("cell 1"));
        assert_eq!(source_of(&doc, 0), "x = 2\ny = 3\n");
        assert_eq!(doc["cells"][0]["outputs"], json!([]));
        assert!(doc["cells"][0]["execution_count"].is_null());
    }

    #[test]
    fn a_markdown_cell_keeps_no_output_field_surprise() {
        let mut doc = notebook(&[("markdown", "# alt\n")]);
        edit_document(&mut doc, "replace", 1, "# neu\n", "").unwrap();
        assert_eq!(source_of(&doc, 0), "# neu\n");
        assert!(doc["cells"][0].get("outputs").is_none());
    }

    #[test]
    fn inserting_puts_the_cell_where_it_belongs() {
        let mut doc = notebook(&[("markdown", "a\n"), ("code", "b\n")]);
        edit_document(&mut doc, "insert", 2, "neu = True\n", "code").unwrap();
        assert_eq!(doc["cells"].as_array().unwrap().len(), 3);
        assert_eq!(source_of(&doc, 1), "neu = True\n");
        assert_eq!(doc["cells"][1]["outputs"], json!([]));
        // at the end is a valid position too
        edit_document(&mut doc, "insert", 4, "## Schluss\n", "markdown").unwrap();
        assert_eq!(source_of(&doc, 3), "## Schluss\n");
    }

    #[test]
    fn deleting_removes_exactly_one_cell() {
        let mut doc = notebook(&[("code", "1\n"), ("code", "2\n"), ("code", "3\n")]);
        edit_document(&mut doc, "delete", 2, "", "").unwrap();
        assert_eq!(doc["cells"].as_array().unwrap().len(), 2);
        assert_eq!(source_of(&doc, 1), "3\n");
    }

    #[test]
    fn a_wrong_cell_or_position_is_refused_and_changes_nothing() {
        let mut doc = notebook(&[("code", "1\n")]);
        let before = doc.clone();
        assert!(edit_document(&mut doc, "replace", 2, "x", "").is_err());
        assert!(edit_document(&mut doc, "replace", 0, "x", "").is_err());
        assert!(edit_document(&mut doc, "delete", 1, "", "").unwrap_err().contains("only cell"));
        assert!(edit_document(&mut doc, "insert", 3, "x", "").is_err());
        assert!(edit_document(&mut doc, "insert", 1, "x", "prose").is_err());
        assert!(edit_document(&mut doc, "frobnicate", 1, "x", "").is_err());
        assert_eq!(doc, before, "a refused edit must not touch the document");
    }

    #[test]
    fn something_that_is_no_notebook_is_refused() {
        let mut value = json!({"notebook": true});
        let error = edit_document(&mut value, "replace", 1, "x", "").unwrap_err();
        assert!(error.contains("no cells array"), "{error}");
    }

    #[test]
    fn source_lines_end_in_a_newline_like_jupyter_writes_them() {
        assert_eq!(lines("a\nb"), json!(["a\n", "b"]));
        assert_eq!(lines(""), json!([]));
    }
}
