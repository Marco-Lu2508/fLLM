//! The catalog of the server-side agent tools (`GET /api/studio/agent/tools`): each tool as an OpenAI
//! function schema plus the permission category the page asks the user about (`read`, `edit`, `command`,
//! `network`), so the page does not have to hard-code the tools.
use super::*;

fn tool(name: &str, permission: &str, description: &str, properties: Value, required: &[&str]) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
        "permission": permission,
    })
}

pub(super) fn catalog() -> Vec<Value> {
    vec![
        tool(
            "list_dir",
            "read",
            "List the files and folders of a directory inside the workspace.",
            json!({"path": {"type": "string", "description": "Folder relative to the workspace (default: the workspace root)."}}),
            &[],
        ),
        tool(
            "read_file",
            "read",
            "Read a text file with line numbers. Jupyter notebooks (.ipynb) are shown as text cells with their outputs. Use offset and limit for large files.",
            json!({
                "path": {"type": "string", "description": "File path relative to the workspace."},
                "offset": {"type": "integer", "description": "First line to read (1-based)."},
                "limit": {"type": "integer", "description": "Maximum number of lines (default and maximum 2000)."},
            }),
            &["path"],
        ),
        tool(
            "write_file",
            "edit",
            "Create a file or overwrite it completely.",
            json!({
                "path": {"type": "string", "description": "File path relative to the workspace."},
                "content": {"type": "string", "description": "The complete new content."},
            }),
            &["path", "content"],
        ),
        tool(
            "edit_file",
            "edit",
            "Replace text in a file. old_string must match exactly and be unique unless replace_all is set.",
            json!({
                "path": {"type": "string"},
                "old_string": {"type": "string", "description": "The exact text to replace."},
                "new_string": {"type": "string", "description": "The replacement text."},
                "replace_all": {"type": "boolean", "description": "Replace every occurrence."},
            }),
            &["path", "old_string", "new_string"],
        ),
        tool(
            "multi_edit",
            "edit",
            "Apply several edits to one file in order, all or nothing (one undo step). Each edit sees the result of the previous one.",
            json!({
                "path": {"type": "string"},
                "edits": {
                    "type": "array",
                    "description": "The edits, applied in order.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_string": {"type": "string"},
                            "new_string": {"type": "string"},
                            "replace_all": {"type": "boolean"},
                        },
                        "required": ["old_string", "new_string"],
                    },
                },
            }),
            &["path", "edits"],
        ),
        tool(
            "glob",
            "read",
            "Find files by glob pattern (`**` crosses folders, `*` stays in one), newest first.",
            json!({"pattern": {"type": "string", "description": "For example `src/**/*.rs`."}}),
            &["pattern"],
        ),
        tool(
            "grep",
            "read",
            "Search file contents with a regular expression.",
            json!({
                "pattern": {"type": "string", "description": "Regular expression."},
                "path": {"type": "string", "description": "File or folder to search (default: the workspace)."},
                "glob": {"type": "string", "description": "Only files matching this glob, for example `*.rs`."},
                "output_mode": {"type": "string", "enum": ["content", "files", "count"], "description": "content: matching lines (default); files: only file names; count: matches per file."},
                "context": {"type": "integer", "description": "Lines of context around each match (content mode, at most 20)."},
                "case_insensitive": {"type": "boolean"},
                "head_limit": {"type": "integer", "description": "Maximum number of matches (or files) to list. Default 200."},
            }),
            &["pattern"],
        ),
        tool(
            "bash",
            "command",
            "Run a shell command in the workspace. A `cd` stays in effect for the next call (inside the workspace). With background=true the command keeps running and a job id is returned.",
            json!({
                "command": {"type": "string"},
                "timeout_ms": {"type": "integer", "description": "Time limit in milliseconds (default 120000, maximum 600000). Not used for background jobs."},
                "background": {"type": "boolean", "description": "Start the command as a background job (dev servers, watchers)."},
            }),
            &["command"],
        ),
        tool(
            "bash_output",
            "read",
            "Read the output of a background job and see whether it still runs.",
            json!({
                "job_id": {"type": "string"},
                "since": {"type": "integer", "description": "The next_since value of the previous call: return only newer output."},
            }),
            &["job_id"],
        ),
        tool("bash_kill", "read", "Stop a background job.", json!({"job_id": {"type": "string"}}), &["job_id"]),
        tool(
            "web_fetch",
            "network",
            "Download a web page (http or https) and return it as readable text.",
            json!({
                "url": {"type": "string"},
                "max_chars": {"type": "integer", "description": "Maximum characters returned (default 20000)."},
            }),
            &["url"],
        ),
        tool(
            "web_search",
            "network",
            "Search the web (best effort, no API key). Returns title, URL and snippet per result.",
            json!({
                "query": {"type": "string"},
                "limit": {"type": "integer", "description": "Maximum number of results (default 8, maximum 20)."},
            }),
            &["query"],
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::super::agent::AgentSession;
    use super::*;

    #[test]
    fn every_tool_has_a_schema_and_a_known_permission() {
        let tools = catalog();
        let mut names = std::collections::HashSet::new();
        for t in &tools {
            let name = t["function"]["name"].as_str().unwrap();
            assert!(names.insert(name), "duplicate tool {name}");
            assert!(["read", "edit", "command", "network"].contains(&t["permission"].as_str().unwrap()), "{name}");
            assert_eq!(t["type"], "function");
            assert_eq!(t["function"]["parameters"]["type"], "object");
            let props = t["function"]["parameters"]["properties"].as_object().unwrap();
            for required in t["function"]["parameters"]["required"].as_array().unwrap() {
                assert!(props.contains_key(required.as_str().unwrap()), "{name}: required argument without a property");
            }
        }
        for expected in [
            "list_dir", "read_file", "write_file", "edit_file", "multi_edit", "glob", "grep", "bash", "bash_output", "bash_kill",
            "web_fetch", "web_search",
        ] {
            assert!(names.contains(expected), "{expected} missing");
        }
    }

    #[test]
    fn every_workspace_tool_of_the_catalog_is_implemented() {
        let session = AgentSession::default();
        for t in catalog() {
            let name = t["function"]["name"].as_str().unwrap();
            if name.starts_with("web_") {
                continue; // async, covered by agent_net
            }
            // no workspace yet: a known tool fails on that, an unknown one says so
            let error = session.run_tool(name, &json!({})).unwrap_err();
            assert!(!error.contains("unknown tool"), "{name} is in the catalog but not implemented");
        }
    }
}
