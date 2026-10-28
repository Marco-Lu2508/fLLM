//! The catalog of the server-side agent tools (`GET /api/studio/agent/tools`): each tool as an OpenAI
//! function schema plus the permission category the page asks the user about (`read`, `edit`, `command`,
//! `network`, `memory`), so the page does not have to hard-code the tools.
use super::*;

pub(super) fn tool(name: &str, permission: &str, description: &str, properties: Value, required: &[&str]) -> Value {
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
            "List a folder of the workspace as a tree (folders first). Use depth 2 or 3 to see the layout of a project at a glance; build output and ignored folders are shown but not opened. To find files by name use glob; to find text use grep.",
            json!({
                "path": {"type": "string", "description": "Folder relative to the workspace (default: the workspace root)."},
                "depth": {"type": "integer", "description": "Levels to show, 1 to 4 (default 1)."},
            }),
            &[],
        ),
        tool(
            "read_file",
            "read",
            "Read a text file; every line is shown with its line number. Read a file before you change it. Long files: use offset and limit (at most 2000 lines per call). Jupyter notebooks (.ipynb) are shown as text cells with their outputs. Binary files and images are refused.",
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
            "Create a new file or replace a file completely. For changes to an existing file use edit_file or patch instead: they are safer and show a small diff.",
            json!({
                "path": {"type": "string", "description": "File path relative to the workspace."},
                "content": {"type": "string", "description": "The complete new content."},
            }),
            &["path", "content"],
        ),
        tool(
            "edit_file",
            "edit",
            "Change a file by replacing text. old_string must be copied exactly from the file (read it first) and occur only once, unless replace_all is set; include a few unchanged lines around the change to make it unique. Differences in indentation are tolerated when the match is unique. If the edit fails, the error says where the closest text is; read that part again.",
            json!({
                "path": {"type": "string"},
                "old_string": {"type": "string", "description": "The exact text to replace."},
                "new_string": {"type": "string", "description": "The replacement text."},
                "replace_all": {"type": "boolean", "description": "Replace every occurrence (for renames)."},
            }),
            &["path", "old_string", "new_string"],
        ),
        tool(
            "multi_edit",
            "edit",
            "Apply several edits to one file in order, all or nothing (one undo step). Each edit sees the result of the previous one. Prefer this to several edit_file calls on the same file.",
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
            "patch",
            "edit",
            "Apply a unified diff to one or more files, all or nothing. Use it for changes that touch several places or several files. Format: `--- a/path` and `+++ b/path` headers, then hunks that start with `@@ -line,count +line,count @@`; every hunk line starts with a space (unchanged), `-` (removed) or `+` (added), with 2 or 3 unchanged lines around each change. Line numbers only need to be roughly right: hunks are placed by their content. A new file has `--- /dev/null`, a deleted file `+++ /dev/null`.",
            json!({"patch": {"type": "string", "description": "The unified diff."}}),
            &["patch"],
        ),
        tool(
            "glob",
            "read",
            "Find files by name pattern (`**` crosses folders, `*` stays inside one), newest first. Files that .gitignore excludes are skipped.",
            json!({"pattern": {"type": "string", "description": "For example `src/**/*.rs` or `**/*.test.js`."}}),
            &["pattern"],
        ),
        tool(
            "grep",
            "read",
            "Search file contents with a regular expression (Rust syntax). Files that .gitignore excludes are skipped. Use output_mode files to see only which files match, count to see how many matches there are, and context to see the lines around a match.",
            json!({
                "pattern": {"type": "string", "description": "Regular expression, or plain text with literal=true."},
                "path": {"type": "string", "description": "File or folder to search (default: the workspace)."},
                "glob": {"type": "string", "description": "Only files matching this glob, for example `*.rs`."},
                "output_mode": {"type": "string", "enum": ["content", "files", "count"], "description": "content: matching lines (default); files: only file names; count: matches per file."},
                "context": {"type": "integer", "description": "Lines of context around each match (content mode, at most 20)."},
                "case_insensitive": {"type": "boolean"},
                "literal": {"type": "boolean", "description": "Treat the pattern as plain text, not a regular expression."},
                "word": {"type": "boolean", "description": "Match whole words only."},
                "head_limit": {"type": "integer", "description": "Maximum number of matches (or files) to list. Default 200."},
            }),
            &["pattern"],
        ),
        tool(
            "bash",
            "command",
            "Run a shell command in the workspace and return its output (stdout and stderr; long output keeps its beginning and its end). A `cd` stays in effect for the next call (inside the workspace). Use it for building, testing, git and running programs, not for reading or searching files (use read_file, grep and glob for that). Give a short timeout for quick commands. With background=true the command keeps running (dev servers, watchers) and a job id is returned.",
            json!({
                "command": {"type": "string"},
                "timeout_ms": {"type": "integer", "description": "Time limit in milliseconds (default 120000, maximum 600000). Not used for background jobs."},
                "background": {"type": "boolean", "description": "Start the command as a background job (dev servers, watchers)."},
            }),
            &["command"],
        ),
        tool(
            "review",
            "read",
            "Review the changes of this workspace: the git diff plus the findings of a fixed set of rules (secrets, eval/exec, shell with interpolation, SQL by concatenation, conflict markers, plain http, TODO, debug output, unwrap, unsafe, swallowed errors), each with file and line. Read the diff, open the files, and say for each finding whether it is real. This tool changes nothing. Use scope \"staged\" for the index and \"branch\" with base for a comparison against another branch.",
            json!({
                "scope": {"type": "string", "enum": ["diff", "staged", "branch"], "description": "diff (default): the uncommitted changes. staged: what is in the index. branch: everything since the base branch."},
                "base": {"type": "string", "description": "With scope \"branch\": the branch, tag or commit to compare against (default HEAD)."},
                "security": {"type": "boolean", "description": "Only the security rules (secrets, eval, shell, SQL, conflict markers, plain http)."},
            }),
            &[],
        ),
        tool(
            "diagnostics",
            "command",
            "Check the code for errors without running it: a syntax check of one file (Python, JavaScript, JSON) or a build check of the project (cargo check, tsc --noEmit, go vet) when the workspace has the matching project file. Returns the first problems. Run it after editing code, before you tell the user the work is done.",
            json!({
                "path": {"type": "string", "description": "A file to check, or empty to check the whole project."},
                "timeout_s": {"type": "integer", "description": "Time limit in seconds (5 to 300, default 120)."},
            }),
            &[],
        ),
        tool(
            "session_note",
            "memory",
            "Leave a handover note for the next session in this workspace, or read the notes that wait here. Chats live in the browser, so a closed session cannot be talked to; a note waits in `.fllm/messages.json` and the Studio shows it the next time a chat is opened. Put in what a colleague would need: what was tried, what is left, which file or branch matters. With `list` true it shows the notes instead of writing one.",
            json!({
                "text": {"type": "string", "description": "The note. A few sentences, no secrets."},
                "to": {"type": "string", "description": "Optional: the title of the chat it is meant for. Empty means every chat of this workspace sees it."},
                "list": {"type": "boolean", "description": "Show the notes of this workspace instead of writing one."},
            }),
            &[],
        ),
        tool(
            "mcp_resource",
            "read",
            "Read a resource that an MCP server publishes (a file, a database record, a document it owns). Call it with only the server to list its resources and their uris, then again with a uri to read one. Resources are read-only and are neither tools nor prompts.",
            json!({
                "server": {"type": "string", "description": "The name of the MCP server, as the MCP page shows it."},
                "uri": {"type": "string", "description": "The uri of the resource to read; leave empty to list the resources of the server."},
            }),
            &["server"],
        ),
        tool(
            "notebook_edit",
            "edit",
            "Change a Jupyter notebook (.ipynb) cell by cell instead of rewriting its JSON: replace the source of a cell, insert a cell or delete one. Cells are counted from 1, like read_file shows them. Replacing the source of a code cell clears its old output, because it no longer matches the code. Read the notebook with read_file first.",
            json!({
                "path": {"type": "string", "description": "The .ipynb file."},
                "action": {"type": "string", "enum": ["replace", "insert", "delete"]},
                "cell": {"type": "integer", "description": "1-based cell number. For insert: the position the new cell gets (1 to count+1)."},
                "source": {"type": "string", "description": "The new source for replace and insert (ignored for delete)."},
                "cell_type": {"type": "string", "enum": ["code", "markdown", "raw"], "description": "Only for insert; default code."},
            }),
            &["path", "action", "cell"],
        ),
        tool(
            "lsp",
            "read",
            "Ask the language server of this project instead of guessing: \"diagnostics\" reports the problems a real language server sees in one file (types, imports, borrows), \"symbols\" searches the workspace for a symbol by name, and \"definition\" finds where a symbol at a position is defined. Needs a language server on PATH (rust-analyzer, pyright-langserver, typescript-language-server, gopls, clangd); when none is installed the tool says so, and the diagnostics tool still works without one.",
            json!({
                "action": {"type": "string", "enum": ["diagnostics", "symbols", "definition"]},
                "path": {"type": "string", "description": "File for diagnostics and definition (default for diagnostics: the first source file of the workspace)."},
                "query": {"type": "string", "description": "With symbols: the name to look for."},
                "line": {"type": "integer", "description": "With definition: 1-based line."},
                "character": {"type": "integer", "description": "With definition: 1-based column."},
            }),
            &["action"],
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
            "memory",
            "memory",
            "Your long-term memory for this workspace (scope workspace) and for this user in every workspace (scope global). Entries are short facts you will want next time: user preferences, project conventions, commands that work, mistakes to avoid. Save one when you learn something durable that is not obvious from the code, or when the user asks you to remember something. Do not store secrets, long text or things that only matter for the current task. Actions: list, add (text), update (id, text), remove (id). List before you add so you do not store duplicates.",
            json!({
                "action": {"type": "string", "enum": ["list", "add", "update", "remove"]},
                "scope": {"type": "string", "enum": ["workspace", "global"], "description": "Default workspace."},
                "text": {"type": "string", "description": "One short sentence (add, update)."},
                "id": {"type": "integer", "description": "The entry number from the list (update, remove)."},
            }),
            &["action"],
        ),
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
            assert!(["read", "edit", "command", "network", "memory"].contains(&t["permission"].as_str().unwrap()), "{name}");
            assert_eq!(t["type"], "function");
            assert_eq!(t["function"]["parameters"]["type"], "object");
            let props = t["function"]["parameters"]["properties"].as_object().unwrap();
            for required in t["function"]["parameters"]["required"].as_array().unwrap() {
                assert!(props.contains_key(required.as_str().unwrap()), "{name}: required argument without a property");
            }
        }
        for expected in [
            "list_dir", "read_file", "write_file", "edit_file", "multi_edit", "patch", "glob", "grep", "bash", "diagnostics", "bash_output",
            "bash_kill", "memory", "web_fetch", "web_search",
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
