//! Finding MCP servers: a search of the official MCP registry (registry.modelcontextprotocol.io) and of Smithery (registry.smithery.ai).
//! Both are public lists. The page asks this route (the browser cannot reach them directly) and gets every hit with the ways to run it
//! already worked out: the command and arguments of a package (`npx`, `uvx`, `docker`) or the address of a remote server, with the
//! variables the user has to fill in. Nothing is installed here; the page puts a choice into the form of Settings > Tools (MCP), where
//! the user sees exactly what will run and saves it.
use super::*;
use axum::extract::Query;
use std::collections::HashMap;
use std::time::Duration;

const OFFICIAL: &str = "https://registry.modelcontextprotocol.io";
const SMITHERY: &str = "https://registry.smithery.ai";
const MAX_ANSWER: usize = 4 << 20;

async fn get_json(url: &str) -> Result<Value, String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .redirect(reqwest::redirect::Policy::custom(|attempt| {
            let host = attempt.url().host_str().unwrap_or("");
            if attempt.url().scheme() == "https" && (host == "registry.modelcontextprotocol.io" || host == "registry.smithery.ai") && attempt.previous().len() < 3 {
                attempt.follow()
            } else {
                attempt.stop()
            }
        }))
        .build()
        .map_err(|e| e.to_string())?;
    let response = client.get(url).header("accept", "application/json").send().await.map_err(|e| format!("the registry did not answer: {e}"))?;
    if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
        return Err("the registry says: too many requests, try again in a minute".into());
    }
    if !response.status().is_success() {
        return Err(format!("the registry answered with status {}", response.status().as_u16()));
    }
    let bytes = response.bytes().await.map_err(|e| e.to_string())?;
    if bytes.len() > MAX_ANSWER {
        return Err("the answer of the registry is too large".into());
    }
    serde_json::from_slice(&bytes).map_err(|_| "the registry answered with something that is not JSON".to_owned())
}

/// A name the MCP list accepts, made from the name in a registry (`io.github.owner/repo-name`).
pub(super) fn suggested_name(registry_name: &str) -> String {
    let last = registry_name.rsplit('/').next().unwrap_or(registry_name);
    let cleaned: String = last.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c.to_ascii_lowercase() } else { '-' }).collect();
    let trimmed = cleaned.trim_matches('-');
    trimmed.chars().take(48).collect::<String>()
}

fn text(value: &Value, key: &str) -> String {
    value.get(key).and_then(Value::as_str).unwrap_or("").to_owned()
}

/// What a value of the registry asks of the user: a fixed value, or a text with `{placeholders}` they must fill in.
fn variable(item: &Value) -> Value {
    let fixed = text(item, "value");
    let default = text(item, "default");
    json!({
        "name": text(item, "name"),
        "description": text(item, "description"),
        "required": item.get("isRequired").and_then(Value::as_bool).unwrap_or(false),
        "secret": item.get("isSecret").and_then(Value::as_bool).unwrap_or(false),
        "value": if fixed.is_empty() { default } else { fixed },
    })
}

/// How to start a package of the official registry, or `None` when fLLM cannot (a format other than npm, PyPI or an OCI image).
fn package_candidate(package: &Value, title: &str) -> Option<Value> {
    let kind = text(package, "registryType");
    let identifier = text(package, "identifier");
    let version = text(package, "version");
    if identifier.is_empty() || package.pointer("/transport/type").and_then(Value::as_str).is_some_and(|t| t != "stdio") {
        return None;
    }
    let env: Vec<Value> = package["environmentVariables"].as_array().map(|list| list.iter().map(variable).collect()).unwrap_or_default();
    let mut args: Vec<String> = Vec::new();
    // the arguments of the package itself, after its name
    let mut tail: Vec<String> = Vec::new();
    for argument in package["packageArguments"].as_array().cloned().unwrap_or_default() {
        let given = {
            let v = text(&argument, "value");
            if v.is_empty() { text(&argument, "default") } else { v }
        };
        let shown = if given.is_empty() { format!("<{}>", Some(text(&argument, "valueHint")).filter(|h| !h.is_empty()).unwrap_or_else(|| text(&argument, "name"))) } else { given };
        if argument["type"] == "named" {
            tail.push(text(&argument, "name"));
            tail.push(shown);
        } else {
            tail.push(shown);
        }
    }
    let (command, label) = match kind.as_str() {
        "npm" => {
            args.push("-y".into());
            args.push(if version.is_empty() { identifier.clone() } else { format!("{identifier}@{version}") });
            ("npx", format!("npm package {identifier}"))
        }
        "pypi" => {
            args.push(if version.is_empty() { identifier.clone() } else { format!("{identifier}=={version}") });
            ("uvx", format!("Python package {identifier}"))
        }
        "oci" => {
            args.extend(["run".into(), "-i".into(), "--rm".into()]);
            for e in &env {
                if let Some(name) = e["name"].as_str().filter(|n| !n.is_empty()) {
                    args.push("-e".into());
                    args.push(name.to_owned());
                }
            }
            args.push(identifier.clone());
            ("docker", format!("container image {identifier}"))
        }
        _ => return None,
    };
    args.extend(tail);
    Some(json!({"kind": "local", "label": label, "name": suggested_name(title), "command": command, "args": args, "env": env}))
}

fn official_entry(entry: &Value) -> Option<Value> {
    let server = entry.get("server")?;
    let name = text(server, "name");
    if name.is_empty() {
        return None;
    }
    let mut candidates: Vec<Value> = server["packages"].as_array().map(|list| list.iter().filter_map(|p| package_candidate(p, &name)).collect()).unwrap_or_default();
    for remote in server["remotes"].as_array().cloned().unwrap_or_default() {
        let url = text(&remote, "url");
        if text(&remote, "type") != "streamable-http" || !url.starts_with("https://") && !url.starts_with("http://") {
            continue;
        }
        let headers: Vec<Value> = remote["headers"].as_array().map(|list| list.iter().map(variable).collect()).unwrap_or_default();
        candidates.push(json!({"kind": "remote", "label": format!("remote server {url}"), "name": suggested_name(&name), "url": url, "headers": headers}));
    }
    Some(json!({
        "id": name,
        "title": name,
        "description": text(server, "description"),
        "version": text(server, "version"),
        "repository": server.pointer("/repository/url").and_then(Value::as_str).unwrap_or(""),
        "website": text(server, "websiteUrl"),
        "candidates": candidates,
    }))
}

fn smithery_entry(entry: &Value) -> Option<Value> {
    let qualified = text(entry, "qualifiedName");
    if qualified.is_empty() || !qualified.chars().all(|c| c.is_ascii_alphanumeric() || "-_./@".contains(c)) {
        return None;
    }
    let remote = entry.get("remote").and_then(Value::as_bool).unwrap_or(false);
    let candidates = if remote {
        vec![json!({
            "kind": "remote",
            "label": "remote server on Smithery (needs your Smithery API key)",
            "name": suggested_name(&qualified),
            "url": format!("https://server.smithery.ai/{qualified}/mcp"),
            "headers": [{"name": "Authorization", "description": "Your Smithery API key (smithery.ai/account/api-keys), written as: Bearer <key>", "required": true, "secret": true, "value": "Bearer "}],
        })]
    } else {
        Vec::new()
    };
    Some(json!({
        "id": qualified,
        "title": Some(text(entry, "displayName")).filter(|t| !t.is_empty()).unwrap_or_else(|| qualified.clone()),
        "description": text(entry, "description"),
        "verified": entry.get("verified").and_then(Value::as_bool).unwrap_or(false),
        "uses": entry.get("useCount").cloned().unwrap_or(Value::Null),
        "website": text(entry, "homepage"),
        "candidates": candidates,
    }))
}

fn encode(q: &str) -> String {
    let mut out = String::new();
    for c in q.chars().take(100) {
        if c.is_ascii_alphanumeric() || "-_.~".contains(c) {
            out.push(c);
        } else {
            let mut buffer = [0u8; 4];
            for byte in c.encode_utf8(&mut buffer).bytes() {
                out.push_str(&format!("%{byte:02X}"));
            }
        }
    }
    out
}

/// `GET /api/studio/mcp/registry?source=official|smithery&q=<words>&cursor=<next>`.
pub(in crate::server) async fn search(Query(query): Query<HashMap<String, String>>) -> Response {
    let q = encode(query.get("q").map(|q| q.trim()).unwrap_or(""));
    let source = query.get("source").map(String::as_str).unwrap_or("official");
    let cursor = encode(query.get("cursor").map(String::as_str).unwrap_or(""));
    let result = if source == "smithery" {
        let page: u32 = query.get("cursor").and_then(|c| c.parse().ok()).unwrap_or(1);
        get_json(&format!("{SMITHERY}/servers?q={q}&page={page}&pageSize=30")).await.map(|v| {
            let servers: Vec<Value> = v["servers"].as_array().map(|l| l.iter().filter_map(smithery_entry).collect()).unwrap_or_default();
            let more = v.pointer("/pagination/totalPages").and_then(Value::as_u64).is_some_and(|t| u64::from(page) < t);
            (servers, more.then(|| (page + 1).to_string()))
        })
    } else {
        let mut url = format!("{OFFICIAL}/v0/servers?limit=30&version=latest");
        if !q.is_empty() {
            url.push_str(&format!("&search={q}"));
        }
        if !cursor.is_empty() {
            url.push_str(&format!("&cursor={cursor}"));
        }
        get_json(&url).await.map(|v| {
            let servers: Vec<Value> = v["servers"].as_array().map(|l| l.iter().filter_map(official_entry).collect()).unwrap_or_default();
            let next = v.pointer("/metadata/nextCursor").and_then(Value::as_str).filter(|_| servers.len() >= 30).map(str::to_owned);
            (servers, next)
        })
    };
    match result {
        Ok((servers, next)) => Json(json!({"source": source, "servers": servers, "next": next})).into_response(),
        Err(message) => api_error(StatusCode::BAD_GATEWAY, &message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_made_from_registry_names() {
        assert_eq!(suggested_name("io.github.owner/Repo_Name"), "repo_name");
        assert_eq!(suggested_name("ai.smithery/Hint Services"), "hint-services");
        assert_eq!(suggested_name("plain"), "plain");
    }

    #[test]
    fn an_npm_package_becomes_an_npx_command_with_its_variables() {
        let entry = json!({"server": {"name": "io.github.x/files", "description": "d", "version": "1.0.0",
            "packages": [{"registryType": "npm", "identifier": "@x/files", "version": "1.2.0", "transport": {"type": "stdio"},
                "environmentVariables": [{"name": "TOKEN", "description": "t", "isRequired": true, "isSecret": true}],
                "packageArguments": [{"type": "positional", "valueHint": "folder", "isRequired": true}]}]}});
        let hit = official_entry(&entry).unwrap();
        let c = &hit["candidates"][0];
        assert_eq!(c["command"], "npx");
        assert_eq!(c["args"], json!(["-y", "@x/files@1.2.0", "<folder>"]));
        assert_eq!(c["env"][0]["name"], "TOKEN");
        assert_eq!(c["env"][0]["secret"], true);
        assert_eq!(c["name"], "files");
    }

    #[test]
    fn python_images_and_remotes_are_handled_and_the_rest_is_left_out() {
        let entry = json!({"server": {"name": "a/b", "packages": [
            {"registryType": "pypi", "identifier": "mcp-b", "version": "0.3", "transport": {"type": "stdio"}},
            {"registryType": "oci", "identifier": "ghcr.io/a/b:1", "transport": {"type": "stdio"}, "environmentVariables": [{"name": "K"}]},
            {"registryType": "nuget", "identifier": "A.B", "transport": {"type": "stdio"}},
            {"registryType": "npm", "identifier": "streamy", "transport": {"type": "streamable-http"}}],
            "remotes": [{"type": "sse", "url": "https://x/sse"}, {"type": "streamable-http", "url": "https://x/mcp", "headers": [{"name": "Authorization", "value": "Bearer {key}"}]}]}});
        let hit = official_entry(&entry).unwrap();
        let list = hit["candidates"].as_array().unwrap();
        assert_eq!(list.len(), 3);
        assert_eq!(list[0]["command"], "uvx");
        assert_eq!(list[0]["args"], json!(["mcp-b==0.3"]));
        assert_eq!(list[1]["command"], "docker");
        assert_eq!(list[1]["args"], json!(["run", "-i", "--rm", "-e", "K", "ghcr.io/a/b:1"]));
        assert_eq!(list[2]["kind"], "remote");
        assert_eq!(list[2]["url"], "https://x/mcp");
        assert_eq!(list[2]["headers"][0]["value"], "Bearer {key}");
    }

    #[test]
    fn smithery_servers_are_remote_only() {
        let hit = smithery_entry(&json!({"qualifiedName": "github", "displayName": "GitHub", "remote": true, "useCount": 5})).unwrap();
        assert_eq!(hit["candidates"][0]["url"], "https://server.smithery.ai/github/mcp");
        let local = smithery_entry(&json!({"qualifiedName": "local-only", "remote": false})).unwrap();
        assert!(local["candidates"].as_array().unwrap().is_empty());
        assert!(smithery_entry(&json!({"qualifiedName": "../x?y"})).is_none());
    }

    #[test]
    fn queries_are_percent_encoded() {
        assert_eq!(encode("git hub&x=1"), "git%20hub%26x%3D1");
    }
}
