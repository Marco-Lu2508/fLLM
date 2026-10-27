//! Web search that works without hosting anything, and takes extra engines when you want them.
//!
//! Default engines need no key and no server: DuckDuckGo over its HTML endpoint, and Wikipedia over its public API. Both
//! are queried, the answers are merged and deduplicated by URL. Extra engines can be added in Settings > Search: a SearXNG
//! instance (yours or a public one that allows `format=json`), Brave with an API key, or any endpoint that speaks the same
//! JSON as SearXNG.
//!
//! Every search reports per engine what happened — which answered, which was blocked, which had nothing. A search that
//! failed is never shown as "no results", because that would be a lie about the network.
//!
//! `search.json` next to `studio.json`: `{"engines": [{"kind": "ddg"}, {"kind": "searxng", "url": "http://..."}], "limit": 8}`
use super::agent::{agent_disabled, disabled_reply};
use super::*;
use serde_json::{json, Value};
use std::time::Duration;

const MAX_RESULTS: usize = 20;
const DEFAULT_RESULTS: usize = 8;
const TIMEOUT: Duration = Duration::from_secs(20);

fn file() -> std::path::PathBuf {
    crate::hub::studio_data_file("search.json")
}

fn read() -> Value {
    let Ok(text) = std::fs::read_to_string(file()) else { return json!({}) };
    serde_json::from_str::<Value>(&text).unwrap_or_else(|_| json!({}))
}

/// The engines to use: the configured ones, or the two that need nothing.
pub(super) fn engines() -> Vec<Value> {
    let configured = read().get("engines").and_then(Value::as_array).cloned().unwrap_or_default();
    let clean: Vec<Value> = configured
        .into_iter()
        .filter(|e| matches!(e.get("kind").and_then(Value::as_str), Some("ddg" | "wikipedia" | "searxng" | "brave")))
        .collect();
    if clean.is_empty() {
        return vec![json!({"kind": "ddg", "name": "DuckDuckGo"}), json!({"kind": "wikipedia", "name": "Wikipedia"})];
    }
    clean
}

fn label(engine: &Value) -> String {
    match engine.get("name").and_then(Value::as_str) {
        Some(name) if !name.trim().is_empty() => name.trim().to_owned(),
        _ => match engine.get("kind").and_then(Value::as_str).unwrap_or("?") {
            "ddg" => "DuckDuckGo".to_owned(),
            "wikipedia" => "Wikipedia".to_owned(),
            "searxng" => engine.get("url").and_then(Value::as_str).unwrap_or("SearXNG").to_owned(),
            other => other.to_owned(),
        },
    }
}

fn limit_of(body: &Value) -> usize {
    body.get("limit").and_then(Value::as_u64).map(|n| n as usize).unwrap_or(DEFAULT_RESULTS).clamp(1, MAX_RESULTS)
}

fn client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .timeout(TIMEOUT)
        .user_agent("Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0")
        .build()
        .map_err(|e| format!("cannot build the client: {e}"))
}

/// One hit; the same shape for every engine.
fn hit(title: &str, url: &str, snippet: &str) -> Value {
    json!({"title": title.trim(), "url": url.trim(), "snippet": snippet.trim()})
}

/// DuckDuckGo over the HTML endpoint (no key). A captcha is reported as blocked, not as "nothing found".
async fn search_ddg(http: &reqwest::Client, query: &str, limit: usize) -> Result<Vec<Value>, String> {
    let response = http
        .post("https://html.duckduckgo.com/html/")
        .form(&[("q", query)])
        .header("accept", "text/html")
        .send()
        .await
        .map_err(|e| format!("DuckDuckGo did not answer: {e}"))?;
    let status = response.status();
    let html = response.text().await.map_err(|e| format!("cannot read the answer: {e}"))?;
    if !status.is_success() {
        return Err(format!("DuckDuckGo answered HTTP {status}"));
    }
    // the parsing and the captcha detection are shared with the old single-engine tool
    let hits = super::agent_net::interpret_search(status.as_u16(), &html, limit)
        .map_err(|e| format!("{e}; add a SearXNG engine in Settings > Search"))?;
    Ok(hits.into_iter().map(|h| hit(&h.title, &h.url, &h.snippet)).collect())
}

/// Wikipedia over its public API (no key): good for facts, not for products or news.
async fn search_wikipedia(http: &reqwest::Client, query: &str, limit: usize) -> Result<Vec<Value>, String> {
    let mut out = Vec::new();
    for language in ["de", "en"] {
        let url = reqwest::Url::parse_with_params(
            &format!("https://{language}.wikipedia.org/w/api.php"),
            [
                ("action", "query"),
                ("list", "search"),
                ("srsearch", query),
                ("srlimit", &limit.to_string()),
                ("format", "json"),
            ],
        )
        .map_err(|e| e.to_string())?;
        let Ok(response) = http.get(url).send().await else { continue };
        if !response.status().is_success() {
            continue;
        }
        let Ok(body) = response.json::<Value>().await else { continue };
        for item in body.pointer("/query/search").and_then(Value::as_array).into_iter().flatten() {
            let title = item.get("title").and_then(Value::as_str).unwrap_or("");
            let snippet = item.get("snippet").and_then(Value::as_str).unwrap_or("");
            let snippet = regex_snippet(snippet);
            let page = format!("https://{language}.wikipedia.org/wiki/{}", title.replace(' ', "_"));
            out.push(hit(title, &page, &snippet));
        }
        if !out.is_empty() {
            break; // the language that found something wins
        }
    }
    Ok(out)
}

/// `snippet` comes as HTML (`<span class="searchmatch">…</span>`): tags out, entities in.
fn regex_snippet(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut inside = false;
    for ch in text.chars() {
        match ch {
            '<' => inside = true,
            '>' => inside = false,
            c if !inside => out.push(c),
            _ => {}
        }
    }
    out.replace("&quot;", "\"").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&#39;", "'")
}

/// A SearXNG instance that allows `format=json`, or any endpoint with that shape.
async fn search_searxng(http: &reqwest::Client, base: &str, query: &str, limit: usize) -> Result<Vec<Value>, String> {
    if base.trim().is_empty() {
        return Err("this SearXNG engine has no address".to_owned());
    }
    let url = reqwest::Url::parse_with_params(
        &format!("{}/search", base.trim().trim_end_matches('/')),
        [("q", query), ("format", "json"), ("language", "auto")],
    )
    .map_err(|e| format!("the address is not valid: {e}"))?;
    let response = http.get(url).send().await.map_err(|e| format!("no answer: {e}"))?;
    if !response.status().is_success() {
        return Err(format!("HTTP {} (public instances often forbid the JSON API)", response.status()));
    }
    let body: Value = response.json().await.map_err(|_| "the answer is no JSON (is `format=json` allowed there?)".to_owned())?;
    let mut out = Vec::new();
    for item in body.get("results").and_then(Value::as_array).into_iter().flatten().take(limit) {
        let title = item.get("title").and_then(Value::as_str).unwrap_or("");
        let url = item.get("url").and_then(Value::as_str).unwrap_or("");
        let snippet = item.get("content").and_then(Value::as_str).unwrap_or("");
        if !url.is_empty() {
            out.push(hit(title, url, snippet));
        }
    }
    Ok(out)
}

/// Brave Search with an API key.
async fn search_brave(http: &reqwest::Client, key: &str, query: &str, limit: usize) -> Result<Vec<Value>, String> {
    if key.trim().is_empty() {
        return Err("this Brave engine has no API key".to_owned());
    }
    let response = http
        .get("https://api.search.brave.com/res/v1/web/search")
        .query(&[("q", query), ("count", &limit.to_string())])
        .header("X-Subscription-Token", key.trim())
        .header("Accept", "application/json")
        .send()
        .await
        .map_err(|e| format!("no answer: {e}"))?;
    if !response.status().is_success() {
        return Err(format!("Brave answered HTTP {}", response.status()));
    }
    let body: Value = response.json().await.map_err(|_| "the answer is no JSON".to_owned())?;
    let mut out = Vec::new();
    for item in body.pointer("/web/results").and_then(Value::as_array).into_iter().flatten().take(limit) {
        let title = item.get("title").and_then(Value::as_str).unwrap_or("");
        let url = item.get("url").and_then(Value::as_str).unwrap_or("");
        let snippet = item.get("description").and_then(Value::as_str).unwrap_or("");
        out.push(hit(title, url, snippet));
    }
    Ok(out)
}

async fn run_engine(http: &reqwest::Client, engine: &Value, query: &str, limit: usize) -> Result<Vec<Value>, String> {
    match engine.get("kind").and_then(Value::as_str).unwrap_or("") {
        "ddg" => search_ddg(http, query, limit).await,
        "wikipedia" => search_wikipedia(http, query, limit).await,
        "searxng" => search_searxng(http, engine.get("url").and_then(Value::as_str).unwrap_or(""), query, limit).await,
        "brave" => search_brave(http, engine.get("key").and_then(Value::as_str).unwrap_or(""), query, limit).await,
        other => Err(format!("unknown engine '{other}'")),
    }
}

/// Queries every engine, merges by URL and says per engine what happened.
pub(super) async fn search(query: &str, limit: usize) -> Value {
    let query = query.trim();
    let mut results: Vec<Value> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    let mut reports: Vec<Value> = Vec::new();
    if query.is_empty() {
        return json!({"ok": false, "error": "the query is empty", "results": [], "engines": []});
    }
    let http = match client() {
        Ok(client) => client,
        Err(e) => return json!({"ok": false, "error": e, "results": [], "engines": []}),
    };
    for engine in engines() {
        let name = label(&engine);
        match run_engine(&http, &engine, query, limit).await {
            Ok(hits) => {
                let mut added = 0;
                for item in hits {
                    let url = item.get("url").and_then(Value::as_str).unwrap_or("").to_owned();
                    if url.is_empty() || seen.iter().any(|s| s == &url) {
                        continue;
                    }
                    seen.push(url);
                    results.push(item);
                    added += 1;
                }
                reports.push(json!({"engine": name, "ok": true, "found": added}));
            }
            Err(error) => reports.push(json!({"engine": name, "ok": false, "error": error})),
        }
        if results.len() >= limit {
            break;
        }
    }
    results.truncate(limit);
    json!({"ok": true, "query": query, "results": results, "engines": reports})
}

/// The same as a text block for the model.
pub(super) fn as_text(value: &Value) -> String {
    let mut out = String::new();
    let results = value.get("results").and_then(Value::as_array).cloned().unwrap_or_default();
    if results.is_empty() {
        out.push_str("No results.\n");
    }
    for (index, item) in results.iter().enumerate() {
        out.push_str(&format!(
            "{}. {}\n   {}\n   {}\n",
            index + 1,
            item.get("title").and_then(Value::as_str).unwrap_or(""),
            item.get("url").and_then(Value::as_str).unwrap_or(""),
            item.get("snippet").and_then(Value::as_str).unwrap_or("")
        ));
    }
    let reports: Vec<String> = value
        .get("engines")
        .and_then(Value::as_array)
        .map(|list| {
            list.iter()
                .map(|r| {
                    let name = r.get("engine").and_then(Value::as_str).unwrap_or("");
                    if r.get("ok").and_then(Value::as_bool).unwrap_or(false) {
                        format!("{name}: {} hit(s)", r.get("found").and_then(Value::as_u64).unwrap_or(0))
                    } else {
                        format!("{name}: failed ({})", r.get("error").and_then(Value::as_str).unwrap_or("?"))
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    if !reports.is_empty() {
        out.push_str(&format!("\nSearched with: {}", reports.join(" · ")));
    }
    out
}

/// `POST /api/studio/web/search` (and `/api/studio/search`): `{query, limit?}`.
pub(in crate::server) async fn search_route(Json(body): Json<Value>) -> Response {
    if agent_disabled() {
        return disabled_reply();
    }
    let query = body.get("query").and_then(Value::as_str).unwrap_or("");
    let value = search(query, limit_of(&body)).await;
    let status = if value.get("ok").and_then(Value::as_bool).unwrap_or(false) { StatusCode::OK } else { StatusCode::BAD_REQUEST };
    (status, Json(value)).into_response()
}

/// `GET /api/studio/search/engines`: the engines and where they come from.
pub(in crate::server) async fn engines_route() -> Response {
    let list: Vec<Value> = engines().into_iter().map(|e| {
        let mut entry = e.clone();
        entry["name"] = json!(label(&e));
        entry
    }).collect();
    Json(json!({"engines": list, "file": file().display().to_string()})).into_response()
}

/// `POST /api/studio/search/engines` `{engines: [{kind, name?, url?, key?}], limit?}`.
pub(in crate::server) async fn engines_save(Json(body): Json<Value>) -> Response {
    let Some(list) = body.get("engines").and_then(Value::as_array) else {
        return api_error(StatusCode::BAD_REQUEST, "engines has to be a list");
    };
    let mut clean: Vec<Value> = Vec::new();
    for item in list {
        let kind = item.get("kind").and_then(Value::as_str).unwrap_or("");
        if !["ddg", "wikipedia", "searxng", "brave"].contains(&kind) {
            return api_error(StatusCode::BAD_REQUEST, "an engine has to be ddg, wikipedia, searxng or brave");
        }
        if kind == "searxng" {
            let url = item.get("url").and_then(Value::as_str).unwrap_or("").trim();
            if !(url.starts_with("http://") || url.starts_with("https://")) {
                return api_error(StatusCode::BAD_REQUEST, "a SearXNG engine needs an http(s) address");
            }
        }
        let mut entry = json!({"kind": kind});
        for key in ["name", "url", "key"] {
            if let Some(text) = item.get(key).and_then(Value::as_str) {
                let text = text.trim();
                if !text.is_empty() {
                    entry[key] = json!(text.chars().take(300).collect::<String>());
                }
            }
        }
        clean.push(entry);
    }
    if clean.len() > 12 {
        return api_error(StatusCode::BAD_REQUEST, "at most 12 engines");
    }
    let mut document = read();
    document["engines"] = json!(clean);
    if let Some(limit) = body.get("limit").and_then(Value::as_u64) {
        document["limit"] = json!(limit.clamp(1, MAX_RESULTS as u64));
    }
    let path = file();
    if let Some(parent) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            return api_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("cannot create {}: {e}", parent.display()));
        }
    }
    let text = serde_json::to_string_pretty(&document).unwrap_or_else(|_| "{}".to_owned());
    if let Err(e) = std::fs::write(&path, text) {
        return api_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("cannot write {}: {e}", path.display()));
    }
    engines_route().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn without_a_file_two_engines_need_nothing() {
        let list = engines();
        assert!(list.iter().any(|e| e["kind"] == "ddg"));
        assert!(list.iter().any(|e| e["kind"] == "wikipedia"));
    }

    #[test]
    fn snippets_lose_their_markup() {
        let text = r#"<span class="searchmatch">Axum</span> ist ein &quot;Web&quot;-Framework &amp; mehr"#;
        assert_eq!(regex_snippet(text), "Axum ist ein \"Web\"-Framework & mehr");
    }

    #[test]
    fn the_limit_is_clamped() {
        assert_eq!(limit_of(&json!({"limit": 3})), 3);
        assert_eq!(limit_of(&json!({"limit": 0})), 1);
        assert_eq!(limit_of(&json!({"limit": 999})), MAX_RESULTS);
        assert_eq!(limit_of(&json!({})), DEFAULT_RESULTS);
    }

    #[test]
    fn an_empty_query_is_an_error_not_an_empty_result() {
        let value = futures_block(search("   ", 5));
        assert_eq!(value["ok"], false);
    }

    /// The search is async; the test needs the answer, so it gets its own small runtime.
    fn futures_block<F: std::future::Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread().enable_all().build().expect("runtime").block_on(future)
    }
}
