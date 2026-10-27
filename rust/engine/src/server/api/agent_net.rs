//! The network tools of the coding agent and the chat page: `web_fetch` (a page as readable text) and
//! `web_search` (best effort through an HTML search endpoint that needs no key).
//!
//! Safety: only http and https, no credentials in the URL, the host name is resolved here and every answer
//! must be a public address (loopback, private, link-local, metadata and reserved ranges are refused), the
//! connection is pinned to the checked address, redirects are followed by hand with the same checks on every
//! hop, and the body and the time are limited. `FLLM_AGENT_ALLOW_PRIVATE_URLS=1` lifts the address check
//! (for documentation servers on the local network).
use super::agent::{agent_disabled, disabled_reply, result_json};
use super::*;
use crate::server::remote_image::is_blocked;
use regex::Regex;
use std::{net::{IpAddr, SocketAddr}, sync::OnceLock, time::Duration};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const TOTAL_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_BODY: usize = 2 << 20;
const MAX_REDIRECTS: usize = 5;
const DEFAULT_CHARS: usize = 20_000;
const MAX_CHARS: usize = 100_000;
const DEFAULT_RESULTS: usize = 8;
const MAX_RESULTS: usize = 20;
const USER_AGENT: &str = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0";

/// `FLLM_AGENT_ALLOW_PRIVATE_URLS=1` lets the network tools reach loopback and private addresses.
fn allow_private_from_env() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("FLLM_AGENT_ALLOW_PRIVATE_URLS").is_ok_and(|v| v.trim() == "1"))
}

struct Fetched {
    url: String,
    status: u16,
    content_type: String,
    body: Vec<u8>,
    /// The body was longer than the cap and was cut.
    cut: bool,
}

/// Resolves `host:port` and returns an address that passes the policy.
async fn checked_address(host: &str, port: u16, allow_private: bool) -> Result<SocketAddr, String> {
    let addresses: Vec<SocketAddr> = if let Ok(ip) = host.trim_matches(['[', ']']).parse::<IpAddr>() {
        vec![SocketAddr::new(ip, port)]
    } else {
        tokio::net::lookup_host((host, port))
            .await
            .map_err(|e| format!("cannot resolve '{host}': {e}"))?
            .collect()
    };
    if addresses.is_empty() {
        return Err(format!("'{host}' has no address"));
    }
    if !allow_private {
        // every answer must be public: a name that also resolves to an internal address is refused
        if let Some(bad) = addresses.iter().find(|a| is_blocked(a.ip())) {
            return Err(format!(
                "URL refused: host '{host}' resolves to {} (private, loopback, link-local and reserved addresses are blocked; set FLLM_AGENT_ALLOW_PRIVATE_URLS=1 to allow them)",
                bad.ip()
            ));
        }
    }
    Ok(addresses[0])
}

async fn fetch_raw(url: &str, accept: &str, allow_private: bool) -> Result<Fetched, String> {
    match tokio::time::timeout(TOTAL_TIMEOUT, fetch_inner(url, accept, allow_private)).await {
        Ok(result) => result,
        Err(_) => Err(format!("the request took longer than {} s", TOTAL_TIMEOUT.as_secs())),
    }
}

async fn fetch_inner(url: &str, accept: &str, allow_private: bool) -> Result<Fetched, String> {
    let mut current = reqwest::Url::parse(url.trim()).map_err(|e| format!("invalid URL: {e}"))?;
    for hop in 0..=MAX_REDIRECTS {
        if !matches!(current.scheme(), "http" | "https") {
            return Err("only http and https URLs are allowed".into());
        }
        if !current.username().is_empty() || current.password().is_some() {
            return Err("URLs with credentials are not allowed".into());
        }
        let host = current.host_str().ok_or("the URL has no host")?.to_owned();
        let port = current.port_or_known_default().ok_or("the URL has no port")?;
        // checked again on every hop, so a redirect cannot lead to an internal address
        let address = checked_address(&host, port, allow_private).await?;
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .resolve(&host, address)
            .timeout(TOTAL_TIMEOUT)
            .connect_timeout(CONNECT_TIMEOUT)
            .user_agent(USER_AGENT)
            .no_proxy()
            .build()
            .map_err(|e| format!("cannot start the request: {e}"))?;
        let mut response = client
            .get(current.clone())
            .header("accept", accept)
            .header("accept-language", "en")
            .send()
            .await
            .map_err(|e| format!("request failed: {}", e.without_url()))?;
        let status = response.status();
        if status.is_redirection() {
            if hop == MAX_REDIRECTS {
                break;
            }
            let location = response
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|v| v.to_str().ok())
                .ok_or("the server redirected without a Location header")?;
            current = current.join(location).map_err(|e| format!("invalid redirect target: {e}"))?;
            continue;
        }
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_ascii_lowercase();
        let mut body: Vec<u8> = Vec::new();
        let mut cut = false;
        while let Some(chunk) = response.chunk().await.map_err(|e| format!("download failed: {}", e.without_url()))? {
            if body.len() + chunk.len() > MAX_BODY {
                body.extend_from_slice(&chunk[..MAX_BODY - body.len()]);
                cut = true;
                break;
            }
            body.extend_from_slice(&chunk);
        }
        return Ok(Fetched { url: current.to_string(), status: status.as_u16(), content_type, body, cut });
    }
    Err(format!("the URL redirects more than {MAX_REDIRECTS} times"))
}

/// A fetched page as text.
#[derive(Debug)]
pub(super) struct Page {
    url: String,
    status: u16,
    content_type: String,
    text: String,
    truncated: bool,
}

async fn fetch_page(url: &str, max_chars: Option<usize>, allow_private: bool) -> Result<Page, String> {
    let max_chars = max_chars.unwrap_or(DEFAULT_CHARS).clamp(500, MAX_CHARS);
    let fetched = fetch_raw(url, "text/html,application/xhtml+xml,text/plain,application/json;q=0.9,*/*;q=0.5", allow_private).await?;
    if !(200..300).contains(&fetched.status) {
        return Err(format!("the server answered HTTP status {}", fetched.status));
    }
    let ct = fetched.content_type.as_str();
    let is_html = ct.contains("html");
    let is_text = ct.is_empty()
        || ct.starts_with("text/")
        || ["json", "xml", "javascript", "yaml", "toml", "markdown"].iter().any(|k| ct.contains(k));
    if !is_html && (!is_text || fetched.body.contains(&0)) {
        return Err(format!("the page is not text (content type '{}')", if ct.is_empty() { "unknown" } else { ct }));
    }
    let raw = String::from_utf8_lossy(&fetched.body);
    let text = if is_html { html_to_text(&raw) } else { raw.into_owned() };
    let mut truncated = fetched.cut;
    let text = if text.chars().count() > max_chars {
        truncated = true;
        text.chars().take(max_chars).collect()
    } else {
        text
    };
    Ok(Page { url: fetched.url, status: fetched.status, content_type: fetched.content_type, text, truncated })
}

fn describe(page: &Page) -> String {
    format!(
        "URL: {}\nStatus: {}\nContent-Type: {}\n\n{}{}",
        page.url,
        page.status,
        if page.content_type.is_empty() { "unknown" } else { page.content_type.as_str() },
        page.text,
        if page.truncated { "\n... content cut" } else { "" }
    )
}

struct HtmlPatterns {
    title: Regex,
    blocks: Vec<Regex>,
    comment: Regex,
    link: Regex,
    list_item: Regex,
    breaks: Regex,
    tag: Regex,
}

fn patterns() -> &'static HtmlPatterns {
    static P: OnceLock<HtmlPatterns> = OnceLock::new();
    P.get_or_init(|| {
        let re = |p: &str| Regex::new(p).expect("fixed pattern");
        HtmlPatterns {
            title: re(r"(?is)<title[^>]*>(.*?)</title\s*>"),
            blocks: ["head", "title", "script", "style", "noscript", "svg", "template", "iframe"]
                .iter()
                .map(|t| re(&format!(r"(?is)<{t}\b.*?</{t}\s*>")))
                .collect(),
            comment: re(r"(?s)<!--.*?-->"),
            link: re(r#"(?is)<a\s[^>]*?href\s*=\s*["']([^"']*)["'][^>]*>(.*?)</a\s*>"#),
            list_item: re(r"(?i)<li\b[^>]*>"),
            breaks: re(r"(?i)</?(?:p|div|br|ul|ol|tr|table|section|article|header|footer|nav|main|aside|h[1-6]|pre|blockquote|hr|form|dd|dt|dl|figure|figcaption)\b[^>]*>"),
            tag: re(r"(?s)<[^>]*>"),
        }
    })
}

/// `&amp;`, `&#39;`, `&#x27;` and a few more named entities replaced by their characters.
fn decode_entities(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find('&') {
        out.push_str(&rest[..at]);
        rest = &rest[at..];
        let decoded = rest[1..].find(';').filter(|&end| end <= 10).and_then(|end| {
            let name = &rest[1..1 + end];
            let ch = match name {
                "amp" => Some('&'),
                "lt" => Some('<'),
                "gt" => Some('>'),
                "quot" => Some('"'),
                "apos" => Some('\''),
                "nbsp" => Some(' '),
                "ndash" => Some('\u{2013}'),
                "mdash" => Some('\u{2014}'),
                "hellip" => Some('\u{2026}'),
                _ => name
                    .strip_prefix("#x")
                    .or_else(|| name.strip_prefix("#X"))
                    .and_then(|h| u32::from_str_radix(h, 16).ok())
                    .or_else(|| name.strip_prefix('#').and_then(|d| d.parse().ok()))
                    .and_then(char::from_u32),
            };
            ch.map(|c| (c, end + 2))
        });
        match decoded {
            Some((c, used)) => {
                out.push(c);
                rest = &rest[used..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// A web page reduced to readable text: scripts and styles gone, blocks on their own lines, links as
/// `text (url)`, entities decoded, white space collapsed.
pub(super) fn html_to_text(html: &str) -> String {
    let p = patterns();
    let title = p.title.captures(html).map(|c| decode_entities(p.tag.replace_all(&c[1], "").trim())).filter(|t| !t.is_empty());
    let mut text = p.comment.replace_all(html, "").into_owned();
    for block in &p.blocks {
        text = block.replace_all(&text, " ").into_owned();
    }
    text = p
        .link
        .replace_all(&text, |c: &regex::Captures| {
            let href = c[1].trim();
            let inner = p.tag.replace_all(&c[2], "");
            let inner = inner.trim();
            if inner.is_empty() || !(href.starts_with("http://") || href.starts_with("https://")) {
                inner.to_owned()
            } else {
                format!("{inner} ({href})")
            }
        })
        .into_owned();
    text = p.list_item.replace_all(&text, "\n- ").into_owned();
    text = p.breaks.replace_all(&text, "\n").into_owned();
    text = decode_entities(&p.tag.replace_all(&text, ""));
    let mut out = String::new();
    if let Some(title) = title {
        out.push_str(&format!("Title: {title}\n\n"));
    }
    let mut blank = true;
    for line in text.lines() {
        let line = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if line.is_empty() {
            if !blank {
                out.push('\n');
            }
            blank = true;
        } else {
            out.push_str(&line);
            out.push('\n');
            blank = false;
        }
    }
    out.trim_end().to_owned()
}

/// One search result.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct SearchHit {
    pub(super) title: String,
    pub(super) url: String,
    pub(super) snippet: String,
}

/// The target of a result link: the search engine wraps it as `/l/?uddg=<encoded url>`.
fn unwrap_result_url(href: &str) -> Option<String> {
    let href = decode_entities(href.trim());
    let base = reqwest::Url::parse("https://duckduckgo.com/").ok()?;
    let url = base.join(&href).ok()?;
    let target = if url.host_str().is_some_and(|h| h.ends_with("duckduckgo.com")) {
        if url.path() == "/y.js" {
            return None; // an advertisement
        }
        url.query_pairs().find(|(k, _)| k == "uddg").map(|(_, v)| v.into_owned())?
    } else {
        url.to_string()
    };
    (target.starts_with("http://") || target.starts_with("https://")).then_some(target)
}

fn parse_search_results(html: &str, limit: usize) -> Vec<SearchHit> {
    static P: OnceLock<(Regex, Regex, Regex)> = OnceLock::new();
    let (anchor, href, snippet) = P.get_or_init(|| {
        (
            Regex::new(r#"(?is)<a\s([^>]*class\s*=\s*"[^"]*\bresult__a\b[^"]*"[^>]*)>(.*?)</a\s*>"#).expect("fixed pattern"),
            Regex::new(r#"(?is)href\s*=\s*"([^"]*)""#).expect("fixed pattern"),
            Regex::new(r#"(?is)<(?:a|td|div)\s[^>]*class\s*=\s*"[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|td|div)\s*>"#).expect("fixed pattern"),
        )
    });
    let tag = &patterns().tag;
    let found: Vec<_> = anchor.captures_iter(html).collect();
    let mut hits = Vec::new();
    for (index, caps) in found.iter().enumerate() {
        if hits.len() >= limit {
            break;
        }
        let whole = caps.get(0).expect("whole match");
        let Some(url) = href.captures(&caps[1]).and_then(|h| unwrap_result_url(&h[1])) else { continue };
        let title = decode_entities(tag.replace_all(&caps[2], "").trim());
        let segment_end = found.get(index + 1).and_then(|n| n.get(0)).map_or(html.len(), |m| m.start());
        let text = snippet
            .captures(&html[whole.end()..segment_end])
            .map(|s| decode_entities(&tag.replace_all(&s[1], "")).split_whitespace().collect::<Vec<_>>().join(" "))
            .unwrap_or_default();
        hits.push(SearchHit { title: title.split_whitespace().collect::<Vec<_>>().join(" "), url, snippet: text });
    }
    hits
}

/// The hits of a search page, or the reason there are none.
pub(super) fn interpret_search(status: u16, html: &str, limit: usize) -> Result<Vec<SearchHit>, String> {
    let blocked = "web search is blocked or rate limited by the search endpoint right now; try again later, or fetch a specific URL with web_fetch";
    if matches!(status, 202 | 403 | 418 | 429 | 503) {
        return Err(blocked.into());
    }
    if !(200..300).contains(&status) {
        return Err(format!("the search endpoint answered HTTP status {status}"));
    }
    let hits = parse_search_results(html, limit);
    if !hits.is_empty() {
        return Ok(hits);
    }
    let lower = html.to_ascii_lowercase();
    if lower.contains("anomaly") || lower.contains("captcha") || lower.contains("unusual traffic") {
        return Err(blocked.into());
    }
    if lower.contains("no results") || lower.contains("no more results") {
        return Ok(hits);
    }
    Err("the search page has no results in a format this tool understands (the endpoint may have changed or blocked the request)".into())
}

#[cfg(test)]
fn format_hits(hits: &[SearchHit]) -> String {
    if hits.is_empty() {
        return "no results".into();
    }
    hits.iter()
        .enumerate()
        .map(|(i, h)| format!("{}. {}\n   {}\n   {}\n", i + 1, h.title, h.url, h.snippet))
        .collect::<Vec<_>>()
        .join("\n")
}

fn size_arg(args: &Value, key: &str) -> Option<usize> {
    args.get(key).and_then(Value::as_u64).map(|n| usize::try_from(n).unwrap_or(usize::MAX))
}

/// A public page as text, for other features that fetch a file (a skill from a URL). Same checks as `web_fetch`.
pub(super) async fn fetch_public_text(url: &str) -> Result<String, String> {
    let page = fetch_page(url, Some(200_000), allow_private_from_env()).await?;
    Ok(page.text)
}

/// Runs a network tool of the agent; `None` when `name` is not one of them.
pub(super) async fn run_tool(name: &str, args: &Value) -> Option<Result<String, String>> {
    match name {
        "web_fetch" => Some(match args.get("url").and_then(Value::as_str) {
            None => Err("the argument 'url' is missing".into()),
            Some(url) => fetch_page(url, size_arg(args, "max_chars"), allow_private_from_env()).await.map(|p| describe(&p)),
        }),
        "web_search" => Some(match args.get("query").and_then(Value::as_str) {
            None => Err("the argument 'query' is missing".into()),
            Some(query) => {
                // every engine the user configured (DuckDuckGo and Wikipedia need no setup)
                let limit = size_arg(args, "limit").unwrap_or(DEFAULT_RESULTS).clamp(1, MAX_RESULTS);
                let value = super::websearch::search(query, limit).await;
                if value.get("ok").and_then(Value::as_bool).unwrap_or(false) {
                    Ok(super::websearch::as_text(&value))
                } else {
                    Err(value.get("error").and_then(Value::as_str).unwrap_or("the search failed").to_owned())
                }
            }
        }),
        _ => None,
    }
}

/// `POST /api/studio/web/fetch`: `{url, max_chars?}` answers `{ok, url, status, content_type, truncated, text}`
/// or `{ok: false, error}`.
pub(in crate::server) async fn fetch_route(Json(body): Json<Value>) -> Response {
    if agent_disabled() {
        return disabled_reply();
    }
    let Some(url) = body.get("url").and_then(Value::as_str) else {
        return api_error(StatusCode::BAD_REQUEST, "the field 'url' is missing");
    };
    match fetch_page(url, size_arg(&body, "max_chars"), allow_private_from_env()).await {
        Ok(p) => Json(json!({"ok": true, "url": p.url, "status": p.status, "content_type": p.content_type, "truncated": p.truncated, "text": p.text}))
            .into_response(),
        Err(e) => Json(result_json(Err(e))).into_response(),
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};

    /// A tiny HTTP server on a free local port; `route` maps the request path to a complete raw response.
    fn serve(route: fn(&str) -> Vec<u8>) -> u16 {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                std::thread::spawn(move || {
                    let mut stream = stream;
                    let mut request = Vec::new();
                    let mut chunk = [0u8; 1024];
                    while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                        match stream.read(&mut chunk) {
                            Ok(0) | Err(_) => return,
                            Ok(n) => request.extend_from_slice(&chunk[..n]),
                        }
                    }
                    let text = String::from_utf8_lossy(&request).into_owned();
                    let path = text.split_whitespace().nth(1).unwrap_or("/").to_owned();
                    let _ = stream.write_all(&route(&path));
                });
            }
        });
        port
    }

    fn response(status: &str, headers: &str, body: &[u8]) -> Vec<u8> {
        let mut out = format!("HTTP/1.1 {status}\r\n{headers}content-length: {}\r\nconnection: close\r\n\r\n", body.len()).into_bytes();
        out.extend_from_slice(body);
        out
    }

    #[test]
    fn html_is_reduced_to_readable_text() {
        let html = "<html><head><title>My &amp; Page</title><style>p{color:red}</style></head><body>\
            <script>alert(1)</script><h1>Hello</h1><p>One   <b>two</b><br>three &lt;x&gt; &#39;q&#39; &#x41;</p>\
            <ul><li>first</li><li>second</li></ul><!-- hidden --><a href=\"https://example.org/a\">a link</a> \
            <a href=\"/relative\">local</a></body></html>";
        let text = html_to_text(html);
        assert!(text.starts_with("Title: My & Page"), "{text}");
        assert!(!text.contains("alert") && !text.contains("color:red") && !text.contains("hidden"), "{text}");
        assert!(text.contains("Hello\n\nOne two\nthree <x> 'q' A"), "{text}");
        assert!(text.contains("- first\n- second"), "{text}");
        assert!(text.contains("a link (https://example.org/a)") && text.contains("local") && !text.contains("/relative"), "{text}");
    }

    #[test]
    fn entities_that_are_not_entities_stay() {
        assert_eq!(decode_entities("a & b &amp;lt; &unknown; &#xZZ; end&"), "a & b &lt; &unknown; &#xZZ; end&");
    }

    const SEARCH_PAGE: &str = r#"<div class="results">
        <div class="result"><h2 class="result__title"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fdocs%3Fa%3D1&amp;rut=abc">Example &amp; <b>Docs</b></a></h2>
        <a class="result__snippet" href="//duckduckgo.com/l/?uddg=x">The   <b>official</b> docs.</a></div>
        <div class="result result--ad"><a class="result__a" href="https://duckduckgo.com/y.js?ad_provider=x">Buy now</a></div>
        <div class="result"><a class="result__a" href="https://direct.example/page">Direct</a>
        <a class="result__snippet" href="x">Second snippet</a></div>
        <div class="result"><a class="result__a" href="ftp://nope.example/">Bad scheme</a></div>
        </div>"#;

    #[test]
    fn search_results_are_parsed_and_unwrapped() {
        let hits = parse_search_results(SEARCH_PAGE, 10);
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert_eq!(hits[0], SearchHit { title: "Example & Docs".into(), url: "https://example.org/docs?a=1".into(), snippet: "The official docs.".into() });
        assert_eq!(hits[1].url, "https://direct.example/page");
        assert_eq!(hits[1].snippet, "Second snippet");
        assert_eq!(parse_search_results(SEARCH_PAGE, 1).len(), 1);
        let text = format_hits(&hits);
        assert!(text.starts_with("1. Example & Docs\n   https://example.org/docs?a=1\n   The official docs."), "{text}");
    }

    #[test]
    fn a_blocked_search_gives_a_clear_error() {
        for status in [202, 403, 429] {
            assert!(interpret_search(status, "", 5).unwrap_err().contains("blocked or rate limited"));
        }
        assert!(interpret_search(200, "<div class=\"anomaly-modal\">", 5).unwrap_err().contains("blocked"));
        assert!(interpret_search(200, "<div>No results.</div>", 5).unwrap().is_empty());
        assert!(interpret_search(200, "<html>something else</html>", 5).unwrap_err().contains("format"));
        assert!(interpret_search(500, "", 5).unwrap_err().contains("500"));
        assert_eq!(interpret_search(200, SEARCH_PAGE, 5).unwrap().len(), 2);
    }

    #[tokio::test]
    async fn private_and_odd_targets_are_refused_before_any_connection() {
        for (url, expect) in [
            ("ftp://example.com/a", "http and https"),
            ("file:///etc/passwd", "http and https"),
            ("http://user:pw@example.com/", "credentials"),
            ("http://127.0.0.1:1/", "refused"),
            ("http://[::1]/", "refused"),
            ("http://169.254.169.254/latest/meta-data/", "refused"),
            ("http://10.1.2.3/", "refused"),
            ("http://localhost/", "refused"),
            ("not a url", "invalid URL"),
        ] {
            let error = fetch_page(url, None, false).await.err().unwrap_or_default();
            assert!(error.contains(expect), "{url}: {error}");
        }
        let tool = run_tool("web_fetch", &json!({"url": "http://127.0.0.1:1/"})).await.unwrap().unwrap_err();
        assert!(tool.contains("FLLM_AGENT_ALLOW_PRIVATE_URLS"), "{tool}");
        assert!(run_tool("web_fetch", &json!({})).await.unwrap().is_err());
        assert!(run_tool("read_file", &json!({})).await.is_none());
    }

    #[tokio::test]
    async fn pages_are_fetched_converted_limited_and_redirects_are_bounded() {
        let port = serve(|path| match path {
            "/page" => response("200 OK", "content-type: text/html; charset=utf-8\r\n", b"<title>T</title><p>Hello <b>world</b></p><script>x()</script>"),
            "/plain" => response("200 OK", "content-type: text/plain\r\n", b"just text"),
            "/missing" => response("404 Not Found", "content-type: text/plain\r\n", b"no"),
            "/image" => response("200 OK", "content-type: image/png\r\n", &[0x89, b'P', b'N', b'G', 0]),
            "/big" => response("200 OK", "content-type: text/plain\r\n", &vec![b'a'; MAX_BODY + 4096]),
            "/loop" => response("302 Found", "location: /loop\r\n", b""),
            "/hop" => response("302 Found", "location: /page\r\n", b""),
            _ => response("500 Oops", "", b""),
        });
        let base = format!("http://127.0.0.1:{port}");
        let page = fetch_page(&format!("{base}/page"), None, true).await.unwrap();
        assert_eq!(page.status, 200);
        assert_eq!(page.text, "Title: T\n\nHello world");
        assert!(!page.truncated);
        assert_eq!(fetch_page(&format!("{base}/plain"), None, true).await.unwrap().text, "just text");
        assert!(fetch_page(&format!("{base}/hop"), None, true).await.unwrap().url.ends_with("/page"), "a redirect is followed");
        assert!(fetch_page(&format!("{base}/missing"), None, true).await.unwrap_err().contains("404"));
        assert!(fetch_page(&format!("{base}/image"), None, true).await.unwrap_err().contains("not text"));
        assert!(fetch_page(&format!("{base}/loop"), None, true).await.unwrap_err().contains("redirects more than"));
        // the body cap and the character limit cut the text
        let big = fetch_page(&format!("{base}/big"), Some(1000), true).await.unwrap();
        assert!(big.truncated && big.text.len() == 1000);
        let capped = fetch_raw(&format!("{base}/big"), "*/*", true).await.unwrap();
        assert!(capped.cut && capped.body.len() == MAX_BODY);
        // the tool output carries the final URL and status
        let out = describe(&page);
        assert!(out.starts_with(&format!("URL: {base}/page\nStatus: 200")), "{out}");
    }
}
