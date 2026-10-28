//! Self-contained browser app (the Studio). One page built from the files in `webui/`: the shell markup, the
//! stylesheet and the script parts, all embedded in the binary and inlined, so the page needs nothing from the
//! network. API credentials never enter the page source.
use axum::{
    http::header,
    response::{Html, IntoResponse, Response},
};
use std::sync::OnceLock;

const SHELL: &str = include_str!("webui/shell.html");
const STYLE: [&str; 4] = [
    include_str!("webui/style.css"),
    include_str!("webui/chat.css"),
    include_str!("webui/code.css"),
    // the design system v2 comes last: it refines the older sheets without rewriting every selector
    include_str!("webui/theme.css"),
];
/// The script parts, joined in this order inside one function scope.
const SCRIPT: [&str; 34] = [
    include_str!("webui/core.js"),
    include_str!("webui/prompts.js"),
    include_str!("webui/markdown.js"),
    include_str!("webui/sampler.js"),
    include_str!("webui/chat.js"),
    include_str!("webui/artifacts.js"),
    include_str!("webui/rag.js"),
    include_str!("webui/projects.js"),
    include_str!("webui/research.js"),
    include_str!("webui/tools.js"),
    include_str!("webui/codeprompts.js"),
    include_str!("webui/code.js"),
    include_str!("webui/computer.js"),
    include_str!("webui/panels.js"),
    include_str!("webui/mcp.js"),
    include_str!("webui/hooks.js"),
    include_str!("webui/agents.js"),
    include_str!("webui/usage.js"),
    include_str!("webui/plugins.js"),
    include_str!("webui/inbox.js"),
    include_str!("webui/runtime.js"),
    include_str!("webui/searchengines.js"),
    include_str!("webui/skilllibrary.js"),
    include_str!("webui/providers.js"),
    include_str!("webui/skills.js"),
    include_str!("webui/sessions.js"),
    include_str!("webui/npu.js"),
    include_str!("webui/models.js"),
    include_str!("webui/engine.js"),
    include_str!("webui/listeners-panels.js"),
    include_str!("webui/desktop.js"),
    include_str!("webui/voice.js"),
    include_str!("webui/widgets.js"),
    include_str!("webui/boot.js"),
];

fn built() -> &'static str {
    static PAGE: OnceLock<String> = OnceLock::new();
    PAGE.get_or_init(|| {
        SHELL
            .replace("/*__STYLE__*/", &STYLE.concat())
            .replace("/*__SCRIPT__*/", &format!("(() => {{\n'use strict';\n{}\n}})();", SCRIPT.concat()))
    })
}

/// Development mode: `FLLM_STUDIO_DEV=<dir>` serves the page from that directory on every request instead of the
/// copy embedded at build time, so a change to a CSS or JS file shows up on reload without a rebuild. The embedded
/// copy stays the default (a release binary carries the page and needs nothing on disk).
fn dev_file(dir: &std::path::Path, name: &str) -> Option<String> {
    std::fs::read_to_string(dir.join(name)).ok()
}

fn dev_page(dir: &std::path::Path) -> Option<String> {
    let shell = dev_file(dir, "shell.html")?;
    let mut style = String::new();
    for name in ["style.css", "chat.css", "code.css", "theme.css"] {
        style.push_str(&dev_file(dir, name)?);
    }
    let mut script = String::new();
    for name in [
        "core.js", "prompts.js", "markdown.js", "sampler.js", "chat.js", "artifacts.js", "rag.js",
        "projects.js", "research.js", "tools.js", "codeprompts.js", "code.js", "computer.js", "panels.js", "mcp.js",
        "hooks.js", "agents.js", "usage.js", "plugins.js", "inbox.js", "runtime.js", "searchengines.js", "skilllibrary.js", "providers.js", "skills.js", "sessions.js", "npu.js", "models.js", "engine.js", "listeners-panels.js",
        "desktop.js", "voice.js", "widgets.js", "boot.js",
    ] {
        script.push_str(&dev_file(dir, name)?);
    }
    Some(
        shell
            .replace("/*__STYLE__*/", &style)
            .replace("/*__SCRIPT__*/", &format!("(() => {{\n'use strict';\n{}\n}})();", script)),
    )
}

fn dev_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("FLLM_STUDIO_DEV").map(std::path::PathBuf::from).filter(|p| p.is_dir())
}

pub fn page(api_key_required: bool) -> Response {
    let dev = dev_dir().and_then(|dir| dev_page(&dir));
    let page = match dev {
        Some(page) => page,
        None => built().to_string(),
    }
    .replace("__API_KEY_REQUIRED__", if api_key_required { "true" } else { "false" });
    ([
        (header::CACHE_CONTROL, "no-store"),
        (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
        (header::CONTENT_SECURITY_POLICY, "default-src 'none'; connect-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self' data: blob:; worker-src 'self'; font-src 'self' data:; base-uri 'none'; frame-ancestors 'none'; form-action 'none'"),
    ], Html(page)).into_response()
}

/// The Web Worker that runs JavaScript code blocks of the chat. It is a separate script (not a Blob) so that it carries
/// its own policy: it may evaluate code but can load and connect to nothing, while the page keeps a policy without
/// `unsafe-eval`.
pub fn runner() -> Response {
    ([
        (header::CONTENT_TYPE, "text/javascript; charset=utf-8"),
        (header::CACHE_CONTROL, "no-store"),
        (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
        (header::CONTENT_SECURITY_POLICY, "default-src 'none'; script-src 'unsafe-eval'"),
    ], include_str!("webui/runner.js")).into_response()
}