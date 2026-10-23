//! Self-contained browser app (the Studio). One page built from the files in `webui/`: the shell markup, the
//! stylesheet and the script parts, all embedded in the binary and inlined, so the page needs nothing from the
//! network. API credentials never enter the page source.
use axum::{
    http::header,
    response::{Html, IntoResponse, Response},
};
use std::sync::OnceLock;

const SHELL: &str = include_str!("webui/shell.html");
const STYLE: [&str; 3] = [include_str!("webui/style.css"), include_str!("webui/chat.css"), include_str!("webui/code.css")];
/// The script parts, joined in this order inside one function scope.
const SCRIPT: [&str; 10] = [
    include_str!("webui/core.js"),
    include_str!("webui/markdown.js"),
    include_str!("webui/sampler.js"),
    include_str!("webui/chat.js"),
    include_str!("webui/rag.js"),
    include_str!("webui/code.js"),
    include_str!("webui/panels.js"),
    include_str!("webui/mcp.js"),
    include_str!("webui/models.js"),
    include_str!("webui/listeners-panels.js"),
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

pub fn page(api_key_required: bool) -> Response {
    let page = built().replace("__API_KEY_REQUIRED__", if api_key_required { "true" } else { "false" });
    ([
        (header::CACHE_CONTROL, "no-store"),
        (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
        (header::CONTENT_SECURITY_POLICY, "default-src 'none'; connect-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self' data:; base-uri 'none'; frame-ancestors 'none'; form-action 'none'"),
    ], Html(page)).into_response()
}