//! Self-contained browser chat client. API credentials never enter the page source.
use axum::{
    http::header,
    response::{Html, IntoResponse, Response},
};

pub fn page(api_key_required: bool) -> Response {
    let page = include_str!("webui.html").replace(
        "__API_KEY_REQUIRED__",
        if api_key_required { "true" } else { "false" },
    );
    ([
        (header::CACHE_CONTROL, "no-store"),
        (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
        (header::CONTENT_SECURITY_POLICY, "default-src 'none'; connect-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'"),
    ], Html(page)).into_response()
}
