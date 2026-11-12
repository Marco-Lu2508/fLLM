//! fLLM desktop app, native: the Studio written in Rust with iced, without a web view.
//!
//! The window talks to an fLLM server over HTTP (the same `/v1` API every client uses), so everything the server can do is in reach and the
//! window stays small. This first stage is the chat; Models, Settings and Code follow (see `README.md`).

pub mod api;
pub mod app;
pub mod md;
pub mod models;
pub mod store;
pub mod theme;

/// Starts the window.
pub fn run() -> iced::Result {
    app::run()
}
