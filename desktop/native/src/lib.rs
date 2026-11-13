//! fLLM desktop app, native: the Studio written in Rust with iced, without a web view.
//!
//! The window talks to an fLLM server over HTTP (the same `/v1` API every client uses), so everything the server can do is in reach and the
//! window stays small. This first stage is the chat; Models, Settings and Code follow (see `README.md`).

pub mod api;
pub mod app;
pub mod assistants;
pub mod artifacts;
pub mod chatlist;
pub mod code;
pub mod engine_settings;
pub mod hero;
pub mod md;
pub mod models;
pub mod palette;
pub mod prompts;
pub mod memory;
pub mod promptlib;
pub mod speech;
pub mod tools;
pub mod usage;
pub mod research;
pub mod rag;
pub mod settings;
pub mod tex;
pub mod shortcuts;
pub mod tour;
pub mod sidecar;
pub mod store;
pub mod theme;

/// Starts the window.
pub fn run() -> iced::Result {
    app::run()
}
