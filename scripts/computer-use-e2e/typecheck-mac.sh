#!/bin/bash
# Type-checks and lints the macOS layer of computer use (rust/engine/src/server/api/agent_computer_mac.rs) for both macOS targets
# without a Mac. The whole engine cannot be checked this way (its C dependencies need the macOS SDK), so the file is checked in a
# scratch crate next to stubs of the shared types it uses. Needs docker. Usage: scripts/computer-use-e2e/typecheck-mac.sh [repo]
set -eu
repo="$(cd "${1:-.}" && pwd)"
work="$(mktemp -d)"
mkdir -p "$work/src"
cp "$repo/rust/engine/src/server/api/agent_computer_mac.rs" "$work/src/agent_computer_mac.rs"
cat > "$work/Cargo.toml" <<'EOT'
[package]
name = "cua-mac"
version = "0.0.0"
edition = "2021"
rust-version = "1.89"
[lib]
path = "src/lib.rs"
EOT
cat > "$work/src/lib.rs" <<'EOT'
//! Stubs of the shared types of agent_computer.rs, so that the macOS layer can be checked on its own.
pub mod agent_computer {
    #[derive(Clone, Debug, PartialEq)]
    pub struct DisplayInfo { pub index: usize, pub name: String, pub primary: bool, pub origin_x: i32, pub origin_y: i32, pub width: u32, pub height: u32, pub scale: f64, pub ui_scale: f64 }
    pub struct Frame { pub width: u32, pub height: u32, pub rgba: Vec<u8> }
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Button { Left, Middle, Right }
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Named { Enter, Tab, Escape, Backspace, Delete, Insert, Home, End, PageUp, PageDown, Up, Down, Left, Right, Space, CapsLock, PrintScreen, Menu, Meta, F(u8) }
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Key { Char(char), Named(Named) }
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Chord { pub ctrl: bool, pub alt: bool, pub shift: bool, pub meta: bool, pub key: Key }
}
#[cfg(target_os = "macos")]
pub mod agent_computer_mac;
EOT
docker run --rm -v "$work:/src" -e CARGO_TARGET_DIR=/tmp/t rust:1-trixie bash -c '
  rustup component add clippy >/dev/null 2>&1
  for t in aarch64-apple-darwin x86_64-apple-darwin; do
    rustup target add "$t" >/dev/null 2>&1
    echo "== $t"
    cd /src && cargo clippy --target "$t" -- -D warnings 2>&1 | tail -40
  done'
rm -rf "$work"
