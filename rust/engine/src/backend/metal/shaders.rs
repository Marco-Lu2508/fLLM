//! Metal shader sources. Owner: package 6 (MSL sources embedded by build.rs from metal/*.metal).
//!
//! `build.rs` embeds `rust/engine/metal/*.metal` and generates `metal_sources.rs` in `OUT_DIR` with
//! `SOURCES: &[(&str, &str)]` (file name, MSL source; `common.metal` first). The backend compiles them with `newLibraryWithSource`.

include!(concat!(env!("OUT_DIR"), "/metal_sources.rs"));

/// All sources concatenated in registry order.
pub fn all_sources() -> String {
    SOURCES.iter().map(|(_, s)| *s).collect::<Vec<_>>().join("\n")
}