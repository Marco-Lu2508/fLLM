//! Version string of the binary: crate version, git commit, build date, cargo features and compiled backends.
//!
//! The commit and the date come from `build.rs`; without a git checkout (a source tarball) the commit is `unknown`.
//! `SOURCE_DATE_EPOCH` fixes the date for reproducible builds.

/// Crate version (the Cargo workspace version).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
/// Short commit hash of the build, or `unknown`.
pub const GIT_SHORT: &str = env!("FLLM_GIT_SHORT");
/// UTC build date, `YYYY-MM-DD`.
pub const BUILD_DATE: &str = env!("FLLM_BUILD_DATE");
/// Enabled cargo features of `fllm-engine`, comma separated, `none` when empty.
pub const FEATURES: &str = env!("FLLM_FEATURES");

/// `fllm-rs 0.1.2 (git abc1234, 2026-10-15) features: avx512 backends: cpu`
pub fn long_version() -> String {
    format!(
        "fllm-rs {VERSION} (git {GIT_SHORT}, {BUILD_DATE}) features: {FEATURES} backends: {}",
        crate::devices::compiled_backends().join(",")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_version_has_every_part() {
        let text = long_version();
        assert!(text.starts_with(&format!("fllm-rs {VERSION} (git ")), "{text}");
        assert!(text.contains(&format!(", {BUILD_DATE}) features: {FEATURES} backends: cpu")), "{text}");
        assert_eq!(BUILD_DATE.len(), 10, "date {BUILD_DATE}");
        assert!(GIT_SHORT == "unknown" || GIT_SHORT.bytes().all(|b| b.is_ascii_hexdigit()), "commit {GIT_SHORT}");
        assert!(!VERSION.is_empty() && !FEATURES.is_empty());
    }
}
