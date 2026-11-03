//! Diagnostics of the Vulkan backend: the knobs that make a stall visible instead of silent.
//!
//! | variable | meaning |
//! |---|---|
//! | `FLLM_VK_TRACE=1` | one line per uploaded weight and per compiled pipeline (name, size, milliseconds) |
//! | `FLLM_VK_WAIT_TIMEOUT_S=N` | longest wait for one GPU submission before it is reported as a hang (default 180, `0` = wait forever) |
//! | `FLLM_VK_SLOW_MS=N` | a pipeline compile or a weight upload that takes longer is logged even without `FLLM_VK_TRACE` (default 3000) |

use std::time::Duration;

/// `name` is set to something other than `0`.
fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| !v.is_empty() && v != "0")
}

/// `FLLM_VK_TRACE` is set (read once).
pub fn trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| env_flag("FLLM_VK_TRACE"))
}

/// Default of `FLLM_VK_WAIT_TIMEOUT_S`: a single submission is one decode step or one prompt chunk; three minutes is far beyond either.
pub const DEFAULT_WAIT_TIMEOUT_S: u64 = 180;

/// The wait limit from the text of `FLLM_VK_WAIT_TIMEOUT_S` (`None`: unset or unreadable = the default; `0` = no limit).
pub fn parse_wait_timeout(value: Option<&str>) -> Duration {
    match value.and_then(|v| v.trim().parse::<u64>().ok()) {
        Some(0) => Duration::from_secs(u64::MAX / 4),
        Some(s) => Duration::from_secs(s),
        None => Duration::from_secs(DEFAULT_WAIT_TIMEOUT_S),
    }
}

/// Longest wait for one submission (read once).
pub fn wait_timeout() -> Duration {
    static LIMIT: std::sync::OnceLock<Duration> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| parse_wait_timeout(std::env::var("FLLM_VK_WAIT_TIMEOUT_S").ok().as_deref()))
}

/// Threshold from the text of `FLLM_VK_SLOW_MS`.
pub fn parse_slow(value: Option<&str>) -> Duration {
    Duration::from_millis(value.and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(3000))
}

/// An operation slower than this is logged (read once).
pub fn slow_threshold() -> Duration {
    static LIMIT: std::sync::OnceLock<Duration> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| parse_slow(std::env::var("FLLM_VK_SLOW_MS").ok().as_deref()))
}

/// Log `what` when tracing is on or when it took longer than the slow threshold.
pub fn note_duration(what: &str, took: Duration) {
    if trace() || took >= slow_threshold() {
        eprintln!("[fllm-vk] {what}: {} ms", took.as_millis());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wait_limit_defaults_parses_and_can_be_switched_off() {
        assert_eq!(parse_wait_timeout(None), Duration::from_secs(180));
        assert_eq!(parse_wait_timeout(Some("garbage")), Duration::from_secs(180));
        assert_eq!(parse_wait_timeout(Some(" 30 ")), Duration::from_secs(30));
        assert!(parse_wait_timeout(Some("0")) > Duration::from_secs(365 * 24 * 3600));
    }

    #[test]
    fn slow_threshold_defaults_to_three_seconds() {
        assert_eq!(parse_slow(None), Duration::from_millis(3000));
        assert_eq!(parse_slow(Some("250")), Duration::from_millis(250));
    }
}
