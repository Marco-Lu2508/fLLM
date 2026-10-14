//! Tunables of the native Metal backend, read once from `FLLM_METAL_*` environment variables.
//!
//! The parsing takes a lookup function so it is checkable without touching the process environment.

use std::path::PathBuf;

/// Default number of dispatches after which a command buffer is committed so that the GPU starts while the CPU
/// still encodes the rest of the step.
pub const DEFAULT_DISPATCHES_PER_COMMAND_BUFFER: usize = 160;

/// Upper bound of the hazard-tracker window (accesses remembered between two barriers) before a barrier is forced.
pub const DEFAULT_HAZARD_WINDOW: usize = 256;

/// Environment-controlled settings; `MetalConfig::from_env()` is what the backend uses.
#[derive(Clone, Debug, PartialEq)]
pub struct MetalConfig {
    /// `FLLM_METAL_SERIAL=1`: serial dispatch type (the driver orders every dispatch after the previous one, no barriers
    /// needed). Slower; the switch to try first when output is wrong, because it removes barrier bugs from the picture.
    pub serial_dispatch: bool,
    /// `FLLM_METAL_CB_DISPATCHES=N`: commit a command buffer after N dispatches (0 = only on flush).
    pub dispatches_per_command_buffer: usize,
    /// `FLLM_METAL_MEM_LIMIT_MB=N`: refuse allocations beyond N MiB in total (default: the device's
    /// `recommendedMaxWorkingSetSize`).
    pub memory_limit_bytes: Option<u64>,
    /// `FLLM_METAL_LIB=path`: load this precompiled `.metallib` instead of compiling the embedded sources.
    pub metallib_path: Option<PathBuf>,
    /// `FLLM_METAL_SINGLE_LIB=1`: compile all sources into one library (default: one library per source file,
    /// compiled on first use, in parallel when pre-warmed).
    pub single_library: bool,
    /// `FLLM_METAL_FAST_MATH=0|1` (default 1): the compiler's fast-math mode.
    pub fast_math: bool,
    /// `FLLM_METAL_NO_NOCOPY=1`: never wrap host memory without a copy; always copy weights into Metal buffers.
    pub force_copy_weights: bool,
    /// `FLLM_METAL_RESIDENCY=0|1` (default 1): keep long-lived buffers in a residency set when the OS has them (macOS 15).
    pub residency_sets: bool,
    /// `FLLM_METAL_DEBUG=1`: object labels and debug groups for GPU captures.
    pub debug_labels: bool,
    /// `FLLM_METAL_MSL=major.minor`: Metal Shading Language version for the source compiler (default: the newest
    /// that the OS supports, at most 3.2).
    pub msl_version: Option<(u16, u16)>,
    /// Accesses remembered by the hazard tracker before a barrier is forced.
    pub hazard_window: usize,
}

impl Default for MetalConfig {
    fn default() -> Self {
        Self {
            serial_dispatch: false,
            dispatches_per_command_buffer: DEFAULT_DISPATCHES_PER_COMMAND_BUFFER,
            memory_limit_bytes: None,
            metallib_path: None,
            single_library: false,
            fast_math: true,
            force_copy_weights: false,
            residency_sets: true,
            debug_labels: false,
            msl_version: None,
            hazard_window: DEFAULT_HAZARD_WINDOW,
        }
    }
}

/// `1`, `true`, `on`, `yes` → `Some(true)`; `0`, `false`, `off`, `no` → `Some(false)`; anything else `None`.
pub fn parse_bool(text: &str) -> Option<bool> {
    match text.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" => Some(true),
        "0" | "false" | "off" | "no" => Some(false),
        _ => None,
    }
}

impl MetalConfig {
    /// Settings from the process environment.
    pub fn from_env() -> Self {
        Self::from_lookup(|key| std::env::var(key).ok())
    }

    /// Settings from an arbitrary variable lookup (tests pass a closure over a map).
    pub fn from_lookup<F: Fn(&str) -> Option<String>>(lookup: F) -> Self {
        let mut config = Self::default();
        let flag =
            |key: &str, default: bool| lookup(key).and_then(|v| parse_bool(&v)).unwrap_or(default);
        config.serial_dispatch = flag("FLLM_METAL_SERIAL", config.serial_dispatch);
        config.single_library = flag("FLLM_METAL_SINGLE_LIB", config.single_library);
        config.fast_math = flag("FLLM_METAL_FAST_MATH", config.fast_math);
        config.force_copy_weights = flag("FLLM_METAL_NO_NOCOPY", config.force_copy_weights);
        config.residency_sets = flag("FLLM_METAL_RESIDENCY", config.residency_sets);
        config.debug_labels = flag("FLLM_METAL_DEBUG", config.debug_labels);
        if let Some(n) =
            lookup("FLLM_METAL_CB_DISPATCHES").and_then(|v| v.trim().parse::<usize>().ok())
        {
            config.dispatches_per_command_buffer = n;
        }
        if let Some(mb) =
            lookup("FLLM_METAL_MEM_LIMIT_MB").and_then(|v| v.trim().parse::<u64>().ok())
        {
            config.memory_limit_bytes = Some(mb.saturating_mul(1024 * 1024));
        }
        config.metallib_path = lookup("FLLM_METAL_LIB")
            .filter(|v| !v.trim().is_empty())
            .map(PathBuf::from);
        config.msl_version =
            lookup("FLLM_METAL_MSL").and_then(|v| super::platform::parse_msl_version(&v));
        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn config_of(pairs: &[(&str, &str)]) -> MetalConfig {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        MetalConfig::from_lookup(|key| map.get(key).cloned())
    }

    #[test]
    fn defaults_without_environment() {
        let config = config_of(&[]);
        assert_eq!(config, MetalConfig::default());
        assert!(config.fast_math && config.residency_sets && !config.serial_dispatch);
        assert_eq!(
            config.dispatches_per_command_buffer,
            DEFAULT_DISPATCHES_PER_COMMAND_BUFFER
        );
    }

    #[test]
    fn reads_every_variable() {
        let config = config_of(&[
            ("FLLM_METAL_SERIAL", "1"),
            ("FLLM_METAL_CB_DISPATCHES", "32"),
            ("FLLM_METAL_MEM_LIMIT_MB", "2048"),
            ("FLLM_METAL_LIB", "/tmp/x.metallib"),
            ("FLLM_METAL_SINGLE_LIB", "on"),
            ("FLLM_METAL_FAST_MATH", "off"),
            ("FLLM_METAL_NO_NOCOPY", "true"),
            ("FLLM_METAL_RESIDENCY", "0"),
            ("FLLM_METAL_DEBUG", "yes"),
            ("FLLM_METAL_MSL", "3.1"),
        ]);
        assert!(
            config.serial_dispatch
                && config.single_library
                && config.force_copy_weights
                && config.debug_labels
        );
        assert!(!config.fast_math && !config.residency_sets);
        assert_eq!(config.dispatches_per_command_buffer, 32);
        assert_eq!(config.memory_limit_bytes, Some(2048 * 1024 * 1024));
        assert_eq!(config.metallib_path, Some(PathBuf::from("/tmp/x.metallib")));
        assert_eq!(config.msl_version, Some((3, 1)));
    }

    #[test]
    fn garbage_keeps_defaults() {
        let config = config_of(&[
            ("FLLM_METAL_SERIAL", "maybe"),
            ("FLLM_METAL_CB_DISPATCHES", "many"),
            ("FLLM_METAL_LIB", "  "),
        ]);
        assert_eq!(config, MetalConfig::default());
    }
}
