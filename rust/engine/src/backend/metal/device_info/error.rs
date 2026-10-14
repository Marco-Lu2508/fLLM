//! Error type of the native Metal backend (all four core modules use it). Plain data, no Metal types, so the
//! `Display` texts are checkable on any host.

use std::fmt;

/// What can go wrong between the Rust code and the Metal driver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetalError {
    /// No Metal device at the requested index (or none at all).
    NoDevice(String),
    /// The device or the operating system lacks a needed feature.
    Unsupported(String),
    /// An allocation would exceed the per-buffer limit or the memory budget.
    OutOfMemory { requested: usize, limit: usize },
    /// Buffer creation, bounds or alignment problem.
    Buffer(String),
    /// A Metal source unit failed to compile; `message` is the compiler log.
    Compile { unit: String, message: String },
    /// A compute pipeline could not be created (unknown function, bad function constants, driver refusal).
    Pipeline { function: String, message: String },
    /// A dispatch or an encoder call was malformed (zero or oversized thread group, parameter blob too large).
    InvalidDispatch(String),
    /// A command buffer finished with an error (GPU fault, timeout, out of memory on the device).
    Gpu(String),
    /// Could not create a queue, command buffer or encoder.
    Command(String),
}

impl fmt::Display for MetalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoDevice(msg) => write!(f, "no Metal device: {msg}"),
            Self::Unsupported(msg) => write!(f, "not supported by this Metal device: {msg}"),
            Self::OutOfMemory { requested, limit } => {
                write!(
                    f,
                    "Metal out of memory: {requested} bytes requested, limit {limit} bytes"
                )
            }
            Self::Buffer(msg) => write!(f, "Metal buffer error: {msg}"),
            Self::Compile { unit, message } => {
                write!(f, "Metal shader compile error in {unit}: {message}")
            }
            Self::Pipeline { function, message } => {
                write!(f, "Metal pipeline error for '{function}': {message}")
            }
            Self::InvalidDispatch(msg) => write!(f, "invalid Metal dispatch: {msg}"),
            Self::Gpu(msg) => write!(f, "Metal command buffer failed: {msg}"),
            Self::Command(msg) => write!(f, "Metal command error: {msg}"),
        }
    }
}

impl std::error::Error for MetalError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_mentions_the_cause() {
        let e = MetalError::Compile {
            unit: "x.metal".into(),
            message: "expected ';'".into(),
        };
        let text = e.to_string();
        assert!(text.contains("x.metal") && text.contains("expected"));
        assert!(MetalError::OutOfMemory {
            requested: 5,
            limit: 3
        }
        .to_string()
        .contains("5 bytes"));
    }
}
