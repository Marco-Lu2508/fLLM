//! Error type of the Vulkan core (instance, memory, pipelines, commands).
//!
//! One enum for the whole layer. It converts to `String` (what `backend::create` returns) and to [`BackendError`], so that the integration
//! layer can use `?` in either kind of function.

use crate::backend::BackendError;
use ash::vk;
use std::fmt;

/// Result alias of the Vulkan core.
pub type VkResult<T> = Result<T, VkError>;

/// What can go wrong in the Vulkan core.
#[derive(Debug, Clone, PartialEq)]
pub enum VkError {
    /// `libvulkan` (the loader) is missing or unusable.
    Loader(String),
    /// No usable device, or the requested index does not exist.
    NoDevice(String),
    /// The device lacks a feature or an extension that the layer requires (for example Vulkan 1.2 or timeline semaphores).
    Unsupported(String),
    /// The driver could not provide device or host memory (`requested` is 0 when the failing call does not say how much).
    OutOfMemory { requested: u64 },
    /// `vkDeviceLost`: the device is gone (driver reset, TDR, unplugged). The context cannot be used any more.
    DeviceLost,
    /// A Vulkan call failed.
    Vulkan { call: &'static str, result: vk::Result },
    /// The caller passed something invalid (size 0, out-of-range slice, too many bindings, ...).
    InvalidArgument(String),
    /// A file operation of the pipeline cache failed (never fatal for the context).
    Io(String),
}

impl fmt::Display for VkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Loader(msg) => write!(f, "Vulkan loader unavailable: {msg}"),
            Self::NoDevice(msg) => write!(f, "no Vulkan device: {msg}"),
            Self::Unsupported(msg) => write!(f, "Vulkan device lacks a required capability: {msg}"),
            Self::OutOfMemory { requested: 0 } => write!(f, "Vulkan: out of memory"),
            Self::OutOfMemory { requested } => write!(f, "Vulkan: out of memory (requested {requested} bytes)"),
            Self::DeviceLost => write!(f, "Vulkan device lost"),
            Self::Vulkan { call, result } => write!(f, "{call} failed: {result:?}"),
            Self::InvalidArgument(msg) => write!(f, "Vulkan: invalid argument: {msg}"),
            Self::Io(msg) => write!(f, "Vulkan pipeline cache I/O: {msg}"),
        }
    }
}

impl std::error::Error for VkError {}

impl VkError {
    /// Classify a raw result of the call `call` (out-of-memory and device-lost get their own variants).
    pub fn from_result(call: &'static str, result: vk::Result) -> Self {
        match result {
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY => Self::OutOfMemory { requested: 0 },
            vk::Result::ERROR_DEVICE_LOST => Self::DeviceLost,
            other => Self::Vulkan { call, result: other },
        }
    }

    /// The same error, with the number of bytes the failed allocation asked for (only touches `OutOfMemory`).
    pub fn with_requested(self, bytes: u64) -> Self {
        match self {
            Self::OutOfMemory { .. } => Self::OutOfMemory { requested: bytes },
            other => other,
        }
    }
}

/// `.map_err(vk_call("vkCreateBuffer"))` for a `Result<T, vk::Result>`.
pub fn vk_call(call: &'static str) -> impl Fn(vk::Result) -> VkError {
    move |result| VkError::from_result(call, result)
}

impl From<VkError> for String {
    fn from(error: VkError) -> Self {
        error.to_string()
    }
}

impl From<VkError> for BackendError {
    fn from(error: VkError) -> Self {
        match error {
            VkError::OutOfMemory { requested } => BackendError::OutOfMemory {
                requested: requested as usize,
                available: 0,
            },
            VkError::NoDevice(msg) | VkError::Loader(msg) => BackendError::DeviceNotFound(msg),
            VkError::InvalidArgument(msg) => BackendError::BufferError(msg),
            other => BackendError::DriverError(other.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification() {
        assert_eq!(VkError::from_result("x", vk::Result::ERROR_DEVICE_LOST), VkError::DeviceLost);
        assert_eq!(
            VkError::from_result("x", vk::Result::ERROR_OUT_OF_DEVICE_MEMORY).with_requested(42),
            VkError::OutOfMemory { requested: 42 }
        );
        let other = VkError::from_result("vkFoo", vk::Result::ERROR_INITIALIZATION_FAILED);
        assert_eq!(other.to_string(), "vkFoo failed: ERROR_INITIALIZATION_FAILED");
        assert_eq!(VkError::DeviceLost.with_requested(1), VkError::DeviceLost);
    }

    #[test]
    fn conversions() {
        let s: String = VkError::NoDevice("none".into()).into();
        assert!(s.contains("none"));
        let b: BackendError = VkError::OutOfMemory { requested: 7 }.into();
        assert!(matches!(b, BackendError::OutOfMemory { requested: 7, .. }));
        let b: BackendError = VkError::Loader("no libvulkan".into()).into();
        assert!(matches!(b, BackendError::DeviceNotFound(_)));
    }
}
