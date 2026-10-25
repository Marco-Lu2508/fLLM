//! Backend / stream implementation of the npu backend. Owner: package 10 (selection between the NPU providers).
//!
//! `create(index)` picks the provider of the `index`-th NPU in the order `fllm-rs devices` lists them: Intel NPUs through OpenVINO
//! first, then (each behind its own cargo feature, implemented by package 11 in its own file) AMD XDNA, Qualcomm Hexagon, Apple
//! Neural Engine. Within a provider, the index passed on is the position in that provider's own `list`.
//!
//! The OpenVINO provider is a *hybrid* backend ([`hybrid::NpuHybridBackend`]): the NPU runs only the large projections of prompts
//! (`quantized_linear` with enough rows); everything else is the CPU backend's. See `OPENVINO.md` for why a stand-alone NPU backend
//! cannot work and what the host decoder needs in order to use the engine without this wrapper.

use crate::backend::Backend;
use std::sync::Arc;

#[cfg(feature = "npu-openvino")]
pub mod hybrid;

/// Create the backend of the `index`-th NPU device.
pub fn create(index: usize) -> Result<Arc<dyn Backend>, String> {
    #[allow(unused_mut)]
    let mut remaining = index;
    #[allow(unused_mut)]
    let mut providers: Vec<&str> = Vec::new();
    #[cfg(feature = "npu-openvino")]
    {
        providers.push("openvino");
        let found = super::openvino::devices::enumerate().len();
        if remaining < found {
            let backend = hybrid::NpuHybridBackend::open(remaining)?;
            return Ok(Arc::new(backend));
        }
        remaining -= found;
    }
    #[cfg(feature = "npu-xdna")]
    {
        providers.push("xdna");
        let found = super::xdna::list(0).len();
        if remaining < found {
            return super::xdna::create(remaining);
        }
        remaining -= found;
    }
    #[cfg(feature = "npu-qnn")]
    {
        providers.push("hexagon");
        let found = super::hexagon::list(0).len();
        if remaining < found {
            return super::hexagon::create(remaining);
        }
        remaining -= found;
    }
    #[cfg(feature = "npu-ane")]
    {
        providers.push("ane");
        let found = super::ane::list(0).len();
        if remaining < found {
            return super::ane::create(remaining);
        }
        remaining -= found;
    }
    let _ = remaining;
    #[cfg(feature = "npu-openvino")]
    {
        if super::probe::detect().iter().any(|d| d.vendor == super::probe::Vendor::Intel) {
            return Err(format!(
                "no usable NPU: the operating system lists an Intel NPU, but OpenVINO does not ({})",
                super::status::state_text(super::status::state())
            ));
        }
    }
    Err(format!(
        "no NPU device {index} (providers in this build: {}); `fllm-rs devices` lists what was found. For Intel NPUs the OpenVINO runtime, the NPU \
         plugin and the driver are required (rust/engine/src/backend/npu/OPENVINO.md)",
        providers.join(", ")
    ))
}
