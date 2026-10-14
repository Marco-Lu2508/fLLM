//! NPU backends (Intel OpenVINO, AMD XDNA, Qualcomm QNN, Apple Neural Engine): ane. Owner: package 11.
//!
//! Apple Neural Engine through Core ML ML Programs. Study: `STUDY_ANE.md` next to this file. What exists: an emitter for a one-layer
//! fp16 ML Program (`mil.rs` text, `mlpackage.rs` protobuf + weight blob + manifest; pure Rust, compiled and unit-tested on every host)
//! and, on Apple platforms with the feature `npu-ane`, a small Objective-C runtime layer (`objc.rs`) and the Core ML probe
//! (`coreml.rs`) that compiles, loads (computeUnits = cpuAndNeuralEngine) and runs that package. What does not exist: any LLM compute
//! path; Core ML offers no way to run fLLM's quantized formats on the Neural Engine (see the study).
//! Verification level: L0 (type-checked for aarch64-apple-darwin; never run).

#[cfg(all(feature = "npu-ane", target_vendor = "apple"))]
pub mod coreml;
pub mod mil;
pub mod mlpackage;
#[cfg(all(feature = "npu-ane", target_vendor = "apple"))]
pub mod objc;

#[cfg(all(feature = "npu-ane", target_vendor = "apple"))]
pub use coreml::{ComputeUnits, ProbeReport};

#[cfg(all(feature = "npu-ane", target_vendor = "apple"))]
mod enabled {
    use super::coreml;
    use crate::backend::stub::StubBackend;
    use crate::backend::Backend;
    use crate::devices::{DeviceInfo, DeviceType};
    use std::sync::Arc;

    /// The Neural Engine of this Mac as device `first_index` (Apple Silicon only).
    pub fn list(first_index: usize) -> Vec<DeviceInfo> {
        let Some(info) = coreml::detect() else {
            return Vec::new();
        };
        let cores = info
            .neural_engine_cores
            .map_or_else(String::new, |c| format!(", {c} cores"));
        vec![DeviceInfo {
            index: first_index,
            name: format!("Apple Neural Engine ({})", info.chip),
            description: format!("npu-ane: Core ML ML Program, computeUnits=cpuAndNeuralEngine{cores}; L0, no LLM compute path (probe only)"),
            dev_type: DeviceType::Accelerator,
            free_memory_bytes: 0,
            total_memory_bytes: info.memory_bytes,
        }]
    }

    /// There is no Neural Engine compute path for LLM inference: reports the state of the machine and that nothing is implemented.
    pub fn create(index: usize) -> Result<Arc<dyn Backend>, String> {
        let info = coreml::detect().ok_or_else(|| {
            "npu-ane: no Apple Neural Engine found (Apple Silicon Mac required)".to_owned()
        })?;
        StubBackend::new("npu-ane", index).map(|b| Arc::new(b) as Arc<dyn Backend>).map_err(|e| {
            format!("{e}; the Neural Engine of {} is present, but fLLM has no Core ML compute path (see src/backend/npu/STUDY_ANE.md)", info.chip)
        })
    }

    /// Compiles, loads and runs the one-layer fp16 ML Program through Core ML. For the test round on an Apple Silicon Mac.
    pub fn probe_linear(
        m: usize,
        k: usize,
        n: usize,
        units: coreml::ComputeUnits,
    ) -> Result<coreml::ProbeReport, String> {
        coreml::probe_linear(m, k, n, units)
    }
}

#[cfg(all(feature = "npu-ane", target_vendor = "apple"))]
pub use enabled::{create, list, probe_linear};

#[cfg(not(all(feature = "npu-ane", target_vendor = "apple")))]
mod disabled {
    use crate::backend::Backend;
    use crate::devices::DeviceInfo;
    use std::sync::Arc;

    /// Without the feature `npu-ane` on an Apple platform there is nothing to list.
    pub fn list(_first_index: usize) -> Vec<DeviceInfo> {
        Vec::new()
    }

    /// Without the feature `npu-ane` on an Apple platform the backend does not exist.
    pub fn create(_index: usize) -> Result<Arc<dyn Backend>, String> {
        Err(if cfg!(feature = "npu-ane") {
            "npu-ane: the Apple Neural Engine is only available on Apple platforms".to_owned()
        } else {
            "npu-ane: this build was compiled without the feature npu-ane".to_owned()
        })
    }
}

#[cfg(not(all(feature = "npu-ane", target_vendor = "apple")))]
pub use disabled::{create, list};

#[cfg(test)]
mod tests {
    /// There is no compute path: `create` must always report an error, and `list` must cope with a missing library.
    #[test]
    fn create_never_succeeds_and_list_does_not_panic() {
        let _ = super::list(0);
        assert!(super::create(0).is_err());
    }
}
