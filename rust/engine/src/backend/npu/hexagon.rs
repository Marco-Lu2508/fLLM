//! NPU backends (Intel OpenVINO, AMD XDNA, Qualcomm QNN, Apple Neural Engine): hexagon. Owner: package 11.
//!
//! Qualcomm Hexagon NPU (HTP) through the Qualcomm AI Engine Direct SDK (QNN), loaded at run time from `libQnnHtp.so` / `QnnHtp.dll`.
//! Study: `STUDY_HEXAGON.md` next to this file. What exists: the layout of the QNN structures (`qnn.rs`, always compiled and
//! size-checked), the run-time binding and device query (`runtime.rs`), a single-MatMul fp16 graph (`graph.rs`) and a wrapper for the
//! Genie LLM runtime (`genie.rs`), all behind the feature `npu-qnn`. What does not exist: any LLM compute path from fLLM's own ops.
//! Verification level: L0 (compiles; never run on a Snapdragon device).

#[cfg(feature = "npu-qnn")]
pub mod genie;
#[cfg(feature = "npu-qnn")]
pub mod graph;
pub mod qnn;
#[cfg(feature = "npu-qnn")]
pub mod runtime;

#[cfg(feature = "npu-qnn")]
pub use graph::ProbeReport;

#[cfg(feature = "npu-qnn")]
mod enabled {
    use super::{genie, qnn, runtime};
    use crate::backend::stub::StubBackend;
    use crate::backend::Backend;
    use crate::devices::{DeviceInfo, DeviceType};
    use std::sync::Arc;

    /// The HTP devices if the QNN HTP library loads and reports at least one on-chip device. Empty otherwise.
    pub fn list(first_index: usize) -> Vec<DeviceInfo> {
        let Ok(q) = runtime::qnn() else {
            return Vec::new();
        };
        let Ok(devices) = q.devices() else {
            return Vec::new();
        };
        let genie_note = match genie::genie() {
            Ok(g) => format!(
                "; Genie API {}.{}.{}",
                g.api_version.0, g.api_version.1, g.api_version.2
            ),
            Err(_) => String::new(),
        };
        devices
            .iter()
            .enumerate()
            .map(|(i, d)| DeviceInfo {
                index: first_index + i,
                name: format!("Qualcomm Hexagon NPU (HTP {}, SoC model {})", qnn::htp_arch_name(d.arch), d.soc_model),
                description: format!(
                    "npu-qnn: {} core API {}.{}.{}, VTCM {} KiB{}; L0, no LLM compute path (probe and Genie only)",
                    q.provider,
                    q.api_version.core.major,
                    q.api_version.core.minor,
                    q.api_version.core.patch,
                    d.vtcm_bytes / 1024,
                    genie_note
                ),
                dev_type: DeviceType::Accelerator,
                free_memory_bytes: 0,
                total_memory_bytes: 0,
            })
            .collect()
    }

    /// There is no Hexagon compute path for fLLM's own ops: reports the state of the machine and that nothing is implemented.
    pub fn create(index: usize) -> Result<Arc<dyn Backend>, String> {
        let q = runtime::qnn()?;
        StubBackend::new("npu-qnn", index).map(|b| Arc::new(b) as Arc<dyn Backend>).map_err(|e| {
            format!("{e}; the QNN HTP library ({}) is present, but fLLM has no Hexagon compute path (see src/backend/npu/STUDY_HEXAGON.md)", q.path)
        })
    }

    /// Runs the single-MatMul fp16 graph through QNN on the HTP. For the test round on a Snapdragon device.
    pub fn probe_matmul_f16(m: usize, k: usize, n: usize) -> Result<super::ProbeReport, String> {
        runtime::qnn()?.probe_matmul_f16(m, k, n)
    }
}

#[cfg(feature = "npu-qnn")]
pub use enabled::{create, list, probe_matmul_f16};

#[cfg(not(feature = "npu-qnn"))]
mod disabled {
    use crate::backend::Backend;
    use crate::devices::DeviceInfo;
    use std::sync::Arc;

    /// Without the feature `npu-qnn` there is nothing to list.
    pub fn list(_first_index: usize) -> Vec<DeviceInfo> {
        Vec::new()
    }

    /// Without the feature `npu-qnn` the backend does not exist.
    pub fn create(_index: usize) -> Result<Arc<dyn Backend>, String> {
        Err("npu-qnn: this build was compiled without the feature npu-qnn".to_owned())
    }
}

#[cfg(not(feature = "npu-qnn"))]
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
