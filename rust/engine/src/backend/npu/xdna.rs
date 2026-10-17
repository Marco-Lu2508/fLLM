//! NPU backends (Intel OpenVINO, AMD XDNA, Qualcomm QNN, Apple Neural Engine): xdna. Owner: package 11.
//!
//! AMD XDNA (Ryzen AI) through ONNX Runtime with the VitisAI execution provider. Study: `docs/research/npu-study-xdna.md`.
//! What exists: the run-time binding of the ONNX Runtime C API (`ort.rs`, feature `npu-xdna`), detection of the device
//! (`pci.rs`, library + provider check), a minimal ONNX emitter for a single `MatMul` (`onnx.rs`) and the probe that runs it
//! (`probe_matmul`). What does not exist: any LLM compute path; `create` reports that and the study explains why.
//! Verification level: L0 (compiles; never run on a Ryzen AI machine).

#[cfg(any(feature = "npu-xdna", feature = "npu-qnn"))]
pub mod dl;
pub mod onnx;
#[cfg(feature = "npu-xdna")]
pub mod ort;
pub mod pb;
pub mod pci;
pub mod testdata;

#[cfg(feature = "npu-xdna")]
pub use ort::ProbeReport;

#[cfg(feature = "npu-xdna")]
mod enabled {
    use super::{ort, pci};
    use crate::backend::stub::StubBackend;
    use crate::backend::Backend;
    use crate::devices::{DeviceInfo, DeviceType};
    use std::sync::Arc;

    /// Why XDNA cannot be used on this machine, or the loaded runtime if it can.
    fn usable_runtime() -> Result<&'static ort::Runtime, String> {
        let rt = ort::runtime()?;
        if !rt.has_vitisai() {
            return Err(format!(
                "ONNX Runtime {} at {} does not list {}; the Ryzen AI build is required (providers: {})",
                rt.version,
                rt.path,
                ort::VITISAI_PROVIDER,
                rt.providers.join(", ")
            ));
        }
        Ok(rt)
    }

    /// The XDNA NPU as device `first_index`, if ONNX Runtime with the VitisAI provider is loadable. Empty otherwise.
    pub fn list(first_index: usize) -> Vec<DeviceInfo> {
        let Ok(rt) = usable_runtime() else {
            return Vec::new();
        };
        let pci_npus = pci::scan();
        let name = match pci_npus.first() {
            Some(npu) => format!("AMD XDNA NPU ({})", npu.apu),
            None => "AMD XDNA NPU".to_owned(),
        };
        vec![DeviceInfo {
            index: first_index,
            name,
            description: format!(
                "npu-xdna: ONNX Runtime {} (API {}) with VitisAI EP; L0, no LLM compute path (probe only)",
                rt.version, rt.api_version
            ),
            dev_type: DeviceType::Accelerator,
            free_memory_bytes: 0,
            total_memory_bytes: 0,
        }]
    }

    /// There is no XDNA compute path for LLM inference: reports the state of the machine and that nothing is implemented.
    pub fn create(index: usize) -> Result<Arc<dyn Backend>, String> {
        let rt = usable_runtime()?;
        StubBackend::new("npu-xdna", index).map(|b| Arc::new(b) as Arc<dyn Backend>).map_err(|e| {
            format!(
                "{e}; ONNX Runtime {} with the VitisAI EP is present, but fLLM has no XDNA compute path (see docs/research/npu-study-xdna.md)",
                rt.version
            )
        })
    }

    /// Runs the single-MatMul probe (`y[m,n] = x[m,k] . w[k,n]`) through ONNX Runtime and the VitisAI EP. For the test round on a Ryzen AI
    /// machine; see [`ort::Runtime::probe_matmul`].
    pub fn probe_matmul(m: usize, k: usize, n: usize) -> Result<ort::ProbeReport, String> {
        usable_runtime()?.probe_matmul(m, k, n)
    }
}

#[cfg(feature = "npu-xdna")]
pub use enabled::{create, list, probe_matmul};

#[cfg(not(feature = "npu-xdna"))]
mod disabled {
    use crate::backend::Backend;
    use crate::devices::DeviceInfo;
    use std::sync::Arc;

    /// Without the feature `npu-xdna` there is nothing to list.
    pub fn list(_first_index: usize) -> Vec<DeviceInfo> {
        Vec::new()
    }

    /// Without the feature `npu-xdna` the backend does not exist.
    pub fn create(_index: usize) -> Result<Arc<dyn Backend>, String> {
        Err("npu-xdna: this build was compiled without the feature npu-xdna".to_owned())
    }
}

#[cfg(not(feature = "npu-xdna"))]
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
