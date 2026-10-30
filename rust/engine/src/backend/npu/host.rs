//! The NPU engine of the host decoder (`FLLM_NPU_HOST=1`).
//!
//! The host decoder (`compute.rs`, the fast CPU path with memory-mapped weights) can hand the large projections of a prompt to an NPU
//! engine before it computes them with its CPU kernels. That needs no second copy of the quantized weights and no CPU-backend buffers, so
//! it is the cheap way to use an NPU; the `--device npu` backend (a hybrid with the reference CPU backend) is the thorough one.
//!
//! The engine is chosen on first use: the native AMD XDNA engine when the shim, XRT and a device are there, else the Intel NPU through
//! OpenVINO. `FLLM_NPU_HOST` unset, `0`, `off`, `false` or `no`: no engine, nothing changes.

use super::openvino::engine::NpuOutput;
use crate::backend::QuantType;
use std::sync::OnceLock;

/// What the host decoder needs from an NPU engine.
pub trait HostOffload: Send + Sync {
    /// Fewest prompt rows worth a call.
    fn min_batch(&self) -> usize;
    /// See `XdnaEngine::offload_host`: the first rows of `x` through the projection `raw` (`rows x cols`, format `quant`), without bias.
    fn offload(&self, quant: QuantType, raw: &[u8], rows: usize, cols: usize, x: &[f32], batch: usize) -> Option<NpuOutput>;
}

impl HostOffload for std::sync::Arc<super::openvino::engine::NpuEngine> {
    fn min_batch(&self) -> usize {
        self.plan.min_batch
    }

    fn offload(&self, quant: QuantType, raw: &[u8], rows: usize, cols: usize, x: &[f32], batch: usize) -> Option<NpuOutput> {
        self.offload_host(quant, raw, rows, cols, x, batch)
    }
}

#[cfg(feature = "npu-xdna-native")]
impl HostOffload for std::sync::Arc<super::xdna::native::engine::XdnaEngine> {
    fn min_batch(&self) -> usize {
        self.settings.min_rows
    }

    fn offload(&self, quant: QuantType, raw: &[u8], rows: usize, cols: usize, x: &[f32], batch: usize) -> Option<NpuOutput> {
        self.offload_host(quant, raw, rows, cols, x, batch)
    }
}

/// The engine for the host decoder, opened on first use; `None` when `FLLM_NPU_HOST` is not set or no NPU can be used (logged once).
pub fn engine() -> Option<&'static dyn HostOffload> {
    static HOST: OnceLock<Option<Box<dyn HostOffload>>> = OnceLock::new();
    HOST.get_or_init(|| {
        let wanted = std::env::var("FLLM_NPU_HOST").is_ok_and(|v| !matches!(v.trim(), "" | "0" | "off" | "false" | "no"));
        if !wanted {
            return None;
        }
        #[cfg(feature = "npu-xdna-native")]
        if super::xdna::native::available().is_ok() {
            match super::xdna::native::engine::XdnaEngine::open(0) {
                Ok(engine) => return Some(Box::new(engine) as Box<dyn HostOffload>),
                Err(error) => crate::backend::log_once("npu-host-engine-xdna", &format!("FLLM_NPU_HOST: the native XDNA engine did not open: {error}")),
            }
        }
        super::openvino::host_engine().map(|e| Box::new(std::sync::Arc::clone(e)) as Box<dyn HostOffload>)
    })
    .as_deref()
}
