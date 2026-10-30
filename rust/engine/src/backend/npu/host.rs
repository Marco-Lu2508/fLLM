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

/// `--device npu` on a computer with an Intel NPU: the host decoder (fast CPU kernels, memory-mapped weights) runs the model and hands the
/// long prompt projections to the NPU, exactly like `FLLM_NPU_HOST=1`; decode stays on the processor. The hybrid backend (every product
/// through the reference CPU backend, a second copy of the weights) is used only with `--npu-mode all` or `FLLM_NPU_HYBRID=1`.
///
/// Returns the device the rest of the start-up should use (`cpu`), or `None` when the request is not for an Intel NPU or the hybrid
/// backend was asked for. When the NPU cannot be used the reason is recorded for the Hardware page and the model runs on the processor.
pub fn route_device(device: &str) -> Option<String> {
    let name = device.trim().to_ascii_lowercase();
    let index = match name.as_str() {
        "npu" | "openvino" => 0,
        other => other.strip_prefix("npu:").or_else(|| other.strip_prefix("openvino:"))?.parse::<usize>().ok()?,
    };
    if std::env::var("FLLM_NPU_HYBRID").is_ok_and(|v| !matches!(v.trim(), "" | "0" | "off" | "false" | "no")) {
        return None;
    }
    // an AMD NPU with fLLM's native kernels keeps its own route (docs/npu-xdna.md)
    #[cfg(feature = "npu-xdna-native")]
    if super::xdna::native::available().is_ok() {
        return None;
    }
    if index != 0 {
        return None;
    }
    let found = super::openvino::devices::enumerate();
    let Some(first) = found.first() else {
        let message = super::status::record_fallback(&format!("no usable NPU: {}", super::status::state_text(super::status::state())));
        eprintln!("[fllm-rs] --device {device}: {message}");
        return Some("cpu".to_owned());
    };
    // an explicit --device npu wins over FLLM_NPU_HOST=0
    std::env::set_var("FLLM_NPU_HOST", "1");
    eprintln!(
        "[fllm-rs] --device {device}: the model runs on the processor and its long prompt projections on the NPU ({}); --npu-mode all or FLLM_NPU_HYBRID=1 selects the hybrid backend",
        first.full_name
    );
    Some("cpu".to_owned())
}

#[cfg(test)]
mod tests {
    use super::route_device;

    #[test]
    fn only_requests_for_the_first_npu_are_rerouted() {
        // none of these is an NPU request (or none for the first NPU), so nothing may change and nothing may be opened
        for device in ["auto", "cpu", "vulkan:0", "cuda", "npu:x", "npu:1", "npu:", "hip:0"] {
            assert_eq!(route_device(device), None, "{device}");
        }
    }
}
