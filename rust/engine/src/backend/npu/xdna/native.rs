//! AMD XDNA NPU, driven natively: fLLM's own GEMM program for the NPU, loaded and run through XRT (`--device npu`).
//!
//! This is not the FastFlowLM route (`backend/npu/flm.rs`, which runs AMD's proprietary kernels in a separate server) and not the ONNX
//! Runtime probe in the parent module. The pieces:
//!
//! * [`kernels`]: the `.xclbin` and instruction streams, built from the open `mlir-aie` / Peano toolchain (embedded in the binary);
//! * [`shim`]: run-time binding of a small C++ library over XRT (`rust/engine/xdna/shim`); nothing is linked at build time;
//! * [`engine`]: the offload engine (bf16 GEMM tiles, quantized weights dequantized into NPU buffers) used by the hybrid CPU backend
//!   (`backend/npu/backend/hybrid.rs`), exactly like the Intel NPU route.
//!
//! No NPU, no XRT or no shim: [`list`] returns nothing and [`create`] an error saying which of the three is missing. Details,
//! measurements and the Windows notes: `docs/npu-xdna.md`.

pub mod bench;
pub mod dequant;
pub mod engine;
pub mod kernels;
pub mod shim;

use crate::backend::npu::backend::hybrid::NpuHybridBackend;
use crate::backend::Backend;
use crate::devices::{DeviceInfo, DeviceType};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

static PAUSED: AtomicBool = AtomicBool::new(false);

/// While paused the engine hands every projection back to the CPU. The self test (`selftest-gpu --device npu`) pauses it for its exact
/// pass, which holds the backend to the f32 error bound the NPU's bf16 products cannot meet by design.
pub fn pause_offload(paused: bool) {
    PAUSED.store(paused, Ordering::Relaxed);
}

pub(crate) fn offload_paused() -> bool {
    PAUSED.load(Ordering::Relaxed)
}

/// Number of NPUs usable natively, or the reason there is none.
pub fn available() -> Result<usize, String> {
    if !engine::Settings::from_env().enabled {
        return Err("FLLM_XDNA=0".to_owned());
    }
    let shim = shim::Shim::get()?;
    match shim.device_count()? {
        0 => Err("XRT lists no NPU device (is the amdxdna driver loaded and /dev/accel/accel0 readable?)".to_owned()),
        n => Ok(n),
    }
}

/// The NPUs the native backend can drive, numbered from `first_index`. Empty when the shim, XRT or the device is missing.
pub fn list(first_index: usize) -> Vec<DeviceInfo> {
    let Ok(count) = available() else {
        return Vec::new();
    };
    let name = crate::backend::npu::probe::detect()
        .into_iter()
        .find(|d| d.vendor == crate::backend::npu::probe::Vendor::Amd)
        .map(|d| d.name)
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| "AMD XDNA NPU".to_owned());
    // the NPU shares the memory of the computer: report that, so the planners see what is really there
    let free = crate::devices::host_free_memory_bytes();
    (0..count)
        .map(|i| DeviceInfo {
            index: first_index + i,
            name: name.clone(),
            description: "npu-xdna: native kernels through XRT (bf16 GEMM offload of prompt projections, hybrid with CPU)".to_owned(),
            dev_type: DeviceType::Accelerator,
            free_memory_bytes: free,
            total_memory_bytes: free,
        })
        .collect()
}

/// Open device `index` as a backend: the CPU backend with the large prompt projections on the NPU.
pub fn create(index: usize) -> Result<Arc<dyn Backend>, String> {
    available()?;
    let engine = engine::XdnaEngine::open(index)?;
    Ok(Arc::new(NpuHybridBackend::with_engine(engine, index)))
}

/// What the Studio and `fllm-rs amd-npu status` say about the native route.
pub fn status() -> Value {
    let compiled = true;
    match available() {
        Ok(devices) => json!({
            "compiled": compiled,
            "available": true,
            "devices": devices,
            "shim": shim::Shim::get().ok().map(|s| s.path.display().to_string()),
            "summary": "native fLLM kernels through XRT",
        }),
        Err(reason) => json!({
            "compiled": compiled,
            "available": false,
            "reason": reason,
            "summary": "native XDNA backend not usable here",
        }),
    }
}
