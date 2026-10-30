//! Intel NPU through OpenVINO (feature `npu-openvino`).
//!
//! * [`ffi`] / [`session`]: the OpenVINO C API loaded at run time (no link dependency) and safe wrappers around it;
//! * [`ir`] / [`model`] / [`quant`]: an OpenVINO IR v11 writer, the graph of one projection (`MatMul` with INT4 / INT8 / FP16 weights)
//!   and the conversion of fLLM's weight formats into the NPU's formats;
//! * [`plan`]: which projections and how many rows are worth running on the NPU (`should_offload`, `plan`);
//! * [`engine`]: the registry of projections, lazy compilation, chunked execution and the CPU fallback;
//! * [`devices`]: the NPU inventory for `fllm-rs devices`.
//!
//! What the NPU does and does not do, the limits and the measurement plan are in `OPENVINO.md` next to this file. Verification level
//! of everything here: L0 (compiled and unit tested without hardware; no NPU, no OpenVINO runtime and no real model were involved).
//!
//! Third-party material: only the public C API signatures and the documented IR format of OpenVINO (Apache-2.0) were used; no
//! source code was copied.
#![cfg(feature = "npu-openvino")]

pub mod bench;
pub mod devices;
pub mod engine;
pub mod ffi;
pub mod ir;
pub mod model;
pub mod plan;
pub mod quant;
pub mod session;

pub use engine::{NpuEngine, WeightKey};
pub use plan::{plan, should_offload, OffloadPlan, PlanEntry, WeightShape};

static HOST: std::sync::OnceLock<Option<std::sync::Arc<NpuEngine>>> = std::sync::OnceLock::new();

/// The host-decoder engine if it has been opened already (never opens it): what the status page reports.
pub fn host_engine_opened() -> Option<&'static std::sync::Arc<NpuEngine>> {
    HOST.get().and_then(Option::as_ref)
}

/// The engine for a *host* decoder (`FLLM_NPU_HOST=1`): opened on first use on the first NPU, `None` when the variable is not set or
/// no NPU can be opened (the reason is logged once). A host decoder calls [`NpuEngine::offload_host`] on it before computing a
/// projection with its CPU kernels; see OPENVINO.md for the call sites.
pub fn host_engine() -> Option<&'static std::sync::Arc<NpuEngine>> {
    HOST.get_or_init(|| {
        let wanted = std::env::var("FLLM_NPU_HOST")
            .is_ok_and(|v| !matches!(v.trim(), "" | "0" | "off" | "false" | "no"));
        if !wanted {
            return None;
        }
        match NpuEngine::open(0) {
            Ok(engine) => Some(engine),
            Err(error) => {
                crate::backend::log_once(
                    "npu-host-engine",
                    &format!("FLLM_NPU_HOST is set but no NPU can be used: {error}"),
                );
                None
            }
        }
    })
    .as_ref()
}

/// Compile and run a small projection on the first NPU and compare it with the host (L1 check for the NPU machine).
pub fn selfcheck(index: usize) -> Result<String, String> {
    let version = session::runtime_version()?;
    let engine = NpuEngine::open(index)?;
    let report = engine.selfcheck()?;
    Ok(format!("OpenVINO {version}: {report}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::QuantType;

    /// Writes sample IR files into `$FLLM_NPU_DUMP_IR` when that variable is set, for the parser check described in OPENVINO.md
    /// (`python3 -c "import openvino as ov; ov.Core().read_model('x.xml')"`); without the variable it only builds them.
    #[test]
    fn sample_irs_build_and_can_be_dumped() {
        let mut raw = Vec::new();
        for i in 0..(64 * 256 / 32) {
            raw.extend_from_slice(&half::f16::from_f32(0.03 + (i % 5) as f32 * 0.01).to_le_bytes());
            raw.extend((0..16).map(|j| (i * 31 + j * 17) as u8));
        }
        let q4_0 = quant::convert(
            QuantType::Q4_0,
            &raw,
            64,
            256,
            &quant::QuantPolicy::default(),
        )
        .unwrap();
        let grouped = quant::convert(
            QuantType::Q4_0,
            &raw,
            64,
            256,
            &quant::QuantPolicy {
                group: 128,
                lossless: false,
                eight: false,
            },
        )
        .unwrap();
        let channel = quant::convert(
            QuantType::Q4_0,
            &raw,
            64,
            256,
            &quant::QuantPolicy {
                group: 0,
                lossless: false,
                eight: false,
            },
        )
        .unwrap();
        let floats: Vec<u8> = (0..64 * 256)
            .flat_map(|i| ((i % 17) as f32 * 0.01 - 0.08).to_le_bytes())
            .collect();
        let f16w = quant::convert(
            QuantType::F32,
            &floats,
            64,
            256,
            &quant::QuantPolicy::default(),
        )
        .unwrap();
        let mut q8 = Vec::new();
        for i in 0..(64 * 256 / 32) {
            q8.extend_from_slice(&half::f16::from_f32(0.01 + (i % 3) as f32 * 0.01).to_le_bytes());
            q8.extend((0..32).map(|j| (i * 3 + j) as u8));
        }
        let int8 = quant::convert(
            QuantType::Q8_0,
            &q8,
            64,
            256,
            &quant::QuantPolicy::default(),
        )
        .unwrap();
        let dir = std::env::var_os("FLLM_NPU_DUMP_IR")
            .filter(|v| !v.is_empty())
            .map(std::path::PathBuf::from);
        for (name, weight) in [
            ("q4_0_g32", &q4_0),
            ("int4_g128", &grouped),
            ("int4_channel", &channel),
            ("f16", &f16w),
            ("int8_g32", &int8),
        ] {
            let ir = model::build_linear_ir(weight, 16, &format!("sample_{name}")).unwrap();
            assert!(ir.xml.contains("<net name=\"sample_"));
            if let Some(dir) = &dir {
                std::fs::create_dir_all(dir).unwrap();
                std::fs::write(dir.join(format!("sample_{name}.xml")), &ir.xml).unwrap();
                std::fs::write(dir.join(format!("sample_{name}.bin")), &ir.bin).unwrap();
                // the dequantized matrix (f32, row-major) for the numeric comparison in Python
                let dequantized: Vec<u8> = weight
                    .dequantize()
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                std::fs::write(dir.join(format!("sample_{name}.weights")), dequantized).unwrap();
                let codes: Vec<String> = weight.signed_codes().iter().map(i8::to_string).collect();
                std::fs::write(dir.join(format!("sample_{name}.codes")), codes.join(" ")).unwrap();
            }
        }
    }

    /// Parser-level check against a real OpenVINO C runtime (no device, no compile): loads the library, enumerates devices, probes the
    /// element type numbering and lets OpenVINO's IR reader parse the generated graphs.
    /// `FLLM_OPENVINO_C_LIB=/path/libopenvino_c.so cargo test --features npu-openvino -- --ignored openvino_reader`
    #[test]
    #[ignore = "needs the OpenVINO C runtime"]
    fn openvino_reader_accepts_generated_ir() {
        let runtime = session::Runtime::shared().expect("OpenVINO C runtime");
        println!(
            "runtime: {} ({})",
            session::runtime_version().unwrap(),
            runtime.api().path.display()
        );
        println!("element types: {:?}", runtime.api().element_types());
        println!("devices: {:?}", runtime.available_devices());
        let mut raw = Vec::new();
        for i in 0..(64 * 256 / 32) {
            raw.extend_from_slice(&half::f16::from_f32(0.03 + (i % 5) as f32 * 0.01).to_le_bytes());
            raw.extend((0..16).map(|j| (i * 31 + j * 17) as u8));
        }
        for policy in [
            quant::QuantPolicy::default(),
            quant::QuantPolicy {
                group: 128,
                lossless: false,
                eight: false,
            },
            quant::QuantPolicy {
                group: 0,
                lossless: false,
                eight: false,
            },
        ] {
            let weight = quant::convert(QuantType::Q4_0, &raw, 64, 256, &policy).unwrap();
            let ir = model::build_linear_ir(&weight, 32, "reader_check").unwrap();
            runtime
                .check_ir(&ir.xml, &ir.bin)
                .unwrap_or_else(|e| panic!("reader rejected the IR for {policy:?}: {e}"));
        }
        let mut broken = model::build_linear_ir(
            &quant::convert(
                QuantType::Q4_0,
                &raw,
                64,
                256,
                &quant::QuantPolicy::default(),
            )
            .unwrap(),
            32,
            "x",
        )
        .unwrap();
        broken.xml = broken
            .xml
            .replace("transpose_b=\"true\"", "transpose_b=\"maybe\"");
        assert!(
            runtime.check_ir(&broken.xml, &broken.bin).is_err(),
            "the reader must reject a bad attribute"
        );
    }

    /// Needs an Intel NPU with its driver and the OpenVINO runtime: `cargo test --features npu-openvino -- --ignored npu_hardware`.
    #[test]
    #[ignore = "needs an Intel NPU and the OpenVINO runtime"]
    fn npu_hardware_selfcheck() {
        let report = selfcheck(0).expect("NPU self-check");
        println!("{report}");
    }
}
