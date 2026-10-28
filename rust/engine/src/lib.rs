// The NEON dot-product and f16 intrinsics were stabilised after the MSRV of the x86 builds (1.89); only aarch64 builds need the newer
// compiler, so the lint that compares them with the MSRV is silenced there.
#![cfg_attr(target_arch = "aarch64", allow(clippy::incompatible_msrv))]
pub mod autotune;
pub mod beam;
pub mod backend;
pub mod cache;
pub mod telemetry;
pub mod chat_template;
pub mod bench;
pub mod bench_kernel;
pub mod comm;
pub mod compute;
pub mod config;
pub mod cpu_arm;
pub mod cpu_attention;
mod cpu_bnb;
pub mod cpu_avx512;
pub mod cpu_elementwise;
pub(crate) mod cpu_delta;
pub(crate) mod cpu_kda;
pub mod cpu_float;
pub mod cpu_kernels;
#[cfg(target_arch = "x86_64")]
mod cpu_kquant;
mod cpu_profile;
#[cfg(target_arch = "x86_64")]
mod cpu_q4z;
pub mod cpu_quantized;
pub mod cpu_selftest;
mod cpu_schedule;
mod cpu_spin;
pub mod cpu_threads;
pub mod cpu_topology;
pub mod draft;
pub(crate) mod kv_blocks;
pub mod speculative;
pub mod encoder;
pub mod encoder_decoder;
pub mod encoder_heads;
pub mod encoder_lora;
pub mod encoder_server;
pub mod metrics;
pub mod webui;
#[cfg(feature = "rust-gpu")]
pub mod gpu;
pub mod mem;
pub mod parallel;
pub mod parameters;
pub mod reasoning;
pub mod thinking;
pub mod rope;
pub mod rust_decoder;
pub mod sampling;
pub mod structured;
pub mod template_json;
pub mod tool_grammar;
pub mod tool_parsers;
pub mod tools;
pub mod vision;

pub mod runtime;
pub mod scheduler;
pub mod server;
pub mod tiny_model;
pub use runtime::{BatchStats, InferenceRuntime, TokenEvent};

pub mod tokenizer;

#[cfg(test)]
mod import_tests;
#[cfg(test)]
mod tokenizer_convert_tests;

#[cfg(test)]
mod gemma_tests;
#[cfg(test)]
mod spec_tests;

pub mod hybrid_config;

#[cfg(test)]
mod hybrid_tests;
#[cfg(test)]
mod device_fallback_tests;

#[cfg(test)]
mod msl_check;

#[cfg(test)]
mod tiny_models;

pub mod devices;
pub mod lowbit;
pub mod mla_latent;
pub mod model;
pub use devices::{compiled_backends, device_caps, is_backend_available, list_devices, DeviceCaps, DeviceInfo, DeviceType};
pub use lowbit::{
    dequantize_fp8_e4m3, dequantize_fp8_e5m2, dequantize_nvfp4, gemv_fp8_e4m3, gemv_fp8_e5m2,
    gemv_nvfp4,
};
pub use model::GgufModel;

#[cfg(test)]
mod nvfp4_tests {
    use super::*;

    #[test]
    fn test_nvfp4_dequantize_parity() {
        // 0xF1 in E2M1: low nibble 0x1 -> +0.5, high nibble 0xF -> -6.0
        let mut packed = [0u8; 8];
        packed[0] = 0xF1;
        // 0x38 in E4M3: sign=0, exp=7 (bias 7 -> 2^0 = 1.0), mant=0 -> 1.0
        let scales = [0x38u8];
        let mut dst = [0.0f32; 16];
        dequantize_nvfp4(&packed, &scales, 1.0, &mut dst).expect("dequantize failed");
        assert!((dst[0] - 0.5).abs() < 1e-6);
        assert!((dst[1] - (-6.0)).abs() < 1e-6);
        for item in dst.iter().skip(2) {
            assert_eq!(*item, 0.0);
        }
    }

    #[test]
    fn test_nvfp4_gemv() {
        let packed = [0xF1, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let scales = [0x38];
        let mut x = [0.0f32; 16];
        x[0] = 2.0;
        x[1] = 1.0;
        let mut y = [0.0f32; 1];
        gemv_nvfp4(&x, &packed, &scales, 1.0, &mut y, 1, 16).expect("gemv failed");
        // y[0] = 2.0 * 0.5 + 1.0 * (-6.0) = 1.0 - 6.0 = -5.0
        assert!((y[0] - (-5.0)).abs() < 1e-5);
    }

    #[test]
    fn test_fp8_e4m3_parity_and_gemv() {
        // byte 0x38 in E4M3 is exactly 1.0f
        // byte 0x7E in E4M3 is exactly 448.0f
        // byte 0xB8 in E4M3 is exactly -1.0f
        let src = [0x38u8, 0x7Eu8, 0xB8u8, 0x00u8];
        let mut dst = [0.0f32; 4];
        dequantize_fp8_e4m3(&src, 1.0, &mut dst).expect("e4m3 dequantize failed");
        assert!((dst[0] - 1.0).abs() < 1e-6);
        assert!((dst[1] - 448.0).abs() < 1e-4);
        assert!((dst[2] - (-1.0)).abs() < 1e-6);
        assert_eq!(dst[3], 0.0);

        // GEMV test: row 0 = [1.0, -1.0]
        let weights = [0x38u8, 0xB8u8];
        let x = [3.0f32, 2.0f32];
        let mut y = [0.0f32; 1];
        gemv_fp8_e4m3(&x, &weights, 1.0, &mut y, 1, 2).expect("e4m3 gemv failed");
        // y[0] = 3.0 * 1.0 + 2.0 * (-1.0) = 1.0
        assert!((y[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_fp8_e5m2_parity_and_gemv() {
        // byte 0x3C in E5M2 is exactly 1.0f
        // byte 0x7B in E5M2 is exactly 57344.0f
        // byte 0xBC in E5M2 is exactly -1.0f
        let src = [0x3Cu8, 0x7Bu8, 0xBCu8, 0x00u8];
        let mut dst = [0.0f32; 4];
        dequantize_fp8_e5m2(&src, 1.0, &mut dst).expect("e5m2 dequantize failed");
        assert!((dst[0] - 1.0).abs() < 1e-6);
        assert!((dst[1] - 57344.0).abs() < 1e-1);
        assert!((dst[2] - (-1.0)).abs() < 1e-6);
        assert_eq!(dst[3], 0.0);

        // GEMV test: row 0 = [1.0, -1.0]
        let weights = [0x3Cu8, 0xBCu8];
        let x = [5.0f32, 2.0f32];
        let mut y = [0.0f32; 1];
        gemv_fp8_e5m2(&x, &weights, 1.0, &mut y, 1, 2).expect("e5m2 gemv failed");
        // y[0] = 5.0 * 1.0 + 2.0 * (-1.0) = 3.0
        assert!((y[0] - 3.0).abs() < 1e-5);
    }
}

pub mod quant_extra;
pub mod quant_iq;

pub mod cli;
pub mod hub;
pub mod version;
pub mod import;
pub mod kv_format;
pub mod kv_geometry;
pub mod catalog;
pub mod ppl;
pub use rust_decoder::{LayerStateSnapshot, RecurrentState, SequenceSnapshot};
