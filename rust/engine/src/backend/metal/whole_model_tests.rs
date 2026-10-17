//! Whole-model tests of the native Metal backend (run by `cargo test --features metal` on an Apple machine; skipped when the machine
//! has no Metal device).
//!
//! Every test builds a tiny random model of one architecture (`crate::tiny_models`: Llama, Qwen3, Mixtral, Qwen3-MoE, Qwen3.5 with
//! gated DeltaNet and an attention output gate), encodes it as F32, F16 or one of the quantized formats, loads it on the host path
//! and on the Metal backend, prefills a 6-token prompt and decodes 16 tokens. The logits of all 16 steps must agree with the host
//! within the tolerance (relative to the largest logit), and a free-running greedy decode must produce the same tokens (up to a
//! step at which the host's two best logits are closer than the tolerance). No layer may fall back to the host.
//!
//! Output (with `--nocapture`), one line per model:
//!
//! ```text
//! metal llama-f32: 16 steps, largest logit difference 3.1e-6 (tolerance 5.0e-3), 16/16 free-running greedy tokens equal, 0 host layer(s), 0 host feed-forward block(s), host tokens [...]
//! ```
//!
//! ```text
//! cargo test --release -p fllm-engine --features metal --lib whole_model -- --nocapture --test-threads=1
//! FLLM_MODEL_TEST_TOL=0.05 ...        # one tolerance for all models of the run
//! FLLM_METAL_SERIAL=1 ...             # first thing to try when a model fails: rules out missing barriers
//! ```

use super::backend::MetalBackend;
use super::device_info::all_devices;
use crate::backend::Backend;
use crate::tiny_models::{build_gguf, compare_with_host, scratch, tolerance, Arch, Encoding};
use std::sync::{Arc, Mutex, OnceLock};

/// One backend for all tests of the process (the pipelines are compiled once). `None`: no Metal device.
fn backend() -> Option<Arc<dyn Backend>> {
    static BACKEND: OnceLock<Mutex<Option<Arc<dyn Backend>>>> = OnceLock::new();
    if all_devices().is_empty() {
        return None;
    }
    let cell = BACKEND.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if guard.is_none() {
        let backend = MetalBackend::new(0).unwrap_or_else(|e| panic!("a Metal device exists but the backend did not start: {e}"));
        *guard = Some(Arc::new(backend));
    }
    guard.clone()
}

fn check(label: &str, arch: Arch, encoding: Encoding, default_tolerance: f32) {
    let Some(backend) = backend() else {
        eprintln!("metal {label}: no Metal device on this machine; skipped");
        return;
    };
    let dir = scratch(label);
    let gguf = build_gguf(&dir, arch, encoding, 0x9e37_79b9_7f4a_7c15);
    let result = compare_with_host(label, &gguf, backend, tolerance(default_tolerance), (0, 0));
    let _ = std::fs::remove_dir_all(&dir);
    match result {
        Ok(line) => eprintln!("metal {line}"),
        Err(error) => panic!("metal {error}"),
    }
}

// Tolerances are relative to the largest logit. Float weights: the kernels accumulate in float, the KV cache is half precision on
// both sides, so the difference is rounding only. Quantized weights: the host path and the GPU kernels quantize the activations
// differently (the host path in blocks of 256 for the K formats), which shows at the percent level on these tiny models.
const FLOAT: f32 = 5e-3;
const QUANT: f32 = 8e-2;

#[test]
fn whole_model_llama_f32() {
    check("llama-f32", Arch::Llama, Encoding::F32, FLOAT);
}

#[test]
fn whole_model_llama_f16() {
    check("llama-f16", Arch::Llama, Encoding::F16, FLOAT);
}

#[test]
fn whole_model_qwen3_f32() {
    check("qwen3-f32", Arch::Qwen3, Encoding::F32, FLOAT);
}

#[test]
fn whole_model_llama_q8_0() {
    check("llama-q8_0", Arch::Llama, Encoding::Quant("q8_0"), QUANT);
}

#[test]
fn whole_model_llama_q4_0() {
    check("llama-q4_0", Arch::Llama, Encoding::Quant("q4_0"), QUANT);
}

#[test]
fn whole_model_llama_q4_k_m() {
    // Q4_K for most tensors, Q6_K for the output head and the last layer's attn_v / ffn_down
    check("llama-q4_k_m", Arch::Llama, Encoding::Quant("q4_k_m"), QUANT);
}

#[test]
fn whole_model_qwen3_q6_k() {
    check("qwen3-q6_k", Arch::Qwen3, Encoding::Quant("q6_k"), QUANT);
}

#[test]
fn whole_model_qwen3_q5_k_m() {
    check("qwen3-q5_k_m", Arch::Qwen3, Encoding::Quant("q5_k_m"), QUANT);
}

#[test]
fn whole_model_mixtral_f16() {
    check("mixtral-f16", Arch::Mixtral, Encoding::F16, FLOAT);
}

#[test]
fn whole_model_mixtral_q4_k_m() {
    check("mixtral-q4_k_m", Arch::Mixtral, Encoding::Quant("q4_k_m"), QUANT);
}

#[test]
fn whole_model_qwen3_moe_q8_0() {
    check("qwen3moe-q8_0", Arch::Qwen3Moe, Encoding::Quant("q8_0"), QUANT);
}

#[test]
fn whole_model_qwen35_f32() {
    check("qwen35-f32", Arch::Qwen35, Encoding::F32, FLOAT);
}

#[test]
fn whole_model_qwen35_q8_0() {
    check("qwen35-q8_0", Arch::Qwen35, Encoding::Quant("q8_0"), QUANT);
}
