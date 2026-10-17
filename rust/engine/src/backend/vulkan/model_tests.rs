//! Whole-model check of the native Vulkan backend: a tiny random Llama-like model, a prompt and 8 greedy tokens on the CPU and on Vulkan.
//!
//! The tests need a native Vulkan device and are skipped (they print why and pass) without one: `FLLM_ALLOW_SOFTWARE_GPU=1` admits a software
//! driver (Mesa lavapipe, `scripts/vulkan-soft-test.sh model`), a GitHub runner has none. They switch the process-wide exact-math flag, so run
//! them with `--test-threads=1`. Names contain `vulkan_whole_model` (the filter of the script).

use crate::backend::vulkan::backend::set_exact_math;
use crate::import_tests::{build_pair, convert, scratch, Kind};
use crate::rust_decoder::RustDecoder;
use crate::GgufModel;
use fllm_checkpoint::OutputDType;
use std::path::Path;

const PROMPT: [i64; 5] = [1, 17, 4, 33, 9];
const NEW_TOKENS: usize = 8;

/// Greedy generation: one prefill of the prompt, then one row per step. Returns the generated ids and the logits of the first step.
fn generate(gguf: &Path, device: &str) -> Result<(Vec<i64>, Vec<f32>), String> {
    let model = GgufModel::load(gguf.to_str().unwrap(), 64, 1, 2)?;
    let decoder = RustDecoder::load(&model, device)?;
    let mut key = vec![0u16; model.key_buffer_len().map_err(|e| e.to_string())?];
    let mut value = vec![0u16; model.value_buffer_len().map_err(|e| e.to_string())?];
    let stride = model.table_stride();
    let table: Vec<i64> = (0..stride as i64).collect();
    let argmax = |logits: &[f32]| logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as i64).unwrap();

    let tables: Vec<i64> = PROMPT.iter().flat_map(|_| table.iter().copied()).collect();
    let lengths: Vec<i64> = (1..=PROMPT.len() as i64).collect();
    let mut logits = decoder.forward(&PROMPT, &lengths, &tables, &mut key, &mut value, &[PROMPT.len() - 1], false)?;
    let first = logits.clone();
    let mut generated = Vec::new();
    let mut length = PROMPT.len() as i64;
    for _ in 0..NEW_TOKENS {
        let next = argmax(&logits);
        generated.push(next);
        length += 1;
        logits = decoder.forward(&[next], &[length], &table, &mut key, &mut value, &[0], false)?;
    }
    Ok((generated, first))
}

/// `None` (and a message) where no native Vulkan device can be opened.
fn device_available() -> bool {
    match crate::backend::vulkan::backend::VulkanBackend::new(0) {
        Ok(backend) => {
            eprintln!("[vulkan_whole_model] device: {}", crate::backend::Backend::name(&backend));
            true
        }
        Err(error) => {
            eprintln!("[vulkan_whole_model] skipped: no native Vulkan device ({error})");
            false
        }
    }
}

fn compare(kind: Option<Kind>, exact_tokens: bool, logit_tolerance: f32) {
    if !device_available() {
        return;
    }
    let (quant_dir, ref_dir) = (scratch("vk-quant"), scratch("vk-ref"));
    // the float model: the F32 checkpoint; the quantized one: the same weights in the format `kind` (converted by the importer)
    build_pair(kind.unwrap_or(Kind::CtInt8), 7, &quant_dir, &ref_dir);
    let gguf = if kind.is_some() { convert(&quant_dir, OutputDType::Preserve) } else { convert(&ref_dir, OutputDType::F32) };
    set_exact_math(true);
    let (want, want_first) = generate(&gguf, "cpu").expect("CPU run");
    let (have, have_first) = generate(&gguf, "vulkan").expect("Vulkan run");
    set_exact_math(false);
    let scale = want_first.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    let worst = want_first.iter().zip(&have_first).fold(0.0f32, |m, (a, b)| m.max((a - b).abs())) / scale;
    eprintln!("[vulkan_whole_model] first-step logits: worst difference {worst:.3e} of the largest logit {scale:.3}; CPU {want:?} Vulkan {have:?}");
    assert!(worst <= logit_tolerance, "first-step logits differ by {worst:e} (tolerance {logit_tolerance:e}), worst element relative to {scale}");
    if exact_tokens {
        assert_eq!(have, want, "greedy tokens differ (CPU left, Vulkan right)");
    }
    let _ = std::fs::remove_dir_all(&quant_dir);
    let _ = std::fs::remove_dir_all(&ref_dir);
}

#[test]
fn vulkan_whole_model_float_greedy_tokens_match_the_cpu() {
    compare(None, true, 2e-4);
}

/// Q8_0 weights: the CPU multiplies int8 weights with int8-quantized activations, the exact Vulkan shaders with f32 activations, so only the
/// logits are compared (to the error of the 8-bit activations), and the tokens are reported.
#[test]
fn vulkan_whole_model_q8_0_logits_match_the_cpu() {
    compare(Some(Kind::CtInt8), false, 3e-2);
}
