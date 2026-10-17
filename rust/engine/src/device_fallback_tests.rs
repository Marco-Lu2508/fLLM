//! Device selection on a machine without a (usable) GPU: what GitHub's runners are (Windows: WARP and SwiftShader only). Both tests adapt to
//! the machine they run on, so they also pass on a developer machine with a GPU.

use crate::import_tests::{build_pair, convert, scratch, Kind};
use crate::rust_decoder::RustDecoder;
use crate::GgufModel;
use fllm_checkpoint::OutputDType;

fn tiny_model() -> std::path::PathBuf {
    let (quant_dir, ref_dir) = (scratch("fallback-quant"), scratch("fallback-ref"));
    build_pair(Kind::CtInt8, 3, &quant_dir, &ref_dir);
    convert(&ref_dir, OutputDType::F32)
}

fn run_one_token(decoder: &RustDecoder, model: &GgufModel) -> Vec<f32> {
    let mut key = vec![0u16; model.key_buffer_len().unwrap()];
    let mut value = vec![0u16; model.value_buffer_len().unwrap()];
    let table: Vec<i64> = (0..model.table_stride() as i64).collect();
    decoder.forward(&[1], &[1], &table, &mut key, &mut value, &[0], false).unwrap()
}

#[test]
fn auto_falls_back_to_the_cpu_without_a_hardware_gpu() {
    let gguf = tiny_model();
    let model = GgufModel::load(gguf.to_str().unwrap(), 64, 1, 2).unwrap();
    // `auto` must always end on a working device: an accelerator that exists and can take the model, else the CPU
    let decoder = RustDecoder::load(&model, "auto").expect("--device auto must not fail");
    let logits = run_one_token(&decoder, &model);
    assert!(!logits.is_empty() && logits.iter().all(|v| v.is_finite()));
}

/// An explicit `dx12` without a hardware DirectX 12 adapter (every non-Windows machine, a Windows runner with WARP only) is an error that
/// says what is missing; it is neither a crash nor a silent software run.
#[cfg(feature = "rust-gpu")]
#[test]
fn explicit_dx12_without_hardware_is_a_clear_error() {
    if crate::gpu::list_adapters(0).iter().any(|d| d.description.starts_with("Dx12")) {
        eprintln!("skipped: this machine has a hardware DirectX 12 adapter");
        return;
    }
    let gguf = tiny_model();
    let model = GgufModel::load(gguf.to_str().unwrap(), 64, 1, 2).unwrap();
    let error = match RustDecoder::load(&model, "dx12") {
        Ok(_) => panic!("--device dx12 loaded the model without a hardware adapter"),
        Err(error) => error,
    };
    eprintln!("dx12 without hardware: {error}");
    assert!(error.to_lowercase().contains("dx12"), "the message does not name the API: {error}");
    assert!(
        error.contains("no ") || error.contains("not "),
        "the message does not say that nothing was found: {error}"
    );
}
