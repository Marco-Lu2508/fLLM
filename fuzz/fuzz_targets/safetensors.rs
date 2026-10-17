//! The safetensors header and offset checks must reject malformed files with an error.
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let path = std::env::temp_dir().join(format!("fllm-fuzz-{}.safetensors", std::process::id()));
    if std::fs::write(&path, data).is_err() {
        return;
    }
    let _ = fllm_checkpoint::inspect_checkpoint(&path);
});
