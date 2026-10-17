//! The GGUF parser must reject any byte string without panicking, hanging or allocating without bound.
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let path = std::env::temp_dir().join(format!("fllm-fuzz-{}.gguf", std::process::id()));
    if std::fs::write(&path, data).is_err() {
        return;
    }
    if let Ok(model) = fllm_gguf_probe::MappedGguf::open_single(&path) {
        // a file that parses must also answer questions about its tensors without panicking
        let index = model.index();
        for tensor in index.tensors.iter().take(8) {
            let _ = model.tensor_bytes(&tensor.name);
        }
    }
});
