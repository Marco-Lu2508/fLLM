//! A chat template comes from a model repository; rendering it with arbitrary messages must end (fuel limit)
//! and must not panic. Input layout: the first line is the template, the rest is a JSON message list.
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else { return };
    let (template, messages) = text.split_once('\n').unwrap_or((text, "[]"));
    let messages = serde_json::from_str(messages).unwrap_or_else(|_| serde_json::json!([{"role": "user", "content": "hi"}]));
    let _ = fllm_engine::server::render_template_for_fuzz(template, &messages);
});
