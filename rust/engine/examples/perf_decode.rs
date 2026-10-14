//! Direct decode benchmark without the HTTP server:
//! `perf_decode MODEL.gguf [threads] [tokens] [kv: f16|q8] [ctx] [max-batched-tokens]`
//! Reports prefill/decode; optional seventh argument selects native|rust.
use fllm_engine::{runtime::InferenceRuntime, tokenizer::TokenizerBundle};
use std::{env, time::Instant};

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args
        .get(1)
        .expect("usage: perf_decode MODEL.gguf [threads] [tokens] [f16|q8]");
    let threads: i32 = args.get(2).map_or(10, |v| v.parse().unwrap());
    let max_tokens: usize = args.get(3).map_or(64, |v| v.parse().unwrap());
    let q8 = args.get(4).is_none_or(|v| v == "q8");
    let ctx: usize = args.get(5).map_or(4096, |v| v.parse().unwrap());
    let batched: usize = args.get(6).map_or(512, |v| v.parse().unwrap());
    let compute = args.get(7).map_or("native", String::as_str);
    assert!(matches!(compute, "native" | "rust"));
    let mut model = fllm_engine::GgufModel::load(path, ctx, 1, threads).unwrap();
    if q8 {
        model.set_kv_kind(2);
    }
    let tok = TokenizerBundle::from_gguf(model.mapping()).unwrap();
    let eos: Vec<i64> = tok.eos_ids().iter().map(|x| *x as i64).collect();
    let prompt: Vec<i64> = tok
        .encode("Write a long and detailed story about a dragon who learns to read.")
        .unwrap()
        .into_iter()
        .map(|x| x as i64)
        .collect();
    let mut runtime = InferenceRuntime::new(model, 1, batched, eos).unwrap();
    if compute == "rust" {
        runtime.attach_compute("cpu").unwrap();
    }
    runtime
        .submit("bench".into(), prompt.clone(), max_tokens, 0.0)
        .unwrap();
    let start = Instant::now();
    let mut first: Option<Instant> = None;
    let mut generated = 0usize;
    while runtime.active_requests() > 0 {
        for event in runtime.step().unwrap() {
            let _ = event;
            generated += 1;
            if first.is_none() {
                first = Some(Instant::now());
            }
        }
    }
    let end = Instant::now();
    let st = runtime.batch_stats();
    println!(
        "stats: forward_calls={} rows_total={} max_rows={} prefill_rows={} decode_rows={} scheduler={:.1}ms build={:.1}ms model_forward={:.1}ms postprocess={:.1}ms",
        st.forward_calls, st.rows_total, st.max_rows, st.prefill_rows, st.decode_rows,
        st.scheduler_ns as f64 / 1e6, st.batch_build_ns as f64 / 1e6, st.model_forward_ns as f64 / 1e6, st.postprocess_ns as f64 / 1e6
    );
    let first = first.unwrap_or(end);
    let prefill = (first - start).as_secs_f64();
    let decode = (end - first).as_secs_f64();
    println!(
        "prompt {} tok: prefill {:.3}s ({:.1} tok/s); generated {} tok: decode {:.3}s ({:.2} tok/s)",
        prompt.len(),
        prefill,
        prompt.len() as f64 / prefill,
        generated,
        decode,
        (generated.saturating_sub(1)) as f64 / decode
    );
}
