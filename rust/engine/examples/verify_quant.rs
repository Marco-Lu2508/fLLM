use fllm_engine::compute::{Linear, Tensor};
use fllm_gguf_probe::MappedGguf;
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mapping = MappedGguf::open(args.first().ok_or("GGUF fixture is required")?)?;
    let device = args.get(1).map_or("cpu", String::as_str);
    #[cfg(feature = "rust-gpu")]
    let gpu = if device != "cpu" {
        Some(fllm_engine::gpu::GpuContext::new(device)?)
    } else {
        None
    };
    #[cfg(not(feature = "rust-gpu"))]
    if device != "cpu" {
        return Err("build with rust-gpu".into());
    }
    let mut results = Vec::new();
    for info in &mapping.index().tensors {
        let tensor = Tensor::load(&mapping, &info.name)?;
        if tensor.shape.len() != 2 || info.name == "token_embd.weight" {
            continue; // norms and embeddings are not projections
        }
        let cols = tensor.shape[0];
        let rows = tensor.shape[1];
        #[allow(unused_mut)]
        let mut linear = Linear::load(
            &mapping,
            info.name
                .strip_suffix(".weight")
                .ok_or("invalid fixture name")?,
            rows,
            cols,
        )?;
        // "fast" exercises the packed CPU kernels used by the decoder (8-bit activations)
        if args.get(2).is_some_and(|a| a == "fast") {
            linear.enable_fast_cpu();
        }
        #[cfg(feature = "rust-gpu")]
        if let Some(gpu) = &gpu {
            linear.enable_gpu(gpu)?;
        }
        let input = (0..2 * cols)
            .map(|j| ((j % 97) as f32 - 48.0) / 49.0)
            .collect::<Vec<_>>();
        results.push(
            json!({"name": info.name, "weights": tensor.data, "output": linear.forward(&input)?}),
        );
    }
    println!("{}", json!({"device": device,"results": results}));
    Ok(())
}
