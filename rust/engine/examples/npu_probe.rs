//! Runs the NPU probes of package 11 on the machine at hand (test-round helper; level L0 until someone runs it on the hardware).
//!
//! ```text
//! cargo run --release -p fllm-engine --example npu_probe --features npu-xdna -- xdna   [M K N]
//! cargo run --release -p fllm-engine --example npu_probe --features npu-qnn  -- hexagon [M K N]
//! cargo run --release -p fllm-engine --example npu_probe --features npu-ane  -- ane    [M K N]
//! cargo run --release -p fllm-engine --example npu_probe --features npu-xdna,npu-qnn,npu-ane -- list
//! ```
//!
//! `list` prints the devices the three backends find; the probe names run one `y = x . w^T` layer through the vendor stack (ONNX Runtime +
//! VitisAI on AMD XDNA, QNN on the Hexagon HTP, Core ML on the Apple Neural Engine) and print the report. `M K N` default to `1 4096 4096`.
//! Without the matching feature the example only explains how to build it.

fn dims(args: &[String]) -> (usize, usize, usize) {
    let parse =
        |i: usize, default: usize| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default);
    (parse(1, 1), parse(2, 4096), parse(3, 4096))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let which = args.first().map_or("list", String::as_str);
    #[allow(unused_variables)]
    let (m, k, n) = dims(&args);
    match which {
        "list" => list_devices(),
        "xdna" => {
            #[cfg(feature = "npu-xdna")]
            println!(
                "{:#?}",
                fllm_engine::backend::npu::xdna::probe_matmul(m, k, n)
            );
            #[cfg(not(feature = "npu-xdna"))]
            eprintln!("build with --features npu-xdna");
        }
        "hexagon" => {
            #[cfg(feature = "npu-qnn")]
            println!(
                "{:#?}",
                fllm_engine::backend::npu::hexagon::probe_matmul_f16(m, k, n)
            );
            #[cfg(not(feature = "npu-qnn"))]
            eprintln!("build with --features npu-qnn");
        }
        "ane" => {
            #[cfg(all(feature = "npu-ane", target_vendor = "apple"))]
            println!(
                "{:#?}",
                fllm_engine::backend::npu::ane::probe_linear(
                    m,
                    k,
                    n,
                    fllm_engine::backend::npu::ane::ComputeUnits::CpuAndNeuralEngine
                )
            );
            #[cfg(not(all(feature = "npu-ane", target_vendor = "apple")))]
            eprintln!("build with --features npu-ane on an Apple Silicon Mac");
        }
        other => eprintln!("unknown probe '{other}'; use list | xdna | hexagon | ane"),
    }
}

#[allow(unused_mut, unused_variables)]
fn list_devices() {
    let mut found = 0usize;
    #[cfg(any(feature = "npu-xdna", feature = "npu-qnn", feature = "npu-ane"))]
    {
        use fllm_engine::backend::npu::{ane, hexagon, xdna};
        for (provider, devices) in [
            ("xdna", xdna::list(found)),
            ("hexagon", hexagon::list(found)),
            ("ane", ane::list(found)),
        ] {
            for device in devices {
                println!(
                    "[{provider}] #{} {} - {}",
                    device.index, device.name, device.description
                );
                found += 1;
            }
        }
    }
    if found == 0 {
        println!(
            "no NPU found (features compiled: xdna={}, qnn={}, ane={})",
            cfg!(feature = "npu-xdna"),
            cfg!(feature = "npu-qnn"),
            cfg!(feature = "npu-ane")
        );
    }
}
