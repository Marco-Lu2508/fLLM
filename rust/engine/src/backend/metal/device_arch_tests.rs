//! The element-wise kernels of the arch ops on the Metal device, against formulas in double: activations (all five modes, plain and gated),
//! clamping, the softplus attention gate and the data movement of expanded multi-head latent attention. They are cheap and run before any whole
//! model does, so a broken kernel is named here and not only as a wrong logit. Skipped when the machine has no Metal device.
//!
//! ```text
//! cargo test --release -p fllm-engine --lib --features metal device_arch_ops -- --nocapture
//! ```

use super::backend::MetalBackend;
use super::device_info::all_devices;
use crate::backend::{Backend, DeviceBuffer, MlaShape, Stream};
use std::sync::Arc;

struct Rig {
    backend: Arc<dyn Backend>,
    stream: Box<dyn Stream>,
}

impl Rig {
    fn new() -> Option<Self> {
        if all_devices().is_empty() {
            return None;
        }
        let backend: Arc<dyn Backend> =
            Arc::new(MetalBackend::new(0).unwrap_or_else(|e| panic!("a Metal device exists but the backend did not start: {e}")));
        let stream = backend.create_stream().expect("stream");
        Some(Self { backend, stream })
    }

    fn upload(&mut self, data: &[f32]) -> DeviceBuffer {
        let mut buffer = self.backend.allocate_buffer(data.len() * 4).expect("buffer");
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.stream.upload(&mut buffer, &bytes).expect("upload");
        buffer
    }

    fn zeros(&mut self, n: usize) -> DeviceBuffer {
        self.backend.allocate_buffer(n * 4).expect("buffer")
    }

    fn download(&mut self, buffer: &DeviceBuffer) -> Vec<f32> {
        self.stream.synchronize().expect("synchronize");
        let mut bytes = vec![0u8; buffer.size];
        self.stream.download(&mut bytes, buffer).expect("download");
        bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
    }
}

fn values(n: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            amp * (((state >> 40) as f32 / (1u64 << 23) as f32) - 1.0)
        })
        .collect()
}

/// GELU with the exact erf: the Chebyshev fit of erfc of Numerical Recipes in double, like the host path of the engine.
fn gelu_erf(x: f64) -> f64 {
    let z = x * std::f64::consts::FRAC_1_SQRT_2;
    let t = 1.0 / (1.0 + 0.5 * z.abs());
    let tau = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98 + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    let erf = if z >= 0.0 { 1.0 - tau } else { tau - 1.0 };
    0.5 * x * (1.0 + erf)
}

fn activation(mode: i32, x: f64) -> f64 {
    match mode {
        0 => x / (1.0 + (-x).exp()),
        1 => 0.5 * x * (1.0 + (0.797_884_560_802_865_4 * (x + 0.044_715 * x * x * x)).tanh()),
        2 => gelu_erf(x),
        3 => x.max(0.0),
        _ => x.max(0.0) * x.max(0.0),
    }
}

/// Error against `want`: absolute below 1, relative above.
fn assert_close(label: &str, got: &[f32], want: &[f64], tolerance: f64) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let mut worst = (0.0f64, 0usize);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (f64::from(*g) - w).abs() / w.abs().max(1.0);
        if err > worst.0 || err.is_nan() {
            worst = (err, i);
        }
    }
    assert!(
        worst.0 <= tolerance,
        "metal {label}: largest error {:.3e} (tolerance {tolerance:.0e}) at element {}: device {} reference {}",
        worst.0,
        worst.1,
        got[worst.1],
        want[worst.1]
    );
    eprintln!("metal {label}: largest error {:.3e}", worst.0);
}

#[test]
fn device_arch_ops_activations() {
    let Some(mut rig) = Rig::new() else {
        eprintln!("metal device_arch_ops_activations: no Metal device; skipped");
        return;
    };
    let n = 3000;
    let gate = values(n, 5, 8.0);
    let up = values(n, 6, 2.0);
    for mode in 0..5 {
        let (g, u) = (rig.upload(&gate), rig.upload(&up));
        let mut out = rig.zeros(n);
        rig.stream.activation(&mut out, &g, mode).expect("activation");
        let want: Vec<f64> = gate.iter().map(|&x| activation(mode, f64::from(x))).collect();
        let got = rig.download(&out);
        assert_close(&format!("activation mode {mode}"), &got, &want, 3e-6);

        let mut out = rig.zeros(n);
        rig.stream.activation_glu(&mut out, &g, &u, mode).expect("activation_glu");
        let want: Vec<f64> = gate.iter().zip(&up).map(|(&x, &y)| activation(mode, f64::from(x)) * f64::from(y)).collect();
        let got = rig.download(&out);
        assert_close(&format!("activation_glu mode {mode}"), &got, &want, 3e-6);
    }
}

#[test]
fn device_arch_ops_clamp_and_softplus_gate() {
    let Some(mut rig) = Rig::new() else {
        eprintln!("metal device_arch_ops_clamp_and_softplus_gate: no Metal device; skipped");
        return;
    };
    let n = 5000;
    let x = values(n, 7, 10.0);
    let mut buffer = rig.upload(&x);
    rig.stream.clamp(&mut buffer, 3.5).expect("clamp");
    let want: Vec<f64> = x.iter().map(|&v| f64::from(v).clamp(-3.5, 3.5)).collect();
    let got = rig.download(&buffer);
    assert_close("clamp", &got, &want, 1e-7);

    let gate = values(n, 8, 30.0);
    for beta in [1.0f32, 2.0] {
        let mut buffer = rig.upload(&x);
        let g = rig.upload(&gate);
        rig.stream.softplus_gate_mul(&mut buffer, &g, beta).expect("softplus_gate_mul");
        let want: Vec<f64> = x
            .iter()
            .zip(&gate)
            .map(|(&v, &s)| f64::from(v) * (f64::from(beta) * f64::from(s)).exp().ln_1p() / f64::from(beta))
            .collect();
        let got = rig.download(&buffer);
        assert_close(&format!("softplus_gate_mul beta {beta}"), &got, &want, 2e-6);
    }
}

#[test]
fn device_arch_ops_mla_data_movement() {
    let Some(mut rig) = Rig::new() else {
        eprintln!("metal device_arch_ops_mla_data_movement: no Metal device; skipped");
        return;
    };
    let rows = 5usize;
    let shape = MlaShape { heads: 3, nope: 16, rope: 8, value: 12, kv_lora: 24 };
    let (dk, width) = (shape.nope + shape.rope, shape.kv_lora + shape.rope);
    let iota = |n: usize, base: f32| (0..n).map(|i| base + i as f32).collect::<Vec<f32>>();
    let kv_a = iota(rows * width, 0.0);
    let q_full = iota(rows * shape.heads * dk, 1000.0);
    let kv = iota(rows * shape.heads * (shape.nope + shape.value), 5000.0);
    let (kv_a_b, q_full_b, kv_b) = (rig.upload(&kv_a), rig.upload(&q_full), rig.upload(&kv));

    let mut latent = rig.zeros(rows * shape.kv_lora);
    rig.stream.mla_latent(&mut latent, &kv_a_b, rows, shape.kv_lora, width).expect("mla_latent");
    let want: Vec<f64> = (0..rows)
        .flat_map(|r| (0..shape.kv_lora).map(move |c| (r, c)))
        .map(|(r, c)| f64::from(kv_a[r * width + c]))
        .collect();
    let got = rig.download(&latent);
    assert_close("mla_latent", &got, &want, 0.0);

    let mut q = rig.zeros(rows * shape.heads * dk);
    let mut k = rig.zeros(rows * shape.heads * dk);
    let mut v = rig.zeros(rows * shape.heads * shape.value);
    rig.stream.mla_assemble(&mut q, &mut k, &mut v, &q_full_b, &kv_a_b, &kv_b, rows, &shape).expect("mla_assemble");
    let (mut wq, mut wk, mut wv) = (Vec::new(), Vec::new(), Vec::new());
    for r in 0..rows {
        for h in 0..shape.heads {
            let at = r * shape.heads + h;
            for d in 0..shape.rope {
                wq.push(f64::from(q_full[at * dk + shape.nope + d]));
                wk.push(f64::from(kv_a[r * width + shape.kv_lora + d]));
            }
            for d in 0..shape.nope {
                wq.push(f64::from(q_full[at * dk + d]));
                wk.push(f64::from(kv[at * (shape.nope + shape.value) + d]));
            }
            for e in 0..shape.value {
                wv.push(f64::from(kv[at * (shape.nope + shape.value) + shape.nope + e]));
            }
        }
    }
    let (gq, gk, gv) = (rig.download(&q), rig.download(&k), rig.download(&v));
    assert_close("mla_assemble q", &gq, &wq, 0.0);
    assert_close("mla_assemble k", &gk, &wk, 0.0);
    assert_close("mla_assemble v", &gv, &wv, 0.0);
}
