//! Fixture writer for the MSL emulator (`metal/emu_ops/run_emu.sh`; package p06b). Not a mirror of the kernels any more: the kernels themselves are
//! executed on the CPU by the emulator, and this test only produces what they are compared with. For every operation and a set of shapes it asks
//! the planner of this module for the dispatches (kernel name, grid, threadgroup size, parameter bytes, buffer slots), computes the expected
//! result with the engine's CPU backend (`backend/cpu.rs`: rms_norm, layer_norm, rope, swiglu, geglu, residual_add, embedding_lookup,
//! kv_cache_write, attention_paged) or, where the CPU backend has no such operation, with a short reference below, and writes inputs, plan and
//! expected outputs into `$FLLM_EMU_DIR/<case>/` (format: see `emu_ops/emu_main.cpp`). Without the environment variable the test does nothing, so it
//! is harmless in a normal `cargo test`.

use super::attention::*;
use super::embedding::*;
use super::*;
use crate::backend::cpu::CpuBackend;
use crate::backend::{
    AttentionParams, Backend, DeviceBuffer, DeviceKvCache, QuantType, RopeParams, Stream,
};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn floats(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.unit() * scale).collect()
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.next() >> 33) as u8).collect()
    }
    fn permutation(&mut self, n: usize) -> Vec<i64> {
        let mut v: Vec<i64> = (0..n as i64).collect();
        for i in (1..n).rev() {
            v.swap(i, self.below(i + 1));
        }
        v
    }
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn i32s(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn from_f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

struct Cpu {
    be: CpuBackend,
    st: Box<dyn Stream>,
}

impl Cpu {
    fn new() -> Self {
        let be = CpuBackend::new();
        let st = be.create_stream().unwrap();
        Self { be, st }
    }
    fn buf(&mut self, bytes: &[u8]) -> DeviceBuffer {
        let mut b = self.be.allocate_buffer(bytes.len()).unwrap();
        self.st.upload(&mut b, bytes).unwrap();
        b
    }
    fn zeros(&mut self, bytes: usize) -> DeviceBuffer {
        self.buf(&vec![0u8; bytes])
    }
    fn f32buf(&mut self, v: &[f32]) -> DeviceBuffer {
        self.buf(&f32s(v))
    }
    fn read(&mut self, b: &DeviceBuffer) -> Vec<u8> {
        let mut out = vec![0u8; b.size];
        self.st.download(&mut out, b).unwrap();
        out
    }
}

pub(super) enum Init {
    Bytes(Vec<u8>),
    Zero(usize),
}

/// One expected result: (buffer, expected bytes, dtype `f32` / `i32` / `u8`, absolute tolerance, relative tolerance).
pub(super) type Expect<'a> = (&'a str, Vec<u8>, &'a str, f64, f64);

/// Consumer of the cases: the fixture writer of the CPU emulator, or (Mac only, `gpu_check.rs`) a runner on a real Metal device.
pub(super) trait Sink {
    /// One case: the dispatches `plans` run in order over the named buffers (see [`gather_buffers`]). `inputs` supplies the buffers of the
    /// `SlotSource::Buffer` slots by slot name; `expects` are the results to compare.
    fn emit(
        &mut self,
        name: &str,
        plans: &[OpPlan],
        inputs: &[(&str, Vec<u8>)],
        expects: &[Expect],
    );
}

/// Name of the buffer a slot is bound to: slots of the plans of one case with the same name share a buffer.
pub(super) fn buffer_name(slot: &BufferSlot) -> String {
    match &slot.source {
        SlotSource::Buffer | SlotSource::Host(_) => slot.name.to_string(),
        SlotSource::Scratch { id, .. } => format!("scratch{id}"),
        _ => "dummy".to_string(),
    }
}

/// The named buffers of a case with their initial contents: inputs for `Buffer` slots, the host arrays of the plans, zeroed scratch (the largest
/// size any plan asks for) and a 64-byte dummy. Also runs the source conformance check of every plan.
pub(super) fn gather_buffers(
    name: &str,
    plans: &[OpPlan],
    inputs: &[(&str, Vec<u8>)],
) -> BTreeMap<String, Init> {
    let mut bufs: BTreeMap<String, Init> = BTreeMap::new();
    let mut used: Vec<&str> = Vec::new();
    for plan in plans {
        super::tests::check_plan(plan);
        for slot in &plan.slots {
            match &slot.source {
                SlotSource::Buffer => {
                    let (_, data) = inputs
                        .iter()
                        .find(|(n, _)| *n == slot.name)
                        .unwrap_or_else(|| panic!("{name}: no input for slot {}", slot.name));
                    used.push(slot.name);
                    bufs.entry(slot.name.to_string())
                        .or_insert_with(|| Init::Bytes(data.clone()));
                }
                SlotSource::Host(b) => match bufs
                    .entry(slot.name.to_string())
                    .or_insert_with(|| Init::Bytes(b.to_vec()))
                {
                    Init::Bytes(old) => assert_eq!(
                        old,
                        b.as_ref(),
                        "{name}: host array {} differs between plans",
                        slot.name
                    ),
                    Init::Zero(_) => {
                        panic!("{name}: slot {} is both host array and scratch", slot.name)
                    }
                },
                SlotSource::Scratch { id, bytes } => {
                    let e = bufs.entry(format!("scratch{id}")).or_insert(Init::Zero(0));
                    if let Init::Zero(n) = e {
                        *n = (*n).max(*bytes);
                    }
                }
                SlotSource::Dummy => {
                    bufs.entry("dummy".to_string()).or_insert(Init::Zero(64));
                }
                SlotSource::RunState(_) => {
                    panic!("{name}: run state slots are not part of p06b")
                }
            }
        }
    }
    for (n, _) in inputs {
        assert!(
            used.contains(n),
            "{name}: input {n} is not used by any plan"
        );
    }
    bufs
}

thread_local! {
    /// Capabilities the attention cases plan for (the defaults on the host; the hardware check sets the values of the device).
    static TUNING: std::cell::Cell<AttentionTuning> = std::cell::Cell::new(AttentionTuning::default());
}

/// Set the attention capabilities of this thread (`simdgroup_matrix`, SIMD width, thread limit).
#[allow(dead_code)] // used by gpu_check.rs, which is switched on after the Metal core (p05) is merged
pub(super) fn set_device_tuning(t: AttentionTuning) {
    TUNING.with(|c| c.set(t));
}

fn device_tuning() -> AttentionTuning {
    TUNING.with(|c| c.get())
}

struct Fixtures {
    root: PathBuf,
    names: Vec<String>,
}

impl Fixtures {
    fn from_env() -> Option<Self> {
        let root = PathBuf::from(std::env::var_os("FLLM_EMU_DIR")?);
        std::fs::create_dir_all(&root).unwrap();
        Some(Self {
            root,
            names: Vec::new(),
        })
    }

    fn finish(&self) {
        std::fs::write(self.root.join("cases.txt"), self.names.join("\n") + "\n").unwrap();
        eprintln!(
            "emu_fixtures: {} cases in {}",
            self.names.len(),
            self.root.display()
        );
    }
}

impl Sink for Fixtures {
    fn emit(
        &mut self,
        name: &str,
        plans: &[OpPlan],
        inputs: &[(&str, Vec<u8>)],
        expects: &[Expect],
    ) {
        let dir = self.root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let bufs = gather_buffers(name, plans, inputs);
        let mut text = String::new();
        for (bname, init) in &bufs {
            match init {
                Init::Bytes(b) => {
                    std::fs::write(dir.join(format!("{bname}.bin")), b).unwrap();
                    writeln!(text, "buf {bname} {bname}.bin").unwrap();
                }
                Init::Zero(n) => writeln!(text, "buf {bname} zero:{n}").unwrap(),
            }
        }
        for plan in plans {
            let hex: String = if plan.constants.is_empty() {
                "-".into()
            } else {
                plan.constants.iter().map(|b| format!("{b:02x}")).collect()
            };
            let names: Vec<String> = plan.slots.iter().map(buffer_name).collect();
            let (g, t) = (plan.threadgroups, plan.threads_per_group);
            writeln!(
                text,
                "run {} {} {} {} {} {} {} {} {}",
                plan.kernel_name,
                g[0],
                g[1],
                g[2],
                t[0],
                t[1],
                t[2],
                hex,
                names.join(" ")
            )
            .unwrap();
        }
        for (bname, want, dtype, abs, rel) in expects {
            std::fs::write(dir.join(format!("{bname}.exp")), want).unwrap();
            writeln!(text, "expect {bname} {bname}.exp {dtype} {abs} {rel}").unwrap();
        }
        std::fs::write(dir.join("case.txt"), text).unwrap();
        self.names.push(name.to_string());
    }
}
// ---------------------------------------------------------------------------------------------------------------------------------------------
// Norms, RoPE, activations, elementwise
// ---------------------------------------------------------------------------------------------------------------------------------------------

fn norm_cases(fx: &mut dyn Sink) {
    let mut rng = Rng::new(11);
    let mut cpu = Cpu::new();
    for (rows, cols) in [(3usize, 64usize), (2, 100), (1, 4096), (5, 1024), (2, 33)] {
        for has_weight in [true, false] {
            let x = rng.floats(rows * cols, 2.0);
            let w = rng.floats(cols, 1.0);
            let ones = vec![1.0f32; cols];
            let (xb, wb, mut ob) = (
                cpu.f32buf(&x),
                cpu.f32buf(if has_weight { &w } else { &ones }),
                cpu.zeros(rows * cols * 4),
            );
            cpu.st.rms_norm(&mut ob, &xb, &wb, 1e-6).unwrap();
            let want = cpu.read(&ob);
            let plan = plan_rms_norm(rows, cols, 1e-6, has_weight).unwrap();
            let mut inputs = vec![("x", f32s(&x)), ("out", vec![0u8; rows * cols * 4])];
            if has_weight {
                inputs.push(("weight", f32s(&w)));
            }
            fx.emit(
                &format!("rms_{rows}x{cols}_w{}", u8::from(has_weight)),
                &[plan],
                &inputs,
                &[("out", want, "f32", 2e-5, 2e-5)],
            );
        }
    }
    // a slice of wider rows (offset and stride in floats), written into a narrower output
    {
        let (rows, cols) = (3usize, 64usize);
        let layout = RowLayout {
            x_stride: 300,
            x_off: 8,
            o_stride: 80,
            o_off: 4,
        };
        let x = rng.floats(layout.x_off + rows * layout.x_stride, 2.0);
        let w = rng.floats(cols, 1.0);
        let mut gathered = Vec::new();
        for r in 0..rows {
            gathered.extend_from_slice(&x[layout.x_off + r * layout.x_stride..][..cols]);
        }
        let (xb, wb, mut ob) = (
            cpu.f32buf(&gathered),
            cpu.f32buf(&w),
            cpu.zeros(rows * cols * 4),
        );
        cpu.st.rms_norm(&mut ob, &xb, &wb, 1e-5).unwrap();
        let normed = from_f32s(&cpu.read(&ob));
        let out_len = layout.o_off + rows * layout.o_stride;
        let mut want = vec![0.0f32; out_len];
        for r in 0..rows {
            want[layout.o_off + r * layout.o_stride..][..cols]
                .copy_from_slice(&normed[r * cols..][..cols]);
        }
        let plan = plan_rms_norm_strided(rows, cols, 1e-5, true, layout).unwrap();
        fx.emit(
            "rms_strided",
            &[plan],
            &[
                ("x", f32s(&x)),
                ("weight", f32s(&w)),
                ("out", vec![0u8; out_len * 4]),
            ],
            &[("out", f32s(&want), "f32", 2e-5, 2e-5)],
        );
    }
    for (rows, cols) in [(3usize, 64usize), (2, 100), (4, 2048), (2, 33)] {
        for has_bias in [true, false] {
            let x = rng.floats(rows * cols, 3.0);
            let w = rng.floats(cols, 1.0);
            let b = rng.floats(cols, 0.5);
            let (xb, wb, bb, mut ob) = (
                cpu.f32buf(&x),
                cpu.f32buf(&w),
                cpu.f32buf(&b),
                cpu.zeros(rows * cols * 4),
            );
            cpu.st
                .layer_norm(&mut ob, &xb, &wb, has_bias.then_some(&bb), 1e-5)
                .unwrap();
            let want = cpu.read(&ob);
            let plan = plan_layer_norm(rows, cols, 1e-5, has_bias).unwrap();
            let mut inputs = vec![
                ("x", f32s(&x)),
                ("weight", f32s(&w)),
                ("out", vec![0u8; rows * cols * 4]),
            ];
            if has_bias {
                inputs.push(("bias", f32s(&b)));
            }
            fx.emit(
                &format!("layernorm_{rows}x{cols}_b{}", u8::from(has_bias)),
                &[plan],
                &inputs,
                &[("out", want, "f32", 3e-5, 3e-5)],
            );
        }
    }
}

fn rope_cases(fx: &mut dyn Sink) {
    let mut rng = Rng::new(21);
    let mut cpu = Cpu::new();
    for (heads, kv_heads, dim, rope_dim, neox) in [
        (6usize, 2usize, 64usize, 64usize, true),
        (4, 4, 64, 32, false),
        (8, 2, 128, 128, true),
        (3, 1, 80, 16, false),
        (2, 2, 96, 96, false),
    ] {
        let positions: Vec<usize> = (0..7).map(|_| rng.below(60)).collect();
        let half = rope_dim / 2;
        let table_rows = 64;
        let mut cos = Vec::new();
        let mut sin = Vec::new();
        for p in 0..table_rows {
            for d in 0..half {
                let a = p as f32 * 0.37 / (1.0 + d as f32);
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
        let rows = positions.len();
        let q = rng.floats(rows * heads * dim, 1.5);
        let k = rng.floats(rows * kv_heads * dim, 1.5);
        let (mut qb, mut kb) = (cpu.f32buf(&q), cpu.f32buf(&k));
        let params = RopeParams {
            heads,
            kv_heads,
            dim,
            rope_dim,
            neox,
            cos: Arc::new(cos.clone()),
            sin: Arc::new(sin.clone()),
        };
        cpu.st.rope(&mut qb, &mut kb, &positions, &params).unwrap();
        let (wq, wk) = (cpu.read(&qb), cpu.read(&kb));
        let plan = plan_rope(&positions, heads, kv_heads, dim, rope_dim, neox, table_rows).unwrap();
        fx.emit(
            &format!(
                "rope_h{heads}k{kv_heads}d{dim}r{rope_dim}n{}",
                u8::from(neox)
            ),
            &[plan],
            &[
                ("q", f32s(&q)),
                ("k", f32s(&k)),
                ("cos", f32s(&cos)),
                ("sin", f32s(&sin)),
            ],
            &[("q", wq, "f32", 1e-5, 1e-5), ("k", wk, "f32", 1e-5, 1e-5)],
        );
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
}

fn activation_cases(fx: &mut dyn Sink) {
    let mut rng = Rng::new(31);
    let mut cpu = Cpu::new();
    // dense, through the CPU backend where it has the operation
    for count in [1000usize, 7, 4096] {
        let g = rng.floats(count, 4.0);
        let u = rng.floats(count, 2.0);
        let (gb, ub) = (cpu.f32buf(&g), cpu.f32buf(&u));
        for (mode, tag) in [(GluMode::Swiglu, "swiglu"), (GluMode::Geglu, "geglu")] {
            let mut ob = cpu.zeros(count * 4);
            if matches!(mode, GluMode::Swiglu) {
                cpu.st.swiglu(&mut ob, &gb, &ub).unwrap();
            } else {
                cpu.st.geglu(&mut ob, &gb, &ub).unwrap();
            }
            let want = cpu.read(&ob);
            let plan = plan_glu(mode, count).unwrap();
            fx.emit(
                &format!("act_{tag}_{count}"),
                &[plan],
                &[
                    ("gate", f32s(&g)),
                    ("up", f32s(&u)),
                    ("out", vec![0u8; count * 4]),
                ],
                &[("out", want, "f32", 2e-5, 2e-5)],
            );
        }
    }
    // all modes with a fused gate/up layout: rows of [gate | up], output narrower than the input rows
    let (n_rows, width) = (5usize, 300usize);
    let fused = rng.floats(n_rows * 2 * width, 3.0);
    let layout = GluLayout {
        gate_stride: 2 * width,
        gate_off: 0,
        up_stride: 2 * width,
        up_off: width,
        out_stride: width,
    };
    let (alpha, limit) = (1.702f32, 7.0f32);
    let modes = [
        (GluMode::Swiglu, "swiglu"),
        (GluMode::Geglu, "geglu"),
        (GluMode::SwigluOai { alpha, limit }, "swiglu_oai"),
        (GluMode::Silu, "silu"),
        (GluMode::Gelu, "gelu"),
        (GluMode::Relu2, "relu2"),
        (GluMode::Relu, "relu"),
    ];
    for (mode, tag) in modes {
        let mut want = vec![0.0f32; n_rows * width];
        for r in 0..n_rows {
            for c in 0..width {
                let g = fused[r * layout.gate_stride + layout.gate_off + c];
                let u = fused[r * layout.up_stride + layout.up_off + c];
                want[r * width + c] = match mode {
                    GluMode::Swiglu => silu(g) * u,
                    GluMode::Geglu => gelu_tanh(g) * u,
                    GluMode::SwigluOai { .. } => {
                        let gc = g.min(limit);
                        let uc = u.clamp(-limit, limit);
                        gc / (1.0 + (-alpha * gc).exp()) * (uc + 1.0)
                    }
                    GluMode::Silu => silu(g),
                    GluMode::Gelu => gelu_tanh(g),
                    GluMode::Relu2 => g.max(0.0) * g.max(0.0),
                    GluMode::Relu => g.max(0.0),
                };
            }
        }
        let plan = plan_glu_strided(mode, n_rows, width, layout).unwrap();
        // gate and up are two views of the same fused buffer
        let mut inputs = vec![
            ("gate", f32s(&fused)),
            ("out", vec![0u8; n_rows * width * 4]),
        ];
        if mode.gated() {
            inputs.push(("up", f32s(&fused)));
        }
        fx.emit(
            &format!("act_fused_{tag}"),
            &[plan],
            &inputs,
            &[("out", f32s(&want), "f32", 3e-5, 3e-5)],
        );
    }
}

fn elementwise_cases(fx: &mut dyn Sink) {
    let mut rng = Rng::new(41);
    let mut cpu = Cpu::new();
    for count in [1usize, 255, 256, 257, 5000] {
        let x = rng.floats(count, 2.0);
        let r = rng.floats(count, 2.0);
        let mut xb = cpu.f32buf(&x);
        let rb = cpu.f32buf(&r);
        cpu.st.residual_add(&mut xb, &rb, 0.75).unwrap();
        let want = cpu.read(&xb);
        let plan = plan_residual_add(count, 0.75).unwrap();
        fx.emit(
            &format!("residual_{count}"),
            &[plan],
            &[("x", f32s(&x)), ("residual", f32s(&r))],
            &[("x", want, "f32", 1e-6, 1e-6)],
        );

        let gate = rng.floats(count, 5.0);
        let want: Vec<f32> = x
            .iter()
            .zip(&gate)
            .map(|(x, g)| x / (1.0 + (-g).exp()))
            .collect();
        let plan = plan_sigmoid_gate_mul(count).unwrap();
        fx.emit(
            &format!("sigmoid_mul_{count}"),
            &[plan],
            &[("x", f32s(&x)), ("gate", f32s(&gate))],
            &[("x", f32s(&want), "f32", 1e-5, 1e-5)],
        );
    }
    for (rows, heads, dk, per_head) in [
        (3usize, 4usize, 64usize, true),
        (3, 4, 64, false),
        (2, 3, 300, true),
        (2, 3, 300, false),
        (1, 2, 17, false),
    ] {
        let projected = rng.floats(rows * 2 * heads * dk, 1.0);
        let width = heads * dk;
        let (mut q, mut gate) = (vec![0.0f32; rows * width], vec![0.0f32; rows * width]);
        for r in 0..rows {
            for h in 0..heads {
                for d in 0..dk {
                    let base = r * 2 * width;
                    let (sq, sg) = if per_head {
                        (base + h * 2 * dk + d, base + h * 2 * dk + dk + d)
                    } else {
                        (base + h * dk + d, base + width + h * dk + d)
                    };
                    q[r * width + h * dk + d] = projected[sq];
                    gate[r * width + h * dk + d] = projected[sg];
                }
            }
        }
        let plan = plan_split_q_gate(rows, heads, dk, per_head).unwrap();
        fx.emit(
            &format!(
                "split_q_gate_{rows}x{heads}x{dk}_{}",
                if per_head { "ph" } else { "flat" }
            ),
            &[plan],
            &[
                ("projected", f32s(&projected)),
                ("q", vec![0u8; rows * width * 4]),
                ("gate", vec![0u8; rows * width * 4]),
            ],
            &[
                ("q", f32s(&q), "f32", 0.0, 0.0),
                ("gate", f32s(&gate), "f32", 0.0, 0.0),
            ],
        );
    }
    // argmax: random rows, a tie (lowest index wins), NaNs that must not win, an all -inf row
    for vocab in [5usize, 100, 1000, 33000] {
        let rows = 4;
        let mut logits = rng.floats(rows * vocab, 10.0);
        let mut want = Vec::new();
        for r in 0..rows {
            let row = &mut logits[r * vocab..(r + 1) * vocab];
            match r {
                1 => {
                    // tie of the maximum at two indices
                    let (a, b) = (vocab / 3, vocab - 1);
                    row[a] = 99.0;
                    row[b] = 99.0;
                }
                2 => {
                    // NaN and -inf entries; the real maximum is further on
                    row[0] = f32::NAN;
                    if vocab > 2 {
                        row[1] = f32::NEG_INFINITY;
                    }
                    let m = vocab / 2;
                    row[m] = 50.0;
                }
                3 => row.iter_mut().for_each(|v| *v = f32::NEG_INFINITY),
                _ => {}
            }
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (i, &v) in row.iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best = i;
                }
            }
            want.push(best as i32);
        }
        let plan = plan_argmax(rows, vocab).unwrap();
        fx.emit(
            &format!("argmax_{vocab}"),
            &[plan],
            &[("logits", f32s(&logits)), ("out", vec![0u8; rows * 4])],
            &[("out", i32s(&want), "i32", 0.0, 0.0)],
        );
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------------------------------------------------------------------------

/// Byte offsets of the binary16 fields of a block (replaced by sane values so that the decoded numbers are finite).
fn f16_offsets(q: QuantType) -> &'static [usize] {
    match q {
        QuantType::Q4_0 | QuantType::Q5_0 | QuantType::Q8_0 | QuantType::IQ4_NL => &[0],
        QuantType::Q4_1 | QuantType::Q5_1 | QuantType::Q4_K | QuantType::Q5_K => &[0, 2],
        QuantType::Q2_K => &[80, 82],
        QuantType::Q3_K => &[108],
        QuantType::Q6_K => &[208],
        _ => &[],
    }
}

fn embedding_cases(fx: &mut dyn Sink) {
    let mut rng = Rng::new(51);
    let mut cpu = Cpu::new();
    let (rows, cols) = (7usize, 512usize);
    let ids: Vec<i64> = vec![3, 0, 6, 3, 5];
    let quants = [
        QuantType::F32,
        QuantType::F16,
        QuantType::BF16,
        QuantType::Q4_0,
        QuantType::Q4_1,
        QuantType::Q5_0,
        QuantType::Q5_1,
        QuantType::Q8_0,
        QuantType::Q2_K,
        QuantType::Q3_K,
        QuantType::Q4_K,
        QuantType::Q5_K,
        QuantType::Q6_K,
        QuantType::IQ4_NL,
        QuantType::Mxfp4,
    ];
    for q in quants {
        let table = EmbeddingTable::gguf(q, rows, cols);
        let (_, qk, bytes) = embed_kernel(q).unwrap();
        let row_bytes = cols / qk * bytes;
        let mut raw = rng.bytes(rows * row_bytes);
        match q {
            QuantType::F32 => raw = f32s(&rng.floats(rows * cols, 1.0)),
            QuantType::F16 => {
                raw = (0..rows * cols)
                    .flat_map(|_| half::f16::from_f32(rng.unit()).to_le_bytes())
                    .collect()
            }
            QuantType::BF16 => {
                raw = (0..rows * cols)
                    .flat_map(|_| half::bf16::from_f32(rng.unit()).to_le_bytes())
                    .collect()
            }
            QuantType::Mxfp4 => {
                for blk in raw.chunks_exact_mut(bytes) {
                    blk[0] = 120 + (blk[0] % 14);
                }
            }
            _ => {
                for blk in raw.chunks_exact_mut(bytes) {
                    for &o in f16_offsets(q) {
                        let v = half::f16::from_f32(rng.unit() * 0.5).to_le_bytes();
                        blk[o] = v[0];
                        blk[o + 1] = v[1];
                    }
                }
            }
        }
        let w = cpu
            .be
            .upload_weight("emb", &raw, q, rows, cols, None)
            .unwrap();
        let mut ob = cpu.zeros(ids.len() * cols * 4);
        cpu.st.embedding_lookup(&mut ob, &ids, &w).unwrap();
        let want = cpu.read(&ob);
        let plan = plan_embedding(&table, &ids).unwrap();
        fx.emit(
            &format!("embed_{}", format!("{q:?}").to_lowercase()),
            &[plan],
            &[("weights", raw), ("out", vec![0u8; ids.len() * cols * 4])],
            &[("out", want, "f32", 1e-6, 1e-6)],
        );
    }
    // MXFP4 in the planar device layout of package p06a: code plane, then the scale plane
    {
        let (_, qk, bytes) = embed_kernel(QuantType::Mxfp4).unwrap();
        let blocks = rows * cols / qk;
        let mut raw = rng.bytes(blocks * bytes);
        for blk in raw.chunks_exact_mut(bytes) {
            blk[0] = 120 + (blk[0] % 14);
        }
        let w = cpu
            .be
            .upload_weight("emb", &raw, QuantType::Mxfp4, rows, cols, None)
            .unwrap();
        let mut ob = cpu.zeros(ids.len() * cols * 4);
        cpu.st.embedding_lookup(&mut ob, &ids, &w).unwrap();
        let want = cpu.read(&ob);
        let mut image: Vec<u8> = raw
            .chunks_exact(bytes)
            .flat_map(|b| b[1..].to_vec())
            .collect();
        assert_eq!(image.len(), blocks * 16);
        image.extend(raw.chunks_exact(bytes).map(|b| b[0]));
        let plan = plan_embedding(&EmbeddingTable::mxfp4_planar(rows, cols), &ids).unwrap();
        fx.emit(
            "embed_mxfp4_planar",
            &[plan],
            &[("weights", image), ("out", vec![0u8; ids.len() * cols * 4])],
            &[("out", want, "f32", 1e-6, 1e-6)],
        );
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// KV write and attention
// ---------------------------------------------------------------------------------------------------------------------------------------------

const LAYERS: usize = 3;
const POOL_BLOCKS: usize = 24;
const BLOCK: usize = 16;
const STRIDE: usize = 20;

fn kv_cache(
    cpu: &mut Cpu,
    kind: i32,
    kv_heads: usize,
    dk: usize,
    dv: usize,
    block: usize,
) -> DeviceKvCache {
    cpu.be
        .allocate_kv_cache(LAYERS, POOL_BLOCKS, block, kv_heads, dk, dv, kind)
        .unwrap()
}

fn kv_write_cases(fx: &mut dyn Sink) {
    let mut rng = Rng::new(61);
    let mut cpu = Cpu::new();
    for (kind, kv_heads, dk, dv, layer) in [
        (1, 2usize, 64usize, 64usize, 1usize),
        (1, 3, 80, 64, 2),
        (2, 2, 64, 64, 1),
        (2, 2, 80, 80, 2),
        (2, 1, 192, 128, 0),
        (2, 3, 48, 40, 1),
        (1, 1, 33, 33, 2),
    ] {
        let kv = kv_cache(&mut cpu, kind, kv_heads, dk, dv, BLOCK);
        // garbage everywhere: neighbours of the written slots must stay as they are
        let (kp0, vp0) = (rng.bytes(kv.key_pool.size), rng.bytes(kv.value_pool.size));
        let (mut kpb, mut vpb) = (kv.key_pool.clone(), kv.value_pool.clone());
        cpu.st.upload(&mut kpb, &kp0).unwrap();
        cpu.st.upload(&mut vpb, &vp0).unwrap();
        let slots: Vec<usize> = rng
            .permutation(kv.slots())
            .into_iter()
            .take(9)
            .map(|s| s as usize)
            .collect();
        let k = rng.floats(slots.len() * kv_heads * dk, 2.0);
        let v = rng.floats(slots.len() * kv_heads * dv, 2.0);
        let (kb, vb) = (cpu.f32buf(&k), cpu.f32buf(&v));
        cpu.st.kv_cache_write(&kv, layer, &slots, &kb, &vb).unwrap();
        let (kp1, vp1) = (cpu.read(&kv.key_pool), cpu.read(&kv.value_pool));
        let layout = KvLayout::from_cache(&kv, layer).unwrap();
        let plans = plan_kv_write(&layout, &slots).unwrap();
        fx.emit(
            &format!(
                "kvwrite_{}_h{kv_heads}d{dk}v{dv}l{layer}",
                if kind == 1 { "f16" } else { "q8" }
            ),
            &plans,
            &[
                ("k", f32s(&k)),
                ("v", f32s(&v)),
                ("key_pool", kp0),
                ("value_pool", vp0),
            ],
            &[
                ("key_pool", kp1, "u8", 0.0, 0.0),
                ("value_pool", vp1, "u8", 0.0, 0.0),
            ],
        );
    }
}

struct AttnCase {
    name: String,
    kind: i32,
    heads: usize,
    kv_heads: usize,
    dk: usize,
    dv: usize,
    layer: usize,
    /// tokens per KV block
    block: usize,
    lengths: Vec<i64>,
    /// rows share block tables in this many runs (0: every row has its own table, 1: one prompt, n: n prompts)
    table_groups: usize,
    window: Option<usize>,
    chunk: Option<usize>,
    softcap: Option<f32>,
    sinks: bool,
    alibi: bool,
    tuning: AttentionTuning,
    seed: u64,
}

impl AttnCase {
    fn new(
        name: &str,
        kind: i32,
        heads: usize,
        kv_heads: usize,
        dk: usize,
        dv: usize,
        lengths: Vec<i64>,
    ) -> Self {
        Self {
            name: name.to_string(),
            kind,
            heads,
            kv_heads,
            dk,
            dv,
            layer: 1,
            block: BLOCK,
            lengths,
            table_groups: 0,
            window: None,
            chunk: None,
            softcap: None,
            sinks: false,
            alibi: false,
            tuning: device_tuning(),
            seed: name
                .bytes()
                .fold(7u64, |h, b| h.wrapping_mul(131).wrapping_add(u64::from(b))),
        }
    }
}

fn attention_case(fx: &mut dyn Sink, c: AttnCase) {
    let mut rng = Rng::new(c.seed);
    let mut cpu = Cpu::new();
    let kv = kv_cache(&mut cpu, c.kind, c.kv_heads, c.dk, c.dv, c.block);
    // random keys / values in every layer, written through the CPU backend (so the q8 encoding is the engine's)
    let all_slots: Vec<usize> = (0..kv.slots()).collect();
    for layer in 0..LAYERS {
        let k = rng.floats(kv.slots() * c.kv_heads * c.dk, 1.0);
        let v = rng.floats(kv.slots() * c.kv_heads * c.dv, 1.0);
        let (kb, vb) = (cpu.f32buf(&k), cpu.f32buf(&v));
        cpu.st
            .kv_cache_write(&kv, layer, &all_slots, &kb, &vb)
            .unwrap();
    }
    let rows = c.lengths.len();
    let q = rng.floats(rows * c.heads * c.dk, 1.5);
    let mut tables = Vec::new();
    let shared: Vec<Vec<i64>> = (0..c.table_groups.max(1))
        .map(|_| rng.permutation(POOL_BLOCKS))
        .collect();
    for r in 0..rows {
        let perm = if c.table_groups > 0 {
            shared[r * c.table_groups / rows].clone()
        } else {
            rng.permutation(POOL_BLOCKS)
        };
        tables.extend_from_slice(&perm[..STRIDE]);
    }
    let params = AttentionParams {
        num_heads: c.heads,
        num_kv_heads: c.kv_heads,
        head_dim: c.dk,
        value_dim: c.dv,
        stride: STRIDE,
        block_size: c.block,
        // the CPU backend quantizes the query for q8 caches on its fast path (exactly 1/sqrt(dk), no modifiers); a different scale makes it
        // take the scalar path, which is the reference for the semantics
        scale: (1.0 / (c.dk as f32).sqrt()) * if c.kind == 2 { 0.97 } else { 1.0 },
        softcap: c.softcap,
        window: c.window,
        chunk: c.chunk,
        sinks: c.sinks.then(|| Arc::new(rng.floats(c.heads, 1.0))),
        alibi_slopes: c
            .alibi
            .then(|| Arc::new((0..c.heads).map(|h| 0.02 + 0.03 * h as f32).collect())),
    };
    let qb = cpu.f32buf(&q);
    let mut ob = cpu.zeros(rows * c.heads * c.dv * 4);
    cpu.st
        .attention_paged(&mut ob, &qb, &kv, &tables, &c.lengths, c.layer, &params)
        .unwrap();
    let want = cpu.read(&ob);
    let layout = KvLayout::from_cache(&kv, c.layer).unwrap();
    let plans = plan_attention(&layout, &params, &tables, &c.lengths, &c.tuning).unwrap();
    let (kp, vp) = (cpu.read(&kv.key_pool), cpu.read(&kv.value_pool));
    fx.emit(
        &c.name,
        &plans,
        &[
            ("q", f32s(&q)),
            ("kpool", kp),
            ("vpool", vp),
            ("out", vec![0u8; rows * c.heads * c.dv * 4]),
        ],
        &[("out", want, "f32", 3e-5, 3e-4)],
    );
}

fn decode_only() -> AttentionTuning {
    AttentionTuning {
        use_prefill_kernel: false,
        ..device_tuning()
    }
}

fn attention_cases(fx: &mut dyn Sink) {
    for (kind, tag) in [(1, "f16"), (2, "q8")] {
        for (dk, dv, heads, kv_heads, len) in [
            (64usize, 64usize, 8usize, 2usize, 300i64),
            (80, 80, 4, 2, 200),
            (96, 96, 4, 4, 130),
            (128, 128, 4, 1, 190),
            (192, 128, 4, 2, 100),
            (32, 32, 2, 2, 50),
            (256, 256, 2, 1, 150),
            (160, 160, 2, 2, 90),
            (64, 128, 4, 2, 100),
            (128, 64, 4, 2, 100),
        ] {
            let mut c = AttnCase::new(
                &format!("decode_{tag}_{dk}_{dv}"),
                kind,
                heads,
                kv_heads,
                dk,
                dv,
                vec![len],
            );
            c.tuning = decode_only();
            attention_case(fx, c);
        }
    }
    {
        let mut c = AttnCase::new("decode_f16_576_512", 1, 2, 1, 576, 512, vec![40]);
        c.tuning = decode_only();
        attention_case(fx, c);
        let mut c = AttnCase::new("decode_q8_576_512", 2, 2, 1, 576, 512, vec![40]);
        c.tuning = decode_only();
        attention_case(fx, c);
    }
    // short context (no split), a continuous batch with different lengths and block tables, and forced fine splitting
    for (name, lengths, tuning) in [
        ("decode_short", vec![20i64], decode_only()),
        ("decode_batch", vec![17, 80, 33, 160, 1], decode_only()),
        (
            "decode_fine_splits",
            vec![200],
            AttentionTuning {
                target_groups: 4096,
                min_split_keys: 8,
                max_splits: 64,
                ..decode_only()
            },
        ),
    ] {
        let mut c = AttnCase::new(name, 1, 4, 2, 64, 64, lengths);
        c.tuning = tuning;
        attention_case(fx, c);
    }
    // masks and modifiers on the decode kernel
    for (name, window, chunk, softcap, sinks, alibi) in [
        ("window", Some(48usize), None, None, false, false),
        ("chunk", None, Some(64usize), None, false, false),
        ("softcap", None, None, Some(8.0f32), false, false),
        ("sinks", None, None, None, true, false),
        ("alibi", None, None, None, false, true),
        ("all", Some(70), None, Some(10.0), true, true),
    ] {
        for kind in [1, 2] {
            let mut c = AttnCase::new(
                &format!("decode_{name}_{}", if kind == 1 { "f16" } else { "q8" }),
                kind,
                4,
                2,
                64,
                64,
                vec![150, 33],
            );
            c.window = window;
            c.chunk = chunk;
            c.softcap = softcap;
            c.sinks = sinks;
            c.alibi = alibi;
            c.tuning = AttentionTuning {
                target_groups: 64,
                min_split_keys: 16,
                ..decode_only()
            };
            attention_case(fx, c);
        }
    }
    // one split (short context, default splitting rules): sink, ALiBi, soft cap and window handled inside the decode kernel itself, not by the merge
    for kind in [1, 2] {
        let tag = if kind == 1 { "f16" } else { "q8" };
        let mut c = AttnCase::new(
            &format!("decode_single_split_sinks_{tag}"),
            kind,
            4,
            2,
            64,
            64,
            vec![60, 20],
        );
        c.sinks = true;
        c.tuning = decode_only();
        attention_case(fx, c);
        let mut c = AttnCase::new(
            &format!("decode_single_split_all_{tag}"),
            kind,
            4,
            2,
            64,
            64,
            vec![60, 20],
        );
        c.window = Some(40);
        c.softcap = Some(10.0);
        c.sinks = true;
        c.alibi = true;
        c.tuning = decode_only();
        attention_case(fx, c);
    } // prompt kernel: rows of one prompt share a block table, lengths 61.. (partial tiles, a second tile)
    for (kind, tag) in [(1, "f16"), (2, "q8")] {
        for (dk, dv, heads, kv_heads, rows) in [
            (64usize, 64usize, 4usize, 2usize, 40usize),
            (80, 80, 2, 2, 12),
            (96, 96, 2, 1, 33),
            (128, 128, 2, 2, 20),
            (192, 128, 2, 1, 16),
            (72, 72, 2, 2, 9),
            (32, 32, 2, 2, 10),
            (256, 256, 2, 1, 12),
            (112, 112, 2, 2, 20),
            (64, 128, 2, 2, 10),
            (128, 64, 2, 1, 40),
        ] {
            let lengths: Vec<i64> = (0..rows as i64).map(|r| 61 + r).collect();
            let mut c = AttnCase::new(
                &format!("prefill_{tag}_{dk}_{dv}_r{rows}"),
                kind,
                heads,
                kv_heads,
                dk,
                dv,
                lengths,
            );
            c.table_groups = 1;
            attention_case(fx, c);
        }
    }
    for (name, window, chunk, softcap, sinks, alibi) in [
        ("window", Some(40usize), None, None, false, false),
        ("chunk", None, Some(32usize), None, false, false),
        ("softcap_sinks_alibi", None, None, Some(6.0f32), true, true),
        ("all", Some(50), None, Some(12.0), true, true),
    ] {
        for kind in [1, 2] {
            let lengths: Vec<i64> = (0..36).map(|r| 20 + r).collect();
            let mut c = AttnCase::new(
                &format!("prefill_{name}_{}", if kind == 1 { "f16" } else { "q8" }),
                kind,
                4,
                2,
                64,
                64,
                lengths,
            );
            c.table_groups = 1;
            c.window = window;
            c.chunk = chunk;
            c.softcap = softcap;
            c.sinks = sinks;
            c.alibi = alibi;
            attention_case(fx, c);
        }
    }
    // block sizes other than 16 (index arithmetic of the block table lookup)
    for (block, kind) in [(8usize, 1), (24, 2), (32, 1), (1, 1)] {
        let mut c = AttnCase::new(
            &format!("decode_block{block}_k{kind}"),
            kind,
            4,
            2,
            64,
            64,
            if block == 1 {
                vec![18, 7]
            } else {
                vec![120, 37]
            },
        );
        c.block = block;
        c.tuning = decode_only();
        attention_case(fx, c);
        if block != 1 {
            let mut c = AttnCase::new(
                &format!("prefill_block{block}_k{kind}"),
                kind,
                4,
                2,
                64,
                64,
                (0..34).map(|r| 40 + r).collect(),
            );
            c.block = block;
            c.table_groups = 1;
            attention_case(fx, c);
        }
    }
    // two prompts in one batch (two prompt dispatches with different tables), rows with tables of their own (decode kernel) and a small tail
    {
        let mut lengths: Vec<i64> = (0..10).map(|r| 30 + r).collect();
        lengths.extend((0..12).map(|r| 50 + r));
        let mut c = AttnCase::new("prefill_two_prompts", 1, 2, 2, 64, 64, lengths);
        c.table_groups = 2;
        attention_case(fx, c);
        let c = AttnCase::new(
            "decode_independent_rows",
            1,
            2,
            2,
            64,
            64,
            (0..20).map(|r| 25 + r).collect(),
        );
        attention_case(fx, c);
        let mut c = AttnCase::new(
            "prefill_with_decode_tail",
            2,
            2,
            2,
            64,
            64,
            (0..35).map(|r| 40 + r).collect(),
        );
        c.table_groups = 1;
        attention_case(fx, c);
    }
}
/// Every case of this file, in order.
pub(super) fn run_all_cases(fx: &mut dyn Sink) {
    norm_cases(fx);
    rope_cases(fx);
    activation_cases(fx);
    elementwise_cases(fx);
    embedding_cases(fx);
    kv_write_cases(fx);
    attention_cases(fx);
}

#[test]
fn emu_fixtures() {
    let Some(mut fx) = Fixtures::from_env() else {
        eprintln!("emu_fixtures: FLLM_EMU_DIR is not set, nothing written");
        return;
    };
    run_all_cases(&mut fx);
    fx.finish();
}
