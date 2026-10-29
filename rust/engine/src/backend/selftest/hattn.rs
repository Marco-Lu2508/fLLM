//! Self-test cases: Hopper and FlashInfer attention paths (CUDA backend, group `hattn`).
//!
//! Every case runs `attention_paged` on the f16 cache with one set of the attention switches (`hopper_attn::AttnSwitch`) and compares the
//! result with the f64 reference of the `attention` group; the label names the path that really ran (`[Fa3]`, `[FlashInfer]`, `[Cascade]`,
//! `[Builtin]`). A case that is meant for a path this build or device cannot run is skipped; one whose path does not run although it could is
//! a failure ("expected path X, ran Y"), so a silent fall-back to the built-in kernels cannot pass for a verified FA3 / FlashInfer kernel.
//! With `FLLM_SELFTEST_BENCH=1` the group adds timing cases at model-like sizes (no reference, the time of one call averaged over repetitions).
use super::*;
use crate::backend::cuda::hopper_attn::{attn_caps, attn_last_path, reset_attn_last_path, set_attn_switch, AttnPath, AttnSwitch};

/// The path a case is meant to exercise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Expect {
    /// whichever path the switches select
    Any,
    Fa3,
    FlashInfer,
    Cascade,
}

/// Values of the four switches (see `AttnSwitch`).
#[derive(Clone, Copy)]
struct Switches {
    fi: i32,
    decode: i32,
    cascade: i32,
    backend: i32,
}

impl Switches {
    /// The defaults of the engine: FlashInfer variants off, builtin decode, no cascade, backend auto.
    const DEFAULT: Self = Self { fi: 0, decode: 0, cascade: 0, backend: 0 };
    /// Variants, ragged batches and window through FlashInfer.
    const FI: Self = Self { fi: 1, ..Self::DEFAULT };
    /// The FlashInfer FA2 variants even on Hopper (FA3 off).
    const FI_FA2: Self = Self { fi: 1, backend: 1, ..Self::DEFAULT };
    const DECODE: Self = Self { decode: 1, ..Self::DEFAULT };
    const CASCADE: Self = Self { decode: 1, cascade: 1, ..Self::DEFAULT };
    /// FA2 only: what the engine did before the FA3 port.
    const FA2: Self = Self { backend: 1, ..Self::DEFAULT };

    fn apply(self) -> [i32; 4] {
        [
            set_attn_switch(AttnSwitch::Fi, self.fi),
            set_attn_switch(AttnSwitch::Decode, self.decode),
            set_attn_switch(AttnSwitch::Cascade, self.cascade),
            set_attn_switch(AttnSwitch::Backend, self.backend),
        ]
    }
}

fn restore(old: [i32; 4]) {
    set_attn_switch(AttnSwitch::Fi, old[0]);
    set_attn_switch(AttnSwitch::Decode, old[1]);
    set_attn_switch(AttnSwitch::Cascade, old[2]);
    set_attn_switch(AttnSwitch::Backend, old[3]);
}

fn path_of(expect: Expect) -> Option<AttnPath> {
    match expect {
        Expect::Any => None,
        Expect::Fa3 => Some(AttnPath::Fa3),
        Expect::FlashInfer => Some(AttnPath::FlashInfer),
        Expect::Cascade => Some(AttnPath::Cascade),
    }
}

impl Tester<'_> {
    pub(super) fn hattn_cases(&mut self) {
        let caps = attn_caps();
        let f16 = |dk: usize, sequences: &[usize]| AttnCase::new(1, dk, sequences);
        // Prompt runs the FlashInfer path takes over: sliding window, soft cap, sinks, ragged batches, head widths 64 / 128 / 256.
        let prompt: Vec<(&str, AttnCase)> = vec![
            ("window prefill", AttnCase { prefill: 24, window: Some(40), ..f16(64, &[90]) }),
            ("window prefill d128", AttnCase { prefill: 40, window: Some(33), ..f16(128, &[150]) }),
            ("soft cap prefill", AttnCase { prefill: 20, softcap: Some(30.0), scale: Some(0.5), ..f16(64, &[64]) }),
            ("sinks prefill", AttnCase { prefill: 17, sinks: true, ..f16(64, &[50]) }),
            ("sinks + window prefill", AttnCase { prefill: 20, sinks: true, window: Some(32), ..f16(64, &[70, 130, 33]) }),
            ("ragged prefill", AttnCase { prefill: 20, ..f16(128, &[70, 130, 33]) }),
            ("ragged prefill d256", AttnCase { prefill: 20, heads: 4, kv_heads: 1, ..f16(256, &[64, 100]) }),
            ("window prefill gemma", AttnCase { prefill: 32, heads: 8, kv_heads: 4, softcap: Some(50.0), window: Some(48), ..f16(256, &[120]) }),
        ];
        for (label, case) in prompt {
            self.hattn_run(label, Switches::FI_FA2, Expect::FlashInfer, caps_for(caps.flashinfer), case);
        }
        // Decode on tensor cores: GQA groups of 4 and 8, long contexts (split KV), options, head widths.
        let decode: Vec<(&str, AttnCase)> = vec![
            ("decode group 4", f16(128, &[1, 17, 33, 64, 100])),
            ("decode group 8", AttnCase { heads: 16, kv_heads: 2, ..f16(64, &[200, 40]) }),
            ("decode group 1", AttnCase { heads: 4, kv_heads: 4, ..f16(64, &[90, 5]) }),
            (
                "decode long",
                AttnCase { pool_blocks: 220, stride: 110, ..f16(128, &[1500, 900, 40]) },
            ),
            ("decode window + sinks", AttnCase { window: Some(24), sinks: true, ..f16(64, &[100, 10]) }),
            ("decode soft cap", AttnCase { softcap: Some(5.0), scale: Some(0.5), ..f16(64, &[50]) }),
            ("decode d256", AttnCase { heads: 4, kv_heads: 1, ..f16(256, &[200, 31]) }),
            ("speculative rows", AttnCase { prefill: 3, ..f16(128, &[60, 21]) }),
        ];
        for (label, case) in decode {
            self.hattn_run(label, Switches::DECODE, Expect::FlashInfer, caps_for(caps.flashinfer), case);
        }
        // Cascade attention: all sequences share the leading 8 blocks (128 tokens); the last case adds a sequence that is too short to share.
        let cascade: Vec<(&str, AttnCase)> = vec![
            ("cascade", AttnCase { shared_blocks: 8, ..f16(128, &[200, 180, 300, 150]) }),
            ("cascade sinks", AttnCase { shared_blocks: 8, sinks: true, ..f16(64, &[200, 180]) }),
            ("cascade soft cap", AttnCase { shared_blocks: 8, softcap: Some(20.0), ..f16(128, &[200, 150, 333]) }),
            ("cascade + single", AttnCase { shared_blocks: 8, ..f16(128, &[200, 180, 5]) }),
        ];
        for (label, case) in cascade {
            self.hattn_run(label, Switches::CASCADE, Expect::Cascade, caps_for(caps.flashinfer), case);
        }
        // FA3 on Hopper: plain prompts through the engine's prompt attention (default switches), ragged and windowed ones through FLLM_ATTN_FI=1.
        let fa3: Vec<(&str, Switches, AttnCase)> = vec![
            ("prefill d64", Switches::DEFAULT, AttnCase { prefill: 64, ..f16(64, &[200]) }),
            ("prefill d128", Switches::DEFAULT, AttnCase { prefill: 64, ..f16(128, &[200]) }),
            ("window prefill d256", Switches::FI, AttnCase { prefill: 40, window: Some(50), heads: 4, kv_heads: 1, ..f16(256, &[120]) }),
            ("prefill gqa 8", Switches::DEFAULT, AttnCase { prefill: 48, heads: 16, kv_heads: 2, ..f16(128, &[150]) }),
            ("ragged prefill", Switches::FI, AttnCase { prefill: 20, ..f16(128, &[70, 130, 33]) }),
            ("window prefill", Switches::FI, AttnCase { prefill: 40, window: Some(50), ..f16(64, &[130]) }),
            ("window prefill d128", Switches::FI, AttnCase { prefill: 40, window: Some(50), ..f16(128, &[130, 90]) }),
        ];
        for (label, sw, case) in fa3 {
            self.hattn_run(&format!("fa3 {label}"), sw, Expect::Fa3, caps_for(caps.fa3), case);
        }
        // The same plain prompt with FA3 switched off must still match (the FA2 kernel, or the built-in one without FlashInfer headers).
        self.hattn_run("fa2 prefill d128", Switches::FA2, Expect::Any, caps_for(true), AttnCase { prefill: 64, ..f16(128, &[200]) });
        if std::env::var("FLLM_SELFTEST_BENCH").is_ok_and(|v| !v.is_empty() && v != "0") {
            self.hattn_bench();
        }
    }

    /// One case under one set of switches.
    fn hattn_run(&mut self, label: &str, switches: Switches, expect: Expect, available: Option<&'static str>, case: AttnCase) {
        let shape = case.describe();
        if let Some(reason) = available {
            self.record_skip(&format!("hattn {label}"), &shape, reason);
            return;
        }
        let old = switches.apply();
        reset_attn_last_path();
        let r = self.attention(&case);
        let path = attn_last_path();
        restore(old);
        let r = r.and_then(|mut o| {
            // The tensor-core kernels round the probabilities and the output to f16 (the budget of the reference is that of an FP32 kernel): one f16
            // unit roundoff of the output scale on top, 2.5 times what the runs on an L4 showed
            if matches!(path, AttnPath::FlashInfer | AttnPath::Cascade) {
                o.tol += 4.8828125e-4;
            }
            Ok(o)
        });
        let r = r.and_then(|o| match path_of(expect) {
            Some(want) if want != path => Err(BackendError::DriverError(format!("expected the {want:?} path to run, the {path:?} path ran (switches: fi {}, decode {}, cascade {}, backend {})", switches.fi, switches.decode, switches.cascade, switches.backend))),
            _ => Ok(o),
        });
        self.record(&format!("hattn {label} [{path:?}]"), &shape, r);
    }

    /// Timing cases (FLLM_SELFTEST_BENCH=1): model-like sizes, each path against the one it replaces. The device time of one call, averaged.
    fn hattn_bench(&mut self) {
        let caps = attn_caps();
        let big = |dk: usize, heads: usize, kv_heads: usize, sequences: &[usize]| AttnCase {
            heads,
            kv_heads,
            block_size: 16,
            stride: sequences.iter().map(|n| n.div_ceil(16)).max().unwrap_or(1) + 2,
            pool_blocks: sequences.iter().map(|n| n.div_ceil(16)).sum::<usize>() + 4,
            ..AttnCase::new(1, dk, sequences)
        };
        let mut items: Vec<(String, Switches, AttnCase)> = Vec::new();
        for (name, sw) in [("fa2/builtin", Switches::FA2), ("default", Switches::DEFAULT)] {
            items.push((format!("prefill 2048 of 4096 {name}"), sw, AttnCase { prefill: 2048, ..big(128, 32, 8, &[4096]) }));
        }
        for (name, sw) in [("builtin", Switches::DEFAULT), ("flashinfer", Switches::FI)] {
            items.push((
                format!("prefill 2048 window 1024 {name}"),
                sw,
                AttnCase { prefill: 2048, window: Some(1024), ..big(128, 32, 8, &[4096]) },
            ));
            items.push((
                format!("prefill 512 sinks {name}"),
                sw,
                AttnCase { prefill: 512, sinks: true, ..big(64, 64, 8, &[2048]) },
            ));
        }
        for (name, sw) in [("builtin", Switches::DEFAULT), ("flashinfer", Switches::DECODE)] {
            items.push((format!("decode 32 x 4096 {name}"), sw, big(128, 32, 8, &[4096; 32])));
            items.push((format!("decode 8 x 16384 {name}"), sw, big(128, 32, 8, &[16384; 8])));
            items.push((format!("decode 32 x 2048 group 4 {name}"), sw, big(128, 32, 8, &[2048; 32])));
        }
        for (name, sw) in [("builtin", Switches::DEFAULT), ("cascade", Switches::CASCADE)] {
            // 64 sequences with a shared prefix of 2048 tokens and 256 own tokens
            items.push((format!("decode 64 shared 2048 + 256 {name}"), sw, AttnCase { shared_blocks: 128, ..big(128, 32, 8, &[2304; 64]) }));
        }
        for (label, sw, case) in items {
            let shape = case.describe();
            let old = sw.apply();
            reset_attn_last_path();
            let r = self.attention_time(&case, 3, 12);
            let path = attn_last_path();
            restore(old);
            if !caps.flashinfer && !caps.fa3 {
                self.record_skip(&format!("hattn bench {label}"), &shape, "this build has no FlashInfer headers or the GPU is older than sm_75");
                continue;
            }
            self.record(&format!("hattn bench {label} [{path:?}]"), &shape, r);
        }
    }

    /// The time of `attention_paged` on random cache contents: warm-up calls, then the average of `reps` calls. No reference.
    fn attention_time(&mut self, case: &AttnCase, warm: usize, reps: usize) -> Result<Outcome, BackendError> {
        let (dk, dv, heads, kv_heads) = (case.dk, case.dv, case.heads, case.kv_heads);
        let batch = self.batch(case);
        let rows = batch.lengths.len();
        let q = self.rng.vector(rows * heads * dk, 1.0);
        let params = AttentionParams {
            num_heads: heads,
            num_kv_heads: kv_heads,
            head_dim: dk,
            value_dim: dv,
            stride: case.stride,
            block_size: case.block_size,
            scale: 1.0 / (dk as f32).sqrt(),
            softcap: case.softcap,
            window: case.window,
            chunk: None,
            sinks: case.sinks.then(|| Arc::new(self.rng.vector(heads, 2.0))),
            alibi_slopes: None,
        };
        let layers = 1usize;
        let layer = 0usize;
        let cache = self.device.allocate_kv_cache(layers, case.pool_blocks, case.block_size, kv_heads, dk, dv, case.kind)?;
        let mut stream = self.device.create_stream()?;
        for mut pool in [cache.key_pool.clone(), cache.value_pool.clone()] {
            // random f16 values: one period of a short random table repeated over the pool
            let table: Vec<u16> = self.rng.vector(4096, 1.0).iter().map(|&v| f16::from_f32(v).to_bits()).collect();
            let mut bytes = Vec::with_capacity(pool.size);
            while bytes.len() + 2 <= pool.size {
                let w = table[(bytes.len() / 2) % table.len()];
                bytes.extend_from_slice(&w.to_le_bytes());
            }
            bytes.resize(pool.size, 0);
            stream.upload(&mut pool, &bytes)?;
        }
        let qb = put(self.device, stream.as_mut(), &q)?;
        let mut out = self.device.allocate_buffer(rows * heads * dv * 4)?;
        let mut run = |stream: &mut dyn Stream| stream.attention_paged(&mut out, &qb, &cache, &batch.tables, &batch.lengths, layer, &params);
        for _ in 0..warm {
            run(stream.as_mut())?;
        }
        stream.synchronize()?;
        let start = Instant::now();
        for _ in 0..reps {
            run(stream.as_mut())?;
        }
        stream.synchronize()?;
        let ms = start.elapsed().as_secs_f64() * 1e3 / reps as f64;
        Ok(Outcome { device: 0.0, cpu: 0.0, tol: 1.0, cpu_tol: 1.0, ms })
    }
}

/// `None` when the path can run, else the reason for skipping.
fn caps_for(available: bool) -> Option<&'static str> {
    (!available).then_some("this build or GPU does not have the path (FlashInfer headers at build time, sm_75+; FA3: an sm_90a build on a compute capability 9.0 GPU)")
}
