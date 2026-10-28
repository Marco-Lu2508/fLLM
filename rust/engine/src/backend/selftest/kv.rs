//! Self-test cases for the KV cache formats beyond f16 / q8_0 (`crate::kv_format`): writes and paged attention on the
//! device against the CPU backend and an f64 reference, keys and values in independent formats.
use super::*;
use crate::kv_format::{pack_kind, unpack_kind, KvFormat, KvLayout};

/// Layout of the key (`key`) or value pool of a generic KV kind; `None` for the original f16 / q8_0 kinds.
pub(super) fn layout_of(kind: i32, key: bool, slots: usize, kv_heads: usize, dim: usize) -> Option<KvLayout> {
    if matches!(kind, 1 | 2) {
        return None;
    }
    let (k, v) = unpack_kind(kind)?;
    Some(KvLayout::new(if key { k } else { v }, slots, kv_heads, dim))
}

pub(super) fn kind_name(kind: i32) -> String {
    match unpack_kind(kind) {
        Some((k, v)) if k == v => k.name().to_owned(),
        Some((k, v)) => format!("{}/{}", k.name(), v.name()),
        None => format!("kind{kind}"),
    }
}

/// Formats with their own checks here (f16 is covered by the original ones).
const GENERIC_FORMATS: [KvFormat; 12] = [
    KvFormat::Q8_0,
    KvFormat::Bf16,
    KvFormat::Fp8E4m3,
    KvFormat::Fp8E5m2,
    KvFormat::Int8,
    KvFormat::Q5_0,
    KvFormat::Q5_1,
    KvFormat::Q4_0,
    KvFormat::Q4_1,
    KvFormat::Iq4Nl,
    KvFormat::Nvfp4,
    KvFormat::F32,
];

/// Mixed key / value formats.
const MIXED_FORMATS: [(KvFormat, KvFormat); 7] = [
    (KvFormat::F16, KvFormat::Q8_0),
    (KvFormat::Q8_0, KvFormat::F16),
    (KvFormat::F16, KvFormat::Q4_0),
    (KvFormat::Fp8E4m3, KvFormat::F16),
    (KvFormat::Q8_0, KvFormat::Q4_0),
    (KvFormat::Q5_0, KvFormat::Fp8E4m3),
    (KvFormat::Nvfp4, KvFormat::Bf16),
];

fn all_kinds() -> Vec<i32> {
    let mut kinds: Vec<i32> = GENERIC_FORMATS.iter().map(|&f| pack_kind(f, f)).collect();
    kinds.extend(MIXED_FORMATS.iter().map(|&(k, v)| pack_kind(k, v)));
    kinds
}

fn generic_attention_cases() -> Vec<AttnCase> {
    let mut cases = Vec::new();
    for kind in all_kinds() {
        let mut list = vec![
            AttnCase::new(kind, 64, &[37]),
            AttnCase::new(kind, 128, &[1, 17, 33, 64, 100]),
            AttnCase { block_size: 5, stride: 16, pool_blocks: 33, ..AttnCase::new(kind, 80, &[23, 61]) },
            AttnCase { stride: 48, pool_blocks: 128, ..AttnCase::new(kind, 128, &[700]) },
            AttnCase { prefill: 20, ..AttnCase::new(kind, 64, &[40]) },
            AttnCase { prefill: 20, ..AttnCase::new(kind, 128, &[50, 33]) },
            AttnCase { window: Some(24), ..AttnCase::new(kind, 64, &[100, 10]) },
            AttnCase { softcap: Some(5.0), scale: Some(0.5), ..AttnCase::new(kind, 64, &[50]) },
            AttnCase { sinks: true, ..AttnCase::new(kind, 64, &[45, 2]) },
            AttnCase { alibi: true, heads: 4, kv_heads: 4, ..AttnCase::new(kind, 32, &[60]) },
            AttnCase { dv: 32, ..AttnCase::new(kind, 64, &[33, 9]) },
            AttnCase { heads: 2, kv_heads: 1, ..AttnCase::new(kind, 576, &[20]) },
        ];
        let Some((k, v)) = unpack_kind(kind) else { continue };
        list.retain(|c| k.check_dim(c.dk).is_ok() && v.check_dim(c.dv).is_ok());
        cases.extend(list);
    }
    cases
}

impl Tester<'_> {
    /// Whether the device implements the generic KV formats (the wgpu backends do not).
    fn kv_formats_supported(&self) -> bool {
        self.device
            .allocate_kv_cache(1, 2, 16, 1, 32, 32, pack_kind(KvFormat::Bf16, KvFormat::Bf16))
            .is_ok()
    }

    pub(super) fn run_kv_formats(&mut self) {
        if !self.kv_formats_supported() {
            println!("kv formats         skipped: {} implements only the f16 and q8_0 KV cache", self.device.name());
            return;
        }
        let mut kinds = all_kinds();
        kinds.push(pack_kind(KvFormat::F16, KvFormat::F16));
        for kind in kinds {
            let Some((k, v)) = unpack_kind(kind) else { continue };
            for (dim, block_size) in [(32, 16), (64, 16), (80, 5), (128, 16)] {
                if k.check_dim(dim).is_err() || v.check_dim(dim).is_err() {
                    continue;
                }
                for rows in [1, 13] {
                    let r = self.kv_write_generic(kind, dim, block_size, rows);
                    self.record(
                        &format!("kv_write {}", kind_name(kind)),
                        &format!("{rows} rows 2x{dim} block {block_size}"),
                        r,
                    );
                }
            }
        }
        for case in generic_attention_cases() {
            let label = format!("attention {}", kind_name(case.kind));
            let shape = case.describe();
            let r = self.attention(&case);
            self.record(&label, &shape, r);
        }
    }

    /// The fast prompt path of the CUDA backend: quantized prompts are expanded to FP16 and run on the tensor-core kernels.
    pub(super) fn run_kv_formats_fast(&mut self) {
        if !self.kv_formats_supported() {
            return;
        }
        self.fast = true;
        for fmt in GENERIC_FORMATS {
            let kind = pack_kind(fmt, fmt);
            for case in [
                AttnCase { prefill: 40, ..AttnCase::new(kind, 64, &[90]) },
                AttnCase { prefill: 33, ..AttnCase::new(kind, 128, &[70, 50]) },
            ] {
                let label = format!("attention {} fast", kind_name(kind));
                let shape = case.describe();
                let r = self.attention(&case);
                self.record(&label, &shape, r);
            }
        }
        self.fast = false;
    }

    /// Write rows into a fresh cache on the device and on the CPU backend: each decoded value must be within the
    /// format's quantization step of its source, and the two pools must agree (nearly) bit for bit.
    fn kv_write_generic(&mut self, kind: i32, dim: usize, block_size: usize, rows: usize) -> Result<Outcome, BackendError> {
        let (layers, layer, kv_heads) = (3usize, 1usize, 2usize);
        let pool_blocks = 33usize;
        let slots_total = pool_blocks * block_size;
        let mut slots: Vec<usize> = Vec::new();
        while slots.len() < rows {
            let s = (self.rng.next() % slots_total as u64) as usize;
            if !slots.contains(&s) {
                slots.push(s);
            }
        }
        let kv = kv_heads * dim;
        let mut k = self.rng.vector(rows * kv, 2.0);
        let mut v = self.rng.vector(rows * kv, 2.0);
        // an all-zero group, and one large outlier per row like real key channels
        k[..dim.min(32)].fill(0.0);
        for r in 0..rows {
            k[r * kv + dim / 2 + 1] *= 12.0;
            v[r * kv + 3] *= 7.0;
        }
        let (kf, vf) = unpack_kind(kind).ok_or_else(|| BackendError::UnsupportedOperation("kv kind".into()))?;
        let words = [kf.layer_words(slots_total, kv_heads, dim), vf.layer_words(slots_total, kv_heads, dim)];
        let op = |backend: &dyn Backend, stream: &mut dyn Stream, ms: &mut f64| -> Result<Vec<f32>, BackendError> {
            let cache = backend.allocate_kv_cache(layers, pool_blocks, block_size, kv_heads, dim, dim, kind)?;
            let kb = put(backend, stream, &k)?;
            let vb = put(backend, stream, &v)?;
            timed(ms, stream, |s| s.kv_cache_write(&cache, layer, &slots, &kb, &vb))?;
            let mut out = Vec::new();
            for (pool, w) in [(&cache.key_pool, words[0]), (&cache.value_pool, words[1])] {
                let mut bytes = vec![0u8; layers * w * 2];
                stream.download(&mut bytes, pool)?;
                out.extend(bytes.as_chunks::<2>().0.iter().map(|c| f32::from(u16::from_le_bytes(*c))));
            }
            Ok(out)
        };
        let (device, cpu, ms) = self.both(&op)?;
        let evaluate = |pools: &[f32]| -> f64 {
            let all = words_of(pools);
            let (kw, vw) = all.split_at(layers * words[0]);
            // (error, element, decoded value, source value, allowed error) of the worst element
            let mut worst_at = (0.0f64, 0usize, 0.0f64, 0.0f64, 1.0f64);
            for (part, ((pool, src), fmt, w)) in [((kw, &k), kf, words[0]), ((vw, &v), vf, words[1])].into_iter().enumerate() {
                let layout = KvLayout::new(fmt, slots_total, kv_heads, dim);
                let view = PoolView {
                    words: pool,
                    layer_words: w,
                    slots: slots_total,
                    kv,
                    group: layout.group,
                    q8: false,
                    layout: Some(layout),
                };
                for (r, &slot) in slots.iter().enumerate() {
                    for c in 0..kv {
                        let x = f64::from(src[r * kv + c]);
                        let have = view.value(layer, slot, c);
                        let g = c / layout.group;
                        let amax = src[r * kv + g * layout.group..r * kv + (g + 1) * layout.group]
                            .iter()
                            .fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
                        let e = if amax == 0.0 {
                            have.abs() * 1e30
                        } else {
                            (have - x).abs() / amax / fmt.max_relative_error()
                        };
                        let at = (part * rows * kv + r * kv + c, have, x, amax * fmt.max_relative_error());
                        if e.is_nan() {
                            return note_worst(f64::INFINITY, at.0, 2 * rows * kv, at.1 as f32, at.2, at.3);
                        }
                        if e > worst_at.0 {
                            worst_at = (e, at.0, at.1, at.2, at.3);
                        }
                    }
                }
            }
            // element = (keys 0, values 1) * rows * kv + row * kv + channel; "got" is the decoded value, "expected" the source value and
            // the error scale the largest error the format allows for the element's group
            note_worst(worst_at.0, worst_at.1, 2 * rows * kv, worst_at.2 as f32, worst_at.3, worst_at.4)
        };
        // the CPU backend and the device must produce the same words (ties of the rounding may differ in a few places)
        let differing = device.iter().zip(&cpu).filter(|(a, b)| a != b).count() as f64 / device.len().max(1) as f64;
        let parity = differing / 0.003;
        if let Some(i) = device.iter().zip(&cpu).position(|(a, b)| a != b) {
            // element = word of the pools (keys, then values), "got" the device word, "expected" the CPU word
            note_worst(parity, i, device.len(), device[i], f64::from(cpu[i]), 0.003 * device.len() as f64);
        }
        Ok(Outcome { device: evaluate(&device).max(parity), cpu: evaluate(&cpu), tol: 1.0, cpu_tol: 1.0, ms })
    }
}
