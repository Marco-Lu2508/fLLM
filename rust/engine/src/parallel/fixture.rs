//! Tiny random GGUF models for the parallel self-test: a writer for the file format and recipes for Llama-, Qwen-, Gemma- and
//! MoE-style networks in any weight format. The weights are random; what matters is that the single-device and the multi-rank
//! runs read exactly the same bytes.
use crate::cpu_selftest::{random_weights, Lcg};
use std::path::Path;

pub enum Meta {
    U32(u32),
    F32(f32),
    Str(String),
    Bool(bool),
}

pub struct TensorOut {
    pub name: String,
    /// GGUF order: `[columns, rows]` or `[columns, rows, experts]`.
    pub dims: Vec<u64>,
    pub type_id: u32,
    pub data: Vec<u8>,
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

pub fn gguf_bytes(meta: &[(String, Meta)], tensors: &[TensorOut]) -> Vec<u8> {
    const ALIGN: usize = 32;
    let mut out = b"GGUF".to_vec();
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(meta.len() as u64).to_le_bytes());
    for (key, value) in meta {
        put_str(&mut out, key);
        match value {
            Meta::U32(v) => {
                out.extend_from_slice(&4u32.to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
            Meta::F32(v) => {
                out.extend_from_slice(&6u32.to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
            Meta::Bool(v) => {
                out.extend_from_slice(&7u32.to_le_bytes());
                out.push(u8::from(*v));
            }
            Meta::Str(v) => {
                out.extend_from_slice(&8u32.to_le_bytes());
                put_str(&mut out, v);
            }
        }
    }
    let mut offset = 0usize;
    let mut offsets = Vec::new();
    for t in tensors {
        offset = offset.next_multiple_of(ALIGN);
        offsets.push(offset);
        offset += t.data.len();
        put_str(&mut out, &t.name);
        out.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&t.type_id.to_le_bytes());
        out.extend_from_slice(&(*offsets.last().unwrap_or(&0) as u64).to_le_bytes());
    }
    out.resize(out.len().next_multiple_of(ALIGN), 0);
    let base = out.len();
    for (t, at) in tensors.iter().zip(&offsets) {
        out.resize(base + at, 0);
        out.extend_from_slice(&t.data);
    }
    out
}

/// Weight formats of the groups of tensors of a model.
#[derive(Clone, Copy, Debug)]
pub struct Formats {
    pub attn: u32,
    pub ffn: u32,
    pub expert: u32,
    pub embed: u32,
    pub head: u32,
}

#[derive(Clone, Debug)]
pub struct MoeShape {
    pub experts: usize,
    pub used: usize,
    pub expert_ff: usize,
    /// Width of the always-active shared expert, if any.
    pub shared_ff: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct TinySpec {
    pub name: &'static str,
    pub arch: &'static str,
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub ff: usize,
    pub vocab: usize,
    pub formats: Formats,
    pub moe: Option<MoeShape>,
    pub qk_norm: bool,
    pub qkv_bias: bool,
    /// Gemma-2 style: post-attention / post-FFN norms, sliding window, soft-capping, tied embeddings.
    pub gemma: bool,
    pub seed: u64,
}

pub const F32: u32 = 0;
pub const F16: u32 = 1;
pub const Q8_0: u32 = 8;
pub const Q4_K: u32 = 12;
pub const Q6_K: u32 = 14;
pub const NVFP4: u32 = 40;
pub const Q4Z32: u32 = 0x8000_0010;
pub const Q4Z128: u32 = 0x8000_0012;
pub const Q4ZP128: u32 = 0x8000_0016;

const DENSE: (usize, usize, usize, usize, usize) = (4, 256, 2048, 512, 16);

fn base(name: &'static str, arch: &'static str, formats: Formats, seed: u64) -> TinySpec {
    let (layers, hidden, ff, vocab, heads) = DENSE;
    TinySpec { name, arch, layers, hidden, heads, kv_heads: 4, head_dim: 128, ff, vocab, formats, moe: None, qk_norm: false, qkv_bias: false, gemma: false, seed }
}

impl TinySpec {
    /// Names of the recipes `named` knows.
    pub const CASES: &'static [&'static str] =
        &["llama-q4k", "qwen3-q8", "qwen2-q4z", "gemma2-f32", "llama-nvfp4", "llama-q4zp", "moe-q4k", "moe-shared-f32", "llama-f16"];

    pub fn named(name: &str) -> Option<TinySpec> {
        let all = |f: u32| Formats { attn: f, ffn: f, expert: f, embed: F32, head: F32 };
        Some(match name {
            "llama-q4k" => base("llama-q4k", "llama", all(Q4_K), 11),
            "qwen3-q8" => TinySpec { qk_norm: true, formats: Formats { embed: Q8_0, head: Q8_0, ..all(Q8_0) }, ..base("qwen3-q8", "qwen3", all(Q8_0), 12) },
            "qwen2-q4z" => TinySpec { qkv_bias: true, formats: Formats { head: F16, ..all(Q4Z128) }, ..base("qwen2-q4z", "qwen2", all(Q4Z128), 13) },
            "gemma2-f32" => TinySpec { gemma: true, formats: Formats { embed: F16, ..all(F32) }, ..base("gemma2-f32", "gemma2", all(F32), 14) },
            "llama-nvfp4" => base("llama-nvfp4", "llama", Formats { attn: NVFP4, ffn: NVFP4, expert: NVFP4, embed: F32, head: Q6_K }, 15),
            "llama-q4zp" => base("llama-q4zp", "llama", Formats { attn: Q4ZP128, ffn: Q4Z32, expert: Q4Z32, embed: F32, head: F32 }, 16),
            "llama-f16" => base("llama-f16", "llama", all(F16), 17),
            "moe-q4k" => TinySpec {
                layers: 4,
                moe: Some(MoeShape { experts: 8, used: 2, expert_ff: 1024, shared_ff: None }),
                qk_norm: true,
                ..base("moe-q4k", "qwen3moe", Formats { attn: Q4_K, ffn: Q4_K, expert: Q4_K, embed: F32, head: F32 }, 18)
            },
            "moe-shared-f32" => TinySpec {
                layers: 4,
                moe: Some(MoeShape { experts: 8, used: 3, expert_ff: 512, shared_ff: Some(512) }),
                qkv_bias: true,
                ..base("moe-shared-f32", "qwen2moe", Formats { attn: Q8_0, ffn: Q8_0, expert: F32, embed: F32, head: F32 }, 19)
            },
            _ => return None,
        })
    }

    fn matrix(&self, kind: u32, rows: usize, cols: usize, rng: &mut Lcg) -> Vec<u8> {
        match kind {
            F32 | F16 | 30 => {
                let amplitude = 1.7 / (cols as f32).sqrt();
                let mut out = Vec::with_capacity(rows * cols * 4);
                for _ in 0..rows * cols {
                    let v = rng.unit() * amplitude;
                    match kind {
                        F32 => out.extend_from_slice(&v.to_le_bytes()),
                        F16 => out.extend_from_slice(&half::f16::from_f32(v).to_le_bytes()),
                        _ => out.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes()),
                    }
                }
                out
            }
            _ => {
                let mut bytes = random_weights(kind, rows, cols, rng);
                if fllm_gguf_probe::q4zp_group(kind).is_some() {
                    // permutations inside blocks of 256 columns, so that every tensor-parallel cut is legal
                    let at = bytes.len() - cols * 4;
                    let chunk = 256.min(cols);
                    for c0 in (0..cols).step_by(chunk) {
                        let mut perm: Vec<u32> = (0..chunk).map(|i| (c0 + i) as u32).collect();
                        for i in (1..chunk).rev() {
                            perm.swap(i, (rng.next() as usize) % (i + 1));
                        }
                        for (i, p) in perm.iter().enumerate() {
                            bytes[at + (c0 + i) * 4..at + (c0 + i) * 4 + 4].copy_from_slice(&p.to_le_bytes());
                        }
                    }
                }
                bytes
            }
        }
    }

    fn vector(&self, n: usize, rng: &mut Lcg, around: f32) -> Vec<u8> {
        (0..n).flat_map(|_| (around + 0.1 * rng.unit()).to_le_bytes()).collect()
    }

    pub fn tensors(&self) -> Vec<TensorOut> {
        let mut rng = Lcg(self.seed);
        let mut out: Vec<TensorOut> = Vec::new();
        let (h, dk) = (self.hidden, self.head_dim);
        let f = self.formats;
        let push = |out: &mut Vec<TensorOut>, name: String, dims: Vec<u64>, type_id: u32, data: Vec<u8>| out.push(TensorOut { name, dims, type_id, data });
        let mat = |s: &Self, out: &mut Vec<TensorOut>, name: String, kind: u32, rows: usize, cols: usize, rng: &mut Lcg| {
            let data = s.matrix(kind, rows, cols, rng);
            push(out, name, vec![cols as u64, rows as u64], kind, data);
        };
        mat(self, &mut out, "token_embd.weight".into(), f.embed, self.vocab, h, &mut rng);
        let norm = self.vector(h, &mut rng, 1.0);
        push(&mut out, "output_norm.weight".into(), vec![h as u64], F32, norm);
        if !self.gemma {
            mat(self, &mut out, "output.weight".into(), f.head, self.vocab, h, &mut rng);
        }
        for l in 0..self.layers {
            let p = format!("blk.{l}");
            let v = self.vector(h, &mut rng, 1.0);
            push(&mut out, format!("{p}.attn_norm.weight"), vec![h as u64], F32, v);
            mat(self, &mut out, format!("{p}.attn_q.weight"), f.attn, self.heads * dk, h, &mut rng);
            mat(self, &mut out, format!("{p}.attn_k.weight"), f.attn, self.kv_heads * dk, h, &mut rng);
            mat(self, &mut out, format!("{p}.attn_v.weight"), f.attn, self.kv_heads * dk, h, &mut rng);
            mat(self, &mut out, format!("{p}.attn_output.weight"), f.attn, h, self.heads * dk, &mut rng);
            if self.qkv_bias {
                for (n, rows) in [("attn_q", self.heads * dk), ("attn_k", self.kv_heads * dk), ("attn_v", self.kv_heads * dk)] {
                    let bias: Vec<u8> = (0..rows).flat_map(|_| (0.3 * rng.unit()).to_le_bytes()).collect();
                    push(&mut out, format!("{p}.{n}.bias"), vec![rows as u64], F32, bias);
                }
            }
            if self.qk_norm {
                for n in ["attn_q_norm", "attn_k_norm"] {
                    let v = self.vector(dk, &mut rng, 1.0);
                    push(&mut out, format!("{p}.{n}.weight"), vec![dk as u64], F32, v);
                }
            }
            let v = self.vector(h, &mut rng, 1.0);
            push(&mut out, format!("{p}.ffn_norm.weight"), vec![h as u64], F32, v);
            if self.gemma {
                for n in ["post_attention_norm", "post_ffw_norm"] {
                    let v = self.vector(h, &mut rng, 1.0);
                    push(&mut out, format!("{p}.{n}.weight"), vec![h as u64], F32, v);
                }
            }
            match &self.moe {
                None => {
                    mat(self, &mut out, format!("{p}.ffn_gate.weight"), f.ffn, self.ff, h, &mut rng);
                    mat(self, &mut out, format!("{p}.ffn_up.weight"), f.ffn, self.ff, h, &mut rng);
                    mat(self, &mut out, format!("{p}.ffn_down.weight"), f.ffn, h, self.ff, &mut rng);
                }
                Some(m) => {
                    mat(self, &mut out, format!("{p}.ffn_gate_inp.weight"), F32, m.experts, h, &mut rng);
                    for (n, rows, cols) in [("gate", m.expert_ff, h), ("up", m.expert_ff, h), ("down", h, m.expert_ff)] {
                        let data = self.matrix(f.expert, rows * m.experts, cols, &mut rng);
                        push(&mut out, format!("{p}.ffn_{n}_exps.weight"), vec![cols as u64, rows as u64, m.experts as u64], f.expert, data);
                    }
                    if let Some(sf) = m.shared_ff {
                        mat(self, &mut out, format!("{p}.ffn_gate_shexp.weight"), f.ffn, sf, h, &mut rng);
                        mat(self, &mut out, format!("{p}.ffn_up_shexp.weight"), f.ffn, sf, h, &mut rng);
                        mat(self, &mut out, format!("{p}.ffn_down_shexp.weight"), f.ffn, h, sf, &mut rng);
                        let g: Vec<u8> = (0..h).flat_map(|_| (0.2 * rng.unit()).to_le_bytes()).collect();
                        push(&mut out, format!("{p}.ffn_gate_inp_shexp.weight"), vec![h as u64], F32, g);
                    }
                }
            }
        }
        out
    }

    pub fn metadata(&self) -> Vec<(String, Meta)> {
        let a = self.arch;
        let mut m: Vec<(String, Meta)> = vec![
            ("general.architecture".into(), Meta::Str(a.into())),
            (format!("{a}.embedding_length"), Meta::U32(self.hidden as u32)),
            (format!("{a}.block_count"), Meta::U32(self.layers as u32)),
            (format!("{a}.attention.head_count"), Meta::U32(self.heads as u32)),
            (format!("{a}.attention.head_count_kv"), Meta::U32(self.kv_heads as u32)),
            (format!("{a}.attention.key_length"), Meta::U32(self.head_dim as u32)),
            (format!("{a}.attention.value_length"), Meta::U32(self.head_dim as u32)),
            (format!("{a}.feed_forward_length"), Meta::U32(self.ff as u32)),
            (format!("{a}.context_length"), Meta::U32(256)),
            (format!("{a}.attention.layer_norm_rms_epsilon"), Meta::F32(1e-5)),
            (format!("{a}.rope.freq_base"), Meta::F32(10_000.0)),
            (format!("{a}.rope.dimension_count"), Meta::U32(self.head_dim as u32)),
        ];
        if let Some(moe) = &self.moe {
            m.push((format!("{a}.expert_count"), Meta::U32(moe.experts as u32)));
            m.push((format!("{a}.expert_used_count"), Meta::U32(moe.used as u32)));
            m.push((format!("{a}.expert_feed_forward_length"), Meta::U32(moe.expert_ff as u32)));
            m.push((format!("{a}.expert_weights_norm"), Meta::Bool(true)));
        }
        if self.gemma {
            m.push((format!("{a}.attention.sliding_window"), Meta::U32(8)));
            m.push((format!("{a}.final_logit_softcapping"), Meta::F32(30.0)));
            m.push((format!("{a}.attn_logit_softcapping"), Meta::F32(50.0)));
        }
        m
    }

    pub fn write(&self, path: &Path) -> Result<(), String> {
        std::fs::write(path, gguf_bytes(&self.metadata(), &self.tensors())).map_err(|e| format!("cannot write {}: {e}", path.display()))
    }
}

/// A unique scratch path for a fixture file.
pub fn scratch_path(tag: &str) -> std::path::PathBuf {
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!("fllm-parallel-{}-{n}-{tag}.gguf", std::process::id()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_recipe_writes_a_gguf_that_loads() {
        for name in TinySpec::CASES {
            let spec = TinySpec::named(name).unwrap_or_else(|| panic!("{name}"));
            let path = scratch_path(name);
            spec.write(&path).unwrap();
            let model = crate::model::GgufModel::load(path.to_str().unwrap(), 64, 2, 1);
            let _ = std::fs::remove_file(&path);
            let model = model.unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(model.layer_count(), spec.layers, "{name}");
            assert_eq!(model.heads(), spec.heads, "{name}");
        }
    }
}
