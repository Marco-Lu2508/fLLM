//! Layers that run a second mixer next to the attention (Falcon-H1) and the shared pieces of the
//! hybrid families that the main decoder file keeps out of its way: the gate/norm tail of the
//! Mamba-2 scan with its three orderings.
use super::*;
use crate::config::MambaNorm;

/// The Mamba-2 branch that runs in parallel with the attention of the same layer (Falcon-H1): both read the
/// layer's normed input and their output projections are summed before the residual add.
pub(super) struct ParallelMixer {
    pub(super) mamba: Mamba,
    /// `ssm_out`: inner width back to the hidden size.
    pub(super) out: Linear,
}

impl ParallelMixer {
    /// Loads the Mamba-2 tensors of layer `p` next to an attention block, `None` when the layer has none.
    pub(super) fn load(
        cfg: &ModelConfig,
        mapping: &MappedGguf,
        p: &str,
        layer: usize,
        linear: &dyn Fn(&str, usize, usize) -> Result<Linear, String>,
    ) -> Result<Option<Box<Self>>, String> {
        let Some(m) = &cfg.mamba else { return Ok(None) };
        if !has_tensor(mapping, &format!("{p}.ssm_in.weight")) || !has_tensor(mapping, &format!("{p}.attn_q.weight")) {
            return Ok(None);
        }
        let mamba = load_mamba(mapping, p, layer, cfg.hidden, m, linear)?;
        let out = linear(&format!("{p}.ssm_out"), cfg.hidden, m.inner)?;
        Ok(Some(Box::new(Self { mamba, out })))
    }
}

/// Weights of one Mamba-2 mixer from the tensors `{p}.ssm_*`.
pub(super) fn load_mamba(
    mapping: &MappedGguf,
    p: &str,
    layer: usize,
    hidden: usize,
    m: &crate::config::MambaConfig,
    linear: &dyn Fn(&str, usize, usize) -> Result<Linear, String>,
) -> Result<Mamba, String> {
    let conv_dim = m.inner + 2 * m.groups * m.state;
    let conv = Tensor::load(mapping, &format!("{p}.ssm_conv1d.weight"))?.data;
    if conv.len() != conv_dim * m.conv_kernel {
        return Err(format!("invalid ssm_conv1d in layer {layer}"));
    }
    let bias_name = format!("{p}.ssm_conv1d.bias");
    Ok(Mamba {
        input: linear(&format!("{p}.ssm_in"), m.inner + conv_dim + m.heads, hidden)?,
        conv,
        conv_bias: if has_tensor(mapping, &bias_name) {
            Some(vector(mapping, &bias_name, conv_dim)?)
        } else {
            None
        },
        dt_bias: vector(mapping, &format!("{p}.ssm_dt.bias"), m.heads)?,
        decay: vector(mapping, &format!("{p}.ssm_a"), m.heads)?,
        skip: vector(mapping, &format!("{p}.ssm_d"), m.heads)?,
        norm: if m.norm == MambaNorm::NoNorm {
            Vec::new()
        } else {
            vector(mapping, &format!("{p}.ssm_norm.weight"), m.inner)?
        },
    })
}

/// The gate and the RMS norm behind the scan of one token: `scan` (inner width) is replaced by the mixer output
/// before its output projection.
pub(super) fn gate_and_norm(mc: &crate::config::MambaConfig, weight: &[f32], eps: f32, scan: &mut [f32], gate: &[f32]) {
    let silu = |g: f32| g / (1.0 + (-g).exp());
    let norm = |scan: &mut [f32]| {
        let size = scan.len() / mc.norm_groups.max(1);
        for (chunk, weight) in scan.chunks_mut(size).zip(weight.chunks(size)) {
            let inv = (chunk.iter().map(|v| v * v).sum::<f32>() / size as f32 + eps).sqrt().recip();
            for (v, w) in chunk.iter_mut().zip(weight) {
                *v *= inv * w;
            }
        }
    };
    match mc.norm {
        MambaNorm::GateThenNorm => {
            for (v, g) in scan.iter_mut().zip(gate) {
                *v *= silu(*g);
            }
            norm(scan);
        }
        MambaNorm::NoNorm => {
            for (v, g) in scan.iter_mut().zip(gate) {
                *v *= silu(*g);
            }
        }
        MambaNorm::NormThenGate => {
            norm(scan);
            for (v, g) in scan.iter_mut().zip(gate) {
                *v *= silu(*g);
            }
        }
    }
}

impl RustDecoder {
    /// Output of the parallel branch of layer `i` for the normed input `normed`; the caller adds it to the attention output.
    pub(super) fn parallel_branch(
        &self,
        parallel: &ParallelMixer,
        i: usize,
        normed: &[f32],
        tables: &[i64],
        lengths: &[i64],
    ) -> Result<Vec<f32>, String> {
        self.mamba_forward(&parallel.mamba, &parallel.out, i, normed, tables, lengths)
    }
}

/// Learned weights of the RMS norms on the time step, B and C of a Mamba-1 mixer (Jamba: `ssm_dt_norm`, `ssm_b_norm`,
/// `ssm_c_norm`); `None` when the layer has none.
pub(super) fn load_inner_norms(
    mapping: &MappedGguf,
    p: &str,
    m: &crate::config::Mamba1Config,
) -> Result<Option<Box<[Vec<f32>; 3]>>, String> {
    if !has_tensor(mapping, &format!("{p}.ssm_dt_norm.weight")) {
        return Ok(None);
    }
    if m.bc_rms_eps.is_none() {
        return Err("Mamba-1 norm weights need ssm.dt_b_c_rms_eps".into());
    }
    Ok(Some(Box::new([
        vector(mapping, &format!("{p}.ssm_dt_norm.weight"), m.rank)?,
        vector(mapping, &format!("{p}.ssm_b_norm.weight"), m.state)?,
        vector(mapping, &format!("{p}.ssm_c_norm.weight"), m.state)?,
    ])))
}

/// The embedding output of the pass in flight, for layers that read it again later (Zamba2's shared blocks).
#[derive(Default)]
pub(super) struct EmbedStash(std::sync::Mutex<Vec<f32>>);

impl EmbedStash {
    pub(super) fn set(&self, embeddings: &[f32]) {
        let mut guard = self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.clear();
        guard.extend_from_slice(embeddings);
    }

    /// A copy of the stashed rows; `len` is the expected number of values.
    pub(super) fn get(&self, len: usize) -> Result<Vec<f32>, String> {
        let guard = self.0.lock().map_err(|_| "embedding stash lock poisoned")?;
        if guard.len() != len {
            return Err(format!("embedding stash holds {} values, the pass needs {len}", guard.len()));
        }
        Ok(guard.clone())
    }
}

impl RustDecoder {
    /// Remembers the embedding output of this pass when some layer needs it again.
    pub(super) fn stash_embeddings(&self, embeddings: &[f32]) {
        if self.cfg.hybrid.zamba.is_some() {
            self.embed_stash.set(embeddings);
        }
    }
}
