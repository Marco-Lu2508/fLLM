//! Configuration of the hybrid layer families that do not fit the generic `ModelConfig` fields: the shared transformer
//! block of Zamba2 and the Kimi Delta Attention layers. Parsed from GGUF metadata like the rest of the config.
use serde_json::{Map, Value};

/// Zamba2: a Mamba-2 backbone with shared transformer blocks that are applied before some of the Mamba layers.
///
/// Each application ("call") of a shared block reads the concatenation of the residual stream and the embedding output,
/// owns a KV layer (its layer index) and carries its own low-rank adapters on the shared projections.
#[derive(Clone, Debug, PartialEq)]
pub struct ZambaConfig {
    /// Number of distinct shared blocks; call `n` uses block `n % blocks`.
    pub blocks: usize,
    /// Rank of the per-call adapters.
    pub rank: usize,
    /// Layers that start with a shared-block call.
    pub layers: Vec<bool>,
    /// The attention projections q, k, v carry per-call adapters too (`use_shared_attention_adapter`).
    pub attention_adapters: bool,
}

impl ZambaConfig {
    /// Shared block used by layer `layer` (`None` for a plain Mamba layer).
    pub fn block_of(&self, layer: usize) -> Option<usize> {
        if !self.layers.get(layer).copied().unwrap_or(false) {
            return None;
        }
        Some(self.layers[..layer].iter().filter(|&&b| b).count() % self.blocks.max(1))
    }
}

/// Kimi Delta Attention (Kimi-Linear, Ling 3.0, Kimi K3, GLM-5.3): a gated delta rule with a decay per key channel.
#[derive(Clone, Debug, PartialEq)]
pub struct KdaConfig {
    /// Layers that use KDA (the others keep their attention).
    pub layers: Vec<bool>,
    pub conv_kernel: usize,
    pub heads: usize,
    pub head_dim: usize,
    /// GLM-5 style bounded gate `bound * sigmoid(...)` instead of Kimi's softplus gate.
    pub lower_bound: Option<f32>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct HybridConfig {
    pub zamba: Option<ZambaConfig>,
    pub kda: Option<KdaConfig>,
}

impl HybridConfig {
    /// Whether some layer keeps a recurrent state of its own.
    pub fn stateful(&self) -> bool {
        self.kda.is_some()
    }
}

fn value<'a>(meta: &'a Map<String, Value>, arch: &str, name: &str) -> Option<&'a Value> {
    meta.get(&format!("{arch}.{name}")).or_else(|| meta.get(name))
}

fn uint(meta: &Map<String, Value>, arch: &str, name: &str) -> Option<usize> {
    value(meta, arch, name).and_then(Value::as_u64).map(|v| v as usize)
}

fn bools(meta: &Map<String, Value>, arch: &str, name: &str, layers: usize) -> Result<Option<Vec<bool>>, String> {
    match value(meta, arch, name) {
        None => Ok(None),
        Some(Value::Array(a)) if a.len() == layers => a
            .iter()
            .map(|v| v.as_bool().ok_or_else(|| format!("{arch}.{name} must hold booleans")))
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
        Some(_) => Err(format!("{arch}.{name} must have one entry per layer")),
    }
}

pub fn parse(meta: &Map<String, Value>, arch: &str, layers: usize) -> Result<HybridConfig, String> {
    let mut out = HybridConfig::default();
    if let Some(flags) = bools(meta, arch, "kda.layer_flags", layers)? {
        let need = |name: &str| uint(meta, arch, name).filter(|&n| n > 0).ok_or_else(|| format!("missing {arch}.{name}"));
        let kernel = need("ssm.conv_kernel")?;
        if kernel < 2 {
            return Err("the KDA short convolution needs at least two taps".into());
        }
        out.kda = Some(KdaConfig {
            layers: flags,
            conv_kernel: kernel,
            heads: need("kda.head_count")?,
            head_dim: need("kda.head_dim")?,
            lower_bound: value(meta, arch, "kda.gate_lower_bound").and_then(Value::as_f64).map(|v| v as f32),
        });
    }
    if arch == "zamba2" {
        let flags = bools(meta, arch, "shared_layer_flags", layers)?.ok_or("missing zamba2.shared_layer_flags")?;
        let blocks = uint(meta, arch, "shared_blocks").unwrap_or(1);
        if blocks == 0 || !flags.iter().any(|&b| b) {
            return Err("zamba2 needs at least one shared block call".into());
        }
        out.zamba = Some(ZambaConfig {
            blocks,
            rank: uint(meta, arch, "adapter_rank").unwrap_or(0),
            layers: flags,
            attention_adapters: value(meta, arch, "shared_attention_adapter").and_then(Value::as_bool).unwrap_or(false),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn meta(extra: Value) -> Map<String, Value> {
        let mut m = Map::new();
        m.insert("zamba2.shared_layer_flags".into(), json!([false, true, false, true, true]));
        m.insert("zamba2.shared_blocks".into(), json!(2));
        for (k, v) in extra.as_object().cloned().unwrap_or_default() {
            m.insert(k, v);
        }
        m
    }

    #[test]
    fn shared_blocks_cycle_over_the_calls() {
        let cfg = parse(&meta(json!({})), "zamba2", 5).unwrap().zamba.unwrap();
        let blocks: Vec<Option<usize>> = (0..5).map(|l| cfg.block_of(l)).collect();
        assert_eq!(blocks, [None, Some(0), None, Some(1), Some(0)]);
    }

    #[test]
    fn kda_layers_come_from_the_flags() {
        let mut m = Map::new();
        m.insert("kimi-linear.kda.layer_flags".into(), json!([true, true, true, false]));
        m.insert("kimi-linear.ssm.conv_kernel".into(), json!(4));
        m.insert("kimi-linear.kda.head_count".into(), json!(2));
        m.insert("kimi-linear.kda.head_dim".into(), json!(8));
        let kda = parse(&m, "kimi-linear", 4).unwrap().kda.unwrap();
        assert_eq!((kda.heads, kda.head_dim, kda.conv_kernel, kda.lower_bound), (2, 8, 4, None));
        assert!(parse(&m, "kimi-linear", 4).unwrap().stateful());
        m.insert("kimi-linear.ssm.conv_kernel".into(), json!(1));
        assert!(parse(&m, "kimi-linear", 4).is_err());
        m.remove("kimi-linear.kda.head_dim");
        m.insert("kimi-linear.ssm.conv_kernel".into(), json!(4));
        assert!(parse(&m, "kimi-linear", 4).is_err());
    }

    #[test]
    fn bad_flags_are_rejected() {
        assert!(parse(&meta(json!({"zamba2.shared_layer_flags": [true]})), "zamba2", 5).is_err());
        assert!(parse(&meta(json!({"zamba2.shared_blocks": 0})), "zamba2", 5).is_err());
        assert!(parse(&meta(json!({"zamba2.shared_layer_flags": [false, false, false, false, false]})), "zamba2", 5).is_err());
        assert!(parse(&Map::new(), "llama", 5).unwrap().zamba.is_none());
    }
}
