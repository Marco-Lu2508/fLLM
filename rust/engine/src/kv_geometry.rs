//! Layout of the KV cache pools when layers differ in KV heads or head width, own no cache (they read the
//! cache of an earlier layer) or both.
//!
//! The key pool and the value pool are each one `u16` array. Layer `l` owns the region
//! `offsets[l] .. offsets[l] + words[l]`; a layer that shares the cache of another layer owns an empty
//! region and resolves to the owner with `ModelConfig::layer_kv_source`. A model with one geometry for all
//! layers gets `offsets[l] = l * words`, the layout every backend used before per-layer geometry existed.
//!
//! All layers address the pool with the same block table: block `b` holds the same token positions in every
//! layer, so the scheduler's block accounting does not depend on the geometry. Sliding-window layers keep
//! their blocks for the whole sequence in this layout (see docs/changes/gemma4.md for the consequences).
use crate::{config::ModelConfig, kv_format::KvFormat};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvPoolLayout {
    /// First 16-bit word of each layer's key region.
    pub key_offsets: Vec<usize>,
    /// First 16-bit word of each layer's value region.
    pub value_offsets: Vec<usize>,
    /// Words of each layer's key region.
    pub key_words: Vec<usize>,
    pub value_words: Vec<usize>,
    pub key_total: usize,
    pub value_total: usize,
}

impl KvPoolLayout {
    /// Layout for `slots` cache slots (`pool_blocks * block_size`).
    pub fn new(cfg: &ModelConfig, key: KvFormat, value: KvFormat, slots: usize) -> Self {
        let layers = cfg.layers;
        let mut layout = Self {
            key_offsets: Vec::with_capacity(layers),
            value_offsets: Vec::with_capacity(layers),
            key_words: Vec::with_capacity(layers),
            value_words: Vec::with_capacity(layers),
            key_total: 0,
            value_total: 0,
        };
        for l in 0..layers {
            let (k, v) = if cfg.layer_kv_source(l).is_some() {
                (0, 0)
            } else {
                let heads = cfg.layer_kv_heads(l);
                (
                    key.layer_words(slots, heads, cfg.layer_head_dim(l)),
                    value.layer_words(slots, heads, cfg.layer_value_dim(l)),
                )
            };
            layout.key_offsets.push(layout.key_total);
            layout.value_offsets.push(layout.value_total);
            layout.key_words.push(k);
            layout.value_words.push(v);
            layout.key_total += k;
            layout.value_total += v;
        }
        layout
    }

    /// The layout of the layers `range` alone, numbered from 0: what a pipeline stage allocates for its own layers.
    pub fn stage(&self, range: std::ops::Range<usize>) -> Self {
        let (key_base, value_base) = (self.key_offsets[range.start], self.value_offsets[range.start]);
        let key_offsets: Vec<usize> = self.key_offsets[range.clone()].iter().map(|o| o - key_base).collect();
        let value_offsets: Vec<usize> = self.value_offsets[range.clone()].iter().map(|o| o - value_base).collect();
        let key_words = self.key_words[range.clone()].to_vec();
        let value_words = self.value_words[range].to_vec();
        Self {
            key_total: key_words.iter().sum(),
            value_total: value_words.iter().sum(),
            key_offsets,
            value_offsets,
            key_words,
            value_words,
        }
    }

    /// The layer whose region `layer` reads.
    pub fn owner(cfg: &ModelConfig, layer: usize) -> usize {
        cfg.layer_kv_source(layer).unwrap_or(layer)
    }

    /// Classify model layers into distinct geometry groups (vLLM hybrid KV manager style).
    pub fn geometry_groups(cfg: &ModelConfig) -> Vec<GeometryGroup> {
        let mut groups: Vec<GeometryGroup> = Vec::new();
        for l in 0..cfg.layers {
            if cfg.layer_kv_source(l).is_some() {
                continue; // Shared KV layer borrows another layer's cache
            }
            let kv_heads = cfg.layer_kv_heads(l);
            let head_dim = cfg.layer_head_dim(l);
            let value_dim = cfg.layer_value_dim(l);
            let sliding_window = cfg.layer_window(l);
            if let Some(g) = groups.iter_mut().find(|g| {
                g.kv_heads == kv_heads
                    && g.head_dim == head_dim
                    && g.value_dim == value_dim
                    && g.sliding_window == sliding_window
            }) {
                g.layers.push(l);
            } else {
                let id = groups.len();
                groups.push(GeometryGroup {
                    group_id: id,
                    kv_heads,
                    head_dim,
                    value_dim,
                    sliding_window,
                    layers: vec![l],
                });
            }
        }
        groups
    }
}

/// A group of layers sharing identical KV cache geometry and window characteristics.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GeometryGroup {
    pub group_id: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub value_dim: usize,
    pub sliding_window: Option<usize>,
    pub layers: Vec<usize>,
}

/// Bytes of cache one token takes in layer `layer` (0 for a layer that reads the cache of an earlier one).
pub fn bytes_per_token_layer(cfg: &ModelConfig, layer: usize, key: KvFormat, value: KvFormat) -> u64 {
    if cfg.layer_kv_source(layer).is_some() {
        return 0;
    }
    let heads = cfg.layer_kv_heads(layer);
    key.bytes_per_slot(heads, cfg.layer_head_dim(layer)) + value.bytes_per_slot(heads, cfg.layer_value_dim(layer))
}

/// Bytes of cache one token takes in all layers together (what the memory planner budgets per slot).
pub fn bytes_per_token(cfg: &ModelConfig, key: KvFormat, value: KvFormat) -> u64 {
    (0..cfg.layers).map(|l| bytes_per_token_layer(cfg, l, key, value)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LayerGeom;

    /// Build a config through the public parser would need a GGUF; the tests patch a minimal one instead.
    fn config(geoms: Option<Vec<LayerGeom>>) -> ModelConfig {
        let mut cfg = crate::config::tests_support::minimal();
        cfg.layers = 4;
        cfg.layer_geoms = geoms;
        cfg
    }

    fn geom(kv_heads: usize, head_dim: usize, source: Option<usize>) -> LayerGeom {
        LayerGeom { heads: 4, kv_heads, head_dim, value_dim: head_dim, kv_source: source, ff: 8 }
    }

    #[test]
    fn uniform_models_keep_the_fixed_stride() {
        let cfg = config(None);
        let layout = KvPoolLayout::new(&cfg, KvFormat::F16, KvFormat::F16, 32);
        let words = KvFormat::F16.layer_words(32, cfg.kv_heads, cfg.head_dim);
        assert_eq!(layout.key_offsets, vec![0, words, 2 * words, 3 * words]);
        assert_eq!(layout.key_total, 4 * words);
        assert_eq!(layout.value_total, 4 * words);
    }

    #[test]
    fn regions_follow_each_layers_geometry_and_shared_layers_own_nothing() {
        let cfg = config(Some(vec![geom(2, 16, None), geom(1, 32, None), geom(2, 16, Some(0)), geom(1, 32, Some(1))]));
        let layout = KvPoolLayout::new(&cfg, KvFormat::F16, KvFormat::F16, 16);
        assert_eq!(layout.key_words, vec![16 * 2 * 16, 16 * 32, 0, 0]);
        assert_eq!(layout.key_offsets, vec![0, 512, 1024, 1024]);
        assert_eq!(layout.key_total, 1024);
        assert_eq!(KvPoolLayout::owner(&cfg, 3), 1);
        assert_eq!(KvPoolLayout::owner(&cfg, 1), 1);
        // 2 * (2 * 16 + 1 * 32) 16-bit values per token
        assert_eq!(bytes_per_token(&cfg, KvFormat::F16, KvFormat::F16), 2 * 2 * (32 + 32));
    }

    #[test]
    fn geometry_groups_partitions_layers_by_kv_shape_and_window() {
        let cfg = config(Some(vec![geom(2, 16, None), geom(1, 32, None), geom(2, 16, Some(0)), geom(1, 32, Some(1))]));
        let groups = KvPoolLayout::geometry_groups(&cfg);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].layers, vec![0]);
        assert_eq!(groups[1].layers, vec![1]);
    }
}
