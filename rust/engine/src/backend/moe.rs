//! Device mixture-of-experts: the parameters of the routing, sorting and grouped-GEMM operations of `Stream` (`moe_route_ex`, `moe_align`,
//! `moe_grouped_linear`, `moe_combine_ex`, `moe_gate_scale`), the sizes the sorted buffers need, and reference implementations of the
//! routing and of the sort that the self test and the unit tests compare the device with.
//!
//! The device operations never read anything back: the sorted buffers have an upper bound that depends only on the number of
//! token-expert pairs, the number of experts and the tile, so a decode step with these operations can be recorded in a CUDA graph.

use super::{DeviceBuffer, QuantType};

/// Gating and selection of one mixture-of-experts layer (the device twin of `MoeConfig` plus the layer's selection bias).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoeRoute {
    /// Scores are the sigmoid of the logits instead of their softmax.
    pub sigmoid: bool,
    /// Divide the selected scores by their sum.
    pub normalize: bool,
    /// Multiplier of the final weights (`routed_scaling_factor`).
    pub scale: f32,
    /// Added to the sum the selected scores are divided by (LFM2-MoE: 1e-6).
    pub eps: f32,
    /// Expert groups of the group-limited selection (`n_group`); 0 or 1 selects from all experts.
    pub groups: usize,
    /// Groups that stay eligible (`topk_group`).
    pub groups_used: usize,
    /// Rank groups by the sum of their two best selection values instead of the best one (DeepSeek-V3 with the correction bias).
    pub group_top2: bool,
}

impl MoeRoute {
    /// Softmax routing of the classic top-k models.
    pub fn softmax(normalize: bool, scale: f32) -> Self {
        MoeRoute { sigmoid: false, normalize, scale, eps: 0.0, groups: 0, groups_used: 0, group_top2: false }
    }
}

/// Expert-major order of the token-expert pairs of one layer. Pair `p = token * k + j` is the `j`-th expert of `token`.
pub struct MoeSort {
    /// Pair numbers grouped by expert, every expert padded to a multiple of `block` with the value `pairs` (i32).
    pub sorted: DeviceBuffer,
    /// Local expert of every block of `block` entries of `sorted`, -1 for unused blocks (i32).
    pub block_expert: DeviceBuffer,
    /// Length of the used part of `sorted` (one i32).
    pub post: DeviceBuffer,
    pub pairs: usize,
    pub block: usize,
    pub max_blocks: usize,
}

/// Bytes of the three buffers of a [`MoeSort`] for `pairs` pairs, `local` experts and tiles of `block` pairs: (sorted, block_expert, post).
pub fn moe_sort_bytes(pairs: usize, local: usize, block: usize) -> (usize, usize, usize) {
    let padded = moe_max_padded(pairs, local, block);
    (padded * 4, padded / block.max(1) * 4, 4)
}

/// Upper bound of the padded length: every expert may waste up to `block - 1` entries.
pub fn moe_max_padded(pairs: usize, local: usize, block: usize) -> usize {
    (pairs + local * block.saturating_sub(1)).next_multiple_of(block.max(1))
}

/// Pairs per tile of the grouped GEMM: the narrow tile when the experts get few tokens each (decode batches, small prompts), the wide one
/// otherwise. The kernels exist for 16 and 64.
pub fn moe_block_for(pairs: usize, local: usize) -> usize {
    if pairs <= local.max(1) * 24 {
        16
    } else {
        64
    }
}

/// `gate/up` of a grouped product: the activation applied in the epilogue.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoeGlu {
    /// 0 SwiGLU, 1 GeGLU (tanh), 2 gpt-oss clamped SwiGLU.
    pub mode: i32,
    pub alpha: f32,
    pub limit: f32,
}

/// One grouped product `out[p] = W[expert(p)] x[p / x_div]` over all experts (`Stream::moe_grouped_linear`). With `table_up` the call is the
/// fused gate/up projection: `out[p] = glu(Wg x, Wu x)` (`rows` is the intermediate width).
pub struct MoeGrouped<'a> {
    pub out: &'a mut DeviceBuffer,
    pub x: &'a DeviceBuffer,
    /// Device addresses of the experts' weights (`upload_pointer_table`); entry `i` is local expert `i`.
    pub table: &'a DeviceBuffer,
    pub table_up: Option<&'a DeviceBuffer>,
    pub bias: Option<&'a DeviceBuffer>,
    pub bias_up: Option<&'a DeviceBuffer>,
    /// Routing weight of every pair, multiplied into the activations (Llama 4 scales the expert input).
    pub input_weights: Option<&'a DeviceBuffer>,
    pub sort: &'a MoeSort,
    pub quant_type: QuantType,
    pub rows: usize,
    pub cols: usize,
    pub x_div: usize,
    pub glu: Option<MoeGlu>,
}

/// Scores of one row of logits and the selection values after the group-limited masking (f64): what [`route_reference`] ranks.
pub fn route_scores(logits: &[f32], bias: Option<&[f32]>, route: &MoeRoute) -> (Vec<f64>, Vec<f64>) {
    let experts = logits.len();
    let scores: Vec<f64> = if route.sigmoid {
        logits.iter().map(|&x| 1.0 / (1.0 + (-f64::from(x)).exp())).collect()
    } else {
        let max = logits.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
        let exps: Vec<f64> = logits.iter().map(|&v| f64::from(v - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|e| e / sum).collect()
    };
    let mut select: Vec<f64> = scores.iter().enumerate().map(|(e, s)| s + bias.map_or(0.0, |b| f64::from(b[e]))).collect();
    if route.groups > 1 {
        let per = experts / route.groups;
        let mut ranked: Vec<(usize, f64)> = (0..route.groups)
            .map(|g| {
                let mut group = select[g * per..(g + 1) * per].to_vec();
                group.sort_by(|a, b| b.total_cmp(a));
                (g, if route.group_top2 && per >= 2 { group[0] + group[1] } else { group[0] })
            })
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        for &(g, _) in &ranked[route.groups_used.min(route.groups)..] {
            select[g * per..(g + 1) * per].fill(f64::NEG_INFINITY);
        }
    }
    (scores, select)
}

/// Reference routing in f64 for one row of logits: the chosen experts in selection order and their weights. Same rules as the host router of
/// the decoder (scores, selection bias, group-limited selection with ties to the lower index, renormalisation, scale).
pub fn route_reference(logits: &[f32], bias: Option<&[f32]>, k: usize, route: &MoeRoute) -> Vec<(usize, f64)> {
    let (scores, select) = route_scores(logits, bias, route);
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| select[b].total_cmp(&select[a]).then(a.cmp(&b)));
    let chosen = &order[..k];
    let total: f64 = chosen.iter().map(|&e| scores[e]).sum::<f64>() + f64::from(route.eps);
    chosen
        .iter()
        .map(|&e| (e, (if route.normalize { scores[e] / total } else { scores[e] }) * f64::from(route.scale)))
        .collect()
}

/// Reference of the sort: for every local expert the set of pairs routed to it (ascending), after dropping the experts outside
/// `[offset, offset + local)`.
pub fn align_reference(ids: &[i32], offset: usize, local: usize) -> Vec<Vec<usize>> {
    let mut per = vec![Vec::new(); local];
    for (p, &id) in ids.iter().enumerate() {
        let e = i64::from(id) - offset as i64;
        if e >= 0 && (e as usize) < local {
            per[e as usize].push(p);
        }
    }
    per
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sort_sizes_bound_the_padding() {
        // 10 pairs over 4 experts in tiles of 16: every expert can need one tile
        assert_eq!(moe_max_padded(10, 4, 16), 80);
        let (sorted, blocks, post) = moe_sort_bytes(2048 * 8, 128, 64);
        assert_eq!(sorted, moe_max_padded(2048 * 8, 128, 64) * 4);
        assert_eq!(blocks, moe_max_padded(2048 * 8, 128, 64) / 64 * 4);
        assert_eq!(post, 4);
        assert_eq!(moe_max_padded(0, 8, 16), 128);
    }

    #[test]
    fn tile_follows_the_tokens_per_expert() {
        assert_eq!(moe_block_for(8, 64), 16);
        assert_eq!(moe_block_for(64 * 8, 128), 16);
        assert_eq!(moe_block_for(8192 * 8, 128), 64);
        assert_eq!(moe_block_for(100, 0), 64);
    }

    #[test]
    fn softmax_reference_matches_the_textbook() {
        let logits = [1.0f32, 3.0, 2.0, 0.0];
        let r = route_reference(&logits, None, 2, &MoeRoute::softmax(true, 1.0));
        assert_eq!(r.iter().map(|x| x.0).collect::<Vec<_>>(), vec![1, 2]);
        let (e1, e2) = (3.0f64.exp(), 2.0f64.exp());
        assert!((r[0].1 - e1 / (e1 + e2)).abs() < 1e-12);
        assert!((r[0].1 + r[1].1 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn sigmoid_bias_and_groups_select_like_deepseek() {
        // 8 experts in 4 groups of 2; group 1 holds the best single score but the bias moves the pick
        let logits = [0.0f32, 0.0, 4.0, -4.0, 1.0, 1.0, 0.5, 0.5];
        let bias = [0.0f32, 0.0, 0.0, 0.0, 0.3, 0.3, 0.0, 0.0];
        let route = MoeRoute { sigmoid: true, normalize: true, scale: 2.5, eps: 0.0, groups: 4, groups_used: 2, group_top2: true };
        let r = route_reference(&logits, Some(&bias), 3, &route);
        // group values (top two selection values): g0 1.0, g1 sigmoid(4)+sigmoid(-4), g2 2*sigmoid(1)+0.6, g3 2*sigmoid(0.5)
        let sig = |x: f64| 1.0 / (1.0 + (-x).exp());
        let g: Vec<f64> = vec![1.0, sig(4.0) + sig(-4.0), 2.0 * sig(1.0) + 0.6, 2.0 * sig(0.5)];
        let mut order: Vec<usize> = (0..4).collect();
        order.sort_by(|&a, &b| g[b].total_cmp(&g[a]).then(a.cmp(&b)));
        let allowed = &order[..2];
        for (e, _) in &r {
            assert!(allowed.contains(&(e / 2)), "expert {e} is outside the best two groups {allowed:?}");
        }
        assert!((r.iter().map(|x| x.1).sum::<f64>() - 2.5).abs() < 1e-9, "normalised weights times the scale");
    }

    #[test]
    fn align_reference_drops_foreign_experts() {
        let per = align_reference(&[3, 1, 3, 0, 2, 1], 1, 2);
        assert_eq!(per, vec![vec![1, 5], vec![4]]);
    }
}
