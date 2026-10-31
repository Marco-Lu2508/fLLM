//! Plans of the lightning indexer and the sparse attention of DeepSeek-V3.2 / GLM-5 (`Stream::dsa_select`, `Stream::dsa_attention`). Shaders:
//! `misc_dsa_keys`, `misc_dsa_scores`, `misc_dsa_topk`, `attn_sparse`; layouts in `shaders/vulkan/INTERFACE_OPS.md`.

use super::attention::KvLayout;
use super::*;
use crate::backend::{AttentionParams, DsaParams};

/// Multiply-adds of one group of the indexer scoring (`dsa_select` runs each group as its own submission). 2e9 is about 50 ms of a Radeon 840M (4 CUs,
/// measured 30 to 90 GMAC/s on this kernel); the driver resets a device whose job runs for seconds ("Vulkan device lost").
pub const GROUP_MACS: u64 = 2_000_000_000;
/// (row, head, position) triples of one sparse attention dispatch (each is its own submission when the batch needs more than one).
pub const ATTN_TRIPLES: usize = 1 << 19;
/// Widest query / key of the sparse attention shader (shared memory of the query).
pub const MAX_DK: usize = 512;

/// `dsa_select`: groups of plans, each to be run and submitted on its own: the key write, then for groups of rows the scoring and the top-k selection (with
/// the k-pool indexer: the pool scores, the top-k of pools into the first entries of the row of `sel`, and a last group that expands the chosen pools to positions). Slots: `store` (rw), `k`, `gate`, `ape`, `q`, `w`, `sel`; `slots`/`meta` are host
/// arrays, `scores` is scratch (id 0).
pub fn plan_dsa_select(p: &DsaParams, slots: &[usize], block_tables: &[i64], lengths: &[i64], store_slots: usize) -> Result<Vec<Vec<OpPlan>>, BackendError> {
    let rows = lengths.len();
    if rows == 0 || slots.len() != rows {
        return Err(invalid("indexer: no rows, or one slot per row expected"));
    }
    if p.heads == 0 || p.dim == 0 || p.topk == 0 || p.stride == 0 || p.block_size == 0 || block_tables.len() != rows * p.stride {
        return Err(invalid("indexer: empty shape or block tables of the wrong size"));
    }
    if slots.iter().any(|&s| s >= store_slots) {
        return Err(invalid("indexer: a slot lies outside the key store"));
    }
    let need = |n: i64| (n as usize).div_ceil(p.block_size);
    for (r, &n) in lengths.iter().enumerate() {
        if n <= 0 || need(n) > p.stride {
            return Err(invalid(format!("indexer row {r}: length {n} does not fit its block table")));
        }
        if block_tables[r * p.stride..][..need(n)].iter().any(|&b| b < 0 || (b as usize + 1) * p.block_size > store_slots) {
            return Err(invalid(format!("indexer row {r}: block outside the key store")));
        }
    }
    let max_len = lengths.iter().map(|&n| n as usize).max().unwrap_or(1);
    let mut meta = pack_i32(lengths)?;
    meta.extend_from_slice(&pack_i32(block_tables)?);
    let meta = Arc::new(meta);
    let pooled = p.kpool > 1;
    let width = p.store_width();
    if pooled && (p.dim > 128 || p.kpool > 16 || p.topk < p.kpool || p.topk / p.kpool > 1024) {
        return Err(BackendError::UnsupportedOperation("k-pool indexer: dim over 128, pools over 16 tokens, fewer kept tokens than one pool or over 1024 kept pools".into()));
    }
    let mut groups = vec![vec![plan(
        "misc_dsa_keys",
        Push::new().u("count", rows * p.dim).u("dim", p.dim).u("width", width).b("has_gate", pooled),
        groups_1d(rows * p.dim, 256),
        Slots::new().rw("store").read("k").read_opt("gate", pooled).host("slots", Arc::new(pack_u32(slots)?)),
    )?]];
    // pools: the topk kernel selects `topk / kpool` pools among `length / kpool`
    let kp = if pooled { p.topk / p.kpool } else { p.topk };
    let units = |n: usize| if pooled { n / p.kpool } else { n };
    let scored = units(max_len) > kp;
    let out_stride = if pooled { p.sel_width() } else { p.topk };
    // the scoring is cut into groups of rows by work: a group is its own submission and holds about GROUP_MACS multiply-adds, so that it ends within
    // a fraction of a second on a small GPU that other programs share (the driver resets the device when a job runs for seconds)
    let pair_macs = (p.heads * p.dim) as u64;
    let mut row0 = 0;
    while row0 < rows {
        let mut count = 0usize;
        let mut work = 0u64;
        while row0 + count < rows && count < MAX_GROUPS_PER_DIM as usize && (count == 0 || !scored || work < GROUP_MACS) {
            work += units(lengths[row0 + count] as usize) as u64 * pair_macs;
            count += 1;
        }
        let mut group = Vec::new();
        if scored && pooled {
            group.push(plan(
                "misc_dsa_pscores",
                Push::new()
                    .u("heads", p.heads)
                    .u("dim", p.dim)
                    .u("width", width)
                    .u("stride", p.stride)
                    .u("block_size", p.block_size)
                    .u("kpool", p.kpool)
                    .u("score_stride", max_len)
                    .u("row0", row0)
                    .u("rows_total", rows)
                    .f("scale", p.scale)
                    .f("head_scale", p.head_scale),
                [div_up(max_len / p.kpool, 64).min(MAX_GROUPS_PER_DIM as usize) as u32, count as u32, 1],
                Slots::new().read("q").read("w").read("store").read("ape").host("meta", meta.clone()).scratch("scores", 0, count * max_len * 4, Access::ReadWrite),
            )?);
        } else if scored {
            group.push(plan(
                if p.dim.is_multiple_of(4) && p.dim <= 128 && std::env::var_os("FLLM_DSA_SCALAR").is_none() { "misc_dsa_scores_v4" } else { "misc_dsa_scores" },
                Push::new()
                    .u("heads", p.heads)
                    .u("dim", p.dim)
                    .u("width", width)
                    .u("stride", p.stride)
                    .u("block_size", p.block_size)
                    .u("score_stride", max_len)
                    .u("row0", row0)
                    .u("rows_total", rows)
                    .f("scale", p.scale)
                    .f("head_scale", p.head_scale),
                [div_up(max_len, 128).min(MAX_GROUPS_PER_DIM as usize) as u32, count as u32, 1],
                Slots::new().read("q").read("w").read("store").host("meta", meta.clone()).scratch("scores", 0, count * max_len * 4, Access::ReadWrite),
            )?);
        }
        group.push(plan(
            "misc_dsa_topk",
            Push::new().u("topk", kp).u("score_stride", max_len).u("row0", row0).u("rows_total", rows).u("len_div", if pooled { p.kpool } else { 1 }).u("out_stride", out_stride),
            [count as u32, 1, 1],
            Slots::new().scratch("scores", 0, if scored { count * max_len * 4 } else { 16 }, Access::Read).host("meta", meta.clone()).write("sel"),
        )?);
        groups.push(group);
        row0 += count;
    }
    if pooled {
        groups.push(vec![plan(
            "misc_dsa_expand",
            Push::new().u("kp", kp).u("kpool", p.kpool).u("width_out", p.sel_width()).b("always_tail", p.always_tail).u("rows_total", rows),
            groups_rows(rows),
            Slots::new().host("meta", meta.clone()).rw("sel"),
        )?]);
    }
    Ok(groups)
}

/// `dsa_attention` over the f16 cache layout `kv`: one dispatch. Slots: `q`, `kpool`, `vpool`, `sel`, `out`; `meta` (the block tables) and `bias` (the sinks) are host arrays.
pub fn plan_dsa_attention(kv: &KvLayout, params: &AttentionParams, topk: usize, block_tables: &[i64]) -> Result<Vec<OpPlan>, BackendError> {
    let (heads, kv_heads, dk, dv) = (params.num_heads, params.num_kv_heads, params.head_dim, params.value_dim);
    if kv.kind != 1 {
        return Err(BackendError::UnsupportedOperation("sparse attention needs an f16 KV cache".into()));
    }
    if params.stride == 0 || block_tables.is_empty() || !block_tables.len().is_multiple_of(params.stride) {
        return Err(invalid("sparse attention: block tables are not whole rows"));
    }
    let rows = block_tables.len() / params.stride;
    if heads == 0 || kv_heads == 0 || !heads.is_multiple_of(kv_heads) || topk == 0 {
        return Err(invalid("sparse attention: heads, kv heads or top-k"));
    }
    if kv.kv_heads != kv_heads || kv.head_dim != dk || kv.value_dim != dv || kv.block_size != params.block_size {
        return Err(invalid("sparse attention parameters do not match the KV cache"));
    }
    if dk > MAX_DK || dv > 4 * 128 {
        return Err(BackendError::UnsupportedOperation(format!("sparse attention head widths {dk}/{dv} exceed {MAX_DK}")));
    }
    if heads > MAX_GROUPS_PER_DIM as usize {
        return Err(BackendError::UnsupportedOperation("sparse attention: too many rows or heads for one dispatch".into()));
    }
    let meta = Arc::new(pack_i32(block_tables)?);
    let mut sinks = vec![0.0f32; heads];
    if let Some(s) = &params.sinks {
        if s.len() < heads {
            return Err(invalid("sparse attention: fewer sink values than heads"));
        }
        sinks.copy_from_slice(&s[..heads]);
    }
    let bias = Arc::new(pack_f32(&sinks));
    // one dispatch covers at most 4M (row, head, position) triples, so that it ends within a fraction of a second on a small GPU
    let chunk = (ATTN_TRIPLES / (heads * topk)).clamp(1, rows);
    let mut plans = Vec::new();
    let mut row0 = 0;
    while row0 < rows {
        let count = chunk.min(rows - row0);
        plans.push(plan(
        "attn_sparse",
        Push::new()
            .u("topk", topk)
            .u("heads", heads)
            .u("kv_heads", kv_heads)
            .u("dk", dk)
            .u("dv", dv)
            .u("stride", params.stride)
            .u("block_size", params.block_size)
            .f("scale", params.scale)
            .u("k_base", kv.key_base)
            .u("v_base", kv.value_base)
            .b("flags", params.sinks.is_some())
            .u("row0", row0),
        [heads as u32, count as u32, 1],
        Slots::new().read("q").read("kpool").read("vpool").read("sel").host("meta", meta.clone()).host("bias", bias.clone()).write("out"),
    )?);
        row0 += count;
    }
    Ok(plans)
}
