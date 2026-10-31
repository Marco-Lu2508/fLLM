//! Plans of the block-sparse attention of MiniMax-M3 (`Stream::block_select`, `Stream::block_attention`). Shaders: `misc_dsa_keys` (the indexer keys go
//! to the same kind of per-slot store as in the lightning indexer), `misc_block_scores`, `misc_dsa_topk` (over the block scores), `attn_block_sparse`;
//! layouts in `shaders/vulkan/INTERFACE_OPS.md`.

use super::attention::KvLayout;
use super::dsa::{ATTN_TRIPLES, GROUP_MACS, MAX_DK};
use super::*;
use crate::backend::{AttentionParams, BlockSparseParams};

/// `block_select`: groups of plans, each to be run and submitted on its own: the key write, then for groups of rows the block scores and the top-k
/// selection. Slots: `store` (rw), `k`, `q`, `sel`; `slots`/`meta` are host arrays, `scores` is scratch (id 0).
pub fn plan_block_select(p: &BlockSparseParams, slots: &[usize], block_tables: &[i64], lengths: &[i64], store_slots: usize) -> Result<Vec<Vec<OpPlan>>, BackendError> {
    let rows = lengths.len();
    if rows == 0 || slots.len() != rows {
        return Err(invalid("block indexer: no rows, or one slot per row expected"));
    }
    if p.heads == 0 || p.dim == 0 || p.topk == 0 || p.block == 0 || p.stride == 0 || p.block_size == 0 || block_tables.len() != rows * p.stride {
        return Err(invalid("block indexer: empty shape or block tables of the wrong size"));
    }
    if slots.iter().any(|&s| s >= store_slots) {
        return Err(invalid("block indexer: a slot lies outside the key store"));
    }
    let need = |n: i64| (n as usize).div_ceil(p.block_size);
    for (r, &n) in lengths.iter().enumerate() {
        if n <= 0 || need(n) > p.stride {
            return Err(invalid(format!("block indexer row {r}: length {n} does not fit its block table")));
        }
        if block_tables[r * p.stride..][..need(n)].iter().any(|&b| b < 0 || (b as usize + 1) * p.block_size > store_slots) {
            return Err(invalid(format!("block indexer row {r}: block outside the key store")));
        }
    }
    let max_len = lengths.iter().map(|&n| n as usize).max().unwrap_or(1);
    let nb = max_len.div_ceil(p.block);
    if nb > MAX_GROUPS_PER_DIM as usize || p.heads > MAX_GROUPS_PER_DIM as usize {
        return Err(BackendError::UnsupportedOperation(format!("block indexer: {nb} blocks of {} keys exceed one dispatch dimension", p.block)));
    }
    let mut meta = pack_i32(lengths)?;
    meta.extend_from_slice(&pack_i32(block_tables)?);
    let meta = Arc::new(meta);
    // the blocks of every (row, head) row of scores: the top-k shader reads them as the row lengths
    let counts: Vec<i64> = lengths.iter().flat_map(|&n| std::iter::repeat_n((n + p.block as i64 - 1) / p.block as i64, p.heads)).collect();
    let counts = Arc::new(pack_i32(&counts)?);
    let mut groups = vec![vec![plan(
        "misc_dsa_keys",
        Push::new().u("count", rows * p.dim).u("dim", p.dim).u("width", p.dim).b("has_gate", false),
        groups_1d(rows * p.dim, 256),
        Slots::new().rw("store").read("k").read_opt("gate", false).host("slots", Arc::new(pack_u32(slots)?)),
    )?]];
    // the scoring is cut into groups of rows by work: a group is its own submission and holds about GROUP_MACS multiply-adds
    let pair_macs = (p.heads * p.dim) as u64;
    let max_rows = (MAX_GROUPS_PER_DIM as usize / p.heads).max(1);
    let mut row0 = 0;
    while row0 < rows {
        let mut count = 0usize;
        let mut work = 0u64;
        while row0 + count < rows && count < max_rows && (count == 0 || work < GROUP_MACS) {
            work += lengths[row0 + count] as u64 * pair_macs;
            count += 1;
        }
        let score_bytes = count * p.heads * nb * 4;
        groups.push(vec![
            plan(
                "misc_block_scores",
                Push::new()
                    .u("heads", p.heads)
                    .u("dim", p.dim)
                    .u("block", p.block)
                    .u("local", p.local)
                    .u("stride", p.stride)
                    .u("block_size", p.block_size)
                    .u("score_stride", nb)
                    .u("row0", row0)
                    .u("rows_total", rows),
                [nb as u32, p.heads as u32, count as u32],
                Slots::new().read("q").read("store").host("meta", meta.clone()).scratch("scores", 0, score_bytes, Access::ReadWrite),
            )?,
            plan(
                "misc_dsa_topk",
                Push::new().u("topk", p.topk).u("score_stride", nb).u("row0", row0 * p.heads).u("rows_total", rows * p.heads).u("len_div", 1).u("out_stride", p.topk),
                [(count * p.heads) as u32, 1, 1],
                Slots::new().scratch("scores", 0, score_bytes, Access::Read).host("meta", counts.clone()).write("sel"),
            )?,
        ]);
        row0 += count;
    }
    Ok(groups)
}

/// `block_attention` over the f16 cache layout `kv`: one dispatch per chunk of rows. Slots: `q`, `kpool`, `vpool`, `sel`, `out`; `meta` (lengths, block tables) is a host array.
pub fn plan_block_attention(kv: &KvLayout, params: &AttentionParams, p: &BlockSparseParams, block_tables: &[i64], lengths: &[i64]) -> Result<Vec<OpPlan>, BackendError> {
    let (heads, kv_heads, dk, dv) = (params.num_heads, params.num_kv_heads, params.head_dim, params.value_dim);
    if kv.kind != 1 {
        return Err(BackendError::UnsupportedOperation("block-sparse attention needs an f16 KV cache".into()));
    }
    if params.stride == 0 || block_tables.is_empty() || !block_tables.len().is_multiple_of(params.stride) {
        return Err(invalid("block-sparse attention: block tables are not whole rows"));
    }
    let rows = block_tables.len() / params.stride;
    if lengths.len() != rows || heads == 0 || kv_heads == 0 || !heads.is_multiple_of(kv_heads) || p.heads == 0 || !heads.is_multiple_of(p.heads) || p.topk == 0 || p.block == 0 {
        return Err(invalid("block-sparse attention: heads, kv heads, indexer heads, top-k or lengths"));
    }
    if kv.kv_heads != kv_heads || kv.head_dim != dk || kv.value_dim != dv || kv.block_size != params.block_size {
        return Err(invalid("block-sparse attention parameters do not match the KV cache"));
    }
    if dk > MAX_DK || dv > 4 * 128 {
        return Err(BackendError::UnsupportedOperation(format!("block-sparse attention head widths {dk}/{dv} exceed {MAX_DK}")));
    }
    if heads > MAX_GROUPS_PER_DIM as usize {
        return Err(BackendError::UnsupportedOperation("block-sparse attention: too many heads for one dispatch".into()));
    }
    let mut meta = pack_i32(lengths)?;
    meta.extend_from_slice(&pack_i32(block_tables)?);
    let meta = Arc::new(meta);
    // one dispatch covers at most ATTN_TRIPLES (row, head, position) triples, so that it ends within a fraction of a second on a small GPU
    let chunk = (ATTN_TRIPLES / (heads * p.topk * p.block)).clamp(1, rows).min(MAX_GROUPS_PER_DIM as usize);
    let mut plans = Vec::new();
    let mut row0 = 0;
    while row0 < rows {
        let count = chunk.min(rows - row0);
        plans.push(plan(
            "attn_block_sparse",
            Push::new()
                .u("topk", p.topk)
                .u("groups", p.heads)
                .u("block", p.block)
                .u("heads", heads)
                .u("kv_heads", kv_heads)
                .u("dk", dk)
                .u("dv", dv)
                .u("stride", params.stride)
                .u("block_size", params.block_size)
                .f("scale", params.scale)
                .u("k_base", kv.key_base)
                .u("v_base", kv.value_base)
                .u("row0", row0)
                .u("rows_total", rows),
            [heads as u32, count as u32, 1],
            Slots::new().read("q").read("kpool").read("vpool").read("sel").host("meta", meta.clone()).write("out"),
        )?);
        row0 += count;
    }
    Ok(plans)
}
