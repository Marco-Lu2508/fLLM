//! Plans of the lightning indexer and the sparse attention of DeepSeek-V3.2 / GLM-5 (`Stream::dsa_select`, `Stream::dsa_attention`). Shaders:
//! `misc_dsa_keys`, `misc_dsa_scores`, `misc_dsa_topk`, `attn_sparse`; layouts in `shaders/vulkan/INTERFACE_OPS.md`.

use super::attention::KvLayout;
use super::*;
use crate::backend::{AttentionParams, DsaParams};

/// Scores of one chunk of rows stay below this many floats (the chunk loop of the plan keeps the scratch buffer at 128 MiB).
pub const SCORE_ELEMS: usize = 32 << 20;
/// Widest query / key of the sparse attention shader (shared memory of the query).
pub const MAX_DK: usize = 512;

/// `dsa_select`: the key write, then for chunks of rows the scoring and the top-k selection. Slots: `store` (rw), `k`, `q`, `w`, `sel`; `slots`/`meta` are host
/// arrays, `scores` is scratch (id 0).
pub fn plan_dsa_select(p: &DsaParams, slots: &[usize], block_tables: &[i64], lengths: &[i64], store_slots: usize) -> Result<Vec<OpPlan>, BackendError> {
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
    let mut plans = vec![plan(
        "misc_dsa_keys",
        Push::new().u("count", rows * p.dim).u("dim", p.dim),
        groups_1d(rows * p.dim, 256),
        Slots::new().rw("store").read("k").host("slots", Arc::new(pack_u32(slots)?)),
    )?];
    let chunk = if max_len > p.topk { (SCORE_ELEMS / max_len).clamp(1, rows).min(MAX_GROUPS_PER_DIM as usize) } else { rows.min(MAX_GROUPS_PER_DIM as usize) };
    let mut row0 = 0;
    while row0 < rows {
        let count = chunk.min(rows - row0);
        if max_len > p.topk {
            plans.push(plan(
                "misc_dsa_scores",
                Push::new()
                    .u("heads", p.heads)
                    .u("dim", p.dim)
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
        plans.push(plan(
            "misc_dsa_topk",
            Push::new().u("topk", p.topk).u("score_stride", max_len).u("row0", row0).u("rows_total", rows),
            [count as u32, 1, 1],
            Slots::new().scratch("scores", 0, if max_len > p.topk { count * max_len * 4 } else { 16 }, Access::Read).host("meta", meta.clone()).write("sel"),
        )?);
        row0 += count;
    }
    Ok(plans)
}

/// `dsa_attention` over the f16 cache layout `kv`: one dispatch. Slots: `q`, `kpool`, `vpool`, `sel`, `out`; `meta` (the block tables) and `bias` (the sinks) are host arrays.
pub fn plan_dsa_attention(kv: &KvLayout, params: &AttentionParams, topk: usize, block_tables: &[i64]) -> Result<OpPlan, BackendError> {
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
    if rows > MAX_GROUPS_PER_DIM as usize || heads > MAX_GROUPS_PER_DIM as usize {
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
    plan(
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
            .b("flags", params.sinks.is_some()),
        [heads as u32, rows as u32, 1],
        Slots::new().read("q").read("kpool").read("vpool").read("sel").host("meta", meta).host("bias", bias).write("out"),
    )
}
