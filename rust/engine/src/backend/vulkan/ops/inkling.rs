//! Plans of the Inkling operations: the convolutions (`Stream::conv_resid`, shader `conv_resid`) and the relative-position attention
//! (`Stream::rel_attention`, shader `attn_rel`); layouts in `shaders/vulkan/INTERFACE_OPS.md`.

use super::attention::KvLayout;
use super::dsa::{ATTN_TRIPLES, MAX_DK};
use super::recurrent::RunSpec;
use super::*;
use crate::backend::RelAttnParams;

/// Whether the `conv_resid` shader takes this convolution (taps 2..8).
pub fn conv_resid_supported(channels: usize, taps: usize) -> bool {
    channels > 0 && (2..=8).contains(&taps)
}

/// `Stream::conv_resid` for one run (`conv_resid`, slots `x`, `conv`, `state`, `out`); `state_offset` is the byte offset of the layer's history.
pub fn plan_conv_resid(run_index: usize, run: &RunSpec, channels: usize, taps: usize, state_offset: usize) -> Result<OpPlan, BackendError> {
    if !conv_resid_supported(channels, taps) || !state_offset.is_multiple_of(4) || run.rows == 0 {
        return Err(invalid("convolution: unsupported taps, offset or empty run"));
    }
    let push = Push::new().u("channels", channels).u("taps", taps).u("row0", run.first_row).u("n", run.rows).b("reset", run.reset).u("state_off", state_offset / 4);
    plan("conv_resid", push, [div_up(channels, 128) as u32, 1, 1], Slots::new().read("x").read("conv").run_state("state", run_index).write("out"))
}

/// Widest relative-bias channel count of the `attn_rel` shader.
pub const MAX_REL: usize = 128;

/// Whether the `attn_rel` shader takes these shapes.
pub fn rel_attention_supported(p: &RelAttnParams) -> bool {
    p.heads > 0 && p.kv_heads > 0 && p.heads.is_multiple_of(p.kv_heads) && p.head_dim > 0 && p.head_dim <= MAX_DK && p.d_rel > 0 && p.d_rel <= MAX_REL && p.extent > 0
}

/// `rel_attention` over the f16 cache layout `kv`: one dispatch per chunk of rows. Slots: `q`, `rel`, `profile`, `kpool`, `vpool`, `out`; `meta` (lengths, block tables) is a host array.
pub fn plan_rel_attention(kv: &KvLayout, p: &RelAttnParams, block_tables: &[i64], lengths: &[i64]) -> Result<Vec<OpPlan>, BackendError> {
    if kv.kind != 1 {
        return Err(BackendError::UnsupportedOperation("relative-position attention needs an f16 KV cache".into()));
    }
    if !rel_attention_supported(p) {
        return Err(BackendError::UnsupportedOperation("relative-position attention: head width or relative channels beyond the shader".into()));
    }
    if p.stride == 0 || block_tables.is_empty() || !block_tables.len().is_multiple_of(p.stride) {
        return Err(invalid("relative-position attention: block tables are not whole rows"));
    }
    let rows = block_tables.len() / p.stride;
    if lengths.len() != rows || lengths.iter().any(|&n| n <= 0) || p.heads > MAX_GROUPS_PER_DIM as usize {
        return Err(invalid("relative-position attention: lengths, or too many heads for one dispatch"));
    }
    if kv.kv_heads != p.kv_heads || kv.head_dim != p.head_dim || kv.value_dim != p.head_dim || kv.block_size != p.block_size {
        return Err(invalid("relative-position attention parameters do not match the KV cache"));
    }
    let mut meta = pack_i32(lengths)?;
    meta.extend_from_slice(&pack_i32(block_tables)?);
    let meta = Arc::new(meta);
    // one dispatch covers at most ATTN_TRIPLES (row, head, visible key) triples, so that it ends within a fraction of a second on a small GPU
    let visible = |n: i64| if p.window > 0 { (n as usize).min(p.window) } else { n as usize };
    let mut plans = Vec::new();
    let mut row0 = 0;
    while row0 < rows {
        let mut count = 0usize;
        let mut work = 0usize;
        while row0 + count < rows && count < MAX_GROUPS_PER_DIM as usize && (count == 0 || work + p.heads * visible(lengths[row0 + count]) <= ATTN_TRIPLES) {
            work += p.heads * visible(lengths[row0 + count]);
            count += 1;
        }
        plans.push(plan(
            "attn_rel",
            Push::new()
                .u("heads", p.heads)
                .u("kv_heads", p.kv_heads)
                .u("hd", p.head_dim)
                .u("d_rel", p.d_rel)
                .u("extent", p.extent)
                .u("window", p.window)
                .u("log_floor", p.log_floor)
                .f("log_alpha", p.log_alpha)
                .f("scale", p.scale)
                .u("stride", p.stride)
                .u("block_size", p.block_size)
                .u("k_base", kv.key_base)
                .u("v_base", kv.value_base)
                .u("row0", row0)
                .u("rows_total", rows),
            [p.heads as u32, count as u32, 1],
            Slots::new().read("q").read("rel").read("profile").read("kpool").read("vpool").host("meta", meta.clone()).write("out"),
        )?);
        row0 += count;
    }
    Ok(plans)
}
