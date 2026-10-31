//! Plans of the DeepSeek-V4 attention operations (`Stream::v4_rope`, `v4_compress`, `v4_attention`, `v4_ring_write`). Shaders: `v4_rope`, `v4_compress`, `v4_attn`,
//! `v4_ring`; layouts in `shaders/vulkan/INTERFACE_OPS.md`.

use super::*;
use crate::backend::{V4Pool, V4Shape};

/// Widest key of the attention shader (shared memory of the query).
pub const MAX_HD: usize = 512;
/// (row, head, key) triples of one attention dispatch (a bigger batch is split into dispatches that are submitted one after the other): about 130 M
/// multiply-adds at the widest key (a dot product and an accumulation of 512 values per triple), a few tens of milliseconds on a Radeon 840M, far from the
/// time after which the driver resets a device.
pub const ATTN_TRIPLES: usize = 1 << 17;
/// Tokens one compressor dispatch walks (it takes one serial step per token).
pub const COMPRESS_PIECE: usize = 512;

/// A run of the batch as the plans need it: its first row and length, and the position of its first row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct V4RunSpec {
    pub first_row: usize,
    pub rows: usize,
    pub position: usize,
}

/// Whether the shaders take a layer of this shape.
pub fn v4_supported(s: &V4Shape) -> bool {
    s.head_dim > 0
        && s.head_dim <= MAX_HD
        && s.rope > 0
        && s.rope <= s.head_dim
        && s.rope.is_multiple_of(2)
        && s.window > 0
        && s.heads > 0
        && s.o_groups > 0
        && s.heads.is_multiple_of(s.o_groups)
        && s.heads <= MAX_GROUPS_PER_DIM as usize
}

/// `v4_rope`: `rows * groups * rope / 2` pairs. Slots: `x` (rw), `cs`.
pub fn plan_v4_rope(rows: usize, groups: usize, dim: usize, rope: usize, sign: f32) -> Result<OpPlan, BackendError> {
    if rows == 0 || groups == 0 || rope == 0 || rope > dim || !rope.is_multiple_of(2) {
        return Err(invalid("DeepSeek-V4 rotation: empty tensor, or an odd or too wide rotated part"));
    }
    let pairs = rows * groups * (rope / 2);
    plan("v4_rope", Push::new().u("pairs", pairs).u("groups", groups).u("dim", dim).u("rope", rope).f("sign", sign), groups_1d(pairs, 256), Slots::new().rw("x").read("cs"))
}

/// `v4_compress`: for every run pieces of at most [`COMPRESS_PIECE`] tokens, one workgroup each, in order. Slots: `kv`, `gate`, `pos`, `norm`, `cs`, `state` (run `i`).
#[allow(clippy::too_many_arguments)]
pub fn plan_v4_compress(which: V4Pool, s: &V4Shape, runs: &[V4RunSpec], state_offset: usize, eps: f32) -> Result<Vec<OpPlan>, BackendError> {
    if s.ratio == 0 || !state_offset.is_multiple_of(4) {
        return Err(invalid("DeepSeek-V4 compressor: no ratio, or an unaligned state offset"));
    }
    let (dim, coff) = match which {
        V4Pool::Attention => (s.head_dim, s.coff()),
        V4Pool::Index => (s.index.map_or(0, |i| i.dim), 2),
    };
    if dim == 0 || dim > MAX_HD {
        return Err(invalid("DeepSeek-V4 compressor: no entry width"));
    }
    let lay = s.layout();
    let (buf_kv, buf_gate, ov_kv, ov_gate, entries) = match which {
        V4Pool::Attention => (lay.buf_kv, lay.buf_gate, lay.ov_kv, lay.ov_gate, lay.entries),
        V4Pool::Index => (lay.index_buf_kv, lay.index_buf_gate, lay.index_ov_kv, lay.index_ov_gate, lay.index_entries),
    };
    let base = state_offset / 4;
    let mut plans = Vec::new();
    for (i, run) in runs.iter().enumerate() {
        let mut done = 0;
        while done < run.rows {
            let n = COMPRESS_PIECE.min(run.rows - done);
            plans.push(plan(
                "v4_compress",
                Push::new()
                    .u("row0", run.first_row + done)
                    .u("n", n)
                    .u("position0", run.position + done)
                    .u("dim", dim)
                    .u("coff", coff)
                    .u("ratio", s.ratio)
                    .u("rope", s.rope)
                    .f("eps", eps)
                    .u("off_buf_kv", base + buf_kv)
                    .u("off_buf_gate", base + buf_gate)
                    .u("off_ov_kv", base + ov_kv)
                    .u("off_ov_gate", base + ov_gate)
                    .u("off_entries", base + entries),
                [1, 1, 1],
                Slots::new().read("kv").read("gate").read("pos").read("norm").read("cs").run_state("state", i),
            )?);
            done += n;
        }
    }
    Ok(plans)
}

/// `v4_attention`: the rows of every run split into dispatches of at most [`ATTN_TRIPLES`] (row, head, key) triples; each group of the result is to be
/// submitted on its own. Slots: `q`, `kv`, `sinks`, `sel` (a dummy without), `cs`, `state` (run `i`), `out`.
pub fn plan_v4_attention(s: &V4Shape, runs: &[V4RunSpec], state_offset: usize, has_sel: bool, rows_total: usize) -> Result<Vec<Vec<OpPlan>>, BackendError> {
    if !v4_supported(s) || !state_offset.is_multiple_of(4) {
        return Err(BackendError::UnsupportedOperation("DeepSeek-V4 attention for this shape".into()));
    }
    let lay = s.layout();
    let base = state_offset / 4;
    let topk = s.index.map_or(0, |i| i.topk);
    if has_sel && topk == 0 {
        return Err(invalid("DeepSeek-V4 attention: a selection without an indexer"));
    }
    let mut groups: Vec<Vec<OpPlan>> = vec![Vec::new()];
    let mut work = 0usize;
    for (i, run) in runs.iter().enumerate() {
        let last = run.position + run.rows;
        let visible = last.checked_div(s.ratio).unwrap_or(0);
        let entries = if has_sel { topk.min(visible) } else { visible };
        let keys = s.window + entries;
        let per_row = s.heads * keys;
        let chunk = (ATTN_TRIPLES / per_row.max(1)).clamp(1, run.rows.max(1));
        let mut done = 0;
        while done < run.rows {
            let count = chunk.min(run.rows - done);
            if work > 0 && work + count * per_row > ATTN_TRIPLES {
                groups.push(Vec::new());
                work = 0;
            }
            work += count * per_row;
            if let Some(group) = groups.last_mut() {
                group.push(plan(
                    "v4_attn",
                    Push::new()
                        .u("row0", run.first_row + done)
                        .u("position0", run.position + done)
                        .u("run_row0", run.first_row)
                        .u("run_position0", run.position)
                        .u("heads", s.heads)
                        .u("hd", s.head_dim)
                        .u("rope", s.rope)
                        .u("window", s.window)
                        .u("ratio", s.ratio)
                        .u("topk", topk)
                        .u("hpg", s.heads / s.o_groups)
                        .u("rows_total", rows_total)
                        .u("off_ring", base + lay.ring)
                        .u("off_entries", base + lay.entries)
                        .f("scale", (s.head_dim as f32).powf(-0.5))
                        .b("has_sel", has_sel),
                    [s.heads as u32, count as u32, 1],
                    Slots::new().read("q").read("kv").read("sinks").read_opt("sel", has_sel).read("cs").run_state("state", i).write("out"),
                )?);
            }
            done += count;
        }
    }
    groups.retain(|g| !g.is_empty());
    Ok(groups)
}

/// `v4_ring`: one dispatch per run. Slots: `kv`, `state` (run `i`).
pub fn plan_v4_ring(s: &V4Shape, runs: &[V4RunSpec], state_offset: usize) -> Result<Vec<OpPlan>, BackendError> {
    if !state_offset.is_multiple_of(4) {
        return Err(invalid("DeepSeek-V4 ring: unaligned state offset"));
    }
    let lay = s.layout();
    let base = state_offset / 4;
    let mut plans = Vec::new();
    for (i, run) in runs.iter().enumerate().filter(|(_, r)| r.rows > 0) {
        let total = run.rows.min(s.window) * s.head_dim;
        plans.push(plan(
            "v4_ring",
            Push::new().u("row0", run.first_row).u("n", run.rows).u("position0", run.position).u("hd", s.head_dim).u("window", s.window).u("off_ring", base + lay.ring),
            groups_1d(total, 256),
            Slots::new().read("kv").run_state("state", i),
        )?);
    }
    Ok(plans)
}
