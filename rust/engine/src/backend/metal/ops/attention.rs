//! Plans of the KV-cache write and of paged attention (package p06b). Kernels: `kv_write.metal` (`fllm_kv_write_f16`, `fllm_kv_write_q8`),
//! `attn_decode.metal` (`fllm_attn_decode_{f16,q8}_e<E>`, `fllm_attn_merge_f32`) and `attn_prefill.metal` (`fllm_attn_prefill_{f16,q8}_k<DKF>v<DVF>`);
//! layouts and grids in `metal/INTERFACE_OPS.md`.

use super::*;
use crate::backend::{kv_layer_words, kv_quant_group, AttentionParams, DeviceKvCache};

/// Widest key or value head the attention kernels take (`32 * 18`, the largest decode instantiation).
pub const MAX_HEAD_WIDTH: usize = 576;
/// Most splits of the key range (`fllm_attn_merge_f32` keeps the weights of the splits in threadgroup memory).
pub const MAX_SPLITS: usize = 256;
/// Query rows of one threadgroup of the prompt kernel (four simdgroups of eight rows).
pub const PREFILL_ROWS: usize = 32;
/// Threads of a decode threadgroup (eight simdgroups; the kernel assumes exactly this many).
pub const DECODE_THREADS: u32 = 256;
/// Threads of a prompt-kernel threadgroup (four simdgroups).
pub const PREFILL_THREADS: u32 = 128;
/// Threads of a merge threadgroup.
pub const MERGE_THREADS: u32 = 128;

/// Per-thread element counts `E` (channels per lane = `ceil(width / 32)`) of the decode instantiations, ascending.
pub const DECODE_E: [usize; 10] = [1, 2, 3, 4, 5, 6, 8, 12, 16, 18];
/// `(DKF, DVF)` = (key width, value width) in 8-wide fragments of the prompt-kernel instantiations, cheapest first.
pub const PREFILL_FRAGMENTS: [(usize, usize); 8] = [
    (8, 8),
    (10, 10),
    (12, 12),
    (14, 14),
    (16, 16),
    (24, 16),
    (24, 24),
    (32, 32),
];

/// Geometry of one layer of a [`DeviceKvCache`] as seen by the kernels. Positions are 16-bit words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvLayout {
    /// 1 = f16, 2 = q8_0.
    pub kind: i32,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub value_dim: usize,
    pub block_size: usize,
    pub pool_blocks: usize,
    /// Start of the layer in the key pool, in 16-bit words from the start of the bound buffer (`layer * words_per_layer_k`).
    pub key_base: usize,
    /// Same for the value pool.
    pub value_base: usize,
}

impl KvLayout {
    /// The layer `layer_idx` of `kv`, the pools bound as a whole.
    pub fn from_cache(kv: &DeviceKvCache, layer_idx: usize) -> Result<Self, BackendError> {
        if layer_idx >= kv.layers {
            return Err(invalid(format!(
                "layer {layer_idx} >= {} cached layers",
                kv.layers
            )));
        }
        let layout = Self {
            kind: kv.kv_kind,
            kv_heads: kv.kv_heads,
            head_dim: kv.head_dim,
            value_dim: kv.value_dim,
            block_size: kv.block_size,
            pool_blocks: kv.pool_blocks,
            key_base: layer_idx * kv.words_per_layer_k,
            value_base: layer_idx * kv.words_per_layer_v,
        };
        layout.check()?;
        Ok(layout)
    }

    pub fn slots(&self) -> usize {
        self.pool_blocks * self.block_size
    }

    /// Values per slot of the key / value pool.
    pub fn key_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    pub fn value_width(&self) -> usize {
        self.kv_heads * self.value_dim
    }

    /// 16-bit words of one layer of the key / value pool.
    pub fn key_words(&self) -> usize {
        kv_layer_words(self.kind, self.slots(), self.kv_heads, self.head_dim)
    }

    pub fn value_words(&self) -> usize {
        kv_layer_words(self.kind, self.slots(), self.kv_heads, self.value_dim)
    }

    /// q8: 16-bit words from the layer start to the first scale (0 for f16).
    pub fn scale_words(&self, dim: usize) -> usize {
        if self.kind == 2 {
            (self.slots() * self.kv_heads * dim).div_ceil(2)
        } else {
            0
        }
    }

    /// q8: `log2(group)` when the group is 32, 0 when the group is the whole head (`kv_quant_group`); 0 for f16.
    pub fn group_shift(&self, dim: usize) -> usize {
        if self.kind == 2 && kv_quant_group(dim) == 32 {
            5
        } else {
            0
        }
    }

    fn check(&self) -> Result<(), BackendError> {
        if !matches!(self.kind, 1 | 2) {
            return Err(BackendError::UnsupportedOperation(format!(
                "KV cache kind {} (expected 1 = f16 or 2 = q8_0)",
                self.kind
            )));
        }
        if self.kv_heads == 0
            || self.head_dim == 0
            || self.value_dim == 0
            || self.block_size == 0
            || self.pool_blocks == 0
        {
            return Err(invalid("empty KV layout"));
        }
        // the kernels address the pools with 64-bit offsets; slot ids and slot * block_size must fit 32 bits
        if self.slots() > u32::MAX as usize {
            return Err(BackendError::UnsupportedOperation(
                "more than 2^32 KV slots".into(),
            ));
        }
        Ok(())
    }
}

/// `Stream::kv_cache_write` (`fllm_kv_write_f16` / `fllm_kv_write_q8`): row `r` of `k` (`kv_heads x head_dim` floats) and `v` (`kv_heads x value_dim`)
/// go to slot `slots[r]`. Two independent dispatches (keys, values; no barrier needed between them). The slots are uploaded with the dispatches.
pub fn plan_kv_write(kv: &KvLayout, slots: &[usize]) -> Result<Vec<OpPlan>, BackendError> {
    kv.check()?;
    if slots.is_empty() {
        return Err(invalid("kv_cache_write without rows"));
    }
    if let Some(&s) = slots.iter().find(|&&s| s >= kv.slots()) {
        return Err(invalid(format!("KV slot {s} >= {} pool slots", kv.slots())));
    }
    let host = Arc::new(pack_u32(slots)?);
    let rows = slots.len();
    let mut plans = Vec::with_capacity(2);
    for (src, pool, dim, base) in [
        ("k", "key_pool", kv.head_dim, kv.key_base),
        ("v", "value_pool", kv.value_dim, kv.value_base),
    ] {
        let width = kv.kv_heads * dim;
        let slots_slot = Slots::new().read(src).host("slots", host.clone()).rw(pool);
        if kv.kind == 1 {
            let tx = round_up(width, 32).min(256);
            let params = Params::new()
                .u64("base", base as u64)
                .u64("scale_base", 0)
                .u("rows", rows)
                .u("kv", width)
                .u("group", 0);
            plans.push(plan(
                "fllm_kv_write_f16",
                params,
                [grid_dim(div_up(width, tx))?, grid_dim(rows)?, 1],
                [tx as u32, 1, 1],
                slots_slot,
            )?);
        } else {
            let group = kv_quant_group(dim);
            let params = Params::new()
                .u64("base", base as u64)
                .u64("scale_base", kv.scale_words(dim) as u64)
                .u("rows", rows)
                .u("kv", width)
                .u("group", group);
            plans.push(plan(
                "fllm_kv_write_q8",
                params,
                [grid_dim(width / group)?, grid_dim(rows)?, 1],
                [32, 1, 1],
                slots_slot,
            )?);
        }
    }
    Ok(plans)
}

/// Choices of the attention planner.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionTuning {
    /// Threadgroups the decode kernel should launch (rows x heads x splits) to fill the device; the planner splits the key range until it gets there.
    pub target_groups: usize,
    /// Most splits per (row, head) (at most [`MAX_SPLITS`]).
    pub max_splits: usize,
    /// Keys a split should have at least (a split of fewer keys costs more in the merge than it saves).
    pub min_split_keys: usize,
    /// Use the tiled prompt kernel (`simdgroup_matrix`) for runs of rows that share a block table.
    pub use_prefill_kernel: bool,
    /// Fewest rows of a run that the prompt kernel takes (below it the decode kernel is cheaper).
    pub prefill_min_rows: usize,
    /// SIMD width of the device (`Pipeline::simd_width`); the attention kernels need 32 (every Apple GPU).
    pub simd_width: usize,
    /// The device has `simdgroup_matrix` (`MetalDeviceInfo::simdgroup_matrix`); without it everything runs on the decode kernel.
    pub simdgroup_matrix: bool,
    /// Largest threadgroup the pipelines accept (`Pipeline::max_threads()`, at most 1024). The decode kernel needs [`DECODE_THREADS`], the prompt
    /// kernel [`PREFILL_THREADS`] (below that the rows go to the decode kernel), the merge kernel [`MERGE_THREADS`].
    pub max_threads: usize,
}

impl Default for AttentionTuning {
    fn default() -> Self {
        Self {
            target_groups: 256,
            max_splits: 64,
            min_split_keys: 128,
            use_prefill_kernel: true,
            prefill_min_rows: 8,
            simd_width: 32,
            simdgroup_matrix: true,
            max_threads: 1024,
        }
    }
}

/// Smallest decode instantiation (channels per lane) that covers both head widths, `None` above [`MAX_HEAD_WIDTH`].
pub fn decode_e(dk: usize, dv: usize) -> Option<usize> {
    let need = dk.max(dv).div_ceil(32);
    DECODE_E.iter().copied().find(|&e| e >= need)
}

/// Kernel name of the decode instantiation for `e` channels per lane (`None`: `e` is not in [`DECODE_E`]).
pub fn decode_kernel(q8: bool, e: usize) -> Option<&'static str> {
    Some(match (q8, e) {
        (false, 1) => "fllm_attn_decode_f16_e1",
        (false, 2) => "fllm_attn_decode_f16_e2",
        (false, 3) => "fllm_attn_decode_f16_e3",
        (false, 4) => "fllm_attn_decode_f16_e4",
        (false, 5) => "fllm_attn_decode_f16_e5",
        (false, 6) => "fllm_attn_decode_f16_e6",
        (false, 8) => "fllm_attn_decode_f16_e8",
        (false, 12) => "fllm_attn_decode_f16_e12",
        (false, 16) => "fllm_attn_decode_f16_e16",
        (false, 18) => "fllm_attn_decode_f16_e18",
        (true, 1) => "fllm_attn_decode_q8_e1",
        (true, 2) => "fllm_attn_decode_q8_e2",
        (true, 3) => "fllm_attn_decode_q8_e3",
        (true, 4) => "fllm_attn_decode_q8_e4",
        (true, 5) => "fllm_attn_decode_q8_e5",
        (true, 6) => "fllm_attn_decode_q8_e6",
        (true, 8) => "fllm_attn_decode_q8_e8",
        (true, 12) => "fllm_attn_decode_q8_e12",
        (true, 16) => "fllm_attn_decode_q8_e16",
        (true, 18) => "fllm_attn_decode_q8_e18",
        _ => return None,
    })
}

/// The cheapest prompt-kernel instantiation `(DKF, DVF)` that covers the head widths (`None`: none does, the decode kernel takes the rows).
pub fn prefill_fragments(dk: usize, dv: usize) -> Option<(usize, usize)> {
    let (nk, nv) = (dk.div_ceil(8), dv.div_ceil(8));
    PREFILL_FRAGMENTS
        .iter()
        .copied()
        .find(|&(fk, fv)| fk >= nk && fv >= nv)
}

/// Kernel name of the prompt-kernel instantiation `(DKF, DVF)`.
pub fn prefill_kernel(q8: bool, frags: (usize, usize)) -> Option<&'static str> {
    Some(match (q8, frags) {
        (false, (8, 8)) => "fllm_attn_prefill_f16_k8v8",
        (false, (10, 10)) => "fllm_attn_prefill_f16_k10v10",
        (false, (12, 12)) => "fllm_attn_prefill_f16_k12v12",
        (false, (14, 14)) => "fllm_attn_prefill_f16_k14v14",
        (false, (16, 16)) => "fllm_attn_prefill_f16_k16v16",
        (false, (24, 16)) => "fllm_attn_prefill_f16_k24v16",
        (false, (24, 24)) => "fllm_attn_prefill_f16_k24v24",
        (false, (32, 32)) => "fllm_attn_prefill_f16_k32v32",
        (true, (8, 8)) => "fllm_attn_prefill_q8_k8v8",
        (true, (10, 10)) => "fllm_attn_prefill_q8_k10v10",
        (true, (12, 12)) => "fllm_attn_prefill_q8_k12v12",
        (true, (14, 14)) => "fllm_attn_prefill_q8_k14v14",
        (true, (16, 16)) => "fllm_attn_prefill_q8_k16v16",
        (true, (24, 16)) => "fllm_attn_prefill_q8_k24v16",
        (true, (24, 24)) => "fllm_attn_prefill_q8_k24v24",
        (true, (32, 32)) => "fllm_attn_prefill_q8_k32v32",
        _ => return None,
    })
}

/// Plans for `Stream::attention_paged` on the cache layout `kv`: validates the batch (like `AttentionParams::validate`, without a `DeviceKvCache`),
/// groups the rows into dispatch segments and returns the dispatches in execution order (a barrier between consecutive plans where the declared
/// accesses conflict; segments are independent of each other). Slots: `q` (`rows x heads x head_dim` f32), `kpool`, `vpool` (the pools of the cache),
/// `out` (`rows x heads x value_dim`); `meta` and `bias` are host arrays, `part` is scratch (id 0, size = the maximum over the plans).
pub fn plan_attention(
    kv: &KvLayout,
    params: &AttentionParams,
    block_tables: &[i64],
    lengths: &[i64],
    tuning: &AttentionTuning,
) -> Result<Vec<OpPlan>, BackendError> {
    kv.check()?;
    let rows = lengths.len();
    let (heads, kv_heads, dk, dv) = (
        params.num_heads,
        params.num_kv_heads,
        params.head_dim,
        params.value_dim,
    );
    if rows == 0 {
        return Err(invalid("attention without rows"));
    }
    if tuning.simd_width != 32 {
        return Err(BackendError::UnsupportedOperation(format!(
            "attention kernels need 32-wide SIMD groups, the device has {}",
            tuning.simd_width
        )));
    }
    if kv_heads == 0 || heads == 0 || !heads.is_multiple_of(kv_heads) {
        return Err(invalid(format!(
            "attention heads {heads} not a multiple of kv heads {kv_heads}"
        )));
    }
    if kv.kv_heads != kv_heads
        || kv.head_dim != dk
        || kv.value_dim != dv
        || kv.block_size != params.block_size
    {
        return Err(invalid("attention parameters do not match the KV cache"));
    }
    if dk > MAX_HEAD_WIDTH || dv > MAX_HEAD_WIDTH {
        return Err(BackendError::UnsupportedOperation(format!(
            "attention head widths {dk}/{dv} exceed {MAX_HEAD_WIDTH}"
        )));
    }
    if params.stride == 0 || block_tables.len() != rows * params.stride {
        return Err(invalid(format!(
            "block tables hold {} entries, expected {rows} rows x stride {}",
            block_tables.len(),
            params.stride
        )));
    }
    for (name, values) in [
        ("sinks", &params.sinks),
        ("alibi slopes", &params.alibi_slopes),
    ] {
        if values.as_ref().is_some_and(|v| v.len() < heads) {
            return Err(invalid(format!("{name} shorter than {heads} heads")));
        }
    }
    if params.chunk == Some(0) || params.window == Some(0) {
        return Err(invalid("zero attention chunk or window"));
    }
    if params.softcap.is_some_and(|c| c == 0.0 || !c.is_finite()) || !params.scale.is_finite() {
        return Err(invalid(
            "attention scale or soft cap is not a finite non-zero number",
        ));
    }
    let bs = params.block_size;
    for (r, &n) in lengths.iter().enumerate() {
        if n <= 0 || (n as usize).div_ceil(bs) > params.stride {
            return Err(invalid(format!(
                "row {r}: length {n} does not fit its block table"
            )));
        }
        let need = (n as usize).div_ceil(bs);
        if block_tables[r * params.stride..][..need]
            .iter()
            .any(|&b| b < 0 || b as usize >= kv.pool_blocks)
        {
            return Err(invalid(format!("row {r}: KV block outside the pool")));
        }
    }

    // host arrays shared by all dispatches
    let mut meta = pack_i32(lengths)?;
    meta.extend_from_slice(&pack_i32(block_tables)?);
    let mut bias = vec![0.0f32; 2 * heads];
    if let Some(s) = &params.sinks {
        bias[..heads].copy_from_slice(&s[..heads]);
    }
    if let Some(a) = &params.alibi_slopes {
        bias[heads..].copy_from_slice(&a[..heads]);
    }
    let common = Common {
        kv,
        params,
        rows,
        flags: usize::from(params.sinks.is_some())
            | (usize::from(params.alibi_slopes.is_some()) << 1),
        meta: Arc::new(meta),
        bias: Arc::new(pack_f32(&bias)),
        tuning,
    };

    let start_of = |n: usize| -> usize {
        match (params.chunk, params.window) {
            (Some(c), _) => (n - 1) / c * c,
            (None, Some(w)) => n.saturating_sub(w),
            (None, None) => 0,
        }
    };
    let need = |n: i64| (n as usize).div_ceil(bs);
    let table = |r: usize| &block_tables[r * params.stride..(r + 1) * params.stride];
    let prefill_ok = tuning.use_prefill_kernel
        && tuning.simdgroup_matrix
        && tuning.max_threads >= PREFILL_THREADS as usize
        && tuning.prefill_min_rows > 1
        && prefill_fragments(dk, dv).is_some();

    // segments of rows: (first, end, prompt kernel)
    let mut segments: Vec<(usize, usize, bool)> = Vec::new();
    let mut i = 0;
    while i < rows {
        let mut take = 1;
        if prefill_ok {
            // longest run of <= PREFILL_ROWS rows whose block tables are prefixes of one table (checked through the longest row seen so far)
            let mut longest = i;
            while take < PREFILL_ROWS && i + take < rows {
                let q = i + take;
                let m = need(lengths[q]).min(need(lengths[longest]));
                if table(q)[..m] != table(longest)[..m] {
                    break;
                }
                if need(lengths[q]) > need(lengths[longest]) {
                    longest = q;
                }
                take += 1;
            }
        }
        if prefill_ok && take >= tuning.prefill_min_rows.max(2) {
            // consecutive full tiles are merged into one dispatch
            let extend = matches!(segments.last(), Some(&(f, e, true)) if e == i && (e - f).is_multiple_of(PREFILL_ROWS));
            if extend {
                if let Some(s) = segments.last_mut() {
                    s.1 = i + take;
                }
            } else {
                segments.push((i, i + take, true));
            }
            i += take;
        } else {
            let extend = matches!(segments.last(), Some(&(_, e, false)) if e == i);
            if extend {
                if let Some(s) = segments.last_mut() {
                    s.1 = i + 1;
                }
            } else {
                segments.push((i, i + 1, false));
            }
            i += 1;
        }
    }

    let mut plans = Vec::new();
    for (first, end, prefill) in segments {
        if prefill {
            plans.push(common.prefill_plan(first, end)?);
        } else {
            let n_eff = (first..end)
                .map(|r| lengths[r] as usize - start_of(lengths[r] as usize))
                .max()
                .unwrap_or(1);
            common.decode_plans(first, end, n_eff, &mut plans)?;
        }
    }
    Ok(plans)
}

/// What the plans of one `plan_attention` call share.
struct Common<'a> {
    kv: &'a KvLayout,
    params: &'a AttentionParams,
    rows: usize,
    flags: usize,
    meta: Arc<Vec<u8>>,
    bias: Arc<Vec<u8>>,
    tuning: &'a AttentionTuning,
}

impl Common<'_> {
    /// The `o_attn_params` struct of a dispatch over the global rows `row0 .. rows_end` with `nsplit` splits.
    fn attn_params(&self, nsplit: usize, row0: usize, rows_end: usize) -> Params {
        let (kv, p) = (self.kv, self.params);
        Params::new()
            .u64("k_base", kv.key_base as u64)
            .u64("v_base", kv.value_base as u64)
            .u64("k_scale", kv.scale_words(kv.head_dim) as u64)
            .u64("v_scale", kv.scale_words(kv.value_dim) as u64)
            .u("heads", p.num_heads)
            .u("kv_heads", p.num_kv_heads)
            .u("dk", p.head_dim)
            .u("dv", p.value_dim)
            .u("stride", p.stride)
            .u("block_size", p.block_size)
            .f("scale", p.scale)
            .f("softcap", p.softcap.unwrap_or(0.0))
            .u("window", p.window.unwrap_or(0))
            .u("chunk", p.chunk.unwrap_or(0))
            .u("flags", self.flags)
            .u("nsplit", nsplit)
            .u("rows_total", self.rows)
            .u("row0", row0)
            .u("rows_end", rows_end)
            .u("k_gshift", kv.group_shift(kv.head_dim))
            .u("v_gshift", kv.group_shift(kv.value_dim))
    }

    /// Decode kernel (plus merge) for the rows `first .. end`; `n_eff` is the longest key range among them.
    fn decode_plans(
        &self,
        first: usize,
        end: usize,
        n_eff: usize,
        plans: &mut Vec<OpPlan>,
    ) -> Result<(), BackendError> {
        if self.tuning.max_threads < DECODE_THREADS as usize {
            return Err(BackendError::UnsupportedOperation(format!(
                "the decode attention kernel needs {DECODE_THREADS} threads per group, the device allows {}",
                self.tuning.max_threads
            )));
        }
        let (p, t) = (self.params, self.tuning);
        let seg_rows = end - first;
        let e = decode_e(p.head_dim, p.value_dim)
            .ok_or_else(|| invalid("attention head width above the decode kernels"))?;
        let name = decode_kernel(self.kv.kind == 2, e)
            .ok_or_else(|| invalid("no decode kernel for this head width"))?;
        let base_groups = seg_rows * p.num_heads;
        let want = div_up(t.target_groups, base_groups.max(1));
        let by_keys = (n_eff / t.min_split_keys.max(1)).max(1);
        let nsplit = want.min(by_keys).min(t.max_splits).clamp(1, MAX_SPLITS);
        let records = base_groups * nsplit;
        let part_bytes = (records * 2 + records * p.value_dim) * 4;
        let slots = Slots::new()
            .read("q")
            .read("kpool")
            .read("vpool")
            .host("meta", self.meta.clone())
            .host("bias", self.bias.clone())
            .write("out")
            .scratch("part", 0, part_bytes, Access::Write);
        plans.push(plan(
            name,
            self.attn_params(nsplit, first, end),
            [
                grid_dim(nsplit)?,
                grid_dim(p.num_heads)?,
                grid_dim(seg_rows)?,
            ],
            [DECODE_THREADS, 1, 1],
            slots,
        )?);
        if nsplit > 1 {
            let params = Params::new()
                .u("heads", p.num_heads)
                .u("dv", p.value_dim)
                .u("nsplit", nsplit)
                .u("flags", self.flags)
                .u("row0", first);
            let slots = Slots::new()
                .host("bias", self.bias.clone())
                .write("out")
                .scratch("part", 0, part_bytes, Access::Read);
            plans.push(plan(
                "fllm_attn_merge_f32",
                params,
                [grid_dim(p.num_heads)?, grid_dim(seg_rows)?, 1],
                [MERGE_THREADS, 1, 1],
                slots,
            )?);
        }
        Ok(())
    }

    /// Prompt kernel for the rows `first .. end` (tiles of [`PREFILL_ROWS`] rows, one threadgroup per tile and query head).
    fn prefill_plan(&self, first: usize, end: usize) -> Result<OpPlan, BackendError> {
        let p = self.params;
        let frags = prefill_fragments(p.head_dim, p.value_dim)
            .ok_or_else(|| invalid("no prompt kernel for this head width"))?;
        let name = prefill_kernel(self.kv.kind == 2, frags)
            .ok_or_else(|| invalid("no prompt kernel for this head width"))?;
        let slots = Slots::new()
            .read("q")
            .read("kpool")
            .read("vpool")
            .host("meta", self.meta.clone())
            .host("bias", self.bias.clone())
            .write("out");
        plan(
            name,
            self.attn_params(1, first, end),
            [
                grid_dim(div_up(end - first, PREFILL_ROWS))?,
                grid_dim(p.num_heads)?,
                1,
            ],
            [PREFILL_THREADS, 1, 1],
            slots,
        )
    }
}
