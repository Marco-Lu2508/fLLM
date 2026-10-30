//! Plans of the KV-cache write and of paged attention (decode with split-KV, prompt tiles). Shaders: `kv_write_f16`, `kv_write_q8`, `attn_decode_*`,
//! `attn_merge`, `attn_prefill_*`; layouts in `shaders/vulkan/INTERFACE_OPS.md`.

use super::*;
use crate::backend::{kv_layer_words, kv_quant_group, AttentionParams, DeviceKvCache};

/// Widest key or value head the attention shaders take (shared memory of the query / accumulators).
pub const MAX_HEAD_WIDTH: usize = 512;
/// Most splits of the key range (`attn_merge` keeps the weights of the splits in shared memory).
pub const MAX_SPLITS: usize = 256;
/// Query rows of one workgroup of `attn_prefill`.
pub const PREFILL_ROWS: usize = 16;
/// Query rows per workgroup of the wide prompt kernel (`attn_prefill_*_r64`).
pub const PREFILL_ROWS_WIDE: usize = 64;

/// Geometry of one layer of a [`DeviceKvCache`] as seen by the shaders. Positions are 16-bit words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvLayout {
    /// 1 = f16, 2 = q8_0.
    pub kind: i32,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub value_dim: usize,
    pub block_size: usize,
    pub pool_blocks: usize,
    /// Start of the layer in the key pool, in 16-bit words from the start of the bound window (`layer * words_per_layer_k` for a pool bound as a whole;
    /// a sub-window binding subtracts the window's first word).
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
            kv_heads: kv.layer_kv_heads(layer_idx),
            head_dim: kv.layer_head_dim(layer_idx),
            value_dim: kv.layer_value_dim(layer_idx),
            block_size: kv.block_size,
            pool_blocks: kv.pool_blocks,
            key_base: kv.layer_key_offset(layer_idx),
            value_base: kv.layer_value_offset(layer_idx),
        };
        layout.check()?;
        Ok(layout)
    }

    /// The same layer with other window-relative starts (pool bound as a sub-window).
    pub fn rebased(mut self, key_base: usize, value_base: usize) -> Result<Self, BackendError> {
        self.key_base = key_base;
        self.value_base = value_base;
        self.check()?;
        Ok(self)
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

    /// q8: 16-bit words from the layer start to the first scale.
    pub fn scale_base(&self, dim: usize) -> usize {
        if self.kind == 2 {
            (self.slots() * self.kv_heads * dim).div_ceil(2)
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
        // the shaders index bytes (2 * word) and words with 32 bits
        let limit = u32::MAX as usize;
        let overflow = |base: usize, words: usize| {
            base.checked_add(words)
                .and_then(|w| w.checked_mul(2))
                .is_none_or(|b| b > limit)
        };
        if overflow(self.key_base, self.key_words())
            || overflow(self.value_base, self.value_words())
        {
            return Err(BackendError::UnsupportedOperation(
                "KV pool window larger than 4 GiB: bind a sub-window and rebase the layer".into(),
            ));
        }
        Ok(())
    }
}

/// `Stream::kv_cache_write` (`kv_write_f16` / `kv_write_q8`): row `r` of `k` (`kv_heads x head_dim` floats) and `v` (`kv_heads x value_dim`) go to slot
/// `slots[r]`. Two independent dispatches (keys, values; no barrier needed between them). The slots are uploaded with the dispatches.
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
        let slots_slot = Slots::new().read(src).host("slots", host.clone());
        let slots_slot = slots_slot.rw(pool);
        if kv.kind == 1 {
            let push = Push::new().u("rows", rows).u("kv", width).u("base", base);
            plans.push(plan(
                "kv_write_f16",
                push,
                groups_1d(rows * (width / 2 + 1), 256),
                slots_slot,
            )?);
        } else {
            let group = kv_quant_group(dim);
            let push = Push::new()
                .u("rows", rows)
                .u("kv", width)
                .u("group", group)
                .u("base", base)
                .u("scale_base", kv.scale_base(dim));
            plans.push(plan(
                "kv_write_q8",
                push,
                groups_1d(rows * (width / group), 128),
                slots_slot,
            )?);
        }
    }
    Ok(plans)
}

/// Choices of the attention planner.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionTuning {
    /// Workgroups the decode kernel should launch (rows x heads x splits) to fill the device; the planner splits the key range until it gets there.
    pub target_groups: usize,
    /// Most splits per (row, head).
    pub max_splits: usize,
    /// Keys a split should have at least (a split of fewer keys costs more in the merge than it saves).
    pub min_split_keys: usize,
    /// Use the tiled prompt kernel for runs of rows that share a block table.
    pub use_prefill_kernel: bool,
    /// Fewest rows of a run that the prompt kernel takes (below it the decode kernel is cheaper).
    pub prefill_min_rows: usize,
    /// Query rows per workgroup of the prompt kernel: [`PREFILL_ROWS`] (128 invocations) or [`PREFILL_ROWS_WIDE`] (512 invocations, the
    /// `_r64` variants), see [`AttentionTuning::for_device`].
    pub prefill_rows: usize,
    /// The device runs the cooperative-matrix prompt kernel (`attn_prefill_cm_d*`: 64 rows per workgroup, f16 cache, key = value width 64 or 128).
    pub coop_prefill: bool,
}

impl Default for AttentionTuning {
    fn default() -> Self {
        Self {
            target_groups: 512,
            max_splits: 64,
            min_split_keys: 64,
            use_prefill_kernel: true,
            prefill_min_rows: 4,
            prefill_rows: PREFILL_ROWS,
            coop_prefill: false,
        }
    }
}

impl AttentionTuning {
    /// The defaults with the wide prompt tile on devices whose workgroups take 512 invocations (`FLLM_VK_ATTN_ROWS=16` keeps the narrow one).
    pub fn for_device(max_workgroup: u32) -> Self {
        static NARROW: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let narrow = *NARROW.get_or_init(|| std::env::var("FLLM_VK_ATTN_ROWS").is_ok_and(|v| v.trim() == "16"));
        let wide = !narrow && max_workgroup as usize >= 8 * PREFILL_ROWS_WIDE;
        Self { prefill_rows: if wide { PREFILL_ROWS_WIDE } else { PREFILL_ROWS }, ..Self::default() }
    }

    /// The prompt kernel on cooperative matrices where the device has them with 32-wide subgroups (NVIDIA); `FLLM_VK_ATTN_COOP=0` keeps the scalar kernel.
    pub fn with_coop_prefill(self, caps: &crate::backend::vulkan::weights::plan::VkCapsLike) -> Self {
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let off = *OFF.get_or_init(|| std::env::var("FLLM_VK_ATTN_COOP").is_ok_and(|v| matches!(v.trim(), "0" | "off")));
        let ok = !off && !crate::backend::vulkan::backend::is_exact_math() && caps.coop_matrix && caps.fp16 && caps.vendor == 0x10de && caps.subgroup_size == 32 && caps.max_workgroup >= 128 && caps.max_shared >= 40 * 1024;
        Self { coop_prefill: ok && self.prefill_rows == PREFILL_ROWS_WIDE, ..self }
    }
}

/// Plans for `Stream::attention_paged` on the cache layout `kv`: validates the batch (like `AttentionParams::validate`, without a `DeviceKvCache`),
/// groups the rows into dispatch segments and returns the dispatches in execution order (barrier between consecutive plans; segments are independent
/// of each other). Slots: `q` (`rows x heads x head_dim` f32), `kpool`, `vpool` (the pools of the cache), `out` (`rows x heads x value_dim`);
/// `meta` and `bias` are host arrays, `part` is scratch (id 0, size = the maximum over the plans).
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
    let meta = Arc::new(meta);
    let mut bias = vec![0.0f32; 2 * heads];
    if let Some(s) = &params.sinks {
        bias[..heads].copy_from_slice(&s[..heads]);
    }
    if let Some(a) = &params.alibi_slopes {
        bias[heads..].copy_from_slice(&a[..heads]);
    }
    let bias = Arc::new(pack_f32(&bias));
    let flags =
        usize::from(params.sinks.is_some()) | (usize::from(params.alibi_slopes.is_some()) << 1);
    let common = Common {
        kv,
        params,
        rows,
        flags,
        meta,
        bias,
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

    // segments of rows: (first, end, prompt kernel)
    let mut segments: Vec<(usize, usize, bool)> = Vec::new();
    let mut i = 0;
    while i < rows {
        let mut take = 1;
        if tuning.use_prefill_kernel && tuning.prefill_min_rows > 1 {
            // longest run of <= prefill_rows rows whose block tables are prefixes of one table (checked pairwise through the longest row seen so far)
            let mut longest = i;
            while take < tuning.prefill_rows && i + take < rows {
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
        if take >= tuning.prefill_min_rows.max(2) && tuning.use_prefill_kernel {
            let br = tuning.prefill_rows;
            let extend = matches!(segments.last(), Some(&(f, e, true)) if e == i && (e - f).is_multiple_of(br) && (e - f + take) <= MAX_GROUPS_PER_DIM as usize * br);
            if extend {
                if let Some(s) = segments.last_mut() {
                    s.1 = i + take;
                }
            } else {
                segments.push((i, i + take, true));
            }
            i += take;
        } else {
            let extend = matches!(segments.last(), Some(&(f, e, false)) if e == i && (e - f) < MAX_GROUPS_PER_DIM as usize);
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
    fn is_q8(&self) -> bool {
        self.kv.kind == 2
    }

    fn slots(&self, part: Option<usize>) -> Slots {
        let s = Slots::new()
            .read("q")
            .read("kpool")
            .read("vpool")
            .host("meta", self.meta.clone())
            .host("bias", self.bias.clone())
            .write("out");
        match part {
            Some(bytes) => s.scratch("part", 0, bytes, Access::ReadWrite),
            None => s,
        }
    }

    /// The push fields the two attention shaders share, up to `flags`.
    fn head_push(&self) -> Push {
        let p = self.params;
        Push::new()
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
    }

    fn cache_push(&self, push: Push) -> Push {
        let kv = self.kv;
        push.u("k_base", kv.key_base)
            .u("v_base", kv.value_base)
            .u("k_scale", kv.scale_base(kv.head_dim))
            .u("v_scale", kv.scale_base(kv.value_dim))
            .u("k_group", kv_quant_group(kv.head_dim))
            .u("v_group", kv_quant_group(kv.value_dim))
    }

    fn decode_plans(
        &self,
        first: usize,
        end: usize,
        n_eff: usize,
        out: &mut Vec<OpPlan>,
    ) -> Result<(), BackendError> {
        let p = self.params;
        let seg_rows = end - first;
        let want = self
            .tuning
            .target_groups
            .div_ceil(seg_rows * p.num_heads)
            .max(1);
        let by_len = (n_eff / self.tuning.min_split_keys.max(1)).max(1);
        let nsplit = want
            .min(by_len)
            .min(self.tuning.max_splits.max(1))
            .min(MAX_SPLITS);
        let records = seg_rows * p.num_heads * nsplit;
        let push = self.cache_push(
            self.head_push()
                .u("nsplit", nsplit)
                .u("rows_total", self.rows)
                .u("row0", first),
        );
        let part_bytes = records * (2 + p.value_dim) * 4;
        let slots = if nsplit == 1 {
            self.slots(None).dummy("part", Access::ReadWrite)
        } else {
            self.slots(Some(part_bytes))
        };
        let name = if self.is_q8() {
            "attn_decode_q8"
        } else {
            "attn_decode_f16"
        };
        out.push(plan(
            name,
            push,
            [nsplit as u32, p.num_heads as u32, seg_rows as u32],
            slots,
        )?);
        if nsplit > 1 {
            let merge_push = Push::new()
                .u("heads", p.num_heads)
                .u("dv", p.value_dim)
                .u("nsplit", nsplit)
                .u("flags", self.flags)
                .u("row0", first);
            let merge_slots = Slots::new()
                .host("bias", self.bias.clone())
                .write("out")
                .scratch("part", 0, part_bytes, Access::Read);
            out.push(plan(
                "attn_merge",
                merge_push,
                [p.num_heads as u32, seg_rows as u32, 1],
                merge_slots,
            )?);
        }
        Ok(())
    }

    fn prefill_plan(&self, first: usize, end: usize) -> Result<OpPlan, BackendError> {
        let p = self.params;
        let wide = self.tuning.prefill_rows == PREFILL_ROWS_WIDE;
        let cm = if self.tuning.coop_prefill && wide && !self.is_q8() && p.head_dim == p.value_dim {
            match p.head_dim {
                128 => Some("attn_prefill_cm_d128"),
                64 => Some("attn_prefill_cm_d64"),
                _ => None,
            }
            .filter(|n| crate::backend::vulkan::shaders::SHADERS.iter().any(|(s, _)| s == n))
        } else {
            None
        };
        let name = match (self.is_q8(), p.value_dim, wide) {
            (false, 0..=128, false) => "attn_prefill_f16_d128",
            (false, 129..=256, false) => "attn_prefill_f16_d256",
            (false, _, false) => "attn_prefill_f16_d512",
            (true, 0..=128, false) => "attn_prefill_q8_d128",
            (true, 129..=256, false) => "attn_prefill_q8_d256",
            (true, _, false) => "attn_prefill_q8_d512",
            (false, 0..=128, true) => "attn_prefill_f16_d128_r64",
            (false, 129..=256, true) => "attn_prefill_f16_d256_r64",
            (false, _, true) => "attn_prefill_f16_d512_r64",
            (true, 0..=128, true) => "attn_prefill_q8_d128_r64",
            (true, 129..=256, true) => "attn_prefill_q8_d256_r64",
            (true, _, true) => "attn_prefill_q8_d512_r64",
        };
        let name = cm.unwrap_or(name);
        let push = self.cache_push(
            self.head_push()
                .u("rows_total", self.rows)
                .u("row0", first)
                .u("rows_end", end),
        );
        // the field order of the shader: ... flags, rows_total, row0, rows_end, k_base, ...
        let blocks = div_up(end - first, self.tuning.prefill_rows);
        plan(
            name,
            push,
            [blocks as u32, p.num_heads as u32, 1],
            self.slots(None),
        )
    }
}
