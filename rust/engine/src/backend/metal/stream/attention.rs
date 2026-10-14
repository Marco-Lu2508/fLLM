//! KV-cache write and paged attention.

use super::exec::{dev, invalid, Bound};
use super::MetalStream;
use crate::backend::metal::backend::shared::buffer_ref;
use crate::backend::metal::ops::attention::{plan_attention, plan_kv_write, AttentionTuning, KvLayout};
use crate::backend::{validate_kv_write, AttentionParams, BackendError, DeviceBuffer, DeviceKvCache};

impl MetalStream {
    /// Stores row `r` of `k` and `v` in slot `slots[r]` of layer `layer_idx` (two dispatches, keys and values).
    pub(super) fn op_kv_write(
        &mut self,
        kv: &DeviceKvCache,
        layer_idx: usize,
        slots: &[usize],
        k: &DeviceBuffer,
        v: &DeviceBuffer,
    ) -> Result<(), BackendError> {
        validate_kv_write(kv, layer_idx, slots, k.size, v.size)?;
        if slots.is_empty() {
            return Ok(());
        }
        let layout = KvLayout::from_cache(kv, layer_idx)?;
        let plans = plan_kv_write(&layout, slots)?;
        self.run_plans(
            &plans,
            &[
                ("k", dev(k)?),
                ("v", dev(v)?),
                ("key_pool", Bound::Buf(buffer_ref(&kv.key_pool)?)),
                ("value_pool", Bound::Buf(buffer_ref(&kv.value_pool)?)),
            ],
            &[],
        )
    }

    /// Planner settings from the device: SIMD width and thread limit of this GPU, `simdgroup_matrix` for the prompt kernel.
    fn attention_tuning(&self) -> AttentionTuning {
        let info = self.shared.ctx.device.info();
        AttentionTuning {
            simd_width: self.shared.caps.simd_width as usize,
            simdgroup_matrix: info.simdgroup_matrix,
            max_threads: info.max_threads_per_threadgroup[0],
            ..AttentionTuning::default()
        }
    }

    /// Row `r` of the batch attends to positions `..lengths[r]` through its row of `block_tables`.
    ///
    /// The planner picks the kernels (decode with split-K and merge, or the `simdgroup_matrix` prompt kernel for runs of rows that share a block
    /// table); when a pipeline cannot run its planned threadgroup (register pressure of the large prompt kernels) the batch is planned again with that
    /// limit, which sends the rows to the decode kernel.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn op_attention(
        &mut self,
        out: &mut DeviceBuffer,
        q: &DeviceBuffer,
        kv: &DeviceKvCache,
        block_tables: &[i64],
        lengths: &[i64],
        layer_idx: usize,
        params: &AttentionParams,
    ) -> Result<(), BackendError> {
        params.validate(kv, block_tables, lengths, layer_idx)?;
        let (dk, dv, heads) = (params.head_dim, params.value_dim, params.num_heads);
        let rows = lengths.len();
        if rows == 0 {
            return Ok(());
        }
        if q.size < rows * heads * dk * 4 || out.size < rows * heads * dv * 4 {
            return Err(invalid("attention q/out smaller than rows x heads x dim"));
        }
        let layout = KvLayout::from_cache(kv, layer_idx)?;
        let mut tuning = self.attention_tuning();
        let mut tries = 0;
        let plans = loop {
            let plans = plan_attention(&layout, params, block_tables, lengths, &tuning)?;
            let mut limit = tuning.max_threads;
            for plan in &plans {
                let pipeline = self.pipeline(&plan.kernel_name, &[])?;
                let threads: usize = plan.threads_per_group.iter().map(|&t| t as usize).product();
                if threads > pipeline.max_threads() {
                    limit = limit.min(pipeline.max_threads());
                }
            }
            if limit == tuning.max_threads {
                break plans;
            }
            tries += 1;
            if tries > 2 {
                return Err(BackendError::UnsupportedOperation(
                    "attention kernels do not fit the threadgroup limits of this GPU".into(),
                ));
            }
            tuning.max_threads = limit;
        };
        self.run_plans(
            &plans,
            &[
                ("q", dev(q)?),
                ("kpool", Bound::Buf(buffer_ref(&kv.key_pool)?)),
                ("vpool", Bound::Buf(buffer_ref(&kv.value_pool)?)),
                ("out", dev(out)?),
            ],
            &[],
        )
    }
}
