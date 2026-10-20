//! GPU time per shader, for tuning a device: `FLLM_VK_PROFILE=1` writes a timestamp after every dispatch of the recorded command buffers
//! and sums the time between consecutive timestamps per pipeline name (so a dispatch is charged the barrier wait in front of it as well).
//! A table of the biggest consumers goes to stderr every [`REPORT_EVERY`] submissions. Off by default; costs nothing then.

use super::super::instance::error::{vk_call, VkResult};
use super::super::instance::VkContext;
use ash::vk;
use std::collections::HashMap;
use std::sync::Arc;

/// Query slots per command buffer; dispatches beyond that are not timed.
pub const SLOTS: u32 = 1024;
/// Command buffers whose slots can be in flight at once.
const RING: u32 = 4;
const REPORT_EVERY: u64 = 256;

/// The timestamps one command buffer wrote: first slot, number of slots used and the pipeline name of each dispatch.
pub struct Recorded {
    pub base: u32,
    pub used: u32,
    pub names: Vec<String>,
}

pub struct Profiler {
    pool: vk::QueryPool,
    period_ns: f64,
    mask: u64,
    next_ring: u32,
    total: HashMap<String, (u64, f64)>,
    reads: u64,
}

impl Profiler {
    /// `None` unless `FLLM_VK_PROFILE` is set and the queue supports timestamps.
    pub fn new(ctx: &Arc<VkContext>) -> Option<Self> {
        if std::env::var_os("FLLM_VK_PROFILE").is_none_or(|v| v.is_empty() || v == "0") {
            return None;
        }
        let core = ctx.core();
        // SAFETY: queries on the physical device the logical device was created from.
        let (bits, period) = unsafe {
            let families = core.instance().get_physical_device_queue_family_properties(core.physical);
            let props = core.instance().get_physical_device_properties(core.physical);
            (
                families.get(core.queue_family as usize).map_or(0, |f| f.timestamp_valid_bits),
                props.limits.timestamp_period,
            )
        };
        if bits == 0 {
            eprintln!("[fllm-rs] FLLM_VK_PROFILE: the queue has no timestamps");
            return None;
        }
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(SLOTS * RING);
        // SAFETY: plain creation on a valid device.
        let pool = unsafe { core.device.create_query_pool(&info, None) }.ok()?;
        eprintln!("[fllm-rs] FLLM_VK_PROFILE: timing every dispatch (table every {REPORT_EVERY} submissions)");
        Some(Self {
            pool,
            period_ns: f64::from(period),
            mask: if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 },
            next_ring: 0,
            total: HashMap::new(),
            reads: 0,
        })
    }

    /// Reset the slots of the next ring entry and write the first timestamp; returns the base slot.
    pub fn start(&mut self, device: &ash::Device, cb: vk::CommandBuffer) -> u32 {
        let base = (self.next_ring % RING) * SLOTS;
        self.next_ring = self.next_ring.wrapping_add(1);
        // SAFETY: `cb` is recording; the slots belong to this ring entry, whose earlier results were read before it came round again.
        unsafe {
            device.cmd_reset_query_pool(cb, self.pool, base, SLOTS);
            device.cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, self.pool, base);
        }
        base
    }

    /// Timestamp after a dispatch (slot `base + used`).
    pub fn stamp(&self, device: &ash::Device, cb: vk::CommandBuffer, slot: u32) {
        // SAFETY: `cb` is recording and `slot` is below `base + SLOTS`.
        unsafe { device.cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, self.pool, slot) };
    }

    /// Add the finished command buffer `rec` to the totals; prints the table now and then.
    pub fn collect(&mut self, device: &ash::Device, rec: &Recorded) -> VkResult<()> {
        if rec.used < 2 {
            return Ok(());
        }
        let mut stamps = vec![0u64; rec.used as usize];
        // SAFETY: the submission that wrote these slots has completed (timeline).
        unsafe {
            device
                .get_query_pool_results(self.pool, rec.base, &mut stamps, vk::QueryResultFlags::TYPE_64)
                .map_err(vk_call("vkGetQueryPoolResults"))?;
        }
        for (i, name) in rec.names.iter().enumerate() {
            let (a, b) = (stamps[i] & self.mask, stamps[i + 1] & self.mask);
            let ticks = b.wrapping_sub(a) & self.mask;
            let e = self.total.entry(name.clone()).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += ticks as f64 * self.period_ns;
        }
        self.reads += 1;
        if self.reads % REPORT_EVERY == 0 {
            self.report();
        }
        Ok(())
    }

    fn report(&self) {
        let mut rows: Vec<(&String, &(u64, f64))> = self.total.iter().collect();
        rows.sort_by(|a, b| b.1 .1.total_cmp(&a.1 .1));
        let sum: f64 = rows.iter().map(|r| r.1 .1).sum();
        eprintln!("[fllm-rs] GPU time per shader over {} command buffers (sum {:.1} ms per buffer):", self.reads, sum / 1e6 / self.reads as f64);
        eprintln!("[fllm-rs]   {:<28} {:>9} {:>10} {:>9} {:>6}", "shader", "calls/buf", "us/call", "ms/buf", "%");
        for (name, (n, ns)) in rows.iter().take(14) {
            eprintln!(
                "[fllm-rs]   {:<28} {:>9.1} {:>10.1} {:>9.3} {:>5.1}%",
                name,
                *n as f64 / self.reads as f64,
                ns / *n as f64 / 1e3,
                ns / 1e6 / self.reads as f64,
                100.0 * ns / sum
            );
        }
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        // SAFETY: all submissions of the recorder are complete.
        unsafe { device.destroy_query_pool(self.pool, None) };
    }
}
