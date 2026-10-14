//! Operations the Vulkan stream does not run on the device: they go through the CPU backend (copy to the host, compute, copy back). Rare
//! paths only: sampling with temperature, the stand-alone `moe_route`, and attention with head widths beyond the shaders' 512.

use super::VulkanStream;
use crate::backend::cpu::{CpuBackend, CpuStream};
use crate::backend::vulkan::backend::mem::view_of;
use crate::backend::{warn_fallback_reason, AttentionParams, Backend, BackendError, DeviceBuffer, DeviceKvCache, SamplingParams, Stream};

/// The CPU backend used for the fallbacks (created on first use).
pub(super) struct Cpu {
    backend: CpuBackend,
    stream: CpuStream,
}

impl Cpu {
    pub fn new() -> Self {
        Self {
            backend: CpuBackend::new(),
            stream: CpuStream,
        }
    }
}

impl VulkanStream {
    fn take_cpu(&mut self) -> Cpu {
        self.cpu.take().unwrap_or_else(Cpu::new)
    }

    /// Device buffer -> CPU-backend buffer.
    fn copy_to_host(&mut self, cpu: &mut Cpu, buffer: &DeviceBuffer) -> Result<DeviceBuffer, BackendError> {
        let mut bytes = vec![0u8; buffer.size];
        self.do_download(&mut bytes, buffer)?;
        let mut host = cpu.backend.allocate_buffer(buffer.size)?;
        cpu.stream.upload(&mut host, &bytes)?;
        Ok(host)
    }

    /// CPU-backend buffer -> device buffer.
    fn copy_from_host(&mut self, cpu: &mut Cpu, host: &DeviceBuffer, dst: &mut DeviceBuffer) -> Result<(), BackendError> {
        let mut bytes = vec![0u8; dst.size.min(host.size)];
        cpu.stream.download(&mut bytes, host)?;
        self.do_upload(dst, &bytes)
    }

    pub(super) fn moe_route_on_cpu(
        &mut self,
        weights_out: &mut DeviceBuffer,
        indices_out: &mut DeviceBuffer,
        router_logits: &DeviceBuffer,
        num_experts: usize,
        top_k: usize,
    ) -> Result<(), BackendError> {
        warn_fallback_reason(
            "moe_route",
            "the Vulkan backend routes with moe_topk; this stand-alone call runs on the host",
        );
        let mut cpu = self.take_cpu();
        let result = (|| {
            let logits = self.copy_to_host(&mut cpu, router_logits)?;
            let mut weights = cpu.backend.allocate_buffer(weights_out.size)?;
            let mut indices = cpu.backend.allocate_buffer(indices_out.size)?;
            cpu.stream.moe_route(&mut weights, &mut indices, &logits, num_experts, top_k)?;
            self.copy_from_host(&mut cpu, &weights, weights_out)?;
            self.copy_from_host(&mut cpu, &indices, indices_out)
        })();
        self.cpu = Some(cpu);
        result
    }

    pub(super) fn sampling_on_cpu(
        &mut self,
        token_ids_out: &mut [u32],
        logits: &DeviceBuffer,
        params: &SamplingParams,
    ) -> Result<(), BackendError> {
        warn_fallback_reason(
            "sampling_logits",
            "sampling with temperature is not implemented on the Vulkan device (argmax_rows is)",
        );
        let mut cpu = self.take_cpu();
        let result = (|| {
            let host = self.copy_to_host(&mut cpu, logits)?;
            cpu.stream.sampling_logits(token_ids_out, &host, params)
        })();
        self.cpu = Some(cpu);
        result
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_on_cpu(
        &mut self,
        out: &mut DeviceBuffer,
        q: &DeviceBuffer,
        kv: &DeviceKvCache,
        block_tables: &[i64],
        lengths: &[i64],
        layer_idx: usize,
        params: &AttentionParams,
    ) -> Result<(), BackendError> {
        warn_fallback_reason(
            "attention_paged",
            &format!(
                "head width {}/{} exceeds the Vulkan kernels' 512; the whole KV pool is copied to the host on every call",
                params.head_dim, params.value_dim
            ),
        );
        // the pools must be single allocations
        view_of(&kv.key_pool)?;
        view_of(&kv.value_pool)?;
        let mut cpu = self.take_cpu();
        let result = (|| {
            let host_q = self.copy_to_host(&mut cpu, q)?;
            let host_kv = DeviceKvCache {
                key_pool: self.copy_to_host(&mut cpu, &kv.key_pool)?,
                value_pool: self.copy_to_host(&mut cpu, &kv.value_pool)?,
                ..kv.clone()
            };
            let mut host_out = cpu.backend.allocate_buffer(out.size)?;
            cpu.stream
                .attention_paged(&mut host_out, &host_q, &host_kv, block_tables, lengths, layer_idx, params)?;
            self.copy_from_host(&mut cpu, &host_out, out)
        })();
        self.cpu = Some(cpu);
        result
    }
}
