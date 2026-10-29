//! fLLM's own WGSL matrix-vector/batch projection kernel and Rust GPU driver.
//! wgpu supplies device/command transport, never model inference.
use std::{sync::Arc, time::Duration};
use wgpu::util::DeviceExt;

const LINEAR_SHADER: &str = r#"
struct Shape { rows: u32, cols: u32, batch: u32, kind: u32 }
@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> shape: Shape;
var<workgroup> partial: array<f32, 64>;
fn byte_at(offset: u32) -> u32 { return (weights[offset / 4u] >> ((offset % 4u) * 8u)) & 255u; }
fn half_at(offset: u32) -> f32 { return unpack2x16float(byte_at(offset) | (byte_at(offset + 1u) << 8u)).x; }
fn signed_byte(value: u32) -> f32 { return f32(i32(value << 24u) >> 24u); }
fn weight_at(index: u32) -> f32 {
    if (shape.kind == 0u) { return bitcast<f32>(weights[index]); }
    if (shape.kind < 8u) {
        let affine = shape.kind == 3u || shape.kind == 7u; let five = shape.kind == 6u || shape.kind == 7u;
        let offset = select(2u, 4u, affine); let qoffset = offset + select(0u, 4u, five);
        let size = qoffset + 16u; let start = (index / 32u) * size; let j = index % 32u;
        let byte = byte_at(start + qoffset + j % 16u); var q = (byte >> select(0u, 4u, j >= 16u)) & 15u;
        if (five) { q |= ((byte_at(start + offset + j / 8u) >> (j % 8u)) & 1u) << 4u; }
        if (affine) { return half_at(start) * f32(q) + half_at(start + 2u); }
        return half_at(start) * (f32(q) - select(8.0, 16.0, five));
    }
    if (shape.kind == 8u) {
        let start = (index / 32u) * 34u;
        return half_at(start) * signed_byte(byte_at(start + 2u + index % 32u));
    }
    let local = index % 256u;
    if (shape.kind == 12u || shape.kind == 13u) {
        let five = shape.kind == 13u; let start = (index / 256u) * select(144u, 176u, five); let group = local / 32u;
        var scale: u32; var minimum: u32;
        if (group < 4u) { scale = byte_at(start + 4u + group) & 63u; minimum = byte_at(start + 8u + group) & 63u; }
        else { scale = (byte_at(start + 8u + group) & 15u) | ((byte_at(start + group) >> 6u) << 4u);
            minimum = (byte_at(start + 8u + group) >> 4u) | ((byte_at(start + 4u + group) >> 6u) << 4u); }
        var q = (byte_at(start + select(16u, 48u, five) + (group / 2u) * 32u + local % 32u) >> ((group % 2u) * 4u)) & 15u;
        if (five) { q |= ((byte_at(start + 16u + local % 32u) >> group) & 1u) << 4u; }
        return half_at(start) * f32(scale) * f32(q) - half_at(start + 2u) * f32(minimum);
    }
    let start = (index / 256u) * 210u; let n = local / 128u; let group = (local % 128u) / 32u; let j = local % 32u;
    let low = (byte_at(start + n * 64u + (group % 2u) * 32u + j) >> ((group / 2u) * 4u)) & 15u;
    let high = (byte_at(start + 128u + n * 32u + j) >> (group * 2u)) & 3u;
    let scale = signed_byte(byte_at(start + 192u + n * 8u + group * 2u + j / 16u));
    return half_at(start + 208u) * scale * (f32(low | (high << 4u)) - 32.0);
}
@compute @workgroup_size(64)
fn linear(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let row = group.x; let batch = group.y;
    var sum = 0.0;
    for (var col = lane; col < shape.cols; col += 64u) {
        sum += weight_at(row * shape.cols + col) * input[batch * shape.cols + col];
    }
    partial[lane] = sum; workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if (lane < stride) { partial[lane] += partial[lane + stride]; }
        workgroupBarrier();
    }
    if (lane == 0u) { output[batch * shape.rows + row] = partial[0]; }
}
"#;

pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    pub name: String,
}
impl GpuContext {
    pub fn new(backend: &str) -> Result<Arc<Self>, String> {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = match backend {
            "vulkan" => wgpu::Backends::VULKAN,
            "metal" => wgpu::Backends::METAL,
            "dx12" => wgpu::Backends::DX12,
            "opengl" | "gl" => wgpu::Backends::GL,
            "gpu" | "wgpu" => wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
            _ => return Err(format!("unsupported Rust GPU backend {backend}")),
        };
        let instance = wgpu::Instance::new(descriptor);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            ..Default::default()
        }))
        .map_err(|e| format!("no native GPU adapter: {e}"))?;
        let info = adapter.get_info();
        if matches!(
            info.device_type,
            wgpu::DeviceType::Cpu | wgpu::DeviceType::Other
        ) {
            return Err(format!(
                "{} is not a hardware GPU; refusing silent CPU fallback",
                info.name
            ));
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("fllm-rust-compute"),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .map_err(|e| format!("cannot initialize GPU: {e}"))?;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fllm-linear"),
            source: wgpu::ShaderSource::Wgsl(LINEAR_SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fllm-linear"),
            layout: None,
            module: &module,
            entry_point: Some("linear"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(format!("GPU kernel validation failed: {e}"));
        }
        let name = format!("{} ({:?})", info.name, info.backend);
        eprintln!("[fllm-rs] Own Rust/WGSL projection backend: {name}");
        Ok(Arc::new(Self {
            device,
            queue,
            pipeline,
            name,
        }))
    }
    pub fn upload(
        self: &Arc<Self>,
        weights: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<GpuLinear, String> {
        let size = weights
            .len()
            .checked_mul(4)
            .ok_or("GPU weight size overflow")?;
        if size > self.device.limits().max_storage_buffer_binding_size as usize
            || weights.len() != rows * cols
        {
            return Err("projection exceeds GPU storage buffer limits".into());
        }
        let weight = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fllm-resident-weight"),
                contents: bytemuck::cast_slice(weights),
                usage: wgpu::BufferUsages::STORAGE,
            });
        Ok(GpuLinear {
            context: self.clone(),
            weight,
            rows,
            cols,
            kind: 0,
        })
    }
    pub fn upload_chunks(
        self: &Arc<Self>,
        weights: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<GpuLinear>, String> {
        let chunk_rows = (self.device.limits().max_storage_buffer_binding_size as usize
            / (cols * 4))
            .min(self.device.limits().max_compute_workgroups_per_dimension as usize);
        if chunk_rows == 0 {
            return Err("GPU cannot hold one projection row".into());
        }
        if weights.len() != rows * cols {
            return Err("invalid projection dimensions".into());
        }
        weights
            .chunks(chunk_rows * cols)
            .map(|chunk| self.upload(chunk, chunk.len() / cols, cols))
            .collect()
    }
    pub fn upload_quantized_chunks(
        self: &Arc<Self>,
        packed: &[u8],
        kind: u32,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<GpuLinear>, String> {
        let (block, size) = match kind {
            2 => (32, 18),
            3 => (32, 20),
            6 => (32, 22),
            7 => (32, 24),
            8 => (32, 34),
            12 => (256, 144),
            13 => (256, 176),
            14 => (256, 210),
            _ => return Err("unsupported GPU quantization".into()),
        };
        if !cols.is_multiple_of(block) {
            return Err("invalid quantized GPU row length".into());
        }
        let row_bytes = cols / block * size;
        if packed.len() != rows * row_bytes {
            return Err("invalid GPU packed weight dimensions".into());
        }
        let chunk_rows = (self.device.limits().max_storage_buffer_binding_size as usize
            / row_bytes)
            .min(self.device.limits().max_compute_workgroups_per_dimension as usize);
        if chunk_rows == 0 {
            return Err("GPU cannot hold one packed projection row".into());
        }
        packed
            .chunks(chunk_rows * row_bytes)
            .map(|chunk| {
                let mut bytes = chunk.to_vec();
                bytes.resize(bytes.len().div_ceil(4) * 4, 0);
                let weight = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("fllm-quantized-weight"),
                        contents: &bytes,
                        usage: wgpu::BufferUsages::STORAGE,
                    });
                Ok(GpuLinear {
                    context: self.clone(),
                    weight,
                    rows: chunk.len() / row_bytes,
                    cols,
                    kind,
                })
            })
            .collect()
    }
}
pub struct GpuLinear {
    context: Arc<GpuContext>,
    weight: wgpu::Buffer,
    rows: usize,
    cols: usize,
    kind: u32,
}
impl GpuLinear {
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        if input.is_empty() || !input.len().is_multiple_of(self.cols) {
            return Err("invalid GPU projection input".into());
        }
        let batch = input.len() / self.cols;
        let device = &self.context.device;
        let limits = device.limits();
        let output_size = batch
            .checked_mul(self.rows)
            .and_then(|n| n.checked_mul(4))
            .ok_or("GPU output size overflow")? as u64;
        if output_size > limits.max_storage_buffer_binding_size
            || input.len() * 4 > limits.max_storage_buffer_binding_size as usize
            || self.rows > limits.max_compute_workgroups_per_dimension as usize
            || batch > limits.max_compute_workgroups_per_dimension as usize
        {
            return Err("GPU projection exceeds device limits".into());
        }
        let x = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fllm-input"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let shape = [self.rows as u32, self.cols as u32, batch as u32, self.kind];
        let shape = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fllm-shape"),
            contents: bytemuck::cast_slice(&shape),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fllm-output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fllm-readback"),
            size: output_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fllm-linear"),
            layout: &self.context.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.weight.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: shape.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fllm-compute"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fllm-linear"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.context.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(self.rows as u32, batch as u32, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_size);
        let index = self.context.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(index),
                timeout: Some(Duration::from_secs(30)),
            })
            .map_err(|e| format!("GPU submission failed: {e}"))?;
        rx.recv_timeout(Duration::from_secs(1))
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        let data = readback
            .slice(..)
            .get_mapped_range()
            .map_err(|e| e.to_string())?;
        let result = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
        drop(data);
        readback.unmap();
        if result.iter().any(|v| !v.is_finite()) {
            return Err("GPU projection produced non-finite output".into());
        }
        Ok(result)
    }
}

/// Hardware adapters reachable through wgpu, numbered from `first_index`.
pub fn list_adapters(first_index: usize) -> Vec<crate::devices::DeviceInfo> {
    use crate::devices::{DeviceInfo, DeviceType};
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12 | wgpu::Backends::GL;
    let backends = descriptor.backends;
    let instance = wgpu::Instance::new(descriptor);
    let all = pollster::block_on(instance.enumerate_adapters(backends));
    // The same graphics card often shows up twice, through its Vulkan / DirectX / Metal driver and through OpenGL: OpenGL is the
    // fallback for cards that have no other driver, so it is listed only for adapters that nothing else serves.
    let served: Vec<String> = all
        .iter()
        .map(wgpu::Adapter::get_info)
        .filter(|i| i.backend != wgpu::Backend::Gl && !matches!(i.device_type, wgpu::DeviceType::Cpu | wgpu::DeviceType::Other))
        .map(|i| i.name)
        .collect();
    all.into_iter()
        .filter(|adapter| {
            let info = adapter.get_info();
            info.backend != wgpu::Backend::Gl || !served.contains(&info.name)
        })
        .filter_map(|adapter| {
            let info = adapter.get_info();
            let dev_type = match info.device_type {
                wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::VirtualGpu => DeviceType::Gpu,
                wgpu::DeviceType::IntegratedGpu => DeviceType::IntegratedGpu,
                wgpu::DeviceType::Cpu | wgpu::DeviceType::Other => return None,
            };
            let (free, total) = adapter_memory(&adapter);
            Some((info, dev_type, free, total))
        })
        .enumerate()
        .map(|(offset, (info, dev_type, free, total))| DeviceInfo {
            index: first_index + offset,
            name: info.name.clone(),
            description: format!("{:?}, driver {}", info.backend, info.driver),
            dev_type,
            free_memory_bytes: free,
            total_memory_bytes: total,
        })
        .collect()
}

/// `(free, total)` bytes of device-local memory. Vulkan reports the heap sizes; they are used for
/// both values because the free amount is not queried (another process may hold part of it, so
/// the planner's utilisation factor is the safety margin). Other backends report `(0, 0)`.
#[cfg(not(target_vendor = "apple"))]
fn adapter_memory(adapter: &wgpu::Adapter) -> (u64, u64) {
    // SAFETY: the raw Vulkan handles are only read to query memory properties and are not kept.
    unsafe {
        let Some(hal) = adapter.as_hal::<wgpu::hal::api::Vulkan>() else {
            return (0, 0);
        };
        let properties = hal
            .shared_instance()
            .raw_instance()
            .get_physical_device_memory_properties(hal.raw_physical_device());
        let total: u64 = properties.memory_heaps[..properties.memory_heap_count as usize]
            .iter()
            .filter(|heap| heap.flags.as_raw() & 1 != 0) // VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
            .map(|heap| heap.size)
            .sum();
        (total, total)
    }
}

#[cfg(target_vendor = "apple")]
fn adapter_memory(_adapter: &wgpu::Adapter) -> (u64, u64) {
    (0, 0)
}
