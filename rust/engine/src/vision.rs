//! Native Rust SigLIP image encoder and Idefics3 modality projector.
use crate::compute::{self, LayerNorm, Linear, Tensor};
use base64::Engine;
use fllm_gguf_probe::MappedGguf;
use serde_json::Value;

struct VisionLayer {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    up: Linear,
    down: Linear,
}
pub struct VisionEncoder {
    patch: Linear,
    position: Tensor,
    layers: Vec<VisionLayer>,
    final_norm: LayerNorm,
    projector: Linear,
    size: usize,
    patch_size: usize,
    hidden: usize,
    heads: usize,
    scale: usize,
    mean: [f32; 3],
    std: [f32; 3],
    pool: rayon::ThreadPool,
    pub output_hidden: usize,
    pub tokens: usize,
}
impl VisionEncoder {
    pub fn load(path: &str, threads: usize, device: &str) -> Result<Self, String> {
        let mapping = MappedGguf::open(path).map_err(|e| e.to_string())?;
        let meta = &mapping.index().metadata;
        if meta.get("clip.projector_type").and_then(Value::as_str) != Some("idefics3") {
            return Err(
                "native Rust vision currently supports SigLIP/Idefics3 projectors (SmolVLM)".into(),
            );
        }
        let integer = |key: &str| -> Result<usize, String> {
            meta.get(key)
                .and_then(Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n > 0)
                .ok_or_else(|| format!("missing/invalid vision metadata {key}"))
        };
        let size = integer("clip.vision.image_size")?;
        let patch_size = integer("clip.vision.patch_size")?;
        let hidden = integer("clip.vision.embedding_length")?;
        let heads = integer("clip.vision.attention.head_count")?;
        let intermediate = integer("clip.vision.feed_forward_length")?;
        let blocks = integer("clip.vision.block_count")?;
        let scale = integer("clip.vision.projector.scale_factor")?;
        let output_hidden = integer("clip.vision.projection_dim")?;
        if size > 1024
            || hidden > 4096
            || blocks > 64
            || size % patch_size != 0
            || (size / patch_size) % scale != 0
            || hidden % heads != 0
        {
            return Err("unsupported vision dimensions".into());
        }
        let grid = size / patch_size;
        let patches = grid * grid;
        let eps = meta
            .get("clip.vision.attention.layer_norm_epsilon")
            .and_then(Value::as_f64)
            .unwrap_or(1e-6) as f32;
        if !eps.is_finite() || eps <= 0.0 {
            return Err("invalid vision layer norm epsilon".into());
        }
        let triple = |key: &str| -> Result<[f32; 3], String> {
            let a = meta
                .get(key)
                .and_then(Value::as_array)
                .filter(|a| a.len() == 3)
                .ok_or_else(|| format!("invalid {key}"))?;
            let mut out = [0.0; 3];
            for i in 0..3 {
                out[i] = a[i].as_f64().ok_or("invalid pixel normalization")? as f32;
                if !out[i].is_finite() {
                    return Err("non-finite pixel normalization".into());
                }
            }
            Ok(out)
        };
        let mean = triple("clip.vision.image_mean")?;
        let std = triple("clip.vision.image_std")?;
        if std.iter().any(|&v| v <= 0.0) {
            return Err("invalid pixel standard deviation".into());
        }
        let mut patch = Tensor::load(&mapping, "v.patch_embd.weight")?;
        if patch.shape != [patch_size, patch_size, 3, hidden] {
            return Err("unsupported patch convolution dimensions".into());
        }
        patch.shape = vec![patch_size * patch_size * 3, hidden];
        let patch = Linear::from_parts(
            patch,
            hidden,
            patch_size * patch_size * 3,
            Some(Tensor::load(&mapping, "v.patch_embd.bias")?.data),
        )?;
        let position = Tensor::load(&mapping, "v.position_embd.weight")?;
        position.matrix(patches, hidden, "vision positions")?;
        let mut layers = Vec::new();
        for i in 0..blocks {
            let p = format!("v.blk.{i}");
            layers.push(VisionLayer {
                q: Linear::load(&mapping, &format!("{p}.attn_q"), hidden, hidden)?,
                k: Linear::load(&mapping, &format!("{p}.attn_k"), hidden, hidden)?,
                v: Linear::load(&mapping, &format!("{p}.attn_v"), hidden, hidden)?,
                out: Linear::load(&mapping, &format!("{p}.attn_out"), hidden, hidden)?,
                norm1: LayerNorm::load(&mapping, &format!("{p}.ln1"), hidden, eps)?,
                norm2: LayerNorm::load(&mapping, &format!("{p}.ln2"), hidden, eps)?,
                // The GGUF CLIP converter names SigLIP fc1 down and fc2 up.
                up: Linear::load(&mapping, &format!("{p}.ffn_down"), intermediate, hidden)?,
                down: Linear::load(&mapping, &format!("{p}.ffn_up"), hidden, intermediate)?,
            });
        }
        let final_norm = LayerNorm::load(&mapping, "v.post_ln", hidden, eps)?;
        let projector = Linear::load(
            &mapping,
            "mm.model.fc",
            output_hidden,
            hidden * scale * scale,
        )?;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads.clamp(1, 256))
            .build()
            .map_err(|e| e.to_string())?;
        let mut vision = Self {
            patch,
            position,
            layers,
            final_norm,
            projector,
            size,
            patch_size,
            hidden,
            heads,
            scale,
            mean,
            std,
            pool,
            output_hidden,
            tokens: patches / (scale * scale),
        };
        vision.set_device(device)?;
        Ok(vision)
    }
    fn set_device(&mut self, device: &str) -> Result<(), String> {
        if matches!(device, "cpu" | "auto") {
            return Ok(());
        }
        #[cfg(feature = "rust-gpu")]
        {
            let gpu = crate::gpu::GpuContext::new(device)?;
            self.patch.enable_gpu(&gpu)?;
            self.projector.enable_gpu(&gpu)?;
            for l in &mut self.layers {
                for linear in [
                    &mut l.q,
                    &mut l.k,
                    &mut l.v,
                    &mut l.out,
                    &mut l.up,
                    &mut l.down,
                ] {
                    linear.enable_gpu(&gpu)?;
                }
            }
            Ok(())
        }
        #[cfg(not(feature = "rust-gpu"))]
        Err("vision GPU projections require --features rust-gpu".into())
    }
    pub fn encode_data_url(&self, url: &str) -> Result<Vec<Vec<f32>>, String> {
        let (header, encoded) = url
            .split_once(',')
            .ok_or("image must be a PNG/JPEG base64 data URL")?;
        if !matches!(header, "data:image/png;base64" | "data:image/jpeg;base64")
            || encoded.len() > 12 * 1024 * 1024
        {
            return Err("image must be a PNG/JPEG base64 data URL up to 9 MiB".into());
        }
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map_err(|_| "invalid image base64")?;
        let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes))
            .with_guessed_format()
            .map_err(|e| e.to_string())?;
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(8192);
        limits.max_image_height = Some(8192);
        limits.max_alloc = Some(128 * 1024 * 1024);
        reader.limits(limits);
        let rgb = reader
            .decode()
            .map_err(|e| format!("cannot decode image: {e}"))?
            .resize_exact(
                self.size as u32,
                self.size as u32,
                image::imageops::FilterType::CatmullRom,
            )
            .to_rgb8();
        let mut pixels = vec![0.0; self.size * self.size * 3];
        for (i, p) in rgb.pixels().enumerate() {
            for c in 0..3 {
                pixels[c * self.size * self.size + i] =
                    (p[c] as f32 / 255.0 - self.mean[c]) / self.std[c];
            }
        }
        self.encode_pixels(&pixels)
    }
    /// Normalized RGB pixels in CHW order. Public for reproducible numerical checks.
    pub fn encode_pixels(&self, pixels: &[f32]) -> Result<Vec<Vec<f32>>, String> {
        if pixels.len() != self.size * self.size * 3 || pixels.iter().any(|x| !x.is_finite()) {
            return Err("invalid vision pixels".into());
        }
        self.pool.install(|| {
            let grid = self.size / self.patch_size;
            let mut patches = Vec::with_capacity(pixels.len());
            for y in 0..grid {
                for x in 0..grid {
                    for c in 0..3 {
                        for dy in 0..self.patch_size {
                            for dx in 0..self.patch_size {
                                patches.push(
                                    pixels[c * self.size * self.size
                                        + (y * self.patch_size + dy) * self.size
                                        + x * self.patch_size
                                        + dx],
                                );
                            }
                        }
                    }
                }
            }
            let mut x = self.patch.forward(&patches)?;
            compute::add(&mut x, &self.position.data);
            for l in &self.layers {
                let mut n = x.clone();
                l.norm1.apply(&mut n);
                let q = l.q.forward(&n)?;
                let k = l.k.forward(&n)?;
                let v = l.v.forward(&n)?;
                let a = compute::attention(&q, &k, &v, self.hidden, self.heads);
                compute::add(&mut x, &l.out.forward(&a)?);
                n = x.clone();
                l.norm2.apply(&mut n);
                let mut up = l.up.forward(&n)?;
                for v in &mut up {
                    *v = compute::gelu(*v);
                }
                compute::add(&mut x, &l.down.forward(&up)?);
            }
            self.final_norm.apply(&mut x);
            let shuffled = pixel_shuffle(&x, grid, self.hidden, self.scale)?;
            let output = self.projector.forward(&shuffled)?;
            if output.iter().any(|v| !v.is_finite()) {
                return Err("non-finite vision projection".into());
            }
            Ok(output
                .chunks_exact(self.output_hidden)
                .map(<[f32]>::to_vec)
                .collect())
        })
    }
}
fn pixel_shuffle(x: &[f32], grid: usize, hidden: usize, scale: usize) -> Result<Vec<f32>, String> {
    if scale == 0 || !grid.is_multiple_of(scale) || x.len() != grid * grid * hidden {
        return Err("invalid pixel shuffle dimensions".into());
    }
    let mut out = Vec::with_capacity(x.len());
    for y in 0..grid / scale {
        for xx in 0..grid / scale {
            for dy in 0..scale {
                for dx in 0..scale {
                    let start = ((y * scale + dy) * grid + xx * scale + dx) * hidden;
                    out.extend_from_slice(&x[start..start + hidden]);
                }
            }
        }
    }
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shuffle_preserves_patch_channel_order() {
        let x = (0..16).map(|n| n as f32).collect::<Vec<_>>();
        assert_eq!(
            pixel_shuffle(&x, 4, 1, 2).unwrap(),
            vec![0., 1., 4., 5., 2., 3., 6., 7., 8., 9., 12., 13., 10., 11., 14., 15.]
        );
        assert!(pixel_shuffle(&x, 4, 1, 3).is_err());
    }
}
