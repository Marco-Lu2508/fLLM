//! Native Rust image encoders: the [`MultimodalEncoder`] interface, the SigLIP/Idefics3 projector
//! of SmolVLM and (in the submodules) the Qwen-VL tower, M-RoPE positions and shared image
//! preprocessing.
use crate::compute::{self, LayerNorm, Linear, Tensor};
use fllm_gguf_probe::MappedGguf;
use serde_json::Value;

pub mod audio;
pub mod mrope;
pub mod pipeline;
pub mod preprocess;
pub mod qwen_vl;
pub mod siglip;

use mrope::MropePositions;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaKind {
    Image,
    Video,
}

/// The embeddings of one image or clip and the token grid they occupy.
#[derive(Clone, Debug)]
pub struct EncodedMedia {
    /// One embedding per placeholder token, in grid order.
    pub embeddings: Vec<Vec<f32>>,
    /// Token grid (time, height, width) after merging; the product is `embeddings.len()`.
    pub grid: [usize; 3],
    pub kind: MediaKind,
    /// Frame rate of a clip (some families use it for time positions).
    pub fps: Option<f32>,
    /// DeepStack features (Qwen3-VL): per level, one row per placeholder token, added to the hidden
    /// state of decoder layer `level` at the placeholder positions.
    pub deepstack: Vec<Vec<Vec<f32>>>,
    /// Lengths of the runs of placeholder tokens when the item is split by text (Qwen3-VL video
    /// frames); empty means one run holding every embedding.
    pub runs: Vec<usize>,
    /// Seconds announced before each run of a clip (Qwen3-VL).
    pub timestamps: Vec<f32>,
}

impl EncodedMedia {
    pub fn new(embeddings: Vec<Vec<f32>>, grid: [usize; 3], kind: MediaKind) -> Self {
        Self { embeddings, grid, kind, fps: None, deepstack: Vec::new(), runs: Vec::new(), timestamps: Vec::new() }
    }
}

/// A vision (or audio) tower with its preprocessing: bytes in, embeddings plus the prompt text
/// that reserves their positions out. The chat template renders one `marker` per media item; the
/// server replaces it with `expand` and the decoder overwrites the token embeddings at the
/// resulting `token_id` positions.
pub trait MultimodalEncoder: Send + Sync {
    fn family(&self) -> &'static str;
    /// Width of the embeddings (must equal the hidden size of the language model).
    fn output_hidden(&self) -> usize;
    /// Text the chat template emits for one item of this kind; `None` when unsupported.
    fn marker(&self, kind: MediaKind) -> Option<&str>;
    /// Text that replaces the marker once the number of tokens is known.
    fn expand(&self, media: &EncodedMedia) -> String;
    /// Vocabulary id whose positions receive the embeddings.
    fn token_id(&self, kind: MediaKind) -> Option<i64>;
    fn encode_image(&self, bytes: &[u8]) -> Result<EncodedMedia, String>;
    /// A clip given as encoded frames (PNG/JPEG/GIF) at `fps`.
    fn encode_video(&self, _frames: &[Vec<u8>], _fps: f32) -> Result<EncodedMedia, String> {
        Err(format!("{} models do not accept video", self.family()))
    }
    /// Whether the language model needs three-axis (M-RoPE) position ids for media prompts.
    fn mrope_layout(&self) -> bool {
        false
    }
    /// Position ids for a prompt of `total` tokens. Every span is (index of its first placeholder
    /// token, its token grid, the item it belongs to); an item split into several runs of tokens
    /// contributes one span per run.
    fn positions(&self, _total: usize, _spans: &[(usize, [usize; 3], &EncodedMedia)]) -> Result<Option<MropePositions>, String> {
        Ok(None)
    }
    /// Custom visible lengths per token for bidirectional attention (Gemma 3 image tokens or PaliGemma prefix-LM).
    fn visible_lengths(&self, _total: usize, _spans: &[(usize, [usize; 3], &EncodedMedia)]) -> Result<Option<Vec<usize>>, String> {
        Ok(None)
    }
}

/// The vision tower stored in a model GGUF by `fllm-rs convert` (`fllm.vision.*` metadata and
/// `v.hf.*` tensors), or `None` when the file has none or its family has no encoder in this build.
pub fn load_from_model(
    mapping: &MappedGguf,
    threads: usize,
    device: &str,
) -> Result<Option<std::sync::Arc<dyn MultimodalEncoder>>, String> {
    let meta = &mapping.index().metadata;
    let Some(family) = meta.get("fllm.vision.family").and_then(Value::as_str) else {
        return Ok(None);
    };
    let config: Value = serde_json::from_str(
        meta.get("fllm.vision.config")
            .and_then(Value::as_str)
            .ok_or("model has fllm.vision.family but no fllm.vision.config")?,
    )
    .map_err(|e| format!("invalid fllm.vision.config: {e}"))?;
    match family {
        "qwen2_vl" | "qwen2_5_vl" | "qwen3_vl" => Ok(Some(std::sync::Arc::new(qwen_vl::QwenVlEncoder::load(
            mapping, family, &config, threads, device,
        )?))),
        "paligemma" | "gemma3" | "mllama" | "llava" => Ok(Some(std::sync::Arc::new(siglip::SiglipEncoder::load(
            mapping, family, &config, threads, device,
        )?))),
        "whisper" => Ok(Some(std::sync::Arc::new(audio::WhisperMultimodalEncoder::load(
            mapping, family, &config, threads, device,
        )?))),
        other => Err(format!("unsupported vision family {other:?} in the model file")),
    }
}

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
        if !crate::encoder::helper_model_uses_gpu(device) {
            eprintln!("[fllm-rs] this model part runs on the CPU ({device} is used for the language model only)");
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
        self.encode_bytes(&preprocess::resolve_image(url, None)?)
    }
    /// Decodes an image (any format the `image` crate was built with), scales it to the square
    /// input of the tower and encodes it.
    pub fn encode_bytes(&self, bytes: &[u8]) -> Result<Vec<Vec<f32>>, String> {
        let rgb = image::DynamicImage::ImageRgb8(preprocess::decode_image(bytes)?)
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
impl MultimodalEncoder for VisionEncoder {
    fn family(&self) -> &'static str {
        "idefics3"
    }
    fn output_hidden(&self) -> usize {
        self.output_hidden
    }
    fn marker(&self, kind: MediaKind) -> Option<&str> {
        (kind == MediaKind::Image).then_some("<image>")
    }
    fn expand(&self, media: &EncodedMedia) -> String {
        format!(
            "<fake_token_around_image><global-img>{}<fake_token_around_image>",
            "<image>".repeat(media.embeddings.len())
        )
    }
    fn token_id(&self, _kind: MediaKind) -> Option<i64> {
        None
    }
    fn encode_image(&self, bytes: &[u8]) -> Result<EncodedMedia, String> {
        let embeddings = self.encode_bytes(bytes)?;
        let side = (embeddings.len() as f64).sqrt() as usize;
        let grid = [1, side, embeddings.len() / side.max(1)];
        Ok(EncodedMedia::new(embeddings, grid, MediaKind::Image))
    }
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
