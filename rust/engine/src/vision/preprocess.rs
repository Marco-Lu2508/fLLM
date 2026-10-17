//! Image input handling shared by all vision families: URL and data-URL resolution, decoding with
//! resource limits, resizing, normalisation and patchification.
use base64::Engine;
use image::{imageops::FilterType, DynamicImage, RgbImage};

/// Resource limits of one request, set from the command line (`--max-image-pixels`,
/// `--limit-mm-per-prompt`) and read through [`MediaLimits::get`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MediaLimits {
    /// Largest accepted decoded image (width x height) before any resize.
    pub max_decode_pixels: usize,
    /// Largest image budget handed to the encoder: images are scaled down to fit (families that
    /// have their own dynamic-resolution rule apply the smaller of the two bounds).
    pub max_image_pixels: usize,
    /// Images (and video clips) per prompt.
    pub max_media_per_prompt: usize,
    /// Raw bytes of one image after base64 decoding.
    pub max_image_bytes: usize,
    /// Frames taken from one video clip.
    pub max_video_frames: usize,
    /// Images encoded at the same time (each uses the vision thread pool).
    pub concurrency: usize,
}

impl Default for MediaLimits {
    fn default() -> Self {
        Self {
            max_decode_pixels: 8192 * 8192,
            max_image_pixels: 1_003_520,
            max_media_per_prompt: 4,
            max_image_bytes: 9 * 1024 * 1024,
            max_video_frames: 32,
            concurrency: 2,
        }
    }
}

static LIMITS: std::sync::OnceLock<std::sync::Mutex<MediaLimits>> = std::sync::OnceLock::new();

impl MediaLimits {
    /// The limits in force: defaults, then the environment (`FLLM_MAX_IMAGE_PIXELS`,
    /// `FLLM_MAX_MEDIA_PER_PROMPT`), then whatever [`MediaLimits::set`] stored.
    pub fn get() -> Self {
        *LIMITS
            .get_or_init(|| std::sync::Mutex::new(Self::from_env()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub fn set(limits: Self) {
        *LIMITS
            .get_or_init(|| std::sync::Mutex::new(Self::from_env()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = limits;
    }

    fn from_env() -> Self {
        let mut limits = Self::default();
        let read = |name: &str| std::env::var(name).ok().and_then(|v| v.trim().parse::<usize>().ok());
        if let Some(n) = read("FLLM_MAX_IMAGE_PIXELS").filter(|&n| n >= 784) {
            limits.max_image_pixels = n;
        }
        if let Some(n) = read("FLLM_MAX_MEDIA_PER_PROMPT").filter(|&n| n > 0) {
            limits.max_media_per_prompt = n;
        }
        if let Some(n) = read("FLLM_VISION_CONCURRENCY").filter(|&n| (1..=16).contains(&n)) {
            limits.concurrency = n;
        }
        limits
    }
}

/// Fetches the bytes behind a remote image URL. The server supplies an implementation with SSRF
/// protection when remote images are enabled; without one only data URLs are accepted.
pub trait ImageSource: Send + Sync {
    fn fetch(&self, url: &str) -> Result<Vec<u8>, String>;
}

static IMAGE_SOURCE: std::sync::OnceLock<std::sync::Arc<dyn ImageSource>> = std::sync::OnceLock::new();

/// Installs the fetcher for remote image URLs (the server does this at start-up when remote images
/// are allowed). Returns false when one is installed already.
pub fn set_image_source(source: std::sync::Arc<dyn ImageSource>) -> bool {
    IMAGE_SOURCE.set(source).is_ok()
}

/// Bytes of the image a chat or API request points at: a `data:` URL (any `image/*` or
/// `application/octet-stream` media type, base64) or, when `source` is given, a remote URL.
pub fn resolve_image(url: &str, source: Option<&dyn ImageSource>) -> Result<Vec<u8>, String> {
    let limits = MediaLimits::get();
    if let Some(rest) = url.strip_prefix("data:") {
        let (header, encoded) = rest.split_once(',').ok_or("malformed data URL")?;
        if !header.ends_with(";base64") {
            return Err("image data URLs must be base64 encoded".into());
        }
        let media_type = header.trim_end_matches(";base64");
        if !(media_type.is_empty() || media_type.starts_with("image/") || media_type == "application/octet-stream") {
            return Err(format!("unsupported data URL media type {media_type:?}"));
        }
        if encoded.len() > limits.max_image_bytes / 3 * 4 + 8 {
            return Err(format!("image larger than {} bytes", limits.max_image_bytes));
        }
        return decode_base64(encoded);
    }
    let source = source.or_else(|| IMAGE_SOURCE.get().map(|s| &**s));
    match (url.starts_with("http://") || url.starts_with("https://"), source) {
        (true, Some(source)) => {
            let bytes = source.fetch(url)?;
            if bytes.len() > limits.max_image_bytes {
                return Err(format!("image larger than {} bytes", limits.max_image_bytes));
            }
            Ok(bytes)
        }
        (true, None) => Err("remote image URLs are disabled; send a base64 data URL or start the server with --allow-remote-images".into()),
        _ => Err("image must be a base64 data URL".into()),
    }
}

/// Standard base64 with whitespace tolerated (clients wrap long lines).
pub fn decode_base64(encoded: &str) -> Result<Vec<u8>, String> {
    let compact: String = encoded.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    base64::engine::general_purpose::STANDARD
        .decode(compact.as_bytes())
        .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(compact.as_bytes()))
        .map_err(|_| "invalid base64".to_owned())
}

/// Decodes PNG/JPEG (and the other formats the `image` crate was built with) under the pixel and
/// allocation limits.
pub fn decode_image(bytes: &[u8]) -> Result<RgbImage, String> {
    let limits = MediaLimits::get();
    if bytes.len() > limits.max_image_bytes {
        return Err(format!("image larger than {} bytes", limits.max_image_bytes));
    }
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| e.to_string())?;
    let mut codec = image::Limits::default();
    codec.max_image_width = Some(16384);
    codec.max_image_height = Some(16384);
    codec.max_alloc = Some(256 * 1024 * 1024);
    reader.limits(codec);
    let decoded = reader.decode().map_err(|e| format!("cannot decode image: {e}"))?;
    if decoded.width() as usize * decoded.height() as usize > limits.max_decode_pixels {
        return Err("image has too many pixels".into());
    }
    Ok(flatten_alpha(decoded))
}

/// RGB with transparent areas composited on white (what PIL-based processors do via `convert("RGB")` on
/// RGBA images is to drop alpha; white background is the convention of the model cards).
fn flatten_alpha(image: DynamicImage) -> RgbImage {
    if !image.color().has_alpha() {
        return image.to_rgb8();
    }
    let rgba = image.to_rgba8();
    let mut out = RgbImage::new(rgba.width(), rgba.height());
    for (dst, src) in out.pixels_mut().zip(rgba.pixels()) {
        let a = u32::from(src[3]);
        for c in 0..3 {
            dst[c] = ((u32::from(src[c]) * a + 255 * (255 - a) + 127) / 255) as u8;
        }
    }
    out
}

/// Frames of an animation as PNG files and the frame rate of the selected frames.
pub type SampledFrames = (Vec<Vec<u8>>, Option<f32>);

/// Frames of an animated GIF or APNG as PNG files, at most `max` of them sampled evenly over the
/// animation, with the frame rate of the selected frames. `None` for stills and other formats.
pub fn animated_frames(bytes: &[u8], max: usize) -> Result<Option<SampledFrames>, String> {
    use image::AnimationDecoder;
    const MAX_ANIMATION_FRAMES: usize = 4096;
    let limits = MediaLimits::get();
    let cursor = std::io::Cursor::new(bytes);
    let frames = match image::guess_format(bytes) {
        Ok(image::ImageFormat::Gif) => {
            let decoder = image::codecs::gif::GifDecoder::new(cursor).map_err(|e| e.to_string())?;
            let (w, h) = image::ImageDecoder::dimensions(&decoder);
            if w as usize * h as usize > limits.max_decode_pixels {
                return Err("animation has too many pixels".into());
            }
            let frames = decoder.into_frames().take(MAX_ANIMATION_FRAMES).collect::<Result<Vec<_>, _>>();
            frames.map_err(|e| format!("cannot decode animation: {e}"))?
        }
        Ok(image::ImageFormat::Png) => {
            let decoder = image::codecs::png::PngDecoder::new(cursor).map_err(|e| e.to_string())?;
            if !decoder.is_apng().map_err(|e| e.to_string())? {
                return Ok(None);
            }
            let (w, h) = image::ImageDecoder::dimensions(&decoder);
            if w as usize * h as usize > limits.max_decode_pixels {
                return Err("animation has too many pixels".into());
            }
            let frames = decoder.apng().map_err(|e| e.to_string())?.into_frames().take(MAX_ANIMATION_FRAMES).collect::<Result<Vec<_>, _>>();
            frames.map_err(|e| format!("cannot decode animation: {e}"))?
        }
        _ => return Ok(None),
    };
    if frames.len() < 2 {
        return Ok(None);
    }
    let total = frames.len();
    let take = max.max(1).min(total);
    let chosen: Vec<usize> = (0..take).map(|i| i * total / take).collect();
    let delay_ms: f64 = chosen
        .iter()
        .map(|&i| {
            let (n, d) = frames[i].delay().numer_denom_ms();
            f64::from(n) / f64::from(d.max(1))
        })
        .sum::<f64>()
        / take as f64;
    // the frames not selected stand in for the time between the selected ones
    let fps = (delay_ms > 0.0).then(|| (1000.0 / delay_ms) as f32 * take as f32 / total as f32);
    let mut out = Vec::with_capacity(take);
    for &i in &chosen {
        let mut png = Vec::new();
        image::DynamicImage::ImageRgba8(frames[i].buffer().clone())
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .map_err(|e| e.to_string())?;
        out.push(png);
    }
    Ok(Some((out, fps)))
}

/// Resize to `width` x `height` with the bicubic (Catmull-Rom) kernel, the filter of the reference
/// processors; identical size returns the input unchanged.
pub fn resize(image: &RgbImage, width: u32, height: u32) -> RgbImage {
    if image.width() == width && image.height() == height {
        return image.clone();
    }
    image::imageops::resize(image, width, height, FilterType::CatmullRom)
}

/// Qwen2-VL `smart_resize`: dimensions rounded to multiples of `factor` whose area lies within
/// `[min_pixels, max_pixels]` while keeping the aspect ratio as far as possible.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize), String> {
    if height == 0 || width == 0 || factor == 0 {
        return Err("empty image".into());
    }
    if height.max(width) as f64 / height.min(width) as f64 > 200.0 {
        return Err("image aspect ratio above 200:1 is not supported".into());
    }
    let round = |v: f64| (v / factor as f64).round_ties_even() as usize * factor;
    let (h, w) = (height as f64, width as f64);
    let (mut hb, mut wb) = (round(h).max(factor), round(w).max(factor));
    if hb * wb > max_pixels {
        let beta = (h * w / max_pixels as f64).sqrt();
        hb = factor.max(((h / beta / factor as f64).floor() as usize) * factor);
        wb = factor.max(((w / beta / factor as f64).floor() as usize) * factor);
    } else if hb * wb < min_pixels {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        hb = ((h * beta / factor as f64).ceil() as usize) * factor;
        wb = ((w * beta / factor as f64).ceil() as usize) * factor;
    }
    Ok((hb, wb))
}

/// Flattened patches of Qwen2-VL style inputs.
///
/// `frames` are equally sized RGB frames; consecutive groups of `temporal` frames form one
/// temporal patch (the last group repeats its final frame; a single image is repeated). Rows come
/// in merge-window order (`merge x merge` neighbouring patches are adjacent) and hold
/// `channel x temporal x patch x patch` values, normalised with `mean` / `std`. Returns the patches
/// and the (time, height, width) grid in patch units.
pub fn qwen_patches(
    frames: &[RgbImage],
    patch: usize,
    temporal: usize,
    merge: usize,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<(Vec<f32>, [usize; 3]), String> {
    let first = frames.first().ok_or("no frames")?;
    let (width, height) = (first.width() as usize, first.height() as usize);
    if frames.iter().any(|f| f.width() as usize != width || f.height() as usize != height) {
        return Err("frames differ in size".into());
    }
    if patch == 0 || temporal == 0 || merge == 0 || height % (patch * merge) != 0 || width % (patch * merge) != 0 {
        return Err("image size must be a multiple of patch size x merge size".into());
    }
    let (gh, gw) = (height / patch, width / patch);
    let gt = frames.len().div_ceil(temporal);
    let row_len = 3 * temporal * patch * patch;
    let mut out = vec![0.0f32; gt * gh * gw * row_len];
    // (v / 255 - mean) / std, evaluated as in the reference (rescale first, then normalise)
    let value = |frame: &RgbImage, x: usize, y: usize, c: usize| -> f32 {
        (f32::from(frame.get_pixel(x as u32, y as u32)[c]) / 255.0 - mean[c]) / std[c]
    };
    let blocks_w = gw / merge;
    let blocks_h = gh / merge;
    for t in 0..gt {
        for bh in 0..blocks_h {
            for bw in 0..blocks_w {
                for mh in 0..merge {
                    for mw in 0..merge {
                        let row = ((t * blocks_h + bh) * blocks_w + bw) * merge * merge + mh * merge + mw;
                        let base = row * row_len;
                        let (py0, px0) = ((bh * merge + mh) * patch, (bw * merge + mw) * patch);
                        for c in 0..3 {
                            for ti in 0..temporal {
                                let frame = &frames[(t * temporal + ti).min(frames.len() - 1)];
                                for py in 0..patch {
                                    for px in 0..patch {
                                        out[base + ((c * temporal + ti) * patch + py) * patch + px] =
                                            value(frame, px0 + px, py0 + py, c);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok((out, [gt, gh, gw]))
}

/// Finds the best-fitting grid of tiles `[width_tiles, height_tiles]` from `candidate_grids`
/// for an image of `(width, height)` up to `max_tiles`.
pub fn best_fit_grid(
    width: u32,
    height: u32,
    candidate_grids: &[[usize; 2]],
    max_tiles: usize,
) -> [usize; 2] {
    if candidate_grids.is_empty() {
        return [1, 1];
    }
    let orig_aspect = width as f64 / (height as f64).max(1.0);
    let mut best_grid = candidate_grids[0];
    let mut best_score = f64::INFINITY;
    for &grid in candidate_grids {
        let count = grid[0] * grid[1];
        if count == 0 || count > max_tiles {
            continue;
        }
        let grid_aspect = grid[0] as f64 / grid[1] as f64;
        let aspect_ratio_diff = (orig_aspect / grid_aspect).ln().abs();
        let score = aspect_ratio_diff * 1000.0 + (count as f64);
        if score < best_score {
            best_score = score;
            best_grid = grid;
        }
    }
    best_grid
}

/// Slices an image into `grid[0] x grid[1]` equal rectangular tiles in row-major order.
pub fn slice_into_tiles(image: &RgbImage, grid: [usize; 2]) -> Result<Vec<RgbImage>, String> {
    let (gw, gh) = (grid[0] as u32, grid[1] as u32);
    if gw == 0 || gh == 0 {
        return Err("tile grid dimensions must be non-zero".into());
    }
    let (w, h) = image.dimensions();
    let tile_w = w / gw;
    let tile_h = h / gh;
    if tile_w == 0 || tile_h == 0 {
        return Err("image is smaller than tile grid".into());
    }
    let mut tiles = Vec::with_capacity((gw * gh) as usize);
    for gy in 0..gh {
        for gx in 0..gw {
            let mut tile = RgbImage::new(tile_w, tile_h);
            for y in 0..tile_h {
                for x in 0..tile_w {
                    let p = image.get_pixel(gx * tile_w + x, gy * tile_h + y);
                    tile.put_pixel(x, y, *p);
                }
            }
            tiles.push(tile);
        }
    }
    Ok(tiles)
}

/// Pixel unshuffle (downsample spatial resolution by factor `scale`, packing spatial blocks into channels).
/// Input shape: `h x w x c`. Output shape: `(h/scale) x (w/scale) x (c * scale * scale)`.
pub fn pixel_unshuffle(x: &[f32], h: usize, w: usize, c: usize, scale: usize) -> Result<Vec<f32>, String> {
    if scale == 0 || !h.is_multiple_of(scale) || !w.is_multiple_of(scale) || x.len() != h * w * c {
        return Err("invalid pixel unshuffle dimensions".into());
    }
    let new_h = h / scale;
    let new_w = w / scale;
    let new_c = c * scale * scale;
    let mut out = vec![0.0f32; new_h * new_w * new_c];
    for y in 0..new_h {
        for xx in 0..new_w {
            let out_idx = (y * new_w + xx) * new_c;
            let mut channel_offset = 0;
            for dy in 0..scale {
                for dx in 0..scale {
                    let in_idx = ((y * scale + dy) * w + xx * scale + dx) * c;
                    out[out_idx + channel_offset..out_idx + channel_offset + c]
                        .copy_from_slice(&x[in_idx..in_idx + c]);
                    channel_offset += c;
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
    fn smart_resize_matches_the_reference_rule() {
        // already aligned: unchanged
        assert_eq!(smart_resize(84, 112, 28, 3136, 1_003_520).unwrap(), (84, 112));
        // rounds to the nearest multiple (banker's rounding on exact halves)
        assert_eq!(smart_resize(100, 100, 28, 3136, 1_003_520).unwrap(), (112, 112));
        assert_eq!(smart_resize(42, 42, 28, 3136, 1_003_520).unwrap(), (56, 56));
        // too large: scaled down by the area ratio and floored to the factor
        let (h, w) = smart_resize(4000, 3000, 28, 3136, 1_003_520).unwrap();
        assert!(h * w <= 1_003_520 && h % 28 == 0 && w % 28 == 0);
        assert_eq!((h, w), (1148, 840));
        // too small: scaled up and ceiled
        let (h, w) = smart_resize(20, 30, 28, 3136, 1_003_520).unwrap();
        assert!(h * w >= 3136 && h % 28 == 0 && w % 28 == 0);
        assert!(smart_resize(10, 3000, 28, 3136, 1_003_520).is_err());
    }

    #[test]
    fn patches_follow_merge_window_order() {
        // 2x4 patches of 1 pixel, merge 2: the first row of the output is patch (0,0), then (0,1),
        // (1,0), (1,1) of the first 2x2 block, then the second block
        let mut image = RgbImage::new(4, 2);
        for y in 0..2u32 {
            for x in 0..4u32 {
                image.put_pixel(x, y, image::Rgb([(y * 4 + x) as u8 * 10, 0, 0]));
            }
        }
        let (patches, grid) = qwen_patches(&[image], 1, 2, 2, [0.0; 3], [1.0; 3]).unwrap();
        assert_eq!(grid, [1, 2, 4]);
        let row_len = 3 * 2;
        let reds: Vec<f32> = (0..8).map(|r| (patches[r * row_len] * 255.0).round()).collect();
        assert_eq!(reds, vec![0., 10., 40., 50., 20., 30., 60., 70.]);
        // the single image fills both temporal slots
        assert_eq!(patches[0], patches[1]);
    }

    #[test]
    fn data_urls_resolve_and_remote_is_gated() {
        let png = {
            let mut bytes = Vec::new();
            let image = RgbImage::new(2, 2);
            image
                .write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png)
                .unwrap();
            bytes
        };
        let url = format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(&png));
        assert_eq!(resolve_image(&url, None).unwrap(), png);
        assert!(resolve_image("https://example.invalid/a.png", None).is_err());
        assert!(resolve_image("data:text/plain;base64,AAAA", None).is_err());
        assert!(resolve_image("file:///etc/passwd", None).is_err());
        struct Fixed(Vec<u8>);
        impl ImageSource for Fixed {
            fn fetch(&self, _: &str) -> Result<Vec<u8>, String> {
                Ok(self.0.clone())
            }
        }
        assert_eq!(resolve_image("https://x.invalid/a.png", Some(&Fixed(png.clone()))).unwrap(), png);
        assert_eq!(decode_image(&png).unwrap().dimensions(), (2, 2));
    }

    #[test]
    fn animated_gif_is_sampled_into_frames() {
        let mut gif = Vec::new();
        {
            let mut encoder = image::codecs::gif::GifEncoder::new(&mut gif);
            for shade in [0u8, 80, 160, 240] {
                let frame = image::Frame::from_parts(
                    image::RgbaImage::from_pixel(4, 4, image::Rgba([shade, 0, 0, 255])),
                    0,
                    0,
                    image::Delay::from_numer_denom_ms(100, 1),
                );
                encoder.encode_frame(frame).unwrap();
            }
        }
        let (frames, fps) = animated_frames(&gif, 8).unwrap().unwrap();
        assert_eq!(frames.len(), 4);
        assert!((fps.unwrap() - 10.0).abs() < 0.5);
        assert_eq!(decode_image(&frames[2]).unwrap().get_pixel(0, 0)[0], 160);
        // sampled down: every other frame, half the rate
        let (frames, fps) = animated_frames(&gif, 2).unwrap().unwrap();
        assert_eq!(frames.len(), 2);
        assert!((fps.unwrap() - 5.0).abs() < 0.5);
        assert_eq!(decode_image(&frames[1]).unwrap().get_pixel(0, 0)[0], 160);
        // a still PNG is not an animation
        let mut png = Vec::new();
        RgbImage::new(2, 2).write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).unwrap();
        assert!(animated_frames(&png, 8).unwrap().is_none());
    }

    #[test]
    fn alpha_is_composited_on_white() {
        let mut image = image::RgbaImage::new(1, 1);
        image.put_pixel(0, 0, image::Rgba([0, 0, 0, 0]));
        let flat = flatten_alpha(DynamicImage::ImageRgba8(image));
        assert_eq!(flat.get_pixel(0, 0).0, [255, 255, 255]);
    }
    #[test]
    fn anyres_best_fit_and_slice() {
        let candidates = vec![[1, 1], [1, 2], [2, 1], [2, 2], [1, 3], [3, 1]];
        assert_eq!(best_fit_grid(800, 400, &candidates, 4), [2, 1]);
        assert_eq!(best_fit_grid(400, 800, &candidates, 4), [1, 2]);
        assert_eq!(best_fit_grid(500, 500, &candidates, 4), [1, 1]);

        let mut image = RgbImage::new(4, 4);
        for y in 0..4u32 {
            for x in 0..4u32 {
                image.put_pixel(x, y, image::Rgb([(y * 4 + x) as u8, 0, 0]));
            }
        }
        let tiles = slice_into_tiles(&image, [2, 2]).unwrap();
        assert_eq!(tiles.len(), 4);
        assert_eq!(tiles[0].dimensions(), (2, 2));
        assert_eq!(tiles[0].get_pixel(0, 0)[0], 0);
        assert_eq!(tiles[1].get_pixel(0, 0)[0], 2);
        assert_eq!(tiles[2].get_pixel(0, 0)[0], 8);
        assert_eq!(tiles[3].get_pixel(0, 0)[0], 10);
    }

    #[test]
    fn unshuffle_packs_spatial_blocks_into_channels() {
        let x: Vec<f32> = (0..16).map(|v| v as f32).collect();
        // 4x4 image, 1 channel, scale 2 -> 2x2 image, 4 channels
        let unshuffled = pixel_unshuffle(&x, 4, 4, 1, 2).unwrap();
        assert_eq!(unshuffled.len(), 16);
        // Top-left 2x2 block: [0, 1, 4, 5]
        assert_eq!(&unshuffled[0..4], &[0.0, 1.0, 4.0, 5.0]);
        // Top-right 2x2 block: [2, 3, 6, 7]
        assert_eq!(&unshuffled[4..8], &[2.0, 3.0, 6.0, 7.0]);
    }
}
