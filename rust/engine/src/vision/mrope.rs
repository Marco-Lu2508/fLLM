//! Multimodal rotary position embedding (M-RoPE) of the Qwen2-VL family.
//!
//! Every token has three position ids (time, height, width). Text tokens carry the same value on
//! all three axes, so for them the rotation is the ordinary RoPE; image and video tokens carry the
//! grid coordinates. The rotary lanes of a head are split into sections and each section reads its
//! angle from one axis.
use crate::rope::RopeTables;
use serde_json::{Map, Value};

/// How the rotary lanes of a head are assigned to the three position axes.
#[derive(Clone, Debug, PartialEq)]
pub struct MropeLayout {
    /// Lanes per axis (time, height, width); they add up to half the rotary dimensions.
    pub sections: Vec<usize>,
    /// Qwen3-VL: the axes alternate lane by lane (t, h, w, t, h, w, ...) instead of forming
    /// contiguous blocks.
    pub interleaved: bool,
}

impl MropeLayout {
    /// Reads `fllm.rope.mrope_sections` and `fllm.rope.mrope_interleaved` written by the converter.
    pub fn from_metadata(meta: &Map<String, Value>) -> Result<Option<Self>, String> {
        let Some(sections) = meta.get("fllm.rope.mrope_sections").and_then(Value::as_array) else {
            return Ok(None);
        };
        let sections = sections
            .iter()
            .map(|v| v.as_u64().and_then(|n| usize::try_from(n).ok()))
            .collect::<Option<Vec<_>>>()
            .ok_or("invalid fllm.rope.mrope_sections")?;
        if sections.len() < 3 || sections.iter().sum::<usize>() == 0 {
            return Err("fllm.rope.mrope_sections needs time, height and width entries".into());
        }
        let interleaved = meta
            .get("fllm.rope.mrope_interleaved")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        Ok(Some(Self { sections, interleaved }))
    }

    /// The sections must cover exactly the rotary lanes of a head (`rope_dims / 2`).
    pub fn check(&self, rope_dims: usize) -> Result<(), String> {
        let total: usize = self.sections.iter().take(3).sum();
        if total * 2 != rope_dims {
            return Err(format!(
                "mrope sections {:?} do not cover {rope_dims} rotary dimensions",
                self.sections
            ));
        }
        Ok(())
    }

    /// Position axis (0 time, 1 height, 2 width) read by rotary lane `lane`.
    pub fn axis(&self, lane: usize) -> usize {
        if self.interleaved {
            let limit = |axis: usize| self.sections[axis] * 3;
            return match lane % 3 {
                1 if lane < limit(1) => 1,
                2 if lane < limit(2) => 2,
                _ => 0,
            };
        }
        let mut end = 0;
        for (axis, &len) in self.sections.iter().take(3).enumerate() {
            end += len;
            if lane < end {
                return axis;
            }
        }
        2
    }

    /// Axis of every lane for `half` rotary pairs.
    pub fn axes(&self, half: usize) -> Vec<u8> {
        (0..half).map(|lane| self.axis(lane) as u8).collect()
    }
}

/// Rotates the first `rope_dims` values of every head of `rows` rows; `positions[r]` holds the
/// (time, height, width) ids of row `r` and `axes` the axis of every rotary lane.
#[allow(clippy::too_many_arguments)]
pub fn apply_rope_mrope(
    x: &mut [f32],
    heads: usize,
    dim: usize,
    rope_dims: usize,
    neox: bool,
    positions: &[[u32; 3]],
    axes: &[u8],
    tables: &RopeTables,
) {
    let half = tables.half;
    debug_assert_eq!(half * 2, rope_dims);
    debug_assert_eq!(axes.len(), half);
    for (r, position) in positions.iter().enumerate() {
        for h in 0..heads {
            let head = &mut x[(r * heads + h) * dim..(r * heads + h) * dim + rope_dims];
            for d in 0..half {
                let p = position[usize::from(axes[d])] as usize;
                let (cos, sin) = (tables.cos[p * half + d], tables.sin[p * half + d]);
                let (a_index, b_index) = if neox { (d, d + half) } else { (2 * d, 2 * d + 1) };
                let (a, b) = (head[a_index], head[b_index]);
                head[a_index] = a * cos - b * sin;
                head[b_index] = b * cos + a * sin;
            }
        }
    }
}

/// Where the placeholder tokens of one image or video sit in the prompt.
#[derive(Clone, Debug, PartialEq)]
pub struct MediaSpan {
    /// Index of the first placeholder token.
    pub start: usize,
    /// Token grid after merging: (time, height, width); the span holds `t * h * w` tokens.
    pub grid: [usize; 3],
    /// Increment of the time position per temporal step (1 for images; videos of Qwen2.5-VL use
    /// seconds per step x tokens per second). The product is truncated, as in the reference.
    pub time_interval: f32,
}

/// Position ids of a prompt with media and the offset that continues them during generation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MropePositions {
    pub prompt: Vec<[u32; 3]>,
    /// Added to the token index of generated tokens (on all axes): the media tokens occupy fewer
    /// position ids than tokens, so this is usually negative.
    pub delta: i64,
}

impl MropePositions {
    /// Position ids of the token at `index` of the whole sequence.
    pub fn at(&self, index: usize) -> [u32; 3] {
        match self.prompt.get(index) {
            Some(p) => *p,
            None => {
                let p = (index as i64 + self.delta).max(0) as u32;
                [p; 3]
            }
        }
    }
}

/// The `get_rope_index` rule of Qwen2-VL: text advances one id per token on all axes; a media span
/// starts at the current id, its tokens take (start + t * interval, start + row, start + column),
/// and the text after it continues at start + max(height, width).
pub fn qwen_positions(total: usize, spans: &[MediaSpan]) -> Result<MropePositions, String> {
    let mut prompt = Vec::with_capacity(total);
    let mut current: u32 = 0;
    for span in spans {
        let [t, h, w] = span.grid;
        let count = t * h * w;
        if span.start < prompt.len() || span.start + count > total || count == 0 {
            return Err("media spans overlap or exceed the prompt".into());
        }
        while prompt.len() < span.start {
            prompt.push([current; 3]);
            current += 1;
        }
        for ti in 0..t {
            for hi in 0..h {
                for wi in 0..w {
                    prompt.push([
                        current + (ti as f32 * span.time_interval) as u32,
                        current + hi as u32,
                        current + wi as u32,
                    ]);
                }
            }
        }
        current += h.max(w) as u32;
    }
    while prompt.len() < total {
        prompt.push([current; 3]);
        current += 1;
    }
    let largest = prompt.iter().flat_map(|p| p.iter().copied()).max().unwrap_or(0);
    Ok(MropePositions {
        delta: i64::from(largest) + 1 - total as i64,
        prompt,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn sections_map_lanes_to_axes() {
        let layout = MropeLayout { sections: vec![2, 3, 3], interleaved: false };
        assert_eq!(layout.axes(8), vec![0, 0, 1, 1, 1, 2, 2, 2]);
        let layout = MropeLayout { sections: vec![4, 2, 2], interleaved: true };
        assert_eq!(layout.axes(8), vec![0, 1, 2, 0, 1, 2, 0, 0]);
    }

    #[test]
    fn layout_must_cover_the_rotary_dimensions() {
        let layout = MropeLayout { sections: vec![16, 24, 24], interleaved: false };
        assert!(layout.check(128).is_ok());
        assert!(layout.check(64).is_err());
    }

    #[test]
    fn layout_from_metadata() {
        let meta = json!({"fllm.rope.mrope_sections": [16, 24, 24]});
        let layout = MropeLayout::from_metadata(meta.as_object().unwrap()).unwrap().unwrap();
        assert_eq!(layout.sections, vec![16, 24, 24]);
        assert!(!layout.interleaved);
        assert!(MropeLayout::from_metadata(&Map::new()).unwrap().is_none());
        let bad = json!({"fllm.rope.mrope_sections": [1, 2]});
        assert!(MropeLayout::from_metadata(bad.as_object().unwrap()).is_err());
    }

    #[test]
    fn image_positions_follow_the_grid() {
        // 2 text tokens, a 2x3 image (6 tokens), 2 text tokens
        let spans = [MediaSpan { start: 2, grid: [1, 2, 3], time_interval: 1.0 }];
        let p = qwen_positions(10, &spans).unwrap();
        assert_eq!(p.prompt[0], [0, 0, 0]);
        assert_eq!(p.prompt[1], [1, 1, 1]);
        assert_eq!(p.prompt[2], [2, 2, 2]);
        assert_eq!(p.prompt[4], [2, 2, 4]);
        assert_eq!(p.prompt[5], [2, 3, 2]);
        assert_eq!(p.prompt[7], [2, 3, 4]);
        // text continues at 2 + max(2, 3) = 5
        assert_eq!(p.prompt[8], [5, 5, 5]);
        assert_eq!(p.prompt[9], [6, 6, 6]);
        assert_eq!(p.delta, 7 - 10);
        assert_eq!(p.at(10), [7, 7, 7]);
        assert_eq!(p.at(3), [2, 2, 3]);
    }

    #[test]
    fn video_time_axis_uses_the_interval() {
        let spans = [MediaSpan { start: 0, grid: [2, 1, 1], time_interval: 4.0 }];
        let p = qwen_positions(3, &spans).unwrap();
        assert_eq!(p.prompt, vec![[0, 0, 0], [4, 0, 0], [1, 1, 1]]);
        assert_eq!(p.delta, 5 - 3);
        assert!(qwen_positions(1, &spans).is_err());
    }

    #[test]
    fn mrope_with_equal_axes_matches_plain_rope() {
        let half = 4;
        let tables = RopeTables {
            half,
            cos: (0..6 * half).map(|i| (i as f32 * 0.37).cos()).collect(),
            sin: (0..6 * half).map(|i| (i as f32 * 0.37).sin()).collect(),
        };
        let original: Vec<f32> = (0..2 * 2 * 8).map(|i| (i as f32 * 0.11).sin()).collect();
        let layout = MropeLayout { sections: vec![1, 1, 2], interleaved: false };
        let mut a = original.clone();
        apply_rope_mrope(&mut a, 2, 8, 8, true, &[[3, 3, 3], [5, 5, 5]], &layout.axes(half), &tables);
        let mut b = original;
        for (r, p) in [3usize, 5].into_iter().enumerate() {
            for h in 0..2 {
                let head = &mut b[(r * 2 + h) * 8..(r * 2 + h) * 8 + 8];
                for d in 0..half {
                    let (c, s) = (tables.cos[p * half + d], tables.sin[p * half + d]);
                    let (x, y) = (head[d], head[d + half]);
                    head[d] = x * c - y * s;
                    head[d + half] = y * c + x * s;
                }
            }
        }
        assert_eq!(a, b);
    }
}
