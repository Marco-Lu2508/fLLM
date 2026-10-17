//! Insertion of encoded media into a prompt: marker expansion, placeholder positions, embedding
//! overrides and position ids.
use super::mrope::MropePositions;
use super::{EncodedMedia, MediaKind, MultimodalEncoder};

/// Everything the decoder needs besides the token ids to run a prompt with media.
#[derive(Clone, Debug, Default)]
pub struct MediaInputs {
    /// Embedding that replaces the token embedding at a prompt index.
    pub overrides: Vec<(usize, Vec<f32>)>,
    /// Three-axis position ids for families with multimodal rotary embeddings.
    pub mrope: Option<MropePositions>,
    /// DeepStack features per decoder layer: (prompt index, row to add to the hidden state).
    pub deepstack: Vec<Vec<(usize, Vec<f32>)>>,
    /// Custom visible lengths per row in attention (e.g. bidirectional image attention or prefix-LM).
    pub visible_lengths: Option<Vec<usize>>,
}

impl MediaInputs {
    pub fn is_empty(&self) -> bool {
        self.overrides.is_empty()
    }
}

/// A prompt with its placeholders expanded and tokenized.
pub struct MediaPrompt {
    pub prompt: String,
    pub ids: Vec<i64>,
    pub inputs: MediaInputs,
}

/// Replaces every marker in the rendered chat `rendered` by the placeholder tokens of the
/// corresponding item of `media` (markers and items pair up in order, per kind), tokenizes the
/// result with `tokenize` and locates the placeholder tokens. `atomic` resolves a marker text to
/// its single vocabulary id for encoders that do not know it themselves.
pub fn build_prompt(
    encoder: &dyn MultimodalEncoder,
    rendered: &str,
    media: Vec<EncodedMedia>,
    tokenize: &dyn Fn(&str) -> Result<Vec<i64>, String>,
    atomic: &dyn Fn(&str) -> Result<Option<i64>, String>,
) -> Result<MediaPrompt, String> {
    let kinds = [MediaKind::Image, MediaKind::Video];
    let mut found: Vec<(usize, MediaKind, usize)> = Vec::new();
    for kind in kinds {
        let wanted = media.iter().filter(|m| m.kind == kind).count();
        match encoder.marker(kind) {
            Some(marker) => {
                let at: Vec<usize> = rendered.match_indices(marker).map(|(i, _)| i).collect();
                if at.len() != wanted {
                    return Err(format!(
                        "the chat template placed {} {} markers but the request has {} items; marker text in messages is reserved",
                        at.len(),
                        marker,
                        wanted
                    ));
                }
                found.extend(at.into_iter().map(|i| (i, kind, marker.len())));
            }
            None if wanted > 0 => return Err(format!("{} models do not accept {kind:?} input", encoder.family())),
            None => {}
        }
    }
    found.sort_unstable_by_key(|f| f.0);
    // item order of the prompt: the n-th marker of a kind takes the n-th item of that kind
    let mut queues = [Vec::new(), Vec::new()];
    for (i, m) in media.iter().enumerate() {
        queues[usize::from(m.kind == MediaKind::Video)].push(i);
    }
    let mut cursor = [0usize; 2];
    let mut order = Vec::with_capacity(found.len());
    let mut prompt = String::with_capacity(rendered.len());
    let mut last = 0;
    for (at, kind, len) in &found {
        let q = usize::from(*kind == MediaKind::Video);
        let item = queues[q][cursor[q]];
        cursor[q] += 1;
        prompt.push_str(&rendered[last..*at]);
        prompt.push_str(&encoder.expand(&media[item]));
        last = at + len;
        order.push(item);
    }
    prompt.push_str(&rendered[last..]);
    let ids = tokenize(&prompt)?;

    // the items in the order of their markers
    let mut slots: Vec<Option<EncodedMedia>> = media.into_iter().map(Some).collect();
    let ordered = order
        .iter()
        .map(|&i| slots[i].take().ok_or("media item used twice"))
        .collect::<Result<Vec<_>, _>>()?;
    let inputs = place_media(encoder, &ids, ordered, atomic)?;
    Ok(MediaPrompt { prompt, ids, inputs })
}

/// Locates the placeholder tokens of `media` (items in prompt order) in the tokenized prompt `ids`
/// and builds the embedding overrides, DeepStack rows and position ids.
pub fn place_media(
    encoder: &dyn MultimodalEncoder,
    ids: &[i64],
    media: Vec<EncodedMedia>,
    atomic: &dyn Fn(&str) -> Result<Option<i64>, String>,
) -> Result<MediaInputs, String> {
    let kinds = [MediaKind::Image, MediaKind::Video];
    let mut token_of = [None, None];
    for kind in kinds {
        let q = usize::from(kind == MediaKind::Video);
        if let Some(marker) = encoder.marker(kind) {
            token_of[q] = match encoder.token_id(kind) {
                Some(id) => Some(id),
                None => atomic(marker)?,
            };
        }
    }
    // locate the runs of placeholder tokens of every item
    let mut runs_of: Vec<Vec<(usize, usize)>> = vec![Vec::new(); media.len()];
    let mut used = vec![false; ids.len()];
    let mut next_search = [0usize; 2];
    for (item, m) in media.iter().enumerate() {
        let q = usize::from(m.kind == MediaKind::Video);
        let token = token_of[q].ok_or("the vision decoder has no atomic placeholder token")?;
        let count = m.embeddings.len();
        let lengths = if m.runs.is_empty() { vec![count] } else { m.runs.clone() };
        if lengths.iter().sum::<usize>() != count || lengths.contains(&0) {
            return Err("media runs do not add up to the embeddings".into());
        }
        for len in lengths {
            let start = (next_search[q]..ids.len())
                .find(|&p| ids[p] == token && !used[p])
                .ok_or("placeholder tokens are missing from the tokenized prompt")?;
            if start + len > ids.len() || ids[start..start + len].iter().any(|&id| id != token) {
                return Err("placeholder tokens are not contiguous after tokenization".into());
            }
            used[start..start + len].iter_mut().for_each(|u| *u = true);
            next_search[q] = start + len;
            runs_of[item].push((start, len));
        }
    }
    let placeholders: usize = media.iter().map(|m| m.embeddings.len()).sum();
    let total_tokens: usize = token_of
        .iter()
        .flatten()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .map(|t| ids.iter().filter(|&&id| id == t).count())
        .sum();
    if total_tokens != placeholders {
        return Err("image token positions do not match the encoder output".into());
    }
    let mut spans: Vec<(usize, [usize; 3], &EncodedMedia)> = Vec::new();
    for (item, m) in media.iter().enumerate() {
        for &(start, len) in &runs_of[item] {
            // an item in several runs shows one frame per run
            let grid = if runs_of[item].len() > 1 { [1, m.grid[1], m.grid[2]] } else { m.grid };
            debug_assert_eq!(grid.iter().product::<usize>(), len);
            spans.push((start, grid, m));
        }
    }
    let mrope = if media.is_empty() { None } else { encoder.positions(ids.len(), &spans)? };
    let visible_lengths = if media.is_empty() { None } else { encoder.visible_lengths(ids.len(), &spans)? };
    let mut overrides = Vec::new();
    let mut deepstack: Vec<Vec<(usize, Vec<f32>)>> = Vec::new();
    for (item, m) in media.into_iter().enumerate() {
        let mut at = Vec::with_capacity(m.embeddings.len());
        for &(start, len) in &runs_of[item] {
            at.extend(start..start + len);
        }
        for (level, rows) in m.deepstack.into_iter().enumerate() {
            if rows.len() != at.len() {
                return Err("deepstack features do not match the placeholder tokens".into());
            }
            if deepstack.len() <= level {
                deepstack.resize(level + 1, Vec::new());
            }
            deepstack[level].extend(at.iter().copied().zip(rows));
        }
        overrides.extend(at.into_iter().zip(m.embeddings));
    }
    overrides.sort_by_key(|o| o.0);
    Ok(MediaInputs { overrides, mrope, deepstack, visible_lengths })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Toy;
    impl MultimodalEncoder for Toy {
        fn family(&self) -> &'static str {
            "toy"
        }
        fn output_hidden(&self) -> usize {
            2
        }
        fn marker(&self, kind: MediaKind) -> Option<&str> {
            Some(if kind == MediaKind::Image { "<img>" } else { "<vid>" })
        }
        fn expand(&self, media: &EncodedMedia) -> String {
            let word = if media.kind == MediaKind::Image { "IMG" } else { "VID" };
            format!("[ {} ]", vec![word; media.embeddings.len()].join(" "))
        }
        fn token_id(&self, kind: MediaKind) -> Option<i64> {
            Some(if kind == MediaKind::Image { 100 } else { 200 })
        }
        fn encode_image(&self, _: &[u8]) -> Result<EncodedMedia, String> {
            unreachable!()
        }
        fn mrope_layout(&self) -> bool {
            true
        }
        fn positions(&self, total: usize, spans: &[(usize, [usize; 3], &EncodedMedia)]) -> Result<Option<MropePositions>, String> {
            let spans: Vec<_> = spans
                .iter()
                .map(|(s, grid, _)| super::super::mrope::MediaSpan { start: *s, grid: *grid, time_interval: 1.0 })
                .collect();
            super::super::mrope::qwen_positions(total, &spans).map(Some)
        }
    }

    fn tokenize(text: &str) -> Result<Vec<i64>, String> {
        Ok(text
            .split_whitespace()
            .map(|w| match w {
                "IMG" => 100,
                "VID" => 200,
                "[" => 1,
                "]" => 2,
                _ => 5,
            })
            .collect())
    }

    fn media(kind: MediaKind, grid: [usize; 3], tag: f32) -> EncodedMedia {
        let n = grid.iter().product();
        EncodedMedia::new((0..n).map(|i| vec![tag, i as f32]).collect(), grid, kind)
    }

    #[test]
    fn markers_expand_in_order_and_positions_are_found() {
        let items = vec![
            media(MediaKind::Image, [1, 1, 2], 1.0),
            media(MediaKind::Video, [2, 1, 1], 2.0),
            media(MediaKind::Image, [1, 1, 1], 3.0),
        ];
        let rendered = "a <img> b <vid> c <img> d";
        let built = build_prompt(&Toy, rendered, items, &tokenize, &|_| Ok(None)).unwrap();
        assert_eq!(built.ids, vec![5, 1, 100, 100, 2, 5, 1, 200, 200, 2, 5, 1, 100, 2, 5]);
        let at: Vec<usize> = built.inputs.overrides.iter().map(|o| o.0).collect();
        assert_eq!(at, vec![2, 3, 7, 8, 12]);
        assert_eq!(built.inputs.overrides[0].1, vec![1.0, 0.0]);
        assert_eq!(built.inputs.overrides[2].1, vec![2.0, 0.0]);
        assert_eq!(built.inputs.overrides[4].1, vec![3.0, 0.0]);
        let positions = built.inputs.mrope.unwrap();
        assert_eq!(positions.prompt.len(), built.ids.len());
        // text after the first image continues at start (2) + max(h, w) = 4
        assert_eq!(positions.prompt[4], [4, 4, 4]);
    }

    #[test]
    fn items_split_into_runs_keep_their_rows_and_deepstack() {
        // a two-frame clip whose frames are separated by text, with one deepstack level
        struct Framed;
        impl MultimodalEncoder for Framed {
            fn family(&self) -> &'static str {
                "framed"
            }
            fn output_hidden(&self) -> usize {
                2
            }
            fn marker(&self, kind: MediaKind) -> Option<&str> {
                (kind == MediaKind::Video).then_some("<vid>")
            }
            fn expand(&self, _: &EncodedMedia) -> String {
                "[ VID VID ] t [ VID VID ]".into()
            }
            fn token_id(&self, _: MediaKind) -> Option<i64> {
                Some(200)
            }
            fn encode_image(&self, _: &[u8]) -> Result<EncodedMedia, String> {
                unreachable!()
            }
            fn positions(&self, total: usize, spans: &[(usize, [usize; 3], &EncodedMedia)]) -> Result<Option<MropePositions>, String> {
                assert_eq!(spans.iter().map(|s| (s.0, s.1)).collect::<Vec<_>>(), vec![(2, [1, 1, 2]), (7, [1, 1, 2])]);
                super::super::mrope::qwen_positions(
                    total,
                    &spans
                        .iter()
                        .map(|(s, g, _)| super::super::mrope::MediaSpan { start: *s, grid: *g, time_interval: 1.0 })
                        .collect::<Vec<_>>(),
                )
                .map(Some)
            }
        }
        let mut clip = media(MediaKind::Video, [2, 1, 2], 7.0);
        clip.runs = vec![2, 2];
        clip.deepstack = vec![(0..4).map(|i| vec![i as f32, 0.0]).collect()];
        let built = build_prompt(&Framed, "a <vid> b", vec![clip], &tokenize, &|_| Ok(None)).unwrap();
        assert_eq!(built.ids, vec![5, 1, 200, 200, 2, 5, 1, 200, 200, 2, 5]);
        let at: Vec<usize> = built.inputs.overrides.iter().map(|o| o.0).collect();
        assert_eq!(at, vec![2, 3, 7, 8]);
        let ds: Vec<usize> = built.inputs.deepstack[0].iter().map(|o| o.0).collect();
        assert_eq!(ds, at);
        assert_eq!(built.inputs.deepstack[0][3].1, vec![3.0, 0.0]);
    }

    #[test]
    fn marker_count_must_match_the_items() {
        let items = vec![media(MediaKind::Image, [1, 1, 1], 1.0)];
        assert!(build_prompt(&Toy, "no marker", items.clone(), &tokenize, &|_| Ok(None)).is_err());
        assert!(build_prompt(&Toy, "<img> <img>", items, &tokenize, &|_| Ok(None)).is_err());
        // text that looks like a marker without a matching item is rejected too
        assert!(build_prompt(&Toy, "<vid>", Vec::new(), &tokenize, &|_| Ok(None)).is_err());
    }

    #[test]
    fn text_only_prompts_pass_through() {
        let built = build_prompt(&Toy, "a b c", Vec::new(), &tokenize, &|_| Ok(None)).unwrap();
        assert_eq!(built.ids, vec![5, 5, 5]);
        assert!(built.inputs.is_empty());
    }
}
