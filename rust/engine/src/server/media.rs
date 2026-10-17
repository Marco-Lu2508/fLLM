//! Media parts of chat requests (images, video clips): collection from the message list, encoding
//! on blocking threads under the vision admission limit, and prompt construction.
use super::{chat_environment, AppState};
use crate::vision::pipeline::{self, MediaInputs};
use crate::vision::preprocess::{self, MediaLimits};
use crate::vision::{EncodedMedia, MediaKind, MultimodalEncoder};
use serde_json::{json, Value};
use std::sync::Arc;

/// One image or clip of a request, before decoding.
enum MediaRequest {
    Image { url: String },
    /// Frames as URLs (data URLs or remote), or one URL of an animated image.
    Video { urls: Vec<String>, fps: f32 },
}

impl MediaRequest {
    fn kind(&self) -> MediaKind {
        match self {
            Self::Image { .. } => MediaKind::Image,
            Self::Video { .. } => MediaKind::Video,
        }
    }
}

/// The content-part types that carry media.
fn is_media_part(kind: &str) -> bool {
    matches!(kind, "image_url" | "input_image" | "image" | "video_url" | "video")
}

/// True when any message of the request holds an image or video part.
pub(super) fn request_has_media(messages: &[Value]) -> bool {
    messages.iter().any(|m| {
        m.get("content")
            .and_then(Value::as_array)
            .is_some_and(|parts| parts.iter().any(|p| p.get("type").and_then(Value::as_str).is_some_and(is_media_part)))
    })
}

fn url_of(part: &Value, key: &str) -> Option<String> {
    let value = part.get(key)?;
    value
        .as_str()
        .or_else(|| value.get("url").and_then(Value::as_str))
        .map(str::to_owned)
}

fn video_request(part: &Value) -> Result<MediaRequest, String> {
    let fps = part
        .get("fps")
        .or_else(|| part.get("video_url").and_then(|v| v.get("fps")))
        .and_then(Value::as_f64)
        .map_or(2.0, |f| f as f32);
    if !(fps.is_finite() && fps > 0.0 && fps <= 240.0) {
        return Err("video fps must be between 0 and 240".into());
    }
    // {"type":"video","video":[frame, ...]} (Qwen style) or {"type":"video_url","video_url":{"frames":[...]}}
    let list = part
        .get("video")
        .or_else(|| part.get("video_url").and_then(|v| v.get("frames")))
        .and_then(Value::as_array);
    if let Some(list) = list {
        let urls = list
            .iter()
            .map(|f| f.as_str().map(str::to_owned).ok_or("video frames must be image URLs"))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(MediaRequest::Video { urls, fps });
    }
    let url = url_of(part, "video_url")
        .or_else(|| part.get("video").and_then(Value::as_str).map(str::to_owned))
        .ok_or("video part needs video_url.url or a frame list")?;
    Ok(MediaRequest::Video { urls: vec![url], fps })
}

/// Splits the messages into template-ready messages (media parts become `{"type":"image"}` or
/// `{"type":"video"}`) and the media requests in order of appearance.
fn collect(messages: &[Value]) -> Result<(Vec<Value>, Vec<MediaRequest>), String> {
    let mut normalized = Vec::new();
    let mut media = Vec::new();
    for message in messages {
        let role = message.get("role").and_then(Value::as_str).ok_or("missing chat role")?;
        if !matches!(role, "system" | "user" | "assistant") {
            return Err("chat requests with media support system, user and assistant roles".into());
        }
        let mut parts = Vec::new();
        match message.get("content") {
            Some(Value::String(text)) => parts.push(json!({"type":"text","text":text})),
            Some(Value::Array(content)) => {
                for part in content {
                    match part.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            let text = part.get("text").and_then(Value::as_str).ok_or("text part must contain text")?;
                            parts.push(json!({"type":"text","text":text}));
                        }
                        Some("image_url" | "input_image" | "image") => {
                            let url = url_of(part, "image_url")
                                .or_else(|| url_of(part, "image"))
                                .ok_or("image part needs image_url.url")?;
                            media.push(MediaRequest::Image { url });
                            parts.push(json!({"type":"image"}));
                        }
                        Some("video_url" | "video") => {
                            media.push(video_request(part)?);
                            parts.push(json!({"type":"video"}));
                        }
                        _ => return Err("unsupported content part for this model".into()),
                    }
                }
            }
            _ => return Err("chat content must be text or content parts".into()),
        }
        if parts.is_empty() {
            parts.push(json!({"type":"text","text":""}));
        }
        normalized.push(json!({"role":role,"content":parts}));
    }
    let limit = MediaLimits::get().max_media_per_prompt;
    if media.len() > limit {
        return Err(format!("at most {limit} images or videos per prompt (--limit-mm-per-prompt)"));
    }
    Ok((normalized, media))
}

/// Decodes and encodes the requested media on blocking threads. At most `vision_admission`
/// requests run at the same time; others wait.
async fn encode(state: &AppState, requests: Vec<MediaRequest>) -> Result<Vec<EncodedMedia>, String> {
    if requests.is_empty() {
        return Ok(Vec::new());
    }
    let encoder = state.vision.clone().ok_or("this server has no vision model loaded")?;
    for request in &requests {
        if encoder.marker(request.kind()).is_none() {
            return Err(format!("{} models do not accept {:?} input", encoder.family(), request.kind()));
        }
    }
    // http(s) URLs are downloaded here (only with --allow-remote-images, SSRF-checked); the encoder only reads data URLs
    let requests = resolve_remote(state, requests).await?;
    let permit = state
        .vision_admission
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| "image encoder is shutting down")?;
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        requests.iter().map(|r| encode_one(&*encoder, r)).collect::<Result<Vec<_>, _>>()
    })
    .await
    .map_err(|_| "image encoder worker failed")?
}

/// Replaces every URL of the requests by a data URL (`remote_image::resolve_all` refuses private addresses and, unless
/// the operator allowed it, every remote URL).
async fn resolve_remote(state: &AppState, requests: Vec<MediaRequest>) -> Result<Vec<MediaRequest>, String> {
    let mut urls: Vec<String> = Vec::new();
    for request in &requests {
        match request {
            MediaRequest::Image { url } => urls.push(url.clone()),
            MediaRequest::Video { urls: frames, .. } => urls.extend(frames.iter().cloned()),
        }
    }
    let mut resolved = super::remote_image::resolve_all(
        urls,
        state.http.options.allow_remote_images,
        super::remote_image::Policy::default(),
    )
    .await?
    .into_iter();
    let mut next = || resolved.next().ok_or_else(|| "internal error: image URL count changed".to_owned());
    requests
        .into_iter()
        .map(|request| match request {
            MediaRequest::Image { .. } => Ok(MediaRequest::Image { url: next()? }),
            MediaRequest::Video { urls: frames, fps } => {
                let urls = frames.iter().map(|_| next()).collect::<Result<Vec<_>, _>>()?;
                Ok(MediaRequest::Video { urls, fps })
            }
        })
        .collect()
}

fn encode_one(encoder: &dyn MultimodalEncoder, request: &MediaRequest) -> Result<EncodedMedia, String> {
    match request {
        MediaRequest::Image { url } => encoder.encode_image(&preprocess::resolve_image(url, None)?),
        MediaRequest::Video { urls, fps } => {
            let limits = MediaLimits::get();
            let (frames, fps) = if let [single] = urls.as_slice() {
                let bytes = preprocess::resolve_image(single, None)?;
                match preprocess::animated_frames(&bytes, limits.max_video_frames)? {
                    Some((frames, rate)) => (frames, rate.unwrap_or(*fps)),
                    None => (vec![bytes], *fps),
                }
            } else {
                if urls.len() > limits.max_video_frames {
                    return Err(format!("a video can have at most {} frames", limits.max_video_frames));
                }
                (urls.iter().map(|u| preprocess::resolve_image(u, None)).collect::<Result<Vec<_>, _>>()?, *fps)
            };
            encoder.encode_video(&frames, fps)
        }
    }
}

/// Encodes one image for `/v1/images/embeddings`.
pub(super) async fn encode_single_image(state: &AppState, url: &str) -> Result<Vec<Vec<f32>>, String> {
    let mut encoded = encode(state, vec![MediaRequest::Image { url: url.to_owned() }]).await?;
    Ok(encoded.remove(0).embeddings)
}

/// Renders the chat template for messages with media and returns the expanded prompt, its token
/// ids and the embeddings / position ids that go with it.
pub(super) async fn prepare_media_chat(
    state: &AppState,
    messages: &[Value],
) -> Result<(String, Vec<i64>, MediaInputs), String> {
    let encoder: Arc<dyn MultimodalEncoder> = state.vision.clone().ok_or("vision model is not loaded")?;
    let (normalized, requests) = collect(messages)?;
    let rendered = chat_environment()
        .render_str(&state.chat_template, json!({"messages":normalized,"add_generation_prompt":true}))
        .map_err(|e| format!("vision chat template failed: {e}"))?;
    let media = encode(state, requests).await?;
    let tokenize = |text: &str| -> Result<Vec<i64>, String> {
        Ok(state.tokenizer.encode(text)?.into_iter().map(i64::from).collect())
    };
    let atomic = |text: &str| -> Result<Option<i64>, String> {
        let ids = state.tokenizer.encode_without_bos(text)?;
        Ok(match ids.as_slice() {
            [id] => Some(i64::from(*id)),
            _ => None,
        })
    };
    let built = pipeline::build_prompt(&*encoder, &rendered, media, &tokenize, &atomic)?;
    if built.ids.len() >= state.max_context {
        return Err("image prompt exceeds model context".into());
    }
    Ok((built.prompt, built.ids, built.inputs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn media_parts_are_collected_in_order() {
        let messages = vec![json!({"role":"user","content":[
            {"type":"text","text":"look"},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}},
            {"type":"video_url","video_url":{"url":"data:image/gif;base64,AAAA"},"fps":4},
            {"type":"video","video":["data:image/png;base64,AAAA","data:image/png;base64,BBBB"]}
        ]})];
        assert!(request_has_media(&messages));
        let (normalized, media) = collect(&messages).unwrap();
        assert_eq!(media.len(), 3);
        assert!(matches!(&media[0], MediaRequest::Image { url } if url.starts_with("data:image/png")));
        assert!(matches!(&media[1], MediaRequest::Video { urls, fps } if urls.len() == 1 && *fps == 4.0));
        assert!(matches!(&media[2], MediaRequest::Video { urls, .. } if urls.len() == 2));
        let types: Vec<_> = normalized[0]["content"].as_array().unwrap().iter().map(|p| p["type"].as_str().unwrap()).collect();
        assert_eq!(types, vec!["text", "image", "video", "video"]);
    }

    #[test]
    fn text_only_and_invalid_parts() {
        let text = vec![json!({"role":"user","content":"hi"})];
        assert!(!request_has_media(&text));
        let bad = vec![json!({"role":"user","content":[{"type":"input_audio","input_audio":{}}]})];
        assert!(collect(&bad).is_err());
        let tool = vec![json!({"role":"tool","content":"x"})];
        assert!(collect(&tool).is_err());
        let fps = vec![json!({"role":"user","content":[{"type":"video_url","video_url":{"url":"u"},"fps":0}]})];
        assert!(collect(&fps).is_err());
    }
}
