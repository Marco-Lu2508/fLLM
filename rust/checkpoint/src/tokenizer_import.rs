//! Tokenizers that ship without a Hugging Face `tokenizer.json`, converted to one in memory:
//!
//! * SentencePiece `tokenizer.model` / `spiece.model` / `sentencepiece.bpe.model` (protobuf, parsed by hand),
//! * tiktoken rank files (`tiktoken.model`, `qwen.tiktoken`, GLM's `tokenizer.model`, any `*.tiktoken`),
//! * Mistral `tekken.json` (mistral-common v3 and later).
//!
//! The result is the same JSON a `tokenizers` conversion would write, so the converter stores it as
//! `tokenizer.huggingface.json` and the engine loads it with the `tokenizers` crate. Every conversion is checked
//! against the reference implementation (`sentencepiece`, `tiktoken`, `mistral_common`) in the engine tests.
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// A converted tokenizer.
pub(crate) struct Imported {
    /// Text of the equivalent `tokenizer.json`.
    pub json: String,
    /// Values for `tokenizer_config.json` keys the checkpoint does not set itself (`bos_token`, `eos_token`, ...).
    pub defaults: Map<String, Value>,
    /// `tokenizer.ggml.model` to record (`llama` for SentencePiece, `gpt2` for byte-level BPE).
    pub ggml_model: &'static str,
    /// Human-readable origin for the log line.
    pub source: String,
}

/// Converts the tokenizer files in `dir` (without needing the weights) and returns the `tokenizer.json` text. Used by
/// `examples/tokenizer_json.rs` and the reference comparison in `scripts/verify_tokenizers.py`.
pub fn tokenizer_json_from_dir(dir: &Path) -> Result<String, String> {
    let read = |name: &str| -> Value { fs::read(dir.join(name)).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(Value::Null) };
    let config = read("config.json");
    let tc = read("tokenizer_config.json");
    load(dir, &config, &tc)?.map(|i| i.json).ok_or_else(|| "no tekken.json, tiktoken or SentencePiece tokenizer file found".to_owned())
}

/// Looks for `tekken.json`, a tiktoken rank file and a SentencePiece model in `dir`, in that order.
pub(crate) fn load(dir: &Path, model_config: &Value, tc: &Value) -> Result<Option<Imported>, String> {
    let tekken = dir.join("tekken.json");
    if tekken.is_file() {
        let text = fs::read_to_string(&tekken).map_err(|e| format!("read tekken.json: {e}"))?;
        return tekken_to_hf(&text).map(Some);
    }
    let mut candidates: Vec<String> = ["tiktoken.model", "qwen.tiktoken", "tokenizer.model", "spiece.model", "sentencepiece.bpe.model", "sentencepiece.model"]
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
    if let Ok(entries) = fs::read_dir(dir) {
        let mut extra: Vec<String> = entries
            .flatten()
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|n| n.ends_with(".tiktoken") && !candidates.contains(n))
            .collect();
        extra.sort();
        candidates.extend(extra);
    }
    for name in candidates {
        let path = dir.join(&name);
        if !path.is_file() {
            continue;
        }
        let bytes = fs::read(&path).map_err(|e| format!("read {name}: {e}"))?;
        if looks_like_tiktoken(&bytes) {
            return tiktoken_to_hf(dir, &bytes, model_config, tc, &name).map(Some);
        }
        return spm_to_hf(&bytes, &name).map(Some);
    }
    Ok(None)
}

/// tiktoken files are text lines `base64(token) rank`; SentencePiece models are protobuf (first byte 0x0a).
fn looks_like_tiktoken(bytes: &[u8]) -> bool {
    let head = &bytes[..bytes.len().min(256)];
    let Some(line_end) = head.iter().position(|&b| b == b'\n') else {
        return false;
    };
    let Ok(line) = std::str::from_utf8(&head[..line_end]) else {
        return false;
    };
    let mut parts = line.trim_end().split(' ');
    match (parts.next(), parts.next(), parts.next()) {
        (Some(token), Some(rank), None) => {
            !token.is_empty() && token.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'/' | b'=')) && rank.bytes().all(|b| b.is_ascii_digit()) && !rank.is_empty()
        }
        _ => false,
    }
}

// ---------------------------------------------------------------- base64

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub(crate) fn base64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let n = (u32::from(chunk[0]) << 16) | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8) | u32::from(*chunk.get(2).unwrap_or(&0));
        out.push(B64[(n >> 18) as usize & 63] as char);
        out.push(B64[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { B64[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { B64[n as usize & 63] as char } else { '=' });
    }
    out
}

pub(crate) fn base64_decode(text: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let mut acc = 0u32;
    let mut bits = 0u32;
    for b in text.bytes() {
        let v = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' => break,
            _ => return Err(format!("invalid base64 character {:?}", b as char)),
        };
        acc = (acc << 6) | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------- SentencePiece

/// Protobuf wire reader, enough for `sentencepiece_model.proto`.
struct Wire<'a> {
    buf: &'a [u8],
    pos: usize,
}

enum Field<'a> {
    Varint(u64),
    Fixed32(u32),
    Bytes(&'a [u8]),
}

impl<'a> Wire<'a> {
    fn varint(&mut self) -> Result<u64, String> {
        let mut value = 0u64;
        for shift in (0..70).step_by(7) {
            let byte = *self.buf.get(self.pos).ok_or("SentencePiece model is truncated")?;
            self.pos += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err("SentencePiece model has an invalid varint".into())
    }

    fn next(&mut self) -> Result<Option<(u32, Field<'a>)>, String> {
        if self.pos >= self.buf.len() {
            return Ok(None);
        }
        let key = self.varint()?;
        let number = (key >> 3) as u32;
        let field = match key & 7 {
            0 => Field::Varint(self.varint()?),
            1 => {
                self.pos += 8;
                Field::Varint(0)
            }
            2 => {
                let len = usize::try_from(self.varint()?).map_err(|_| "SentencePiece field length overflow")?;
                let end = self.pos.checked_add(len).filter(|&e| e <= self.buf.len()).ok_or("SentencePiece model is truncated")?;
                let bytes = &self.buf[self.pos..end];
                self.pos = end;
                Field::Bytes(bytes)
            }
            5 => {
                let bytes: [u8; 4] = self.buf.get(self.pos..self.pos + 4).ok_or("SentencePiece model is truncated")?.try_into().unwrap();
                self.pos += 4;
                Field::Fixed32(u32::from_le_bytes(bytes))
            }
            other => return Err(format!("unsupported protobuf wire type {other} in SentencePiece model")),
        };
        if self.pos > self.buf.len() {
            return Err("SentencePiece model is truncated".into());
        }
        Ok(Some((number, field)))
    }
}

/// `SentencePiece.Type`.
const PIECE_NORMAL: i32 = 1;
const PIECE_UNKNOWN: i32 = 2;
const PIECE_CONTROL: i32 = 3;
const PIECE_USER_DEFINED: i32 = 4;

#[derive(Debug, Clone)]
pub(crate) struct SpmPiece {
    pub text: String,
    pub score: f32,
    pub kind: i32,
}

/// The parts of `sentencepiece_model.proto` that determine tokenization.
#[derive(Debug)]
pub(crate) struct SpmModel {
    pub pieces: Vec<SpmPiece>,
    /// 1 unigram, 2 BPE, 3 word, 4 char
    pub model_type: i32,
    pub byte_fallback: bool,
    pub unk_id: i32,
    pub bos_id: i32,
    pub eos_id: i32,
    pub pad_id: i32,
    pub unk_piece: String,
    pub normalizer_name: String,
    pub charsmap: Vec<u8>,
    pub add_dummy_prefix: bool,
    pub remove_extra_whitespaces: bool,
    pub escape_whitespaces: bool,
    pub treat_whitespace_as_suffix: bool,
}

pub(crate) fn parse_spm(bytes: &[u8]) -> Result<SpmModel, String> {
    let mut model = SpmModel {
        pieces: Vec::new(),
        model_type: 1,
        byte_fallback: false,
        unk_id: 0,
        bos_id: 1,
        eos_id: 2,
        pad_id: -1,
        unk_piece: "<unk>".into(),
        normalizer_name: String::new(),
        charsmap: Vec::new(),
        add_dummy_prefix: true,
        remove_extra_whitespaces: true,
        escape_whitespaces: true,
        treat_whitespace_as_suffix: false,
    };
    let mut wire = Wire { buf: bytes, pos: 0 };
    while let Some((number, field)) = wire.next()? {
        match (number, field) {
            (1, Field::Bytes(body)) => {
                let mut piece = SpmPiece { text: String::new(), score: 0.0, kind: PIECE_NORMAL };
                let mut inner = Wire { buf: body, pos: 0 };
                while let Some((n, f)) = inner.next()? {
                    match (n, f) {
                        (1, Field::Bytes(b)) => piece.text = String::from_utf8(b.to_vec()).map_err(|_| "SentencePiece piece is not UTF-8")?,
                        (2, Field::Fixed32(v)) => piece.score = f32::from_bits(v),
                        (3, Field::Varint(v)) => piece.kind = v as i32,
                        _ => {}
                    }
                }
                model.pieces.push(piece);
            }
            (2, Field::Bytes(body)) => {
                let mut inner = Wire { buf: body, pos: 0 };
                while let Some((n, f)) = inner.next()? {
                    match (n, f) {
                        (3, Field::Varint(v)) => model.model_type = v as i32,
                        (35, Field::Varint(v)) => model.byte_fallback = v != 0,
                        (24, Field::Varint(v)) => model.treat_whitespace_as_suffix = v != 0,
                        (40, Field::Varint(v)) => model.unk_id = v as i64 as i32,
                        (41, Field::Varint(v)) => model.bos_id = v as i64 as i32,
                        (42, Field::Varint(v)) => model.eos_id = v as i64 as i32,
                        (43, Field::Varint(v)) => model.pad_id = v as i64 as i32,
                        (45, Field::Bytes(b)) => model.unk_piece = String::from_utf8_lossy(b).into_owned(),
                        _ => {}
                    }
                }
            }
            (3, Field::Bytes(body)) => {
                let mut inner = Wire { buf: body, pos: 0 };
                while let Some((n, f)) = inner.next()? {
                    match (n, f) {
                        (1, Field::Bytes(b)) => model.normalizer_name = String::from_utf8_lossy(b).into_owned(),
                        (2, Field::Bytes(b)) => model.charsmap = b.to_vec(),
                        (3, Field::Varint(v)) => model.add_dummy_prefix = v != 0,
                        (4, Field::Varint(v)) => model.remove_extra_whitespaces = v != 0,
                        (5, Field::Varint(v)) => model.escape_whitespaces = v != 0,
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    if model.pieces.is_empty() {
        return Err("SentencePiece model has no pieces".into());
    }
    Ok(model)
}

/// Merge list of a SentencePiece BPE model: every split of a piece into two vocabulary pieces, ordered by the score
/// of the merged piece (best first), as `transformers.convert_slow_tokenizer` derives it.
fn spm_merges(pieces: &[SpmPiece]) -> Vec<(String, String)> {
    let mut ids: HashMap<&str, usize> = HashMap::with_capacity(pieces.len());
    for (id, p) in pieces.iter().enumerate() {
        ids.entry(p.text.as_str()).or_insert(id);
    }
    struct Merge<'a> {
        left: &'a str,
        right: &'a str,
        score: f32,
        left_chars: usize,
        right_chars: usize,
    }
    let mut merges: Vec<Merge<'_>> = Vec::new();
    for piece in pieces {
        if piece.kind == PIECE_CONTROL || piece.kind == PIECE_UNKNOWN {
            continue;
        }
        let text = piece.text.as_str();
        let mut local: Vec<(usize, usize, Merge<'_>)> = Vec::new();
        for (cut, _) in text.char_indices().skip(1) {
            let (l, r) = text.split_at(cut);
            if let (Some(&li), Some(&ri)) = (ids.get(l), ids.get(r)) {
                local.push((li, ri, Merge { left: l, right: r, score: piece.score, left_chars: l.chars().count(), right_chars: r.chars().count() }));
            }
        }
        local.sort_by_key(|(li, ri, _)| (*li, *ri));
        merges.extend(local.into_iter().map(|(_, _, m)| m));
    }
    // Python: sorted(key=(score, len(left), len(right)), reverse=True); a stable descending sort
    merges.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.left_chars.cmp(&a.left_chars))
            .then(b.right_chars.cmp(&a.right_chars))
    });
    merges.into_iter().map(|m| (m.left.to_owned(), m.right.to_owned())).collect()
}

fn added_token(id: usize, content: &str, special: bool) -> Value {
    json!({"id": id, "content": content, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": special})
}

fn spm_to_hf(bytes: &[u8], name: &str) -> Result<Imported, String> {
    let m = parse_spm(bytes).map_err(|e| format!("{name}: {e}"))?;
    if m.model_type != 1 && m.model_type != 2 {
        return Err(format!("{name}: SentencePiece model type {} (word/char) is not supported; only unigram and BPE models are", m.model_type));
    }
    if m.treat_whitespace_as_suffix {
        return Err(format!("{name}: SentencePiece treat_whitespace_as_suffix is not supported"));
    }
    if m.charsmap.is_empty() && !matches!(m.normalizer_name.as_str(), "identity" | "") {
        return Err(format!(
            "{name}: normalizer rule {:?} has no embedded charsmap; only models with a precompiled charsmap or the identity rule are supported",
            m.normalizer_name
        ));
    }
    let mut seen: HashMap<&str, usize> = HashMap::new();
    for (id, p) in m.pieces.iter().enumerate() {
        if let Some(first) = seen.insert(p.text.as_str(), id) {
            return Err(format!("{name}: piece {:?} appears twice (ids {first} and {id})", p.text));
        }
    }
    // SentencePiece normalization: charsmap, strip and collapse spaces, dummy prefix, escape spaces
    let mut normalizers: Vec<Value> = Vec::new();
    if !m.charsmap.is_empty() {
        normalizers.push(json!({"type": "Precompiled", "precompiled_charsmap": base64_encode(&m.charsmap)}));
    }
    if m.remove_extra_whitespaces {
        normalizers.push(json!({"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "}));
        normalizers.push(json!({"type": "Replace", "pattern": {"Regex": "\\A | \\z"}, "content": ""}));
    }
    // The dummy prefix goes in front of every text segment between added tokens, like transformers' legacy Llama
    // tokenizers and llama.cpp do: a chat template that writes `<s>` into the text gets the prefix after it, which is
    // what SentencePiece produces when the BOS id is added separately.
    if m.add_dummy_prefix {
        normalizers.push(json!({"type": "Prepend", "prepend": " "}));
    }
    if m.escape_whitespaces {
        normalizers.push(json!({"type": "Replace", "pattern": {"String": " "}, "content": "\u{2581}"}));
    }
    let pre_tokenizer = Value::Null;
    // normalized text that starts with the dummy prefix is rendered back without it
    let mut decoder = Vec::<Value>::new();
    if m.escape_whitespaces {
        decoder.push(json!({"type": "Replace", "pattern": {"String": "\u{2581}"}, "content": " "}));
    }
    if m.byte_fallback {
        decoder.push(json!({"type": "ByteFallback"}));
    }
    // Strip works on the fused text, not on every token
    decoder.push(json!({"type": "Fuse"}));
    if m.add_dummy_prefix {
        decoder.push(json!({"type": "Strip", "content": " ", "start": 1, "stop": 0}));
    }
    let model_json = if m.model_type == 1 {
        let vocab: Vec<Value> = m.pieces.iter().map(|p| json!([p.text, f64::from(p.score)])).collect();
        let unk_id = usize::try_from(m.unk_id).ok().filter(|&u| u < m.pieces.len());
        json!({"type": "Unigram", "unk_id": unk_id, "vocab": vocab, "byte_fallback": m.byte_fallback})
    } else {
        let vocab: Map<String, Value> = m.pieces.iter().enumerate().map(|(i, p)| (p.text.clone(), json!(i))).collect();
        let merges: Vec<Value> = spm_merges(&m.pieces).into_iter().map(|(l, r)| json!([l, r])).collect();
        json!({"type": "BPE", "dropout": null, "unk_token": m.unk_piece, "continuing_subword_prefix": null, "end_of_word_suffix": null,
            "fuse_unk": true, "byte_fallback": m.byte_fallback, "ignore_merges": false, "vocab": vocab, "merges": merges})
    };
    let added: Vec<Value> = m
        .pieces
        .iter()
        .enumerate()
        .filter(|(_, p)| matches!(p.kind, PIECE_UNKNOWN | PIECE_CONTROL | PIECE_USER_DEFINED))
        .map(|(id, p)| added_token(id, &p.text, p.kind != PIECE_USER_DEFINED))
        .collect();
    let json = json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
        "normalizer": if normalizers.is_empty() { Value::Null } else { json!({"type": "Sequence", "normalizers": normalizers}) },
        "pre_tokenizer": pre_tokenizer, "post_processor": null,
        "decoder": if decoder.is_empty() { Value::Null } else { json!({"type": "Sequence", "decoders": decoder}) },
        "model": model_json,
    });
    let mut defaults = Map::new();
    let piece_at = |id: i32| usize::try_from(id).ok().and_then(|i| m.pieces.get(i)).map(|p| p.text.clone());
    if let Some(t) = piece_at(m.bos_id) {
        defaults.insert("bos_token".into(), json!(t));
    }
    if let Some(t) = piece_at(m.eos_id) {
        defaults.insert("eos_token".into(), json!(t));
    }
    if let Some(t) = piece_at(m.unk_id) {
        defaults.insert("unk_token".into(), json!(t));
    }
    if let Some(t) = piece_at(m.pad_id) {
        defaults.insert("pad_token".into(), json!(t));
    }
    // LlamaTokenizer's default: a BOS token is prepended whenever the model defines one
    defaults.insert("add_bos_token".into(), json!(m.bos_id >= 0));
    let kind = if m.model_type == 1 { "unigram" } else { "BPE" };
    Ok(Imported {
        json: json.to_string(),
        defaults,
        ggml_model: "llama",
        source: format!("SentencePiece {kind} model {name} ({} pieces{})", m.pieces.len(), if m.byte_fallback { ", byte fallback" } else { "" }),
    })
}

// ---------------------------------------------------------------- byte-level BPE (tiktoken, tekken)

/// GPT-2's byte to printable character table.
pub(crate) fn byte_to_char() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut extra = 0u32;
    for b in 0..=255u32 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        table[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            let c = char::from_u32(256 + extra).unwrap();
            extra += 1;
            c
        };
    }
    table
}

fn bytes_to_text(bytes: &[u8], table: &[char; 256]) -> String {
    bytes.iter().map(|&b| table[b as usize]).collect()
}

/// tiktoken's `bpe()` helper: merge the cheapest pair until only parts below `max_rank` remain.
fn split_below(ranks: &HashMap<Vec<u8>, u32>, token: &[u8], max_rank: u32) -> Vec<Vec<u8>> {
    let mut parts: Vec<Vec<u8>> = token.iter().map(|&b| vec![b]).collect();
    loop {
        let mut best: Option<(usize, u32)> = None;
        for i in 0..parts.len().saturating_sub(1) {
            let mut joined = parts[i].clone();
            joined.extend_from_slice(&parts[i + 1]);
            if let Some(&rank) = ranks.get(&joined) {
                if best.is_none_or(|(_, r)| rank < r) {
                    best = Some((i, rank));
                }
            }
        }
        match best {
            Some((i, rank)) if rank < max_rank => {
                let right = parts.remove(i + 1);
                parts[i].extend_from_slice(&right);
            }
            _ => return parts,
        }
    }
}

/// Everything needed to describe a byte-level BPE vocabulary in the Hugging Face format.
struct ByteBpe {
    /// token bytes in id order (ids start at `first_id`)
    tokens: Vec<Vec<u8>>,
    first_id: u32,
    pattern: String,
    /// `(id, content)` of added special tokens
    specials: Vec<(u32, String)>,
}

fn byte_bpe_to_json(spec: &ByteBpe) -> Result<String, String> {
    let table = byte_to_char();
    let mut ranks: HashMap<Vec<u8>, u32> = HashMap::with_capacity(spec.tokens.len());
    for (i, t) in spec.tokens.iter().enumerate() {
        if ranks.insert(t.clone(), i as u32).is_some() {
            return Err(format!("duplicate token {t:?} in the rank file"));
        }
    }
    for b in 0..=255u8 {
        if !ranks.contains_key(&vec![b]) {
            return Err(format!("the rank file has no single-byte token for 0x{b:02x}"));
        }
    }
    let mut vocab = Map::new();
    let mut merges = Vec::<Value>::new();
    for (i, t) in spec.tokens.iter().enumerate() {
        vocab.insert(bytes_to_text(t, &table), json!(spec.first_id + i as u32));
        if t.len() > 1 {
            let parts = split_below(&ranks, t, i as u32);
            if parts.len() == 2 {
                merges.push(json!([bytes_to_text(&parts[0], &table), bytes_to_text(&parts[1], &table)]));
            }
        }
    }
    // Special tokens are part of the model vocabulary too: `tokenizers` gives an added token that is not in the vocabulary a
    // fresh id after the vocabulary, ignoring the id in the file. Ids that are neither a rank nor a named special token
    // (gaps before reserved tokens) become placeholders.
    let regular_end = spec.first_id + spec.tokens.len() as u32;
    let mut taken: HashMap<u32, String> = HashMap::new();
    for (id, name) in &spec.specials {
        if *id >= spec.first_id && *id < regular_end {
            return Err(format!("special token {name:?} id {id} collides with a regular token"));
        }
        if vocab.contains_key(name) {
            return Err(format!("special token {name:?} has the same text as a regular token"));
        }
        if taken.insert(*id, name.clone()).is_some() {
            return Err(format!("two special tokens share id {id}"));
        }
    }
    let end = taken.keys().max().map_or(regular_end, |m| (*m + 1).max(regular_end));
    for id in regular_end..end {
        taken.entry(id).or_insert_with(|| format!("<|fllm_unused_{id}|>"));
    }
    let mut added = Vec::<Value>::new();
    let mut ordered: Vec<(&u32, &String)> = taken.iter().collect();
    ordered.sort();
    for (id, name) in ordered {
        vocab.insert(name.clone(), json!(id));
        added.push(added_token(*id as usize, name, true));
    }
    let doc = json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added, "normalizer": null,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": spec.pattern}, "behavior": "Removed", "invert": true},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}]},
        "post_processor": null,
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
        "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "", "end_of_word_suffix": "",
            "fuse_unk": false, "byte_fallback": false, "ignore_merges": true, "vocab": vocab, "merges": merges},
    });
    Ok(doc.to_string())
}

// ---------------------------------------------------------------- tiktoken

pub(crate) const PATTERN_CL100K: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
pub(crate) const PATTERN_O200K: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
/// Qwen (first generation) splits digits one by one.
pub(crate) const PATTERN_QWEN1: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
/// Kimi K2: Han runs are separate pieces and the cased-letter classes exclude Han.
pub(crate) const PATTERN_KIMI: &str = concat!(
    r"[\p{Han}]+|",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|",
    r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
);

/// Token families that define their split pattern and special tokens in remote code instead of a data file.
struct Family {
    name: &'static str,
    pattern: &'static str,
    /// named special tokens at ids `base + index` (Qwen1) when the checkpoint lists none
    specials: fn(usize) -> Vec<(u32, String)>,
    /// number of `<|reserved_token_N|>` ids that follow the base ranks (Kimi K2)
    reserved: usize,
}

fn no_specials(_: usize) -> Vec<(u32, String)> {
    Vec::new()
}

fn qwen1_specials(base: usize) -> Vec<(u32, String)> {
    let mut v: Vec<String> = vec!["<|endoftext|>".into(), "<|im_start|>".into(), "<|im_end|>".into()];
    v.extend((0..205).map(|i| format!("<|extra_{i}|>")));
    v.into_iter().enumerate().map(|(i, s)| ((base + i) as u32, s)).collect()
}

const FAMILIES: &[Family] = &[
    Family { name: "kimi", pattern: PATTERN_KIMI, specials: no_specials, reserved: 256 },
    Family { name: "qwen1", pattern: PATTERN_QWEN1, specials: qwen1_specials, reserved: 0 },
    Family { name: "glm", pattern: PATTERN_CL100K, specials: no_specials, reserved: 0 },
    Family { name: "cl100k", pattern: PATTERN_CL100K, specials: no_specials, reserved: 0 },
    Family { name: "o200k", pattern: PATTERN_O200K, specials: no_specials, reserved: 0 },
];

fn family_by_name(name: &str) -> Option<&'static Family> {
    FAMILIES.iter().find(|f| f.name == name)
}

/// Which family a checkpoint with tiktoken files belongs to, judged by `config.json` and the tokenizer files.
fn detect_family(dir: &Path, config: &Value, tc: &Value, file: &str) -> Option<&'static Family> {
    if let Ok(name) = std::env::var("FLLM_TIKTOKEN_PATTERN") {
        if let Some(f) = family_by_name(name.trim()) {
            return Some(f);
        }
    }
    let model_type = config.get("model_type").and_then(Value::as_str).unwrap_or_default();
    let arch = config
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .and_then(Value::as_str)
        .unwrap_or_default();
    let class = tc.get("tokenizer_class").and_then(Value::as_str).unwrap_or_default();
    let auto_map = tc.get("auto_map").map(Value::to_string).unwrap_or_default();
    if model_type == "kimi_k2" || arch.starts_with("KimiK2") || auto_map.contains("tokenization_kimi") || (class == "TikTokenTokenizer" && file == "tiktoken.model") {
        return family_by_name("kimi");
    }
    if model_type == "qwen" || arch == "QWenLMHeadModel" || file == "qwen.tiktoken" {
        return family_by_name("qwen1");
    }
    if matches!(model_type, "chatglm" | "glm4") && file == "tokenizer.model" || auto_map.contains("ChatGLM4Tokenizer") {
        return family_by_name("glm");
    }
    if file.contains("o200k") {
        return family_by_name("o200k");
    }
    if file.contains("cl100k") {
        return family_by_name("cl100k");
    }
    // a pattern string in the checkpoint's tokenizer code is picked up by `pattern_from_code`
    let _ = dir;
    None
}

/// Extracts a `pat_str` raw-string literal (or a `"|".join([...])` of raw strings) from the tokenizer's Python code.
pub(crate) fn pattern_from_code(source: &str) -> Option<String> {
    let start = source.find("pat_str").or_else(|| source.find("PAT_STR"))?;
    let rest = &source[start..];
    let eq = rest.find('=')?;
    let rest = &rest[eq + 1..];
    // end of the statement: the first line that is not part of the literal or list
    let mut parts = Vec::<String>::new();
    let mut cursor = rest;
    let join = rest.trim_start().starts_with("\"|\".join");
    if join {
        // skip the separator literal and the opening parenthesis of `"|".join(`
        cursor = &rest[rest.find(".join").unwrap_or(0) + ".join".len()..];
    }
    loop {
        let trimmed = cursor.trim_start_matches([' ', '\t', '\r', '\n', ',', '(', '[']);
        let (prefix_len, quote) = if let Some(r) = trimmed.strip_prefix("r\"\"\"") {
            (4, ("\"\"\"", r))
        } else if let Some(r) = trimmed.strip_prefix("r\"") {
            (2, ("\"", r))
        } else if let Some(r) = trimmed.strip_prefix("r'") {
            (2, ("'", r))
        } else if let Some(r) = trimmed.strip_prefix("\"") {
            (1, ("\"", r))
        } else {
            break;
        };
        let _ = prefix_len;
        let (terminator, body) = quote;
        let end = body.find(terminator)?;
        let mut text = body[..end].to_owned();
        if !trimmed.starts_with('r') {
            // ordinary string literal: undo the escapes the regex needs
            text = text.replace("\\\\", "\u{1}").replace("\\n", "\n").replace("\\r", "\r").replace('\u{1}', "\\");
        }
        parts.push(text);
        cursor = &body[end + terminator.len()..];
        if !join {
            break;
        }
        let after = cursor.trim_start();
        if !(after.starts_with(',') || after.starts_with(']') || after.starts_with(')')) {
            break;
        }
        if after.starts_with(']') || after.starts_with(')') {
            break;
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("|"))
    }
}

fn find_pattern_in_code(dir: &Path) -> Option<String> {
    let mut files: Vec<_> = fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "py")).collect();
    files.sort();
    for p in files {
        if let Ok(text) = fs::read_to_string(&p) {
            if text.contains("tiktoken") {
                if let Some(pat) = pattern_from_code(&text) {
                    return Some(pat);
                }
            }
        }
    }
    None
}

fn tiktoken_to_hf(dir: &Path, bytes: &[u8], config: &Value, tc: &Value, file: &str) -> Result<Imported, String> {
    let text = std::str::from_utf8(bytes).map_err(|_| format!("{file}: not UTF-8"))?;
    let mut tokens: Vec<Option<Vec<u8>>> = Vec::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let (token, rank) = line.split_once(' ').ok_or_else(|| format!("{file}: line {} is not 'token rank'", n + 1))?;
        let rank: usize = rank.trim().parse().map_err(|_| format!("{file}: line {} has an invalid rank", n + 1))?;
        if rank > 10_000_000 {
            return Err(format!("{file}: rank {rank} exceeds the safety limit"));
        }
        if tokens.len() <= rank {
            tokens.resize(rank + 1, None);
        }
        if tokens[rank].is_some() {
            return Err(format!("{file}: rank {rank} appears twice"));
        }
        tokens[rank] = Some(base64_decode(token).map_err(|e| format!("{file}: line {}: {e}", n + 1))?);
    }
    let tokens: Vec<Vec<u8>> = tokens
        .into_iter()
        .enumerate()
        .map(|(i, t)| t.ok_or_else(|| format!("{file}: rank {i} is missing (ranks must be contiguous)")))
        .collect::<Result<_, _>>()?;
    let base = tokens.len();
    let family = detect_family(dir, config, tc, file);
    // an explicit pattern: `pat_str` in tokenizer_config.json or a `pat_str.txt` next to the rank file
    let explicit = tc
        .get("pat_str")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| fs::read_to_string(dir.join("pat_str.txt")).ok().map(|s| s.trim_end_matches(['\r', '\n']).to_owned()));
    let pattern = match (explicit, family, find_pattern_in_code(dir)) {
        (Some(p), _, _) => p,
        (None, Some(f), _) => f.pattern.to_owned(),
        (None, None, Some(p)) => p,
        (None, None, None) => {
            return Err(format!(
                "{file} is a tiktoken rank file but its split pattern is unknown: no known family matched and no tokenizer code with a pat_str was found; \
                 set FLLM_TIKTOKEN_PATTERN to one of {}",
                FAMILIES.iter().map(|f| f.name).collect::<Vec<_>>().join(", ")
            ))
        }
    };
    // special tokens: the checkpoint's own list first, then the family's constants
    let mut specials: Vec<(u32, String)> = Vec::new();
    if let Some(decoder) = tc.get("added_tokens_decoder").and_then(Value::as_object) {
        for (id, entry) in decoder {
            if let (Ok(id), Some(content)) = (id.parse::<u32>(), entry.get("content").and_then(Value::as_str)) {
                specials.push((id, content.to_owned()));
            }
        }
    }
    if let Some(f) = family {
        if specials.is_empty() {
            specials = (f.specials)(base);
        }
        if f.reserved > 0 {
            for i in 0..f.reserved {
                let id = (base + i) as u32;
                if !specials.iter().any(|(sid, _)| *sid == id) {
                    specials.push((id, format!("<|reserved_token_{i}|>")));
                }
            }
        }
    }
    specials.sort();
    specials.dedup();
    let spec = ByteBpe { tokens, first_id: 0, pattern, specials: specials.clone() };
    let json = byte_bpe_to_json(&spec).map_err(|e| format!("{file}: {e}"))?;
    let mut defaults = Map::new();
    if tc.get("eos_token").is_none() {
        if let Some((_, name)) = specials.iter().find(|(_, n)| matches!(n.as_str(), "<|endoftext|>" | "<|im_end|>" | "[EOS]")) {
            defaults.insert("eos_token".into(), json!(name));
        }
    }
    Ok(Imported {
        json,
        defaults,
        ggml_model: "gpt2",
        source: format!("tiktoken ranks {file} ({base} tokens, family {})", family.map_or("from code", |f| f.name)),
    })
}

// ---------------------------------------------------------------- tekken

/// Special tokens of mistral-common tokenizers before v7 (and the base of every later list), in rank order.
const TEKKEN_DEFAULT_SPECIALS: [&str; 20] = [
    "<unk>", "<s>", "</s>", "[INST]", "[/INST]", "[AVAILABLE_TOOLS]", "[/AVAILABLE_TOOLS]", "[TOOL_RESULTS]", "[/TOOL_RESULTS]", "[TOOL_CALLS]", "[IMG]", "<pad>", "[IMG_BREAK]",
    "[IMG_END]", "[PREFIX]", "[MIDDLE]", "[SUFFIX]", "[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]", "[TOOL_CONTENT]",
];

fn tekken_to_hf(text: &str) -> Result<Imported, String> {
    let doc: Value = serde_json::from_str(text).map_err(|e| format!("invalid tekken.json: {e}"))?;
    let config = doc.get("config").ok_or("tekken.json has no config")?;
    let pattern = config.get("pattern").and_then(Value::as_str).ok_or("tekken.json config has no pattern")?;
    let num_special = config.get("default_num_special_tokens").and_then(Value::as_u64).ok_or("tekken.json config has no default_num_special_tokens")? as usize;
    let vocab_size = config.get("default_vocab_size").and_then(Value::as_u64).ok_or("tekken.json config has no default_vocab_size")? as usize;
    let version = config.get("version").and_then(Value::as_str).unwrap_or("v3");
    let vocab = doc.get("vocab").and_then(Value::as_array).ok_or("tekken.json has no vocab")?;
    let inner = vocab_size.checked_sub(num_special).ok_or("tekken.json default_vocab_size is smaller than the special token count")?;
    if vocab.len() < inner {
        return Err(format!("tekken.json lists {} tokens but needs {inner}", vocab.len()));
    }
    let mut tokens = Vec::with_capacity(inner);
    for (i, entry) in vocab.iter().take(inner).enumerate() {
        if entry.get("rank").and_then(Value::as_u64) != Some(i as u64) {
            return Err(format!("tekken.json vocab entry {i} has rank {:?}", entry.get("rank")));
        }
        let b64 = entry.get("token_bytes").and_then(Value::as_str).ok_or("tekken.json vocab entry without token_bytes")?;
        tokens.push(base64_decode(b64)?);
    }
    let mut specials: Vec<(u32, String)> = match doc.get("special_tokens").and_then(Value::as_array) {
        Some(list) => list
            .iter()
            .map(|s| {
                Ok((
                    s.get("rank").and_then(Value::as_u64).ok_or("tekken.json special token without rank")? as u32,
                    s.get("token_str").and_then(Value::as_str).ok_or("tekken.json special token without token_str")?.to_owned(),
                ))
            })
            .collect::<Result<_, &str>>()?,
        None if matches!(version, "v1" | "v2" | "v3" | "v7") => TEKKEN_DEFAULT_SPECIALS.iter().enumerate().map(|(i, s)| (i as u32, (*s).to_owned())).collect(),
        None => return Err(format!("tekken.json (version {version}) lists no special_tokens")),
    };
    let named = specials.len();
    for i in named..num_special {
        specials.push((i as u32, format!("<SPECIAL_{i}>")));
    }
    if specials.len() != num_special || specials.iter().any(|(r, _)| *r as usize >= num_special) {
        return Err("tekken.json special tokens do not fit default_num_special_tokens".into());
    }
    // tekken ids: special tokens first (0..num_special), then regular tokens shifted by num_special
    let spec = ByteBpe { tokens, first_id: num_special as u32, pattern: pattern.to_owned(), specials: specials.clone() };
    let value: Value = serde_json::from_str(&byte_bpe_to_json(&spec)?).map_err(|e| e.to_string())?;
    let mut defaults = Map::new();
    for (id, name) in &specials {
        match (*id, name.as_str()) {
            (_, "<s>") => {
                defaults.insert("bos_token".into(), json!(name));
            }
            (_, "</s>") => {
                defaults.insert("eos_token".into(), json!(name));
            }
            (_, "<unk>") => {
                defaults.insert("unk_token".into(), json!(name));
            }
            (_, "<pad>") => {
                defaults.insert("pad_token".into(), json!(name));
            }
            _ => {}
        }
    }
    defaults.insert("add_bos_token".into(), json!(true));
    Ok(Imported {
        json: value.to_string(),
        defaults,
        ggml_model: "gpt2",
        source: format!("Mistral tekken {version} ({inner} tokens, {num_special} special)"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_round_trip() {
        for data in [&b""[..], b"a", b"ab", b"abc", b"\x00\xff\x10\x80", b"hello world!"] {
            assert_eq!(base64_decode(&base64_encode(data)).unwrap(), data);
        }
        assert_eq!(base64_encode(b"Man"), "TWFu");
        assert_eq!(base64_encode(b"Ma"), "TWE=");
        assert_eq!(base64_decode("IGA=").unwrap(), b" `");
    }

    #[test]
    fn byte_table_matches_gpt2() {
        let t = byte_to_char();
        assert_eq!(t[b'A' as usize], 'A');
        assert_eq!(t[b' ' as usize], '\u{120}'); // Ġ
        assert_eq!(t[b'\n' as usize], '\u{10a}'); // Ċ
        assert_eq!(t[0], '\u{100}');
    }

    #[test]
    fn tiktoken_and_spm_are_told_apart() {
        assert!(looks_like_tiktoken(b"IQ== 0\nIg== 1\n"));
        assert!(!looks_like_tiktoken(&[0x0a, 0x0b, b'<', b'u', b'n', b'k', b'>']));
    }

    #[test]
    fn pattern_extraction_from_python_source() {
        let kimi = "class T:\n    pat_str = \"|\".join(\n        [\n            r\"\"\"[\\p{Han}]+\"\"\",\n            r\"\"\"\\p{N}{1,3}\"\"\",\n            r\"\"\"\\s+\"\"\",\n        ]\n    )\n    other = 1\n";
        assert_eq!(pattern_from_code(kimi).unwrap(), "[\\p{Han}]+|\\p{N}{1,3}|\\s+");
        let glm = "        pat_str = \"(?i:'s|'t)|\\\\p{L}+|\\\\s+\"\n        self.pat_str = re.compile(pat_str)\n";
        assert_eq!(pattern_from_code(glm).unwrap(), "(?i:'s|'t)|\\p{L}+|\\s+");
        let qwen = "PAT_STR = r\"\"\"(?i:'s|'t)|\\p{N}\"\"\"\nENDOFTEXT = 1\n";
        assert_eq!(pattern_from_code(qwen).unwrap(), "(?i:'s|'t)|\\p{N}");
        assert!(pattern_from_code("x = 1").is_none());
    }

    #[test]
    fn merges_follow_tiktoken_rank_order() {
        // tokens: bytes a, b, c then "ab" (rank 3) and "abc" (rank 4)
        let mut ranks: HashMap<Vec<u8>, u32> = HashMap::new();
        for (i, t) in [&b"a"[..], b"b", b"c", b"ab", b"abc"].iter().enumerate() {
            ranks.insert(t.to_vec(), i as u32);
        }
        assert_eq!(split_below(&ranks, b"abc", 4), vec![b"ab".to_vec(), b"c".to_vec()]);
        assert_eq!(split_below(&ranks, b"ab", 3), vec![b"a".to_vec(), b"b".to_vec()]);
    }
}
