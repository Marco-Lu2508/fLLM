use fllm_gguf_probe::MappedGguf;
use serde_json::Value;
use std::collections::HashSet;
use tokenizers::decoders::byte_fallback::ByteFallback;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::decoders::fuse::Fuse;
use tokenizers::decoders::metaspace::Metaspace as MetaspaceDecoder;
use tokenizers::decoders::sequence::Sequence as DecoderSequence;
use tokenizers::models::bpe::BPE;
use tokenizers::models::unigram::Unigram;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::SplitDelimiterBehavior;
use tokenizers::Tokenizer;

pub struct TokenizerBundle {
    tokenizer: Tokenizer,
    special: Vec<(String, u32)>,
    skipped_special_ids: HashSet<u32>,
    vocab_size: usize,
    eos_ids: Vec<u32>,
    bos_ids: Vec<u32>,
    pub chat_template: String,
    add_bos: bool,
    add_eos: bool,
    grammar_bytes: Vec<Vec<u8>>,
    sentencepiece: Option<SentencePieceOptions>,
    /// llama.cpp's SentencePiece (bigram merge) encoder for `tokenizer.ggml.model == "llama"`.
    spm: Option<Box<SpmEncoder>>,
    fim: Option<Box<Fim>>,
}

/// Fill-in-the-middle control tokens of a code model, as llama.cpp reads them from the GGUF
/// metadata keys `tokenizer.ggml.fim_*_token_id` or looks up their literals in the vocabulary.
#[derive(Clone, Copy)]
pub struct Fim {
    pub prefix: u32,
    pub suffix: u32,
    pub middle: u32,
    pub pad: Option<u32>,
    pub rep: Option<u32>,
    pub sep: Option<u32>,
    pub family: FimFamily,
}

/// Prompt layout a FIM family expects (llama.cpp `llama_fim_*`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FimFamily {
    /// Qwen2.5-Coder, CodeGemma: `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`.
    Qwen,
    /// StarCoder, StarCoder2, Granite code: `<fim_prefix>`, `<fim_suffix>`, `<fim_middle>`.
    StarCoder,
    /// CodeLlama: `<PRE>`, `<SUF>`, `<MID>`, with a space after `<PRE>`, before `<SUF>` and before `<MID>`.
    CodeLlama,
    /// DeepSeek-Coder: `<｜fim▁begin｜>`, `<｜fim▁hole｜>`, `<｜fim▁end｜>` (or the prefix/suffix/middle spelling).
    DeepSeek,
    /// Codestral: `[SUFFIX]suffix[PREFIX]prefix`, the model continues after the prefix.
    Codestral,
}

/// Control-token literals of the known FIM families, in prefix/suffix/middle order.
const FIM_LITERALS: [(FimFamily, [&str; 3]); 6] = [
    (
        FimFamily::Qwen,
        ["<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"],
    ),
    (FimFamily::StarCoder, ["<fim_prefix>", "<fim_suffix>", "<fim_middle>"]),
    (FimFamily::CodeLlama, ["<PRE>", "<SUF>", "<MID>"]),
    (
        FimFamily::DeepSeek,
        [
            "<｜fim▁begin｜>",
            "<｜fim▁hole｜>",
            "<｜fim▁end｜>",
        ],
    ),
    (
        FimFamily::DeepSeek,
        [
            "<｜fim▁prefix｜>",
            "<｜fim▁suffix｜>",
            "<｜fim▁middle｜>",
        ],
    ),
    (FimFamily::Codestral, ["[PREFIX]", "[SUFFIX]", "[MIDDLE]"]),
];

#[derive(Clone, Copy)]
struct SentencePieceOptions {
    add_space_prefix: bool,
    remove_extra_whitespaces: bool,
}

/// Control tokens that end an assistant turn in common chat formats.
const END_OF_GENERATION_TOKENS: [&str; 9] = [
    "<|im_end|>",
    "<|eot_id|>",
    "<|eom_id|>",
    "<|end|>",
    "<|return|>",
    "<|endoftext|>",
    "<|end_of_text|>",
    "<end_of_turn>",
    "<｜end▁of▁sentence｜>",
];

impl TokenizerBundle {
    pub fn from_gguf(mapping: &MappedGguf) -> Result<Self, String> {
        let metadata = &mapping.index().metadata;
        let tokens = metadata
            .get("tokenizer.ggml.tokens")
            .and_then(Value::as_array)
            .ok_or_else(|| "GGUF has no tokenizer.ggml.tokens array".to_owned())?;
        if tokens.is_empty() {
            return Err("GGUF tokenizer vocabulary is empty".to_owned());
        }
        let tokens = tokens
            .iter()
            .map(|token| {
                token
                    .as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| "GGUF tokenizer token must be a string".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(last_id) = tokens.len().checked_sub(1) {
            u32::try_from(last_id).map_err(|_| "GGUF vocabulary exceeds u32 IDs")?;
        }
        let byte_level = tokens.iter().take(4096).any(|token| token.contains('Ġ'));
        let pre_type = metadata
            .get("tokenizer.ggml.pre")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let serialized_hf = metadata.get("tokenizer.huggingface.json");
        let mut spm: Option<Box<SpmEncoder>> = None;
        let (tokenizer, sentencepiece, grammar_byte_fallback) = if let Some(json) = serialized_hf {
            let json = json
                .as_str()
                .ok_or_else(|| "tokenizer.huggingface.json must be a string".to_owned())?;
            (
                tokenizer_from_hf_json(
                    json.as_bytes(),
                    &tokens,
                    metadata
                        .get("tokenizer.ggml.token_type")
                        .and_then(Value::as_array)
                        .map(Vec::as_slice),
                )?,
                None,
                false,
            )
        } else {
            let model_name = metadata
                .get("tokenizer.ggml.model")
                .and_then(Value::as_str)
                .ok_or_else(|| "GGUF has no string tokenizer.ggml.model".to_owned())?;
            let preserve_legacy_byte_bpe = pre_type == "k2-horizon"
                || pre_type == "k2_horizon"
                || (byte_level && (model_name == "gpt2" || model_name.starts_with("qwen")));
            if preserve_legacy_byte_bpe {
                (
                    build_legacy_bpe(metadata, &tokens, byte_level, pre_type)?,
                    None,
                    false,
                )
            } else if model_name == "gemma4" {
                (build_gemma4_bpe(metadata, &tokens)?, None, true)
            } else if matches!(model_name, "llama" | "t5") {
                let (tokenizer, options, byte_fallback, encoder) =
                    build_unigram(metadata, &tokens, model_name)?;
                spm = encoder.map(Box::new);
                (tokenizer, Some(options), byte_fallback)
            } else if matches!(
                model_name,
                "gpt2" | "qwen2" | "qwen3" | "hybriddna" | "whitespace"
            ) {
                (
                    build_legacy_bpe(metadata, &tokens, byte_level, pre_type)?,
                    None,
                    false,
                )
            } else {
                return Err(format!(
                    "unsupported GGUF tokenizer model '{model_name}'; provide tokenizer.huggingface.json"
                ));
            }
        };
        let special_ids = special_ids(metadata, &tokens)?;
        let mut special = special_ids
            .iter()
            .map(|id| (tokens[*id as usize].clone(), *id))
            .collect::<Vec<_>>();
        special.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.0.cmp(&b.0)));
        let preserved = ["<tool_call>", "</tool_call>"]
            .into_iter()
            .collect::<HashSet<_>>();
        // User-defined tokens that are plain whitespace (Gemma has newline, double newline, tab and space runs) are
        // ordinary text:
        // llama.cpp renders every user-defined token and hides only control and unused ones.
        let skipped_special_ids = special
            .iter()
            .filter(|(token, _)| !preserved.contains(token.as_str()))
            .filter(|(token, _)| !token.chars().all(char::is_whitespace))
            .map(|(_, id)| *id)
            .collect();
        let mut eos_ids = ["tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"]
            .into_iter()
            .filter_map(|key| metadata.get(key).and_then(Value::as_u64))
            .filter_map(|id| u32::try_from(id).ok())
            .collect::<Vec<_>>();
        if let Some(value) = metadata.get("tokenizer.huggingface.eos_token_ids") {
            let ids = value
                .as_array()
                .ok_or("HF EOS token IDs must be an array")?;
            for value in ids {
                let id = value
                    .as_u64()
                    .and_then(|id| u32::try_from(id).ok())
                    .filter(|&id| (id as usize) < tokens.len())
                    .ok_or("HF EOS token ID is outside the vocabulary")?;
                if !eos_ids.contains(&id) {
                    eos_ids.push(id);
                }
            }
        }
        // Chat templates end a turn with a dedicated token that GGUF files often leave out of
        // `eos_token_id` (llama.cpp calls these end-of-generation tokens).
        for (id, token) in tokens.iter().enumerate() {
            if END_OF_GENERATION_TOKENS.contains(&token.as_str()) {
                if let Ok(id) = u32::try_from(id) {
                    if !eos_ids.contains(&id) {
                        eos_ids.push(id);
                    }
                }
            }
        }
        if let Some(bad) = eos_ids.iter().find(|&&id| id as usize >= tokens.len()) {
            return Err(format!(
                "EOS token ID {bad} is outside the GGUF vocabulary of {} tokens",
                tokens.len()
            ));
        }
        // `tokenizer.ggml.add_eos_token` is written by the checkpoint converter; it appends the
        // EOS token on `encode` exactly like llama.cpp does for `add_special` tokenization.
        let add_eos = metadata
            .get("tokenizer.ggml.add_eos_token")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            && !eos_ids.is_empty();
        let eos_ids = if eos_ids.is_empty() { vec![0] } else { eos_ids };
        let mut bos_ids: Vec<u32> = metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(Value::as_u64)
            .and_then(|id| u32::try_from(id).ok())
            .into_iter()
            .collect();
        // llama.cpp's default special tokens of byte-level BPE vocabularies: BOS is token 11 unless the file names one
        // (Falcon3 files name only EOS = 11 and are prompted with it in front)
        if bos_ids.is_empty() && metadata.get("tokenizer.ggml.model").and_then(Value::as_str) == Some("gpt2") && tokens.len() > 11 {
            bos_ids.push(11);
        }
        let add_bos = metadata
            .get("tokenizer.ggml.add_bos_token")
            .and_then(Value::as_bool)
            .unwrap_or_else(|| {
                default_add_bos(
                    metadata.get("tokenizer.ggml.model").and_then(Value::as_str),
                    metadata.get("tokenizer.ggml.pre").and_then(Value::as_str),
                    !bos_ids.is_empty(),
                )
            });
        let byte_map = byte_level_map();
        let grammar_bytes = tokens
            .iter()
            .enumerate()
            .map(|(id, token)| {
                if special_ids.contains(&(id as u32)) {
                    let mut bytes = vec![0xff];
                    bytes.extend_from_slice(token.as_bytes());
                    bytes
                } else if grammar_byte_fallback {
                    byte_fallback_token(token)
                        .map(|byte| vec![byte])
                        .unwrap_or_else(|| token.replace('▁', " ").into_bytes())
                } else if byte_level || pre_type == "k2-horizon" || pre_type == "k2_horizon" {
                    token
                        .chars()
                        .flat_map(|c| {
                            byte_map
                                .get(&c)
                                .copied()
                                .map(|b| vec![b])
                                .unwrap_or_else(|| c.to_string().into_bytes())
                        })
                        .collect()
                } else {
                    token.replace('▁', " ").into_bytes()
                }
            })
            .collect();
        let fim = fim_tokens(metadata, &special).map(Box::new);
        Ok(Self {
            tokenizer,
            special,
            skipped_special_ids,
            vocab_size: tokens.len(),
            eos_ids,
            bos_ids,
            chat_template: metadata
                .get("tokenizer.chat_template")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            add_bos,
            add_eos,
            grammar_bytes,
            sentencepiece,
            spm,
            fim,
        })
    }

    /// FIM control tokens of this checkpoint, if the GGUF declares or the vocabulary contains them.
    pub fn fim(&self) -> Option<&Fim> {
        self.fim.as_deref()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    pub fn eos_ids(&self) -> &[u32] {
        &self.eos_ids
    }
    pub fn bos_ids(&self) -> &[u32] {
        &self.bos_ids
    }

    pub fn grammar_factory(&self) -> Result<llguidance::ParserFactory, String> {
        use llguidance::toktrie::{ApproximateTokEnv, TokEnv, TokRxInfo, TokTrie};
        let info = TokRxInfo::new(self.vocab_size as u32, self.eos_ids[0]);
        let trie = TokTrie::from(&info, &self.grammar_bytes).with_eos_tokens(&self.eos_ids);
        let env: TokEnv = std::sync::Arc::new(ApproximateTokEnv::new(trie));
        llguidance::ParserFactory::new_simple(&env)
            .map_err(|e| format!("cannot initialize grammar tokenizer: {e}"))
    }

    /// Encodes with special tokens: BOS follows `tokenizer.ggml.add_bos_token`, EOS follows
    /// `tokenizer.ggml.add_eos_token`, and neither is repeated if the text already carries it.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        let mut ids = self.encode_with_bos(text, self.add_bos)?;
        if self.add_eos {
            if let Some(&eos) = self.eos_ids.first() {
                if !ids.ends_with(&[eos]) {
                    ids.push(eos);
                }
            }
        }
        Ok(ids)
    }

    pub fn encode_without_bos(&self, text: &str) -> Result<Vec<u32>, String> {
        self.encode_with_bos(text, false)
    }

    fn encode_with_bos(&self, text: &str, add_bos: bool) -> Result<Vec<u32>, String> {
        let mut ids = Vec::new();
        let starts_with_bos = self.bos_ids.iter().any(|&bos_id| {
            self.special
                .iter()
                .any(|(lit, id)| *id == bos_id && text.starts_with(lit))
        });
        if add_bos && !starts_with_bos {
            if let Some(&bos) = self.bos_ids.first() {
                ids.push(bos);
            }
        }
        if self.special.is_empty() {
            ids.extend(self.encode_plain(text)?);
            return Ok(ids);
        }
        let mut cursor = 0;
        while cursor < text.len() {
            let next = self
                .special
                .iter()
                .filter_map(|(literal, id)| {
                    text[cursor..]
                        .find(literal)
                        .map(|offset| (cursor + offset, literal, *id))
                })
                .min_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.len().cmp(&a.1.len())));
            let Some((start, literal, id)) = next else {
                ids.extend(self.encode_plain(&text[cursor..])?);
                break;
            };
            if start > cursor {
                ids.extend(self.encode_plain(&text[cursor..start])?);
            }
            ids.push(id);
            cursor = start + literal.len();
        }
        Ok(ids)
    }

    /// Encodes text with the model's tokenizer, without BOS, EOS or special-token parsing.
    pub fn encode_plain(&self, text: &str) -> Result<Vec<u32>, String> {
        if let (Some(spm), Some(options)) = (&self.spm, self.sentencepiece) {
            return Ok(spm.encode(text, options.add_space_prefix));
        }
        let sentencepiece_text = self
            .sentencepiece
            .map(|options| normalize_sentencepiece_input(text, options));
        let text = sentencepiece_text.as_deref().unwrap_or(text);
        // GGUF flags control BOS insertion in TokenizerBundle; disabling the HF
        // post-processor here prevents serialized templates from inserting it twice.
        self.tokenizer
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| format!("GGUF tokenization failed: {error}"))
    }

    /// Id of a special token with exactly this literal text (for example `</think>`).
    pub fn special_token_id(&self, literal: &str) -> Option<u32> {
        self.special
            .iter()
            .find(|(text, _)| text == literal)
            .map(|(_, id)| *id)
    }

    /// Decodes like `decode(ids, true)` but writes the given marker tokens as their literal text,
    /// so reasoning tags that are special tokens stay visible to the chat parser.
    pub fn decode_keeping(&self, ids: &[u32], keep: &[(u32, String)]) -> Result<String, String> {
        if keep.is_empty() {
            return self.decode(ids, true);
        }
        let mut output = String::new();
        let mut start = 0;
        for (index, id) in ids.iter().enumerate() {
            if let Some((_, literal)) = keep.iter().find(|(marker, _)| marker == id) {
                output.push_str(&self.decode(&ids[start..index], true)?);
                output.push_str(literal);
                start = index + 1;
            }
        }
        output.push_str(&self.decode(&ids[start..], true)?);
        Ok(output)
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String, String> {
        if !skip_special || self.skipped_special_ids.is_empty() {
            return self
                .tokenizer
                .decode(ids, skip_special)
                .map_err(|error| format!("GGUF detokenization failed: {error}"));
        }
        let mut output = String::new();
        let mut ordinary = Vec::new();
        for id in ids {
            if self.skipped_special_ids.contains(id) {
                if !ordinary.is_empty() {
                    output.push_str(
                        &self
                            .tokenizer
                            .decode(&ordinary, true)
                            .map_err(|error| format!("GGUF detokenization failed: {error}"))?,
                    );
                    ordinary.clear();
                }
            } else {
                ordinary.push(*id);
            }
        }
        if !ordinary.is_empty() {
            output.push_str(
                &self
                    .tokenizer
                    .decode(&ordinary, true)
                    .map_err(|error| format!("GGUF detokenization failed: {error}"))?,
            );
        }
        Ok(output)
    }
}

/// Regular expressions llama.cpp applies, in order, before byte-level BPE for a given
/// `tokenizer.ggml.pre` value (llama.cpp `llm_tokenizer_bpe::regex_exprs`). Every entry of
/// the returned slice is one split pass over the text, in llama.cpp order. `None` means the
/// classic GPT-2 splitting, which `build_legacy_bpe` announces with a warning.
fn pre_patterns(pre_type: &str) -> Option<&'static [&'static str]> {
    const LLAMA3: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const QWEN2: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const K2: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?(?:\p{L}|\p{M}|\u200C|\u200D)+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const GPT4O: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const DEEPSEEK3: [&str; 3] = [
        r"\p{N}{1,3}",
        r"[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+",
        r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##,
    ];
    const DEFAULT: [&str; 4] = [
        r"[\p{P}\$\+<=>\^~\|]+",
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)",
        r"\p{N}+",
        r"[0-9][0-9][0-9]",
    ];
    const MINICPM5: [&str; 2] = [
        r"\p{N}{1,3}",
        r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    ];
    const DEEPSEEK_LLM: [&str; 6] = [
        r"[\r\n]",
        r"\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+",
        r"\s?[!-/:-~！-／：-～‘-‟　-。]+",
        r"\s+$",
        r"[一-龥ࠀ-一가-퟿]+",
        r"\p{N}+",
    ];
    const DEEPSEEK_CODER: [&str; 5] = [
        r"[\r\n]",
        r"\s?\p{L}+",
        r"\s?\p{P}+",
        r"[一-龥ࠀ-一가-퟿]+",
        r"\p{N}",
    ];
    const SPARK2_5: [&str; 4] = [
        r"\p{N}{1,3}",
        r"[一-龥぀-ゟ゠-ヿ]+",
        r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+|[\r\n]|\s+(?!\S)|\s+"##,
        r"\p{N}",
    ];
    const YOUTU: [&str; 2] = [
        r"[가-힣ㄱ-ㆎ]+|[！…“”‘’—：；，、-〿︰-﹏]+|[ㄅ-ㄯ]+|[一-龥぀-ゟ゠-ヿ]+",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    ];
    const FALCON: [&str; 3] = [
        r"[\p{P}\$\+<=>\^~\|`]+",
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)",
        r"[0-9][0-9][0-9]",
    ];
    const GPT2: [&str; 1] = [
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)",
    ];
    const STARCODER: [&str; 2] = [
        r"\p{N}",
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)",
    ];
    const JAIS2: [&str; 1] = [
        r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s{512}(?!\S)|\s{256}(?!\S)|\s{128}(?!\S)|\s{64}(?!\S)|\s{32}(?!\S)|\s{16}(?!\S)|\s{8}(?!\S)|\s{4}(?!\S)|\s{1,2}(?!\S)|\s{1}",
    ];
    const PORO: [&str; 1] = [
        r" ?[^(\s|.,!?…。，、।۔،)]+",
    ];
    const VIKING: [&str; 2] = [
        r" ?[^(\s|.,!?…。，、।۔،)]+",
        r"\p{N}",
    ];
    const TEKKEN: [&str; 1] = [
        r"[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))*((?=[\p{L}])([^A-Z]))+|[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))+((?=[\p{L}])([^A-Z]))*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    ];
    const EXAONE_MOE: [&str; 1] = [
        r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?(?:\p{L}\p{M}*(?: \p{L}\p{M}*)*)+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]?|\s*[\r\n]|\s+(?!\S)|\s+",
    ];
    const CHAMELEON: [&str; 6] = [
        r"<sentinel:[0-9]+>",
        r"(IMGIMG)((A|B|C|D|E|F|G|H|I){1,4})Z",
        r"([\t\n]|    |  )",
        r"\p{N}",
        r"[\p{P}!-/:-@\[-`{-~]",
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)",
    ];
    const GRANITE_EMB_MULTI: [&str; 1] = [
        r"[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))*((?=[\p{L}\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))+((?=[\p{L}\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    ];
    const TINY_AYA: [&str; 2] = [
        r"\d{1,3}(?=(?:\d{3})*\b)",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    ];
    const BAILINGMOE: [&str; 1] = [
        r"'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+",
    ];
    const SEED_CODER: [&str; 1] = [
        r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1}| ?[^\s\p{L}\p{N}\r\n]+|\s*[\r\n]+|\s+(?!\S)|\s+",
    ];
    const UFAKZEKA: [&str; 1] = [
        r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    ];
    const LAGUNA: [&str; 2] = [
        r"[^\n]+|[\n]+",
        r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    ];
    match pre_type {
        "llama-bpe" | "llama3" | "dbrx" | "smaug-bpe" | "llama-v3" | "falcon3" | "falcon-h1" | "pixtral" | "midm-2.0" | "lfm2" | "jina-v5-nano" | "chatglm-bpe" | "glm4" | "glm5" => Some(&[LLAMA3]),
        "qwen2" | "qwen35" | "qwen3" | "deepseek-r1-qwen" | "kormo" | "f2llmv2" | "stablelm2" | "megrez" | "hunyuan" | "grok-2" | "solar-open" => Some(&[QWEN2]),
        "k2-horizon" | "k2_horizon" => Some(&[K2]),
        "gpt-4o" | "llama4" | "kimi-k2" | "minimax-m2" | "kanana2" | "talkie" => Some(&[GPT4O]),
        "deepseek-v3" | "hunyuan-dense" | "hy_v4" | "joyai-llm" => Some(&DEEPSEEK3),
        "default" => Some(&DEFAULT),
        "minicpm5" => Some(&MINICPM5),
        "deepseek-llm" => Some(&DEEPSEEK_LLM),
        "deepseek-coder" => Some(&DEEPSEEK_CODER),
        "spark2_5" => Some(&SPARK2_5),
        "youtu" => Some(&YOUTU),
        "falcon" => Some(&FALCON),
        "mpt" | "gpt-2" | "phi-2" | "jina-es" | "jina-de" | "gigachat" | "jina-v2-es" | "jina-v2-de" | "a.x-4.0" | "mellum" | "modern-bert" | "jina-v1-en" | "jina-v2-code" | "roberta-bpe" | "olmo" | "jais" | "exaone4" | "trillion" | "granite-docling" => Some(&GPT2),
        "starcoder" | "refact" | "command-r" | "smollm" | "codeshell" | "exaone" | "minerva-7b" | "mellum2" => Some(&STARCODER),
        "jais-2" => Some(&JAIS2),
        "poro-chat" | "bloom" | "gpt3-finnish" => Some(&PORO),
        "viking" => Some(&VIKING),
        "tekken" => Some(&TEKKEN),
        "exaone-moe" => Some(&EXAONE_MOE),
        "chameleon" => Some(&CHAMELEON),
        "granite-embed-multi-97m" => Some(&GRANITE_EMB_MULTI),
        "tiny_aya" | "cohere2moe" => Some(&TINY_AYA),
        "bailingmoe" | "bailingmoe2" | "llada-moe" => Some(&BAILINGMOE),
        "seed-coder" => Some(&SEED_CODER),
        "ufakzeka" => Some(&UFAKZEKA),
        "laguna" => Some(&LAGUNA),
        _ => None,
    }
}

/// The tokenizer of Gemma 4 (`tokenizer.ggml.model == "gemma4"`): BPE over the vocabulary with the merges of the file, spaces
/// written as `▁` (a normalizer, as in the checkpoint's tokenizer.json), `<0xHH>` byte tokens for what the vocabulary lacks, and
/// no prefix space.
fn build_gemma4_bpe(metadata: &serde_json::Map<String, Value>, tokens: &[String]) -> Result<Tokenizer, String> {
    use tokenizers::decoders::DecoderWrapper;
    use tokenizers::normalizers::Replace;
    let mut vocab = ahash::AHashMap::with_capacity(tokens.len());
    for (id, token) in tokens.iter().enumerate() {
        vocab.insert(token.clone(), u32::try_from(id).map_err(|_| "GGUF vocabulary exceeds u32 IDs")?);
    }
    let merges = metadata
        .get("tokenizer.ggml.merges")
        .and_then(Value::as_array)
        .ok_or("the Gemma 4 GGUF has no tokenizer.ggml.merges")?
        .iter()
        .filter_map(Value::as_str)
        .filter_map(|merge| {
            let (left, right) = merge.split_once(' ')?;
            let joined = format!("{left}{right}");
            (vocab.contains_key(left) && vocab.contains_key(right) && vocab.contains_key(&joined))
                .then(|| (left.to_owned(), right.to_owned()))
        })
        .collect::<Vec<_>>();
    let unknown = metadata
        .get("tokenizer.ggml.unknown_token_id")
        .and_then(Value::as_u64)
        .and_then(|id| tokens.get(id as usize).cloned())
        .unwrap_or_else(|| "<unk>".to_owned());
    let model = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .byte_fallback(true)
        .unk_token(unknown)
        .fuse_unk(true)
        .ignore_merges(false)
        .build()
        .map_err(|error| format!("cannot build the Gemma 4 tokenizer: {error}"))?;
    let mut tokenizer = Tokenizer::new(model);
    tokenizer
        .with_normalizer(Some(Replace::new(" ", "▁").map_err(|error| error.to_string())?))
        .map_err(|error| error.to_string())?;
    tokenizer.with_decoder(Some(DecoderSequence::new(vec![
        DecoderWrapper::Replace(Replace::new("▁", " ").map_err(|error| error.to_string())?),
        DecoderWrapper::ByteFallback(ByteFallback::default()),
        DecoderWrapper::Fuse(Fuse::default()),
    ])));
    Ok(tokenizer)
}

fn build_legacy_bpe(
    metadata: &serde_json::Map<String, Value>,
    tokens: &[String],
    byte_level: bool,
    pre_type: &str,
) -> Result<Tokenizer, String> {
    let mut vocab = ahash::AHashMap::with_capacity(tokens.len());
    for (id, token) in tokens.iter().enumerate() {
        let id = u32::try_from(id).map_err(|_| "GGUF vocabulary exceeds u32 IDs")?;
        vocab.insert(token.clone(), id);
    }
    let merges = metadata
        .get("tokenizer.ggml.merges")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter_map(|merge| {
            let (left, right) = merge.split_once(' ')?;
            // Some checkpoints (GLM-4) list merges whose result is not a vocabulary entry; such a merge can never
            // produce a token, so it is dropped instead of failing the whole tokenizer.
            let joined = format!("{left}{right}");
            (vocab.contains_key(left) && vocab.contains_key(right) && vocab.contains_key(&joined))
                .then(|| (left.to_owned(), right.to_owned()))
        })
        .collect::<Vec<_>>();
    let model = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .ignore_merges(false)
        .build()
        .map_err(|error| format!("cannot build GGUF BPE tokenizer: {error}"))?;
    let mut tokenizer = Tokenizer::new(model);
    if let Some(patterns) = pre_patterns(pre_type) {
        let mut steps = Vec::new();
        for pattern in patterns {
            let split = Split::new(
                SplitPattern::Regex((*pattern).to_string()),
                SplitDelimiterBehavior::Isolated,
                false,
            )
            .map_err(|error| format!("cannot create {pre_type} split pre-tokenizer: {error}"))?;
            steps.push(split.into());
        }
        steps.push(ByteLevel::new(false, true, false).into());
        tokenizer.with_pre_tokenizer(Some(Sequence::new(steps)));
        tokenizer.with_decoder(Some(ByteLevelDecoder::new(false, true, true)));
    } else if byte_level {
        if !pre_type.is_empty() {
            warn_unknown_pre(pre_type);
        }
        tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));
        tokenizer.with_decoder(Some(ByteLevelDecoder::new(false, true, true)));
    } else {
        tokenizer.with_decoder(Some(Fuse::default()));
    }
    Ok(tokenizer)
}

/// Announces an unknown `tokenizer.ggml.pre` before the tokenizer falls back to the GPT-2
/// splitting, so a new pre-tokenizer type shows up in the log instead of silently producing
/// different tokens. Every value is reported once per process.
fn warn_unknown_pre(pre_type: &str) {
    use std::sync::{Mutex, OnceLock};
    static WARNED: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let warned = WARNED.get_or_init(|| Mutex::new(HashSet::new()));
    let mut warned = warned.lock().unwrap_or_else(|error| error.into_inner());
    if warned.insert(pre_type.to_owned()) {
        eprintln!(
            "WARNING: unknown tokenizer.ggml.pre '{pre_type}': falling back to GPT-2 splitting. Token ids can differ from the model's \
             reference tokenizer (digits, whitespace, punctuation); convert the checkpoint with fllm-rs convert (it stores the exact \
             tokenizer) or report the name"
        );
    }
}

fn tokenizer_from_hf_json(
    bytes: &[u8],
    gguf_tokens: &[String],
    token_types: Option<&[Value]>,
) -> Result<Tokenizer, String> {
    let tokenizer = Tokenizer::from_bytes(bytes)
        .map_err(|error| format!("cannot parse tokenizer.huggingface.json: {error}"))?;
    let vocab = tokenizer.get_vocab(true);
    if vocab.len() > gguf_tokens.len()
        || token_types.is_some_and(|types| types.len() != gguf_tokens.len())
    {
        return Err(format!(
            "tokenizer.huggingface.json vocabulary has {} entries but GGUF has {}",
            vocab.len(),
            gguf_tokens.len()
        ));
    }
    for (id, token) in gguf_tokens.iter().enumerate() {
        let expected_id = u32::try_from(id).map_err(|_| "GGUF vocabulary exceeds u32 IDs")?;
        if vocab.get(token).copied() != Some(expected_id) {
            // HF embeddings often have trailing alignment rows. Only metadata
            // explicitly marked UNUSED may extend the exact tokenizer mapping.
            if !vocab.contains_key(token)
                && token_types
                    .and_then(|types| types.get(id))
                    .and_then(Value::as_u64)
                    == Some(5)
            {
                continue;
            }
            return Err(format!(
                "tokenizer.huggingface.json vocabulary ID mismatch at GGUF ID {id} for token {token:?}"
            ));
        }
    }
    for (token, &id) in &vocab {
        if gguf_tokens.get(id as usize) != Some(token) {
            return Err(format!(
                "tokenizer.huggingface.json token {token:?} ID {id} has no matching GGUF token"
            ));
        }
    }
    Ok(tokenizer)
}

fn build_unigram(
    metadata: &serde_json::Map<String, Value>,
    tokens: &[String],
    model_name: &str,
) -> Result<(Tokenizer, SentencePieceOptions, bool, Option<SpmEncoder>), String> {
    let charsmap = metadata
        .get("tokenizer.ggml.precompiled_charsmap")
        .map(|value| {
            value
                .as_array()
                .ok_or_else(|| "tokenizer.ggml.precompiled_charsmap must be an array".to_owned())
        })
        .transpose()?;
    // SentencePiece's normalization table (nmt_nfkc): T5-style models (llama.cpp's UGM tokenizer) apply it; the
    // `llama` model name is tokenized like llama.cpp's SPM tokenizer, which ignores the table.
    let charsmap_bytes: Vec<u8> = charsmap
        .map(|values| {
            values
                .iter()
                .map(|v| v.as_u64().and_then(|b| u8::try_from(b).ok()).ok_or_else(|| "tokenizer.ggml.precompiled_charsmap must hold bytes".to_owned()))
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_default();

    let scores = if let Some(values) = metadata.get("tokenizer.ggml.scores") {
        let values = values
            .as_array()
            .ok_or_else(|| "tokenizer.ggml.scores must be an array".to_owned())?;
        if values.len() != tokens.len() {
            return Err("tokenizer.ggml.scores must match the token vocabulary length".into());
        }
        values
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .filter(|score| score.is_finite())
                    .ok_or_else(|| "GGUF Unigram scores must be finite numbers".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        vec![0.0; tokens.len()]
    };

    let token_types = metadata
        .get("tokenizer.ggml.token_type")
        .map(|value| {
            value
                .as_array()
                .ok_or_else(|| "tokenizer.ggml.token_type must be an array".to_owned())
        })
        .transpose()?;
    let token_types = token_types.map(Vec::as_slice);
    if let Some(types) = token_types {
        if types.len() != tokens.len() {
            return Err("GGUF token_types must match the token vocabulary length".into());
        }
    }
    let explicit_unknown = metadata
        .get("tokenizer.ggml.unknown_token_id")
        .map(|value| {
            value.as_u64().ok_or_else(|| {
                "tokenizer.ggml.unknown_token_id must be an unsigned integer".to_owned()
            })
        })
        .transpose()?;
    let unknown_id = explicit_unknown
        .or_else(|| {
            token_types.and_then(|types| {
                types
                    .iter()
                    .position(|token_type| token_type.as_i64() == Some(2))
                    .map(|id| id as u64)
            })
        })
        .or(match model_name {
            // Defaults from llama.cpp's GGUF vocabulary loader for these two
            // model names; explicit GGUF unknown_token_id takes precedence.
            "llama" => Some(0),
            "t5" => Some(2),
            _ => None,
        })
        .ok_or_else(|| "GGUF Unigram tokenizer has no unknown_token_id".to_owned())?;
    let unknown_id =
        usize::try_from(unknown_id).map_err(|_| "GGUF Unigram unknown_token_id exceeds usize")?;
    if unknown_id >= tokens.len() {
        return Err(format!(
            "GGUF Unigram unknown_token_id {unknown_id} is outside the vocabulary"
        ));
    }

    let has_byte_tokens = has_complete_byte_fallback_vocab(tokens);
    let byte_fallback = match metadata.get("tokenizer.ggml.byte_fallback") {
        Some(value) => value
            .as_bool()
            .ok_or_else(|| "tokenizer.ggml.byte_fallback must be a boolean".to_owned())?,
        None => has_byte_tokens,
    };
    if byte_fallback && !has_byte_tokens {
        return Err(
            "GGUF Unigram byte_fallback is enabled but the vocabulary lacks all 256 <0xHH> tokens"
                .into(),
        );
    }
    if let Some(types) = token_types {
        let declares_byte_tokens = types.iter().any(|kind| kind.as_i64() == Some(6));
        if declares_byte_tokens && !has_byte_tokens {
            return Err(
                "GGUF declares byte tokens but does not contain a complete byte-fallback vocabulary"
                    .into(),
            );
        }
    }

    let spm = (model_name == "llama").then(|| SpmEncoder::new(tokens, &scores, unknown_id as u32));
    let pieces = tokens.iter().cloned().zip(scores).collect::<Vec<_>>();
    let model = Unigram::from(pieces, Some(unknown_id), byte_fallback)
        .map_err(|error| format!("cannot build GGUF Unigram tokenizer: {error}"))?;
    let mut tokenizer = Tokenizer::new(model);
    let add_space_default = model_name == "llama";
    let options = SentencePieceOptions {
        add_space_prefix: metadata_bool(
            metadata,
            "tokenizer.ggml.add_space_prefix",
            add_space_default,
        )?,
        remove_extra_whitespaces: metadata_bool(
            metadata,
            "tokenizer.ggml.remove_extra_whitespaces",
            false,
        )?,
    };
    if model_name == "t5" && !charsmap_bytes.is_empty() {
        use base64::Engine as _;
        let table = serde_json::json!({
            "type": "Precompiled",
            "precompiled_charsmap": base64::engine::general_purpose::STANDARD.encode(&charsmap_bytes),
        });
        // the tokenizers crate panics instead of failing on a malformed table
        let normalizer = std::panic::catch_unwind(|| serde_json::from_value::<tokenizers::NormalizerWrapper>(table))
            .map_err(|_| "tokenizer.ggml.precompiled_charsmap is not a valid SentencePiece normalization table".to_owned())?
            .map_err(|error| format!("invalid tokenizer.ggml.precompiled_charsmap: {error}"))?;
        tokenizer
            .with_normalizer(Some(normalizer))
            .map_err(|error| format!("cannot use tokenizer.ggml.precompiled_charsmap: {error}"))?;
    }
    // Whitespace cleanup and the initial prefix are applied in
    // encode_plain; Never here prevents the pre-tokenizer adding it twice.
    tokenizer.with_pre_tokenizer(Some(Metaspace::new('▁', PrependScheme::Never, true)));
    let decoder_scheme = if options.add_space_prefix {
        PrependScheme::First
    } else {
        PrependScheme::Never
    };
    if byte_fallback {
        tokenizer.with_decoder(Some(DecoderSequence::new(vec![
            ByteFallback::new().into(),
            MetaspaceDecoder::new('▁', decoder_scheme, true).into(),
        ])));
    } else {
        tokenizer.with_decoder(Some(MetaspaceDecoder::new('▁', decoder_scheme, true)));
    }
    Ok((tokenizer, options, byte_fallback, spm))
}

/// SentencePiece tokenization exactly as llama.cpp does it (`llm_tokenizer_spm`): every space becomes U+2581, the
/// text is cut into characters, and the adjacent pair whose concatenation is a piece with the highest score is
/// merged first (the leftmost one among equal scores) until no pair is left. Characters that are not pieces fall
/// back to their `<0xHH>` byte tokens. There is no pre-tokenizer: newlines, tabs and runs of spaces stay in the text.
struct SpmEncoder {
    vocab: ahash::AHashMap<String, u32>,
    scores: Vec<f32>,
    bytes: [Option<u32>; 256],
    unknown: u32,
}

impl SpmEncoder {
    fn new(tokens: &[String], scores: &[f64], unknown: u32) -> Self {
        let mut vocab = ahash::AHashMap::with_capacity(tokens.len());
        let mut bytes = [None; 256];
        for (id, token) in tokens.iter().enumerate() {
            vocab.entry(token.clone()).or_insert(id as u32);
            if let Some(byte) = byte_fallback_token(token) {
                bytes[byte as usize].get_or_insert(id as u32);
            }
        }
        Self {
            vocab,
            scores: scores.iter().map(|&s| s as f32).collect(),
            bytes,
            unknown,
        }
    }

    fn encode(&self, text: &str, add_space_prefix: bool) -> Vec<u32> {
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;
        if text.is_empty() {
            return Vec::new();
        }
        let mut escaped = String::with_capacity(text.len() + 4);
        if add_space_prefix {
            escaped.push('\u{2581}');
        }
        escaped.extend(text.chars().map(|c| if c == ' ' { '\u{2581}' } else { c }));

        struct Symbol {
            text: String,
            prev: isize,
            next: isize,
        }
        let mut symbols = escaped
            .chars()
            .enumerate()
            .map(|(i, c)| Symbol {
                text: c.to_string(),
                prev: i as isize - 1,
                next: i as isize + 1,
            })
            .collect::<Vec<_>>();
        if let Some(last) = symbols.last_mut() {
            last.next = -1;
        }
        #[derive(PartialEq)]
        struct Bigram {
            score: f32,
            left: usize,
            right: usize,
            size: usize,
        }
        impl Eq for Bigram {}
        impl PartialOrd for Bigram {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for Bigram {
            // max-heap: higher score first, then the smaller left index
            fn cmp(&self, other: &Self) -> Ordering {
                self.score.total_cmp(&other.score).then(other.left.cmp(&self.left))
            }
        }
        let mut heap = BinaryHeap::new();
        let push = |heap: &mut BinaryHeap<Bigram>, symbols: &[Symbol], left: isize, right: isize| {
            if left < 0 || right < 0 {
                return;
            }
            let (l, r) = (left as usize, right as usize);
            let mut joined = String::with_capacity(symbols[l].text.len() + symbols[r].text.len());
            joined.push_str(&symbols[l].text);
            joined.push_str(&symbols[r].text);
            if let Some(&id) = self.vocab.get(&joined) {
                heap.push(Bigram {
                    score: self.scores[id as usize],
                    left: l,
                    right: r,
                    size: joined.len(),
                });
            }
        };
        for i in 1..symbols.len() {
            push(&mut heap, &symbols, i as isize - 1, i as isize);
        }
        while let Some(top) = heap.pop() {
            let (l, r) = (top.left, top.right);
            if symbols[l].text.is_empty()
                || symbols[r].text.is_empty()
                || symbols[l].text.len() + symbols[r].text.len() != top.size
            {
                continue;
            }
            let right_text = std::mem::take(&mut symbols[r].text);
            symbols[l].text.push_str(&right_text);
            let next = symbols[r].next;
            symbols[l].next = next;
            if next >= 0 {
                symbols[next as usize].prev = l as isize;
            }
            let prev = symbols[l].prev;
            push(&mut heap, &symbols, prev, l as isize);
            push(&mut heap, &symbols, l as isize, next);
        }
        let mut ids = Vec::new();
        let mut at: isize = if symbols.is_empty() { -1 } else { 0 };
        while at >= 0 {
            let symbol = &symbols[at as usize];
            if let Some(&id) = self.vocab.get(&symbol.text) {
                ids.push(id);
            } else {
                for byte in symbol.text.bytes() {
                    ids.push(self.bytes[byte as usize].unwrap_or(self.unknown));
                }
            }
            at = symbol.next;
        }
        ids
    }
}

fn metadata_bool(
    metadata: &serde_json::Map<String, Value>,
    key: &str,
    default: bool,
) -> Result<bool, String> {
    metadata
        .get(key)
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| format!("{key} must be a boolean"))
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}

fn normalize_sentencepiece_input(text: &str, options: SentencePieceOptions) -> String {
    if options.remove_extra_whitespaces {
        let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
        if collapsed.is_empty() {
            String::new()
        } else {
            // GGML's SentencePiece normalization emits one leading space for
            // each non-whitespace span when remove_extra_whitespaces is set.
            format!(" {collapsed}")
        }
    } else {
        let normalized = text
            .chars()
            .map(|ch| if ch.is_whitespace() { ' ' } else { ch })
            .collect::<String>();
        if options.add_space_prefix && normalized.chars().any(|ch| !ch.is_whitespace()) {
            format!(" {normalized}")
        } else {
            normalized
        }
    }
}

fn has_complete_byte_fallback_vocab(tokens: &[String]) -> bool {
    let mut found = [false; 256];
    for token in tokens {
        if let Some(byte) = byte_fallback_token(token) {
            found[byte as usize] = true;
        }
    }
    found.into_iter().all(|present| present)
}

/// Resolves the FIM control tokens of a checkpoint: the GGUF metadata keys first (as llama.cpp
/// does), then the literals of the known code-model families.
fn fim_tokens(
    metadata: &serde_json::Map<String, Value>,
    special: &[(String, u32)],
) -> Option<Fim> {
    let literal = |text: &str| -> Option<u32> {
        special
            .iter()
            .find(|(token, _)| token == text)
            .map(|(_, id)| *id)
    };
    let id = |key: &str| -> Option<u32> {
        metadata
            .get(key)
            .and_then(Value::as_u64)
            .and_then(|id| u32::try_from(id).ok())
    };
    if let (Some(prefix), Some(suffix), Some(middle)) = (
        id("tokenizer.ggml.fim_pre_token_id"),
        id("tokenizer.ggml.fim_suf_token_id"),
        id("tokenizer.ggml.fim_mid_token_id"),
    ) {
        let family = FIM_LITERALS
            .iter()
            .find(|(_, markers)| {
                [prefix, suffix, middle]
                    .iter()
                    .zip(markers)
                    .all(|(token, marker)| literal(marker) == Some(*token))
            })
            .map_or(FimFamily::Qwen, |(family, _)| *family);
        return Some(Fim {
            prefix,
            suffix,
            middle,
            pad: id("tokenizer.ggml.fim_pad_token_id"),
            rep: id("tokenizer.ggml.fim_rep_token_id").or_else(|| literal("<|repo_name|>")),
            sep: id("tokenizer.ggml.fim_sep_token_id"),
            family,
        });
    }
    FIM_LITERALS
        .iter()
        .find_map(|(family, markers)| {
            Some(Fim {
                prefix: literal(markers[0])?,
                suffix: literal(markers[1])?,
                middle: literal(markers[2])?,
                pad: id("tokenizer.ggml.fim_pad_token_id"),
                rep: id("tokenizer.ggml.fim_rep_token_id").or_else(|| literal("<|repo_name|>")),
                sep: id("tokenizer.ggml.fim_sep_token_id"),
                family: *family,
            })
        })
}

fn byte_fallback_token(token: &str) -> Option<u8> {
    let digits = token.strip_prefix("<0x")?.strip_suffix('>')?;
    if digits.len() != 2 {
        return None;
    }
    u8::from_str_radix(digits, 16).ok()
}

fn byte_level_map() -> std::collections::HashMap<char, u8> {
    let mut bytes: Vec<u8> = (b'!'..=b'~')
        .chain(0xa1..=0xac)
        .chain(0xae..=0xff)
        .collect();
    let mut chars: Vec<u32> = bytes.iter().map(|b| *b as u32).collect();
    let mut next = 256;
    for byte in 0u8..=255 {
        if !bytes.contains(&byte) {
            bytes.push(byte);
            chars.push(next);
            next += 1;
        }
    }
    chars
        .into_iter()
        .zip(bytes)
        .map(|(c, b)| (char::from_u32(c).unwrap(), b))
        .collect()
}

fn special_ids(
    metadata: &serde_json::Map<String, Value>,
    tokens: &[String],
) -> Result<Vec<u32>, String> {
    let mut ids = Vec::new();
    if let Some(types) = metadata
        .get("tokenizer.ggml.token_type")
        .and_then(Value::as_array)
    {
        if types.len() != tokens.len() {
            return Err("GGUF token_types must match the token vocabulary length".to_owned());
        }
        for (index, token_type) in types.iter().enumerate() {
            if matches!(token_type.as_i64(), Some(3..=5)) {
                ids.push(index as u32);
            }
        }
    } else {
        for (index, token) in tokens.iter().enumerate() {
            if token.len() > 4 && token.starts_with("<|") && token.ends_with("|>") {
                ids.push(index as u32);
            }
        }
    }
    Ok(ids)
}

/// Whether a prompt starts with BOS when the GGUF does not say (`tokenizer.ggml.add_bos_token` missing): llama.cpp adds it
/// for SentencePiece vocabularies and the Llama-3 / Tekken style BPE ones; the other byte-level BPE models (gpt-oss,
/// Qwen, ...) are prompted without it.
fn default_add_bos(model: Option<&str>, pre: Option<&str>, has_bos: bool) -> bool {
    let spm = matches!(model, Some("llama" | "t5" | "gemma4"))
        || matches!(
            pre,
            Some(
                "llama3" | "llama-v3" | "llama-bpe" | "falcon3" | "falcon-h1" | "pixtral" | "midm-2.0" | "lfm2"
                    | "jina-v5-nano" | "tekken" | "chameleon"
            )
        );
    spm && has_bos
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn bos_default_follows_llama_cpp() {
        assert!(default_add_bos(Some("llama"), None, true));
        assert!(default_add_bos(Some("gpt2"), Some("llama-bpe"), true));
        assert!(default_add_bos(Some("gpt2"), Some("tekken"), true));
        assert!(!default_add_bos(Some("gpt2"), Some("gpt-4o"), true), "gpt-oss is prompted without BOS");
        assert!(!default_add_bos(Some("gpt2"), Some("qwen2"), true));
        assert!(!default_add_bos(Some("llama"), None, false), "no BOS token, nothing to add");
    }

    fn unigram_metadata(tokens: &[String], scores: Vec<f32>) -> serde_json::Map<String, Value> {
        let types = tokens
            .iter()
            .enumerate()
            .map(|(id, token)| {
                if id == 0 {
                    2
                } else if byte_fallback_token(token).is_some() {
                    6
                } else {
                    1
                }
            })
            .collect::<Vec<_>>();
        serde_json::from_value(json!({
            "tokenizer.ggml.scores": scores,
            "tokenizer.ggml.token_type": types,
            "tokenizer.ggml.unknown_token_id": 0,
            "tokenizer.ggml.byte_fallback": tokens.iter().any(|t| byte_fallback_token(t).is_some()),
            "tokenizer.ggml.add_space_prefix": false,
            "tokenizer.ggml.remove_extra_whitespaces": false,
        }))
        .unwrap()
    }

    #[test]
    fn hf_tokenizer_json_requires_exact_gguf_token_ids() {
        let json = br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"a":0,"b":1},"merges":[]}}"#;
        assert!(tokenizer_from_hf_json(json, &["a".into(), "b".into()], None).is_ok());
        let padded = ["a".into(), "b".into(), "<unused>".into()];
        assert!(
            tokenizer_from_hf_json(json, &padded, Some(&[json!(1), json!(1), json!(5)])).is_ok()
        );
        assert!(tokenizer_from_hf_json(json, &padded, None).is_err());
        assert!(
            tokenizer_from_hf_json(json, &padded, Some(&[json!(1), json!(1), json!(1)])).is_err()
        );
        let mismatch = tokenizer_from_hf_json(json, &["b".into(), "a".into()], None);
        match mismatch {
            Err(error) => assert!(error.contains("ID mismatch")),
            Ok(_) => panic!("expected HF/GGUF token ID mismatch to be rejected"),
        }
    }

    /// Compares the Gemma 4 tokenizer built from a GGUF with the checkpoint's own tokenizer.json (both files are needed:
    /// `FLLM_TEST_GEMMA4_GGUF` and `FLLM_TEST_GEMMA4_TOKENIZER`).
    #[test]
    #[ignore = "needs the Gemma 4 GGUF and tokenizer.json (set FLLM_TEST_GEMMA4_GGUF and FLLM_TEST_GEMMA4_TOKENIZER)"]
    fn gemma4_gguf_tokenizer_matches_the_checkpoint_tokenizer() {
        let (Ok(gguf), Ok(json)) = (std::env::var("FLLM_TEST_GEMMA4_GGUF"), std::env::var("FLLM_TEST_GEMMA4_TOKENIZER")) else {
            return;
        };
        let mapping = MappedGguf::open(std::path::Path::new(&gguf)).unwrap();
        let ours = TokenizerBundle::from_gguf(&mapping).unwrap();
        let reference = Tokenizer::from_file(&json).unwrap();
        for text in [
            "Hello, world!",
            "  leading and  double  spaces",
            "def f(x):\n    return x * 2\n",
            "Grüße aus München — 你好，世界 🙂",
            "line one\n\nline two\ttabbed",
            "<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n",
            "1234567890 3.14159",
        ] {
            let expected = reference.encode(text, false).unwrap().get_ids().to_vec();
            let got = ours.encode_without_bos(text).unwrap();
            assert_eq!(got, expected, "token ids differ for {text:?}");
            assert_eq!(ours.decode(&got, false).unwrap(), text, "round trip differs for {text:?}");
        }
    }

    #[test]
    fn gguf_llama_tokenizer_merges_by_score_like_llama_cpp() {
        // SentencePiece BPE vocabulary: every character and every intermediate merge is a piece, and the score
        // is the merge priority (higher first). "▁capital" must win over its characters even though the sum of
        // the character scores is higher.
        let tokens = [
            "<unk>", "▁hi", "▁hé", "▁a", "▁b", "▁", "a", "b", "h", "i", "é", "▁h", "▁c", "ap", "it", "al", "▁cap",
            "▁capit", "▁capital", "c", "p", "t", "l",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
        let scores = vec![
            0.0, -30.0, -31.0, -20.0, -21.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -10.0, -8.0, -9.0, -9.5, -9.7, -40.0,
            -50.0, -60.0, -1.0, -1.5, -1.7, -1.9,
        ];
        let mut metadata = unigram_metadata(&tokens, scores);
        metadata.insert("tokenizer.ggml.add_space_prefix".into(), Value::Bool(true));
        metadata.insert(
            "tokenizer.ggml.remove_extra_whitespaces".into(),
            Value::Bool(true),
        );
        let (_, options, _, spm) = build_unigram(&metadata, &tokens, "llama").unwrap();
        assert!(options.add_space_prefix);
        let spm = spm.expect("llama models use the SentencePiece encoder");
        assert_eq!(spm.encode("a b", true), vec![3, 4]);
        assert_eq!(spm.encode("hi", true), vec![1]);
        assert_eq!(spm.encode("h\u{e9}", true), vec![2]);
        assert_eq!(spm.encode("capital", true), vec![18]);
        // a space-less prefix and unknown characters
        assert_eq!(spm.encode("hi", false), vec![8, 9]);
        assert_eq!(spm.encode("z", false), vec![0]);
    }

    #[test]
    fn gguf_unigram_byte_fallback_roundtrips_utf8() {
        let mut tokens = vec!["<unk>".to_owned()];
        tokens.extend((0u16..=255).map(|byte| format!("<0x{byte:02X}>")));
        let metadata = unigram_metadata(&tokens, vec![0.0; tokens.len()]);
        let (tokenizer, _, byte_fallback, _) = build_unigram(&metadata, &tokens, "llama").unwrap();
        assert!(byte_fallback);
        let ids = tokenizer.encode("é", false).unwrap().get_ids().to_vec();
        assert_eq!(ids, vec![1u32 + 0xc3, 1u32 + 0xa9]);
        assert_eq!(tokenizer.decode(&ids, false).unwrap(), "é");
    }

    #[test]
    fn gguf_unigram_rejects_a_malformed_precompiled_normalizer() {
        let tokens = vec!["<unk>".to_owned(), "▁hi".to_owned()];
        let mut metadata = unigram_metadata(&tokens, vec![0.0, 1.0]);
        metadata.insert(
            "tokenizer.ggml.precompiled_charsmap".into(),
            json!([0, 1, 2]),
        );
        match build_unigram(&metadata, &tokens, "t5") {
            Err(error) => assert!(error.contains("precompiled_charsmap"), "{error}"),
            Ok(_) => panic!("expected a malformed normalization table to be rejected"),
        }
        // the llama.cpp SPM tokenizer behind the `llama` model name ignores the table
        assert!(build_unigram(&metadata, &tokens, "llama").is_ok());
    }

    #[test]
    fn gguf_unigram_applies_the_precompiled_normalizer_of_t5_models() {
        use base64::Engine as _;
        // the nmt_nfkc table of a SentencePiece model trained for the converter tests
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/tokenizers/spm_unigram_nfkc");
        let converted: Value = serde_json::from_str(&fllm_checkpoint::tokenizer_json_from_dir(&dir).unwrap()).unwrap();
        let table = converted["normalizer"]["normalizers"][0]["precompiled_charsmap"].as_str().expect("precompiled normalizer");
        let bytes = base64::engine::general_purpose::STANDARD.decode(table).unwrap();
        let tokens = ["<unk>", "▁", "f", "i", "▁fi"].map(str::to_owned).to_vec();
        let mut metadata = unigram_metadata(&tokens, vec![0.0, -2.0, -3.0, -3.0, -1.0]);
        metadata.insert("tokenizer.ggml.precompiled_charsmap".into(), json!(bytes));
        let (tokenizer, ..) = build_unigram(&metadata, &tokens, "t5").unwrap();
        // U+FB01 (the "fi" ligature) normalizes to "fi", the one-piece token
        assert_eq!(tokenizer.encode(" \u{fb01}", false).unwrap().get_ids(), &[4]);
        let (plain, ..) = build_unigram(&unigram_metadata(&tokens, vec![0.0, -2.0, -3.0, -3.0, -1.0]), &tokens, "t5").unwrap();
        assert_ne!(plain.encode(" \u{fb01}", false).unwrap().get_ids(), &[4]);
    }
}
