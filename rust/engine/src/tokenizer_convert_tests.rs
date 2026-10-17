//! Tokenizers converted from SentencePiece / tiktoken / tekken files must encode exactly like the reference
//! implementations. The fixtures in `testdata/tokenizers/` are tiny models trained by `scripts/make_tokenizer_fixtures.py`
//! together with the ids `sentencepiece`, `tiktoken` and `mistral_common` produce for a corpus of tricky strings.
use serde_json::Value;
use std::path::PathBuf;
use tokenizers::Tokenizer;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata").join("tokenizers").join(name)
}

fn check(name: &str) -> Tokenizer {
    check_except(name, &[])
}

/// `known` lists strings on which the converted tokenizer is known to differ (documented in docs/changes/formats.md).
fn check_except(name: &str, known: &[&str]) -> Tokenizer {
    let dir = fixture(name);
    let json = fllm_checkpoint::tokenizer_json_from_dir(&dir).unwrap_or_else(|e| panic!("{name}: {e}"));
    let tokenizer = Tokenizer::from_bytes(json.as_bytes()).unwrap_or_else(|e| panic!("{name}: converted JSON does not load: {e}"));
    let golden: Value = serde_json::from_slice(&std::fs::read(dir.join("golden.json")).unwrap()).unwrap();
    let strings = golden["strings"].as_array().unwrap();
    let ids = golden["ids"].as_array().unwrap();
    assert!(strings.len() > 100 && strings.len() == ids.len());
    let mut failures = Vec::new();
    for (s, expected) in strings.iter().zip(ids) {
        let s = s.as_str().unwrap();
        let expected: Vec<u32> = expected.as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let got = tokenizer.encode(s, false).unwrap().get_ids().to_vec();
        if got != expected && !known.contains(&s) {
            failures.push(format!("{s:?}\n  reference {expected:?}\n  converted {got:?}"));
        }
    }
    assert!(failures.is_empty(), "{name}: {} of {} strings differ:\n{}", failures.len(), strings.len(), failures.iter().take(6).cloned().collect::<Vec<_>>().join("\n"));
    tokenizer
}

#[test]
fn sentencepiece_unigram_with_nfkc_charsmap() {
    // a base letter followed by five combining acute accents: NFKC composes the first pair, the remaining marks are unknown;
    // SentencePiece then prefers [dummy prefix, composed letter] to the single piece "dummy prefix + composed letter"
    check_except("spm_unigram_nfkc", &["a\u{301}\u{301}\u{301}\u{301}\u{301}"]);
}

#[test]
fn sentencepiece_unigram_with_byte_fallback() {
    check("spm_unigram_bytefb");
}

#[test]
fn sentencepiece_bpe_llama_style() {
    let tokenizer = check("spm_bpe_llama");
    // user-defined symbols are matched in text; every segment between them gets the dummy prefix (transformers' legacy
    // Llama tokenizers and llama.cpp do the same), so the text after the symbol decodes with one more leading space
    let ids = tokenizer.encode("hi <|user|> there", false).unwrap();
    assert!(ids.get_tokens().iter().any(|t| t == "<|user|>"), "{:?}", ids.get_tokens());
    assert_eq!(tokenizer.decode(ids.get_ids(), false).unwrap(), "hi <|user|>  there");
    let plain = tokenizer.encode("no symbols here", false).unwrap();
    assert_eq!(tokenizer.decode(plain.get_ids(), false).unwrap(), "no symbols here");
}

#[test]
fn sentencepiece_bpe_plain() {
    check("spm_bpe_plain");
}

#[test]
fn tiktoken_ranks_with_cl100k_pattern() {
    let tokenizer = check("tiktoken_small");
    let ids = tokenizer.encode("a<|im_end|>b", false).unwrap();
    assert!(ids.get_ids().contains(&(tokenizer.token_to_id("<|im_end|>").unwrap())));
    let text = "Hello, world! \u{4f60}\u{597d} \u{1F600}";
    assert_eq!(tokenizer.decode(tokenizer.encode(text, false).unwrap().get_ids(), false).unwrap(), text);
}

#[test]
fn tekken_vocabulary_with_special_ids_first() {
    let tokenizer = check("tekken_small");
    let golden: Value = serde_json::from_slice(&std::fs::read(fixture("tekken_small").join("golden.json")).unwrap()).unwrap();
    for (name, id) in golden["specials"].as_object().unwrap() {
        assert_eq!(tokenizer.token_to_id(name), id.as_u64().map(|v| v as u32), "{name}");
    }
    assert_eq!(tokenizer.token_to_id("<s>"), Some(1));
}
