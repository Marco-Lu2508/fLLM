//! The RWKV "world" vocabulary (`rwkv_vocab_v20230424.txt` next to the weights of the Hugging Face RWKV repositories) as GGUF
//! tokenizer metadata (`tokenizer.ggml.model = rwkv`), in the form llama.cpp writes it: every token is the Python `repr` of its bytes
//! without the quotes.
//!
//! Every line of the file is `<id> <python literal> <byte length>`, the literal a `str` or a `bytes` literal.
use crate::gguf::Meta;
use std::path::Path;

/// Number of tokens and the metadata entries of the tokenizer.
pub(crate) type TokenizerMeta = (usize, Vec<(String, Meta)>);

const FILE: &str = "rwkv_vocab_v20230424.txt";

/// Bytes of a Python `str` or `bytes` literal (`'abc'`, `b'\x00'`, `"it's"`), with the usual escapes.
fn literal_bytes(text: &str) -> Result<Vec<u8>, String> {
    let text = text.trim();
    let (is_bytes, body) = match text.strip_prefix('b') {
        Some(rest) => (true, rest),
        None => (false, text),
    };
    let quote = body.chars().next().filter(|c| matches!(c, '\'' | '"')).ok_or_else(|| format!("RWKV vocabulary: not a literal: {text}"))?;
    let inner = body
        .strip_prefix(quote)
        .and_then(|b| b.strip_suffix(quote))
        .ok_or_else(|| format!("RWKV vocabulary: unterminated literal: {text}"))?;
    let mut out = Vec::new();
    let mut chars = inner.chars();
    let push = |c: char, out: &mut Vec<u8>| {
        if is_bytes {
            out.push(c as u32 as u8);
        } else {
            let mut buf = [0u8; 4];
            out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
        }
    };
    while let Some(c) = chars.next() {
        if c != '\\' {
            push(c, &mut out);
            continue;
        }
        let e = chars.next().ok_or("RWKV vocabulary: dangling backslash")?;
        let mut take = |n: usize| -> Result<u32, String> {
            let digits: String = chars.by_ref().take(n).collect();
            u32::from_str_radix(&digits, 16).map_err(|_| format!("RWKV vocabulary: bad escape in {text}"))
        };
        match e {
            'n' => push('\n', &mut out),
            't' => push('\t', &mut out),
            'r' => push('\r', &mut out),
            '\\' | '\'' | '"' => push(e, &mut out),
            'x' => {
                let v = take(2)?;
                if is_bytes {
                    out.push(v as u8);
                } else {
                    push(char::from_u32(v).ok_or("RWKV vocabulary: bad escape")?, &mut out);
                }
            }
            'u' => push(char::from_u32(take(4)?).ok_or("RWKV vocabulary: bad escape")?, &mut out),
            'U' => push(char::from_u32(take(8)?).ok_or("RWKV vocabulary: bad escape")?, &mut out),
            other => {
                push('\\', &mut out);
                push(other, &mut out);
            }
        }
    }
    Ok(out)
}

/// The `repr` of `bytes` without `b'` and `'` (what llama.cpp stores).
fn escape(bytes: &[u8]) -> String {
    let mut out = String::new();
    for &b in bytes {
        match b {
            b'\\' => out.push_str("\\\\"),
            b'\t' => out.push_str("\\t"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            0x20..=0x7e => out.push(b as char),
            _ => out.push_str(&format!("\\x{b:02x}")),
        }
    }
    out
}

/// Tokenizer metadata for a directory that holds the RWKV world vocabulary and no `tokenizer.json`; `None` for any other directory.
pub(crate) fn meta(dir: &Path, model_vocab: u64) -> Result<Option<TokenizerMeta>, String> {
    let path = dir.join(FILE);
    if dir.join("tokenizer.json").is_file() || !path.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path).map_err(|e| format!("read {FILE}: {e}"))?;
    let target = usize::try_from(model_vocab).map_err(|_| "vocabulary size does not fit usize")?;
    let mut tokens: Vec<Option<Vec<u8>>> = vec![None; target.max(1)];
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let (id, rest) = line.split_once(' ').ok_or_else(|| format!("RWKV vocabulary: bad line {line:?}"))?;
        let (literal, len) = rest.rsplit_once(' ').ok_or_else(|| format!("RWKV vocabulary: bad line {line:?}"))?;
        let id: usize = id.parse().map_err(|_| format!("RWKV vocabulary: bad id in {line:?}"))?;
        let bytes = literal_bytes(literal)?;
        if len.trim().parse::<usize>().ok() != Some(bytes.len()) {
            return Err(format!("RWKV vocabulary: length mismatch in {line:?}"));
        }
        if id >= tokens.len() {
            return Err(format!("RWKV vocabulary: id {id} is outside the model vocabulary of {} tokens", tokens.len()));
        }
        tokens[id] = Some(bytes);
    }
    // id 0 is the end-of-text marker; ids the file does not list are padding
    let pieces: Vec<String> = tokens
        .iter()
        .enumerate()
        .map(|(id, t)| match (id, t) {
            (0, _) => "<s>".to_owned(),
            (_, Some(bytes)) => escape(bytes),
            (_, None) => format!("[PAD{id}]"),
        })
        .collect();
    let types = (0..pieces.len()).map(|id| if id == 0 { 3 } else { 1 }).collect();
    let meta = vec![
        ("tokenizer.ggml.model".to_owned(), Meta::Str("rwkv".into())),
        ("tokenizer.ggml.tokens".to_owned(), Meta::Strs(pieces)),
        ("tokenizer.ggml.token_type".to_owned(), Meta::U32s(types)),
        ("tokenizer.ggml.eos_token_id".to_owned(), Meta::U32(0)),
        // "\n\n" ends a chat turn
        ("tokenizer.ggml.eot_token_id".to_owned(), Meta::U32(261)),
        ("tokenizer.chat_template".to_owned(), Meta::Str("rwkv-world".into())),
    ];
    Ok(Some((target, meta)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_literals_become_bytes_and_back() {
        assert_eq!(literal_bytes("b'\\x00'").unwrap(), vec![0]);
        assert_eq!(literal_bytes("'\\t'").unwrap(), b"\t");
        assert_eq!(literal_bytes("'\\xe4'").unwrap(), "\u{e4}".as_bytes());
        assert_eq!(literal_bytes("b'\\xe4'").unwrap(), vec![0xe4]);
        assert_eq!(literal_bytes("\"it's\"").unwrap(), b"it's");
        assert_eq!(literal_bytes("'\\u4e2d'").unwrap(), "\u{4e2d}".as_bytes());
        assert_eq!(escape(&[0, b'a', b'\n', b'\\', 0xc3, 0xbc]), "\\x00a\\n\\\\\\xc3\\xbc");
    }

    #[test]
    fn directory_without_the_file_is_not_rwkv() {
        let dir = std::env::temp_dir().join(format!("fllm-rwkv-vocab-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(meta(&dir, 8).unwrap().is_none());
        std::fs::write(dir.join(FILE), "1 b'\\x00' 1\n2 '\\n' 1\n3 'ab' 2\n").unwrap();
        let (count, meta) = meta(&dir, 6).unwrap().unwrap();
        assert_eq!(count, 6);
        let tokens = meta.iter().find_map(|(k, v)| (k == "tokenizer.ggml.tokens").then_some(v)).unwrap();
        match tokens {
            Meta::Strs(t) => assert_eq!(t, &["<s>", "\\x00", "\\n", "ab", "[PAD4]", "[PAD5]"]),
            _ => panic!("tokens are a string array"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
