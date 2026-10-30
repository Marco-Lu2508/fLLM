//! The RWKV "world" tokenizer (`tokenizer.ggml.model == "rwkv"`): the vocabulary is a list of byte strings and a text is cut
//! greedily into the longest vocabulary entry at every position; there is no pre-tokenisation and no merge table.
//!
//! llama.cpp files store each entry as the Python `repr` of its bytes without the quotes (`\x00`, `\n`, `\\`, `\'`, `\xc3\xbc`, plain
//! printable ASCII otherwise), which [`unescape`] reverses.
use std::collections::HashMap;

/// Bytes of a vocabulary entry written as the `repr` of a Python `bytes` object (without `b'` and `'`).
pub(crate) fn unescape(text: &str) -> Vec<u8> {
    let raw = text.as_bytes();
    let mut out = Vec::with_capacity(raw.len());
    let mut i = 0;
    while i < raw.len() {
        if raw[i] != b'\\' || i + 1 >= raw.len() {
            out.push(raw[i]);
            i += 1;
            continue;
        }
        let hex = |at: usize| raw.get(at).and_then(|&b| (b as char).to_digit(16));
        match raw[i + 1] {
            b'x' => match (hex(i + 2), hex(i + 3)) {
                (Some(hi), Some(lo)) => {
                    out.push((hi * 16 + lo) as u8);
                    i += 4;
                }
                _ => {
                    out.push(b'\\');
                    i += 1;
                }
            },
            b'n' => {
                out.push(b'\n');
                i += 2;
            }
            b't' => {
                out.push(b'\t');
                i += 2;
            }
            b'r' => {
                out.push(b'\r');
                i += 2;
            }
            other @ (b'\\' | b'\'' | b'"') => {
                out.push(other);
                i += 2;
            }
            _ => {
                out.push(b'\\');
                i += 1;
            }
        }
    }
    out
}

pub(crate) struct RwkvVocab {
    /// Bytes of every token id.
    tokens: Vec<Vec<u8>>,
    /// Trie over the token bytes: `(node << 8) | byte -> child`.
    children: HashMap<u64, u32>,
    /// Token that ends at a node (`u32::MAX`: none).
    ends: Vec<u32>,
}

impl RwkvVocab {
    pub(crate) fn new(escaped: &[String]) -> Result<Self, String> {
        let tokens: Vec<Vec<u8>> = escaped.iter().map(|t| unescape(t)).collect();
        let mut children = HashMap::new();
        let mut ends = vec![u32::MAX];
        for (id, bytes) in tokens.iter().enumerate() {
            if bytes.is_empty() {
                continue;
            }
            let mut node = 0u32;
            for &b in bytes {
                let key = (u64::from(node) << 8) | u64::from(b);
                node = *children.entry(key).or_insert_with(|| {
                    ends.push(u32::MAX);
                    (ends.len() - 1) as u32
                });
            }
            // the first entry wins when two ids spell the same bytes
            if ends[node as usize] == u32::MAX {
                ends[node as usize] = u32::try_from(id).map_err(|_| "RWKV vocabulary exceeds u32 IDs")?;
            }
        }
        Ok(Self { tokens, children, ends })
    }

    pub(crate) fn token_bytes(&self, id: u32) -> &[u8] {
        self.tokens.get(id as usize).map_or(&[], Vec::as_slice)
    }

    /// Greedy longest-match encoding. A byte that no entry starts with is skipped (the world vocabulary has all 256 single bytes).
    pub(crate) fn encode(&self, text: &str) -> Vec<u32> {
        let bytes = text.as_bytes();
        let mut ids = Vec::with_capacity(bytes.len() / 3 + 1);
        let mut at = 0;
        while at < bytes.len() {
            let (mut node, mut best, mut best_len) = (0u32, u32::MAX, 0usize);
            for (k, &b) in bytes[at..].iter().enumerate() {
                match self.children.get(&((u64::from(node) << 8) | u64::from(b))) {
                    Some(&next) => {
                        node = next;
                        if self.ends[node as usize] != u32::MAX {
                            best = self.ends[node as usize];
                            best_len = k + 1;
                        }
                    }
                    None => break,
                }
            }
            if best == u32::MAX {
                at += 1;
            } else {
                ids.push(best);
                at += best_len;
            }
        }
        ids
    }

    pub(crate) fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|&id| self.token_bytes(id).iter().copied()).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unescapes_python_bytes_reprs() {
        assert_eq!(unescape("\\x00"), vec![0]);
        assert_eq!(unescape("\\n\\t"), b"\n\t");
        assert_eq!(unescape("\\\\"), b"\\");
        assert_eq!(unescape("\\'"), b"'");
        assert_eq!(unescape(" G\\xc3\\xbc"), " Gü".as_bytes());
        assert_eq!(unescape("abc"), b"abc");
    }

    #[test]
    fn greedy_longest_match_and_round_trip() {
        let vocab: Vec<String> = ["<s>", "a", "b", "ab", "abc", "c", "\\xc3", "\\xbc", "\\xc3\\xbc", " "].iter().map(|s| s.to_string()).collect();
        let v = RwkvVocab::new(&vocab).unwrap();
        // "abc" is one token; "abab" cuts into ab, ab; "ü" is the two-byte entry
        assert_eq!(v.encode("abc"), vec![4]);
        assert_eq!(v.encode("abab"), vec![3, 3]);
        assert_eq!(v.encode("a ü"), vec![1, 9, 8]);
        assert_eq!(v.decode(&v.encode("ab c ü")), "ab c ü");
        // a prefix that does not complete falls back to the longest shorter entry
        assert_eq!(v.encode("abx"), vec![3]);
    }
}
