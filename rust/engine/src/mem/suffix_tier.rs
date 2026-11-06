//! The suffix cache on disk, so that speculation does not start cold after a restart.
//!
//! One file per model, `suffix-<digest>.bin` in the directory named by `FLLM_SUFFIX_DIR` (the tier is off without the variable). The
//! digest is the fingerprint of the model file (`ModelFacts::digest`): token ids are only meaningful for the tokenizer they came from, so
//! a file of another model is thrown away, not read. The index is not stored; it is rebuilt from the tokens on load.
//!
//! Layout, all integers little endian:
//!
//! ```text
//! "FLLMSUF1" | digest_len u32 | digest bytes | document_count u64
//! per document: salt u64 | token_count u32 | tokens i32 * token_count
//! ```
//!
//! The file is written under a temporary name and renamed, so a crash leaves the old file or none, never half of one. A truncated or
//! damaged file is treated as absent. Token ids are stored as `i32`; a document with a larger id (no model has a vocabulary of that
//! size) is not stored.
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"FLLMSUF1";
/// More documents than this in one file are a damaged file, not a cache (the budget of the cache bounds the real number).
const MAX_DOCUMENTS: u64 = 16 << 20;
const MAX_DIGEST: usize = 256;

/// A finished generation: the namespace it belongs to and its tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredDoc {
    pub salt: u64,
    pub tokens: Vec<i64>,
}

/// `<dir>/suffix-<digest>.bin`, or `None` when the digest has characters that do not belong into a file name.
pub fn path_for(dir: &Path, digest: &str) -> Option<PathBuf> {
    if digest.is_empty() || digest.len() > MAX_DIGEST || !digest.chars().all(|c| c.is_ascii_alphanumeric()) {
        return None;
    }
    Some(dir.join(format!("suffix-{digest}.bin")))
}

/// Encodes the documents of one model.
pub fn encode(digest: &str, docs: &[StoredDoc]) -> Vec<u8> {
    let usable: Vec<&StoredDoc> = docs.iter().filter(|d| d.tokens.iter().all(|&t| i32::try_from(t).is_ok())).collect();
    let mut out = Vec::with_capacity(32 + usable.iter().map(|d| 12 + 4 * d.tokens.len()).sum::<usize>());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(digest.len() as u32).to_le_bytes());
    out.extend_from_slice(digest.as_bytes());
    out.extend_from_slice(&(usable.len() as u64).to_le_bytes());
    for doc in usable {
        out.extend_from_slice(&doc.salt.to_le_bytes());
        out.extend_from_slice(&(doc.tokens.len() as u32).to_le_bytes());
        for &token in &doc.tokens {
            out.extend_from_slice(&(token as i32).to_le_bytes());
        }
    }
    out
}

fn take<'a>(bytes: &mut &'a [u8], n: usize) -> Option<&'a [u8]> {
    if bytes.len() < n {
        return None;
    }
    let (head, tail) = bytes.split_at(n);
    *bytes = tail;
    Some(head)
}

/// Decodes a file. `None` for anything that is not a complete file of this model: wrong magic, another digest, a truncated body, a
/// count that the rest of the file cannot hold, trailing bytes.
pub fn decode(digest: &str, mut bytes: &[u8]) -> Option<Vec<StoredDoc>> {
    if take(&mut bytes, 8)? != MAGIC {
        return None;
    }
    let digest_len = u32::from_le_bytes(take(&mut bytes, 4)?.try_into().ok()?) as usize;
    if digest_len > MAX_DIGEST || take(&mut bytes, digest_len)? != digest.as_bytes() {
        return None;
    }
    let count = u64::from_le_bytes(take(&mut bytes, 8)?.try_into().ok()?);
    // every document takes at least its 12 byte header: a larger count cannot be true
    if count > MAX_DOCUMENTS || count.saturating_mul(12) > bytes.len() as u64 {
        return None;
    }
    let mut docs = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let salt = u64::from_le_bytes(take(&mut bytes, 8)?.try_into().ok()?);
        let n = u32::from_le_bytes(take(&mut bytes, 4)?.try_into().ok()?) as usize;
        let raw = take(&mut bytes, n.checked_mul(4)?)?;
        let tokens = raw.as_chunks::<4>().0.iter().map(|c| i64::from(i32::from_le_bytes(*c))).collect();
        docs.push(StoredDoc { salt, tokens });
    }
    bytes.is_empty().then_some(docs)
}

/// Writes the file atomically.
pub fn save(path: &Path, digest: &str, docs: &[StoredDoc]) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let temporary = path.with_extension("bin.part");
    let written = fs::File::create(&temporary).and_then(|mut file| {
        file.write_all(&encode(digest, docs))?;
        file.sync_data()
    });
    if let Err(error) = written {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    fs::rename(&temporary, path)
}

/// Reads the file of this model. A file that cannot be trusted is removed, so that it is not tried again at every start.
pub fn load(path: &Path, digest: &str) -> Vec<StoredDoc> {
    let Ok(bytes) = fs::read(path) else { return Vec::new() };
    match decode(digest, &bytes) {
        Some(docs) => docs,
        None => {
            let _ = fs::remove_file(path);
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("fllm-suffix-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn docs() -> Vec<StoredDoc> {
        vec![
            StoredDoc { salt: 0, tokens: vec![10, 11, 12, 13] },
            StoredDoc { salt: 7, tokens: vec![1, 2, 3] },
            StoredDoc { salt: u64::MAX, tokens: vec![] },
            StoredDoc { salt: 3, tokens: vec![0, 151_935, 42] },
        ]
    }

    #[test]
    fn a_file_round_trips_with_its_salts() {
        let d = dir("roundtrip");
        let path = path_for(&d, "abc123").unwrap();
        save(&path, "abc123", &docs()).unwrap();
        assert_eq!(load(&path, "abc123"), docs());
        assert!(!path.with_extension("bin.part").exists(), "the temporary name is gone");
    }

    #[test]
    fn a_file_of_another_model_is_thrown_away() {
        let d = dir("foreign");
        let path = path_for(&d, "modela").unwrap();
        save(&path, "modela", &docs()).unwrap();
        assert_eq!(load(&path, "modela"), docs(), "the file is good for its own model");
        assert!(load(&path, "modelb").is_empty());
        assert!(!path.exists(), "a file that does not belong to this model is removed");
    }

    #[test]
    fn a_truncated_or_damaged_file_is_absent_and_removed() {
        let d = dir("damaged");
        let path = path_for(&d, "abc").unwrap();
        let good = encode("abc", &docs());
        for cut in [0usize, 5, 12, good.len() - 1, good.len() / 2] {
            fs::write(&path, &good[..cut]).unwrap();
            assert!(load(&path, "abc").is_empty(), "a file cut at {cut} was trusted");
            assert!(!path.exists());
        }
        let mut wrong_magic = good.clone();
        wrong_magic[0] ^= 0xff;
        fs::write(&path, &wrong_magic).unwrap();
        assert!(load(&path, "abc").is_empty());
        let mut trailing = good.clone();
        trailing.push(0);
        fs::write(&path, &trailing).unwrap();
        assert!(load(&path, "abc").is_empty(), "bytes after the last document mean the file is not what it says");
        // a count that the rest of the file cannot hold
        let mut huge = good.clone();
        huge[8 + 4 + 3..8 + 4 + 3 + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(decode("abc", &huge).is_none());
    }

    #[test]
    fn a_missing_file_is_an_empty_cache() {
        let d = dir("missing");
        assert!(load(&d.join("suffix-none.bin"), "none").is_empty());
    }

    #[test]
    fn a_digest_that_is_not_a_file_name_gets_no_path() {
        let d = dir("name");
        for bad in ["", "../escape", "a/b", "a b", "a\0b"] {
            assert!(path_for(&d, bad).is_none(), "{bad:?}");
        }
        assert!(path_for(&d, "0123abcDEF").is_some());
    }

    #[test]
    fn a_token_that_does_not_fit_in_32_bits_keeps_its_document_out() {
        let kept = StoredDoc { salt: 1, tokens: vec![5, 6] };
        let skipped = StoredDoc { salt: 2, tokens: vec![5, i64::from(i32::MAX) + 1] };
        let decoded = decode("d", &encode("d", &[kept.clone(), skipped])).unwrap();
        assert_eq!(decoded, vec![kept]);
    }
}
