//! The document search of the Studio (`rust/engine/src/webui/rag.js`, the part between `@pure-begin`
//! and `@pure-end`): a document is cut into passages of about 500 tokens and the passages are ranked
//! against the question with BM25, so no embedding model is needed. With vectors in the index the
//! ranking is hybrid: 60 % keyword, 40 % meaning. `tests/fixtures/rag_fixtures.json` holds the results
//! the original JavaScript produces (the generator is `tests/fixtures/rag_fixtures.js`); the tests
//! require the port to reproduce them.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

/// The same numbers as in rag.js.
pub const CHUNK_CHARS: usize = 1800;
pub const OVERLAP_CHARS: usize = 240;
/// The extracted text kept per document in the storage.
pub const MAX_DOC_CHARS: usize = 600_000;
/// The extracted text kept per chat.
pub const MAX_CHAT_CHARS: usize = 1_500_000;
/// A document goes into the message whole when it takes less than this share of the context.
pub const FIT_SHARE: f64 = 0.35;
/// A passage without a shared word needs this cosine to count as a hit.
pub const VEC_MIN: f64 = 0.3;
/// What a text file may weigh before the window refuses to read it (the Studio's `MAX_TEXT_ATTACHMENT`).
pub const MAX_TEXT_ATTACHMENT: u64 = 20 * 1024 * 1024;

/// The words that carry no meaning in a search (the same list as in rag.js).
const STOP: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at", "by", "for", "with", "from", "as", "is", "are", "was", "were",
    "be", "been", "it", "its", "this", "that", "these", "those", "i", "you", "he", "she", "we", "they", "me", "my", "your", "our", "their",
    "what", "which", "who", "whom", "how", "why", "when", "where", "do", "does", "did", "not", "no", "so", "than", "then", "there", "here",
    "can", "could", "will", "would", "should", "may", "might", "about", "into", "over", "under", "also", "just", "very", "der", "die",
    "das", "und", "oder", "aber", "ist", "sind", "war", "ein", "eine", "einer", "den", "dem", "des", "zu", "von", "mit", "auf", "fuer",
    "bei", "als", "wie", "le", "la", "les", "un", "une", "du", "de", "et", "en", "est",
];

static WORDS: OnceLock<regex::Regex> = OnceLock::new();
static PARA_SPLIT: OnceLock<regex::Regex> = OnceLock::new();

fn words() -> &'static regex::Regex {
    WORDS.get_or_init(|| regex::Regex::new(r"[\p{L}\p{N}]+").expect("word pattern"))
}

fn para_split() -> &'static regex::Regex {
    PARA_SPLIT.get_or_init(|| regex::Regex::new(r"\n\s*\n").expect("paragraph pattern"))
}

/// One word of a question or a document: lower-cased, without stop words, a trailing plural folded
/// (the `ragTokenize` of rag.js).
pub fn tokenize(text: &str) -> Vec<String> {
    // the Studio lower-cases the whole text first and then takes the words
    let lowered = text.to_lowercase();
    let mut out = Vec::new();
    for m in words().find_iter(&lowered) {
        let w = m.as_str();
        if STOP.contains(&w) {
            continue;
        }
        let chars: Vec<char> = w.chars().collect();
        let folded: String = if chars.len() > 3 && chars[chars.len() - 1] == 's' && chars[chars.len() - 2] != 's' {
            w[..w.len() - 1].to_string()
        } else {
            w.to_string()
        };
        let len = folded.chars().count();
        if len > 1 || folded.chars().any(|c| c.is_ascii_digit()) {
            out.push(folded);
        }
    }
    out
}

/// Cuts text into passages of `CHUNK_CHARS` characters; each passage starts with the tail of the
/// previous one (the `ragChunk` of rag.js).
pub fn chunk(text: &str) -> Vec<String> {
    chunk_with(text, CHUNK_CHARS, OVERLAP_CHARS)
}

pub fn chunk_with(text: &str, size: usize, overlap: usize) -> Vec<String> {
    let mut units: Vec<(String, &'static str)> = Vec::new();
    for para in para_split().split(text) {
        let p = para.trim();
        if p.is_empty() {
            continue;
        }
        if p.chars().count() <= size {
            units.push((p.to_string(), "\n\n"));
            continue;
        }
        let mut first = true;
        for sentence in split_sentences(p) {
            let mut rest: Vec<char> = sentence.trim().chars().collect();
            while rest.len() > size {
                // lastIndexOf(' ', size): a space at an index up to AND including `size`
                let upto = size.min(rest.len() - 1);
                let found = rest[..=upto].iter().rposition(|&c| c == ' ');
                // a space in the first half of the window (or none at all) is no place to cut
                let cut = match found {
                    Some(idx) if (idx as f64) >= (size as f64) * 0.5 => idx,
                    _ => size.min(rest.len()),
                };
                let head: String = rest[..cut].iter().collect();
                units.push((head.trim().to_string(), if first { "\n\n" } else { " " }));
                first = false;
                let tail: String = rest[cut..].iter().collect();
                rest = tail.trim().chars().collect();
            }
            if !rest.is_empty() {
                units.push((rest.iter().collect::<String>(), if first { "\n\n" } else { " " }));
                first = false;
            }
        }
    }
    let mut chunks: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut cur_len = 0usize;
    for (t, sep) in units {
        let t_len = t.chars().count();
        if !cur.is_empty() && cur_len + sep.len() + t_len > size {
            chunks.push(cur.trim().to_string());
            // the new passage begins with the tail of the old one, cut at the next word
            let mut tail: String = if overlap > 0 && cur_len > overlap {
                cur.chars().skip(cur_len - overlap).collect()
            } else if overlap > 0 {
                cur.clone()
            } else {
                String::new()
            };
            if let Some(space) = tail.find(' ') {
                if space + 1 < tail.chars().count() {
                    tail = tail[space + 1..].to_string();
                }
            }
            cur = if tail.is_empty() { t.clone() } else { format!("{tail} {t}") };
            cur_len = cur.chars().count();
        } else {
            if !cur.is_empty() {
                cur.push_str(sep);
            }
            cur.push_str(&t);
            cur_len = cur.chars().count();
        }
    }
    if !cur.trim().is_empty() {
        chunks.push(cur.trim().to_string());
    }
    chunks
}

/// The paragraphs of `p`, cut after a sentence end that is followed by whitespace, or at a line break
/// (the split of rag.js: `/(?<=[.!?])\s+|\n/`). `&[str]` may end with an empty sentence; the caller
/// skips it like the `if (rest)` of rag.js.
fn split_sentences(p: &str) -> Vec<&str> {
    let bytes = p.as_bytes();
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < p.len() {
        let c = bytes[i];
        if c == b'\n' {
            out.push(&p[start..i]);
            i += 1;
            start = i;
        } else if c == b'.' || c == b'!' || c == b'?' {
            // the separator `\\s+` after the punctuation is consumed with it
            let mut j = i + 1;
            while j < p.len() {
                let ch = p[j..].chars().next().expect("char at a boundary");
                if ch.is_whitespace() {
                    j += ch.len_utf8();
                } else {
                    break;
                }
            }
            if j > i + 1 {
                out.push(&p[start..i + 1]);
                i = j;
                start = j;
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    out.push(&p[start..]);
    out
}

/// A passage of a document as it travels with a message and is shown under it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Passage {
    #[serde(default)]
    pub doc_id: String,
    pub doc: String,
    /// Which passage of the document (1-based) and how many the document has.
    pub n: usize,
    pub m: usize,
    pub text: String,
    /// The rank score; not shown, not needed after the passages are chosen.
    #[serde(default)]
    pub score: f64,
}

/// One passage of the index with its term counts.
struct Chunk {
    doc_id: String,
    doc: String,
    n: usize,
    m: usize,
    text: String,
    tf: HashMap<String, usize>,
    len: usize,
}

/// The passages of every document with the document frequencies of BM25 (the `index` of rag.js).
pub struct Index {
    chunks: Vec<Chunk>,
    df: HashMap<String, usize>,
    avg: f64,
    /// Vectors of the passages when an embeddings server answered (filled later, once per index).
    pub vecs: Option<Vec<Vec<f64>>>,
    /// Why no vectors could be fetched (the message the Studio keeps).
    pub vecs_failed: Option<String>,
}

/// The documents an index is built from: their ids, names and texts.
pub struct IndexedDoc<'a> {
    pub id: &'a str,
    pub name: &'a str,
    pub text: &'a str,
}

pub fn build_index(docs: &[IndexedDoc<'_>]) -> Index {
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut df: HashMap<String, usize> = HashMap::new();
    for doc in docs {
        let parts = chunk(doc.text);
        let m = parts.len();
        for (i, text) in parts.iter().enumerate() {
            let terms = tokenize(text);
            let mut tf: HashMap<String, usize> = HashMap::new();
            for w in &terms {
                *tf.entry(w.clone()).or_insert(0) += 1;
            }
            for w in tf.keys() {
                *df.entry(w.clone()).or_insert(0) += 1;
            }
            let len = terms.len().max(1);
            chunks.push(Chunk {
                doc_id: doc.id.to_string(),
                doc: doc.name.to_string(),
                n: i + 1,
                m,
                text: text.clone(),
                tf,
                len,
            });
        }
    }
    let total: usize = chunks.iter().map(|c| c.len).sum();
    let avg = if chunks.is_empty() { 0.0 } else { total as f64 / chunks.len() as f64 };
    Index { chunks, df, avg, vecs: None, vecs_failed: None }
}

/// The `k` best passages for the query, best first (only passages that share a word with it). With
/// vectors in the index the ranking is hybrid: 60 % keyword scaled to the best hit, 40 % meaning.
pub fn search(index: &Index, query: &str, k: usize, query_vec: Option<&[f64]>) -> Vec<Passage> {
    // the Set of rag.js: a term counts once, the first occurrence keeps its place
    let tokens = tokenize(query);
    let mut seen: HashSet<&str> = HashSet::new();
    let terms: Vec<String> = tokens.iter().filter(|t| seen.insert(t.as_str())).cloned().collect();
    let vec = match query_vec {
        Some(v) if index.vecs.as_ref().is_some_and(|vs| vs.len() == index.chunks.len()) => Some(v),
        _ => None,
    };
    if (terms.is_empty() && vec.is_none()) || index.chunks.is_empty() {
        return Vec::new();
    }
    let n = index.chunks.len() as f64;
    let k1 = 1.5;
    let b = 0.75;
    let mut rows: Vec<(usize, f64)> = Vec::with_capacity(index.chunks.len());
    let mut best = 0.0f64;
    for (i, c) in index.chunks.iter().enumerate() {
        let mut bm = 0.0f64;
        for w in &terms {
            let Some(&f) = c.tf.get(w) else { continue };
            let d = *index.df.get(w).unwrap_or(&0) as f64;
            let idf = (1.0 + (n - d + 0.5) / (d + 0.5)).ln();
            bm += idf * (f as f64 * (k1 + 1.0)) / (f as f64 + k1 * (1.0 - b + b * c.len as f64 / index.avg));
        }
        if bm > best {
            best = bm;
        }
        rows.push((i, bm));
    }
    let mut scored: Vec<(usize, f64)> = Vec::new();
    for (i, bm) in rows {
        let Some(vec) = vec else {
            if bm > 0.0 {
                scored.push((i, bm));
            }
            continue;
        };
        let cos = cosine(vec, &index.vecs.as_ref().expect("checked above")[i]);
        // a passage without a shared word has to be clearly close in meaning
        if bm <= 0.0 && cos < VEC_MIN {
            continue;
        }
        let keyword = if best > 0.0 { bm / best } else { 0.0 };
        scored.push((i, 0.6 * keyword + 0.4 * cos.max(0.0)));
    }
    // stable, descending: passages with the same score keep their order, like the sort of rag.js
    scored.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
        .into_iter()
        .take(k)
        .map(|(i, score)| {
            let c = &index.chunks[i];
            Passage { doc_id: c.doc_id.clone(), doc: c.doc.clone(), n: c.n, m: c.m, text: c.text.clone(), score }
        })
        .collect()
}

/// Cosine similarity of two vectors (0 when one of them is empty or all zeros).
pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb).sqrt()
}

/// The first passages of every document: used when the question names nothing to search for
/// ("summarize this").
pub fn opening(index: &Index, k: usize) -> Vec<Passage> {
    let mut doc_ids: Vec<&str> = Vec::new();
    for c in &index.chunks {
        if !doc_ids.contains(&c.doc_id.as_str()) {
            doc_ids.push(&c.doc_id);
        }
    }
    let per = k.max(1).div_ceil(doc_ids.len().max(1));
    let mut out: Vec<Passage> = Vec::new();
    for id in doc_ids {
        for c in index.chunks.iter().filter(|x| x.doc_id == id).take(per) {
            out.push(Passage { doc_id: c.doc_id.clone(), doc: c.doc.clone(), n: c.n, m: c.m, text: c.text.clone(), score: 0.0 });
        }
        if out.len() >= k {
            break;
        }
    }
    out.truncate(k);
    out
}

/// How many passages fit: about 40 % of the context, between 2 and 8.
pub fn passage_count(max_context: u64) -> usize {
    let per_passage = CHUNK_CHARS as f64 / 3.6;
    (((max_context as f64) * 0.4 / per_passage) as usize).clamp(2, 8)
}

/// The text put in front of the question: passages with their source labels (the `ragPrefix` of
/// rag.js, without the web results, which the model's own tools do now).
pub fn prefix(passages: &[Passage]) -> String {
    if passages.is_empty() {
        return String::new();
    }
    let mut out = String::from("Passages from the attached documents. Use them to answer, and name the source in brackets, for example [report.pdf, chunk 3/42], when you rely on one.\n\n");
    out.push_str(
        &passages
            .iter()
            .map(|p| format!("[{}, chunk {}/{}]\n{}", p.doc, p.n, p.m, p.text))
            .collect::<Vec<_>>()
            .join("\n\n"),
    );
    out.push_str("\n\n");
    out.push_str("---\n");
    out
}

/// The chosen passages in the order the Studio sends them: passages of one document in their order,
/// everything else untouched (the stable sort of `a.docId === b.docId ? a.n - b.n : 0`).
pub fn arrange(mut found: Vec<Passage>) -> Vec<Passage> {
    found.sort_by(|a, b| if a.doc_id == b.doc_id { a.n.cmp(&b.n) } else { std::cmp::Ordering::Equal });
    found
}

/// One line of a passage as it is shown under a message (the Studio's `ragSnippet`).
pub fn snippet(text: &str) -> String {
    let t = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if t.chars().count() > 700 {
        let cut: String = t.chars().take(700).collect();
        format!("{cut}...")
    } else {
        t
    }
}

/// The documents of a list by name: the first two and a count (the Studio's `ragNames`).
pub fn names(list: &[String]) -> String {
    let mut unique: Vec<&String> = Vec::new();
    for n in list {
        if !unique.contains(&n) {
            unique.push(n);
        }
    }
    if unique.len() > 2 {
        format!("{}, {} and {} more", unique[0], unique[1], unique.len() - 2)
    } else {
        unique.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" and ")
    }
}

/// What the fit of a document decided (the `prjFitDoc` of projects.js).
pub enum Fit {
    Whole(String),
    Cut { text: String, note: String },
    Message(String),
}

impl Fit {
    /// The text to store, with the line to show when it was cut; `None` when it was refused.
    pub fn stored(self) -> Option<(String, Option<String>)> {
        match self {
            Fit::Whole(t) => Some((t, None)),
            Fit::Cut { text, note } => Some((text, Some(note))),
            Fit::Message(_) => None,
        }
    }

    /// The line to show when the document was refused.
    pub fn error(self) -> Option<String> {
        match self {
            Fit::Message(m) => Some(m),
            _ => None,
        }
    }
}

/// Can this document go into the project knowledge? Cuts it to the room left, or refuses with the
/// reason. `used_project` and `used_total` are the characters already stored.
pub fn fit_doc(used_project: usize, used_total: usize, name: &str, text: &str) -> Fit {
    if text.trim().is_empty() {
        return Fit::Message(format!("{name} has no readable text."));
    }
    let project_room = limits::PROJECT_USAGES.saturating_sub(used_project);
    let total_room = limits::TOTAL_USAGES.saturating_sub(used_total);
    if project_room < limits::MIN_ROOM {
        return Fit::Message(format!(
            "{name} was not added: the knowledge of this project is full ({} characters). Remove a document first.",
            group_sep(limits::PROJECT_USAGES)
        ));
    }
    if total_room < limits::MIN_ROOM {
        return Fit::Message(format!(
            "{name} was not added: the browser storage kept for projects is full ({} characters in all projects). Remove a document first.",
            group_sep(limits::TOTAL_USAGES)
        ));
    }
    let room = limits::MAX_DOC.min(project_room).min(total_room);
    if text.chars().count() > room {
        let cut: String = text.chars().take(room).collect();
        return Fit::Cut { text: cut, note: format!("{name} is long: only its first {} characters are kept in the project.", group_sep(room)) };
    }
    Fit::Whole(text.to_string())
}

/// `toLocaleString('en-US')`: 2000000 -> "2,000,000".
fn group_sep(n: usize) -> String {
    let digits = n.to_string();
    let mut parts: Vec<&str> = digits.as_bytes().rchunks(3).map(|c| std::str::from_utf8(c).unwrap_or("")).collect();
    parts.reverse();
    parts.join(",")
}

/// The limits of `PRJ_LIMITS` in the Studio's projects.js.
pub mod limits {
    /// The characters of one document of a project knowledge.
    pub const MAX_DOC: usize = 600_000;
    /// The characters of the knowledge of one project.
    pub const PROJECT_USAGES: usize = 2_000_000;
    /// The characters of the knowledge of all projects together (the browser storage).
    pub const TOTAL_USAGES: usize = 3_500_000;
    /// A project is full below this room (the Studio stops accepting documents before it is exactly full).
    pub const MIN_ROOM: usize = 2_000;
    /// The instructions of a project.
    pub const INSTRUCTIONS: usize = 8_000;
    /// The name of a project.
    pub const NAME: usize = 60;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Deserialize)]
    struct Fixture {
        docs: Vec<Doc>,
        index: IndexInfo,
        cases: Vec<Case>,
    }

    #[derive(Deserialize)]
    struct Doc {
        id: String,
        name: String,
        text: String,
    }

    #[derive(Deserialize)]
    struct IndexInfo {
        n: usize,
        avg: f64,
        df: HashMap<String, usize>,
    }

    #[derive(Deserialize)]
    struct Case {
        kind: String,
        #[serde(default)]
        query: String,
        #[serde(default)]
        tokens: Vec<String>,
        #[serde(default)]
        id: String,
        #[serde(default)]
        chunks: Vec<String>,
        #[serde(default)]
        hits: Vec<Hit>,
        #[serde(default)]
        k: usize,
        #[serde(default)]
        a: Vec<f64>,
        #[serde(default)]
        b: Vec<f64>,
        #[serde(default)]
        v: f64,
        #[serde(default)]
        passages: Vec<Passage>,
        #[serde(default)]
        prefix: String,
    }

    #[derive(Deserialize)]
    struct Hit {
        #[serde(rename = "docId", default)]
        doc_id: Option<String>,
        #[serde(default)]
        n: usize,
        #[serde(default)]
        m: usize,
        #[serde(default)]
        score: Option<f64>,
        #[serde(default)]
        text: String,
    }

    fn fixture() -> Fixture {
        serde_json::from_str(include_str!("../tests/fixtures/rag_fixtures.json")).expect("fixture json")
    }

    fn index_of(f: &Fixture) -> Index {
        let docs: Vec<IndexedDoc<'_>> = f.docs.iter().map(|d| IndexedDoc { id: &d.id, name: &d.name, text: &d.text }).collect();
        build_index(&docs)
    }

    #[test]
    fn tokenizing_is_the_one_of_the_studio() {
        let f = fixture();
        let mut checked = 0;
        for c in &f.cases {
            if c.kind == "tokenize" {
                assert_eq!(tokenize(&c.query), c.tokens, "tokens of {:?}", c.query);
                checked += 1;
            }
        }
        assert_eq!(checked, 8, "the fixture holds eight token cases");
    }

    #[test]
    fn the_passages_of_every_document_are_the_ones_of_the_studio() {
        let f = fixture();
        let mut checked = 0;
        for c in &f.cases {
            if c.kind == "chunk" {
                let doc = f.docs.iter().find(|d| d.id == c.id).expect("fixture doc");
                assert_eq!(chunk(&doc.text), c.chunks, "chunks of {}", c.id);
                checked += 1;
            }
        }
        assert_eq!(checked, 3, "the fixture holds three documents");
    }

    #[test]
    fn the_index_counts_what_the_studio_counts() {
        let f = fixture();
        let index = index_of(&f);
        assert_eq!(index.chunks.len(), f.index.n);
        assert!((index.avg - f.index.avg).abs() < 1e-9, "avg {} vs {}", index.avg, f.index.avg);
        assert_eq!(index.df, f.index.df, "document frequencies");
    }

    #[test]
    fn every_search_finds_the_passages_in_the_order_of_the_studio() {
        let f = fixture();
        let index = index_of(&f);
        let mut checked = 0;
        for c in &f.cases {
            if c.kind != "search" {
                continue;
            }
            let got = search(&index, &c.query, 4, None);
            assert_eq!(got.len(), c.hits.len(), "hits of {:?}", c.query);
            for (g, want) in got.iter().zip(c.hits.iter()) {
                assert_eq!(Some(g.doc_id.as_str()), want.doc_id.as_deref(), "doc of {:?}", c.query);
                assert_eq!(g.n, want.n, "passage number of {:?}", c.query);
                assert_eq!(g.m, want.m, "passage count of {:?}", c.query);
                assert_eq!(g.text, want.text, "text of {:?}", c.query);
                let want_score = want.score.unwrap_or(0.0);
                assert!((g.score - want_score).abs() < 1e-9, "score of {:?}: {} vs {}", c.query, g.score, want_score);
            }
            checked += 1;
        }
        assert_eq!(checked, 8, "the fixture holds eight search cases");
    }

    #[test]
    fn the_opening_passages_and_the_cosine_match() {
        let f = fixture();
        let index = index_of(&f);
        for c in &f.cases {
            match c.kind.as_str() {
                "opening" => {
                    let got = opening(&index, c.k);
                    let want: Vec<(String, usize, usize)> = c.hits.iter().map(|h| (h.doc_id.clone().unwrap_or_default(), h.n, h.m)).collect();
                    let got: Vec<(String, usize, usize)> = got.iter().map(|p| (p.doc_id.clone(), p.n, p.m)).collect();
                    assert_eq!(got, want, "opening of k={}", c.k);
                }
                "cosine" => {
                    let v = cosine(&c.a, &c.b);
                    assert!((v - c.v).abs() < 1e-12, "cosine {:?} {:?}: {} vs {}", c.a, c.b, v, c.v);
                }
                _ => {}
            }
        }
    }

    #[test]
    fn the_prefix_of_the_prompt_is_the_one_of_the_studio() {
        let f = fixture();
        let c = f.cases.iter().find(|c| c.kind == "prefix").expect("prefix case");
        let found = search(&index_of(&f), "servers in the data center", 3, None);
        let got = prefix(&arrange(found));
        assert_eq!(got, c.prefix);
        // the stored passages alone give the same text
        assert_eq!(prefix(&c.passages), c.prefix);
        // without passages there is nothing in front of the question
        assert_eq!(prefix(&[]), "");
    }

    #[test]
    fn the_passage_count_stays_between_two_and_eight() {
        assert_eq!(passage_count(4096), 3);
        assert_eq!(passage_count(100), 2, "a small context still gets two");
        assert_eq!(passage_count(2048), 2, "floor(1.6384) is 1, the clamp lifts it to 2");
        assert_eq!(passage_count(1_000_000), 8, "a huge context gets eight");
    }

    #[test]
    fn folding_a_plural_never_cuts_a_double_s_and_keeps_digits() {
        assert_eq!(tokenize("glasses bottles"), vec!["glasse", "bottle"], "a trailing s folds, as in the Studio");
        assert_eq!(tokenize("glass class"), vec!["glass", "class"], "a word ending in ss is not folded");
        assert_eq!(tokenize("4"), vec!["4"], "a single digit is a word");
        assert_eq!(tokenize("a i"), Vec::<String>::new(), "single letters are stop words or too short");
        assert_eq!(tokenize("ÜBER Werte"), vec!["über", "werte"]);
        assert_eq!(tokenize("the of and"), Vec::<String>::new(), "stop words only");
    }

    #[test]
    fn the_room_of_a_project_document_is_cut_or_refused_like_in_the_studio() {
        let whole = fit_doc(0, 0, "a.txt", "x").stored().expect("fits");
        assert_eq!(whole.0, "x");
        assert_eq!(whole.1, None);
        // a text over the per-document room is cut, with the line the Studio shows
        let long = "y".repeat(limits::MAX_DOC + 5);
        let cut = fit_doc(0, 0, "big.txt", &long).stored().expect("cut");
        assert_eq!(cut.0.chars().count(), limits::MAX_DOC);
        assert_eq!(cut.1.as_deref(), Some("big.txt is long: only its first 600,000 characters are kept in the project."));
        // the project is full
        let err = fit_doc(limits::PROJECT_USAGES - 10, 0, "c.txt", "text").error();
        assert_eq!(err.as_deref(), Some("c.txt was not added: the knowledge of this project is full (2,000,000 characters). Remove a document first."));
        // every project together is full
        let err = fit_doc(0, limits::TOTAL_USAGES - 10, "d.txt", "text").error();
        assert_eq!(
            err.as_deref(),
            Some("d.txt was not added: the browser storage kept for projects is full (3,500,000 characters in all projects). Remove a document first.")
        );
        // nothing readable
        let err = fit_doc(0, 0, "e.txt", " \n ").error();
        assert_eq!(err.as_deref(), Some("e.txt has no readable text."));
    }

    #[test]
    fn snippet_and_names_are_shown_like_in_the_studio() {
        assert_eq!(snippet("  a   b \n c "), "a b c");
        let long: String = "w".repeat(800);
        let s = snippet(&long);
        assert_eq!(s.chars().count(), 703, "700 characters and three dots");
        assert_eq!(names(&["a.pdf".into(), "b.md".into(), "c.txt".into()]), "a.pdf, b.md and 1 more");
        assert_eq!(names(&["a.pdf".into(), "b.md".into()]), "a.pdf and b.md");
        assert_eq!(names(&["a.pdf".into(), "a.pdf".into()]), "a.pdf", "a name counts once");
        assert_eq!(names(&[]), "");
    }
}
