//! The command palette of the Studio (`palette.js`, the `@pure-begin`..`@pure-end` block for the
//! ranking, the rest for the commands): what scores, in which order the rows come and which row an
//! arrow key takes. The drawing lives in `app.rs` (`palette_window`), the commands in `app.rs`
//! (`palette_commands`), the chat rows come from the store the way the sidebar searches it.

/// The kind of a row — the groups the palette draws (`labelOf` of `palette.js`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Command,
    Chat,
    CodeSession,
    Setting,
}

impl Kind {
    /// The name of the group over the block (`labelOf` of `palette.js`).
    pub fn label(self) -> &'static str {
        match self {
            Kind::Command => "Commands",
            Kind::Chat => "Chats",
            Kind::CodeSession => "Code sessions",
            Kind::Setting => "Settings",
        }
    }
}

/// One row of the palette (`{ id, kind, title, sub?, keywords?, hint?, weak?, run }` of
/// `palette.js`; the run lives on the app side as a [`crate::app::PaletteAction`]).
#[derive(Debug, Clone, PartialEq)]
pub struct PaletteItem {
    pub id: String,
    pub kind: Kind,
    pub title: String,
    /// A line under the title: the text around a hit in a chat, or a settings hint.
    pub sub: String,
    /// Words that also find the row, scored under the title.
    pub keywords: String,
    /// The keys of the shortcut on the right (`kbd` of the row).
    pub hint: String,
    /// A row that is only shown for a query and always last in its block.
    pub weak: bool,
}

impl PaletteItem {
    pub fn new(id: impl Into<String>, kind: Kind, title: impl Into<String>) -> Self {
        PaletteItem {
            id: id.into(),
            kind,
            title: title.into(),
            sub: String::new(),
            keywords: String::new(),
            hint: String::new(),
            weak: false,
        }
    }

    pub fn keywords(mut self, words: &str) -> Self {
        self.keywords = words.to_string();
        self
    }

    pub fn hint(mut self, keys: impl Into<String>) -> Self {
        self.hint = keys.into();
        self
    }

    pub fn weak(mut self) -> Self {
        self.weak = true;
        self
    }

    pub fn with_sub(mut self, text: impl Into<String>) -> Self {
        self.sub = text.into();
        self
    }
}

/// Scores one text for a query: 0 = no match. Word starts count more than a hit inside a word; a
/// spread-out match counts least (`paletteScore` of `palette.js`).
pub fn palette_score(text: &str, query: &str) -> i64 {
    let t = text.to_lowercase();
    let q = query.to_lowercase();
    let q = q.trim();
    if q.is_empty() {
        return 1;
    }
    if t == q {
        return 120;
    }
    if t.starts_with(q) {
        return 100;
    }
    let words: Vec<&str> = q.split_whitespace().filter(|w| !w.is_empty()).collect();
    if words.is_empty() {
        return 0;
    }
    let t_chars: Vec<char> = t.chars().collect();
    let mut total: i64 = 0;
    for w in &words {
        let w_chars: Vec<char> = w.chars().collect();
        let mut s: i64 = 0;
        if t.starts_with(w) {
            s = 90;
        } else if word_boundary_hit(&t_chars, &w_chars).is_some() {
            s = 70;
        } else if t.contains(w) {
            s = 40;
        } else if w_chars.len() >= 3 {
            // letters in order, starting at the start of a word and close together ("kbd" finds
            // "keyboard"); never scattered over the text
            let mut st = 0usize;
            while st < t_chars.len() && s == 0 {
                let ok_start = t_chars[st] == w_chars[0]
                    && (st == 0 || !t_chars[st - 1].is_alphanumeric());
                if !ok_start {
                    st += 1;
                    continue;
                }
                let mut i = 1usize;
                let mut at = st;
                while i < w_chars.len() && at + 1 < t_chars.len() {
                    at += 1;
                    if t_chars[at] == w_chars[i] {
                        i += 1;
                    }
                }
                if i == w_chars.len() && at - st < w_chars.len() * 3 {
                    s = 12;
                }
                st += 1;
            }
        }
        if s == 0 {
            return 0;
        }
        total += s;
    }
    (total as f64 / words.len() as f64).round() as i64
}

/// The first place where `word` sits at a word boundary of `t`: the text before it is nothing or
/// no letter or digit (`(^|[^\p{L}\p{N}])word` of boot.js). Nothing = no such place.
fn word_boundary_hit(t: &[char], word: &[char]) -> Option<usize> {
    if word.is_empty() || t.len() < word.len() {
        return None;
    }
    for start in 0..=t.len() - word.len() {
        if start > 0 && t[start - 1].is_alphanumeric() {
            continue;
        }
        if t[start..start + word.len()] == *word {
            return Some(start);
        }
    }
    None
}

/// Orders palette rows for a query. Without a query the commands come first (the first few), then
/// the chats in the order given. With a query every row whose title, keywords or sub text matches
/// is kept, best first; a title match beats a keyword match, which beats a text match — the result
/// is the indexes of `items` in their new order (`rankPalette` of `palette.js`, `limit` 40 there).
pub fn rank_palette(query: &str, items: &[PaletteItem], limit: usize) -> Vec<usize> {
    let q = query.trim();
    let cap = if limit == 0 { 40 } else { limit };
    if q.is_empty() {
        let mut out: Vec<usize> = items
            .iter()
            .enumerate()
            .filter(|(_, i)| i.kind == Kind::Command)
            .map(|(n, _)| n)
            .take(7)
            .collect();
        out.extend(
            items
                .iter()
                .enumerate()
                .filter(|(_, i)| i.kind == Kind::Chat)
                .map(|(n, _)| n)
                .take(6),
        );
        out.truncate(cap);
        return out;
    }

    // every row that matches something, with its score and its place in the list
    let mut scored: Vec<(usize, i64, usize)> = Vec::new();
    for (order, item) in items.iter().enumerate() {
        let title = palette_score(&item.title, q);
        let kw = if item.keywords.is_empty() {
            0
        } else {
            palette_score(&item.keywords, q)
        };
        let sub = if item.sub.is_empty() {
            0
        } else {
            palette_score(&item.sub, q)
        };
        let score = if item.weak {
            1
        } else {
            (title * 3).max(kw * 2).max(sub)
        };
        if score > 0 {
            scored.push((order, score, order));
        }
    }
    scored.sort_by(|a, b| b.1.cmp(&a.1).then(a.2.cmp(&b.2)));

    // one block per kind; the block with the best match comes first, a weak entry stays last in
    // its block
    let mut best: Vec<(Kind, i64)> = Vec::new();
    for (order, score, _) in &scored {
        if items[*order].weak {
            continue;
        }
        let kind = items[*order].kind;
        if !best.iter().any(|(k, _)| *k == kind) {
            best.push((kind, *score));
        }
    }
    let rank = |kind: Kind| -> i64 {
        best.iter()
            .find(|(k, _)| *k == kind)
            .map(|(_, s)| -*s)
            .unwrap_or(0)
    };
    // the place in the best-first list is what keeps the order inside one block
    let mut placed: Vec<(usize, usize)> = scored
        .iter()
        .enumerate()
        .map(|(i, (order, ..))| (*order, i))
        .collect();
    placed.sort_by(|a, b| {
        let ka = items[a.0].kind;
        let kb = items[b.0].kind;
        rank(ka).cmp(&rank(kb)).then(if ka == kb {
            a.1.cmp(&b.1)
        } else {
            ka.label().cmp(kb.label())
        })
    });
    placed.truncate(cap);
    placed.into_iter().map(|(order, _)| order).collect()
}

/// The next active row for an arrow key: wraps around at both ends (`paletteStep` of
/// `palette.js`; `-1` is no row yet and `Home`/`End` jump).
pub fn palette_step(index: i32, count: usize, key: &str) -> i32 {
    if count == 0 {
        return -1;
    }
    let count = count as i32;
    match key {
        "Home" => 0,
        "End" => count - 1,
        "ArrowDown" => {
            if index < 0 {
                0
            } else {
                (index + 1) % count
            }
        }
        "ArrowUp" => {
            if index <= 0 {
                count - 1
            } else {
                index - 1
            }
        }
        _ => index,
    }
}

/// The text around a hit (`snippet` of `boot.js`): thirty characters before, the hit, fifty after,
/// the spaces folded, an ellipsis where the text continues.
pub fn snippet(text: &str, q: &str) -> String {
    let lower = text.to_lowercase();
    let Some(byte) = lower.find(&q.to_lowercase()) else {
        return String::new();
    };
    let char_index = text[..byte].chars().count();
    let q_len = text[byte..].chars().count().min(q.chars().count());
    let chars: Vec<char> = text.chars().collect();
    let from = char_index.saturating_sub(30);
    let to = (char_index + q_len + 50).min(chars.len());
    let body: String = chars[from..to]
        .iter()
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    format!("{}{}…", if from > 0 { "…" } else { "" }, body)
}
