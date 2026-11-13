//! The memory of the user: short facts that go into the prompt of every chat, a *remember that …*
//! at the start of a message, and the `memory` tool of the model. The rules are the ones of the
//! Studio's `projects.js` (`memExtract` to `memSelect`): a fact is never a secret, the list holds a
//! hundred entries and the prompt gets the most useful ones within its character cap.

use serde::{Deserialize, Serialize};

/// The limits of the Studio's `PRJ_LIMITS` that belong to the memory.
pub mod limits {
    /// One entry: the characters the Studio counts and cuts at (`memory`).
    pub const MEMORY: usize = 400;
    /// How many entries the list holds (`memories`).
    pub const MEMORIES: usize = 100;
    /// The category of an entry (`category`).
    pub const CATEGORY: usize = 24;
    /// What *About me* may hold (`about`).
    pub const ABOUT: usize = 2_000;
    /// The characters of the memory block in the prompt (`memoryPrompt`).
    pub const PROMPT: usize = 3_000;
    /// The same for a small model (`memoryPromptCompact`).
    pub const PROMPT_COMPACT: usize = 1_200;
}

/// One remembered fact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Memory {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub text: String,
    /// When it was remembered (milliseconds since 1970, like the Studio's `Date.now()`).
    #[serde(default)]
    pub created_at: i64,
    #[serde(default)]
    pub updated_at: i64,
    /// How often the model came back to it; two and more are sent first.
    #[serde(default)]
    pub uses: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
}

/// A change of the list by the tool, so it can be taken back (the Studio's `change` of memToolApply).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Change {
    pub kind: String,
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before: Option<Memory>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after: Option<Memory>,
}

/// One line under an answer: what the memory tool did, with the undo of a change (the Studio's
/// tool card of a node).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCard {
    #[serde(default)]
    pub label: String,
    /// `done` or `error`.
    #[serde(default)]
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub change: Option<Change>,
    #[serde(default)]
    pub undone: bool,
    #[serde(default)]
    pub quiet: bool,
}

/// What the tool answered, and what it changed.
pub struct ToolOutcome {
    pub list: Vec<Memory>,
    /// What the model is told.
    pub text: String,
    /// What happened, for the card with the undo.
    pub change: Option<Change>,
}

/// `"remember that …"` (or `"remember: …"`) at the start of a message gives the thing to remember,
/// else `None`. The first paragraph only, whitespace collapsed, cut to the limit (memExtract).
pub fn extract(text: &str) -> Option<String> {
    let re = regex::Regex::new(r"(?i)^\s*(?:please\s+)?remember\s*(?:that\b|:)\s*([\s\S]+)$").expect("the pattern compiles");
    let caps = re.captures(text)?;
    let paragraph = regex::Regex::new(r"\n\s*\n").expect("the pattern compiles");
    let first = paragraph.split(&caps[1]).next().unwrap_or("");
    let flat = first.split_whitespace().collect::<Vec<_>>().join(" ");
    let cut: String = flat.chars().take(limits::MEMORY).collect();
    (!cut.is_empty()).then_some(cut)
}

/// Does a text look like a password, an access key, a card number or similar? Such things are never
/// saved to memory (memSecret).
pub fn secret(text: &str) -> bool {
    let re = regex::Regex::new;
    let key_value = re(r"(?i)\b(?:pass(?:word|wd|phrase|code)?|pin|secret|api[ _-]?key|access[ _-]?key|private[ _-]?key|token|credentials?|cvv|cvc)\b\s*(?:is|are|:|=)\s*(?:\S*\d\S*|\S{6,})")
        .expect("the pattern compiles");
    if key_value.is_match(text) {
        return true;
    }
    let pem = re(r"-----BEGIN [A-Z ]*PRIVATE KEY-----").expect("the pattern compiles");
    if pem.is_match(text) {
        return true;
    }
    let tokens = re(r"\b(?:sk|pk|rk|ghp|gho|ghs|xox[abp]|AIza|hf)[-_][A-Za-z0-9_-]{16,}").expect("the pattern compiles");
    if tokens.is_match(text) {
        return true;
    }
    let aws = re(r"\bAKIA[0-9A-Z]{16}\b").expect("the pattern compiles");
    if aws.is_match(text) {
        return true;
    }
    // the JS pattern ends with a look-ahead the Rust crate does not have: the token is matched
    // first, the boundary behind it is checked by hand
    let long_token = re(r"(?:^|\s)[A-Za-z0-9+/_-]{32,}={0,2}").expect("the pattern compiles");
    for m in long_token.find_iter(text) {
        let boundary = match text[m.end()..].chars().next() {
            None => true,
            Some(c) => c.is_whitespace() || matches!(c, '.' | ',' | ';'),
        };
        if boundary {
            return true;
        }
    }
    // a card number: the digits pass the Luhn checksum
    let digits_run = re(r"\b(?:\d[ -]?){13,19}\b").expect("the pattern compiles");
    for m in digits_run.find_iter(text) {
        let digits: String = m.as_str().chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.len() < 13 || digits.len() > 19 {
            continue;
        }
        let mut sum = 0;
        // from the right, like the Studio: every second digit from there is doubled
        for (i, c) in digits.chars().rev().enumerate() {
            let mut d = c.to_digit(10).unwrap_or(0);
            if i % 2 == 1 {
                d *= 2;
                if d > 9 {
                    d -= 9;
                }
            }
            sum += d;
        }
        if sum % 10 == 0 {
            return true;
        }
    }
    let iban = re(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,4})?\b").expect("the pattern compiles");
    iban.is_match(text)
}

/// The category as the Studio keeps it: lowercase, letters digits space dash underscore, cut (memCategory).
pub fn category(c: Option<&str>) -> Option<String> {
    let raw = c.unwrap_or("").to_lowercase();
    let cleaned: String = raw.chars().filter(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || matches!(ch, ' ' | '_' | '-')).collect();
    let cleaned = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
    let cleaned: String = cleaned.chars().take(limits::CATEGORY).collect();
    (!cleaned.is_empty()).then_some(cleaned)
}

/// The text as an entry holds it: whitespace collapsed, cut (memNorm).
pub fn norm(text: &str) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    flat.chars().take(limits::MEMORY).collect()
}

/// Clean entries from untrusted data (the store, the server copy). Older entries only had id, text
/// and created (memSanitize).
pub fn sanitize(raw: &[serde_json::Value], make_id: impl Fn() -> String) -> Vec<Memory> {
    raw.iter()
        .filter(|m| m.get("text").and_then(serde_json::Value::as_str).is_some_and(|t| !t.trim().is_empty()))
        .take(limits::MEMORIES)
        .map(|m| {
            let text: String = m.get("text").and_then(serde_json::Value::as_str).unwrap_or_default().chars().take(limits::MEMORY).collect();
            let created_at = m.get("createdAt").and_then(serde_json::Value::as_i64).unwrap_or(0);
            let id: String = m
                .get("id")
                .and_then(serde_json::Value::as_str)
                .map(|s| s.chars().take(64).collect())
                .filter(|s: &String| !s.is_empty())
                .unwrap_or_else(&make_id);
            let updated_at = m.get("updatedAt").and_then(serde_json::Value::as_i64).unwrap_or(created_at);
            let uses = m.get("uses").and_then(serde_json::Value::as_f64).unwrap_or(0.0).round().clamp(0.0, 1000.0) as u32;
            let cat = category(m.get("category").and_then(serde_json::Value::as_str));
            Memory { id, text, created_at, updated_at, uses, category: cat }
        })
        .collect()
}

/// Adds a memory unless it is empty, a secret, a duplicate or the list is full (memAdd). The status
/// says which it is, in the words the Studio shows the user.
pub fn add(list: &[Memory], text: &str, now: i64, make_id: impl Fn() -> String, cat: Option<&str>) -> (Vec<Memory>, &'static str, Option<Memory>) {
    let t = norm(text);
    if t.is_empty() {
        return (list.to_vec(), "empty", None);
    }
    if secret(&t) {
        return (list.to_vec(), "secret", None);
    }
    if list.iter().any(|m| m.text.to_lowercase() == t.to_lowercase()) {
        return (list.to_vec(), "duplicate", None);
    }
    if list.len() >= limits::MEMORIES {
        return (list.to_vec(), "full", None);
    }
    let entry = Memory { id: make_id(), text: t, created_at: now, updated_at: now, uses: 0, category: category(cat) };
    let mut next = list.to_vec();
    next.push(entry.clone());
    (next, "added", Some(entry))
}

/// Entries the model may name: by id, by position in the list (1 = first) or by a part of the text
/// (memFind).
pub fn find<'a>(list: &'a [Memory], id: &str, part: &str) -> Vec<&'a Memory> {
    let id = id.trim();
    if !id.is_empty() {
        let by_id: Vec<&Memory> = list.iter().filter(|m| m.id == id).collect();
        if !by_id.is_empty() {
            return by_id;
        }
        if let Ok(n) = id.parse::<usize>() {
            if n >= 1 && n <= list.len() {
                return vec![&list[n - 1]];
            }
        }
        return Vec::new();
    }
    let part = part.to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ");
    if part.is_empty() {
        return Vec::new();
    }
    list.iter().filter(|m| m.text.to_lowercase().contains(&part)).collect()
}

/// The `memory` tool of the chat (memToolApply). `args`: `{ action: add | update | remove | list,
/// text, category, id, match }`. The text is what the model is told; the change is for the card.
pub fn tool_apply(list: &[Memory], args: &serde_json::Value, now: i64, make_id: impl Fn() -> String) -> ToolOutcome {
    let object = args.as_object();
    let str_arg = |key: &str| object.and_then(|o| o.get(key)).and_then(serde_json::Value::as_str).unwrap_or_default().to_string();
    let action = str_arg("action").to_lowercase();
    let out = |text: String, next: Vec<Memory>, change: Option<Change>| ToolOutcome { list: next, text, change };
    if action == "list" {
        let text = if list.is_empty() {
            "The memory is empty.".to_string()
        } else {
            list.iter()
                .enumerate()
                .map(|(i, m)| {
                    format!(
                        "{}. [{}]{} {}",
                        i + 1,
                        m.id,
                        m.category.as_ref().map(|c| format!(" ({c})")).unwrap_or_default(),
                        m.text
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        };
        return out(text, list.to_vec(), None);
    }
    if action == "add" {
        let (next, status, entry) = add(list, &str_arg("text"), now, &make_id, Some(&str_arg("category")));
        return match status {
            "empty" => out("Error: \"text\" is required for add.".into(), list.to_vec(), None),
            "secret" => out("Not saved: this looks like a password, key or card number, which must never be stored.".into(), list.to_vec(), None),
            "full" => out(format!("Error: the memory is full ({} entries). Remove an entry first.", limits::MEMORIES), list.to_vec(), None),
            "duplicate" => {
                let wanted = norm(&str_arg("text")).to_lowercase();
                let next = list
                    .iter()
                    .map(|m| {
                        let mut m = m.clone();
                        if m.text.to_lowercase() == wanted {
                            m.uses += 1;
                        }
                        m
                    })
                    .collect();
                out("Already remembered.".into(), next, None)
            }
            _ => {
                let entry = entry.expect("added");
                out("Saved.".into(), next, Some(Change { kind: "add".into(), id: entry.id.clone(), before: None, after: Some(entry) }))
            }
        };
    }
    if action == "update" || action == "remove" {
        let found = find(list, &str_arg("id"), &str_arg("match"));
        if found.is_empty() {
            return out(
                "Error: no matching entry. Use action \"list\" to see the entries, then pass \"id\" or part of the old text as \"match\".".into(),
                list.to_vec(),
                None,
            );
        }
        if found.len() > 1 {
            let matches: String = found
                .iter()
                .take(5)
                .map(|m| format!("[{}] {}", m.id, m.text))
                .collect::<Vec<_>>()
                .join("\n");
            return out(format!("Error: {} entries match; be more specific or pass \"id\". Matches:\n{}", found.len(), matches), list.to_vec(), None);
        }
        let before = found[0].clone();
        if action == "remove" {
            let next = list.iter().filter(|m| m.id != before.id).cloned().collect();
            return out(
                "Removed.".into(),
                next,
                Some(Change { kind: "remove".into(), id: before.id.clone(), before: Some(before), after: None }),
            );
        }
        let text = norm(&str_arg("text"));
        if text.is_empty() {
            return out("Error: \"text\" is required for update.".into(), list.to_vec(), None);
        }
        if secret(&text) {
            return out("Not saved: this looks like a password, key or card number, which must never be stored.".into(), list.to_vec(), None);
        }
        let mut after = before.clone();
        after.text = text;
        after.updated_at = now;
        after.uses += 1;
        if object.is_some_and(|o| o.contains_key("category")) {
            after.category = category(Some(&str_arg("category")));
        }
        let next = list.iter().map(|m| if m.id == before.id { after.clone() } else { m.clone() }).collect();
        return out(
            "Updated.".into(),
            next,
            Some(Change { kind: "update".into(), id: before.id.clone(), before: Some(before), after: Some(after) }),
        );
    }
    out("Error: \"action\" must be add, update, remove or list.".into(), list.to_vec(), None)
}

/// Takes back one change of `tool_apply` (the card's undo). Returns the new list, or `None` when it
/// no longer applies (memUndo).
pub fn undo(list: &[Memory], change: &Change) -> Option<Vec<Memory>> {
    match change.kind.as_str() {
        "add" => list.iter().any(|m| m.id == change.id).then(|| list.iter().filter(|m| m.id != change.id).cloned().collect()),
        "update" => list
            .iter()
            .any(|m| m.id == change.id)
            .then(|| list.iter().map(|m| if m.id == change.id { change.before.clone().expect("an update has a before") } else { m.clone() }).collect()),
        "remove" => {
            if list.iter().any(|m| m.id == change.id) || list.len() >= limits::MEMORIES {
                None
            } else {
                let mut next = list.to_vec();
                next.push(change.before.clone().expect("a remove has a before"));
                Some(next)
            }
        }
        _ => None,
    }
}

/// The memories that go into the prompt, within `cap` characters. Priority: entries that were come
/// back to (uses >= 2) first, then the rest, each group newest first; the chosen ones keep their
/// original order (memSelect). Returns the lines and how many were left out.
pub fn select(memories: &[Memory], cap: usize) -> (Vec<String>, usize) {
    let mut ranked: Vec<(usize, &Memory)> = memories.iter().enumerate().collect();
    ranked.sort_by(|a, b| {
        let ha = u32::from(a.1.uses >= 2);
        let hb = u32::from(b.1.uses >= 2);
        hb.cmp(&ha).then(b.0.cmp(&a.0))
    });
    let mut chosen: Vec<usize> = Vec::new();
    let mut used = 0;
    for (i, m) in ranked {
        let cost = m.text.chars().count() + 3;
        if used + cost > cap {
            continue;
        }
        chosen.push(i);
        used += cost;
    }
    chosen.sort_unstable();
    (chosen.iter().map(|i| memories[*i].text.clone()).collect(), memories.len() - chosen.len())
}


#[cfg(test)]
mod tests {
    use super::*;

    fn mem(id: &str, text: &str, uses: u32) -> Memory {
        Memory { id: id.into(), text: text.into(), created_at: 1, updated_at: 1, uses, category: None }
    }

    fn v(o: serde_json::Value) -> serde_json::Value {
        o
    }

    #[test]
    fn remember_that_at_the_start_gives_the_first_paragraph() {
        assert_eq!(extract("remember that the pump runs at 40k rpm").as_deref(), Some("the pump runs at 40k rpm"));
        assert_eq!(extract("Please remember: I prefer metric units").as_deref(), Some("I prefer metric units"));
        assert_eq!(extract("REMEMBER THAT I fly on the left seat").as_deref(), Some("I fly on the left seat"));
        assert_eq!(extract("remember that\n\nand this is another paragraph").as_deref(), Some("and this is another paragraph"));
        assert_eq!(extract("  remember   that   extra   spaces  ").as_deref(), Some("extra spaces"));
        assert_eq!(extract("what should I remember?"), None, "only at the start");
        assert_eq!(extract("remember"), None, "without a thing to remember");
        assert_eq!(extract("remember that    "), None, "without a thing to remember");
        assert_eq!(extract("remembers that x"), None, "the word must be remember");
    }

    #[test]
    fn a_secret_is_never_saved() {
        assert!(secret("password: hunter2secret"));
        assert!(secret("my api key is abc12def34ghi"));
        assert!(secret("-----BEGIN RSA PRIVATE KEY----- MIIE"));
        assert!(secret("token=ghp_abcdefghijklmnop"));
        assert!(secret("4111 1111 1111 1111"), "a card number by Luhn");
        assert!(secret("DE89 3704 0044 0532 0130 00"), "an IBAN");
        assert!(!secret("The pump runs dry."));
        assert!(!secret("My pinboard has 12 cards"), "a pin without a value is no secret");
    }

    #[test]
    fn adding_checks_empty_secret_duplicate_and_full() {
        let now = 7;
        let mut list = Vec::new();
        let (next, status, _entry) = add(&list, "  Prefers   metric units ", now, || "m1".into(), Some("Profile!"));
        assert_eq!(status, "added");
        assert_eq!(next[0].text, "Prefers metric units");
        assert_eq!(next[0].category.as_deref(), Some("profile"), "the category is cleaned");
        list = next;
        let (next, status, _) = add(&list, "prefers METRIC units", now, || "m2".into(), None);
        assert_eq!(status, "duplicate", "duplicates ignore case");
        assert_eq!(next.len(), 1);
        let (_, status, _) = add(&list, "password: hunter2secret", now, || "m3".into(), None);
        assert_eq!(status, "secret");
        let (_, status, _) = add(&list, "   ", now, || "m4".into(), None);
        assert_eq!(status, "empty");
        let full: Vec<Memory> = (0..limits::MEMORIES).map(|i| mem(&format!("m{i}"), &format!("fact {i}"), 0)).collect();
        let (_, status, _) = add(&full, "one more", now, || "m5".into(), None);
        assert_eq!(status, "full");
    }

    #[test]
    fn the_tool_adds_lists_updates_and_removes() {
        let now = 10;
        let list: Vec<Memory> = vec![mem("a1", "Flies the C172", 0), mem("a2", "Prefers metric units", 1)];
        // list, in the Studio's numbering
        let out = tool_apply(&list, &v(serde_json::json!({"action": "list"})), now, || "x".into());
        assert_eq!(out.text, "1. [a1] Flies the C172\n2. [a2] Prefers metric units");
        assert!(out.change.is_none());
        // add
        let out = tool_apply(&list, &v(serde_json::json!({"action": "add", "text": "Based in Berlin", "category": "profile"})), now, || "a3".into());
        assert_eq!(out.text, "Saved.");
        assert_eq!(out.list.len(), 3);
        assert_eq!(out.change.as_ref().unwrap().kind, "add");
        // a duplicate counts as a use
        let out = tool_apply(&list, &v(serde_json::json!({"action": "add", "text": "flies the c172"})), now, || "a4".into());
        assert_eq!(out.text, "Already remembered.");
        assert_eq!(out.list[0].uses, 1, "the entry was used once more");
        // update by a part of the text
        let out = tool_apply(&list, &v(serde_json::json!({"action": "update", "text": "Flies the DA40", "match": "C172"})), now, || "x".into());
        assert_eq!(out.text, "Updated.");
        assert_eq!(out.list[0].text, "Flies the DA40");
        assert_eq!(out.change.as_ref().unwrap().kind, "update");
        // remove by id
        let out = tool_apply(&list, &v(serde_json::json!({"action": "remove", "id": "a2"})), now, || "x".into());
        assert_eq!(out.text, "Removed.");
        assert_eq!(out.list.len(), 1);
        // the errors the model gets
        let out = tool_apply(&list, &v(serde_json::json!({"action": "nope"})), now, || "x".into());
        assert_eq!(out.text, "Error: \"action\" must be add, update, remove or list.");
        let out = tool_apply(&list, &v(serde_json::json!({"action": "remove", "match": "zzz"})), now, || "x".into());
        assert!(out.text.starts_with("Error: no matching entry."));
        let out = tool_apply(&list, &v(serde_json::json!({"action": "add", "text": "password: hunter2secret"})), now, || "x".into());
        assert!(out.text.starts_with("Not saved:"), "a secret is refused with the model's line");
    }

    #[test]
    fn a_change_can_be_taken_back() {
        let list: Vec<Memory> = vec![mem("a1", "Flies the C172", 0)];
        let added = Change { kind: "add".into(), id: "a2".into(), before: None, after: Some(mem("a2", "New fact", 0)) };
        let mut with = list.clone();
        with.push(mem("a2", "New fact", 0));
        assert_eq!(undo(&with, &added).unwrap().len(), 1, "the added entry goes away");
        assert!(undo(&list, &added).is_none(), "it is not there any more");
        let removed = Change { kind: "remove".into(), id: "a1".into(), before: Some(mem("a1", "Flies the C172", 0)), after: None };
        assert_eq!(undo(&[], &removed).unwrap()[0].id, "a1", "the removed entry comes back");
        let updated = Change { kind: "update".into(), id: "a1".into(), before: Some(mem("a1", "Flies the C172", 0)), after: Some(mem("a1", "Flies the DA40", 1)) };
        let now_list = vec![mem("a1", "Flies the DA40", 1)];
        assert_eq!(undo(&now_list, &updated).unwrap()[0].text, "Flies the C172", "the old text comes back");
    }

    #[test]
    fn the_prompt_gets_the_useful_entries_within_its_cap() {
        let list: Vec<Memory> = vec![
            mem("a1", "old but used twice", 2),
            mem("a2", "new fact one", 0),
            mem("a3", "new fact two", 0),
        ];
        let (lines, dropped) = select(&list, 3_000);
        assert_eq!(dropped, 0);
        assert_eq!(lines.len(), 3);
        // the used one is sent first in the block, but the chosen keep their order in the list
        // the used entry wins the budget of a small cap (its 18 characters plus 3 cost 21)
        let (lines, dropped) = select(&list, 21);
        assert_eq!(lines, vec!["old but used twice".to_string()], "uses >= 2 are sent first");
        assert_eq!(dropped, 2, "the cap leaves the others out");
        let (lines, dropped) = select(&list, 14);
        assert_eq!(lines.len(), 0, "nothing fits");
        assert_eq!(dropped, 3);
    }

    #[test]
    fn old_entries_from_a_store_are_cleaned() {
        let raw = vec![
            serde_json::json!({"id": "m1", "text": "Keeps a logbook", "createdAt": 5}),
            serde_json::json!({"id": "", "text": "   "}),
            serde_json::json!({"text": "x".repeat(500), "uses": 9999, "category": "Prefernces!"}),
        ];
        let list = sanitize(&raw, || "generated".into());
        assert_eq!(list.len(), 2, "an entry without text is dropped");
        assert_eq!(list[0].id, "m1");
        assert_eq!(list[0].updated_at, 5, "without updatedAt the createdAt counts");
        assert_eq!(list[1].text.chars().count(), limits::MEMORY, "the text is cut");
        assert_eq!(list[1].uses, 1000, "the uses are clamped");
        assert_eq!(list[1].category.as_deref(), Some("prefernces"), "the category is cleaned");
    }
}
