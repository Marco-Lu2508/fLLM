//! The prompt library: the saved prompts of Settings > Personalization and the menu under a "/" in
//! the message box (the Studio's `plibMatch` and the rows of `plibList` in projects.js).

use serde::{Deserialize, Serialize};

/// The limits of the Studio (`PRJ_LIMITS`).
pub mod limits {
    /// A title of a saved prompt (`PRJ_LIMITS.promptTitle`).
    pub const TITLE: usize = 60;
    /// The text of a saved prompt (`PRJ_LIMITS.promptText`).
    pub const TEXT: usize = 8000;
    /// How many prompts the library holds (`PRJ_LIMITS.prompts`).
    pub const PROMPTS: usize = 100;
    /// How many entries the menu under a "/" shows (`plibMatch` slices to eight).
    pub const MENU: usize = 8;
}

/// One saved prompt (`state.prj.prompts`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SavedPrompt {
    pub id: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub text: String,
}

/// The built-in commands of the box (`SLASH_COMMANDS`): a "/" means one of them first, so the
/// library keeps out of their way.
fn builtin(q: &str) -> bool {
    let first = q.split_whitespace().next().unwrap_or("");
    matches!(first, "clear" | "new" | "system" | "export" | "help" | "retry")
}

/// Saved prompts that match what is typed after a leading "/" (the built-in commands are left
/// alone): the indices into `prompts`, the ones whose title starts with the search first, eight at
/// most (`projects.js` plibMatch — checked against that code in the tests).
pub fn match_prompts(prompts: &[SavedPrompt], text: &str) -> Vec<usize> {
    if !text.starts_with('/') {
        return Vec::new();
    }
    let q: String = text.chars().skip(1).collect();
    let q = q.to_lowercase();
    let q = q.trim();
    if builtin(q) {
        return Vec::new();
    }
    let mut hits: Vec<usize> = prompts
        .iter()
        .enumerate()
        .filter(|(_, p)| p.title.to_lowercase().contains(q))
        .map(|(i, _)| i)
        .collect();
    // the ones whose title starts with the search first; inside a group the order of the list (the
    // sort of the Studio is stable, so a Rust sort_by_key does the same)
    hits.sort_by_key(|&i| {
        if prompts[i].title.to_lowercase().starts_with(q) {
            0
        } else {
            1
        }
    });
    hits.truncate(limits::MENU);
    hits
}

/// The small line under a title in the menu: every whitespace collapsed to one space, the first
/// eighty characters (`plibUpdateSlash`).
pub fn preview(text: &str) -> String {
    let collapsed: String = text.split_whitespace().collect::<Vec<&str>>().join(" ");
    collapsed.chars().take(80).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_menu_finds_the_prompts_the_way_the_studio_does() {
        // the fixtures are produced by plibMatch of the Studio itself (see fixtures/)
        let data: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/promptlib_fixtures.json")).expect("the fixtures are json");
        let prompts: Vec<SavedPrompt> = serde_json::from_value(data["prompts"].clone()).expect("the prompts");
        assert_eq!(prompts.len(), 11, "the fixture library");
        let cases = data["cases"].as_array().expect("the cases");
        assert_eq!(cases.len(), 16, "every input of the fixture list runs");
        for case in cases {
            let text = case["text"].as_str().expect("the input");
            let expected: Vec<String> = case["indices"]
                .as_array()
                .expect("the match of the Studio")
                .iter()
                .map(|p| p["id"].as_str().expect("an id").to_string())
                .collect();
            let got: Vec<String> = match_prompts(&prompts, text)
                .into_iter()
                .map(|i| prompts[i].id.clone())
                .collect();
            assert_eq!(got, expected, "for {text:?}");
        }
    }

    #[test]
    fn a_line_after_the_slash_is_no_search() {
        // the caller (plibUpdateSlash) only asks while it is one line; a search that would span
        // two of them simply finds nothing, because the line break is part of what is searched for
        let prompts = vec![SavedPrompt { id: "pr1".into(), title: "One".into(), text: "T".into() }];
        assert!(match_prompts(&prompts, "/One\nmore").is_empty(), "no title contains a line break");
        assert_eq!(match_prompts(&prompts, "One"), Vec::<usize>::new(), "without the slash there is no menu");
    }

    #[test]
    fn the_preview_collapses_the_whitespace_and_stops_at_eighty_characters() {
        assert_eq!(preview("  a\n\tb   c  "), "a b c");
        let long: String = "x".repeat(120);
        assert_eq!(preview(&long).chars().count(), 80, "no ellipsis, just cut (the Studio slices)");
        assert_eq!(preview("Ünïcodé läuft"), "Ünïcodé läuft", "characters, not bytes");
    }

    #[test]
    fn the_library_holds_what_the_studio_holds() {
        assert_eq!(limits::TITLE, 60, "PRJ_LIMITS.promptTitle");
        assert_eq!(limits::TEXT, 8000, "PRJ_LIMITS.promptText");
        assert_eq!(limits::PROMPTS, 100, "PRJ_LIMITS.prompts");
        assert_eq!(limits::MENU, 8, "plibMatch");
    }
}
