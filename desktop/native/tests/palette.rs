//! The command palette against what `palette.js` of the Studio answers: the fixtures are produced
//! by the `@pure-begin`..`@pure-end` block of that file itself — the score of a text for a query,
//! the order of the rows in blocks per kind, and the row an arrow key takes.

use fllm_app::palette::{
    palette_score, palette_step, rank_palette, snippet, Kind, PaletteItem,
};
use serde_json::Value;

fn fixtures() -> Value {
    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/palette_fixtures.json"
    ))
    .expect("the fixtures of palette.js are there");
    serde_json::from_str(&raw).expect("the fixtures are json")
}

fn item_of(v: &Value) -> PaletteItem {
    let kind = match v["kind"].as_str().unwrap_or("Command") {
        "Chat" => Kind::Chat,
        "Setting" => Kind::Setting,
        "Code session" => Kind::CodeSession,
        _ => Kind::Command,
    };
    PaletteItem {
        id: v["id"].as_str().unwrap_or_default().to_string(),
        kind,
        title: v["title"].as_str().unwrap_or_default().to_string(),
        sub: v["sub"].as_str().unwrap_or_default().to_string(),
        keywords: v["keywords"].as_str().unwrap_or_default().to_string(),
        hint: String::new(),
        weak: v["weak"].as_bool().unwrap_or(false),
    }
}

/// One text scored for a query the way `paletteScore` of the Studio scores it: 0 = no match.
#[test]
fn a_text_scores_the_way_palette_js_scores_it() {
    let f = fixtures();
    for case in f["score"].as_array().expect("score cases") {
        let fields = case.as_array().expect("a case array");
        let [text, query, expected] = fields.as_slice() else {
            panic!("the case has its three fields");
        };
        let text = text.as_str().unwrap();
        let query = query.as_str().unwrap();
        assert_eq!(
            palette_score(text, query),
            expected.as_i64().unwrap(),
            "paletteScore({text:?}, {query:?}) of palette.js"
        );
    }
}

/// The rows in the order the Studio puts them: without a query the first commands then the chats,
/// with one the best first and one block per kind (`rankPalette` of `palette.js`).
#[test]
fn the_rows_are_ranked_the_way_palette_js_ranks_them() {
    let f = fixtures();
    for case in f["rank"].as_array().expect("rank cases") {
        let fields = case.as_array().expect("a case array");
        let [query, items_json, limit, expected] = fields.as_slice() else {
            panic!("the case has its four fields");
        };
        let query = query.as_str().unwrap();
        let items: Vec<PaletteItem> = items_json
            .as_array()
            .expect("the items")
            .iter()
            .map(item_of)
            .collect();
        let ranked = rank_palette(query, &items, limit.as_u64().unwrap_or(40) as usize);
        let got: Vec<&str> = ranked.iter().map(|i| items[*i].id.as_str()).collect();
        let want: Vec<&str> = expected
            .as_array()
            .expect("the ids")
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(got, want, "rankPalette({query:?}) of palette.js");
    }
}

/// The arrow keys wrap around at both ends and Home and End jump (`paletteStep` of `palette.js`).
#[test]
fn the_arrow_keys_take_the_row_of_palette_js() {
    let f = fixtures();
    for case in f["step"].as_array().expect("step cases") {
        let fields = case.as_array().expect("a case array");
        let [index, count, key, expected] = fields.as_slice() else {
            panic!("the case has its four fields");
        };
        assert_eq!(
            palette_step(
                index.as_i64().unwrap() as i32,
                count.as_u64().unwrap() as usize,
                key.as_str().unwrap()
            ),
            expected.as_i64().unwrap() as i32,
            "paletteStep({index}, {count}, {key:?}) of palette.js"
        );
    }
}

/// The text around a hit is thirty characters before, the hit, fifty after, the spaces folded and
/// an ellipsis where the text continues (`snippet` of `boot.js`).
#[test]
fn the_snippet_around_a_hit_is_the_one_of_boot_js() {
    let long = format!("{} gold price record today {}", "a".repeat(50), "b".repeat(80));
    let s = snippet(&long, "gold");
    assert!(s.starts_with('…'), "the text before is cut: {s:?}");
    assert!(s.contains("gold price"), "the hit with what follows: {s:?}");
    assert!(s.ends_with('…'), "the text after continues: {s:?}");
    // a hit at the start has no ellipsis before it
    let short = snippet("gold price", "gold");
    assert_eq!(short, "gold price…", "nothing before the hit: {short:?}");
    // nothing found is nothing
    assert_eq!(snippet("gold price", "silver"), "");
    // the spaces around the hit are folded to one
    // the hit sits within the first thirty characters, so nothing is cut before it
    let spaces = snippet("gold    price  today", "price");
    assert_eq!(spaces, "gold price today…", "the spaces are folded: {spaces:?}");
    // behind the thirtieth character the text before it is cut with an ellipsis
    let far = snippet(&format!("{} price", "g".repeat(40)), "price");
    assert!(far.starts_with('…') && far.contains("price"), "cut before: {far:?}");
}
