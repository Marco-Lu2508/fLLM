//! Usage and cost: tokens per model, and what they cost with the prices the user entered — the
//! `usageModule` of the Studio (`usage.js`).
//!
//! - Counted from the usage block every stream returns (`prompt_tokens`, `completion_tokens`).
//! - Prices come from `GET/POST /api/studio/prices`; nothing is prefilled, because a local model
//!   costs nothing and provider prices differ per plan. Unpriced models show tokens without a
//!   dollar amount.
//! - A pattern is matched against the model name; the longest match wins, `*` matches any tail
//!   (for example `gpt-6*`).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// How many models are counted at most: when a new one arrives the first key steps aside
/// (`MAX_MODELS` of `usage.js`).
pub const MAX_MODELS: usize = 60;

/// What one model has seen: answers, input tokens, output tokens.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Entry {
    pub calls: u64,
    pub prompt: u64,
    pub completion: u64,
}

/// The counters of this machine (`fllm.studio.usage` of the Studio's localStorage, in the store
/// of the app instead).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct UsageData {
    #[serde(default)]
    pub by_model: BTreeMap<String, Entry>,
    #[serde(default)]
    pub since: String,
}

/// One price row: what a million tokens cost with a provider (`prices.json`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Price {
    pub pattern: String,
    pub input_per_million: f64,
    pub output_per_million: f64,
    pub note: String,
    /// What the two number fields show: the values as text (the inputs of the Studio are text
    /// too; `String(0)` is "0", `String(1.5)` is "1.5"). Not sent to the server.
    #[serde(skip)]
    pub input_text: String,
    #[serde(skip)]
    pub output_text: String,
}

impl Price {
    /// Puts the two numbers into the text the fields show.
    pub fn refresh_text(&mut self) {
        self.input_text = format!("{}", self.input_per_million);
        self.output_text = format!("{}", self.output_per_million);
    }
}

/// What the counters add up to.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Totals {
    pub calls: u64,
    pub prompt: u64,
    pub completion: u64,
    /// The dollars of the models that have a price.
    pub cost: f64,
    /// How many of the counted models have a price.
    pub priced: u64,
}

/// The key of a model in the counters: `unknown` without a name, cut after 120 characters —
/// `modelKey` of `usage.js`.
pub fn model_key(model: &str) -> String {
    let key = if model.is_empty() { "unknown" } else { model };
    key.chars().take(120).collect()
}

/// Adds one answer's usage (nothing without tokens; the first key steps aside beyond 60 models)
/// — `record` of `usage.js`.
pub fn record(data: &mut UsageData, model: &str, prompt: u64, completion: u64) {
    if prompt == 0 && completion == 0 {
        return;
    }
    let key = model_key(model);
    let entry = data.by_model.entry(key).or_default();
    entry.calls += 1;
    entry.prompt += prompt;
    entry.completion += completion;
    if data.by_model.len() > MAX_MODELS {
        let first = data.by_model.keys().next().cloned();
        if let Some(key) = first {
            data.by_model.remove(&key);
        }
    }
}

/// The price row that fits the model name best (longest pattern wins; `*` matches any tail) —
/// `priceFor` of `usage.js`.
pub fn price_for<'a>(model: &str, prices: &'a [Price]) -> Option<&'a Price> {
    let name = model.to_lowercase();
    let mut best: Option<(&'a Price, usize)> = None;
    for row in prices {
        let pattern = row.pattern.to_lowercase().trim().to_string();
        if pattern.is_empty() {
            continue;
        }
        let hit = if let Some(prefix) = pattern.strip_suffix('*') {
            name.starts_with(prefix)
        } else {
            name == pattern || name.contains(&pattern)
        };
        let length = pattern.chars().count();
        if hit && best.is_none_or(|(_, best_length)| length > best_length) {
            best = Some((row, length));
        }
    }
    best.map(|(row, _)| row)
}

/// What the answers of one model cost with the price that fits it — `costOf` of `usage.js`.
pub fn cost_of(model: &str, entry: &Entry, prices: &[Price]) -> Option<f64> {
    let price = price_for(model, prices)?;
    Some(
        entry.prompt as f64 / 1e6 * price.input_per_million
            + entry.completion as f64 / 1e6 * price.output_per_million,
    )
}

/// Everything together — `totals` of `usage.js`.
pub fn totals(data: &UsageData, prices: &[Price]) -> Totals {
    let mut out = Totals::default();
    for (model, entry) in &data.by_model {
        out.calls += entry.calls;
        out.prompt += entry.prompt;
        out.completion += entry.completion;
        if let Some(cost) = cost_of(model, entry, prices) {
            out.cost += cost;
            out.priced += 1;
        }
    }
    out
}

/// The rows of the table: the models by everything they have seen, the biggest first —
/// `renderUsage` of `usage.js`.
pub fn rows_sorted(data: &UsageData) -> Vec<(&String, &Entry)> {
    let mut rows: Vec<(&String, &Entry)> = data.by_model.iter().collect();
    rows.sort_by(|a, b| {
        (b.1.prompt + b.1.completion)
            .cmp(&(a.1.prompt + a.1.completion))
            .then(a.0.cmp(b.0))
    });
    rows
}

/// How many tokens: millions with two decimals, thousands with one, small numbers as they are —
/// `fmtTokens` of `usage.js`.
pub fn fmt_tokens(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

/// How many dollars: a dash without a price, four decimals below a cent, two otherwise —
/// `fmtMoney` of `usage.js`.
pub fn fmt_money(n: Option<f64>) -> String {
    match n {
        None => String::from("—"),
        Some(v) if v < 0.01 => format!("${v:.4}"),
        Some(v) => format!("${v:.2}"),
    }
}

/// The line above the table — `renderUsage` of `usage.js`.
pub fn totals_text(data: &UsageData, prices: &[Price]) -> String {
    let t = totals(data, prices);
    format!(
        "{} in / {} out in {} answer(s)",
        fmt_tokens(t.prompt),
        fmt_tokens(t.completion),
        t.calls
    ) + &if t.priced > 0 {
        format!(" · about {}", fmt_money(Some(t.cost)))
    } else {
        String::from(" · no prices entered, so no dollar amount")
    }
}

/// What `/cost` says about it — `fllmUsageSummary` of `usage.js`.
pub fn summary_text(data: &UsageData, prices: &[Price]) -> String {
    let t = totals(data, prices);
    format!(
        "{} in / {} out in {} answer(s)",
        fmt_tokens(t.prompt),
        fmt_tokens(t.completion),
        t.calls
    ) + &if t.priced > 0 {
        format!(", about {}", fmt_money(Some(t.cost)))
    } else {
        String::from(" (no prices entered)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn price(pattern: &str, input: f64, output: f64) -> Price {
        Price {
            pattern: pattern.into(),
            input_per_million: input,
            output_per_million: output,
            note: String::new(),
            input_text: String::new(),
            output_text: String::new(),
        }
    }

    /// One answer lands in the counters; without tokens nothing happens; the key is cut.
    #[test]
    fn one_answer_lands_in_the_counters() {
        let mut data = UsageData::default();
        record(&mut data, "qwen3-4b", 922, 52);
        record(&mut data, "qwen3-4b", 10, 5);
        assert_eq!(data.by_model["qwen3-4b"].calls, 2, "two answers");
        assert_eq!(data.by_model["qwen3-4b"].prompt, 932);
        record(&mut data, "", 0, 0);
        assert_eq!(data.by_model.len(), 1, "nothing without tokens, and the name becomes unknown only with tokens");
        record(&mut data, &"x".repeat(500), 1, 1);
        let long = data.by_model.keys().find(|k| k.len() == 120).expect("the key is cut at 120");
        assert!(long.starts_with("xxx"));
    }

    /// Beyond sixty models the first key steps aside (MAX_MODELS of the Studio).
    #[test]
    fn beyond_sixty_models_the_first_steps_aside() {
        let mut data = UsageData::default();
        for i in 0..61 {
            record(&mut data, &format!("model-{i:02}"), 1, 1);
        }
        assert_eq!(data.by_model.len(), 60, "sixty stay");
    }

    /// The price row that fits wins by the longest pattern; `*` matches any tail, an empty
    /// pattern fits nothing, and a model without a row has no cost.
    #[test]
    fn the_longest_pattern_wins_and_a_star_matches_any_tail() {
        let prices = vec![price("gpt*", 1.0, 2.0), price("gpt-6-mini", 0.1, 0.2), price("  ", 9.0, 9.0)];
        assert_eq!(
            price_for("gpt-6-mini", &prices).map(|p| p.pattern.as_str()),
            Some("gpt-6-mini"),
            "the longer pattern wins over the star"
        );
        assert_eq!(price_for("gpt-5", &prices).map(|p| p.pattern.as_str()), Some("gpt*"), "the star matches any tail");
        assert!(price_for("claude", &prices).is_none(), "nothing fits");
        let entry = Entry { calls: 1, prompt: 1_000_000, completion: 500_000 };
        assert_eq!(cost_of("gpt-6-mini", &entry, &prices), Some(0.1 + 0.1), "a million in, half a million out");
        assert_eq!(cost_of("claude", &entry, &prices), None, "no price, no dollars");
    }

    /// The totals count every model and only the dollars of the priced ones.
    #[test]
    fn the_totals_count_everything_and_only_the_priced_dollars() {
        let mut data = UsageData::default();
        record(&mut data, "gpt-6-mini", 1_000_000, 500_000);
        record(&mut data, "local-model", 1_000, 10);
        let prices = vec![price("gpt*", 1.0, 2.0)];
        let t = totals(&data, &prices);
        assert_eq!((t.calls, t.prompt, t.completion), (2, 1_001_000, 500_010));
        assert_eq!(t.priced, 1, "only the model with a price");
        // a million input tokens at one dollar, half a million output tokens at two
        assert!((t.cost - 2.0).abs() < 1e-9, "one dollar plus one dollar");
    }

    /// Tokens and dollars are written the way the Studio writes them, and so are the two lines.
    #[test]
    fn the_words_are_the_ones_of_the_studio() {
        assert_eq!(fmt_tokens(999), "999");
        assert_eq!(fmt_tokens(1_500), "1.5k");
        assert_eq!(fmt_tokens(2_500_000), "2.50M");
        assert_eq!(fmt_money(None), "—");
        assert_eq!(fmt_money(Some(0.0012)), "$0.0012", "below a cent: four decimals");
        assert_eq!(fmt_money(Some(1.5)), "$1.50");
        let mut data = UsageData::default();
        assert_eq!(
            totals_text(&data, &[]),
            "0 in / 0 out in 0 answer(s) · no prices entered, so no dollar amount"
        );
        assert_eq!(summary_text(&data, &[]), "0 in / 0 out in 0 answer(s) (no prices entered)");
        record(&mut data, "gpt-6-mini", 1_000_000, 500_000);
        let prices = vec![price("gpt*", 1.0, 2.0)];
        assert_eq!(
            totals_text(&data, &prices),
            "1.00M in / 500.0k out in 1 answer(s) · about $2.00"
        );
        assert_eq!(summary_text(&data, &prices), "1.00M in / 500.0k out in 1 answer(s), about $2.00");
    }

    /// The rows of the table stand by everything they have seen, the biggest first.
    #[test]
    fn the_rows_are_ordered_by_everything_they_have_seen() {
        let mut data = UsageData::default();
        record(&mut data, "small", 1, 1);
        record(&mut data, "big", 100, 100);
        record(&mut data, "middle", 50, 50);
        let order: Vec<&str> = rows_sorted(&data).iter().map(|(m, _)| m.as_str()).collect();
        assert_eq!(order, vec!["big", "middle", "small"]);
    }
}
