//! Row 80 of PARITY.md: *Specs and steering files* — the specs of `.fllm/specs/<name>/` as the
//! Code side panel keeps them, the commands that start one and run its tasks, and the notices
//! the page words for them (`specCommand`, `runSpecTask`, `specAction` and `specsHtml` of
//! `code.js`; the list itself comes from `GET /api/studio/agent/specs`).
//!
//! The steering files of the row are the server's: they arrive with `POST
//! /api/studio/agent/instructions` (the `always` ones and the index) and reach the model over
//! `Context` like every other instruction file — nothing of them is drawn separately, as on the
//! page (it has no steering pane either).

use serde::Deserialize;
use serde_json::Value;

/// One task of a spec as the list carries it (`parse_tasks` → `list_json`).
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct SpecTask {
    #[serde(default)]
    pub index: u64,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub done: bool,
    #[serde(default)]
    pub depth: u64,
}

/// One spec of the folder: its counts, which of the three files are written, and its tasks.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct Spec {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub requirements: bool,
    #[serde(default)]
    pub design: bool,
    #[serde(default)]
    pub tasks_file: bool,
    #[serde(default)]
    pub total: usize,
    #[serde(default)]
    pub done: usize,
    #[serde(default)]
    pub tasks: Vec<SpecTask>,
}

/// `Number(word)` of the page for one word of `/spec run NAME N`: decimal with a sign, a
/// fraction or an exponent, `Infinity`, and the `0x` / `0o` / `0b` prefixes; anything else is
/// `NaN` (the words come out of a whitespace split, so an empty one cannot occur).
pub fn js_number(word: &str) -> f64 {
    let lower = word.to_ascii_lowercase();
    for (prefix, radix) in [("0x", 16), ("0o", 8), ("0b", 2)] {
        if let Some(digits) = lower.strip_prefix(prefix) {
            return u128::from_str_radix(digits, radix).map(|n| n as f64).unwrap_or(f64::NAN);
        }
    }
    let (sign, rest) = match word.strip_prefix('-') {
        Some(rest) => (-1.0, rest),
        None => (1.0, word.strip_prefix('+').unwrap_or(word)),
    };
    if rest == "Infinity" {
        return sign * f64::INFINITY;
    }
    // Rust also reads `inf`, `nan` and `infinity`; the page does not
    if !rest.starts_with(|c: char| c.is_ascii_digit() || c == '.') || !rest.chars().all(|c| c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '+' | '-')) {
        return f64::NAN;
    }
    rest.parse::<f64>().map(|n| sign * n).unwrap_or(f64::NAN)
}

/// What `Number(words[2]) ||` leaves of the asked task: `NaN` and `0` are falsy and fall
/// through to the first open task.
pub fn asked_task(word: Option<&str>) -> Option<f64> {
    word.map(js_number).filter(|n| *n != 0.0 && !n.is_nan())
}

/// The next task `run` would do: the number that was asked for (`Number(...)` of the page, so
/// a negative or a fraction goes through as it is), else the first open task, and `None` when
/// every task is done (`case 'spec'`, the `run` branch).
pub fn next_index(spec: &Spec, asked: Option<f64>) -> Option<f64> {
    if let Some(n) = asked {
        return Some(n);
    }
    spec.tasks.iter().find(|t| !t.done).map(|t| t.index as f64)
}

/// A task number the way `JSON.stringify` writes it: a whole number without a fraction, and
/// `null` for what JSON cannot hold (`Infinity`).
pub fn index_json(index: f64) -> Value {
    if !index.is_finite() {
        Value::Null
    } else if index.fract() == 0.0 && index.abs() < 9e15 {
        Value::from(index as i64)
    } else {
        Value::from(index)
    }
}

/// A task number the way a template literal writes it (`${index}`).
pub fn index_text(index: f64) -> String {
    if index.is_infinite() {
        return if index > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if index.fract() == 0.0 && index.abs() < 1e21 {
        return format!("{index:.0}");
    }
    format!("{index}")
}

/// `GET /api/studio/agent/specs` → the list (`loadSpecs` of the page).
pub async fn fetch_specs(base: String) -> Result<Vec<Spec>, String> {
    let v = crate::code::call_get(&base, "/api/studio/agent/specs").await?;
    Ok(parse_specs(&v))
}

/// The `specs` of an answer of that route (`data.specs || []` where the page reads it).
pub fn parse_specs(v: &Value) -> Vec<Spec> {
    v.get("specs")
        .and_then(Value::as_array)
        .map(|rows| rows.iter().filter_map(|row| serde_json::from_value(row.clone()).ok()).collect())
        .unwrap_or_default()
}

/// `POST /api/studio/agent/specs` (`specAction`): the answer, or the error the page throws —
/// the message of its body, its own fallback without one, and a request that did not arrive
/// as it is.
pub async fn spec_act(base: String, body: Value) -> Result<Value, String> {
    match crate::code::call_post(&base, "/api/studio/agent/specs", body).await {
        Ok(v) => Ok(v),
        Err(e) => Err(if e.starts_with("the server answered") {
            "The spec action failed.".to_string()
        } else {
            e
        }),
    }
}

/// `tick` for a box that was just ticked, `untick` for one that was cleared (`t.checked ? 'tick'
/// : 'untick'` of the side's handler).
pub fn tick_action(done: bool) -> &'static str {
    if done {
        "tick"
    } else {
        "untick"
    }
}

/// What the list shows for a run the specs started (`/spec run ${name} ${index}`).
pub fn run_display(name: &str, index: f64) -> String {
    format!("/spec run {name} {}", index_text(index))
}

/// `margin-left: ${t.depth * 12}px` of a task of the side.
pub fn task_indent(depth: u64) -> f32 {
    (depth * 12) as f32
}

/// The title of a file chip (`${file}${sp[k] ? '' : ' (not written yet)'}`).
pub fn chip_title(file: &str, written: bool) -> String {
    if written {
        file.to_string()
    } else {
        format!("{file} (not written yet)")
    }
}

/// The body of `tick` / `untick` (the checkbox of a task in the side panel).
pub fn tick_body(action: &str, name: &str, index: u64) -> Value {
    serde_json::json!({ "action": action, "name": name, "index": index })
}

/// The body of `task_prompt` (the *Run next task* button and `/spec run`).
pub fn task_prompt_body(name: &str, index: f64) -> Value {
    serde_json::json!({ "action": "task_prompt", "name": name, "index": index_json(index) })
}

/// The body of `new_prompt` (`/spec NAME what you want`).
pub fn new_prompt_body(name: &str, description: &str) -> Value {
    serde_json::json!({ "action": "new_prompt", "name": name, "description": description })
}

/// What `/spec run` decides against one list (the page: find, reload once when the name is
/// missing, pick with `next_index`).
#[derive(Debug, Clone, PartialEq)]
pub enum RunDecision {
    /// The task to run: the spec's name as the list has it, and the index.
    Act { name: String, index: f64 },
    /// The list does not hold the name: reload and decide again (`if (!sp) await loadSpecs()`).
    NeedReload,
    /// After a reload: there is no such spec.
    NotFound,
    /// The spec exists and every task of it is done.
    AllDone,
}

/// `case 'spec'`, the `run` branch, over the list the window currently has.
pub fn decide_run(current: &[Spec], name: &str, asked: Option<f64>) -> RunDecision {
    let Some(sp) = current.iter().find(|s| s.name == name) else {
        return RunDecision::NeedReload;
    };
    match next_index(sp, asked) {
        Some(index) => RunDecision::Act { name: sp.name.clone(), index },
        None => RunDecision::AllDone,
    }
}

/// The listing of `/spec` with no argument: the specs with their counts, or the pitch of the
/// page for the folder without one (`case 'spec'`, the empty branch).
pub fn listing_notice(specs: &[Spec]) -> String {
    if specs.is_empty() {
        return "No specs yet. /spec NAME what you want writes requirements.md, design.md and tasks.md under .fllm/specs/NAME and stops after each file for your review.".to_string();
    }
    let rows: Vec<String> = specs.iter().map(|s| format!("{}: {}/{} tasks done", s.name, s.done, s.total)).collect();
    format!(
        "Specs of this folder:\n{}\n/spec run NAME [N] does the next open task (or task N); /spec NAME what you want starts a new spec.",
        rows.join("\n")
    )
}

/// `There is no spec "{name}". /spec lists them.` — the page's answer for a name it does not
/// have (after reloading once).
pub fn no_spec_notice(name: &str) -> String {
    format!("There is no spec \"{name}\". /spec lists them.")
}

/// `Every task of this spec is done.`
pub const ALL_DONE_NOTICE: &str = "Every task of this spec is done.";

/// What the side says while the folder has no spec (the words of `specsHtml`, its two mono
/// spans inline).
pub const PITCH: &str = "Plan a feature before coding it: /spec name what you want writes requirements, design and a task list under .fllm/specs/ and builds it task by task.";

/// The usage line of `/spec NAME what you want`.
pub const SPEC_USAGE: &str = "Usage: /spec NAME what you want. Example: /spec csv-export Export the report as CSV.";

#[cfg(test)]
mod tests {
    use super::*;

    /// The index picker of `/spec run`: the asked number when it is one, the first open task
    /// otherwise, and nothing when the list is through.
    #[test]
    fn the_run_picks_the_task_the_page_picks() {
        let spec = Spec {
            name: "export".into(),
            total: 3,
            done: 1,
            tasks: vec![
                SpecTask { index: 1, text: "a".into(), done: true, depth: 0 },
                SpecTask { index: 2, text: "b".into(), done: false, depth: 0 },
                SpecTask { index: 3, text: "c".into(), done: false, depth: 1 },
            ],
            ..Default::default()
        };
        assert_eq!(next_index(&spec, asked_task(Some("3"))), Some(3.0), "the number that was asked for");
        assert_eq!(next_index(&spec, asked_task(Some("0"))), Some(2.0), "zero is falsy on the page");
        assert_eq!(next_index(&spec, asked_task(Some("x"))), Some(2.0), "NaN is falsy on the page");
        assert_eq!(next_index(&spec, asked_task(Some("-1"))), Some(-1.0), "a negative number is truthy");
        assert_eq!(next_index(&spec, None), Some(2.0), "the first open task");
        let done = Spec { tasks: vec![SpecTask { index: 1, done: true, ..Default::default() }], ..Default::default() };
        assert_eq!(next_index(&done, None), None, "nothing left");
    }

    /// The small words of the side and of the run: the action of a box, the display of a run,
    /// the indent of a nested task, the title of a chip.
    #[test]
    fn the_small_words_of_the_side() {
        assert_eq!(tick_action(true), "tick");
        assert_eq!(tick_action(false), "untick");
        assert_eq!(run_display("export", 2.0), "/spec run export 2");
        assert_eq!(run_display("export", 1.5), "/spec run export 1.5");
        assert_eq!(task_indent(0), 0.0);
        assert_eq!(task_indent(1), 12.0);
        assert_eq!(task_indent(3), 36.0);
        assert_eq!(chip_title("design.md", true), "design.md");
        assert_eq!(chip_title("design.md", false), "design.md (not written yet)");
    }

    /// `Number(word)` for the words the page can be given, and the two ways a task number is
    /// written out (`JSON.stringify` and a template literal).
    #[test]
    fn a_task_number_reads_and_writes_like_javascript() {
        for (word, want) in [("2", 2.0), ("+3", 3.0), ("-1", -1.0), ("1.5", 1.5), (".5", 0.5), ("5.", 5.0), ("1e1", 10.0), ("0x2", 2.0), ("0X10", 16.0), ("0o10", 8.0), ("0b11", 3.0), ("007", 7.0)] {
            assert_eq!(js_number(word), want, "Number({word:?})");
        }
        for word in ["x", "inf", "nan", "infinity", "1_0", "--1", "1.2.3", ".", "e5", "0x", "0xg", "-0x2", "1e", "+"] {
            assert!(js_number(word).is_nan(), "Number({word:?}) is NaN");
        }
        assert_eq!(js_number("Infinity"), f64::INFINITY);
        assert_eq!(js_number("-Infinity"), f64::NEG_INFINITY);
        assert_eq!(index_json(2.0), serde_json::json!(2), "a whole number goes out as an integer");
        assert_eq!(index_json(1.5), serde_json::json!(1.5));
        assert_eq!(index_json(f64::INFINITY), Value::Null, "JSON.stringify(Infinity) is null");
        assert_eq!(index_text(10.0), "10");
        assert_eq!(index_text(-1.0), "-1");
        assert_eq!(index_text(1.5), "1.5");
        assert_eq!(index_text(f64::INFINITY), "Infinity");
    }

    /// Both listings, word for word.
    #[test]
    fn the_listing_is_the_pages_text() {
        assert_eq!(
            listing_notice(&[]),
            "No specs yet. /spec NAME what you want writes requirements.md, design.md and tasks.md under .fllm/specs/NAME and stops after each file for your review."
        );
        let specs = vec![
            Spec { name: "csv-export".into(), done: 1, total: 3, ..Default::default() },
            Spec { name: "search".into(), done: 3, total: 3, ..Default::default() },
        ];
        assert_eq!(
            listing_notice(&specs),
            "Specs of this folder:\ncsv-export: 1/3 tasks done\nsearch: 3/3 tasks done\n/spec run NAME [N] does the next open task (or task N); /spec NAME what you want starts a new spec."
        );
    }
}
