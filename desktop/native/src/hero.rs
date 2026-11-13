//! The start screen: what the engine is doing right now — the pure things of the hero block of the
//! Studio (`panels.js`): the title of a model file, the plan of what to offer when nothing is
//! loaded, and the points of the two-minute trace line.
//!
//! The hero of the Studio shows the heading of the moment — *No model loaded*, the name of the
//! model, *Ready* — with the facts under it (parameters, quantization, context), the buttons of
//! what can be done about it (`startPlan`) and the line of tokens per second of the last two
//! minutes. `startPlan` counts every model that is on the computer (downloads, files the user
//! added, files found in the usual places); the one used last is first, then the recommended one
//! when it is on disk, then the largest that fits into the free memory.

use crate::models::{CatalogEntry, Overview};

/// The name of a model file for a person: the file name without folders and `.gguf` (an Ollama
/// model keeps its `name:tag`) — `modelFileTitle` of `panels.js`.
pub fn model_file_title(text: &str) -> String {
    let last = text.rsplit(['/', '\\']).next().unwrap_or("");
    let name = if last.len() >= 5 && last[last.len() - 5..].eq_ignore_ascii_case(".gguf") {
        &last[..last.len() - 5]
    } else {
        last
    };
    name.to_string()
}

/// One model that is on this computer: what `models/load` takes, what people see, how big it is,
/// and where it came from (`downloaded`, `added`, or the source that found it).
#[derive(Debug, Clone, PartialEq)]
pub struct PlanEntry {
    pub reference: String,
    pub name: String,
    /// The file name people see (`modelFileTitle` of `name`).
    pub title: String,
    pub size: u64,
    pub where_: String,
    /// The one that was used last.
    pub last: bool,
    /// Fits into the free memory (with 15 percent of room, as the Studio does).
    pub fits: bool,
}

/// What the start screen offers when nothing is loaded: `load` with the first model and the
/// others beside it, or `download` with the recommended one — `startPlan` of `panels.js`.
#[derive(Debug, Clone, PartialEq)]
pub struct StartPlan {
    pub mode: &'static str,
    pub primary: Option<PlanEntry>,
    pub others: Vec<PlanEntry>,
    pub download: Option<CatalogEntry>,
}

/// What is on this computer, what the hub recommends and what was used last, in the order of the
/// Studio: the model used last first, then the recommended one when it is on disk, then the
/// largest that fits into the free memory; every one counts once.
pub fn start_plan(overview: &Overview, last_model: &str) -> StartPlan {
    let free = overview.memory;
    let catalog = &overview.catalog;
    let recommended = overview.recommended.first().map(String::as_str).unwrap_or("");
    let rec = catalog.iter().find(|c| c.id == recommended).cloned();
    let rec_ref = rec.as_ref().and_then(|c| c.installed_ref.clone()).unwrap_or_default();
    let last = last_model;
    let last_title = model_file_title(last).to_lowercase();
    let last_entry = catalog.iter().find(|c| c.id == last);
    let last_ref = last_entry.and_then(|c| c.installed_ref.clone()).unwrap_or_default();

    let mut seen: Vec<String> = Vec::new();
    let mut all: Vec<PlanEntry> = Vec::new();
    let add = |reference: &str, name: &str, size: u64, where_: String, all: &mut Vec<PlanEntry>,
                   seen: &mut Vec<String>| {
        let key = reference.to_lowercase();
        if reference.is_empty() || seen.contains(&key) {
            return;
        }
        seen.push(key);
        all.push(PlanEntry {
            reference: reference.into(),
            name: name.into(),
            title: model_file_title(name),
            size,
            where_,
            last: false,
            fits: false,
        });
    };
    for i in &overview.installed {
        // the plan shows what the Studio's add() is given for each list (the parse decides it)
        add(
            &i.reference,
            &i.shown,
            i.bytes,
            i.where_.clone(),
            &mut all,
            &mut seen,
        );
    }

    let is_last = |m: &PlanEntry| {
        !last.is_empty()
            && (m.reference == last || m.reference == last_ref || m.title.to_lowercase() == last_title)
    };
    let fits = |m: &PlanEntry| free == 0 || m.size == 0 || (m.size as f64) * 1.15 < free as f64;
    let rank = |m: &PlanEntry| {
        if is_last(m) {
            0u8
        } else if !rec_ref.is_empty() && m.reference == rec_ref {
            1
        } else if fits(m) {
            2
        } else {
            3
        }
    };
    all.sort_by(|x, y| {
        rank(x)
            .cmp(&rank(y))
            .then(y.size.cmp(&x.size))
            .then(x.title.cmp(&y.title))
    });

    if !all.is_empty() {
        // every model knows its two little words for the view (the Studio puts them on the first
        // one in its own object; the view shows them beside the others as well)
        for entry in &mut all {
            let last_flag = is_last(entry);
            let fits_flag = fits(entry);
            entry.last = last_flag;
            entry.fits = fits_flag;
        }
        let mut iter = all.into_iter();
        let primary = iter.next().expect("the list is not empty");
        let others = iter.collect();
        StartPlan { mode: "load", primary: Some(primary), others, download: None }
    } else {
        StartPlan { mode: "download", primary: None, others: Vec::new(), download: rec }
    }
}

/// The facts line under the heading: `3 billion parameters, Q4_K_M, 4096 tokens of context` — the
/// `updateHero` of `panels.js` in one piece (the part after the parameter size, which is the
/// engine's own `parameter_size`).
pub fn hero_facts(parameters: u64, parameter_size: &str, quantization: &str, max_context: u64) -> String {
    let mut out = String::new();
    if parameters > 0 {
        let (value, unit) = if parameters.is_multiple_of(1_000_000_000) {
            (parameters / 1_000_000_000, "billion")
        } else if parameters.is_multiple_of(1_000_000) {
            (parameters / 1_000_000, "million")
        } else {
            (parameters, "parameters")
        };
        if unit == "parameters" {
            out.push_str(&format!("{value} parameters"));
        } else {
            out.push_str(&format!("{value} {unit} parameters"));
        }
    }
    let mut tail = Vec::new();
    if !parameter_size.is_empty() {
        tail.push(parameter_size.to_string());
    }
    if !quantization.is_empty() {
        tail.push(quantization.to_string());
    }
    if max_context > 0 {
        tail.push(format!("{max_context} tokens of context"));
    }
    if !tail.is_empty() {
        if !out.is_empty() {
            out.push_str(", ");
        }
        out.push_str(&tail.join(", "));
    }
    out
}

/// The line of tokens per second: the samples of the last two minutes onto the polyline of the
/// Studio — `step` of 600 by 59, the newest samples on the right (the line grows from there), the
/// line always beginning at (0, 63) — `updateHero` of `panels.js`.
pub fn trace_line(samples: &[f64], size: usize) -> String {
    let top = 10.0f64.max(samples.iter().cloned().fold(0.0f64, f64::max));
    let step = 600.0 / (size.saturating_sub(1)).max(1) as f64;
    let offset = size.saturating_sub(samples.len());
    let mut points = Vec::new();
    for (i, s) in samples.iter().enumerate() {
        let x = (offset + i) as f64 * step;
        let y = 63.0 - (s / top).clamp(0.0, 1.0) * 58.0;
        points.push(format!("{x:.1},{y:.1}"));
    }
    // the Studio joins the string itself: the point at the left, then the samples
    format!("0,63 {}", points.join(" "))
}

/// The rate of the trace: tokens per second between two moments of the server's own clock — the
/// first one is zero, an older clock or a reset token count stays zero — `updateHero`.
pub fn trace_rate(tokens: u64, uptime: f64, last_tokens: Option<u64>, last_uptime: Option<f64>) -> f64 {
    match (last_tokens, last_uptime) {
        (Some(last), Some(at)) if uptime > at && tokens >= last => {
            (tokens - last) as f64 / (uptime - at)
        }
        _ => 0.0,
    }
}

/// How big a file is for a person: `1.0 GB` from a billion, `340 MB` below, nothing for zero —
/// `sizeText` of `panels.js`.
pub fn size_text(bytes: u64) -> String {
    if bytes == 0 {
        String::new()
    } else if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1e9)
    } else {
        format!("{} MB", (bytes as f64 / 1e6).round() as u64)
    }
}

/// The little words under a button: what it weighs, the one word of where it stands (`used last`
/// instead of the source when it was used last), and the warning only when it may not fit —
/// the `hero-start-meta` spans of `renderHeroStart`.
pub fn meta_line(size: u64, where_: &str, last: bool, fits: bool) -> String {
    let mut parts = Vec::new();
    let size_text = size_text(size);
    if !size_text.is_empty() {
        parts.push(size_text);
    }
    let word = if last { "used last" } else { where_ };
    if !word.is_empty() {
        parts.push(word.to_string());
    }
    let mut out = parts.join(", ");
    if !fits {
        if !out.is_empty() {
            out.push_str(", ");
        }
        out.push_str("may not fit into free memory");
    }
    out
}

/// The meta line of the download: name, quantization and weight in one line, with the warning
/// only when the variant does not fit into memory — `renderHeroStart` of `panels.js`.
pub fn download_meta(name: &str, quant: &str, bytes: u64, fits: bool) -> String {
    let mut parts = Vec::new();
    for part in [name, quant, &size_text(bytes)] {
        if !part.is_empty() {
            parts.push(part.to_string());
        }
    }
    let mut out = parts.join(", ");
    if !fits {
        if !out.is_empty() {
            out.push_str(", ");
        }
        out.push_str("may not fit into free memory");
    }
    out
}

/// What stands beside the line: the rate while a slot is running (one decimal), zero when the
/// model waits for a question — `heroTps` and the label of `updateHero`.
pub fn trace_text(rate: f64, busy: bool) -> (String, &'static str) {
    if busy {
        (format!("{rate:.1}"), "tokens per second, the model is answering")
    } else {
        (String::from("0"), "tokens per second, the model is waiting for a question")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{CatalogEntry, Installed, Variant};

    fn entry(id: &str, ref_: &str) -> CatalogEntry {
        CatalogEntry {
            id: id.into(),
            display_name: id.into(),
            default_quant: "Q4_K_M".into(),
            parameters_b: 3.0,
            context: 4096,
            vision: false,
            fits: true,
            installed_ref: Some(ref_.into()),
            variants: vec![Variant { quant: "Q4_K_M".into(), bytes: 2_000_000_000, fits: true }],
        }
    }

    fn installed(name: &str, reference: &str, bytes: u64, where_: &str) -> Installed {
        Installed {
            name: name.into(),
            reference: reference.into(),
            bytes,
            where_: where_.into(),
            shown: reference.into(),
        }
    }

    /// The little words under the buttons are the ones of the Studio: weight in GB or MB, one
    /// word of where it stands, and the warning only when it may not fit.
    #[test]
    fn the_meta_lines_are_the_ones_of_the_studio() {
        assert_eq!(size_text(0), "");
        assert_eq!(size_text(2_500_000_000), "2.5 GB");
        assert_eq!(size_text(340_000_000), "340 MB");
        assert_eq!(meta_line(1_000_000_000, "downloaded", false, true), "1.0 GB, downloaded");
        assert_eq!(meta_line(1_000_000_000, "downloaded", true, true), "1.0 GB, used last");
        assert_eq!(
            meta_line(4_000_000_000, "added", false, false),
            "4.0 GB, added, may not fit into free memory"
        );
        assert_eq!(download_meta("Qwen3 4B", "Q4_K_M", 2_500_000_000, true), "Qwen3 4B, Q4_K_M, 2.5 GB");
        assert_eq!(
            download_meta("Qwen3 4B", "Q4_K_M", 2_500_000_000, false),
            "Qwen3 4B, Q4_K_M, 2.5 GB, may not fit into free memory"
        );
    }

    /// The title of a file is the file name without folders and without `.gguf`.
    #[test]
    fn the_title_of_a_file_is_the_file_name_without_gguf() {
        assert_eq!(model_file_title("/models/qwen/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
                   "Qwen3-4B-Instruct-2507-Q4_K_M");
        assert_eq!(model_file_title("C:\\Users\\a\\qwen.gguf"), "qwen");
        assert_eq!(model_file_title("qwen3.5:4b"), "qwen3.5:4b", "an Ollama name keeps its tag");
        assert_eq!(model_file_title(""), "");
    }

    /// Nothing on disk: the plan is a download of the recommended model.
    #[test]
    fn without_a_model_on_disk_the_plan_is_the_download() {
        let overview = Overview {
            loaded: None,
            loaded_detail: String::new(),
            memory: 8_000_000_000,
            installed: vec![],
            catalog: vec![entry("qwen3-4b", "")],
            recommended: vec!["qwen3-4b".into()],
        };
        let plan = start_plan(&overview, "");
        assert_eq!(plan.mode, "download");
        assert!(plan.primary.is_none() && plan.others.is_empty());
        assert_eq!(plan.download.as_ref().unwrap().id, "qwen3-4b");
    }

    /// The model used last is first, the recommended one beside it, and a model that does not fit
    /// into the memory ranks last.
    #[test]
    fn the_model_used_last_is_first_and_the_big_one_that_does_not_fit_is_last() {
        let overview = Overview {
            loaded: None,
            loaded_detail: String::new(),
            memory: 3_000_000_000, // 15 percent of room: 2 GB fits, 4 GB does not
            installed: vec![
                installed("big.gguf", "/m/big.gguf", 4_000_000_000, "downloaded"),
                installed("used.gguf", "/m/used.gguf", 1_000_000_000, "added"),
                installed("rec.gguf", "/m/rec.gguf", 2_000_000_000, "found"),
            ],
            catalog: vec![entry("used", "/m/used.gguf"), entry("rec", "")],
            recommended: vec!["rec".into()],
        };
        let plan = start_plan(&overview, "/m/used.gguf");
        assert_eq!(plan.mode, "load");
        let primary = plan.primary.as_ref().unwrap();
        assert_eq!(primary.title, "used", "the model used last comes first");
        assert!(primary.last && primary.fits);
        assert_eq!(plan.others.len(), 2);
        assert_eq!(plan.others[1].title, "big", "the one that does not fit is last");
        assert!(!plan.others[1].fits);
        assert_eq!(plan.others[0].title, "rec", "the recommended one fits");
    }

    /// The same model counts once — the reference decides, in lower case (the Set of the Studio).
    #[test]
    fn a_model_that_is_on_the_disk_twice_counts_once() {
        let overview = Overview {
            loaded: None,
            loaded_detail: String::new(),
            memory: 0,
            installed: vec![
                installed("a.gguf", "/M/A.GGUF", 1_000_000_000, "added"),
                installed("a.gguf", "/m/a.gguf", 999, "downloaded"),
            ],
            catalog: vec![],
            recommended: vec![],
        };
        let plan = start_plan(&overview, "");
        assert_eq!(plan.others.len(), 0, "the second one is the same model");
        assert_eq!(plan.primary.as_ref().unwrap().reference, "/M/A.GGUF");
    }

    /// The facts line is the one of the Studio: parameters with their unit, then the size of the
    /// parameters, the quantization and the context.
    #[test]
    fn the_facts_line_is_the_one_of_the_studio() {
        assert_eq!(
            hero_facts(3_000_000_000, "3B", "Q4_K_M", 32768),
            "3 billion parameters, 3B, Q4_K_M, 32768 tokens of context"
        );
        assert_eq!(hero_facts(0, "", "", 0), "", "without a model there are no facts");
    }

    /// The rate of the trace is the tokens between two moments of the server's own clock, and the
    /// line is the one of the Studio: it begins at (0, 63), the samples stand on the right.
    #[test]
    fn the_rate_and_the_line_are_the_ones_of_the_studio() {
        assert_eq!(trace_rate(100, 10.0, Some(70), Some(8.0)), 15.0, "30 tokens in two seconds");
        assert_eq!(trace_rate(100, 10.0, None, None), 0.0, "the first moment is zero");
        assert_eq!(trace_rate(50, 8.0, Some(70), Some(8.0)), 0.0, "a reset count stays zero");
        assert_eq!(trace_rate(100, 8.0, Some(70), Some(8.0)), 0.0, "an older clock stays zero");

        let line = trace_line(&[], 60);
        assert_eq!(line, "0,63 ", "without samples the line is the point at the left");
        let line = trace_line(&[10.0, 5.0], 60);
        assert!(line.starts_with("0,63 "), "the line begins at the bottom left: {:?}", &line[..10]);
        // 60 minus the two samples on the left, step 600/59: ten is the top (y 5.0), half of it y 34.0
        assert_eq!(line, "0,63 589.8,5.0 600.0,34.0", "the samples stand on the right: {line}");
        let (text, label) = trace_text(12.34, true);
        assert_eq!(text, "12.3", "one decimal while the model answers");
        assert_eq!(label, "tokens per second, the model is answering");
        let (text, label) = trace_text(12.34, false);
        assert_eq!(text, "0", "zero while it waits");
        assert_eq!(label, "tokens per second, the model is waiting for a question");
    }
}
