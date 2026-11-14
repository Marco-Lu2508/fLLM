//! Is the GPU doing the work? The pure parts of the Studio's `gpucheck.js`: what to show for the
//! `device_fallbacks` of the status, and one result line of the GPU check.
//!
//!  * The amber notice *"Part of this model runs on the CPU"* (Settings > Hardware, the footer) is
//!    built from `device_fallbacks` of `GET /api/studio/status` (the engine's `placement.rs`).
//!  * The *"Test this GPU"* button of every device card runs `POST /api/studio/gpu-check` — a
//!    quantized matrix product on the device and on the CPU — and answers with result lines.

use serde_json::Value;

/// What to show for the `device_fallbacks` object of the status. `show` is false when everything
/// runs where it should (or no GPU is involved); `level` is `cpu` (the whole model), `mostly`
/// (half or more) or `partial` — `fallbackNotice` of `gpucheck.js`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FallbackNotice {
    pub show: bool,
    pub level: String,
    pub title: String,
    pub short: String,
    pub summary: String,
    pub detail: String,
    pub lines: Vec<String>,
    pub footer_title: String,
}

/// The strings beside one device while it is tested and when it answered — `gpuCheckHtml` of
/// `gpucheck.js`.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct GpuResult {
    pub device: String,
    pub status: String,
    pub verdict: String,
}

/// One result line of the check: the text, and `ok` / `warn` / `bad` for its colour —
/// `gpuVerdictLine` of `gpucheck.js`.
#[derive(Debug, Clone, PartialEq)]
pub struct VerdictLine {
    pub text: String,
    pub cls: &'static str,
}

/// What the *Processor check* card shows after `POST /api/studio/selftest-cpu`
/// (`btnRunSimdSelftest` of `listeners-panels.js`): one line over the table and one line under it.
#[derive(Debug, Clone, PartialEq)]
pub struct Selftest {
    /// *Processor check: X of Y checks passed in Z ms, N skipped (…)*
    pub title: String,
    /// The closing sentence of the Studio, with *All correct.* or *Some results deviate.*
    pub note: String,
    /// One row per feature level of the report.
    pub rows: Vec<SelftestRow>,
}

/// One feature level of the self-test: how many checks it has, how many of them passed and failed.
#[derive(Debug, Clone, PartialEq)]
pub struct SelftestRow {
    pub tier: String,
    pub checks: u64,
    pub passed: u64,
    pub failed: u64,
    pub ok: bool,
}

impl Selftest {
    pub fn from_json(v: &Value) -> Self {
        let failed = v.get("failed").and_then(Value::as_u64).unwrap_or(0);
        let checks = v.get("checks").and_then(Value::as_u64).unwrap_or(0);
        let skipped = v.get("skipped").and_then(Value::as_u64).unwrap_or(0);
        let duration = v.get("duration_ms").and_then(Value::as_f64).unwrap_or(0.0);
        let ok_all = v.get("ok").and_then(Value::as_bool).unwrap_or(false) && failed == 0;
        let title = format!(
            "Processor check: {} of {} checks passed in {} ms, {} skipped (feature not supported by this processor)",
            checks.saturating_sub(failed),
            checks,
            duration.round() as i64,
            skipped
        );
        let note = format!(
            "The calculations the model needs are checked against exact results for every processor feature level. {}",
            if ok_all { "All correct." } else { "Some results deviate." }
        );
        let rows = v
            .get("tested_tiers")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|c| SelftestRow {
                tier: c.get("tier").and_then(Value::as_str).unwrap_or("").to_string(),
                checks: c.get("checks").and_then(Value::as_u64).unwrap_or(0),
                passed: c.get("passed").and_then(Value::as_u64).unwrap_or(0),
                failed: c.get("failed").and_then(Value::as_u64).unwrap_or(0),
                ok: c.get("ok").and_then(Value::as_bool).unwrap_or(false),
            })
            .collect();
        Selftest { title, note, rows }
    }
}

/// The devices the check of one card runs over (`runGpuCheck` of `gpucheck.js`): every interface
/// of the physical device behind the button — the button's own id when no group holds it.
pub fn check_members(devices: &[crate::tour::Device], id: &str) -> Vec<String> {
    let filtered: Vec<crate::tour::Device> = devices
        .iter()
        .filter(|d| d.backend != "cpu")
        .cloned()
        .collect();
    for group in crate::npu::merge_devices(&filtered, id) {
        // the Studio compares the ids as they stand (`g.id === id`)
        if group.primary.id == id {
            return group.members.iter().map(|m| m.id.clone()).collect();
        }
    }
    vec![id.to_string()]
}

pub fn gpu_verdict_line(result: &GpuResult) -> VerdictLine {
    let cls = if result.status == "ok" {
        "ok"
    } else if result.status == "slow" {
        "warn"
    } else {
        "bad"
    };
    VerdictLine { text: format!("{}: {}", result.device, result.verdict), cls }
}

/// The pieces of one card's test area: whether the button shows (*Testing ...* while it runs) and
/// the lines of what came back. The Studio skips the button for `cpu` and `npu` devices — they
/// have nothing to test with this route.
pub struct GpuCheckView {
    pub show_button: bool,
    pub label: &'static str,
    pub lines: Vec<VerdictLine>,
}

pub fn gpu_check_view(backend: &str, busy: bool, results: &[GpuResult]) -> GpuCheckView {
    GpuCheckView {
        show_button: backend != "cpu" && backend != "npu",
        label: if busy { "Testing ..." } else { "Test this GPU" },
        lines: results.iter().map(gpu_verdict_line).collect(),
    }
}

/// What the `device_fallbacks` object of the status has to say. Without one — or when nothing
/// falls back — everything runs where it should and there is nothing to show.
pub fn fallback_notice(df: Option<&Value>) -> FallbackNotice {
    let Some(df) = df.filter(|d| d.get("fallback").and_then(Value::as_bool).unwrap_or(false)) else {
        return FallbackNotice::default();
    };
    let text = |key: &str| df.get(key).and_then(Value::as_str).unwrap_or("").to_string();
    let mode = text("mode");
    let device = text("device");
    let backend = text("backend");
    let where_ = if device.is_empty() {
        String::from("the graphics card")
    } else if backend.is_empty() {
        device.clone()
    } else {
        format!("{device} ({backend})")
    };
    let title = match mode.as_str() {
        "cpu" => String::from("The whole model runs on the CPU"),
        "mostly_cpu" => String::from("Weights in GPU memory, compute mostly on the CPU"),
        _ => String::from("Part of this model runs on the CPU"),
    };
    let reasons: Vec<String> = df
        .get("reasons")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|r| {
            format!(
                "{} ({})",
                r.get("reason").and_then(Value::as_str).unwrap_or(""),
                r.get("operations").and_then(Value::as_str).unwrap_or("")
            )
        })
        .collect();
    let ops: Vec<String> = df
        .get("operation_fallbacks")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|o| !o.get("by_design").and_then(Value::as_bool).unwrap_or(false))
        .map(|o| {
            format!(
                "{}: {} ({} calls)",
                o.get("operation").and_then(Value::as_str).unwrap_or(""),
                o.get("reason").and_then(Value::as_str).unwrap_or(""),
                o.get("calls").and_then(Value::as_u64).unwrap_or(0)
            )
        })
        .collect();
    // the headline and the notices are the detail under the summary
    let mut detail_parts: Vec<String> = Vec::new();
    if let Some(headline) = df.get("headline").and_then(Value::as_str).filter(|s| !s.is_empty()) {
        detail_parts.push(headline.to_string());
    }
    for notice in df.get("notices").and_then(Value::as_array).into_iter().flatten() {
        if let Some(s) = notice.as_str().filter(|s| !s.is_empty()) {
            detail_parts.push(s.to_string());
        }
    }
    let detail = detail_parts.join(" ");

    let mut lines: Vec<String> = Vec::new();
    let whole_model = df.get("whole_model_on_cpu").and_then(Value::as_bool).unwrap_or(false);
    let operations_on_cpu = df.get("operations_on_cpu").and_then(Value::as_u64).unwrap_or(0);
    if whole_model {
        lines.push(format!("{where_} is not used for this model."));
    } else if operations_on_cpu > 0 {
        let operations_total = df.get("operations_total").and_then(Value::as_u64).unwrap_or(0);
        let layers_on_device = df.get("layers_on_device").and_then(Value::as_u64).unwrap_or(0);
        lines.push(format!(
            "{operations_on_cpu} of {operations_total} operations (attention or feed-forward blocks of the layers) run on the CPU. \
             {layers_on_device} layers run completely on {where_}."
        ));
    }
    for r in &reasons {
        lines.push(format!("Why: {r}"));
    }
    for o in &ops {
        lines.push(format!("Operation on the CPU: {o}"));
    }

    let summary = {
        let s = text("summary");
        if s.is_empty() { title.clone() } else { s }
    };
    let short = if mode == "cpu" {
        String::from("Model runs on the CPU")
    } else {
        String::from("Part of the model runs on the CPU")
    };
    // the footer stands in one line: title, detail, and where the rest of it lives (the join of
    // the Studio squeezes the spaces its template leaves)
    let base = if detail.is_empty() {
        format!("{title}. Open Settings, Hardware for the details, or run \"fllm-rs doctor --gpu\".")
    } else {
        format!("{title}. {detail} Open Settings, Hardware for the details, or run \"fllm-rs doctor --gpu\".")
    };
    let footer_title = base.split_whitespace().collect::<Vec<_>>().join(" ");
    FallbackNotice {
        show: true,
        level: if mode == "cpu" {
            String::from("cpu")
        } else if mode == "mostly_cpu" {
            String::from("mostly")
        } else {
            String::from("partial")
        },
        title,
        short,
        summary,
        detail,
        lines,
        footer_title,
    }
}

impl GpuResult {
    pub fn from_json(v: &Value) -> Vec<GpuResult> {
        v.get("results")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|r| GpuResult {
                device: r.get("device").and_then(Value::as_str).unwrap_or("").to_string(),
                status: r.get("status").and_then(Value::as_str).unwrap_or("").to_string(),
                verdict: r.get("verdict").and_then(Value::as_str).unwrap_or("").to_string(),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Without fallbacks there is nothing to say, and a false flag says nothing either.
    #[test]
    fn without_fallbacks_there_is_nothing_to_show() {
        assert!(!fallback_notice(None).show);
        let no = json!({"fallback": false, "mode": "cpu"});
        assert!(!fallback_notice(Some(&no)).show, "the flag decides, not the mode");
    }

    /// The three modes are the three titles of the Studio, and `short` distinguishes the whole
    /// model from a part of it.
    #[test]
    fn the_three_modes_are_the_three_titles_of_the_studio() {
        let whole = json!({"fallback": true, "mode": "cpu", "device": "NVIDIA GPU", "backend": "cuda",
            "whole_model_on_cpu": true});
        let n = fallback_notice(Some(&whole));
        assert_eq!(n.title, "The whole model runs on the CPU");
        assert_eq!(n.short, "Model runs on the CPU");
        assert_eq!(n.level, "cpu");
        assert_eq!(n.lines, vec!["NVIDIA GPU (cuda) is not used for this model."]);
        // the footer is the title and the detail in one line (the Studio's template with its join)
        assert_eq!(
            n.footer_title,
            "The whole model runs on the CPU. Open Settings, Hardware for the details, or run \"fllm-rs doctor --gpu\".",
            "{}",
            n.footer_title
        );
        assert!(!n.footer_title.contains("  "), "the footer is squeezed into one line");

        let mostly = json!({"fallback": true, "mode": "mostly_cpu"});
        assert_eq!(
            fallback_notice(Some(&mostly)).title,
            "Weights in GPU memory, compute mostly on the CPU"
        );
        let partial = json!({"fallback": true, "mode": "layer_offload"});
        let n = fallback_notice(Some(&partial));
        assert_eq!(n.title, "Part of this model runs on the CPU");
        assert_eq!(n.level, "partial");
        assert_eq!(n.short, "Part of the model runs on the CPU");
        // without a device the Studio says the plain words
        assert_eq!(n.lines.len(), 0, "no counts and no device: nothing under the title");
    }

    /// The counts, the reasons and the operations are the lines of the Studio; operations that are
    /// there by design are not shown.
    #[test]
    fn the_counts_reasons_and_operations_are_the_lines_of_the_studio() {
        let df = json!({
            "fallback": true, "mode": "layer_offload",
            "device": "GPU", "backend": "vulkan",
            "operations_on_cpu": 12, "operations_total": 48, "layers_on_device": 30,
            "reasons": [{"reason": "no kernel", "operations": "attention"}],
            "operation_fallbacks": [
                {"operation": "rms_norm", "reason": "unsupported", "calls": 4, "by_design": false},
                {"operation": "rope", "reason": "by design", "calls": 9, "by_design": true},
            ],
            "headline": "Some blocks run on the processor.",
            "notices": ["The card has 4 GB."],
            "summary": "Half or more of the work is on the CPU.",
        });
        let n = fallback_notice(Some(&df));
        assert_eq!(
            n.lines,
            vec![
                "12 of 48 operations (attention or feed-forward blocks of the layers) run on the CPU. 30 layers run completely on GPU (vulkan).",
                "Why: no kernel (attention)",
                "Operation on the CPU: rms_norm: unsupported (4 calls)",
            ],
            "the by-design operation stays out"
        );
        assert_eq!(n.summary, "Half or more of the work is on the CPU.");
        assert_eq!(n.detail, "Some blocks run on the processor. The card has 4 GB.");
        assert_eq!(n.level, "partial");
    }

    /// One result line of the check: `ok` is green, `slow` is a warning, everything else bad.
    #[test]
    fn one_result_line_carries_its_colour() {
        let line = gpu_verdict_line(&GpuResult {
            device: "cuda:0".into(),
            status: "ok".into(),
            verdict: "the card does the work (3.1 GB/s)".into(),
        });
        assert_eq!(line.text, "cuda:0: the card does the work (3.1 GB/s)");
        assert_eq!(line.cls, "ok");
        let slow = gpu_verdict_line(&GpuResult { device: "cuda:0".into(), status: "slow".into(), verdict: "half of it".into() });
        assert_eq!(slow.cls, "warn");
        let bad = gpu_verdict_line(&GpuResult { device: "cpu".into(), status: "error".into(), verdict: "The check could not run.".into() });
        assert_eq!(bad.cls, "bad");
    }

    /// The button only exists for devices the route can test, and it says *Testing ...* while it
    /// runs; the lines follow the answers.
    #[test]
    fn the_button_exists_for_testable_devices_only() {
        assert!(!gpu_check_view("cpu", false, &[]).show_button, "the CPU has nothing to test here");
        assert!(!gpu_check_view("npu", false, &[]).show_button, "nor the NPU");
        let view = gpu_check_view("cuda", false, &[]);
        assert!(view.show_button);
        assert_eq!(view.label, "Test this GPU");
        assert_eq!(gpu_check_view("cuda", true, &[]).label, "Testing ...");
        let with = gpu_check_view("cuda", false, &[GpuResult { device: "cuda:0".into(), status: "ok".into(), verdict: "fine".into() }]);
        assert_eq!(with.lines.len(), 1);
        assert_eq!(with.lines[0].text, "cuda:0: fine");
    }

    fn device(id: &str, name: &str, backend: &str, api: &str, bytes: u64) -> crate::tour::Device {
        crate::tour::Device {
            id: id.into(),
            name: name.into(),
            backend: backend.into(),
            api: api.into(),
            dev_type: String::new(),
            total_memory_bytes: bytes,
            free_memory_bytes: 0,
            available: true,
            has_id: !id.is_empty(),
        }
    }

    /// The check of one card runs over every interface of the physical device behind the button,
    /// and over nothing else (`runGpuCheck` of `gpucheck.js`).
    #[test]
    fn the_check_of_a_card_runs_over_its_members() {
        let devices = vec![
            device("cpu0", "Intel Core i5", "cpu", "cpu", 16_000_000_000u64),
            device("cuda:0", "GeForce RTX 4060", "cuda", "cuda", 8_000_000_000u64),
            device("vulkan:1", "GeForce RTX 4060", "vulkan", "vulkan", 8_100_000_000u64),
            device("vulkan:2", "GeForce RTX 4060", "vulkan", "wgpu", 8_100_000_000u64),
            device("npu0", "Intel AI Boost", "npu", "npu", 0),
        ];
        // the same card through three interfaces is one check: the processor is left out, the
        // entries of the device come in the order the report gave them
        assert_eq!(
            check_members(&devices, "cuda:0"),
            vec!["cuda:0", "vulkan:1", "vulkan:2"],
            "every interface of the card is tested"
        );
        // the wanted id decides which group is meant (the Studio compares them as they stand)
        assert_eq!(check_members(&devices, "CUDA:0"), vec!["CUDA:0"], "an id of no group is its own check");
        // an NPU is never merged: its group holds it alone
        assert_eq!(check_members(&devices, "npu0"), vec!["npu0"]);
        // nothing to go on: the button's own id, once
        assert_eq!(check_members(&[], "cuda:0"), vec!["cuda:0"]);
        let no_id = vec![device("", "Nameless", "cuda", "cuda", 1_000u64)];
        assert_eq!(check_members(&no_id, "cuda:0"), vec!["cuda:0"], "an entry without an id is no group");
    }

    /// The title and the closing line of the processor check are the Studio's own words
    /// (`btnRunSimdSelftest` of `listeners-panels.js`).
    #[test]
    fn the_processor_check_says_what_the_studio_says() {
        let report = Selftest::from_json(&json!({
            "ok": true, "checks": 120, "failed": 0, "skipped": 4, "duration_ms": 5123.4,
            "tested_tiers": [
                {"tier": "avx2-fma", "checks": 40, "passed": 40, "failed": 0, "ok": true},
                {"tier": "scalar", "checks": 40, "passed": 39, "failed": 1, "ok": false},
            ],
        }));
        assert_eq!(
            report.title,
            "Processor check: 120 of 120 checks passed in 5123 ms, 4 skipped (feature not supported by this processor)"
        );
        assert!(
            report.note.ends_with(" All correct."),
            "{}",
            report.note
        );
        assert_eq!(report.rows.len(), 2);
        assert_eq!((report.rows[0].tier.as_str(), report.rows[0].ok), ("avx2-fma", true));
        // a row says what the server says about it, not what the totals say
        assert!(!report.rows[1].ok);
        // one failed check, however the server flags itself: the note deviates
        let bad = Selftest::from_json(&json!({"ok": true, "checks": 10, "failed": 2, "duration_ms": 1.4}));
        assert!(bad.title.starts_with("Processor check: 8 of 10 checks passed in 1 ms"), "{}", bad.title);
        assert!(bad.note.ends_with("Some results deviate."), "{}", bad.note);
        // nothing at all: the line still reads
        let empty = Selftest::from_json(&json!({}));
        assert_eq!(empty.rows.len(), 0);
        assert!(empty.title.starts_with("Processor check: 0 of 0"), "{}", empty.title);
    }
}
