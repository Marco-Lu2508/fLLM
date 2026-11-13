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
}
