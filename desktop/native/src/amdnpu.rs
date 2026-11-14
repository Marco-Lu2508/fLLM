//! The AMD NPU card (Settings > Hardware): fLLM has no kernels for the XDNA NPU of a Ryzen AI
//! processor, FastFlowLM has. The pure heart of the Studio's `amdnpu.js` — what the card says and
//! which of its buttons exist — ported whole and checked against that file itself
//! (`amdnpu_fixtures.js` runs the `@pure` zone of the Studio and produces the cases).

use serde_json::Value;

/// The line under the card's buttons, always (`AMD_NPU_NOTE` of `amdnpu.js`).
pub const AMD_NPU_NOTE: &str = "FastFlowLM is a separate program (its NPU kernels are AMD's, free to use). It has its own model files, so a model here is not the same file as one in your fLLM model list.";

/// One entry of the model list: the tag, how the option reads, and whether it is on disk.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AmdModel {
    pub tag: String,
    pub label: String,
    pub installed: bool,
}

/// One row of the hardware the server found (`hardware` of the API state).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct AmdHardware {
    pub label: String,
    pub problem: Option<String>,
}

/// Where FastFlowLM is fetched from (`flm.install`), shown while it is missing.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct AmdInstall {
    pub text: String,
    pub url: String,
}

/// Everything the card shows, from the API state and the model chosen in the list
/// (`amdNpuModel` of `amdnpu.js`). No drawing here.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AmdNpuCard {
    /// False when there is no AMD NPU and no FastFlowLM on the computer: the card stays away.
    pub visible: bool,
    pub summary: String,
    pub native_text: String,
    pub native_ready: bool,
    pub tone: String,
    pub hardware: Vec<AmdHardware>,
    pub install: Option<AmdInstall>,
    pub problems: Vec<String>,
    pub version: Option<String>,
    pub models: Vec<AmdModel>,
    pub selected: String,
    /// Whether *Run on the NPU* exists. The Studio's chain ends on `picked.installed`, which is
    /// `undefined` when the entry says nothing about being on disk — then the key is not in its
    /// answer at all, so this stays `None` and is left out of the serialised card too.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start: Option<bool>,
    pub stop: bool,
    pub pull: bool,
    pub cancel_pull: bool,
    pub busy: bool,
    pub pull_text: String,
    pub error: Option<String>,
    pub note: String,
}

impl Default for AmdNpuCard {
    fn default() -> Self {
        AmdNpuCard {
            visible: false,
            summary: String::new(),
            native_text: String::new(),
            native_ready: false,
            tone: String::from("muted"),
            hardware: Vec::new(),
            install: None,
            problems: Vec::new(),
            version: None,
            models: Vec::new(),
            selected: String::new(),
            start: Some(false),
            stop: false,
            pull: false,
            cancel_pull: false,
            busy: false,
            pull_text: String::new(),
            error: None,
            note: String::from(AMD_NPU_NOTE),
        }
    }
}

/// What the Studio counts as true: everything but `false`, `null`, `0`, `""` and a missing value.
fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0 && !f.is_nan()),
        Some(Value::String(s)) => !s.is_empty(),
        Some(_) => true,
    }
}

/// How the Studio's list sorts two tags (`a.tag.localeCompare(b.tag)`): by the letters without
/// their case first, and when those are the same the lower case one before the upper.
fn tag_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    let (lower_a, lower_b) = (a.to_lowercase(), b.to_lowercase());
    if lower_a != lower_b {
        return lower_a.cmp(&lower_b);
    }
    // `a`-`z` first, then `A`-`Z`, everything else after them (it only ever breaks a tie
    // between two tags that are equal without their case)
    fn rank(c: char) -> u32 {
        match c {
            'a'..='z' => c as u32 - 'a' as u32,
            'A'..='Z' => 26 + (c as u32 - 'A' as u32),
            other => 1000 + other as u32,
        }
    }
    a.chars().map(rank).cmp(b.chars().map(rank))
}

/// A field of an object as text, empty when it is not a string.
fn text<'a>(v: Option<&'a Value>, key: &str) -> &'a str {
    v.and_then(|o| o.get(key)).and_then(Value::as_str).unwrap_or("")
}

/// The summary line of the card (`amdNpuText` of `amdnpu.js`): what FastFlowLM is doing right now.
pub fn amd_npu_text(s: &Value, serving: &Value, problems: &[String]) -> String {
    // the Studio's own `running` is the model only while it runs, and empty for the two lines
    // that speak of a start or a failure
    let running = if text(Some(serving), "state") == "running" {
        text(Some(serving), "model")
    } else {
        ""
    };
    if text(Some(s), "state") == "flm_missing" {
        return String::from(
            "An AMD NPU was found. fLLM runs models on it through FastFlowLM, which you install yourself (a free download).",
        );
    }
    if text(Some(s), "state") == "needs_attention" {
        return format!(
            "FastFlowLM reports a problem with the NPU setup: {}",
            if problems.is_empty() { String::from("see the details") } else { problems.join("; ") }
        );
    }
    match text(Some(serving), "state") {
        "running" => format!(
            "Running {running} on the NPU. Pick \"AMD NPU (FastFlowLM) · {running}\" in the model list of the chat."
        ),
        // the two lines below take the model from the state itself, not from `running`
        "starting" => format!(
            "Starting {} on the NPU ... (the first start of a model can take a few minutes)",
            text(Some(serving), "model")
        ),
        "failed" => format!("{} did not start.", text(Some(serving), "model")),
        _ => String::from("Ready. Choose a model and run it on the NPU."),
    }
}

/// The status line of fLLM's own (native) XDNA kernels (`amdNpuNativeText`): empty when the build
/// has none of them.
pub fn amd_npu_native_text(n: Option<&Value>) -> String {
    let compiled = n.and_then(|n| n.get("compiled"));
    if !truthy(compiled) {
        return String::new();
    }
    let n = n.unwrap_or(&Value::Null);
    if truthy(n.get("available")) {
        return String::from("Native fLLM kernels: ready. Start the server with FLLM_NPU_HOST=1 and the long prompts of any model run on the NPU, the rest on the processor (or choose NPU under Advanced when you load a model).");
    }
    let reason = text(Some(n), "reason");
    if reason.is_empty() {
        String::from("Native fLLM kernels: not available. FastFlowLM, below, does not need them.")
    } else {
        format!(
            "Native fLLM kernels: not available ({reason}). FastFlowLM, below, does not need them."
        )
    }
}

/// Everything the card shows, from the API state `s` and the model chosen in the list
/// (`amdNpuModel` of `amdnpu.js`). No drawing here.
pub fn amd_npu_model(s: Option<&Value>, chosen: &str) -> AmdNpuCard {
    let Some(s) = s else {
        return AmdNpuCard::default();
    };
    if text(Some(s), "state") == "no_npu" {
        return AmdNpuCard { visible: false, ..AmdNpuCard::default() };
    }
    let flm = s.get("flm").filter(|v| !v.is_null()).unwrap_or(&Value::Null);
    let serving = s.get("serving").filter(|v| !v.is_null()).unwrap_or(&Value::Null);
    let pull = s.get("pull").filter(|v| !v.is_null()).unwrap_or(&Value::Null);
    let problems: Vec<String> = s
        .pointer("/validation/problems")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect())
        .unwrap_or_default();

    // the list, installed first and otherwise by tag: `(b.installed - a.installed) || a.tag.localeCompare(b.tag)`
    let mut entries: Vec<(String, bool, Option<bool>)> = s
        .get("models")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .map(|m| {
                    (
                        m.get("tag").and_then(Value::as_str).unwrap_or("").to_string(),
                        truthy(m.get("installed")),
                        m.get("installed").and_then(Value::as_bool),
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    let mut order: Vec<usize> = (0..entries.len()).collect();
    order.sort_by(|&x, &y| {
        // `b.installed - a.installed`: with both sides known the difference decides, an unknown
        // side makes it NaN and NaN is falsy, so the tags decide (as in the Studio)
        let diff = match (entries[x].2, entries[y].2) {
            (Some(a), Some(b)) => Some(i32::from(b) - i32::from(a)),
            _ => None,
        };
        match diff {
            Some(d) if d != 0 => d.cmp(&0),
            _ => tag_cmp(&entries[x].0, &entries[y].0),
        }
    });
    let sorted: Vec<(String, bool, Option<bool>)> =
        order.iter().map(|&i| entries[i].clone()).collect();
    entries = sorted;

    let serving_state = text(Some(serving), "state");
    let running = if serving_state == "running" { text(Some(serving), "model") } else { "" };
    let starting = if serving_state == "starting" { text(Some(serving), "model") } else { "" };
    let downloading = truthy(pull.get("busy"));

    // which model the list holds: what was chosen, else the running one, else the first that is
    // on disk, else the first of the list
    let tag = if entries.iter().any(|(t, _, _)| t == chosen) {
        chosen.to_string()
    } else if !running.is_empty() {
        running.to_string()
    } else {
        entries
            .iter()
            .find(|(_, installed, _)| *installed)
            .or_else(|| entries.first())
            .map(|(t, _, _)| t.clone())
            .unwrap_or_default()
    };
    // the object of the chosen tag (`picked`), whether it is on disk, and whether its entry says
    // anything about that at all (`picked.installed` may be missing)
    let picked = entries.iter().find(|(t, _, _)| *t == tag);
    let picked_known = picked.is_some();
    let picked_installed = picked.map(|(_, installed, _)| *installed).unwrap_or(false);
    let picked_says = picked.is_some_and(|(_, _, raw)| raw.is_some());

    let state = text(Some(s), "state");
    let ready = state == "ready";
    let mut error = {
        let pull_error = text(Some(pull), "error");
        if !pull_error.is_empty() {
            Some(pull_error.to_string())
        } else if serving_state == "failed" && !text(Some(serving), "error").is_empty() {
            Some(text(Some(serving), "error").to_string())
        } else if !text(Some(s), "error").is_empty() {
            Some(text(Some(s), "error").to_string())
        } else {
            None
        }
    };
    if text(Some(pull), "error") == "cancelled" {
        error = None;
    }

    let models: Vec<AmdModel> = entries
        .iter()
        .map(|(t, installed, _)| AmdModel {
            tag: t.clone(),
            label: if *installed { t.clone() } else { format!("{t} (not downloaded)") },
            installed: *installed,
        })
        .collect();

    let pull_model = text(Some(pull), "model");
    let pull_line = text(Some(pull), "line");
    let pull_text = if downloading {
        format!(
            "Downloading {pull_model}: {}",
            if pull_line.is_empty() { String::from("...") } else { pull_line.to_string() }
        )
    } else if pull_line == "done" && !pull_model.is_empty() {
        format!("{pull_model} downloaded.")
    } else {
        String::new()
    };

    AmdNpuCard {
        visible: true,
        summary: amd_npu_text(s, serving, &problems),
        native_text: amd_npu_native_text(s.get("native")),
        native_ready: truthy(s.pointer("/native/available")),
        tone: if !running.is_empty() {
            String::from("good")
        } else if state == "needs_attention" || error.is_some() {
            String::from("warn")
        } else {
            String::from("muted")
        },
        hardware: s
            .get("hardware")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .map(|h| {
                        let problem = text(Some(h), "problem");
                        AmdHardware {
                            label: text(Some(h), "name").to_string(),
                            // `h.problem || null`
                            problem: (!problem.is_empty()).then(|| problem.to_string()),
                        }
                    })
                    .collect()
            })
            .unwrap_or_default(),
        install: if state == "flm_missing" {
            s.pointer("/flm/install").and_then(|i| {
                if i.is_object() {
                    Some(AmdInstall {
                        text: text(Some(i), "text").to_string(),
                        url: text(Some(i), "url").to_string(),
                    })
                } else {
                    None
                }
            })
        } else {
            None
        },
        problems,
        version: {
            let v = text(Some(flm), "version");
            (!v.is_empty()).then(|| v.to_string())
        },
        models,
        selected: tag.clone(),
        // `ready && !!picked && picked.installed && !downloading && running !== tag && starting !== tag`
        start: if ready && picked_known && !picked_says {
            None
        } else {
            Some(
                ready
                    && picked_installed
                    && !downloading
                    && running != tag
                    && starting != tag,
            )
        },
        stop: !running.is_empty() || !starting.is_empty(),
        pull: ready && picked_known && !picked_installed && !downloading,
        cancel_pull: downloading,
        busy: !starting.is_empty() || downloading,
        pull_text,
        error,
        note: String::from(AMD_NPU_NOTE),
    }
}
