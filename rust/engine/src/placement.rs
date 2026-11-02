//! Where the work of a model really runs: the ledger of everything a GPU backend hands to the CPU.
//!
//! A model "on the GPU" can still compute a part of itself on the host: layers or feed-forward blocks without device kernels, feed-forward
//! weights the memory planner kept in host memory, a whole model a device could not take, single operations of a backend that copies
//! its inputs to the host and back (`warn_fallback_reason`). None of these is an error, so none of them stopped anything; the model just
//! ran slower than the user expected and nobody knew why. This module counts them per run, logs one WARN line when the model is loaded
//! and when the first operation falls back, and hands the numbers to `/api/studio/status`, `/metrics`, the Studio and `fllm-rs serve`.
//!
//! `FLLM_STRICT_DEVICE=1` (or `--strict-device`) turns every such hand-over into an error that names the reason.
//!
//! Unit: a *block* is the attention (or other mixer) half or the feed-forward half of one layer; a model has two per layer.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

static STRICT: AtomicBool = AtomicBool::new(false);

/// Is the strict device mode on (`FLLM_STRICT_DEVICE=1`, `--strict-device`)? A hand-over to the CPU is then an error.
pub fn strict() -> bool {
    STRICT.load(Ordering::Relaxed) || std::env::var("FLLM_STRICT_DEVICE").is_ok_and(|v| !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false"))
}

/// Switches the strict device mode on (`--strict-device`).
pub fn set_strict(on: bool) {
    STRICT.store(on, Ordering::Relaxed);
}

/// The text of the error a strict run ends with.
pub fn strict_error(what: &str) -> String {
    format!("strict device mode (FLLM_STRICT_DEVICE): {what}; unset FLLM_STRICT_DEVICE / --strict-device to let the CPU take that part")
}

/// Where one layer runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerPlace {
    Device,
    /// The mixer is on the device, the feed-forward block runs on the host.
    FfnOnHost,
    /// The feed-forward block is on the device, the (recurrent) mixer runs on the host.
    MixerOnHost,
    Host,
}

#[derive(Clone, Debug, Default)]
struct OpStat {
    reason: String,
    calls: u64,
}

#[derive(Default)]
struct State {
    /// `(device name, backend id)` the model was attached to.
    device: Option<(String, String)>,
    blocks_total: usize,
    host_blocks: usize,
    host_reasons: BTreeMap<String, usize>,
    layers: Vec<LayerPlace>,
    /// Why the whole model runs on the CPU although a device was asked for or found.
    whole_model: Option<String>,
    ops: BTreeMap<String, OpStat>,
    notices: Vec<String>,
    op_warned: bool,
    /// The placement of the main model is recorded; a draft model or a second decoder loaded later does not overwrite it.
    placed: bool,
}

static STATE: Mutex<State> = Mutex::new(State {
    device: None,
    blocks_total: 0,
    host_blocks: 0,
    host_reasons: BTreeMap::new(),
    layers: Vec::new(),
    whole_model: None,
    ops: BTreeMap::new(),
    notices: Vec::new(),
    op_warned: false,
    placed: false,
});

/// Operations that run on the host by design on every backend (sampling a token from the logits is a few microseconds on the CPU): they are
/// listed with their call counts but are neither a warning nor an error in the strict mode.
fn is_minor(op: &str) -> bool {
    matches!(op, "sampling_logits")
}

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn label(device: &(String, String)) -> String {
    format!("GPU {} ({})", device.0, device.1)
}

/// The model is about to be loaded on `device` of `backend`. The first load of a process is the main model; later ones (a draft model for
/// speculation) leave its record alone.
pub fn begin_load(device: &str, backend: &str) {
    let mut s = state();
    if s.placed {
        return;
    }
    *s = State { device: Some((device.to_owned(), backend.to_owned())), ..State::default() };
}

/// Forgets everything (tests).
pub fn reset() {
    *state() = State::default();
}

/// A device was found or asked for but the model runs on the CPU as a whole.
pub fn note_whole_model_on_cpu(device: &str, backend: &str, reason: &str) {
    let mut s = state();
    if s.placed {
        return;
    }
    s.placed = true;
    s.device = Some((device.to_owned(), backend.to_owned()));
    s.whole_model = Some(reason.to_owned());
    drop(s);
    eprintln!("[fllm-rs] WARN: GPU {device} ({backend}): the whole model runs on the CPU: {reason}");
}

/// Something near the fallbacks worth showing in the Studio (a backend that was replaced by another one).
pub fn note_notice(text: &str) {
    let mut s = state();
    if !s.notices.iter().any(|n| n == text) {
        s.notices.push(text.to_owned());
        drop(s);
        eprintln!("[fllm-rs] WARN: {text}");
    }
}

/// The result of the layer placement of one model load: `layers` (one entry per layer of the stage), the blocks that run on the host with
/// their reasons.
pub fn note_layers(layers: Vec<LayerPlace>, host_reasons: BTreeMap<String, usize>) {
    let mut s = state();
    if s.placed {
        return;
    }
    s.placed = true;
    s.blocks_total = layers.len() * 2;
    s.host_blocks = host_reasons.values().sum();
    s.host_reasons = host_reasons;
    s.layers = layers;
    let line = (s.host_blocks > 0).then(|| headline_of(&s));
    drop(s);
    if let Some(Some(line)) = line {
        eprintln!("[fllm-rs] WARN: {line}");
    }
}

/// A backend operation ran on the CPU instead of the device (`warn_fallback_reason` calls this on every use). Logs the summary line once, when
/// the first operation falls back.
pub fn note_op(op: &str, reason: &str) {
    let mut s = state();
    let entry = s.ops.entry(op.to_owned()).or_default();
    entry.calls += 1;
    if entry.reason.is_empty() {
        entry.reason = reason.to_owned();
    }
    let first = !s.op_warned && !is_minor(op);
    s.op_warned |= first;
    let who = s.device.as_ref().map_or_else(|| "GPU".to_owned(), label);
    drop(s);
    if first {
        eprintln!("[fllm-rs] WARN: {who}: the operation '{op}' runs on the CPU: {reason} (counted as device_fallbacks in /api/studio/status; FLLM_STRICT_DEVICE=1 makes it an error)");
    }
}

/// Like [`note_op`] for the places that can refuse: in the strict device mode the hand-over is an error that names the operation and the reason.
pub fn op_fallback(op: &str, reason: &str) -> Result<(), String> {
    note_op(op, reason);
    if strict() && !is_minor(op) {
        return Err(strict_error(&format!("the operation '{op}' cannot run on the device ({reason})")));
    }
    Ok(())
}

fn reasons_text(reasons: &BTreeMap<String, usize>) -> String {
    let mut list: Vec<(&String, &usize)> = reasons.iter().collect();
    list.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    list.iter().map(|(r, n)| format!("{r} ({n})")).collect::<Vec<_>>().join("; ")
}

/// `GPU <name> (<backend>): <N> of <M> operations run on the CPU: <reasons>`
fn headline_of(s: &State) -> Option<String> {
    let device = s.device.as_ref()?;
    if let Some(why) = &s.whole_model {
        return Some(format!("{}: the whole model runs on the CPU: {why}", label(device)));
    }
    if s.host_blocks == 0 {
        return None;
    }
    Some(format!("{}: {} of {} operations run on the CPU: {}", label(device), s.host_blocks, s.blocks_total, reasons_text(&s.host_reasons)))
}

/// Everything that is known about the placement of the loaded model.
#[derive(Clone, Debug)]
pub struct Report {
    pub device: Option<(String, String)>,
    pub blocks_total: usize,
    pub host_blocks: usize,
    pub reasons: Vec<(String, usize)>,
    pub whole_model_reason: Option<String>,
    pub ops: Vec<(String, String, u64)>,
    pub notices: Vec<String>,
    pub layers_device: usize,
    pub layers_ffn_host: usize,
    pub layers_mixer_host: usize,
    pub layers_host: usize,
}

pub fn report() -> Report {
    let s = state();
    let mut reasons: Vec<(String, usize)> = s.host_reasons.iter().map(|(r, n)| (r.clone(), *n)).collect();
    reasons.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    Report {
        device: s.device.clone(),
        blocks_total: s.blocks_total,
        host_blocks: s.host_blocks,
        reasons,
        whole_model_reason: s.whole_model.clone(),
        ops: s.ops.iter().map(|(o, st)| (o.clone(), st.reason.clone(), st.calls)).collect(),
        notices: s.notices.clone(),
        layers_device: s.layers.iter().filter(|l| **l == LayerPlace::Device).count(),
        layers_ffn_host: s.layers.iter().filter(|l| **l == LayerPlace::FfnOnHost).count(),
        layers_mixer_host: s.layers.iter().filter(|l| **l == LayerPlace::MixerOnHost).count(),
        layers_host: s.layers.iter().filter(|l| **l == LayerPlace::Host).count(),
    }
}

impl Report {
    /// `device` (nothing runs on the CPU that should not), `partial` (a small part), `mostly_cpu` (half or more of the model: the weights
    /// sit in device memory, the compute runs on the CPU), `cpu` (the whole model), `none` (no accelerator involved).
    pub fn mode(&self) -> &'static str {
        if self.device.is_none() {
            "none"
        } else if self.whole_model_reason.is_some() {
            "cpu"
        } else if self.blocks_total > 0 && self.host_blocks * 2 >= self.blocks_total {
            "mostly_cpu"
        } else if self.host_blocks > 0 || self.ops.iter().any(|o| !is_minor(&o.0)) {
            "partial"
        } else {
            "device"
        }
    }

    /// Does anything run on the CPU that was meant for the device?
    pub fn any(&self) -> bool {
        matches!(self.mode(), "partial" | "mostly_cpu" | "cpu")
    }

    /// The one-line summary, `None` when everything runs on the device.
    pub fn headline(&self) -> Option<String> {
        let device = self.device.as_ref()?;
        if let Some(why) = &self.whole_model_reason {
            return Some(format!("{}: the whole model runs on the CPU: {why}", label(device)));
        }
        if self.host_blocks > 0 {
            let reasons: BTreeMap<String, usize> = self.reasons.iter().cloned().collect();
            return Some(format!("{}: {} of {} operations run on the CPU: {}", label(device), self.host_blocks, self.blocks_total, reasons_text(&reasons)));
        }
        let major: Vec<&(String, String, u64)> = self.ops.iter().filter(|o| !is_minor(&o.0)).collect();
        if let Some((op, reason, calls)) = major.first() {
            let more = major.len() - 1;
            return Some(format!(
                "{}: {} operation kind(s) run on the CPU, first '{op}' ({calls} calls): {reason}{}",
                label(device),
                major.len(),
                if more > 0 { format!(" (+{more} more)") } else { String::new() }
            ));
        }
        None
    }

    /// What the user should read in the Studio: the mode in plain words.
    pub fn plain(&self) -> Option<String> {
        let what = match self.mode() {
            "cpu" => "The whole model runs on the CPU.".to_owned(),
            "mostly_cpu" => "Weights in device memory, compute mostly on the CPU: more than half of the model runs on the CPU, so it is much slower than the GPU can be.".to_owned(),
            "partial" => "Part of this model runs on the CPU.".to_owned(),
            _ => return None,
        };
        Some(what)
    }

    pub fn to_json(&self) -> serde_json::Value {
        let reason_list: Vec<serde_json::Value> = self.reasons.iter().map(|(r, n)| serde_json::json!({ "reason": r, "operations": n })).collect();
        let ops: Vec<serde_json::Value> = self.ops.iter().map(|(o, r, c)| serde_json::json!({ "operation": o, "reason": r, "calls": c, "by_design": is_minor(o) })).collect();
        serde_json::json!({
            "mode": self.mode(),
            "fallback": self.any(),
            "device": self.device.as_ref().map(|d| d.0.clone()),
            "backend": self.device.as_ref().map(|d| d.1.clone()),
            "operations_total": self.blocks_total,
            "operations_on_cpu": self.host_blocks,
            "layers_on_device": self.layers_device,
            "layers_feed_forward_on_cpu": self.layers_ffn_host,
            "layers_mixer_on_cpu": self.layers_mixer_host,
            "layers_on_cpu": self.layers_host,
            "whole_model_on_cpu": self.whole_model_reason.is_some(),
            "reasons": reason_list,
            "operation_fallbacks": ops,
            "operation_fallback_calls": self.ops.iter().map(|o| o.2).sum::<u64>(),
            "notices": self.notices,
            "summary": self.plain(),
            "headline": self.headline(),
            "strict": strict(),
        })
    }

    pub fn prometheus(&self) -> String {
        let mut out = String::new();
        out.push_str("# HELP fllm_device_fallback_operations Layer blocks (attention or feed-forward half of a layer) that run on the CPU although a GPU was chosen\n# TYPE fllm_device_fallback_operations gauge\n");
        out.push_str(&format!("fllm_device_fallback_operations {}\n", self.host_blocks));
        out.push_str("# HELP fllm_device_fallback_whole_model 1 when the whole model runs on the CPU although a GPU was chosen or found\n# TYPE fllm_device_fallback_whole_model gauge\n");
        out.push_str(&format!("fllm_device_fallback_whole_model {}\n", u8::from(self.whole_model_reason.is_some())));
        out.push_str("# HELP fllm_device_fallback_calls_total Calls of backend operations that ran on the CPU instead of the device\n# TYPE fllm_device_fallback_calls_total counter\n");
        for (op, _, calls) in &self.ops {
            out.push_str(&format!("fllm_device_fallback_calls_total{{operation=\"{}\"}} {calls}\n", op.replace('"', "'")));
        }
        out
    }
}

/// The per-layer placement as lines for the start of `fllm-rs serve`: ranges of layers with the same place.
pub fn layer_summary() -> Vec<String> {
    let s = state();
    let mut lines = Vec::new();
    let Some(device) = s.device.clone() else {
        return lines;
    };
    if let Some(why) = &s.whole_model {
        lines.push(format!("placement: every layer runs on the CPU ({why})"));
        return lines;
    }
    let mut start = 0usize;
    while start < s.layers.len() {
        let place = s.layers[start];
        let mut end = start;
        while end + 1 < s.layers.len() && s.layers[end + 1] == place {
            end += 1;
        }
        let range = if start == end { format!("layer {start}") } else { format!("layers {start}-{end}") };
        let what = match place {
            LayerPlace::Device => format!("on the GPU ({})", device.1),
            LayerPlace::FfnOnHost => "attention on the GPU, feed-forward on the CPU".to_owned(),
            LayerPlace::MixerOnHost => "recurrent mixer on the CPU, feed-forward on the GPU".to_owned(),
            LayerPlace::Host => "on the CPU".to_owned(),
        };
        lines.push(format!("placement: {range}: {what}"));
        start = end + 1;
    }
    lines
}

/// The expected speed class of the placement (the measured speed of a device is what `fllm-rs doctor --gpu` prints).
pub fn speed_class() -> &'static str {
    match report().mode() {
        "device" => "GPU speed (every layer on the device)",
        "partial" => "between CPU and GPU speed (part of the model on the CPU)",
        "mostly_cpu" => "close to CPU speed (most of the compute is on the CPU)",
        "cpu" => "CPU speed",
        _ => "CPU speed (no accelerator)",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn a_model_on_the_device_reports_nothing() {
        let _g = LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();
        begin_load("Test GPU", "cuda");
        note_layers(vec![LayerPlace::Device; 4], BTreeMap::new());
        let r = report();
        assert_eq!(r.mode(), "device");
        assert!(r.headline().is_none() && !r.any());
        assert_eq!(layer_summary(), vec!["placement: layers 0-3: on the GPU (cuda)".to_owned()]);
    }

    #[test]
    fn the_headline_names_the_counts_and_reasons() {
        let _g = LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();
        begin_load("Test GPU", "cuda");
        let mut reasons = BTreeMap::new();
        reasons.insert("not enough device memory".to_owned(), 5usize);
        note_layers(vec![LayerPlace::Device, LayerPlace::Device, LayerPlace::FfnOnHost, LayerPlace::FfnOnHost, LayerPlace::Host], reasons);
        let r = report();
        assert_eq!(r.headline().unwrap(), "GPU Test GPU (cuda): 5 of 10 operations run on the CPU: not enough device memory (5)");
        assert_eq!(r.mode(), "mostly_cpu");
        assert!(r.plain().unwrap().contains("compute mostly on the CPU"));
        let j = r.to_json();
        assert_eq!(j["operations_on_cpu"], 5);
        assert_eq!(j["layers_feed_forward_on_cpu"], 2);
        assert_eq!(layer_summary().len(), 3);
    }

    #[test]
    fn operations_are_counted_per_call_and_a_whole_model_is_cpu() {
        let _g = LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();
        begin_load("Test GPU", "vulkan");
        note_layers(vec![LayerPlace::Device; 2], BTreeMap::new());
        note_op("moe_route", "no kernel");
        note_op("moe_route", "no kernel");
        let r = report();
        assert_eq!(r.mode(), "partial");
        assert_eq!(r.ops, vec![("moe_route".to_owned(), "no kernel".to_owned(), 2)]);
        assert!(r.prometheus().contains("fllm_device_fallback_calls_total{operation=\"moe_route\"} 2"));
        note_op("sampling_logits", "host sampling");
        assert_eq!(report().ops.len(), 2);
        assert_eq!(report().mode(), "partial");
        reset();
        begin_load("Test GPU", "vulkan");
        note_whole_model_on_cpu("Test GPU", "vulkan", "no kernels");
        assert_eq!(report().mode(), "cpu");
        assert!(report().headline().unwrap().contains("the whole model runs on the CPU"));
    }
}
