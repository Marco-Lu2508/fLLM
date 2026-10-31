//! What a model can do about thinking, and the thinking budget.
//!
//! * [`ThinkingCaps::detect`] reads the chat template, the reasoning family the server picked and the architecture and says
//!   whether the model can think at all and what can be controlled: nothing (`none`), an on/off switch (`toggle`), an effort
//!   level handed to the template (`effort`), or nothing because the model always thinks (`always`).
//! * [`BudgetMachine`] is the decision logic of "budget forcing": once the reasoning part has produced the budget's number of
//!   tokens and is still open, the server ends it itself. It is a pure state machine; the server feeds it one token at a time
//!   and does the forcing (see `ReasoningRx` in `server.rs`).
use crate::reasoning::Family;
use serde_json::{json, Value};

/// Budget of the `low` effort level, in reasoning tokens.
pub const BUDGET_LOW: usize = 512;
/// Budget of the `medium` effort level, in reasoning tokens.
pub const BUDGET_MEDIUM: usize = 2048;

/// What the user can control about the thinking of a model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Control {
    /// The model has no thinking mode.
    None,
    /// The template has a switch (`enable_thinking`, `thinking`); levels are budgets enforced by the server.
    Toggle,
    /// The template takes an effort level (`reasoning_effort`); the model itself scales its thinking.
    Effort,
    /// The model always thinks; there is nothing to switch.
    Always,
}

impl Control {
    pub fn name(self) -> &'static str {
        match self {
            Control::None => "none",
            Control::Toggle => "toggle",
            Control::Effort => "effort",
            Control::Always => "always",
        }
    }
}

/// Thinking facts of the loaded model, reported as `capabilities.thinking` by the status route.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThinkingCaps {
    pub control: Control,
    /// Template variable the switch or level goes into (`enable_thinking`, `thinking`, `reasoning_effort`); empty without one.
    pub variable: &'static str,
    /// Levels a client can choose, from the least to the most thinking.
    pub levels: Vec<&'static str>,
    /// The level the model uses when the request says nothing.
    pub default: &'static str,
}

/// `true` when the template uses `name` as a variable: an identifier of exactly that name that is neither a field
/// (`message.thinking`) nor inside a string.
fn uses_variable(template: &str, name: &str) -> bool {
    let bytes = template.as_bytes();
    let mut from = 0;
    while let Some(at) = template[from..].find(name) {
        let start = from + at;
        let end = start + name.len();
        let before = start.checked_sub(1).map(|i| bytes[i]);
        let after = bytes.get(end).copied();
        let ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
        let blocked_before = before.is_some_and(|b| ident(b) || matches!(b, b'.' | b'\'' | b'"' | b'/' | b'<' | b'|' | b':'));
        let blocked_after = after.is_some_and(|b| ident(b) || matches!(b, b'\'' | b'"'));
        if !blocked_before && !blocked_after {
            return true;
        }
        from = end;
    }
    false
}

/// The template thinks unless told not to (`enable_thinking is defined and enable_thinking is false` opens an empty block).
fn default_on(template: &str, variable: &str) -> bool {
    [
        format!("{variable} is defined and {variable} is false"),
        format!("{variable} is false"),
        format!("{variable} == false"),
        format!("{variable} is not true"),
        format!("not {variable}"),
    ]
    .iter()
    .any(|pattern| template.contains(pattern.as_str()))
        // a template that does not look at the variable when it is missing, but sets it to true first, thinks by default too
        || template.contains(&format!("set {variable} = true"))
}

/// `true` when the template itself puts the model into a reasoning block (a template without a switch).
///
/// Almost every Qwen3 template mentions `<think>` and `</think>` to show earlier reasoning in the history, including the
/// non-thinking Instruct-2507 releases, so the mention alone proves nothing. What makes a model think all the time is a
/// generation prompt that opens the block (`<|im_start|>assistant\n<think>\n`): the text after the last
/// `add_generation_prompt` decides. A template without that variable falls back to any mention of a reasoning tag.
fn writes_reasoning_block(template: &str) -> bool {
    const TAGS: [&str; 3] = ["<think", "</think", "<|channel>thought"];
    let any = |text: &str| TAGS.iter().any(|tag| text.contains(tag));
    match template.rfind("add_generation_prompt") {
        Some(at) => {
            let tail = &template[at..];
            // an `if` before the generation prompt may also open with a tag of its own; only the part after it counts
            tail.contains("<think") || tail.contains("<|channel>thought")
        }
        None => any(template),
    }
}

impl ThinkingCaps {
    fn none() -> Self {
        Self { control: Control::None, variable: "", levels: Vec::new(), default: "off" }
    }

    pub fn supported(&self) -> bool {
        self.control != Control::None
    }

    /// Reads the capabilities from the chat template, the reasoning family and the GGUF architecture.
    pub fn detect(chat_template: &str, family: Family, arch: &str) -> Self {
        let arch = arch.to_ascii_lowercase();
        let has_toggle = uses_variable(chat_template, "enable_thinking");
        let has_thinking = uses_variable(chat_template, "thinking");
        let has_effort = uses_variable(chat_template, "reasoning_effort");
        let harmony = family == Family::GptOss || matches!(arch.as_str(), "gpt-oss" | "gptoss" | "gpt_oss");
        // the template may use `enable_thinking` for a switch and `reasoning_effort` for a level; the switch decides
        if harmony || (has_effort && !has_toggle && !has_thinking) {
            return Self { control: Control::Effort, variable: "reasoning_effort", levels: vec!["low", "medium", "high"], default: "medium" };
        }
        if has_toggle || has_thinking {
            let variable = if has_toggle { "enable_thinking" } else { "thinking" };
            let default = if default_on(chat_template, variable) { "high" } else { "off" };
            return Self { control: Control::Toggle, variable, levels: vec!["off", "low", "medium", "high"], default };
        }
        // No switch. A model whose template writes or expects a reasoning block thinks all the time; a family that was named
        // explicitly (`--reasoning-parser`) or found by a template marker other than `<think>` counts as one.
        let block = writes_reasoning_block(chat_template);
        let marked = !matches!(family, Family::None | Family::Think);
        if block || marked {
            return Self { control: Control::Always, variable: "", levels: vec!["high"], default: "high" };
        }
        Self::none()
    }

    pub fn to_json(&self) -> Value {
        json!({
            "supported": self.supported(),
            "control": self.control.name(),
            "variable": self.variable,
            "levels": self.levels,
            "default": self.default,
        })
    }
}

/// The reasoning budget a `reasoning_effort` level stands for on a model with a thinking switch: `None` is unlimited.
pub fn budget_for_effort(effort: &str) -> Option<usize> {
    match effort.trim().to_ascii_lowercase().as_str() {
        "low" | "minimal" => Some(BUDGET_LOW),
        "medium" => Some(BUDGET_MEDIUM),
        _ => None,
    }
}

/// Whether `reasoning_effort` asks for no thinking at all.
pub fn effort_is_off(effort: &str) -> bool {
    matches!(effort.trim().to_ascii_lowercase().as_str(), "none" | "off" | "disabled")
}

/// The reasoning budget of a request. `thinking_budget` (tokens, at least 1) wins; on a model with a thinking switch
/// `reasoning_effort` low/medium/high stands for a budget as well. `None` means unlimited.
pub fn request_budget(field: Option<&Value>, effort: Option<&str>, control: Control) -> Result<Option<usize>, String> {
    match field {
        None | Some(Value::Null) => {}
        Some(value) => {
            return match value.as_u64().filter(|n| *n >= 1) {
                Some(n) => Ok(Some(usize::try_from(n).unwrap_or(usize::MAX))),
                None => Err("thinking_budget must be a whole number of tokens, at least 1".into()),
            };
        }
    }
    Ok(match (control, effort) {
        (Control::Toggle, Some(effort)) => budget_for_effort(effort),
        _ => None,
    })
}

/// What the text produced so far says about the reasoning block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Probe {
    /// No reasoning block has started (the model answers without thinking), so there is nothing to cut.
    Absent,
    /// Inside the block.
    Open,
    /// Inside the block, but the text ends in the middle of the closing tag.
    Closing,
    /// The block has ended.
    Closed,
}

/// What the server has to do after a token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Continue,
    /// End the reasoning block now: append the end-of-thinking text and generate the answer.
    Force,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Counting,
    Decided,
}

/// Budget forcing as a state machine. `on_token` is called after every generated token; from the budget's token on it asks
/// `probe` (the server decodes and parses the text) whether the reasoning block is still open, and answers `Force` once.
#[derive(Clone, Debug)]
pub struct BudgetMachine {
    budget: usize,
    seen: usize,
    phase: Phase,
}

impl BudgetMachine {
    pub fn new(budget: usize) -> Self {
        Self { budget: budget.max(1), seen: 0, phase: Phase::Counting }
    }

    pub fn budget(&self) -> usize {
        self.budget
    }

    /// The generation ended (or the block was closed some other way): nothing is forced any more.
    pub fn finish(&mut self) {
        self.phase = Phase::Decided;
    }

    pub fn on_token(&mut self, probe: impl FnOnce() -> Probe) -> Step {
        self.seen += 1;
        if self.phase == Phase::Decided || self.seen < self.budget {
            return Step::Continue;
        }
        match probe() {
            Probe::Open => {
                self.phase = Phase::Decided;
                Step::Force
            }
            // the closing tag is being written: look again after the next token
            Probe::Closing => Step::Continue,
            Probe::Absent | Probe::Closed => {
                self.phase = Phase::Decided;
                Step::Continue
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN3: &str = "{%- if enable_thinking is defined and enable_thinking is false %}{{ '<think>\\n\\n</think>\\n\\n' }}{%- endif %}";
    const QWEN3_OFF_BY_DEFAULT: &str = "{%- if enable_thinking %}{{ '<think>\\n' }}{%- endif %}";
    const GPT_OSS: &str = "<|start|>assistant<|channel|>analysis<|message|> Reasoning: {{ reasoning_effort | default('medium') }}";
    const DEEPSEEK_R1: &str = "{{ '<|Assistant|><think>\\n' }}";
    const LLAMA3: &str = "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>";
    const GEMMA3: &str = "<start_of_turn>{{ role }}\n{{ message['content'] }}<end_of_turn>\n";
    const GEMMA4: &str = "{%- if enable_thinking is defined and enable_thinking -%}<|think|>{%- endif -%}{{ '<|channel>thought' }}";
    const GRANITE: &str = "{%- if thinking %}Here is my thought process:{%- endif %}{{ message.thinking }}";
    const GLM: &str = "{%- if enable_thinking is defined and enable_thinking is false -%}/nothink{%- endif -%}<think>";
    const DEEPSEEK_V31: &str = "{%- if not thinking -%}</think>{%- else -%}<think>{%- endif -%}";
    const MISTRAL_PLAIN: &str = "[INST] {{ message['content'] }} [/INST]";

    fn caps(template: &str, family: Family, arch: &str) -> ThinkingCaps {
        ThinkingCaps::detect(template, family, arch)
    }

    #[test]
    fn qwen_style_toggle_that_thinks_by_default() {
        let c = caps(QWEN3, Family::Think, "qwen3");
        assert_eq!((c.control, c.variable, c.default), (Control::Toggle, "enable_thinking", "high"));
        assert_eq!(c.levels, ["off", "low", "medium", "high"]);
        let json = c.to_json();
        assert_eq!(json["supported"], true);
        assert_eq!(json["control"], "toggle");
    }

    #[test]
    fn toggle_that_is_off_by_default() {
        let c = caps(QWEN3_OFF_BY_DEFAULT, Family::Think, "x");
        assert_eq!((c.control, c.default), (Control::Toggle, "off"));
        let c = caps(GEMMA4, Family::Gemma4, "gemma4");
        assert_eq!((c.control, c.variable, c.default), (Control::Toggle, "enable_thinking", "off"));
    }

    #[test]
    fn thinking_variable_is_not_a_message_field() {
        let c = caps(GRANITE, Family::Granite, "granite");
        assert_eq!((c.control, c.variable), (Control::Toggle, "thinking"));
        let c = caps(DEEPSEEK_V31, Family::Think, "deepseek2");
        assert_eq!((c.control, c.variable, c.default), (Control::Toggle, "thinking", "high"));
        // `message.thinking` alone is history rendering, not a switch
        assert_eq!(caps("{{ message.thinking }}{{ 'thinking' }}", Family::Think, "x").control, Control::None);
    }

    #[test]
    fn effort_models() {
        let c = caps(GPT_OSS, Family::GptOss, "gpt-oss");
        assert_eq!((c.control, c.variable, c.default), (Control::Effort, "reasoning_effort", "medium"));
        assert_eq!(c.levels, ["low", "medium", "high"]);
        // the architecture alone is enough when the template is unusual
        assert_eq!(caps("{{ x }}", Family::None, "gpt-oss").control, Control::Effort);
    }

    #[test]
    fn models_that_always_think() {
        let c = caps(DEEPSEEK_R1, Family::Think, "qwen2");
        assert_eq!((c.control, c.levels.as_slice(), c.default), (Control::Always, ["high"].as_slice(), "high"));
        assert_eq!(caps("{{ x }}", Family::Mistral, "mistral3").control, Control::Always);
    }

    // Qwen3 2507 releases: both show earlier reasoning in the history (`<think>` and `</think>` appear in the text),
    // only the Thinking one opens the block in the generation prompt.
    const QWEN3_HISTORY: &str = "{%- if '</think>' in content %}{%- set reasoning_content = content.split('</think>')[0].split('<think>')[-1] %}{%- endif %}\
        {{- '<|im_start|>assistant\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}";
    const QWEN3_INSTRUCT_2507: &str = concat!(
        "{%- if '</think>' in content %}{%- set reasoning_content = content.split('</think>')[0].split('<think>')[-1] %}{%- endif %}",
        "{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}"
    );
    const QWEN3_THINKING_2507: &str = concat!(
        "{%- if '</think>' in content %}{%- set reasoning_content = content.split('</think>')[0].split('<think>')[-1] %}{%- endif %}",
        "{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n<think>\\n' }}\n{%- endif %}"
    );
    const R1_DISTILL: &str = "{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\\n'}}{% endif %}";
    const QWEN3_HYBRID: &str = concat!(
        "{%- if '</think>' in content %}{%- set reasoning_content = content.split('</think>')[0] %}{%- endif %}",
        "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}",
        "{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}{%- endif %}"
    );

    #[test]
    fn template_detection_table_by_family() {
        // (name, template, family the server picks, arch, expected control)
        let table: [(&str, &str, Family, &str, Control); 7] = [
            ("Qwen3 hybrid", QWEN3_HYBRID, Family::Think, "qwen3", Control::Toggle),
            ("Qwen3-Instruct-2507 never thinks", QWEN3_INSTRUCT_2507, Family::Think, "qwen3", Control::None),
            ("Qwen3-Thinking-2507 always thinks", QWEN3_THINKING_2507, Family::Think, "qwen3", Control::Always),
            ("DeepSeek-R1 distill", R1_DISTILL, Family::Think, "qwen2", Control::Always),
            ("gpt-oss harmony", GPT_OSS, Family::GptOss, "gpt-oss", Control::Effort),
            ("GLM with switch", GLM, Family::Think, "glm4moe", Control::Toggle),
            ("Gemma 3", GEMMA3, Family::Think, "gemma3", Control::None),
        ];
        for (name, template, family, arch, expected) in table {
            assert_eq!(caps(template, family, arch).control, expected, "{name}");
        }
        // a history-only mention without a generation prompt in the text still counts as a reasoning template
        assert_eq!(caps(QWEN3_HISTORY, Family::Think, "qwen3").control, Control::Always);
    }

    #[test]
    fn glm_with_switch_is_a_toggle() {
        assert_eq!(caps(GLM, Family::Think, "glm4moe").control, Control::Toggle);
    }

    #[test]
    fn models_without_thinking() {
        for (template, arch) in [(LLAMA3, "llama"), (GEMMA3, "gemma3"), (MISTRAL_PLAIN, "llama")] {
            let c = caps(template, Family::Think, arch);
            assert_eq!(c.control, Control::None, "{arch}");
            assert!(!c.supported());
            assert_eq!(c.to_json()["supported"], false);
            assert!(c.levels.is_empty());
        }
        assert_eq!(caps(LLAMA3, Family::None, "llama").control, Control::None);
    }

    #[test]
    fn effort_names_map_to_budgets() {
        assert_eq!(budget_for_effort("low"), Some(512));
        assert_eq!(budget_for_effort(" Medium "), Some(2048));
        assert_eq!(budget_for_effort("high"), None);
        assert_eq!(budget_for_effort("unknown"), None);
    }

    #[test]
    fn request_budget_rules() {
        let n = |v: Value| request_budget(Some(&v), None, Control::Toggle);
        assert_eq!(n(json!(300)), Ok(Some(300)));
        assert_eq!(request_budget(None, None, Control::Toggle), Ok(None));
        assert_eq!(request_budget(Some(&Value::Null), Some("low"), Control::Toggle), Ok(Some(512)));
        for bad in [json!(0), json!(-4), json!("100"), json!(1.5), json!(true)] {
            assert!(n(bad.clone()).is_err(), "{bad}");
        }
        // an explicit budget beats the effort; effort only maps on models with a switch
        assert_eq!(request_budget(Some(&json!(77)), Some("low"), Control::Toggle), Ok(Some(77)));
        assert_eq!(request_budget(None, Some("medium"), Control::Toggle), Ok(Some(2048)));
        assert_eq!(request_budget(None, Some("high"), Control::Toggle), Ok(None));
        assert_eq!(request_budget(None, Some("low"), Control::Effort), Ok(None));
        assert_eq!(request_budget(None, Some("low"), Control::Always), Ok(None));
        assert!(effort_is_off("None") && effort_is_off("off") && !effort_is_off("low"));
    }

    fn run(machine: &mut BudgetMachine, probes: &[Probe]) -> Vec<Step> {
        probes.iter().map(|p| machine.on_token(|| *p)).collect()
    }

    #[test]
    fn forces_exactly_once_at_the_budget() {
        let mut m = BudgetMachine::new(3);
        let steps = run(&mut m, &[Probe::Open; 6]);
        assert_eq!(steps, [Step::Continue, Step::Continue, Step::Force, Step::Continue, Step::Continue, Step::Continue]);
    }

    #[test]
    fn does_not_probe_before_the_budget() {
        let mut m = BudgetMachine::new(4);
        for _ in 0..3 {
            assert_eq!(m.on_token(|| panic!("probed too early")), Step::Continue);
        }
    }

    #[test]
    fn closed_or_absent_block_is_left_alone_and_not_probed_again() {
        for probe in [Probe::Closed, Probe::Absent] {
            let mut m = BudgetMachine::new(2);
            assert_eq!(m.on_token(|| Probe::Open), Step::Continue);
            assert_eq!(m.on_token(|| probe), Step::Continue);
            assert_eq!(m.on_token(|| panic!("probed after the decision")), Step::Continue);
        }
    }

    #[test]
    fn waits_for_a_half_written_closing_tag() {
        let mut m = BudgetMachine::new(2);
        let steps = run(&mut m, &[Probe::Open, Probe::Closing, Probe::Closing, Probe::Closed, Probe::Open]);
        assert_eq!(steps, [Step::Continue; 5]);
        let mut m = BudgetMachine::new(2);
        assert_eq!(run(&mut m, &[Probe::Open, Probe::Closing, Probe::Open]), [Step::Continue, Step::Continue, Step::Force]);
    }

    #[test]
    fn finished_generation_forces_nothing() {
        let mut m = BudgetMachine::new(2);
        m.on_token(|| Probe::Open);
        m.finish();
        assert_eq!(m.on_token(|| Probe::Open), Step::Continue);
    }

    #[test]
    fn budget_of_zero_counts_as_one() {
        assert_eq!(BudgetMachine::new(0).budget(), 1);
        assert_eq!(BudgetMachine::new(0).on_token(|| Probe::Open), Step::Force);
    }
}
