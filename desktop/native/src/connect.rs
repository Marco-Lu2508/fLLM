//! Settings > Connect other apps: the four examples of `spane-claude-code` and the settings the
//! server writes for a coding tool (`connect.js` of the Studio).

use crate::app::Message;
use crate::models::card;
use crate::theme::{self, Tokens};
use iced::widget::{button, column, pick_list, row, scrollable, text, Space};
use iced::{Alignment, Element, Font, Length};

/// The four examples of the first card (`#sdkSnippetTabs`): the id, the label of the tab and the
/// language the header shows for it.
///
/// The language is `bash` for all four: `#sdkSnippetLang` carries `bash` in `shell.html` and
/// nothing in the Studio ever writes it again — the header says *bash* even for the Python and
/// TypeScript examples, and this file keeps that.
pub const SDKS: [(&str, &str); 4] = [
    ("claude", "Claude Code CLI"),
    ("python", "OpenAI Python"),
    ("ts", "OpenAI TypeScript"),
    ("curl", "cURL"),
];

/// The header of the code card of the first card.
pub const SDK_LANG: &str = "bash";

/// The description of the first card (its `<p class="app-item-desc">`).
pub const SDK_HELP: &str = "fLLM speaks the same languages as the big AI services, so coding tools such as Claude Code, Cursor or Continue, and the official libraries, can use your model. Copy one of these examples.";

/// The description of the second card (the `<p class="app-item-desc">` of `spane-claude-code`,
/// with its `<span class="mono">` words set like the rest of the sentence).
pub const CONNECT_HELP: &str = "Pick your tool: you get the exact settings for this server and the model that is loaded now. The same text comes from fllm-rs connect TOOL in a terminal, which can also write the files for you (--write).";

/// What the header of the second card says about the key (`statusLine` of `connect.js`): a
/// placeholder is always fine here, and a server that wants a real key is said so.
pub fn status_line(auth_required: bool) -> String {
    if auth_required {
        "The key \"fllm\" is a placeholder; this server wants a real key, put yours in its place.".into()
    } else {
        "The key \"fllm\" is a placeholder: this server accepts any value.".into()
    }
}

/// One of the shells of the second card (`selConnectShell`): what it is called, and the name in
/// the answer of the server.
pub const SHELLS: [(&str, &str); 3] = [("sh", "bash / zsh"), ("powershell", "PowerShell"), ("cmd", "cmd")];

/// Where the shell select starts: PowerShell on Windows, a shell of the family everywhere else.
pub fn default_shell() -> &'static str {
    if cfg!(windows) {
        "powershell"
    } else {
        "sh"
    }
}

/// The settings of a tool as the server describes them (`GET /api/studio/connect?tool=`).
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct Plan {
    #[serde(default)]
    pub notes: Vec<String>,
    #[serde(default)]
    pub env: std::collections::HashMap<String, Vec<String>>,
    #[serde(default)]
    pub files: Vec<File>,
    #[serde(default)]
    pub snippet: Option<Snippet>,
    #[serde(default)]
    pub start: Vec<String>,
    #[serde(default)]
    pub tools: Vec<Tool>,
}

/// One file of the plan: where it goes, what it holds, or why it cannot be written here.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct File {
    pub path: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub problem: Option<String>,
}

impl File {
    /// The title of the block: the path, and the reason when the file cannot be created
    /// (`render()` of `connect.js`).
    pub fn title(&self) -> String {
        if self.problem.is_some() {
            format!("{} (cannot be created here)", self.path)
        } else {
            self.path.clone()
        }
    }

    /// What the block holds and what its Copy button copies: the content when there is one, and
    /// the reason when there is not.
    pub fn body(&self) -> String {
        self.content
            .clone()
            .or_else(|| self.problem.clone())
            .unwrap_or_default()
    }
}

/// The little program of the second card.
#[derive(Debug, Clone, Default, PartialEq, serde::Deserialize)]
pub struct Snippet {
    pub title: String,
    pub text: String,
}

/// One tool of the select (`TOOLS` of the engine, handed out by the server).
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Tool {
    pub id: String,
    pub what: String,
}

/// How a block of the second card is drawn: a plain line of text (the notes), a title with lines
/// under it (`<strong>` of `connect.js`), or a code card with a Copy button.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Plain,
    Strong,
    Card,
}

/// One block of the second card, drawn in the order `render()` of `connect.js` builds them.
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub title: String,
    pub body: String,
    pub kind: Kind,
}

/// The blocks of the second card in the order of the page: the notes, the environment of the
/// shell that is chosen, the files, the snippet and *Start*.
pub fn blocks(plan: &Plan, shell: &str) -> Vec<Block> {
    let mut out = vec![Block {
        title: String::new(),
        body: plan.notes.join("\n"),
        kind: Kind::Plain,
    }];
    if let Some(lines) = plan.env.get(shell) {
        if !lines.is_empty() {
            out.push(Block { title: "Environment".into(), body: lines.join("\n"), kind: Kind::Strong });
        }
    }
    for file in &plan.files {
        out.push(Block { title: file.title(), body: file.body(), kind: Kind::Card });
    }
    if let Some(snippet) = &plan.snippet {
        out.push(Block { title: snippet.title.clone(), body: snippet.text.clone(), kind: Kind::Card });
    }
    if !plan.start.is_empty() {
        out.push(Block { title: "Start".into(), body: plan.start.join("\n"), kind: Kind::Strong });
    }
    out
}

/// The label of a shell of the select (`selConnectShell` shows the label): nothing is selected
/// for a shell the page does not know.
pub fn shell_label(id: &str) -> Option<&'static str> {
    SHELLS.iter().find(|(name, _)| *name == id).map(|(_, label)| *label)
}

/// The name of a shell from the label of the select.
pub fn shell_id(label: &str) -> &'static str {
    SHELLS.iter().find(|(_, name)| *name == label).map(|(id, _)| *id).unwrap_or("sh")
}

/// The example of the tab that is chosen (`updateCliAndSdkSnippets` of `panels.js`), with the
/// origin of the window, the key of `state.apiKey` (nothing stores one here, so the example keeps
/// the `fllm-sk-local` of the Studio), the model that is loaded and the temperature that is in
/// the box of the sampler.
pub fn snippet(lang: &str, origin: &str, key: &str, model: &str, temperature: f32) -> String {
    let temperature = format!("{temperature}");
    match lang {
        "python" => format!(
            r#"from openai import OpenAI

client = OpenAI(base_url="{origin}/v1", api_key="{key}")
res = client.chat.completions.create(
    model="{model}",
    messages=[{{"role": "user", "content": "Hello from fLLM!"}}],
    temperature={temperature},
)
print(res.choices[0].message.content)"#
        ),
        "ts" => format!(
            r#"import OpenAI from 'openai';

const client = new OpenAI({{ baseURL: '{origin}/v1', apiKey: '{key}' }});
const res = await client.chat.completions.create({{
  model: '{model}',
  messages: [{{ role: 'user', content: 'Hello from fLLM!' }}],
}});
console.log(res.choices[0]?.message?.content);"#
        ),
        "curl" => format!(
            r#"curl {origin}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ***" \
  -d '{{"model":"{model}","messages":[{{"role":"user","content":"Hello!"}}]}}'"#
        ),
        // `claude` — and the tabs only offer these four
        _ => format!(
            r#"export ANTHROPIC_BASE_URL="{origin}"
export ANTHROPIC_API_KEY="{key}"

# Run Claude Code against your local fLLM Rust server:
claude --model {model}"#
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The example of the first tab, written the way `updateCliAndSdkSnippets` writes it (the
    /// line breaks, the quotes and the `Hello from fLLM!` of the Studio included).
    #[test]
    fn the_four_examples_are_the_ones_of_the_studio() {
        let origin = "http://127.0.0.1:8000";
        let key = "fllm-sk-local";
        let model = "gpt-oss-20b";
        assert_eq!(
            snippet("claude", origin, key, model, 0.7),
            "export ANTHROPIC_BASE_URL=\"http://127.0.0.1:8000\"\nexport ANTHROPIC_API_KEY=\"fllm-sk-local\"\n\n# Run Claude Code against your local fLLM Rust server:\nclaude --model gpt-oss-20b"
        );
        assert_eq!(
            snippet("python", origin, key, model, 0.7),
            "from openai import OpenAI\n\nclient = OpenAI(base_url=\"http://127.0.0.1:8000/v1\", api_key=\"fllm-sk-local\")\nres = client.chat.completions.create(\n    model=\"gpt-oss-20b\",\n    messages=[{\"role\": \"user\", \"content\": \"Hello from fLLM!\"}],\n    temperature=0.7,\n)\nprint(res.choices[0].message.content)"
        );
        assert_eq!(
            snippet("ts", origin, key, model, 0.7),
            "import OpenAI from 'openai';\n\nconst client = new OpenAI({ baseURL: 'http://127.0.0.1:8000/v1', apiKey: 'fllm-sk-local' });\nconst res = await client.chat.completions.create({\n  model: 'gpt-oss-20b',\n  messages: [{ role: 'user', content: 'Hello from fLLM!' }],\n});\nconsole.log(res.choices[0]?.message?.content);"
        );
        assert_eq!(
            snippet("curl", origin, key, model, 0.7),
            "curl http://127.0.0.1:8000/v1/chat/completions \\\n  -H \"Content-Type: application/json\" \\\n  -H \"Authorization: Bearer ***\" \\\n  -d '{\"model\":\"gpt-oss-20b\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}'"
        );
        // the temperature is written the way a browser writes it in a template literal
        assert!(
            snippet("python", origin, key, model, 1.0).contains("temperature=1,"),
            "{}",
            snippet("python", origin, key, model, 1.0)
        );
    }

    /// The line about the key: always the placeholder sentence, with the half of the Studio's
    /// `statusLine` that fits the server (`state.apiKeyRequired` of the page).
    #[test]
    fn the_line_about_the_key_has_the_two_halves_of_the_studio() {
        assert_eq!(
            status_line(false),
            "The key \"fllm\" is a placeholder: this server accepts any value."
        );
        assert_eq!(
            status_line(true),
            "The key \"fllm\" is a placeholder; this server wants a real key, put yours in its place."
        );
    }

    /// The shell of the select: its label comes back to its name, and an unknown one is not
    /// selected at all.
    #[test]
    fn the_shell_selects_by_its_label() {
        assert_eq!(shell_id("bash / zsh"), "sh");
        assert_eq!(shell_id("PowerShell"), "powershell");
        assert_eq!(shell_id("cmd"), "cmd");
        assert_eq!(shell_label("sh"), Some("bash / zsh"));
        assert_eq!(shell_label("fish"), None);
        // the page starts at PowerShell on Windows and at a shell of the family everywhere
        // else — which one is checked here for the machine this test runs on
        if cfg!(windows) {
            assert_eq!(default_shell(), "powershell", "the page asks for PowerShell on Windows");
        } else {
            assert_eq!(default_shell(), "sh", "the page asks for sh away from Windows");
        }
    }

    /// The blocks of the plan, in the order `render()` of `connect.js` builds them: notes first,
    /// the environment of the shell that is chosen, every file, the snippet and *Start*.
    #[test]
    fn the_blocks_come_in_the_order_of_the_page() {
        let plan: Plan = serde_json::from_value(json!({
            "notes": ["first note", "second note"],
            "env": {
                "powershell": ["$env:FOO = 'bar'"],
                "sh": ["export FOO=bar"],
                "cmd": []
            },
            "files": [
                {"path": ".claude/settings.local.json", "content": "{\"a\": 1}"},
                {"path": "/etc/thing", "problem": "permission denied"}
            ],
            "snippet": {"title": "Settings", "text": "the program of the plan"},
            "start": ["claude", "--model", "fllm"],
            "tools": [{"id": "claude-code", "what": "Claude Code"}]
        }))
        .expect("a plan of the server");

        let sh: Vec<(String, Kind)> = blocks(&plan, "sh")
            .into_iter()
            .map(|b| (b.title, b.kind))
            .collect();
        assert_eq!(
            sh,
            vec![
                (String::new(), Kind::Plain),
                ("Environment".into(), Kind::Strong),
                (".claude/settings.local.json".into(), Kind::Card),
                ("/etc/thing (cannot be created here)".into(), Kind::Card),
                ("Settings".into(), Kind::Card),
                ("Start".into(), Kind::Strong),
            ],
            "the titles and how they are drawn"
        );

        // the shell is chosen: cmd holds nothing, so its environment is not drawn
        let cmd = blocks(&plan, "cmd");
        assert!(
            !cmd.iter().any(|b| b.title == "Environment"),
            "an empty environment is not drawn"
        );

        let powershell = blocks(&plan, "powershell");
        assert_eq!(powershell[1].body, "$env:FOO = 'bar'");

        // the file that cannot be created carries its reason where its content would be
        let failed = blocks(&plan, "sh").remove(3);
        assert_eq!(failed.body, "permission denied");
        // the notes are one line of text, and the program of the plan is copied as it came
        let all = blocks(&plan, "sh");
        assert_eq!(all[0].body, "first note\nsecond note");
        assert_eq!(all[4].body, "the program of the plan");
        assert_eq!(all[5].body, "claude\n--model\nfllm");
    }

    /// A plan with nothing in it still gives its notes, and the page writes *Not built in yet.*
    #[test]
    fn an_empty_plan_still_shows_its_notes() {
        let plan = Plan::default();
        let out = blocks(&plan, "sh");
        assert_eq!(out.len(), 1, "only the notes: {}", out.len());
        assert!(out[0].body.is_empty());
    }
}

/// The data of both cards, read from the answer of the server and the store.
#[derive(Clone, Copy)]
pub struct ConnectView<'a> {
    pub sdk_lang: &'a str,
    pub origin: &'a str,
    pub model: &'a str,
    pub temperature: f32,
    pub tool: &'a str,
    pub shell: &'a str,
    pub plan: Option<&'a Plan>,
    pub say: &'a str,
    pub auth_required: bool,
}

/// One block of the second card: a plain line of text, a title with lines under it when the
/// Studio writes `<strong>`, or a code card whose Copy button reads *Copied* for 1500 ms
/// (`render()` of `connect.js`).
fn block_element<'a>(t: Tokens, b: Block, copied: &Option<String>) -> Element<'a, Message> {
    match b.kind {
        Kind::Plain => text(b.body).size(13).color(t.fg_soft).into(),
        Kind::Strong => column![
            text(b.title)
                .size(13)
                .color(t.fg)
                .font(Font { weight: iced::font::Weight::Semibold, ..Font::DEFAULT }),
            text(b.body).size(13).color(t.fg_soft).font(Font::MONOSPACE),
        ]
        .spacing(2)
        .into(),
        Kind::Card => {
            let Block { title, body, .. } = b;
            let label = if copied.as_deref() == Some(body.as_str()) { "Copied" } else { "Copy" };
            let mut head = row![text(title).size(13).color(t.fg)]
                .push(Space::new().width(Length::Fill))
                .align_y(Alignment::Center);
            if !body.is_empty() {
                head = head.push(
                    button(text(label).size(12))
                        .on_press(Message::CopyBlock(body.clone()))
                        .padding([3, 8])
                        .style(theme::ghost_button(t)),
                );
            }
            column![head, text(body).size(13).color(t.fg_soft).font(Font::MONOSPACE)]
                .spacing(4)
                .into()
        }
    }
}

/// The code card of the first card: what is written in it and a Copy button beside the language
/// (`.code-card` with `#preSdkSnippet`). The text is copied into the card, so the card does not
/// borrow the example — which lives only as long as this call.
fn code_card<'a>(t: Tokens, title: &str, body: &str, copy: Message) -> Element<'a, Message> {
    // the texts go into the card as they are, so the card holds them and not the caller
    let title = title.to_string();
    let body = body.to_string();
    let head = row![
        text(title).size(13).color(t.muted).font(Font::MONOSPACE),
        Space::new().width(Length::Fill),
        button(text("Copy").size(12))
            .on_press(copy)
            .padding([3, 8])
            .style(theme::ghost_button(t)),
    ]
    .align_y(Alignment::Center);
    column![head, scrollable(text(body).size(13).color(t.fg_soft).font(Font::MONOSPACE))]
        .spacing(6)
        .into()
}

/// The two cards of `spane-claude-code`: the examples the Studio shows for every way of talking
/// to it, and the settings the server writes for a coding tool.
pub fn section<'a>(t: Tokens, v: ConnectView<'a>, copied: &Option<String>) -> Element<'a, Message> {
    // ---- Connect other apps -------------------------------------------------------------
    let example = v.snippet();
    let mut tabs = row![].spacing(6);
    for (id, label) in SDKS {
        tabs = tabs.push(
            button(text(label).size(13))
                .on_press(Message::SdkTab(id))
                .style(theme::row_button(t, id == v.sdk_lang)),
        );
    }
    let first = column![
        text(SDK_HELP).size(13).color(t.fg_soft),
        tabs,
        code_card(t, SDK_LANG, &example, Message::SdkCopy),
    ]
    .spacing(10);

    // ---- Connect a coding tool ----------------------------------------------------------
    let mut second = column![].spacing(10);
    // one sentence, as every description of these cards is drawn (`app-item-desc` is one
    // paragraph: its `mono` words are set like the rest of it, not put beside it)
    second = second.push(text(CONNECT_HELP).size(13).color(t.fg_soft));

    // the two selects: the tool comes from the answer of the server, and the select shows what
    // it is called there — the id comes back out of the label, the way `AmdNpuChosen` reads it
    let tool_labels: Vec<String> = v
        .plan
        .into_iter()
        .flat_map(|plan| plan.tools.iter().map(|tool| format!("{} - {}", tool.id, tool.what)))
        .collect();
    let selected_tool = v
        .plan
        .and_then(|plan| plan.tools.iter().find(|tool| tool.id == v.tool))
        .map(|tool| format!("{} - {}", tool.id, tool.what));
    second = second.push(
        row![
            pick_list(tool_labels, selected_tool, |label: String| {
                Message::ConnectTool(label.split(" - ").next().unwrap_or(label.as_str()).to_string())
            })
            .placeholder("Select a tool"),
            pick_list(
                SHELLS.iter().map(|(_, label)| *label).collect::<Vec<_>>(),
                shell_label(v.shell),
                |label: &'static str| Message::ConnectShell(shell_id(label)),
            )
            .placeholder("Shell"),
        ]
        .spacing(10),
    );

    // `ctOut` of the page: a read that failed writes its message there and leaves it standing
    // until the next one succeeds, a read that has not come yet says so, and an answer gives its
    // blocks in the order above
    if !v.say.is_empty() {
        second = second.push(text(v.say).size(13).color(t.danger));
    } else if let Some(plan) = v.plan {
        for b in blocks(plan, v.shell) {
            second = second.push(block_element(t, b, copied));
        }
    } else {
        second = second.push(text("Reading the settings...").size(13).color(t.muted));
    }
    // the line about the key: always the placeholder sentence of the Studio's `statusLine`
    second = second.push(text(status_line(v.auth_required)).size(12).color(t.muted));

    column![
        card(t, "Connect other apps", first.into()),
        card(t, "Connect a coding tool", second.into()),
    ]
    .spacing(14)
    .into()
}

impl ConnectView<'_> {
    /// The example of the chosen tab.
    pub fn snippet(&self) -> String {
        snippet(self.sdk_lang, self.origin, "fllm-sk-local", self.model, self.temperature)
    }
}
