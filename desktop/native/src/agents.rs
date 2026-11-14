//! The **Sub-agents** card: Markdown files in `agents/` the coding agent can hand a job to.
//!
//! Port of `agentsSettings` in `rust/engine/src/webui/agents.js` (the whole file), including the
//! pure helpers of its `@pure-begin` block — the name rule, the two groups of tools, what the
//! form sends and the line that says what a sub-agent may do.

use iced::{
    Font, Length,
    alignment::Alignment,
    widget::{button, checkbox, column, container, row, text, text_editor, text_input, Space},
    Element,
};
use serde::Deserialize;

use crate::app::Message;
use crate::models::card;
use crate::theme::{self, Tokens};

/// The two sub-agents the app ships with (`AGENT_BUILTINS` of the page).
pub const BUILTINS: [(&str, &str); 2] = [
    (
        "explore",
        "Reads and searches the project and reports what it found. It cannot change anything.",
    ),
    (
        "general",
        "Does a whole job on its own: it may also edit files and run commands.",
    ),
];

/// The names of the tools that may never be given to a sub-agent (`AGENT_HIDDEN_TOOLS`).
pub const HIDDEN_TOOLS: [&str; 3] = ["task", "spawn_agent", "create_agent"];

/// The tools that change things or start processes (`AGENT_ACTING_TOOLS`): a sub-agent that may
/// call one of them is not read-only.
pub const ACTING_TOOLS: [&str; 7] = [
    "write_file", "edit_file", "multi_edit", "patch", "bash", "bash_kill", "memory",
];

/// The tools of the coding agent as the page keeps them, for the times the server has no route
/// for them (`TOOL_DEFS` of `code.js` — the names and permissions, which is what the boxes of the
/// form use; their descriptions are the `title` of a box, which this window does not draw).
pub const BUILTIN_TOOL_DEFS: [(&str, &str); 8] = [
    ("list_dir", "read"),
    ("read_file", "read"),
    ("write_file", "edit"),
    ("edit_file", "edit"),
    ("glob", "read"),
    ("grep", "read"),
    ("bash", "command"),
    ("create_agent", "edit"),
];

/// One tool of a list (`agentToolGroups` reads name and permission).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tool {
    pub name: String,
    pub permission: String,
}

/// The two groups of the form in the order the page writes them (`agentToolGroups`): what only
/// reads (the default) and what changes things, each by its name.
pub fn tool_groups(tools: &[Tool]) -> (Vec<Tool>, Vec<Tool>) {
    let mut read: Vec<Tool> = Vec::new();
    let mut act: Vec<Tool> = Vec::new();
    for tool in tools {
        if tool.name.is_empty() || HIDDEN_TOOLS.contains(&tool.name.as_str()) || tool.permission == "computer" {
            continue;
        }
        if tool.permission == "read" && !ACTING_TOOLS.contains(&tool.name.as_str()) {
            read.push(tool.clone());
        } else {
            act.push(tool.clone());
        }
    }
    read.sort_by(|a, b| a.name.cmp(&b.name));
    act.sort_by(|a, b| a.name.cmp(&b.name));
    (read, act)
}

/// The tools the page keeps in `BUILTIN_TOOL_DEFS` as its own list, for the times without a model
/// (`if (!agentRoutesUp())` of `loadTools`).
pub fn builtin_tools() -> Vec<Tool> {
    BUILTIN_TOOL_DEFS
        .iter()
        .map(|(name, permission)| Tool {
            name: (*name).to_string(),
            permission: (*permission).to_string(),
        })
        .collect()
}

/// The same list, already in its two groups.
pub fn builtin_groups() -> (Vec<Tool>, Vec<Tool>) {
    tool_groups(&builtin_tools())
}

/// Which boxes start checked: the agent's own tool list, or — a new one, or an old one without a
/// list — everything that only reads (`agentCheckedTools`).
pub fn checked_tools(agent_tools: Option<&[String]>, read: &[Tool]) -> Vec<String> {
    match agent_tools {
        Some(list) if !list.is_empty() => list.to_vec(),
        _ => read.iter().map(|tool| tool.name.clone()).collect(),
    }
}

/// `true` for a name the page accepts for a sub-agent (`AGENT_NAME_RE`:
/// `^[a-z0-9][a-z0-9_-]{0,63}$`).
pub fn valid_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    if bytes.is_empty() || bytes.len() > 64 {
        return false;
    }
    let first = bytes[0];
    if !(first.is_ascii_lowercase() || first.is_ascii_digit()) {
        return false;
    }
    bytes[1..]
        .iter()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == b'-' || *c == b'_')
}

/// What the form sends, or the words the page says instead (`agentPayload`).
///
/// `workspace` is the folder the Code page has open, or nothing; `is_new` is whether the form
/// creates one; `existing` are the names the list already holds.
///
/// The page passes those nine pieces as one form object (`agentPayload(form, workspace, isNew,
/// existingNames)`); here they are the arguments of the call.
#[allow(clippy::too_many_arguments)]
pub fn payload(
    name: &str,
    description: &str,
    body: &str,
    tools: &[String],
    model: &str,
    max_steps: &str,
    workspace_scope: bool,
    workspace: Option<&str>,
    is_new: bool,
    existing: &[String],
) -> Result<serde_json::Value, String> {
    let name = name.trim();
    if !valid_name(name) {
        return Err("Give the sub-agent a short name: lowercase letters, digits, - and _ (for example code-reviewer).".to_string());
    }
    if BUILTINS.iter().any(|(builtin, _)| *builtin == name) {
        return Err(format!("\"{name}\" is a built-in sub-agent. Pick another name."));
    }
    if is_new && existing.iter().any(|existing| existing == name) {
        return Err(format!(
            "You already have a sub-agent called \"{name}\". Pick another name or edit that one."
        ));
    }
    let body = body.trim();
    if body.is_empty() {
        return Err(
            "Write the instructions: what this sub-agent does and what its report should contain."
                .to_string(),
        );
    }
    let scope = if workspace_scope { "workspace" } else { "user" };
    if scope == "workspace" && workspace.is_none() {
        return Err(
            "Open a folder on the Code page first, or store the sub-agent for all folders."
                .to_string(),
        );
    }
    let steps: f64 = max_steps.trim().parse().unwrap_or(0.0);
    let steps = if steps.is_nan() { 0.0 } else { steps };
    let steps = steps.floor();
    let steps = steps.clamp(0.0, 60.0) as i64;
    // the description is collapsed the way the page collapses it: every run of whitespace
    // becomes one space
    let description: String = description.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut seen: Vec<String> = Vec::new();
    for tool in tools {
        let tool = tool.trim();
        if !tool.is_empty() && !seen.iter().any(|seen| seen == tool) {
            seen.push(tool.to_string());
        }
    }
    Ok(serde_json::json!({
        "name": name,
        "description": description,
        "body": body,
        "tools": seen,
        "model": model.trim(),
        "max_steps": steps,
        "scope": scope,
        "workspace": workspace,
    }))
}

/// One line about what a sub-agent may do for the list (`agentReach`).
pub fn reach(agent_tools: Option<&[String]>, act: &[Tool]) -> String {
    let tools = agent_tools.unwrap_or(&[]);
    if tools.is_empty() {
        return "reads and searches only".to_string();
    }
    let mut acting: Vec<&str> = ACTING_TOOLS.to_vec();
    acting.extend(act.iter().map(|tool| tool.name.as_str()));
    let writes = tools.iter().filter(|tool| acting.contains(&tool.as_str())).count();
    if writes > 0 {
        format!("can change files or run commands ({} tools)", tools.len())
    } else {
        format!("reads and searches ({} tools)", tools.len())
    }
}

/// One sub-agent as the list answers it (`GET /api/studio/agents`).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Agent {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub body: String,
    #[serde(default)]
    pub tools: Vec<String>,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub max_steps: i64,
    #[serde(default)]
    pub scope: String,
    /// A sub-agent an installed plugin brings: shown, not editable here.
    #[serde(default)]
    pub read_only: bool,
}

/// The folders the server names (`data.folders`).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Folders {
    #[serde(default)]
    pub user: String,
}

/// The answer of `GET /api/studio/agents`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct AgentsAnswer {
    #[serde(default)]
    pub agents: Vec<Agent>,
    #[serde(default)]
    pub folders: Folders,
}

/// The form while it is open: the boxes as they are typed, the tools the boxes may name and the
/// little run at the end that tries the instructions.
#[derive(Default)]
pub struct AgentEdit {
    /// The name the form was opened for — empty for a new one; the boxes may not change it.
    pub original: String,
    pub is_new: bool,
    pub name: String,
    pub description: String,
    pub body: text_editor::Content,
    pub model: String,
    pub max_steps: String,
    pub workspace_scope: bool,
    pub checked: Vec<String>,
    /// What the **Run the sample** answer says (`#agentStatus` of the form).
    pub status: String,
    pub try_open: bool,
    pub try_job: text_editor::Content,
    /// The words of `#agentTryOut`: the page hides the box until it has any.
    pub try_out: Option<String>,
}

impl AgentEdit {
    /// The form the **New sub-agent** button opens: nothing is filled in.
    pub fn new() -> Self {
        AgentEdit {
            original: String::new(),
            is_new: true,
            ..AgentEdit::default()
        }
    }

    /// The form the **Edit** button opens for one agent (`startEdit`): its own fields, and the
    /// boxes that start checked once the groups are known (`agentCheckedTools`).
    pub fn open(agent: &Agent, groups: &(Vec<Tool>, Vec<Tool>)) -> Self {
        let checked = checked_tools(Some(&agent.tools), &groups.0);
        AgentEdit {
            original: agent.name.clone(),
            is_new: false,
            name: agent.name.clone(),
            description: agent.description.clone(),
            body: text_editor::Content::with_text(&agent.body),
            model: agent.model.clone(),
            max_steps: agent.max_steps.to_string(),
            workspace_scope: agent.scope == "workspace",
            checked,
            status: String::new(),
            try_open: false,
            try_job: text_editor::Content::default(),
            try_out: None,
        }
    }

    /// Whether one of the boxes of the form starts checked.
    pub fn is_checked(&self, name: &str) -> bool {
        self.checked.iter().any(|checked| checked == name)
    }

    /// Ticking or unticking one box, with the state the box reports.
    pub fn toggle(&mut self, name: &str, on: bool) {
        let at = self.checked.iter().position(|checked| checked == name);
        match (on, at) {
            (true, None) => self.checked.push(name.to_string()),
            (false, Some(at)) => {
                self.checked.remove(at);
            }
            _ => {}
        }
    }
}

/// What the list draws as its sections, in the order the page writes them.
#[derive(Default)]
pub struct AgentsView<'a> {
    pub agents: &'a [Agent],
    pub folders_user: &'a str,
    pub notice: &'a str,
    pub groups: (&'a [Tool], &'a [Tool]),
    pub editing: Option<&'a AgentEdit>,
    pub workspace: &'a str,
}

/// One row of the list: its name with its chip, what it is for, what it may do, and its buttons.
fn agent_row<'a>(
    t: Tokens,
    agent: &'a Agent,
    act: &[Tool],
    chip: Option<&'static str>,
) -> Element<'a, Message> {
    let name = agent.name.clone();
    let scope = agent.scope.clone();
    let mut head = row![text(agent.name.clone()).size(14).color(t.fg)].spacing(6);
    if let Some(chip) = chip {
        head = head.push(chip_label(t, chip));
    } else if agent.scope == "workspace" {
        head = head.push(chip_label(t, "this folder only"));
    }
    let description = if agent.description.is_empty() {
        "No description yet: say when the coding agent should use it."
    } else {
        &agent.description
    };
    let mut reach_line = reach(Some(&agent.tools), act);
    if !agent.model.is_empty() {
        reach_line.push_str(&format!(" · model {}", agent.model));
    }
    let body = if chip.is_some() {
        // the built-in and the plugin rows have no buttons and no reach line
        column![head, text(description).size(13).color(t.fg_soft)].spacing(3)
    } else {
        column![
            head,
            text(description).size(13).color(t.fg_soft),
            text(reach_line).size(13).color(t.fg_soft),
        ]
        .spacing(3)
    };
    if chip.is_some() {
        return column![body].padding([4, 0]).spacing(4).into();
    }
    column![
        row![
            body,
            Space::new().width(Length::Fill),
            column![
                button(text("Edit").size(12))
                    .on_press(Message::AgentEdit(name.clone(), scope.clone()))
                    .padding([2, 8])
                    .style(theme::ghost_button(t)),
                button(text("Delete").size(12))
                    .on_press(Message::AgentDeleteAsk(name, scope))
                    .padding([2, 8])
                    .style(theme::ghost_button(t)),
            ]
            .spacing(4),
        ]
        .spacing(8)
        .align_y(Alignment::Center),
    ]
    .padding([4, 0])
    .spacing(4)
    .into()
}

/// The little chip of a row (`quant-chip` of the page).
fn chip_label<'a>(t: Tokens, label: &'a str) -> Element<'a, Message> {
    container(text(label).size(11).color(t.muted))
        .padding([1, 6])
        .style(move |theme: &iced::Theme| {
            let palette = theme.extended_palette();
            container::Style {
                background: Some(iced::Background::Color(palette.background.weak.color)),
                border: iced::border::rounded(6),
                ..Default::default()
            }
        })
        .into()
}

/// One box of a tool (`agentToolBoxes`): its name, next to the box that gives it.
fn tool_box<'a>(_t: Tokens, tool: &Tool, checked: bool) -> Element<'a, Message> {
    let name = tool.name.clone();
    // the box of the page carries the name of the tool in `mono`; a checkbox of the window
    // takes its label as its words, without a font of its own
    checkbox(checked)
        .label(tool.name.clone())
        .size(16)
        .text_size(13)
        .on_toggle(move |value| Message::AgentToolToggle(name.clone(), value))
        .into()
}

/// A labelled box of the form with its small line under it (`agent-field` + `small`).
fn field<'a>(t: Tokens, label: &'a str, control: Element<'a, Message>, small: &'a str) -> Element<'a, Message> {
    column![
        text(label)
            .size(13)
            .color(t.fg)
            .font(Font {
                weight: iced::font::Weight::Semibold,
                ..Font::DEFAULT
            }),
        control,
        text(small).size(11).color(t.muted),
    ]
    .spacing(6)
    .into()
}

/// The form (`renderForm` of the page).
fn form<'a>(t: Tokens, v: &AgentsView<'a>, edit: &'a AgentEdit) -> Element<'a, Message> {
    let title = if edit.is_new {
        "New sub-agent".to_string()
    } else {
        format!("Edit {}", edit.original)
    };
    let mut body = column![text(title).size(15).color(t.fg)].spacing(12);

    // Name: the box may not be changed for an agent that exists
    let name_control: Element<'a, Message> = if edit.is_new {
        input(t, "code-reviewer", &edit.name, Message::AgentName)
    } else {
        text_input("code-reviewer", &edit.name)
            .size(13)
            .padding([6, 10])
            .font(Font::MONOSPACE)
            .style(move |_, status| text_input::Style {
                background: iced::Background::Color(t.surface_2),
                border: iced::Border {
                    radius: 8.0.into(),
                    width: 1.0,
                    color: if matches!(status, text_input::Status::Focused { .. }) {
                        t.fg_soft
                    } else {
                        t.separator
                    },
                },
                icon: t.muted,
                placeholder: t.muted,
                value: t.muted,
                selection: t.primary_soft,
            })
            .into()
    };
    body = body.push(field(
        t,
        "Name",
        name_control,
        "The coding agent picks it by this name. Lowercase letters, digits, - and _.",
    ));
    body = body.push(field(
        t,
        "When to use it",
        input(
            t,
            "Reviews a change for bugs and missing tests",
            &edit.description,
            Message::AgentDescription,
        ),
        "One sentence. The coding agent reads it to decide whether this sub-agent fits a job.",
    ));
    body = body.push(field(
        t,
        "Instructions",
        area(
            t,
            &edit.body,
            "You review code changes. Read the diff, name the file and line of each problem and explain why it is one. End with a short list of what to fix first.",
            Message::AgentBody,
            160.0,
        ),
        "The system prompt of this sub-agent. It does not see your conversation, only the job it is given.",
    ));

    // Allowed tools: the two groups, and the tools the file names that this build does not list
    let mut tools = column![text("Allowed tools")
        .size(13)
        .color(t.fg)
        .font(Font {
            weight: iced::font::Weight::Semibold,
            ..Font::DEFAULT
        })]
    .spacing(6);
    tools = tools.push(text("Read and search").size(12).color(t.muted));
    if v.groups.0.is_empty() {
        tools = tools.push(text("No tool list available yet.").size(13).color(t.fg_soft));
    } else {
        tools = tools.push(tool_boxes(t, v.groups.0, edit));
    }
    let known: Vec<&str> = v
        .groups
        .0
        .iter()
        .chain(v.groups.1.iter())
        .map(|tool| tool.name.as_str())
        .collect();
    let extra: Vec<Tool> = edit
        .checked
        .iter()
        .filter(|name| !known.contains(&name.as_str()))
        .map(|name| Tool {
            name: name.clone(),
            permission: String::new(),
        })
        .collect();
    if !v.groups.1.is_empty() || !extra.is_empty() {
        tools = tools.push(text("Change files and run commands").size(12).color(t.muted));
        let mut boxes: Vec<Tool> = v.groups.1.to_vec();
        boxes.extend(extra);
        tools = tools.push(tool_boxes(t, &boxes, edit));
    }
    tools = tools.push(text("A sub-agent with only reading tools cannot change your files.")
        .size(11)
        .color(t.muted));
    body = body.push(tools);

    // Model and step limit
    body = body.push(
        row![
            field(
                t,
                "Model",
                input(t, "Same as the coding agent", &edit.model, Message::AgentModel),
                "Leave empty to use the model that runs the session.",
            ),
            field(
                t,
                "Step limit",
                input(t, "0", &edit.max_steps, Message::AgentSteps),
                "Most tool calls before it must report. 0 means the default (16).",
            ),
        ]
        .spacing(12),
    );

    // Available in: the page's second radio is disabled without a folder, and says so
    let workspace_open = !v.workspace.is_empty();
    let only_label: &'static str = if workspace_open {
        "Only the open folder"
    } else {
        "Only the open folder (open a folder on the Code page first)"
    };
    let scope_picker = column![text("Available in")
        .size(13)
        .color(t.fg)
        .font(Font {
            weight: iced::font::Weight::Semibold,
            ..Font::DEFAULT
        })]
    .push(
        crate::settings::segmented(
            t,
            vec![
                ("All folders", !edit.workspace_scope, Message::AgentScope(false)),
                (only_label, edit.workspace_scope, Message::AgentScope(true)),
            ],
        ),
    );
    body = body.push(scope_picker);

    // The buttons of the form with its own status line
    body = body.push(
        row![
            button(text(if edit.is_new {
                "Create sub-agent"
            } else {
                "Save changes"
            })
            .size(13))
            .on_press(Message::AgentSave)
            .padding([6, 12])
            .style(theme::primary_button(t)),
            button(text("Cancel").size(13))
                .on_press(Message::AgentCancel)
                .padding([6, 12])
                .style(theme::ghost_button(t)),
            text(edit.status.clone()).size(12).color(t.fg_soft),
        ]
        .spacing(8)
        .align_y(Alignment::Center),
    );

    // Try the instructions
    let mut try_box = column![
        button(
            text(if edit.try_open {
                "Hide the sample"
            } else {
                "Try the instructions"
            })
            .size(13)
        )
        .on_press(Message::AgentTryOpen)
        .padding([4, 8])
        .style(theme::ghost_button(t)),
    ]
    .spacing(6);
    if edit.try_open {
        try_box = try_box.push(
            text("Sends a sample job to the loaded model with these instructions. It has no tools here, so it can only answer in words.")
                .size(13)
                .color(t.fg_soft),
        );
        try_box = try_box.push(area(
            t,
            &edit.try_job,
            "Review this function: function add(a, b) { return a - b; }",
            Message::AgentTryJob,
            56.0,
        ));
        try_box = try_box.push(
            button(text("Run the sample").size(13))
                .on_press(Message::AgentTryRun)
                .padding([6, 12])
                .style(theme::ghost_button(t)),
        );
        if let Some(out) = &edit.try_out {
            try_box = try_box.push(
                text(out.clone())
                    .size(12)
                    .font(Font::MONOSPACE)
                    .color(t.fg_soft),
            );
        }
    }
    body = body.push(try_box);

    card(t, "Sub-agents-form", body.into())
}

/// The boxes of one group of tools.
fn tool_boxes<'a>(t: Tokens, tools: &[Tool], edit: &'a AgentEdit) -> Element<'a, Message> {
    let mut line = row![].spacing(8);
    for tool in tools {
        line = line.push(tool_box(t, tool, edit.is_checked(&tool.name)));
    }
    line.into()
}

/// The card of `spane-agents`.
pub fn section<'a>(t: Tokens, v: AgentsView<'a>) -> Element<'a, Message> {
    let head = row![
        column![
            text("Sub-agents").size(16),
            text("Helpers that the coding agent can hand a job to. Each works on its own with its own instructions and reports back.")
                .size(13)
                .color(t.fg_soft),
        ]
        .spacing(4),
        Space::new().width(Length::Fill),
        button(text("New sub-agent").size(13))
            .on_press(Message::AgentNew)
            .padding([6, 12])
            .style(theme::primary_button(t)),
    ]
    .spacing(12)
    .align_y(Alignment::Center);

    let mut body = column![head].spacing(12);

    // Built in
    body = body.push(section_head(t, "Built in"));
    for (name, description) in BUILTINS.iter() {
        body = body.push(builtin_row(t, name, description));
    }

    // Your sub-agents
    body = body.push(section_head(t, "Your sub-agents"));
    let own: Vec<&Agent> = v.agents.iter().filter(|agent| !agent.read_only).collect();
    if own.is_empty() {
        body = body.push(
            text("You have not created one yet. A sub-agent is useful for a job you repeat, for example reviewing a change or writing tests. Choose New sub-agent to make one.")
                .size(13)
                .color(t.fg_soft),
        );
    } else {
        for agent in own {
            body = body.push(agent_row(t, agent, v.groups.1, None));
        }
    }

    // From plugins
    let plugins: Vec<&Agent> = v.agents.iter().filter(|agent| agent.read_only).collect();
    if !plugins.is_empty() {
        body = body.push(section_head(t, "From plugins"));
        for agent in plugins {
            body = body.push(agent_row(t, agent, v.groups.1, Some("from a plugin")));
        }
    }

    if !v.notice.is_empty() {
        body = body.push(text(v.notice).size(13).color(t.fg_soft));
    }

    if let Some(edit) = v.editing {
        body = body.push(form(t, &v, edit));
    }

    // The line under the list: the folder the files are written to
    if !v.folders_user.is_empty() {
        body = body.push(
            row![
                text("Stored as one file per sub-agent in").size(13).color(t.fg_soft),
                button(text("the agents folder").size(13))
                    .on_press(Message::AgentPathCopy)
                    .padding([2, 6])
                    .style(theme::ghost_button(t)),
                text("(click to copy the path).").size(13).color(t.fg_soft),
            ]
            .spacing(4)
            .align_y(Alignment::Center),
        );
    }

    card(t, "Sub-agents", body.into())
}

/// One section title of the list (`agents-section`).
fn section_head<'a>(t: Tokens, title: &'a str) -> Element<'a, Message> {
    text(title)
        .size(13)
        .color(t.muted)
        .font(Font {
            weight: iced::font::Weight::Semibold,
            ..Font::DEFAULT
        })
        .into()
}

/// One of the two built-in rows: name, its chip and what it is for — without the buttons the
/// page's own rows have.
fn builtin_row<'a>(t: Tokens, name: &'a str, description: &'a str) -> Element<'a, Message> {
    column![
        row![
            text(name).size(14).color(t.fg),
            chip_label(t, "built in"),
        ]
        .spacing(6)
        .align_y(Alignment::Center),
        text(description).size(13).color(t.fg_soft),
    ]
    .spacing(3)
    .padding([4, 0])
    .into()
}

/// The `ui-input` of the page.
fn input<'a>(t: Tokens, placeholder: &'a str, value: &'a str, typed: fn(String) -> Message) -> Element<'a, Message> {
    text_input(placeholder, value)
        .on_input(typed)
        .size(13)
        .padding([6, 10])
        .font(Font::MONOSPACE)
        .style(move |_, status| text_input::Style {
            background: iced::Background::Color(t.surface_2),
            border: iced::Border {
                radius: 8.0.into(),
                width: 1.0,
                color: if matches!(status, text_input::Status::Focused { .. }) {
                    t.fg_soft
                } else {
                    t.separator
                },
            },
            icon: t.muted,
            placeholder: t.muted,
            value: t.fg,
            selection: t.primary_soft,
        })
        .into()
}

/// One of the text boxes of the form (`textarea`).
fn area<'a>(
    t: Tokens,
    content: &'a text_editor::Content,
    placeholder: &'a str,
    action: fn(text_editor::Action) -> Message,
    height: f32,
) -> Element<'a, Message> {
    text_editor(content)
        .on_action(action)
        .placeholder(placeholder)
        .size(13)
        .padding(8)
        .height(Length::Fixed(height))
        .style(move |_, status| text_editor::Style {
            background: iced::Background::Color(t.surface_2),
            border: iced::Border {
                radius: 8.0.into(),
                width: 1.0,
                color: if matches!(status, text_editor::Status::Focused { .. }) {
                    t.fg_soft
                } else {
                    t.separator
                },
            },
            placeholder: t.muted,
            value: t.fg,
            selection: iced::Color { a: 0.35, ..t.primary },
        })
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(name: &str, permission: &str) -> Tool {
        Tool {
            name: name.to_string(),
            permission: permission.to_string(),
        }
    }

    /// The name rule of the page: `^[a-z0-9][a-z0-9_-]{0,63}$`.
    #[test]
    fn a_name_is_what_the_page_accepts() {
        assert!(valid_name("code-reviewer"));
        assert!(valid_name("a"));
        assert!(valid_name("a1_b2-c3"));
        assert!(!valid_name(""));
        assert!(!valid_name("Code"));
        assert!(!valid_name("-code"));
        assert!(!valid_name("_code"));
        assert!(!valid_name("code reviewer"));
        assert!(!valid_name("code.reviewer"));
        assert!(!valid_name(&"a".repeat(65)));
        assert!(valid_name(&"a".repeat(64)));
    }

    fn names(list: &[Tool]) -> Vec<&str> {
        list.iter().map(|tool| tool.name.as_str()).collect()
    }

    /// The two groups: what only reads, what changes things, each by name, without the tools a
    /// sub-agent may never have and without the ones that start a process for a computer.
    #[test]
    fn the_tools_are_in_the_two_groups_of_the_page() {
        let (read, act) = tool_groups(&[
            tool("read_file", "read"),
            tool("bash", "command"),
            tool("list_dir", "read"),
            tool("write_file", "edit"),
            tool("task", "command"),
            tool("screenshot", "computer"),
            tool("run", "computer"),
        ]);
        assert_eq!(names(&read), vec!["list_dir", "read_file"], "by name");
        assert_eq!(names(&act), vec!["bash", "write_file"]);
        assert!(
            !read.iter().any(|tool| tool.name == "task") && !act.iter().any(|tool| tool.name == "run"),
            "the hidden and the computer tools are not given out"
        );
    }

    /// A sub-agent that may change things is not read-only (`agentReach`).
    #[test]
    fn the_line_of_a_row_says_what_it_may_do() {
        let (_, act) = builtin_groups();
        assert_eq!(reach(None, &act), "reads and searches only");
        assert_eq!(
            reach(Some(&["read_file".to_string()]), &act),
            "reads and searches (1 tools)"
        );
        assert_eq!(
            reach(Some(&["read_file".to_string(), "bash".to_string()]), &act),
            "can change files or run commands (2 tools)"
        );
        assert_eq!(
            reach(Some(&["read_file".to_string()]), &[tool("custom_write", "edit")]),
            "reads and searches (1 tools)",
            "only the tools of the page count as changing"
        );
    }

    /// The boxes that start checked: the agent's own list, or everything that only reads.
    #[test]
    fn the_boxes_start_the_way_the_page_starts_them() {
        let (read, _) = builtin_groups();
        let empty: Vec<String> = Vec::new();
        assert_eq!(
            checked_tools(Some(&empty), &read),
            read.iter().map(|tool| tool.name.clone()).collect::<Vec<_>>(),
            "an agent without a list gets the reading tools"
        );
        let own = vec!["bash".to_string()];
        assert_eq!(checked_tools(Some(&own), &read), own, "its own list is kept");
    }

    /// What the form sends, with the clamps and the words of `agentPayload`.
    #[test]
    fn the_form_sends_what_the_page_sends() {
        let value = payload(
            " code-reviewer ",
            "Reviews\n  a change",
            "Read the diff.",
            &[" bash ".to_string(), "bash".to_string(), String::new()],
            " llama-4 ",
            "900",
            false,
            Some("/home/m/proj"),
            true,
            &[],
        )
        .unwrap();
        assert_eq!(value["name"], serde_json::json!("code-reviewer"));
        assert_eq!(
            value["description"],
            serde_json::json!("Reviews a change"),
            "the runs of whitespace become one space"
        );
        assert_eq!(
            value["tools"],
            serde_json::json!(["bash"]),
            "the names are trimmed and only said once"
        );
        assert_eq!(value["max_steps"], serde_json::json!(60), "60 is the top");
        assert_eq!(value["scope"], serde_json::json!("user"));
        assert_eq!(value["workspace"], serde_json::json!("/home/m/proj"));
        let steps = payload(
            "x",
            "",
            "body",
            &[],
            "",
            "7.9",
            false,
            None,
            true,
            &[],
        )
        .unwrap();
        assert_eq!(steps["max_steps"], serde_json::json!(7), "whole steps");
        let zero = payload("x", "", "body", &[], "", "", false, None, true, &[]).unwrap();
        assert_eq!(zero["max_steps"], serde_json::json!(0), "0 means the default");
    }

    /// The words the page says instead of an answer, in its order.
    #[test]
    fn the_form_says_what_the_page_says() {
        let err = |name: &str, body: &str, workspace: bool, existing: &[&str]| {
            payload(
                name,
                "",
                body,
                &[],
                "",
                "",
                workspace,
                if workspace { None } else { Some("/home/m") },
                true,
                &existing.iter().map(|name| name.to_string()).collect::<Vec<_>>(),
            )
            .unwrap_err()
        };
        assert_eq!(
            err("", "", false, &[]),
            "Give the sub-agent a short name: lowercase letters, digits, - and _ (for example code-reviewer)."
        );
        assert_eq!(
            err("Code", "", false, &[]),
            "Give the sub-agent a short name: lowercase letters, digits, - and _ (for example code-reviewer)."
        );
        assert_eq!(
            err("explore", "x", false, &[]),
            "\"explore\" is a built-in sub-agent. Pick another name."
        );
        assert_eq!(
            err("mine", "", false, &["mine"]),
            "You already have a sub-agent called \"mine\". Pick another name or edit that one."
        );
        assert_eq!(
            err("mine", "   ", false, &[]),
            "Write the instructions: what this sub-agent does and what its report should contain."
        );
        assert_eq!(
            err("mine", "x", true, &[]),
            "Open a folder on the Code page first, or store the sub-agent for all folders."
        );
    }
}
