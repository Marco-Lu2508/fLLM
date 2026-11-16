//! The blocks of the side of the Code page that go with the sessions (row 79): *Tasks*, *Changed
//! files*, *Sessions*, *Memory*, *Permissions* and the style of the agent — `renderCodeSide`,
//! `sessionsHtml`, `memoryHtml`, `permissionsHtml` and the handlers behind their buttons.

use super::*;
use crate::sessions::SavedTodo;
use iced::widget::{column, pick_list, row};

fn semibold() -> Font {
    Font { weight: iced::font::Weight::Semibold, ..Font::default() }
}

fn head<'a>(title: &'a str, count: Option<String>, t: Tokens) -> Element<'a, Message> {
    let mut r = row![text(title).size(14).font(semibold())].spacing(8).align_y(Alignment::Center);
    if let Some(c) = count {
        r = r.push(text(c).size(12).color(t.muted));
    }
    r.into()
}

fn small<'a>(label: &'a str, msg: Option<Message>, t: Tokens) -> Element<'a, Message> {
    button(text(label).size(12)).on_press_maybe(msg).padding([2, 8]).style(theme::ghost_button(t)).into()
}

fn sess(m: SessMsg) -> Message {
    Message::Code(CodeMsg::Sess(m))
}

/// *Tasks*: the list the agent keeps (`todoListHtml`).
pub fn tasks_block<'a>(s: &'a CodeState, t: Tokens) -> Element<'a, Message> {
    let done = s.todos.iter().filter(|x| x.status == TodoStatus::Completed).count();
    let mut b = column![head("Tasks", (!s.todos.is_empty()).then(|| format!("{done}/{}", s.todos.len())), t)].spacing(6);
    if s.todos.is_empty() {
        b = b.push(text("For bigger jobs the agent keeps a task list here.").size(12).color(t.muted));
    }
    for todo in &s.todos {
        let mark = match todo.status {
            TodoStatus::Completed => "✓",
            TodoStatus::InProgress => "●",
            TodoStatus::Pending => "○",
        };
        let colour = if todo.status == TodoStatus::Completed { t.muted } else { t.fg };
        b = b.push(row![text(mark).size(12).color(colour), text(todo.content.as_str()).size(12).color(colour)].spacing(6));
    }
    b.into()
}

/// *Changed files*: the files of the server's list of changes, the newest thirty, and *Rewind ...*.
pub fn changes_block<'a>(s: &'a CodeState, t: Tokens) -> Element<'a, Message> {
    let mut unique: Vec<(&str, &str)> = Vec::new();
    for c in &s.changes {
        if let Some(e) = unique.iter_mut().find(|(p, _)| *p == c.path) {
            e.1 = c.kind.as_str();
        } else {
            unique.push((c.path.as_str(), c.kind.as_str()));
        }
    }
    let mut b = column![head("Changed files", (!unique.is_empty()).then(|| unique.len().to_string()), t)].spacing(4);
    if unique.is_empty() {
        return b.push(text("Nothing changed yet.").size(12).color(t.muted)).into();
    }
    let skip = unique.len().saturating_sub(30);
    for (path, kind) in unique.into_iter().skip(skip) {
        let mut r = row![text(path).size(12).font(Font::MONOSPACE).wrapping(text::Wrapping::None)].spacing(6);
        if !kind.is_empty() {
            r = r.push(text(kind).size(11).color(t.muted));
        }
        b = b.push(r);
    }
    if s.changes_known {
        b = b.push(small("Rewind ...", Some(sess(SessMsg::RewindStart)), t));
    }
    b.into()
}

/// *Sessions*: the saved ones of this folder, searchable, with their four buttons.
pub fn sessions_block<'a>(s: &'a CodeState, t: Tokens) -> Element<'a, Message> {
    let mut b = column![head("Sessions", None, t)].spacing(6);
    if s.sessions.len() > 3 {
        b = b.push(text_input("Search sessions", &s.session_query).size(12).padding([4, 8]).on_input(|v| sess(SessMsg::Search(v))));
    }
    let shown: Vec<&crate::sessions::Record> = crate::sessions::filter_sessions(&s.sessions, &s.session_query).into_iter().take(12).collect();
    if shown.is_empty() {
        let msg = if s.sessions.is_empty() { "Sessions of this folder are saved automatically." } else { "No session matches." };
        return b.push(text(msg).size(12).color(t.muted)).into();
    }
    for rec in shown {
        let current = rec.id == s.session_id;
        let title = if current { s.title_now() } else { rec.shown_title() };
        if let Some((id, draft)) = &s.rename {
            if *id == rec.id {
                b = b.push(
                    row![
                        text_input("Title", draft).size(12).padding([3, 6]).on_input(|v| sess(SessMsg::RenameEdit(v))).on_submit(sess(SessMsg::RenameOk)),
                        small("Rename", Some(sess(SessMsg::RenameOk)), t),
                        small("Cancel", Some(sess(SessMsg::RenameCancel)), t),
                    ]
                    .spacing(4)
                    .align_y(Alignment::Center),
                );
                continue;
            }
        }
        if s.confirm_delete.as_deref() == Some(rec.id.as_str()) {
            b = b.push(
                column![
                    text("Delete this session?").size(12).font(semibold()),
                    text("The conversation is removed from this computer. File changes stay as they are.").size(11).color(t.muted),
                    row![small("Delete", Some(sess(SessMsg::DeleteOk)), t), small("Cancel", Some(sess(SessMsg::DeleteCancel)), t)].spacing(4),
                ]
                .spacing(3),
            );
            continue;
        }
        let id = rec.id.clone();
        let label = button(text(title).size(13).font(if current { semibold() } else { Font::default() }).wrapping(text::Wrapping::None))
            .on_press_maybe((!s.running && !current).then(|| sess(SessMsg::Resume(id.clone()))))
            .padding([2, 4])
            .style(pick_style(t, false, !current));
        let idle = (!s.running).then_some(());
        let acts = row![
            small("Rename", Some(sess(SessMsg::RenameStart(id.clone()))), t),
            small("Fork", idle.map(|_| sess(SessMsg::ForkOf(id.clone()))), t),
            small("Export", Some(sess(SessMsg::ExportOf(id.clone()))), t),
            small("Delete", Some(sess(SessMsg::DeleteAsk(id.clone()))), t),
        ]
        .spacing(2);
        b = b.push(column![label, acts].spacing(1));
    }
    b.into()
}

const MEMORY_TEXT: &str = "Short facts the agent keeps between sessions, one per line. They are part of its prompt. The agent adds entries itself when it learns something lasting; edit or delete them here.";
const RULES_TEXT: &str = "One rule per line: category: pattern, optionally for one agent: [plan] bash: ls*. Categories: bash, edit, network, mcp, read, memory. Risky programs need a rule that names them, for example bash: git status*.";

fn editor<'a>(content: &'a Content, placeholder: &'a str, rows: f32, on: impl Fn(Action) -> SessMsg + 'a, t: Tokens) -> Element<'a, Message> {
    text_editor(content).placeholder(placeholder).on_action(move |a| sess(on(a))).padding(6).size(12).height(Length::Fixed(rows * 18.0)).style(crate::skills::editor_style(t)).into()
}

fn toggle_head<'a>(title: &'a str, count: usize, open: bool, msg: SessMsg, t: Tokens) -> Element<'a, Message> {
    let mut r = row![text(if open { "▾" } else { "▸" }).size(12), text(title).size(14).font(semibold())].spacing(6).align_y(Alignment::Center);
    if count > 0 {
        r = r.push(text(count.to_string()).size(12).color(t.muted));
    }
    button(r).on_press(sess(msg)).padding(0).style(theme::ghost_button(t)).into()
}

/// *Memory*: the facts of this workspace and of everywhere, one per line.
pub fn memory_block<'a>(s: &'a CodeState, t: Tokens) -> Element<'a, Message> {
    let n = s.context.memory_workspace.len() + s.context.memory_global.len();
    let mut b = column![toggle_head("Memory", n, s.mem_open, SessMsg::MemToggle, t)].spacing(6);
    if !s.mem_open {
        return b.into();
    }
    b = b.push(text(MEMORY_TEXT).size(12).color(t.muted));
    b = b.push(text("This workspace (.fllm/memory.md)").size(12));
    b = b.push(editor(&s.mem_ws, "Tests run with: npm test", 5.0, SessMsg::MemEditWs, t));
    b = b.push(text("Everywhere (this user)").size(12));
    b = b.push(editor(&s.mem_gl, "Prefers short answers", 4.0, SessMsg::MemEditGl, t));
    let mut r = row![button(text("Save memory").size(12)).on_press(sess(SessMsg::MemSave)).padding([4, 10]).style(theme::secondary_button(t))].spacing(6);
    if s.mem_dirty.0 || s.mem_dirty.1 {
        r = r.push(small("Discard changes", Some(sess(SessMsg::MemReset)), t));
    }
    b.push(r).into()
}

/// *Permissions*: the allow and deny rules of this folder.
pub fn permissions_block<'a>(s: &'a CodeState, t: Tokens) -> Element<'a, Message> {
    let n = crate::commands::Rules::parse_side(&s.rules.allow).len() + crate::commands::Rules::parse_side(&s.rules.deny).len();
    let mut b = column![toggle_head("Permissions", n, s.perms_open, SessMsg::PermsToggle, t)].spacing(6);
    if !s.perms_open {
        return b.into();
    }
    b = b.push(text("What each agent may do without asking. Pattern rules below refine this: deny beats everything, an allow rule skips the question.").size(12).color(t.muted));
    b = b.push(text(RULES_TEXT).size(12).color(t.muted));
    b = b.push(text("Always allow").size(12));
    b = b.push(editor(&s.rules_allow, "bash: npm test*\nedit: src/**\nnetwork: *", 4.0, SessMsg::RulesEditAllow, t));
    b = b.push(text("Always deny").size(12));
    b = b.push(editor(&s.rules_deny, "bash: rm *\nedit: .env", 3.0, SessMsg::RulesEditDeny, t));
    b.push(button(text("Save rules").size(12)).on_press(sess(SessMsg::RulesSave)).padding([4, 10]).style(theme::secondary_button(t))).into()
}

/// The *Style of the agent* row of the Settings block.
pub fn style_row<'a>(s: &'a CodeState, t: Tokens) -> Element<'a, Message> {
    let labels: Vec<&'static str> = crate::commands::PERSONALITIES.iter().map(|(_, label, _)| *label).collect();
    let current = crate::commands::PERSONALITIES.iter().find(|(k, _, _)| *k == s.personality).map(|(_, l, _)| *l);
    column![
        text("Style of the agent").size(12),
        pick_list(labels, current, |label: &'static str| sess(SessMsg::Personality(label.to_string()))).text_size(13),
        text("Only changes how the agent talks, not what it may do. /personality name does the same.").size(12).color(t.muted),
    ]
    .spacing(4)
    .into()
}

/// The meter of the header: the conversation against the context of the model (`updateCodeMeter`).
pub fn meter_text(s: &CodeState) -> Option<String> {
    if s.ctx_max == 0 || s.items.is_empty() {
        return None;
    }
    let used = s.code_estimate(None);
    Some(format!("Context ~{} of {}", crate::usage::fmt_tokens(used as u64), crate::usage::fmt_tokens(s.ctx_max as u64)))
}

/// The notices with buttons below the conversation (`notice-choices`).
pub fn choices_view<'a>(s: &'a CodeState, t: Tokens) -> Vec<Element<'a, Message>> {
    s.choices
        .iter()
        .enumerate()
        .map(|(at, set)| {
            let mut buttons = column![].spacing(4);
            for (i, (label, _)) in set.rows.iter().enumerate() {
                buttons = buttons.push(button(text(label.as_str()).size(13)).on_press(sess(SessMsg::Choice(at, i))).padding([4, 12]).style(theme::secondary_button(t)));
            }
            column![text(set.text.as_str()).size(13).color(t.muted), buttons].spacing(6).into()
        })
        .collect()
}

impl CodeState {
    pub(super) fn side_update(&mut self, m: SessMsg, env: &Env) -> Task<Message> {
        match m {
            SessMsg::MemToggle => {
                self.mem_open = !self.mem_open;
                if self.mem_open && self.mem_dirty == (false, false) {
                    self.sync_memory_editors();
                }
                Task::none()
            }
            SessMsg::PermsToggle => {
                self.perms_open = !self.perms_open;
                if self.perms_open && !self.rules_dirty {
                    self.sync_rule_editors();
                }
                Task::none()
            }
            SessMsg::MemEditWs(a) => {
                self.mem_dirty.0 |= a.is_edit();
                self.mem_ws.perform(a);
                Task::none()
            }
            SessMsg::MemEditGl(a) => {
                self.mem_dirty.1 |= a.is_edit();
                self.mem_gl.perform(a);
                Task::none()
            }
            SessMsg::MemReset => {
                self.mem_dirty = (false, false);
                self.sync_memory_editors();
                Task::none()
            }
            SessMsg::MemSave => {
                let lines = |c: &Content| -> String {
                    let items: Vec<String> = c.text().split('\n').map(|l| l.trim().trim_start_matches('-').trim().to_string()).filter(|l| !l.is_empty()).collect();
                    format!("# Memory\n\n{}{}", items.iter().map(|l| format!("- {l}")).collect::<Vec<_>>().join("\n"), if items.is_empty() { "" } else { "\n" })
                };
                let mut posts = Vec::new();
                if self.mem_dirty.0 {
                    posts.push(("workspace", lines(&self.mem_ws)));
                }
                if self.mem_dirty.1 {
                    posts.push(("global", lines(&self.mem_gl)));
                }
                if posts.is_empty() {
                    return Task::none();
                }
                let base = env.base.to_string();
                Task::perform(
                    async move {
                        for (scope, text) in posts {
                            call(&base, reqwest::Method::POST, "/api/studio/agent/memory", Some(json!({ "scope": scope, "op": "set", "text": text }))).await.map_err(|e| if e.trim().is_empty() { "The memory could not be saved.".to_string() } else { e })?;
                        }
                        Ok(())
                    },
                    |r| Message::Code(CodeMsg::Sess(SessMsg::MemSaved(r))),
                )
            }
            SessMsg::MemSaved(r) => match r {
                Ok(()) => {
                    self.mem_dirty = (false, false);
                    self.pending_toast = Some("Memory saved".into());
                    Task::perform(fetch_context(env.base.to_string()), |r| Message::Code(CodeMsg::ContextLoaded(r)))
                }
                Err(e) => {
                    self.pending_toast = Some(e);
                    Task::none()
                }
            },
            SessMsg::RulesEditAllow(a) => {
                self.rules_dirty |= a.is_edit();
                self.rules_allow.perform(a);
                Task::none()
            }
            SessMsg::RulesEditDeny(a) => {
                self.rules_dirty |= a.is_edit();
                self.rules_deny.perform(a);
                Task::none()
            }
            SessMsg::RulesSave => {
                let allow = self.rules_allow.text().trim_end_matches('\n').to_string();
                let deny = self.rules_deny.text().trim_end_matches('\n').to_string();
                let n = crate::commands::Rules::parse_side(&allow).len() + crate::commands::Rules::parse_side(&deny).len();
                self.rules = crate::commands::Rules { allow, deny };
                self.pending_prefs.get_or_insert_with(Default::default).rules = Some(self.rules.clone());
                self.rules_dirty = false;
                self.pending_toast = Some(format!("{n} rules saved"));
                Task::none()
            }
            SessMsg::Personality(label) => {
                if let Some((key, _, _)) = crate::commands::PERSONALITIES.iter().find(|(_, l, _)| *l == label) {
                    self.personality = key.to_string();
                    self.pending_prefs.get_or_insert_with(Default::default).personality = Some(key.to_string());
                }
                Task::none()
            }
            _ => Task::none(),
        }
    }

    /// `/skill`: the model writes a skill from this session; nothing is saved before you agree.
    pub(super) fn skill_command(&mut self, env: &Env) -> Task<Message> {
        if self.skill_asking || self.skill.is_some() {
            return Task::none();
        }
        let messages: Vec<crate::skills::SessionMessage> = self
            .wire(None)
            .into_iter()
            .map(|m| crate::skills::SessionMessage {
                calls: m
                    .tool_calls
                    .as_ref()
                    .and_then(|c| c.as_array())
                    .map(|a| a.iter().map(|c| (c.pointer("/function/name").and_then(|n| n.as_str()).unwrap_or("undefined").to_string(), c.pointer("/function/arguments").and_then(|n| n.as_str()).unwrap_or("").to_string())).collect())
                    .unwrap_or_default(),
                role: m.role,
                content: m.content,
            })
            .collect();
        let transcript = crate::skills::transcript(&messages, 14000);
        // the page says nothing at all when there is nothing to learn from yet
        if transcript.trim().is_empty() {
            return Task::none();
        }
        let Some(model) = env.model.map(str::to_string) else {
            self.pending_toast = Some("Could not write a skill: no model is loaded".into());
            return Task::none();
        };
        self.skill_asking = true;
        let request = vec![WireMessage::new("user", crate::skills::proposal_prompt(&transcript))];
        Task::perform(super::session_cmds::ask_raw(env.base.to_string(), model, request, 1600, 0.3), |r| Message::Code(CodeMsg::Sess(SessMsg::SkillWritten(r))))
    }

    /// The dialog *Save this as a skill?* (`skEnsureProposalModal`).
    pub fn skill_window(&self, t: Tokens) -> Option<Element<'_, Message>> {
        let draft = self.skill.as_ref()?;
        let sess = |m: SessMsg| Message::Code(CodeMsg::Sess(m));
        let scope = crate::settings::segmented(
            t,
            vec![
                ("For every project", !draft.workspace, sess(SessMsg::SkillScope(false))),
                ("This folder only", draft.workspace, sess(SessMsg::SkillScope(true))),
            ],
        );
        let scope = if self.workspace.is_some() { scope } else { text("This folder only is not possible without a folder.").size(11).color(t.muted).into() };
        let window = container(
            column![
                text("Save this as a skill?").size(16).font(semibold()),
                text("The model wrote this skill from the session. Edit it if needed; it is only saved when you press Save.").size(12).color(t.muted),
                row![
                    iced::widget::text_input("name", &draft.name).on_input(move |s| Message::Code(CodeMsg::Sess(SessMsg::SkillName(s)))).font(iced::Font::MONOSPACE).size(13).padding([6, 10]),
                    scope,
                ]
                .spacing(8)
                .align_y(iced::Alignment::Center),
                iced::widget::text_editor(&draft.editor).on_action(move |a| Message::Code(CodeMsg::Sess(SessMsg::SkillEdit(a)))).font(iced::Font::MONOSPACE).size(12).height(Length::Fixed(300.0)).style(crate::skills::editor_style(t)),
                row![
                    Space::new().width(Length::Fill),
                    button(text("Discard").size(13)).on_press(sess(SessMsg::SkillDiscard)).padding([6, 12]).style(crate::theme::ghost_button(t)),
                    button(text(if draft.busy { "Saving…" } else { "Save skill" }).size(13)).on_press_maybe((!draft.busy).then(|| sess(SessMsg::SkillSave))).padding([6, 14]).style(crate::theme::primary_button(t)),
                ]
                .spacing(8),
            ]
            .spacing(10),
        )
        .padding(16)
        .width(640)
        .style(crate::theme::card(t, crate::theme::R_LG, false));
        Some(container(iced::widget::opaque(window)).center_x(Length::Fill).center_y(Length::Fill).width(Length::Fill).height(Length::Fill).padding(24).into())
    }
}

#[allow(dead_code)]
fn _todo_type(_: &SavedTodo) {}
