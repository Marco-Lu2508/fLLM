//! The list of chats in the sidebar: search, pinned and archived chats, rename, duplicate, export, delete (one or many) and the date groups.

use crate::app::{ellipsize, icon, muted_svg, Message, ICON_MORE, ICON_PIN};
use crate::store::{ChatRecord, Record, Role, Store};
use crate::theme::{self, Tokens};
use chrono::{Datelike, Local, TimeZone};
use iced::widget::{button, column, container, mouse_area, row, svg, text, text_input, Space};
use iced::{Alignment, Element, Length, Padding};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Markdown,
    Json,
}

#[derive(Debug, Clone)]
pub enum ListMsg {
    Search(String),
    /// Opens or closes the row of actions under a chat.
    Menu(Option<u64>),
    Pin(u64),
    Archive(u64),
    RenameStart(u64),
    RenameInput(String),
    /// `true` keeps the new name.
    RenameEnd(bool),
    Duplicate(u64),
    Export(u64, Format),
    Delete(u64),
    SelectMode(bool),
    Toggle(u64),
    SelectAll,
    DeleteSelected,
    ArchiveSelected,
    /// Asks which project these chats go to (the menu asks for one chat, the select row for all).
    MoveAsk,
    MoveSelected,
    /// The chat menu switches your instructions and memory for this chat (menuPersonal).
    Personal(u64),
    /// The chat menu switches memory for this chat (menuMemory).
    Memory(u64),
    ShowArchived(bool),
}

/// What the window has to do after a message of the list.
#[derive(Debug, Clone, PartialEq)]
pub enum Effect {
    Nothing,
    /// The chats changed: write them.
    Save,
    /// These chats are gone; the open one has to be closed when it is among them.
    Deleted(Vec<u64>),
    /// A line for the user ("Saved to ...").
    Toast(String),
    /// These chats should be moved to a project: the window opens the picker.
    AskProject(Vec<u64>),
    /// The switch of the chat menu went the other way: what it now is (PersonalToggled / MemoryToggled).
    PersonalToggled(bool),
    MemoryToggled(bool),
}

#[derive(Debug, Clone, Default)]
pub struct ChatList {
    pub search: String,
    pub menu: Option<u64>,
    pub renaming: Option<(u64, String)>,
    pub select: bool,
    pub selected: BTreeSet<u64>,
    pub show_archived: bool,
}

/// The day a chat was last used, as the group it belongs to.
pub fn group_of(updated: i64, now: i64) -> &'static str {
    if updated <= 0 {
        return "Older";
    }
    let day = |t: i64| Local.timestamp_opt(t, 0).single().map(|d| d.date_naive().num_days_from_ce());
    let (Some(a), Some(b)) = (day(updated), day(now)) else { return "Older" };
    match b - a {
        i32::MIN..=0 => "Today",
        1 => "Yesterday",
        2..=7 => "Previous 7 days",
        8..=30 => "Previous 30 days",
        _ => "Older",
    }
}

fn matches(c: &ChatRecord, q: &str) -> bool {
    let q = q.to_lowercase();
    c.title.to_lowercase().contains(&q) || c.messages.iter().any(|m| m.content.to_lowercase().contains(&q))
}

pub fn markdown_of(c: &ChatRecord) -> String {
    let mut out = format!("# {}\n\n", c.title);
    for m in &c.messages {
        out.push_str(match m.role {
            Role::User => "## You\n\n",
            Role::Assistant => "## Assistant\n\n",
            // a tool answer belongs to the model, not to the exported text
            Role::Tool => continue,
        });
        out.push_str(m.content.trim());
        out.push_str("\n\n");
    }
    out
}

pub fn json_of(c: &ChatRecord) -> String {
    let messages: Vec<serde_json::Value> = c.messages.iter().map(|m| serde_json::json!({"role": m.role.wire(), "content": m.content})).collect();
    serde_json::to_string_pretty(&serde_json::json!({"title": c.title, "messages": messages})).unwrap_or_default()
}

/// A file name from a title: letters, digits, `-` and `_`.
pub fn file_stem(title: &str) -> String {
    let s: String = title.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '-' }).collect();
    let s = s.split('-').filter(|p| !p.is_empty()).collect::<Vec<_>>().join("-");
    if s.is_empty() { "chat".into() } else { s.chars().take(60).collect() }
}

/// The chats of an imported file: an array of them or one chat, in the shape `json_of` writes
/// (an id that is already taken gets a new one — the `importThreads` of the Studio). The error is
/// the line the Studio shows when the file is no JSON at all.
pub fn parse_import(text: &str, next_id: &mut u64, taken: &std::collections::HashSet<u64>) -> Result<Vec<ChatRecord>, String> {
    let value: serde_json::Value = serde_json::from_str(text).map_err(|_| "The file is not valid JSON.".to_string())?;
    let list = match value {
        serde_json::Value::Array(items) => items,
        other => vec![other],
    };
    let mut out = Vec::new();
    for raw in list {
        let Some(messages) = raw.get("messages").and_then(|m| m.as_array()) else { continue };
        // an id of the file that is free stays, any other chat gets the next one of this window
        // (`state.threads.some((x) => x.id === t.id)` of importThreads)
        let id = match raw.get("id").and_then(|i| i.as_u64()) {
            Some(id) if !taken.contains(&id) => id,
            _ => *next_id,
        };
        *next_id = (*next_id).max(id + 1);
        let chat = ChatRecord::new(
            id,
            raw.get("title").and_then(|t| t.as_str()).unwrap_or(""),
            messages
                .iter()
                .filter_map(|m| {
                    let content = m.get("content")?.as_str()?.to_string();
                    let role = match m.get("role")?.as_str().unwrap_or("assistant") {
                        "user" => Role::User,
                        "tool" => Role::Tool,
                        _ => Role::Assistant,
                    };
                    Some(Record {
                        role,
                        content,
                        reasoning: String::new(),
                        stats: None,
                        passages: None,
                        images: Vec::new(),
                        files: Vec::new(),
                        tool_calls: None,
                        tool_call_id: None,
                        tool_cards: Vec::new(),
                        sources: Vec::new(),
                        research: None,
                        logprobs: Vec::new(),
                    })
                })
                .collect(),
        );
        out.push(chat);
    }
    Ok(out)
}

pub fn export_dir() -> std::path::PathBuf {
    if let Some(dir) = std::env::var_os("FLLM_EXPORT_DIR") {
        return dir.into();
    }
    dirs::download_dir().or_else(dirs::home_dir).unwrap_or_else(|| std::path::PathBuf::from("."))
}

impl ChatList {
    /// The chats to show, in groups: pinned first, then by the day of the last use. A search looks in titles and messages of every chat.
    pub fn groups<'a>(&self, store: &'a Store, now: i64) -> (Vec<(String, Vec<&'a ChatRecord>)>, usize) {
        let q = self.search.trim();
        let mut live: Vec<&ChatRecord> = Vec::new();
        let mut archived: Vec<&ChatRecord> = Vec::new();
        for c in &store.chats {
            if !q.is_empty() && !matches(c, q) {
                continue;
            }
            if c.archived {
                // an archived chat stays reachable under Archived, project chats included (the Studio's toast says so)
                archived.push(c);
                continue;
            }
            // the chats of a project are shown under their project, not here (the Studio's prjSidebarView);
            // a search looks through every chat, project chats included
            if q.is_empty() && c.project.is_some() {
                continue;
            }
            live.push(c)
        }
        let by_use = |v: &mut Vec<&ChatRecord>| v.sort_by_key(|c| std::cmp::Reverse(c.updated));
        by_use(&mut live);
        by_use(&mut archived);
        let archived_count = archived.len();
        let mut groups: Vec<(String, Vec<&ChatRecord>)> = Vec::new();
        let pinned: Vec<&ChatRecord> = live.iter().copied().filter(|c| c.pinned).collect();
        if !pinned.is_empty() {
            groups.push(("Pinned".into(), pinned));
        }
        for c in live.iter().copied().filter(|c| !c.pinned) {
            let g = group_of(c.updated, now);
            match groups.last_mut() {
                Some((name, list)) if name == g => list.push(c),
                _ => groups.push((g.to_string(), vec![c])),
            }
        }
        if !archived.is_empty() && (self.show_archived || !q.is_empty()) {
            groups.push(("Archived".into(), archived));
        }
        (groups, archived_count)
    }

    pub fn update(&mut self, msg: ListMsg, store: &mut Store, live: Option<u64>, now: i64) -> Effect {
        match msg {
            ListMsg::Search(q) => {
                self.search = q;
                Effect::Nothing
            }
            ListMsg::Menu(id) => {
                self.menu = if self.menu == id { None } else { id };
                self.renaming = None;
                Effect::Nothing
            }
            ListMsg::Pin(id) => {
                if let Some(c) = store.chat_mut(id) {
                    c.pinned = !c.pinned;
                }
                self.menu = None;
                Effect::Save
            }
            ListMsg::Archive(id) => {
                if let Some(c) = store.chat_mut(id) {
                    c.archived = !c.archived;
                    c.pinned = false;
                }
                self.menu = None;
                Effect::Save
            }
            ListMsg::RenameStart(id) => {
                let title = store.chat(id).map(|c| c.title.clone()).unwrap_or_default();
                self.renaming = Some((id, title));
                self.menu = None;
                Effect::Nothing
            }
            ListMsg::RenameInput(t) => {
                if let Some((_, name)) = &mut self.renaming {
                    *name = t.chars().take(120).collect();
                }
                Effect::Nothing
            }
            ListMsg::RenameEnd(keep) => {
                let Some((id, name)) = self.renaming.take() else { return Effect::Nothing };
                let name = name.trim();
                if keep && !name.is_empty() {
                    if let Some(c) = store.chat_mut(id) {
                        c.title = name.to_string();
                        return Effect::Save;
                    }
                }
                Effect::Nothing
            }
            ListMsg::Duplicate(id) => {
                self.menu = None;
                let Some(src) = store.chat(id).cloned() else { return Effect::Nothing };
                let new = store.new_chat();
                if let Some(c) = store.chat_mut(new) {
                    c.title = format!("{} (copy)", src.title);
                    c.messages = src.messages;
                    c.updated = now;
                }
                Effect::Save
            }
            ListMsg::Export(id, format) => {
                self.menu = None;
                let Some(c) = store.chat(id) else { return Effect::Nothing };
                let (body, ext) = match format {
                    Format::Markdown => (markdown_of(c), "md"),
                    Format::Json => (json_of(c), "json"),
                };
                let path = export_dir().join(format!("fllm-{}-{}.{ext}", file_stem(&c.title), Local.timestamp_opt(now, 0).single().map(|d| d.format("%Y-%m-%d").to_string()).unwrap_or_default()));
                match std::fs::write(&path, body) {
                    Ok(()) => Effect::Toast(format!("Saved to {}", path.display())),
                    Err(e) => Effect::Toast(format!("The chat could not be saved to {}: {e}", path.display())),
                }
            }
            ListMsg::Delete(id) => {
                if live == Some(id) {
                    return Effect::Nothing;
                }
                self.menu = None;
                store.chats.retain(|c| c.id != id);
                self.selected.remove(&id);
                Effect::Deleted(vec![id])
            }
            ListMsg::SelectMode(on) => {
                self.select = on;
                self.selected.clear();
                self.menu = None;
                Effect::Nothing
            }
            ListMsg::Toggle(id) => {
                if !self.selected.remove(&id) {
                    self.selected.insert(id);
                }
                Effect::Nothing
            }
            ListMsg::SelectAll => {
                let all: Vec<u64> = self.groups(store, now).0.iter().flat_map(|(_, l)| l.iter().map(|c| c.id)).collect();
                if all.iter().all(|id| self.selected.contains(id)) {
                    self.selected.clear();
                } else {
                    self.selected.extend(all);
                }
                Effect::Nothing
            }
            ListMsg::DeleteSelected => {
                let ids: Vec<u64> = self.selected.iter().copied().filter(|id| live != Some(*id)).collect();
                store.chats.retain(|c| !ids.contains(&c.id));
                self.selected.clear();
                self.select = false;
                Effect::Deleted(ids)
            }
            ListMsg::ArchiveSelected => {
                for c in store.chats.iter_mut().filter(|c| self.selected.contains(&c.id)) {
                    c.archived = true;
                    c.pinned = false;
                }
                self.selected.clear();
                self.select = false;
                Effect::Save
            }
            ListMsg::MoveAsk => {
                let ids = self.menu.map(|id| vec![id]).unwrap_or_default();
                self.menu = None;
                if ids.is_empty() { Effect::Nothing } else { Effect::AskProject(ids) }
            }
            ListMsg::MoveSelected => {
                let ids: Vec<u64> = self.selected.iter().copied().collect();
                self.selected.clear();
                self.select = false;
                if ids.is_empty() { Effect::Nothing } else { Effect::AskProject(ids) }
            }
            ListMsg::Personal(id) => {
                let off = store.chat_mut(id).map(|c| {
                    c.no_personal = !c.no_personal;
                    c.no_personal
                });
                self.menu = None;
                off.map(Effect::PersonalToggled).unwrap_or(Effect::Nothing)
            }
            ListMsg::Memory(id) => {
                let off = store.chat_mut(id).map(|c| {
                    c.no_memory = !c.no_memory;
                    c.no_memory
                });
                self.menu = None;
                off.map(Effect::MemoryToggled).unwrap_or(Effect::Nothing)
            }
            ListMsg::ShowArchived(on) => {
                self.show_archived = on;
                Effect::Nothing
            }
        }
    }

    /// The part of the sidebar under "Chats": search box, groups, rows.
    pub fn view<'a>(&'a self, t: Tokens, store: &'a Store, current: Option<u64>, hover: Option<u64>, can_act: bool, now: i64) -> Element<'a, Message> {
        let (groups, archived_count) = self.groups(store, now);
        let search = text_input("Search chats", &self.search)
            .on_input(|s| Message::List(ListMsg::Search(s)))
            .size(13)
            .padding([6, 10])
            .style(move |_, status| text_input::Style {
                background: iced::Background::Color(t.fill),
                border: iced::Border { radius: 8.0.into(), width: 1.0, color: if matches!(status, text_input::Status::Focused { .. }) { t.fg_soft } else { iced::Color::TRANSPARENT } },
                icon: t.muted,
                placeholder: t.muted,
                value: t.fg,
                selection: t.primary_soft,
            });
        let mode_button = button(text(if self.select { "Cancel" } else { "Select" }).size(12))
            .on_press(Message::List(ListMsg::SelectMode(!self.select)))
            .padding([2, 8])
            .style(theme::ghost_button(t));
        // the import beside it (the btnImportThreads of the Studio's chat header)
        let import_button = button(text("Import").size(12))
            .on_press(Message::ImportChats)
            .padding([2, 8])
            .style(theme::ghost_button(t));
        let mut col = column![row![search, mode_button, import_button].spacing(6).align_y(Alignment::Center)].spacing(6);
        if self.select {
            let n = self.selected.len();
            col = col.push(
                row![
                    button(text("All").size(12)).on_press(Message::List(ListMsg::SelectAll)).padding([3, 8]).style(theme::secondary_button(t)),
                    button(text(format!("Move {n}")).size(12)).on_press_maybe((n > 0).then_some(Message::List(ListMsg::MoveSelected))).padding([3, 8]).style(theme::secondary_button(t)),
                    button(text(format!("Archive {n}")).size(12)).on_press_maybe((n > 0).then_some(Message::List(ListMsg::ArchiveSelected))).padding([3, 8]).style(theme::secondary_button(t)),
                    button(text(format!("Delete {n}")).size(12)).on_press_maybe((n > 0 && can_act).then_some(Message::List(ListMsg::DeleteSelected))).padding([3, 8]).style(theme::secondary_button(t)),
                ]
                .spacing(6),
            );
        }
        let mut list = column![].spacing(2);
        if groups.is_empty() {
            let msg = if self.search.trim().is_empty() { "No chats yet" } else { "No chat matches" };
            list = list.push(container(text(msg).size(13).color(t.muted)).padding([6, 10]));
        }
        for (name, chats) in groups {
            list = list.push(container(text(name).size(11).color(t.muted)).padding(Padding { top: 8.0, right: 10.0, bottom: 2.0, left: 10.0 }));
            for c in chats {
                list = list.push(self.row(t, c, current, hover, can_act));
            }
        }
        if archived_count > 0 && self.search.trim().is_empty() {
            list = list.push(
                button(text(if self.show_archived { "Hide archived".to_string() } else { format!("Archived ({archived_count})") }).size(12))
                    .on_press(Message::List(ListMsg::ShowArchived(!self.show_archived)))
                    .padding([4, 10])
                    .style(theme::ghost_button(t)),
            );
        }
        col.push(iced::widget::scrollable(list).height(Length::Fill)).height(Length::Fill).into()
    }

    fn row<'a>(&'a self, t: Tokens, c: &'a ChatRecord, current: Option<u64>, hover: Option<u64>, can_act: bool) -> Element<'a, Message> {
        let selected = current == Some(c.id);
        let hovered = hover == Some(c.id);
        let open_menu = self.menu == Some(c.id);
        if let Some((id, name)) = &self.renaming {
            if *id == c.id {
                let input = text_input("Name", name)
                    .on_input(|s| Message::List(ListMsg::RenameInput(s)))
                    .on_submit(Message::List(ListMsg::RenameEnd(true)))
                    .size(14)
                    .padding([6, 8])
                    .style(move |_, _| text_input::Style {
                        background: iced::Background::Color(t.surface_2),
                        border: iced::Border { radius: 8.0.into(), width: 1.0, color: t.fg_soft },
                        icon: t.muted,
                        placeholder: t.muted,
                        value: t.fg,
                        selection: t.primary_soft,
                    });
                return row![
                    input,
                    button(text("OK").size(12)).on_press(Message::List(ListMsg::RenameEnd(true))).padding([4, 8]).style(theme::secondary_button(t)),
                    button(text("×").size(14)).on_press(Message::List(ListMsg::RenameEnd(false))).padding([2, 8]).style(theme::ghost_button(t)),
                ]
                .spacing(4)
                .align_y(Alignment::Center)
                .into();
            }
        }
        let title = {
            let shown = ellipsize(&c.title, 24);
            if c.pinned {
                row![svg(icon(ICON_PIN)).width(12).height(12).style(muted_svg(t)), text(shown).size(14).wrapping(text::Wrapping::None)].spacing(6).align_y(Alignment::Center)
            } else {
                row![text(shown).size(14).wrapping(text::Wrapping::None)]
            }
        };
        let main_press = if self.select { Some(Message::List(ListMsg::Toggle(c.id))) } else { can_act.then_some(Message::OpenChat(c.id)) };
        let lead: Element<'a, Message> = if self.select {
            let on = self.selected.contains(&c.id);
            container(Space::new().width(12).height(12))
                .style(move |_| container::Style {
                    background: Some(iced::Background::Color(if on { t.primary } else { iced::Color::TRANSPARENT })),
                    border: iced::Border { radius: 3.0.into(), width: 1.5, color: if on { t.primary } else { t.fg_soft } },
                    ..Default::default()
                })
                .into()
        } else {
            Space::new().width(0).into()
        };
        let more: Element<'a, Message> = if !self.select && (selected || hovered || open_menu) {
            button(svg(icon(ICON_MORE)).width(14).height(14).style(muted_svg(t)))
                .on_press(Message::List(ListMsg::Menu(Some(c.id))))
                .padding(5)
                .style(theme::ghost_button(t))
                .into()
        } else {
            Space::new().width(24).height(24).into()
        };
        let line = row![
            lead,
            button(title).on_press_maybe(main_press).padding([7, 10]).width(Length::Fill).style(theme::plain_button(t)),
            more,
        ]
        .align_y(Alignment::Center);
        let mut block = column![container(line).padding(Padding { top: 0.0, right: 4.0, bottom: 0.0, left: 4.0 })];
        if open_menu {
            let item = |label: &'static str, msg: ListMsg, enabled: bool| button(text(label).size(12)).on_press_maybe(enabled.then_some(Message::List(msg))).padding([3, 8]).style(theme::secondary_button(t));
            block = block.push(
                container(
                    column![
                        row![
                            item(if c.pinned { "Unpin" } else { "Pin" }, ListMsg::Pin(c.id), true),
                            item("Rename", ListMsg::RenameStart(c.id), true),
                            item(if c.archived { "Unarchive" } else { "Archive" }, ListMsg::Archive(c.id), true),
                            item("Move", ListMsg::MoveAsk, true),
                            item(
                                if c.no_personal { "Personalization: off" } else { "Personalization: on" },
                                ListMsg::Personal(c.id),
                                true,
                            ),
                            item(if c.no_memory { "Memory: off" } else { "Memory: on" }, ListMsg::Memory(c.id), true),
                        ]
                        .spacing(4),
                        row![
                            item("Duplicate", ListMsg::Duplicate(c.id), true),
                            item("Markdown", ListMsg::Export(c.id, Format::Markdown), true),
                            item("JSON", ListMsg::Export(c.id, Format::Json), true),
                        ]
                        .spacing(4),
                        row![button(text("Delete").size(12))
                            .on_press_maybe(can_act.then_some(Message::List(ListMsg::Delete(c.id))))
                            .padding([3, 8])
                            .style(move |_, status| {
                                let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
                                button::Style {
                                    background: Some(iced::Background::Color(if hovered { iced::Color { a: 0.16, ..t.danger } } else { t.fill })),
                                    text_color: t.danger,
                                    border: iced::border::rounded(8),
                                    ..Default::default()
                                }
                            })]
                        .spacing(4),
                    ]
                    .spacing(4),
                )
                .padding(Padding { top: 2.0, right: 8.0, bottom: 6.0, left: 12.0 }),
            );
        }
        let framed = container(block).style(move |_| container::Style {
            background: if selected { Some(iced::Background::Color(t.primary_soft)) } else if hovered || open_menu { Some(iced::Background::Color(t.fill)) } else { None },
            border: iced::border::rounded(theme::R_SM),
            ..Default::default()
        });
        mouse_area(framed).on_enter(Message::Hover(Some(c.id))).on_exit(Message::Hover(None)).into()
    }
}
