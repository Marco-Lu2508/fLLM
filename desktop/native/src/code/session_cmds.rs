//! Row 79: the commands over the sessions of a folder — `/resume`, `/sessions`, `/fork`, `/rename`,
//! `/export`, `/revert`, `/retry`, `/rewind`, `/compact`, `/model` — and the title the model gives
//! a session. They are the page's `resumeSession`, `forkSession`, `renameSession`, `exportSession`,
//! `revertToMessage`, `unrevert`, `showRewind`, `rewindTo`, `compactSession` and `generateTitle`.

use super::*;
use crate::sessions::{self, Mark, SavedTodo};

#[derive(Debug, Clone)]
pub enum SessMsg {
    Nop,
    /// A button of a notice with choices (the row of the list below the conversation).
    Choice(usize, usize),
    /// The *Fork* and *Revert* buttons of one of your messages (the index of the item).
    TurnFork(usize),
    TurnRevert(usize),
    Resume(String),
    ForkOf(String),
    ExportOf(String),
    Search(String),
    RenameStart(String),
    RenameEdit(String),
    RenameOk,
    RenameCancel,
    DeleteAsk(String),
    DeleteOk,
    DeleteCancel,
    RewindStart,
    RewindList(Result<Vec<Change>, String>),
    RewindDone(Result<Value, String>),
    RevertList(usize, Result<Vec<Change>, String>),
    RevertUndone(usize, Result<Value, String>),
    UnrevertDone(Result<Value, String>),
    Compacted { start: usize, before: usize, resume: bool, note: usize, result: Result<String, String> },
    TitleMade(String, Result<String, String>),
    MemToggle,
    PermsToggle,
    MemEditWs(Action),
    MemEditGl(Action),
    MemSave,
    MemReset,
    MemSaved(Result<(), String>),
    RulesEditAllow(Action),
    RulesEditDeny(Action),
    RulesSave,
    Personality(String),
}

/// `new Date(time).toLocaleString()` the way an English page shows it.
pub fn locale_time(ms: f64) -> String {
    use chrono::TimeZone;
    chrono::Local.timestamp_millis_opt(ms as i64).single().map(|d| d.format("%-m/%-d/%Y, %-I:%M:%S %p").to_string()).unwrap_or_default()
}

fn now_ms() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_millis() as f64)
}

/// `GET /api/studio/agent/changes` and `POST …/undo` answer in the page's `{ ok, output, restored, error }`.
async fn undo_call(base: String, body: Value) -> Result<Value, String> {
    call(&base, reqwest::Method::POST, "/api/studio/agent/undo", Some(body)).await
}

/// One answer of the model that is no stream (`codeChat` with `stream: false`).
async fn ask_once(base: String, model: String, messages: Vec<WireMessage>, max_tokens: usize) -> Result<String, String> {
    let body = json!({ "model": model, "messages": messages, "stream": false, "temperature": 0.2, "max_tokens": max_tokens });
    let resp = reqwest::Client::new().post(format!("{base}/v1/chat/completions")).json(&body).send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("The server answered with error {}", resp.status().as_u16()));
    }
    let data: Value = resp.json().await.map_err(|e| e.to_string())?;
    let text = data.pointer("/choices/0/message/content").and_then(Value::as_str).unwrap_or("");
    Ok(sessions::answer_of(text).trim().to_string())
}

fn export_dir() -> std::path::PathBuf {
    std::env::var_os("FLLM_EXPORT_DIR").map(std::path::PathBuf::from).or_else(dirs::download_dir).or_else(dirs::document_dir).or_else(dirs::home_dir).unwrap_or_else(|| std::path::PathBuf::from("."))
}

impl CodeState {
    /// `sessionTitleNow()`: the title it has, else the first words of the first message.
    pub fn title_now(&self) -> String {
        if self.title.is_empty() {
            sessions::heuristic_title(sessions::first_user_text(&self.saved_all()).as_deref())
        } else {
            self.title.clone()
        }
    }

    pub(super) fn note(&mut self, text: impl Into<String>) {
        self.items.push(Item::Note(text.into()));
    }

    pub(super) fn push_choices(&mut self, text: &str, action: ChoiceAction, rows: Vec<(String, String)>) {
        self.choices.push(Choices { text: text.to_string(), action, rows });
    }

    /// Whether the conversation can go back to before the message at this item (it has a mark and
    /// was not summarised).
    pub fn turn_allowed(&self, item: usize) -> bool {
        self.marks.iter().any(|m| m.item == item) && self.compaction.as_ref().is_none_or(|c| item >= c.from_item) && matches!(self.items.get(item), Some(Item::User { .. }))
    }

    /// `turnChoices()`: your messages of this session, newest first, at most fifteen.
    pub fn turn_rows(&self) -> Vec<(String, String)> {
        self.items
            .iter()
            .enumerate()
            .rev()
            .filter(|(i, _)| self.turn_allowed(*i))
            .take(15)
            .filter_map(|(i, it)| if let Item::User { text, .. } = it { Some((crate::computer::slice_units(&text.split_whitespace().collect::<Vec<_>>().join(" "), 80), i.to_string())) } else { None })
            .collect()
    }

    /// `cutBefore(view, messages, itemId)`: the session as it was before the message at `item`.
    fn cut_prefix(&self, item: usize) -> Option<(Vec<Saved>, Vec<SavedTodo>, Vec<Mark>, String)> {
        if !self.turn_allowed(item) {
            return None;
        }
        let mark = self.marks.iter().find(|m| m.item == item)?;
        let text = if let Some(Item::User { text, .. }) = self.items.get(item) { text.clone() } else { return None };
        let at = self.saved_index(item);
        let items = self.saved_all()[..at].to_vec();
        let marks = self.marks.iter().filter(|m| m.item < item).map(|m| Mark { item: self.saved_index(m.item), ..m.clone() }).collect();
        Some((items, mark.todos.clone(), marks, text))
    }

    // ---- the commands --------------------------------------------------------------------------

    pub(super) fn session_step(&mut self, step: crate::commands::Step, env: &Env) -> Task<Message> {
        use crate::commands::Step;
        match step {
            Step::Sessions => {
                let rows: Vec<(String, String)> =
                    self.sessions.iter().filter(|s| s.id != self.session_id).map(|s| (format!("{}  ({})", s.shown_title(), locale_time(s.time)), s.id.clone())).collect();
                if rows.is_empty() {
                    self.note("There is no earlier session for this folder.");
                } else {
                    self.push_choices("Pick a session to continue:", ChoiceAction::Resume, rows);
                }
                Task::none()
            }
            Step::Fork => {
                let turns = self.turn_rows();
                if turns.is_empty() {
                    return self.fork(None, env);
                }
                let mut rows = vec![("The end of this conversation".to_string(), String::new())];
                rows.extend(turns);
                self.push_choices("Fork: the new session continues from before the message you pick (its text goes back to the input).", ChoiceAction::Fork, rows);
                Task::none()
            }
            Step::Revert => {
                let rows = self.turn_rows();
                if rows.is_empty() {
                    self.note("There is no message to revert to.");
                } else {
                    self.push_choices("Revert: pick the message to go back to. The conversation and the file changes since then are taken back; you can undo the revert.", ChoiceAction::Revert, rows);
                }
                Task::none()
            }
            Step::Rename(Some(arg)) => {
                self.title = sessions::clean_title(&arg);
                self.title_locked = true;
                self.wants_save = true;
                Task::none()
            }
            Step::Rename(None) => {
                if self.session_id.is_empty() && !self.items.is_empty() {
                    self.session_id = sessions::new_id();
                }
                if !self.session_id.is_empty() {
                    self.rename = Some((self.session_id.clone(), self.title_now()));
                }
                Task::none()
            }
            Step::Export => self.export(&self.session_id.clone(), env),
            Step::Rewind => self.sess_update(SessMsg::RewindStart, env),
            Step::Compact(focus) => self.compact_command(focus, env),
            Step::PickModel(names) => {
                if names.is_empty() {
                    self.note("No model is loaded.");
                } else {
                    self.push_choices("Pick the model for the agent:", ChoiceAction::Model, names.into_iter().map(|n| (n.clone(), n)).collect());
                }
                Task::none()
            }
            Step::Memory => {
                self.mem_open = true;
                self.mem_dirty = (false, false);
                self.sync_memory_editors();
                Task::none()
            }
            Step::Retry => {
                let last = self.items.iter().enumerate().rev().find(|(i, it)| matches!(it, Item::User { .. }) && self.turn_allowed(*i)).map(|(i, _)| i);
                match last {
                    None => {
                        self.note("There is no message to retry.");
                        Task::none()
                    }
                    Some(i) => {
                        self.retry_pending = true;
                        self.start_revert(i, env)
                    }
                }
            }
            Step::Skill => self.skill_command(env),
            _ => Task::none(),
        }
    }

    pub fn sync_memory_editors(&mut self) {
        self.mem_ws = Content::with_text(&self.context.memory_workspace.join("\n"));
        self.mem_gl = Content::with_text(&self.context.memory_global.join("\n"));
    }

    pub fn sync_rule_editors(&mut self) {
        self.rules_allow = Content::with_text(&self.rules.allow);
        self.rules_deny = Content::with_text(&self.rules.deny);
        self.rules_dirty = false;
    }

    /// `resumeSession(id)`.
    fn resume(&mut self, id: &str, env: &Env) -> Task<Message> {
        if self.running {
            self.pending_toast = Some("Stop the agent first.".into());
            return Task::none();
        }
        self.persist(env.session_file);
        let Some(rec) = self.sessions.iter().find(|s| s.id == id).cloned() else {
            self.pending_toast = Some("That session is gone.".into());
            return Task::none();
        };
        self.apply_record(rec);
        Task::none()
    }

    /// `forkSession(itemId)`: a new session from before one of your messages, or from the end.
    fn fork(&mut self, at: Option<usize>, env: &Env) -> Task<Message> {
        if self.running {
            self.pending_toast = Some("Stop the agent first.".into());
            return Task::none();
        }
        let (items, todos, marks, text) = match at {
            Some(i) => match self.cut_prefix(i) {
                Some(cut) => cut,
                None => {
                    self.pending_toast = Some("That part of the conversation was compacted. Fork from a later message.".into());
                    return Task::none();
                }
            },
            None => (self.saved_all(), self.todos.iter().map(todo_saved).collect(), self.marks.iter().map(|m| Mark { item: self.saved_index(m.item), ..m.clone() }).collect(), String::new()),
        };
        self.persist(env.session_file);
        let parent = self.title_now();
        let base = parent.strip_suffix(" (fork)").unwrap_or(&parent).to_string();
        let compaction = self.compaction.clone();
        self.reset_session();
        self.session_id = sessions::new_id();
        self.title = crate::computer::slice_units(&format!("{base} (fork)"), 64);
        self.title_locked = true;
        self.marks = marks;
        self.compaction = compaction;
        self.restore(items);
        self.todos = todos.iter().map(todo_loaded).collect();
        self.persist(env.session_file);
        if !text.is_empty() {
            self.editor = Content::with_text(&text);
        }
        self.note(format!("Forked from \"{parent}\". The original session is unchanged."));
        Task::none()
    }

    /// `exportSession(id)`: the Markdown goes to the downloads folder (the page offers a download).
    fn export(&mut self, id: &str, _env: &Env) -> Task<Message> {
        let rec = if id.is_empty() || id == self.session_id {
            self.record().map(|r| (r.title, r.items))
        } else {
            self.sessions.iter().find(|s| s.id == id).map(|s| (s.shown_title(), s.items.clone()))
        };
        let Some((title, items)) = rec.filter(|(_, items)| !items.is_empty()) else {
            self.pending_toast = Some("Nothing to export.".into());
            return Task::none();
        };
        let root = self.workspace.as_ref().map(|w| w.root.clone()).unwrap_or_default();
        let md = sessions::export_markdown(&title, now_ms(), &items, &root);
        let path = export_dir().join(format!("{}.md", sessions::file_safe(&title)));
        match std::fs::write(&path, md) {
            Ok(()) => self.pending_toast = Some(format!("Saved {}", path.display())),
            Err(e) => self.pending_toast = Some(format!("The file could not be written: {e}")),
        }
        Task::none()
    }

    fn refresh_changes(&self, env: &Env) -> Task<Message> {
        Task::perform(fetch_changes(env.base.to_string()), |r| Message::Code(CodeMsg::ChangesLoaded(r)))
    }

    // ---- revert --------------------------------------------------------------------------------

    /// `revertToMessage(itemId)`, the first half: the changes are read to find where the files go back to.
    fn start_revert(&mut self, item: usize, env: &Env) -> Task<Message> {
        if self.running {
            self.retry_pending = false;
            self.pending_toast = Some("Stop the agent first.".into());
            return Task::none();
        }
        if !self.turn_allowed(item) {
            self.retry_pending = false;
            self.pending_toast = Some("That part of the conversation was compacted. Revert to a later message.".into());
            return Task::none();
        }
        Task::perform(fetch_changes(env.base.to_string()), move |r| Message::Code(CodeMsg::Sess(SessMsg::RevertList(item, r))))
    }

    /// The second half: the conversation is cut and the text goes back to the box.
    fn apply_revert(&mut self, item: usize, restored: usize, env: &Env) -> Task<Message> {
        let Some(mark) = self.marks.iter().find(|m| m.item == item).cloned() else { return Task::none() };
        let text = if let Some(Item::User { text, .. }) = self.items.get(item) { text.clone() } else { return Task::none() };
        self.reverted = Some(Reverted { items: self.saved_all(), todos: self.todos.iter().map(todo_saved).collect(), marks: self.marks.clone(), compaction: self.compaction.clone(), restored, title: self.title.clone() });
        self.items.truncate(item);
        self.marks.retain(|m| m.item < item);
        self.todos = mark.todos.iter().map(todo_loaded).collect();
        self.choices.clear();
        self.persist(env.session_file);
        self.editor = Content::with_text(&text);
        let moved = if restored > 0 { format!(" and put {restored} file change{} back", if restored == 1 { "" } else { "s" }) } else { String::new() };
        self.push_choices(&format!("Reverted to before that message{moved}. Its text is in the input."), ChoiceAction::Unrevert, vec![("Undo the revert".into(), "unrevert".into())]);
        let refresh = self.refresh_changes(env);
        if std::mem::take(&mut self.retry_pending) && !text.trim().is_empty() && !self.running {
            return Task::batch([refresh, self.update(CodeMsg::Send, env)]);
        }
        refresh
    }

    /// `unrevert()`.
    fn start_unrevert(&mut self, env: &Env) -> Task<Message> {
        let Some(r) = &self.reverted else {
            self.pending_toast = Some("Nothing to restore.".into());
            return Task::none();
        };
        if self.running {
            self.pending_toast = Some("Stop the agent first.".into());
            return Task::none();
        }
        if r.restored > 0 {
            let n = r.restored;
            return Task::perform(undo_call(env.base.to_string(), json!({ "redo": n })), |r| Message::Code(CodeMsg::Sess(SessMsg::UnrevertDone(r))));
        }
        self.apply_unrevert(env)
    }

    fn apply_unrevert(&mut self, env: &Env) -> Task<Message> {
        let Some(r) = self.reverted.take() else { return Task::none() };
        self.items.clear();
        self.marks = r.marks;
        self.compaction = r.compaction;
        self.restore(r.items);
        self.todos = r.todos.iter().map(todo_loaded).collect();
        self.editor = Content::new();
        self.choices.clear();
        self.persist(env.session_file);
        self.note("The revert was undone.");
        self.refresh_changes(env)
    }

    // ---- rewind --------------------------------------------------------------------------------

    fn rewind_rows(&self) -> Vec<(String, String)> {
        use chrono::TimeZone;
        self.changes
            .iter()
            .rev()
            .take(15)
            .map(|c| {
                let when = if c.time > 0.0 {
                    let ms = if c.time > 1e12 { c.time } else { c.time * 1000.0 };
                    chrono::Local.timestamp_millis_opt(ms as i64).single().map(|d| format!("  ({})", d.format("%-I:%M:%S %p"))).unwrap_or_default()
                } else {
                    String::new()
                };
                (format!("{} {}{}", if c.kind.is_empty() { "change" } else { c.kind.as_str() }, c.path, when), c.id.to_string())
            })
            .collect()
    }

    // ---- compaction ----------------------------------------------------------------------------

    /// `codeEstimate()`: the system prompt, the tool list and the messages in tokens.
    pub fn code_estimate(&self, extra: Option<&str>) -> usize {
        let wire = self.wire(extra);
        sessions::estimate_tokens(&wire[0].content) + sessions::estimate_tokens(&self.tools_json.to_string()) + sessions::estimate_messages(&wire[1..])
    }

    fn compact_command(&mut self, focus: String, env: &Env) -> Task<Message> {
        if self.workspace.is_none() || env.model.is_none() {
            return Task::none();
        }
        match self.begin_compaction(focus, env, false) {
            Some(task) => task,
            None => {
                self.note("There is not enough conversation to compact yet.");
                Task::none()
            }
        }
    }

    /// The summary is asked for; `None` when there is nothing to summarise (`splitForCompaction`).
    pub(super) fn begin_compaction(&mut self, focus: String, env: &Env, resume: bool) -> Option<Task<Message>> {
        let model = env.model?.to_string();
        let wire = self.wire(env.extra);
        let start = sessions::split_for_compaction(&wire[1..], 6, 3)?;
        let before = self.code_estimate(env.extra);
        let mut messages = vec![wire[0].clone()];
        messages.extend(wire[1..1 + start].iter().cloned().map(|mut m| {
            m.parts = None;
            m
        }));
        messages.push(WireMessage::new("user", sessions::summary_request(&focus)));
        let prompt = sessions::estimate_tokens(&messages[0].content) + sessions::estimate_tokens(&self.tools_json.to_string()) + sessions::estimate_messages(&messages[1..]);
        let room = sessions::compute_max_tokens(env.ctx_max, prompt, 1200);
        let base = env.base.to_string();
        let hook = Task::perform(fire_hook(base.clone(), "pre_compact", "", json!({ "tokens": before })), |_| Message::Code(CodeMsg::Sess(SessMsg::Nop)));
        self.note("Compacting the conversation ...");
        let note = self.items.len() - 1;
        self.compacting = true;
        self.running = true;
        let work = Task::perform(
            async move {
                if room == 0 {
                    return Err(sessions::TOO_SMALL.to_string());
                }
                ask_once(base, model, messages, room).await
            },
            move |result| Message::Code(CodeMsg::Sess(SessMsg::Compacted { start, before, resume, note, result })),
        );
        Some(Task::batch([hook, work]))
    }

    fn finish_compaction(&mut self, start: usize, before: usize, resume: bool, note: usize, result: Result<String, String>, env: &Env) -> Task<Message> {
        self.compacting = false;
        let summary = result.unwrap_or_default();
        if summary.is_empty() {
            if let Some(Item::Note(t)) = self.items.get_mut(note) {
                *t = "Compacting failed; older steps are trimmed instead.".into();
            }
            self.running = resume;
            return if resume { self.start_stream_now(env) } else { Task::none() };
        }
        let wire = self.wire(env.extra);
        let msgs = &wire[1..];
        let tail_first_user = msgs.get(start).is_none_or(|m| m.role == "user");
        let mut files: Vec<String> = Vec::new();
        for c in &self.changes {
            if !files.contains(&c.path) {
                files.push(c.path.clone());
            }
        }
        let todos: Vec<SavedTodo> = self.todos.iter().map(todo_saved).collect();
        let prefix: Vec<(String, String)> = sessions::build_prefix(&summary, &todos, &files, tail_first_user).into_iter().map(|m| (m.role, m.content)).collect();
        let (old_prefix, old_covered) = self.compaction.as_ref().map_or((0, 0), |c| (c.prefix.len(), c.covered));
        let covered = old_covered + start.saturating_sub(old_prefix);
        // the first message of yours the summary does not cover
        let keep = self.compaction.take();
        let items_now = self.items.len();
        let mut from_item = items_now;
        for i in (0..items_now).rev() {
            if matches!(self.items[i], Item::User { .. }) {
                let raw = self.wire_of(&self.items[..i], None).len() - 1;
                if raw >= covered {
                    from_item = i;
                } else {
                    break;
                }
            }
        }
        let _ = keep;
        self.compaction = Some(sessions::Compaction { prefix, covered, from_item });
        let after = self.code_estimate(env.extra);
        if let Some(Item::Note(t)) = self.items.get_mut(note) {
            *t = format!("Context compacted: {start} earlier messages became a summary (about {before} to {after} tokens). You still see everything here.");
        }
        self.persist(env.session_file);
        self.running = resume;
        if resume {
            self.start_stream_now(env)
        } else {
            Task::none()
        }
    }

    // ---- the title -----------------------------------------------------------------------------

    /// `generateTitle()`: after the first answer the model names the session.
    pub fn title_task(&mut self, env: &Env) -> Task<Message> {
        if self.title_locked || self.titling || !self.title.is_empty() || self.session_id.is_empty() || self.running {
            return Task::none();
        }
        let Some(model) = env.model.map(str::to_string) else { return Task::none() };
        let Some(first) = sessions::first_user_text(&self.saved_all()) else { return Task::none() };
        self.titling = true;
        let id = self.session_id.clone();
        let messages = vec![WireMessage::new("system", "You write very short titles for coding sessions."), WireMessage::new("user", sessions::title_request(&sessions::heuristic_title(Some(&first))))];
        let base = env.base.to_string();
        Task::perform(ask_once(base, model, messages, 400), move |r| Message::Code(CodeMsg::Sess(SessMsg::TitleMade(id.clone(), r))))
    }

    // ---- the messages --------------------------------------------------------------------------

    pub(super) fn sess_update(&mut self, m: SessMsg, env: &Env) -> Task<Message> {
        match m {
            SessMsg::Nop => Task::none(),
            SessMsg::Choice(at, row) => {
                let Some(set) = self.choices.get(at).cloned() else { return Task::none() };
                let Some((_, data)) = set.rows.get(row).cloned() else { return Task::none() };
                self.choices.remove(at);
                match set.action {
                    ChoiceAction::Resume => self.resume(&data, env),
                    ChoiceAction::Rewind => match data.parse::<u64>() {
                        Ok(id) => Task::perform(undo_call(env.base.to_string(), json!({ "to": id })), |r| Message::Code(CodeMsg::Sess(SessMsg::RewindDone(r)))),
                        Err(_) => Task::none(),
                    },
                    ChoiceAction::Fork => self.fork(data.parse().ok(), env),
                    ChoiceAction::Revert => match data.parse() {
                        Ok(i) => self.start_revert(i, env),
                        Err(_) => Task::none(),
                    },
                    ChoiceAction::Unrevert => self.start_unrevert(env),
                    ChoiceAction::Model => {
                        self.pending_model = Some(data);
                        Task::none()
                    }
                }
            }
            SessMsg::TurnFork(i) => self.fork(Some(i), env),
            SessMsg::TurnRevert(i) => self.start_revert(i, env),
            SessMsg::Resume(id) => self.resume(&id, env),
            SessMsg::ForkOf(id) => {
                if id != self.session_id {
                    let t = self.resume(&id, env);
                    if self.session_id != id {
                        return t;
                    }
                }
                self.fork(None, env)
            }
            SessMsg::ExportOf(id) => self.export(&id, env),
            SessMsg::Search(q) => {
                self.session_query = q;
                Task::none()
            }
            SessMsg::RenameStart(id) => {
                let current = if id == self.session_id { self.title_now() } else { self.sessions.iter().find(|s| s.id == id).map(|s| s.shown_title()).unwrap_or_default() };
                self.rename = Some((id, current));
                Task::none()
            }
            SessMsg::RenameEdit(v) => {
                if let Some((_, draft)) = self.rename.as_mut() {
                    *draft = v;
                }
                Task::none()
            }
            SessMsg::RenameCancel => {
                self.rename = None;
                Task::none()
            }
            SessMsg::RenameOk => {
                let Some((id, draft)) = self.rename.take() else { return Task::none() };
                let title = sessions::clean_title(&draft);
                if title.is_empty() {
                    return Task::none();
                }
                if id == self.session_id {
                    self.title = title;
                    self.title_locked = true;
                    self.persist(env.session_file);
                } else if let Some(root) = self.workspace.as_ref().map(|w| w.root.clone()) {
                    let mut all = Sessions::load(env.session_file);
                    if let Some(rec) = all.0.get_mut(&root).and_then(|l| l.iter_mut().find(|s| s.id == id)) {
                        rec.title = title;
                        rec.title_locked = true;
                    }
                    let _ = all.save(env.session_file);
                    self.sessions = all.of(&root).to_vec();
                }
                Task::none()
            }
            SessMsg::DeleteAsk(id) => {
                self.confirm_delete = Some(id);
                Task::none()
            }
            SessMsg::DeleteCancel => {
                self.confirm_delete = None;
                Task::none()
            }
            SessMsg::DeleteOk => {
                let Some(id) = self.confirm_delete.take() else { return Task::none() };
                let Some(root) = self.workspace.as_ref().map(|w| w.root.clone()) else { return Task::none() };
                let mut all = Sessions::load(env.session_file);
                all.remove(&root, &id);
                let _ = all.save(env.session_file);
                self.sessions = all.of(&root).to_vec();
                if id == self.session_id {
                    let stop = if self.running { self.stop_now(env) } else { Task::none() };
                    self.reset_session();
                    return stop;
                }
                Task::none()
            }
            SessMsg::RewindStart => Task::perform(fetch_changes(env.base.to_string()), |r| Message::Code(CodeMsg::Sess(SessMsg::RewindList(r)))),
            SessMsg::RewindList(r) => {
                match r {
                    Err(_) => {
                        self.changes_known = false;
                        self.note("Rewind is not available on this server. /undo reverts the last file change.");
                    }
                    Ok(list) => {
                        self.changes_known = true;
                        self.changes = list;
                        let rows = self.rewind_rows();
                        if rows.is_empty() {
                            self.note("No file changes to rewind.");
                        } else {
                            self.push_choices("Rewind: pick the oldest change to take back. It and every later change are undone.", ChoiceAction::Rewind, rows);
                        }
                    }
                }
                Task::none()
            }
            SessMsg::RewindDone(r) => {
                let text = match r {
                    Ok(v) if v.get("ok").and_then(Value::as_bool) == Some(false) => v.pointer("/error/message").and_then(Value::as_str).or_else(|| v.get("error").and_then(Value::as_str)).unwrap_or("The rewind failed.").to_string(),
                    Ok(v) => format!("Rewound: {}", v.get("output").and_then(Value::as_str).filter(|s| !s.is_empty()).unwrap_or("done")),
                    Err(e) => e,
                };
                self.note(text);
                self.refresh_changes(env)
            }
            SessMsg::RevertList(item, r) => {
                if let Ok(list) = r {
                    self.changes_known = true;
                    self.changes = list;
                } else {
                    self.changes_known = false;
                }
                let mark = self.marks.iter().find(|m| m.item == item).map_or(0, |m| m.changes);
                let ids: Vec<u64> = self.changes.iter().map(|c| c.id).collect();
                match self.changes_known.then(|| sessions::first_change_after(&ids, mark)).flatten() {
                    Some(first) => Task::perform(undo_call(env.base.to_string(), json!({ "to": first })), move |r| Message::Code(CodeMsg::Sess(SessMsg::RevertUndone(item, r)))),
                    None => self.apply_revert(item, 0, env),
                }
            }
            SessMsg::RevertUndone(item, r) => {
                let failed = match &r {
                    Ok(v) if v.get("ok").and_then(Value::as_bool) == Some(false) => Some(v.pointer("/error/message").and_then(Value::as_str).or_else(|| v.get("error").and_then(Value::as_str)).unwrap_or("The files could not be put back.").to_string()),
                    Err(e) => Some(e.clone()),
                    _ => None,
                };
                if let Some(why) = failed {
                    self.retry_pending = false;
                    self.note(format!("Revert stopped: {why}"));
                    return self.refresh_changes(env);
                }
                let restored = r.ok().and_then(|v| v.get("restored").and_then(Value::as_array).map(Vec::len)).unwrap_or(0);
                self.apply_revert(item, restored, env)
            }
            SessMsg::UnrevertDone(r) => {
                let failed = match &r {
                    Ok(v) if v.get("ok").and_then(Value::as_bool) == Some(false) => Some(v.pointer("/error/message").and_then(Value::as_str).or_else(|| v.get("error").and_then(Value::as_str)).unwrap_or("").to_string()),
                    Err(e) => Some(e.clone()),
                    _ => None,
                };
                if let Some(why) = failed {
                    self.note(format!("The files could not be restored: {why}"));
                    return Task::none();
                }
                self.apply_unrevert(env)
            }
            SessMsg::Compacted { start, before, resume, note, result } => self.finish_compaction(start, before, resume, note, result, env),
            SessMsg::TitleMade(id, r) => {
                self.titling = false;
                let title = sessions::clean_title(&r.unwrap_or_default());
                if !title.is_empty() && id == self.session_id && !self.title_locked {
                    self.title = title;
                    self.persist(env.session_file);
                }
                Task::none()
            }
            other => self.side_update(other, env),
        }
    }
}
