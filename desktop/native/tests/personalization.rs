//! Personalization and memory: what of yours the prompt carries, "remember that …", the switches of
//! the chat menu, the Settings section, the *Remember* button, the copy on the server and the
//! `memory` tool of the model with its card and undo.

use fllm_app::api::{ModelInfo, StreamEvent, SyncAnswer};
use fllm_app::app::{App, Message, PersonalSync};
use fllm_app::chatlist::ListMsg;
use fllm_app::memory::Memory;
use fllm_app::store::{ChatRecord, Record, Role, Store};
use iced::widget::text_editor::{Action, Edit};
use iced_test::Simulator;
use std::sync::Arc;

fn rec(role: Role, content: &str) -> Record {
    Record {
        role,
        content: content.into(),
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
    }
}

fn app_with(store: Store) -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    store.save(&path).unwrap();
    let (mut app, _t) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("m", 4096))));
    (app, dir)
}

fn saved(dir: &tempfile::TempDir) -> Store {
    Store::load(&dir.path().join("chats.json"))
}

fn sim(app: &App) -> Simulator<'_, Message> {
    let settings = iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() };
    Simulator::with_size(settings, iced::Size::new(1280.0, 900.0), app.view())
}

fn paste(text: &str) -> Action {
    Action::Edit(Edit::Paste(Arc::new(text.to_string())))
}

fn memory(id: &str, text: &str, uses: u32) -> Memory {
    Memory { id: id.into(), text: text.into(), created_at: 1, updated_at: 1, uses, category: None }
}

#[test]
fn about_me_and_the_memories_are_in_the_prompt_in_the_studios_order() {
    let mut store = Store::default();
    store.prefs.personal.about = "I am a nurse. Keep answers short.".into();
    store.prefs.personal.memories.push(memory("m1", "Prefers metric units", 0));
    let (app, _dir) = app_with(store);

    let full = app.system_prompt();
    let about_at = full.find("About the user and how they want answers (in their own words):").expect("the about block");
    let memory_at = full.find("Saved memory about the user (use it when it is relevant, do not bring it up otherwise):").expect("the memory block");
    assert!(full[about_at..].contains("I am a nurse."), "the text itself follows");
    assert!(full[memory_at..].contains("- Prefers metric units"), "the entries as lines");
    assert!(about_at < memory_at, "about me before the memory (composeSystem)");

    // one chat says: your instructions and memory are not used here
    let mut chat = ChatRecord::new(0, "quiet", vec![rec(Role::User, "hi"), rec(Role::Assistant, "yo")]);
    chat.no_personal = true;
    let mut store = Store::default();
    store.prefs.personal.about = "I am a nurse.".into();
    store.prefs.personal.memories.push(memory("m1", "Prefers metric units", 0));
    store.chats.push(chat);
    store.next_id = 1;
    let (mut app, _dir) = app_with(store);
    let _ = app.update(Message::OpenChat(0));
    let quiet = app.system_prompt();
    assert!(!quiet.contains("About the user"), "no personalization in this chat");
    assert!(!quiet.contains("Saved memory about the user"), "no memory in this chat");

    // memory alone can be switched off for one chat
    let mut chat = ChatRecord::new(0, "quiet", vec![rec(Role::User, "hi"), rec(Role::Assistant, "yo")]);
    chat.no_memory = true;
    let mut store = Store::default();
    store.prefs.personal.about = "I am a nurse.".into();
    store.prefs.personal.memories.push(memory("m1", "Prefers metric units", 0));
    store.chats.push(chat);
    store.next_id = 1;
    let (mut app, _dir) = app_with(store);
    let _ = app.update(Message::OpenChat(0));
    let quiet = app.system_prompt();
    assert!(quiet.contains("About the user"), "the about me stays");
    assert!(!quiet.contains("Saved memory about the user"), "only the memory is out");
}

#[test]
fn remember_that_at_the_start_of_a_question_saves_the_thing() {
    let (mut app, dir) = app_with(Store::default());
    let question = paste("remember that I always carry a spare oil can");
    let _ = app.update(Message::Edit(question));
    let _ = app.update(Message::Send);
    let after = saved(&dir);
    assert_eq!(after.prefs.personal.memories.len(), 1, "the fact is remembered");
    assert_eq!(after.prefs.personal.memories[0].text, "I always carry a spare oil can");
    assert_eq!(app.toast(), Some("Remembered. See Settings > Personalization."), "the line of the Studio");

    // a secret is refused with the Studio's line
    let (mut app, _dir) = app_with(Store::default());
    let _ = app.update(Message::Edit(paste("remember that password: hunter2secret")));
    let _ = app.update(Message::Send);
    assert_eq!(app.toast(), Some("That looks like a password, key or card number. It is not saved to memory."));
    assert!(app.prefs().personal.memories.is_empty(), "nothing is saved");

    // a chat that says no does not capture
    let mut store = Store::default();
    let mut chat = ChatRecord::new(0, "quiet", vec![rec(Role::User, "hi"), rec(Role::Assistant, "yo")]);
    chat.no_memory = true;
    store.chats.push(chat);
    store.next_id = 1;
    let (mut app, _dir) = app_with(store);
    let _ = app.update(Message::OpenChat(0));
    let _ = app.update(Message::Edit(paste("remember that I always carry a spare oil can")));
    let _ = app.update(Message::Send);
    assert!(app.prefs().personal.memories.is_empty(), "memory is switched off in this chat");
}

#[test]
fn the_chat_menu_switches_personalization_and_memory_with_the_lines_of_the_studio() {
    let mut store = Store::default();
    let mut chat = ChatRecord::new(0, "one", vec![rec(Role::User, "hi"), rec(Role::Assistant, "yo")]);
    chat.updated = 1;
    store.chats.push(chat);
    store.next_id = 1;
    let (mut app, dir) = app_with(store);

    let _ = app.update(Message::List(ListMsg::Personal(0)));
    assert!(saved(&dir).chats[0].no_personal, "the switch went over");
    assert_eq!(app.toast(), Some("Your instructions and memory are not used in this chat."));
    let _ = app.update(Message::List(ListMsg::Personal(0)));
    assert!(!saved(&dir).chats[0].no_personal, "and back again");
    assert_eq!(app.toast(), Some("Your instructions and memory are used in this chat."));

    let _ = app.update(Message::List(ListMsg::Memory(0)));
    assert!(saved(&dir).chats[0].no_memory);
    assert_eq!(app.toast(), Some("Memory is not used or changed in this chat."));
    let _ = app.update(Message::List(ListMsg::Memory(0)));
    assert!(!saved(&dir).chats[0].no_memory);
    assert_eq!(app.toast(), Some("Memory is used in this chat."));
}

#[test]
fn the_settings_section_draws_and_adds_and_forgets_memories() {
    let mut store = Store::default();
    store.prefs.personal.about = "I am a nurse.".into();
    store.prefs.personal.memories.push(memory("m1", "Prefers metric units", 0));
    let (mut app, dir) = app_with(store);
    let _ = app.update(Message::Go(fllm_app::app::View::Settings));
    {
        let mut ui = sim(&app);
        ui.find("Custom instructions").expect("the about section");
        ui.find("Remembered facts").expect("the memory section");
        ui.find("Prefers metric units").expect("the entry as a row");
        ui.find("1 of 100 entries. All of them are sent to the model.").expect("the usage line of the Studio");
        ui.find("Add something to remember").expect("the row to add one");
        ui.find("Forget everything").expect("the button at the end");
    }
    // add one through the row
    let _ = app.update(Message::MemNewInput("Flies the DA40".into()));
    let _ = app.update(Message::MemAddNow);
    assert_eq!(saved(&dir).prefs.personal.memories.len(), 2);
    // the tick of the window writes the copy to the server (nothing is running here: it only skips
    // when the text is already up there)
    let _ = app.update(Message::Tick);
    // change it through the row, then delete it again
    let id = saved(&dir).prefs.personal.memories[1].id.clone();
    let _ = app.update(Message::MemEditStart(id.clone()));
    let _ = app.update(Message::MemEditInput("Flies the C172".into()));
    let _ = app.update(Message::MemEditEnd(true));
    assert_eq!(saved(&dir).prefs.personal.memories[1].text, "Flies the C172");
    let _ = app.update(Message::MemDelete(id));
    assert_eq!(saved(&dir).prefs.personal.memories.len(), 1);
    // forget everything asks first
    let _ = app.update(Message::MemClear);
    {
        let mut ui = sim(&app);
        ui.find("Forget everything?").expect("the dialog");
        ui.find("All remembered facts will be deleted.").expect("its sentence");
    }
    assert_eq!(saved(&dir).prefs.personal.memories.len(), 1, "nothing is gone before the yes");
    let _ = app.update(Message::MemClearOk);
    assert!(saved(&dir).prefs.personal.memories.is_empty());
}

#[test]
fn the_remember_button_under_a_message_saves_what_it_is_prefilled_with() {
    let mut store = Store::default();
    let mut chat = ChatRecord::new(0, "one", vec![rec(Role::User, "I fly   the  DA40"), rec(Role::Assistant, "Nice.")]);
    chat.updated = 1;
    store.chats.push(chat);
    store.next_id = 1;
    let (mut app, dir) = app_with(store);
    let _ = app.update(Message::OpenChat(0));

    let _ = app.update(Message::RememberOpen(0));
    {
        let mut ui = sim(&app);
        ui.find("Remember").expect("the dialog");
        ui.find("This is added to the instructions of every chat. Edit it first if you like. Memories stay on this computer.")
            .expect("the sentence of the Studio");
        ui.find("I fly the DA40").expect("the question, whitespace collapsed");
    }
    let _ = app.update(Message::RememberEdit("I fly the DA40, left seat".into()));
    let _ = app.update(Message::RememberOk);
    let after = saved(&dir);
    assert_eq!(after.prefs.personal.memories.len(), 1);
    assert_eq!(after.prefs.personal.memories[0].text, "I fly the DA40, left seat");

    // switched off in the settings: the button says so and saves nothing
    let _ = app.update(Message::UseMemory(false));
    let _ = app.update(Message::RememberOpen(1));
    assert_eq!(app.toast(), Some("Memory is switched off in Settings > Personalization."));
    assert_eq!(saved(&dir).prefs.personal.memories.len(), 1, "nothing new");
}

#[test]
fn the_copy_on_the_server_is_taken_over_on_a_first_run() {
    let (mut app, _dir) = app_with(Store::default());
    assert!(!app.prefs().personal_loaded, "no local copy yet");
    let entries = vec![serde_json::json!({"id": "s1", "text": "Likes tea", "createdAt": 5})];
    let _ = app.update(Message::PersonalSynced(SyncAnswer::Got(PersonalSync::Import(entries))));
    assert!(app.prefs().personal_loaded, "the copy is local now");
    assert_eq!(app.prefs().personal.memories[0].text, "Likes tea");
    assert_eq!(app.toast(), Some("1 remembered fact restored from the copy on this computer."));
    // a refused server switches the copy off; unreachable does not (the next tick asks again)
    let _ = app.update(Message::DismissToast);
    let _ = app.update(Message::PersonalSynced(SyncAnswer::Refused));
    let _ = app.update(Message::Tick);
    assert_eq!(app.toast(), None, "nothing to say about it");
}

#[test]
fn the_memory_tool_of_the_model_runs_and_its_card_can_be_taken_back() {
    let (mut app, dir) = app_with(Store::default());
    assert!(app.memory_tool_offered(), "on by default");
    let question = paste("What do I fly?");
    let _ = app.update(Message::Edit(question));
    let _ = app.update(Message::Send);
    // the answer calls the tool, in parts, and ends
    let args = serde_json::json!({"action": "add", "text": "Flies the DA40", "category": "profile"}).to_string();
    let _ = app.update(Message::Stream(
        0,
        StreamEvent::ToolCall { index: 0, id: Some("call_1".into()), name: Some("memory".into()), arguments: args },
    ));
    let _ = app.update(Message::Stream(0, StreamEvent::Done));
    // the memory is saved, the answer asks again (a second round is running)
    let after = saved(&dir);
    assert_eq!(after.prefs.personal.memories.len(), 1, "the tool saved the fact");
    assert_eq!(after.prefs.personal.memories[0].text, "Flies the DA40");
    assert!(after.prefs.personal.memories[0].category.as_deref() == Some("profile"));

    // the request of the second round carries the call and the answer of the tool
    let wire = app.wire_messages();
    let roles: Vec<&str> = wire.iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, ["system", "user", "assistant", "tool"], "the shape of a tool round: {roles:?}");
    assert!(wire[2].tool_calls.is_some(), "the call travels back with the assistant");
    assert_eq!(wire[2].tool_calls.as_ref().unwrap()[0]["function"]["name"], "memory");
    assert_eq!(wire[3].tool_call_id.as_deref(), Some("call_1"), "the tool message answers the call");
    assert_eq!(wire[3].content, "Saved.", "what the model is told");

    // the card under the answer, with its undo
    {
        let mut ui = sim(&app);
        ui.find("Saved to memory: Flies the DA40").expect("the card under the answer");
        ui.find("Undo").expect("the undo of the change");
    }
    let _ = app.update(Message::MemoryUndo(1, 0));
    assert!(saved(&dir).prefs.personal.memories.is_empty(), "the change is taken back");
    assert_eq!(app.toast(), Some("Memory change undone"));
    {
        let mut ui = sim(&app);
        ui.find("Undo").expect_err("the card is used up");
    }

    // an unknown tool is answered with the Studio's error, the round ends without a save
    let (mut app, _dir) = app_with(Store::default());
    let _ = app.update(Message::Edit(paste("Do something")));
    let _ = app.update(Message::Send);
    let _ = app.update(Message::Stream(
        0,
        StreamEvent::ToolCall { index: 0, id: Some("call_2".into()), name: Some("nope".into()), arguments: "{}".into() },
    ));
    let _ = app.update(Message::Stream(0, StreamEvent::Done));
    let wire = app.wire_messages();
    assert_eq!(wire.last().unwrap().content, "Error: unknown tool 'nope'.");
    assert!(app.prefs().personal.memories.is_empty());
}

#[test]
fn a_memory_that_does_not_fit_the_prompt_is_left_out_and_the_line_says_so() {
    let mut store = Store::default();
    // three entries that together are over the cap of the prompt (the model of the test has a small
    // context, so its prompt is the compact one with 1,200 characters)
    store.prefs.personal.memories = (0..3).map(|i| memory(&format!("m{i}"), &format!("fact {i} {}", "x".repeat(1000)), 0)).collect();
    let (mut app, _dir) = app_with(store);
    let _ = app.update(Message::Go(fllm_app::app::View::Settings));
    {
        let mut ui = sim(&app);
        ui.find("3 of 100 entries. The 1 oldest do not fit into the 3,000 characters sent to the model and are skipped.")
            .expect("the count and the line of the Studio");
    }
    let full = app.system_prompt();
    let block_at = full.find("Saved memory about the user").expect("the block is there");
    // the entries are ranked newest first, so the newest one gets the room of the compact prompt
    assert!(full[block_at..].contains("fact 2"), "the newest entry made it in");
    assert!(!full.contains("fact 0"), "the oldest one did not fit");
}


/// Writes the three views of the feature as PNGs into the snapshot dir of the run.
#[test]
fn the_personalization_views_are_painted() {
    // the Settings section with about me and two entries
    let mut store = Store::default();
    store.prefs.personal.about = "I am a nurse. Keep answers short, use metric units.".into();
    store.prefs.personal.memories.push(memory("m1", "Prefers metric units", 2));
    store.prefs.personal.memories.push(memory("m2", "Flies the DA40", 0));
    let (mut app, _dir) = app_with(store);
    let _ = app.update(Message::Go(fllm_app::app::View::Settings));
    save_png_h(&app, "mem-settings", 1500.0);

    // the dialog of the Remember button under a question
    let mut store = Store::default();
    let mut chat = ChatRecord::new(0, "one", vec![rec(Role::User, "I fly the DA40"), rec(Role::Assistant, "Nice.")]);
    chat.updated = 1;
    store.chats.push(chat);
    store.next_id = 1;
    let (mut app, _dir) = app_with(store);
    let _ = app.update(Message::OpenChat(0));
    let _ = app.update(Message::RememberOpen(0));
    save_png(&app, "mem-remember");

    // the card of the memory tool under an answer
    let (mut app, _dir) = app_with(Store::default());
    let _ = app.update(Message::Edit(paste("What do I fly?")));
    let _ = app.update(Message::Send);
    let args = serde_json::json!({"action": "add", "text": "Flies the DA40"}).to_string();
    let _ = app.update(Message::Stream(
        0,
        StreamEvent::ToolCall { index: 0, id: Some("call_1".into()), name: Some("memory".into()), arguments: args },
    ));
    let _ = app.update(Message::Stream(0, StreamEvent::Done));
    save_png(&app, "mem-card");
}

/// Writes one PNG of the given view into the snapshot dir of the run (the pattern of `tests/ui.rs`).
fn save_png(app: &App, name: &str) {
    save_png_h(app, name, 900.0)
}

fn save_png_h(app: &App, name: &str, height: f32) {
    let theme = app.theme();
    let settings = iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() };
    let mut ui = Simulator::with_size(settings, iced::Size::new(1280.0, height), app.view());
    let shot = ui.snapshot(&theme).expect("the software renderer draws the view");
    let dir = std::path::PathBuf::from(std::env::var("FLLM_SNAPSHOT_DIR").unwrap_or_else(|_| "/opt/data/cache/scratch/fLLM/desktop/native/ui-snapshots".into()));
    std::fs::create_dir_all(&dir).unwrap();
    let prefix = format!("{name}-");
    for old in std::fs::read_dir(&dir).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with(&prefix)) {
        let _ = std::fs::remove_file(old.path());
    }
    shot.matches_image(dir.join(format!("{name}.png"))).expect("the picture is written");
    assert!(std::fs::read_dir(&dir).unwrap().flatten().any(|e| e.file_name().to_string_lossy().starts_with(&prefix)), "the picture exists");
}
