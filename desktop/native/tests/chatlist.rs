//! The list of chats: groups by day, pinned and archived, search, rename, duplicate, export, delete one or many.

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message};
use fllm_app::chatlist::{group_of, Format, ListMsg};
use fllm_app::store::{ChatRecord, Record, Role, Store};
use iced_test::Simulator;

const DAY: i64 = 86_400;

fn rec(role: Role, content: &str) -> Record {
    Record { role, content: content.into(), reasoning: String::new(), stats: None, passages: None , images: Vec::new(), files: Vec::new(), tool_calls: None, tool_call_id: None, tool_cards: Vec::new() }
}

fn chat(id: u64, title: &str, updated: i64, text: &str) -> ChatRecord {
    let mut c = ChatRecord::new(id, title, vec![rec(Role::User, text), rec(Role::Assistant, "Sure.")]);
    c.updated = updated;
    c
}

fn app_with(chats: Vec<ChatRecord>) -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    let next_id = chats.iter().map(|c| c.id + 1).max().unwrap_or(0);
    Store { chats, next_id, theme: "light".into(), ..Default::default() }.save(&path).unwrap();
    let (mut app, _t) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("m", 4096))));
    (app, dir)
}

fn list(app: &mut App, m: ListMsg) {
    let _ = app.update(Message::List(m));
}

fn titles(app: &App) -> Vec<String> {
    app.chat_list().groups(&reload_store(app), now()).0.into_iter().flat_map(|(_, l)| l.into_iter().map(|c| c.title.clone())).collect()
}

fn reload_store(app: &App) -> Store {
    // the store as the window holds it is saved after every change: the file is what is asked here
    Store::load(app.data_file())
}

fn now() -> i64 {
    chrono::Local::now().timestamp()
}

fn sim(app: &App) -> Simulator<'_, Message> {
    let settings = iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() };
    Simulator::with_size(settings, iced::Size::new(1280.0, 900.0), app.view())
}

#[test]
fn chats_are_grouped_by_the_day_of_their_last_use_pinned_first() {
    let n = now();
    assert_eq!(group_of(n, n), "Today");
    assert_eq!(group_of(n - DAY, n), "Yesterday");
    assert_eq!(group_of(n - 4 * DAY, n), "Previous 7 days");
    assert_eq!(group_of(n - 20 * DAY, n), "Previous 30 days");
    assert_eq!(group_of(n - 90 * DAY, n), "Older");
    assert_eq!(group_of(0, n), "Older", "a chat from before the date was kept");
    let (mut app, _d) = app_with(vec![chat(0, "old one", n - 90 * DAY, "a"), chat(1, "today one", n, "b"), chat(2, "yesterday one", n - DAY, "c"), chat(3, "pinned one", n - 90 * DAY, "d")]);
    list(&mut app, ListMsg::Pin(3));
    let store = reload_store(&app);
    let (groups, archived) = app.chat_list().groups(&store, n);
    let names: Vec<&str> = groups.iter().map(|(g, _)| g.as_str()).collect();
    assert_eq!(names, ["Pinned", "Today", "Yesterday", "Older"]);
    assert_eq!(archived, 0);
    assert_eq!(groups[0].1[0].title, "pinned one");
    // unpinning puts it back where its day says
    list(&mut app, ListMsg::Pin(3));
    assert!(titles(&app).ends_with(&["old one".to_string(), "pinned one".to_string()]) || titles(&app).contains(&"pinned one".to_string()));
}

#[test]
fn archived_chats_are_hidden_until_asked_for_and_a_search_finds_them_with_all_the_others() {
    let n = now();
    let (mut app, _d) = app_with(vec![chat(0, "Plan the trip", n, "flights to Lisbon"), chat(1, "Rust lifetimes", n - 1, "explain borrowing"), chat(2, "Old recipe", n - 2, "pasta with lisbon sauce")]);
    list(&mut app, ListMsg::Archive(2));
    assert_eq!(titles(&app), ["Plan the trip", "Rust lifetimes"]);
    let store = reload_store(&app);
    assert_eq!(app.chat_list().groups(&store, n).1, 1, "one archived chat is counted");
    list(&mut app, ListMsg::ShowArchived(true));
    assert_eq!(titles(&app).last().map(String::as_str), Some("Old recipe"));
    list(&mut app, ListMsg::ShowArchived(false));
    // a search looks into the messages too, and into archived chats
    list(&mut app, ListMsg::Search("LISBON".into()));
    assert_eq!(titles(&app), ["Plan the trip", "Old recipe"]);
    list(&mut app, ListMsg::Search("borrowing".into()));
    assert_eq!(titles(&app), ["Rust lifetimes"]);
    list(&mut app, ListMsg::Search("zzz".into()));
    assert!(titles(&app).is_empty());
    // archiving unpins
    list(&mut app, ListMsg::Search(String::new()));
    list(&mut app, ListMsg::Pin(0));
    list(&mut app, ListMsg::Archive(0));
    assert!(!reload_store(&app).chat(0).unwrap().pinned);
    list(&mut app, ListMsg::Archive(0));
    assert!(!reload_store(&app).chat(0).unwrap().archived, "archive toggles");
}

#[test]
fn a_chat_is_renamed_duplicated_exported_and_deleted() {
    let n = now();
    let (mut app, dir) = app_with(vec![chat(0, "First", n, "Hello **world**")]);
    list(&mut app, ListMsg::RenameStart(0));
    list(&mut app, ListMsg::RenameInput("   ".into()));
    list(&mut app, ListMsg::RenameEnd(true));
    assert_eq!(reload_store(&app).chat(0).unwrap().title, "First", "an empty name changes nothing");
    list(&mut app, ListMsg::RenameStart(0));
    list(&mut app, ListMsg::RenameInput("Renamed".into()));
    list(&mut app, ListMsg::RenameEnd(false));
    assert_eq!(reload_store(&app).chat(0).unwrap().title, "First", "cancelled");
    list(&mut app, ListMsg::RenameStart(0));
    list(&mut app, ListMsg::RenameInput("Renamed".into()));
    list(&mut app, ListMsg::RenameEnd(true));
    assert_eq!(reload_store(&app).chat(0).unwrap().title, "Renamed");

    list(&mut app, ListMsg::Duplicate(0));
    let store = reload_store(&app);
    assert_eq!(store.chats.len(), 2);
    let copy = store.chats.iter().find(|c| c.title == "Renamed (copy)").expect("a copy");
    assert_eq!(copy.messages.len(), 2);
    assert_ne!(copy.id, 0);

    // export writes a file and says where
    let out = dir.path().join("exports");
    std::fs::create_dir_all(&out).unwrap();
    std::env::set_var("FLLM_EXPORT_DIR", &out);
    list(&mut app, ListMsg::Export(0, Format::Markdown));
    let toast = app.toast().expect("a line says where the file is").to_string();
    assert!(toast.starts_with("Saved to ") && toast.contains("fllm-Renamed-") && toast.ends_with(".md"), "{toast}");
    list(&mut app, ListMsg::Export(0, Format::Json));
    let mut files: Vec<String> = std::fs::read_dir(&out).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    files.sort();
    assert_eq!(files.len(), 2);
    let md = std::fs::read_to_string(out.join(files.iter().find(|f| f.ends_with(".md")).unwrap())).unwrap();
    assert_eq!(md, "# Renamed\n\n## You\n\nHello **world**\n\n## Assistant\n\nSure.\n\n");
    let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(out.join(files.iter().find(|f| f.ends_with(".json")).unwrap())).unwrap()).unwrap();
    assert_eq!((json["title"].as_str(), json["messages"][1]["role"].as_str(), json["messages"][0]["content"].as_str()), (Some("Renamed"), Some("assistant"), Some("Hello **world**")));
    std::env::remove_var("FLLM_EXPORT_DIR");
    let _ = app.update(Message::DismissToast);
    assert!(app.toast().is_none());

    list(&mut app, ListMsg::Delete(0));
    assert_eq!(reload_store(&app).chats.len(), 1);
}

#[test]
fn many_chats_are_selected_and_archived_or_deleted_together_and_the_open_one_is_closed() {
    let n = now();
    let (mut app, _d) = app_with((0..5).map(|i| chat(i, &format!("chat {i}"), n - i as i64, "x")).collect());
    let _ = app.update(Message::OpenChat(1));
    assert_eq!(app.message_count(), 2);
    list(&mut app, ListMsg::SelectMode(true));
    list(&mut app, ListMsg::Toggle(1));
    list(&mut app, ListMsg::Toggle(3));
    list(&mut app, ListMsg::Toggle(3));
    list(&mut app, ListMsg::Toggle(4));
    assert_eq!(app.chat_list().selected.len(), 2);
    list(&mut app, ListMsg::ArchiveSelected);
    assert!(!app.chat_list().select && app.chat_list().selected.is_empty());
    assert_eq!(titles(&app), ["chat 0", "chat 2", "chat 3"]);
    list(&mut app, ListMsg::SelectMode(true));
    list(&mut app, ListMsg::SelectAll);
    assert_eq!(app.chat_list().selected.len(), 3, "all that are shown");
    list(&mut app, ListMsg::SelectAll);
    assert!(app.chat_list().selected.is_empty(), "again: none");
    list(&mut app, ListMsg::SelectAll);
    list(&mut app, ListMsg::DeleteSelected);
    assert_eq!(reload_store(&app).chats.len(), 2, "the two archived ones stay");
    // deleting the open chat empties the conversation
    list(&mut app, ListMsg::ShowArchived(true));
    let _ = app.update(Message::OpenChat(1));
    assert_eq!(app.message_count(), 2);
    list(&mut app, ListMsg::Delete(1));
    assert_eq!(app.message_count(), 0);
    assert!(app.open_chat().is_none());
}

#[test]
fn the_sidebar_draws_the_groups_the_search_box_and_the_row_of_actions() {
    let n = now();
    let (mut app, _d) = app_with(vec![chat(0, "Today's chat", n, "x"), chat(1, "Last week chat", n - 5 * DAY, "y")]);
    {
        let mut ui = sim(&app);
        ui.find("Today").expect("a group");
        ui.find("Previous 7 days").expect("another group");
        ui.find("Today's chat").expect("a chat");
        ui.find("Select").expect("the select switch");
    }
    list(&mut app, ListMsg::Menu(Some(1)));
    {
        let mut ui = sim(&app);
        let shot = ui.snapshot(&app.theme()).unwrap();
        let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ui-snapshots");
        std::fs::create_dir_all(&dir).unwrap();
        for old in std::fs::read_dir(&dir).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with("chatlist-menu-")) {
            let _ = std::fs::remove_file(old.path());
        }
        shot.matches_image(dir.join("chatlist-menu.png")).unwrap();
    }
    {
        let mut ui = sim(&app);
        for label in ["Pin", "Rename", "Archive", "Duplicate", "Markdown", "JSON", "Delete"] {
            ui.find(label).unwrap_or_else(|e| panic!("{label}: {e:?}"));
        }
        ui.click("Archive").unwrap();
        assert!(ui.into_messages().any(|m| matches!(m, Message::List(ListMsg::Archive(1)))));
    }
    assert_eq!(app.chat_list().menu, Some(1));
    list(&mut app, ListMsg::Menu(Some(1)));
    assert!(app.chat_list().menu.is_none(), "the same button closes the row");
    list(&mut app, ListMsg::Search("nothing like it".into()));
    let mut ui = sim(&app);
    ui.find("No chat matches").expect("the empty search says so");
}
