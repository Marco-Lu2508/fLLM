//! Projects: the rows in the sidebar, the page with instructions and knowledge, moving chats into a
//! project, the passages that travel with a question and the project inside the system prompt.

use fllm_app::api::ModelInfo;
use fllm_app::app::{App, Message, View};
use fllm_app::assistants::{Assistant, SysMode};
use fllm_app::chatlist::ListMsg;
use fllm_app::store::{ChatRecord, Project, ProjectDoc, Record, Role, Store};
use iced::widget::text_editor;
use iced_test::Simulator;

fn rec(role: Role, content: &str) -> Record {
    Record { role, content: content.into(), reasoning: String::new(), stats: None, passages: None , images: Vec::new(), files: Vec::new(), tool_calls: None, tool_call_id: None, tool_cards: Vec::new(), sources: Vec::new(), research: None, logprobs: Vec::new() }
}

fn app_with(store: Store) -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    store.save(&path).unwrap();
    let (mut app, _t) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Ok(ModelInfo::new("m", 4096))));
    (app, dir)
}

fn project(id: &str, name: &str) -> Project {
    Project { id: id.into(), name: name.into(), ..Default::default() }
}

fn doc(id: &str, name: &str, text: &str) -> ProjectDoc {
    ProjectDoc { id: id.into(), name: name.into(), text: text.into(), pages: None, chars: text.chars().count(), added: 0 }
}

fn saved(dir: &tempfile::TempDir) -> Store {
    Store::load(&dir.path().join("chats.json"))
}

fn titles(app: &App, store: &Store) -> Vec<String> {
    app.chat_list()
        .groups(store, chrono::Local::now().timestamp())
        .0
        .into_iter()
        .flat_map(|(_, list)| list.into_iter().map(|c| c.title.clone()))
        .collect()
}

fn sim(app: &App) -> Simulator<'_, Message> {
    let settings = iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() };
    Simulator::with_size(settings, iced::Size::new(1280.0, 900.0), app.view())
}

#[test]
fn a_project_is_created_renamed_and_the_rename_can_be_thrown_away() {
    let (mut app, dir) = app_with(Store::default());
    let _ = app.update(Message::PrjNew);
    let _ = app.update(Message::PrjNewInput("  Work notes  ".into()));
    let _ = app.update(Message::PrjNewEnd(true));
    let store = saved(&dir);
    assert_eq!(store.projects.len(), 1);
    assert_eq!(store.projects[0].name, "Work notes", "the name is trimmed");
    assert_eq!(app.page(), View::Project, "a new project opens its page");
    let id = store.projects[0].id.clone();
    assert!(id.starts_with("prj-"), "the id looks like the Studio's uid: {id}");

    let _ = app.update(Message::PrjRenameStart(id.clone()));
    let _ = app.update(Message::PrjRenameInput("Docs".into()));
    let _ = app.update(Message::PrjRenameEnd(true));
    assert_eq!(saved(&dir).projects[0].name, "Docs");

    // the cross throws the typed name away and keeps the old one
    let _ = app.update(Message::PrjRenameStart(id.clone()));
    let _ = app.update(Message::PrjRenameInput("Nope".into()));
    let _ = app.update(Message::PrjRenameEnd(false));
    assert_eq!(saved(&dir).projects[0].name, "Docs");

    // an empty name does not rename either
    let _ = app.update(Message::PrjRenameStart(id));
    let _ = app.update(Message::PrjRenameInput("   ".into()));
    let _ = app.update(Message::PrjRenameEnd(true));
    assert_eq!(saved(&dir).projects[0].name, "Docs");
}

#[test]
fn deleting_a_project_frees_its_chats_and_says_what_it_did() {
    let mut store = Store::default();
    store.projects.push(project("p1", "Work"));
    let mut in_project = ChatRecord::new(0, "in project", vec![rec(Role::User, "hi"), rec(Role::Assistant, "yo")]);
    in_project.project = Some("p1".into());
    store.chats.push(in_project);
    store.chats.push(ChatRecord::new(1, "alone", vec![rec(Role::User, "hey"), rec(Role::Assistant, "yes")]));
    store.next_id = 2;
    let (mut app, dir) = app_with(store);

    let _ = app.update(Message::PrjDelete("p1".into()));
    {
        // the Studio asks before it deletes (the askDialog of prjDelete)
        let mut ui = sim(&app);
        ui.find("Delete project?").expect("the dialog opens first");
        ui.find("\"Work\" and its 0 documents will be deleted. Its 1 chat stay in your list.").expect("the sentence of the Studio");
    }
    assert!(saved(&dir).projects.len() == 1, "nothing is deleted before the yes");
    let _ = app.update(Message::PrjDeleteOk);
    let after = saved(&dir);
    assert!(after.projects.is_empty(), "the project is gone");
    assert!(after.chats.iter().all(|c| c.project.is_none()), "its chats stay, without a project");
    assert_eq!(after.chats.len(), 2, "no chat is deleted with the project");
    assert_eq!(app.toast(), None, "the Studio says nothing after the dialog");
    assert_eq!(app.page(), View::Chat, "an open project page closes");
}

#[test]
fn the_sidebar_lists_the_chats_of_a_project_only_without_a_search_archived_included() {
    let mut store = Store::default();
    store.projects.push(project("p1", "Work"));
    let mut inside = ChatRecord::new(0, "inside", vec![rec(Role::User, "a"), rec(Role::Assistant, "b")]);
    inside.project = Some("p1".into());
    let mut old = ChatRecord::new(1, "archived one", vec![rec(Role::User, "c"), rec(Role::Assistant, "d")]);
    old.project = Some("p1".into());
    old.archived = true;
    store.chats.push(inside);
    store.chats.push(old);
    store.chats.push(ChatRecord::new(2, "alone", vec![rec(Role::User, "e"), rec(Role::Assistant, "f")]));
    store.next_id = 3;
    let (mut app, _dir) = app_with(store.clone());

    let live = titles(&app, &store);
    assert!(live.contains(&"alone".to_string()), "a chat without a project stays in the list: {live:?}");
    assert!(!live.contains(&"inside".to_string()), "the chat of the project is only shown under its project: {live:?}");
    let _ = app.update(Message::List(ListMsg::ShowArchived(true)));
    let (groups, archived) = app.chat_list().groups(&store, chrono::Local::now().timestamp());
    assert_eq!(archived, 1, "the archived chat of a project is still reachable");
    let archived_titles: Vec<&str> = groups
        .iter()
        .filter(|(name, _)| name == "Archived")
        .flat_map(|(_, list)| list.iter().map(|c| c.title.as_str()))
        .collect();
    assert!(archived_titles.contains(&"archived one"), "under Archived, when it is shown: {archived_titles:?}");

    // with a search every chat is looked at, the project's chats included
    let _ = app.update(Message::List(ListMsg::Search("inside".into())));
    let live = titles(&app, &store);
    assert!(live.contains(&"inside".to_string()), "a search finds the chats of projects too: {live:?}");
}

#[test]
fn a_new_chat_joins_the_open_project_and_its_question_carries_the_passages() {
    let mut store = Store::default();
    let mut p = project("p1", "Work");
    p.instructions = "Answer as an engineer.".into();
    p.docs.push(doc("d1", "manual.pdf", "The turbo pump runs at forty thousand revolutions per minute and must never be started dry."));
    store.projects.push(p);
    let (mut app, dir) = app_with(store);

    let _ = app.update(Message::PrjOpen("p1".into()));
    assert_eq!(app.page(), View::Project);
    let _ = app.update(Message::PrjNewChat);
    assert_eq!(app.open_chat(), None, "the new chat starts empty");
    let question = std::sync::Arc::new("When may the turbo pump start dry?".to_string());
    let _ = app.update(Message::Edit(text_editor::Action::Edit(text_editor::Edit::Paste(question))));
    let _ = app.update(Message::Send);

    let after = saved(&dir);
    let _chat = after.chats.iter().find(|c| c.project.as_deref() == Some("p1")).expect("the new chat belongs to the project");
    assert_eq!(after.chats.len(), 1, "exactly one chat was created");

    // the passages travel with the newest question, before it
    let wire = app.wire_messages();
    let user = wire.iter().find(|m| m.role == "user").expect("the question is on the wire");
    assert!(
        user.content.starts_with("Passages from the attached documents"),
        "the passages lead: {}",
        &user.content[..120.min(user.content.len())]
    );
    assert!(user.content.contains("forty thousand"), "a passage of the knowledge is in it");
    assert!(user.content.ends_with("When may the turbo pump start dry?"), "the question stays after the passages");

    // the chip under the question names the documents; a click opens the passages
    {
        let mut ui = sim(&app);
        ui.find("When may the turbo pump start dry?").expect("the question bubble is drawn");
        ui.find("Using 1 passage from manual.pdf").expect("the chip under the question");
        ui.find("forty thousand").expect_err("the passages are behind a click");
    }
    let _ = app.update(Message::RagToggle(0));
    {
        let mut ui = sim(&app);
        ui.find("manual.pdf, chunk 1/1").expect("the source line of the passage");
        ui.find("The turbo pump runs at forty thousand revolutions per minute and must never be started dry.").expect("the passage text");
    }

    // the system prompt carries the instructions of the project with the Studio's header
    let system = app.system_prompt();
    let header = system.find("Instructions for the project \"Work\":").expect("the header line");
    assert!(system[header..].contains("Answer as an engineer."), "the instructions follow the header");
}

#[test]
fn the_project_instructions_sit_between_an_override_and_the_assistants_own_text() {
    let mut store = Store::default();
    let mut p = project("p1", "Work");
    p.instructions = "Mind the project rules.".into();
    store.projects.push(p);
    store.assistants.push(Assistant {
        id: "a1".into(),
        name: "Helper".into(),
        icon: "bot".into(),
        color: "patina".into(),
        system: "Always be brief.".into(),
        temperature: 0.7,
        top_p: 0.95,
        top_k: 20,
        model: String::new(),
        starters: vec![],
        sys_mode: SysMode::Override,
        sys_text: "You are terse.".into(),
        builtin: false,
    });
    store.prefs.assistant = "a1".into();
    let mut chat = ChatRecord::new(0, "in project", vec![rec(Role::User, "a"), rec(Role::Assistant, "b")]);
    chat.project = Some("p1".into());
    // like a chat that was started while this assistant was chosen (the Studio's t.preset)
    chat.assistant = Some("a1".into());
    store.chats.push(chat);
    store.next_id = 1;
    let (mut app, _dir) = app_with(store);

    let _ = app.update(Message::OpenChat(0));
    let system = app.system_prompt();
    let override_at = system.find("You are terse.").expect("the override is the base");
    let header_at = system.find("Instructions for the project").expect("the project follows");
    let assistant_at = system.find("Always be brief.").expect("the assistant's own text comes last");
    assert!(override_at < header_at && header_at < assistant_at, "order: defaults, project, assistant — got {system:?}");
}

#[test]
fn a_broken_project_is_dropped_and_the_rest_is_clamped_on_load() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    let long_name = "n".repeat(200);
    let long_instructions = "i".repeat(9000);
    let long_text = "t".repeat(700_000);
    let json = format!(
        r#"{{"projects":[{{"id":"","name":"broken"}},{{"id":"p1","name":"{long_name}","instructions":"{long_instructions}",
        "docs":[{{"id":"d1","name":"doc.pdf","text":"{long_text}","chars":9999999}}],"created":1,"updated":2}}],"chats":[],"next_id":1,"theme":"light"}}"#
    );
    std::fs::write(&path, json).unwrap();
    let store = Store::load(&path);
    assert_eq!(store.projects.len(), 1, "the project without an id is dropped");
    let p = &store.projects[0];
    assert_eq!(p.name.chars().count(), fllm_app::rag::limits::NAME, "the name is cut to the limit");
    assert_eq!(p.instructions.chars().count(), fllm_app::rag::limits::INSTRUCTIONS, "the instructions are cut to the limit");
    assert_eq!(p.docs[0].text.chars().count(), fllm_app::rag::MAX_DOC_CHARS, "the document is cut to the limit");
    assert!(p.docs[0].chars <= fllm_app::rag::MAX_DOC_CHARS, "the counted characters follow the text");
}

#[test]
fn the_page_draws_its_sections_and_a_document_can_be_added_and_removed_again() {
    let mut store = Store::default();
    store.projects.push(project("p1", "Work"));
    store.chats.push(ChatRecord::new(0, "in project", vec![rec(Role::User, "a"), rec(Role::Assistant, "b")]));
    store.next_id = 1;
    let (mut app, dir) = app_with(store);

    let _ = app.update(Message::PrjOpen("p1".into()));
    {
        let mut ui = sim(&app);
        ui.find("Work").expect("the name of the project");
        ui.find("Instructions for every chat of this project").expect("the label of the box");
        ui.find("Knowledge").expect("the knowledge section");
        ui.find("No documents yet. Add files that every chat of this project should be able to search.").expect("the empty knowledge");
        ui.find("Add files").expect("the button of the knowledge");
        ui.find("in project").expect("the chat of the project");
        ui.find("New chat in project").expect("the action of the head");
        ui.find("Delete project").expect("the action at the bottom");
    }

    // a document arrives through the route of the server (the dialog answers in the same shape)
    let _ = app.update(Message::PrjExtracted(Ok(("manual.pdf".into(), "Pump rules: never start dry.".into(), Some(3), false))));
    assert!(
        app.toast().is_some_and(|t| t.contains("manual.pdf added to \"Work\"")),
        "the line of prjAddFiles: {:?}",
        app.toast()
    );
    {
        let mut ui = sim(&app);
        ui.find("manual.pdf").expect("the document is listed");
        ui.find("28 B, 3 pages").expect("its size and pages are shown");
    }
    let doc_id = saved(&dir).projects[0].docs[0].id.clone();

    let _ = app.update(Message::PrjDocRemove(doc_id));
    assert!(saved(&dir).projects[0].docs.is_empty(), "the document is gone from the store");
    let mut ui = sim(&app);
    ui.find("No documents yet. Add files that every chat of this project should be able to search.").expect("the knowledge is empty again");
    ui.find("manual.pdf").expect_err("the row is gone");
}


/// Writes the four views of the Projects feature as PNGs into `FLLM_SNAPSHOT_DIR`, so they can be
/// looked at; the asserts keep a blank render from passing.
#[test]
fn the_project_views_are_painted() {
    let mut store = Store::default();
    let mut p = project("p1", "Work");
    p.instructions = "Answer as an engineer.".into();
    p.docs.push(doc("d1", "manual.pdf", "The turbo pump runs at forty thousand revolutions per minute and must never be started dry."));
    store.projects.push(p);
    let mut inside = ChatRecord::new(0, "inside the project", vec![rec(Role::User, "a"), rec(Role::Assistant, "b")]);
    inside.project = Some("p1".into());
    store.chats.push(inside);
    store.chats.push(ChatRecord::new(1, "a chat of its own", vec![rec(Role::User, "c"), rec(Role::Assistant, "d")]));
    store.next_id = 2;
    let (mut app, _dir) = app_with(store);

    // the sidebar with the project and its chat below it
    let _ = app.update(Message::PrjToggle("p1".into()));
    save_png(&app, "prj-sidebar");

    // the page: head, usage, instructions, knowledge with the document, chats, actions
    let _ = app.update(Message::PrjOpen("p1".into()));
    save_png(&app, "prj-page");

    // the dialog that asks before the project goes away
    let _ = app.update(Message::PrjDelete("p1".into()));
    save_png(&app, "prj-delete");
    let _ = app.update(Message::PrjDeleteCancel);

    // a question of the project with its passage chip open under it
    let _ = app.update(Message::PrjDeleteCancel);
    let _ = app.update(Message::PrjNewChat);
    let question = std::sync::Arc::new("When may the turbo pump start dry?".to_string());
    let _ = app.update(Message::Edit(text_editor::Action::Edit(text_editor::Edit::Paste(question))));
    let _ = app.update(Message::Send);
    let _ = app.update(Message::RagToggle(0));
    save_png(&app, "prj-chip");
    let _ = app.update(Message::DeleteChat(0));

    // the picker that moves a chat into a project
    let _ = app.update(Message::PrjClose);
    let _ = app.update(Message::List(ListMsg::Menu(Some(1))));
    let _ = app.update(Message::List(ListMsg::MoveAsk));
    save_png(&app, "prj-pick");
}

/// Writes one PNG of the given view into the snapshot dir of the run (the pattern of `tests/ui.rs`).
fn save_png(app: &App, name: &str) {
    let theme = app.theme();
    let settings = iced::Settings { fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()], default_font: fllm_app::app::UI_FONT, ..Default::default() };
    let mut ui = Simulator::with_size(settings, iced::Size::new(1280.0, 900.0), app.view());
    let shot = ui.snapshot(&theme).expect("the software renderer draws the view");
    let dir = std::path::PathBuf::from(std::env::var("FLLM_SNAPSHOT_DIR").unwrap_or_else(|_| format!("{}/ui-snapshots", env!("CARGO_TARGET_TMPDIR"))));
    std::fs::create_dir_all(&dir).unwrap();
    let prefix = format!("{name}-");
    for old in std::fs::read_dir(&dir).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with(&prefix)) {
        let _ = std::fs::remove_file(old.path());
    }
    shot.matches_image(dir.join(format!("{name}.png"))).expect("the picture is written");
    assert!(std::fs::read_dir(&dir).unwrap().flatten().any(|e| e.file_name().to_string_lossy().starts_with(&prefix)), "the picture exists");
}
