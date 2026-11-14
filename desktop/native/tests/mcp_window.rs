//! Settings > MCP servers in the window: the list of what runs, *Find servers* and the form that
//! writes a server into `mcp.json` (`spane-mcp`).

use fllm_app::app::{App, Message, View};
use fllm_app::mcp::TOOLS_HELP;
use fllm_app::store::Store;
use iced::Size;
use iced_test::Simulator;
use serde_json::json;

fn settings() -> iced::Settings {
    iced::Settings {
        fonts: vec![fllm_app::app::FONT_REGULAR.into(), fllm_app::app::FONT_SEMIBOLD.into()],
        default_font: fllm_app::app::UI_FONT,
        ..Default::default()
    }
}

fn app(theme: &str) -> (App, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.json");
    Store { theme: theme.into(), ..Default::default() }.save(&path).unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Err("no server at http://127.0.0.1:9: connection refused".into())));
    (app, dir)
}

fn sim(app: &App) -> Simulator<'_, Message> {
    Simulator::with_size(settings(), Size::new(1280.0, 6000.0), app.view())
}

/// The three cards start the way the page builds them: the description and the warning of the
/// first one, nothing to show yet, and the form with its six fields.
#[test]
fn the_three_cards_start_the_way_the_page_builds_them() {
    let (mut app, _dir) = app("dark");
    let _ = app.update(Message::Go(View::Settings));

    let mut ui = sim(&app);
    ui.find("Tools (MCP)").expect("the title of the first card");
    ui.find(TOOLS_HELP).expect("the description as one sentence");
    // the warning and the description of the second card are `rich_text` — the window cannot look
    // inside one, so their sentences are held by the unit test of the module
    ui.find("No MCP servers yet. Add one below.").expect("the empty list");
    ui.find("Find servers").expect("the title of the second card");
    // the two links of this description are spans of a `rich_text`, which the window cannot look
    // inside — the sentence and both links are held by the unit test of the module
    ui.find("Search").expect("the button of the search");
    ui.find("Add a server").expect("the title of the third card");
    ui.find("Name").expect("the first field");
    ui.find("URL (a remote server instead of a command)").expect("the second field");
    ui.find("Command").expect("the third field");
    ui.find("Arguments (one per line)").expect("the fourth field");
    ui.find("Environment variables (NAME=value, one per line)").expect("the fifth field");
    ui.find("HTTP headers for a remote server (NAME=value, one per line)").expect("the sixth field");
    ui.find("Save and start").expect("the button that saves");
    ui.find("Clear").expect("the button beside it");
    assert!(
        ui.find("Show more").is_err(),
        "there is nothing to show more of before an answer came"
    );
    assert!(
        ui.find("Saved in").is_err(),
        "the line under the form comes with the list"
    );
}

/// The list follows its answer, and the buttons of the form work on it the way the page's do.
#[test]
fn the_list_follows_its_answer_and_the_form_works() {
    let (mut app, _dir) = app("light");
    let _ = app.update(Message::Go(View::Settings));

    let answer: fllm_app::mcp::ServersAnswer = serde_json::from_value(json!({
        "servers": [
            {
                "name": "files",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem"],
                "url": "",
                "headers": [],
                "env_keys": ["API_KEY"],
                "state": "ready",
                "message": null,
                "tool_count": 1,
                "tools": [{"name": "read_file", "description": "Read one file"}],
                "stderr": null
            },
            {
                "name": "remote",
                "command": "",
                "args": [],
                "url": "https://mcp.example.com/mcp",
                "headers": [],
                "env_keys": [],
                "state": "starting",
                "message": "waiting for the port",
                "tool_count": 0,
                "tools": [],
                "stderr": "some noise"
            }
        ],
        "config_file": "/home/m/.fllm/mcp.json",
        "config_error": null
    }))
    .expect("an answer of the servers");
    let _ = app.update(Message::McpLoaded(Ok(answer)));

    let mut ui = sim(&app);
    ui.find("Saved in /home/m/.fllm/mcp.json").expect("the line under the form");
    ui.find("files").expect("the name of the first server");
    ui.find("ready - 1 tool",).expect("its state with one tool, and `tool` for one");
    ui.find("npx -y @modelcontextprotocol/server-filesystem").expect("the line under it");
    ui.find("Tools").expect("the tools of the server are drawn open");
    ui.find("read_file").expect("the name of one of them");
    ui.find("remote").expect("the name of the second server");
    ui.find("starting").expect("its state");
    ui.find("HTTP https://mcp.example.com/mcp").expect("its address instead of a command");
    ui.find("waiting for the port").expect("the message under it");
    ui.find("some noise").expect("the error output of the server");
    assert!(
        ui.find("No MCP servers yet. Add one below.").is_err(),
        "the list has something in it now"
    );
    drop(ui);

    // Edit fills the six boxes and changes the title of the form
    let _ = app.update(Message::McpEdit("files".into()));
    assert_eq!(app.mcp_name, "files", "the name is in the box and locked");
    assert_eq!(app.mcp_args.text(), "-y\n@modelcontextprotocol/server-filesystem");
    assert_eq!(app.mcp_env.text(), "API_KEY=", "the values stay on disk");
    assert_eq!(
        app.mcp_env_hint,
        "Saved values are not shown. A variable left without a value keeps its saved value."
    );
    let mut ui = sim(&app);
    ui.find("Edit files").expect("the title of the form while a server is edited");
    drop(ui);

    // Clear puts the form back
    let _ = app.update(Message::McpClear);
    assert!(app.mcp_name.is_empty() && app.mcp_env.text().is_empty(), "the boxes are empty");

    // a line that is not NAME=value stops the save with the toast of the page
    // the box takes the text one character at a time, the way a person types it
    for c in "BROKEN".chars() {
        let _ = app.update(Message::McpEnv(iced::widget::text_editor::Action::Edit(
            iced::widget::text_editor::Edit::Insert(c),
        )));
    }
    let _ = app.update(Message::McpSave);
    assert_eq!(
        app.toast.as_deref(),
        Some("Environment line \"BROKEN\" needs the form NAME=value"),
        "the toast of `mcpSave`"
    );

    // the question of the page appears with the name of the server in it
    let _ = app.update(Message::McpRemove("files".into()));
    let mut ui = sim(&app);
    ui.find("Remove the MCP server \"files\"?").expect("the question of `confirmDialog`");
    drop(ui);
    let _ = app.update(Message::McpRemoveCancel);
    let mut ui = sim(&app);
    assert!(
        ui.find("Remove the MCP server \"files\"?").is_err(),
        "the answer of Cancel puts the question away"
    );
}

/// The search of the registry: its answer, its lines and *Use* filling the form with it.
#[test]
fn the_registry_search_fills_the_form() {
    let (mut app, _dir) = app("light");
    let _ = app.update(Message::Go(View::Settings));

    let _ = app.update(Message::RegFound(Ok(serde_json::from_value(json!({
        "source": "official",
        "servers": [
            {
                "title": "GitHub MCP",
                "verified": true,
                "version": "1.2.0",
                "description": "GitHub tools",
                "candidates": [
                    {
                        "kind": "local",
                        "label": "npm github",
                        "name": "github",
                        "command": "npx",
                        "args": ["-y", "@github/mcp-server"],
                        "env": []
                    }
                ]
            },
            { "title": "One without a package", "candidates": [] }
        ],
        "next": "cursor-2"
    }))
    .expect("an answer of the registry"))));
    assert_eq!(app.reg_status, "2 servers.", "the line of the page after an answer");

    let mut ui = sim(&app);
    ui.find("2 servers.").expect("the status line");
    ui.find("GitHub MCP").expect("the title of the first one");
    ui.find("verified").expect("the chip of it");
    ui.find("1.2.0").expect("its version");
    ui.find("GitHub tools").expect("its description");
    ui.find("Use: npm github").expect("the button that fills the form");
    ui.find("One without a package").expect("the second one");
    ui.find(
        "fLLM cannot run this one yet (no npm, Python or container package and no HTTP address).",
    )
    .expect("the sentence for one fLLM cannot run");
    ui.find("Show more").expect("the answer had a cursor, so more may come");
    drop(ui);

    // *Use* clears the form first and then fills it, with the line under the list
    let _ = app.update(Message::RegUse(0, 0));
    assert_eq!(app.mcp_name, "github");
    assert_eq!(app.mcp_command, "npx");
    assert_eq!(app.mcp_args.text(), "-y\n@github/mcp-server");
    assert_eq!(
        app.reg_status,
        "Filled in the form below with \u{201c}GitHub MCP\u{201d}. Check it, then press Save."
    );
    let mut ui = sim(&app);
    ui.find(app.reg_status.as_str()).expect("the line of *Use*");
    drop(ui);

    // an answer that came later is added to what was found before
    let _ = app.update(Message::RegMoreLoaded(Ok(serde_json::from_value(json!({
        "servers": [{ "title": "The next page", "candidates": [] }],
        "next": null
    }))
    .expect("more of the registry"))));
    assert_eq!(app.reg_status, "3 servers.", "the page counts what it has together");
    let mut ui = sim(&app);
    ui.find("The next page").expect("the server of the second page");
    assert!(ui.find("Show more").is_err(), "the cursor is used up");
    drop(ui);

    // an answer the registry refused says so the way the page says it
    let _ = app.update(Message::RegFound(Err("Request failed (503)".into())));
    let mut ui = sim(&app);
    ui.find("The registry did not answer.").expect("the sentence of the page");
    drop(ui);
}

/// The Resources button of a row: the sentence while it reads, the list of what came, the text of
/// one of them and the sentence for a server that publishes none.
#[test]
fn the_resources_of_a_row_are_read_and_shown() {
    let (mut app, _dir) = app("light");
    let _ = app.update(Message::Go(View::Settings));

    // the box the Resources button fills sits in the row of its server, so the row has to be
    // there before it is read
    let _ = app.update(Message::McpLoaded(Ok(serde_json::from_value(json!({
        "servers": [
            {"name": "files", "command": "npx", "args": [], "url": "",
             "headers": [], "env_keys": [], "state": "ready", "message": null,
             "tool_count": 0, "tools": [], "stderr": null},
            {"name": "other", "command": "", "args": [], "url": "",
             "headers": [], "env_keys": [], "state": "ready", "message": null,
             "tool_count": 0, "tools": [], "stderr": null}
        ],
        "config_file": "/home/m/.fllm/mcp.json"
    }))
    .expect("an answer of the servers"))));

    let _ = app.update(Message::McpResources("files".into()));
    let mut ui = sim(&app);
    ui.find("Reading the resources ...").expect("the sentence before the answer");
    drop(ui);

    let _ = app.update(Message::McpResourcesLoaded(
        "files".into(),
        Ok(serde_json::from_value(json!({
            "resources": [
                {"uri": "file:///notes.txt", "name": "Notes", "mime_type": "text/plain", "description": "the notes"}
            ]
        }))
        .expect("an answer of the resources")),
    ));
    let mut ui = sim(&app);
    ui.find("1 resource(s)").expect("the count of them");
    ui.find("file:///notes.txt").expect("the address of the resource");
    ui.find("Notes [text/plain] the notes").expect("what it is");
    ui.find("Read").expect("the button that reads it");
    drop(ui);

    // reading one gives its text under the list
    let _ = app.update(Message::McpRead("files".into(), "file:///notes.txt".into()));
    let mut ui = sim(&app);
    ui.find("Reading ...").expect("the sentence before the text");
    drop(ui);
    let _ = app.update(Message::McpReadLoaded(
        "files".into(),
        "file:///notes.txt".into(),
        Ok("the words of the file".into()),
    ));
    let mut ui = sim(&app);
    ui.find("the words of the file").expect("the text of the resource");
    drop(ui);

    // a server that publishes nothing says so
    let _ = app.update(Message::McpResourcesLoaded("other".into(), Ok(Default::default())));
    let _ = app.update(Message::McpResources("other".into()));
    let mut ui = sim(&app);
    ui.find("Reading the resources ...").expect("the other row reads too");
    drop(ui);
    let _ = app.update(Message::McpResourcesLoaded("other".into(), Ok(Default::default())));
    let mut ui = sim(&app);
    ui.find("This server publishes no resources.").expect("the sentence for a server without any");
    drop(ui);
}
