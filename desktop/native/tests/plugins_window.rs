//! Settings > Plugins, Hooks and Sub-agents in the window: the three cards `plugins.js`,
//! `hooks.js` and `agents.js` build (`spane-plugins`, `spane-hooks`, `spane-agents`).

use fllm_app::app::{App, Message, View};
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
    Store { theme: theme.into(), ..Default::default() }
        .save(&path)
        .unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Err(
        "no server at http://127.0.0.1:9: connection refused".into(),
    )));
    (app, dir)
}

fn sim(app: &App) -> Simulator<'_, Message> {
    Simulator::with_size(settings(), Size::new(1280.0, 14000.0), app.view())
}

/// One plugin of the list of `refresh()`.
fn plugins_answer() -> fllm_app::plugins::PluginsAnswer {
    serde_json::from_value(json!({
        "folder": "/home/m/.fllm/plugins",
        "plugins": [{
            "name": "review-tools",
            "version": "1.2.0",
            "description": "Checks a change before it is committed.",
            "skills": 2,
            "agents": 1,
            "commands": 0,
            "hooks": 0,
            "author": "ada",
            "path": "/home/m/.fllm/plugins/review-tools"
        }]
    }))
    .expect("an answer of the plugins")
}

/// The three cards start the way the page builds them: their titles, their descriptions, the
/// empty lists and the sentences over them.
#[test]
fn the_three_cards_start_the_way_the_page_builds_them() {
    let (mut app, _dir) = app("dark");
    let _ = app.update(Message::Go(View::Settings));

    let mut ui = sim(&app);
    // the one card of the plugins pane
    ui.find("Plugins").expect("the title of the plugins card");
    ui.find(
        "A plugin is a folder that brings several things at once: skills (skills/<name>/SKILL.md), sub-agents (agents/<name>.md), slash commands (commands/<name>.md) and hooks (hooks.json), described by plugin.json. Nothing is copied into your own files: the contributions stay in the plugin folder, so removing the plugin removes everything it added.",
    )
    .expect("the description of the plugins card as one sentence");
    ui.find("No plugins installed.").expect("the empty list of plugins");
    ui.find("Install from folder").expect("the first install button");
    ui.find("Install from repository").expect("the second install button");

    // the two paragraphs over the list of hooks
    ui.find(fllm_app::hooks::DESC1).expect("the first paragraph of hooks");
    ui.find(fllm_app::hooks::DESC2).expect("the second paragraph of hooks");
    ui.find("Hooks").expect("the title of the hooks card");
    for (event, title, hint) in fllm_app::hooks::EVENTS {
        ui.find(title).expect("the title of an event");
        ui.find(hint).expect("the hint under it");
        ui.find(event).expect("the name of the event in mono next to its title");
    }
    ui.find("+ Add hook").expect("the button under a list of hooks");
    ui.find("Save").expect("the button that writes the list");
    ui.find("Reload").expect("the button that reads it again");
    assert!(
        ui.find("From plugins").is_err(),
        "the block of the plugins is not there while no plugin hook came"
    );
    drop(ui);

    // the card of sub-agents with its two built-ins and its empty list
    let mut ui = sim(&app);
    ui.find("Sub-agents").expect("the title of the card");
    ui.find(
        "Helpers that the coding agent can hand a job to. Each works on its own with its own instructions and reports back.",
    )
    .expect("the description of the card");
    ui.find("New sub-agent").expect("the button over the list");
    ui.find("Built in").expect("the first section of the list");
    ui.find("explore").expect("the first built-in");
    ui.find("Reads and searches the project and reports what it found. It cannot change anything.")
        .expect("what the first built-in does");
    ui.find("general").expect("the second built-in");
    ui.find("Your sub-agents").expect("the second section");
    ui.find(
        "You have not created one yet. A sub-agent is useful for a job you repeat, for example reviewing a change or writing tests. Choose New sub-agent to make one.",
    )
    .expect("the empty list of your own sub-agents");
}

/// The plugins card: what an answer holds, what the three buttons say and the question before a
/// plugin is taken away.
#[test]
fn the_plugins_card_works_the_way_the_page_works() {
    let (mut app, _dir) = app("dark");
    let _ = app.update(Message::Go(View::Settings));
    let _ = app.update(Message::PluginsLoaded(Ok(plugins_answer())));

    let mut ui = sim(&app);
    ui.find("review-tools").expect("the name of the plugin");
    ui.find("v1.2.0").expect("the version behind the name");
    ui.find("Checks a change before it is committed.").expect("what it is for");
    ui.find("2 skill(s) · 1 sub-agent(s) · by ada")
        .expect("what it brings, with only the counts the page keeps");
    ui.find("/home/m/.fllm/plugins/review-tools").expect("where it lies");
    ui.find("Plugins live in /home/m/.fllm/plugins").expect("the folder line");
    drop(ui);

    // the path box decides what an install says, as the page's `value.trim()` does
    let _ = app.update(Message::PluginPath("   ".into()));
    let _ = app.update(Message::PluginInstall);
    assert_eq!(
        app.plugins_status,
        "Give the absolute path of the plugin folder.",
        "nothing to install without a path"
    );
    let _ = app.update(Message::PluginPath("/srv/plugins/one".into()));
    let _ = app.update(Message::PluginInstall);
    assert_eq!(app.plugins_status, "Installing ...", "the words before the answer");
    let _ = app.update(Message::PluginInstalled(Ok(json!({}))));
    assert!(app.plugin_path.is_empty(), "the box is emptied");
    assert_eq!(
        app.plugins_status,
        "Installed. Its skills, agents, commands and hooks are available now.",
        "the words after it"
    );

    // the repository row has its own two sentences
    let _ = app.update(Message::PluginClone);
    assert_eq!(
        app.plugins_status,
        "Give the address of a repository with a plugin.json."
    );
    let _ = app.update(Message::PluginUrl("https://github.com/ada/one".into()));
    let _ = app.update(Message::PluginClone);
    assert_eq!(app.plugins_status, "Cloning ...");
    let _ = app.update(Message::PluginCloned(Ok(json!({}))));
    assert!(app.plugin_url.is_empty() && app.plugin_ref.is_empty());
    assert_eq!(
        app.plugins_status,
        "Installed from the repository. Its skills, agents, commands and hooks are available now."
    );

    // a failure of the server is shown the way it was written
    let _ = app.update(Message::PluginInstalled(Err("not a folder with a plugin.json".into())));
    assert_eq!(app.plugins_status, "not a folder with a plugin.json");
    let _ = app.update(Message::PluginsLoaded(Err(
        "no server at http://127.0.0.1:9: connection refused".into(),
    )));
    assert_eq!(
        app.plugins_status,
        "Could not read the plugins: no server at http://127.0.0.1:9: connection refused",
        "the catch of refresh()"
    );

    // Remove asks first, with the words of the page
    let _ = app.update(Message::PluginRemoveAsk("review-tools".into()));
    let mut ui = sim(&app);
    ui.find("Remove the plugin \"review-tools\" and everything it contributes?")
        .expect("the question of confirmDialog");
    drop(ui);
    let _ = app.update(Message::PluginRemoveOk);
    assert!(app.plugin_remove.is_none(), "the question is answered");
    let _ = app.update(Message::PluginRemoved("review-tools".into(), Ok(json!({}))));
    assert_eq!(app.plugins_status, "Removed review-tools.");
    let _ = app.update(Message::PluginRemoveCancel);
}

/// The hooks card: the list as an answer brings it, the rows as the page edits them, and the
/// lines of a test.
#[test]
fn the_hooks_card_works_the_way_the_page_works() {
    let (mut app, _dir) = app("dark");
    let _ = app.update(Message::Go(View::Settings));
    let _ = app.update(Message::HooksLoaded(Ok(serde_json::from_value(json!({
        "hooks": {
            "pre_tool_call": [{
                "matcher": "bash",
                "command": "cargo fmt --quiet",
                "prompt": "",
                "on": "",
                "timeout_seconds": 45,
                "enabled": true
            }]
        },
        "file": "/home/m/.fllm/hooks.json",
        "from_plugins": [{
            "event": "stop",
            "matcher": null,
            "command": "echo done",
            "plugin": "review-tools",
            "enabled": true,
            "timeout_seconds": 12
        }]
    }))
    .expect("an answer of the hooks"))));

    assert_eq!(
        app.hooks_rows[0].len(),
        1,
        "the hook of the first event is a row"
    );
    assert_eq!(app.hooks_rows[0][0].command, "cargo fmt --quiet");
    assert_eq!(app.hooks_rows[0][0].timeout, "45", "the box holds the timeout");
    assert!(
        app.hooks_status.is_empty(),
        "the status line is empty after a load that worked"
    );

    let mut ui = sim(&app);
    // what stands in a box is the state of the row — a `text_input` has no text of the tree to
    // look at, which the assertions above hold
    ui.find("/home/m/.fllm/hooks.json").expect("the file the list lives in");
    ui.find("From plugins").expect("the block of the hooks a plugin brings");
    ui.find(
        "These run like your own hooks. Remove or change them in the plugin (Settings > Plugins), not here.",
    )
    .expect("what the block says");
    ui.find("echo done").expect("the command of the plugin hook");
    ui.find("review-tools").expect("the plugin it comes from");
    ui.find("12 s").expect("how long it may run");
    drop(ui);

    // a row is typed into and taken away the way the page does it
    let _ = app.update(Message::HookField(
        0,
        0,
        fllm_app::hooks::HookField::Command,
        "cargo clippy --quiet".into(),
    ));
    assert_eq!(app.hooks_rows[0][0].command, "cargo clippy --quiet");
    let _ = app.update(Message::HookAdd(0));
    assert_eq!(app.hooks_rows[0].len(), 2, "+ Add hook appends an empty row");
    assert_eq!(
        app.hooks_rows[0][1].timeout, "30",
        "a new row starts with the 30 seconds of the page"
    );
    assert!(app.hooks_rows[0][1].enabled, "and its switch starts on");
    let _ = app.update(Message::HookRemove(0, 1));
    assert_eq!(app.hooks_rows[0].len(), 1);
    assert_eq!(
        app.hooks_status,
        "Removed. Press Save to write it.",
        "the words after a removal"
    );

    // Save sends what is kept and says what the page says
    let _ = app.update(Message::HooksSave);
    let _ = app.update(Message::HooksSaved(Ok(json!({ "hooks": {} }))));
    assert_eq!(
        app.hooks_status,
        "Saved. The next tool call runs them.",
        "the words of save()"
    );
    assert!(
        app.hooks_rows.iter().all(|rows| rows.is_empty()),
        "the answer the server wrote holds no hooks"
    );

    // a test on a list without a hook does nothing, as `if (!hook) return;` does
    let _ = app.update(Message::HookTest(0, 0));
    assert_eq!(
        app.hooks_status,
        "Saved. The next tool call runs them.",
        "the line is untouched"
    );

    // a hook that runs says what it ran
    let _ = app.update(Message::HookAdd(0));
    let _ = app.update(Message::HookField(
        0,
        0,
        fllm_app::hooks::HookField::Command,
        "cargo fmt --quiet".into(),
    ));
    let _ = app.update(Message::HookTest(0, 0));
    assert_eq!(app.hooks_status, "Running cargo fmt --quiet ...");
    let _ = app.update(Message::HookTested(Ok(json!({
        "blocked": false,
        "ran": [{ "command": "cargo fmt --quiet", "exit_code": 0, "milliseconds": 9, "stdout": "" }]
    }))));
    assert_eq!(
        app.hooks_status,
        "passed\n  cargo fmt --quiet -> exit 0 (9 ms)",
        "the lines of test()"
    );

    let _ = app.update(Message::HooksLoaded(Err("connection refused".into())));
    assert_eq!(
        app.hooks_status,
        "Could not load the hooks: connection refused",
        "the catch of load()"
    );
}

/// The sub-agents card: the list with its sections, the form behind the buttons and the words
/// they say.
#[test]
fn the_sub_agents_card_works_the_way_the_page_works() {
    let (mut app, _dir) = app("dark");
    let _ = app.update(Message::Go(View::Settings));
    let _ = app.update(Message::AgentsLoaded(Ok(serde_json::from_value(json!({
        "agents": [
            {
                "name": "reviewer",
                "description": "Reviews a change for bugs and missing tests.",
                "body": "You review code changes.",
                "tools": ["bash", "read_file"],
                "model": "llama-4",
                "max_steps": 8,
                "scope": "workspace",
                "read_only": false
            },
            {
                "name": "brought",
                "description": "",
                "body": "",
                "tools": [],
                "model": "",
                "max_steps": 0,
                "scope": "user",
                "read_only": true
            }
        ],
        "folders": { "user": "/home/m/.fllm/agents" }
    }))
    .expect("an answer of the sub-agents"))));

    let mut ui = sim(&app);
    ui.find("reviewer").expect("your own sub-agent");
    ui.find("this folder only").expect("the chip of a folder-bound one");
    ui.find("Reviews a change for bugs and missing tests.").expect("what it is for");
    ui.find("can change files or run commands (2 tools) · model llama-4")
        .expect("what it may do, with its model");
    ui.find("From plugins").expect("the third section of the list");
    ui.find("brought").expect("the sub-agent of a plugin");
    ui.find("from a plugin").expect("its chip");
    ui.find("Stored as one file per sub-agent in").expect("the line under the list");
    ui.find("the agents folder").expect("the button that copies the path");
    ui.find("(click to copy the path).").expect("what the button does");
    drop(ui);

    // the form waits for the tools the page reads first (`startEdit` awaits `loadTools`)
    let _ = app.update(Message::AgentNew);
    assert!(app.agent_editing.is_none(), "no form before the tools came");
    assert!(app.agent_pending.is_some(), "the form is waiting");
    let _ = app.update(Message::AgentToolsLoaded(Ok(fllm_app::agents::builtin_tools())));
    let edit = app.agent_editing.as_ref().expect("the form is open now");
    assert_eq!(
        edit.checked,
        vec![
            "glob".to_string(),
            "grep".to_string(),
            "list_dir".to_string(),
            "read_file".to_string()
        ],
        "a new one starts with everything that only reads"
    );
    assert!(edit.is_new, "and nothing is filled in for it");

    let mut ui = sim(&app);
    for line in [
        "New sub-agent",
        "Name",
        "The coding agent picks it by this name. Lowercase letters, digits, - and _.",
        "When to use it",
        "One sentence. The coding agent reads it to decide whether this sub-agent fits a job.",
        "Instructions",
        "The system prompt of this sub-agent. It does not see your conversation, only the job it is given.",
        "Allowed tools",
        "Read and search",
        "Change files and run commands",
        "A sub-agent with only reading tools cannot change your files.",
        "Model",
        "Leave empty to use the model that runs the session.",
        "Step limit",
        "Most tool calls before it must report. 0 means the default (16).",
        "Available in",
        "All folders",
        "Only the open folder (open a folder on the Code page first)",
        "Create sub-agent",
        "Cancel",
        "Try the instructions",
    ] {
        ui.find(line).expect("a line of the form");
    }
    drop(ui);

    // the box of the sample opens with its own words
    let _ = app.update(Message::AgentTryOpen);
    let mut ui = sim(&app);
    ui.find(
        "Sends a sample job to the loaded model with these instructions. It has no tools here, so it can only answer in words.",
    )
    .expect("what the sample does");
    ui.find("Run the sample").expect("the button of the sample");
    drop(ui);

    // the two guards of the sample, then the answer
    let _ = app.update(Message::AgentTryRun);
    assert_eq!(
        app.agent_editing.as_ref().unwrap().status,
        "Write the instructions first."
    );
    let _ = app.update(Message::AgentBody(iced::widget::text_editor::Action::Edit(
        iced::widget::text_editor::Edit::Insert('x'),
    )));
    let _ = app.update(Message::AgentTryRun);
    assert_eq!(
        app.agent_editing.as_ref().unwrap().status,
        "Describe a sample job to try."
    );
    let _ = app.update(Message::AgentTried(Ok("(no answer)".into())));
    assert_eq!(
        app.agent_editing.as_ref().unwrap().try_out.as_deref(),
        Some("(no answer)"),
        "the words of an answer without words"
    );

    // the form refuses a name the page refuses, with its words
    let _ = app.update(Message::AgentSave);
    assert_eq!(
        app.agent_editing.as_ref().unwrap().status,
        "Give the sub-agent a short name: lowercase letters, digits, - and _ (for example code-reviewer)."
    );
    let _ = app.update(Message::AgentName("mine".into()));
    let _ = app.update(Message::AgentSave);
    let _ = app.update(Message::AgentSaved(
        "mine".to_string(),
        true,
        Ok(json!({})),
    ));
    assert!(app.agent_editing.is_none(), "the form closes");
    assert_eq!(
        app.agents_notice,
        "Created \"mine\". The coding agent can use it from its next run.",
        "what the list says after it"
    );

    // editing an existing one keeps its name and says the shorter sentence
    let _ = app.update(Message::AgentEdit("reviewer".into(), "workspace".into()));
    let _ = app.update(Message::AgentToolsLoaded(Ok(fllm_app::agents::builtin_tools())));
    let edit = app.agent_editing.as_ref().expect("the form is open");
    assert!(!edit.is_new);
    assert_eq!(edit.name, "reviewer", "the name box is not part of the form now");
    assert_eq!(edit.checked, vec!["bash".to_string(), "read_file".to_string()]);
    assert_eq!(edit.max_steps, "8");
    assert!(edit.workspace_scope, "its scope is kept");
    let _ = app.update(Message::AgentSaved(
        "reviewer".to_string(),
        false,
        Ok(json!({})),
    ));
    assert_eq!(app.agents_notice, "Saved \"reviewer\".");

    // Delete asks first, with the words of the page
    let _ = app.update(Message::AgentDeleteAsk("reviewer".into(), "workspace".into()));
    let mut ui = sim(&app);
    ui.find("Delete the sub-agent \"reviewer\"? Its file is removed.")
        .expect("the question of confirmDialog");
    drop(ui);
    let _ = app.update(Message::AgentDeleteOk);
    assert!(app.agent_delete.is_none());
    let _ = app.update(Message::AgentDeleted("reviewer".into(), Ok(json!({}))));
    assert_eq!(app.agents_notice, "Deleted \"reviewer\".");

    // the path of the folder goes to the clipboard and says so
    let _ = app.update(Message::AgentPathCopy);
    assert_eq!(app.agents_notice, "Path copied.");

    // a list that could not be read says so the way the catch of load() does
    let _ = app.update(Message::AgentsLoaded(Err(
        "the server answered with error 500".into(),
    )));
    assert_eq!(
        app.agents_notice,
        "The sub-agents could not be loaded: the server answered with error 500"
    );
}
