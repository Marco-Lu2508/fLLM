//! Settings > Skills in the window: the list, what the agent learned, the form, the tasks, the
//! task form and the two cards `skilllibrary.js` appends (`spane-skills`).

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
    Store { theme: theme.into(), ..Default::default() }.save(&path).unwrap();
    let (mut app, _task) = App::with("http://127.0.0.1:9".into(), path);
    let _ = app.update(Message::Connected(Err("no server at http://127.0.0.1:9: connection refused".into())));
    (app, dir)
}

fn sim(app: &App) -> Simulator<'_, Message> {
    Simulator::with_size(settings(), Size::new(1280.0, 9000.0), app.view())
}

/// All seven cards start the way the page builds them: their titles, their buttons and the empty
/// sentences of the lists that hold nothing yet.
#[test]
fn the_seven_cards_start_the_way_the_page_builds_them() {
    let (mut app, _dir) = app("dark");
    let _ = app.update(Message::Go(View::Settings));
    // the line over the tasks comes with the list, which the page reads on every click of its
    // menu entry
    let _ = app.update(Message::ScLoaded(Ok(serde_json::from_value(json!({
        "clock_running": false, "note": "", "tasks": []
    }))
    .expect("an answer of the schedule"))));

    let mut ui = sim(&app);
    for title in [
        "Skills",
        "Skill improvements",
        "Write a skill",
        "Scheduled tasks",
        "New task",
        "Skills that come with the app",
        "Find skills on GitHub",
    ] {
        ui.find(title).expect("the title of a card of the pane");
    }
    ui.find(
        "A skill is a saved procedure (a Markdown file) that the model can read when a task matches it, and write itself after a task that is worth repeating. Skills of the user apply everywhere; skills of a workspace folder are kept in that folder.",
    )
    .expect("the description of the first card as one sentence");
    ui.find(
        "The coding agent learns from your Code sessions on its own (switch it off in Settings > Personalization). When it reads a skill, how it went is noted: finished or failed, whether the checks passed, retries, and your thumbs. After a failure, a thumb down, many retries or five uses it revises the skill itself: the previous text is kept, and if one of the next two uses of the revised skill goes badly the previous text is put back. After a complex task that went well it writes a new skill. A skill you wrote or edited is never overwritten: the agent writes a copy named NAME-agent next to it. Every change is listed below and under What I learned with an Undo button. The model that is loaded writes the drafts, so a small model makes small changes.",
    )
    .expect("the long description of the second card as one sentence");
    ui.find("One click writes them into your skills folder — as a normal file you can change afterwards.")
        .expect("the description of the library card");
    ui.find("No skills yet. Write one below, or let the model propose one after a long task.")
        .expect("the empty list of skills");
    ui.find("Nothing was learned automatically yet.").expect("the empty feed");
    ui.find("Nothing to show yet. Switch on \"Learn from this session\" in the Code page: when the agent reads a skill, how it went is noted here, and after a failure or a few uses it can draft a better version.")
        .expect("the empty list of revisions");
    ui.find("No scheduled tasks yet.").expect("the empty list of tasks");
    ui.find("The library is empty.").expect("the empty library");
    ui.find("Save skill").expect("the button of the form");
    ui.find("New from template").expect("the button beside it");
    ui.find("Save task").expect("the button of the task form");
    ui.find("Consolidate now").expect("the button under the lists");
    ui.find("Import from URL").expect("the button of the library");
    ui.find("Show skills").expect("the button of the finder");
    ui.find("Save token").expect("the button of the directory");
    ui.find("Remove it").expect("the button beside it");
    ui.find("Search").expect("the button of the directory search");
    ui.find(
        "Also accepted: \"every 2 hours\", \"hourly at :15\", or a cron line (minute hour day month weekday). Local time.",
    )
    .expect("the sentence under the schedule box");
    ui.find(
        "The scheduler starts when a model is loaded. Tasks can be defined now.",
    )
    .expect("the line over the tasks before the clock runs");
    assert!(
        ui.find("Next runs:").is_err(),
        "the preview of a schedule comes with a box that holds one"
    );
}

/// The list follows its answer, the form of a skill takes what the server has, and deleting asks
/// first with the question of the page.
#[test]
fn the_list_and_the_form_work_the_way_the_page_works() {
    let (mut app, _dir) = app("light");
    let _ = app.update(Message::Go(View::Settings));

    let answer: fllm_app::skills::SkillsAnswer = serde_json::from_value(json!({
        "skills": [{"name": "my-skill", "description": "What it does", "when": "for tests", "scope": "user"}],
        "user_folder": "/home/m/.fllm/skills"
    }))
    .expect("an answer of the skills");
    let _ = app.update(Message::SkLoaded(Ok(answer)));

    let mut ui = sim(&app);
    ui.find("my-skill").expect("the name of the skill");
    ui.find("user").expect("the chip of its scope");
    ui.find("What it does Use when: for tests").expect("what it is for");
    ui.find("Your skills are saved in /home/m/.fllm/skills. Workspace skills are saved in <folder>/.fllm/skills/<name>/SKILL.md.")
        .expect("the line under the list");
    drop(ui);

    // Edit reads the skill and puts it into the form
    let _ = app.update(Message::SkEdit("my-skill".into(), "user".into()));
    let _ = app.update(Message::SkRead(
        "my-skill".into(),
        "user".into(),
        Ok("---\nname: my-skill\n---\n1. Step\n".into()),
    ));
    assert_eq!(app.sk_name, "my-skill");
    assert_eq!(app.sk_text.text(), "---\nname: my-skill\n---\n1. Step\n");

    // the template starts the box the way the page starts it
    let _ = app.update(Message::SkNew);
    assert!(app.sk_name.is_empty(), "the name is cleared with it");
    assert_eq!(app.sk_scope, "user");
    assert_eq!(
        app.sk_text.text(),
        fllm_app::skills::template(),
        "the box holds the template of the page"
    );

    // a name that is no skill name stops the save with the toast of the page
    app.sk_name = "Not a name".into();
    let _ = app.update(Message::SkSave);
    assert_eq!(
        app.toast.as_deref(),
        Some("The name uses lowercase letters, digits, - and _ (up to 64 characters)."),
        "the toast of `skSave`"
    );

    // The name of the front matter goes into the empty name box while its text is written
    // (`skText` input of the page). The parse that decides it is held by the unit test of the
    // module; a keystroke in this window lands where the cursor stands — at the front of the box,
    // where the front matter begins — so the box is not typed into here.

    // deleting asks first, with the name in the question
    let _ = app.update(Message::SkDelete("my-skill".into(), "user".into()));
    let mut ui = sim(&app);
    ui.find("Delete the skill \"my-skill\"?").expect("the question of `askDialog`");
    ui.find("The file is removed from disk.").expect("the sentence under it");
    drop(ui);
    let _ = app.update(Message::SkDeleteOk);
    let _ = app.update(Message::SkDeleted(
        "my-skill".into(),
        "user".into(),
        Ok(serde_json::json!({})),
    ));
    assert!(app.sk_remove.is_none(), "the dialog is away again");
}

/// The list of tasks, the answers under a row and the preview of what a schedule means.
#[test]
fn the_tasks_and_their_answers_are_drawn_the_way_the_page_draws_them() {
    let (mut app, _dir) = app("light");
    let _ = app.update(Message::Go(View::Settings));

    let answer: fllm_app::skills::ScheduleAnswer = serde_json::from_value(json!({
        "clock_running": false,
        "note": "the server is asleep",
        "tasks": [{
            "id": "t1", "name": "morning", "prompt": "do it", "when": "daily at 08:30",
            "mode": "agent", "folder": "/home/m/proj", "notify": false, "enabled": true,
            "deliver": "telegram:1234", "running": false, "next_run": 1700000000,
            "last": {"time": 1699990000, "ok": false, "delivery": "401"}
        }]
    }))
    .expect("an answer of the schedule");
    let _ = app.update(Message::ScLoaded(Ok(answer)));

    let mut ui = sim(&app);
    ui.find("The scheduler starts when a model is loaded. Tasks can be defined now.")
        .expect("the line of the page when its clock does not run");
    ui.find("morning").expect("the name of the task");
    ui.find("daily at 08:30").expect("the chip of its schedule");
    ui.find("agent").expect("the chip of its kind");
    ui.find("telegram").expect("the chip of where it also sends to");
    // the line under the task is drawn as one text, so it is found whole
    let runs = format!(
        "Next run: {}. Last run: {} (failed; delivery 401). Folder: /home/m/proj",
        fllm_app::skills::when(1700000000),
        fllm_app::skills::when(1699990000)
    );
    ui.find(runs.as_str()).expect("the whole line under the task");
    for button in ["Run now", "Results", "Edit", "Delete"] {
        ui.find(button).expect("a button of the row");
    }
    drop(ui);

    // the answers of a task are read when its button opens them
    let _ = app.update(Message::ScAction("results".into(), "t1".into()));
    let _ = app.update(Message::ScResultsLoaded(
        "t1".into(),
        Ok(serde_json::from_value(json!({
            "results": [{"time": 1699990000, "ok": true, "seconds": 1.5, "text": "the answer"}]
        }))
        .expect("an answer of the results")),
    ));
    let state = app.sc_results.get("t1").expect("the answers of the task were read");
    assert!(state.shown, "the box is open");
    let answers = match &state.filled {
        Some(Ok(list)) if list.len() == 1 => list.clone(),
        other => panic!("the answers could not be read: {other:?}"),
    };
    assert_eq!(
        fllm_app::skills::result_line(&answers[0]),
        format!("{} - ok (1.5s)", fllm_app::skills::when(1699990000)),
        "the line over the text"
    );
    // the box of the answers is open and holds the text of the answer; the line over it is the
    // one `result_line` writes, which the unit test of the module holds to the page
    let mut ui = sim(&app);
    ui.find("the answer").expect("what came out of it");
    drop(ui);

    // the preview of a schedule: the next runs, and the sentence of one that means nothing
    let _ = app.update(Message::ScPreviewed(Ok(fllm_app::skills::CheckAnswer {
        ok: true,
        next: vec![1700000000],
        error: None,
    })));
    let mut ui = sim(&app);
    ui.find(format!("Next runs: {}", fllm_app::skills::when(1700000000)).as_str())
        .expect("the preview of the box");
    drop(ui);
    let _ = app.update(Message::ScPreviewed(Ok(fllm_app::skills::CheckAnswer {
        ok: false,
        next: vec![],
        error: Some("I do not know what that means.".into()),
    })));
    assert_eq!(app.sc_preview, "I do not know what that means.");
    assert!(app.sc_preview_err, "the line is drawn in the colour of an error");

    // a task that is edited puts itself into the form and changes its title
    let _ = app.update(Message::ScAction("edit".into(), "t1".into()));
    assert_eq!(app.sc_name, "morning");
    assert_eq!(app.sc_mode, "agent");
    assert_eq!(app.sc_folder, "/home/m/proj");
    let mut ui = sim(&app);
    ui.find("Edit the task").expect("the title of the form while a task is edited");
    drop(ui);

    // and deleting it asks with the name of the task
    let _ = app.update(Message::ScAction("remove".into(), "t1".into()));
    let mut ui = sim(&app);
    ui.find("Delete the task \"morning\"?").expect("the question of `askDialog`");
    ui.find("Its saved results are deleted too.").expect("the sentence under it");
    drop(ui);
}

/// What the learning loop wants to change, its dialog, and the two cards of `skilllibrary.js`.
#[test]
fn the_revisions_and_the_library_work_the_way_the_page_works() {
    let (mut app, _dir) = app("light");
    let _ = app.update(Message::Go(View::Settings));

    let _ = app.update(Message::LnStatusLoaded(Ok(serde_json::from_value(json!({
        "skills": [{"name": "a"}], "memory": [], "profile": [], "running": false
    }))
    .expect("an answer of the learn overview"))));
    assert_eq!(app.ln_pending, "Waiting for a decision: 1 skill revision.");

    let _ = app.update(Message::LnLogLoaded(Ok(serde_json::from_value(json!({
        "log": [
            {"id": "c1", "kind": "skill_revise", "name": "my-skill", "t": 1699990000,
             "session": "s1", "reason": "a failure", "diff": "+ new line\n- old line", "undone": false},
            {"id": "c2", "kind": "memory_add", "text": "not about a skill", "t": 1699990001, "undone": false}
        ]
    }))
    .expect("an answer of the log"))));
    assert_eq!(app.ln_feed.len(), 1, "only the changes about a skill are in this card");

    let _ = app.update(Message::LnRevisionsLoaded(Ok(serde_json::from_value(json!({
        "skills": [{
            "name": "my-skill", "scope": "user", "workspace": "", "uses": 5, "failed": 1,
            "up": 2, "down": 0, "retries": 0, "pending": false, "due": null,
            "versions": [{"version": 3, "action": "revised", "t": 1699990000}],
            "revision": {"trigger": "a failure", "reason": "the second step was wrong",
                         "diff": "+ fixed"},
            "recent": [{"t": 1699990000, "outcome": "failed", "feedback": null,
                        "verify": false, "note": "timeouts"}]
        }]
    }))
    .expect("an answer of the revisions"))));

    let mut ui = sim(&app);
    ui.find("Revised the skill \"my-skill\"").expect("the title of the change");
    ui.find("Undo").expect("the button beside it");
    ui.find("+ new line").expect("the added line of the diff in green");
    ui.find("- old line").expect("the line that went in red");
    ui.find("my-skill").expect("the name of the row");
    ui.find("5 uses, 1 failed, thumbs 2 up / 0 down").expect("what the row counts");
    ui.find("Improve now").expect("the button of the row");
    ui.find("Roll back").expect("the button beside it");
    ui.find("Draft revision (a failure): the second step was wrong")
        .expect("what the model wants to change");
    ui.find("Approve").expect("the button that applies it");
    ui.find("Reject").expect("the button beside it");
    ui.find("The skill stays as it is until you approve.").expect("the sentence under them");
    ui.find("Recent uses").expect("the uses of the skill");
    ui.find(
        format!("{} · failed · checks failed · timeouts", fllm_app::skills::when(1699990000)).as_str(),
    )
    .expect("one of the uses, with its time in front");
    // the label of a version is what `versions_label` writes, which the unit test of the module
    // holds to the page — a `pick_list` is drawn by the renderer, so its label is not a text of
    // the tree the window can look at
    drop(ui);

    // Roll back asks first, with the words of the page
    let _ = app.update(Message::LnAction(
        "rollback".into(),
        "my-skill".into(),
        "user".into(),
        String::new(),
        Some(3),
    ));
    let mut ui = sim(&app);
    ui.find("Roll back \"my-skill\"?").expect("the question of `askDialog`");
    ui.find("The text of version 3 is put back. The current text is kept as a new version, so this can be undone.")
        .expect("the sentence under it");
    drop(ui);
    let _ = app.update(Message::LnConfirmOk);
    assert!(app.ln_confirm.is_none(), "the dialog is away again");

    // the two cards of the library
    let _ = app.update(Message::LibLoaded(Ok(serde_json::from_value(json!({
        "skills": [{"name": "git-workflow", "description": "How to branch", "when": "for repos",
                    "installed": false}]
    }))
    .expect("an answer of the library"))));
    let _ = app.update(Message::RepoLoaded(Ok(serde_json::from_value(json!({
        "repo": "anthropics/skills", "truncated": true, "skills": [
            {"name": "pdf", "description": "PDF tools", "path": "pdf/SKILL.md",
             "url": "https://x/pdf", "installed": false}
        ]
    }))
    .expect("an answer of the repository"))));

    let mut ui = sim(&app);
    ui.find("git-workflow").expect("a skill of the library");
    ui.find("when: for repos").expect("when it is for");
    ui.find("Install").expect("the button that writes it into the folder");
    ui.find("pdf").expect("a skill of the repository");
    ui.find("1 skill in anthropics/skills (the list is cut off). ")
        .expect("the line of the answer");
    drop(ui);

    // a directory that is not configured says so, and an installed row says *installed*
    let _ = app.update(Message::DirLoaded(Ok(serde_json::from_value(json!({
        "configured": false, "skills": []
    }))
    .expect("an answer of the directory"))));
    let mut ui = sim(&app);
    ui.find("Save a Vercel token first.").expect("the sentence of the page");
    drop(ui);
    let _ = app.update(Message::LibInstalled(
        "git-workflow".into(),
        Ok(serde_json::json!({})),
    ));
    assert_eq!(
        app.lib_status,
        "Installed git-workflow. It is in your skills folder now and can be edited there.",
        "the line of `skilllibrary.js` after an install"
    );
}
