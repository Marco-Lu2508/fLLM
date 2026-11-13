//! The artifacts of an answer: what is a previewable block, how blocks of a conversation become
//! families with versions, the file names and the table of a CSV — checked against the fixtures
//! the Studio's own `@pure` block in `artifacts.js` produced.

use fllm_app::artifacts::*;
use serde_json::Value;

fn fixtures() -> Value {
    let raw = include_str!("fixtures/artifacts_fixtures.json");
    serde_json::from_str(raw).expect("the fixtures of the Studio's code")
}

#[test]
fn the_kinds_and_extensions_are_those_of_the_studio() {
    let f = fixtures();
    for (lang, want) in f["limits"]["kind_by_lang"].as_object().unwrap() {
        let got = kind_by_lang(lang).map(|k| serde_json::to_value(k).unwrap());
        assert_eq!(got.as_ref(), Some(want), "kind of {lang}");
    }
    for (lang, want) in f["limits"]["ext_by_lang"].as_object().unwrap() {
        assert_eq!(ext_by_lang(lang), want.as_str(), "ext of {lang}");
    }
    for (kind, ext) in f["limits"]["kind_ext"].as_object().unwrap() {
        let k: Kind = serde_json::from_value(Value::String(kind.clone())).unwrap();
        assert_eq!(k.ext(), ext, "ext of {kind}");
    }
    for (kind, mime) in f["limits"]["kind_mime"].as_object().unwrap() {
        let k: Kind = serde_json::from_value(Value::String(kind.clone())).unwrap();
        assert_eq!(k.mime(), mime, "mime of {kind}");
    }
    // the whole lookup of artifactKind, and the JavaScript languages that can run
    for case in f["kind"].as_array().unwrap() {
        let (lang, want) = (case[0].as_str().unwrap(), case[1].as_str().unwrap());
        let got = serde_json::to_value(artifact_kind(lang)).unwrap();
        assert_eq!(got, Value::String(want.to_string()), "artifactKind({lang})");
    }
    assert!(is_js_lang("js") && is_js_lang("JavaScript") && is_js_lang("mjs") && is_js_lang("node"));
    assert!(!is_js_lang("py") && !is_js_lang(""));
}

#[test]
fn a_title_comes_out_of_every_info_string_the_studio_reads() {
    let f = fixtures();
    for case in f["parse_info"].as_array().unwrap() {
        let (info, want) = (case[0].as_str().unwrap(), case[1].as_str().unwrap());
        assert_eq!(parse_info(info), want, "parseInfo({info:?})");
    }
}

#[test]
fn the_previewable_blocks_of_a_message_are_the_ones_of_the_studio() {
    let f = fixtures();
    for case in f["extract"].as_array().unwrap() {
        let (name, want) = (case[0].as_str().unwrap(), &case[1]);
        let text = extract_text_of(name);
        let got: Value = serde_json::to_value(extract_artifacts(&text)).unwrap();
        assert_eq!(got, *want, "{name}");
    }
}

/// The message texts of the extract cases, in the order of the fixture file.
fn extract_text_of(name: &str) -> String {
    match name {
        "a plain fence" => "```js\nconsole.log(1);\n```".to_string(),
        "an html page" => "```html\n<h1>Hi</h1>\n```".to_string(),
        "with a title on a code block" => "```python title=\"calc.py\"\nprint(1)\n```".to_string(),
        "tilde fence with info" => "~~~ svg title=\"logo\"\n<circle/>\n~~~".to_string(),
        "unclosed fence is skipped" => "```html\n<h1>never closed".to_string(),
        "closing fence needs only backticks" => "```md\n# Hi\n``` trailing junk not allowed\n```".to_string(),
        "markdown variant" => "```markdown\n# Doc\n```".to_string(),
        "tsv is a table" => "```tsv\ta\tb\n1\t2\n```".to_string(),
        "the info string may be long" => "```js {wrap} title=\"Run me\" other\nx\n```".to_string(),
        "cr lf line endings" => "```html\r\n<p>ok</p>\r\n```".to_string(),
        "body keeps blank lines" => "```csv\na,b\n\n1,2\n```".to_string(),
        "four backticks" => "````html\n<p>deep</p>\n````".to_string(),
        "kind code without a title is not an artifact" => "```rust\nfn main() {}\n```".to_string(),
        "indented fence" => "  ```html\n  <p>x</p>\n  ```".to_string(),
        other => panic!("unknown extract case: {other}"),
    }
}

#[test]
fn blocks_of_a_conversation_group_into_the_families_of_the_studio() {
    let f = fixtures();
    for case in f["families"].as_array().unwrap() {
        let (name, want) = (case[0].as_str().unwrap(), &case[1]);
        let msgs = family_messages(name);
        let got: Value = serde_json::to_value(build_families(&msgs)).unwrap();
        assert_eq!(got, *want, "{name}");
    }
}

/// The messages of the family cases, in the order of the fixture file.
fn family_messages(name: &str) -> Vec<(String, String)> {
    let list: &[(&str, &str)] = match name {
        "two messages with the same title" => &[("m1", "```html title=\"a\"\nv1\n```"), ("m2", "```html title=\"a\"\nv2\n```")],
        "an untitled block continues the newest of its kind" => &[
            ("m1", "```csv\na,b\n```"),
            ("m2", "```html\n<p>x</p>\n```"),
            ("m3", "```csv\na,b,c\n```"),
        ],
        "identical code stays with its own family" => &[("m1", "```md\nsame\n```"), ("m2", "```html\nx\n```"), ("m3", "```md\nsame\n```")],
        "one message changes a family at most once" => &[
            ("m1", "```html title=\"a\"\none\n```"),
            ("m2", "```html title=\"a\"\ntwo\n``` \n```html title=\"a\"\nthree\n```"),
        ],
        "titles are compared without case" => &[("m1", "```js title=\"Tool\"\nx\n```"), ("m2", "```js title=\"tool\"\ny\n```")],
        "several kinds start their own families" => &[("m1", "```html\n<p/>\n```\n```svg\n<circle/>\n```"), ("m2", "```html\n<p>2</p>\n```")],
        other => panic!("unknown family case: {other}"),
    };
    list.iter().map(|(id, text)| (id.to_string(), text.to_string())).collect()
}

#[test]
fn the_download_file_is_the_one_the_studio_names() {
    let f = fixtures();
    for case in f["file_name"].as_array().unwrap() {
        let (a, want) = (&case[0], case[1].as_str().unwrap());
        let kind: Kind = serde_json::from_value(a["kind"].clone()).unwrap();
        let got = file_name(a["title"].as_str().unwrap_or(""), kind, a["lang"].as_str().unwrap());
        assert_eq!(got, want, "{a}");
    }
}

#[test]
fn the_table_of_a_csv_is_the_one_the_studio_parses() {
    let f = fixtures();
    for (i, case) in f["parse_csv"].as_array().unwrap().iter().enumerate() {
        let text = csv_text_of(i);
        assert_eq!(serde_json::to_value(parse_csv(&text)).unwrap(), *case, "case {i}");
    }
    // the table of the panel: the header as the top row, and the note under the 500th row
    let rows = vec![
        vec!["name".to_string(), "age".to_string()],
        vec!["Ada".to_string(), "36".to_string()],
    ];
    let md = csv_markdown(&rows);
    assert!(md.starts_with("| name | age |\n| --- | --- |"), "the table: {md:?}");
    assert!(md.contains("| Ada | 36 |"), "the table: {md:?}");
    assert!(!md.contains("rows."), "a small table has no note");
    let empty = csv_markdown(&[]);
    assert_eq!(empty, "The table is empty.", "the empty table of the panel");
    let many: Vec<Vec<String>> = (0..600).map(|i| vec![format!("r{i}"), format!("{}", i + 1)]).collect();
    let md = csv_markdown(&many);
    assert!(md.contains("Showing the first 500 of 600 rows."), "the note of the Studio: {md:?}");
}

/// One run of JavaScript, the way the Studio's card runs it: the messages come over the channel
/// until the runner falls silent (`runBlock` with `runner.js` behind it).
fn run(code: &str) -> Vec<fllm_app::artifacts::RunEvent> {
    use futures::StreamExt;
    let (tx, rx) = fllm_app::artifacts::run_channel();
    let code = code.to_string();
    let handle = std::thread::spawn(move || fllm_app::artifacts::run_javascript(&code, tx));
    let out = futures::executor::block_on(async move {
        let mut out = Vec::new();
        let mut rx = rx;
        while let Some(ev) = rx.next().await {
            out.push(ev);
        }
        out
    });
    handle.join().expect("the runner thread ends");
    out
}

#[test]
fn the_runner_prints_its_console_and_reports_what_came_out() {
    use fllm_app::artifacts::{CodeRun, RunEvent, RunStatus};
    let events = run("console.log('hi', 2); 41 + 1");
    let logs: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            RunEvent::Log { level, text } => Some(format!("{level}:{text}")),
            _ => None,
        })
        .collect();
    assert_eq!(logs, ["log:hi 2"], "{events:?}");
    let last = events.last().expect("the run says it is done");
    let RunEvent::Done { value, error, .. } = last else { panic!("done: {events:?}") };
    assert_eq!(value.as_deref(), Some("42"), "the value of the last line: {events:?}");
    assert!(error.is_none(), "{events:?}");
    // the card under the block shows the same
    let mut run = CodeRun::starting();
    for ev in &events {
        run.apply(ev.clone());
    }
    let RunStatus::Finished { failed, line, .. } = &run.status else { panic!("the card is finished: {run:?}") };
    assert!(!*failed, "{run:?}");
    assert!(line.starts_with("Done in "), "the line of the Studio: {line:?}");
    assert!(run.lines.iter().any(|l| l.level == "value" && l.text == "-> 42"), "{run:?}");
}

#[test]
fn the_runner_reports_an_error_and_the_blocked_network() {
    use fllm_app::artifacts::RunEvent;
    let events = run("noSuchFunction()");
    let RunEvent::Done { error, .. } = events.last().expect("done") else { panic!("{events:?}") };
    let error = error.as_deref().expect("the error of the run");
    assert!(error.starts_with("ReferenceError:"), "name and message, like the Studio: {error:?}");
    assert!(error.contains("noSuchFunction"), "{error:?}");

    let events = run("fetch('https://example.com')");
    let RunEvent::Done { error, .. } = events.last().expect("done") else { panic!("{events:?}") };
    assert_eq!(error.as_deref(), Some("Error: fetch is disabled: code blocks run offline"), "{events:?}");
}

#[test]
fn the_runner_waits_for_an_async_answer_and_cuts_its_output() {
    use fllm_app::artifacts::RunEvent;
    // the async fallback of runner.js: the promise is awaited and its value reported
    let events = run("(async () => { console.log('later'); return 7; })()");
    assert!(events.iter().any(|e| matches!(e, RunEvent::Log { level, text, .. } if text == "later")), "{events:?}");
    let RunEvent::Done { value, error, .. } = events.last().expect("done") else { panic!("{events:?}") };
    assert!(error.is_none(), "{events:?}");
    assert_eq!(value.as_deref(), Some("7"), "the promise's value: {events:?}");

    // 500 lines and then silence, with the one warning of the Studio (limits.RUN_LINES)
    let events = run("for (let i = 0; i < 520; i++) console.log('line', i);");
    let logs: Vec<&RunEvent> = events.iter().filter(|e| matches!(e, RunEvent::Log { .. })).collect();
    assert_eq!(logs.len(), limits_run_lines() + 1, "500 lines and the warning: {}", logs.len());
    let RunEvent::Log { level, text } = logs.last().unwrap() else { unreachable!() };
    assert_eq!((*level, text.as_str()), ("warn", "... output truncated"), "the cut of the Studio");
    assert!(matches!(events.last(), Some(RunEvent::Done { .. })), "the run still ends: {events:?}");
}

fn limits_run_lines() -> usize {
    fllm_app::artifacts::limits::RUN_LINES
}

#[test]
fn a_syntax_error_reports_the_first_one_and_the_runner_stops_at_five_seconds() {
    use fllm_app::artifacts::{CodeRun, RunEvent, RunStatus};
    // runner.js reports the first SyntaxError when the async fallback fails to parse too
    let events = run("let = broken %%%;");
    let RunEvent::Done { error, .. } = events.last().expect("done") else { panic!("{events:?}") };
    let error = error.as_deref().expect("an error");
    assert!(error.starts_with("SyntaxError:"), "the first error: {error:?}");

    // the five seconds of RUN_MS stop a run that never ends
    let events = run("while (true) {}");
    let RunEvent::Failed(why) = events.last().expect("the stop") else { panic!("{events:?}") };
    assert_eq!(why.as_str(), format!("Stopped after {} s (time limit)", fllm_app::artifacts::limits::RUN_MS / 1000), "{events:?}");
    let mut card = CodeRun::starting();
    for ev in &events {
        card.apply(ev.clone());
    }
    assert!(card.warn, "the stop of the Studio is a warning: {card:?}");
    let RunStatus::Stopped(text) = &card.status else { panic!("{card:?}") };
    assert!(text.contains("time limit"), "{text:?}");
}

/// The texts of the CSV cases, in the order of the fixture file.
fn csv_text_of(i: usize) -> String {
    match i {
        0 => "a,b,c\n1,2,3".to_string(),
        1 => "name;age\n\"Smith; Jr\";42".to_string(),
        2 => "x\ty\n1\t2".to_string(),
        3 => "\"quoted, field\",b\n\"line\nbreak\",2".to_string(),
        4 => String::new(),
        5 => "single".to_string(),
        6 => "a,b\n\n1,2".to_string(),
        7 => "a,,b\n1,,3".to_string(),
        8 => "empty,rows,too\n,,\nreal,data,here".to_string(),
        other => panic!("unknown csv case: {other}"),
    }
}
