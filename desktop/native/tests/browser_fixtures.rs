//! Row 85: the browser panel against the page's own `browser.js`. Every scenario of
//! `tests/fixtures/browser_fixtures.js` runs the page in a vm (a stub world, a table of answers),
//! and what each step leaves behind — the requests it sent, the words and flags of the panel, the
//! comments text — is compared with what the port does for the same step.

use fllm_app::browser::*;
use serde_json::Value;

fn fixtures() -> Value {
    let raw = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/browser_fixtures.json")).expect("the fixtures of the page");
    serde_json::from_str(&raw).expect("fixtures as json")
}

/// The words of a piece of the page's HTML: the tags gone, the four entities of `escapeHtml` back.
fn words(html: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for c in html.chars() {
        match c {
            '<' => in_tag = true,
            '>' if in_tag => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&amp;", "&")
}

fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => (x.as_f64().unwrap() - y.as_f64().unwrap()).abs() < 1e-9,
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same(p, q)),
        (Value::Object(x), Value::Object(y)) => x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| same(v, w))),
        _ => a == b,
    }
}

fn os_of(platform: &str) -> &'static str {
    if platform.starts_with("Win") {
        "windows"
    } else if platform.starts_with("Mac") {
        "macos"
    } else {
        "linux"
    }
}

/// The calls of a step that went to one path (`""` is the status).
fn calls_to<'a>(step: &'a Value, path: &str, method: &str) -> Vec<&'a Value> {
    step["calls"].as_array().unwrap().iter().filter(|c| c["path"].as_str().unwrap().strip_prefix("/api/studio/agent/browser").unwrap() == path && c["method"] == method).collect()
}

fn answer_of(call: &Value) -> Option<Value> {
    call["ok"].as_bool().unwrap_or(true).then(|| call["response"].clone())
}

/// What the window does with an input: sends it and reads the answer from the recorded call.
fn send(panel: &mut Panel, input: &Input, step: &Value, toasts: &mut Vec<String>, name: &str) {
    let sent = calls_to(step, "/input", "POST");
    let call = sent.iter().find(|c| c["body"]["type"] == input.body()["type"]).unwrap_or_else(|| panic!("{name}: the page sent no {:?}: {sent:?}", input.body()));
    assert!(same(&call["body"], &input.body()), "{name}: the body of the request: {} / {}", call["body"], input.body());
    let answer = if call["response"].is_null() && call["ok"].as_bool() == Some(true) { Err("x".to_string()) } else { Ok(call["response"].clone()) };
    // a call that threw is recorded with no answer; the page toasts its message
    let step_toasts: Vec<String> = step["toasts"].as_array().unwrap().iter().filter_map(Value::as_str).map(str::to_string).collect();
    let answer = match (&answer, step_toasts.iter().find(|t| t.starts_with("The browser did not answer: "))) {
        (_, Some(t)) => Err(t.trim_start_matches("The browser did not answer: ").to_string()),
        _ => answer,
    };
    toasts.extend(panel.input_answered(input, answer));
}

fn state_matches(panel: &Panel, step: &Value, name: &str) {
    let want = &step["state"];
    let view = panel.view();
    assert_eq!(view.state, want["state"].as_str().unwrap(), "{name}: the state word");
    let empty_html = want["empty"]["html"].as_str().unwrap();
    let mut shown = view.empty.text.clone();
    if let Some((label, _)) = view.empty.button {
        shown.push_str(label);
    }
    assert_eq!(shown, words(empty_html), "{name}: what the stage says");
    assert_eq!(view.empty.button.map(|(_, primary)| primary), empty_html.contains("<button").then(|| empty_html.contains("btn-primary")), "{name}: the button of the stage");
    assert_eq!(!view.show_image, want["img"]["hidden"].as_bool().unwrap(), "{name}: the picture is shown");
    assert_eq!(panel.image.as_deref(), want["img"]["src"].as_str(), "{name}: the picture");
    assert_eq!(!view.live_dot, want["liveDot"].as_bool().unwrap(), "{name}: the live dot");
    for id in ["brBack", "brForward", "brReload", "brStop", "brComment"] {
        assert_eq!(!view.back_enabled, want["disabled"][id].as_bool().unwrap(), "{name}: {id} disabled");
    }
    assert_eq!(!view.url_enabled, want["urlDisabled"].as_bool().unwrap(), "{name}: the address box");
    assert_eq!(view.url_input, want["url"].as_str().unwrap(), "{name}: the address");
    assert_eq!(view.foot, want["foot"].as_str().unwrap(), "{name}: the line under the picture");
    assert_eq!(view.commenting, want["commenting"].as_bool().unwrap(), "{name}: comment mode");
    assert_eq!(panel.comments_text(), words(want["comments"].as_str().unwrap()), "{name}: the comments");
    assert_eq!(panel.note_text(), words(want["note"]["html"].as_str().unwrap()), "{name}: the note");
    if panel.draft.is_some() {
        let pin = panel.pin(640.0).map(|p| format!("{}px {}px {}px {}px", fmt(p[0]), fmt(p[1]), fmt(p[2]), fmt(p[3])));
        let s = &want["pin"]["style"];
        let page = s["left"].as_str().map(|_| format!("{} {} {} {}", s["left"].as_str().unwrap(), s["top"].as_str().unwrap(), s["width"].as_str().unwrap(), s["height"].as_str().unwrap()));
        assert_eq!(pin, page, "{name}: the frame around the element");
    }
}

fn fmt(n: f64) -> String {
    fllm_app::loopguard::js_number(n)
}

#[test]
fn the_panel_does_what_the_page_does() {
    let data = fixtures();
    let scenarios = data["scenarios"].as_array().unwrap();
    assert!(scenarios.len() >= 24);
    let mut steps_seen = 0;
    for sc in scenarios {
        let sname = sc["name"].as_str().unwrap();
        let rect = &sc["opts"]["stageRect"];
        let rect = Rect { left: rect["left"].as_f64().unwrap(), top: rect["top"].as_f64().unwrap(), width: rect["width"].as_f64().unwrap(), height: rect["height"].as_f64().unwrap() };
        let mut panel = Panel::new(os_of(sc["opts"]["platform"].as_str().unwrap()));
        for (n, step) in sc["steps"].as_array().unwrap().iter().enumerate() {
            let name = format!("{sname} / {}", step["step"].as_str().unwrap());
            steps_seen += 1;
            let mut toasts: Vec<String> = Vec::new();
            let status_call = |step: &Value| calls_to(step, "", "GET").first().map(|c| answer_of(c));
            if n == 0 {
                // the page reads the status when it starts
                if let Some(a) = status_call(step) {
                    panel.apply_status(a.as_ref());
                }
                state_matches(&panel, step, &name);
                continue;
            }
            let action = &step["action"];
            let f = |k: &str| action[k].as_f64().unwrap();
            match action["kind"].as_str().unwrap() {
                "open" => {
                    panel.open = true;
                    let a = status_call(step).expect("the status is read when the panel opens");
                    panel.apply_status(a.as_ref());
                }
                "hide" => panel.open = false,
                "tick" => {
                    let reads = panel.tick_reads_status();
                    let page_reads = status_call(step).is_some();
                    assert_eq!(reads, page_reads, "{name}: the status is read");
                    if reads {
                        panel.apply_status(status_call(step).unwrap().as_ref());
                    }
                    let frames = calls_to(step, "/frame", "POST");
                    assert_eq!(panel.tick_asks_frame(), !frames.is_empty(), "{name}: a frame is asked for");
                    if let Some(frame) = frames.first() {
                        assert_eq!(frame["body"]["since"].as_str().unwrap(), panel.key, "{name}: since");
                        if frame["ok"].as_bool().unwrap_or(true) {
                            panel.apply_frame(&frame["response"]);
                        }
                    }
                }
                "click" => {
                    let input = panel.click((f("x"), f("y")), rect, action["onButton"].as_bool().unwrap());
                    match input {
                        Some(i) => send(&mut panel, &i, step, &mut toasts, &name),
                        None => assert!(calls_to(step, "/input", "POST").is_empty(), "{name}: the page sent something the port did not"),
                    }
                }
                "dblclick" => match panel.dblclick((f("x"), f("y")), rect) {
                    Some(i) => send(&mut panel, &i, step, &mut toasts, &name),
                    None => assert!(calls_to(step, "/input", "POST").is_empty(), "{name}: the page sent something the port did not"),
                },
                "wheel" => {
                    for e in action["events"].as_array().unwrap() {
                        let unit = match e["mode"].as_i64().unwrap() {
                            1 => WheelUnit::Lines,
                            2 => WheelUnit::Pages,
                            _ => WheelUnit::Pixels,
                        };
                        panel.wheel((e["x"].as_f64().unwrap(), e["y"].as_f64().unwrap()), rect, (e["dx"].as_f64().unwrap(), e["dy"].as_f64().unwrap()), unit);
                    }
                    match panel.flush_wheel() {
                        Some(i) => send(&mut panel, &i, step, &mut toasts, &name),
                        None => assert!(calls_to(step, "/input", "POST").is_empty(), "{name}: the page sent a wheel the port did not"),
                    }
                }
                "key" => {
                    let flag = |k: &str| action[k].as_bool().unwrap_or(false);
                    let press = KeyPress { key: action["key"].as_str().unwrap().to_string(), ctrl: flag("ctrl"), alt: flag("alt"), shift: flag("shift"), meta: flag("meta"), composing: flag("composing") };
                    match panel.key(&press) {
                        Some(i) => send(&mut panel, &i, step, &mut toasts, &name),
                        None => assert!(calls_to(step, "/input", "POST").is_empty(), "{name}: the page sent a key the port did not"),
                    }
                }
                "paste" => match panel.paste(action["text"].as_str().unwrap()) {
                    Some(i) => send(&mut panel, &i, step, &mut toasts, &name),
                    None => assert!(calls_to(step, "/input", "POST").is_empty(), "{name}"),
                },
                "brBack" | "brForward" | "brReload" => {
                    let i = match action["kind"].as_str().unwrap() {
                        "brBack" => Input::Back,
                        "brForward" => Input::Forward,
                        _ => Input::Reload,
                    };
                    send(&mut panel, &i, step, &mut toasts, &name);
                }
                "submit" => {
                    panel.url_input = action["url"].as_str().unwrap().to_string();
                    match panel.submit() {
                        Some(i) => {
                            send(&mut panel, &i, step, &mut toasts, &name);
                            // `loadStatus()` follows
                            let a = status_call(step).expect("the status is read after going");
                            panel.apply_status(a.as_ref());
                        }
                        None => assert!(step["calls"].as_array().unwrap().is_empty(), "{name}"),
                    }
                }
                "close" => {
                    if action["confirm"].as_bool().unwrap() {
                        send(&mut panel, &Input::Close, step, &mut toasts, &name);
                        panel.closed();
                        let a = status_call(step).expect("the status is read after closing");
                        panel.apply_status(a.as_ref());
                    } else {
                        assert!(step["calls"].as_array().unwrap().is_empty(), "{name}: nothing is sent when you say no");
                    }
                }
                "comment" => panel.toggle_commenting(),
                "note_add" => {
                    panel.add_comment(action["note"].as_str().unwrap());
                }
                "note_cancel" => panel.cancel_note(),
                "del" => panel.delete_comment(action["i"].as_u64().unwrap() as usize),
                "take" => {
                    let text = panel.take_comments();
                    assert_eq!(text, step["extra"].as_str().unwrap_or_default(), "{name}: the comments as text");
                }
                "send" => {
                    assert_eq!(panel.send_text(""), step["extra"]["box"].as_str().unwrap(), "{name}");
                    assert_eq!(step["extra"]["clicked"], true);
                }
                "url" => assert_eq!(panel.current_url(), step["extra"].as_str().unwrap_or_default(), "{name}: the address for the rules"),
                other => panic!("{other}"),
            }
            let page_toasts: Vec<String> = step["toasts"].as_array().unwrap().iter().filter_map(Value::as_str).map(str::to_string).collect();
            assert_eq!(toasts, page_toasts, "{name}: the toasts");
            state_matches(&panel, step, &name);
        }
    }
    assert!(steps_seen > 150, "{steps_seen}");
}

// ---- Settings > Browser ----

use fllm_app::browsercards::*;

fn load_of(call: &Value) -> LoadResult {
    if let Some(msg) = call["response"].as_str().filter(|_| false) {
        return LoadResult::Failed(msg.to_string());
    }
    match call["ok"].as_bool() {
        Some(true) | None => LoadResult::Ok(call["response"].clone()),
        Some(false) => LoadResult::Refused(call["response"].clone()),
    }
}

fn saved_of(call: &Value) -> SaveResult {
    match call["ok"].as_bool() {
        Some(true) | None => SaveResult::Ok(call["response"].clone()),
        Some(false) => SaveResult::Refused(call["response"].clone()),
    }
}

#[test]
fn the_texts_of_the_cards_are_the_pages() {
    let data = fixtures();
    let html = data["settings"][0]["paneHtml"].as_str().unwrap();
    // the page's own words, in its order
    let mut want = String::new();
    for part in [
        ENABLE_TITLE, ENABLE_TEXT, ENABLE_LABEL, RUN_TITLE, SHOW_LABEL, SHOW_TEXT, KEEP_LABEL, KEEP_TEXT, LOCAL_LABEL,
        "Only localhost, while a workspace is open (default)", "None", "Localhost and the local network (10.x, 192.168.x ...)",
        LOCAL_TEXT, EXE_LABEL, EXE_TEXT, "Save", SAFETY_TITLE, SAFETY_TEXT,
    ] {
        want.push_str(part);
    }
    // the page's labels start with a space after the box
    assert_eq!(words(html).replace(" Let the", "Let the").replace(" Show the", "Show the").replace(" Keep logins", "Keep logins"), want);
    assert!(html.contains(&format!("placeholder=\"{EXE_PLACEHOLDER}\"")));
    for (value, label) in [("auto", LOCAL_CHOICES[0].1), ("off", LOCAL_CHOICES[1].1), ("all", LOCAL_CHOICES[2].1)] {
        assert!(html.contains(&format!("<option value=\"{value}\">{label}</option>")), "{value}");
    }
}

#[test]
fn the_cards_follow_the_page_step_by_step() {
    let data = fixtures();
    assert!(data["settings"].as_array().unwrap().len() >= 10);
    for sc in data["settings"].as_array().unwrap() {
        let sname = sc["name"].as_str().unwrap();
        let mut pane = BrowserPane::default();
        for (n, step) in sc["steps"].as_array().unwrap().iter().enumerate() {
            let name = format!("{sname} / {}", step["step"].as_str().unwrap());
            let calls = step["calls"].as_array().unwrap();
            let action = &step["action"];
            let to = |suffix: &str, c: &&Value| c["path"].as_str().unwrap().ends_with(suffix);
            match action["kind"].as_str().unwrap() {
                "load" => {
                    let call = calls.iter().find(|c| to("browser", c) && c["method"] == "GET").expect("the status is read");
                    // a request that threw is recorded without a status: the page says it could not read
                    let result = if sname == "the server is not there" { LoadResult::Failed("connection refused".into()) } else { load_of(call) };
                    pane.loaded(result);
                }
                "enable" => {
                    let checked = action["checked"].as_bool().unwrap();
                    pane.enabled = checked;
                    assert_eq!(step["confirms"].as_array().unwrap().len(), usize::from(checked), "{name}: asked once when switching on");
                    if checked {
                        assert_eq!(step["confirms"][0].as_str().unwrap(), CONFIRM_ENABLE);
                        if action["confirm"].as_bool().unwrap() {
                            let call = calls.iter().find(|c| to("/settings", c)).expect("it is saved");
                            assert_eq!(call["body"], serde_json::json!({ "enabled": true }), "{name}");
                            pane.saved(saved_of(call));
                        } else {
                            assert!(calls.is_empty(), "{name}: nothing is sent when you say no");
                            pane.enabled = false;
                        }
                    } else {
                        let call = calls.iter().find(|c| to("/settings", c)).expect("it is saved");
                        assert_eq!(call["body"], serde_json::json!({ "enabled": false }), "{name}");
                        pane.saved(saved_of(call));
                    }
                }
                "save" => {
                    pane.show = action["show"].as_bool().unwrap();
                    pane.keep = action["keep"].as_bool().unwrap();
                    pane.local = action["local"].as_str().unwrap().to_string();
                    pane.exe = action["exe"].as_str().unwrap().to_string();
                    let call = calls.iter().find(|c| to("/settings", c)).expect("it is saved");
                    assert_eq!(pane.save_body(), call["body"], "{name}: the request");
                    let result = if sname == "saving does not reach the server" { SaveResult::Failed("offline".into()) } else { saved_of(call) };
                    pane.saved(result);
                }
                other => panic!("{other}"),
            }
            let want = &step["state"];
            assert_eq!((pane.enabled, pane.show, pane.keep), (want["enabled"].as_bool().unwrap(), want["show"].as_bool().unwrap(), want["keep"].as_bool().unwrap()), "{name}: the boxes");
            assert_eq!((pane.local.as_str(), pane.exe.as_str()), (want["local"].as_str().unwrap(), want["exe"].as_str().unwrap()), "{name}: the choice and the program");
            assert_eq!(pane.found, want["found"].as_str().unwrap(), "{name}: the line of what was found");
            assert_eq!(pane.found_bad, want["foundColor"].as_str().is_some_and(|c| !c.is_empty()), "{name}: drawn as a problem");
            assert_eq!(pane.profile, want["profile"].as_str().unwrap(), "{name}: the profile line");
            assert_eq!(pane.message, want["msg"].as_str().unwrap(), "{name}: the message");
            let _ = n;
        }
    }
}
