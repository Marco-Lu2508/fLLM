//! Row 86: the pure block of the page's `computer.js` against the port. The fixtures are the
//! page's own functions run in a vm (`tests/fixtures/computer_fixtures.js`): the lines that name
//! an action, the questions, the marks and their SVG, the pictures kept in a conversation, the
//! panic keys.

use fllm_app::computer::*;
use serde_json::Value;

fn fixtures() -> Value {
    let raw = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/computer_fixtures.json")).expect("the fixtures of the page");
    serde_json::from_str(&raw).expect("fixtures as json")
}

/// A number of the page is a number: `100` and `100.0` are the same.
fn norm(v: &Value) -> Value {
    match v {
        Value::Number(n) => Value::String(format!("{}", n.as_f64().unwrap_or(f64::NAN))),
        Value::Array(a) => Value::Array(a.iter().map(norm).collect()),
        Value::Object(m) => Value::Object(m.iter().map(|(k, v)| (k.clone(), norm(v))).collect()),
        other => other.clone(),
    }
}

fn args_of(v: &Value) -> Value {
    v.clone()
}

#[test]
fn the_lines_and_questions_of_the_actions_are_the_pages() {
    let data = fixtures();
    assert!(data["subjects"].as_array().unwrap().len() >= 300);
    for row in data["subjects"].as_array().unwrap() {
        let (name, args) = (row["name"].as_str().unwrap(), args_of(&row["args"]));
        assert_eq!(computer_subject(name, &args), row["result"].as_str().unwrap(), "computerSubject({name}, {args})");
    }
    for row in data["questions"].as_array().unwrap() {
        let (name, args) = (row["name"].as_str().unwrap(), args_of(&row["args"]));
        assert_eq!(computer_question(name, &args), row["result"].as_str().unwrap(), "computerQuestionText({name}, {args})");
    }
}

#[test]
fn the_marks_and_their_picture_are_the_pages() {
    let data = fixtures();
    for row in data["marks"].as_array().unwrap() {
        let (name, args) = (row["name"].as_str().unwrap(), args_of(&row["args"]));
        let got = Value::Array(computer_marks(name, &args).iter().map(Mark::to_json).collect());
        assert_eq!(norm(&got), norm(&row["result"]), "computerMarks({name}, {args})");
    }
    for row in data["svgs"].as_array().unwrap() {
        let marks = computer_marks(row["name"].as_str().unwrap(), &row["args"]);
        let svg = marks_svg(row["w"].as_f64().unwrap(), row["h"].as_f64().unwrap(), &marks);
        assert_eq!(svg, row["result"].as_str().unwrap_or_default(), "marksSvg of {row}");
    }
}

fn msgs(v: &Value) -> Vec<Msg> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|m| Msg {
            role: m["role"].as_str().unwrap().to_string(),
            content: m["content"].as_str().unwrap().to_string(),
            images: m.get("images").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default(),
        })
        .collect()
}

/// What the page leaves of a message: without an `images` key at all when the pictures are gone.
fn page_form(list: &[Msg]) -> Value {
    Value::Array(
        list.iter()
            .map(|m| {
                let mut o = serde_json::json!({ "role": m.role, "content": m.content });
                if !m.images.is_empty() {
                    o["images"] = serde_json::json!(m.images);
                }
                o
            })
            .collect(),
    )
}

#[test]
fn the_pictures_of_a_conversation_are_kept_and_dropped_as_the_page_does() {
    let data = fixtures();
    for set in data["images"].as_array().unwrap() {
        let name = set["name"].as_str().unwrap();
        let original = msgs(&set["messages"]);
        // the page's empty `images: []` is no carrier and stays on the message; the port has no
        // difference between an empty list and none, so what is compared is the visible text
        let mut kept = original.clone();
        assert_eq!(drop_old_images(&mut kept, 2) as u64, set["dropped"].as_u64().unwrap(), "{name}: how many were dropped");
        assert_eq!(page_form(&kept), norm_images(&set["afterKeep2"]), "{name}: keep two");
        let mut kept1 = original.clone();
        assert_eq!(drop_old_images(&mut kept1, 1) as u64, set["dropped1"].as_u64().unwrap(), "{name}: dropped when one stays");
        assert_eq!(page_form(&kept1), norm_images(&set["afterKeep1"]), "{name}: keep one");
        assert_eq!(page_form(&without_images(&original)), norm_images(&set["without"]), "{name}: without pictures");
        assert_eq!(Value::Array(to_wire(&original)), set["wire"].clone().as_array().map(|a| Value::Array(a.iter().map(wire_form).collect())).unwrap(), "{name}: the wire form");
    }
}

/// The page keeps `images: []` on a message that had it; the port holds none either way.
fn norm_images(v: &Value) -> Value {
    Value::Array(
        v.as_array()
            .unwrap()
            .iter()
            .map(|m| {
                let mut m = m.clone();
                if m.get("images").and_then(Value::as_array).is_some_and(Vec::is_empty) {
                    m.as_object_mut().unwrap().remove("images");
                }
                m
            })
            .collect(),
    )
}

/// A message of the page's wire form without the `images` it kept when it had an empty list.
fn wire_form(m: &Value) -> Value {
    let mut m = m.clone();
    if m.get("images").and_then(Value::as_array).is_some_and(Vec::is_empty) {
        m.as_object_mut().unwrap().remove("images");
    }
    m
}

#[test]
fn the_panic_keys_are_the_pages() {
    let data = fixtures();
    for row in data["streaks"].as_array().unwrap() {
        let times: Vec<f64> = row["times"].as_array().unwrap().iter().filter_map(Value::as_f64).collect();
        let (list, fire) = escape_streak_default(&times, row["now"].as_f64().unwrap());
        let want: Vec<f64> = row["result"]["times"].as_array().unwrap().iter().filter_map(Value::as_f64).collect();
        assert_eq!((list, fire), (want, row["result"]["fire"].as_bool().unwrap()), "escapeStreak of {row}");
    }
    for row in data["streaks2"].as_array().unwrap() {
        let times: Vec<f64> = row["times"].as_array().unwrap().iter().filter_map(Value::as_f64).collect();
        let (list, fire) = escape_streak(&times, row["now"].as_f64().unwrap(), 100.0, 2);
        let want: Vec<f64> = row["result"]["times"].as_array().unwrap().iter().filter_map(Value::as_f64).collect();
        assert_eq!((list, fire), (want, row["result"]["fire"].as_bool().unwrap()), "escapeStreak(100, 2) of {row}");
    }
    for row in data["panic"].as_array().unwrap() {
        let e = &row["e"];
        let flag = |k: &str| e.get(k).and_then(Value::as_bool).unwrap_or(false);
        let got = is_panic_chord(e.get("key").and_then(Value::as_str).unwrap_or(""), flag("ctrlKey"), flag("altKey"), flag("shiftKey"));
        assert_eq!(got, row["result"].as_bool().unwrap(), "isPanicChord of {e}");
    }
}
