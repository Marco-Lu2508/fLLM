//! Rows 83 and 84: the words and shapes of the tool cards against the page's own `toolLabel`,
//! `diffRows`, `outputHtml` and `defaultOpen`.

use fllm_app::toolcard::*;
use serde_json::Value;

fn fixtures() -> Value {
    let raw = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/toolcard_fixtures.json")).expect("the fixtures of the page");
    serde_json::from_str(&raw).expect("fixtures as json")
}

/// A label's word as the page's template shows it: `String(x)` of JavaScript (a number is its
/// digits, an array its items joined by commas, an object `[object Object]`).
fn text_of(v: &Value) -> String {
    fllm_app::computer::js_string(v)
}

#[test]
fn every_tool_is_named_as_the_page_names_it() {
    let data = fixtures();
    assert!(data["labels"].as_array().unwrap().len() > 1000);
    for row in data["labels"].as_array().unwrap() {
        let name = row["name"].as_str().unwrap();
        let args = row.get("args").cloned().unwrap_or(Value::Null);
        let (verb, target) = label(name, &args, None);
        assert_eq!([verb.as_str(), target.as_str()], [text_of(&row["label"][0]).as_str(), text_of(&row["label"][1]).as_str()], "toolLabel({name}, {args})");
    }
}

#[test]
fn a_tool_of_an_mcp_server_is_named_by_its_tool_and_server() {
    let data = fixtures();
    for row in data["mcp"].as_array().unwrap() {
        let (verb, target) = label(row["name"].as_str().unwrap(), &row["args"], Some(("srv", "tool")));
        assert_eq!([verb.as_str(), target.as_str()], [text_of(&row["label"][0]).as_str(), text_of(&row["label"][1]).as_str()]);
    }
}

#[test]
fn an_edit_shows_what_it_removes_and_adds_as_the_page_does() {
    let data = fixtures();
    for row in data["diffs"].as_array().unwrap() {
        let args = row.get("args").cloned().unwrap_or(Value::Null);
        let got: Vec<(String, String)> = diff_rows(row["name"].as_str().unwrap(), &args)
            .into_iter()
            .map(|(k, l)| (match k { Kind::Add => "add", Kind::Del => "del", Kind::More => "more", Kind::Ctx => "ctx" }.to_string(), l))
            .collect();
        let want: Vec<(String, String)> = row["rows"].as_array().unwrap().iter().map(|r| (r[0].as_str().unwrap().to_string(), r[1].as_str().unwrap().to_string())).collect();
        assert_eq!(got, want, "diffRows({}, {args})", row["name"]);
    }
}

#[test]
fn an_output_is_cut_and_shown_in_full_as_the_page_does() {
    let data = fixtures();
    for row in data["outputs"].as_array().unwrap() {
        let text = match &row["text"] {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        let (shown, more) = output_view(&text, row["expanded"].as_bool().unwrap());
        assert_eq!(shown, row["shown"].as_str().unwrap(), "the text of {:?}", &text[..text.len().min(30)]);
        let want_more = row["more"].as_str().unwrap();
        let want_label = want_more.strip_prefix("<button class=\"btn btn-xs btn-ghost tool-more\" data-expand=\"i1\">").and_then(|s| s.strip_suffix("</button>"));
        assert_eq!(more.as_deref(), want_label, "the button of {:?}", &text[..text.len().min(30)]);
    }
}

#[test]
fn a_card_starts_open_when_the_page_opens_it() {
    let data = fixtures();
    for row in data["opens"].as_array().unwrap() {
        let item = &row["item"];
        let status = match item["status"].as_str().unwrap() {
            "pending" => Status::Pending,
            "running" => Status::Running,
            "ok" => Status::Ok,
            "error" => Status::Error,
            _ => Status::Denied,
        };
        let got = default_open(item["name"].as_str().unwrap(), status, item.get("output").is_some(), item.get("image").is_some(), item.get("shot").is_some(), item.get("check").is_some());
        assert_eq!(got, row["open"].as_bool().unwrap(), "defaultOpen({item})");
    }
}
