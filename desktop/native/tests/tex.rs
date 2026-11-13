//! The math of an answer: the small LaTeX subset typeset the way the Studio's markdown.js typesets
//! it, and the places where an answer writes math inline — both checked against fixtures the
//! JavaScript itself produced.

use fllm_app::tex::*;

fn fixtures() -> serde_json::Value {
    let raw = include_str!("fixtures/tex_fixtures.json");
    serde_json::from_str(raw).expect("the fixtures of the Studio's code")
}

#[test]
fn the_typesetter_gives_the_html_of_the_studio() {
    let f = fixtures();
    let mut typesetted = 0;
    for case in f["tex"].as_array().unwrap() {
        let (src, inline, display) = (case[0].as_str().unwrap(), case[1].as_str().unwrap_or(""), case[2].as_str().unwrap_or(""));
        let got = typeset(src, false).map(|n| to_html(&n, false));
        match (inline.is_empty(), got) {
            (true, None) => {}
            (true, Some(html)) => panic!("{src:?}: the Studio shows the source, we typeset it: {html}"),
            (false, None) => panic!("{src:?}: the Studio typesets it, we show the source"),
            (false, Some(html)) => {
                assert_eq!(html, inline, "typeset({src:?}, inline)");
                typesetted += 1;
            }
        }
        let got = typeset(src, true).map(|n| to_html(&n, true));
        match (display.is_empty(), got) {
            (true, None) => {}
            (true, Some(html)) => panic!("{src:?} display: the Studio shows the source: {html}"),
            (false, None) => panic!("{src:?} display: the Studio typesets it"),
            (false, Some(html)) => {
                assert_eq!(html, display, "typeset({src:?}, display)");
                typesetted += 1;
            }
        }
    }
    assert!(typesetted >= 60, "most of the cases typeset: {typesetted}");
}

#[test]
fn math_span_is_the_chip_or_the_typeset_source_of_the_studio() {
    let f = fixtures();
    for case in f["span"].as_array().unwrap() {
        let (src, display, want) = (case[0].as_str().unwrap(), case[1].as_bool().unwrap(), case[2].as_str().unwrap());
        assert_eq!(span_html(src, display), want, "mathSpan({src:?}, {display})");
    }
}

#[test]
fn the_places_of_inline_math_are_those_of_the_studio() {
    let f = fixtures();
    for case in f["patterns"].as_array().unwrap() {
        let (text, want) = (case[0].as_str().unwrap(), &case[1]);
        let hits: Vec<serde_json::Value> = math_hits(text)
            .iter()
            .map(|h| serde_json::json!({ "start": h.start, "end": h.end, "display": h.display, "body": h.body }))
            .collect();
        assert_eq!(serde_json::Value::Array(hits), *want, "the math of {text:?}");
    }
}
