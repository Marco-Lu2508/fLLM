//! The start plan of the native app against the Studio's own: twelve cases from the pure heart of
//! `panels.js` (loaded by `panels_fixtures.js`), each compared field by field.

use fllm_app::hero::{model_file_title, start_plan, StartPlan};
use fllm_app::models::parse_overview;
use serde_json::Value;

fn fixtures() -> Vec<Value> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/panels_fixtures.json");
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("run `node panels_fixtures.js` first: {e}"));
    serde_json::from_str(&raw).expect("the fixtures are a list")
}

/// What the Studio answers for the plan, in the shape of the port: the primary model with its two
/// flags, the others with what they are, and the id of the download.
fn shape(plan: &StartPlan) -> Value {
    let entry = |e: &fllm_app::hero::PlanEntry| {
        serde_json::json!({
            "ref": e.reference,
            "title": e.title,
            "size": e.size,
            "where": e.where_,
        })
    };
    serde_json::json!({
        "mode": plan.mode,
        "primary": plan.primary.as_ref().map(|p| {
            let mut v = entry(p);
            v["last"] = serde_json::json!(p.last);
            v["fits"] = serde_json::json!(p.fits);
            v
        }),
        "others": plan.others.iter().map(entry).collect::<Vec<_>>(),
        "download": plan.download.as_ref().map(|d| serde_json::json!({ "id": d.id })),
    })
}

#[test]
fn every_case_of_the_studio_gives_the_same_answer() {
    let cases = fixtures();
    assert!(cases.len() >= 12, "the generator found the pure zone: {} cases", cases.len());
    for case in &cases {
        let function = case["fn"].as_str().unwrap();
        let args = case["args"].as_array().unwrap();
        let note = case["note"].as_str().unwrap_or("");
        if function == "modelFileTitle" {
            let text = match &args[0] {
                Value::String(s) => s.as_str(),
                Value::Null => "",
                other => panic!("a title case with {other:?}"),
            };
            assert_eq!(
                model_file_title(text),
                case["got"].as_str().unwrap(),
                "modelFileTitle({text:?}) {note}"
            );
        } else {
            let hub = &args[0];
            let last = match &args[1] {
                Value::String(s) => s.as_str(),
                Value::Null => "",
                other => panic!("a last model with {other:?}"),
            };
            let overview = parse_overview(hub);
            let plan = start_plan(&overview, last);
            let got = case["got"].clone();
            let mine = shape(&plan);
            assert_eq!(
                mine, got,
                "startPlan({last:?}) {note}\n  Studio: {got:#}\n  native: {mine:#}"
            );
        }
    }
}

/// The fields the Studio's answer does not carry (the flags of the others) are harmless: they are
/// what the view needs to show the little words beside the models.
#[test]
fn the_others_of_the_plan_carry_their_flags_as_well() {
    let overview = parse_overview(&serde_json::json!({
        "local": [
            {"name": "big.gguf", "path": "/m/big.gguf", "size_bytes": 4_000_000_000u64},
            {"name": "small.gguf", "path": "/m/small.gguf", "size_bytes": 1_000_000_000u64},
        ],
        "available_memory_bytes": 3_000_000_000u64,
    }));
    let plan = start_plan(&overview, "");
    // the small one fits (rank 2), the big one does not (rank 3): the small one comes first
    let primary = plan.primary.as_ref().expect("there is a model");
    assert_eq!(primary.title, "small");
    assert!(primary.fits, "1.15 * 1 GB is under 3 GB");
    assert_eq!(plan.others.len(), 1, "the big one is beside it");
    assert!(!plan.others[0].fits, "1.15 * 4 GB is over 3 GB");
    assert!(!plan.others[0].last, "nothing was used last");
}
