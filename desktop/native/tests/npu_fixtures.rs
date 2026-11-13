//! The pure NPU and device functions against the Studio's own: fifty-six cases from the `@pure`
//! zone of `npu.js` (loaded by `npu_fixtures.js`), each compared field by field. The cards that
//! show them are `hardware_window.rs`.

use fllm_app::npu::{
    device_backend_label, device_choices, device_key, device_memory, device_rank, device_role,
    merge_devices, npu_card_model, npu_megabytes, npu_progress_model, running_device,
};
use fllm_app::tour::Device;
use serde_json::{json, Value};

fn fixtures() -> Vec<Value> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/npu_fixtures.json");
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("run `node npu_fixtures.js` first: {e}"));
    serde_json::from_str(&raw).expect("the fixtures are a list")
}

/// One device of the fixtures' `{id, name, backend, api, total_memory_bytes, available}` shape.
fn to_device(v: &Value) -> Device {
    let text = |key: &str| v.get(key).and_then(Value::as_str).unwrap_or("").to_string();
    Device {
        id: text("id"),
        name: text("name"),
        backend: text("backend"),
        api: text("api"),
        dev_type: text("dev_type"),
        total_memory_bytes: v.get("total_memory_bytes").and_then(Value::as_u64).unwrap_or(0),
        free_memory_bytes: v.get("free_memory_bytes").and_then(Value::as_u64).unwrap_or(0),
        available: v.get("available").and_then(Value::as_bool).unwrap_or(true),
        has_id: v.get("has_id").and_then(Value::as_bool).unwrap_or(true),
    }
}

/// A JSON value as the option the ports take: null is nothing, anything else is the object.
fn obj(v: &Value) -> Option<&Value> {
    if v.is_null() { None } else { Some(v) }
}

fn to_devices(value: &Value) -> Vec<Device> {
    value.as_array().map(|a| a.iter().map(to_device).collect()).unwrap_or_default()
}

#[test]
fn every_case_of_the_studio_gives_the_same_answer() {
    let cases = fixtures();
    assert!(cases.len() >= 56, "the generator found the pure zone: {} cases", cases.len());
    for case in &cases {
        let function = case["fn"].as_str().unwrap();
        let note = case["note"].as_str().unwrap_or("");
        let args = case["args"].as_array().unwrap();
        let theirs = case["got"].clone();

        let mine = match function {
            "npuMegabytes" => json!(npu_megabytes(args[0].as_u64().unwrap_or(0))),
            "npuProgressModel" => {
                let p = npu_progress_model(obj(&args[0]));
                json!({"visible": p.visible, "percent": p.percent, "text": p.text})
            }
            "npuCardModel" => {
                let card = npu_card_model(obj(&args[0]));
                json!({
                    "summary": card.summary,
                    "offload": card.offload,
                    "tone": card.tone,
                    "hardware": card.hardware.iter().map(|h| json!({
                        "label": h.label, "badge": h.badge, "supported": h.supported, "problem": h.problem,
                    })).collect::<Vec<_>>(),
                    "button": card.button.as_ref().map(|b| json!({"action": b.action, "label": b.label})),
                    "cancel": card.cancel,
                    "remove": card.remove,
                    "progress": {"visible": card.progress.visible, "percent": card.progress.percent, "text": card.progress.text},
                    "error": card.error,
                    "formats": card.formats.iter().map(|f| json!({"title": f.title, "items": f.items})).collect::<Vec<_>>(),
                    "explanation": card.explanation,
                })
            }
            "deviceKey" => json!(device_key(&to_device(&args[0]))),
            "deviceRank" => json!(device_rank(&to_device(&args[0]))),
            "mergeDevices" => {
                let devices = to_devices(&args[0]);
                let wanted = args[1].as_str().unwrap_or("");
                let groups = merge_devices(&devices, wanted);
                json!(groups
                    .iter()
                    .map(|g| json!({
                        "id": g.primary.id,
                        "name": g.primary.name,
                        "backend": g.primary.backend,
                        "api": g.primary.api,
                        "dev_type": g.primary.dev_type,
                        "total_memory_bytes": g.primary.total_memory_bytes,
                        "available": g.primary.available,
                        "has_id": g.primary.has_id,
                        "members": g.members.iter().map(|m| json!({"id": m.id, "backend": m.backend, "api": m.api})).collect::<Vec<_>>(),
                        "others": g.others.iter().map(|(o, label)| json!({"id": o.id, "backend": o.backend, "api": o.api, "label": label})).collect::<Vec<_>>(),
                    }))
                    .collect::<Vec<_>>())
            }
            "deviceBackendLabel" => json!(device_backend_label(&to_device(&args[0]))),
            "deviceChoices" => {
                // the first argument is the hardware report; its devices are what is offered
                let devices = to_devices(args[0].get("devices").unwrap_or(&args[0]));
                let npu = obj(&args[1]);
                let current = args[2].as_str().unwrap_or("");
                let choices = device_choices(&devices, npu, current);
                json!(choices
                    .iter()
                    .map(|c| {
                        let mut entry = json!({"value": c.value, "label": c.label, "warning": c.warning});
                        // the Studio only writes the key when it is true
                        if c.alternative {
                            entry["alternative"] = json!(true);
                        }
                        entry
                    })
                    .collect::<Vec<_>>())
            }
            "runningDevice" => {
                // the second argument is the hardware report: its devices and its automatic choice
                let hw = &args[1];
                let devices = to_devices(hw.get("devices").unwrap_or(&Value::Array(Default::default())));
                let r = running_device(
                    args[0].as_str().unwrap_or(""),
                    &devices,
                    hw.get("auto_choice").and_then(Value::as_str).unwrap_or(""),
                    args[2].as_str().unwrap_or(""),
                );
                json!({"label": r.label, "title": r.title, "auto": r.auto})
            }
            "deviceMemory" => {
                // the Studio reads the free memory from the device itself (one argument)
                let d = to_device(&args[0]);
                let m = device_memory(&d);
                json!({"known": m.known, "pct": m.pct, "level": m.level})
            }
            "deviceRole" => {
                // the first argument is the device; its members decide, or its own id when it has none
                let d = &args[0];
                let members: Vec<String> = d
                    .get("members")
                    .and_then(Value::as_array)
                    .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect())
                    .unwrap_or_default();
                let own = d.get("id").and_then(Value::as_str).unwrap_or("");
                json!(device_role(
                    &members,
                    own,
                    args[1].as_str().unwrap_or(""),
                    args[2].as_str().unwrap_or(""),
                    args[3].as_bool().unwrap_or(false),
                ))
            }
            other => panic!("a case for {other}"),
        };
        assert_eq!(mine, theirs, "{function}: {note}");
    }
}
