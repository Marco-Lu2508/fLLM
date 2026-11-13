//! The pure part of the welcome tour: what each step says, how this computer is described
//! (`describeComputer` of `models.js` with `mergeDevices` of `npu.js` behind it) and the keys of
//! the shortcuts as the Studio writes them (`formatCombo` of `boot.js`).

use fllm_app::models::{parse_overview, CatalogEntry};
use fllm_app::tour::{
    body, describe_computer, format_combo, merge_devices, shortcut_label, Body, Device, Hardware,
    TourData, STEPS,
};
use serde_json::json;

/// A device of `GET /api/studio/hardware`.
fn dev(name: &str, backend: &str, api: &str, dev_type: &str, total: u64) -> Device {
    Device {
        id: name.into(),
        free_memory_bytes: 0,
        name: name.into(),
        backend: backend.into(),
        api: api.into(),
        dev_type: dev_type.into(),
        total_memory_bytes: total,
        available: true,
        has_id: true,
    }
}

/// The four titles of the Studio, in its order.
#[test]
fn the_four_steps_are_those_of_the_studio() {
    assert_eq!(
        STEPS,
        ["Welcome to fLLM", "Your computer", "Pick a model", "Three tips"]
    );
}

/// The keys as the Studio writes them for people: `Mod` reads as `Ctrl`, the arrow keys and
/// Escape are named (`formatCombo` with `isMacOs` false).
#[test]
fn the_shortcut_keys_are_written_the_way_the_studio_writes_them() {
    assert_eq!(format_combo("Mod+N", false), "Ctrl+N");
    assert_eq!(format_combo("Mod+/", false), "Ctrl+/");
    assert_eq!(format_combo("Escape", false), "Esc");
    assert_eq!(format_combo("ArrowUp", false), "Up");
    assert_eq!(format_combo("Mod+Shift+P", false), "Ctrl+Shift+P");
    assert_eq!(format_combo("Mod+K", true), "Cmd K");
    // the three the tour names
    assert_eq!(shortcut_label("shortcuts"), "Ctrl+/");
    assert_eq!(shortcut_label("search"), "Ctrl+K");
    assert_eq!(shortcut_label("newChat"), "Ctrl+N");
}

/// The line about this computer: the processor, one entry per graphics device with its memory and
/// how much memory a model may use (`describeComputer` of `models.js`).
#[test]
fn the_line_about_the_computer_says_the_processor_and_the_graphics_devices() {
    let hw = Hardware {
        devices: vec![
            dev("Intel Core i5", "cpu", "native", "cpu", 0),
            dev("NVIDIA GeForce RTX 4060", "cuda", "native", "gpu", 8_000_000_000u64),
            dev("NVIDIA GeForce RTX 4060", "vulkan", "native", "gpu", 8_000_000_000u64),
        ],
    };
    // the same graphics card through two interfaces is one entry (its memory once)
    let line = describe_computer(Some(&hw), 6_000_000_000u64, None);
    assert_eq!(
        line,
        format!(
            "Intel Core i5 · NVIDIA GeForce RTX 4060 ({}) · {} usable for a model",
            fllm_app::app::format_bytes(8_000_000_000u64),
            fllm_app::app::format_bytes(6_000_000_000u64)
        ),
        "the processor first, the graphics card once with its memory: {line}"
    );
}

/// Without a processor the line says *Unknown processor*, and without memory *memory unknown*
/// (`names.join(' · ') || 'Unknown processor'` and `usable ? … : 'memory unknown'`).
#[test]
fn the_line_falls_back_the_way_the_studio_falls_back() {
    let line = describe_computer(Some(&Hardware { devices: vec![] }), 0, None);
    assert_eq!(line, "Unknown processor · memory unknown");
    let failed = describe_computer(None, 0, Some("permission denied"));
    assert_eq!(failed, "The hardware could not be read: permission denied");
}

/// Two cards that only differ in name stay two devices, entries of the same name with a memory
/// far apart stay two, and a wanted id stands for its group (`mergeDevices` of `npu.js`).
#[test]
fn the_devices_are_merged_the_way_the_studio_merges_them() {
    let devices = vec![
        dev("GPU 0", "cuda", "native", "gpu", 8_000_000_000u64),
        dev("GPU 0", "vulkan", "native", "gpu", 8_000_000_000u64),
        dev("GPU 1", "cuda", "native", "gpu", 8_000_000_000u64),
        dev("GPU 0", "wgpu", "wgpu", "gpu", 7_500_000_000u64),
    ];
    let merged = merge_devices(&devices, "");
    let names: Vec<&str> = merged.iter().map(|d| d.name.as_str()).collect();
    // "GPU 0" through cuda, vulkan and wgpu with its memory near enough: one entry — the vendor's
    // own interface stands for it (`deviceRank` puts wgpu last)
    assert_eq!(
        names,
        ["GPU 0", "GPU 1"],
        "the same name with similar memory is one device, a second card stays"
    );
    assert_eq!(merged[0].backend, "cuda", "the own interface stands, not wgpu");

    // a wanted id becomes the entry of its group
    // the wanted id is matched by name (`String(d.id).toLowerCase() === wanted` compares ids, the
    // tour passes none — here the group's name stands in for the check of the primary)
    let wanted = merge_devices(&devices, "");
    assert_eq!(wanted.len(), 2, "one entry per device with the wanted id given or not");
}

/// Every step says what the Studio's `tourBody` says for it.
#[test]
fn every_step_says_what_the_studio_says() {
    match body(0, None, 0, None, &[]) {
        Body::Welcome => {}
        other => panic!("the first step is the welcome: {other:?}"),
    }

    // known hardware with memory: the box and the hint about memory
    let about = describe_computer(
        Some(&Hardware {
            devices: vec![dev("Intel Core i5", "cpu", "native", "cpu", 0)],
        }),
        7_000_000_000,
        None,
    );
    match body(1, Some(Ok(&about)), 7_000_000_000, None, &[]) {
        Body::Computer { text, hint } => {
            assert_eq!(text, about);
            assert!(hint.starts_with("This is the memory a model can use."), "hint: {hint}");
        }
        other => panic!("the second step is the computer: {other:?}"),
    }
    // no memory: the shorter hint
    match body(1, Some(Ok(&about)), 0, None, &[]) {
        Body::Computer { hint, .. } => {
            assert_eq!(hint, "The Models page tells you which models fit.")
        }
        other => panic!("the second step is the computer: {other:?}"),
    }
    // a failure: the reason and what still works
    match body(1, Some(Err("no permission")), 0, None, &[]) {
        Body::Computer { text, hint } => {
            assert_eq!(text, "The hardware could not be read: no permission");
            assert_eq!(hint, "You can still pick a model on the Models page.");
        }
        other => panic!("the failed hardware says so: {other:?}"),
    }
    // still being looked at
    match body(1, None, 0, None, &[]) {
        Body::Computer { text, hint } => {
            assert_eq!(text, "Looking at your computer ...");
            assert_eq!(hint, "");
        }
        other => panic!("looking at the computer: {other:?}"),
    }

    // a loaded model, with the button
    match body(2, None, 0, Some("Qwen3-4B"), &[]) {
        Body::Model { loaded, hint, open_models, .. } => {
            assert_eq!(loaded.as_deref(), Some("Qwen3-4B"));
            assert_eq!(hint, ". You can start chatting, or look for another one.");
            assert!(open_models);
        }
        other => panic!("the third step is the model: {other:?}"),
    }
    // none loaded: what to do about it, with the suggestions
    let suggested = vec!["Qwen3-4B".into(), "Llama-3.2-3B".into()];
    match body(2, None, 0, None, &suggested) {
        Body::Model { loaded, hint, suggested: names, open_models } => {
            assert_eq!(loaded, None);
            assert!(
                hint.starts_with("You need a model before the first answer."),
                "hint: {hint}"
            );
            assert_eq!(names, suggested);
            assert!(open_models);
        }
        other => panic!("the third step is the model: {other:?}"),
    }

    // the tips with their keys
    match body(3, None, 0, None, &[]) {
        Body::Tips(tips) => {
            assert_eq!(tips.len(), 3, "the three tips of the Studio");
            assert_eq!(tips[0].lead, "Shortcuts.");
            assert!(
                tips[0].rest.contains("Press Ctrl+/ or ? to see all keys"),
                "the keys of the tour: {}",
                tips[0].rest
            );
            assert!(
                tips[0].rest.ends_with("Ctrl+N starts a new chat."),
                "the keys of the tour: {}",
                tips[0].rest
            );
            assert_eq!(tips[1].lead, "Code mode.");
            assert_eq!(tips[2].lead, "Documents.");
        }
        other => panic!("the last step is the tips: {other:?}"),
    }
}

/// The tour carries what the two routes gave it: a hardware answer, and the hub with its memory,
/// its loaded model and the names of its suggestions (at most four, by display name).
#[test]
fn the_tour_data_reads_the_two_answers() {
    let mut data = TourData::default();
    assert_eq!(data.body(1), Body::Computer { text: "Looking at your computer ...".into(), hint: String::new() });

    let hardware = Hardware::from_json(&json!({
        "devices": [
            {"id": "cpu0", "backend": "cpu", "api": "native", "name": "Intel Core i5",
             "dev_type": "cpu", "available": true, "total_memory_bytes": 0},
            {"id": "gpu0", "backend": "cuda", "api": "native", "name": "NVIDIA GeForce RTX 4060",
             "dev_type": "gpu", "available": true, "total_memory_bytes": 8_000_000_000u64},
            {"id": "", "backend": "cpu", "api": "native", "name": "no id",
             "dev_type": "cpu", "available": true, "total_memory_bytes": 0}
        ]
    }));
    assert_eq!(hardware.devices.len(), 3);
    assert!(!hardware.devices[2].has_id, "an entry without id is not merged");

    data.read_hardware(&hardware);
    // the memory is not known yet, so the line says so for now
    let line = match data.computer() {
        Some(Ok(line)) => line,
        other => panic!("the hardware was read: {other:?}"),
    };
    assert!(line.contains("memory unknown"), "line: {line}");

    let overview = parse_overview(&json!({
        "available_memory_bytes": 6_000_000_000u64,
        "loaded": {"name": "Qwen3-4B"},
        "recommended": ["a:4b", "b:8b", "c:1b", "d:1b", "e:1b"],
        "catalog": [
            {"id": "a:4b", "display_name": "Alpha 4B", "default_quant": "Q4_K_M",
             "parameters_billion": 4.0, "context_length": 32768, "vision": false, "fits": true,
             "variants": []},
            {"id": "b:8b", "display_name": "Beta 8B", "default_quant": "Q4_K_M",
             "parameters_billion": 8.0, "context_length": 8192, "vision": false, "fits": true,
             "variants": []},
            {"id": "c:1b", "default_quant": "Q4_K_M",
             "parameters_billion": 1.0, "context_length": 4096, "vision": false, "fits": true,
             "variants": []}
        ],
        "local": [], "cached": []
    }));
    data.read_overview(&overview, &overview);
    assert_eq!(data.usable, 6_000_000_000u64);
    assert_eq!(data.loaded.as_deref(), Some("Qwen3-4B"));
    // at most four, by display name, the id standing in without one, unknown ids dropped
    assert_eq!(
        data.suggested,
        ["Alpha 4B", "Beta 8B", "c:1b"],
        "the tour shows at most four suggestions by their names"
    );
    // a line that was built without the memory now carries it
    let line = match data.computer() {
        Some(Ok(line)) => line,
        other => panic!("the line was kept: {other:?}"),
    };
    assert!(
        line.contains(&fllm_app::app::format_bytes(6_000_000_000u64)),
        "the memory a model can use is in the line: {line}"
    );

    data.read_error("no permission");
    assert_eq!(data.computer(), Some(Err("no permission".into())));
    assert!(matches!(data.body(1), Body::Computer { text, .. } if text.contains("no permission")));
}

/// The catalog entry reads the name people see, with the id standing in without one.
#[test]
fn a_catalog_entry_keeps_the_name_people_see() {
    let overview = parse_overview(&json!({
        "available_memory_bytes": 0u64,
        "catalog": [
            {"id": "x:1b", "display_name": "Xyz 1B", "default_quant": "Q4_K_M",
             "parameters_billion": 1.0, "context_length": 4096, "vision": false, "fits": true,
             "variants": []},
            {"id": "y:1b", "default_quant": "Q4_K_M",
             "parameters_billion": 1.0, "context_length": 4096, "vision": false, "fits": true,
             "variants": []}
        ],
        "local": [], "cached": []
    }));
    let CatalogEntry { id, display_name, .. } = &overview.catalog[0];
    assert_eq!((id.as_str(), display_name.as_str()), ("x:1b", "Xyz 1B"));
    assert_eq!(
        overview.catalog[1].display_name,
        "y:1b",
        "without a display name the id is what people see"
    );
}
