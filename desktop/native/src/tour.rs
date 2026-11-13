//! The welcome tour of the Studio (`voice.js`, the `-- the welcome tour --` block): four steps
//! with what each says, the description of this computer (`describeComputer` of `models.js` with
//! `mergeDevices` of `npu.js` behind it) and the keys of the shortcuts as `formatCombo` writes
//! them. The drawing lives in `app.rs` (`tour_window`), the first start and the button of
//! Settings > General open it.

use crate::app::format_bytes;

/// The titles of the four steps, in order (`STEPS` of `voice.js`).
pub const STEPS: [&str; 4] = ["Welcome to fLLM", "Your computer", "Pick a model", "Three tips"];

/// One line of the tips of the last step: a strong lead and the rest (`tourBody` step 3).
#[derive(Debug, Clone, PartialEq)]
pub struct Tip {
    pub lead: &'static str,
    pub rest: String,
}

/// The two paragraphs of the first step (`tourBody` step 0 of `voice.js`).
pub const WELCOME: [&str; 2] = [
    "fLLM runs language models on your own computer, on the processor or on any graphics card. Your chats, files and models stay on this machine.",
    "It is a chat, a coding agent and a model manager in one program, and it speaks the same API as the big services, so other tools can use it too.",
];

/// What the body of a step has to say — the branches of `tourBody` of `voice.js`, each its own.
#[derive(Debug, Clone, PartialEq)]
pub enum Body {
    /// Step 1: the two paragraphs of the welcome.
    Welcome,
    /// Step 2: this computer; `text` is `Looking at your computer ...` while it is not known yet.
    Computer { text: String, hint: String },
    /// Step 3: the model; `suggested` is at most four names, the button opens the Models page.
    Model {
        loaded: Option<String>,
        hint: String,
        suggested: Vec<String>,
        open_models: bool,
    },
    /// Step 4: the three tips with their keys.
    Tips(Vec<Tip>),
}

/// What the tour has to say — the values of `state` its `tourBody` reads: the line about this
/// computer (while it is unknown it is `None`, with the reason of a failure it is `Err`), how much
/// memory a model may use, the loaded model and the names the hub suggests.
#[derive(Debug, Clone, Default)]
pub struct TourData {
    /// The hardware answer, kept so the line can be built with any memory there is.
    pub hardware: Option<Hardware>,
    /// The reason the hardware could not be read (`state.hardwareError`).
    pub error: Option<String>,
    pub usable: u64,
    pub loaded: Option<String>,
    pub suggested: Vec<String>,
}

impl TourData {
    /// The line about this computer as it stands now: the failure if there is one, the hardware
    /// with the memory that is known by now, or nothing while it is still being looked at.
    pub fn computer(&self) -> Option<Result<String, String>> {
        if let Some(error) = &self.error {
            return Some(Err(error.clone()));
        }
        self.hardware
            .as_ref()
            .map(|hw| Ok(describe_computer(Some(hw), self.usable, None)))
    }

    /// The body of a step out of what is known now.
    pub fn body(&self, step: usize) -> Body {
        let computer = self.computer();
        body(
            step,
            computer
                .as_ref()
                .map(|r| r.as_ref().map(String::as_str).map_err(String::as_str)),
            self.usable,
            self.loaded.as_deref(),
            &self.suggested,
        )
    }

    /// The hardware answer from `GET /api/studio/hardware` (`describeComputer` reads its devices).
    pub fn read_hardware(&mut self, hardware: &Hardware) {
        self.hardware = Some(hardware.clone());
    }

    /// The reason the hardware could not be read (`describeComputer` with `state.hardwareError`).
    pub fn read_error(&mut self, error: &str) {
        self.error = Some(error.to_string());
    }

    /// The memory of a model, the loaded model and the suggestions — the hub answer. The line
    /// about the computer is built again with the memory it now knows.
    pub fn read_overview(&mut self, overview: &crate::models::Overview, catalog: &crate::models::Overview) {
        self.usable = overview.memory;
        self.loaded = overview.loaded.clone();
        // the tour names at most four suggested models, by display name (`byId` of `voice.js`)
        self.suggested = overview
            .recommended
            .iter()
            .filter_map(|id| catalog.catalog.iter().find(|e| &e.id == id).map(|e| e.display_name.clone()))
            .take(4)
            .collect();
    }
}

/// The hardware answer read into what the tour needs (`devices` of `GET /api/studio/hardware`).
impl Hardware {
    pub fn from_json(v: &serde_json::Value) -> Self {
        let devices = v
            .get("devices")
            .and_then(serde_json::Value::as_array)
            .map(|a| {
                a.iter()
                    .map(|d| Device {
                        name: d.get("name").and_then(serde_json::Value::as_str).unwrap_or_default().to_string(),
                        backend: d.get("backend").and_then(serde_json::Value::as_str).unwrap_or_default().to_string(),
                        api: d.get("api").and_then(serde_json::Value::as_str).unwrap_or_default().to_string(),
                        dev_type: d.get("dev_type").and_then(serde_json::Value::as_str).unwrap_or_default().to_string(),
                        total_memory_bytes: d
                            .get("total_memory_bytes")
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or_default(),
                        available: d
                            .get("available")
                            .and_then(serde_json::Value::as_bool)
                            .unwrap_or(true),
                        has_id: d
                            .get("id")
                            .and_then(serde_json::Value::as_str)
                            .is_some_and(|s| !s.is_empty()),
                    })
                    .collect()
            })
            .unwrap_or_default();
        Hardware { devices }
    }
}

/// The keys of a shortcut as people see them (`shortcutLabel`/`formatCombo` of `boot.js` with the
/// defaults of the three shortcuts the tour names, `Mod` read as `Ctrl` off macOS).
pub fn shortcut_label(id: &str) -> String {
    let combo = match id {
        "newChat" => "Mod+N",
        "search" => "Mod+K",
        "shortcuts" => "Mod+/",
        _ => return String::new(),
    };
    format_combo(combo, false)
}

/// The keys of `combo` as people see them (`formatCombo` of `boot.js`: the names of the arrow keys
/// and Escape, `Mod` read as `Cmd` on macOS and `Ctrl` elsewhere, the parts joined by a space on
/// macOS and by a `+` everywhere else).
pub fn format_combo(combo: &str, is_mac: bool) -> String {
    let mut mods: Vec<&str> = Vec::new();
    let mut key = "";
    for part in combo.split('+') {
        match part {
            "Mod" | "Alt" | "Shift" | "Ctrl" | "Meta" => mods.push(part),
            _ => key = part,
        }
    }
    if key.is_empty() {
        return String::new();
    }
    let named = match key {
        "ArrowUp" => "Up",
        "ArrowDown" => "Down",
        "ArrowLeft" => "Left",
        "ArrowRight" => "Right",
        "Escape" => "Esc",
        other => other,
    };
    let mut parts: Vec<String> = mods
        .iter()
        .map(|m| match *m {
            "Mod" => String::from(if is_mac { "Cmd" } else { "Ctrl" }),
            "Alt" if is_mac => String::from("Option"),
            other => other.to_string(),
        })
        .collect();
    parts.push(named.to_string());
    parts.join(if is_mac { " " } else { "+" })
}

/// One device of `GET /api/studio/hardware` as the tour needs it: its name, its kind and its
/// memory (`devices` of the route, the fields `describeComputer` reads).
#[derive(Debug, Clone, PartialEq)]
pub struct Device {
    pub name: String,
    pub backend: String,
    pub api: String,
    pub dev_type: String,
    pub total_memory_bytes: u64,
    pub available: bool,
    pub has_id: bool,
}

/// The hardware of `GET /api/studio/hardware`: only what the tour reads.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Hardware {
    pub devices: Vec<Device>,
}

/// The name of a device without the marks it may carry (`deviceKey` of `npu.js`: `(R)` and `(TM)`
/// out, spaces folded, lower case — entries with the same key are the same physical device).
fn device_key(d: &Device) -> String {
    let mut out = String::with_capacity(d.name.len());
    let chars: Vec<char> = d.name.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '(' && i + 1 < chars.len() && (chars[i + 1] == 'R' || chars[i + 1] == 'T') {
            let mut j = i + 1;
            let mut close = None;
            while j < chars.len() && j < i + 5 {
                if chars[j] == ')' {
                    close = Some(j);
                    break;
                }
                j += 1;
            }
            if let Some(close) = close {
                let mark: String = chars[i + 1..close].iter().collect();
                if mark == "R" || mark == "TM" {
                    i = close + 1;
                    continue;
                }
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
}

/// Lower is better: the vendor's own interface first, then the native graphics API, the generic
/// (wgpu) entries last (`deviceRank` of `npu.js`).
fn device_rank(d: &Device) -> usize {
    if d.api == "wgpu" {
        return 3;
    }
    let backend = d.backend.to_lowercase();
    if ["cuda", "hip", "metal", "sycl"].contains(&backend.as_str()) {
        0
    } else {
        1
    }
}

/// One entry per physical device, the processor and NPUs never merged (`mergeDevices` of `npu.js`:
/// entries with the same name and a similar memory size are one device, an entry joins a group only
/// when the group has no entry of the same backend and interface yet, the best interface stands for
/// the device; `wantedId` would become the entry of its group — the tour passes none).
pub fn merge_devices<'a>(devices: &'a [Device], wanted_id: &str) -> Vec<&'a Device> {
    let wanted = wanted_id.to_lowercase();
    let list: Vec<&Device> = devices.iter().filter(|d| d.has_id).collect();
    let mergeable = |d: &Device| {
        let backend = d.backend.to_lowercase();
        backend != "cpu" && backend != "npu" && !device_key(d).is_empty()
    };
    // the groups, in the order their first entry came
    let mut groups: Vec<Vec<&Device>> = Vec::new();
    let mut group_mergeable: Vec<bool> = Vec::new();
    for d in list {
        let is_mergeable = mergeable(d);
        let found = if is_mergeable {
            groups.iter().enumerate().find(|(gi, items)| {
                if !group_mergeable[*gi] || device_key(items[0]) != device_key(d) {
                    return false;
                }
                if items
                    .iter()
                    .any(|o| o.backend == d.backend && o.api == d.api)
                {
                    return false;
                }
                let a = items[0].total_memory_bytes;
                let b = d.total_memory_bytes;
                a == 0 || b == 0 || a.min(b) as f64 / a.max(b) as f64 >= 0.5
            })
        } else {
            None
        };
        match found {
            Some((gi, _)) => groups[gi].push(d),
            None => {
                groups.push(vec![d]);
                group_mergeable.push(is_mergeable);
            }
        }
    }
    let mut out = Vec::with_capacity(groups.len());
    for (gi, items) in groups.iter().enumerate() {
        if !group_mergeable[gi] {
            out.push(items[0]);
            continue;
        }
        let mut ordered: Vec<&&Device> = items.iter().collect();
        ordered.sort_by_key(|d| device_rank(d));
        let primary = items
            .iter()
            .find(|d| d.name.to_lowercase() == wanted)
            .unwrap_or(ordered[0]);
        out.push(primary);
    }
    out
}

/// The line about this computer (`describeComputer` of `models.js`): the processor, one entry per
/// graphics device with its memory, and how much memory a model may use.
pub fn describe_computer(hardware: Option<&Hardware>, usable: u64, error: Option<&str>) -> String {
    let (Some(hw), None) = (hardware, error) else {
        return match error {
            Some(e) => format!("The hardware could not be read: {e}"),
            None => String::from("Unknown processor · memory unknown"),
        };
    };
    let mut names: Vec<String> = Vec::new();
    for d in hw.devices.iter().filter(|d| d.backend == "cpu") {
        names.insert(0, d.name.clone());
    }
    let graphics: Vec<Device> = hw
        .devices
        .iter()
        .filter(|d| {
            let backend = d.backend.to_lowercase();
            backend != "cpu" && backend != "npu" && d.available
        })
        .cloned()
        .collect();
    for d in merge_devices(&graphics, "") {
        let label = if d.dev_type == "gpu" {
            format!("{} ({})", d.name, format_bytes(d.total_memory_bytes))
        } else {
            d.name.clone()
        };
        if !names.contains(&label) {
            names.push(label);
        }
    }
    let head = if names.is_empty() {
        String::from("Unknown processor")
    } else {
        names.join(" · ")
    };
    let memory = if usable > 0 {
        format!("{} usable for a model", format_bytes(usable))
    } else {
        String::from("memory unknown")
    };
    format!("{head} · {memory}")
}

/// The body of a step, with what the Studio knows at this moment (`tourBody` of `voice.js`).
/// `computer` is the line of [`describe_computer`]; the caller says whether the hardware is still
/// being looked at (`None`), could not be read (`Err`) or is known (`Ok`).
pub fn body(
    step: usize,
    computer: Option<Result<&str, &str>>,
    usable: u64,
    loaded: Option<&str>,
    suggested: &[String],
) -> Body {
    match step {
        0 => Body::Welcome,
        1 => match computer {
            Some(Ok(text)) => Body::Computer {
                text: text.to_string(),
                hint: if usable > 0 {
                    String::from(
                        "This is the memory a model can use. Bigger models are smarter but need more memory and are slower; the Models page tells you which ones fit.",
                    )
                } else {
                    String::from("The Models page tells you which models fit.")
                },
            },
            Some(Err(e)) => Body::Computer {
                text: format!("The hardware could not be read: {e}"),
                hint: String::from("You can still pick a model on the Models page."),
            },
            None => Body::Computer {
                text: String::from("Looking at your computer ..."),
                hint: String::new(),
            },
        },
        2 => Body::Model {
            loaded: loaded.map(str::to_string),
            hint: if loaded.is_some() {
                String::from(". You can start chatting, or look for another one.")
            } else {
                String::from(
                    "You need a model before the first answer. The Models page suggests ones that run well on this computer and downloads them with one click.",
                )
            },
            suggested: suggested.to_vec(),
            open_models: true,
        },
        _ => Body::Tips(vec![
            Tip {
                lead: "Shortcuts.",
                rest: format!(
                    "Press {} or ? to see all keys and change them. {} searches, {} starts a new chat.",
                    shortcut_label("shortcuts"),
                    shortcut_label("search"),
                    shortcut_label("newChat")
                ),
            },
            Tip {
                lead: "Code mode.",
                rest: String::from(
                    "Switch to Code in the sidebar: an agent that reads and edits the files of a folder you choose and runs commands, asking you first.",
                ),
            },
            Tip {
                lead: "Documents.",
                rest: String::from(
                    "Attach a PDF, Word, Excel or text file with the plus button, or drop it into the chat, and ask about it.",
                ),
            },
        ]),
    }
}
