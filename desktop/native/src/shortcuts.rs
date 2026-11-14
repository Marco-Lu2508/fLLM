//! The keyboard shortcuts of the Studio (`boot.js`, the `@pure-begin`..`@pure-end` block): the
//! list, the fixed keys nobody may take, and the rules for a combination — normal form, how it is
//! shown to people, which keys are in force, whether a new combination may be given, and which
//! shortcut a combination triggers. `format_combo` lives in `tour.rs` (the tour shows the same
//! keys). The drawing of the overview is `shortcuts_window` in `app.rs`.

use crate::tour::{format_combo, is_mac};
use std::collections::HashMap;

/// One shortcut of the list (`SHORTCUTS` of `boot.js`): its id, its default keys and what it says
/// to people.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shortcut {
    pub id: &'static str,
    pub keys: &'static str,
    pub desc: &'static str,
}

/// The shortcuts of the Studio, in its order (`SHORTCUTS` of `boot.js`).
pub const SHORTCUTS: [Shortcut; 15] = [
    Shortcut { id: "newChat", keys: "Mod+N", desc: "New chat (new session in Code)" },
    Shortcut { id: "search", keys: "Mod+K", desc: "Search chats and settings" },
    Shortcut { id: "focusComposer", keys: "Shift+Escape", desc: "Put the cursor in the message field" },
    Shortcut { id: "toggleSidebar", keys: "Mod+B", desc: "Show or hide the sidebar" },
    Shortcut { id: "stop", keys: "Escape", desc: "Stop the answer or the agent, close a window" },
    Shortcut { id: "surfaceChat", keys: "Alt+1", desc: "Go to Chat" },
    Shortcut { id: "surfaceCode", keys: "Alt+2", desc: "Go to Code" },
    Shortcut { id: "openModels", keys: "Alt+3", desc: "Open Models" },
    Shortcut { id: "openSettings", keys: "Mod+,", desc: "Open Settings" },
    Shortcut { id: "nextChat", keys: "Alt+ArrowDown", desc: "Next chat in the list" },
    Shortcut { id: "prevChat", keys: "Alt+ArrowUp", desc: "Previous chat in the list" },
    Shortcut { id: "regenerate", keys: "Alt+R", desc: "Regenerate the last answer" },
    Shortcut { id: "shortcuts", keys: "Mod+/", desc: "This overview (also ? outside a text field)" },
    Shortcut { id: "codeAgent", keys: "Alt+B", desc: "Code: switch between the build and the plan agent" },
    Shortcut { id: "codeSessions", keys: "Alt+L", desc: "Code: pick an earlier session of this folder" },
];

/// Keys with a fixed meaning that no shortcut may take (`FIXED_KEYS` of `boot.js`). The overview
/// does not show the row of `?`.
pub const FIXED_KEYS: [(&str, &str); 4] = [
    ("Enter", "Send"),
    ("Shift+Enter", "New line"),
    ("ArrowUp", "Edit your last message (empty field)"),
    ("?", "This overview"),
];

/// The order of the modifiers in the normal form (`MOD_ORDER` of `boot.js`).
pub const MOD_ORDER: [&str; 4] = ["Mod", "Ctrl", "Alt", "Shift"];

/// One combination: its modifiers in `MOD_ORDER` and its key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Combo {
    pub mods: Vec<&'static str>,
    pub key: String,
}

/// The combination of a text, or nothing when there is none (`parseCombo` of `boot.js`): every
/// leading `+`-separated word that is a modifier is one, the rest is the key (one letter is
/// upper case), and the modifiers end up in the fixed order.
pub fn parse_combo(text: &str) -> Option<Combo> {
    let s = text.trim();
    if s.is_empty() {
        return None;
    }
    let mut mods: Vec<&'static str> = Vec::new();
    let mut rest = s;
    loop {
        let i = rest.find('+');
        match i {
            Some(i) if i > 0 => {
                // a word that is no modifier ends the list of modifiers (the `break` of boot.js)
                let Some(head) = MOD_ORDER
                    .iter()
                    .find(|m| m.eq_ignore_ascii_case(&rest[..i]))
                    .copied()
                else {
                    break;
                };
                if !mods.contains(&head) {
                    mods.push(head);
                }
                rest = &rest[i + 1..];
            }
            _ => break,
        }
    }
    let key = if rest.chars().count() == 1 {
        rest.to_uppercase()
    } else {
        rest.to_string()
    };
    mods.sort_by_key(|m| MOD_ORDER.iter().position(|o| o == m).unwrap_or(usize::MAX));
    Some(Combo { mods, key })
}

/// The normal form of a combination text (modifiers in a fixed order, letters in upper case).
pub fn norm_combo(text: &str) -> String {
    match parse_combo(text) {
        Some(p) => {
            let mut parts: Vec<String> = p.mods.iter().map(|m| m.to_string()).collect();
            parts.push(p.key);
            parts.join("+")
        }
        None => String::new(),
    }
}

/// The keys in force: the defaults with the saved changes on top; changes for unknown ids are
/// ignored (`effectiveCombos` of `boot.js`).
pub fn effective_combos(overrides: &HashMap<String, String>) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for s in SHORTCUTS.iter() {
        let o = overrides
            .get(s.id)
            .map(|o| norm_combo(o))
            .filter(|o| !o.is_empty());
        out.insert(s.id.to_string(), o.unwrap_or_else(|| norm_combo(s.keys)));
    }
    out
}

/// Whether `combo` may be given to shortcut `id`; on yes the normal form comes back, on no the
/// reason the Studio gives (`checkCombo` of `boot.js`).
pub fn check_combo(combos: &HashMap<String, String>, id: &str, text: &str) -> Result<String, String> {
    let combo = norm_combo(text);
    let Some(p) = parse_combo(&combo) else {
        return Err(String::from("Press a key combination."));
    };
    let named = p.key.chars().count() > 1;
    let plain_named_ok = named
        && !["Enter", "Tab", "Space", "Backspace", "Delete"].contains(&p.key.as_str());
    if p.mods.is_empty() && !plain_named_ok {
        return Err(String::from(
            "Use at least one of Ctrl, Alt or Cmd with the key, so typing is not disturbed.",
        ));
    }
    if p.mods.len() == 1 && p.mods[0] == "Shift" && !named {
        return Err(String::from("Shift alone is used for typing capital letters."));
    }
    for (keys, desc) in FIXED_KEYS {
        if norm_combo(keys) == combo {
            return Err(format!("{} is fixed: {}.", format_combo(&combo, false), desc));
        }
    }
    for s in SHORTCUTS.iter() {
        if s.id != id && combos.get(s.id).map(String::as_str) == Some(combo.as_str()) {
            return Err(format!(
                "{} is already used for: {}.",
                // the Studio writes these two messages with `formatCombo(combo, false)`, so the
                // reason of a refused key reads the same on every platform (`checkCombo`)
                format_combo(&combo, false),
                s.desc
            ));
        }
    }
    Ok(combo)
}

/// The shortcut id that `combo` triggers, or nothing (`shortcutFor` of `boot.js`).
pub fn shortcut_for(combos: &HashMap<String, String>, combo: &str) -> Option<&'static str> {
    SHORTCUTS
        .iter()
        .find(|s| combos.get(s.id).map(String::as_str) == Some(combo))
        .map(|s| s.id)
}

/// The keys of `combo` as people see them in the list (`formatCombo` of `boot.js`): what this
/// platform's `Mod` is called, joined the way this platform joins it.
pub fn format(combo: &str) -> String {
    format_combo(combo, is_mac())
}

/// The combination of a keyboard event, or nothing while only a modifier key is down
/// (`comboFromEvent` of `boot.js`): the modifiers are read as the Studio reads them — `Mod` is the
/// command key on macOS and the control key elsewhere, and Shift counts only for letters, digits
/// and named keys, because Shift is part of a typed symbol such as `?`.
pub fn combo_of_key(key: &iced::keyboard::Key, modifiers: iced::keyboard::Modifiers, is_mac: bool) -> String {
    // the modifier keys alone are no combination (the first line of comboFromEvent)
    if matches!(
        key,
        iced::keyboard::Key::Named(
            iced::keyboard::key::Named::Shift
                | iced::keyboard::key::Named::Control
                | iced::keyboard::key::Named::Alt
                | iced::keyboard::key::Named::AltGraph
                | iced::keyboard::key::Named::Meta
                | iced::keyboard::key::Named::Super
        )
    ) {
        return String::new();
    }
    let named = |n: &iced::keyboard::key::Named| match n {
        iced::keyboard::key::Named::Escape => String::from("Escape"),
        iced::keyboard::key::Named::Enter => String::from("Enter"),
        iced::keyboard::key::Named::Tab => String::from("Tab"),
        iced::keyboard::key::Named::Space => String::from("Space"),
        iced::keyboard::key::Named::Backspace => String::from("Backspace"),
        iced::keyboard::key::Named::Delete => String::from("Delete"),
        iced::keyboard::key::Named::ArrowUp => String::from("ArrowUp"),
        iced::keyboard::key::Named::ArrowDown => String::from("ArrowDown"),
        iced::keyboard::key::Named::ArrowLeft => String::from("ArrowLeft"),
        iced::keyboard::key::Named::ArrowRight => String::from("ArrowRight"),
        other => format!("{other:?}"),
    };
    let raw = match key {
        iced::keyboard::Key::Named(n) => named(n).to_string(),
        iced::keyboard::Key::Character(c) if c.as_str() == " " => String::from("Space"),
        iced::keyboard::Key::Character(c) => c.to_string(),
        iced::keyboard::Key::Unidentified => return String::new(),
    };
    if raw.is_empty() {
        return String::new();
    }
    let mut mods: Vec<&str> = Vec::new();
    let command = if is_mac { modifiers.logo() } else { modifiers.control() };
    if command {
        mods.push("Mod");
    }
    if is_mac && modifiers.control() {
        mods.push("Ctrl");
    }
    if modifiers.alt() {
        mods.push("Alt");
    }
    let is_named = raw.chars().count() > 1;
    let is_alnum = raw.chars().count() == 1 && raw.chars().next().is_some_and(|c| c.is_alphanumeric());
    if modifiers.shift() && (is_named || is_alnum) {
        mods.push("Shift");
    }
    let key_part = if raw.chars().count() == 1 {
        raw.to_uppercase()
    } else {
        raw
    };
    let mut parts: Vec<String> = mods.iter().map(|m| m.to_string()).collect();
    parts.push(key_part);
    norm_combo(&parts.join("+"))
}
