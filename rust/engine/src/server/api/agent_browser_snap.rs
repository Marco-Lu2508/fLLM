//! The page as text for a model, and keys as CDP events. Pure functions over canned data, tested without a browser.
//!
//! The snapshot comes from the accessibility tree of the browser (`Accessibility.getFullAXTree`): one line per
//! meaningful node, indented by depth, and every element a person could use (link, button, field, checkbox, menu item,
//! tab ...) carries a numeric `ref`. A ref stays the same for the same DOM node as long as the document stays, so a
//! click on `ref 7` still means the same button after a second snapshot.
use serde_json::{json, Value};
use std::collections::HashMap;

/// Roles a person can operate: they get a `ref`.
const INTERACTIVE: &[&str] = &[
    "link", "button", "textbox", "searchbox", "combobox", "checkbox", "radio", "switch", "menuitem", "menuitemcheckbox", "menuitemradio",
    "tab", "option", "slider", "spinbutton", "listbox", "treeitem", "textarea", "DisclosureTriangle", "PopUpButton", "ColorWell", "DateTime",
];
/// Roles that give a page its outline and are shown with their name.
const LANDMARKS: &[&str] = &[
    "banner", "navigation", "main", "contentinfo", "complementary", "search", "form", "region", "dialog", "alertdialog", "alert", "status", "table",
    "list", "tablist", "menu", "toolbar", "article", "figure", "img", "image", "tree", "grid", "heading", "listitem", "row", "cell", "columnheader", "rowheader",
];

const NAME_CAP: usize = 160;
const TEXT_CAP: usize = 240;

/// What a `ref` points at.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct RefTarget {
    pub backend: i64,
    pub role: String,
    pub name: String,
}

/// The numbering of the interactive elements of one document.
#[derive(Default, Debug)]
pub(super) struct RefMap {
    by_backend: HashMap<i64, u32>,
    by_ref: HashMap<u32, RefTarget>,
    next: u32,
}

impl RefMap {
    /// The ref of a DOM node: the same number as before when it was numbered already.
    pub(super) fn assign(&mut self, backend: i64, role: &str, name: &str) -> u32 {
        if let Some(&n) = self.by_backend.get(&backend) {
            self.by_ref.insert(n, RefTarget { backend, role: role.to_owned(), name: name.to_owned() });
            return n;
        }
        self.next += 1;
        self.by_backend.insert(backend, self.next);
        self.by_ref.insert(self.next, RefTarget { backend, role: role.to_owned(), name: name.to_owned() });
        self.next
    }
    pub(super) fn get(&self, n: u32) -> Option<&RefTarget> {
        self.by_ref.get(&n)
    }
    /// A new document: all numbers are forgotten.
    pub(super) fn reset(&mut self) {
        *self = Self::default();
    }
    pub(super) fn len(&self) -> usize {
        self.by_ref.len()
    }
}

fn text_of(value: &Value) -> String {
    match value {
        Value::Object(o) => o.get("value").map(text_of).unwrap_or_default(),
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        _ => String::new(),
    }
}

fn clip(text: &str, cap: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= cap {
        flat
    } else {
        let mut cut: String = flat.chars().take(cap).collect();
        cut.push('…');
        cut
    }
}

fn property<'a>(node: &'a Value, name: &str) -> Option<&'a Value> {
    node.get("properties")?.as_array()?.iter().find(|p| p.get("name").and_then(Value::as_str) == Some(name)).and_then(|p| p.get("value"))
}

fn flag(node: &Value, name: &str) -> bool {
    property(node, name).is_some_and(|v| matches!(v.get("value"), Some(Value::Bool(true))) || text_of(v) == "true")
}

/// How the page is described to the model.
pub(super) struct SnapshotOptions<'a> {
    pub title: &'a str,
    pub url: &'a str,
    pub max_chars: usize,
}

/// The accessibility tree as indented text. `refs` numbers the interactive elements.
pub(super) fn format_snapshot(nodes: &[Value], refs: &mut RefMap, options: &SnapshotOptions<'_>) -> String {
    let mut by_id: HashMap<&str, &Value> = HashMap::new();
    for node in nodes {
        if let Some(id) = node.get("nodeId").and_then(Value::as_str) {
            by_id.insert(id, node);
        }
    }
    let root = nodes.iter().find(|n| n.get("parentId").is_none()).or_else(|| nodes.first());
    let mut lines: Vec<String> = Vec::new();
    if let Some(root) = root {
        walk(root, &by_id, refs, 0, false, &mut lines);
    }
    let mut out = format!("Page: {}\nURL: {}\n", clip(options.title, NAME_CAP), options.url);
    if lines.is_empty() {
        out.push_str("(the page has no content yet; it may still be loading, or it is empty)\n");
    }
    let mut used = out.len();
    let mut shown = 0usize;
    for line in &lines {
        if used + line.len() + 1 > options.max_chars {
            break;
        }
        used += line.len() + 1;
        shown += 1;
        out.push_str(line);
        out.push('\n');
    }
    if shown < lines.len() {
        out.push_str(&format!("... {} more lines not shown (the page is long): use browser_scroll and take a new snapshot, or look at a part of the page.\n", lines.len() - shown));
    }
    out.push_str(&format!("[{} interactive elements; use their ref with browser_click, browser_type, browser_scroll]", refs.len()));
    out
}

fn walk(node: &Value, by_id: &HashMap<&str, &Value>, refs: &mut RefMap, depth: usize, parent_named: bool, lines: &mut Vec<String>) {
    let role = text_of(node.get("role").unwrap_or(&Value::Null));
    let ignored = node.get("ignored").and_then(Value::as_bool).unwrap_or(false);
    let name = clip(&text_of(node.get("name").unwrap_or(&Value::Null)), NAME_CAP);
    let backend = node.get("backendDOMNodeId").and_then(Value::as_i64);
    let children: Vec<&Value> = node.get("childIds").and_then(Value::as_array).map(|ids| ids.iter().filter_map(|i| i.as_str().and_then(|i| by_id.get(i).copied())).collect()).unwrap_or_default();
    let descend = |refs: &mut RefMap, lines: &mut Vec<String>, depth: usize, named: bool| {
        for child in &children {
            walk(child, by_id, refs, depth, named, lines);
        }
    };
    if ignored {
        descend(refs, lines, depth, parent_named);
        return;
    }
    let indent = "  ".repeat(depth.min(12));
    match role.as_str() {
        "InlineTextBox" | "LineBreak" | "none" | "presentation" => {
            descend(refs, lines, depth, parent_named);
        }
        "StaticText" | "text" => {
            // the name of a link or button already says this text
            if !parent_named && !name.is_empty() {
                lines.push(format!("{indent}- text: {}", clip(&name, TEXT_CAP)));
            }
        }
        r if INTERACTIVE.contains(&r) => {
            let mut line = format!("{indent}- {r}");
            if !name.is_empty() {
                line.push_str(&format!(" \"{name}\""));
            }
            let value = clip(&text_of(node.get("value").unwrap_or(&Value::Null)), 120);
            if !value.is_empty() {
                line.push_str(&format!(" value=\"{value}\""));
            }
            if let Some(checked) = property(node, "checked") {
                line.push_str(match text_of(checked).as_str() {
                    "true" => " checked",
                    "mixed" => " mixed",
                    _ => " unchecked",
                });
            }
            for (key, label) in [("disabled", "disabled"), ("selected", "selected"), ("required", "required"), ("readonly", "readonly")] {
                if flag(node, key) {
                    line.push(' ');
                    line.push_str(label);
                }
            }
            if let Some(expanded) = property(node, "expanded") {
                line.push_str(if text_of(expanded) == "true" { " expanded" } else { " collapsed" });
            }
            if r == "link" {
                if let Some(url) = property(node, "url") {
                    let url = text_of(url);
                    if !url.is_empty() {
                        line.push_str(&format!(" url={}", clip(&url, 200)));
                    }
                }
            }
            if let Some(backend) = backend {
                let n = refs.assign(backend, r, &name);
                line.push_str(&format!(" [ref={n}]"));
            }
            lines.push(line);
            descend(refs, lines, depth + 1, true);
        }
        r if LANDMARKS.contains(&r) => {
            let mut line = format!("{indent}- {r}");
            if !name.is_empty() {
                line.push_str(&format!(" \"{name}\""));
            }
            if r == "heading" {
                if let Some(level) = property(node, "level") {
                    line.push_str(&format!(" [level={}]", text_of(level)));
                }
            }
            lines.push(line);
            descend(refs, lines, depth + 1, !name.is_empty() && matches!(r, "heading" | "img" | "image"));
        }
        _ => descend(refs, lines, depth, parent_named),
    }
}

// ---- keys -------------------------------------------------------------------------------------------------------

/// Modifier bits of CDP: Alt 1, Control 2, Meta 4, Shift 8.
fn modifier_bit(name: &str) -> Option<u32> {
    match name.to_ascii_lowercase().as_str() {
        "alt" | "option" => Some(1),
        "control" | "ctrl" => Some(2),
        "meta" | "cmd" | "command" | "super" => Some(4),
        "shift" => Some(8),
        _ => None,
    }
}

/// `(key, code, windowsVirtualKeyCode, text)` of a named key or a single character.
fn key_info(name: &str) -> Option<(String, String, u32, String)> {
    let s = |a: &str, b: &str, c: u32, d: &str| Some((a.to_owned(), b.to_owned(), c, d.to_owned()));
    match name.to_ascii_lowercase().as_str() {
        "enter" | "return" => s("Enter", "Enter", 13, "\r"),
        "tab" => s("Tab", "Tab", 9, ""),
        "escape" | "esc" => s("Escape", "Escape", 27, ""),
        "backspace" => s("Backspace", "Backspace", 8, ""),
        "delete" | "del" => s("Delete", "Delete", 46, ""),
        "space" | " " => s(" ", "Space", 32, " "),
        "arrowup" | "up" => s("ArrowUp", "ArrowUp", 38, ""),
        "arrowdown" | "down" => s("ArrowDown", "ArrowDown", 40, ""),
        "arrowleft" | "left" => s("ArrowLeft", "ArrowLeft", 37, ""),
        "arrowright" | "right" => s("ArrowRight", "ArrowRight", 39, ""),
        "home" => s("Home", "Home", 36, ""),
        "end" => s("End", "End", 35, ""),
        "pageup" => s("PageUp", "PageUp", 33, ""),
        "pagedown" => s("PageDown", "PageDown", 34, ""),
        "insert" => s("Insert", "Insert", 45, ""),
        f if f.len() >= 2 && f.starts_with('f') && f[1..].parse::<u32>().is_ok_and(|n| (1..=12).contains(&n)) => {
            let n: u32 = f[1..].parse().ok()?;
            Some((format!("F{n}"), format!("F{n}"), 111 + n, String::new()))
        }
        _ => {
            let mut chars = name.chars();
            let c = chars.next()?;
            if chars.next().is_some() {
                return None;
            }
            let upper = c.to_ascii_uppercase();
            let (code, vk) = if c.is_ascii_alphabetic() {
                (format!("Key{upper}"), upper as u32)
            } else if c.is_ascii_digit() {
                (format!("Digit{c}"), c as u32)
            } else {
                (String::new(), 0)
            };
            Some((c.to_string(), code, vk, c.to_string()))
        }
    }
}

/// The CDP `Input.dispatchKeyEvent` parameter objects for a key such as `Enter`, `ArrowDown`, `a` or `Control+a`: a key
/// down and a key up.
pub(super) fn key_events(spec: &str) -> Result<Vec<Value>, String> {
    let spec = spec.trim();
    if spec.is_empty() {
        return Err("the key is empty".into());
    }
    // "+" itself is a key: "+" and "Control++" end in it
    let (mods, key_name) = if spec == "+" {
        ("", "+")
    } else if let Some(head) = spec.strip_suffix("++") {
        (head, "+")
    } else {
        spec.rsplit_once('+').unwrap_or(("", spec))
    };
    if key_name.is_empty() {
        return Err(format!("no key in '{spec}'"));
    }
    let mut modifiers = 0;
    for m in mods.split('+').filter(|m| !m.is_empty()) {
        modifiers |= modifier_bit(m).ok_or_else(|| format!("'{m}' is not a modifier (Control, Shift, Alt, Meta)"))?;
    }
    let (key, code, vk, text) = key_info(key_name).ok_or_else(|| format!("unknown key '{key_name}'. Use a name such as Enter, Tab, Escape, Backspace, ArrowDown, PageDown, F5, or one character"))?;
    let shifted = modifiers & 8 != 0;
    let key = if shifted && key.chars().count() == 1 { key.to_uppercase() } else { key };
    // with Control, Alt or Meta held a key is a shortcut, not typed text
    let typed = modifiers & 7 == 0 && !text.is_empty();
    let text = if typed && shifted { text.to_uppercase() } else { text };
    let command = if modifiers & 6 != 0 {
        match key.to_ascii_lowercase().as_str() {
            "a" => Some("selectAll"),
            "c" => Some("copy"),
            "x" => Some("cut"),
            "v" => Some("paste"),
            "z" => Some("undo"),
            "y" => Some("redo"),
            _ => None,
        }
    } else {
        None
    };
    let mut down = json!({
        "type": if typed { "keyDown" } else { "rawKeyDown" },
        "modifiers": modifiers,
        "key": key,
        "code": code,
        "windowsVirtualKeyCode": vk,
        "nativeVirtualKeyCode": vk,
    });
    if typed {
        down["text"] = json!(text);
        down["unmodifiedText"] = json!(text);
    }
    if let Some(command) = command {
        down["commands"] = json!([command]);
    }
    let up = json!({"type": "keyUp", "modifiers": modifiers, "key": down["key"], "code": down["code"], "windowsVirtualKeyCode": vk, "nativeVirtualKeyCode": vk});
    Ok(vec![down, up])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, role: &str, name: &str, backend: Option<i64>, children: &[&str], parent: Option<&str>) -> Value {
        let mut v = json!({"nodeId": id, "ignored": false, "role": {"type": "role", "value": role}, "name": {"type": "computedString", "value": name}, "childIds": children});
        if let Some(b) = backend {
            v["backendDOMNodeId"] = json!(b);
        }
        if let Some(p) = parent {
            v["parentId"] = json!(p);
        }
        v
    }

    fn page() -> Vec<Value> {
        let mut link = node("5", "link", "Pricing", Some(105), &["6"], Some("2"));
        link["properties"] = json!([{"name": "url", "value": {"type": "string", "value": "https://example.com/pricing"}}]);
        let mut field = node("8", "textbox", "Email", Some(108), &[], Some("7"));
        field["value"] = json!({"type": "string", "value": "me@example.com"});
        field["properties"] = json!([{"name": "required", "value": {"type": "booleanOrUndefined", "value": true}}]);
        let mut check = node("9", "checkbox", "Subscribe", Some(109), &[], Some("7"));
        check["properties"] = json!([{"name": "checked", "value": {"type": "tristate", "value": "false"}}]);
        let mut gone = node("11", "button", "Hidden", Some(111), &[], Some("1"));
        gone["ignored"] = json!(true);
        let mut heading = node("3", "heading", "Welcome", Some(103), &["4"], Some("2"));
        heading["properties"] = json!([{"name": "level", "value": {"type": "integer", "value": 1}}]);
        vec![
            node("1", "RootWebArea", "Shop", Some(1), &["2", "7", "10", "11"], None),
            node("2", "navigation", "Main", Some(102), &["3", "5"], Some("1")),
            heading,
            node("4", "StaticText", "Welcome", None, &[], Some("3")),
            link,
            node("6", "StaticText", "Pricing", None, &[], Some("5")),
            node("7", "form", "", Some(107), &["8", "9", "12"], Some("1")),
            field,
            check,
            node("10", "paragraph", "", Some(110), &["13"], Some("1")),
            gone,
            node("12", "button", "Sign up", Some(112), &["14"], Some("7")),
            node("13", "StaticText", "Free for 30 days.", None, &[], Some("10")),
            node("14", "StaticText", "Sign up", None, &[], Some("12")),
        ]
    }

    #[test]
    fn a_page_becomes_compact_text_with_refs_on_what_a_person_can_use() {
        let mut refs = RefMap::default();
        let text = format_snapshot(&page(), &mut refs, &SnapshotOptions { title: "Shop", url: "https://example.com/", max_chars: 10_000 });
        let expected = "Page: Shop\nURL: https://example.com/\n\
- navigation \"Main\"\n  - heading \"Welcome\" [level=1]\n  - link \"Pricing\" url=https://example.com/pricing [ref=1]\n\
- form\n  - textbox \"Email\" value=\"me@example.com\" required [ref=2]\n  - checkbox \"Subscribe\" unchecked [ref=3]\n  - button \"Sign up\" [ref=4]\n\
- text: Free for 30 days.\n\
[4 interactive elements; use their ref with browser_click, browser_type, browser_scroll]";
        // the ignored (hidden) button is left out, and so are the duplicate texts of named elements
        assert_eq!(text, expected, "\n{text}");
        assert_eq!(refs.get(2), Some(&RefTarget { backend: 108, role: "textbox".into(), name: "Email".into() }));
        assert_eq!(refs.get(99), None);
    }

    #[test]
    fn refs_stay_the_same_for_the_same_node_and_restart_with_a_new_document() {
        let mut refs = RefMap::default();
        let a = refs.assign(500, "button", "Go");
        let b = refs.assign(501, "link", "Home");
        assert_eq!((a, b), (1, 2));
        assert_eq!(refs.assign(500, "button", "Go (renamed)"), 1, "same node, same ref");
        assert_eq!(refs.get(1).unwrap().name, "Go (renamed)");
        assert_eq!(refs.assign(777, "tab", "Docs"), 3);
        // a second snapshot of the same page numbers the same elements the same way
        let mut again = RefMap::default();
        let o = SnapshotOptions { title: "", url: "u", max_chars: 10_000 };
        let first = format_snapshot(&page(), &mut again, &o);
        let second = format_snapshot(&page(), &mut again, &o);
        assert_eq!(first, second);
        refs.reset();
        assert_eq!(refs.len(), 0);
        assert_eq!(refs.assign(500, "button", "Go"), 1, "numbering starts again in a new document");
    }

    #[test]
    fn a_long_page_is_cut_with_a_note_and_an_empty_one_says_so() {
        let mut nodes = vec![node("1", "RootWebArea", "Long", Some(1), &[], None)];
        let mut ids: Vec<String> = Vec::new();
        for i in 0..400 {
            let id = format!("n{i}");
            nodes.push(node(&id, "link", &format!("Link number {i}"), Some(1000 + i), &[], Some("1")));
            ids.push(id);
        }
        nodes[0]["childIds"] = json!(ids);
        let mut refs = RefMap::default();
        let text = format_snapshot(&nodes, &mut refs, &SnapshotOptions { title: "Long", url: "https://x/", max_chars: 2000 });
        assert!(text.len() < 2600, "{}", text.len());
        assert!(text.contains("more lines not shown"), "{text}");
        assert!(text.contains("Link number 0") && !text.contains("Link number 399"));
        let empty = format_snapshot(&[], &mut RefMap::default(), &SnapshotOptions { title: "", url: "about:blank", max_chars: 1000 });
        assert!(empty.contains("no content yet"), "{empty}");
        // very long names are clipped
        let long = vec![node("1", "RootWebArea", "", Some(1), &["2"], None), node("2", "button", &"x".repeat(1000), Some(2), &[], Some("1"))];
        let clipped = format_snapshot(&long, &mut RefMap::default(), &SnapshotOptions { title: "", url: "u", max_chars: 5000 });
        assert!(clipped.contains('…') && clipped.len() < 600);
    }

    #[test]
    fn keys_map_to_cdp_events() {
        let enter = key_events("Enter").unwrap();
        assert_eq!(enter[0]["type"], "keyDown");
        assert_eq!(enter[0]["key"], "Enter");
        assert_eq!(enter[0]["windowsVirtualKeyCode"], 13);
        assert_eq!(enter[0]["text"], "\r");
        assert_eq!(enter[1]["type"], "keyUp");
        let arrow = key_events("ArrowDown").unwrap();
        assert_eq!(arrow[0]["type"], "rawKeyDown");
        assert_eq!(arrow[0]["windowsVirtualKeyCode"], 40);
        assert!(arrow[0].get("text").is_none(), "arrow keys type nothing");
        let letter = key_events("a").unwrap();
        assert_eq!((letter[0]["code"].as_str(), letter[0]["text"].as_str()), (Some("KeyA"), Some("a")));
        let select_all = key_events("Control+a").unwrap();
        assert_eq!(select_all[0]["modifiers"], 2);
        assert_eq!(select_all[0]["type"], "rawKeyDown", "a shortcut is not typed text");
        assert_eq!(select_all[0]["commands"], json!(["selectAll"]));
        assert_eq!(key_events("Meta+c").unwrap()[0]["commands"], json!(["copy"]));
        let shift_tab = key_events("Shift+Tab").unwrap();
        assert_eq!((shift_tab[0]["modifiers"].as_u64(), shift_tab[0]["key"].as_str()), (Some(8), Some("Tab")));
        assert_eq!(key_events("Shift+a").unwrap()[0]["text"], "A");
        assert_eq!(key_events("f5").unwrap()[0]["windowsVirtualKeyCode"], 116);
        assert_eq!(key_events("Escape").unwrap()[0]["windowsVirtualKeyCode"], 27);
        assert_eq!(key_events("Space").unwrap()[0]["text"], " ");
        assert_eq!(key_events("+").unwrap()[0]["key"], "+");
        assert_eq!(key_events("Control++").unwrap()[0]["key"], "+");
        assert_eq!(key_events("5").unwrap()[0]["code"], "Digit5");
        for bad in ["", "Hyper+a", "Enterr", "Control+", "F13"] {
            assert!(key_events(bad).is_err(), "{bad:?}");
        }
    }
}
