//! Multi-agent routing for the messaging gateway: different chats get different agents. A route file maps a chat to a profile
//! (its own persona `SOUL.md` and its own sessions), a workspace folder, a set of permissions, a model and a system text:
//!
//! ```json
//! {"routes": [
//!   {"chat": "123456789", "profile": "home", "folder": "/srv/notes", "allow": ["edit"]},
//!   {"chat": "-1001234567", "profile": "work", "folder": "/srv/app", "allow": ["edit", "command"], "system": "Answer in German."}
//! ]}
//! ```
//!
//! A chat that has a route is allowed (the file is written by the owner on this machine); a chat without one uses the defaults
//! of the command line, if it is in `--chat`. Reading is always allowed; `allow` names the other categories (`edit`, `command`,
//! `network`, `memory`, or `all`). Nothing a chat message says can change its route.
use crate::agent_cli::{self, LoopConfig};
use serde_json::Value;
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

pub const FILE_NAME: &str = "gateway-routes.json";
const MAX_ROUTES: usize = 100;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct RouteSpec {
    pub chat: String,
    pub profile: Option<String>,
    pub folder: Option<String>,
    pub allow: Option<Vec<String>>,
    pub system: Option<String>,
    pub model: Option<String>,
}

/// What a routed chat runs with.
#[derive(Clone)]
pub struct Route {
    pub profile: Option<String>,
    pub folder: String,
    pub sessions_dir: PathBuf,
    pub soul: Option<String>,
    pub loop_cfg: LoopConfig,
    pub system_extra: Option<String>,
}

pub fn parse_routes(text: &str) -> Result<Vec<RouteSpec>, String> {
    let value: Value = serde_json::from_str(text).map_err(|e| format!("the routes file is not valid JSON: {e}"))?;
    let list = value.get("routes").unwrap_or(&value).as_array().ok_or("the routes file needs a list of routes")?;
    if list.len() > MAX_ROUTES {
        return Err(format!("at most {MAX_ROUTES} routes are allowed"));
    }
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    for item in list {
        let chat = item["chat"].as_str().map(str::to_owned).or_else(|| item["chat"].as_i64().map(|n| n.to_string())).filter(|c| !c.trim().is_empty() && !c.chars().any(char::is_whitespace)).ok_or("every route needs a \"chat\" (the chat or room id)")?;
        if !seen.insert(chat.clone()) {
            return Err(format!("chat {chat} has two routes"));
        }
        let allow = item.get("allow").map(|a| a.as_array().map(|l| l.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect::<Vec<_>>()).ok_or("\"allow\" must be a list")).transpose()?;
        let text = |k: &str| item[k].as_str().map(str::trim).filter(|s| !s.is_empty()).map(str::to_owned);
        out.push(RouteSpec { chat, profile: text("profile"), folder: text("folder"), allow, system: text("system"), model: text("model") });
    }
    Ok(out)
}

pub fn allow_set(list: &[String]) -> Result<BTreeSet<String>, String> {
    let mut set = BTreeSet::from(["read".to_owned()]);
    for c in list {
        match c.as_str() {
            "all" => set.extend(["edit", "command", "network", "memory"].map(str::to_owned)),
            "read" | "edit" | "command" | "network" | "memory" => {
                set.insert(c.clone());
            }
            other => return Err(format!("route: unknown permission '{other}' (edit, command, network, memory, all)")),
        }
    }
    Ok(set)
}

/// Turns a spec into a ready route; the defaults come from the gateway's own command line.
pub fn build(spec: &RouteSpec, default_cfg: &LoopConfig, default_folder: &str, default_extra: Option<&str>) -> Result<Route, String> {
    let folder_path = PathBuf::from(spec.folder.as_deref().unwrap_or(default_folder));
    let folder = folder_path.canonicalize().map(|p| agent_cli::plain_path(&p)).map_err(|e| format!("route {}: the folder {} cannot be used: {e}", spec.chat, folder_path.display()))?;
    if !folder.is_dir() {
        return Err(format!("route {}: {} is not a folder", spec.chat, folder.display()));
    }
    let base = agent_cli::profile_base(spec.profile.as_deref())?;
    let mut cfg = default_cfg.clone();
    if let Some(list) = &spec.allow {
        cfg.allow = allow_set(list)?;
    }
    if let Some(model) = &spec.model {
        cfg.model = Some(model.clone());
    }
    cfg.rules = crate::agent_rules::load_hierarchy(&agent_cli::studio_folder(), &folder)?;
    let extra = match (spec.system.as_deref(), default_extra) {
        (Some(s), Some(d)) => Some(format!("{d}\n\n{s}")),
        (Some(s), None) => Some(s.to_owned()),
        (None, d) => d.map(str::to_owned),
    };
    Ok(Route { profile: spec.profile.clone(), folder: folder.display().to_string(), sessions_dir: base.clone(), soul: agent_cli::load_soul(&base), loop_cfg: cfg, system_extra: extra })
}

pub fn build_all(specs: &[RouteSpec], default_cfg: &LoopConfig, default_folder: &str, default_extra: Option<&str>) -> Result<HashMap<String, Route>, String> {
    specs.iter().map(|s| build(s, default_cfg, default_folder, default_extra).map(|r| (s.chat.clone(), r))).collect()
}

/// The routes file to use: the one named by `--routes`, else `gateway-routes.json` of the profile folder when it exists.
pub fn file_for(explicit: Option<&Path>, base: &Path) -> Option<PathBuf> {
    match explicit {
        Some(p) => Some(p.to_path_buf()),
        None => Some(base.join(FILE_NAME)).filter(|p| p.is_file()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_are_parsed_and_checked() {
        let specs = parse_routes(r#"{"routes": [{"chat": "-100", "profile": "work", "folder": "/srv/app", "allow": ["edit", "command"], "system": " German "}, {"chat": 7}]}"#).unwrap();
        assert_eq!(specs.len(), 2);
        assert_eq!((specs[0].chat.as_str(), specs[0].profile.as_deref(), specs[0].system.as_deref()), ("-100", Some("work"), Some("German")));
        assert_eq!(specs[1].chat, "7");
        assert_eq!(specs[1].allow, None);
        assert!(parse_routes(r#"[{"chat": "1"}, {"chat": "1"}]"#).unwrap_err().contains("two routes"));
        assert!(parse_routes(r#"[{"profile": "x"}]"#).is_err());
        assert!(parse_routes("nonsense").is_err());
        assert!(parse_routes(r#"[{"chat": "1", "allow": "edit"}]"#).is_err());
    }

    #[test]
    fn permissions_start_at_read_and_only_known_names_widen_them() {
        assert_eq!(allow_set(&[]).unwrap().len(), 1);
        assert!(allow_set(&["edit".into()]).unwrap().contains("edit"));
        assert_eq!(allow_set(&["all".into()]).unwrap().len(), 5);
        assert!(allow_set(&["root".into()]).is_err());
    }

    #[test]
    fn a_route_takes_its_folder_permissions_and_extra_text() {
        let dir = std::env::temp_dir().join(format!("fllm-route-{}-{}", std::process::id(), crate::agent_extras::fresh_session_name(5)));
        std::fs::create_dir_all(&dir).unwrap();
        let default = LoopConfig { model: Some("m".into()), allow: BTreeSet::from(["read".to_owned()]), rules: Default::default(), max_steps: 5, max_tokens: 100, temperature: 0.0 };
        let spec = RouteSpec { chat: "1".into(), folder: Some(dir.display().to_string()), allow: Some(vec!["edit".into()]), system: Some("Be brief.".into()), ..RouteSpec::default() };
        let route = build(&spec, &default, ".", Some("Base.")).unwrap();
        assert!(route.loop_cfg.allow.contains("edit") && route.loop_cfg.allow.contains("read"));
        assert_eq!(route.system_extra.as_deref(), Some("Base.\n\nBe brief."));
        assert!(route.folder.ends_with(dir.file_name().unwrap().to_str().unwrap()));
        let missing = RouteSpec { chat: "2".into(), folder: Some(dir.join("nope").display().to_string()), ..RouteSpec::default() };
        assert!(build(&missing, &default, ".", None).is_err_and(|e| e.contains("cannot be used")));
        let _ = std::fs::remove_dir_all(dir);
    }
}
