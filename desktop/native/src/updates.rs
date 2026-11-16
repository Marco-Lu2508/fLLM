//! Row 47 of PARITY.md: a look for a newer version. The Tauri shell had an updater; this window
//! asks GitHub which release is the newest and offers its page. It does not replace the program
//! itself: that needs a signature of the bundles that the release does not carry yet.
//!
//! The question goes out only when you press *Check for updates*, or at the start when you have
//! switched that on in Settings > Desktop app. Nothing but the question is sent.

use serde_json::Value;

/// Where the releases are.
pub const RELEASES_URL: &str = "https://api.github.com/repos/Marco-Lu2508/fLLM/releases?per_page=30";

/// The three numbers of a version in a tag (`v0.2.2`, `fllm-0.3.0-rc1`): the first group `a.b.c`.
pub fn parse_version(tag: &str) -> Option<(u64, u64, u64)> {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re = RE.get_or_init(|| regex::Regex::new(r"(\d+)\.(\d+)\.(\d+)").expect("version"));
    let c = re.captures(tag)?;
    Some((c[1].parse().ok()?, c[2].parse().ok()?, c[3].parse().ok()?))
}

#[derive(Debug, Clone, PartialEq)]
pub struct Release {
    pub tag: String,
    pub url: String,
    pub version: (u64, u64, u64),
}

/// The newest release of the list the server of GitHub sent (drafts do not count; a prerelease does, the releases of this program all are).
pub fn newest(list: &Value) -> Option<Release> {
    list.as_array()?
        .iter()
        .filter(|r| !r["draft"].as_bool().unwrap_or(false))
        .filter_map(|r| {
            let tag = r["tag_name"].as_str()?.to_string();
            Some(Release { version: parse_version(&tag)?, url: r["html_url"].as_str()?.to_string(), tag })
        })
        .max_by_key(|r| r.version)
}

/// Whether `release` is newer than the version this program has.
pub fn is_newer(release: &Release, current: &str) -> bool {
    parse_version(current).is_some_and(|c| release.version > c)
}

pub async fn check(url: &str) -> Result<Option<Release>, String> {
    let resp = reqwest::Client::new()
        .get(url)
        .header("User-Agent", concat!("fllm-app/", env!("CARGO_PKG_VERSION")))
        .header("Accept", "application/vnd.github+json")
        .timeout(std::time::Duration::from_secs(15))
        .send()
        .await
        .map_err(|e| format!("Could not look for updates: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("Could not look for updates: GitHub answered {}", resp.status().as_u16()));
    }
    let list: Value = resp.json().await.map_err(|e| format!("Could not look for updates: {e}"))?;
    Ok(newest(&list))
}

/// What the card says.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum State {
    #[default]
    Idle,
    Checking,
    Newest,
    Available(Release),
    Failed(String),
}

impl State {
    pub fn line(&self, current: &str) -> String {
        match self {
            State::Idle => format!("This is version {current}."),
            State::Checking => "Looking for a newer version ...".into(),
            State::Newest => format!("You have the newest version ({current})."),
            State::Available(r) => format!("Version {} is there; you have {current}.", r.tag.trim_start_matches('v')),
            State::Failed(e) => e.clone(),
        }
    }

    pub fn of(result: Result<Option<Release>, String>, current: &str) -> State {
        match result {
            Ok(Some(r)) if is_newer(&r, current) => State::Available(r),
            Ok(_) => State::Newest,
            Err(e) => State::Failed(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn versions_are_read_from_tags() {
        assert_eq!(parse_version("v0.2.2"), Some((0, 2, 2)));
        assert_eq!(parse_version("fllm-1.10.3-rc1"), Some((1, 10, 3)));
        assert_eq!(parse_version("nightly"), None);
        assert_eq!(parse_version("v1.2"), None);
    }

    #[test]
    fn the_newest_release_wins_by_number_and_not_by_the_order_of_the_list() {
        let list = json!([
            { "tag_name": "v0.2.2", "html_url": "https://x/0.2.2", "draft": false },
            { "tag_name": "v0.10.0", "html_url": "https://x/0.10.0", "draft": false },
            { "tag_name": "v9.9.9", "html_url": "https://x/draft", "draft": true },
            { "tag_name": "nightly", "html_url": "https://x/nightly", "draft": false },
        ]);
        let r = newest(&list).unwrap();
        assert_eq!((r.tag.as_str(), r.url.as_str()), ("v0.10.0", "https://x/0.10.0"), "0.10 is newer than 0.2, a draft does not count");
        assert!(is_newer(&r, "0.2.2") && !is_newer(&r, "0.10.0") && !is_newer(&r, "1.0.0"));
        assert!(newest(&json!([])).is_none() && newest(&json!({})).is_none());
    }

    #[test]
    fn the_card_says_what_it_found() {
        let r = Release { tag: "v0.3.0".into(), url: "u".into(), version: (0, 3, 0) };
        assert_eq!(State::of(Ok(Some(r.clone())), "0.2.2"), State::Available(r.clone()));
        assert_eq!(State::of(Ok(Some(r.clone())), "0.3.0"), State::Newest, "the same version is the newest");
        assert_eq!(State::of(Ok(None), "0.2.2"), State::Newest);
        assert_eq!(State::Available(r).line("0.2.2"), "Version 0.3.0 is there; you have 0.2.2.");
        assert_eq!(State::Newest.line("0.2.2"), "You have the newest version (0.2.2).");
        assert_eq!(State::of(Err("Could not look for updates: x".into()), "0.2.2").line("0.2.2"), "Could not look for updates: x");
    }
}
