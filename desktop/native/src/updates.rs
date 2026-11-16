//! Row 47 of PARITY.md: a look for a newer version. The window asks GitHub which release is the
//! newest; `selfupdate.rs` downloads the bundle of the same build as yours (a CUDA app gets the
//! CUDA bundle, a standard app the standard one) and puts it in place.
//!
//! The question goes out when you press *Check for updates*, and at the start while the switch in
//! Settings > Desktop app is on. Nothing but the question is sent.

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
    pub assets: Vec<Asset>,
}

/// A file of a release.
#[derive(Debug, Clone, PartialEq)]
pub struct Asset {
    pub name: String,
    pub url: String,
    pub size: u64,
}

impl Release {
    pub fn asset(&self, name: &str) -> Option<&Asset> {
        self.assets.iter().find(|a| a.name == name)
    }
}

/// The newest release of the list the server of GitHub sent (drafts do not count; a prerelease does, the releases of this program all are).
pub fn newest(list: &Value) -> Option<Release> {
    list.as_array()?
        .iter()
        .filter(|r| !r["draft"].as_bool().unwrap_or(false))
        .filter_map(|r| {
            let tag = r["tag_name"].as_str()?.to_string();
            let assets = r["assets"]
                .as_array()
                .map(|list| {
                    list.iter()
                        .filter_map(|a| Some(Asset { name: a["name"].as_str()?.to_string(), url: a["browser_download_url"].as_str()?.to_string(), size: a["size"].as_u64().unwrap_or(0) }))
                        .collect()
                })
                .unwrap_or_default();
            Some(Release { version: parse_version(&tag)?, url: r["html_url"].as_str()?.to_string(), tag, assets })
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
    /// A newer release is there and has not been fetched.
    Available,
    Downloading { written: u64, total: u64 },
    Verifying,
    Unpacking,
    /// The new version is unpacked and waits for the restart.
    Ready,
    Failed(String),
}

impl State {
    /// The line under the title; `tag` is the release that was found.
    pub fn line(&self, current: &str, tag: &str) -> String {
        let version = tag.trim_start_matches('v');
        match self {
            State::Idle => format!("This is version {current}."),
            State::Checking => "Looking for a newer version ...".into(),
            State::Newest => format!("You have the newest version ({current})."),
            State::Available => format!("Version {version} is there; you have {current}."),
            State::Downloading { written, total } if *total > 0 => format!("Downloading version {version}: {} of {} ({} %)", megabytes(*written), megabytes(*total), written * 100 / total),
            State::Downloading { .. } => format!("Downloading version {version} ..."),
            State::Verifying => format!("Checking the download of version {version} ..."),
            State::Unpacking => format!("Unpacking version {version} ..."),
            State::Ready => format!("Version {version} is ready. Restart to use it; you have {current}."),
            State::Failed(e) => e.clone(),
        }
    }

    /// The look is over: `Available` only for a release that is newer than `current`.
    pub fn of(result: &Result<Option<Release>, String>, current: &str) -> State {
        match result {
            Ok(Some(r)) if is_newer(r, current) => State::Available,
            Ok(_) => State::Newest,
            Err(e) => State::Failed(e.clone()),
        }
    }

    /// A download is on its way.
    pub fn busy(&self) -> bool {
        matches!(self, State::Checking | State::Downloading { .. } | State::Verifying | State::Unpacking)
    }
}

fn megabytes(bytes: u64) -> String {
    format!("{:.1} MB", bytes as f64 / 1_000_000.0)
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
            { "tag_name": "v0.2.2", "html_url": "https://x/0.2.2", "draft": false, "assets": [{ "name": "a.zip", "browser_download_url": "https://x/a.zip", "size": 7 }, { "name": "broken" }] },
            { "tag_name": "v0.10.0", "html_url": "https://x/0.10.0", "draft": false },
            { "tag_name": "v9.9.9", "html_url": "https://x/draft", "draft": true },
            { "tag_name": "nightly", "html_url": "https://x/nightly", "draft": false },
        ]);
        let r = newest(&list).unwrap();
        assert_eq!((r.tag.as_str(), r.url.as_str()), ("v0.10.0", "https://x/0.10.0"), "0.10 is newer than 0.2, a draft does not count");
        assert!(is_newer(&r, "0.2.2") && !is_newer(&r, "0.10.0") && !is_newer(&r, "1.0.0"));
        assert!(newest(&json!([])).is_none() && newest(&json!({})).is_none());
        let old = newest(&json!([{ "tag_name": "v0.2.2", "html_url": "u", "draft": false, "assets": [{ "name": "a.zip", "browser_download_url": "https://x/a.zip", "size": 7 }, { "name": "broken" }] }])).unwrap();
        assert_eq!(old.assets, vec![Asset { name: "a.zip".into(), url: "https://x/a.zip".into(), size: 7 }], "a file with no address is not listed");
        assert!(old.asset("a.zip").is_some() && old.asset("b.zip").is_none());
    }

    #[test]
    fn the_card_says_what_it_found() {
        let r = Release { tag: "v0.3.0".into(), url: "u".into(), version: (0, 3, 0), assets: vec![] };
        assert_eq!(State::of(&Ok(Some(r.clone())), "0.2.2"), State::Available);
        assert_eq!(State::of(&Ok(Some(r.clone())), "0.3.0"), State::Newest, "the same version is the newest");
        assert_eq!(State::of(&Ok(None), "0.2.2"), State::Newest);
        assert_eq!(State::Available.line("0.2.2", "v0.3.0"), "Version 0.3.0 is there; you have 0.2.2.");
        assert_eq!(State::Newest.line("0.2.2", ""), "You have the newest version (0.2.2).");
        assert_eq!(State::of(&Err("Could not look for updates: x".into()), "0.2.2").line("0.2.2", ""), "Could not look for updates: x");
        assert_eq!(State::Downloading { written: 5_000_000, total: 20_000_000 }.line("0.2.2", "v0.3.0"), "Downloading version 0.3.0: 5.0 MB of 20.0 MB (25 %)");
        assert_eq!(State::Ready.line("0.2.2", "v0.3.0"), "Version 0.3.0 is ready. Restart to use it; you have 0.2.2.");
        assert!(State::Verifying.busy() && !State::Ready.busy() && !State::Available.busy());
    }
}
