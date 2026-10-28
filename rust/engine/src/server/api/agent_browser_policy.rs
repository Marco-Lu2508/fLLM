//! Rules of the in-app browser that need no running browser: the settings, which URLs the agent may open, where a
//! Chromium-family browser lives on this machine and with which flags it is started. Everything here is portable
//! and unit-tested; the session that uses it is in `agent_browser`.
//!
//! The settings live in `browser.json` next to `studio.json`:
//! `{"enabled": false, "show_window": false, "local_addresses": "auto", "executable": "", "persistent_profile": true}`.
use serde_json::{json, Value};
use std::{
    net::{IpAddr, Ipv4Addr, Ipv6Addr},
    path::{Path, PathBuf},
};

// ---- settings ---------------------------------------------------------------------------------------------------

/// Which local addresses a page may be loaded from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LocalAddresses {
    /// `localhost`, `127.0.0.0/8` and `::1`, but only while a workspace is open (testing a dev server is the main use).
    Auto,
    /// No local address at all.
    Off,
    /// Loopback and the private ranges of the local network (10.x, 172.16/12, 192.168.x, unique-local IPv6).
    All,
}

impl LocalAddresses {
    pub(super) fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "auto" | "workspace" => Some(Self::Auto),
            "off" | "none" => Some(Self::Off),
            "all" | "on" => Some(Self::All),
            _ => None,
        }
    }
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Off => "off",
            Self::All => "all",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Settings {
    /// The agent may use the browser at all (default off).
    pub enabled: bool,
    /// Run the browser with a visible window (to log in to a site once); default off (headless).
    pub show_window: bool,
    pub local: LocalAddresses,
    /// Path or name of the browser; empty: look for one.
    pub executable: String,
    /// Keep cookies between sessions in the profile folder (default on); off: a throw-away profile per session.
    pub persistent_profile: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self { enabled: false, show_window: false, local: LocalAddresses::Auto, executable: String::new(), persistent_profile: true }
    }
}

impl Settings {
    pub(super) fn from_json(value: &Value) -> Self {
        let d = Self::default();
        Self {
            enabled: value.get("enabled").and_then(Value::as_bool).unwrap_or(d.enabled),
            show_window: value.get("show_window").and_then(Value::as_bool).unwrap_or(d.show_window),
            local: value.get("local_addresses").and_then(Value::as_str).and_then(LocalAddresses::parse).unwrap_or(d.local),
            executable: value.get("executable").and_then(Value::as_str).unwrap_or_default().trim().to_owned(),
            persistent_profile: value.get("persistent_profile").and_then(Value::as_bool).unwrap_or(d.persistent_profile),
        }
    }
    pub(super) fn to_json(&self) -> Value {
        json!({
            "enabled": self.enabled,
            "show_window": self.show_window,
            "local_addresses": self.local.name(),
            "executable": self.executable,
            "persistent_profile": self.persistent_profile,
        })
    }
    /// The settings with the changes of a request applied; `Err` names a value that is not accepted.
    pub(super) fn updated(&self, body: &Value) -> Result<Self, String> {
        let mut next = self.clone();
        if let Some(v) = body.get("enabled").and_then(Value::as_bool) {
            next.enabled = v;
        }
        if let Some(v) = body.get("show_window").and_then(Value::as_bool) {
            next.show_window = v;
        }
        if let Some(v) = body.get("persistent_profile").and_then(Value::as_bool) {
            next.persistent_profile = v;
        }
        if let Some(v) = body.get("local_addresses") {
            next.local = v.as_str().and_then(LocalAddresses::parse).ok_or("local_addresses has to be auto, off or all")?;
        }
        if let Some(v) = body.get("executable").and_then(Value::as_str) {
            if v.len() > 1024 || v.contains('\0') {
                return Err("executable is not a usable path".into());
            }
            next.executable = v.trim().to_owned();
        }
        Ok(next)
    }
}

fn settings_file() -> PathBuf {
    crate::hub::studio_data_file("browser.json")
}

pub(super) fn read_settings() -> Settings {
    std::fs::read_to_string(settings_file())
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .map(|v| Settings::from_json(&v))
        .unwrap_or_default()
}

pub(super) fn write_settings(settings: &Settings) -> Result<(), String> {
    let path = settings_file();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    let text = serde_json::to_string_pretty(&settings.to_json()).unwrap_or_else(|_| "{}".into());
    std::fs::write(&path, text).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

/// The folder with the browser profile (cookies, local storage) of fLLM. Never the profile of the user's own browser.
pub(super) fn profile_dir() -> PathBuf {
    crate::hub::studio_data_file("browser-profile")
}

// ---- URL policy -------------------------------------------------------------------------------------------------

/// Where an address leads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Reach {
    Public,
    /// This machine (`localhost`, 127.0.0.0/8, ::1).
    Loopback,
    /// The local network (RFC 1918, carrier-grade NAT, unique-local IPv6).
    Private,
    /// Never reachable: link-local (cloud metadata is 169.254.169.254), unspecified, multicast, broadcast, reserved.
    Forbidden,
}

pub(super) fn reach_of_ip(ip: IpAddr) -> Reach {
    match ip {
        IpAddr::V4(v4) => reach_v4(v4),
        IpAddr::V6(v6) => match v6.to_ipv4_mapped() {
            Some(v4) => reach_v4(v4),
            None => reach_v6(v6),
        },
    }
}

fn reach_v4(ip: Ipv4Addr) -> Reach {
    let [a, b, ..] = ip.octets();
    if ip.is_loopback() {
        Reach::Loopback
    } else if ip.is_link_local() || ip.is_unspecified() || ip.is_broadcast() || ip.is_multicast() || a == 0 || a >= 240 || (a == 100 && b == 100) {
        // 100.100.100.200 is the metadata address of Alibaba Cloud, inside the carrier-grade NAT range
        Reach::Forbidden
    } else if crate::server::remote_image::is_blocked(IpAddr::V4(ip)) {
        Reach::Private
    } else {
        Reach::Public
    }
}

fn reach_v6(ip: Ipv6Addr) -> Reach {
    let s = ip.segments();
    if ip.is_loopback() {
        Reach::Loopback
    } else if ip.is_unspecified() || ip.is_multicast() || (s[0] & 0xffc0) == 0xfe80 || ip == Ipv6Addr::new(0xfd00, 0x0ec2, 0, 0, 0, 0, 0, 0x0254) {
        Reach::Forbidden
    } else if crate::server::remote_image::is_blocked(IpAddr::V6(ip)) {
        // NAT64 and 6to4 addresses that embed a private IPv4 address are blocked by the shared check; none of them is local
        if (s[0] & 0xfe00) == 0xfc00 {
            Reach::Private
        } else {
            Reach::Forbidden
        }
    } else {
        Reach::Public
    }
}

/// The URL policy for one navigation or request.
#[derive(Clone, Copy, Debug)]
pub(super) struct UrlPolicy {
    pub local: LocalAddresses,
    /// A workspace is open on the Code page.
    pub workspace: bool,
}

impl UrlPolicy {
    /// Whether a destination of this kind may be loaded.
    pub(super) fn allows(&self, reach: Reach) -> bool {
        match reach {
            Reach::Public => true,
            Reach::Loopback => match self.local {
                LocalAddresses::All => true,
                LocalAddresses::Auto => self.workspace,
                LocalAddresses::Off => false,
            },
            Reach::Private => self.local == LocalAddresses::All,
            Reach::Forbidden => false,
        }
    }

    fn refusal(&self, host: &str, reach: Reach) -> String {
        match reach {
            Reach::Loopback => match self.local {
                LocalAddresses::Off => format!("refused: '{host}' is this machine and local addresses are switched off (Settings > Browser)"),
                _ => format!("refused: '{host}' is this machine; local addresses are allowed only while a workspace is open on the Code page, or when Settings > Browser allows all local addresses"),
            },
            Reach::Private => format!("refused: '{host}' is on the local network; Settings > Browser > Local addresses has to be set to 'all' to allow it"),
            _ => format!("refused: '{host}' is a link-local, metadata or reserved address and is never reachable from the browser tools"),
        }
    }
}

/// What the syntax of a URL allows, before any address is looked at: only http and https, no credentials, a host.
/// `about:blank` is accepted as the empty page.
pub(super) fn parse_url(text: &str) -> Result<reqwest::Url, String> {
    let text = text.trim();
    if text.is_empty() {
        return Err("the URL is empty".into());
    }
    let url = reqwest::Url::parse(text).map_err(|e| format!("invalid URL '{text}': {e}"))?;
    match url.scheme() {
        "http" | "https" => {}
        "about" if url.as_str() == "about:blank" => return Ok(url),
        other => return Err(format!("refused: the scheme '{other}:' is not allowed, only http and https pages can be opened")),
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err("refused: URLs with a user name or password are not allowed".into());
    }
    if url.host().is_none() {
        return Err("the URL has no host".into());
    }
    Ok(url)
}

/// What can be decided from the URL alone: `Some(reach)` for an IP address or a `localhost` name, `None` for a name
/// that has to be resolved first.
pub(super) fn reach_by_name(url: &reqwest::Url) -> Option<Reach> {
    // the URL parser has already turned 2130706433, 0x7f.1 and the like into dotted addresses
    let host = url.host_str()?.trim_matches(['[', ']']);
    if let Ok(ip) = host.parse::<IpAddr>() {
        return Some(reach_of_ip(ip));
    }
    let name = host.trim_end_matches('.').to_ascii_lowercase();
    (name == "localhost" || name.ends_with(".localhost")).then_some(Reach::Loopback)
}

/// The least reachable of several answers of the resolver (a name with one forbidden address is forbidden).
pub(super) fn worst_reach(addresses: &[IpAddr]) -> Reach {
    let rank = |r: Reach| match r {
        Reach::Public => 0,
        Reach::Loopback => 1,
        Reach::Private => 2,
        Reach::Forbidden => 3,
    };
    addresses.iter().map(|ip| reach_of_ip(*ip)).max_by_key(|r| rank(*r)).unwrap_or(Reach::Forbidden)
}

/// The whole check for a navigation: syntax, then the address (resolved through DNS for a name, and checked again for
/// every answer so that a name pointing at an internal address is refused).
pub(super) async fn check_url(text: &str, policy: UrlPolicy) -> Result<reqwest::Url, String> {
    let url = parse_url(text)?;
    if url.scheme() == "about" {
        return Ok(url);
    }
    let host = url.host_str().unwrap_or_default().to_owned();
    let reach = match reach_by_name(&url) {
        Some(reach) => reach,
        None => {
            let port = url.port_or_known_default().unwrap_or(80);
            let found: Vec<IpAddr> = tokio::net::lookup_host((host.as_str(), port))
                .await
                .map_err(|e| format!("cannot resolve '{host}': {e}"))?
                .map(|a| a.ip())
                .collect();
            if found.is_empty() {
                return Err(format!("'{host}' has no address"));
            }
            worst_reach(&found)
        }
    };
    if policy.allows(reach) {
        Ok(url)
    } else {
        Err(policy.refusal(&host, reach))
    }
}

/// Whether a page at this URL counts as local, which decides if it may load local resources (a public page must not
/// reach into the local machine through the browser).
pub(super) fn is_local_page(url: &str) -> bool {
    match parse_url(url) {
        Ok(u) if u.scheme() == "about" => true,
        Ok(u) => matches!(reach_by_name(&u), Some(Reach::Loopback | Reach::Private)),
        Err(_) => false,
    }
}

// ---- finding the browser ----------------------------------------------------------------------------------------

/// The places a browser is installed to on this system, best first. The inputs are passed in so that every system
/// can be tested on every machine.
pub(super) fn candidate_paths(os: &str, program_files: &[String], local_app_data: &str, home: &str) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    match os {
        "windows" => {
            let apps: [(&str, &str); 4] = [
                ("Google", "Chrome\\Application\\chrome.exe"),
                ("Microsoft", "Edge\\Application\\msedge.exe"),
                ("BraveSoftware", "Brave-Browser\\Application\\brave.exe"),
                ("Chromium", "Application\\chrome.exe"),
            ];
            let mut roots: Vec<&str> = program_files.iter().map(String::as_str).filter(|s| !s.is_empty()).collect();
            if !local_app_data.is_empty() {
                roots.push(local_app_data);
            }
            for (vendor, rest) in apps {
                for root in &roots {
                    out.push(Path::new(root).join(vendor).join(rest));
                }
            }
        }
        "macos" => {
            let apps = [
                "Google Chrome.app/Contents/MacOS/Google Chrome",
                "Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
                "Brave Browser.app/Contents/MacOS/Brave Browser",
                "Chromium.app/Contents/MacOS/Chromium",
            ];
            for app in apps {
                out.push(Path::new("/Applications").join(app));
                if !home.is_empty() {
                    out.push(Path::new(home).join("Applications").join(app));
                }
            }
        }
        _ => {
            for dir in ["/usr/bin", "/usr/local/bin", "/snap/bin", "/opt/google/chrome", "/opt/microsoft/msedge", "/opt/brave.com/brave"] {
                for name in linux_names() {
                    out.push(Path::new(dir).join(name));
                }
            }
        }
    }
    out
}

/// Executable names searched on the PATH of Linux and other Unix systems.
pub(super) fn linux_names() -> &'static [&'static str] {
    &["google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge", "microsoft-edge-stable", "brave-browser", "brave", "chrome"]
}

fn on_path(name: &str) -> Option<PathBuf> {
    let paths = std::env::var_os("PATH")?;
    let exts: Vec<&str> = if cfg!(windows) { vec!["", ".exe"] } else { vec![""] };
    for dir in std::env::split_paths(&paths) {
        for ext in &exts {
            let file = dir.join(format!("{name}{ext}"));
            if file.is_file() {
                return Some(file);
            }
        }
    }
    None
}

/// The browser to start: `FLLM_BROWSER`, then the setting, then the usual places, then the PATH.
pub(super) fn find_browser(settings: &Settings) -> Result<PathBuf, String> {
    let named = |what: &str, value: &str| -> Result<PathBuf, String> {
        let direct = PathBuf::from(value);
        if direct.is_file() {
            return Ok(direct);
        }
        on_path(value).ok_or_else(|| format!("{what} names '{value}', which is not a file and not on the PATH"))
    };
    if let Ok(value) = std::env::var("FLLM_BROWSER") {
        if !value.trim().is_empty() {
            return named("FLLM_BROWSER", value.trim());
        }
    }
    if !settings.executable.is_empty() {
        return named("The browser setting", &settings.executable);
    }
    let program_files: Vec<String> = ["ProgramFiles", "ProgramFiles(x86)"].iter().filter_map(|k| std::env::var(k).ok()).collect();
    let local = std::env::var("LOCALAPPDATA").unwrap_or_default();
    let home = std::env::var("HOME").unwrap_or_default();
    for path in candidate_paths(std::env::consts::OS, &program_files, &local, &home) {
        if path.is_file() {
            return Ok(path);
        }
    }
    for name in linux_names() {
        if let Some(path) = on_path(name) {
            return Ok(path);
        }
    }
    Err(no_browser_message())
}

pub(super) fn no_browser_message() -> String {
    "no Chromium-based browser was found on this machine. Install Google Chrome, Chromium, Microsoft Edge or Brave \
(Windows has Edge already; on Debian or Ubuntu: apt install chromium; in Docker: add chromium to the image), \
or set FLLM_BROWSER (or Settings > Browser > Browser program) to its path."
        .to_owned()
}

/// The command line of the browser. `port_zero` lets the browser pick a free debugging port (it reports it in
/// `DevToolsActivePort` inside the profile folder). `no_sandbox` is needed when running as root, as in a container.
pub(super) fn launch_args(profile: &Path, show_window: bool, no_sandbox: bool, width: u32, height: u32) -> Vec<String> {
    let mut args = vec![
        format!("--user-data-dir={}", profile.display()),
        "--remote-debugging-port=0".to_owned(),
        "--remote-debugging-address=127.0.0.1".to_owned(),
        "--no-first-run".to_owned(),
        "--no-default-browser-check".to_owned(),
        "--disable-extensions".to_owned(),
        "--disable-sync".to_owned(),
        "--disable-background-networking".to_owned(),
        "--disable-component-update".to_owned(),
        "--disable-default-apps".to_owned(),
        "--disable-popup-blocking".to_owned(),
        "--disable-features=Translate,MediaRouter,OptimizationHints".to_owned(),
        "--password-store=basic".to_owned(),
        "--use-mock-keychain".to_owned(),
        "--deny-permission-prompts".to_owned(),
        "--hide-crash-restore-bubble".to_owned(),
        format!("--window-size={width},{height}"),
    ];
    if !show_window {
        args.push("--headless=new".to_owned());
        args.push("--hide-scrollbars".to_owned());
        args.push("--mute-audio".to_owned());
    }
    if no_sandbox {
        args.push("--no-sandbox".to_owned());
        args.push("--disable-dev-shm-usage".to_owned());
    }
    args.push("about:blank".to_owned());
    args
}

/// The port and the path of the browser WebSocket from the `DevToolsActivePort` file (line 1: the port, line 2: the path).
pub(super) fn parse_active_port(text: &str) -> Option<(u16, String)> {
    let mut lines = text.lines();
    let port: u16 = lines.next()?.trim().parse().ok().filter(|p| *p != 0)?;
    let path = lines.next()?.trim().to_owned();
    path.starts_with('/').then_some((port, path))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(local: LocalAddresses, workspace: bool) -> UrlPolicy {
        UrlPolicy { local, workspace }
    }

    fn verdict(url: &str, p: UrlPolicy) -> Result<(), String> {
        let parsed = parse_url(url)?;
        match reach_by_name(&parsed) {
            Some(reach) if p.allows(reach) => Ok(()),
            Some(reach) => Err(p.refusal(parsed.host_str().unwrap_or_default(), reach)),
            None => Ok(()), // a public-looking name: decided after resolving it
        }
    }

    #[test]
    fn only_http_and_https_pages_can_be_opened() {
        let open = policy(LocalAddresses::All, true);
        for ok in ["http://example.com/", "https://example.com/a?b=1#c", "HTTPS://Example.com"] {
            assert!(verdict(ok, open).is_ok(), "{ok}");
        }
        for bad in [
            "file:///etc/passwd", "file:///C:/Windows/win.ini", "chrome://settings", "chrome-extension://abc/x.html", "devtools://devtools/bundled/inspector.html",
            "view-source:https://example.com", "javascript:alert(1)", "data:text/html,<h1>x</h1>", "ftp://example.com/x", "blob:https://example.com/uuid", "about:config",
            "ws://127.0.0.1:9222/", "", "   ", "not a url", "http://",
        ] {
            assert!(verdict(bad, open).is_err(), "{bad} must be refused");
        }
        assert!(parse_url("about:blank").is_ok(), "the empty page is internal");
        assert!(parse_url("http://user:pw@example.com/").unwrap_err().contains("user name"));
        assert!(parse_url("http://user@example.com/").is_err());
    }

    #[test]
    fn local_addresses_follow_the_setting_and_the_workspace() {
        let none = policy(LocalAddresses::Auto, false);
        let with = policy(LocalAddresses::Auto, true);
        for url in ["http://localhost:3000/", "http://127.0.0.1:8080/x", "http://[::1]:5173/", "http://app.localhost/", "http://127.1/", "http://2130706433/", "http://0x7f.0.0.1/"] {
            assert!(verdict(url, none).is_err(), "{url} without a workspace");
            assert!(verdict(url, with).is_ok(), "{url} with a workspace");
            assert!(verdict(url, policy(LocalAddresses::Off, true)).is_err(), "{url} with local addresses off");
            assert!(verdict(url, policy(LocalAddresses::All, false)).is_ok(), "{url} with all");
        }
    }

    #[test]
    fn the_local_network_needs_the_all_setting_and_metadata_is_never_reachable() {
        let auto = policy(LocalAddresses::Auto, true);
        let all = policy(LocalAddresses::All, true);
        for url in ["http://10.0.0.5/", "http://192.168.1.1/", "http://172.16.3.4/", "http://172.31.255.1/", "http://100.64.1.1/", "http://[fd12:3456::1]/", "http://[::ffff:10.0.0.1]/", "http://3232235777/"] {
            assert!(verdict(url, auto).is_err(), "{url} must be refused by default");
            assert!(verdict(url, all).is_ok(), "{url} is allowed by 'all'");
        }
        for url in [
            "http://169.254.169.254/latest/meta-data/", "http://[fe80::1]/", "http://[fd00:ec2::254]/", "http://100.100.100.200/", "http://0.0.0.0/", "http://224.0.0.1/",
            "http://255.255.255.255/", "http://[::]/", "http://[::ffff:169.254.169.254]/", "http://0/", "http://0xa9fea9fe/",
        ] {
            assert!(verdict(url, all).is_err(), "{url} must never be reachable");
        }
        for url in ["http://8.8.8.8/", "https://93.184.216.34/", "http://[2606:4700:4700::1111]/", "http://172.32.0.1/", "http://[::ffff:8.8.8.8]/"] {
            assert!(verdict(url, policy(LocalAddresses::Off, false)).is_ok(), "{url} is public");
        }
    }

    #[test]
    fn a_name_is_judged_by_the_worst_of_its_addresses() {
        let a: IpAddr = "93.184.216.34".parse().unwrap();
        let lo: IpAddr = "127.0.0.1".parse().unwrap();
        let meta: IpAddr = "169.254.169.254".parse().unwrap();
        assert_eq!(worst_reach(&[a]), Reach::Public);
        assert_eq!(worst_reach(&[a, lo]), Reach::Loopback, "a rebinding answer is judged by its loopback address");
        assert_eq!(worst_reach(&[lo, a, meta]), Reach::Forbidden);
        assert_eq!(worst_reach(&[]), Reach::Forbidden);
    }

    #[tokio::test]
    async fn check_url_resolves_names_and_refuses_what_they_point_at() {
        // `localtest.me`-style names cannot be used offline; a literal and a localhost name cover the async path
        let strict = policy(LocalAddresses::Off, true);
        assert!(check_url("http://127.0.0.1:1/", strict).await.is_err());
        assert!(check_url("http://localhost/", strict).await.is_err());
        assert!(check_url("http://localhost:1/", policy(LocalAddresses::Auto, true)).await.is_ok());
        assert!(check_url("file:///etc/passwd", policy(LocalAddresses::All, true)).await.is_err());
        let blocked = check_url("http://169.254.169.254/", policy(LocalAddresses::All, true)).await.unwrap_err();
        assert!(blocked.contains("never reachable"), "{blocked}");
        assert!(check_url("about:blank", strict).await.is_ok());
    }

    #[test]
    fn a_public_page_is_not_local_and_a_local_one_is() {
        assert!(is_local_page("http://localhost:3000/"));
        assert!(is_local_page("http://192.168.1.2/"));
        assert!(is_local_page("about:blank"));
        assert!(!is_local_page("https://example.com/"));
        assert!(!is_local_page("file:///etc/passwd"));
    }

    #[test]
    fn settings_default_to_off_and_reject_unknown_values() {
        let d = Settings::default();
        assert!(!d.enabled && !d.show_window);
        assert_eq!(d.local, LocalAddresses::Auto);
        assert!(d.persistent_profile);
        assert_eq!(Settings::from_json(&json!({})), d);
        assert_eq!(Settings::from_json(&d.to_json()), d);
        let next = d.updated(&json!({"enabled": true, "local_addresses": "all", "executable": "  /usr/bin/chromium "})).unwrap();
        assert!(next.enabled);
        assert_eq!(next.local, LocalAddresses::All);
        assert_eq!(next.executable, "/usr/bin/chromium");
        assert!(d.updated(&json!({"local_addresses": "everything"})).is_err());
        assert!(d.updated(&json!({"local_addresses": 5})).is_err());
        assert_eq!(Settings::from_json(&json!({"local_addresses": "junk"})).local, LocalAddresses::Auto);
    }

    #[test]
    fn browser_locations_per_system() {
        let win = candidate_paths("windows", &["C:\\Program Files".into(), "C:\\Program Files (x86)".into()], "C:\\Users\\a\\AppData\\Local", "");
        let names: Vec<String> = win.iter().map(|p| p.to_string_lossy().replace('\\', "/")).collect();
        assert!(names.iter().any(|n| n == "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"), "Edge is on every Windows 10 and 11");
        assert!(names.iter().any(|n| n.ends_with("AppData/Local/Google/Chrome/Application/chrome.exe")));
        let chrome = names.iter().position(|n| n.contains("Google/Chrome")).unwrap();
        let edge = names.iter().position(|n| n.contains("Edge")).unwrap();
        assert!(chrome < edge, "Chrome is preferred to Edge");
        let mac = candidate_paths("macos", &[], "", "/Users/a");
        assert_eq!(mac[0], Path::new("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"));
        assert!(mac.iter().any(|p| p.starts_with("/Users/a/Applications")));
        let linux = candidate_paths("linux", &[], "", "/home/a");
        assert!(linux.contains(&PathBuf::from("/usr/bin/chromium")));
        assert!(linux.contains(&PathBuf::from("/usr/bin/google-chrome")));
        assert!(linux_names().contains(&"chromium-browser") && linux_names().contains(&"microsoft-edge"));
    }

    #[test]
    fn the_browser_starts_headless_in_its_own_profile_without_asking_anything() {
        let profile = Path::new("/data/fllm/browser-profile");
        let args = launch_args(profile, false, false, 1280, 800);
        assert!(args.contains(&"--headless=new".to_owned()));
        assert!(args.contains(&"--remote-debugging-port=0".to_owned()));
        assert!(args.contains(&"--remote-debugging-address=127.0.0.1".to_owned()));
        assert!(args.contains(&"--user-data-dir=/data/fllm/browser-profile".to_owned()));
        assert!(args.contains(&"--no-first-run".to_owned()) && args.contains(&"--disable-extensions".to_owned()));
        assert!(args.contains(&"--window-size=1280,800".to_owned()));
        assert!(!args.contains(&"--no-sandbox".to_owned()), "the sandbox of the browser stays on unless running as root");
        assert_eq!(args.last().unwrap(), "about:blank");
        let headed = launch_args(profile, true, true, 1000, 700);
        assert!(!headed.iter().any(|a| a.starts_with("--headless")), "showing the window means not headless");
        assert!(headed.contains(&"--no-sandbox".to_owned()));
    }

    #[test]
    fn the_debugging_port_is_read_from_the_active_port_file() {
        assert_eq!(parse_active_port("41893\n/devtools/browser/0b1e-22\n"), Some((41893, "/devtools/browser/0b1e-22".to_owned())));
        assert_eq!(parse_active_port("41893\r\n/devtools/browser/x\r\n"), Some((41893, "/devtools/browser/x".to_owned())));
        for bad in ["", "41893\n", "0\n/devtools/browser/x", "abc\n/x", "41893\nnot-a-path"] {
            assert_eq!(parse_active_port(bad), None, "{bad:?}");
        }
    }
}
