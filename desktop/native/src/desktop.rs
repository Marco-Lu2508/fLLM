//! Row 47 of PARITY.md: what the app does as a program of the desktop — start with the system, keep a
//! quick-entry hotkey, open its folder of logs. The settings of `desktop.js` (*Desktop app*) and what
//! `main.rs` of the shell they came from did for them.
//!
//! This file has the parts that need no window: the file of settings, the words of a key
//! combination, the entry the system starts the program from at sign-in and the opening of a folder.
//! `shell.rs` has the parts that talk to the windowing system (the hotkey, the tray).

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// The key combination that brings the window to the front with a new chat, until the user chooses another.
pub const DEFAULT_QUICK_ENTRY: &str = "Ctrl+Alt+Space";
/// Given to the program when the system starts it at sign-in.
pub const AUTOSTART_ARG: &str = "--autostarted";

/// What the app keeps in `desktop.json` next to its chats.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Desktop {
    /// Closing the window hides it instead of quitting; the tray icon brings it back.
    pub keep_in_tray: bool,
    /// The global hotkey (`""` = off).
    pub quick_entry: String,
    /// Where the window was and how big when the app was closed (`window-state` of the old shell).
    pub window: Option<WindowState>,
    /// Look for a newer version on GitHub when the app starts (off until you switch it on: nothing is sent without it).
    pub check_updates: bool,
}

impl Default for Desktop {
    fn default() -> Self {
        Desktop { keep_in_tray: false, quick_entry: DEFAULT_QUICK_ENTRY.to_string(), window: None, check_updates: false }
    }
}

/// The size and place of the window, in logical pixels.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WindowState {
    pub width: f32,
    pub height: f32,
    pub x: Option<f32>,
    pub y: Option<f32>,
}

impl WindowState {
    /// What of a remembered window may be used: a size that is not tiny or absurd, and a place that is not far outside any screen.
    pub fn usable(&self) -> Option<WindowState> {
        let ok = |v: f32, lo: f32, hi: f32| v.is_finite() && v >= lo && v <= hi;
        if !ok(self.width, 480.0, 16000.0) || !ok(self.height, 320.0, 16000.0) {
            return None;
        }
        let place = match (self.x, self.y) {
            (Some(x), Some(y)) if ok(x, -16000.0, 16000.0) && ok(y, -16000.0, 16000.0) => (Some(x), Some(y)),
            _ => (None, None),
        };
        Some(WindowState { width: self.width, height: self.height, x: place.0, y: place.1 })
    }

    /// The settings the window is made with.
    pub fn settings(&self) -> (iced::Size, iced::window::Position) {
        let position = match (self.x, self.y) {
            (Some(x), Some(y)) => iced::window::Position::Specific(iced::Point::new(x, y)),
            _ => iced::window::Position::Default,
        };
        (iced::Size::new(self.width, self.height), position)
    }
}

/// Where the settings of the desktop are kept.
pub fn settings_file() -> PathBuf {
    crate::store::data_file().with_file_name("desktop.json")
}

impl Desktop {
    /// A missing or unreadable file is the defaults.
    pub fn load(path: &Path) -> Desktop {
        std::fs::read_to_string(path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        std::fs::write(path, serde_json::to_string_pretty(self).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
    }
}

/// What a press of the keys in the hotkey field comes to (`hotkeyText` of `desktop.js`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Hotkey {
    /// A modifier alone: nothing happens yet.
    Waiting,
    /// A combination without Ctrl, Alt or Cmd: the sentence that says so.
    Refused(&'static str),
    /// The combination in the form the app understands (`Ctrl+Alt+Space`).
    Chosen(String),
}

/// `hotkeyText(e)` for a key of the window.
pub fn hotkey_of(key: &iced::keyboard::Key, mods: iced::keyboard::Modifiers) -> Hotkey {
    use iced::keyboard::{key::Named, Key};
    let name = match key {
        Key::Character(c) => {
            let c = c.as_str();
            if c == " " { "Space".to_string() } else { c.to_uppercase() }
        }
        Key::Named(Named::Space) => "Space".to_string(),
        Key::Named(Named::Shift | Named::Control | Named::Alt | Named::Super | Named::Meta | Named::AltGraph) => return Hotkey::Waiting,
        Key::Named(n) => format!("{n:?}"),
        Key::Unidentified => return Hotkey::Waiting,
    };
    if !mods.control() && !mods.alt() && !mods.logo() {
        return Hotkey::Refused("Use at least one of Ctrl, Alt or Cmd.");
    }
    let mut parts: Vec<String> = Vec::new();
    if mods.control() {
        parts.push("Ctrl".into());
    }
    if mods.alt() {
        parts.push("Alt".into());
    }
    if mods.shift() {
        parts.push("Shift".into());
    }
    if mods.logo() {
        parts.push("Super".into());
    }
    parts.push(name);
    Hotkey::Chosen(parts.join("+"))
}

/// The system the entry is written for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Os {
    Linux,
    Mac,
    Windows,
}

impl Os {
    pub fn current() -> Os {
        if cfg!(windows) {
            Os::Windows
        } else if cfg!(target_os = "macos") {
            Os::Mac
        } else {
            Os::Linux
        }
    }
}

/// The entry that starts the program at sign-in: where it goes below the home (or config) folder, and what it says.
/// Windows keeps it in the registry instead: there the text is the value of the `Run` key.
pub fn autostart_entry(os: Os, exe: &Path) -> (PathBuf, String) {
    let exe_text = exe.display().to_string();
    match os {
        Os::Linux => (
            PathBuf::from("autostart/fllm.desktop"),
            format!("[Desktop Entry]\nType=Application\nName=fLLM\nComment=Starts fLLM when you sign in\nExec=\"{exe_text}\" {AUTOSTART_ARG}\nX-GNOME-Autostart-enabled=true\n"),
        ),
        Os::Mac => (
            PathBuf::from("Library/LaunchAgents/com.fllm.app.plist"),
            format!(
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n<plist version=\"1.0\">\n<dict>\n  <key>Label</key>\n  <string>com.fllm.app</string>\n  <key>ProgramArguments</key>\n  <array>\n    <string>{exe_text}</string>\n    <string>{AUTOSTART_ARG}</string>\n  </array>\n  <key>RunAtLoad</key>\n  <true/>\n</dict>\n</plist>\n"
            ),
        ),
        Os::Windows => (PathBuf::from("Run/fLLM"), format!("\"{exe_text}\" {AUTOSTART_ARG}")),
    }
}

/// The folder an entry of Linux or macOS goes below: `$XDG_CONFIG_HOME` (or `~/.config`) for the first, the home for the second.
fn entry_base(os: Os, home: &Path, xdg_config: Option<&Path>) -> PathBuf {
    match os {
        Os::Linux => xdg_config.map(Path::to_path_buf).unwrap_or_else(|| home.join(".config")),
        _ => home.to_path_buf(),
    }
}

const RUN_KEY: &str = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run";

/// Whether the program starts at sign-in.
pub fn autostart_enabled(os: Os, home: &Path, xdg_config: Option<&Path>) -> bool {
    match os {
        Os::Windows => std::process::Command::new("reg").args(["query", RUN_KEY, "/v", "fLLM"]).output().is_ok_and(|o| o.status.success()),
        _ => entry_base(os, home, xdg_config).join(autostart_entry(os, Path::new("")).0).is_file(),
    }
}

/// Switches the start at sign-in on or off.
pub fn set_autostart(os: Os, on: bool, exe: &Path, home: &Path, xdg_config: Option<&Path>) -> Result<(), String> {
    let (rel, content) = autostart_entry(os, exe);
    if os == Os::Windows {
        let status = if on {
            std::process::Command::new("reg").args(["add", RUN_KEY, "/v", "fLLM", "/t", "REG_SZ", "/d", &content, "/f"]).status()
        } else {
            std::process::Command::new("reg").args(["delete", RUN_KEY, "/v", "fLLM", "/f"]).status()
        };
        return match status {
            Ok(s) if s.success() || !on => Ok(()),
            Ok(s) => Err(format!("The registry did not take the entry ({s}).")),
            Err(e) => Err(format!("The registry could not be reached: {e}")),
        };
    }
    let file = entry_base(os, home, xdg_config).join(rel);
    if on {
        if let Some(dir) = file.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        std::fs::write(&file, content).map_err(|e| e.to_string())
    } else {
        match std::fs::remove_file(&file) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.to_string()),
        }
    }
}

/// `autostart` for this user, this system and this program.
pub fn autostart_now(on: Option<bool>) -> Result<bool, String> {
    let home = dirs::home_dir().ok_or("The home folder is not known.")?;
    let xdg = std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from).filter(|p| p.is_absolute());
    let os = Os::current();
    if let Some(on) = on {
        let exe = std::env::current_exe().map_err(|e| e.to_string())?;
        set_autostart(os, on, &exe, &home, xdg.as_deref())?;
    }
    Ok(autostart_enabled(os, &home, xdg.as_deref()))
}

/// Opens a folder in the file manager of the system.
pub fn open_folder(path: &Path) -> Result<(), String> {
    let program = if cfg!(windows) {
        "explorer"
    } else if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    };
    std::process::Command::new(program).arg(path).spawn().map(|_| ()).map_err(|e| format!("The folder could not be opened: {e}"))
}

/// The folder of the logs of the server (`server.log` lies in it): made when it is not there yet.
pub fn log_folder() -> PathBuf {
    crate::store::data_file().parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."))
}

#[cfg(test)]
mod tests {
    use super::*;
    use iced::keyboard::{key::Named, Key, Modifiers};

    #[test]
    fn the_settings_start_with_the_hotkey_on_and_the_tray_off_and_survive_a_restart() {
        let d = tempfile::tempdir().unwrap();
        let file = d.path().join("sub/desktop.json");
        assert_eq!(Desktop::load(&file), Desktop { keep_in_tray: false, quick_entry: "Ctrl+Alt+Space".into(), ..Default::default() });
        Desktop { keep_in_tray: true, quick_entry: String::new(), ..Default::default() }.save(&file).unwrap();
        assert_eq!(Desktop::load(&file), Desktop { keep_in_tray: true, quick_entry: String::new(), ..Default::default() });
        // a damaged file is the defaults, and a file with only one setting keeps the other's default
        std::fs::write(&file, "{ not json").unwrap();
        assert_eq!(Desktop::load(&file), Desktop::default());
        std::fs::write(&file, r#"{"keep_in_tray": true}"#).unwrap();
        assert_eq!(Desktop::load(&file), Desktop { keep_in_tray: true, quick_entry: "Ctrl+Alt+Space".into(), ..Default::default() });
    }

    #[test]
    fn the_keys_become_the_words_of_the_page() {
        let ctrl_alt = Modifiers::CTRL | Modifiers::ALT;
        assert_eq!(hotkey_of(&Key::Named(Named::Space), ctrl_alt), Hotkey::Chosen("Ctrl+Alt+Space".into()));
        assert_eq!(hotkey_of(&Key::Character("k".into()), Modifiers::CTRL | Modifiers::SHIFT), Hotkey::Chosen("Ctrl+Shift+K".into()));
        assert_eq!(hotkey_of(&Key::Character("7".into()), Modifiers::LOGO), Hotkey::Chosen("Super+7".into()));
        assert_eq!(hotkey_of(&Key::Named(Named::F5), Modifiers::ALT), Hotkey::Chosen("Alt+F5".into()));
        assert_eq!(hotkey_of(&Key::Named(Named::ArrowUp), Modifiers::CTRL), Hotkey::Chosen("Ctrl+ArrowUp".into()));
        // a modifier alone waits, a combination without Ctrl, Alt or Cmd is refused
        assert_eq!(hotkey_of(&Key::Named(Named::Control), Modifiers::CTRL), Hotkey::Waiting);
        assert_eq!(hotkey_of(&Key::Named(Named::Shift), Modifiers::SHIFT), Hotkey::Waiting);
        assert_eq!(hotkey_of(&Key::Character("a".into()), Modifiers::empty()), Hotkey::Refused("Use at least one of Ctrl, Alt or Cmd."));
        assert_eq!(hotkey_of(&Key::Character("a".into()), Modifiers::SHIFT), Hotkey::Refused("Use at least one of Ctrl, Alt or Cmd."));
    }

    #[test]
    fn linux_starts_the_program_from_a_desktop_entry_with_the_flag() {
        let d = tempfile::tempdir().unwrap();
        let home = d.path().join("home");
        let exe = Path::new("/opt/fllm/fllm-app");
        assert!(!autostart_enabled(Os::Linux, &home, None));
        set_autostart(Os::Linux, true, exe, &home, None).unwrap();
        let file = home.join(".config/autostart/fllm.desktop");
        let text = std::fs::read_to_string(&file).unwrap();
        assert!(text.contains("Exec=\"/opt/fllm/fllm-app\" --autostarted"), "{text}");
        assert!(text.starts_with("[Desktop Entry]\nType=Application\nName=fLLM\n"));
        assert!(autostart_enabled(Os::Linux, &home, None));
        set_autostart(Os::Linux, false, exe, &home, None).unwrap();
        assert!(!file.exists() && !autostart_enabled(Os::Linux, &home, None));
        // switching off what is not on is no error
        set_autostart(Os::Linux, false, exe, &home, None).unwrap();
        // $XDG_CONFIG_HOME wins over ~/.config
        let xdg = d.path().join("xdg");
        set_autostart(Os::Linux, true, exe, &home, Some(&xdg)).unwrap();
        assert!(xdg.join("autostart/fllm.desktop").is_file());
        assert!(autostart_enabled(Os::Linux, &home, Some(&xdg)) && !autostart_enabled(Os::Linux, &home, None));
    }

    #[test]
    fn macos_starts_it_from_a_launch_agent_and_windows_from_the_run_key() {
        let (rel, plist) = autostart_entry(Os::Mac, Path::new("/Applications/fLLM.app/Contents/MacOS/fllm-app"));
        assert_eq!(rel, PathBuf::from("Library/LaunchAgents/com.fllm.app.plist"));
        assert!(plist.contains("<string>/Applications/fLLM.app/Contents/MacOS/fllm-app</string>\n    <string>--autostarted</string>"));
        assert!(plist.contains("<key>RunAtLoad</key>\n  <true/>"));
        let d = tempfile::tempdir().unwrap();
        set_autostart(Os::Mac, true, Path::new("/x/fllm-app"), d.path(), None).unwrap();
        assert!(d.path().join("Library/LaunchAgents/com.fllm.app.plist").is_file());
        assert!(autostart_enabled(Os::Mac, d.path(), None));
        let (_, value) = autostart_entry(Os::Windows, Path::new(r"C:\Program Files\fLLM\fllm-app.exe"));
        assert_eq!(value, r#""C:\Program Files\fLLM\fllm-app.exe" --autostarted"#);
    }
}
