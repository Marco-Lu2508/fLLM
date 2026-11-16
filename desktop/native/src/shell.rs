//! Row 47 of PARITY.md: the parts of the desktop shell that talk to the windowing system — the
//! global quick-entry hotkey and the tray icon.
//!
//! The window has no way to ask the system for either, so this file owns them: it is made once,
//! after the window exists (`Shell::new` on the first tick of the clock the app listens to), and the
//! app looks at what happened to it with `poll`. What the system does with them is not something a
//! test can see; `desktop.rs` holds what is.
//!
//! On Linux the tray is the desktop's StatusNotifier (over D-Bus, without GTK): a desktop with no
//! host for it gives no icon, and then closing the window quits (`has_tray`). The hotkey works on
//! X11; some Wayland desktops do not hand out global keys (the page's own words say so).

use global_hotkey::{hotkey::HotKey, GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};

/// What happened in the system's part of the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellEvent {
    /// The quick-entry hotkey: the window comes to the front with a new chat.
    QuickEntry,
    /// *Open fLLM* of the tray, or a click on its icon.
    Open,
    /// *Restart server* of the tray.
    Restart,
    /// *Open log folder* of the tray.
    Logs,
    /// *Quit* of the tray.
    Quit,
}

/// `Ctrl+Alt+Space` as the hotkey of the system, or the sentence of the page that says it is not one.
pub fn parse_hotkey(keys: &str) -> Result<HotKey, String> {
    keys.parse::<HotKey>().map_err(|_| format!("\"{keys}\" is not a valid key combination."))
}

pub struct Shell {
    manager: Option<GlobalHotKeyManager>,
    registered: Option<HotKey>,
    /// Why there is no hotkey manager (a system that does not give out global keys).
    pub manager_error: Option<String>,
        tray: Option<tray::Tray>,
}

impl Shell {
    pub fn new() -> Shell {
        let (manager, manager_error) = match GlobalHotKeyManager::new() {
            Ok(m) => (Some(m), None),
            Err(e) => (None, Some(format!("The hotkey could not be set up: {e}"))),
        };
        Shell {
            manager,
            registered: None,
            manager_error,
                        tray: tray::Tray::new(),
        }
    }

    /// Registers `keys` as the only global hotkey; an empty text switches it off (`register_quick_entry` of the old shell).
    pub fn set_hotkey(&mut self, keys: &str) -> Result<(), String> {
        let Some(manager) = self.manager.as_ref() else {
            return if keys.is_empty() { Ok(()) } else { Err(self.manager_error.clone().unwrap_or_else(|| "The hotkey could not be set up.".into())) };
        };
        if let Some(old) = self.registered.take() {
            let _ = manager.unregister(old);
        }
        if keys.is_empty() {
            return Ok(());
        }
        let hotkey = parse_hotkey(keys)?;
        manager.register(hotkey).map_err(|e| format!("The hotkey could not be registered (another program may use it): {e}"))?;
        self.registered = Some(hotkey);
        Ok(())
    }

    /// Whether a tray icon is there to bring the window back.
    pub fn has_tray(&self) -> bool {
        self.tray.is_some()
    }

    /// What happened since the last look.
    pub fn poll(&mut self) -> Vec<ShellEvent> {
        let mut events = Vec::new();
        while let Ok(event) = GlobalHotKeyEvent::receiver().try_recv() {
            if event.state == HotKeyState::Pressed && self.registered.is_some_and(|h| h.id() == event.id) {
                events.push(ShellEvent::QuickEntry);
            }
        }
                if let Some(tray) = self.tray.as_ref() {
            events.extend(tray.poll());
        }
        events
    }
}

impl Default for Shell {
    fn default() -> Self {
        Shell::new()
    }
}

mod tray {
    use super::ShellEvent;
    use tray_icon::{
        menu::{Menu, MenuEvent, MenuId, MenuItem},
        MouseButton, MouseButtonState, TrayIcon, TrayIconBuilder, TrayIconEvent,
    };

    /// The icon of the program, as the tray shows it.
    const ICON: &[u8] = include_bytes!("../../app-icon.png");

    pub struct Tray {
        _icon: TrayIcon,
        open: MenuId,
        restart: MenuId,
        logs: MenuId,
        quit: MenuId,
    }

    impl Tray {
        /// The menu of the old shell: *Open fLLM*, *Restart server*, *Open log folder*, *Quit*.
        pub fn new() -> Option<Tray> {
            let open = MenuItem::new("Open fLLM", true, None);
            let restart = MenuItem::new("Restart server", true, None);
            let logs = MenuItem::new("Open log folder", true, None);
            let quit = MenuItem::new("Quit", true, None);
            let menu = Menu::with_items(&[&open, &restart, &logs, &quit]).ok()?;
            let image = image::load_from_memory(ICON).ok()?.into_rgba8();
            let (w, h) = image.dimensions();
            let icon = tray_icon::Icon::from_rgba(image.into_raw(), w, h).ok()?;
            let tray = TrayIconBuilder::new().with_tooltip("fLLM").with_menu(Box::new(menu)).with_menu_on_left_click(false).with_icon(icon).build().ok()?;
            Some(Tray { _icon: tray, open: open.id().clone(), restart: restart.id().clone(), logs: logs.id().clone(), quit: quit.id().clone() })
        }

        pub fn poll(&self) -> Vec<ShellEvent> {
            let mut events = Vec::new();
            while let Ok(event) = MenuEvent::receiver().try_recv() {
                if event.id == self.open {
                    events.push(ShellEvent::Open);
                } else if event.id == self.restart {
                    events.push(ShellEvent::Restart);
                } else if event.id == self.logs {
                    events.push(ShellEvent::Logs);
                } else if event.id == self.quit {
                    events.push(ShellEvent::Quit);
                }
            }
            while let Ok(event) = TrayIconEvent::receiver().try_recv() {
                if let TrayIconEvent::Click { button: MouseButton::Left, button_state: MouseButtonState::Up, .. } = event {
                    events.push(ShellEvent::Open);
                }
            }
            events
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_key_combination_is_one_or_a_sentence_says_it_is_not() {
        assert!(parse_hotkey("Ctrl+Alt+Space").is_ok());
        assert!(parse_hotkey("Ctrl+Shift+K").is_ok());
        assert!(parse_hotkey("Super+7").is_ok());
        assert!(parse_hotkey("Alt+F5").is_ok());
        assert_eq!(parse_hotkey("Banana").unwrap_err(), "\"Banana\" is not a valid key combination.");
    }
}
