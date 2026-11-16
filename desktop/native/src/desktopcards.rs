//! Row 47 of PARITY.md: *Settings > Desktop app* (`desktop.js`): keep running in the tray, start at
//! sign-in, the quick-entry hotkey, the server's log folder and its restart. The words are the
//! page's. What the system does with the choices is in `desktop.rs` and `shell.rs`.

use crate::app::Message;
use crate::desktop::{self, Desktop, Hotkey};
use crate::shell::{self, Shell};
use crate::models::card;
use crate::theme::{self, Tokens};
use iced::widget::{button, checkbox, column, container, row, text};
use iced::{Alignment, Element, Font, Length, Task};
use std::path::Path;

pub const TITLE: &str = "Desktop app";
pub const TRAY_TITLE: &str = "Keep running in the tray";
pub const TRAY_TEXT: &str = "Closing the window hides it instead of quitting. Quit from the tray icon.";
pub const AUTOSTART_TITLE: &str = "Start fLLM when I sign in";
pub const AUTOSTART_TEXT: &str = "Starts the app when you sign in to this computer.";
pub const HOTKEY_TITLE: &str = "Quick entry hotkey";
pub const HOTKEY_TEXT: &str = "Works from any program: brings fLLM to the front with a new chat. Click the field and press the keys; it needs Ctrl, Alt or Cmd. It does not work on some Linux desktops (Wayland).";
pub const VARIANT_TITLE: &str = "Graphics backends";
pub const VARIANT_TEXT: &str = "This installation holds one server build only.";
pub const LOG_TITLE: &str = "Server log";
pub const UPDATE_TITLE: &str = "Updates";
pub const UPDATE_AUTO_TITLE: &str = "Look for updates at the start";
pub const UPDATE_AUTO_TEXT: &str = "Asks GitHub which version is the newest when the app starts. Nothing else is sent, and nothing is asked while this is off. A newer version is installed from its download page.";
pub const SERVER_TITLE: &str = "Server";
pub const SERVER_TEXT: &str = "Stops the server and starts it again. The page reloads when it is ready.";

/// What the pane holds (the fields of `refresh` in `desktop.js`).
#[derive(Debug, Clone, Default)]
pub struct DesktopPane {
    pub s: Desktop,
    pub autostart: bool,
    /// The hotkey field has the focus and waits for keys.
    pub recording: bool,
    /// The line at the end of the card (`desktopStatus`).
    pub status: String,
    /// Where the file of settings is kept.
    pub file: std::path::PathBuf,
    /// What the look for a newer version found.
    pub update: crate::updates::State,
}

#[derive(Debug, Clone)]
pub enum DkMsg {
    Tray(bool),
    Autostart(bool),
    /// What the system says of the start at sign-in after a change (or the look at it).
    AutostartIs(Result<bool, String>),
    /// The field of the hotkey was clicked.
    Record,
    Off,
    Default,
    Logs,
    /// The *Check for updates* button.
    UpdateCheck,
    Updated(Result<Option<crate::updates::Release>, String>),
    UpdateOpen,
    UpdateAuto(bool),
}

impl DesktopPane {
    pub fn load(file: &Path) -> DesktopPane {
        DesktopPane { s: Desktop::load(file), file: file.to_path_buf(), ..Default::default() }
    }

    /// Looks at the start at sign-in (`refresh` of the page).
    pub fn refresh() -> Task<Message> {
        Task::perform(async { desktop::autostart_now(None) }, |r| Message::Dk(DkMsg::AutostartIs(r)))
    }

    fn save(&mut self) {
        if let Err(e) = self.s.save(&self.file) {
            self.status = e;
        }
    }

    /// `set_quick_entry`: the combination is registered first, and kept only when it works.
    pub fn set_quick_entry(&mut self, value: &str, shell: Option<&mut Shell>) {
        let value = value.trim().to_string();
        if let Some(shell) = shell {
            if let Err(e) = shell.set_hotkey(&value) {
                let _ = shell.set_hotkey(&self.s.quick_entry);
                self.status = e;
                return;
            }
        } else if !value.is_empty() {
            if let Err(e) = shell::parse_hotkey(&value) {
                self.status = e;
                return;
            }
        }
        self.status.clear();
        self.s.quick_entry = value;
        self.save();
    }

    /// A key while the field waits for one (`keydown` of `inpQuickEntry`): `true` when it was used.
    pub fn key(&mut self, key: &iced::keyboard::Key, mods: iced::keyboard::Modifiers, shell: Option<&mut Shell>) -> bool {
        if !self.recording {
            return false;
        }
        // Tab leaves the field, as in the page
        if matches!(key, iced::keyboard::Key::Named(iced::keyboard::key::Named::Tab)) {
            self.recording = false;
            return false;
        }
        match desktop::hotkey_of(key, mods) {
            Hotkey::Waiting => {}
            Hotkey::Refused(sentence) => self.status = sentence.to_string(),
            Hotkey::Chosen(keys) => {
                self.recording = false;
                self.set_quick_entry(&keys, shell);
            }
        }
        true
    }

    pub fn update(&mut self, m: DkMsg, shell: Option<&mut Shell>) -> Task<Message> {
        match m {
            DkMsg::Tray(on) => {
                self.s.keep_in_tray = on;
                self.status.clear();
                self.save();
                Task::none()
            }
            DkMsg::Autostart(on) => {
                self.status.clear();
                // the system is asked; the box shows what it answers
                self.autostart = on;
                Task::perform(async move { desktop::autostart_now(Some(on)) }, |r| Message::Dk(DkMsg::AutostartIs(r)))
            }
            DkMsg::AutostartIs(Ok(on)) => {
                self.autostart = on;
                Task::none()
            }
            DkMsg::AutostartIs(Err(e)) => {
                self.status = e;
                DesktopPane::refresh()
            }
            DkMsg::Record => {
                self.recording = true;
                Task::none()
            }
            DkMsg::Off => {
                self.recording = false;
                self.set_quick_entry("", shell);
                Task::none()
            }
            DkMsg::Default => {
                self.recording = false;
                self.set_quick_entry(desktop::DEFAULT_QUICK_ENTRY, shell);
                Task::none()
            }
            DkMsg::UpdateCheck => {
                self.update = crate::updates::State::Checking;
                Task::perform(async { crate::updates::check(crate::updates::RELEASES_URL).await }, |r| Message::Dk(DkMsg::Updated(r)))
            }
            DkMsg::Updated(r) => {
                self.update = crate::updates::State::of(r, env!("CARGO_PKG_VERSION"));
                Task::none()
            }
            DkMsg::UpdateOpen => {
                if let crate::updates::State::Available(r) = &self.update {
                    crate::app::open_link(&r.url);
                }
                Task::none()
            }
            DkMsg::UpdateAuto(on) => {
                self.s.check_updates = on;
                self.save();
                Task::none()
            }
            DkMsg::Logs => {
                let folder = desktop::log_folder();
                let _ = std::fs::create_dir_all(&folder);
                if let Err(e) = desktop::open_folder(&folder) {
                    self.status = e;
                }
                Task::none()
            }
        }
    }
}

fn item<'a>(t: Tokens, title: &'a str, desc: String, action: Element<'a, Message>) -> Element<'a, Message> {
    row![
        column![text(title).size(13).color(t.fg).font(Font { weight: iced::font::Weight::Semibold, ..Font::default() }), text(desc).size(12).color(t.muted)].spacing(2).width(Length::Fill),
        action,
    ]
    .spacing(14)
    .align_y(Alignment::Center)
    .into()
}

pub fn view<'a>(t: Tokens, pane: &'a DesktopPane, log_folder: String) -> Element<'a, Message> {
    let tray_box = checkbox(pane.s.keep_in_tray).on_toggle(|on| Message::Dk(DkMsg::Tray(on)));
    let field = button(
        text(if pane.recording {
            "Press the keys …".to_string()
        } else if pane.s.quick_entry.is_empty() {
            "Off".to_string()
        } else {
            pane.s.quick_entry.clone()
        })
        .size(13)
        .font(Font::MONOSPACE),
    )
    .on_press(Message::Dk(DkMsg::Record))
    .width(170)
    .padding([5, 10])
    .style({
        let recording = pane.recording;
        move |th, st| if recording { theme::primary_button(t)(th, st) } else { theme::secondary_button(t)(th, st) }
    });
    let hotkey = row![
        field,
        button(text("Off").size(12)).on_press(Message::Dk(DkMsg::Off)).padding([4, 10]).style(theme::ghost_button(t)),
        button(text("Default").size(12)).on_press(Message::Dk(DkMsg::Default)).padding([4, 10]).style(theme::ghost_button(t)),
    ]
    .spacing(6)
    .align_y(Alignment::Center);
    // the one build there is: the box of the page with one entry, which is not to be changed
    let variants: Element<'a, Message> = container(text("Automatic").size(13).color(t.muted))
        .padding([5, 10])
        .width(230)
        .style(move |_| container::Style { border: iced::Border { width: 1.0, color: t.separator, radius: 6.0.into() }, ..Default::default() })
        .into();
    let checking = pane.update == crate::updates::State::Checking;
    let mut update_actions = row![button(text("Check for updates").size(12)).on_press_maybe((!checking).then_some(Message::Dk(DkMsg::UpdateCheck))).padding([4, 10]).style(theme::secondary_button(t))].spacing(6);
    if matches!(pane.update, crate::updates::State::Available(_)) {
        update_actions = update_actions.push(button(text("Open the download page").size(12)).on_press(Message::Dk(DkMsg::UpdateOpen)).padding([4, 10]).style(theme::primary_button(t)));
    }
    let update_actions: Element<'a, Message> = update_actions.into();
    let mut body = column![
        item(t, TRAY_TITLE, TRAY_TEXT.to_string(), tray_box.into()),
        item(t, AUTOSTART_TITLE, AUTOSTART_TEXT.into(), checkbox(pane.autostart).on_toggle(|on| Message::Dk(DkMsg::Autostart(on))).into()),
        item(t, HOTKEY_TITLE, HOTKEY_TEXT.into(), hotkey.into()),
        item(t, VARIANT_TITLE, VARIANT_TEXT.into(), variants),
        item(t, LOG_TITLE, log_folder, button(text("Open log folder").size(12)).on_press(Message::Dk(DkMsg::Logs)).padding([4, 10]).style(theme::secondary_button(t)).into()),
        item(t, UPDATE_TITLE, pane.update.line(env!("CARGO_PKG_VERSION")), update_actions),
        item(t, UPDATE_AUTO_TITLE, UPDATE_AUTO_TEXT.into(), checkbox(pane.s.check_updates).on_toggle(|on| Message::Dk(DkMsg::UpdateAuto(on))).into()),
        item(t, SERVER_TITLE, SERVER_TEXT.into(), button(text("Restart server").size(12)).on_press(Message::RestartServer).padding([4, 10]).style(theme::secondary_button(t)).into()),
    ]
    .spacing(14);
    body = body.push(text(pane.status.as_str()).size(12).color(t.muted));
    card(t, TITLE, body.into())
}
