//! The design tokens of `design-system/fllm-studio/MASTER.md`, as iced colors and widget styles.
//! One accent (copper), system gray neutrals, hairlines, no decoration that carries no meaning.

use iced::border::Radius;
use iced::widget::{button, container};
use iced::{Background, Border, Color, Shadow, Theme};

const fn rgb(hex: u32) -> Color {
    Color { r: ((hex >> 16) & 0xff) as f32 / 255.0, g: ((hex >> 8) & 0xff) as f32 / 255.0, b: (hex & 0xff) as f32 / 255.0, a: 1.0 }
}

fn alpha(c: Color, a: f32) -> Color {
    Color { a, ..c }
}

/// One accent of the Studio (`ACCENT_COLORS` of `core.js`): what people see, what the setting
/// keeps, the colour of the thumb and the two primaries — the dark theme takes `primary`, the
/// light one `primary_light`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Accent {
    pub name: &'static str,
    pub value: &'static str,
    pub thumb: Color,
    pub primary: Color,
    pub primary_light: Color,
}

/// The eight accents of the Studio, in its order.
pub const ACCENTS: [Accent; 8] = [
    Accent { name: "Copper", value: "patina", thumb: rgb(0xe08a4b), primary: rgb(0xe08a4b), primary_light: rgb(0xa64b1a) },
    Accent { name: "Signal", value: "emerald", thumb: rgb(0x4ade80), primary: rgb(0x4ade80), primary_light: rgb(0x15803d) },
    Accent { name: "Indigo", value: "indigo", thumb: rgb(0x8b8cf9), primary: rgb(0x8b8cf9), primary_light: rgb(0x4338ca) },
    Accent { name: "Cyan", value: "cyan", thumb: rgb(0x22d3ee), primary: rgb(0x22d3ee), primary_light: rgb(0x0e7490) },
    Accent { name: "Rose", value: "rose", thumb: rgb(0xfb7185), primary: rgb(0xfb7185), primary_light: rgb(0xbe123c) },
    Accent { name: "Violet", value: "purple", thumb: rgb(0xc084fc), primary: rgb(0xc084fc), primary_light: rgb(0x7e22ce) },
    Accent { name: "Steel", value: "blue", thumb: rgb(0x94a3b8), primary: rgb(0x94a3b8), primary_light: rgb(0x334155) },
    Accent { name: "Lime", value: "green", thumb: rgb(0xa3e635), primary: rgb(0xa3e635), primary_light: rgb(0x4d7c0f) },
];

/// The accent of the setting; an unknown value falls back to the first one, as the Studio does.
pub fn accent_of(value: &str) -> Accent {
    ACCENTS.iter().find(|a| a.value == value).copied().unwrap_or(ACCENTS[0])
}

/// The colour under an untinted user message: `--secondary` of the Studio.
pub fn secondary(dark: bool) -> Color {
    if dark { rgb(0x1e2b35) } else { rgb(0xe0e8e5) }
}

/// Every color of one theme. The values are the ones of `refine.css`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tokens {
    pub dark: bool,
    pub bg: Color,
    pub surface: Color,
    pub surface_2: Color,
    pub fg: Color,
    pub fg_soft: Color,
    pub muted: Color,
    pub fill: Color,
    pub fill_hover: Color,
    pub separator: Color,
    pub primary: Color,
    pub primary_soft: Color,
    pub on_primary: Color,
    pub danger: Color,
    pub ok: Color,
}

impl Tokens {
    /// The colours of one theme with the chosen accent: the dark theme takes the accent's own
    /// colour, the light one its lighter twin, and the text on it is dark or light by the
    /// luminance the Studio computes (`--primary`, `--primary-foreground` of `core.js`).
    pub fn new(dark: bool, accent: &str) -> Self {
        let chosen = accent_of(accent);
        if dark {
            let primary = chosen.primary;
            let on_primary = on_accent(primary);
            Tokens {
                dark,
                bg: rgb(0x161618),
                surface: rgb(0x1c1c1e),
                surface_2: rgb(0x232326),
                fg: rgb(0xf5f5f7),
                fg_soft: rgb(0xd1d1d6),
                muted: rgb(0xa1a1a6),
                fill: alpha(rgb(0x787880), 0.24),
                fill_hover: alpha(rgb(0x787880), 0.32),
                separator: alpha(rgb(0x545458), 0.65),
                primary,
                primary_soft: alpha(primary, 0.16),
                on_primary,
                danger: rgb(0xff6961),
                ok: rgb(0x32d74b),
            }
        } else {
            let primary = chosen.primary_light;
            let on_primary = on_accent(primary);
            Tokens {
                dark,
                bg: rgb(0xf5f5f7),
                surface: rgb(0xffffff),
                surface_2: rgb(0xf5f5f7),
                fg: rgb(0x1d1d1f),
                fg_soft: rgb(0x424245),
                muted: rgb(0x636366),
                fill: alpha(rgb(0x787880), 0.12),
                fill_hover: alpha(rgb(0x787880), 0.18),
                separator: alpha(rgb(0x3c3c43), 0.16),
                primary,
                primary_soft: alpha(primary, 0.14),
                on_primary,
                danger: rgb(0xc0392b),
                ok: rgb(0x1e8e3e),
            }
        }
    }

}

/// The relative luminance of a colour, the way `updateAccent` of `core.js` computes it.
fn luminance(c: Color) -> f64 {
    0.2126 * c.r as f64 + 0.7152 * c.g as f64 + 0.0722 * c.b as f64
}

/// The text on the accent: dark on a light colour, light on a dark one (the `lum > 0.45` of
/// `core.js` with its two colours).
pub fn on_accent(primary: Color) -> Color {
    if luminance(primary) > 0.45 { rgb(0x06161a) } else { rgb(0xf6fbfa) }
}

impl Tokens {
    /// The iced theme: its palette feeds the widgets that this file does not style itself (scrollbars, the markdown view, the code highlighter).
    pub fn iced_theme(&self) -> Theme {
        Theme::custom(
            if self.dark { "fLLM dark" } else { "fLLM light" }.to_string(),
            iced::theme::Palette { background: self.bg, text: self.fg, primary: self.primary, success: self.ok, warning: self.primary, danger: self.danger },
        )
    }
}

pub const R_SM: f32 = 8.0;
pub const R_MD: f32 = 10.0;
pub const R_LG: f32 = 14.0;
pub const R_XL: f32 = 22.0;

fn border(radius: f32, width: f32, color: Color) -> Border {
    Border { color, width, radius: Radius::from(radius) }
}

/// Plain page area.
pub fn page(t: Tokens) -> impl Fn(&Theme) -> container::Style {
    move |_| container::Style { background: Some(Background::Color(t.bg)), text_color: Some(t.fg), ..Default::default() }
}

/// The sidebar: one flat material, a hairline on the right.
pub fn sidebar(t: Tokens) -> impl Fn(&Theme) -> container::Style {
    move |_| container::Style {
        background: Some(Background::Color(if t.dark { t.surface } else { rgb(0xebebf0) })),
        text_color: Some(t.fg),
        border: border(0.0, 0.0, Color::TRANSPARENT),
        ..Default::default()
    }
}

/// A raised surface (the composer, cards): a hairline ring and a soft shadow.
pub fn card(t: Tokens, radius: f32, focused: bool) -> impl Fn(&Theme) -> container::Style {
    move |_| container::Style {
        background: Some(Background::Color(t.surface)),
        text_color: Some(t.fg),
        border: border(radius, if focused { 1.0 } else { 0.5 }, if focused { alpha(t.fg, 0.45) } else { t.separator }),
        shadow: Shadow { color: Color { a: if t.dark { 0.35 } else { 0.08 }, ..Color::BLACK }, offset: iced::Vector::new(0.0, 4.0), blur_radius: 18.0 },
        ..Default::default()
    }
}

/// The user's message, one corner tighter than the others like a speech bubble: tinted with the
/// accent (and the text on it) when that switch is on, the `--secondary` of the Studio when it is
/// off (`colored-user-bubble` of `style.css`).
pub fn bubble(t: Tokens, tint: bool) -> impl Fn(&Theme) -> container::Style {
    move |_| container::Style {
        background: Some(Background::Color(if tint { t.primary } else { secondary(t.dark) })),
        text_color: Some(if tint { t.on_primary } else { t.fg }),
        border: Border { color: Color::TRANSPARENT, width: 0.0, radius: Radius { top_left: 20.0, top_right: 20.0, bottom_right: 6.0, bottom_left: 20.0 } },
        ..Default::default()
    }
}

/// A code block or other inset surface.
pub fn inset(t: Tokens) -> impl Fn(&Theme) -> container::Style {
    move |_| container::Style {
        background: Some(Background::Color(t.surface_2)),
        text_color: Some(t.fg),
        border: border(R_MD, 0.5, t.separator),
        ..Default::default()
    }
}

/// A small label on a quiet fill (the quantization badge, the model name).
pub fn chip(t: Tokens) -> impl Fn(&Theme) -> container::Style {
    move |_| container::Style { background: Some(Background::Color(t.fill)), text_color: Some(t.fg_soft), border: border(999.0, 0.0, Color::TRANSPARENT), ..Default::default() }
}

/// A row of the sidebar's navigation: the accent tint marks the page you are on.
pub fn nav_button(t: Tokens, selected: bool) -> impl Fn(&Theme, button::Status) -> button::Style {
    row_button_with(t, selected, true)
}

/// A row of a list (chats): transparent, the quiet fill on hover and for the selected one.
pub fn row_button(t: Tokens, selected: bool) -> impl Fn(&Theme, button::Status) -> button::Style {
    row_button_with(t, selected, true)
}

fn row_button_with(t: Tokens, selected: bool, accent: bool) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |_, status| {
        let bg = if selected {
            Some(Background::Color(if accent { t.primary_soft } else { t.fill_hover }))
        } else {
            match status {
                button::Status::Hovered | button::Status::Pressed => Some(Background::Color(t.fill)),
                _ => None,
            }
        };
        button::Style { background: bg, text_color: t.fg, border: border(R_SM, 0.0, Color::TRANSPARENT), ..Default::default() }
    }
}

/// A button that only carries a label: the row around it shows hover and selection.
pub fn plain_button(t: Tokens) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |_, _| button::Style { background: None, text_color: t.fg, border: border(R_SM, 0.0, Color::TRANSPARENT), ..Default::default() }
}

/// A button without a background (icon buttons): the fill appears on hover.
pub fn ghost_button(t: Tokens) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |_, status| {
        let (bg, fg) = match status {
            button::Status::Hovered => (Some(Background::Color(t.fill)), t.fg),
            button::Status::Pressed => (Some(Background::Color(t.fill_hover)), t.fg),
            button::Status::Disabled => (None, alpha(t.muted, 0.5)),
            button::Status::Active => (None, t.muted),
        };
        button::Style { background: bg, text_color: fg, border: border(R_SM, 0.0, Color::TRANSPARENT), ..Default::default() }
    }
}

/// The one filled action of a view: the accent, or a gray fill while there is nothing to do.
pub fn primary_button(t: Tokens) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |_, status| {
        let (bg, fg) = match status {
            button::Status::Disabled => (t.fill, t.muted),
            button::Status::Hovered => (Color { a: 0.92, ..t.primary }, t.on_primary),
            button::Status::Pressed => (Color { a: 0.82, ..t.primary }, t.on_primary),
            button::Status::Active => (t.primary, t.on_primary),
        };
        button::Style { background: Some(Background::Color(bg)), text_color: fg, border: border(999.0, 0.0, Color::TRANSPARENT), ..Default::default() }
    }
}

/// The quiet action next to a row (Unload, Pause, Cancel): a gray fill, so it reads as a button and not as a label.
pub fn secondary_button(t: Tokens) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |_, status| {
        let (bg, fg) = match status {
            button::Status::Disabled => (t.fill, alpha(t.muted, 0.6)),
            button::Status::Hovered => (t.fill_hover, t.fg),
            button::Status::Pressed => (t.fill_hover, t.fg),
            button::Status::Active => (t.fill, t.fg),
        };
        button::Style { background: Some(Background::Color(bg)), text_color: fg, border: border(999.0, 0.0, Color::TRANSPARENT), ..Default::default() }
    }
}

/// Contrast ratio of two opaque colors (WCAG 2.x), for the tests that guard the palette.
pub fn contrast(a: Color, b: Color) -> f32 {
    fn lum(c: Color) -> f32 {
        let f = |v: f32| if v <= 0.03928 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) };
        0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b)
    }
    let (la, lb) = (lum(a), lum(b));
    (la.max(lb) + 0.05) / (la.min(lb) + 0.05)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_colors_keep_their_contrast_in_both_themes() {
        for dark in [false, true] {
            let t = Tokens::new(dark, "patina");
            assert!(contrast(t.fg, t.bg) >= 7.0, "primary text, dark={dark}");
            assert!(contrast(t.fg_soft, t.surface) >= 7.0, "soft text, dark={dark}");
            assert!(contrast(t.muted, t.surface) >= 4.5, "secondary text on a card, dark={dark}");
            assert!(contrast(t.muted, t.bg) >= 4.5, "secondary text on the page, dark={dark}");
            assert!(contrast(t.on_primary, t.primary) >= 4.5, "text on the accent, dark={dark}");
            assert!(contrast(t.primary, t.surface) >= 3.0, "the accent as an icon on a card, dark={dark}");
        }
    }

    /// Every accent of the Studio keeps the contrast in both themes (the text on it follows the
    /// luminance the Studio computes, and an unknown value falls back to Copper).
    #[test]
    fn every_accent_keeps_the_contrast_and_an_unknown_one_falls_back() {
        for accent in ACCENTS {
            for dark in [false, true] {
                let t = Tokens::new(dark, accent.value);
                assert_eq!(t.primary, if dark { accent.primary } else { accent.primary_light });
                assert!(
                    contrast(t.on_primary, t.primary) >= 4.5,
                    "text on {} in {:?}",
                    accent.value,
                    if dark { "dark" } else { "light" }
                );
                assert!(contrast(t.primary, t.surface) >= 3.0, "{} on a card", accent.value);
                // the thumb people click is the colour of the Studio
                assert_eq!(accent.thumb, accent.primary, "{} keeps one colour", accent.value);
            }
        }
        assert_eq!(accent_of("nope"), ACCENTS[0], "an unknown accent is Copper");
        assert_eq!(ACCENTS.len(), 8, "the eight of the Studio");
        assert_eq!(ACCENTS[0].name, "Copper", "the first is copper, the colour of Rust");
    }

    /// The untinted bubble is the `--secondary` of the Studio, in both themes.
    #[test]
    fn the_untinted_bubble_is_the_secondary_of_the_studio() {
        assert_eq!(secondary(false), rgb(0xe0e8e5));
        assert_eq!(secondary(true), rgb(0x1e2b35));
    }
}
