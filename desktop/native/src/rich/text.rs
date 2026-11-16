//! Text as outlines: a chart draws its words as paths of the window's own font, so it looks the same
//! on every system and needs no font installed where the picture is drawn.

use skrifa::instance::{LocationRef, Size};
use skrifa::outline::{DrawSettings, OutlinePen};
use skrifa::{FontRef, MetadataProvider};
use std::fmt::Write;
use std::sync::OnceLock;

/// Where the text stands at its point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Anchor {
    Start,
    Middle,
    End,
}

pub struct Face {
    font: FontRef<'static>,
}

/// The regular face of the interface font.
pub fn regular() -> &'static Face {
    static F: OnceLock<Face> = OnceLock::new();
    F.get_or_init(|| Face { font: FontRef::new(crate::app::FONT_REGULAR).expect("the interface font is a font") })
}

/// The semibold face of the interface font.
pub fn semibold() -> &'static Face {
    static F: OnceLock<Face> = OnceLock::new();
    F.get_or_init(|| Face { font: FontRef::new(crate::app::FONT_SEMIBOLD).expect("the interface font is a font") })
}

struct Pen<'a> {
    out: &'a mut String,
    scale: f32,
    x: f32,
    y: f32,
}

impl Pen<'_> {
    fn p(&self, x: f32, y: f32) -> (f32, f32) {
        (self.x + x * self.scale, self.y - y * self.scale)
    }
}

impl OutlinePen for Pen<'_> {
    fn move_to(&mut self, x: f32, y: f32) {
        let (x, y) = self.p(x, y);
        let _ = write!(self.out, "M{x:.2} {y:.2}");
    }
    fn line_to(&mut self, x: f32, y: f32) {
        let (x, y) = self.p(x, y);
        let _ = write!(self.out, "L{x:.2} {y:.2}");
    }
    fn quad_to(&mut self, cx0: f32, cy0: f32, x: f32, y: f32) {
        let (a, b) = self.p(cx0, cy0);
        let (x, y) = self.p(x, y);
        let _ = write!(self.out, "Q{a:.2} {b:.2} {x:.2} {y:.2}");
    }
    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        let (a, b) = self.p(cx0, cy0);
        let (c, d) = self.p(cx1, cy1);
        let (x, y) = self.p(x, y);
        let _ = write!(self.out, "C{a:.2} {b:.2} {c:.2} {d:.2} {x:.2} {y:.2}");
    }
    fn close(&mut self) {
        self.out.push('Z');
    }
}

impl Face {
    /// The width of `text` at `size` pixels.
    pub fn width(&self, text: &str, size: f32) -> f32 {
        let upem = self.font.metrics(Size::unscaled(), LocationRef::default()).units_per_em as f32;
        let charmap = self.font.charmap();
        let metrics = self.font.glyph_metrics(Size::unscaled(), LocationRef::default());
        text.chars()
            .map(|c| {
                let gid = charmap.map(c).or_else(|| charmap.map('?'));
                gid.and_then(|g| metrics.advance_width(g)).unwrap_or(upem * 0.5) * size / upem
            })
            .sum()
    }

    /// The outline of `text` as the `d` of an SVG path: its baseline is at `y`.
    pub fn path(&self, text: &str, size: f32, x: f32, y: f32, anchor: Anchor) -> String {
        let upem = self.font.metrics(Size::unscaled(), LocationRef::default()).units_per_em as f32;
        let scale = size / upem;
        let width = self.width(text, size);
        let mut cursor = match anchor {
            Anchor::Start => x,
            Anchor::Middle => x - width / 2.0,
            Anchor::End => x - width,
        };
        let charmap = self.font.charmap();
        let metrics = self.font.glyph_metrics(Size::unscaled(), LocationRef::default());
        let outlines = self.font.outline_glyphs();
        let mut d = String::new();
        for c in text.chars() {
            let Some(gid) = charmap.map(c).or_else(|| charmap.map('?')) else { continue };
            let advance = metrics.advance_width(gid).unwrap_or(upem * 0.5) * scale;
            if let Some(glyph) = outlines.get(gid) {
                let mut pen = Pen { out: &mut d, scale, x: cursor, y };
                let _ = glyph.draw(DrawSettings::unhinted(Size::unscaled(), LocationRef::default()), &mut pen);
            }
            cursor += advance;
        }
        d
    }

    /// `text` cut to fit `max` pixels, with an ellipsis where it was cut.
    pub fn fit(&self, text: &str, size: f32, max: f32) -> String {
        if self.width(text, size) <= max {
            return text.to_string();
        }
        let mut cut: Vec<char> = text.chars().collect();
        while !cut.is_empty() {
            cut.pop();
            let candidate: String = cut.iter().collect::<String>().trim_end().to_string() + "…";
            if self.width(&candidate, size) <= max {
                return candidate;
            }
        }
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_word_has_a_width_and_an_outline() {
        let f = regular();
        let w = f.width("Revenue", 12.0);
        assert!(w > 30.0 && w < 80.0, "{w}");
        assert!(f.width("Revenue", 24.0) > 1.9 * w);
        let d = f.path("Hi", 12.0, 10.0, 20.0, Anchor::Start);
        assert!(d.starts_with('M') && d.contains('Z'), "{d}");
        assert!(f.path("", 12.0, 0.0, 0.0, Anchor::Start).is_empty());
        // the three anchors put the same word at three places
        let at = |a| f.path("Hi", 12.0, 100.0, 20.0, a);
        assert!(at(Anchor::Start) != at(Anchor::Middle) && at(Anchor::Middle) != at(Anchor::End));
    }

    #[test]
    fn a_long_label_is_cut_with_an_ellipsis() {
        let f = regular();
        let cut = f.fit("A very long label of a bar", 12.0, 60.0);
        assert!(cut.ends_with('…') && f.width(&cut, 12.0) <= 60.0, "{cut}");
        assert_eq!(f.fit("Short", 12.0, 60.0), "Short");
        assert_eq!(f.fit("Anything", 12.0, 2.0), "");
    }
}
