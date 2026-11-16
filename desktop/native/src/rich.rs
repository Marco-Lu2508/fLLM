//! Rich blocks of an answer: typeset math, charts, diagrams and SVG pictures, drawn in the chat and
//! on the Code page like the answers of ChatGPT or Claude show them.
//!
//! Everything here ends in an SVG that the window shows with its `svg` widget, so it is sharp at any
//! size and takes the colors of the theme. The work (typesetting, layout) happens once per source
//! and theme: `art` keeps the results, the page is drawn from them every frame.
//!
//! * `math` — `latex-rust`, a TeX-faithful layout with the STIX Two Math font: fractions, roots,
//!   sums and integrals with limits, matrices, `cases`, `align`, accents, delimiters that grow.
//! * `chart` (in `chart.rs`) — a block ```chart with a JSON like Chart.js takes: bar, line, area,
//!   pie, doughnut, scatter, radar.
//! * `diagram` — ```mermaid: flowcharts, sequence, class, state, ER, Gantt, pie, mindmap and more
//!   (`mermaid-rs-renderer`).
//! * a block ```svg is shown as the picture it is.

use crate::theme::Tokens;
use iced::widget::svg;
use iced::Color;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex, OnceLock};

pub mod chart;
pub mod text;

/// A finished drawing: the SVG, its natural size and where the baseline of the text it stands in
/// is (math inside a line of words), all in logical pixels.
#[derive(Debug, Clone)]
pub struct Art {
    pub handle: svg::Handle,
    pub width: f32,
    pub height: f32,
    /// Distance from the top to the baseline; for a picture that has none: the height.
    pub baseline: f32,
    /// The SVG text itself (for *Save as SVG*).
    pub source: std::sync::Arc<str>,
}

impl Art {
    pub fn new(svg_text: String, width: f32, height: f32, baseline: f32) -> Art {
        Art { handle: svg::Handle::from_memory(svg_text.clone().into_bytes()), width, height, baseline, source: svg_text.into() }
    }
}

/// What the model is told when *Charts, diagrams and math in answers* is on: the formats the window
/// draws. It is short on purpose: it is part of every request.
pub const INSTRUCTIONS: &str = "The app draws these, so use them when a picture or a formula says it better than words, and not otherwise:\n- Math in LaTeX: $x^2$ inline, $$ ... $$ for a formula on its own line.\n- A chart in a ```chart block, JSON like Chart.js: {\"type\":\"bar|line|area|pie|doughnut|scatter|radar\",\"title\":\"...\",\"labels\":[...],\"series\":[{\"name\":\"...\",\"values\":[...]}]} (a pie or a bar chart of one series may use \"values\" directly).\n- A diagram in a ```mermaid block (flowchart, sequenceDiagram, classDiagram, stateDiagram-v2, erDiagram, gantt, pie, mindmap).\n- A picture in a ```svg block.";

/// How many results are kept (an answer has a handful; a long session has hundreds).
const KEEP: usize = 400;

type Cache = HashMap<String, Result<Art, String>>;

static CACHE: LazyLock<Mutex<Cache>> = LazyLock::new(|| Mutex::new(HashMap::new()));

/// The result for `key`, made by `make` the first time.
fn art(key: String, make: impl FnOnce() -> Result<Art, String>) -> Result<Art, String> {
    if let Some(hit) = CACHE.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        return hit.clone();
    }
    let made = make();
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if cache.len() >= KEEP {
        cache.clear();
    }
    cache.insert(key, made.clone());
    made
}

/// How many results are kept now (for the tests).
pub fn kept() -> usize {
    CACHE.lock().unwrap_or_else(|e| e.into_inner()).len()
}

/// The longest source that is typeset or drawn (more is shown as it is).
pub const MAX_SOURCE: usize = 20_000;

fn rgb(c: Color) -> latex_rust::Color {
    let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    latex_rust::Color::rgb(byte(c.r), byte(c.g), byte(c.b))
}

/// `#rrggbb` of a color, for SVG attributes.
pub fn hex(c: Color) -> String {
    let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}", byte(c.r), byte(c.g), byte(c.b))
}

fn font() -> Result<&'static latex_rust::MathFont, String> {
    static FONT: OnceLock<Result<latex_rust::MathFont, String>> = OnceLock::new();
    FONT.get_or_init(|| latex_rust::MathFont::stix_two_math().map_err(|e| e.to_string())).as_ref().map_err(Clone::clone)
}

/// Points of the SVG per pixel of the window: an SVG in `pt` is 4/3 as many pixels.
const PX_PER_PT: f32 = 4.0 / 3.0;

fn pt_to_px(d: &latex_rust::Dim) -> f32 {
    d.to_dec_string().parse::<f32>().unwrap_or(0.0) * PX_PER_PT
}

/// Typeset math. `size` is the size of the text around it in pixels; `display` sets it as a line of
/// its own (larger operators, limits above and below).
pub fn math(source: &str, display: bool, size: f32, color: Color) -> Result<Art, String> {
    let source = source.trim();
    if source.is_empty() {
        return Err("There is no math here.".into());
    }
    if source.chars().count() > MAX_SOURCE {
        return Err("This is too long to typeset.".into());
    }
    let key = format!("math\u{1}{display}\u{1}{size}\u{1}{}\u{1}{source}", hex(color));
    art(key, || {
        let font = font()?;
        let ast = latex_rust::parse(source).map_err(|e| e.to_string())?;
        let style = if display { latex_rust::MathStyle::Display } else { latex_rust::MathStyle::Text };
        // an em of the font is the size of the text: `size` pixels are `size * 3/4` points
        let em_pt = latex_rust::Dim::parse(&format!("{:.3}", size / PX_PER_PT));
        let tree = latex_rust::layout_with_em_size_pt(&ast, font, style, &em_pt).map_err(|e| e.to_string())?;
        let options = latex_rust::SvgOptions { font_size_pt: em_pt, color: rgb(color), display };
        let image = latex_rust::render_svg_with_metrics(&tree, font, &options).map_err(|e| e.to_string())?;
        let (w, h, b) = (pt_to_px(&image.width_pt), pt_to_px(&image.height_pt), pt_to_px(&image.baseline_pt));
        if !(w > 0.0 && h > 0.0) {
            return Err("There is nothing to draw.".into());
        }
        // the viewport of the SVG is exactly the ink: a rounded pixel at its edge would cut a stroke,
        // so it gets a margin of its own on every side (the baseline moves down with the top)
        let (svg, margin) = with_margin(&image.svg, 1.2);
        Ok(Art::new(svg, w + 2.0 * margin * PX_PER_PT, h + 2.0 * margin * PX_PER_PT, b + margin * PX_PER_PT))
    })
}

/// The SVG of `latex-rust` with `margin` points more room on every side; returns it with the margin used
/// (0 when its header was not what it should be).
fn with_margin(svg: &str, margin: f32) -> (String, f32) {
    static HEADER: OnceLock<regex::Regex> = OnceLock::new();
    let re = HEADER.get_or_init(|| {
        regex::Regex::new(r#"width="([\d.]+)pt" height="([\d.]+)pt" viewBox="(-?[\d.]+) (-?[\d.]+) ([\d.]+) ([\d.]+)""#).expect("a fixed pattern")
    });
    let Some(c) = re.captures(svg) else { return (svg.to_string(), 0.0) };
    let n = |i: usize| c[i].parse::<f32>().unwrap_or(0.0);
    let (w, h, x, y, vw, vh) = (n(1), n(2), n(3), n(4), n(5), n(6));
    let header = format!(
        r#"width="{}pt" height="{}pt" viewBox="{} {} {} {}""#,
        w + 2.0 * margin,
        h + 2.0 * margin,
        x - margin,
        y - margin,
        vw + 2.0 * margin,
        vh + 2.0 * margin
    );
    (svg.replacen(&c[0], &header, 1), margin)
}

/// A Mermaid diagram.
pub fn diagram(source: &str, t: Tokens, dark: bool) -> Result<Art, String> {
    let source = source.trim();
    if source.is_empty() {
        return Err("There is no diagram here.".into());
    }
    if source.chars().count() > MAX_SOURCE {
        return Err("This is too long to draw.".into());
    }
    let key = format!("mermaid\u{1}{dark}\u{1}{}\u{1}{}\u{1}{source}", hex(t.fg), hex(t.primary));
    art(key, || diagram_svg(source, t, dark).map(|svg| text_to_paths(&svg)).and_then(|svg| from_svg(&svg)))
}

/// The colors of a Mermaid diagram from the colors of the window.
fn mermaid_theme(t: Tokens) -> mermaid_rs_renderer::Theme {
    let mix = |a: Color, b: Color, k: f32| Color::from_rgb(a.r * (1.0 - k) + b.r * k, a.g * (1.0 - k) + b.g * k, a.b * (1.0 - k) + b.b * k);
    let mut th = if t.dark { mermaid_rs_renderer::Theme::dark() } else { mermaid_rs_renderer::Theme::modern() };
    let node = mix(t.surface, t.primary, if t.dark { 0.22 } else { 0.10 });
    let edge = mix(t.surface, t.primary, 0.55);
    let sub = mix(t.surface, t.fg, 0.05);
    th.font_family = "Inter, 'Segoe UI', Helvetica, Arial, sans-serif".into();
    th.background = "none".to_string();
    th.primary_color = hex(node);
    th.primary_text_color = hex(t.fg);
    th.primary_border_color = hex(edge);
    th.line_color = hex(t.fg_soft);
    th.text_color = hex(t.fg);
    th.secondary_color = hex(mix(t.surface, t.fg, 0.08));
    th.tertiary_color = hex(sub);
    th.edge_label_background = hex(t.surface);
    th.cluster_background = hex(sub);
    th.cluster_border = hex(mix(t.surface, t.fg, 0.25));
    th.sequence_actor_fill = hex(node);
    th.sequence_actor_border = hex(edge);
    th.sequence_actor_line = hex(mix(t.surface, t.fg, 0.4));
    th.sequence_note_fill = hex(mix(t.surface, t.primary, 0.18));
    th.sequence_note_border = hex(edge);
    th.sequence_activation_fill = hex(sub);
    th.sequence_activation_border = hex(edge);
    th
}

fn diagram_svg(source: &str, t: Tokens, _dark: bool) -> Result<String, String> {
    // the engine of the crate may panic on a source it does not understand: that is an error here
    let source = source.to_string();
    let theme = mermaid_theme(t);
    std::panic::catch_unwind(move || {
        let options = mermaid_rs_renderer::RenderOptions { theme, ..Default::default() };
        mermaid_rs_renderer::render_with_options(&source, options).map_err(|e| e.to_string())
    })
    .map_err(|_| "This diagram could not be drawn.".to_string())?
}

/// A picture of an SVG the model wrote: its size is the one it names, or its `viewBox`.
pub fn picture(source: &str) -> Result<Art, String> {
    let source = source.trim();
    if source.chars().count() > MAX_SOURCE * 5 {
        return Err("This is too long to draw.".into());
    }
    let key = format!("svg\u{1}{source}");
    art(key, || from_svg(source))
}

/// The size of an SVG as the window draws it, and the SVG ready to be shown. Nothing in it is run: a
/// script, a link to a file or an address is not followed by the renderer of the window.
pub fn from_svg(source: &str) -> Result<Art, String> {
    let tree = usvg::Tree::from_str(source, &usvg::Options::default()).map_err(|e| format!("This is not a picture the window can draw: {e}"))?;
    let size = tree.size();
    let (w, h) = (size.width(), size.height());
    if !(w >= 1.0 && h >= 1.0 && w.is_finite() && h.is_finite()) {
        return Err("This picture has no size.".into());
    }
    Ok(Art::new(source.to_string(), w, h, h))
}

fn re(cell: &'static OnceLock<regex::Regex>, pattern: &str) -> &'static regex::Regex {
    cell.get_or_init(|| regex::Regex::new(pattern).expect("a fixed pattern"))
}

fn decode_entities(s: &str) -> String {
    s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&#39;", "'").replace("&apos;", "'").replace("&amp;", "&")
}

/// Every `<text>` of an SVG becomes the outline of its words in the window's own font. A renderer
/// that draws SVG text needs the fonts of the system; this does not, and looks the same everywhere.
pub fn text_to_paths(svg: &str) -> String {
    static TEXT: OnceLock<regex::Regex> = OnceLock::new();
    static ATTR: OnceLock<regex::Regex> = OnceLock::new();
    static SPAN: OnceLock<regex::Regex> = OnceLock::new();
    static TAG: OnceLock<regex::Regex> = OnceLock::new();
    let text_re = re(&TEXT, r"(?s)<text\b([^>]*)>(.*?)</text>");
    let attr_re = re(&ATTR, r#"([A-Za-z:_-]+)\s*=\s*"([^"]*)""#);
    let span_re = re(&SPAN, r"(?s)<tspan\b([^>]*)>(.*?)</tspan>");
    let tag_re = re(&TAG, r"<[^>]*>");
    let attrs = |s: &str| -> HashMap<String, String> { attr_re.captures_iter(s).map(|c| (c[1].to_string(), c[2].to_string())).collect() };
    let number = |m: &HashMap<String, String>, k: &str| m.get(k).and_then(|v| v.trim().trim_end_matches("px").parse::<f32>().ok());
    text_re
        .replace_all(svg, |c: &regex::Captures| {
            let a = attrs(&c[1]);
            let size = number(&a, "font-size").unwrap_or(14.0);
            let bold = a.get("font-weight").is_some_and(|w| w == "bold" || w.parse::<u32>().is_ok_and(|n| n >= 600));
            let anchor = match a.get("text-anchor").map(String::as_str) {
                Some("middle") => text::Anchor::Middle,
                Some("end") => text::Anchor::End,
                _ => text::Anchor::Start,
            };
            let fill = a.get("fill").cloned().unwrap_or_else(|| "#000000".into());
            let (x0, mut y) = (number(&a, "x").unwrap_or(0.0), number(&a, "y").unwrap_or(0.0));
            // the text hangs from the middle of its line when the renderer asked for it
            let shift = match a.get("dominant-baseline").map(String::as_str) {
                Some("middle") | Some("central") => size * 0.35,
                Some("hanging") => size * 0.8,
                _ => 0.0,
            };
            let face = if bold { text::semibold() } else { text::regular() };
            let mut d = String::new();
            let mut line = |x: f32, y: f32, s: &str| {
                let s = decode_entities(s.trim());
                if !s.is_empty() {
                    d.push_str(&face.path(&s, size, x, y + shift, anchor));
                }
            };
            if span_re.is_match(&c[2]) {
                for s in span_re.captures_iter(&c[2]) {
                    let sa = attrs(&s[1]);
                    y += number(&sa, "dy").unwrap_or(0.0);
                    if let Some(sy) = number(&sa, "y") {
                        y = sy;
                    }
                    line(number(&sa, "x").unwrap_or(x0), y, &tag_re.replace_all(&s[2], ""));
                }
            } else {
                line(x0, y, &tag_re.replace_all(&c[2], ""));
            }
            if d.is_empty() {
                return String::new();
            }
            let extra: String = ["transform", "opacity", "fill-opacity"].iter().filter_map(|k| a.get(*k).map(|v| format!(" {k}=\"{v}\""))).collect();
            format!("<path d=\"{d}\" fill=\"{fill}\"{extra}/>")
        })
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_viewport_of_a_formula_has_room_around_its_ink() {
        let tight = r#"<svg xmlns="http://www.w3.org/2000/svg" width="10pt" height="5pt" viewBox="0.0 0.0 10 5"><g/></svg>"#;
        let (out, margin) = with_margin(tight, 1.0);
        assert_eq!(margin, 1.0);
        assert!(out.contains(r#"width="12pt" height="7pt" viewBox="-1 -1 12 7""#), "{out}");
        assert_eq!(with_margin("<svg/>", 1.0), ("<svg/>".to_string(), 0.0), "something else is left alone");
        let a = math("x", false, 16.0, Color::BLACK).unwrap();
        let tree = usvg::Tree::from_str(&a.source, &usvg::Options::default()).unwrap();
        let ink = tree.root().abs_bounding_box();
        assert!(ink.top() >= 0.9 && ink.bottom() <= tree.size().height() - 0.9 && ink.left() >= 0.9, "{ink:?} in {:?}", tree.size());
        assert!((a.width - tree.size().width()).abs() < 0.01 && (a.height - tree.size().height()).abs() < 0.01);
    }

    #[test]
    fn the_text_of_an_svg_becomes_outlines() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="40"><text x="50" y="20" text-anchor="middle" font-size="14" fill="#112233"><tspan x="50" dy="0">Hello &amp; bye</tspan></text><text x="5" y="35" font-size="10" transform="rotate(-10 5 35)">Two</text></svg>"##;
        let out = text_to_paths(svg);
        assert!(!out.contains("<text") && !out.contains("tspan"), "{out}");
        assert_eq!(out.matches("<path").count(), 2);
        assert!(out.contains("fill=\"#112233\"") && out.contains("transform=\"rotate(-10 5 35)\""));
        assert!(from_svg(&out).is_ok(), "still a picture");
        // an SVG with no text is left alone
        let plain = r#"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10"/></svg>"#;
        assert_eq!(text_to_paths(plain), plain);
        // an empty text leaves nothing behind
        assert!(!text_to_paths(r#"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><text x="1" y="2"> </text></svg>"#).contains("<path"));
    }
}
