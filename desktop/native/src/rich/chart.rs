//! Charts: a fenced block ```chart holds a JSON like the one Chart.js takes, and the window draws it
//! as ChatGPT or Claude would show a chart — bars, lines, areas, pies, doughnuts, scatter plots and
//! radar charts with axes, grid, legend and values.
//!
//! The words of a chart are paths of the window's own font (`text.rs`), the colors come from the
//! theme, and what the model writes is never put into the SVG as markup: numbers are numbers,
//! colors are checked, text becomes outlines.
//!
//! Accepted, in the order they are looked for:
//! * Chart.js: `{"type":"bar","data":{"labels":[..],"datasets":[{"label":"..","data":[..]}]},"options":{..}}`
//! * short: `{"type":"bar","title":"..","labels":[..],"values":[..]}` or `"series":[{"name":"..","values":[..]}]`
//! * pairs: `{"type":"pie","data":[{"label":"A","value":3},{"label":"B","value":5}]}` or `[["A",3],["B",5]]`

use super::text::{regular, semibold, Anchor};
use super::Art;
use crate::theme::Tokens;
use iced::Color;
use serde_json::Value;
use std::fmt::Write;

/// The most points a series and the most series a chart draws.
pub const MAX_POINTS: usize = 500;
pub const MAX_SERIES: usize = 24;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Bar,
    Line,
    Pie,
    Scatter,
    Radar,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Series {
    pub name: String,
    /// One value for every label (`None`: a gap).
    pub values: Vec<Option<f64>>,
    /// The points of a scatter plot.
    pub points: Vec<(f64, f64)>,
    pub color: Option<String>,
    /// A color for every value (pie slices, single bars).
    pub colors: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Spec {
    pub kind: Kind,
    pub title: String,
    pub labels: Vec<String>,
    pub series: Vec<Series>,
    pub stacked: bool,
    pub horizontal: bool,
    pub area: bool,
    pub smooth: bool,
    pub doughnut: bool,
    pub x_title: String,
    pub y_title: String,
    pub legend: Option<bool>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    /// Added to the numbers that are written out (`€`, ` ms`, `%`).
    pub unit: String,
}

/// The colors of the series when the chart names none.
fn palette(t: Tokens) -> [Color; 8] {
    let c = |r: u8, g: u8, b: u8| Color::from_rgb8(r, g, b);
    if t.dark {
        [t.primary, c(0xfb, 0x92, 0x3c), c(0x34, 0xd3, 0x99), c(0xc0, 0x84, 0xfc), c(0xf8, 0x71, 0x71), c(0x22, 0xd3, 0xee), c(0xfa, 0xcc, 0x15), c(0xf4, 0x72, 0xb6)]
    } else {
        [t.primary, c(0xea, 0x7a, 0x0c), c(0x0f, 0xa9, 0x7a), c(0x93, 0x4c, 0xe0), c(0xe5, 0x48, 0x48), c(0x08, 0x91, 0xb2), c(0xca, 0x9a, 0x04), c(0xdb, 0x2a, 0x84)]
    }
}

// ---- reading the JSON ----------------------------------------------------------------------------

fn text_of(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.trim().to_string()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

fn number_of(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64().filter(|f| f.is_finite()),
        Value::String(s) => s.trim().trim_end_matches('%').replace(',', ".").parse::<f64>().ok().filter(|f| f.is_finite()),
        _ => None,
    }
}

/// A color that is safe to put in an attribute: `#rgb`, `#rrggbb`, `rgb(..)`, `rgba(..)` or a plain name.
pub fn clean_color(s: &str) -> Option<String> {
    let s = s.trim();
    let hex = s.strip_prefix('#').is_some_and(|h| matches!(h.len(), 3 | 4 | 6 | 8) && h.bytes().all(|b| b.is_ascii_hexdigit()));
    let func = (s.starts_with("rgb(") || s.starts_with("rgba(") || s.starts_with("hsl(") || s.starts_with("hsla(")) && s.ends_with(')') && s.bytes().all(|b| b.is_ascii_digit() || b" (),.%-+abcdefghilnorsg".contains(&b));
    let name = !s.is_empty() && s.len() <= 24 && s.bytes().all(|b| b.is_ascii_alphabetic());
    (hex || func || name).then(|| s.to_string())
}

fn colors_of(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::String(s)) => clean_color(s).into_iter().collect(),
        Some(Value::Array(a)) => a.iter().filter_map(|c| c.as_str().and_then(clean_color)).collect(),
        _ => Vec::new(),
    }
}

fn series_values(v: &Value) -> Vec<Option<f64>> {
    v.as_array().map(|a| a.iter().take(MAX_POINTS).map(number_of).collect()).unwrap_or_default()
}

fn points_of(v: &Value) -> Vec<(f64, f64)> {
    v.as_array()
        .map(|a| {
            a.iter()
                .take(MAX_POINTS)
                .filter_map(|p| match p {
                    Value::Object(o) => Some((number_of(o.get("x")?)?, number_of(o.get("y")?)?)),
                    Value::Array(xy) if xy.len() >= 2 => Some((number_of(&xy[0])?, number_of(&xy[1])?)),
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default()
}

/// What a type name stands for.
struct Shape {
    kind: Kind,
    horizontal: bool,
    stacked: bool,
    area: bool,
    smooth: bool,
    doughnut: bool,
}

fn shape_of(name: &str) -> Result<Shape, String> {
    let n: String = name.to_ascii_lowercase().chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    let s = |kind, horizontal, stacked, area, smooth, doughnut| Shape { kind, horizontal, stacked, area, smooth, doughnut };
    Ok(match n.as_str() {
        "bar" | "column" | "columns" | "histogram" => s(Kind::Bar, false, false, false, false, false),
        "horizontalbar" | "hbar" | "barh" | "barhorizontal" => s(Kind::Bar, true, false, false, false, false),
        "stackedbar" | "stackedcolumn" | "barstacked" => s(Kind::Bar, false, true, false, false, false),
        "line" => s(Kind::Line, false, false, false, false, false),
        "spline" | "smooth" => s(Kind::Line, false, false, false, true, false),
        "area" | "filledline" => s(Kind::Line, false, false, true, false, false),
        "stackedarea" => s(Kind::Line, false, true, true, false, false),
        "pie" => s(Kind::Pie, false, false, false, false, false),
        "doughnut" | "donut" | "ring" => s(Kind::Pie, false, false, false, false, true),
        "scatter" | "bubble" | "points" => s(Kind::Scatter, false, false, false, false, false),
        "radar" | "spider" | "polararea" => s(Kind::Radar, false, false, false, false, false),
        _ => return Err(format!("The chart type \"{name}\" is not known. Use bar, line, area, pie, doughnut, scatter or radar.")),
    })
}

/// Reads the text of a chart block.
pub fn parse(source: &str) -> Result<Spec, String> {
    let v: Value = serde_json::from_str(source.trim()).map_err(|e| format!("The chart is not valid JSON: {e}"))?;
    from_value(&v)
}

/// Whether the text fails only because it stops too early (an answer that is still being written).
pub fn unfinished(source: &str) -> bool {
    serde_json::from_str::<Value>(source.trim()).is_err_and(|e| e.is_eof())
}

fn from_value(v: &Value) -> Result<Spec, String> {
    let obj = v.as_object().ok_or("The chart has to be a JSON object.")?;
    let type_name = obj.get("type").and_then(Value::as_str).unwrap_or("bar");
    let Shape { kind, mut horizontal, mut stacked, area, smooth, doughnut } = shape_of(type_name)?;
    let options = obj.get("options");
    let opt = |path: &[&str]| -> Option<&Value> {
        let mut cur = options?;
        for p in path {
            cur = cur.get(*p)?;
        }
        Some(cur)
    };
    let title = obj
        .get("title")
        .and_then(text_of)
        .or_else(|| opt(&["plugins", "title", "text"]).and_then(text_of))
        .or_else(|| opt(&["title", "text"]).and_then(text_of))
        .unwrap_or_default();
    if opt(&["indexAxis"]).and_then(Value::as_str) == Some("y") || obj.get("horizontal").and_then(Value::as_bool) == Some(true) {
        horizontal = true;
    }
    if opt(&["scales", "x", "stacked"]).and_then(Value::as_bool) == Some(true) || opt(&["scales", "y", "stacked"]).and_then(Value::as_bool) == Some(true) || obj.get("stacked").and_then(Value::as_bool) == Some(true) {
        stacked = true;
    }
    let smooth = smooth || obj.get("smooth").and_then(Value::as_bool) == Some(true);
    let data = obj.get("data");
    let mut labels: Vec<String> = data
        .and_then(|d| d.get("labels"))
        .or_else(|| obj.get("labels"))
        .or_else(|| obj.get("categories"))
        .and_then(Value::as_array)
        .map(|a| a.iter().take(MAX_POINTS).map(|l| text_of(l).unwrap_or_default()).collect())
        .unwrap_or_default();
    let mut series: Vec<Series> = Vec::new();
    let datasets = data.and_then(|d| d.get("datasets")).or_else(|| obj.get("datasets")).or_else(|| obj.get("series")).and_then(Value::as_array);
    if let Some(sets) = datasets {
        for (i, set) in sets.iter().take(MAX_SERIES).enumerate() {
            let name = set.get("label").or_else(|| set.get("name")).or_else(|| set.get("title")).and_then(text_of).unwrap_or_else(|| format!("Series {}", i + 1));
            let raw = set.get("data").or_else(|| set.get("values")).or_else(|| set.get("y")).unwrap_or(&Value::Null);
            let points = if kind == Kind::Scatter { points_of(raw) } else { Vec::new() };
            let color_value = set.get("backgroundColor").or_else(|| set.get("color")).or_else(|| set.get("borderColor"));
            let mut colors = colors_of(color_value);
            if colors.is_empty() {
                colors = colors_of(obj.get("colors"));
            }
            let own = if colors.len() == 1 { colors.first().cloned() } else { None };
            series.push(Series { name, values: if kind == Kind::Scatter { Vec::new() } else { series_values(raw) }, points, color: own, colors: if colors.len() > 1 { colors } else { Vec::new() } });
        }
    } else {
        // a single series: `values`, or `data` as numbers, objects or pairs
        let raw = obj.get("values").or(data).unwrap_or(&Value::Null);
        let colors = colors_of(obj.get("colors"));
        match raw.as_array() {
            Some(items) if !items.is_empty() && kind == Kind::Scatter => series.push(Series { name: String::new(), values: Vec::new(), points: points_of(raw), color: None, colors }),
            Some(items) if !items.is_empty() && items.iter().all(Value::is_object) => {
                let mut values = Vec::new();
                for item in items.iter().take(MAX_POINTS) {
                    let Some(o) = item.as_object() else { continue };
                    let label = o.get("label").or_else(|| o.get("name")).or_else(|| o.get("x")).or_else(|| o.get("category")).and_then(text_of).unwrap_or_default();
                    let value = o.get("value").or_else(|| o.get("y")).or_else(|| o.get("count")).and_then(number_of);
                    if labels.len() < values.len() + 1 {
                        labels.push(label);
                    }
                    values.push(value);
                }
                series.push(Series { name: String::new(), values, points: Vec::new(), color: None, colors });
            }
            Some(items) if !items.is_empty() && kind != Kind::Scatter && items.iter().all(|i| i.as_array().is_some_and(|p| p.len() == 2)) => {
                let mut values = Vec::new();
                for item in items.iter().take(MAX_POINTS) {
                    let Some(p) = item.as_array() else { continue };
                    if labels.len() < values.len() + 1 {
                        labels.push(text_of(&p[0]).unwrap_or_default());
                    }
                    values.push(number_of(&p[1]));
                }
                series.push(Series { name: String::new(), values, points: Vec::new(), color: None, colors });
            }
            Some(_) => series.push(Series { name: String::new(), values: series_values(raw), points: Vec::new(), color: None, colors }),
            None => {}
        }
    }
    series.retain(|s| s.values.iter().any(Option::is_some) || !s.points.is_empty());
    if series.is_empty() {
        return Err("The chart has no numbers to draw. Give it \"data\" (labels and datasets) or \"labels\" and \"values\".".into());
    }
    if kind != Kind::Scatter {
        let n = series.iter().map(|s| s.values.len()).max().unwrap_or(0);
        while labels.len() < n {
            labels.push((labels.len() + 1).to_string());
        }
        labels.truncate(n);
    }
    let axis_title = |axis: &str| opt(&["scales", axis, "title", "text"]).and_then(text_of).or_else(|| obj.get(&format!("{axis}Label")).and_then(text_of)).or_else(|| obj.get(&format!("{axis}_label")).and_then(text_of)).unwrap_or_default();
    Ok(Spec {
        kind,
        title,
        labels,
        series,
        stacked,
        horizontal,
        area,
        smooth,
        doughnut,
        x_title: axis_title("x"),
        y_title: axis_title("y"),
        legend: opt(&["plugins", "legend", "display"]).and_then(Value::as_bool).or_else(|| obj.get("legend").and_then(Value::as_bool)),
        min: opt(&["scales", "y", "min"]).and_then(number_of).or_else(|| obj.get("min").and_then(number_of)),
        max: opt(&["scales", "y", "max"]).and_then(number_of).or_else(|| obj.get("max").and_then(number_of)),
        unit: obj.get("unit").and_then(text_of).unwrap_or_default(),
    })
}

// ---- numbers ---------------------------------------------------------------------------------------

/// A number as it is written on a chart: `1.2k`, `3.5M`, `0.25`.
pub fn format_number(v: f64) -> String {
    let a = v.abs();
    let (scaled, suffix) = if a >= 1e9 {
        (v / 1e9, "B")
    } else if a >= 1e6 {
        (v / 1e6, "M")
    } else if a >= 1e4 {
        (v / 1e3, "k")
    } else {
        (v, "")
    };
    let mut s = if scaled.fract().abs() < 1e-9 || scaled.abs() >= 100.0 {
        format!("{scaled:.0}")
    } else if scaled.abs() >= 10.0 {
        format!("{scaled:.1}")
    } else {
        format!("{scaled:.2}")
    };
    if s.contains('.') {
        s = s.trim_end_matches('0').trim_end_matches('.').to_string();
    }
    if s == "-0" {
        s = "0".into();
    }
    format!("{s}{suffix}")
}

/// Tick values from `lo` to `hi`: round numbers, about `count` of them.
pub fn nice_ticks(lo: f64, hi: f64, count: usize) -> Vec<f64> {
    if !(lo.is_finite() && hi.is_finite()) {
        return vec![0.0, 1.0];
    }
    let (lo, hi) = if (hi - lo).abs() < 1e-12 { (lo.min(0.0), lo.max(0.0).max(1.0)) } else { (lo, hi) };
    let raw = (hi - lo) / count.max(1) as f64;
    let mag = 10f64.powf(raw.log10().floor());
    let step = [1.0, 2.0, 2.5, 5.0, 10.0].iter().map(|m| m * mag).find(|s| *s >= raw * 0.999).unwrap_or(10.0 * mag);
    let start = (lo / step).floor() * step;
    let mut out = Vec::new();
    let mut v = start;
    while v <= hi + step * 0.999 && out.len() < 50 {
        out.push((v / step).round() * step);
        if v >= hi - step * 1e-9 {
            break;
        }
        v += step;
    }
    if out.len() < 2 {
        out.push(start + step);
    }
    out
}

// ---- drawing ---------------------------------------------------------------------------------------

struct Canvas {
    w: f32,
    body: String,
    t: Tokens,
}

fn f(v: f32) -> String {
    let s = format!("{v:.2}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

impl Canvas {
    #[allow(clippy::too_many_arguments)]
    fn text(&mut self, s: &str, size: f32, x: f32, y: f32, anchor: Anchor, fill: &str, bold: bool) {
        if s.is_empty() {
            return;
        }
        let face = if bold { semibold() } else { regular() };
        let d = face.path(s, size, x, y, anchor);
        if !d.is_empty() {
            let _ = write!(self.body, "<path d=\"{d}\" fill=\"{fill}\"/>");
        }
    }

    /// Text with a halo of the surface color, so a grid line behind it does not strike it through.
    fn label(&mut self, s: &str, size: f32, x: f32, y: f32, anchor: Anchor, fill: &str) {
        if s.is_empty() {
            return;
        }
        let d = regular().path(s, size, x, y, anchor);
        if !d.is_empty() {
            let halo = super::hex(self.t.surface);
            let _ = write!(self.body, "<path d=\"{d}\" fill=\"{fill}\" stroke=\"{halo}\" stroke-width=\"3.5\" stroke-linejoin=\"round\" paint-order=\"stroke\"/>");
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn rotated(&mut self, s: &str, size: f32, x: f32, y: f32, angle: f32, anchor: Anchor, fill: &str) {
        let d = regular().path(s, size, x, y, anchor);
        if !d.is_empty() {
            let _ = write!(self.body, "<path d=\"{d}\" fill=\"{fill}\" transform=\"rotate({} {} {})\"/>", f(angle), f(x), f(y));
        }
    }

    fn line(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, stroke: &str, width: f32) {
        let _ = write!(self.body, "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{stroke}\" stroke-width=\"{}\"/>", f(x1), f(y1), f(x2), f(y2), f(width));
    }

    /// A rectangle with its own radius at every corner (top-left, top-right, bottom-right, bottom-left).
    fn rect(&mut self, x: f32, y: f32, w: f32, h: f32, fill: &str, radius: [f32; 4]) {
        if w <= 0.0 || h <= 0.0 {
            return;
        }
        let [tl, tr, br, bl] = radius.map(|r| r.min(w / 2.0).min(h / 2.0).max(0.0));
        let _ = write!(
            self.body,
            "<path d=\"M{} {}H{}Q{} {} {} {}V{}Q{} {} {} {}H{}Q{} {} {} {}V{}Q{} {} {} {}Z\" fill=\"{fill}\"/>",
            f(x + tl),
            f(y),
            f(x + w - tr),
            f(x + w),
            f(y),
            f(x + w),
            f(y + tr),
            f(y + h - br),
            f(x + w),
            f(y + h),
            f(x + w - br),
            f(y + h),
            f(x + bl),
            f(x),
            f(y + h),
            f(x),
            f(y + h - bl),
            f(y + tl),
            f(x),
            f(y),
            f(x + tl),
            f(y),
        );
    }
}

fn series_color(spec: &Spec, i: usize, pal: &[Color; 8]) -> String {
    spec.series.get(i).and_then(|s| s.color.clone()).unwrap_or_else(|| super::hex(pal[i % pal.len()]))
}

fn point_color(s: &Series, i: usize, pal: &[Color; 8]) -> String {
    s.colors.get(i).cloned().unwrap_or_else(|| super::hex(pal[i % pal.len()]))
}

/// Whether a `#rrggbb` color is light enough for dark text on it.
fn is_light(color: &str) -> bool {
    let h = color.trim_start_matches('#');
    if h.len() != 6 {
        return false;
    }
    let ch = |i: usize| u8::from_str_radix(&h[i..i + 2], 16).map(f32::from).unwrap_or(0.0) / 255.0;
    0.2126 * ch(0) + 0.7152 * ch(2) + 0.0722 * ch(4) > 0.62
}

fn with_unit(spec: &Spec, v: f64) -> String {
    format!("{}{}", format_number(v), spec.unit)
}

/// The SVG of a chart and its size.
pub fn draw(spec: &Spec, t: Tokens) -> (String, f32, f32) {
    let pal = palette(t);
    let w = 640.0f32;
    let h = match spec.kind {
        Kind::Pie => 330.0,
        Kind::Radar => 380.0,
        Kind::Bar if spec.horizontal => {
            let rows = spec.labels.len() as f32 * if spec.stacked { 1.0 } else { spec.series.len().max(1) as f32 * 0.7 };
            (150.0 + 34.0 * rows).clamp(220.0, 900.0)
        }
        _ => 360.0,
    };
    let fg = super::hex(t.fg);
    let soft = super::hex(t.fg_soft);
    let muted = super::hex(t.muted);
    // the line of the grid: the separator of the theme laid over the surface (an opaque, calm grey)
    let a = t.separator.a;
    let grid = super::hex(Color::from_rgb(t.surface.r * (1.0 - a) + t.separator.r * a, t.surface.g * (1.0 - a) + t.separator.g * a, t.surface.b * (1.0 - a) + t.separator.b * a));
    let mut c = Canvas { w, body: String::new(), t };
    let legend_wanted = spec.legend.unwrap_or(spec.kind != Kind::Pie && spec.series.len() > 1);
    let mut top = 14.0f32;
    if !spec.title.is_empty() {
        let title = semibold().fit(&spec.title, 15.0, w - 40.0);
        c.text(&title, 15.0, w / 2.0, top + 14.0, Anchor::Middle, &fg, true);
        top += 30.0;
    }
    if legend_wanted && spec.kind != Kind::Pie {
        let items: Vec<(String, String)> = spec.series.iter().enumerate().map(|(i, s)| (s.name.clone(), series_color(spec, i, &pal))).collect();
        top = legend_row(&mut c, &items, top, &soft);
    }
    let h = match spec.kind {
        Kind::Bar | Kind::Line | Kind::Scatter => cartesian(&mut c, spec, &pal, top, h, (&soft, &muted, &grid)),
        Kind::Pie => {
            pie(&mut c, spec, &pal, top, h, (&fg, &soft, &muted));
            h
        }
        Kind::Radar => {
            radar(&mut c, spec, &pal, top, h, (&soft, &muted, &grid));
            h
        }
    };
    let svg = format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">{}</svg>", f(w), f(h), f(w), f(h), c.body);
    (svg, w, h)
}

fn legend_row(c: &mut Canvas, items: &[(String, String)], top: f32, text_color: &str) -> f32 {
    let size = 12.0;
    let mut rows: Vec<Vec<(String, String, f32)>> = vec![Vec::new()];
    let mut x = 0.0;
    for (name, color) in items {
        let name = regular().fit(name, size, 150.0);
        let width = 16.0 + regular().width(&name, size) + 18.0;
        if x + width > c.w - 40.0 && rows.last().is_some_and(|r| !r.is_empty()) {
            rows.push(Vec::new());
            x = 0.0;
        }
        if let Some(row) = rows.last_mut() {
            row.push((name, color.clone(), width));
        }
        x += width;
    }
    let mut y = top + 10.0;
    for row in &rows {
        let total: f32 = row.iter().map(|r| r.2).sum::<f32>() - 18.0;
        let mut x = (c.w - total) / 2.0;
        for (name, color, width) in row {
            let _ = write!(c.body, "<rect x=\"{}\" y=\"{}\" width=\"10\" height=\"10\" rx=\"3\" fill=\"{color}\"/>", f(x), f(y - 9.0));
            c.text(name, size, x + 16.0, y, Anchor::Start, text_color, false);
            x += width;
        }
        y += 20.0;
    }
    y - 4.0
}

type Colors<'a> = (&'a str, &'a str, &'a str);

/// Bars, lines and scatter plots share the frame: axes, ticks, grid. Returns the height used.
fn cartesian(c: &mut Canvas, spec: &Spec, pal: &[Color; 8], top: f32, h: f32, colors: Colors) -> f32 {
    let (soft, muted, grid) = colors;
    let scatter = spec.kind == Kind::Scatter;
    let n = if scatter { 0 } else { spec.labels.len().max(1) };
    let horizontal = spec.kind == Kind::Bar && spec.horizontal;
    // the value range
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let (mut xlo, mut xhi) = (f64::INFINITY, f64::NEG_INFINITY);
    if scatter {
        for s in &spec.series {
            for (x, y) in &s.points {
                lo = lo.min(*y);
                hi = hi.max(*y);
                xlo = xlo.min(*x);
                xhi = xhi.max(*x);
            }
        }
    } else if spec.stacked {
        for i in 0..n {
            let (mut pos, mut neg) = (0.0f64, 0.0f64);
            for s in &spec.series {
                let v = s.values.get(i).copied().flatten().unwrap_or(0.0);
                if v >= 0.0 {
                    pos += v;
                } else {
                    neg += v;
                }
            }
            hi = hi.max(pos);
            lo = lo.min(neg);
        }
    } else {
        for s in &spec.series {
            for v in s.values.iter().flatten() {
                lo = lo.min(*v);
                hi = hi.max(*v);
            }
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        lo = 0.0;
        hi = 1.0;
    }
    if scatter {
        let room = (hi - lo).max(1e-9) * 0.06;
        lo -= room;
        hi += room;
        let xroom = (xhi - xlo).max(1e-9) * 0.06;
        xlo -= xroom;
        xhi += xroom;
    }
    // bars and areas start at zero
    if spec.kind == Kind::Bar || spec.area {
        lo = lo.min(0.0);
        hi = hi.max(0.0);
    }
    if let Some(m) = spec.min {
        lo = m;
    }
    if let Some(m) = spec.max {
        hi = m;
    }
    if hi <= lo {
        hi = lo + 1.0;
    }
    let ticks = nice_ticks(lo, hi, 5);
    let (tmin, tmax) = (*ticks.first().unwrap_or(&lo), *ticks.last().unwrap_or(&hi));
    let vmin = if spec.min.is_some() { lo } else { tmin.min(lo) };
    let vmax = if spec.max.is_some() { hi } else { tmax.max(hi) };
    let span = (vmax - vmin).max(1e-12);
    let label_font = 11.5f32;

    // the room the labels need
    let value_labels: Vec<String> = ticks.iter().map(|v| with_unit(spec, *v)).collect();
    let widest_value = value_labels.iter().map(|s| regular().width(s, label_font)).fold(0.0f32, f32::max);
    let cat_labels: Vec<String> = if scatter { Vec::new() } else { spec.labels.clone() };
    let widest_cat = cat_labels.iter().map(|s| regular().width(s, label_font)).fold(0.0f32, f32::max);
    let left = if horizontal { (widest_cat.min(170.0) + 34.0).max(60.0) } else { widest_value + 30.0 };
    let right = 24.0f32;
    let plot_w = c.w - left - right;
    let x_title_room = if spec.x_title.is_empty() { 0.0 } else { 22.0 };
    let y_title_room = if spec.y_title.is_empty() { 0.0 } else { 18.0 };
    let band_w = plot_w / n.max(1) as f32;
    let rotate = !horizontal && !scatter && cat_labels.iter().any(|l| regular().width(l, label_font) > band_w - 6.0);
    let bottom_labels = if horizontal { 26.0 } else if rotate { 28.0 + widest_cat.min(110.0) * 0.55 } else { 30.0 };
    let top = top + 8.0 + y_title_room + if spec.kind == Kind::Bar && !horizontal { 8.0 } else { 0.0 };
    let plot_h = (h - top - bottom_labels - x_title_room - 10.0).max(80.0);
    let new_h = top + plot_h + bottom_labels + x_title_room + 10.0;
    let (px, py) = (left, top);
    let y_of = |v: f64| py + plot_h - ((v - vmin) / span) as f32 * plot_h;
    let x_of_value = |v: f64| px + ((v - vmin) / span) as f32 * plot_w;

    // grid and value axis
    for (tick, label) in ticks.iter().zip(&value_labels) {
        if horizontal {
            let x = x_of_value(*tick);
            if x < px - 0.5 || x > px + plot_w + 0.5 {
                continue;
            }
            c.line(x, py, x, py + plot_h, grid, 1.0);
            c.text(label, label_font, x, py + plot_h + 18.0, Anchor::Middle, muted, false);
        } else {
            let y = y_of(*tick);
            if y < py - 0.5 || y > py + plot_h + 0.5 {
                continue;
            }
            let zero = tick.abs() < 1e-12;
            c.line(px, y, px + plot_w, y, grid, if zero { 1.5 } else { 1.0 });
            c.text(label, label_font, px - 8.0, y + 4.0, Anchor::End, muted, false);
        }
    }
    if !spec.y_title.is_empty() {
        c.text(&spec.y_title, 12.0, px, py - 12.0, Anchor::Start, soft, false);
    }
    if !spec.x_title.is_empty() {
        c.text(&spec.x_title, 12.0, px + plot_w / 2.0, new_h - 8.0, Anchor::Middle, soft, false);
    }

    match spec.kind {
        Kind::Bar => {
            let groups = if spec.stacked { 1 } else { spec.series.len().max(1) };
            for i in 0..n {
                let name = cat_labels.get(i).cloned().unwrap_or_default();
                let label = regular().fit(&name, label_font, if horizontal { left - 22.0 } else { band_w - 4.0 });
                let thickness = if horizontal { plot_h / n as f32 } else { band_w } * 0.72;
                let each = thickness / groups as f32;
                let centre = if horizontal { py + plot_h / n as f32 * (i as f32 + 0.5) } else { px + band_w * (i as f32 + 0.5) };
                if horizontal {
                    c.text(&label, label_font, px - 8.0, centre + 4.0, Anchor::End, soft, false);
                } else if rotate {
                    c.rotated(&name, label_font, centre + 4.0, py + plot_h + 16.0, -35.0, Anchor::End, soft);
                } else {
                    c.text(&label, label_font, centre, py + plot_h + 18.0, Anchor::Middle, soft, false);
                }
                let (mut pos_base, mut neg_base) = (0.0f64, 0.0f64);
                for (si, s) in spec.series.iter().enumerate() {
                    let Some(v) = s.values.get(i).copied().flatten() else { continue };
                    let color = if spec.series.len() == 1 && !s.colors.is_empty() {
                        point_color(s, i, pal)
                    } else {
                        series_color(spec, si, pal)
                    };
                    let (from, to) = if spec.stacked {
                        if v >= 0.0 {
                            pos_base += v;
                            (pos_base - v, pos_base)
                        } else {
                            neg_base += v;
                            (neg_base - v, neg_base)
                        }
                    } else {
                        (0.0f64.clamp(vmin, vmax), v.clamp(vmin, vmax))
                    };
                    let outer = !spec.stacked || si + 1 == spec.series.len();
                    let r = if outer { 4.0 } else { 0.0 };
                    let offset = if spec.stacked { 0.0 } else { each * si as f32 };
                    if horizontal {
                        let (x0, x1) = (x_of_value(from.min(to)), x_of_value(from.max(to)));
                        let y0 = centre - thickness / 2.0 + offset;
                        let rad = if v >= 0.0 { [0.0, r, r, 0.0] } else { [r, 0.0, 0.0, r] };
                        c.rect(x0, y0 + 1.0, x1 - x0, each - 2.0, &color, rad);
                        if n * groups <= 40 && !spec.stacked {
                            let (x, anchor) = if v >= 0.0 { (x1 + 6.0, Anchor::Start) } else { (x0 - 6.0, Anchor::End) };
                            c.label(&with_unit(spec, v), 11.0, x, y0 + each / 2.0 + 4.0, anchor, soft);
                        }
                    } else {
                        let (y0, y1) = (y_of(from.max(to)), y_of(from.min(to)));
                        let x0 = centre - thickness / 2.0 + offset;
                        let rad = if v >= 0.0 { [r, r, 0.0, 0.0] } else { [0.0, 0.0, r, r] };
                        c.rect(x0 + 1.0, y0, each - 2.0, y1 - y0, &color, rad);
                        if n * groups <= 36 && !spec.stacked && each > 18.0 {
                            let y = if v >= 0.0 { y0 - 5.0 } else { y1 + 13.0 };
                            if y > py - 16.0 {
                                c.label(&with_unit(spec, v), 11.0, x0 + each / 2.0, y, Anchor::Middle, soft);
                            }
                        }
                    }
                }
            }
            if horizontal {
                c.line(px, py, px, py + plot_h, grid, 1.0);
            }
        }
        Kind::Line => {
            let pad = 18.0f32.min(plot_w / 8.0);
            let step = if n > 1 { (plot_w - 2.0 * pad) / (n as f32 - 1.0) } else { 0.0 };
            let x_at = |i: usize| if n > 1 { px + pad + step * i as f32 } else { px + plot_w / 2.0 };
            // labels: thinned out when they would overlap
            let every = (((widest_cat + 10.0) / step.max(1.0)).ceil().max(1.0)) as usize;
            for (i, label) in cat_labels.iter().enumerate() {
                if i % every == 0 {
                    c.text(label, label_font, x_at(i), py + plot_h + 18.0, Anchor::Middle, soft, false);
                }
            }
            let mut base = vec![0.0f64; n];
            let surface = super::hex(c.t.surface);
            for (si, s) in spec.series.iter().enumerate() {
                let color = series_color(spec, si, pal);
                let mut tops = vec![f64::NAN; n];
                let mut pts: Vec<Option<(f32, f32)>> = Vec::with_capacity(n);
                for (i, top_value) in tops.iter_mut().enumerate() {
                    let v = s.values.get(i).copied().flatten();
                    if let Some(v) = v {
                        *top_value = if spec.stacked { base[i] + v } else { v };
                    }
                    pts.push(v.map(|_| (x_at(i), y_of(top_value.clamp(vmin, vmax)))));
                }
                let floor_y = |i: usize| if spec.stacked { y_of(base[i].clamp(vmin, vmax)) } else { y_of(0.0f64.clamp(vmin, vmax)) };
                // runs of points without a gap
                let mut runs: Vec<Vec<(usize, (f32, f32))>> = Vec::new();
                for (i, p) in pts.iter().enumerate() {
                    if let Some(p) = p {
                        if runs.last().and_then(|r| r.last()).is_none_or(|l| l.0 + 1 != i) {
                            runs.push(Vec::new());
                        }
                        if let Some(run) = runs.last_mut() {
                            run.push((i, *p));
                        }
                    }
                }
                for run in &runs {
                    let path = curve(&run.iter().map(|r| r.1).collect::<Vec<_>>(), spec.smooth);
                    if spec.area {
                        let mut d = path.clone();
                        if spec.stacked {
                            for (i, (x, _)) in run.iter().rev() {
                                let _ = write!(d, "L{} {}", f(*x), f(floor_y(*i)));
                            }
                        } else if let (Some(first), Some(last)) = (run.first(), run.last()) {
                            let _ = write!(d, "L{} {}L{} {}", f(last.1 .0), f(floor_y(last.0)), f(first.1 .0), f(floor_y(first.0)));
                        }
                        d.push('Z');
                        let _ = write!(c.body, "<path d=\"{d}\" fill=\"{color}\" fill-opacity=\"0.18\"/>");
                    }
                    if run.len() > 1 {
                        let _ = write!(c.body, "<path d=\"{path}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"2.5\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>");
                    }
                }
                if n <= 60 {
                    for (x, y) in pts.iter().flatten() {
                        let _ = write!(c.body, "<circle cx=\"{}\" cy=\"{}\" r=\"3.6\" fill=\"{surface}\" stroke=\"{color}\" stroke-width=\"2\"/>", f(*x), f(*y));
                    }
                }
                if n <= 14 && spec.series.len() <= 2 && !spec.stacked {
                    for (i, p) in pts.iter().enumerate() {
                        if let (Some((x, y)), Some(v)) = (p, s.values.get(i).copied().flatten()) {
                            c.label(&with_unit(spec, v), 11.0, *x, *y - 9.0, Anchor::Middle, soft);
                        }
                    }
                }
                if spec.stacked {
                    for (b, t) in base.iter_mut().zip(&tops) {
                        if t.is_finite() {
                            *b = *t;
                        }
                    }
                }
            }
        }
        Kind::Scatter => {
            let (mut xa, mut xb) = (xlo, xhi);
            if !xa.is_finite() || !xb.is_finite() {
                xa = 0.0;
                xb = 1.0;
            }
            let xt = nice_ticks(xa, xb, 6);
            let xmin = xt.first().copied().unwrap_or(xa).min(xa);
            let xmax = xt.last().copied().unwrap_or(xb).max(xb);
            let xspan = (xmax - xmin).max(1e-12);
            for tick in &xt {
                let x = px + ((tick - xmin) / xspan) as f32 * plot_w;
                if x < px - 0.5 || x > px + plot_w + 0.5 {
                    continue;
                }
                c.line(x, py, x, py + plot_h, grid, 1.0);
                c.text(&format_number(*tick), label_font, x, py + plot_h + 18.0, Anchor::Middle, muted, false);
            }
            for (si, s) in spec.series.iter().enumerate() {
                let color = series_color(spec, si, pal);
                for (x, y) in &s.points {
                    let (cx, cy) = (px + ((x - xmin) / xspan) as f32 * plot_w, y_of(*y));
                    let _ = write!(c.body, "<circle cx=\"{}\" cy=\"{}\" r=\"4.4\" fill=\"{color}\" fill-opacity=\"0.78\" stroke=\"{color}\" stroke-width=\"1\"/>", f(cx), f(cy));
                }
            }
        }
        _ => {}
    }
    new_h
}

/// A path through points: straight pieces, or a smooth curve (Catmull-Rom as Béziers).
fn curve(points: &[(f32, f32)], smooth: bool) -> String {
    let mut d = String::new();
    for (i, (x, y)) in points.iter().enumerate() {
        if i == 0 {
            let _ = write!(d, "M{} {}", f(*x), f(*y));
        } else if smooth && points.len() > 2 {
            let p0 = points[i.saturating_sub(2)];
            let p1 = points[i - 1];
            let p2 = points[i];
            let p3 = points[(i + 1).min(points.len() - 1)];
            let (c1x, c1y) = (p1.0 + (p2.0 - p0.0) / 6.0, p1.1 + (p2.1 - p0.1) / 6.0);
            let (c2x, c2y) = (p2.0 - (p3.0 - p1.0) / 6.0, p2.1 - (p3.1 - p1.1) / 6.0);
            let _ = write!(d, "C{} {} {} {} {} {}", f(c1x), f(c1y), f(c2x), f(c2y), f(p2.0), f(p2.1));
        } else {
            let _ = write!(d, "L{} {}", f(*x), f(*y));
        }
    }
    d
}

fn pie(c: &mut Canvas, spec: &Spec, pal: &[Color; 8], top: f32, h: f32, colors: Colors) {
    let (fg, soft, muted) = colors;
    let s = &spec.series[0];
    let values: Vec<(usize, f64)> = s.values.iter().enumerate().filter_map(|(i, v)| v.filter(|v| *v > 0.0).map(|v| (i, v))).collect();
    let total: f64 = values.iter().map(|v| v.1).sum();
    if total <= 0.0 {
        c.text("No positive values to draw.", 13.0, c.w / 2.0, h / 2.0, Anchor::Middle, muted, false);
        return;
    }
    let r = ((h - top - 24.0) / 2.0).clamp(60.0, 120.0);
    let legend_w = values
        .iter()
        .map(|(i, v)| 18.0 + regular().width(spec.labels.get(*i).map(String::as_str).unwrap_or(""), 13.0).min(190.0) + 24.0 + regular().width(&with_unit(spec, *v), 12.0))
        .fold(0.0f32, f32::max);
    let block = 2.0 * r + 40.0 + legend_w;
    let cx = ((c.w - block) / 2.0).max(24.0) + r;
    let cy = top + 10.0 + r;
    let inner = if spec.doughnut { r * 0.6 } else { 0.0 };
    let mut angle = -std::f32::consts::FRAC_PI_2;
    let surface = super::hex(c.t.surface);
    for (i, v) in &values {
        let sweep = (*v / total) as f32 * std::f32::consts::TAU;
        let color = point_color(s, *i, pal);
        let (a0, a1) = (angle, angle + sweep.min(std::f32::consts::TAU - 0.0001));
        let (x0, y0, x1, y1) = (cx + r * a0.cos(), cy + r * a0.sin(), cx + r * a1.cos(), cy + r * a1.sin());
        let large = i32::from(sweep > std::f32::consts::PI);
        let d = if inner > 0.0 {
            let (ix0, iy0, ix1, iy1) = (cx + inner * a0.cos(), cy + inner * a0.sin(), cx + inner * a1.cos(), cy + inner * a1.sin());
            format!("M{} {}A{} {} 0 {large} 1 {} {}L{} {}A{} {} 0 {large} 0 {} {}Z", f(x0), f(y0), f(r), f(r), f(x1), f(y1), f(ix1), f(iy1), f(inner), f(inner), f(ix0), f(iy0))
        } else {
            format!("M{} {}L{} {}A{} {} 0 {large} 1 {} {}Z", f(cx), f(cy), f(x0), f(y0), f(r), f(r), f(x1), f(y1))
        };
        let _ = write!(c.body, "<path d=\"{d}\" fill=\"{color}\" stroke=\"{surface}\" stroke-width=\"2\" stroke-linejoin=\"round\"/>");
        let share = *v / total;
        if share >= 0.05 {
            let mid = (a0 + a1) / 2.0;
            let rr = if inner > 0.0 { (r + inner) / 2.0 } else { r * 0.64 };
            let on_color = if is_light(&color) { "#14141a" } else { "#ffffff" };
            c.text(&format!("{:.0}%", share * 100.0), 12.0, cx + rr * mid.cos(), cy + rr * mid.sin() + 4.0, Anchor::Middle, on_color, true);
        }
        angle = a1;
    }
    if spec.doughnut {
        c.text(&with_unit(spec, total), 18.0, cx, cy + 2.0, Anchor::Middle, fg, true);
        c.text("Total", 11.5, cx, cy + 18.0, Anchor::Middle, muted, false);
    }
    // the legend to the right: dot, label, value
    let lx = cx + r + 40.0;
    let rows = values.len();
    let row_h = 24.0f32.min((h - top - 20.0) / rows.max(1) as f32);
    let mut y = cy - row_h * rows as f32 / 2.0 + 12.0;
    for (i, v) in &values {
        let color = point_color(s, *i, pal);
        let _ = write!(c.body, "<rect x=\"{}\" y=\"{}\" width=\"11\" height=\"11\" rx=\"3.5\" fill=\"{color}\"/>", f(lx), f(y - 10.0));
        let name = regular().fit(spec.labels.get(*i).map(String::as_str).unwrap_or(""), 13.0, 190.0);
        c.text(&name, 13.0, lx + 18.0, y, Anchor::Start, soft, false);
        c.text(&with_unit(spec, *v), 12.0, lx + legend_w, y, Anchor::End, muted, false);
        y += row_h;
    }
}

fn radar(c: &mut Canvas, spec: &Spec, pal: &[Color; 8], top: f32, h: f32, colors: Colors) {
    let (soft, muted, grid) = colors;
    let n = spec.labels.len().max(3);
    let max = spec.max.unwrap_or_else(|| spec.series.iter().flat_map(|s| s.values.iter().flatten().copied()).fold(0.0f64, f64::max)).max(1e-9);
    let ticks = nice_ticks(0.0, max, 4);
    let top_tick = ticks.last().copied().unwrap_or(max).max(max);
    let r = ((h - top - 60.0) / 2.0).max(60.0);
    let (cx, cy) = (c.w / 2.0, top + 30.0 + r);
    let at = |i: usize, frac: f32| {
        let a = -std::f32::consts::FRAC_PI_2 + std::f32::consts::TAU * i as f32 / n as f32;
        (cx + r * frac * a.cos(), cy + r * frac * a.sin())
    };
    for tick in ticks.iter().skip(1) {
        let frac = (*tick / top_tick) as f32;
        let ring: Vec<String> = (0..n)
            .map(|i| {
                let (x, y) = at(i, frac);
                format!("{} {}", f(x), f(y))
            })
            .collect();
        let _ = write!(c.body, "<polygon points=\"{}\" fill=\"none\" stroke=\"{grid}\" stroke-width=\"1\"/>", ring.join(" "));
    }
    for i in 0..n {
        let (x, y) = at(i, 1.0);
        c.line(cx, cy, x, y, grid, 1.0);
        let dx = x - cx;
        let anchor = if dx.abs() < 8.0 { Anchor::Middle } else if dx > 0.0 { Anchor::Start } else { Anchor::End };
        let label = spec.labels.get(i).map(String::as_str).unwrap_or("");
        let pad = if dx.abs() < 8.0 { 0.0 } else if dx > 0.0 { 8.0 } else { -8.0 };
        let lift = if y < cy - 4.0 { -8.0 } else if y > cy + 4.0 { 16.0 } else { 4.0 };
        c.text(&regular().fit(label, 12.0, 110.0), 12.0, x + pad, y + lift, anchor, soft, false);
    }
    let shapes: Vec<(String, Vec<(f32, f32)>)> = spec
        .series
        .iter()
        .enumerate()
        .map(|(si, s)| (series_color(spec, si, pal), (0..n).map(|i| at(i, (s.values.get(i).copied().flatten().unwrap_or(0.0) / top_tick).clamp(0.0, 1.0) as f32)).collect()))
        .collect();
    let points = |pts: &[(f32, f32)]| pts.iter().map(|(x, y)| format!("{} {}", f(*x), f(*y))).collect::<Vec<_>>().join(" ");
    // the fills first, so no fill paints over the outline of another series
    for (color, pts) in &shapes {
        let _ = write!(c.body, "<polygon points=\"{}\" fill=\"{color}\" fill-opacity=\"0.16\"/>", points(pts));
    }
    for (color, pts) in &shapes {
        let _ = write!(c.body, "<polygon points=\"{}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"2.4\" stroke-linejoin=\"round\"/>", points(pts));
        for (x, y) in pts {
            let _ = write!(c.body, "<circle cx=\"{}\" cy=\"{}\" r=\"3.2\" fill=\"{color}\"/>", f(*x), f(*y));
        }
    }
    // the numbers of the rings stand on top of everything, beside the first axis
    for tick in ticks.iter().skip(1) {
        let (x, y) = at(0, (*tick / top_tick) as f32);
        c.label(&format_number(*tick), 10.5, x + 6.0, y + 4.0, Anchor::Start, muted);
    }
}

/// A chart from the text of a block: parsed, drawn and ready for the window.
pub fn render(source: &str, t: Tokens) -> Result<Art, String> {
    let key = format!("chart\u{1}{}\u{1}{}\u{1}{source}", t.dark, super::hex(t.primary));
    super::art(key, || {
        let spec = parse(source)?;
        let (svg, w, h) = draw(&spec, t);
        Ok(Art::new(svg, w, h, h))
    })
}
