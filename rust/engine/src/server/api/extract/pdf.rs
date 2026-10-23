//! Text from the text layer of a PDF. The file is scanned for its objects (no cross-reference table needed, so
//! damaged files and incremental updates still work), object streams are unpacked, the page tree gives the page
//! order and each page's content stream is interpreted for its text-showing operators. Fonts are mapped to
//! Unicode through their `ToUnicode` CMap or `Differences` list, else bytes are read as Latin-1. Line breaks come
//! from the vertical movement of the text position; the order is the order of the content stream.
use super::inflate::inflate_zlib;
use std::{collections::HashMap, rc::Rc};

const STREAM_LIMIT: usize = 64 << 20;
const TEXT_LIMIT: usize = 8 << 20;

#[derive(Clone, Debug)]
enum Obj {
    Null,
    Bool,
    Num(f64),
    Name(String),
    Str(Vec<u8>),
    Arr(Vec<Obj>),
    Dict(Dict),
    Ref(u32),
    Op(String),
}

type Dict = HashMap<String, Obj>;
static NULL: Obj = Obj::Null;

#[derive(Debug)]
pub(super) struct Extracted {
    pub text: String,
    pub pages: usize,
    pub truncated: bool,
}

struct Parser<'a> {
    data: &'a [u8],
    pos: usize,
    refs: bool,
}

fn is_space(b: u8) -> bool {
    matches!(b, 0 | 9 | 10 | 12 | 13 | 32)
}

fn is_delim(b: u8) -> bool {
    matches!(b, b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'/' | b'%')
}

impl Parser<'_> {
    fn skip_space(&mut self) {
        while let Some(&b) = self.data.get(self.pos) {
            if is_space(b) {
                self.pos += 1;
            } else if b == b'%' {
                while self.data.get(self.pos).is_some_and(|&c| c != b'\n' && c != b'\r') {
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
    }

    fn word(&mut self) -> &[u8] {
        let start = self.pos;
        while self.data.get(self.pos).is_some_and(|&b| !is_space(b) && !is_delim(b)) {
            self.pos += 1;
        }
        &self.data[start..self.pos]
    }

    fn parse(&mut self, depth: usize) -> Option<Obj> {
        if depth > 40 {
            return None;
        }
        self.skip_space();
        let b = *self.data.get(self.pos)?;
        match b {
            b'/' => {
                self.pos += 1;
                let raw = self.word().to_vec();
                let mut name = Vec::new();
                let mut i = 0;
                while i < raw.len() {
                    if raw[i] == b'#' && i + 2 < raw.len() &&raw.get(i + 1..i + 3).is_some_and(|h| h.iter().all(u8::is_ascii_hexdigit)) {
                        name.push(u8::from_str_radix(std::str::from_utf8(&raw[i + 1..i + 3]).ok()?, 16).ok()?);
                        i += 3;
                    } else {
                        name.push(raw[i]);
                        i += 1;
                    }
                }
                Some(Obj::Name(String::from_utf8_lossy(&name).into_owned()))
            }
            b'(' => {
                self.pos += 1;
                let mut out = Vec::new();
                let mut level = 1;
                while let Some(&c) = self.data.get(self.pos) {
                    self.pos += 1;
                    match c {
                        b'\\' => {
                            let e = *self.data.get(self.pos)?;
                            self.pos += 1;
                            match e {
                                b'n' => out.push(b'\n'),
                                b'r' => out.push(b'\r'),
                                b't' => out.push(b'\t'),
                                b'b' => out.push(8),
                                b'f' => out.push(12),
                                b'0'..=b'7' => {
                                    let mut v = u32::from(e - b'0');
                                    for _ in 0..2 {
                                        match self.data.get(self.pos) {
                                            Some(&d @ b'0'..=b'7') => {
                                                v = v * 8 + u32::from(d - b'0');
                                                self.pos += 1;
                                            }
                                            _ => break,
                                        }
                                    }
                                    out.push(v as u8);
                                }
                                b'\r' => {
                                    if self.data.get(self.pos) == Some(&b'\n') {
                                        self.pos += 1;
                                    }
                                }
                                b'\n' => {}
                                other => out.push(other),
                            }
                        }
                        b'(' => {
                            level += 1;
                            out.push(c);
                        }
                        b')' => {
                            level -= 1;
                            if level == 0 {
                                return Some(Obj::Str(out));
                            }
                            out.push(c);
                        }
                        _ => out.push(c),
                    }
                }
                None
            }
            b'<' if self.data.get(self.pos + 1) == Some(&b'<') => {
                self.pos += 2;
                let mut dict = Dict::new();
                loop {
                    self.skip_space();
                    if self.data.get(self.pos..self.pos + 2) == Some(b">>") {
                        self.pos += 2;
                        return Some(Obj::Dict(dict));
                    }
                    let Obj::Name(key) = self.parse(depth + 1)? else { return None };
                    let value = self.parse(depth + 1)?;
                    dict.insert(key, value);
                }
            }
            b'<' => {
                self.pos += 1;
                let mut digits = Vec::new();
                while let Some(&c) = self.data.get(self.pos) {
                    self.pos += 1;
                    if c == b'>' {
                        break;
                    }
                    if c.is_ascii_hexdigit() {
                        digits.push(c);
                    }
                }
                if digits.len() % 2 == 1 {
                    digits.push(b'0');
                }
                Some(Obj::Str(digits.chunks(2).filter_map(|p| u8::from_str_radix(std::str::from_utf8(p).ok()?, 16).ok()).collect()))
            }
            b'[' => {
                self.pos += 1;
                let mut list = Vec::new();
                loop {
                    self.skip_space();
                    if self.data.get(self.pos) == Some(&b']') {
                        self.pos += 1;
                        return Some(Obj::Arr(list));
                    }
                    list.push(self.parse(depth + 1)?);
                }
            }
            b')' | b'>' | b']' | b'{' | b'}' => {
                self.pos += 1;
                Some(Obj::Op(char::from(b).to_string()))
            }
            _ => {
                let word = self.word().to_vec();
                if word.is_empty() {
                    self.pos += 1;
                    return Some(Obj::Null);
                }
                let text = String::from_utf8_lossy(&word).into_owned();
                if let Ok(n) = text.parse::<f64>() {
                    if self.refs && text.bytes().all(|c| c.is_ascii_digit()) {
                        let save = self.pos;
                        self.skip_space();
                        let generation = self.word().to_vec();
                        self.skip_space();
                        if !generation.is_empty() && generation.iter().all(u8::is_ascii_digit) && self.data.get(self.pos) == Some(&b'R') && !self.data.get(self.pos + 1).is_some_and(|&c| !is_space(c) && !is_delim(c)) {
                            self.pos += 1;
                            return Some(Obj::Ref(n as u32));
                        }
                        self.pos = save;
                    }
                    return Some(Obj::Num(n));
                }
                Some(match text.as_str() {
                    "true" | "false" => Obj::Bool,
                    "null" => Obj::Null,
                    _ => Obj::Op(text),
                })
            }
        }
    }
}

fn find(data: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    data.get(from..)?.windows(needle.len()).position(|w| w == needle).map(|p| p + from)
}

struct Doc<'a> {
    data: &'a [u8],
    objs: HashMap<u32, Obj>,
    /// Stream data ranges of the objects that have a stream.
    streams: HashMap<u32, (usize, usize)>,
    roots: Vec<Obj>,
    encrypted: bool,
}

impl<'a> Doc<'a> {
    fn scan(data: &'a [u8]) -> Doc<'a> {
        let mut doc = Doc { data, objs: HashMap::new(), streams: HashMap::new(), roots: Vec::new(), encrypted: false };
        let mut i = 0;
        while let Some(p) = find(data, b"obj", i) {
            i = p + 3;
            if p == 0 || !is_space(data[p - 1]) || data.get(p + 3).is_some_and(u8::is_ascii_alphanumeric) {
                continue;
            }
            let mut j = p;
            let back = |j: &mut usize, digits: bool| {
                let end = *j;
                while *j > 0 && (if digits { data[*j - 1].is_ascii_digit() } else { is_space(data[*j - 1]) }) {
                    *j -= 1;
                }
                end - *j
            };
            if back(&mut j, false) == 0 {
                continue;
            }
            let gen_len = back(&mut j, true);
            if gen_len == 0 || back(&mut j, false) == 0 {
                continue;
            }
            let num_end = j;
            let num_len = back(&mut j, true);
            if num_len == 0 || num_len > 10 {
                continue;
            }
            let Some(id) = std::str::from_utf8(&data[num_end - num_len..num_end]).ok().and_then(|s| s.parse::<u32>().ok()) else { continue };
            let mut parser = Parser { data, pos: p + 3, refs: true };
            let Some(obj) = parser.parse(0) else { continue };
            parser.skip_space();
            let mut end = parser.pos;
            if let (Obj::Dict(dict), true) = (&obj, data.get(end..end + 6) == Some(b"stream")) {
                let mut start = end + 6;
                if data.get(start) == Some(&b'\r') {
                    start += 1;
                }
                if data.get(start) == Some(&b'\n') {
                    start += 1;
                }
                let declared = match dict.get("Length") {
                    Some(Obj::Num(n)) if *n >= 0.0 => Some(*n as usize),
                    _ => None,
                };
                let trusted = declared.filter(|&n| {
                    let mut k = start + n;
                    while data.get(k).is_some_and(|&b| is_space(b)) {
                        k += 1;
                    }
                    data.get(k..k + 9) == Some(b"endstream")
                });
                let stop = match trusted {
                    Some(n) => start + n,
                    None => {
                        let mut e = find(data, b"endstream", start).unwrap_or(data.len());
                        if e > start && data[e - 1] == b'\n' {
                            e -= 1;
                        }
                        if e > start && data[e - 1] == b'\r' {
                            e -= 1;
                        }
                        e
                    }
                };
                doc.streams.insert(id, (start, stop.min(data.len())));
                end = stop;
            }
            doc.objs.insert(id, obj);
            i = i.max(end);
        }
        // trailers of classic files; cross-reference streams are objects with the same keys
        let mut at = 0;
        while let Some(p) = find(data, b"trailer", at) {
            at = p + 7;
            let mut parser = Parser { data, pos: at, refs: true };
            if let Some(Obj::Dict(d)) = parser.parse(0) {
                doc.note_trailer(&d);
            }
        }
        let dicts: Vec<Dict> = doc.objs.values().filter_map(|o| if let Obj::Dict(d) = o { Some(d.clone()) } else { None }).collect();
        for d in &dicts {
            if matches!(d.get("Type"), Some(Obj::Name(n)) if n == "XRef") {
                doc.note_trailer(d);
            }
        }
        doc.unpack_object_streams();
        doc
    }

    fn note_trailer(&mut self, d: &Dict) {
        if d.contains_key("Encrypt") {
            self.encrypted = true;
        }
        if let Some(root) = d.get("Root") {
            self.roots.push(root.clone());
        }
    }

    fn unpack_object_streams(&mut self) {
        let ids: Vec<u32> = self
            .objs
            .iter()
            .filter(|(_, o)| matches!(o, Obj::Dict(d) if matches!(d.get("Type"), Some(Obj::Name(n)) if n == "ObjStm")))
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            let Some(Obj::Dict(dict)) = self.objs.get(&id).cloned() else { continue };
            let (Some(Obj::Num(n)), Some(Obj::Num(first))) = (dict.get("N"), dict.get("First")) else { continue };
            let Some(body) = self.stream_data(id) else { continue };
            let mut header = Parser { data: &body, pos: 0, refs: false };
            let mut table = Vec::new();
            for _ in 0..(*n as usize).min(1_000_000) {
                let (Some(Obj::Num(num)), Some(Obj::Num(off))) = (header.parse(0), header.parse(0)) else { break };
                table.push((num as u32, off as usize));
            }
            for (num, off) in table {
                let mut p = Parser { data: &body, pos: *first as usize + off, refs: true };
                if let Some(obj) = p.parse(0) {
                    self.objs.entry(num).or_insert(obj);
                }
            }
        }
    }

    fn resolve<'b>(&'b self, mut obj: &'b Obj) -> &'b Obj {
        for _ in 0..16 {
            match obj {
                Obj::Ref(id) => obj = self.objs.get(id).unwrap_or(&NULL),
                _ => return obj,
            }
        }
        &NULL
    }

    fn dict<'b>(&'b self, obj: &'b Obj) -> Option<&'b Dict> {
        match self.resolve(obj) {
            Obj::Dict(d) => Some(d),
            _ => None,
        }
    }

    fn get<'b>(&'b self, dict: &'b Dict, key: &str) -> &'b Obj {
        dict.get(key).map_or(&NULL, |o| self.resolve(o))
    }

    /// The decoded stream of object `id`, or `None` when it has none or uses an unsupported filter.
    fn stream_data(&self, id: u32) -> Option<Vec<u8>> {
        let &(start, end) = self.streams.get(&id)?;
        let Obj::Dict(dict) = self.objs.get(&id)? else { return None };
        let mut bytes = self.data.get(start..end)?.to_vec();
        let filters: Vec<String> = match self.get(dict, "Filter") {
            Obj::Name(n) => vec![n.clone()],
            Obj::Arr(a) => a.iter().filter_map(|f| if let Obj::Name(n) = self.resolve(f) { Some(n.clone()) } else { None }).collect(),
            _ => Vec::new(),
        };
        let parms: Vec<Option<&Dict>> = match self.get(dict, "DecodeParms") {
            Obj::Dict(d) => vec![Some(d)],
            Obj::Arr(a) => a.iter().map(|p| self.dict(p)).collect(),
            _ => Vec::new(),
        };
        for (index, filter) in filters.iter().enumerate() {
            match filter.as_str() {
                "FlateDecode" | "Fl" => {
                    let (out, _) = inflate_zlib(&bytes, STREAM_LIMIT);
                    if out.is_empty() && !bytes.is_empty() {
                        return None;
                    }
                    bytes = out;
                    if let Some(Some(p)) = parms.get(index) {
                        let predictor = if let Obj::Num(n) = self.get(p, "Predictor") { *n as usize } else { 1 };
                        if predictor >= 10 {
                            let columns = if let Obj::Num(n) = self.get(p, "Columns") { (*n as usize).max(1) } else { 1 };
                            let colors = if let Obj::Num(n) = self.get(p, "Colors") { (*n as usize).max(1) } else { 1 };
                            bytes = png_predictor(&bytes, columns * colors, colors);
                        }
                    }
                }
                "ASCIIHexDecode" | "AHx" => {
                    let digits: Vec<u8> = bytes.iter().copied().take_while(|&b| b != b'>').filter(u8::is_ascii_hexdigit).collect();
                    bytes = digits.chunks(2).filter_map(|p| u8::from_str_radix(std::str::from_utf8(p).ok()?, 16).ok().or_else(|| u8::from_str_radix(&format!("{}0", char::from(p[0])), 16).ok())).collect();
                }
                _ => return None,
            }
        }
        Some(bytes)
    }
}

/// Undoes the PNG row filters of a predictor-encoded stream (`row` bytes per row, `bpp` bytes per pixel).
fn png_predictor(data: &[u8], row: usize, bpp: usize) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(data.len());
    let mut previous = vec![0u8; row];
    for chunk in data.chunks(row + 1) {
        let kind = chunk[0];
        let mut cur: Vec<u8> = chunk[1..].to_vec();
        cur.resize(row, 0);
        for i in 0..row {
            let left = if i >= bpp { cur[i - bpp] } else { 0 };
            let up = previous[i];
            let up_left = if i >= bpp { previous[i - bpp] } else { 0 };
            let add = match kind {
                1 => left,
                2 => up,
                3 => ((u16::from(left) + u16::from(up)) / 2) as u8,
                4 => {
                    let (a, b, c) = (i32::from(left), i32::from(up), i32::from(up_left));
                    let p = a + b - c;
                    let (pa, pb, pc) = ((p - a).abs(), (p - b).abs(), (p - c).abs());
                    (if pa <= pb && pa <= pc { a } else if pb <= pc { b } else { c }) as u8
                }
                _ => 0,
            };
            cur[i] = cur[i].wrapping_add(add);
        }
        out.extend_from_slice(&cur);
        previous = cur;
    }
    out
}

#[derive(Default)]
struct Font {
    two_byte: bool,
    map: HashMap<u32, String>,
    differences: HashMap<u8, String>,
}

impl Font {
    fn decode(&self, bytes: &[u8], out: &mut String) {
        if self.two_byte {
            for pair in bytes.chunks(2) {
                let code = pair.iter().fold(0u32, |a, &b| a << 8 | u32::from(b));
                if let Some(s) = self.map.get(&code) {
                    out.push_str(s);
                }
            }
            return;
        }
        for &b in bytes {
            if let Some(s) = self.map.get(&u32::from(b)).or_else(|| self.differences.get(&b)) {
                out.push_str(s);
            } else if b >= 32 || b == b'\t' {
                out.push(cp1252(b));
            }
        }
    }
}

fn cp1252(b: u8) -> char {
    match b {
        0x80 => '\u{20ac}',
        0x85 => '\u{2026}',
        0x91 => '\u{2018}',
        0x92 => '\u{2019}',
        0x93 => '\u{201c}',
        0x94 => '\u{201d}',
        0x95 => '\u{2022}',
        0x96 => '\u{2013}',
        0x97 => '\u{2014}',
        0x99 => '\u{2122}',
        _ => char::from(b),
    }
}

fn glyph_text(name: &str) -> Option<String> {
    let mut chars = name.chars();
    if let (Some(c), None) = (chars.next(), chars.next()) {
        return Some(c.to_string());
    }
    if let Some(hex) = name.strip_prefix("uni").filter(|h| h.len() >= 4) {
        let units: Vec<u16> = hex.as_bytes().chunks(4).filter_map(|c| u16::from_str_radix(std::str::from_utf8(c).ok()?, 16).ok()).collect();
        return Some(String::from_utf16_lossy(&units));
    }
    let known = match name {
        "space" => " ",
        "hyphen" | "minus" => "-",
        "period" => ".",
        "comma" => ",",
        "colon" => ":",
        "semicolon" => ";",
        "slash" => "/",
        "backslash" => "\\",
        "parenleft" => "(",
        "parenright" => ")",
        "bracketleft" => "[",
        "bracketright" => "]",
        "braceleft" => "{",
        "braceright" => "}",
        "quotesingle" => "'",
        "quotedbl" => "\"",
        "quoteright" => "\u{2019}",
        "quoteleft" => "\u{2018}",
        "quotedblleft" => "\u{201c}",
        "quotedblright" => "\u{201d}",
        "endash" => "\u{2013}",
        "emdash" => "\u{2014}",
        "bullet" => "\u{2022}",
        "ellipsis" => "\u{2026}",
        "fi" => "fi",
        "fl" => "fl",
        "ff" => "ff",
        "ffi" => "ffi",
        "ffl" => "ffl",
        "exclam" => "!",
        "question" => "?",
        "ampersand" => "&",
        "percent" => "%",
        "dollar" => "$",
        "numbersign" => "#",
        "at" => "@",
        "asterisk" => "*",
        "plus" => "+",
        "equal" => "=",
        "less" => "<",
        "greater" => ">",
        "underscore" => "_",
        "zero" => "0",
        "one" => "1",
        "two" => "2",
        "three" => "3",
        "four" => "4",
        "five" => "5",
        "six" => "6",
        "seven" => "7",
        "eight" => "8",
        "nine" => "9",
        "adieresis" => "\u{e4}",
        "odieresis" => "\u{f6}",
        "udieresis" => "\u{fc}",
        "Adieresis" => "\u{c4}",
        "Odieresis" => "\u{d6}",
        "Udieresis" => "\u{dc}",
        "germandbls" => "\u{df}",
        "eacute" => "\u{e9}",
        "egrave" => "\u{e8}",
        "agrave" => "\u{e0}",
        _ => return None,
    };
    Some(known.to_owned())
}

enum CTok {
    Hex(Vec<u8>),
    Open,
    Close,
}

fn cmap_tokens(section: &str) -> Vec<CTok> {
    let bytes = section.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'<' => {
                let end = bytes[i..].iter().position(|&b| b == b'>').map_or(bytes.len(), |p| i + p);
                let digits: Vec<u8> = bytes[i + 1..end].iter().copied().filter(u8::is_ascii_hexdigit).collect();
                out.push(CTok::Hex(digits.chunks(2).filter_map(|p| u8::from_str_radix(std::str::from_utf8(p).ok()?, 16).ok()).collect()));
                i = end + 1;
            }
            b'[' => {
                out.push(CTok::Open);
                i += 1;
            }
            b']' => {
                out.push(CTok::Close);
                i += 1;
            }
            _ => i += 1,
        }
    }
    out
}

fn utf16(bytes: &[u8]) -> Vec<u16> {
    bytes.chunks(2).map(|p| p.iter().fold(0u16, |a, &b| a << 8 | u16::from(b))).collect()
}

fn code_of(bytes: &[u8]) -> u32 {
    bytes.iter().take(4).fold(0u32, |a, &b| a << 8 | u32::from(b))
}

fn parse_cmap(text: &str) -> HashMap<u32, String> {
    let mut map = HashMap::new();
    let mut rest = text;
    while let Some(start) = rest.find("beginbf") {
        let kind_range = rest[start + 7..].starts_with("range");
        let after = &rest[start + 7..];
        let end_marker = if kind_range { "endbfrange" } else { "endbfchar" };
        let Some(end) = after.find(end_marker) else { break };
        let tokens = cmap_tokens(&after[..end]);
        let mut it = tokens.into_iter().peekable();
        if kind_range {
            while let (Some(CTok::Hex(lo)), Some(CTok::Hex(hi))) = (it.next(), it.next()) {
                let (lo, hi) = (code_of(&lo), code_of(&hi));
                match it.next() {
                    Some(CTok::Hex(dst)) => {
                        let units = utf16(&dst);
                        for code in lo..=hi.min(lo.saturating_add(65_535)) {
                            let mut u = units.clone();
                            if let Some(last) = u.last_mut() {
                                *last = last.wrapping_add((code - lo) as u16);
                            }
                            map.insert(code, String::from_utf16_lossy(&u));
                        }
                    }
                    Some(CTok::Open) => {
                        let mut code = lo;
                        while let Some(CTok::Hex(dst)) = it.peek() {
                            map.insert(code, String::from_utf16_lossy(&utf16(dst)));
                            code += 1;
                            it.next();
                        }
                        it.next();
                    }
                    _ => break,
                }
            }
        } else {
            while let (Some(CTok::Hex(src)), Some(CTok::Hex(dst))) = (it.next(), it.next()) {
                map.insert(code_of(&src), String::from_utf16_lossy(&utf16(&dst)));
            }
        }
        rest = &after[end + end_marker.len()..];
    }
    map
}

type Matrix = [f64; 6];
const IDENTITY: Matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

/// `a` applied first, then `b`.
fn mul(a: &Matrix, b: &Matrix) -> Matrix {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
        a[4] * b[0] + a[5] * b[2] + b[4],
        a[4] * b[1] + a[5] * b[3] + b[5],
    ]
}

fn numbers(operands: &[Obj]) -> Vec<f64> {
    operands.iter().filter_map(|o| if let Obj::Num(n) = o { Some(*n) } else { None }).collect()
}

struct PageText {
    out: String,
    last: Option<(f64, f64)>,
    moved: bool,
}

impl PageText {
    /// Starts a fragment at text position (`x`, `y`) for text of height `eff`: a line break when the position
    /// moved down or up by more than part of the height, a space when it moved along the line.
    fn begin(&mut self, x: f64, y: f64, eff: f64) {
        if let Some((_, ly)) = self.last {
            if self.moved {
                let dy = (y - ly).abs();
                if dy > eff * 0.4 {
                    if !self.out.ends_with('\n') {
                        self.out.push('\n');
                    }
                    if dy > eff * 1.9 {
                        self.out.push('\n');
                    }
                } else if !self.out.ends_with([' ', '\n']) {
                    self.out.push(' ');
                }
            }
        }
        self.last = Some((x, y));
        self.moved = false;
    }
}

struct Interpreter<'d> {
    fonts: &'d HashMap<String, Rc<Font>>,
}

impl Interpreter<'_> {
    fn run(&self, content: &[u8]) -> String {
        let mut parser = Parser { data: content, pos: 0, refs: false };
        let mut operands: Vec<Obj> = Vec::new();
        let mut page = PageText { out: String::new(), last: None, moved: false };
        let (mut ctm, mut stack) = (IDENTITY, Vec::<Matrix>::new());
        let (mut tm, mut tlm) = (IDENTITY, IDENTITY);
        let (mut font, mut size, mut leading): (Option<Rc<Font>>, f64, f64) = (None, 10.0, 0.0);
        while let Some(obj) = parser.parse(0) {
            let Obj::Op(op) = obj else {
                operands.push(obj);
                if operands.len() > 4096 {
                    operands.clear();
                }
                continue;
            };
            let n = numbers(&operands);
            let mut shown: Option<Vec<(Vec<u8>, bool)>> = None;
            match op.as_str() {
                "q" => stack.push(ctm),
                "Q" => ctm = stack.pop().unwrap_or(IDENTITY),
                "cm" if n.len() == 6 => {
                    ctm = mul(&[n[0], n[1], n[2], n[3], n[4], n[5]], &ctm);
                    page.moved = true;
                }
                "BT" => {
                    tm = IDENTITY;
                    tlm = IDENTITY;
                }
                "Tf" => {
                    if let (Some(Obj::Name(name)), Some(Obj::Num(s))) = (operands.first(), operands.get(1)) {
                        font = self.fonts.get(name).cloned();
                        size = if s.abs() > 0.01 { s.abs() } else { 10.0 };
                    }
                }
                "TL" if !n.is_empty() => leading = n[0],
                "Tm" if n.len() == 6 => {
                    tm = [n[0], n[1], n[2], n[3], n[4], n[5]];
                    tlm = tm;
                    page.moved = true;
                }
                "Td" | "TD" if n.len() == 2 => {
                    if op == "TD" {
                        leading = -n[1];
                    }
                    tlm = mul(&[1.0, 0.0, 0.0, 1.0, n[0], n[1]], &tlm);
                    tm = tlm;
                    page.moved = true;
                }
                "T*" => {
                    tlm = mul(&[1.0, 0.0, 0.0, 1.0, 0.0, -leading], &tlm);
                    tm = tlm;
                    page.moved = true;
                }
                "Tj" => {
                    if let Some(Obj::Str(s)) = operands.first() {
                        shown = Some(vec![(s.clone(), false)]);
                    }
                }
                "'" | "\"" => {
                    tlm = mul(&[1.0, 0.0, 0.0, 1.0, 0.0, -leading], &tlm);
                    tm = tlm;
                    page.moved = true;
                    if let Some(Obj::Str(s)) = operands.last() {
                        shown = Some(vec![(s.clone(), false)]);
                    }
                }
                "TJ" => {
                    if let Some(Obj::Arr(items)) = operands.first() {
                        let mut parts = Vec::new();
                        for item in items {
                            match item {
                                Obj::Str(s) => parts.push((s.clone(), false)),
                                Obj::Num(k) if *k < -200.0 => parts.push((Vec::new(), true)),
                                _ => {}
                            }
                        }
                        shown = Some(parts);
                    }
                }
                "BI" => {
                    // inline image: skip its data up to "EI"
                    let mut at = parser.pos;
                    while let Some(p) = find(content, b"EI", at) {
                        let before = p == 0 || is_space(content[p - 1]);
                        let after = content.get(p + 2).is_none_or(|&b| is_space(b));
                        at = p + 2;
                        if before && after {
                            break;
                        }
                    }
                    parser.pos = at.min(content.len());
                }
                _ => {}
            }
            if let Some(parts) = shown {
                let mut text = String::new();
                for (bytes, gap) in &parts {
                    if *gap {
                        if !text.ends_with(' ') && !text.is_empty() {
                            text.push(' ');
                        }
                    } else if let Some(f) = &font {
                        f.decode(bytes, &mut text);
                    } else {
                        Font::default().decode(bytes, &mut text);
                    }
                }
                if !text.trim().is_empty() {
                    let x = tm[4] * ctm[0] + tm[5] * ctm[2] + ctm[4];
                    let y = tm[4] * ctm[1] + tm[5] * ctm[3] + ctm[5];
                    let eff = size * tm[3].hypot(tm[2]).max(0.01) * ctm[3].hypot(ctm[2]).max(0.01);
                    page.begin(x, y, eff);
                    page.out.push_str(&text);
                }
            }
            operands.clear();
        }
        page.out
    }
}

impl Doc<'_> {
    fn load_font(&self, obj: &Obj) -> Font {
        let mut font = Font::default();
        let Some(dict) = self.dict(obj) else { return font };
        font.two_byte = matches!(self.get(dict, "Subtype"), Obj::Name(n) if n == "Type0");
        if let Obj::Ref(id) = dict.get("ToUnicode").unwrap_or(&NULL) {
            if let Some(bytes) = self.stream_data(*id) {
                font.map = parse_cmap(&String::from_utf8_lossy(&bytes));
            }
        }
        if let Obj::Dict(enc) = self.get(dict, "Encoding") {
            if let Obj::Arr(diffs) = self.get(enc, "Differences") {
                let mut code = 0u32;
                for item in diffs {
                    match self.resolve(item) {
                        Obj::Num(n) => code = *n as u32,
                        Obj::Name(name) => {
                            if let (Some(text), Ok(c)) = (glyph_text(name), u8::try_from(code)) {
                                font.differences.insert(c, text);
                            }
                            code += 1;
                        }
                        _ => {}
                    }
                }
            }
        }
        font
    }

    fn collect_pages(&self, node: &Obj, resources: &Obj, out: &mut Vec<(Obj, Obj)>, seen: &mut Vec<u32>, depth: usize) {
        if depth > 32 || out.len() > 20_000 {
            return;
        }
        if let Obj::Ref(id) = node {
            if seen.contains(id) {
                return;
            }
            seen.push(*id);
        }
        let Some(dict) = self.dict(node) else { return };
        let res = dict.get("Resources").unwrap_or(resources);
        if let Obj::Arr(kids) = self.get(dict, "Kids") {
            for kid in kids {
                self.collect_pages(kid, res, out, seen, depth + 1);
            }
        } else if matches!(self.get(dict, "Type"), Obj::Name(n) if n == "Page") || dict.contains_key("Contents") {
            out.push((dict.get("Contents").cloned().unwrap_or(Obj::Null), res.clone()));
        }
    }

    fn pages(&self) -> Vec<(Obj, Obj)> {
        let mut out = Vec::new();
        for root in self.roots.iter().rev() {
            if let Some(catalog) = self.dict(root) {
                if let Some(pages) = catalog.get("Pages") {
                    self.collect_pages(pages, &Obj::Null, &mut out, &mut Vec::new(), 0);
                }
            }
            if !out.is_empty() {
                return out;
            }
        }
        // no usable page tree: take every page object in file order
        let mut ids: Vec<u32> = self.objs.iter().filter(|(_, o)| matches!(o, Obj::Dict(d) if matches!(d.get("Type"), Some(Obj::Name(n)) if n == "Page"))).map(|(id, _)| *id).collect();
        ids.sort_unstable();
        for id in ids {
            self.collect_pages(&Obj::Ref(id), &Obj::Null, &mut out, &mut Vec::new(), 0);
        }
        out
    }

    fn page_content(&self, contents: &Obj) -> Vec<u8> {
        let mut out = Vec::new();
        let mut add = |obj: &Obj| {
            if let Obj::Ref(id) = obj {
                if let Some(bytes) = self.stream_data(*id) {
                    out.extend_from_slice(&bytes);
                    out.push(b'\n');
                }
            }
        };
        match self.resolve(contents) {
            Obj::Arr(list) => list.iter().for_each(&mut add),
            _ => add(contents),
        }
        out
    }
}

pub(super) fn extract(data: &[u8]) -> Result<Extracted, String> {
    if !data.windows(5).take(1024).any(|w| w == b"%PDF-") {
        return Err("this file is not a PDF".into());
    }
    let doc = Doc::scan(data);
    if doc.encrypted {
        return Err("this PDF is encrypted; remove the password and try again".into());
    }
    let pages = doc.pages();
    if pages.is_empty() {
        return Err("no pages were found in this PDF; the file may be damaged".into());
    }
    let mut cache: HashMap<u32, Rc<Font>> = HashMap::new();
    let mut texts: Vec<String> = Vec::new();
    let mut total = 0;
    let mut truncated = false;
    for (contents, resources) in &pages {
        let mut fonts: HashMap<String, Rc<Font>> = HashMap::new();
        if let Some(res) = doc.dict(resources) {
            if let Obj::Dict(list) = doc.get(res, "Font") {
                for (name, entry) in list {
                    let font = match entry {
                        Obj::Ref(id) => cache.entry(*id).or_insert_with(|| Rc::new(doc.load_font(entry))).clone(),
                        _ => Rc::new(doc.load_font(entry)),
                    };
                    fonts.insert(name.clone(), font);
                }
            }
        }
        let text = Interpreter { fonts: &fonts }.run(&doc.page_content(contents));
        total += text.len();
        texts.push(text.trim().to_owned());
        if total > TEXT_LIMIT {
            truncated = true;
            break;
        }
    }
    if texts.iter().all(|t| t.is_empty()) {
        return Err("this PDF has no text layer (scanned); run it through OCR first".into());
    }
    Ok(Extracted { text: texts.join("\n\n"), pages: pages.len(), truncated })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Wraps data in a stored (uncompressed) zlib stream.
    pub(crate) fn stored_zlib(data: &[u8]) -> Vec<u8> {
        let mut out = vec![0x78, 0x01];
        let mut chunks = data.chunks(60_000).peekable();
        if chunks.peek().is_none() {
            out.extend_from_slice(&[1, 0, 0, 0xff, 0xff]);
        }
        while let Some(c) = chunks.next() {
            out.push(u8::from(chunks.peek().is_none()));
            out.extend_from_slice(&(c.len() as u16).to_le_bytes());
            out.extend_from_slice(&(!(c.len() as u16)).to_le_bytes());
            out.extend_from_slice(c);
        }
        out.extend_from_slice(&[0, 0, 0, 0]);
        out
    }

    fn build_pdf(compress: bool) -> Vec<u8> {
        let content = b"BT /F1 12 Tf 72 700 Td (Hello) Tj 50 0 Td (World) Tj 0 -20 Td [(Se) -300 (cond)] TJ ET\nBT /F2 12 Tf 72 600 Td <0001000200030004> Tj ET";
        let cmap = b"/CIDInit begincmap 2 beginbfchar <0001> <0041> <0002> <0042> endbfchar 1 beginbfrange <0003> <0004> <0043> endbfrange endcmap";
        let mut pdf: Vec<u8> = b"%PDF-1.4\n".to_vec();
        let add_stream = |pdf: &mut Vec<u8>, id: u32, dict: &str, body: &[u8]| {
            let body = if compress { stored_zlib(body) } else { body.to_vec() };
            let filter = if compress { "/Filter /FlateDecode " } else { "" };
            pdf.extend_from_slice(format!("{id} 0 obj\n<< {dict} {filter}/Length {} >>\nstream\n", body.len()).as_bytes());
            pdf.extend_from_slice(&body);
            pdf.extend_from_slice(b"\nendstream\nendobj\n");
        };
        pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
        pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 /Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> >>\nendobj\n");
        pdf.extend_from_slice(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /Contents 4 0 R >>\nendobj\n");
        add_stream(&mut pdf, 4, "", content);
        pdf.extend_from_slice(b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n");
        pdf.extend_from_slice(b"6 0 obj\n<< /Type /Font /Subtype /Type0 /ToUnicode 7 0 R >>\nendobj\n");
        add_stream(&mut pdf, 7, "", cmap);
        pdf.extend_from_slice(b"trailer\n<< /Root 1 0 R /Size 8 >>\n%%EOF\n");
        pdf
    }

    #[test]
    fn reads_text_with_lines_spaces_and_cmaps() {
        for compress in [false, true] {
            let got = extract(&build_pdf(compress)).unwrap();
            assert_eq!(got.pages, 1);
            assert_eq!(got.text, "Hello World\nSe cond\n\nABCD", "compress={compress}");
        }
    }

    #[test]
    fn scanned_and_broken_files_give_clear_errors() {
        let mut pdf: Vec<u8> = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] >>\nendobj\n3 0 obj\n<< /Type /Page /Contents 4 0 R >>\nendobj\n".to_vec();
        pdf.extend_from_slice(b"4 0 obj\n<< /Length 12 >>\nstream\n0 0 10 10 re f\nendstream\nendobj\ntrailer << /Root 1 0 R >>");
        assert!(extract(&pdf).unwrap_err().contains("no text layer"));
        assert!(extract(b"hello").unwrap_err().contains("not a PDF"));
        let enc = b"%PDF-1.4\ntrailer << /Root 1 0 R /Encrypt 9 0 R >>";
        assert!(extract(enc).unwrap_err().contains("encrypted"));
        // truncated input must not panic
        let full = build_pdf(true);
        for cut in (0..full.len()).step_by(7) {
            let _ = extract(&full[..cut]);
        }
    }

    #[test]
    fn cmap_ranges_with_arrays() {
        let map = parse_cmap("beginbfrange <10> <12> [<0041> <0042> <00430044>] endbfrange beginbfchar <20> <D83DDE00> endbfchar");
        assert_eq!(map[&0x10], "A");
        assert_eq!(map[&0x12], "CD");
        assert_eq!(map[&0x20], "\u{1F600}");
    }

    #[test]
    fn png_predictor_up_filter() {
        assert_eq!(png_predictor(&[2, 1, 2, 3, 2, 1, 1, 1], 3, 1), vec![1, 2, 3, 2, 3, 4]);
    }
}
