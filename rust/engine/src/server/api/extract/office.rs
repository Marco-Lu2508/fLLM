//! Text from DOCX, XLSX, EPUB and HTML files. XML is read with a tiny tag scanner: these formats only need the
//! text nodes and the structure tags (paragraphs, rows, cells), not a full parser.
use super::inflate::Zip;

const ENTRY_LIMIT: usize = 64 << 20;

/// One piece of markup: a tag (name without `<`, `/` and attributes, plus whether it closes or is self-closing) or text.
enum Piece<'a> {
    Text(&'a str),
    Tag { name: &'a str, attrs: &'a str, close: bool, empty: bool },
}

fn pieces(xml: &str) -> Vec<Piece<'_>> {
    let mut out = Vec::new();
    let mut rest = xml;
    while !rest.is_empty() {
        match rest.find('<') {
            None => {
                out.push(Piece::Text(rest));
                break;
            }
            Some(0) => {
                if let Some(body) = rest.strip_prefix("<!--") {
                    rest = body.find("-->").map_or("", |e| &body[e + 3..]);
                    continue;
                }
                if let Some(body) = rest.strip_prefix("<![CDATA[") {
                    let end = body.find("]]>").unwrap_or(body.len());
                    out.push(Piece::Text(&body[..end]));
                    rest = body.get(end + 3..).unwrap_or("");
                    continue;
                }
                let Some(end) = rest.find('>') else { break };
                let inner = &rest[1..end];
                rest = &rest[end + 1..];
                if inner.starts_with('?') || inner.starts_with('!') {
                    continue;
                }
                let close = inner.starts_with('/');
                let empty = inner.ends_with('/');
                let inner = inner.trim_start_matches('/').trim_end_matches('/');
                let (name, attrs) = inner.split_once(|c: char| c.is_whitespace()).unwrap_or((inner, ""));
                out.push(Piece::Tag { name, attrs, close, empty });
            }
            Some(at) => {
                out.push(Piece::Text(&rest[..at]));
                rest = &rest[at..];
            }
        }
    }
    out
}

/// The value of attribute `key` in a tag's attribute text.
fn attr<'a>(attrs: &'a str, key: &str) -> Option<&'a str> {
    let mut from = 0;
    while let Some(p) = attrs[from..].find(key) {
        let at = from + p;
        let before_ok = at == 0 || attrs[..at].ends_with(|c: char| c.is_whitespace() || c == ':');
        let after = &attrs[at + key.len()..];
        if before_ok {
            if let Some(v) = after.trim_start().strip_prefix('=') {
                let v = v.trim_start();
                let quote = v.chars().next()?;
                if quote == '"' || quote == '\'' {
                    let v = &v[1..];
                    return v.find(quote).map(|e| &v[..e]);
                }
            }
        }
        from = at + key.len();
    }
    None
}

pub(super) fn decode_entities(text: &str) -> String {
    if !text.contains('&') {
        return text.to_owned();
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find('&') {
        out.push_str(&rest[..at]);
        rest = &rest[at..];
        let Some(end) = rest.find(';').filter(|&e| e <= 10) else {
            out.push('&');
            rest = &rest[1..];
            continue;
        };
        let name = &rest[1..end];
        let ch = match name {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            "nbsp" => Some(' '),
            "ndash" => Some('\u{2013}'),
            "mdash" => Some('\u{2014}'),
            "hellip" => Some('\u{2026}'),
            _ => name.strip_prefix('#').and_then(|n| match n.strip_prefix(['x', 'X']) {
                Some(hex) => u32::from_str_radix(hex, 16).ok(),
                None => n.parse().ok(),
            })
            .and_then(char::from_u32),
        };
        match ch {
            Some(c) => {
                out.push(c);
                rest = &rest[end + 1..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

fn tidy(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut blank = 0;
    for line in text.lines() {
        let line = line.trim_end();
        if line.trim().is_empty() {
            blank += 1;
            if blank > 1 {
                continue;
            }
        } else {
            blank = 0;
        }
        out.push_str(line);
        out.push('\n');
    }
    out.trim().to_owned()
}

/// Text of a WordprocessingML `document.xml`: one line per paragraph, table cells joined by ` | `.
pub(super) fn docx_xml_text(xml: &str) -> String {
    let mut out = String::new();
    let mut in_text = false;
    let mut cell_start = true;
    let mut in_cell = false;
    for piece in pieces(xml) {
        match piece {
            Piece::Text(t) if in_text => out.push_str(&decode_entities(t)),
            Piece::Text(_) => {}
            Piece::Tag { name, close, empty, .. } => match (name, close) {
                ("w:t", false) => in_text = !empty,
                ("w:t", true) => in_text = false,
                ("w:tab", _) => out.push('\t'),
                ("w:br" | "w:cr", _) => out.push('\n'),
                ("w:p", true) => out.push(if in_cell { ' ' } else { '\n' }),
                ("w:tr", false) => cell_start = true,
                ("w:tr", true) => {
                    out.truncate(out.trim_end().len());
                    out.push('\n');
                }
                ("w:tc", false) => {
                    if !cell_start {
                        out.truncate(out.trim_end().len());
                        out.push_str(" | ");
                    }
                    cell_start = false;
                    in_cell = true;
                }
                ("w:tc", true) => in_cell = false,
                _ => {}
            },
        }
    }
    tidy(&out)
}

pub(super) fn docx(data: &[u8]) -> Result<String, String> {
    let zip = Zip::open(data)?;
    let main = zip.read("word/document.xml", ENTRY_LIMIT).ok_or("this DOCX file has no word/document.xml")?;
    let mut text = docx_xml_text(&String::from_utf8_lossy(&main));
    for extra in ["word/footnotes.xml", "word/endnotes.xml"] {
        if let Some(bytes) = zip.read(extra, ENTRY_LIMIT) {
            let more = docx_xml_text(&String::from_utf8_lossy(&bytes));
            if !more.is_empty() {
                text.push_str("\n\n");
                text.push_str(&more);
            }
        }
    }
    Ok(text)
}

fn shared_strings(xml: &str) -> Vec<String> {
    let mut list = Vec::new();
    let mut current = String::new();
    let mut in_item = false;
    let mut in_text = false;
    for piece in pieces(xml) {
        match piece {
            Piece::Text(t) if in_text => current.push_str(&decode_entities(t)),
            Piece::Text(_) => {}
            Piece::Tag { name, close, empty, .. } => match (name, close) {
                ("si", false) => {
                    in_item = !empty;
                    current.clear();
                    if empty {
                        list.push(String::new());
                    }
                }
                ("si", true) => {
                    in_item = false;
                    list.push(std::mem::take(&mut current));
                }
                ("t", false) => in_text = in_item && !empty,
                ("t", true) => in_text = false,
                _ => {}
            },
        }
    }
    list
}

fn sheet_text(xml: &str, strings: &[String]) -> String {
    let mut out = String::new();
    let mut kind = String::new();
    let mut value = String::new();
    let mut capture = false;
    let mut row_has_cell = false;
    for piece in pieces(xml) {
        match piece {
            Piece::Text(t) if capture => value.push_str(&decode_entities(t)),
            Piece::Text(_) => {}
            Piece::Tag { name, attrs, close, empty } => match (name, close) {
                ("row", false) => row_has_cell = false,
                ("row", true) => {
                    if row_has_cell {
                        out.push('\n');
                    }
                }
                ("c", false) => {
                    kind = attr(attrs, "t").unwrap_or("").to_owned();
                    value.clear();
                    if empty {
                        out.push_str(if row_has_cell { "\t" } else { "" });
                        row_has_cell = true;
                    }
                }
                ("c", true) => {
                    if row_has_cell {
                        out.push('\t');
                    }
                    row_has_cell = true;
                    if kind == "s" {
                        if let Some(s) = value.trim().parse::<usize>().ok().and_then(|i| strings.get(i)) {
                            out.push_str(s);
                        }
                    } else {
                        out.push_str(value.trim());
                    }
                }
                ("v" | "t", false) => capture = !empty,
                ("v" | "t", true) => capture = false,
                _ => {}
            },
        }
    }
    out
}

pub(super) fn xlsx(data: &[u8]) -> Result<String, String> {
    let zip = Zip::open(data)?;
    let strings = zip.read("xl/sharedStrings.xml", ENTRY_LIMIT).map(|b| shared_strings(&String::from_utf8_lossy(&b))).unwrap_or_default();
    let mut names: Vec<String> = Vec::new();
    if let Some(book) = zip.read("xl/workbook.xml", ENTRY_LIMIT) {
        for piece in pieces(&String::from_utf8_lossy(&book)) {
            if let Piece::Tag { name: "sheet", attrs, close: false, .. } = piece {
                names.push(decode_entities(attr(attrs, "name").unwrap_or("")));
            }
        }
    }
    let mut sheets: Vec<(usize, String)> = zip
        .names()
        .filter_map(|n| {
            let num = n.strip_prefix("xl/worksheets/sheet")?.strip_suffix(".xml")?.parse::<usize>().ok()?;
            Some((num, n.to_owned()))
        })
        .collect();
    sheets.sort();
    if sheets.is_empty() {
        return Err("this XLSX file has no worksheets".into());
    }
    let mut out = String::new();
    for (index, (_, path)) in sheets.iter().enumerate() {
        let Some(bytes) = zip.read(path, ENTRY_LIMIT) else { continue };
        let body = sheet_text(&String::from_utf8_lossy(&bytes), &strings);
        if body.trim().is_empty() {
            continue;
        }
        let title = names.get(index).filter(|n| !n.is_empty()).cloned().unwrap_or_else(|| format!("Sheet {}", index + 1));
        out.push_str(&format!("## {title}\n{}\n\n", body.trim_end()));
    }
    Ok(out.trim().to_owned())
}

/// Readable text of an HTML page: scripts, styles and comments dropped, block elements become line breaks.
pub(super) fn html_text(html: &str) -> String {
    let mut out = String::new();
    let mut skip: Option<String> = None;
    for piece in pieces(html) {
        match piece {
            Piece::Text(t) => {
                if skip.is_none() {
                    let decoded = decode_entities(t);
                    let mut last_space = out.ends_with(|c: char| c.is_whitespace());
                    for c in decoded.chars() {
                        if c.is_whitespace() {
                            if !last_space {
                                out.push(' ');
                            }
                            last_space = true;
                        } else {
                            out.push(c);
                            last_space = false;
                        }
                    }
                }
            }
            Piece::Tag { name, close, .. } => {
                let name = name.to_ascii_lowercase();
                if let Some(current) = &skip {
                    if close && *current == name {
                        skip = None;
                    }
                    continue;
                }
                match name.as_str() {
                    "script" | "style" | "noscript" | "template" | "svg" | "head" if !close => skip = Some(name.clone()),
                    "br" | "p" | "div" | "section" | "article" | "header" | "footer" | "nav" | "ul" | "ol" | "table" | "tr" | "blockquote" | "pre" | "hr"
                    | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "li" | "dt" | "dd" | "figure" | "figcaption" | "details" | "summary" | "main" => {
                        let heading = name.starts_with('h') && name.len() == 2 && !close;
                        out.push_str(if heading || (close && matches!(name.as_str(), "p" | "table" | "ul" | "ol")) { "\n\n" } else { "\n" });
                    }
                    "td" | "th" if close => out.push_str(" | "),
                    _ => {}
                }
            }
        }
    }
    let out: String = out.lines().map(|l| l.trim().trim_end_matches('|').trim_end()).collect::<Vec<_>>().join("\n");
    tidy(&out)
}

pub(super) fn epub(data: &[u8]) -> Result<String, String> {
    let zip = Zip::open(data)?;
    let container = zip.read("META-INF/container.xml", ENTRY_LIMIT).ok_or("this EPUB file has no META-INF/container.xml")?;
    let container = String::from_utf8_lossy(&container).into_owned();
    let opf_path = pieces(&container)
        .into_iter()
        .find_map(|p| match p {
            Piece::Tag { name: "rootfile", attrs, .. } => attr(attrs, "full-path").map(str::to_owned),
            _ => None,
        })
        .ok_or("this EPUB file names no package document")?;
    let opf = String::from_utf8_lossy(&zip.read(&opf_path, ENTRY_LIMIT).ok_or("the EPUB package document is missing")?).into_owned();
    let base = opf_path.rsplit_once('/').map_or("", |(dir, _)| dir);
    let mut manifest: Vec<(String, String)> = Vec::new();
    let mut spine: Vec<String> = Vec::new();
    for piece in pieces(&opf) {
        if let Piece::Tag { name, attrs, close: false, .. } = piece {
            match name {
                "item" => {
                    if let (Some(id), Some(href)) = (attr(attrs, "id"), attr(attrs, "href")) {
                        manifest.push((id.to_owned(), href.to_owned()));
                    }
                }
                "itemref" => {
                    if let Some(id) = attr(attrs, "idref") {
                        spine.push(id.to_owned());
                    }
                }
                _ => {}
            }
        }
    }
    let mut out = String::new();
    for id in spine {
        let Some((_, href)) = manifest.iter().find(|(i, _)| *i == id) else { continue };
        let href = decode_entities(href.split('#').next().unwrap_or(""));
        let path = if base.is_empty() { href } else { format!("{base}/{href}") };
        if let Some(bytes) = zip.read(&path, ENTRY_LIMIT) {
            let text = html_text(&String::from_utf8_lossy(&bytes));
            if !text.is_empty() {
                out.push_str(&text);
                out.push_str("\n\n");
            }
        }
    }
    if out.trim().is_empty() {
        return Err("no readable text was found in this EPUB file".into());
    }
    Ok(out.trim().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_drops_scripts_and_keeps_structure() {
        let html = "<html><head><title>x</title><style>p{}</style></head><body><h1>Title</h1><p>One &amp; two</p><script>var a=1;</script><ul><li>a</li><li>b</li></ul><table><tr><td>1</td><td>2</td></tr></table></body></html>";
        let text = html_text(html);
        assert!(text.contains("Title"));
        assert!(text.contains("One & two"));
        assert!(!text.contains("var a"));
        assert!(!text.contains("p{}"));
        assert!(text.contains("1 | 2"), "{text}");
    }

    #[test]
    fn docx_paragraphs_and_tables() {
        let xml = r#"<w:document><w:body><w:p><w:r><w:t>Hello</w:t></w:r><w:r><w:t xml:space="preserve"> world</w:t></w:r></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>A</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>B</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>"#;
        assert_eq!(docx_xml_text(xml), "Hello world\nA | B");
    }

    #[test]
    fn entities() {
        assert_eq!(decode_entities("a &lt; b &#65; &#x42; &unknown; &"), "a < b A B &unknown; &");
    }
}
