//! `POST /api/studio/extract?name=file.ext`: text from a document the Studio page attaches to a chat. The body is
//! the raw file (at most 50 MB). PDF (text layer), DOCX, XLSX, EPUB and HTML are unpacked here, everything else
//! is read as text. The reply is `{text, pages?, truncated}`. The readers use no extra libraries: see `inflate`
//! (DEFLATE and ZIP), `pdf` and `office`.
//!
//! The route changes nothing but is a POST under `/api/studio/`, so it follows the same rule as the other
//! Studio routes: without an API key only this machine may call it.
use super::*;
use axum::extract::Query;
use std::collections::HashMap;

mod inflate;
mod office;
mod pdf;

pub(in crate::server) const MAX_UPLOAD_BYTES: usize = 50 << 20;
/// Text returned for one document at most (characters), so the reply and the browser storage stay bounded.
const MAX_TEXT_BYTES: usize = 8 << 20;

#[derive(Debug)]
pub(super) struct Document {
    pub text: String,
    pub pages: Option<usize>,
    pub truncated: bool,
}

fn extension(name: &str) -> String {
    name.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase()).unwrap_or_default()
}

fn plain_text(bytes: &[u8]) -> Result<String, String> {
    if bytes.iter().take(8192).any(|&b| b == 0) {
        return Err("this file is not a text file and its format is not supported".into());
    }
    let text = match std::str::from_utf8(bytes) {
        Ok(s) => s.to_owned(),
        Err(_) => String::from_utf8_lossy(bytes).into_owned(),
    };
    Ok(text.strip_prefix('\u{feff}').map(str::to_owned).unwrap_or(text))
}

/// Reads the text of a document by the extension of its name (content sniffing for PDF and ZIP-based files).
pub(super) fn extract_document(name: &str, bytes: &[u8]) -> Result<Document, String> {
    let ext = extension(name);
    let mut pages = None;
    let text = if ext == "pdf" || bytes.starts_with(b"%PDF-") {
        let got = pdf::extract(bytes)?;
        pages = Some(got.pages);
        if got.truncated {
            return Ok(finish(got.text, pages, true));
        }
        got.text
    } else {
        match ext.as_str() {
            "docx" => office::docx(bytes)?,
            "xlsx" => office::xlsx(bytes)?,
            "epub" => office::epub(bytes)?,
            "html" | "htm" | "xhtml" => office::html_text(&plain_text(bytes)?),
            "doc" | "xls" | "ppt" | "pptx" | "odt" | "rtf" => {
                return Err(format!("the .{ext} format is not supported; save the file as PDF, DOCX or text"));
            }
            _ => plain_text(bytes)?,
        }
    };
    if text.trim().is_empty() {
        return Err("no text was found in this file".into());
    }
    Ok(finish(text, pages, false))
}

fn finish(mut text: String, pages: Option<usize>, truncated: bool) -> Document {
    let mut truncated = truncated;
    if text.len() > MAX_TEXT_BYTES {
        let mut cut = MAX_TEXT_BYTES;
        while !text.is_char_boundary(cut) {
            cut -= 1;
        }
        text.truncate(cut);
        truncated = true;
    }
    Document { text, pages, truncated }
}

/// `POST /api/studio/extract?name=report.pdf` with the file as body.
pub(in crate::server) async fn extract(Query(query): Query<HashMap<String, String>>, body: Bytes) -> Response {
    if body.is_empty() {
        return api_error(StatusCode::BAD_REQUEST, "the request body is empty; send the file as the body");
    }
    let name = query.get("name").cloned().unwrap_or_default();
    match tokio::task::spawn_blocking(move || extract_document(&name, &body)).await {
        Ok(Ok(doc)) => {
            let mut reply = json!({"text": doc.text, "truncated": doc.truncated});
            if let Some(pages) = doc.pages {
                reply["pages"] = json!(pages);
            }
            Json(reply).into_response()
        }
        Ok(Err(message)) => api_error(StatusCode::UNPROCESSABLE_ENTITY, &message),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stored_raw(data: &[u8]) -> Vec<u8> {
        let mut out = vec![1];
        out.extend_from_slice(&(data.len() as u16).to_le_bytes());
        out.extend_from_slice(&(!(data.len() as u16)).to_le_bytes());
        out.extend_from_slice(data);
        out
    }

    /// A ZIP archive with the given entries, each stored without compression.
    fn zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut central = Vec::new();
        for (name, data) in entries {
            let offset = out.len() as u32;
            let mut local = b"PK\x03\x04".to_vec();
            local.extend_from_slice(&[20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            local.extend_from_slice(&(data.len() as u32).to_le_bytes());
            local.extend_from_slice(&(data.len() as u32).to_le_bytes());
            local.extend_from_slice(&(name.len() as u16).to_le_bytes());
            local.extend_from_slice(&[0, 0]);
            local.extend_from_slice(name.as_bytes());
            local.extend_from_slice(data);
            out.extend_from_slice(&local);
            let mut c = b"PK\x01\x02".to_vec();
            c.extend_from_slice(&[20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            c.extend_from_slice(&(data.len() as u32).to_le_bytes());
            c.extend_from_slice(&(data.len() as u32).to_le_bytes());
            c.extend_from_slice(&(name.len() as u16).to_le_bytes());
            c.extend_from_slice(&[0; 12]);
            c.extend_from_slice(&offset.to_le_bytes());
            c.extend_from_slice(name.as_bytes());
            central.extend_from_slice(&c);
        }
        let start = out.len() as u32;
        out.extend_from_slice(&central);
        out.extend_from_slice(b"PK\x05\x06\0\0\0\0");
        out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        out.extend_from_slice(&(central.len() as u32).to_le_bytes());
        out.extend_from_slice(&start.to_le_bytes());
        out.extend_from_slice(&[0, 0]);
        out
    }

    fn hex(text: &str) -> Vec<u8> {
        (0..text.len()).step_by(2).map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap()).collect()
    }

    #[test]
    fn inflate_reads_zlib_fixed_and_dynamic_blocks() {
        let (out, ok) = inflate::inflate_zlib(&hex("789ccb48cdc9c90700062c0215"), 1 << 20);
        assert!(ok);
        assert_eq!(out, b"hello");
        let (out, ok) = inflate::inflate_zlib(&hex("789c4b4c84010014e103cb"), 1 << 20);
        assert!(ok);
        assert_eq!(out, b"aaaaaaaaaa");
        // dynamic Huffman block (made by an independent DEFLATE encoder) with back references
        let packed = hex("9dd55b16c1500c46e177a3c810e40f2d66e37268397a68d56df41633b09fb3f653be95e4b64b365dd9ad49761ddbedc9367d7974b62f4f3b8ee7cb60e59efadf38afdf2fdb95c3247f1b078d4013a09981660e9a0a34356816a059929d22084482130a4e2c38c1e04483130e4e3c3801e144848808a1db40448888101121224244848808111122228288082222d0bb20228288082222888820228288883f457c00");
        let (out, ok) = inflate::inflate_raw(&packed, 1 << 20);
        assert!(ok);
        let expected: String = (0..40).map(|i| format!("line {i}: the quick brown fox jumps over the lazy dog\n")).collect();
        assert_eq!(String::from_utf8(out).unwrap(), expected);
        // limits and damage are handled without panics
        let (out, ok) = inflate::inflate_raw(&packed, 100);
        assert!(!ok);
        assert_eq!(out.len(), 100);
        let (_, ok) = inflate::inflate_raw(&packed[..40], 1 << 20);
        assert!(!ok);
    }

    #[test]
    fn docx_xlsx_epub_and_html() {
        let docx = zip(&[("word/document.xml", br#"<w:document><w:body><w:p><w:r><w:t>First &amp; second</w:t></w:r></w:p><w:p><w:r><w:t>Third</w:t></w:r></w:p></w:body></w:document>"#)]);
        assert_eq!(extract_document("a.docx", &docx).unwrap().text, "First & second\nThird");

        let xlsx = zip(&[
            ("xl/workbook.xml", br#"<workbook><sheets><sheet name="Data" sheetId="1" r:id="rId1"/></sheets></workbook>"#),
            ("xl/sharedStrings.xml", br#"<sst><si><t>Name</t></si><si><t>Score</t></si><si><t>Ann</t></si></sst>"#),
            ("xl/worksheets/sheet1.xml", br#"<worksheet><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row><row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2"><v>42</v></c></row></sheetData></worksheet>"#),
        ]);
        assert_eq!(extract_document("a.xlsx", &xlsx).unwrap().text, "## Data\nName\tScore\nAnn\t42");

        let epub = zip(&[
            ("META-INF/container.xml", br#"<container><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>"#),
            ("OEBPS/content.opf", br#"<package><manifest><item id="c1" href="ch1.xhtml"/><item id="c2" href="ch2.xhtml"/></manifest><spine><itemref idref="c2"/><itemref idref="c1"/></spine></package>"#),
            ("OEBPS/ch1.xhtml", b"<html><body><p>One</p></body></html>"),
            ("OEBPS/ch2.xhtml", b"<html><body><p>Two</p></body></html>"),
        ]);
        assert_eq!(extract_document("a.epub", &epub).unwrap().text, "Two\n\nOne");

        let html = extract_document("a.html", b"<p>Hi <b>there</b></p><script>x()</script>").unwrap();
        assert_eq!(html.text, "Hi there");
        assert!(extract_document("a.docx", b"not a zip").is_err());
    }

    #[test]
    fn deflated_zip_entry_and_plain_text() {
        let body = stored_raw(b"<w:t>x</w:t>");
        assert!(inflate::inflate_raw(&body, 100).1);
        let doc = extract_document("notes.md", "\u{feff}# Title\nbody".as_bytes()).unwrap();
        assert_eq!(doc.text, "# Title\nbody");
        assert_eq!(doc.pages, None);
        assert!(extract_document("x.bin", &[1, 2, 0, 3]).is_err());
        assert!(extract_document("x.doc", b"abc").unwrap_err().contains("not supported"));
        assert!(extract_document("empty.txt", b"  \n").is_err());
    }

    #[test]
    fn pdf_goes_through_the_dispatcher() {
        let pdf = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] >>\nendobj\n3 0 obj\n<< /Type /Page /Contents 4 0 R /Resources << >> >>\nendobj\n4 0 obj\n<< /Length 30 >>\nstream\nBT /F1 12 Tf 10 10 Td (Hi) Tj ET\nendstream\nendobj\ntrailer << /Root 1 0 R >>";
        let doc = extract_document("x.pdf", pdf).unwrap();
        assert_eq!(doc.text, "Hi");
        assert_eq!(doc.pages, Some(1));
    }
}
