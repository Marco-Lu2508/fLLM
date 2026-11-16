//! Row 47 of PARITY.md: a picture on the clipboard (a screenshot, a copied image) is attached with
//! Ctrl+V / Cmd+V, as the `paste` of the chat box of the page attaches it. The text of the clipboard
//! is the business of the text box; this module looks only when there is no text on it.

/// The picture on the clipboard as PNG, or `None` when there is none, or there is text on it.
pub fn read() -> Result<Option<Vec<u8>>, String> {
    let mut clipboard = arboard::Clipboard::new().map_err(|e| format!("The clipboard could not be read: {e}"))?;
    if clipboard.get_text().is_ok_and(|t| !t.is_empty()) {
        return Ok(None);
    }
    match clipboard.get_image() {
        Ok(img) => png_of(img.width, img.height, &img.bytes).map(Some),
        // nothing of a picture on it: not an error
        Err(arboard::Error::ContentNotAvailable) => Ok(None),
        Err(e) => Err(format!("The clipboard could not be read: {e}")),
    }
}

/// RGBA pixels as a PNG file.
pub fn png_of(width: usize, height: usize, rgba: &[u8]) -> Result<Vec<u8>, String> {
    let buffer = image::RgbaImage::from_raw(width as u32, height as u32, rgba.to_vec()).ok_or("The picture on the clipboard is not whole.")?;
    let mut out = std::io::Cursor::new(Vec::new());
    buffer.write_to(&mut out, image::ImageFormat::Png).map_err(|e| format!("The picture could not be made: {e}"))?;
    Ok(out.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixels_become_a_png_that_reads_back() {
        let rgba: Vec<u8> = (0..4 * 3 * 2).map(|i| i as u8).collect();
        let png = png_of(3, 2, &rgba).unwrap();
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
        let back = image::load_from_memory(&png).unwrap().to_rgba8();
        assert_eq!((back.width(), back.height()), (3, 2));
        assert_eq!(back.into_raw(), rgba);
    }

    #[test]
    fn a_picture_that_is_not_whole_is_refused() {
        assert_eq!(png_of(3, 2, &[0; 5]).unwrap_err(), "The picture on the clipboard is not whole.");
    }
}
