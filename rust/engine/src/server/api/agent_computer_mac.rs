//! Computer use on macOS through the system frameworks (CoreGraphics, ApplicationServices) declared by hand, so that no
//! extra crate is needed: `CGDisplayCreateImage` for screenshots, `CGEventPost` for the mouse and the keyboard,
//! `CGEventSourceKeyState` for the emergency stop chord.
//!
//! NOT VERIFIED ON A REAL MAC: the code is type-checked for the macOS targets only. macOS asks the user for two permissions
//! that belong to the *application* that runs fLLM (fLLM.app, or the terminal the server was started from): Screen Recording
//! (screenshots) and Accessibility (mouse and keyboard). Without them screenshots come back without windows and input is
//! dropped silently, so both are checked first and explained.
use super::agent_computer::{Button, Chord, DisplayInfo, Frame, Key, Named};
use std::ffi::{c_char, c_void};
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[repr(C)]
#[derive(Clone, Copy)]
struct CGPoint {
    x: f64,
    y: f64,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct CGSize {
    width: f64,
    height: f64,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct CGRect {
    origin: CGPoint,
    size: CGSize,
}

type Ref = *mut c_void;

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {
    fn CGGetActiveDisplayList(max: u32, displays: *mut u32, count: *mut u32) -> i32;
    fn CGMainDisplayID() -> u32;
    fn CGDisplayBounds(display: u32) -> CGRect;
    fn CGDisplayCopyDisplayMode(display: u32) -> Ref;
    fn CGDisplayModeGetPixelWidth(mode: Ref) -> usize;
    fn CGDisplayModeGetPixelHeight(mode: Ref) -> usize;
    fn CGDisplayModeRelease(mode: Ref);
    fn CGDisplayCreateImage(display: u32) -> Ref;
    fn CGImageGetWidth(image: Ref) -> usize;
    fn CGImageGetHeight(image: Ref) -> usize;
    fn CGImageGetBytesPerRow(image: Ref) -> usize;
    fn CGImageGetBitsPerPixel(image: Ref) -> usize;
    fn CGImageGetBitmapInfo(image: Ref) -> u32;
    fn CGImageGetDataProvider(image: Ref) -> Ref;
    fn CGImageRelease(image: Ref);
    fn CGDataProviderCopyData(provider: Ref) -> Ref;
    fn CGEventCreate(source: Ref) -> Ref;
    fn CGEventGetLocation(event: Ref) -> CGPoint;
    fn CGEventCreateMouseEvent(source: Ref, kind: u32, position: CGPoint, button: u32) -> Ref;
    fn CGEventCreateKeyboardEvent(source: Ref, keycode: u16, down: bool) -> Ref;
    fn CGEventCreateScrollWheelEvent2(source: Ref, units: u32, count: u32, wheel1: i32, wheel2: i32, wheel3: i32) -> Ref;
    fn CGEventKeyboardSetUnicodeString(event: Ref, length: usize, text: *const u16);
    fn CGEventSetFlags(event: Ref, flags: u64);
    fn CGEventSetIntegerValueField(event: Ref, field: u32, value: i64);
    fn CGEventPost(tap: u32, event: Ref);
    fn CGEventSourceKeyState(state: i32, keycode: u16) -> bool;
}

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFDataGetBytePtr(data: Ref) -> *const u8;
    fn CFDataGetLength(data: Ref) -> isize;
    fn CFRelease(object: Ref);
}

#[link(name = "ApplicationServices", kind = "framework")]
extern "C" {
    fn AXIsProcessTrusted() -> bool;
}

extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

/// Calls a `bool f(void)` function of the system by name; `None` when this macOS does not have it.
fn optional_bool_fn(name: &[u8]) -> Option<bool> {
    // RTLD_DEFAULT is (void *)-2 on macOS.
    // SAFETY: `name` is NUL terminated; a found symbol with this name has the signature `bool (*)(void)`.
    unsafe {
        let p = dlsym(-2isize as *mut c_void, name.as_ptr().cast());
        if p.is_null() {
            return None;
        }
        let f: extern "C" fn() -> bool = std::mem::transmute(p);
        Some(f())
    }
}

const MOUSE_MOVED: u32 = 5;
const HID_TAP: u32 = 0;
const CLICK_STATE_FIELD: u32 = 1;
const FLAG_SHIFT: u64 = 0x2_0000;
const FLAG_CONTROL: u64 = 0x4_0000;
const FLAG_ALT: u64 = 0x8_0000;
const FLAG_COMMAND: u64 = 0x10_0000;

pub fn platform() -> &'static str {
    "macos"
}

fn screen_permission() -> Result<(), String> {
    if optional_bool_fn(b"CGPreflightScreenCaptureAccess\0") == Some(false) {
        return Err("macOS has not allowed this app to record the screen. Open System Settings > Privacy & Security > Screen Recording, switch fLLM on (or the terminal that started the server), then restart the app".to_owned());
    }
    Ok(())
}

fn input_permission() -> Result<(), String> {
    // SAFETY: a plain call.
    if unsafe { AXIsProcessTrusted() } {
        Ok(())
    } else {
        Err("macOS has not allowed this app to control the computer. Open System Settings > Privacy & Security > Accessibility, switch fLLM on (or the terminal that started the server), then restart the app".to_owned())
    }
}

/// Called when the user switches computer use on: asking makes macOS show its own dialog and list fLLM in System Settings.
pub fn request_access() {
    if optional_bool_fn(b"CGPreflightScreenCaptureAccess\0") == Some(false) {
        let _ = optional_bool_fn(b"CGRequestScreenCaptureAccess\0");
    }
}

pub fn available() -> Result<(), String> {
    screen_permission()?;
    input_permission()
}

pub fn displays() -> Result<Vec<DisplayInfo>, String> {
    let mut ids = [0u32; 16];
    let mut count = 0u32;
    // SAFETY: the buffer holds 16 ids and `count` receives how many were written.
    if unsafe { CGGetActiveDisplayList(16, ids.as_mut_ptr(), &mut count) } != 0 || count == 0 {
        return Err("macOS lists no display".to_owned());
    }
    // SAFETY: plain calls.
    let main = unsafe { CGMainDisplayID() };
    let mut list = Vec::new();
    for id in ids.iter().take(count as usize) {
        // SAFETY: the mode is released after use.
        let (bounds, pw, ph) = unsafe {
            let bounds = CGDisplayBounds(*id);
            let mode = CGDisplayCopyDisplayMode(*id);
            let (pw, ph) = if mode.is_null() { (bounds.size.width as usize, bounds.size.height as usize) } else { (CGDisplayModeGetPixelWidth(mode), CGDisplayModeGetPixelHeight(mode)) };
            if !mode.is_null() {
                CGDisplayModeRelease(mode);
            }
            (bounds, pw, ph)
        };
        let scale = if bounds.size.width > 0.0 { (pw as f64 / bounds.size.width).max(1.0) } else { 1.0 };
        list.push(DisplayInfo {
            index: 0,
            name: format!("display {id}"),
            primary: *id == main,
            origin_x: bounds.origin.x as i32,
            origin_y: bounds.origin.y as i32,
            width: pw as u32,
            height: ph as u32,
            scale,
            ui_scale: scale,
        });
    }
    list.sort_by_key(|d| (!d.primary, d.origin_x, d.origin_y));
    for (i, d) in list.iter_mut().enumerate() {
        d.index = i;
    }
    Ok(list)
}

/// The display id behind a [`DisplayInfo`] (found again by its position, because the list is sorted).
fn display_id(d: &DisplayInfo) -> Result<u32, String> {
    let mut ids = [0u32; 16];
    let mut count = 0u32;
    // SAFETY: as in `displays`.
    unsafe { CGGetActiveDisplayList(16, ids.as_mut_ptr(), &mut count) };
    for id in ids.iter().take(count as usize) {
        // SAFETY: a plain call.
        let b = unsafe { CGDisplayBounds(*id) };
        if b.origin.x as i32 == d.origin_x && b.origin.y as i32 == d.origin_y {
            return Ok(*id);
        }
    }
    Err("the display is gone; list the displays again".to_owned())
}

pub fn capture(d: &DisplayInfo, region: (u32, u32, u32, u32)) -> Result<Frame, String> {
    screen_permission()?;
    let id = display_id(d)?;
    // SAFETY: the image and its data are released before returning; the byte range is checked against the length.
    unsafe {
        let image = CGDisplayCreateImage(id);
        if image.is_null() {
            return Err("macOS returned no picture of the display (Screen Recording is not allowed for this app)".to_owned());
        }
        let (w, h, stride, bpp, info) = (CGImageGetWidth(image), CGImageGetHeight(image), CGImageGetBytesPerRow(image), CGImageGetBitsPerPixel(image), CGImageGetBitmapInfo(image));
        let data = CGDataProviderCopyData(CGImageGetDataProvider(image));
        if data.is_null() || bpp != 32 {
            if !data.is_null() {
                CFRelease(data);
            }
            CGImageRelease(image);
            return Err(format!("unsupported picture format ({bpp} bits per pixel)"));
        }
        let bytes = std::slice::from_raw_parts(CFDataGetBytePtr(data), CFDataGetLength(data).max(0) as usize);
        let (rx, ry, rw, rh) = (region.0 as usize, region.1 as usize, region.2 as usize, region.3 as usize);
        let result = if rx + rw > w || ry + rh > h || rw == 0 || rh == 0 || bytes.len() < stride * h {
            Err(format!("the region is outside the picture ({w}x{h})"))
        } else {
            // byte order 32 little: B G R A in memory, otherwise A R G B
            let little = info & 0x7000 == 0x2000;
            let mut rgba = Vec::with_capacity(rw * rh * 4);
            for row in ry..ry + rh {
                let (pixels, _) = bytes[row * stride + rx * 4..row * stride + (rx + rw) * 4].as_chunks::<4>();
                for px in pixels {
                    if little {
                        rgba.extend_from_slice(&[px[2], px[1], px[0], 255]);
                    } else {
                        rgba.extend_from_slice(&[px[1], px[2], px[3], 255]);
                    }
                }
            }
            Ok(Frame { width: rw as u32, height: rh as u32, rgba })
        };
        CFRelease(data);
        CGImageRelease(image);
        result
    }
}

pub fn cursor() -> Result<(i32, i32), String> {
    // SAFETY: the event is released after the position was read.
    unsafe {
        let e = CGEventCreate(std::ptr::null_mut());
        if e.is_null() {
            return Err("cannot read the pointer position".to_owned());
        }
        let p = CGEventGetLocation(e);
        CFRelease(e);
        Ok((p.x.round() as i32, p.y.round() as i32))
    }
}

fn post(event: Ref) -> Result<(), String> {
    if event.is_null() {
        return Err("macOS refused to create the input event".to_owned());
    }
    // SAFETY: a valid event; it is released after posting.
    unsafe {
        CGEventPost(HID_TAP, event);
        CFRelease(event);
    }
    Ok(())
}

/// The button that is held right now (moves are drags then), and the click counter for double clicks.
struct MouseState {
    held: Option<Button>,
    last_up: Option<(Instant, Button, i64)>,
    clicks: i64,
}
static MOUSE: Mutex<MouseState> = Mutex::new(MouseState { held: None, last_up: None, clicks: 1 });

fn button_index(b: Button) -> u32 {
    match b {
        Button::Left => 0,
        Button::Right => 1,
        Button::Middle => 2,
    }
}

fn mouse_event(kind: u32, x: f64, y: f64, button: Button, clicks: Option<i64>) -> Result<(), String> {
    // SAFETY: a plain call; the result is checked in `post`.
    let e = unsafe { CGEventCreateMouseEvent(std::ptr::null_mut(), kind, CGPoint { x, y }, button_index(button)) };
    if let (false, Some(n)) = (e.is_null(), clicks) {
        // SAFETY: `e` is a valid event.
        unsafe { CGEventSetIntegerValueField(e, CLICK_STATE_FIELD, n) };
    }
    post(e)
}

pub fn move_to(x: i32, y: i32) -> Result<(), String> {
    input_permission()?;
    let held = MOUSE.lock().unwrap_or_else(|e| e.into_inner()).held;
    let kind = match held {
        None => MOUSE_MOVED,
        Some(Button::Left) => 6,
        Some(Button::Right) => 7,
        Some(Button::Middle) => 27,
    };
    mouse_event(kind, f64::from(x), f64::from(y), held.unwrap_or(Button::Left), None)
}

pub fn button(b: Button, down: bool) -> Result<(), String> {
    input_permission()?;
    let (x, y) = cursor()?;
    let mut s = MOUSE.lock().unwrap_or_else(|e| e.into_inner());
    if down {
        s.clicks = match s.last_up {
            Some((t, lb, n)) if lb == b && t.elapsed() < Duration::from_millis(500) => (n + 1).min(3),
            _ => 1,
        };
        s.held = Some(b);
    } else {
        s.held = None;
        s.last_up = Some((Instant::now(), b, s.clicks));
    }
    let kind = match (b, down) {
        (Button::Left, true) => 1,
        (Button::Left, false) => 2,
        (Button::Right, true) => 3,
        (Button::Right, false) => 4,
        (Button::Middle, true) => 25,
        (Button::Middle, false) => 26,
    };
    let clicks = s.clicks;
    drop(s);
    mouse_event(kind, f64::from(x), f64::from(y), b, Some(clicks))
}

pub fn scroll(dx: i32, dy: i32) -> Result<(), String> {
    input_permission()?;
    // lines; a positive wheel value scrolls up / to the left
    // SAFETY: a plain call; the result is checked in `post`.
    let e = unsafe { CGEventCreateScrollWheelEvent2(std::ptr::null_mut(), 1, 2, -dy, -dx, 0) };
    post(e)
}

/// ANSI keyboard virtual key codes of the characters that can be part of a shortcut.
fn char_keycode(c: char) -> Option<u16> {
    Some(match c {
        'a' => 0, 's' => 1, 'd' => 2, 'f' => 3, 'h' => 4, 'g' => 5, 'z' => 6, 'x' => 7, 'c' => 8, 'v' => 9, 'b' => 11, 'q' => 12, 'w' => 13,
        'e' => 14, 'r' => 15, 'y' => 16, 't' => 17, '1' => 18, '2' => 19, '3' => 20, '4' => 21, '6' => 22, '5' => 23, '=' | '+' => 24, '9' => 25,
        '7' => 26, '-' => 27, '8' => 28, '0' => 29, ']' => 30, 'o' => 31, 'u' => 32, '[' => 33, 'i' => 34, 'p' => 35, 'l' => 37, 'j' => 38,
        '\'' => 39, 'k' => 40, ';' => 41, '\\' => 42, ',' => 43, '/' => 44, 'n' => 45, 'm' => 46, '.' => 47, '`' => 50,
        _ => return None,
    })
}

fn named_keycode(n: Named) -> Option<u16> {
    Some(match n {
        Named::Enter => 36,
        Named::Tab => 48,
        Named::Space => 49,
        Named::Backspace => 51,
        Named::Escape => 53,
        Named::Meta => 55,
        Named::CapsLock => 57,
        Named::Delete => 117,
        Named::Insert => 114,
        Named::Home => 115,
        Named::End => 119,
        Named::PageUp => 116,
        Named::PageDown => 121,
        Named::Left => 123,
        Named::Right => 124,
        Named::Down => 125,
        Named::Up => 126,
        Named::F(n) => match n {
            1 => 122,
            2 => 120,
            3 => 99,
            4 => 118,
            5 => 96,
            6 => 97,
            7 => 98,
            8 => 100,
            9 => 101,
            10 => 109,
            11 => 103,
            12 => 111,
            _ => return None,
        },
        Named::PrintScreen | Named::Menu => return None,
    })
}

pub fn press(chord: &Chord) -> Result<(), String> {
    input_permission()?;
    let code = match chord.key {
        Key::Named(n) => named_keycode(n).ok_or_else(|| format!("the {n:?} key does not exist on a Mac keyboard"))?,
        Key::Char(c) => char_keycode(c).ok_or_else(|| format!("cannot type '{c}' as a key on this keyboard layout"))?,
    };
    let mut flags = 0u64;
    if chord.ctrl {
        flags |= FLAG_CONTROL;
    }
    if chord.alt {
        flags |= FLAG_ALT;
    }
    if chord.shift {
        flags |= FLAG_SHIFT;
    }
    if chord.meta && chord.key != Key::Named(Named::Meta) {
        flags |= FLAG_COMMAND;
    }
    for down in [true, false] {
        // SAFETY: a plain call; the result is checked in `post`.
        let e = unsafe { CGEventCreateKeyboardEvent(std::ptr::null_mut(), code, down) };
        if !e.is_null() {
            // SAFETY: `e` is a valid event.
            unsafe { CGEventSetFlags(e, flags) };
        }
        post(e)?;
    }
    Ok(())
}

pub fn type_char(c: char) -> Result<(), String> {
    input_permission()?;
    let mut units = [0u16; 2];
    let text = c.encode_utf16(&mut units);
    for down in [true, false] {
        // SAFETY: the string is copied by the call; the result is checked in `post`.
        let e = unsafe { CGEventCreateKeyboardEvent(std::ptr::null_mut(), 0, down) };
        if !e.is_null() {
            // SAFETY: `e` is a valid event and `text` is valid for its length.
            unsafe { CGEventKeyboardSetUnicodeString(e, text.len(), text.as_ptr()) };
        }
        post(e)?;
    }
    Ok(())
}

/// Whether Ctrl+Alt+Shift+Esc is held down right now (the emergency stop).
pub fn panic_down() -> bool {
    let down = |codes: [u16; 2]| codes.iter().any(|c| {
        // SAFETY: a plain call (state 0 = the combined session state).
        unsafe { CGEventSourceKeyState(0, *c) }
    });
    down([59, 62]) && down([58, 61]) && down([56, 60]) && down([53, 53])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shortcut_keys_have_mac_key_codes() {
        assert_eq!(char_keycode('a'), Some(0));
        assert_eq!(char_keycode('s'), Some(1));
        assert_eq!(char_keycode('z'), Some(6));
        assert_eq!(char_keycode('é'), None);
        assert_eq!(named_keycode(Named::Enter), Some(36));
        assert_eq!(named_keycode(Named::F(5)), Some(96));
        assert_eq!(named_keycode(Named::F(13)), None);
    }
}
