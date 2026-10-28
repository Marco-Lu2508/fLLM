//! Computer use on Windows: GDI `BitBlt` for screenshots, `SendInput` for the mouse and the keyboard, `GetAsyncKeyState`
//! for the emergency stop chord. Thin glue over `windows-sys` (already a dependency); the decisions (coordinates, chords,
//! limits) are made in `agent_computer` and unit tested there.
//!
//! NOT VERIFIED ON A REAL WINDOWS MACHINE: the code is cross-compiled and lint-checked for `x86_64-pc-windows-msvc` only.
//! Things to know: the process makes itself per-monitor DPI aware so that coordinates are physical pixels; windows that run
//! as administrator do not accept input from a normal process (UIPI); protected video content is captured black; the lock
//! screen and the secure desktop (UAC prompts) cannot be seen or operated.
use super::agent_computer::{Button, Chord, DisplayInfo, Frame, Key, Named};
use std::mem::{size_of, zeroed};
use std::ptr::{null, null_mut};
use std::sync::Once;
use windows_sys::Win32::Foundation::{LPARAM, POINT, RECT};
use windows_sys::Win32::Graphics::Gdi::{
    BitBlt, CreateCompatibleBitmap, CreateCompatibleDC, DeleteDC, DeleteObject, EnumDisplayMonitors, GetDC, GetDIBits, GetMonitorInfoW, ReleaseDC, SelectObject,
    BITMAPINFO, BITMAPINFOHEADER, BI_RGB, CAPTUREBLT, DIB_RGB_COLORS, HDC, HMONITOR, MONITORINFO, MONITORINFOEXW, SRCCOPY,
};
use windows_sys::Win32::UI::HiDpi::{GetDpiForMonitor, SetProcessDpiAwarenessContext, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, MDT_EFFECTIVE_DPI};
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, SendInput, VkKeyScanW, INPUT, INPUT_0, INPUT_KEYBOARD, INPUT_MOUSE, KEYBDINPUT, KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, KEYEVENTF_UNICODE,
    MOUSEEVENTF_HWHEEL, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP, MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP, MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP,
    MOUSEEVENTF_WHEEL, MOUSEINPUT,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{GetCursorPos, SetCursorPos};

pub fn platform() -> &'static str {
    "windows"
}

/// Coordinates are physical pixels only when the process is per-monitor DPI aware; ask for it once (it fails harmlessly when the
/// application manifest already decided).
fn dpi_aware() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // SAFETY: a plain call without pointers.
        unsafe {
            SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        }
    });
}

/// Nothing to ask for on Windows.
pub fn request_access() {}

pub fn available() -> Result<(), String> {
    dpi_aware();
    // SAFETY: GetDC(NULL) is the device context of the whole screen; it is released right away.
    let dc = unsafe { GetDC(null_mut()) };
    if dc.is_null() {
        return Err("Windows gives this process no access to the screen (a service without a desktop, or a locked session)".to_owned());
    }
    // SAFETY: the context was obtained above.
    unsafe { ReleaseDC(null_mut(), dc) };
    Ok(())
}

unsafe extern "system" fn collect_monitor(monitor: HMONITOR, _dc: HDC, _rect: *mut RECT, data: LPARAM) -> i32 {
    // SAFETY: `data` is the address of the Vec that `displays` passes and keeps alive during the enumeration.
    let list = unsafe { &mut *(data as *mut Vec<DisplayInfo>) };
    let mut info: MONITORINFOEXW = unsafe { zeroed() };
    info.monitorInfo.cbSize = size_of::<MONITORINFOEXW>() as u32;
    // SAFETY: MONITORINFOEXW starts with a MONITORINFO and cbSize says how large it is.
    if unsafe { GetMonitorInfoW(monitor, (&mut info as *mut MONITORINFOEXW).cast::<MONITORINFO>()) } == 0 {
        return 1;
    }
    let r = info.monitorInfo.rcMonitor;
    let (mut dpi_x, mut dpi_y) = (96u32, 96u32);
    // SAFETY: both out parameters are valid for the call.
    unsafe { GetDpiForMonitor(monitor, MDT_EFFECTIVE_DPI, &mut dpi_x, &mut dpi_y) };
    let name_len = info.szDevice.iter().position(|c| *c == 0).unwrap_or(info.szDevice.len());
    let name = String::from_utf16_lossy(&info.szDevice[..name_len]).trim_start_matches("\\\\.\\").to_owned();
    list.push(DisplayInfo {
        index: list.len(),
        name,
        primary: info.monitorInfo.dwFlags & 1 != 0,
        origin_x: r.left,
        origin_y: r.top,
        width: (r.right - r.left).max(0) as u32,
        height: (r.bottom - r.top).max(0) as u32,
        scale: 1.0,
        ui_scale: f64::from(dpi_x.max(96)) / 96.0,
    });
    1
}

pub fn displays() -> Result<Vec<DisplayInfo>, String> {
    dpi_aware();
    let mut list: Vec<DisplayInfo> = Vec::new();
    // SAFETY: the callback only touches the Vec behind the pointer, which outlives the call.
    let ok = unsafe { EnumDisplayMonitors(null_mut(), null(), Some(collect_monitor), (&mut list as *mut Vec<DisplayInfo>) as LPARAM) };
    if ok == 0 || list.is_empty() {
        return Err("Windows lists no display".to_owned());
    }
    // the main display first, then by position
    list.sort_by_key(|d| (!d.primary, d.origin_x, d.origin_y));
    for (i, d) in list.iter_mut().enumerate() {
        d.index = i;
    }
    Ok(list)
}

pub fn capture(d: &DisplayInfo, region: (u32, u32, u32, u32)) -> Result<Frame, String> {
    dpi_aware();
    let (rx, ry, rw, rh) = region;
    if rw == 0 || rh == 0 {
        return Err("the region is empty".to_owned());
    }
    let (w, h) = (rw as i32, rh as i32);
    // SAFETY: every handle is created and released in this block; the pixel buffer is as large as GetDIBits is told.
    unsafe {
        let screen = GetDC(null_mut());
        if screen.is_null() {
            return Err("cannot open the screen".to_owned());
        }
        let memory = CreateCompatibleDC(screen);
        let bitmap = CreateCompatibleBitmap(screen, w, h);
        if memory.is_null() || bitmap.is_null() {
            if !bitmap.is_null() {
                DeleteObject(bitmap);
            }
            if !memory.is_null() {
                DeleteDC(memory);
            }
            ReleaseDC(null_mut(), screen);
            return Err("cannot create a picture of this size".to_owned());
        }
        let old = SelectObject(memory, bitmap);
        let copied = BitBlt(memory, 0, 0, w, h, screen, d.origin_x + rx as i32, d.origin_y + ry as i32, SRCCOPY | CAPTUREBLT);
        let mut bgra = vec![0u8; rw as usize * rh as usize * 4];
        let mut info: BITMAPINFO = zeroed();
        info.bmiHeader.biSize = size_of::<BITMAPINFOHEADER>() as u32;
        info.bmiHeader.biWidth = w;
        info.bmiHeader.biHeight = -h; // top-down rows
        info.bmiHeader.biPlanes = 1;
        info.bmiHeader.biBitCount = 32;
        info.bmiHeader.biCompression = BI_RGB;
        // the bitmap must not be selected into a device context while GetDIBits reads it
        SelectObject(memory, old);
        let lines = if copied != 0 { GetDIBits(memory, bitmap, 0, rh, bgra.as_mut_ptr().cast(), &mut info, DIB_RGB_COLORS) } else { 0 };
        DeleteObject(bitmap);
        DeleteDC(memory);
        ReleaseDC(null_mut(), screen);
        if copied == 0 || lines == 0 {
            return Err("Windows refused to copy the screen (a locked session, the secure desktop or a remote session without a display)".to_owned());
        }
        for px in bgra.chunks_exact_mut(4) {
            px.swap(0, 2);
            px[3] = 255;
        }
        Ok(Frame { width: rw, height: rh, rgba: bgra })
    }
}

pub fn cursor() -> Result<(i32, i32), String> {
    dpi_aware();
    let mut p = POINT { x: 0, y: 0 };
    // SAFETY: `p` is valid for the call.
    if unsafe { GetCursorPos(&mut p) } == 0 {
        return Err("cannot read the pointer position".to_owned());
    }
    Ok((p.x, p.y))
}

pub fn move_to(x: i32, y: i32) -> Result<(), String> {
    dpi_aware();
    // SAFETY: a plain call.
    if unsafe { SetCursorPos(x, y) } == 0 {
        return Err("Windows refused to move the pointer (the secure desktop is showing, or the window in front runs as administrator)".to_owned());
    }
    Ok(())
}

fn send(inputs: &[INPUT]) -> Result<(), String> {
    // SAFETY: the slice is valid for `len` INPUT structures of the given size.
    let sent = unsafe { SendInput(inputs.len() as u32, inputs.as_ptr(), size_of::<INPUT>() as i32) };
    if sent as usize != inputs.len() {
        return Err("Windows blocked the input (another program with higher rights is in front, or the secure desktop is showing)".to_owned());
    }
    Ok(())
}

fn mouse(flags: u32, data: u32) -> INPUT {
    INPUT { r#type: INPUT_MOUSE, Anonymous: INPUT_0 { mi: MOUSEINPUT { dx: 0, dy: 0, mouseData: data, dwFlags: flags, time: 0, dwExtraInfo: 0 } } }
}

fn key_event(vk: u16, scan: u16, flags: u32) -> INPUT {
    INPUT { r#type: INPUT_KEYBOARD, Anonymous: INPUT_0 { ki: KEYBDINPUT { wVk: vk, wScan: scan, dwFlags: flags, time: 0, dwExtraInfo: 0 } } }
}

pub fn button(b: Button, down: bool) -> Result<(), String> {
    let flag = match (b, down) {
        (Button::Left, true) => MOUSEEVENTF_LEFTDOWN,
        (Button::Left, false) => MOUSEEVENTF_LEFTUP,
        (Button::Right, true) => MOUSEEVENTF_RIGHTDOWN,
        (Button::Right, false) => MOUSEEVENTF_RIGHTUP,
        (Button::Middle, true) => MOUSEEVENTF_MIDDLEDOWN,
        (Button::Middle, false) => MOUSEEVENTF_MIDDLEUP,
    };
    send(&[mouse(flag, 0)])
}

pub fn scroll(dx: i32, dy: i32) -> Result<(), String> {
    const NOTCH: i32 = 120;
    let mut events = Vec::new();
    if dy != 0 {
        // a positive wheel delta scrolls up; the tool's positive dy means down
        events.push(mouse(MOUSEEVENTF_WHEEL, (-dy * NOTCH) as u32));
    }
    if dx != 0 {
        events.push(mouse(MOUSEEVENTF_HWHEEL, (dx * NOTCH) as u32));
    }
    send(&events)
}

const VK_SHIFT: u16 = 0x10;
const VK_CONTROL: u16 = 0x11;
const VK_MENU: u16 = 0x12;
const VK_LWIN: u16 = 0x5B;
const VK_ESCAPE: u16 = 0x1B;

/// The virtual-key code of a named key and whether it is an "extended" key.
fn named_vk(n: Named) -> (u16, bool) {
    match n {
        Named::Enter => (0x0D, false),
        Named::Tab => (0x09, false),
        Named::Escape => (VK_ESCAPE, false),
        Named::Backspace => (0x08, false),
        Named::Delete => (0x2E, true),
        Named::Insert => (0x2D, true),
        Named::Home => (0x24, true),
        Named::End => (0x23, true),
        Named::PageUp => (0x21, true),
        Named::PageDown => (0x22, true),
        Named::Left => (0x25, true),
        Named::Up => (0x26, true),
        Named::Right => (0x27, true),
        Named::Down => (0x28, true),
        Named::Space => (0x20, false),
        Named::CapsLock => (0x14, false),
        Named::PrintScreen => (0x2C, true),
        Named::Menu => (0x5D, true),
        Named::Meta => (VK_LWIN, true),
        Named::F(n) => (0x70 + u16::from(n.clamp(1, 24)) - 1, false),
    }
}

pub fn press(chord: &Chord) -> Result<(), String> {
    let (vk, extended, needs_shift) = match chord.key {
        Key::Named(n) => {
            let (vk, ext) = named_vk(n);
            (vk, ext, false)
        }
        Key::Char(c) => {
            let mut units = [0u16; 2];
            let unit = c.encode_utf16(&mut units);
            if unit.len() != 1 {
                return Err(format!("cannot type '{c}' as a key"));
            }
            // SAFETY: a plain call.
            let scan = unsafe { VkKeyScanW(unit[0]) };
            if scan == -1 {
                return Err(format!("cannot type '{c}' with the keyboard layout of this computer"));
            }
            ((scan & 0xff) as u16, false, scan & 0x100 != 0)
        }
    };
    let mut mods = Vec::new();
    if chord.ctrl {
        mods.push(VK_CONTROL);
    }
    if chord.alt {
        mods.push(VK_MENU);
    }
    if chord.shift || needs_shift {
        mods.push(VK_SHIFT);
    }
    if chord.meta && chord.key != Key::Named(Named::Meta) {
        mods.push(VK_LWIN);
    }
    let ext = if extended { KEYEVENTF_EXTENDEDKEY } else { 0 };
    let mut events = Vec::new();
    for m in &mods {
        events.push(key_event(*m, 0, if *m == VK_LWIN { KEYEVENTF_EXTENDEDKEY } else { 0 }));
    }
    events.push(key_event(vk, 0, ext));
    events.push(key_event(vk, 0, ext | KEYEVENTF_KEYUP));
    for m in mods.iter().rev() {
        events.push(key_event(*m, 0, KEYEVENTF_KEYUP | if *m == VK_LWIN { KEYEVENTF_EXTENDEDKEY } else { 0 }));
    }
    send(&events)
}

/// Types one character as a Unicode event: independent of the keyboard layout.
pub fn type_char(c: char) -> Result<(), String> {
    let mut units = [0u16; 2];
    let mut events = Vec::new();
    for unit in c.encode_utf16(&mut units).iter() {
        events.push(key_event(0, *unit, KEYEVENTF_UNICODE));
        events.push(key_event(0, *unit, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP));
    }
    send(&events)
}

fn down(vk: u16) -> bool {
    // SAFETY: a plain call.
    (unsafe { GetAsyncKeyState(i32::from(vk)) } as u16) & 0x8000 != 0
}

/// Whether Ctrl+Alt+Shift+Esc is held down right now (the emergency stop).
pub fn panic_down() -> bool {
    down(VK_CONTROL) && down(VK_MENU) && down(VK_SHIFT) && down(VK_ESCAPE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn function_keys_and_extended_keys_map_to_virtual_keys() {
        assert_eq!(named_vk(Named::F(1)), (0x70, false));
        assert_eq!(named_vk(Named::F(12)), (0x7B, false));
        assert_eq!(named_vk(Named::Left), (0x25, true));
        assert_eq!(named_vk(Named::Enter), (0x0D, false));
        assert_eq!(named_vk(Named::Meta), (0x5B, true));
    }
}
