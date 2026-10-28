//! Computer use on Linux through the X11 wire protocol, written directly against the protocol (a Unix socket, a handful
//! of requests) so that it needs no extra crate and no system library: `GetImage` for screenshots, the XTEST extension
//! for the mouse and the keyboard, `QueryPointer` and `QueryKeymap` for the pointer and the panic chord, and RANDR for the
//! list of monitors.
//!
//! Only a local X server is supported (`DISPLAY=:N`, Xorg, Xvfb, XWayland is refused). A Wayland session cannot be
//! screenshot or driven this way: the compositor does not allow it without a portal, and XWayland only shows X11 windows.
use super::agent_computer::{Button, Chord, DisplayInfo, Frame, Key, Named};
use std::{
    collections::HashMap,
    io::{Read, Write},
    os::unix::net::UnixStream,
    sync::Mutex,
    time::Duration,
};

const IO_TIMEOUT: Duration = Duration::from_secs(5);

pub fn platform() -> &'static str {
    "x11"
}

// ---- the connection -------------------------------------------------------------------------------

struct Conn {
    stream: UnixStream,
    seq: u16,
    /// Errors that arrived for requests without a reply; reported by the next [`Conn::sync`].
    errors: Vec<String>,
    root: u32,
    width: u16,
    height: u16,
    min_keycode: u8,
    max_keycode: u8,
    /// Bits per pixel and scanline padding (bits) of the root depth, and the byte order of the image data.
    bpp: u8,
    pad: u8,
    lsb_first: bool,
    masks: (u32, u32, u32),
    xtest: Option<u8>,
    randr: Option<u8>,
    /// keysym -> (keycode, column in the keymap), the lowest column first.
    keysyms: HashMap<u32, (u8, usize)>,
}

fn u16_at(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}
fn u32_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}
fn i16_at(b: &[u8], i: usize) -> i16 {
    i16::from_le_bytes([b[i], b[i + 1]])
}
fn padded(n: usize) -> usize {
    (n + 3) & !3
}

/// `:99`, `:0.0` -> the display number. Remote displays are refused.
fn display_number(display: &str) -> Result<u32, String> {
    let rest = display.strip_prefix("unix").unwrap_or(display);
    let Some(num) = rest.strip_prefix(':') else {
        return Err(format!("DISPLAY={display} names a remote X server; only a local one (DISPLAY=:0) is supported"));
    };
    num.split('.').next().unwrap_or("").parse().map_err(|_| format!("DISPLAY={display} is not understood"))
}

fn hostname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname").map(|s| s.trim().to_owned()).unwrap_or_default()
}

/// The MIT-MAGIC-COOKIE-1 of the display from the Xauthority file, when there is one.
fn cookie(number: u32) -> Option<Vec<u8>> {
    let path = std::env::var_os("XAUTHORITY").map(std::path::PathBuf::from).or_else(|| std::env::var_os("HOME").map(|h| std::path::Path::new(&h).join(".Xauthority")))?;
    let data = std::fs::read(path).ok()?;
    find_cookie(&data, number, &hostname())
}

/// Parses an Xauthority file (big-endian counted strings) and picks the cookie of this display.
fn find_cookie(data: &[u8], number: u32, host: &str) -> Option<Vec<u8>> {
    fn take<'a>(data: &'a [u8], i: &mut usize) -> Option<&'a [u8]> {
        let len = u16::from_be_bytes([*data.get(*i)?, *data.get(*i + 1)?]) as usize;
        let slice = data.get(*i + 2..*i + 2 + len)?;
        *i += 2 + len;
        Some(slice)
    }
    let mut i = 0usize;
    let mut best: Option<Vec<u8>> = None;
    while i + 2 <= data.len() {
        let family = u16::from_be_bytes([data[i], data[i + 1]]);
        i += 2;
        let address = take(data, &mut i)?;
        let num = take(data, &mut i)?;
        let name = take(data, &mut i)?;
        let value = take(data, &mut i)?;
        let num_ok = num.is_empty() || std::str::from_utf8(num).ok().and_then(|s| s.parse::<u32>().ok()) == Some(number);
        if !num_ok || name != b"MIT-MAGIC-COOKIE-1" {
            continue;
        }
        // family 256 is "local" (the address is the host name), 65535 is "wild"
        if family == 256 && address == host.as_bytes() {
            return Some(value.to_vec());
        }
        if family == 65535 || family == 256 {
            best.get_or_insert_with(|| value.to_vec());
        }
    }
    best
}

impl Conn {
    fn open() -> Result<Conn, String> {
        if std::env::var_os("WAYLAND_DISPLAY").is_some() || std::env::var("XDG_SESSION_TYPE").is_ok_and(|v| v.eq_ignore_ascii_case("wayland")) {
            return Err("this is a Wayland session: Wayland does not let a program look at the screen or press keys for other programs (only XWayland windows would be visible). Log in to an X11 session (\"Xorg\" in the login screen) to use computer use".to_owned());
        }
        let display = std::env::var("DISPLAY").map_err(|_| "there is no graphical session: DISPLAY is not set (a server without a desktop, or a service that does not run inside the desktop session of the user)".to_owned())?;
        let number = display_number(&display)?;
        let path = format!("/tmp/.X11-unix/X{number}");
        let stream = UnixStream::connect(&path)
            .or_else(|first| {
                // the abstract socket of the same name (Linux)
                use std::os::linux::net::SocketAddrExt;
                std::os::unix::net::SocketAddr::from_abstract_name(path.as_bytes()).and_then(|a| UnixStream::connect_addr(&a)).map_err(|_| first)
            })
            .map_err(|e| format!("cannot connect to the X server {display}: {e}"))?;
        stream.set_read_timeout(Some(IO_TIMEOUT)).map_err(|e| e.to_string())?;
        stream.set_write_timeout(Some(IO_TIMEOUT)).map_err(|e| e.to_string())?;
        let mut conn = Conn {
            stream,
            seq: 0,
            errors: Vec::new(),
            root: 0,
            width: 0,
            height: 0,
            min_keycode: 8,
            max_keycode: 255,
            bpp: 32,
            pad: 32,
            lsb_first: true,
            masks: (0xff0000, 0xff00, 0xff),
            xtest: None,
            randr: None,
            keysyms: HashMap::new(),
        };
        conn.handshake(&cookie(number).unwrap_or_default())?;
        conn.xtest = conn.extension("XTEST")?;
        conn.randr = conn.extension("RANDR")?;
        conn.load_keymap()?;
        Ok(conn)
    }

    fn handshake(&mut self, cookie: &[u8]) -> Result<(), String> {
        let name: &[u8] = if cookie.is_empty() { b"" } else { b"MIT-MAGIC-COOKIE-1" };
        let mut req = vec![b'l', 0, 11, 0, 0, 0];
        req.extend_from_slice(&(name.len() as u16).to_le_bytes());
        req.extend_from_slice(&(cookie.len() as u16).to_le_bytes());
        req.extend_from_slice(&[0, 0]);
        req.extend_from_slice(name);
        req.resize(req.len() + padded(name.len()) - name.len(), 0);
        req.extend_from_slice(cookie);
        req.resize(req.len() + padded(cookie.len()) - cookie.len(), 0);
        self.stream.write_all(&req).map_err(|e| format!("X server: {e}"))?;
        let mut head = [0u8; 8];
        self.stream.read_exact(&mut head).map_err(|e| format!("X server: {e}"))?;
        let mut body = vec![0u8; u16_at(&head, 6) as usize * 4];
        self.stream.read_exact(&mut body).map_err(|e| format!("X server: {e}"))?;
        if head[0] != 1 {
            let reason = String::from_utf8_lossy(body.get(..head[1] as usize).unwrap_or(&[])).trim().to_owned();
            return Err(format!("the X server refused the connection: {reason}. If it asks for authorization, run the app as the user of the desktop session (XAUTHORITY must point to the cookie file)"));
        }
        parse_setup(&body, self)
    }

    fn send(&mut self, bytes: &[u8]) -> Result<u16, String> {
        self.stream.write_all(bytes).map_err(|e| format!("X server: {e}"))?;
        self.seq = self.seq.wrapping_add(1);
        Ok(self.seq)
    }

    /// Reads messages until the reply of request `seq`; errors of other requests are remembered, events are dropped.
    fn reply(&mut self, seq: u16) -> Result<Vec<u8>, String> {
        loop {
            let mut head = [0u8; 32];
            self.stream.read_exact(&mut head).map_err(|e| format!("X server: {e}"))?;
            match head[0] {
                0 => {
                    let text = format!("X error {} (request {}.{})", head[1], head[10], u16_at(&head, 8));
                    if u16_at(&head, 2) == seq {
                        return Err(text);
                    }
                    self.errors.push(text);
                }
                1 => {
                    let extra = u32_at(&head, 4) as usize * 4;
                    let mut buf = head.to_vec();
                    if extra > 0 {
                        if extra > 256 << 20 {
                            return Err("X server: the reply is too large".to_owned());
                        }
                        buf.resize(32 + extra, 0);
                        self.stream.read_exact(&mut buf[32..]).map_err(|e| format!("X server: {e}"))?;
                    }
                    if u16_at(&head, 2) == seq {
                        return Ok(buf);
                    }
                }
                // an event: not wanted (nothing is selected), but GenericEvent has extra data
                35 => {
                    let extra = u32_at(&head, 4) as usize * 4;
                    let mut skip = vec![0u8; extra.min(1 << 20)];
                    let _ = self.stream.read_exact(&mut skip);
                }
                _ => {}
            }
        }
    }

    /// A round trip: everything sent before has been processed; errors of earlier requests are reported.
    fn sync(&mut self) -> Result<(), String> {
        let seq = self.send(&[43, 0, 1, 0])?; // GetInputFocus
        self.reply(seq)?;
        match self.errors.drain(..).next() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    fn extension(&mut self, name: &str) -> Result<Option<u8>, String> {
        let mut req = vec![98, 0];
        req.extend_from_slice(&((2 + padded(name.len()) / 4) as u16).to_le_bytes());
        req.extend_from_slice(&(name.len() as u16).to_le_bytes());
        req.extend_from_slice(&[0, 0]);
        req.extend_from_slice(name.as_bytes());
        req.resize(8 + padded(name.len()), 0);
        let seq = self.send(&req)?;
        let r = self.reply(seq)?;
        Ok(if r[8] != 0 { Some(r[9]) } else { None })
    }

    fn load_keymap(&mut self) -> Result<(), String> {
        let count = self.max_keycode - self.min_keycode + 1;
        let seq = self.send(&[101, 0, 2, 0, self.min_keycode, count, 0, 0])?;
        let r = self.reply(seq)?;
        let per = r[1] as usize;
        if per == 0 {
            return Ok(());
        }
        for (i, chunk) in r[32..].chunks_exact(4 * per).enumerate() {
            let keycode = self.min_keycode.saturating_add(i as u8);
            for col in 0..per {
                let sym = u32_at(chunk, col * 4);
                if sym == 0 {
                    continue;
                }
                let entry = self.keysyms.entry(sym).or_insert((keycode, col));
                if col < entry.1 {
                    *entry = (keycode, col);
                }
            }
        }
        Ok(())
    }

    fn fake(&mut self, kind: u8, detail: u8, x: i16, y: i16) -> Result<(), String> {
        let Some(major) = self.xtest else {
            return Err("the X server has no XTEST extension, which is needed to press keys and move the mouse".to_owned());
        };
        let mut req = vec![0u8; 36];
        req[0] = major;
        req[1] = 2; // XTestFakeInput
        req[2..4].copy_from_slice(&9u16.to_le_bytes());
        req[4] = kind;
        req[5] = detail;
        req[12..16].copy_from_slice(&self.root.to_le_bytes());
        req[24..26].copy_from_slice(&x.to_le_bytes());
        req[26..28].copy_from_slice(&y.to_le_bytes());
        self.send(&req)?;
        Ok(())
    }

    fn pointer(&mut self) -> Result<(i32, i32), String> {
        let mut req = vec![38, 0, 2, 0];
        req.extend_from_slice(&self.root.to_le_bytes());
        let seq = self.send(&req)?;
        let r = self.reply(seq)?;
        Ok((i32::from(i16_at(&r, 16)), i32::from(i16_at(&r, 18))))
    }

    fn keysym(&self, sym: u32) -> Option<(u8, usize)> {
        self.keysyms.get(&sym).copied()
    }
}

/// The part of the connection setup reply that is needed: the first screen and its pixel format.
fn parse_setup(body: &[u8], c: &mut Conn) -> Result<(), String> {
    if body.len() < 32 {
        return Err("X server: the setup reply is too short".to_owned());
    }
    let vendor = u16_at(body, 16) as usize;
    let screens = body[20] as usize;
    let formats = body[21] as usize;
    c.lsb_first = body[22] == 0;
    c.min_keycode = body[26];
    c.max_keycode = body[27];
    let formats_at = 32 + padded(vendor);
    let screen_at = formats_at + formats * 8;
    if screens == 0 || body.len() < screen_at + 40 {
        return Err("X server: no screen in the setup reply".to_owned());
    }
    let s = &body[screen_at..];
    c.root = u32_at(s, 0);
    c.width = u16_at(s, 20);
    c.height = u16_at(s, 22);
    let root_visual = u32_at(s, 32);
    let root_depth = s[38];
    let depths = s[39] as usize;
    for f in 0..formats {
        let at = formats_at + f * 8;
        if body[at] == root_depth {
            c.bpp = body[at + 1];
            c.pad = body[at + 2];
        }
    }
    let mut at = 40usize;
    for _ in 0..depths {
        if s.len() < at + 8 {
            break;
        }
        let visuals = u16_at(s, at + 2) as usize;
        at += 8;
        for _ in 0..visuals {
            if s.len() < at + 24 {
                break;
            }
            if u32_at(s, at) == root_visual {
                c.masks = (u32_at(s, at + 8), u32_at(s, at + 12), u32_at(s, at + 16));
            }
            at += 24;
        }
    }
    Ok(())
}

static CONN: Mutex<Option<Conn>> = Mutex::new(None);

/// Runs `f` on the shared connection (opened on first use); a failure closes it so that the next call starts clean.
fn with_conn<T>(f: impl FnOnce(&mut Conn) -> Result<T, String>) -> Result<T, String> {
    let mut guard = CONN.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Some(Conn::open()?);
    }
    let conn = guard.as_mut().ok_or("no connection")?;
    let result = f(conn);
    if result.is_err() {
        *guard = None;
    }
    result
}

// ---- the platform interface -----------------------------------------------------------------------

pub fn available() -> Result<(), String> {
    with_conn(|_| Ok(()))
}

pub fn displays() -> Result<Vec<DisplayInfo>, String> {
    with_conn(|c| {
        let root = DisplayInfo { index: 0, name: "screen".to_owned(), primary: true, origin_x: 0, origin_y: 0, width: u32::from(c.width), height: u32::from(c.height), scale: 1.0, ui_scale: 1.0 };
        let Some(major) = c.randr else { return Ok(vec![root]) };
        // RRQueryVersion(1, 5), then RRGetMonitors (RANDR 1.5)
        let mut q = vec![major, 0, 3, 0];
        q.extend_from_slice(&1u32.to_le_bytes());
        q.extend_from_slice(&5u32.to_le_bytes());
        let seq = c.send(&q)?;
        if c.reply(seq).is_err() {
            return Ok(vec![root]);
        }
        let mut g = vec![major, 42, 3, 0];
        g.extend_from_slice(&c.root.to_le_bytes());
        g.extend_from_slice(&[1, 0, 0, 0]);
        let seq = c.send(&g)?;
        let Ok(r) = c.reply(seq) else { return Ok(vec![root]) };
        let n = u32_at(&r, 12) as usize;
        let mut at = 32usize;
        let mut list = Vec::new();
        for i in 0..n {
            if r.len() < at + 24 {
                break;
            }
            let outputs = u16_at(&r, at + 6) as usize;
            let (x, y, w, h) = (i32::from(i16_at(&r, at + 8)), i32::from(i16_at(&r, at + 10)), u32::from(u16_at(&r, at + 12)), u32::from(u16_at(&r, at + 14)));
            let primary = r[at + 4] != 0;
            at += 24 + outputs * 4;
            if w == 0 || h == 0 {
                continue;
            }
            list.push(DisplayInfo { index: i, name: format!("monitor {i}"), primary, origin_x: x, origin_y: y, width: w, height: h, scale: 1.0, ui_scale: 1.0 });
        }
        if list.is_empty() {
            return Ok(vec![root]);
        }
        // the main monitor first, then by position
        list.sort_by_key(|d| (!d.primary, d.origin_x, d.origin_y));
        for (i, d) in list.iter_mut().enumerate() {
            d.index = i;
            d.primary = i == 0;
        }
        Ok(list)
    })
}

pub fn capture(d: &DisplayInfo, region: (u32, u32, u32, u32)) -> Result<Frame, String> {
    with_conn(|c| {
        let (rx, ry, rw, rh) = region;
        let x = d.origin_x + rx as i32;
        let y = d.origin_y + ry as i32;
        if x < 0 || y < 0 || x as u32 + rw > u32::from(c.width) || y as u32 + rh > u32::from(c.height) || rw == 0 || rh == 0 {
            return Err(format!("the region {rw}x{rh} at {x},{y} is outside the screen ({}x{})", c.width, c.height));
        }
        let mut req = vec![73, 2, 5, 0];
        req.extend_from_slice(&c.root.to_le_bytes());
        req.extend_from_slice(&(x as i16).to_le_bytes());
        req.extend_from_slice(&(y as i16).to_le_bytes());
        req.extend_from_slice(&(rw as u16).to_le_bytes());
        req.extend_from_slice(&(rh as u16).to_le_bytes());
        req.extend_from_slice(&u32::MAX.to_le_bytes());
        let seq = c.send(&req)?;
        let r = c.reply(seq)?;
        let data = r.get(32..).ok_or("X server: empty image")?;
        let rgba = convert_pixels(data, rw as usize, rh as usize, c.bpp, c.pad, c.lsb_first, c.masks)?;
        Ok(Frame { width: rw, height: rh, rgba })
    })
}

/// X11 ZPixmap data -> RGBA, for the pixel formats that exist in practice (32, 24 and 16 bits per pixel).
fn convert_pixels(data: &[u8], w: usize, h: usize, bpp: u8, pad: u8, lsb_first: bool, masks: (u32, u32, u32)) -> Result<Vec<u8>, String> {
    let bytes = usize::from(bpp) / 8;
    if !matches!(bytes, 2..=4) {
        return Err(format!("the screen uses {bpp} bits per pixel, which is not supported"));
    }
    let pad_bits = usize::from(pad.max(8));
    let stride = (w * usize::from(bpp)).div_ceil(pad_bits) * pad_bits / 8;
    if data.len() < stride * h {
        return Err("X server: the image is shorter than expected".to_owned());
    }
    let channel = |mask: u32| -> (u32, u32) { (mask.trailing_zeros(), mask.count_ones().max(1)) };
    let (rs, rb) = channel(masks.0);
    let (gs, gb) = channel(masks.1);
    let (bs, bb) = channel(masks.2);
    let scale = |v: u32, bits: u32| -> u8 {
        if bits >= 8 {
            (v >> (bits - 8)) as u8
        } else {
            (v * 255 / ((1 << bits) - 1)) as u8
        }
    };
    let mut out = Vec::with_capacity(w * h * 4);
    for row in 0..h {
        let line = &data[row * stride..row * stride + w * bytes];
        for px in line.chunks_exact(bytes) {
            let mut v = 0u32;
            if lsb_first {
                for (i, b) in px.iter().enumerate() {
                    v |= u32::from(*b) << (8 * i);
                }
            } else {
                for b in px {
                    v = (v << 8) | u32::from(*b);
                }
            }
            out.push(scale((v & masks.0) >> rs, rb));
            out.push(scale((v & masks.1) >> gs, gb));
            out.push(scale((v & masks.2) >> bs, bb));
            out.push(255);
        }
    }
    Ok(out)
}

pub fn cursor() -> Result<(i32, i32), String> {
    with_conn(|c| c.pointer())
}

pub fn move_to(x: i32, y: i32) -> Result<(), String> {
    with_conn(|c| {
        c.fake(6, 0, x.clamp(0, i32::from(i16::MAX)) as i16, y.clamp(0, i32::from(i16::MAX)) as i16)?;
        c.sync()
    })
}

fn button_number(b: Button) -> u8 {
    match b {
        Button::Left => 1,
        Button::Middle => 2,
        Button::Right => 3,
    }
}

pub fn button(b: Button, down: bool) -> Result<(), String> {
    with_conn(|c| {
        c.fake(if down { 4 } else { 5 }, button_number(b), 0, 0)?;
        c.sync()
    })
}

pub fn scroll(dx: i32, dy: i32) -> Result<(), String> {
    with_conn(|c| {
        // wheel buttons: 4 up, 5 down, 6 left, 7 right
        for (amount, plus, minus) in [(dy, 5u8, 4u8), (dx, 7u8, 6u8)] {
            let b = if amount > 0 { plus } else { minus };
            for _ in 0..amount.unsigned_abs().min(50) {
                c.fake(4, b, 0, 0)?;
                c.fake(5, b, 0, 0)?;
            }
        }
        c.sync()
    })
}

fn named_keysym(n: Named) -> u32 {
    match n {
        Named::Enter => 0xff0d,
        Named::Tab => 0xff09,
        Named::Escape => 0xff1b,
        Named::Backspace => 0xff08,
        Named::Delete => 0xffff,
        Named::Insert => 0xff63,
        Named::Home => 0xff50,
        Named::End => 0xff57,
        Named::PageUp => 0xff55,
        Named::PageDown => 0xff56,
        Named::Up => 0xff52,
        Named::Down => 0xff54,
        Named::Left => 0xff51,
        Named::Right => 0xff53,
        Named::Space => 0x20,
        Named::CapsLock => 0xffe5,
        Named::PrintScreen => 0xff61,
        Named::Menu => 0xff67,
        Named::Meta => 0xffeb,
        Named::F(n) => 0xffbd + u32::from(n),
    }
}

/// The keysym of a character: Latin-1 as it is, everything else in the Unicode range of keysyms.
fn char_keysym(c: char) -> u32 {
    let code = c as u32;
    if (0x20..=0x7e).contains(&code) || (0xa0..=0xff).contains(&code) {
        code
    } else {
        0x0100_0000 + code
    }
}

const SHIFT_L: u32 = 0xffe1;
const CONTROL_L: u32 = 0xffe3;
const ALT_L: u32 = 0xffe9;
const SUPER_L: u32 = 0xffeb;
const LEVEL3: u32 = 0xfe03;

/// Presses the keys of `keysym` (with the modifiers it needs, plus `extra`) and releases them again.
fn tap(c: &mut Conn, keysym: u32, extra: &[u32], what: &str) -> Result<(), String> {
    let (code, col) = c.keysym(keysym).ok_or_else(|| format!("cannot type {what} with the keyboard layout of this computer"))?;
    let mut mods: Vec<u32> = extra.to_vec();
    // column 0: plain, 1: shift, 2: AltGr, 3: AltGr+shift
    if col % 2 == 1 && !mods.contains(&SHIFT_L) {
        mods.push(SHIFT_L);
    }
    if (col == 2 || col == 3) && !mods.contains(&LEVEL3) {
        mods.push(LEVEL3);
    }
    if col > 3 {
        return Err(format!("cannot type {what} with the keyboard layout of this computer"));
    }
    let mut codes = Vec::new();
    for m in &mods {
        codes.push(c.keysym(*m).ok_or_else(|| format!("the keyboard layout has no key for a modifier needed for {what}"))?.0);
    }
    for k in &codes {
        c.fake(2, *k, 0, 0)?;
    }
    c.fake(2, code, 0, 0)?;
    c.fake(3, code, 0, 0)?;
    for k in codes.iter().rev() {
        c.fake(3, *k, 0, 0)?;
    }
    c.sync()
}

pub fn press(chord: &Chord) -> Result<(), String> {
    with_conn(|c| {
        let mut mods = Vec::new();
        if chord.ctrl {
            mods.push(CONTROL_L);
        }
        if chord.alt {
            mods.push(ALT_L);
        }
        if chord.shift {
            mods.push(SHIFT_L);
        }
        if chord.meta && chord.key != Key::Named(Named::Meta) {
            mods.push(SUPER_L);
        }
        let sym = match chord.key {
            Key::Named(n) => named_keysym(n),
            Key::Char(ch) => char_keysym(ch),
        };
        let label = match chord.key {
            Key::Char(ch) => format!("'{ch}'"),
            Key::Named(n) => format!("the {n:?} key"),
        };
        tap(c, sym, &mods, &label)
    })
}

pub fn type_char(ch: char) -> Result<(), String> {
    with_conn(|c| tap(c, char_keysym(ch), &[], &format!("'{ch}'")))
}

/// Whether Ctrl+Alt+Shift+Esc is held down right now (the emergency stop). Uses its own connection.
pub fn panic_down() -> bool {
    static WATCH: Mutex<Option<Conn>> = Mutex::new(None);
    let mut guard = WATCH.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        match Conn::open() {
            Ok(c) => *guard = Some(c),
            Err(_) => {
                drop(guard);
                std::thread::sleep(Duration::from_secs(2));
                return false;
            }
        }
    }
    let Some(c) = guard.as_mut() else { return false };
    let result = (|| -> Result<bool, String> {
        let seq = c.send(&[44, 0, 1, 0])?; // QueryKeymap
        let r = c.reply(seq)?;
        let held = |sym: u32| c.keysym(sym).is_some_and(|(code, _)| r[8 + usize::from(code) / 8] & (1 << (code % 8)) != 0);
        // either side of each modifier counts
        let any = |a: u32, b: u32| held(a) || held(b);
        Ok(any(CONTROL_L, 0xffe4) && any(ALT_L, 0xffea) && any(SHIFT_L, 0xffe2) && held(0xff1b))
    })();
    match result {
        Ok(v) => v,
        Err(_) => {
            *guard = None;
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(family: u16, address: &str, number: &str, name: &str, data: &[u8]) -> Vec<u8> {
        let mut v = family.to_be_bytes().to_vec();
        for part in [address.as_bytes(), number.as_bytes(), name.as_bytes(), data] {
            v.extend_from_slice(&(part.len() as u16).to_be_bytes());
            v.extend_from_slice(part);
        }
        v
    }

    #[test]
    fn display_names_are_parsed() {
        assert_eq!(display_number(":0").unwrap(), 0);
        assert_eq!(display_number(":99.0").unwrap(), 99);
        assert_eq!(display_number("unix:1").unwrap(), 1);
        assert!(display_number("example.org:0").is_err());
        assert!(display_number("localhost:10.0").is_err(), "remote displays are refused");
        assert!(display_number(":x").is_err());
    }

    #[test]
    fn the_cookie_of_this_display_and_host_is_picked() {
        let mut file = entry(256, "otherhost", "0", "MIT-MAGIC-COOKIE-1", &[1; 16]);
        file.extend(entry(256, "myhost", "1", "MIT-MAGIC-COOKIE-1", &[2; 16]));
        file.extend(entry(256, "myhost", "0", "MIT-MAGIC-COOKIE-1", &[3; 16]));
        file.extend(entry(256, "myhost", "0", "XDM-AUTHORIZATION-1", &[4; 16]));
        assert_eq!(find_cookie(&file, 0, "myhost").unwrap(), vec![3; 16]);
        assert_eq!(find_cookie(&file, 1, "myhost").unwrap(), vec![2; 16]);
        // another host name: the first cookie of the display is the best guess (containers change the host name)
        assert_eq!(find_cookie(&file, 0, "renamed").unwrap(), vec![1; 16]);
        assert!(find_cookie(&file, 7, "myhost").is_none());
        assert!(find_cookie(&[], 0, "x").is_none());
        // a cut file does not panic
        assert!(find_cookie(&file[..file.len() - 5], 9, "myhost").is_none());
        let wild = entry(65535, "", "", "MIT-MAGIC-COOKIE-1", &[9; 16]);
        assert_eq!(find_cookie(&wild, 5, "x").unwrap(), vec![9; 16]);
    }

    #[test]
    fn pixels_of_the_common_formats_become_rgba() {
        // 32 bits, BGRX in memory (little endian 0x00RRGGBB)
        let data = [10u8, 20, 30, 0, 40, 50, 60, 0];
        let out = convert_pixels(&data, 2, 1, 32, 32, true, (0xff0000, 0xff00, 0xff)).unwrap();
        assert_eq!(out, vec![30, 20, 10, 255, 60, 50, 40, 255]);
        // 24 bits per pixel with rows padded to 32 bits
        let data = [1u8, 2, 3, 4, 5, 6, 0, 0, 7, 8, 9, 10, 11, 12, 0, 0];
        let out = convert_pixels(&data, 2, 2, 24, 32, true, (0xff0000, 0xff00, 0xff)).unwrap();
        assert_eq!(out.len(), 16);
        assert_eq!(&out[0..4], &[3, 2, 1, 255]);
        assert_eq!(&out[8..12], &[9, 8, 7, 255], "the second row starts after the padding");
        // 16 bits (5-6-5)
        let px = (0b11111u16 << 11).to_le_bytes();
        let out = convert_pixels(&px, 1, 1, 16, 16, true, (0xf800, 0x7e0, 0x1f)).unwrap();
        assert_eq!(out, vec![255, 0, 0, 255]);
        assert!(convert_pixels(&data, 2, 2, 8, 8, true, (0, 0, 0)).is_err());
        assert!(convert_pixels(&data[..4], 2, 2, 32, 32, true, (0xff0000, 0xff00, 0xff)).is_err());
    }

    #[test]
    fn keysyms_follow_the_x11_tables() {
        assert_eq!(char_keysym('a'), 0x61);
        assert_eq!(char_keysym('A'), 0x41);
        assert_eq!(char_keysym('é'), 0xe9);
        assert_eq!(char_keysym('€'), 0x0100_20ac);
        assert_eq!(named_keysym(Named::F(1)), 0xffbe);
        assert_eq!(named_keysym(Named::F(12)), 0xffc9);
        assert_eq!(named_keysym(Named::Enter), 0xff0d);
        assert_eq!(named_keysym(Named::Space), 0x20);
    }
}
