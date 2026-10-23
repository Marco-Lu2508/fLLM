//! A small DEFLATE decoder (RFC 1951) with the zlib wrapper (RFC 1950), plus a reader for ZIP archives.
//! PDF streams, DOCX, XLSX and EPUB files all use them. Output is capped so a crafted file cannot exhaust memory.

const LENGTH_BASE: [u16; 29] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LENGTH_EXTRA: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];

struct Huffman {
    count: [u16; 16],
    symbol: Vec<u16>,
}

impl Huffman {
    fn new(lengths: &[u8]) -> Huffman {
        let mut count = [0u16; 16];
        for &l in lengths {
            count[usize::from(l & 15)] += 1;
        }
        let mut offsets = [0u16; 16];
        for i in 1..15 {
            offsets[i + 1] = offsets[i] + count[i];
        }
        let mut symbol = vec![0u16; lengths.len()];
        for (sym, &l) in lengths.iter().enumerate() {
            if l != 0 {
                let slot = &mut offsets[usize::from(l & 15)];
                if let Some(s) = symbol.get_mut(usize::from(*slot)) {
                    *s = sym as u16;
                }
                *slot += 1;
            }
        }
        Huffman { count, symbol }
    }
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
    bits: u32,
    nbits: u32,
}

impl Reader<'_> {
    fn take(&mut self, n: u32) -> Option<u32> {
        while self.nbits < n {
            let byte = *self.data.get(self.pos)?;
            self.pos += 1;
            self.bits |= u32::from(byte) << self.nbits;
            self.nbits += 8;
        }
        let value = self.bits & ((1u32 << n) - 1);
        self.bits >>= n;
        self.nbits -= n;
        Some(value)
    }

    fn decode(&mut self, h: &Huffman) -> Option<u16> {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..16 {
            code |= self.take(1)? as i32;
            let count = i32::from(h.count[len]);
            if code - count < first {
                return h.symbol.get((index + (code - first)) as usize).copied();
            }
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        None
    }
}

fn fixed_tables() -> (Huffman, Huffman) {
    let mut lengths = [0u8; 288];
    for (i, l) in lengths.iter_mut().enumerate() {
        *l = match i {
            0..=143 => 8,
            144..=255 => 9,
            256..=279 => 7,
            _ => 8,
        };
    }
    (Huffman::new(&lengths), Huffman::new(&[5u8; 30]))
}

fn dynamic_tables(r: &mut Reader) -> Option<(Huffman, Huffman)> {
    const ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
    let nlen = r.take(5)? as usize + 257;
    let ndist = r.take(5)? as usize + 1;
    let ncode = r.take(4)? as usize + 4;
    if nlen > 286 || ndist > 30 {
        return None;
    }
    let mut lengths = [0u8; 320];
    for &slot in ORDER.iter().take(ncode) {
        lengths[slot] = r.take(3)? as u8;
    }
    let code_table = Huffman::new(&lengths[..19]);
    let mut lengths = [0u8; 320];
    let mut index = 0;
    while index < nlen + ndist {
        let symbol = r.decode(&code_table)?;
        if symbol < 16 {
            lengths[index] = symbol as u8;
            index += 1;
        } else {
            let (value, repeat) = match symbol {
                16 => {
                    if index == 0 {
                        return None;
                    }
                    (lengths[index - 1], 3 + r.take(2)? as usize)
                }
                17 => (0, 3 + r.take(3)? as usize),
                _ => (0, 11 + r.take(7)? as usize),
            };
            if index + repeat > nlen + ndist {
                return None;
            }
            for _ in 0..repeat {
                lengths[index] = value;
                index += 1;
            }
        }
    }
    Some((Huffman::new(&lengths[..nlen]), Huffman::new(&lengths[nlen..nlen + ndist])))
}

/// Decodes raw DEFLATE data. Returns the bytes produced (at most `limit`) and whether the stream ended cleanly;
/// on damaged data the part decoded so far is still returned.
pub(super) fn inflate_raw(data: &[u8], limit: usize) -> (Vec<u8>, bool) {
    let mut out: Vec<u8> = Vec::new();
    let mut r = Reader { data, pos: 0, bits: 0, nbits: 0 };
    let ok = (|| -> Option<()> {
        loop {
            let last = r.take(1)?;
            match r.take(2)? {
                0 => {
                    r.bits = 0;
                    r.nbits = 0;
                    let head = r.data.get(r.pos..r.pos + 4)?;
                    let len = usize::from(u16::from_le_bytes([head[0], head[1]]));
                    let inv = u16::from_le_bytes([head[2], head[3]]);
                    if inv != !(len as u16) {
                        return None;
                    }
                    let block = r.data.get(r.pos + 4..r.pos + 4 + len)?;
                    r.pos += 4 + len;
                    out.extend_from_slice(block);
                }
                kind @ (1 | 2) => {
                    let (lit, dist) = if kind == 1 { fixed_tables() } else { dynamic_tables(&mut r)? };
                    loop {
                        let sym = r.decode(&lit)?;
                        if sym < 256 {
                            out.push(sym as u8);
                        } else if sym == 256 {
                            break;
                        } else {
                            let i = usize::from(sym - 257);
                            let len = usize::from(*LENGTH_BASE.get(i)?) + r.take(u32::from(*LENGTH_EXTRA.get(i)?))? as usize;
                            let d = usize::from(r.decode(&dist)?);
                            let distance = usize::from(*DIST_BASE.get(d)?) + r.take(u32::from(*DIST_EXTRA.get(d)?))? as usize;
                            if distance > out.len() {
                                return None;
                            }
                            for _ in 0..len {
                                out.push(out[out.len() - distance]);
                            }
                        }
                        if out.len() > limit {
                            return None;
                        }
                    }
                }
                _ => return None,
            }
            if out.len() > limit {
                return None;
            }
            if last == 1 {
                return Some(());
            }
        }
    })()
    .is_some();
    out.truncate(limit);
    (out, ok)
}

/// Decodes a zlib stream (the format of PDF `FlateDecode`); a stream without the header is read as raw DEFLATE.
pub(super) fn inflate_zlib(data: &[u8], limit: usize) -> (Vec<u8>, bool) {
    let zlib = data.len() > 2 && data[0] & 0x0f == 8 && (u32::from(data[0]) << 8 | u32::from(data[1])) % 31 == 0;
    inflate_raw(if zlib { &data[2..] } else { data }, limit)
}

fn le16(b: &[u8], at: usize) -> Option<usize> {
    Some(usize::from(u16::from_le_bytes(b.get(at..at + 2)?.try_into().ok()?)))
}

fn le32(b: &[u8], at: usize) -> Option<usize> {
    Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?) as usize)
}

/// The entries of a ZIP archive, read through the central directory.
pub(super) struct Zip<'a> {
    data: &'a [u8],
    entries: Vec<ZipEntry>,
}

struct ZipEntry {
    name: String,
    method: usize,
    compressed: usize,
    offset: usize,
}

impl<'a> Zip<'a> {
    pub(super) fn open(data: &'a [u8]) -> Result<Zip<'a>, String> {
        let bad = || "the file is not a valid ZIP-based document".to_owned();
        let start = data.len().saturating_sub(65_557);
        let eocd = data[start..].windows(4).rposition(|w| w == b"PK\x05\x06").map(|p| p + start).ok_or_else(bad)?;
        let count = le16(data, eocd + 10).ok_or_else(bad)?;
        let mut at = le32(data, eocd + 16).ok_or_else(bad)?;
        let mut entries = Vec::new();
        for _ in 0..count {
            if data.get(at..at + 4) != Some(b"PK\x01\x02") {
                return Err(bad());
            }
            let method = le16(data, at + 10).ok_or_else(bad)?;
            let compressed = le32(data, at + 20).ok_or_else(bad)?;
            let name_len = le16(data, at + 28).ok_or_else(bad)?;
            let extra_len = le16(data, at + 30).ok_or_else(bad)?;
            let comment_len = le16(data, at + 32).ok_or_else(bad)?;
            let offset = le32(data, at + 42).ok_or_else(bad)?;
            let name = String::from_utf8_lossy(data.get(at + 46..at + 46 + name_len).ok_or_else(bad)?).into_owned();
            entries.push(ZipEntry { name, method, compressed, offset });
            at += 46 + name_len + extra_len + comment_len;
        }
        Ok(Zip { data, entries })
    }

    pub(super) fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|e| e.name.as_str())
    }

    /// The content of one entry (at most `limit` bytes), or `None` when it is missing or unreadable.
    pub(super) fn read(&self, name: &str, limit: usize) -> Option<Vec<u8>> {
        let e = self.entries.iter().find(|e| e.name == name)?;
        if self.data.get(e.offset..e.offset + 4)? != b"PK\x03\x04" {
            return None;
        }
        let start = e.offset + 30 + le16(self.data, e.offset + 26)? + le16(self.data, e.offset + 28)?;
        let raw = self.data.get(start..start.checked_add(e.compressed)?)?;
        match e.method {
            0 => Some(raw.iter().copied().take(limit).collect()),
            8 => {
                let (out, ok) = inflate_raw(raw, limit);
                ok.then_some(out)
            }
            _ => None,
        }
    }
}
