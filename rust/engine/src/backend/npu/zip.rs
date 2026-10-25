//! A small ZIP reader with its own DEFLATE decoder, used to unpack the OpenVINO runtime from the official `openvino` wheel (a ZIP file).
//!
//! The engine has no archive dependency and the setup must not need Python, `unzip` or `bsdtar` on the machine, so the format is read
//! here: the central directory, stored and deflated entries, CRC-32 and size checks. Not supported (and refused with a message): ZIP64,
//! encryption, other compression methods. The decoder follows RFC 1951 (stored, fixed and dynamic Huffman blocks) in the plain
//! bit-by-bit canonical-code way, which is slow but small and easy to check; the runtime is about 200 MB once unpacked.
//!
//! Safety: entry names with `..`, absolute paths, drive letters or backslashes are refused; the declared sizes are limited and the
//! decoder never produces more bytes than declared; every entry's CRC-32 is checked.

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Bounds for one archive.
pub struct Limits {
    /// Most bytes one entry may have unpacked.
    pub max_entry: u64,
    /// Most bytes all kept entries may have unpacked together.
    pub max_total: u64,
    /// Most entries in the archive.
    pub max_entries: usize,
}

struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    buffer: u64,
    count: u32,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8]) -> Self {
        Bits { data, pos: 0, buffer: 0, count: 0 }
    }

    fn take(&mut self, n: u32) -> Result<u32, String> {
        while self.count < n {
            let byte = *self.data.get(self.pos).ok_or("compressed data ends too early")?;
            self.pos += 1;
            self.buffer |= u64::from(byte) << self.count;
            self.count += 8;
        }
        let value = (self.buffer & ((1u64 << n) - 1)) as u32;
        self.buffer >>= n;
        self.count -= n;
        Ok(value)
    }

    /// Drops the bits left in the current byte.
    fn align(&mut self) {
        self.buffer = 0;
        self.count = 0;
    }
}

/// A canonical Huffman code: how many codes of each length, and the symbols in code order.
struct Huffman {
    count: [u16; 16],
    symbol: Vec<u16>,
}

impl Huffman {
    fn new(lengths: &[u8]) -> Result<Huffman, String> {
        let mut count = [0u16; 16];
        for &len in lengths {
            count[usize::from(len)] += 1;
        }
        let mut left: i32 = 1;
        for &codes_of_length in &count[1..16] {
            left <<= 1;
            left -= i32::from(codes_of_length);
            if left < 0 {
                return Err("over-subscribed Huffman code".into());
            }
        }
        let mut offsets = [0u16; 16];
        for len in 1..15 {
            offsets[len + 1] = offsets[len] + count[len];
        }
        let mut symbol = vec![0u16; lengths.len()];
        for (sym, &len) in lengths.iter().enumerate() {
            if len != 0 {
                symbol[usize::from(offsets[usize::from(len)])] = sym as u16;
                offsets[usize::from(len)] += 1;
            }
        }
        Ok(Huffman { count, symbol })
    }

    fn decode(&self, bits: &mut Bits) -> Result<u16, String> {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..16 {
            code |= bits.take(1)? as i32;
            let count = i32::from(self.count[len]);
            if code - count < first {
                return self.symbol.get((index + (code - first)) as usize).copied().ok_or_else(|| "bad Huffman symbol".to_owned());
            }
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        Err("invalid Huffman code".into())
    }
}

const LENGTH_BASE: [u16; 29] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LENGTH_EXTRA: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];

fn codes(bits: &mut Bits, out: &mut Vec<u8>, limit: usize, lit: &Huffman, dist: &Huffman) -> Result<(), String> {
    loop {
        let sym = usize::from(lit.decode(bits)?);
        if sym < 256 {
            if out.len() >= limit {
                return Err("unpacked data is larger than declared".into());
            }
            out.push(sym as u8);
        } else if sym == 256 {
            return Ok(());
        } else {
            let sym = sym - 257;
            if sym >= 29 {
                return Err("invalid length code".into());
            }
            let length = usize::from(LENGTH_BASE[sym]) + bits.take(u32::from(LENGTH_EXTRA[sym]))? as usize;
            let d = usize::from(dist.decode(bits)?);
            if d >= 30 {
                return Err("invalid distance code".into());
            }
            let distance = usize::from(DIST_BASE[d]) + bits.take(u32::from(DIST_EXTRA[d]))? as usize;
            if distance > out.len() {
                return Err("distance reaches before the start of the data".into());
            }
            if out.len() + length > limit {
                return Err("unpacked data is larger than declared".into());
            }
            let start = out.len() - distance;
            for i in 0..length {
                let byte = out[start + i];
                out.push(byte);
            }
        }
    }
}

/// Unpacks raw DEFLATE data; fails when the result would be longer than `limit` bytes.
pub fn inflate(data: &[u8], limit: usize) -> Result<Vec<u8>, String> {
    let mut bits = Bits::new(data);
    let mut out: Vec<u8> = Vec::with_capacity(limit.min(1 << 26));
    loop {
        let last = bits.take(1)?;
        match bits.take(2)? {
            0 => {
                bits.align();
                let header = data.get(bits.pos..bits.pos + 4).ok_or("stored block ends too early")?;
                let len = usize::from(u16::from_le_bytes([header[0], header[1]]));
                let inverse = u16::from_le_bytes([header[2], header[3]]);
                if inverse != !(len as u16) {
                    return Err("stored block length check failed".into());
                }
                let start = bits.pos + 4;
                let block = data.get(start..start + len).ok_or("stored block ends too early")?;
                if out.len() + len > limit {
                    return Err("unpacked data is larger than declared".into());
                }
                out.extend_from_slice(block);
                bits.pos = start + len;
            }
            1 => {
                let mut lengths = [0u8; 288];
                lengths[..144].fill(8);
                lengths[144..256].fill(9);
                lengths[256..280].fill(7);
                lengths[280..].fill(8);
                codes(&mut bits, &mut out, limit, &Huffman::new(&lengths)?, &Huffman::new(&[5u8; 30])?)?;
            }
            2 => {
                let nlen = bits.take(5)? as usize + 257;
                let ndist = bits.take(5)? as usize + 1;
                let ncode = bits.take(4)? as usize + 4;
                if nlen > 286 || ndist > 30 {
                    return Err("too many length or distance codes".into());
                }
                const ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
                let mut lengths = [0u8; 320];
                for &slot in ORDER.iter().take(ncode) {
                    lengths[slot] = bits.take(3)? as u8;
                }
                let code_code = Huffman::new(&lengths[..19])?;
                let mut lengths = [0u8; 320];
                let mut index = 0;
                while index < nlen + ndist {
                    let sym = code_code.decode(&mut bits)?;
                    if sym < 16 {
                        lengths[index] = sym as u8;
                        index += 1;
                    } else {
                        let (value, repeat) = match sym {
                            16 => {
                                if index == 0 {
                                    return Err("repeat with no previous length".into());
                                }
                                (lengths[index - 1], 3 + bits.take(2)? as usize)
                            }
                            17 => (0, 3 + bits.take(3)? as usize),
                            _ => (0, 11 + bits.take(7)? as usize),
                        };
                        if index + repeat > nlen + ndist {
                            return Err("too many code lengths".into());
                        }
                        lengths[index..index + repeat].fill(value);
                        index += repeat;
                    }
                }
                if lengths[256] == 0 {
                    return Err("no end-of-block code".into());
                }
                let lit = Huffman::new(&lengths[..nlen])?;
                let dist = Huffman::new(&lengths[nlen..nlen + ndist])?;
                codes(&mut bits, &mut out, limit, &lit, &dist)?;
            }
            _ => return Err("invalid block type".into()),
        }
        if last == 1 {
            return Ok(out);
        }
    }
}

fn crc32_table() -> &'static [u32; 256] {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [0u32; 256];
        for (i, slot) in table.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 == 1 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            }
            *slot = c;
        }
        table
    })
}

pub fn crc32(data: &[u8]) -> u32 {
    let table = crc32_table();
    !data.iter().fold(!0u32, |c, &b| table[((c ^ u32::from(b)) & 0xFF) as usize] ^ (c >> 8))
}

/// One file of the archive.
#[derive(Clone, Debug)]
pub struct Entry {
    pub name: String,
    method: u16,
    crc: u32,
    compressed: u64,
    size: u64,
    header_offset: u64,
}

fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}
fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// The entries of the archive, read from its central directory.
pub fn list<R: Read + Seek>(file: &mut R, limits: &Limits) -> Result<Vec<Entry>, String> {
    let length = file.seek(SeekFrom::End(0)).map_err(|e| e.to_string())?;
    let tail_len = length.min(22 + 65535);
    file.seek(SeekFrom::Start(length - tail_len)).map_err(|e| e.to_string())?;
    let mut tail = vec![0u8; tail_len as usize];
    file.read_exact(&mut tail).map_err(|e| e.to_string())?;
    let eocd = (0..tail.len().saturating_sub(21))
        .rev()
        .find(|&i| tail[i..i + 4] == [0x50, 0x4B, 0x05, 0x06])
        .ok_or("not a ZIP file (no end-of-directory record)")?;
    let record = &tail[eocd..];
    let count = u16_at(record, 10) as usize;
    let directory_size = u64::from(u32_at(record, 12));
    let directory_offset = u64::from(u32_at(record, 16));
    if count == 0xFFFF || directory_size == 0xFFFF_FFFF || directory_offset == 0xFFFF_FFFF {
        return Err("ZIP64 archives are not supported".into());
    }
    if count > limits.max_entries || directory_size > (64 << 20) {
        return Err("the archive has too many entries".into());
    }
    file.seek(SeekFrom::Start(directory_offset)).map_err(|e| e.to_string())?;
    let mut directory = vec![0u8; directory_size as usize];
    file.read_exact(&mut directory).map_err(|e| e.to_string())?;
    let mut entries = Vec::with_capacity(count);
    let mut at = 0usize;
    for _ in 0..count {
        let head = directory.get(at..at + 46).ok_or("central directory is cut off")?;
        if head[..4] != [0x50, 0x4B, 0x01, 0x02] {
            return Err("central directory is damaged".into());
        }
        let flags = u16_at(head, 8);
        let name_len = usize::from(u16_at(head, 28));
        let extra_len = usize::from(u16_at(head, 30));
        let comment_len = usize::from(u16_at(head, 32));
        let name = directory.get(at + 46..at + 46 + name_len).ok_or("central directory is cut off")?;
        if flags & 1 != 0 {
            return Err("encrypted entries are not supported".into());
        }
        entries.push(Entry {
            name: String::from_utf8_lossy(name).into_owned(),
            method: u16_at(head, 10),
            crc: u32_at(head, 16),
            compressed: u64::from(u32_at(head, 20)),
            size: u64::from(u32_at(head, 24)),
            header_offset: u64::from(u32_at(head, 42)),
        });
        at += 46 + name_len + extra_len + comment_len;
    }
    Ok(entries)
}

/// The relative path an entry name stands for, `None` when the name could leave the target directory.
pub fn safe_path(name: &str) -> Option<PathBuf> {
    if name.is_empty() || name.contains('\\') || name.contains(':') || name.starts_with('/') || name.contains('\0') {
        return None;
    }
    let mut out = PathBuf::new();
    for part in name.split('/') {
        match part {
            "" | "." => continue,
            ".." => return None,
            other => out.push(other),
        }
    }
    if out.as_os_str().is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Reads and checks one entry.
pub fn read_entry<R: Read + Seek>(file: &mut R, entry: &Entry, limits: &Limits) -> Result<Vec<u8>, String> {
    if entry.size > limits.max_entry || entry.compressed > limits.max_entry {
        return Err(format!("{}: larger than the limit", entry.name));
    }
    file.seek(SeekFrom::Start(entry.header_offset)).map_err(|e| e.to_string())?;
    let mut head = [0u8; 30];
    file.read_exact(&mut head).map_err(|e| e.to_string())?;
    if head[..4] != [0x50, 0x4B, 0x03, 0x04] {
        return Err(format!("{}: local header is damaged", entry.name));
    }
    let skip = i64::from(u16_at(&head, 26)) + i64::from(u16_at(&head, 28));
    file.seek(SeekFrom::Current(skip)).map_err(|e| e.to_string())?;
    let mut packed = vec![0u8; entry.compressed as usize];
    file.read_exact(&mut packed).map_err(|e| format!("{}: {e}", entry.name))?;
    let data = match entry.method {
        0 => packed,
        8 => inflate(&packed, entry.size as usize).map_err(|e| format!("{}: {e}", entry.name))?,
        other => return Err(format!("{}: compression method {other} is not supported", entry.name)),
    };
    if data.len() as u64 != entry.size {
        return Err(format!("{}: size does not match the directory", entry.name));
    }
    if crc32(&data) != entry.crc {
        return Err(format!("{}: CRC-32 check failed", entry.name));
    }
    Ok(data)
}

/// Unpacks the entries for which `keep(name)` is true below `destination`. `step(done, total)` is called after each file with the
/// unpacked bytes so far; a `true` from `cancelled()` ends the work. Returns the names written.
pub fn extract(
    archive: &Path,
    destination: &Path,
    keep: &dyn Fn(&str) -> bool,
    limits: &Limits,
    cancelled: &dyn Fn() -> bool,
    step: &mut dyn FnMut(u64, u64),
) -> Result<Vec<String>, String> {
    let mut file = std::fs::File::open(archive).map_err(|e| format!("cannot open {}: {e}", archive.display()))?;
    let entries = list(&mut file, limits)?;
    let wanted: Vec<&Entry> = entries.iter().filter(|e| !e.name.ends_with('/') && keep(&e.name)).collect();
    let total: u64 = wanted.iter().map(|e| e.size).sum();
    if total > limits.max_total {
        return Err("the archive is larger than the limit when unpacked".into());
    }
    let mut done = 0u64;
    let mut written = Vec::new();
    for entry in wanted {
        if cancelled() {
            return Err("cancelled".into());
        }
        let relative = safe_path(&entry.name).ok_or_else(|| format!("{}: unsafe file name in the archive", entry.name))?;
        let data = read_entry(&mut file, entry, limits)?;
        let target = destination.join(relative);
        if let Some(dir) = target.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        }
        let mut out = std::fs::File::create(&target).map_err(|e| format!("cannot write {}: {e}", target.display()))?;
        out.write_all(&data).map_err(|e| format!("cannot write {}: {e}", target.display()))?;
        done += entry.size;
        written.push(entry.name.clone());
        step(done, total);
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Made with Python's zipfile: one deflated entry (dynamic Huffman), one stored entry outside `openvino/libs/` and one stored inside.
    const SAMPLE: [u8; 524] = [
        0x50, 0x4b, 0x03, 0x04, 0x14, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x21, 0x00, 0xb3, 0x99, 0x5f, 0xfa, 0x8c, 0x00, 0x00, 0x00, 0x75, 0x01,
        0x00, 0x00, 0x13, 0x00, 0x00, 0x00, 0x6f, 0x70, 0x65, 0x6e, 0x76, 0x69, 0x6e, 0x6f, 0x2f, 0x6c, 0x69, 0x62, 0x73, 0x2f, 0x61, 0x2e, 0x74, 0x78,
        0x74, 0xd5, 0x8d, 0xb9, 0x11, 0x83, 0x30, 0x14, 0x44, 0x73, 0x57, 0xb1, 0x05, 0x78, 0x34, 0x36, 0xc3, 0x38, 0x71, 0x13, 0x0e, 0x68, 0x00, 0xc1,
        0x07, 0x09, 0x84, 0x3e, 0xe8, 0x04, 0xaa, 0xb7, 0x52, 0x02, 0x17, 0xe0, 0x64, 0x37, 0x78, 0x7b, 0x34, 0x8a, 0xb0, 0x45, 0xdd, 0xcd, 0x90, 0x8e,
        0xb3, 0xc5, 0xc0, 0x3b, 0xa6, 0xb8, 0xac, 0x1e, 0x9c, 0xc8, 0x21, 0x14, 0x6c, 0xda, 0xf3, 0x40, 0xcf, 0xa3, 0xc0, 0xa7, 0x2d, 0xb9, 0xe5, 0x80,
        0x2c, 0xa1, 0xac, 0x83, 0xc2, 0xa0, 0x13, 0x15, 0x74, 0x92, 0x85, 0xd1, 0x5b, 0x64, 0x57, 0xba, 0xa3, 0x17, 0x68, 0xfe, 0x68, 0xb5, 0x95, 0x5d,
        0x4f, 0xc3, 0xa8, 0xf4, 0x34, 0x9b, 0xc5, 0xf2, 0xba, 0x39, 0x1f, 0x62, 0xca, 0xfb, 0x71, 0xfe, 0x26, 0x50, 0x64, 0x0c, 0x5f, 0x34, 0xb3, 0x33,
        0xfd, 0xfd, 0x62, 0x6f, 0xf0, 0x4a, 0x36, 0x69, 0xcb, 0xe5, 0x51, 0x7a, 0x54, 0x8f, 0xea, 0x25, 0x6a, 0xf1, 0xbc, 0x7d, 0x01, 0x50, 0x4b, 0x03,
        0x04, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x21, 0x00, 0x67, 0xba, 0x8e, 0xeb, 0x03, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x11,
        0x00, 0x00, 0x00, 0x6f, 0x74, 0x68, 0x65, 0x72, 0x2f, 0x73, 0x6b, 0x69, 0x70, 0x70, 0x65, 0x64, 0x2e, 0x62, 0x69, 0x6e, 0x78, 0x79, 0x7a, 0x50,
        0x4b, 0x03, 0x04, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x21, 0x00, 0x11, 0x55, 0xd7, 0x99, 0x0b, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00,
        0x00, 0x18, 0x00, 0x00, 0x00, 0x6f, 0x70, 0x65, 0x6e, 0x76, 0x69, 0x6e, 0x6f, 0x2f, 0x6c, 0x69, 0x62, 0x73, 0x2f, 0x73, 0x74, 0x6f, 0x72, 0x65,
        0x64, 0x2e, 0x74, 0x78, 0x74, 0x73, 0x74, 0x6f, 0x72, 0x65, 0x64, 0x20, 0x64, 0x61, 0x74, 0x61, 0x50, 0x4b, 0x01, 0x02, 0x14, 0x00, 0x14, 0x00,
        0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x21, 0x00, 0xb3, 0x99, 0x5f, 0xfa, 0x8c, 0x00, 0x00, 0x00, 0x75, 0x01, 0x00, 0x00, 0x13, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x01, 0x00, 0x00, 0x00, 0x00, 0x6f, 0x70, 0x65, 0x6e, 0x76, 0x69, 0x6e, 0x6f, 0x2f, 0x6c,
        0x69, 0x62, 0x73, 0x2f, 0x61, 0x2e, 0x74, 0x78, 0x74, 0x50, 0x4b, 0x01, 0x02, 0x14, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x21,
        0x00, 0x67, 0xba, 0x8e, 0xeb, 0x03, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x11, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x80, 0x01, 0xbd, 0x00, 0x00, 0x00, 0x6f, 0x74, 0x68, 0x65, 0x72, 0x2f, 0x73, 0x6b, 0x69, 0x70, 0x70, 0x65, 0x64, 0x2e, 0x62, 0x69, 0x6e,
        0x50, 0x4b, 0x01, 0x02, 0x14, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x21, 0x00, 0x11, 0x55, 0xd7, 0x99, 0x0b, 0x00, 0x00, 0x00,
        0x0b, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x01, 0xef, 0x00, 0x00, 0x00, 0x6f, 0x70,
        0x65, 0x6e, 0x76, 0x69, 0x6e, 0x6f, 0x2f, 0x6c, 0x69, 0x62, 0x73, 0x2f, 0x73, 0x74, 0x6f, 0x72, 0x65, 0x64, 0x2e, 0x74, 0x78, 0x74, 0x50, 0x4b,
        0x05, 0x06, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x03, 0x00, 0xc6, 0x00, 0x00, 0x00, 0x30, 0x01, 0x00, 0x00, 0x00, 0x00,
    ];

    const LIMITS: Limits = Limits { max_entry: 1 << 20, max_total: 1 << 22, max_entries: 100 };

    #[test]
    fn crc_matches_the_known_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn lists_and_reads_deflated_and_stored_entries() {
        let mut file = std::io::Cursor::new(SAMPLE.to_vec());
        let entries = list(&mut file, &LIMITS).unwrap();
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["openvino/libs/a.txt", "other/skipped.bin", "openvino/libs/stored.txt"]);
        let text = read_entry(&mut file, &entries[0], &LIMITS).unwrap();
        assert_eq!(text.len(), 373);
        assert!(text.starts_with(b"The quick brown fox"));
        assert!(text.ends_with(b"openvino libs 2026.4.1\n"));
        assert_eq!(read_entry(&mut file, &entries[2], &LIMITS).unwrap(), b"stored data");
    }

    #[test]
    fn a_damaged_archive_is_refused() {
        let mut broken = SAMPLE.to_vec();
        broken[60] ^= 0x55; // inside the deflated data
        let mut file = std::io::Cursor::new(broken);
        let entries = list(&mut file, &LIMITS).unwrap();
        assert!(read_entry(&mut file, &entries[0], &LIMITS).is_err());
        let mut short = std::io::Cursor::new(SAMPLE[..100].to_vec());
        assert!(list(&mut short, &LIMITS).is_err());
        // declared sizes above the limit
        let mut file = std::io::Cursor::new(SAMPLE.to_vec());
        let tiny = Limits { max_entry: 100, ..LIMITS };
        let entries = list(&mut file, &tiny).unwrap();
        assert!(read_entry(&mut file, &entries[0], &tiny).is_err());
    }

    #[test]
    fn unsafe_names_are_refused() {
        assert_eq!(safe_path("openvino/libs/a.so"), Some(PathBuf::from("openvino").join("libs").join("a.so")));
        for bad in ["../x", "a/../../x", "/etc/passwd", "C:/x", "a\\b", "", "./"] {
            assert!(safe_path(bad).is_none(), "{bad}");
        }
    }

    #[test]
    fn extract_keeps_only_the_wanted_entries() {
        let base = std::env::temp_dir().join(format!("fllm-zip-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let archive = base.join("a.zip");
        std::fs::write(&archive, SAMPLE).unwrap();
        let out = base.join("out");
        let mut last = (0, 0);
        let written = extract(&archive, &out, &|n| n.starts_with("openvino/libs/"), &LIMITS, &|| false, &mut |d, t| last = (d, t)).unwrap();
        assert_eq!(written.len(), 2);
        assert_eq!(last, (373 + 11, 373 + 11));
        assert_eq!(std::fs::read(out.join("openvino").join("libs").join("stored.txt")).unwrap(), b"stored data");
        assert!(!out.join("other").exists());
        let cancelled = extract(&archive, &base.join("o2"), &|_| true, &LIMITS, &|| true, &mut |_, _| {});
        assert_eq!(cancelled.unwrap_err(), "cancelled");
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn inflate_handles_fixed_blocks_and_rejects_garbage() {
        // "aaaaaaaaaa" as a fixed-Huffman block (zlib.compressobj(wbits=-15) output)
        let packed = [0x4b, 0x4c, 0x84, 0x01, 0x00];
        assert_eq!(inflate(&packed, 100).unwrap(), b"aaaaaaaaaa");
        assert!(inflate(&packed, 5).is_err(), "output beyond the declared size");
        assert!(inflate(&[0x07], 10).is_err(), "reserved block type");
        assert!(inflate(&[], 10).is_err());
    }
}
