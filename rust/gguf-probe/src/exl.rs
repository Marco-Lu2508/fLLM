//! Native in-memory / on-disk layouts of ExLlamaV2 (EXL2) and ExLlamaV3 (EXL3) weight matrices.
//!
//! The two fLLM-specific GGUF tensor types `EXL2` (0x8000_0030) and `EXL3` (0x8000_0031) keep the quantized weights of
//! an EXL checkpoint packed: nothing is expanded to 16 bit when the model is loaded. A tensor is one self-describing
//! little-endian blob (so its length cannot be derived from the GGUF dimensions alone: the blob header carries it):
//!
//! ```text
//! header (64 bytes)
//!    0  magic "EXL2" / "EXL3"      4  version = 1           8  rows (output features)   12  cols (input features)
//!   16  total blob length (u64)   24  data_off (first row / tile column)                28  row_bytes
//!   32  groups (EXL2)             36  param (EXL3: bits | codebook << 8)
//!   40  off_a    44  off_b    48  off_c        (section offsets, 16-byte aligned; see below)   52..64 zero
//! ```
//!
//! # EXL2
//!
//! * `off_a`: `groups` table entries of 16 bytes `{ n: u32, bits: u16, max: f16, woff: u32, start: u32 }`. Group `g` holds
//!   `n` consecutive inputs *in stored order* starting at stored input `start`; each input is a `bits`-wide code
//!   (2..=6 or 8) and `max` is the group's `q_scale_max`. `woff` is the 32-bit word offset of the group's bitstream inside a row.
//! * `off_b`: `perm`, `cols` x u32: stored input `k` multiplies original activation `x[perm[k]]` (`q_perm`; the inverse of the
//!   checkpoint's `q_invperm`).
//! * Row `r` (output feature) is at `data_off + r * row_bytes`: the bitstreams of all groups in order (every group starts on a
//!   32-bit word, the codes are LSB-first and may straddle words), followed by the 4-bit scale index of every group
//!   (eight per word, group `g` in word `g / 8` at bit `4 * (g % 8)`). The weight of stored input `i` of group `g` is
//!   `(code - 2^(bits-1)) * (s + 1)^2 * (max / 256)`, `s` the scale index.
//!
//! # EXL3
//!
//! * `off_a`: `suh`, `cols` x f16 (input sign / scale vector); `off_b`: `svh`, `rows` x f16; `off_c`: `out_map`, `rows` x u32, the
//!   output row each result channel is written to (a permutation: the converter's row reordering of attention weights).
//! * Trellis tiles from `data_off`: tile column `no` (16 output channels) is `row_bytes = cols / 16 * 32 * K` bytes, the tiles of
//!   its `cols / 16` input tiles `ki` in order, `32 * K` bytes each (the checkpoint's `trellis[ki][no]` words, 16 x 16 values as
//!   a tail-biting ring of 256 K-bit symbols, see docs/exl-formats.md). `param & 0xff` is `K` (1..=8), `param >> 8` the codebook
//!   (0 default, 1 mcg, 2 mul1).
//! * `y = diag(svh) * H * (R^T * (H * diag(suh) * x))` with `H` the normalized 128x128 Hadamard transform applied to every
//!   128-block and `R` the trellis matrix.
//!
//! Layout provenance and the MIT notice of the upstream projects: docs/exl-formats.md.

use half::f16;
use std::sync::OnceLock;

pub const EXL2_TYPE: u32 = 0x8000_0030;
pub const EXL3_TYPE: u32 = 0x8000_0031;
pub const HEADER_LEN: usize = 64;
pub const GROUP_LEN: usize = 16;
const VERSION: u32 = 1;
const MAGIC2: [u8; 4] = *b"EXL2";
const MAGIC3: [u8; 4] = *b"EXL3";

/// Whether `type_id` is one of the native EXL GGUF tensor types.
pub fn is_exl_type(type_id: u32) -> bool {
    type_id == EXL2_TYPE || type_id == EXL3_TYPE
}

fn align16(n: usize) -> usize {
    (n + 15) & !15
}

fn u32_at(b: &[u8], at: usize) -> Result<u32, String> {
    b.get(at..at + 4)
        .map(|s| u32::from_le_bytes(s.try_into().unwrap()))
        .ok_or_else(|| "EXL tensor truncated".to_owned())
}

fn word(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]])
}

/// Blob length stored in the header (the tensors of a GGUF file are validated against it).
pub fn payload_len(head: &[u8], type_id: u32) -> Result<u64, String> {
    let magic = if type_id == EXL2_TYPE { MAGIC2 } else { MAGIC3 };
    if head.len() < HEADER_LEN || head[..4] != magic {
        return Err("EXL tensor header missing or of the wrong type".into());
    }
    Ok(u64::from_le_bytes(head[16..24].try_into().unwrap()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub exl3: bool,
    pub rows: usize,
    pub cols: usize,
    pub total: usize,
    pub data_off: usize,
    pub row_bytes: usize,
    pub groups: usize,
    pub param: u32,
    pub off_a: usize,
    pub off_b: usize,
    pub off_c: usize,
}

impl Header {
    pub fn write(&self) -> [u8; HEADER_LEN] {
        let mut h = [0u8; HEADER_LEN];
        h[..4].copy_from_slice(if self.exl3 { &MAGIC3 } else { &MAGIC2 });
        let mut put = |at: usize, v: u32| h[at..at + 4].copy_from_slice(&v.to_le_bytes());
        put(4, VERSION);
        put(8, self.rows as u32);
        put(12, self.cols as u32);
        put(24, self.data_off as u32);
        put(28, self.row_bytes as u32);
        put(32, self.groups as u32);
        put(36, self.param);
        put(40, self.off_a as u32);
        put(44, self.off_b as u32);
        put(48, self.off_c as u32);
        h[16..24].copy_from_slice(&(self.total as u64).to_le_bytes());
        h
    }

    /// Parse and bounds-check the header of a whole blob.
    pub fn parse(blob: &[u8]) -> Result<Self, String> {
        if blob.len() < HEADER_LEN {
            return Err("EXL tensor shorter than its header".into());
        }
        let exl3 = match &blob[..4] {
            m if m == MAGIC2 => false,
            m if m == MAGIC3 => true,
            _ => return Err("EXL tensor magic mismatch".into()),
        };
        if u32_at(blob, 4)? != VERSION {
            return Err("unsupported EXL tensor version".into());
        }
        let field = |at: usize| u32_at(blob, at).map(|v| v as usize);
        let total = u64::from_le_bytes(blob[16..24].try_into().unwrap());
        let h = Self {
            exl3,
            rows: field(8)?,
            cols: field(12)?,
            total: usize::try_from(total).map_err(|_| "EXL tensor length overflow".to_owned())?,
            data_off: field(24)?,
            row_bytes: field(28)?,
            groups: field(32)?,
            param: u32_at(blob, 36)?,
            off_a: field(40)?,
            off_b: field(44)?,
            off_c: field(48)?,
        };
        if h.rows == 0 || h.cols == 0 || h.total != blob.len() {
            return Err("EXL tensor header inconsistent with its size".into());
        }
        let units = if exl3 { h.rows / 16 } else { h.rows };
        let data_end = units
            .checked_mul(h.row_bytes)
            .and_then(|n| n.checked_add(h.data_off))
            .ok_or("EXL tensor size overflow")?;
        if data_end != h.total || h.data_off % 16 != 0 || h.data_off < HEADER_LEN {
            return Err("EXL tensor data section inconsistent".into());
        }
        Ok(h)
    }
}

// ---------------------------------------------------------------------------------------------------------------- EXL2

/// One group of the EXL2 table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Exl2Group {
    /// Inputs in the group.
    pub n: usize,
    pub bits: usize,
    /// `q_scale_max` as stored (f16 bits).
    pub max_bits: u16,
    /// Word offset of the group's bitstream inside a row.
    pub woff: usize,
    /// First stored input of the group.
    pub start: usize,
}

impl Exl2Group {
    /// 32-bit words of the group's bitstream.
    pub fn words(&self) -> usize {
        (self.n * self.bits).div_ceil(32)
    }
    /// `q_scale_max / 256`: multiplied with `(s + 1)^2` it is the scale of the group.
    pub fn max_scaled(&self) -> f32 {
        f16::from_bits(self.max_bits).to_f32() / 256.0
    }
    /// The offset subtracted from every code.
    pub fn zero(&self) -> i32 {
        1 << (self.bits - 1)
    }
}

pub fn exl2_width_ok(bits: usize) -> bool {
    matches!(bits, 2..=6 | 8)
}

/// Geometry of an EXL2 blob, shared by the writer and the reader.
#[derive(Clone, Debug)]
pub struct Exl2Layout {
    pub header: Header,
    /// `(n, bits)` per group in stored order.
    pub groups: Vec<(usize, usize)>,
}

impl Exl2Layout {
    pub fn new(rows: usize, cols: usize, groups: &[(usize, usize)]) -> Result<Self, String> {
        if rows == 0 || cols == 0 || groups.is_empty() {
            return Err("EXL2 dimensions must be nonzero".into());
        }
        if groups.iter().map(|g| g.0).sum::<usize>() != cols || groups.iter().any(|&(n, b)| n == 0 || !exl2_width_ok(b)) {
            return Err("EXL2 groups do not tile the input dimension".into());
        }
        let words: usize = groups.iter().map(|&(n, b)| (n * b).div_ceil(32)).sum();
        let row_bytes = 4 * (words + groups.len().div_ceil(8));
        let off_a = HEADER_LEN;
        let off_b = align16(off_a + GROUP_LEN * groups.len());
        let data_off = align16(off_b + 4 * cols);
        let total = rows
            .checked_mul(row_bytes)
            .and_then(|n| n.checked_add(data_off))
            .filter(|&n| n < u32::MAX as usize && rows < u32::MAX as usize && cols < u32::MAX as usize)
            .ok_or("EXL2 tensor exceeds the 4 GiB blob limit")?;
        Ok(Self {
            header: Header { exl3: false, rows, cols, total, data_off, row_bytes, groups: groups.len(), param: 0, off_a, off_b, off_c: 0 },
            groups: groups.to_vec(),
        })
    }

    /// Everything in front of the row data: header, group table (with `maxima`, f16 bits per group) and `perm`.
    pub fn prefix(&self, maxima: &[u16], perm: &[u32]) -> Vec<u8> {
        assert_eq!(maxima.len(), self.groups.len());
        assert_eq!(perm.len(), self.header.cols);
        let h = &self.header;
        let mut out = vec![0u8; h.data_off];
        out[..HEADER_LEN].copy_from_slice(&h.write());
        let (mut woff, mut start) = (0u32, 0u32);
        for (g, &(n, bits)) in self.groups.iter().enumerate() {
            let at = h.off_a + g * GROUP_LEN;
            out[at..at + 4].copy_from_slice(&(n as u32).to_le_bytes());
            out[at + 4..at + 6].copy_from_slice(&(bits as u16).to_le_bytes());
            out[at + 6..at + 8].copy_from_slice(&maxima[g].to_le_bytes());
            out[at + 8..at + 12].copy_from_slice(&woff.to_le_bytes());
            out[at + 12..at + 16].copy_from_slice(&start.to_le_bytes());
            woff += (n * bits).div_ceil(32) as u32;
            start += n as u32;
        }
        for (k, &p) in perm.iter().enumerate() {
            out[h.off_b + 4 * k..h.off_b + 4 * k + 4].copy_from_slice(&p.to_le_bytes());
        }
        out
    }
}

/// Append `codes` (each `< 2^bits`) as an LSB-first bitstream padded to a 32-bit word.
pub fn pack_codes(codes: &[u8], bits: usize, out: &mut Vec<u8>) {
    let mut buf = 0u64;
    let mut have = 0usize;
    for &c in codes {
        buf |= u64::from(c) << have;
        have += bits;
        if have >= 32 {
            out.extend_from_slice(&(buf as u32).to_le_bytes());
            buf >>= 32;
            have -= 32;
        }
    }
    if have > 0 {
        out.extend_from_slice(&(buf as u32).to_le_bytes());
    }
}

/// Unpack `out.len()` codes of `bits` width from an LSB-first bitstream and store `code - 2^(bits-1)`.
pub fn unpack_centered(words: &[u8], bits: usize, out: &mut [f32]) {
    let mask = (1u64 << bits) - 1;
    let zero = (1i32 << (bits - 1)) as f32;
    let (mut buf, mut have, mut next) = (0u64, 0usize, 0usize);
    for slot in out.iter_mut() {
        if have < bits {
            buf |= u64::from(word(words, next)) << have;
            have += 32;
            next += 1;
        }
        *slot = (buf & mask) as f32 - zero;
        buf >>= bits;
        have -= bits;
    }
}

/// Reader over a validated EXL2 blob.
#[derive(Clone, Copy, Debug)]
pub struct Exl2View<'a> {
    pub header: Header,
    blob: &'a [u8],
}

impl<'a> Exl2View<'a> {
    /// Structural validation, linear in the number of groups.
    pub fn parse(blob: &'a [u8]) -> Result<Self, String> {
        let header = Header::parse(blob)?;
        if header.exl3 || header.groups == 0 {
            return Err("not an EXL2 tensor".into());
        }
        let table_end = header.off_a.checked_add(GROUP_LEN * header.groups).ok_or("EXL2 table overflow")?;
        let perm_end = header.off_b.checked_add(4 * header.cols).ok_or("EXL2 perm overflow")?;
        if header.off_a < HEADER_LEN || table_end > header.off_b || perm_end > header.data_off {
            return Err("EXL2 sections overlap".into());
        }
        let view = Self { header, blob };
        let (mut woff, mut start) = (0usize, 0usize);
        for g in 0..header.groups {
            let e = view.group(g);
            if e.n == 0 || !exl2_width_ok(e.bits) || e.woff != woff || e.start != start {
                return Err(format!("EXL2 group {g} is inconsistent"));
            }
            let m = f16::from_bits(e.max_bits).to_f32();
            if !m.is_finite() || m < 0.0 {
                return Err(format!("EXL2 group {g} has an invalid scale"));
            }
            woff += e.words();
            start += e.n;
        }
        if start != header.cols || header.row_bytes != 4 * (woff + header.groups.div_ceil(8)) {
            return Err("EXL2 groups do not match the row size".into());
        }
        Ok(view)
    }

    /// Full validation: structure plus `perm` being a permutation (linear in the number of inputs; do it once at load).
    pub fn validate(blob: &'a [u8]) -> Result<Self, String> {
        let v = Self::parse(blob)?;
        let mut seen = vec![false; v.header.cols];
        for k in 0..v.header.cols {
            let p = v.perm(k);
            if p >= v.header.cols || std::mem::replace(&mut seen[p], true) {
                return Err(format!("EXL2 permutation invalid at {k}"));
            }
        }
        Ok(v)
    }

    pub fn rows(&self) -> usize {
        self.header.rows
    }
    pub fn cols(&self) -> usize {
        self.header.cols
    }
    pub fn groups(&self) -> usize {
        self.header.groups
    }

    pub fn group(&self, g: usize) -> Exl2Group {
        let at = self.header.off_a + g * GROUP_LEN;
        let b = &self.blob[at..at + GROUP_LEN];
        Exl2Group {
            n: u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize,
            bits: u16::from_le_bytes(b[4..6].try_into().unwrap()) as usize,
            max_bits: u16::from_le_bytes(b[6..8].try_into().unwrap()),
            woff: u32::from_le_bytes(b[8..12].try_into().unwrap()) as usize,
            start: u32::from_le_bytes(b[12..16].try_into().unwrap()) as usize,
        }
    }

    /// Original input that stored input `k` multiplies.
    pub fn perm(&self, k: usize) -> usize {
        word(&self.blob[self.header.off_b..], k) as usize
    }

    pub fn row(&self, r: usize) -> &'a [u8] {
        let at = self.header.data_off + r * self.header.row_bytes;
        &self.blob[at..at + self.header.row_bytes]
    }

    /// The bitstream of group `e` inside `row`.
    pub fn group_words(&self, row: &'a [u8], e: &Exl2Group) -> &'a [u8] {
        &row[4 * e.woff..4 * (e.woff + e.words())]
    }

    /// Scale of group `g` (index `e`) in `row`: `(s + 1)^2 * max / 256`.
    pub fn scale(&self, row: &[u8], g: usize, e: &Exl2Group) -> f32 {
        let words = self.header.row_bytes / 4 - self.header.groups.div_ceil(8);
        let s = (word(row, words + g / 8) >> (4 * (g % 8))) & 15;
        ((s + 1) * (s + 1)) as f32 * e.max_scaled()
    }

    /// Dequantize row `r` into original input order (`out.len() == cols`), F32 scale arithmetic.
    pub fn dequantize_row(&self, r: usize, out: &mut [f32]) {
        let row = self.row(r);
        let mut tmp = Vec::new();
        for g in 0..self.header.groups {
            let e = self.group(g);
            tmp.resize(e.n, 0.0);
            unpack_centered(self.group_words(row, &e), e.bits, &mut tmp);
            let scale = self.scale(row, g, &e);
            for (i, &c) in tmp.iter().enumerate() {
                out[self.perm(e.start + i)] = c * scale;
            }
        }
    }

    /// The whole matrix, row-major `[rows, cols]`.
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = vec![0.0; self.header.rows * self.header.cols];
        for (r, row) in out.chunks_exact_mut(self.header.cols).enumerate() {
            self.dequantize_row(r, row);
        }
        out
    }

    /// A blob holding only rows `first..first + n` (the tables are copied).
    pub fn slice_rows(&self, first: usize, n: usize) -> Result<Vec<u8>, String> {
        if n == 0 || first.checked_add(n).is_none_or(|e| e > self.header.rows) {
            return Err("EXL2 row range out of bounds".into());
        }
        let h = &self.header;
        let mut out = self.blob[..h.data_off].to_vec();
        out.extend_from_slice(&self.blob[h.data_off + first * h.row_bytes..h.data_off + (first + n) * h.row_bytes]);
        let total = out.len();
        out[8..12].copy_from_slice(&(n as u32).to_le_bytes());
        out[16..24].copy_from_slice(&(total as u64).to_le_bytes());
        Ok(out)
    }
}

// ---------------------------------------------------------------------------------------------------------------- EXL3

pub fn exl3_param(bits: usize, codebook: u8) -> u32 {
    bits as u32 | (u32::from(codebook) << 8)
}

/// Geometry of an EXL3 blob, shared by the writer and the reader.
#[derive(Clone, Debug)]
pub struct Exl3Layout {
    pub header: Header,
}

impl Exl3Layout {
    pub fn new(rows: usize, cols: usize, bits: usize, codebook: u8) -> Result<Self, String> {
        if !(1..=8).contains(&bits) || codebook > 2 {
            return Err("unsupported EXL3 width or codebook".into());
        }
        if rows == 0 || cols == 0 || rows % 128 != 0 || cols % 128 != 0 {
            return Err("EXL3 dimensions must be multiples of 128".into());
        }
        let off_a = HEADER_LEN;
        let off_b = align16(off_a + 2 * cols);
        let off_c = align16(off_b + 2 * rows);
        let data_off = align16(off_c + 4 * rows);
        let row_bytes = cols / 16 * 32 * bits;
        let total = (rows / 16)
            .checked_mul(row_bytes)
            .and_then(|n| n.checked_add(data_off))
            .filter(|&n| n < u32::MAX as usize && rows < u32::MAX as usize && cols < u32::MAX as usize)
            .ok_or("EXL3 tensor exceeds the 4 GiB blob limit")?;
        Ok(Self {
            header: Header { exl3: true, rows, cols, total, data_off, row_bytes, groups: 0, param: exl3_param(bits, codebook), off_a, off_b, off_c },
        })
    }

    /// Everything in front of the trellis tiles: header, `suh` / `svh` (f16 bits) and `out_map`.
    pub fn prefix(&self, suh: &[u16], svh: &[u16], out_map: &[u32]) -> Vec<u8> {
        let h = &self.header;
        assert_eq!(suh.len(), h.cols);
        assert_eq!(svh.len(), h.rows);
        assert_eq!(out_map.len(), h.rows);
        let mut out = vec![0u8; h.data_off];
        out[..HEADER_LEN].copy_from_slice(&h.write());
        for (i, v) in suh.iter().enumerate() {
            out[h.off_a + 2 * i..h.off_a + 2 * i + 2].copy_from_slice(&v.to_le_bytes());
        }
        for (i, v) in svh.iter().enumerate() {
            out[h.off_b + 2 * i..h.off_b + 2 * i + 2].copy_from_slice(&v.to_le_bytes());
        }
        for (i, v) in out_map.iter().enumerate() {
            out[h.off_c + 4 * i..h.off_c + 4 * i + 4].copy_from_slice(&v.to_le_bytes());
        }
        out
    }
}

/// Codebook value of a 16-bit trellis state (F16 arithmetic of the upstream decoder, result as F32).
pub fn codebook_value(state: u16, codebook: u8) -> f32 {
    let h = |x: f32| f16::from_f32(x).to_f32();
    let x = u32::from(state);
    if codebook == 2 {
        let x = x.wrapping_mul(0x83dc_d12d);
        let sum = x.to_le_bytes().iter().map(|&b| u16::from(b)).sum::<u16>() + 0x6400;
        return h(f16::from_bits(sum).to_f32().mul_add(f16::from_bits(0x1eee).to_f32(), f16::from_bits(0xc931).to_f32()));
    }
    let x = if codebook == 1 { x.wrapping_mul(0xcbac_1fed) } else { x.wrapping_mul(89_226_354).wrapping_add(64_248_484) };
    let x = (x & 0x8fff_8fff) ^ 0x3b60_3b60;
    h(f16::from_bits(x as u16).to_f32() + f16::from_bits((x >> 16) as u16).to_f32())
}

/// All 65536 codebook values of `codebook` (0, 1 or 2), computed once.
pub fn codebook_lut(codebook: u8) -> &'static [f32] {
    static LUTS: [OnceLock<Vec<f32>>; 3] = [OnceLock::new(), OnceLock::new(), OnceLock::new()];
    LUTS[usize::from(codebook.min(2))].get_or_init(|| (0..=u16::MAX).map(|s| codebook_value(s, codebook.min(2))).collect())
}

/// 16-bit window of a tail-biting ring of `words.len()` u32 words, MSB-first from bit `start` (wrapping).
fn window16(words: &[u32], start: usize) -> u16 {
    let w = start >> 5;
    let off = start & 31;
    let hi = u64::from(words[w]);
    let lo = u64::from(words[(w + 1) % words.len()]);
    ((((hi << 32) | lo) >> (48 - off)) & 0xffff) as u16
}

/// Position inside a decoded 16x16 tile of the symbol `8 * lane + j`: `(row, column)`.
pub fn tile_position(index: usize) -> (usize, usize) {
    const ROWS: [usize; 8] = [0, 1, 8, 9, 0, 1, 8, 9];
    let (lane, j) = (index / 8, index % 8);
    (2 * (lane % 4) + ROWS[j], lane / 4 + if j < 4 { 0 } else { 8 })
}

/// Normalized in-place Hadamard transform of 128 values (natural order).
pub fn hadamard128(v: &mut [f32]) {
    let mut stride = 1;
    while stride < 128 {
        for begin in (0..128).step_by(stride * 2) {
            for i in begin..begin + stride {
                let (a, b) = (v[i], v[i + stride]);
                v[i] = a + b;
                v[i + stride] = a - b;
            }
        }
        stride *= 2;
    }
    let norm = 128f32.sqrt();
    for x in v.iter_mut().take(128) {
        *x /= norm;
    }
}

/// Reader over a validated EXL3 blob.
#[derive(Clone, Copy, Debug)]
pub struct Exl3View<'a> {
    pub header: Header,
    blob: &'a [u8],
}

impl<'a> Exl3View<'a> {
    pub fn parse(blob: &'a [u8]) -> Result<Self, String> {
        let header = Header::parse(blob)?;
        if !header.exl3 {
            return Err("not an EXL3 tensor".into());
        }
        let bits = (header.param & 0xff) as usize;
        let codebook = header.param >> 8;
        if !(1..=8).contains(&bits) || codebook > 2 || header.rows % 128 != 0 || header.cols % 128 != 0 {
            return Err("EXL3 tensor parameters invalid".into());
        }
        if header.row_bytes != header.cols / 16 * 32 * bits
            || header.off_a + 2 * header.cols > header.off_b
            || header.off_b + 2 * header.rows > header.off_c
            || header.off_c + 4 * header.rows > header.data_off
        {
            return Err("EXL3 sections inconsistent".into());
        }
        Ok(Self { header, blob })
    }

    /// Full validation: structure plus `out_map` being a permutation and the sign vectors being finite.
    pub fn validate(blob: &'a [u8]) -> Result<Self, String> {
        let v = Self::parse(blob)?;
        let mut seen = vec![false; v.header.rows];
        for o in 0..v.header.rows {
            let d = v.out_row(o);
            if d >= v.header.rows || std::mem::replace(&mut seen[d], true) {
                return Err(format!("EXL3 output map invalid at {o}"));
            }
            if !v.svh(o).is_finite() {
                return Err("EXL3 output transform is not finite".into());
            }
        }
        if (0..v.header.cols).any(|i| !v.suh(i).is_finite()) {
            return Err("EXL3 input transform is not finite".into());
        }
        Ok(v)
    }

    pub fn rows(&self) -> usize {
        self.header.rows
    }
    pub fn cols(&self) -> usize {
        self.header.cols
    }
    pub fn bits(&self) -> usize {
        (self.header.param & 0xff) as usize
    }
    pub fn codebook(&self) -> u8 {
        (self.header.param >> 8) as u8
    }
    pub fn suh(&self, i: usize) -> f32 {
        let at = self.header.off_a + 2 * i;
        f16::from_bits(u16::from_le_bytes([self.blob[at], self.blob[at + 1]])).to_f32()
    }
    pub fn svh(&self, o: usize) -> f32 {
        let at = self.header.off_b + 2 * o;
        f16::from_bits(u16::from_le_bytes([self.blob[at], self.blob[at + 1]])).to_f32()
    }
    /// Output row that result channel `o` is written to.
    pub fn out_row(&self, o: usize) -> usize {
        word(&self.blob[self.header.off_c..], o) as usize
    }

    /// The `32 * K` bytes of the tile of output tile column `no` and input tile `ki`.
    pub fn tile(&self, no: usize, ki: usize) -> &'a [u8] {
        let len = 32 * self.bits();
        let at = self.header.data_off + no * self.header.row_bytes + ki * len;
        &self.blob[at..at + len]
    }

    /// Decode the 16x16 tile into `out[row * 16 + col]` (`row`: input within the tile, `col`: output channel).
    pub fn decode_tile(&self, no: usize, ki: usize, lut: &[f32], out: &mut [f32; 256]) {
        let bits = self.bits();
        let bytes = self.tile(no, ki);
        let mut words = [0u32; 64];
        let nw = 8 * bits;
        for (i, w) in words[..nw].iter_mut().enumerate() {
            *w = word(bytes, i);
        }
        let ring = 256 * bits;
        for index in 0..256 {
            let state = window16(&words[..nw], ((index + 1) * bits + ring - 16) % ring);
            let (r, c) = tile_position(index);
            out[r * 16 + c] = lut[usize::from(state)];
        }
    }

    /// The whole matrix, row-major `[rows, cols]` (rows in output order), F32 arithmetic.
    pub fn dequantize(&self) -> Vec<f32> {
        let (rows, cols) = (self.header.rows, self.header.cols);
        let lut = codebook_lut(self.codebook());
        let mut out = vec![0.0; rows * cols];
        let mut block = vec![0.0f32; 128 * 128];
        let mut tile = [0.0f32; 256];
        let mut column = [0.0f32; 128];
        for ob in 0..rows / 128 {
            for ib in 0..cols / 128 {
                for n in 0..8 {
                    for k in 0..8 {
                        self.decode_tile(ob * 8 + n, ib * 8 + k, lut, &mut tile);
                        for r in 0..16 {
                            for c in 0..16 {
                                block[(k * 16 + r) * 128 + n * 16 + c] = tile[r * 16 + c];
                            }
                        }
                    }
                }
                for o in 0..128 {
                    for i in 0..128 {
                        column[i] = block[i * 128 + o];
                    }
                    hadamard128(&mut column);
                    for i in 0..128 {
                        block[i * 128 + o] = column[i] * self.suh(ib * 128 + i);
                    }
                }
                for i in 0..128 {
                    let row = &mut block[i * 128..(i + 1) * 128];
                    hadamard128(row);
                    for (o, x) in row.iter_mut().enumerate() {
                        *x *= self.svh(ob * 128 + o);
                        out[self.out_row(ob * 128 + o) * cols + ib * 128 + i] = *x;
                    }
                }
            }
        }
        out
    }
}

/// Expand any native EXL blob to a row-major F32 matrix (the fallback of backends without a native kernel).
pub fn dequantize(blob: &[u8], type_id: u32, count: usize) -> Result<Vec<f32>, String> {
    let (values, exl3) = if type_id == EXL3_TYPE {
        (Exl3View::validate(blob)?.dequantize(), true)
    } else {
        (Exl2View::validate(blob)?.dequantize(), false)
    };
    if values.len() != count {
        return Err(format!("EXL{} tensor holds {} values, expected {count}", if exl3 { 3 } else { 2 }, values.len()));
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_all_widths_straddle_words() {
        for bits in [2usize, 3, 4, 5, 6, 8] {
            for n in [1usize, 5, 31, 32, 33, 64, 100] {
                let codes: Vec<u8> = (0..n).map(|i| ((i * 37 + 11) & ((1 << bits) - 1)) as u8).collect();
                let mut bytes = Vec::new();
                pack_codes(&codes, bits, &mut bytes);
                assert_eq!(bytes.len(), 4 * (n * bits).div_ceil(32), "bits {bits} n {n}");
                let mut out = vec![0.0; n];
                unpack_centered(&bytes, bits, &mut out);
                for i in 0..n {
                    assert_eq!(out[i], f32::from(codes[i]) - (1 << (bits - 1)) as f32, "bits {bits} n {n} i {i}");
                }
            }
        }
    }

    #[test]
    fn three_bit_codes_cross_the_32_bit_word_boundary() {
        // code 10 starts at bit 30: two bits in word 0, one bit in word 1
        let mut codes = vec![0u8; 12];
        codes[10] = 0b101;
        let mut bytes = Vec::new();
        pack_codes(&codes, 3, &mut bytes);
        assert_eq!(word(&bytes, 0) >> 30, 0b01);
        assert_eq!(word(&bytes, 1) & 1, 1);
        let mut out = vec![0.0; 12];
        unpack_centered(&bytes, 3, &mut out);
        assert_eq!(out[10], 1.0);
    }

    fn sample_exl2(rows: usize) -> (Vec<u8>, Vec<(usize, usize)>) {
        let groups = vec![(32usize, 2usize), (32, 3), (64, 4), (32, 5), (32, 6), (32, 8), (40, 3)];
        let cols: usize = groups.iter().map(|g| g.0).sum();
        let layout = Exl2Layout::new(rows, cols, &groups).unwrap();
        let maxima: Vec<u16> = (0..groups.len()).map(|g| f16::from_f32(64.0 + g as f32).to_bits()).collect();
        let perm: Vec<u32> = (0..cols as u32).rev().collect();
        let mut blob = layout.prefix(&maxima, &perm);
        for r in 0..rows {
            let mut nib = vec![0u32; groups.len().div_ceil(8)];
            for (g, &(n, bits)) in groups.iter().enumerate() {
                let codes: Vec<u8> = (0..n).map(|i| ((i + r * 3 + g) & ((1 << bits) - 1)) as u8).collect();
                pack_codes(&codes, bits, &mut blob);
                nib[g / 8] |= (((r + g) % 16) as u32) << (4 * (g % 8));
            }
            for w in nib {
                blob.extend_from_slice(&w.to_le_bytes());
            }
        }
        (blob, groups)
    }

    #[test]
    fn exl2_blob_round_trip_and_dequantize() {
        let (blob, groups) = sample_exl2(5);
        let v = Exl2View::validate(&blob).unwrap();
        assert_eq!(v.groups(), groups.len());
        assert_eq!(v.cols(), 264);
        let m = v.dequantize();
        let g_index = |col_packed: usize| {
            let mut s = 0;
            for (g, &(n, _)) in groups.iter().enumerate() {
                if col_packed < s + n {
                    return (g, col_packed - s);
                }
                s += n;
            }
            unreachable!()
        };
        for r in 0..5 {
            for k in 0..264 {
                let (g, i) = g_index(k);
                let bits = groups[g].1;
                let code = ((i + r * 3 + g) & ((1 << bits) - 1)) as i32 - (1 << (bits - 1));
                let s = ((r + g) % 16) as f32 + 1.0;
                let expect = code as f32 * s * s * (64.0 + g as f32) / 256.0;
                // permutation is a full reversal: stored input k is original input cols - 1 - k
                assert_eq!(m[r * 264 + (263 - k)], expect, "row {r} stored input {k}");
            }
        }
        let part = v.slice_rows(2, 2).unwrap();
        let pv = Exl2View::validate(&part).unwrap();
        assert_eq!(pv.rows(), 2);
        assert_eq!(pv.dequantize(), m[2 * 264..4 * 264]);
        assert!(v.slice_rows(4, 2).is_err());
    }

    #[test]
    fn exl2_corruption_is_rejected() {
        let (blob, _) = sample_exl2(2);
        let mut bad = blob.clone();
        bad[HEADER_LEN + 8] ^= 1; // woff of group 0
        assert!(Exl2View::parse(&bad).is_err());
        let mut bad = blob.clone();
        let perm_at = Header::parse(&blob).unwrap().off_b;
        bad[perm_at..perm_at + 4].copy_from_slice(&5u32.to_le_bytes()); // duplicate index
        assert!(Exl2View::validate(&bad).is_err());
        assert!(Exl2View::parse(&blob[..blob.len() - 4]).is_err());
        let mut bad = blob.clone();
        bad[HEADER_LEN + 4] = 7; // 7-bit group
        assert!(Exl2View::parse(&bad).is_err());
        assert_eq!(payload_len(&blob, EXL2_TYPE).unwrap(), blob.len() as u64);
        assert!(payload_len(&blob, EXL3_TYPE).is_err());
    }

    #[test]
    fn tile_positions_cover_the_tile_once() {
        let mut seen = [false; 256];
        for i in 0..256 {
            let (r, c) = tile_position(i);
            assert!(r < 16 && c < 16);
            assert!(!std::mem::replace(&mut seen[r * 16 + c], true));
        }
    }

    #[test]
    fn window_reads_wrap_around_the_ring() {
        let words = [0xdead_beefu32, 0x0123_4567];
        // bits 56..64 and 0..8 : 0x67 then 0xde
        assert_eq!(window16(&words, 56), 0x67de);
        assert_eq!(window16(&words, 0), 0xdead);
        assert_eq!(window16(&words, 16), 0xbeef);
        assert_eq!(window16(&words, 24), 0xef01);
    }

    #[test]
    fn hadamard_is_orthonormal_and_involutive() {
        let mut v: Vec<f32> = (0..128).map(|i| (i as f32 * 0.37).sin()).collect();
        let orig = v.clone();
        hadamard128(&mut v);
        let e0: f32 = orig.iter().map(|x| x * x).sum();
        let e1: f32 = v.iter().map(|x| x * x).sum();
        assert!((e0 - e1).abs() < 1e-3);
        hadamard128(&mut v);
        for (a, b) in v.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
