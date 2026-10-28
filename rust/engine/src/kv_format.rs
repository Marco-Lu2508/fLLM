//! KV cache storage formats shared by the CPU decoder, the CPU backend and (through the same layout
//! numbers) the CUDA kernels in `cuda/kv_quant.cu`.
//!
//! A cache pool holds `slots x kv` values per layer, `kv = kv_heads * dim`, in 16-bit words
//! (little endian). Keys and values may use different formats. The layout of one layer:
//!
//! | region  | content                                                                              |
//! |---------|--------------------------------------------------------------------------------------|
//! | codes   | f32: 2 words per value; f16/bf16: 1 word; 8 bit: 2 values per word; 4 bit and the low |
//! |         | 4 bits of a 5 bit code: 4 values per word (value `i` in bits `4 * (i % 4)`)          |
//! | high    | 5 bit formats: the 5th bit, 16 values per word (value `i` in bit `i % 16`)           |
//! | scales  | one f16 per group of `group` values (nvfp4: one UE4M3 byte per 16 values)            |
//! | minima  | one f16 per group (the `_1` formats)                                                 |
//!
//! Every region starts on a word. The f16 and q8_0 layouts are the original ones of fLLM
//! (`kv_kind` 1 and 2); the other formats extend them.
//!
//! `kv_kind` (an `i32` in `Model`, `DeviceKvCache`, ...) packs the two formats: the low byte is the
//! key format, the next byte the value format, and a plain format code means "same for both".
//! `kv_kind == 1` and `== 2` therefore still mean "f16 keys and values" / "q8_0 keys and values".

use half::f16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum KvFormat {
    F16 = 1,
    Q8_0 = 2,
    Bf16 = 3,
    Fp8E4m3 = 4,
    Fp8E5m2 = 5,
    Int8 = 6,
    Q5_0 = 7,
    Q5_1 = 8,
    Q4_0 = 9,
    Q4_1 = 10,
    Iq4Nl = 11,
    Nvfp4 = 12,
    F32 = 13,
}

/// Non-linear 4-bit grid of IQ4_NL (the public ggml format table).
pub const IQ4_NL_GRID: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];
/// Magnitudes of the E2M1 values (sign in bit 3).
pub const FP4_GRID: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
/// Midpoints between neighbouring E2M1 magnitudes; a magnitude above `FP4_MID[i]` rounds to at least `i + 1`.
pub const FP4_MID: [f32; 7] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];

impl KvFormat {
    pub const ALL: [KvFormat; 13] = [
        KvFormat::F16,
        KvFormat::Q8_0,
        KvFormat::Bf16,
        KvFormat::Fp8E4m3,
        KvFormat::Fp8E5m2,
        KvFormat::Int8,
        KvFormat::Q5_0,
        KvFormat::Q5_1,
        KvFormat::Q4_0,
        KvFormat::Q4_1,
        KvFormat::Iq4Nl,
        KvFormat::Nvfp4,
        KvFormat::F32,
    ];

    pub fn code(self) -> i32 {
        self as u8 as i32
    }

    pub fn from_code(code: i32) -> Option<Self> {
        Self::ALL.iter().copied().find(|f| f.code() == code)
    }

    /// Parse the names of `--kv-cache-dtype` (vLLM / SGLang) and `--cache-type-k/-v` (llama.cpp).
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name.trim().to_ascii_lowercase().as_str() {
            "auto" | "f16" | "fp16" | "half" | "float16" => Self::F16,
            "bf16" | "bfloat16" => Self::Bf16,
            "fp8" | "fp8_e4m3" | "e4m3" | "fp8e4m3" => Self::Fp8E4m3,
            "fp8_e5m2" | "e5m2" | "fp8e5m2" => Self::Fp8E5m2,
            "int8" | "i8" => Self::Int8,
            "q8_0" | "q8" => Self::Q8_0,
            "q5_0" => Self::Q5_0,
            "q5_1" => Self::Q5_1,
            "q4_0" => Self::Q4_0,
            "q4_1" => Self::Q4_1,
            "iq4_nl" => Self::Iq4Nl,
            "nvfp4" | "fp4" | "fp4_e2m1" => Self::Nvfp4,
            "f32" | "fp32" | "float32" => Self::F32,
            _ => return None,
        })
    }

    pub const NAMES: &'static str =
        "f16, bf16, fp8_e4m3 (fp8), fp8_e5m2, int8, q8_0, q5_0, q5_1, q4_0, q4_1, iq4_nl, nvfp4, f32";

    pub fn name(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::Fp8E4m3 => "fp8_e4m3",
            Self::Fp8E5m2 => "fp8_e5m2",
            Self::Int8 => "int8",
            Self::Q8_0 => "q8_0",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Iq4Nl => "iq4_nl",
            Self::Nvfp4 => "nvfp4",
            Self::F32 => "f32",
        }
    }

    /// Formats the pre-existing CUDA / wgpu attention kernels read (`kv_kind` 1 and 2).
    pub fn is_legacy(self) -> bool {
        matches!(self, Self::F16 | Self::Q8_0)
    }

    /// `Some(reason)` for formats that are offered with a start-up warning.
    pub fn experimental_note(self) -> Option<&'static str> {
        match self {
            Self::Nvfp4 => Some(
                "4-bit E2M1 values with an FP8 scale per 16 values lose a lot of precision; other engines reported \
                 gibberish and loops on long generations. Use only for experiments",
            ),
            _ => None,
        }
    }

    /// Start-up warning for key formats whose accuracy depends on the model: the keys of some models (large outlier
    /// channels, e.g. Qwen2.5-1.5B: top-1 agreement with the FP16 cache drops to 1-6 of 30 prompts with fp8, q5 and q4)
    /// do not survive fewer than 8 bits, while Llama-3.1-8B stays within about 0.2 in log-probability.
    pub fn key_accuracy_note(self) -> Option<&'static str> {
        match self {
            Self::Fp8E4m3 | Self::Fp8E5m2 | Self::Q5_0 | Self::Q5_1 | Self::Q4_0 | Self::Q4_1 | Self::Iq4Nl | Self::Nvfp4 => Some(
                "keys below 8 bit of precision can break models with large key outliers (Qwen2.5 does); check the output \
                 or keep the keys at q8_0 / int8 (--cache-type-k q8_0) and quantize only the values",
            ),
            _ => None,
        }
    }

    fn code_bits(self) -> usize {
        match self {
            Self::F32 => 32,
            Self::F16 | Self::Bf16 => 16,
            Self::Fp8E4m3 | Self::Fp8E5m2 | Self::Int8 | Self::Q8_0 => 8,
            Self::Q5_0 | Self::Q5_1 => 5,
            Self::Q4_0 | Self::Q4_1 | Self::Iq4Nl | Self::Nvfp4 => 4,
        }
    }

    /// Bits per value including scales and minima for heads of width `dim`.
    pub fn bits_per_value(self, dim: usize) -> f64 {
        let group = self.group(dim) as f64;
        self.code_bits() as f64
            + (self.scale_bits() + self.min_bits()) as f64 / group
    }

    fn scale_bits(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::Bf16 | Self::Fp8E5m2 => 0,
            Self::Nvfp4 => 8,
            _ => 16,
        }
    }

    fn min_bits(self) -> usize {
        match self {
            Self::Q5_1 | Self::Q4_1 => 16,
            _ => 0,
        }
    }

    /// Values sharing one scale: 32 for the block formats (the whole head when the width is not a
    /// multiple of 32), a whole head for int8 / fp8_e4m3 (one scale per token and head), 16 for nvfp4.
    pub fn group(self, dim: usize) -> usize {
        match self {
            Self::Q8_0 | Self::Q5_0 | Self::Q5_1 | Self::Q4_0 | Self::Q4_1 | Self::Iq4Nl => {
                if dim.is_multiple_of(32) {
                    32
                } else {
                    dim
                }
            }
            Self::Int8 | Self::Fp8E4m3 => dim,
            Self::Nvfp4 => 16,
            _ => dim,
        }
    }

    /// Largest error of a decoded value relative to the largest magnitude of its group that the format may show
    /// (rounding to the nearest code plus the rounding of the scale).
    pub fn max_relative_error(self) -> f64 {
        match self {
            Self::F32 => 1e-6,
            Self::F16 => 1e-3,
            Self::Bf16 => 8e-3,
            Self::Q8_0 | Self::Int8 => 0.0085,
            Self::Fp8E4m3 => 0.075,
            Self::Fp8E5m2 => 0.14,
            Self::Q5_0 | Self::Q5_1 => 0.07,
            Self::Q4_0 | Self::Q4_1 => 0.15,
            Self::Iq4Nl => 0.16,
            Self::Nvfp4 => 0.4,
        }
    }

    /// Reject head widths whose groups would not start on a word of every region.
    pub fn check_dim(self, dim: usize) -> Result<(), String> {
        let group = self.group(dim);
        let ok = match self {
            Self::F32 | Self::F16 | Self::Bf16 | Self::Q8_0 => true,
            Self::Fp8E5m2 | Self::Int8 | Self::Fp8E4m3 => group.is_multiple_of(2),
            Self::Q4_0 | Self::Q4_1 | Self::Iq4Nl => group.is_multiple_of(4),
            Self::Q5_0 | Self::Q5_1 => group.is_multiple_of(16),
            Self::Nvfp4 => dim.is_multiple_of(32),
        };
        if ok {
            Ok(())
        } else {
            Err(format!(
                "KV cache format {} does not support a head width of {dim}",
                self.name()
            ))
        }
    }

    /// Bytes one slot (token) of one layer takes for `kv_heads` heads of width `dim`.
    pub fn bytes_per_slot(self, kv_heads: usize, dim: usize) -> u64 {
        let kv = kv_heads * dim;
        let bits = kv * self.code_bits() + kv / self.group(dim) * (self.scale_bits() + self.min_bits());
        bits.div_ceil(8) as u64
    }

    /// 16-bit words of one layer.
    pub fn layer_words(self, slots: usize, kv_heads: usize, dim: usize) -> usize {
        KvLayout::new(self, slots, kv_heads, dim).words
    }
}

/// Split a packed `kv_kind` into (key, value) formats.
pub fn unpack_kind(kind: i32) -> Option<(KvFormat, KvFormat)> {
    if kind <= 0 {
        return None;
    }
    let k = KvFormat::from_code(kind & 0xff)?;
    let v = match kind >> 8 {
        0 => k,
        c => KvFormat::from_code(c)?,
    };
    Some((k, v))
}

/// Pack (key, value) formats into a `kv_kind`.
pub fn pack_kind(k: KvFormat, v: KvFormat) -> i32 {
    if k == v {
        k.code()
    } else {
        k.code() | v.code() << 8
    }
}

/// Region offsets (in 16-bit words) of one layer of a pool.
#[derive(Clone, Copy, Debug)]
pub struct KvLayout {
    pub fmt: KvFormat,
    pub slots: usize,
    pub kv: usize,
    pub dim: usize,
    pub group: usize,
    pub hi_off: usize,
    pub scale_off: usize,
    pub min_off: usize,
    pub words: usize,
}

impl KvLayout {
    pub fn new(fmt: KvFormat, slots: usize, kv_heads: usize, dim: usize) -> Self {
        let kv = kv_heads * dim;
        let n = slots * kv;
        let group = fmt.group(dim);
        let groups = slots * (kv / group);
        let code_words = match fmt {
            KvFormat::F32 => 2 * n,
            KvFormat::F16 | KvFormat::Bf16 => n,
            KvFormat::Fp8E4m3 | KvFormat::Fp8E5m2 | KvFormat::Int8 | KvFormat::Q8_0 => n.div_ceil(2),
            _ => n.div_ceil(4),
        };
        let hi_words = if fmt.code_bits() == 5 { n.div_ceil(16) } else { 0 };
        let scale_words = match fmt.scale_bits() {
            0 => 0,
            8 => groups.div_ceil(2),
            _ => groups,
        };
        let min_words = if fmt.min_bits() > 0 { groups } else { 0 };
        let hi_off = code_words;
        let scale_off = hi_off + hi_words;
        let min_off = scale_off + scale_words;
        Self { fmt, slots, kv, dim, group, hi_off, scale_off, min_off, words: min_off + min_words }
    }
}

// ---------------------------------------------------------------------------------------------
// FP8 conversion (integer arithmetic, identical in cuda/kv_quant.cu): round to nearest even,
// saturating at the largest finite value (e4m3: 448, e5m2: 57344).

/// `f32` to an FP8 code with `mbits` mantissa bits and exponent bias `bias` (sign in bit 7).
fn f32_to_fp8(x: f32, mbits: u32, bias: i32, max_code: u32) -> u8 {
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let a = x.abs();
    if a.is_nan() {
        return 0;
    }
    let min_normal = f32::from_bits(((1 - bias + 127) as u32) << 23);
    let code = if a < min_normal {
        // subnormals: multiples of 2^(1 - bias - mbits)
        let scaled = a * f32::from_bits(((bias - 1 + mbits as i32 + 127) as u32) << 23);
        scaled.round_ties_even() as u32
    } else {
        let bits = a.to_bits();
        let shift = 23 - mbits;
        let r = (bits + ((1 << (shift - 1)) - 1) + ((bits >> shift) & 1)) >> shift;
        r - (((127 - bias) as u32) << mbits)
    };
    sign | code.min(max_code) as u8
}

fn fp8_to_f32(b: u8, mbits: u32, bias: i32) -> f32 {
    let e = ((b & 0x7f) >> mbits) as i32;
    let m = (b & ((1 << mbits) - 1)) as u32;
    let v = if e == 0 {
        m as f32 * f32::from_bits(((1 - bias - mbits as i32 + 127) as u32) << 23)
    } else {
        f32::from_bits((((e - bias + 127) as u32) << 23) | (m << (23 - mbits)))
    };
    if b & 0x80 != 0 {
        -v
    } else {
        v
    }
}

pub fn f32_to_e4m3(x: f32) -> u8 {
    f32_to_fp8(x, 3, 7, 126)
}
pub fn e4m3_to_f32(b: u8) -> f32 {
    fp8_to_f32(b, 3, 7)
}
pub fn f32_to_e5m2(x: f32) -> u8 {
    f32_to_fp8(x, 2, 15, 123)
}
pub fn e5m2_to_f32(b: u8) -> f32 {
    fp8_to_f32(b, 2, 15)
}

pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) | 0x40) as u16;
    }
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn h(bits: u16) -> f32 {
    f16::from_bits(bits).to_f32()
}
fn hb(x: f32) -> u16 {
    f16::from_f32(x).to_bits()
}

/// Element with the largest magnitude (first of equals), with its sign.
fn signed_max(x: &[f32]) -> f32 {
    let mut v = 0.0f32;
    for &e in x {
        if e.abs() > v.abs() {
            v = e;
        }
    }
    v
}

/// Quantize one group. `codes` receives one code per value; returns (scale bits, minimum bits).
/// For nvfp4 the scale is the UE4M3 byte.
pub fn quantize_group(fmt: KvFormat, x: &[f32], codes: &mut [u8]) -> (u16, u16) {
    debug_assert_eq!(x.len(), codes.len());
    match fmt {
        KvFormat::Q8_0 => {
            let mut scale = 0u16;
            crate::cpu_kernels::encode_q8_cache_group(x, codes, &mut scale);
            (scale, 0)
        }
        KvFormat::Int8 => {
            let mut scale = 0u16;
            crate::cpu_kernels::encode_q8_cache_group_scalar(x, codes, &mut scale);
            (scale, 0)
        }
        KvFormat::Fp8E4m3 => {
            let amax = x.iter().fold(0.0f32, |m, e| m.max(e.abs()));
            let sb = hb(amax / 448.0);
            let d = h(sb);
            for (c, &e) in codes.iter_mut().zip(x) {
                *c = if d == 0.0 { 0 } else { f32_to_e4m3(e / d) };
            }
            (sb, 0)
        }
        KvFormat::Q4_0 | KvFormat::Q5_0 => {
            let (levels, zero) = if fmt == KvFormat::Q4_0 { (15, 8.0f32) } else { (31, 16.0f32) };
            let sb = hb(signed_max(x) / -zero);
            let d = h(sb);
            for (c, &e) in codes.iter_mut().zip(x) {
                *c = if d == 0.0 {
                    zero as u8
                } else {
                    ((e / d) + (zero + 0.5)).floor().clamp(0.0, levels as f32) as u8
                };
            }
            (sb, 0)
        }
        KvFormat::Q4_1 | KvFormat::Q5_1 => {
            let levels = if fmt == KvFormat::Q4_1 { 15.0f32 } else { 31.0f32 };
            let mn = x.iter().copied().fold(f32::INFINITY, f32::min);
            let mx = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sb = hb((mx - mn) / levels);
            let mb = hb(mn);
            let (d, m) = (h(sb), h(mb));
            for (c, &e) in codes.iter_mut().zip(x) {
                *c = if d == 0.0 {
                    0
                } else {
                    (((e - m) / d) + 0.5).floor().clamp(0.0, levels) as u8
                };
            }
            (sb, mb)
        }
        KvFormat::Iq4Nl => {
            let v = signed_max(x);
            if v == 0.0 {
                codes.fill(8);
                return (0, 0);
            }
            let d0 = v / -127.0;
            let (mut sxg, mut sgg) = (0.0f32, 0.0f32);
            for (c, &e) in codes.iter_mut().zip(x) {
                let t = e / d0;
                let mut best = 0usize;
                let mut best_err = f32::INFINITY;
                for (i, &g) in IQ4_NL_GRID.iter().enumerate() {
                    let err = (t - g as f32).abs();
                    if err < best_err {
                        best_err = err;
                        best = i;
                    }
                }
                *c = best as u8;
                let g = IQ4_NL_GRID[best] as f32;
                sxg = e.mul_add(g, sxg);
                sgg += g * g;
            }
            (hb(sxg / sgg), 0)
        }
        KvFormat::Nvfp4 => {
            let amax = x.iter().fold(0.0f32, |m, e| m.max(e.abs()));
            let sb = f32_to_e4m3(amax / 6.0);
            let d = e4m3_to_f32(sb);
            for (c, &e) in codes.iter_mut().zip(x) {
                let sign = if e.is_sign_negative() { 8u8 } else { 0 };
                let idx = if d == 0.0 {
                    0
                } else {
                    let a = e.abs() / d;
                    FP4_MID.iter().filter(|&&mid| a > mid).count() as u8
                };
                *c = sign | idx;
            }
            (sb as u16, 0)
        }
        KvFormat::F32 | KvFormat::F16 | KvFormat::Bf16 | KvFormat::Fp8E5m2 => {
            unreachable!("unscaled formats are encoded per value")
        }
    }
}

/// Decode one code of a group with scale/minimum `s`, `m` (see [`quantize_group`]).
#[inline]
fn dequant_code(fmt: KvFormat, code: u32, s: u16, m: u16) -> f32 {
    match fmt {
        KvFormat::Q8_0 | KvFormat::Int8 => (code as u8 as i8) as f32 * h(s),
        KvFormat::Fp8E4m3 => e4m3_to_f32(code as u8) * h(s),
        KvFormat::Q4_0 => (code as f32 - 8.0) * h(s),
        KvFormat::Q5_0 => (code as f32 - 16.0) * h(s),
        KvFormat::Q4_1 | KvFormat::Q5_1 => code as f32 * h(s) + h(m),
        KvFormat::Iq4Nl => IQ4_NL_GRID[code as usize & 15] as f32 * h(s),
        KvFormat::Nvfp4 => {
            let v = FP4_GRID[code as usize & 7] * e4m3_to_f32(s as u8);
            if code & 8 != 0 {
                -v
            } else {
                v
            }
        }
        _ => unreachable!(),
    }
}

fn get_field(words: &[u16], base: usize, idx: usize, bits: usize) -> u32 {
    let per = 16 / bits;
    let w = words[base + idx / per] as u32;
    (w >> ((idx % per) * bits)) & ((1 << bits) - 1)
}

fn put_field(words: &mut [u16], base: usize, idx: usize, bits: usize, value: u32) {
    let per = 16 / bits;
    let shift = (idx % per) * bits;
    let mask = (((1u32 << bits) - 1) << shift) as u16;
    let w = &mut words[base + idx / per];
    *w = (*w & !mask) | ((value << shift) as u16 & mask);
}

impl KvLayout {
    /// `(offset, words per block)` of every region when the blocks of `block_size` slots each take a
    /// whole number of words in every region (needed to copy a block as a unit); `None` otherwise.
    pub fn block_regions(&self, block_size: usize) -> Option<Vec<(usize, usize)>> {
        let n = block_size * self.kv;
        let groups = block_size * (self.kv / self.group);
        let mut regions = Vec::new();
        let (code, hi) = match self.fmt {
            KvFormat::F32 => (2 * n, 0),
            KvFormat::F16 | KvFormat::Bf16 => (n, 0),
            KvFormat::Fp8E4m3 | KvFormat::Fp8E5m2 | KvFormat::Int8 | KvFormat::Q8_0 => {
                (n.is_multiple_of(2).then_some(n / 2)?, 0)
            }
            KvFormat::Q5_0 | KvFormat::Q5_1 => (
                n.is_multiple_of(4).then_some(n / 4)?,
                n.is_multiple_of(16).then_some(n / 16)?,
            ),
            _ => (n.is_multiple_of(4).then_some(n / 4)?, 0),
        };
        regions.push((0, code));
        if hi > 0 {
            regions.push((self.hi_off, hi));
        }
        match self.fmt.scale_bits() {
            0 => {}
            8 => regions.push((self.scale_off, groups.is_multiple_of(2).then_some(groups / 2)?)),
            _ => regions.push((self.scale_off, groups)),
        }
        if self.fmt.min_bits() > 0 {
            regions.push((self.min_off, groups));
        }
        Some(regions)
    }

    /// Value `c` (`head * dim + d`) of `slot`; `layer` is the slice of one layer.
    pub fn value(&self, layer: &[u16], slot: usize, c: usize) -> f32 {
        let i = slot * self.kv + c;
        let g = slot * (self.kv / self.group) + c / self.group;
        match self.fmt {
            KvFormat::F32 => f32::from_bits(layer[2 * i] as u32 | (layer[2 * i + 1] as u32) << 16),
            KvFormat::F16 => h(layer[i]),
            KvFormat::Bf16 => f32::from_bits((layer[i] as u32) << 16),
            KvFormat::Fp8E5m2 => e5m2_to_f32(get_field(layer, 0, i, 8) as u8),
            KvFormat::Q8_0 | KvFormat::Int8 | KvFormat::Fp8E4m3 => {
                dequant_code(self.fmt, get_field(layer, 0, i, 8), layer[self.scale_off + g], 0)
            }
            KvFormat::Q5_0 | KvFormat::Q5_1 => {
                let code = get_field(layer, 0, i, 4) | get_field(layer, self.hi_off, i, 1) << 4;
                let m = if self.fmt == KvFormat::Q5_1 { layer[self.min_off + g] } else { 0 };
                dequant_code(self.fmt, code, layer[self.scale_off + g], m)
            }
            KvFormat::Nvfp4 => {
                let s = get_field(layer, self.scale_off, g, 8) as u16;
                dequant_code(self.fmt, get_field(layer, 0, i, 4), s, 0)
            }
            KvFormat::Q4_0 | KvFormat::Q4_1 | KvFormat::Iq4Nl => {
                let m = if self.fmt == KvFormat::Q4_1 { layer[self.min_off + g] } else { 0 };
                dequant_code(self.fmt, get_field(layer, 0, i, 4), layer[self.scale_off + g], m)
            }
        }
    }

    /// Decode a whole slot (`kv` values) into `out`.
    pub fn decode_slot(&self, layer: &[u16], slot: usize, out: &mut [f32]) {
        for (c, o) in out.iter_mut().enumerate().take(self.kv) {
            *o = self.value(layer, slot, c);
        }
    }

    /// Encode `row` (`kv` values) into `slot`.
    pub fn encode_slot(&self, layer: &mut [u16], slot: usize, row: &[f32]) {
        debug_assert_eq!(row.len(), self.kv);
        let base = slot * self.kv;
        match self.fmt {
            KvFormat::F32 => {
                for (j, &x) in row.iter().enumerate() {
                    let b = x.to_bits();
                    layer[2 * (base + j)] = b as u16;
                    layer[2 * (base + j) + 1] = (b >> 16) as u16;
                }
            }
            KvFormat::F16 => {
                let dst = &mut layer[base..base + self.kv];
                if !crate::cpu_float::encode_f16(row, dst) {
                    for (d, &x) in dst.iter_mut().zip(row) {
                        *d = hb(x);
                    }
                }
            }
            KvFormat::Bf16 => {
                for (j, &x) in row.iter().enumerate() {
                    layer[base + j] = f32_to_bf16(x);
                }
            }
            KvFormat::Fp8E5m2 => {
                for (j, &x) in row.iter().enumerate() {
                    put_field(layer, 0, base + j, 8, f32_to_e5m2(x) as u32);
                }
            }
            _ => {
                let mut codes = vec![0u8; self.group];
                let groups = self.kv / self.group;
                for (g, chunk) in row.chunks_exact(self.group).enumerate() {
                    let (s, m) = quantize_group(self.fmt, chunk, &mut codes);
                    let gi = slot * groups + g;
                    let first = base + g * self.group;
                    for (j, &c) in codes.iter().enumerate() {
                        match self.fmt.code_bits() {
                            8 => put_field(layer, 0, first + j, 8, c as u32),
                            5 => {
                                put_field(layer, 0, first + j, 4, c as u32 & 15);
                                put_field(layer, self.hi_off, first + j, 1, c as u32 >> 4);
                            }
                            _ => put_field(layer, 0, first + j, 4, c as u32),
                        }
                    }
                    if self.fmt == KvFormat::Nvfp4 {
                        put_field(layer, self.scale_off, gi, 8, s as u32);
                    } else {
                        layer[self.scale_off + gi] = s;
                    }
                    if self.fmt.min_bits() > 0 {
                        layer[self.min_off + gi] = m;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|i| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let u = (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
                // a few outliers like real key channels
                u * if i % 29 == 0 { 40.0 } else { 3.0 }
            })
            .collect()
    }

    /// Host model of `kvq_write_warp_kernel` (cuda/kv_quant.cu) for the 4 and 5 bit formats: lane `l` of the warp holds the values
    /// `l + 32 k` of the unit, the element of largest magnitude is found with a ballot over the lanes, the nibbles are packed with
    /// `__shfl_down_sync` and the fifth bits are collected with `__ballot_sync`. The arithmetic is that of `quantize_group`; what the model
    /// checks is the lane arithmetic around it (which element is "the first of largest magnitude", where each word goes) against
    /// `encode_slot`. A device ballot that returns garbage (the `kvq_ballot32` that called itself) breaks exactly these two places.
    fn warp_encode(fmt: KvFormat, x: &[f32], first: usize, layout: &KvLayout, pool: &mut [u16]) {
        let unit = x.len();
        let nk = unit.div_ceil(32);
        let ok = |k: usize, lane: usize| lane + 32 * k < unit;
        let v = |k: usize, lane: usize| if ok(k, lane) { x[lane + 32 * k] } else { 0.0 };
        let ballot = |pred: &dyn Fn(usize) -> bool| (0..32).fold(0u32, |m, lane| m | (u32::from(pred(lane)) << lane));
        let amax = (0..nk).flat_map(|k| (0..32).map(move |l| (k, l))).fold(0.0f32, |m, (k, l)| m.max(v(k, l).abs()));
        let (scale, min, codes): (u16, u16, Vec<Vec<u32>>) = if matches!(fmt, KvFormat::Q4_1 | KvFormat::Q5_1) {
            let levels = if fmt == KvFormat::Q4_1 { 15.0f32 } else { 31.0f32 };
            let mn = (0..nk).flat_map(|k| (0..32).map(move |l| (k, l))).filter(|&(k, l)| ok(k, l)).fold(f32::INFINITY, |m, (k, l)| m.min(v(k, l)));
            let mx = (0..nk).flat_map(|k| (0..32).map(move |l| (k, l))).filter(|&(k, l)| ok(k, l)).fold(f32::NEG_INFINITY, |m, (k, l)| m.max(v(k, l)));
            let (sb, mb) = (hb((mx - mn) / levels), hb(mn));
            let (d, m) = (h(sb), h(mb));
            let codes = (0..nk).map(|k| (0..32).map(|l| if d == 0.0 { 0 } else { (((v(k, l) - m) / d) + 0.5).floor().clamp(0.0, levels) as u32 }).collect()).collect();
            (sb, mb, codes)
        } else {
            let mut vmax = 0.0f32;
            let mut found = false;
            for k in 0..nk {
                let mask = ballot(&|l| ok(k, l) && v(k, l).abs() == amax);
                let cand = v(k, if mask != 0 { mask.trailing_zeros() as usize } else { 0 });
                if !found && mask != 0 && amax > 0.0 {
                    vmax = cand;
                    found = true;
                }
            }
            let (zero, levels) = if fmt == KvFormat::Q4_0 { (8.0f32, 15.0f32) } else { (16.0f32, 31.0f32) };
            let sb = hb(vmax / -zero);
            let d = h(sb);
            let codes = (0..nk)
                .map(|k| (0..32).map(|l| if d == 0.0 { zero as u32 } else { ((v(k, l) / d) + (zero + 0.5)).floor().clamp(0.0, levels) as u32 }).collect())
                .collect();
            (sb, 0, codes)
        };
        let five = matches!(fmt, KvFormat::Q5_0 | KvFormat::Q5_1);
        for (k, code) in codes.iter().enumerate() {
            for lane in 0..32usize {
                let idx0 = first + 32 * k + lane;
                if lane % 4 == 0 && ok(k, lane) {
                    // lanes beyond 31 do not exist; the lanes a packing lane reads are inside its quad
                    let nib = |o: usize| code[lane + o] & 15;
                    pool[idx0 >> 2] = (nib(0) | nib(1) << 4 | nib(2) << 8 | nib(3) << 12) as u16;
                }
            }
            if five {
                let bits = ballot(&|l| ok(k, l) && (code[l] >> 4) & 1 == 1);
                let w0 = layout.hi_off + ((first + 32 * k) >> 4);
                if 32 * k < unit {
                    pool[w0] = (bits & 0xffff) as u16;
                }
                if 32 * k + 16 < unit {
                    pool[w0 + 1] = (bits >> 16) as u16;
                }
            }
        }
        let gfirst = first / layout.fmt.group(layout.dim);
        pool[layout.scale_off + gfirst] = scale;
        if matches!(fmt, KvFormat::Q4_1 | KvFormat::Q5_1) {
            pool[layout.min_off + gfirst] = min;
        }
    }

    #[test]
    fn warp_encoder_model_matches_the_cpu_layout() {
        for fmt in [KvFormat::Q4_0, KvFormat::Q5_0, KvFormat::Q4_1, KvFormat::Q5_1] {
            for dim in [32usize, 64, 80, 96, 128] {
                let (slots, kv_heads) = (7usize, 2usize);
                let layout = KvLayout::new(fmt, slots, kv_heads, dim);
                let unit = fmt.group(dim);
                for seed in 1..6u32 {
                    let mut want = vec![0u16; layout.words];
                    let mut got = vec![0u16; layout.words];
                    for slot in [0usize, 3, 6] {
                        let row = sample(layout.kv, seed * 17 + slot as u32);
                        layout.encode_slot(&mut want, slot, &row);
                        for u in 0..layout.kv / unit {
                            warp_encode(fmt, &row[u * unit..(u + 1) * unit], slot * layout.kv + u * unit, &layout, &mut got);
                        }
                    }
                    assert_eq!(got, want, "{} dim {dim} seed {seed}", fmt.name());
                }
            }
        }
    }

    #[test]
    fn kind_packing_round_trips() {
        for &k in &KvFormat::ALL {
            assert_eq!(unpack_kind(pack_kind(k, k)), Some((k, k)));
            for &v in &KvFormat::ALL {
                assert_eq!(unpack_kind(pack_kind(k, v)), Some((k, v)));
            }
        }
        assert_eq!(pack_kind(KvFormat::F16, KvFormat::F16), 1);
        assert_eq!(pack_kind(KvFormat::Q8_0, KvFormat::Q8_0), 2);
        assert_eq!(unpack_kind(0), None);
        assert_eq!(unpack_kind(99), None);
    }

    #[test]
    fn fp8_codes_round_trip_and_saturate() {
        for b in 0..=126u8 {
            assert_eq!(f32_to_e4m3(e4m3_to_f32(b)), b, "e4m3 {b}");
            assert_eq!(f32_to_e4m3(-e4m3_to_f32(b)), b | 0x80);
        }
        for b in 0..=123u8 {
            assert_eq!(f32_to_e5m2(e5m2_to_f32(b)), b, "e5m2 {b}");
        }
        assert_eq!(e4m3_to_f32(126), 448.0);
        assert_eq!(e5m2_to_f32(123), 57344.0);
        assert_eq!(f32_to_e4m3(1e9), 126);
        assert_eq!(f32_to_e5m2(f32::INFINITY), 123);
        assert_eq!(f32_to_e4m3(0.0), 0);
        // ties to even: 1.0625 lies between 1.0 (0x38) and 1.125 (0x39)
        assert_eq!(f32_to_e4m3(1.0625), 0x38);
        assert_eq!(f32_to_e4m3(1.1875), 0x3a);
    }

    #[test]
    fn layouts_keep_the_original_q8_numbers() {
        let l = KvLayout::new(KvFormat::Q8_0, 64, 2, 128);
        assert_eq!(l.scale_off, (64 * 256usize).div_ceil(2));
        assert_eq!(l.words, l.scale_off + 64 * 8);
        let f = KvLayout::new(KvFormat::F16, 64, 2, 128);
        assert_eq!(f.words, 64 * 256);
    }

    #[test]
    fn every_format_round_trips_within_its_quantization_step() {
        let (slots, heads, dim) = (6usize, 2usize, 128usize);
        // largest relative error of the group maximum one can accept per format
        for &fmt in &KvFormat::ALL {
            fmt.check_dim(dim).unwrap();
            let layout = KvLayout::new(fmt, slots, heads, dim);
            let mut layer = vec![0u16; layout.words];
            let rows: Vec<Vec<f32>> = (0..slots).map(|s| sample(heads * dim, 7 + s as u32)).collect();
            for (s, row) in rows.iter().enumerate() {
                layout.encode_slot(&mut layer, s, row);
            }
            let tolerance = fmt.max_relative_error() as f32;
            for (s, row) in rows.iter().enumerate() {
                let mut out = vec![0.0f32; heads * dim];
                layout.decode_slot(&layer, s, &mut out);
                for (g, (a, b)) in row.chunks(layout.group).zip(out.chunks(layout.group)).enumerate() {
                    let amax = a.iter().fold(0.0f32, |m, x| m.max(x.abs()));
                    for (x, y) in a.iter().zip(b) {
                        let err = (x - y).abs();
                        assert!(
                            err <= tolerance * amax + 1e-6,
                            "{} slot {s} group {g}: {x} -> {y} (group max {amax})",
                            fmt.name()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn writes_do_not_disturb_neighbouring_slots() {
        for &fmt in &KvFormat::ALL {
            let (slots, heads, dim) = (4usize, 1usize, 64usize);
            let layout = KvLayout::new(fmt, slots, heads, dim);
            let mut layer = vec![0u16; layout.words];
            let a = sample(dim, 1);
            let b = sample(dim, 2);
            layout.encode_slot(&mut layer, 1, &a);
            let mut before = vec![0.0f32; dim];
            layout.decode_slot(&layer, 1, &mut before);
            layout.encode_slot(&mut layer, 0, &b);
            layout.encode_slot(&mut layer, 2, &b);
            let mut after = vec![0.0f32; dim];
            layout.decode_slot(&layer, 1, &mut after);
            assert_eq!(before, after, "{}", fmt.name());
        }
    }

    #[test]
    fn constant_and_zero_groups_decode_exactly_enough() {
        for &fmt in &KvFormat::ALL {
            let layout = KvLayout::new(fmt, 2, 1, 64);
            let mut layer = vec![0u16; layout.words];
            layout.encode_slot(&mut layer, 0, &[0.0; 64]);
            layout.encode_slot(&mut layer, 1, &[0.5; 64]);
            let mut out = vec![1.0f32; 64];
            layout.decode_slot(&layer, 0, &mut out);
            assert!(out.iter().all(|&x| x == 0.0), "{}", fmt.name());
            layout.decode_slot(&layer, 1, &mut out);
            assert!(out.iter().all(|&x| (x - 0.5).abs() < 0.04), "{}: {out:?}", fmt.name());
        }
    }

    #[test]
    fn outlier_channel_keeps_the_small_values() {
        // one key channel 3 orders of magnitude above the rest, as in several model families
        let mut row: Vec<f32> = (0..128).map(|i| ((i * 37 % 29) as f32 - 14.0) * 0.07).collect();
        row[17] = 2500.0;
        for &fmt in &KvFormat::ALL {
            let layout = KvLayout::new(fmt, 1, 1, 128);
            let mut layer = vec![0u16; layout.words];
            layout.encode_slot(&mut layer, 0, &row);
            let mut out = vec![0.0f32; 128];
            layout.decode_slot(&layer, 0, &mut out);
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for i in (0..128).filter(|&i| i != 17) {
                num += f64::from(out[i] - row[i]).powi(2);
                den += f64::from(row[i]).powi(2);
            }
            println!("{:10} relative error of the small channels {:.4}", fmt.name(), (num / den).sqrt());
        }
    }

    #[test]
    fn bytes_per_slot_matches_layout() {
        for &fmt in &KvFormat::ALL {
            let l = KvLayout::new(fmt, 1024, 8, 128);
            assert_eq!(l.words as u64 * 2, 1024 * fmt.bytes_per_slot(8, 128), "{}", fmt.name());
        }
        assert_eq!(KvFormat::F16.bytes_per_slot(8, 128), 2048);
        assert_eq!(KvFormat::Q8_0.bytes_per_slot(8, 128), 1024 + 64);
        assert_eq!(KvFormat::Nvfp4.bytes_per_slot(8, 128), 512 + 64);
    }
}
