//! Test-only Rust mirror of the chunk decoders in `shaders/vulkan/_q_fmt_*.glsl`: the same word reads, shifts and masks on the device layout
//! of `layout.rs`. The tests compare it with `compute::dequantize`, which checks the repacking and the index arithmetic of the shaders
//! without a GPU (the GLSL text itself is only compiled and validated, never run, in this round).

use super::layout::DeviceLayout;

/// One decoded chunk of 8 weights: `weight[i] = ws * q[i] + wm`.
#[derive(Debug, Clone, Copy)]
pub struct Chunk {
    pub q: [i8; 8],
    pub ws: f32,
    pub wm: f32,
}

fn word(buf: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([buf[4 * i], buf[4 * i + 1], buf[4 * i + 2], buf[4 * i + 3]])
}

/// binary16 number `i` of a plane of halves starting at word `base`
fn f16at(buf: &[u8], base: u32, i: usize) -> f32 {
    let v = word(buf, base as usize + (i >> 1));
    half::f16::from_bits(((v >> (16 * (i & 1))) & 0xffff) as u16).to_f32()
}

fn half2(v: u32) -> (f32, f32) {
    (
        half::f16::from_bits((v & 0xffff) as u16).to_f32(),
        half::f16::from_bits((v >> 16) as u16).to_f32(),
    )
}

fn sub8(x: u32, c: u32) -> u32 {
    ((x | 0x8080_8080).wrapping_sub(c.wrapping_mul(0x0101_0101))) ^ 0x8080_8080
}

fn lanes(p: u32) -> [i8; 4] {
    [
        p as u8 as i8,
        (p >> 8) as u8 as i8,
        (p >> 16) as u8 as i8,
        (p >> 24) as u8 as i8,
    ]
}

const KV_IQ4NL: [i32; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];
const KV_FP4: [i32; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

fn lut(table: &[i32; 16], n: u32) -> u32 {
    (table[(n & 15) as usize] as u32 & 255)
        | ((table[((n >> 8) & 15) as usize] as u32 & 255) << 8)
        | ((table[((n >> 16) & 15) as usize] as u32 & 255) << 16)
        | ((table[((n >> 24) & 15) as usize] as u32) << 24)
}

fn e8m0_half(x: u32) -> f32 {
    f32::from_bits(if x < 2 {
        0x0020_0000u32 << x
    } else {
        (x - 1) << 23
    })
}

fn ue4m3_half(x: u32) -> f32 {
    if x == 0 || x == 0x7f {
        return 0.0;
    }
    let (e, m) = ((x >> 3) & 15, x & 7);
    if e == 0 {
        return m as f32 * (1.0 / 1024.0);
    }
    0.5 * f32::from_bits(((e + 120) << 23) | (m << 20))
}

fn spread(qh: u32) -> (u32, u32) {
    (
        ((qh & 15) * 0x0020_4081) & 0x0101_0101,
        (((qh >> 4) & 15) * 0x0020_4081) & 0x0101_0101,
    )
}

/// `wchunk(g, cin)` of the format of `layout` on the device buffer `buf`.
pub fn chunk(layout: &DeviceLayout, buf: &[u8], g: usize, cin: usize) -> Chunk {
    let off1 = layout.plane_word_offset[1];
    let off2 = layout.plane_word_offset[2];
    let off3 = layout.plane_word_offset[3];
    let w = |i: usize| word(buf, i);
    let (p0, p1, ws, wm): (u32, u32, f32, f32);
    match layout.info.tag {
        "q4_0" => {
            let b = g * 4 + ((cin & 1) << 1);
            let sh = ((cin >> 1) << 2) as u32;
            p0 = sub8((w(b) >> sh) & 0x0F0F_0F0F, 8);
            p1 = sub8((w(b + 1) >> sh) & 0x0F0F_0F0F, 8);
            ws = f16at(buf, off1, g);
            wm = 0.0;
        }
        "q4_1" => {
            let b = g * 4 + ((cin & 1) << 1);
            let sh = ((cin >> 1) << 2) as u32;
            p0 = (w(b) >> sh) & 0x0F0F_0F0F;
            p1 = (w(b + 1) >> sh) & 0x0F0F_0F0F;
            (ws, wm) = half2(w(off1 as usize + g));
        }
        "q5_0" => {
            let b = g * 4 + ((cin & 1) << 1);
            let sh = ((cin >> 1) << 2) as u32;
            let qh = w(off1 as usize + g) >> (cin << 3);
            let (h0, h1) = spread(qh);
            p0 = sub8(((w(b) >> sh) & 0x0F0F_0F0F) | (h0 << 4), 16);
            p1 = sub8(((w(b + 1) >> sh) & 0x0F0F_0F0F) | (h1 << 4), 16);
            ws = f16at(buf, off2, g);
            wm = 0.0;
        }
        "q5_1" => {
            let b = g * 4 + ((cin & 1) << 1);
            let sh = ((cin >> 1) << 2) as u32;
            let qh = w(off1 as usize + g) >> (cin << 3);
            let (h0, h1) = spread(qh);
            p0 = ((w(b) >> sh) & 0x0F0F_0F0F) | (h0 << 4);
            p1 = ((w(b + 1) >> sh) & 0x0F0F_0F0F) | (h1 << 4);
            (ws, wm) = half2(w(off2 as usize + g));
        }
        "q8_0" => {
            let b = g * 8 + (cin << 1);
            p0 = w(b);
            p1 = w(b + 1);
            ws = f16at(buf, off1, g);
            wm = 0.0;
        }
        "iq4_nl" => {
            let b = g * 4 + ((cin & 1) << 1);
            let sh = ((cin >> 1) << 2) as u32;
            p0 = lut(&KV_IQ4NL, (w(b) >> sh) & 0x0F0F_0F0F);
            p1 = lut(&KV_IQ4NL, (w(b + 1) >> sh) & 0x0F0F_0F0F);
            ws = f16at(buf, off1, g);
            wm = 0.0;
        }
        "mxfp4" => {
            let b = g * 4 + ((cin & 1) << 1);
            let sh = ((cin >> 1) << 2) as u32;
            p0 = lut(&KV_FP4, (w(b) >> sh) & 0x0F0F_0F0F);
            p1 = lut(&KV_FP4, (w(b + 1) >> sh) & 0x0F0F_0F0F);
            ws = e8m0_half((w(off1 as usize + (g >> 2)) >> ((g & 3) << 3)) & 255);
            wm = 0.0;
        }
        "nvfp4" => {
            let s = cin >> 1;
            let b = g * 8 + (s << 1);
            let sh = ((cin & 1) << 2) as u32;
            p0 = lut(&KV_FP4, (w(b) >> sh) & 0x0F0F_0F0F);
            p1 = lut(&KV_FP4, (w(b + 1) >> sh) & 0x0F0F_0F0F);
            ws = ue4m3_half((w(off1 as usize + g) >> (s << 3)) & 255);
            wm = 0.0;
        }
        "q4z32" | "q4z64" | "q4z128" => {
            let v = w(g * (layout.info.qk / 8) + cin);
            let sz = w(off1 as usize + g);
            let zero = (sz >> 16) & 15;
            p0 = sub8(v & 0x0F0F_0F0F, zero);
            p1 = sub8((v >> 4) & 0x0F0F_0F0F, zero);
            ws = half2(sz).0;
            wm = 0.0;
        }
        "q2_k" => {
            let n = cin >> 4;
            let r = cin & 15;
            let sub = r >> 2;
            let j4 = r & 3;
            let b = g * 16 + (n << 3) + (j4 << 1);
            let sh = (sub << 1) as u32;
            p0 = (w(b) >> sh) & 0x0303_0303;
            p1 = (w(b + 1) >> sh) & 0x0303_0303;
            let si = (n << 3) + (sub << 1) + (j4 >> 1);
            let sc = (w(off1 as usize + g * 4 + (si >> 2)) >> ((si & 3) << 3)) & 255;
            let dm = half2(w(off2 as usize + g));
            ws = dm.0 * (sc & 15) as f32;
            wm = -dm.1 * (sc >> 4) as f32;
        }
        "q3_k" => {
            let outer = cin >> 4;
            let r = cin & 15;
            let sub = r >> 2;
            let j4 = r & 3;
            let b = g * 16 + (outer << 3) + (j4 << 1);
            let sh = (sub << 1) as u32;
            let hb = ((outer << 2) + sub) as u32;
            let hm = g * 8 + (j4 << 1);
            p0 = sub8(
                ((w(b) >> sh) & 0x0303_0303) | (((w(off1 as usize + hm) >> hb) & 0x0101_0101) << 2),
                4,
            );
            p1 = sub8(
                ((w(b + 1) >> sh) & 0x0303_0303)
                    | (((w(off1 as usize + hm + 1) >> hb) & 0x0101_0101) << 2),
                4,
            );
            let si = (outer << 3) + (sub << 1) + (j4 >> 1);
            let sc = (w(off2 as usize + g * 4 + (si >> 2)) >> ((si & 3) << 3)) as u8 as i8;
            ws = f32::from_bits(w(off3 as usize + g)) * sc as f32;
            wm = 0.0;
        }
        "q4_k" => {
            let g8 = cin >> 2;
            let j4 = cin & 3;
            let b = g * 32 + ((g8 >> 1) << 3) + (j4 << 1);
            let sh = ((g8 & 1) << 2) as u32;
            p0 = (w(b) >> sh) & 0x0F0F_0F0F;
            p1 = (w(b + 1) >> sh) & 0x0F0F_0F0F;
            let q = off1 as usize + g * 4 + (g8 >> 2);
            let lane = ((g8 & 3) << 3) as u32;
            let sc = ((w(q) >> lane) & 255) as f32;
            let mn = ((w(q + 2) >> lane) & 255) as f32;
            let dm = half2(w(off2 as usize + g));
            ws = dm.0 * sc;
            wm = -dm.1 * mn;
        }
        "q5_k" => {
            let g8 = cin >> 2;
            let j4 = cin & 3;
            let b = g * 32 + ((g8 >> 1) << 3) + (j4 << 1);
            let sh = ((g8 & 1) << 2) as u32;
            let h = g * 8 + (j4 << 1);
            p0 = ((w(b) >> sh) & 0x0F0F_0F0F) | (((w(off1 as usize + h) >> g8) & 0x0101_0101) << 4);
            p1 = ((w(b + 1) >> sh) & 0x0F0F_0F0F)
                | (((w(off1 as usize + h + 1) >> g8) & 0x0101_0101) << 4);
            let q = off2 as usize + g * 4 + (g8 >> 2);
            let lane = ((g8 & 3) << 3) as u32;
            let sc = ((w(q) >> lane) & 255) as f32;
            let mn = ((w(q + 2) >> lane) & 255) as f32;
            let dm = half2(w(off3 as usize + g));
            ws = dm.0 * sc;
            wm = -dm.1 * mn;
        }
        "q6_k" => {
            let n = cin >> 4;
            let r = cin & 15;
            let grp = r >> 2;
            let j4 = r & 3;
            let b = g * 32 + (n << 4) + ((grp & 1) << 3) + (j4 << 1);
            let sh = ((grp >> 1) << 2) as u32;
            let h = g * 16 + (n << 3) + (j4 << 1);
            let hs = (grp << 1) as u32;
            p0 = sub8(
                ((w(b) >> sh) & 0x0F0F_0F0F) | (((w(off1 as usize + h) >> hs) & 0x0303_0303) << 4),
                32,
            );
            p1 = sub8(
                ((w(b + 1) >> sh) & 0x0F0F_0F0F)
                    | (((w(off1 as usize + h + 1) >> hs) & 0x0303_0303) << 4),
                32,
            );
            let si = (n << 3) + (j4 >> 1) + (grp << 1);
            let sc = (w(off2 as usize + g * 4 + (si >> 2)) >> ((si & 3) << 3)) as u8 as i8;
            ws = f32::from_bits(w(off3 as usize + g)) * sc as f32;
            wm = 0.0;
        }
        other => panic!("no mirror for {other}"),
    }
    let (a, b) = (lanes(p0), lanes(p1));
    Chunk {
        q: [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]],
        ws,
        wm,
    }
}
