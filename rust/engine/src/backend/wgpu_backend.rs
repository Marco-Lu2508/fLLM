//! wgpu accelerator backend for fLLM.
//!
//! Resident device weights, command streams and WGSL compute kernels for linear projections
//! (F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K), embedding lookup, RMSNorm, RoPE,
//! SwiGLU/GeGLU, residual addition, KV-cache writes and paged attention.
//!
//! Conventions every kernel shares:
//! - Workgroups have 64 lanes. Per-row kernels number their workgroups `x + y * nx` so a
//!   dispatch is never limited to 65535 rows; element-wise kernels loop with a grid stride.
//! - Parameters live in one per-stream arena buffer. Binding 0 is this dispatch's 32-word
//!   header (dynamic offset, aligned to `min_storage_buffer_offset_alignment`), binding 1 the
//!   whole arena; header fields named `*_base` index variable-length data (positions, slots,
//!   lengths, block tables, sinks) inside it. The arena is written with one `write_buffer`
//!   per submit, so a decode step creates no uniform buffers and, after its first run, no
//!   bind groups (they are cached per buffer window).
//! - Weights and KV pools are bound as `array<u32>` and decoded in-kernel; their byte layout
//!   is the GGUF block layout and the KV layout documented on [`DeviceKvCache`].

use super::*;
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use wgpu::util::DeviceExt;

static WGPU_BUFFER_ID: AtomicU64 = AtomicU64::new(1);

/// u32 words of a dispatch header (binding 0).
const HEADER_WORDS: usize = 32;
/// Initial parameter arena; it grows when one dispatch needs more (long block tables).
const ARENA_WORDS: usize = 1 << 18;
/// Bind groups cached per stream before the cache is reset.
const MAX_BIND_GROUPS: usize = 4096;
/// Widest key or value head the attention kernel holds in workgroup memory / registers.
const MAX_ATTENTION_DIM: usize = 512;
/// GGUF types the linear and embedding kernels decode natively. Other types are dequantized
/// to F32 once at upload.
const NATIVE_KINDS: [u32; 11] = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 30];
/// Marker the upload puts into `hdr[3]` for block-wise FP8 (GGUF `FP8B` types) that the WGSL decoder
/// reads directly: high nibble 0xF8, then log2(block_rows) and log2(block_cols) in the low byte.
/// No GGUF type id reaches that pattern (the `FP8B` ids carry 0x8000 in the upper half).
const FP8B_NATIVE: u32 = 0xF800_0000;
/// Marker for bitsandbytes tensors the WGSL decoder reads in the file layout: high byte 0xF4, then
/// the family (0 NF4, 1 FP4, 2 INT8) in bits 5-6 and log2(values per block) in bits 0-4.
const BNB_NATIVE: u32 = 0xF400_0000;
/// "Absent" marker for optional arena arrays.
const NONE: u32 = u32::MAX;

/// Header and arena bindings shared by every kernel.
const PRELUDE: &str = r#"
@group(0) @binding(0) var<storage, read> hdr: array<u32, 32>;
@group(0) @binding(1) var<storage, read> arena: array<u32>;
const NONE: u32 = 0xffffffffu;
fn hdr_f32(i: u32) -> f32 { return bitcast<f32>(hdr[i]); }
"#;

/// Weight decoding (binding 2). Element `index` of a row-major `rows x cols` weight of GGUF
/// type `kind`; blocks hold 32 (Q4_0..Q8_0) or 256 (K-quants) consecutive elements of a row.
/// Byte layouts (ggml, little endian, `d`/`m` are f16):
///   Q4_0 18 B: d, 16 B nibbles (j < 16 low nibble of byte j, else high nibble of byte j-16)
///   Q4_1 20 B: d, m, nibbles             value = d * q + m
///   Q5_0 22 B: d, 4 B high bits, nibbles value = d * (q - 16)
///   Q5_1 24 B: d, m, 4 B high bits, nibbles
///   Q8_0 34 B: d, 32 signed bytes
///   Q4_K 144 B: d, dmin, 12 B packed 6-bit scales/mins, 128 B nibbles (two 32-groups per 32 B)
///   Q5_K 176 B: as Q4_K plus 32 B high bits before the nibbles
///   Q6_K 210 B: 128 B low nibbles, 64 B 2-bit highs, 16 signed scales, d at byte 208
/// Weight buffers are padded to a multiple of 4 bytes so the last word is always readable.
const WEIGHT_DECODE: &str = r#"
@group(0) @binding(2) var<storage, read> weights: array<u32>;
fn byte_at(offset: u32) -> u32 { return (weights[offset / 4u] >> ((offset % 4u) * 8u)) & 255u; }
fn half_at(offset: u32) -> f32 { return unpack2x16float(byte_at(offset) | (byte_at(offset + 1u) << 8u)).x; }
fn signed_byte(value: u32) -> f32 { return f32(i32(value << 24u) >> 24u); }
// E4M3FN code to float, exactly `decode_code` of cpu_fp8.rs: exponent and mantissa moved into an
// F32 (bias 127 instead of 7), subnormals are mantissa * 2^-9, 0x7f/0xff are NaN.
fn e4m3_value(code: u32) -> f32 {
    let mag = code & 0x7fu;
    let normal = bitcast<f32>((mag << 20u) + (120u << 23u));
    let subnormal = f32(mag & 7u) * (1.0 / 512.0);
    var value = select(normal, subnormal, (mag >> 3u) == 0u);
    if (mag == 0x7fu) { value = bitcast<f32>(0x7fc00000u); }
    return bitcast<f32>(bitcast<u32>(value) | ((code & 0x80u) << 24u));
}
// Block-wise FP8 (`FP8B`): `rows * cols` E4M3 codes row-major, then one F32 scale per (row, tile
// column) pair, tiles row-major; the upload passes log2(block_rows)/log2(block_cols) in `kind`
// under the FP8_NATIVE marker. A weight is `e4m3(code) * scale`, the `weight_scale_inv` convention.
fn fp8b_weight(index: u32, kind: u32) -> f32 {
    let rows = hdr[0]; let cols = hdr[1];
    let block_rows = 1u << ((kind >> 4u) & 15u);
    let block_cols = 1u << (kind & 15u);
    let row = index / cols; let col = index % cols;
    let tile_cols = (cols + block_cols - 1u) / block_cols;
    let tile = (row / block_rows) * tile_cols + col / block_cols;
    let base = rows * cols + 4u * tile;
    let bits = byte_at(base) | (byte_at(base + 1u) << 8u) | (byte_at(base + 2u) << 16u) | (byte_at(base + 3u) << 24u);
    return e4m3_value(byte_at(index)) * bitcast<f32>(bits);
}
// bitsandbytes (NF4 / FP4 / LLM.int8()) in the file layout: per block an f32 scale followed by the
// code bytes (block/2 packed nibbles, or block int8 codes). The upload passes the family
// (0 NF4, 1 FP4, 2 INT8) and log2(values per block) under the BNB_NATIVE marker.
const NF4_BOOK: array<u32, 16> = array<u32, 16>(0xbf800000u, 0xbf3239b1u, 0xbf066b30u, 0xbeca32a0u,
    0xbe91a24du, 0xbe3d353fu, 0xbdba7871u, 0x00000000u, 0x3da2faffu, 0x3e24cae3u, 0x3e7c04ddu,
    0x3ead033au, 0x3ee1a4b8u, 0x3f1007abu, 0x3f3913b3u, 0x3f800000u);
const FP4_BOOK: array<u32, 16> = array<u32, 16>(0x00000000u, 0x3baaaaabu, 0x3f2aaaabu, 0x3f800000u,
    0x3eaaaaabu, 0x3f000000u, 0x3e2aaaabu, 0x3e800000u, 0x80000000u, 0xbbaaaaabu, 0xbf2aaaabu,
    0xbf800000u, 0xbeaaaaabu, 0xbf000000u, 0xbe2aaaabu, 0xbe800000u);
fn bits_at(offset: u32) -> u32 {
    let bits = byte_at(offset) | (byte_at(offset + 1u) << 8u) | (byte_at(offset + 2u) << 16u) | (byte_at(offset + 3u) << 24u);
    return bits;
}
fn bnb_weight(index: u32, kind: u32) -> f32 {
    let family = (kind >> 5u) & 3u;
    let block = 1u << (kind & 31u);
    let local = index % block;
    let codes = select(block / 2u, block, family == 2u);
    let start = (index / block) * (4u + codes);
    let scale = bitcast<f32>(bits_at(start));
    if (family == 2u) {
        return signed_byte(byte_at(start + 4u + local)) * scale;
    }
    let byte = byte_at(start + 4u + local / 2u);
    let nib = select(byte & 15u, byte >> 4u, local % 2u == 0u);
    return bitcast<f32>(select(FP4_BOOK[nib], NF4_BOOK[nib], family == 0u)) * scale;
}
fn weight_at(index: u32, kind: u32) -> f32 {
    if ((kind & 0xf8000000u) == 0xf8000000u) { return fp8b_weight(index, kind); }
    if ((kind & 0xfc000000u) == 0xf4000000u) { return bnb_weight(index, kind); }
    if (kind == 0u) { return bitcast<f32>(weights[index]); }
    if (kind == 1u) { return unpack2x16float(weights[index / 2u])[index % 2u]; }
    if (kind == 30u) { return bitcast<f32>(((weights[index / 2u] >> ((index % 2u) * 16u)) & 0xffffu) << 16u); }
    if (kind < 8u) {
        let affine = kind == 3u || kind == 7u; let five = kind == 6u || kind == 7u;
        let offset = select(2u, 4u, affine); let qoffset = offset + select(0u, 4u, five);
        let size = qoffset + 16u; let start = (index / 32u) * size; let j = index % 32u;
        let byte = byte_at(start + qoffset + j % 16u); var q = (byte >> select(0u, 4u, j >= 16u)) & 15u;
        if (five) { q |= ((byte_at(start + offset + j / 8u) >> (j % 8u)) & 1u) << 4u; }
        if (affine) { return half_at(start) * f32(q) + half_at(start + 2u); }
        return half_at(start) * (f32(q) - select(8.0, 16.0, five));
    }
    if (kind == 8u) {
        let start = (index / 32u) * 34u;
        return half_at(start) * signed_byte(byte_at(start + 2u + index % 32u));
    }
    let local = index % 256u;
    if (kind == 12u || kind == 13u) {
        let five = kind == 13u; let start = (index / 256u) * select(144u, 176u, five); let group = local / 32u;
        var scale: u32; var minimum: u32;
        if (group < 4u) { scale = byte_at(start + 4u + group) & 63u; minimum = byte_at(start + 8u + group) & 63u; }
        else { scale = (byte_at(start + 8u + group) & 15u) | ((byte_at(start + group) >> 6u) << 4u);
            minimum = (byte_at(start + 8u + group) >> 4u) | ((byte_at(start + 4u + group) >> 6u) << 4u); }
        var q = (byte_at(start + select(16u, 48u, five) + (group / 2u) * 32u + local % 32u) >> ((group % 2u) * 4u)) & 15u;
        if (five) { q |= ((byte_at(start + 16u + local % 32u) >> group) & 1u) << 4u; }
        return half_at(start) * f32(scale) * f32(q) - half_at(start + 2u) * f32(minimum);
    }
    let start = (index / 256u) * 210u; let n = local / 128u; let group = (local % 128u) / 32u; let j = local % 32u;
    let low = (byte_at(start + n * 64u + (group % 2u) * 32u + j) >> ((group / 2u) * 4u)) & 15u;
    let high = (byte_at(start + 128u + n * 32u + j) >> (group * 2u)) & 3u;
    let scale = signed_byte(byte_at(start + 192u + n * 8u + group * 2u + j / 16u));
    return half_at(start + 208u) * scale * (f32(low | (high << 4u)) - 32.0);
}
"#;

/// hdr: 0 rows, 1 cols, 2 batch, 3 kind, 4 has_bias. One workgroup per output row and tile of
/// four batch rows (z), so each weight element is decoded once per four tokens.
const LINEAR: &str = r#"
@group(0) @binding(3) var<storage, read> input: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<storage, read> bias: array<f32>;
var<workgroup> partial: array<vec4<f32>, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    let rows = hdr[0]; let cols = hdr[1]; let batch = hdr[2]; let kind = hdr[3];
    let row = wid.x + wid.y * nwg.x;
    let b0 = wid.z * 4u;
    if (row >= rows || b0 >= batch) { return; }
    let nb = min(4u, batch - b0);
    var acc = vec4<f32>(0.0);
    for (var col = lane; col < cols; col += 64u) {
        let w = weight_at(row * cols + col, kind);
        let xi = b0 * cols + col;
        acc.x += w * input[xi];
        if (nb > 1u) { acc.y += w * input[xi + cols]; }
        if (nb > 2u) { acc.z += w * input[xi + 2u * cols]; }
        if (nb > 3u) { acc.w += w * input[xi + 3u * cols]; }
    }
    partial[lane] = acc;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s / 2u) {
        if (lane < s) { partial[lane] += partial[lane + s]; }
        workgroupBarrier();
    }
    if (lane == 0u) {
        var b = 0.0;
        if (hdr[4] != 0u) { b = bias[row]; }
        let total = partial[0];
        output[b0 * rows + row] = total.x + b;
        if (nb > 1u) { output[(b0 + 1u) * rows + row] = total.y + b; }
        if (nb > 2u) { output[(b0 + 2u) * rows + row] = total.z + b; }
        if (nb > 3u) { output[(b0 + 3u) * rows + row] = total.w + b; }
    }
}
"#;

/// Decode-sized linear for the 144-byte (Q4_K) and 176-byte (Q5_K) super-blocks, which are word aligned: eight lanes share
/// one super-block of 256 values (one 32-value group each), read it as whole 32-bit words and fold scale and minimum in once
/// per group: sum = d * scale * sum(q * x) - dmin * min * sum(x). Four tokens per workgroup like `LINEAR`.
/// hdr: 0 rows, 1 cols, 2 batch, 3 kind (12 or 13), 4 has_bias.
const LINEAR_KQ: &str = r#"
@group(0) @binding(3) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<storage, read> bias: array<f32>;
var<workgroup> partial: array<vec4<f32>, 64>;
fn scale_min(w1: u32, w2: u32, w3: u32, g: u32) -> vec2<f32> {
    // 12 bytes of 6-bit scales/mins: words w1 (bytes 0-3), w2 (4-7), w3 (8-11)
    var sc: u32; var mn: u32;
    if (g < 4u) {
        sc = (w1 >> (g * 8u)) & 63u;
        mn = (w2 >> (g * 8u)) & 63u;
    } else {
        let h = g - 4u;
        let lo = (w3 >> (h * 8u)) & 255u;
        sc = (lo & 15u) | (((w1 >> (h * 8u)) >> 6u) & 3u) << 4u;
        mn = (lo >> 4u) | (((w2 >> (h * 8u)) >> 6u) & 3u) << 4u;
    }
    return vec2<f32>(f32(sc), f32(mn));
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    let rows = hdr[0]; let cols = hdr[1]; let batch = hdr[2]; let kind = hdr[3];
    let row = wid.x + wid.y * nwg.x;
    let b0 = wid.z * 4u;
    if (row >= rows || b0 >= batch) { return; }
    let nb = min(4u, batch - b0);
    let five = kind == 13u;
    let block_words = select(36u, 44u, five);
    let sb_count = cols / 256u;
    let g = lane % 8u;
    let sb_lane = lane / 8u;
    let qs_word = select(4u, 12u, five);          // first word of the nibbles
    let cols4 = cols / 4u;
    var acc = vec4<f32>(0.0);
    for (var sb = sb_lane; sb < sb_count; sb += 8u) {
        let base = (row * sb_count + sb) * block_words;
        let dd = unpack2x16float(weights[base]);
        let sm = scale_min(weights[base + 1u], weights[base + 2u], weights[base + 3u], g);
        let wq = base + qs_word + (g / 2u) * 8u;     // 8 words = 32 bytes of this group pair
        let shift = (g % 2u) * 4u;
        let hq = base + 4u + 0u;                      // Q5_K high-bit bytes (32 B) follow the scales
        var qx = vec4<f32>(0.0);
        var sx = vec4<f32>(0.0);
        let x4 = (sb * 256u + g * 32u) / 4u;
        for (var k = 0u; k < 8u; k++) {
            let word = weights[wq + k];
            var q = vec4<f32>(f32((word >> shift) & 15u), f32((word >> (8u + shift)) & 15u),
                              f32((word >> (16u + shift)) & 15u), f32((word >> (24u + shift)) & 15u));
            if (five) {
                let hw = weights[hq + k];
                q += 16.0 * vec4<f32>(f32((hw >> g) & 1u), f32((hw >> (8u + g)) & 1u),
                                      f32((hw >> (16u + g)) & 1u), f32((hw >> (24u + g)) & 1u));
            }
            let xa = input[b0 * cols4 + x4 + k];
            qx.x += dot(q, xa); sx.x += xa.x + xa.y + xa.z + xa.w;
            if (nb > 1u) { let xb = input[(b0 + 1u) * cols4 + x4 + k]; qx.y += dot(q, xb); sx.y += xb.x + xb.y + xb.z + xb.w; }
            if (nb > 2u) { let xc = input[(b0 + 2u) * cols4 + x4 + k]; qx.z += dot(q, xc); sx.z += xc.x + xc.y + xc.z + xc.w; }
            if (nb > 3u) { let xd = input[(b0 + 3u) * cols4 + x4 + k]; qx.w += dot(q, xd); sx.w += xd.x + xd.y + xd.z + xd.w; }
        }
        acc += dd.x * sm.x * qx - dd.y * sm.y * sx;
    }
    partial[lane] = acc;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s / 2u) {
        if (lane < s) { partial[lane] += partial[lane + s]; }
        workgroupBarrier();
    }
    if (lane == 0u) {
        var b = 0.0;
        if (hdr[4] != 0u) { b = bias[row]; }
        let total = partial[0];
        output[b0 * rows + row] = total.x + b;
        if (nb > 1u) { output[(b0 + 1u) * rows + row] = total.y + b; }
        if (nb > 2u) { output[(b0 + 2u) * rows + row] = total.z + b; }
        if (nb > 3u) { output[(b0 + 3u) * rows + row] = total.w + b; }
    }
}
"#;

/// Decode-sized linear for Q6_K (210-byte super-blocks, only 2-byte aligned): eight lanes share a super-block, each takes one
/// 32-value group (`n` = 128-value half, `group` = 32-value quarter inside it) and reads its low nibbles, high bit pairs and
/// two scales as 32-bit words at arbitrary byte offsets (`u32_at`). value = d * scale * (q - 32).
/// hdr: 0 rows, 1 cols, 2 batch, 3 kind (14), 4 has_bias.
const LINEAR_Q6: &str = r#"
@group(0) @binding(3) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<storage, read> bias: array<f32>;
var<workgroup> partial: array<vec4<f32>, 64>;
fn u32_at(off: u32) -> u32 {
    let w = off >> 2u; let s = (off & 3u) * 8u;
    if (s == 0u) { return weights[w]; }
    return (weights[w] >> s) | (weights[w + 1u] << (32u - s));
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    let rows = hdr[0]; let cols = hdr[1]; let batch = hdr[2];
    let row = wid.x + wid.y * nwg.x;
    let b0 = wid.z * 4u;
    if (row >= rows || b0 >= batch) { return; }
    let nb = min(4u, batch - b0);
    let sb_count = cols / 256u;
    let n = (lane % 8u) / 4u;
    let group = lane % 4u;
    let sb_lane = lane / 8u;
    let cols4 = cols / 4u;
    var acc = vec4<f32>(0.0);
    for (var sb = sb_lane; sb < sb_count; sb += 8u) {
        let start = (row * sb_count + sb) * 210u;
        let d = unpack2x16float(u32_at(start + 208u) & 0xffffu).x;
        let sc = u32_at(start + 192u + n * 8u + group * 2u);     // bytes 0 and 1 are this group's two scales
        let s0 = f32(i32(sc << 24u) >> 24u);
        let s1 = f32(i32(sc << 16u) >> 24u);
        let ql = start + n * 64u + (group % 2u) * 32u;
        let qh = start + 128u + n * 32u;
        let lsh = (group / 2u) * 4u;
        let hsh = group * 2u;
        var q0 = vec4<f32>(0.0);
        var q1 = vec4<f32>(0.0);
        let x4 = (sb * 256u + n * 128u + group * 32u) / 4u;
        for (var k = 0u; k < 8u; k++) {
            let wl = u32_at(ql + k * 4u);
            let wh = u32_at(qh + k * 4u);
            let q = vec4<f32>(
                f32(((wl >> lsh) & 15u) | (((wh >> hsh) & 3u) << 4u)) - 32.0,
                f32(((wl >> (8u + lsh)) & 15u) | (((wh >> (8u + hsh)) & 3u) << 4u)) - 32.0,
                f32(((wl >> (16u + lsh)) & 15u) | (((wh >> (16u + hsh)) & 3u) << 4u)) - 32.0,
                f32(((wl >> (24u + lsh)) & 15u) | (((wh >> (24u + hsh)) & 3u) << 4u)) - 32.0);
            var t = vec4<f32>(dot(q, input[b0 * cols4 + x4 + k]), 0.0, 0.0, 0.0);
            if (nb > 1u) { t.y = dot(q, input[(b0 + 1u) * cols4 + x4 + k]); }
            if (nb > 2u) { t.z = dot(q, input[(b0 + 2u) * cols4 + x4 + k]); }
            if (nb > 3u) { t.w = dot(q, input[(b0 + 3u) * cols4 + x4 + k]); }
            if (k < 4u) { q0 += t; } else { q1 += t; }
        }
        acc += d * (s0 * q0 + s1 * q1);
    }
    partial[lane] = acc;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s / 2u) {
        if (lane < s) { partial[lane] += partial[lane + s]; }
        workgroupBarrier();
    }
    if (lane == 0u) {
        var b = 0.0;
        if (hdr[4] != 0u) { b = bias[row]; }
        let total = partial[0];
        output[b0 * rows + row] = total.x + b;
        if (nb > 1u) { output[(b0 + 1u) * rows + row] = total.y + b; }
        if (nb > 2u) { output[(b0 + 2u) * rows + row] = total.z + b; }
        if (nb > 3u) { output[(b0 + 3u) * rows + row] = total.w + b; }
    }
}
"#;

/// Prompt-sized linear: a 64 x 64 tile of the output (64 weight rows x 64 tokens) per workgroup of 16 x 16 threads, each
/// thread owning a 4 x 4 block. Per step of 32 values of k the workgroup decodes its weight tile and loads its activation
/// tile into workgroup memory once, so every weight element is decoded once per 64 tokens instead of once per 4.
/// hdr: 0 rows, 1 cols, 2 batch, 3 kind, 4 has_bias.
const LINEAR_TILED: &str = r#"
@group(0) @binding(3) var<storage, read> input: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<storage, read> bias: array<f32>;
const TILE: u32 = 64u;
const STEP: u32 = 32u;
const LD: u32 = 33u;
var<workgroup> wtile: array<f32, 2112>;
var<workgroup> xtile: array<f32, 2112>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let rows = hdr[0]; let cols = hdr[1]; let batch = hdr[2]; let kind = hdr[3];
    let r0 = wid.x * TILE; let t0 = wid.y * TILE;
    let tid = lid.y * 16u + lid.x;
    var acc0 = vec4<f32>(0.0); var acc1 = vec4<f32>(0.0); var acc2 = vec4<f32>(0.0); var acc3 = vec4<f32>(0.0);
    for (var k0 = 0u; k0 < cols; k0 += STEP) {
        for (var i = 0u; i < 8u; i++) {
            let idx = tid + i * 256u;
            let r = idx / STEP; let k = idx % STEP;
            let row = r0 + r; let kk = k0 + k;
            var w = 0.0;
            if (row < rows && kk < cols) { w = weight_at(row * cols + kk, kind); }
            wtile[r * LD + k] = w;
            let tok = t0 + r;
            var x = 0.0;
            if (tok < batch && kk < cols) { x = input[tok * cols + kk]; }
            xtile[r * LD + k] = x;
        }
        workgroupBarrier();
        for (var k = 0u; k < STEP; k++) {
            let w = vec4<f32>(wtile[(lid.y * 4u) * LD + k], wtile[(lid.y * 4u + 1u) * LD + k],
                              wtile[(lid.y * 4u + 2u) * LD + k], wtile[(lid.y * 4u + 3u) * LD + k]);
            let x = vec4<f32>(xtile[(lid.x * 4u) * LD + k], xtile[(lid.x * 4u + 1u) * LD + k],
                              xtile[(lid.x * 4u + 2u) * LD + k], xtile[(lid.x * 4u + 3u) * LD + k]);
            acc0 += w.x * x; acc1 += w.y * x; acc2 += w.z * x; acc3 += w.w * x;
        }
        workgroupBarrier();
    }
    // acc<i>[j]: weight row r0 + 4 * lid.y + i, token t0 + 4 * lid.x + j
    for (var i = 0u; i < 4u; i++) {
        let row = r0 + lid.y * 4u + i;
        if (row >= rows) { continue; }
        var b = 0.0;
        if (hdr[4] != 0u) { b = bias[row]; }
        var a = acc0;
        if (i == 1u) { a = acc1; } else if (i == 2u) { a = acc2; } else if (i == 3u) { a = acc3; }
        for (var j = 0u; j < 4u; j++) {
            let tok = t0 + lid.x * 4u + j;
            if (tok < batch) { output[tok * rows + row] = a[j] + b; }
        }
    }
}
"#;

/// hdr: 0 vocab rows, 1 cols, 2 token count, 3 kind, 4 ids_base.
const EMBEDDING: &str = r#"
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let cols = hdr[1]; let total = hdr[2] * cols;
    for (var i = gid.x; i < total; i += nwg.x * 64u) {
        let id = arena[hdr[4] + i / cols];
        output[i] = weight_at(id * cols + i % cols, hdr[3]);
    }
}
"#;

/// hdr: 0 rows, 1 cols, 2 eps (f32 bits). One workgroup per row.
const RMS_NORM: &str = r#"
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read> weight: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    let rows = hdr[0]; let cols = hdr[1];
    let r = wid.x + wid.y * nwg.x;
    if (r >= rows) { return; }
    var sum = 0.0;
    for (var c = lane; c < cols; c += 64u) {
        let v = input[r * cols + c];
        sum += v * v;
    }
    partial[lane] = sum;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s / 2u) {
        if (lane < s) { partial[lane] += partial[lane + s]; }
        workgroupBarrier();
    }
    let inv = 1.0 / sqrt(partial[0] / f32(cols) + hdr_f32(2u));
    for (var c = lane; c < cols; c += 64u) {
        output[r * cols + c] = input[r * cols + c] * (inv * weight[c]);
    }
}
"#;

/// erf as the decoder's host code computes it (Chebyshev fit of erfc, Numerical Recipes, fractional error below 1.2e-7).
const ERF: &str = r#"
fn erf_fit(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.5 * abs(x));
    let poly = -1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806
        + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))));
    let tau = t * exp(-x * x + poly);
    if (x >= 0.0) { return 1.0 - tau; }
    return tau - 1.0;
}
"#;

/// hdr: 0 count, 1 mode (0 SiLU, 1 GELU tanh approximation, 2 GELU erf, 3 ReLU, 4 ReLU squared), 2 plain (1: `up` is not
/// applied, `output = act(gate)`). tanh's argument is clamped to +-15, where it is 1 in f32 already, so drivers computing
/// tanh via exp cannot return NaN.
const GLU: &str = r#"
@group(0) @binding(2) var<storage, read> gate: array<f32>;
@group(0) @binding(3) var<storage, read> up: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let count = hdr[0];
    for (var i = gid.x; i < count; i += nwg.x * 64u) {
        let g = gate[i];
        var a: f32;
        switch (hdr[1]) {
            case 0u: { a = g / (1.0 + exp(-g)); }
            case 1u: { a = 0.5 * g * (1.0 + tanh(clamp(0.7978846 * (g + 0.044715 * g * g * g), -15.0, 15.0))); }
            case 2u: { a = 0.5 * g * (1.0 + erf_fit(g * 0.70710678)); }
            case 3u: { a = max(g, 0.0); }
            default: { let r = max(g, 0.0); a = r * r; }
        }
        if (hdr[2] == 0u) { a = a * up[i]; }
        output[i] = a;
    }
}
"#;

/// hdr: 0 count, 1 mode (0: `x = clamp(x, -limit, limit)`, 1: `x *= softplus(beta * gate) / beta`), 2 parameter (f32 bits).
const EWISE: &str = r#"
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@group(0) @binding(3) var<storage, read> gate: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let count = hdr[0]; let param = hdr_f32(2u);
    for (var i = gid.x; i < count; i += nwg.x * 64u) {
        if (hdr[1] == 0u) {
            x[i] = clamp(x[i], -param, param);
        } else {
            let scaled = param * gate[i];
            var factor = gate[i];
            if (scaled <= 20.0) { factor = log(1.0 + exp(scaled)) / param; }
            x[i] = x[i] * factor;
        }
    }
}
"#;

/// hdr: 0 count, 1 mode, 2 rows, 3 heads, 4 nope, 5 rope, 6 value, 7 kv_lora. One invocation per output element. Modes:
/// 0 queries `[rotary | content]` from a = q_full; 1 keys `[shared rotary | content]` from a = kv_a, b = kv;
/// 2 values from a = kv; 3 latent from a = kv_a. (Layouts as `mla_qkv` of the decoder.)
const MLA_MOVE: &str = r#"
@group(0) @binding(2) var<storage, read> src_a: array<f32>;
@group(0) @binding(3) var<storage, read> src_b: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let count = hdr[0]; let mode = hdr[1];
    let heads = hdr[3]; let nope = hdr[4]; let rope = hdr[5]; let value = hdr[6]; let kv_lora = hdr[7];
    let dk = nope + rope;
    for (var i = gid.x; i < count; i += nwg.x * 64u) {
        if (mode == 0u) {
            let at = i / dk; let d = i % dk;
            if (d < rope) { dst[i] = src_a[at * dk + nope + d]; } else { dst[i] = src_a[at * dk + d - rope]; }
        } else if (mode == 1u) {
            let at = i / dk; let d = i % dk;
            if (d < rope) { dst[i] = src_a[(at / heads) * (kv_lora + rope) + kv_lora + d]; }
            else { dst[i] = src_b[at * (nope + value) + d - rope]; }
        } else if (mode == 2u) {
            let at = i / value; let e = i % value;
            dst[i] = src_a[at * (nope + value) + nope + e];
        } else {
            let r = i / kv_lora; let c = i % kv_lora;
            dst[i] = src_a[r * (kv_lora + rope) + c];
        }
    }
}
"#;

/// hdr: 0 rows, 1 cols, 2 eps (f32 bits), 3 has_bias. One workgroup per row; mean and variance in two passes (biased variance,
/// eps inside the square root, like the CPU backend).
const LAYER_NORM: &str = r#"
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read> weight: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
var<workgroup> partial: array<f32, 64>;
fn reduce_sum(lane: u32, value: f32) -> f32 {
    partial[lane] = value;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s / 2u) {
        if (lane < s) { partial[lane] += partial[lane + s]; }
        workgroupBarrier();
    }
    let total = partial[0];
    workgroupBarrier();
    return total;
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    let rows = hdr[0]; let cols = hdr[1];
    let r = wid.x + wid.y * nwg.x;
    if (r >= rows) { return; }
    var sum = 0.0;
    for (var c = lane; c < cols; c += 64u) { sum += input[r * cols + c]; }
    let mean = reduce_sum(lane, sum) / f32(cols);
    var sq = 0.0;
    for (var c = lane; c < cols; c += 64u) {
        let d = input[r * cols + c] - mean;
        sq += d * d;
    }
    let inv = 1.0 / sqrt(reduce_sum(lane, sq) / f32(cols) + hdr_f32(2u));
    for (var c = lane; c < cols; c += 64u) {
        var v = (input[r * cols + c] - mean) * inv * weight[c];
        if (hdr[3] != 0u) { v += bias[c]; }
        output[r * cols + c] = v;
    }
}
"#;

/// hdr: 0 count, 1 scale (f32 bits). `x += residual * scale`.
const RESIDUAL: &str = r#"
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@group(0) @binding(3) var<storage, read> residual: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let count = hdr[0]; let scale = hdr_f32(1u);
    for (var i = gid.x; i < count; i += nwg.x * 64u) {
        x[i] = x[i] + residual[i] * scale;
    }
}
"#;

/// hdr: 0 rows, 1 heads, 2 kv_heads, 3 dim, 4 half (rope_dim / 2), 5 neox, 6 positions_base,
/// 7 sin offset. `table` is `[cos | sin]`, each `positions x half`. One invocation rotates one
/// pair: (d, d + half) for NeoX, (2d, 2d + 1) otherwise; query heads first, then key heads.
const ROPE: &str = r#"
@group(0) @binding(2) var<storage, read_write> q: array<f32>;
@group(0) @binding(3) var<storage, read_write> k: array<f32>;
@group(0) @binding(4) var<storage, read> table: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let heads = hdr[1]; let kvh = hdr[2]; let dim = hdr[3]; let hf = hdr[4];
    let per_row = (heads + kvh) * hf;
    let total = hdr[0] * per_row;
    for (var i = gid.x; i < total; i += nwg.x * 64u) {
        let r = i / per_row; let h = (i % per_row) / hf; let d = i % hf;
        let pos = arena[hdr[6] + r];
        let c = table[pos * hf + d];
        let s = table[hdr[7] + pos * hf + d];
        var ia = d; var ib = d + hf;
        if (hdr[5] == 0u) { ia = 2u * d; ib = 2u * d + 1u; }
        if (h < heads) {
            let base = (r * heads + h) * dim;
            let a = q[base + ia]; let b = q[base + ib];
            q[base + ia] = a * c - b * s;
            q[base + ib] = b * c + a * s;
        } else {
            let base = (r * kvh + h - heads) * dim;
            let a = k[base + ia]; let b = k[base + ib];
            k[base + ia] = a * c - b * s;
            k[base + ib] = b * c + a * s;
        }
    }
}
"#;

/// hdr: 0 rows, 1 kv (values per slot), 2 kind (1 f16, 2 q8), 3 group, 4 layer base (16-bit
/// word inside the bound window), 5 scale base (16-bit words from the layer start), 6
/// slots_base. Neighbouring slots and scales share 32-bit words (a layer may even start at an
/// odd 16-bit word), so partial words are merged with atomic and/or; a word owned entirely by
/// one invocation is stored directly.
/// q8 matches `encode_q8_cache_group`: d = amax / 127, scale = f16(d), q = round half away
/// from zero of x / d, clamped to +-127. WGSL division is not correctly rounded (2.5 ULP), so
/// a value exactly between two steps may round to the other neighbour.
const KV_WRITE: &str = r#"
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read_write> pool: array<atomic<u32>>;
fn store_bits(word: u32, mask: u32, value: u32) {
    if (mask == 0xffffffffu) {
        atomicStore(&pool[word], value);
    } else {
        atomicAnd(&pool[word], ~mask);
        atomicOr(&pool[word], value & mask);
    }
}
fn store_half(index: u32, bits: u32) {
    let shift = (index % 2u) * 16u;
    store_bits(index / 2u, 0xffffu << shift, (bits & 0xffffu) << shift);
}
fn f16_bits(x: f32) -> u32 { return pack2x16float(vec2<f32>(x, 0.0)) & 0xffffu; }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let rows = hdr[0]; let kv = hdr[1]; let base = hdr[4];
    let stride_n = nwg.x * 64u;
    if (hdr[2] == 1u) {
        let total = rows * kv;
        for (var i = gid.x; i < total; i += stride_n) {
            let slot = arena[hdr[6] + i / kv];
            store_half(base + slot * kv + i % kv, f16_bits(src[i]));
        }
        return;
    }
    let group = hdr[3]; let groups = kv / group;
    let total = rows * groups;
    for (var i = gid.x; i < total; i += stride_n) {
        let r = i / groups; let g = i % groups;
        let slot = arena[hdr[6] + r];
        let first = r * kv + g * group;
        var amax = 0.0;
        for (var j = 0u; j < group; j += 1u) { amax = max(amax, abs(src[first + j])); }
        let d = amax / 127.0;
        store_half(base + hdr[5] + slot * groups + g, f16_bits(d));
        let byte0 = base * 2u + slot * kv + g * group;
        var word = byte0 / 4u; var mask = 0u; var value = 0u;
        for (var j = 0u; j < group; j += 1u) {
            let b = byte0 + j;
            if (b / 4u != word) {
                store_bits(word, mask, value);
                word = b / 4u; mask = 0u; value = 0u;
            }
            var q = 0.0;
            if (d != 0.0) {
                let t = src[first + j] / d;
                q = clamp(sign(t) * floor(abs(t) + 0.5), -127.0, 127.0);
            }
            let shift = (b % 4u) * 8u;
            mask |= 0xffu << shift;
            value |= (u32(i32(q)) & 0xffu) << shift;
        }
        store_bits(word, mask, value);
    }
}
"#;

/// hdr: 0 rows, 1 heads, 2 kv_heads, 3 dk, 4 dv, 5 stride, 6 block_size, 7 scale (f32),
/// 8 softcap (f32, 0 = off), 9 window (0 = off), 10 chunk (0 = off), 11 kind (1 f16, 2 q8),
/// 12/13 key/value layer base (16-bit word in the bound window), 14/15 key/value scale base
/// (16-bit words from the layer start), 16/17 key/value q8 group, 18 lengths_base,
/// 19 tables_base, 20 sinks_base, 21 alibi_base (NONE = absent).
/// One workgroup per (row, head). Keys are scored 64 at a time (one per lane) and folded into
/// an online softmax; each lane accumulates value dims lane, lane + 64, ... (dv <= 512). The
/// running maximum starts at a finite floor instead of -inf so `exp(m - m_new)` never sees
/// inf - inf; with a sink it starts at the sink logit and the denominator at exp(0) = 1.
const ATTENTION: &str = r#"
@group(0) @binding(2) var<storage, read> q: array<f32>;
@group(0) @binding(3) var<storage, read> kpool: array<u32>;
@group(0) @binding(4) var<storage, read> vpool: array<u32>;
@group(0) @binding(5) var<storage, read_write> dst: array<f32>;
const FLOOR: f32 = -3.0e38;
var<workgroup> qs: array<f32, 512>;
var<workgroup> probs: array<f32, 64>;
var<workgroup> slots: array<u32, 64>;
fn k_half(i: u32) -> f32 { return unpack2x16float(kpool[i / 2u])[i % 2u]; }
fn v_half(i: u32) -> f32 { return unpack2x16float(vpool[i / 2u])[i % 2u]; }
fn k_byte(b: u32) -> f32 { return f32(i32(((kpool[b / 4u] >> ((b % 4u) * 8u)) & 255u) << 24u) >> 24u); }
fn v_byte(b: u32) -> f32 { return f32(i32(((vpool[b / 4u] >> ((b % 4u) * 8u)) & 255u) << 24u) >> 24u); }
// dot(q, key) for kv head `kh` of `slot`; q8 groups are summed before applying their scale.
fn key_dot(slot: u32, kh: u32) -> f32 {
    let dk = hdr[3]; let kv = hdr[2] * dk; let c0 = kh * dk;
    var s = 0.0;
    if (hdr[11] == 1u) {
        let w0 = hdr[12] + slot * kv + c0;
        for (var d = 0u; d < dk; d += 1u) { s += qs[d] * k_half(w0 + d); }
        return s;
    }
    let group = hdr[16];
    let b0 = hdr[12] * 2u + slot * kv + c0;
    let sc0 = hdr[12] + hdr[14] + slot * (kv / group) + c0 / group;
    for (var g = 0u; g < dk / group; g += 1u) {
        var part = 0.0;
        for (var j = 0u; j < group; j += 1u) {
            let d = g * group + j;
            part += qs[d] * k_byte(b0 + d);
        }
        s += part * k_half(sc0 + g);
    }
    return s;
}
// Value channel `c` (`kv_head * dv + d`) of `slot`.
fn value_at(slot: u32, c: u32) -> f32 {
    let kv = hdr[2] * hdr[4];
    if (hdr[11] == 1u) { return v_half(hdr[13] + slot * kv + c); }
    let group = hdr[17];
    return v_byte(hdr[13] * 2u + slot * kv + c) * v_half(hdr[13] + hdr[15] + slot * (kv / group) + c / group);
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    let heads = hdr[1];
    let item = wid.x + wid.y * nwg.x;
    if (item >= hdr[0] * heads) { return; }
    let r = item / heads; let h = item % heads;
    let kh = h / (heads / hdr[2]);
    let dk = hdr[3]; let dv = hdr[4]; let bs = hdr[6];
    for (var d = lane; d < dk; d += 64u) { qs[d] = q[item * dk + d]; }
    let n = arena[hdr[18] + r];
    var start = 0u;
    if (hdr[10] != 0u) {
        start = (n - 1u) / hdr[10] * hdr[10];
    } else if (hdr[9] != 0u && n > hdr[9]) {
        start = n - hdr[9];
    }
    var m = FLOOR;
    var l = 0.0;
    if (hdr[20] != NONE) { m = bitcast<f32>(arena[hdr[20] + h]); l = 1.0; }
    var acc: array<f32, 8>;
    let scale = hdr_f32(7u); let cap = hdr_f32(8u);
    let table = hdr[19] + r * hdr[5];
    workgroupBarrier();
    for (var t0 = start; t0 < n; t0 += 64u) {
        let p = t0 + lane;
        var s = FLOOR;
        if (p < n) {
            let slot = arena[table + p / bs] * bs + p % bs;
            slots[lane] = slot;
            s = key_dot(slot, kh) * scale;
            if (hdr[21] != NONE) { s += bitcast<f32>(arena[hdr[21] + h]) * (f32(p) - f32(n - 1u)); }
            if (cap != 0.0) { s = cap * tanh(clamp(s / cap, -15.0, 15.0)); }
        }
        probs[lane] = s;
        workgroupBarrier();
        let count = min(64u, n - t0);
        var tile_max = FLOOR;
        for (var j = 0u; j < count; j += 1u) { tile_max = max(tile_max, probs[j]); }
        let m_new = max(m, tile_max);
        let correction = exp(m - m_new);
        workgroupBarrier();
        var e = 0.0;
        if (p < n) { e = exp(s - m_new); }
        probs[lane] = e;
        workgroupBarrier();
        var tile_sum = 0.0;
        for (var j = 0u; j < count; j += 1u) { tile_sum += probs[j]; }
        l = l * correction + tile_sum;
        for (var i = 0u; i < 8u; i += 1u) {
            let d = lane + i * 64u;
            if (d < dv) {
                var a = acc[i] * correction;
                for (var j = 0u; j < count; j += 1u) { a += probs[j] * value_at(slots[j], kh * dv + d); }
                acc[i] = a;
            }
        }
        m = m_new;
        workgroupBarrier();
    }
    let inv = 1.0 / l;
    for (var i = 0u; i < 8u; i += 1u) {
        let d = lane + i * 64u;
        if (d < dv) { dst[item * dv + d] = acc[i] * inv; }
    }
}
"#;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum KernelId {
    Linear,
    Embedding,
    RmsNorm,
    Glu,
    Residual,
    Rope,
    KvWrite,
    Attention,
    LinearTiled,
    LinearKq,
    LinearQ6,
    LayerNorm,
    Ewise,
    MlaMove,
}

impl KernelId {
    const ALL: [KernelId; 14] = [
        KernelId::Linear,
        KernelId::Embedding,
        KernelId::RmsNorm,
        KernelId::Glu,
        KernelId::Residual,
        KernelId::Rope,
        KernelId::KvWrite,
        KernelId::Attention,
        KernelId::LinearTiled,
        KernelId::LinearKq,
        KernelId::LinearQ6,
        KernelId::LayerNorm,
        KernelId::Ewise,
        KernelId::MlaMove,
    ];

    fn label(self) -> &'static str {
        match self {
            KernelId::Linear => "fllm-linear",
            KernelId::Embedding => "fllm-embedding",
            KernelId::RmsNorm => "fllm-rms-norm",
            KernelId::Glu => "fllm-glu",
            KernelId::Residual => "fllm-residual",
            KernelId::Rope => "fllm-rope",
            KernelId::KvWrite => "fllm-kv-write",
            KernelId::Attention => "fllm-attention",
            KernelId::LinearTiled => "fllm-linear-tiled",
            KernelId::LinearKq => "fllm-linear-kq",
            KernelId::LinearQ6 => "fllm-linear-q6",
            KernelId::LayerNorm => "fllm-layer-norm",
            KernelId::Ewise => "fllm-ewise",
            KernelId::MlaMove => "fllm-mla-move",
        }
    }

    /// Complete WGSL module.
    fn source(self) -> String {
        let body = match self {
            KernelId::Linear => [WEIGHT_DECODE, LINEAR].concat(),
            KernelId::Embedding => [WEIGHT_DECODE, EMBEDDING].concat(),
            KernelId::RmsNorm => RMS_NORM.to_string(),
            KernelId::Glu => [ERF, GLU].concat(),
            KernelId::LayerNorm => LAYER_NORM.to_string(),
            KernelId::Ewise => EWISE.to_string(),
            KernelId::MlaMove => MLA_MOVE.to_string(),
            KernelId::Residual => RESIDUAL.to_string(),
            KernelId::Rope => ROPE.to_string(),
            KernelId::KvWrite => KV_WRITE.to_string(),
            KernelId::Attention => ATTENTION.to_string(),
            KernelId::LinearTiled => [WEIGHT_DECODE, LINEAR_TILED].concat(),
            KernelId::LinearKq => [WEIGHT_DECODE, LINEAR_KQ].concat(),
            KernelId::LinearQ6 => [WEIGHT_DECODE, LINEAR_Q6].concat(),
        };
        format!("{PRELUDE}{body}")
    }

    /// Bindings 2.. in order; `true` marks a read-write storage buffer.
    fn buffers(self) -> &'static [bool] {
        match self {
            KernelId::Linear => &[false, false, true, false],
            KernelId::Embedding => &[false, true],
            KernelId::RmsNorm => &[false, false, true],
            KernelId::Glu => &[false, false, true],
            KernelId::Residual => &[true, false],
            KernelId::Rope => &[true, true, false],
            KernelId::KvWrite => &[false, true],
            KernelId::Attention => &[false, false, false, true],
            KernelId::LinearTiled => &[false, false, true, false],
            KernelId::LinearKq => &[false, false, true, false],
            KernelId::LinearQ6 => &[false, false, true, false],
            KernelId::LayerNorm => &[false, false, false, true],
            KernelId::Ewise => &[true, false],
            KernelId::MlaMove => &[false, false, true],
        }
    }
}

struct Kernel {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

/// Raw GPU context wrapping device, queue, and compiled compute pipelines.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    kernels: Vec<Kernel>,
    /// Bound in place of an absent optional buffer (linear without bias).
    dummy: wgpu::Buffer,
    dummy_id: u64,
    pub limits: wgpu::Limits,
    /// First uncaptured device error (validation, out of memory); reported by the next
    /// synchronising call instead of wgpu's default handler, which panics.
    error: Arc<Mutex<Option<String>>>,
    pub name: String,
    pub device_id: usize,
    /// The adapter is a software rasteriser (lavapipe, WARP), admitted by FLLM_ALLOW_SOFTWARE_GPU.
    pub software: bool,
}

fn driver(msg: impl std::fmt::Display) -> BackendError {
    BackendError::DriverError(msg.to_string())
}

fn invalid(msg: impl Into<String>) -> BackendError {
    BackendError::InvalidShape(msg.into())
}

/// The `hdr[3]` marker for a block-wise FP8 tensor the WGSL decoder can read directly, or `None` when
/// the upload has to expand it (several slabs / a trailer that does not match the bytes).
/// The `hdr[3]` marker for a bitsandbytes tensor the WGSL decoder reads in the file layout, or `None`
/// when the block grid does not fit the shape (then the upload expands it to F32 as before).
fn bnb_native_kind(kind: u32, raw_bytes: &[u8], rows: usize, cols: usize) -> Option<u32> {
    let layout = fllm_gguf_probe::bnb::layout(kind)?;
    if cols % layout.block != 0 || rows * (cols / layout.block) * layout.bytes != raw_bytes.len() {
        return None;
    }
    let family = match layout.family {
        fllm_gguf_probe::bnb::Family::Nf4 => 0u32,
        fllm_gguf_probe::bnb::Family::Fp4 => 1,
        fllm_gguf_probe::bnb::Family::Int8 => 2,
    };
    Some(BNB_NATIVE | (family << 5) | layout.block.trailing_zeros())
}

fn fp8b_native_kind(raw_bytes: &[u8], kind: u32) -> Option<u32> {
    let layout = fllm_gguf_probe::Fp8bLayout::parse(kind, raw_bytes).ok()?;
    if layout.slabs != 1 {
        return None; // the kernels index one matrix; expert stacks keep the exact expansion
    }
    let (lr, lc) = (layout.block_rows.trailing_zeros(), layout.block_cols.trailing_zeros());
    (lr <= 15 && lc <= 15).then_some(FP8B_NATIVE | (lr << 4) | lc)
}

fn round4(n: u64) -> u64 {
    n.div_ceil(4) * 4
}

/// Environment switch that admits a software renderer (lavapipe, SwiftShader, WARP) as a GPU.
const SOFTWARE_ENV: &str = "FLLM_ALLOW_SOFTWARE_GPU";

/// What kind of adapter wgpu found; software renderers are refused unless [`SOFTWARE_ENV`] is set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AdapterClass {
    Discrete,
    Integrated,
    Virtual,
    Software,
}

impl AdapterClass {
    fn of(kind: wgpu::DeviceType) -> Self {
        match kind {
            wgpu::DeviceType::DiscreteGpu => Self::Discrete,
            wgpu::DeviceType::IntegratedGpu => Self::Integrated,
            wgpu::DeviceType::VirtualGpu => Self::Virtual,
            wgpu::DeviceType::Cpu | wgpu::DeviceType::Other => Self::Software,
        }
    }
}

/// Which of `adapters` (all adapters of the requested API, in wgpu's order) `<api>:index` means. Index 0 is the first discrete GPU, else the
/// first hardware adapter; index N is the N-th hardware adapter in the order `fllm-rs devices` lists them. Software renderers (WARP on
/// Windows, lavapipe and SwiftShader on Vulkan) are used only when no hardware adapter exists and `allow_software` is set. The errors say
/// what was found and what to do about it.
pub(crate) fn choose_adapter(api: &str, adapters: &[(AdapterClass, String)], index: usize, allow_software: bool) -> Result<usize, String> {
    let hardware: Vec<usize> = (0..adapters.len()).filter(|&i| adapters[i].0 != AdapterClass::Software).collect();
    let software: Vec<usize> = (0..adapters.len()).filter(|&i| adapters[i].0 == AdapterClass::Software).collect();
    let api_label = if matches!(api, "gpu" | "wgpu" | "auto") { "GPU".to_owned() } else { format!("{api} GPU") };
    if hardware.is_empty() {
        if software.is_empty() {
            let hint = if cfg!(windows) {
                " (install the graphics driver of the GPU vendor; a remote-desktop session or a virtual machine without GPU passthrough only offers the software renderer)"
            } else {
                " (is a graphics driver / Vulkan ICD installed?)"
            };
            return Err(format!("no {api_label} adapter found{hint}"));
        }
        let names: Vec<&str> = software.iter().map(|&i| adapters[i].1.as_str()).collect();
        if allow_software && index == 0 {
            return Ok(software[0]);
        }
        return Err(format!(
            "no hardware {api_label} adapter found; only software renderers are present ({}), which are refused because they are far too slow              for inference; set {SOFTWARE_ENV}=1 to use one anyway (testing), or run on the CPU with --device cpu",
            names.join(", ")
        ));
    }
    if index == 0 {
        return Ok(hardware.iter().copied().find(|&i| adapters[i].0 == AdapterClass::Discrete).unwrap_or(hardware[0]));
    }
    hardware.get(index).copied().ok_or_else(|| format!("no {api_label} adapter number {index} ({} hardware adapters found)", hardware.len()))
}

impl GpuContext {
    pub fn new(backend_name: &str, device_id: usize) -> Result<Arc<Self>, BackendError> {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = match backend_name {
            "vulkan" => wgpu::Backends::VULKAN,
            "metal" => wgpu::Backends::METAL,
            "dx12" => wgpu::Backends::DX12,
            "opengl" | "gl" => wgpu::Backends::GL,
            "gpu" | "wgpu" | "auto" => {
                wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12
            }
            _ => {
                return Err(BackendError::DeviceNotFound(format!(
                    "unsupported GPU backend {backend_name}"
                )))
            }
        };

        let backends = descriptor.backends;
        let instance = wgpu::Instance::new(descriptor);
        let mut adapters: Vec<wgpu::Adapter> = pollster::block_on(instance.enumerate_adapters(backends));
        let candidates: Vec<(AdapterClass, String)> = adapters
            .iter()
            .map(|a| {
                let info = a.get_info();
                (AdapterClass::of(info.device_type), format!("{} ({:?})", info.name, info.backend))
            })
            .collect();
        let allow_software = std::env::var_os(SOFTWARE_ENV).is_some_and(|v| !v.is_empty() && v != "0");
        let chosen = choose_adapter(backend_name, &candidates, device_id, allow_software).map_err(BackendError::DeviceNotFound)?;
        let adapter = adapters.swap_remove(chosen);
        let info = adapter.get_info();
        let software = AdapterClass::of(info.device_type) == AdapterClass::Software;

        // The defaults cap storage bindings at 128 MiB, less than an LM head or a KV pool;
        // take whatever the adapter offers for buffer sizes and keep the rest at defaults.
        let adapter_limits = adapter.limits();
        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            max_buffer_size: adapter_limits.max_buffer_size,
            ..wgpu::Limits::default()
        }
        // an OpenGL adapter offers less than the defaults (fewer storage buffers, less workgroup memory): ask for no more than it has
        .or_worse_values_from(&adapter_limits);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("fllm-gpu-context"),
            required_limits,
            ..Default::default()
        }))
        .map_err(|e| BackendError::DriverError(format!("cannot initialize GPU: {e}")))?;

        let error = Arc::new(Mutex::new(None::<String>));
        let sink = Arc::clone(&error);
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            let mut slot = sink
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            slot.get_or_insert_with(|| e.to_string());
        }));

        let kernels = KernelId::ALL
            .iter()
            .map(|&id| Self::kernel(&device, id))
            .collect();
        let dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fllm-dummy"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let limits = device.limits();
        let mut name = format!("{} ({:?})", info.name, info.backend);
        if software {
            name.push_str(" [software]");
        }
        let context = Self {
            device,
            queue,
            kernels,
            dummy,
            dummy_id: WGPU_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            limits,
            error,
            name,
            device_id,
            software,
        };
        // Shader compilation and pipeline validation errors arrive through the error handler.
        context
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(60)),
            })
            .map_err(driver)?;
        context.check()?;
        Ok(Arc::new(context))
    }

    fn kernel(device: &wgpu::Device, id: KernelId) -> Kernel {
        let storage = |binding: u32, read_only: bool, dynamic: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: dynamic,
                min_binding_size: if dynamic {
                    NonZeroU64::new(HEADER_WORDS as u64 * 4)
                } else {
                    None
                },
            },
            count: None,
        };
        let mut entries = vec![storage(0, true, true), storage(1, true, false)];
        for (i, &writable) in id.buffers().iter().enumerate() {
            entries.push(storage(2 + i as u32, !writable, false));
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(id.label()),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(id.label()),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(id.label()),
            source: wgpu::ShaderSource::Wgsl(id.source().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(id.label()),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Kernel { layout, pipeline }
    }

    /// Report (and clear) the first device error since the last check.
    fn check(&self) -> Result<(), BackendError> {
        let mut slot = self
            .error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match slot.take() {
            Some(e) => Err(BackendError::DriverError(format!("GPU error: {e}"))),
            None => Ok(()),
        }
    }

    fn offset_alignment(&self) -> usize {
        (self.limits.min_storage_buffer_offset_alignment as usize).max(4)
    }

    fn max_binding(&self) -> u64 {
        self.limits
            .max_storage_buffer_binding_size
            .min(u64::from(u32::MAX))
    }

    fn create_storage(&self, label: &str, size: usize) -> Result<wgpu::Buffer, BackendError> {
        let padded = round4(size.max(4) as u64);
        if padded > self.limits.max_buffer_size {
            return Err(BackendError::OutOfMemory {
                requested: size,
                available: self.limits.max_buffer_size as usize,
            });
        }
        Ok(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: padded,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
    }
}

/// GPU buffer wrapper holding a native `wgpu::Buffer`.
pub struct WgpuBufferInner {
    pub buffer: wgpu::Buffer,
}

fn get_wgpu_buf(buf: &DeviceBuffer) -> Result<&wgpu::Buffer, BackendError> {
    let inner = buf
        .handle
        .downcast_ref::<WgpuBufferInner>()
        .ok_or_else(|| BackendError::BufferError("invalid wgpu buffer handle".into()))?;
    Ok(&inner.buffer)
}

fn wrap(buffer: wgpu::Buffer, size: usize, device_id: usize) -> DeviceBuffer {
    DeviceBuffer::new(
        WGPU_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
        size,
        device_id,
        Arc::new(WgpuBufferInner { buffer }),
    )
}

/// Hardware GPU accelerator backend backed by wgpu.
pub struct WgpuBackend {
    pub context: Arc<GpuContext>,
    capabilities: BackendCapabilities,
}

impl WgpuBackend {
    pub fn new(backend_name: &str, device_id: usize) -> Result<Self, BackendError> {
        let context = GpuContext::new(backend_name, device_id)?;
        let supported_quants = NATIVE_KINDS
            .iter()
            .map(|&k| QuantType::from_gguf_type_id(k))
            .collect();
        let capabilities = BackendCapabilities {
            name: context.name.clone(),
            supported_quants,
            supports_rms_norm: true,
            supports_grouped_rms_norm: false,
            supports_layer_norm: true,
            supports_rope: true,
            supports_swiglu: true,
            supports_geglu: true,
            supports_paged_attention: true,
            supports_kv_cache_write: true,
            supports_embedding_lookup: true,
            supports_device_sampling: false, // host fallback
            sample_on_device: false,
            supports_residual_add: true,
            supports_moe: false, // host fallback
            max_batch_size: 512,
            max_context: 131072,
        };
        Ok(Self {
            context,
            capabilities,
        })
    }
}

impl Backend for WgpuBackend {
    fn name(&self) -> &str {
        &self.context.name
    }

    fn device_id(&self) -> usize {
        self.context.device_id
    }

    fn capabilities(&self) -> &BackendCapabilities {
        &self.capabilities
    }

    fn create_stream(&self) -> Result<Box<dyn Stream>, BackendError> {
        Ok(Box::new(WgpuStream::new(self.context.clone())?))
    }

    fn allocate_buffer(&self, size: usize) -> Result<DeviceBuffer, BackendError> {
        let buffer = self.context.create_storage("fllm-device-buffer", size)?;
        Ok(wrap(buffer, size, self.context.device_id))
    }

    fn free_buffer(&self, _buffer: DeviceBuffer) -> Result<(), BackendError> {
        Ok(())
    }

    fn upload_weight(
        &self,
        name: &str,
        raw_bytes: &[u8],
        quant_type: QuantType,
        rows: usize,
        cols: usize,
        bias: Option<&[f32]>,
    ) -> Result<DeviceWeight, BackendError> {
        let kind = quant_type.to_gguf_type_id();
        let converted;
        let expanded16: Vec<u8>;
        let native_fp8b = if fllm_gguf_probe::fp8b_type(kind) { fp8b_native_kind(raw_bytes, kind) } else { None };
        let native_bnb = if native_fp8b.is_none() && fllm_gguf_probe::bnb::is_bnb(kind) {
            bnb_native_kind(kind, raw_bytes, rows, cols)
        } else {
            None
        };
        let (bytes, quant_type) = if let Some(native) = native_fp8b {
            // WGSL decodes the blob as it is on disk: codes, then one F32 scale per tile.
            (raw_bytes, QuantType::Other(native))
        } else if let Some(native) = native_bnb {
            // bitsandbytes blocks stay as they are: f32 scale, then the code bytes.
            (raw_bytes, QuantType::Other(native))
        } else if fllm_gguf_probe::fp8b_type(kind) {
            // Block-wise FP8 has no WGSL decoder: the codes are expanded once with their tile scales, exactly, to binary16 (2 bytes per weight, not
            // the 4 of the generic F32 expansion below)
            let values = crate::compute::dequantize(raw_bytes, kind, rows * cols)
                .map_err(|e| BackendError::UnsupportedOperation(format!("{name}: {e}")))?;
            log_once(
                "wgpu-fp8b-f16",
                &format!("Block-wise FP8 weights have no WGSL decoder; expanded to F16 on upload (first: {name}, {} MiB)", (values.len() * 2) >> 20),
            );
            if values.iter().any(|v| !v.is_finite() || v.abs() > 65504.0) {
                return Err(invalid(format!("{name}: a dequantized block-wise FP8 weight does not fit binary16")));
            }
            expanded16 = values.iter().flat_map(|&v| half::f16::from_f32(v).to_le_bytes()).collect();
            (expanded16.as_slice(), QuantType::F16)
        } else if NATIVE_KINDS.contains(&kind) {
            let (block, width) = (quant_type.block_size(), quant_type.block_bytes());
            if !cols.is_multiple_of(block) || raw_bytes.len() != rows * (cols / block) * width {
                return Err(invalid(format!(
                    "{name}: {} bytes are not {rows} x {cols} {quant_type:?}",
                    raw_bytes.len()
                )));
            }
            (raw_bytes, quant_type)
        } else {
            // One-time conversion keeps every projection on the device; it costs memory
            // (4 bytes per weight), which the log line states.
            converted = crate::compute::dequantize(raw_bytes, kind, rows * cols)
                .map_err(|e| BackendError::UnsupportedOperation(format!("{name}: {e}")))?;
            log_once(
                &format!("wgpu-dequant-{kind}"),
                &format!(
                    "{quant_type:?} weights have no WGSL decoder; dequantized to F32 on upload \
                     (first: {name}, {} MiB)",
                    (converted.len() * 4) >> 20
                ),
            );
            (bytemuck::cast_slice::<f32, u8>(&converted), QuantType::F32)
        };
        if rows.checked_mul(cols).is_none_or(|n| n > u32::MAX as usize) {
            return Err(invalid(format!(
                "{name}: {rows} x {cols} elements exceed 32-bit kernel indexing"
            )));
        }
        if round4(bytes.len() as u64) > self.context.max_binding() {
            return Err(BackendError::OutOfMemory {
                requested: bytes.len(),
                available: self.context.max_binding() as usize,
            });
        }
        // create_buffer_init pads to a multiple of 4 bytes; the kernels rely on that.
        let buffer = self
            .context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(name),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let bias_buf = match bias {
            Some(b) if b.len() != rows => {
                return Err(invalid(format!(
                    "{name}: bias has {} values for {rows} rows",
                    b.len()
                )))
            }
            Some(b) => {
                let b_buffer =
                    self.context
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(name),
                            contents: bytemuck::cast_slice(b),
                            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                        });
                Some(wrap(b_buffer, b.len() * 4, self.context.device_id))
            }
            None => None,
        };
        self.context.check()?;

        Ok(DeviceWeight {
            name: name.to_string(),
            buffer: wrap(buffer, bytes.len(), self.context.device_id),
            quant_type,
            rows,
            cols,
            bias: bias_buf,
            scale: 1.0,
            perm: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn allocate_kv_cache(
        &self,
        layers: usize,
        pool_blocks: usize,
        block_size: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        kv_kind: i32,
    ) -> Result<DeviceKvCache, BackendError> {
        if !matches!(kv_kind, 1 | 2) {
            let names = crate::kv_format::unpack_kind(kv_kind)
                .map_or_else(|| kv_kind.to_string(), |(k, v)| format!("keys {}, values {}", k.name(), v.name()));
            return Err(BackendError::UnsupportedOperation(format!(
                "the wgpu backend (Vulkan, DX12, Metal) only supports the f16 and q8_0 KV cache, not {names}; use --device cuda or --device cpu for the other formats"
            )));
        }
        let (words_per_layer_k, words_per_layer_v) = DeviceKvCache::geometry(
            layers,
            pool_blocks,
            block_size,
            kv_heads,
            head_dim,
            value_dim,
            kv_kind,
        )?;
        // Kernels bind one layer at a time (offset rounded down to the binding alignment).
        let slack = self.context.offset_alignment() as u64 + 4;
        for words in [words_per_layer_k, words_per_layer_v] {
            if words as u64 * 2 + slack > self.context.max_binding() {
                return Err(BackendError::OutOfMemory {
                    requested: words * 2,
                    available: self.context.max_binding() as usize,
                });
            }
        }
        // Pools are whole 32-bit words so they can be uploaded and bound without a tail.
        let key_pool = self.allocate_buffer((layers * words_per_layer_k * 2).div_ceil(4) * 4)?;
        let value_pool = self.allocate_buffer((layers * words_per_layer_v * 2).div_ceil(4) * 4)?;

        Ok(DeviceKvCache {
            key_pool,
            value_pool,
            kv_kind,
            layers,
            pool_blocks,
            block_size,
            kv_heads,
            head_dim,
            value_dim,
            words_per_layer_k,
            words_per_layer_v,
            layout: None,
            layer_geoms: None,
        })
    }
}

/// One storage binding of a dispatch.
#[derive(Clone, Copy)]
struct Bind<'a> {
    buffer: &'a wgpu::Buffer,
    id: u64,
    offset: u64,
    size: u64,
}

/// Cache key of a bind group: kernel, arena and every buffer window.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct BindKey {
    kernel: KernelId,
    arena: u64,
    windows: [(u64, u64, u64); 4],
}

/// A rotary table uploaded once per (cos, sin) pair, recognised by `Arc` identity.
struct RopeTable {
    cos: Arc<Vec<f32>>,
    sin: Arc<Vec<f32>>,
    buffer: DeviceBuffer,
}

/// Asynchronous execution stream for wgpu.
pub struct WgpuStream {
    context: Arc<GpuContext>,
    encoder: Option<wgpu::CommandEncoder>,
    arena: wgpu::Buffer,
    arena_id: u64,
    /// Host mirror of the arena; `used` words are written before the next submit.
    host: Vec<u32>,
    used: usize,
    bind_groups: HashMap<BindKey, wgpu::BindGroup>,
    staging: Option<wgpu::Buffer>,
    map_state: Arc<AtomicU8>,
    rope_tables: Vec<RopeTable>,
    cpu: cpu::CpuBackend,
    cpu_stream: cpu::CpuStream,
}

impl WgpuStream {
    pub fn new(context: Arc<GpuContext>) -> Result<Self, BackendError> {
        let arena = Self::create_arena(&context, ARENA_WORDS)?;
        Ok(Self {
            context,
            encoder: None,
            arena,
            arena_id: WGPU_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            host: vec![0; ARENA_WORDS],
            used: 0,
            bind_groups: HashMap::new(),
            staging: None,
            map_state: Arc::new(AtomicU8::new(0)),
            rope_tables: Vec::new(),
            cpu: cpu::CpuBackend::new(),
            cpu_stream: cpu::CpuStream,
        })
    }

    fn create_arena(context: &GpuContext, words: usize) -> Result<wgpu::Buffer, BackendError> {
        if words as u64 * 4 > context.max_binding() {
            return Err(BackendError::OutOfMemory {
                requested: words * 4,
                available: context.max_binding() as usize,
            });
        }
        Ok(context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fllm-param-arena"),
            size: words as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
    }

    /// Submit recorded work, writing the parameter arena first.
    fn flush(&mut self) -> Result<(), BackendError> {
        if self.used > 0 {
            self.context.queue.write_buffer(
                &self.arena,
                0,
                bytemuck::cast_slice(&self.host[..self.used]),
            );
            self.used = 0;
        }
        if let Some(enc) = self.encoder.take() {
            self.context.queue.submit([enc.finish()]);
        }
        Ok(())
    }

    /// Reserve `headers` zeroed dispatch headers plus `data` words of arena data for one
    /// operation. Returns the first header word, the word stride between headers and the
    /// first data word. Flushes when the arena is full and grows it when one operation needs
    /// more than it holds.
    fn reserve(
        &mut self,
        headers: usize,
        data: usize,
    ) -> Result<(usize, usize, usize), BackendError> {
        let align = self.context.offset_alignment() / 4;
        let stride = HEADER_WORDS.div_ceil(align) * align;
        let need = stride * headers + data;
        let mut start = self.used.div_ceil(align) * align;
        if start + need > self.host.len() {
            self.flush()?;
            start = 0;
            if need > self.host.len() {
                let words = need.next_power_of_two();
                self.arena = Self::create_arena(&self.context, words)?;
                self.arena_id = WGPU_BUFFER_ID.fetch_add(1, Ordering::Relaxed);
                self.host.resize(words, 0);
                self.bind_groups.clear();
            }
        }
        self.host[start..start + stride * headers].fill(0);
        self.used = start + need;
        Ok((start, stride, start + stride * headers))
    }

    fn bind<'a>(&self, buf: &'a DeviceBuffer) -> Result<Bind<'a>, BackendError> {
        let buffer = get_wgpu_buf(buf)?;
        let align = self.context.offset_alignment();
        if !buf.offset.is_multiple_of(align) {
            return Err(BackendError::BufferError(format!(
                "buffer offset {} is not aligned to {align} bytes for a storage binding",
                buf.offset
            )));
        }
        // Bind the rest of the allocation rather than the logical size: kernels index with the
        // counts in their header (checked on the host), and a pooled buffer reused with a
        // different row count then keeps hitting the same cached bind group.
        let available = buffer.size().saturating_sub(buf.offset as u64);
        let logical = round4(buf.size.max(4) as u64);
        let whole = round4(buf.capacity.saturating_sub(buf.offset).max(4) as u64).min(available);
        let size = if whole >= logical && whole <= self.context.max_binding() {
            whole
        } else {
            logical
        };
        if size > available || size > self.context.max_binding() {
            return Err(BackendError::BufferError(format!(
                "buffer window {}+{size} exceeds its allocation or the binding limit",
                buf.offset
            )));
        }
        Ok(Bind {
            buffer,
            id: buf.id,
            offset: buf.offset as u64,
            size,
        })
    }

    /// Bind `len` bytes at byte `start` of `buf` (a KV layer), rounding the offset down to
    /// the binding alignment. Returns the binding and the 16-bit word at which `start` lies.
    fn bind_window<'a>(
        &self,
        buf: &'a DeviceBuffer,
        start: usize,
        len: usize,
    ) -> Result<(Bind<'a>, u32), BackendError> {
        let buffer = get_wgpu_buf(buf)?;
        let absolute = (buf.offset + start) as u64;
        if !absolute.is_multiple_of(2) || start + len > buf.size {
            return Err(BackendError::BufferError(
                "KV layer window out of range".into(),
            ));
        }
        let align = self.context.offset_alignment() as u64;
        let offset = absolute / align * align;
        let end = round4(absolute + len as u64).min(buffer.size());
        let size = end - offset;
        if size > self.context.max_binding() {
            return Err(BackendError::BufferError(
                "KV layer exceeds the storage binding limit".into(),
            ));
        }
        Ok((
            Bind {
                buffer,
                id: buf.id,
                offset,
                size,
            },
            ((absolute - offset) / 2) as u32,
        ))
    }

    /// Workgroup grid `(x, y)` covering `groups` per-row workgroups.
    fn grid(&self, groups: usize) -> Result<(u32, u32), BackendError> {
        let max = self.context.limits.max_compute_workgroups_per_dimension as usize;
        let x = groups.clamp(1, max);
        let y = groups.div_ceil(x);
        if y > max {
            return Err(invalid(format!(
                "{groups} workgroups exceed the dispatch grid"
            )));
        }
        Ok((x as u32, y as u32))
    }

    /// Workgroups for a grid-stride kernel over `items` elements.
    fn stride_groups(&self, items: usize) -> u32 {
        let max = self.context.limits.max_compute_workgroups_per_dimension as usize;
        items.div_ceil(64).clamp(1, max) as u32
    }

    fn dispatch(
        &mut self,
        kernel: KernelId,
        header: usize,
        binds: &[Bind<'_>],
        groups: (u32, u32, u32),
    ) -> Result<(), BackendError> {
        // profiling aid: FLLM_WGPU_SKIP=attention,linear-kq ... leaves out the kernels whose label contains one of the words
        // (the output is garbage, only the timing of the others is meaningful)
        static SKIP: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
        let skip = SKIP.get_or_init(|| {
            std::env::var("FLLM_WGPU_SKIP").map(|v| v.split(',').map(str::to_owned).collect()).unwrap_or_default()
        });
        if skip.iter().any(|w| !w.is_empty() && kernel.label().contains(w.as_str())) {
            return Ok(());
        }
        let writable = kernel.buffers();
        debug_assert_eq!(binds.len(), writable.len());
        for (i, a) in binds.iter().enumerate() {
            for (j, b) in binds.iter().enumerate().skip(i + 1) {
                if a.id == b.id && a.id != self.context.dummy_id && (writable[i] || writable[j]) {
                    return Err(invalid(format!(
                        "{}: a written buffer is also bound as another operand",
                        kernel.label()
                    )));
                }
            }
        }
        let mut windows = [(0u64, 0u64, 0u64); 4];
        for (w, b) in windows.iter_mut().zip(binds) {
            *w = (b.id, b.offset, b.size);
        }
        let key = BindKey {
            kernel,
            arena: self.arena_id,
            windows,
        };
        let context = Arc::clone(&self.context);
        let entry = &context.kernels[kernel as usize];
        let group = match self.bind_groups.get(&key) {
            Some(group) => group.clone(),
            None => {
                if self.bind_groups.len() >= MAX_BIND_GROUPS {
                    self.bind_groups.clear();
                }
                let mut entries = vec![
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.arena,
                            offset: 0,
                            size: NonZeroU64::new(HEADER_WORDS as u64 * 4),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.arena.as_entire_binding(),
                    },
                ];
                for (i, b) in binds.iter().enumerate() {
                    entries.push(wgpu::BindGroupEntry {
                        binding: 2 + i as u32,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: b.buffer,
                            offset: b.offset,
                            size: NonZeroU64::new(b.size),
                        }),
                    });
                }
                let group = context
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(kernel.label()),
                        layout: &entry.layout,
                        entries: &entries,
                    });
                self.bind_groups.insert(key, group.clone());
                group
            }
        };
        let encoder = self.encoder.get_or_insert_with(|| {
            context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fllm-stream-encoder"),
                })
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(kernel.label()),
            timestamp_writes: None,
        });
        pass.set_pipeline(&entry.pipeline);
        pass.set_bind_group(0, &group, &[(header * 4) as u32]);
        pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        Ok(())
    }

    fn dummy_bind(context: &GpuContext) -> Bind<'_> {
        Bind {
            buffer: &context.dummy,
            id: context.dummy_id,
            offset: 0,
            size: 16,
        }
    }

    /// Copy a device buffer into a new CPU buffer (host fallbacks only).
    fn copy_to_host(&mut self, buf: &DeviceBuffer) -> Result<DeviceBuffer, BackendError> {
        let mut bytes = vec![0u8; buf.size];
        self.download(&mut bytes, buf)?;
        let mut host = self.cpu.allocate_buffer(buf.size)?;
        self.cpu_stream.upload(&mut host, &bytes)?;
        Ok(host)
    }

    fn weight_to_host(&mut self, w: &DeviceWeight) -> Result<DeviceWeight, BackendError> {
        Ok(DeviceWeight {
            buffer: self.copy_to_host(&w.buffer)?,
            bias: w.bias.as_ref().map(|b| self.copy_to_host(b)).transpose()?,
            scale: 1.0,
            ..w.clone()
        })
    }

    /// Copy a CPU buffer back to the device (host fallbacks only).
    fn copy_from_host(
        &mut self,
        host: &DeviceBuffer,
        dst: &mut DeviceBuffer,
    ) -> Result<(), BackendError> {
        let mut bytes = vec![0u8; dst.size.min(host.size)];
        self.cpu_stream.download(&mut bytes, host)?;
        self.upload(dst, &bytes)
    }

    fn rope_table(&mut self, params: &RopeParams) -> Result<DeviceBuffer, BackendError> {
        if let Some(t) = self
            .rope_tables
            .iter()
            .find(|t| Arc::ptr_eq(&t.cos, &params.cos) && Arc::ptr_eq(&t.sin, &params.sin))
        {
            return Ok(t.buffer.clone());
        }
        let bytes = (params.cos.len() + params.sin.len()) * 4;
        let buffer = wrap(
            self.context.create_storage("fllm-rope-table", bytes)?,
            bytes,
            self.context.device_id,
        );
        let raw = get_wgpu_buf(&buffer)?;
        self.context
            .queue
            .write_buffer(raw, 0, bytemuck::cast_slice(&params.cos));
        self.context.queue.write_buffer(
            raw,
            params.cos.len() as u64 * 4,
            bytemuck::cast_slice(&params.sin),
        );
        if self.rope_tables.len() >= 8 {
            self.rope_tables.remove(0);
        }
        self.rope_tables.push(RopeTable {
            cos: Arc::clone(&params.cos),
            sin: Arc::clone(&params.sin),
            buffer: buffer.clone(),
        });
        Ok(buffer)
    }

    fn glu(
        &mut self,
        out: &DeviceBuffer,
        gate: &DeviceBuffer,
        up: &DeviceBuffer,
        mode: u32,
        plain: bool,
    ) -> Result<(), BackendError> {
        let count = gate.size / 4;
        if up.size != gate.size || out.size < gate.size || !gate.size.is_multiple_of(4) {
            return Err(invalid("gated activation operands differ in size"));
        }
        if count == 0 {
            return Ok(());
        }
        let (h, _, _) = self.reserve(1, 0)?;
        self.host[h] = count as u32;
        self.host[h + 1] = mode;
        self.host[h + 2] = u32::from(plain);
        let binds = [self.bind(gate)?, self.bind(up)?, self.bind(out)?];
        let groups = self.stride_groups(count);
        self.dispatch(KernelId::Glu, h, &binds, (groups, 1, 1))
    }

    /// In-place element-wise kernel (`mode` 0 clamp to +-`param`, 1 softplus gate with beta `param`).
    fn ewise(&mut self, x: &mut DeviceBuffer, gate: Option<&DeviceBuffer>, mode: u32, param: f32) -> Result<(), BackendError> {
        let count = x.size / 4;
        if count == 0 {
            return Ok(());
        }
        fits_u32(count, "element-wise operand")?;
        let (h, _, _) = self.reserve(1, 0)?;
        self.host[h..h + 3].copy_from_slice(&[count as u32, mode, param.to_bits()]);
        let context = Arc::clone(&self.context);
        let gate_bind = match gate {
            Some(g) => self.bind(g)?,
            None => Self::dummy_bind(&context),
        };
        let binds = [self.bind(x)?, gate_bind];
        let groups = self.stride_groups(count);
        self.dispatch(KernelId::Ewise, h, &binds, (groups, 1, 1))
    }

    /// One MLA data-movement pass (`MLA_MOVE` modes).
    #[allow(clippy::too_many_arguments)]
    fn mla_move(
        &mut self,
        mode: u32,
        count: usize,
        rows: usize,
        shape: &MlaShape,
        a: &DeviceBuffer,
        b: &DeviceBuffer,
        dst: &DeviceBuffer,
    ) -> Result<(), BackendError> {
        if count == 0 {
            return Ok(());
        }
        fits_u32(count, "MLA operand")?;
        fits_u32(rows * (shape.kv_lora + shape.rope), "MLA latent rows")?;
        let (h, _, _) = self.reserve(1, 0)?;
        self.host[h..h + 8].copy_from_slice(&[
            count as u32,
            mode,
            rows as u32,
            shape.heads as u32,
            shape.nope as u32,
            shape.rope as u32,
            shape.value as u32,
            shape.kv_lora as u32,
        ]);
        let binds = [self.bind(a)?, self.bind(b)?, self.bind(dst)?];
        let groups = self.stride_groups(count);
        self.dispatch(KernelId::MlaMove, h, &binds, (groups, 1, 1))
    }
}

/// Trait activation mode (0 SiLU, 1 GELU tanh, 2 GELU erf, 3 ReLU, 4 ReLU squared) to the kernel's.
fn activation_mode(mode: i32) -> Result<u32, BackendError> {
    u32::try_from(mode).ok().filter(|m| *m <= 4).ok_or_else(|| invalid(format!("unknown activation mode {mode}")))
}

fn fits_u32(n: usize, what: &str) -> Result<u32, BackendError> {
    u32::try_from(n).map_err(|_| invalid(format!("{what} ({n}) exceeds 32-bit kernel indexing")))
}

impl Stream for WgpuStream {
    fn synchronize(&mut self) -> Result<(), BackendError> {
        self.flush()?;
        self.context
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(driver)?;
        self.context.check()
    }

    fn upload(&mut self, dst: &mut DeviceBuffer, src: &[u8]) -> Result<(), BackendError> {
        if src.len() > dst.size {
            return Err(BackendError::BufferError(format!(
                "upload overflow: src {} > dst {}",
                src.len(),
                dst.size
            )));
        }
        if src.is_empty() {
            return Ok(());
        }
        if !dst.offset.is_multiple_of(4) || !src.len().is_multiple_of(4) {
            return Err(BackendError::BufferError(format!(
                "upload of {} bytes at offset {} is not 4-byte aligned",
                src.len(),
                dst.offset
            )));
        }
        // Recorded commands may still read `dst`; submit them before the write lands.
        self.flush()?;
        let buf = get_wgpu_buf(dst)?;
        self.context.queue.write_buffer(buf, dst.offset as u64, src);
        Ok(())
    }

    fn download(&mut self, dst: &mut [u8], src: &DeviceBuffer) -> Result<(), BackendError> {
        if dst.len() > src.size {
            return Err(BackendError::BufferError(format!(
                "download overflow: dst {} > src {}",
                dst.len(),
                src.size
            )));
        }
        if dst.is_empty() {
            return Ok(());
        }
        self.flush()?;
        let src_buf = get_wgpu_buf(src)?;
        let copy_len = round4(dst.len() as u64);
        if !src.offset.is_multiple_of(4) || src.offset as u64 + copy_len > src_buf.size() {
            return Err(BackendError::BufferError(
                "download window is unaligned or exceeds its allocation".into(),
            ));
        }
        // One staging buffer per stream, replaced only when a larger download arrives.
        let staging = match &self.staging {
            Some(s) if s.size() >= copy_len => s.clone(),
            _ => {
                let s = self.context.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("fllm-download-staging"),
                    size: copy_len.max(1 << 16),
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.staging = Some(s.clone());
                s
            }
        };

        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fllm-download-encoder"),
                });
        encoder.copy_buffer_to_buffer(src_buf, src.offset as u64, &staging, 0, copy_len);
        let index = self.context.queue.submit([encoder.finish()]);

        self.map_state.store(0, Ordering::Release);
        let state = Arc::clone(&self.map_state);
        staging
            .slice(..copy_len)
            .map_async(wgpu::MapMode::Read, move |res| {
                state.store(if res.is_ok() { 1 } else { 2 }, Ordering::Release);
            });
        let polled = self
            .context
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(index),
                timeout: Some(Duration::from_secs(60)),
            })
            .map_err(|e| driver(format!("GPU submission failed: {e}")));
        let checked = polled.and_then(|_| self.context.check());
        let mapped = self.map_state.load(Ordering::Acquire);
        if checked.is_err() || mapped != 1 {
            // A failed or pending mapping must not be reused.
            self.staging = None;
            checked?;
            return Err(driver(if mapped == 2 {
                "mapping the download buffer failed"
            } else {
                "download buffer was not mapped after the submission completed"
            }));
        }
        {
            let view = staging
                .slice(..copy_len)
                .get_mapped_range()
                .map_err(|e| driver(format!("get_mapped_range error: {e}")))?;
            dst.copy_from_slice(&view[..dst.len()]);
        }
        staging.unmap();
        Ok(())
    }

    fn copy_buffer(
        &mut self,
        dst: &mut DeviceBuffer,
        src: &DeviceBuffer,
    ) -> Result<(), BackendError> {
        let copy_len = dst.size.min(src.size) as u64;
        if copy_len == 0 {
            return Ok(());
        }
        if dst.id == src.id {
            return Err(BackendError::BufferError(
                "copy_buffer within one allocation".into(),
            ));
        }
        if !copy_len.is_multiple_of(4)
            || !src.offset.is_multiple_of(4)
            || !dst.offset.is_multiple_of(4)
        {
            return Err(BackendError::BufferError(
                "copy_buffer offsets and size must be multiples of 4".into(),
            ));
        }
        let s = get_wgpu_buf(src)?;
        let d = get_wgpu_buf(dst)?;
        let context = Arc::clone(&self.context);
        let encoder = self.encoder.get_or_insert_with(|| {
            context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fllm-stream-encoder"),
                })
        });
        encoder.copy_buffer_to_buffer(s, src.offset as u64, d, dst.offset as u64, copy_len);
        Ok(())
    }

    fn embedding_lookup(
        &mut self,
        out: &mut DeviceBuffer,
        ids: &[i64],
        weight: &DeviceWeight,
    ) -> Result<(), BackendError> {
        let kind = weight.quant_type.to_gguf_type_id();
        if !NATIVE_KINDS.contains(&kind) {
            return Err(BackendError::UnsupportedQuant(weight.quant_type));
        }
        if let Some(&id) = ids.iter().find(|&&id| id < 0 || id as usize >= weight.rows) {
            return Err(invalid(format!("embedding token id {id} out of bounds")));
        }
        let total = ids.len() * weight.cols;
        if total == 0 {
            return Ok(());
        }
        fits_u32(total, "embedding output")?;
        if out.size < total * 4 {
            return Err(invalid("embedding output smaller than ids x hidden"));
        }
        let (h, _, data) = self.reserve(1, ids.len())?;
        for (slot, &id) in self.host[data..data + ids.len()].iter_mut().zip(ids) {
            *slot = id as u32;
        }
        self.host[h..h + 5].copy_from_slice(&[
            weight.rows as u32,
            weight.cols as u32,
            ids.len() as u32,
            kind,
            data as u32,
        ]);
        let binds = [self.bind(&weight.buffer)?, self.bind(out)?];
        let groups = self.stride_groups(total);
        self.dispatch(KernelId::Embedding, h, &binds, (groups, 1, 1))
    }

    fn rms_norm(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        weight: &DeviceBuffer,
        eps: f32,
    ) -> Result<(), BackendError> {
        let cols = weight.size / 4;
        if cols == 0 || !x.size.is_multiple_of(cols * 4) || out.size < x.size {
            return Err(invalid("rms_norm input is not rows of the weight width"));
        }
        let rows = x.size / (cols * 4);
        if rows == 0 {
            return Ok(());
        }
        fits_u32(rows * cols, "rms_norm input")?;
        let (gx, gy) = self.grid(rows)?;
        let (h, _, _) = self.reserve(1, 0)?;
        self.host[h..h + 3].copy_from_slice(&[rows as u32, cols as u32, eps.to_bits()]);
        let binds = [self.bind(x)?, self.bind(weight)?, self.bind(out)?];
        self.dispatch(KernelId::RmsNorm, h, &binds, (gx, gy, 1))
    }

    fn layer_norm(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        weight: &DeviceBuffer,
        bias: Option<&DeviceBuffer>,
        eps: f32,
    ) -> Result<(), BackendError> {
        let cols = weight.size / 4;
        if cols == 0 || !x.size.is_multiple_of(cols * 4) || out.size < x.size {
            return Err(invalid("layer_norm input is not rows of the weight width"));
        }
        if bias.is_some_and(|b| b.size < weight.size) {
            return Err(invalid("layer_norm bias is shorter than its weight"));
        }
        let rows = x.size / (cols * 4);
        if rows == 0 {
            return Ok(());
        }
        fits_u32(rows * cols, "layer_norm input")?;
        let (gx, gy) = self.grid(rows)?;
        let (h, _, _) = self.reserve(1, 0)?;
        self.host[h..h + 4].copy_from_slice(&[rows as u32, cols as u32, eps.to_bits(), u32::from(bias.is_some())]);
        let context = Arc::clone(&self.context);
        let bias_bind = match bias {
            Some(b) => self.bind(b)?,
            None => Self::dummy_bind(&context),
        };
        let binds = [self.bind(x)?, self.bind(weight)?, bias_bind, self.bind(out)?];
        self.dispatch(KernelId::LayerNorm, h, &binds, (gx, gy, 1))
    }

    fn supports_arch_ops(&self) -> bool {
        true
    }

    fn activation(&mut self, out: &mut DeviceBuffer, x: &DeviceBuffer, mode: i32) -> Result<(), BackendError> {
        self.glu(out, x, x, activation_mode(mode)?, true)
    }

    fn activation_glu(
        &mut self,
        out: &mut DeviceBuffer,
        gate: &DeviceBuffer,
        up: &DeviceBuffer,
        mode: i32,
    ) -> Result<(), BackendError> {
        self.glu(out, gate, up, activation_mode(mode)?, false)
    }

    fn softplus_gate_mul(&mut self, x: &mut DeviceBuffer, gate: &DeviceBuffer, beta: f32) -> Result<(), BackendError> {
        if !x.size.is_multiple_of(4) || gate.size < x.size || beta <= 0.0 {
            return Err(invalid("attention gate shorter than its target"));
        }
        self.ewise(x, Some(gate), 1, beta)
    }

    fn clamp(&mut self, x: &mut DeviceBuffer, limit: f32) -> Result<(), BackendError> {
        if !x.size.is_multiple_of(4) {
            return Err(invalid("clamp target is not a whole number of floats"));
        }
        self.ewise(x, None, 0, limit)
    }

    fn mla_latent(
        &mut self,
        latent: &mut DeviceBuffer,
        kv_a: &DeviceBuffer,
        rows: usize,
        kv_lora: usize,
        width: usize,
    ) -> Result<(), BackendError> {
        if width < kv_lora || kv_a.size < rows * width * 4 || latent.size < rows * kv_lora * 4 {
            return Err(invalid("mla_latent buffers are smaller than rows x width"));
        }
        let shape = MlaShape { heads: 1, nope: 0, rope: width - kv_lora, value: 1, kv_lora };
        self.mla_move(3, rows * kv_lora, rows, &shape, kv_a, kv_a, latent)
    }

    #[allow(clippy::too_many_arguments)]
    fn mla_assemble(
        &mut self,
        q: &mut DeviceBuffer,
        k: &mut DeviceBuffer,
        v: &mut DeviceBuffer,
        q_full: &DeviceBuffer,
        kv_a: &DeviceBuffer,
        kv: &DeviceBuffer,
        rows: usize,
        shape: &MlaShape,
    ) -> Result<(), BackendError> {
        let (dk, width) = (shape.nope + shape.rope, shape.kv_lora + shape.rope);
        let items = rows * shape.heads;
        if q_full.size < items * dk * 4
            || kv_a.size < rows * width * 4
            || kv.size < items * (shape.nope + shape.value) * 4
            || q.size < items * dk * 4
            || k.size < items * dk * 4
            || v.size < items * shape.value * 4
        {
            return Err(invalid("mla_assemble buffers are smaller than rows x heads x width"));
        }
        self.mla_move(0, items * dk, rows, shape, q_full, q_full, q)?;
        self.mla_move(1, items * dk, rows, shape, kv_a, kv, k)?;
        self.mla_move(2, items * shape.value, rows, shape, kv, kv, v)
    }

    fn rope(
        &mut self,
        q: &mut DeviceBuffer,
        k: &mut DeviceBuffer,
        positions: &[usize],
        params: &RopeParams,
    ) -> Result<(), BackendError> {
        params.validate(positions, q.size, k.size)?;
        let rows = positions.len();
        let half = params.rope_dim / 2;
        let total = rows * (params.heads + params.kv_heads) * half;
        if total == 0 {
            return Ok(());
        }
        fits_u32(total, "rope pairs")?;
        fits_u32(params.cos.len() * 2, "rope table")?;
        fits_u32(rows * params.heads * params.dim, "rope queries")?;
        let table = self.rope_table(params)?;
        let (h, _, data) = self.reserve(1, rows)?;
        for (slot, &p) in self.host[data..data + rows].iter_mut().zip(positions) {
            *slot = p as u32;
        }
        self.host[h..h + 8].copy_from_slice(&[
            rows as u32,
            params.heads as u32,
            params.kv_heads as u32,
            params.dim as u32,
            half as u32,
            u32::from(params.neox),
            data as u32,
            params.cos.len() as u32,
        ]);
        let binds = [self.bind(q)?, self.bind(k)?, self.bind(&table)?];
        let groups = self.stride_groups(total);
        self.dispatch(KernelId::Rope, h, &binds, (groups, 1, 1))
    }

    fn quantized_linear(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        weight: &DeviceWeight,
        bias: Option<&DeviceBuffer>,
    ) -> Result<(), BackendError> {
        let kind = weight.quant_type.to_gguf_type_id();
        if !NATIVE_KINDS.contains(&kind) {
            // upload_weight converts every other type, so this weight did not come from it.
            return Err(BackendError::UnsupportedQuant(weight.quant_type));
        }
        let (rows, cols) = (weight.rows, weight.cols);
        if rows == 0 || cols == 0 || !x.size.is_multiple_of(cols * 4) {
            return Err(invalid("linear input dimensions do not match"));
        }
        let batch = x.size / (cols * 4);
        if batch == 0 {
            return Ok(());
        }
        if out.size < batch * rows * 4 {
            return Err(invalid("linear output smaller than batch x rows"));
        }
        if bias.is_some_and(|b| b.size < rows * 4) {
            return Err(invalid("linear bias shorter than the output rows"));
        }
        fits_u32(batch * cols, "linear input")?;
        fits_u32(batch * rows, "linear output")?;
        let tiled = batch >= 16;
        let (gx, gy) = if tiled { (rows.div_ceil(64) as u32, batch.div_ceil(64) as u32) } else { self.grid(rows)? };
        let gz = if tiled { 1 } else { batch.div_ceil(4) };
        if gz > self.context.limits.max_compute_workgroups_per_dimension as usize
            || (tiled && (gx.max(gy) as usize) > self.context.limits.max_compute_workgroups_per_dimension as usize)
        {
            return Err(invalid(format!(
                "linear batch {batch} exceeds the dispatch grid"
            )));
        }
        let (h, _, _) = self.reserve(1, 0)?;
        self.host[h..h + 5].copy_from_slice(&[
            rows as u32,
            cols as u32,
            batch as u32,
            kind,
            u32::from(bias.is_some()),
        ]);
        let context = Arc::clone(&self.context);
        let bias_bind = match bias {
            Some(b) => self.bind(b)?,
            None => Self::dummy_bind(&context),
        };
        let binds = [
            self.bind(&weight.buffer)?,
            self.bind(x)?,
            self.bind(out)?,
            bias_bind,
        ];
        let id = if tiled {
            KernelId::LinearTiled
        } else if kind == 14 && cols % 256 == 0 {
            KernelId::LinearQ6
        } else if matches!(kind, 12 | 13) && cols % 256 == 0 {
            KernelId::LinearKq
        } else {
            KernelId::Linear
        };
        self.dispatch(id, h, &binds, (gx, gy, gz as u32))
    }

    fn swiglu(
        &mut self,
        out: &mut DeviceBuffer,
        gate: &DeviceBuffer,
        up: &DeviceBuffer,
    ) -> Result<(), BackendError> {
        self.glu(out, gate, up, 0, false)
    }

    fn geglu(
        &mut self,
        out: &mut DeviceBuffer,
        gate: &DeviceBuffer,
        up: &DeviceBuffer,
    ) -> Result<(), BackendError> {
        self.glu(out, gate, up, 1, false)
    }

    fn residual_add(
        &mut self,
        x: &mut DeviceBuffer,
        residual: &DeviceBuffer,
        scale: f32,
    ) -> Result<(), BackendError> {
        if !x.size.is_multiple_of(4) || residual.size < x.size {
            return Err(invalid("residual shorter than its target"));
        }
        let count = x.size / 4;
        if count == 0 {
            return Ok(());
        }
        let (h, _, _) = self.reserve(1, 0)?;
        self.host[h..h + 2].copy_from_slice(&[count as u32, scale.to_bits()]);
        let binds = [self.bind(x)?, self.bind(residual)?];
        let groups = self.stride_groups(count);
        self.dispatch(KernelId::Residual, h, &binds, (groups, 1, 1))
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_paged(
        &mut self,
        out: &mut DeviceBuffer,
        q: &DeviceBuffer,
        kv: &DeviceKvCache,
        block_tables: &[i64],
        lengths: &[i64],
        layer_idx: usize,
        params: &AttentionParams,
    ) -> Result<(), BackendError> {
        params.validate(kv, block_tables, lengths, layer_idx)?;
        let (dk, dv, heads) = (params.head_dim, params.value_dim, params.num_heads);
        let rows = lengths.len();
        if rows == 0 {
            return Ok(());
        }
        if q.size < rows * heads * dk * 4 || out.size < rows * heads * dv * 4 {
            return Err(invalid("attention q/out smaller than rows x heads x dim"));
        }
        if dk > MAX_ATTENTION_DIM || dv > MAX_ATTENTION_DIM {
            warn_fallback_reason(
                "attention_paged",
                &format!("head width {dk}/{dv} exceeds the WGSL kernel's {MAX_ATTENTION_DIM}; the whole KV pool is copied every call"),
            );
            return self.attention_host(out, q, kv, block_tables, lengths, layer_idx, params);
        }
        let slots = kv.slots();
        fits_u32(slots * kv.kv_heads * dk.max(dv) * 2, "KV layer bytes")?;
        fits_u32(rows * heads * dk.max(dv), "attention queries")?;
        let sinks = params.sinks.as_ref().map(|v| v.as_slice());
        let alibi = params.alibi_slopes.as_ref().map(|v| v.as_slice());
        let words =
            rows + block_tables.len() + sinks.map_or(0, |_| heads) + alibi.map_or(0, |_| heads);
        let (h, _, data) = self.reserve(1, words)?;
        let lengths_base = data;
        let tables_base = lengths_base + rows;
        let mut next = tables_base + block_tables.len();
        for (slot, &n) in self.host[lengths_base..tables_base].iter_mut().zip(lengths) {
            *slot = n as u32;
        }
        // validate() checked every block a row reads; unread entries may be any value.
        for (slot, &b) in self.host[tables_base..next].iter_mut().zip(block_tables) {
            *slot = b.clamp(0, u32::MAX as i64) as u32;
        }
        let mut place = |values: Option<&[f32]>, host: &mut [u32]| -> u32 {
            match values {
                Some(v) => {
                    for (slot, &x) in host[next..next + heads].iter_mut().zip(v) {
                        *slot = x.to_bits();
                    }
                    next += heads;
                    (next - heads) as u32
                }
                None => NONE,
            }
        };
        let sinks_base = place(sinks, &mut self.host);
        let alibi_base = place(alibi, &mut self.host);

        let (wk, wv) = (kv.words_per_layer_k, kv.words_per_layer_v);
        let (kwin, kbase) = self.bind_window(&kv.key_pool, layer_idx * wk * 2, wk * 2)?;
        let (vwin, vbase) = self.bind_window(&kv.value_pool, layer_idx * wv * 2, wv * 2)?;
        let (kk, kvv) = (kv.kv_heads * dk, kv.kv_heads * dv);
        let header = [
            rows as u32,
            heads as u32,
            params.num_kv_heads as u32,
            dk as u32,
            dv as u32,
            params.stride as u32,
            params.block_size as u32,
            params.scale.to_bits(),
            params.softcap.unwrap_or(0.0).to_bits(),
            params.window.map_or(0, |w| w.min(u32::MAX as usize) as u32),
            params.chunk.map_or(0, |c| c.min(u32::MAX as usize) as u32),
            kv.kv_kind as u32,
            kbase,
            vbase,
            (slots * kk).div_ceil(2) as u32,
            (slots * kvv).div_ceil(2) as u32,
            kv_quant_group(dk) as u32,
            kv_quant_group(dv) as u32,
            lengths_base as u32,
            tables_base as u32,
            sinks_base,
            alibi_base,
        ];
        self.host[h..h + header.len()].copy_from_slice(&header);
        let (gx, gy) = self.grid(rows * heads)?;
        let binds = [self.bind(q)?, kwin, vwin, self.bind(out)?];
        self.dispatch(KernelId::Attention, h, &binds, (gx, gy, 1))
    }

    fn kv_cache_write(
        &mut self,
        kv: &DeviceKvCache,
        layer_idx: usize,
        slots: &[usize],
        k: &DeviceBuffer,
        v: &DeviceBuffer,
    ) -> Result<(), BackendError> {
        validate_kv_write(kv, layer_idx, slots, k.size, v.size)?;
        let rows = slots.len();
        if rows == 0 {
            return Ok(());
        }
        let (h, stride, data) = self.reserve(2, rows)?;
        for (dst, &s) in self.host[data..data + rows].iter_mut().zip(slots) {
            *dst = s as u32;
        }
        let pools = [
            (&kv.key_pool, k, kv.head_dim, kv.words_per_layer_k),
            (&kv.value_pool, v, kv.value_dim, kv.words_per_layer_v),
        ];
        for (n, (pool, src, dim, words)) in pools.into_iter().enumerate() {
            let width = kv.kv_heads * dim;
            let group = kv_quant_group(dim);
            fits_u32(rows * width, "kv_cache_write values")?;
            fits_u32(kv.slots() * width * 2, "KV layer bytes")?;
            let (window, base) = self.bind_window(pool, layer_idx * words * 2, words * 2)?;
            let header = h + n * stride;
            self.host[header..header + 7].copy_from_slice(&[
                rows as u32,
                width as u32,
                kv.kv_kind as u32,
                group as u32,
                base,
                (kv.slots() * width).div_ceil(2) as u32,
                data as u32,
            ]);
            let items = if kv.kv_kind == 2 {
                rows * (width / group)
            } else {
                rows * width
            };
            let groups = self.stride_groups(items);
            let binds = [self.bind(src)?, window];
            self.dispatch(KernelId::KvWrite, header, &binds, (groups, 1, 1))?;
        }
        Ok(())
    }

    fn moe_route(
        &mut self,
        weights_out: &mut DeviceBuffer,
        indices_out: &mut DeviceBuffer,
        router_logits: &DeviceBuffer,
        num_experts: usize,
        top_k: usize,
    ) -> Result<(), BackendError> {
        warn_fallback_reason("moe_route", "no WGSL router kernel yet");
        let cl = self.copy_to_host(router_logits)?;
        let mut cw = self.cpu.allocate_buffer(weights_out.size)?;
        let mut ci = self.cpu.allocate_buffer(indices_out.size)?;
        self.cpu_stream
            .moe_route(&mut cw, &mut ci, &cl, num_experts, top_k)?;
        self.copy_from_host(&cw, weights_out)?;
        self.copy_from_host(&ci, indices_out)
    }

    fn moe_expert_gemm(
        &mut self,
        out: &mut DeviceBuffer,
        x: &DeviceBuffer,
        expert_weights: &[DeviceWeight],
        expert_indices: &DeviceBuffer,
        expert_weights_buf: &DeviceBuffer,
        top_k: usize,
    ) -> Result<(), BackendError> {
        warn_fallback_reason(
            "moe_expert_gemm",
            "no WGSL expert kernel yet; expert weights are copied every call",
        );
        let cx = self.copy_to_host(x)?;
        let cidx = self.copy_to_host(expert_indices)?;
        let cwts = self.copy_to_host(expert_weights_buf)?;
        let experts = expert_weights
            .iter()
            .map(|w| self.weight_to_host(w))
            .collect::<Result<Vec<_>, _>>()?;
        let mut co = self.cpu.allocate_buffer(out.size)?;
        self.cpu_stream
            .moe_expert_gemm(&mut co, &cx, &experts, &cidx, &cwts, top_k)?;
        self.copy_from_host(&co, out)
    }

    fn sampling_logits(
        &mut self,
        token_ids_out: &mut [u32],
        logits: &DeviceBuffer,
        params: &SamplingParams,
    ) -> Result<(), BackendError> {
        warn_fallback_reason("sampling_logits", "no WGSL sampling kernel yet");
        let cl = self.copy_to_host(logits)?;
        self.cpu_stream.sampling_logits(token_ids_out, &cl, params)
    }
}

impl WgpuStream {
    /// Paged attention on the host for head widths the kernel does not hold.
    #[allow(clippy::too_many_arguments)]
    fn attention_host(
        &mut self,
        out: &mut DeviceBuffer,
        q: &DeviceBuffer,
        kv: &DeviceKvCache,
        block_tables: &[i64],
        lengths: &[i64],
        layer_idx: usize,
        params: &AttentionParams,
    ) -> Result<(), BackendError> {
        let cq = self.copy_to_host(q)?;
        let host_kv = DeviceKvCache {
            key_pool: self.copy_to_host(&kv.key_pool)?,
            value_pool: self.copy_to_host(&kv.value_pool)?,
            ..kv.clone()
        };
        let mut co = self.cpu.allocate_buffer(out.size)?;
        self.cpu_stream.attention_paged(
            &mut co,
            &cq,
            &host_kv,
            block_tables,
            lengths,
            layer_idx,
            params,
        )?;
        self.copy_from_host(&co, out)
    }
}

/// Weight formats the linear and embedding kernels decode natively.
pub fn native_formats() -> Vec<String> {
    NATIVE_KINDS
        .iter()
        .map(|&k| format!("{:?}", QuantType::from_gguf_type_id(k)))
        .collect()
}

/// Every WGSL module, for validation without a device.
pub fn shader_sources() -> Vec<(&'static str, String)> {
    KernelId::ALL
        .iter()
        .map(|&id| (id.label(), id.source()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{choose_adapter, AdapterClass};
    use wgpu::naga;

    fn adapters(list: &[(AdapterClass, &str)]) -> Vec<(AdapterClass, String)> {
        list.iter().map(|(c, n)| (*c, (*n).to_owned())).collect()
    }

    #[test]
    fn index_zero_prefers_a_discrete_gpu_and_ignores_software() {
        use AdapterClass::*;
        let list = adapters(&[(Software, "WARP"), (Integrated, "iGPU"), (Discrete, "dGPU")]);
        assert_eq!(choose_adapter("dx12", &list, 0, false), Ok(2));
        assert_eq!(choose_adapter("dx12", &list, 0, true), Ok(2));
        assert_eq!(choose_adapter("dx12", &list, 1, false), Ok(2));
        assert!(choose_adapter("dx12", &list, 2, false).unwrap_err().contains("adapter number 2 (2 hardware adapters found)"));
        let only_integrated = adapters(&[(Integrated, "iGPU")]);
        assert_eq!(choose_adapter("vulkan", &only_integrated, 0, false), Ok(0));
    }

    #[test]
    fn software_only_is_refused_with_an_actionable_message() {
        use AdapterClass::*;
        let list = adapters(&[(Software, "Microsoft Basic Render Driver (Dx12)")]);
        let error = choose_adapter("dx12", &list, 0, false).unwrap_err();
        assert!(error.contains("no hardware dx12 GPU adapter") && error.contains("Basic Render Driver"), "{error}");
        assert!(error.contains("FLLM_ALLOW_SOFTWARE_GPU=1") && error.contains("--device cpu"), "{error}");
        assert_eq!(choose_adapter("dx12", &list, 0, true), Ok(0));
        // a software renderer is only ever number 0
        assert!(choose_adapter("dx12", &list, 1, true).is_err());
    }

    #[test]
    fn no_adapter_at_all_says_so() {
        let error = choose_adapter("vulkan", &[], 0, true).unwrap_err();
        assert!(error.starts_with("no vulkan GPU adapter found"), "{error}");
        assert!(choose_adapter("gpu", &[], 0, false).unwrap_err().starts_with("no GPU adapter found"));
    }

    #[test]
    fn shaders_parse_and_validate() {
        for (label, source) in super::shader_sources() {
            let module = naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(&source)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(), // wgpu enables the f16-unpack capability on real devices
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        }
    }

    /// bitsandbytes blocks: the WGSL decoder's arithmetic (mirrored here) against the reference
    /// dequantizer, for every family and both block sizes.
    #[test]
    fn bnb_native_matches_the_reference_dequantizer() {
        use super::{bnb_native_kind, BNB_NATIVE};
        // the bit patterns the WGSL module carries
        let nf4: [u32; 16] = [
            0xbf800000, 0xbf3239b1, 0xbf066b30, 0xbeca32a0, 0xbe91a24d, 0xbe3d353f, 0xbdba7871, 0x00000000,
            0x3da2faff, 0x3e24cae3, 0x3e7c04dd, 0x3ead033a, 0x3ee1a4b8, 0x3f1007ab, 0x3f3913b3, 0x3f800000,
        ];
        let fp4: [u32; 16] = [
            0x00000000, 0x3baaaaab, 0x3f2aaaab, 0x3f800000, 0x3eaaaaab, 0x3f000000, 0x3e2aaaab, 0x3e800000,
            0x80000000, 0xbbaaaaab, 0xbf2aaaab, 0xbf800000, 0xbeaaaaab, 0xbf000000, 0xbe2aaaab, 0xbe800000,
        ];
        // the tables must be the ones bitsandbytes defines
        for (i, value) in nf4.iter().enumerate() {
            assert_eq!(f32::from_bits(*value), fllm_gguf_probe::bnb::NF4_CODE[i], "NF4 code {i}");
        }
        for (i, value) in fp4.iter().enumerate() {
            assert_eq!(f32::from_bits(*value), fllm_gguf_probe::bnb::FP4_CODE[i], "FP4 code {i}");
        }
        for (id, block) in [
            (fllm_gguf_probe::bnb::NF4_B64, 64usize),
            (fllm_gguf_probe::bnb::NF4_B128, 128),
            (fllm_gguf_probe::bnb::FP4_B64, 64),
            (fllm_gguf_probe::bnb::FP4_B128, 128),
        ] {
            let (rows, cols) = (2usize, block * 2);
            let layout = fllm_gguf_probe::bnb::layout(id).expect("layout");
            let mut bytes = Vec::new();
            for b in 0..rows * cols / block {
                bytes.extend_from_slice(&(0.5f32 * (b as f32 + 1.0)).to_le_bytes());
                for j in 0..block / 2 {
                    bytes.push((((j * 13 + b) % 256) as u8).wrapping_mul(7).wrapping_add(3));
                }
            }
            let marker = bnb_native_kind(id, &bytes, rows, cols).expect("the WGSL decoder takes this");
            assert_eq!(marker & 0xFC00_0000, BNB_NATIVE);
            let expected_family = match layout.family {
                fllm_gguf_probe::bnb::Family::Nf4 => 0u32,
                fllm_gguf_probe::bnb::Family::Fp4 => 1,
                fllm_gguf_probe::bnb::Family::Int8 => 2,
            };
            assert_eq!((marker >> 5) & 3, expected_family, "family in the marker");
            let family = (marker >> 5) & 3;
            let per_block = 1usize << (marker & 31);
            assert_eq!(per_block, block);
            let book: &[u32; 16] = if family == 0 { &nf4 } else { &fp4 };
            let reference = fllm_gguf_probe::bnb::dequantize(&bytes, id, rows * cols).expect("reference");
            for index in 0..rows * cols {
                let local = index % per_block;
                let start = (index / per_block) * (4 + per_block / 2);
                let scale = f32::from_le_bytes([
                    bytes[start], bytes[start + 1], bytes[start + 2], bytes[start + 3],
                ]);
                let byte = bytes[start + 4 + local / 2];
                let nib = if local % 2 == 0 { byte >> 4 } else { byte & 15 };
                let mine = f32::from_bits(book[nib as usize]) * scale;
                assert_eq!(mine, reference[index], "block {per_block}, index {index}");
            }
        }
        // INT8: one scale per block, one signed byte per value
        let id = fllm_gguf_probe::bnb::INT8_ROW;
        let (rows, cols) = (2usize, 128usize);
        let mut bytes = Vec::new();
        for b in 0..rows * cols / 64 {
            bytes.extend_from_slice(&(0.25f32 * (b as f32 + 1.0)).to_le_bytes());
            for j in 0..64 {
                bytes.push(((j * 37 + b * 11) % 256) as u8);
            }
        }
        let marker = bnb_native_kind(id, &bytes, rows, cols).expect("int8 native");
        assert_eq!((marker >> 5) & 3, 2);
        let reference = fllm_gguf_probe::bnb::dequantize(&bytes, id, rows * cols).expect("reference");
        for index in 0..rows * cols {
            let local = index % 64;
            let start = (index / 64) * 68;
            let scale = f32::from_le_bytes([
                bytes[start], bytes[start + 1], bytes[start + 2], bytes[start + 3],
            ]);
            let mine = bytes[start + 4 + local] as i8 as f32 * scale;
            assert_eq!(mine, reference[index], "int8 index {index}");
        }
        // a block grid that does not divide the row stays on the expansion
        assert_eq!(bnb_native_kind(fllm_gguf_probe::bnb::NF4_B64, &[0u8; 36 * 3], 1, 192 + 8), None);
    }

    /// The native FP8B path must decode exactly what the upload-time expansion produced before it:
    /// same codes, same tile scales, same marker wiring. The arithmetic here mirrors
    /// `fp8b_weight`/`e4m3_value` in the WGSL module line by line.
    #[test]
    fn fp8b_native_matches_the_reference_dequantizer() {
        use super::{fp8b_native_kind, FP8B_NATIVE};
        let decode = |code: u8| -> f32 {
            let v = u32::from(code);
            let mag = v & 0x7f;
            let normal = f32::from_bits((mag << 20) + (120 << 23));
            let subnormal = (mag & 7) as f32 * (1.0 / 512.0);
            let value = if mag >> 3 == 0 { subnormal } else { normal };
            let value = if mag == 0x7f { f32::NAN } else { value };
            f32::from_bits(value.to_bits() | ((v & 0x80) << 24))
        };
        for (block_rows, block_cols, rows, cols) in
            [(1usize, 8usize, 2usize, 16usize), (2, 8, 4, 24), (128, 128, 128, 256)]
        {
            let id = fllm_gguf_probe::fp8b_type_id(block_rows, block_cols).expect("fp8b type id");
            let layout = fllm_gguf_probe::Fp8bLayout::new(id, cols, rows, 1).expect("layout");
            let codes = [0x00u8, 0x01, 0x38, 0xb8, 0x4e, 0x7e, 0x7f];
            let mut bytes: Vec<u8> = (0..rows * cols).map(|i| codes[i % codes.len()]).collect();
            for tile in 0..layout.tiles_per_slab() {
                bytes.extend_from_slice(&(0.125f32 * (tile as f32 + 1.0)).to_le_bytes());
            }
            bytes.extend_from_slice(&layout.trailer());

            let marker = fp8b_native_kind(&bytes, id).expect("the WGSL decoder takes this blob");
            assert_eq!(marker & 0xF800_0000, FP8B_NATIVE);
            let (br, bc) = (1usize << ((marker >> 4) & 15), 1usize << (marker & 15));
            assert_eq!((br, bc), (block_rows, block_cols), "marker must round-trip the tile shape");

            let reference = crate::compute::dequantize(&bytes, id, rows * cols).expect("reference");
            for index in 0..rows * cols {
                let (row, col) = (index / cols, index % cols);
                let tile_cols = cols.div_ceil(bc);
                let tile = (row / br) * tile_cols + col / bc;
                let base = rows * cols + 4 * tile;
                let scale =
                    f32::from_le_bytes([bytes[base], bytes[base + 1], bytes[base + 2], bytes[base + 3]]);
                let mine = decode(bytes[index]) * scale;
                let want = reference[index];
                assert!(
                    mine == want || (mine.is_nan() && want.is_nan()),
                    "{rows}x{cols} tiles {br}x{bc}, index {index}: {mine} != {want}"
                );
            }
        }
        // Stacked experts (several slabs) keep the exact expansion: the kernels index one matrix.
        let id = fllm_gguf_probe::fp8b_type_id(1, 8).expect("fp8b type id");
        let layout = fllm_gguf_probe::Fp8bLayout::new(id, 8, 1, 2).expect("layout");
        let mut bytes = vec![0x38u8; layout.elements()];
        for _ in 0..layout.tiles() {
            bytes.extend_from_slice(&1.0f32.to_le_bytes());
        }
        bytes.extend_from_slice(&layout.trailer());
        assert_eq!(fp8b_native_kind(&bytes, id), None, "multi-slab must not take the native path");
    }
}
