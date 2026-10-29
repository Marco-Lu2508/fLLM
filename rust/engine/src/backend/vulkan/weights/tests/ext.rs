//! Tests of the native layouts of block-wise FP8, bitsandbytes and MLX affine weights: the repacking, the Rust transcription of the chunk decoders
//! (`_q_fmt_{fp8b,nf4,int8r,mlx}.glsl`) against `compute::dequantize`, slicing by rows and the kernel selection of the planner. No GPU involved; the
//! shaders themselves run in the Vulkan self test (`fllm-rs selftest-gpu --device vulkan`, see `docs/gpu-backends.md`).

use super::super::mirror::{chunk, chunk_f};
use super::*;
use fllm_gguf_probe::bnb;

fn ext_sample(quant: QuantType, rows: usize, cols: usize, seed: u32) -> Vec<u8> {
    let mut rng = Lcg(seed.into());
    match quant.mlx_affine() {
        Some(m) => crate::cpu_mlx::random_weights(m, rows, cols, &mut rng),
        None => random_weights(quant.to_gguf_type_id(), rows, cols, &mut rng),
    }
}

fn fp8b(block_rows: usize, block_cols: usize) -> QuantType {
    QuantType::from_gguf_type_id(fllm_gguf_probe::fp8b_type_id(block_rows, block_cols).unwrap())
}

fn mlx(bits: u32, group: u32, bf16: bool) -> QuantType {
    QuantType::Other(fllm_gguf_probe::MlxAffine { bits, group, bf16 }.type_id())
}

/// Every MLX type of the capability list plus a group (96) that is no power of two, the integer bitsandbytes type, and the float-chunk types.
fn int_chunk_formats() -> Vec<QuantType> {
    let mut all: Vec<QuantType> = QuantType::mlx_affine_common();
    for bits in [2, 3, 4, 5, 6, 8] {
        all.push(mlx(bits, 96, false));
        all.push(mlx(bits, 96, true));
    }
    all.push(QuantType::Int8Row);
    all
}

fn float_chunk_formats() -> Vec<QuantType> {
    let mut all = vec![QuantType::Nf4B64, QuantType::Nf4B128, QuantType::Fp4B64, QuantType::Fp4B128];
    for (r, c) in [(128, 128), (32, 32), (64, 8), (1, 128), (128, 32), (16, 16), (1, 64), (4, 1024)] {
        all.push(fp8b(r, c));
    }
    all
}

/// A width every format of the list divides: the block, the group and (FP8) a few tiles plus a partial one.
fn width(info: &FormatInfo) -> usize {
    match info.geometry {
        Geometry::Fp8b { block_cols, .. } => (block_cols * 2 + 8).max(24),
        _ => info.src_qk.max(info.qk) * 2,
    }
}

#[test]
fn extended_formats_are_listed_and_have_aligned_planes() {
    let listed = extended_formats();
    assert!(listed.contains(&QuantType::Fp8Block) && listed.contains(&QuantType::Int8Row) && listed.contains(&mlx(4, 64, true)));
    for quant in int_chunk_formats().into_iter().chain(float_chunk_formats()) {
        let info = format_info(quant).unwrap_or_else(|| panic!("{quant:?} has no format info"));
        let cols = width(&info);
        let layout = DeviceLayout::new(quant, 7, cols).unwrap_or_else(|e| panic!("{quant:?} {cols}: {e}"));
        for plane in 0..4 {
            assert_eq!(layout.plane_byte_offset(plane) % 16, 0, "{quant:?} plane {plane}");
        }
        assert_eq!(layout.total_bytes % 16, 0);
        // the device image is the payload of the file plus small tables; FP8 repeats the scale of a tile for the rows of the tile (at most 4 bytes per
        // 8 weights for the narrowest tile the kernels take)
        let file = 7 * cols / info.src_qk * info.src_block_bytes;
        assert!(layout.total_bytes >= file.min(7 * cols) && layout.total_bytes <= file * 3 / 2 + 7 * cols / 2 + 256, "{quant:?}: {} bytes for {file} in the file", layout.total_bytes);
        assert_eq!(info.float_chunk, !matches!(info.geometry, Geometry::Mlx { .. }) && quant != QuantType::Int8Row, "{quant:?}");
    }
}

/// Integer-chunk formats (MLX, int8): `ws * q + wm` of every chunk against the reference weights.
#[test]
fn mlx_and_int8_chunks_decode_to_the_reference_weights() {
    for quant in int_chunk_formats() {
        let info = format_info(quant).unwrap();
        let (rows, cols) = (5usize, width(&info));
        let raw = ext_sample(quant, rows, cols, 0x5eed_0001);
        let packed = pack_weight(quant, &raw, rows, cols).expect("pack");
        let want = crate::compute::dequantize(&raw, quant.to_gguf_type_id(), rows * cols).expect("reference");
        let bpr = cols / info.qk;
        for r in 0..rows {
            for b in 0..bpr {
                for cin in 0..info.qk / 8 {
                    let c = chunk(&packed.layout, &packed.bytes, r * bpr + b, cin);
                    for i in 0..8 {
                        let got = c.ws * c.q[i] as f32 + c.wm;
                        let at = r * cols + b * info.qk + cin * 8 + i;
                        // 8-bit MLX codes are stored as code - 128 with the offset folded into the bias: a few more roundings
                        let tol = 1e-5 * (c.ws * c.q[i] as f32).abs().max(c.wm.abs()) + 1e-7;
                        assert!((got - want[at]).abs() <= tol, "{quant:?} row {r} block {b} chunk {cin} value {i}: {got} vs {}", want[at]);
                    }
                }
            }
        }
    }
}

/// Float-chunk formats (block-wise FP8, NF4, FP4): the decoded values are the reference weights, bit for bit (same single multiplication).
#[test]
fn fp8_nf4_fp4_chunks_decode_exactly() {
    for quant in float_chunk_formats() {
        let info = format_info(quant).unwrap();
        let (rows, cols) = (9usize, width(&info));
        let raw = ext_sample(quant, rows, cols, 0x5eed_0002);
        let packed = pack_weight(quant, &raw, rows, cols).expect("pack");
        let want = crate::compute::dequantize(&raw, quant.to_gguf_type_id(), rows * cols).expect("reference");
        let bpr = cols / info.qk;
        for r in 0..rows {
            for b in 0..bpr {
                for cin in 0..info.qk / 8 {
                    let v = chunk_f(&packed.layout, &packed.bytes, r * bpr + b, cin);
                    for (i, got) in v.iter().enumerate() {
                        let at = r * cols + b * info.qk + cin * 8 + i;
                        assert_eq!(got.to_bits(), want[at].to_bits(), "{quant:?} row {r} block {b} chunk {cin} value {i}: {got} vs {}", want[at]);
                    }
                }
            }
        }
    }
}

/// The dot product through the chunk decoders (the arithmetic of `mv_scalar.comp`) against the f64 dot product of the reference weights.
#[test]
fn row_dots_through_the_new_decoders_match_the_reference() {
    for quant in int_chunk_formats().into_iter().chain(float_chunk_formats()) {
        let info = format_info(quant).unwrap();
        let (rows, cols) = (4usize, width(&info));
        let raw = ext_sample(quant, rows, cols, 0x5eed_0003);
        let packed = pack_weight(quant, &raw, rows, cols).unwrap();
        let weights = crate::compute::dequantize(&raw, quant.to_gguf_type_id(), rows * cols).expect("reference");
        let x = unit_vector(cols, 17);
        let bpr = cols / info.qk;
        for r in 0..rows {
            let (mut got, mut mag, mut want) = (0.0f64, 0.0f64, 0.0f64);
            for ch in 0..cols / 8 {
                let (g, cin) = (r * bpr + ch / (info.qk / 8), ch % (info.qk / 8));
                if info.float_chunk {
                    let v = chunk_f(&packed.layout, &packed.bytes, g, cin);
                    got += (0..8).map(|i| v[i] as f64 * x[ch * 8 + i] as f64).sum::<f64>();
                } else {
                    let c = chunk(&packed.layout, &packed.bytes, g, cin);
                    let (mut dot, mut sum) = (0.0f32, 0.0f32);
                    for i in 0..8 {
                        dot += c.q[i] as f32 * x[ch * 8 + i];
                        sum += x[ch * 8 + i];
                    }
                    got += (c.ws * dot + c.wm * sum) as f64;
                }
            }
            for k in 0..cols {
                let t = weights[r * cols + k] as f64 * x[k] as f64;
                want += t;
                mag += t.abs();
            }
            assert!((got - want).abs() <= 1e-5 * mag + 1e-6, "{quant:?} row {r}: {got} vs {want}");
        }
    }
}

/// Slices of rows decode like the whole weight (FP8 tiles of 128 rows are cut inside the tile; MLX and bitsandbytes cut at file rows).
#[test]
fn row_slices_of_the_new_formats_decode_like_the_whole_weight() {
    for quant in [fp8b(128, 128), fp8b(32, 16), mlx(3, 64, true), mlx(8, 32, false), QuantType::Nf4B64, QuantType::Int8Row] {
        let info = format_info(quant).unwrap();
        let (rows, cols) = (150usize, width(&info).max(64));
        let cols = cols.next_multiple_of(info.src_qk.max(info.qk).max(match info.geometry { Geometry::Fp8b { block_cols, .. } => block_cols, _ => 1 }));
        let raw = ext_sample(quant, rows, cols, 0x5eed_0004);
        let slices = pack_weight_slices(quant, &raw, rows, cols, 64).unwrap();
        assert_eq!(slices.iter().map(|s| (s.row_base, s.packed.layout.rows)).collect::<Vec<_>>(), [(0, 64), (64, 64), (128, 22)], "{quant:?}");
        let want = crate::compute::dequantize(&raw, quant.to_gguf_type_id(), rows * cols).unwrap();
        let bpr = cols / info.qk;
        for s in &slices {
            // the slice made by the upload path (`pack_slice`) is the same image
            let via_upload = crate::backend::vulkan::backend::upload::pack_slice(quant, &raw, rows, cols, s.row_base, s.packed.layout.rows).unwrap();
            assert_eq!(via_upload.bytes, s.packed.bytes, "{quant:?} slice at {}", s.row_base);
            for r in 0..s.packed.layout.rows {
                for blk in 0..bpr {
                    for cin in 0..info.qk / 8 {
                        let g = r * bpr + blk;
                        let values: Vec<f32> = if info.float_chunk {
                            chunk_f(&s.packed.layout, &s.packed.bytes, g, cin).to_vec()
                        } else {
                            let c = chunk(&s.packed.layout, &s.packed.bytes, g, cin);
                            (0..8).map(|i| c.ws * c.q[i] as f32 + c.wm).collect()
                        };
                        for (i, got) in values.iter().enumerate() {
                            let at = (s.row_base + r) * cols + blk * info.qk + cin * 8 + i;
                            assert!((got - want[at]).abs() <= 1e-5 * want[at].abs() + 1e-6, "{quant:?} slice {} row {r} block {blk}: {got} vs {}", s.row_base, want[at]);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn shapes_the_kernels_cannot_take_are_rejected_by_the_layout() {
    // FP8 chunks of 8 columns must not straddle tiles, and a row is a whole number of chunks
    assert!(DeviceLayout::new(fp8b(16, 1), 4, 16).is_err());
    assert!(DeviceLayout::new(fp8b(16, 4), 4, 16).is_err());
    assert!(DeviceLayout::new(fp8b(128, 128), 4, 100).is_err());
    assert!(DeviceLayout::new(fp8b(128, 128), 4, 104).is_ok());
    // a row of MLX codes is a whole number of groups
    assert!(DeviceLayout::new(mlx(4, 64, false), 4, 96).is_err());
    assert!(DeviceLayout::new(mlx(4, 64, false), 4, 128).is_ok());
    // a wrong tensor length is an error, not a short read
    let raw = ext_sample(fp8b(128, 128), 3, 128, 1);
    assert!(pack_weight(fp8b(128, 128), &raw[1..], 3, 128).is_err());
    assert!(pack_weight(fp8b(128, 128), &raw, 4, 128).is_err());
    let raw = ext_sample(mlx(4, 64, false), 3, 128, 1);
    assert!(pack_weight(mlx(4, 64, false), &raw[..raw.len() - 1], 3, 128).is_err());
}

#[test]
fn float_chunk_formats_plan_only_the_f32_shaders_and_integer_formats_all_variants() {
    let caps = VkCapsLike { int_dot: true, ..VkCapsLike::default() };
    let lossy = lossy_options();
    for quant in [fp8b(128, 128), fp8b(32, 16), QuantType::Nf4B64, QuantType::Nf4B128, QuantType::Fp4B64, QuantType::Fp4B128] {
        let info = format_info(quant).unwrap();
        let cols = width(&info);
        for batch in [1usize, 8, 9, 64, 256] {
            let plan = plan_quantized_linear_with(quant, 70, cols, batch, &caps, &lossy).unwrap();
            assert_eq!(plan.variant, Variant::Scalar, "{quant:?} batch {batch}");
            assert_eq!(plan.shader_name, format!("{}_{}_s", if batch <= 8 { "mv" } else { "mm" }, info.tag));
            assert!(plan.activation_prepass.is_none());
        }
    }
    for quant in [mlx(4, 64, false), mlx(3, 32, true), mlx(8, 128, false), QuantType::Int8Row] {
        let info = format_info(quant).unwrap();
        let cols = width(&info);
        let matvec = plan_quantized_linear_with(quant, 70, cols, 3, &caps, &lossy).unwrap();
        assert_eq!(matvec.shader_name, format!("mv_{}_i", info.tag));
        let tiled = plan_quantized_linear_with(quant, 70, cols, 64, &caps, &lossy).unwrap();
        assert_eq!(tiled.shader_name, format!("mmq_{}_i", info.tag));
        let exact = plan_quantized_linear_with(quant, 70, cols, 64, &caps, &PlanOptions { exact: true, ..PlanOptions::default() }).unwrap();
        assert_eq!(exact.shader_name, format!("mm_{}_s", info.tag));
    }
}

/// The format parameter travels as push constant `off2` (word 7 of `QPush`).
#[test]
fn the_format_parameter_is_the_third_offset_of_the_push_constants() {
    let caps = VkCapsLike::default();
    let word = |bytes: &[u8], i: usize| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
    let plan = plan_quantized_linear(fp8b(128, 32), 70, 256, 1, &caps).unwrap();
    assert_eq!(word(&plan.push_constants, 7), 32);
    assert_eq!(word(&plan.push_constants, 5), 256 / 8, "blocks per row");
    assert_eq!(word(&plan.push_constants, 6), plan.layout.plane_word_offset[1]);
    let plan = plan_quantized_linear(mlx(4, 128, true), 70, 256, 1, &caps).unwrap();
    assert_eq!(word(&plan.push_constants, 7), 4 | (1 << 16));
    assert_eq!(word(&plan.push_constants, 5), 256 / 32);
    let plan = plan_quantized_linear(mlx(4, 64, false), 70, 256, 1, &caps).unwrap();
    assert_eq!(word(&plan.push_constants, 7), 2);
    // a planar format still gets the offset of its plane 2
    let plan = plan_quantized_linear(QuantType::Q5_0, 70, 256, 1, &caps).unwrap();
    assert_eq!(word(&plan.push_constants, 7), plan.layout.plane_word_offset[2]);
}

#[test]
fn bitsandbytes_blocks_split_into_codes_and_scales() {
    for (quant, kind) in [(QuantType::Nf4B64, bnb::NF4_B64), (QuantType::Fp4B128, bnb::FP4_B128), (QuantType::Int8Row, bnb::INT8_ROW)] {
        let block = bnb::layout(kind).unwrap();
        let (rows, cols) = (3usize, block.block * 2);
        let raw = ext_sample(quant, rows, cols, 9);
        let packed = pack_weight(quant, &raw, rows, cols).unwrap();
        let (codes, scales) = (packed.layout.plane_byte_offset(0), packed.layout.plane_byte_offset(1));
        for (g, b) in raw.chunks_exact(block.bytes).enumerate() {
            assert_eq!(&packed.bytes[codes + g * (block.bytes - 4)..codes + (g + 1) * (block.bytes - 4)], &b[4..], "{quant:?} codes of block {g}");
            assert_eq!(&packed.bytes[scales + g * 4..scales + g * 4 + 4], &b[..4], "{quant:?} scale of block {g}");
        }
    }
}
