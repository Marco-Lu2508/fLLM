//! CPU tests of the Metal weight layouts and the dispatch planner (no GPU, no Metal objects).
//!
//! The decoders of the kernels themselves (`metal/common.metal`) are checked by `metal/emu` (the MSL text run as C++); the dump test
//! at the end of this file writes the vectors that check reads.

use super::sample::{reference_weights, sample_weights, Lcg};
use super::*;
use crate::backend::QuantType;
use half::f16;
use std::path::PathBuf;

/// GGUF type of the blocks of `quant` (the Q4ZP formats store the Q4Z blocks followed by a permutation table).
fn block_kind(quant: QuantType) -> u32 {
    let id = quant.to_gguf_type_id();
    if quant.is_q4zp() {
        id - 4
    } else {
        id
    }
}

fn dims(info: &FormatInfo) -> (usize, usize) {
    (9, info.qk.max(8) * 3)
}

// ---- Rust mirrors of the plane decoders of common.metal (the repacked formats) ---------------------------------------------------

const FP4: [i32; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

fn e8m0_half(x: u8) -> f32 {
    f32::from_bits(if x < 2 {
        0x0020_0000u32 << x
    } else {
        (x as u32 - 1) << 23
    })
}

/// `q_fmt_mxfp4::dec8`
fn dec8_mxfp4(image: &[u8], plane1: usize, g: usize, c: usize) -> [f32; 8] {
    let d = e8m0_half(image[plane1 + g]);
    let mut o = [0.0; 8];
    for (k, v) in o.iter_mut().enumerate() {
        let byte = image[g * 16 + (c & 1) * 8 + k];
        let nib = if c >> 1 == 0 { byte & 15 } else { byte >> 4 };
        *v = FP4[nib as usize] as f32 * d;
    }
    o
}

/// `q_fmt_q4z<GROUP>::dec8`
fn dec8_q4z(image: &[u8], plane1: usize, group: usize, g: usize, c: usize) -> [f32; 8] {
    let word = |at: usize| u32::from_le_bytes(image[at..at + 4].try_into().unwrap());
    let u = word(g * group / 2 + c * 4);
    let sz = word(plane1 + g * 4);
    let scale = f16::from_bits((sz & 0xffff) as u16).to_f32();
    let zero = ((sz >> 16) & 15) as f32;
    let mut o = [0.0; 8];
    for (k, v) in o.iter_mut().enumerate() {
        *v = (((u >> (4 * k)) & 15) as f32 - zero) * scale;
    }
    o
}

/// `q_fmt_mlx<BITS, GROUP>::dec8`; `bf16`: bit 63 of the plane-1 offset (`MLX_BF16_FLAG`)
fn dec8_mlx(image: &[u8], plane1: usize, bits: usize, group: usize, bf16: bool, g: usize, c: usize) -> [f32; 8] {
    let word = u32::from_le_bytes(image[plane1 + g * 4..plane1 + g * 4 + 4].try_into().unwrap());
    let convert = |raw: u32| if bf16 { f32::from_bits(raw << 16) } else { f16::from_bits(raw as u16).to_f32() };
    let (scale, bias) = (convert(word & 0xffff), convert(word >> 16));
    let start = g * (group * bits / 8) + c * bits;
    let mut v = 0u64;
    for (i, &byte) in image[start..start + bits].iter().enumerate() {
        v |= u64::from(byte) << (8 * i);
    }
    std::array::from_fn(|k| ((v >> (bits * k)) & ((1u64 << bits) - 1)) as f32 * scale + bias)
}

#[test]
fn every_format_has_consistent_geometry() {
    for quant in supported_formats() {
        let info = format_info(quant).unwrap();
        assert!(info.qk.is_multiple_of(8), "{quant:?}");
        // block geometry of the file against the engine's own table (the dense formats use 8-value pseudo blocks)
        if info.src_qk > 8 {
            assert_eq!(info.src_qk, quant.block_size(), "{quant:?}");
            assert_eq!(info.src_block_bytes, quant.block_bytes(), "{quant:?}");
        }
        assert_eq!(info.src_qk, info.qk);
        // kernel tag round trip: the tag names a kernel in every family file
        assert!(!info.tag.is_empty());
        let (rows, cols) = dims(&info);
        let planes = device_layout(quant, rows, cols).unwrap();
        if planes.layout == DeviceLayout::Planar {
            assert_eq!(planes.plane1 % 16, 0, "{quant:?}");
            assert_eq!(planes.total_bytes % 16, 0, "{quant:?}");
        }
        assert!(format_info(quant).is_some());
    }
    assert!(format_info(QuantType::IQ2_XS).is_none());
    assert!(format_info(QuantType::Q8_K).is_none());
    assert!(format_info(QuantType::Nvfp4Marlin).is_none());
}

#[test]
fn raw_formats_are_borrowed_unchanged() {
    for quant in supported_formats() {
        let info = format_info(quant).unwrap();
        if info.repack != super::format::Repack::None {
            continue;
        }
        let (rows, cols) = dims(&info);
        let raw = sample_weights(quant, rows, cols, 5);
        assert_eq!(
            raw.len(),
            expected_raw_len(quant, rows, cols).unwrap(),
            "{quant:?}"
        );
        let packed = pack_weight(quant, rows, cols, &raw).unwrap();
        assert!(
            matches!(packed.bytes, std::borrow::Cow::Borrowed(_)),
            "{quant:?} must not copy"
        );
        assert_eq!(&packed.bytes[..], &raw[..], "{quant:?}");
        assert_eq!(packed.planes.layout, DeviceLayout::Raw);
        assert_eq!(packed.planes.total_bytes as usize, raw.len());
        assert!(packed.perm.is_none());
        assert_eq!(
            packed.planes.plane1,
            if info.row_scales || info.tiled {
                (rows * cols) as u64
            } else {
                0
            }
        );
    }
}

#[test]
fn raw_alignment_assumptions_of_the_kernels_hold() {
    // formats read with 32 bit loads need a block size that is a multiple of 4 (so every block of an aligned image is aligned)
    for quant in supported_formats() {
        let info = format_info(quant).unwrap();
        if info.repack != super::format::Repack::None {
            continue;
        }
        let wide = matches!(
            quant,
            QuantType::Q4_1
                | QuantType::Q5_1
                | QuantType::Q2_K
                | QuantType::Q4_K
                | QuantType::Q5_K
                | QuantType::NvFp4
                | QuantType::IQ4_XS
                | QuantType::F32
                | QuantType::F16
                | QuantType::BF16
                | QuantType::Fp8E4m3
                | QuantType::Fp8E5m2
                | QuantType::Fp8Marlin
                | QuantType::Fp8Block
                | QuantType::Nf4B64
                | QuantType::Nf4B128
                | QuantType::Fp4B64
                | QuantType::Fp4B128
                | QuantType::Int8Row
        );
        if wide {
            assert_eq!(info.block_bytes % 4, 0, "{quant:?}");
        } else {
            assert_eq!(info.block_bytes % 2, 0, "{quant:?}");
        }
    }
}

#[test]
fn repacked_planes_decode_to_the_reference() {
    let mut repacked = vec![
        QuantType::Mxfp4,
        QuantType::Q4Z32,
        QuantType::Q4Z64,
        QuantType::Q4Z128,
        QuantType::Q4ZP32,
        QuantType::Q4ZP64,
        QuantType::Q4ZP128,
    ];
    repacked.extend(supported_formats().into_iter().filter(|q| q.mlx_affine().is_some()));
    for quant in repacked {
        let info = format_info(quant).unwrap();
        let (rows, cols) = (7usize, info.qk * 3);
        let raw = sample_weights(quant, rows, cols, 77);
        let blocks_len = rows * cols / info.qk * info.src_block_bytes;
        let want =
            crate::compute::dequantize(&raw[..blocks_len], block_kind(quant), rows * cols).unwrap();
        let packed = pack_weight(quant, rows, cols, &raw).unwrap();
        assert_eq!(packed.planes.layout, DeviceLayout::Planar);
        assert!(matches!(packed.bytes, std::borrow::Cow::Owned(_)));
        assert_eq!(packed.bytes.len() as u64, packed.planes.total_bytes);
        let plane1 = packed.planes.plane1 as usize;
        let nb = cols / info.qk;
        for r in 0..rows {
            for b in 0..nb {
                for c in 0..info.qk / 8 {
                    let got = if quant == QuantType::Mxfp4 {
                        dec8_mxfp4(&packed.bytes, plane1, r * nb + b, c)
                    } else if let Some(m) = quant.mlx_affine() {
                        dec8_mlx(&packed.bytes, plane1, m.bits as usize, m.group as usize, m.bf16, r * nb + b, c)
                    } else {
                        dec8_q4z(&packed.bytes, plane1, info.qk, r * nb + b, c)
                    };
                    for (i, value) in got.iter().enumerate() {
                        let at = r * cols + b * info.qk + c * 8 + i;
                        assert_eq!(
                            *value, want[at],
                            "{quant:?} row {r} block {b} chunk {c} value {i}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn q4zp_permutation_is_split_off_and_validated() {
    let (rows, cols) = (3usize, 128usize);
    let raw = sample_weights(QuantType::Q4ZP64, rows, cols, 7);
    let packed = pack_weight(QuantType::Q4ZP64, rows, cols, &raw).unwrap();
    let perm = packed.perm.expect("permutation");
    assert_eq!(perm.len(), cols);
    let mut sorted = perm.clone();
    sorted.sort_unstable();
    assert!(sorted.iter().enumerate().all(|(i, &p)| i as u32 == p));
    // out of range
    let n = raw.len();
    let mut bad = raw.clone();
    bad[n - 4..].copy_from_slice(&(cols as u32).to_le_bytes());
    assert!(matches!(
        pack_weight(QuantType::Q4ZP64, rows, cols, &bad),
        Err(WeightError::BadPermutation(_))
    ));
    // duplicate
    let mut dup = raw.clone();
    let first: [u8; 4] = dup[n - cols * 4..n - cols * 4 + 4].try_into().unwrap();
    dup[n - 4..].copy_from_slice(&first);
    assert!(matches!(
        pack_weight(QuantType::Q4ZP64, rows, cols, &dup),
        Err(WeightError::BadPermutation(_))
    ));
    // missing table / Q4Z with a table
    assert!(matches!(
        pack_weight(QuantType::Q4ZP64, rows, cols, &raw[..n - 4]),
        Err(WeightError::BadLength { .. })
    ));
    assert!(pack_weight(QuantType::Q4Z64, rows, cols, &raw).is_err());
}

#[test]
fn bad_shapes_and_lengths_are_rejected() {
    assert!(matches!(
        pack_weight(QuantType::Q4_K, 4, 200, &[]),
        Err(WeightError::BadShape { .. })
    ));
    assert!(matches!(
        pack_weight(QuantType::Q4_K, 0, 256, &[]),
        Err(WeightError::BadShape { .. })
    ));
    assert!(matches!(
        pack_weight(QuantType::F16, 2, 12, &[0u8; 48]),
        Err(WeightError::BadShape { .. })
    )); // cols % 8
    assert!(matches!(
        pack_weight(QuantType::IQ2_XS, 4, 256, &[]),
        Err(WeightError::UnsupportedFormat(_))
    ));
    assert!(matches!(
        pack_weight(QuantType::Q4_0, 1, 32, &[0u8; 17]),
        Err(WeightError::BadLength {
            expected: 18,
            got: 17,
            ..
        })
    ));
    assert!(matches!(
        expected_raw_len(QuantType::Q8_K, 1, 256),
        Err(WeightError::UnsupportedFormat(_))
    ));
}

#[test]
fn fp8m_keeps_its_row_scales_behind_the_codes() {
    let (rows, cols) = (4usize, 16usize);
    let raw = sample_weights(QuantType::Fp8Marlin, rows, cols, 3);
    assert_eq!(raw.len(), rows * cols + rows * 2);
    let packed = pack_weight(QuantType::Fp8Marlin, rows, cols, &raw).unwrap();
    assert_eq!(packed.planes.plane1 as usize, rows * cols);
    assert_eq!(packed.bytes.len(), raw.len());
    let want = reference_weights(QuantType::Fp8Marlin, &raw, rows, cols).unwrap();
    // E4M3 codes times the row scale in the epilogue = the reference
    let scale = |r: usize| {
        f16::from_le_bytes([raw[rows * cols + 2 * r], raw[rows * cols + 2 * r + 1]]).to_f32()
    };
    for r in 0..rows {
        for c in 0..cols {
            let code = raw[r * cols + c];
            let e = ((code >> 3) & 15) as i32;
            let m = (code & 7) as f32;
            let v = if e == 0 {
                m / 512.0
            } else {
                (1.0 + m / 8.0) * 2f32.powi(e - 7)
            };
            let v = if code & 0x80 != 0 { -v } else { v };
            assert_eq!(v * scale(r), want[r * cols + c]);
        }
    }
}

// ---- planner -----------------------------------------------------------------------------------------------------------------

fn metal_dir() -> PathBuf {
    match std::env::var_os("FLLM_METAL_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("metal"),
    }
}

fn kernel_sources() -> String {
    let mut files: Vec<_> = std::fs::read_dir(metal_dir())
        .expect("rust/engine/metal")
        .map(|e| e.unwrap().path())
        .filter(|p| {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            name.starts_with("q_") && name.ends_with(".metal")
        })
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no q_*.metal in {:?}", metal_dir());
    files
        .iter()
        .map(|p| std::fs::read_to_string(p).unwrap())
        .collect()
}

#[test]
fn planned_kernels_exist_in_the_metal_sources() {
    let text = kernel_sources();
    for name in all_kernel_names() {
        let declared = text.contains(&format!("host_name(\"{name}\")"))
            || text.contains(&format!("kernel void {name}("));
        assert!(
            declared,
            "kernel {name} is planned but not instantiated in metal/q_*.metal (run q_gen.py)"
        );
    }
    // and the other way round: no orphan instantiation
    let known = all_kernel_names();
    for part in text.split("host_name(\"").skip(1) {
        let name = part.split('"').next().unwrap();
        assert!(
            known.iter().any(|k| k == name),
            "metal/q_*.metal defines {name}, which the planner does not know"
        );
    }
}

#[test]
fn qparams_layout_matches_the_msl_struct() {
    assert_eq!(std::mem::size_of::<QParams>(), QParams::SIZE);
    assert_eq!(std::mem::align_of::<QParams>(), 8);
    let p = QParams {
        plane1: 0x0102_0304_0506_0708,
        rows: 1,
        cols: 2,
        batch: 3,
        flags: 3,
        scale: 0.5,
        out_stride: 6,
        row_base: 7,
        x_stride: 8,
        nb_blocks: 9,
        pad: 0,
    };
    let b = p.to_bytes();
    assert_eq!(&b[0..8], &0x0102_0304_0506_0708u64.to_le_bytes());
    assert_eq!(&b[8..12], &1u32.to_le_bytes());
    assert_eq!(&b[24..28], &0.5f32.to_le_bytes());
    assert_eq!(&b[44..48], &0u32.to_le_bytes());
    assert_eq!(std::mem::size_of::<PermuteParams>(), 16);
    // the MSL side: same field order
    let common = std::fs::read_to_string(metal_dir().join("common.metal")).unwrap();
    let block = common
        .split("struct q_params {")
        .nth(1)
        .unwrap()
        .split('}')
        .next()
        .unwrap();
    let names: Vec<&str> = block
        .lines()
        .filter_map(|l| l.trim().split("//").next())
        .filter(|l| l.trim_end().ends_with(';'))
        .map(|l| {
            l.trim()
                .trim_end_matches(';')
                .split_whitespace()
                .last()
                .unwrap()
        })
        .collect();
    assert_eq!(
        names,
        [
            "plane1",
            "rows",
            "cols",
            "batch",
            "flags",
            "scale",
            "out_stride",
            "row_base",
            "x_stride",
            "nb_blocks",
            "pad"
        ]
    );
}

fn caps() -> MetalCapsLike {
    MetalCapsLike::default()
}

#[test]
fn matvec_and_matmul_selection() {
    let c = caps();
    for batch in 1..=8usize {
        let p = plan_quantized_linear(QuantType::Q4_K, 4096, 4096, batch, &c).unwrap();
        assert_eq!(p.op, LinearOp::MatVec);
        assert_eq!(
            p.constants,
            [PlanConstant {
                index: FC_NB,
                value: batch as u32
            }]
        );
        assert_eq!(
            p.kernel_name,
            format!("fllm_q_mv_q4_k_{}", if batch <= 4 { "r4" } else { "r2" })
        );
        assert_eq!(p.threads_per_group, [64, 1, 1]);
        assert_eq!(
            p.threadgroups,
            [4096 / if batch <= 4 { 8 } else { 4 }, 1, 1]
        );
    }
    let p = plan_quantized_linear(QuantType::Q4_K, 4096, 4096, 9, &c).unwrap();
    assert_eq!(p.op, LinearOp::MatMul);
    assert_eq!(p.kernel_name, "fllm_q_mm_q4_k_h");
    assert_eq!(p.threadgroups, [64, 1, 1]);
    assert_eq!(p.threads_per_group, [128, 1, 1]);
    assert_eq!(p.threadgroup_memory, [(0, 14336)]);
    assert!(p.constants.is_empty() && p.half_tiles);
    let p = plan_quantized_linear(QuantType::Q4_K, 100, 256, 33, &c).unwrap();
    assert_eq!(p.threadgroups, [2, 2, 1]);
    // exact mode and F32 weights use float tiles
    let exact = MetalCapsLike { exact: true, ..c };
    let p = plan_quantized_linear(QuantType::Q4_K, 64, 256, 16, &exact).unwrap();
    assert_eq!(
        (
            p.kernel_name.as_str(),
            p.threadgroup_memory.clone(),
            p.half_tiles
        ),
        ("fllm_q_mm_q4_k_f", vec![(0, 20480)], false)
    );
    let p = plan_quantized_linear(QuantType::F32, 64, 256, 16, &c).unwrap();
    assert_eq!(p.kernel_name, "fllm_q_mm_f32_f");
    // without simdgroup_matrix a large batch becomes several columns of the matrix-vector kernel
    let none = MetalCapsLike {
        simdgroup_matrix: false,
        ..c
    };
    let p = plan_quantized_linear(QuantType::Q8_0, 64, 256, 21, &none).unwrap();
    assert_eq!(p.op, LinearOp::MatVec);
    assert_eq!(
        p.constants,
        [PlanConstant {
            index: FC_NB,
            value: 8
        }]
    );
    assert_eq!(p.threadgroups, [16, 3, 1]);
    // a simd width other than 32 cannot run the matrix-matrix kernels
    let wide = MetalCapsLike {
        simd_width: 64,
        ..c
    };
    let p = plan_quantized_linear(QuantType::Q8_0, 64, 256, 40, &wide).unwrap();
    assert_eq!((p.op, p.threads_per_group), (LinearOp::MatVec, [128, 1, 1]));
    // a device that cannot even run two simdgroups
    let small = MetalCapsLike {
        max_threads: 32,
        ..c
    };
    assert!(matches!(
        plan_quantized_linear(QuantType::Q8_0, 64, 256, 1, &small),
        Err(PlanError::Device(_))
    ));
}

#[test]
fn parameters_and_builders() {
    let p = plan_quantized_linear(QuantType::Mxfp4, 40, 96, 2, &caps()).unwrap();
    assert_eq!(p.params.rows, 40);
    assert_eq!(p.params.cols, 96);
    assert_eq!(p.params.batch, 2);
    assert_eq!(p.params.nb_blocks, 3);
    assert_eq!(p.params.out_stride, 40);
    assert_eq!(p.params.x_stride, 96);
    assert_eq!(p.params.scale, 1.0);
    assert_eq!(
        p.params.plane1,
        device_layout(QuantType::Mxfp4, 40, 96).unwrap().plane1
    );
    assert!(!p.reads_output());
    let p = p
        .with_bias()
        .with_residual()
        .with_scale(0.25)
        .with_output(8, 64);
    assert_eq!(p.params.flags, FLAG_BIAS | FLAG_RESIDUAL);
    assert!(p.reads_output());
    assert_eq!(
        (p.params.scale, p.params.row_base, p.params.out_stride),
        (0.25, 8, 64)
    );
    assert_eq!(
        p.slots,
        BufferSlots {
            weights: 0,
            x: 1,
            out: 2,
            bias: 3,
            params: 4,
            threadgroup_memory: 0
        }
    );
    // dense formats count 8-value blocks
    let p = plan_quantized_linear(QuantType::F16, 5, 24, 1, &caps()).unwrap();
    assert_eq!(p.params.nb_blocks, 3);
    // input stride: the main kernel, or only the prepass of act-order weights
    let p = p.with_input_stride(64);
    assert_eq!(p.params.x_stride, 64);
    let p = plan_quantized_linear(QuantType::Q4ZP32, 8, 64, 2, &caps())
        .unwrap()
        .with_input_stride(128);
    assert_eq!(p.params.x_stride, 64);
    assert_eq!(p.activation_prepass.unwrap().params.x_stride, 128);
}

#[test]
fn act_order_weights_get_a_prepass() {
    let p = plan_quantized_linear(QuantType::Q4ZP64, 64, 256, 3, &caps()).unwrap();
    let pre = p.activation_prepass.expect("prepass");
    assert_eq!(pre.kernel_name, "fllm_q_permute_x");
    assert_eq!(pre.output_bytes, 3 * 256 * 4);
    assert_eq!(pre.threadgroups, [3, 1, 1]);
    assert_eq!(
        pre.params,
        PermuteParams {
            batch: 3,
            cols: 256,
            x_stride: 256,
            pad: 0
        }
    );
    assert_eq!(p.kernel_name, "fllm_q_mv_q4z64_r4");
    assert!(plan_quantized_linear(QuantType::Q4Z64, 64, 256, 3, &caps())
        .unwrap()
        .activation_prepass
        .is_none());
}

#[test]
fn planner_rejects_bad_requests() {
    let c = caps();
    assert!(matches!(
        plan_quantized_linear(QuantType::Q4_0, 8, 32, 0, &c),
        Err(PlanError::BadSize(_))
    ));
    assert!(matches!(
        plan_quantized_linear(QuantType::Q4_0, 8, 40, 1, &c),
        Err(PlanError::Weight(WeightError::BadShape { .. }))
    ));
    assert!(matches!(
        plan_quantized_linear(QuantType::IQ3_S, 8, 256, 1, &c),
        Err(PlanError::Weight(WeightError::UnsupportedFormat(_)))
    ));
    assert!(matches!(
        plan_quantized_linear(
            QuantType::Q4_0,
            8,
            32,
            1,
            &MetalCapsLike { simd_width: 0, ..c }
        ),
        Err(PlanError::Device(_))
    ));
    assert!(matches!(
        plan_quantized_linear(QuantType::Q4_0, 8, 4096, usize::MAX / 2, &c),
        Err(PlanError::BadSize(_))
    ));
}

/// Every dispatch covers all rows and all tokens exactly once, within the device limits (randomised).
#[test]
fn grids_cover_the_output_exactly() {
    let mut rng = Lcg(2024);
    let formats = supported_formats();
    for _ in 0..4000 {
        let quant = formats[(rng.next_u64() as usize) % formats.len()];
        let info = format_info(quant).unwrap();
        let rows = 1 + (rng.next_u64() as usize) % 700;
        let cols = info.qk * (1 + (rng.next_u64() as usize) % 12);
        let batch = 1 + (rng.next_u64() as usize) % 90;
        let c = MetalCapsLike {
            simdgroup_matrix: !rng.next_u64().is_multiple_of(4),
            exact: rng.next_u64().is_multiple_of(3),
            ..caps()
        };
        let p = plan_quantized_linear(quant, rows, cols, batch, &c).unwrap();
        let [gx, gy, gz] = p.threadgroups;
        assert_eq!(gz, 1);
        let threads: u32 = p.threads_per_group.iter().product();
        assert!(threads <= c.max_threads);
        let shared: u32 = p.threadgroup_memory.iter().map(|m| m.1).sum();
        assert!(shared <= 32768, "{shared}");
        let (rows_per_group, tokens_per_group) = match p.op {
            LinearOp::MatMul => (64usize, 32usize),
            LinearOp::MatVec => {
                let nr = if p.kernel_name.ends_with("_r4") { 4 } else { 2 };
                (2 * nr, p.constants[0].value as usize)
            }
        };
        assert!(
            gx as usize * rows_per_group >= rows && (gx as usize - 1) * rows_per_group < rows,
            "{} rows {rows}",
            p.kernel_name
        );
        assert!(
            gy as usize * tokens_per_group >= batch && (gy as usize - 1) * tokens_per_group < batch,
            "{} batch {batch}",
            p.kernel_name
        );
        assert_eq!(p.params.nb_blocks as usize * info.qk, cols);
        assert!(all_kernel_names().contains(&p.kernel_name));
    }
}

#[test]
fn mlx_formats_agree_with_the_type_ids_of_the_checkpoint_and_plan_their_kernels() {
    let mlx: Vec<QuantType> = supported_formats().into_iter().filter(|q| q.mlx_affine().is_some()).collect();
    // 6 widths x 3 group sizes x 2 scale formats
    assert_eq!(mlx.len(), 36);
    assert_eq!(mlx, QuantType::mlx_affine_common(), "the capability list and the table are the same set");
    for quant in mlx {
        let m = quant.mlx_affine().unwrap();
        let info = format_info(quant).unwrap();
        assert_eq!(quant.to_gguf_type_id(), m.type_id());
        assert_eq!(info.tag, format!("mlx{}g{}", m.bits, m.group));
        assert_eq!(info.repack, super::format::Repack::Mlx);
        assert_eq!(info.block_bytes, m.group as usize * m.bits as usize / 8);
        assert_eq!(info.src_block_bytes, m.block_bytes() as usize);
        // the kernel planes of 7 rows of 3 groups
        let (rows, cols) = (7usize, m.group as usize * 3);
        let planes = device_layout(quant, rows, cols).unwrap();
        assert_eq!(planes.layout, DeviceLayout::Planar);
        assert_eq!(planes.plane1 % 16, 0);
        let plan = plan_quantized_linear(quant, rows, cols, 1, &caps()).unwrap();
        assert_eq!(plan.kernel_name, format!("fllm_q_mv_mlx{}g{}_r4", m.bits, m.group));
        // bfloat16 scales are flagged in bit 63 of plane 1, which stays the offset in the low bits
        assert_eq!(plan.params.plane1 & !MLX_BF16_FLAG, planes.plane1);
        assert_eq!(plan.params.plane1 & MLX_BF16_FLAG != 0, m.bf16);
        let mm = plan_quantized_linear(quant, rows, cols, 33, &caps()).unwrap();
        assert_eq!(mm.kernel_name, format!("fllm_q_mm_mlx{}g{}_h", m.bits, m.group));
        assert!(all_kernel_names().contains(&plan.kernel_name) && all_kernel_names().contains(&mm.kernel_name));
    }
}

#[test]
fn mlx_planes_hold_every_code_scale_and_bias_of_the_file_blocks() {
    for bits in [2u32, 3, 4, 5, 6, 8] {
        for (group, bf16) in [(32u32, false), (64, true), (128, false)] {
            let quant = QuantType::Other(fllm_gguf_probe::MlxAffine { bits, group, bf16 }.type_id());
            let (rows, cols) = (5usize, group as usize * 2);
            let raw = sample_weights(quant, rows, cols, 99);
            let packed = pack_weight(quant, rows, cols, &raw).unwrap();
            let block = group as usize * bits as usize / 8 + 4;
            let plane1 = packed.planes.plane1 as usize;
            for g in 0..rows * cols / group as usize {
                // the 4 header bytes of the file block are the plane-1 word, the rest are the codes in place
                assert_eq!(&packed.bytes[plane1 + 4 * g..plane1 + 4 * g + 4], &raw[g * block..g * block + 4], "{bits} bits group {g}");
                let codes = block - 4;
                assert_eq!(&packed.bytes[g * codes..(g + 1) * codes], &raw[g * block + 4..(g + 1) * block], "{bits} bits group {g}");
            }
            assert!(packed.perm.is_none());
            assert!(pack_weight(quant, rows, cols, &raw[1..]).is_err(), "wrong length");
            assert!(pack_weight(quant, rows, group as usize * 2 + 8, &raw).is_err(), "columns are no whole number of groups");
        }
    }
}

// ---- bitsandbytes and block-wise FP8: Rust mirrors of the decoders of common.metal ------------------------------------------------

/// The `uint[16]` initializer of the program-scope table `name` in common.metal.
fn msl_table(name: &str) -> Vec<u32> {
    let common = std::fs::read_to_string(metal_dir().join("common.metal")).unwrap();
    let at = common.find(&format!("constant uint {name}[16] = {{")).unwrap_or_else(|| panic!("table {name}"));
    let body = common[at..].split('{').nth(1).unwrap().split('}').next().unwrap();
    body.split(',')
        .map(|t| t.trim().trim_end_matches('u'))
        .filter(|t| !t.is_empty())
        .map(|t| u32::from_str_radix(t.trim_start_matches("0x"), 16).unwrap())
        .collect()
}

fn word_at(image: &[u8], at: usize) -> u32 {
    assert_eq!(at % 4, 0, "the kernels issue 32 bit loads here");
    u32::from_le_bytes(image[at..at + 4].try_into().unwrap())
}

/// `q_fmt_bnb4<BLOCK, FP4>::dec8` (`book`: the bit patterns of the table the format reads)
fn dec8_bnb4(image: &[u8], block: usize, book: &[u32], g: usize, c: usize) -> [f32; 8] {
    let bytes = 4 + block / 2;
    let b = g * bytes;
    let d = f32::from_bits(word_at(image, b));
    let u = word_at(image, b + 4 + 4 * c);
    let mut o = [0.0; 8];
    for k in 0..4 {
        let byte = ((u >> (8 * k)) & 255) as usize;
        o[2 * k] = f32::from_bits(book[byte >> 4]) * d;
        o[2 * k + 1] = f32::from_bits(book[byte & 15]) * d;
    }
    o
}

/// `q_fmt_int8r::dec8` (`q_s8` is the sign extension of the low byte)
fn dec8_int8r(image: &[u8], g: usize, c: usize) -> [f32; 8] {
    let b = g * 68;
    let d = f32::from_bits(word_at(image, b));
    let mut o = [0.0; 8];
    for i in 0..2 {
        let u = word_at(image, b + 4 + 8 * c + 4 * i);
        for k in 0..4 {
            o[4 * i + k] = f32::from((u >> (8 * k)) as u8 as i8) * d;
        }
    }
    o
}

/// `q_e4m3`
fn msl_e4m3(x: u32) -> f32 {
    let (e, m) = ((x >> 3) & 15, x & 7);
    let v = if e == 0 {
        m as f32 * (1.0 / 512.0)
    } else if e == 15 && m == 7 {
        f32::from_bits(0x7FC0_0000)
    } else {
        f32::from_bits(((e + 120) << 23) | (m << 20))
    };
    if x & 0x80 != 0 {
        -v
    } else {
        v
    }
}

/// `q_fmt_fp8b::dec8p` with the parameter block the planner built
fn dec8p_fp8b(image: &[u8], row: usize, blk: usize, p: &QParams) -> [f32; 8] {
    let (lbr, lbc) = (p.pad & 0xff, (p.pad >> 8) & 0xff);
    let col = blk * 8;
    let tiles_per_row = (p.cols as usize + ((1usize << lbc) - 1)) >> lbc;
    let scales = p.plane1 as usize + (row >> lbr) * tiles_per_row * 4;
    let base = row * p.cols as usize + col;
    assert_eq!(base % 4, 0, "the kernels issue 32 bit loads here");
    std::array::from_fn(|at| {
        let tile = if lbc >= 3 { col >> lbc } else { (col + at) >> lbc };
        let s = f32::from_bits(word_at(image, scales + 4 * tile));
        msl_e4m3(u32::from(image[base + at])) * s
    })
}

#[test]
fn bnb_code_books_of_the_kernels_are_the_rust_books_bit_for_bit() {
    use fllm_gguf_probe::bnb;
    let bits = |book: &[f32; 16]| book.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(msl_table("q_kv_nf4"), bits(&bnb::NF4_CODE));
    assert_eq!(msl_table("q_kv_bnbfp4"), bits(&bnb::FP4_CODE));
}

#[test]
fn bnb_formats_decode_chunk_by_chunk_to_the_reference() {
    use fllm_gguf_probe::bnb;
    let nf4 = msl_table("q_kv_nf4");
    let fp4 = msl_table("q_kv_bnbfp4");
    let formats = [
        (QuantType::Nf4B64, 64, 0),
        (QuantType::Nf4B128, 128, 0),
        (QuantType::Fp4B64, 64, 1),
        (QuantType::Fp4B128, 128, 1),
        (QuantType::Int8Row, 64, 2),
    ];
    for (quant, block, kind) in formats {
        let info = format_info(quant).unwrap();
        assert_eq!((info.qk, info.src_qk, info.family), (block, block, Family::Bnb));
        assert_eq!(info.block_bytes, quant.bnb_layout().unwrap().bytes);
        assert_eq!(info.block_bytes % 4, 0);
        let (rows, cols) = (5usize, block * 3);
        let raw = sample_weights(quant, rows, cols, 41);
        let want = reference_weights(quant, &raw, rows, cols).unwrap();
        let packed = pack_weight(quant, rows, cols, &raw).unwrap();
        assert!(matches!(packed.bytes, std::borrow::Cow::Borrowed(_)), "{quant:?} is used as stored");
        // the loop of the kernel bodies: chunk ci of a row is block ci / CPB, chunk ci % CPB of it
        let cpb = block / 8;
        let nb = cols / block;
        for r in 0..rows {
            for ci in 0..cols / 8 {
                let (blk, c) = (ci / cpb, ci % cpb);
                let g = r * nb + blk;
                let got = match kind {
                    0 => dec8_bnb4(&packed.bytes, block, &nf4, g, c),
                    1 => dec8_bnb4(&packed.bytes, block, &fp4, g, c),
                    _ => dec8_int8r(&packed.bytes, g, c),
                };
                for (k, v) in got.iter().enumerate() {
                    let w = want[r * cols + ci * 8 + k];
                    assert_eq!(v.to_bits(), w.to_bits(), "{quant:?} row {r} chunk {ci} value {k}: {v} vs {w}");
                }
            }
        }
    }
    // the plans name the kernels of the family
    for (quant, tag) in [(QuantType::Nf4B64, "nf4b64"), (QuantType::Fp4B128, "fp4b128"), (QuantType::Int8Row, "int8r")] {
        let plan = plan_quantized_linear(quant, 8, 256, 1, &caps()).unwrap();
        assert_eq!(plan.kernel_name, format!("fllm_q_mv_{tag}_r4"));
        assert_eq!(plan.params.pad, 0);
        assert_eq!(plan.params.plane1, 0);
        assert!(all_kernel_names().contains(&plan.kernel_name));
    }
    assert_eq!(QuantType::Nf4B64.to_gguf_type_id(), bnb::NF4_B64);
}

#[test]
fn fp8b_tiles_are_looked_up_like_the_cpu_kernel_for_every_shape() {
    use fllm_gguf_probe::{fp8b_shape, Fp8bLayout};
    let mut formats = vec![QuantType::Fp8Block];
    formats.extend(check_formats().into_iter().filter(|q| matches!(q, QuantType::Other(id) if fp8b_shape(*id).is_some())));
    // shapes the table does not list: very wide and very tall tiles, 4 x 4
    formats.extend([(4usize, 4usize), (1, 1), (4096, 2), (2, 32768)].into_iter().map(|(r, c)| QuantType::from_gguf_type_id(fllm_gguf_probe::fp8b_type_id(r, c).unwrap())));
    assert!(formats.len() >= 11);
    for quant in formats {
        let info = format_info(quant).unwrap_or_else(|| panic!("{quant:?}"));
        assert!(info.tiled && info.family == Family::Fp8b && info.tag == "fp8b" && info.qk == 8);
        let (tr, tc) = fp8b_shape(quant.to_gguf_type_id()).unwrap();
        // rows and columns that are not whole tiles, and a matrix smaller than a tile
        for (rows, cols) in [(37usize, 72usize), (9, 24), (130, 264)] {
            let raw = sample_weights(quant, rows, cols, 5 + rows as u64);
            let layout = Fp8bLayout::parse(quant.to_gguf_type_id(), &raw).unwrap();
            assert_eq!((layout.rows, layout.cols, layout.slabs), (rows, cols, 1));
            assert_eq!(raw.len(), expected_raw_len(quant, rows, cols).unwrap());
            let want = reference_weights(quant, &raw, rows, cols).unwrap();
            let packed = pack_weight(quant, rows, cols, &raw).unwrap();
            assert!(matches!(packed.bytes, std::borrow::Cow::Borrowed(_)), "used as stored");
            assert_eq!(packed.planes.plane1 as usize, rows * cols);
            assert_eq!(packed.planes.total_bytes as usize, raw.len());
            let plan = plan_quantized_linear(quant, rows, cols, 1, &caps()).unwrap();
            assert_eq!(plan.kernel_name, "fllm_q_mv_fp8b_r4");
            assert_eq!(plan.params.plane1, (rows * cols) as u64);
            assert_eq!(plan.params.pad, tr.trailing_zeros() | (tc.trailing_zeros() << 8), "{quant:?}");
            assert_eq!(plan.params.nb_blocks as usize, cols / 8);
            assert_eq!(plan_quantized_linear(quant, rows, cols, 33, &caps()).unwrap().kernel_name, "fllm_q_mm_fp8b_h");
            for r in 0..rows {
                for ci in 0..cols / 8 {
                    // (CPB is 1: blk = ci, c = 0)
                    let got = dec8p_fp8b(&packed.bytes, r, ci, &plan.params);
                    for (k, v) in got.iter().enumerate() {
                        let w = want[r * cols + ci * 8 + k];
                        assert_eq!(v.to_bits(), w.to_bits(), "{quant:?} {rows}x{cols} row {r} chunk {ci} value {k}: {v} vs {w}");
                    }
                }
            }
        }
    }
}

#[test]
fn fp8b_weights_that_are_not_one_matrix_of_the_shape_are_refused() {
    use fllm_gguf_probe::Fp8bLayout;
    let kind = QuantType::Fp8Block.to_gguf_type_id();
    let raw = sample_weights(QuantType::Fp8Block, 16, 32, 1);
    assert!(pack_weight(QuantType::Fp8Block, 16, 32, &raw).is_ok());
    // another shape with the same number of elements: the trailer says 16 x 32
    assert!(pack_weight(QuantType::Fp8Block, 32, 16, &raw).is_err());
    // a stack of two matrices (experts) is no single weight, whatever its length
    let stack = Fp8bLayout::new(kind, 32, 16, 2).unwrap();
    let codes = vec![0x38u8; stack.elements()];
    let scales = vec![1.0f32; stack.tiles()];
    let bytes = crate::cpu_fp8::pack(&stack, &codes, &scales);
    assert!(pack_weight(QuantType::Fp8Block, 16, 32, &bytes).is_err());
    assert!(pack_weight(QuantType::Fp8Block, 32, 32, &bytes).is_err());
    // an old tensor without the trailer
    assert!(pack_weight(QuantType::Fp8Block, 16, 32, &raw[..raw.len() - 16]).is_err());
    let mut bad = raw.clone();
    let n = bad.len();
    bad[n - 16] ^= 1;
    assert!(pack_weight(QuantType::Fp8Block, 16, 32, &bad).is_err());
    // columns that are no multiple of 8
    assert!(matches!(expected_raw_len(QuantType::Fp8Block, 4, 12), Err(WeightError::BadShape { .. })));
}

// ---- vectors for the CPU emulation of the kernels -----------------------------------------------------------------------------

/// Writes the device images and the reference dequantization of every format to `$FLLM_METAL_DUMP_DIR/<tag>[_variant].bin` (no-op when
/// the variable is unset). `metal/emu/run_emu.sh` runs the MSL kernels as C++ on them.
///
/// Header (little endian u32 words): magic `0x4d544c51`, rows, cols, qk, blocks per row, plane1 low, plane1 high, device bytes,
/// permutation flag, GGUF type id, `q_params::pad` (the tile shape of block-wise FP8); then the device image, `rows * cols` f32 reference weights (original column order), and for
/// act-order weights `cols` u32 permutation entries.
#[test]
fn dump_vectors_for_the_msl_emulation() {
    let Ok(dir) = std::env::var("FLLM_METAL_DUMP_DIR") else {
        return;
    };
    for quant in check_formats() {
        let info = format_info(quant).unwrap();
        let mut shapes = vec![(24usize, 256usize, ""), (72usize, 512usize, "_big")];
        // an odd row count: the last threadgroup of every kernel is partly outside the matrix
        shapes.push((37, 2048, "_odd"));
        if info.qk == 8 {
            shapes.push((24, 72, "_tail")); // K not a multiple of the 32-wide slice of the matrix-matrix kernel
        }
        for (rows, cols, suffix) in shapes {
            let raw = sample_weights(quant, rows, cols, 0xc0ffee ^ quant.to_gguf_type_id() as u64);
            let packed = pack_weight(quant, rows, cols, &raw).unwrap();
            let want = reference_weights(quant, &raw, rows, cols).unwrap();
            let mut out = Vec::new();
            let planes = packed.planes;
            // MLX affine weights with bfloat16 scales carry the flag in bit 63 of plane 1, as the planner puts it into q_params
            let bf16 = quant.mlx_affine().is_some_and(|m| m.bf16);
            let plane1 = planes.plane1 | if bf16 { MLX_BF16_FLAG } else { 0 };
            let pad = plan_quantized_linear(quant, rows, cols, 1, &caps()).unwrap().params.pad;
            for v in [
                0x4d54_4c51u32,
                rows as u32,
                cols as u32,
                info.qk as u32,
                (cols / info.qk) as u32,
                plane1 as u32,
                (plane1 >> 32) as u32,
                packed.bytes.len() as u32,
                packed.perm.is_some() as u32,
                quant.to_gguf_type_id(),
                pad,
            ] {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out.extend_from_slice(&packed.bytes);
            for v in &want {
                out.extend_from_slice(&v.to_le_bytes());
            }
            if let Some(perm) = &packed.perm {
                for p in perm {
                    out.extend_from_slice(&p.to_le_bytes());
                }
            }
            let tag = if info.tiled && quant != QuantType::Fp8Block {
                let (r, c) = fllm_gguf_probe::fp8b_shape(quant.to_gguf_type_id()).unwrap();
                format!("{}t{r}x{c}", info.tag)
            } else if info.permuted {
                format!("{}p", info.tag)
            } else if bf16 {
                format!("{}b", info.tag)
            } else {
                info.tag.to_owned()
            };
            std::fs::write(format!("{dir}/{tag}{suffix}.bin"), out).unwrap();
        }
    }
}
