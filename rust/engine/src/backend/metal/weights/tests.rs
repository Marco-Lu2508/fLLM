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
            if info.row_scales {
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
    for quant in [
        QuantType::Mxfp4,
        QuantType::Q4Z32,
        QuantType::Q4Z64,
        QuantType::Q4Z128,
        QuantType::Q4ZP32,
        QuantType::Q4ZP64,
        QuantType::Q4ZP128,
    ] {
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

// ---- vectors for the CPU emulation of the kernels -----------------------------------------------------------------------------

/// Writes the device images and the reference dequantization of every format to `$FLLM_METAL_DUMP_DIR/<tag>[_variant].bin` (no-op when
/// the variable is unset). `metal/emu/run_emu.sh` runs the MSL kernels as C++ on them.
///
/// Header (little endian u32 words): magic `0x4d544c51`, rows, cols, qk, blocks per row, plane1 low, plane1 high, device bytes,
/// permutation flag, GGUF type id; then the device image, `rows * cols` f32 reference weights (original column order), and for
/// act-order weights `cols` u32 permutation entries.
#[test]
fn dump_vectors_for_the_msl_emulation() {
    let Ok(dir) = std::env::var("FLLM_METAL_DUMP_DIR") else {
        return;
    };
    for quant in supported_formats() {
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
            for v in [
                0x4d54_4c51u32,
                rows as u32,
                cols as u32,
                info.qk as u32,
                (cols / info.qk) as u32,
                planes.plane1 as u32,
                (planes.plane1 >> 32) as u32,
                packed.bytes.len() as u32,
                packed.perm.is_some() as u32,
                quant.to_gguf_type_id(),
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
            let tag = if info.permuted {
                format!("{}p", info.tag)
            } else {
                info.tag.to_owned()
            };
            std::fs::write(format!("{dir}/{tag}{suffix}.bin"), out).unwrap();
        }
    }
}
