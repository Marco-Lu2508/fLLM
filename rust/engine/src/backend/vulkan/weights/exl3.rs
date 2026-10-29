//! Dispatch plan of an EXL3 product on the Vulkan backend: `y = diag(svh) H (R^T (H diag(suh) x))` as three passes per chunk of up to 128 tokens
//! (`exl3_pre.comp`, `exl3_mm.comp`, `exl3_post.comp`, the same structure as the CUDA kernels). Pure CPU code, like the plan of the other formats.

use super::plan::MAX_GROUPS_PER_DIM;
use crate::backend::BackendError;

/// Tokens per pass: the scratch holds the transformed input and the product of that many rows.
pub const EXL3_CHUNK: usize = 128;

/// Push-constant bytes of the three shaders (`exl3_pre`: cols, tok0; `exl3_mm`: rows, cols, chunk; `exl3_post`: rows, tok0, flags, scale).
pub const PRE_PUSH_BYTES: u32 = 8;
pub const MM_PUSH_BYTES: u32 = 12;
pub const POST_PUSH_BYTES: u32 = 16;

/// One pass over `chunk` tokens starting at token `tok0`.
#[derive(Debug, Clone, PartialEq)]
pub struct Exl3Pass {
    pub tok0: usize,
    pub chunk: usize,
    pub pre_push: Vec<u8>,
    pub pre_groups: [u32; 3],
    pub mm_push: Vec<u8>,
    pub mm_groups: [u32; 3],
    pub post_push: Vec<u8>,
    pub post_groups: [u32; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub struct Exl3Plan {
    pub passes: Vec<Exl3Pass>,
    /// Specialization constant 0 of `exl3_mm`: groups of 16 tokens per pass (1, 2, 4 or 8).
    pub tile_groups: u32,
    /// Bytes of the scratch for the transformed input (`x2`) and for the product (`z`) of the largest pass.
    pub x2_bytes: usize,
    pub z_bytes: usize,
}

fn words(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Plan of `batch` tokens through a `rows x cols` EXL3 weight (both multiples of 128); `scale` is the tensor factor, `has_bias` adds `bias[out_map[o]]`.
pub fn plan_exl3(rows: usize, cols: usize, batch: usize, has_bias: bool, scale: f32) -> Result<Exl3Plan, BackendError> {
    let invalid = |msg: String| BackendError::InvalidShape(msg);
    if rows == 0 || cols == 0 || batch == 0 || !rows.is_multiple_of(128) || !cols.is_multiple_of(128) {
        return Err(invalid(format!("EXL3 product of {batch} tokens with a {rows} x {cols} weight (rows and columns must be multiples of 128)")));
    }
    if (rows / 16) as u64 > MAX_GROUPS_PER_DIM {
        return Err(BackendError::UnsupportedOperation(format!("EXL3 weight with {rows} rows is too tall for one dispatch")));
    }
    let to_u32 = |v: usize| u32::try_from(v).map_err(|_| invalid(format!("EXL3 dimension {v} exceeds 32 bits")));
    let (rows32, cols32) = (to_u32(rows)?, to_u32(cols)?);
    if (batch as u64) * (rows.max(cols) as u64) >= u64::from(u32::MAX) {
        return Err(invalid(format!("{batch} x {} activations exceed 32-bit indexing", rows.max(cols))));
    }
    let widest = batch.min(EXL3_CHUNK);
    let tile_groups = match widest {
        0..=16 => 1,
        17..=32 => 2,
        33..=64 => 4,
        _ => 8,
    };
    let mut passes = Vec::new();
    let mut tok0 = 0;
    while tok0 < batch {
        let chunk = (batch - tok0).min(EXL3_CHUNK);
        passes.push(Exl3Pass {
            tok0,
            chunk,
            pre_push: words(&[cols32, to_u32(tok0)?]),
            pre_groups: [(cols / 128) as u32, chunk as u32, 1],
            mm_push: words(&[rows32, cols32, chunk as u32]),
            mm_groups: [(rows / 16) as u32, 1, 1],
            post_push: words(&[rows32, to_u32(tok0)?, u32::from(has_bias), scale.to_bits()]),
            post_groups: [(rows / 128) as u32, chunk as u32, 1],
        });
        tok0 += chunk;
    }
    Ok(Exl3Plan { passes, tile_groups, x2_bytes: widest * cols * 4, z_bytes: widest * rows * 4 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passes_cover_the_batch_in_chunks_of_128() {
        let p = plan_exl3(384, 256, 300, true, 0.5).unwrap();
        assert_eq!(p.passes.iter().map(|q| (q.tok0, q.chunk)).collect::<Vec<_>>(), [(0, 128), (128, 128), (256, 44)]);
        assert_eq!(p.tile_groups, 8);
        assert_eq!((p.x2_bytes, p.z_bytes), (128 * 256 * 4, 128 * 384 * 4));
        let q = &p.passes[2];
        assert_eq!(q.pre_groups, [2, 44, 1]);
        assert_eq!(q.mm_groups, [24, 1, 1]);
        assert_eq!(q.post_groups, [3, 44, 1]);
        let word = |b: &[u8], i: usize| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!((word(&q.pre_push, 0), word(&q.pre_push, 1)), (256, 256));
        assert_eq!((word(&q.mm_push, 0), word(&q.mm_push, 1), word(&q.mm_push, 2)), (384, 256, 44));
        assert_eq!((word(&q.post_push, 0), word(&q.post_push, 1), word(&q.post_push, 2), word(&q.post_push, 3)), (384, 256, 1, 0.5f32.to_bits()));
        assert_eq!(q.pre_push.len() as u32, PRE_PUSH_BYTES);
        assert_eq!(q.mm_push.len() as u32, MM_PUSH_BYTES);
        assert_eq!(q.post_push.len() as u32, POST_PUSH_BYTES);
    }

    #[test]
    fn the_tile_width_follows_the_largest_pass() {
        for (batch, groups) in [(1usize, 1u32), (16, 1), (17, 2), (32, 2), (33, 4), (64, 4), (65, 8), (128, 8), (129, 8), (1000, 8)] {
            assert_eq!(plan_exl3(128, 128, batch, false, 1.0).unwrap().tile_groups, groups, "batch {batch}");
        }
    }

    #[test]
    fn shapes_the_kernels_cannot_take_are_errors() {
        assert!(plan_exl3(128, 100, 1, false, 1.0).is_err());
        assert!(plan_exl3(100, 128, 1, false, 1.0).is_err());
        assert!(plan_exl3(128, 128, 0, false, 1.0).is_err());
        assert!(plan_exl3(16 * 70_000, 128, 1, false, 1.0).is_err());
    }
}
