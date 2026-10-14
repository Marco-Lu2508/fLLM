//! Plans of the embedding lookup (`embed_float.metal`, `embed_quant.metal`). Package p06b.
//!
//! The table is a `DeviceWeight` in the device layout of package p06a (`weights/layout.rs`): float rows and all block formats except MXFP4 are the
//! bytes of the GGUF file (rows of quantization blocks, [`TableLayout::Gguf`]); MXFP4 is re-sorted into two planes
//! ([`TableLayout::Planar`]: the 16 code bytes of every block, then the E8M0 scale byte of every block). The kernels decode one output element per
//! thread (`o_deq_*` in `common_ops.metal`). Formats without a kernel (IQ1/IQ2/IQ3, IQ4_XS, Q8_K, NVFP4, Q4Z, FP8) return `UnsupportedQuant`: the
//! stream dequantizes the rows on the host in that case.

use super::*;
use crate::backend::QuantType;

/// How the device image of an embedding table is laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TableLayout {
    /// The bytes of the GGUF tensor: rows of float values or of quantization blocks.
    #[default]
    Gguf,
    /// MXFP4 as repacked by package p06a: plane 0 = the 16 code bytes of every block, plane 1 (at `round_up(blocks * 16, 16)`) = the E8M0 scale byte
    /// of every block; blocks are numbered `row * (cols / 32) + block`.
    Planar,
}

/// Geometry of an embedding table for [`plan_embedding`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmbeddingTable {
    pub quant: QuantType,
    /// Vocabulary size (rows of the table).
    pub rows: usize,
    /// Hidden size (columns).
    pub cols: usize,
    pub layout: TableLayout,
}

impl EmbeddingTable {
    /// A table in the layout of the GGUF file.
    pub fn gguf(quant: QuantType, rows: usize, cols: usize) -> Self {
        Self {
            quant,
            rows,
            cols,
            layout: TableLayout::Gguf,
        }
    }

    /// An MXFP4 table in the planar device layout of package p06a.
    pub fn mxfp4_planar(rows: usize, cols: usize) -> Self {
        Self {
            quant: QuantType::Mxfp4,
            rows,
            cols,
            layout: TableLayout::Planar,
        }
    }
}

/// Kernel name, values per block and bytes per block of the formats of the embedding kernels in the GGUF layout (`None`: no kernel). F32 / F16 /
/// BF16 have no blocks: one value, 4 / 2 / 2 bytes.
pub fn embed_kernel(quant: QuantType) -> Option<(&'static str, usize, usize)> {
    use QuantType as Q;
    Some(match quant {
        Q::F32 => ("fllm_embed_f32", 1, 4),
        Q::F16 => ("fllm_embed_f16", 1, 2),
        Q::BF16 => ("fllm_embed_bf16", 1, 2),
        Q::Q4_0 => ("fllm_embed_q4_0", 32, 18),
        Q::Q4_1 => ("fllm_embed_q4_1", 32, 20),
        Q::Q5_0 => ("fllm_embed_q5_0", 32, 22),
        Q::Q5_1 => ("fllm_embed_q5_1", 32, 24),
        Q::Q8_0 => ("fllm_embed_q8_0", 32, 34),
        Q::Q2_K => ("fllm_embed_q2_k", 256, 84),
        Q::Q3_K => ("fllm_embed_q3_k", 256, 110),
        Q::Q4_K => ("fllm_embed_q4_k", 256, 144),
        Q::Q5_K => ("fllm_embed_q5_k", 256, 176),
        Q::Q6_K => ("fllm_embed_q6_k", 256, 210),
        Q::IQ4_NL => ("fllm_embed_iq4_nl", 32, 18),
        Q::Mxfp4 => ("fllm_embed_mxfp4", 32, 17),
        _ => return None,
    })
}

/// Name of the kernel for the planar MXFP4 layout.
pub const EMBED_MXFP4_PLANAR: &str = "fllm_embed_mxfp4_planar";

impl EmbeddingTable {
    /// Bytes of one table row in the GGUF layout, or `None` when the format has no kernel or `cols` is not a whole number of blocks. For the planar
    /// layout this is the size of the code plane of one row (the scale plane adds one byte per block).
    pub fn row_bytes(&self) -> Option<usize> {
        let (_, qk, bytes) = embed_kernel(self.quant)?;
        let bytes = match self.layout {
            TableLayout::Gguf => bytes,
            TableLayout::Planar => 16,
        };
        self.cols.is_multiple_of(qk).then(|| self.cols / qk * bytes)
    }
}

/// `Stream::embedding_lookup`: `out[t] = table[ids[t]]` for every id (checked against the vocabulary here). The ids are uploaded with the dispatch
/// (slot `ids`); `weights` is the table, `out` receives `ids.len() x cols` floats.
pub fn plan_embedding(table: &EmbeddingTable, ids: &[i64]) -> Result<OpPlan, BackendError> {
    let (name, _, _) =
        embed_kernel(table.quant).ok_or(BackendError::UnsupportedQuant(table.quant))?;
    let name = match table.layout {
        TableLayout::Gguf => name,
        TableLayout::Planar if table.quant == QuantType::Mxfp4 => EMBED_MXFP4_PLANAR,
        TableLayout::Planar => return Err(BackendError::UnsupportedQuant(table.quant)),
    };
    if ids.is_empty() {
        return Err(invalid("embedding lookup without ids"));
    }
    if table.cols == 0 || table.rows == 0 {
        return Err(invalid("empty embedding table"));
    }
    let row_bytes = table.row_bytes().ok_or_else(|| {
        invalid(format!(
            "{:?} embedding table {} x {} is not a whole number of blocks",
            table.quant, table.rows, table.cols
        ))
    })?;
    if let Some(&id) = ids.iter().find(|&&id| id < 0 || id as usize >= table.rows) {
        return Err(invalid(format!("embedding token id {id} out of bounds")));
    }
    // byte offset of the scale plane of the planar layout (`round_up(blocks * 16, 16)`, blocks * 16 already is a multiple of 16)
    let plane1 = match table.layout {
        TableLayout::Planar => table.rows * row_bytes,
        TableLayout::Gguf => 0,
    };
    let id_words: Vec<usize> = ids.iter().map(|&i| i as usize).collect();
    let tx = round_up(table.cols, 32).min(256);
    let params = Params::new()
        .u64("plane1", plane1 as u64)
        .u("n_ids", ids.len())
        .u("cols", table.cols)
        .u("row_bytes", row_bytes);
    plan(
        name,
        params,
        [grid_dim(div_up(table.cols, tx))?, grid_dim(ids.len())?, 1],
        [tx as u32, 1, 1],
        Slots::new()
            .read("weights")
            .host("ids", Arc::new(pack_u32(&id_words)?))
            .write("out"),
    )
}
