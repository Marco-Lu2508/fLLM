//! Whole-block operations on the paged KV cache: copy a block (the fork of `n > 1` choices), export it to host memory and import it back
//! (preemption by swap). A block is stored per layer in several regions (codes, high bits, scales, minima; see `kv_format::KvLayout`),
//! so a block is a list of pieces, `(offset, length)` in 16-bit words of the key or value pool. The same pieces serve the host arrays of
//! the CPU decoder and the device pools of the accelerator backends, which only need `Stream::upload` / `download` / `copy_buffer` on
//! slices of the pool buffers. Exported data is the concatenation of the pieces (keys first), as little-endian 16-bit words.

use crate::backend::{BackendError, DeviceBuffer, DeviceKvCache, Stream};

/// The pieces of one block in the key and in the value pool: `(word offset, words)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BlockPieces {
    pub keys: Vec<(usize, usize)>,
    pub values: Vec<(usize, usize)>,
}

impl BlockPieces {
    /// `layers` layers of `layer_words` words each; `regions` are the `(offset, words per block)` of the regions inside a layer.
    pub fn new(
        layers: usize,
        key_layer_words: usize,
        key_regions: &[(usize, usize)],
        value_layer_words: usize,
        value_regions: &[(usize, usize)],
        block: usize,
    ) -> Self {
        let pieces = |layer_words: usize, regions: &[(usize, usize)]| {
            let mut out = Vec::with_capacity(layers * regions.len());
            for layer in 0..layers {
                for &(offset, words) in regions {
                    out.push((layer * layer_words + offset + block * words, words));
                }
            }
            out
        };
        Self { keys: pieces(key_layer_words, key_regions), values: pieces(value_layer_words, value_regions) }
    }

    /// Bytes of an exported block.
    pub fn bytes(&self) -> usize {
        2 * (self.keys.iter().chain(&self.values).map(|p| p.1).sum::<usize>())
    }

    pub fn check_fits(&self, key_words: usize, value_words: usize) -> Result<(), String> {
        let fits = |pieces: &[(usize, usize)], total: usize| pieces.iter().all(|&(o, w)| o + w <= total);
        if fits(&self.keys, key_words) && fits(&self.values, value_words) {
            Ok(())
        } else {
            Err("a KV block lies outside its pool".into())
        }
    }
}

/// Copy a block inside host arrays.
#[cfg(test)]
pub(crate) fn host_copy(key: &mut [u16], value: &mut [u16], from: &BlockPieces, to: &BlockPieces) -> Result<(), String> {
    from.check_fits(key.len(), value.len())?;
    to.check_fits(key.len(), value.len())?;
    for (pool, a, b) in [(&mut *key, &from.keys, &to.keys), (&mut *value, &from.values, &to.values)] {
        for (&(src, words), &(dst, w2)) in a.iter().zip(b) {
            if words != w2 {
                return Err("mismatched KV block pieces".into());
            }
            pool.copy_within(src..src + words, dst);
        }
    }
    Ok(())
}

pub(crate) fn host_export(key: &[u16], value: &[u16], block: &BlockPieces) -> Result<Vec<u8>, String> {
    block.check_fits(key.len(), value.len())?;
    let mut out = Vec::with_capacity(block.bytes());
    for (pool, pieces) in [(key, &block.keys), (value, &block.values)] {
        for &(offset, words) in pieces {
            for w in &pool[offset..offset + words] {
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
    }
    Ok(out)
}

pub(crate) fn host_import(key: &mut [u16], value: &mut [u16], block: &BlockPieces, data: &[u8]) -> Result<(), String> {
    block.check_fits(key.len(), value.len())?;
    if data.len() != block.bytes() {
        return Err(format!("swapped KV block has {} bytes, expected {}", data.len(), block.bytes()));
    }
    let mut at = 0;
    for (pool, pieces) in [(key, &block.keys), (value, &block.values)] {
        for &(offset, words) in pieces {
            for w in &mut pool[offset..offset + words] {
                *w = u16::from_le_bytes([data[at], data[at + 1]]);
                at += 2;
            }
        }
    }
    Ok(())
}

/// The pieces of `block` in the device pools of `kv`.
pub(crate) fn device_pieces(kv: &DeviceKvCache, block: usize) -> Result<BlockPieces, String> {
    if block >= kv.pool_blocks {
        return Err("KV block outside the pool".into());
    }
    if kv.layout.is_some() {
        return layered_pieces(kv, block);
    }
    let (key_layout, value_layout) = kv.layouts();
    let (Some(key_regions), Some(value_regions)) = (key_layout.block_regions(kv.block_size), value_layout.block_regions(kv.block_size)) else {
        return Err("this KV format has blocks that are not whole words".into());
    };
    Ok(BlockPieces::new(kv.layers, kv.words_per_layer_k, &key_regions, kv.words_per_layer_v, &value_regions, block))
}

/// The pieces of `block` in pools whose layers have their own geometry (`DeviceKvCache::layout`); layers that read an earlier layer's cache
/// have no region and no pieces.
fn layered_pieces(kv: &DeviceKvCache, block: usize) -> Result<BlockPieces, String> {
    let (key_format, value_format) = kv.formats();
    let mut out = BlockPieces { keys: Vec::new(), values: Vec::new() };
    for layer in 0..kv.layers {
        let heads = kv.layer_kv_heads(layer);
        for (pieces, format, dim, start, words) in [
            (&mut out.keys, key_format, kv.layer_head_dim(layer), kv.layer_key_offset(layer), kv.layer_key_words(layer)),
            (&mut out.values, value_format, kv.layer_value_dim(layer), kv.layer_value_offset(layer), kv.layer_value_words(layer)),
        ] {
            if words == 0 {
                continue;
            }
            let layout = crate::kv_format::KvLayout::new(format, kv.slots(), heads, dim);
            let regions = layout.block_regions(kv.block_size).ok_or("this KV format has blocks that are not whole words")?;
            pieces.extend(regions.iter().map(|&(offset, per_block)| (start + offset + block * per_block, per_block)));
        }
    }
    Ok(out)
}

fn slice(pool: &DeviceBuffer, offset: usize, words: usize) -> Result<DeviceBuffer, BackendError> {
    pool.slice(offset * 2, words * 2)
}

pub(crate) fn device_copy(stream: &mut dyn Stream, kv: &DeviceKvCache, src: usize, dst: usize) -> Result<(), String> {
    let (a, b) = (device_pieces(kv, src)?, device_pieces(kv, dst)?);
    let err = |e: BackendError| e.to_string();
    for (pool, from, to) in [(&kv.key_pool, &a.keys, &b.keys), (&kv.value_pool, &a.values, &b.values)] {
        for (&(s, words), &(d, _)) in from.iter().zip(to) {
            let source = slice(pool, s, words).map_err(err)?;
            let mut target = slice(pool, d, words).map_err(err)?;
            stream.copy_buffer(&mut target, &source).map_err(err)?;
        }
    }
    Ok(())
}

pub(crate) fn device_export(stream: &mut dyn Stream, kv: &DeviceKvCache, block: usize) -> Result<Vec<u8>, String> {
    let pieces = device_pieces(kv, block)?;
    let err = |e: BackendError| e.to_string();
    let mut out = Vec::with_capacity(pieces.bytes());
    for (pool, list) in [(&kv.key_pool, &pieces.keys), (&kv.value_pool, &pieces.values)] {
        for &(offset, words) in list {
            let part = slice(pool, offset, words).map_err(err)?;
            let mut bytes = vec![0u8; words * 2];
            stream.download(&mut bytes, &part).map_err(err)?;
            out.extend_from_slice(&bytes);
        }
    }
    Ok(out)
}

pub(crate) fn device_import(stream: &mut dyn Stream, kv: &DeviceKvCache, block: usize, data: &[u8]) -> Result<(), String> {
    let pieces = device_pieces(kv, block)?;
    if data.len() != pieces.bytes() {
        return Err(format!("swapped KV block has {} bytes, expected {}", data.len(), pieces.bytes()));
    }
    let err = |e: BackendError| e.to_string();
    let mut at = 0;
    for (pool, list) in [(&kv.key_pool, &pieces.keys), (&kv.value_pool, &pieces.values)] {
        for &(offset, words) in list {
            let mut part = slice(pool, offset, words).map_err(err)?;
            stream.upload(&mut part, &data[at..at + words * 2]).map_err(err)?;
            at += words * 2;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{cpu::CpuBackend, Backend};
    use crate::kv_format::{KvFormat, KvLayout};

    const BLOCKS: usize = 4;
    const BLOCK_SIZE: usize = 16;
    const HEADS: usize = 2;
    const DIM: usize = 64;
    const LAYERS: usize = 3;

    fn fill(words: &mut [u16], seed: u16) {
        for (i, w) in words.iter_mut().enumerate() {
            *w = (i as u16).wrapping_mul(2654).wrapping_add(seed);
        }
    }

    #[test]
    fn host_copy_export_and_import_move_exactly_one_block_for_every_format() {
        for &format in &KvFormat::ALL {
            let layout = KvLayout::new(format, BLOCKS * BLOCK_SIZE, HEADS, DIM);
            let regions = layout.block_regions(BLOCK_SIZE).expect("whole-word blocks");
            let layer_words = layout.words;
            let mut key = vec![0u16; layer_words * LAYERS];
            let mut value = vec![0u16; layer_words * LAYERS];
            fill(&mut key, 1);
            fill(&mut value, 7);
            let pieces = |b| BlockPieces::new(LAYERS, layer_words, &regions, layer_words, &regions, b);
            let before = (key.clone(), value.clone());
            // export block 0, copy 0 -> 2, import the export into 3
            let exported = host_export(&key, &value, &pieces(0)).unwrap();
            assert_eq!(exported.len(), pieces(0).bytes());
            host_copy(&mut key, &mut value, &pieces(0), &pieces(2)).unwrap();
            host_import(&mut key, &mut value, &pieces(3), &exported).unwrap();
            assert_eq!(host_export(&key, &value, &pieces(2)).unwrap(), exported, "{}", format.name());
            assert_eq!(host_export(&key, &value, &pieces(3)).unwrap(), exported, "{}", format.name());
            // block 1 and everything outside blocks 2 and 3 is unchanged
            assert_eq!(host_export(&key, &value, &pieces(1)).unwrap(), host_export(&before.0, &before.1, &pieces(1)).unwrap());
            let mut expected = before.clone();
            host_copy(&mut expected.0, &mut expected.1, &pieces(0), &pieces(2)).unwrap();
            host_copy(&mut expected.0, &mut expected.1, &pieces(0), &pieces(3)).unwrap();
            assert_eq!((key, value), expected, "{}", format.name());
        }
    }

    #[test]
    fn import_rejects_the_wrong_size_and_pieces_must_fit() {
        let layout = KvLayout::new(KvFormat::F16, BLOCKS * BLOCK_SIZE, HEADS, DIM);
        let regions = layout.block_regions(BLOCK_SIZE).unwrap();
        let p = BlockPieces::new(1, layout.words, &regions, layout.words, &regions, 0);
        let mut k = vec![0u16; layout.words];
        let mut v = vec![0u16; layout.words];
        assert!(host_import(&mut k, &mut v, &p, &[0u8; 3]).is_err());
        let far = BlockPieces::new(1, layout.words, &regions, layout.words, &regions, BLOCKS + 1);
        assert!(host_export(&k, &v, &far).is_err());
    }

    /// The device path against the host path: a `DeviceKvCache` whose pools are buffers of the CPU backend, driven only through
    /// `Stream::upload` / `download` / `copy_buffer` on slices.
    #[test]
    fn device_operations_match_the_host_operations() {
        for kv_kind in [crate::kv_format::pack_kind(KvFormat::F16, KvFormat::F16), crate::kv_format::pack_kind(KvFormat::Q8_0, KvFormat::Q8_0)] {
            let backend = CpuBackend::new();
            let (wk, wv) = DeviceKvCache::geometry(LAYERS, BLOCKS, BLOCK_SIZE, HEADS, DIM, DIM, kv_kind).unwrap();
            let mut stream = backend.create_stream().unwrap();
            let key_pool = backend.allocate_buffer(LAYERS * wk * 2).unwrap();
            let value_pool = backend.allocate_buffer(LAYERS * wv * 2).unwrap();
            let kv = DeviceKvCache {
                key_pool: key_pool.clone(),
                value_pool: value_pool.clone(),
                kv_kind,
                layers: LAYERS,
                pool_blocks: BLOCKS,
                block_size: BLOCK_SIZE,
                kv_heads: HEADS,
                head_dim: DIM,
                value_dim: DIM,
                words_per_layer_k: wk,
                words_per_layer_v: wv,
                layout: None,
                layer_geoms: None,
            };
            let mut host_key = vec![0u16; LAYERS * wk];
            let mut host_value = vec![0u16; LAYERS * wv];
            fill(&mut host_key, 3);
            fill(&mut host_value, 11);
            let bytes = |w: &[u16]| w.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
            let mut kb = key_pool.clone();
            stream.upload(&mut kb, &bytes(&host_key)).unwrap();
            let mut vb = value_pool.clone();
            stream.upload(&mut vb, &bytes(&host_value)).unwrap();

            let (kl, vl) = kv.layouts();
            let (kr, vr) = (kl.block_regions(BLOCK_SIZE).unwrap(), vl.block_regions(BLOCK_SIZE).unwrap());
            let pieces = |b| BlockPieces::new(LAYERS, wk, &kr, wv, &vr, b);
            // export agrees with the host export
            let exported = device_export(stream.as_mut(), &kv, 1).unwrap();
            assert_eq!(exported, host_export(&host_key, &host_value, &pieces(1)).unwrap());
            // copy 1 -> 2 and import into 3, then compare whole pools with the host arrays after the same operations
            device_copy(stream.as_mut(), &kv, 1, 2).unwrap();
            device_import(stream.as_mut(), &kv, 3, &exported).unwrap();
            host_copy(&mut host_key, &mut host_value, &pieces(1), &pieces(2)).unwrap();
            host_import(&mut host_key, &mut host_value, &pieces(3), &exported).unwrap();
            let mut got_k = vec![0u8; LAYERS * wk * 2];
            stream.download(&mut got_k, &key_pool).unwrap();
            let mut got_v = vec![0u8; LAYERS * wv * 2];
            stream.download(&mut got_v, &value_pool).unwrap();
            assert_eq!(got_k, bytes(&host_key));
            assert_eq!(got_v, bytes(&host_value));
            assert!(device_pieces(&kv, BLOCKS).is_err());
        }
    }
}
