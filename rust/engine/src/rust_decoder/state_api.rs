//! Copying the recurrent state of a sequence out of the decoder and back in.
//!
//! Layers without a KV cache (Mamba-1/2, Gated DeltaNet, KDA, ShortConv, RWKV, ...) fold the whole history of a sequence into a
//! small fixed-size state, so a prefix cache or a speculative-decoding rollback cannot work by pointing at old KV blocks: the
//! state at the wanted position has to be copied. The decoder keeps these states per sequence, keyed by the first KV block
//! of the sequence (`tables[row * stride]`), on the host (`states`) and, for layer kinds with device kernels, in one device
//! buffer per sequence (`device_states`). This module snapshots and restores both.
//!
//! Typical use (speculative decoding): `snapshot_state(first)` before verifying `k` draft tokens, then `restore_state(first, &snapshot)`
//! when some of them are rejected and the verified prefix is decoded again; a prefix cache stores one snapshot per cached prefix
//! and restores it into the key of the new sequence.
use super::*;

/// A state that can be copied out and put back.
pub trait RecurrentState {
    /// Plain data holding everything needed to continue from this point.
    type Snapshot: Clone;
    /// Number of tokens folded into the state.
    fn tokens(&self) -> usize;
    fn snapshot(&self) -> Self::Snapshot;
    fn restore(&mut self, snapshot: &Self::Snapshot);
}

/// Recurrent state of one host layer of one sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerStateSnapshot {
    /// Tokens folded into the state.
    pub tokens: usize,
    /// Convolution history (`kernel - 1` inputs per channel), empty for layers without a convolution.
    pub conv: Vec<f32>,
    /// Recurrent matrix/vector state.
    pub rec: Vec<f32>,
}

impl RecurrentState for LayerState {
    type Snapshot = LayerStateSnapshot;
    fn tokens(&self) -> usize {
        self.len
    }
    fn snapshot(&self) -> LayerStateSnapshot {
        LayerStateSnapshot { tokens: self.len, conv: self.conv.clone(), rec: self.rec.clone() }
    }
    fn restore(&mut self, snapshot: &LayerStateSnapshot) {
        self.len = snapshot.tokens;
        self.conv.clone_from(&snapshot.conv);
        self.rec.clone_from(&snapshot.rec);
    }
}

/// Everything a sequence keeps outside the KV cache.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SequenceSnapshot {
    /// One entry per decoder layer; `None` for layers without host state.
    pub host: Vec<Option<LayerStateSnapshot>>,
    /// Bytes of the device state buffer (all layers of the kind that runs on the device) and the tokens folded into each layer.
    pub device: Option<(Vec<u8>, Vec<usize>)>,
}

impl SequenceSnapshot {
    /// Tokens folded into the state (the smallest count over all layers that hold one).
    pub fn tokens(&self) -> Option<usize> {
        let host = self.host.iter().flatten().map(|l| l.tokens);
        let device = self.device.iter().flat_map(|(_, lens)| lens.iter().copied());
        host.chain(device).min()
    }

    /// Memory held by the snapshot.
    pub fn bytes(&self) -> usize {
        let host: usize = self.host.iter().flatten().map(|l| (l.conv.len() + l.rec.len()) * 4).sum();
        host + self.device.as_ref().map_or(0, |(b, _)| b.len())
    }
}

impl RustDecoder {
    /// Whether the model keeps state that a snapshot has to cover.
    pub fn has_recurrent_state(&self) -> bool {
        self.v4.is_some() || self.inkling.is_some() || self.layers.iter().any(|l| !matches!(l.attn, Attn::Std { .. } | Attn::Mla(_) | Attn::None) || l.parallel.is_some())
    }

    /// Copies the recurrent state of the sequence whose first KV block is `first`. `None`: the sequence holds no state yet.
    pub fn snapshot_state(&self, first: i64) -> Result<Option<SequenceSnapshot>, String> {
        let host = {
            let store = self.states.lock().map_err(|_| "recurrent state lock poisoned")?;
            store
                .map
                .get(&first)
                .map(|seq| seq.layers.iter().map(|l| l.as_ref().map(RecurrentState::snapshot)).collect::<Vec<_>>())
        };
        let device = {
            let store = self.device_states.lock().map_err(|_| "recurrent state lock poisoned")?;
            match (store.map.get(&first), &self.backend_stream) {
                (Some(seq), Some(stream)) => {
                    let mut bytes = vec![0u8; seq.buf.size];
                    let mut stream = stream.lock().map_err(|_| "device stream lock poisoned")?;
                    stream.synchronize().map_err(|e| e.to_string())?;
                    stream.download(&mut bytes, &seq.buf).map_err(|e| e.to_string())?;
                    Some((bytes, seq.lens.clone()))
                }
                _ => None,
            }
        };
        if host.is_none() && device.is_none() {
            return Ok(None);
        }
        Ok(Some(SequenceSnapshot { host: host.unwrap_or_default(), device }))
    }

    /// Puts a snapshot back as the state of the sequence whose first KV block is `first` (creating the entry when needed).
    /// The next forward pass for that sequence must continue at `snapshot.tokens()`.
    pub fn restore_state(&self, first: i64, snapshot: &SequenceSnapshot) -> Result<(), String> {
        if !snapshot.host.is_empty() {
            if snapshot.host.len() != self.layers.len() {
                return Err(format!("snapshot has {} layers, the model {}", snapshot.host.len(), self.layers.len()));
            }
            let mut store = self.states.lock().map_err(|_| "recurrent state lock poisoned")?;
            store.clock += 1;
            let clock = store.clock;
            if !store.map.contains_key(&first) && store.map.len() >= self.state_capacity {
                if let Some(oldest) = store.map.iter().min_by_key(|(_, s)| s.used).map(|(k, _)| *k) {
                    store.map.remove(&oldest);
                }
            }
            let layers = self.layers.len();
            let seq = store
                .map
                .entry(first)
                .or_insert_with(|| SeqState { used: clock, layers: (0..layers).map(|_| None).collect() });
            seq.used = clock;
            for (slot, saved) in seq.layers.iter_mut().zip(&snapshot.host) {
                match (slot.as_mut(), saved) {
                    (Some(state), Some(saved)) => state.restore(saved),
                    (_, Some(saved)) => {
                        *slot = Some(LayerState { len: saved.tokens, conv: saved.conv.clone(), rec: saved.rec.clone() })
                    }
                    (_, None) => *slot = None,
                }
            }
        }
        if let Some((bytes, lens)) = &snapshot.device {
            let (Some(backend), Some(stream)) = (&self.backend, &self.backend_stream) else {
                return Err("snapshot holds device state but the decoder has no accelerator".into());
            };
            let mut store = self.device_states.lock().map_err(|_| "recurrent state lock poisoned")?;
            store.clock += 1;
            let clock = store.clock;
            if !store.map.contains_key(&first) {
                if store.map.len() >= self.state_capacity {
                    if let Some(key) = store.map.iter().min_by_key(|(_, s)| s.used).map(|(k, _)| *k) {
                        if let Some(old) = store.map.remove(&key) {
                            store.free.push(old.buf);
                        }
                    }
                }
                let buf = match store.free.pop() {
                    Some(buf) => buf,
                    None => backend.allocate_buffer(bytes.len()).map_err(|e| e.to_string())?,
                };
                store.map.insert(first, DeviceSeq { used: clock, buf, lens: lens.clone() });
            }
            let seq = store.map.get_mut(&first).ok_or("recurrent state vanished")?;
            if seq.buf.size != bytes.len() {
                return Err("snapshot size differs from the device state buffer".into());
            }
            let mut stream = stream.lock().map_err(|_| "device stream lock poisoned")?;
            stream.upload(&mut seq.buf, bytes).map_err(|e| e.to_string())?;
            stream.synchronize().map_err(|e| e.to_string())?;
            seq.lens.clone_from(lens);
            seq.used = clock;
        }
        Ok(())
    }

    /// Forgets the recurrent state of a finished sequence.
    pub fn forget_state(&self, first: i64) {
        if let Ok(mut store) = self.states.lock() {
            store.map.remove(&first);
        }
        if let Ok(mut store) = self.device_states.lock() {
            if let Some(old) = store.map.remove(&first) {
                store.free.push(old.buf);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_state_snapshot_round_trips() {
        let mut state = LayerState { len: 3, conv: vec![1.0, 2.0], rec: vec![0.5; 4] };
        let saved = state.snapshot();
        state.len = 9;
        state.conv.fill(7.0);
        state.rec.push(1.0);
        state.restore(&saved);
        assert_eq!((state.len, state.conv.as_slice(), state.rec.len()), (3, &[1.0, 2.0][..], 4));
        assert_eq!(state.tokens(), 3);
    }

    #[test]
    fn sequence_snapshot_reports_tokens_and_bytes() {
        let layer = |tokens| Some(LayerStateSnapshot { tokens, conv: vec![0.0; 2], rec: vec![0.0; 6] });
        let snap = SequenceSnapshot { host: vec![layer(5), None, layer(4)], device: Some((vec![0; 16], vec![7])) };
        assert_eq!(snap.tokens(), Some(4));
        assert_eq!(snap.bytes(), 2 * 8 * 4 + 16);
        assert_eq!(SequenceSnapshot::default().tokens(), None);
    }
}
