//! Disaggregated prefill / decode: moving the KV cache of a prompt from the server that computed it (a prefill worker) to the server that
//! generates from it (a decode worker).
//!
//! The unit is the **prefix chain** of a prompt: the full KV blocks the paged cache holds for the prompt's leading tokens, with the token
//! ids of every block. The decode worker imports the chain into its own paged cache and registers it in its own prefix cache, recomputing
//! the chain hashes with its own seed, so the request that follows hits the cache for all full blocks and only the last partial block is
//! computed again. Nothing in the scheduler knows about disaggregation: a decode worker that received a chain behaves like a server that
//! had served the prompt before.
//!
//! * [`KvChain`] and its wire format: magic, a fingerprint of the cache geometry (a decode worker with another model, block size or KV
//!   format refuses the chain), block count and sizes, the block bytes and a checksum. Lengths are checked against the data actually
//!   present before anything is allocated.
//! * [`KvTransport`]: how the bytes travel. [`HttpTransport`] (TCP, a POST to the decode worker's `/admin/disagg/import`) is the one the
//!   server uses; [`DirTransport`] hands chains over through a directory (shared memory such as `/dev/shm` on one node, a network file
//!   system otherwise). An RDMA or NVLink transport implements the same trait; none exists yet.
//!
//! Single node first: the block bytes go through host memory (`device_kv_export` / `device_kv_import`), which is what a swap does too.
//! What this does not do: send only the layers a pipeline stage owns, convert between different tensor-parallel layouts, or overlap the
//! transfer with the prefill of the next request.
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

const MAGIC: &[u8; 8] = b"FLLMKV01";
/// More than this many blocks in one chain is a corrupt message, not a prompt.
const MAX_BLOCKS: usize = 1 << 22;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvBlock {
    /// The tokens this block holds (`block_size` of them).
    pub tokens: Vec<i64>,
    /// The exported KV bytes of the block (all layers, keys and values).
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvChain {
    /// Geometry of the cache the blocks came from; the importer compares it with its own.
    pub fingerprint: String,
    /// The request's prefix salt (0 without per-key isolation): part of the chain identity.
    pub salt: u64,
    pub block_size: usize,
    pub blocks: Vec<KvBlock>,
}

fn fnv(data: &[u8]) -> u64 {
    data.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(*b)).wrapping_mul(0x100_0000_01b3))
}

fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.at.checked_add(n).filter(|e| *e <= self.data.len()).ok_or("the KV message is truncated")?;
        let slice = &self.data[self.at..end];
        self.at = end;
        Ok(slice)
    }
    fn u64(&mut self) -> Result<u64, String> {
        let bytes: [u8; 8] = self.take(8)?.try_into().map_err(|_| "the KV message is truncated".to_owned())?;
        Ok(u64::from_le_bytes(bytes))
    }
}

impl KvChain {
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(64 + self.blocks.iter().map(|b| b.bytes.len() + 8 * b.tokens.len() + 16).sum::<usize>());
        out.extend_from_slice(MAGIC);
        put_u64(&mut out, self.fingerprint.len() as u64);
        out.extend_from_slice(self.fingerprint.as_bytes());
        put_u64(&mut out, self.salt);
        put_u64(&mut out, self.block_size as u64);
        put_u64(&mut out, self.blocks.len() as u64);
        for b in &self.blocks {
            put_u64(&mut out, b.tokens.len() as u64);
            for t in &b.tokens {
                out.extend_from_slice(&t.to_le_bytes());
            }
            put_u64(&mut out, b.bytes.len() as u64);
            out.extend_from_slice(&b.bytes);
        }
        let sum = fnv(&out);
        put_u64(&mut out, sum);
        out
    }

    pub fn decode(data: &[u8]) -> Result<Self, String> {
        if data.len() < MAGIC.len() + 8 || &data[..MAGIC.len()] != MAGIC {
            return Err("not a KV chain message (wrong magic)".into());
        }
        let (body, tail) = data.split_at(data.len() - 8);
        let expected = u64::from_le_bytes(tail.try_into().map_err(|_| "the KV message is truncated".to_owned())?);
        if fnv(body) != expected {
            return Err("the KV message is corrupt (checksum mismatch)".into());
        }
        let mut r = Reader { data: body, at: MAGIC.len() };
        let fp_len = r.u64()? as usize;
        let fingerprint = String::from_utf8(r.take(fp_len)?.to_vec()).map_err(|_| "the KV fingerprint is not text".to_owned())?;
        let salt = r.u64()?;
        let block_size = r.u64()? as usize;
        let count = r.u64()? as usize;
        if count > MAX_BLOCKS {
            return Err(format!("the KV message announces {count} blocks"));
        }
        let mut blocks = Vec::new();
        for _ in 0..count {
            let n_tokens = r.u64()? as usize;
            let token_bytes = r.take(n_tokens.checked_mul(8).ok_or("the KV message is corrupt")?)?;
            let tokens = token_bytes.as_chunks::<8>().0.iter().map(|c| i64::from_le_bytes(*c)).collect();
            let n_bytes = r.u64()? as usize;
            let bytes = r.take(n_bytes)?.to_vec();
            blocks.push(KvBlock { tokens, bytes });
        }
        if r.at != body.len() {
            return Err("the KV message has trailing data".into());
        }
        Ok(Self { fingerprint, salt, block_size, blocks })
    }

    pub fn payload_bytes(&self) -> usize {
        self.blocks.iter().map(|b| b.bytes.len()).sum()
    }
}

/// How a chain gets to the decode worker.
pub trait KvTransport: Send + Sync {
    fn name(&self) -> &'static str;
    /// Deliver `payload` (an encoded [`KvChain`]) to `target` (a URL for HTTP, a directory for [`DirTransport`]).
    fn send(&self, target: &str, payload: &[u8]) -> Result<(), String>;
}

/// TCP: a POST to the decode worker's `/admin/disagg/import` (blocking; call it off the async threads).
pub struct HttpTransport {
    pub api_key: Option<String>,
    pub timeout: Duration,
}

impl Default for HttpTransport {
    fn default() -> Self {
        Self { api_key: std::env::var("FLLM_API_KEY").ok().filter(|k| !k.is_empty()), timeout: Duration::from_secs(120) }
    }
}

impl KvTransport for HttpTransport {
    fn name(&self) -> &'static str {
        "http"
    }

    fn send(&self, target: &str, payload: &[u8]) -> Result<(), String> {
        let client = reqwest::blocking::Client::builder().timeout(self.timeout).build().map_err(|e| e.to_string())?;
        let url = format!("{}/admin/disagg/import", target.trim_end_matches('/'));
        let mut request = client.post(&url).header("content-type", "application/octet-stream").body(payload.to_vec());
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }
        let response = request.send().map_err(|e| format!("cannot reach {url}: {}", e.without_url()))?;
        if response.status().is_success() {
            Ok(())
        } else {
            let status = response.status();
            Err(format!("{url} answered {status}: {}", response.text().unwrap_or_default().chars().take(300).collect::<String>()))
        }
    }
}

/// A directory shared by both workers: each chain becomes one file, written under a temporary name and renamed, so a reader never sees a
/// half-written chain. Point it at `/dev/shm/...` for shared memory on one node.
pub struct DirTransport;

impl DirTransport {
    /// Chains waiting in `dir`, oldest name first; each file is removed once read.
    pub fn drain(dir: &Path) -> Result<Vec<Vec<u8>>, String> {
        let mut names: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "kvchain"))
            .collect();
        names.sort();
        let mut out = Vec::new();
        for path in names {
            out.push(std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?);
            std::fs::remove_file(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        }
        Ok(out)
    }
}

impl KvTransport for DirTransport {
    fn name(&self) -> &'static str {
        "dir"
    }

    fn send(&self, target: &str, payload: &[u8]) -> Result<(), String> {
        let dir = Path::new(target);
        std::fs::create_dir_all(dir).map_err(|e| format!("{target}: {e}"))?;
        let id = format!("{:020}-{:016x}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos()), fnv(payload));
        let tmp = dir.join(format!("{id}.tmp"));
        let mut file = std::fs::File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
        file.write_all(payload).and_then(|()| file.sync_all()).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, dir.join(format!("{id}.kvchain"))).map_err(|e| format!("{target}: {e}"))
    }
}

/// The transport for a target string: a directory path (starting with `/`, `.` or a drive letter, or `dir:`) uses the shared directory,
/// anything else is an HTTP URL.
pub fn transport_for(target: &str) -> Box<dyn KvTransport> {
    if target.starts_with("dir:") || (!target.contains("://") && (target.starts_with('/') || target.starts_with('.') || target.as_bytes().get(1) == Some(&b':'))) {
        Box::new(DirTransport)
    } else {
        Box::new(HttpTransport::default())
    }
}

pub fn strip_dir_prefix(target: &str) -> &str {
    target.strip_prefix("dir:").unwrap_or(target)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain(blocks: usize) -> KvChain {
        KvChain {
            fingerprint: "llama:2:16:4096".into(),
            salt: 7,
            block_size: 4,
            blocks: (0..blocks).map(|b| KvBlock { tokens: vec![b as i64, -1, 3, 4], bytes: (0..100).map(|i| (i + b) as u8).collect() }).collect(),
        }
    }

    #[test]
    fn a_chain_survives_the_wire() {
        for n in [0usize, 1, 5] {
            let c = chain(n);
            assert_eq!(KvChain::decode(&c.encode()).unwrap(), c);
        }
        assert_eq!(chain(3).payload_bytes(), 300);
    }

    #[test]
    fn damaged_messages_are_refused() {
        let good = chain(2).encode();
        assert!(KvChain::decode(&good[..good.len() - 1]).is_err());
        assert!(KvChain::decode(&good[..20]).is_err());
        assert!(KvChain::decode(b"hello world, this is not a chain").is_err());
        let mut flipped = good.clone();
        flipped[40] ^= 1;
        assert!(KvChain::decode(&flipped).unwrap_err().contains("checksum"));
        // a length field that promises more than the message holds, with a valid checksum
        let mut forged = good[..good.len() - 8].to_vec();
        let at = forged.len() - 100 - 8;
        forged[at..at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        let sum = fnv(&forged);
        forged.extend_from_slice(&sum.to_le_bytes());
        assert!(KvChain::decode(&forged).is_err());
    }

    #[test]
    fn the_directory_transport_hands_chains_over_in_order_and_once() {
        let dir = std::env::temp_dir().join(format!("fllm-kv-test-{}-{}", std::process::id(), fnv(b"order")));
        let _ = std::fs::remove_dir_all(&dir);
        let transport = DirTransport;
        let (a, b) = (chain(1).encode(), chain(2).encode());
        transport.send(dir.to_str().unwrap(), &a).unwrap();
        transport.send(dir.to_str().unwrap(), &b).unwrap();
        let got = DirTransport::drain(&dir).unwrap();
        assert_eq!(got, vec![a, b]);
        assert!(DirTransport::drain(&dir).unwrap().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn targets_pick_their_transport() {
        assert_eq!(transport_for("http://10.0.0.5:8000").name(), "http");
        assert_eq!(transport_for("/dev/shm/kv").name(), "dir");
        assert_eq!(transport_for("dir:kv-inbox").name(), "dir");
        assert_eq!(transport_for("C:/temp/kv").name(), "dir");
        assert_eq!(strip_dir_prefix("dir:kv-inbox"), "kv-inbox");
    }
}
