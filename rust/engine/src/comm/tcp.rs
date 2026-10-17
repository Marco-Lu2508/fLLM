//! Multi-node rendezvous and the socket transport.
//!
//! `TcpStore` is a tiny key-value server (rank 0's node runs it on `--master-addr:--master-port`): ranks publish small values (their
//! listening address, an `ncclUniqueId`) and read the values of the others, blocking until they exist. `SocketFabric` connects all
//! ranks in a full mesh of TCP connections and carries the control traffic and the host collectives between nodes.
use super::fabric::{default_timeout, Fabric, Inbox};
use super::CommResult;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

const OP_SET: u8 = 1;
const OP_GET: u8 = 2;

fn io_error(context: &str, error: std::io::Error) -> String {
    format!("{context}: {error}")
}

fn read_exact_vec(stream: &mut TcpStream, len: usize) -> std::io::Result<Vec<u8>> {
    // never trust a length field with an allocation: grow while reading
    let mut out = Vec::new();
    let mut limited = stream.take(len as u64);
    limited.read_to_end(&mut out)?;
    if out.len() != len {
        return Err(std::io::ErrorKind::UnexpectedEof.into());
    }
    Ok(out)
}

#[derive(Default)]
struct StoreState {
    values: HashMap<String, Vec<u8>>,
}

struct StoreShared {
    state: Mutex<StoreState>,
    changed: Condvar,
    stop: AtomicBool,
}

/// The server half. Dropping it stops accepting connections.
pub struct TcpStore {
    addr: SocketAddr,
    shared: Arc<StoreShared>,
}

impl TcpStore {
    /// Listen on `bind` (`"0.0.0.0:29500"`, port 0 picks a free one).
    pub fn serve(bind: &str) -> CommResult<Self> {
        let listener = TcpListener::bind(bind).map_err(|e| io_error(&format!("cannot listen on {bind}"), e))?;
        let addr = listener.local_addr().map_err(|e| io_error("local address", e))?;
        let shared = Arc::new(StoreShared { state: Mutex::new(StoreState::default()), changed: Condvar::new(), stop: AtomicBool::new(false) });
        let accept_shared = shared.clone();
        std::thread::Builder::new()
            .name("fllm-store".into())
            .spawn(move || {
                for stream in listener.incoming() {
                    if accept_shared.stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let Ok(stream) = stream else { continue };
                    let shared = accept_shared.clone();
                    let _ = std::thread::Builder::new().name("fllm-store-conn".into()).spawn(move || serve_connection(stream, shared));
                }
            })
            .map_err(|e| io_error("cannot start the store thread", e))?;
        Ok(Self { addr, shared })
    }

    pub fn addr(&self) -> SocketAddr {
        self.addr
    }
}

impl Drop for TcpStore {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        // wake the accept loop
        let _ = TcpStream::connect_timeout(&self.addr, Duration::from_millis(200));
    }
}

fn serve_connection(mut stream: TcpStream, shared: Arc<StoreShared>) {
    let _ = stream.set_nodelay(true);
    loop {
        let mut op = [0u8; 1];
        if stream.read_exact(&mut op).is_err() {
            return;
        }
        let mut len = [0u8; 4];
        if stream.read_exact(&mut len).is_err() {
            return;
        }
        let key_len = u32::from_le_bytes(len) as usize;
        if key_len > 4096 {
            return;
        }
        let Ok(key) = read_exact_vec(&mut stream, key_len) else { return };
        let key = String::from_utf8_lossy(&key).into_owned();
        match op[0] {
            OP_SET => {
                let mut len = [0u8; 8];
                if stream.read_exact(&mut len).is_err() {
                    return;
                }
                let value_len = u64::from_le_bytes(len) as usize;
                if value_len > 64 << 20 {
                    return;
                }
                let Ok(value) = read_exact_vec(&mut stream, value_len) else { return };
                shared.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner).values.insert(key, value);
                shared.changed.notify_all();
                if stream.write_all(&[0]).is_err() {
                    return;
                }
            }
            OP_GET => {
                let mut ms = [0u8; 8];
                if stream.read_exact(&mut ms).is_err() {
                    return;
                }
                let deadline = Instant::now() + Duration::from_millis(u64::from_le_bytes(ms));
                let mut state = shared.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                let found = loop {
                    if let Some(value) = state.values.get(&key) {
                        break Some(value.clone());
                    }
                    let now = Instant::now();
                    if now >= deadline {
                        break None;
                    }
                    state = shared.changed.wait_timeout(state, deadline - now).unwrap_or_else(std::sync::PoisonError::into_inner).0;
                };
                drop(state);
                let reply = match found {
                    Some(value) => {
                        let mut reply = vec![0u8];
                        reply.extend_from_slice(&(value.len() as u64).to_le_bytes());
                        reply.extend_from_slice(&value);
                        reply
                    }
                    None => vec![1u8],
                };
                if stream.write_all(&reply).is_err() {
                    return;
                }
            }
            _ => return,
        }
    }
}

/// The client half: one connection, used by one thread at a time.
pub struct StoreClient {
    stream: Mutex<TcpStream>,
    timeout: Duration,
}

impl StoreClient {
    /// Connect to the store at `addr`, retrying until it is up (the other nodes start in any order).
    pub fn connect(addr: &str, timeout: Duration) -> CommResult<Self> {
        let deadline = Instant::now() + timeout;
        let mut last = String::new();
        loop {
            let resolved: Vec<SocketAddr> = addr.to_socket_addrs().map_err(|e| io_error(&format!("cannot resolve {addr}"), e))?.collect();
            for candidate in &resolved {
                match TcpStream::connect_timeout(candidate, Duration::from_secs(2)) {
                    Ok(stream) => {
                        let _ = stream.set_nodelay(true);
                        return Ok(Self { stream: Mutex::new(stream), timeout });
                    }
                    Err(e) => last = e.to_string(),
                }
            }
            if Instant::now() >= deadline {
                return Err(format!("cannot reach the rendezvous store at {addr} within {} s ({last})", timeout.as_secs()));
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    /// The address of this machine on the interface that reaches the store (what the other nodes should connect to).
    pub fn local_ip(&self) -> String {
        let stream = self.stream.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        stream.local_addr().map_or_else(|_| "127.0.0.1".to_owned(), |a| a.ip().to_string())
    }

    pub fn set(&self, key: &str, value: &[u8]) -> CommResult<()> {
        let mut stream = self.stream.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut request = vec![OP_SET];
        request.extend_from_slice(&(key.len() as u32).to_le_bytes());
        request.extend_from_slice(key.as_bytes());
        request.extend_from_slice(&(value.len() as u64).to_le_bytes());
        request.extend_from_slice(value);
        stream.write_all(&request).map_err(|e| io_error("store set", e))?;
        let mut ack = [0u8; 1];
        stream.read_exact(&mut ack).map_err(|e| io_error("store set", e))?;
        Ok(())
    }

    /// Block until `key` exists (at most the client's timeout).
    pub fn get(&self, key: &str) -> CommResult<Vec<u8>> {
        let mut stream = self.stream.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut request = vec![OP_GET];
        request.extend_from_slice(&(key.len() as u32).to_le_bytes());
        request.extend_from_slice(key.as_bytes());
        request.extend_from_slice(&(self.timeout.as_millis() as u64).to_le_bytes());
        stream.write_all(&request).map_err(|e| io_error("store get", e))?;
        let mut status = [0u8; 1];
        stream.set_read_timeout(Some(self.timeout + Duration::from_secs(5))).map_err(|e| io_error("store get", e))?;
        stream.read_exact(&mut status).map_err(|e| io_error("store get", e))?;
        if status[0] != 0 {
            return Err(format!("timed out after {} s waiting for '{key}' in the rendezvous store (is every node started?)", self.timeout.as_secs()));
        }
        let mut len = [0u8; 8];
        stream.read_exact(&mut len).map_err(|e| io_error("store get", e))?;
        read_exact_vec(&mut stream, u64::from_le_bytes(len) as usize).map_err(|e| io_error("store get", e))
    }
}

/// Where the rendezvous happens (`--nnodes --node-rank --master-addr --master-port`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RendezvousConfig {
    pub nnodes: usize,
    pub node_rank: usize,
    pub master_addr: String,
    pub master_port: u16,
}

impl RendezvousConfig {
    pub fn single_node() -> Self {
        Self { nnodes: 1, node_rank: 0, master_addr: "127.0.0.1".into(), master_port: 29500 }
    }

    pub fn validate(&self) -> CommResult<()> {
        if self.nnodes == 0 || self.node_rank >= self.nnodes {
            return Err(format!("--node-rank {} is not valid for --nnodes {}", self.node_rank, self.nnodes));
        }
        if self.nnodes > 1 && self.master_addr.trim().is_empty() {
            return Err("--master-addr is required when --nnodes is larger than 1".into());
        }
        Ok(())
    }
}

/// Node 0 starts the store; every node (node 0 included) returns a client.
pub struct Rendezvous {
    _server: Option<TcpStore>,
    pub client: Arc<StoreClient>,
}

impl Rendezvous {
    pub fn start(config: &RendezvousConfig) -> CommResult<Self> {
        config.validate()?;
        let timeout = default_timeout();
        let server = if config.node_rank == 0 { Some(TcpStore::serve(&format!("0.0.0.0:{}", config.master_port))?) } else { None };
        let target = match &server {
            Some(server) => format!("127.0.0.1:{}", server.addr().port()),
            None => format!("{}:{}", config.master_addr, config.master_port),
        };
        let client = Arc::new(StoreClient::connect(&target, timeout)?);
        Ok(Self { _server: server, client })
    }
}

struct SocketShared {
    rank: usize,
    size: usize,
    inbox: Inbox,
    writers: Vec<Mutex<Option<TcpStream>>>,
    aborted: Arc<AtomicBool>,
}

/// Full mesh of TCP connections between all ranks of the world. Frame: `chan: u32, len: u64, bytes`.
pub struct SocketFabric {
    shared: Arc<SocketShared>,
}

impl SocketFabric {
    /// Join the mesh as `rank` of `size`. `host` is the address the other ranks reach this rank on. The ranks find each other
    /// through `store` (`fabric/<group>/addr/<rank>`); `group` separates several meshes that share one store.
    pub fn connect(store: &StoreClient, group: &str, rank: usize, size: usize, host: &str) -> CommResult<Arc<dyn Fabric>> {
        let timeout = default_timeout();
        let listener = TcpListener::bind("0.0.0.0:0").map_err(|e| io_error("cannot open a listening socket", e))?;
        let port = listener.local_addr().map_err(|e| io_error("local address", e))?.port();
        store.set(&format!("fabric/{group}/addr/{rank}"), format!("{host}:{port}").as_bytes())?;
        let aborted = Arc::new(AtomicBool::new(false));
        let shared = Arc::new(SocketShared {
            rank,
            size,
            inbox: Inbox::new(aborted.clone(), timeout),
            writers: (0..size).map(|_| Mutex::new(None)).collect(),
            aborted,
        });
        // higher ranks connect to lower ranks; this rank accepts from the higher ones
        let incoming = size - 1 - rank;
        let accept_shared = shared.clone();
        let accept = std::thread::Builder::new()
            .name("fllm-accept".into())
            .spawn(move || -> CommResult<()> {
                for _ in 0..incoming {
                    let (mut stream, _) = listener.accept().map_err(|e| io_error("accept", e))?;
                    let mut peer = [0u8; 8];
                    stream.read_exact(&mut peer).map_err(|e| io_error("handshake", e))?;
                    let peer = u64::from_le_bytes(peer) as usize;
                    if peer >= accept_shared.size || peer <= accept_shared.rank {
                        return Err(format!("unexpected connection from rank {peer}"));
                    }
                    adopt(&accept_shared, peer, stream)?;
                }
                Ok(())
            })
            .map_err(|e| io_error("cannot start the accept thread", e))?;
        for peer in 0..rank {
            let address = String::from_utf8_lossy(&store.get(&format!("fabric/{group}/addr/{peer}"))?).into_owned();
            let mut stream = connect_retry(&address, timeout)?;
            stream.write_all(&(rank as u64).to_le_bytes()).map_err(|e| io_error("handshake", e))?;
            adopt(&shared, peer, stream)?;
        }
        accept.join().map_err(|_| "the accept thread panicked".to_owned())??;
        Ok(Arc::new(SocketFabric { shared }))
    }
}

fn connect_retry(address: &str, timeout: Duration) -> CommResult<TcpStream> {
    let deadline = Instant::now() + timeout;
    loop {
        match TcpStream::connect(address) {
            Ok(stream) => return Ok(stream),
            Err(e) if Instant::now() >= deadline => return Err(format!("cannot connect to {address}: {e}")),
            Err(_) => std::thread::sleep(Duration::from_millis(100)),
        }
    }
}

/// Start the reader thread for `peer` and remember the write half.
fn adopt(shared: &Arc<SocketShared>, peer: usize, stream: TcpStream) -> CommResult<()> {
    let _ = stream.set_nodelay(true);
    let mut reader = stream.try_clone().map_err(|e| io_error("clone socket", e))?;
    *shared.writers[peer].lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(stream);
    let shared = shared.clone();
    std::thread::Builder::new()
        .name(format!("fllm-recv-{peer}"))
        .spawn(move || loop {
            let mut head = [0u8; 12];
            if reader.read_exact(&mut head).is_err() {
                shared.inbox.close(peer);
                return;
            }
            let chan = u32::from_le_bytes([head[0], head[1], head[2], head[3]]);
            let len = u64::from_le_bytes([head[4], head[5], head[6], head[7], head[8], head[9], head[10], head[11]]) as usize;
            match read_exact_vec(&mut reader, len) {
                Ok(data) => shared.inbox.push(peer, chan, data),
                Err(_) => {
                    shared.inbox.close(peer);
                    return;
                }
            }
        })
        .map_err(|e| io_error("cannot start the receive thread", e))?;
    Ok(())
}

impl Fabric for SocketFabric {
    fn world_size(&self) -> usize {
        self.shared.size
    }
    fn world_rank(&self) -> usize {
        self.shared.rank
    }
    fn send(&self, dst: usize, chan: u32, data: Vec<u8>) -> CommResult<()> {
        if dst == self.shared.rank {
            self.shared.inbox.push(dst, chan, data);
            return Ok(());
        }
        let writer = self.shared.writers.get(dst).ok_or_else(|| format!("rank {dst} is outside the world of {}", self.shared.size))?;
        let mut guard = writer.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let stream = guard.as_mut().ok_or_else(|| format!("no connection to rank {dst}"))?;
        let mut frame = Vec::with_capacity(12 + data.len());
        frame.extend_from_slice(&chan.to_le_bytes());
        frame.extend_from_slice(&(data.len() as u64).to_le_bytes());
        frame.extend_from_slice(&data);
        stream.write_all(&frame).map_err(|e| format!("send to rank {dst}: {e}"))
    }
    fn recv(&self, src: usize, chan: u32) -> CommResult<Vec<u8>> {
        self.shared.inbox.pop(src, chan)
    }
    fn abort(&self) {
        self.shared.aborted.store(true, Ordering::Relaxed);
        self.shared.inbox.wake_all();
        for writer in &self.shared.writers {
            if let Some(stream) = writer.lock().unwrap_or_else(std::sync::PoisonError::into_inner).as_ref() {
                let _ = stream.shutdown(std::net::Shutdown::Both);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comm::mesh::MeshComm;
    use crate::comm::Communicator;

    #[test]
    fn the_store_hands_values_to_waiting_clients() {
        let store = TcpStore::serve("127.0.0.1:0").unwrap();
        let address = store.addr().to_string();
        let reader_address = address.clone();
        let reader = std::thread::spawn(move || {
            let client = StoreClient::connect(&reader_address, Duration::from_secs(10)).unwrap();
            client.get("late/key").unwrap()
        });
        std::thread::sleep(Duration::from_millis(100));
        let writer = StoreClient::connect(&address, Duration::from_secs(10)).unwrap();
        writer.set("late/key", b"unique-id-bytes").unwrap();
        assert_eq!(reader.join().unwrap(), b"unique-id-bytes");
        let short = StoreClient::connect(&address, Duration::from_millis(300)).unwrap();
        assert!(short.get("never").unwrap_err().contains("timed out"));
    }

    #[test]
    fn four_ranks_over_loopback_sockets_run_collectives() {
        let size = 4;
        let store = TcpStore::serve("127.0.0.1:0").unwrap();
        let address = store.addr().to_string();
        let handles: Vec<_> = (0..size)
            .map(|rank| {
                let address = address.clone();
                std::thread::spawn(move || -> CommResult<Vec<f32>> {
                    let client = StoreClient::connect(&address, Duration::from_secs(20))?;
                    let fabric = SocketFabric::connect(&client, "t", rank, size, "127.0.0.1")?;
                    let comm = MeshComm::world(fabric, "socket");
                    let mut v: Vec<f32> = (0..9).map(|i| (rank * 10 + i) as f32).collect();
                    comm.all_reduce(&mut v)?;
                    let mut id = if rank == 0 { b"nccl-unique-id".to_vec() } else { Vec::new() };
                    comm.broadcast_bytes(&mut id, 0)?;
                    assert_eq!(id, b"nccl-unique-id");
                    let pair = comm.split(rank / 2, rank)?;
                    let mut p = [1.0f32];
                    pair.all_reduce(&mut p)?;
                    assert_eq!(p[0], 2.0);
                    comm.barrier()?;
                    Ok(v)
                })
            })
            .collect();
        for handle in handles {
            let v = handle.join().unwrap().unwrap();
            let expected: Vec<f32> = (0..9).map(|i| (0..size).map(|r| (r * 10 + i) as f32).sum()).collect();
            assert_eq!(v, expected);
        }
    }

    #[test]
    fn rendezvous_settings_are_validated() {
        assert!(RendezvousConfig::single_node().validate().is_ok());
        let bad = RendezvousConfig { nnodes: 2, node_rank: 2, master_addr: "a".into(), master_port: 1 };
        assert!(bad.validate().is_err());
        let missing = RendezvousConfig { nnodes: 2, node_rank: 1, master_addr: String::new(), master_port: 1 };
        assert!(missing.validate().unwrap_err().contains("--master-addr"));
    }
}
