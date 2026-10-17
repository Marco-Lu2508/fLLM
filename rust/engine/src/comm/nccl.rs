//! NCCL through `dlopen`: the library is looked up when a parallel group starts, so release binaries run on machines without NCCL and
//! tell the user what to install. Symbols used: ncclGetVersion, ncclGetUniqueId, ncclCommInitRank, ncclCommDestroy, ncclAllReduce,
//! ncclAllGather, ncclReduceScatter, ncclBroadcast, ncclSend, ncclRecv, ncclGroupStart, ncclGroupEnd, ncclGetErrorString.
//! The ranks are threads of one process; every rank creates its communicator with `ncclCommInitRank` and the unique id travels over the
//! host-memory control group (for several nodes that group runs over TCP).
use super::cuda_ffi::{set_device, DeviceMem, RankStream};
use super::{bytes_to_f32s, f32s_to_bytes, CommResult, Communicator, DType, DeviceCollectives, DevPtr, StreamHandle};
use libloading::Library;
use std::ffi::{c_char, c_int, c_void, CStr};
use std::sync::{Arc, Mutex, OnceLock};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct UniqueId {
    internal: [u8; 128],
}

type Comm = *mut c_void;
type Stream = *mut c_void;
type GetVersion = unsafe extern "C" fn(*mut c_int) -> c_int;
type GetUniqueId = unsafe extern "C" fn(*mut UniqueId) -> c_int;
type CommInitRank = unsafe extern "C" fn(*mut Comm, c_int, UniqueId, c_int) -> c_int;
type CommDestroy = unsafe extern "C" fn(Comm) -> c_int;
type AllReduce = unsafe extern "C" fn(*const c_void, *mut c_void, usize, c_int, c_int, Comm, Stream) -> c_int;
type AllGather = unsafe extern "C" fn(*const c_void, *mut c_void, usize, c_int, Comm, Stream) -> c_int;
type ReduceScatter = unsafe extern "C" fn(*const c_void, *mut c_void, usize, c_int, c_int, Comm, Stream) -> c_int;
type Broadcast = unsafe extern "C" fn(*const c_void, *mut c_void, usize, c_int, c_int, Comm, Stream) -> c_int;
type SendRecv = unsafe extern "C" fn(*const c_void, usize, c_int, c_int, Comm, Stream) -> c_int;
type RecvOnly = unsafe extern "C" fn(*mut c_void, usize, c_int, c_int, Comm, Stream) -> c_int;
type Group = unsafe extern "C" fn() -> c_int;
type ErrorString = unsafe extern "C" fn(c_int) -> *const c_char;

/// The loaded library and its entry points.
pub struct NcclLib {
    _library: Library,
    pub version: i32,
    get_unique_id: GetUniqueId,
    comm_init_rank: CommInitRank,
    comm_destroy: CommDestroy,
    all_reduce: AllReduce,
    all_gather: AllGather,
    reduce_scatter: ReduceScatter,
    broadcast: Broadcast,
    send: SendRecv,
    recv: RecvOnly,
    group_start: Group,
    group_end: Group,
    error_string: ErrorString,
}

const INSTALL_HINT: &str = "install the NCCL runtime library (Debian/Ubuntu: `sudo apt install libnccl2`, other systems: the nccl package of your CUDA \
                            repository), or point FLLM_NCCL_PATH at libnccl.so.2, or use --distributed-backend p2p";

impl NcclLib {
    /// Load libnccl (once per process).
    pub fn open() -> CommResult<Arc<NcclLib>> {
        static LIB: OnceLock<Result<Arc<NcclLib>, String>> = OnceLock::new();
        LIB.get_or_init(Self::load).clone()
    }

    fn load() -> CommResult<Arc<NcclLib>> {
        let mut candidates: Vec<String> = Vec::new();
        if let Ok(path) = std::env::var("FLLM_NCCL_PATH") {
            candidates.push(path);
        }
        candidates.extend(["libnccl.so.2", "libnccl.so"].map(String::from));
        let mut last = String::new();
        let mut library = None;
        for name in &candidates {
            // SAFETY: loading a shared library runs its initializers; NCCL is a plain CUDA library.
            match unsafe { Library::new(name) } {
                Ok(l) => {
                    library = Some(l);
                    break;
                }
                Err(e) => last = e.to_string(),
            }
        }
        let library = library.ok_or_else(|| format!("libnccl could not be loaded ({last}); {INSTALL_HINT}"))?;
        // SAFETY (all symbol lookups): the signatures are those of nccl.h; the function pointers are copied out and `_library` keeps
        // the library mapped for as long as the struct lives.
        unsafe {
            macro_rules! sym {
                ($name:literal, $ty:ty) => {
                    *library.get::<$ty>($name).map_err(|e| format!("libnccl lacks {}: {e}; {INSTALL_HINT}", String::from_utf8_lossy(&$name[..$name.len() - 1])))?
                };
            }
            let get_version: GetVersion = sym!(b"ncclGetVersion\0", GetVersion);
            let mut version: c_int = 0;
            get_version(&mut version);
            Ok(Arc::new(NcclLib {
                version,
                get_unique_id: sym!(b"ncclGetUniqueId\0", GetUniqueId),
                comm_init_rank: sym!(b"ncclCommInitRank\0", CommInitRank),
                comm_destroy: sym!(b"ncclCommDestroy\0", CommDestroy),
                all_reduce: sym!(b"ncclAllReduce\0", AllReduce),
                all_gather: sym!(b"ncclAllGather\0", AllGather),
                reduce_scatter: sym!(b"ncclReduceScatter\0", ReduceScatter),
                broadcast: sym!(b"ncclBroadcast\0", Broadcast),
                send: sym!(b"ncclSend\0", SendRecv),
                recv: sym!(b"ncclRecv\0", RecvOnly),
                group_start: sym!(b"ncclGroupStart\0", Group),
                group_end: sym!(b"ncclGroupEnd\0", Group),
                error_string: sym!(b"ncclGetErrorString\0", ErrorString),
                _library: library,
            }))
        }
    }

    fn check(&self, what: &str, code: c_int) -> CommResult<()> {
        if code == 0 {
            return Ok(());
        }
        // SAFETY: ncclGetErrorString returns a static string.
        let text = unsafe { CStr::from_ptr((self.error_string)(code)) }.to_string_lossy().into_owned();
        Err(format!("{what} failed: NCCL error {code} ({text})"))
    }

    /// `2.18.3` style text of the loaded version.
    pub fn version_text(&self) -> String {
        let v = self.version;
        if v >= 10000 {
            format!("{}.{}.{}", v / 10000, (v % 10000) / 100, v % 100)
        } else {
            format!("{}.{}.{}", v / 1000, (v % 1000) / 100, v % 100)
        }
    }
}

/// One rank's NCCL communicator, bound to its device and a stream of its own.
pub struct NcclComm {
    lib: Arc<NcclLib>,
    comm: Comm,
    rank: usize,
    size: usize,
    device: usize,
    stream: RankStream,
    /// Same ranks, host memory: control messages, unique ids, splits.
    host: Arc<dyn Communicator>,
    staging: Mutex<Option<DeviceMem>>,
}

// SAFETY: the communicator handle is used by the rank's thread; NCCL calls on one communicator are serialized by the callers
// (SPMD), and the staging buffer is behind a mutex.
unsafe impl Send for NcclComm {}
unsafe impl Sync for NcclComm {}

impl NcclComm {
    /// Collective over the ranks of `host`: create the NCCL communicator of the same group on `device`.
    pub fn connect(host: &Arc<dyn Communicator>, device: usize) -> CommResult<Arc<NcclComm>> {
        let lib = NcclLib::open()?;
        let (rank, size) = (host.rank(), host.size());
        set_device(device)?;
        let mut id_bytes = Vec::new();
        if rank == 0 {
            let mut id = UniqueId { internal: [0; 128] };
            // SAFETY: valid out pointer.
            lib.check("ncclGetUniqueId", unsafe { (lib.get_unique_id)(&mut id) })?;
            id_bytes = id.internal.to_vec();
        }
        host.broadcast_bytes(&mut id_bytes, 0)?;
        let mut id = UniqueId { internal: [0; 128] };
        if id_bytes.len() != 128 {
            return Err("the NCCL unique id has the wrong size".into());
        }
        id.internal.copy_from_slice(&id_bytes);
        let mut comm: Comm = std::ptr::null_mut();
        // SAFETY: blocks until all `size` ranks have called it with the same id.
        lib.check(&format!("ncclCommInitRank (rank {rank} of {size}, device {device})"), unsafe {
            (lib.comm_init_rank)(&mut comm, size as c_int, id, rank as c_int)
        })?;
        Ok(Arc::new(NcclComm { lib, comm, rank, size, device, stream: RankStream::new(device)?, host: host.clone(), staging: Mutex::new(None) }))
    }

    pub fn device(&self) -> usize {
        self.device
    }

    pub fn stream(&self) -> &RankStream {
        &self.stream
    }

    fn check(&self, what: &str, code: c_int) -> CommResult<()> {
        self.lib.check(what, code)
    }

    /// Run `f(base)` with a device buffer of at least `bytes` bytes.
    fn staged<T>(&self, bytes: usize, f: impl FnOnce(&DeviceMem) -> CommResult<T>) -> CommResult<T> {
        let mut guard = self.staging.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if guard.as_ref().is_none_or(|m| m.bytes < bytes) {
            *guard = Some(DeviceMem::new(self.device, bytes.next_power_of_two())?);
        }
        match guard.as_ref() {
            Some(mem) => f(mem),
            None => Err("no staging buffer".into()),
        }
    }

    fn all_to_all_device(&self, send: DevPtr, send_counts: &[usize], recv: DevPtr, recv_counts: &[usize], dtype: DType, stream: StreamHandle) -> CommResult<()> {
        if send_counts.len() != self.size || recv_counts.len() != self.size {
            return Err(format!("all_to_all needs {} counts per direction", self.size));
        }
        let width = dtype.size() as u64;
        set_device(self.device)?;
        // SAFETY: group calls bracket the sends and receives; the pointers are the caller's device buffers.
        unsafe {
            self.check("ncclGroupStart", (self.lib.group_start)())?;
            let (mut so, mut ro) = (0u64, 0u64);
            for peer in 0..self.size {
                let r = (self.lib.send)((send + so * width) as *const c_void, send_counts[peer], dtype.nccl_code(), peer as c_int, self.comm, stream as Stream);
                let r2 = (self.lib.recv)((recv + ro * width) as *mut c_void, recv_counts[peer], dtype.nccl_code(), peer as c_int, self.comm, stream as Stream);
                so += send_counts[peer] as u64;
                ro += recv_counts[peer] as u64;
                self.check("ncclSend", r)?;
                self.check("ncclRecv", r2)?;
            }
            self.check("ncclGroupEnd", (self.lib.group_end)())
        }
    }
}

impl Drop for NcclComm {
    fn drop(&mut self) {
        let _ = set_device(self.device);
        // SAFETY: destroyed once, on the rank's thread.
        unsafe { (self.lib.comm_destroy)(self.comm) };
    }
}

impl DeviceCollectives for NcclComm {
    fn all_reduce(&self, buf: DevPtr, count: usize, dtype: DType, stream: StreamHandle) -> CommResult<()> {
        set_device(self.device)?;
        // SAFETY: in-place reduction on the caller's device buffer.
        self.check("ncclAllReduce", unsafe {
            (self.lib.all_reduce)(buf as *const c_void, buf as *mut c_void, count, dtype.nccl_code(), 0, self.comm, stream as Stream)
        })
    }
    fn all_gather(&self, send: DevPtr, recv: DevPtr, count: usize, dtype: DType, stream: StreamHandle) -> CommResult<()> {
        set_device(self.device)?;
        // SAFETY: the caller's device buffers.
        self.check("ncclAllGather", unsafe { (self.lib.all_gather)(send as *const c_void, recv as *mut c_void, count, dtype.nccl_code(), self.comm, stream as Stream) })
    }
    fn reduce_scatter(&self, send: DevPtr, recv: DevPtr, count: usize, dtype: DType, stream: StreamHandle) -> CommResult<()> {
        set_device(self.device)?;
        // SAFETY: the caller's device buffers.
        self.check("ncclReduceScatter", unsafe {
            (self.lib.reduce_scatter)(send as *const c_void, recv as *mut c_void, count, dtype.nccl_code(), 0, self.comm, stream as Stream)
        })
    }
    fn broadcast(&self, buf: DevPtr, count: usize, dtype: DType, root: usize, stream: StreamHandle) -> CommResult<()> {
        set_device(self.device)?;
        // SAFETY: in-place broadcast on the caller's device buffer.
        self.check("ncclBroadcast", unsafe {
            (self.lib.broadcast)(buf as *const c_void, buf as *mut c_void, count, dtype.nccl_code(), root as c_int, self.comm, stream as Stream)
        })
    }
    fn all_to_all(&self, send: DevPtr, send_counts: &[usize], recv: DevPtr, recv_counts: &[usize], dtype: DType, stream: StreamHandle) -> CommResult<()> {
        self.all_to_all_device(send, send_counts, recv, recv_counts, dtype, stream)
    }
    fn send(&self, buf: DevPtr, count: usize, dtype: DType, peer: usize, stream: StreamHandle) -> CommResult<()> {
        set_device(self.device)?;
        // SAFETY: the caller's device buffer.
        self.check("ncclSend", unsafe { (self.lib.send)(buf as *const c_void, count, dtype.nccl_code(), peer as c_int, self.comm, stream as Stream) })
    }
    fn recv(&self, buf: DevPtr, count: usize, dtype: DType, peer: usize, stream: StreamHandle) -> CommResult<()> {
        set_device(self.device)?;
        // SAFETY: the caller's device buffer.
        self.check("ncclRecv", unsafe { (self.lib.recv)(buf as *mut c_void, count, dtype.nccl_code(), peer as c_int, self.comm, stream as Stream) })
    }
}

impl Communicator for NcclComm {
    fn rank(&self) -> usize {
        self.rank
    }
    fn size(&self) -> usize {
        self.size
    }
    fn backend(&self) -> &'static str {
        "nccl"
    }

    fn all_reduce(&self, data: &mut [f32]) -> CommResult<()> {
        let bytes = f32s_to_bytes(data);
        self.staged(bytes.len(), |mem| {
            mem.upload(&bytes)?;
            DeviceCollectives::all_reduce(self, mem.addr(), data.len(), DType::F32, self.stream.handle())?;
            self.stream.sync()?;
            let mut out = vec![0u8; bytes.len()];
            mem.download(&mut out)?;
            bytes_to_f32s(&out, data)
        })
    }

    fn all_gather(&self, send: &[f32], recv: &mut [f32]) -> CommResult<()> {
        if recv.len() != self.size * send.len() {
            return Err("all_gather: receive buffer has the wrong size".into());
        }
        let bytes = f32s_to_bytes(send);
        self.staged(bytes.len() * (1 + self.size), |mem| {
            mem.upload(&bytes)?;
            let out_at = mem.addr() + bytes.len() as u64;
            DeviceCollectives::all_gather(self, mem.addr(), out_at, send.len(), DType::F32, self.stream.handle())?;
            self.stream.sync()?;
            let mut all = vec![0u8; bytes.len() * (1 + self.size)];
            mem.download(&mut all)?;
            bytes_to_f32s(&all[bytes.len()..], recv)
        })
    }

    fn reduce_scatter(&self, send: &[f32], recv: &mut [f32]) -> CommResult<()> {
        if send.len() != self.size * recv.len() {
            return Err("reduce_scatter: send buffer has the wrong size".into());
        }
        let bytes = f32s_to_bytes(send);
        let out_bytes = recv.len() * 4;
        self.staged(bytes.len() + out_bytes, |mem| {
            mem.upload(&bytes)?;
            let out_at = mem.addr() + bytes.len() as u64;
            DeviceCollectives::reduce_scatter(self, mem.addr(), out_at, recv.len(), DType::F32, self.stream.handle())?;
            self.stream.sync()?;
            let mut all = vec![0u8; bytes.len() + out_bytes];
            mem.download(&mut all)?;
            bytes_to_f32s(&all[bytes.len()..], recv)
        })
    }

    fn broadcast(&self, data: &mut [f32], root: usize) -> CommResult<()> {
        let bytes = f32s_to_bytes(data);
        self.staged(bytes.len(), |mem| {
            mem.upload(&bytes)?;
            DeviceCollectives::broadcast(self, mem.addr(), data.len(), DType::F32, root, self.stream.handle())?;
            self.stream.sync()?;
            let mut out = vec![0u8; bytes.len()];
            mem.download(&mut out)?;
            bytes_to_f32s(&out, data)
        })
    }

    fn all_to_all(&self, send: &[f32], send_counts: &[usize], recv: &mut [f32], recv_counts: &[usize]) -> CommResult<()> {
        if send_counts.iter().sum::<usize>() != send.len() || recv_counts.iter().sum::<usize>() != recv.len() {
            return Err("all_to_all: the counts do not add up to the buffer sizes".into());
        }
        let bytes = f32s_to_bytes(send);
        let out_bytes = recv.len() * 4;
        self.staged(bytes.len() + out_bytes, |mem| {
            mem.upload(&bytes)?;
            let out_at = mem.addr() + bytes.len() as u64;
            self.all_to_all_device(mem.addr(), send_counts, out_at, recv_counts, DType::F32, self.stream.handle())?;
            self.stream.sync()?;
            let mut all = vec![0u8; bytes.len() + out_bytes];
            mem.download(&mut all)?;
            bytes_to_f32s(&all[bytes.len()..], recv)
        })
    }

    fn send(&self, data: &[f32], peer: usize) -> CommResult<()> {
        let bytes = f32s_to_bytes(data);
        self.staged(bytes.len(), |mem| {
            mem.upload(&bytes)?;
            DeviceCollectives::send(self, mem.addr(), data.len(), DType::F32, peer, self.stream.handle())?;
            self.stream.sync()
        })
    }

    fn recv(&self, data: &mut [f32], peer: usize) -> CommResult<()> {
        let n = data.len() * 4;
        self.staged(n, |mem| {
            DeviceCollectives::recv(self, mem.addr(), data.len(), DType::F32, peer, self.stream.handle())?;
            self.stream.sync()?;
            let mut out = vec![0u8; n];
            mem.download(&mut out)?;
            bytes_to_f32s(&out, data)
        })
    }

    fn barrier(&self) -> CommResult<()> {
        self.host.barrier()
    }
    fn broadcast_bytes(&self, data: &mut Vec<u8>, root: usize) -> CommResult<()> {
        self.host.broadcast_bytes(data, root)
    }
    fn send_bytes(&self, data: &[u8], peer: usize) -> CommResult<()> {
        self.host.send_bytes(data, peer)
    }
    fn recv_bytes(&self, peer: usize) -> CommResult<Vec<u8>> {
        self.host.recv_bytes(peer)
    }
    fn exchange_counts(&self, send_counts: &[usize]) -> CommResult<Vec<usize>> {
        self.host.exchange_counts(send_counts)
    }
    fn split(&self, color: usize, key: usize) -> CommResult<Arc<dyn Communicator>> {
        let sub = self.host.split(color, key)?;
        Ok(NcclComm::connect(&sub, self.device)? as Arc<dyn Communicator>)
    }
    fn device(&self) -> Option<&dyn DeviceCollectives> {
        Some(self)
    }
    fn abort(&self) {
        self.host.abort();
    }
}
