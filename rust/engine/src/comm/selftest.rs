//! `fllm-rs selftest-comm`: every collective on data whose result is known exactly (small integers in f32, so sums are exact).
//! The host-memory implementation always runs; with a CUDA or HIP build and libnccl / librccl the same checks run through NCCL / RCCL on host
//! slices and on device buffers: with one rank on a single GPU, with all GPUs when there are two or more. The custom one-shot all-reduce runs
//! with several ranks sharing GPU 0 as well as across GPUs.
use super::local::{run_ranks, LocalComm};
use super::{CommResult, Communicator};
use std::sync::Arc;

pub const HELP: &str = "usage: fllm-rs selftest-comm [--ranks N]\n\
Checks all-reduce, all-gather, reduce-scatter, broadcast, all-to-all (uneven splits) and send/recv with exact expected results:\n\
first on the host-memory communicator (N threads, default 4), then, in a CUDA build with at least two GPUs and libnccl installed,\n\
through NCCL (RCCL in a HIP build) with one rank on host slices and device buffers, with all visible GPUs when there are two or more,
and the custom one-shot all-reduce kernel (also with several ranks sharing GPU 0).";

fn value(rank: usize, i: usize) -> f32 {
    ((rank + 1) * (i % 7 + 1)) as f32
}

/// The checks every communicator must pass; `n` ranks, called by each rank.
pub fn check_collectives(comm: &Arc<dyn Communicator>) -> CommResult<usize> {
    let (rank, n) = (comm.rank(), comm.size());
    let mut checks = 0;
    let triangle = (n * (n + 1) / 2) as f32;
    let mut v: Vec<f32> = (0..37).map(|i| value(rank, i)).collect();
    comm.all_reduce(&mut v)?;
    for (i, x) in v.iter().enumerate() {
        if *x != triangle * (i % 7 + 1) as f32 {
            return Err(format!("all_reduce: element {i} is {x}, expected {}", triangle * (i % 7 + 1) as f32));
        }
    }
    checks += 1;
    let mine: Vec<f32> = (0..5).map(|i| value(rank, i)).collect();
    let mut all = vec![0.0; 5 * n];
    comm.all_gather(&mine, &mut all)?;
    for r in 0..n {
        for i in 0..5 {
            if all[r * 5 + i] != value(r, i) {
                return Err(format!("all_gather: block {r} element {i} is {}, expected {}", all[r * 5 + i], value(r, i)));
            }
        }
    }
    checks += 1;
    let send: Vec<f32> = (0..3 * n).map(|i| value(rank, i)).collect();
    let mut part = vec![0.0; 3];
    comm.reduce_scatter(&send, &mut part)?;
    for (i, x) in part.iter().enumerate() {
        let expected: f32 = (0..n).map(|r| value(r, rank * 3 + i)).sum();
        if *x != expected {
            return Err(format!("reduce_scatter: element {i} is {x}, expected {expected}"));
        }
    }
    checks += 1;
    let root = n - 1;
    let mut b = if rank == root { vec![3.0, 1.0, 4.0, 1.0, 5.0] } else { vec![0.0; 5] };
    comm.broadcast(&mut b, root)?;
    if b != [3.0, 1.0, 4.0, 1.0, 5.0] {
        return Err(format!("broadcast: got {b:?}"));
    }
    checks += 1;
    let send_counts: Vec<usize> = (0..n).map(|d| rank + d + 1).collect();
    let recv_counts = comm.exchange_counts(&send_counts)?;
    let mut sendbuf = Vec::new();
    for (d, c) in send_counts.iter().enumerate() {
        sendbuf.extend(std::iter::repeat_n((100 * rank + d) as f32, *c));
    }
    let mut recvbuf = vec![0.0; recv_counts.iter().sum()];
    comm.all_to_all(&sendbuf, &send_counts, &mut recvbuf, &recv_counts)?;
    let mut at = 0;
    for (s, c) in recv_counts.iter().enumerate() {
        if recvbuf[at..at + c].iter().any(|x| *x != (100 * s + rank) as f32) {
            return Err(format!("all_to_all: the block from rank {s} is wrong"));
        }
        at += c;
    }
    checks += 1;
    if n > 1 {
        comm.send(&[rank as f32, 9.0], (rank + 1) % n)?;
        let mut got = [0.0; 2];
        comm.recv(&mut got, (rank + n - 1) % n)?;
        if got != [((rank + n - 1) % n) as f32, 9.0] {
            return Err(format!("send/recv: got {got:?}"));
        }
        checks += 1;
    }
    comm.barrier()?;
    Ok(checks)
}

#[cfg(feature = "cuda")]
fn check_device(comm: &Arc<dyn Communicator>, device: usize) -> CommResult<usize> {
    use super::cuda_ffi::{DeviceMem, RankStream};
    use super::DType;
    let Some(dev) = comm.device() else { return Err("this communicator has no device collectives".into()) };
    let (rank, n) = (comm.rank(), comm.size());
    let stream = RankStream::new(device)?;
    let h = stream.handle();
    let to_bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let from_bytes = |b: &[u8]| b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect::<Vec<f32>>();
    let mut checks = 0;
    // all-reduce on a device buffer
    let data: Vec<f32> = (0..1000).map(|i| value(rank, i)).collect();
    let buf = DeviceMem::new(device, data.len() * 4)?;
    buf.upload(&to_bytes(&data))?;
    super::cuda_ffi::default_stream_sync()?;
    dev.all_reduce(buf.addr(), data.len(), DType::F32, h)?;
    stream.sync()?;
    let mut out = vec![0u8; data.len() * 4];
    buf.download(&mut out)?;
    let triangle = (n * (n + 1) / 2) as f32;
    for (i, x) in from_bytes(&out).iter().enumerate() {
        if *x != triangle * (i % 7 + 1) as f32 {
            return Err(format!("device all_reduce: element {i} is {x}"));
        }
    }
    checks += 1;
    // all-gather and reduce-scatter
    let mine: Vec<f32> = (0..16).map(|i| value(rank, i)).collect();
    let send = DeviceMem::new(device, 16 * 4 * n)?;
    let recv = DeviceMem::new(device, 16 * 4 * n)?;
    send.upload(&to_bytes(&mine))?;
    super::cuda_ffi::default_stream_sync()?;
    dev.all_gather(send.addr(), recv.addr(), 16, DType::F32, h)?;
    stream.sync()?;
    let mut out = vec![0u8; 16 * 4 * n];
    recv.download(&mut out)?;
    let all = from_bytes(&out);
    for r in 0..n {
        for i in 0..16 {
            if all[r * 16 + i] != value(r, i) {
                return Err(format!("device all_gather: block {r} element {i} is {}", all[r * 16 + i]));
            }
        }
    }
    checks += 1;
    let full: Vec<f32> = (0..16 * n).map(|i| value(rank, i)).collect();
    send.upload(&to_bytes(&full))?;
    super::cuda_ffi::default_stream_sync()?;
    dev.reduce_scatter(send.addr(), recv.addr(), 16, DType::F32, h)?;
    stream.sync()?;
    let mut out = vec![0u8; 16 * 4];
    recv.download(&mut out)?;
    for (i, x) in from_bytes(&out).iter().enumerate() {
        let expected: f32 = (0..n).map(|r| value(r, rank * 16 + i)).sum();
        if *x != expected {
            return Err(format!("device reduce_scatter: element {i} is {x}, expected {expected}"));
        }
    }
    checks += 1;
    Ok(checks)
}

/// The custom one-shot all-reduce on device buffers: many sizes, consecutive calls (the two staging halves), several calls in flight at once,
/// f32 / f16 / bf16. Values are small integers, so every sum is exact in every type and must be identical on all ranks.
/// Ranks may share one GPU (their kernels then wait for each other on the device), so nothing here synchronizes the whole device or frees
/// memory while another rank's kernel can still be running: buffers are allocated up front and released after a barrier.
#[cfg(feature = "cuda")]
fn check_custom(host: &Arc<dyn Communicator>, device: usize) -> CommResult<usize> {
    use super::cuda_ffi::{default_stream_sync, DeviceMem, RankStream};
    use super::oneshot::CustomAllReduce;
    use super::DType;
    const MAX: usize = 70_001;
    const QUEUED: usize = 24;
    let (rank, n) = (host.rank(), host.size());
    let car = CustomAllReduce::connect(host, device, 1 << 20)?;
    let stream = RankStream::new(device)?;
    let h = stream.handle();
    let triangle = (n * (n + 1) / 2) as f32;
    let mut checks = 0;
    let encode = |vals: &[f32], dtype: DType| -> Vec<u8> {
        match dtype {
            DType::F32 => vals.iter().flat_map(|x| x.to_le_bytes()).collect(),
            DType::F16 => vals.iter().flat_map(|x| half::f16::from_f32(*x).to_bits().to_le_bytes()).collect(),
            _ => vals.iter().flat_map(|x| half::bf16::from_f32(*x).to_bits().to_le_bytes()).collect(),
        }
    };
    let decode = |bytes: &[u8], dtype: DType| -> Vec<f32> {
        match dtype {
            DType::F32 => bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
            DType::F16 => bytes.as_chunks::<2>().0.iter().map(|c| half::f16::from_bits(u16::from_le_bytes(*c)).to_f32()).collect(),
            _ => bytes.as_chunks::<2>().0.iter().map(|c| half::bf16::from_bits(u16::from_le_bytes(*c)).to_f32()).collect(),
        }
    };
    let buf = DeviceMem::new(device, MAX * 4)?;
    let queued: Vec<DeviceMem> = (0..QUEUED).map(|_| DeviceMem::new(device, 3000 * 4)).collect::<Result<_, _>>()?;
    let result = (|| -> CommResult<()> {
        for dtype in [DType::F32, DType::F16, DType::Bf16] {
            for count in [1usize, 7, 37, 1000, 4096, MAX] {
                for rep in 0..3 {
                    let vals: Vec<f32> = (0..count).map(|i| value(rank, i + rep)).collect();
                    buf.upload(&encode(&vals, dtype))?;
                    // the synchronous upload may return before the transfer is done; the kernel runs on another (non-blocking) stream
                    default_stream_sync()?;
                    car.all_reduce(buf.addr(), count, dtype, h)?;
                    stream.sync()?;
                    let mut out = vec![0u8; count * dtype.size()];
                    buf.download(&mut out)?;
                    let got = decode(&out, dtype);
                    let wrong: Vec<usize> = (0..count).filter(|&i| got[i] != triangle * ((i + rep) % 7 + 1) as f32).collect();
                    if let Some(&i) = wrong.first() {
                        return Err(format!(
                            "custom all-reduce {dtype:?} x {count} (repeat {rep}): {} of {count} elements wrong, first {i} is {}, expected {}",
                            wrong.len(),
                            got[i],
                            triangle * ((i + rep) % 7 + 1) as f32
                        ));
                    }
                    checks += 1;
                }
            }
        }
        // 24 calls queued on the stream without waiting in between
        for (k, b) in queued.iter().enumerate() {
            let vals: Vec<f32> = (0..3000).map(|i| value(rank, i + k)).collect();
            b.upload(&encode(&vals, DType::F32))?;
        }
        default_stream_sync()?;
        for b in &queued {
            car.all_reduce(b.addr(), 3000, DType::F32, h)?;
        }
        stream.sync()?;
        for (k, b) in queued.iter().enumerate() {
            let mut out = vec![0u8; 3000 * 4];
            b.download(&mut out)?;
            for (i, x) in decode(&out, DType::F32).iter().enumerate() {
                let expected = triangle * ((i + k) % 7 + 1) as f32;
                if *x != expected {
                    return Err(format!("custom all-reduce, queued call {k}: element {i} is {x}, expected {expected}"));
                }
            }
        }
        checks += 1;
        Ok(())
    })();
    // nobody frees memory or tears down the staging buffers while a peer may still be inside a kernel
    let _ = stream.sync();
    let _ = host.barrier();
    result.map(|()| checks)
}

#[cfg(feature = "cuda")]
fn gpu_part() -> Result<(), String> {
    let count = super::cuda_ffi::device_count()?;
    if count == 0 {
        println!("SKIP  gpu collectives: no GPU visible");
        return Ok(());
    }
    // the one-shot kernel, with N ranks sharing device 0 (no second GPU needed: this exercises the flag protocol and the kernel)
    for ranks in [1usize, 2, 4, 8] {
        let world = LocalComm::world(ranks);
        let results = run_ranks(&world, |_, host| check_custom(&host, 0))?;
        println!("PASS  custom one-shot all-reduce, {ranks} rank(s) on GPU 0: {} checks per rank, exact", results[0]);
    }
    let lib = match super::nccl::NcclLib::open() {
        Ok(lib) => lib,
        Err(e) => {
            println!("SKIP  {}: {e}", super::nccl::LIB_LABEL);
            return Ok(());
        }
    };
    println!("{} {} on {count} GPU(s)", super::nccl::LIB_LABEL, lib.version_text());
    // one rank: the real library, real device buffers, the real stream ordering (all collectives are identities on one rank)
    {
        let world = LocalComm::world(1);
        let results = run_ranks(&world, |rank, host| {
            let comm: Arc<dyn Communicator> = super::nccl::NcclComm::connect(&host, rank)?;
            Ok((check_collectives(&comm)?, check_device(&comm, rank)?))
        })?;
        println!("PASS  {} with 1 rank: host slices ({} checks) and device buffers ({} checks)", super::nccl::LIB_LABEL, results[0].0, results[0].1);
    }
    if count < 2 {
        println!("SKIP  {} multi-GPU: {count} GPU visible; the multi-rank checks need at least 2", super::nccl::LIB_LABEL);
        return Ok(());
    }
    match super::topology::report() {
        Ok(text) => print!("{text}"),
        Err(e) => println!("topology: {e}"),
    }
    let world = LocalComm::world(count);
    let results = run_ranks(&world, |rank, host| {
        let comm: Arc<dyn Communicator> = super::nccl::NcclComm::connect(&host, rank)?;
        let host_checks = check_collectives(&comm)?;
        let device_checks = check_device(&comm, rank)?;
        Ok((host_checks, device_checks))
    })?;
    println!("PASS  {} host slices ({} checks per rank) and device buffers ({} checks per rank) on {count} GPUs", super::nccl::LIB_LABEL, results[0].0, results[0].1);
    let world = LocalComm::world(count);
    let results = run_ranks(&world, |rank, host| check_custom(&host, rank));
    match results {
        Ok(r) => println!("PASS  custom one-shot all-reduce across {count} GPUs (peer access): {} checks per rank, exact", r[0]),
        Err(e) => println!("SKIP  custom one-shot all-reduce across {count} GPUs: {e}"),
    }
    Ok(())
}

pub fn run(args: &[String]) -> Result<(), String> {
    let mut ranks = 4usize;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--ranks" => {
                i += 1;
                ranks = args.get(i).and_then(|v| v.parse().ok()).ok_or("--ranks needs a number")?;
            }
            "--help" | "-h" => {
                println!("{HELP}");
                return Ok(());
            }
            other => return Err(format!("unknown option {other}\n{HELP}")),
        }
        i += 1;
    }
    for n in [2, ranks.max(2), 8] {
        let world = LocalComm::world(n);
        let counts = run_ranks(&world, |_, comm| check_collectives(&comm))?;
        println!("PASS  local ({n} ranks): {} collective checks per rank, exact", counts[0]);
    }
    #[cfg(feature = "cuda")]
    gpu_part()?;
    #[cfg(not(feature = "cuda"))]
    println!("SKIP  nccl / rccl: this build has no GPU support");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_host_communicator_passes_the_exact_checks() {
        for n in [1usize, 2, 3, 8] {
            let world = LocalComm::world(n);
            let counts = run_ranks(&world, |_, comm| check_collectives(&comm)).unwrap();
            assert!(counts.iter().all(|c| *c >= 5));
        }
    }
}
