//! `fllm-rs selftest-comm`: every collective on data whose result is known exactly (small integers in f32, so sums are exact).
//! The host-memory implementation always runs; with a CUDA build, two or more GPUs and libnccl the same checks run through NCCL on host
//! slices and on device buffers. With fewer than two GPUs the GPU part is skipped with a message.
use super::local::{run_ranks, LocalComm};
use super::{CommResult, Communicator};
use std::sync::Arc;

pub const HELP: &str = "usage: fllm-rs selftest-comm [--ranks N]\n\
Checks all-reduce, all-gather, reduce-scatter, broadcast, all-to-all (uneven splits) and send/recv with exact expected results:\n\
first on the host-memory communicator (N threads, default 4), then, in a CUDA build with at least two GPUs and libnccl installed,\n\
through NCCL on host slices and on device buffers across all visible GPUs.";

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

#[cfg(all(feature = "cuda", not(feature = "hip")))]
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

#[cfg(all(feature = "cuda", not(feature = "hip")))]
fn gpu_part() -> Result<(), String> {
    let count = super::cuda_ffi::device_count()?;
    if count < 2 {
        println!("SKIP  nccl: {count} GPU(s) visible; the NCCL checks need at least 2");
        return Ok(());
    }
    let lib = match super::nccl::NcclLib::open() {
        Ok(lib) => lib,
        Err(e) => {
            println!("SKIP  nccl: {e}");
            return Ok(());
        }
    };
    println!("nccl {} on {count} GPUs", lib.version_text());
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
    println!("PASS  nccl host slices ({} checks per rank) and device buffers ({} checks per rank) on {count} GPUs", results[0].0, results[0].1);
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
    #[cfg(all(feature = "cuda", not(feature = "hip")))]
    gpu_part()?;
    #[cfg(not(all(feature = "cuda", not(feature = "hip"))))]
    println!("SKIP  nccl: this build has no CUDA support");
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
