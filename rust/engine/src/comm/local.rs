//! `LocalComm`: all ranks are threads of one process and exchange data through host memory. It needs no GPU, serves the CPU
//! simulation of tensor / pipeline / expert parallelism, and carries the control traffic of the device back ends.
use super::fabric::{Fabric, LocalFabric};
use super::mesh::MeshComm;
use super::Communicator;
use std::sync::Arc;
use std::time::Duration;

/// The endpoints of an in-process world and the means to abort it.
pub struct LocalWorld {
    pub comms: Vec<Arc<dyn Communicator>>,
    fabric: Arc<dyn Fabric>,
}

impl LocalWorld {
    /// Wake every rank blocked in a receive with an error.
    pub fn abort(&self) {
        self.fabric.abort();
    }
}

pub struct LocalComm;

impl LocalComm {
    /// `size` communicators, one per rank, to be moved into the rank's thread.
    pub fn world(size: usize) -> LocalWorld {
        Self::from_fabrics(LocalFabric::create(size))
    }

    pub fn world_with_timeout(size: usize, timeout: Duration) -> LocalWorld {
        Self::from_fabrics(LocalFabric::create_with_timeout(size, timeout))
    }

    fn from_fabrics(fabrics: Vec<Arc<dyn Fabric>>) -> LocalWorld {
        let fabric = fabrics[0].clone();
        let comms = fabrics.into_iter().map(|f| MeshComm::world(f, "local") as Arc<dyn Communicator>).collect();
        LocalWorld { comms, fabric }
    }
}

/// Run `body(rank, comm)` for every rank on its own thread and collect the results in rank order. The first failure (an `Err` or a
/// panic) aborts the world so that the other ranks do not wait for a message that never comes.
pub fn run_ranks<T: Send>(
    world: &LocalWorld,
    body: impl Fn(usize, Arc<dyn Communicator>) -> Result<T, String> + Sync,
) -> Result<Vec<T>, String> {
    std::thread::scope(|scope| {
        let handles: Vec<_> = world
            .comms
            .iter()
            .enumerate()
            .map(|(rank, comm)| {
                let comm = comm.clone();
                let body = &body;
                scope.spawn(move || {
                    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| body(rank, comm)));
                    let result = match outcome {
                        Ok(result) => result,
                        Err(payload) => {
                            let text = payload
                                .downcast_ref::<&str>()
                                .map(|s| (*s).to_owned())
                                .or_else(|| payload.downcast_ref::<String>().cloned())
                                .unwrap_or_else(|| "panic".into());
                            Err(format!("rank {rank} panicked: {text}"))
                        }
                    };
                    if result.is_err() {
                        // wake the ranks that wait for this one at once
                        world.abort();
                    }
                    result
                })
            })
            .collect();
        let mut results = Vec::new();
        let mut first_error: Option<String> = None;
        for (rank, handle) in handles.into_iter().enumerate() {
            match handle.join() {
                Ok(Ok(value)) => results.push(value),
                Ok(Err(error)) => {
                    world.abort();
                    // an abort wakes the other ranks with a derived error: keep the root cause
                    if first_error.as_ref().is_none_or(|e| e.contains("aborted")) {
                        first_error = Some(format!("rank {rank}: {error}"));
                    }
                }
                Err(_) => {
                    world.abort();
                    first_error.get_or_insert_with(|| format!("rank {rank} panicked"));
                }
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(results),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(rank: usize, len: usize) -> Vec<f32> {
        (0..len).map(|i| ((rank * 31 + i * 7) % 13) as f32 * 0.25 - 1.0).collect()
    }

    #[test]
    fn all_reduce_sums_in_rank_order_and_every_rank_gets_the_same_bits() {
        for size in [1usize, 2, 3, 4, 8] {
            let world = LocalComm::world(size);
            let results = run_ranks(&world, |rank, comm| {
                let mut v = data(rank, 37);
                comm.all_reduce(&mut v)?;
                Ok(v)
            })
            .unwrap();
            let mut expected = data(0, 37);
            for r in 1..size {
                for (e, d) in expected.iter_mut().zip(data(r, 37)) {
                    *e += d;
                }
            }
            for v in &results {
                assert_eq!(v, &expected, "size {size}");
            }
        }
    }

    #[test]
    fn gather_scatter_broadcast_and_barrier() {
        let size = 4;
        let world = LocalComm::world(size);
        run_ranks(&world, |rank, comm| {
            let mine = data(rank, 5);
            let mut gathered = vec![0.0; 5 * size];
            comm.all_gather(&mine, &mut gathered)?;
            for r in 0..size {
                assert_eq!(&gathered[r * 5..(r + 1) * 5], &data(r, 5)[..]);
            }
            // reduce_scatter: block r of everybody summed
            let send = data(rank, 3 * size);
            let mut out = vec![0.0; 3];
            comm.reduce_scatter(&send, &mut out)?;
            let mut expected = vec![0.0f32; 3];
            for src in 0..size {
                for (e, d) in expected.iter_mut().zip(&data(src, 3 * size)[rank * 3..rank * 3 + 3]) {
                    *e += d;
                }
            }
            assert_eq!(out, expected);
            // broadcast from rank 2
            let mut b = if rank == 2 { vec![4.0, 5.0, 6.0] } else { vec![0.0; 3] };
            comm.broadcast(&mut b, 2)?;
            assert_eq!(b, vec![4.0, 5.0, 6.0]);
            let mut bytes = if rank == 1 { b"step".to_vec() } else { Vec::new() };
            comm.broadcast_bytes(&mut bytes, 1)?;
            assert_eq!(bytes, b"step");
            comm.barrier()?;
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn all_to_all_with_uneven_splits() {
        let size = 3;
        let world = LocalComm::world(size);
        run_ranks(&world, |rank, comm| {
            // rank r sends (r + dst) elements valued 100 * r + dst to dst
            let send_counts: Vec<usize> = (0..size).map(|d| rank + d).collect();
            let mut send = Vec::new();
            for (d, &c) in send_counts.iter().enumerate() {
                send.extend(std::iter::repeat_n((100 * rank + d) as f32, c));
            }
            let recv_counts = comm.exchange_counts(&send_counts)?;
            assert_eq!(recv_counts, (0..size).map(|s| s + rank).collect::<Vec<_>>());
            let mut recv = vec![0.0; recv_counts.iter().sum()];
            comm.all_to_all(&send, &send_counts, &mut recv, &recv_counts)?;
            let mut offset = 0;
            for (s, &c) in recv_counts.iter().enumerate() {
                assert!(recv[offset..offset + c].iter().all(|&v| v == (100 * s + rank) as f32));
                offset += c;
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn point_to_point_does_not_block_the_sender_and_split_forms_groups() {
        let size = 4;
        let world = LocalComm::world(size);
        run_ranks(&world, |rank, comm| {
            // ring: everybody sends first, then receives
            comm.send(&[rank as f32], (rank + 1) % size)?;
            let mut got = [0.0];
            comm.recv(&mut got, (rank + size - 1) % size)?;
            assert_eq!(got[0], ((rank + size - 1) % size) as f32);
            // two groups {0,2} and {1,3}, ordered by descending key
            let group = comm.split(rank % 2, size - rank)?;
            assert_eq!(group.size(), 2);
            assert_eq!(group.rank(), usize::from(rank < 2));
            let mut v = [rank as f32];
            group.all_reduce(&mut v)?;
            assert_eq!(v[0], if rank % 2 == 0 { 2.0 } else { 4.0 });
            // the parent keeps working, and a group of a group too
            let mut w = [1.0];
            comm.all_reduce(&mut w)?;
            assert_eq!(w[0], 4.0);
            let inner = group.split(0, group.rank())?;
            assert_eq!(inner.size(), 2);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn a_failing_rank_aborts_the_others() {
        let world = LocalComm::world_with_timeout(3, Duration::from_secs(30));
        let started = std::time::Instant::now();
        let result = run_ranks(&world, |rank, comm| {
            if rank == 1 {
                return Err("boom".to_owned());
            }
            comm.barrier()
        });
        assert!(result.unwrap_err().contains("boom"));
        assert!(started.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn mismatched_buffer_sizes_are_reported() {
        let world = LocalComm::world(2);
        let result = run_ranks(&world, |rank, comm| {
            let mut v = vec![1.0; if rank == 0 { 4 } else { 5 }];
            comm.all_reduce(&mut v)
        });
        assert!(result.is_err());
    }
}
