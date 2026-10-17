//! Byte transport between ranks: an in-process fabric (queues in host memory) and the inbox both fabrics share.
//! Messages between one pair of ranks on one channel arrive in the order they were sent.
use super::CommResult;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// A rank's view of the transport that connects all ranks of the world.
pub trait Fabric: Send + Sync {
    fn world_size(&self) -> usize;
    fn world_rank(&self) -> usize;
    /// Never blocks on the receiver.
    fn send(&self, dst: usize, chan: u32, data: Vec<u8>) -> CommResult<()>;
    fn recv(&self, src: usize, chan: u32) -> CommResult<Vec<u8>>;
    /// Wake every blocked receiver with an error (a worker failed; the others must not wait forever).
    fn abort(&self);
}

/// Receive timeout: `FLLM_COMM_TIMEOUT_SECS` (default 600).
pub fn default_timeout() -> Duration {
    let secs = std::env::var("FLLM_COMM_TIMEOUT_SECS").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(600);
    Duration::from_secs(secs.max(1))
}

#[derive(Default)]
struct InboxState {
    queues: HashMap<(usize, u32), VecDeque<Vec<u8>>>,
    closed: HashSet<usize>,
}

/// Messages addressed to one rank, by (sender, channel).
pub struct Inbox {
    state: Mutex<InboxState>,
    ready: Condvar,
    aborted: Arc<AtomicBool>,
    timeout: Duration,
}

impl Inbox {
    pub fn new(aborted: Arc<AtomicBool>, timeout: Duration) -> Self {
        Self { state: Mutex::new(InboxState::default()), ready: Condvar::new(), aborted, timeout }
    }

    pub fn push(&self, src: usize, chan: u32, data: Vec<u8>) {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        state.queues.entry((src, chan)).or_default().push_back(data);
        self.ready.notify_all();
    }

    /// The connection to `src` is gone: queued messages can still be read, then `pop` fails.
    pub fn close(&self, src: usize) {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        state.closed.insert(src);
        self.ready.notify_all();
    }

    pub fn wake_all(&self) {
        let _guard = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        self.ready.notify_all();
    }

    pub fn pop(&self, src: usize, chan: u32) -> CommResult<Vec<u8>> {
        let deadline = Instant::now() + self.timeout;
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            if let Some(data) = state.queues.get_mut(&(src, chan)).and_then(VecDeque::pop_front) {
                return Ok(data);
            }
            if self.aborted.load(Ordering::Relaxed) {
                return Err("communication aborted: another rank failed".into());
            }
            if state.closed.contains(&src) {
                return Err(format!("rank {src} disconnected"));
            }
            let now = Instant::now();
            if now >= deadline {
                return Err(format!(
                    "timed out after {} s waiting for a message from rank {src} (a peer crashed or the ranks run different steps; FLLM_COMM_TIMEOUT_SECS changes the limit)",
                    self.timeout.as_secs()
                ));
            }
            let wait = (deadline - now).min(Duration::from_millis(200));
            state = self.ready.wait_timeout(state, wait).unwrap_or_else(std::sync::PoisonError::into_inner).0;
        }
    }
}

struct LocalWorld {
    inboxes: Vec<Inbox>,
    aborted: Arc<AtomicBool>,
}

/// All ranks in one process: sending pushes into the receiver's inbox.
pub struct LocalFabric {
    rank: usize,
    world: Arc<LocalWorld>,
}

impl LocalFabric {
    /// One fabric endpoint per rank.
    pub fn create(size: usize) -> Vec<Arc<dyn Fabric>> {
        Self::create_with_timeout(size, default_timeout())
    }

    pub fn create_with_timeout(size: usize, timeout: Duration) -> Vec<Arc<dyn Fabric>> {
        let aborted = Arc::new(AtomicBool::new(false));
        let world = Arc::new(LocalWorld { inboxes: (0..size).map(|_| Inbox::new(aborted.clone(), timeout)).collect(), aborted });
        (0..size).map(|rank| Arc::new(LocalFabric { rank, world: world.clone() }) as Arc<dyn Fabric>).collect()
    }
}

impl Fabric for LocalFabric {
    fn world_size(&self) -> usize {
        self.world.inboxes.len()
    }
    fn world_rank(&self) -> usize {
        self.rank
    }
    fn send(&self, dst: usize, chan: u32, data: Vec<u8>) -> CommResult<()> {
        let inbox = self.world.inboxes.get(dst).ok_or_else(|| format!("rank {dst} is outside the world of {}", self.world.inboxes.len()))?;
        inbox.push(self.rank, chan, data);
        Ok(())
    }
    fn recv(&self, src: usize, chan: u32) -> CommResult<Vec<u8>> {
        self.world.inboxes[self.rank].pop(src, chan)
    }
    fn abort(&self) {
        self.world.aborted.store(true, Ordering::Relaxed);
        for inbox in &self.world.inboxes {
            inbox.wake_all();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_keep_their_order_per_pair_and_channel() {
        let ranks = LocalFabric::create(2);
        ranks[0].send(1, 0, vec![1]).unwrap();
        ranks[0].send(1, 7, vec![9]).unwrap();
        ranks[0].send(1, 0, vec![2]).unwrap();
        assert_eq!(ranks[1].recv(0, 7).unwrap(), vec![9]);
        assert_eq!(ranks[1].recv(0, 0).unwrap(), vec![1]);
        assert_eq!(ranks[1].recv(0, 0).unwrap(), vec![2]);
    }

    #[test]
    fn a_missing_message_times_out_and_abort_wakes_receivers() {
        let ranks = LocalFabric::create_with_timeout(2, Duration::from_millis(300));
        let error = ranks[1].recv(0, 0).unwrap_err();
        assert!(error.contains("timed out"), "{error}");
        let waiting = ranks[1].clone();
        let handle = std::thread::spawn(move || waiting.recv(0, 1));
        std::thread::sleep(Duration::from_millis(50));
        ranks[0].abort();
        let result = handle.join().unwrap();
        assert!(result.unwrap_err().contains("aborted"));
    }
}
