//! Coalescing identical requests that are in flight at the same time: the first one computes, the others wait for its
//! answer instead of sending the same prompt through the model again.
//!
//! This is not a cache: nothing is stored beyond the moment the answer exists. It catches the burst — ten people pressing
//! send on the same prompt, a retry storm, a dashboard that polls the same question — and it cannot serve stale data.
//!
//! Only requests with a deterministic fingerprint take part (the same one the output cache uses), so two requests are
//! combined only when their answers would be identical.
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;

/// Capacity 1 is enough: followers only ever need the one answer (or the one "nothing came").
const CHANNEL: usize = 1;

pub struct Coalescer {
    inflight: Mutex<HashMap<[u8; 32], broadcast::Sender<Option<serde_json::Value>>>>,
    leaders: AtomicU64,
    followers: AtomicU64,
}

/// What this request is allowed to do.
pub enum Role {
    /// Nobody else is running this request: this one computes and publishes.
    Leader(Leader),
    /// Somebody else is already computing it: wait for that answer.
    Follower(broadcast::Receiver<Option<serde_json::Value>>),
}

/// The claim of a leader. Publishing the answer wakes the followers; dropping it without publishing tells them that
/// nothing came, which is better than leaving them waiting forever.
pub struct Leader {
    coalescer: Arc<Coalescer>,
    key: [u8; 32],
    published: bool,
}

impl Coalescer {
    pub fn new() -> Arc<Coalescer> {
        Arc::new(Coalescer { inflight: Mutex::new(HashMap::new()), leaders: AtomicU64::new(0), followers: AtomicU64::new(0) })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<[u8; 32], broadcast::Sender<Option<serde_json::Value>>>> {
        self.inflight.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Lead this request, or follow the one that already runs.
    pub fn join(self: &Arc<Self>, key: [u8; 32]) -> Role {
        let mut inflight = self.lock();
        if let Some(sender) = inflight.get(&key) {
            let receiver = sender.subscribe();
            self.followers.fetch_add(1, Ordering::Relaxed);
            return Role::Follower(receiver);
        }
        let (sender, _) = broadcast::channel(CHANNEL);
        inflight.insert(key, sender);
        self.leaders.fetch_add(1, Ordering::Relaxed);
        Role::Leader(Leader { coalescer: Arc::clone(self), key, published: false })
    }

    /// How many requests computed an answer others waited for, and how many waited.
    pub fn counters(&self) -> (u64, u64) {
        (self.leaders.load(Ordering::Relaxed), self.followers.load(Ordering::Relaxed))
    }

    pub fn stats(&self) -> serde_json::Value {
        serde_json::json!({
            "inflight": self.lock().len(),
            "leaders": self.leaders.load(Ordering::Relaxed),
            "followers": self.followers.load(Ordering::Relaxed),
        })
    }
}

impl Leader {
    /// Hands the answer to everyone who joined, and frees the key.
    pub fn publish(mut self, answer: serde_json::Value) {
        if let Some(sender) = self.coalescer.lock().remove(&self.key) {
            // nobody listening is fine: the send just fails
            let _ = sender.send(Some(answer));
        }
        self.published = true;
    }
}

impl Drop for Leader {
    fn drop(&mut self) {
        if self.published {
            return;
        }
        // the leader failed or went away: tell the followers instead of letting them wait
        if let Some(sender) = self.coalescer.lock().remove(&self.key) {
            let _ = sender.send(None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn key(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    #[tokio::test]
    async fn the_first_request_leads_and_the_second_follows() {
        let coalescer = Coalescer::new();
        let leader = match coalescer.join(key(1)) {
            Role::Leader(leader) => leader,
            Role::Follower(_) => panic!("the first one has to lead"),
        };
        let mut follower = match coalescer.join(key(1)) {
            Role::Follower(receiver) => receiver,
            Role::Leader(_) => panic!("the second one has to follow"),
        };
        leader.publish(json!({"answer": 42}));
        let received = follower.recv().await.expect("an answer").expect("not empty");
        assert_eq!(received["answer"], 42);
        // the key is free again
        assert!(matches!(coalescer.join(key(1)), Role::Leader(_)));
    }

    #[tokio::test]
    async fn different_requests_do_not_block_each_other() {
        let coalescer = Coalescer::new();
        let _first = match coalescer.join(key(1)) {
            Role::Leader(leader) => leader,
            Role::Follower(_) => panic!(),
        };
        assert!(matches!(coalescer.join(key(2)), Role::Leader(_)), "another key leads on its own");
    }

    #[tokio::test]
    async fn a_leader_that_fails_wakes_the_followers() {
        let coalescer = Coalescer::new();
        let leader = match coalescer.join(key(3)) {
            Role::Leader(leader) => leader,
            Role::Follower(_) => panic!(),
        };
        let mut follower = match coalescer.join(key(3)) {
            Role::Follower(receiver) => receiver,
            Role::Leader(_) => panic!(),
        };
        drop(leader); // for example a panic or an early error
        assert!(follower.recv().await.expect("a message").is_none(), "the follower learns that nothing came");
    }

    #[tokio::test]
    async fn many_followers_all_get_the_answer() {
        let coalescer = Coalescer::new();
        let leader = match coalescer.join(key(4)) {
            Role::Leader(leader) => leader,
            Role::Follower(_) => panic!(),
        };
        let mut followers: Vec<_> = (0..5)
            .map(|_| match coalescer.join(key(4)) {
                Role::Follower(receiver) => receiver,
                Role::Leader(_) => panic!(),
            })
            .collect();
        leader.publish(json!({"answer": "einmal gerechnet"}));
        for follower in followers.iter_mut() {
            let value = follower.recv().await.expect("an answer").expect("not empty");
            assert_eq!(value["answer"], "einmal gerechnet");
        }
        assert_eq!(coalescer.stats()["followers"], 5);
    }

    #[test]
    fn the_stats_count_leaders_and_followers() {
        let coalescer = Coalescer::new();
        let _leader = match coalescer.join(key(5)) {
            Role::Leader(leader) => leader,
            Role::Follower(_) => panic!(),
        };
        let _follower = coalescer.join(key(5));
        let stats = coalescer.stats();
        assert_eq!(stats["leaders"], 1);
        assert_eq!(stats["followers"], 1);
        assert_eq!(stats["inflight"], 1);
    }
}
