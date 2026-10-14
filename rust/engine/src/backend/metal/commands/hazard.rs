//! Dependency tracking between the dispatches of one concurrent compute encoder.
//!
//! In a `MTLDispatchTypeConcurrent` encoder the GPU may overlap consecutive dispatches, and Metal does not order
//! them even for hazard-tracked buffers. The recorder therefore declares which byte ranges of which buffer each
//! dispatch reads and writes, and this tracker says when a barrier has to be inserted before the dispatch: when it
//! reads something an earlier dispatch of the same window wrote (read after write), or writes something an
//! earlier one read or wrote (write after read, write after write). Disjoint ranges of the same buffer do not
//! conflict. The window of remembered accesses starts anew after every barrier.

/// How a dispatch uses a buffer range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    Read,
    Write,
    /// Read and write (in-place update, accumulation).
    ReadWrite,
}

impl Access {
    fn writes(self) -> bool {
        !matches!(self, Access::Read)
    }
}

/// Buffer id that stands for "any buffer" (accesses through address tables whose targets are not tracked one by one).
const ALL_BUFFERS: u64 = u64::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Entry {
    buffer: u64,
    start: usize,
    end: usize,
    write: bool,
}

impl Entry {
    fn overlaps(&self, other: &Entry) -> bool {
        (self.buffer == other.buffer || self.buffer == ALL_BUFFERS || other.buffer == ALL_BUFFERS)
            && self.start < other.end
            && other.start < self.end
    }

    fn conflicts(&self, other: &Entry) -> bool {
        (self.write || other.write) && self.overlaps(other)
    }
}

/// Accesses of the current barrier window plus those of the dispatch being declared.
#[derive(Debug)]
pub struct HazardTracker {
    window: Vec<Entry>,
    pending: Vec<Entry>,
    max_entries: usize,
    barriers: u64,
}

impl HazardTracker {
    /// `max_entries`: most accesses remembered between two barriers; reaching it forces a barrier (keeps the check cheap).
    pub fn new(max_entries: usize) -> Self {
        Self {
            window: Vec::with_capacity(64),
            pending: Vec::with_capacity(16),
            max_entries: max_entries.max(8),
            barriers: 0,
        }
    }

    /// Declare that the dispatch under construction touches `[start, end)` of buffer `buffer`.
    pub fn declare(&mut self, buffer: u64, start: usize, end: usize, access: Access) {
        if start >= end {
            return;
        }
        let entry = Entry {
            buffer,
            start,
            end,
            write: access.writes(),
        };
        // same range declared twice by one dispatch (a buffer bound at two indices): keep the stronger entry
        if let Some(existing) = self
            .pending
            .iter_mut()
            .find(|e| e.buffer == buffer && e.start == start && e.end == end)
        {
            existing.write |= entry.write;
            return;
        }
        self.pending.push(entry);
    }

    /// Declare an access to memory that is not tracked per buffer (everything reachable through an address table).
    pub fn declare_all(&mut self, access: Access) {
        self.declare(ALL_BUFFERS, 0, usize::MAX, access);
    }

    /// Close the dispatch under construction: `true` if a barrier has to be encoded before it. The accesses of the
    /// dispatch join the window (which was cleared first when a barrier is due).
    pub fn commit_dispatch(&mut self) -> bool {
        let mut barrier = self.window.len() + self.pending.len() > self.max_entries;
        if !barrier {
            barrier = self
                .pending
                .iter()
                .any(|p| self.window.iter().any(|w| p.conflicts(w)));
        }
        if barrier {
            self.window.clear();
            self.barriers += 1;
        }
        for entry in self.pending.drain(..) {
            // merge with an identical range already in the window
            if let Some(existing) = self
                .window
                .iter_mut()
                .find(|w| w.buffer == entry.buffer && w.start == entry.start && w.end == entry.end)
            {
                existing.write |= entry.write;
            } else {
                self.window.push(entry);
            }
        }
        barrier
    }

    /// Forget the window (a barrier was encoded explicitly, or a new encoder starts). Drops undeclared pending accesses too.
    pub fn reset(&mut self) {
        self.window.clear();
        self.pending.clear();
    }

    /// Drop the accesses declared for a dispatch that will not be encoded (the window stays).
    pub fn discard_pending(&mut self) {
        self.pending.clear();
    }

    /// Barriers requested so far.
    pub fn barriers(&self) -> u64 {
        self.barriers
    }

    /// Accesses currently remembered (for tests and statistics).
    pub fn window_len(&self) -> usize {
        self.window.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dispatch(tracker: &mut HazardTracker, accesses: &[(u64, usize, usize, Access)]) -> bool {
        for &(buffer, start, end, access) in accesses {
            tracker.declare(buffer, start, end, access);
        }
        tracker.commit_dispatch()
    }

    #[test]
    fn independent_reads_share_a_window() {
        let mut t = HazardTracker::new(64);
        assert!(!dispatch(
            &mut t,
            &[(1, 0, 100, Access::Read), (2, 0, 100, Access::Write)]
        ));
        assert!(!dispatch(
            &mut t,
            &[(1, 0, 100, Access::Read), (3, 0, 100, Access::Write)]
        ));
        assert_eq!(t.barriers(), 0);
    }

    #[test]
    fn read_after_write_needs_a_barrier() {
        let mut t = HazardTracker::new(64);
        assert!(!dispatch(
            &mut t,
            &[(1, 0, 100, Access::Read), (2, 0, 100, Access::Write)]
        ));
        assert!(dispatch(
            &mut t,
            &[(2, 0, 100, Access::Read), (3, 0, 100, Access::Write)]
        ));
        // after the barrier the window holds only the second dispatch
        assert!(!dispatch(
            &mut t,
            &[(1, 0, 100, Access::Read), (4, 0, 100, Access::Write)]
        ));
        assert_eq!(t.barriers(), 1);
    }

    #[test]
    fn write_after_read_and_write_after_write() {
        let mut t = HazardTracker::new(64);
        assert!(!dispatch(&mut t, &[(1, 0, 100, Access::Read)]));
        assert!(
            dispatch(&mut t, &[(1, 50, 150, Access::Write)]),
            "WAR on overlapping range"
        );
        assert!(dispatch(&mut t, &[(1, 60, 70, Access::Write)]), "WAW");
        assert!(
            !dispatch(&mut t, &[(1, 0, 10, Access::Write)]),
            "disjoint from the 60..70 write that opened the window"
        );
    }

    #[test]
    fn disjoint_ranges_of_one_buffer_do_not_conflict() {
        let mut t = HazardTracker::new(64);
        assert!(!dispatch(&mut t, &[(7, 0, 64, Access::Write)]));
        assert!(!dispatch(&mut t, &[(7, 64, 128, Access::Write)]));
        assert!(!dispatch(&mut t, &[(7, 128, 192, Access::Read)]));
        assert!(
            dispatch(&mut t, &[(7, 100, 130, Access::Read)]),
            "overlaps the write of 64..128"
        );
    }

    #[test]
    fn in_place_update_is_one_access() {
        let mut t = HazardTracker::new(64);
        // a buffer bound at two indices, read and written by the same dispatch: no self-conflict
        assert!(!dispatch(
            &mut t,
            &[(5, 0, 32, Access::Read), (5, 0, 32, Access::ReadWrite)]
        ));
        assert!(dispatch(&mut t, &[(5, 0, 32, Access::Read)]));
    }

    #[test]
    fn untracked_access_conflicts_with_everything_written() {
        let mut t = HazardTracker::new(64);
        assert!(!dispatch(&mut t, &[(1, 0, 8, Access::Read)]));
        t.declare_all(Access::Read);
        assert!(!t.commit_dispatch(), "read after read");
        assert!(
            dispatch(&mut t, &[(1, 0, 8, Access::Write)]),
            "write conflicts with the earlier wildcard read"
        );
        t.declare_all(Access::Write);
        assert!(t.commit_dispatch());
    }

    #[test]
    fn window_overflow_forces_a_barrier() {
        let mut t = HazardTracker::new(8);
        let mut barriers = 0;
        for i in 0..40u64 {
            if dispatch(&mut t, &[(i, 0, 8, Access::Read)]) {
                barriers += 1;
            }
        }
        assert!(barriers >= 4, "got {barriers}");
        assert!(t.window_len() <= 8);
    }

    #[test]
    fn reset_clears_the_window() {
        let mut t = HazardTracker::new(64);
        assert!(!dispatch(&mut t, &[(1, 0, 8, Access::Write)]));
        t.reset();
        assert!(!dispatch(&mut t, &[(1, 0, 8, Access::Read)]));
    }

    #[test]
    fn empty_ranges_are_ignored() {
        let mut t = HazardTracker::new(64);
        assert!(!dispatch(&mut t, &[(1, 10, 10, Access::Write)]));
        assert!(!dispatch(&mut t, &[(1, 0, 100, Access::Read)]));
    }
}
