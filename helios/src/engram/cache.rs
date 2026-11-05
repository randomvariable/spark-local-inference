/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Fixed-capacity, four-way row storage for immutable embedding tables.

use crate::Result;

const WAYS: usize = 4;
/// Marks a row read ahead of its first use. A pinned row was admitted without
/// the frequency test, because a row predicted for the next step is about to be
/// used however rarely it was seen before, and no insertion may evict it until
/// it is used or its pin goes stale. The bit lives in the recency stamp, which
/// can never reach it, so an entry stays two words.
const PINNED: u64 = 1 << 63;

#[derive(Clone, Copy)]
struct Entry {
    id: i64,
    accessed: u64,
}

impl Entry {
    fn stamp(self) -> u64 {
        self.accessed & !PINNED
    }

    fn pinned(self) -> bool {
        self.accessed & PINNED != 0
    }
}

pub(super) struct RowCache {
    entries: Vec<Entry>,
    payload: Vec<u8>,
    row_bytes: usize,
    clock: u64,
    frequency: Frequency,
}

/// Bounded approximate usage counts, including requests that miss the row cache.
struct Frequency {
    counters: Vec<u8>,
    samples: usize,
}

impl Frequency {
    fn indices(&self, id: i64) -> [usize; 4] {
        let hash = twox_hash::XxHash3_64::oneshot(&id.to_le_bytes());
        let width = self.counters.len() / 4;
        std::array::from_fn(|i| {
            i * width + (hash.rotate_right(i as u32 * 16) as usize & (width - 1))
        })
    }

    fn estimate(&self, id: i64) -> u8 {
        self.indices(id)
            .map(|i| self.counters[i])
            .into_iter()
            .min()
            .unwrap()
    }

    fn observe(&mut self, id: i64) {
        self.samples += 1;
        if self.samples >= self.counters.len() * 10 {
            for value in &mut self.counters {
                *value /= 2;
            }
            self.samples = 0;
        }
        for index in self.indices(id) {
            self.counters[index] = self.counters[index].saturating_add(1);
        }
    }
}

pub(super) struct Insertion {
    /// The row is resident after the call.
    pub admitted: bool,
    /// It took a previously empty slot.
    pub added: bool,
    /// It displaced a different row.
    pub evicted: bool,
}

impl Insertion {
    const REFUSED: Self = Self {
        admitted: false,
        added: false,
        evicted: false,
    };
}

impl RowCache {
    pub fn new(budget: usize, rows: usize, row_bytes: usize) -> Result<Self> {
        let history_bytes = (budget / 64).clamp(16, 1 << 20).next_power_of_two();
        let payload_budget = budget
            .checked_sub(history_bytes)
            .ok_or("embedding row cache budget cannot hold usage counters")?;
        let set_bytes = row_bytes
            .checked_add(size_of::<Entry>())
            .and_then(|n| n.checked_mul(WAYS))
            .ok_or("embedding row cache size overflow")?;
        let sets = (payload_budget / set_bytes).min(rows.div_ceil(WAYS));
        if sets == 0 {
            return Err("embedding row cache budget cannot hold one four-row set".into());
        }
        let capacity = sets * WAYS;
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(capacity)
            .map_err(|e| format!("embedding row cache index: {e}"))?;
        entries.resize(
            capacity,
            Entry {
                id: -1,
                accessed: 0,
            },
        );
        let mut payload = Vec::new();
        payload
            .try_reserve_exact(capacity * row_bytes)
            .map_err(|e| format!("embedding row cache payload: {e}"))?;
        payload.resize(capacity * row_bytes, 0);
        // Commit every host page before the integrated GPU's KV budget reads
        // MemAvailable. A lazily zeroed allocation would leave that budget too large.
        for page in payload.chunks_mut(4096) {
            unsafe { page.as_mut_ptr().write_volatile(0) };
        }
        let mut counters = Vec::new();
        counters
            .try_reserve_exact(history_bytes)
            .map_err(|e| format!("embedding row cache usage counters: {e}"))?;
        counters.resize(history_bytes, 0);
        for page in counters.chunks_mut(4096) {
            unsafe { page.as_mut_ptr().write_volatile(0) };
        }
        Ok(Self {
            entries,
            payload,
            row_bytes,
            clock: 0,
            frequency: Frequency {
                counters,
                samples: 0,
            },
        })
    }

    pub fn capacity(&self) -> usize {
        self.entries.len()
    }

    pub fn bytes(&self) -> usize {
        self.entries.capacity() * size_of::<Entry>()
            + self.payload.capacity()
            + self.frequency.counters.capacity()
    }

    fn set(&self, id: i64) -> std::ops::Range<usize> {
        let hash = twox_hash::XxHash3_64::oneshot(&id.to_le_bytes());
        let first = (hash as usize % (self.entries.len() / WAYS)) * WAYS;
        first..first + WAYS
    }

    /// Serve a row and count the use. A read-ahead row stops being protected
    /// once it is used and competes on frequency and recency from then on.
    pub fn get(&mut self, id: i64) -> Option<&[u8]> {
        self.frequency.observe(id);
        let slot = self.set(id).find(|&slot| self.entries[slot].id == id)?;
        self.clock = self.clock.saturating_add(1);
        self.entries[slot].accessed = self.clock;
        Some(&self.payload[slot * self.row_bytes..(slot + 1) * self.row_bytes])
    }

    /// Whether `id` is resident, pinning it for an upcoming use if so. A
    /// prediction is not a use: the frequency estimate is left to the gather
    /// that consumes the row, so warming never counts a row twice.
    pub fn pin(&mut self, id: i64) -> bool {
        let Some(slot) = self.set(id).find(|&slot| self.entries[slot].id == id) else {
            return false;
        };
        self.clock = self.clock.saturating_add(1);
        self.entries[slot].accessed = self.clock | PINNED;
        true
    }

    pub fn insert(&mut self, id: i64, weights: &[u8], scales: &[u8]) -> Insertion {
        debug_assert!(id >= 0);
        debug_assert_eq!(weights.len() + scales.len(), self.row_bytes);
        self.clock = self.clock.saturating_add(1);
        match self.admit(id, self.clock) {
            Some(slot) => self.place(slot, id, self.clock, weights, scales),
            None => Insertion::REFUSED,
        }
    }

    /// Insert a row read ahead of its first use. It is admitted regardless of
    /// how often it was seen, and refused only when every way of its set already
    /// holds a protected read-ahead row.
    pub fn insert_ahead(&mut self, id: i64, weights: &[u8], scales: &[u8]) -> Insertion {
        debug_assert!(id >= 0);
        debug_assert_eq!(weights.len() + scales.len(), self.row_bytes);
        self.clock = self.clock.saturating_add(1);
        match self.victim(id) {
            Some(slot) => self.place(slot, id, self.clock | PINNED, weights, scales),
            None => Insertion::REFUSED,
        }
    }

    fn place(
        &mut self,
        slot: usize,
        id: i64,
        accessed: u64,
        weights: &[u8],
        scales: &[u8],
    ) -> Insertion {
        let previous = self.entries[slot].id;
        self.entries[slot] = Entry { id, accessed };
        self.fill(id, weights, scales);
        Insertion {
            admitted: true,
            added: previous == -1,
            evicted: previous >= 0 && previous != id,
        }
    }

    /// A read-ahead row stays protected for one turnover of the cache's
    /// operations. A row predicted but never requested, such as one for a
    /// cancelled request, then ages back into an ordinary entry.
    fn protected(&self, entry: Entry) -> bool {
        entry.pinned() && self.clock.saturating_sub(entry.stamp()) < self.entries.len() as u64
    }

    /// The slot `id` would occupy: its own, an empty way, or the way with the
    /// lowest usage and then the oldest use among those not protected.
    fn victim(&self, id: i64) -> Option<usize> {
        let set = self.set(id);
        set.clone()
            .find(|&slot| self.entries[slot].id == id)
            .or_else(|| set.clone().find(|&slot| self.entries[slot].id == -1))
            .or_else(|| {
                set.filter(|&slot| !self.protected(self.entries[slot]))
                    .min_by_key(|&slot| {
                        let entry = self.entries[slot];
                        (self.frequency.estimate(entry.id), entry.stamp())
                    })
            })
    }

    fn admit(&self, id: i64, accessed: u64) -> Option<usize> {
        let slot = self.victim(id)?;
        let previous = self.entries[slot];
        (previous.id < 0
            || previous.id == id
            || (self.frequency.estimate(id), accessed)
                >= (self.frequency.estimate(previous.id), previous.stamp()))
            .then_some(slot)
    }

    fn fill(&mut self, id: i64, weights: &[u8], scales: &[u8]) {
        let slot = self
            .set(id)
            .find(|&slot| self.entries[slot].id == id)
            .unwrap();
        let row = &mut self.payload[slot * self.row_bytes..(slot + 1) * self.row_bytes];
        row[..weights.len()].copy_from_slice(weights);
        row[weights.len()..].copy_from_slice(scales);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collisions_evict_the_least_recent_row_and_keep_both_planes() {
        let mut cache = RowCache::new(16 + 4 * (size_of::<Entry>() + 3), 100, 3).unwrap();
        for id in 0..4 {
            let result = cache.insert(id, &[id as u8, 17], &[42]);
            assert!(result.added && !result.evicted);
        }
        // Admission compares usage before recency. Give the incoming row the
        // same frequency as an untouched resident to exercise the recency tie.
        assert!(cache.get(4).is_none());
        assert_eq!(cache.get(0), Some(&[0, 17, 42][..]));
        let result = cache.insert(4, &[4, 18], &[43]);
        assert!(!result.added && result.evicted);
        assert!(cache.get(1).is_none());
        assert_eq!(cache.get(0), Some(&[0, 17, 42][..]));
        assert_eq!(cache.get(4), Some(&[4, 18, 43][..]));
        let result = cache.insert(4, &[9, 19], &[44]);
        assert!(!result.added && !result.evicted);
        assert_eq!(cache.get(4), Some(&[9, 19, 44][..]));
    }

    #[test]
    fn payload_and_index_stay_within_the_budget() {
        for budget in [1024, 4096, 65536] {
            let cache = RowCache::new(budget, 100000, 90).unwrap();
            assert!(cache.bytes() <= budget);
            assert_eq!(cache.capacity() % WAYS, 0);
            assert!(cache.bytes() >= cache.capacity() * 106);
        }
        assert!(RowCache::new(423, 100, 90).is_err());
    }

    #[test]
    fn frequently_used_rows_survive_scans_and_repeated_misses_gain_admission() {
        // Four resident rows compete for one set, with enough history counters
        // to distinguish a scan from repeated use despite hash approximation.
        let mut cache = RowCache::new(16384, 1000, 2048).unwrap();
        let payload = [0; 2048];
        assert_eq!(cache.capacity(), 4);
        for id in 0..4 {
            cache.get(id);
            cache.insert(id, &payload, &[]);
        }
        for _ in 0..20 {
            for id in 0..4 {
                assert!(cache.get(id).is_some());
            }
        }
        for id in 4..12 {
            assert!(cache.get(id).is_none());
            assert!(!cache.insert(id, &payload, &[]).evicted);
        }
        for id in 0..4 {
            assert!(cache.get(id).is_some());
        }
        for _ in 0..100 {
            if cache.get(12).is_some() {
                break;
            }
            cache.insert(12, &payload, &[]);
        }
        assert!(cache.get(12).is_some());
    }

    #[test]
    fn read_ahead_rows_pass_frequent_rows_and_stay_until_used() {
        // One four-way set whose residents are all in steady use, as after a long
        // conversation: an ordinary miss for a never-seen row is refused.
        let mut cache = RowCache::new(16384, 1000, 2048).unwrap();
        let payload = [0; 2048];
        assert_eq!(cache.capacity(), 4);
        for id in 0..4 {
            cache.insert(id, &payload, &[]);
            for _ in 0..20 {
                cache.get(id);
            }
        }
        assert!(!cache.insert(10, &payload, &[]).admitted);
        // A row predicted for the next step is admitted anyway, and the next
        // read-ahead row must not displace it before that step uses it.
        let first = cache.insert_ahead(10, &payload, &[]);
        assert!(first.admitted && first.evicted);
        assert!(cache.insert_ahead(11, &payload, &[]).evicted);
        assert!(cache.pin(10), "the first read-ahead row was evicted before use");
        // Once used it is an ordinary, rarely used row, so it is the next victim
        // rather than the rows the conversation keeps using.
        assert!(cache.get(10).is_some());
        assert!(cache.get(11).is_some());
        assert!(cache.insert_ahead(12, &payload, &[]).evicted);
        assert!(cache.get(10).is_none() || cache.get(11).is_none());
        assert_eq!((0..4).filter(|&id| cache.pin(id)).count(), 2);
    }

    #[test]
    fn a_set_of_fresh_read_ahead_rows_refuses_more_until_the_pins_go_stale() {
        // Two sets, so the protection horizon (one turnover, eight operations)
        // outlasts one set's four read-ahead insertions.
        let mut cache = RowCache::new(17200, 1000, 2048).unwrap();
        let payload = [0; 2048];
        assert_eq!(cache.capacity(), 8);
        let first = cache.set(0).start;
        let (same, other): (Vec<i64>, Vec<i64>) =
            (0..64).partition(|&id| cache.set(id).start == first);
        for &id in &same[..4] {
            assert!(cache.insert_ahead(id, &payload, &[]).admitted);
        }
        // Every way holds a row a coming step needs: neither read-ahead nor an
        // ordinary miss may take one.
        assert!(!cache.insert_ahead(same[4], &payload, &[]).admitted);
        assert!(!cache.insert(same[4], &payload, &[]).admitted);
        // Rows predicted but never requested, as for a cancelled request, stop
        // being protected after a full turnover of the cache's operations.
        for &id in &other[..4] {
            cache.insert(id, &payload, &[]);
        }
        let late = cache.insert_ahead(same[4], &payload, &[]);
        assert!(late.admitted && late.evicted);
        assert!(!cache.pin(same[0]), "the oldest stale pin is the victim");
    }

    #[test]
    fn usage_counts_age_as_requests_shift() {
        let mut frequency = Frequency {
            counters: vec![0; 1024],
            samples: 0,
        };
        for _ in 0..100 {
            frequency.observe(7);
        }
        let before = frequency.estimate(7);
        for _ in 0..frequency.counters.len() * 20 {
            frequency.observe(99);
        }
        assert!(frequency.estimate(7) < before);
    }
}
