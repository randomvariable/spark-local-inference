/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host- or graph-produced n-gram requests and asynchronous compact-row delivery.

mod cache;
#[cfg(unix)]
mod prepared;
mod reader;
mod row_ids;
pub(crate) use row_ids::{compress_into, history_depth, query_row_ids};
use crate::{
    Error, Result,
    cuda::{Event, PinnedHost, Stream},
    execution::{ExecutionManifest, GraphKind, Upload},
};
#[cfg(unix)]
pub use prepared::{PlaneSource, PreparedTables, SourceSegment, TableSource};
pub(crate) use reader::Plane;
#[cfg(feature = "server")]
use std::fmt::Write;
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
};

/// Delivery, row-cache, and advisory-prefetch counters for one embedding table.
///
/// The single reader thread that owns a table's `Reader` writes every disk and
/// cache field, and the submitting thread writes only the prefetch queue gauge,
/// so each value needs no lock: counters are monotone and gauges hold the last
/// published value. The loader reads them once at startup through
/// `Stats::cache_bytes`.
#[derive(Default)]
pub(crate) struct TableCounters {
    /// Declared `EngramTable.layer`, used only as a metric label.
    layer: u32,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    resident_rows: AtomicU64,
    capacity_rows: AtomicU64,
    cache_bytes: AtomicU64,
    resident_scale_bytes: AtomicU64,
    disk_reads: AtomicU64,
    disk_bytes: AtomicU64,
    disk_read_calls: AtomicU64,
    disk_read_bytes: AtomicU64,
    disk_unique_blocks: AtomicU64,
    disk_coalesced_reads: AtomicU64,
    disk_submit_calls: AtomicU64,
    disk_plan_ns: AtomicU64,
    disk_submit_ns: AtomicU64,
    disk_wait_ns: AtomicU64,
    disk_complete_ns: AtomicU64,
    disk_staging_bytes: AtomicU64,
    disk_queue_depth: AtomicU64,
    disk_queue_depth_requested: AtomicU64,
    disk_alignment_bytes: AtomicU64,
    disk_uring_flags: AtomicU64,
    disk_fixed_files: AtomicU64,
    prefetch_rows: AtomicU64,
    prefetch_rows_loaded: AtomicU64,
    prefetch_rows_dropped: AtomicU64,
    prefetch_errors: AtomicU64,
    prefetch_queue_batches: AtomicU64,
    /// Steps whose published row ids included a row this table's prediction
    /// missed. Advisory: served rows always come from the published ids.
    prediction_mismatches: AtomicU64,
}

impl TableCounters {
    fn with_layer(layer: u32) -> Arc<Self> {
        Arc::new(Self {
            layer,
            ..Default::default()
        })
    }

    // Counter reads for the device delivery test, which observes the advisory
    // path from outside this module; the serving path publishes these through
    // `Stats::render` instead.
    #[cfg(test)]
    pub(crate) fn prefetch_rows(&self) -> u64 {
        self.prefetch_rows.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn prefetch_rows_loaded(&self) -> u64 {
        self.prefetch_rows_loaded.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn disk_read_calls(&self) -> u64 {
        self.disk_read_calls.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn prediction_mismatches(&self) -> u64 {
        self.prediction_mismatches.load(Ordering::Relaxed)
    }
}

#[derive(Default)]
pub(crate) struct Stats {
    gathers: AtomicU64,
    gather_ns: AtomicU64,
    /// Steps whose predicted rows were handed to the reader threads.
    prefetch_submissions: AtomicU64,
    /// Steps the prefetcher could not queue because the advisory channel was
    /// full. Skips because prefetch is disabled do no work to lose and are not
    /// counted here.
    prefetch_dropped_submissions: AtomicU64,
    /// One when the loader enabled advisory prefetch, so a zero row-cache hit
    /// rate distinguishes a disabled prefetcher from one that cannot keep up.
    prefetch_enabled: AtomicU64,
    tables: Vec<Arc<TableCounters>>,
}

impl Stats {
    /// Build the counters for one reader per table, labelled by its declared layer.
    pub(crate) fn with_tables(layers: impl IntoIterator<Item = u32>) -> Self {
        Self {
            tables: layers.into_iter().map(TableCounters::with_layer).collect(),
            ..Default::default()
        }
    }

    pub(crate) fn table(&self, index: usize) -> Option<Arc<TableCounters>> {
        self.tables.get(index).cloned()
    }

    /// Committed host bytes across every embedding row cache.
    pub(crate) fn cache_bytes(&self) -> u64 {
        self.tables
            .iter()
            .map(|table| table.cache_bytes.load(Ordering::Relaxed))
            .sum()
    }

    pub(crate) fn set_prefetch_enabled(&self, enabled: bool) {
        self.prefetch_enabled
            .store(u64::from(enabled), Ordering::Relaxed);
    }

    /// Count one step's advisory attempt, whether or not any batch was accepted.
    /// Steps skipped because prefetch is disabled never reach this call: they do
    /// no work to lose.
    pub(crate) fn record_submission(&self, handed: bool) {
        let counter = if handed {
            &self.prefetch_submissions
        } else {
            &self.prefetch_dropped_submissions
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(feature = "server")]
    fn line(text: &mut String, name: &str, kind: &str, help: &str, value: u64) {
        let _ = writeln!(
            text,
            "# HELP flare_engram_{name} {help}\n# TYPE flare_engram_{name} {kind}\nflare_engram_{name} {value}"
        );
    }

    #[cfg(feature = "server")]
    pub(crate) fn render(&self, text: &mut String) {
        for (name, kind, help, counter) in [
            (
                "gathers_total",
                "counter",
                "Embedding row gathers, excluding the wait for GPU-produced IDs.",
                &self.gathers,
            ),
            (
                "gather_nanoseconds_total",
                "counter",
                "Wall time inside embedding row gathers.",
                &self.gather_ns,
            ),
            (
                "prefetch_submissions_total",
                "counter",
                "Steps whose predicted next-step rows were handed to the reader threads.",
                &self.prefetch_submissions,
            ),
            (
                "prefetch_dropped_submissions_total",
                "counter",
                "Steps whose predicted rows were discarded because the advisory queue was full. Steps skipped because prefetch is disabled are not counted.",
                &self.prefetch_dropped_submissions,
            ),
            (
                "prefetch_enabled",
                "gauge",
                "One while advisory prefetch is enabled, so a low hit rate distinguishes a disabled prefetcher from one that cannot keep up.",
                &self.prefetch_enabled,
            ),
        ] {
            Self::line(text, name, kind, help, counter.load(Ordering::Relaxed));
        }
        for (name, kind, help, sample) in TABLE_SAMPLES {
            let total: u64 = self
                .tables
                .iter()
                .map(|table| sample(table).load(Ordering::Relaxed))
                .sum();
            Self::line(text, name, kind, help, total);
        }
        for (name, kind, help, sample) in TABLE_SAMPLES {
            let _ = writeln!(
                text,
                "# HELP flare_engram_table_{name} {help}\n# TYPE flare_engram_table_{name} {kind}"
            );
            for (index, table) in self.tables.iter().enumerate() {
                let _ = writeln!(
                    text,
                    "flare_engram_table_{name}{{table=\"{index}\",layer=\"{}\"}} {}",
                    table.layer,
                    sample(table).load(Ordering::Relaxed)
                );
            }
        }
    }
}

/// One sampled field of `TableCounters`, used to declare each family once and
/// then emit one labelled sample per embedding table.
#[cfg(feature = "server")]
type TableSample = (
    &'static str,
    &'static str,
    &'static str,
    fn(&TableCounters) -> &AtomicU64,
);

/// Each selector is a non-capturing closure, so it coerces to a function pointer
/// and the whole table stays `const`. `sample` names the selector slot so each
/// entry reads as the field it samples.
#[cfg(feature = "server")]
const fn sample(
    select: fn(&TableCounters) -> &AtomicU64,
) -> fn(&TableCounters) -> &AtomicU64 {
    select
}

#[cfg(feature = "server")]
const TABLE_SAMPLES: &[TableSample] = &[
    (
        "row_cache_hits_total",
        "counter",
        "Cached row hits; duplicate IDs within one gather count once.",
        sample(|table| &table.hits),
    ),
    (
        "row_cache_misses_total",
        "counter",
        "Cached row misses; duplicate IDs within one gather count once.",
        sample(|table| &table.misses),
    ),
    (
        "row_cache_evictions_total",
        "counter",
        "Resident rows replaced by other row IDs.",
        sample(|table| &table.evictions),
    ),
    (
        "row_cache_resident_rows",
        "gauge",
        "Rows held in the embedding row cache.",
        sample(|table| &table.resident_rows),
    ),
    (
        "row_cache_capacity_rows",
        "gauge",
        "Maximum rows held in the embedding row cache.",
        sample(|table| &table.capacity_rows),
    ),
    (
        "row_cache_bytes",
        "gauge",
        "Allocated embedding row payload, usage counters, index and bounded miss workspace bytes.",
        sample(|table| &table.cache_bytes),
    ),
    (
        "resident_scale_bytes",
        "gauge",
        "Resident scale-plane allocation bytes for disk-backed embedding tables, separate from the row cache.",
        sample(|table| &table.resident_scale_bytes),
    ),
    (
        "disk_reads_total",
        "counter",
        "Direct-I/O embedding read requests queued.",
        sample(|table| &table.disk_reads),
    ),
    (
        "disk_bytes_total",
        "counter",
        "Bytes requested by direct-I/O embedding reads.",
        sample(|table| &table.disk_bytes),
    ),
    (
        "disk_read_calls_total",
        "counter",
        "Direct-I/O embedding reads completed, counting each registered-buffer read once.",
        sample(|table| &table.disk_read_calls),
    ),
    (
        "disk_read_bytes_total",
        "counter",
        "Bytes returned by completed direct-I/O embedding reads.",
        sample(|table| &table.disk_read_bytes),
    ),
    (
        "disk_unique_blocks_total",
        "counter",
        "Distinct alignment-sized blocks covered by embedding row reads.",
        sample(|table| &table.disk_unique_blocks),
    ),
    (
        "disk_coalesced_reads_total",
        "counter",
        "Embedding reads that merged more than one block into one request.",
        sample(|table| &table.disk_coalesced_reads),
    ),
    (
        "disk_submit_calls_total",
        "counter",
        "io_uring enter calls that submitted or waited for embedding reads.",
        sample(|table| &table.disk_submit_calls),
    ),
    (
        "disk_staging_bytes",
        "gauge",
        "Registered direct-I/O staging bytes held by the embedding reader.",
        sample(|table| &table.disk_staging_bytes),
    ),
    (
        "disk_queue_depth",
        "gauge",
        "io_uring queue depth the reader actually registered after the locked-memory clamp and halving ladder.",
        sample(|table| &table.disk_queue_depth),
    ),
    (
        "disk_queue_depth_requested",
        "gauge",
        "io_uring queue depth the reader asked the kernel for before clamping.",
        sample(|table| &table.disk_queue_depth_requested),
    ),
    (
        "disk_alignment_bytes",
        "gauge",
        "Smallest per-file direct-I/O fragment floor in use, from statx when the filesystem reports one.",
        sample(|table| &table.disk_alignment_bytes),
    ),
    (
        "disk_uring_flags",
        "gauge",
        "IORING_SETUP_* bits the kernel accepted for the embedding reader's ring.",
        sample(|table| &table.disk_uring_flags),
    ),
    (
        "disk_fixed_files",
        "gauge",
        "One while the embedding reader submits registered file descriptors instead of raw ones.",
        sample(|table| &table.disk_fixed_files),
    ),
    (
        "disk_plan_nanoseconds_total",
        "counter",
        "Wall time building, sorting and coalescing disk row reads.",
        sample(|table| &table.disk_plan_ns),
    ),
    (
        "disk_submit_nanoseconds_total",
        "counter",
        "Cumulative worker wall time preparing and issuing disk read batches; Linux kernel submission is included in wait time.",
        sample(|table| &table.disk_submit_ns),
    ),
    (
        "disk_wait_nanoseconds_total",
        "counter",
        "Cumulative worker wall time in disk completion waits, including Linux io_uring submission.",
        sample(|table| &table.disk_wait_ns),
    ),
    (
        "disk_complete_nanoseconds_total",
        "counter",
        "Cumulative worker wall time processing disk completions and copying rows, excluding completion waits.",
        sample(|table| &table.disk_complete_ns),
    ),
    (
        "prefetch_rows_total",
        "counter",
        "Rows the advisory queue accepted for warming.",
        sample(|table| &table.prefetch_rows),
    ),
    (
        "prefetch_rows_loaded_total",
        "counter",
        "Rows the reader thread fetched and published into the row cache from the advisory queue.",
        sample(|table| &table.prefetch_rows_loaded),
    ),
    (
        "prefetch_rows_dropped_total",
        "counter",
        "Rows discarded because the advisory queue was full.",
        sample(|table| &table.prefetch_rows_dropped),
    ),
    (
        "prefetch_errors_total",
        "counter",
        "Disk or cache failures the advisory path swallowed instead of reporting; the real gather still reports them.",
        sample(|table| &table.prefetch_errors),
    ),
    (
        "prefetch_queue_batches",
        "gauge",
        "Advisory prefetch batches waiting in the reader thread's queue.",
        sample(|table| &table.prefetch_queue_batches),
    ),
    (
        "prediction_mismatches_total",
        "counter",
        "Steps whose published row ids included a row this table's prediction missed. Advisory: served rows always come from the published ids.",
        sample(|table| &table.prediction_mismatches),
    ),
];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Storage {
    #[default]
    Disk,
    Ram,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Table {
    /// Declared layer this table serves, used to label metrics and to select the
    /// per-layer row-addressing geometry.
    pub(crate) layer: u32,
    pub(crate) planes: [Plane; 2],
    pub(crate) heads: usize,
    /// Rows in the complete table, before rank partitioning.
    pub(crate) rows: u64,
    /// First global row of this rank's partition, zero when the table is shared.
    pub(crate) first_row: u64,
    /// Declared n-gram row addressing, validated at load.
    pub(crate) geometry: RowGeometry,
    pub(crate) weights: String,
    pub(crate) scales: String,
    pub(crate) hashes: String,
    pub(crate) event: String,
    pub(crate) request: Option<String>,
}

/// Declared n-gram row addressing for one table, from `EngramHash`.
///
/// The preparation-time resident catalog has no declared addressing: its tables
/// never predict rows, and every field is empty so a prediction yields no rows
/// rather than a guess.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct RowGeometry {
    pub(crate) method: RowMethod,
    /// One window length per lookup head, in output-column order.
    pub(crate) orders: Vec<u32>,
    /// Multipliers by lag: the current token first, then its predecessors.
    pub(crate) multipliers: Vec<u64>,
    pub(crate) primes: Vec<u64>,
    pub(crate) offsets: Vec<u64>,
}

/// How the package addresses rows, from `EngramHash.method`. Both forms mix the
/// window the same way; they differ in how a position outside the committed
/// history is filled in. The default selects no normalization, and an empty
/// [`RowGeometry`] therefore predicts nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum RowMethod {
    /// Normalize through the token map, then mix and take the modulus.
    #[default]
    XorModulo,
    /// Bound the history at the pad token, then mix and take the modulus.
    EosXorModulo,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Index {
    pub(crate) tables: Vec<Table>,
    /// Raw token id to normalized vocabulary id, empty when the method has none.
    pub(crate) token_map: Vec<u32>,
    /// Token that fills a window position with no committed history behind it, and
    /// the segment separator for the EOS-bounded form.
    pub(crate) pad_token_id: u32,
}

impl Table {
    fn staging_payload_bytes(&self, max_tokens: usize) -> Result<usize> {
        self.planes.iter().try_fold(0usize, |total, plane| total.checked_add(plane.row_bytes))
            .and_then(|bytes| bytes.checked_mul(self.heads))
            .and_then(|bytes| bytes.checked_mul(max_tokens))
            .ok_or_else(|| "embedding staging exceeds host range".into())
    }
}

pub(crate) fn staging_bytes(index: &Index, max_tokens: usize) -> Result<u64> {
    index.tables.iter().try_fold(0u64, |total, table| {
        let payload = table.staging_payload_bytes(max_tokens)? as u64;
        payload.checked_add(4).and_then(|bytes| bytes.checked_mul(STAGING_FRAMES as u64))
            .and_then(|bytes| total.checked_add(bytes))
            .ok_or_else(|| "embedding staging exceeds the memory budget".into())
    })
}

/// Bytes the per-table reader threads can reach in their bounded planning
/// tables, sized for the smallest fragment floor `statx` may report. The loader
/// reserves these alongside the staging frames: they belong to disk delivery,
/// not to the row cache, so configuring a cache budget never shrinks them.
pub(crate) fn planning_bytes(index: &Index, max_tokens: usize) -> u64 {
    index
        .tables
        .iter()
        .map(|table| {
            let rows = max_tokens.saturating_mul(table.heads);
            u64::try_from(reader::planning_bytes(&table.planes, rows)).unwrap_or(u64::MAX)
        })
        .sum()
}

struct Staging {
    data: PinnedHost,
    ready: PinnedHost,
    error: Mutex<Option<Error>>,
}
impl Staging {
    fn signal(&self, generation: u32) {
        unsafe { &*(self.ready.address() as *const AtomicU32) }
            .store(generation, Ordering::Release);
    }
}
enum Requests {
    Host(Vec<i64>),
    Device {
        address: usize,
        flag: usize,
        sequence: u64,
        rows: usize,
    },
}
impl Requests {
    fn rows(&self) -> usize {
        match self {
            Self::Host(ids) => ids.len(),
            Self::Device { rows, .. } => *rows,
        }
    }
    /// Whether `read` can return without waiting: host ids always can, device ids
    /// once the graph published this sequence, and a cancelled request at once so
    /// `read` reports the cancellation.
    fn ready(&self, cancelled: &AtomicBool) -> bool {
        match self {
            Self::Host(_) => true,
            Self::Device { flag, sequence, .. } => {
                unsafe { &*(*flag as *const AtomicU64) }.load(Ordering::Acquire) == *sequence
                    || cancelled.load(Ordering::Acquire)
            }
        }
    }
    fn read(self, cancelled: &AtomicBool) -> Result<Vec<i64>> {
        match self {
            Self::Host(ids) => Ok(ids),
            Self::Device {
                address,
                flag,
                sequence,
                rows,
            } => {
                let ready = unsafe { &*(flag as *const AtomicU64) };
                while ready.load(Ordering::Acquire) != sequence {
                    if cancelled.load(Ordering::Acquire) {
                        return Err("n-gram request cancelled".into());
                    }
                    std::hint::spin_loop();
                }
                Ok(unsafe { std::slice::from_raw_parts(address as *const i64, rows) }.to_vec())
            }
        }
    }
}
const STAGING_FRAMES: usize = 2;
/// Advisory prefetch batches one reader thread may hold. A batch beyond it is
/// discarded. With `PREFETCH_BATCH_ROWS` it bounds read-ahead to about a million
/// rows per table, enough to keep a prefill chunk's rows ahead of the graph.
const PREFETCH_SLOTS: u64 = 256;
/// Rows in one advisory batch. A gather whose ids are published while a batch is
/// being read starts when that batch completes, so this bounds the delay warming
/// can add to a step; it is far below a prefill chunk's row count.
const PREFETCH_BATCH_ROWS: usize = 4096;
/// A reader thread parked with advisory batches waiting re-checks its
/// submission channel at this interval, so a batch handed over between steps is
/// gathered without waiting for the next step to arrive.
const PREFETCH_POLL: std::time::Duration = std::time::Duration::from_micros(200);
/// Recycled row-id buffers kept per table: one per queued advisory batch, plus the
/// prediction scratch and expectations of the frames in flight, so read-ahead that
/// keeps the queue full still allocates nothing in steady state.
const POOLED_BUFFERS: usize = PREFETCH_SLOTS as usize + 2 * STAGING_FRAMES + 2;

struct Job {
    frame: usize,
    generation: u32,
    /// Submission sequence, so an advisory prediction can be compared with the
    /// ids the graph actually published for the same step.
    sequence: u64,
    requests: Requests,
}
#[derive(Clone, Copy, Default)]
struct State {
    sequence: u64,
    generation: u32,
    rows: usize,
    submitted: usize,
    completed: usize,
}
impl State {
    fn prepare(&mut self, rank: usize, sequence: u64, rows: usize, ranks: usize) -> Result<bool> {
        if sequence == 0 || rank >= ranks {
            return Err("invalid Engram submission identity".into());
        }
        let fresh = self.sequence != sequence;
        if fresh {
            if self.sequence > sequence
                || (self.sequence != 0 && self.completed != (1 << ranks) - 1)
            {
                return Err("Engram staging reused before all ranks completed".into());
            }
            *self = Self {
                sequence,
                generation: self.generation.wrapping_add(1),
                rows,
                submitted: 0,
                completed: 0,
            };
        }
        if self.rows != rows {
            return Err("ranks submitted different Engram row counts".into());
        }
        if self.submitted & (1 << rank) != 0 {
            return Err("rank submitted Engram generation twice".into());
        }
        self.submitted |= 1 << rank;
        Ok(fresh)
    }

    fn finish(&mut self, rank: usize, sequence: u64) -> Result<()> {
        if self.sequence != sequence || self.submitted & (1 << rank) == 0 {
            return Err("Engram completion does not match its submission".into());
        }
        if self.completed & (1 << rank) != 0 {
            return Err("rank completed Engram generation twice".into());
        }
        self.completed |= 1 << rank;
        Ok(())
    }
}

/// Row ids predicted for a recent submission, by sequence, kept until the reader
/// thread compares them with what the graph published.
type Predictions = Arc<Mutex<Vec<(u64, Vec<i64>)>>>;

/// Whether every predicted row is among the rows the graph published.
///
/// The prediction covers only host-known positions: the pending token of a
/// decode lane and the prompt positions of a prefill lane, never graph-produced
/// drafts. A correct prediction is therefore a subset of what the graph publishes,
/// and a predicted row the graph did not publish is the disagreement this check
/// exists to report. `expected` is sorted and deduplicated; `published` is scratch.
fn predicted_rows_published(expected: &[i64], ids: &[i64], published: &mut Vec<i64>) -> bool {
    published.clear();
    published.extend_from_slice(ids);
    published.sort_unstable();
    let mut rest = &published[..];
    expected.iter().all(|id| {
        let skip = rest.partition_point(|value| value < id);
        rest = &rest[skip..];
        rest.first() == Some(id)
    })
}

/// What the reader thread does next.
enum Work {
    /// A real gather whose ids are readable, which always outranks warming.
    Gather(Job),
    /// One advisory batch, which warms the cache without taking a frame.
    Warm(Vec<i64>),
    /// Nothing was ready: the poll interval expired, a wake token arrived, or the
    /// held gather is still waiting for its graph to publish ids.
    Wait,
    /// The submission channel is closed and drained.
    Stop,
}

/// Choose the reader thread's next unit of work.
///
/// A submission is taken into `held` as soon as it arrives, but it is gathered
/// only once its ids are readable. Device ids exist only after the step's graph
/// starts, which is after the previous step's graph finishes, so a thread that
/// blocked on them would sit idle through the whole replay that advisory work is
/// meant to overlap. While the held gather waits, advisory batches run one at a
/// time, so the gather starts at most one batch after its ids appear.
///
/// A closed advisory channel is never a shutdown signal: real gathers may still be
/// queued, and a thread that exited on it would strand the frame every rank is
/// waiting on. With nothing held or ready the thread blocks on the wake token, or
/// polls briefly while `queued` advisory batches wait, so a batch handed over
/// between steps is warmed without waiting for the next submission.
fn next_work(
    receiver: &mpsc::Receiver<Job>,
    advisory: &mpsc::Receiver<Vec<i64>>,
    wake: &mpsc::Receiver<()>,
    queued: u64,
    held: &mut Option<Job>,
    cancelled: &AtomicBool,
) -> Work {
    if held.is_none() {
        match receiver.try_recv() {
            Ok(job) => *held = Some(job),
            Err(mpsc::TryRecvError::Disconnected) => return Work::Stop,
            Err(mpsc::TryRecvError::Empty) => {}
        }
    }
    if let Some(job) = held.as_ref() {
        if job.requests.ready(cancelled) {
            return Work::Gather(held.take().unwrap());
        }
        return match advisory.try_recv() {
            Ok(ids) => Work::Warm(ids),
            Err(_) => Work::Wait,
        };
    }
    if let Ok(ids) = advisory.try_recv() {
        return Work::Warm(ids);
    }
    if queued == 0 {
        // A send on the advisory channel cannot release a thread parked on the
        // submission channel, so idleness is waited out on a token both producers
        // post after enqueueing. The token stays in the channel until it is read,
        // so a wake that races this park cannot be lost, and a closed wake channel
        // only sends the thread back to re-probe, where a drained submission
        // channel decides whether to stop.
        wake.recv().ok();
        Work::Wait
    } else {
        match receiver.recv_timeout(PREFETCH_POLL) {
            // Re-probe through `held`, so a device gather is still taken only
            // once its ids are readable.
            Ok(job) => {
                *held = Some(job);
                Work::Wait
            }
            Err(mpsc::RecvTimeoutError::Timeout) => Work::Wait,
            Err(mpsc::RecvTimeoutError::Disconnected) => Work::Stop,
        }
    }
}

struct Layer {
    staging: Arc<[Staging; STAGING_FRAMES]>,
    sender: Option<mpsc::SyncSender<Job>>,
    /// Advisory warming batches waiting for the reader thread. Its sender is
    /// dropped with the layer, but a closed advisory channel is never a reason
    /// for the thread to stop: real gathers may still be queued.
    prefetch: Option<mpsc::SyncSender<Vec<i64>>>,
    /// Coalescing wake-up for the thread parked in `next_work`. Both producers
    /// post one token after enqueueing; capacity one means a pending token makes
    /// further posts free, and the queue's memory means a racing park still sees it.
    wake: mpsc::SyncSender<()>,
    counters: Arc<TableCounters>,
    /// Rows one gather serves, the chunk bound for advisory batches.
    max_rows: usize,
    worker: Option<JoinHandle<()>>,
    state: Mutex<[State; STAGING_FRAMES]>,
    predicted: Predictions,
    /// Recycled row-id buffers, so a step's prediction and the batches it hands to
    /// the reader thread allocate nothing in steady state.
    pool: Arc<Mutex<Vec<Vec<i64>>>>,
    cancelled: Arc<AtomicBool>,
}
impl Drop for Layer {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
        self.sender.take();
        self.prefetch.take();
        // The thread waits on this token rather than on the submission channel, so
        // closing the work senders does not release it by itself. Post one and let
        // it re-probe: the drained, now-closed submission channel ends the loop.
        let _ = self.wake.try_send(());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

pub(crate) struct Shared {
    pub index: Arc<Index>,
    layers: Vec<Layer>,
    ranks: usize,
    max_tokens: usize,
    stats: Arc<Stats>,
    storage: Storage,
    prefetch: bool,
}
impl Shared {
    #[cfg(test)]
    pub(crate) fn layout_fixture(index: Index, ranks: usize) -> Arc<Self> {
        Arc::new(Self {
            index: Arc::new(index),
            layers: Vec::new(),
            ranks,
            max_tokens: 4096,
            stats: Arc::new(Stats::default()),
            storage: Storage::Disk,
            prefetch: false,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        index: Index,
        files: &[Arc<std::fs::File>],
        ranks: usize,
        max_tokens: usize,
        context: usize,
        device: i32,
        storage: Storage,
        cache_bytes: u64,
        workers: usize,
        resident_scales: bool,
        prefetch: bool,
    ) -> Result<Arc<Self>> {
        if storage == Storage::Ram && cache_bytes != 0 {
            return Err("embedding row cache requires disk table storage".into());
        }
        if prefetch && storage != Storage::Disk {
            return Err("embedding row prefetch requires disk table storage".into());
        }
        let index = Arc::new(index);
        let stats = Arc::new(Stats::with_tables(
            index.tables.iter().map(|table| table.layer),
        ));
        // Publish the enabled state once, so a zero hit rate tells an operator
        // whether the prefetcher was off or simply could not keep up.
        stats.set_prefetch_enabled(prefetch && cache_bytes != 0);
        let table_bytes: u128 = index
            .tables
            .iter()
            .flat_map(|table| &table.planes)
            .map(|plane| plane.nbytes as u128)
            .sum();
        let readers = index
            .tables
            .iter()
            .enumerate()
            .map(|(ordinal, table)| {
                let bytes: u128 = table.planes.iter().map(|plane| plane.nbytes as u128).sum();
                let budget = usize::try_from(cache_bytes as u128 * bytes / table_bytes)
                    .map_err(|_| "embedding row cache exceeds host address range")?;
                let rows = max_tokens
                    .checked_mul(table.heads)
                    .ok_or("embedding row capacity exceeds host range")?;
                reader::Reader::from_files(
                    files,
                    &table.planes,
                    storage == Storage::Ram,
                    budget,
                    rows,
                    stats.clone(),
                    stats.table(ordinal).expect("one counter set per table"),
                    workers,
                    resident_scales,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Self::with_readers(
            index,
            readers,
            ranks,
            max_tokens,
            context,
            device,
            stats,
            storage,
            prefetch && cache_bytes != 0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn with_readers(
        index: Arc<Index>,
        readers: Vec<reader::Reader>,
        ranks: usize,
        max_tokens: usize,
        context: usize,
        device: i32,
        stats: Arc<Stats>,
        storage: Storage,
        prefetch: bool,
    ) -> Result<Arc<Self>> {
        if ranks == 0 || ranks > 16 || max_tokens == 0 || readers.len() != index.tables.len() {
            return Err("invalid embedding delivery geometry".into());
        }
        let stream = Stream::open(context, device)?;
        let mut layers = Vec::new();
        for (ordinal, (table, mut reader)) in index.tables.iter().zip(readers).enumerate() {
            let staging_bytes = table.staging_payload_bytes(max_tokens)?;
            let mut staging = Vec::with_capacity(STAGING_FRAMES);
            for _ in 0..STAGING_FRAMES {
                staging.push(Staging {
                    data: stream.mapped_host_buffer(staging_bytes)?,
                    ready: stream.mapped_host_buffer(4)?,
                    error: Mutex::new(None),
                });
            }
            let staging: Arc<[Staging; STAGING_FRAMES]> = Arc::new(
                staging
                    .try_into()
                    .map_err(|_| "invalid Engram staging frame count")?,
            );
            let (sender, receiver) = mpsc::sync_channel::<Job>(STAGING_FRAMES);
            let (prefetch_sender, prefetch_receiver) = mpsc::sync_channel::<Vec<i64>>(
                PREFETCH_SLOTS as usize,
            );
            let (wake_sender, wake_receiver) = mpsc::sync_channel::<()>(1);
            let counters = stats.table(ordinal).expect("one counter set per table");
            let buffers = staging.clone();
            let planes = table.planes;
            let row_bytes = planes.iter().map(|p| p.row_bytes).sum::<usize>();
            let max_rows = max_tokens
                .checked_mul(table.heads)
                .ok_or("embedding row capacity exceeds host range")?;
            let cancelled = Arc::new(AtomicBool::new(false));
            let stop = cancelled.clone();
            let queued = counters.clone();
            let predictions: Predictions = Arc::new(Mutex::new(Vec::new()));
            let observed = predictions.clone();
            let pool = Arc::new(Mutex::new(Vec::new()));
            let recycled = pool.clone();
            let worker = thread::Builder::new()
                .name(format!("flare-{}", table.event))
                .spawn(move || {
                    let mut held = None;
                    // Sorted copy of a step's published ids, reused across steps.
                    let mut published = Vec::new();
                    loop {
                        // Readable submissions always outrank advisory warming.
                        let job = match next_work(
                            &receiver,
                            &prefetch_receiver,
                            &wake_receiver,
                            queued.prefetch_queue_batches.load(Ordering::Acquire),
                            &mut held,
                            &stop,
                        ) {
                            Work::Gather(job) => job,
                            Work::Warm(ids) => {
                                queued
                                    .prefetch_queue_batches
                                    .fetch_sub(1, Ordering::Release);
                                let warmed = std::panic::catch_unwind(
                                    std::panic::AssertUnwindSafe(|| {
                                        reader.prefetch(&planes, &ids)
                                    }),
                                );
                                match warmed {
                                    Ok(Ok(_)) => {}
                                    // A failed advisory read is swallowed. The real
                                    // gather for these rows reports the same disk
                                    // condition, and losing cache warmth must never
                                    // poison a frame or stop serving. Upstream instead
                                    // disables its cache here because its cache is the
                                    // only source of truth for those bytes.
                                    Ok(Err(_)) | Err(_) => {
                                        queued.prefetch_errors.fetch_add(1, Ordering::Relaxed);
                                    }
                                }
                                // Hand the batch's buffer back for the next step.
                                let mut spare = ids;
                                spare.clear();
                                let mut pool = recycled.lock().unwrap();
                                if pool.len() < POOLED_BUFFERS {
                                    pool.push(spare);
                                }
                                drop(pool);
                                continue;
                            }
                            Work::Wait => {
                                // A held gather is waiting for its graph: keep
                                // observing the flag as closely as the old blocking
                                // read did, between advisory batches.
                                std::hint::spin_loop();
                                continue;
                            }
                            Work::Stop => break,
                        };
                        // Each frame stays owned until every rank retires its consuming graph.
                        // The reader alone writes its data before publishing the ready flag.
                        let buffers = &buffers[job.frame];
                        let output = unsafe {
                            std::slice::from_raw_parts_mut(
                                buffers.data.address() as *mut u8,
                                job.requests.rows() * row_bytes,
                            )
                        };
                        let mut ids = Vec::new();
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            if stop.load(Ordering::Acquire) {
                                return Err("n-gram request cancelled".into());
                            }
                            ids = job.requests.read(&stop)?;
                            reader.gather(&planes, &ids, output)
                        }))
                        .unwrap_or_else(|_| Err("Engram reader panicked".into()));
                        if let Err(error) = result {
                            output.fill(0);
                            *buffers.error.lock().unwrap() = Some(error);
                            stop.store(true, Ordering::Release);
                        }
                        // Release failed reads too, so graph shutdown can drain before reporting the error.
                        buffers.signal(job.generation);
                        // Check the prediction only after the frame is released, so
                        // the comparison never delays the graph.
                        let expected = {
                            let mut predicted = observed.lock().unwrap();
                            predicted
                                .iter()
                                .position(|(entry, _)| *entry == job.sequence)
                                .map(|position| predicted.remove(position).1)
                        };
                        if let Some(mut expected) = expected {
                            if !ids.is_empty()
                                && !predicted_rows_published(&expected, &ids, &mut published)
                            {
                                queued.prediction_mismatches.fetch_add(1, Ordering::Relaxed);
                            }
                            expected.clear();
                            let mut pool = recycled.lock().unwrap();
                            if pool.len() < POOLED_BUFFERS {
                                pool.push(expected);
                            }
                        }
                    }
                })
                .map_err(|e| format!("start Engram reader: {e}"))?;
            layers.push(Layer {
                staging,
                cancelled,
                sender: Some(sender),
                prefetch: Some(prefetch_sender),
                wake: wake_sender,
                counters,
                max_rows,
                worker: Some(worker),
                state: Mutex::new(std::array::from_fn(|_| State::default())),
                predicted: predictions,
                pool,
            });
        }
        Ok(Arc::new(Self {
            index,
            layers,
            ranks,
            max_tokens,
            stats,
            storage,
            prefetch,
        }))
    }

    /// Take a recycled row-id buffer for one table, or a fresh one the first time.
    fn take_buffer(&self, table: usize) -> Vec<i64> {
        self.layers
            .get(table)
            .and_then(|layer| layer.pool.lock().unwrap().pop())
            .unwrap_or_default()
    }

    /// Return a used row-id buffer, capped so the pool cannot grow with the batch.
    fn return_buffer(&self, table: usize, mut buffer: Vec<i64>) {
        buffer.clear();
        if let Some(layer) = self.layers.get(table) {
            let mut pool = layer.pool.lock().unwrap();
            if pool.len() < POOLED_BUFFERS {
                pool.push(buffer);
            }
        }
    }

    /// Whether advisory warming can do anything: it is enabled and the tables are
    /// read from disk into a nonzero row cache.
    pub(crate) fn prefetching(&self) -> bool {
        self.prefetch && self.storage == Storage::Disk && self.stats.cache_bytes() != 0
    }

    /// Query positions one advisory batch covers for `table`.
    pub(crate) fn batch_tokens(&self, table: usize) -> usize {
        self.index
            .tables
            .get(table)
            .map_or(1, |table| (PREFETCH_BATCH_ROWS / table.heads.max(1)).max(1))
    }

    /// Query positions `table` may hold read ahead of the step being launched: a
    /// quarter of its row cache, so protected read-ahead rows never crowd out the
    /// rows in steady use, and no more than its advisory queue holds.
    pub(crate) fn read_ahead_tokens(&self, table: usize) -> usize {
        let Some(entry) = self.index.tables.get(table) else {
            return 0;
        };
        let capacity = self
            .stats
            .table(table)
            .map_or(0, |counters| counters.capacity_rows.load(Ordering::Relaxed) as usize);
        let rows = (capacity / 4).min(PREFETCH_SLOTS as usize * PREFETCH_BATCH_ROWS);
        rows / entry.heads.max(1)
    }

    /// Append to `out` the rows the query positions `[first, end)` of `tokens`
    /// will request from `table`. Committed history before `first` comes from
    /// `tokens` too. `excluded` reports a position that never enters an n-gram
    /// history, such as an image position. Rows outside this rank's partition and
    /// unaddressable windows report nothing.
    pub(crate) fn predict_rows(
        &self,
        table: usize,
        tokens: &[u32],
        first: usize,
        end: usize,
        excluded: impl Fn(usize) -> bool,
        out: &mut Vec<i64>,
    ) {
        let Some(entry) = self.index.tables.get(table) else {
            return;
        };
        if first >= end || tokens.len() < end {
            return;
        }
        let base = first.saturating_sub(history_depth(&entry.geometry) - 1);
        // Compress and derive into recycled buffers: a prefill window is as wide
        // as its token count, so the submission path must not allocate per step.
        let mut compressed = self.take_buffer(table);
        compress_into(
            &self.index,
            entry,
            base,
            &tokens[base..end],
            excluded,
            &mut compressed,
        );
        let (history, queries) = compressed.split_at(compressed.len() - (end - first));
        let mut ids = self.take_buffer(table);
        query_row_ids(&self.index, entry, history, queries, &mut ids);
        out.extend(ids.iter().copied().filter(|&id| id >= 0));
        self.return_buffer(table, ids);
        self.return_buffer(table, compressed);
    }

    /// Leave the rows predicted for `sequence` for the reader thread to check
    /// against the ids the graph publishes for that step. `ids` is taken and
    /// replaced with a recycled buffer.
    pub(crate) fn expect(&self, table: usize, sequence: u64, ids: &mut Vec<i64>) {
        let Some(layer) = self.layers.get(table) else {
            return;
        };
        if ids.is_empty() {
            return;
        }
        let mut expected = std::mem::replace(ids, self.take_buffer(table));
        expected.sort_unstable();
        expected.dedup();
        let mut predicted = layer.predicted.lock().unwrap();
        // Two frames may be in flight and a step can be superseded before the
        // reader reaches it; keep the most recent few and let older ones age out.
        predicted.retain(|(entry, _)| *entry != sequence);
        predicted.push((sequence, expected));
        if predicted.len() > 2 * STAGING_FRAMES {
            let (_, stale) = predicted.remove(0);
            self.return_buffer(table, stale);
        }
    }

    /// Hand predicted row ids to one table's reader thread for cache warming, in
    /// batches of at most `PREFETCH_BATCH_ROWS`.
    ///
    /// Advisory by contract: it never blocks, never reports an error, and is a
    /// no-op unless the table's rows can actually be cached. A batch that finds
    /// the queue full is discarded together with every later batch, matching the
    /// upstream prefetch queue, which also drops the newest request rather than
    /// delaying a serving thread. Returns whether every row was queued, so a
    /// caller tracking read-ahead progress can retry the rest later.
    pub(crate) fn prefetch(&self, table: usize, ids: &[i64]) -> bool {
        if !self.prefetch || self.storage != Storage::Disk {
            return false;
        }
        let Some(layer) = self.layers.get(table) else {
            return false;
        };
        if layer.cancelled.load(Ordering::Acquire) || ids.is_empty() || layer.max_rows == 0 {
            return false;
        }
        let Some(sender) = layer.prefetch.as_ref() else {
            return false;
        };
        let batch_rows = layer.max_rows.min(PREFETCH_BATCH_ROWS);
        for (index, chunk) in ids.chunks(batch_rows).enumerate() {
            let refuse = || {
                layer
                    .counters
                    .prefetch_rows_dropped
                    .fetch_add((ids.len() - index * batch_rows) as u64, Ordering::Relaxed);
                false
            };
            let depth = layer.counters.prefetch_queue_batches.load(Ordering::Acquire);
            if depth >= PREFETCH_SLOTS {
                return refuse();
            }
            let mut batch = self.take_buffer(table);
            batch.extend_from_slice(chunk);
            if let Err(error) = sender.try_send(batch) {
                // `try_send` hands the buffer back, so it returns to the pool
                // rather than being freed.
                let batch = match error {
                    mpsc::TrySendError::Full(batch) | mpsc::TrySendError::Disconnected(batch) => {
                        batch
                    }
                };
                self.return_buffer(table, batch);
                return refuse();
            }
            layer
                .counters
                .prefetch_queue_batches
                .fetch_add(1, Ordering::Release);
            layer
                .counters
                .prefetch_rows
                .fetch_add(chunk.len() as u64, Ordering::Relaxed);
            // Post after enqueueing, so a thread that parks between the probes
            // still finds the token waiting. A pending token makes this free.
            let _ = layer.wake.try_send(());
        }
        true
    }

    #[cfg(any(feature = "server", test))]
    pub(crate) fn stats(&self) -> Arc<Stats> {
        self.stats.clone()
    }

    pub(crate) fn cache_bytes(&self) -> u64 {
        self.stats.cache_bytes()
    }
    fn validate_start(&self, table: usize, rank: usize, sequence: u64, rows: usize) -> Result<()> {
        if rows > self.max_tokens * self.index.tables[table].heads {
            return Err("Engram input exceeds staged capacity".into());
        }
        let layer = &self.layers[table];
        let states = layer.state.lock().unwrap();
        for staging in layer.staging.iter() {
            if let Some(error) = staging.error.lock().unwrap().as_ref() {
                return Err(error.clone());
            }
        }
        let mut state = states[sequence as usize % STAGING_FRAMES];
        state.prepare(rank, sequence, rows, self.ranks)?;
        Ok(())
    }

    fn start(
        &self,
        table: usize,
        rank: usize,
        sequence: u64,
        requests: Requests,
    ) -> Result<(usize, u32)> {
        let rows = requests.rows();
        if rows > self.max_tokens * self.index.tables[table].heads {
            return Err("Engram input exceeds staged capacity".into());
        }
        let layer = &self.layers[table];
        let mut states = layer.state.lock().unwrap();
        for staging in layer.staging.iter() {
            if let Some(error) = staging.error.lock().unwrap().as_ref() {
                return Err(error.clone());
            }
        }
        let frame = sequence as usize % STAGING_FRAMES;
        let mut state = states[frame];
        let fresh = state.prepare(rank, sequence, rows, self.ranks)?;
        if fresh {
            layer
                .sender
                .as_ref()
                .unwrap()
                .send(Job {
                    frame,
                    generation: state.generation,
                    sequence,
                    requests,
                })
                .map_err(|_| "Engram reader disconnected")?;
            // Same reason as the advisory path: the token, not the queue, is what
            // releases a thread parked between steps.
            let _ = layer.wake.try_send(());
        }
        states[frame] = state;
        Ok((rows, state.generation))
    }
    fn finish(&self, rank: usize, sequence: u64) -> Result<()> {
        let mut error = None;
        for layer in &self.layers {
            let mut states = layer.state.lock().unwrap();
            let frame = sequence as usize % STAGING_FRAMES;
            states[frame].finish(rank, sequence)?;
            error = error.or_else(|| layer.staging[frame].error.lock().unwrap().clone());
        }
        error.map_or(Ok(()), Err)
    }
}

#[derive(Clone)]
pub struct Binding {
    pub(crate) shared: Arc<Shared>,
    pub(crate) rank: usize,
    pub(crate) events: Vec<usize>,
}
impl std::fmt::Debug for Binding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngramBinding")
            .field("rank", &self.rank)
            .finish()
    }
}
struct Transfer {
    stream: Stream,
    flags: [usize; STAGING_FRAMES],
    weights: usize,
    scales: usize,
    event: usize,
    request: Option<(usize, usize)>,
}
pub(crate) struct Transfers {
    binding: Binding,
    layers: Vec<Transfer>,
}
impl Transfers {
    pub fn new(binding: Binding, manifest: &ExecutionManifest) -> Result<Self> {
        if let Some(buffer) = manifest
            .buffers
            .iter()
            .find(|buffer| buffer.name == "readiness_timestamps")
        {
            if buffer.shape[1] / 3 != binding.shared.index.tables.len() {
                return Err(
                    "readiness interval count differs from the embedding table catalog".into(),
                );
            }
        }
        let mut layers = Vec::new();
        for (i, table) in binding.shared.index.tables.iter().enumerate() {
            let stream = Stream::open(manifest.context, manifest.device)?;
            let mut flags = [0; STAGING_FRAMES];
            for (flag, staging) in flags
                .iter_mut()
                .zip(binding.shared.layers[i].staging.iter())
            {
                *flag = stream.mapped_address(&staging.ready)?;
            }
            let request = table
                .request
                .as_ref()
                .map(|name| {
                    let buffer = manifest.buffer(name)?;
                    let address = stream.host_address(buffer.address, buffer.nbytes)?;
                    Ok::<_, Error>((address, address + buffer.nbytes - 8))
                })
                .transpose()?;
            layers.push(Transfer {
                stream,
                flags,
                weights: manifest.buffer(&table.weights)?.address,
                scales: manifest.buffer(&table.scales)?.address,
                event: binding.events[i],
                request,
            });
        }
        Ok(Self { binding, layers })
    }
    pub fn validate(&self, sequence: u64, tokens: usize, uploads: &[Upload<'_>]) -> Result<()> {
        for (i, (transfer, table)) in self
            .layers
            .iter()
            .zip(&self.binding.shared.index.tables)
            .enumerate()
        {
            let rows = if transfer.request.is_some() {
                tokens
                    .checked_mul(table.heads)
                    .ok_or("Engram row count overflow")?
            } else {
                let upload = uploads
                    .iter()
                    .find(|u| u.buffer == table.hashes)
                    .ok_or("step lacks n-gram hash IDs")?;
                if upload.offset != 0 || upload.bytes.len() % (table.heads * 8) != 0 {
                    return Err("invalid n-gram hash upload".into());
                }
                upload.bytes.len() / 8
            };
            self.binding
                .shared
                .validate_start(i, self.binding.rank, sequence, rows)?;
        }
        Ok(())
    }

    pub fn enqueue(
        &self,
        sequence: u64,
        tokens: usize,
        uploads: &[Upload<'_>],
        inputs_ready: &Event,
    ) -> Result<()> {
        self.validate(sequence, tokens, uploads)?;
        for (i, (transfer, table)) in self
            .layers
            .iter()
            .zip(&self.binding.shared.index.tables)
            .enumerate()
        {
            let heads = table.heads;
            let requests = if let Some((address, flag)) = transfer.request {
                Requests::Device {
                    address,
                    flag,
                    sequence,
                    rows: tokens * heads,
                }
            } else {
                let upload = uploads
                    .iter()
                    .find(|u| u.buffer == table.hashes)
                    .ok_or("step lacks n-gram hash IDs")?;
                if upload.offset != 0 || upload.bytes.len() % (heads * 8) != 0 {
                    return Err("invalid n-gram hash upload".into());
                }
                Requests::Host(
                    upload
                        .bytes
                        .chunks_exact(8)
                        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
                        .collect(),
                )
            };
            let (rows, generation) =
                self.binding
                    .shared
                    .start(i, self.binding.rank, sequence, requests)?;
            let frame = sequence as usize % STAGING_FRAMES;
            let staging = &self.binding.shared.layers[i].staging[frame];
            // The model stream records inputs_ready after the preceding graph, so DMA
            // cannot overwrite tables still consumed by that graph. Each graph waits for
            // its host gather before the following graph can overwrite the mailbox.
            transfer.stream.wait(inputs_ready)?;
            unsafe {
                transfer
                    .stream
                    .wait_value(transfer.flags[frame], generation)?;
                transfer.stream.upload(
                    transfer.weights,
                    &staging.data,
                    0,
                    rows * table.planes[0].row_bytes,
                )?;
                transfer.stream.upload(
                    transfer.scales,
                    &staging.data,
                    rows * table.planes[0].row_bytes,
                    rows * table.planes[1].row_bytes,
                )?;
                // The consumer is enqueued before a following submission re-records this event.
                transfer.stream.record_borrowed(transfer.event)?;
            }
        }
        Ok(())
    }
    pub fn finish(&self, sequence: u64) -> Result<()> {
        self.binding.shared.finish(self.binding.rank, sequence)
    }
    pub fn cancel(&self) {
        for layer in &self.binding.shared.layers {
            layer.cancelled.store(true, Ordering::Release);
        }
    }
    pub fn drain(&self) -> Result<()> {
        let mut error = None;
        for transfer in &self.layers {
            error = error.or(transfer.stream.synchronize().err());
        }
        error.map_or(Ok(()), Err)
    }
}

pub(crate) fn consumes(kind: GraphKind) -> bool {
    matches!(
        kind,
        GraphKind::Prefill
            | GraphKind::ScoredPrefill
            | GraphKind::Decode
            | GraphKind::Speculative
            | GraphKind::Mixed
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staging_budget_covers_two_payloads_and_ready_flags_per_table() {
        let table = Table {
            layer: 1,
            planes: [Plane { file: 0, offset: 0, nbytes: 1024, row_bytes: 64 },
                Plane { file: 0, offset: 1024, nbytes: 128, row_bytes: 8 }],
            heads: 4, rows: 16, first_row: 0, geometry: RowGeometry::default(),
            weights: "weights".into(), scales: "scales".into(),
            hashes: "hashes".into(), event: "ready".into(), request: None,
        };
        let index = Index {
            tables: vec![table.clone(), table],
            token_map: Vec::new(),
            pad_token_id: 0,
        };
        assert_eq!(staging_bytes(&index, 128).unwrap(), 2 * 2 * (128 * 4 * (64 + 8) + 4));
        assert!(staging_bytes(&index, usize::MAX).is_err());
    }

    #[test]
    fn two_frames_require_all_rank_retirement_before_reuse() {
        let mut states: [State; STAGING_FRAMES] = std::array::from_fn(|_| State::default());
        for sequence in 1..=2 {
            let state = &mut states[sequence as usize % STAGING_FRAMES];
            assert!(state.prepare(0, sequence, 24, 2).unwrap());
            assert!(!state.prepare(1, sequence, 24, 2).unwrap());
        }
        let first = &mut states[1];
        first.finish(0, 1).unwrap();
        assert!(first.prepare(0, 3, 48, 2).is_err());
        assert_eq!(first.sequence, 1);
        first.finish(1, 1).unwrap();
        assert!(first.prepare(0, 3, 48, 2).unwrap());
        assert_eq!(first.generation, 2);
        assert_eq!(first.rows, 48);
        assert_eq!(states[0].sequence, 2);
        assert_eq!(states[0].completed, 0);
    }

    #[test]
    fn frame_rejects_stale_duplicate_and_mismatched_rank_work() {
        let mut state = State::default();
        assert!(state.prepare(0, 0, 24, 2).is_err());
        assert!(state.prepare(2, 1, 24, 2).is_err());
        state.prepare(0, 1, 24, 2).unwrap();
        assert!(state.prepare(0, 1, 24, 2).is_err());
        assert!(state.prepare(1, 1, 48, 2).is_err());
        assert!(state.finish(1, 1).is_err());
        assert!(state.finish(0, 3).is_err());
        state.prepare(1, 1, 24, 2).unwrap();
        state.finish(0, 1).unwrap();
        assert!(state.finish(0, 1).is_err());
        state.finish(1, 1).unwrap();
        state.prepare(0, 3, 24, 2).unwrap();
        assert!(state.prepare(1, 1, 24, 2).is_err());
    }

    #[test]
    fn device_request_wait_observes_publication_and_cancellation() {
        let ids = [17i64, 31, 47];
        let flag = AtomicU64::new(0);
        let cancelled = AtomicBool::new(false);
        let request = || Requests::Device {
            address: ids.as_ptr() as usize,
            flag: &flag as *const AtomicU64 as usize,
            sequence: 7,
            rows: ids.len(),
        };
        thread::scope(|scope| {
            let worker = scope.spawn(|| request().read(&cancelled));
            flag.store(7, Ordering::Release);
            assert_eq!(worker.join().unwrap().unwrap(), ids);
        });
        flag.store(0, Ordering::Release);
        cancelled.store(true, Ordering::Release);
        assert!(request().read(&cancelled).is_err());
    }

    fn job(sequence: u64) -> Job {
        Job {
            frame: (sequence as usize) % STAGING_FRAMES,
            generation: sequence as u32,
            sequence,
            requests: Requests::Host(Vec::new()),
        }
    }

    /// The reader loop's channels and held job, driven one decision at a time.
    struct Loop {
        receiver: mpsc::Receiver<Job>,
        advisory: mpsc::Receiver<Vec<i64>>,
        wake: mpsc::Receiver<()>,
        held: Option<Job>,
        cancelled: AtomicBool,
    }

    impl Loop {
        fn new() -> (mpsc::Sender<Job>, mpsc::Sender<Vec<i64>>, mpsc::Sender<()>, Self) {
            let (sender, receiver) = mpsc::channel();
            let (advisory_sender, advisory) = mpsc::channel();
            let (wake_sender, wake) = mpsc::channel();
            let state = Self {
                receiver,
                advisory,
                wake,
                held: None,
                cancelled: AtomicBool::new(false),
            };
            (sender, advisory_sender, wake_sender, state)
        }

        fn next(&mut self, queued: u64) -> Work {
            next_work(
                &self.receiver,
                &self.advisory,
                &self.wake,
                queued,
                &mut self.held,
                &self.cancelled,
            )
        }

        /// Re-probe through `Wait` as the reader thread does.
        fn settle(&mut self, queued: u64) -> Work {
            loop {
                match self.next(queued) {
                    Work::Wait => continue,
                    work => return work,
                }
            }
        }
    }

    #[test]
    fn reader_thread_prefers_a_readable_gather_over_advisory_warming() {
        let (sender, advisory, _wake, mut state) = Loop::new();
        // Advisory work is queued first, so a priority inversion would show up here.
        advisory.send(vec![7, 12]).unwrap();
        sender.send(job(1)).unwrap();
        assert!(
            matches!(state.next(1), Work::Gather(_)),
            "a readable submission must outrank a queued batch"
        );
        assert!(matches!(state.next(1), Work::Warm(_)));
        // Only a closed and drained submission channel ends the thread.
        drop(sender);
        assert!(matches!(state.next(0), Work::Stop));
    }

    #[test]
    fn an_unpublished_device_gather_yields_to_warming_until_its_ids_appear() {
        let ids = [17i64, 31, 47];
        let flag = AtomicU64::new(0);
        let (sender, advisory, _wake, mut state) = Loop::new();
        sender
            .send(Job {
                frame: 1,
                generation: 1,
                sequence: 5,
                requests: Requests::Device {
                    address: ids.as_ptr() as usize,
                    flag: &flag as *const AtomicU64 as usize,
                    sequence: 5,
                    rows: ids.len(),
                },
            })
            .unwrap();
        advisory.send(vec![1]).unwrap();
        advisory.send(vec![2]).unwrap();
        // The step's graph has not started, so its rows cannot be read yet. The
        // thread holds the job and spends the wait on read-ahead, which is the
        // replay window advisory work exists to use.
        assert!(matches!(state.next(2), Work::Warm(batch) if batch == [1]));
        assert!(matches!(state.next(1), Work::Warm(batch) if batch == [2]));
        assert!(state.held.is_some(), "the submission must stay held");
        assert!(matches!(state.next(0), Work::Wait));
        advisory.send(vec![3]).unwrap();
        flag.store(5, Ordering::Release);
        // Publication makes the held gather outrank the batch behind it.
        assert!(matches!(state.next(1), Work::Gather(job) if job.sequence == 5));
        assert!(matches!(state.next(1), Work::Warm(batch) if batch == [3]));
        drop(sender);
        assert!(matches!(state.next(0), Work::Stop));
    }

    #[test]
    fn a_cancelled_unpublished_gather_is_released_to_report_cancellation() {
        let flag = AtomicU64::new(0);
        let (sender, _advisory, _wake, mut state) = Loop::new();
        sender
            .send(Job {
                frame: 0,
                generation: 1,
                sequence: 4,
                requests: Requests::Device {
                    address: 0,
                    flag: &flag as *const AtomicU64 as usize,
                    sequence: 4,
                    rows: 0,
                },
            })
            .unwrap();
        assert!(matches!(state.next(0), Work::Wait));
        // A closed submission channel does not end the thread while a gather is
        // held: the frame every rank waits on still has to be released.
        drop(sender);
        assert!(matches!(state.next(0), Work::Wait));
        // A graph that will never publish must not pin the thread: cancellation
        // hands the job to the gather path, whose read reports the cancellation
        // and still releases the frame.
        state.cancelled.store(true, Ordering::Release);
        assert!(matches!(state.next(0), Work::Gather(job) if job.sequence == 4));
        assert!(matches!(state.next(0), Work::Stop));
    }

    #[test]
    fn a_closed_advisory_channel_is_never_a_shutdown_signal() {
        let (sender, advisory, _wake, mut state) = Loop::new();
        sender.send(job(3)).unwrap();
        // Dropping the advisory sender with real work still queued must not end the
        // thread: a graph would wait forever on the frame that work publishes.
        drop(advisory);
        assert!(matches!(state.next(0), Work::Gather(_)));
        drop(sender);
        assert!(matches!(state.next(0), Work::Stop));
    }

    #[test]
    fn a_parked_reader_wakes_for_a_submission() {
        let (sender, _advisory, wake, mut state) = Loop::new();
        // A Receiver is not shareable, so the parked thread owns every channel.
        let parked = thread::spawn(move || state.settle(0));
        // `Shared::start` enqueues the job and then posts the token, which is what
        // releases a thread blocked between steps.
        sender.send(job(9)).unwrap();
        wake.send(()).unwrap();
        assert!(matches!(parked.join().unwrap(), Work::Gather(job) if job.sequence == 9));
    }

    #[test]
    fn a_parked_reader_wakes_for_an_advisory_batch() {
        let (sender, advisory, wake, mut state) = Loop::new();
        let parked = thread::spawn(move || state.settle(0));
        // `Shared::prefetch` touches only the advisory channel and the token. A
        // thread parked on the submission channel alone never sees it, and then the
        // step the batch was meant to warm still waits on the disk.
        advisory.send(vec![4, 5]).unwrap();
        wake.send(()).unwrap();
        assert!(
            matches!(parked.join().unwrap(), Work::Warm(ids) if ids == vec![4, 5]),
            "an idle reader must warm the rows predicted for the next step"
        );
        drop(sender);
    }

    #[test]
    fn a_stray_wake_token_reprobes_instead_of_stopping() {
        let (sender, _advisory, wake, mut state) = Loop::new();
        // Coalescing means a token can be posted with nothing behind it, or already
        // consumed by a competing wake. Either way the thread re-probes and stays.
        wake.send(()).unwrap();
        assert!(matches!(state.next(0), Work::Wait));
        drop(sender);
    }

    #[test]
    fn predicted_rows_must_all_be_published() {
        let mut scratch = Vec::new();
        // Graph-produced draft rows (90, 91) were never predicted, and a row may be
        // published by several positions: neither is a disagreement.
        assert!(predicted_rows_published(&[3, 8, 12], &[12, 90, 3, 8, 91, 3], &mut scratch));
        assert!(predicted_rows_published(&[], &[5], &mut scratch));
        // A predicted row the graph did not request means the host derivation
        // computed a different id for a position it claimed to know.
        assert!(!predicted_rows_published(&[3, 9], &[3, 8, 12], &mut scratch));
        assert!(!predicted_rows_published(&[13], &[3, 8, 12], &mut scratch));
    }

    #[cfg(feature = "server")]
    #[test]
    fn metrics_publish_aggregates_and_one_labelled_series_per_table() {
        let stats = Stats::with_tables([7u32, 9]);
        let first = stats.table(0).unwrap();
        let second = stats.table(1).unwrap();
        first.hits.fetch_add(41, Ordering::Relaxed);
        first.disk_reads.fetch_add(3, Ordering::Relaxed);
        first.disk_queue_depth.store(128, Ordering::Relaxed);
        first
            .disk_queue_depth_requested
            .store(256, Ordering::Relaxed);
        first.disk_alignment_bytes.store(512, Ordering::Relaxed);
        first.prediction_mismatches.store(2, Ordering::Relaxed);
        second.hits.fetch_add(1, Ordering::Relaxed);
        stats.record_submission(true);
        stats.set_prefetch_enabled(true);
        let mut text = String::new();
        stats.render(&mut text);
        // Aggregate families keep their documented names and sum every table.
        assert!(
            text.contains("flare_engram_row_cache_hits_total 42\n"),
            "aggregate hits\n{text}"
        );
        assert!(text.contains("# TYPE flare_engram_disk_reads_total counter\n"));
        // Each table is addressable by its declared layer.
        assert!(
            text.contains(
                "flare_engram_table_row_cache_hits_total{table=\"0\",layer=\"7\"} 41\n"
            ),
            "labelled first table\n{text}"
        );
        assert!(
            text.contains(
                "flare_engram_table_row_cache_hits_total{table=\"1\",layer=\"9\"} 1\n"
            ),
            "labelled second table\n{text}"
        );
        assert!(
            text.contains(
                "flare_engram_table_disk_queue_depth{table=\"0\",layer=\"7\"} 128\n"
            )
        );
        assert!(
            text.contains(
                "flare_engram_table_disk_queue_depth_requested{table=\"0\",layer=\"7\"} 256\n"
            )
        );
        assert!(
            text.contains(
                "flare_engram_table_disk_alignment_bytes{table=\"0\",layer=\"7\"} 512\n"
            )
        );
        assert!(
            text.contains(
                "flare_engram_table_prediction_mismatches_total{table=\"0\",layer=\"7\"} 2\n"
            )
        );
        assert!(text.contains("flare_engram_prefetch_submissions_total 1\n"));
        assert!(text.contains("flare_engram_prefetch_dropped_submissions_total 0\n"));
        assert!(text.contains("flare_engram_prefetch_enabled 1\n"));
        // The families the loader documentation already promises are still emitted.
        for name in [
            "flare_engram_gathers_total",
            "flare_engram_gather_nanoseconds_total",
            "flare_engram_row_cache_bytes",
            "flare_engram_resident_scale_bytes",
            "flare_engram_disk_plan_nanoseconds_total",
            "flare_engram_disk_wait_nanoseconds_total",
        ] {
            assert!(text.contains(&format!("# TYPE {name} ")), "missing {name}");
        }
        // One HELP and one TYPE line per family, however many samples it carries.
        let families = text
            .lines()
            .filter(|line| line.starts_with("# TYPE "))
            .count();
        let declared = text
            .lines()
            .filter(|line| line.starts_with("# HELP "))
            .count();
        assert_eq!(families, declared);
    }
}
