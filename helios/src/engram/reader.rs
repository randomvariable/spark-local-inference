/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Bounded, direct-I/O row gathering into the same buffers used by resident tables.

use super::{Stats, TableCounters, cache::RowCache};
use crate::Result;
#[cfg(unix)]
use io_uring::{IoUring, opcode, types};
use std::{
    fs::File,
    sync::{Arc, atomic::Ordering},
    time::Instant,
};

#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd};

#[cfg(test)]
use std::path::Path;

#[cfg(windows)]
mod windows;

const BLOCK: usize = 4096;
/// Smallest fragment floor a filesystem may report. An NVMe logical block is
/// 512 bytes, and `statx` never returns a usable read alignment below it.
const GRAN_MIN: usize = 512;
const CHUNK: usize = 64 << 10;
/// Queue depth the Linux reader asks for. Upstream's reader found a 64-slot ring
/// left the device about a sixth full at its measured completion rate, so 256 is
/// the floor this reader raises its depth to; a deeper explicit request is
/// honoured as-is and the depth that actually landed is reported.
const QUEUE_DEPTH_DEFAULT: usize = 256;
/// A ring needs at least two slots to overlap a submission with a completion,
/// and it is where the queue-depth halving ladder stops.
const QUEUE_DEPTH_MIN: usize = 2;
/// Windows readers share this queue and staging budget across workers.
#[cfg(windows)]
const DEPTH: usize = 128;
const LOAD_CHUNK: usize = 4 << 20;
const LOAD_READERS: usize = 4;

/// `IORING_SETUP_*` bits the crate's builder can request, from the kernel uapi
/// header. Cooperative task work keeps completion handling off the
/// inter-processor-interrupt path; the task-run flag only advertises pending
/// task work, which the crate's own reaping does not need but costs nothing.
const IORING_SETUP_COOP_TASKRUN: u32 = 1 << 10;
const IORING_SETUP_TASKRUN_FLAG: u32 = 1 << 11;

/// Setup-flag sets to try, best first. Plain is the fallback for older kernels.
///
/// `IORING_SETUP_SINGLE_ISSUER` and `DEFER_TASKRUN` are deliberately not
/// requested: the kernel binds a single-issuer ring to the first submitting task
/// and answers a second one with `-EEXIST`, which would permanently poison a
/// working reader if any other thread ever submits, and `DEFER_TASKRUN` cannot
/// be used without it.
#[cfg(unix)]
const RING_FLAGS: &[(bool, bool)] = &[(true, true), (true, false), (false, false)];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Plane {
    pub file: usize,
    pub offset: u64,
    pub nbytes: usize,
    pub row_bytes: usize,
}

impl Plane {
    pub(crate) fn resident_bytes(self) -> Result<usize> {
        ((self.offset % BLOCK as u64) as usize)
            .checked_add(self.nbytes)
            .and_then(|n| n.checked_add(BLOCK - 1))
            .map(|n| (n / BLOCK * BLOCK).max(BLOCK))
            .ok_or_else(|| "resident embedding allocation exceeds host range".into())
    }
}

#[derive(Clone, Copy)]
struct Fragment {
    file: usize,
    source: u64,
    destination: usize,
    nbytes: usize,
}

#[derive(Clone)]
struct Read {
    file: usize,
    offset: u64,
    nbytes: usize,
    required: usize,
    fragments: std::ops::Range<usize>,
}

/// Windows staging: the aligned allocation its I/O ring registers. Linux uses
/// [`MappedRegion`] so the region is pre-faulted and huge-page backed.
#[cfg(windows)]
struct Aligned(usize, usize);
#[cfg(windows)]
impl Aligned {
    fn new(nbytes: usize) -> Result<Self> {
        Ok(Self(
            crate::os::allocate_aligned(nbytes).map_err(|e| format!("Engram allocation: {e}"))?,
            nbytes,
        ))
    }
    fn at(&self, slot: usize) -> *mut u8 {
        (self.0 + slot * CHUNK) as *mut u8
    }
}

/// Registered direct-I/O staging for the Linux reader, mapped by the kernel with
/// its page tables already populated rather than taken from the allocator. Every
/// `READ_FIXED` completion writes into this region, so pre-faulting it and
/// backing it with huge pages keeps page faults out of the gather the GPU waits
/// on. Windows keeps the aligned allocation its I/O ring registers directly.
#[cfg(unix)]
struct MappedRegion {
    address: usize,
    nbytes: usize,
}

#[cfg(unix)]
impl MappedRegion {
    fn new(nbytes: usize) -> Result<Self> {
        Ok(Self {
            address: crate::os::map_host(nbytes)
                .map_err(|e| format!("Engram allocation: {e}"))?,
            nbytes,
        })
    }
    fn at(&self, slot: usize) -> *mut u8 {
        (self.address + slot * CHUNK) as *mut u8
    }
}

#[cfg(unix)]
impl Drop for MappedRegion {
    fn drop(&mut self) {
        // SAFETY: the address and length came from `map_host` above, and the
        // ring that registered them is destroyed before this region is unmapped.
        unsafe { crate::os::unmap_host(self.address, self.nbytes) };
    }
}

/// Power-of-two alignment the kernel will accept for a direct read, or zero when
/// the value is unusable. Reads are checked against the device's logical block
/// size, so 512 is the smallest floor worth honouring and anything above the
/// historical split is worse than what the reader did before.
#[cfg(target_os = "linux")]
fn usable_align(candidate: u32) -> usize {
    let value = candidate as usize;
    if (GRAN_MIN..=BLOCK).contains(&value) && value.is_power_of_two() {
        value
    } else {
        0
    }
}

/// Per-file direct-I/O read alignment, from `statx`.
///
/// `iomap` checks a direct read against `bdev_logical_block_size()` only, and
/// `statx` publishes that number per file: ext4 answers with the logical block
/// size, and XFS puts it in the dedicated read field while keeping the
/// allocation-unit hint in the write field. A read path must therefore prefer
/// `STATX_DIO_READ_ALIGN` and fall back to `STATX_DIOALIGN`, which is how the
/// kernel's own direct-I/O consumers treat the pair. Anything the probe cannot
/// answer - no `statx`, a filesystem that leaves both bits unset (btrfs and
/// tmpfs do), an older library or kernel, or a device that reports more than
/// 4 KiB - keeps the historical [`BLOCK`] split.
#[cfg(target_os = "linux")]
fn dio_gran(file: &std::fs::File) -> usize {
    use std::os::fd::AsRawFd;
    // glibc exposes `struct statx` and these fields under `_GNU_SOURCE`; the
    // read-alignment mask is newer than the constant in some headers, so the
    // probe asks for both and trusts only what `stx_mask` reports.
    const STATX_DIO_READ_ALIGN: u32 = 0x8000;
    let mut probe: libc::statx = unsafe { std::mem::zeroed() };
    let ready = libc::STATX_DIOALIGN | STATX_DIO_READ_ALIGN;
    let result = unsafe {
        libc::statx(
            file.as_raw_fd(),
            c"".as_ptr(),
            libc::AT_EMPTY_PATH,
            ready,
            &mut probe,
        )
    };
    if result != 0 {
        return BLOCK;
    }
    let mask = probe.stx_mask;
    if mask & STATX_DIO_READ_ALIGN != 0 {
        let align = usable_align(probe.stx_dio_read_offset_align);
        if align != 0 {
            return align;
        }
    }
    if mask & libc::STATX_DIOALIGN != 0 {
        let align = usable_align(probe.stx_dio_offset_align);
        if align != 0 {
            return align;
        }
    }
    BLOCK
}

/// Non-Linux platforms have no `statx` read-alignment report, so the reader
/// keeps the historical split.
#[cfg(unix)]
#[cfg(not(target_os = "linux"))]
fn dio_gran(_file: &std::fs::File) -> usize {
    BLOCK
}

/// How many `CHUNK`-sized staging slots the locked-memory budget can cover:
/// `io_uring` charges registered buffers against `RLIMIT_MEMLOCK`. This is only
/// an estimate to avoid a doomed first attempt - the limit is per-process and
/// shared with every other ring - so registration remains the authoritative
/// answer and still halves the queue on refusal.
#[cfg(unix)]
fn slot_budget(requested: usize) -> usize {
    let mut limit: libc::rlimit = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut limit) } != 0 {
        return requested;
    }
    if limit.rlim_cur == libc::RLIM_INFINITY {
        return requested;
    }
    let fits = usize::try_from(limit.rlim_cur / CHUNK as u64).unwrap_or(usize::MAX);
    fits.min(requested)
}

#[cfg(test)]
pub(super) fn open_file(path: &Path) -> Result<File> {
    crate::os::direct_file(path, false).map_err(|e| format!("open Engram package: {e}").into())
}

pub(crate) struct Resident {
    #[cfg(unix)]
    storage: ResidentMapping,
    #[cfg(windows)]
    storage: Aligned,
    #[cfg(unix)]
    file: File,
    skip: usize,
    nbytes: usize,
}

#[cfg(unix)]
struct ResidentMapping {
    address: usize,
    nbytes: usize,
}

#[cfg(unix)]
impl ResidentMapping {
    fn new(file: &File, nbytes: usize, writable: bool) -> Result<Self> {
        let protection = libc::PROT_READ | if writable { libc::PROT_WRITE } else { 0 };
        let address = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                nbytes,
                protection,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if address == libc::MAP_FAILED {
            return Err(format!(
                "map resident embedding table: {}",
                std::io::Error::last_os_error()
            )
            .into());
        }
        let mapping = Self {
            address: address as usize,
            nbytes,
        };
        // The sealed read-only remap has its own page tables, even after loading
        // populated the backing RAM. Prepare them before row delivery starts.
        if !writable && unsafe { libc::madvise(address, nbytes, libc::MADV_POPULATE_READ) } != 0 {
            let error = std::io::Error::last_os_error();
            return Err(format!("populate resident embedding page tables: {error}").into());
        }
        Ok(mapping)
    }
}

#[cfg(unix)]
impl Drop for ResidentMapping {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.address as *mut libc::c_void, self.nbytes) };
    }
}

#[cfg(unix)]
const RESIDENT_SEALS: i32 =
    libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;

#[cfg(unix)]
fn seal_resident(file: &File) -> Result<()> {
    // The loading threads have joined and their writable mapping is gone.
    // Kernel-held page references can still make F_SEAL_WRITE temporarily busy.
    // Publish only after every required seal succeeds; persistent busy state fails.
    for attempt in 0..4 {
        if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_ADD_SEALS, RESIDENT_SEALS) } == 0 {
            return Ok(());
        }
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() != Some(libc::EBUSY) || attempt == 3 {
            return Err(format!("seal resident embedding table: {error}").into());
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    unreachable!()
}

impl Resident {
    pub(super) fn load(file: &File, plane: Plane) -> Result<Self> {
        Self::initialize(plane, |address, extent, required| {
            let skip = (plane.offset % BLOCK as u64) as usize;
            let offset = plane.offset - skip as u64;
            std::thread::scope(|scope| -> Result<()> {
                let mut workers = Vec::new();
                for reader in 0..LOAD_READERS.min(extent.div_ceil(LOAD_CHUNK)) {
                    workers.push(
                        std::thread::Builder::new()
                            .name(format!("flare-table-loader-{reader}"))
                            .spawn_scoped(scope, move || -> Result<()> {
                                for begin in (reader * LOAD_CHUNK..extent)
                                    .step_by(LOAD_READERS * LOAD_CHUNK)
                                {
                                    let nbytes = LOAD_CHUNK.min(extent - begin);
                                    let source = offset
                                        .checked_add(begin as u64)
                                        .ok_or("resident Engram file offset exceeds range")?;
                                    // SAFETY: Each reader initializes a disjoint aligned allocation range.
                                    let count = unsafe { crate::os::read_at_raw(file, (address + begin) as *mut u8, nbytes, source) }
                                        .map_err(|e| format!("resident Engram read at {source}: {e}"))?;
                                    let needed = nbytes.min(required - begin);
                                    if count < needed {
                                        return Err(format!(
                                            "resident Engram read at {source} returned {count}, requires {needed} bytes"
                                        ).into());
                                    }
                                }
                                Ok(())
                            })
                            .map_err(|e| format!("start resident Engram loader: {e}"))?,
                    );
                }
                for worker in workers {
                    worker
                        .join()
                        .map_err(|_| "resident Engram loader panicked")??;
                }
                Ok(())
            })
        })
    }

    #[cfg(unix)]
    pub(super) fn load_segments(plane: Plane, segments: &[(File, u64, usize)]) -> Result<Self> {
        if plane.offset != 0 {
            return Err("assembled embedding planes must start at offset zero".into());
        }
        Self::initialize(plane, |address, _, _| {
            let mut destination = 0usize;
            for (file, offset, nbytes) in segments {
                let end = destination
                    .checked_add(*nbytes)
                    .filter(|end| *end <= plane.nbytes)
                    .ok_or("embedding source segments exceed their plane")?;
                let mut copied = 0;
                while copied < *nbytes {
                    let count = (*nbytes - copied).min(LOAD_CHUNK);
                    // SAFETY: Source segments initialize consecutive, disjoint ranges
                    // inside the resident allocation before it becomes immutable.
                    let read = unsafe {
                        crate::os::read_at_raw(
                            file,
                            (address + destination + copied) as *mut u8,
                            count,
                            offset + copied as u64,
                        )
                    }
                    .map_err(|error| format!("read embedding source segment: {error}"))?;
                    if read == 0 {
                        return Err(
                            "embedding source segment ended before its declared size".into()
                        );
                    }
                    copied += read;
                }
                destination = end;
            }
            if destination != plane.nbytes {
                return Err("embedding source segments do not fill their plane".into());
            }
            Ok(())
        })
    }

    fn initialize(
        plane: Plane,
        fill: impl FnOnce(usize, usize, usize) -> Result<()>,
    ) -> Result<Self> {
        let skip = (plane.offset % BLOCK as u64) as usize;
        let required = skip
            .checked_add(plane.nbytes)
            .ok_or("resident Engram plane exceeds host range")?;
        let extent = plane.resident_bytes()?;
        #[cfg(unix)]
        let backing = {
            let descriptor = unsafe {
                libc::memfd_create(
                    c"flare-embedding-table".as_ptr(),
                    libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
                )
            };
            if descriptor < 0 {
                return Err(format!(
                    "create resident embedding table: {}",
                    std::io::Error::last_os_error()
                )
                .into());
            }
            // The descriptor is newly created and owned exclusively until published.
            let backing = unsafe { File::from_raw_fd(descriptor) };
            backing
                .set_len(extent as u64)
                .map_err(|e| format!("size resident embedding table: {e}"))?;
            backing
        };
        #[cfg(unix)]
        let storage = ResidentMapping::new(&backing, extent, true)?;
        #[cfg(unix)]
        let address = storage.address;
        #[cfg(windows)]
        let storage = Aligned::new(extent)?;
        #[cfg(windows)]
        let address = storage.0;
        fill(address, extent, required)?;
        #[cfg(unix)]
        {
            drop(storage);
            seal_resident(&backing)?;
            Self::from_shared(backing, plane)
        }
        #[cfg(windows)]
        {
            Ok(Self {
                storage,
                skip,
                nbytes: plane.nbytes,
            })
        }
    }

    #[cfg(unix)]
    pub(super) fn from_shared(file: File, plane: Plane) -> Result<Self> {
        let seals = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) };
        if seals < 0 || seals & RESIDENT_SEALS != RESIDENT_SEALS {
            return Err("resident embedding table must have immutable size and contents".into());
        }
        let skip = (plane.offset % BLOCK as u64) as usize;
        let extent = plane.resident_bytes()?;
        if file
            .metadata()
            .map_err(|e| format!("inspect resident embedding table: {e}"))?
            .len()
            != extent as u64
        {
            return Err("shared resident embedding table extent differs from its plane".into());
        }
        Ok(Self {
            storage: ResidentMapping::new(&file, extent, false)?,
            file,
            skip,
            nbytes: plane.nbytes,
        })
    }

    #[cfg(unix)]
    pub(super) fn descriptor(&self) -> i32 {
        self.file.as_raw_fd()
    }

    fn bytes(&self) -> &[u8] {
        #[cfg(unix)]
        let address = self.storage.address;
        #[cfg(windows)]
        let address = self.storage.0;
        // SAFETY: Every logical byte was initialized by successful reads, and
        // scoped readers joined before this immutable view becomes available.
        unsafe { std::slice::from_raw_parts((address + self.skip) as *const u8, self.nbytes) }
    }
}
#[cfg(windows)]
impl Drop for Aligned {
    fn drop(&mut self) {
        unsafe { crate::os::free_aligned(self.0, self.1) };
    }
}

pub(crate) struct DiskReader {
    // Destroying the ring cancels and drains outstanding I/O before the buffers are freed.
    #[cfg(unix)]
    ring: IoUring,
    #[cfg(windows)]
    windows: windows::Readers,
    #[cfg(unix)]
    staging: MappedRegion,
    #[cfg(windows)]
    files: Vec<Arc<File>>,
    /// Per-source facts the planner needs: file length and fragment floor.
    #[cfg(unix)]
    meta: Vec<FileMeta>,
    #[cfg(unix)]
    depth: usize,
    /// Slots not currently holding an in-flight read, popped from the back.
    #[cfg(unix)]
    free: Vec<u16>,
    /// Read index each in-flight slot carries.
    #[cfg(unix)]
    slot_job: Vec<usize>,
    fragments: Vec<Fragment>,
    reads: Vec<Read>,
    resident_scales: Option<Resident>,
    counters: Arc<TableCounters>,
}

/// Split each requested row into pieces no wider than the file's fragment floor,
/// zeroing a row the caller marked absent.
///
/// Each piece begins at the offset it ends at, so a read never crosses a boundary
/// the kernel would reject for direct I/O.
fn plan_fragments(
    plane: &Plane,
    ids: &[i64],
    gran: usize,
    base: usize,
    output: &mut [u8],
    fragments: &mut Vec<Fragment>,
) {
    for (row, id) in ids.iter().copied().enumerate() {
        let destination = base + row * plane.row_bytes;
        if id < 0 {
            output[destination..destination + plane.row_bytes].fill(0);
            continue;
        }
        let source = plane.offset + id as u64 * plane.row_bytes as u64;
        let mut copied = 0;
        while copied < plane.row_bytes {
            let address = source + copied as u64;
            let nbytes = (plane.row_bytes - copied).min(gran - address as usize % gran);
            fragments.push(Fragment {
                file: plane.file,
                source: address,
                destination: destination + copied,
                nbytes,
            });
            copied += nbytes;
        }
    }
}

/// Merge runs of fragments that occupy consecutive blocks of one file into a
/// single read, up to one staging slot.
///
/// The fragments must be sorted by file and source. The sort exists to make
/// adjacent blocks mergeable; byte results are identical whether or not runs
/// merge, only the request count changes. Returns the distinct block count and
/// how many reads cover more than one block.
fn plan_reads(
    fragments: &[Fragment],
    reads: &mut Vec<Read>,
    gran: impl Fn(usize) -> u64,
    size: impl Fn(usize) -> u64,
) -> (u64, u64) {
    let mut cursor = 0;
    let mut blocks = 0u64;
    let mut coalesced = 0u64;
    while cursor < fragments.len() {
        let begin = cursor;
        let file = fragments[begin].file;
        let floor = gran(file);
        let first = fragments[begin].source / floor * floor;
        let mut last = first;
        let mut required = 0usize;
        while cursor < fragments.len() {
            let fragment = fragments[cursor];
            if fragment.file != file {
                break;
            }
            let block = fragment.source / floor * floor;
            if block != last {
                if block < last || block - last != floor || block - first >= CHUNK as u64 {
                    break;
                }
                last = block;
            }
            required = required.max((fragment.source - first) as usize + fragment.nbytes);
            cursor += 1;
        }
        blocks += (last - first) / floor + 1;
        if last != first {
            coalesced += 1;
        }
        let aligned = (last - first) as usize + floor as usize;
        let available = size(file).saturating_sub(first) as usize;
        reads.push(Read {
            file,
            offset: first,
            nbytes: aligned.min(available),
            required,
            fragments: begin..cursor,
        });
    }
    (blocks, coalesced)
}

/// The queue depth the reader starts at: the requested depth raised to the
/// default, clamped by what the locked-memory budget can cover, and never below
/// the two slots a submission needs to overlap a completion.
#[cfg(unix)]
fn start_depth(requested: usize, budget: usize) -> usize {
    requested.min(budget).max(QUEUE_DEPTH_MIN)
}

/// The next rung of the halving ladder, or `None` once it has bottomed out.
#[cfg(unix)]
fn shallower(depth: usize) -> Option<usize> {
    (depth > QUEUE_DEPTH_MIN).then(|| (depth / 2).max(QUEUE_DEPTH_MIN))
}

/// Bytes the reader's bounded planning tables can reach for one gather.
///
/// Each row splits into at most `row_bytes / GRAN_MIN + 2` fragments at the
/// smallest floor `statx` may report, and each run of contiguous fragments into
/// one read. These tables belong to the disk reader rather than to the row
/// cache, so the loader reserves them alongside the staging frames instead of
/// shrinking a configured cache budget.
pub(crate) fn planning_bytes(planes: &[Plane; 2], rows: usize) -> usize {
    let per_row = planes
        .iter()
        .map(|plane| plane.row_bytes / GRAN_MIN + 2)
        .sum::<usize>();
    rows.saturating_mul(per_row.saturating_mul(size_of::<Fragment>() + size_of::<Read>()))
}

/// One source file's planning facts, probed once when the reader opens.
#[cfg(unix)]
#[derive(Clone)]
struct FileMeta {
    bytes: u64,
    gran: usize,
}

impl DiskReader {
    /// Fragment floor for one source: its reported direct-I/O read alignment on
    /// Linux, and the historical 4 KiB split everywhere else.
    fn gran(&self, file: usize) -> usize {
        #[cfg(unix)]
        return self.meta[file].gran;
        #[cfg(not(unix))]
        {
            let _ = file;
            BLOCK
        }
    }

    /// Length of one source, used to clamp the tail of a coalesced read to
    /// bytes that exist. Windows keeps the previous unclamped request length.
    fn file_bytes(&self, file: usize) -> u64 {
        #[cfg(unix)]
        return self.meta[file].bytes;
        #[cfg(not(unix))]
        {
            let _ = file;
            u64::MAX
        }
    }
}

// Accumulate locally so timing does not add atomic operations to each completion.
// Drop also records completed phases when a gather exits with an I/O error.
struct DiskTimings<'a> {
    stats: &'a TableCounters,
    plan: u64,
    submit: u64,
    wait: u64,
    complete: u64,
}

impl Drop for DiskTimings<'_> {
    fn drop(&mut self) {
        for (counter, value) in [
            (&self.stats.disk_plan_ns, self.plan),
            (&self.stats.disk_submit_ns, self.submit),
            (&self.stats.disk_wait_ns, self.wait),
            (&self.stats.disk_complete_ns, self.complete),
        ] {
            counter.fetch_add(value, Ordering::Relaxed);
        }
    }
}

impl DiskReader {
    fn from_files(files: &[Arc<File>], counters: Arc<TableCounters>, workers: usize) -> Result<Self> {
        #[cfg(unix)]
        if workers != 1 {
            return Err("parallel embedding disk workers require Windows".into());
        }
        #[cfg(unix)]
        {
            // One probe per source: the file length clamps a coalesced read to
            // real bytes, and the direct-I/O alignment sets the fragment floor.
            let meta = files
                .iter()
                .map(|file| {
                    let bytes = file
                        .metadata()
                        .map_err(|e| format!("stat Engram table: {e}"))?
                        .len();
                    Ok(FileMeta { bytes, gran: dio_gran(file) })
                })
                .collect::<Result<Vec<_>>>()?;
            let requested = QUEUE_DEPTH_DEFAULT;
            let mut depth = start_depth(requested, slot_budget(requested));
            // Registered buffers are pinned and charged against the locked-memory
            // limit, which is a per-process budget shared with every other ring,
            // so registration is the only authoritative answer: on refusal, halve
            // the queue rather than failing the reader. A shallower ring is slower
            // but correct, and the reported depth is the one that landed.
            loop {
                match Self::configure(depth, requested, files, meta.clone(), counters.clone()) {
                    Ok(reader) => return Ok(reader),
                    // At the floor the refusal is fatal, and the caller reports it.
                    Err(error) => depth = shallower(depth).ok_or(error)?,
                }
            }
        }
        #[cfg(windows)]
        {
            let windows = windows::Readers::new(files, workers)?;
            Ok(Self {
                windows,
                files: files.to_vec(),
                fragments: Vec::new(),
                reads: Vec::new(),
                resident_scales: None,
                counters,
            })
        }
    }

    /// Allocate and register everything one queue depth needs, or fail without
    /// leaving a ring, pinned buffers, or a dangling mapping behind, so the
    /// caller can simply retry at half the slots.
    #[cfg(unix)]
    fn configure(
        depth: usize,
        depth_requested: usize,
        files: &[Arc<File>],
        meta: Vec<FileMeta>,
        counters: Arc<TableCounters>,
    ) -> Result<Self> {
        let staging = MappedRegion::new(depth * CHUNK)?;
        let mut ring = None;
        let mut setup_bits = 0u32;
        for &(coop, taskrun_flag) in RING_FLAGS {
            let mut builder = IoUring::builder();
            if coop {
                builder.setup_coop_taskrun();
            }
            if taskrun_flag {
                builder.setup_taskrun_flag();
            }
            match builder.build(depth as u32) {
                Ok(candidate) => {
                    ring = Some(candidate);
                    setup_bits = (u32::from(coop) * IORING_SETUP_COOP_TASKRUN)
                        | (u32::from(taskrun_flag) * IORING_SETUP_TASKRUN_FLAG);
                    break;
                }
                // Only a rejected setup request steps down a rung; a resource
                // error is the same error for every rung.
                Err(_) => continue,
            }
        }
        let ring = ring.ok_or("create Engram io_uring: no supported setup flags")?;
        let descriptors: Vec<_> = (0..depth)
            .map(|slot| libc::iovec {
                iov_base: staging.at(slot).cast(),
                iov_len: CHUNK,
            })
            .collect();
        // SAFETY: every descriptor addresses CHUNK bytes of `staging`, which
        // outlives the ring: the struct declares staging after ring, so the
        // kernel's registered buffers are released before the mapping is unmapped.
        if let Err(error) = unsafe { ring.submitter().register_buffers(&descriptors) } {
            return Err(format!("register Engram staging: {error}").into());
        }
        // Registered files let each read skip the descriptor-table lookup, and the
        // crate sets IOSQE_FIXED_FILE for a `types::Fixed` operand. A kernel or
        // container policy that refuses file registration is not fatal: the reader
        // keeps submitting raw descriptors, and the gauge records which path landed.
        let raw: Vec<_> = files.iter().map(|file| file.as_raw_fd()).collect();
        let fixed_files = ring.submitter().register_files(&raw).is_ok();
        // Every value here describes the ring that finally came up, so a halving
        // retry must overwrite rather than accumulate.
        counters
            .disk_staging_bytes
            .store(depth as u64 * CHUNK as u64, Ordering::Relaxed);
        counters
            .disk_queue_depth
            .store(depth as u64, Ordering::Relaxed);
        counters
            .disk_queue_depth_requested
            .store(depth_requested as u64, Ordering::Relaxed);
        counters.disk_alignment_bytes.store(
            meta.iter().map(|entry| entry.gran).min().unwrap_or(BLOCK) as u64,
            Ordering::Relaxed,
        );
        counters
            .disk_uring_flags
            .store(u64::from(setup_bits), Ordering::Relaxed);
        counters
            .disk_fixed_files
            .store(u64::from(fixed_files), Ordering::Relaxed);
        Ok(Self {
            ring,
            staging,
            meta,
            depth,
            free: (0..depth as u16).rev().collect(),
            slot_job: vec![0; depth],
            fragments: Vec::new(),
            reads: Vec::new(),
            resident_scales: None,
            counters,
        })
    }

    pub fn gather(&mut self, planes: &[Plane; 2], ids: &[i64], output: &mut [u8]) -> Result<()> {
        let mut timing = DiskTimings {
            stats: &self.counters,
            plan: 0,
            submit: 0,
            wait: 0,
            complete: 0,
        };
        let begin = Instant::now();
        let mut fragments = std::mem::take(&mut self.fragments);
        fragments.clear();
        let mut base = 0;
        for (ordinal, plane) in planes.iter().enumerate() {
            if ordinal == 1 {
                if let Some(scales) = &self.resident_scales {
                    gather_resident(scales, plane.row_bytes, ids, &mut output[base..]);
                    base += ids.len() * plane.row_bytes;
                    continue;
                }
            }
            let gran = self.gran(plane.file);
            plan_fragments(plane, ids, gran, base, output, &mut fragments);
            base += ids.len() * plane.row_bytes;
        }
        fragments.sort_unstable_by_key(|f| (f.file, f.source));
        let mut reads = std::mem::take(&mut self.reads);
        reads.clear();
        let (blocks, coalesced) = plan_reads(
            &fragments,
            &mut reads,
            |file| self.gran(file) as u64,
            |file| self.file_bytes(file),
        );
        self.fragments = fragments;
        self.reads = reads;
        self.counters
            .disk_unique_blocks
            .fetch_add(blocks, Ordering::Relaxed);
        self.counters
            .disk_coalesced_reads
            .fetch_add(coalesced, Ordering::Relaxed);
        timing.plan = begin.elapsed().as_nanos() as u64;
        self.counters
            .disk_reads
            .fetch_add(self.reads.len() as u64, Ordering::Relaxed);
        self.counters.disk_bytes.fetch_add(
            self.reads.iter().map(|read| read.nbytes as u64).sum(),
            Ordering::Relaxed,
        );
        #[cfg(unix)]
        {
            // Every slot is free at the start of a gather: this thread owns the ring,
            // and a previous gather either reaped each read it armed or returned an
            // error that cancelled the table. Re-issuing the list keeps a failed
            // gather from leaking slots into later ones.
            self.free.clear();
            self.free.extend((0..self.depth as u16).rev());
            let mut armed = 0usize;
            let mut next = 0usize;
            let mut failure: Option<crate::Error> = None;
            while next < self.reads.len() || armed != 0 {
                // Refill from every slot the previous reap handed back, so the
                // queue stays saturated across the wave instead of one
                // completion at a time. Once a read failed, stop arming: an I/O
                // error then leaves no speculative refill entries, and the
                // already-submitted reads are still drained before returning.
                let mut prepared = 0usize;
                while failure.is_none() && next < self.reads.len() && !self.free.is_empty() {
                    let slot = self.free.pop().expect("checked non-empty");
                    let read = self.reads[next].clone();
                    self.slot_job[slot as usize] = next;
                    next += 1;
                    let entry = opcode::ReadFixed::new(
                        types::Fixed(read.file as u32),
                        self.staging.at(slot as usize),
                        read.nbytes as u32,
                        slot,
                    )
                    .offset(read.offset)
                    .build()
                    .user_data(slot as u64);
                    // SAFETY: The ring has `depth` entries and at most `depth`
                    // slots are handed out, so the submission queue has room.
                    match unsafe { self.ring.submission().push(&entry) } {
                        Ok(()) => prepared += 1,
                        Err(_) => failure = Some("Engram submission queue full".into()),
                    }
                }
                armed += prepared;
                if armed == 0 {
                    break;
                }
                // One enter submits the whole armed wave and blocks until half
                // the ring is ready to refill, replacing wait-one-CQE granularity
                // that left the device idle between reaps.
                let begin = Instant::now();
                let wait = armed.min(self.depth / 2).max(1);
                let submitted = self.ring.submit_and_wait(wait);
                timing.wait += begin.elapsed().as_nanos() as u64;
                self.counters
                    .disk_submit_calls
                    .fetch_add(1, Ordering::Relaxed);
                match submitted {
                    Ok(_) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(error) => {
                        return Err(format!("submit Engram reads: {error}").into());
                    }
                }
                let begin = Instant::now();
                {
                    let mut completion = self.ring.completion();
                    while armed != 0 {
                        let Some(entry) = completion.next() else { break };
                        armed -= 1;
                        let slot = entry.user_data() as usize;
                        self.free.push(slot as u16);
                        let read = self.reads[self.slot_job[slot]].clone();
                        self.counters
                            .disk_read_calls
                            .fetch_add(1, Ordering::Relaxed);
                        let count = entry.result();
                        if count < 0 || (count as usize) < read.required {
                            if failure.is_none() {
                                failure = Some(
                                    format!(
                                        "Engram read at {} returned {count}, requires {} bytes",
                                        read.offset, read.required
                                    )
                                    .into(),
                                );
                            }
                            continue;
                        }
                        self.counters
                            .disk_read_bytes
                            .fetch_add(count as u64, Ordering::Relaxed);
                        for fragment in &self.fragments[read.fragments.clone()] {
                            let source = unsafe {
                                std::slice::from_raw_parts(
                                    self.staging
                                        .at(slot)
                                        .add((fragment.source - read.offset) as usize),
                                    fragment.nbytes,
                                )
                            };
                            output[fragment.destination..fragment.destination + fragment.nbytes]
                                .copy_from_slice(source);
                        }
                    }
                }
                timing.complete += begin.elapsed().as_nanos() as u64;
            }
            if let Some(error) = failure {
                return Err(error);
            }
        }
        #[cfg(windows)]
        self.windows
            .gather(&self.reads, &self.fragments, output, &self.counters)?;

        Ok(())
    }
}

struct CachedReader {
    disk: DiskReader,
    cache: RowCache,
    requests: Vec<(i64, usize)>,
    miss_ids: Vec<i64>,
    miss_groups: Vec<(usize, usize)>,
    miss_output: Vec<u8>,
    max_rows: usize,
}

impl CachedReader {
    fn new(
        disk: DiskReader,
        planes: &[Plane; 2],
        budget: usize,
        max_rows: usize,
    ) -> Result<Self> {
        if max_rows == 0 {
            return Err("embedding row cache requires a nonzero gather capacity".into());
        }
        let row_bytes = planes.iter().map(|plane| plane.row_bytes).sum::<usize>();
        let workspace = max_rows
            .checked_mul(
                size_of::<(i64, usize)>()
                    + size_of::<i64>()
                    + size_of::<(usize, usize)>()
                    + row_bytes,
            )
            .and_then(|n| n.checked_add(size_of::<Self>()))
            .ok_or("embedding row cache workspace size overflow")?;
        let payload_budget = budget
            .checked_sub(workspace)
            .ok_or("embedding row cache budget cannot hold its bounded miss workspace")?;
        fn allocated<T: Copy>(len: usize, value: T) -> Result<Vec<T>> {
            let mut data = Vec::new();
            data.try_reserve_exact(len)
                .map_err(|e| format!("embedding row cache workspace: {e}"))?;
            data.resize(len, value);
            // Touch the entire workspace before KV sizing, including vectors
            // whose length is cleared after allocation.
            for index in (0..len).step_by((4096 / size_of::<T>()).max(1)) {
                unsafe { data.as_mut_ptr().add(index).write_volatile(data[index]) };
            }
            Ok(data)
        }
        let mut requests = allocated(max_rows, (0i64, 0usize))?;
        let mut miss_ids = allocated(max_rows, 0i64)?;
        let mut miss_groups = allocated(max_rows, (0usize, 0usize))?;
        requests.clear();
        miss_ids.clear();
        miss_groups.clear();
        let miss_output = allocated(max_rows * row_bytes, 0u8)?;
        let cache = RowCache::new(
            payload_budget,
            planes[0].nbytes / planes[0].row_bytes,
            row_bytes,
        )?;
        let bytes = cache.bytes() + workspace;
        disk.counters
            .capacity_rows
            .fetch_add(cache.capacity() as u64, Ordering::Relaxed);
        disk.counters
            .cache_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(Self {
            disk,
            cache,
            requests,
            miss_ids,
            miss_groups,
            miss_output,
            max_rows,
        })
    }

    fn gather(&mut self, planes: &[Plane; 2], ids: &[i64], output: &mut [u8]) -> Result<()> {
        if ids.len() > self.max_rows {
            return Err("embedding gather exceeds the row cache workspace capacity".into());
        }
        let [weights, scales] = planes.map(|plane| plane.row_bytes);
        let scale_base = ids.len() * weights;
        self.requests.clear();
        self.miss_ids.clear();
        self.miss_groups.clear();
        for (row, &id) in ids.iter().enumerate() {
            if id < 0 {
                output[row * weights..(row + 1) * weights].fill(0);
                output[scale_base + row * scales..scale_base + (row + 1) * scales].fill(0);
            } else {
                self.requests.push((id, row));
            }
        }
        self.requests.sort_unstable_by_key(|&(id, _)| id);
        let mut begin = 0;
        let mut hits = 0;
        while begin < self.requests.len() {
            let id = self.requests[begin].0;
            let end = begin + self.requests[begin..].partition_point(|&(next, _)| next == id);
            if let Some(payload) = self.cache.get(id) {
                hits += 1;
                for &(_, row) in &self.requests[begin..end] {
                    output[row * weights..(row + 1) * weights].copy_from_slice(&payload[..weights]);
                    output[scale_base + row * scales..scale_base + (row + 1) * scales]
                        .copy_from_slice(&payload[weights..]);
                }
            } else {
                self.miss_ids.push(id);
                self.miss_groups.push((begin, end));
            }
            begin = end;
        }
        self.disk.counters.hits.fetch_add(hits, Ordering::Relaxed);
        self.disk
            .counters
            .misses
            .fetch_add(self.miss_ids.len() as u64, Ordering::Relaxed);
        if self.miss_ids.is_empty() {
            return Ok(());
        }
        let count = self.miss_ids.len();
        self.disk.gather(
            planes,
            &self.miss_ids,
            &mut self.miss_output[..count * (weights + scales)],
        )?;
        for (index, (&id, &(begin, end))) in self.miss_ids.iter().zip(&self.miss_groups).enumerate()
        {
            let w = &self.miss_output[index * weights..(index + 1) * weights];
            let s = &self.miss_output
                [count * weights + index * scales..count * weights + (index + 1) * scales];
            for &(_, row) in &self.requests[begin..end] {
                output[row * weights..(row + 1) * weights].copy_from_slice(w);
                output[scale_base + row * scales..scale_base + (row + 1) * scales]
                    .copy_from_slice(s);
            }
            // Only complete successful disk gathers enter the immutable cache.
            let inserted = self.cache.insert(id, w, s);
            self.disk
                .counters
                .resident_rows
                .fetch_add(u64::from(inserted.added), Ordering::Relaxed);
            self.disk
                .counters
                .evictions
                .fetch_add(u64::from(inserted.evicted), Ordering::Relaxed);
        }
        Ok(())
    }

    /// Warm the row cache with predicted ids, without a staging frame.
    ///
    /// Advisory by contract: an id that is negative, outside the table, already
    /// resident, or a repeat of another id in the same call is dropped without
    /// an error, and a failed read is returned for the caller to count rather
    /// than disabling the cache. Upstream makes a failed miss fetch terminal
    /// because its cache is the only source of truth for those bytes; here the
    /// real gather still reads them and reports the same condition, so losing
    /// cache warmth must not be allowed to poison a frame or stop serving.
    fn prefetch(&mut self, planes: &[Plane; 2], ids: &[i64]) -> Result<usize> {
        if ids.is_empty() {
            return Ok(0);
        }
        let [weights, scales] = planes.map(|plane| plane.row_bytes);
        let rows = planes[0].nbytes / planes[0].row_bytes;
        self.miss_ids.clear();
        self.miss_ids.extend(
            ids.iter()
                .copied()
                .filter(|&id| id >= 0 && (id as usize) < rows),
        );
        self.miss_ids.sort_unstable();
        self.miss_ids.dedup();
        self.miss_ids.retain(|&id| self.cache.get(id).is_none());
        if self.miss_ids.len() > self.max_rows {
            self.disk
                .counters
                .prefetch_rows_dropped
                .fetch_add((self.miss_ids.len() - self.max_rows) as u64, Ordering::Relaxed);
            self.miss_ids.truncate(self.max_rows);
        }
        let count = self.miss_ids.len();
        if count == 0 {
            return Ok(0);
        }
        self.disk.gather(
            planes,
            &self.miss_ids,
            &mut self.miss_output[..count * (weights + scales)],
        )?;
        let mut admitted = 0usize;
        for index in 0..count {
            let id = self.miss_ids[index];
            let w = &self.miss_output[index * weights..(index + 1) * weights];
            let s = &self.miss_output
                [count * weights + index * scales..count * weights + (index + 1) * scales];
            // Only complete successful reads enter the immutable cache.
            let inserted = self.cache.insert(id, w, s);
            admitted += usize::from(inserted.added);
            self.disk
                .counters
                .resident_rows
                .fetch_add(u64::from(inserted.added), Ordering::Relaxed);
            self.disk
                .counters
                .evictions
                .fetch_add(u64::from(inserted.evicted), Ordering::Relaxed);
        }
        self.disk
            .counters
            .prefetch_rows_loaded
            .fetch_add(admitted as u64, Ordering::Relaxed);
        Ok(admitted)
    }
}

enum Data {
    Disk(DiskReader),
    Ram([Arc<Resident>; 2]),
    Cached(Box<CachedReader>),
}

fn gather_resident(data: &Resident, row_bytes: usize, ids: &[i64], output: &mut [u8]) {
    let data = data.bytes();
    for (row, &id) in ids.iter().enumerate() {
        let target = &mut output[row * row_bytes..(row + 1) * row_bytes];
        if id < 0 {
            target.fill(0);
        } else {
            let start = id as usize * row_bytes;
            target.copy_from_slice(&data[start..start + row_bytes]);
        }
    }
}
pub(crate) struct Reader {
    data: Data,
    stats: Arc<Stats>,
}
impl Reader {
    #[cfg(test)]
    fn open(
        path: &Path,
        planes: &[Plane; 2],
        resident: bool,
        cache_bytes: usize,
        max_rows: usize,
        stats: Arc<Stats>,
        counters: Arc<TableCounters>,
    ) -> Result<Self> {
        Self::from_files(
            &[Arc::new(open_file(path)?)],
            planes,
            resident,
            cache_bytes,
            max_rows,
            stats,
            counters,
            1,
            false,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn from_files(
        files: &[Arc<File>],
        planes: &[Plane; 2],
        resident: bool,
        cache_bytes: usize,
        max_rows: usize,
        stats: Arc<Stats>,
        counters: Arc<TableCounters>,
        workers: usize,
        resident_scales: bool,
    ) -> Result<Self> {
        if resident && cache_bytes != 0 {
            return Err("embedding row cache requires disk table storage".into());
        }
        if planes.iter().any(|plane| plane.file >= files.len()) {
            return Err("embedding plane references an unknown file".into());
        }
        let data = if resident {
            Data::Ram([
                Arc::new(Resident::load(&files[planes[0].file], planes[0])?),
                Arc::new(Resident::load(&files[planes[1].file], planes[1])?),
            ])
        } else {
            let mut disk = DiskReader::from_files(files, counters.clone(), workers)?;
            if resident_scales {
                disk.resident_scales = Some(Resident::load(&files[planes[1].file], planes[1])?);
                counters
                    .resident_scale_bytes
                    .fetch_add(planes[1].resident_bytes()? as u64, Ordering::Relaxed);
            }
            if cache_bytes != 0 {
                Data::Cached(Box::new(CachedReader::new(
                    disk,
                    planes,
                    cache_bytes,
                    max_rows,
                )?))
            } else {
                Data::Disk(disk)
            }
        };
        Ok(Self { data, stats })
    }
    #[cfg(unix)]
    pub(super) fn resident(data: [Arc<Resident>; 2], stats: Arc<Stats>) -> Self {
        Self {
            data: Data::Ram(data),
            stats,
        }
    }
    pub fn gather(&mut self, planes: &[Plane; 2], ids: &[i64], output: &mut [u8]) -> Result<()> {
        if output.len() != ids.len() * planes.iter().map(|p| p.row_bytes).sum::<usize>() {
            return Err("Engram staging size differs from row count".into());
        }
        for plane in planes {
            if ids
                .iter()
                .any(|id| *id < -1 || (*id >= 0 && *id as usize >= plane.nbytes / plane.row_bytes))
            {
                return Err("Engram row exceeds its table".into());
            }
        }
        let begin = Instant::now();
        let result = match &mut self.data {
            Data::Disk(reader) => reader.gather(planes, ids, output),
            Data::Cached(reader) => reader.gather(planes, ids, output),
            Data::Ram(data) => {
                let mut base = 0;
                for (plane, data) in planes.iter().zip(data) {
                    gather_resident(data, plane.row_bytes, ids, &mut output[base..]);
                    base += ids.len() * plane.row_bytes;
                }
                Ok(())
            }
        };
        self.stats.gathers.fetch_add(1, Ordering::Relaxed);
        self.stats.gather_ns.fetch_add(
            begin.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        result
    }

    /// Warm the row cache with predicted ids. Returns the rows newly resident.
    ///
    /// A reader without a row cache has nothing to warm: disk storage reads the
    /// same bytes again, and RAM storage never touches the disk. That is the
    /// same gate upstream applies when its cache size is unset, so an advisory
    /// call against such a reader is a no-op rather than an error.
    pub fn prefetch(&mut self, planes: &[Plane; 2], ids: &[i64]) -> Result<usize> {
        match &mut self.data {
            Data::Cached(reader) => reader.prefetch(planes, ids),
            Data::Disk(_) | Data::Ram(_) => Ok(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;

    #[test]
    fn resident_scales_preserve_bytes_and_cache_hits_without_reading_the_scale_file() {
        for (weight_row_bytes, scale_row_bytes) in [(256, 8), (80, 10)] {
            check_resident_scale_rows(weight_row_bytes, scale_row_bytes);
        }
    }

    fn check_resident_scale_rows(weight_row_bytes: usize, scale_row_bytes: usize) {
        let root = std::env::temp_dir().join(format!("helios-engram-scales-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let rows = 4097;
        let planes = [
            Plane { file: 0, offset: 4093, nbytes: rows * weight_row_bytes, row_bytes: weight_row_bytes },
            Plane { file: 1, offset: 4093, nbytes: rows * scale_row_bytes, row_bytes: scale_row_bytes },
        ];
        let row_bytes = weight_row_bytes + scale_row_bytes;
        let data: Vec<Vec<u8>> = planes.iter().map(|plane| {
            (0..plane.offset as usize + plane.nbytes).map(|i| (i * 19 + i / 257) as u8).collect()
        }).collect();
        for cache in [0, 65536] {
            let mut files = Vec::new();
            for (i, bytes) in data.iter().enumerate() {
                let path = root.join(i.to_string());
                std::fs::write(&path, bytes).unwrap();
                files.push(Arc::new(open_file(&path).unwrap()));
            }
            let stats = Arc::new(Stats::default());
            let counters = Arc::new(TableCounters::default());
            let mut reader = Reader::from_files(&files, &planes, false, cache, 8, stats.clone(), counters.clone(), 1, true).unwrap();
            assert_eq!(counters.resident_scale_bytes.load(Ordering::Relaxed), planes[1].resident_bytes().unwrap() as u64);
            OpenOptions::new().write(true).open(root.join("1")).unwrap().set_len(0).unwrap();
            for ids in [vec![0, 4096, -1, 0], vec![4096, 0, -1, 0], vec![17, -1, 19]] {
                let mut expected = Vec::new();
                for (plane, bytes) in planes.iter().zip(&data) {
                    for &id in &ids {
                        if id < 0 {
                            expected.extend(std::iter::repeat_n(0, plane.row_bytes));
                        } else {
                            let start = plane.offset as usize + id as usize * plane.row_bytes;
                            expected.extend_from_slice(&bytes[start..start + plane.row_bytes]);
                        }
                    }
                }
                let mut output = vec![0xa5; expected.len()];
                reader.gather(&planes, &ids, &mut output).unwrap();
                assert_eq!(output, expected);
            }
            assert!(counters.disk_reads.load(Ordering::Relaxed) > 0);
            if cache > 0 {
                assert_eq!(counters.hits.load(Ordering::Relaxed), 2);
                assert_eq!(counters.misses.load(Ordering::Relaxed), 4);
                let reads = counters.disk_reads.load(Ordering::Relaxed);
                reader.gather(&planes, &[17, 19], &mut vec![0; row_bytes * 2]).unwrap();
                assert_eq!(counters.disk_reads.load(Ordering::Relaxed), reads);
            }
            assert!(reader.gather(&planes, &[rows as i64], &mut vec![0; row_bytes]).is_err());
            assert!(reader.gather(&planes, &[-2], &mut vec![0; row_bytes]).is_err());
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(target_os = "linux")]
    fn unsealed_resident_file() -> File {
        let descriptor = unsafe {
            libc::memfd_create(
                c"helios-resident-seal-test".as_ptr(),
                libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
            )
        };
        assert!(descriptor >= 0);
        let file = unsafe { File::from_raw_fd(descriptor) };
        file.set_len(BLOCK as u64).unwrap();
        file
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn resident_sealing_waits_for_unmapped_pages_held_by_io() {
        use std::io::Read;
        use std::sync::mpsc::{self, RecvTimeoutError};
        use std::time::Duration;

        let file = unsealed_resident_file();
        let mapping = ResidentMapping::new(&file, BLOCK, true).unwrap();
        unsafe { std::ptr::write_bytes(mapping.address as *mut u8, 0x39, BLOCK) };
        let mut descriptors = [-1; 2];
        assert_eq!(
            unsafe { libc::pipe2(descriptors.as_mut_ptr(), libc::O_CLOEXEC) },
            0
        );
        let mut reader = unsafe { File::from_raw_fd(descriptors[0]) };
        let writer = unsafe { File::from_raw_fd(descriptors[1]) };
        let vector = libc::iovec {
            iov_base: mapping.address as *mut libc::c_void,
            iov_len: BLOCK,
        };
        assert_eq!(
            unsafe { libc::vmsplice(writer.as_raw_fd(), &vector, 1, 0) },
            BLOCK as isize
        );
        drop(mapping);
        assert_eq!(
            unsafe { libc::fcntl(file.as_raw_fd(), libc::F_ADD_SEALS, RESIDENT_SEALS) },
            -1
        );
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::EBUSY)
        );

        std::thread::scope(|scope| {
            let (send, receive) = mpsc::channel();
            let file = &file;
            scope.spawn(move || send.send(seal_resident(file)).unwrap());
            // The I/O reference remains live beyond the kernel's initial pin wait.
            assert!(matches!(
                receive.recv_timeout(Duration::from_millis(250)),
                Err(RecvTimeoutError::Timeout)
            ));
            let mut bytes = [0; BLOCK];
            reader.read_exact(&mut bytes).unwrap();
            assert_eq!(bytes, [0x39; BLOCK]);
            receive.recv_timeout(Duration::from_secs(2)).unwrap().unwrap();
        });
        assert_eq!(
            unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) } & RESIDENT_SEALS,
            RESIDENT_SEALS
        );
        assert!(file.set_len(1).is_err());
        let value = 0u8;
        assert_eq!(
            unsafe { libc::pwrite(file.as_raw_fd(), (&value as *const u8).cast(), 1, 0) },
            -1
        );
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::EPERM)
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn resident_sealing_rejects_a_retained_writable_mapping() {
        let file = unsealed_resident_file();
        let initial_seals = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) };
        let mapping = ResidentMapping::new(&file, BLOCK, true).unwrap();
        assert!(seal_resident(&file).is_err());
        assert_eq!(
            unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) },
            initial_seals
        );
        drop(mapping);
        seal_resident(&file).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn shared_resident_rows_are_readable_without_first_access_page_faults() {
        let path = std::env::temp_dir().join(format!(
            "helios-engram-populated-{}.bin",
            std::process::id()
        ));
        let rows = 65536;
        let second = rows * 256;
        let bytes: Vec<_> = (0..second + rows * 8)
            .map(|i| (i * 19 + i / 7) as u8)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let planes = [
            Plane {
                file: 0,
                offset: 0,
                nbytes: second,
                row_bytes: 256,
            },
            Plane {
                file: 0,
                offset: second as u64,
                nbytes: rows * 8,
                row_bytes: 8,
            },
        ];
        let file = open_file(&path).unwrap();
        let original = planes.map(|plane| Arc::new(Resident::load(&file, plane).unwrap()));
        let shared = std::array::from_fn(|i| {
            Arc::new(
                Resident::from_shared(original[i].file.try_clone().unwrap(), planes[i]).unwrap(),
            )
        });
        std::fs::remove_file(path).unwrap();
        let ids: Vec<_> = (0..128).map(|i| ((i * 509 + 17) % rows) as i64).collect();
        let mut output = vec![0; ids.len() * 264];
        let mut expected = Vec::with_capacity(output.len());
        for plane in &planes {
            for &id in &ids {
                let first = plane.offset as usize + id as usize * plane.row_bytes;
                expected.extend_from_slice(&bytes[first..first + plane.row_bytes]);
            }
        }
        for data in [original, shared] {
            let mut reader = Reader::resident(data, Arc::default());
            // Prime the copy code and output storage without touching the selected table pages.
            reader.gather(&planes, &[0], &mut output[..264]).unwrap();
            output.fill(0xA5);
            let mut before = unsafe { std::mem::zeroed::<libc::rusage>() };
            let mut after = unsafe { std::mem::zeroed::<libc::rusage>() };
            assert_eq!(
                unsafe { libc::getrusage(libc::RUSAGE_THREAD, &mut before) },
                0
            );
            reader.gather(&planes, &ids, &mut output).unwrap();
            assert_eq!(
                unsafe { libc::getrusage(libc::RUSAGE_THREAD, &mut after) },
                0
            );
            assert_eq!(
                after.ru_minflt, before.ru_minflt,
                "row gathering faulted resident pages"
            );
            assert_eq!(after.ru_majflt, before.ru_majflt);
            assert_eq!(output, expected);
        }
    }

    #[test]
    #[cfg(unix)]
    fn shared_resident_plane_outlives_source_and_loader_and_rejects_mutation() {
        let path =
            std::env::temp_dir().join(format!("helios-engram-shared-{}.bin", std::process::id()));
        let bytes: Vec<_> = (0..12345).map(|i| (i * 19 + i / 7) as u8).collect();
        std::fs::write(&path, &bytes).unwrap();
        let plane = Plane {
            file: 0,
            offset: 97,
            nbytes: 8192,
            row_bytes: 128,
        };
        let original = Resident::load(&open_file(&path).unwrap(), plane).unwrap();
        let peer = Resident::from_shared(original.file.try_clone().unwrap(), plane).unwrap();
        use std::os::unix::fs::MetadataExt;
        assert_eq!(
            original.file.metadata().unwrap().ino(),
            peer.file.metadata().unwrap().ino()
        );
        assert!(original.file.set_len(1).is_err());
        let value = 255u8;
        assert_eq!(
            unsafe { libc::pwrite(original.descriptor(), (&value as *const u8).cast(), 1, 97) },
            -1
        );
        let wrong = Plane {
            nbytes: plane.nbytes + BLOCK,
            ..plane
        };
        assert!(Resident::from_shared(original.file.try_clone().unwrap(), wrong).is_err());
        std::fs::remove_file(path).unwrap();
        drop(original);
        assert_eq!(peer.bytes(), &bytes[97..97 + 8192]);
    }

    #[test]
    fn resident_load_covers_parallel_chunks_and_rejects_truncated_planes() {
        let path =
            std::env::temp_dir().join(format!("helios-engram-bulk-{}.bin", std::process::id()));
        let rows = LOAD_CHUNK * 5 / 80 + 1;
        let first = 123;
        let second = first + rows * 80 + 37;
        let bytes: Vec<_> = (0..second + rows * 10)
            .map(|i| (i * 13 + i / 257) as u8)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let planes = [
            Plane {
                file: 0,
                offset: first as u64,
                nbytes: rows * 80,
                row_bytes: 80,
            },
            Plane {
                file: 0,
                offset: second as u64,
                nbytes: rows * 10,
                row_bytes: 10,
            },
        ];
        let file = open_file(&path).unwrap();
        for plane in planes {
            let data = Resident::load(&file, plane).unwrap();
            let begin = plane.offset as usize;
            assert_eq!(data.bytes(), &bytes[begin..begin + plane.nbytes]);
        }
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len((first + LOAD_CHUNK + 17) as u64)
            .unwrap();
        assert!(
            Resident::load(&file, planes[0])
                .err()
                .unwrap()
                .to_string()
                .contains("requires")
        );
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn warm_rows_avoid_disk_and_failed_misses_never_enter_the_cache() {
        let path =
            std::env::temp_dir().join(format!("helios-engram-cache-{}.bin", std::process::id()));
        let rows = 128;
        let first = 4093;
        let second = first + rows * 80 + 7;
        let bytes: Vec<u8> = (0..second + rows * 10)
            .map(|i| (i * 13 + i / 257) as u8)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let planes = [
            Plane {
                file: 0,
                offset: first as u64,
                nbytes: rows * 80,
                row_bytes: 80,
            },
            Plane {
                file: 0,
                offset: second as u64,
                nbytes: rows * 10,
                row_bytes: 10,
            },
        ];
        let stats = Arc::new(Stats::default());
        let counters = Arc::new(TableCounters::default());
        let mut reader = Reader::open(&path, &planes, false, 65536, 32, stats.clone(), counters.clone()).unwrap();
        assert!(counters.cache_bytes.load(Ordering::Relaxed) <= 65536);
        fn check(reader: &mut Reader, planes: &[Plane; 2], bytes: &[u8], ids: &[i64]) {
            let mut expected = Vec::new();
            for plane in planes {
                for &id in ids {
                    if id < 0 {
                        expected.extend(std::iter::repeat_n(0, plane.row_bytes));
                    } else {
                        let start = plane.offset as usize + id as usize * plane.row_bytes;
                        expected.extend_from_slice(&bytes[start..start + plane.row_bytes]);
                    }
                }
            }
            let mut output = vec![0xa5; ids.len() * 90];
            reader.gather(planes, ids, &mut output).unwrap();
            assert_eq!(output, expected);
        }
        check(&mut reader, &planes, &bytes, &[2, 9, 2, -1, 17]);
        assert_eq!(counters.misses.load(Ordering::Relaxed), 3);
        assert_eq!(counters.resident_rows.load(Ordering::Relaxed), 3);
        let reads = counters.disk_reads.load(Ordering::Relaxed);
        assert!(reads > 0);
        check(&mut reader, &planes, &bytes, &[17, 9, -1, 2, 17]);
        assert_eq!(counters.hits.load(Ordering::Relaxed), 3);
        assert_eq!(counters.disk_reads.load(Ordering::Relaxed), reads);
        assert!(reader.gather(&planes, &[128], &mut [0; 90]).is_err());
        assert!(reader.gather(&planes, &[-2], &mut [0; 90]).is_err());
        assert!(reader.gather(&planes, &[2; 33], &mut [0; 33 * 90]).is_err());
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(0)
            .unwrap();
        check(&mut reader, &planes, &bytes, &[9, 2]);
        assert!(reader.gather(&planes, &[20], &mut [0; 90]).is_err());
        assert_eq!(counters.resident_rows.load(Ordering::Relaxed), 3);
        std::fs::write(&path, &bytes).unwrap();
        check(&mut reader, &planes, &bytes, &[20]);
        assert_eq!(counters.resident_rows.load(Ordering::Relaxed), 4);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn disk_and_ram_gather_identical_rows_across_unaligned_planes() {
        let path = std::env::temp_dir().join(format!("helios-engram-{}.bin", std::process::id()));
        let rows = 4097;
        let first = 4093;
        let second = first + rows * 256 + 7;
        let bytes: Vec<_> = (0..second + rows * 8)
            .map(|i| (i * 13 + i / 257) as u8)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let planes = [
            Plane {
                file: 0,
                offset: first as u64,
                nbytes: rows * 256,
                row_bytes: 256,
            },
            Plane {
                file: 0,
                offset: second as u64,
                nbytes: rows * 8,
                row_bytes: 8,
            },
        ];
        let ids: Vec<_> = (0..8192)
            .map(|i| {
                if i % 19 == 0 {
                    -1
                } else {
                    (i * 101 % rows) as i64
                }
            })
            .collect();
        let mut expected = Vec::new();
        for plane in &planes {
            for id in &ids {
                if *id == -1 {
                    expected.extend(std::iter::repeat_n(0, plane.row_bytes));
                } else {
                    let offset = plane.offset as usize + *id as usize * plane.row_bytes;
                    expected.extend_from_slice(&bytes[offset..offset + plane.row_bytes]);
                }
            }
        }
        for (resident, cache_bytes) in [(false, 0), (true, 0), (false, 3 << 20)] {
            let mut reader = Reader::open(
                &path,
                &planes,
                resident,
                cache_bytes,
                ids.len(),
                Arc::default(),
                Arc::default(),
            )
            .unwrap();
            for _ in 0..3 {
                let mut actual = vec![0xA5; expected.len()];
                reader.gather(&planes, &ids, &mut actual).unwrap();
                assert_eq!(actual, expected);
            }
            assert!(
                reader
                    .gather(&planes, &[rows as i64], &mut [0; 264])
                    .is_err()
            );
        }
        let mut reader =
            Reader::open(&path, &planes, false, 0, ids.len(), Arc::default(), Arc::default()).unwrap();
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(4096)
            .unwrap();
        assert!(reader.gather(&planes, &[100], &mut [0; 264]).is_err());
        std::fs::remove_file(&path).unwrap();
    }
    #[test]
    fn equal_offsets_in_separate_files_never_coalesce() {
        let root = std::env::temp_dir().join(format!("helios-engram-files-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let planes = [
            Plane {
                file: 0,
                offset: 4093,
                nbytes: 80 * 128,
                row_bytes: 80,
            },
            Plane {
                file: 1,
                offset: 4093,
                nbytes: 10 * 128,
                row_bytes: 10,
            },
        ];
        let mut files = Vec::new();
        for (i, plane) in planes.iter().enumerate() {
            let path = root.join(i.to_string());
            std::fs::write(
                &path,
                vec![i as u8 + 1; plane.offset as usize + plane.nbytes],
            )
            .unwrap();
            files.push(Arc::new(open_file(&path).unwrap()));
        }
        for (resident, cache) in [(false, 0), (false, 65536), (true, 0)] {
            let mut reader =
                Reader::from_files(
                    &files,
                    &planes,
                    resident,
                    cache,
                    16,
                    Arc::default(),
                    Arc::default(),
                    1,
                    false,
                )
                .unwrap();
            for _ in 0..3 {
                let mut output = vec![0xa5; 4 * 90];
                reader
                    .gather(&planes, &[17, 2, -1, 17], &mut output)
                    .unwrap();
                assert_eq!(&output[..80], &[1; 80]);
                assert_eq!(&output[160..240], &[0; 80]);
                assert_eq!(&output[320..340], &[2; 20]);
                assert_eq!(&output[340..350], &[0; 10]);
                assert_eq!(&output[350..360], &[2; 10]);
            }
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    /// A 2560-byte row read at file offset 2560 splits on the file's floor.
    #[test]
    fn fragment_floor_follows_the_files_direct_io_alignment() {
        let plane = Plane {
            file: 0,
            offset: 0,
            nbytes: 64 * 2560,
            row_bytes: 2560,
        };
        for (gran, expected) in [
            (BLOCK, vec![(2560u64, 1536usize), (4096, 1024)]),
            (
                GRAN_MIN,
                vec![(2560, 512), (3072, 512), (3584, 512), (4096, 512), (4608, 512)],
            ),
        ] {
            let mut output = vec![0xa5; 2560];
            let mut fragments = Vec::new();
            plan_fragments(&plane, &[1], gran, 0, &mut output, &mut fragments);
            assert_eq!(
                fragments
                    .iter()
                    .map(|f| (f.source, f.nbytes))
                    .collect::<Vec<_>>(),
                expected,
                "floor {gran}"
            );
            // Every fragment starts where the previous one ended, and together
            // they cover the row exactly once.
            assert_eq!(fragments.first().unwrap().destination, 0);
            assert_eq!(
                fragments.iter().map(|f| f.nbytes).sum::<usize>()
                    + if fragments.is_empty() { 0 } else { 0 },
                2560
            );
        }
    }

    #[test]
    fn reads_merge_only_consecutive_blocks_within_one_slot() {
        let plane = Plane {
            file: 0,
            offset: 0,
            nbytes: 64 * 2560,
            row_bytes: 2560,
        };
        let mut output = vec![0; 2560];
        let mut fragments = Vec::new();
        plan_fragments(&plane, &[1], GRAN_MIN, 0, &mut output, &mut fragments);
        fragments.sort_unstable_by_key(|f| (f.file, f.source));
        let mut reads = Vec::new();
        let (blocks, coalesced) = plan_reads(
            &fragments,
            &mut reads,
            |_| GRAN_MIN as u64,
            |_| 64 * 2560,
        );
        assert_eq!(reads.len(), 1, "a contiguous run becomes one read");
        assert_eq!(reads[0].offset, 2560);
        assert_eq!(reads[0].nbytes, 2560);
        assert_eq!(reads[0].required, 2560);
        assert_eq!(blocks, 5);
        assert_eq!(coalesced, 1);
        // A file shorter than the aligned extent clamps the request to real bytes,
        // and the short read is then reported by the completion check.
        let mut clamped = Vec::new();
        let _ = plan_reads(&fragments, &mut clamped, |_| GRAN_MIN as u64, |_| 3000);
        assert_eq!(clamped[0].nbytes, 440);
        assert!(clamped[0].required > clamped[0].nbytes);
    }

    #[test]
    fn reads_never_merge_across_files_or_beyond_one_slot() {
        let floor = 512u64;
        let fragments = [
            Fragment {
                file: 0,
                source: 512,
                destination: 0,
                nbytes: 512,
            },
            Fragment {
                file: 1,
                source: 512,
                destination: 512,
                nbytes: 512,
            },
            // Same file as the first, but 64 KiB away: beyond one staging slot.
            Fragment {
                file: 0,
                source: 512 + CHUNK as u64,
                destination: 1024,
                nbytes: 512,
            },
        ];
        let mut reads = Vec::new();
        let (blocks, coalesced) = plan_reads(&fragments, &mut reads, |_| floor, |_| u64::MAX);
        assert_eq!(reads.len(), 3, "equal offsets in separate files never merge");
        assert_eq!(blocks, 3);
        assert_eq!(coalesced, 0);
        assert_eq!(reads[1].file, 1);
    }

    #[cfg(unix)]
    #[test]
    fn queue_depth_ladder_clamps_then_halves_to_the_floor() {
        // `start_depth` takes the budget in slots, as `slot_budget` reports it.
        assert_eq!(start_depth(QUEUE_DEPTH_DEFAULT, usize::MAX), 256);
        assert_eq!(start_depth(QUEUE_DEPTH_DEFAULT, 3), 3);
        assert_eq!(start_depth(QUEUE_DEPTH_DEFAULT, 1), QUEUE_DEPTH_MIN);
        assert_eq!(start_depth(1024, usize::MAX), 1024);
        assert_eq!(shallower(256), Some(128));
        assert_eq!(shallower(4), Some(2));
        assert_eq!(shallower(QUEUE_DEPTH_MIN), None);
    }

    #[test]
    fn planning_covers_the_smallest_allowed_fragment_floor() {
        let planes = [
            Plane {
                file: 0,
                offset: 0,
                nbytes: 64 * 2560,
                row_bytes: 2560,
            },
            Plane {
                file: 0,
                offset: 64 * 2560,
                nbytes: 64 * 8,
                row_bytes: 8,
            },
        ];
        // Each row can split into `row_bytes / GRAN_MIN + 2` fragments, and each
        // fragment run into one read.
        assert!(planning_bytes(&planes, 8192) > 0);
        let wider = [
            planes[0],
            Plane {
                row_bytes: 1024,
                nbytes: 64 * 1024,
                ..planes[1]
            },
        ];
        assert!(planning_bytes(&wider, 8192) > planning_bytes(&planes, 8192));
        assert_eq!(planning_bytes(&planes, 0), 0);
    }

    #[test]
    fn prefetch_warms_the_cache_and_the_real_gather_then_avoids_the_disk() {
        let root = std::env::temp_dir().join(format!("helios-engram-prefetch-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let rows = 64;
        let row_bytes = 90;
        let path = root.join("table.bin");
        let bytes: Vec<u8> = (0..rows * row_bytes)
            .map(|i| (i * 31 + i / 257) as u8)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let planes = [
            Plane {
                file: 0,
                offset: 0,
                nbytes: rows * 80,
                row_bytes: 80,
            },
            Plane {
                file: 0,
                offset: (rows * 80) as u64,
                nbytes: rows * 10,
                row_bytes: 10,
            },
        ];
        let counters = Arc::new(TableCounters::default());
        let mut reader = Reader::from_files(
            &[Arc::new(open_file(&path).unwrap())],
            &planes,
            false,
            1 << 20,
            16,
            Arc::default(),
            counters.clone(),
            1,
            false,
        )
        .unwrap();
        // Out-of-range, negative, and duplicate ids are dropped silently; the two
        // distinct in-range rows are read once each.
        assert_eq!(
            reader.prefetch(&planes, &[7, 7, 64, -1, 12]).unwrap(),
            2
        );
        assert_eq!(counters.prefetch_rows_loaded.load(Ordering::Relaxed), 2);
        assert_eq!(counters.misses.load(Ordering::Relaxed), 0, "prefetch is not a lookup");
        let reads = counters.disk_reads.load(Ordering::Relaxed);
        assert!(reads > 0);
        let mut expected = Vec::new();
        for (plane, row_bytes) in planes.iter().zip([80, 10]) {
            for &id in &[7, 7, 12] {
                let start = plane.offset as usize + id as usize * row_bytes;
                expected.extend_from_slice(&bytes[start..start + row_bytes]);
            }
        }
        let mut output = vec![0xa5; expected.len()];
        reader.gather(&planes, &[7, 7, 12], &mut output).unwrap();
        assert_eq!(output, expected);
        assert_eq!(counters.hits.load(Ordering::Relaxed), 2);
        assert_eq!(
            counters.disk_reads.load(Ordering::Relaxed),
            reads,
            "the prefetched rows must not be read again"
        );
        // A second identical prefetch is a no-op: everything is already resident.
        assert_eq!(reader.prefetch(&planes, &[7, 12]).unwrap(), 0);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn prefetch_without_a_row_cache_does_no_disk_work() {
        let root = std::env::temp_dir().join(format!("helios-engram-nocache-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let rows = 64;
        let path = root.join("table.bin");
        std::fs::write(&path, vec![7u8; rows * 90]).unwrap();
        let planes = [
            Plane {
                file: 0,
                offset: 0,
                nbytes: rows * 80,
                row_bytes: 80,
            },
            Plane {
                file: 0,
                offset: (rows * 80) as u64,
                nbytes: rows * 10,
                row_bytes: 10,
            },
        ];
        let counters = Arc::new(TableCounters::default());
        let mut reader = Reader::from_files(
            &[Arc::new(open_file(&path).unwrap())],
            &planes,
            false,
            0,
            16,
            Arc::default(),
            counters.clone(),
            1,
            false,
        )
        .unwrap();
        assert_eq!(reader.prefetch(&planes, &[7, 12, 64, -1]).unwrap(), 0);
        assert_eq!(counters.disk_reads.load(Ordering::Relaxed), 0);
        // The real path still reads, and still rejects a row beyond the table.
        reader.gather(&planes, &[7], &mut vec![0; 90]).unwrap();
        assert!(counters.disk_reads.load(Ordering::Relaxed) > 0);
        assert!(reader.gather(&planes, &[64], &mut vec![0; 90]).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    /// A wave abandoned after a short read must leave the reader usable: the ring
    /// is drained before the error returns, the slot list is re-issued for the next
    /// gather, and a restored file then serves every requested row.
    #[test]
    #[cfg(unix)]
    fn a_failed_gather_drains_and_the_next_gather_serves_every_row() {
        let root = std::env::temp_dir().join(format!("helios-engram-slots-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("table.bin");
        let rows = 400usize;
        let (weight_bytes, scale_bytes) = (240usize, 16usize);
        let bytes: Vec<u8> = (0..rows * (weight_bytes + scale_bytes))
            .map(|i| (i * 37 + i / 101) as u8)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let planes = [
            Plane {
                file: 0,
                offset: 0,
                nbytes: rows * weight_bytes,
                row_bytes: weight_bytes,
            },
            Plane {
                file: 0,
                offset: (rows * weight_bytes) as u64,
                nbytes: rows * scale_bytes,
                row_bytes: scale_bytes,
            },
        ];
        let mut reader = Reader::open(
            &path,
            &planes,
            false,
            0,
            rows,
            Arc::default(),
            Arc::new(TableCounters::default()),
        )
        .unwrap();
        // More reads than the ring has slots, so several waves are armed.
        let ids: Vec<i64> = (0..rows as i64).collect();
        let mut output = vec![0u8; rows * (weight_bytes + scale_bytes)];
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len((weight_bytes + scale_bytes) as u64)
            .unwrap();
        assert!(reader.gather(&planes, &ids, &mut output).is_err());
        std::fs::write(&path, &bytes).unwrap();
        // A leaked slot would stop arming later reads and leave this buffer stale
        // rather than error, so compare every byte.
        reader.gather(&planes, &ids, &mut output).unwrap();
        assert_eq!(output, bytes);
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[cfg(test)]
mod proptests {
    //! Invariants of the row planner over arbitrary geometries: every requested
    //! byte is covered exactly once, no read crosses its staging slot or its
    //! file's fragment floor, and the prefetch normalisation never invents a row.
    use super::*;
    use proptest::prelude::*;

    fn arbitrary_plane(seed: u64) -> Plane {
        let row_bytes = 1 + (seed % 4096) as usize;
        let rows = 1 + ((seed >> 13) % 64) as usize;
        let offset = ((seed >> 26) % 512) as u64;
        Plane {
            file: 0,
            offset,
            nbytes: rows * row_bytes,
            row_bytes,
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        /// Each in-range row is covered by fragments whose total length equals
        /// the row, each beginning where the previous one ended, and none
        /// crossing a multiple of the file's fragment floor.
        #[test]
        fn fragments_cover_every_row_exactly_once(
            seed in any::<u64>(),
            floor in prop::sample::select(vec![BLOCK, GRAN_MIN, 1024, 2048]),
            count in 1usize..24,
        ) {
            let plane = arbitrary_plane(seed);
            let rows = plane.nbytes / plane.row_bytes;
            let ids: Vec<i64> = (0..count).map(|i| (i * 7 % rows) as i64).collect();
            let mut output = vec![0xa5; count * plane.row_bytes];
            let mut fragments = Vec::new();
            plan_fragments(&plane, &ids, floor, 0, &mut output, &mut fragments);
            let mut covered = vec![0u32; count * plane.row_bytes];
            for fragment in &fragments {
                prop_assert!(fragment.nbytes > 0);
                prop_assert!(fragment.nbytes <= floor);
                prop_assert!(
                    fragment.source % floor as u64 + fragment.nbytes as u64 <= floor as u64,
                    "a fragment may not cross a {floor}-byte boundary"
                );
                for byte in fragment.destination..fragment.destination + fragment.nbytes {
                    covered[byte] += 1;
                }
            }
            for (row, &id) in ids.iter().enumerate() {
                let base = row * plane.row_bytes;
                for offset in 0..plane.row_bytes {
                    prop_assert_eq!(covered[base + offset], 1, "row {} byte {}", id, offset);
                }
            }
        }

        /// Reads never overlap a file boundary, never exceed one staging slot, and
        /// together cover every fragment they were given.
        #[test]
        fn reads_stay_within_one_slot_and_cover_their_fragments(
            seed in any::<u64>(),
            floor in prop::sample::select(vec![BLOCK, GRAN_MIN, 1024]),
            count in 1usize..16,
        ) {
            let plane = arbitrary_plane(seed);
            let rows = plane.nbytes / plane.row_bytes;
            let ids: Vec<i64> = (0..count).map(|i| (i * 13 % rows) as i64).collect();
            let mut output = vec![0u8; count * plane.row_bytes];
            let mut fragments = Vec::new();
            plan_fragments(&plane, &ids, floor, 0, &mut output, &mut fragments);
            fragments.sort_unstable_by_key(|f| (f.file, f.source));
            let size = (plane.offset + plane.nbytes as u64) as u64;
            let mut reads = Vec::new();
            let (blocks, coalesced) = plan_reads(
                &fragments,
                &mut reads,
                |_| floor as u64,
                |_| size,
            );
            prop_assert!(blocks >= reads.len() as u64);
            prop_assert!(coalesced <= reads.len() as u64);
            for read in &reads {
                prop_assert!(read.nbytes <= CHUNK);
                prop_assert!(read.required <= read.nbytes);
                prop_assert_eq!(read.offset % floor as u64, 0);
                prop_assert!(read.fragments.start < read.fragments.end);
                prop_assert!(read.fragments.end <= fragments.len());
                for fragment in &fragments[read.fragments.clone()] {
                    prop_assert_eq!(fragment.file, read.file);
                    let end = (fragment.source - read.offset) as usize + fragment.nbytes;
                    prop_assert!(end <= read.required, "fragment outside its read");
                }
            }
            // Every fragment belongs to exactly one read.
            let mut assigned = vec![0u32; fragments.len()];
            for read in &reads {
                for index in read.fragments.clone() {
                    assigned[index] += 1;
                }
            }
            prop_assert!(assigned.iter().all(|&count| count == 1), "fragment coverage");
        }

        /// Prefetch admission is idempotent and never counts a row twice.
        #[test]
        fn prefetch_normalisation_is_idempotent(
            rows in 8usize..40,
            ids in prop::collection::vec(-3i64..60, 1..48),
        ) {
            let root = std::env::temp_dir().join(format!(
                "helios-engram-proptest-{}-{}",
                std::process::id(),
                rows,
            ));
            std::fs::create_dir_all(&root).unwrap();
            let path = root.join("table.bin");
            std::fs::write(&path, vec![3u8; rows * 90]).unwrap();
            let planes = [
                Plane { file: 0, offset: 0, nbytes: rows * 80, row_bytes: 80 },
                Plane { file: 0, offset: (rows * 80) as u64, nbytes: rows * 10, row_bytes: 10 },
            ];
            let counters = Arc::new(TableCounters::default());
            let mut reader = Reader::from_files(
                &[Arc::new(open_file(&path).unwrap())],
                &planes,
                false,
                1 << 20,
                64,
                Arc::default(),
                counters.clone(),
                1,
                false,
            )
            .unwrap();
            let first = reader.prefetch(&planes, &ids).unwrap();
            let loaded = counters.prefetch_rows_loaded.load(Ordering::Relaxed) as usize;
            prop_assert_eq!(first, loaded);
            let distinct = ids
                .iter()
                .filter(|&&id| id >= 0 && (id as usize) < rows)
                .collect::<std::collections::HashSet<_>>()
                .len();
            prop_assert!(first <= distinct);
            prop_assert!(
                counters.resident_rows.load(Ordering::Relaxed)
                    <= counters.capacity_rows.load(Ordering::Relaxed)
            );
            // Re-prefetching is safe and bounded: a row a set collision evicted may
            // legitimately be read again, so the requirement is no error and no more
            // work than the distinct rows, not a fixed read count.
            let reads = counters.disk_reads.load(Ordering::Relaxed);
            let again = reader.prefetch(&planes, &ids).unwrap();
            prop_assert!(again <= distinct);
            prop_assert!(counters.disk_reads.load(Ordering::Relaxed) >= reads);
            prop_assert!(reader.prefetch(&planes, &[]).unwrap() == 0);
            std::fs::remove_dir_all(root).unwrap();
        }
    }
}
