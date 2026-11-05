/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Row addressing for the compact n-gram embedding tables.
//!
//! The captured graph is the authority: it hashes the committed token window and
//! publishes the row ids its own lookup then consumes. These functions reproduce
//! that mapping on the host so a reader thread can start fetching the rows a step
//! will request before the step is launched. Every parameter comes from the
//! signed package's `Engram` and `EngramHash` messages, so this is shared
//! behaviour over declared data rather than model-specific policy: the only
//! discriminator is the declared [`RowMethod`], which the loader already ties to
//! the request source and the vocabulary map (`package/schema.rs::validate_engram`).
//!
//! A wrong prediction costs disk bandwidth and nothing else, because the rows
//! actually served always come from the ids the graph published;
//! `flare_engram_prediction_mismatches_total` reports the disagreement.

use super::{Index, RowGeometry, RowMethod, Table};

/// The largest lag any head needs, so a caller knows how much committed history a
/// prediction requires: the window for a query is the query itself plus the
/// `depth - 1` tokens before it.
pub(crate) fn history_depth(geometry: &RowGeometry) -> usize {
    geometry.orders.iter().copied().max().unwrap_or(2).max(2) as usize
}

/// Compress raw token ids into the ids the lookup hashes, into `out`.
///
/// A token is dead when it has no normalized entry, when the caller excludes the
/// position, or when it is negative. Engram's declared token map performs the
/// normalization; the EOS-bounded form has none, so it passes tokens through.
/// Dead positions become `-1`, which the derivation below turns into the pad
/// value. `excluded` receives the absolute position, so a caller can drop the
/// image positions that never enter an n-gram history. `out` is caller-owned
/// scratch: a prefill chunk's window is wide enough that the submission path must
/// not allocate per step.
pub(crate) fn compress_into(
    index: &Index,
    table: &Table,
    base: usize,
    tokens: &[u32],
    excluded: impl Fn(usize) -> bool,
    out: &mut Vec<i64>,
) {
    let mapped = |token: u32| -> Option<i64> {
        match table.geometry.method {
            RowMethod::XorModulo => usize::try_from(token)
                .ok()
                .filter(|entry| *entry < index.token_map.len())
                .map(|entry| index.token_map[entry] as i64),
            // The EOS-bounded oracle has no vocabulary map: every token passes
            // through, and only the pad token bounds a window.
            RowMethod::EosXorModulo => Some(token as i64),
        }
    };
    out.clear();
    out.extend(tokens.iter().enumerate().map(|(offset, &token)| {
        if excluded(base + offset) {
            -1
        } else {
            mapped(token).unwrap_or(-1)
        }
    }));
}

/// The value each method substitutes for a position outside the window.
///
/// A window position is filled in the same space as the tokens around it, so the
/// Engram form maps the declared pad token through the same normalization the
/// recipe's hash kernel uses, and the EOS-bounded form, which has no map, uses the
/// declared value directly. The EOS-bounded form also treats that value as the
/// segment separator, so a window never reaches behind the most recent
/// occurrence. A pad id with no map entry leaves the window dead, which the
/// prediction-mismatch counter reports rather than serving wrongly.
fn pad(index: &Index, method: RowMethod) -> i64 {
    match method {
        RowMethod::XorModulo => index
            .token_map
            .get(index.pad_token_id as usize)
            .copied()
            .map_or(-1, |pad| pad as i64),
        RowMethod::EosXorModulo => index.pad_token_id as i64,
    }
}

/// Append the `queries.len() * table.heads` row ids the lookup will request for
/// the query tokens that follow `history`, both already passed through
/// [`compress_into`].
///
/// Ids outside this rank's partition are emitted as `-1` so the caller can drop
/// them: a partitioned table stores only the rows its rank owns.
pub(crate) fn query_row_ids(
    index: &Index,
    table: &Table,
    history: &[i64],
    queries: &[i64],
    out: &mut Vec<i64>,
) {
    out.clear();
    if queries.is_empty() {
        return;
    }
    let geometry = &table.geometry;
    let depth = history_depth(geometry);
    // A package always carries a complete geometry, validated at load. The
    // preparation-time catalog carries none, and a partial geometry must predict
    // nothing rather than index past an array.
    if geometry.orders.len() < table.heads
        || geometry.primes.len() < table.heads
        || geometry.offsets.len() < table.heads
        || geometry.multipliers.len() < depth
    {
        return;
    }
    let pad = pad(index, geometry.method);
    let cache = history.len() + queries.len();
    // Positions before the request's stored history are absent rather than dead:
    // they read as the pad value, matching the committed history a fresh request
    // is seeded with.
    let window = |position: isize| -> i64 {
        let Ok(offset) = usize::try_from(position) else {
            return pad;
        };
        if offset < history.len() {
            history[offset]
        } else if offset < cache {
            queries[offset - history.len()]
        } else {
            pad
        }
    };
    // The EOS-bounded form restarts a segment after every pad token, so a window
    // never reaches behind the most recent one. The boundary only moves forward as
    // query positions advance, so it is tracked incrementally rather than rescanned
    // per head: a prefill chunk is thousands of positions wide. It advances after a
    // row is built, because the oracle bounds a row at the most recent EOS *before*
    // that position: an EOS at the query position is itself the pad value, but the
    // older positions of its own window still count.
    let mut boundary = 0usize;
    for position in 0..history.len() {
        if window(position as isize) == pad {
            boundary = position + 1;
        }
    }
    for current in history.len()..cache {
        for head in 0..table.heads {
            let order = geometry.orders.get(head).copied().unwrap_or(2) as usize;
            let mut rolling = 0i64;
            // The XOR form accumulates a dead token into every older lag; the
            // EOS-bounded form replaces only the positions outside the segment.
            let mut blocked = false;
            for lag in 0..order.min(depth) {
                let position = current as isize - lag as isize;
                let source = match geometry.method {
                    RowMethod::XorModulo => {
                        let value = window(position);
                        blocked |= value < 0;
                        if blocked { pad } else { value }
                    }
                    RowMethod::EosXorModulo => {
                        let value = window(position);
                        if usize::try_from(position).is_ok_and(|position| position < boundary)
                            || value < 0
                        {
                            pad
                        } else {
                            value
                        }
                    }
                };
                rolling ^= source.wrapping_mul(geometry.multipliers[lag] as i64);
            }
            let Some(prime) = geometry.primes.get(head) else {
                out.push(-1);
                continue;
            };
            let row = rolling
                .rem_euclid(*prime as i64)
                .checked_add(
                    geometry
                        .offsets
                        .get(head)
                        .copied()
                        .and_then(|offset| i64::try_from(offset).ok())
                        .unwrap_or(i64::MAX),
                );
            // `decode` rejects an offset plus prime beyond the table, so the only
            // failure left is a signed overflow, which is not a row.
            let local = row.unwrap_or(i64::MAX) - table.first_row as i64;
            out.push(if local < 0 || local as u64 >= table.rows {
                -1
            } else {
                local
            });
        }
        // The next row's segment begins after this position when it holds the pad
        // token, which is how the EOS-bounded oracle bounds a window.
        if window(current as isize) == pad {
            boundary = current + 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Declared geometry the vectors below were produced with. The authority is the
    /// upstream b12x oracle pair: `sequence/engram/host.py::query_row_ids` with
    /// `sequence/engram/reference.py::hash_reference` for the XOR form, and
    /// `sequence/ple_hash/reference.py::ple_hash_ids_reference` with
    /// `::eos_bounded_windows` for the EOS-bounded form. Reproduce a vector by
    /// transcribing those functions literally, keeping their integer semantics:
    /// products and XOR accumulate as wrapping i64, and the modulus is floor
    /// modulo (Python `%`, `torch.remainder`), which is `i64::rem_euclid` here.
    /// Heads are emitted in output-column order, so each head's `orders` entry
    /// selects how many lags reach into the window.
    const ENGRAM_MULTIPLIERS: [u64; 4] = [
        0x9E37_79B9_7F4A_7C15,
        0xBF58_476D_1CE4_E5B9,
        0x94D0_49BB_1331_11EB,
        0xD6E8_FEB8_6659_FD93,
    ];

    fn engram_table(rows: u64, first_row: u64) -> (Index, Table) {
        let heads = 24;
        let index = Index {
            tables: Vec::new(),
            // Identity normalization, so a token is its own compressed id and only an
            // excluded position is dead.
            token_map: (0..40u32).collect(),
            pad_token_id: 2,
        };
        let table = Table {
            layer: 1,
            heads,
            rows,
            first_row,
            geometry: RowGeometry {
                method: RowMethod::XorModulo,
                orders: (2..=4).flat_map(|order| [order; 8]).collect(),
                multipliers: ENGRAM_MULTIPLIERS.to_vec(),
                primes: (0..heads as u64).map(|head| 1009 + 2 * head).collect(),
                offsets: (0..heads as u64).map(|head| 10_000 * head).collect(),
            },
            ..Default::default()
        };
        (index, table)
    }

    fn ple_table(rows: u64) -> (Index, Table) {
        let heads = 16;
        let index = Index {
            tables: Vec::new(),
            token_map: Vec::new(),
            pad_token_id: 7,
        };
        let table = Table {
            layer: 3,
            heads,
            rows,
            first_row: 0,
            geometry: RowGeometry {
                method: RowMethod::EosXorModulo,
                orders: (2..=3).flat_map(|order| [order; 8]).collect(),
                multipliers: vec![1_000_003, 2_000_003, 3_000_007],
                primes: (0..heads as u64).map(|head| 71 + 2 * head).collect(),
                offsets: (0..heads as u64).map(|head| 500 * (head / 8)).collect(),
            },
            ..Default::default()
        };
        (index, table)
    }

    fn predict(index: &Index, table: &Table, history: &[i64], queries: &[i64]) -> Vec<i64> {
        let mut out = Vec::new();
        query_row_ids(index, table, history, queries, &mut out);
        out
    }

    #[test]
    fn engram_window_matches_the_upstream_oracle() {
        let (index, table) = engram_table(231_055, 0);
        assert_eq!(
            predict(&index, &table, &[1, 2, 3], &[4, 5]),
            [
                380, 10100, 20757, 30371, 40658, 50425, 60044, 70892, //
                80170, 90569, 100214, 110207, 120404, 130760, 140509, 150316,
                160558, 170760, 180547, 190954, 200303, 210689, 220366, 230592,
                287, 10539, 20948, 30794, 40077, 50121, 60651, 70335, //
                80187, 90188, 100484, 110179, 120818, 130832, 140608, 150628,
                160709, 170392, 180557, 190931, 200292, 210450, 220958, 230462,
            ]
        );
    }

    /// A dead token makes its own lag and every older lag read as the pad value,
    /// while the lags nearer the current token keep their own terms.
    #[test]
    fn dead_token_pads_its_own_and_every_older_lag() {
        let (index, table) = engram_table(231_055, 0);
        assert_eq!(
            predict(&index, &table, &[1, 2, -1], &[4, 5]),
            [
                39, 10319, 20954, 30839, 40784, 50291, 60654, 70901, //
                80886, 90120, 100140, 110388, 120858, 130791, 140669, 150563,
                160966, 170176, 180596, 190606, 201038, 210685, 220081, 231026,
                287, 10539, 20948, 30794, 40077, 50121, 60651, 70335, //
                80254, 90253, 100241, 110517, 120571, 130544, 140267, 150183,
                160579, 171002, 180466, 190093, 200393, 210497, 220420, 230576,
            ]
        );
        assert_eq!(
            predict(&index, &table, &[1, 2, 3], &[-1, 5]),
            [
                965, 10010, 20938, 30692, 40826, 50919, 60149, 70247, //
                80889, 90055, 100641, 110998, 120399, 130974, 140612, 150725,
                160832, 170864, 180183, 190874, 200659, 210305, 220592, 230128,
                600, 10786, 20140, 30309, 40546, 50453, 60810, 70084, //
                80989, 90599, 100523, 110937, 120396, 130619, 140632, 150579,
                160901, 170534, 180838, 190574, 200932, 210017, 220175, 230458,
            ]
        );
    }

    /// Small multipliers keep every product positive, pinning the modulus itself:
    /// a signed `%` instead of `rem_euclid` would move the near-lag heads.
    #[test]
    fn positive_products_use_the_floor_modulus() {
        let (index, mut table) = engram_table(231_055, 0);
        table.geometry.multipliers = vec![3, 5, 7, 11];
        assert_eq!(
            predict(&index, &table, &[1, 2, 3], &[4]),
            [
                3, 10003, 20003, 30003, 40003, 50003, 60003, 70003, //
                80013, 90013, 100013, 110013, 120013, 130013, 140013, 150013,
                160006, 170006, 180006, 190006, 200006, 210006, 220006, 230006,
            ]
        );
    }

    #[test]
    fn ple_window_matches_the_upstream_oracle() {
        let (index, table) = ple_table(1_000);
        assert_eq!(
            predict(&index, &table, &[1, 2, 3], &[4]),
            [
                9, 42, 22, 39, 42, 52, 79, 27, //
                512, 584, 515, 581, 536, 522, 515, 507,
            ]
        );
        assert_eq!(
            predict(&index, &table, &[3, 4], &[5]),
            [
                9, 52, 45, 6, 1, 63, 16, 40, //
                534, 528, 536, 543, 511, 563, 597, 559,
            ]
        );
    }

    /// An EOS bounds the segment: the window never reaches behind the most recent
    /// EOS, so only the order-three head group changes.
    #[test]
    fn eos_bounds_the_segment() {
        let (index, table) = ple_table(1_000);
        assert_eq!(
            predict(&index, &table, &[1, 7, 3], &[4]),
            [
                9, 42, 22, 39, 42, 52, 79, 27, //
                568, 536, 543, 577, 548, 559, 520, 522,
            ]
        );
    }

    /// A pad token at the query position bounds the *next* row, not its own: the
    /// oracle advances its segment start only after emitting a row, so the older
    /// positions of that row's window still contribute their own tokens.
    #[test]
    fn pad_at_the_query_position_does_not_bound_its_own_row() {
        let (index, table) = ple_table(1_000);
        assert_eq!(
            predict(&index, &table, &[1, 2], &[7]),
            [
                40, 58, 31, 15, 67, 73, 50, 71, //
                525, 508, 561, 516, 522, 590, 573, 550,
            ]
        );
        assert_eq!(
            predict(&index, &table, &[1, 2, 7], &[4]),
            [
                57, 67, 55, 62, 13, 1, 38, 25, //
                517, 525, 553, 565, 590, 518, 565, 586,
            ]
        );
    }

    #[test]
    fn partitioned_table_emits_only_its_own_rows() {
        let (index, table) = engram_table(20_000, 100_000);
        assert_eq!(
            predict(&index, &table, &[1, 2, 3], &[4, 5]),
            [
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 214, 10207, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 484, 10179, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            ]
        );
    }

    /// A geometry the loader would reject must predict nothing rather than panic.
    #[test]
    fn absent_or_partial_geometry_predicts_nothing() {
        let index = Index {
            tables: Vec::new(),
            token_map: Vec::new(),
            pad_token_id: 0,
        };
        let table = Table {
            heads: 24,
            rows: 1000,
            ..Default::default()
        };
        assert!(predict(&index, &table, &[1, 2, 3], &[4]).is_empty());
        let mut partial = table.clone();
        partial.geometry = RowGeometry {
            method: RowMethod::XorModulo,
            orders: vec![2; 24],
            multipliers: vec![3],
            primes: vec![7; 24],
            offsets: vec![0; 24],
        };
        assert!(predict(&index, &partial, &[1, 2, 3], &[4]).is_empty());
    }

    #[test]
    fn compression_normalizes_and_excludes_positions() {
        let (index, table) = engram_table(231_055, 0);
        let mut out = Vec::new();
        compress_into(&index, &table, 0, &[1, 2, 3, 4], |position| position == 1, &mut out);
        assert_eq!(out, [1, -1, 3, 4]);
        // A token with no normalized entry is dead rather than wrapped.
        compress_into(&index, &table, 0, &[1, 99], |_| false, &mut out);
        assert_eq!(out, [1, -1]);
        let (ple_index, ple_table) = ple_table(1_000);
        // The EOS-bounded form has no token map: every token passes through, and
        // only the pad token bounds a window.
        compress_into(&ple_index, &ple_table, 0, &[1, 99], |_| false, &mut out);
        assert_eq!(out, [1, 99]);
        compress_into(&ple_index, &ple_table, 5, &[1, 2], |position| position == 5, &mut out);
        assert_eq!(out, [-1, 2]);
    }
}

#[cfg(test)]
mod proptests {
    //! Properties of the host row derivation over arbitrary declared geometry:
    //! every predicted row is a row of the table, the derivation is a function of
    //! the window it can see, and it never disagrees with itself.
    use super::*;
    use proptest::prelude::*;

    #[derive(Clone, Debug)]
    struct Geometry {
        method: RowMethod,
        max_order: u32,
        multipliers: Vec<u64>,
        primes: Vec<u64>,
        offsets: Vec<u64>,
        rows: u64,
        first_row: u64,
        vocabulary: u32,
    }

    fn geometry(methods: Vec<RowMethod>) -> impl Strategy<Value = Geometry> {
        (
            prop::sample::select(methods),
            2u32..=4,
            prop::collection::vec(1u64..(1u64 << 62), 4),
            1u64..(1u64 << 20),
            0u64..(1u64 << 12),
            1u32..64,
        )
            .prop_flat_map(
                |(method, max_order, multipliers, rows, first_row, vocabulary)| {
                    // One order, modulus, and row offset per lookup head, so the
                    // arrays must all be as long as the head count.
                    let heads = usize::try_from((max_order - 1) * 8).unwrap();
                    (
                        Just(method),
                        Just(max_order),
                        Just(multipliers),
                        prop::collection::vec(2u64..1024, heads),
                        prop::collection::vec(0u64..4096, heads),
                        Just(rows),
                        Just(first_row),
                        Just(vocabulary),
                    )
                        .prop_map(
                            |(
                                method,
                                max_order,
                                multipliers,
                                primes,
                                offsets,
                                rows,
                                first_row,
                                vocabulary,
                            )| {
                                Geometry {
                                    method,
                                    max_order,
                                    multipliers,
                                    primes,
                                    offsets,
                                    rows,
                                    first_row,
                                    vocabulary,
                                }
                            },
                        )
                },
            )
    }

    fn heads_of(declared: &Geometry) -> usize {
        usize::try_from((declared.max_order - 1) * 8).unwrap()
    }

    fn build(declared: &Geometry) -> (Index, Table) {
        let heads = heads_of(declared);
        let index = Index {
            tables: Vec::new(),
            token_map: if declared.method == RowMethod::XorModulo {
                (0..declared.vocabulary).collect()
            } else {
                Vec::new()
            },
            pad_token_id: 2,
        };
        let table = Table {
            layer: 1,
            heads,
            rows: declared.rows,
            first_row: declared.first_row,
            geometry: RowGeometry {
                method: declared.method,
                orders: (2..=declared.max_order)
                    .flat_map(|order| [order; 8])
                    .collect(),
                multipliers: declared.multipliers.clone(),
                primes: declared.primes.clone(),
                offsets: declared.offsets.clone(),
            },
            ..Default::default()
        };
        (index, table)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        /// Every predicted row is either absent or a row of the table, and one
        /// query position yields exactly one row per declared head.
        #[test]
        fn predicted_rows_are_table_rows(
            declared in geometry(vec![RowMethod::XorModulo, RowMethod::EosXorModulo]),
            history in prop::collection::vec(-3i64..70, 0..8),
            queries in prop::collection::vec(-3i64..70, 1..6),
        ) {
            let (index, table) = build(&declared);
            let mut out = Vec::new();
            query_row_ids(&index, &table, &history, &queries, &mut out);
            prop_assert_eq!(out.len(), queries.len() * table.heads);
            for &row in &out {
                prop_assert!(
                    row < 0 || (row as u64) < table.rows,
                    "row {} outside {} rows",
                    row,
                    table.rows
                );
            }
        }

        /// Only the deepest window matters for a row: committed tokens older than
        /// `max(orders) - 1` positions cannot change it. The EOS-bounded form is
        /// excluded because its segment boundary is a whole-request property.
        #[test]
        fn prefix_beyond_the_window_cannot_change_a_row(
            declared in geometry(vec![RowMethod::XorModulo]),
            prefix in prop::collection::vec(-3i64..70, 0..8),
            tail in prop::collection::vec(-3i64..70, 4..10),
        ) {
            let (index, table) = build(&declared);
            let depth = history_depth(&table.geometry);
            let query = *tail.last().unwrap();
            let history = &tail[..tail.len() - 1];
            let mut long_history = Vec::with_capacity(prefix.len() + history.len());
            long_history.extend(prefix.iter().copied());
            long_history.extend(history.iter().copied());
            let mut far = Vec::new();
            let mut near = Vec::new();
            query_row_ids(&index, &table, &long_history, &[query], &mut far);
            let keep = depth - 1;
            let start = long_history.len().saturating_sub(keep);
            query_row_ids(&index, &table, &long_history[start..], &[query], &mut near);
            prop_assert_eq!(far, near);
        }

        /// The derivation is deterministic and total: no window can make it panic or
        /// return a different answer the second time.
        #[test]
        fn derivation_is_deterministic(
            declared in geometry(vec![RowMethod::XorModulo, RowMethod::EosXorModulo]),
            history in prop::collection::vec(-3i64..70, 0..8),
            queries in prop::collection::vec(-3i64..70, 1..6),
        ) {
            let (index, table) = build(&declared);
            let mut first = Vec::new();
            let mut second = Vec::new();
            query_row_ids(&index, &table, &history, &queries, &mut first);
            query_row_ids(&index, &table, &history, &queries, &mut second);
            prop_assert_eq!(first, second);
        }
    }
}
