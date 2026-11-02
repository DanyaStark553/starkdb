//! Hybrid Three-Channel Retrieval (`bound_attack`) and Reusable Thread-Local Scratchpad.
//!
//! Combines:
//! - Channel V: Two-Phase HNSW (SQ8 coarse + f16/f32 exact rerank) or selective brute-force scan
//! - Channel T: Resident BM25F lexical search normalized by exact theoretical upper bound $UB(q)$
//! - Channel G: Multi-source spreading activation (`activate_ids`) excluding seed self-energy
//! - Channel R: Exponential recency decay $\exp(-\Delta t / 7\text{d})$

use crate::format::FLAG_DELETED;
use crate::graph::{StarkGraph, TelemetryMode};
use crate::vectors::{dot_product_f16, dot_product_f32, quantize_sq8};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

pub const MAX_SCRATCH_DIMS: usize = 2048;
const TAU_V: f32 = 0.25;
const RECENCY_HALF_LIFE_SECS: f32 = 7.0 * 86_400.0; // 7 days in seconds

/// Fusion weights for the four scoring signals in `bound_attack`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FusionWeights {
    pub w_v: f32,
    pub w_t: f32,
    pub w_g: f32,
    pub w_r: f32,
}

impl Default for FusionWeights {
    fn default() -> Self {
        Self {
            w_v: 0.45,
            w_t: 0.30,
            w_g: 0.20,
            w_r: 0.05,
        }
    }
}

/// Structured input query for `StarkGraph::bound_attack`.
#[derive(Debug, Clone)]
pub struct BoundQuery<'a> {
    pub vector: Option<&'a [f32]>,
    pub text: Option<&'a str>,
    pub anchors: &'a [u32],
    pub tags_all: &'a [&'a str],
    pub node_type: Option<u16>,
    pub since: Option<u32>,
    pub until: Option<u32>,
    pub top_k: usize,
    pub depth: u8,
    pub weights: FusionWeights,
}

impl<'a> Default for BoundQuery<'a> {
    fn default() -> Self {
        Self {
            vector: None,
            text: None,
            anchors: &[],
            tags_all: &[],
            node_type: None,
            since: None,
            until: None,
            top_k: 10,
            depth: 2,
            weights: FusionWeights::default(),
        }
    }
}

/// Single fused result item returned by `bound_attack`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundHit {
    pub node_id: u32,
    pub name: String,
    pub score: f32,
    pub s_v: f32,
    pub s_t: f32,
    pub s_g: f32,
    pub s_r: f32,
    pub depth: usize,
}

/// Per-thread reusable scratchpad for zero-allocation steady-state query execution.
/// `stamp`, `allow_stamp`, and `acc` grow monotonically to `max_nodes` and are invalidated in $O(1)$ via `epoch`.
pub struct QueryScratch {
    pub epoch: u32,
    pub allow_epoch: u32,
    pub stamp: Vec<u32>,
    pub allow_stamp: Vec<u32>,
    /// Per-node channel accumulators: `[s_v, s_t, s_g, bfs_max_energy]`
    pub acc: Vec<[f32; 4]>,
    /// Per-node minimum BFS hop depth when reached by spreading activation
    pub hop_depth: Vec<u8>,
    pub touched: Vec<u32>,
    pub bfs: VecDeque<(u32, u8, f32)>,
    pub q_i8: Box<[i8; MAX_SCRATCH_DIMS]>,
}

impl Default for QueryScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryScratch {
    pub fn new() -> Self {
        Self {
            epoch: 1,
            allow_epoch: 1,
            stamp: Vec::new(),
            allow_stamp: Vec::new(),
            acc: Vec::new(),
            hop_depth: Vec::new(),
            touched: Vec::with_capacity(256),
            bfs: VecDeque::with_capacity(256),
            q_i8: Box::new([0i8; MAX_SCRATCH_DIMS]),
        }
    }

    /// Prepares the scratchpad for a new query over a graph with `max_nodes` capacity.
    /// Zero-cost in steady state: increments `epoch` in $O(1)$ without zeroing `stamp` or `acc`.
    #[inline]
    pub fn prepare(&mut self, max_nodes: usize) {
        if self.stamp.len() < max_nodes {
            self.stamp.resize(max_nodes, 0);
            self.allow_stamp.resize(max_nodes, 0);
            self.acc.resize(max_nodes, [0.0; 4]);
            self.hop_depth.resize(max_nodes, 0);
        }
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.stamp.fill(0);
            self.epoch = 1;
        }
        self.touched.clear();
        self.bfs.clear();
    }

    #[inline]
    pub fn next_allow_epoch(&mut self) -> u32 {
        self.allow_epoch = self.allow_epoch.wrapping_add(1);
        if self.allow_epoch == 0 {
            self.allow_stamp.fill(0);
            self.allow_epoch = 1;
        }
        self.allow_epoch
    }

    /// Marks `node_id` as touched in the current `epoch`, resetting its accumulator on first visit.
    /// Returns `true` if this was the first visit in the current `epoch`.
    #[inline]
    pub fn touch(&mut self, node_id: u32) -> bool {
        let idx = node_id as usize;
        if idx >= self.stamp.len() {
            let new_len = (idx + 1).next_power_of_two();
            self.stamp.resize(new_len, 0);
            self.allow_stamp.resize(new_len, 0);
            self.acc.resize(new_len, [0.0; 4]);
            self.hop_depth.resize(new_len, 0);
        }
        if self.stamp[idx] != self.epoch {
            self.stamp[idx] = self.epoch;
            self.acc[idx] = [0.0; 4];
            self.hop_depth[idx] = 0;
            self.touched.push(node_id);
            true
        } else {
            false
        }
    }

    #[inline]
    pub fn is_touched(&self, node_id: u32) -> bool {
        self.stamp
            .get(node_id as usize)
            .map(|&s| s == self.epoch)
            .unwrap_or(false)
    }
}

thread_local! {
    static THREAD_SCRATCH: RefCell<QueryScratch> = RefCell::new(QueryScratch::new());
}

/// Runs a closure with the current thread's reusable `QueryScratch`.
#[inline]
pub fn with_scratch<F, R>(max_nodes: usize, f: F) -> R
where
    F: FnOnce(&mut QueryScratch) -> R,
{
    THREAD_SCRATCH.with(|cell| {
        let mut borrow = cell.borrow_mut();
        borrow.prepare(max_nodes);
        f(&mut borrow)
    })
}

impl StarkGraph {
    /// Executes the Three-Channel `bound_attack` hybrid retrieval algorithm.
    /// Operates purely over `u32` `node_id`s in `QueryScratch`, materializing `String` names
    /// only for the final `top_k` results.
    pub fn bound_attack(&self, q: &BoundQuery<'_>) -> Result<Vec<BoundHit>> {
        if q.top_k == 0 {
            return Ok(Vec::new());
        }

        let nodes = self.storage.nodes();
        let max_nodes = nodes.len().max(self.storage.header().max_nodes as usize);
        if max_nodes == 0 {
            return Ok(Vec::new());
        }

        let now_sec = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as u32)
            .unwrap_or(0);

        with_scratch(max_nodes, |scratch| {
            // 1. Tag filter setup in O(sum |P_tag|) with O(1) stamp lookup and zero heap sort
            let has_tag_filter = !q.tags_all.is_empty();
            let mut active_allow_epoch = 0u32;
            let mut tag_allowed_count = 0usize;
            let mut single_tag_slice: &[u32] = &[];

            if has_tag_filter {
                let first_slice = self.bm25.postings_for_tag(q.tags_all[0]);
                if first_slice.is_empty() {
                    return Ok(Vec::new());
                }
                let mut cur_ep = scratch.next_allow_epoch();
                for &nid in first_slice {
                    if let Some(slot) = scratch.allow_stamp.get_mut(nid as usize) {
                        *slot = cur_ep;
                    }
                }
                tag_allowed_count = first_slice.len();
                if q.tags_all.len() == 1 {
                    single_tag_slice = first_slice;
                } else {
                    for &tag in &q.tags_all[1..] {
                        let next_slice = self.bm25.postings_for_tag(tag);
                        if next_slice.is_empty() {
                            return Ok(Vec::new());
                        }
                        let prev_ep = cur_ep;
                        cur_ep = scratch.next_allow_epoch();
                        let mut matched = 0usize;
                        for &nid in next_slice {
                            if let Some(slot) = scratch.allow_stamp.get_mut(nid as usize) {
                                if *slot == prev_ep {
                                    *slot = cur_ep;
                                    matched += 1;
                                }
                            }
                        }
                        if matched == 0 {
                            return Ok(Vec::new());
                        }
                        tag_allowed_count = matched;
                    }
                }
                active_allow_epoch = cur_ep;
            }

            let total_active = (self.bm25.n_docs as usize).max(self.hnsw.active_count).max(1);
            let sigma: f32 = if has_tag_filter {
                (tag_allowed_count as f32 / total_active as f32).clamp(0.001, 1.0)
            } else if q.node_type.is_some() || q.since.is_some() || q.until.is_some() {
                0.25
            } else {
                1.0
            };

            let check_node_meta = |nid: u32| -> bool {
                let Some(n) = nodes.get(nid as usize) else {
                    return false;
                };
                if (n.flags & FLAG_DELETED) != 0 {
                    return false;
                }
                if let Some(nt) = q.node_type {
                    if n.node_type != nt {
                        return false;
                    }
                }
                if let Some(since_ts) = q.since {
                    if n.created_at < since_ts {
                        return false;
                    }
                }
                if let Some(until_ts) = q.until {
                    if n.created_at > until_ts {
                        return false;
                    }
                }
                true
            };

            let mut has_v_channel = false;
            let mut has_t_channel = false;
            let mut has_g_channel = false;

            // 2. Channel V: Vector Search (Two-Phase HNSW SQ8 or selective scan)
            if let Some(vec_q) = q.vector {
                let norm_sq: f32 = vec_q.iter().map(|&x| x * x).sum();
                if norm_sq > 0.0 {
                    has_v_channel = true;
                    let norm_buf: Vec<f32>;
                    let q_norm: &[f32] = if (norm_sq - 1.0).abs() < 1e-4 {
                        vec_q
                    } else {
                        let inv = 1.0 / norm_sq.sqrt();
                        norm_buf = vec_q.iter().map(|&x| x * inv).collect();
                        &norm_buf
                    };

                    let k_v = (q.top_k * 2).max(16);
                    let is_f16 = self.storage.header().vector_format == 1;
                    let estimated_candidates = (sigma * (self.hnsw.active_count as f32)) as usize;

                    if (has_tag_filter && q.tags_all.len() == 1 && single_tag_slice.len() <= 256)
                        || (estimated_candidates <= 128 && sigma < 1.0)
                        || self.hnsw.active_count < 256
                    {
                        let mut v_hits: smallvec::SmallVec<[(f32, u32); 64]> =
                            smallvec::SmallVec::new();
                        if has_tag_filter && q.tags_all.len() == 1 {
                            // Fast SQ8 two-phase selective scan over single_tag_slice when SQ8 is ready
                            if self.storage.has_sq8() && q_norm.len() <= MAX_SCRATCH_DIMS {
                                let dims = q_norm.len();
                                let q_meta = quantize_sq8(q_norm, &mut scratch.q_i8[..dims]);
                                let mut coarse: smallvec::SmallVec<[(f32, u32); 256]> =
                                    smallvec::SmallVec::new();
                                for &nid in single_tag_slice {
                                    if !check_node_meta(nid) {
                                        continue;
                                    }
                                    let vid = nodes[nid as usize].vector_id;
                                    if vid == u32::MAX {
                                        continue;
                                    }
                                    if let Ok((v_i8, v_meta)) = self.storage.get_vector_sq8(vid) {
                                        let dot = crate::vectors::dot_sq8(
                                            &scratch.q_i8[..dims],
                                            v_i8,
                                        );
                                        let sim = (dot as f32) * q_meta.scale * v_meta.scale;
                                        if sim > TAU_V * 0.8 {
                                            coarse.push((sim, nid));
                                        }
                                    }
                                }
                                let rerank_k = k_v.min(coarse.len());
                                if coarse.len() > rerank_k && rerank_k > 0 {
                                    coarse.select_nth_unstable_by(rerank_k - 1, |a, b| {
                                        b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal)
                                    });
                                    coarse.truncate(rerank_k);
                                }
                                for (_, nid) in coarse {
                                    let vid = nodes[nid as usize].vector_id;
                                    let sim = if is_f16 {
                                        self.storage
                                            .get_vector_f16(vid)
                                            .map(|v| dot_product_f16(q_norm, v))
                                            .unwrap_or(-1.0)
                                    } else {
                                        self.storage
                                            .get_vector_f32(vid)
                                            .map(|v| dot_product_f32(q_norm, v))
                                            .unwrap_or(-1.0)
                                    };
                                    if sim > TAU_V {
                                        v_hits.push((sim, nid));
                                    }
                                }
                            } else {
                                for &nid in single_tag_slice {
                                    if !check_node_meta(nid) {
                                        continue;
                                    }
                                    let vid = nodes[nid as usize].vector_id;
                                    if vid == u32::MAX {
                                        continue;
                                    }
                                    let sim = if is_f16 {
                                        self.storage
                                            .get_vector_f16(vid)
                                            .map(|v| dot_product_f16(q_norm, v))
                                            .unwrap_or(-1.0)
                                    } else {
                                        self.storage
                                            .get_vector_f32(vid)
                                            .map(|v| dot_product_f32(q_norm, v))
                                            .unwrap_or(-1.0)
                                    };
                                    if sim > TAU_V {
                                        v_hits.push((sim, nid));
                                    }
                                }
                            }
                        } else {
                            for (idx, n) in nodes.iter().enumerate() {
                                let nid = idx as u32;
                                if n.vector_id == u32::MAX || !check_node_meta(nid) {
                                    continue;
                                }
                                if has_tag_filter
                                    && scratch.allow_stamp.get(idx).copied().unwrap_or(0)
                                        != active_allow_epoch
                                {
                                    continue;
                                }
                                let sim = if is_f16 {
                                    self.storage
                                        .get_vector_f16(n.vector_id)
                                        .map(|v| dot_product_f16(q_norm, v))
                                        .unwrap_or(-1.0)
                                } else {
                                    self.storage
                                        .get_vector_f32(n.vector_id)
                                        .map(|v| dot_product_f32(q_norm, v))
                                        .unwrap_or(-1.0)
                                };
                                if sim > TAU_V {
                                    v_hits.push((sim, nid));
                                }
                            }
                        }
                        if v_hits.len() > k_v {
                            v_hits.select_nth_unstable_by(k_v - 1, |a, b| {
                                b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal)
                            });
                            v_hits.truncate(k_v);
                        }
                        for (sim, nid) in v_hits {
                            let s_v = ((sim - TAU_V) / (1.0 - TAU_V)).clamp(0.0, 1.0);
                            if s_v > 0.0 {
                                scratch.touch(nid);
                                scratch.acc[nid as usize][0] = s_v;
                            }
                        }
                    } else if self.storage.has_sq8() && q_norm.len() <= MAX_SCRATCH_DIMS {
                        let dims = q_norm.len();
                        let q_meta = quantize_sq8(q_norm, &mut scratch.q_i8[..dims]);
                        let allow_slice = &scratch.allow_stamp[..];
                        let raw_hits = self.hnsw.search_two_phase_filtered(
                            q_norm,
                            &scratch.q_i8[..dims],
                            q_meta.scale,
                            k_v,
                            &self.storage,
                            |nid| {
                                (!has_tag_filter
                                    || allow_slice.get(nid as usize).copied().unwrap_or(0)
                                        == active_allow_epoch)
                                    && check_node_meta(nid)
                            },
                        );
                        for (sim, nid) in raw_hits {
                            let s_v = ((sim - TAU_V) / (1.0 - TAU_V)).clamp(0.0, 1.0);
                            if s_v > 0.0 {
                                scratch.touch(nid);
                                scratch.acc[nid as usize][0] = s_v;
                            }
                        }
                    } else {
                        let allow_slice = &scratch.allow_stamp[..];
                        let raw_hits = self.hnsw.search_filtered(
                            q_norm,
                            k_v,
                            &self.storage,
                            |nid| {
                                (!has_tag_filter
                                    || allow_slice.get(nid as usize).copied().unwrap_or(0)
                                        == active_allow_epoch)
                                    && check_node_meta(nid)
                            },
                        );
                        for (sim, nid) in raw_hits {
                            let s_v = ((sim - TAU_V) / (1.0 - TAU_V)).clamp(0.0, 1.0);
                            if s_v > 0.0 {
                                scratch.touch(nid);
                                scratch.acc[nid as usize][0] = s_v;
                            }
                        }
                    }
                }
            }

            // 3. Channel T: Resident BM25F Search normalized by UB(q)
            if let Some(text_q) = q.text {
                if !text_q.trim().is_empty() {
                    has_t_channel = true;
                    let k_t = (q.top_k * 2).max(32);
                    // Borrow allow_stamp pointer for O(1) filter check inside search_into_filtered
                    let allow_ptr = scratch.allow_stamp.as_ptr();
                    let allow_len = scratch.allow_stamp.len();
                    let (n_terms, ub_q) =
                        self.bm25
                            .search_into_filtered(text_q, scratch, k_t, |nid| {
                                if has_tag_filter {
                                    let idx = nid as usize;
                                    if idx >= allow_len {
                                        return false;
                                    }
                                    // SAFETY: allow_stamp is not resized while checking existing node IDs < max_nodes
                                    let st = unsafe { *allow_ptr.add(idx) };
                                    if st != active_allow_epoch {
                                        return false;
                                    }
                                }
                                check_node_meta(nid)
                            });
                    if n_terms > 0 && ub_q > 0.0 {
                        let inv_ub = 1.0 / ub_q;
                        for &nid in &scratch.touched {
                            let raw = scratch.acc[nid as usize][1];
                            if raw > 0.0 {
                                scratch.acc[nid as usize][1] = (raw * inv_ub).clamp(0.0, 1.0);
                            }
                        }
                    }
                }
            }

            // 4. Channel G: Multi-Source Spreading Activation (`activate_ids`)
            if q.depth > 0 {
                // Collect top-8 preliminary seeds from V U T with E_0 = max(s_v, s_t)
                let mut prelim: smallvec::SmallVec<[(u32, f32); 16]> = smallvec::SmallVec::new();
                for &nid in &scratch.touched {
                    let [s_v, s_t, _, _] = scratch.acc[nid as usize];
                    let e0 = s_v.max(s_t);
                    if e0 > 0.0 {
                        prelim.push((nid, e0));
                    }
                }

                if prelim.len() > 8 {
                    prelim.select_nth_unstable_by(7, |a, b| {
                        b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
                    });
                    prelim.truncate(8);
                }
                prelim.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

                // Add explicit anchors with E_0 = 1.0
                for &anchor_id in q.anchors {
                    if let Some(existing) = prelim.iter_mut().find(|(id, _)| *id == anchor_id) {
                        existing.1 = 1.0;
                    } else {
                        prelim.push((anchor_id, 1.0));
                    }
                }

                if !prelim.is_empty() {
                    has_g_channel = true;
                    self.activate_ids(
                        &prelim,
                        q.depth.min(6),
                        0.75,
                        0.10,
                        scratch,
                        TelemetryMode::Sampled,
                    );
                }
            }

            // 5. Fusion across touched candidates (excluding anchors and missing channels)
            let w_v = if has_v_channel { q.weights.w_v } else { 0.0 };
            let w_t = if has_t_channel { q.weights.w_t } else { 0.0 };
            let w_g = if has_g_channel { q.weights.w_g } else { 0.0 };
            let w_r = q.weights.w_r;
            let denom = (w_v + w_t + w_g + w_r).max(1e-6);

            let mut candidates: Vec<(f32, u32)> = Vec::with_capacity(scratch.touched.len());
            for &nid in &scratch.touched {
                if q.anchors.contains(&nid) {
                    continue;
                }
                if has_tag_filter
                    && scratch
                        .allow_stamp
                        .get(nid as usize)
                        .copied()
                        .unwrap_or(0)
                        != active_allow_epoch
                {
                    continue;
                }
                if !check_node_meta(nid) {
                    continue;
                }
                let [s_v, s_t, s_g, _] = scratch.acc[nid as usize];
                if s_v <= 0.0 && s_t <= 0.0 && s_g <= 0.0 {
                    continue;
                }

                let n = &nodes[nid as usize];
                let last_act = self
                    .telemetry
                    .read_node(nid)
                    .map(|(_, la)| {
                        if la > 0 {
                            la
                        } else {
                            n.last_activated.max(n.created_at)
                        }
                    })
                    .unwrap_or_else(|| n.last_activated.max(n.created_at));

                let dt = now_sec.saturating_sub(last_act) as f32;
                let s_r = (-dt / RECENCY_HALF_LIFE_SECS).exp().clamp(0.0, 1.0);
                // Store s_r in acc[nid][3] for final reporting
                scratch.acc[nid as usize][3] = s_r;

                let fused = (w_v * s_v + w_t * s_t + w_g * s_g + w_r * s_r) / denom;
                candidates.push((fused, nid));
            }

            // 6. Top-K selection via O(C) quickselect (`select_nth_unstable_by`)
            let cmp = |a: &(f32, u32), b: &(f32, u32)| {
                b.0.partial_cmp(&a.0)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.1.cmp(&b.1))
            };
            if candidates.len() > q.top_k {
                candidates.select_nth_unstable_by(q.top_k - 1, cmp);
                candidates.truncate(q.top_k);
            }
            candidates.sort_by(cmp);

            // 7. Materialize String names ONLY for the final K results
            let mut hits = Vec::with_capacity(candidates.len());
            for (score, nid) in candidates {
                let n = &nodes[nid as usize];
                let name = self
                    .storage
                    .get_string(n.name_offset, n.name_len as u32)
                    .unwrap_or("Unknown")
                    .to_string();
                let [s_v, s_t, s_g, s_r] = scratch.acc[nid as usize];
                let depth = scratch.hop_depth[nid as usize] as usize;
                hits.push(BoundHit {
                    node_id: nid,
                    name,
                    score,
                    s_v,
                    s_t,
                    s_g,
                    s_r,
                    depth,
                });
            }

            Ok(hits)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::StarkStorage;
    use std::collections::HashSet;

    #[test]
    fn test_bound_attack_dedup_and_fusion() {
        let tmp_dir = std::env::temp_dir().join("starkdb_test_bound_attack_dedup");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let db_path = tmp_dir.join("bound.starkdb");

        let dims = 64usize;
        let mut db =
            StarkGraph::new(StarkStorage::open_or_create(&db_path, dims as u16).unwrap()).unwrap();

        let id_anchor = db
            .upsert_node(
                "server_root",
                Some("Entity"),
                Some("infra"),
                Some("core server anchor"),
                None,
                None,
            )
            .unwrap()
            .id;
        let id_target = db
            .upsert_node(
                "tauri_ipc_handler",
                Some("Software"),
                Some("rust,tauri"),
                Some("tauri ipc command handler"),
                None,
                None,
            )
            .unwrap()
            .id;
        let id_neighbor = db
            .upsert_node(
                "webview_bridge",
                Some("Software"),
                Some("rust,tauri"),
                Some("frontend bridge module"),
                None,
                None,
            )
            .unwrap()
            .id;

        // Connect anchor -> target AND target <-> neighbor so target is hit by V, T, and G simultaneously
        db.remember("server_root", "hosts", "tauri_ipc_handler", 0.9)
            .unwrap();
        db.remember("tauri_ipc_handler", "calls", "webview_bridge", 0.8)
            .unwrap();
        db.remember("webview_bridge", "returns", "tauri_ipc_handler", 0.8)
            .unwrap();

        let mut v_target = vec![0.0f32; dims];
        v_target[0] = 1.0;
        let mut v_neighbor = vec![0.0f32; dims];
        v_neighbor[0] = 0.8;
        v_neighbor[1] = 0.6;

        db.attach_vector(id_target, &v_target).unwrap();
        db.attach_vector(id_neighbor, &v_neighbor).unwrap();

        // 1. Query with both Vector + Text + Anchor + Tag filter
        let anchors = [id_anchor];
        let tags = ["tauri"];
        let q = BoundQuery {
            vector: Some(&v_target),
            text: Some("tauri ipc"),
            anchors: &anchors,
            tags_all: &tags,
            node_type: None,
            since: None,
            until: None,
            top_k: 10,
            depth: 2,
            weights: FusionWeights {
                w_v: 0.45,
                w_t: 0.30,
                w_g: 0.25,
                w_r: 0.0,
            },
        };

        let hits = db.bound_attack(&q).unwrap();
        assert!(!hits.is_empty());

        // Verify zero duplicate node_ids
        let mut seen = HashSet::new();
        for h in &hits {
            assert!(
                seen.insert(h.node_id),
                "Duplicate node_id {} in bound_attack output",
                h.node_id
            );
            // Verify anchor is excluded from results
            assert_ne!(h.node_id, id_anchor, "Anchor must be excluded from output");
        }

        // Top hit must be id_target (matches V=1.0, T>0, and G from anchor)
        assert_eq!(hits[0].node_id, id_target);
        assert!((hits[0].s_v - 1.0).abs() < 1e-4);
        assert!(hits[0].s_t > 0.0);
        assert!(hits[0].s_g > 0.0);

        // 2. Missing channel exclusion test: Text-only query (vector = None, depth = 0, w_r = 0)
        // Since only Channel T is active, fused score S(d) must equal s_t(d) (w_t / w_t = 1.0)
        let q_text_only = BoundQuery {
            vector: None,
            text: Some("tauri ipc"),
            anchors: &[],
            tags_all: &[],
            node_type: None,
            since: None,
            until: None,
            top_k: 10,
            depth: 0,
            weights: FusionWeights {
                w_v: 0.45,
                w_t: 0.30,
                w_g: 0.20,
                w_r: 0.0,
            },
        };
        let text_hits = db.bound_attack(&q_text_only).unwrap();
        assert!(!text_hits.is_empty());
        for h in &text_hits {
            assert!(
                (h.score - h.s_t).abs() < 1e-5,
                "When only Channel T is active and w_r=0, fused score ({}) must equal s_t ({})",
                h.score,
                h.s_t
            );
        }

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }
}
