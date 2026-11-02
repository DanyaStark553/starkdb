use crate::format::FLAG_DELETED;
use crate::graph::StarkGraph;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tracing::{info, warn};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuarantineRecord {
    pub timestamp: u64,
    pub entity_type: String, // "node" or "synapse"
    pub id: u32,
    pub reason: String,
    pub raw_offsets: String,
}

pub fn quarantine_path_for_db(db_path: &Path) -> PathBuf {
    let mut p = db_path.to_path_buf();
    let file_name = p.file_name().unwrap_or_default().to_string_lossy().to_string();
    p.set_file_name(format!("{}.quarantine.jsonl", file_name));
    p
}

pub fn append_quarantine_entry(db_path: &Path, record: &QuarantineRecord) -> Result<()> {
    use std::io::Write;
    let qpath = quarantine_path_for_db(db_path);
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&qpath)?;
    let line = serde_json::to_string(record)?;
    writeln!(file, "{}", line)?;
    Ok(())
}

pub fn count_quarantined_records(db_path: &Path) -> usize {
    use std::io::BufRead;
    let qpath = quarantine_path_for_db(db_path);
    if !qpath.exists() {
        return 0;
    }
    if let Ok(file) = std::fs::File::open(&qpath) {
        std::io::BufReader::new(file).lines().count()
    } else {
        0
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntegrityReport {
    pub healthy: bool,
    pub header_a_valid: bool,
    pub header_b_valid: bool,
    pub active_generation: u64,
    pub total_nodes: usize,
    pub valid_nodes: usize,
    pub corrupted_nodes: usize,
    pub total_synapses: usize,
    pub valid_synapses: usize,
    pub corrupted_synapses: usize,
    pub healed_count: usize,
    pub quarantined_total_count: usize,
    pub details: Vec<String>,
    pub duration_us: u128,
}

/// Flat cache-resident bitset for fast O(1) membership checks without hash allocations
struct NodeBitSet {
    words: Vec<u64>,
}

impl NodeBitSet {
    fn new(capacity: usize) -> Self {
        let num_words = capacity.div_ceil(64);
        Self {
            words: vec![0u64; num_words.max(1)],
        }
    }

    #[inline]
    fn insert(&mut self, id: u32) {
        let idx = (id as usize) / 64;
        if idx >= self.words.len() {
            self.words.resize(idx + 1, 0);
        }
        self.words[idx] |= 1u64 << ((id as usize) % 64);
    }

    #[inline]
    fn contains(&self, id: u32) -> bool {
        let idx = (id as usize) / 64;
        if idx < self.words.len() {
            (self.words[idx] & (1u64 << ((id as usize) % 64))) != 0
        } else {
            false
        }
    }
}

impl StarkGraph {
    /// Phase 1: High-speed read-only integrity verification.
    /// Operates under a shared read-lock and never mutates data on disk or in memory.
    pub fn check_integrity_readonly(&self) -> Result<IntegrityReport> {
        let start = Instant::now();
        let mut details = Vec::new();
        let mut corrupted_nodes = 0;
        let mut corrupted_synapses = 0;

        let hdr = *self.storage.header();
        let strings_bytes = hdr.strings_bytes;

        // 1. Validate Dual A/B Header integrity
        let (valid_a, valid_b) = self.storage.verify_dual_headers();
        if !valid_a || !valid_b {
            details.push(format!("Header state anomaly detected: Page A valid={}, Page B valid={}", valid_a, valid_b));
        }

        let node_count = hdr.node_count as usize;
        let synapse_count = hdr.synapse_count as usize;

        // 2. Read-only scan of Nodes using flat cache-friendly NodeBitSet
        let mut valid_nodes = 0;
        let mut active_node_ids = NodeBitSet::new(node_count);

        {
            let nodes = self.storage.nodes();
            for node in nodes.iter().take(node_count) {
                if (node.flags & FLAG_DELETED) != 0 {
                    continue;
                }

                let mut is_corrupt = false;

                if (node.name_offset as u64) + (node.name_len as u64) > strings_bytes {
                    is_corrupt = true;
                    details.push(format!("Node #{} has invalid name offset ({}+{} > {})", node.id, node.name_offset, node.name_len, strings_bytes));
                }

                if node.tags_len > 0 && (node.tags_offset as u64) + (node.tags_len as u64) > strings_bytes {
                    is_corrupt = true;
                    details.push(format!("Node #{} has invalid tags offset", node.id));
                }

                if node.content_len > 0 && (node.content_offset as u64) + (node.content_len as u64) > strings_bytes {
                    is_corrupt = true;
                    details.push(format!("Node #{} has invalid content offset", node.id));
                }

                if !node.base_weight.is_finite() || node.base_weight < 0.0 {
                    is_corrupt = true;
                    details.push(format!("Node #{} has non-finite or negative base_weight", node.id));
                }

                if is_corrupt {
                    corrupted_nodes += 1;
                } else {
                    valid_nodes += 1;
                    active_node_ids.insert(node.id);
                }
            }
        }

        // 3. Read-only scan of Synapses
        let mut valid_synapses = 0;
        {
            let synapses = self.storage.synapses();
            for syn in synapses.iter().take(synapse_count) {
                if (syn.flags & FLAG_DELETED) != 0 {
                    continue;
                }

                let mut is_corrupt = false;

                if !active_node_ids.contains(syn.source_id) {
                    is_corrupt = true;
                    details.push(format!("Synapse #{} references deleted or non-existent source #{}", syn.id, syn.source_id));
                }
                if !active_node_ids.contains(syn.target_id) {
                    is_corrupt = true;
                    details.push(format!("Synapse #{} references deleted or non-existent target #{}", syn.id, syn.target_id));
                }

                if (syn.rel_offset as u64) + (syn.rel_len as u64) > strings_bytes {
                    is_corrupt = true;
                    details.push(format!("Synapse #{} has invalid relation string offset", syn.id));
                }

                if !syn.weight.is_finite() || syn.weight < 0.0 {
                    is_corrupt = true;
                    details.push(format!("Synapse #{} has non-finite or negative weight", syn.id));
                }

                if is_corrupt {
                    corrupted_synapses += 1;
                } else {
                    valid_synapses += 1;
                }
            }
        }

        let quarantined_total_count = count_quarantined_records(&self.storage.path);
        let duration_us = start.elapsed().as_micros();
        let healthy = corrupted_nodes == 0 && corrupted_synapses == 0 && valid_a && valid_b;

        Ok(IntegrityReport {
            healthy,
            header_a_valid: valid_a,
            header_b_valid: valid_b,
            active_generation: self.storage.header().generation,
            total_nodes: node_count,
            valid_nodes,
            corrupted_nodes,
            total_synapses: synapse_count,
            valid_synapses,
            corrupted_synapses,
            healed_count: 0,
            quarantined_total_count,
            details,
            duration_us,
        })
    }

    /// Phase 2: Self-healing repair pass under exclusive write lock.
    /// Isolates corruptions, records persistent quarantine records, recycles slots into free lists,
    /// synchronizes headers, and rebuilds indices.
    pub fn scrub_and_self_heal(&mut self) -> Result<IntegrityReport> {
        let start = Instant::now();
        let mut details = Vec::new();
        let mut healed_count = 0;
        let mut corrupted_nodes = 0;
        let mut corrupted_synapses = 0;
        let db_path = self.storage.path.clone();

        let hdr = *self.storage.header();
        let strings_bytes = hdr.strings_bytes;

        // 1. Validate Dual A/B Header integrity
        let (valid_a, valid_b) = self.storage.verify_dual_headers();
        if !valid_a || !valid_b {
            warn!("Scrub: detected header page mismatch (A={}, B={}). Re-synchronizing headers...", valid_a, valid_b);
            self.storage.synchronize_headers()?;
            healed_count += 1;
            details.push("Synchronized dual header pages A and B to matching generation and CRC32.".to_string());
        }

        let node_count = hdr.node_count as usize;
        let synapse_count = hdr.synapse_count as usize;

        // 2. Scan and audit Nodes
        let mut valid_nodes = 0;
        let mut active_node_ids = NodeBitSet::new(node_count);
        let mut current_free_nodes_head = self.storage.header().free_nodes_head;

        {
            let nodes = self.storage.nodes_mut();
            for i in 0..nodes.len().min(node_count) {
                let node = &mut nodes[i];
                if (node.flags & FLAG_DELETED) != 0 {
                    continue;
                }

                let mut is_corrupt = false;
                let mut reasons = Vec::new();

                if (node.name_offset as u64) + (node.name_len as u64) > strings_bytes {
                    is_corrupt = true;
                    reasons.push(format!("invalid name offset ({}+{} > {})", node.name_offset, node.name_len, strings_bytes));
                }

                if node.tags_len > 0 && (node.tags_offset as u64) + (node.tags_len as u64) > strings_bytes {
                    is_corrupt = true;
                    reasons.push(format!("invalid tags offset ({}+{} > {})", node.tags_offset, node.tags_len, strings_bytes));
                }

                if node.content_len > 0 && (node.content_offset as u64) + (node.content_len as u64) > strings_bytes {
                    is_corrupt = true;
                    reasons.push(format!("invalid content offset ({}+{} > {})", node.content_offset, node.content_len, strings_bytes));
                }

                if !node.base_weight.is_finite() || node.base_weight < 0.0 {
                    node.base_weight = 1.0;
                    healed_count += 1;
                    details.push(format!("Node #{} had non-finite base_weight; repaired to 1.0", node.id));
                }

                if is_corrupt {
                    corrupted_nodes += 1;
                    healed_count += 1;

                    let raw_offsets = format!(
                        "name:{}+{}, tags:{}+{}, content:{}+{}, pool:{}",
                        node.name_offset, node.name_len,
                        node.tags_offset, node.tags_len,
                        node.content_offset, node.content_len,
                        strings_bytes
                    );
                    let record = QuarantineRecord {
                        timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
                        entity_type: "node".to_string(),
                        id: node.id,
                        reason: reasons.join("; "),
                        raw_offsets,
                    };
                    let _ = append_quarantine_entry(&db_path, &record);

                    // Slot recycling: link into free_nodes_head stack
                    node.flags |= FLAG_DELETED;
                    node.content_offset = current_free_nodes_head;
                    current_free_nodes_head = node.id;

                    details.push(format!("Self-healed & Quarantined: Isolated corrupted Node #{} and linked to free list", node.id));
                } else {
                    valid_nodes += 1;
                    active_node_ids.insert(node.id);
                }
            }
        }
        self.storage.header_mut().free_nodes_head = current_free_nodes_head;

        // 3. Scan and audit Synapses (Edges)
        let mut valid_synapses = 0;
        let mut current_free_synapses_head = self.storage.header().free_synapses_head;
        {
            let synapses = self.storage.synapses_mut();
            for i in 0..synapses.len().min(synapse_count) {
                let syn = &mut synapses[i];
                if (syn.flags & FLAG_DELETED) != 0 {
                    continue;
                }

                let mut is_corrupt = false;
                let mut reasons = Vec::new();

                if !active_node_ids.contains(syn.source_id) {
                    is_corrupt = true;
                    reasons.push(format!("source #{} is deleted or non-existent", syn.source_id));
                }
                if !active_node_ids.contains(syn.target_id) {
                    is_corrupt = true;
                    reasons.push(format!("target #{} is deleted or non-existent", syn.target_id));
                }

                if (syn.rel_offset as u64) + (syn.rel_len as u64) > strings_bytes {
                    is_corrupt = true;
                    reasons.push(format!("invalid relation offset ({}+{} > {})", syn.rel_offset, syn.rel_len, strings_bytes));
                }

                if !syn.weight.is_finite() || syn.weight < 0.0 {
                    syn.weight = 0.5;
                    healed_count += 1;
                    details.push(format!("Synapse #{} had non-finite weight; repaired to 0.5", syn.id));
                }

                if is_corrupt {
                    corrupted_synapses += 1;
                    healed_count += 1;

                    let raw_offsets = format!(
                        "source:{}, target:{}, rel:{}+{}, weight:{}",
                        syn.source_id, syn.target_id,
                        syn.rel_offset, syn.rel_len, syn.weight
                    );
                    let record = QuarantineRecord {
                        timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
                        entity_type: "synapse".to_string(),
                        id: syn.id,
                        reason: reasons.join("; "),
                        raw_offsets,
                    };
                    let _ = append_quarantine_entry(&db_path, &record);

                    // Slot recycling: link into free_synapses_head stack
                    syn.flags |= FLAG_DELETED;
                    syn.rel_offset = current_free_synapses_head;
                    current_free_synapses_head = syn.id;

                    details.push(format!("Self-healed & Quarantined: Isolated dangling Synapse #{} and linked to free list", syn.id));
                } else {
                    valid_synapses += 1;
                }
            }
        }
        self.storage.header_mut().free_synapses_head = current_free_synapses_head;

        // 4. If any repairs occurred, commit header, flush and rebuild indices
        if healed_count > 0 {
            info!("Scrub complete: repaired {} anomaly/anomalies in database. Rebuilding index...", healed_count);
            self.storage.commit_header()?;
            self.storage.flush()?;
            self.rebuild_indices();
        }

        let quarantined_total_count = count_quarantined_records(&self.storage.path);
        let duration_us = start.elapsed().as_micros();
        let healthy = corrupted_nodes == 0 && corrupted_synapses == 0;

        Ok(IntegrityReport {
            healthy,
            header_a_valid: valid_a,
            header_b_valid: valid_b,
            active_generation: self.storage.header().generation,
            total_nodes: node_count,
            valid_nodes,
            corrupted_nodes,
            total_synapses: synapse_count,
            valid_synapses,
            corrupted_synapses,
            healed_count,
            quarantined_total_count,
            details,
            duration_us,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::TOTAL_HEADER_SIZE;
    use crate::storage::StarkStorage;
    use std::io::{Seek, SeekFrom, Write};

    #[test]
    fn test_scrub_healthy_graph() {
        let temp_dir = std::env::temp_dir();
        let test_path = temp_dir.join(format!("test_starkdb_scrub_{}.starkdb", std::process::id()));
        let _ = std::fs::remove_file(&test_path);
        let _ = std::fs::remove_file(quarantine_path_for_db(&test_path));

        let storage = StarkStorage::open_or_create(&test_path, 128).unwrap();
        let mut graph = StarkGraph::new(storage).unwrap();

        graph.upsert_node("IronMan", Some("Hero"), Some("avenger"), Some("Tony Stark"), Some(1.0), Some(true)).unwrap();
        graph.upsert_node("Jarvis", Some("AI"), Some("system"), Some("Assistant"), Some(1.0), Some(true)).unwrap();
        graph.remember("IronMan", "created", "Jarvis", 0.95).unwrap();

        // Read-only check
        let ro_report = graph.check_integrity_readonly().unwrap();
        assert!(ro_report.healthy);
        assert_eq!(ro_report.healed_count, 0);
        assert_eq!(ro_report.corrupted_nodes, 0);
        assert_eq!(ro_report.corrupted_synapses, 0);

        // Mutating scrub
        let report = graph.scrub_and_self_heal().unwrap();
        assert!(report.healthy);
        assert_eq!(report.healed_count, 0);
        assert_eq!(report.corrupted_nodes, 0);
        assert_eq!(report.corrupted_synapses, 0);
        assert!(report.header_a_valid);
        assert!(report.header_b_valid);

        let _ = std::fs::remove_file(&test_path);
        let _ = std::fs::remove_file(quarantine_path_for_db(&test_path));
    }

    #[test]
    fn test_scrub_self_healing_anomalies() {
        let temp_dir = std::env::temp_dir();
        let test_path = temp_dir.join(format!("test_starkdb_disk_heal_{}.starkdb", std::process::id()));
        let qpath = quarantine_path_for_db(&test_path);
        let _ = std::fs::remove_file(&test_path);
        let _ = std::fs::remove_file(&qpath);

        // 1. Create a populated database
        {
            let storage = StarkStorage::open_or_create(&test_path, 128).unwrap();
            let mut graph = StarkGraph::new(storage).unwrap();
            graph.upsert_node("Thor", Some("God"), Some("asgard"), Some("Mjolnir"), Some(1.0), Some(true)).unwrap();
            graph.upsert_node("Loki", Some("God"), Some("mischief"), Some("Dagger"), Some(1.0), Some(false)).unwrap();
            graph.remember("Thor", "brother_of", "Loki", 0.8).unwrap();
        } // Drop: unlocks and closes file descriptor cleanly

        // 2. Corrupt bytes directly on physical disk via raw file descriptor
        {
            let mut raw_file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&test_path)
                .expect("Failed to open physical test file for raw corruption");
            
            // In NodeRecord (offset TOTAL_HEADER_SIZE = 512):
            // id: u32 (0..4), flags: u16 (4..6), node_type: u16 (6..8), name_offset: u32 (8..12)
            // So byte offset 520 (512 + 8) is the name_offset of Node #0.
            let corrupt_offset = TOTAL_HEADER_SIZE as u64 + 8;
            raw_file.seek(SeekFrom::Start(corrupt_offset)).unwrap();
            let bad_bytes: [u8; 4] = [0xFF, 0xFF, 0xFF, 0x00]; // 16,777,215 (out of bounds)
            raw_file.write_all(&bad_bytes).unwrap();
            raw_file.sync_all().unwrap();
        }

        // 3. Mount "cold" from disk and perform Phase 1 (Read-Only Check)
        {
            let storage = StarkStorage::open_or_create(&test_path, 128).unwrap();
            let graph = StarkGraph::new(storage).unwrap();
            let ro_report = graph.check_integrity_readonly().unwrap();
            assert!(!ro_report.healthy, "Read-only check must report database as unhealthy");
            assert_eq!(ro_report.corrupted_nodes, 1, "Must detect 1 corrupted node on disk");
            assert_eq!(ro_report.healed_count, 0, "Read-only check MUST NOT mutate or heal records");
        }

        // 4. Mount cold and perform Phase 2 (Self-Healing & Quarantine)
        {
            let storage = StarkStorage::open_or_create(&test_path, 128).unwrap();
            let mut graph = StarkGraph::new(storage).unwrap();

            let heal_report = graph.scrub_and_self_heal().unwrap();
            assert_eq!(heal_report.corrupted_nodes, 1);
            assert_eq!(heal_report.corrupted_synapses, 1);
            assert_eq!(heal_report.healed_count, 2);
            assert_eq!(heal_report.quarantined_total_count, 2);

            // Verify slot recycling: free_nodes_head points to Node #0 and free_synapses_head points to Synapse #0
            assert_eq!(graph.storage.header().free_nodes_head, 0);
            assert_eq!(graph.storage.header().free_synapses_head, 0);
            assert_eq!(graph.storage.nodes()[0].flags & FLAG_DELETED, FLAG_DELETED);
            assert_eq!(graph.storage.synapses()[0].flags & FLAG_DELETED, FLAG_DELETED);

            // Verify quarantine file exists and contains both quarantined entities
            assert!(qpath.exists(), "Quarantine log file must be created on disk");
            let qcontent = std::fs::read_to_string(&qpath).unwrap();
            assert!(qcontent.contains("\"id\":0"));
            assert!(qcontent.contains("\"entity_type\":\"node\""));
            assert!(qcontent.contains("\"entity_type\":\"synapse\""));
        }

        // 5. Cold remount verification: verify database is now clean and healthy
        {
            let storage = StarkStorage::open_or_create(&test_path, 128).unwrap();
            let graph = StarkGraph::new(storage).unwrap();
            let cold_report = graph.check_integrity_readonly().unwrap();
            assert!(cold_report.healthy, "Database must now be fully healthy on cold remount");
            assert_eq!(cold_report.corrupted_nodes, 0);
            assert_eq!(cold_report.corrupted_synapses, 0);
            assert_eq!(cold_report.quarantined_total_count, 2, "Quarantine history must be preserved");
        }

        let _ = std::fs::remove_file(&test_path);
        let _ = std::fs::remove_file(&qpath);
    }
}
