use crate::format::{
    node_type_to_str, str_to_node_type, NodeRecord, SynapseRecord, FLAG_DELETED, FLAG_PINNED,
};
use crate::storage::{current_timestamp, StarkStorage};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatabaseInfo {
    pub name: String,
    pub filename: String,
    pub active: bool,
    pub nodes: usize,
    pub synapses: usize,
    pub vectors: u32,
    pub size_bytes: usize,
    pub api_key: String,
    pub api_key_enabled: bool,
    pub allow_remote: bool,
    pub connection_uri: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DbConnectionConfig {
    pub api_key: String,
    pub api_key_enabled: bool,
    pub allow_remote: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EnvMetaStore {
    pub brain_dropped: bool,
    pub configs: HashMap<String, DbConnectionConfig>,
}

fn generate_api_key(db_name: &str) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(987654321);
    let mut hash: u64 = 0xcbf29ce484222325 ^ (nanos as u64);
    for b in db_name.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let h2 = hash.wrapping_mul(0x9e3779b97f4a7c15) ^ ((nanos >> 32) as u64);
    format!("sk-stark-{}-{:016x}{:08x}", db_name, hash, (h2 & 0xffffffff))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeView {
    pub id: u32,
    pub name: String,
    pub node_type: String,
    pub tags: Vec<String>,
    pub content: String,
    pub base_weight: f32,
    pub pinned: bool,
    pub has_vector: bool,
    pub created_at: u32,
    pub last_activated: u32,
    pub activation_count: u32,
    pub out_degree: usize,
    pub in_degree: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SynapseView {
    pub id: u32,
    pub source_id: u32,
    pub source_name: String,
    pub target_id: u32,
    pub target_name: String,
    pub relation: String,
    pub weight: f32,
    pub last_used: u32,
    pub activation_count: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActivatedNode {
    pub id: u32,
    pub name: String,
    pub node_type: String,
    pub tags: Vec<String>,
    pub content: String,
    pub energy: f32,
    pub depth: usize,
    pub path: Vec<String>,
}

#[derive(Clone, Copy, Debug)]
pub struct EdgeRef {
    pub synapse_id: u32,
    pub neighbor_id: u32,
    pub weight: f32,
    pub hot_score: f32,
}

#[inline]
pub fn compute_synapse_hot_score(weight: f32, activation_count: u32) -> f32 {
    weight + ((1.0 + activation_count as f32).ln() * 0.15)
}

#[repr(C)]
pub struct TelemetrySlot {
    pub hits: std::sync::atomic::AtomicU32,
    pub last_ts: std::sync::atomic::AtomicU32,
}

/// Lock-Free atomic telemetry table (`v0.7 & v0.8 Brave Evolution`)
/// Allows concurrent read-only `spreading_activation(&self)` threads under `RwLock::read()`
/// to increment `activation_count` and update `last_activated` / `last_used` without acquiring a write lock.
pub struct LockFreeTelemetry {
    pub node_slots: Vec<TelemetrySlot>,
    pub synapse_slots: Vec<TelemetrySlot>,
    pub dirty_ops: std::sync::atomic::AtomicU32,
}

impl Default for LockFreeTelemetry {
    fn default() -> Self {
        Self::new()
    }
}

impl LockFreeTelemetry {
    pub fn new() -> Self {
        Self {
            node_slots: Vec::new(),
            synapse_slots: Vec::new(),
            dirty_ops: std::sync::atomic::AtomicU32::new(0),
        }
    }

    pub fn sync_from_storage(&mut self, storage: &StarkStorage) {
        use std::sync::atomic::{AtomicU32, Ordering};
        let max_nodes = storage.header().max_nodes as usize;
        let max_synapses = storage.header().max_synapses as usize;
        let nodes = storage.nodes();
        let synapses = storage.synapses();

        self.node_slots.clear();
        self.node_slots.reserve(max_nodes);

        for i in 0..max_nodes {
            if let Some(n) = nodes.get(i) {
                self.node_slots.push(TelemetrySlot {
                    hits: AtomicU32::new(n.activation_count),
                    last_ts: AtomicU32::new(n.last_activated),
                });
            } else {
                self.node_slots.push(TelemetrySlot {
                    hits: AtomicU32::new(0),
                    last_ts: AtomicU32::new(0),
                });
            }
        }

        self.synapse_slots.clear();
        self.synapse_slots.reserve(max_synapses);

        for i in 0..max_synapses {
            if let Some(s) = synapses.get(i) {
                self.synapse_slots.push(TelemetrySlot {
                    hits: AtomicU32::new(s.activation_count),
                    last_ts: AtomicU32::new(s.last_used),
                });
            } else {
                self.synapse_slots.push(TelemetrySlot {
                    hits: AtomicU32::new(0),
                    last_ts: AtomicU32::new(0),
                });
            }
        }

        self.dirty_ops.store(0, Ordering::Relaxed);
    }

    #[inline]
    pub fn ensure_node_slot(&mut self, id: u32, max_nodes: u32, hits: u32, ts: u32) {
        use std::sync::atomic::{AtomicU32, Ordering};
        let target_len = (max_nodes as usize).max(id as usize + 1);
        while self.node_slots.len() < target_len {
            self.node_slots.push(TelemetrySlot {
                hits: AtomicU32::new(0),
                last_ts: AtomicU32::new(0),
            });
        }
        let slot = &self.node_slots[id as usize];
        slot.hits.store(hits, Ordering::Relaxed);
        slot.last_ts.store(ts, Ordering::Relaxed);
    }

    #[inline]
    pub fn ensure_synapse_slot(&mut self, id: u32, max_synapses: u32, hits: u32, ts: u32) {
        use std::sync::atomic::{AtomicU32, Ordering};
        let target_len = (max_synapses as usize).max(id as usize + 1);
        while self.synapse_slots.len() < target_len {
            self.synapse_slots.push(TelemetrySlot {
                hits: AtomicU32::new(0),
                last_ts: AtomicU32::new(0),
            });
        }
        let slot = &self.synapse_slots[id as usize];
        slot.hits.store(hits, Ordering::Relaxed);
        slot.last_ts.store(ts, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn mark_dirty(&self) {
        use std::sync::atomic::Ordering;
        if self.dirty_ops.load(Ordering::Relaxed) == 0 {
            self.dirty_ops.store(1, Ordering::Relaxed);
        }
    }

    #[inline(always)]
    pub fn record_node_hit(&self, id: u32, now: u32) {
        use std::sync::atomic::Ordering;
        if let Some(slot) = self.node_slots.get(id as usize) {
            slot.hits.fetch_add(1, Ordering::Relaxed);
            slot.last_ts.store(now, Ordering::Relaxed);
        }
    }

    #[inline(always)]
    pub fn record_node_hit_sampled(&self, id: u32, now: u32, sample_weight: u32) {
        use std::sync::atomic::Ordering;
        if let Some(slot) = self.node_slots.get(id as usize) {
            slot.hits.fetch_add(sample_weight, Ordering::Relaxed);
            slot.last_ts.store(now, Ordering::Relaxed);
        }
    }

    #[inline(always)]
    pub fn record_synapse_hit(&self, id: u32, now: u32) {
        use std::sync::atomic::Ordering;
        if let Some(slot) = self.synapse_slots.get(id as usize) {
            slot.hits.fetch_add(1, Ordering::Relaxed);
            slot.last_ts.store(now, Ordering::Relaxed);
        }
    }

    #[inline(always)]
    pub fn record_synapse_hit_sampled(&self, id: u32, now: u32, sample_weight: u32) {
        use std::sync::atomic::Ordering;
        if let Some(slot) = self.synapse_slots.get(id as usize) {
            slot.hits.fetch_add(sample_weight, Ordering::Relaxed);
            slot.last_ts.store(now, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn read_node(&self, id: u32) -> Option<(u32, u32)> {
        use std::sync::atomic::Ordering;
        self.node_slots.get(id as usize).map(|slot| {
            (
                slot.hits.load(Ordering::Relaxed),
                slot.last_ts.load(Ordering::Relaxed),
            )
        })
    }

    #[inline]
    pub fn get_node_stats(&self, id: u32, fallback_hits: u32, fallback_ts: u32) -> (u32, u32) {
        use std::sync::atomic::Ordering;
        if let Some(slot) = self.node_slots.get(id as usize) {
            let hits = slot.hits.load(Ordering::Relaxed).max(fallback_hits);
            let ts = slot.last_ts.load(Ordering::Relaxed).max(fallback_ts);
            (hits, ts)
        } else {
            (fallback_hits, fallback_ts)
        }
    }

    #[inline]
    pub fn get_synapse_stats(&self, id: u32, fallback_hits: u32, fallback_ts: u32) -> (u32, u32) {
        use std::sync::atomic::Ordering;
        if let Some(slot) = self.synapse_slots.get(id as usize) {
            let hits = slot.hits.load(Ordering::Relaxed).max(fallback_hits);
            let ts = slot.last_ts.load(Ordering::Relaxed).max(fallback_ts);
            (hits, ts)
        } else {
            (fallback_hits, fallback_ts)
        }
    }
}

/// Controls telemetry updates during graph traversal (`activate_ids`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TelemetryMode {
    None,
    Sampled,
    Exact,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompactionStats {
    pub original_size_bytes: usize,
    pub compacted_size_bytes: usize,
    pub bytes_saved: usize,
    pub active_nodes: usize,
    pub active_synapses: usize,
    pub vectors: u32,
}

/// Per-database handle with independent `tokio::sync::RwLock<StarkGraph>` and write-lock telemetry.
pub struct DbHandle {
    pub name: String,
    pub filename: String,
    pub graph: tokio::sync::RwLock<StarkGraph>,
    pub cached_nodes: std::sync::atomic::AtomicUsize,
    pub cached_synapses: std::sync::atomic::AtomicUsize,
    pub cached_vectors: std::sync::atomic::AtomicU32,
    pub cached_size_bytes: std::sync::atomic::AtomicUsize,
    pub last_write_ts: std::sync::atomic::AtomicU64,
    pub write_lock_samples_us: std::sync::Mutex<VecDeque<u64>>,
}

impl DbHandle {
    pub fn new(name: String, graph: StarkGraph) -> Self {
        let filename = graph
            .storage
            .path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("brain.starkdb")
            .to_string();
        let nodes = graph.active_node_count();
        let synapses = graph.active_synapse_count();
        let vectors = graph.storage.header().vector_count;
        let size_bytes = graph.storage.file_size_bytes();
        Self {
            name,
            filename,
            graph: tokio::sync::RwLock::new(graph),
            cached_nodes: std::sync::atomic::AtomicUsize::new(nodes),
            cached_synapses: std::sync::atomic::AtomicUsize::new(synapses),
            cached_vectors: std::sync::atomic::AtomicU32::new(vectors),
            cached_size_bytes: std::sync::atomic::AtomicUsize::new(size_bytes),
            last_write_ts: std::sync::atomic::AtomicU64::new(current_timestamp()),
            write_lock_samples_us: std::sync::Mutex::new(VecDeque::with_capacity(256)),
        }
    }

    pub fn refresh_cached_metrics(&self, g: &StarkGraph) {
        use std::sync::atomic::Ordering;
        self.cached_nodes
            .store(g.active_node_count(), Ordering::Relaxed);
        self.cached_synapses
            .store(g.active_synapse_count(), Ordering::Relaxed);
        self.cached_vectors
            .store(g.storage.header().vector_count, Ordering::Relaxed);
        self.cached_size_bytes
            .store(g.storage.file_size_bytes(), Ordering::Relaxed);
        self.last_write_ts
            .store(current_timestamp(), Ordering::Relaxed);
    }

    pub fn record_write_lock_us(&self, duration_us: u64) {
        if let Ok(mut guard) = self.write_lock_samples_us.lock() {
            if guard.len() >= 256 {
                guard.pop_front();
            }
            guard.push_back(duration_us);
        }
    }

    /// Returns `(p50_us, p99_us)` of recent write-lock durations on this database.
    pub fn write_lock_percentiles_us(&self) -> (u64, u64) {
        let Ok(guard) = self.write_lock_samples_us.lock() else {
            return (0, 0);
        };
        if guard.is_empty() {
            return (0, 0);
        }
        let mut sorted: Vec<u64> = guard.iter().copied().collect();
        sorted.sort_unstable();
        let p50 = sorted[sorted.len() / 2];
        let p99_idx = ((sorted.len() * 99) / 100).min(sorted.len() - 1);
        let p99 = sorted[p99_idx];
        (p50, p99)
    }
}

/// Helper to remove all physical segment files (`*.s*.starkdb`), manifest, and caches for a DB base path.
pub fn remove_all_db_artifacts(base_path: &Path) {
    let _ = std::fs::remove_file(base_path);
    let _ = std::fs::remove_file(StarkStorage::manifest_path_for(base_path));
    let _ = std::fs::remove_file(base_path.with_extension("hnsw.cache"));
    let _ = std::fs::remove_file(base_path.with_extension("bm25.cache"));
    if let (Some(parent), Some(stem)) = (
        base_path.parent(),
        base_path.file_stem().and_then(|s| s.to_str()),
    ) {
        let dir = if parent.as_os_str().is_empty() {
            Path::new(".")
        } else {
            parent
        };
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if let Some(fname) = p.file_name().and_then(|s| s.to_str()) {
                    let prefix = format!("{}.s", stem);
                    if fname.starts_with(&prefix) && fname.ends_with(".starkdb") {
                        let _ = std::fs::remove_file(&p);
                    }
                }
            }
        }
    }
}

/// Manages multiple independent `.starkdb` storage environments with per-DB locks (`Arc<DbHandle>`).
pub struct StarkEnvironmentManager {
    pub data_dir: PathBuf,
    pub vector_dims: u16,
    pub active_db_name: String,
    pub databases: HashMap<String, std::sync::Arc<DbHandle>>,
    pub meta: EnvMetaStore,
}

impl StarkEnvironmentManager {
    pub fn new<P: AsRef<Path>>(default_db_path: P, vector_dims: u16) -> Result<Self> {
        let default_path = default_db_path.as_ref();
        let data_dir = PathBuf::from("databases");
        let _ = std::fs::create_dir_all(&data_dir);

        let meta_path = data_dir.join("starkdb_meta.json");
        let mut meta: EnvMetaStore = if let Ok(raw) = std::fs::read_to_string(&meta_path) {
            serde_json::from_str(&raw).unwrap_or_default()
        } else {
            EnvMetaStore::default()
        };

        let mut databases: HashMap<String, std::sync::Arc<DbHandle>> = HashMap::new();

        // Load default brain.starkdb unless user explicitly dropped it
        let default_manifest = StarkStorage::manifest_path_for(default_path);
        if !meta.brain_dropped || default_path.exists() || default_manifest.exists() {
            let main_storage = StarkStorage::open_or_create(default_path, vector_dims)?;
            let main_graph = StarkGraph::new(main_storage)?;
            databases.insert(
                "brain".to_string(),
                std::sync::Arc::new(DbHandle::new("brain".to_string(), main_graph)),
            );
        }

        // Scan databases/ directory for all .starkdb environments (manifests, segment files, or legacy files)
        if let Ok(entries) = std::fs::read_dir(&data_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                let fname = match path.file_name().and_then(|s| s.to_str()) {
                    Some(f) => f,
                    None => continue,
                };
                let candidate_name = if let Some(prefix) = fname.strip_suffix(".starkdb.manifest") {
                    Some(prefix.to_string())
                } else if path.extension().and_then(|e| e.to_str()) == Some("starkdb") {
                    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                        if let Some((prefix, suffix)) = stem.rsplit_once(".s") {
                            if suffix.len() == 6 && suffix.chars().all(|c| c.is_ascii_digit()) {
                                Some(prefix.to_string())
                            } else {
                                Some(stem.to_string())
                            }
                        } else {
                            Some(stem.to_string())
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };

                if let Some(name) = candidate_name {
                    if name.is_empty() {
                        continue;
                    }
                    if let std::collections::hash_map::Entry::Vacant(e) =
                        databases.entry(name.clone())
                    {
                        let logical_path = data_dir.join(format!("{}.starkdb", name));
                        if let Ok(st) = StarkStorage::open_or_create(&logical_path, vector_dims) {
                            if let Ok(g) = StarkGraph::new(st) {
                                e.insert(std::sync::Arc::new(DbHandle::new(name, g)));
                            }
                        }
                    }
                }
            }
        }

        // Ensure every loaded DB has a connection config & API key
        for name in databases.keys() {
            meta.configs
                .entry(name.clone())
                .or_insert_with(|| DbConnectionConfig {
                    api_key: generate_api_key(name),
                    api_key_enabled: false,
                    allow_remote: false,
                });
        }

        let active_db_name = if databases.contains_key("brain") {
            "brain".to_string()
        } else {
            databases.keys().next().cloned().unwrap_or_default()
        };

        let mgr = Self {
            data_dir,
            vector_dims,
            active_db_name,
            databases,
            meta,
        };
        mgr.save_meta();
        Ok(mgr)
    }

    pub fn save_meta(&self) {
        let meta_path = self.data_dir.join("starkdb_meta.json");
        if let Ok(json) = serde_json::to_string_pretty(&self.meta) {
            let _ = std::fs::write(meta_path, json);
        }
    }

    pub fn has_active_graph(&self) -> bool {
        !self.active_db_name.is_empty() && self.databases.contains_key(&self.active_db_name)
    }

    pub fn active_handle(&self) -> Option<std::sync::Arc<DbHandle>> {
        self.databases.get(&self.active_db_name).cloned()
    }

    pub fn get_handle(&self, name: &str) -> Option<std::sync::Arc<DbHandle>> {
        self.databases.get(name.trim()).cloned()
    }

    pub fn resolve_db_by_api_key(&self, key: &str) -> Option<String> {
        let trimmed = key.trim();
        if trimmed.is_empty() {
            return None;
        }
        for (db_name, cfg) in &self.meta.configs {
            if cfg.api_key == trimmed && self.databases.contains_key(db_name) {
                return Some(db_name.clone());
            }
        }
        None
    }

    pub fn list_databases(&self) -> Vec<DatabaseInfo> {
        use std::sync::atomic::Ordering;
        let mut list: Vec<DatabaseInfo> = self
            .databases
            .iter()
            .map(|(name, handle)| {
                if let Ok(g) = handle.graph.try_read() {
                    handle.refresh_cached_metrics(&g);
                }
                let cfg = self.meta.configs.get(name);
                let api_key = cfg
                    .map(|c| c.api_key.clone())
                    .unwrap_or_else(|| generate_api_key(name));
                let api_key_enabled = cfg.map(|c| c.api_key_enabled).unwrap_or(false);
                let allow_remote = cfg.map(|c| c.allow_remote).unwrap_or(false);
                let host = if allow_remote { "0.0.0.0" } else { "127.0.0.1" };

                DatabaseInfo {
                    name: name.clone(),
                    filename: handle.filename.clone(),
                    active: *name == self.active_db_name,
                    nodes: handle.cached_nodes.load(Ordering::Relaxed),
                    synapses: handle.cached_synapses.load(Ordering::Relaxed),
                    vectors: handle.cached_vectors.load(Ordering::Relaxed),
                    size_bytes: handle.cached_size_bytes.load(Ordering::Relaxed),
                    api_key,
                    api_key_enabled,
                    allow_remote,
                    connection_uri: format!("starkdb://{}:9099/{}", host, name),
                }
            })
            .collect();
        list.sort_by(|a, b| a.name.cmp(&b.name));
        list
    }

    pub fn create_database(&mut self, raw_name: &str) -> Result<DatabaseInfo> {
        let clean: String = raw_name
            .trim()
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
            .collect();
        if clean.is_empty() {
            bail!("Invalid database name");
        }
        if self.databases.contains_key(&clean) {
            bail!("Database '{}' already exists", clean);
        }

        let path = if clean == "brain" {
            self.meta.brain_dropped = false;
            PathBuf::from("brain.starkdb")
        } else {
            self.data_dir.join(format!("{}.starkdb", clean))
        };

        let storage = StarkStorage::open_or_create(&path, self.vector_dims)?;
        let graph = StarkGraph::new(storage)?;
        self.databases.insert(
            clean.clone(),
            std::sync::Arc::new(DbHandle::new(clean.clone(), graph)),
        );
        self.active_db_name = clean.clone();

        self.meta
            .configs
            .entry(clean.clone())
            .or_insert_with(|| DbConnectionConfig {
                api_key: generate_api_key(&clean),
                api_key_enabled: false,
                allow_remote: false,
            });
        self.save_meta();

        Ok(self
            .list_databases()
            .into_iter()
            .find(|d| d.name == clean)
            .unwrap())
    }

    pub fn use_database(&mut self, name: &str) -> Result<DatabaseInfo> {
        let clean = name.trim();
        if !self.databases.contains_key(clean) {
            bail!("Database '{}' does not exist", clean);
        }
        self.active_db_name = clean.to_string();
        Ok(self
            .list_databases()
            .into_iter()
            .find(|d| d.name == clean)
            .unwrap())
    }

    pub fn drop_database(&mut self, name: &str) -> Result<String> {
        let clean = name.trim();
        let removed_handle = self
            .databases
            .remove(clean)
            .ok_or_else(|| anyhow::anyhow!("Database '{}' not found", clean))?;
        let path = {
            let g = removed_handle
                .graph
                .try_read()
                .map_err(|_| anyhow::anyhow!("Database '{}' is currently busy", clean))?;
            g.storage.path.clone()
        };
        drop(removed_handle);
        remove_all_db_artifacts(&path);

        if clean == "brain" {
            self.meta.brain_dropped = true;
            remove_all_db_artifacts(Path::new("brain.starkdb"));
            remove_all_db_artifacts(&self.data_dir.join("brain.starkdb"));
        }
        self.meta.configs.remove(clean);
        self.save_meta();

        if self.active_db_name == clean {
            let mut remaining: Vec<String> = self.databases.keys().cloned().collect();
            remaining.sort();
            self.active_db_name = remaining.first().cloned().unwrap_or_default();
        }
        Ok(clean.to_string())
    }

    pub fn truncate_database(&mut self, name: &str) -> Result<DatabaseInfo> {
        let clean = name.trim();
        let removed_handle = self
            .databases
            .remove(clean)
            .ok_or_else(|| anyhow::anyhow!("Database '{}' not found", clean))?;
        let path = {
            let g = removed_handle
                .graph
                .try_read()
                .map_err(|_| anyhow::anyhow!("Database '{}' is currently busy", clean))?;
            g.storage.path.clone()
        };
        drop(removed_handle);
        remove_all_db_artifacts(&path);

        let storage = StarkStorage::open_or_create(&path, self.vector_dims)?;
        let graph = StarkGraph::new(storage)?;
        self.databases.insert(
            clean.to_string(),
            std::sync::Arc::new(DbHandle::new(clean.to_string(), graph)),
        );
        self.active_db_name = clean.to_string();

        Ok(self
            .list_databases()
            .into_iter()
            .find(|d| d.name == clean)
            .unwrap())
    }

    pub fn compact_database(&mut self, name: &str) -> Result<CompactionStats> {
        let clean = name.trim();
        let handle = self
            .databases
            .get(clean)
            .ok_or_else(|| anyhow::anyhow!("Database '{}' not found", clean))?;
        let mut g = handle
            .graph
            .try_write()
            .map_err(|_| anyhow::anyhow!("Database '{}' is currently busy", clean))?;
        let stats = g.compact()?;
        handle.refresh_cached_metrics(&g);
        Ok(stats)
    }

    pub fn check_database_readonly(&self, name: &str) -> Result<crate::scrub::IntegrityReport> {
        let clean = name.trim();
        let handle = self
            .databases
            .get(clean)
            .ok_or_else(|| anyhow::anyhow!("Database '{}' not found", clean))?;
        let g = handle
            .graph
            .try_read()
            .map_err(|_| anyhow::anyhow!("Database '{}' is currently busy", clean))?;
        g.check_integrity_readonly()
    }

    pub fn scrub_database(&mut self, name: &str) -> Result<crate::scrub::IntegrityReport> {
        let clean = name.trim();
        let handle = self
            .databases
            .get(clean)
            .ok_or_else(|| anyhow::anyhow!("Database '{}' not found", clean))?;
        let mut g = handle
            .graph
            .try_write()
            .map_err(|_| anyhow::anyhow!("Database '{}' is currently busy", clean))?;
        let res = g.scrub_and_self_heal()?;
        handle.refresh_cached_metrics(&g);
        Ok(res)
    }

    pub fn update_db_config(
        &mut self,
        name: &str,
        api_key_enabled: bool,
        allow_remote: bool,
    ) -> Result<DatabaseInfo> {
        let clean = name.trim();
        if !self.databases.contains_key(clean) {
            bail!("Database '{}' does not exist", clean);
        }
        let entry = self
            .meta
            .configs
            .entry(clean.to_string())
            .or_insert_with(|| DbConnectionConfig {
                api_key: generate_api_key(clean),
                api_key_enabled,
                allow_remote,
            });
        entry.api_key_enabled = api_key_enabled;
        entry.allow_remote = allow_remote;
        self.save_meta();

        Ok(self
            .list_databases()
            .into_iter()
            .find(|d| d.name == clean)
            .unwrap())
    }

    pub fn regenerate_api_key(&mut self, name: &str) -> Result<DatabaseInfo> {
        let clean = name.trim();
        if !self.databases.contains_key(clean) {
            bail!("Database '{}' does not exist", clean);
        }
        let new_key = generate_api_key(clean);
        let entry = self
            .meta
            .configs
            .entry(clean.to_string())
            .or_insert_with(|| DbConnectionConfig {
                api_key: new_key.clone(),
                api_key_enabled: false,
                allow_remote: false,
            });
        entry.api_key = new_key;
        self.save_meta();

        Ok(self
            .list_databases()
            .into_iter()
            .find(|d| d.name == clean)
            .unwrap())
    }
}

/// Dirty-bitset change journal for non-blocking Shadow Vacuum (Milestone M5).
#[derive(Clone, Debug)]
pub struct VacuumJournal {
    pub snap_gen: u64,
    pub dirty_nodes: Vec<u64>,
    pub dirty_synapses: Vec<u64>,
    pub dirty_vectors: Vec<u64>,
    pub pending: u32,
    pub has_vector_ops: bool,
}

impl VacuumJournal {
    pub fn new(snap_gen: u64, max_nodes: u32, max_synapses: u32) -> Self {
        let n_words = (max_nodes as usize).div_ceil(64);
        let s_words = (max_synapses as usize).div_ceil(64);
        Self {
            snap_gen,
            dirty_nodes: vec![0u64; n_words.max(1)],
            dirty_synapses: vec![0u64; s_words.max(1)],
            dirty_vectors: vec![0u64; n_words.max(1)],
            pending: 0,
            has_vector_ops: false,
        }
    }

    #[inline]
    fn set_bit(bits: &mut Vec<u64>, id: u32) -> bool {
        let w = (id as usize) >> 6;
        let b = (id as usize) & 63;
        if w >= bits.len() {
            bits.resize(w + 1, 0);
        }
        let mask = 1u64 << b;
        let was_zero = (bits[w] & mask) == 0;
        bits[w] |= mask;
        was_zero
    }

    #[inline]
    pub fn mark_node(&mut self, id: u32) {
        if Self::set_bit(&mut self.dirty_nodes, id) {
            self.pending = self.pending.saturating_add(1);
        }
    }

    #[inline]
    pub fn mark_synapse(&mut self, id: u32) {
        if Self::set_bit(&mut self.dirty_synapses, id) {
            self.pending = self.pending.saturating_add(1);
        }
    }

    #[inline]
    pub fn mark_vector(&mut self, node_id: u32) {
        self.has_vector_ops = true;
        Self::set_bit(&mut self.dirty_vectors, node_id);
        self.mark_node(node_id);
    }

    fn drain_bits(bits: &mut [u64]) -> Vec<u32> {
        let mut out = Vec::new();
        for (w_idx, word) in bits.iter_mut().enumerate() {
            let mut v = *word;
            *word = 0;
            while v != 0 {
                let tz = v.trailing_zeros();
                out.push(((w_idx << 6) as u32) + tz);
                v &= v - 1;
            }
        }
        out
    }

    pub fn take_dirty(&mut self) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let nodes = Self::drain_bits(&mut self.dirty_nodes);
        let synapses = Self::drain_bits(&mut self.dirty_synapses);
        let vectors = Self::drain_bits(&mut self.dirty_vectors);
        self.pending = 0;
        self.has_vector_ops = false;
        (nodes, synapses, vectors)
    }
}

/// Pre-sorted adjacency plan prepared under read-lock for two-phase Hot-Tier adaptation.
pub struct HotTierPlan {
    pub generation: u64,
    pub adj_out: HashMap<u32, Vec<EdgeRef>>,
    pub adj_in: HashMap<u32, Vec<EdgeRef>>,
    pub hot_tier_nodes: Vec<u32>,
}

pub struct StarkGraph {
    pub storage: StarkStorage,
    name_to_id: HashMap<String, u32>,
    adj_out: HashMap<u32, Vec<EdgeRef>>,
    adj_in: HashMap<u32, Vec<EdgeRef>>,
    synapse_lookup: HashMap<(u32, u32, String), u32>,
    pub hnsw: crate::vectors::HnswIndex,
    pub bm25: crate::bm25::Bm25Index,
    pub telemetry: LockFreeTelemetry,
    pub hot_tier_nodes: Vec<u32>,
    pub vacuum: Option<VacuumJournal>,
}

impl StarkGraph {
    pub fn new(storage: StarkStorage) -> Result<Self> {
        let mut graph = Self {
            storage,
            name_to_id: HashMap::new(),
            adj_out: HashMap::new(),
            adj_in: HashMap::new(),
            synapse_lookup: HashMap::new(),
            hnsw: crate::vectors::HnswIndex::new(),
            bm25: crate::bm25::Bm25Index::new(),
            telemetry: LockFreeTelemetry::new(),
            hot_tier_nodes: Vec::new(),
            vacuum: None,
        };
        graph.rebuild_indices();
        Ok(graph)
    }

    /// Flushes any accumulated lock-free read telemetry (`activation_count` & `last_activated`)
    /// back into the underlying `mmap` records.
    pub fn sync_telemetry_to_mmap(&mut self) {
        use std::sync::atomic::Ordering;
        if self.telemetry.dirty_ops.swap(0, Ordering::Relaxed) == 0 {
            return;
        }
        let nodes = self.storage.nodes_mut();
        for (idx, node) in nodes.iter_mut().enumerate() {
            if (node.flags & FLAG_DELETED) != 0 {
                continue;
            }
            if let Some(slot) = self.telemetry.node_slots.get(idx) {
                let live_h = slot.hits.load(Ordering::Relaxed);
                if live_h > node.activation_count {
                    node.activation_count = live_h;
                }
                let live_t = slot.last_ts.load(Ordering::Relaxed);
                if live_t > node.last_activated {
                    node.last_activated = live_t;
                }
            }
        }

        let synapses = self.storage.synapses_mut();
        for (idx, syn) in synapses.iter_mut().enumerate() {
            if (syn.flags & FLAG_DELETED) != 0 {
                continue;
            }
            if let Some(slot) = self.telemetry.synapse_slots.get(idx) {
                let live_h = slot.hits.load(Ordering::Relaxed);
                if live_h > syn.activation_count {
                    syn.activation_count = live_h;
                }
                let live_t = slot.last_ts.load(Ordering::Relaxed);
                if live_t > syn.last_used {
                    syn.last_used = live_t;
                }
            }
        }
    }

    /// Phase 1 of Two-Phase Hot-Tier Adaptation (`&self` under read-lock):
    /// Computes updated `hot_score`s and sorts cloned adjacency lists without blocking readers.
    pub fn plan_hot_tier(&self) -> HotTierPlan {
        let generation = self.storage.header().generation;
        let synapses = self.storage.synapses();
        let mut new_out = self.adj_out.clone();
        for edges in new_out.values_mut() {
            for e in edges.iter_mut() {
                if let Some(syn) = synapses.get(e.synapse_id as usize) {
                    let (hits, _) = self.telemetry.get_synapse_stats(
                        e.synapse_id,
                        syn.activation_count,
                        syn.last_used,
                    );
                    e.weight = syn.weight;
                    e.hot_score = compute_synapse_hot_score(syn.weight, hits);
                }
            }
            if edges.len() > 1 {
                edges.sort_by(|a, b| {
                    b.hot_score
                        .partial_cmp(&a.hot_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        let mut new_in = self.adj_in.clone();
        for edges in new_in.values_mut() {
            for e in edges.iter_mut() {
                if let Some(syn) = synapses.get(e.synapse_id as usize) {
                    let (hits, _) = self.telemetry.get_synapse_stats(
                        e.synapse_id,
                        syn.activation_count,
                        syn.last_used,
                    );
                    e.weight = syn.weight;
                    e.hot_score = compute_synapse_hot_score(syn.weight, hits);
                }
            }
            if edges.len() > 1 {
                edges.sort_by(|a, b| {
                    b.hot_score
                        .partial_cmp(&a.hot_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        let mut node_scores: Vec<(u32, f32)> = self
            .storage
            .nodes()
            .iter()
            .filter(|n| (n.flags & FLAG_DELETED) == 0)
            .map(|n| {
                let (hits, _) =
                    self.telemetry
                        .get_node_stats(n.id, n.activation_count, n.last_activated);
                (n.id, compute_synapse_hot_score(n.base_weight, hits))
            })
            .collect();
        node_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        node_scores.truncate(64);
        let hot_tier_nodes = node_scores.into_iter().map(|(id, _)| id).collect();

        HotTierPlan {
            generation,
            adj_out: new_out,
            adj_in: new_in,
            hot_tier_nodes,
        }
    }

    /// Phase 2 of Two-Phase Hot-Tier Adaptation (`&mut self` under write-lock, microseconds):
    /// Applies the pre-sorted `HotTierPlan` via `std::mem::replace` if `generation` is unchanged.
    pub fn apply_hot_tier(&mut self, plan: HotTierPlan) -> bool {
        self.sync_telemetry_to_mmap();
        if self.storage.header().generation != plan.generation {
            return false;
        }
        self.adj_out = plan.adj_out;
        self.adj_in = plan.adj_in;
        self.hot_tier_nodes = plan.hot_tier_nodes;
        true
    }

    /// Brave Evolution Hot-Tier Adaptation (`v0.7 & v0.8`):
    /// Syncs lock-free telemetry and re-sorts all adjacency lists (`adj_out` & `adj_in`)
    /// in descending order of combined `hot_score` (`weight` + `activation_count`),
    /// promoting the hottest synapses to index 0 (first CPU L1 cache line).
    pub fn adapt_hot_tier(&mut self) {
        self.sync_telemetry_to_mmap();
        let plan = self.plan_hot_tier();
        let _ = self.apply_hot_tier(plan);
    }

    pub fn hnsw_cache_path(&self) -> PathBuf {
        self.storage.path.with_extension("hnsw.cache")
    }

    pub fn bm25_cache_path(&self) -> PathBuf {
        self.storage.path.with_extension("bm25.cache")
    }

    pub fn save_hnsw_cache(&self) -> Result<()> {
        let cache_path = self.hnsw_cache_path();
        let gen = self.storage.header().generation;
        let vec_cnt = self.storage.header().vector_count;
        let bytes = self.hnsw.serialize_cache(gen, vec_cnt);
        std::fs::write(&cache_path, bytes)
            .with_context(|| format!("Failed to write HNSW cache to {}", cache_path.display()))?;
        Ok(())
    }

    pub fn save_bm25_cache(&self) -> Result<()> {
        let cache_path = self.bm25_cache_path();
        let gen = self.storage.header().generation;
        self.bm25.save_to_file(&cache_path, gen)
    }

    pub fn try_load_hnsw_cache(&mut self, expected_gen: u64, expected_vec_cnt: u32) -> Result<bool> {
        let cache_path = self.hnsw_cache_path();
        if !cache_path.exists() {
            return Ok(false);
        }
        let bytes = match std::fs::read(&cache_path) {
            Ok(b) => b,
            Err(_) => return Ok(false),
        };
        self.hnsw.deserialize_cache(&bytes, expected_gen, expected_vec_cnt)
    }

    pub fn try_load_bm25_cache(&mut self, expected_gen: u64, expected_docs: u32) -> Result<bool> {
        let cache_path = self.bm25_cache_path();
        let loaded = self.bm25.load_from_file(&cache_path, expected_gen)?;
        Ok(loaded && self.bm25.n_docs == expected_docs)
    }

    pub fn total_telemetry_hits(&self) -> u64 {
        use std::sync::atomic::Ordering;
        let node_hits: u64 = self
            .telemetry
            .node_slots
            .iter()
            .map(|s| s.hits.load(Ordering::Relaxed) as u64)
            .sum();
        let syn_hits: u64 = self
            .telemetry
            .synapse_slots
            .iter()
            .map(|s| s.hits.load(Ordering::Relaxed) as u64)
            .sum();
        node_hits + syn_hits
    }

    pub fn outgoing_edges(&self, node_id: u32) -> Option<&[EdgeRef]> {
        self.adj_out.get(&node_id).map(|v| v.as_slice())
    }

    pub fn rebuild_indices(&mut self) {
        self.sync_telemetry_to_mmap();
        self.name_to_id.clear();
        self.adj_out.clear();
        self.adj_in.clear();
        self.synapse_lookup.clear();

        self.telemetry.sync_from_storage(&self.storage);

        let gen = self.storage.header().generation;
        let vec_cnt = self.storage.header().vector_count;
        let active_docs = self.active_node_count() as u32;
        let loaded_from_cache =
            (vec_cnt > 0) && self.try_load_hnsw_cache(gen, vec_cnt).unwrap_or(false);
        if !loaded_from_cache {
            self.hnsw.clear();
        }
        let loaded_bm25_cache =
            (active_docs > 0) && self.try_load_bm25_cache(gen, active_docs).unwrap_or(false);
        if !loaded_bm25_cache {
            self.bm25.clear();
        }

        let node_count = self.storage.header().node_count as usize;
        for idx in 0..node_count {
            let (
                flags,
                node_id,
                name_offset,
                name_len,
                tags_offset,
                tags_len,
                content_offset,
                content_len,
                vector_id,
            ) = {
                let node = &self.storage.nodes()[idx];
                (
                    node.flags,
                    node.id,
                    node.name_offset,
                    node.name_len,
                    node.tags_offset,
                    node.tags_len,
                    node.content_offset,
                    node.content_len,
                    node.vector_id,
                )
            };
            if (flags & FLAG_DELETED) != 0 {
                continue;
            }
            let name_str = self
                .storage
                .get_string(name_offset, name_len as u32)
                .unwrap_or("");
            if !name_str.is_empty() {
                self.name_to_id.insert(name_str.to_lowercase(), node_id);
            }
            if !loaded_bm25_cache {
                let tags_str = self
                    .storage
                    .get_string(tags_offset, tags_len as u32)
                    .unwrap_or("");
                let content_str = self
                    .storage
                    .get_string(content_offset, content_len)
                    .unwrap_or("");
                self.bm25.index_doc(node_id, name_str, tags_str, content_str);
            }
            if !loaded_from_cache && vector_id != u32::MAX {
                self.hnsw.insert(node_id, vector_id, &self.storage);
            }
        }

        if !loaded_from_cache && vec_cnt > 0 {
            let _ = self.save_hnsw_cache();
        }
        if !loaded_bm25_cache && active_docs > 0 {
            let _ = self.save_bm25_cache();
        }

        for syn in self.storage.synapses() {
            if (syn.flags & FLAG_DELETED) != 0 {
                continue;
            }

            if let Ok(rel) = self.storage.get_string(syn.rel_offset, syn.rel_len as u32) {
                let key = (syn.source_id, syn.target_id, rel.trim().to_lowercase());
                self.synapse_lookup.insert(key, syn.id);
            }

            let hot_score = compute_synapse_hot_score(syn.weight, syn.activation_count);

            self.adj_out
                .entry(syn.source_id)
                .or_default()
                .push(EdgeRef {
                    synapse_id: syn.id,
                    neighbor_id: syn.target_id,
                    weight: syn.weight,
                    hot_score,
                });

            self.adj_in
                .entry(syn.target_id)
                .or_default()
                .push(EdgeRef {
                    synapse_id: syn.id,
                    neighbor_id: syn.source_id,
                    weight: syn.weight,
                    hot_score,
                });
        }

        // Sort adjacency lists by descending hot_score (Brave Evolution Hot-Tier)
        self.adapt_hot_tier();

        // Reconstruct free lists if uninitialized
        if self.storage.header().free_nodes_head == u32::MAX {
            let deleted_nodes: Vec<u32> = self
                .storage
                .nodes()
                .iter()
                .enumerate()
                .filter(|(_, n)| (n.flags & FLAG_DELETED) != 0)
                .map(|(i, _)| i as u32)
                .collect();
            for idx in deleted_nodes {
                let old_head = self.storage.header().free_nodes_head;
                self.storage.nodes_mut()[idx as usize].content_offset = old_head;
                self.storage.header_mut().free_nodes_head = idx;
            }
        }
        if self.storage.header().free_synapses_head == u32::MAX {
            let deleted_synapses: Vec<u32> = self
                .storage
                .synapses()
                .iter()
                .enumerate()
                .filter(|(_, s)| (s.flags & FLAG_DELETED) != 0)
                .map(|(i, _)| i as u32)
                .collect();
            for idx in deleted_synapses {
                let old_head = self.storage.header().free_synapses_head;
                self.storage.synapses_mut()[idx as usize].rel_offset = old_head;
                self.storage.header_mut().free_synapses_head = idx;
            }
        }
    }

    pub fn active_node_count(&self) -> usize {
        self.storage
            .nodes()
            .iter()
            .filter(|n| (n.flags & FLAG_DELETED) == 0)
            .count()
    }

    pub fn active_synapse_count(&self) -> usize {
        self.storage
            .synapses()
            .iter()
            .filter(|s| (s.flags & FLAG_DELETED) == 0)
            .count()
    }

    pub fn seed_starter_brain(&mut self) -> Result<()> {
        Ok(())
    }

    pub fn get_or_create_node(&mut self, name: &str, node_type: u16) -> Result<u32> {
        let (id, _) = self.get_or_create_node_inner(name, node_type, true)?;
        Ok(id)
    }

    fn get_or_create_node_inner(
        &mut self,
        name: &str,
        node_type: u16,
        index_bm25: bool,
    ) -> Result<(u32, bool)> {
        let trimmed = name.trim();
        let key = trimmed.to_lowercase();
        if let Some(&id) = self.name_to_id.get(&key) {
            return Ok((id, true));
        }

        let new_id = if self.storage.header().free_nodes_head != u32::MAX {
            let recycled_id = self.storage.header().free_nodes_head;
            let next_free = self.storage.nodes()[recycled_id as usize].content_offset;
            self.storage.header_mut().free_nodes_head = next_free;
            recycled_id
        } else {
            self.storage.ensure_node_capacity(1)?;
            let id = self.storage.header().node_count;
            self.storage.header_mut().node_count += 1;
            id
        };

        let (name_offset, name_len) = self.storage.append_string(trimmed)?;
        let now = current_timestamp() as u32;

        let record = NodeRecord {
            id: new_id,
            node_type,
            flags: 0,
            name_offset,
            name_len,
            tags_len: 0,
            tags_offset: 0,
            content_offset: 0,
            content_len: 0,
            vector_id: u32::MAX,
            created_at: now,
            last_activated: now,
            base_weight: 0.85,
            activation_count: 1,
        };

        self.storage.nodes_mut()[new_id as usize] = record;
        self.telemetry
            .ensure_node_slot(new_id, self.storage.header().max_nodes, 1, now);
        self.name_to_id.insert(key, new_id);
        if index_bm25 {
            self.bm25.index_doc(new_id, trimmed, "", "");
        }
        if let Some(j) = &mut self.vacuum {
            j.mark_node(new_id);
        }

        Ok((new_id, false))
    }

    pub fn upsert_node(
        &mut self,
        name: &str,
        node_type: Option<&str>,
        tags: Option<&str>,
        content: Option<&str>,
        base_weight: Option<f32>,
        pinned: Option<bool>,
    ) -> Result<NodeView> {
        let t_code = node_type.map(str_to_node_type).unwrap_or(0);
        let (node_id, existed) = self.get_or_create_node_inner(name, t_code, false)?;

        // Read existing text before mutation to cleanly remove old BM25 postings
        if existed {
            let (old_name, old_tags, old_content) = {
                let n = &self.storage.nodes()[node_id as usize];
                let nm = self
                    .storage
                    .get_string(n.name_offset, n.name_len as u32)
                    .unwrap_or("")
                    .to_string();
                let tg = self
                    .storage
                    .get_string(n.tags_offset, n.tags_len as u32)
                    .unwrap_or("")
                    .to_string();
                let ct = self
                    .storage
                    .get_string(n.content_offset, n.content_len)
                    .unwrap_or("")
                    .to_string();
                (nm, tg, ct)
            };
            self.bm25
                .remove_doc(node_id, &old_name, &old_tags, &old_content);
        }

        let new_tags = if let Some(t) = tags {
            Some(self.storage.append_string(t.trim())?)
        } else {
            None
        };

        let new_content = if let Some(c) = content {
            Some(self.storage.append_payload(c.trim())?)
        } else {
            None
        };

        let now = current_timestamp() as u32;
        let max_nodes = self.storage.header().max_nodes;
        let live_hits = {
            let node = &mut self.storage.nodes_mut()[node_id as usize];
            node.flags &= !FLAG_DELETED;
            if node_type.is_some() {
                node.node_type = t_code;
            }
            if let Some((off, len)) = new_tags {
                node.tags_offset = off;
                node.tags_len = len;
            }
            if let Some((off, len)) = new_content {
                node.content_offset = off;
                node.content_len = len;
            }
            if let Some(w) = base_weight {
                node.base_weight = w.clamp(0.01, 1.0);
            }
            if let Some(pin) = pinned {
                if pin {
                    node.flags |= FLAG_PINNED;
                } else {
                    node.flags &= !FLAG_PINNED;
                }
            }
            node.last_activated = now;
            node.activation_count
        };
        let (synced_hits, _) = self.telemetry.get_node_stats(node_id, live_hits, now);
        self.storage.nodes_mut()[node_id as usize].activation_count = synced_hits;
        self.telemetry
            .ensure_node_slot(node_id, max_nodes, synced_hits, now);

        // Re-index updated fields into BM25
        {
            let n = &self.storage.nodes()[node_id as usize];
            let nm = self
                .storage
                .get_string(n.name_offset, n.name_len as u32)
                .unwrap_or("");
            let tg = self
                .storage
                .get_string(n.tags_offset, n.tags_len as u32)
                .unwrap_or("");
            let ct = self
                .storage
                .get_string(n.content_offset, n.content_len)
                .unwrap_or("");
            self.bm25.index_doc(node_id, nm, tg, ct);
        }
        if let Some(j) = &mut self.vacuum {
            j.mark_node(node_id);
        }

        self.storage.flush()?;
        self.get_node_view(node_id)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update_node_by_id(
        &mut self,
        node_id: u32,
        name: Option<&str>,
        node_type: Option<&str>,
        tags: Option<&str>,
        content: Option<&str>,
        base_weight: Option<f32>,
        pinned: Option<bool>,
    ) -> Result<NodeView> {
        if node_id >= self.storage.header().node_count {
            bail!("Node #{} does not exist", node_id);
        }

        let (old_name_str, old_tags_str, old_content_str, old_flags) = {
            let n = &self.storage.nodes()[node_id as usize];
            let nm = self
                .storage
                .get_string(n.name_offset, n.name_len as u32)
                .unwrap_or("")
                .to_string();
            let tg = self
                .storage
                .get_string(n.tags_offset, n.tags_len as u32)
                .unwrap_or("")
                .to_string();
            let ct = self
                .storage
                .get_string(n.content_offset, n.content_len)
                .unwrap_or("")
                .to_string();
            (nm, tg, ct, n.flags)
        };
        if (old_flags & FLAG_DELETED) != 0 {
            bail!("Node #{} is deleted", node_id);
        }

        self.bm25
            .remove_doc(node_id, &old_name_str, &old_tags_str, &old_content_str);

        let new_name = if let Some(n) = name {
            if !n.trim().is_empty() {
                Some((n.trim().to_string(), self.storage.append_string(n.trim())?))
            } else {
                None
            }
        } else {
            None
        };

        let new_tags = if let Some(t) = tags {
            Some(self.storage.append_string(t.trim())?)
        } else {
            None
        };

        let new_content = if let Some(c) = content {
            Some(self.storage.append_payload(c.trim())?)
        } else {
            None
        };

        let now = current_timestamp() as u32;
        let max_nodes = self.storage.header().max_nodes;
        let live_hits = {
            let node = &mut self.storage.nodes_mut()[node_id as usize];
            if let Some((_, (off, len))) = &new_name {
                node.name_offset = *off;
                node.name_len = *len;
            }
            if let Some(nt) = node_type {
                node.node_type = str_to_node_type(nt);
            }
            if let Some((off, len)) = new_tags {
                node.tags_offset = off;
                node.tags_len = len;
            }
            if let Some((off, len)) = new_content {
                node.content_offset = off;
                node.content_len = len;
            }
            if let Some(w) = base_weight {
                node.base_weight = w.clamp(0.01, 1.0);
            }
            if let Some(pin) = pinned {
                if pin {
                    node.flags |= FLAG_PINNED;
                } else {
                    node.flags &= !FLAG_PINNED;
                }
            }
            node.last_activated = now;
            node.activation_count
        };
        let (synced_hits, _) = self.telemetry.get_node_stats(node_id, live_hits, now);
        self.storage.nodes_mut()[node_id as usize].activation_count = synced_hits;
        self.telemetry
            .ensure_node_slot(node_id, max_nodes, synced_hits, now);

        if let Some((clean_name, _)) = new_name {
            self.name_to_id.remove(&old_name_str.to_lowercase());
            self.name_to_id.insert(clean_name.to_lowercase(), node_id);
        }

        {
            let n = &self.storage.nodes()[node_id as usize];
            let nm = self
                .storage
                .get_string(n.name_offset, n.name_len as u32)
                .unwrap_or("");
            let tg = self
                .storage
                .get_string(n.tags_offset, n.tags_len as u32)
                .unwrap_or("");
            let ct = self
                .storage
                .get_string(n.content_offset, n.content_len)
                .unwrap_or("");
            self.bm25.index_doc(node_id, nm, tg, ct);
        }
        if let Some(j) = &mut self.vacuum {
            j.mark_node(node_id);
        }

        self.storage.flush()?;
        self.get_node_view(node_id)
    }

    pub fn delete_node(&mut self, id_or_name: &str) -> Result<u32> {
        let target_id = if let Ok(id) = id_or_name.trim().parse::<u32>() {
            id
        } else if let Some(id) = self.find_node_id(id_or_name) {
            id
        } else {
            bail!("Node '{}' not found", id_or_name);
        };

        if target_id >= self.storage.header().node_count {
            bail!("Node #{} out of range", target_id);
        }

        let (
            node_flags,
            name_offset,
            name_len,
            tags_offset,
            tags_len,
            content_offset,
            content_len,
            old_vec_id,
        ) = {
            let node = &self.storage.nodes()[target_id as usize];
            (
                node.flags,
                node.name_offset,
                node.name_len,
                node.tags_offset,
                node.tags_len,
                node.content_offset,
                node.content_len,
                node.vector_id,
            )
        };
        if (node_flags & FLAG_DELETED) != 0 {
            return Ok(target_id);
        }

        // Remove document from BM25 before content_offset is overwritten by free_nodes_head
        {
            let old_name = self
                .storage
                .get_string(name_offset, name_len as u32)
                .unwrap_or("")
                .to_string();
            let old_tags = self
                .storage
                .get_string(tags_offset, tags_len as u32)
                .unwrap_or("")
                .to_string();
            let old_content = self
                .storage
                .get_string(content_offset, content_len)
                .unwrap_or("")
                .to_string();
            self.bm25
                .remove_doc(target_id, &old_name, &old_tags, &old_content);
            self.name_to_id.remove(&old_name.to_lowercase());
        }

        let old_free_head = self.storage.header().free_nodes_head;
        let max_nodes = self.storage.header().max_nodes;
        {
            let node = &mut self.storage.nodes_mut()[target_id as usize];
            node.flags |= FLAG_DELETED;
            node.content_offset = old_free_head;
            node.vector_id = u32::MAX;
        }
        self.storage.header_mut().free_nodes_head = target_id;
        self.storage.header_mut().dead_string_bytes = self
            .storage
            .header()
            .dead_string_bytes
            .saturating_add(name_len as u64 + tags_len as u64 + content_len as u64);

        if old_vec_id != u32::MAX {
            self.storage.release_vector(old_vec_id)?;
        }
        self.telemetry.ensure_node_slot(target_id, max_nodes, 0, 0);
        self.hnsw.mark_deleted(target_id);
        if let Some(j) = &mut self.vacuum {
            j.mark_node(target_id);
            if old_vec_id != u32::MAX {
                j.mark_vector(target_id);
            }
        }

        if let Some(out_edges) = self.adj_out.remove(&target_id) {
            for e in out_edges {
                let (rel_offset, rel_len) = {
                    let syn = &self.storage.synapses()[e.synapse_id as usize];
                    (syn.rel_offset, syn.rel_len)
                };
                let rel_str = self
                    .storage
                    .get_string(rel_offset, rel_len as u32)
                    .ok()
                    .map(|s| s.trim().to_lowercase());

                let old_syn_head = self.storage.header().free_synapses_head;
                {
                    let syn = &mut self.storage.synapses_mut()[e.synapse_id as usize];
                    syn.flags |= FLAG_DELETED;
                    syn.rel_offset = old_syn_head;
                }
                self.storage.header_mut().free_synapses_head = e.synapse_id;
                if let Some(j) = &mut self.vacuum {
                    j.mark_synapse(e.synapse_id);
                }

                if let Some(rel) = rel_str {
                    self.synapse_lookup.remove(&(target_id, e.neighbor_id, rel));
                }
                if let Some(in_edges) = self.adj_in.get_mut(&e.neighbor_id) {
                    in_edges.retain(|edge| edge.synapse_id != e.synapse_id);
                }
            }
        }

        if let Some(in_edges) = self.adj_in.remove(&target_id) {
            for e in in_edges {
                let (rel_offset, rel_len) = {
                    let syn = &self.storage.synapses()[e.synapse_id as usize];
                    (syn.rel_offset, syn.rel_len)
                };
                let rel_str = self
                    .storage
                    .get_string(rel_offset, rel_len as u32)
                    .ok()
                    .map(|s| s.trim().to_lowercase());

                let old_syn_head = self.storage.header().free_synapses_head;
                {
                    let syn = &mut self.storage.synapses_mut()[e.synapse_id as usize];
                    syn.flags |= FLAG_DELETED;
                    syn.rel_offset = old_syn_head;
                }
                self.storage.header_mut().free_synapses_head = e.synapse_id;
                if let Some(j) = &mut self.vacuum {
                    j.mark_synapse(e.synapse_id);
                }

                if let Some(rel) = rel_str {
                    self.synapse_lookup.remove(&(e.neighbor_id, target_id, rel));
                }
                if let Some(out_edges) = self.adj_out.get_mut(&e.neighbor_id) {
                    out_edges.retain(|edge| edge.synapse_id != e.synapse_id);
                }
            }
        }

        self.storage.flush()?;
        Ok(target_id)
    }

    pub fn remember(
        &mut self,
        source: &str,
        relation: &str,
        target: &str,
        weight: f32,
    ) -> Result<u32> {
        let source_id = self.get_or_create_node(source, 0)?;
        let target_id = self.get_or_create_node(target, 0)?;
        let rel_clean = relation.trim();
        let rel_lower = rel_clean.to_lowercase();
        let now = current_timestamp() as u32;

        let existing_id = self
            .synapse_lookup
            .get(&(source_id, target_id, rel_lower.clone()))
            .copied();

        if let Some(syn_id) = existing_id {
            let clamped_weight = weight.clamp(0.01, 1.0);
            let max_synapses = self.storage.header().max_synapses;
            let new_hits = {
                let syn = &mut self.storage.synapses_mut()[syn_id as usize];
                let (prev_hits, _) =
                    self.telemetry
                        .get_synapse_stats(syn_id, syn.activation_count, syn.last_used);
                let hits = prev_hits.saturating_add(1);
                syn.weight = clamped_weight;
                syn.last_used = now;
                syn.activation_count = hits;
                hits
            };
            self.telemetry
                .ensure_synapse_slot(syn_id, max_synapses, new_hits, now);
            let new_hot_score = compute_synapse_hot_score(clamped_weight, new_hits);

            if let Some(edges) = self.adj_out.get_mut(&source_id) {
                for e in edges.iter_mut() {
                    if e.synapse_id == syn_id {
                        e.weight = clamped_weight;
                        e.hot_score = new_hot_score;
                    }
                }
                if edges.len() > 1 {
                    edges.sort_by(|a, b| {
                        b.hot_score
                            .partial_cmp(&a.hot_score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
            }
            if let Some(edges) = self.adj_in.get_mut(&target_id) {
                for e in edges.iter_mut() {
                    if e.synapse_id == syn_id {
                        e.weight = clamped_weight;
                        e.hot_score = new_hot_score;
                    }
                }
                if edges.len() > 1 {
                    edges.sort_by(|a, b| {
                        b.hot_score
                            .partial_cmp(&a.hot_score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
            }
            if let Some(j) = &mut self.vacuum {
                j.mark_synapse(syn_id);
            }
            self.storage.flush()?;
            return Ok(syn_id);
        }

        let syn_id = if self.storage.header().free_synapses_head != u32::MAX {
            let recycled_id = self.storage.header().free_synapses_head;
            let next_free = self.storage.synapses()[recycled_id as usize].rel_offset;
            self.storage.header_mut().free_synapses_head = next_free;
            recycled_id
        } else {
            self.storage.ensure_synapse_capacity(1)?;
            let id = self.storage.header().synapse_count;
            self.storage.header_mut().synapse_count += 1;
            id
        };

        let (rel_offset, rel_len) = self.storage.append_string(rel_clean)?;

        let syn = SynapseRecord {
            id: syn_id,
            source_id,
            target_id,
            rel_offset,
            rel_len,
            flags: 0,
            weight: weight.clamp(0.01, 1.0),
            last_used: now,
            activation_count: 1,
        };

        self.storage.synapses_mut()[syn_id as usize] = syn;
        self.telemetry
            .ensure_synapse_slot(syn_id, self.storage.header().max_synapses, 1, now);
        let hot_score = compute_synapse_hot_score(syn.weight, 1);

        let out_edges = self.adj_out.entry(source_id).or_default();
        out_edges.push(EdgeRef {
            synapse_id: syn_id,
            neighbor_id: target_id,
            weight: syn.weight,
            hot_score,
        });
        if out_edges.len() > 1 {
            out_edges.sort_by(|a, b| {
                b.hot_score
                    .partial_cmp(&a.hot_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let in_edges = self.adj_in.entry(target_id).or_default();
        in_edges.push(EdgeRef {
            synapse_id: syn_id,
            neighbor_id: source_id,
            weight: syn.weight,
            hot_score,
        });
        if in_edges.len() > 1 {
            in_edges.sort_by(|a, b| {
                b.hot_score
                    .partial_cmp(&a.hot_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        self.synapse_lookup
            .insert((source_id, target_id, rel_lower), syn_id);
        if let Some(j) = &mut self.vacuum {
            j.mark_synapse(syn_id);
        }

        self.storage.flush()?;
        Ok(syn_id)
    }

    pub fn update_synapse(
        &mut self,
        synapse_id: u32,
        relation: Option<&str>,
        weight: Option<f32>,
    ) -> Result<SynapseView> {
        if synapse_id >= self.storage.header().synapse_count {
            bail!("Synapse #{} does not exist", synapse_id);
        }

        let old_syn = self.storage.synapses()[synapse_id as usize];
        if (old_syn.flags & FLAG_DELETED) != 0 {
            bail!("Synapse #{} is deleted", synapse_id);
        }

        let old_rel = self
            .storage
            .get_string(old_syn.rel_offset, old_syn.rel_len as u32)
            .ok()
            .map(|s| s.to_string());

        let new_rel = if let Some(r) = relation {
            if !r.trim().is_empty() {
                Some((r.trim().to_string(), self.storage.append_string(r.trim())?))
            } else {
                None
            }
        } else {
            None
        };

        let now = current_timestamp() as u32;
        let max_synapses = self.storage.header().max_synapses;
        let (final_weight, synced_hits) = {
            let syn = &mut self.storage.synapses_mut()[synapse_id as usize];
            if let Some((_, (off, len))) = &new_rel {
                syn.rel_offset = *off;
                syn.rel_len = *len;
            }
            if let Some(w) = weight {
                syn.weight = w.clamp(0.01, 1.0);
            }
            let (hits, _) =
                self.telemetry
                    .get_synapse_stats(synapse_id, syn.activation_count, syn.last_used);
            syn.activation_count = hits;
            syn.last_used = now;
            (syn.weight, hits)
        };
        self.telemetry
            .ensure_synapse_slot(synapse_id, max_synapses, synced_hits, now);
        let new_hot_score = compute_synapse_hot_score(final_weight, synced_hits);

        if let Some((clean_rel, _)) = new_rel {
            if let Some(old) = old_rel {
                self.synapse_lookup.remove(&(
                    old_syn.source_id,
                    old_syn.target_id,
                    old.to_lowercase(),
                ));
            }
            self.synapse_lookup.insert(
                (
                    old_syn.source_id,
                    old_syn.target_id,
                    clean_rel.to_lowercase(),
                ),
                synapse_id,
            );
        }

        if let Some(edges) = self.adj_out.get_mut(&old_syn.source_id) {
            for e in edges.iter_mut() {
                if e.synapse_id == synapse_id {
                    e.weight = final_weight;
                    e.hot_score = new_hot_score;
                }
            }
            if edges.len() > 1 {
                edges.sort_by(|a, b| {
                    b.hot_score
                        .partial_cmp(&a.hot_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }
        if let Some(edges) = self.adj_in.get_mut(&old_syn.target_id) {
            for e in edges.iter_mut() {
                if e.synapse_id == synapse_id {
                    e.weight = final_weight;
                    e.hot_score = new_hot_score;
                }
            }
            if edges.len() > 1 {
                edges.sort_by(|a, b| {
                    b.hot_score
                        .partial_cmp(&a.hot_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }
        if let Some(j) = &mut self.vacuum {
            j.mark_synapse(synapse_id);
        }

        self.storage.flush()?;
        self.get_synapse_view(synapse_id)
    }

    pub fn delete_synapse(&mut self, synapse_id: u32) -> Result<u32> {
        if synapse_id >= self.storage.header().synapse_count {
            bail!("Synapse #{} does not exist", synapse_id);
        }
        let (flags, source_id, target_id, rel_offset, rel_len) = {
            let syn = &self.storage.synapses()[synapse_id as usize];
            (
                syn.flags,
                syn.source_id,
                syn.target_id,
                syn.rel_offset,
                syn.rel_len,
            )
        };
        if (flags & FLAG_DELETED) != 0 {
            return Ok(synapse_id);
        }

        let old_head = self.storage.header().free_synapses_head;
        {
            let syn = &mut self.storage.synapses_mut()[synapse_id as usize];
            syn.flags |= FLAG_DELETED;
            syn.rel_offset = old_head;
        }
        self.storage.header_mut().free_synapses_head = synapse_id;
        if let Some(j) = &mut self.vacuum {
            j.mark_synapse(synapse_id);
        }

        if let Ok(rel) = self.storage.get_string(rel_offset, rel_len as u32) {
            self.synapse_lookup
                .remove(&(source_id, target_id, rel.trim().to_lowercase()));
        }

        if let Some(edges) = self.adj_out.get_mut(&source_id) {
            edges.retain(|e| e.synapse_id != synapse_id);
        }
        if let Some(edges) = self.adj_in.get_mut(&target_id) {
            edges.retain(|e| e.synapse_id != synapse_id);
        }

        self.storage.flush()?;
        Ok(synapse_id)
    }

    pub fn get_node_name(&self, node_id: u32) -> Result<String> {
        let nodes = self.storage.nodes();
        let node = nodes
            .get(node_id as usize)
            .ok_or_else(|| anyhow::anyhow!("Node {} not found", node_id))?;
        let name = self
            .storage
            .get_string(node.name_offset, node.name_len as u32)?;
        Ok(name.to_string())
    }

    pub fn get_node_view(&self, node_id: u32) -> Result<NodeView> {
        let nodes = self.storage.nodes();
        let node = nodes
            .get(node_id as usize)
            .ok_or_else(|| anyhow::anyhow!("Node {} not found", node_id))?;
        let name = self
            .storage
            .get_string(node.name_offset, node.name_len as u32)?
            .to_string();
        let tags_raw = self
            .storage
            .get_string(node.tags_offset, node.tags_len as u32)
            .unwrap_or("");
        let tags: Vec<String> = if tags_raw.is_empty() {
            Vec::new()
        } else {
            tags_raw
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        };
        let content = self
            .storage
            .get_string(node.content_offset, node.content_len)
            .unwrap_or("")
            .to_string();

        let out_degree = self.adj_out.get(&node.id).map(|v| v.len()).unwrap_or(0);
        let in_degree = self.adj_in.get(&node.id).map(|v| v.len()).unwrap_or(0);
        let (live_hits, live_ts) =
            self.telemetry
                .get_node_stats(node.id, node.activation_count, node.last_activated);

        Ok(NodeView {
            id: node.id,
            name,
            node_type: node_type_to_str(node.node_type).to_string(),
            tags,
            content,
            base_weight: node.base_weight,
            pinned: (node.flags & FLAG_PINNED) != 0,
            has_vector: node.vector_id != u32::MAX,
            created_at: node.created_at,
            last_activated: live_ts,
            activation_count: live_hits,
            out_degree,
            in_degree,
        })
    }

    pub fn get_synapse_view(&self, syn_id: u32) -> Result<SynapseView> {
        let synapses = self.storage.synapses();
        let syn = synapses
            .get(syn_id as usize)
            .ok_or_else(|| anyhow::anyhow!("Synapse {} not found", syn_id))?;
        let relation = self
            .storage
            .get_string(syn.rel_offset, syn.rel_len as u32)
            .unwrap_or("LINK")
            .to_string();
        let source_name = self.get_node_name(syn.source_id).unwrap_or_default();
        let target_name = self.get_node_name(syn.target_id).unwrap_or_default();
        let (live_hits, live_ts) =
            self.telemetry
                .get_synapse_stats(syn.id, syn.activation_count, syn.last_used);

        Ok(SynapseView {
            id: syn.id,
            source_id: syn.source_id,
            source_name,
            target_id: syn.target_id,
            target_name,
            relation,
            weight: syn.weight,
            last_used: live_ts,
            activation_count: live_hits,
        })
    }

    pub fn list_nodes(
        &self,
        search: Option<&str>,
        type_filter: Option<&str>,
        tag_filter: Option<&str>,
        limit: usize,
    ) -> Vec<NodeView> {
        let q = search.map(|s| s.trim().to_lowercase()).unwrap_or_default();
        let tf = type_filter
            .map(|s| s.trim().to_lowercase())
            .unwrap_or_default();
        let tag_f = tag_filter
            .map(|s| s.trim().to_lowercase())
            .unwrap_or_default();

        let mut list = Vec::new();
        for node in self.storage.nodes() {
            if (node.flags & FLAG_DELETED) != 0 {
                continue;
            }
            if let Ok(view) = self.get_node_view(node.id) {
                if !tf.is_empty() && tf != "all" && view.node_type.to_lowercase() != tf {
                    continue;
                }
                if !tag_f.is_empty()
                    && !view.tags.iter().any(|t| t.to_lowercase().contains(&tag_f))
                {
                    continue;
                }
                if !q.is_empty() {
                    let matched = view.name.to_lowercase().contains(&q)
                        || view.content.to_lowercase().contains(&q)
                        || view.tags.iter().any(|t| t.to_lowercase().contains(&q));
                    if !matched {
                        continue;
                    }
                }
                list.push(view);
                if list.len() >= limit {
                    break;
                }
            }
        }
        list
    }

    pub fn list_synapses(&self, search: Option<&str>, limit: usize) -> Vec<SynapseView> {
        let q = search.map(|s| s.trim().to_lowercase()).unwrap_or_default();
        let mut list = Vec::new();
        for syn in self.storage.synapses() {
            if (syn.flags & FLAG_DELETED) != 0 {
                continue;
            }
            if let Ok(view) = self.get_synapse_view(syn.id) {
                if !q.is_empty() {
                    let matched = view.source_name.to_lowercase().contains(&q)
                        || view.target_name.to_lowercase().contains(&q)
                        || view.relation.to_lowercase().contains(&q);
                    if !matched {
                        continue;
                    }
                }
                list.push(view);
                if list.len() >= limit {
                    break;
                }
            }
        }
        list
    }

    pub fn find_node_id(&self, name_or_query: &str) -> Option<u32> {
        let clean = name_or_query.trim().to_lowercase();
        if let Some(&id) = self.name_to_id.get(&clean) {
            return Some(id);
        }
        for (name, &id) in &self.name_to_id {
            if name.contains(&clean) || clean.contains(name) {
                return Some(id);
            }
        }
        None
    }

    /// 100% Read-Only Lock-Free Spreading Activation (`&self`) — `v1.0 Ultimate Legacy`.
    /// Multiple reader threads can execute this simultaneously under `RwLock::read()`.
    /// Uses first-visit deduplication and unbiased 1/8 probabilistic sampling (`(rng & 7) == 0 -> fetch_add(8)`)
    /// to eliminate cache-line ping-pong on hub nodes under 16+ concurrent reader threads.
    pub fn spreading_activation(
        &self,
        seed_concept: &str,
        max_depth: usize,
        decay_factor: f32,
        energy_threshold: f32,
    ) -> Result<Vec<ActivatedNode>> {
        let seed_id = match self.find_node_id(seed_concept) {
            Some(id) => id,
            None => return Ok(Vec::new()),
        };
        self.spreading_activation_by_id(seed_id, max_depth, decay_factor, energy_threshold)
    }

    /// Multi-source lock-free graph activation operating purely on `u32` `node_id`s inside `QueryScratch`.
    /// Excludes initial seed self-energy `E_0` from `scratch.acc[id][2]` (`s_g`) so only energy
    /// arriving via synapses (`depth >= 1`) contributes to the graph channel score in `bound_attack`.
    pub fn activate_ids(
        &self,
        seeds: &[(u32, f32)],
        max_depth: u8,
        decay_factor: f32,
        energy_threshold: f32,
        scratch: &mut crate::hybrid::QueryScratch,
        telemetry: TelemetryMode,
    ) -> usize {
        thread_local! {
            static TL_ACT_RNG: std::cell::Cell<u64> = const { std::cell::Cell::new(0x9E3779B97F4A7C15) };
        }
        let sample_weight: u32 = match telemetry {
            TelemetryMode::None => 0,
            TelemetryMode::Exact => 1,
            TelemetryMode::Sampled => {
                let hit = TL_ACT_RNG.with(|c| {
                    let s = c.get().wrapping_add(1);
                    c.set(s);
                    (s & 7) == 0
                });
                if hit { 8 } else { 0 }
            }
        };

        let now = current_timestamp() as u32;
        if sample_weight > 0 {
            self.telemetry.mark_dirty();
        }

        scratch.bfs.clear();
        for &(seed_id, e0) in seeds {
            if e0 <= 0.0 {
                continue;
            }
            let first = scratch.touch(seed_id);
            let idx = seed_id as usize;
            if e0 > scratch.acc[idx][3] {
                scratch.acc[idx][3] = e0;
            }
            scratch.bfs.push_back((seed_id, 0, e0));
            if first && sample_weight > 0 {
                self.telemetry
                    .record_node_hit_sampled(seed_id, now, sample_weight);
            }
        }

        while let Some((curr_id, depth, curr_energy)) = scratch.bfs.pop_front() {
            if depth >= max_depth {
                continue;
            }
            let next_depth = depth + 1;

            if let Some(out_edges) = self.adj_out.get(&curr_id) {
                for &edge in out_edges {
                    let next_energy = curr_energy * edge.weight * decay_factor;
                    if next_energy < energy_threshold {
                        continue;
                    }
                    let first = scratch.touch(edge.neighbor_id);
                    let nb_idx = edge.neighbor_id as usize;
                    if first && sample_weight > 0 {
                        self.telemetry
                            .record_synapse_hit_sampled(edge.synapse_id, now, sample_weight);
                        self.telemetry
                            .record_node_hit_sampled(edge.neighbor_id, now, sample_weight);
                    }
                    if next_energy > scratch.acc[nb_idx][2] {
                        scratch.acc[nb_idx][2] = next_energy;
                    }
                    if next_energy > scratch.acc[nb_idx][3] {
                        scratch.acc[nb_idx][3] = next_energy;
                        if scratch.hop_depth[nb_idx] == 0 || next_depth < scratch.hop_depth[nb_idx]
                        {
                            scratch.hop_depth[nb_idx] = next_depth;
                        }
                        scratch
                            .bfs
                            .push_back((edge.neighbor_id, next_depth, next_energy));
                    }
                }
            }

            if let Some(in_edges) = self.adj_in.get(&curr_id) {
                for &edge in in_edges {
                    let next_energy = curr_energy * edge.weight * decay_factor * 0.85;
                    if next_energy < energy_threshold {
                        continue;
                    }
                    let first = scratch.touch(edge.neighbor_id);
                    let nb_idx = edge.neighbor_id as usize;
                    if first && sample_weight > 0 {
                        self.telemetry
                            .record_synapse_hit_sampled(edge.synapse_id, now, sample_weight);
                        self.telemetry
                            .record_node_hit_sampled(edge.neighbor_id, now, sample_weight);
                    }
                    if next_energy > scratch.acc[nb_idx][2] {
                        scratch.acc[nb_idx][2] = next_energy;
                    }
                    if next_energy > scratch.acc[nb_idx][3] {
                        scratch.acc[nb_idx][3] = next_energy;
                        if scratch.hop_depth[nb_idx] == 0 || next_depth < scratch.hop_depth[nb_idx]
                        {
                            scratch.hop_depth[nb_idx] = next_depth;
                        }
                        scratch
                            .bfs
                            .push_back((edge.neighbor_id, next_depth, next_energy));
                    }
                }
            }
        }

        scratch.touched.len()
    }

    pub fn spreading_activation_by_id(
        &self,
        seed_id: u32,
        max_depth: usize,
        decay_factor: f32,
        energy_threshold: f32,
    ) -> Result<Vec<ActivatedNode>> {
        thread_local! {
            static TL_SAMPLE_RNG: std::cell::Cell<u64> = const { std::cell::Cell::new(0x9E3779B97F4A7C15) };
        }
        let sample_this_query = TL_SAMPLE_RNG.with(|c| {
            let s = c.get().wrapping_add(1);
            c.set(s);
            (s & 7) == 0
        });

        let now = current_timestamp() as u32;
        // Tracking (energy, depth, parent_id) with preallocated capacity - eliminates heap string allocations during BFS!
        let mut activated_energy: HashMap<u32, (f32, usize, u32)> = HashMap::with_capacity(128);
        let mut queue: VecDeque<(u32, f32, usize, u32)> = VecDeque::with_capacity(128);

        queue.push_back((seed_id, 1.0, 0, seed_id));
        activated_energy.insert(seed_id, (1.0, 0, seed_id));

        if sample_this_query {
            self.telemetry.mark_dirty();
            self.telemetry.record_node_hit_sampled(seed_id, now, 8);
        }

        while let Some((curr_id, curr_energy, depth, _)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            // Directly iterate outgoing edges (direction_factor = 1.0) in Hot-Tier order
            if let Some(out_edges) = self.adj_out.get(&curr_id) {
                for &edge in out_edges {
                    let next_energy = curr_energy * edge.weight * decay_factor;
                    if next_energy < energy_threshold {
                        continue;
                    }

                    let (should_visit, is_first_visit) = match activated_energy.get(&edge.neighbor_id) {
                        Some(&(existing_energy, _, _)) => (next_energy > existing_energy, false),
                        None => (true, true),
                    };

                    if should_visit {
                        activated_energy.insert(edge.neighbor_id, (next_energy, depth + 1, curr_id));
                        queue.push_back((edge.neighbor_id, next_energy, depth + 1, curr_id));

                        if is_first_visit && sample_this_query {
                            self.telemetry.record_synapse_hit_sampled(edge.synapse_id, now, 8);
                            self.telemetry.record_node_hit_sampled(edge.neighbor_id, now, 8);
                        }
                    }
                }
            }

            // Directly iterate incoming edges (direction_factor = 0.85) in Hot-Tier order
            if let Some(in_edges) = self.adj_in.get(&curr_id) {
                for &edge in in_edges {
                    let next_energy = curr_energy * edge.weight * decay_factor * 0.85;
                    if next_energy < energy_threshold {
                        continue;
                    }

                    let (should_visit, is_first_visit) = match activated_energy.get(&edge.neighbor_id) {
                        Some(&(existing_energy, _, _)) => (next_energy > existing_energy, false),
                        None => (true, true),
                    };

                    if should_visit {
                        activated_energy.insert(edge.neighbor_id, (next_energy, depth + 1, curr_id));
                        queue.push_back((edge.neighbor_id, next_energy, depth + 1, curr_id));

                        if is_first_visit && sample_this_query {
                            self.telemetry.record_synapse_hit_sampled(edge.synapse_id, now, 8);
                            self.telemetry.record_node_hit_sampled(edge.neighbor_id, now, 8);
                        }
                    }
                }
            }
        }

        let nodes_slice = self.storage.nodes();
        let mut results: Vec<ActivatedNode> = Vec::with_capacity(activated_energy.len());
        for (&node_id, &(energy, depth, _)) in &activated_energy {
            if let Some(node) = nodes_slice.get(node_id as usize) {
                let Ok(name_str) = self.storage.get_string(node.name_offset, node.name_len as u32) else {
                    continue;
                };
                let name = name_str.to_string();
                let tags_raw = self
                    .storage
                    .get_string(node.tags_offset, node.tags_len as u32)
                    .unwrap_or("");
                let tags: Vec<String> = if tags_raw.is_empty() {
                    Vec::new()
                } else {
                    tags_raw
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect()
                };
                let content = self
                    .storage
                    .get_string(node.content_offset, node.content_len)
                    .unwrap_or("")
                    .to_string();

                // Reconstruct path by walking back parent links to seed_id
                let mut path_ids = Vec::with_capacity(depth + 1);
                let mut curr = node_id;
                path_ids.push(curr);
                while curr != seed_id {
                    if let Some(&(_, _, parent)) = activated_energy.get(&curr) {
                        if parent == curr {
                            break;
                        }
                        path_ids.push(parent);
                        curr = parent;
                    } else {
                        break;
                    }
                }
                path_ids.reverse();
                let path: Vec<String> = path_ids
                    .into_iter()
                    .map(|id| self.get_node_name(id).unwrap_or_default())
                    .collect();

                results.push(ActivatedNode {
                    id: node_id,
                    name,
                    node_type: node_type_to_str(node.node_type).to_string(),
                    tags,
                    content,
                    energy,
                    depth,
                    path,
                });
            }
        }

        results.sort_by(|a, b| {
            b.energy
                .partial_cmp(&a.energy)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(results)
    }

    pub fn consolidate(&mut self, half_life_days: f32, prune_below: f32) -> Result<usize> {
        self.sync_telemetry_to_mmap();
        let now = current_timestamp();
        let seconds_in_day = 86400.0;
        let lambda = 0.693 / (half_life_days.max(0.1) * seconds_in_day);
        let mut pruned_count = 0;

        let pinned_nodes: std::collections::HashSet<u32> = self
            .storage
            .nodes()
            .iter()
            .filter(|n| (n.flags & FLAG_DELETED) == 0 && (n.flags & FLAG_PINNED) != 0)
            .map(|n| n.id)
            .collect();

        let synapses = self.storage.synapses_mut();
        for syn in synapses.iter_mut() {
            if (syn.flags & FLAG_DELETED) != 0 {
                continue;
            }
            if pinned_nodes.contains(&syn.source_id) && pinned_nodes.contains(&syn.target_id) {
                continue;
            }

            let delta_t = (now.saturating_sub(syn.last_used as u64)) as f32;
            let retention = (-lambda * delta_t).exp();
            syn.weight = (syn.weight * retention).clamp(0.0, 1.0);

            if syn.weight < prune_below {
                syn.flags |= FLAG_DELETED;
                pruned_count += 1;
            }
        }

        self.storage.header_mut().last_consolidated = now;
        self.rebuild_indices();
        self.storage.flush()?;
        Ok(pruned_count)
    }

    /// Phase 0 (`&self` under read-lock): Scans active records and computes exact storage waste metrics.
    pub fn vacuum_plan(&self, force: bool) -> Option<VacuumPlan> {
        let mut live_string_bytes: u64 = 0;
        let nodes = self.storage.nodes();
        for n in nodes {
            if (n.flags & FLAG_DELETED) == 0 {
                live_string_bytes = live_string_bytes
                    .saturating_add(n.name_len as u64)
                    .saturating_add(n.tags_len as u64)
                    .saturating_add(n.content_len as u64);
            }
        }
        let synapses = self.storage.synapses();
        for s in synapses {
            if (s.flags & FLAG_DELETED) == 0 {
                live_string_bytes = live_string_bytes.saturating_add(s.rel_len as u64);
            }
        }
        let total_string_bytes = self.storage.header().strings_bytes;
        let dead_string_bytes = total_string_bytes.saturating_sub(live_string_bytes);
        let dead_vector_slots = self.storage.header().dead_vector_slots;
        let total_vector_slots = self.storage.header().vector_count;
        let hnsw_tombstones = self.hnsw.tombstone_count();
        let bm25_dead_slots = self.bm25.dead_slots;
        let file_size_bytes = self.storage.file_size_bytes();

        let dead_vec_bytes = (dead_vector_slots as u64)
            .saturating_mul((self.storage.bytes_per_vector() + self.storage.header().vector_dims as usize + 8) as u64);
        let total_dead = dead_string_bytes.saturating_add(dead_vec_bytes);
        let waste_ratio = (total_dead as f32) / (file_size_bytes.max(1) as f32);

        let should_vacuum = force
            || (waste_ratio >= 0.30 && file_size_bytes >= 4 * 1024 * 1024)
            || bm25_dead_slots > 2048
            || hnsw_tombstones > 256;

        Some(VacuumPlan {
            dead_string_bytes,
            total_string_bytes,
            dead_vector_slots,
            total_vector_slots,
            hnsw_tombstones,
            bm25_dead_slots,
            file_size_bytes,
            waste_ratio,
            should_vacuum,
        })
    }

    /// Phase 1 (`&mut self` under write-lock, microseconds):
    /// Activates `VacuumJournal` dirty-bitsets and allocates the `ShadowSegment` temporary file.
    pub fn vacuum_begin(&mut self) -> Result<(VacuumTicket, ShadowSegment)> {
        self.sync_telemetry_to_mmap();
        let hdr = *self.storage.header();
        let snap_gen = hdr.generation;
        let next_seq = self.storage.active_seq.saturating_add(1);

        // Compute trimmed tail counts (preserving all stable IDs up to the last non-deleted slot)
        let nodes = self.storage.nodes();
        let mut trimmed_node_count = nodes.len() as u32;
        while trimmed_node_count > 0
            && (nodes[(trimmed_node_count - 1) as usize].flags & FLAG_DELETED) != 0
        {
            trimmed_node_count -= 1;
        }

        let synapses = self.storage.synapses();
        let mut trimmed_synapse_count = synapses.len() as u32;
        while trimmed_synapse_count > 0
            && (synapses[(trimmed_synapse_count - 1) as usize].flags & FLAG_DELETED) != 0
        {
            trimmed_synapse_count -= 1;
        }

        self.vacuum = Some(VacuumJournal::new(
            snap_gen,
            hdr.max_nodes.max(trimmed_node_count),
            hdr.max_synapses.max(trimmed_synapse_count),
        ));

        let final_seg_path = StarkStorage::segment_path_for_seq(&self.storage.path, next_seq);
        let temp_path = PathBuf::from(format!("{}.vacuum_tmp", final_seg_path.display()));
        if temp_path.exists() {
            let _ = std::fs::remove_file(&temp_path);
        }

        let parent = self.storage.path.parent().unwrap_or(Path::new("."));
        let hw = crate::hardware::HardwareProfile::detect(parent);
        let mut shadow_storage = StarkStorage::open_raw_segment_file(
            &temp_path,
            &self.storage.path,
            next_seq,
            self.storage.active_seq,
            hdr.vector_dims,
            &hw,
            true,
        )?;
        shadow_storage.header_mut().vector_format = hdr.vector_format;
        shadow_storage.header_mut().created_at = hdr.created_at;
        shadow_storage.header_mut().last_consolidated = hdr.last_consolidated;

        if trimmed_node_count > 0 {
            shadow_storage.ensure_node_capacity(trimmed_node_count)?;
            shadow_storage.header_mut().node_count = trimmed_node_count;
        }
        if trimmed_synapse_count > 0 {
            shadow_storage.ensure_synapse_capacity(trimmed_synapse_count)?;
            shadow_storage.header_mut().synapse_count = trimmed_synapse_count;
        }

        let ticket = VacuumTicket {
            snap_gen,
            next_seq,
            node_count: trimmed_node_count,
            synapse_count: trimmed_synapse_count,
        };
        let shadow = ShadowSegment {
            temp_path,
            final_seg_path,
            next_seq,
            storage: shadow_storage,
            vec_remap: vec![u32::MAX; hdr.vector_count as usize],
            node_vec_map: vec![u32::MAX; trimmed_node_count as usize],
            hnsw_shadow: None,
            bm25_shadow: None,
            synapses_copied: false,
            catch_up_rounds: 0,
        };

        Ok((ticket, shadow))
    }

    /// Helper to copy a single `NodeRecord` at its exact `nid` from `self.storage` into `shadow`.
    fn copy_node_to_shadow(&self, nid: u32, shadow: &mut ShadowSegment) -> Result<()> {
        let idx = nid as usize;
        if idx >= self.storage.nodes().len() {
            return Ok(());
        }
        if nid >= shadow.storage.header().node_count {
            let needed = (nid + 1).saturating_sub(shadow.storage.header().node_count);
            shadow.storage.ensure_node_capacity(needed)?;
            shadow.storage.header_mut().node_count = nid + 1;
        }
        if idx >= shadow.node_vec_map.len() {
            shadow.node_vec_map.resize(idx + 1, u32::MAX);
        }

        let src_node = self.storage.nodes()[idx];
        if (src_node.flags & FLAG_DELETED) != 0 {
            // Release any previously copied vector in shadow if this node was deleted during catch-up
            let prev_shadow_vid = shadow.node_vec_map[idx];
            if prev_shadow_vid != u32::MAX {
                let _ = shadow.storage.release_vector(prev_shadow_vid);
                shadow.node_vec_map[idx] = u32::MAX;
            }
            let mut del_rec = src_node;
            del_rec.flags |= FLAG_DELETED;
            del_rec.vector_id = u32::MAX;
            shadow.storage.nodes_mut()[idx] = del_rec;
            return Ok(());
        }

        let name_str = self
            .storage
            .get_string(src_node.name_offset, src_node.name_len as u32)
            .unwrap_or("");
        let tags_str = self
            .storage
            .get_string(src_node.tags_offset, src_node.tags_len as u32)
            .unwrap_or("");
        let content_str = self
            .storage
            .get_string(src_node.content_offset, src_node.content_len)
            .unwrap_or("");

        let (name_offset, name_len) = shadow.storage.append_string(name_str)?;
        let (tags_offset, tags_len) = shadow.storage.append_string(tags_str)?;
        let (content_offset, content_len) = shadow.storage.append_payload(content_str)?;

        let new_vid = if src_node.vector_id != u32::MAX {
            let old_vid = src_node.vector_id as usize;
            if old_vid >= shadow.vec_remap.len() {
                shadow.vec_remap.resize(old_vid + 1, u32::MAX);
            }
            let existing_shadow_vid = shadow.node_vec_map[idx];
            if existing_shadow_vid != u32::MAX {
                shadow.storage.overwrite_raw_vector_from(
                    existing_shadow_vid,
                    &self.storage,
                    src_node.vector_id,
                )?;
                shadow.vec_remap[old_vid] = existing_shadow_vid;
                existing_shadow_vid
            } else {
                let copied_vid = shadow
                    .storage
                    .copy_raw_vector_from(&self.storage, src_node.vector_id)?;
                shadow.vec_remap[old_vid] = copied_vid;
                shadow.node_vec_map[idx] = copied_vid;
                copied_vid
            }
        } else {
            u32::MAX
        };

        let (live_hits, live_ts) =
            self.telemetry
                .get_node_stats(nid, src_node.activation_count, src_node.last_activated);

        let dst_node = NodeRecord {
            id: nid,
            node_type: src_node.node_type,
            flags: src_node.flags,
            name_offset,
            name_len,
            tags_len,
            tags_offset,
            content_offset,
            content_len,
            vector_id: new_vid,
            created_at: src_node.created_at,
            last_activated: live_ts,
            base_weight: src_node.base_weight,
            activation_count: live_hits,
        };
        shadow.storage.nodes_mut()[idx] = dst_node;
        Ok(())
    }

    /// Helper to copy a single `SynapseRecord` at its exact `sid` from `self.storage` into `shadow`.
    fn copy_synapse_to_shadow(&self, sid: u32, shadow: &mut ShadowSegment) -> Result<()> {
        let idx = sid as usize;
        if idx >= self.storage.synapses().len() {
            return Ok(());
        }
        if sid >= shadow.storage.header().synapse_count {
            let needed = (sid + 1).saturating_sub(shadow.storage.header().synapse_count);
            shadow.storage.ensure_synapse_capacity(needed)?;
            shadow.storage.header_mut().synapse_count = sid + 1;
        }
        let src_syn = self.storage.synapses()[idx];
        if (src_syn.flags & FLAG_DELETED) != 0 {
            let mut del_syn = src_syn;
            del_syn.flags |= FLAG_DELETED;
            shadow.storage.synapses_mut()[idx] = del_syn;
            return Ok(());
        }
        let rel_str = self
            .storage
            .get_string(src_syn.rel_offset, src_syn.rel_len as u32)
            .unwrap_or("LINK");
        let (rel_offset, rel_len) = shadow.storage.append_string(rel_str)?;
        let (live_hits, live_ts) =
            self.telemetry
                .get_synapse_stats(sid, src_syn.activation_count, src_syn.last_used);

        let dst_syn = SynapseRecord {
            id: sid,
            source_id: src_syn.source_id,
            target_id: src_syn.target_id,
            rel_offset,
            rel_len,
            flags: src_syn.flags,
            weight: src_syn.weight,
            last_used: live_ts,
            activation_count: live_hits,
        };
        shadow.storage.synapses_mut()[idx] = dst_syn;
        Ok(())
    }

    /// Rebuilds the internal `free_nodes_head` and `free_synapses_head` chains inside `shadow.storage`.
    fn rebuild_shadow_freelists(shadow: &mut ShadowSegment) {
        let mut free_n_head = u32::MAX;
        let n_cnt = shadow.storage.header().node_count as usize;
        for idx in (0..n_cnt).rev() {
            let n = &mut shadow.storage.nodes_mut()[idx];
            if (n.flags & FLAG_DELETED) != 0 {
                n.content_offset = free_n_head;
                free_n_head = idx as u32;
            }
        }
        shadow.storage.header_mut().free_nodes_head = free_n_head;

        let mut free_s_head = u32::MAX;
        let s_cnt = shadow.storage.header().synapse_count as usize;
        for idx in (0..s_cnt).rev() {
            let s = &mut shadow.storage.synapses_mut()[idx];
            if (s.flags & FLAG_DELETED) != 0 {
                s.rel_offset = free_s_head;
                free_s_head = idx as u32;
            }
        }
        shadow.storage.header_mut().free_synapses_head = free_s_head;
    }

    /// Phase 2 (`&self` under read-lock per chunk):
    /// Copies a chunk of `NodeRecord`s (and when reaching `ticket.node_count`, copies synapses and builds
    /// repaired HNSW + compacted BM25 shadow indices). Returns `Ok(true)` when the initial copy is complete.
    pub fn vacuum_copy_chunk(
        &self,
        ticket: &VacuumTicket,
        shadow: &mut ShadowSegment,
        range: std::ops::Range<u32>,
    ) -> Result<bool> {
        let end = range.end.min(ticket.node_count);
        for nid in range.start..end {
            self.copy_node_to_shadow(nid, shadow)?;
        }

        if end >= ticket.node_count && !shadow.synapses_copied {
            for sid in 0..ticket.synapse_count {
                self.copy_synapse_to_shadow(sid, shadow)?;
            }
            let mut hnsw_clone = self.hnsw.clone();
            hnsw_clone.repair_tombstones(&self.storage);
            hnsw_clone.remap_vector_ids(&shadow.vec_remap);
            shadow.hnsw_shadow = Some(hnsw_clone);

            let mut bm25_clone = self.bm25.clone();
            bm25_clone.compact_arena();
            shadow.bm25_shadow = Some(bm25_clone);

            shadow.synapses_copied = true;
            return Ok(true);
        }

        Ok(shadow.synapses_copied)
    }

    /// Phase 3 (`&mut self`): Replays dirty node/synapse/vector IDs accumulated since `vacuum_begin`
    /// or the previous catch-up pass. Returns the number of newly dirtied items pending after replay.
    pub fn vacuum_catch_up(&mut self, shadow: &mut ShadowSegment) -> Result<u32> {
        let Some((dirty_nodes, dirty_synapses, dirty_vectors)) =
            self.vacuum.as_mut().map(|j| j.take_dirty())
        else {
            return Ok(0);
        };

        for nid in dirty_nodes {
            self.copy_node_to_shadow(nid, shadow)?;
        }
        for sid in dirty_synapses {
            self.copy_synapse_to_shadow(sid, shadow)?;
        }
        if let Some(ref mut hnsw_sh) = shadow.hnsw_shadow {
            for nid in dirty_vectors {
                let idx = nid as usize;
                if let Some(n) = shadow.storage.nodes().get(idx) {
                    if (n.flags & FLAG_DELETED) != 0 || n.vector_id == u32::MAX {
                        hnsw_sh.mark_deleted(nid);
                    } else {
                        hnsw_sh.insert(nid, n.vector_id, &shadow.storage);
                    }
                }
            }
        }
        // Keep BM25 shadow synchronized with live BM25 + arena compaction
        let mut bm25_latest = self.bm25.clone();
        bm25_latest.compact_arena();
        shadow.bm25_shadow = Some(bm25_latest);

        shadow.catch_up_rounds += 1;
        Ok(self.vacuum.as_ref().map(|j| j.pending).unwrap_or(0))
    }

    /// Phase 4 (`&mut self` under write-lock, target < 2 ms):
    /// Replays final dirty items (<= 32), max-merges lock-free telemetry, flushes the shadow segment,
    /// atomically flips `<dbname>.manifest`, swaps `storage`/`hnsw`/`bm25`, and keeps the old segment as `rollback_seq` (C).
    pub fn vacuum_finish(&mut self, mut shadow: ShadowSegment) -> Result<VacuumStats> {
        use std::sync::atomic::Ordering;
        let original_size_bytes = self.storage.file_size_bytes();
        let old_seq = self.storage.active_seq;
        let new_seq = shadow.next_seq;
        let new_vacuum_epoch = self.storage.header().vacuum_epoch.saturating_add(1);
        let new_gen = self.storage.header().generation.saturating_add(1);
        let vector_dims = self.storage.header().vector_dims;

        // 1. Final replay of any remaining dirty items
        let _ = self.vacuum_catch_up(&mut shadow)?;
        Self::rebuild_shadow_freelists(&mut shadow);

        // 2. Max-merge live lock-free telemetry directly into shadow records
        {
            let sh_nodes = shadow.storage.nodes_mut();
            for (idx, node) in sh_nodes.iter_mut().enumerate() {
                if (node.flags & FLAG_DELETED) != 0 {
                    continue;
                }
                if let Some(slot) = self.telemetry.node_slots.get(idx) {
                    let h = slot.hits.load(Ordering::Relaxed);
                    if h > node.activation_count {
                        node.activation_count = h;
                    }
                    let ts = slot.last_ts.load(Ordering::Relaxed);
                    if ts > node.last_activated {
                        node.last_activated = ts;
                    }
                }
            }
            let sh_syns = shadow.storage.synapses_mut();
            for (idx, syn) in sh_syns.iter_mut().enumerate() {
                if (syn.flags & FLAG_DELETED) != 0 {
                    continue;
                }
                if let Some(slot) = self.telemetry.synapse_slots.get(idx) {
                    let h = slot.hits.load(Ordering::Relaxed);
                    if h > syn.activation_count {
                        syn.activation_count = h;
                    }
                    let ts = slot.last_ts.load(Ordering::Relaxed);
                    if ts > syn.last_used {
                        syn.last_used = ts;
                    }
                }
            }
        }

        // 3. Finalize shadow header and flush to disk
        shadow.storage.header_mut().dead_string_bytes = 0;
        shadow.storage.header_mut().dead_vector_slots = 0;
        shadow.storage.header_mut().vacuum_epoch = new_vacuum_epoch;
        shadow.storage.header_mut().generation = new_gen;
        shadow.storage.flush()?;

        let temp_path = shadow.temp_path.clone();
        let final_seg_path = shadow.final_seg_path.clone();
        let base_path = self.storage.path.clone();
        let hnsw_new = shadow.hnsw_shadow.take().unwrap_or_else(|| self.hnsw.clone());
        let bm25_new = shadow.bm25_shadow.take().unwrap_or_else(|| self.bm25.clone());
        let catch_up_rounds = shadow.catch_up_rounds;

        // Drop shadow storage handle so *.vacuum_tmp can be renamed to *.s{:06}.starkdb on Windows
        drop(shadow);
        if final_seg_path.exists() {
            let _ = std::fs::remove_file(&final_seg_path);
        }
        std::fs::rename(&temp_path, &final_seg_path)?;

        let parent = base_path.parent().unwrap_or(Path::new("."));
        let hw = crate::hardware::HardwareProfile::detect(parent);
        let final_storage = StarkStorage::open_raw_segment_file(
            &final_seg_path,
            &base_path,
            new_seq,
            old_seq,
            vector_dims,
            &hw,
            false,
        )?;
        let compacted_size_bytes = final_storage.file_size_bytes();

        // 4. Atomic manifest flip: active_seq = new_seq, rollback_seq = old_seq (C-rollback point)
        StarkStorage::write_manifest_flip(&base_path, new_seq, old_seq, new_vacuum_epoch)?;

        // 5. Swap storage, hnsw, and bm25 in memory; drop old storage (leaving old segment file on disk as C)
        let prev_rollback = self.storage.rollback_seq;
        let old_storage = std::mem::replace(&mut self.storage, final_storage);
        drop(old_storage);
        // Clean up any older rollback segment from 2 epochs ago
        if prev_rollback != u64::MAX && prev_rollback != old_seq && prev_rollback != new_seq {
            let ancient_path = StarkStorage::segment_path_for_seq(&base_path, prev_rollback);
            let _ = std::fs::remove_file(ancient_path);
        }

        self.hnsw = hnsw_new;
        self.bm25 = bm25_new;
        self.vacuum = None;
        self.telemetry.dirty_ops.store(0, Ordering::Relaxed);

        let _ = self.save_hnsw_cache();
        let _ = self.save_bm25_cache();

        Ok(VacuumStats {
            old_seq,
            new_seq,
            vacuum_epoch: new_vacuum_epoch,
            original_size_bytes,
            compacted_size_bytes,
            bytes_saved: original_size_bytes.saturating_sub(compacted_size_bytes),
            active_nodes: self.active_node_count(),
            active_synapses: self.active_synapse_count(),
            vectors: self.storage.header().vector_count,
            catch_up_rounds,
        })
    }

    /// Aborts an in-progress Shadow Vacuum and removes any temporary shadow segment files.
    pub fn vacuum_abort(&mut self, shadow: Option<ShadowSegment>) {
        self.vacuum = None;
        if let Some(sh) = shadow {
            let tmp = sh.temp_path.clone();
            let fin = sh.final_seg_path.clone();
            drop(sh);
            let _ = std::fs::remove_file(tmp);
            let _ = std::fs::remove_file(fin);
        }
    }

    /// Executes a complete ID-stable Shadow Vacuum cycle across chunks of 1,024 nodes.
    pub fn run_shadow_vacuum(&mut self) -> Result<VacuumStats> {
        let (ticket, mut shadow) = self.vacuum_begin()?;
        let mut cursor = 0u32;
        loop {
            let next = (cursor + 1024).min(ticket.node_count);
            let done = self.vacuum_copy_chunk(&ticket, &mut shadow, cursor..next)?;
            cursor = next;
            if done || cursor >= ticket.node_count {
                if !shadow.synapses_copied {
                    let _ = self.vacuum_copy_chunk(
                        &ticket,
                        &mut shadow,
                        ticket.node_count..ticket.node_count,
                    )?;
                }
                break;
            }
        }
        for _ in 0..8 {
            let pending = self.vacuum_catch_up(&mut shadow)?;
            let has_vec = self
                .vacuum
                .as_ref()
                .map(|j| j.has_vector_ops)
                .unwrap_or(false);
            if pending <= 32 && !has_vec {
                break;
            }
        }
        self.vacuum_finish(shadow)
    }

    /// Full offline compaction (`COMPACT`): renumbers active `node_id` and `synapse_id` densely from `0..N`,
    /// preserving `created_at`, `activation_count`, `last_activated`, `last_used`, and raw f16/SQ8 vectors,
    /// and builds HNSW & BM25 once at the end.
    pub fn compact(&mut self) -> Result<CompactionStats> {
        self.sync_telemetry_to_mmap();
        let original_size_bytes = self.storage.file_size_bytes();
        let base_path = self.storage.path.clone();
        let active_seg_path = self.storage.active_segment_path.clone();
        let active_seq = self.storage.active_seq;
        let rollback_seq = self.storage.rollback_seq;
        let vector_dims = self.storage.header().vector_dims;
        let vector_format = self.storage.header().vector_format;
        let created_at_hdr = self.storage.header().created_at;
        let last_cons_hdr = self.storage.header().last_consolidated;
        let vac_epoch = self.storage.header().vacuum_epoch;

        let temp_path = active_seg_path.with_extension("compact_tmp");
        if temp_path.exists() {
            let _ = std::fs::remove_file(&temp_path);
        }

        // Collect all active nodes with full metadata and raw vector slot index
        struct SavedNode {
            old_id: u32,
            node_type: u16,
            flags: u16,
            name: String,
            tags: String,
            content: String,
            old_vector_id: u32,
            created_at: u32,
            last_activated: u32,
            base_weight: f32,
            activation_count: u32,
        }
        let mut active_nodes = Vec::new();
        for node in self.storage.nodes() {
            if (node.flags & FLAG_DELETED) == 0 {
                let name = self
                    .storage
                    .get_string(node.name_offset, node.name_len as u32)
                    .unwrap_or("")
                    .to_string();
                let tags = self
                    .storage
                    .get_string(node.tags_offset, node.tags_len as u32)
                    .unwrap_or("")
                    .to_string();
                let content = self
                    .storage
                    .get_string(node.content_offset, node.content_len)
                    .unwrap_or("")
                    .to_string();
                let (hits, last_act) =
                    self.telemetry
                        .get_node_stats(node.id, node.activation_count, node.last_activated);
                active_nodes.push(SavedNode {
                    old_id: node.id,
                    node_type: node.node_type,
                    flags: node.flags & !FLAG_DELETED,
                    name,
                    tags,
                    content,
                    old_vector_id: node.vector_id,
                    created_at: node.created_at,
                    last_activated: last_act,
                    base_weight: node.base_weight,
                    activation_count: hits,
                });
            }
        }

        struct SavedSynapse {
            source_old_id: u32,
            target_old_id: u32,
            relation: String,
            flags: u16,
            weight: f32,
            last_used: u32,
            activation_count: u32,
        }
        let mut active_synapses = Vec::new();
        for syn in self.storage.synapses() {
            if (syn.flags & FLAG_DELETED) == 0 {
                let rel = self
                    .storage
                    .get_string(syn.rel_offset, syn.rel_len as u32)
                    .unwrap_or("LINK")
                    .to_string();
                let (hits, lu) =
                    self.telemetry
                        .get_synapse_stats(syn.id, syn.activation_count, syn.last_used);
                active_synapses.push(SavedSynapse {
                    source_old_id: syn.source_id,
                    target_old_id: syn.target_id,
                    relation: rel,
                    flags: syn.flags & !FLAG_DELETED,
                    weight: syn.weight,
                    last_used: lu,
                    activation_count: hits,
                });
            }
        }

        let parent = base_path.parent().unwrap_or(Path::new("."));
        let hw = crate::hardware::HardwareProfile::detect(parent);

        {
            let mut new_storage = StarkStorage::open_raw_segment_file(
                &temp_path,
                &base_path,
                active_seq,
                rollback_seq,
                vector_dims,
                &hw,
                true,
            )?;
            new_storage.header_mut().vector_format = vector_format;
            new_storage.header_mut().created_at = created_at_hdr;
            new_storage.header_mut().last_consolidated = last_cons_hdr;
            new_storage.header_mut().vacuum_epoch = vac_epoch.saturating_add(1);

            let n_total = active_nodes.len() as u32;
            if n_total > 0 {
                new_storage.ensure_node_capacity(n_total)?;
                new_storage.header_mut().node_count = n_total;
            }

            let mut old_to_new_node = HashMap::with_capacity(active_nodes.len());
            for (idx, sn) in active_nodes.into_iter().enumerate() {
                let new_id = idx as u32;
                let (name_offset, name_len) = new_storage.append_string(&sn.name)?;
                let (tags_offset, tags_len) = new_storage.append_string(&sn.tags)?;
                let (content_offset, content_len) = new_storage.append_payload(&sn.content)?;
                let new_vid = if sn.old_vector_id != u32::MAX {
                    new_storage.copy_raw_vector_from(&self.storage, sn.old_vector_id)?
                } else {
                    u32::MAX
                };
                new_storage.nodes_mut()[idx] = NodeRecord {
                    id: new_id,
                    node_type: sn.node_type,
                    flags: sn.flags,
                    name_offset,
                    name_len,
                    tags_len,
                    tags_offset,
                    content_offset,
                    content_len,
                    vector_id: new_vid,
                    created_at: sn.created_at,
                    last_activated: sn.last_activated,
                    base_weight: sn.base_weight,
                    activation_count: sn.activation_count,
                };
                old_to_new_node.insert(sn.old_id, new_id);
            }

            let mut valid_syns = Vec::with_capacity(active_synapses.len());
            for ss in active_synapses {
                if let (Some(&src_new), Some(&dst_new)) = (
                    old_to_new_node.get(&ss.source_old_id),
                    old_to_new_node.get(&ss.target_old_id),
                ) {
                    valid_syns.push((src_new, dst_new, ss));
                }
            }

            let s_total = valid_syns.len() as u32;
            if s_total > 0 {
                new_storage.ensure_synapse_capacity(s_total)?;
                new_storage.header_mut().synapse_count = s_total;
            }
            for (idx, (src_new, dst_new, ss)) in valid_syns.into_iter().enumerate() {
                let (rel_offset, rel_len) = new_storage.append_string(&ss.relation)?;
                new_storage.synapses_mut()[idx] = SynapseRecord {
                    id: idx as u32,
                    source_id: src_new,
                    target_id: dst_new,
                    rel_offset,
                    rel_len,
                    flags: ss.flags,
                    weight: ss.weight,
                    last_used: ss.last_used,
                    activation_count: ss.activation_count,
                };
            }
            new_storage.flush()?;
        }

        let compacted_size_bytes = std::fs::metadata(&temp_path)?.len() as usize;
        let bytes_saved = original_size_bytes.saturating_sub(compacted_size_bytes);

        let dummy_path = std::env::temp_dir().join(format!(
            "starkdb_swap_{}_{}.tmp",
            std::process::id(),
            current_timestamp()
        ));
        let dummy_storage = StarkStorage::open_or_create(&dummy_path, vector_dims)?;
        let old_storage = std::mem::replace(&mut self.storage, dummy_storage);
        drop(old_storage);

        let backup_path = active_seg_path.with_extension("compact_bak");
        let _ = std::fs::remove_file(&backup_path);
        let _ = std::fs::rename(&active_seg_path, &backup_path);
        std::fs::rename(&temp_path, &active_seg_path)?;
        let _ = std::fs::remove_file(&backup_path);

        let fresh_storage = StarkStorage::open_raw_segment_file(
            &active_seg_path,
            &base_path,
            active_seq,
            rollback_seq,
            vector_dims,
            &hw,
            false,
        )?;
        let dummy_storage = std::mem::replace(&mut self.storage, fresh_storage);
        drop(dummy_storage);
        let _ = std::fs::remove_file(&dummy_path);

        let _ = std::fs::remove_file(self.hnsw_cache_path());
        let _ = std::fs::remove_file(self.bm25_cache_path());
        // Single-pass index rebuild (builds HNSW and BM25 once!)
        self.rebuild_indices();

        Ok(CompactionStats {
            original_size_bytes,
            compacted_size_bytes,
            bytes_saved,
            active_nodes: self.active_node_count(),
            active_synapses: self.active_synapse_count(),
            vectors: self.storage.header().vector_count,
        })
    }
}

/// Storage waste evaluation returned by `StarkGraph::vacuum_plan` (Phase 0).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VacuumPlan {
    pub dead_string_bytes: u64,
    pub total_string_bytes: u64,
    pub dead_vector_slots: u32,
    pub total_vector_slots: u32,
    pub hnsw_tombstones: usize,
    pub bm25_dead_slots: u32,
    pub file_size_bytes: usize,
    pub waste_ratio: f32,
    pub should_vacuum: bool,
}

/// Snapshot metadata returned by `StarkGraph::vacuum_begin` (Phase 1).
#[derive(Clone, Debug)]
pub struct VacuumTicket {
    pub snap_gen: u64,
    pub next_seq: u64,
    pub node_count: u32,
    pub synapse_count: u32,
}

/// Result metrics returned by `StarkGraph::vacuum_finish` (Phase 4).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VacuumStats {
    pub old_seq: u64,
    pub new_seq: u64,
    pub vacuum_epoch: u64,
    pub original_size_bytes: usize,
    pub compacted_size_bytes: usize,
    pub bytes_saved: usize,
    pub active_nodes: usize,
    pub active_synapses: usize,
    pub vectors: u32,
    pub catch_up_rounds: u32,
}

/// In-progress shadow segment state built during Phase 2 and Phase 3 of Shadow Vacuum.
pub struct ShadowSegment {
    pub temp_path: PathBuf,
    pub final_seg_path: PathBuf,
    pub next_seq: u64,
    pub storage: StarkStorage,
    pub vec_remap: Vec<u32>,
    pub node_vec_map: Vec<u32>,
    pub hnsw_shadow: Option<crate::vectors::HnswIndex>,
    pub bm25_shadow: Option<crate::bm25::Bm25Index>,
    pub synapses_copied: bool,
    pub catch_up_rounds: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{DbHeader, NodeRecord, SynapseRecord};
    use crate::migration::{export_database, import_database};
    use std::time::Instant;

    #[test]
    fn test_binary_struct_sizes_and_interning() {
        assert_eq!(std::mem::size_of::<DbHeader>(), 256);
        assert_eq!(std::mem::size_of::<crate::format::DbHeaderV3>(), 128);
        assert_eq!(std::mem::size_of::<NodeRecord>(), 48);
        assert_eq!(std::mem::size_of::<SynapseRecord>(), 32);

        let tmp_dir = std::env::temp_dir().join("starkdb_test_interning");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let db_path = tmp_dir.join("intern.starkdb");

        let mut db = StarkGraph::new(StarkStorage::open_or_create(&db_path, 384).unwrap()).unwrap();
        // Pre-allocation: 512 (A/B headers) + 32*48 (1536) + 64*32 (2048) + 4096 = 8192 bytes (2x 4KB pages)
        assert_eq!(db.storage.file_size_bytes(), 8192);

        let (off1, len1) = db.storage.append_string("DEVELOPS").unwrap();
        let bytes_after_first = db.storage.header().strings_bytes;
        let (off2, len2) = db.storage.append_string("DEVELOPS").unwrap();
        let bytes_after_second = db.storage.header().strings_bytes;

        assert_eq!(off1, off2, "Subslice interning should reuse the exact same arena offset");
        assert_eq!(len1, len2);
        assert_eq!(
            bytes_after_first, bytes_after_second,
            "Repeated relation/tag strings must allocate 0 extra bytes"
        );

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_1500_nodes_1700_synapses_footprint_and_spreading_activation() {
        let tmp_dir = std::env::temp_dir().join("starkdb_test_1500");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let db_path = tmp_dir.join("brain_1500.starkdb");

        let mut db = StarkGraph::new(StarkStorage::open_or_create(&db_path, 384).unwrap()).unwrap();
        let node_types = ["Entity", "Software", "Concept", "Hardware", "Document", "Fact", "Rule"];
        let relations = [
            "DEVELOPS",
            "USES",
            "POWERS",
            "RUNS_ON",
            "CONNECTS_TO",
            "PART_OF",
            "OPTIMIZES",
            "REMEMBERS",
        ];

        for i in 0..1500 {
            let name = format!("Concept_Node_{:04}", i);
            let ntype = node_types[i % node_types.len()];
            let tags = format!("core,group_{},stark", i % 12);
            let content = format!(
                "Вузол когнітивної пам'яті #{} з описом для StarkOS та Wing Accel.",
                i
            );
            db.upsert_node(
                &name,
                Some(ntype),
                Some(&tags),
                Some(&content),
                Some(0.85),
                Some(i < 10),
            )
            .unwrap();
        }

        for e in 0..1700 {
            let src = format!("Concept_Node_{:04}", e % 1500);
            let dst = format!("Concept_Node_{:04}", (e * 7 + 13) % 1500);
            let rel = relations[e % relations.len()];
            db.remember(&src, rel, &dst, 0.90).unwrap();
        }

        let active_nodes = db.active_node_count();
        let active_synapses = db.active_synapse_count();
        let file_bytes = db.storage.file_size_bytes();
        let strings_bytes = db.storage.header().strings_bytes;
        let payload_bytes = 128 + active_nodes * 48 + active_synapses * 32 + strings_bytes as usize;

        println!(
            "[BENCHMARK 1500x1700] active_nodes={} active_synapses={} payload_bytes={} ({:.2} KB) file_size_bytes={} ({:.2} KB) strings_bytes={}",
            active_nodes,
            active_synapses,
            payload_bytes,
            payload_bytes as f64 / 1024.0,
            file_bytes,
            file_bytes as f64 / 1024.0,
            strings_bytes
        );

        assert_eq!(active_nodes, 1500);
        assert!(active_synapses >= 1690);
        assert!(
            file_bytes < 400 * 1024,
            "Expected file size < 400 KB (with dynamic hardware mmap page alignment), got {} bytes",
            file_bytes
        );

        let t0 = Instant::now();
        let activated = db
            .spreading_activation("Concept_Node_0000", 3, 0.75, 0.15)
            .unwrap();
        let elapsed_us = t0.elapsed().as_micros();
        println!(
            "[SPREADING ACTIVATION] 3-hop traversal across 1500 nodes / 1700 synapses returned {} nodes in {} µs",
            activated.len(),
            elapsed_us
        );
        assert!(!activated.is_empty());

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_sql_and_mongo_dump_and_restore() {
        let tmp_dir = std::env::temp_dir().join("starkdb_test_migration");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();

        let mut src_db =
            StarkGraph::new(StarkStorage::open_or_create(tmp_dir.join("src.starkdb"), 384).unwrap())
                .unwrap();
        src_db
            .upsert_node(
                "Даня",
                Some("Entity"),
                Some("author"),
                Some("Creator of StarkOS"),
                Some(1.0),
                Some(true),
            )
            .unwrap();
        src_db
            .upsert_node(
                "StarkDB",
                Some("Software"),
                Some("rust,db"),
                Some("Wing Accel Engine"),
                Some(0.95),
                Some(true),
            )
            .unwrap();
        src_db.remember("Даня", "DEVELOPS", "StarkDB", 0.99).unwrap();

        let sql_dump = export_database(&src_db, "src", "sql").unwrap();
        assert!(sql_dump.contains("CREATE TABLE IF NOT EXISTS stark_nodes"));
        assert!(sql_dump.contains("INSERT INTO stark_nodes"));
        assert!(sql_dump.contains("INSERT INTO stark_synapses"));

        let mut dst_sql_db = StarkGraph::new(
            StarkStorage::open_or_create(tmp_dir.join("dst_sql.starkdb"), 384).unwrap(),
        )
        .unwrap();
        let (nodes_sql, syn_sql) = import_database(&mut dst_sql_db, "sql", &sql_dump).unwrap();
        assert_eq!(nodes_sql, 2);
        assert_eq!(syn_sql, 1);

        let mongo_dump = export_database(&src_db, "src", "mongo").unwrap();
        let mut dst_mongo_db = StarkGraph::new(
            StarkStorage::open_or_create(tmp_dir.join("dst_mongo.starkdb"), 384).unwrap(),
        )
        .unwrap();
        let (nodes_mongo, syn_mongo) =
            import_database(&mut dst_mongo_db, "mongo", &mongo_dump).unwrap();
        assert_eq!(nodes_mongo, 2);
        assert_eq!(syn_mongo, 1);

        let chroma_dump = export_database(&src_db, "src", "chromadb").unwrap();
        let mut dst_chroma_db = StarkGraph::new(
            StarkStorage::open_or_create(tmp_dir.join("dst_chroma.starkdb"), 384).unwrap(),
        )
        .unwrap();
        let (nodes_chroma, syn_chroma) =
            import_database(&mut dst_chroma_db, "chromadb", &chroma_dump).unwrap();
        assert_eq!(nodes_chroma, 2);
        assert_eq!(syn_chroma, 1);

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_ab_dual_header_recovery_and_crc_protection() {
        let tmp_dir = std::env::temp_dir().join("starkdb_test_ab_recovery");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let db_path = tmp_dir.join("ab_robust.starkdb");

        // Step 1: Create database and insert some concepts
        {
            let mut db = StarkGraph::new(StarkStorage::open_or_create(&db_path, 384).unwrap()).unwrap();
            db.remember("Quantum", "POWERS", "JarvisCore", 0.95).unwrap();
            db.storage.flush().unwrap();
        }

        // Step 2: Simulate crash during write that corrupts Page A (first 256 bytes)
        {
            use std::io::{Seek, SeekFrom, Write};
            let mut f = std::fs::OpenOptions::new().write(true).open(&db_path).unwrap();
            f.seek(SeekFrom::Start(0)).unwrap();
            f.write_all(&[0xDE; 256]).unwrap();
            f.flush().unwrap();
        }

        // Step 3: Reopen database - must successfully recover from Header Page B without crashing!
        {
            let db = StarkGraph::new(StarkStorage::open_or_create(&db_path, 384).unwrap()).unwrap();
            assert_eq!(db.list_nodes(None, None, None, 10).len(), 2);
            assert_eq!(db.list_synapses(None, 10).len(), 1);
            let syn = &db.list_synapses(None, 10)[0];
            assert_eq!(syn.relation, "POWERS");
        }

        // Step 4: Corrupt both Page A and Page B with garbage
        {
            use std::io::{Seek, SeekFrom, Write};
            let mut f = std::fs::OpenOptions::new().write(true).open(&db_path).unwrap();
            f.seek(SeekFrom::Start(256)).unwrap();
            f.write_all(&[0xFF; 256]).unwrap();
            f.flush().unwrap();
        }

        // Step 5: Opening fully corrupted header returns a clean Error, NOT a panic
        let err = StarkStorage::open_or_create(&db_path, 384);
        assert!(err.is_err(), "Must return error when both pages are corrupt");
        let msg = format!("{}", err.err().unwrap());
        assert!(msg.contains("Corrupted database file"), "Error message should be descriptive: {}", msg);

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_file_locking_prevents_concurrent_access() {
        let tmp_dir = std::env::temp_dir().join("starkdb_test_locking");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let db_path = tmp_dir.join("locked.starkdb");

        let storage1 = StarkStorage::open_or_create(&db_path, 384).unwrap();

        // Second open of the same file while first is still alive must fail with descriptive lock error
        let storage2_res = StarkStorage::open_or_create(&db_path, 384);
        assert!(storage2_res.is_err(), "Second concurrent open must fail with lock error");
        let err_msg = format!("{}", storage2_res.err().unwrap());
        assert!(err_msg.contains("locked by another running StarkDB instance"), "Expected lock error, got: {}", err_msg);

        drop(storage1);

        // After dropping storage1, opening again should succeed cleanly
        let storage3_res = StarkStorage::open_or_create(&db_path, 384);
        assert!(storage3_res.is_ok(), "Opening after drop should succeed");

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_crash_recovery_1000_iterations() {
        use std::io::{Seek, SeekFrom, Write};

        let tmp_dir = std::env::temp_dir().join("starkdb_test_crash_1000");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let db_path = tmp_dir.join("crash_1000.starkdb");

        // Initial setup
        {
            let mut db = StarkGraph::new(StarkStorage::open_or_create(&db_path, 384).unwrap()).unwrap();
            db.remember("RootConcept", "ANCHORS", "Genesis", 1.0).unwrap();
            db.storage.flush().unwrap();
        }

        // 1000 stress mutations with randomized page corruption
        for i in 0..1000 {
            // Open, do work, flush
            {
                let mut db = StarkGraph::new(StarkStorage::open_or_create(&db_path, 384).unwrap()).unwrap();
                let src = format!("Node_{}", i);
                let tgt = format!("Node_{}", i + 1);
                db.remember(&src, "CONNECTS", &tgt, 0.8).unwrap();
                db.storage.flush().unwrap();
            }

            // Simulate partial power failure or crash damaging one of the pages
            let corrupt_page_a = (i % 2) == 0;
            let offset = if corrupt_page_a { 0 } else { 256 };
            {
                let mut f = std::fs::OpenOptions::new().write(true).open(&db_path).unwrap();
                f.seek(SeekFrom::Start(offset)).unwrap();
                f.write_all(&[(i as u8).wrapping_add(0x33); 256]).unwrap();
                f.flush().unwrap();
            }

            // Verify the database always opens without panic or error
            let db = StarkGraph::new(StarkStorage::open_or_create(&db_path, 384).unwrap()).unwrap();
            assert!(db.storage.header().node_count >= 2);
        }

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_free_list_recycling_and_compaction() {
        let tmp_dir = std::env::temp_dir().join("starkdb_test_freelist");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let db_path = tmp_dir.join("recycle.starkdb");

        let mut db = StarkGraph::new(StarkStorage::open_or_create(&db_path, 384).unwrap()).unwrap();

        // 1. Insert 10 initial nodes
        for i in 0..10 {
            db.upsert_node(&format!("Initial_{}", i), Some("Entity"), None, Some("Payload data"), None, None).unwrap();
        }
        assert_eq!(db.storage.header().node_count, 10);
        assert_eq!(db.active_node_count(), 10);

        // 2. Delete 4 nodes
        db.delete_node("Initial_2").unwrap();
        db.delete_node("Initial_5").unwrap();
        db.delete_node("Initial_7").unwrap();
        db.delete_node("Initial_9").unwrap();

        assert_eq!(db.active_node_count(), 6);
        assert_ne!(db.storage.header().free_nodes_head, u32::MAX, "Free list head must not be MAX after deletes");

        // 3. Insert 4 new nodes - should recycle slots without incrementing node_count beyond 10!
        let n1 = db.get_or_create_node("Recycled_A", 0).unwrap();
        let n2 = db.get_or_create_node("Recycled_B", 0).unwrap();
        let n3 = db.get_or_create_node("Recycled_C", 0).unwrap();
        let n4 = db.get_or_create_node("Recycled_D", 0).unwrap();

        assert_eq!(db.storage.header().node_count, 10, "Allocated node count must remain 10 via free-list slot reuse");
        assert_eq!(db.active_node_count(), 10);
        let recycled_slots = [n1, n2, n3, n4];
        for slot in recycled_slots {
            assert!(slot < 10, "Slot {} must be within the recycled range < 10", slot);
        }

        // 4. Test synapses free list recycling
        db.remember("Recycled_A", "CONNECTS_TO", "Recycled_B", 0.9).unwrap();
        db.remember("Recycled_B", "CONNECTS_TO", "Recycled_C", 0.9).unwrap();
        db.remember("Recycled_C", "CONNECTS_TO", "Recycled_D", 0.9).unwrap();
        assert_eq!(db.storage.header().synapse_count, 3);

        db.delete_synapse(1).unwrap();
        assert_eq!(db.active_synapse_count(), 2);
        assert_eq!(db.storage.header().free_synapses_head, 1);

        let syn_recycled = db.remember("Recycled_A", "POWERS", "Recycled_D", 0.95).unwrap();
        assert_eq!(syn_recycled, 1, "Deleted synapse slot 1 must be reused");
        assert_eq!(db.storage.header().synapse_count, 3, "Synapse count must not expand beyond 3");

        // 5. Test Compaction
        let stats = db.compact().unwrap();
        assert_eq!(stats.active_nodes, 10);
        assert_eq!(stats.active_synapses, 3);
        assert!(stats.compacted_size_bytes <= stats.original_size_bytes);

        // Verify query after compaction
        let activated = db.spreading_activation("Recycled_A", 2, 0.8, 0.1).unwrap();
        assert!(!activated.is_empty(), "Compacted graph must be fully traversable");

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_cho_z_concurrent_burst_stress() {
        use std::sync::{Arc, RwLock};
        use std::thread;

        let tmp_dir = std::env::temp_dir().join("starkdb_test_cho_z_burst");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let db_path = tmp_dir.join("cho_z.starkdb");

        let dims: u16 = 128;
        let mut initial_db = StarkGraph::new(StarkStorage::open_or_create(&db_path, dims).unwrap()).unwrap();

        // Seed initial nodes, vectors and synapses
        for i in 0..30 {
            let name = format!("Seed_{}", i);
            let n = initial_db.upsert_node(&name, Some("Concept"), Some("seed"), Some("Initial seed payload"), Some(1.0), Some(false)).unwrap();
            let vec: Vec<f32> = (0..dims).map(|d| ((i * 5 + d as usize) as f32 * 0.05).sin()).collect();
            initial_db.attach_vector(n.id, &vec).unwrap();
        }
        for i in 0..40 {
            let src = format!("Seed_{}", i % 30);
            let tgt = format!("Seed_{}", (i * 3 + 1) % 30);
            initial_db.remember(&src, "LINK", &tgt, 0.8).unwrap();
        }
        initial_db.storage.flush().unwrap();

        let db = Arc::new(RwLock::new(initial_db));
        let mut handles = Vec::new();

        // 4 Mutator threads: insert nodes, huge payloads (triggering mmap resize and grow_sections), and synapses
        for t in 0..4 {
            let db_ref = Arc::clone(&db);
            handles.push(thread::spawn(move || {
                for i in 0..40 {
                    let name = format!("Burst_T{}_{}", t, i);
                    let big_payload = format!("Heavy payload data for thread {} iteration {}: {}", t, i, "X".repeat(1024));
                    let tags = format!("burst,t{},iter{}", t, i);
                    let _node_id = {
                        let mut g = db_ref.write().unwrap();
                        let n = g.upsert_node(&name, Some("Software"), Some(&tags), Some(&big_payload), Some(0.9), Some(false)).unwrap();
                        let vec: Vec<f32> = (0..dims).map(|d| ((t * 100 + i + d as usize) as f32 * 0.01).cos()).collect();
                        g.attach_vector(n.id, &vec).unwrap();
                        n.id
                    };
                    {
                        let mut g = db_ref.write().unwrap();
                        let target = format!("Seed_{}", (t * 7 + i) % 30);
                        let _ = g.remember(&name, "BURST_LINK", &target, 0.85);
                    }
                }
            }));
        }

        // 8 Reader / Query threads: run spreading_activation and vector search concurrently
        for t in 0..8 {
            let db_ref = Arc::clone(&db);
            handles.push(thread::spawn(move || {
                let query_vec: Vec<f32> = (0..dims).map(|d| ((t + d as usize) as f32 * 0.02).sin()).collect();
                for i in 0..50 {
                    // Vector search and Spreading activation both under shared read-lock!
                    {
                        let g = db_ref.read().unwrap();
                        let matches = g.search_vectors(&query_vec, 5).unwrap();
                        assert!(matches.len() <= 5);
                        let seed = format!("Seed_{}", (t + i) % 30);
                        let activated = g.spreading_activation(&seed, 2, 0.75, 0.1).unwrap();
                        let _ = activated.len();
                    }
                }
            }));
        }

        // 2 Auditor threads: run check_integrity_readonly under shared read-lock
        for _ in 0..2 {
            let db_ref = Arc::clone(&db);
            handles.push(thread::spawn(move || {
                for _ in 0..20 {
                    let g = db_ref.read().unwrap();
                    let rep = g.check_integrity_readonly().unwrap();
                    assert!(rep.healthy);
                    thread::sleep(std::time::Duration::from_millis(5));
                }
            }));
        }

        // 2 Deleter / Free-list recycler threads
        for t in 0..2 {
            let db_ref = Arc::clone(&db);
            handles.push(thread::spawn(move || {
                for i in 0..20 {
                    let del_name = format!("Burst_T{}_{}", t, i);
                    {
                        let mut g = db_ref.write().unwrap();
                        let _ = g.delete_node(&del_name);
                    }
                    {
                        let mut g = db_ref.write().unwrap();
                        let rec_name = format!("Recycled_T{}_{}", t, i);
                        let _ = g.get_or_create_node(&rec_name, 0);
                    }
                }
            }));
        }

        // Wait for all 16 threads to complete without any panic or segmentation fault
        for handle in handles {
            handle.join().expect("Cho-Z thread panicked during burst concurrency test!");
        }

        // Final verification: 100% healthy integrity report
        {
            let g = db.read().unwrap();
            let final_report = g.check_integrity_readonly().unwrap();
            assert!(final_report.healthy, "Final database state must be 100% healthy after burst stress!");
            assert_eq!(final_report.corrupted_nodes, 0);
            assert_eq!(final_report.corrupted_synapses, 0);
            assert!(final_report.header_a_valid);
            assert!(final_report.header_b_valid);
            assert!(final_report.valid_nodes > 50);
        }

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_lock_free_concurrent_spreading_activation() {
        use std::sync::{Arc, RwLock};
        use std::thread;

        let tmp_dir = std::env::temp_dir().join("starkdb_test_lock_free_sa");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let db_path = tmp_dir.join("brave_lockfree.starkdb");

        let mut initial_db =
            StarkGraph::new(StarkStorage::open_or_create(&db_path, 128).unwrap()).unwrap();

        let hub_id = initial_db
            .upsert_node("Hub_Core", Some("Concept"), Some("hub"), None, Some(1.0), Some(true))
            .unwrap()
            .id;
        let static_id = initial_db
            .upsert_node("Static_Node", Some("Concept"), None, None, Some(0.9), Some(false))
            .unwrap()
            .id;
        let hot_id = initial_db
            .upsert_node("Dynamic_Hot_Node", Some("Concept"), None, None, Some(0.9), Some(false))
            .unwrap()
            .id;

        // Synapse A: higher initial weight (0.82) -> initially at index 0 of Hub_Core's adj_out
        let syn_static = initial_db
            .remember("Hub_Core", "STATIC_LINK", "Static_Node", 0.82)
            .unwrap();
        // Synapse B: lower initial weight (0.75) -> initially at index 1 of Hub_Core's adj_out
        let syn_hot = initial_db
            .remember("Hub_Core", "HOT_LINK", "Dynamic_Hot_Node", 0.75)
            .unwrap();

        // Verify initial Hot-Tier order: syn_static (0.82) is ahead of syn_hot (0.75)
        {
            let edges = initial_db.outgoing_edges(hub_id).unwrap();
            assert_eq!(edges.len(), 2);
            assert_eq!(edges[0].synapse_id, syn_static);
            assert_eq!(edges[1].synapse_id, syn_hot);
            assert_eq!(edges[0].neighbor_id, static_id);
            assert_eq!(edges[1].neighbor_id, hot_id);
        }

        let db = Arc::new(RwLock::new(initial_db));
        let mut handles = Vec::new();

        // Spawn 16 concurrent reader threads executing spreading_activation ONLY under RwLock::read()!
        let threads = 16usize;
        let iters_per_thread = 50usize;
        for _ in 0..threads {
            let db_ref = Arc::clone(&db);
            handles.push(thread::spawn(move || {
                for _ in 0..iters_per_thread {
                    let g = db_ref.read().unwrap();
                    // 1-hop activation from Dynamic_Hot_Node traverses incoming syn_hot to Hub_Core
                    let res = g
                        .spreading_activation("Dynamic_Hot_Node", 1, 0.90, 0.20)
                        .unwrap();
                    assert_eq!(res.len(), 2);
                }
            }));
        }

        for h in handles {
            h.join()
                .expect("Concurrent lock-free spreading_activation reader thread panicked!");
        }

        // Verify unbiased 1/8 sampled lock-free atomic telemetry counts before & after adapt_hot_tier
        // Each of the 16 fresh threads starts with TL_SAMPLE_RNG = 0x9E3779B97F4A7C15 (...15 & 7 == 5),
        // and across 50 iterations hits (s & 7) == 0 exactly 6 times per thread -> 6 * 8 = 48 per thread -> 1 + 16*48 = 769!
        let sampled_hot_activations = {
            let g = db.read().unwrap();
            let hot_syn_view = g.get_synapse_view(syn_hot).unwrap();
            let static_syn_view = g.get_synapse_view(syn_static).unwrap();
            assert!(
                hot_syn_view.activation_count >= 700 && hot_syn_view.activation_count <= 900,
                "Sampled 1/8 lock-free counter must approximate 801 (got {})",
                hot_syn_view.activation_count
            );
            assert_eq!(
                (hot_syn_view.activation_count - 1) % 8,
                0,
                "1/8 sampled increments must step by 8"
            );
            assert_eq!(static_syn_view.activation_count, 1);
            hot_syn_view.activation_count
        };

        // Trigger Brave Evolution Hot-Tier adaptation and verify syn_hot is promoted to index 0 (first CPU cache line)
        {
            let mut g = db.write().unwrap();
            g.adapt_hot_tier();

            let edges = g.outgoing_edges(hub_id).unwrap();
            assert_eq!(
                edges[0].synapse_id, syn_hot,
                "Frequently activated synapse must be promoted to index 0 of adj_out!"
            );
            assert_eq!(edges[1].synapse_id, syn_static);
            assert!(edges[0].hot_score > edges[1].hot_score);

            // Also verify mmap record was synced
            assert_eq!(
                g.storage.synapses()[syn_hot as usize].activation_count,
                sampled_hot_activations
            );
        }

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_savior_hnsw_persistence_and_fast_load() {
        let tmp_dir = std::env::temp_dir().join(format!(
            "stark_savior_test_{}_{}",
            std::process::id(),
            current_timestamp()
        ));
        let _ = std::fs::create_dir_all(&tmp_dir);
        let db_path = tmp_dir.join("savior_hnsw.starkdb");
        let dims = 64usize;
        let num_nodes = 500usize;

        let make_vector = |seed: usize| -> Vec<f32> {
            let mut v = vec![0.0f32; dims];
            for (d, val) in v.iter_mut().enumerate().take(dims) {
                *val = (((seed * 31 + d * 7) as f32) * 0.05).sin();
            }
            v
        };

        // 1. Create database and populate 500+ vector nodes and synapses
        {
            let storage = StarkStorage::open_or_create(&db_path, dims as u16).unwrap();
            let mut graph = StarkGraph::new(storage).unwrap();

            for i in 0..num_nodes {
                let name = format!("Concept_{}", i);
                let node_id = graph.get_or_create_node(&name, 0).unwrap();
                let vec = make_vector(i);
                graph.attach_vector(node_id, &vec).unwrap();
                if i > 0 {
                    let parent = format!("Concept_{}", i / 2);
                    let _ = graph.remember(&parent, "RELATED", &name, 0.85).unwrap();
                }
            }

            // Verify HNSW cache was saved to disk
            let _ = graph.save_hnsw_cache();
            let cache_path = graph.hnsw_cache_path();
            assert!(cache_path.exists(), "HNSW cache file must exist on disk");
            let cache_meta = std::fs::metadata(&cache_path).unwrap();
            assert!(cache_meta.len() > 64, "HNSW cache must have header and non-empty payload");

            // Perform baseline vector search
            let query = make_vector(42);
            let baseline_hits = graph.search_vectors(&query, 10).unwrap();
            assert_eq!(baseline_hits.len(), 10);
            assert_eq!(baseline_hits[0].node_id, 42); // Exact query should be top-1 match
        }

        // 2. Re-open database with existing valid HNSW cache: measure fast load time
        let baseline_query = make_vector(42);
        let cached_hits = {
            let storage = StarkStorage::open_or_create(&db_path, dims as u16).unwrap();
            let start = std::time::Instant::now();
            let graph = StarkGraph::new(storage).unwrap();
            let load_elapsed = start.elapsed();

            println!(
                "⚡ Savior HNSW fast cache reload time (graph + HNSW): {:.2?} ({} active vectors in cache)",
                load_elapsed, graph.hnsw.active_count
            );
            assert_eq!(graph.hnsw.active_count, num_nodes);
            assert!(graph.hnsw.max_layer > 0);

            // Fast reload without distance recalculations takes < 25 ms even in unoptimized debug mode (and < 1 ms in release)
            assert!(load_elapsed.as_millis() < 100, "Cache load must be fast (< 100 ms in debug)");

            let hits = graph.search_vectors(&baseline_query, 10).unwrap();
            assert_eq!(hits.len(), 10);
            assert_eq!(hits[0].node_id, 42);
            hits
        };

        // 3. Invalidate cache: corrupt 1 byte of the cache payload and verify graceful self-healing fallback
        {
            let cache_path = {
                let storage = StarkStorage::open_or_create(&db_path, dims as u16).unwrap();
                let graph = StarkGraph::new(storage).unwrap();
                graph.hnsw_cache_path()
            };

            let mut cache_bytes = std::fs::read(&cache_path).unwrap();
            // Corrupt byte in payload past the 64-byte header
            if cache_bytes.len() > 70 {
                cache_bytes[70] ^= 0xFF;
                std::fs::write(&cache_path, &cache_bytes).unwrap();
            }

            // Re-open: corrupted cache CRC fails, engine gracefully rebuilds HNSW and writes fresh cache
            let storage_fresh = StarkStorage::open_or_create(&db_path, dims as u16).unwrap();
            let recovered_graph = StarkGraph::new(storage_fresh).unwrap();
            assert_eq!(recovered_graph.hnsw.active_count, num_nodes);
            assert!(recovered_graph.hnsw.max_layer > 0);

            let recovered_hits = recovered_graph.search_vectors(&baseline_query, 10).unwrap();
            assert_eq!(recovered_hits.len(), 10);
            assert_eq!(recovered_hits[0].node_id, 42);
            assert_eq!(recovered_hits[0].score, cached_hits[0].score);
        }

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_shadow_vacuum_concurrent_writes() {
        use std::sync::{Arc, RwLock};
        use std::thread;

        let tmp_dir = std::env::temp_dir().join(format!(
            "stark_shadow_vac_conc_{}_{}",
            std::process::id(),
            current_timestamp()
        ));
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let db_path = tmp_dir.join("conc_vac.starkdb");
        let dims = 32usize;

        let mut initial_db =
            StarkGraph::new(StarkStorage::open_or_create(&db_path, dims as u16).unwrap()).unwrap();

        let make_vec = |seed: usize| -> Vec<f32> {
            (0..dims)
                .map(|d| (((seed * 17 + d * 13) as f32) * 0.11).sin())
                .collect()
        };

        // Populate 200 nodes with vectors & synapses
        for i in 0..200 {
            let v = initial_db
                .upsert_node(
                    &format!("BaseNode_{}", i),
                    Some("Concept"),
                    Some("initial,batch"),
                    Some(&format!("Payload content for node {} with padding text", i)),
                    Some(0.85),
                    Some(false),
                )
                .unwrap();
            assert_eq!(v.id, i as u32);
            initial_db.attach_vector(v.id, &make_vec(i)).unwrap();
            if i > 0 {
                initial_db
                    .remember(
                        &format!("BaseNode_{}", i - 1),
                        "LINKS_TO",
                        &format!("BaseNode_{}", i),
                        0.8,
                    )
                    .unwrap();
            }
        }

        // Delete 40 nodes to create dead strings, dead vectors, and HNSW tombstones
        for i in (10..90).step_by(2) {
            initial_db.delete_node(&format!("BaseNode_{}", i)).unwrap();
        }

        let db = Arc::new(RwLock::new(initial_db));

        // Phase 1: Begin Shadow Vacuum under brief write lock
        let (ticket, mut shadow) = {
            let mut g = db.write().unwrap();
            g.vacuum_begin().unwrap()
        };

        // Spawn a concurrent writer thread that interleaves mutations between Phase 2 chunks
        let db_writer = Arc::clone(&db);
        let writer_handle = thread::spawn(move || {
            for w in 0..25 {
                {
                    let mut g = db_writer.write().unwrap();
                    // Update an existing active node
                    let target_id = (w * 4 + 1) as u32;
                    let _ = g
                        .update_node_by_id(
                            target_id,
                            None,
                            None,
                            Some("concurrent,updated"),
                            Some(&format!("Updated by concurrent writer step {}", w)),
                            Some(0.95),
                            None,
                        )
                        .unwrap();

                    // Insert a brand new node + vector + synapse during Vacuum
                    let new_name = format!("ConcurrentLive_{}", w);
                    let nv = g
                        .upsert_node(
                            &new_name,
                            Some("Fact"),
                            Some("live,writer"),
                            Some("written while shadow vacuum was copying chunks"),
                            Some(0.99),
                            Some(true),
                        )
                        .unwrap();
                    let vec: Vec<f32> = (0..dims)
                        .map(|d| (((w * 31 + d) as f32) * 0.2).cos())
                        .collect();
                    g.attach_vector(nv.id, &vec).unwrap();
                    g.remember("BaseNode_0", "CONCURRENT_EDGE", &new_name, 0.91)
                        .unwrap();
                }
                thread::sleep(std::time::Duration::from_millis(1));
            }
        });

        // Phase 2: Copy in small chunks of 32 nodes under read-lock, yielding between chunks
        let mut cursor = 0u32;
        loop {
            let next = (cursor + 32).min(ticket.node_count);
            let done = {
                let g = db.read().unwrap();
                g.vacuum_copy_chunk(&ticket, &mut shadow, cursor..next)
                    .unwrap()
            };
            cursor = next;
            if done || cursor >= ticket.node_count {
                break;
            }
            thread::sleep(std::time::Duration::from_millis(1));
        }

        writer_handle.join().unwrap();

        // Phase 3 & Phase 4: Catch-up dirty bitsets and finish atomic manifest flip
        let stats = {
            let mut g = db.write().unwrap();
            let _ = g.vacuum_catch_up(&mut shadow).unwrap();
            g.vacuum_finish(shadow).unwrap()
        };

        assert_eq!(stats.old_seq, 0);
        assert_eq!(stats.new_seq, 1);
        assert_eq!(stats.vacuum_epoch, 1);

        // Verify ID stability and 100% preservation of concurrent writes
        {
            let g = db.read().unwrap();
            assert_eq!(g.storage.active_seq, 1);
            assert_eq!(g.storage.rollback_seq, 0);

            // Surviving original nodes must keep their exact stable IDs
            let n1 = g.get_node_view(1).unwrap();
            assert_eq!(n1.name, "BaseNode_1");
            assert_eq!(n1.content, "Updated by concurrent writer step 0");

            let n199 = g.get_node_view(199).unwrap();
            assert_eq!(n199.name, "BaseNode_199");

            // All 25 concurrently inserted nodes + synapses + vectors must exist
            for w in 0..25 {
                let cname = format!("ConcurrentLive_{}", w);
                let cid = g
                    .find_node_id(&cname)
                    .expect("Concurrent node must exist after Shadow Vacuum");
                let cview = g.get_node_view(cid).unwrap();
                assert!(cview.has_vector);
                assert!(cview.pinned);
            }
        }

        // Drop and reopen from disk via manifest to verify persistence
        drop(db);
        let reopened =
            StarkGraph::new(StarkStorage::open_or_create(&db_path, dims as u16).unwrap()).unwrap();
        assert_eq!(reopened.storage.active_seq, 1);
        for w in 0..25 {
            assert!(reopened
                .find_node_id(&format!("ConcurrentLive_{}", w))
                .is_some());
        }

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_vacuum_crash_injection() {
        let tmp_dir = std::env::temp_dir().join(format!(
            "stark_vac_crash_{}_{}",
            std::process::id(),
            current_timestamp()
        ));
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let db_path = tmp_dir.join("crash_vac.starkdb");
        let dims = 32usize;

        // 1. Create baseline DB with 50 nodes
        {
            let mut db =
                StarkGraph::new(StarkStorage::open_or_create(&db_path, dims as u16).unwrap())
                    .unwrap();
            for i in 0..50 {
                let nid = db.get_or_create_node(&format!("Node_{}", i), 1).unwrap();
                let v: Vec<f32> = (0..dims).map(|d| ((i + d) as f32 * 0.1).sin()).collect();
                db.attach_vector(nid, &v).unwrap();
            }
            for i in 0..10 {
                db.delete_node(&format!("Node_{}", i)).unwrap();
            }

            // Crash Point A: Abort right after Phase 1 (`vacuum_begin`)
            let (_ticket, shadow_a) = db.vacuum_begin().unwrap();
            let orphan_tmp = shadow_a.temp_path.clone();
            assert!(orphan_tmp.exists());
            drop(shadow_a);
            db.vacuum = None;
        }

        // Reopen after Crash Point A: orphan .vacuum_tmp must be cleaned up, 40 nodes intact
        {
            let mut db =
                StarkGraph::new(StarkStorage::open_or_create(&db_path, dims as u16).unwrap())
                    .unwrap();
            assert_eq!(db.active_node_count(), 40);

            // Crash Point B: Abort mid-Phase 2 (`vacuum_copy_chunk` partial copy)
            let (ticket, mut shadow_b) = db.vacuum_begin().unwrap();
            let _ = db.vacuum_copy_chunk(&ticket, &mut shadow_b, 0..20).unwrap();
            drop(shadow_b);
            db.vacuum = None;
        }

        // Reopen after Crash Point B and complete a full Shadow Vacuum (flips manifest to seq=1, rollback_seq=0)
        let seg1_path = {
            let mut db =
                StarkGraph::new(StarkStorage::open_or_create(&db_path, dims as u16).unwrap())
                    .unwrap();
            assert_eq!(db.active_node_count(), 40);
            let stats = db.run_shadow_vacuum().unwrap();
            assert_eq!(stats.new_seq, 1);
            assert_eq!(db.storage.active_seq, 1);
            assert_eq!(db.storage.rollback_seq, 0);
            db.storage.active_segment_path.clone()
        };

        // Crash Point C: Simulate catastrophic corruption of active segment seq=1 (zero out both A and B headers!)
        // When reopening, `StarkStorage::open_or_create` must detect invalid active segment seq=1
        // and automatically roll back to C-segment (`rollback_seq = 0`) via `.manifest`!
        {
            let mut seg1_bytes = std::fs::read(&seg1_path).unwrap();
            for b in seg1_bytes.iter_mut().take(512) {
                *b = 0;
            }
            std::fs::write(&seg1_path, &seg1_bytes).unwrap();
        }

        let recovered_db =
            StarkGraph::new(StarkStorage::open_or_create(&db_path, dims as u16).unwrap()).unwrap();
        assert_eq!(
            recovered_db.storage.active_seq, 0,
            "Database must automatically roll back to C-segment (seq=0) when active segment (seq=1) is corrupted"
        );
        assert_eq!(recovered_db.active_node_count(), 40);
        for i in 10..50 {
            assert!(
                recovered_db.find_node_id(&format!("Node_{}", i)).is_some(),
                "Node_{} must be preserved after C-segment rollback",
                i
            );
        }

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_interrupted_section_grow_rollback() {
        use std::io::{Seek, SeekFrom, Write};

        let tmp_dir = std::env::temp_dir().join(format!(
            "stark_grow_crash_{}_{}",
            std::process::id(),
            current_timestamp()
        ));
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let db_path = tmp_dir.join("grow_crash.starkdb");
        let dims = 64u16;

        let (initial_gen, initial_active_page, initial_nodes_offset) = {
            let mut db =
                StarkGraph::new(StarkStorage::open_or_create(&db_path, dims).unwrap()).unwrap();
            for i in 0..50 {
                let src = format!("Concept_{}", i);
                let tgt = format!("Concept_{}", (i + 1) % 50);
                db.remember(&src, "REL", &tgt, 0.75).unwrap();
            }
            db.storage.flush().unwrap();
            let hdr = db.storage.header();
            (hdr.generation, hdr.active_page, hdr.nodes_offset)
        };

        // Verifies rejection of torn/corrupted inactive header and clean rollback
        // to the intact active page when an interrupted section expansion leaves orphaned bytes:
        // 1. Data was appended/copied to the tail of the file
        // 2. The inactive header page was only partially written (64 bytes of 256) with invalid CRC32
        let inactive_page_offset = if initial_active_page == 0 { 256u64 } else { 0u64 };
        {
            let mut f = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&db_path)
                .unwrap();

            // Simulate partial/torn header write to inactive page
            f.seek(SeekFrom::Start(inactive_page_offset)).unwrap();
            let torn_hdr_slice = [0xAAu8; 64];
            f.write_all(&torn_hdr_slice).unwrap();

            // Append orphaned grown section bytes to the end of the file
            f.seek(SeekFrom::End(0)).unwrap();
            f.write_all(&[0x55u8; 8192]).unwrap();
            f.flush().unwrap();
        }

        // Reopen database - must safely detect invalid inactive header CRC,
        // ignore the orphaned tail data, and roll back to the intact active page!
        {
            let mut db =
                StarkGraph::new(StarkStorage::open_or_create(&db_path, dims).unwrap()).unwrap();
            let hdr = db.storage.header();
            assert_eq!(
                hdr.generation, initial_gen,
                "Generation must remain at pre-crash value"
            );
            assert_eq!(
                hdr.active_page, initial_active_page,
                "Active page must remain on the uncorrupted page"
            );
            assert_eq!(
                hdr.nodes_offset, initial_nodes_offset,
                "Nodes section offset must remain rolled back to original"
            );
            assert_eq!(db.active_node_count(), 50, "All 50 nodes must remain intact");

            // Verify nodes and synapses survived without any data corruption
            let check = db.check_integrity_readonly().unwrap();
            assert_eq!(check.corrupted_nodes, 0, "Zero nodes corrupted");
            assert_eq!(check.corrupted_synapses, 0, "Zero synapses corrupted");
            assert_eq!(check.valid_nodes, 50, "All 50 nodes are valid");
            assert_eq!(check.valid_synapses, 50, "All 50 synapses are valid");

            // Verify database can continue normal mutations after rollback and heal the header state
            db.remember("Concept_0", "VERIFIED", "Concept_49", 0.99)
                .unwrap();
            db.storage.flush().unwrap();

            assert!(
                db.storage.header().generation > initial_gen,
                "Generation must cleanly advance after recovery and flush (was {}, now {})",
                initial_gen,
                db.storage.header().generation
            );

            let check_after = db.check_integrity_readonly().unwrap();
            assert!(
                check_after.healthy,
                "Integrity check must be 100% healthy with both headers valid after post-recovery flush"
            );
        }

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_crash_child_worker() {
        let db_path = match std::env::var("STARKDB_CHILD_DB_PATH") {
            Ok(p) => std::path::PathBuf::from(p),
            Err(_) => return,
        };

        let dims = 64u16;
        let mut db = StarkGraph::new(StarkStorage::open_or_create(&db_path, dims).unwrap()).unwrap();

        println!("CHILD_READY");
        use std::io::Write;
        let _ = std::io::stdout().flush();

        // Aggressively insert nodes and relations to force repeated section expansions
        for i in 0..100_000 {
            let src = format!("BombConcept_{}", i);
            let tgt = format!("BombTarget_{}", i);
            let _ = db.remember(&src, "EXPLODE", &tgt, 0.9);
            if i % 50 == 0 {
                let _ = db.storage.flush();
            }
        }
    }

    #[test]
    fn test_real_subprocess_sigkill() {
        use std::process::{Command, Stdio};
        use std::io::{BufRead, BufReader};
        use std::time::Duration;

        let tmp_dir = std::env::temp_dir().join(format!(
            "stark_sigkill_{}_{}",
            std::process::id(),
            current_timestamp()
        ));
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let db_path = tmp_dir.join("sigkill.starkdb");
        let dims = 64u16;

        // Step 1: Pre-populate database with known initial dataset and commit
        let initial_gen = {
            let mut db = StarkGraph::new(StarkStorage::open_or_create(&db_path, dims).unwrap()).unwrap();
            for i in 0..50 {
                db.remember(&format!("Seed_{}", i), "LINK", &format!("Seed_{}", (i + 1) % 50), 0.8).unwrap();
            }
            db.storage.flush().unwrap();
            db.storage.header().generation
        };

        // Step 2: Spawn child process pointing to the same database
        let current_exe = std::env::current_exe().expect("failed to obtain current test executable");
        let mut child = Command::new(&current_exe)
            .arg("graph::tests::test_crash_child_worker")
            .arg("--exact")
            .arg("--nocapture")
            .env("STARKDB_CHILD_DB_PATH", db_path.to_str().unwrap())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("failed to spawn crash child process");

        // Wait for child to start mutating
        if let Some(stdout) = child.stdout.take() {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            let _ = reader.read_line(&mut line);
        }

        // Let child perform concurrent writes/grow for 30ms, then abruptly KILL with SIGKILL / TerminateProcess
        std::thread::sleep(Duration::from_millis(30));

        // Real OS-level process termination (kill -9 equivalent)
        let kill_res = child.kill();
        assert!(kill_res.is_ok(), "Child process must be killed");
        let status = child.wait().expect("failed to wait on killed child");
        assert!(!status.success(), "Killed child process must not report clean success exit");

        // Step 3: Wait briefly for OS to release file locks on Windows
        std::thread::sleep(Duration::from_millis(100));

        // Step 4: Reopen database and verify that crash recovery restores consistent state
        let mut reopen_attempts = 0;
        let db = loop {
            match StarkStorage::open_or_create(&db_path, dims) {
                Ok(storage) => match StarkGraph::new(storage) {
                    Ok(graph) => break graph,
                    Err(_) if reopen_attempts < 20 => {
                        reopen_attempts += 1;
                        std::thread::sleep(Duration::from_millis(50));
                    }
                    Err(e) => panic!("Failed to open graph after kill: {:?}", e),
                },
                Err(_) if reopen_attempts < 20 => {
                    reopen_attempts += 1;
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(e) => panic!("Failed to open storage after kill: {:?}", e),
            }
        };

        let hdr = db.storage.header();
        assert!(
            hdr.generation >= initial_gen,
            "Generation must be at least initial committed generation"
        );
        assert!(
            hdr.node_count >= 50,
            "Must preserve at least the 50 initially committed nodes"
        );

        // Run read-only integrity audit
        let check = db.check_integrity_readonly().expect("Integrity check must succeed");
        assert_eq!(check.corrupted_nodes, 0, "Zero corrupted nodes allowed after process kill");
        assert_eq!(check.corrupted_synapses, 0, "Zero corrupted synapses allowed after process kill");
        assert!(check.valid_nodes >= 50, "All original valid nodes preserved");

        // Verify database is completely operational for new mutations
        let mut db_mut = db;
        db_mut.remember("PostKill_Node", "SURVIVED", "Seed_0", 1.0).unwrap();
        db_mut.storage.flush().unwrap();

        let post_check = db_mut.check_integrity_readonly().unwrap();
        assert!(post_check.healthy, "Database must be 100% healthy after new commit");

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }
}

