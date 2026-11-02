use crate::graph::{
    ActivatedNode, DatabaseInfo, NodeView, StarkEnvironmentManager, StarkGraph, SynapseView,
    VacuumStats,
};
use crate::hybrid::{BoundHit, BoundQuery, FusionWeights};
use crate::storage::current_timestamp;
use crate::vectors::VectorMatch;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Deserialize, Debug)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum StarkCommand {
    Ping,
    Stats,
    ListDatabases,
    CreateDatabase {
        name: String,
    },
    UseDatabase {
        name: String,
    },
    DropDatabase {
        name: String,
    },
    TruncateDatabase {
        name: String,
    },
    UpdateDbConfig {
        name: String,
        #[serde(default)]
        api_key_enabled: bool,
        #[serde(default)]
        allow_remote: bool,
    },
    RegenerateApiKey {
        name: String,
    },
    ExportDatabase {
        #[serde(default)]
        name: Option<String>,
        format: String,
    },
    ImportDatabase {
        #[serde(default)]
        name: Option<String>,
        format: String,
        payload: String,
    },
    Remember {
        source: String,
        relation: String,
        target: String,
        #[serde(default = "default_weight")]
        weight: f32,
    },
    UpsertNode {
        name: String,
        node_type: Option<String>,
        tags: Option<String>,
        content: Option<String>,
        base_weight: Option<f32>,
        pinned: Option<bool>,
    },
    UpdateNode {
        id: u32,
        name: Option<String>,
        node_type: Option<String>,
        tags: Option<String>,
        content: Option<String>,
        base_weight: Option<f32>,
        pinned: Option<bool>,
    },
    DeleteNode {
        target: String,
    },
    UpdateSynapse {
        id: u32,
        relation: Option<String>,
        weight: Option<f32>,
    },
    DeleteSynapse {
        id: u32,
    },
    FindNodes {
        search: Option<String>,
        node_type: Option<String>,
        tag: Option<String>,
        #[serde(default = "default_limit")]
        limit: usize,
    },
    FindSynapses {
        search: Option<String>,
        #[serde(default = "default_limit")]
        limit: usize,
    },
    Activate {
        concept: String,
        #[serde(default = "default_depth")]
        depth: usize,
        #[serde(default = "default_decay")]
        decay: f32,
        #[serde(default = "default_threshold")]
        threshold: f32,
    },
    AttachVector {
        node: String,
        vector: Vec<f32>,
    },
    Recall {
        vector: Vec<f32>,
        #[serde(default = "default_top_k")]
        top_k: usize,
    },
    Hybrid {
        vector: Vec<f32>,
        #[serde(default = "default_top_vectors")]
        top_vectors: usize,
        #[serde(default = "default_depth")]
        depth: usize,
    },
    BoundAttack {
        #[serde(default)]
        text: Option<String>,
        #[serde(default)]
        vector: Option<Vec<f32>>,
        #[serde(default)]
        anchors: Vec<String>,
        #[serde(default)]
        tags: Vec<String>,
        #[serde(default)]
        node_type: Option<String>,
        #[serde(default)]
        since: Option<u32>,
        #[serde(default)]
        until: Option<u32>,
        #[serde(default = "default_top_k")]
        top_k: usize,
        #[serde(default = "default_depth")]
        depth: usize,
        #[serde(default)]
        weights: Option<FusionWeights>,
    },
    TextSearch {
        query: String,
        #[serde(default = "default_top_k")]
        top_k: usize,
    },
    Consolidate {
        #[serde(default = "default_half_life")]
        half_life_days: f32,
        #[serde(default = "default_prune")]
        prune_below: f32,
    },
    Compact {
        #[serde(default)]
        name: Option<String>,
    },
    Vacuum {
        #[serde(default)]
        name: Option<String>,
        #[serde(default)]
        force: bool,
    },
    Quantize {
        #[serde(default)]
        name: Option<String>,
        mode: String,
    },
    Scrub {
        #[serde(default)]
        name: Option<String>,
    },
    Hardware,
}

fn default_weight() -> f32 {
    0.9
}
fn default_depth() -> usize {
    2
}
fn default_decay() -> f32 {
    0.75
}
fn default_threshold() -> f32 {
    0.1
}
fn default_top_k() -> usize {
    5
}
fn default_top_vectors() -> usize {
    3
}
fn default_half_life() -> f32 {
    30.0
}
fn default_prune() -> f32 {
    0.05
}
fn default_limit() -> usize {
    500
}

#[derive(Serialize, Debug)]
pub struct DbStats {
    pub active_db: String,
    pub codename: String,
    pub version: u32,
    pub active_nodes: usize,
    pub allocated_nodes: u32,
    pub max_nodes: u32,
    pub active_synapses: usize,
    pub allocated_synapses: u32,
    pub max_synapses: u32,
    pub vectors: u32,
    pub max_vectors: u32,
    pub vector_dims: u16,
    pub vector_format: String,
    pub string_bytes: u64,
    pub file_size_bytes: usize,
    pub hnsw_layers: usize,
    pub hnsw_indexed_vectors: usize,
    pub hnsw_tombstones: usize,
    pub lock_free_hits_total: u64,
    pub bm25_terms: usize,
    pub bm25_postings: usize,
    pub sq8_ready: bool,
    pub vacuum_epoch: u64,
    pub segment_seq: u64,
    pub dead_string_bytes: u64,
    pub dead_vector_slots: u32,
    pub write_lock_p50_us: u64,
    pub write_lock_p99_us: u64,
}

#[derive(Serialize, Debug)]
pub struct CommandResponse {
    pub status: String,
    pub elapsed_us: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub synapse_id: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_id: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node: Option<NodeView>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub synapse: Option<SynapseView>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nodes: Option<Vec<NodeView>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub synapses: Option<Vec<SynapseView>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub activated: Option<Vec<ActivatedNode>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_matches: Option<Vec<VectorMatch>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bound_hits: Option<Vec<BoundHit>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub databases: Option<Vec<DatabaseInfo>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<DbStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compaction: Option<crate::graph::CompactionStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vacuum_stats: Option<VacuumStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub integrity: Option<crate::scrub::IntegrityReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hardware: Option<crate::hardware::HardwareProfile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pruned: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exported_data: Option<String>,
}

impl CommandResponse {
    pub fn ok(elapsed_us: u64) -> Self {
        Self {
            status: "ok".to_string(),
            elapsed_us,
            message: None,
            synapse_id: None,
            node_id: None,
            node: None,
            synapse: None,
            nodes: None,
            synapses: None,
            activated: None,
            vector_matches: None,
            bound_hits: None,
            databases: None,
            stats: None,
            compaction: None,
            vacuum_stats: None,
            integrity: None,
            hardware: None,
            pruned: None,
            exported_data: None,
        }
    }

    pub fn err(msg: impl Into<String>, elapsed_us: u64) -> Self {
        Self {
            status: "error".to_string(),
            elapsed_us,
            message: Some(msg.into()),
            synapse_id: None,
            node_id: None,
            node: None,
            synapse: None,
            nodes: None,
            synapses: None,
            activated: None,
            vector_matches: None,
            bound_hits: None,
            databases: None,
            stats: None,
            compaction: None,
            vacuum_stats: None,
            integrity: None,
            hardware: None,
            pruned: None,
            exported_data: None,
        }
    }
}

fn tokenize_starkql(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_quote: Option<char> = None;
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];
        if let Some(q) = in_quote {
            if c == '\\' && i + 1 < chars.len() {
                current.push(chars[i + 1]);
                i += 2;
                continue;
            }
            if c == q {
                in_quote = None;
                tokens.push(current.clone());
                current.clear();
            } else {
                current.push(c);
            }
        } else if c == '"' || c == '\'' {
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
            in_quote = Some(c);
        } else if c.is_whitespace()
            || c == '('
            || c == ')'
            || c == '['
            || c == ']'
            || c == '{'
            || c == '}'
            || c == ','
            || c == ';'
        {
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
        } else {
            current.push(c);
        }
        i += 1;
    }

    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn parse_time_spec(raw: &str) -> Option<u32> {
    let s = raw.trim().to_ascii_lowercase();
    let now = current_timestamp() as u32;
    if let Some(days) = s.strip_suffix('d') {
        let d: u32 = days.parse().ok()?;
        return Some(now.saturating_sub(d.saturating_mul(86_400)));
    }
    if let Some(hours) = s.strip_suffix('h') {
        let h: u32 = hours.parse().ok()?;
        return Some(now.saturating_sub(h.saturating_mul(3_600)));
    }
    if let Some(mins) = s.strip_suffix('m') {
        let m: u32 = mins.parse().ok()?;
        return Some(now.saturating_sub(m.saturating_mul(60)));
    }
    s.parse::<u32>().ok()
}

pub fn build_db_stats(
    graph: &StarkGraph,
    active_db: String,
    write_lock_percentiles: (u64, u64),
) -> DbStats {
    let h = *graph.storage.header();
    let active_nodes = graph.active_node_count();
    let active_synapses = graph.active_synapse_count();
    let file_size_bytes = graph.storage.file_size_bytes();
    DbStats {
        active_db,
        codename: "StarkDB Ultimate Legacy (v1.0.0)".to_string(),
        version: h.version,
        active_nodes,
        allocated_nodes: h.node_count,
        max_nodes: h.max_nodes,
        active_synapses,
        allocated_synapses: h.synapse_count,
        max_synapses: h.max_synapses,
        vectors: h.vector_count,
        max_vectors: h.max_vectors,
        vector_dims: h.vector_dims,
        vector_format: if graph.storage.has_sq8() {
            "f16+sq8".to_string()
        } else if h.vector_format == 1 {
            "f16".to_string()
        } else {
            "f32".to_string()
        },
        string_bytes: h.strings_bytes,
        file_size_bytes,
        hnsw_layers: graph.hnsw.max_layer,
        hnsw_indexed_vectors: graph.hnsw.active_count,
        hnsw_tombstones: graph.hnsw.tombstone_count(),
        lock_free_hits_total: graph.total_telemetry_hits(),
        bm25_terms: graph.bm25.terms.len(),
        bm25_postings: graph.bm25.post_doc.len(),
        sq8_ready: graph.storage.has_sq8(),
        vacuum_epoch: h.vacuum_epoch,
        segment_seq: graph.storage.active_seq,
        dead_string_bytes: h.dead_string_bytes,
        dead_vector_slots: h.dead_vector_slots,
        write_lock_p50_us: write_lock_percentiles.0,
        write_lock_p99_us: write_lock_percentiles.1,
    }
}

impl StarkCommand {
    pub fn parse_line(line: &str) -> Result<Self> {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            bail!("Empty query");
        }
        if trimmed.starts_with('{') {
            let cmd: StarkCommand = serde_json::from_str(trimmed)?;
            return Ok(cmd);
        }

        let tokens = tokenize_starkql(trimmed);
        if tokens.is_empty() {
            bail!("Empty query");
        }

        let verb = tokens[0].to_uppercase();
        match verb.as_str() {
            "PING" => Ok(StarkCommand::Ping),
            "STATS" | "STATUS" => Ok(StarkCommand::Stats),
            "SHOW" => Ok(StarkCommand::ListDatabases),
            "USE" => {
                if tokens.len() < 2 {
                    bail!("Usage: USE <database_name>");
                }
                Ok(StarkCommand::UseDatabase {
                    name: tokens[1].clone(),
                })
            }
            "DROP" => {
                if tokens.len() >= 3 && tokens[1].eq_ignore_ascii_case("DATABASE") {
                    Ok(StarkCommand::DropDatabase {
                        name: tokens[2].clone(),
                    })
                } else if tokens.len() >= 2 {
                    Ok(StarkCommand::DropDatabase {
                        name: tokens[1].clone(),
                    })
                } else {
                    bail!("Usage: DROP DATABASE <name>");
                }
            }
            "ACTIVATE" => {
                if tokens.len() < 2 {
                    bail!("Usage: ACTIVATE \"<concept>\" [DEPTH 2] [DECAY 0.75]");
                }
                let concept = tokens[1].clone();
                let mut depth = 2usize;
                let mut decay = 0.75f32;
                let mut threshold = 0.1f32;

                let mut i = 2;
                while i < tokens.len() {
                    let kw = tokens[i].to_uppercase();
                    if kw == "DEPTH" && i + 1 < tokens.len() {
                        depth = tokens[i + 1].parse().unwrap_or(2);
                        i += 2;
                    } else if kw == "DECAY" && i + 1 < tokens.len() {
                        decay = tokens[i + 1].parse().unwrap_or(0.75);
                        i += 2;
                    } else if kw == "THRESHOLD" && i + 1 < tokens.len() {
                        threshold = tokens[i + 1].parse().unwrap_or(0.1);
                        i += 2;
                    } else if i == 2 {
                        depth = tokens[i].parse().unwrap_or(2);
                        i += 1;
                    } else if i == 3 {
                        decay = tokens[i].parse().unwrap_or(0.75);
                        i += 1;
                    } else {
                        i += 1;
                    }
                }

                Ok(StarkCommand::Activate {
                    concept,
                    depth,
                    decay,
                    threshold,
                })
            }
            "BOUND" | "BOUND_ATTACK" => {
                if tokens.len() < 2 {
                    bail!("Usage: BOUND \"<text>\" [ANCHOR <name>] [SINCE <7d|24h|ts>] [TAG <tag>] [TYPE <type>] [DEPTH <d>] [TOP <k>]");
                }
                let mut text: Option<String> = None;
                let mut anchors: Vec<String> = Vec::new();
                let mut tags: Vec<String> = Vec::new();
                let mut node_type: Option<String> = None;
                let mut since: Option<u32> = None;
                let mut until: Option<u32> = None;
                let mut depth = 2usize;
                let mut top_k = 5usize;

                let mut i = 1;
                while i < tokens.len() {
                    let kw = tokens[i].to_uppercase();
                    if kw == "ANCHOR" && i + 1 < tokens.len() {
                        anchors.push(tokens[i + 1].clone());
                        i += 2;
                    } else if (kw == "TAG" || kw == "TAGS") && i + 1 < tokens.len() {
                        for t in tokens[i + 1].split(',') {
                            let clean = t.trim();
                            if !clean.is_empty() {
                                tags.push(clean.to_string());
                            }
                        }
                        i += 2;
                    } else if kw == "TYPE" && i + 1 < tokens.len() {
                        node_type = Some(tokens[i + 1].clone());
                        i += 2;
                    } else if kw == "SINCE" && i + 1 < tokens.len() {
                        since = parse_time_spec(&tokens[i + 1]);
                        i += 2;
                    } else if kw == "UNTIL" && i + 1 < tokens.len() {
                        until = parse_time_spec(&tokens[i + 1]);
                        i += 2;
                    } else if kw == "DEPTH" && i + 1 < tokens.len() {
                        depth = tokens[i + 1].parse().unwrap_or(2);
                        i += 2;
                    } else if (kw == "TOP" || kw == "LIMIT") && i + 1 < tokens.len() {
                        top_k = tokens[i + 1].parse().unwrap_or(5);
                        i += 2;
                    } else if text.is_none() {
                        if !tokens[i].trim().is_empty() {
                            text = Some(tokens[i].clone());
                        }
                        i += 1;
                    } else {
                        i += 1;
                    }
                }

                Ok(StarkCommand::BoundAttack {
                    text,
                    vector: None,
                    anchors,
                    tags,
                    node_type,
                    since,
                    until,
                    top_k,
                    depth,
                    weights: None,
                })
            }
            "SEARCH" | "TEXT_SEARCH" | "BM25" => {
                if tokens.len() < 2 {
                    bail!("Usage: SEARCH \"<query>\" [TOP <k>]");
                }
                let query = tokens[1].clone();
                let mut top_k = 5usize;
                let mut i = 2;
                while i < tokens.len() {
                    let kw = tokens[i].to_uppercase();
                    if (kw == "TOP" || kw == "LIMIT") && i + 1 < tokens.len() {
                        top_k = tokens[i + 1].parse().unwrap_or(5);
                        i += 2;
                    } else if i == 2 {
                        top_k = tokens[i].parse().unwrap_or(5);
                        i += 1;
                    } else {
                        i += 1;
                    }
                }
                Ok(StarkCommand::TextSearch { query, top_k })
            }
            "REMEMBER" | "LINK" => {
                let filtered: Vec<String> = tokens[1..]
                    .iter()
                    .filter(|t| {
                        let s = t.as_str();
                        s != "-"
                            && s != "->"
                            && s != "-["
                            && s != "]->"
                            && !s.eq_ignore_ascii_case("weight:")
                            && !s.eq_ignore_ascii_case("weight")
                    })
                    .map(|t| {
                        t.trim_start_matches('-')
                            .trim_end_matches("->")
                            .trim_start_matches(':')
                            .to_string()
                    })
                    .filter(|t| !t.is_empty())
                    .collect();

                if filtered.len() < 3 {
                    bail!("Usage: REMEMBER \"<source>\" \"<relation>\" \"<target>\" [weight]");
                }

                let source = filtered[0].clone();
                let relation = filtered[1].clone();
                let (target, weight) = if filtered.len() >= 4 {
                    if let Ok(w) = filtered[2].parse::<f32>() {
                        (filtered[3].clone(), w)
                    } else {
                        let w = filtered[3].parse::<f32>().unwrap_or(0.9);
                        (filtered[2].clone(), w)
                    }
                } else {
                    (filtered[2].clone(), 0.9)
                };

                Ok(StarkCommand::Remember {
                    source,
                    relation,
                    target,
                    weight,
                })
            }
            "FIND" | "SELECT" => {
                let target_table = tokens
                    .get(1)
                    .map(|s| s.to_uppercase())
                    .unwrap_or_else(|| "NODES".to_string());
                if target_table.starts_with("SYNAPSE") || target_table.starts_with("EDGE") {
                    let search = tokens.get(2).cloned();
                    Ok(StarkCommand::FindSynapses {
                        search,
                        limit: 500,
                    })
                } else {
                    let mut search = None;
                    let mut node_type = None;
                    let mut tag = None;
                    let mut limit = 500usize;

                    let mut i = 2;
                    while i < tokens.len() {
                        let kw = tokens[i].to_uppercase();
                        if kw == "WHERE"
                            || kw == "FROM"
                            || kw == "*"
                            || kw == "="
                            || kw == "NODES"
                        {
                            i += 1;
                        } else if kw == "TYPE" && i + 1 < tokens.len() {
                            let val = if tokens[i + 1] == "=" && i + 2 < tokens.len() {
                                i += 3;
                                tokens[i - 1].clone()
                            } else {
                                i += 2;
                                tokens[i - 1].clone()
                            };
                            node_type = Some(val);
                        } else if kw == "TAG" && i + 1 < tokens.len() {
                            let val = if tokens[i + 1] == "=" && i + 2 < tokens.len() {
                                i += 3;
                                tokens[i - 1].clone()
                            } else {
                                i += 2;
                                tokens[i - 1].clone()
                            };
                            tag = Some(val);
                        } else if kw == "LIMIT" && i + 1 < tokens.len() {
                            limit = tokens[i + 1].parse().unwrap_or(500);
                            i += 2;
                        } else {
                            search = Some(tokens[i].clone());
                            i += 1;
                        }
                    }

                    Ok(StarkCommand::FindNodes {
                        search,
                        node_type,
                        tag,
                        limit,
                    })
                }
            }
            "CREATE" | "UPSERT" => {
                if tokens.len() >= 3 && tokens[1].eq_ignore_ascii_case("DATABASE") {
                    return Ok(StarkCommand::CreateDatabase {
                        name: tokens[2].clone(),
                    });
                }
                if tokens.len() < 3 {
                    bail!("Usage: CREATE DATABASE <name> | UPSERT NODE \"<name>\" [TYPE <type>] [TAGS <tags>] [CONTENT <json>]");
                }
                let name = tokens[2].clone();
                let mut node_type = None;
                let mut tags = None;
                let mut content = None;
                let mut base_weight = None;
                let mut pinned = None;

                let mut i = 3;
                while i < tokens.len() {
                    let kw = tokens[i].to_uppercase();
                    if kw == "TYPE" && i + 1 < tokens.len() {
                        node_type = Some(tokens[i + 1].clone());
                        i += 2;
                    } else if kw == "TAGS" && i + 1 < tokens.len() {
                        tags = Some(tokens[i + 1].clone());
                        i += 2;
                    } else if kw == "CONTENT" && i + 1 < tokens.len() {
                        content = Some(tokens[i + 1].clone());
                        i += 2;
                    } else if kw == "WEIGHT" && i + 1 < tokens.len() {
                        base_weight = tokens[i + 1].parse().ok();
                        i += 2;
                    } else if kw == "PINNED" {
                        pinned = Some(true);
                        i += 1;
                    } else {
                        i += 1;
                    }
                }

                Ok(StarkCommand::UpsertNode {
                    name,
                    node_type,
                    tags,
                    content,
                    base_weight,
                    pinned,
                })
            }
            "FORGET" | "DELETE" => {
                if tokens.len() < 2 {
                    bail!("Usage: FORGET NODE \"<name_or_id>\" | FORGET SYNAPSE <id>");
                }
                let sub = tokens[1].to_uppercase();
                if sub.starts_with("SYNAPSE") || sub.starts_with("EDGE") {
                    if tokens.len() < 3 {
                        bail!("Usage: FORGET SYNAPSE <id>");
                    }
                    let id = tokens[2].parse::<u32>()?;
                    Ok(StarkCommand::DeleteSynapse { id })
                } else if sub.starts_with("NODE") {
                    if tokens.len() < 3 {
                        bail!("Usage: FORGET NODE \"<name_or_id>\"");
                    }
                    Ok(StarkCommand::DeleteNode {
                        target: tokens[2].clone(),
                    })
                } else {
                    Ok(StarkCommand::DeleteNode {
                        target: tokens[1].clone(),
                    })
                }
            }
            "CONSOLIDATE" => {
                let mut half_life = 30.0f32;
                let mut prune = 0.05f32;
                let mut i = 1;
                while i < tokens.len() {
                    let kw = tokens[i].to_uppercase();
                    if (kw == "DECAY" || kw == "HALFLIFE") && i + 1 < tokens.len() {
                        half_life = tokens[i + 1].parse().unwrap_or(30.0);
                        i += 2;
                    } else if (kw == "PRUNE" || kw == "BELOW") && i + 1 < tokens.len() {
                        prune = tokens[i + 1].parse().unwrap_or(0.05);
                        i += 2;
                    } else if i == 1 {
                        half_life = tokens[i].parse().unwrap_or(30.0);
                        i += 1;
                    } else if i == 2 {
                        prune = tokens[i].parse().unwrap_or(0.05);
                        i += 1;
                    } else {
                        i += 1;
                    }
                }
                Ok(StarkCommand::Consolidate {
                    half_life_days: half_life,
                    prune_below: prune,
                })
            }
            "DUMP" | "EXPORT" => {
                let format = tokens
                    .get(1)
                    .cloned()
                    .unwrap_or_else(|| "sql".to_string());
                let name = tokens.get(2).cloned();
                Ok(StarkCommand::ExportDatabase { name, format })
            }
            "COMPACT" => {
                let name = tokens.get(1).cloned();
                Ok(StarkCommand::Compact { name })
            }
            "VACUUM" => {
                let mut name = None;
                let mut force = false;
                for tok in &tokens[1..] {
                    if tok.eq_ignore_ascii_case("FORCE") {
                        force = true;
                    } else if name.is_none() {
                        name = Some(tok.clone());
                    }
                }
                Ok(StarkCommand::Vacuum { name, force })
            }
            "QUANTIZE" => {
                let mode = tokens
                    .get(1)
                    .cloned()
                    .unwrap_or_else(|| "sq8".to_string());
                let name = tokens.get(2).cloned();
                Ok(StarkCommand::Quantize { name, mode })
            }
            "SCRUB" | "HEAL" | "INTEGRITY" => {
                let name = tokens.get(1).cloned();
                Ok(StarkCommand::Scrub { name })
            }
            "HARDWARE" | "SYSINFO" => Ok(StarkCommand::Hardware),
            cmd => bail!("Unknown StarkQL command: '{}'.", cmd),
        }
    }

    #[inline]
    pub fn is_read_only(&self) -> bool {
        matches!(
            self,
            StarkCommand::Ping
                | StarkCommand::Stats
                | StarkCommand::ListDatabases
                | StarkCommand::ExportDatabase { .. }
                | StarkCommand::FindNodes { .. }
                | StarkCommand::FindSynapses { .. }
                | StarkCommand::Activate { .. }
                | StarkCommand::Recall { .. }
                | StarkCommand::Hybrid { .. }
                | StarkCommand::BoundAttack { .. }
                | StarkCommand::TextSearch { .. }
                | StarkCommand::Hardware
        )
    }

    #[inline]
    pub fn is_env_command(&self) -> bool {
        matches!(
            self,
            StarkCommand::Ping
                | StarkCommand::Hardware
                | StarkCommand::ListDatabases
                | StarkCommand::CreateDatabase { .. }
                | StarkCommand::UseDatabase { .. }
                | StarkCommand::DropDatabase { .. }
                | StarkCommand::TruncateDatabase { .. }
                | StarkCommand::UpdateDbConfig { .. }
                | StarkCommand::RegenerateApiKey { .. }
        )
    }

    #[inline]
    pub fn target_db_override(&self) -> Option<&str> {
        match self {
            StarkCommand::ExportDatabase { name, .. }
            | StarkCommand::ImportDatabase { name, .. }
            | StarkCommand::Compact { name }
            | StarkCommand::Vacuum { name, .. }
            | StarkCommand::Quantize { name, .. }
            | StarkCommand::Scrub { name } => {
                name.as_deref().filter(|s| !s.trim().is_empty())
            }
            _ => None,
        }
    }

    pub fn execute_readonly_on_graph(
        self,
        graph: &StarkGraph,
        db_name: &str,
        write_lock_percentiles: (u64, u64),
        start: Instant,
    ) -> CommandResponse {
        match self {
            StarkCommand::Stats => {
                let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                r.stats = Some(build_db_stats(
                    graph,
                    db_name.to_string(),
                    write_lock_percentiles,
                ));
                r
            }
            StarkCommand::ExportDatabase { format, .. } => {
                match crate::migration::export_database(graph, db_name, &format) {
                    Ok(dump) => {
                        let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                        r.message = Some(format!(
                            "Exported '{}' to {} format ({} bytes)",
                            db_name,
                            format.to_uppercase(),
                            dump.len()
                        ));
                        r.exported_data = Some(dump);
                        r
                    }
                    Err(e) => {
                        CommandResponse::err(e.to_string(), start.elapsed().as_micros() as u64)
                    }
                }
            }
            StarkCommand::FindNodes {
                search,
                node_type,
                tag,
                limit,
            } => {
                let list = graph.list_nodes(
                    search.as_deref(),
                    node_type.as_deref(),
                    tag.as_deref(),
                    limit,
                );
                let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                r.nodes = Some(list);
                r
            }
            StarkCommand::FindSynapses { search, limit } => {
                let list = graph.list_synapses(search.as_deref(), limit);
                let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                r.synapses = Some(list);
                r
            }
            StarkCommand::Activate {
                concept,
                depth,
                decay,
                threshold,
            } => match graph.spreading_activation(&concept, depth, decay, threshold) {
                Ok(nodes) => {
                    let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                    r.activated = Some(nodes);
                    r
                }
                Err(e) => CommandResponse::err(e.to_string(), start.elapsed().as_micros() as u64),
            },
            StarkCommand::Recall { vector, top_k } => match graph.search_vectors(&vector, top_k) {
                Ok(matches) => {
                    let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                    r.vector_matches = Some(matches);
                    r
                }
                Err(e) => CommandResponse::err(e.to_string(), start.elapsed().as_micros() as u64),
            },
            StarkCommand::Hybrid {
                vector,
                top_vectors,
                depth,
            } => match graph.hybrid_recall(&vector, top_vectors, depth) {
                Ok(nodes) => {
                    let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                    r.activated = Some(nodes);
                    r
                }
                Err(e) => CommandResponse::err(e.to_string(), start.elapsed().as_micros() as u64),
            },
            StarkCommand::BoundAttack {
                text,
                vector,
                anchors,
                tags,
                node_type,
                since,
                until,
                top_k,
                depth,
                weights,
            } => {
                let anchor_ids: Vec<u32> = anchors
                    .iter()
                    .filter_map(|a| {
                        a.trim()
                            .parse::<u32>()
                            .ok()
                            .or_else(|| graph.find_node_id(a))
                    })
                    .collect();
                let tag_refs: Vec<&str> = tags.iter().map(|s| s.as_str()).collect();
                let nt_code = node_type
                    .as_deref()
                    .filter(|s| !s.trim().is_empty())
                    .map(crate::format::str_to_node_type);
                let q = BoundQuery {
                    vector: vector.as_deref(),
                    text: text.as_deref(),
                    anchors: &anchor_ids,
                    tags_all: &tag_refs,
                    node_type: nt_code,
                    since,
                    until,
                    top_k,
                    depth: depth.min(255) as u8,
                    weights: weights.unwrap_or_default(),
                };
                match graph.bound_attack(&q) {
                    Ok(hits) => {
                        let activated: Vec<ActivatedNode> = hits
                            .iter()
                            .filter_map(|h| {
                                let v = graph.get_node_view(h.node_id).ok()?;
                                Some(ActivatedNode {
                                    id: h.node_id,
                                    name: h.name.clone(),
                                    node_type: v.node_type,
                                    tags: v.tags,
                                    content: v.content,
                                    energy: h.score,
                                    depth: h.depth,
                                    path: vec![format!(
                                        "bound(v={:.2}, t={:.2}, g={:.2}, r={:.2})",
                                        h.s_v, h.s_t, h.s_g, h.s_r
                                    )],
                                })
                            })
                            .collect();
                        let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                        r.activated = Some(activated);
                        r.bound_hits = Some(hits);
                        r
                    }
                    Err(e) => {
                        CommandResponse::err(e.to_string(), start.elapsed().as_micros() as u64)
                    }
                }
            }
            StarkCommand::TextSearch { query, top_k } => {
                let raw_hits = graph.bm25.search(&query, top_k.max(1));
                let mut nodes = Vec::with_capacity(raw_hits.len());
                let mut bound_hits = Vec::with_capacity(raw_hits.len());
                for (score, doc_id) in raw_hits {
                    if let Ok(view) = graph.get_node_view(doc_id) {
                        bound_hits.push(BoundHit {
                            node_id: doc_id,
                            name: view.name.clone(),
                            score,
                            s_v: 0.0,
                            s_t: score,
                            s_g: 0.0,
                            s_r: 1.0,
                            depth: 0,
                        });
                        nodes.push(view);
                    }
                }
                let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                r.nodes = Some(nodes);
                r.bound_hits = Some(bound_hits);
                r
            }
            _ => CommandResponse::err(
                "Unsupported read-only command on graph",
                start.elapsed().as_micros() as u64,
            ),
        }
    }

    pub fn execute_on_graph(
        self,
        graph: &mut StarkGraph,
        db_name: &str,
        write_lock_percentiles: (u64, u64),
        start: Instant,
    ) -> CommandResponse {
        if self.is_read_only() {
            return self.execute_readonly_on_graph(graph, db_name, write_lock_percentiles, start);
        }
        match self {
            StarkCommand::ImportDatabase {
                format, payload, ..
            } => match crate::migration::import_database(graph, &format, &payload) {
                Ok((n_cnt, s_cnt)) => {
                    let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                    r.message = Some(format!(
                        "Imported {} nodes and {} synapses from {} into '{}'",
                        n_cnt,
                        s_cnt,
                        format.to_uppercase(),
                        db_name
                    ));
                    r
                }
                Err(e) => CommandResponse::err(e.to_string(), start.elapsed().as_micros() as u64),
            },
            StarkCommand::Compact { .. } => match graph.compact() {
                Ok(stats) => {
                    let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                    r.message = Some(format!(
                        "Database '{}' compacted: {:.1} KB -> {:.1} KB (saved {:.1} KB)",
                        db_name,
                        stats.original_size_bytes as f64 / 1024.0,
                        stats.compacted_size_bytes as f64 / 1024.0,
                        stats.bytes_saved as f64 / 1024.0,
                    ));
                    r.compaction = Some(stats);
                    r
                }
                Err(e) => CommandResponse::err(e.to_string(), start.elapsed().as_micros() as u64),
            },
            StarkCommand::Vacuum { .. } => match graph.run_shadow_vacuum() {
                Ok(stats) => {
                    let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                    r.message = Some(format!(
                        "Shadow Vacuum on '{}' finished (epoch {} -> seq {}): {:.1} KB -> {:.1} KB (saved {:.1} KB)",
                        db_name,
                        stats.vacuum_epoch,
                        stats.new_seq,
                        stats.original_size_bytes as f64 / 1024.0,
                        stats.compacted_size_bytes as f64 / 1024.0,
                        stats.bytes_saved as f64 / 1024.0,
                    ));
                    r.vacuum_stats = Some(stats);
                    r
                }
                Err(e) => CommandResponse::err(e.to_string(), start.elapsed().as_micros() as u64),
            },
            StarkCommand::Quantize { mode, .. } => {
                let enable = mode.eq_ignore_ascii_case("sq8")
                    || mode.eq_ignore_ascii_case("on")
                    || mode == "1";
                let res = graph
                    .storage
                    .set_sq8_mode(enable)
                    .and_then(|_| graph.storage.flush());
                match res {
                    Ok(()) => {
                        let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                        r.message = Some(format!(
                            "Quantization mode on '{}' set to {}",
                            db_name,
                            if enable { "SQ8 (f16+sq8)" } else { "OFF (f16)" }
                        ));
                        r
                    }
                    Err(e) => {
                        CommandResponse::err(e.to_string(), start.elapsed().as_micros() as u64)
                    }
                }
            }
            StarkCommand::Scrub { .. } => match graph.scrub_and_self_heal() {
                Ok(rep) => {
                    let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                    r.message = Some(format!(
                        "Integrity scrub for '{}' finished: healthy={}, healed {} anomaly(ies) in {} µs",
                        db_name, rep.healthy, rep.healed_count, rep.duration_us
                    ));
                    r.integrity = Some(rep);
                    r
                }
                Err(e) => CommandResponse::err(e.to_string(), start.elapsed().as_micros() as u64),
            },
            StarkCommand::Remember {
                source,
                relation,
                target,
                weight,
            } => match graph.remember(&source, &relation, &target, weight) {
                Ok(syn_id) => {
                    let syn_view = graph.get_synapse_view(syn_id).ok();
                    let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                    r.message = Some(format!(
                        "Synapse #{} linked: ({}) -[{}]-> ({})",
                        syn_id, source, relation, target
                    ));
                    r.synapse_id = Some(syn_id);
                    r.synapse = syn_view;
                    r
                }
                Err(e) => CommandResponse::err(e.to_string(), start.elapsed().as_micros() as u64),
            },
            StarkCommand::UpsertNode {
                name,
                node_type,
                tags,
                content,
                base_weight,
                pinned,
            } => match graph.upsert_node(
                &name,
                node_type.as_deref(),
                tags.as_deref(),
                content.as_deref(),
                base_weight,
                pinned,
            ) {
                Ok(view) => {
                    let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                    r.node_id = Some(view.id);
                    r.message = Some(format!("Node #{} '{}' saved", view.id, view.name));
                    r.node = Some(view);
                    r
                }
                Err(e) => CommandResponse::err(e.to_string(), start.elapsed().as_micros() as u64),
            },
            StarkCommand::UpdateNode {
                id,
                name,
                node_type,
                tags,
                content,
                base_weight,
                pinned,
            } => match graph.update_node_by_id(
                id,
                name.as_deref(),
                node_type.as_deref(),
                tags.as_deref(),
                content.as_deref(),
                base_weight,
                pinned,
            ) {
                Ok(view) => {
                    let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                    r.node_id = Some(view.id);
                    r.message = Some(format!("Node #{} updated", view.id));
                    r.node = Some(view);
                    r
                }
                Err(e) => CommandResponse::err(e.to_string(), start.elapsed().as_micros() as u64),
            },
            StarkCommand::DeleteNode { target } => match graph.delete_node(&target) {
                Ok(id) => {
                    let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                    r.node_id = Some(id);
                    r.message = Some(format!("Node #{} deleted", id));
                    r
                }
                Err(e) => CommandResponse::err(e.to_string(), start.elapsed().as_micros() as u64),
            },
            StarkCommand::UpdateSynapse {
                id,
                relation,
                weight,
            } => match graph.update_synapse(id, relation.as_deref(), weight) {
                Ok(view) => {
                    let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                    r.synapse_id = Some(view.id);
                    r.synapse = Some(view);
                    r.message = Some(format!("Synapse #{} updated", id));
                    r
                }
                Err(e) => CommandResponse::err(e.to_string(), start.elapsed().as_micros() as u64),
            },
            StarkCommand::DeleteSynapse { id } => match graph.delete_synapse(id) {
                Ok(sid) => {
                    let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                    r.synapse_id = Some(sid);
                    r.message = Some(format!("Synapse #{} deleted", sid));
                    r
                }
                Err(e) => CommandResponse::err(e.to_string(), start.elapsed().as_micros() as u64),
            },
            StarkCommand::AttachVector { node, vector } => match graph.get_or_create_node(&node, 0)
            {
                Ok(node_id) => match graph.attach_vector(node_id, &vector) {
                    Ok(vec_id) => {
                        let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                        r.message = Some(format!(
                            "Vector slot {} attached to node #{}",
                            vec_id, node_id
                        ));
                        r.node_id = Some(node_id);
                        r
                    }
                    Err(e) => {
                        CommandResponse::err(e.to_string(), start.elapsed().as_micros() as u64)
                    }
                },
                Err(e) => CommandResponse::err(e.to_string(), start.elapsed().as_micros() as u64),
            },
            StarkCommand::Consolidate {
                half_life_days,
                prune_below,
            } => match graph.consolidate(half_life_days, prune_below) {
                Ok(pruned) => {
                    let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                    r.message = Some(format!(
                        "Memory consolidated: {} weak synapses pruned",
                        pruned
                    ));
                    r.pruned = Some(pruned);
                    r
                }
                Err(e) => CommandResponse::err(e.to_string(), start.elapsed().as_micros() as u64),
            },
            _ => CommandResponse::err(
                "Unsupported write command on graph",
                start.elapsed().as_micros() as u64,
            ),
        }
    }

    pub fn execute(self, env: &mut StarkEnvironmentManager) -> CommandResponse {
        let start = Instant::now();
        match self {
            StarkCommand::Ping => {
                let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                r.message = Some("PONG // STARKDB ULTIMATE LEGACY (v1.0.0)".to_string());
                r
            }
            StarkCommand::ListDatabases => {
                let dbs = env.list_databases();
                let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                r.databases = Some(dbs);
                r
            }
            StarkCommand::CreateDatabase { name } => match env.create_database(&name) {
                Ok(info) => {
                    let dbs = env.list_databases();
                    let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                    r.message = Some(format!(
                        "Database '{}' created ({:.1} KB)",
                        info.name,
                        info.size_bytes as f64 / 1024.0
                    ));
                    r.databases = Some(dbs);
                    r
                }
                Err(e) => CommandResponse::err(e.to_string(), start.elapsed().as_micros() as u64),
            },
            StarkCommand::UseDatabase { name } => match env.use_database(&name) {
                Ok(info) => {
                    let dbs = env.list_databases();
                    let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                    r.message = Some(format!("Switched to database '{}'", info.name));
                    r.databases = Some(dbs);
                    r
                }
                Err(e) => CommandResponse::err(e.to_string(), start.elapsed().as_micros() as u64),
            },
            StarkCommand::DropDatabase { name } => match env.drop_database(&name) {
                Ok(dropped) => {
                    let dbs = env.list_databases();
                    let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                    r.message = Some(format!("Database '{}' deleted", dropped));
                    r.databases = Some(dbs);
                    r
                }
                Err(e) => CommandResponse::err(e.to_string(), start.elapsed().as_micros() as u64),
            },
            StarkCommand::TruncateDatabase { name } => match env.truncate_database(&name) {
                Ok(info) => {
                    let dbs = env.list_databases();
                    let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                    r.message = Some(format!("Database '{}' emptied (reset to 7.6 KB)", info.name));
                    r.databases = Some(dbs);
                    r
                }
                Err(e) => CommandResponse::err(e.to_string(), start.elapsed().as_micros() as u64),
            },
            StarkCommand::UpdateDbConfig {
                name,
                api_key_enabled,
                allow_remote,
            } => match env.update_db_config(&name, api_key_enabled, allow_remote) {
                Ok(_info) => {
                    let dbs = env.list_databases();
                    let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                    r.message = Some(format!("Connection settings updated for '{}'", name));
                    r.databases = Some(dbs);
                    r
                }
                Err(e) => CommandResponse::err(e.to_string(), start.elapsed().as_micros() as u64),
            },
            StarkCommand::RegenerateApiKey { name } => match env.regenerate_api_key(&name) {
                Ok(_info) => {
                    let dbs = env.list_databases();
                    let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                    r.message = Some(format!("New API key generated for '{}'", name));
                    r.databases = Some(dbs);
                    r
                }
                Err(e) => CommandResponse::err(e.to_string(), start.elapsed().as_micros() as u64),
            },
            StarkCommand::Hardware => {
                let prof = crate::hardware::HardwareProfile::detect("databases");
                let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                r.message = Some(format!(
                    "Hardware Profile: {} Cores | {:.1} GB RAM | Mode: {}",
                    prof.cpu_cores,
                    prof.total_ram_mb as f64 / 1024.0,
                    prof.mode_name
                ));
                r.hardware = Some(prof);
                r
            }
            other_cmd => {
                let is_import = matches!(other_cmd, StarkCommand::ImportDatabase { .. });
                let target_name = other_cmd
                    .target_db_override()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| env.active_db_name.clone());
                let Some(handle) = env.databases.get(&target_name).cloned() else {
                    return CommandResponse::err(
                        if target_name.is_empty() {
                            "No database is currently mounted. Create a database first.".to_string()
                        } else {
                            format!("Database '{}' is not mounted", target_name)
                        },
                        start.elapsed().as_micros() as u64,
                    );
                };
                let pct = handle.write_lock_percentiles_us();
                let lock_start = Instant::now();
                let Ok(mut g) = handle.graph.try_write() else {
                    return CommandResponse::err(
                        format!("Database '{}' is currently busy", target_name),
                        start.elapsed().as_micros() as u64,
                    );
                };
                let mut resp = other_cmd.execute_on_graph(&mut g, &target_name, pct, start);
                handle.record_write_lock_us(lock_start.elapsed().as_micros() as u64);
                handle.refresh_cached_metrics(&g);
                drop(g);
                if is_import {
                    resp.databases = Some(env.list_databases());
                }
                resp
            }
        }
    }

    pub fn execute_readonly(self, env: &StarkEnvironmentManager) -> CommandResponse {
        let start = Instant::now();
        match self {
            StarkCommand::Ping => {
                let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                r.message = Some("PONG // STARKDB ULTIMATE LEGACY (v1.0.0)".to_string());
                r
            }
            StarkCommand::ListDatabases => {
                let list = env.list_databases();
                let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                r.databases = Some(list);
                r
            }
            StarkCommand::Hardware => {
                let prof = crate::hardware::HardwareProfile::detect("databases");
                let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                r.message = Some(format!(
                    "Hardware Profile: {} Cores | {:.1} GB RAM | Mode: {}",
                    prof.cpu_cores,
                    prof.total_ram_mb as f64 / 1024.0,
                    prof.mode_name
                ));
                r.hardware = Some(prof);
                r
            }
            other_cmd => {
                let target_name = other_cmd
                    .target_db_override()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| env.active_db_name.clone());
                let Some(handle) = env.databases.get(&target_name).cloned() else {
                    return CommandResponse::err(
                        if target_name.is_empty() {
                            "No database is currently mounted. Create a database first.".to_string()
                        } else {
                            format!("Database '{}' is not mounted", target_name)
                        },
                        start.elapsed().as_micros() as u64,
                    );
                };
                let pct = handle.write_lock_percentiles_us();
                let Ok(g) = handle.graph.try_read() else {
                    return CommandResponse::err(
                        format!("Database '{}' is currently busy", target_name),
                        start.elapsed().as_micros() as u64,
                    );
                };
                other_cmd.execute_readonly_on_graph(&g, &target_name, pct, start)
            }
        }
    }
}
