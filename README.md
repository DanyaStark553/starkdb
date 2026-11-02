# StarkDB 👑 (Ultimate Legacy)

**High-Performance Embedded Hybrid Graph & Vector Database for AI Agents & Cognitive Systems**

[![Rust](https://img.shields.io/badge/rust-stable-brightgreen.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)
[![Zero-Copy](https://img.shields.io/badge/architecture-zero--copy%20mmap-orange.svg)]()
[![SIMD](https://img.shields.io/badge/simd-AVX2%20%2B%20ARM64%20NEON-red.svg)]()
[![Crash-Proof](https://img.shields.io/badge/durability-A%2FB%20Dual%20Header%20%2B%20CRC32-blueviolet.svg)]()

---

## Executive Summary

Modern Autonomous AI Agents and Local LLMs struggle with long-term memory. Typical agent architectures glue together three disparate databases:
1. **Neo4j** for entity relationship graphs (heavy JVM footprint, 1.5GB+ idle RAM, multi-millisecond network hops).
2. **Chroma / Pinecone / Qdrant** for vector similarity search.
3. **Elasticsearch / Meilisearch** for keyword search.

**StarkDB** replaces this fragmented stack with a **single, zero-dependency, ultra-fast embedded engine written in Rust**. It unifies **Associative Graph Traversal** (Spreading Activation), **Vector Similarity** (HNSW with SQ8 SIMD quantization), and **Lexical Full-Text Search** (BM25F) into a single memory-mapped binary.

---

## Key Architectural Highlights

* **Zero-Copy Memory-Mapped Engine (`mmap`):** Direct mapped binary layout (48-byte node record, 32-byte synapse record). Zero serialization overhead; instant cold starts (<15 ms).
* **Tri-Channel Hybrid Retrieval (`bound_attack`):** Fuses Vector similarity, Lexical BM25F, Graph Spreading Activation, and exponential recency decay into an optimal Pareto-ranked context.
* **Symmetric SQ8 Vector Quantization:** Compresses embeddings by 75% (from 32-bit floats to 8-bit integers) with dynamic CPU dispatch:
  * **x86_64:** Hardware **AVX2 + FMA** integer dot product with zero saturation risk on `[-127, 127]`.
  * **ARM64:** Native **NEON** dual-lane accumulation (`vmull_s8` / `vpadalq_s16`).
  * **Fallback:** Highly-unrolled 4-way scalar kernel.
* **A/B Dual Header Pages (1000-Crash Durability):** 256-byte cacheline-aligned dual headers with monotonically increasing transaction generation counters and CRC32 checksums. Immune to sudden power outages and hard process terminations (`SIGKILL`).
* **Non-Blocking Shadow Vacuum:** Online 4-phase defragmentation and compaction without stalling reader or writer threads.
* **Standalone 3D Workbench GUI:** Integrated Web & Desktop GUI with an interactive Canvas-based 3D graph visualizer, table inspector, query console, and live hardware telemetry. Zero external CDN dependencies—runs in air-gapped environments.

---

## Objective Comparative Benchmarks

*All benchmarks measured on an Intel Core x86_64 workstation with NVMe storage under release profile (LTO enabled).*

| Feature / Metric | **StarkDB v1.0** | **Neo4j (Community)** | **ChromaDB / Qdrant** | **SQLite (Relational)** |
| :--- | :--- | :--- | :--- | :--- |
| **Architectural Model** | **Unified Hybrid (Graph + Vector + BM25)** | Graph Only (LPG) | Vector Only (Dense Embeddings) | Relational (B-Tree) |
| **Idle Memory Footprint** | **~18 – 28 MB** (mmap demand-paged) | **~1.2 – 2.4 GB** (JVM runtime) | **~350 – 800 MB** (Python/Rust) | **~5 – 10 MB** |
| **Cold Boot Latency** | **< 15 ms** (via `.hnsw.cache`) | **12 – 25 seconds** (JVM init) | **1.5 – 3.5 seconds** | **< 5 ms** |
| **3-Hop Associative Walk** | **~3.2 µs** (Lock-Free BitSet) | **~4.5 – 12.0 ms** (Recursive Cypher) | *Not Supported* | **~8 – 15 ms** (Recursive CTE) |
| **Vector Dot Product** | **110 – 160 µs** (SQ8 SIMD) | *Plugin / Extension required* | **~250 – 700 µs** (f32) | *External extension only* |
| **Power-Loss Durability** | **A/B Dual Header + CRC32** | Write-Ahead Log (WAL) | WAL / Periodic Snapshots | WAL / Rollback Journal |
| **External Dependencies** | **Zero (Single self-contained binary)** | Java JRE 17+ | Python runtime / Docker daemon | Zero |

---

## Command Syntax & API Reference

StarkDB exposes an intuitive command language across the CLI REPL, Raw TCP Socket (`127.0.0.1:9099`), and HTTP REST API (`127.0.0.1:7474`).

### 1. Associative Memory & Graph Operations

```sql
-- Remember a directed relationship with an associative weight (0.0 to 1.0)
REMEMBER <source_entity> <relation> <target_entity> [weight]
-- Example:
REMEMBER "AutonomousAgent" "USES" "StarkDB" 0.95

-- Activate context using Spreading Activation (energy propagation across hops)
-- Syntax: ACTIVATE <seed_concept> [depth=2] [decay=0.75] [threshold=0.10]
ACTIVATE "StarkDB" 2 0.75 0.10

-- Insert or update a knowledge node
UPSERT NODE "StarkDB" TYPE="Software" TAGS="rust,database,mmap" CONTENT="Zero-copy hybrid DB" WEIGHT=1.0 PINNED=true

-- Query nodes or synapses
FIND NODES [filter_keyword]
FIND SYNAPSES [filter_keyword]
```

### 2. Vector Search (HNSW SQ8)

```sql
-- Attach an embedding vector to an existing node
ATTACH_VECTOR "StarkDB" [0.034, -0.128, 0.452, ...]

-- Find nearest neighbors by vector similarity (Cosine / Dot product)
RECALL [0.034, -0.128, 0.452, ...] [top_k=5]
```

### 3. Tri-Channel Hybrid Retrieval (`BOUND_ATTACK`)

Execute a multi-modal query across Vector space, Lexical BM25F index, and Graph topology simultaneously:

```json
{
  "cmd": "bound_attack",
  "text": "embedded memory-mapped database",
  "vector": [0.034, -0.128, 0.452, ...],
  "anchors": ["StarkDB"],
  "tags": ["rust"],
  "top_k": 5,
  "depth": 2,
  "weights": {
    "w_v": 0.45,
    "w_t": 0.30,
    "w_g": 0.20,
    "w_r": 0.05
  }
}
```

### 4. Database Administration & Self-Healing

```sql
PING                     -- Heartbeat check
STATS                    -- Print active node/synapse counts, HNSW layers, and memory metrics
SCRUB                    -- Run Two-Phase Self-Healing audit and slot recycling
VACUUM                   -- Execute non-blocking Shadow Vacuum compaction
HARDWARE                 -- Probe system seek penalty, RAM capacity, and optimal page sizes
```

---

## Python Client SDK Example

A lightweight, zero-dependency client is included in [`starkdb_client.py`](starkdb_client.py):

```python
from starkdb_client import StarkDBClient

# Connect to the local StarkDB daemon (default 127.0.0.1:9099)
client = StarkDBClient()

# 1. Store knowledge graph associations
client.remember("Tony Stark", "CREATED", "JarvisCore", weight=0.98)
client.remember("JarvisCore", "RUNS_ON", "StarkDB", weight=1.0)

# 2. Attach vector embeddings
client.attach_vector("StarkDB", [0.012, -0.045, 0.891, ...])

# 3. Associative Recall for LLM Prompt Augmentation
activated = client.activate("Tony Stark", depth=2, decay=0.8)
for node in activated:
    print(f"Node: {node['name']} | Activation Energy: {node['energy']:.3f}")

# 4. Hybrid Tri-Channel Search (Vector + Text + Graph)
hits = client.bound_attack(
    text="high-performance associative memory",
    vector=[0.012, -0.045, 0.891, ...],
    top_k=5
)
for hit in hits:
    print(f"Hit: {hit['name']} | Score: {hit['score']:.4f}")
```

---

## HTTP REST API

StarkDB provides an HTTP interface at `http://127.0.0.1:7474`:

```bash
# Execute any StarkDB command via JSON query
curl -X POST http://127.0.0.1:7474/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "FIND NODES"}'

# Retrieve real-time server telemetry
curl http://127.0.0.1:7474/api/stats
```

---

## Quickstart & Installation

### Prerequisites
* Rust 1.75+ (`cargo`, `rustc`)

### 1. Build from Source
```bash
git clone https://github.com/DanyaStark553/starkdb.git
cd starkdb
cargo build --release
```

The optimized binary will be located at `target/release/starkdb` (or `starkdb.exe` on Windows).

### 2. Run the Background Daemon & Desktop Workbench
```bash
# Starts the background daemon and opens the interactive 3D Workbench
./target/release/starkdb workbench

# Or run headless server directly
./target/release/starkdb --headless serve
```

### 3. Run Benchmarks
```bash
cargo bench
```

### 4. Run Stress & Durability Tests
```bash
cargo test
```

---

## Security & Privacy

* **Strict Localhost Binding:** StarkDB binds exclusively to `127.0.0.1` by default. No external traffic can connect unless explicitly configured.
* **No Telemetry / No Call-Home:** StarkDB never connects to external servers or collects telemetry. Your knowledge graph and vector embeddings remain 100% on your local storage.

---

## License

This project is dual-licensed under either:
* **MIT License** ([LICENSE](LICENSE))
* **Apache License, Version 2.0**

Choose the license that best fits your commercial or open-source requirements.
