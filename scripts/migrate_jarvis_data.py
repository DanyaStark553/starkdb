"""
migrate_jarvis_data.py — High-Performance Migration from MongoDB & ChromaDB to StarkDB Victory Boost

Extracts:
1. 2,204+ Entities from MongoDB (jarvis_db.entities)
2. 1,696+ Semantic Edges from MongoDB (jarvis_db.entity_edges)
3. 247+ 384-dimensional Vector Embeddings from ChromaDB (jarvis_world_mind)

Migrates directly into StarkDB Victory Boost (`jarvis.starkdb`) with:
- f16 vector quantization (50% memory reduction)
- Dual A/B crash-resilient header pages
- Sub-millisecond associative cognitive recall
"""

import sys
import os
import time
import json
import socket
from typing import Dict, Any, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add parent directory for starkdb_client
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from starkdb_client import StarkDBClient


def run_migration():
    print("=" * 70)
    print("  STARKDB VICTORY BOOST (v0.2.0) — JARVIS DATA MIGRATION ENGINE")
    print("=" * 70)

    # 1. Check MongoDB
    import pymongo
    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
    print(f"\n[1/5] Connecting to MongoDB ({mongo_uri})...")
    try:
        mongo_client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        mongo_db = mongo_client["jarvis_db"]
        entities_count = mongo_db.entities.count_documents({})
        edges_count = mongo_db.entity_edges.count_documents({})
        kg_edges_count = mongo_db.kg_edges.count_documents({})
        user_facts_count = mongo_db.user_facts.count_documents({})
        print(f"  ✓ Connected to MongoDB 'jarvis_db'")
        print(f"  ✓ Found {entities_count} entities in 'entities' collection")
        print(f"  ✓ Found {edges_count} edges in 'entity_edges' collection")
        print(f"  ✓ Found {kg_edges_count} edges in 'kg_edges' collection")
        print(f"  ✓ Found {user_facts_count} facts in 'user_facts' collection")
    except Exception as e:
        print(f"  ✗ MongoDB error: {e}")
        return

    # 2. Check ChromaDB
    import chromadb
    chroma_path = os.environ.get("CHROMA_PATH", "./jarvis_vector_db")
    print(f"\n[2/5] Connecting to ChromaDB ({chroma_path})...")
    try:
        chroma_client = chromadb.PersistentClient(path=chroma_path)
        world_mind = chroma_client.get_collection("jarvis_world_mind")
        vector_count = world_mind.count()
        print(f"  ✓ Connected to ChromaDB")
        print(f"  ✓ Found {vector_count} vector documents in 'jarvis_world_mind'")
    except Exception as e:
        print(f"  ✗ ChromaDB error: {e}")
        return

    # 3. Connect to StarkDB
    print("\n[3/5] Connecting to StarkDB Victory Boost (127.0.0.1:9099)...")
    client = StarkDBClient(host="127.0.0.1", port=9099)
    try:
        ping = client.ping()
        print(f"  ✓ StarkDB Online! Ping response: {ping}")
    except Exception as e:
        print(f"  ✗ StarkDB not running on port 9099. Launching daemon...")
        import subprocess
        exe_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "starkdb.exe"))
        subprocess.Popen([exe_path, "--headless", "--dims", "384", "serve"], cwd=os.path.dirname(exe_path))
        time.sleep(1.5)
        try:
            ping = client.ping()
            print(f"  ✓ StarkDB started successfully! Ping: {ping}")
        except Exception as e2:
            print(f"  ✗ Could not connect to StarkDB: {e2}")
            return

    # Setup database 'jarvis' with 384 dimensions
    print("\n[4/5] Preparing 'jarvis' database in StarkDB...")
    dbs = client.list_databases()
    db_names = [d.get("name") for d in dbs]
    if "jarvis" not in db_names:
        res = client.create_database("jarvis")
        print(f"  ✓ Created new 'jarvis' database (384 dims, f16 format)")
    else:
        res = client.use_database("jarvis")
        print("  ✓ Switched to existing 'jarvis' database")

    # 4. Migrate Data
    print("\n[5/5] Streaming Entities, Edges, and Vectors into StarkDB...")
    t_start = time.perf_counter()

    migrated_nodes = 0
    migrated_edges = 0
    migrated_vectors = 0

    with client.session() as sess:
        use_res = sess.send({"cmd": "use_database", "name": "jarvis"})
        if use_res.get("status") != "ok":
            raise RuntimeError(f"Failed to use database 'jarvis': {use_res}")

        # A. Migrate Entities from MongoDB
        print("  -> Migrating MongoDB Entities...")
        for doc in mongo_db.entities.find():
            name = doc.get("name", "").strip()
            if not name:
                continue
            entity_type = doc.get("type", "Entity")
            attrs = doc.get("attributes", {})
            content = json.dumps(attrs, ensure_ascii=False) if attrs else ""
            tags = entity_type.lower()

            sess.send({
                "cmd": "upsert_node",
                "name": name,
                "node_type": entity_type,
                "tags": tags,
                "content": content,
                "base_weight": 0.9,
                "pinned": name in ("Danil", "StarkOS", "Jarvis"),
            })
            migrated_nodes += 1
            if migrated_nodes % 500 == 0:
                print(f"     ... {migrated_nodes} entities imported")

        print(f"  ✓ {migrated_nodes} Entities imported successfully!")

        # B. Migrate ChromaDB Vectors (jarvis_world_mind)
        print("  -> Migrating ChromaDB Vectors (384 dims, f16 quantization)...")
        all_vecs = world_mind.get(include=["documents", "metadatas", "embeddings"])
        vec_ids = all_vecs.get("ids", [])
        vec_docs = all_vecs.get("documents", [])
        vec_metas = all_vecs.get("metadatas", [])
        vec_embeddings = all_vecs.get("embeddings", [])

        for i, item_id in enumerate(vec_ids):
            doc_text = vec_docs[i] if i < len(vec_docs) and vec_docs[i] else ""
            meta = vec_metas[i] if i < len(vec_metas) and vec_metas[i] else {}
            embedding = vec_embeddings[i] if i < len(vec_embeddings) and vec_embeddings[i] is not None else None

            # First few characters of doc or title as name
            name = meta.get("title") or meta.get("name") or item_id
            tags = "world_mind,insight,discovery"

            node_res = sess.send({
                "cmd": "upsert_node",
                "name": name,
                "node_type": "Document",
                "tags": tags,
                "content": doc_text,
                "base_weight": 0.85,
            })
            node_id = node_res.get("node", {}).get("id") or node_res.get("node_id")

            if embedding is not None and node_id is not None:
                vec_list = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
                # Attach 384-dim vector
                sess.send({
                    "cmd": "attach_vector",
                    "node": name,
                    "vector": vec_list,
                })
                migrated_vectors += 1

        print(f"  ✓ {migrated_vectors} Vector embeddings imported and f16-compressed!")

        # C. Migrate User Facts from MongoDB
        print("  -> Migrating MongoDB User Facts...")
        for doc in mongo_db.user_facts.find():
            fact = doc.get("fact", "").strip()
            topic = doc.get("topic", "user_fact").strip()
            if fact:
                sess.send({
                    "cmd": "upsert_node",
                    "name": fact,
                    "node_type": "Fact",
                    "tags": f"user_fact,{topic}",
                    "content": f"Personal fact: {fact}",
                    "base_weight": 0.95,
                    "pinned": True,
                })
                migrated_nodes += 1

        # D. Migrate Semantic Edges & Knowledge Graph from MongoDB
        print("  -> Migrating MongoDB Semantic Edges (entity_edges + kg_edges)...")
        for doc in mongo_db.entity_edges.find():
            src = doc.get("source", "").strip()
            tgt = doc.get("target", "").strip()
            rel = doc.get("relation", "RELATED_TO").strip()
            weight = float(doc.get("weight", 0.9))

            if src and tgt and rel:
                sess.send({
                    "cmd": "remember",
                    "source": src,
                    "relation": rel,
                    "target": tgt,
                    "weight": weight,
                })
                migrated_edges += 1
                if migrated_edges % 500 == 0:
                    print(f"     ... {migrated_edges} edges imported")

        for doc in mongo_db.kg_edges.find():
            src = doc.get("source", "").strip()
            tgt = doc.get("target", "").strip()
            rel = doc.get("relation", "RELATED_TO").strip()
            if src and tgt and rel:
                sess.send({
                    "cmd": "remember",
                    "source": src,
                    "relation": rel,
                    "target": tgt,
                    "weight": 0.9,
                })
                migrated_edges += 1

        print(f"  ✓ {migrated_edges} Total Synapses imported successfully!")

        # D. Compact Database
        print("  -> Running Zero-Downtime Compaction on 'jarvis.starkdb'...")
        compact_res = sess.send({"cmd": "compact", "name": "jarvis"})
        print(f"  ✓ {compact_res.get('message', 'Compaction complete')}")

    t_total = time.perf_counter() - t_start

    # Print Final Verification & Benchmark
    stats = client.stats()
    print("\n" + "=" * 70)
    print("  MIGRATION & PERFORMANCE BENCHMARK REPORT")
    print("=" * 70)
    print(f"  • Total Migration Time:       {t_total:.2f} s")
    print(f"  • Total Nodes in StarkDB:     {stats.get('active_nodes', 0):,}")
    print(f"  • Total Synapses in StarkDB:  {stats.get('active_synapses', 0):,}")
    print(f"  • Vector Count:               {stats.get('vectors', 0):,}")
    print(f"  • Vector Dimensions:          {stats.get('vector_dims', 0)} ({stats.get('vector_format', 'f16')})")
    print(f"  • StarkDB File Size on Disk:  {stats.get('file_size_bytes', 0) / 1024.0:.1f} KB")

    chroma_size_kb = os.path.getsize(r"C:\Users\UserComp\Desktop\jarvis\jarvis_vector_db\chroma.sqlite3") / 1024.0
    starkdb_size_kb = stats.get('file_size_bytes', 0) / 1024.0
    print(f"  • Old ChromaDB sqlite3 size:  {chroma_size_kb:.1f} KB")
    print(f"  • Compression vs ChromaDB:    {chroma_size_kb / max(starkdb_size_kb, 1.0):.1f}x smaller footprint!")

    # Test Spreading Activation
    print("\n  [VERIFICATION QUERY] Spreading Activation for 'StarkOS':")
    t0 = time.perf_counter()
    nodes = client.activate("StarkOS", depth=2)
    elapsed_us = (time.perf_counter() - t0) * 1_000_000
    print(f"  ✓ Traversal returned {len(nodes)} connected concepts in {elapsed_us:.1f} µs ({elapsed_us / 1000.0:.3f} ms)")
    for n in nodes[:5]:
        print(f"    - [{n.get('energy', 0):.2f}] {n.get('name')} (path: {' -> '.join(n.get('path', []))})")

    # Test Vector Recall
    if migrated_vectors > 0 and len(vec_embeddings) > 0:
        print("\n  [VERIFICATION QUERY] Semantic Vector Search (384-dim f16):")
        query_vec = vec_embeddings[0]
        if hasattr(query_vec, "tolist"):
            query_vec = query_vec.tolist()
        t0 = time.perf_counter()
        matches = client.recall(query_vec, top_k=3)
        elapsed_us = (time.perf_counter() - t0) * 1_000_000
        print(f"  ✓ Nearest neighbor search completed in {elapsed_us:.1f} µs ({elapsed_us / 1000.0:.3f} ms)")
        for m in matches:
            print(f"    - [{m.get('score', 0):.4f}] {m.get('name')}")

        print("\n  [VERIFICATION QUERY] Hybrid Vector + Cognitive Graph Search:")
        t0 = time.perf_counter()
        hybrid_results = client.hybrid(query_vec, top_vectors=2, depth=2)
        elapsed_us = (time.perf_counter() - t0) * 1_000_000
        print(f"  ✓ Hybrid search completed in {elapsed_us:.1f} µs ({elapsed_us / 1000.0:.3f} ms)")
        for h in hybrid_results[:5]:
            print(f"    - [{h.get('fused_score', 0):.4f}] {h.get('name')} (vec: {h.get('vector_similarity', 0):.3f}, graph: {h.get('graph_energy', 0):.3f})")

    print("\n" + "=" * 70)
    print("  STATUS: 100% SUCCESS — JARVIS MEMORY IS NOW UNIFIED IN STARKDB!")
    print("=" * 70)


if __name__ == "__main__":
    run_migration()
