use criterion::{black_box, criterion_group, criterion_main, Criterion};
use starkdb::graph::StarkGraph;
use starkdb::storage::StarkStorage;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

fn build_bench_graph(name: &str) -> (PathBuf, StarkGraph) {
    let tmp_dir = std::env::temp_dir().join(format!("starkdb_bench_{}", name));
    let _ = std::fs::remove_dir_all(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).expect("Failed to create bench temp dir");
    let db_path = tmp_dir.join("dataset.starkdb");

    let dims: u16 = 1536;
    let storage = StarkStorage::open_or_create(&db_path, dims).expect("Failed to open storage");
    let mut graph = StarkGraph::new(storage).expect("Failed to init graph");

    println!("⚡ Building benchmark dataset: 5,000 nodes, 10,000 synapses, 1,000 f16 vectors (dim=1536)...");

    // 1. Insert 5,000 nodes
    for i in 0..5000 {
        let node_name = format!("node_{}", i);
        let tags = format!("tag_{},group_{}", i % 25, i % 5);
        let content = format!("Payload content for node {} in StarkDB benchmark", i);
        graph
            .upsert_node(
                &node_name,
                Some("Concept"),
                Some(&tags),
                Some(&content),
                Some(1.0),
                Some(false),
            )
            .expect("Failed to upsert node");
    }

    // 2. Insert 10,000 synapses connecting nodes
    for i in 0..10000 {
        let src = format!("node_{}", i % 5000);
        let tgt = format!("node_{}", (i * 7 + 1) % 5000);
        let rel = format!("rel_{}", i % 12);
        let weight = 0.5 + ((i % 50) as f32) * 0.01;
        let _ = graph.remember(&src, &rel, &tgt, weight);
    }

    // 3. Attach 1,000 f16 vectors (dim=1536) to first 1,000 nodes
    for i in 0..1000 {
        let mut vec = vec![0.0f32; dims as usize];
        for (d, val) in vec.iter_mut().enumerate().take(dims as usize) {
            *val = ((i * 13 + d) as f32 * 0.004).sin();
        }
        graph.attach_vector(i as u32, &vec).expect("Failed to attach vector");
    }

    println!("✅ Benchmark dataset ready: {} nodes, {} synapses, {} vectors",
        graph.active_node_count(),
        graph.active_synapse_count(),
        graph.storage.header().vector_count
    );

    (db_path, graph)
}

fn bench_hot_paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("strike_god_v0_4");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(4));

    let (_db_path, mut graph) = build_bench_graph("hot_paths");

    // 1. bench_spreading_activation_3_hops: 3-hop Lock-Free (&self) traversal on 5,000 nodes / 10,000 synapses
    group.bench_function("bench_spreading_activation_3_hops", |b| {
        b.iter(|| {
            let res = graph.spreading_activation(
                black_box("node_0"),
                black_box(3),
                black_box(0.75),
                black_box(0.10),
            );
            black_box(res).unwrap()
        })
    });

    // 2. bench_f16_cosine_similarity_top10: Search top-10 in 1,000 f16 vectors (dim=1536) via HNSW + AVX2/F16C
    let mut query_vector: Vec<f32> = (0..1536).map(|d| ((d as f32) * 0.007).cos()).collect();
    let q_norm: f32 = query_vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if q_norm > 0.0 {
        for v in &mut query_vector {
            *v /= q_norm;
        }
    }
    group.bench_function("bench_f16_cosine_similarity_top10", |b| {
        b.iter(|| {
            let res = graph.search_vectors(black_box(&query_vector), black_box(10));
            black_box(res).unwrap()
        })
    });

    // 3. bench_readonly_scrub: Full integrity check on 5,000 nodes and 10,000 synapses
    group.bench_function("bench_readonly_scrub", |b| {
        b.iter(|| {
            let res = graph.check_integrity_readonly();
            black_box(res).unwrap()
        })
    });

    // 4. bench_node_insert_and_intern: Insert new node with string interning
    static COUNTER: AtomicUsize = AtomicUsize::new(100_000);
    group.bench_function("bench_node_insert_and_intern", |b| {
        b.iter(|| {
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            let name = format!("intern_test_node_{}", id);
            let tags = "benchmark,interning,hot_path";
            let content = "Benchmark node content to test string interning in string pool";
            let res = graph.upsert_node(
                black_box(&name),
                black_box(Some("BenchNode")),
                black_box(Some(tags)),
                black_box(Some(content)),
                black_box(Some(0.95)),
                black_box(Some(false)),
            );
            black_box(res).unwrap()
        })
    });

    // 5. bench_hnsw_vs_bruteforce_10k_vectors: Search top-10 across 10,000 f16 vectors (dim=1536)
    println!("⚡ Extending dataset to 10,000 nodes & 10,000 f16 vectors (dim=1536) for HNSW 10k benchmark...");
    while graph.storage.header().node_count < 10000 {
        let idx = graph.storage.header().node_count;
        let node_name = format!("vec_node_10k_{}", idx);
        graph
            .get_or_create_node(&node_name, 0)
            .expect("Failed to create node for 10k bench");
    }
    for i in 1000..10000 {
        let mut vec = vec![0.0f32; 1536];
        for (d, val) in vec.iter_mut().enumerate().take(1536) {
            *val = ((i * 17 + d * 3) as f32 * 0.0037).sin()
                + (((i % 97) * 29 + d) as f32 * 0.011).cos() * 0.5;
        }
        graph
            .attach_vector(i as u32, &vec)
            .expect("Failed to attach vector for 10k bench");
    }
    println!(
        "✅ 10k HNSW Index ready: active_vectors={}, max_layer={}",
        graph.hnsw.active_count,
        graph.hnsw.max_layer
    );

    // Measure single-run Brute-Force 10k vs HNSW 10k reference for log verification
    {
        let t_bf = std::time::Instant::now();
        let bf_iters = 20;
        for _ in 0..bf_iters {
            let _ = black_box(graph.search_vectors_bruteforce(black_box(&query_vector), 10).unwrap());
        }
        let avg_bf_us = t_bf.elapsed().as_micros() as f64 / bf_iters as f64;

        let t_hnsw = std::time::Instant::now();
        let hnsw_iters = 20;
        for _ in 0..hnsw_iters {
            let _ = black_box(graph.search_vectors_hnsw(black_box(&query_vector), 10).unwrap());
        }
        let avg_hnsw_us = t_hnsw.elapsed().as_micros() as f64 / hnsw_iters as f64;
        println!(
            "📊 [10k f16 Vectors (dim=1536) Direct Comparison] Brute-Force SIMD: {:.2} µs | HNSW + SIMD: {:.2} µs (Speedup: {:.2}x)",
            avg_bf_us,
            avg_hnsw_us,
            avg_bf_us / avg_hnsw_us.max(0.001)
        );
    }

    group.bench_function("bench_hnsw_vs_bruteforce_10k_vectors", |b| {
        b.iter(|| {
            let res = graph.search_vectors(black_box(&query_vector), black_box(10));
            black_box(res).unwrap()
        })
    });

    // 6. bench_hnsw_sq8_10k: Direct Two-Phase SQ8 AVX2 + F16C rerank on 10,000 vectors (dim=1536)
    let mut query_i8 = [0i8; 1536];
    let q_meta = starkdb::vectors::quantize_sq8(&query_vector, &mut query_i8);
    group.bench_function("bench_hnsw_sq8_10k", |b| {
        b.iter(|| {
            let hits = graph.hnsw.search_two_phase(
                black_box(&query_vector),
                black_box(&query_i8),
                black_box(q_meta.scale),
                black_box(10),
                &graph.storage,
            );
            black_box(hits)
        })
    });

    // 7. bench_bound_attack_10k: Three-channel hybrid retrieval (BM25F + SQ8 HNSW + Graph BFS) on 10k nodes/vectors
    let bound_anchors = [0u32];
    let bound_tags = ["tag_17"];
    let bound_query = starkdb::hybrid::BoundQuery {
        vector: Some(&query_vector),
        text: Some("Payload content node_42 StarkDB benchmark"),
        anchors: &bound_anchors,
        tags_all: &bound_tags,
        node_type: None,
        since: None,
        until: None,
        top_k: 10,
        depth: 2,
        weights: starkdb::hybrid::FusionWeights::default(),
    };
    group.bench_function("bench_bound_attack_10k", |b| {
        b.iter(|| {
            let hits = graph.bound_attack(black_box(&bound_query));
            black_box(hits).unwrap()
        })
    });

    // 8. bench_spreading_activation_16_threads: 16 concurrent reader threads on shared StarkGraph
    {
        use std::sync::{Arc, Barrier};
        let shared_graph = Arc::new(graph);
        let start_barrier = Arc::new(Barrier::new(17));
        let end_barrier = Arc::new(Barrier::new(17));
        let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut workers = Vec::with_capacity(16);

        for _ in 0..16 {
            let g = Arc::clone(&shared_graph);
            let sb = Arc::clone(&start_barrier);
            let eb = Arc::clone(&end_barrier);
            let sf = Arc::clone(&stop_flag);
            workers.push(std::thread::spawn(move || loop {
                sb.wait();
                if sf.load(Ordering::Relaxed) {
                    break;
                }
                let res = g.spreading_activation(
                    black_box("node_0"),
                    black_box(3),
                    black_box(0.75),
                    black_box(0.10),
                );
                let _ = black_box(res);
                eb.wait();
            }));
        }

        group.bench_function("bench_spreading_activation_16_threads", |b| {
            b.iter(|| {
                start_barrier.wait();
                end_barrier.wait();
            })
        });

        stop_flag.store(true, Ordering::Relaxed);
        start_barrier.wait();
        for w in workers {
            let _ = w.join();
        }
    }

    // 8. bench_attach_10k_build_dim256: Geometric growth + SQ8 + f16 arena allocation of 10,000 vectors (dim=256)
    // NOTE: Uses dim=256 specifically to measure allocation throughput independent of 1536-D SIMD math.
    let sample_vec_256: Vec<f32> = (0..256).map(|d| ((d as f32) * 0.013).sin()).collect();
    group.sample_size(10);
    group.bench_function("bench_attach_10k_build_dim256", |b| {
        b.iter(|| {
            let tmp_path = std::env::temp_dir().join("starkdb_bench_10k_build.starkdb");
            let _ = std::fs::remove_file(&tmp_path);
            let mut st = StarkStorage::open_or_create(&tmp_path, 256).unwrap();
            for _ in 0..10_000 {
                let vid = st.write_vector(black_box(&sample_vec_256)).unwrap();
                black_box(vid);
            }
            st.flush().unwrap();
            drop(st);
            let _ = std::fs::remove_file(&tmp_path);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_hot_paths);
criterion_main!(benches);
