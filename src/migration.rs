use crate::graph::StarkGraph;
use anyhow::{bail, Result};
use serde_json::{json, Value};

pub fn export_database(graph: &StarkGraph, db_name: &str, format: &str) -> Result<String> {
    let nodes = graph.list_nodes(None, None, None, 100_000);
    let synapses = graph.list_synapses(None, 200_000);

    match format.trim().to_lowercase().as_str() {
        "sql" | "mysql" | "postgres" | "sqlite" => {
            let mut out = String::new();
            out.push_str(&format!(
                "-- StarkDB: Ultimate Legacy (v1.0.0) SQL Migration Dump\n-- Source Database: {}\n-- Nodes: {}, Synapses: {}\n\n",
                db_name,
                nodes.len(),
                synapses.len()
            ));
            out.push_str(
                "CREATE TABLE IF NOT EXISTS stark_nodes (\n  id INTEGER PRIMARY KEY,\n  name VARCHAR(255) NOT NULL,\n  node_type VARCHAR(64),\n  tags TEXT,\n  content TEXT,\n  base_weight REAL,\n  pinned INTEGER,\n  activation_count INTEGER\n);\n\n",
            );
            out.push_str(
                "CREATE TABLE IF NOT EXISTS stark_synapses (\n  id INTEGER PRIMARY KEY,\n  source_id INTEGER,\n  source_name VARCHAR(255),\n  relation VARCHAR(128),\n  target_id INTEGER,\n  target_name VARCHAR(255),\n  weight REAL,\n  activation_count INTEGER\n);\n\n",
            );

            for n in &nodes {
                let tags_str = n.tags.join(",");
                out.push_str(&format!(
                    "INSERT INTO stark_nodes (id, name, node_type, tags, content, base_weight, pinned, activation_count) VALUES ({}, '{}', '{}', '{}', '{}', {:.4}, {}, {});\n",
                    n.id,
                    sql_escape(&n.name),
                    sql_escape(&n.node_type),
                    sql_escape(&tags_str),
                    sql_escape(&n.content),
                    n.base_weight,
                    if n.pinned { 1 } else { 0 },
                    n.activation_count
                ));
            }
            out.push('\n');
            for s in &synapses {
                out.push_str(&format!(
                    "INSERT INTO stark_synapses (id, source_id, source_name, relation, target_id, target_name, weight, activation_count) VALUES ({}, {}, '{}', '{}', {}, '{}', {:.4}, {});\n",
                    s.id,
                    s.source_id,
                    sql_escape(&s.source_name),
                    sql_escape(&s.relation),
                    s.target_id,
                    sql_escape(&s.target_name),
                    s.weight,
                    s.activation_count
                ));
            }
            Ok(out)
        }
        "mongo" | "mongodb" | "bson" => {
            let mongo_nodes: Vec<Value> = nodes
                .iter()
                .map(|n| {
                    let parsed_content = serde_json::from_str::<Value>(&n.content)
                        .unwrap_or_else(|_| Value::String(n.content.clone()));
                    json!({
                        "_id": n.id,
                        "name": n.name,
                        "node_type": n.node_type,
                        "tags": n.tags,
                        "content": parsed_content,
                        "base_weight": n.base_weight,
                        "pinned": n.pinned,
                        "activation_count": n.activation_count
                    })
                })
                .collect();

            let mongo_edges: Vec<Value> = synapses
                .iter()
                .map(|s| {
                    json!({
                        "_id": s.id,
                        "source": s.source_name,
                        "relation": s.relation,
                        "target": s.target_name,
                        "weight": s.weight,
                        "activation_count": s.activation_count
                    })
                })
                .collect();

            let bundle = json!({
                "engine": "StarkDB: Ultimate Legacy (v1.0.0)",
                "database": db_name,
                "collections": {
                    "nodes": mongo_nodes,
                    "entity_edges": mongo_edges
                }
            });
            Ok(serde_json::to_string_pretty(&bundle)?)
        }
        "chroma" | "chromadb" => {
            let mut ids = Vec::new();
            let mut documents = Vec::new();
            let mut metadatas = Vec::new();

            for n in &nodes {
                ids.push(format!("node_{}", n.id));
                documents.push(if n.content.is_empty() {
                    n.name.clone()
                } else {
                    n.content.clone()
                });
                metadatas.push(json!({
                    "name": n.name,
                    "node_type": n.node_type,
                    "tags": n.tags.join(","),
                    "base_weight": n.base_weight,
                    "pinned": n.pinned
                }));
            }

            let relations: Vec<Value> = synapses
                .iter()
                .map(|s| {
                    json!({
                        "source": s.source_name,
                        "relation": s.relation,
                        "target": s.target_name,
                        "weight": s.weight
                    })
                })
                .collect();

            let chroma_bundle = json!({
                "collection_name": db_name,
                "ids": ids,
                "documents": documents,
                "metadatas": metadatas,
                "relations": relations
            });
            Ok(serde_json::to_string_pretty(&chroma_bundle)?)
        }
        "starkql" | "native" => {
            let mut out = String::new();
            out.push_str(&format!("-- StarkQL Backup for database '{}'\n", db_name));
            for n in &nodes {
                let tags = n.tags.join(",");
                let pin_str = if n.pinned { " PINNED" } else { "" };
                out.push_str(&format!(
                    "UPSERT NODE \"{}\" TYPE \"{}\" TAGS \"{}\" CONTENT '{}' WEIGHT {:.3}{}\n",
                    n.name.replace('"', "\\\""),
                    n.node_type,
                    tags.replace('"', "\\\""),
                    n.content.replace('\'', "\\'"),
                    n.base_weight,
                    pin_str
                ));
            }
            for s in &synapses {
                out.push_str(&format!(
                    "REMEMBER \"{}\" \"{}\" \"{}\" {:.3}\n",
                    s.source_name.replace('"', "\\\""),
                    s.relation.replace('"', "\\\""),
                    s.target_name.replace('"', "\\\""),
                    s.weight
                ));
            }
            Ok(out)
        }
        other => bail!(
            "Unsupported export format '{}'. Use: sql, mongo, chromadb, or starkql",
            other
        ),
    }
}

pub fn import_database(
    graph: &mut StarkGraph,
    format: &str,
    payload: &str,
) -> Result<(usize, usize)> {
    let trimmed = payload.trim();
    if trimmed.is_empty() {
        bail!("Import payload is empty");
    }

    match format.trim().to_lowercase().as_str() {
        "mongo" | "mongodb" | "bson" => import_from_mongo(graph, trimmed),
        "chroma" | "chromadb" => import_from_chroma(graph, trimmed),
        "sql" | "mysql" | "postgres" | "sqlite" => import_from_sql(graph, trimmed),
        "starkql" | "native" => import_from_starkql(graph, trimmed),
        other => bail!(
            "Unsupported import format '{}'. Use: mongo, chromadb, sql, or starkql",
            other
        ),
    }
}

fn import_from_mongo(graph: &mut StarkGraph, payload: &str) -> Result<(usize, usize)> {
    let mut nodes_added = 0usize;
    let mut synapses_added = 0usize;

    if let Ok(val) = serde_json::from_str::<Value>(payload) {
        if let Some(obj) = val.as_object() {
            // Check for {"collections": {"nodes": [...], "entity_edges": [...]}} or {"nodes": [...], "entity_edges": [...]}
            let root = obj
                .get("collections")
                .and_then(|c| c.as_object())
                .unwrap_or(obj);

            if let Some(arr) = root.get("nodes").and_then(|v| v.as_array()) {
                for item in arr {
                    if import_single_mongo_doc(graph, item, &mut nodes_added, &mut synapses_added) {
                        // handled
                    }
                }
            }
            for edge_key in ["entity_edges", "synapses", "edges", "relations"] {
                if let Some(arr) = root.get(edge_key).and_then(|v| v.as_array()) {
                    for item in arr {
                        let _ = import_single_mongo_doc(
                            graph,
                            item,
                            &mut nodes_added,
                            &mut synapses_added,
                        );
                    }
                }
            }

            if nodes_added > 0 || synapses_added > 0 {
                return Ok((nodes_added, synapses_added));
            }

            // Single document object
            let _ = import_single_mongo_doc(graph, &val, &mut nodes_added, &mut synapses_added);
            return Ok((nodes_added, synapses_added));
        } else if let Some(arr) = val.as_array() {
            for item in arr {
                let _ = import_single_mongo_doc(graph, item, &mut nodes_added, &mut synapses_added);
            }
            return Ok((nodes_added, synapses_added));
        }
    }

    // Fallback: JSONL (one JSON document per line, standard mongoexport format)
    for line in payload.lines() {
        let l = line.trim();
        if l.is_empty() {
            continue;
        }
        if let Ok(item) = serde_json::from_str::<Value>(l) {
            let _ = import_single_mongo_doc(graph, &item, &mut nodes_added, &mut synapses_added);
        }
    }

    Ok((nodes_added, synapses_added))
}

fn import_single_mongo_doc(
    graph: &mut StarkGraph,
    item: &Value,
    nodes_added: &mut usize,
    synapses_added: &mut usize,
) -> bool {
    let Some(obj) = item.as_object() else {
        return false;
    };

    // Check if this is an edge / synapse document (e.g. Jarvis entity_edges)
    let src = obj
        .get("source")
        .or_else(|| obj.get("source_name"))
        .or_else(|| obj.get("from"))
        .and_then(|v| v.as_str());
    let dst = obj
        .get("target")
        .or_else(|| obj.get("target_name"))
        .or_else(|| obj.get("to"))
        .and_then(|v| v.as_str());

    if let (Some(s), Some(t)) = (src, dst) {
        let rel = obj
            .get("relation")
            .or_else(|| obj.get("rel"))
            .or_else(|| obj.get("type"))
            .and_then(|v| v.as_str())
            .unwrap_or("RELATED_TO");
        let w = obj
            .get("weight")
            .and_then(|v| v.as_f64())
            .map(|f| f as f32)
            .unwrap_or(0.9);
        if graph.remember(s, rel, t, w).is_ok() {
            *synapses_added += 1;
            return true;
        }
    }

    // Otherwise treat as a Node document
    let name = obj
        .get("name")
        .or_else(|| obj.get("title"))
        .or_else(|| obj.get("concept"))
        .or_else(|| obj.get("key"))
        .or_else(|| obj.get("_id"))
        .map(|v| match v {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        });

    if let Some(node_name) = name {
        let node_type = obj
            .get("node_type")
            .or_else(|| obj.get("type"))
            .or_else(|| obj.get("category"))
            .and_then(|v| v.as_str())
            .unwrap_or("Document");

        let tags_str = match obj.get("tags") {
            Some(Value::Array(arr)) => arr
                .iter()
                .filter_map(|x| x.as_str())
                .collect::<Vec<_>>()
                .join(","),
            Some(Value::String(s)) => s.clone(),
            _ => String::new(),
        };

        let content_str = match obj.get("content") {
            Some(Value::String(s)) => s.clone(),
            Some(other) => serde_json::to_string(other).unwrap_or_default(),
            None => serde_json::to_string(obj).unwrap_or_default(),
        };

        let weight = obj
            .get("base_weight")
            .or_else(|| obj.get("weight"))
            .and_then(|v| v.as_f64())
            .map(|f| f as f32);

        let pinned = obj.get("pinned").and_then(|v| v.as_bool());

        if graph
            .upsert_node(
                &node_name,
                Some(node_type),
                Some(&tags_str),
                Some(&content_str),
                weight,
                pinned,
            )
            .is_ok()
        {
            *nodes_added += 1;
            return true;
        }
    }

    false
}

fn import_from_chroma(graph: &mut StarkGraph, payload: &str) -> Result<(usize, usize)> {
    let val: Value = serde_json::from_str(payload)?;
    let mut nodes_added = 0usize;
    let mut synapses_added = 0usize;

    if let Some(obj) = val.as_object() {
        let ids = obj.get("ids").and_then(|v| v.as_array());
        let docs = obj.get("documents").and_then(|v| v.as_array());
        let metas = obj.get("metadatas").and_then(|v| v.as_array());
        let embeds = obj.get("embeddings").and_then(|v| v.as_array());

        if let Some(id_list) = ids {
            for (i, id_val) in id_list.iter().enumerate() {
                let raw_id = id_val.as_str().unwrap_or("chroma_item");
                let doc_str = docs
                    .and_then(|d| d.get(i))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let meta_obj = metas.and_then(|m| m.get(i)).and_then(|v| v.as_object());

                let name = meta_obj
                    .and_then(|m| m.get("name").or_else(|| m.get("title")))
                    .and_then(|v| v.as_str())
                    .unwrap_or(raw_id);
                let node_type = meta_obj
                    .and_then(|m| m.get("node_type").or_else(|| m.get("type")))
                    .and_then(|v| v.as_str())
                    .unwrap_or("Document");
                let tags = meta_obj
                    .and_then(|m| m.get("tags"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("chromadb");

                if let Ok(view) =
                    graph.upsert_node(name, Some(node_type), Some(tags), Some(doc_str), None, None)
                {
                    nodes_added += 1;
                    if let Some(emb_arr) =
                        embeds.and_then(|e| e.get(i)).and_then(|v| v.as_array())
                    {
                        let vec_f32: Vec<f32> = emb_arr
                            .iter()
                            .filter_map(|x| x.as_f64().map(|f| f as f32))
                            .collect();
                        if !vec_f32.is_empty() {
                            let _ = graph.attach_vector(view.id, &vec_f32);
                        }
                    }
                }
            }
        }

        if let Some(rels) = obj.get("relations").and_then(|v| v.as_array()) {
            for r in rels {
                let _ = import_single_mongo_doc(graph, r, &mut nodes_added, &mut synapses_added);
            }
        }
    } else if let Some(arr) = val.as_array() {
        for item in arr {
            let _ = import_single_mongo_doc(graph, item, &mut nodes_added, &mut synapses_added);
        }
    }

    Ok((nodes_added, synapses_added))
}

fn import_from_sql(graph: &mut StarkGraph, payload: &str) -> Result<(usize, usize)> {
    let mut nodes_added = 0usize;
    let mut synapses_added = 0usize;

    for line in payload.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("--") {
            continue;
        }
        let upper = trimmed.to_uppercase();
        if !upper.starts_with("INSERT INTO") {
            continue;
        }

        let Some(values_pos) = upper.find("VALUES") else {
            continue;
        };
        let header_part = &trimmed[..values_pos];
        let values_part = &trimmed[values_pos + 6..];

        let is_synapse_table = header_part.to_lowercase().contains("synapse")
            || header_part.to_lowercase().contains("edge");

        let tuple_str = values_part
            .trim()
            .trim_end_matches(';')
            .trim()
            .trim_start_matches('(')
            .trim_end_matches(')');
        let fields = parse_sql_tuple(tuple_str);

        if is_synapse_table {
            // Either (id, source_id, source_name, relation, target_id, target_name, weight, ...)
            // or (source, relation, target, weight)
            if fields.len() >= 7 {
                let src = &fields[2];
                let rel = &fields[3];
                let dst = &fields[5];
                let w = fields[6].parse::<f32>().unwrap_or(0.9);
                if graph.remember(src, rel, dst, w).is_ok() {
                    synapses_added += 1;
                }
            } else if fields.len() >= 3 {
                let src = &fields[0];
                let rel = &fields[1];
                let dst = &fields[2];
                let w = fields.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.9);
                if graph.remember(src, rel, dst, w).is_ok() {
                    synapses_added += 1;
                }
            }
        } else if fields.len() >= 2 {
            // (id, name, node_type, tags, content, base_weight, pinned, ...)
            let name = if fields[0].parse::<u64>().is_ok() {
                &fields[1]
            } else {
                &fields[0]
            };
            let node_type = fields.get(2).map(|s| s.as_str()).unwrap_or("Entity");
            let tags = fields.get(3).map(|s| s.as_str()).unwrap_or("");
            let content = fields.get(4).map(|s| s.as_str()).unwrap_or("");
            let weight = fields.get(5).and_then(|s| s.parse::<f32>().ok());
            let pinned = fields.get(6).map(|s| s == "1" || s.eq_ignore_ascii_case("true"));

            if graph
                .upsert_node(name, Some(node_type), Some(tags), Some(content), weight, pinned)
                .is_ok()
            {
                nodes_added += 1;
            }
        }
    }

    Ok((nodes_added, synapses_added))
}

fn import_from_starkql(graph: &mut StarkGraph, payload: &str) -> Result<(usize, usize)> {
    let mut nodes_added = 0usize;
    let mut synapses_added = 0usize;

    for line in payload.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("--") || trimmed.starts_with('#') {
            continue;
        }
        if let Ok(cmd) = crate::protocol::StarkCommand::parse_line(trimmed) {
            match cmd {
                crate::protocol::StarkCommand::UpsertNode {
                    name,
                    node_type,
                    tags,
                    content,
                    base_weight,
                    pinned,
                } => {
                    if graph
                        .upsert_node(
                            &name,
                            node_type.as_deref(),
                            tags.as_deref(),
                            content.as_deref(),
                            base_weight,
                            pinned,
                        )
                        .is_ok()
                    {
                        nodes_added += 1;
                    }
                }
                crate::protocol::StarkCommand::Remember {
                    source,
                    relation,
                    target,
                    weight,
                } if graph.remember(&source, &relation, &target, weight).is_ok() => {
                    synapses_added += 1;
                }
                _ => {}
            }
        }
    }

    Ok((nodes_added, synapses_added))
}

fn sql_escape(s: &str) -> String {
    s.replace('\'', "''")
}

fn parse_sql_tuple(input: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut in_quote = false;
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];
        if in_quote {
            if c == '\'' {
                if i + 1 < chars.len() && chars[i + 1] == '\'' {
                    current.push('\'');
                    i += 2;
                    continue;
                } else {
                    in_quote = false;
                }
            } else {
                current.push(c);
            }
        } else if c == '\'' {
            in_quote = true;
        } else if c == ',' {
            result.push(current.trim().to_string());
            current.clear();
        } else {
            current.push(c);
        }
        i += 1;
    }
    result.push(current.trim().to_string());
    result
}
