use crate::graph::{DbHandle, StarkEnvironmentManager, VacuumStats};
use crate::protocol::{CommandResponse, StarkCommand};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::RwLock;
use tracing::{error, info};

const INDEX_HTML: &str = include_str!("../ui/index.html");
const LOGO_PNG: &[u8] = include_bytes!("../assets/logo_banner.png");
const ICON_PNG: &[u8] = include_bytes!("../assets/icon.png");
const FAVICON_ICO: &[u8] = include_bytes!("../assets/favicon.ico");

#[derive(Deserialize)]
struct RawQueryWrapper {
    query: String,
    #[serde(default)]
    db: Option<String>,
    #[serde(default)]
    api_key: Option<String>,
}

#[derive(Deserialize)]
struct AutostartRequest {
    enable: bool,
}

#[derive(Serialize)]
pub struct DaemonStatusInfo {
    pub status: String,
    pub daemon_mode: String,
    pub pid: u32,
    pub tcp_port: u16,
    pub http_port: u16,
    pub autostart_enabled: bool,
    pub executable: String,
}

pub fn is_autostart_enabled() -> bool {
    #[cfg(target_os = "windows")]
    {
        if let Ok(output) = std::process::Command::new("reg")
            .args([
                "query",
                r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run",
                "/v",
                "StarkDB_WingAccel",
            ])
            .output()
        {
            return output.status.success();
        }
        false
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(home) = std::env::var("HOME") {
            let service_path =
                std::path::PathBuf::from(home).join(".config/systemd/user/starkdb.service");
            return service_path.exists();
        }
        false
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        false
    }
}

pub fn set_autostart(enable: bool) -> Result<bool> {
    let exe = std::env::current_exe()?;
    let exe_str = exe.to_string_lossy().to_string();

    #[cfg(target_os = "windows")]
    {
        if enable {
            let cmd_val = format!("\"{}\" daemon", exe_str);
            let _ = std::process::Command::new("reg")
                .args([
                    "add",
                    r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run",
                    "/v",
                    "StarkDB_WingAccel",
                    "/t",
                    "REG_SZ",
                    "/d",
                    &cmd_val,
                    "/f",
                ])
                .output()?;
        } else {
            let _ = std::process::Command::new("reg")
                .args([
                    "delete",
                    r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run",
                    "/v",
                    "StarkDB_WingAccel",
                    "/f",
                ])
                .output();
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(home) = std::env::var("HOME") {
            let dir = std::path::PathBuf::from(home).join(".config/systemd/user");
            let service_path = dir.join("starkdb.service");
            if enable {
                let _ = std::fs::create_dir_all(&dir);
                let workdir = exe
                    .parent()
                    .unwrap_or(std::path::Path::new("."))
                    .to_string_lossy();
                let unit = format!(
                    "[Unit]\nDescription=StarkDB Wing Accel Background Service\nAfter=network.target\n\n[Service]\nType=simple\nWorkingDirectory={}\nExecStart={} serve --headless\nRestart=always\n\n[Install]\nWantedBy=default.target\n",
                    workdir, exe_str
                );
                let _ = std::fs::write(&service_path, unit);
            } else if service_path.exists() {
                let _ = std::fs::remove_file(&service_path);
            }
        }
    }
    Ok(is_autostart_enabled())
}

pub struct StarkServer {
    env: Arc<RwLock<StarkEnvironmentManager>>,
    tcp_port: u16,
    http_port: u16,
}

impl StarkServer {
    pub fn new(env: StarkEnvironmentManager, tcp_port: u16, http_port: u16) -> Self {
        Self {
            env: Arc::new(RwLock::new(env)),
            tcp_port,
            http_port,
        }
    }

    /// Executes four-phase ID-stable Shadow Vacuum with non-blocking chunked copy
    /// so reader and writer threads on `handle` are never starved during compaction.
    pub async fn run_nonblocking_vacuum(
        handle: &Arc<DbHandle>,
        force: bool,
    ) -> Result<VacuumStats> {
        let (ticket, mut shadow) = {
            let lock_start = Instant::now();
            let mut g = handle.graph.write().await;
            let should = g
                .vacuum_plan(force)
                .map(|p| p.should_vacuum)
                .unwrap_or(false);
            if !should {
                let size = g.storage.file_size_bytes();
                let seq = g.storage.active_seq;
                let epoch = g.storage.header().vacuum_epoch;
                let nodes = g.active_node_count();
                let syns = g.active_synapse_count();
                let vecs = g.storage.header().vector_count;
                handle.record_write_lock_us(lock_start.elapsed().as_micros() as u64);
                return Ok(VacuumStats {
                    old_seq: seq,
                    new_seq: seq,
                    vacuum_epoch: epoch,
                    original_size_bytes: size,
                    compacted_size_bytes: size,
                    bytes_saved: 0,
                    active_nodes: nodes,
                    active_synapses: syns,
                    vectors: vecs,
                    catch_up_rounds: 0,
                });
            }
            let res = g.vacuum_begin();
            handle.record_write_lock_us(lock_start.elapsed().as_micros() as u64);
            res?
        };

        let mut cursor = 0u32;
        loop {
            let next = (cursor + 1024).min(ticket.node_count);
            let copy_res = {
                let g = handle.graph.read().await;
                g.vacuum_copy_chunk(&ticket, &mut shadow, cursor..next)
            };
            match copy_res {
                Ok(done) => {
                    cursor = next;
                    if done || cursor >= ticket.node_count {
                        if !shadow.synapses_copied {
                            let fin_res = {
                                let g = handle.graph.read().await;
                                g.vacuum_copy_chunk(
                                    &ticket,
                                    &mut shadow,
                                    ticket.node_count..ticket.node_count,
                                )
                            };
                            if let Err(e) = fin_res {
                                let mut g = handle.graph.write().await;
                                g.vacuum_abort(Some(shadow));
                                return Err(e);
                            }
                        }
                        break;
                    }
                }
                Err(e) => {
                    let mut g = handle.graph.write().await;
                    g.vacuum_abort(Some(shadow));
                    return Err(e);
                }
            }
            tokio::task::yield_now().await;
        }

        for _ in 0..8 {
            let pending = {
                let lock_start = Instant::now();
                let mut g = handle.graph.write().await;
                let res = g.vacuum_catch_up(&mut shadow);
                handle.record_write_lock_us(lock_start.elapsed().as_micros() as u64);
                match res {
                    Ok(p) => p,
                    Err(e) => {
                        g.vacuum_abort(Some(shadow));
                        return Err(e);
                    }
                }
            };
            if pending <= 32 {
                break;
            }
            tokio::task::yield_now().await;
        }

        let stats = {
            let lock_start = Instant::now();
            let mut g = handle.graph.write().await;
            let res = g.vacuum_finish(shadow);
            handle.record_write_lock_us(lock_start.elapsed().as_micros() as u64);
            if res.is_ok() {
                handle.refresh_cached_metrics(&g);
            }
            res?
        };

        Ok(stats)
    }

    /// Dispatches a `StarkCommand` using fine-grained per-DB locking (`DbHandle::graph`)
    /// so operations on one database never block readers or writers on other databases.
    pub async fn dispatch_command(
        env: &Arc<RwLock<StarkEnvironmentManager>>,
        cmd: StarkCommand,
        target_db: Option<String>,
        provided_key: Option<String>,
    ) -> CommandResponse {
        let start = Instant::now();

        match &cmd {
            StarkCommand::Ping => {
                let mut r = CommandResponse::ok(start.elapsed().as_micros() as u64);
                r.message = Some("PONG // STARKDB ULTIMATE LEGACY (v1.0.0)".to_string());
                return r;
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
                return r;
            }
            StarkCommand::ListDatabases => {
                let e = env.read().await;
                return cmd.execute_readonly(&e);
            }
            _ if cmd.is_env_command() => {
                let mut e = env.write().await;
                return cmd.execute(&mut e);
            }
            _ => {}
        }

        // Resolve target DB handle under a brief shared read lock on the environment registry
        let (db_name, handle_opt) = {
            let e = env.read().await;
            let mut resolved = cmd.target_db_override().map(|s| s.to_string());
            if resolved.is_none() {
                if let Some(ref key) = provided_key {
                    if let Some(key_db) = e.resolve_db_by_api_key(key) {
                        resolved = Some(key_db);
                    }
                }
            }
            if resolved.is_none() {
                if let Some(ref tdb) = target_db {
                    if !tdb.trim().is_empty() {
                        resolved = Some(tdb.trim().to_string());
                    }
                }
            }
            let name = resolved.unwrap_or_else(|| e.active_db_name.clone());
            let handle = e.databases.get(&name).cloned();
            (name, handle)
        };

        // Keep active_db_name synced if the HTTP workbench explicitly selected a different DB
        if let Some(ref tdb) = target_db {
            let trimmed = tdb.trim();
            if !trimmed.is_empty() && handle_opt.is_some() {
                let needs_switch = {
                    let e = env.read().await;
                    e.active_db_name != trimmed
                };
                if needs_switch {
                    let mut e = env.write().await;
                    let _ = e.use_database(trimmed);
                }
            }
        }

        let Some(handle) = handle_opt else {
            return CommandResponse::err(
                if db_name.is_empty() {
                    "No database is currently mounted. Create a database first.".to_string()
                } else {
                    format!("Database '{}' is not mounted", db_name)
                },
                start.elapsed().as_micros() as u64,
            );
        };

        if cmd.is_read_only() {
            let pct = handle.write_lock_percentiles_us();
            let g = handle.graph.read().await;
            return cmd.execute_readonly_on_graph(&g, &db_name, pct, start);
        }

        if let StarkCommand::Vacuum { force, .. } = cmd {
            match Self::run_nonblocking_vacuum(&handle, force).await {
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
                    return r;
                }
                Err(e) => {
                    return CommandResponse::err(
                        e.to_string(),
                        start.elapsed().as_micros() as u64,
                    );
                }
            }
        }

        let is_import = matches!(cmd, StarkCommand::ImportDatabase { .. });
        let pct = handle.write_lock_percentiles_us();
        let mut resp = {
            let lock_start = Instant::now();
            let mut g = handle.graph.write().await;
            let r = cmd.execute_on_graph(&mut g, &db_name, pct, start);
            handle.record_write_lock_us(lock_start.elapsed().as_micros() as u64);
            handle.refresh_cached_metrics(&g);
            r
        };

        if is_import {
            let e = env.read().await;
            resp.databases = Some(e.list_databases());
        }
        resp
    }

    pub async fn run_all(&self) -> Result<()> {
        let tcp_env = Arc::clone(&self.env);
        let tcp_port = self.tcp_port;

        let http_env = Arc::clone(&self.env);
        let http_port = self.http_port;

        let tcp_task = tokio::spawn(async move {
            if let Err(e) = Self::serve_tcp(tcp_env, tcp_port).await {
                error!("TCP server error: {}", e);
            }
        });

        let http_task = tokio::spawn(async move {
            if let Err(e) = Self::serve_http(http_env, tcp_port, http_port).await {
                error!("HTTP GUI server error: {}", e);
            }
        });

        // Multi-phase non-blocking ScrubSentinel (v1.0 Ultimate Legacy):
        // Iterates per-DB handles every 60s without holding the environment lock.
        let scrub_env = Arc::clone(&self.env);
        let scrub_task = tokio::spawn(async move {
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;

                let handles: Vec<(String, Arc<DbHandle>)> = {
                    let guard = scrub_env.read().await;
                    guard
                        .databases
                        .iter()
                        .map(|(k, v)| (k.clone(), Arc::clone(v)))
                        .collect()
                };

                for (db_name, handle) in handles {
                    // Phase 1: Read-only verification + planning under per-DB read lock
                    let ro_state = {
                        let g = handle.graph.read().await;
                        let ro_rep = g.check_integrity_readonly();
                        let has_rollback = g.storage.rollback_seq != u64::MAX;
                        let dirty_ops = g.telemetry.dirty_ops.load(Ordering::Relaxed);
                        let hot_plan = if dirty_ops > 0 {
                            Some(g.plan_hot_tier())
                        } else {
                            None
                        };
                        let vac_plan = g.vacuum_plan(false);
                        let needs_sq8 =
                            g.storage.header().vector_count >= 4096 && !g.storage.has_sq8();
                        (
                            ro_rep,
                            has_rollback,
                            dirty_ops,
                            hot_plan,
                            vac_plan,
                            needs_sq8,
                        )
                    };

                    let (ro_rep, has_rollback, dirty_ops, hot_plan, vac_plan, needs_sq8) = ro_state;
                    match ro_rep {
                        Ok(rep) => {
                            if rep.healthy && rep.header_a_valid && rep.header_b_valid {
                                tracing::debug!(
                                    "🛡️ ScrubSentinel (Ultimate Legacy): '{}' PASS (100% HEALTHY, {} nodes, {} synapses).",
                                    db_name, rep.valid_nodes, rep.valid_synapses
                                );

                                if has_rollback || hot_plan.is_some() || needs_sq8 {
                                    let lock_start = Instant::now();
                                    let mut g = handle.graph.write().await;
                                    if has_rollback {
                                        let _ = g.storage.cleanup_rollback_segment();
                                    }
                                    if let Some(plan) = hot_plan {
                                        g.apply_hot_tier(plan);
                                        tracing::debug!(
                                            "🔥 Hot-Tier two-phase apply for '{}': synced {} dirty hits",
                                            db_name,
                                            dirty_ops
                                        );
                                    }
                                    if needs_sq8 && g.storage.set_sq8_mode(true).is_ok() {
                                        info!(
                                            "⚡ ScrubSentinel: Auto-enabled SQ8 quantized vectors for '{}'",
                                            db_name
                                        );
                                    }
                                    let _ = g.storage.flush();
                                    handle.record_write_lock_us(
                                        lock_start.elapsed().as_micros() as u64,
                                    );
                                    handle.refresh_cached_metrics(&g);
                                }

                                if vac_plan.map(|p| p.should_vacuum).unwrap_or(false) {
                                    match Self::run_nonblocking_vacuum(&handle, false).await {
                                        Ok(vstats) => {
                                            info!(
                                                "🧹 ScrubSentinel Shadow Vacuum on '{}' finished: {:.1} KB -> {:.1} KB (saved {:.1} KB, epoch {})",
                                                db_name,
                                                vstats.original_size_bytes as f64 / 1024.0,
                                                vstats.compacted_size_bytes as f64 / 1024.0,
                                                vstats.bytes_saved as f64 / 1024.0,
                                                vstats.vacuum_epoch
                                            );
                                        }
                                        Err(e) => {
                                            tracing::warn!(
                                                "🧹 ScrubSentinel Shadow Vacuum skipped/aborted on '{}': {}",
                                                db_name,
                                                e
                                            );
                                        }
                                    }
                                }

                                // Persist warm-start HNSW & BM25 caches under shared read lock
                                {
                                    let g = handle.graph.read().await;
                                    let _ = g.save_hnsw_cache();
                                    let _ = g.save_bm25_cache();
                                }
                            } else {
                                tracing::warn!(
                                    "⚠️ ScrubSentinel: Anomaly in '{}' (healthy={}, nodes_corrupt={}, syn_corrupt={}). Healing...",
                                    db_name, rep.healthy, rep.corrupted_nodes, rep.corrupted_synapses
                                );
                                let lock_start = Instant::now();
                                let mut g = handle.graph.write().await;
                                match g.scrub_and_self_heal() {
                                    Ok(heal_rep) => {
                                        info!(
                                            "🛡️ ScrubSentinel: Healed {} sector(s) in '{}' in {} µs.",
                                            heal_rep.healed_count, db_name, heal_rep.duration_us
                                        );
                                    }
                                    Err(e) => {
                                        error!(
                                            "🚨 ScrubSentinel healing failure in '{}': {}",
                                            db_name, e
                                        );
                                    }
                                }
                                handle
                                    .record_write_lock_us(lock_start.elapsed().as_micros() as u64);
                                handle.refresh_cached_metrics(&g);
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                "🛡️ ScrubSentinel read-only check error on '{}': {}",
                                db_name,
                                e
                            );
                        }
                    }
                }
            }
        });

        tokio::select! {
            _ = async { tokio::join!(tcp_task, http_task, scrub_task) } => {}
            _ = tokio::signal::ctrl_c() => {
                info!("🛑 Graceful shutdown signal received (Ctrl+C). Flushing telemetry and caches...");
                let handles: Vec<(String, Arc<DbHandle>)> = {
                    let guard = self.env.read().await;
                    guard
                        .databases
                        .iter()
                        .map(|(k, v)| (k.clone(), Arc::clone(v)))
                        .collect()
                };
                for (db_name, handle) in handles {
                    let mut g = handle.graph.write().await;
                    g.sync_telemetry_to_mmap();
                    let _ = g.storage.flush();
                    let _ = g.save_hnsw_cache();
                    let _ = g.save_bm25_cache();
                    info!("✅ Database '{}' flushed and caches saved.", db_name);
                }
            }
        }
        Ok(())
    }

    async fn serve_tcp(env: Arc<RwLock<StarkEnvironmentManager>>, port: u16) -> Result<()> {
        let addr = format!("127.0.0.1:{}", port);
        let listener = TcpListener::bind(&addr)
            .await
            .with_context(|| format!("Failed to bind TCP listener to {}", addr))?;

        info!("⚡ StarkDB IPC socket listening on tcp://{}", addr);

        loop {
            let (socket, _peer) = match listener.accept().await {
                Ok(conn) => conn,
                Err(e) => {
                    error!("Error accepting TCP connection: {}", e);
                    continue;
                }
            };

            let env_clone = Arc::clone(&env);
            tokio::spawn(async move {
                let (reader, mut writer) = socket.into_split();
                let mut buf_reader = BufReader::new(reader);
                let mut line = String::new();

                while let Ok(n) = buf_reader.read_line(&mut line).await {
                    if n == 0 {
                        break;
                    }

                    let response = match StarkCommand::parse_line(&line) {
                        Ok(cmd) => Self::dispatch_command(&env_clone, cmd, None, None).await,
                        Err(err) => CommandResponse::err(err.to_string(), 0),
                    };

                    line.clear();

                    if let Ok(json) = serde_json::to_string(&response) {
                        let _ = writer.write_all(json.as_bytes()).await;
                        let _ = writer.write_all(b"\n").await;
                        let _ = writer.flush().await;
                    }
                }
            });
        }
    }

    async fn serve_http(
        env: Arc<RwLock<StarkEnvironmentManager>>,
        tcp_port: u16,
        http_port: u16,
    ) -> Result<()> {
        let addr = format!("127.0.0.1:{}", http_port);
        let listener = TcpListener::bind(&addr)
            .await
            .with_context(|| format!("Failed to bind HTTP GUI listener to {}", addr))?;

        info!(
            "🚀 StarkDB Ultimate Legacy Workbench active at http://{}",
            addr
        );

        loop {
            let (stream, _) = match listener.accept().await {
                Ok(conn) => conn,
                Err(e) => {
                    error!("HTTP accept error: {}", e);
                    continue;
                }
            };

            let env_clone = Arc::clone(&env);
            tokio::spawn(async move {
                let _ = Self::handle_http_client(stream, env_clone, tcp_port, http_port).await;
            });
        }
    }

    async fn handle_http_client(
        mut stream: TcpStream,
        env: Arc<RwLock<StarkEnvironmentManager>>,
        tcp_port: u16,
        http_port: u16,
    ) -> Result<()> {
        let mut buf = vec![0u8; 65536];
        let mut total_read = 0usize;

        loop {
            let n = stream.read(&mut buf[total_read..]).await?;
            if n == 0 {
                return Ok(());
            }
            total_read += n;
            if find_subslice(&buf[..total_read], b"\r\n\r\n").is_some() {
                break;
            }
            if total_read >= buf.len() {
                break;
            }
        }

        let header_end = find_subslice(&buf[..total_read], b"\r\n\r\n")
            .map(|p| p + 4)
            .unwrap_or(total_read);
        let header_str = String::from_utf8_lossy(&buf[..header_end]).to_string();
        let first_line = header_str.lines().next().unwrap_or("");
        let parts: Vec<&str> = first_line.split_whitespace().collect();
        if parts.len() < 2 {
            return Ok(());
        }
        let method = parts[0];
        let path = parts[1];

        let mut content_length = 0usize;
        let mut header_api_key: Option<String> = None;
        for line in header_str.lines() {
            let lower = line.to_ascii_lowercase();
            if let Some(rest) = lower.strip_prefix("content-length:") {
                content_length = rest.trim().parse().unwrap_or(0);
            } else if let Some(_rest) = lower.strip_prefix("x-stark-key:") {
                header_api_key = Some(line[12..].trim().to_string());
            } else if let Some(_rest) = lower.strip_prefix("authorization:") {
                let val = line[14..].trim();
                if val.to_ascii_lowercase().starts_with("bearer ") {
                    header_api_key = Some(val[7..].trim().to_string());
                } else {
                    header_api_key = Some(val.to_string());
                }
            }
        }

        let mut body_bytes = buf[header_end..total_read].to_vec();
        while body_bytes.len() < content_length {
            let mut chunk = vec![0u8; 8192];
            let n = stream.read(&mut chunk).await?;
            if n == 0 {
                break;
            }
            body_bytes.extend_from_slice(&chunk[..n]);
        }

        if method == "GET" && (path == "/" || path == "/index.html") {
            let html = std::fs::read_to_string("ui/index.html")
                .unwrap_or_else(|_| INDEX_HTML.to_string());
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
                html.len()
            );
            stream.write_all(resp.as_bytes()).await?;
            stream.write_all(html.as_bytes()).await?;
            stream.flush().await?;
            return Ok(());
        }

        if method == "GET" && path.starts_with("/logo.png") {
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
                LOGO_PNG.len()
            );
            stream.write_all(resp.as_bytes()).await?;
            stream.write_all(LOGO_PNG).await?;
            stream.flush().await?;
            return Ok(());
        }

        if method == "GET" && path.starts_with("/icon.png") {
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
                ICON_PNG.len()
            );
            stream.write_all(resp.as_bytes()).await?;
            stream.write_all(ICON_PNG).await?;
            stream.flush().await?;
            return Ok(());
        }

        if method == "GET" && path.starts_with("/favicon.ico") {
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: image/x-icon\r\nContent-Length: {}\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
                FAVICON_ICO.len()
            );
            stream.write_all(resp.as_bytes()).await?;
            stream.write_all(FAVICON_ICO).await?;
            stream.flush().await?;
            return Ok(());
        }

        if method == "GET" && path == "/api/daemon" {
            let exe = std::env::current_exe()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| "starkdb".to_string());
            let info = DaemonStatusInfo {
                status: "ok".to_string(),
                daemon_mode: "Background Service (Detached)".to_string(),
                pid: std::process::id(),
                tcp_port,
                http_port,
                autostart_enabled: is_autostart_enabled(),
                executable: exe,
            };
            let json = serde_json::to_string(&info)?;
            let http_resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                json.len(),
                json
            );
            stream.write_all(http_resp.as_bytes()).await?;
            stream.flush().await?;
            return Ok(());
        }

        if method == "POST" && path == "/api/daemon/autostart" {
            let body_str = String::from_utf8_lossy(&body_bytes);
            let enable = serde_json::from_str::<AutostartRequest>(&body_str)
                .map(|r| r.enable)
                .unwrap_or(false);
            let enabled_now = set_autostart(enable).unwrap_or(false);
            let exe = std::env::current_exe()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| "starkdb".to_string());
            let info = DaemonStatusInfo {
                status: "ok".to_string(),
                daemon_mode: "Background Service (Detached)".to_string(),
                pid: std::process::id(),
                tcp_port,
                http_port,
                autostart_enabled: enabled_now,
                executable: exe,
            };
            let json = serde_json::to_string(&info)?;
            let http_resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                json.len(),
                json
            );
            stream.write_all(http_resp.as_bytes()).await?;
            stream.flush().await?;
            return Ok(());
        }

        if method == "POST" && path == "/api/query" {
            let body_str = String::from_utf8_lossy(&body_bytes);
            let mut target_db: Option<String> = None;
            let mut provided_key: Option<String> = header_api_key;
            let parsed_cmd =
                if let Ok(wrapper) = serde_json::from_str::<RawQueryWrapper>(&body_str) {
                    target_db = wrapper.db;
                    if wrapper.api_key.is_some() {
                        provided_key = wrapper.api_key;
                    }
                    StarkCommand::parse_line(&wrapper.query)
                } else {
                    StarkCommand::parse_line(&body_str)
                };

            let response = match parsed_cmd {
                Ok(cmd) => Self::dispatch_command(&env, cmd, target_db, provided_key).await,
                Err(err) => CommandResponse::err(err.to_string(), 0),
            };

            let json = serde_json::to_string(&response)?;
            let http_resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                json.len(),
                json
            );
            stream.write_all(http_resp.as_bytes()).await?;
            stream.flush().await?;
            return Ok(());
        }

        let not_found =
            "HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\nConnection: close\r\n\r\nNot Found";
        stream.write_all(not_found.as_bytes()).await?;
        stream.flush().await?;
        Ok(())
    }
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}
