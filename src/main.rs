use anyhow::Result;
use clap::{Parser, Subcommand};
use starkdb::graph::StarkEnvironmentManager;
use starkdb::hardware;
use starkdb::server::{set_autostart, StarkServer};
use std::io::{self, BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(name = "starkdb")]
#[command(author = "Danya <starkos>")]
#[command(version = "1.0.0")]
#[command(
    about = "StarkDB // Ultimate Legacy (v1.0.0) — Open-Source Zero-Copy Hybrid Vector (HNSW SQ8), Lexical (BM25F) & Associative Graph Database",
    long_about = None
)]
struct Cli {
    #[arg(short, long, default_value = "brain.starkdb")]
    db_path: PathBuf,

    /// Raw TCP IPC port for Jarvis / Python client
    #[arg(short, long, default_value_t = 9099)]
    port: u16,

    /// Desktop Workbench port
    #[arg(long, default_value_t = 7474)]
    http_port: u16,

    #[arg(long, default_value_t = 768)]
    dims: u16,

    /// Run headless in background without opening the desktop GUI window
    #[arg(long, default_value_t = false)]
    headless: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Ensure background daemon is running and open StarkDB Workbench window
    Workbench,
    /// Ensure StarkDB background daemon is running silently in the background (no GUI window)
    Daemon,
    /// Stop the running background StarkDB daemon
    Stop,
    /// Configure automatic background startup on OS boot
    Autostart {
        #[arg(long, default_value_t = true)]
        enable: bool,
        #[arg(long, default_value_t = false)]
        disable: bool,
    },
    /// Internal foreground server loop (used by the detached background daemon)
    Serve,
    /// Interactive CLI REPL
    Repl,
    /// Quick command to remember an association
    Remember {
        source: String,
        relation: String,
        target: String,
        #[arg(default_value_t = 0.9)]
        weight: f32,
    },
    /// Quick command to activate context from a concept (Bidirectional)
    Activate {
        concept: String,
        #[arg(short, long, default_value_t = 2)]
        depth: usize,
        #[arg(long, default_value_t = 0.75)]
        decay: f32,
    },
    /// Print DB & daemon stats
    Stats,
    /// Deep integrity audit & self-healing (God Reboot Engine)
    Scrub {
        #[arg(short, long)]
        db: Option<String>,
    },
    /// Audit and display detected hardware profile (Zero-Configuration Engine)
    Hardware,
}

fn is_daemon_online(port: u16) -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    TcpStream::connect_timeout(&addr, Duration::from_millis(180)).is_ok()
}

fn send_daemon_command(port: u16, line: &str) -> Result<String> {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_millis(800))?;
    stream.write_all(line.as_bytes())?;
    stream.write_all(b"\n")?;
    stream.flush()?;
    let mut reader = BufReader::new(stream);
    let mut resp = String::new();
    reader.read_line(&mut resp)?;
    Ok(resp)
}

fn spawn_detached_daemon(cli: &Cli) -> Result<bool> {
    if is_daemon_online(cli.port) {
        return Ok(false); // Already running
    }

    let exe = std::env::current_exe()?;
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("--db-path")
        .arg(&cli.db_path)
        .arg("--port")
        .arg(cli.port.to_string())
        .arg("--http-port")
        .arg(cli.http_port.to_string())
        .arg("--dims")
        .arg(cli.dims.to_string())
        .arg("--headless")
        .arg("serve");

    cmd.stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        const DETACHED_PROCESS: u32 = 0x00000008;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x00000200;
        cmd.creation_flags(CREATE_NO_WINDOW | DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP);
    }

    cmd.spawn()?;

    // Wait up to 2.5 seconds for the background daemon to bind TCP port
    for _ in 0..25 {
        if is_daemon_online(cli.port) {
            return Ok(true);
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    Ok(is_daemon_online(cli.port))
}

fn open_desktop_app_window(port: u16) {
    let url = format!("http://127.0.0.1:{}", port);

    // Prefer native Tauri workbench window
    let candidates = [
        PathBuf::from("starkdb-workbench.exe"),
        PathBuf::from("target/release/starkdb-workbench.exe"),
        PathBuf::from("src-tauri/target/release/starkdb-workbench.exe"),
    ];

    if let Ok(curr_exe) = std::env::current_exe() {
        if let Some(parent) = curr_exe.parent() {
            let candidate = parent.join("starkdb-workbench.exe");
            if candidate.exists() && std::process::Command::new(candidate).spawn().is_ok() {
                return;
            }
        }
    }

    for candidate in &candidates {
        if candidate.exists() && std::process::Command::new(candidate).spawn().is_ok() {
            return;
        }
    }

    #[cfg(target_os = "windows")]
    {
        // Fallback standalone desktop app window (no address bar, no browser tabs)
        let _ = std::process::Command::new("cmd")
            .args([
                "/C",
                "start",
                "",
                "msedge",
                &format!("--app={}", url),
                "--window-size=1380,860",
            ])
            .spawn();
    }
    #[cfg(target_os = "linux")]
    {
        let _ = std::process::Command::new("xdg-open").arg(&url).spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(&url).spawn();
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    let cli = Cli::parse();

    // If `--headless` was passed without a subcommand, treat it as background daemon launch
    let selected_cmd = match cli.command {
        Some(ref c) => c,
        None => {
            if cli.headless {
                &Commands::Daemon
            } else {
                &Commands::Workbench
            }
        }
    };

    match selected_cmd {
        Commands::Workbench => {
            let started_new = spawn_detached_daemon(&cli)?;
            if started_new {
                println!(
                    "👑 StarkDB // Ultimate Legacy (v1.0.0) background service started (127.0.0.1:{})",
                    cli.port
                );
            } else {
                println!(
                    "👑 StarkDB // Ultimate Legacy (v1.0.0) background service is already online (127.0.0.1:{})",
                    cli.port
                );
            }
            println!(
                "🖥️  Opening StarkDB Workbench (127.0.0.1:{}) — closing the window leaves the DB running in background.",
                cli.http_port
            );
            open_desktop_app_window(cli.http_port);
        }
        Commands::Daemon => {
            let started_new = spawn_detached_daemon(&cli)?;
            if started_new {
                println!(
                    "✅ StarkDB // Ultimate Legacy (v1.0.0) background daemon started silently on 127.0.0.1:{} (GUI endpoint: 127.0.0.1:{})",
                    cli.port, cli.http_port
                );
            } else {
                println!(
                    "✅ StarkDB // Ultimate Legacy (v1.0.0) background daemon is already running on 127.0.0.1:{}",
                    cli.port
                );
            }
        }
        Commands::Stop => {
            #[cfg(target_os = "windows")]
            {
                let my_pid = std::process::id();
                let _ = std::process::Command::new("taskkill")
                    .args([
                        "/F",
                        "/FI",
                        &format!("PID ne {}", my_pid),
                        "/IM",
                        "starkdb.exe",
                    ])
                    .output();
            }
            #[cfg(target_os = "linux")]
            {
                let _ = std::process::Command::new("pkill")
                    .args(["-f", "starkdb.*serve"])
                    .output();
            }
            println!("🛑 StarkDB background service stopped.");
        }
        Commands::Autostart { enable, disable } => {
            let target_state = if *disable { false } else { *enable };
            let now_enabled = set_autostart(target_state)?;
            if now_enabled {
                println!("✅ StarkDB background service registered for automatic startup on OS boot.");
            } else {
                println!("ℹ️  StarkDB automatic startup on OS boot disabled.");
            }
        }
        Commands::Serve => {
            let env = StarkEnvironmentManager::new(&cli.db_path, cli.dims)?;
            println!("==============================================================");
            println!("   👑 StarkDB // Ultimate Legacy (v1.0.0) — Server Engine");
            println!("   Primary DB File : {:?}", cli.db_path);
            println!("   IPC TCP Socket  : 127.0.0.1:{}", cli.port);
            println!("   Workbench Port  : 127.0.0.1:{}", cli.http_port);
            println!("==============================================================");

            if !cli.headless {
                let port = cli.http_port;
                std::thread::spawn(move || {
                    std::thread::sleep(Duration::from_millis(350));
                    open_desktop_app_window(port);
                });
            }

            let server = StarkServer::new(env, cli.port, cli.http_port);
            server.run_all().await?;
        }
        Commands::Repl => {
            let _ = spawn_detached_daemon(&cli);
            println!("StarkDB // Ultimate Legacy (v1.0.0) Interactive REPL (connected to 127.0.0.1:{}). Type 'exit' to quit.", cli.port);
            let stdin = io::stdin();
            let mut stdout = io::stdout();

            loop {
                print!("starkdb> ");
                stdout.flush()?;

                let mut input = String::new();
                if stdin.lock().read_line(&mut input)? == 0 {
                    break;
                }

                let trimmed = input.trim();
                if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
                    break;
                }
                if trimmed.is_empty() {
                    continue;
                }

                match send_daemon_command(cli.port, trimmed) {
                    Ok(raw) => {
                        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&raw) {
                            println!("{}", serde_json::to_string_pretty(&val)?);
                        } else {
                            println!("{}", raw.trim());
                        }
                    }
                    Err(e) => println!("Error communicating with daemon: {}", e),
                }
            }
        }
        Commands::Remember {
            source,
            relation,
            target,
            weight,
        } => {
            let _ = spawn_detached_daemon(&cli);
            let q = format!(
                "REMEMBER \"{}\" \"{}\" \"{}\" {}",
                source, relation, target, weight
            );
            let resp = send_daemon_command(cli.port, &q)?;
            println!("{}", resp.trim());
        }
        Commands::Activate {
            concept,
            depth,
            decay,
        } => {
            let _ = spawn_detached_daemon(&cli);
            let q = format!("ACTIVATE \"{}\" DEPTH {} DECAY {}", concept, depth, decay);
            let resp = send_daemon_command(cli.port, &q)?;
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&resp) {
                println!("{}", serde_json::to_string_pretty(&val)?);
            } else {
                println!("{}", resp.trim());
            }
        }
        Commands::Stats => {
            let _ = spawn_detached_daemon(&cli);
            let resp = send_daemon_command(cli.port, "STATS")?;
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&resp) {
                println!("{}", serde_json::to_string_pretty(&val)?);
            } else {
                println!("{}", resp.trim());
            }
        }
        Commands::Scrub { db } => {
            let _ = spawn_detached_daemon(&cli);
            let cmd = if let Some(target) = db {
                format!("SCRUB \"{}\"", target)
            } else {
                "SCRUB".to_string()
            };
            let resp = send_daemon_command(cli.port, &cmd)?;
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&resp) {
                println!("{}", serde_json::to_string_pretty(&val)?);
            } else {
                println!("{}", resp.trim());
            }
        }
        Commands::Hardware => {
            let prof = hardware::HardwareProfile::detect("databases");
            println!("{}", serde_json::to_string_pretty(&prof)?);
        }
    }

    Ok(())
}
