//! MCP CLI commands
//!
//! Implements `rk mcp` subcommands for managing MCP servers and tools.
//!
//! When built with `--features daemon`, enables:
//! - Background daemon process for persistent MCP connections
//! - `rk mcp daemon start/stop/status` commands
//! - Automatic daemon detection for tool calls

#![allow(dead_code)]

use clap::{Parser, Subcommand};
use reasonkit::mcp::McpRegistry;

#[cfg(feature = "daemon")]
use reasonkit::mcp::daemon::{call_tool, DaemonManager, DaemonStatus};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Parse arguments
    let cli = McpCli::parse();

    // Run command
    run_mcp_command(cli).await
}

#[derive(Parser)]
#[command(name = "rk-mcp")]
#[command(author = "ReasonKit Team <team@reasonkit.sh>")]
#[command(version)]
#[command(about = "ReasonKit MCP — Model Context Protocol Server")]
#[command(long_about = r#"ReasonKit MCP — Model Context Protocol Server

Part of The Reasoning Engine suite. This tool provides MCP server
functionality, exposing ThinkTools to any MCP-compatible AI client.

THINKTOOLS EXPOSED:
  • GigaThink    - Divergent exploration (10+ perspectives)
  • LaserLogic   - Precision deductive reasoning
  • BedRock      - First principles decomposition
  • ProofGuard   - Multi-source verification
  • BrutalHonesty - Adversarial self-critique

MODES:
  stdio       Standard I/O for direct integration (default)
  http        HTTP server for network access

EXAMPLES:
  rk-mcp serve                  # Start stdio MCP server
  rk-mcp serve --port 3000      # Start HTTP server
  rk-mcp list-tools             # Show available tools
  rk-mcp daemon start           # Run as background daemon

WEBSITE: https://reasonkit.sh
DOCS:    https://reasonkit.sh/docs/mcp-server
"#)]
pub struct McpCli {
    #[command(subcommand)]
    command: McpCommands,
}

#[derive(Subcommand)]
pub enum McpCommands {
    /// Start the MCP server (alias for serve-mcp)
    Serve {
        /// Module to serve (default: all ThinkTools)
        #[arg(long, default_value = "all")]
        module: String,
        /// Port for HTTP mode (optional, default: stdio mode)
        #[arg(long)]
        port: Option<u16>,
    },

    /// List registered MCP servers
    ListServers,

    /// List available tools from all servers
    ListTools,

    /// List available prompts from all servers
    ListPrompts,

    /// Call a tool (uses daemon if running, otherwise direct mode)
    CallTool {
        /// Tool name
        name: String,
        /// JSON arguments
        args: String,
    },

    /// Get a prompt
    GetPrompt {
        /// Prompt name
        name: String,
        /// JSON arguments
        #[arg(default_value = "{}")]
        args: String,
    },

    /// Manage the MCP daemon
    Daemon {
        #[command(subcommand)]
        action: DaemonAction,
    },

    /// Internal: Run as daemon server (called by `daemon start`)
    #[cfg(feature = "daemon")]
    #[command(hide = true)]
    ServeDaemon {
        /// Unix socket path
        #[arg(long)]
        socket: String,
    },
}

#[derive(Subcommand)]
pub enum DaemonAction {
    /// Start the daemon
    Start,
    /// Stop the daemon
    Stop,
    /// Check daemon status
    Status,
    /// Restart the daemon
    Restart,
}

pub async fn run_mcp_command(cli: McpCli) -> anyhow::Result<()> {
    match cli.command {
        McpCommands::Serve { module, port } => {
            if port.is_some() {
                eprintln!("HTTP mode not yet implemented. Using stdio mode.");
            }

            match module.as_str() {
                "all" | "thinktools" => {
                    println!("Starting ReasonKit MCP Server (ThinkTools)...");
                    reasonkit::mcp::server::run_server().await?;
                }
                "delta" => {
                    println!("Starting ReasonKit MCP Server (Delta Protocol)...");
                    // TODO: Implement delta-specific server
                    reasonkit::mcp::server::run_server().await?;
                }
                _ => {
                    eprintln!(
                        "Unknown module: {}. Available: all, thinktools, delta",
                        module
                    );
                    std::process::exit(1);
                }
            }
        }

        McpCommands::ListServers => {
            let registry = McpRegistry::new();
            let servers = registry.list_servers().await;
            if servers.is_empty() {
                println!("No MCP servers registered.");
                println!("\nAvailable built-in servers:");
                println!("  reasonkit       ReasonKit ThinkTools (GigaThink, LaserLogic, etc.)");
                println!("  reasonkit-delta ReasonKit Delta Protocol");
            } else {
                println!("{:<36}  {:<20}  Status", "ID", "Name");
                println!("────────────────────────────────────────────────────────────────");
                for server in servers {
                    let status = if let Some(check) = server.last_health_check {
                        format!("{:?}", check.status)
                    } else {
                        "Unknown".to_string()
                    };
                    println!("{:<36}  {:<20}  {}", server.id, server.name, status);
                }
            }
        }

        McpCommands::ListTools => {
            // List built-in tools even without registry
            println!("{:<20}  {:<20}  Description", "Tool", "Server");
            println!("────────────────────────────────────────────────────────────────");
            println!(
                "{:<20}  {:<20}  Expansive creative thinking (10+ perspectives)",
                "gigathink", "reasonkit"
            );
            println!(
                "{:<20}  {:<20}  Precision deductive reasoning with fallacy detection",
                "laserlogic", "reasonkit"
            );
            println!(
                "{:<20}  {:<20}  First principles decomposition",
                "bedrock", "reasonkit"
            );
            println!(
                "{:<20}  {:<20}  Multi-source verification (3+ sources)",
                "proofguard", "reasonkit"
            );
            println!(
                "{:<20}  {:<20}  Adversarial self-critique",
                "brutalhonesty", "reasonkit"
            );
            println!(
                "{:<20}  {:<20}  Delta Protocol verification",
                "delta_verify", "reasonkit-delta"
            );

            // Also check registry
            let registry = McpRegistry::new();
            if let Ok(tools) = registry.list_all_tools().await {
                for tool in tools {
                    println!(
                        "{:<20}  {:<20}  {}",
                        tool.name,
                        tool.server_name.unwrap_or_default(),
                        tool.description.unwrap_or_default()
                    );
                }
            }
        }

        McpCommands::ListPrompts => {
            let registry = McpRegistry::new();
            match registry.list_all_prompts().await {
                Ok(prompts) => {
                    if prompts.is_empty() {
                        println!("No prompts available.");
                    } else {
                        println!("{:<20}  Description", "Prompt");
                        println!(
                            "────────────────────────────────────────────────────────────────"
                        );
                        for prompt in prompts {
                            println!(
                                "{:<20}  {}",
                                prompt.name,
                                prompt.description.unwrap_or_default()
                            );
                        }
                    }
                }
                Err(e) => eprintln!("Error listing prompts: {}", e),
            }
        }

        McpCommands::CallTool { name, args } => {
            // Parse args as JSON
            let args_json: serde_json::Value = match serde_json::from_str(&args) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("Invalid JSON args: {}", e);
                    std::process::exit(1);
                }
            };

            #[cfg(feature = "daemon")]
            {
                // Daemon feature enabled - call tool via daemon or direct mode
                println!("Calling tool '{}'...", name);

                match call_tool(&name, args_json).await {
                    Ok(result) => {
                        // Print tool result
                        for content in result.content {
                            match content {
                                reasonkit::mcp::tools::ToolResultContent::Text { text } => {
                                    println!("{}", text);
                                }
                                reasonkit::mcp::tools::ToolResultContent::Image {
                                    data,
                                    mime_type,
                                } => {
                                    println!("[Image: {} bytes, {}]", data.len(), mime_type);
                                }
                                reasonkit::mcp::tools::ToolResultContent::Resource {
                                    uri,
                                    mime_type,
                                } => {
                                    println!(
                                        "[Resource: {} ({})]",
                                        uri,
                                        mime_type.unwrap_or_default()
                                    );
                                }
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Tool call failed: {}", e);
                        std::process::exit(1);
                    }
                }
            }

            #[cfg(not(feature = "daemon"))]
            {
                println!("Calling tool '{}'...", name);
                eprintln!("Direct tool calling requires daemon feature.");
                eprintln!("Rebuild with: cargo build --features daemon");
                eprintln!("Or use the MCP server via stdio: rk-core mcp serve");
                eprintln!("Tool: {}, Args: {}", name, args_json);
                std::process::exit(1);
            }
        }

        McpCommands::GetPrompt { name, args } => {
            println!("Getting prompt '{}' with args: {}", name, args);
            println!("(Prompt retrieval not yet implemented)");
        }

        McpCommands::Daemon { action } => {
            #[cfg(feature = "daemon")]
            {
                let manager = match DaemonManager::new() {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("Failed to initialize daemon manager: {}", e);
                        std::process::exit(1);
                    }
                };

                match action {
                    DaemonAction::Start => {
                        println!("Starting MCP daemon...");
                        if let Err(e) = manager.start().await {
                            eprintln!("Failed to start daemon: {}", e);
                            std::process::exit(1);
                        }
                        println!("MCP daemon started successfully.");
                    }
                    DaemonAction::Stop => {
                        println!("Stopping MCP daemon...");
                        if let Err(e) = manager.stop().await {
                            eprintln!("Failed to stop daemon: {}", e);
                            std::process::exit(1);
                        }
                        println!("MCP daemon stopped.");
                    }
                    DaemonAction::Status => {
                        let status = manager.status().await;
                        match status {
                            DaemonStatus::Running { pid, uptime_secs } => {
                                println!("Daemon: running (PID {}, uptime {}s)", pid, uptime_secs);
                            }
                            DaemonStatus::Stopped => {
                                println!("Daemon: not running");
                            }
                            DaemonStatus::Stale => {
                                println!("Daemon: stale (PID file exists but process is dead)");
                            }
                        }
                    }
                    DaemonAction::Restart => {
                        println!("Restarting MCP daemon...");
                        if let Err(e) = manager.restart().await {
                            eprintln!("Failed to restart daemon: {}", e);
                            std::process::exit(1);
                        }
                        println!("MCP daemon restarted.");
                    }
                }
            }

            #[cfg(not(feature = "daemon"))]
            {
                match action {
                    DaemonAction::Start | DaemonAction::Stop | DaemonAction::Restart => {
                        eprintln!("Daemon mode requires the daemon feature.");
                        eprintln!("Rebuild with: cargo build --features daemon");
                        std::process::exit(1);
                    }
                    DaemonAction::Status => {
                        println!("Daemon: feature not enabled");
                        eprintln!("Rebuild with: cargo build --features daemon");
                    }
                }
            }
        }

        #[cfg(feature = "daemon")]
        McpCommands::ServeDaemon { socket: _socket } => {
            // Run as daemon server (internal command)
            use reasonkit::mcp::daemon::{setup_signal_handlers, DaemonServer};

            println!("Starting MCP daemon server...");

            // Setup signal handlers (returns shutdown channel)
            let shutdown_tx = setup_signal_handlers()?;

            // Create and run server
            let server = DaemonServer::new(shutdown_tx).await?;
            server.run().await?;
        }
    }

    Ok(())
}
