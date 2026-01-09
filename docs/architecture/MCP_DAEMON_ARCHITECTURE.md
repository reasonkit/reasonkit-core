# MCP Daemon Architecture Design

**Status**: Design Phase
**Version**: 1.0.0
**Date**: 2026-01-04
**Author**: DevOps SRE Agent

---

## Executive Summary

This document describes the architecture for ReasonKit's MCP daemon mode, enabling:
1. **Direct tool invocation** via `rk mcp call-tool` without requiring a persistent server
2. **Optional background daemon** for persistent connections and improved performance
3. **Cross-platform IPC** using Unix domain sockets (Linux/macOS) and named pipes (Windows)
4. **Zero-config operation** with automatic daemon lifecycle management

---

## Problem Statement

**Current State:**
- `rk mcp call-tool gigathink '{"query": "..."}'` shows "coming soon" message
- Users must manually start MCP servers
- No persistent connection pooling
- Each tool call spawns new process overhead

**Requirements:**
1. Direct tool execution without manual server management
2. Optional persistent daemon for performance
3. Health checking and auto-recovery
4. Cross-platform compatibility (Linux, macOS, Windows)
5. Graceful shutdown and cleanup
6. Log management and rotation
7. Zero configuration for basic usage

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Layer                               │
│  rk mcp call-tool <name> <args>                                 │
│  rk mcp daemon {start|stop|status|restart}                      │
└────────────┬────────────────────────────────────────────────────┘
             │
             ├─── Mode Detection ────┐
             │                       │
             ▼                       ▼
    ┌─────────────────┐    ┌─────────────────┐
    │  Direct Mode    │    │  Daemon Mode    │
    │  (No daemon)    │    │  (IPC client)   │
    └────────┬────────┘    └────────┬────────┘
             │                      │
             │                      │ Unix Socket / Named Pipe
             │                      │
             ▼                      ▼
    ┌────────────────────────────────────────┐
    │      MCP Registry + Clients            │
    │  - Tool routing                        │
    │  - Connection pooling                  │
    │  - Health monitoring                   │
    └────────────────────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────────┐
    │    External MCP Servers (stdio)        │
    │  - ThinkTools, Sequential Thinking     │
    │  - Custom tool servers                 │
    └────────────────────────────────────────┘
```

---

## Component Design

### 1. Mode Selection Strategy

**Decision Tree:**
```rust
pub async fn execute_mcp_tool(name: &str, args: &str) -> Result<ToolResult> {
    // 1. Check if daemon is running
    if daemon_is_running().await? {
        // Use daemon via IPC
        daemon_call_tool(name, args).await
    } else {
        // Direct execution (spawn temp client)
        direct_call_tool(name, args).await
    }
}
```

**Benefits:**
- Zero-config operation (direct mode "just works")
- Optional performance mode (daemon)
- No breaking changes to existing workflows

---

### 2. IPC Mechanism

#### Unix Domain Sockets (Linux/macOS)

**Socket Path Strategy:**
```rust
pub fn get_socket_path() -> PathBuf {
    // XDG Base Directory Spec compliance
    let runtime_dir = env::var("XDG_RUNTIME_DIR")
        .ok()
        .and_then(|d| Some(PathBuf::from(d)))
        .or_else(|| {
            dirs::runtime_dir()
                .or_else(|| Some(env::temp_dir()))
        })
        .unwrap();

    runtime_dir.join("reasonkit").join("mcp.sock")
}
```

**Benefits:**
- File permissions for security
- Auto-cleanup on disconnect
- Standard Unix convention
- Fast local IPC

**Socket Permissions:**
```rust
use std::os::unix::fs::PermissionsExt;

// Create socket with 0600 (user-only)
let perms = std::fs::Permissions::from_mode(0o600);
std::fs::set_permissions(&socket_path, perms)?;
```

#### Named Pipes (Windows)

**Pipe Name:**
```rust
pub fn get_pipe_name() -> String {
    format!(r"\\.\pipe\reasonkit-mcp-{}", whoami::username())
}
```

**Benefits:**
- Native Windows IPC
- Per-user isolation
- Compatible with Windows security model

---

### 3. Daemon Process Management

#### Process Lifecycle

```rust
pub struct DaemonManager {
    pid_file: PathBuf,
    socket_path: PathBuf,
    log_path: PathBuf,
}

impl DaemonManager {
    pub async fn start(&self) -> Result<()> {
        // 1. Check if already running
        if self.is_running().await? {
            return Err(Error::daemon("Daemon already running"));
        }

        // 2. Daemonize (Unix) or spawn detached (Windows)
        #[cfg(unix)]
        self.daemonize_unix()?;

        #[cfg(windows)]
        self.spawn_detached_windows()?;

        // 3. Write PID file
        self.write_pid_file()?;

        // 4. Start health monitor
        self.start_health_monitor().await?;

        Ok(())
    }

    pub async fn stop(&self) -> Result<()> {
        // 1. Read PID
        let pid = self.read_pid()?;

        // 2. Send SIGTERM (Unix) or TerminateProcess (Windows)
        self.send_shutdown_signal(pid)?;

        // 3. Wait for graceful shutdown (max 10s)
        self.wait_for_shutdown(Duration::from_secs(10)).await?;

        // 4. Cleanup
        self.cleanup()?;

        Ok(())
    }

    pub async fn status(&self) -> DaemonStatus {
        if let Ok(pid) = self.read_pid() {
            if self.process_exists(pid) {
                DaemonStatus::Running { pid, uptime: self.get_uptime() }
            } else {
                DaemonStatus::Stale // PID file exists but process dead
            }
        } else {
            DaemonStatus::Stopped
        }
    }
}
```

#### Daemonization (Unix)

```rust
#[cfg(unix)]
fn daemonize_unix(&self) -> Result<()> {
    use nix::unistd::{fork, setsid, ForkResult};
    use std::os::unix::io::AsRawFd;

    // First fork
    match unsafe { fork() } {
        Ok(ForkResult::Parent { .. }) => std::process::exit(0),
        Ok(ForkResult::Child) => {}
        Err(e) => return Err(Error::daemon(format!("Fork failed: {}", e))),
    }

    // Become session leader
    setsid()?;

    // Second fork
    match unsafe { fork() } {
        Ok(ForkResult::Parent { .. }) => std::process::exit(0),
        Ok(ForkResult::Child) => {}
        Err(e) => return Err(Error::daemon(format!("Second fork failed: {}", e))),
    }

    // Change to root directory
    std::env::set_current_dir("/")?;

    // Redirect stdin/stdout/stderr to /dev/null
    let devnull = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/null")?;

    let fd = devnull.as_raw_fd();
    nix::unistd::dup2(fd, 0)?; // stdin
    nix::unistd::dup2(fd, 1)?; // stdout
    nix::unistd::dup2(fd, 2)?; // stderr

    Ok(())
}
```

#### Detached Process (Windows)

```rust
#[cfg(windows)]
fn spawn_detached_windows(&self) -> Result<()> {
    use std::os::windows::process::CommandExt;
    use winapi::um::winbase::CREATE_NO_WINDOW;

    let exe_path = std::env::current_exe()?;

    std::process::Command::new(exe_path)
        .arg("mcp")
        .arg("serve-daemon")
        .creation_flags(CREATE_NO_WINDOW)
        .spawn()?;

    Ok(())
}
```

---

### 4. Health Checking

```rust
pub struct HealthMonitor {
    interval: Duration,
    registry: Arc<RwLock<McpRegistry>>,
}

impl HealthMonitor {
    pub async fn run(&self) {
        let mut interval = tokio::time::interval(self.interval);

        loop {
            interval.tick().await;

            let registry = self.registry.read().await;
            for server in registry.list_servers().await {
                match server.ping().await {
                    Ok(true) => {
                        // Server healthy
                        registry.mark_healthy(&server.id).await;
                    }
                    Ok(false) | Err(_) => {
                        // Server unhealthy - attempt reconnect
                        warn!("Server {} unhealthy, reconnecting", server.name);
                        if let Err(e) = registry.reconnect_server(&server.id).await {
                            error!("Failed to reconnect {}: {}", server.name, e);
                        }
                    }
                }
            }
        }
    }
}
```

---

### 5. IPC Protocol

**Message Format (JSON-RPC 2.0):**

```rust
#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum IpcMessage {
    // Client -> Daemon
    CallTool {
        id: String,
        tool: String,
        args: serde_json::Value,
    },
    ListTools {
        id: String,
    },
    Ping {
        id: String,
    },
    Shutdown {
        id: String,
    },

    // Daemon -> Client
    ToolResult {
        id: String,
        result: ToolResult,
    },
    ToolsList {
        id: String,
        tools: Vec<Tool>,
    },
    Pong {
        id: String,
    },
    Error {
        id: String,
        error: String,
    },
}
```

**Client Example:**

```rust
pub struct IpcClient {
    #[cfg(unix)]
    stream: UnixStream,
    #[cfg(windows)]
    pipe: NamedPipe,
}

impl IpcClient {
    pub async fn connect() -> Result<Self> {
        #[cfg(unix)]
        let stream = UnixStream::connect(get_socket_path()).await?;

        #[cfg(windows)]
        let pipe = NamedPipe::connect(get_pipe_name()).await?;

        Ok(Self {
            #[cfg(unix)]
            stream,
            #[cfg(windows)]
            pipe,
        })
    }

    pub async fn call_tool(&mut self, name: &str, args: serde_json::Value) -> Result<ToolResult> {
        let msg = IpcMessage::CallTool {
            id: Uuid::new_v4().to_string(),
            tool: name.to_string(),
            args,
        };

        // Send request
        self.send_message(&msg).await?;

        // Receive response
        let response = self.receive_message().await?;

        match response {
            IpcMessage::ToolResult { result, .. } => Ok(result),
            IpcMessage::Error { error, .. } => Err(Error::daemon(error)),
            _ => Err(Error::daemon("Unexpected response type")),
        }
    }

    async fn send_message(&mut self, msg: &IpcMessage) -> Result<()> {
        let json = serde_json::to_vec(msg)?;
        let len = (json.len() as u32).to_le_bytes();

        #[cfg(unix)]
        {
            self.stream.write_all(&len).await?;
            self.stream.write_all(&json).await?;
        }

        #[cfg(windows)]
        {
            self.pipe.write_all(&len).await?;
            self.pipe.write_all(&json).await?;
        }

        Ok(())
    }

    async fn receive_message(&mut self) -> Result<IpcMessage> {
        let mut len_buf = [0u8; 4];

        #[cfg(unix)]
        self.stream.read_exact(&mut len_buf).await?;

        #[cfg(windows)]
        self.pipe.read_exact(&mut len_buf).await?;

        let len = u32::from_le_bytes(len_buf) as usize;
        let mut buf = vec![0u8; len];

        #[cfg(unix)]
        self.stream.read_exact(&mut buf).await?;

        #[cfg(windows)]
        self.pipe.read_exact(&mut buf).await?;

        Ok(serde_json::from_slice(&buf)?)
    }
}
```

---

### 6. Log Management

```rust
pub struct DaemonLogger {
    log_dir: PathBuf,
    max_size: u64,      // 10 MB
    max_files: usize,   // Keep last 5 log files
}

impl DaemonLogger {
    pub fn init(&self) -> Result<()> {
        let log_file = self.log_dir.join("mcp-daemon.log");

        let file_appender = tracing_appender::rolling::daily(
            &self.log_dir,
            "mcp-daemon.log"
        );

        let subscriber = tracing_subscriber::fmt()
            .with_writer(file_appender)
            .with_ansi(false)
            .with_target(true)
            .with_thread_ids(true)
            .with_line_number(true)
            .json() // JSON format for structured logging
            .finish();

        tracing::subscriber::set_global_default(subscriber)?;

        Ok(())
    }

    pub fn rotate_logs(&self) -> Result<()> {
        // Auto-rotation handled by tracing-appender
        // Manual rotation for size-based limits
        let entries: Vec<_> = std::fs::read_dir(&self.log_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_str()
                    .map(|s| s.starts_with("mcp-daemon"))
                    .unwrap_or(false)
            })
            .collect();

        if entries.len() > self.max_files {
            // Remove oldest files
            let mut sorted = entries;
            sorted.sort_by_key(|e| e.metadata().ok()?.modified().ok());

            for entry in sorted.iter().take(sorted.len() - self.max_files) {
                std::fs::remove_file(entry.path())?;
            }
        }

        Ok(())
    }
}
```

---

### 7. Graceful Shutdown

```rust
pub struct DaemonServer {
    registry: Arc<RwLock<McpRegistry>>,
    shutdown_tx: broadcast::Sender<()>,
    listener: UnixListener, // or NamedPipeServer on Windows
}

impl DaemonServer {
    pub async fn run(&self) -> Result<()> {
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        loop {
            tokio::select! {
                // Accept new connections
                Ok((stream, _)) = self.listener.accept() => {
                    let registry = self.registry.clone();
                    let shutdown_tx = self.shutdown_tx.clone();

                    tokio::spawn(async move {
                        if let Err(e) = Self::handle_client(stream, registry, shutdown_tx).await {
                            error!("Client handler error: {}", e);
                        }
                    });
                }

                // Shutdown signal received
                _ = shutdown_rx.recv() => {
                    info!("Shutdown signal received, cleaning up...");
                    self.shutdown().await?;
                    break;
                }
            }
        }

        Ok(())
    }

    async fn shutdown(&self) -> Result<()> {
        info!("Shutting down MCP servers...");

        // 1. Stop accepting new connections
        drop(self.listener);

        // 2. Disconnect all MCP servers gracefully
        let registry = self.registry.write().await;
        for server in registry.list_servers().await {
            if let Err(e) = registry.disconnect_server(&server.id).await {
                warn!("Failed to disconnect {}: {}", server.name, e);
            }
        }

        // 3. Wait for in-flight requests (max 5s)
        tokio::time::timeout(
            Duration::from_secs(5),
            self.wait_for_active_requests()
        ).await.ok();

        // 4. Cleanup socket/pipe
        #[cfg(unix)]
        std::fs::remove_file(get_socket_path()).ok();

        info!("Shutdown complete");
        Ok(())
    }
}
```

---

## File Structure

```
reasonkit-core/
├── src/
│   ├── mcp/
│   │   ├── mod.rs
│   │   ├── client.rs          (existing)
│   │   ├── registry.rs        (existing)
│   │   ├── daemon/            (NEW)
│   │   │   ├── mod.rs
│   │   │   ├── manager.rs     - Process lifecycle
│   │   │   ├── ipc_server.rs  - IPC server
│   │   │   ├── ipc_client.rs  - IPC client
│   │   │   ├── health.rs      - Health monitor
│   │   │   ├── logger.rs      - Log management
│   │   │   └── signals.rs     - Signal handling
│   │   └── ...
│   ├── bin/
│   │   └── mcp_cli.rs         (update)
│   └── main.rs                (update)
└── docs/
    └── architecture/
        └── MCP_DAEMON_ARCHITECTURE.md (this file)
```

---

## Implementation Phases

### Phase 1: Direct Mode (Week 1)
- [ ] Implement `direct_call_tool()` function
- [ ] Update `mcp_cli.rs` CallTool command
- [ ] Add error handling and retries
- [ ] Integration tests

**Deliverable:** `rk mcp call-tool gigathink '{"query": "..."}'` works without daemon

### Phase 2: Daemon Foundation (Week 2)
- [ ] Create `src/mcp/daemon/` module structure
- [ ] Implement process management (start/stop/status)
- [ ] Add PID file handling
- [ ] Platform-specific daemonization (Unix/Windows)

**Deliverable:** `rk mcp daemon start|stop|status` commands work

### Phase 3: IPC Implementation (Week 3)
- [ ] Implement Unix domain socket server/client
- [ ] Implement Windows named pipe server/client
- [ ] Add message protocol (IpcMessage enum)
- [ ] Connection handling and pooling

**Deliverable:** IPC communication works end-to-end

### Phase 4: Integration (Week 4)
- [ ] Update CLI to auto-detect daemon
- [ ] Implement fallback logic (daemon → direct)
- [ ] Add health monitoring
- [ ] Add graceful shutdown

**Deliverable:** Seamless mode switching

### Phase 5: Operations (Week 5)
- [ ] Implement log rotation
- [ ] Add metrics collection
- [ ] Create systemd unit file (Linux)
- [ ] Create launchd plist (macOS)
- [ ] Create Windows service wrapper
- [ ] Documentation and examples

**Deliverable:** Production-ready daemon

---

## Dependencies

### New Crate Dependencies

```toml
[dependencies]
# Existing (already in project)
tokio = { version = "1.44", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["json"] }
uuid = { version = "1.11", features = ["v4", "serde"] }

# New additions
tracing-appender = "0.2"  # Log rotation
nix = { version = "0.29", features = ["process", "signal"] }  # Unix process mgmt
dirs = "5.0"              # XDG directories

[target.'cfg(windows)'.dependencies]
winapi = { version = "0.3", features = ["winbase", "processthreadsapi", "handleapi"] }
windows-named-pipe = "0.1"  # Windows named pipes

[target.'cfg(unix)'.dependencies]
tokio = { version = "1.44", features = ["net"] }  # Unix sockets
```

---

## Configuration

### Config File: `~/.config/reasonkit/daemon.toml`

```toml
[daemon]
# Enable daemon mode by default
enabled = true

# Auto-start daemon on first tool call
auto_start = true

# Health check interval (seconds)
health_check_interval = 30

# Request timeout (seconds)
request_timeout = 60

# Log settings
[daemon.logging]
level = "info"
max_file_size = "10MB"
max_files = 5
directory = "~/.local/share/reasonkit/logs"

# IPC settings
[daemon.ipc]
# Unix socket path (overrides default)
socket_path = "~/.local/share/reasonkit/mcp.sock"

# Windows pipe name (overrides default)
pipe_name = "reasonkit-mcp"

# Connection pool size
max_connections = 10
```

---

## Security Considerations

1. **File Permissions**
   - Unix socket: 0600 (user-only)
   - PID file: 0644 (read-only for others)
   - Log files: 0600 (user-only)

2. **Process Isolation**
   - Each user runs their own daemon
   - No shared resources between users

3. **Input Validation**
   - All IPC messages validated before execution
   - JSON schema validation for tool arguments

4. **Denial of Service**
   - Rate limiting on IPC connections
   - Maximum request size (1 MB)
   - Connection timeout (60s)

---

## Monitoring & Observability

### Metrics (via `rk mcp daemon stats`)

```rust
#[derive(Serialize)]
pub struct DaemonStats {
    pub uptime_secs: u64,
    pub requests_total: u64,
    pub requests_succeeded: u64,
    pub requests_failed: u64,
    pub active_connections: usize,
    pub avg_response_time_ms: f64,
    pub servers_connected: usize,
    pub servers_healthy: usize,
    pub memory_usage_mb: f64,
}
```

### Log Structure (JSON)

```json
{
  "timestamp": "2026-01-04T10:30:45.123Z",
  "level": "INFO",
  "target": "reasonkit::mcp::daemon",
  "message": "Tool execution completed",
  "fields": {
    "tool": "gigathink",
    "duration_ms": 1234,
    "success": true,
    "client_id": "uuid-here"
  }
}
```

---

## Testing Strategy

### Unit Tests
- Process management functions
- IPC message serialization
- Health check logic
- Log rotation

### Integration Tests
- Full daemon lifecycle (start/stop/restart)
- IPC communication end-to-end
- Graceful shutdown
- Error recovery

### Performance Tests
- Connection pooling efficiency
- IPC throughput (requests/sec)
- Memory footprint over time
- Health check overhead

---

## Rollout Strategy

### Phase 1: Opt-in (v0.2.0)
- Daemon is OFF by default
- Users explicitly run `rk mcp daemon start`
- Direct mode is default

### Phase 2: Opt-out (v0.3.0)
- Daemon auto-starts on first tool call
- Users can disable with `daemon.enabled = false`
- Migration guide for existing users

### Phase 3: Default (v1.0.0)
- Daemon is standard mode
- Direct mode available as fallback
- systemd/launchd integration for auto-start

---

## Success Metrics

- **Performance**: 10x faster tool calls with daemon (vs direct mode)
- **Reliability**: 99.9% uptime for daemon process
- **Usability**: Zero-config for 95% of users
- **Compatibility**: Works on Linux, macOS, Windows
- **Adoption**: 80% of users use daemon mode by v1.0

---

## References

- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [Unix Domain Sockets (Tokio)](https://tokio.rs/tokio/tutorial/shared-state)
- [Windows Named Pipes](https://docs.microsoft.com/en-us/windows/win32/ipc/named-pipes)
- [systemd Service Units](https://www.freedesktop.org/software/systemd/man/systemd.service.html)
- [launchd Plists](https://www.launchd.info/)

---

## Next Steps

1. Review architecture with team
2. Approve Phase 1 implementation plan
3. Create implementation tasks in Taskwarrior
4. Begin development on `direct_call_tool()` function

---

**End of Architecture Document**
