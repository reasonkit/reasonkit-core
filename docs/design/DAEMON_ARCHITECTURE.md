# ReasonKit Daemon (`rkd`) Architecture
> **Status:** DRAFT
> **Target:** v1.1.0

## 1. Executive Summary

The ReasonKit Daemon (`rkd`) is a background process responsible for:
1.  **Filesystem Watching:** Detecting changes in project directories to trigger re-indexing or protocol reloading.
2.  **MCP Server Management:** Managing the lifecycle of local MCP servers.
3.  **State Persistence:** Maintaining long-running state (indexes, traces) independent of CLI invocations.
4.  **IPC:** Providing a stable interface for IDE extensions and the CLI.

## 2. Core Components

### 2.1 Process Architecture

```
┌──────────────┐      IPC       ┌──────────────┐
│   CLI / IDE  │ <───────────>  │     rkd      │
└──────────────┘                └──────┬───────┘
                                       │
                         ┌─────────────┼──────────────┐
                         ▼             ▼              ▼
                   ┌──────────┐  ┌──────────┐  ┌──────────┐
                   │ Watcher  │  │ MCP Mgr  │  │ Indexer  │
                   └──────────┘  └──────────┘  └──────────┘
```

### 2.2 Filesystem Watcher (`notify`)

We use the `notify` crate (debounced) to monitor:
*   `protocols/*.toml` -> Hot-reload protocols.
*   `docs/**/*.md` -> Trigger incremental RAG indexing.
*   `config.toml` -> Reload configuration.

**Configuration:**
```toml
[daemon.watch]
paths = ["./docs", "./protocols"]
debounce_ms = 500
ignore = ["**/node_modules", "**/target"]
```

### 2.3 MCP Server Manager

The daemon acts as the host for local MCP servers.
*   Starts servers on demand or at startup.
*   Monitors health (heartbeats).
*   Restarts crashed servers.
*   Proxies JSON-RPC messages between clients and servers.

### 2.4 IPC Interface

We will use **JSON-RPC over Domain Sockets** (Unix) or **Named Pipes** (Windows).
*   Socket path: `~/.local/share/reasonkit/rkd.sock`
*   Protocol: Standard JSON-RPC 2.0.

**Core Methods:**
*   `daemon.status()`
*   `daemon.reload()`
*   `mcp.list_servers()`
*   `mcp.start_server(name)`
*   `rag.index_status()`

## 3. Implementation Plan

### Phase 1: Skeleton
*   Basic `clap` subcommand `rk-core daemon`.
*   Pidfile management (ensure single instance).
*   Signal handling (SIGINT/SIGTERM).

### Phase 2: Watcher
*   Integrate `notify` crate.
*   Event loop to handle file change events.
*   Debouncing logic.

### Phase 3: IPC
*   Implement Unix Domain Socket server.
*   Define `DaemonRequest` and `DaemonResponse` types.

### Phase 4: Integration
*   Connect Watcher events to Indexer actions.
*   Connect CLI `rk-core status` to Daemon IPC.

## 4. Rust Crate Dependencies

*   `notify = "6.1"` (Watching)
*   `tokio = { version = "1", features = ["full"] }` (Async runtime)
*   `interprocess = "1.2"` (Cross-platform IPC)
*   `daemonize = "0.5"` (Backgrounding - Linux/macOS)
*   `windows-service = "0.6"` (Windows Service - optional)

## 5. Security Considerations

*   **Socket Permissions:** The IPC socket must be accessible ONLY by the user (mode `600`).
*   **Resource Limits:** Daemon should respect cgroups/ulimits to prevent resource exhaustion.
