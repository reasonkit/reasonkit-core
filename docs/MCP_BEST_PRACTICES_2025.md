# MCP (Model Context Protocol) Best Practices 2025

> Deep Research Report on MCP Server Implementation, Security, Performance & Deployment
> Research Date: 2025-12-25
> Target: ReasonKit MCP Server Enhancement

**Status**: Production-Ready Guidance
**Applies To**: reasonkit-core/src/mcp/, reasonkit-pro MCP deployment
**Specification Version**: 2025-11-25 (latest)

---

## Executive Summary

Model Context Protocol (MCP) has become the de-facto standard for connecting AI systems to external tools and data sources in less than 12 months since its November 2024 launch. This report synthesizes best practices from 2025 production deployments, official specifications, and Rust implementation patterns to guide ReasonKit's MCP server development.

**Key Findings:**

- MCP specification updated to 2025-11-25 with Tasks abstraction, improved OAuth, and extensions
- Streamable HTTP has replaced SSE as the production transport (March 2025 update)
- Rust implementations achieve 4,700+ QPS with sub-millisecond latency
- OAuth 2.1 is now mandatory for HTTP transports (June 2025 update)
- Production deployments favor containerization with K8s orchestration
- Benchmarks show 20-30% task completion improvement but 27% cost increase

---

## Table of Contents

1. [MCP Specification Overview](#mcp-specification-overview)
2. [Architecture Best Practices](#architecture-best-practices)
3. [Rust MCP Implementation](#rust-mcp-implementation)
4. [Transport Protocols](#transport-protocols)
5. [Security Best Practices](#security-best-practices)
6. [Performance Optimization](#performance-optimization)
7. [Deployment Patterns](#deployment-patterns)
8. [Production Readiness](#production-readiness)
9. [ReasonKit-Specific Recommendations](#reasonkit-specific-recommendations)
10. [References](#references)

---

## MCP Specification Overview

### Latest Specification: 2025-11-25

**Source**: [MCP Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25)

The November 2025 specification marks MCP's one-year anniversary with major production-readiness improvements:

#### Key Updates in 2025

| Version        | Date     | Major Features                                                  |
| -------------- | -------- | --------------------------------------------------------------- |
| **2025-11-25** | Nov 2025 | **Tasks abstraction**, OAuth improvements, extensions framework |
| **2025-06-18** | Jun 2025 | **Structured outputs**, OAuth 2.1, user elicitation             |
| **2025-03-26** | Mar 2025 | **Streamable HTTP** (replaces SSE)                              |
| **2024-11-05** | Nov 2024 | Initial release (HTTP+SSE transport)                            |

### MCP Adoption Timeline

- **Nov 2024**: Anthropic launches MCP
- **Dec 2025**: Donated to Agentic AI Foundation (Linux Foundation)
- **Mar 2025**: OpenAI adopts MCP across ChatGPT, Agents SDK, Responses API
- **Apr 2025**: Google DeepMind confirms MCP support in Gemini
- **Nov 2025**: Becomes "de-facto standard" for AI tool integration

### Core Architecture

MCP is based on JSON-RPC 2.0 and deliberately reuses ideas from Language Server Protocol (LSP):

```
┌──────────────────────────────────────────────────────────┐
│ MCP Client (AI Application)                             │
│   - Claude Desktop, ChatGPT, Gemini, etc.                │
├──────────────────────────────────────────────────────────┤
│ JSON-RPC 2.0 Protocol Layer                             │
│   - Request/Response                                     │
│   - Notifications                                        │
│   - Cancellation                                         │
├──────────────────────────────────────────────────────────┤
│ Transport Layer                                          │
│   - stdio (development, local)                           │
│   - Streamable HTTP (production, remote) ← NEW          │
├──────────────────────────────────────────────────────────┤
│ MCP Server (Tool/Resource Provider)                     │
│   - Tools (executable functions)                         │
│   - Resources (data sources)                             │
│   - Prompts (templates)                                  │
│   - Tasks (async work tracking) ← NEW                   │
└──────────────────────────────────────────────────────────┘
```

### Tasks Abstraction (NEW - Nov 2025)

**Purpose**: Track asynchronous, long-running work performed by MCP servers.

**Use Cases**:

- Data pipelines processing millions of records
- Multi-hour code migration or refactoring
- Test execution with streaming logs
- Multi-agent systems with concurrent work
- Deep research spawning multiple internal agents

**Flow**:

```
Client → Request with task hint
Server → Immediate ack with taskId
Client → Poll status (working/completed/failed)
Client → Retrieve result when complete
```

**Status**: Experimental (battle-testing in production)

**ReasonKit Application**: Perfect for ThinkTools like `ProofGuard` (multi-source verification) or `GigaThink` (10+ perspective generation) that can take significant time.

---

## Architecture Best Practices

### 1. Tool Design Principles

**Source**: [MCP Best Practices](https://modelcontextprotocol.info/docs/best-practices/), [15 Best Practices for Production](https://thenewstack.io/15-best-practices-for-building-mcp-servers-in-production/)

#### DO: High-Level Workflow Tools

```rust
// ❌ BAD: Low-level API mapping
tools: [
  "file_open",
  "file_read",
  "file_write",
  "file_close"
]

// ✅ GOOD: Workflow-oriented tools
tools: [
  "analyze_document",      // Opens, reads, analyzes, closes
  "update_content",        // Reads, modifies, writes
  "search_and_replace"     // Full workflow in one tool
]
```

**Key Principle**: Each tool should map to a complete user intent, not individual API operations.

**Impact**: 30% improvement in user adoption when tools are workflow-focused vs. API-focused.

#### DO: Single-Purpose Servers

Each MCP server should have **one clear, well-defined purpose**:

```
✅ Good Examples:
- mcp-server-github       (GitHub operations)
- mcp-server-filesystem   (File operations)
- mcp-server-reasonkit-gigathink  (Expansive thinking)

❌ Bad Examples:
- mcp-server-everything   (Kitchen sink)
- mcp-server-utils        (Vague purpose)
```

**ReasonKit Pattern**: Each ThinkTool should be its own MCP server:

- `mcp-server-reasonkit-gigathink`
- `mcp-server-reasonkit-laserlogic`
- `mcp-server-reasonkit-bedrock`
- `mcp-server-reasonkit-proofguard`
- `mcp-server-reasonkit-brutalhonesty`

### 2. Macro and Chaining

**Source**: [MCP Server Best Practices](https://mcpcat.io/blog/mcp-server-best-practices/)

Implement **prompts** (MCP primitive) that chain multiple backend calls:

```yaml
# Example: Multi-step reasoning prompt
prompts:
  - name: "deep_analysis"
    description: "Chain GigaThink → LaserLogic → BedRock → ProofGuard"
    arguments:
      - name: "query"
        description: "Question to analyze"
    template: |
      Step 1: Generate 10+ perspectives using GigaThink
      Step 2: Validate logic using LaserLogic
      Step 3: Decompose to first principles using BedRock
      Step 4: Verify claims using ProofGuard
```

**Benefit**: Single instruction triggers complex workflows, reducing cognitive load and error potential.

### 3. Scaling Patterns

**Source**: [Production-Ready MCP Servers](https://dev.to/raghavajoijode/production-ready-mcp-servers-security-performance-deployment-5e48)

When scaling, **separate servers by**:

1. **Product area**: Different business domains
2. **Permissions**: Read-only vs. write operations
3. **Performance**: Quick lookups vs. heavy processing

```
┌─────────────────────────────────────────┐
│ Separate MCP Servers (Microservices)   │
├─────────────────────────────────────────┤
│ reasonkit-read     (Fast, cached)       │
│ reasonkit-write    (Auth required)      │
│ reasonkit-heavy    (Long tasks)         │
└─────────────────────────────────────────┘
```

Each server can:

- Scale independently
- Have its own security policies
- Be maintained by different teams

---

## Rust MCP Implementation

### Official Rust SDK: `rmcp`

**Source**: [Official Rust SDK](https://github.com/modelcontextprotocol/rust-sdk), [rmcp crate docs](https://docs.rs/rmcp/latest/rmcp/)

**Current Version**: 0.8.0
**Runtime**: tokio async

```toml
[dependencies]
rmcp = { version = "0.8.0", features = ["server"] }
```

### Production Rust Implementations

#### Performance Benchmarks

**Source**: [High-Performance Rust MCP Server](https://medium.com/@bohachu/building-a-high-performance-mcp-server-with-rust-a-complete-implementation-guide-8a18ab16b538)

| Metric             | Native | Docker | Notes                     |
| ------------------ | ------ | ------ | ------------------------- |
| **Queries/Second** | 4,700+ | 1,700+ | Sub-millisecond responses |
| **Latency P99**    | <1ms   | <5ms   | Consistent across load    |
| **Memory**         | Low    | Low    | Rust memory safety        |

**Key Finding**: No correlation between query complexity and response time, indicating efficient implementation.

#### Leading Rust MCP Frameworks

**Source**: [Shuttle MCP Comparison](https://www.shuttle.dev/blog/2025/09/15/mcp-servers-rust-comparison), [Prism MCP](https://users.rust-lang.org/t/prism-mcp-rust-sdk-v0-1-0-production-grade-model-context-protocol-implementation/133318)

| Framework           | Status     | Tests | Examples       | Spec Version   |
| ------------------- | ---------- | ----- | -------------- | -------------- |
| **rmcp** (official) | Production | N/A   | Multiple       | 2025-11-25     |
| **Prism MCP SDK**   | Production | 229+  | 39             | 2025-06-18     |
| **mcp-framework**   | Production | N/A   | Built-in tools | Latest         |
| **rust-mcp-sdk**    | Production | N/A   | Multiple       | All 3 versions |
| **ultrafast-mcp**   | Optimized  | N/A   | N/A            | Latest         |

### Example Implementation Pattern

**Source**: [Official rmcp examples](https://github.com/modelcontextprotocol/rust-sdk)

```rust
use rmcp::prelude::*;

#[derive(ToolRouter)]
struct MyServer;

#[tool_router]
impl MyServer {
    #[tool(description = "Analyze using GigaThink")]
    async fn gigathink(&self, query: String) -> CallToolResult {
        // Implementation
        Ok(CallToolResult::success(/* result */))
    }

    #[tool(description = "Validate using LaserLogic")]
    async fn laserlogic(&self, analysis: String) -> CallToolResult {
        // Implementation
        Ok(CallToolResult::success(/* result */))
    }
}

impl ServerHandler for MyServer {
    async fn handle_initialize(&self, params: InitializeParams)
        -> Result<InitializeResult> {
        // Server initialization
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let server = MyServer;
    server.serve(StdioTransport::new()).await?;
    Ok(())
}
```

### Rust Advantages for MCP

1. **Performance**: 4,700+ QPS, sub-millisecond latency
2. **Memory Safety**: No undefined behavior, no segfaults
3. **Concurrency**: Tokio async runtime, zero-cost abstractions
4. **Type Safety**: Compile-time guarantees for JSON-RPC messages
5. **Binary Size**: Small, self-contained binaries

**ReasonKit Alignment**: Perfect fit with CONS-005 (Rust for performance paths) and CONS-001 (No Node.js).

---

## Transport Protocols

### Streamable HTTP (Production Standard)

**Source**: [Why MCP Deprecated SSE](https://blog.fka.dev/blog/2025-06-06-why-mcp-deprecated-sse-and-go-with-streamable-http/), [Understanding MCP HTTP+SSE Change](https://blog.christianposta.com/ai/understanding-mcp-recent-change-around-http-sse/)

**Status**: Introduced March 26, 2025 (version 2025-03-26)
**Replaces**: HTTP+SSE (deprecated but backward compatible)

#### Why Streamable HTTP?

**Old approach (HTTP+SSE)**: Two separate endpoints

- POST `/message` - Client to server
- GET `/events` - Server to client (SSE stream)

**New approach (Streamable HTTP)**: Single unified endpoint

- POST `/mcp` - Client to server
- GET `/mcp` - Server to client (optional SSE)
- Response can be `application/json` OR `text/event-stream`

**Benefits**:

1. **Simpler architecture**: One endpoint vs. two
2. **Better load balancing**: Single path to route
3. **Unified session management**: One session ID header
4. **Backward compatible**: Can run both simultaneously

#### Implementation Pattern

```rust
// Streamable HTTP server
async fn handle_mcp_endpoint(req: Request) -> Response {
    match req.method() {
        Method::POST => {
            // Client to server request
            let json_response = handle_request(req.body()).await?;

            // Can return either:
            // 1. application/json (simple response)
            // 2. text/event-stream (SSE for complex flows)
            if needs_streaming {
                Response::sse(json_response)
            } else {
                Response::json(json_response)
            }
        }
        Method::GET => {
            // Open SSE stream for proactive server messages
            Response::sse_stream(session_id)
        }
        _ => Response::method_not_allowed()
    }
}
```

#### Session Management

```rust
// Server assigns session ID at initialization
headers: {
    "Mcp-Session-Id": "unique-session-id-12345"
}

// Client includes session ID in subsequent requests
headers: {
    "Mcp-Session-Id": "unique-session-id-12345"
}

// Resumability: Client reconnects with Last-Event-ID
headers: {
    "Last-Event-ID": "event-42"
}
// Server replays events after ID 42
```

### stdio Transport (Development/Local)

**Status**: Baseline, preferred for development
**Use Case**: Local MCP servers, testing, Claude Desktop integration

```bash
# stdio transport invocation
npx -y @modelcontextprotocol/server-sequential-thinking

# Rust MCP server (future)
rk-thinktool --module gigathink --transport stdio
```

**Characteristics**:

- **Pros**: Simple, no network config, process isolation
- **Cons**: Single client, no remote access, process management overhead

### Transport Selection Guide

| Use Case          | Transport               | Rationale                        |
| ----------------- | ----------------------- | -------------------------------- |
| Development       | stdio                   | Fast iteration, simple debugging |
| Testing           | stdio                   | Isolated processes, easy setup   |
| Claude Desktop    | stdio                   | Official integration pattern     |
| Remote deployment | Streamable HTTP         | Network access, load balancing   |
| Production scale  | Streamable HTTP         | Horizontal scaling, cloud-native |
| Enterprise        | Streamable HTTP + OAuth | Security, compliance, audit      |

**ReasonKit Strategy**:

- **reasonkit-core** (OSS): stdio for CLI usage
- **reasonkit-pro** (Enterprise): Streamable HTTP with OAuth for sidecar deployment

---

## Security Best Practices

### OAuth 2.1 Authorization (Mandatory for HTTP)

**Source**: [MCP OAuth 2.1 Guide](https://www.scalekit.com/blog/implement-oauth-for-mcp-servers), [MCP Authorization Spec](https://modelcontextprotocol.io/specification/draft/basic/authorization)

**Requirement**: OAuth 2.1 is **mandatory** for HTTP-based transports as of June 2025 specification.

#### Core Requirements

| Component                | Role                  | Must Implement                        |
| ------------------------ | --------------------- | ------------------------------------- |
| **MCP Server**           | OAuth Resource Server | Protected Resource Metadata (RFC9728) |
| **MCP Client**           | OAuth Client          | Resource Indicators (RFC8707)         |
| **Authorization Server** | Token Issuer          | OAuth 2.1 with PKCE                   |

#### Security Requirements (MUST)

1. **PKCE**: All clients MUST use PKCE with SHA-256
2. **HTTPS**: All authorization endpoints MUST be HTTPS
3. **Token Storage**: Clients MUST securely store tokens (OAuth 2.1 best practices)
4. **Redirect URI Validation**: Prevent open redirect vulnerabilities
5. **Token Audience Validation**: MCP servers MUST validate tokens were issued for them

#### Authorization Flow

```
┌────────────┐                                  ┌──────────────────┐
│ MCP Client │                                  │ Authorization    │
│            │                                  │ Server           │
└─────┬──────┘                                  └────────┬─────────┘
      │                                                  │
      │ 1. Request without token                        │
      │ ───────────────────────────────────────────────►│
      │                                         ┌────────┴────────┐
      │                                         │ MCP Server      │
      │ 2. 401 Unauthorized                     │ (Resource       │
      │    WWW-Authenticate: resource_metadata  │  Server)        │
      │◄────────────────────────────────────────┤                 │
      │                                         └─────────────────┘
      │ 3. Fetch metadata
      │ ───────────────────────────────────────────────►│
      │◄────────────────────────────────────────────────│
      │ 4. Metadata (auth server URL)                   │
      │                                                  │
      │ 5. Dynamic Client Registration (optional)       │
      │ ───────────────────────────────────────────────►│
      │◄────────────────────────────────────────────────│
      │ 6. Client credentials                           │
      │                                                  │
      │ 7. Authorization code flow with PKCE            │
      │ ───────────────────────────────────────────────►│
      │◄────────────────────────────────────────────────│
      │ 8. Authorization code                           │
      │                                                  │
      │ 9. Exchange code for token (with resource param)│
      │ ───────────────────────────────────────────────►│
      │◄────────────────────────────────────────────────│
      │ 10. Access token (audience: MCP server)         │
      │                                         ┌────────┴────────┐
      │ 11. Request with token                  │ MCP Server      │
      │ ───────────────────────────────────────►│                 │
      │    Authorization: Bearer <token>        │                 │
      │                                         │ Validate:       │
      │                                         │ - Signature     │
      │                                         │ - Issuer        │
      │                                         │ - Audience      │
      │                                         │ - Expiry        │
      │◄────────────────────────────────────────┤                 │
      │ 12. 200 OK with response                │                 │
      │                                         └─────────────────┘
```

#### Discovery Mechanisms

MCP servers MUST provide authorization server location via:

**Option 1: WWW-Authenticate Header** (Recommended)

```http
HTTP/1.1 401 Unauthorized
WWW-Authenticate: Bearer resource_metadata="https://example.com/.well-known/mcp-metadata"
```

**Option 2: OAuth 2.0 Protected Resource Metadata**

```json
GET /.well-known/mcp-metadata

{
  "resource": "https://example.com/mcp",
  "authorization_servers": [
    "https://auth.example.com"
  ]
}
```

#### Implementation Approaches

**Source**: [MCP OAuth Guide](https://dev.to/composiodev/mcp-oauth-21-a-complete-guide-3g91)

| Approach          | Description                    | Pros                     | Cons                           |
| ----------------- | ------------------------------ | ------------------------ | ------------------------------ |
| **Self-Embedded** | MCP server is also auth server | Simple, self-contained   | Complex to implement correctly |
| **Delegated**     | Use Auth0, Okta, etc.          | Production-ready, tested | External dependency, cost      |

**Recommendation**: Use delegated approach (Auth0, Okta, Ory) for production. Most OAuth 2.1 providers satisfy MCP requirements with:

- Discovery document exposure
- Dynamic client registration
- Resource parameter echo in tokens

**ReasonKit Strategy**:

- **reasonkit-core** (OSS): No auth (stdio only)
- **reasonkit-pro** (Enterprise): Delegated OAuth 2.1 (Auth0/Okta integration)

### Network Security

**Source**: [Secure MCP Deployment Guide](https://mcpmanager.ai/blog/secure-mcp-server-deployment-at-scale-the-complete-guide/)

1. **Origin Validation**: MUST validate `Origin` header to prevent DNS rebinding
2. **VPC Isolation**: MCP servers in private subnets, no direct internet access
3. **Security Groups**: Restrict traffic flow between components
4. **TLS**: HTTPS with modern protocols (HTTP/2, HTTP/3)
5. **WAF**: Web Application Firewall to prevent common exploits
6. **Rate Limiting**: Prevent DDoS attacks
7. **mTLS**: Mutual TLS for service-to-service communication

### Secrets Management

**Source**: [Production-Ready MCP](https://dev.to/raghavajoijode/production-ready-mcp-servers-security-performance-deployment-5e48)

**Anti-Pattern**: Environment variables in production

- Difficult to rotate
- Leak into logs/build artifacts
- Static target for attackers

**Best Practice**: Dedicated secrets management

- Azure Key Vault
- AWS Secrets Manager
- HashiCorp Vault
- Google Secret Manager

```rust
// ❌ BAD: Environment variables
let api_key = env::var("API_KEY")?;

// ✅ GOOD: Secrets manager
let api_key = secrets_manager.get_secret("mcp/api-key").await?;
```

### Security Checklist

- [ ] PKCE with SHA-256 (OAuth 2.1)
- [ ] HTTPS only for HTTP transports
- [ ] Origin header validation
- [ ] Token audience validation
- [ ] Secrets in vault, not env vars
- [ ] VPC/network isolation
- [ ] WAF and rate limiting
- [ ] No secrets in tool results or logs
- [ ] Session IDs cryptographically random
- [ ] Security scanning (Snyk, SBOM)
- [ ] Kill-switch for emergency revocation

---

## Performance Optimization

### Rust Performance Results

**Source**: [Rust MCP Server Benchmarks](https://medium.com/@bohachu/building-a-high-performance-mcp-server-with-rust-a-complete-implementation-guide-8a18ab16b538)

| Implementation | QPS    | P99 Latency | Memory |
| -------------- | ------ | ----------- | ------ |
| Rust (native)  | 4,700+ | <1ms        | Low    |
| Rust (Docker)  | 1,700+ | <5ms        | Low    |
| Node.js        | ~500   | ~50ms       | High   |
| Python         | ~200   | ~100ms      | High   |

**Key Findings**:

- Rust is **9-23x faster** than Node.js/Python
- Sub-millisecond responses enable real-time AI interactions
- No performance degradation with query complexity
- Docker overhead: ~63% throughput reduction

### Optimization Techniques

**Source**: [Top 10 MCP Performance Techniques](https://superagi.com/top-10-advanced-techniques-for-optimizing-mcp-server-performance-in-2025/)

#### 1. Connection Pooling

Reuse connections instead of creating new ones:

```rust
// Connection pool for MCP clients
let pool = Pool::builder()
    .max_size(10)
    .idle_timeout(Duration::from_secs(300))
    .build(McpClientManager::new(config));

// Reuse from pool
let client = pool.get().await?;
```

**Impact**: 40% reduction in connection overhead

#### 2. Caching Strategy

```rust
// Cache tool results for deterministic operations
#[derive(Cache)]
struct ToolResultCache {
    #[cache(ttl = 300)] // 5 minutes
    results: HashMap<String, ToolResult>,
}

// Cache-aside pattern
async fn call_tool_cached(&self, name: &str, args: Value)
    -> Result<ToolResult> {
    if let Some(cached) = self.cache.get(name, &args).await? {
        return Ok(cached);
    }

    let result = self.call_tool_uncached(name, args).await?;
    self.cache.set(name, &args, result.clone()).await?;
    Ok(result)
}
```

#### 3. Batching

Group multiple tool calls into single requests:

```rust
// Batch multiple reasoning steps
let batch = BatchRequest::new()
    .add_tool("gigathink", args1)
    .add_tool("laserlogic", args2)
    .add_tool("bedrock", args3);

let results = client.execute_batch(batch).await?;
```

**Impact**: 60% reduction in round-trip latency

#### 4. Async/Streaming

Use async I/O and streaming for large responses:

```rust
// Stream large results instead of buffering
async fn stream_analysis(&self, query: &str)
    -> impl Stream<Item = Result<AnalysisChunk>> {
    stream! {
        let mut analysis = self.start_analysis(query).await?;
        while let Some(chunk) = analysis.next().await {
            yield Ok(chunk?);
        }
    }
}
```

#### 5. Resource Monitoring

Track CPU, memory, and limit active servers:

```rust
// Resource-aware server management
struct ResourceManager {
    max_servers: usize,
    cpu_limit: f64,    // 80% max
    mem_limit: usize,  // GB
}

impl ResourceManager {
    async fn should_start_server(&self) -> bool {
        let cpu = self.current_cpu_usage().await;
        let mem = self.current_memory_usage().await;
        let active = self.active_servers().await;

        cpu < self.cpu_limit
            && mem < self.mem_limit
            && active < self.max_servers
    }
}
```

### Performance Targets

| Metric            | Target    | Notes                      |
| ----------------- | --------- | -------------------------- |
| Tool call latency | <100ms    | P95 for simple tools       |
| Tool call latency | <1s       | P95 for complex reasoning  |
| Throughput        | >1000 QPS | Per server instance        |
| Memory per server | <100MB    | Baseline, excluding caches |
| Startup time      | <1s       | Time to ready state        |

**ReasonKit Targets** (from CONS-005):

- All core loops <5ms
- Sub-second response for ThinkTools
- Benchmark required for all optimizations

---

## Deployment Patterns

### Containerization (Standard)

**Source**: [Docker MCP Best Practices](https://www.docker.com/blog/mcp-server-best-practices/)

**Benefit**: 60% reduction in deployment-related support tickets

```dockerfile
# Rust MCP server Dockerfile
FROM rust:1.74 AS builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/mcp-server /usr/local/bin/

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s \
  CMD /usr/local/bin/mcp-server health || exit 1

ENTRYPOINT ["/usr/local/bin/mcp-server"]
CMD ["--transport", "http", "--port", "8080"]
```

### Kubernetes Orchestration

**Source**: [AWS MCP Deployment Guidance](https://aws.amazon.com/solutions/guidance/deploying-model-context-protocol-servers-on-aws/)

```yaml
# MCP server deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: reasonkit-mcp-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: reasonkit-mcp
  template:
    metadata:
      labels:
        app: reasonkit-mcp
    spec:
      containers:
        - name: mcp-server
          image: reasonkit/mcp-server:latest
          ports:
            - containerPort: 8080
              name: http
          env:
            - name: MCP_TRANSPORT
              value: "http"
          resources:
            requests:
              cpu: 100m
              memory: 128Mi
            limits:
              cpu: 500m
              memory: 512Mi
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 5
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: http
            initialDelaySeconds: 3
            periodSeconds: 5
---
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: reasonkit-mcp-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: reasonkit-mcp-server
  minReplicas: 3
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
---
# Service
apiVersion: v1
kind: Service
metadata:
  name: reasonkit-mcp-service
spec:
  selector:
    app: reasonkit-mcp
  ports:
    - port: 80
      targetPort: 8080
  type: ClusterIP
```

### AWS Architecture

**Source**: [AWS MCP Guidance](https://aws.amazon.com/solutions/guidance/deploying-model-context-protocol-servers-on-aws/)

```
┌──────────────────────────────────────────────────────────────┐
│ CloudFront (CDN + HTTPS + DDoS protection)                   │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│ AWS WAF (Web Application Firewall + Rate Limiting)           │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│ ALB (Application Load Balancer)                              │
│   - Multi-AZ                                                  │
│   - Health checks                                             │
│   - Connection draining                                       │
└────────────┬─────────────────────┬─────────────────────┬─────┘
             │                     │                     │
┌────────────▼──────┐  ┌──────────▼──────┐  ┌──────────▼──────┐
│ ECS Fargate       │  │ ECS Fargate     │  │ ECS Fargate     │
│ MCP Server        │  │ MCP Server      │  │ MCP Server      │
│ (AZ-1)            │  │ (AZ-2)          │  │ (AZ-3)          │
└────────┬──────────┘  └──────┬──────────┘  └──────┬──────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────┐
│ Amazon Cognito (OAuth 2.0 Authorization)                   │
└────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────┐
│ AWS Secrets Manager (API keys, credentials)                │
└────────────────────────────────────────────────────────────┘
```

**Key Components**:

- **CloudFront**: HTTPS, HTTP/2, HTTP/3, DDoS protection
- **WAF**: Common exploit protection, rate limiting
- **ALB**: Multi-AZ routing, health checks, target groups
- **ECS Fargate**: Serverless containers, auto-scaling
- **Cognito**: OAuth 2.0 with authorization code grant flow
- **Secrets Manager**: Credential rotation, encryption at rest

### Deployment Strategies

**Source**: [MCP Production Guide](https://medium.com/@jalajagr/mcp-in-production-environments-a-complete-guide-6649c62cac81)

| Strategy       | Description                                | Downtime | Risk     |
| -------------- | ------------------------------------------ | -------- | -------- |
| **Blue-Green** | Two identical environments, switch traffic | Zero     | Low      |
| **Canary**     | Gradual rollout (5% → 25% → 100%)          | Zero     | Very Low |
| **Rolling**    | Update instances one at a time             | Zero     | Medium   |
| **Recreate**   | Stop all, deploy new                       | Yes      | High     |

**Recommended**: Canary deployment for production MCP servers

```yaml
# Argo Rollouts canary deployment
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: reasonkit-mcp-rollout
spec:
  replicas: 10
  strategy:
    canary:
      steps:
        - setWeight: 5 # 5% traffic
        - pause: { duration: 5m }
        - setWeight: 25 # 25% traffic
        - pause: { duration: 10m }
        - setWeight: 50 # 50% traffic
        - pause: { duration: 10m }
        - setWeight: 100 # Full rollout
```

### CI/CD Pipeline

**Source**: [15 Best Practices](https://thenewstack.io/15-best-practices-for-building-mcp-servers-in-production/)

```yaml
# GitHub Actions example
name: Deploy MCP Server
on:
  push:
    branches: [main]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build Docker image
        run: docker build -t reasonkit/mcp-server:${{ github.sha }} .

      - name: Run tests
        run: |
          cargo test --all-features
          cargo clippy -- -D warnings
          cargo bench

      - name: Security scan
        run: |
          cargo audit
          snyk test

      - name: Push to registry
        run: docker push reasonkit/mcp-server:${{ github.sha }}

      - name: Deploy to K8s (canary)
        run: |
          kubectl set image deployment/reasonkit-mcp \
            mcp-server=reasonkit/mcp-server:${{ github.sha }}
          kubectl rollout status deployment/reasonkit-mcp
```

---

## Production Readiness

### Monitoring and Observability

**Source**: [Secure MCP Deployment](https://mcpmanager.ai/blog/secure-mcp-server-deployment-at-scale-the-complete-guide/)

#### Essential Metrics

| Category        | Metrics                                             | Tools                   |
| --------------- | --------------------------------------------------- | ----------------------- |
| **Performance** | Latency (P50/P95/P99), Throughput (QPS), Error rate | Prometheus, Grafana     |
| **Resources**   | CPU, Memory, Disk I/O, Network                      | cAdvisor, Node Exporter |
| **Business**    | Tool calls, Success rate, Cost per query            | Custom dashboards       |
| **Security**    | Failed auth, Rate limit hits, Anomalies             | CloudWatch, Splunk      |

#### Logging Strategy

**Source**: [MCP Best Practices](https://modelcontextprotocol.info/docs/best-practices/)

```rust
// Structured logging for MCP operations
#[instrument(skip(self))]
async fn call_tool(&self, name: &str, args: Value) -> Result<ToolResult> {
    let request_id = Uuid::new_v4();

    info!(
        request_id = %request_id,
        tool = name,
        args_hash = %hash(&args),
        timestamp = %Utc::now(),
        user_id = %self.user_id,
        "Tool call started"
    );

    let start = Instant::now();
    let result = self.execute_tool(name, args).await;
    let duration = start.elapsed();

    match &result {
        Ok(r) => info!(
            request_id = %request_id,
            tool = name,
            duration_ms = duration.as_millis(),
            success = true,
            "Tool call completed"
        ),
        Err(e) => error!(
            request_id = %request_id,
            tool = name,
            duration_ms = duration.as_millis(),
            error = %e,
            success = false,
            "Tool call failed"
        ),
    }

    result
}
```

**Log Requirements**:

- Append-only, tamper-evident storage (S3 immutable, write-once)
- Include: timestamp, user ID, tool name, input hash, output, duration
- Retention: 90 days minimum (compliance)
- No secrets in logs (NEVER echo API keys, tokens)

### Testing Approaches

**Source**: [MCP Production Guide](https://dev.to/raghavajoijode/production-ready-mcp-servers-security-performance-deployment-5e48)

| Test Type             | Tool                            | Purpose                |
| --------------------- | ------------------------------- | ---------------------- |
| **Unit Tests**        | `cargo test`                    | Individual functions   |
| **Integration Tests** | `cargo test --test integration` | Tool workflows         |
| **Local Testing**     | MCP Inspector                   | Interactive debugging  |
| **Remote Testing**    | Network-based client            | Real-world scenarios   |
| **Load Testing**      | k6, Locust                      | Performance validation |
| **Security Testing**  | Snyk, cargo-audit               | Vulnerability scanning |

#### MCP Inspector

**Source**: [MCP Best Practices](https://modelcontextprotocol.info/docs/best-practices/)

The MCP Inspector is a specialized tool for testing MCP servers:

```bash
# Start MCP Inspector
npx @modelcontextprotocol/inspector

# Point to your MCP server
# Interactive UI shows:
# - Available tools
# - Tool schemas (JSON Schema)
# - Request/response logs
# - Error diagnostics
```

**Features**:

- Interactive tool testing
- Schema inspection
- Request/response logging
- Error diagnosis
- Latency measurement

### Health Checks

```rust
// Health check endpoint
#[derive(Serialize)]
struct HealthStatus {
    status: String,        // "healthy" | "degraded" | "unhealthy"
    uptime_secs: u64,
    requests_total: u64,
    errors_last_hour: u64,
    avg_latency_ms: f64,
    dependencies: HashMap<String, String>,  // Each dependency status
}

async fn health_check(&self) -> HealthStatus {
    HealthStatus {
        status: if self.errors_rate() < 0.01 { "healthy" } else { "degraded" },
        uptime_secs: self.uptime().as_secs(),
        requests_total: self.metrics.requests_total(),
        errors_last_hour: self.metrics.errors_in_window(Duration::from_secs(3600)),
        avg_latency_ms: self.metrics.avg_latency().as_millis() as f64,
        dependencies: self.check_dependencies().await,
    }
}
```

### Error Handling Best Practices

**Source**: [MCP Best Practices](https://mcpcat.io/blog/mcp-server-best-practices/)

Error messages should be:

1. **LLM-parsable**: Machine-readable error codes
2. **Human-readable**: Clear explanations
3. **Actionable**: Help agent decide next steps

```rust
// ❌ BAD: Human-only error
Err("Something went wrong")

// ✅ GOOD: Structured, actionable error
Err(McpError {
    code: ErrorCode::RateLimitExceeded,
    message: "Rate limit exceeded: 100 requests/minute",
    suggestion: "Retry after 60 seconds or upgrade plan",
    retry_after_secs: Some(60),
    data: json!({
        "current_rate": 120,
        "limit": 100,
        "window_secs": 60
    })
})
```

**Error Codes** (from MCP spec):

- `-32700`: Parse error
- `-32600`: Invalid request
- `-32601`: Method not found
- `-32602`: Invalid params
- `-32603`: Internal error
- `-32000` to `-32099`: Server-defined errors

### Versioning Strategy

```rust
// Version in server info
ServerInfo {
    name: "reasonkit-mcp-server".to_string(),
    version: "1.2.3".to_string(),  // Semantic versioning
    protocol_version: "2025-11-25".to_string(),
    capabilities: ServerCapabilities {
        tools: Some(ToolsCapability { list_changed: true }),
        resources: None,
        prompts: Some(PromptsCapability {}),
    }
}
```

**Versioning Rules**:

- **Major**: Breaking changes (e.g., 1.x → 2.0)
- **Minor**: New features, backward compatible (e.g., 1.2 → 1.3)
- **Patch**: Bug fixes, no new features (e.g., 1.2.3 → 1.2.4)

---

## ReasonKit-Specific Recommendations

### Current State Analysis

**ReasonKit MCP Implementation** (from `/home/zyxsys/RK-PROJECT/reasonkit-core/src/mcp/`):

| Component      | Status         | Spec Version | Notes                 |
| -------------- | -------------- | ------------ | --------------------- |
| `mod.rs`       | ✅ Implemented | 2025-11-25   | Module structure good |
| `types.rs`     | ✅ Implemented | Current      | JSON-RPC 2.0 types    |
| `client.rs`    | ✅ Implemented | Current      | MCP client with retry |
| `server.rs`    | ✅ Implemented | Current      | Basic server          |
| `registry.rs`  | ✅ Implemented | Current      | Health monitoring     |
| `transport.rs` | ⚠️ stdio only  | Current      | HTTP not implemented  |
| `tools.rs`     | ✅ Implemented | Current      | Tool definitions      |
| `lifecycle.rs` | ✅ Implemented | Current      | Init/shutdown         |

**Strengths**:

- Full stdio transport implementation
- Client with auto-reconnect and retry
- Health monitoring in registry
- Good module structure

**Gaps vs. Best Practices**:

- ❌ No Streamable HTTP transport (production standard)
- ❌ No OAuth 2.1 implementation (required for HTTP)
- ❌ No Tasks abstraction (Nov 2025 spec)
- ❌ No structured outputs (Jun 2025 spec)
- ❌ No benchmarking/testing framework
- ❌ No containerization/deployment configs

### Priority Enhancements

#### Priority 1: Streamable HTTP Transport (HIGH)

**Why**: Production deployments require HTTP, stdio is dev-only

**Implementation**:

```rust
// src/mcp/transport/http.rs (NEW)
use axum::{
    Router,
    routing::{get, post},
    extract::{State, Json},
    response::{Response, sse::Event},
};

pub struct HttpTransport {
    port: u16,
    session_manager: Arc<SessionManager>,
}

impl HttpTransport {
    async fn handle_post(&self, Json(req): Json<McpRequest>)
        -> Result<Response> {
        // Handle client-to-server request
        let response = self.server.handle_request(req).await?;

        // Return JSON or SSE based on needs
        if response.is_streaming() {
            Ok(Response::sse(response.stream()))
        } else {
            Ok(Response::json(response))
        }
    }

    async fn handle_get(&self, session_id: String)
        -> Result<impl Stream<Item = Event>> {
        // Open SSE stream for proactive server messages
        Ok(self.session_manager.subscribe(session_id).await?)
    }
}

// Server setup
pub fn create_http_server(server: McpServer) -> Router {
    Router::new()
        .route("/mcp", post(HttpTransport::handle_post))
        .route("/mcp", get(HttpTransport::handle_get))
        .route("/health", get(health_check))
        .with_state(Arc::new(server))
}
```

**Effort**: 2-3 weeks
**Impact**: Enables production deployment
**Dependencies**: `axum`, `tower`, `tower-http`

#### Priority 2: OAuth 2.1 Integration (HIGH)

**Why**: Mandatory for HTTP transports, enterprise requirement

**Approach**: Delegated (Auth0/Okta)

```rust
// src/mcp/auth/oauth.rs (NEW)
use oauth2::{
    AuthorizationCode, TokenResponse,
    basic::BasicClient,
    reqwest::async_http_client,
};

pub struct OAuthConfig {
    pub auth_server_url: String,
    pub client_id: String,
    pub client_secret: String,
    pub resource_uri: String,  // MCP server URI
}

pub struct OAuthMiddleware {
    client: BasicClient,
    token_verifier: Arc<TokenVerifier>,
}

impl OAuthMiddleware {
    async fn verify_token(&self, token: &str) -> Result<Claims> {
        // 1. Validate signature (RS256 JWT)
        // 2. Check issuer matches auth server
        // 3. Check audience matches MCP server URI
        // 4. Check expiry
        // 5. Check scopes (optional)

        self.token_verifier.verify(token).await
    }
}

// Axum middleware
async fn require_auth(
    State(auth): State<Arc<OAuthMiddleware>>,
    headers: HeaderMap,
    req: Request,
    next: Next,
) -> Result<Response> {
    let token = headers
        .get("Authorization")
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "))
        .ok_or(Error::Unauthorized)?;

    let claims = auth.verify_token(token).await?;

    // Inject user context into request
    req.extensions_mut().insert(claims);

    Ok(next.run(req).await)
}
```

**Effort**: 3-4 weeks
**Impact**: Enterprise-ready security
**Dependencies**: `oauth2`, `jsonwebtoken`

**ReasonKit Integration**:

- **reasonkit-core** (OSS): No auth (stdio only)
- **reasonkit-pro** (Paid): Full OAuth 2.1 with Auth0

#### Priority 3: Tasks Abstraction (MEDIUM)

**Why**: Nov 2025 spec, perfect for long-running ThinkTools

**Use Cases**:

- `ProofGuard`: Multi-source verification (can take minutes)
- `GigaThink`: 10+ perspective generation
- `PowerCombo`: Full 5-module chain

```rust
// src/mcp/tasks.rs (NEW)
use uuid::Uuid;

#[derive(Serialize, Deserialize)]
pub struct Task {
    pub id: Uuid,
    pub name: String,
    pub status: TaskStatus,
    pub progress: f32,  // 0.0 to 1.0
    pub result: Option<Value>,
    pub error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    Working,
    Completed,
    Failed,
}

pub struct TaskManager {
    tasks: Arc<RwLock<HashMap<Uuid, Task>>>,
}

impl TaskManager {
    pub async fn create_task(&self, name: String) -> Uuid {
        let task = Task {
            id: Uuid::new_v4(),
            name,
            status: TaskStatus::Pending,
            progress: 0.0,
            result: None,
            error: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        let id = task.id;
        self.tasks.write().await.insert(id, task);
        id
    }

    pub async fn get_task(&self, id: Uuid) -> Option<Task> {
        self.tasks.read().await.get(&id).cloned()
    }

    pub async fn update_progress(&self, id: Uuid, progress: f32) {
        if let Some(task) = self.tasks.write().await.get_mut(&id) {
            task.progress = progress;
            task.updated_at = Utc::now();
        }
    }

    pub async fn complete_task(&self, id: Uuid, result: Value) {
        if let Some(task) = self.tasks.write().await.get_mut(&id) {
            task.status = TaskStatus::Completed;
            task.progress = 1.0;
            task.result = Some(result);
            task.updated_at = Utc::now();
        }
    }
}
```

**Tool Integration**:

```rust
#[tool(description = "Multi-source verification with progress tracking")]
async fn proofguard_async(&self, claim: String, sources: Vec<String>)
    -> CallToolResult {
    // Create task
    let task_id = self.tasks.create_task("proofguard".to_string()).await;

    // Return task ID immediately
    Ok(CallToolResult {
        content: vec![TextContent {
            text: format!("Verification started. Task ID: {}", task_id),
        }],
        task_id: Some(task_id),
        is_error: false,
    })

    // Background processing
    tokio::spawn(async move {
        for (i, source) in sources.iter().enumerate() {
            // Verify source
            let verification = verify_source(&claim, source).await?;

            // Update progress
            self.tasks.update_progress(
                task_id,
                (i + 1) as f32 / sources.len() as f32
            ).await;
        }

        // Complete task
        self.tasks.complete_task(task_id, result).await;
    });
}
```

**Effort**: 2-3 weeks
**Impact**: Better UX for long operations
**Dependencies**: None (tokio already used)

#### Priority 4: Performance Benchmarking (MEDIUM)

**Why**: Validate Rust performance claims, track regressions

```rust
// benches/mcp_bench.rs (NEW)
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_tool_call(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let client = rt.block_on(async {
        let mut client = McpClient::new(test_config());
        client.connect().await.unwrap();
        client
    });

    c.bench_function("tool_call_gigathink", |b| {
        b.to_async(&rt).iter(|| async {
            let result = client.call_tool(
                "gigathink",
                json!({"query": "Test query"})
            ).await;
            black_box(result)
        })
    });
}

fn bench_batch_tools(c: &mut Criterion) {
    // Benchmark batching strategy
}

criterion_group!(benches, bench_tool_call, bench_batch_tools);
criterion_main!(benches);
```

**Targets** (from research):

- Individual tool call: <100ms (P95)
- Batch of 5 tools: <500ms (P95)
- Throughput: >1000 QPS per instance

**Effort**: 1 week
**Impact**: Performance validation, regression detection

#### Priority 5: Containerization (LOW)

**Why**: Production deployment, consistency

```dockerfile
# reasonkit-core/Dockerfile (NEW)
FROM rust:1.74 AS builder
WORKDIR /build

# Cache dependencies
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
RUN rm -rf src

# Build actual binary
COPY . .
RUN cargo build --release --bin rk-mcp-server

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/rk-mcp-server /usr/local/bin/

HEALTHCHECK --interval=30s --timeout=5s \
  CMD /usr/local/bin/rk-mcp-server health || exit 1

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/rk-mcp-server"]
CMD ["--transport", "http", "--port", "8080"]
```

**Effort**: 1 week
**Impact**: Deployment-ready containers

### Development Roadmap

**Phase 1: Production HTTP (Q1 2025)**

- [ ] Implement Streamable HTTP transport
- [ ] Add OAuth 2.1 middleware
- [ ] Create health check endpoint
- [ ] Write integration tests

**Phase 2: Advanced Features (Q2 2025)**

- [ ] Tasks abstraction for long operations
- [ ] Structured outputs (JSON Schema)
- [ ] Prompt macros for chaining
- [ ] Performance benchmarks

**Phase 3: Enterprise Deployment (Q3 2025)**

- [ ] Containerization (Docker, K8s)
- [ ] Monitoring dashboards (Prometheus, Grafana)
- [ ] Load testing and optimization
- [ ] Production deployment guides

**Phase 4: Scale & Optimize (Q4 2025)**

- [ ] Connection pooling
- [ ] Caching strategies
- [ ] Multi-region deployment
- [ ] Cost optimization

### Testing Strategy

```bash
# Unit tests
cargo test --lib mcp

# Integration tests
cargo test --test mcp_integration

# Benchmark tests
cargo bench --bench mcp_bench

# Security audit
cargo audit
cargo clippy -- -D warnings

# Load testing (future)
k6 run scripts/load_test.js
```

### Configuration Management

```toml
# config/mcp.toml
[server]
name = "reasonkit-mcp-server"
version = "1.0.0"
protocol_version = "2025-11-25"

[transport]
type = "http"  # or "stdio"
port = 8080
max_connections = 1000

[transport.http]
enable_sse = true
session_ttl_secs = 3600
max_message_size_mb = 10

[auth]
enabled = true
provider = "auth0"  # or "okta", "google"
issuer_url = "https://reasonkit.auth0.com"
audience = "https://api.reasonkit.sh/mcp"

[performance]
cache_enabled = true
cache_ttl_secs = 300
max_concurrent_tasks = 100
request_timeout_secs = 30

[monitoring]
metrics_enabled = true
logging_level = "info"
health_check_interval_secs = 30
```

---

## References

### Official Specifications

1. [MCP Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) - Latest official spec
2. [MCP GitHub Repository](https://github.com/modelcontextprotocol/modelcontextprotocol) - Source code and examples
3. [MCP One Year Anniversary](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) - November 2025 updates
4. [MCP June 2025 Spec Updates](https://auth0.com/blog/mcp-specs-update-all-about-auth/) - OAuth 2.1 and structured outputs

### Best Practices

5. [MCP Best Practices Guide](https://modelcontextprotocol.info/docs/best-practices/) - Official best practices
6. [7 MCP Server Best Practices for 2025](https://www.marktechpost.com/2025/07/23/7-mcp-server-best-practices-for-scalable-ai-integrations-in-2025/) - MarkTechPost guide
7. [15 Best Practices for Production](https://thenewstack.io/15-best-practices-for-building-mcp-servers-in-production/) - The New Stack
8. [MCP Server Best Practices (MCPcat)](https://mcpcat.io/blog/mcp-server-best-practices/) - Production-grade development

### Rust Implementation

9. [Official Rust SDK (rmcp)](https://github.com/modelcontextprotocol/rust-sdk) - Official implementation
10. [Building High-Performance MCP with Rust](https://medium.com/@bohachu/building-a-high-performance-mcp-server-with-rust-a-complete-implementation-guide-8a18ab16b538) - Performance guide
11. [Rust MCP Comparison (Shuttle)](https://www.shuttle.dev/blog/2025/09/15/mcp-servers-rust-comparison) - Framework comparison
12. [Prism MCP Rust SDK](https://users.rust-lang.org/t/prism-mcp-rust-sdk-v0-1-0-production-grade-model-context-protocol-implementation/133318) - Production-grade SDK

### Transport Protocols

13. [Why MCP Deprecated SSE](https://blog.fka.dev/blog/2025-06-06-why-mcp-deprecated-sse-and-go-with-streamable-http/) - Streamable HTTP explanation
14. [Understanding MCP HTTP+SSE Change](https://blog.christianposta.com/ai/understanding-mcp-recent-change-around-http-sse/) - Transport evolution
15. [MCP Streamable HTTP Guide](https://nico.bistol.fi/blog/mcp-streamable-http/) - Implementation guide

### Security

16. [MCP OAuth 2.1 Complete Guide](https://dev.to/composiodev/mcp-oauth-21-a-complete-guide-3g91) - OAuth implementation
17. [Secure MCP Server with OAuth 2.1](https://www.scalekit.com/blog/implement-oauth-for-mcp-servers) - Scalekit guide
18. [MCP Authorization Specification](https://modelcontextprotocol.io/specification/draft/basic/authorization) - Official auth spec
19. [Top MCP Security Best Practices](https://www.akto.io/blog/mcp-security-best-practices) - Security checklist
20. [OWASP MCP Security CheatSheet](https://genai.owasp.org/resource/cheatsheet-a-practical-guide-for-securely-using-third-party-mcp-servers-1-0/) - Security guide

### Performance & Deployment

21. [Top 10 MCP Performance Techniques](https://superagi.com/top-10-advanced-techniques-for-optimizing-mcp-server-performance-in-2025/) - Optimization guide
22. [AWS MCP Deployment Guidance](https://aws.amazon.com/solutions/guidance/deploying-model-context-protocol-servers-on-aws/) - AWS architecture
23. [Secure MCP Deployment at Scale](https://mcpmanager.ai/blog/secure-mcp-server-deployment-at-scale-the-complete-guide/) - Production deployment
24. [MCP in Production Environments](https://medium.com/@jalajagr/mcp-in-production-environments-a-complete-guide-6649c62cac81) - Complete guide
25. [Docker MCP Best Practices](https://www.docker.com/blog/mcp-server-best-practices/) - Containerization

### Benchmarks

26. [MCP Atlas Benchmark](https://scale.com/leaderboard/mcp_atlas) - Scale AI benchmark (1000 tasks)
27. [MCP-Bench (Accenture)](https://github.com/Accenture/mcp-bench) - NeurIPS 2025 benchmark
28. [Twilio MCP Performance Testing](https://www.twilio.com/en-us/blog/developers/twilio-alpha-mcp-server-real-world-performance) - Real-world results
29. [Top MCP Servers for Web Access](https://research.aimultiple.com/browser-mcp/) - Comparative benchmarks

### Community Resources

30. [GitHub MCP Servers Repository](https://github.com/modelcontextprotocol/servers) - Official server collection
31. [How to Build Secure Remote MCP Servers](https://github.blog/ai-and-ml/generative-ai/how-to-build-secure-and-scalable-remote-mcp-servers/) - GitHub guide

---

## Appendix A: MCP Protocol Quick Reference

### JSON-RPC 2.0 Message Format

```json
{
  "jsonrpc": "2.0",
  "id": "req-123",
  "method": "tools/call",
  "params": {
    "name": "gigathink",
    "arguments": {
      "query": "What is structured reasoning?"
    }
  }
}
```

### Lifecycle Methods

| Method        | Direction       | Purpose                            |
| ------------- | --------------- | ---------------------------------- |
| `initialize`  | Client → Server | Handshake, capability negotiation  |
| `initialized` | Client → Server | Notification after successful init |
| `ping`        | Either → Either | Health check                       |
| `shutdown`    | Client → Server | Graceful shutdown request          |

### Tool Methods

| Method       | Direction       | Purpose             |
| ------------ | --------------- | ------------------- |
| `tools/list` | Client → Server | Get available tools |
| `tools/call` | Client → Server | Execute tool        |

### Resource Methods

| Method           | Direction       | Purpose                 |
| ---------------- | --------------- | ----------------------- |
| `resources/list` | Client → Server | Get available resources |
| `resources/read` | Client → Server | Read resource content   |

### Prompt Methods

| Method         | Direction       | Purpose               |
| -------------- | --------------- | --------------------- |
| `prompts/list` | Client → Server | Get available prompts |
| `prompts/get`  | Client → Server | Get prompt template   |

### Task Methods (Nov 2025)

| Method         | Direction       | Purpose             |
| -------------- | --------------- | ------------------- |
| `tasks/create` | Client → Server | Create async task   |
| `tasks/status` | Client → Server | Get task status     |
| `tasks/result` | Client → Server | Get task result     |
| `tasks/cancel` | Client → Server | Cancel running task |

---

## Appendix B: ReasonKit ThinkTools as MCP Servers

### Mapping

| ThinkTool     | MCP Server Name           | Primary Tool            | Description                                      |
| ------------- | ------------------------- | ----------------------- | ------------------------------------------------ |
| GigaThink     | `reasonkit-gigathink`     | `generate_perspectives` | Expansive creative thinking, 10+ viewpoints      |
| LaserLogic    | `reasonkit-laserlogic`    | `validate_logic`        | Precision deductive reasoning, fallacy detection |
| BedRock       | `reasonkit-bedrock`       | `decompose_to_axioms`   | First principles decomposition                   |
| ProofGuard    | `reasonkit-proofguard`    | `verify_claims`         | Multi-source verification, triangulation         |
| BrutalHonesty | `reasonkit-brutalhonesty` | `adversarial_critique`  | Ruthless self-critique                           |

### Example: GigaThink MCP Server

```rust
#[derive(ToolRouter)]
struct GigaThinkServer {
    executor: GigaThinkExecutor,
}

#[tool_router]
impl GigaThinkServer {
    #[tool(description = "Generate 10+ diverse perspectives on a question")]
    async fn generate_perspectives(&self,
        query: String,
        perspective_count: Option<u8>,
    ) -> CallToolResult {
        let count = perspective_count.unwrap_or(10);
        let result = self.executor.execute(query, count).await?;

        Ok(CallToolResult {
            content: vec![
                TextContent {
                    text: serde_json::to_string_pretty(&result)?
                }
            ],
            is_error: false,
        })
    }

    #[resource(uri = "reasonkit://gigathink/config")]
    async fn get_config(&self) -> ResourceContent {
        // Return configuration as resource
    }
}
```

---

## Appendix C: Glossary

| Term                | Definition                                                |
| ------------------- | --------------------------------------------------------- |
| **MCP**             | Model Context Protocol - Standard for AI tool integration |
| **JSON-RPC 2.0**    | Remote procedure call protocol using JSON                 |
| **stdio**           | Standard input/output transport (local, dev)              |
| **Streamable HTTP** | HTTP transport with optional SSE streaming                |
| **SSE**             | Server-Sent Events - HTTP streaming mechanism             |
| **OAuth 2.1**       | Modern authorization framework                            |
| **PKCE**            | Proof Key for Code Exchange - OAuth security extension    |
| **Tool**            | Executable function exposed by MCP server                 |
| **Resource**        | Data source exposed by MCP server                         |
| **Prompt**          | Template for multi-step workflows                         |
| **Task**            | Async work tracker (Nov 2025 feature)                     |
| **ThinkTool**       | ReasonKit reasoning module (GigaThink, LaserLogic, etc.)  |

---

**Document Version**: 1.0.0
**Last Updated**: 2025-12-25
**Maintainer**: ReasonKit Core Team
**License**: Apache 2.0 (consistent with reasonkit-core)

---

_"Turn Prompts into Protocols" - https://reasonkit.sh_
