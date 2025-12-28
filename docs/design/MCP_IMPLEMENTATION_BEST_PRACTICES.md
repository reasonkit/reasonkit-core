# MCP Server Implementation Best Practices for ReasonKit

> Research Report: Model Context Protocol (MCP) Server Implementation
> Focus: Rust-based MCP servers for RAG/Reasoning tools
> Date: 2025-12-25
> Specification Version: MCP 2025-11-25

---

## Executive Summary

This document provides comprehensive best practices for implementing Model Context Protocol (MCP) servers in Rust, specifically tailored for exposing ReasonKit's RAG and reasoning capabilities. Based on the latest MCP specification (2025-11-25), industry patterns, and consultation with AI reasoning systems, these recommendations prioritize **performance, security, developer experience, and composability**.

**Key Findings:**
- **Tool Design**: Hybrid approach combining coarse-grained session tools with fine-grained atomic tools
- **Transport**: stdio for Claude Desktop, HTTP/SSE for distributed deployments
- **Performance**: Lazy initialization + Moka caching can achieve <1ms P99 latency
- **Security**: Multi-layer defense with input sanitization, resource limits, and sandboxing
- **Testing**: Self-describing tools + MCP Inspector enable rapid iteration

---

## 1. MCP Architecture Overview

### 1.1 Protocol Fundamentals

**MCP Specification Version**: 2025-11-25 ([Official Spec](https://modelcontextprotocol.io/specification/2025-11-25))

The Model Context Protocol is built on **JSON-RPC 2.0** and provides three core primitives:

1. **Tools**: Executable functions that LLMs can invoke
2. **Resources**: Data sources that provide context
3. **Prompts**: Reusable prompt templates

**Key 2025 Updates:**
- Tool calling in sampling requests (server-side agent loops)
- Parallel tool execution support
- Task abstraction for work tracking
- Improved capability declarations (deprecation of ambiguous `includeContext`)

**Transport Options:**
- **stdio**: JSON-RPC over standard input/output (primary for local servers)
- **HTTP with SSE**: Server-Sent Events for server-to-client messages, HTTP POST for client-to-server
- **WebSocket**: (Future consideration)

**Protocol Flow:**
```
Client                          Server
  |                               |
  |---- initialize ------>        |
  |                               |
  |<--- initialized -----         |
  |                               |
  |---- tools/list ------>        |
  |<--- [tools] ---------         |
  |                               |
  |---- tools/call ------>        |
  |       {tool, args}            |
  |<--- result -----------        |
  |                               |
  |---- shutdown -------->        |
  |<--- OK --------------         |
```

**References:**
- [MCP Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25)
- [One Year of MCP Blog Post](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/)
- [MCP GitHub Repository](https://github.com/modelcontextprotocol/modelcontextprotocol)

### 1.2 Official Rust SDK

**Repository**: [modelcontextprotocol/rust-sdk](https://github.com/modelcontextprotocol/rust-sdk)

**Core Crates:**
- `rmcp`: Core protocol implementation
- `rmcp-macros`: Procedural macros for tool generation

**Key Features:**
- Tokio async runtime
- Type-safe tool definitions via `#[tool_router]` and `#[tool]` macros
- Automatic JSON Schema generation
- Built-in protocol compliance

**Basic Server Structure:**
```rust
use rmcp::*;

#[tool_router]
struct MyServer {
    // Server state
}

#[tool]
impl MyServer {
    async fn my_tool(&self, arg: String) -> Result<String> {
        // Tool implementation
        Ok(format!("Processed: {}", arg))
    }
}
```

**References:**
- [Rust MCP SDK GitHub](https://github.com/modelcontextprotocol/rust-sdk)
- [Building MCP Servers in Rust - MCPcat](https://mcpcat.io/guides/building-mcp-server-rust/)
- [Shuttle Rust MCP Tutorial](https://www.shuttle.dev/blog/2025/07/18/how-to-build-a-stdio-mcp-server-in-rust)

---

## 2. Tool Design Patterns

### 2.1 Granularity: Atomic vs Workflow-Based

**The Core Tradeoff**: Token efficiency vs. flexibility

**Problem Statement**: Each tool exposed to an LLM consumes context window space. With ReasonKit's 5+ ThinkTools, naive exposure can degrade performance.

**Best Practice**: **Hybrid Two-Tier Architecture**

```
High-Level (Workflow Tools)        Low-Level (Atomic Tools)
├─ session/create                  ├─ gigathink
├─ session/step                    ├─ laserlogic
├─ session/execute_chain           ├─ bedrock
├─ profile/quick                   ├─ proofguard
├─ profile/balanced                └─ brutalhonesty
├─ profile/paranoid
└─ profile/powercombo
```

**Rationale:**
1. **Workflow tools** (e.g., `profile/balanced`) handle complete reasoning chains in one call
   - Reduces LLM decision points (no multi-turn orchestration)
   - Minimizes token overhead (single description vs. 5 tool descriptions)
   - Better for production/batch processing

2. **Atomic tools** (e.g., `gigathink`) enable fine-grained composition
   - Supports interactive/exploratory use cases
   - Allows LLM to dynamically adjust reasoning strategy
   - Better for research/development workflows

**Implementation Example:**
```rust
pub enum ReasoningTool {
    // Workflow tools
    ProfileQuick,
    ProfileBalanced,
    ProfileParanoid,

    // Atomic tools
    GigaThink,
    LaserLogic,
    BedRock,
    ProofGuard,
    BrutalHonesty,

    // Session management
    SessionCreate,
    SessionStep,
    SessionExecuteChain,
}
```

**Tool Schema Design:**
```json
{
  "name": "profile/balanced",
  "description": "Execute balanced reasoning profile (GigaThink → LaserLogic → BedRock → ProofGuard). Returns comprehensive analysis with ~80% confidence.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": { "type": "string", "description": "Question or problem to analyze" },
      "context": { "type": "string", "description": "Optional background context" },
      "session_id": { "type": "string", "description": "Optional session ID for stateful execution" }
    },
    "required": ["query"]
  }
}
```

**References:**
- [Less is More: 4 MCP Design Patterns](https://www.klavis.ai/blog/less-is-more-mcp-design-patterns-for-ai-agents)
- [MCP Tool Design Best Practices](https://dev.to/klavisai/less-is-more-4-design-patterns-for-building-better-mcp-servers-3gpf)

### 2.2 State Management Between Reasoning Steps

**Challenge**: ThinkTools need to pass context between steps (e.g., BedRock uses GigaThink's perspectives)

**Pattern 1: Session-Based Stateful Execution** (Recommended for ReasonKit)

```rust
pub struct ReasoningSession {
    pub id: Uuid,
    pub profile: ReasoningProfile,
    pub trace: ExecutionTrace,
    pub step_outputs: HashMap<String, StepOutput>,  // Cross-step state
    pub cursor: usize,
    pub status: SessionStatus,
    pub created_at: DateTime<Utc>,
    pub ttl: Duration,
}

pub struct SessionStore {
    sessions: DashMap<Uuid, ReasoningSession>,
    ttl_cleanup_task: tokio::task::JoinHandle<()>,
}

impl SessionStore {
    pub async fn create_session(&self, profile: ReasoningProfile) -> Uuid {
        let session = ReasoningSession {
            id: Uuid::new_v4(),
            profile,
            trace: ExecutionTrace::new(),
            step_outputs: HashMap::new(),
            cursor: 0,
            status: SessionStatus::Active,
            created_at: Utc::now(),
            ttl: Duration::from_secs(3600), // 1 hour default
        };
        let id = session.id;
        self.sessions.insert(id, session);
        id
    }

    pub async fn execute_step(&self, session_id: Uuid, input: String) -> Result<StepOutput> {
        let mut session = self.sessions.get_mut(&session_id)
            .ok_or_else(|| Error::NotFound("Session not found".into()))?;

        let current_step = session.profile.steps[session.cursor].clone();

        // Access previous step outputs for context
        let context = self.build_context_from_previous_steps(&session.step_outputs);

        // Execute step with context
        let output = execute_thinktool(&current_step, &input, &context).await?;

        // Store output for next steps
        session.step_outputs.insert(current_step.name.clone(), output.clone());
        session.cursor += 1;

        Ok(output)
    }
}
```

**Advantages:**
- Server owns state (no client-side session management complexity)
- Efficient for multi-step reasoning chains
- Supports resumption after failures
- Auditability: full trace persisted server-side

**Disadvantages:**
- Requires sticky sessions (can complicate load balancing)
- Memory overhead for active sessions
- Cleanup complexity (TTL-based expiration)

**Pattern 2: Resource-Based State** (Alternative for distributed systems)

```rust
// State exposed via MCP resources, not tools
impl McpServerTrait for ReasoningServer {
    async fn list_resources(&self) -> Result<Vec<Resource>> {
        vec![
            Resource {
                uri: "reasoning://sessions".to_string(),
                name: "Active Sessions".to_string(),
                description: Some("List of active reasoning sessions".into()),
                mime_type: Some("application/json".into()),
            },
            Resource {
                uri: "reasoning://traces/{session_id}".to_string(),
                name: "Execution Trace".to_string(),
                description: Some("Step-by-step reasoning trace".into()),
                mime_type: Some("application/json".into()),
            },
        ]
    }

    async fn read_resource(&self, uri: &str) -> Result<ResourceContent> {
        if let Some(session_id) = parse_trace_uri(uri) {
            let trace = self.storage.get_trace(session_id)?;
            Ok(ResourceContent::json(trace))
        } else {
            Err(Error::NotFound("Resource not found".into()))
        }
    }
}

// Tools reference resources via URIs
pub struct ToolInput {
    query: String,
    context_refs: Vec<String>,  // ["reasoning://traces/abc123/step/1"]
}
```

**Advantages:**
- Stateless server (easier horizontal scaling)
- Cross-session composition (reference outputs from different sessions)
- RESTful architecture (familiar to developers)

**Disadvantages:**
- Client manages URI resolution
- More network overhead (resource fetches before tool calls)
- Complexity in URI schema design

**Recommendation for ReasonKit**: **Use Session-Based (Pattern 1)** for initial implementation
- Aligns with existing `ExecutionTrace` and `ProfileRegistry`
- Simpler client integration (just pass session ID)
- Can migrate to Resource-Based later if distributed deployment needed

### 2.3 Composability and Tool Chaining

**Goal**: Enable LLMs to combine ThinkTools flexibly (e.g., run GigaThink → LaserLogic → custom verification)

**Pattern: Protocol-Driven Declarative Chains**

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolDefinition {
    pub id: String,
    pub name: String,
    pub steps: Vec<ChainStep>,
    pub input_mapping: HashMap<String, String>,
    pub conditions: Vec<BranchCondition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainStep {
    pub id: String,
    pub tool_ref: ToolRef,
    pub input_mapping: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ToolRef {
    Builtin { name: String },              // "gigathink"
    Profile { name: String },              // "profile/balanced"
    Protocol { id: String },               // Reference another protocol
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchCondition {
    pub step_index: usize,
    pub skip_if: String,  // Expression: "confidence > 0.95"
}
```

**Example Protocol Definition (JSON):**
```json
{
  "id": "custom_verification",
  "name": "Custom Verification Chain",
  "steps": [
    {
      "id": "initial_analysis",
      "tool_ref": { "type": "Profile", "name": "profile/balanced" },
      "input_mapping": {}
    },
    {
      "id": "adversarial_check",
      "tool_ref": { "type": "Builtin", "name": "brutalhonesty" },
      "input_mapping": {
        "context": "initial_analysis.output"
      }
    },
    {
      "id": "final_guard",
      "tool_ref": { "type": "Builtin", "name": "proofguard" },
      "input_mapping": {
        "claims": "adversarial_check.identified_weaknesses"
      }
    }
  ],
  "conditions": [
    {
      "step_index": 2,
      "skip_if": "adversarial_check.confidence > 0.95"
    }
  ]
}
```

**MCP Tool Exposure:**
```rust
Tool {
    name: "protocol/define".to_string(),
    description: Some("Define custom reasoning protocol".into()),
    input_schema: protocol_definition_schema(),
}

Tool {
    name: "protocol/execute".to_string(),
    description: Some("Execute protocol by ID".into()),
    input_schema: protocol_execution_schema(),
}
```

**Benefits:**
- LLM can create custom reasoning chains on-the-fly
- Protocols are first-class entities (can be saved, versioned, shared)
- Supports conditional branching (skip steps if confidence high)
- Enables protocol nesting (protocols reference other protocols)

**Implementation Consideration:**
```rust
// Add to ExecutionTrace
pub struct ExecutionTrace {
    pub id: Uuid,
    pub protocol_id: Option<String>,  // Reference to protocol used
    pub steps: Vec<StepTrace>,
    pub total_time_ms: u64,
    pub status: TraceStatus,
}
```

---

## 3. Performance Optimization

### 3.1 Caching Strategies

**Challenge**: Embedding generation and vector search are expensive operations

**Solution: Multi-Layer Caching with Moka**

**Dependency:**
```toml
[dependencies]
moka = { version = "0.12", features = ["future"] }
```

**Layer 1: Embedding Cache** (Hot path, highest hit rate)

```rust
use moka::future::Cache;
use std::hash::{Hash, Hasher};

pub struct EmbeddingCache {
    cache: Cache<QueryKey, Vec<f32>>,
}

#[derive(Clone, Eq, PartialEq)]
pub struct QueryKey {
    text: String,
    model: String,  // "nomic-embed-text-v1.5"
}

impl Hash for QueryKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.text.hash(state);
        self.model.hash(state);
    }
}

impl EmbeddingCache {
    pub fn new() -> Self {
        Self {
            cache: Cache::builder()
                .max_capacity(10_000)           // 10k embeddings
                .time_to_live(Duration::from_secs(3600))  // 1 hour TTL
                .build(),
        }
    }

    pub async fn get_or_compute(
        &self,
        text: &str,
        model: &str,
        compute_fn: impl Future<Output = Vec<f32>>,
    ) -> Vec<f32> {
        let key = QueryKey {
            text: text.to_string(),
            model: model.to_string(),
        };

        self.cache.try_get_with(key, async move {
            Ok::<_, std::convert::Infallible>(compute_fn.await)
        }).await.unwrap()
    }
}
```

**Layer 2: Search Result Cache** (Query-to-results)

```rust
pub struct SearchResultCache {
    cache: Cache<SearchKey, Vec<ScoredDocument>>,
}

#[derive(Clone, Eq, PartialEq, Hash)]
pub struct SearchKey {
    query: String,
    top_k: usize,
    filters: Vec<String>,  // Sorted for cache hits
}

impl SearchResultCache {
    pub fn new() -> Self {
        Self {
            cache: Cache::builder()
                .max_capacity(1_000)
                .time_to_live(Duration::from_secs(600))  // 10 min TTL (shorter than embeddings)
                .build(),
        }
    }
}
```

**Layer 3: Protocol Execution Cache** (Full reasoning chains)

```rust
pub struct ProtocolCache {
    cache: Cache<ProtocolCacheKey, ProtocolResult>,
}

#[derive(Clone, Eq, PartialEq, Hash)]
pub struct ProtocolCacheKey {
    protocol_id: String,
    input_hash: u64,  // Hash of input to detect duplicates
}

impl ProtocolCache {
    pub fn new() -> Self {
        Self {
            cache: Cache::builder()
                .max_capacity(100)              // Fewer entries (large payloads)
                .time_to_live(Duration::from_secs(300))  // 5 min TTL (reasoning can change)
                .build(),
        }
    }
}
```

**Cache Invalidation Strategy:**
```rust
pub enum CacheInvalidation {
    Time(Duration),           // TTL-based (default)
    Event(EventType),         // On document update/delete
    Manual(Vec<String>),      // Explicit key invalidation
}

pub async fn on_document_updated(&self, doc_id: &str) {
    // Invalidate search results that might include this document
    self.search_cache.invalidate_all().await;
    // Embedding cache remains valid (document content embeddings unchanged)
}
```

**Performance Target:**
- Cache hit: < 1ms
- Cache miss (embedding generation): < 50ms (local model) or < 200ms (API call)
- Full reasoning chain: < 5000ms

**Monitoring:**
```rust
pub struct CacheMetrics {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub evictions: AtomicU64,
    pub size: AtomicUsize,
}

impl EmbeddingCache {
    pub async fn metrics(&self) -> CacheMetrics {
        CacheMetrics {
            hits: AtomicU64::new(self.cache.entry_count()),
            // ... populate from cache stats
        }
    }
}
```

**References:**
- [Moka High-Performance Cache](https://github.com/moka-rs/moka)
- [MCP Caching Best Practices](https://gist.github.com/eonist/16f74dea1e0110cee3ef6caff2a5856c)
- [Advanced MCP Performance Techniques](https://superagi.com/top-10-advanced-techniques-for-optimizing-mcp-server-performance-in-2025/)

### 3.2 Lazy Initialization

**Challenge**: Starting MCP server should be fast; defer expensive operations until needed

**Pattern: `LazyLock` + `OnceCell` (Rust 1.80+)**

```rust
use std::sync::LazyLock;

// Static lazy initialization (shared across all requests)
static EMBEDDING_MODEL: LazyLock<EmbeddingModel> = LazyLock::new(|| {
    tracing::info!("Loading embedding model (first request)");
    EmbeddingModel::load("nomic-embed-text-v1.5")
        .expect("Failed to load embedding model")
});

static QDRANT_CLIENT: LazyLock<QdrantClient> = LazyLock::new(|| {
    tracing::info!("Connecting to Qdrant (first request)");
    QdrantClient::from_url("http://localhost:6334")
        .build()
        .expect("Failed to connect to Qdrant")
});

// Per-instance lazy initialization (when global state inappropriate)
pub struct ReasoningServer {
    protocol_registry: OnceCell<ProtocolRegistry>,
}

impl ReasoningServer {
    pub async fn get_protocol_registry(&self) -> &ProtocolRegistry {
        self.protocol_registry.get_or_init(|| async {
            tracing::info!("Loading protocol registry");
            ProtocolRegistry::load_from_disk("protocols/").await
                .expect("Failed to load protocols")
        }).await
    }
}
```

**Initialization Order:**
1. **Server startup**: Minimal (bind transport, register tools) - target <100ms
2. **First tool call**: Load embedding model, connect to Qdrant - one-time cost
3. **Subsequent calls**: Use cached connections/models - fast path

**Benefits:**
- Faster server startup (important for Claude Desktop integration)
- Resources only loaded if actually used
- Memory efficiency (don't load unused models)

**Trade-off:**
- First request has higher latency (acceptable for reasoning tasks)
- Need proper error handling (can't panic in lazy init)

**Error Handling Pattern:**
```rust
pub async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
    let model = self.embedding_model.get_or_try_init(|| async {
        EmbeddingModel::load("nomic-embed-text-v1.5")
            .await
            .map_err(|e| Error::initialization(format!("Failed to load model: {}", e)))
    }).await?;

    model.embed(text).await
}
```

**References:**
- [Rust Lazy Initialization Guide](https://blog.logrocket.com/how-use-lazy-initialization-pattern-rust-1-80/)
- [Lazy Static Best Practices](https://www.somethingsblog.com/2024/11/02/mastering-lazy-static-in-rust-a-guide-to-thread-safe-global-variables-and-shared-constant-data/)

### 3.3 Connection Pooling

**Challenge**: MCP tools may need to connect to external services (Qdrant, LLM APIs, databases)

**Pattern: Shared Connection Pool with `bb8`**

```toml
[dependencies]
bb8 = "0.8"
```

**Qdrant Connection Pool:**
```rust
use bb8::Pool;
use std::sync::Arc;

pub struct QdrantPool {
    pool: Pool<QdrantConnectionManager>,
}

impl QdrantPool {
    pub async fn new(url: &str, pool_size: u32) -> Result<Self> {
        let manager = QdrantConnectionManager::new(url);
        let pool = Pool::builder()
            .max_size(pool_size)
            .connection_timeout(Duration::from_secs(5))
            .idle_timeout(Some(Duration::from_secs(300)))  // 5 min idle
            .build(manager)
            .await?;

        Ok(Self { pool })
    }

    pub async fn search(&self, query: SearchQuery) -> Result<Vec<ScoredDocument>> {
        let conn = self.pool.get().await
            .map_err(|e| Error::network(format!("Pool exhausted: {}", e)))?;

        conn.search(query).await
    }
}
```

**LLM API Connection Pool:**
```rust
pub struct LlmClientPool {
    pool: Pool<HttpClientManager>,
}

impl LlmClientPool {
    pub async fn new(api_key: String) -> Result<Self> {
        let manager = HttpClientManager::new(api_key);
        let pool = Pool::builder()
            .max_size(10)  // Max concurrent API calls
            .build(manager)
            .await?;

        Ok(Self { pool })
    }

    pub async fn generate(&self, prompt: &str) -> Result<String> {
        let client = self.pool.get().await?;
        client.generate(prompt).await
    }
}
```

**Integration with MCP Server:**
```rust
pub struct ReasoningServer {
    qdrant_pool: Arc<QdrantPool>,
    llm_pool: Arc<LlmClientPool>,
    embedding_cache: Arc<EmbeddingCache>,
}

impl McpServerTrait for ReasoningServer {
    async fn call_tool(&self, name: &str, args: Value) -> Result<ToolResult> {
        match name {
            "rag/search" => {
                let query: SearchQuery = serde_json::from_value(args)?;
                let results = self.qdrant_pool.search(query).await?;
                Ok(ToolResult::from_documents(results))
            }
            "reasoning/gigathink" => {
                let input: ThinkToolInput = serde_json::from_value(args)?;
                let response = self.llm_pool.generate(&input.prompt).await?;
                Ok(ToolResult::from_text(response))
            }
            _ => Err(Error::NotFound("Unknown tool".into())),
        }
    }
}
```

**Pool Sizing Guidelines:**
- **Qdrant**: 5-10 connections (I/O bound, keep low to avoid overwhelming Qdrant)
- **LLM APIs**: 3-5 connections (rate-limited by provider, respect limits)
- **Local models**: 1-2 connections (CPU/GPU bound, more connections don't help)

**Monitoring:**
```rust
pub struct PoolMetrics {
    pub active_connections: usize,
    pub idle_connections: usize,
    pub wait_count: u64,
    pub timeouts: u64,
}

impl QdrantPool {
    pub fn metrics(&self) -> PoolMetrics {
        PoolMetrics {
            active_connections: self.pool.state().connections - self.pool.state().idle_connections,
            idle_connections: self.pool.state().idle_connections,
            // ... collect from pool stats
        }
    }
}
```

### 3.4 Timeout Management

**Best Practice: Cascading Timeouts** (Each layer has shorter timeout than parent)

```rust
pub struct TimeoutConfig {
    pub tool_call: Duration,          // 30s (MCP tool call timeout)
    pub llm_generation: Duration,     // 25s (LLM API call)
    pub vector_search: Duration,      // 5s  (Qdrant search)
    pub embedding: Duration,          // 10s (Embedding generation)
    pub health_check: Duration,       // 5s  (Ping timeout)
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            tool_call: Duration::from_secs(30),
            llm_generation: Duration::from_secs(25),
            vector_search: Duration::from_secs(5),
            embedding: Duration::from_secs(10),
            health_check: Duration::from_secs(5),
        }
    }
}
```

**Implementation with `tokio::time::timeout`:**
```rust
pub async fn search_with_timeout(&self, query: SearchQuery) -> Result<Vec<ScoredDocument>> {
    tokio::time::timeout(
        self.config.vector_search,
        self.qdrant_pool.search(query)
    )
    .await
    .map_err(|_| Error::timeout("Vector search timed out"))?
}

pub async fn generate_with_timeout(&self, prompt: &str) -> Result<String> {
    tokio::time::timeout(
        self.config.llm_generation,
        self.llm_pool.generate(prompt)
    )
    .await
    .map_err(|_| Error::timeout("LLM generation timed out"))?
}
```

**Graceful Degradation:**
```rust
pub async fn search_with_fallback(&self, query: SearchQuery) -> Result<Vec<ScoredDocument>> {
    // Try vector search first
    match tokio::time::timeout(
        Duration::from_secs(5),
        self.vector_search(query.clone())
    ).await {
        Ok(Ok(results)) => Ok(results),
        _ => {
            // Fallback to BM25 (faster, less accurate)
            tracing::warn!("Vector search timed out, falling back to BM25");
            self.bm25_search(query).await
        }
    }
}
```

**References:**
- [Rust Async Timeout Patterns](https://tokio.rs/tokio/topics/time)

---

## 4. Security Considerations

### 4.1 Input Sanitization

**Threat Model:**
- **Command Injection**: Malicious input escapes to shell commands
- **Path Traversal**: User-controlled file paths access unauthorized files
- **SQL Injection**: Unsanitized input in database queries
- **Prompt Injection**: Malicious prompts manipulate LLM behavior
- **DoS**: Extremely large inputs exhaust resources

**Defense Patterns:**

**1. JSON Schema Validation (First Line of Defense)**

```rust
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Validate)]
pub struct SearchQuery {
    #[validate(length(min = 1, max = 10000))]
    pub query: String,

    #[validate(range(min = 1, max = 100))]
    pub top_k: usize,

    #[validate(custom = "validate_filters")]
    pub filters: Option<Vec<String>>,
}

fn validate_filters(filters: &[String]) -> Result<(), validator::ValidationError> {
    if filters.len() > 10 {
        return Err(validator::ValidationError::new("too_many_filters"));
    }
    // Check for injection patterns
    for filter in filters {
        if filter.contains("';") || filter.contains("--") {
            return Err(validator::ValidationError::new("invalid_filter"));
        }
    }
    Ok(())
}
```

**2. Path Validation (Prevent Traversal)**

```rust
use std::path::{Path, PathBuf};

pub struct SafePathValidator {
    allowed_root: PathBuf,
}

impl SafePathValidator {
    pub fn validate(&self, user_path: &str) -> Result<PathBuf> {
        let requested = Path::new(user_path);

        // Reject absolute paths
        if requested.is_absolute() {
            return Err(Error::validation("Absolute paths not allowed"));
        }

        // Reject path traversal
        if user_path.contains("..") {
            return Err(Error::validation("Path traversal not allowed"));
        }

        // Canonicalize and verify under allowed root
        let full_path = self.allowed_root.join(requested)
            .canonicalize()
            .map_err(|_| Error::validation("Invalid path"))?;

        if !full_path.starts_with(&self.allowed_root) {
            return Err(Error::validation("Path outside allowed directory"));
        }

        Ok(full_path)
    }
}
```

**3. Command Injection Prevention**

```rust
use tokio::process::Command;

// BAD: Never do this
async fn bad_execute(user_input: &str) -> Result<String> {
    let output = Command::new("sh")
        .arg("-c")
        .arg(format!("echo {}", user_input))  // VULNERABLE!
        .output()
        .await?;
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

// GOOD: Use array of args, no shell interpolation
async fn safe_execute(user_input: &str) -> Result<String> {
    // Validate input first
    if !user_input.chars().all(|c| c.is_alphanumeric() || c.is_whitespace()) {
        return Err(Error::validation("Invalid characters in input"));
    }

    let output = Command::new("echo")
        .arg(user_input)  // Passed as argument, not interpolated
        .output()
        .await?;
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}
```

**4. Prompt Injection Mitigation**

```rust
pub struct PromptSanitizer;

impl PromptSanitizer {
    pub fn sanitize(user_query: &str) -> String {
        // Remove system prompt markers
        let cleaned = user_query
            .replace("<|system|>", "")
            .replace("<|assistant|>", "")
            .replace("IGNORE PREVIOUS INSTRUCTIONS", "");

        // Limit length
        if cleaned.len() > 5000 {
            cleaned[..5000].to_string()
        } else {
            cleaned
        }
    }

    pub fn detect_injection(query: &str) -> bool {
        let suspicious_patterns = [
            "ignore previous",
            "disregard all",
            "new instructions",
            "system:",
            "assistant:",
        ];

        let lower = query.to_lowercase();
        suspicious_patterns.iter().any(|p| lower.contains(p))
    }
}

// Usage in tool
pub async fn call_tool(&self, name: &str, args: Value) -> Result<ToolResult> {
    let query: String = serde_json::from_value(args["query"].clone())?;

    if PromptSanitizer::detect_injection(&query) {
        tracing::warn!("Potential prompt injection detected: {}", query);
        return Err(Error::validation("Suspicious input detected"));
    }

    let sanitized = PromptSanitizer::sanitize(&query);
    // ... proceed with sanitized input
}
```

**References:**
- [MCP Security Best Practices](https://modelcontextprotocol.io/specification/draft/basic/security_best_practices)
- [Input Validation Guide - Writer](https://writer.com/engineering/mcp-security-considerations/)
- [Red Hat MCP Security Controls](https://www.redhat.com/en/blog/model-context-protocol-mcp-understanding-security-risks-and-controls)

### 4.2 Resource Limits

**Goal**: Prevent DoS attacks and resource exhaustion

**Memory Limits (Per Request):**

```rust
use std::sync::Arc;
use tokio::sync::Semaphore;

pub struct ResourceLimiter {
    max_concurrent_requests: Arc<Semaphore>,
    max_result_size_bytes: usize,
    max_search_results: usize,
}

impl ResourceLimiter {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            max_concurrent_requests: Arc::new(Semaphore::new(max_concurrent)),
            max_result_size_bytes: 10 * 1024 * 1024,  // 10 MB
            max_search_results: 100,
        }
    }

    pub async fn execute_with_limit<F, T>(&self, f: F) -> Result<T>
    where
        F: Future<Output = Result<T>>,
    {
        let _permit = self.max_concurrent_requests.acquire().await
            .map_err(|_| Error::resource_exhausted("Too many concurrent requests"))?;

        f.await
    }

    pub fn validate_result_size(&self, size: usize) -> Result<()> {
        if size > self.max_result_size_bytes {
            return Err(Error::resource_exhausted(
                format!("Result too large: {} bytes", size)
            ));
        }
        Ok(())
    }
}
```

**Rate Limiting (Per Client/IP):**

```rust
use std::collections::HashMap;
use tokio::sync::RwLock;
use std::time::Instant;

pub struct RateLimiter {
    requests: RwLock<HashMap<String, Vec<Instant>>>,
    max_requests_per_minute: usize,
}

impl RateLimiter {
    pub fn new(max_requests_per_minute: usize) -> Self {
        Self {
            requests: RwLock::new(HashMap::new()),
            max_requests_per_minute,
        }
    }

    pub async fn check(&self, client_id: &str) -> Result<()> {
        let mut requests = self.requests.write().await;
        let now = Instant::now();

        let client_requests = requests.entry(client_id.to_string()).or_insert_with(Vec::new);

        // Remove requests older than 1 minute
        client_requests.retain(|&instant| now.duration_since(instant).as_secs() < 60);

        if client_requests.len() >= self.max_requests_per_minute {
            return Err(Error::rate_limited("Too many requests"));
        }

        client_requests.push(now);
        Ok(())
    }
}
```

**CPU/Time Limits:**

```rust
pub struct CpuLimiter {
    max_execution_time: Duration,
}

impl CpuLimiter {
    pub async fn execute_with_timeout<F, T>(&self, f: F) -> Result<T>
    where
        F: Future<Output = Result<T>>,
    {
        tokio::time::timeout(self.max_execution_time, f)
            .await
            .map_err(|_| Error::timeout("Execution time limit exceeded"))?
    }
}
```

**Containerized Limits (Docker/Kubernetes):**

```yaml
# docker-compose.yml
services:
  reasonkit-mcp:
    image: reasonkit-core:latest
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
```

**References:**
- [MCP Resource Limits - Datadog](https://www.datadoghq.com/blog/monitor-mcp-servers/)
- [Container Security Best Practices](https://protocolguard.com/resources/mcp-server-hardening/)

### 4.3 Sandboxing

**Threat**: Compromised tool execution could escalate to system-level access

**Defense-in-Depth Layers:**

**Layer 1: Process Isolation (AppArmor/SELinux)**

```bash
# AppArmor profile for reasonkit-mcp
# /etc/apparmor.d/usr.bin.reasonkit-mcp

#include <tunables/global>

/usr/bin/reasonkit-mcp {
  #include <abstractions/base>

  # Allow read access to config and data
  /etc/reasonkit/** r,
  /var/lib/reasonkit/data/** r,

  # Allow write to specific directories only
  /var/lib/reasonkit/sessions/** rw,
  /var/log/reasonkit/** w,

  # Deny network (if not needed)
  deny network,

  # Deny execution of other binaries
  deny /usr/bin/* x,
  deny /bin/* x,
}
```

**Layer 2: Seccomp (System Call Filtering)**

```rust
// Restrict syscalls using seccomp-bpf (requires seccompiler crate)
use seccompiler::{BpfProgram, SeccompAction, SeccompFilter};

pub fn apply_seccomp() -> Result<()> {
    let mut filter = SeccompFilter::new(
        vec![
            // Allow essential syscalls
            "read", "write", "open", "close", "stat",
            "mmap", "munmap", "brk", "futex",
            "clone", "exit", "exit_group",
        ]
        .into_iter()
        .collect(),
        SeccompAction::Allow,
        SeccompAction::Errno(libc::EPERM),  // Default: deny
    );

    filter.apply()?;
    Ok(())
}
```

**Layer 3: Containerization (Minimal Attack Surface)**

```dockerfile
# Dockerfile for reasonkit-mcp
FROM rust:1.75 AS builder
WORKDIR /build
COPY . .
RUN cargo build --release

# Minimal runtime image
FROM gcr.io/distroless/cc-debian12
COPY --from=builder /build/target/release/reasonkit-mcp /app/
USER nonroot:nonroot
ENTRYPOINT ["/app/reasonkit-mcp"]
```

**Layer 4: Network Isolation**

```yaml
# docker-compose.yml
services:
  reasonkit-mcp:
    networks:
      - internal
    # No ports exposed (stdio transport only)

  qdrant:
    networks:
      - internal
    # Only accessible from internal network

networks:
  internal:
    driver: bridge
    internal: true  # No external access
```

**Recommendations for ReasonKit:**
1. **Local deployment (Claude Desktop)**: AppArmor profile sufficient
2. **Cloud deployment**: Full containerization with seccomp + network isolation
3. **Enterprise**: Add SELinux + regular security audits

**References:**
- [MCP Sandboxing Guide](https://mcpmanager.ai/blog/sandbox-mcp-servers/)
- [WorkOS MCP Security Best Practices](https://workos.com/blog/mcp-security-risks-best-practices)
- [OWASP MCP Security Recommendations](https://www.practical-devsecops.com/mcp-security-vulnerabilities/)

### 4.4 Audit Logging

**Goal**: Comprehensive auditability for compliance and security monitoring

```rust
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLog {
    pub timestamp: DateTime<Utc>,
    pub event_type: AuditEventType,
    pub client_id: Option<String>,
    pub session_id: Option<Uuid>,
    pub tool_name: String,
    pub input_hash: String,  // Hash of input (not raw input, for privacy)
    pub result_summary: String,
    pub duration_ms: u64,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditEventType {
    ToolCall,
    ResourceAccess,
    SessionCreated,
    SessionClosed,
    HealthCheck,
    Error,
}

pub struct AuditLogger {
    log_file: tokio::fs::File,
}

impl AuditLogger {
    pub async fn log(&mut self, event: AuditLog) -> Result<()> {
        let json = serde_json::to_string(&event)?;
        self.log_file.write_all(json.as_bytes()).await?;
        self.log_file.write_all(b"\n").await?;
        self.log_file.flush().await?;
        Ok(())
    }
}
```

**Integration with MCP Server:**
```rust
impl McpServerTrait for ReasoningServer {
    async fn call_tool(&self, name: &str, args: Value) -> Result<ToolResult> {
        let start = Instant::now();
        let input_hash = hash_json(&args);

        let result = self.execute_tool_internal(name, args).await;

        let duration_ms = start.elapsed().as_millis() as u64;

        let audit_event = AuditLog {
            timestamp: Utc::now(),
            event_type: AuditEventType::ToolCall,
            client_id: None,  // Extract from request context
            session_id: None,
            tool_name: name.to_string(),
            input_hash,
            result_summary: match &result {
                Ok(r) => format!("Success: {} content items", r.content.len()),
                Err(e) => format!("Error: {}", e),
            },
            duration_ms,
            error: result.as_ref().err().map(|e| e.to_string()),
        };

        self.audit_logger.log(audit_event).await?;

        result
    }
}
```

---

## 5. Integration Patterns

### 5.1 Claude Desktop Integration

**Configuration File**: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)

```json
{
  "mcpServers": {
    "reasonkit-core": {
      "command": "/usr/local/bin/reasonkit-mcp",
      "args": ["--config", "/etc/reasonkit/config.toml"],
      "env": {
        "REASONKIT_LOG_LEVEL": "info",
        "QDRANT_URL": "http://localhost:6334"
      }
    }
  }
}
```

**Server Implementation (stdio transport):**

```rust
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

pub struct StdioServer {
    server: Arc<ReasoningServer>,
}

impl StdioServer {
    pub async fn run(&self) -> Result<()> {
        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let mut reader = BufReader::new(stdin).lines();

        tracing::info!("MCP server started (stdio mode)");

        while let Some(line) = reader.next_line().await? {
            // Parse JSON-RPC request
            let request: McpMessage = serde_json::from_str(&line)?;

            // Dispatch to handler
            let response = self.handle_message(request).await;

            // Write JSON-RPC response to stdout
            let json = serde_json::to_string(&response)?;
            stdout.write_all(json.as_bytes()).await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
        }

        Ok(())
    }

    async fn handle_message(&self, msg: McpMessage) -> McpMessage {
        match msg {
            McpMessage::Request(req) => {
                let response = self.handle_request(req).await;
                McpMessage::Response(response)
            }
            McpMessage::Notification(notif) => {
                self.handle_notification(notif).await;
                // No response for notifications
                return;
            }
            _ => {
                // Invalid message type
                McpMessage::Response(McpResponse::error(
                    RequestId::Null,
                    ErrorCode::InvalidRequest,
                    "Invalid message type",
                ))
            }
        }
    }
}
```

**Critical Rule**: **NEVER write to stdout except JSON-RPC messages**

```rust
// BAD: This breaks stdio transport
println!("Processing request...");  // Goes to stdout!

// GOOD: Use stderr for logging
eprintln!("Processing request...");

// BETTER: Use tracing (automatically goes to stderr)
tracing::info!("Processing request...");
```

**References:**
- [Claude Desktop MCP Integration](https://www.codecademy.com/article/how-to-use-model-context-protocol-mcp-with-claude-step-by-step-guide-with-examples)
- [MCP Inspector for Debugging](https://modelcontextprotocol.io/legacy/tools/debugging)

### 5.2 VS Code/Cursor Integration

**Extension Configuration** (`.vscode/settings.json`):

```json
{
  "mcp.servers": [
    {
      "name": "reasonkit-core",
      "command": "reasonkit-mcp",
      "args": ["--config", "${workspaceFolder}/.reasonkit/config.toml"],
      "env": {
        "REASONKIT_WORKSPACE": "${workspaceFolder}"
      }
    }
  ]
}
```

**Workspace-Aware Tools:**

```rust
pub struct WorkspaceContext {
    pub root_path: PathBuf,
    pub project_type: ProjectType,
}

#[derive(Debug, Clone)]
pub enum ProjectType {
    Rust,
    Python,
    JavaScript,
    Unknown,
}

impl WorkspaceContext {
    pub fn detect(root_path: PathBuf) -> Self {
        let project_type = if root_path.join("Cargo.toml").exists() {
            ProjectType::Rust
        } else if root_path.join("pyproject.toml").exists() {
            ProjectType::Python
        } else if root_path.join("package.json").exists() {
            ProjectType::JavaScript
        } else {
            ProjectType::Unknown
        };

        Self { root_path, project_type }
    }
}

// Tool uses workspace context
pub async fn code_analysis(&self, args: CodeAnalysisArgs) -> Result<ToolResult> {
    let workspace = WorkspaceContext::detect(args.workspace_path);

    // Adjust analysis based on project type
    match workspace.project_type {
        ProjectType::Rust => self.analyze_rust_code(&workspace).await,
        ProjectType::Python => self.analyze_python_code(&workspace).await,
        _ => Err(Error::validation("Unsupported project type")),
    }
}
```

### 5.3 Multi-Server Orchestration

**Problem**: ReasonKit may expose multiple MCP servers (core reasoning, web research, code analysis)

**Pattern: MCP Registry Client** (Already in `/home/zyxsys/RK-PROJECT/reasonkit-core/src/mcp/registry.rs`)

```rust
use reasonkit_core::mcp::{McpRegistry, McpClient, McpClientConfig};

pub struct MultiServerOrchestrator {
    registry: McpRegistry,
}

impl MultiServerOrchestrator {
    pub async fn new() -> Result<Self> {
        let registry = McpRegistry::new();

        // Register reasoning server
        let reasoning_config = McpClientConfig {
            name: "reasoning".to_string(),
            command: "reasonkit-mcp".to_string(),
            args: vec!["--module".into(), "reasoning".into()],
            env: HashMap::new(),
            timeout_secs: 30,
            auto_reconnect: true,
            max_retries: 3,
        };
        registry.register_client(reasoning_config).await?;

        // Register web research server
        let web_config = McpClientConfig {
            name: "web".to_string(),
            command: "reasonkit-web".to_string(),
            args: vec![],
            env: HashMap::new(),
            timeout_secs: 60,  // Longer timeout for web scraping
            auto_reconnect: true,
            max_retries: 3,
        };
        registry.register_client(web_config).await?;

        Ok(Self { registry })
    }

    pub async fn execute_composite_task(&self, query: &str) -> Result<CompositeResult> {
        // 1. Web research
        let web_results = self.registry.call_tool(
            "web",
            "research",
            json!({ "query": query, "max_sources": 10 })
        ).await?;

        // 2. Extract claims
        let claims = extract_claims(&web_results);

        // 3. Verify with ProofGuard
        let verification = self.registry.call_tool(
            "reasoning",
            "proofguard",
            json!({ "claims": claims, "sources": web_results })
        ).await?;

        Ok(CompositeResult {
            raw_research: web_results,
            verified_claims: verification,
        })
    }
}
```

**Load Balancing:**

```rust
pub struct LoadBalancedRegistry {
    servers: Vec<Arc<McpClient>>,
    strategy: LoadBalanceStrategy,
}

pub enum LoadBalanceStrategy {
    RoundRobin,
    LeastConnections,
    ResponseTime,
}

impl LoadBalancedRegistry {
    pub async fn call_tool(&self, tool: &str, args: Value) -> Result<ToolResult> {
        let server = self.select_server(&self.strategy).await;
        server.call_tool(tool, args).await
    }

    async fn select_server(&self, strategy: &LoadBalanceStrategy) -> Arc<McpClient> {
        match strategy {
            LoadBalanceStrategy::RoundRobin => {
                // Simple round-robin
                let idx = self.next_index.fetch_add(1, Ordering::Relaxed) % self.servers.len();
                self.servers[idx].clone()
            }
            LoadBalanceStrategy::LeastConnections => {
                // Select server with fewest active requests
                self.servers.iter()
                    .min_by_key(|s| s.active_requests())
                    .unwrap()
                    .clone()
            }
            LoadBalanceStrategy::ResponseTime => {
                // Select server with lowest avg response time
                self.servers.iter()
                    .min_by(|a, b| a.avg_response_time().cmp(&b.avg_response_time()))
                    .unwrap()
                    .clone()
            }
        }
    }
}
```

**References:**
- [MCP Multi-Server Patterns](https://dev.to/techstuff/part-4-advanced-mcp-patterns-and-tool-chaining-4ll7)

### 5.4 RAG-Specific Integration Patterns

**Pattern: MCP Server as RAG Query Interface**

```rust
pub struct RagMcpServer {
    vector_store: Arc<VectorStore>,
    bm25_index: Arc<Bm25Index>,
    embedding_cache: Arc<EmbeddingCache>,
}

impl McpServerTrait for RagMcpServer {
    fn tools(&self) -> Vec<Tool> {
        vec![
            Tool {
                name: "rag/search".to_string(),
                description: Some("Hybrid search over document corpus".into()),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" },
                        "top_k": { "type": "integer", "default": 10 },
                        "strategy": {
                            "type": "string",
                            "enum": ["vector", "bm25", "hybrid", "fusion"]
                        }
                    },
                    "required": ["query"]
                }),
                server_id: None,
                server_name: None,
            },
            Tool {
                name: "rag/index".to_string(),
                description: Some("Index new documents".into()),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "documents": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": { "type": "string" },
                                    "content": { "type": "string" },
                                    "metadata": { "type": "object" }
                                }
                            }
                        }
                    },
                    "required": ["documents"]
                }),
                server_id: None,
                server_name: None,
            },
        ]
    }

    async fn call_tool(&self, name: &str, args: Value) -> Result<ToolResult> {
        match name {
            "rag/search" => self.search(args).await,
            "rag/index" => self.index(args).await,
            _ => Err(Error::NotFound("Unknown tool".into())),
        }
    }
}

impl RagMcpServer {
    async fn search(&self, args: Value) -> Result<ToolResult> {
        let query: SearchQuery = serde_json::from_value(args)?;

        // Get cached embedding or compute
        let embedding = self.embedding_cache.get_or_compute(
            &query.query,
            "nomic-embed-text-v1.5",
            self.compute_embedding(&query.query)
        ).await;

        // Execute search based on strategy
        let results = match query.strategy.as_deref() {
            Some("vector") => self.vector_store.search(embedding, query.top_k).await?,
            Some("bm25") => self.bm25_index.search(&query.query, query.top_k).await?,
            Some("hybrid") => self.hybrid_search(&query).await?,
            Some("fusion") => self.fusion_search(&query).await?,
            _ => self.hybrid_search(&query).await?,  // Default
        };

        // Return as MCP tool result
        Ok(ToolResult {
            content: results.into_iter().map(|doc| ToolResultContent::Text {
                text: serde_json::to_string(&doc).unwrap(),
            }).collect(),
            is_error: None,
        })
    }
}
```

**Resources for Document Access:**

```rust
impl McpServerTrait for RagMcpServer {
    async fn list_resources(&self) -> Result<Vec<Resource>> {
        vec![
            Resource {
                uri: "rag://documents".to_string(),
                name: "All Documents".to_string(),
                description: Some("List all indexed documents".into()),
                mime_type: Some("application/json".into()),
            },
            Resource {
                uri: "rag://documents/{id}".to_string(),
                name: "Document by ID".to_string(),
                description: Some("Fetch document by ID".into()),
                mime_type: Some("application/json".into()),
            },
            Resource {
                uri: "rag://collections/{name}".to_string(),
                name: "Collection".to_string(),
                description: Some("Documents in a specific collection".into()),
                mime_type: Some("application/json".into()),
            },
        ]
    }

    async fn read_resource(&self, uri: &str) -> Result<ResourceContent> {
        if uri == "rag://documents" {
            let docs = self.vector_store.list_all().await?;
            Ok(ResourceContent::json(docs))
        } else if let Some(id) = parse_document_uri(uri) {
            let doc = self.vector_store.get_by_id(id).await?;
            Ok(ResourceContent::json(doc))
        } else {
            Err(Error::NotFound("Resource not found".into()))
        }
    }
}
```

**Agentic RAG Pattern:**

The key insight from [True Agentic RAG with MCP](https://medium.com/@adkomyagin/true-agentic-rag-how-i-taught-claude-to-talk-to-my-pdfs-using-model-context-protocol-mcp-9b8671b00de1) is that MCP enables **autonomous query formulation** by the LLM:

```
Traditional RAG:
User Question → Fixed query → Vector search → Context → LLM

Agentic RAG with MCP:
User Question → LLM decides search strategy → Tool calls (rag/search, rag/refine, rag/filter) → LLM synthesizes answer
```

**Implementation:**

```rust
// LLM can call multiple tools to refine search
Tool {
    name: "rag/search_and_refine".to_string(),
    description: Some("Initial search, then refine based on relevance".into()),
    // ... schema
}

Tool {
    name: "rag/multi_query".to_string(),
    description: Some("Generate and execute multiple query variations".into()),
    // ... schema
}
```

**References:**
- [True Agentic RAG with MCP](https://medium.com/@adkomyagin/true-agentic-rag-how-i-taught-claude-to-talk-to-my-pdfs-using-model-context-protocol-mcp-9b8671b00de1)
- [Local RAG with Rust and MCP](https://medium.com/@ksaritek/local-rag-with-rust-and-mcp-private-document-search-for-claude-desktop-6fccb37c024e)
- [Qdrant MCP Server](https://qdrant.tech/blog/webinar-vibe-coding-rag/)

---

## 6. Developer Experience

### 6.1 Self-Describing Tools

**Goal**: Tools provide rich metadata for LLMs to understand their purpose and usage

**Pattern: Comprehensive JSON Schema with Examples**

```rust
pub fn gigathink_tool_schema() -> Value {
    json!({
        "type": "object",
        "title": "GigaThink Multi-Perspective Expansion",
        "description": "Generates 10+ diverse perspectives on a problem using structured creative thinking. Best for exploring complex topics, generating ideas, or challenging assumptions.",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question or problem to analyze from multiple angles",
                "minLength": 10,
                "maxLength": 5000,
                "examples": [
                    "Should we adopt microservices architecture?",
                    "How can we reduce customer churn?",
                    "What are the ethical implications of AI in healthcare?"
                ]
            },
            "perspectives_count": {
                "type": "integer",
                "description": "Number of perspectives to generate (default: 10)",
                "minimum": 5,
                "maximum": 20,
                "default": 10
            },
            "focus_areas": {
                "type": "array",
                "description": "Optional domains to emphasize (e.g., 'technical', 'business', 'ethical')",
                "items": {
                    "type": "string",
                    "enum": ["technical", "business", "ethical", "social", "environmental", "legal"]
                },
                "maxItems": 3
            }
        },
        "required": ["query"],
        "examples": [
            {
                "query": "Should we migrate to Rust for our backend?",
                "perspectives_count": 10,
                "focus_areas": ["technical", "business"]
            }
        ]
    })
}
```

**Key Elements for Self-Description:**
1. **Clear purpose**: What does this tool do?
2. **When to use**: Under what circumstances should LLM call this tool?
3. **Input constraints**: Min/max lengths, valid ranges
4. **Concrete examples**: Show typical usage patterns
5. **Output format**: What to expect in the response

**Validation Errors as Learning:**

```rust
impl Tool {
    pub fn validate_input(&self, args: &Value) -> Result<()> {
        let schema = self.input_schema.clone();

        // Use jsonschema crate for validation
        let compiled = jsonschema::JSONSchema::compile(&schema)
            .map_err(|e| Error::validation(format!("Invalid schema: {}", e)))?;

        if let Err(errors) = compiled.validate(args) {
            let error_messages: Vec<String> = errors
                .map(|e| format!("{} at {}", e, e.instance_path))
                .collect();

            return Err(Error::validation(format!(
                "Input validation failed:\n{}",
                error_messages.join("\n")
            )));
        }

        Ok(())
    }
}
```

**Helpful Error Messages:**

```json
{
  "error": {
    "code": -32602,
    "message": "Invalid parameters",
    "data": {
      "validation_errors": [
        "query: String too short (minimum 10 characters)",
        "perspectives_count: Value 25 exceeds maximum of 20"
      ],
      "suggestion": "Try reducing perspectives_count to 20 or less. Query should be a complete sentence.",
      "example": {
        "query": "Should we adopt microservices architecture?",
        "perspectives_count": 10
      }
    }
  }
}
```

**References:**
- [Speakeasy Self-Describing Tools](https://www.speakeasy.com/blog/streamlined-sdk-testing-ai-ready-apis-with-mcp-server-generation)

### 6.2 Documentation Generation

**Pattern: Auto-Generate Docs from Tool Definitions**

```rust
pub struct DocGenerator;

impl DocGenerator {
    pub fn generate_markdown(tools: &[Tool]) -> String {
        let mut doc = String::from("# ReasonKit MCP Tools\n\n");

        for tool in tools {
            doc.push_str(&format!("## {}\n\n", tool.name));

            if let Some(desc) = &tool.description {
                doc.push_str(&format!("{}\n\n", desc));
            }

            doc.push_str("### Input Schema\n\n");
            doc.push_str("```json\n");
            doc.push_str(&serde_json::to_string_pretty(&tool.input_schema).unwrap());
            doc.push_str("\n```\n\n");

            if let Some(examples) = tool.input_schema.get("examples") {
                doc.push_str("### Examples\n\n");
                doc.push_str("```json\n");
                doc.push_str(&serde_json::to_string_pretty(examples).unwrap());
                doc.push_str("\n```\n\n");
            }
        }

        doc
    }

    pub fn generate_openapi(tools: &[Tool]) -> Value {
        json!({
            "openapi": "3.1.0",
            "info": {
                "title": "ReasonKit MCP Server",
                "version": "1.0.0"
            },
            "paths": tools.iter().map(|tool| {
                (format!("/tools/{}", tool.name), json!({
                    "post": {
                        "summary": tool.description,
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": tool.input_schema
                                }
                            }
                        }
                    }
                }))
            }).collect::<serde_json::Map<String, Value>>()
        })
    }
}
```

**Auto-Generated Documentation on Server Startup:**

```rust
impl ReasoningServer {
    pub async fn new() -> Result<Self> {
        let server = Self {
            // ... initialization
        };

        // Generate documentation
        let tools = server.list_tools().await?;
        let markdown_docs = DocGenerator::generate_markdown(&tools);
        let openapi_spec = DocGenerator::generate_openapi(&tools);

        // Write to files
        tokio::fs::write("docs/MCP_TOOLS.md", markdown_docs).await?;
        tokio::fs::write("docs/openapi.json", serde_json::to_string_pretty(&openapi_spec)?).await?;

        tracing::info!("Documentation generated at docs/");

        Ok(server)
    }
}
```

**References:**
- [AWS Code Documentation MCP Server](https://awslabs.github.io/mcp/servers/code-doc-gen-mcp-server)
- [Mintlify MCP Documentation Guide](https://www.mintlify.com/blog/how-to-use-mcp-servers-to-generate-docs)

### 6.3 Testing Strategies

**Level 1: Unit Tests (Individual Tools)**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gigathink_tool() {
        let server = create_test_server().await;

        let input = json!({
            "query": "Should we use Rust for backend?",
            "perspectives_count": 10
        });

        let result = server.call_tool("gigathink", input).await;
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(!output.content.is_empty());

        // Validate output structure
        let text = match &output.content[0] {
            ToolResultContent::Text { text } => text,
            _ => panic!("Expected text content"),
        };

        let parsed: GigaThinkOutput = serde_json::from_str(text).unwrap();
        assert_eq!(parsed.perspectives.len(), 10);
    }

    #[tokio::test]
    async fn test_gigathink_validation() {
        let server = create_test_server().await;

        let invalid_input = json!({
            "query": "too short",  // Less than 10 chars
            "perspectives_count": 25  // Exceeds max
        });

        let result = server.call_tool("gigathink", invalid_input).await;
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(error.to_string().contains("validation"));
    }
}
```

**Level 2: Integration Tests (Tool Chaining)**

```rust
#[tokio::test]
async fn test_reasoning_chain() {
    let server = create_test_server().await;

    // Step 1: GigaThink
    let gigathink_result = server.call_tool("gigathink", json!({
        "query": "Should we adopt microservices?"
    })).await.unwrap();

    let perspectives = extract_perspectives(&gigathink_result);

    // Step 2: LaserLogic (uses GigaThink output)
    let laserlogic_result = server.call_tool("laserlogic", json!({
        "argument": perspectives[0],
        "validate": true
    })).await.unwrap();

    let logical_analysis = extract_analysis(&laserlogic_result);
    assert!(logical_analysis.contains_key("fallacies"));
}
```

**Level 3: MCP Protocol Compliance Tests**

```rust
#[tokio::test]
async fn test_mcp_protocol_compliance() {
    let server = create_test_server().await;

    // Test initialize
    let init_result = server.initialize(json!({
        "protocolVersion": "2025-11-25",
        "capabilities": {},
        "clientInfo": {
            "name": "test-client",
            "version": "1.0.0"
        }
    })).await;
    assert!(init_result.is_ok());

    // Test tools/list
    let tools = server.list_tools().await.unwrap();
    assert!(!tools.is_empty());

    for tool in &tools {
        // Validate schema structure
        assert!(tool.input_schema.is_object());
        assert!(tool.input_schema.get("type").is_some());
    }

    // Test ping
    let ping_result = server.ping().await;
    assert!(ping_result.is_ok());

    // Test shutdown
    let shutdown_result = server.shutdown().await;
    assert!(shutdown_result.is_ok());
}
```

**Level 4: Mock LLM Integration Tests**

```rust
use mockito::Server as MockServer;

#[tokio::test]
async fn test_with_mock_llm() {
    let mut mock_server = MockServer::new_async().await;

    let mock = mock_server.mock("POST", "/v1/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(json!({
            "choices": [{
                "message": {
                    "content": "Mocked LLM response for testing"
                }
            }]
        }).to_string())
        .create_async()
        .await;

    let server = create_test_server_with_llm_url(&mock_server.url()).await;

    let result = server.call_tool("gigathink", json!({
        "query": "Test query"
    })).await;

    assert!(result.is_ok());
    mock.assert_async().await;
}
```

**Level 5: End-to-End Tests with MCP Inspector**

```bash
#!/bin/bash
# test_mcp_server.sh

# Start server in background
./target/release/reasonkit-mcp &
SERVER_PID=$!

# Wait for server to start
sleep 2

# Run MCP Inspector tests
npx @modelcontextprotocol/inspector \
  --command "./target/release/reasonkit-mcp" \
  --test-suite ./tests/mcp_test_suite.json

# Capture exit code
TEST_EXIT=$?

# Cleanup
kill $SERVER_PID

exit $TEST_EXIT
```

**References:**
- [MCP Testing Best Practices](https://testomat.io/blog/mcp-server-testing-tools/)
- [Top MCP Testing Tools 2025](https://testguild.com/top-model-context-protocols-mcp/)
- [MCP Automated Testing Guide](https://www.byteplus.com/en/topic/541524)

### 6.4 Debugging Support

**Pattern: Rich Error Context + Logging**

```rust
use tracing::{error, warn, info, debug, instrument};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

pub fn init_tracing() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("REASONKIT_LOG_LEVEL").unwrap_or_else(|_| "info".into())
        ))
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr))
        .init();
}

#[instrument(skip(self, args), fields(tool_name = %name))]
pub async fn call_tool(&self, name: &str, args: Value) -> Result<ToolResult> {
    debug!("Tool call received: {} with args: {:?}", name, args);

    // Validate input
    let tool = self.tools.get(name)
        .ok_or_else(|| {
            warn!("Tool not found: {}", name);
            Error::NotFound(format!("Tool '{}' not found", name))
        })?;

    if let Err(e) = tool.validate_input(&args) {
        error!("Validation failed for {}: {}", name, e);
        return Err(e);
    }

    // Execute tool
    let start = std::time::Instant::now();
    let result = self.execute_tool_internal(name, args).await;
    let duration_ms = start.elapsed().as_millis();

    match &result {
        Ok(_) => info!("Tool {} succeeded in {}ms", name, duration_ms),
        Err(e) => error!("Tool {} failed in {}ms: {}", name, duration_ms, e),
    }

    result
}
```

**Error Code System:**

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    // Standard JSON-RPC errors
    ParseError = -32700,
    InvalidRequest = -32600,
    MethodNotFound = -32601,
    InvalidParams = -32602,
    InternalError = -32603,

    // MCP-specific errors
    ServerNotInitialized = -32001,
    RequestTimeout = -32002,
    ConnectionClosed = -32000,

    // Application errors
    ValidationError = -1000,
    ResourceExhausted = -1001,
    RateLimited = -1002,
    Unauthorized = -1003,
}

impl ErrorCode {
    pub fn to_json_rpc_error(self, message: impl Into<String>) -> McpError {
        McpError {
            code: self as i32,
            message: message.into(),
            data: None,
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::ParseError => "Invalid JSON was received",
            Self::InvalidRequest => "The JSON sent is not a valid Request object",
            Self::MethodNotFound => "The method does not exist or is not available",
            Self::InvalidParams => "Invalid method parameters",
            Self::InternalError => "Internal server error",
            Self::ServerNotInitialized => "Server has not been initialized",
            Self::RequestTimeout => "Request timed out",
            Self::ConnectionClosed => "Connection closed unexpectedly",
            Self::ValidationError => "Input validation failed",
            Self::ResourceExhausted => "Server resource limit exceeded",
            Self::RateLimited => "Rate limit exceeded",
            Self::Unauthorized => "Unauthorized access",
        }
    }
}
```

**Structured Error Responses:**

```rust
pub struct RichError {
    pub code: ErrorCode,
    pub message: String,
    pub details: ErrorDetails,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetails {
    pub tool_name: Option<String>,
    pub validation_errors: Vec<String>,
    pub suggestion: Option<String>,
    pub documentation_link: Option<String>,
    pub trace_id: Uuid,
}

impl From<RichError> for McpError {
    fn from(err: RichError) -> Self {
        McpError {
            code: err.code as i32,
            message: err.message,
            data: Some(serde_json::to_value(err.details).unwrap()),
        }
    }
}
```

**Debugging Endpoints (Development Mode):**

```rust
#[cfg(debug_assertions)]
impl ReasoningServer {
    pub fn enable_debug_tools(&mut self) {
        self.tools.push(Tool {
            name: "debug/inspect_session".to_string(),
            description: Some("Inspect session state (debug only)".into()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id": { "type": "string" }
                },
                "required": ["session_id"]
            }),
            server_id: None,
            server_name: None,
        });

        self.tools.push(Tool {
            name: "debug/clear_cache".to_string(),
            description: Some("Clear all caches (debug only)".into()),
            input_schema: json!({ "type": "object" }),
            server_id: None,
            server_name: None,
        });
    }
}
```

**References:**
- [MCP Debugging Best Practices](https://www.mcpevals.io/blog/debugging-mcp-servers-tips-and-best-practices)
- [Error Handling in MCP Servers](https://mcpcat.io/guides/error-handling-custom-mcp-servers/)
- [MCP Error Codes Reference](https://www.mcpevals.io/blog/mcp-error-codes)

---

## 7. Recommended Implementation Roadmap for ReasonKit

### Phase 1: Core MCP Server (Week 1-2)

**Deliverables:**
1. ✅ Basic MCP server with stdio transport (already exists in `/src/mcp/`)
2. Expose 5 ThinkTools as atomic tools (gigathink, laserlogic, bedrock, proofguard, brutalhonesty)
3. Add 3 workflow tools (profile/quick, profile/balanced, profile/paranoid)
4. Implement tool schema with validation
5. Claude Desktop integration

**Code Changes:**
```rust
// src/mcp/reasonkit_server.rs
pub struct ReasonKitMcpServer {
    executor: Arc<ProtocolExecutor>,
    profile_registry: Arc<ProfileRegistry>,
}

impl McpServerTrait for ReasonKitMcpServer {
    fn tools(&self) -> Vec<Tool> {
        vec![
            // Atomic tools
            create_thinktool_schema("gigathink", "Multi-perspective expansion"),
            create_thinktool_schema("laserlogic", "Logical validation"),
            create_thinktool_schema("bedrock", "First principles decomposition"),
            create_thinktool_schema("proofguard", "Evidence verification"),
            create_thinktool_schema("brutalhonesty", "Adversarial critique"),

            // Workflow tools
            create_profile_schema("profile/quick", "Fast 3-step analysis"),
            create_profile_schema("profile/balanced", "Standard 5-module chain"),
            create_profile_schema("profile/paranoid", "Maximum verification"),
        ]
    }
}
```

### Phase 2: Session Management (Week 3)

**Deliverables:**
1. Session store with DashMap
2. Session lifecycle tools (create, step, close)
3. TTL-based cleanup
4. Audit logging

**Code Changes:**
```rust
// src/mcp/session.rs
pub struct SessionStore {
    sessions: DashMap<Uuid, ReasoningSession>,
    cleanup_task: Option<tokio::task::JoinHandle<()>>,
}

// Add tools
Tool::new("session/create", "Create reasoning session")
Tool::new("session/step", "Execute next reasoning step")
Tool::new("session/close", "Close and persist session")
```

### Phase 3: Performance Optimization (Week 4)

**Deliverables:**
1. Embedding cache with Moka
2. Lazy initialization of models
3. Connection pooling for Qdrant
4. Benchmarking suite

**Dependencies to add:**
```toml
[dependencies]
moka = { version = "0.12", features = ["future"] }
bb8 = "0.8"
```

### Phase 4: RAG Integration (Week 5)

**Deliverables:**
1. RAG tools (rag/search, rag/index)
2. Resources for document access
3. Hybrid search (vector + BM25)
4. Reranking support

**Tools:**
```rust
Tool::new("rag/search", "Hybrid search over document corpus")
Tool::new("rag/index", "Index new documents")
Tool::new("rag/rerank", "Rerank search results")
```

### Phase 5: Security Hardening (Week 6)

**Deliverables:**
1. Input sanitization for all tools
2. Rate limiting
3. Resource limits
4. AppArmor profile
5. Security audit

### Phase 6: Testing & Documentation (Week 7)

**Deliverables:**
1. Unit tests (80%+ coverage)
2. Integration tests
3. MCP Inspector test suite
4. Auto-generated documentation
5. Usage examples

---

## 8. Key Takeaways

### Tool Design
1. **Hybrid granularity**: Expose both atomic tools and workflow tools
2. **Session-based state**: Use DashMap for server-side session management
3. **Self-describing schemas**: Rich JSON Schema with examples and validation

### Performance
1. **Cache aggressively**: Moka for embeddings (3600s TTL), search results (600s TTL)
2. **Lazy initialization**: Load models on first request, not server startup
3. **Connection pooling**: 5-10 Qdrant connections, 3-5 LLM API connections

### Security
1. **Multi-layer defense**: JSON Schema validation + input sanitization + resource limits + sandboxing
2. **Audit everything**: Comprehensive logging for compliance
3. **Fail safely**: Graceful degradation on timeout/failure

### Developer Experience
1. **Self-describing tools**: Clear descriptions, examples, validation errors
2. **Auto-generate docs**: Markdown + OpenAPI from tool definitions
3. **Rich debugging**: Structured errors with trace IDs, suggestions, documentation links

### Integration
1. **stdio for Claude Desktop**: Critical logging rule (stderr only!)
2. **Resources for data access**: Expose documents/traces via MCP resources
3. **Agentic RAG**: Let LLM decide search strategy via multiple tools

---

## 9. References

### Official Specifications
- [MCP Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25)
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)

### Rust Implementation
- [Official Rust MCP SDK](https://github.com/modelcontextprotocol/rust-sdk)
- [Building MCP Servers in Rust - MCPcat](https://mcpcat.io/guides/building-mcp-server-rust/)
- [Shuttle Rust MCP Tutorials](https://www.shuttle.dev/blog/2025/07/18/how-to-build-a-stdio-mcp-server-in-rust)

### Tool Design
- [Less is More: MCP Design Patterns](https://www.klavis.ai/blog/less-is-more-mcp-design-patterns-for-ai-agents)
- [MCP Architecture Overview](https://modelcontextprotocol.io/docs/learn/architecture)

### Performance
- [Moka High-Performance Cache](https://github.com/moka-rs/moka)
- [MCP Caching Best Practices](https://gist.github.com/eonist/16f74dea1e0110cee3ef6caff2a5856c)
- [Advanced MCP Performance Techniques](https://superagi.com/top-10-advanced-techniques-for-optimizing-mcp-server-performance-in-2025/)

### Security
- [MCP Security Best Practices](https://modelcontextprotocol.io/specification/draft/basic/security_best_practices)
- [Writer MCP Security Guide](https://writer.com/engineering/mcp-security-considerations/)
- [Red Hat MCP Security Controls](https://www.redhat.com/en/blog/model-context-protocol-mcp-understanding-security-risks-and-controls)
- [WorkOS MCP Security Best Practices](https://workos.com/blog/mcp-security-risks-best-practices)

### RAG Integration
- [True Agentic RAG with MCP](https://medium.com/@adkomyagin/true-agentic-rag-how-i-taught-claude-to-talk-to-my-pdfs-using-model-context-protocol-mcp-9b8671b00de1)
- [Local RAG with Rust and MCP](https://medium.com/@ksaritek/local-rag-with-rust-and-mcp-private-document-search-for-claude-desktop-6fccb37c024e)
- [Qdrant MCP Server](https://qdrant.tech/blog/webinar-vibe-coding-rag/)

### Testing
- [MCP Testing Best Practices](https://testomat.io/blog/mcp-server-testing-tools/)
- [Top MCP Testing Tools 2025](https://testguild.com/top-model-context-protocols-mcp/)

### Debugging
- [MCP Debugging Best Practices](https://www.mcpevals.io/blog/debugging-mcp-servers-tips-and-best-practices)
- [Error Handling in MCP Servers](https://mcpcat.io/guides/error-handling-custom-mcp-servers/)
- [MCP Error Codes Reference](https://www.mcpevals.io/blog/mcp-error-codes)

---

**Document Version**: 1.0.0
**Last Updated**: 2025-12-25
**Maintainer**: ReasonKit Core Team
**License**: Apache 2.0 (consistent with reasonkit-core)
