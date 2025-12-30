# MCP Implementation Quick Reference

> TL;DR - Essential patterns for Rust-based MCP servers exposing RAG/reasoning tools

---

## 1. Tool Design Pattern (RECOMMENDED)

### Hybrid Two-Tier Architecture

```rust
// HIGH-LEVEL (Workflow Tools) - Optimize for token efficiency
- session/create           → Create reasoning session
- session/step             → Execute one step in chain
- session/execute_chain    → Run complete chain
- profile/quick            → 3-step fast analysis (gt → ll)
- profile/balanced         → 5-step standard (gt → ll → br → pg)
- profile/paranoid         → Maximum verification (all + validation)

// LOW-LEVEL (Atomic Tools) - Optimize for flexibility
- gigathink               → Multi-perspective expansion
- laserlogic              → Logical validation
- bedrock                 → First principles decomposition
- proofguard              → Evidence verification
- brutalhonesty           → Adversarial critique
```

**Why?**

- Workflow tools reduce LLM decision overhead (1 call vs 5 calls)
- Atomic tools enable custom composition
- Best of both worlds: efficiency + flexibility

---

## 2. State Management Pattern (RECOMMENDED)

### Session-Based Stateful Execution

```rust
pub struct ReasoningSession {
    pub id: Uuid,
    pub profile: ReasoningProfile,
    pub trace: ExecutionTrace,
    pub step_outputs: HashMap<String, StepOutput>,  // Cross-step context
    pub cursor: usize,
    pub status: SessionStatus,
}

pub struct SessionStore {
    sessions: DashMap<Uuid, ReasoningSession>,  // Lock-free concurrent access
    ttl_cleanup_task: tokio::task::JoinHandle<()>,
}
```

**Benefits:**

- Server owns state (simple client integration - just pass session ID)
- Efficient for multi-step chains
- Full audit trail
- Supports resumption after failures

**Trade-off:** Requires sticky sessions (ok for single-server deployment)

---

## 3. Performance Optimization Pattern

### Three-Layer Caching with Moka

```rust
use moka::future::Cache;

// Layer 1: Embedding Cache (1 hour TTL)
cache: Cache::builder()
    .max_capacity(10_000)
    .time_to_live(Duration::from_secs(3600))
    .build()

// Layer 2: Search Result Cache (10 min TTL)
cache: Cache::builder()
    .max_capacity(1_000)
    .time_to_live(Duration::from_secs(600))
    .build()

// Layer 3: Protocol Execution Cache (5 min TTL)
cache: Cache::builder()
    .max_capacity(100)
    .time_to_live(Duration::from_secs(300))
    .build()
```

**Target Performance:**

- Cache hit: < 1ms
- Embedding generation: < 50ms (local) / < 200ms (API)
- Full reasoning chain: < 5000ms

### Lazy Initialization (Rust 1.80+)

```rust
use std::sync::LazyLock;

static EMBEDDING_MODEL: LazyLock<EmbeddingModel> = LazyLock::new(|| {
    EmbeddingModel::load("nomic-embed-text-v1.5").expect("Failed to load")
});

static QDRANT_CLIENT: LazyLock<QdrantClient> = LazyLock::new(|| {
    QdrantClient::from_url("http://localhost:6334").build().expect("Failed to connect")
});
```

**Benefits:**

- Fast server startup (< 100ms)
- Resources loaded only if needed
- First request pays one-time cost

### Connection Pooling with bb8

```rust
use bb8::Pool;

QdrantPool: Pool::builder()
    .max_size(10)  // 5-10 for Qdrant (I/O bound)
    .connection_timeout(Duration::from_secs(5))
    .idle_timeout(Some(Duration::from_secs(300)))
    .build(manager)

LlmPool: Pool::builder()
    .max_size(5)   // 3-5 for LLM APIs (rate limited)
    .build(manager)
```

---

## 4. Security Pattern (CRITICAL)

### Multi-Layer Defense

```rust
// Layer 1: JSON Schema Validation
#[derive(Validate)]
pub struct SearchQuery {
    #[validate(length(min = 1, max = 10000))]
    pub query: String,

    #[validate(range(min = 1, max = 100))]
    pub top_k: usize,
}

// Layer 2: Path Traversal Prevention
if user_path.contains("..") || user_path.starts_with("/") {
    return Err(Error::validation("Invalid path"));
}

// Layer 3: Command Injection Prevention
// NEVER use shell interpolation
Command::new("echo").arg(user_input)  // ✅ Safe
Command::new("sh").arg("-c").arg(format!("echo {}", user_input))  // ❌ VULNERABLE

// Layer 4: Prompt Injection Detection
let suspicious = ["ignore previous", "disregard all", "new instructions"];
if suspicious.iter().any(|p| query.to_lowercase().contains(p)) {
    return Err(Error::validation("Suspicious input detected"));
}
```

### Resource Limits

```rust
pub struct ResourceLimiter {
    max_concurrent_requests: Arc<Semaphore>,  // e.g., 10
    max_result_size_bytes: usize,             // e.g., 10 MB
    max_search_results: usize,                // e.g., 100
}

// Rate limiting
pub struct RateLimiter {
    max_requests_per_minute: usize,  // e.g., 60
}

// Timeout
tokio::time::timeout(Duration::from_secs(30), tool_execution).await?
```

### Sandboxing (Production)

```yaml
# docker-compose.yml
services:
  reasonkit-mcp:
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 4G
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
```

---

## 5. Integration Pattern

### Claude Desktop (stdio)

**CRITICAL RULE:** Never write to stdout except JSON-RPC messages!

```rust
// ❌ BREAKS STDIO TRANSPORT
println!("Processing...");  // Goes to stdout!

// ✅ CORRECT
eprintln!("Processing...");             // stderr
tracing::info!("Processing...");        // stderr via tracing
```

**Config:** `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "reasonkit-core": {
      "command": "/usr/local/bin/reasonkit-mcp",
      "args": ["--config", "/etc/reasonkit/config.toml"]
    }
  }
}
```

### RAG Integration

```rust
// Agentic RAG: LLM decides search strategy
Tool::new("rag/search", "Hybrid search")
Tool::new("rag/multi_query", "Multiple query variations")
Tool::new("rag/search_and_refine", "Iterative refinement")

// Resources for document access
Resource::new("rag://documents/{id}", "Document by ID")
Resource::new("rag://collections/{name}", "Collection")
```

**Pattern:** Let LLM orchestrate RAG pipeline via multiple tool calls

- Better than fixed RAG: LLM can adapt strategy per query
- Claude autonomously decides: initial search → refinement → verification

---

## 6. Developer Experience Pattern

### Self-Describing Tools

```rust
json!({
    "type": "object",
    "title": "GigaThink Multi-Perspective Expansion",
    "description": "Generates 10+ perspectives. Best for exploring complex topics.",
    "properties": {
        "query": {
            "type": "string",
            "minLength": 10,
            "maxLength": 5000,
            "examples": ["Should we adopt microservices?"]
        }
    },
    "required": ["query"]
})
```

**Include:**

- Clear purpose (what/when to use)
- Concrete examples
- Validation constraints
- Helpful error messages

### Auto-Generated Documentation

```rust
impl ReasoningServer {
    pub async fn new() -> Result<Self> {
        // Generate docs on startup
        let markdown = DocGenerator::generate_markdown(&self.tools);
        tokio::fs::write("docs/MCP_TOOLS.md", markdown).await?;
    }
}
```

### Testing Layers

1. **Unit**: Individual tool validation
2. **Integration**: Tool chaining
3. **Protocol Compliance**: MCP spec adherence
4. **E2E**: MCP Inspector test suites

---

## 7. Error Handling Pattern

### Structured Errors with Context

```rust
pub struct RichError {
    pub code: ErrorCode,
    pub message: String,
    pub details: ErrorDetails,
}

pub struct ErrorDetails {
    pub validation_errors: Vec<String>,
    pub suggestion: Option<String>,
    pub documentation_link: Option<String>,
    pub trace_id: Uuid,  // For debugging
}
```

**Example Error Response:**

```json
{
  "error": {
    "code": -32602,
    "message": "Invalid parameters",
    "data": {
      "validation_errors": ["query: String too short (minimum 10 characters)"],
      "suggestion": "Query should be a complete sentence.",
      "documentation_link": "https://reasonkit.sh/docs/tools/gigathink",
      "trace_id": "123e4567-e89b-12d3-a456-426614174000"
    }
  }
}
```

### Retry Pattern with Exponential Backoff

```rust
pub async fn execute_with_retry<F, T>(&self, f: F, max_retries: u32) -> Result<T>
where
    F: Fn() -> Future<Output = Result<T>>,
{
    let mut delay_ms = 100;
    for attempt in 0..max_retries {
        match f().await {
            Ok(result) => return Ok(result),
            Err(e) if should_retry(&e) && attempt < max_retries - 1 => {
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                delay_ms *= 2;  // Exponential backoff
            }
            Err(e) => return Err(e),
        }
    }
}
```

---

## 8. Implementation Roadmap

### Phase 1: Core MCP Server (Week 1-2)

- ✅ stdio transport (already exists)
- Expose 5 ThinkTools + 3 profiles
- Tool schema + validation
- Claude Desktop integration

### Phase 2: Session Management (Week 3)

- DashMap session store
- Session lifecycle tools
- TTL cleanup
- Audit logging

### Phase 3: Performance (Week 4)

- Moka caching (3 layers)
- Lazy initialization
- Connection pooling
- Benchmarking

### Phase 4: RAG Integration (Week 5)

- rag/search, rag/index tools
- Resources for documents
- Hybrid search
- Reranking

### Phase 5: Security (Week 6)

- Input sanitization
- Rate limiting
- Resource limits
- AppArmor/containerization

### Phase 6: Testing & Docs (Week 7)

- 80%+ test coverage
- MCP Inspector suite
- Auto-generated docs
- Usage examples

---

## 9. Critical Dependencies

```toml
[dependencies]
# MCP Protocol
tokio = { version = "1.40", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
uuid = { version = "1.10", features = ["v4", "serde"] }

# Performance
moka = { version = "0.12", features = ["future"] }
bb8 = "0.8"

# Validation
validator = { version = "0.18", features = ["derive"] }
jsonschema = "0.18"

# Logging
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# Concurrency
dashmap = "6.0"
```

---

## 10. Key Metrics to Track

```rust
pub struct ServerMetrics {
    // Performance
    pub avg_response_time_ms: f64,         // Target: < 5000ms
    pub cache_hit_rate: f64,               // Target: > 70%

    // Reliability
    pub uptime_secs: u64,
    pub error_rate: f64,                   // Target: < 1%
    pub timeout_rate: f64,                 // Target: < 0.1%

    // Usage
    pub requests_total: u64,
    pub active_sessions: usize,
    pub tools_called: HashMap<String, u64>,

    // Resources
    pub cache_size_mb: f64,
    pub pool_active_connections: usize,
    pub pool_idle_connections: usize,
}
```

---

## 11. Common Pitfalls to Avoid

1. **❌ Writing to stdout in stdio transport**
   - ✅ Use stderr for all logging (tracing → stderr automatically)

2. **❌ Exposing too many tools (token overhead)**
   - ✅ Use workflow tools for common chains

3. **❌ No input validation**
   - ✅ JSON Schema + custom validators + sanitization

4. **❌ Synchronous blocking operations**
   - ✅ Use async/await throughout

5. **❌ No timeout management**
   - ✅ Cascading timeouts (tool > LLM > DB)

6. **❌ Hardcoded configuration**
   - ✅ Environment variables + config files

7. **❌ Poor error messages**
   - ✅ Rich context + suggestions + trace IDs

8. **❌ No caching**
   - ✅ Multi-layer caching (embeddings, search, protocols)

9. **❌ Missing audit trail**
   - ✅ Log every tool call with input hash

10. **❌ No rate limiting**
    - ✅ Per-client rate limits + resource caps

---

## 12. Quick Wins

### Immediate Impact (< 1 week)

1. Add Moka caching for embeddings → 10-100x speedup on cache hits
2. Lazy load embedding model → 5-10s faster startup
3. Add JSON Schema validation → Catch 80% of invalid inputs
4. Rich error messages → 50% reduction in debugging time

### High Value (1-2 weeks)

1. Session-based execution → Enable multi-step reasoning
2. Workflow tools (profiles) → 5x reduction in LLM tokens
3. Auto-generated docs → Always up-to-date
4. Connection pooling → 2-5x throughput improvement

---

**Quick Reference Version**: 1.0.0
**Last Updated**: 2025-12-25
**Full Guide**: `MCP_IMPLEMENTATION_BEST_PRACTICES.md`
