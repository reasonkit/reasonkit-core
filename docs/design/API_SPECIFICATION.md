# ReasonKit ThinkTools API Specification

> Public API Surface for ReasonKit AI Reasoning Infrastructure
> Version: 1.0.0 | Status: Draft | Date: 2025-12-28

---

## Table of Contents

1. [Overview](#1-overview)
2. [Authentication](#2-authentication)
3. [REST API](#3-rest-api)
4. [Streaming API](#4-streaming-api)
5. [Webhook/Callback Patterns](#5-webhookcallback-patterns)
6. [SDK Interface Design](#6-sdk-interface-design)
7. [Rate Limiting and Quotas](#7-rate-limiting-and-quotas)
8. [OpenAPI 3.1 Specification](#8-openapi-31-specification)

---

## 1. Overview

### 1.1 Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Developer Ergonomics** | Consistent naming, sensible defaults, comprehensive error messages |
| **Observability** | Trace IDs on every request, step-level visibility, structured logs |
| **Cost Transparency** | Token counts per step, estimated cost in every response |
| **Idempotency** | Client-generated request IDs, safe retries for all operations |
| **Streaming First** | Real-time step updates for long-running reasoning chains |

### 1.2 Base URL

```
Production:  https://api.reasonkit.sh/v1
Staging:     https://api.staging.reasonkit.sh/v1
Self-hosted: http://localhost:8080/v1
```

### 1.3 Content Types

| Type | Media Type | Usage |
|------|------------|-------|
| Request Body | `application/json` | All POST/PUT requests |
| Response Body | `application/json` | Standard responses |
| Streaming | `text/event-stream` | SSE streaming responses |
| Webhook Payload | `application/json` | Async callbacks |

---

## 2. Authentication

### 2.1 API Key Authentication

```http
Authorization: Bearer rk_live_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Key Format:**
- `rk_live_*` - Production keys
- `rk_test_*` - Test/sandbox keys (no billing)
- `rk_dev_*` - Development keys (local only)

### 2.2 Organization Context

```http
X-ReasonKit-Org: org_xxxxxxxxxxxxxxxx
```

Optional header for multi-tenant deployments. If omitted, uses the default organization for the API key.

### 2.3 Request Identification

```http
X-Request-ID: req_xxxxxxxxxxxxxxxx
X-Idempotency-Key: idem_xxxxxxxxxxxxxxxx
```

| Header | Purpose | Format |
|--------|---------|--------|
| `X-Request-ID` | Client-generated request identifier for tracing | `req_` + 16-32 alphanumeric |
| `X-Idempotency-Key` | Ensures exactly-once execution for retries | `idem_` + UUID or custom |

---

## 3. REST API

### 3.1 Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/think` | Execute a reasoning request |
| `POST` | `/v1/think/stream` | Execute with SSE streaming |
| `GET` | `/v1/traces/{trace_id}` | Retrieve execution trace |
| `GET` | `/v1/protocols` | List available protocols |
| `GET` | `/v1/protocols/{id}` | Get protocol details |
| `GET` | `/v1/profiles` | List reasoning profiles |
| `GET` | `/v1/profiles/{id}` | Get profile details |
| `POST` | `/v1/webhooks` | Register webhook endpoint |
| `GET` | `/v1/usage` | Get usage statistics |

### 3.2 POST /v1/think - Execute Reasoning

Primary endpoint for synchronous reasoning requests.

**Request:**

```json
{
  "query": "Should we adopt microservices architecture?",
  "protocol": "gigathink",
  "profile": null,
  "context": {
    "team_size": 15,
    "current_architecture": "monolith",
    "timeline": "6 months"
  },
  "options": {
    "min_confidence": 0.8,
    "max_tokens": 8000,
    "temperature": 0.7,
    "save_trace": true,
    "tags": ["architecture", "decision"]
  },
  "idempotency_key": "idem_abc123def456"
}
```

**Request Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | The question or problem to analyze |
| `protocol` | string | No* | Protocol ID (e.g., "gigathink", "laserlogic") |
| `profile` | string | No* | Profile ID (e.g., "quick", "balanced", "paranoid") |
| `context` | object | No | Additional context as key-value pairs |
| `options.min_confidence` | number | No | Minimum confidence threshold (0.0-1.0) |
| `options.max_tokens` | integer | No | Maximum tokens for entire chain |
| `options.temperature` | number | No | LLM temperature (0.0-2.0) |
| `options.save_trace` | boolean | No | Persist trace for later retrieval |
| `options.tags` | array | No | Custom tags for filtering/analytics |
| `idempotency_key` | string | No | For safe retries |

*Either `protocol` or `profile` is required.

**Response (200 OK):**

```json
{
  "id": "exec_7f8a9b0c1d2e3f4g",
  "trace_id": "trace_1a2b3c4d5e6f7g8h",
  "status": "completed",
  "protocol_id": "gigathink",
  "profile_id": null,
  "input": {
    "query": "Should we adopt microservices architecture?"
  },
  "output": {
    "perspectives": [
      {
        "dimension": "Scalability",
        "insight": "Microservices enable independent scaling...",
        "confidence": 0.92
      },
      {
        "dimension": "Team Structure",
        "insight": "Aligns with Conway's Law for autonomous teams...",
        "confidence": 0.88
      }
    ],
    "synthesis": "Based on 7 perspectives analyzed...",
    "recommendation": "Conditional adoption with prerequisites",
    "verdict": "proceed_with_caution"
  },
  "confidence": 0.86,
  "steps": [
    {
      "step_id": "identify_dimensions",
      "status": "completed",
      "confidence": 0.88,
      "duration_ms": 1245,
      "tokens": {
        "input": 156,
        "output": 423,
        "total": 579
      }
    },
    {
      "step_id": "explore_each",
      "status": "completed",
      "confidence": 0.85,
      "duration_ms": 2341,
      "tokens": {
        "input": 312,
        "output": 1247,
        "total": 1559
      }
    },
    {
      "step_id": "synthesize",
      "status": "completed",
      "confidence": 0.87,
      "duration_ms": 892,
      "tokens": {
        "input": 1423,
        "output": 567,
        "total": 1990
      }
    }
  ],
  "tokens": {
    "input": 1891,
    "output": 2237,
    "total": 4128,
    "cost_usd": 0.0147
  },
  "timing": {
    "total_ms": 4478,
    "llm_ms": 4123,
    "processing_ms": 355
  },
  "metadata": {
    "model": "claude-sonnet-4",
    "provider": "anthropic",
    "temperature": 0.7,
    "tags": ["architecture", "decision"]
  },
  "created_at": "2025-12-28T14:30:45.123Z"
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique execution identifier |
| `trace_id` | string | Trace ID for detailed debugging |
| `status` | string | `completed`, `failed`, `timeout` |
| `confidence` | number | Overall confidence score (0.0-1.0) |
| `steps` | array | Per-step execution details |
| `tokens` | object | Total token usage with cost |
| `timing` | object | Execution timing breakdown |

### 3.3 POST /v1/think/stream - Streaming Execution

Same request format as `/v1/think`, but returns Server-Sent Events.

**Request Headers:**

```http
Accept: text/event-stream
Cache-Control: no-cache
```

**Response (200 OK, SSE Stream):**

```
event: execution_started
id: evt_1
data: {"execution_id":"exec_7f8a9b0c1d2e3f4g","trace_id":"trace_1a2b3c4d5e6f7g8h","protocol_id":"gigathink"}

event: step_started
id: evt_2
data: {"step_id":"identify_dimensions","step_index":0,"total_steps":3}

event: step_progress
id: evt_3
data: {"step_id":"identify_dimensions","progress":0.5,"message":"Generating perspectives..."}

event: step_completed
id: evt_4
data: {"step_id":"identify_dimensions","confidence":0.88,"duration_ms":1245,"tokens":{"input":156,"output":423,"total":579},"output":{"type":"list","items":[...]}}

event: step_started
id: evt_5
data: {"step_id":"explore_each","step_index":1,"total_steps":3}

event: step_completed
id: evt_6
data: {"step_id":"explore_each","confidence":0.85,"duration_ms":2341,"tokens":{"input":312,"output":1247,"total":1559},"output":{"type":"list","items":[...]}}

event: step_started
id: evt_7
data: {"step_id":"synthesize","step_index":2,"total_steps":3}

event: step_completed
id: evt_8
data: {"step_id":"synthesize","confidence":0.87,"duration_ms":892,"tokens":{"input":1423,"output":567,"total":1990},"output":{"type":"text","content":"..."}}

event: execution_completed
id: evt_9
data: {"execution_id":"exec_7f8a9b0c1d2e3f4g","status":"completed","confidence":0.86,"total_tokens":4128,"cost_usd":0.0147,"duration_ms":4478}

event: done
id: evt_10
data: [DONE]
```

**SSE Event Types:**

| Event | Description | Data Fields |
|-------|-------------|-------------|
| `execution_started` | Execution has begun | `execution_id`, `trace_id`, `protocol_id` |
| `step_started` | Step execution starting | `step_id`, `step_index`, `total_steps` |
| `step_progress` | Intermediate progress | `step_id`, `progress`, `message` |
| `step_completed` | Step finished | `step_id`, `confidence`, `tokens`, `output` |
| `execution_completed` | Full execution done | Full result summary |
| `error` | Error occurred | `error_code`, `message`, `step_id` |
| `done` | Stream ended | `[DONE]` |

### 3.4 GET /v1/traces/{trace_id} - Retrieve Trace

**Response (200 OK):**

```json
{
  "id": "trace_1a2b3c4d5e6f7g8h",
  "execution_id": "exec_7f8a9b0c1d2e3f4g",
  "protocol_id": "gigathink",
  "protocol_version": "1.0.0",
  "status": "completed",
  "input": {
    "query": "Should we adopt microservices architecture?"
  },
  "output": {
    "perspectives": [...],
    "synthesis": "..."
  },
  "steps": [
    {
      "step_id": "identify_dimensions",
      "index": 0,
      "status": "completed",
      "prompt": "Identify 5-10 distinct dimensions/angles to analyze: Should we adopt microservices architecture?\nContext: {\"team_size\": 15}",
      "raw_response": "1. Scalability\n2. Team Structure\n3. Operational Complexity...",
      "parsed_output": {
        "type": "list",
        "items": [...]
      },
      "confidence": 0.88,
      "duration_ms": 1245,
      "tokens": {
        "input": 156,
        "output": 423,
        "total": 579,
        "cost_usd": 0.0021
      },
      "started_at": "2025-12-28T14:30:45.123Z",
      "completed_at": "2025-12-28T14:30:46.368Z"
    }
  ],
  "confidence": 0.86,
  "tokens": {
    "input": 1891,
    "output": 2237,
    "total": 4128,
    "cost_usd": 0.0147
  },
  "timing": {
    "started_at": "2025-12-28T14:30:45.000Z",
    "completed_at": "2025-12-28T14:30:49.478Z",
    "total_ms": 4478,
    "llm_ms": 4123,
    "processing_ms": 355
  },
  "metadata": {
    "model": "claude-sonnet-4",
    "provider": "anthropic",
    "temperature": 0.7,
    "profile": null,
    "tags": ["architecture", "decision"],
    "environment": "production"
  }
}
```

### 3.5 GET /v1/protocols - List Protocols

**Response (200 OK):**

```json
{
  "protocols": [
    {
      "id": "gigathink",
      "name": "GigaThink",
      "shortcode": "gt",
      "description": "Expansive creative thinking - generate 10+ diverse perspectives",
      "strategy": "expansive",
      "input_fields": {
        "required": ["query"],
        "optional": ["context", "constraints"]
      },
      "steps_count": 4,
      "typical_tokens": 2000,
      "typical_duration_ms": 5000
    },
    {
      "id": "laserlogic",
      "name": "LaserLogic",
      "shortcode": "ll",
      "description": "Precision deductive reasoning with fallacy detection",
      "strategy": "deductive",
      "input_fields": {
        "required": ["argument"],
        "optional": ["context"]
      },
      "steps_count": 4,
      "typical_tokens": 1500,
      "typical_duration_ms": 3000
    },
    {
      "id": "bedrock",
      "name": "BedRock",
      "shortcode": "br",
      "description": "First principles decomposition and axiom identification",
      "strategy": "analytical",
      "input_fields": {
        "required": ["statement"],
        "optional": ["domain"]
      },
      "steps_count": 4,
      "typical_tokens": 1800,
      "typical_duration_ms": 4000
    },
    {
      "id": "proofguard",
      "name": "ProofGuard",
      "shortcode": "pg",
      "description": "Multi-source verification and triangulation",
      "strategy": "verification",
      "input_fields": {
        "required": ["claim"],
        "optional": ["sources"]
      },
      "steps_count": 4,
      "typical_tokens": 2500,
      "typical_duration_ms": 6000
    },
    {
      "id": "brutalhonesty",
      "name": "BrutalHonesty",
      "shortcode": "bh",
      "description": "Adversarial self-critique to find flaws first",
      "strategy": "adversarial",
      "input_fields": {
        "required": ["work"],
        "optional": ["criteria"]
      },
      "steps_count": 4,
      "typical_tokens": 1800,
      "typical_duration_ms": 4000
    }
  ],
  "total": 5
}
```

### 3.6 GET /v1/profiles - List Profiles

**Response (200 OK):**

```json
{
  "profiles": [
    {
      "id": "quick",
      "name": "Quick Analysis",
      "description": "Fast 2-step analysis for rapid insights",
      "chain": ["gigathink", "laserlogic"],
      "min_confidence": 0.70,
      "typical_tokens": 3000,
      "typical_duration_ms": 8000,
      "tags": ["fast", "creative"]
    },
    {
      "id": "balanced",
      "name": "Balanced Analysis",
      "description": "Standard 4-module chain for thorough analysis",
      "chain": ["gigathink", "laserlogic", "bedrock", "proofguard"],
      "min_confidence": 0.80,
      "typical_tokens": 8000,
      "typical_duration_ms": 18000,
      "tags": ["standard", "thorough"]
    },
    {
      "id": "deep",
      "name": "Deep Analysis",
      "description": "Thorough analysis with all 5 tools",
      "chain": ["gigathink", "laserlogic", "bedrock", "proofguard", "brutalhonesty"],
      "min_confidence": 0.85,
      "typical_tokens": 12000,
      "typical_duration_ms": 25000,
      "tags": ["thorough", "analytical"]
    },
    {
      "id": "paranoid",
      "name": "Paranoid Verification",
      "description": "Maximum rigor with multi-pass verification",
      "chain": ["gigathink", "laserlogic", "bedrock", "proofguard", "brutalhonesty", "proofguard"],
      "min_confidence": 0.95,
      "typical_tokens": 18000,
      "typical_duration_ms": 35000,
      "tags": ["rigorous", "verification", "adversarial"]
    },
    {
      "id": "powercombo",
      "name": "PowerCombo Ultimate",
      "description": "All 5 ThinkTools with cross-validation",
      "chain": ["gigathink", "laserlogic", "bedrock", "proofguard", "brutalhonesty", "proofguard"],
      "min_confidence": 0.95,
      "typical_tokens": 25000,
      "typical_duration_ms": 45000,
      "tags": ["ultimate", "all-tools", "maximum-rigor"]
    }
  ],
  "total": 5
}
```

### 3.7 GET /v1/usage - Usage Statistics

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `start_date` | string | ISO 8601 date (default: 30 days ago) |
| `end_date` | string | ISO 8601 date (default: now) |
| `group_by` | string | `day`, `week`, `month` |

**Response (200 OK):**

```json
{
  "period": {
    "start": "2025-11-28T00:00:00Z",
    "end": "2025-12-28T23:59:59Z"
  },
  "summary": {
    "total_executions": 1247,
    "successful_executions": 1198,
    "failed_executions": 49,
    "total_tokens": 5234891,
    "total_cost_usd": 18.67,
    "average_confidence": 0.84,
    "average_duration_ms": 4521
  },
  "by_protocol": {
    "gigathink": {
      "executions": 523,
      "tokens": 1046000,
      "cost_usd": 3.73
    },
    "laserlogic": {
      "executions": 312,
      "tokens": 468000,
      "cost_usd": 1.67
    }
  },
  "by_profile": {
    "balanced": {
      "executions": 245,
      "tokens": 1960000,
      "cost_usd": 6.98
    }
  },
  "daily": [
    {
      "date": "2025-12-27",
      "executions": 45,
      "tokens": 189000,
      "cost_usd": 0.67
    }
  ],
  "quota": {
    "monthly_tokens_limit": 10000000,
    "monthly_tokens_used": 5234891,
    "monthly_tokens_remaining": 4765109,
    "rate_limit_requests_per_minute": 60,
    "rate_limit_tokens_per_minute": 100000
  }
}
```

---

## 4. Streaming API

### 4.1 Server-Sent Events Format

All streaming responses follow the W3C Server-Sent Events specification:

```
event: <event_type>
id: <event_id>
data: <json_payload>
retry: <reconnect_ms>

```

### 4.2 Reconnection Handling

Clients should implement reconnection with the `Last-Event-ID` header:

```http
GET /v1/think/stream HTTP/1.1
Last-Event-ID: evt_6
```

The server will resume from the specified event ID if the execution is still in progress.

### 4.3 Heartbeat Events

For long-running operations, the server sends heartbeat events every 15 seconds:

```
event: heartbeat
id: hb_1
data: {"timestamp":"2025-12-28T14:31:00.000Z","elapsed_ms":15000}

```

### 4.4 Token Streaming (Granular Output)

For real-time token output, use `stream_tokens=true`:

```json
{
  "query": "...",
  "profile": "balanced",
  "options": {
    "stream_tokens": true
  }
}
```

Additional event type:

```
event: token
id: tok_1
data: {"step_id":"synthesize","content":" Based","index":0}

event: token
id: tok_2
data: {"step_id":"synthesize","content":" on","index":1}

event: token
id: tok_3
data: {"step_id":"synthesize","content":" the","index":2}
```

---

## 5. Webhook/Callback Patterns

### 5.1 Register Webhook

**POST /v1/webhooks**

```json
{
  "url": "https://your-app.com/webhooks/reasonkit",
  "events": ["execution.completed", "execution.failed"],
  "secret": "whsec_xxxxxxxxxxxxxxxx"
}
```

**Response (201 Created):**

```json
{
  "id": "wh_1a2b3c4d",
  "url": "https://your-app.com/webhooks/reasonkit",
  "events": ["execution.completed", "execution.failed"],
  "status": "active",
  "created_at": "2025-12-28T14:30:45Z"
}
```

### 5.2 Webhook Events

| Event | Trigger |
|-------|---------|
| `execution.started` | Execution has begun processing |
| `execution.step_completed` | Individual step completed |
| `execution.completed` | Entire execution finished successfully |
| `execution.failed` | Execution failed with error |
| `execution.timeout` | Execution timed out |

### 5.3 Webhook Payload

```json
{
  "id": "evt_webhook_1a2b3c4d",
  "type": "execution.completed",
  "created": "2025-12-28T14:30:49.478Z",
  "data": {
    "execution_id": "exec_7f8a9b0c1d2e3f4g",
    "trace_id": "trace_1a2b3c4d5e6f7g8h",
    "status": "completed",
    "protocol_id": "gigathink",
    "confidence": 0.86,
    "tokens": {
      "total": 4128,
      "cost_usd": 0.0147
    },
    "duration_ms": 4478
  }
}
```

### 5.4 Webhook Signature Verification

All webhook requests include a signature header:

```http
X-ReasonKit-Signature: t=1703772649,v1=5d7e8f9a0b1c2d3e4f5g6h7i8j9k0l
```

**Verification (Python):**

```python
import hmac
import hashlib
import time

def verify_webhook(payload: bytes, signature_header: str, secret: str) -> bool:
    parts = dict(p.split("=") for p in signature_header.split(","))
    timestamp = int(parts["t"])
    expected_sig = parts["v1"]

    # Check timestamp (5 minute tolerance)
    if abs(time.time() - timestamp) > 300:
        return False

    # Compute expected signature
    signed_payload = f"{timestamp}.{payload.decode()}"
    computed_sig = hmac.new(
        secret.encode(),
        signed_payload.encode(),
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(computed_sig, expected_sig)
```

### 5.5 Async Execution with Callback

For long-running operations, use async mode:

**POST /v1/think**

```json
{
  "query": "...",
  "profile": "paranoid",
  "async": true,
  "callback_url": "https://your-app.com/callbacks/reasonkit",
  "callback_events": ["step_completed", "execution_completed"]
}
```

**Immediate Response (202 Accepted):**

```json
{
  "execution_id": "exec_7f8a9b0c1d2e3f4g",
  "trace_id": "trace_1a2b3c4d5e6f7g8h",
  "status": "pending",
  "poll_url": "https://api.reasonkit.sh/v1/executions/exec_7f8a9b0c1d2e3f4g",
  "estimated_duration_ms": 35000
}
```

---

## 6. SDK Interface Design

### 6.1 Rust SDK (Native)

```rust
use reasonkit::{Client, Profile, ThinkOptions, ThinkResult};

// Initialize client
let client = Client::new("rk_live_xxx")?;

// Simple execution
let result = client
    .think("Should we adopt microservices?")
    .profile(Profile::Balanced)
    .execute()
    .await?;

println!("Confidence: {}", result.confidence);
println!("Recommendation: {}", result.output.recommendation);

// Streaming execution
let mut stream = client
    .think("Complex analysis question")
    .profile(Profile::Paranoid)
    .stream()
    .await?;

while let Some(event) = stream.next().await {
    match event? {
        ThinkEvent::StepStarted { step_id, index, total } => {
            println!("[{}/{}] Starting: {}", index + 1, total, step_id);
        }
        ThinkEvent::StepCompleted { step_id, confidence, tokens } => {
            println!("Completed: {} (conf: {:.2}, tokens: {})",
                step_id, confidence, tokens.total);
        }
        ThinkEvent::Completed(result) => {
            println!("Final confidence: {:.2}", result.confidence);
            println!("Total cost: ${:.4}", result.tokens.cost_usd);
        }
        _ => {}
    }
}

// With full options
let result = client
    .think("What are the risks of this approach?")
    .protocol("proofguard")
    .context("domain", "financial services")
    .context("risk_tolerance", "low")
    .options(ThinkOptions {
        min_confidence: 0.9,
        max_tokens: 10000,
        temperature: 0.5,
        save_trace: true,
        tags: vec!["risk-analysis".into(), "finance".into()],
    })
    .idempotency_key("analysis-123")
    .execute()
    .await?;

// Access trace
let trace = client.trace(&result.trace_id).await?;
for step in &trace.steps {
    println!("Step {}: {}", step.step_id, step.status);
    println!("  Prompt: {}", &step.prompt[..100.min(step.prompt.len())]);
    println!("  Tokens: {}", step.tokens.total);
}
```

**Rust SDK Types:**

```rust
/// Main client for ReasonKit API
pub struct Client {
    api_key: String,
    base_url: String,
    http_client: reqwest::Client,
}

/// Think request builder
pub struct ThinkRequest {
    query: String,
    protocol: Option<String>,
    profile: Option<Profile>,
    context: HashMap<String, Value>,
    options: ThinkOptions,
    idempotency_key: Option<String>,
}

/// Execution result
pub struct ThinkResult {
    pub id: String,
    pub trace_id: String,
    pub status: ExecutionStatus,
    pub output: ThinkOutput,
    pub confidence: f64,
    pub steps: Vec<StepSummary>,
    pub tokens: TokenUsage,
    pub timing: Timing,
}

/// Step-level summary
pub struct StepSummary {
    pub step_id: String,
    pub status: StepStatus,
    pub confidence: f64,
    pub duration_ms: u64,
    pub tokens: TokenUsage,
}

/// Token usage with cost
pub struct TokenUsage {
    pub input: u32,
    pub output: u32,
    pub total: u32,
    pub cost_usd: f64,
}

/// Available profiles
pub enum Profile {
    Quick,
    Balanced,
    Deep,
    Paranoid,
    Scientific,
    Decide,
    PowerCombo,
    Custom(String),
}

/// Streaming event types
pub enum ThinkEvent {
    ExecutionStarted { execution_id: String, trace_id: String },
    StepStarted { step_id: String, index: usize, total: usize },
    StepProgress { step_id: String, progress: f64, message: String },
    StepCompleted { step_id: String, confidence: f64, tokens: TokenUsage, output: StepOutput },
    Completed(ThinkResult),
    Error { code: String, message: String, step_id: Option<String> },
}
```

### 6.2 Python SDK (via PyO3 bindings)

```python
from reasonkit import Client, Profile, ThinkOptions
import asyncio

# Initialize client
client = Client("rk_live_xxx")

# Simple execution
result = client.think(
    "Should we adopt microservices?",
    profile=Profile.BALANCED
)

print(f"Confidence: {result.confidence}")
print(f"Recommendation: {result.output.recommendation}")

# Async streaming
async def stream_example():
    stream = client.think_stream(
        "Complex analysis question",
        profile=Profile.PARANOID
    )

    async for event in stream:
        if event.type == "step_started":
            print(f"[{event.index + 1}/{event.total}] Starting: {event.step_id}")
        elif event.type == "step_completed":
            print(f"Completed: {event.step_id} (conf: {event.confidence:.2f})")
        elif event.type == "completed":
            print(f"Final: {event.result.confidence:.2f}")
            print(f"Cost: ${event.result.tokens.cost_usd:.4f}")

asyncio.run(stream_example())

# With full options
result = client.think(
    "What are the risks?",
    protocol="proofguard",
    context={
        "domain": "financial services",
        "risk_tolerance": "low"
    },
    options=ThinkOptions(
        min_confidence=0.9,
        max_tokens=10000,
        temperature=0.5,
        save_trace=True,
        tags=["risk-analysis", "finance"]
    ),
    idempotency_key="analysis-123"
)

# Access detailed trace
trace = client.get_trace(result.trace_id)
for step in trace.steps:
    print(f"Step {step.step_id}: {step.status}")
    print(f"  Confidence: {step.confidence:.2f}")
    print(f"  Tokens: {step.tokens.total}")
    print(f"  Duration: {step.duration_ms}ms")

# Webhook registration
webhook = client.webhooks.create(
    url="https://your-app.com/webhooks/reasonkit",
    events=["execution.completed", "execution.failed"],
    secret="whsec_xxx"
)
```

**Python SDK Classes:**

```python
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from enum import Enum

class Profile(Enum):
    QUICK = "quick"
    BALANCED = "balanced"
    DEEP = "deep"
    PARANOID = "paranoid"
    SCIENTIFIC = "scientific"
    DECIDE = "decide"
    POWERCOMBO = "powercombo"

@dataclass
class ThinkOptions:
    min_confidence: float = 0.8
    max_tokens: int = 8000
    temperature: float = 0.7
    save_trace: bool = False
    tags: List[str] = None

@dataclass
class TokenUsage:
    input: int
    output: int
    total: int
    cost_usd: float

@dataclass
class StepResult:
    step_id: str
    status: str
    confidence: float
    duration_ms: int
    tokens: TokenUsage
    output: Any

@dataclass
class ThinkResult:
    id: str
    trace_id: str
    status: str
    output: Dict[str, Any]
    confidence: float
    steps: List[StepResult]
    tokens: TokenUsage
    timing: Dict[str, int]

class Client:
    def __init__(self, api_key: str, base_url: str = "https://api.reasonkit.sh/v1"):
        ...

    def think(
        self,
        query: str,
        protocol: Optional[str] = None,
        profile: Optional[Profile] = None,
        context: Optional[Dict[str, Any]] = None,
        options: Optional[ThinkOptions] = None,
        idempotency_key: Optional[str] = None
    ) -> ThinkResult:
        ...

    async def think_stream(
        self,
        query: str,
        **kwargs
    ) -> AsyncIterator[ThinkEvent]:
        ...

    def get_trace(self, trace_id: str) -> ExecutionTrace:
        ...

    def list_protocols(self) -> List[Protocol]:
        ...

    def list_profiles(self) -> List[ProfileInfo]:
        ...

    def get_usage(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> UsageReport:
        ...
```

---

## 7. Rate Limiting and Quotas

### 7.1 Rate Limit Headers

All responses include rate limit information:

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1703773200
X-RateLimit-Tokens-Limit: 100000
X-RateLimit-Tokens-Remaining: 85000
X-RateLimit-Tokens-Reset: 1703773200
```

### 7.2 Rate Limit Tiers

| Tier | Requests/min | Tokens/min | Concurrent | Monthly Tokens |
|------|-------------|------------|------------|----------------|
| Free | 10 | 10,000 | 2 | 100,000 |
| Starter | 30 | 50,000 | 5 | 1,000,000 |
| Pro | 60 | 100,000 | 10 | 10,000,000 |
| Enterprise | Custom | Custom | Custom | Unlimited |

### 7.3 Rate Limit Response

**429 Too Many Requests:**

```json
{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "Rate limit exceeded. Please retry after 45 seconds.",
    "type": "requests",
    "limit": 60,
    "current": 61,
    "retry_after": 45
  }
}
```

### 7.4 Token Quota Response

**402 Payment Required:**

```json
{
  "error": {
    "code": "quota_exceeded",
    "message": "Monthly token quota exceeded.",
    "type": "tokens",
    "limit": 1000000,
    "used": 1000000,
    "reset_at": "2026-01-01T00:00:00Z",
    "upgrade_url": "https://reasonkit.sh/pricing"
  }
}
```

### 7.5 Retry Strategy

Recommended exponential backoff:

```python
import time
import random

def retry_with_backoff(func, max_retries=5):
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise

            # Use retry_after if provided, otherwise exponential backoff
            wait_time = e.retry_after or (2 ** attempt + random.uniform(0, 1))
            time.sleep(wait_time)
```

---

## 8. OpenAPI 3.1 Specification

```yaml
openapi: 3.1.0
info:
  title: ReasonKit ThinkTools API
  description: |
    Structured AI reasoning infrastructure. Execute reasoning protocols,
    chain ThinkTools, and get auditable execution traces.
  version: 1.0.0
  contact:
    name: ReasonKit Support
    url: https://reasonkit.sh/support
    email: support@reasonkit.sh
  license:
    name: Apache 2.0
    url: https://www.apache.org/licenses/LICENSE-2.0

servers:
  - url: https://api.reasonkit.sh/v1
    description: Production
  - url: https://api.staging.reasonkit.sh/v1
    description: Staging
  - url: http://localhost:8080/v1
    description: Local Development

security:
  - BearerAuth: []

tags:
  - name: Think
    description: Execute reasoning operations
  - name: Protocols
    description: Available reasoning protocols
  - name: Profiles
    description: Pre-configured reasoning chains
  - name: Traces
    description: Execution traces for debugging
  - name: Webhooks
    description: Async notification configuration
  - name: Usage
    description: Usage statistics and quotas

paths:
  /think:
    post:
      operationId: executeThink
      summary: Execute a reasoning request
      description: |
        Execute a ThinkTool protocol or profile against the provided query.
        Returns the full reasoning result with step-by-step details.
      tags:
        - Think
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ThinkRequest'
            examples:
              simple:
                summary: Simple protocol execution
                value:
                  query: "Should we adopt microservices?"
                  protocol: "gigathink"
              profile:
                summary: Using a profile
                value:
                  query: "What are the risks of this approach?"
                  profile: "balanced"
                  context:
                    domain: "financial services"
                  options:
                    min_confidence: 0.85
                    save_trace: true
      responses:
        '200':
          description: Successful execution
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ThinkResult'
          headers:
            X-Request-ID:
              schema:
                type: string
              description: Request identifier
            X-RateLimit-Remaining:
              schema:
                type: integer
              description: Remaining requests in current window
        '400':
          $ref: '#/components/responses/BadRequest'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '402':
          $ref: '#/components/responses/QuotaExceeded'
        '422':
          $ref: '#/components/responses/ValidationError'
        '429':
          $ref: '#/components/responses/RateLimited'
        '500':
          $ref: '#/components/responses/InternalError'

  /think/stream:
    post:
      operationId: executeThinkStream
      summary: Execute with streaming response
      description: |
        Same as /think but returns Server-Sent Events for real-time progress.
      tags:
        - Think
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ThinkRequest'
      responses:
        '200':
          description: SSE stream
          content:
            text/event-stream:
              schema:
                type: string
                description: Server-Sent Events stream
        '400':
          $ref: '#/components/responses/BadRequest'
        '401':
          $ref: '#/components/responses/Unauthorized'

  /traces/{trace_id}:
    get:
      operationId: getTrace
      summary: Retrieve execution trace
      description: Get detailed trace including prompts, responses, and timing.
      tags:
        - Traces
      parameters:
        - name: trace_id
          in: path
          required: true
          schema:
            type: string
            pattern: '^trace_[a-zA-Z0-9]{16}$'
      responses:
        '200':
          description: Trace details
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ExecutionTrace'
        '404':
          $ref: '#/components/responses/NotFound'

  /protocols:
    get:
      operationId: listProtocols
      summary: List available protocols
      tags:
        - Protocols
      responses:
        '200':
          description: List of protocols
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ProtocolList'

  /protocols/{protocol_id}:
    get:
      operationId: getProtocol
      summary: Get protocol details
      tags:
        - Protocols
      parameters:
        - name: protocol_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Protocol details
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Protocol'
        '404':
          $ref: '#/components/responses/NotFound'

  /profiles:
    get:
      operationId: listProfiles
      summary: List reasoning profiles
      tags:
        - Profiles
      responses:
        '200':
          description: List of profiles
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ProfileList'

  /profiles/{profile_id}:
    get:
      operationId: getProfile
      summary: Get profile details
      tags:
        - Profiles
      parameters:
        - name: profile_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Profile details
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Profile'
        '404':
          $ref: '#/components/responses/NotFound'

  /webhooks:
    post:
      operationId: createWebhook
      summary: Register webhook endpoint
      tags:
        - Webhooks
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/WebhookCreate'
      responses:
        '201':
          description: Webhook created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Webhook'
    get:
      operationId: listWebhooks
      summary: List registered webhooks
      tags:
        - Webhooks
      responses:
        '200':
          description: List of webhooks
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/WebhookList'

  /webhooks/{webhook_id}:
    delete:
      operationId: deleteWebhook
      summary: Delete webhook
      tags:
        - Webhooks
      parameters:
        - name: webhook_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '204':
          description: Webhook deleted

  /usage:
    get:
      operationId: getUsage
      summary: Get usage statistics
      tags:
        - Usage
      parameters:
        - name: start_date
          in: query
          schema:
            type: string
            format: date
        - name: end_date
          in: query
          schema:
            type: string
            format: date
        - name: group_by
          in: query
          schema:
            type: string
            enum: [day, week, month]
      responses:
        '200':
          description: Usage statistics
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UsageReport'

components:
  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: API Key
      description: |
        API key authentication. Keys are prefixed:
        - `rk_live_*` - Production
        - `rk_test_*` - Test/sandbox
        - `rk_dev_*` - Development

  schemas:
    ThinkRequest:
      type: object
      required:
        - query
      properties:
        query:
          type: string
          description: The question or problem to analyze
          minLength: 1
          maxLength: 32000
        protocol:
          type: string
          description: Protocol ID (gigathink, laserlogic, etc.)
          enum: [gigathink, laserlogic, bedrock, proofguard, brutalhonesty]
        profile:
          type: string
          description: Profile ID (quick, balanced, paranoid, etc.)
          enum: [quick, balanced, deep, paranoid, scientific, decide, powercombo]
        context:
          type: object
          description: Additional context as key-value pairs
          additionalProperties: true
        options:
          $ref: '#/components/schemas/ThinkOptions'
        idempotency_key:
          type: string
          description: Key for idempotent retries
          pattern: '^idem_[a-zA-Z0-9_-]{8,64}$'
      oneOf:
        - required: [protocol]
        - required: [profile]

    ThinkOptions:
      type: object
      properties:
        min_confidence:
          type: number
          minimum: 0
          maximum: 1
          default: 0.8
          description: Minimum confidence threshold
        max_tokens:
          type: integer
          minimum: 100
          maximum: 100000
          default: 8000
          description: Maximum tokens for entire chain
        temperature:
          type: number
          minimum: 0
          maximum: 2
          default: 0.7
          description: LLM temperature
        save_trace:
          type: boolean
          default: false
          description: Persist trace for later retrieval
        tags:
          type: array
          items:
            type: string
          maxItems: 10
          description: Custom tags for filtering

    ThinkResult:
      type: object
      properties:
        id:
          type: string
          description: Execution identifier
          pattern: '^exec_[a-zA-Z0-9]{16}$'
        trace_id:
          type: string
          description: Trace identifier
          pattern: '^trace_[a-zA-Z0-9]{16}$'
        status:
          type: string
          enum: [completed, failed, timeout]
        protocol_id:
          type: string
        profile_id:
          type: string
          nullable: true
        input:
          type: object
          additionalProperties: true
        output:
          type: object
          additionalProperties: true
          description: Protocol-specific output
        confidence:
          type: number
          minimum: 0
          maximum: 1
          description: Overall confidence score
        steps:
          type: array
          items:
            $ref: '#/components/schemas/StepSummary'
        tokens:
          $ref: '#/components/schemas/TokenUsage'
        timing:
          $ref: '#/components/schemas/Timing'
        metadata:
          $ref: '#/components/schemas/ExecutionMetadata'
        created_at:
          type: string
          format: date-time

    StepSummary:
      type: object
      properties:
        step_id:
          type: string
        status:
          type: string
          enum: [pending, running, completed, failed, skipped]
        confidence:
          type: number
        duration_ms:
          type: integer
        tokens:
          $ref: '#/components/schemas/TokenUsage'

    TokenUsage:
      type: object
      properties:
        input:
          type: integer
          description: Input/prompt tokens
        output:
          type: integer
          description: Output/completion tokens
        total:
          type: integer
          description: Total tokens
        cost_usd:
          type: number
          format: float
          description: Estimated cost in USD

    Timing:
      type: object
      properties:
        total_ms:
          type: integer
          description: Total execution time
        llm_ms:
          type: integer
          description: Time in LLM calls
        processing_ms:
          type: integer
          description: Time in local processing

    ExecutionMetadata:
      type: object
      properties:
        model:
          type: string
        provider:
          type: string
        temperature:
          type: number
        tags:
          type: array
          items:
            type: string

    ExecutionTrace:
      type: object
      properties:
        id:
          type: string
        execution_id:
          type: string
        protocol_id:
          type: string
        protocol_version:
          type: string
        status:
          type: string
        input:
          type: object
        output:
          type: object
        steps:
          type: array
          items:
            $ref: '#/components/schemas/StepTrace'
        confidence:
          type: number
        tokens:
          $ref: '#/components/schemas/TokenUsage'
        timing:
          type: object
          properties:
            started_at:
              type: string
              format: date-time
            completed_at:
              type: string
              format: date-time
            total_ms:
              type: integer
            llm_ms:
              type: integer
            processing_ms:
              type: integer
        metadata:
          $ref: '#/components/schemas/ExecutionMetadata'

    StepTrace:
      type: object
      properties:
        step_id:
          type: string
        index:
          type: integer
        status:
          type: string
        prompt:
          type: string
          description: Actual prompt sent to LLM
        raw_response:
          type: string
          description: Raw LLM response
        parsed_output:
          type: object
          description: Structured output
        confidence:
          type: number
        duration_ms:
          type: integer
        tokens:
          $ref: '#/components/schemas/TokenUsage'
        started_at:
          type: string
          format: date-time
        completed_at:
          type: string
          format: date-time

    Protocol:
      type: object
      properties:
        id:
          type: string
        name:
          type: string
        shortcode:
          type: string
        description:
          type: string
        strategy:
          type: string
          enum: [expansive, deductive, analytical, adversarial, verification, decision, empirical]
        input_fields:
          type: object
          properties:
            required:
              type: array
              items:
                type: string
            optional:
              type: array
              items:
                type: string
        steps_count:
          type: integer
        typical_tokens:
          type: integer
        typical_duration_ms:
          type: integer

    ProtocolList:
      type: object
      properties:
        protocols:
          type: array
          items:
            $ref: '#/components/schemas/Protocol'
        total:
          type: integer

    Profile:
      type: object
      properties:
        id:
          type: string
        name:
          type: string
        description:
          type: string
        chain:
          type: array
          items:
            type: string
          description: Protocol IDs in execution order
        min_confidence:
          type: number
        typical_tokens:
          type: integer
        typical_duration_ms:
          type: integer
        tags:
          type: array
          items:
            type: string

    ProfileList:
      type: object
      properties:
        profiles:
          type: array
          items:
            $ref: '#/components/schemas/Profile'
        total:
          type: integer

    WebhookCreate:
      type: object
      required:
        - url
        - events
      properties:
        url:
          type: string
          format: uri
          description: Webhook endpoint URL
        events:
          type: array
          items:
            type: string
            enum:
              - execution.started
              - execution.step_completed
              - execution.completed
              - execution.failed
              - execution.timeout
        secret:
          type: string
          description: Shared secret for signature verification

    Webhook:
      type: object
      properties:
        id:
          type: string
        url:
          type: string
        events:
          type: array
          items:
            type: string
        status:
          type: string
          enum: [active, disabled]
        created_at:
          type: string
          format: date-time

    WebhookList:
      type: object
      properties:
        webhooks:
          type: array
          items:
            $ref: '#/components/schemas/Webhook'
        total:
          type: integer

    UsageReport:
      type: object
      properties:
        period:
          type: object
          properties:
            start:
              type: string
              format: date-time
            end:
              type: string
              format: date-time
        summary:
          type: object
          properties:
            total_executions:
              type: integer
            successful_executions:
              type: integer
            failed_executions:
              type: integer
            total_tokens:
              type: integer
            total_cost_usd:
              type: number
            average_confidence:
              type: number
            average_duration_ms:
              type: integer
        by_protocol:
          type: object
          additionalProperties:
            type: object
            properties:
              executions:
                type: integer
              tokens:
                type: integer
              cost_usd:
                type: number
        by_profile:
          type: object
          additionalProperties:
            type: object
            properties:
              executions:
                type: integer
              tokens:
                type: integer
              cost_usd:
                type: number
        daily:
          type: array
          items:
            type: object
            properties:
              date:
                type: string
                format: date
              executions:
                type: integer
              tokens:
                type: integer
              cost_usd:
                type: number
        quota:
          type: object
          properties:
            monthly_tokens_limit:
              type: integer
            monthly_tokens_used:
              type: integer
            monthly_tokens_remaining:
              type: integer
            rate_limit_requests_per_minute:
              type: integer
            rate_limit_tokens_per_minute:
              type: integer

    Error:
      type: object
      properties:
        error:
          type: object
          properties:
            code:
              type: string
            message:
              type: string
            type:
              type: string
            details:
              type: object
              additionalProperties: true

  responses:
    BadRequest:
      description: Invalid request
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error:
              code: "bad_request"
              message: "Invalid JSON in request body"

    Unauthorized:
      description: Authentication failed
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error:
              code: "unauthorized"
              message: "Invalid or missing API key"

    NotFound:
      description: Resource not found
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error:
              code: "not_found"
              message: "Trace not found"

    ValidationError:
      description: Validation failed
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error:
              code: "validation_error"
              message: "Either 'protocol' or 'profile' is required"
              details:
                field: "protocol"

    QuotaExceeded:
      description: Quota exceeded
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error:
              code: "quota_exceeded"
              message: "Monthly token quota exceeded"
              type: "tokens"
              limit: 1000000
              used: 1000000

    RateLimited:
      description: Rate limit exceeded
      headers:
        Retry-After:
          schema:
            type: integer
          description: Seconds to wait before retry
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error:
              code: "rate_limit_exceeded"
              message: "Rate limit exceeded"
              retry_after: 45

    InternalError:
      description: Internal server error
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error:
              code: "internal_error"
              message: "An unexpected error occurred"
```

---

## Appendix A: Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `bad_request` | 400 | Malformed request |
| `unauthorized` | 401 | Invalid/missing API key |
| `forbidden` | 403 | Insufficient permissions |
| `not_found` | 404 | Resource not found |
| `validation_error` | 422 | Request validation failed |
| `rate_limit_exceeded` | 429 | Too many requests |
| `quota_exceeded` | 402 | Token quota exceeded |
| `timeout` | 408 | Execution timed out |
| `llm_error` | 502 | Upstream LLM error |
| `internal_error` | 500 | Internal server error |

## Appendix B: Protocol Input Fields

| Protocol | Required | Optional |
|----------|----------|----------|
| `gigathink` | `query` | `context`, `constraints` |
| `laserlogic` | `argument` | `context` |
| `bedrock` | `statement` | `domain` |
| `proofguard` | `claim` | `sources` |
| `brutalhonesty` | `work` | `criteria` |

## Appendix C: Output Structures

### GigaThink Output
```json
{
  "perspectives": [
    {"dimension": "string", "insight": "string", "confidence": 0.0}
  ],
  "themes": ["string"],
  "synthesis": "string",
  "confidence": 0.0
}
```

### LaserLogic Output
```json
{
  "premises": [{"statement": "string", "valid": true}],
  "conclusion": "string",
  "fallacies": [{"type": "string", "description": "string"}],
  "validity": "valid|invalid|uncertain",
  "confidence": 0.0
}
```

### BedRock Output
```json
{
  "axioms": [{"principle": "string", "foundation": "string"}],
  "decomposition": ["string"],
  "reconstruction": "string",
  "confidence": 0.0
}
```

### ProofGuard Output
```json
{
  "sources": [{"name": "string", "stance": "confirms|contradicts|neutral", "tier": 1}],
  "triangulation_status": "verified|partial|unverified",
  "contradictions": ["string"],
  "confidence": 0.0
}
```

### BrutalHonesty Output
```json
{
  "strengths": ["string"],
  "weaknesses": ["string"],
  "risks": ["string"],
  "verdict": "proceed|caution|reconsider|abandon",
  "confidence": 0.0
}
```

---

*ReasonKit ThinkTools API v1.0.0 | "Turn Prompts into Protocols"*
*https://reasonkit.sh*
