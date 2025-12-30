# WEBHOOK AND EVENT NOTIFICATION SYSTEM

> ReasonKit Pro/Enterprise Event Architecture
> Version: 1.0.0 | Status: Design Phase
> Author: ReasonKit Team

---

## 1. EXECUTIVE SUMMARY

This document specifies the complete webhook and event notification system for ReasonKit Pro/Enterprise. The system provides real-time notifications, third-party integrations, and comprehensive event streaming for auditable AI reasoning workflows.

```
+-------------------------------------------------------------------------+
|                     REASONKIT EVENT ARCHITECTURE                         |
+-------------------------------------------------------------------------+
|                                                                          |
|  +-------------+     +------------------+     +----------------------+   |
|  |   ReasonKit |     |   Event Router   |     |   Delivery System    |   |
|  |   Core      | --> |   & Dispatcher   | --> |   (At-Least-Once)    |   |
|  +-------------+     +------------------+     +----------------------+   |
|                              |                          |                |
|                              v                          v                |
|                      +---------------+          +---------------+        |
|                      | Event Store   |          | Dead Letter   |        |
|                      | (Persistence) |          | Queue (DLQ)   |        |
|                      +---------------+          +---------------+        |
|                                                                          |
|  Delivery Channels:                                                      |
|  - Webhooks (HTTPS POST with HMAC signatures)                           |
|  - WebSocket Streams (Enterprise)                                        |
|  - Server-Sent Events (SSE)                                             |
|  - Email Digests                                                         |
|  - Platform Integrations (Slack, Discord, PagerDuty)                    |
|                                                                          |
+-------------------------------------------------------------------------+
```

---

## 2. EVENT TYPES

### 2.1 Event Taxonomy

All events follow a hierarchical naming convention: `{category}.{action}[.{detail}]`

```
+----------------------------------------------------------------------+
|                        EVENT HIERARCHY                                |
+----------------------------------------------------------------------+
|                                                                       |
|  execution.*                                                          |
|  +-- execution.started                                                |
|  +-- execution.step_completed                                         |
|  +-- execution.completed                                              |
|  +-- execution.failed                                                 |
|  +-- execution.timeout                                                |
|  +-- execution.cancelled                                              |
|                                                                       |
|  trace.*                                                              |
|  +-- trace.created                                                    |
|  +-- trace.updated                                                    |
|  +-- trace.exported                                                   |
|  +-- trace.shared                                                     |
|  +-- trace.archived                                                   |
|                                                                       |
|  account.*                                                            |
|  +-- account.user.created                                             |
|  +-- account.user.upgraded                                            |
|  +-- account.user.downgraded                                          |
|  +-- account.api_key.created                                          |
|  +-- account.api_key.revoked                                          |
|                                                                       |
|  usage.*                                                              |
|  +-- usage.limit_warning        (80% threshold)                       |
|  +-- usage.limit_approaching    (90% threshold)                       |
|  +-- usage.limit_exceeded                                             |
|  +-- usage.quota_reset                                                |
|                                                                       |
|  team.*  (Enterprise)                                                 |
|  +-- team.member_added                                                |
|  +-- team.member_removed                                              |
|  +-- team.member_role_changed                                         |
|  +-- team.settings_changed                                            |
|  +-- team.project_created                                             |
|  +-- team.project_archived                                            |
|                                                                       |
|  security.*  (Enterprise)                                             |
|  +-- security.login_success                                           |
|  +-- security.login_failed                                            |
|  +-- security.mfa_enabled                                             |
|  +-- security.suspicious_activity                                     |
|  +-- security.api_key_used                                            |
|                                                                       |
|  ingestion.*                                                          |
|  +-- ingestion.started                                                |
|  +-- ingestion.completed                                              |
|  +-- ingestion.failed                                                 |
|  +-- ingestion.document_added                                         |
|                                                                       |
+----------------------------------------------------------------------+
```

### 2.2 Execution Events

Events generated during ThinkTool protocol executions.

#### execution.started

Fired when a reasoning execution begins.

```json
{
  "id": "evt_01HZK4J8X2YNMGQP3R5T7W9VBC",
  "type": "execution.started",
  "created_at": "2025-12-28T15:30:00.000Z",
  "data": {
    "execution_id": "exec_01HZK4J8X2YNMGQP3R5T7W9VBD",
    "protocol_id": "powercombo",
    "protocol_version": "1.0.0",
    "profile": "balanced",
    "input_tokens": 1250,
    "estimated_steps": 5,
    "user_id": "usr_abc123",
    "project_id": "prj_xyz789",
    "metadata": {
      "source": "api",
      "client_version": "0.2.0",
      "environment": "production"
    }
  }
}
```

#### execution.step_completed

Fired after each reasoning step completes.

```json
{
  "id": "evt_01HZK4J8X2YNMGQP3R5T7W9VBE",
  "type": "execution.step_completed",
  "created_at": "2025-12-28T15:30:05.000Z",
  "data": {
    "execution_id": "exec_01HZK4J8X2YNMGQP3R5T7W9VBD",
    "step_index": 0,
    "step_id": "gigathink",
    "step_name": "GigaThink",
    "status": "completed",
    "confidence": 0.87,
    "duration_ms": 4823,
    "tokens_used": {
      "input": 1250,
      "output": 892
    },
    "remaining_steps": 4
  }
}
```

#### execution.completed

Fired when execution completes successfully.

```json
{
  "id": "evt_01HZK4J8X2YNMGQP3R5T7W9VBF",
  "type": "execution.completed",
  "created_at": "2025-12-28T15:30:30.000Z",
  "data": {
    "execution_id": "exec_01HZK4J8X2YNMGQP3R5T7W9VBD",
    "protocol_id": "powercombo",
    "profile": "balanced",
    "status": "completed",
    "final_confidence": 0.91,
    "duration_ms": 30450,
    "total_steps": 5,
    "completed_steps": 5,
    "tokens": {
      "input": 6250,
      "output": 4120,
      "total": 10370,
      "estimated_cost_usd": 0.0156
    },
    "quality_metrics": {
      "coherence_score": 0.89,
      "depth_score": 0.92,
      "consistency_score": 0.88
    },
    "trace_id": "trace_01HZK4J8X2YNMGQP3R5T7W9VBG"
  }
}
```

#### execution.failed

Fired when execution fails.

```json
{
  "id": "evt_01HZK4J8X2YNMGQP3R5T7W9VBH",
  "type": "execution.failed",
  "created_at": "2025-12-28T15:30:15.000Z",
  "data": {
    "execution_id": "exec_01HZK4J8X2YNMGQP3R5T7W9VBI",
    "protocol_id": "powercombo",
    "profile": "paranoid",
    "status": "failed",
    "failed_at_step": 3,
    "step_id": "proofguard",
    "error": {
      "code": "VALIDATION_FAILED",
      "category": "reasoning",
      "message": "Source triangulation could not verify claim",
      "recoverable": false,
      "details": {
        "claim_index": 2,
        "verification_attempts": 3,
        "sources_checked": 5
      }
    },
    "partial_result": {
      "completed_steps": 2,
      "last_confidence": 0.72
    },
    "duration_ms": 15230,
    "tokens_used": 4850
  }
}
```

#### execution.timeout

Fired when execution times out.

```json
{
  "id": "evt_01HZK4J8X2YNMGQP3R5T7W9VBJ",
  "type": "execution.timeout",
  "created_at": "2025-12-28T15:32:00.000Z",
  "data": {
    "execution_id": "exec_01HZK4J8X2YNMGQP3R5T7W9VBK",
    "protocol_id": "powercombo",
    "profile": "deep",
    "timeout_ms": 120000,
    "actual_duration_ms": 120003,
    "last_step_index": 3,
    "last_step_id": "bedrock",
    "completed_steps": 3,
    "partial_confidence": 0.78
  }
}
```

### 2.3 Trace Events

Events related to execution trace management.

#### trace.created

```json
{
  "id": "evt_01HZK4J8X2YNMGQP3R5T7W9VBL",
  "type": "trace.created",
  "created_at": "2025-12-28T15:30:31.000Z",
  "data": {
    "trace_id": "trace_01HZK4J8X2YNMGQP3R5T7W9VBG",
    "execution_id": "exec_01HZK4J8X2YNMGQP3R5T7W9VBD",
    "protocol_id": "powercombo",
    "step_count": 5,
    "size_bytes": 45230,
    "retention_days": 90,
    "storage_tier": "hot"
  }
}
```

#### trace.exported

```json
{
  "id": "evt_01HZK4J8X2YNMGQP3R5T7W9VBM",
  "type": "trace.exported",
  "created_at": "2025-12-28T16:00:00.000Z",
  "data": {
    "trace_id": "trace_01HZK4J8X2YNMGQP3R5T7W9VBG",
    "export_format": "json",
    "export_options": {
      "include_prompts": false,
      "include_raw_responses": true,
      "redact_pii": true
    },
    "download_url": "https://api.reasonkit.sh/exports/exp_xxx",
    "expires_at": "2025-12-29T16:00:00.000Z",
    "size_bytes": 38450
  }
}
```

#### trace.shared

```json
{
  "id": "evt_01HZK4J8X2YNMGQP3R5T7W9VBN",
  "type": "trace.shared",
  "created_at": "2025-12-28T16:05:00.000Z",
  "data": {
    "trace_id": "trace_01HZK4J8X2YNMGQP3R5T7W9VBG",
    "share_id": "share_01HZK4J8X2YNMGQP3R5T7W9VBO",
    "shared_by": "usr_abc123",
    "share_type": "link",
    "permissions": ["view"],
    "expires_at": "2025-12-31T23:59:59.000Z",
    "share_url": "https://reasonkit.sh/traces/share/xxx"
  }
}
```

### 2.4 Account Events

```json
// account.user.created
{
  "id": "evt_01HZK4J8X2YNMGQP3R5T7W9VBP",
  "type": "account.user.created",
  "created_at": "2025-12-28T12:00:00.000Z",
  "data": {
    "user_id": "usr_abc123",
    "email_domain": "example.com",
    "plan": "pro",
    "signup_source": "website",
    "referral_code": "REF123"
  }
}

// account.user.upgraded
{
  "id": "evt_01HZK4J8X2YNMGQP3R5T7W9VBQ",
  "type": "account.user.upgraded",
  "created_at": "2025-12-28T14:00:00.000Z",
  "data": {
    "user_id": "usr_abc123",
    "previous_plan": "pro",
    "new_plan": "enterprise",
    "billing_cycle": "annual",
    "effective_date": "2025-12-28T14:00:00.000Z"
  }
}
```

### 2.5 Usage Events

```json
// usage.limit_approaching (90% threshold)
{
  "id": "evt_01HZK4J8X2YNMGQP3R5T7W9VBR",
  "type": "usage.limit_approaching",
  "created_at": "2025-12-28T18:00:00.000Z",
  "data": {
    "user_id": "usr_abc123",
    "resource": "api_calls",
    "current_usage": 9000,
    "limit": 10000,
    "percentage": 90,
    "period": "monthly",
    "period_ends_at": "2025-12-31T23:59:59.000Z",
    "recommendation": "Consider upgrading to increase limits"
  }
}

// usage.limit_exceeded
{
  "id": "evt_01HZK4J8X2YNMGQP3R5T7W9VBS",
  "type": "usage.limit_exceeded",
  "created_at": "2025-12-28T20:00:00.000Z",
  "data": {
    "user_id": "usr_abc123",
    "resource": "api_calls",
    "current_usage": 10500,
    "limit": 10000,
    "overage": 500,
    "overage_billed": true,
    "overage_rate_per_unit": 0.001,
    "action_taken": "throttled"
  }
}
```

### 2.6 Team Events (Enterprise)

```json
// team.member_added
{
  "id": "evt_01HZK4J8X2YNMGQP3R5T7W9VBT",
  "type": "team.member_added",
  "created_at": "2025-12-28T10:00:00.000Z",
  "data": {
    "team_id": "team_xyz789",
    "team_name": "Engineering",
    "user_id": "usr_newmember",
    "added_by": "usr_admin123",
    "role": "member",
    "permissions": ["execute", "view_traces"],
    "invitation_accepted": true
  }
}

// team.settings_changed
{
  "id": "evt_01HZK4J8X2YNMGQP3R5T7W9VBU",
  "type": "team.settings_changed",
  "created_at": "2025-12-28T11:00:00.000Z",
  "data": {
    "team_id": "team_xyz789",
    "changed_by": "usr_admin123",
    "changes": [
      {
        "setting": "default_profile",
        "old_value": "balanced",
        "new_value": "paranoid"
      },
      {
        "setting": "trace_retention_days",
        "old_value": 30,
        "new_value": 90
      }
    ]
  }
}
```

### 2.7 Ingestion Events

```json
// ingestion.completed
{
  "id": "evt_01HZK4J8X2YNMGQP3R5T7W9VBV",
  "type": "ingestion.completed",
  "created_at": "2025-12-28T09:00:00.000Z",
  "data": {
    "ingestion_id": "ing_01HZK4J8X2YNMGQP3R5T7W9VBW",
    "source": "github",
    "source_ref": "anthropics/claude-code",
    "documents_processed": 245,
    "chunks_created": 1820,
    "embeddings_generated": 1820,
    "duration_ms": 45230,
    "size_bytes": 2458900,
    "errors": []
  }
}
```

---

## 3. WEBHOOK ARCHITECTURE

### 3.1 Webhook Registration

#### CLI Interface

```bash
# Add a webhook endpoint
rk-core webhooks add \
  --url https://example.com/webhook \
  --events execution.completed,execution.failed \
  --secret my-webhook-secret-key

# Add with advanced options
rk-core webhooks add \
  --url https://hooks.slack.com/services/xxx \
  --events "execution.*,usage.limit_*" \
  --secret $WEBHOOK_SECRET \
  --description "Slack notifications" \
  --filter '{"profile": ["paranoid", "deep"]}' \
  --retry-policy exponential \
  --max-retries 5

# List webhooks
rk-core webhooks list

# Get webhook details
rk-core webhooks get <webhook_id>

# Update webhook
rk-core webhooks update <webhook_id> \
  --events "execution.*" \
  --enabled true

# Delete webhook
rk-core webhooks delete <webhook_id>

# Test webhook (sends test event)
rk-core webhooks test <webhook_id>

# View delivery logs
rk-core webhooks logs <webhook_id> --limit 50
```

#### API Interface

```http
POST /api/v1/webhooks
Content-Type: application/json
Authorization: Bearer <api_key>

{
  "url": "https://example.com/webhook",
  "description": "Production webhook",
  "events": [
    "execution.completed",
    "execution.failed",
    "execution.timeout"
  ],
  "secret": "whsec_your_secret_key",
  "enabled": true,
  "filter": {
    "profile": ["paranoid", "deep"],
    "project_id": "prj_xyz789"
  },
  "retry_policy": {
    "strategy": "exponential",
    "max_retries": 5,
    "initial_delay_ms": 1000,
    "max_delay_ms": 60000
  },
  "headers": {
    "X-Custom-Header": "value"
  }
}
```

Response:

```json
{
  "id": "whk_01HZK4J8X2YNMGQP3R5T7W9VBX",
  "url": "https://example.com/webhook",
  "description": "Production webhook",
  "events": ["execution.completed", "execution.failed", "execution.timeout"],
  "enabled": true,
  "created_at": "2025-12-28T10:00:00.000Z",
  "signing_secret_prefix": "whsec_...abc"
}
```

### 3.2 Webhook Payload Structure

All webhook deliveries follow a consistent structure:

```json
{
  "id": "evt_01HZK4J8X2YNMGQP3R5T7W9VBY",
  "type": "execution.completed",
  "api_version": "2025-12-01",
  "created_at": "2025-12-28T15:30:30.000Z",
  "idempotency_key": "idem_01HZK4J8X2YNMGQP3R5T7W9VBZ",
  "webhook_id": "whk_01HZK4J8X2YNMGQP3R5T7W9VBX",
  "delivery_attempt": 1,
  "data": {
    "execution_id": "exec_01HZK4J8X2YNMGQP3R5T7W9VBD",
    "protocol_id": "powercombo",
    "profile": "balanced",
    "status": "completed",
    "final_confidence": 0.91,
    "duration_ms": 30450,
    "thinktool_count": 5,
    "tokens": {
      "input": 6250,
      "output": 4120,
      "total": 10370
    }
  }
}
```

### 3.3 Security

#### HMAC Signature Verification

All webhook requests include security headers:

```http
POST /webhook HTTP/1.1
Host: example.com
Content-Type: application/json
X-ReasonKit-Signature: sha256=5d45f...
X-ReasonKit-Timestamp: 1703778630
X-ReasonKit-Webhook-Id: whk_01HZK4J8X2YNMGQP3R5T7W9VBX
X-ReasonKit-Delivery-Id: del_01HZK4J8X2YNMGQP3R5T7W9VC0
X-ReasonKit-Event: execution.completed
```

**Signature Computation:**

```python
import hmac
import hashlib

def verify_webhook(payload: bytes, signature: str, timestamp: str, secret: str) -> bool:
    """Verify ReasonKit webhook signature."""
    # Prevent replay attacks (reject if older than 5 minutes)
    import time
    if abs(time.time() - int(timestamp)) > 300:
        return False

    # Compute expected signature
    signed_payload = f"{timestamp}.{payload.decode('utf-8')}"
    expected_sig = hmac.new(
        secret.encode('utf-8'),
        signed_payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    # Constant-time comparison
    expected = f"sha256={expected_sig}"
    return hmac.compare_digest(expected, signature)
```

**Rust Verification:**

```rust
use hmac::{Hmac, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

pub fn verify_webhook(
    payload: &[u8],
    signature: &str,
    timestamp: &str,
    secret: &str,
) -> bool {
    // Check timestamp freshness (5 minute window)
    let ts: i64 = timestamp.parse().unwrap_or(0);
    let now = chrono::Utc::now().timestamp();
    if (now - ts).abs() > 300 {
        return false;
    }

    // Compute signature
    let signed_payload = format!(
        "{}.{}",
        timestamp,
        String::from_utf8_lossy(payload)
    );

    let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
        .expect("HMAC can take key of any size");
    mac.update(signed_payload.as_bytes());

    let expected = format!("sha256={}", hex::encode(mac.finalize().into_bytes()));

    // Constant-time comparison
    constant_time_eq::constant_time_eq(
        expected.as_bytes(),
        signature.as_bytes()
    )
}
```

#### Security Requirements

| Requirement                   | Enforcement                                                |
| ----------------------------- | ---------------------------------------------------------- |
| **TLS Required**              | Webhook URLs must use HTTPS (except localhost for testing) |
| **Signature Verification**    | Recipients MUST verify HMAC signatures                     |
| **Timestamp Validation**      | Reject events older than 5 minutes                         |
| **Secret Rotation**           | Support for rotating webhook secrets without downtime      |
| **IP Allowlist** (Enterprise) | Optional IP range restrictions                             |

---

## 4. DELIVERY SYSTEM

### 4.1 Delivery Guarantees

| Guarantee         | Implementation                                                         |
| ----------------- | ---------------------------------------------------------------------- |
| **At-Least-Once** | Events may be delivered multiple times                                 |
| **Ordering**      | Events ordered within a single execution; no global ordering guarantee |
| **Idempotency**   | Use `idempotency_key` to deduplicate                                   |
| **Durability**    | Events persisted before delivery attempt                               |

### 4.2 Retry Policy

```
+-------------------------------------------------------------------+
|                        RETRY STRATEGY                              |
+-------------------------------------------------------------------+
|                                                                    |
|  Attempt 1: Immediate                                              |
|  Attempt 2: After 1 second                                         |
|  Attempt 3: After 4 seconds    (exponential backoff with jitter)   |
|  Attempt 4: After 16 seconds                                       |
|  Attempt 5: After 64 seconds                                       |
|  Attempt 6: After 256 seconds (~4 minutes)                         |
|                                                                    |
|  Maximum total wait: ~5 minutes 41 seconds                         |
|                                                                    |
|  Formula: delay = min(initial * 2^attempt, max_delay) + jitter     |
|  Where jitter = random(0, delay * 0.1)                             |
|                                                                    |
+-------------------------------------------------------------------+
```

**Retry Configuration Options:**

```json
{
  "retry_policy": {
    "strategy": "exponential",
    "max_retries": 5,
    "initial_delay_ms": 1000,
    "max_delay_ms": 60000,
    "jitter": true
  }
}
```

**Alternative Strategies:**

| Strategy      | Description                    |
| ------------- | ------------------------------ |
| `exponential` | Exponential backoff (default)  |
| `linear`      | Linear increase: 1s, 2s, 3s... |
| `fixed`       | Fixed delay between retries    |
| `none`        | No retries                     |

### 4.3 Failure Handling

#### Response Code Handling

| Response Code    | Action                               |
| ---------------- | ------------------------------------ |
| 2xx              | Success - No retry                   |
| 3xx              | Follow redirects (max 3)             |
| 408, 429         | Rate limited - Retry with backoff    |
| 4xx (other)      | Client error - No retry, log warning |
| 5xx              | Server error - Retry                 |
| Timeout (30s)    | Retry                                |
| Connection error | Retry                                |

#### Dead Letter Queue (DLQ)

Failed events after all retries are moved to DLQ:

```bash
# View DLQ items
rk-core webhooks dlq list --webhook-id <id>

# Replay single event
rk-core webhooks dlq replay <delivery_id>

# Replay all DLQ items
rk-core webhooks dlq replay-all --webhook-id <id>

# Purge DLQ
rk-core webhooks dlq purge --webhook-id <id> --before 2025-12-01
```

**DLQ Event Structure:**

```json
{
  "delivery_id": "del_01HZK4J8X2YNMGQP3R5T7W9VC1",
  "webhook_id": "whk_01HZK4J8X2YNMGQP3R5T7W9VBX",
  "event": {
    "id": "evt_01HZK4J8X2YNMGQP3R5T7W9VBY",
    "type": "execution.completed",
    "data": {}
  },
  "attempts": [
    {
      "attempt": 1,
      "timestamp": "2025-12-28T15:30:31.000Z",
      "response_code": 500,
      "error": "Internal Server Error"
    },
    {
      "attempt": 5,
      "timestamp": "2025-12-28T15:35:45.000Z",
      "response_code": null,
      "error": "Connection timeout"
    }
  ],
  "dead_lettered_at": "2025-12-28T15:35:45.000Z",
  "expires_at": "2025-12-31T15:35:45.000Z"
}
```

### 4.4 Timeout Configuration

```json
{
  "timeout": {
    "connect_timeout_ms": 5000,
    "response_timeout_ms": 30000,
    "total_timeout_ms": 35000
  }
}
```

---

## 5. INTEGRATION TEMPLATES

### 5.1 Slack Integration

#### Setup via CLI

```bash
# Add Slack webhook
rk-core integrations add slack \
  --webhook-url https://hooks.slack.com/services/T00/B00/xxx \
  --channel "#reasonkit-alerts" \
  --events "execution.failed,usage.limit_*" \
  --mention-on-failure "@oncall"
```

#### Configuration

```json
{
  "type": "slack",
  "webhook_url": "https://hooks.slack.com/services/xxx",
  "channel": "#reasonkit-alerts",
  "events": ["execution.failed", "execution.timeout", "usage.limit_exceeded"],
  "formatting": {
    "include_trace_link": true,
    "include_metrics": true,
    "color_by_status": true
  },
  "mentions": {
    "on_failure": ["@oncall", "@platform-team"],
    "on_usage_exceeded": ["@billing"]
  }
}
```

#### Message Format

```
+---------------------------------------------+
| ReasonKit: Execution Failed                  |
+---------------------------------------------+
| Execution: exec_01HZK4...                   |
| Protocol: powercombo (paranoid)             |
| Error: VALIDATION_FAILED                    |
|                                             |
| Source triangulation could not verify claim |
|                                             |
| Failed at step: ProofGuard (3/5)            |
| Duration: 15.23s                            |
| Tokens used: 4,850                          |
|                                             |
| [View Trace] [View Execution]               |
+---------------------------------------------+
| @oncall @platform-team                      |
+---------------------------------------------+
```

### 5.2 Discord Integration

```bash
rk-core integrations add discord \
  --webhook-url https://discord.com/api/webhooks/xxx \
  --events "execution.*" \
  --thread-per-execution
```

**Rich Embed Format:**

```json
{
  "embeds": [
    {
      "title": "Execution Completed",
      "color": 5763719,
      "fields": [
        { "name": "Protocol", "value": "PowerCombo", "inline": true },
        { "name": "Profile", "value": "balanced", "inline": true },
        { "name": "Confidence", "value": "91%", "inline": true },
        { "name": "Duration", "value": "30.4s", "inline": true },
        { "name": "Tokens", "value": "10,370", "inline": true },
        { "name": "Cost", "value": "$0.016", "inline": true }
      ],
      "footer": { "text": "exec_01HZK4J8X2..." },
      "timestamp": "2025-12-28T15:30:30.000Z"
    }
  ]
}
```

### 5.3 Email Notifications

```bash
rk-core integrations add email \
  --recipients "team@example.com,alerts@example.com" \
  --events "execution.failed,usage.limit_exceeded" \
  --digest hourly \
  --smtp-host smtp.example.com \
  --smtp-port 587
```

**Configuration:**

```json
{
  "type": "email",
  "smtp": {
    "host": "smtp.example.com",
    "port": 587,
    "username": "noreply@example.com",
    "password_env": "SMTP_PASSWORD",
    "tls": true
  },
  "recipients": {
    "default": ["team@example.com"],
    "on_failure": ["oncall@example.com"],
    "on_usage": ["billing@example.com"]
  },
  "digest": {
    "enabled": true,
    "frequency": "hourly",
    "include_summary": true
  },
  "templates": {
    "subject": "[ReasonKit] {{event_type}}: {{execution_id}}",
    "from_name": "ReasonKit Alerts"
  }
}
```

### 5.4 PagerDuty Integration

```bash
rk-core integrations add pagerduty \
  --routing-key xxxxx \
  --events "execution.failed,execution.timeout" \
  --severity-mapping '{"execution.failed":"critical","execution.timeout":"error"}'
```

**Configuration:**

```json
{
  "type": "pagerduty",
  "routing_key_env": "PAGERDUTY_ROUTING_KEY",
  "events": ["execution.failed", "execution.timeout"],
  "severity_mapping": {
    "execution.failed": "critical",
    "execution.timeout": "error",
    "usage.limit_exceeded": "warning"
  },
  "dedup_key_template": "{{execution_id}}",
  "auto_resolve": {
    "enabled": true,
    "on_event": "execution.completed"
  },
  "custom_details": {
    "service": "reasonkit",
    "environment": "production"
  }
}
```

**Event Action Mapping:**

| ReasonKit Event        | PagerDuty Action | Severity |
| ---------------------- | ---------------- | -------- |
| `execution.failed`     | trigger          | critical |
| `execution.timeout`    | trigger          | error    |
| `execution.completed`  | resolve          | -        |
| `usage.limit_exceeded` | trigger          | warning  |

### 5.5 Datadog Integration (Enterprise)

```json
{
  "type": "datadog",
  "api_key_env": "DD_API_KEY",
  "site": "datadoghq.com",
  "events": ["*"],
  "metrics": {
    "enabled": true,
    "prefix": "reasonkit",
    "tags": ["env:production", "service:reasonkit"]
  },
  "logs": {
    "enabled": true,
    "source": "reasonkit",
    "service": "reasonkit-pro"
  }
}
```

### 5.6 Custom Webhook

For custom integrations, use the generic webhook:

```bash
rk-core webhooks add \
  --url https://your-api.com/webhook \
  --events "*" \
  --secret $WEBHOOK_SECRET \
  --headers "X-Custom-Auth:token123" \
  --transform-template ./transform.jq
```

**Custom Transform (JQ Template):**

```jq
# transform.jq
{
  "event": .type,
  "timestamp": .created_at,
  "execution": .data.execution_id,
  "success": (.data.status == "completed"),
  "metrics": {
    "duration": .data.duration_ms,
    "tokens": .data.tokens.total
  }
}
```

---

## 6. FILTERING AND ROUTING

### 6.1 Event Filters

Filter events before delivery based on conditions:

```json
{
  "filter": {
    "type": "and",
    "conditions": [
      {
        "field": "data.profile",
        "operator": "in",
        "value": ["paranoid", "deep"]
      },
      {
        "field": "data.status",
        "operator": "ne",
        "value": "completed"
      },
      {
        "field": "data.duration_ms",
        "operator": "gt",
        "value": 60000
      }
    ]
  }
}
```

**Supported Operators:**

| Operator   | Description        | Example                                                                  |
| ---------- | ------------------ | ------------------------------------------------------------------------ |
| `eq`       | Equal              | `{"field": "status", "operator": "eq", "value": "failed"}`               |
| `ne`       | Not equal          | `{"field": "status", "operator": "ne", "value": "completed"}`            |
| `gt`       | Greater than       | `{"field": "duration_ms", "operator": "gt", "value": 1000}`              |
| `gte`      | Greater or equal   | `{"field": "confidence", "operator": "gte", "value": 0.9}`               |
| `lt`       | Less than          | `{"field": "tokens", "operator": "lt", "value": 10000}`                  |
| `lte`      | Less or equal      | `{"field": "step_count", "operator": "lte", "value": 3}`                 |
| `in`       | In list            | `{"field": "profile", "operator": "in", "value": ["paranoid", "deep"]}`  |
| `contains` | Contains substring | `{"field": "error.message", "operator": "contains", "value": "timeout"}` |
| `regex`    | Regex match        | `{"field": "protocol_id", "operator": "regex", "value": "^power.*"}`     |
| `exists`   | Field exists       | `{"field": "data.error", "operator": "exists", "value": true}`           |

### 6.2 Routing Rules

Route different events to different endpoints:

```json
{
  "routes": [
    {
      "name": "failures-to-pagerduty",
      "events": ["execution.failed", "execution.timeout"],
      "filter": {
        "field": "data.profile",
        "operator": "in",
        "value": ["paranoid", "production"]
      },
      "destination": "integration_pagerduty",
      "priority": 1
    },
    {
      "name": "all-to-slack",
      "events": ["execution.*"],
      "destination": "integration_slack",
      "priority": 2
    },
    {
      "name": "usage-to-email",
      "events": ["usage.*"],
      "destination": "integration_email",
      "priority": 3
    }
  ]
}
```

### 6.3 Event Transformation

Transform events before delivery:

```json
{
  "transform": {
    "type": "jq",
    "expression": "{event: .type, status: .data.status, confidence: (.data.final_confidence * 100 | floor)}"
  }
}
```

---

## 7. MANAGEMENT API

### 7.1 REST Endpoints

```
+-------------------------------------------------------------------+
|                     WEBHOOK MANAGEMENT API                         |
+-------------------------------------------------------------------+
|                                                                    |
|  POST   /api/v1/webhooks              Create webhook               |
|  GET    /api/v1/webhooks              List webhooks                |
|  GET    /api/v1/webhooks/:id          Get webhook details          |
|  PUT    /api/v1/webhooks/:id          Update webhook               |
|  DELETE /api/v1/webhooks/:id          Delete webhook               |
|  POST   /api/v1/webhooks/:id/test     Send test event              |
|  POST   /api/v1/webhooks/:id/rotate   Rotate secret                |
|                                                                    |
|  GET    /api/v1/webhooks/:id/deliveries                           |
|                                       List delivery history        |
|  POST   /api/v1/webhooks/:id/deliveries/:did/replay               |
|                                       Replay delivery              |
|                                                                    |
|  GET    /api/v1/webhooks/:id/dlq      List dead-lettered events   |
|  POST   /api/v1/webhooks/:id/dlq/replay                           |
|                                       Replay all DLQ               |
|  DELETE /api/v1/webhooks/:id/dlq      Purge DLQ                   |
|                                                                    |
|  GET    /api/v1/events                List all events              |
|  GET    /api/v1/events/:id            Get event details            |
|                                                                    |
+-------------------------------------------------------------------+
```

### 7.2 API Examples

#### Create Webhook

```http
POST /api/v1/webhooks
Content-Type: application/json
Authorization: Bearer rk_live_xxx

{
  "url": "https://example.com/webhook",
  "events": ["execution.completed", "execution.failed"],
  "secret": "whsec_your_secret",
  "enabled": true
}
```

#### List Webhooks

```http
GET /api/v1/webhooks?page=1&per_page=20
Authorization: Bearer rk_live_xxx
```

Response:

```json
{
  "data": [
    {
      "id": "whk_01HZK4J8X2YNMGQP3R5T7W9VBX",
      "url": "https://example.com/webhook",
      "events": ["execution.completed", "execution.failed"],
      "enabled": true,
      "created_at": "2025-12-28T10:00:00.000Z",
      "last_delivery": {
        "at": "2025-12-28T15:30:31.000Z",
        "status": "success"
      },
      "stats": {
        "total_deliveries": 1523,
        "success_rate": 0.987,
        "avg_latency_ms": 245
      }
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 3
  }
}
```

#### View Delivery History

```http
GET /api/v1/webhooks/whk_xxx/deliveries?status=failed&limit=10
Authorization: Bearer rk_live_xxx
```

Response:

```json
{
  "data": [
    {
      "id": "del_01HZK4J8X2YNMGQP3R5T7W9VC2",
      "event_id": "evt_01HZK4J8X2YNMGQP3R5T7W9VBY",
      "event_type": "execution.failed",
      "status": "failed",
      "attempts": 5,
      "created_at": "2025-12-28T15:30:31.000Z",
      "last_attempt_at": "2025-12-28T15:35:45.000Z",
      "last_error": "Connection timeout",
      "dead_lettered": true
    }
  ]
}
```

### 7.3 CLI Commands Reference

```bash
# ============================================================
# WEBHOOK MANAGEMENT
# ============================================================

# Create webhook
rk-core webhooks add --url <URL> --events <EVENTS> --secret <SECRET>

# List webhooks
rk-core webhooks list [--json]

# Get webhook details
rk-core webhooks get <WEBHOOK_ID>

# Update webhook
rk-core webhooks update <WEBHOOK_ID> [--url <URL>] [--events <EVENTS>] [--enabled <BOOL>]

# Delete webhook
rk-core webhooks delete <WEBHOOK_ID>

# Test webhook
rk-core webhooks test <WEBHOOK_ID> [--event-type <TYPE>]

# Rotate webhook secret
rk-core webhooks rotate-secret <WEBHOOK_ID>

# ============================================================
# DELIVERY MANAGEMENT
# ============================================================

# View delivery logs
rk-core webhooks logs <WEBHOOK_ID> [--limit <N>] [--status <STATUS>]

# Replay delivery
rk-core webhooks replay <DELIVERY_ID>

# ============================================================
# DEAD LETTER QUEUE
# ============================================================

# List DLQ items
rk-core webhooks dlq list --webhook-id <WEBHOOK_ID>

# Replay single DLQ item
rk-core webhooks dlq replay <DELIVERY_ID>

# Replay all DLQ items for webhook
rk-core webhooks dlq replay-all --webhook-id <WEBHOOK_ID>

# Purge DLQ
rk-core webhooks dlq purge --webhook-id <WEBHOOK_ID> [--before <DATE>]

# ============================================================
# INTEGRATIONS
# ============================================================

# Add Slack integration
rk-core integrations add slack --webhook-url <URL> --events <EVENTS>

# Add Discord integration
rk-core integrations add discord --webhook-url <URL> --events <EVENTS>

# Add PagerDuty integration
rk-core integrations add pagerduty --routing-key <KEY> --events <EVENTS>

# Add email integration
rk-core integrations add email --recipients <EMAILS> --events <EVENTS>

# List integrations
rk-core integrations list

# Remove integration
rk-core integrations remove <INTEGRATION_ID>

# ============================================================
# EVENT BROWSING
# ============================================================

# List recent events
rk-core events list [--type <TYPE>] [--limit <N>]

# Get event details
rk-core events get <EVENT_ID>

# Stream events (live)
rk-core events stream [--types <TYPES>]
```

---

## 8. MONITORING

### 8.1 Metrics

The webhook system exposes Prometheus metrics:

```
# ============================================================
# WEBHOOK METRICS
# ============================================================

# Delivery counters
reasonkit_webhook_deliveries_total{webhook_id, event_type, status}
reasonkit_webhook_retries_total{webhook_id, attempt}

# Latency histograms
reasonkit_webhook_delivery_duration_seconds{webhook_id, event_type}

# Current state gauges
reasonkit_webhook_dlq_size{webhook_id}
reasonkit_webhook_pending_deliveries{webhook_id}

# Rate metrics
reasonkit_webhook_delivery_rate{webhook_id}
reasonkit_webhook_success_rate{webhook_id}

# ============================================================
# EVENT METRICS
# ============================================================

# Event counters
reasonkit_events_total{type}
reasonkit_events_by_hour{type, hour}

# Processing metrics
reasonkit_event_processing_duration_seconds{type}
reasonkit_event_queue_size
```

### 8.2 Logging

Structured logs for all webhook operations:

```json
{
  "timestamp": "2025-12-28T15:30:31.000Z",
  "level": "info",
  "message": "Webhook delivery successful",
  "webhook_id": "whk_01HZK4J8X2YNMGQP3R5T7W9VBX",
  "delivery_id": "del_01HZK4J8X2YNMGQP3R5T7W9VC2",
  "event_id": "evt_01HZK4J8X2YNMGQP3R5T7W9VBY",
  "event_type": "execution.completed",
  "attempt": 1,
  "response_code": 200,
  "duration_ms": 245,
  "payload_size_bytes": 1523
}
```

**Log Levels:**

| Level   | Use Case                                     |
| ------- | -------------------------------------------- |
| `debug` | Detailed delivery attempts, payload contents |
| `info`  | Successful deliveries, webhook changes       |
| `warn`  | Retry attempts, approaching limits           |
| `error` | Delivery failures, DLQ additions             |

### 8.3 Alerting

Built-in alert conditions:

```yaml
# Alert: High webhook failure rate
- name: webhook_high_failure_rate
  condition: >
    rate(reasonkit_webhook_deliveries_total{status="failed"}[5m]) /
    rate(reasonkit_webhook_deliveries_total[5m]) > 0.1
  for: 5m
  severity: warning
  annotations:
    summary: "Webhook failure rate above 10%"

# Alert: DLQ growing
- name: webhook_dlq_growing
  condition: >
    increase(reasonkit_webhook_dlq_size[1h]) > 100
  for: 15m
  severity: warning
  annotations:
    summary: "Webhook DLQ growing rapidly"

# Alert: Webhook endpoint down
- name: webhook_endpoint_down
  condition: >
    increase(reasonkit_webhook_deliveries_total{status="failed"}[15m]) > 10 and
    rate(reasonkit_webhook_deliveries_total{status="success"}[15m]) == 0
  for: 10m
  severity: critical
  annotations:
    summary: "Webhook endpoint appears down"
```

---

## 9. RATE LIMITING

### 9.1 Per-Endpoint Limits

```json
{
  "rate_limits": {
    "per_endpoint": {
      "events_per_minute": 100,
      "events_per_hour": 3000,
      "burst_size": 50
    },
    "global": {
      "events_per_minute": 1000,
      "events_per_hour": 30000
    }
  }
}
```

### 9.2 Backpressure Handling

When rate limits are exceeded:

1. Events are queued (up to queue limit)
2. Delivery slowed to respect limits
3. Oldest events prioritized
4. `usage.rate_limit_hit` event emitted

```json
{
  "id": "evt_01HZK4J8X2YNMGQP3R5T7W9VC3",
  "type": "usage.rate_limit_hit",
  "data": {
    "webhook_id": "whk_01HZK4J8X2YNMGQP3R5T7W9VBX",
    "limit_type": "events_per_minute",
    "limit_value": 100,
    "current_value": 150,
    "queued_events": 50,
    "oldest_queued_age_ms": 45000
  }
}
```

---

## 10. EVENT STREAMING (Enterprise)

### 10.1 WebSocket API

Real-time event streaming for Enterprise customers:

```javascript
// JavaScript Example
const ws = new WebSocket("wss://api.reasonkit.sh/v1/events/stream");

ws.onopen = () => {
  // Authenticate
  ws.send(
    JSON.stringify({
      type: "auth",
      token: "rk_live_xxx",
    }),
  );

  // Subscribe to events
  ws.send(
    JSON.stringify({
      type: "subscribe",
      events: ["execution.*", "trace.*"],
      filter: {
        project_id: "prj_xyz789",
      },
    }),
  );
};

ws.onmessage = (message) => {
  const event = JSON.parse(message.data);
  console.log("Received:", event.type, event.data);
};
```

**WebSocket Protocol:**

```
Client -> Server:
  { "type": "auth", "token": "rk_live_xxx" }
  { "type": "subscribe", "events": ["execution.*"], "filter": {} }
  { "type": "unsubscribe", "subscription_id": "sub_xxx" }
  { "type": "ping" }

Server -> Client:
  { "type": "auth_success", "user_id": "usr_xxx" }
  { "type": "subscribed", "subscription_id": "sub_xxx" }
  { "type": "event", "subscription_id": "sub_xxx", "event": {...} }
  { "type": "pong" }
  { "type": "error", "code": "AUTH_FAILED", "message": "..." }
```

**Reconnection Handling:**

- Automatic reconnection with exponential backoff
- Resume from last received event using `last_event_id`
- Server maintains 24-hour event history for replay

### 10.2 Server-Sent Events (SSE)

Simpler alternative to WebSocket:

```bash
curl -N -H "Authorization: Bearer rk_live_xxx" \
  "https://api.reasonkit.sh/v1/events/stream?events=execution.*"
```

**Response Stream:**

```
event: execution.started
id: evt_01HZK4J8X2YNMGQP3R5T7W9VBY
data: {"execution_id":"exec_xxx","protocol_id":"powercombo",...}

event: execution.completed
id: evt_01HZK4J8X2YNMGQP3R5T7W9VBZ
data: {"execution_id":"exec_xxx","status":"completed",...}
```

---

## 11. RUST TYPE DEFINITIONS

### 11.1 Event Types

```rust
//! Webhook event types for ReasonKit

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Base event structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookEvent {
    /// Unique event identifier
    pub id: String,

    /// Event type (e.g., "execution.completed")
    #[serde(rename = "type")]
    pub event_type: String,

    /// API version for this event
    pub api_version: String,

    /// When the event was created
    pub created_at: DateTime<Utc>,

    /// Idempotency key for deduplication
    pub idempotency_key: String,

    /// Webhook ID this event is being sent to
    #[serde(skip_serializing_if = "Option::is_none")]
    pub webhook_id: Option<String>,

    /// Delivery attempt number
    #[serde(default = "default_attempt")]
    pub delivery_attempt: u32,

    /// Event-specific data
    pub data: serde_json::Value,
}

fn default_attempt() -> u32 { 1 }

/// Execution event types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status")]
pub enum ExecutionEventData {
    #[serde(rename = "started")]
    Started(ExecutionStartedData),

    #[serde(rename = "completed")]
    Completed(ExecutionCompletedData),

    #[serde(rename = "failed")]
    Failed(ExecutionFailedData),

    #[serde(rename = "timeout")]
    Timeout(ExecutionTimeoutData),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStartedData {
    pub execution_id: String,
    pub protocol_id: String,
    pub protocol_version: String,
    pub profile: String,
    pub input_tokens: u64,
    pub estimated_steps: u32,
    pub user_id: Option<String>,
    pub project_id: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionCompletedData {
    pub execution_id: String,
    pub protocol_id: String,
    pub profile: String,
    pub final_confidence: f64,
    pub duration_ms: u64,
    pub total_steps: u32,
    pub completed_steps: u32,
    pub tokens: TokenUsage,
    pub quality_metrics: Option<QualityMetrics>,
    pub trace_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionFailedData {
    pub execution_id: String,
    pub protocol_id: String,
    pub profile: String,
    pub failed_at_step: u32,
    pub step_id: String,
    pub error: ExecutionError,
    pub partial_result: Option<PartialResult>,
    pub duration_ms: u64,
    pub tokens_used: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTimeoutData {
    pub execution_id: String,
    pub protocol_id: String,
    pub profile: String,
    pub timeout_ms: u64,
    pub actual_duration_ms: u64,
    pub last_step_index: u32,
    pub last_step_id: String,
    pub completed_steps: u32,
    pub partial_confidence: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input: u64,
    pub output: u64,
    pub total: u64,
    pub estimated_cost_usd: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub coherence_score: f64,
    pub depth_score: f64,
    pub consistency_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionError {
    pub code: String,
    pub category: ErrorCategory,
    pub message: String,
    pub recoverable: bool,
    pub details: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCategory {
    Reasoning,
    Validation,
    Network,
    RateLimit,
    Internal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialResult {
    pub completed_steps: u32,
    pub last_confidence: f64,
}
```

### 11.2 Webhook Configuration

```rust
//! Webhook configuration types

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use url::Url;

/// Webhook endpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Webhook {
    /// Unique webhook identifier
    pub id: String,

    /// Destination URL (must be HTTPS)
    pub url: Url,

    /// Human-readable description
    pub description: Option<String>,

    /// Events to subscribe to (supports wildcards)
    pub events: Vec<String>,

    /// HMAC signing secret (only prefix exposed)
    #[serde(skip_serializing)]
    pub secret: String,

    /// Whether webhook is enabled
    pub enabled: bool,

    /// Event filter conditions
    pub filter: Option<EventFilter>,

    /// Retry policy configuration
    pub retry_policy: RetryPolicy,

    /// Custom headers to include
    pub headers: Option<std::collections::HashMap<String, String>>,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Last modification timestamp
    pub updated_at: DateTime<Utc>,

    /// Owner user ID
    pub user_id: String,

    /// Associated project ID
    pub project_id: Option<String>,
}

/// Event filter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventFilter {
    #[serde(rename = "type")]
    pub filter_type: FilterType,
    pub conditions: Vec<FilterCondition>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FilterType {
    And,
    Or,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterCondition {
    pub field: String,
    pub operator: FilterOperator,
    pub value: serde_json::Value,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FilterOperator {
    Eq,
    Ne,
    Gt,
    Gte,
    Lt,
    Lte,
    In,
    Contains,
    Regex,
    Exists,
}

/// Retry policy for failed deliveries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Retry strategy
    pub strategy: RetryStrategy,

    /// Maximum retry attempts
    pub max_retries: u32,

    /// Initial delay in milliseconds
    pub initial_delay_ms: u64,

    /// Maximum delay in milliseconds
    pub max_delay_ms: u64,

    /// Whether to add jitter
    pub jitter: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            strategy: RetryStrategy::Exponential,
            max_retries: 5,
            initial_delay_ms: 1000,
            max_delay_ms: 60000,
            jitter: true,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetryStrategy {
    Exponential,
    Linear,
    Fixed,
    None,
}

/// Webhook delivery record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookDelivery {
    pub id: String,
    pub webhook_id: String,
    pub event_id: String,
    pub event_type: String,
    pub status: DeliveryStatus,
    pub attempts: Vec<DeliveryAttempt>,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub dead_lettered: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryStatus {
    Pending,
    InProgress,
    Success,
    Failed,
    DeadLettered,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryAttempt {
    pub attempt: u32,
    pub timestamp: DateTime<Utc>,
    pub response_code: Option<u16>,
    pub duration_ms: u64,
    pub error: Option<String>,
}
```

### 11.3 Webhook Dispatcher

```rust
//! Webhook delivery system

use std::time::Duration;
use tokio::time::sleep;

/// Webhook delivery service
pub struct WebhookDispatcher {
    client: reqwest::Client,
    config: DispatcherConfig,
}

pub struct DispatcherConfig {
    pub connect_timeout: Duration,
    pub response_timeout: Duration,
    pub max_concurrent_deliveries: usize,
}

impl WebhookDispatcher {
    /// Create new dispatcher
    pub fn new(config: DispatcherConfig) -> Self {
        let client = reqwest::Client::builder()
            .connect_timeout(config.connect_timeout)
            .timeout(config.response_timeout)
            .build()
            .expect("Failed to build HTTP client");

        Self { client, config }
    }

    /// Deliver event to webhook
    pub async fn deliver(
        &self,
        webhook: &Webhook,
        event: &WebhookEvent,
    ) -> Result<DeliveryResult, DeliveryError> {
        let payload = serde_json::to_vec(event)?;
        let timestamp = chrono::Utc::now().timestamp().to_string();
        let signature = self.compute_signature(&payload, &timestamp, &webhook.secret);

        let mut attempt = 0;
        let mut last_error = None;

        while attempt <= webhook.retry_policy.max_retries {
            if attempt > 0 {
                let delay = self.compute_delay(&webhook.retry_policy, attempt);
                sleep(delay).await;
            }

            let result = self.try_deliver(
                &webhook.url,
                &payload,
                &signature,
                &timestamp,
                &webhook.headers,
                event,
            ).await;

            match result {
                Ok(response) if response.status().is_success() => {
                    return Ok(DeliveryResult::Success {
                        attempt: attempt + 1,
                        status_code: response.status().as_u16(),
                    });
                }
                Ok(response) if !self.should_retry(response.status().as_u16()) => {
                    return Err(DeliveryError::PermanentFailure {
                        status_code: response.status().as_u16(),
                        attempt: attempt + 1,
                    });
                }
                Ok(response) => {
                    last_error = Some(format!("HTTP {}", response.status()));
                }
                Err(e) => {
                    last_error = Some(e.to_string());
                }
            }

            attempt += 1;
        }

        Err(DeliveryError::MaxRetriesExceeded {
            attempts: attempt,
            last_error: last_error.unwrap_or_else(|| "Unknown".to_string()),
        })
    }

    async fn try_deliver(
        &self,
        url: &url::Url,
        payload: &[u8],
        signature: &str,
        timestamp: &str,
        custom_headers: &Option<std::collections::HashMap<String, String>>,
        event: &WebhookEvent,
    ) -> Result<reqwest::Response, reqwest::Error> {
        let mut request = self.client
            .post(url.as_str())
            .header("Content-Type", "application/json")
            .header("X-ReasonKit-Signature", signature)
            .header("X-ReasonKit-Timestamp", timestamp)
            .header("X-ReasonKit-Event", &event.event_type)
            .header("X-ReasonKit-Delivery-Id", &event.id)
            .body(payload.to_vec());

        if let Some(headers) = custom_headers {
            for (key, value) in headers {
                request = request.header(key, value);
            }
        }

        request.send().await
    }

    fn compute_signature(&self, payload: &[u8], timestamp: &str, secret: &str) -> String {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;

        let signed_payload = format!(
            "{}.{}",
            timestamp,
            String::from_utf8_lossy(payload)
        );

        let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(signed_payload.as_bytes());

        format!("sha256={}", hex::encode(mac.finalize().into_bytes()))
    }

    fn compute_delay(&self, policy: &RetryPolicy, attempt: u32) -> Duration {
        let base_delay = match policy.strategy {
            RetryStrategy::Exponential => {
                policy.initial_delay_ms * 2u64.pow(attempt.saturating_sub(1))
            }
            RetryStrategy::Linear => {
                policy.initial_delay_ms * attempt as u64
            }
            RetryStrategy::Fixed => policy.initial_delay_ms,
            RetryStrategy::None => 0,
        };

        let delay = base_delay.min(policy.max_delay_ms);

        let delay = if policy.jitter {
            use rand::Rng;
            let jitter = rand::thread_rng().gen_range(0..=(delay / 10));
            delay + jitter
        } else {
            delay
        };

        Duration::from_millis(delay)
    }

    fn should_retry(&self, status_code: u16) -> bool {
        matches!(status_code, 408 | 429 | 500..=599)
    }
}

pub enum DeliveryResult {
    Success {
        attempt: u32,
        status_code: u16,
    },
}

pub enum DeliveryError {
    SerializationError(serde_json::Error),
    PermanentFailure {
        status_code: u16,
        attempt: u32,
    },
    MaxRetriesExceeded {
        attempts: u32,
        last_error: String,
    },
}

impl From<serde_json::Error> for DeliveryError {
    fn from(e: serde_json::Error) -> Self {
        DeliveryError::SerializationError(e)
    }
}
```

---

## 12. DOCUMENTATION

### 12.1 Quick Start Guide

````markdown
# Webhook Quick Start

## 1. Create a Webhook

```bash
rk-core webhooks add \
  --url https://your-server.com/webhook \
  --events "execution.completed,execution.failed" \
  --secret $(openssl rand -hex 32)
```
````

## 2. Verify Signatures (Python Example)

```python
import hmac
import hashlib

def verify_signature(payload, signature, timestamp, secret):
    signed_payload = f"{timestamp}.{payload}"
    expected = hmac.new(
        secret.encode(),
        signed_payload.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)
```

## 3. Handle Events

```python
from flask import Flask, request

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    # Verify signature
    signature = request.headers.get('X-ReasonKit-Signature')
    timestamp = request.headers.get('X-ReasonKit-Timestamp')

    if not verify_signature(request.data.decode(), signature, timestamp, SECRET):
        return 'Invalid signature', 401

    event = request.json

    if event['type'] == 'execution.completed':
        # Handle successful execution
        print(f"Execution {event['data']['execution_id']} completed")
    elif event['type'] == 'execution.failed':
        # Handle failure
        print(f"Execution failed: {event['data']['error']['message']}")

    return 'OK', 200
```

```

### 12.2 Event Reference

Full event documentation is available at:
- API Docs: https://docs.reasonkit.sh/webhooks/events
- OpenAPI: https://api.reasonkit.sh/v1/openapi.json

---

## VERSION HISTORY

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-28 | Initial design specification |

---

*"Real-time insights into AI reasoning."*
*ReasonKit Team*
```
