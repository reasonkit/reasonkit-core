# Webhook Event Documentation

This document provides a comprehensive reference for all events emitted by the ReasonKit event system. These events can be consumed via webhooks, WebSockets (Enterprise), or Server-Sent Events (SSE).

## Event Taxonomy

All events follow the naming convention: `{category}.{action}`.

| Category | Description |
|----------|-------------|
| `execution` | Events related to the lifecycle of a reasoning protocol execution. |
| `trace` | Events related to the creation and management of reasoning traces. |
| `usage` | Events triggered by quota thresholds and usage limits. |
| `account` | Events related to user account changes (Pro/Enterprise). |
| `ingestion` | Events related to document ingestion and RAG indexing. |

## Core Execution Events

### `execution.started`
Fired when a reasoning execution begins.

**Payload:**
```json
{
  "type": "execution.started",
  "data": {
    "execution_id": "exec_01HZK...",
    "protocol_id": "gigathink",
    "protocol_version": "1.0.0",
    "profile": "balanced",
    "input_tokens": 1250
  }
}
```

### `execution.step_completed`
Fired after each atomic step within a protocol completes.

**Payload:**
```json
{
  "type": "execution.step_completed",
  "data": {
    "execution_id": "exec_01HZK...",
    "step_id": "identify_dimensions",
    "confidence": 0.87,
    "duration_ms": 4823
  }
}
```

### `execution.completed`
Fired when the entire protocol execution finishes successfully.

**Payload:**
```json
{
  "type": "execution.completed",
  "data": {
    "execution_id": "exec_01HZK...",
    "final_confidence": 0.91,
    "duration_ms": 30450,
    "tokens": {
      "total": 10370
    }
  }
}
```

### `execution.failed`
Fired when an execution fails due to an error (provider down, validation failed, etc.).

**Payload:**
```json
{
  "type": "execution.failed",
  "data": {
    "execution_id": "exec_01HZK...",
    "failed_at_step": 3,
    "error": {
      "code": "VALIDATION_FAILED",
      "message": "Source triangulation could not verify claim"
    }
  }
}
```

## Trace Events

### `trace.created`
Fired when an execution trace is persisted to storage.

### `trace.shared`
Fired when a public share link is generated for a trace.

## Usage & Billing Events

### `usage.limit_warning`
Fired when a user reaches 80% of their monthly token or request quota.

### `usage.limit_exceeded`
Fired when a quota is exhausted and requests are being throttled.

## Security (Enterprise)

### `security.api_key_used`
Fired when a new or rotated API key is first used.

---

## Event Payload Schema

All event payloads include the following top-level fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique event UUID. |
| `type` | string | The event name (e.g., `execution.completed`). |
| `created_at` | string | ISO 8601 timestamp. |
| `idempotency_key` | string | Key to prevent duplicate processing. |
| `data` | object | Event-specific data (see details above). |

## Integration

For details on how to receive these events, see the [Webhook System Guide](WEBHOOK_SYSTEM.md).
