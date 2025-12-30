# API Rate Limiting & Usage Guide

ReasonKit Core implements robust rate limiting to ensure stability and fair usage, especially when interacting with external LLM providers. This guide covers how to configure rate limits, handle errors, and optimize your application's behavior.

## Default Limits

By default, ReasonKit applies the following limits to protect your resources and downstream providers:

| Resource                      | Default Limit    | Scope            |
| ----------------------------- | ---------------- | ---------------- |
| **Concurrent Requests**       | 10               | Per instance     |
| **Tokens per Minute (TPM)**   | Provider Default | Per provider key |
| **Requests per Minute (RPM)** | 60               | Global           |

## Configuration

You can customize rate limits in your `config.toml` file under the `[server]` and `[processing]` sections.

```toml
[server]
# Maximum number of concurrent connections
max_concurrent_requests = 20

# Global rate limit (requests per minute) per client IP
rate_limit_rpm = 60

[processing]
# Artificial delay between steps to prevent hitting provider limits
step_delay_ms = 100
```

## Handling Rate Limits (HTTP 429)

When a limit is exceeded, ReasonKit will return an HTTP `429 Too Many Requests` status code.

**Response Body:**

```json
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests. Please retry after 15 seconds.",
  "retry_after_seconds": 15
}
```

### Headers

ReasonKit includes standard rate limit headers in all responses:

- `X-RateLimit-Limit`: The maximum number of requests allowed in the current window.
- `X-RateLimit-Remaining`: The number of requests remaining in the current window.
- `X-RateLimit-Reset`: The time (in seconds) until the limit resets.

## Client-Side Best Practices

### 1. Exponential Backoff

When you receive a 429 error, **do not retry immediately**. Use an exponential backoff strategy.

**Algorithm:**

1.  Wait `retry_after_seconds` (or a base delay like 1s).
2.  Retry the request.
3.  If it fails again, wait `delay * 2`.
4.  Repeat up to a maximum number of retries (e.g., 5).

### 2. Token Budgeting

ThinkTool protocols can be token-intensive.

- **GigaThink:** ~2.5k tokens
- **ProofGuard:** ~2.2k tokens

Monitor your usage. If you are hitting provider limits (e.g., OpenAI TPM), consider:

- Reducing `max_perspectives` in GigaThink.
- Caching results (ReasonKit has built-in caching; verify it is enabled).
- Using a provider with higher limits or a dedicated throughput tier.

## Provider-Specific Notes

- **OpenAI:** Enforces strictly on TPM and RPM. ReasonKit handles the 429s from OpenAI internally and will retry automatically for internal protocol steps, but will bubble up the error if the provider is persistently unavailable.
- **Anthropic:** Similar to OpenAI, but often with stricter concurrent request limits on lower tiers.
- **Local Models (Ollama/Llama.cpp):** "Rate limiting" here effectively means queuing. If your local GPU is saturated, ReasonKit will queue requests up to `max_concurrent_requests`.

## Monitoring

Rate limit events are logged at the `WARN` level.

```text
2025-12-28T18:30:00Z WARN [reasonkit::server] Rate limit exceeded for ip=192.168.1.50
```

You can view these logs to identify usage patterns and adjust your capacity planning.
