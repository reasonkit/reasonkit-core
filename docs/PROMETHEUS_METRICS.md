# Prometheus Metrics Integration

ReasonKit Core exposes a comprehensive set of Prometheus-compatible metrics for monitoring system health, performance, and usage.

## Configuration

To enable the metrics endpoint, update your `config.toml`:

```toml
[metrics]
enabled = true
endpoint = "/metrics"
port = 9090  # Optional: dedicated port for metrics
```

## Available Metrics

### Execution Metrics

| Metric Name                            | Type      | Description                         | Labels                |
| -------------------------------------- | --------- | ----------------------------------- | --------------------- |
| `reasonkit_execution_total`            | Counter   | Total number of protocol executions | `protocol`, `status`  |
| `reasonkit_execution_duration_seconds` | Histogram | Time taken to execute protocols     | `protocol`            |
| `reasonkit_step_duration_seconds`      | Histogram | Time taken for individual steps     | `step_id`, `protocol` |
| `reasonkit_confidence_score`           | Histogram | Distribution of confidence scores   | `protocol`            |

### LLM Usage Metrics

| Metric Name                     | Type      | Description              | Labels                         |
| ------------------------------- | --------- | ------------------------ | ------------------------------ |
| `reasonkit_llm_requests_total`  | Counter   | Total LLM API requests   | `provider`, `model`            |
| `reasonkit_llm_tokens_total`    | Counter   | Total tokens consumed    | `type` (input/output), `model` |
| `reasonkit_llm_errors_total`    | Counter   | Failed LLM requests      | `provider`, `error_type`       |
| `reasonkit_llm_latency_seconds` | Histogram | Latency of LLM responses | `provider`                     |

### System Metrics

| Metric Name                    | Type    | Description                           | Labels       |
| ------------------------------ | ------- | ------------------------------------- | ------------ |
| `reasonkit_memory_usage_bytes` | Gauge   | Current memory usage                  | -            |
| `reasonkit_active_executions`  | Gauge   | Number of currently running protocols | -            |
| `reasonkit_cache_hits_total`   | Counter | Semantic cache hits                   | `cache_type` |

## Grafana Dashboard Example

A standard Grafana dashboard configuration is available in `grafana/dashboard.json`. It includes:

1.  **Overview Panel:** Request rate, error rate, and average latency.
2.  **Protocol Performance:** Breakdown of duration by protocol type.
3.  **Cost Monitor:** Estimated token costs per hour.
4.  **Quality Watch:** Average confidence scores over time.

## Alerting Rules

Recommended Prometheus alerting rules:

```yaml
groups:
  - name: reasonkit-alerts
    rules:
      - alert: HighErrorRate
        expr: rate(reasonkit_execution_total{status="failed"}[5m]) / rate(reasonkit_execution_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High protocol failure rate detected"

      - alert: LowConfidence
        expr: rate(reasonkit_confidence_score_sum[1h]) / rate(reasonkit_confidence_score_count[1h]) < 0.6
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "Average reasoning confidence dropped below 60%"
```
