# Telemetry Opt-In Design
> **Philosophy:** Privacy Default. Opt-In Value.

## 1. Overview

ReasonKit collects telemetry to improve the reasoning engine. However, we strictly adhere to a **Privacy First** policy. Telemetry is **Opt-In** by default for personal data, and **Anonymized** for usage metrics.

## 2. Opt-In Workflow

### 2.1 First Run Experience (`rk-core init`)

When a user runs `rk-core` for the first time, they are presented with a configuration wizard.

```text
Welcome to ReasonKit v0.1.0!

ReasonKit improves by learning from usage patterns. We would like to collect
anonymous usage data (error rates, performance metrics).

We NEVER collect:
- Your source code
- Content of your prompts
- Personal Identifiable Information (PII)

[?] Enable anonymous telemetry? (Y/n) >
```

### 2.2 Configuration (`config.toml`)

Telemetry settings are explicit in the configuration file.

```toml
[telemetry]
# Master switch. If false, absolutely nothing is sent.
enabled = true

# Sending of crash reports (stack traces)
crash_reporting = true

# Sending of performance metrics (latency, token usage)
metrics = true

# Sending of anonymized reasoning traces (for model improvement)
# This is explicitly OPT-IN and requires a separate flag.
reasoning_data = false
```

## 3. PII Stripping Implementation

Before any data packet leaves the machine, it passes through the `PrivacyFilter`.

**Mechanism:**
1.  **Regex Redaction:** Scans for Email, IP, API Key patterns (`sk-...`), Credit Cards.
2.  **Allowlist Fields:** Only specific JSON fields (`duration_ms`, `error_code`, `protocol_id`) are allowed.
3.  **Hash Salting:** User IDs are hashed with a daily rotating salt to prevent long-term tracking without consent.

## 4. Data Points Collected (If Enabled)

### Standard Telemetry
*   **Version:** `0.1.0`
*   **OS:** `linux-x86_64`
*   **Command:** `think`
*   **Protocol ID:** `gigathink`
*   **Success:** `true`
*   **Duration:** `450ms`
*   **Error Code:** `None`

### Enhanced Telemetry (Opt-In "Reasoning Data")
*   **Step Sequence:** `[analyze, critique, synthesize]`
*   **Confidence Scores:** `[0.8, 0.4, 0.9]`
*   **Token Counts:** `input: 500, output: 200`

## 5. Transparency

Users can view exactly what is being sent by running:
```bash
rk-core telemetry --dry-run
```
This prints the JSON payload to stdout instead of sending it.
