# ReasonKit CLI Cookbook

Practical examples for using `rk` in your daily workflow.

## 🧠 Structured Reasoning

### Quick Answer (Fast)

Use GigaThink for rapid brainstorming.

```bash
rk think "What are creative names for a Rust-based AI tool?" --profile quick
```

### Deep Analysis (Standard)

Use the standard chain (GigaThink -> LaserLogic -> BedRock) for thorough analysis.

```bash
rk think "Should we migrate from microservices to a monolith?" --profile balanced
```

### Maximum Rigor (Paranoid)

Apply all ThinkTools, including adversarial critique and source verification.

```bash
rk think "Analyze the security risks of this architecture" --profile paranoid
```

### Budget-Constrained

Limit execution to a specific time or token budget.

```bash
rk think "Summarize this topic" --budget 30s
rk think "Generate ideas" --budget 1000t
```

---

## 🌐 Deep Research

### Quick Topic Overview

```bash
rk web "Rust async runtime comparison" --depth quick
```

### Comprehensive Report

Deep dive with web search and knowledge base integration.

```bash
rk web "State of AI Agents in 2025" \
  --depth deep \
  --output report.md
```

---

## 🛡️ Truth & Verification

### Verify a Claim

Triangulate a claim against 3 independent sources.

```bash
rk verify "Python 3.13 removes the GIL" --sources 3
```

### Fact-Check with Knowledge Base

Cross-reference a claim against your local documents.

```bash
rk verify "Our architecture requires mTLS" --kb --no-web
```

---

## 🔧 Custom Protocols

### Run a Custom TOML Tool

```bash
rk think "Critique this design" --protocol devils_advocate
```

_(Requires `devils_advocate` to be defined in `protocols/_.toml`)\*

### Debug a Protocol

See the raw execution trace for debugging.

```bash
rk think "Test query" --protocol my_new_tool --save-trace
rk trace view <trace_id>
```

---

## 📚 Knowledge Base (RAG)

### Ingest Documents

```bash
rk ingest ./docs --recursive --type markdown
rk ingest ./papers/whitepaper.pdf
```

### Query Knowledge Base

```bash
rk query "What is the retry policy?" --top-k 5
```

---

## ⚙️ Administration

### Check Status

```bash
rk stats
```

### Export Metrics

```bash
rk metrics report --format json > monthly_report.json
```
