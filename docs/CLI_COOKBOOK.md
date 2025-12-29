# ReasonKit CLI Cookbook

Practical examples for using `rk-core` in your daily workflow.

## 🧠 Structured Reasoning

### Quick Answer (Fast)
Use GigaThink for rapid brainstorming.
```bash
rk-core think "What are creative names for a Rust-based AI tool?" --profile quick
```

### Deep Analysis (Standard)
Use the standard chain (GigaThink -> LaserLogic -> BedRock) for thorough analysis.
```bash
rk-core think "Should we migrate from microservices to a monolith?" --profile balanced
```

### Maximum Rigor (Paranoid)
Apply all ThinkTools, including adversarial critique and source verification.
```bash
rk-core think "Analyze the security risks of this architecture" --profile paranoid
```

### Budget-Constrained
Limit execution to a specific time or token budget.
```bash
rk-core think "Summarize this topic" --budget 30s
rk-core think "Generate ideas" --budget 1000t
```

---

## 🌐 Deep Research

### Quick Topic Overview
```bash
rk-core web "Rust async runtime comparison" --depth quick
```

### Comprehensive Report
Deep dive with web search and knowledge base integration.
```bash
rk-core web "State of AI Agents in 2025" \
  --depth deep \
  --output report.md
```

---

## 🛡️ Truth & Verification

### Verify a Claim
Triangulate a claim against 3 independent sources.
```bash
rk-core verify "Python 3.13 removes the GIL" --sources 3
```

### Fact-Check with Knowledge Base
Cross-reference a claim against your local documents.
```bash
rk-core verify "Our architecture requires mTLS" --kb --no-web
```

---

## 🔧 Custom Protocols

### Run a Custom TOML Tool
```bash
rk-core think "Critique this design" --protocol devils_advocate
```
*(Requires `devils_advocate` to be defined in `protocols/*.toml`)*

### Debug a Protocol
See the raw execution trace for debugging.
```bash
rk-core think "Test query" --protocol my_new_tool --save-trace
rk-core trace view <trace_id>
```

---

## 📚 Knowledge Base (RAG)

### Ingest Documents
```bash
rk-core ingest ./docs --recursive --type markdown
rk-core ingest ./papers/whitepaper.pdf
```

### Query Knowledge Base
```bash
rk-core query "What is the retry policy?" --top-k 5
```

---

## ⚙️ Administration

### Check Status
```bash
rk-core stats
```

### Export Metrics
```bash
rk-core metrics report --format json > monthly_report.json
```
