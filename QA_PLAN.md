# QUALITY ASSURANCE PLAN: ReasonKit Core

> Permanent Guardrails for Rust Excellence
> Version: 1.0.0 | Established: 2025-12-11

---

## EXECUTIVE SUMMARY

This document establishes **permanent, embedded quality gates** for ReasonKit-core development.
All work MUST pass these gates before merge. No exceptions.

```
DEPLOYMENT MODE (VERIFIED 4x):
├── reasonkit-core → CLI Mode (#2) ✓ CONFIRMED
└── reasonkit-pro  → Sidecar Mode (#4) ✓ CONFIRMED (future)

VERIFICATION AXES:
1. Technical Feasibility  ✓
2. User Experience       ✓
3. Operational           ✓
4. Market/Business       ✓
```

---

## 1. Rust QUALITY GATES (MANDATORY)

### Gate 1: Compilation (BLOCKING)

```bash
# MUST PASS before any PR review
cargo build --release 2>&1 | tee build.log
# Exit code MUST be 0
# Warnings count MUST be < 10 (decreasing trend required)
```

| Metric        | Threshold           | Enforcement        |
| ------------- | ------------------- | ------------------ |
| Build success | 100%                | HARD BLOCK         |
| Warnings      | < 10                | SOFT (track trend) |
| Build time    | < 60s (incremental) | MONITORING         |

### Gate 2: Linting (BLOCKING)

```bash
# MUST PASS before merge
cargo clippy -- -D warnings -W clippy::all -W clippy::pedantic
```

| Lint Category         | Action | Rationale           |
| --------------------- | ------ | ------------------- |
| `clippy::all`         | DENY   | Standard safety     |
| `clippy::pedantic`    | WARN   | Quality improvement |
| `clippy::unwrap_used` | WARN   | Production safety   |
| `clippy::expect_used` | ALLOW  | Explicit is OK      |

### Gate 3: Formatting (BLOCKING)

```bash
# MUST PASS - auto-fixable
cargo fmt --check
cargo fmt  # Fix if needed
```

### Gate 4: Testing (BLOCKING)

```bash
# MUST PASS with coverage threshold
cargo test --all-features
cargo test --release  # Also test optimized builds

# Coverage tracking (target: 80%)
cargo tarpaulin --out Html --output-dir target/coverage
```

| Metric                | Threshold | Enforcement        |
| --------------------- | --------- | ------------------ |
| Unit test pass        | 100%      | HARD BLOCK         |
| Integration test pass | 100%      | HARD BLOCK         |
| Coverage              | > 80%     | SOFT (track trend) |
| Doc tests             | 100%      | HARD BLOCK         |

### Gate 5: Benchmarks (MONITORING)

```bash
# Run on performance-critical changes
cargo bench --bench retrieval_bench

# Regression threshold: 5%
# If > 5% regression: INVESTIGATE before merge
```

| Benchmark     | Baseline | Max Regression |
| ------------- | -------- | -------------- |
| BM25 query    | TBD      | 5%             |
| Vector search | TBD      | 5%             |
| Hybrid fusion | TBD      | 5%             |

---

## 2. CODE REVIEW CHECKLIST

Every PR MUST answer these questions:

### Architecture

- [ ] Does this change align with the 5-layer architecture?
- [ ] Is the change in the correct layer?
- [ ] Are cross-layer dependencies minimized?

### Safety

- [ ] No `unsafe` blocks without explicit justification
- [ ] No `.unwrap()` in library code (use `?` or `.expect()`)
- [ ] Error types are informative and actionable
- [ ] No hardcoded secrets or credentials

### Performance

- [ ] Hot paths are allocation-free where possible
- [ ] Large data uses iterators, not collecting to Vec
- [ ] Async operations don't block the runtime
- [ ] Benchmarks added for new performance-critical code

### Documentation

- [ ] Public APIs have doc comments
- [ ] Complex logic has inline comments
- [ ] README updated if user-facing changes
- [ ] CHANGELOG entry added

---

## 3. PERIODIC QUALITY REVIEWS

### Weekly: Automated Metrics

```bash
# Run every Monday via CI
./scripts/quality_metrics.sh

# Outputs:
# - Test coverage trend
# - Warning count trend
# - Benchmark regression report
# - Dependency audit
```

### Bi-Weekly: Code Quality Review

```yaml
Checklist:
  - Review TODO/FIXME count (must decrease)
  - Check dead code (cargo +nightly udeps)
  - Verify error handling completeness
  - Audit unsafe blocks
```

### Monthly: Expert Spec Panel Review

```yaml
Trigger: /sc:spec-panel "cd reasonkit-core"

Panel:
  - Karl Wiegers (Requirements)
  - Michael Nygard (Operations)
  - Martin Fowler (Architecture)
  - Gojko Adzic (Specifications)
  - Sam Newman (Integration)

Pass Criteria:
  - Overall Score: > 7.0/10
  - Production Readiness: > 6.0/10
  - No P0 critical issues
```

### Quarterly: External Audit

```yaml
Focus Areas:
  - Security audit (cargo audit)
  - Performance profiling (flamegraph)
  - API ergonomics review
  - Documentation completeness
```

---

## 4. IMPLEMENTATION TRACKING

### P0: Critical (Must fix before any release)

| Issue             | Current State | Target           | Owner |
| ----------------- | ------------- | ---------------- | ----- |
| Processing module | 5% (4 lines)  | 100%             | NEXT  |
| CLI stubs         | 0% (10 TODOs) | 100%             | NEXT  |
| Error handling    | Basic         | Production-grade | NEXT  |

### P1: High Priority (Next sprint)

| Issue              | Current State | Target    | Owner |
| ------------------ | ------------- | --------- | ----- |
| Qdrant integration | Defined       | Working   | TBD   |
| Test coverage      | ~40%          | 80%       | TBD   |
| Benchmarks         | Template      | Populated | TBD   |

### P2: Medium Priority (Backlog)

| Issue            | Current State | Target        | Owner |
| ---------------- | ------------- | ------------- | ----- |
| OpenAPI spec     | None          | Complete      | TBD   |
| Shell completion | None          | bash/zsh/fish | TBD   |
| RAPTOR tree      | Planned       | Implemented   | TBD   |

---

## 5. EMBEDDED BEST PRACTICES

### Error Handling Pattern

```rust
// CORRECT: Informative, actionable errors
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ReasonKitError {
    #[error("Document not found: {path}")]
    DocumentNotFound { path: String },

    #[error("Embedding failed for chunk {chunk_id}: {source}")]
    EmbeddingFailed {
        chunk_id: String,
        #[source]
        source: reqwest::Error,
    },
}

// WRONG: Don't do this
fn bad() -> Result<(), Box<dyn std::error::Error>> { ... }  // Too generic
value.unwrap();  // Will panic
```

### Async Pattern

```rust
// CORRECT: Non-blocking, cancellation-safe
async fn process_documents(docs: Vec<Document>) -> Result<Vec<Chunk>> {
    let futures = docs.into_iter().map(|doc| async move {
        process_one(doc).await
    });

    futures::future::try_join_all(futures).await
}

// WRONG: Don't block async runtime
async fn bad() {
    std::thread::sleep(Duration::from_secs(1));  // BLOCKS!
}
```

### Testing Pattern

```rust
// CORRECT: Property-based + unit tests
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn chunking_preserves_content(text in "\\PC{1,1000}") {
            let chunks = chunk_text(&text, 100);
            let reconstructed: String = chunks.iter()
                .map(|c| c.text.as_str())
                .collect();
            // Allow for overlap
            assert!(text.starts_with(&reconstructed[..text.len().min(reconstructed.len())]));
        }
    }
}
```

---

## 6. CI/CD INTEGRATION

### GitHub Actions Workflow

```yaml
# .github/workflows/quality-gates.yml
name: Quality Gates

on: [push, pull_request]

jobs:
  gate-1-build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo build --release

  gate-2-lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy
      - run: cargo clippy -- -D warnings

  gate-3-format:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt
      - run: cargo fmt --check

  gate-4-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test --all-features

  gate-5-bench:
    runs-on: ubuntu-latest
    if: contains(github.event.pull_request.labels.*.name, 'performance')
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo bench --bench retrieval_bench -- --noplot
```

---

## 7. METRICS DASHBOARD

### Key Performance Indicators

| KPI                  | Current | Target | Trend |
| -------------------- | ------- | ------ | ----- |
| Test Coverage        | ~40%    | 80%    | -     |
| Clippy Warnings      | TBD     | 0      | -     |
| Build Time (release) | TBD     | <60s   | -     |
| Binary Size          | TBD     | <20MB  | -     |
| P0 Issues            | 3       | 0      | -     |

### Quality Score Formula

```
QUALITY_SCORE = (
    0.25 * (test_coverage / 100) +
    0.20 * (1 - warnings / 100) +
    0.20 * (1 - p0_issues / 10) +
    0.15 * (spec_panel_score / 10) +
    0.10 * (doc_coverage / 100) +
    0.10 * benchmark_stability
) * 10

Current Estimate: 4.5/10
Target: 8.0/10
```

---

## 8. ESCALATION PROTOCOL

### When to Escalate

| Trigger          | Action              | Escalation To    |
| ---------------- | ------------------- | ---------------- |
| Gate failure     | Block merge         | PR author        |
| P0 discovered    | Stop work           | Tech lead        |
| Security issue   | Immediate fix       | Security team    |
| Spec panel < 5.0 | Architecture review | All stakeholders |

### Resolution SLA

| Severity    | Resolution Time | Notification     |
| ----------- | --------------- | ---------------- |
| P0/Critical | 24 hours        | Immediate        |
| P1/High     | 1 week          | Daily standup    |
| P2/Medium   | 1 sprint        | Sprint planning  |
| P3/Low      | Backlog         | Quarterly review |

---

## 9. CHANGE LOG

| Date       | Version | Change                      |
| ---------- | ------- | --------------------------- |
| 2025-12-11 | 1.0.0   | Initial QA Plan established |

---

## 10. COMMITMENT

```
This Quality Assurance Plan is a BINDING COMMITMENT.

All development work on reasonkit-core MUST:
1. Pass all 5 quality gates before merge
2. Follow embedded best practices
3. Maintain or improve quality metrics
4. Undergo periodic expert review

Violations require:
- Root cause analysis
- Corrective action plan
- Prevention measures

Signed: Claude Code (AI Engineering Agent)
Date: 2025-12-11
```

---

_"Quality is not an act, it is a habit." - Aristotle_
_"Designed, Not Dreamed." - ReasonKit_
