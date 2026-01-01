# EXPERT SPECIFICATION REVIEW PROTOCOL

> Periodic Quality Reviews for ReasonKit Core
> Version: 1.0.0 | Established: 2025-12-11

---

## PURPOSE

This protocol establishes **mandatory periodic expert reviews** to maintain code quality,
catch architectural drift, and ensure production readiness. The spec-panel review is a
PROVEN METHOD for systematic quality assessment.

---

## REVIEW SCHEDULE

| Frequency     | Type             | Trigger                 | Mandatory |
| ------------- | ---------------- | ----------------------- | --------- |
| **Monthly**   | Full Spec Panel  | 1st Monday of month     | YES       |
| **Ad-hoc**    | Targeted Review  | Major feature complete  | YES       |
| **Quarterly** | External Audit   | Q1, Q2, Q3, Q4          | YES       |
| **Release**   | Pre-release Gate | Before any version bump | YES       |

---

## EXPERT PANEL COMPOSITION

### Core Panel (Always Included)

| Expert             | Perspective              | Focus Areas                           |
| ------------------ | ------------------------ | ------------------------------------- |
| **Karl Wiegers**   | Requirements Engineering | Specification completeness, ambiguity |
| **Michael Nygard** | Release It! / Production | Operational readiness, failure modes  |
| **Martin Fowler**  | Architecture             | Clean architecture, maintainability   |
| **Gojko Adzic**    | Specification by Example | Executable specs, BDD scenarios       |
| **Sam Newman**     | Microservices            | Service boundaries, integration       |

### Extended Panel (On Request)

| Expert             | Perspective          | When to Include                  |
| ------------------ | -------------------- | -------------------------------- |
| **Linus Torvalds** | Code Quality         | Rust code review, performance    |
| **Bryan Cantrill** | Systems              | Low-level performance, debugging |
| **Dan Abramov**    | Developer Experience | API ergonomics, CLI UX           |
| **Julia Evans**    | Debugging            | Error messages, observability    |

---

## REVIEW INVOCATION

### Via Claude Code Slash Command

```bash
# Full spec panel review (recommended)
/sc:spec-panel "cd reasonkit-core"

# Targeted review (specific area)
/sc:spec-panel "cd reasonkit-core/src/retrieval"

# Compare to previous review
/sc:spec-panel "cd reasonkit-core --compare-last"
```

### Manual Review Checklist

If slash command unavailable, use this checklist:

```markdown
## SPEC PANEL REVIEW - [DATE]

### 1. Requirements Quality (0-10)

- [ ] Clear success criteria defined?
- [ ] Edge cases documented?
- [ ] Error scenarios specified?
- [ ] Performance requirements stated?

### 2. Architecture Clarity (0-10)

- [ ] Layered architecture respected?
- [ ] Dependencies flow correctly?
- [ ] Abstractions appropriate?
- [ ] Cross-cutting concerns handled?

### 3. Production Readiness (0-10)

- [ ] Timeout/retry implemented?
- [ ] Error handling comprehensive?
- [ ] Logging/tracing adequate?
- [ ] Monitoring hooks present?

### 4. Testability (0-10)

- [ ] Unit test coverage > 80%?
- [ ] Integration tests exist?
- [ ] Property-based tests for critical paths?
- [ ] Mocking boundaries clear?

### 5. Documentation (0-10)

- [ ] API documented?
- [ ] README current?
- [ ] Architecture decisions recorded?
- [ ] Runbook available?

### OVERALL SCORE: \_\_\_ / 10
```

---

## SCORING CRITERIA

### Score Interpretation

| Score | Status     | Action Required                    |
| ----- | ---------- | ---------------------------------- |
| 9-10  | Excellent  | Proceed with confidence            |
| 7-8   | Good       | Minor improvements, proceed        |
| 5-6   | Acceptable | Address issues before release      |
| 3-4   | Concerning | Significant work needed            |
| 0-2   | Critical   | STOP - Major intervention required |

### Pass/Fail Thresholds

| Gate               | Minimum Score | For                 |
| ------------------ | ------------- | ------------------- |
| Development        | 5.0           | Continue work       |
| Feature Complete   | 6.0           | Merge to main       |
| Release Candidate  | 7.0           | Version bump        |
| Production Release | 8.0           | Public availability |

---

## ISSUE CLASSIFICATION

### Priority Levels

| Level  | Label    | SLA      | Example                   |
| ------ | -------- | -------- | ------------------------- |
| **P0** | Critical | 24h      | Unimplemented core module |
| **P1** | High     | 1 week   | Missing error handling    |
| **P2** | Medium   | 1 sprint | Suboptimal algorithm      |
| **P3** | Low      | Backlog  | Minor refactoring         |

### Issue Template

```markdown
## [P0] Issue Title

**Expert:** Karl Wiegers
**Category:** Requirements
**Location:** src/processing/mod.rs

**Finding:**
Processing module is only 4 lines - completely unimplemented.

**Impact:**
Blocks all document processing - core functionality missing.

**Recommendation:**
Implement chunking, cleaning, and metadata extraction.

**Acceptance Criteria:**

- [ ] Chunk documents by configurable size
- [ ] Clean text (normalize whitespace, remove artifacts)
- [ ] Extract and preserve metadata
- [ ] Test coverage > 80%
```

---

## REVIEW OUTPUT FORMAT

### Standard Report Structure

```markdown
# SPEC PANEL REVIEW REPORT

Date: YYYY-MM-DD
Project: reasonkit-core
Reviewers: [Panel Members]

## EXECUTIVE SUMMARY

[2-3 sentence overview]

## SCORES

| Dimension     | Score    | Trend |
| ------------- | -------- | ----- |
| Requirements  | X/10     | ↑↓→   |
| Architecture  | X/10     | ↑↓→   |
| Production    | X/10     | ↑↓→   |
| Testability   | X/10     | ↑↓→   |
| Documentation | X/10     | ↑↓→   |
| **OVERALL**   | **X/10** | ↑↓→   |

## P0 ISSUES (Count: N)

[List all P0 issues]

## P1 ISSUES (Count: N)

[List all P1 issues]

## RECOMMENDATIONS

[Prioritized action items]

## NEXT REVIEW

Date: YYYY-MM-DD
Focus Areas: [Specific concerns to monitor]
```

---

## TRACKING & METRICS

### Review History Log

Maintain in `reasonkit-core/docs/review_history.jsonl`:

```json
{
  "date": "2025-12-11",
  "overall": 4.9,
  "requirements": 5.8,
  "architecture": 6.5,
  "production": 2.0,
  "testability": 4.0,
  "p0_count": 3,
  "p1_count": 4
}
```

### Trend Analysis

```bash
# Generate trend report (monthly)
cat docs/review_history.jsonl | \
  jq -s 'sort_by(.date) | .[] | "\(.date): \(.overall)/10"'
```

### Quality Improvement Targets

| Metric           | Current | 30-Day Target | 90-Day Target |
| ---------------- | ------- | ------------- | ------------- |
| Overall Score    | 4.9     | 6.0           | 7.5           |
| P0 Issues        | 3       | 0             | 0             |
| Test Coverage    | ~40%    | 60%           | 80%           |
| Production Ready | 2.0     | 5.0           | 7.0           |

---

## ESCALATION

### When to Escalate

| Trigger             | Escalation Target | Action                      |
| ------------------- | ----------------- | --------------------------- |
| Score < 4.0         | Project Lead      | Architecture review meeting |
| P0 count > 3        | All Stakeholders  | Emergency sprint            |
| Score trending down | Tech Lead         | Root cause analysis         |
| Missed review       | Project Manager   | Schedule enforcement        |

### Escalation Template

```
ESCALATION: ReasonKit Core Quality Review

Date: YYYY-MM-DD
Trigger: [Specific trigger]
Current Score: X/10
Required Score: Y/10

Immediate Actions Required:
1. [Action 1]
2. [Action 2]

Resource Requirements:
- [Resources needed]

Timeline for Resolution:
- [Dates]
```

---

## INTEGRATION WITH CI/CD

### Pre-merge Check (Automated)

```yaml
# .github/workflows/quality-review.yml
name: Quality Check

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  quick-review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run quality metrics
        run: ./scripts/quality_metrics.sh --ci
      - name: Check minimum score
        run: |
          SCORE=$(cat target/quality_metrics.json | jq -r '.quality_score')
          if [ "$SCORE" -lt 5 ]; then
            echo "Quality score $SCORE below minimum 5"
            exit 1
          fi
```

### Monthly Review Automation

```yaml
# .github/workflows/monthly-review.yml
name: Monthly Spec Panel

on:
  schedule:
    - cron: "0 9 1 * *" # 9 AM UTC on 1st of month

jobs:
  spec-panel-reminder:
    runs-on: ubuntu-latest
    steps:
      - name: Create review issue
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: '🔍 Monthly Spec Panel Review Due',
              body: 'Time for the monthly spec panel review. Run: /sc:spec-panel "cd reasonkit-core"',
              labels: ['review', 'monthly']
            })
```

---

## CHANGE LOG

| Date       | Version | Change                       |
| ---------- | ------- | ---------------------------- |
| 2025-12-11 | 1.0.0   | Initial protocol established |

---

## COMMITMENT

```
This Review Protocol is PERMANENTLY EMBEDDED in the development workflow.

All ReasonKit development MUST:
1. Undergo monthly spec panel reviews
2. Pass minimum score thresholds before release
3. Track and trend quality metrics over time
4. Escalate when scores fall below thresholds

This is not optional. This is how we build reliable software.

Established by: Claude Code (AI Engineering Agent)
Date: 2025-12-11
```

---

*"What gets measured gets managed." - Peter Drucker*
*"Designed, Not Dreamed." - ReasonKit*
