# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for ReasonKit. ADRs document the **why** behind significant technical decisions.

## What is an ADR?

An Architecture Decision Record captures:

- **Context**: What situation prompted the decision?
- **Decision**: What choice was made?
- **Consequences**: What are the trade-offs?

ADRs are immutable records. When decisions change, we create new ADRs that supersede old ones.

## ADR Index

| ID                                                      | Title                           | Status   | Date       |
| ------------------------------------------------------- | ------------------------------- | -------- | ---------- |
| [ADR-001](./ADR-001-rust-as-primary-language.md)        | Use Rust as Primary Language    | Accepted | 2024-12-28 |
| [ADR-002](./ADR-002-sqlite-for-audit-trail.md)          | SQLite for Audit Trail          | Accepted | 2024-12-28 |
| [ADR-003](./ADR-003-thinktool-module-architecture.md)   | ThinkTool Module Architecture   | Accepted | 2024-12-28 |
| [ADR-004](./ADR-004-provider-agnostic-llm-interface.md) | Provider-Agnostic LLM Interface | Accepted | 2024-12-28 |
| [ADR-005](./ADR-005-cli-first-distribution.md)          | CLI-First Distribution          | Accepted | 2024-12-28 |

## ADR Status Definitions

| Status         | Meaning                                       |
| -------------- | --------------------------------------------- |
| **Proposed**   | Under discussion, not yet decided             |
| **Accepted**   | Decision made and being implemented           |
| **Deprecated** | No longer relevant; superseded by another ADR |
| **Superseded** | Replaced by a newer ADR (linked)              |

## Creating New ADRs

Use the following template:

```markdown
# ADR-NNN: Title

## Status

**Proposed/Accepted** - YYYY-MM-DD

## Context

[What is the issue that we're seeing that is motivating this decision?]

## Decision

[What is the change that we're proposing and/or doing?]

## Consequences

### Positive

[What becomes easier or possible?]

### Negative

[What becomes harder or impossible?]

### Mitigations

[How do we address the negatives?]

## Related Documents

[Links to related documentation]

## References

[External references, if any]
```

## File Naming Convention

```
ADR-NNN-short-descriptive-title.md
```

Examples:

- `ADR-001-rust-as-primary-language.md`
- `ADR-006-testing-strategy.md`

## References

- [Michael Nygard's ADR Article](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
- [ADR GitHub Organization](https://adr.github.io/)
- [Lightweight ADRs](https://www.thoughtworks.com/radar/techniques/lightweight-architecture-decision-records)
