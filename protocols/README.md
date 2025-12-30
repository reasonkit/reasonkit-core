# ReasonKit Core Protocols

This directory contains behavioral protocols for ThinkTool modules.

## Available Protocols

| Protocol ID      | Name                     | ThinkTool  | Profile                    | File                                     |
| ---------------- | ------------------------ | ---------- | -------------------------- | ---------------------------------------- |
| PROT-PG-DEEP-001 | ProofGuard Deep Research | ProofGuard | paranoid, scientific, deep | `proofguard-deep-research-protocol.yaml` |
| PROT-WS-OPT-001  | Web Search Optimization  | ProofGuard | all                        | `web-search-optimization-protocol.yaml`  |

## Protocol Categories

### Verification Protocols

- **ProofGuard Deep Research** - Multi-source triangulation with structured synthesis

### Search Optimization Protocols

- **Web Search Optimization** - Optimized web search with HyDE, adaptive routing, multi-provider parallel search, and credibility scoring (9/9 tests passing)

### (Future) Reasoning Protocols

- GigaThink expansive exploration
- LaserLogic deductive chains
- BedRock first principles

### (Future) Decision Protocols (Pro)

- DeciDomatic multi-criteria analysis _(Pro)_
- RiskRadar threat assessment _(Pro)_

## Usage

Protocols are invoked through ThinkTool profiles:

```bash
# CLI invocation (future)
rk-core think --profile paranoid "query"

# Explicit protocol
rk-core think --protocol pg-deep "research topic"
```

## Schema

All protocols follow `reasonkit-proofguard-*-v1` schema with:

- Metadata (id, name, triggers)
- Phases (numbered workflow steps)
- Checklists (validation gates)
- Objective measures (testable metrics)

## License

Apache 2.0
