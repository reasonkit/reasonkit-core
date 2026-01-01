# REASONKIT COPILOT AGENTS

This directory contains specialized agent personas for GitHub Copilot CLI.
These agents are designed to enforce the ReasonKit protocols and constraints.

## AVAILABLE AGENTS

### 1. System Architect (`roles/architect.md`)

**Use for:** High-level design, making architectural decisions, reviewing project structure, and enforcing governance.
**Key Traits:** Strategic, authoritative, constraint-aware.

### 2. Rust Engineer (`roles/rust-engineer.md`)

**Use for:** Writing Rust code, optimizing performance, ensuring memory safety, and passing quality gates.
**Key Traits:** Strict, performance-obsessed, safety-first.

### 3. Deep Researcher (`roles/researcher.md`)

**Use for:** Gathering information, verifying claims, and conducting deep analysis.
**Key Traits:** Skeptical, thorough, citation-focused.

### 4. Task Master (`roles/task-master.md`)

**Use for:** Managing tasks, tracking time, and organizing work.
**Key Traits:** Organized, disciplined, process-oriented.

### 5. Python Specialist (`roles/python-specialist.md`)

**Use for:** Writing Python scripts, bindings, and MCP servers.
**Key Traits:** Modern tooling (UV), integration-focused.

### 6. QA Engineer (`roles/qa-engineer.md`)

**Use for:** Running tests, benchmarks, and enforcing the 5 Quality Gates.
**Key Traits:** Detail-oriented, rigorous, gatekeeper.

### 7. Security Guardian (`roles/security-guardian.md`)

**Use for:** Security auditing, secret scanning, and GDPR compliance.
**Key Traits:** Paranoid, thorough, risk-averse.

### 8. DevOps SRE (`roles/devops-sre.md`)

**Use for:** CI/CD pipelines, releases, and infrastructure.
**Key Traits:** Automation-focused, reliable, efficient.

### 9. Technical Writer (`roles/technical-writer.md`)

**Use for:** Documentation, protocol definition, and knowledge management.
**Key Traits:** Clear, authoritative, structured.

## HOW TO USE

### Option 1: Manual Context Loading

Copy the content of the desired agent file (e.g., `roles/rust-engineer.md`) and paste it into your Copilot chat or context.

### Option 2: Custom Instructions

Set the content of the desired agent file as your "Custom Instructions" in your IDE or CLI configuration.

### Option 3: The "Loader" Pattern

Paste the content of `LOADER.md` into your chat. Copilot will ask you which agent you want to activate.

## DIRECTORY STRUCTURE

```
.github/agents/copilot/
├── manifest.json       # Machine-readable list of agents
├── README.md           # This file
├── LOADER.md           # Interactive agent loader prompt
├── MAIN.md             # General-purpose balanced agent
└── roles/              # Specialized agent definitions
    ├── architect.md
    ├── rust-engineer.md
    ├── researcher.md
    ├── task-master.md
    └── python-specialist.md
```
