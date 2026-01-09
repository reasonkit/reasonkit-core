# ADR-005: CLI-First Distribution

## Status

**Accepted** - 2024-12-28

## Context

ReasonKit needs a distribution strategy that enables rapid adoption while maintaining long-term flexibility. We must balance:

1. **Time to Value**: How quickly can developers get productive?
2. **Friction**: What barriers exist to trying ReasonKit?
3. **Integration Depth**: How deeply can ReasonKit integrate with existing workflows?
4. **Monetization Path**: How does distribution support business goals?

We evaluated several distribution approaches:

| Approach          | Time to Value           | Friction                   | Integration | Monetization             |
| ----------------- | ----------------------- | -------------------------- | ----------- | ------------------------ |
| **Library-First** | Medium (needs code)     | Medium (import, learn API) | Deep        | SDK licensing            |
| **CLI-First**     | Immediate (one command) | Very Low (install and run) | Medium      | CLI features, enterprise |
| **SaaS-First**    | Immediate (signup)      | Medium (account, API keys) | Shallow     | Usage-based              |
| **IDE Plugin**    | Medium (install plugin) | Medium (per-IDE effort)    | Deep        | Marketplace              |

### Market Analysis

Successful developer tools often follow a CLI-first pattern:

| Tool               | Initial Distribution | Later Expansion           |
| ------------------ | -------------------- | ------------------------- |
| **Docker**         | CLI (`docker run`)   | Desktop, Compose, Swarm   |
| **kubectl**        | CLI                  | Lens, Rancher, dashboards |
| **git**            | CLI                  | GitHub, GitLab, GUIs      |
| **terraform**      | CLI                  | Cloud, Enterprise         |
| **llm (Willison)** | CLI                  | Datasette integration     |

The pattern: **CLI establishes value, ecosystem builds on it.**

### Developer Workflow Integration

Modern AI developers use command-line tools extensively:

```bash
# Typical AI development workflow
git pull
uv pip install -r requirements.txt
python train.py
llm "explain this error: $(cat error.log)"
rk --profile deep "Is this approach correct?"  # <-- ReasonKit fits here
git commit -m "Fix training loop"
```

CLI tools integrate naturally into this flow.

## Decision

**We will distribute ReasonKit as a CLI-first tool, with library/SDK as secondary.**

### Distribution Hierarchy

```
1. CLI (Primary)     - Immediate value, lowest friction
2. Library (Secondary) - Deep integration for programmatic use
3. MCP Server (Extension) - AI assistant integration
4. API (Enterprise)   - SaaS deployment option
```

### CLI Design Principles

1. **Single Binary**: No runtime dependencies
2. **Immediate Value**: First command produces useful output
3. **Progressive Disclosure**: Simple defaults, powerful options available
4. **Unix Philosophy**: Works with pipes, files, stdin/stdout
5. **Scriptable**: Exit codes, JSON output, machine-readable options

### Command Structure

```bash
# Immediate value (one command)
rk "Should I use microservices for my startup?"

# Profile selection
rk --profile deep "Complex architectural question"
rk --profile paranoid "Security-critical decision"

# Output control
rk --format json "Query"
rk --format markdown "Query" > analysis.md

# Integration with other tools
cat requirements.txt | rk "Are there security vulnerabilities?"
rk "Explain this code" < src/main.rs
git diff | rk "Review these changes"

# Configuration
rk config set llm.provider openai
rk config set llm.model gpt-4
```

### Installation Options

```bash
# Rust developers (signals Rust-first identity)
cargo install reasonkit

# Universal (detects platform, installs binary)
curl -fsSL https://get.reasonkit.sh | bash

# Package managers
brew install reasonkit        # macOS
apt install reasonkit         # Debian/Ubuntu
winget install reasonkit      # Windows

# Container
docker run -it reasonkit/core "Query"

# Ecosystem compatibility
uv pip install reasonkit      # Python bindings
npm install -g @reasonkit/cli # Node.js wrapper
```

### Library Mode (Secondary)

```rust
// Rust library usage
use reasonkit::{ReasoningEngine, Profile};

let engine = ReasoningEngine::new()?;
let result = engine.reason("Query", Profile::Balanced).await?;
println!("Confidence: {}", result.confidence);
```

```python
# Python bindings (via PyO3)
from reasonkit import ReasoningEngine, Profile

engine = ReasoningEngine()
result = engine.reason("Query", Profile.BALANCED)
print(f"Confidence: {result.confidence}")
```

## Consequences

### Positive

1. **Immediate Value**: `cargo install reasonkit && rk "question"` in under 30 seconds
2. **Low Friction**: No accounts, API keys (with local models), or configuration required
3. **Scriptable**: Integrates with CI/CD, shell scripts, automation
4. **Discoverable**: `--help` and `--version` are universal interfaces
5. **Demonstrable**: Easy to show in demos, blog posts, tutorials
6. **SEO Friendly**: Command examples are easily searchable
7. **Offline Capable**: Works without network (with local models)

### Negative

1. **Limited UI**: No graphical interface for complex workflows
2. **Learning Curve**: Command-line unfamiliar to some developers
3. **Integration Depth**: Less control than programmatic API
4. **State Management**: CLI inherently stateless per-invocation

### Mitigations

| Negative          | Mitigation                                            |
| ----------------- | ----------------------------------------------------- |
| Limited UI        | TUI mode (`rk --tui`); web dashboard planned          |
| Learning curve    | Excellent `--help`; interactive mode; examples        |
| Integration depth | Library/SDK for programmatic use                      |
| State management  | SQLite audit log persists state; session continuation |

### Adoption Funnel

```
Stage 1: Discovery
  "Oh, a reasoning CLI tool"
  cargo install reasonkit

Stage 2: First Value
  rk "Is Kubernetes overkill for my project?"
  "Wow, this is thorough"

Stage 3: Integration
  alias think='rk --profile balanced'
  Added to git pre-commit hooks

Stage 4: Library Use
  Using Python bindings in production
  Integrated with existing agentic workflows

Stage 5: Enterprise
  Team licenses, SSO, custom profiles
  Self-hosted deployment
```

## Related Documents

- `../reference/CLI_REFERENCE.md` - CLI documentation
- `../../src/main.rs` - CLI implementation
- ADR-001: Rust enables single-binary distribution

## References

- [Command Line Interface Guidelines](https://clig.dev/)
- [12 Factor CLI Apps](https://medium.com/@jdxcode/12-factor-cli-apps-dd3c227a0e46)
- [Building a CLI with Clap](https://rust-cli.github.io/book/)
- [Simon Willison's LLM CLI](https://llm.datasette.io/)
