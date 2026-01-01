# Getting Started with ReasonKit

> **Goal:** Working AI reasoning in 5 minutes

---

## Quick Links

| I want to...                      | Go to...                                             |
| --------------------------------- | ---------------------------------------------------- |
| **Install and run first command** | [docs/QUICKSTART.md](docs/QUICKSTART.md)             |
| **Step-by-step tutorial**         | [examples/tutorial/](examples/tutorial/)             |
| **Troubleshoot an issue**         | [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)   |
| **See real-world examples**       | [docs/USE_CASES.md](docs/USE_CASES.md)               |
| **Full CLI reference**            | [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md)       |
| **ThinkTool deep dive**           | [docs/THINKTOOLS_GUIDE.md](docs/THINKTOOLS_GUIDE.md) |

---

## 60-Second Quickstart

```bash
# Install
curl -fsSL https://reasonkit.sh/install | bash

# Set API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Run
rk-core think "Should I use microservices?" --profile quick
```

**That's it.** You now have structured AI reasoning.

---

## What Just Happened?

1. **GigaThink** generated 10+ perspectives on your question
2. **LaserLogic** detected logical fallacies and hidden assumptions
3. **ReasonKit** synthesized a verdict with confidence score

This is "structured reasoning" - organized, auditable thinking instead of a wall of text.

---

## Choose Your Depth

| Stakes                | Use                  | Time  |
| --------------------- | -------------------- | ----- |
| Low (Slack message)   | `--profile quick`    | 30s   |
| Medium (PR review)    | `--profile balanced` | 2min  |
| High (Architecture)   | `--profile deep`     | 5min  |
| Critical (Production) | `--profile paranoid` | 10min |

---

## Next Steps

1. **[Interactive Tutorial](examples/tutorial/)** - 10 minutes, hands-on
2. **[Real Examples](docs/USE_CASES.md)** - Code review, architecture, debugging
3. **[Full Docs](docs/)** - Everything you need

---

**Website:** [reasonkit.sh](https://reasonkit.sh)  
**Questions?** [GitHub Discussions](https://github.com/reasonkit/reasonkit-core/discussions)

_Turn prompts into protocols._
