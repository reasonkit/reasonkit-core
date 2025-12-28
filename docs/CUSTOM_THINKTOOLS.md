# Custom ThinkTools Guide

ReasonKit allows you to define your own reasoning protocols (ThinkTools) using TOML configuration files. This enables you to create specialized reasoning patterns tailored to your specific domain or problem type.

## Quick Start

1. Create a directory named `protocols` in your current working directory or `~/.config/reasonkit/protocols/`.
2. Create a file named `my_tools.toml` (or any name ending in `.toml`).
3. Define your custom ThinkTool in the file.

## Protocol Definition (TOML)

ReasonKit protocols are defined using the `thinktool-v2` schema.

```toml
version = "2.0.0"
schema = "thinktool-v2"

[thinktool_modules.devils_advocate]
id = "devils_advocate"
name = "Devil's Advocate"
description = "Systematically challenge an idea from multiple angles"
category = "adversarial"
tier = "custom"
capabilities = ["critique", "adversarial"]
typical_duration = "5s"
token_cost_estimate = "medium"

# Optional: Define standard thinking pattern metadata
[thinktool_modules.devils_advocate.thinking_pattern]
type = "linear"
steps = ["identify_assumptions", "challenge_assumptions", "propose_alternatives"]

# CORE: Define the actual reasoning steps
[[thinktool_modules.devils_advocate.steps]]
id = "identify_assumptions"
action = { type = "analyze", criteria = ["assumptions", "biases"] }
prompt_template = """
Identify the core assumptions underlying this proposal:

Proposal: {{query}}

List explicit and implicit assumptions.
"""
output_format = "list"
min_confidence = 0.7

[[thinktool_modules.devils_advocate.steps]]
id = "challenge_assumptions"
action = { type = "critique", severity = "standard" }
prompt_template = """
Challenge each identified assumption:

Assumptions:
{{identify_assumptions}}

For each assumption, ask: "Under what conditions would this be false?"
"""
output_format = "structured"
min_confidence = 0.6
depends_on = ["identify_assumptions"]

[[thinktool_modules.devils_advocate.steps]]
id = "propose_alternatives"
action = { type = "generate", min_count = 3, max_count = 5 }
prompt_template = """
Based on the challenges, propose 3-5 alternative approaches that avoid these pitfalls.

Challenges:
{{challenge_assumptions}}
"""
output_format = "list"
min_confidence = 0.8
depends_on = ["challenge_assumptions"]
```

## Protocol Reference

### Module Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (snake_case) |
| `name` | string | Human-readable name |
| `description` | string | Brief description of what the tool does |
| `category` | string | `divergent`, `convergent`, `analytical`, `verification`, `adversarial` |
| `steps` | array | List of ProtocolStep objects (see below) |

### Protocol Step

Each step defines a single atomic reasoning operation.

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique step ID (used for referencing) |
| `action` | object | The type of cognitive action (see below) |
| `prompt_template` | string | Handlebars-style template. Can reference input `{{query}}` or previous step outputs `{{step_id}}`. |
| `output_format` | string | `text`, `list`, `structured`, `score`, `boolean` |
| `min_confidence` | float | 0.0 - 1.0 (default 0.7) |
| `depends_on` | array | List of step IDs this step requires input from |

### Action Types

| Action Type | Parameters | Example |
|-------------|------------|---------|
| `generate` | `min_count`, `max_count` | `{ type = "generate", min_count = 5, max_count = 10 }` |
| `analyze` | `criteria` (array) | `{ type = "analyze", criteria = ["feasibility", "cost"] }` |
| `synthesize` | `aggregation` | `{ type = "synthesize", aggregation = "thematic_clustering" }` |
| `validate` | `rules` (array) | `{ type = "validate", rules = ["consistency"] }` |
| `critique` | `severity` | `{ type = "critique", severity = "brutal" }` |
| `decide` | `method` | `{ type = "decide", method = "pros_cons" }` |
| `cross_reference` | `min_sources` | `{ type = "cross_reference", min_sources = 3 }` |

## Advanced: Dynamic Branching

You can add branching logic to steps:

```toml
[[thinktool_modules.my_tool.steps]]
id = "check_safety"
# ...
branch = { type = "confidence_below", threshold = 0.8 } 
# If confidence < 0.8, the executor might trigger a loop or fallback (depending on implementation)
```

## Best Practices

1. **Atomic Steps:** Keep steps focused on a single cognitive task (e.g., "List ideas" vs "List and evaluate ideas").
2. **Clear Dependencies:** Explicitly state `depends_on` to ensure the context is available in the prompt.
3. **Structured Output:** Prefer `structured` or `list` output formats for easier parsing by subsequent steps.
4. **Prompt Engineering:** Use clear instructions in `prompt_template`. The quality of the output depends heavily on the prompt.
