# ReasonKit CLI Workflow Examples

> Practical examples for common reasoning tasks

---

## Quick Start

```bash
# Install ReasonKit
curl -fsSL https://get.reasonkit.sh | bash

# Or build from source
cargo build --release
./scripts/install-rk-commands.sh
```

---

## Common Workflows

### 1. Quick Decision Analysis

When you need a fast, structured analysis:

```bash
# Quick mode: GigaThink + LaserLogic (~60s)
rk-think --quick "Should we use microservices or monolith for this project?"

# Output includes:
# - 10+ perspectives on the decision
# - Logical analysis of each option
# - Confidence score (target: 70%)
```

### 2. Thorough Research

For comprehensive analysis:

```bash
# Deep mode: All 5 ThinkTools (~5min)
rk-think --deep "What are the security implications of this architecture?"

# Or use rk-web for research + web search
rk-web --deep "Compare RAPTOR vs ColBERT for production RAG"
```

### 3. Fact Verification

Verify claims with triangulated sources:

```bash
# Single claim verification
rk verify "GPT-4 has 1.7 trillion parameters" --sources 3

# ProofGuard standalone
rk-pg "The ROI projection of 15% is accurate based on historical data"
```

### 4. Critical Decision Making

Maximum verification for high-stakes decisions:

```bash
# Paranoid mode: All 5 + verification loop (~10min)
rk-think --paranoid "Should we approve this $50K vendor contract?"

# Output includes:
# - Multi-perspective analysis
# - Logical validation
# - First principles decomposition
# - 3-source verification
# - Adversarial critique
# - Confidence score (target: 95%)
```

### 5. Brainstorming Session

Generate diverse perspectives:

```bash
# GigaThink standalone
rk-gt "Generate 10 perspectives on AI safety regulations"

# With creative temperature
rk think --protocol gigathink --temperature 0.9 "Innovation ideas for our product"
```

### 6. Argument Validation

Check logical soundness:

```bash
# LaserLogic standalone
rk-ll "If AI improves productivity, and productivity improves economy, then AI improves economy"

# Full logical analysis
rk think --protocol laserlogic "Analyze this argument: [your argument]"
```

### 7. First Principles Analysis

Break down to fundamentals:

```bash
# BedRock standalone
rk-br "Why do customers choose our product over competitors?"

# Deep decomposition
rk think --protocol bedrock "What are the fundamental assumptions in our business model?"
```

### 8. Self-Critique

Find flaws before others do:

```bash
# BrutalHonesty standalone
rk-bh "Review this proposal and find all weaknesses"

# Adversarial review
rk think --protocol brutalhonesty "Attack this design: [your design]"
```

---

## Advanced Workflows

### Research Pipeline

```bash
#!/bin/bash
# research_topic.sh - Complete research pipeline

TOPIC="$1"
OUTPUT_DIR="./research_output"
mkdir -p "$OUTPUT_DIR"

echo "=== Starting Research Pipeline for: $TOPIC ==="

# Step 1: Brainstorm angles
echo "Step 1: Brainstorming..."
rk-gt "$TOPIC" --format json > "$OUTPUT_DIR/perspectives.json"

# Step 2: Deep research with web
echo "Step 2: Deep research..."
rk-web --deep "$TOPIC" --web --output "$OUTPUT_DIR/research.md"

# Step 3: Verify key claims
echo "Step 3: Verification..."
rk-think --paranoid "Verify the key findings from: $TOPIC" \
  --format json > "$OUTPUT_DIR/verification.json"

# Step 4: Generate summary
echo "Step 4: Summary..."
rk-think --balanced "Summarize findings on: $TOPIC" > "$OUTPUT_DIR/summary.md"

echo "=== Research complete! See $OUTPUT_DIR ==="
```

### Code Review Pipeline

```bash
#!/bin/bash
# code_review.sh - Structured code review

FILE="$1"

echo "=== Reviewing: $FILE ==="

# Security analysis
echo "Security check..."
rk-think --paranoid "Security review of this code: $(cat $FILE)" \
  --format json > review_security.json

# Logic validation
echo "Logic check..."
rk-ll "Is this code logically correct: $(cat $FILE)"

# Adversarial critique
echo "Finding flaws..."
rk-bh "Find all bugs and issues in: $(cat $FILE)"
```

### Decision Support Pipeline

```bash
#!/bin/bash
# decision_support.sh - Structured decision making

DECISION="$1"
OPTIONS="${@:2}"

echo "=== Decision Analysis: $DECISION ==="

# Generate perspectives on each option
for opt in $OPTIONS; do
  echo "Analyzing: $opt"
  rk-gt "Analyze pros and cons of: $opt for decision: $DECISION"
done

# Logical comparison
echo "Comparing options..."
rk-ll "Compare these options for $DECISION: $OPTIONS"

# First principles
echo "First principles analysis..."
rk-br "What are the fundamental factors in deciding: $DECISION"

# Final recommendation
echo "Generating recommendation..."
rk-think --deep "Recommend the best option for: $DECISION given options: $OPTIONS"
```

---

## Output Formats

### Text Output (Default)

```bash
rk-think "Your query"
# Human-readable formatted output
```

### JSON Output

```bash
rk-think "Your query" --format json
# Machine-readable structured output
```

### Save to File

```bash
rk-web --deep "Topic" --output research_report.md
# Saves markdown report to file
```

---

## Environment Variables

```bash
# API Keys
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export TAVILY_API_KEY="tvly-..."

# Configuration
export REASONKIT_DATA_DIR="./data"
export REASONKIT_CONFIG="~/.reasonkit/config.yaml"

# Provider selection
export REASONKIT_PROVIDER="anthropic"  # or openai, openrouter
```

---

## Profile Selection Guide

| Scenario            | Profile        | Command                                |
| ------------------- | -------------- | -------------------------------------- |
| Quick brainstorm    | `--quick`      | `rk-think --quick "idea"`              |
| Daily decisions     | `--balanced`   | `rk-think "question"`                  |
| Research synthesis  | `--deep`       | `rk-think --deep "topic"`              |
| Security review     | `--paranoid`   | `rk-think --paranoid "security issue"` |
| Scientific analysis | `--scientific` | `rk-think --scientific "hypothesis"`   |

---

## Tips & Best Practices

1. **Start with `--quick`** for initial exploration, escalate to `--deep` or `--paranoid` for important decisions.

2. **Use `rk-web`** for research tasks that need web sources.

3. **Use `rk verify`** when specific claims need triangulation.

4. **Save traces** with `--save-trace` for audit trails.

5. **Use JSON output** (`--format json`) for integration with other tools.

6. **Set temperature** higher (0.8-0.9) for creative tasks, lower (0.3-0.5) for analytical tasks.

---

## Troubleshooting

```bash
# Check installation
rk --version

# List available protocols
rk-think --list

# Verbose output for debugging
rk-think -v -v -v "Query"

# Test with mock LLM
rk-think --mock "Test query"
```

---

_ReasonKit CLI Workflow Examples | Apache 2.0_
*https://reasonkit.sh*
