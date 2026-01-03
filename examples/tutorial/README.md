# ReasonKit Interactive Tutorial

> **Time:** 10 minutes total | **Difficulty:** Beginner | **Prerequisites:** rk installed

This tutorial takes you from zero to confidently using ReasonKit for AI reasoning.

---

## Tutorial Structure

| Step | File                                                   | What You Learn             | Time |
| ---- | ------------------------------------------------------ | -------------------------- | ---- |
| 1    | [step1_basic.sh](step1_basic.sh)                       | Basic ThinkTool usage      | 60s  |
| 2    | [step2_profiles.sh](step2_profiles.sh)                 | Choosing the right profile | 60s  |
| 3    | [step3_individual_tools.sh](step3_individual_tools.sh) | Individual ThinkTools      | 90s  |
| 4    | [step4_json_output.sh](step4_json_output.sh)           | Machine-readable output    | 60s  |
| 5    | [step5_audit_trail.sh](step5_audit_trail.sh)           | Execution traces           | 90s  |

---

## Before You Start

### 1. Verify installation

```bash
rk --version
```

### 2. Set your API key

```bash
# Anthropic Claude (recommended)
export ANTHROPIC_API_KEY="sk-ant-..."

# Or OpenAI
export OPENAI_API_KEY="sk-..."

# Or use demo mode (no API key required)
# Just add --mock to any command
```

---

## Run the Tutorial

### Option A: Step by step (recommended)

```bash
cd examples/tutorial

# Step 1
./step1_basic.sh

# Step 2
./step2_profiles.sh

# ... and so on
```

### Option B: All at once

```bash
cd examples/tutorial
./run_all.sh
```

### Option C: Demo mode (no API key)

```bash
cd examples/tutorial
./run_all.sh --mock
```

---

## What You'll Build

By the end of this tutorial, you'll be able to:

1. Run structured AI reasoning from the command line
2. Choose the right profile for your use case
3. Use individual ThinkTools for specific tasks
4. Parse JSON output for automation
5. Review execution traces for debugging

---

## Next Steps

After completing this tutorial:

- **Real-world examples:** [USE_CASES.md](../../docs/process/USE_CASES.md)
- **Full CLI reference:** [CLI_REFERENCE.md](../../docs/reference/CLI_REFERENCE.md)
- **ThinkTool deep dive:** [THINKTOOLS_GUIDE.md](../../docs/thinktools/THINKTOOLS_GUIDE.md)
- **Rust API:** [API_REFERENCE.md](../../docs/reference/API_REFERENCE.md)

---

**Questions?** [GitHub Discussions](https://github.com/reasonkit/reasonkit-core/discussions)
