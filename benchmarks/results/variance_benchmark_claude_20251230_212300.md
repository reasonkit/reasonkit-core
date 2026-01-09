# ReasonKit Variance Reduction Benchmark Report

**Generated:** 2025-12-30T21:23:00.489331+00:00
**Model:** Claude (via claude CLI)
**Runs per question:** 5
**Total questions:** 3

---

## Executive Summary

| Metric                     | Raw Prompts | Structured Prompts | Change      |
| -------------------------- | ----------- | ------------------ | ----------- |
| Mean Agreement Rate (TARa) | 100.0%      | 100.0%             | **+0.0 pp** |
| Inconsistency Rate         | 0.0%        | 0.0%               | **-0.0 pp** |
| Mean Variance Reduction    | -           | -                  | **0.0%**    |

---

## Methodology

Based on academic literature:

- [arXiv:2408.04667](https://arxiv.org/abs/2408.04667) - "Non-Determinism of 'Deterministic' LLM Settings"
- [arXiv:2203.11171](https://arxiv.org/abs/2203.11171) - "Self-Consistency Improves Chain of Thought Reasoning"
- [arXiv:2511.20836](https://arxiv.org/abs/2511.20836) - "Structured Prompting Enables More Robust Evaluation"

### Protocol

1. Each question run **5 times** with identical prompts
2. Two conditions: Raw prompts vs 5-step structured reasoning
3. Metric: TARa (Total Agreement Rate for parsed answers)

---

## Results by Category

### Math (3 questions)

| Raw Agreement | Structured Agreement | Improvement |
| ------------- | -------------------- | ----------- |
| 100.0%        | 100.0%               | +0.0 pp     |

---

## Conclusion

Structured prompting reduced output inconsistency from **0.0%** to **0.0%** (a **0.0%** reduction in variance).

---

_ReasonKit Variance Benchmark v1.0 - Claude CLI_
