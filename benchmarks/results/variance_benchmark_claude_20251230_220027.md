# ReasonKit Variance Reduction Benchmark Report

**Generated:** 2025-12-30T22:00:27.031465+00:00
**Model:** Claude (via claude CLI)
**Runs per question:** 5
**Total questions:** 10

---

## Executive Summary

| Metric                     | Raw Prompts | Structured Prompts | Change      |
| -------------------------- | ----------- | ------------------ | ----------- |
| Mean Agreement Rate (TARa) | 96.0%       | 98.0%              | **+2.0 pp** |
| Inconsistency Rate         | 4.0%        | 2.0%               | **-2.0 pp** |
| Mean Variance Reduction    | -           | -                  | **5.0%**    |

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

### Factual (2 questions)

| Raw Agreement | Structured Agreement | Improvement |
| ------------- | -------------------- | ----------- |
| 100.0%        | 100.0%               | +0.0 pp     |

### Math (3 questions)

| Raw Agreement | Structured Agreement | Improvement |
| ------------- | -------------------- | ----------- |
| 100.0%        | 100.0%               | +0.0 pp     |

### Complex (1 questions)

| Raw Agreement | Structured Agreement | Improvement |
| ------------- | -------------------- | ----------- |
| 80.0%         | 100.0%               | +20.0 pp    |

### Logic (2 questions)

| Raw Agreement | Structured Agreement | Improvement |
| ------------- | -------------------- | ----------- |
| 100.0%        | 100.0%               | +0.0 pp     |

### Decision (2 questions)

| Raw Agreement | Structured Agreement | Improvement |
| ------------- | -------------------- | ----------- |
| 90.0%         | 90.0%                | +0.0 pp     |

---

## Conclusion

Structured prompting reduced output inconsistency from **4.0%** to **2.0%** (a **5.0%** reduction in variance).

---

*ReasonKit Variance Benchmark v1.0 - Claude CLI*
