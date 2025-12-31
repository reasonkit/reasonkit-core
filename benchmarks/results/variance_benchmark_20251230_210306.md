# ReasonKit Variance Reduction Benchmark Report

**Generated:** 2025-12-30T21:03:06.048229
**Model:** mock/simulated-variance
**Temperature:** 0.0
**Runs per question:** 20
**Total questions:** 10

---

## Executive Summary

| Metric                     | Raw Prompts | Structured Prompts | Change       |
| -------------------------- | ----------- | ------------------ | ------------ |
| Mean Agreement Rate (TARa) | 80.0%       | 98.5%              | **+18.5 pp** |
| Inconsistency Rate         | 20.0%       | 1.5%               | **-18.5 pp** |
| Mean Variance Reduction    | -           | -                  | **38.0%**    |

---

## Methodology

Based on academic literature:

- [arXiv:2408.04667](https://arxiv.org/abs/2408.04667) - "Non-Determinism of 'Deterministic' LLM Settings"
- [arXiv:2203.11171](https://arxiv.org/abs/2203.11171) - "Self-Consistency Improves Chain of Thought Reasoning"
- [arXiv:2511.20836](https://arxiv.org/abs/2511.20836) - "Structured Prompting Enables More Robust Evaluation"

### Protocol

1. Each question run **20 times** with identical parameters
2. Temperature: **0.0**
3. Two conditions: Raw prompts vs 5-step structured reasoning
4. Metric: TARa (Total Agreement Rate for parsed answers)

---

## Results by Category

### Logic (2 questions)

| Raw Agreement | Structured Agreement | Improvement |
| ------------- | -------------------- | ----------- |
| 67.5%         | 95.0%                | +27.5 pp    |

### Factual (2 questions)

| Raw Agreement | Structured Agreement | Improvement |
| ------------- | -------------------- | ----------- |
| 92.5%         | 100.0%               | +7.5 pp     |

### Math (3 questions)

| Raw Agreement | Structured Agreement | Improvement |
| ------------- | -------------------- | ----------- |
| 98.3%         | 100.0%               | +1.7 pp     |

### Decision (2 questions)

| Raw Agreement | Structured Agreement | Improvement |
| ------------- | -------------------- | ----------- |
| 65.0%         | 97.5%                | +32.5 pp    |

### Complex (1 questions)

| Raw Agreement | Structured Agreement | Improvement |
| ------------- | -------------------- | ----------- |
| 55.0%         | 100.0%               | +45.0 pp    |

---

## Conclusion

Structured prompting reduced output inconsistency from **20.0%** to **1.5%** (a **38.0%** reduction in variance).

---

_ReasonKit Variance Benchmark v1.0_
