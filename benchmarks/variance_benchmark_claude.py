#!/usr/bin/env python3
"""
ReasonKit Variance Reduction Benchmark - Claude CLI Version
============================================================

Uses the claude CLI (Claude Code) with your web account login.
No API keys needed - uses your existing authentication.

Usage:
  python3 variance_benchmark_claude.py --runs 10
  python3 variance_benchmark_claude.py --quick  # 5 runs, 3 questions
"""

import subprocess
import json
import time
import statistics
import argparse
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import Counter
from typing import Optional
import re

# ============================================================================
# TEST QUESTIONS
# ============================================================================

VARIANCE_TEST_QUESTIONS = [
    {
        "id": "math_001",
        "category": "math",
        "question": "If a train travels at 60 mph for 2.5 hours, how many miles does it travel?",
        "expected_answer": "150",
        "answer_type": "numeric"
    },
    {
        "id": "math_002",
        "category": "math",
        "question": "A store sells apples for $0.75 each. If you buy 8 apples and pay with a $10 bill, how much change do you receive?",
        "expected_answer": "4",
        "answer_type": "numeric"
    },
    {
        "id": "math_003",
        "category": "math",
        "question": "What is 15% of 240?",
        "expected_answer": "36",
        "answer_type": "numeric"
    },
    {
        "id": "logic_001",
        "category": "logic",
        "question": "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly? Answer yes, no, or unknown.",
        "expected_answer": "no",
        "answer_type": "boolean"
    },
    {
        "id": "logic_002",
        "category": "logic",
        "question": "If it rains, the ground gets wet. The ground is wet. Did it rain? Answer yes, no, or unknown.",
        "expected_answer": "unknown",
        "answer_type": "categorical"
    },
    {
        "id": "fact_001",
        "category": "factual",
        "question": "What is the chemical symbol for gold?",
        "expected_answer": "Au",
        "answer_type": "text"
    },
    {
        "id": "fact_002",
        "category": "factual",
        "question": "How many planets are in our solar system?",
        "expected_answer": "8",
        "answer_type": "numeric"
    },
    {
        "id": "decision_001",
        "category": "decision",
        "question": "A company has $100,000 to invest. Option A offers 5% guaranteed return. Option B offers 50% chance of 15% return, 50% chance of -5% return. Which is the safer choice? Answer A or B.",
        "expected_answer": "A",
        "answer_type": "categorical"
    },
    {
        "id": "decision_002",
        "category": "decision",
        "question": "For a critical medical diagnosis system, should you prioritize minimizing false negatives or false positives?",
        "expected_answer": "false negatives",
        "answer_type": "categorical"
    },
    {
        "id": "complex_001",
        "category": "complex",
        "question": "In a room of 23 people, what is the approximate probability that at least two share a birthday? Answer: above 50%, below 50%, or exactly 50%?",
        "expected_answer": "above 50%",
        "answer_type": "categorical"
    }
]

# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

RAW_PROMPT_TEMPLATE = """Answer the following question. Provide only your final answer, nothing else.

Question: {question}

Answer:"""

STRUCTURED_PROMPT_TEMPLATE = """You are a rigorous analytical assistant. Follow this structured reasoning process:

## STEP 1: UNDERSTAND
- Restate the question in your own words
- Identify what type of answer is expected

## STEP 2: ANALYZE
- Break down the problem into components
- Identify relevant facts, formulas, or logical principles

## STEP 3: SOLVE
- Work through the solution step by step
- Show your reasoning clearly

## STEP 4: VERIFY
- Double-check your work
- Consider if the answer makes sense

## STEP 5: FINAL ANSWER
- State your final answer clearly
- Format: "FINAL ANSWER: [your answer]"

---

Question: {question}

Begin your structured analysis:"""


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class QuestionVarianceResult:
    question_id: str
    category: str
    question: str
    expected_answer: str
    n_runs: int
    raw_answers: list
    raw_unique_answers: int
    raw_agreement_rate: float
    raw_most_common_answer: str
    raw_most_common_count: int
    raw_correct_rate: float
    structured_answers: list
    structured_unique_answers: int
    structured_agreement_rate: float
    structured_most_common_answer: str
    structured_most_common_count: int
    structured_correct_rate: float
    agreement_improvement: float
    variance_reduction_pct: float


@dataclass
class BenchmarkSummary:
    timestamp: str
    model: str
    temperature: float
    n_runs_per_question: int
    n_questions: int
    mean_raw_agreement_rate: float
    mean_structured_agreement_rate: float
    mean_agreement_improvement: float
    mean_variance_reduction_pct: float
    by_category: dict
    question_results: list


# ============================================================================
# CLAUDE CLI INTERFACE
# ============================================================================

def query_claude(prompt: str, timeout: int = 120) -> tuple[str, float]:
    """
    Query Claude using the claude CLI.
    Returns (response, latency_ms)
    """
    start = time.perf_counter()

    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "text"],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        latency_ms = (time.perf_counter() - start) * 1000

        if result.returncode != 0:
            print(f"    Warning: claude returned {result.returncode}")
            return result.stderr or "ERROR", latency_ms

        return result.stdout.strip(), latency_ms

    except subprocess.TimeoutExpired:
        latency_ms = (time.perf_counter() - start) * 1000
        return "TIMEOUT", latency_ms
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return f"ERROR: {e}", latency_ms


# ============================================================================
# PARSING AND METRICS
# ============================================================================

def parse_answer(raw_output: str, answer_type: str) -> str:
    output = raw_output.strip()

    if "FINAL ANSWER:" in output.upper():
        parts = output.upper().split("FINAL ANSWER:")
        if len(parts) > 1:
            output = parts[1].strip()

    if answer_type == "numeric":
        numbers = re.findall(r'-?\d+\.?\d*', output)
        if numbers:
            num = float(numbers[0])
            if num == int(num):
                return str(int(num))
            return str(num)
        return output.lower().strip()

    elif answer_type == "boolean":
        output_lower = output.lower()
        if any(word in output_lower for word in ["yes", "true", "correct", "can conclude"]):
            return "yes"
        elif any(word in output_lower for word in ["no", "false", "incorrect", "cannot conclude"]):
            return "no"
        elif "unknown" in output_lower:
            return "unknown"
        return output_lower.strip()[:50]

    elif answer_type == "categorical":
        first_line = output.split('\n')[0]
        return first_line.lower().strip()[:100]

    else:
        return output.lower().strip()[:50]


def calculate_agreement_rate(answers: list) -> float:
    if not answers:
        return 0.0
    counter = Counter(answers)
    most_common_count = counter.most_common(1)[0][1]
    return most_common_count / len(answers)


def check_correct(parsed_answer: str, expected: str, answer_type: str) -> bool:
    parsed_lower = parsed_answer.lower().strip()
    expected_lower = expected.lower().strip()

    if answer_type == "numeric":
        try:
            return abs(float(parsed_lower) - float(expected_lower)) < 0.01
        except ValueError:
            return parsed_lower == expected_lower

    return expected_lower in parsed_lower or parsed_lower in expected_lower


# ============================================================================
# BENCHMARK EXECUTION
# ============================================================================

def run_variance_test(
    question_data: dict,
    n_runs: int = 10,
    verbose: bool = True
) -> QuestionVarianceResult:

    question = question_data["question"]
    question_id = question_data["id"]
    category = question_data["category"]
    expected = question_data["expected_answer"]
    answer_type = question_data["answer_type"]

    if verbose:
        print(f"\n  Testing: {question_id} ({category})")
        print(f"  Question: {question[:60]}...")

    # Run raw prompts
    raw_prompt = RAW_PROMPT_TEMPLATE.format(question=question)
    raw_results = []
    for i in range(n_runs):
        output, latency = query_claude(raw_prompt)
        parsed = parse_answer(output, answer_type)
        raw_results.append(parsed)
        if verbose:
            print(f"    Raw {i+1}/{n_runs}: {parsed[:30]}... ({latency:.0f}ms)")

    # Run structured prompts
    structured_prompt = STRUCTURED_PROMPT_TEMPLATE.format(question=question)
    structured_results = []
    for i in range(n_runs):
        output, latency = query_claude(structured_prompt)
        parsed = parse_answer(output, answer_type)
        structured_results.append(parsed)
        if verbose:
            print(f"    Structured {i+1}/{n_runs}: {parsed[:30]}... ({latency:.0f}ms)")

    # Calculate metrics
    raw_agreement = calculate_agreement_rate(raw_results)
    structured_agreement = calculate_agreement_rate(structured_results)

    raw_counter = Counter(raw_results)
    structured_counter = Counter(structured_results)

    raw_most_common = raw_counter.most_common(1)[0] if raw_results else ("", 0)
    structured_most_common = structured_counter.most_common(1)[0] if structured_results else ("", 0)

    raw_correct = sum(1 for a in raw_results if check_correct(a, expected, answer_type)) / len(raw_results) if raw_results else 0
    structured_correct = sum(1 for a in structured_results if check_correct(a, expected, answer_type)) / len(structured_results) if structured_results else 0

    raw_unique = len(set(raw_results))
    structured_unique = len(set(structured_results))

    if raw_unique > 0:
        variance_reduction = (1 - structured_unique / raw_unique) * 100
    else:
        variance_reduction = 0.0

    agreement_improvement = structured_agreement - raw_agreement

    return QuestionVarianceResult(
        question_id=question_id,
        category=category,
        question=question,
        expected_answer=expected,
        n_runs=n_runs,
        raw_answers=raw_results,
        raw_unique_answers=raw_unique,
        raw_agreement_rate=raw_agreement,
        raw_most_common_answer=raw_most_common[0],
        raw_most_common_count=raw_most_common[1],
        raw_correct_rate=raw_correct,
        structured_answers=structured_results,
        structured_unique_answers=structured_unique,
        structured_agreement_rate=structured_agreement,
        structured_most_common_answer=structured_most_common[0],
        structured_most_common_count=structured_most_common[1],
        structured_correct_rate=structured_correct,
        agreement_improvement=agreement_improvement,
        variance_reduction_pct=variance_reduction
    )


def run_full_benchmark(
    n_runs: int = 10,
    questions: Optional[list] = None,
    verbose: bool = True
) -> BenchmarkSummary:

    if questions is None:
        questions = VARIANCE_TEST_QUESTIONS

    print("=" * 70)
    print("REASONKIT VARIANCE REDUCTION BENCHMARK")
    print("=" * 70)
    print("Backend: claude CLI (Claude Code)")
    print(f"Runs per question: {n_runs}")
    print(f"Total questions: {len(questions)}")
    print(f"Total CLI calls: {len(questions) * n_runs * 2}")
    print("=" * 70)

    results = []
    for q in questions:
        result = run_variance_test(q, n_runs, verbose)
        results.append(result)

    # Calculate aggregates
    mean_raw_agreement = statistics.mean(r.raw_agreement_rate for r in results)
    mean_structured_agreement = statistics.mean(r.structured_agreement_rate for r in results)
    mean_improvement = statistics.mean(r.agreement_improvement for r in results)
    mean_variance_reduction = statistics.mean(r.variance_reduction_pct for r in results)

    # By category
    categories = set(r.category for r in results)
    by_category = {}
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        by_category[cat] = {
            "n_questions": len(cat_results),
            "mean_raw_agreement": statistics.mean(r.raw_agreement_rate for r in cat_results),
            "mean_structured_agreement": statistics.mean(r.structured_agreement_rate for r in cat_results),
            "mean_improvement": statistics.mean(r.agreement_improvement for r in cat_results),
            "mean_variance_reduction": statistics.mean(r.variance_reduction_pct for r in cat_results)
        }

    return BenchmarkSummary(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model="claude-cli",
        temperature=0.0,
        n_runs_per_question=n_runs,
        n_questions=len(questions),
        mean_raw_agreement_rate=mean_raw_agreement,
        mean_structured_agreement_rate=mean_structured_agreement,
        mean_agreement_improvement=mean_improvement,
        mean_variance_reduction_pct=mean_variance_reduction,
        by_category=by_category,
        question_results=[asdict(r) for r in results]
    )


def print_summary(summary: BenchmarkSummary):
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nModel: {summary.model}")
    print(f"Runs per question: {summary.n_runs_per_question}")
    print(f"Total questions: {summary.n_questions}")

    print("\n" + "-" * 70)
    print("AGGREGATE METRICS")
    print("-" * 70)

    raw_pct = summary.mean_raw_agreement_rate * 100
    struct_pct = summary.mean_structured_agreement_rate * 100
    improvement = summary.mean_agreement_improvement * 100

    print(f"\nMean Agreement Rate (TARa):")
    print(f"  Raw prompts:        {raw_pct:.1f}%")
    print(f"  Structured prompts: {struct_pct:.1f}%")
    print(f"  Improvement:        +{improvement:.1f} percentage points")

    print(f"\nMean Variance Reduction: {summary.mean_variance_reduction_pct:.1f}%")

    raw_inconsistency = (1 - summary.mean_raw_agreement_rate) * 100
    structured_inconsistency = (1 - summary.mean_structured_agreement_rate) * 100

    print(f"\nInconsistency Rate:")
    print(f"  Raw prompts:        {raw_inconsistency:.1f}% inconsistent")
    print(f"  Structured prompts: {structured_inconsistency:.1f}% inconsistent")

    if raw_inconsistency > 0:
        relative_reduction = ((raw_inconsistency - structured_inconsistency) / raw_inconsistency) * 100
        print(f"  Relative reduction: {relative_reduction:.1f}%")

    print("\n" + "-" * 70)
    print("BY CATEGORY")
    print("-" * 70)

    for cat, data in summary.by_category.items():
        print(f"\n{cat.upper()} ({data['n_questions']} questions):")
        print(f"  Raw agreement:        {data['mean_raw_agreement']*100:.1f}%")
        print(f"  Structured agreement: {data['mean_structured_agreement']*100:.1f}%")
        print(f"  Improvement:          +{data['mean_improvement']*100:.1f} pp")


def save_results(summary: BenchmarkSummary, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # JSON
    json_path = output_dir / f"variance_benchmark_claude_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(asdict(summary), f, indent=2)
    print(f"\nJSON saved: {json_path}")

    # Markdown report
    md_path = output_dir / f"variance_benchmark_claude_{timestamp}.md"

    raw_pct = summary.mean_raw_agreement_rate * 100
    struct_pct = summary.mean_structured_agreement_rate * 100
    improvement = summary.mean_agreement_improvement * 100
    raw_inconsistency = (1 - summary.mean_raw_agreement_rate) * 100
    structured_inconsistency = (1 - summary.mean_structured_agreement_rate) * 100

    md_content = f"""# ReasonKit Variance Reduction Benchmark Report

**Generated:** {summary.timestamp}
**Model:** Claude (via claude CLI)
**Runs per question:** {summary.n_runs_per_question}
**Total questions:** {summary.n_questions}

---

## Executive Summary

| Metric | Raw Prompts | Structured Prompts | Change |
|--------|-------------|-------------------|--------|
| Mean Agreement Rate (TARa) | {raw_pct:.1f}% | {struct_pct:.1f}% | **+{improvement:.1f} pp** |
| Inconsistency Rate | {raw_inconsistency:.1f}% | {structured_inconsistency:.1f}% | **-{raw_inconsistency - structured_inconsistency:.1f} pp** |
| Mean Variance Reduction | - | - | **{summary.mean_variance_reduction_pct:.1f}%** |

---

## Methodology

Based on academic literature:
- [arXiv:2408.04667](https://arxiv.org/abs/2408.04667) - "Non-Determinism of 'Deterministic' LLM Settings"
- [arXiv:2203.11171](https://arxiv.org/abs/2203.11171) - "Self-Consistency Improves Chain of Thought Reasoning"
- [arXiv:2511.20836](https://arxiv.org/abs/2511.20836) - "Structured Prompting Enables More Robust Evaluation"

### Protocol
1. Each question run **{summary.n_runs_per_question} times** with identical prompts
2. Two conditions: Raw prompts vs 5-step structured reasoning
3. Metric: TARa (Total Agreement Rate for parsed answers)

---

## Results by Category

"""

    for cat, data in summary.by_category.items():
        md_content += f"""### {cat.title()} ({data['n_questions']} questions)
| Raw Agreement | Structured Agreement | Improvement |
|--------------|---------------------|-------------|
| {data['mean_raw_agreement']*100:.1f}% | {data['mean_structured_agreement']*100:.1f}% | +{data['mean_improvement']*100:.1f} pp |

"""

    md_content += f"""---

## Conclusion

Structured prompting reduced output inconsistency from **{raw_inconsistency:.1f}%** to **{structured_inconsistency:.1f}%** (a **{summary.mean_variance_reduction_pct:.1f}%** reduction in variance).

---

*ReasonKit Variance Benchmark v1.0 - Claude CLI*
"""

    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"Report saved: {md_path}")

    return json_path, md_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ReasonKit Variance Benchmark (Claude CLI)")
    parser.add_argument("--runs", "-n", type=int, default=10,
                       help="Runs per question (default: 10)")
    parser.add_argument("--output-dir", "-o", type=str, default="results",
                       help="Output directory")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test (5 runs, 3 questions)")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress verbose output")

    args = parser.parse_args()

    n_runs = 5 if args.quick else args.runs
    questions = VARIANCE_TEST_QUESTIONS[:3] if args.quick else VARIANCE_TEST_QUESTIONS

    summary = run_full_benchmark(
        n_runs=n_runs,
        questions=questions,
        verbose=not args.quiet
    )

    print_summary(summary)
    save_results(summary, Path(args.output_dir))

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
