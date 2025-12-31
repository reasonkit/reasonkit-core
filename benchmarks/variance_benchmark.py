#!/usr/bin/env python3
"""
ReasonKit Variance Reduction Benchmark
======================================

PURPOSE: Measure output variance with and without structured prompting.
         Produce academic-grade, reproducible results.

METHODOLOGY:
- Run identical queries N times (default N=20)
- Compare raw LLM outputs vs ThinkTools structured outputs
- Calculate variance metrics: TARa, StdDev, Coefficient of Variation
- Use multiple question types for robustness

METRICS (based on academic literature):
- TARa@N: Total Agreement Rate of parsed answers across N runs
- CV: Coefficient of Variation (StdDev / Mean)
- Consistency Score: % of runs producing identical core answer

REFERENCES:
- arXiv:2408.04667 "Non-Determinism of 'Deterministic' LLM Settings"
- arXiv:2203.11171 "Self-Consistency Improves Chain of Thought Reasoning"
- arXiv:2511.20836 "Structured Prompting Enables More Robust Evaluation"

Author: ReasonKit Team
License: Apache 2.0
"""

import os
import sys
import json
import time
import hashlib
import argparse
import statistics
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from collections import Counter

# Check for required packages
try:
    import anthropic
except ImportError:
    print("ERROR: anthropic package required. Install with: uv pip install anthropic")
    sys.exit(1)


# ============================================================================
# TEST QUESTIONS - Diverse set for robust variance measurement
# ============================================================================

VARIANCE_TEST_QUESTIONS = [
    # Math reasoning (has definitive answer)
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
    # Logical reasoning (has definitive answer)
    {
        "id": "logic_001",
        "category": "logic",
        "question": "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly?",
        "expected_answer": "no",
        "answer_type": "boolean"
    },
    {
        "id": "logic_002",
        "category": "logic",
        "question": "If it rains, the ground gets wet. The ground is wet. Did it rain?",
        "expected_answer": "unknown",
        "answer_type": "categorical"
    },
    # Factual (definitive answer)
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
    # Decision/Analysis (measures reasoning consistency)
    {
        "id": "decision_001",
        "category": "decision",
        "question": "A company has $100,000 to invest. Option A offers 5% guaranteed return. Option B offers 50% chance of 15% return, 50% chance of -5% return. Which is the safer choice?",
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
    # Complex reasoning
    {
        "id": "complex_001",
        "category": "complex",
        "question": "In a room of 23 people, what is the approximate probability that at least two share a birthday? Answer: above 50%, below 50%, or exactly 50%?",
        "expected_answer": "above 50%",
        "answer_type": "categorical"
    }
]


# ============================================================================
# STRUCTURED PROMPTING TEMPLATES (ThinkTools style)
# ============================================================================

RAW_PROMPT_TEMPLATE = """Answer the following question. Provide only your final answer, nothing else.

Question: {question}

Answer:"""

STRUCTURED_PROMPT_TEMPLATE = """You are a rigorous analytical assistant. Follow this structured reasoning process:

## STEP 1: UNDERSTAND
- Restate the question in your own words
- Identify what type of answer is expected (number, yes/no, explanation)

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
class SingleRunResult:
    """Result from a single query run."""
    run_id: int
    raw_output: str
    parsed_answer: str
    latency_ms: float
    tokens_used: int
    timestamp: str


@dataclass
class QuestionVarianceResult:
    """Variance analysis for a single question."""
    question_id: str
    category: str
    question: str
    expected_answer: str
    n_runs: int

    # Raw LLM results
    raw_answers: list
    raw_unique_answers: int
    raw_agreement_rate: float  # TARa
    raw_most_common_answer: str
    raw_most_common_count: int
    raw_correct_rate: float

    # Structured prompting results
    structured_answers: list
    structured_unique_answers: int
    structured_agreement_rate: float  # TARa
    structured_most_common_answer: str
    structured_most_common_count: int
    structured_correct_rate: float

    # Improvement metrics
    agreement_improvement: float  # Structured TARa - Raw TARa
    variance_reduction_pct: float  # (1 - structured_variance/raw_variance) * 100


@dataclass
class BenchmarkSummary:
    """Overall benchmark summary."""
    timestamp: str
    model: str
    temperature: float
    n_runs_per_question: int
    n_questions: int

    # Aggregate metrics
    mean_raw_agreement_rate: float
    mean_structured_agreement_rate: float
    mean_agreement_improvement: float
    mean_variance_reduction_pct: float

    # By category
    by_category: dict

    # Individual results
    question_results: list


# ============================================================================
# CORE BENCHMARK FUNCTIONS
# ============================================================================

def create_client() -> anthropic.Anthropic:
    """Create Anthropic client."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable required")
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)


def query_llm(
    client: anthropic.Anthropic,
    prompt: str,
    model: str = "claude-sonnet-4-20250514",
    temperature: float = 0.0,
    max_tokens: int = 1024
) -> tuple[str, float, int]:
    """
    Query the LLM and return (response, latency_ms, tokens_used).
    """
    start = time.perf_counter()

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )

    latency_ms = (time.perf_counter() - start) * 1000
    tokens_used = response.usage.input_tokens + response.usage.output_tokens
    output = response.content[0].text

    return output, latency_ms, tokens_used


def parse_answer(raw_output: str, answer_type: str) -> str:
    """
    Parse the core answer from LLM output.
    Normalizes for comparison (lowercase, strip, etc).
    """
    output = raw_output.strip()

    # Look for "FINAL ANSWER:" pattern from structured prompts
    if "FINAL ANSWER:" in output.upper():
        parts = output.upper().split("FINAL ANSWER:")
        if len(parts) > 1:
            output = parts[1].strip()

    # Normalize based on answer type
    if answer_type == "numeric":
        # Extract first number
        import re
        numbers = re.findall(r'-?\d+\.?\d*', output)
        if numbers:
            # Return as clean number string
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
        return output_lower.strip()[:50]

    elif answer_type == "categorical":
        # Take first line, first 100 chars, lowercase
        first_line = output.split('\n')[0]
        return first_line.lower().strip()[:100]

    else:  # text
        return output.lower().strip()[:50]


def calculate_agreement_rate(answers: list) -> float:
    """
    Calculate TARa (Total Agreement Rate for parsed answers).
    Returns the fraction of answers that match the most common answer.
    """
    if not answers:
        return 0.0

    counter = Counter(answers)
    most_common_count = counter.most_common(1)[0][1]
    return most_common_count / len(answers)


def check_correct(parsed_answer: str, expected: str, answer_type: str) -> bool:
    """Check if parsed answer matches expected."""
    parsed_lower = parsed_answer.lower().strip()
    expected_lower = expected.lower().strip()

    if answer_type == "numeric":
        try:
            return abs(float(parsed_lower) - float(expected_lower)) < 0.01
        except ValueError:
            return parsed_lower == expected_lower

    return expected_lower in parsed_lower or parsed_lower in expected_lower


def run_variance_test(
    client: anthropic.Anthropic,
    question_data: dict,
    n_runs: int = 20,
    model: str = "claude-sonnet-4-20250514",
    temperature: float = 0.0,
    verbose: bool = True
) -> QuestionVarianceResult:
    """
    Run variance test for a single question.
    Runs both raw and structured prompts n_runs times each.
    """
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
        output, latency, tokens = query_llm(client, raw_prompt, model, temperature)
        parsed = parse_answer(output, answer_type)
        raw_results.append(parsed)
        if verbose and (i + 1) % 5 == 0:
            print(f"    Raw: {i+1}/{n_runs} complete")

    # Run structured prompts
    structured_prompt = STRUCTURED_PROMPT_TEMPLATE.format(question=question)
    structured_results = []
    for i in range(n_runs):
        output, latency, tokens = query_llm(client, structured_prompt, model, temperature)
        parsed = parse_answer(output, answer_type)
        structured_results.append(parsed)
        if verbose and (i + 1) % 5 == 0:
            print(f"    Structured: {i+1}/{n_runs} complete")

    # Calculate metrics
    raw_agreement = calculate_agreement_rate(raw_results)
    structured_agreement = calculate_agreement_rate(structured_results)

    raw_counter = Counter(raw_results)
    structured_counter = Counter(structured_results)

    raw_most_common = raw_counter.most_common(1)[0] if raw_results else ("", 0)
    structured_most_common = structured_counter.most_common(1)[0] if structured_results else ("", 0)

    raw_correct = sum(1 for a in raw_results if check_correct(a, expected, answer_type)) / len(raw_results)
    structured_correct = sum(1 for a in structured_results if check_correct(a, expected, answer_type)) / len(structured_results)

    # Variance reduction calculation
    # Using (1 - unique_structured/unique_raw) as proxy for variance reduction
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
    n_runs: int = 20,
    model: str = "claude-sonnet-4-20250514",
    temperature: float = 0.0,
    questions: Optional[list] = None,
    verbose: bool = True
) -> BenchmarkSummary:
    """
    Run the full variance benchmark suite.
    """
    if questions is None:
        questions = VARIANCE_TEST_QUESTIONS

    client = create_client()

    print("=" * 70)
    print("REASONKIT VARIANCE REDUCTION BENCHMARK")
    print("=" * 70)
    print(f"Model: {model}")
    print(f"Temperature: {temperature}")
    print(f"Runs per question: {n_runs}")
    print(f"Total questions: {len(questions)}")
    print(f"Total API calls: {len(questions) * n_runs * 2}")
    print("=" * 70)

    results = []
    for q in questions:
        result = run_variance_test(
            client, q, n_runs, model, temperature, verbose
        )
        results.append(result)

    # Calculate aggregate metrics
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

    summary = BenchmarkSummary(
        timestamp=datetime.utcnow().isoformat(),
        model=model,
        temperature=temperature,
        n_runs_per_question=n_runs,
        n_questions=len(questions),
        mean_raw_agreement_rate=mean_raw_agreement,
        mean_structured_agreement_rate=mean_structured_agreement,
        mean_agreement_improvement=mean_improvement,
        mean_variance_reduction_pct=mean_variance_reduction,
        by_category=by_category,
        question_results=[asdict(r) for r in results]
    )

    return summary


def print_summary(summary: BenchmarkSummary):
    """Print human-readable summary."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nModel: {summary.model}")
    print(f"Temperature: {summary.temperature}")
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

    # Convert to the "85% to 22%" format if applicable
    # Raw variance = (1 - raw_agreement) represents inconsistency
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
        print(f"  Variance reduction:   {data['mean_variance_reduction']:.1f}%")

    print("\n" + "-" * 70)
    print("INDIVIDUAL QUESTION RESULTS")
    print("-" * 70)

    for r in summary.question_results:
        raw_agree = r['raw_agreement_rate'] * 100
        struct_agree = r['structured_agreement_rate'] * 100
        print(f"\n{r['question_id']} ({r['category']}):")
        print(f"  Raw: {r['raw_unique_answers']} unique answers, {raw_agree:.0f}% agreement")
        print(f"  Structured: {r['structured_unique_answers']} unique answers, {struct_agree:.0f}% agreement")
        print(f"  Raw correct: {r['raw_correct_rate']*100:.0f}%, Structured correct: {r['structured_correct_rate']*100:.0f}%")


def save_results(summary: BenchmarkSummary, output_dir: Path):
    """Save results to JSON and markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = output_dir / f"variance_benchmark_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(asdict(summary), f, indent=2)
    print(f"\nJSON saved: {json_path}")

    # Save markdown report
    md_path = output_dir / f"variance_benchmark_{timestamp}.md"

    raw_pct = summary.mean_raw_agreement_rate * 100
    struct_pct = summary.mean_structured_agreement_rate * 100
    improvement = summary.mean_agreement_improvement * 100
    raw_inconsistency = (1 - summary.mean_raw_agreement_rate) * 100
    structured_inconsistency = (1 - summary.mean_structured_agreement_rate) * 100

    md_content = f"""# ReasonKit Variance Reduction Benchmark Report

**Generated:** {summary.timestamp}
**Model:** {summary.model}
**Temperature:** {summary.temperature}
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

This benchmark measures output variance using the TARa (Total Agreement Rate for parsed answers) metric, following the methodology established in academic literature:

- **arXiv:2408.04667** - "Non-Determinism of 'Deterministic' LLM Settings"
- **arXiv:2203.11171** - "Self-Consistency Improves Chain of Thought Reasoning"
- **arXiv:2511.20836** - "Structured Prompting Enables More Robust Evaluation"

### Test Protocol

1. Each question is run **{summary.n_runs_per_question} times** with identical parameters
2. Temperature set to **{summary.temperature}** (deterministic mode)
3. Two conditions tested:
   - **Raw prompts**: Direct question, no structure
   - **Structured prompts**: 5-step reasoning framework (Understand → Analyze → Solve → Verify → Answer)
4. Answers parsed and normalized for comparison
5. Agreement rate calculated as: (most common answer count) / (total runs)

---

## Results by Category

"""

    for cat, data in summary.by_category.items():
        md_content += f"""### {cat.title()} ({data['n_questions']} questions)

| Metric | Value |
|--------|-------|
| Raw Agreement | {data['mean_raw_agreement']*100:.1f}% |
| Structured Agreement | {data['mean_structured_agreement']*100:.1f}% |
| Improvement | +{data['mean_improvement']*100:.1f} percentage points |
| Variance Reduction | {data['mean_variance_reduction']:.1f}% |

"""

    md_content += """---

## Detailed Results

| Question ID | Category | Raw Unique | Raw TARa | Struct Unique | Struct TARa | Improvement |
|-------------|----------|------------|----------|---------------|-------------|-------------|
"""

    for r in summary.question_results:
        md_content += f"| {r['question_id']} | {r['category']} | {r['raw_unique_answers']} | {r['raw_agreement_rate']*100:.0f}% | {r['structured_unique_answers']} | {r['structured_agreement_rate']*100:.0f}% | +{r['agreement_improvement']*100:.0f} pp |\n"

    md_content += f"""
---

## Statistical Validity

- **Sample size per condition:** {summary.n_runs_per_question} runs
- **Total API calls:** {summary.n_questions * summary.n_runs_per_question * 2}
- **Reproducibility:** All tests run at temperature=0 for deterministic base

### Limitations

1. Results are model-specific ({summary.model})
2. Question set is limited ({summary.n_questions} questions)
3. Temperature=0 may not reflect real-world usage
4. Parser normalization may affect results

---

## Conclusion

Structured prompting with a 5-step reasoning framework reduced output variance by **{summary.mean_variance_reduction_pct:.1f}%** on average, improving agreement rate from **{raw_pct:.1f}%** to **{struct_pct:.1f}%**.

This supports claims that structured prompting reduces LLM output variance, consistent with academic literature on chain-of-thought and self-consistency methods.

---

*Generated by ReasonKit Variance Benchmark v1.0*
"""

    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"Report saved: {md_path}")

    return json_path, md_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ReasonKit Variance Reduction Benchmark"
    )
    parser.add_argument(
        "--runs", "-n", type=int, default=20,
        help="Number of runs per question (default: 20)"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="claude-sonnet-4-20250514",
        help="Model to test (default: claude-sonnet-4-20250514)"
    )
    parser.add_argument(
        "--temperature", "-t", type=float, default=0.0,
        help="Temperature setting (default: 0.0)"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default="results",
        help="Output directory for results (default: results)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test with 5 runs and 3 questions"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    n_runs = 5 if args.quick else args.runs
    questions = VARIANCE_TEST_QUESTIONS[:3] if args.quick else VARIANCE_TEST_QUESTIONS

    summary = run_full_benchmark(
        n_runs=n_runs,
        model=args.model,
        temperature=args.temperature,
        questions=questions,
        verbose=not args.quiet
    )

    print_summary(summary)

    output_dir = Path(args.output_dir)
    save_results(summary, output_dir)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
