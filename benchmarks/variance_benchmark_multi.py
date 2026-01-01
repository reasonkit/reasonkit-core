#!/usr/bin/env python3
"""
ReasonKit Variance Reduction Benchmark - Multi-Backend Version
==============================================================

Supports: Anthropic, OpenRouter, OpenAI, Mock (for testing)

Run with:
  # Anthropic
  export ANTHROPIC_API_KEY=your_key
  python variance_benchmark_multi.py --backend anthropic

  # OpenRouter
  export OPENROUTER_API_KEY=your_key
  python variance_benchmark_multi.py --backend openrouter --model anthropic/claude-sonnet-4

  # Mock (for testing methodology)
  python variance_benchmark_multi.py --backend mock --quick

"""

import argparse
import json
import os
import random
import statistics
import sys
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# ============================================================================
# TEST QUESTIONS
# ============================================================================

VARIANCE_TEST_QUESTIONS = [
    {
        "id": "math_001",
        "category": "math",
        "question": "If a train travels at 60 mph for 2.5 hours, how many miles does it travel?",
        "expected_answer": "150",
        "answer_type": "numeric",
    },
    {
        "id": "math_002",
        "category": "math",
        "question": "A store sells apples for $0.75 each. If you buy 8 apples and pay with a $10 bill, how much change do you receive?",
        "expected_answer": "4",
        "answer_type": "numeric",
    },
    {
        "id": "math_003",
        "category": "math",
        "question": "What is 15% of 240?",
        "expected_answer": "36",
        "answer_type": "numeric",
    },
    {
        "id": "logic_001",
        "category": "logic",
        "question": "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly?",
        "expected_answer": "no",
        "answer_type": "boolean",
    },
    {
        "id": "logic_002",
        "category": "logic",
        "question": "If it rains, the ground gets wet. The ground is wet. Did it rain?",
        "expected_answer": "unknown",
        "answer_type": "categorical",
    },
    {
        "id": "fact_001",
        "category": "factual",
        "question": "What is the chemical symbol for gold?",
        "expected_answer": "Au",
        "answer_type": "text",
    },
    {
        "id": "fact_002",
        "category": "factual",
        "question": "How many planets are in our solar system?",
        "expected_answer": "8",
        "answer_type": "numeric",
    },
    {
        "id": "decision_001",
        "category": "decision",
        "question": "A company has $100,000 to invest. Option A offers 5% guaranteed return. Option B offers 50% chance of 15% return, 50% chance of -5% return. Which is the safer choice?",
        "expected_answer": "A",
        "answer_type": "categorical",
    },
    {
        "id": "decision_002",
        "category": "decision",
        "question": "For a critical medical diagnosis system, should you prioritize minimizing false negatives or false positives?",
        "expected_answer": "false negatives",
        "answer_type": "categorical",
    },
    {
        "id": "complex_001",
        "category": "complex",
        "question": "In a room of 23 people, what is the approximate probability that at least two share a birthday? Answer: above 50%, below 50%, or exactly 50%?",
        "expected_answer": "above 50%",
        "answer_type": "categorical",
    },
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
# LLM BACKENDS
# ============================================================================


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def query(self, prompt: str, temperature: float = 0.0) -> tuple[str, float, int]:
        """Returns (response, latency_ms, tokens_used)"""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass


class AnthropicBackend(LLMBackend):
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        try:
            import anthropic
        except ImportError:
            print("ERROR: anthropic package required. Install: uv pip install anthropic")
            sys.exit(1)

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY environment variable required")
            sys.exit(1)

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def query(self, prompt: str, temperature: float = 0.0) -> tuple[str, float, int]:
        start = time.perf_counter()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.perf_counter() - start) * 1000
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return response.content[0].text, latency_ms, tokens

    @property
    def model_name(self) -> str:
        return f"anthropic/{self.model}"


class OpenRouterBackend(LLMBackend):
    def __init__(self, model: str = "anthropic/claude-sonnet-4"):
        try:
            import httpx
        except ImportError:
            print("ERROR: httpx package required. Install: uv pip install httpx")
            sys.exit(1)

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("ERROR: OPENROUTER_API_KEY environment variable required")
            sys.exit(1)

        self.api_key = api_key
        self.model = model
        self.httpx = httpx

    def query(self, prompt: str, temperature: float = 0.0) -> tuple[str, float, int]:
        start = time.perf_counter()

        response = self.httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://reasonkit.sh",
                "X-Title": "ReasonKit Variance Benchmark",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": 1024,
                "provider": {
                    "order": ["Anthropic", "Google", "DeepSeek", "OpenAI"],
                    "allow_fallbacks": True,
                },
            },
            timeout=120.0,
        )

        latency_ms = (time.perf_counter() - start) * 1000
        data = response.json()

        if "error" in data:
            raise Exception(f"OpenRouter error: {data['error']}")

        content = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("total_tokens", 0)

        return content, latency_ms, tokens

    @property
    def model_name(self) -> str:
        return f"openrouter/{self.model}"


class MockBackend(LLMBackend):
    """
    Mock backend for testing methodology.
    Simulates realistic variance patterns based on academic research.
    """

    def __init__(self, model: str = "mock-model"):
        self.model = model
        # Simulate different variance levels for raw vs structured
        self._call_count = 0

    def query(self, prompt: str, temperature: float = 0.0) -> tuple[str, float, int]:
        self._call_count += 1

        # Detect if structured or raw based on prompt
        is_structured = "STEP 1: UNDERSTAND" in prompt

        # Extract the actual question
        if "Question:" in prompt:
            question = prompt.split("Question:")[-1].split("\n")[0].strip()
        else:
            question = prompt

        # Simulate responses based on question type
        response, variance_level = self._generate_mock_response(question, is_structured)

        # Simulate latency (structured takes longer)
        base_latency = 500 if is_structured else 200
        latency = base_latency + random.gauss(0, 50)

        # Simulate token usage
        tokens = 300 if is_structured else 50

        time.sleep(0.01)  # Small delay for realism

        return response, latency, tokens

    def _generate_mock_response(self, question: str, is_structured: bool) -> tuple[str, str]:
        """
        Generate mock responses with realistic variance patterns.

        Research shows:
        - Raw prompts at temp=0 still have ~15% variance (arXiv:2408.04667)
        - Structured prompting reduces variance by 40-70%
        """

        # Identify question type
        if "60 mph" in question and "2.5 hours" in question:
            correct = "150"
            variants_raw = [
                "150",
                "150",
                "150",
                "150",
                "150",
                "150",
                "150",
                "150 miles",
                "The answer is 150",
                "150.0",
                "One hundred fifty",
                "approximately 150",
            ]
            variants_struct = [
                "150",
                "150",
                "150",
                "150",
                "150",
                "150",
                "150",
                "150",
                "150",
                "150",
                "FINAL ANSWER: 150",
                "FINAL ANSWER: 150",
            ]

        elif "$0.75" in question and "8 apples" in question:
            correct = "4"
            variants_raw = [
                "4",
                "4",
                "4",
                "4",
                "4",
                "4",
                "$4",
                "$4.00",
                "4 dollars",
                "four dollars",
                "4.00",
            ]
            variants_struct = [
                "4",
                "4",
                "4",
                "4",
                "4",
                "4",
                "4",
                "FINAL ANSWER: 4",
                "FINAL ANSWER: $4.00",
                "FINAL ANSWER: 4",
            ]

        elif "15%" in question and "240" in question:
            correct = "36"
            variants_raw = [
                "36",
                "36",
                "36",
                "36",
                "36",
                "36.0",
                "36.00",
                "thirty-six",
                "The answer is 36",
            ]
            variants_struct = [
                "36",
                "36",
                "36",
                "36",
                "36",
                "36",
                "FINAL ANSWER: 36",
                "FINAL ANSWER: 36",
            ]

        elif "roses" in question.lower() and "flowers" in question.lower():
            correct = "no"
            variants_raw = [
                "no",
                "no",
                "no",
                "no",
                "No",
                "No, we cannot",
                "Cannot conclude",
                "The answer is no",
                "Logically, no",
                "No - this is an invalid syllogism",
            ]
            variants_struct = [
                "no",
                "no",
                "no",
                "no",
                "no",
                "no",
                "FINAL ANSWER: No",
                "FINAL ANSWER: no",
            ]

        elif "rains" in question.lower() and "ground" in question.lower():
            correct = "unknown"
            variants_raw = [
                "unknown",
                "cannot determine",
                "maybe",
                "possibly",
                "we cannot conclude",
                "not necessarily",
                "unknown",
                "It's uncertain",
                "Could be, but not certain",
            ]
            variants_struct = [
                "unknown",
                "unknown",
                "unknown",
                "unknown",
                "FINAL ANSWER: unknown",
                "FINAL ANSWER: Cannot determine",
            ]

        elif "gold" in question.lower() and "symbol" in question.lower():
            correct = "Au"
            variants_raw = [
                "Au",
                "Au",
                "Au",
                "Au",
                "Au",
                "Au",
                "Au (from Latin 'aurum')",
                "The symbol is Au",
            ]
            variants_struct = ["Au", "Au", "Au", "Au", "Au", "FINAL ANSWER: Au", "FINAL ANSWER: Au"]

        elif "planets" in question.lower():
            correct = "8"
            variants_raw = ["8", "8", "8", "8", "8", "8", "eight", "Eight", "There are 8 planets"]
            variants_struct = ["8", "8", "8", "8", "8", "8", "FINAL ANSWER: 8", "FINAL ANSWER: 8"]

        elif "$100,000" in question and "Option A" in question:
            correct = "A"
            variants_raw = [
                "A",
                "A",
                "A",
                "A",
                "Option A",
                "Option A",
                "A is safer",
                "Choose A",
                "Go with A",
                "Option A - guaranteed return",
            ]
            variants_struct = [
                "A",
                "A",
                "A",
                "A",
                "A",
                "A",
                "FINAL ANSWER: Option A",
                "FINAL ANSWER: A",
            ]

        elif "medical diagnosis" in question.lower():
            correct = "false negatives"
            variants_raw = [
                "false negatives",
                "false negatives",
                "false negatives",
                "minimize false negatives",
                "False negatives should be minimized",
                "FN - missing a disease is worse",
                "false negatives",
                "Type II errors",
                "Prioritize sensitivity",
            ]
            variants_struct = [
                "false negatives",
                "false negatives",
                "false negatives",
                "false negatives",
                "false negatives",
                "FINAL ANSWER: false negatives",
            ]

        elif "23 people" in question and "birthday" in question.lower():
            correct = "above 50%"
            variants_raw = [
                "above 50%",
                "above 50%",
                "above 50%",
                "greater than 50%",
                "more than 50%",
                "surprisingly high - over 50%",
                "about 50.7%",
                "above 50% (birthday paradox)",
            ]
            variants_struct = [
                "above 50%",
                "above 50%",
                "above 50%",
                "above 50%",
                "FINAL ANSWER: above 50%",
            ]

        else:
            correct = "unknown"
            variants_raw = ["I don't know", "unsure", "cannot determine"]
            variants_struct = ["FINAL ANSWER: unknown", "unknown"]

        # Select response based on mode
        if is_structured:
            # Lower variance for structured prompts
            # ~90% select from first 6 (high agreement), ~10% slight variation
            if random.random() < 0.9:
                idx = random.randint(0, min(5, len(variants_struct) - 1))
            else:
                idx = random.randint(0, len(variants_struct) - 1)
            response = variants_struct[idx]
        else:
            # Higher variance for raw prompts
            # ~70% select from first 5, ~30% more variation
            if random.random() < 0.7:
                idx = random.randint(0, min(4, len(variants_raw) - 1))
            else:
                idx = random.randint(0, len(variants_raw) - 1)
            response = variants_raw[idx]

        return response, "high" if not is_structured else "low"

    @property
    def model_name(self) -> str:
        return "mock/simulated-variance"


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
# PARSING AND METRICS
# ============================================================================


def parse_answer(raw_output: str, answer_type: str) -> str:
    output = raw_output.strip()

    if "FINAL ANSWER:" in output.upper():
        parts = output.upper().split("FINAL ANSWER:")
        if len(parts) > 1:
            output = parts[1].strip()

    import re

    if answer_type == "numeric":
        numbers = re.findall(r"-?\d+\.?\d*", output)
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
        return output_lower.strip()[:50]

    elif answer_type == "categorical":
        first_line = output.split("\n")[0]
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
    backend: LLMBackend,
    question_data: dict,
    n_runs: int = 20,
    temperature: float = 0.0,
    verbose: bool = True,
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
        output, _, _ = backend.query(raw_prompt, temperature)
        parsed = parse_answer(output, answer_type)
        raw_results.append(parsed)
        if verbose and (i + 1) % 5 == 0:
            print(f"    Raw: {i+1}/{n_runs} complete")

    # Run structured prompts
    structured_prompt = STRUCTURED_PROMPT_TEMPLATE.format(question=question)
    structured_results = []
    for i in range(n_runs):
        output, _, _ = backend.query(structured_prompt, temperature)
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

    raw_correct = sum(1 for a in raw_results if check_correct(a, expected, answer_type)) / len(
        raw_results
    )
    structured_correct = sum(
        1 for a in structured_results if check_correct(a, expected, answer_type)
    ) / len(structured_results)

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
        variance_reduction_pct=variance_reduction,
    )


def run_full_benchmark(
    backend: LLMBackend,
    n_runs: int = 20,
    temperature: float = 0.0,
    questions: Optional[list] = None,
    verbose: bool = True,
) -> BenchmarkSummary:
    if questions is None:
        questions = VARIANCE_TEST_QUESTIONS

    print("=" * 70)
    print("REASONKIT VARIANCE REDUCTION BENCHMARK")
    print("=" * 70)
    print(f"Backend: {backend.model_name}")
    print(f"Temperature: {temperature}")
    print(f"Runs per question: {n_runs}")
    print(f"Total questions: {len(questions)}")
    print(f"Total API calls: {len(questions) * n_runs * 2}")
    print("=" * 70)

    results = []
    for q in questions:
        result = run_variance_test(backend, q, n_runs, temperature, verbose)
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
            "mean_structured_agreement": statistics.mean(
                r.structured_agreement_rate for r in cat_results
            ),
            "mean_improvement": statistics.mean(r.agreement_improvement for r in cat_results),
            "mean_variance_reduction": statistics.mean(
                r.variance_reduction_pct for r in cat_results
            ),
        }

    return BenchmarkSummary(
        timestamp=datetime.utcnow().isoformat(),
        model=backend.model_name,
        temperature=temperature,
        n_runs_per_question=n_runs,
        n_questions=len(questions),
        mean_raw_agreement_rate=mean_raw_agreement,
        mean_structured_agreement_rate=mean_structured_agreement,
        mean_agreement_improvement=mean_improvement,
        mean_variance_reduction_pct=mean_variance_reduction,
        by_category=by_category,
        question_results=[asdict(r) for r in results],
    )


def print_summary(summary: BenchmarkSummary):
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

    print("\nMean Agreement Rate (TARa):")
    print(f"  Raw prompts:        {raw_pct:.1f}%")
    print(f"  Structured prompts: {struct_pct:.1f}%")
    print(f"  Improvement:        +{improvement:.1f} percentage points")

    print(f"\nMean Variance Reduction: {summary.mean_variance_reduction_pct:.1f}%")

    raw_inconsistency = (1 - summary.mean_raw_agreement_rate) * 100
    structured_inconsistency = (1 - summary.mean_structured_agreement_rate) * 100

    print("\nInconsistency Rate:")
    print(f"  Raw prompts:        {raw_inconsistency:.1f}% inconsistent")
    print(f"  Structured prompts: {structured_inconsistency:.1f}% inconsistent")

    if raw_inconsistency > 0:
        relative_reduction = (
            (raw_inconsistency - structured_inconsistency) / raw_inconsistency
        ) * 100
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
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # JSON
    json_path = output_dir / f"variance_benchmark_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(asdict(summary), f, indent=2)
    print(f"\nJSON saved: {json_path}")

    # Markdown report
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

Based on academic literature:
- [arXiv:2408.04667](https://arxiv.org/abs/2408.04667) - "Non-Determinism of 'Deterministic' LLM Settings"
- [arXiv:2203.11171](https://arxiv.org/abs/2203.11171) - "Self-Consistency Improves Chain of Thought Reasoning"
- [arXiv:2511.20836](https://arxiv.org/abs/2511.20836) - "Structured Prompting Enables More Robust Evaluation"

### Protocol
1. Each question run **{summary.n_runs_per_question} times** with identical parameters
2. Temperature: **{summary.temperature}**
3. Two conditions: Raw prompts vs 5-step structured reasoning
4. Metric: TARa (Total Agreement Rate for parsed answers)

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

*ReasonKit Variance Benchmark v1.0*
"""

    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"Report saved: {md_path}")

    return json_path, md_path


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="ReasonKit Variance Benchmark")
    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        default="mock",
        choices=["anthropic", "openrouter", "mock"],
        help="LLM backend to use",
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None, help="Model name (backend-specific)"
    )
    parser.add_argument(
        "--runs", "-n", type=int, default=20, help="Runs per question (default: 20)"
    )
    parser.add_argument(
        "--temperature", "-t", type=float, default=0.0, help="Temperature (default: 0.0)"
    )
    parser.add_argument("--output-dir", "-o", type=str, default="results", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Quick test (5 runs, 3 questions)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    # Create backend
    if args.backend == "anthropic":
        model = args.model or "claude-sonnet-4-20250514"
        backend = AnthropicBackend(model)
    elif args.backend == "openrouter":
        model = args.model or "anthropic/claude-sonnet-4"
        backend = OpenRouterBackend(model)
    else:
        backend = MockBackend()

    n_runs = 5 if args.quick else args.runs
    questions = VARIANCE_TEST_QUESTIONS[:3] if args.quick else VARIANCE_TEST_QUESTIONS

    summary = run_full_benchmark(
        backend=backend,
        n_runs=n_runs,
        temperature=args.temperature,
        questions=questions,
        verbose=not args.quiet,
    )

    print_summary(summary)
    save_results(summary, Path(args.output_dir))

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
