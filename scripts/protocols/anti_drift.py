#!/usr/bin/env python3
"""
Dynamic Beta Protocol: Anti-Drift with Saturation Trigger

Implements the Dynamic Beta protocol for research tasks with automatic
saturation detection to prevent "Zombie Researcher" syndrome.

Uses zlib-based compression analysis to detect information gain.
When new information adds <5% entropy, the Anchor relaxes.

License: Apache-2.0
"""

import argparse
import json
import zlib
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ResearchState:
    """Current state of research session"""

    anchor: str
    accumulated_content: str = ""
    step_count: int = 0
    low_gain_steps: int = 0
    saturation_triggered: bool = False
    mode: str = "RESEARCH"  # RESEARCH | PIVOT | SYNTHESIS
    entities_found: list = field(default_factory=list)
    insights: list = field(default_factory=list)
    history: list = field(default_factory=list)


def calculate_compression_ratio(content: str) -> float:
    """Calculate compression ratio as proxy for information density"""
    if not content:
        return 0.0
    original_size = len(content.encode("utf-8"))
    compressed_size = len(zlib.compress(content.encode("utf-8")))
    return compressed_size / original_size


def calculate_information_gain(old_content: str, new_content: str) -> float:
    """
    Calculate information gain using compression delta.

    If new content adds little compressible information (i.e., it's redundant),
    the compression ratio won't change much = low information gain.
    """
    if not old_content:
        return 1.0  # First content is always high gain

    combined = old_content + "\n" + new_content

    old_ratio = calculate_compression_ratio(old_content)
    combined_ratio = calculate_compression_ratio(combined)

    # Information gain = how much the compression ratio changes
    # Higher delta = more unique/compressible content = more information
    delta = abs(combined_ratio - old_ratio)

    # Normalize: typical range is 0.0-0.1, map to 0.0-1.0
    normalized_gain = min(delta * 10, 1.0)

    return normalized_gain


def check_saturation(
    state: ResearchState, new_content: str, threshold: float = 0.3, max_low_gain_steps: int = 3
) -> tuple[float, bool]:
    """
    Check if research has reached saturation.

    Returns:
        (information_gain, is_saturated)
    """
    gain = calculate_information_gain(state.accumulated_content, new_content)

    if gain < threshold:
        state.low_gain_steps += 1
    else:
        state.low_gain_steps = 0  # Reset on good gain

    is_saturated = state.low_gain_steps >= max_low_gain_steps

    return gain, is_saturated


def process_research_step(
    state: ResearchState, new_content: str, new_entities: list = None, new_insights: list = None
) -> dict:
    """
    Process a research step and check for saturation.

    Returns status dict with recommendations.
    """
    state.step_count += 1

    # Calculate information gain
    gain, is_saturated = check_saturation(state, new_content)

    # Update state
    state.accumulated_content += "\n" + new_content
    if new_entities:
        state.entities_found.extend(new_entities)
    if new_insights:
        state.insights.extend(new_insights)

    # Record history
    state.history.append(
        {
            "step": state.step_count,
            "gain": round(gain, 3),
            "low_gain_steps": state.low_gain_steps,
            "mode": state.mode,
            "timestamp": datetime.now().isoformat(),
        }
    )

    # Handle saturation
    if is_saturated and not state.saturation_triggered:
        state.saturation_triggered = True
        state.mode = "PIVOT"

        return {
            "status": "SATURATION_TRIGGERED",
            "information_gain": round(gain, 3),
            "low_gain_steps": state.low_gain_steps,
            "recommendation": "PIVOT",
            "pivot_directions": [
                "Synthesize patterns from collected data",
                "Analyze relationships between entities",
                "Generate insights about ecosystem dynamics",
                "Identify gaps in coverage and their significance",
            ],
            "message": f"[Dynamic Beta] SATURATION DETECTED after {state.step_count} steps. "
            f"Information gain ({gain:.2%}) below threshold for {state.low_gain_steps} consecutive steps. "
            f"Relaxing anchor to allow lateral pivot.",
        }

    # Normal progress
    status = "WARNING" if state.low_gain_steps > 0 else "OK"

    return {
        "status": status,
        "information_gain": round(gain, 3),
        "low_gain_steps": state.low_gain_steps,
        "mode": state.mode,
        "entities_found": len(state.entities_found),
        "insights_count": len(state.insights),
        "message": f"[Dynamic Beta] Step {state.step_count}: Gain={gain:.2%}, "
        f"Status={status}, Mode={state.mode}",
    }


def run_anti_drift_session(anchor: str, content_stream: list) -> dict:
    """
    Run a complete anti-drift session.

    Args:
        anchor: Primary research goal
        content_stream: List of content strings (one per research step)

    Returns:
        Final session report
    """
    state = ResearchState(anchor=anchor)
    results = []

    print(f'[Dynamic Beta] Anchor: "{anchor}"')
    print("[Dynamic Beta] Starting research session...")
    print("-" * 60)

    for i, content in enumerate(content_stream):
        result = process_research_step(state, content)
        results.append(result)
        print(result["message"])

        if result["status"] == "SATURATION_TRIGGERED":
            print("\n[Dynamic Beta] PIVOTING TO SYNTHESIS MODE")
            print("Pivot directions:")
            for direction in result["pivot_directions"]:
                print(f"  - {direction}")
            break

    print("-" * 60)

    return {
        "anchor": anchor,
        "total_steps": state.step_count,
        "final_mode": state.mode,
        "saturation_triggered": state.saturation_triggered,
        "entities_found": len(state.entities_found),
        "insights": len(state.insights),
        "history": state.history,
        "step_results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Dynamic Beta Protocol: Anti-Drift with Saturation Trigger"
    )
    parser.add_argument("--anchor", "-a", required=True, help="Research anchor/goal")
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.3,
        help="Information gain threshold (default: 0.3)",
    )
    parser.add_argument(
        "--max-low-gain",
        "-m",
        type=int,
        default=3,
        help="Max consecutive low-gain steps before trigger (default: 3)",
    )
    parser.add_argument("--demo", action="store_true", help="Run demo mode with sample data")
    parser.add_argument("--output", "-o", help="Output JSON file")

    args = parser.parse_args()

    if args.demo:
        # Demo with synthetic data showing saturation pattern
        demo_content = [
            "LangChain is a framework for building LLM applications. It has agents, chains, and memory.",
            "AutoGen by Microsoft enables multi-agent conversations. It supports code execution.",
            "CrewAI orchestrates AI agents with roles. It has task delegation features.",
            "DSPy from Stanford optimizes LLM prompts programmatically.",
            "Semantic Kernel from Microsoft integrates LLMs with plugins.",
            # Saturation starts here - redundant/low-value content
            "Another framework called AgentX does similar things to LangChain.",
            "Framework Y also has agents and chains like LangChain.",
            "Yet another tool Z provides LLM orchestration capabilities.",
        ]

        result = run_anti_drift_session(args.anchor, demo_content)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to {args.output}")

        return result

    else:
        # Interactive mode - read content from stdin
        print(f'[Dynamic Beta] Anchor: "{args.anchor}"')
        print("Enter research content (one paragraph per step, empty line to finish):")
        print("-" * 60)

        state = ResearchState(anchor=args.anchor)

        while True:
            try:
                content = input(f"[Step {state.step_count + 1}] > ")
                if not content.strip():
                    break

                result = process_research_step(state, content)
                print(result["message"])

                if result["status"] == "SATURATION_TRIGGERED":
                    print("\n[Dynamic Beta] Session complete - saturation reached")
                    break

            except EOFError:
                break

        print(f"\n[Dynamic Beta] Final: {state.step_count} steps, Mode: {state.mode}")


if __name__ == "__main__":
    main()
