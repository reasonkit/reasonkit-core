import json
import subprocess
import os
import sys
from typing import Optional, List, Dict, Any

from langchain_core.tools import BaseTool

# Pydantic v1 is used by LangChain < 0.2, but v2 is standard now.
# We'll try to use v1 if available (for compatibility) or fallback.
try:
    from langchain_core.pydantic_v1 import BaseModel, Field
except ImportError:
    from pydantic import BaseModel, Field

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.llms import FakeListLLM

# --- Configuration ---
# Path to your compiled rk-core binary
RK_CORE_PATH = os.environ.get("RK_CORE_PATH", "../target/release/rk-core")

if not os.path.exists(RK_CORE_PATH):
    # Try current directory if relative path fails
    if os.path.exists("./target/release/rk-core"):
        RK_CORE_PATH = "./target/release/rk-core"
    else:
        print(f"Error: rk-core binary not found at {RK_CORE_PATH}")
        print("Please build it first: cargo build --release")
        sys.exit(1)

# --- 1. Define the ReasonKit Tool Wrapper ---


class ReasonKitInput(BaseModel):
    query: str = Field(
        description="The input query, claim, or problem to reason about."
    )
    profile: str = Field(
        default="quick",
        description="The reasoning profile: 'quick', 'balanced', 'scientific', 'paranoid'.",
    )


class ReasonKitTool(BaseTool):
    name = "reasonkit_think"
    description = (
        "Use this tool for complex reasoning, brainstorming, or verification. "
        "It runs a structured 'ThinkTool' protocol (e.g., GigaThink -> LaserLogic). "
        "Returns a structured analysis with confidence scores."
    )
    args_schema: type[BaseModel] = ReasonKitInput

    def _run(self, query: str, profile: str = "quick") -> str:
        """Executes the rk-core binary and returns the structured output."""
        try:
            # Construct command: rk-core think "query" --profile profile --format json --mock
            # We use --mock here for demonstration to avoid needing API keys.
            # In production, remove --mock to use real LLMs defined in your config.
            cmd = [
                RK_CORE_PATH,
                "think",
                query,
                "--profile",
                profile,
                "--format",
                "json",
                "--mock",
            ]

            # Run the Rust binary
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Parse the JSON output
            data = json.loads(result.stdout)

            # --- Format the output for the Agent ---
            # We want to give the LLM a concise summary of the structured reasoning.
            summary = []
            summary.append(f"--- ReasonKit Analysis ({profile}) ---")
            summary.append(f"Confidence: {data.get('confidence', 0.0) * 100:.1f}%")

            # Summarize steps
            steps = data.get("steps", [])
            summary.append(f"Steps executed: {len(steps)}")
            for step in steps:
                status = "✓" if step.get("success") else "✗"
                summary.append(f"- {step.get('step_id')}: {status}")

            # Extract key findings from the protocol output
            # (Adapting to the generic JSON output structure of rk-core)
            summary.append("\nKey Findings:")
            output_data = data.get("data", {})

            for key, val in output_data.items():
                if key == "confidence":
                    continue

                # Handle text vs complex objects
                content = ""
                if isinstance(val, dict) and "content" in val:
                    content = val["content"]
                elif isinstance(val, dict) and "items" in val:
                    # Handle lists (like from GigaThink)
                    items = [item.get("content", "") for item in val["items"]]
                    content = "\n  * ".join(items[:5])  # Show first 5 items
                else:
                    content = str(val)

                # Truncate long content for the context window
                if len(content) > 300:
                    content = content[:300] + "..."

                summary.append(f"{key}:\n{content}\n")

            return "\n".join(summary)

        except subprocess.CalledProcessError as e:
            return f"Error executing ReasonKit: {e.stderr}"
        except json.JSONDecodeError:
            return f"Error parsing ReasonKit output: {result.stdout}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"


# --- 2. Run the Agent Demo ---


def run_agent_demo():
    print("🤖 Initializing Glass Box Agent with ReasonKit Tools...")
    print(f"   Using binary: {RK_CORE_PATH}")

    # Initialize the tool
    rk_tool = ReasonKitTool()
    tools = [rk_tool]

    # Mock LLM for demonstration (Deterministic)
    # This simulates how an agent would decide to use the tool.
    llm = FakeListLLM(
        responses=[
            # Thought 1
            "The user is asking about the ethical implications of AGI. This is a complex topic requiring structured analysis. I should use ReasonKit.",
            # Action 1 (Calling the tool)
            "Action: reasonkit_think\nAction Input: 'Analyze the ethical implications of AGI' --profile balanced",
            # Thought 2 (After receiving tool output)
            "Observation: [ReasonKit Output]\nBased on the structured analysis from ReasonKit, I can now synthesize a comprehensive answer.",
            # Final Answer
            "Final Answer: The ethical implications of AGI are multifaceted. ReasonKit's analysis highlights several key dimensions:\n\n1. Safety and Alignment: Ensuring AGI goals match human values.\n2. Economic Impact: Potential for massive labor displacement.\n3. Bias and Fairness: Risk of amplifying existing societal biases.\n\nThe 'balanced' profile analysis provides a confidence of 85% in these findings.",
        ]
    )

    # Standard ReAct Prompt
    template = """Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)

    # Create the Agent
    # Note: handle_parsing_errors=True is important when using FakeListLLM or weaker models
    # that might produce outputs not perfectly matching the expected format.
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )

    # Run the Agent
    query = "Analyze the ethical implications of AGI using a balanced perspective."

    print(f"\n❓ User Query: {query}")
    print("-" * 60)

    try:
        # We manually demonstrate the tool execution first to see the real output
        print("\n[DEMO] Manually executing the ReasonKit tool to show real output:")
        real_output = rk_tool.run({"query": query, "profile": "balanced"})
        print(f"\n📄 REAL TOOL OUTPUT (Glass Box Artifact):\n{real_output}")
        print("-" * 60)

        # Now run the agent simulation
        print("\n[DEMO] Running Agent Loop:")
        agent_executor.invoke({"input": query})

    except Exception as e:
        print(f"Agent execution failed: {e}")


if __name__ == "__main__":
    run_agent_demo()
