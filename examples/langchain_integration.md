# ReasonKit <-> LangChain Integration: "The Glass Box Agent"

This prototype demonstrates how **ReasonKit** (a Rust-based verified reasoning engine) can be integrated into **LangChain** (a Python agent framework) to create a "Glass Box" agent.

Unlike standard "Black Box" agents that think obscurely, this agent delegates critical reasoning steps to ReasonKit's verified `ThinkTools`, providing:

1. **Structured Reasoning**: Chains like `GigaThink` (divergent) -> `LaserLogic` (convergent).
2. **Auditability**: Every reasoning step is a structured artifact, not just text stream.
3. **Verification**: The `ProofGuard` tool provides 3-source triangulation for claims.

## Prerequisites

- Python 3.10+
- `uv` (recommended) or `pip`
- A built `rk` binary (from the Rust project)

## Setup

```bash
# 1. Install dependencies
uv pip install langchain langchain-core langchain-community

# 2. Ensure rk is in your PATH or point to it directly
export RK_CORE_PATH="../target/release/rk"
```

## The Integration Code (`rk_langchain.py`)

```python
import json
import subprocess
import os
from typing import Optional, List, Dict, Any
from langchain_core.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.llms import FakeListLLM  # For demo without API keys

# --- Configuration ---
RK_CORE_PATH = os.environ.get("RK_CORE_PATH", "./target/release/rk")

# --- 1. Define the ReasonKit Tool Wrapper ---

class ReasonKitInput(BaseModel):
    query: str = Field(description="The input query, claim, or problem to reason about.")
    profile: str = Field(default="quick", description="The reasoning profile to use: 'quick', 'balanced', 'scientific', 'paranoid'.")

class ReasonKitTool(BaseTool):
    name = "reasonkit_think"
    description = (
        "Use this tool for complex reasoning, brainstorming, or verification. "
        "It runs a structured 'ThinkTool' protocol (e.g., GigaThink -> LaserLogic). "
        "Returns a structured analysis with confidence scores."
    )
    args_schema: type[BaseModel] = ReasonKitInput

    def _run(self, query: str, profile: str = "quick") -> str:
        """Executes the rk binary and returns the structured output."""
        try:
            # Construct command: rk think "query" --profile profile --format json --mock
            # Note: --mock is used here to avoid needing live API keys for this demo.
            # In production, remove --mock and provide provider credentials.
            cmd = [
                RK_CORE_PATH,
                "think",
                query,
                "--profile", profile,
                "--format", "json",
                "--mock"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)

            # Format the output for the LLM to understand
            summary = []
            summary.append(f"--- ReasonKit Analysis ({profile}) ---")
            summary.append(f"Confidence: {data.get('confidence', 0.0) * 100:.1f}%")

            # Extract key steps (simplified for the agent's context)
            steps = data.get('steps', [])
            for step in steps:
                step_id = step.get('step_id')
                success = "✓" if step.get('success') else "✗"
                summary.append(f"Step {step_id}: {success}")

            # Extract the final synthesized output if available
            # (The JSON structure depends on your specific rk output schema)
            # This is a generic fallback:
            summary.append("\nDetailed Findings:")
            output_data = data.get('data', {})
            for key, val in output_data.items():
                if isinstance(val, dict) and 'content' in val:
                     summary.append(f"{key}: {val['content'][:200]}...") # Truncate for brevity
                else:
                     summary.append(f"{key}: {str(val)[:200]}...")

            return "\n".join(summary)

        except subprocess.CalledProcessError as e:
            return f"Error executing ReasonKit: {e.stderr}"
        except json.JSONDecodeError:
            return f"Error parsing ReasonKit output: {result.stdout}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

# --- 2. Build the Agent ---

def run_agent_demo():
    print("🤖 Initializing Glass Box Agent with ReasonKit Tools...")

    # Initialize the tool
    rk_tool = ReasonKitTool()
    tools = [rk_tool]

    # For this demo, we use a Fake LLM that "knows" to use the tool.
    # In a real scenario, use ChatOpenAI, ChatAnthropic, etc.
    llm = FakeListLLM(responses=[
        "I need to use complex reasoning for this. I will use ReasonKit.",
        "Action: reasonkit_think\nAction Input: 'What are the ethical implications of AGI?'",
        "Observation: [ReasonKit Output]",
        "Based on the structured analysis, here is the answer..."
    ])

    # Simple ReAct Prompt
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

    # Construct the Agent
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Run the Agent
    query = "Analyze the ethical implications of AGI using a balanced perspective."
    print(f"\n❓ User Query: {query}")

    # We mock the execution flow since FakeListLLM is static,
    # but this shows how the tool is integrated.
    try:
        # In a real run with a real LLM, this would call the tool automatically.
        # Here we manually invoke the tool to show it working.
        print("\n⚙️  Agent decides to call ReasonKit...")
        tool_output = rk_tool.run({"query": query, "profile": "balanced"})
        print(f"\n📄 Tool Output (ReasonKit Artifact):\n{tool_output}")

    except Exception as e:
        print(f"Agent execution failed: {e}")

if __name__ == "__main__":
    run_agent_demo()
```

## How It Works

1. **Tool Registration**: We subclass `BaseTool` to create `ReasonKitTool`. This exposes the `rk think` CLI command as a callable function within Python.
2. **Structured Execution**: When the Python agent calls the tool, it shells out to the optimized Rust binary.
3. **Config-Driven**: The agent can select profiles (`quick`, `paranoid`) defined in your YAML configuration, leveraging the work we just finished.
4. **Artifact Return**: Instead of just text, the tool returns a summary of the *verification steps* (Confidence, Step Success), making the reasoning "Glass Box".
