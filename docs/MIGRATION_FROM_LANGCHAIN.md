# Migrating from LangChain to ReasonKit

This guide is designed for developers moving from LangChain (Python/TypeScript) to ReasonKit (Rust). While both frameworks help build LLM applications, ReasonKit focuses on **structured reasoning protocols** rather than generic chains of thought.

## Conceptual Mapping

| LangChain Concept           | ReasonKit Equivalent              | Key Difference                                                                    |
| --------------------------- | --------------------------------- | --------------------------------------------------------------------------------- |
| **Chain** (`LLMChain`)      | **Protocol** (`Protocol`)         | Protocols are static, declarative, and auditable (TOML/YAML), not arbitrary code. |
| **Prompt Template**         | **ThinkTool** (`ProtocolStep`)    | ReasonKit prompts are structured cognitive steps (Analyze, Critique, Synthesize). |
| **Memory** (`BufferMemory`) | **Context** (`ProtocolInput`)     | State is passed explicitly between steps; no hidden global state.                 |
| **Agent**                   | **Executor** (`ProtocolExecutor`) | Executors run defined protocols; less "magic", more predictability.               |
| **Output Parser**           | **Schema Validation**             | Native Rust type safety and JSON schema validation at every step.                 |
| **Retriever**               | **Retrieval Module**              | High-performance Rust-based retrieval (Qdrant/Tantivy) integrated directly.       |

## Core Architectural Shifts

1.  **Code vs. Configuration:**
    - _LangChain:_ Logic is often defined in Python/JS code (loops, conditionals).
    - _ReasonKit:_ Logic is defined in declarative TOML/YAML files. Rust code is the _engine_ that runs them.

2.  **Dynamic vs. Static:**
    - _LangChain:_ Flexible, often "yolo" execution flow.
    - _ReasonKit:_ Structured, predictable, type-safe execution.

3.  **Prompting Philosophy:**
    - _LangChain:_ "Ask the LLM to do X."
    - _ReasonKit:_ "Apply Cognitive Module Y (e.g., GigaThink) to X."

## Migration Step-by-Step

### 1. Simple Chain Migration

**LangChain (Python):**

```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

llm = OpenAI(temperature=0.9)
chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run("colorful socks"))
```

**ReasonKit (TOML Protocol):**

Create `protocols/naming_tool.toml`:

```toml
version = "2.0.0"
schema = "thinktool"

[thinktool_modules.naming_tool]
id = "naming_tool"
name = "Company Namer"
description = "Generates creative company names"
category = "divergent"

[[thinktool_modules.naming_tool.steps]]
id = "brainstorm"
action = { type = "generate", min_count = 5, max_count = 10 }
prompt_template = """
Generate creative company names for a company that makes: {{query}}

Provide a list of 5-10 names with brief rationales.
"""
output_format = "list"
```

**Running it (CLI):**

```bash
rk think naming_tool "colorful socks"
```

### 2. Sequential Chain (Multi-Step)

**LangChain:**

- Step 1: Generate synopsis.
- Step 2: Write review based on synopsis.

**ReasonKit:**
Define multiple steps in your TOML.

```toml
[[thinktool_modules.reviewer.steps]]
id = "synopsis"
action = { type = "analyze" }
prompt_template = "Write a synopsis for: {{query}}"
output_format = "text"

[[thinktool_modules.reviewer.steps]]
id = "review"
action = { type = "critique", severity = "standard" }
prompt_template = """
Write a review based on this synopsis:
{{synopsis}}
"""
output_format = "text"
depends_on = ["synopsis"]
```

### 3. RAG (Retrieval Augmented Generation)

**LangChain:**
`RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=docsearch.as_retriever())`

**ReasonKit:**
RAG is a native capability of the `ProtocolExecutor`. You simply use a protocol that includes a retrieval step or pass context.

- _Note:_ ReasonKit Core provides the **engine**. You configure the storage (Qdrant/Local) in `config.toml`.

## Best Practices for Migration

1.  **Audit your Prompts:** Don't just copy-paste. ReasonKit works best when you decompose big prompts into atomic cognitive steps (Analyze -> Critique -> Synthesize).
2.  **Define Inputs:** Clearly specify what your protocol expects (`{{query}}`, `{{context}}`).
3.  **Leverage Structure:** Use `output_format = "structured"` (JSON) whenever possible to get type-safe outputs you can trust.
4.  **Test with CLI:** Use the `rk` CLI to test your new protocols quickly before integrating them into your application code.

## Why Switch?

- **Performance:** Rust engine overhead is measured in microseconds, not milliseconds.
- **Safety:** Strict type checking and validation prevent runtime "hallucination format" errors.
- **Deployment:** Single binary deployment. No massive Python venv dependency hell.
- **Auditability:** Protocols are data, not code. Easier to version, diff, and review.
