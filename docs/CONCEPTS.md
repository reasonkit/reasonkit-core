# Core Concepts: The Glass Box Architecture

> **"Stop hoping your LLM 'gets it.' Start engineering how it thinks."**

## The "Glass Box" vs. "Black Box"

The fundamental value of ReasonKit is the shift from opaque probabilistic generation to transparent, auditable cognitive processes.

### 1. The Black Box (Standard LLM)
You pour context in, and get a result out. You have no idea if it hallucinated, skipped steps, or just got lucky.

```mermaid
graph LR
    A[User Prompt] --> B{Black Box LLM}
    B --> C[Output]
    B -.-> D[Hidden Hallucinations?]
    B -.-> E[Skipped Logic?]
```

### 2. The Glass Box (ReasonKit)
Every step of the thinking process is exposed, structured, and verified.

```mermaid
graph TD
    User[User Prompt] --> RK[ReasonKit Engine]
    
    subgraph "Visible Cognitive Pipeline"
    RK --> Div[1. Divergent Thinking]
    Div --> |10+ Perspectives| Conv[2. Convergent Analysis]
    Conv --> |Trade-offs| Ground[3. Grounding]
    Ground --> |First Principles| Valid[4. Validation]
    Valid --> |Feasibility Check| Cut[5. Ruthless Cutting]
    end
    
    Cut --> Output[Verified Output]
    
    style Div fill:#e1f5fe,stroke:#01579b
    style Conv fill:#e8f5e9,stroke:#1b5e20
    style Ground fill:#fff3e0,stroke:#e65100
    style Valid fill:#f3e5f5,stroke:#4a148c
    style Cut fill:#ffebee,stroke:#b71c1c
```

---

## The 5-Step Integrity Process

This process is hard-coded into our protocols. It forces the AI to behave like a senior engineer, not a junior chatbot.

### Step 1: Divergent Thinking (Many Ideas)
*   **Module:** `GigaThink`
*   **Action:** Forces the generation of at least 10 distinct perspectives or dimensions.
*   **Why:** Prevents "tunnel vision" and the availability heuristic.

### Step 2: Convergent Analysis (Prioritization)
*   **Module:** `LaserLogic`
*   **Action:** Applies strict criteria to filter the noise.
*   **Why:** Raw creativity is useless without discernment.

### Step 3: Grounding (First Principles)
*   **Module:** `BedRock`
*   **Action:** Decomposes remaining ideas to their fundamental axioms.
*   **Why:** Ensures solutions aren't built on shaky assumptions.

### Step 4: Validation (Feasibility Check)
*   **Module:** `ProofGuard`
*   **Action:** Triangulates claims against external sources or logical rules.
*   **Why:** Trust, but verify. Then verify again.

### Step 5: Ruthless Cutting (Focus)
*   **Module:** `BrutalHonesty`
*   **Action:** Strips away fluff, hedging, and "AI slop."
*   **Why:** The best code is no code. The best answer is the shortest correct one.

---

## Integration Architecture

How ReasonKit fits into your existing agent stack (e.g., LangChain, AutoGen).

```mermaid
sequenceDiagram
    participant User
    participant Agent as LangChain Agent
    participant RK as ReasonKit (Glass Box)
    participant LLM as Raw LLM
    
    User->>Agent: "Analyze this complex risk"
    
    Agent->>Agent: Decides to use tool
    
    Agent->>RK: Invoke "rk-think --paranoid"
    activate RK
    
    RK->>LLM: Step 1: Generate Perspectives
    LLM-->>RK: [List of 10 items]
    
    RK->>LLM: Step 2: Critique & Filter
    LLM-->>RK: [Top 3 Risks]
    
    RK->>LLM: Step 3: Verify Sources
    LLM-->>RK: [Triangulated Facts]
    
    RK-->>Agent: Returns STRUCTURED JSON Artifact
    deactivate RK
    
    Agent->>User: Final Answer (referencing the Artifact)
```
