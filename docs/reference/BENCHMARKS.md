# ReasonKit Reasoning Arena: Community Benchmarks

> **"Don't just vibe check. Verify."**

The **Reasoning Arena** is a proposed standard for evaluating AI reasoning capabilities, moving beyond simple "Chatbot Arena" style preference voting toward verifiable, structural integrity metrics.

## The 5-Step Integrity Check

Unlike raw LLM outputs which are opaque "black boxes," ReasonKit enforces a visible 5-step cognitive pipeline. We benchmark against this standard:

1.  **Divergent Thinking:** Did the system explore at least 5 distinct angles before answering?
2.  **Convergent Analysis:** Did it explicitly weigh trade-offs using defined criteria?
3.  **Grounding:** Can every claim be traced to a first principle or source?
4.  **Validation:** Did it run a self-correction loop _before_ outputting?
5.  **Ruthless Cutting:** Is the final output free of fluff and focused on value?

---

## Standard "Hard Reasoning" Prompts

We propose these 5 prompts to test any reasoning engine (Raw LLM vs. ReasonKit).

### 1. The Policy Loophole Test

**Prompt:**

> "Analyze this policy statement for potential exploit vectors: 'Employees may expense meals up to $50 per day when traveling for business, provided receipts are submitted within 30 days.'"

- **Success Criteria:** Identifies edge cases (e.g., splitting receipts, timezone definitions, "traveling" definition, currency conversion exploits).
- **ReasonKit Goal:** Use `rk-think --paranoid` to map the threat surface.

### 2. The First Principles Decomposition

**Prompt:**

> "Decompose the statement 'We need a blockchain for supply chain transparency' into its fundamental assumptions. Which assumptions are empirical and which are faith-based?"

- **Success Criteria:** Breaks down trust boundaries, immutable ledger requirements, and centralization risks without jargon.
- **ReasonKit Goal:** Use `rk-br` (BedRock) to isolate axioms.

### 3. The Triangulated Fact Check

**Prompt:**

> "Verify the claim: 'The 2024 EU AI Act bans all biometric identification systems in public spaces.'"

- **Success Criteria:** Correctly identifies the _exceptions_ (law enforcement, specific crimes) rather than a blanket "yes/no". Requires citing specific articles.
- **ReasonKit Goal:** Use `rk-core verify` with 3-source rule.

### 4. The Adversarial Stress Test

**Prompt:**

> "Propose a plan to migrate a monolithic bank system to microservices. Then, act as a hostile CTO and destroy that plan."

- **Success Criteria:** The critique must be specific (latency, data consistency, atomic transactions) and not generic.
- **ReasonKit Goal:** Use `rk-bh` (BrutalHonesty) to steelman then attack.

### 5. The "Impossible" Synthesis

**Prompt:**

> "Synthesize a strategy that satisfies both 'Move Fast and Break Things' and 'Zero Trust Security' for a medical device startup."

- **Success Criteria:** Identifies the specific non-overlapping domains where each applies (e.g., "Break things in dev environment," "Zero trust in production firmware").
- **ReasonKit Goal:** Use `rk-think --balanced` to find the dialectic synthesis.

---

## Scoring Methodology

We score responses on the **Reasoning Integrity Score (RIS)** (0-100):

- **Structure (30%):** Is the reasoning path visible?
- **Breadth (20%):** Were alternatives considered?
- **Depth (20%):** Did it reach root causes?
- **Verification (30%):** Are facts triangulated?

**Target:** Raw GPT-4 typically scores ~60. ReasonKit targets >85.
