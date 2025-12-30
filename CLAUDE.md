# REASONKIT-CORE PROJECT CONTEXT

> Structured Prompt Engineering Framework | Auditable AI Reasoning
> "See How Your AI Thinks"

**LICENSE:** Apache 2.0 (fully open source)
**REPOSITORY:** https://github.com/ReasonKit/reasonkit-core
**WEBSITE:** https://reasonkit.sh

---

## WHAT REASONKIT ACTUALLY IS

**ReasonKit is a structured prompt engineering framework with execution tracing and metrics.**

It provides reusable reasoning patterns that organize LLM outputs into auditable, traceable chains.

### The Honest Value Proposition

| What We Claim         | What We Deliver                           | Status       |
| --------------------- | ----------------------------------------- | ------------ |
| Structured reasoning  | Prompt templates that guide output format | ✅ Delivered |
| Auditable traces      | Full execution logging with metrics       | ✅ Delivered |
| Quality measurement   | Confidence scores, grades, reports        | ✅ Delivered |
| Reasoning improvement | **Run benchmarks to verify**              | 🔬 Testing   |

---

## THE POWERCOMBO PROCESS (CENTRAL VALUE)

This 5-step structured thinking process is the core of ReasonKit:

```
┌─────────────────────────────────────────────────────────────────┐
│                    🌈 POWERCOMBO PROCESS                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 💡 DIVERGENT THINKING (GigaThink)                          │
│     → Generate 10+ perspectives, explore widely                 │
│                                                                 │
│  2. ⚡ CONVERGENT ANALYSIS (LaserLogic)                        │
│     → Validate logic, detect fallacies, prioritize             │
│                                                                 │
│  3. 🪨 GROUNDING (BedRock)                                     │
│     → First principles decomposition, find axioms              │
│                                                                 │
│  4. 🛡️ VALIDATION (ProofGuard)                                │
│     → Verify claims, triangulate sources, check evidence       │
│                                                                 │
│  5. 🔥 RUTHLESS CUTTING (BrutalHonesty)                        │
│     → Adversarial critique, cut the fluff, be honest           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Process Works

- **Divergent → Convergent**: Explore widely, then focus ruthlessly
- **Abstract → Concrete**: From ideas to first principles to evidence
- **Constructive → Destructive**: Build up, then attack your own work
- **Traceable**: Every step logged, every decision auditable

---

## THINKTOOLS (STRUCTURED PROMPTS)

| Tool              | Icon | Purpose                     | Output               |
| ----------------- | ---- | --------------------------- | -------------------- |
| **GigaThink**     | 💡   | Multi-perspective expansion | 10+ viewpoints       |
| **LaserLogic**    | ⚡   | Logical validation          | Fallacy detection    |
| **BedRock**       | 🪨   | First principles            | Axiom identification |
| **ProofGuard**    | 🛡️   | Evidence verification       | Source triangulation |
| **BrutalHonesty** | 🔥   | Adversarial critique        | Honest assessment    |
| **PowerCombo**    | 🌈   | All 5 in sequence           | Maximum rigor        |

### Profiles (Pre-configured Chains)

| Profile        | Chain                    | Confidence Target | Use Case      |
| -------------- | ------------------------ | ----------------- | ------------- |
| `--quick`      | gt → ll                  | 70%               | Fast analysis |
| `--balanced`   | gt → ll → br → pg        | 80%               | Standard      |
| `--deep`       | All 5                    | 85%               | Thorough      |
| `--paranoid`   | All 5 + validation pass  | 95%               | Maximum rigor |
| `--powercombo` | All 5 + cross-validation | 95%               | Ultimate mode |

---

## PROVING VALUE (BENCHMARKS)

**We don't claim improvement without evidence.**

```bash
# Run GSM8K benchmark
cargo run --release --bin gsm8k_eval -- --samples 100

# A/B comparison
rk-compare "Should we use microservices?" --profile balanced

# View metrics
rk-core metrics report
```

### What We Measure

| Benchmark | What It Tests     | How to Run                    |
| --------- | ----------------- | ----------------------------- |
| GSM8K     | Math reasoning    | `cargo run --bin gsm8k_eval`  |
| ARC-C     | Science reasoning | `cargo run --bin arc_c_eval`  |
| LogiQA    | Logical deduction | `cargo run --bin logiqa_eval` |

### Interpreting Results

| Delta | Meaning                          |
| ----- | -------------------------------- |
| > +5% | ✅ Meaningful improvement        |
| +1-5% | ⚠️ Marginal, verify cost-benefit |
| 0%    | ⚪ No measurable difference      |
| < 0%  | ❌ ThinkTools performed worse    |

---

## THE REAL VALUE

### What ReasonKit IS Good For:

1. **Debugging AI Responses** - See exactly how reasoning unfolded
2. **Compliance/Audit** - Traceable decision chains for regulated industries
3. **Structured Output** - Consistent format across queries
4. **Quality Metrics** - Measure and track reasoning quality over time
5. **Teaching Tool** - Learn structured thinking patterns

### What ReasonKit is NOT:

- Magic that makes LLMs smarter
- Novel AI research
- A replacement for good prompting skills
- Guaranteed improvement (run benchmarks!)

---

## PROJECT STRUCTURE

```
reasonkit-core/
├── src/
│   ├── thinktool/       # ThinkTools (structured prompts)
│   │   ├── executor.rs  # Protocol chain runner
│   │   ├── profiles.rs  # Profile definitions
│   │   ├── metrics.rs   # Quality measurement
│   │   ├── trace.rs     # Execution tracing
│   │   └── llm.rs       # LLM integration
│   │
│   │
│   └── verification/    # ProofLedger, source tracking
│
├── benchmarks/          # Reproducible evaluation
│   ├── gsm8k_eval.rs    # Math reasoning benchmark
│   └── README.md        # Benchmark documentation
│
└── protocols/           # YAML protocol definitions
```

### Memory Infrastructure (Optional)

**Note:** Memory infrastructure (storage, embedding, retrieval, RAPTOR, indexing) has been migrated to the standalone `reasonkit-mem` crate. Enable the `memory` feature to use these modules:

```toml
[dependencies]
reasonkit-core = { version = "1.0", features = ["memory"] }
```

This automatically includes `reasonkit-mem` as a dependency and re-exports its modules for convenience. The RAG engine (with full LLM integration) remains in `reasonkit-core` and uses `reasonkit-mem` for storage/retrieval operations.

---

## QUICK START

```bash
# Build
cargo build --release

# Run with profile
../target/release/rk-core think --profile balanced "Your question"

# Compare raw vs enhanced
../target/release/rk-compare "Your question" --mock

# View metrics
../target/release/rk-core metrics report

# Run benchmarks
cargo run --release --bin gsm8k_eval
```

---

## DEVELOPMENT PRIORITIES

1. **Run benchmarks** - Prove (or disprove) value with data
2. **Honest documentation** - No claims without evidence
3. **Tracing/debugging** - This is the proven value
4. **Enterprise compliance** - Audit trails for regulated industries

---

## TASK MANAGEMENT (MANDATORY - CONS-007)

> **Axiom:** No work exists without task tracking. ALL AI agents MUST use the full task system.

### Taskwarrior Integration

**ALL AI agents MUST use Taskwarrior for task tracking.**

```bash
# Create task (MANDATORY format for RK-PROJECT)
task add project:rk-project.core.{component} "{description}" priority:{H|M|L} due:{date} +{tags}

# Examples:
task add project:rk-project.core.rag "Implement RAPTOR tree optimization" priority:H due:today +rust +performance
task add project:rk-project.core.thinktools "Add new ThinkTool module" priority:M due:friday +reasoning
task add project:rk-project.core.benchmarks "Run GSM8K evaluation" priority:M due:tomorrow +benchmark

# Start working (CRITICAL: Auto-starts timewarrior!)
task {id} start

# Stop working (pauses time tracking)
task {id} stop

# Complete task (stops timewarrior, records completion)
task {id} done

# Add annotations (progress notes, decisions, blockers)
task {id} annotate "Completed RAG implementation, 15% improvement in retrieval"
task {id} annotate "BLOCKED: Waiting for API key from vendor"
task {id} annotate "DECISION: Using Qdrant over Pinecone for cost reasons"

# View status
task project:rk-project.core list
task project:rk-project.core summary
timew summary :week
```

**Components:**

- `core.rag` → RAG pipeline
- `core.thinktools` → ThinkTool modules
- `core.benchmarks` → Evaluation and benchmarks
- `core.mcp` → MCP server
- `core.telemetry` → Telemetry and logging

**Full Documentation:** See [Taskwarrior docs](https://taskwarrior.org/docs/) for complete reference.

---

## MCP SERVERS, SKILLS & PLUGINS (MAXIMIZE)

### MCP Server Usage

**Agents MUST leverage MCP servers for all compatible operations.**

```yaml
MCP_SERVERS_PRIORITY:
  - sequential-thinking   # ALWAYS use for complex reasoning chains
  - filesystem            # File operations
  - github               # Repository operations
  - memory               # Persistent memory
  - puppeteer            # Web automation
  - fetch                # HTTP requests with caching

USAGE_PATTERN:
  1. Check if MCP server exists for operation
  2. If yes: USE IT (preferred over direct implementation)
  3. If no: Implement in Rust, consider creating MCP server
```

### Skills & Plugins

```yaml
SKILLS_MAXIMIZATION:
  - Use pdf skill for PDF operations
  - Use xlsx skill for spreadsheet operations
  - Use docx skill for document operations
  - Use frontend-design skill for UI work
  - Use mcp-builder skill for MCP server creation

PLUGIN_PRIORITY:
  - api-contract-sync for API validation
  - math for deterministic calculations
  - experienced-engineer agents for specialized tasks
```

### Extensions

```yaml
BROWSER_EXTENSIONS:
  - Use when web research needed
  - Prefer official provider extensions

IDE_EXTENSIONS:
  - Cursor: .cursorrules enforcement
  - VS Code: copilot-instructions.md
  - Windsurf: .windsurfrules
```

**Full Reference:** See [ORCHESTRATOR.md](../../ORCHESTRATOR.md#mcp-servers-skills--plugins-maximize) for complete MCP/Skills/Plugins documentation.

---

## CONSTRAINTS

| Constraint         | Details                                         |
| ------------------ | ----------------------------------------------- |
| Evidence required  | All improvement claims must have benchmark data |
| Honest positioning | "Structured prompts" not "AI enhancement"       |
| Traceability       | Every execution fully logged                    |
| Reproducibility    | Benchmarks must be reproducible                 |

---

_reasonkit-core v1.1.0 | Structured Prompt Engineering Framework | Apache 2.0_
_"See How Your AI Thinks"_
