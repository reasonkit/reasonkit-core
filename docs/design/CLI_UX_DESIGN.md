# CLI UX DESIGN SPECIFICATION

> ReasonKit Core Command Line Interface - User Experience Guidelines
> Version: 1.0.0 | Status: Design Specification
> Author: UX Design Agent

---

## 1. EXECUTIVE SUMMARY

This document provides comprehensive UX/UI design recommendations for the `rk-core` CLI tool. The focus is on creating an intuitive, accessible, and visually appealing terminal experience that helps developers understand and trust AI reasoning processes.

### Design Goals

| Goal | Description | Priority |
|------|-------------|----------|
| **Clarity** | Users always know which step they're in and what's happening | Critical |
| **Scannability** | Key insights are immediately visible without reading everything | High |
| **Accessibility** | Works without colors; screen reader friendly | High |
| **Copy-Paste Friendly** | Output can be easily shared and documented | Medium |
| **Delight** | Professional polish that inspires confidence | Medium |

---

## 2. OUTPUT FORMATTING RECOMMENDATIONS

### 2.1 Visual Hierarchy System

The output should follow a clear 4-level hierarchy:

```
LEVEL 1: Section Headers (Most Important)
-----------------------------------------
Double-line borders, tool branding, bold text
Example: Tool name, main result status

LEVEL 2: Subsection Headers
-----------------------------------------
Single-line borders, section titles
Example: EXECUTION METRICS, OUTPUT DATA

LEVEL 3: Key-Value Data
-----------------------------------------
Indented, labeled pairs with visual indicators
Example: Confidence: 87.3% [HIGH]

LEVEL 4: Content/Detail
-----------------------------------------
Indented content, wrapped text
Example: Actual analysis output, lists
```

### 2.2 Recommended Output Structure

```
[1] BRANDED HEADER (Tool identity + tagline)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[2] QUICK STATUS BAR (At-a-glance summary)
    Status: SUCCESS | Confidence: 87% HIGH | Duration: 2.3s

[3] PROGRESS INDICATOR (For long operations - see Section 3)
    Step 2/5: LaserLogic analyzing...

[4] EXECUTION TRACE (Collapsible/Optional)
    [01] GigaThink      OK  85%  1.2s
    [02] LaserLogic     OK  89%  0.8s
         |
    [05] ProofGuard     OK  91%  0.3s

[5] OUTPUT DATA (The actual results)
    ## Key Findings
    1. First insight here
    2. Second insight here

[6] FOOTER (Metadata + branding)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ReasonKit v0.1.0 | https://reasonkit.sh
```

### 2.3 Spacing and Layout Rules

| Element | Spacing |
|---------|---------|
| Between major sections | 1 blank line |
| Between subsections | No blank line |
| Within content | Natural paragraph breaks |
| Indentation | 2 spaces per level |
| Maximum line width | 76 characters (fits 80-col terminals) |

---

## 3. PROGRESS INDICATORS FOR LONG-RUNNING ANALYSIS

### 3.1 Progress Indicator Patterns

#### Pattern A: Step Progress (Recommended for Profile Execution)

```
Executing: balanced profile (4 steps)

  [1/4] GigaThink     Generating perspectives...
        ████████████████████████░░░░░░░░░░  75%

  Estimated: ~45s remaining
```

#### Pattern B: Spinner with Status (For Single Operations)

```
  [*] Analyzing claim with LaserLogic...
      ⣾ Checking logical validity

  (Alternative spinners: ⣾⣽⣻⢿⡿⣟⣯⣷ or |/-\)
```

#### Pattern C: Time-Based Progress (When Duration is Predictable)

```
  BedRock: First principles decomposition
  [====================          ] 67%  ETA: 12s
```

### 3.2 Progress Output Rules

1. **Write to stderr** - Progress indicators should write to stderr so they don't pollute stdout when piping to JSON parsers
2. **Clear on completion** - Use carriage return (`\r`) to overwrite progress on the same line
3. **Show elapsed time** - For operations over 5 seconds, show elapsed time
4. **Estimate remaining** - If possible, show estimated time remaining

### 3.3 Multi-Step Progress (Profile Chains)

For profile execution with multiple ThinkTools:

```
Executing profile: paranoid (6 steps)

  [01] GigaThink       DONE     1.2s  85%
  [02] LaserLogic      DONE     0.8s  89%
  [03] BedRock         RUNNING  ...   --
       ├─ Identifying axioms
       └─ Decomposing assumptions
  [04] ProofGuard      PENDING  --    --
  [05] BrutalHonesty   PENDING  --    --
  [06] ProofGuard      PENDING  --    --

  Progress: 2/6 complete | Elapsed: 2.0s | ETA: ~4.0s
```

### 3.4 Implementation Guidance (Rust)

```rust
// Write progress to stderr (doesn't interfere with --json)
use std::io::{Write, stderr};

fn show_progress(step: usize, total: usize, name: &str, status: &str) {
    eprint!("\r  [{:02}/{}] {:<15} {}", step, total, name, status);
    stderr().flush().unwrap();
}

// Clear progress line when done
fn clear_progress() {
    eprint!("\r{}\r", " ".repeat(60));
}
```

---

## 4. COLOR SCHEME FOR THINKTOOL OUTPUTS

### 4.1 Base Color Palette

The color scheme should reflect the ReasonKit brand while providing semantic meaning:

| Color | ANSI Code | Hex | Usage |
|-------|-----------|-----|-------|
| **Cyan (Primary)** | `\x1b[36m` | #00d2ff | Primary brand, borders |
| **Green (Success)** | `\x1b[32m` | #00ff9d | Success states, high confidence |
| **Yellow (Warning)** | `\x1b[33m` | #ffcc00 | Warnings, moderate confidence |
| **Red (Error)** | `\x1b[31m` | #ff4d00 | Errors, low confidence |
| **White (Text)** | `\x1b[97m` | #f8fbff | Primary text |
| **Gray (Dim)** | `\x1b[90m` | #8b949e | Secondary text, metadata |
| **Magenta (Accent)** | `\x1b[35m` | #bd34fe | Accents, highlights |

### 4.2 ThinkTool-Specific Themes

Each ThinkTool should have a distinct visual identity:

| Tool | Primary | Secondary | Icon | Rationale |
|------|---------|-----------|------|-----------|
| **GigaThink** | Gold (#FFD700) | Purple | 💡 | Brilliant ideas, creativity |
| **LaserLogic** | Green (#00FF00) | White | ⚡ | Precision, clarity |
| **BedRock** | Amber (#FFBF00) | Gray | 🪨 | Solid foundation |
| **ProofGuard** | White | Blue | 🛡️ | Authority, trust |
| **BrutalHonesty** | Red (#FF0000) | Yellow | 🔥 | Warning, intensity |
| **PowerCombo** | Rainbow gradient | - | 🌈 | All tools combined |

### 4.3 Semantic Colors

| Semantic Use | Color | ANSI | Example |
|--------------|-------|------|---------|
| Success/Completed | Green | `\x1b[32m` | "OK Status" |
| In Progress | Cyan | `\x1b[36m` | "RUNNING..." |
| Pending/Waiting | Gray | `\x1b[90m` | "PENDING" |
| Warning/Moderate | Yellow | `\x1b[33m` | "60% confidence" |
| Error/Failed | Red | `\x1b[31m` | "FAILED" |
| High Value | Bold Green | `\x1b[1;32m` | "95% confidence" |

### 4.4 Confidence Color Gradient

```
95-100%  VERY HIGH   ████████████████████  Bold Green  \x1b[1;32m
80-94%   HIGH        ████████████████░░░░  Green       \x1b[32m
60-79%   MODERATE    ████████████░░░░░░░░  Yellow      \x1b[33m
40-59%   LOW         ████████░░░░░░░░░░░░  Yellow      \x1b[33m
0-39%    VERY LOW    ████░░░░░░░░░░░░░░░░  Red         \x1b[31m
```

### 4.5 Color Application Examples

```
# Tool Header
\x1b[38;5;220m💡 \x1b[1;3;38;5;220mGigaThink\x1b[0m \x1b[38;5;220m·\x1b[0m \x1b[3;38;5;135m10+ Perspectives · Brilliant Ideas\x1b[0m

# Status Line
Status: \x1b[32m✓ SUCCESS\x1b[0m | Confidence: \x1b[1;32m92%\x1b[0m HIGH | Duration: 2.3s

# Progress Bar
Confidence: 87.3% [\x1b[32m████████████████\x1b[0m\x1b[90m░░░░\x1b[0m]

# Error State
\x1b[31m✗ FAILED\x1b[0m: API rate limit exceeded
```

---

## 5. INTERACTIVE MODE DESIGN (REPL-LIKE EXPERIENCE)

### 5.1 REPL Overview

An interactive REPL mode would allow users to have a conversation with ReasonKit without restarting the CLI:

```bash
$ rk-core interactive
# or
$ rk-core repl
# or
$ rk-core shell
```

### 5.2 REPL Interface Design

```
╭─────────────────────────────────────────────────────────────────╮
│  🧠 ReasonKit Interactive Shell v0.1.0                         │
│  Type 'help' for commands, 'exit' to quit                      │
╰─────────────────────────────────────────────────────────────────╯

rk> think "What is chain of thought prompting?"
[GigaThink] Analyzing...

💡 GigaThink · 10+ Perspectives
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Perspectives Generated:
1. Historical context: CoT emerged from...
2. Technical mechanism: The model is...
3. ...

Confidence: 87% | Duration: 1.2s

rk> /profile paranoid
Profile set to: paranoid (6 steps, 95% target)

rk> /model claude-sonnet-4
Model changed to: claude-sonnet-4

rk> Why does it improve reasoning?
[Executing paranoid profile]
...

rk> /history
1. "What is chain of thought prompting?"  (gigathink, 87%)
2. "Why does it improve reasoning?"       (paranoid, 91%)

rk> /export 2 --format markdown > analysis.md
Exported to analysis.md

rk> exit
Goodbye! Session saved to ~/.reasonkit/sessions/2025-12-28.json
```

### 5.3 REPL Commands

| Command | Shortcut | Description |
|---------|----------|-------------|
| `think <query>` | Default | Execute with current settings |
| `/profile <name>` | `/p` | Set reasoning profile |
| `/protocol <name>` | `/t` | Set single protocol |
| `/model <name>` | `/m` | Change LLM model |
| `/provider <name>` | `/pr` | Change LLM provider |
| `/temperature <n>` | `/temp` | Set temperature (0.0-2.0) |
| `/context <text>` | `/c` | Add context for next query |
| `/history` | `/h` | Show query history |
| `/export <n>` | `/e` | Export result n to file |
| `/last` | `/l` | Show last result |
| `/trace <n>` | `/tr` | View trace for result n |
| `/clear` | `/cls` | Clear screen |
| `/help` | `/?` | Show help |
| `exit` | `quit` | Exit REPL |

### 5.4 REPL Features

#### 5.4.1 Command Completion

```
rk> /pro<TAB>
/profile   /protocol   /provider

rk> /profile p<TAB>
paranoid   powercombo

rk> think "<TAB>
[Recent queries shown]
"What is chain of thought prompting?"
"Why does it improve reasoning?"
```

#### 5.4.2 Multi-line Input

```
rk> think """
... Analyze this code for security issues:
...
... fn process_input(data: &str) {
...     let query = format!("SELECT * FROM users WHERE name = '{}'", data);
...     // ...
... }
... """
```

#### 5.4.3 Result References

```
rk> think "What is X?"
[Result #1 stored]

rk> think "Explain more about {1.perspectives[2]}"
# Uses perspective #2 from result #1 as context
```

#### 5.4.4 Session Persistence

```
# Sessions auto-save to:
~/.reasonkit/sessions/
├── 2025-12-28_001.json
├── 2025-12-28_002.json
└── latest.json → 2025-12-28_002.json

# Resume session:
rk> /load ~/.reasonkit/sessions/2025-12-28_001.json
Session loaded: 5 queries, profile: balanced
```

### 5.5 REPL Visual Design

#### Prompt Styling

```
# Default prompt
rk>

# With active profile
rk [paranoid]>

# With context loaded
rk [paranoid] (ctx)>

# Multiline continuation
...>
```

#### Output Formatting in REPL

In REPL mode, use a more compact output format:

```
rk> think "Is Rust better than Go?"

💡 GigaThink Analysis
━━━━━━━━━━━━━━━━━━━━━

📊 Summary
├─ Confidence: 82% [HIGH]
├─ Duration: 1.4s
└─ Tokens: 847

📝 Key Points
1. Rust excels in memory safety without GC
2. Go offers faster compilation and simpler learning curve
3. Choice depends on use case requirements

💭 Verdict
Both languages have distinct strengths. Rust for systems
programming where safety is critical; Go for services
where development speed matters.

[Full result stored as #3. Use '/trace 3' for details]

rk>
```

---

## 6. ERROR MESSAGE UX GUIDELINES

### 6.1 Error Message Structure

Every error should follow this structure:

```
┌─ ERROR TYPE ──────────────────────────────────────────────────────┐
│                                                                    │
│  ✗ WHAT WENT WRONG (Clear, non-technical summary)                │
│                                                                    │
│  Details:                                                          │
│    Technical details for debugging                                 │
│                                                                    │
│  How to Fix:                                                       │
│    1. First suggested action                                       │
│    2. Second suggested action                                      │
│                                                                    │
│  More Info:                                                        │
│    https://docs.reasonkit.sh/errors/E001                          │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 6.2 Error Categories and Examples

#### 6.2.1 Configuration Errors (E1xx)

```
┌─ CONFIGURATION ERROR ─────────────────────────────────────────────┐
│                                                                    │
│  ✗ Missing API Key                                                │
│                                                                    │
│  The ANTHROPIC_API_KEY environment variable is not set.           │
│                                                                    │
│  How to Fix:                                                       │
│    1. Get an API key from https://console.anthropic.com           │
│    2. Set it in your environment:                                  │
│       export ANTHROPIC_API_KEY="sk-ant-..."                       │
│                                                                    │
│    Or use a different provider:                                    │
│       rk-core think "query" --provider openai                      │
│                                                                    │
│  More Info: https://docs.reasonkit.sh/errors/E101                 │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

#### 6.2.2 Network Errors (E2xx)

```
┌─ NETWORK ERROR ───────────────────────────────────────────────────┐
│                                                                    │
│  ✗ Connection Failed                                              │
│                                                                    │
│  Could not connect to the Anthropic API.                          │
│                                                                    │
│  Details:                                                          │
│    Error: Connection timed out after 30s                          │
│    Endpoint: https://api.anthropic.com/v1/messages                │
│                                                                    │
│  How to Fix:                                                       │
│    1. Check your internet connection                              │
│    2. Verify the API is available: https://status.anthropic.com   │
│    3. Try again with: rk-core think "query" --retry 3             │
│                                                                    │
│  More Info: https://docs.reasonkit.sh/errors/E201                 │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

#### 6.2.3 Input Validation Errors (E3xx)

```
┌─ INPUT ERROR ─────────────────────────────────────────────────────┐
│                                                                    │
│  ✗ Invalid Budget Format                                          │
│                                                                    │
│  The budget "50" is not in a valid format.                        │
│                                                                    │
│  Valid formats:                                                    │
│    Time:   30s, 2m, 1h                                            │
│    Tokens: 1000t, 5000tokens                                       │
│    Cost:   $0.50, $5.00                                           │
│                                                                    │
│  Example:                                                          │
│    rk-core think "query" --budget 30s                             │
│    rk-core think "query" --budget $0.50                           │
│                                                                    │
│  More Info: https://docs.reasonkit.sh/errors/E301                 │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

#### 6.2.4 Execution Errors (E4xx)

```
┌─ EXECUTION ERROR ─────────────────────────────────────────────────┐
│                                                                    │
│  ✗ Protocol Execution Failed                                      │
│                                                                    │
│  The LaserLogic protocol failed at step "check_validity".         │
│                                                                    │
│  Details:                                                          │
│    Step: 2 of 3                                                    │
│    Error: LLM response did not match expected format              │
│    Trace ID: a1b2c3d4                                             │
│                                                                    │
│  How to Fix:                                                       │
│    1. Try again (transient LLM errors are common)                 │
│    2. Simplify your query                                         │
│    3. Lower the temperature: --temperature 0.5                    │
│    4. View trace: rk-core trace view a1b2c3d4                     │
│                                                                    │
│  More Info: https://docs.reasonkit.sh/errors/E401                 │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 6.3 Error Message Rules

| Rule | Description |
|------|-------------|
| **No jargon** | Use plain language, not internal error codes alone |
| **Actionable** | Always provide at least one concrete action |
| **Context-aware** | Include relevant details (what was attempted) |
| **Consistent** | Use the same structure for all errors |
| **Non-blaming** | Avoid "you did X wrong"; say "X needs to be Y" |
| **Helpful links** | Point to documentation when relevant |

### 6.4 Warning Messages

For non-fatal issues, use warnings:

```
⚠ WARNING: RAPTOR tree is outdated
   Last built: 7 days ago (12,450 chunks)
   Current documents: 13,210 chunks
   Run 'rk-core index build' to rebuild

   (Continuing with existing index...)
```

### 6.5 Graceful Degradation Messages

```
ℹ INFO: Web search unavailable
   No search API key configured.
   Proceeding with knowledge base only.

   To enable web search:
   export TAVILY_API_KEY="..." or
   export SERPER_API_KEY="..."
```

---

## 7. ACCESSIBILITY GUIDELINES

### 7.1 No-Color Mode

All output must be readable without colors:

```bash
# Force no-color mode
rk-core --color never think "query"

# Or via environment
NO_COLOR=1 rk-core think "query"
TERM=dumb rk-core think "query"
```

No-color output should use:
- ASCII borders instead of box-drawing characters
- Text labels instead of color-coded states
- Explicit status text: "[OK]", "[FAILED]", "[WARN]"

```
# With colors
┌──────────────────────┐
│ ✓ SUCCESS           │
│ Confidence: 87% HIGH │
└──────────────────────┘

# Without colors (--color never)
+----------------------+
| [OK] SUCCESS         |
| Confidence: 87% HIGH |
+----------------------+
```

### 7.2 Screen Reader Compatibility

1. **Avoid decorative characters** in critical information
2. **Use semantic text** alongside icons: "SUCCESS" not just "✓"
3. **Linear structure** - information flows top to bottom
4. **Proper headings** - use consistent section labels

### 7.3 Contrast Ratios

For terminal emulators that support it:
- Foreground text should have 4.5:1 contrast ratio minimum
- Important information (errors, warnings) should be 7:1

### 7.4 Keyboard Navigation in REPL

| Key | Action |
|-----|--------|
| Up/Down | Navigate history |
| Ctrl+R | Reverse search history |
| Ctrl+C | Cancel current operation |
| Ctrl+D | Exit REPL |
| Tab | Auto-complete |
| Ctrl+L | Clear screen |

---

## 8. COPY-PASTE FRIENDLY OUTPUT

### 8.1 Machine-Readable Formats

Always provide clean JSON output:

```bash
rk-core think "query" --format json
```

```json
{
  "protocol_id": "gigathink",
  "success": true,
  "confidence": 0.87,
  "duration_ms": 1234,
  "data": {
    "perspectives": [...],
    "synthesis": "..."
  }
}
```

### 8.2 Markdown Export

```bash
rk-core think "query" --format markdown
# or
rk-core think "query" -o analysis.md
```

Output structure:
```markdown
# GigaThink Analysis

**Query:** What is chain of thought prompting?
**Confidence:** 87% (HIGH)
**Duration:** 1.23s

---

## Perspectives

1. **Historical Context**: CoT prompting emerged from...
2. **Technical Mechanism**: The model generates...

## Synthesis

Chain of thought prompting is a technique that...

---

*Generated by ReasonKit v0.1.0 on 2025-12-28*
```

### 8.3 Plain Text (Default)

The default text output should be:
- Clean enough to paste into documents
- Not depend on terminal-specific features
- Include all essential information without decoration

### 8.4 Quiet Mode

For scripting, provide minimal output:

```bash
rk-core think "query" --quiet
# Returns only: confidence score and verdict

0.87
Chain of thought prompting improves reasoning by...
```

---

## 9. IMPLEMENTATION CHECKLIST

### 9.1 Priority 1 (Critical)

- [ ] Implement stderr progress indicators
- [ ] Add `--color never` support
- [ ] Create consistent error message format
- [ ] Add confidence color gradient

### 9.2 Priority 2 (High)

- [ ] Implement compact REPL mode
- [ ] Add Markdown export format
- [ ] Create ThinkTool-specific themes
- [ ] Implement progress bars for long operations

### 9.3 Priority 3 (Medium)

- [ ] Add command completion in REPL
- [ ] Implement session persistence
- [ ] Add result references in REPL
- [ ] Create quiet mode

### 9.4 Priority 4 (Nice to Have)

- [ ] Add multi-line input support
- [ ] Implement history search
- [ ] Add export to file shortcuts
- [ ] Create shell integration scripts

---

## 10. EXAMPLE OUTPUTS

### 10.1 Successful Execution (Full)

```
💡 GigaThink · 10+ Perspectives · Brilliant Ideas
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────────────┐
│  EXECUTION METRICS                                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Status:        ✓ SUCCESS                                          │
│  Confidence:    87.3% (HIGH) [█████████████████░░░]                │
│  Duration:      2.34s (2340ms)                                     │
│  Tokens:        in: 245  out: 602  total: 847                      │
│  Cost:          $0.0127 USD                                        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  OUTPUT DATA                                                        │
├─────────────────────────────────────────────────────────────────────┤
│  perspectives:                                                      │
│    1. Historical perspective: Chain of thought prompting...        │
│    2. Technical mechanism: By generating intermediate...           │
│    3. Empirical evidence: Studies show 15-30% improvement...       │
│    4. Limitations: Not effective for all task types...             │
│                                                                     │
│  synthesis:                                                         │
│    Chain of thought prompting is a technique that improves         │
│    LLM reasoning by generating explicit intermediate steps...      │
└─────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ReasonKit · Turn Prompts into Protocols · https://reasonkit.sh
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 10.2 Profile Execution with Progress

```
Executing profile: paranoid (6 steps)

  [01] GigaThink       ✓ DONE     1.2s  85%
  [02] LaserLogic      ✓ DONE     0.8s  89%
  [03] BedRock         ● RUNNING  0.4s  --
       └─ Identifying first principles...
  [04] ProofGuard      ○ PENDING  --    --
  [05] BrutalHonesty   ○ PENDING  --    --
  [06] ProofGuard      ○ PENDING  --    --

  ████████████░░░░░░░░░░░░  33% | Elapsed: 2.4s | ETA: ~4.8s
```

### 10.3 Error State

```
┌─ EXECUTION ERROR ─────────────────────────────────────────────────┐
│                                                                    │
│  ✗ Rate Limit Exceeded                                            │
│                                                                    │
│  The Anthropic API rate limit has been reached.                   │
│                                                                    │
│  Details:                                                          │
│    Requests this minute: 60/60                                    │
│    Reset in: 45 seconds                                           │
│                                                                    │
│  How to Fix:                                                       │
│    1. Wait 45 seconds and try again                               │
│    2. Use a different provider: --provider openrouter             │
│    3. Reduce request frequency with --budget 1m                   │
│                                                                    │
│  More Info: https://docs.reasonkit.sh/errors/E429                 │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## VERSION HISTORY

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-28 | Initial UX design specification |

---

*"Great CLIs are discovered through use, refined through feedback."*
*- ReasonKit UX Design*
