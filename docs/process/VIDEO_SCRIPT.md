# ReasonKit 3-Minute Demo Video Script

> **Duration:** 3:00  
> **Format:** Screen recording with narration  
> **Goal:** Show ReasonKit value in action, not explain concepts

---

## Video Outline

| Time      | Section      | What's on screen      | Narration         |
| --------- | ------------ | --------------------- | ----------------- |
| 0:00-0:15 | Hook         | Terminal with problem | The problem       |
| 0:15-0:45 | Install      | One-liner install     | 30-second install |
| 0:45-1:30 | First Run    | Basic command         | See the magic     |
| 1:30-2:15 | Profiles     | Compare quick vs deep | Choose your rigor |
| 2:15-2:45 | Real Example | Actual use case       | This is useful    |
| 2:45-3:00 | CTA          | Website/docs          | Get started       |

---

## Detailed Script

### Scene 1: The Hook (0:00-0:15)

**Screen:** Clean terminal, dark theme

**Type on screen:**

```
$ ask-llm "Should I use microservices?"
```

**Show:** Wall of unstructured text response

**Narration:**

> "You ask an LLM for advice. You get... this. Two thousand words. No structure. No confidence score. No idea if it's right."

**Cut to:** ReasonKit logo flash (0.5s)

---

### Scene 2: Installation (0:15-0:45)

**Screen:** Clean terminal

**Narration:**

> "There's a better way. ReasonKit. One line to install."

**Type on screen:**

```bash
curl -fsSL https://reasonkit.sh/install | bash
```

**Show:** Installation progress (speed up 4x)

**Narration (during install):**

> "ReasonKit is a Rust CLI that turns messy prompts into structured reasoning protocols. Install takes 30 seconds."

**Show:** Installation complete message

**Type on screen:**

```bash
rk --version
# reasonkit-core 0.1.0
```

**Narration:**

> "Done. Let's use it."

---

### Scene 3: First Run (0:45-1:30)

**Screen:** Clean terminal

**Narration:**

> "Same question. Different approach."

**Type on screen:**

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
rk think "Should I use microservices?" --profile quick
```

**Show:** Output appearing in real-time (can speed up slightly)

```
Protocol: quick (GigaThink -> LaserLogic)
Model: claude-sonnet-4

[GigaThink] 10 PERSPECTIVES GENERATED
  1. TEAM SIZE: Microservices need 20+ engineers
  2. DEPLOYMENT: Monolith = 1 deploy
  3. DEBUGGING: Distributed tracing is hard
  ...

[LaserLogic] HIDDEN ASSUMPTIONS DETECTED
  ! Assuming scale is your problem
  ! Assuming team has DevOps maturity

VERDICT: Start with monolith | Confidence: 78% | Time: 1.8s
```

**Narration:**

> "Ten perspectives. Hidden assumptions detected. A verdict with confidence score. In under 2 seconds."

**Pause on output for 2 seconds**

**Narration:**

> "That's structured reasoning. Not a wall of text. Organized thinking you can audit."

---

### Scene 4: Profiles (1:30-2:15)

**Screen:** Split view showing profile comparison

**Narration:**

> "ReasonKit has four profiles. Match the rigor to the stakes."

**Type on screen (left side):**

```bash
rk think "Deploy now?" --profile quick
# Time: 30s, Confidence: 70%
```

**Type on screen (right side):**

```bash
rk think "Deploy now?" --profile paranoid
# Time: 10min, Confidence: 95%
```

**Narration:**

> "Quick for Slack decisions. Paranoid for production deploys. You choose."

**Show:** Quick table of profiles

```
--profile quick     Daily decisions      30s
--profile balanced  Important choices    2min
--profile deep      Major architecture   5min
--profile paranoid  Production releases  10min
```

**Narration:**

> "Each profile runs different combinations of ThinkTools - specialized cognitive modules for specific types of analysis."

---

### Scene 5: Real Example (2:15-2:45)

**Screen:** Clean terminal

**Narration:**

> "Here's a real use case. Code review automation."

**Type on screen:**

```bash
rk think "Review this PR for security issues:
  - Adds user input to SQL query
  - Uses eval() on user data
  - Logs passwords in debug mode" --profile balanced --format json
```

**Show:** JSON output

```json
{
  "verdict": "reject",
  "confidence": 0.95,
  "critical_issues": [
    "SQL injection vulnerability",
    "Arbitrary code execution via eval()",
    "Credential exposure in logs"
  ],
  "recommendation": "Block merge, require security review"
}
```

**Narration:**

> "JSON output for CI/CD integration. High confidence. Actionable. You can build this into your pipeline today."

---

### Scene 6: Call to Action (2:45-3:00)

**Screen:** ReasonKit website hero section or logo

**Narration:**

> "ReasonKit. Turn prompts into protocols."

**Show:**

```
reasonkit.sh

cargo install reasonkit-core
curl -fsSL https://reasonkit.sh/install | bash
```

**Narration:**

> "Open source. Rust-native. Works with any LLM. Get started at reasonkit.sh."

**End card:** ReasonKit logo + "Turn Prompts into Protocols"

---

## Production Notes

### Visual Style

- **Terminal:** Dark theme (Dracula or similar)
- **Font:** JetBrains Mono, 18pt
- **Colors:** Cyan for prompts, Green for success, White for output
- **Transitions:** Cut (no fades, keep it fast)

### Audio

- **Music:** Subtle electronic, upbeat but not distracting
- **Voice:** Professional, confident, fast-paced (not rushed)
- **Mix:** Voice 80%, Music 20%

### Pacing

- No pauses longer than 1.5 seconds
- Speed up installation and long outputs (4x)
- Keep energy high

### Key Frames to Capture

1. The messy LLM output (before)
2. The structured ReasonKit output (after)
3. The confidence score
4. The JSON output for automation
5. The website/installation command

### Recording Tips

- Pre-record all commands, paste them in
- Use `asciinema` or screen recording
- Add subtle cursor animations
- Ensure API responses are fast (or pre-record)

---

## Alternate Versions

### 60-Second Version

- Hook (5s)
- Install (10s)
- First run (25s)
- CTA (20s)

### 30-Second Version

- Problem + Solution in one (15s)
- One command demo (10s)
- CTA (5s)

---

## Thumbnail Ideas

1. Terminal with "78% Confidence" highlighted
2. Before/after split (messy text vs structured output)
3. "AI Reasoning Engine" with terminal background
4. The 5 ThinkTool icons in a chain

---

**File:** `docs/VIDEO_SCRIPT.md`  
**Last Updated:** 2026-01-01
