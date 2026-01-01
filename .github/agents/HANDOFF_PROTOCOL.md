# AGENT HANDOFF PROTOCOL

## PURPOSE

To ensure seamless context transfer between specialized agents in the ReasonKit Swarm.

## THE HANDOFF TRIGGER

When an agent reaches the boundary of its domain, it MUST initiate a handoff.

### Trigger Examples

- **Architect** -> **Rust Engineer**: "Architecture defined. Proceed to implementation."
- **Rust Engineer** -> **QA Engineer**: "Implementation complete. Verify quality gates."
- **Researcher** -> **Architect**: "Research complete. Synthesize into roadmap."
- **Any** -> **Task Master**: "Work identified. Create tracking tasks."

## HANDOFF FORMAT

When handing off, the active agent must provide a **Context Block**:

```markdown
**HANDOFF TO:** [Target Agent Name]
**CONTEXT:** [1-sentence summary of current state]
**ARTIFACTS:**

- [File Path 1] (Created/Modified)
- [File Path 2] (Reference)
  **NEXT ACTION:** [Specific instruction for the next agent]
```

## AGENT CAPABILITIES MATRIX

| From Agent            | To Agent              | Typical Trigger                |
| :-------------------- | :-------------------- | :----------------------------- |
| **Architect**         | **Rust Engineer**     | Implementation of core logic   |
| **Architect**         | **Python Specialist** | Implementation of glue/scripts |
| **Rust Engineer**     | **QA Engineer**       | Pre-merge verification         |
| **Python Specialist** | **QA Engineer**       | Testing scripts/bindings       |
| **QA Engineer**       | **DevOps SRE**        | Release/Deployment             |
| **DevOps SRE**        | **Security Guardian** | Infrastructure Audit           |
| **Researcher**        | **Architect**         | Strategic planning             |

## INSTRUCTIONS FOR HUMANS

If you see a **HANDOFF TO** message:

1. Copy the **Context Block**.
2. Paste it into the chat.
3. Activate the requested agent (via `LOADER.md` or manual context switch).
