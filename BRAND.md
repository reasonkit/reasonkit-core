# ReasonKit Brand Identity System

> **Classification:** CANONICAL SOURCE OF TRUTH
> **Enforcement:** HARD - All projects MUST comply
> **Tagline:** "Turn Prompts into Protocols"
> **Philosophy:** "Designed, Not Dreamed" | "High Logic > Lost Hopes"

---

## Core Identity

```
██████╗ ███████╗ █████╗ ███████╗ ██████╗ ███╗   ██╗██╗  ██╗██╗████████╗
██╔══██╗██╔════╝██╔══██╗██╔════╝██╔═══██╗████╗  ██║██║ ██╔╝██║╚══██╔══╝
██████╔╝█████╗  ███████║███████╗██║   ██║██╔██╗ ██║█████╔╝ ██║   ██║
██╔══██╗██╔══╝  ██╔══██║╚════██║██║   ██║██║╚██╗██║██╔═██╗ ██║   ██║
██║  ██║███████╗██║  ██║███████║╚██████╔╝██║ ╚████║██║  ██╗██║   ██║
╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝   ██║
```

**Mission:** Make AI reasoning structured, auditable, and reliable.
**Website:** https://reasonkit.sh

---

## Color System (NON-NEGOTIABLE)

> **Source of Truth:** `reasonkit-site/main.css` CSS variables
> **Last Synchronized:** 2025-12-30

### Primary Palette

| Name                   | Hex       | RGB                 | CSS Variable | Usage                                   |
| ---------------------- | --------- | ------------------- | ------------ | --------------------------------------- |
| **Cyan** (Primary)     | `#06b6d4` | `rgb(6, 182, 212)`  | `--cyan`     | Primary accent, CTAs, links, highlights |
| **Green** (Success)    | `#10b981` | `rgb(16, 185, 129)` | `--green`    | Success states, confirmations, positive |
| **Purple** (Secondary) | `#a855f7` | `rgb(168, 85, 247)` | `--purple`   | Secondary accent, gradients             |
| **Pink** (Tertiary)    | `#ec4899` | `rgb(236, 72, 153)` | `--pink`     | Gradient endpoints, emphasis            |
| **Orange** (Alert)     | `#f97316` | `rgb(249, 115, 22)` | `--orange`   | Warnings, alerts, attention             |
| **Yellow** (Highlight) | `#fbbf24` | `rgb(251, 191, 36)` | `--yellow`   | Highlights, featured elements           |

### Background & Text

| Name               | Hex       | RGB                  | CSS Variable       | Usage               |
| ------------------ | --------- | -------------------- | ------------------ | ------------------- |
| **Void Black**     | `#030508` | `rgb(3, 5, 8)`       | `--bg-void`        | Primary background  |
| **Deep Black**     | `#0a0d14` | `rgb(10, 13, 20)`    | `--bg-deep`        | Card backgrounds    |
| **Surface**        | `#111827` | `rgb(17, 24, 39)`    | `--bg-surface`     | Elevated surfaces   |
| **Elevated**       | `#1f2937` | `rgb(31, 41, 55)`    | `--bg-elevated`    | Hover states        |
| **White**          | `#f9fafb` | `rgb(249, 250, 251)` | `--text-primary`   | Primary text        |
| **Text Secondary** | `#9ca3af` | `rgb(156, 163, 175)` | `--text-secondary` | Secondary text      |
| **Text Muted**     | `#6b7280` | `rgb(107, 114, 128)` | `--text-muted`     | Muted/disabled text |
| **Text Dim**       | `#4b5563` | `rgb(75, 85, 99)`    | `--text-dim`       | Very muted text     |

### Gradient Definitions

```css
/* Hero Gradient - Tree of Thoughts visualization */
--gradient-hero: linear-gradient(135deg, #06b6d4 0%, #a855f7 50%, #ec4899 100%);

/* Cyan-Purple Gradient */
--gradient-cyan-purple: linear-gradient(135deg, #06b6d4, #a855f7);

/* Purple-Pink Gradient */
--gradient-purple-pink: linear-gradient(135deg, #a855f7, #ec4899);

/* Subtle Glow - For backgrounds and ambient effects */
--gradient-glow: radial-gradient(
  ellipse at center,
  rgba(6, 182, 212, 0.15),
  transparent 70%
);

/* Glow Colors */
--cyan-glow: rgba(6, 182, 212, 0.3);
--purple-glow: rgba(168, 85, 247, 0.3);
--pink-glow: rgba(236, 72, 153, 0.3);
--green-glow: rgba(16, 185, 129, 0.3);
--orange-glow: rgba(249, 115, 22, 0.3);
```

---

## Typography System

> **Source of Truth:** `reasonkit-site/index.html` font imports and `main.css`
> **Last Synchronized:** 2025-12-30

### Font Stack

| Category      | Font             | Fallback                     | CSS                               | Usage                       |
| ------------- | ---------------- | ---------------------------- | --------------------------------- | --------------------------- |
| **Headlines** | Inter            | -apple-system, sans-serif    | `font-family: 'Inter'`            | H1, H2, hero text, body     |
| **Display**   | Playfair Display | Georgia, serif               | `font-family: 'Playfair Display'` | Special emphasis (optional) |
| **Code**      | JetBrains Mono   | SF Mono, Consolas, monospace | `font-family: 'JetBrains Mono'`   | Code, terminals, metrics    |

**Note:** The landing page uses Inter as the primary font for both headlines and body. IBM Plex fonts may be used in documentation and technical materials but Inter is the primary web font.

### Scale (Fluid Typography)

```css
/* Base: 16px */
--text-xs: 0.75rem; /* 12px */
--text-sm: 0.875rem; /* 14px */
--text-base: 1rem; /* 16px */
--text-lg: 1.125rem; /* 18px */
--text-xl: 1.25rem; /* 20px */
--text-2xl: 1.5rem; /* 24px */
--text-3xl: 1.875rem; /* 30px */
--text-4xl: 2.25rem; /* 36px */
--text-5xl: 3rem; /* 48px */
--text-6xl: 3.75rem; /* 60px */
--text-7xl: 4.5rem; /* 72px */
```

### Font Weights

| Weight   | Value | Usage                      |
| -------- | ----- | -------------------------- |
| Light    | 300   | Large display text         |
| Regular  | 400   | Body text                  |
| Medium   | 500   | Emphasis, subheadings      |
| Semibold | 600   | Buttons, labels            |
| Bold     | 700   | Headlines, strong emphasis |

---

## Visual Language

### Tree of Thoughts Motif

The **Tree of Thoughts** is ReasonKit's core visual metaphor:

- Represents hierarchical reasoning structures
- Branching nodes = decision points
- Glowing connections = reasoning pathways
- Root = foundational axioms (BedRock)
- Leaves = conclusions and outputs

**Application:**

- Hero backgrounds (subtle, 5-10% opacity)
- Documentation diagrams
- ThinkTool visualizations
- Marketing materials

### Ambient Orbs

Floating, blurred orbs create depth and movement:

- Cyan, purple, pink colors
- 15% opacity, 60px blur
- Slow floating animations (18-28s cycles)
- Positioned asymmetrically

### Glass Morphism

Cards and containers use subtle glass effects:

```css
background: rgba(10, 15, 26, 0.8);
backdrop-filter: blur(12px);
border: 1px solid rgba(255, 255, 255, 0.05);
```

---

## Component Patterns

### Buttons

```css
/* Primary Button */
.btn-primary {
  background: var(--cyan);
  color: var(--bg-deep);
  font-weight: 600;
  padding: 0.75rem 1.5rem;
  border-radius: 0.5rem;
  transition: all 0.2s ease;
}

/* Ghost Button */
.btn-ghost {
  background: transparent;
  border: 1px solid rgba(255, 255, 255, 0.1);
  color: var(--white);
}
```

### Cards

```css
.card {
  background: rgba(10, 15, 26, 0.6);
  border: 1px solid rgba(255, 255, 255, 0.05);
  border-radius: 1rem;
  padding: 1.5rem;
  backdrop-filter: blur(8px);
}
```

### Badges

```css
.badge {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: rgba(6, 182, 212, 0.08);
  border: 1px solid rgba(6, 182, 212, 0.2);
  border-radius: 100px;
  font-size: 0.8125rem;
  color: var(--cyan);
}
```

### Code Blocks

```css
.code-block {
  background: var(--bg-deep);
  border: 1px solid rgba(255, 255, 255, 0.05);
  border-radius: 0.75rem;
  font-family: "JetBrains Mono", monospace;
  padding: 1rem;
}
```

---

## Animation Guidelines

### Timing Functions

| Name    | Value                | Usage                 |
| ------- | -------------------- | --------------------- |
| Fast    | `0.15s ease`         | Micro-interactions    |
| Normal  | `0.3s ease`          | Standard transitions  |
| Slow    | `0.5s ease-out`      | Page transitions      |
| Ambient | `20-30s ease-in-out` | Background animations |

### Motion Principles

1. **Purposeful** - Every animation serves a function
2. **Subtle** - Never distracting or excessive
3. **Accessible** - Respect `prefers-reduced-motion`
4. **Consistent** - Same timing across similar elements

### Standard Animations

```css
@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes pulse-glow {
  0%,
  100% {
    box-shadow: 0 0 20px rgba(0, 210, 255, 0.3);
  }
  50% {
    box-shadow: 0 0 40px rgba(0, 210, 255, 0.5);
  }
}
```

---

## Voice & Tone

### Writing Principles

| Principle         | Description            | Example                          |
| ----------------- | ---------------------- | -------------------------------- |
| **Authoritative** | Confident expertise    | "ReasonKit validates reasoning." |
| **Technical**     | Precise, specific      | "5-module protocol chain"        |
| **Clear**         | No jargon, direct      | "Turn prompts into protocols"    |
| **Honest**        | No hype, substantiated | "Benchmarked at 95% accuracy"    |

### Do's and Don'ts

**DO:**

- Use active voice
- Be specific with numbers and results
- Explain benefits, not just features
- Use technical terms correctly

**DON'T:**

- Use buzzwords without substance
- Make unverified claims
- Use excessive exclamation marks
- Be vague or wishy-washy

### Taglines & Copy

```
Primary:       "Turn Prompts into Protocols"
Philosophy:    "Designed, Not Dreamed"
Positioning:   "Structure Beats Intelligence"
Technical:     "Auditable Reasoning by Default"
```

---

## Asset Requirements

### Image Generation Prompts

When generating images for ReasonKit, include:

```
REQUIRED ELEMENTS:
- Cyan (#06b6d4) as primary color (landing page standard)
- Void black (#030508) background
- Purple (#a855f7) and pink (#ec4899) accents
- Green (#10b981) for success/validation states
- Orange (#f97316) for alerts/warnings
- Yellow (#fbbf24) for highlights
- Tree/branching/neural network motifs
- Clean, futuristic, premium aesthetic
- NO text (add text programmatically via SVG/CSS)

STYLE KEYWORDS:
- 3D render, hyperrealistic
- Volumetric lighting, ambient glow
- Glass morphism (blur effects)
- Geometric precision
- Dark mode, high contrast
- Neural network / circuit patterns

GLOW EFFECTS:
- Use rgba with 0.3 opacity for glow colors
- Cyan glow: rgba(6, 182, 212, 0.3)
- Purple glow: rgba(168, 85, 247, 0.3)
- Pink glow: rgba(236, 72, 153, 0.3)
```

### Logo Usage

- Minimum clear space: 0.5x logo height
- Minimum size: 32px height
- Always use on dark backgrounds
- Never distort or recolor

### Social Media Sizes

| Platform     | Size      | Format |
| ------------ | --------- | ------ |
| OG Image     | 1200x630  | PNG    |
| Twitter Card | 1200x600  | PNG    |
| LinkedIn     | 1200x627  | PNG    |
| Square       | 1200x1200 | PNG    |

---

## Documentation Standards

### README Format

Every project README should include:

1. ASCII art banner (ReasonKit logo)
2. One-line description
3. Features list with checkmarks
4. Quick install command
5. Code example
6. Link to full docs

### Code Comments

```rust
// ============================================================
// ThinkTool: GigaThink - Expansive Creative Reasoning
// ============================================================
// Generates 10+ perspectives for comprehensive exploration
// Part of the ReasonKit 5-module protocol chain
// ============================================================
```

### Diagram Style

- Use Mermaid or ASCII art
- Cyan (#06b6d4) for primary paths
- Purple (#a855f7) for secondary
- Keep backgrounds transparent/dark

---

## Enforcement Checklist

### For Every Asset

- [ ] Uses correct color palette (no off-brand colors)
- [ ] Uses correct typography (IBM Plex / Inter)
- [ ] Follows glass morphism patterns
- [ ] Respects motion/animation guidelines
- [ ] Voice/tone is authoritative and clear
- [ ] No unverified claims
- [ ] Dark mode optimized

### For Generated Images

- [ ] Contains cyan/purple/pink gradient
- [ ] Deep black background
- [ ] No AI-generated text
- [ ] Tree/neural motif present
- [ ] Premium/futuristic aesthetic

### For Documentation

- [ ] ASCII banner included
- [ ] Consistent formatting
- [ ] Code examples use brand colors in syntax highlighting
- [ ] Diagrams follow color system

---

## File Locations

| Asset Type          | Location                          |
| ------------------- | --------------------------------- |
| Brand Config        | `rk-startup/branding/brand.yaml`  |
| Design Tokens       | `rk-startup/design/tokens.css`    |
| Launch Images       | `reasonkit-site/assets/launch/`   |
| Logo Assets         | `rk-startup/branding/logos/`      |
| Marketing Templates | `rk-startup/marketing/templates/` |
| This Document       | `/RK-PROJECT/BRAND_IDENTITY.md`   |

---

## Version History

| Version | Date       | Changes                                                                                                                                                                                                                                                                                                   |
| ------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.1.0   | 2025-12-30 | **MAJOR:** Synchronized with landing page (main.css) - Updated cyan from #06b6d4 to #06b6d4, green from #10b981 to #10b981, orange from #f97316 to #f97316, background from #030508 to #030508, added yellow #fbbf24. Updated fonts from IBM Plex to Inter/JetBrains Mono. Added CSS variable references. |
| 1.0.0   | 2025-12-30 | Initial canonical brand identity                                                                                                                                                                                                                                                                          |

---

**This document is the SINGLE SOURCE OF TRUTH for ReasonKit brand identity.**

All projects, assets, documentation, and marketing materials MUST comply with these standards.

Deviations require explicit approval and documentation.

---

_"Designed, Not Dreamed" | https://reasonkit.sh_
