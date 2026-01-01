# ReasonKit Brand Playbook

> **Classification:** ULTIMATE MASTER SOURCE OF TRUTH
> **Enforcement:** HARD - ALL AI agents and projects MUST comply
> **Last Updated:** 2025-12-31
> **Supersedes:** All other BRAND\*.md files across RK-PROJECT

---

```
██████╗ ███████╗ █████╗ ███████╗ ██████╗ ███╗   ██╗██╗  ██╗██╗████████╗
██╔══██╗██╔════╝██╔══██╗██╔════╝██╔═══██╗████╗  ██║██║ ██╔╝██║╚══██╔══╝
██████╔╝█████╗  ███████║███████╗██║   ██║██╔██╗ ██║█████╔╝ ██║   ██║
██╔══██╗██╔══╝  ██╔══██║╚════██║██║   ██║██║╚██╗██║██╔═██╗ ██║   ██║
██║  ██║███████╗██║  ██║███████║╚██████╔╝██║ ╚████║██║  ██╗██║   ██║
╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝   ╚═╝

              BRAND PLAYBOOK | DESIGNED, NOT DREAMED
```

---

## Table of Contents

1. [Core Identity](#core-identity)
2. [Color System](#color-system)
3. [Typography](#typography)
4. [Logo System: The Luminous Polyhedron](#logo-system-the-luminous-polyhedron)
5. [Visual Language](#visual-language)
6. [Asset Directory Structure](#asset-directory-structure)
7. [Touchpoint-Specific Guidelines](#touchpoint-specific-guidelines)
8. [AI Agent Instructions](#ai-agent-instructions)
9. [Quick Reference Cards](#quick-reference-cards)

---

## Core Identity

| Attribute       | Value                                                 |
| --------------- | ----------------------------------------------------- |
| **Name**        | ReasonKit                                             |
| **Tagline**     | "Turn Prompts into Protocols"                         |
| **Philosophy**  | "Designed, Not Dreamed" / "High Logic > Lost Hopes"   |
| **Positioning** | "Structure Beats Intelligence"                        |
| **Technical**   | "Auditable Reasoning by Default"                      |
| **Website**     | <https://reasonkit.sh>                                  |
| **Mission**     | Make AI reasoning structured, auditable, and reliable |

### Product Ecosystem

| Project            | License     | Purpose                         | Status |
| ------------------ | ----------- | ------------------------------- | ------ |
| **ReasonKit-core** | Apache 2.0  | Structured reasoning engine     | OSS    |
| **ReasonKit-mem**  | Apache 2.0  | Memory/vector layer (optional)  | OSS    |
| **ReasonKit-web**  | Apache 2.0  | Web sensing MCP sidecar         | OSS    |
| **ReasonKit-pro**  | Proprietary | Advanced ThinkTools, enterprise | Paid   |

---

## Color System

> **Source of Truth:** `reasonkit-site/main.css` CSS variables
> **Enforcement:** NON-NEGOTIABLE - Use ONLY these colors

### Primary Palette

| Name                   | Hex       | RGB                 | CSS Variable | Usage                                   |
| ---------------------- | --------- | ------------------- | ------------ | --------------------------------------- |
| **Cyan** (Primary)     | `#06b6d4` | `rgb(6, 182, 212)`  | `--cyan`     | CTAs, links, highlights, primary accent |
| **Green** (Success)    | `#10b981` | `rgb(16, 185, 129)` | `--green`    | Success states, confirmations           |
| **Purple** (Secondary) | `#a855f7` | `rgb(168, 85, 247)` | `--purple`   | Secondary accent, gradients             |
| **Pink** (Tertiary)    | `#ec4899` | `rgb(236, 72, 153)` | `--pink`     | Gradient endpoints, emphasis            |
| **Orange** (Alert)     | `#f97316` | `rgb(249, 115, 22)` | `--orange`   | Warnings, alerts                        |
| **Yellow** (Highlight) | `#fbbf24` | `rgb(251, 191, 36)` | `--yellow`   | Highlights, featured                    |

### Background & Text

| Name               | Hex       | RGB                  | CSS Variable       | Usage              |
| ------------------ | --------- | -------------------- | ------------------ | ------------------ |
| **Void Black**     | `#030508` | `rgb(3, 5, 8)`       | `--bg-void`        | Primary background |
| **Deep Black**     | `#0a0d14` | `rgb(10, 13, 20)`    | `--bg-deep`        | Card backgrounds   |
| **Surface**        | `#111827` | `rgb(17, 24, 39)`    | `--bg-surface`     | Elevated surfaces  |
| **Elevated**       | `#1f2937` | `rgb(31, 41, 55)`    | `--bg-elevated`    | Hover states       |
| **White**          | `#f9fafb` | `rgb(249, 250, 251)` | `--text-primary`   | Primary text       |
| **Text Secondary** | `#9ca3af` | `rgb(156, 163, 175)` | `--text-secondary` | Secondary text     |
| **Text Muted**     | `#6b7280` | `rgb(107, 114, 128)` | `--text-muted`     | Muted text         |

### Gradients

```css
/* Hero Gradient (Cyan → Purple → Pink) */
--gradient-hero: linear-gradient(135deg, #06b6d4 0%, #a855f7 50%, #ec4899 100%);

/* Cyan-Purple Gradient */
--gradient-cyan-purple: linear-gradient(135deg, #06b6d4, #a855f7);

/* Glow Colors (30% opacity) */
--cyan-glow: rgba(6, 182, 212, 0.3);
--purple-glow: rgba(168, 85, 247, 0.3);
--pink-glow: rgba(236, 72, 153, 0.3);
```

---

## Typography

| Category      | Font             | Fallback                     | Usage                       |
| ------------- | ---------------- | ---------------------------- | --------------------------- |
| **Headlines** | Inter            | -apple-system, sans-serif    | H1, H2, hero text, body     |
| **Display**   | Playfair Display | Georgia, serif               | Special emphasis (optional) |
| **Code**      | JetBrains Mono   | SF Mono, Consolas, monospace | Code, terminals, metrics    |

### Font Weights

| Weight   | Value | Usage                      |
| -------- | ----- | -------------------------- |
| Light    | 300   | Large display text         |
| Regular  | 400   | Body text                  |
| Medium   | 500   | Emphasis, subheadings      |
| Semibold | 600   | Buttons, labels            |
| Bold     | 700   | Headlines, strong emphasis |

---

## Logo System: The Luminous Polyhedron

> **Canonical Logo:** A glass-like hexagonal polyhedron representing multi-faceted, transparent reasoning.

### Visual Metaphor

| Element             | Meaning                                      |
| ------------------- | -------------------------------------------- |
| Transparent facets  | Auditable, see-through reasoning             |
| Multiple faces      | Multi-perspective analysis (GigaThink)       |
| Glowing edges       | Structured protocols lighting the path       |
| Central convergence | All reasoning converges to clear conclusions |

### Logo Variants

| Variant               | File                              | Use Case               |
| --------------------- | --------------------------------- | ---------------------- |
| **Icon Only**         | `logos/logo-icon.svg`             | Favicons, app icons    |
| **Icon Transparent**  | `logos/logo-icon-transparent.svg` | Overlays, watermarks   |
| **Full Horizontal**   | `logos/logo-full-horizontal.svg`  | Headers, footers       |
| **Full with Tagline** | `logos/logo-with-tagline.svg`     | Marketing materials    |
| **Stacked**           | `logos/logo-stacked.svg`          | Square layouts, social |
| **Wordmark**          | `logos/logo-wordmark.svg`         | Text-only situations   |
| **Wordmark Dark**     | `logos/logo-wordmark-dark.svg`    | Dark backgrounds       |
| **Wordmark Light**    | `logos/logo-wordmark-light.svg`   | Light backgrounds      |

### PNG Sizes Available

- `logo-icon-64.png`, `logo-icon-128.png`, `logo-icon-256.png`, `logo-icon-512.png`
- `logo-full-400.png`, `logo-full-800.png`
- `logo-wordmark-300.png`, `logo-wordmark-600.png`

### Usage Rules

- Minimum clear space: 0.5x logo height
- Minimum size: 32px height
- ALWAYS use on dark backgrounds (Void Black preferred)
- NEVER distort, rotate arbitrarily, or recolor
- NEVER add drop shadows (logo has built-in glow effects)

---

## Visual Language

### Tree of Thoughts Motif

The core visual metaphor representing hierarchical reasoning:

- Branching nodes = decision points
- Glowing connections = reasoning pathways
- Root = foundational axioms (BedRock)
- Leaves = conclusions and outputs

### Ambient Orbs

Floating, blurred orbs create depth:

- Cyan, purple, pink colors
- 15% opacity, 60px blur
- 18-28s animation cycles

### Glass Morphism

```css
background: rgba(10, 15, 26, 0.8);
backdrop-filter: blur(12px);
border: 1px solid rgba(255, 255, 255, 0.05);
```

---

## Asset Directory Structure

```
reasonkit-core/brand/                    # CANONICAL ASSET LOCATION
├── BRAND_PLAYBOOK.md                    # THIS FILE - Master source of truth
│
├── logos/                               # Logo variants (18 files)
│   ├── logo-icon.svg                    # Primary icon
│   ├── logo-icon-transparent.svg        # Transparent variant
│   ├── logo-full-horizontal.svg         # Full horizontal
│   ├── logo-stacked.svg                 # Stacked layout
│   ├── logo-with-tagline.svg            # With "Turn Prompts into Protocols"
│   ├── logo-wordmark*.svg               # Text-only variants
│   └── *.png                            # PNG exports (various sizes)
│
├── thinktools/                          # ThinkTool icons (12 files)
│   ├── icon-gigathink.svg               # GigaThink icon
│   ├── icon-laserlogic.svg              # LaserLogic icon
│   ├── icon-bedrock.svg                 # BedRock icon
│   ├── icon-proofguard.svg              # ProofGuard icon
│   ├── icon-brutalhonesty.svg           # BrutalHonesty icon
│   └── icon-*.svg                       # Other feature icons
│
├── badges/                              # Status badges (30 files)
│   ├── badge-rust.svg                   # Rust badge
│   ├── badge-apache-2.svg               # Apache 2.0 license
│   ├── badge-opensource.svg             # Open source badge
│   └── *.png                            # PNG badge exports
│
├── banners/                             # Hero/social banners (22 files)
│   ├── banner-hero.svg                  # Main hero banner
│   ├── banner-og-image.svg              # OG image template
│   ├── github-banner.svg                # GitHub social preview
│   ├── twitter-banner.svg               # Twitter/X banner
│   ├── linkedin-banner.svg              # LinkedIn banner
│   ├── og-social.png                    # Production OG image
│   ├── hero-tree.png                    # Hero tree visual
│   └── cognitive-engine.png             # Cognitive engine visual
│
├── patterns/                            # Background patterns (5 files)
│   ├── pattern-circuit.svg              # Circuit pattern
│   ├── pattern-hexagon.svg              # Hexagon pattern
│   ├── pattern-grid.svg                 # Grid pattern
│   ├── pattern-dots-hex.svg             # Dots pattern
│   └── circuit-void.svg                 # Void circuit pattern
│
├── diagrams/                            # Technical diagrams (13 files)
│   ├── reasontrace-topology.svg         # ReasonTrace visualization
│   ├── how-it-works.svg                 # How it works diagram
│   ├── deployment-modes.svg             # Deployment options
│   ├── thinktool-cards.svg              # ThinkTools overview
│   └── thinktools-chain.svg             # Chain visualization
│
├── favicons/                            # Favicon sizes (9 files)
│   ├── luminous-polyhedron.svg          # Master favicon SVG
│   ├── favicon.svg                      # Standard favicon
│   └── favicon-{16,32,48,64,128,256,512}.png
│
├── avatars/                             # Platform avatars (29 files)
│   ├── avatar-{192,400,800,1024}.png    # Generic sizes
│   ├── github-500.png                   # GitHub avatar
│   ├── twitter-400.png                  # Twitter/X avatar
│   ├── linkedin-400.png                 # LinkedIn avatar
│   └── {platform}-{size}.png            # Platform-specific
│
├── readme/                              # README images (14 files)
│   ├── reasonkit-core_hero.png          # Main README hero
│   ├── powercombo_process.png           # PowerCombo diagram
│   ├── chart_variance_reduction.png     # Performance chart
│   ├── terminal_mockup.png              # Terminal example
│   ├── architecture_diagram.png         # Architecture diagram
│   └── designed_not_dreamed.png         # Footer banner
│
└── launch-svg/                          # Launch SVG assets (5 files)
    ├── github-social-preview.svg        # GitHub preview
    ├── hero-reasoning-engine.svg        # Hero SVG
    ├── terminal-mockup.svg              # Terminal SVG
    ├── thinktools-overview.svg          # ThinkTools overview
    └── variance-reduction-chart.svg     # Chart SVG
```

### Total Asset Count: ~160+ files

---

## Touchpoint-Specific Guidelines

### 1. Website (ReasonKit.sh)

**Primary touchpoint for users discovering ReasonKit.**

| Element         | Asset Location                     | Notes                    |
| --------------- | ---------------------------------- | ------------------------ |
| Favicon         | `favicons/luminous-polyhedron.svg` | Luminous Polyhedron logo |
| Nav Logo        | Inline SVG (Luminous Polyhedron)   | 36x36px, with wordmark   |
| Hero Banner     | `banners/hero-tree.png` or custom  | Cyan/purple gradient     |
| OG Image        | `banners/og-social.png`            | 1200x630px               |
| ThinkTool Icons | `thinktools/icon-*.svg`            | Use in feature cards     |
| Patterns        | `patterns/pattern-*.svg`           | Background accents       |

**Website Asset Paths (RELATIVE):**

```
/assets/brand/logo-*.svg
/assets/brand/favicon*.png
/og-image.png
```

### 2. README.md (ReasonKit-core)

**Primary touchpoint for developers on GitHub.**

| Element        | Asset Path                                    | Notes             |
| -------------- | --------------------------------------------- | ----------------- |
| Hero Image     | `./brand/readme/reasonkit-core_hero.png`      | Main banner       |
| PowerCombo     | `./brand/readme/powercombo_process.png`       | Process diagram   |
| Variance Chart | `./brand/readme/chart_variance_reduction.png` | Performance chart |
| Terminal       | `./brand/readme/terminal_mockup.png`          | CLI example       |
| Architecture   | `./brand/readme/architecture_diagram.png`     | System diagram    |
| Footer         | `./brand/readme/designed_not_dreamed.png`     | Closing banner    |

**README Image Reference Format:**

```markdown
![Description](./brand/readme/filename.png)
```

### 3. Other OSS READMEs (ReasonKit-mem, ReasonKit-web)

Copy relevant assets to each project's `assets/` directory:

- Use consistent hero image style
- Reference `reasonkit-core/brand/` for canonical assets
- Maintain visual consistency

### 4. Social Media

| Platform  | Size         | Asset                           |
| --------- | ------------ | ------------------------------- |
| GitHub    | 1280x640     | `banners/github-banner.svg`     |
| Twitter/X | 1500x500     | `banners/twitter-banner.svg`    |
| LinkedIn  | 1200x627     | `banners/linkedin-banner.svg`   |
| OG/Share  | 1200x630     | `banners/og-social.png`         |
| Avatar    | Per platform | `avatars/{platform}-{size}.png` |

---

## AI Agent Instructions

### For ALL AI Agents Working on ReasonKit

```yaml
BRAND_COMPLIANCE_RULES:
  - ALWAYS use assets from: reasonkit-core/brand/
  - NEVER use off-brand colors (check hex values)
  - NEVER make unverified claims (ProofGuard discipline)
  - ALWAYS use Inter/JetBrains Mono fonts
  - ALWAYS use dark backgrounds (#030508 or #0a0d14)
  - NEVER generate AI text in images (add programmatically)

WHEN_CREATING_IMAGES:
  required_elements:
    - Cyan (#06b6d4) as primary color
    - Void black (#030508) background
    - Purple (#a855f7) and pink (#ec4899) accents
    - Tree/branching/neural network motifs
    - Clean, futuristic, premium aesthetic
    - NO embedded text (add via SVG/CSS later)

WHEN_WRITING_COPY:
  tone:
    - Authoritative
    - Technical
    - Clear
    - Honest (no hype, substantiated claims)
  avoid:
    - Buzzwords without substance
    - Unverified performance claims
    - Excessive exclamation marks
    - Vague or wishy-washy language

ASSET_SELECTION_PRIORITY:
  1. reasonkit-core/brand/           # ALWAYS use this first
  2. reasonkit-core/assets/img/      # For README-specific images
  3. reasonkit-site/assets/          # For website-specific assets
  4. DO NOT use rk-startup assets directly (they're copied to brand/)
```

### Quick Asset Lookup

```
Need a logo?           → brand/logos/
Need a ThinkTool icon? → brand/thinktools/
Need a badge?          → brand/badges/
Need a banner?         → brand/banners/
Need a pattern?        → brand/patterns/
Need a diagram?        → brand/diagrams/
Need a favicon?        → brand/favicons/
Need an avatar?        → brand/avatars/
Need README images?    → brand/readme/
Need launch SVGs?      → brand/launch-svg/
```

---

## Quick Reference Cards

### Color Cheat Sheet

```
PRIMARY:     #06b6d4 (Cyan)
SECONDARY:   #a855f7 (Purple)
TERTIARY:    #ec4899 (Pink)
SUCCESS:     #10b981 (Green)
ALERT:       #f97316 (Orange)
BACKGROUND:  #030508 (Void Black)
CARDS:       #0a0d14 (Deep Black)
TEXT:        #f9fafb (White)
```

### Font Cheat Sheet

```
HEADLINES:  Inter, Bold (700)
BODY:       Inter, Regular (400)
CODE:       JetBrains Mono, Regular (400)
```

### Taglines

```
Primary:    "Turn Prompts into Protocols"
Philosophy: "Designed, Not Dreamed"
Technical:  "Auditable Reasoning by Default"
Competitive: "Structure Beats Intelligence"
```

---

## Version History

| Version | Date       | Changes                                                                                                                                                                   |
| ------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.0.0   | 2025-12-31 | Initial BRAND_PLAYBOOK.md - Consolidated all brand assets into ReasonKit-core/brand/, created comprehensive AI agent instructions, defined touchpoint-specific guidelines |

---

## Enforcement

**This document is the SINGLE SOURCE OF TRUTH for all ReasonKit brand identity.**

All AI agents, projects, documentation, and marketing materials MUST:

1. Use assets from `reasonkit-core/brand/` directory
2. Follow color, typography, and visual guidelines exactly
3. Apply ProofGuard discipline to all claims
4. Maintain consistency across all touchpoints

**Deviations require explicit documentation and approval.**

---

*"Designed, Not Dreamed" | <https://reasonkit.sh>*
