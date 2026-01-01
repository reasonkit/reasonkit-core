# ReasonKit Community Badge System
## "Reasoned By" / "Protocol: ReasonKit" Badges

> **Classification:** Community Marketing & Growth Tool
> **Purpose:** Enable viral growth - every project using ReasonKit becomes an advertisement
> **Philosophy:** "Designed, Not Dreamed" - Professional, branded, consistent

---

## Badge Variants

### 1. "Reasoned By ReasonKit"

**Primary badge** - For projects that use ReasonKit for reasoning/validation.

**Visual Design:**
- Hexagonal icon (matching Luminous Polyhedron logo)
- "Reasoned By" text in Inter font
- "ReasonKit" text with brand gradient (Cyan → Purple → Pink)
- Dark background (Void Black `#030508`)
- Optional: "Inside" text variant

### 2. "Protocol: ReasonKit"

**Technical badge** - For projects that implement ReasonKit protocols.

**Visual Design:**
- Same hexagonal icon
- "Protocol:" prefix in muted text
- "ReasonKit" in brand gradient
- Slightly more technical aesthetic

### 3. "Powered By ReasonKit"

**General badge** - For any project using ReasonKit.

**Visual Design:**
- Same hexagonal icon
- "Powered By" text
- "ReasonKit" in brand gradient

---

## Badge Specifications

### Sizes

| Size | Dimensions | Use Case |
|------|------------|----------|
| **Small** | 120×28px | Inline in README, compact spaces |
| **Medium** | 160×38px | Standard README badge (recommended) |
| **Large** | 200×48px | Hero sections, landing pages |

### Formats

- **SVG** (recommended) - Scalable, crisp at any size
- **PNG** - Fallback for environments that don't support SVG
- **Markdown** - Direct integration in READMEs

---

## Badge Assets

### SVG Implementation

```svg
<svg width="160" height="38" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="160" height="38" fill="#030508" rx="6"/>
  
  <!-- Hexagonal icon -->
  <g transform="translate(12, 8)">
    <polygon 
      points="11,0 22,6 22,16 11,22 0,16 0,6" 
      fill="none" 
      stroke="url(#badgeGradient)" 
      stroke-width="1.5"
    />
    <!-- Inner glow -->
    <polygon 
      points="11,2 20,7 20,15 11,20 2,15 2,7" 
      fill="url(#badgeGradient)" 
      opacity="0.2"
    />
  </g>
  
  <!-- Text -->
  <text x="36" y="24" font-family="Inter, sans-serif" font-size="11" fill="#9ca3af">
    Reasoned By
  </text>
  <text x="36" y="36" font-family="Inter, sans-serif" font-size="13" font-weight="600" fill="url(#badgeGradient)">
    ReasonKit
  </text>
  
  <!-- Gradient definition -->
  <defs>
    <linearGradient id="badgeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#06b6d4;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#a855f7;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#ec4899;stop-opacity:1" />
    </linearGradient>
  </defs>
</svg>
```

---

## README Integration

### Markdown Badge (Recommended)

**Standard Badge:**
```markdown
[![Reasoned By ReasonKit](https://img.shields.io/badge/Reasoned%20By-ReasonKit-06b6d4?style=for-the-badge&logo=data:image/svg+xml;base64,...)](https://reasonkit.sh)
```

**Custom Badge (SVG):**
```markdown
[![Reasoned By ReasonKit](https://reasonkit.sh/badges/reasoned-by.svg)](https://reasonkit.sh)
```

**HTML Badge (for websites):**
```html
<a href="https://reasonkit.sh" target="_blank" rel="noopener noreferrer">
  <img 
    src="https://reasonkit.sh/badges/reasoned-by.svg" 
    alt="Reasoned By ReasonKit" 
    width="160" 
    height="38"
  />
</a>
```

### Badge Placement

**Recommended locations:**
1. **Top of README** - After project title, before description
2. **Features section** - Alongside other technology badges
3. **Footer** - At the bottom with other acknowledgments

**Example README structure:**
```markdown
# My Awesome Project

[![Reasoned By ReasonKit](https://reasonkit.sh/badges/reasoned-by.svg)](https://reasonkit.sh)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> Project description here

## Features

- Feature 1 (validated with ReasonKit)
- Feature 2 (uses ReasonKit protocols)

## Installation

```bash
cargo add reasonkit-core
```

## Acknowledgments

Built with [ReasonKit](https://reasonkit.sh) for structured, auditable reasoning.
```

---

## Badge Generator

### Online Tool

**URL:** `https://reasonkit.sh/badges/generator`

**Parameters:**
- `variant`: `reasoned-by` | `protocol` | `powered-by`
- `size`: `small` | `medium` | `large`
- `style`: `flat` | `glow` | `minimal`

**Example:**
```
https://reasonkit.sh/badges/generator?variant=reasoned-by&size=medium&style=glow
```

### API Endpoint

**Endpoint:** `GET /api/badges/generate`

**Request:**
```bash
curl "https://reasonkit.sh/api/badges/generate?variant=reasoned-by&size=medium" \
  -o badge.svg
```

**Response:** SVG content

---

## Badge Variants

### 1. "Reasoned By ReasonKit"

**Markdown:**
```markdown
[![Reasoned By ReasonKit](https://reasonkit.sh/badges/reasoned-by.svg)](https://reasonkit.sh)
```

**HTML:**
```html
<a href="https://reasonkit.sh">
  <img src="https://reasonkit.sh/badges/reasoned-by.svg" alt="Reasoned By ReasonKit" />
</a>
```

### 2. "Protocol: ReasonKit"

**Markdown:**
```markdown
[![Protocol: ReasonKit](https://reasonkit.sh/badges/protocol.svg)](https://reasonkit.sh)
```

**HTML:**
```html
<a href="https://reasonkit.sh">
  <img src="https://reasonkit.sh/badges/protocol.svg" alt="Protocol: ReasonKit" />
</a>
```

### 3. "Powered By ReasonKit"

**Markdown:**
```markdown
[![Powered By ReasonKit](https://reasonkit.sh/badges/powered-by.svg)](https://reasonkit.sh)
```

**HTML:**
```html
<a href="https://reasonkit.sh">
  <img src="https://reasonkit.sh/badges/powered-by.svg" alt="Powered By ReasonKit" />
</a>
```

---

## Usage Guidelines

### When to Use

✅ **Appropriate:**
- Projects that use ReasonKit for reasoning/validation
- Projects that implement ReasonKit protocols
- Projects that depend on ReasonKit crates
- Documentation sites for ReasonKit-based tools

❌ **Inappropriate:**
- Projects that only mention ReasonKit in passing
- Projects that don't actually use ReasonKit
- Competitor products (unless explicitly partnered)

### Brand Compliance

- **Always link to** `https://reasonkit.sh`
- **Use official badge assets** (don't create custom versions)
- **Maintain aspect ratio** when resizing
- **Don't modify colors** - use brand gradient as-is
- **Respect minimum size** - 120×28px minimum

---

## Badge Assets Location

**Source Files:**
- `reasonkit-core/brand/badges/reasoned-by.svg`
- `reasonkit-core/brand/badges/protocol.svg`
- `reasonkit-core/brand/badges/powered-by.svg`

**CDN URLs:**
- `https://reasonkit.sh/badges/reasoned-by.svg`
- `https://reasonkit.sh/badges/protocol.svg`
- `https://reasonkit.sh/badges/powered-by.svg`

**GitHub Releases:**
- Versioned badge packages in `brand-assets-*` releases

---

## Analytics & Tracking

**Badge clicks are tracked** (anonymously) to measure:
- Community adoption
- Badge placement effectiveness
- Referral traffic to reasonkit.sh

**Privacy:** No personal data collected, only aggregate statistics.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-01 | Initial badge system specification |

---

## Examples

### Real-World Usage

**Example 1: README Top**
```markdown
# My Reasoning Engine

[![Reasoned By ReasonKit](https://reasonkit.sh/badges/reasoned-by.svg)](https://reasonkit.sh)

A powerful reasoning engine built on ReasonKit protocols.
```

**Example 2: Features Section**
```markdown
## Features

- ✅ Structured reasoning (via ReasonKit)
- ✅ Auditable decision trees
- ✅ Multi-perspective analysis

[![Protocol: ReasonKit](https://reasonkit.sh/badges/protocol.svg)](https://reasonkit.sh)
```

**Example 3: Footer**
```markdown
---

Built with [ReasonKit](https://reasonkit.sh) - Turn Prompts into Protocols

[![Powered By ReasonKit](https://reasonkit.sh/badges/powered-by.svg)](https://reasonkit.sh)
```

---

**"Designed, Not Dreamed" - Turn Prompts into Protocols**
*https://reasonkit.sh*
