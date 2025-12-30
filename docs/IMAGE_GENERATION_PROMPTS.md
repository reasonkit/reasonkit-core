# ReasonKit Core v1.0 Launch — Image Generation Prompts

> Optimized prompts for FLUX 2 and Midjourney v6.1  
> Target: Hero visualization for "The Reasoning Engine"  
> Brand: Industrial Cyberpunk Aesthetic | "Designed, Not Dreamed"

**Related:** See [ARCHITECTURE_DIAGRAM_PROMPTS.md](./ARCHITECTURE_DIAGRAM_PROMPTS.md) for system architecture visualization prompts.

---

## 🎯 Primary Recommendation: FLUX 2

**FLUX 2 (Black Forest Labs)** delivers photorealistic renders with precise lighting and technical accuracy. Ideal for hero images that require trust-building realism.

**Use:** [FLUX.2 dev Space](https://hf.co/spaces/black-forest-labs/FLUX.2-dev)

**Why FLUX:**

- Photorealistic quality unmatched by other models
- Precise lighting control for technical accuracy
- Speed and efficiency for iteration
- Professional-grade output for brand trust

---

## 🚀 Main Hero Prompt (FLUX 2)

```
A massive crystalline neural network structure floating in deep space, rendered in electric cyan (#06b6d4) and vibrant green (#10b981) against a near-black void (#030508). The core is a luminous geometric polyhedron with subsurface scattering, connected by transparent data pipes with flowing light particles. Five orbital modules surround the core: a cyan explosive star pattern, sharp green laser beams, solid amber foundation blocks, a purple energy shield, and red warning signals. Holographic UI overlays display "LATENCY: 0.004ms" in green, "CONFIDENCE: 94%" in cyan. Red particles being filtered by a cyan energy field represent hallucination detection. This hero image should evoke a sense of wonder, drawing viewers into the boundless possibilities of the Reasoning Engine. Photorealistic ray-traced lighting, volumetric god rays, cinematic depth of field, 8K resolution, ultra-sharp focus, dramatic low-angle hero shot, wide 21:9 aspect ratio, industrial cyberpunk aesthetic, professional color grading
```

### FLUX 2 Technical Settings

| Setting            | Value             | Purpose                        |
| ------------------ | ----------------- | ------------------------------ |
| **Guidance Scale** | 7.5-8.5           | Higher = more prompt adherence |
| **Steps**          | 30-50             | More steps = finer detail      |
| **Aspect Ratio**   | 21:9 or 16:9      | Hero banner format             |
| **Resolution**     | Maximum available | Best quality output            |

---

## 🎨 Midjourney v6.1 — Alternative

**Midjourney** excels at artistic interpretation and dynamic compositions. Best for social media variants and creative explorations.

### Main Hero Prompt (Midjourney Syntax)

```
/imagine prompt: Massive crystalline neural network floating in deep space, electric cyan and vibrant green light against void black, luminous geometric polyhedron core with subsurface scattering, transparent data pipes with flowing light particles, five orbital reasoning modules: cyan star burst, green laser beams, amber blocks, purple energy shield, red warning signals, holographic UI overlay showing metrics, red particles caught in cyan energy field, photorealistic ray-traced lighting, volumetric god rays, cinematic depth of field, dramatic low-angle hero shot, industrial cyberpunk, professional, trustworthy ::5 no human figures, no generic AI brain imagery, no stock photo aesthetic ::2 --ar 21:9 --v 6.1 --style raw --s 250 --c 15
```

### Midjourney Parameter Breakdown

| Parameter     | Value                  | Purpose             |
| ------------- | ---------------------- | ------------------- |
| `--ar 21:9`   | Cinematic aspect ratio | Hero banner format  |
| `--v 6.1`     | Latest model version   | Best quality        |
| `--style raw` | Less stylization       | More photorealistic |
| `--s 250`     | Stylize value          | Balanced creativity |
| `--c 15`      | Chaos                  | Slight variation    |

---

## 📐 Variant Prompts

### Social Media Square (1:1)

**Purpose:** Engaging, shareable content for social platforms. Centralized symmetrical composition to spark curiosity.

**FLUX:**

```
Centered crystalline neural core with five orbiting reasoning modules, electric cyan (#06b6d4) and green (#10b981) against deep black (#030508), geometric polyhedron center with light rays emanating outward, holographic "REASONKIT" text floating above, transparent data streams, volumetric lighting, symmetrical composition, professional tech aesthetic, 8K resolution, ultra-sharp
```

**Midjourney:**

```
/imagine prompt: Crystalline neural core with five orbiting modules, electric cyan and green light, geometric polyhedron center, light rays, holographic text, data streams, volumetric lighting, symmetrical, professional tech ::5 --ar 1:1 --v 6.1 --style raw --s 200
```

---

### GitHub Banner (2:1)

**Purpose:** Wide horizontal banner for GitHub repository. Visualizes "Turn Prompts into Protocols" transformation.

**FLUX:**

```
Wide horizontal composition: crystalline reasoning engine core on the left, cascading data transformation on the right showing chaotic text becoming structured geometric patterns, electric cyan and green palette against near-black. "Turn Prompts into Protocols" is visualized as literal transformation: text evolving seamlessly into cohesive structures, symbolizing clear, actionable insights. Clean negative space in the center for text overlay, professional tech aesthetic, 8K resolution
```

**Midjourney:**

```
/imagine prompt: Wide horizontal: crystalline core left, text-to-structure transformation right, chaotic text becoming geometric patterns, electric cyan and green, void black background, clean negative space center, professional tech ::5 --ar 2:1 --v 6.1 --style raw --s 200
```

---

## 🧠 ThinkTool Module-Specific Variants

### GigaThink Module 💡

**Concept:** Explosive divergent thinking, maximum exploration.

**FLUX:**

```
Explosive cyan star pattern radiating from geometric core, maximum complexity neural pathways, thousands of interconnected nodes, particle systems showing thought expansion, deep black void, volumetric cyan light, photorealistic, 8K
```

**Midjourney:**

```
/imagine prompt: Explosive cyan star burst, neural pathways radiating outward, interconnected nodes, particle expansion, void black, volumetric cyan light ::5 --ar 16:9 --v 6.1 --style raw --s 250
```

---

### LaserLogic Module ⚡

**Concept:** Precision logical validation, sharp convergence.

**FLUX:**

```
Sharp green laser beams cutting through darkness in precise geometric patterns, crystalline prism splitting light into logical branches, binary decision tree visualization, photorealistic lighting, clinical precision aesthetic
```

**Midjourney:**

```
/imagine prompt: Green laser beams, precise geometric patterns, crystalline prism, logical branches, binary tree, clinical precision ::5 --ar 16:9 --v 6.1 --style raw --s 200
```

---

### BedRock Module 🪨

**Concept:** Solid foundation, first principles, stability.

**FLUX:**

```
Solid amber foundation blocks forming hexagonal base structure, stable geometric patterns conveying gravity and weight through lighting, warm amber (#ff9500) against deep black, tactile stone texture, architectural foundation aesthetic, permanence and robustness, photorealistic, 8K
```

**Midjourney:**

```
/imagine prompt: Amber hexagonal blocks, foundation structure, geometric patterns, stone texture, warm amber and black, architectural, stable ::5 --ar 16:9 --v 6.1 --style raw --s 200
```

---

### ProofGuard Module 🛡️

**Concept:** Protective validation, multi-source verification, 95% filtering capability.

**FLUX:**

```
Purple energy shield surrounding the data core, hexagonal force field patterns, validation checkmarks in green appearing through shield gaps, protective barrier visualization, purple (#a855f7) and green (#10b981) palette, mission-critical defense system, reliability and trust, photorealistic, 8K
```

**Midjourney:**

```
/imagine prompt: Purple energy shield, hexagonal force field, green checkmarks, protective barrier, purple and green palette, defense system ::5 --ar 16:9 --v 6.1 --style raw --s 200
```

---

### BrutalHonesty Module 🔥

**Concept:** Adversarial critique, error detection, quality control.

**FLUX:**

```
Red warning signals pulsing through neural network, error detection highlighted in orange (#f97316), filtering mechanism catching anomalous data patterns, quality control visualization, stark contrast lighting, photorealistic, 8K
```

**Midjourney:**

```
/imagine prompt: Red warning signals, orange error detection, filtering mechanism, anomalous patterns, quality control, stark contrast ::5 --ar 16:9 --v 6.1 --style raw --s 200
```

---

## 🚫 Negative Prompt Template

**Always append to avoid common AI image pitfalls:**

```
Negative: human figures, hands, faces, generic AI brain, glowing head, stock photo aesthetic, soft dreamy effects, cluttered background, watermarks, text errors, blurry, low resolution, anime style, cartoon, illustration style, cheesy, over-stylized, unrealistic proportions
```

---

## 🎨 Brand Color Reference

| Color               | Hex       | Use Case                         |
| ------------------- | --------- | -------------------------------- |
| **Electric Cyan**   | `#06b6d4` | Primary, core glow, main actions |
| **Cyber Green**     | `#10b981` | Success, validation, metrics     |
| **Void Black**      | `#030508` | Background, depth                |
| **Accent Purple**   | `#a855f7` | ProofGuard, shields, protection  |
| **Alert Orange**    | `#f97316` | Warnings, BrutalHonesty, errors  |
| **Starlight White** | `#f9fafb` | Text, highlights, UI elements    |
| **Amber**           | `#ff9500` | BedRock, foundation, stability   |

---

## 📐 Typography Reference

**For Text Overlays:**

- **Headlines:** Inter (authoritative, technical)
- **Body/Metrics:** Inter (clear, modern)
- **Code/Data:** JetBrains Mono (monospace, precise)

**Example Overlays:**

- `LATENCY: 0.004ms` (green, Inter)
- `CONFIDENCE: 94%` (cyan, Inter)
- `REASONKIT` (cyan, Inter)
- `Turn Prompts into Protocols` (white, Inter)

---

## 🔄 Post-Processing Workflow

### Step 1: Generation

1. Generate 4-8 variants with main prompt
2. Select strongest composition
3. Run variations (FLUX variations or Midjourney `--v`)
4. Upscale winner to maximum resolution

### Step 2: Color Correction

1. Color grade to match exact brand hex values
2. Adjust contrast for industrial cyberpunk aesthetic
3. Ensure cyan/green pop against void black
4. Verify color accuracy in sRGB and P3

### Step 3: Text Overlays

1. Add brand typography using specified fonts
2. Place metrics overlays (latency, confidence)
3. Ensure text is readable but not dominant
4. Maintain negative space for composition

### Step 4: Export

1. **Print:** 300 DPI, CMYK color space
2. **Web:** 72 DPI, sRGB color space
3. **Social:** Optimized dimensions per platform
4. **GitHub:** 2:1 aspect ratio, PNG format

---

## 📊 Generation Workflow Summary

```
┌─────────────────────────────────────────────────────────┐
│ 1. GENERATE                                              │
│    • 4-8 variants with main prompt                      │
│    • Test different guidance scales (FLUX)              │
│    • Try different chaos values (Midjourney)            │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ 2. SELECT                                                │
│    • Choose strongest composition                        │
│    • Evaluate technical accuracy                         │
│    • Check brand color alignment                         │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ 3. REFINE                                                │
│    • Run variations on selected                          │
│    • Upscale to maximum resolution                       │
│    • Generate module-specific variants                   │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ 4. POST-PROCESS                                          │
│    • Color correct to brand hex values                   │
│    • Add text overlays (typography)                     │
│    • Composite multiple elements if needed               │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ 5. EXPORT                                                │
│    • Print: 300 DPI, CMYK                                │
│    • Web: 72 DPI, sRGB                                  │
│    • Social: Platform-optimized dimensions               │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 Quality Checklist

Before finalizing any image:

- [ ] Colors match brand hex values exactly
- [ ] Typography uses Inter / Inter
- [ ] No human figures or generic AI brain imagery
- [ ] Industrial cyberpunk aesthetic maintained
- [ ] Resolution sufficient for intended use
- [ ] Text overlays readable but not dominant
- [ ] Negative space preserved for composition
- [ ] Professional color grading applied

---

## 📝 Usage Notes

### For Hero Images

- Use main hero prompt with 21:9 aspect ratio
- Focus on crystalline core with five modules
- Emphasize variance reduction visualization
- Include holographic metrics overlays

### For Social Media

- Use square (1:1) variant for Instagram/Twitter
- Symmetrical composition works best
- Keep text minimal or add in post-processing
- Optimize for mobile viewing

### For Documentation

- Use module-specific variants for individual ThinkTool pages
- Maintain consistent color palette
- Show clear module differentiation
- Professional, trustworthy aesthetic

---

## 🔗 Resources

- **FLUX 2:** [Hugging Face Space](https://hf.co/spaces/black-forest-labs/FLUX.2-dev)
- **Midjourney:** [Official Site](https://www.midjourney.com)
- **Brand Colors:** See color reference table above
- **Typography:** Inter, Inter, JetBrains Mono

---

**Generated:** 2025-12-29  
**For:** ReasonKit Core v1.0 Launch  
**Status:** ✅ Ready for Image Generation

---

_"Designed, Not Dreamed. Turn Prompts into Protocols."_  
*https://reasonkit.sh*
