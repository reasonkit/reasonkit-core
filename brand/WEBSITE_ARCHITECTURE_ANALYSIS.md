# Website Architecture Analysis

> **PowerCombo + BrutalHonesty Assessment**
> **Date:** 2025-12-31
> **Decision Required:** How to structure ReasonKit-site relative to ReasonKit-core

---

## Executive Summary

**RECOMMENDATION: Option A - Website Internal + Assets from ReasonKit-core**

This is the TOP BEST approach because it:

1. Keeps `reasonkit-core` as the SINGLE source of truth for brand assets
2. Maintains clean separation between library code and marketing
3. Doesn't expose marketing strategy to competitors
4. Minimizes maintenance burden

---

## Option Comparison Matrix

| Criterion              | Weight | Option A (Internal) | Option B (OSS Site) | Option C (Merged) |
| ---------------------- | ------ | ------------------- | ------------------- | ----------------- |
| Brand Consistency      | 25%    | ★★★★★ (5)           | ★★★☆☆ (3)           | ★★★★★ (5)         |
| Maintenance Burden     | 20%    | ★★★★★ (5)           | ★★★☆☆ (3)           | ★★☆☆☆ (2)         |
| Security/Privacy       | 20%    | ★★★★★ (5)           | ★★☆☆☆ (2)           | ★★☆☆☆ (2)         |
| Developer UX           | 15%    | ★★★★☆ (4)           | ★★★☆☆ (3)           | ★★☆☆☆ (2)         |
| Deployment Simplicity  | 10%    | ★★★★☆ (4)           | ★★★★★ (5)           | ★★★☆☆ (3)         |
| Community Contribution | 10%    | ★★☆☆☆ (2)           | ★★★★☆ (4)           | ★★☆☆☆ (2)         |
| **WEIGHTED SCORE**     | 100%   | **4.45**            | **3.15**            | **2.80**          |

---

## BrutalHonesty Critique of Each Option

### Option A: Website Internal (RECOMMENDED)

**Honest Pros:**

- Clean separation of concerns
- Brand assets live in the OSS repo (ReasonKit-core/brand/)
- Marketing strategy stays private
- Minimal maintenance overhead
- Website can pull assets from published crate or git submodule

**Honest Cons:**

- Need a mechanism to sync assets (but this is trivial)
- Community can't contribute to website (but website PRs are rare anyway)
- Two repos to manage (but one is internal, low overhead)

**Brutal Truth:** This is the RIGHT approach for a commercial OSS product.
The website is MARKETING, not CODE. Keeping it internal protects competitive
positioning while the BRAND ASSETS (which community needs) are in the OSS repo.

---

### Option B: ReasonKit-site as OSS

**Honest Pros:**

- Transparency
- Community can fix typos, improve docs

**Honest Cons:**

- Exposes marketing copy, positioning, competitor comparisons
- Creates asset synchronization problem (which repo is source of truth?)
- More repos = more maintenance
- Website changes are rare; doesn't justify OSS overhead

**Brutal Truth:** This is VANITY open-sourcing. The website is static HTML with
marketing copy. There's no meaningful contribution opportunity. The cost (exposed
strategy, sync complexity) far outweighs the negligible benefit.

---

### Option C: Merge into ReasonKit-core

**Honest Pros:**

- Truly single repo

**Honest Cons:**

- 22MB website bloat in a Rust library
- Confuses library purpose ("is this a library or a website?")
- Different build systems (cargo vs static hosting)
- Website changes trigger library CI
- Contributors confused about what to work on

**Brutal Truth:** This violates basic software engineering principles. A library
should be a library. Mixing marketing HTML into a Rust crate is architectural
pollution. The "single repo" benefit is not worth the confusion cost.

---

## Implementation Plan for Option A

### Asset Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    REASONKIT-CORE (OSS)                         │
│                                                                 │
│   brand/                    ← CANONICAL ASSET SOURCE            │
│   ├── BRAND_PLAYBOOK.md                                         │
│   ├── logos/                                                    │
│   ├── badges/                                                   │
│   ├── banners/                                                  │
│   └── ...                                                       │
│                                                                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ Assets flow DOWN (sync/copy)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   REASONKIT-SITE (Internal)                     │
│                                                                 │
│   assets/brand/ ← Synced from reasonkit-core/brand/             │
│   index.html                                                    │
│   main.css                                                      │
│   ...                                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Sync Mechanism Options

1. **Git Submodule** (Recommended)

   ```bash
   # In reasonkit-site
   git submodule add ../reasonkit-core brand-source
   # Use brand-source/brand/ in website
   ```

2. **Build-time Copy Script**

   ```bash
   # scripts/sync-brand.sh
   cp -r ../reasonkit-core/brand/* ./assets/brand/
   ```

3. **Symlink** (Development only)

   ```bash
   ln -s ../reasonkit-core/brand ./assets/brand
   ```

---

## Conclusion

**Option A is the CLEAR WINNER.**

The website should remain internal with assets sourced from `reasonkit-core/brand/`.
This maintains:

- Single source of truth for brand (in OSS repo)
- Clean separation of concerns
- Protected marketing strategy
- Minimal maintenance burden

---

*Analysis performed with PowerCombo (GigaThink → LaserLogic → BedRock → ProofGuard → BrutalHonesty)*
