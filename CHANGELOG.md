# Changelog

All notable changes to ReasonKit Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-supremacy] - 2026-01-08

### The "Level Up" Overhaul

This release represents a comprehensive quality and compliance overhaul, transitioning
ReasonKit from prototype to production-ready.

### Changed

- **CLI Branding**: Updated help screen with comprehensive ThinkTools descriptions
  - Added full ThinkTools list: GigaThink, LaserLogic, BedRock, ProofGuard, BrutalHonesty
  - Added Profiles with confidence levels (70%/80%/85%/95%)
  - Added usage examples and documentation links
  - Aligned "about" text with brand identity: "The Reasoning Engine"

### Fixed

- **Compliance**: Removed dishonest marketing claims from website
  - "5,000+ developers" → "a growing community of developers"
  - "340+ Models Supported" → "9 Providers Tested"
  - "+ 330+ more via OpenRouter" → "+ Any model via OpenRouter"
- **Brand Identity**: Enforced CONS-014 (canonical description rule)

### Documentation

- Created Task 1039: LLM Provider Validation Playbook specification
  - Comprehensive testing protocol for 9 providers
  - Evidence-based validation matrix template
  - Follow-up PDF compilation task for Gemini 3 Pro

### Quality

- Verified install.sh first-run experience (ASCII banner, interactive wizard)
- Verified CLI help matches ThinkTools branding
- All changes aligned with ORCHESTRATOR.md quality gates

---

## [0.1.0] - 2025-01-01

### Added

- Initial release of ReasonKit Core
- **ThinkTool Protocol System**
  - Protocol definition and execution engine
  - Step-by-step reasoning workflows
  - Branch conditions and decision points
  - Output formatting (Text, List, Boolean)
  - Protocol validation and error handling
- **Aesthetic Expression Mastery System**
  - Visual design assessment (color, typography, layout)
  - Usability evaluation (UX patterns, interaction design)
  - Accessibility assessment (WCAG 2.1 AA/AAA compliance)
  - 3D rendering evaluation (React Three Fiber, WebGL)
  - Cross-platform design validation (Web, iOS, Android)
  - Performance impact analysis
  - VIBE Benchmark integration (M2-proven benchmarks)
- **Reasoning Profiles**
  - GigaThink: Expansive creative thinking
  - LaserLogic: Precision deductive reasoning
  - BedRock: First principles decomposition
  - ProofGuard: Multi-source verification
  - BrutalHonesty: Adversarial self-critique
- **LLM Integration**
  - Unified LLM client interface
  - Support for multiple providers
  - Token usage tracking
  - Request/response handling
- **Python Bindings** (optional)
  - PyO3 integration for Python interoperability
  - Extension module support

### Fixed

- All build errors resolved
- Test suite compilation issues fixed
  - Fixed aesthetic test type annotations
  - Resolved module import paths
- Security vulnerabilities addressed
  - String format injection protection
  - Unsafe unwrap() usage limited to tests
- Code formatting and linting issues

### Documentation

- Comprehensive README with examples
- API documentation
- Architecture documentation
- Getting started guides

### Performance

- Optimized protocol execution
- Efficient LLM request handling
- Memory-efficient data structures

---

## [Unreleased]

### Planned

- Enhanced reasoning profiles
- Additional LLM provider support
- Performance benchmarks
- More comprehensive examples

---

[0.1.0]: https://github.com/reasonkit/reasonkit-core/releases/tag/v0.1.0
