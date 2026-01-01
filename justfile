# ReasonKit Core - Development Commands
# Usage: just <command>
# Install just: cargo install just

# Default recipe - show available commands
default:
    @just --list

# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY GATES (CONS-009) - All must pass before merge
# ═══════════════════════════════════════════════════════════════════════════════

# Run ALL quality gates (use before every PR)
qa: gate1 gate2 gate3 gate4
    @echo "✅ All quality gates passed!"

# Gate 1: Build (BLOCKING)
gate1:
    @echo "🔨 [Gate 1/5] Building release..."
    cargo build --release

# Gate 2: Lint with Clippy (BLOCKING)
gate2:
    @echo "🔍 [Gate 2/5] Running Clippy..."
    cargo clippy -- -D warnings

# Gate 3: Format check (BLOCKING)
gate3:
    @echo "📝 [Gate 3/5] Checking format..."
    cargo fmt --check

# Gate 4: Run tests (BLOCKING)
gate4:
    @echo "🧪 [Gate 4/5] Running tests..."
    cargo test --all-features

# Gate 5: Benchmarks (MONITORING - not blocking)
gate5:
    @echo "📊 [Gate 5/5] Running benchmarks..."
    cargo bench --bench retrieval_bench || echo "⚠️ Benchmarks need setup"

# Full quality metrics report
metrics:
    @echo "📈 Running full quality metrics..."
    ./scripts/quality_metrics.sh

# CI mode metrics (exits with error code)
metrics-ci:
    ./scripts/quality_metrics.sh --ci

# ═══════════════════════════════════════════════════════════════════════════════
# DEVELOPMENT COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

# Build debug
build:
    cargo build

# Build release
release:
    cargo build --release

# Run tests with output
test:
    cargo test -- --nocapture

# Run tests quietly
test-quiet:
    cargo test

# Run specific test
test-one NAME:
    cargo test {{NAME}} -- --nocapture

# Watch for changes and run tests
watch:
    cargo watch -x test

# Format code
fmt:
    cargo fmt

# Fix clippy warnings automatically
fix:
    cargo clippy --fix --allow-dirty

# Clean build artifacts
clean:
    cargo clean

# ═══════════════════════════════════════════════════════════════════════════════
# CLI COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

# Run CLI with arguments
run *ARGS:
    cargo run --release -- {{ARGS}}

# Show CLI help
help:
    cargo run --release -- --help

# Ingest documents
ingest PATH:
    cargo run --release -- ingest {{PATH}}

# Query the knowledge base
query QUERY:
    cargo run --release -- query "{{QUERY}}"

# Show statistics
stats:
    cargo run --release -- stats

# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

# Generate and open documentation
docs:
    cargo doc --open

# Generate documentation without opening
docs-build:
    cargo doc --no-deps

# Check documentation links (requires lychee)
docs-check:
    @echo "🔗 Checking documentation links..."
    @if command -v lychee >/dev/null; then \
        lychee --exclude-mail "docs/**/*.md" "README.md"; \
    else \
        echo "⚠️ lychee not found. Install with 'cargo install lychee'"; \
    fi

# List documentation structure
docs-tree:
    @echo "📂 Documentation Structure:"
    @tree docs -I "assets|images" --dirsfirst

# ═══════════════════════════════════════════════════════════════════════════════
# SPEC PANEL REVIEWS (Monthly requirement)
# ═══════════════════════════════════════════════════════════════════════════════

# Run spec panel review (requires Claude Code)
spec-panel:
    @echo "🔬 Launching Expert Specification Panel Review..."
    @echo "Run in Claude Code: /sc:spec-panel \"cd $(pwd)\""

# Show review protocol
review-protocol:
    @cat REVIEW_PROTOCOL.md | head -100

# Show QA plan
qa-plan:
    @cat QA_PLAN.md | head -100

# ═══════════════════════════════════════════════════════════════════════════════
# GIT HOOKS SETUP
# ═══════════════════════════════════════════════════════════════════════════════

# Install git hooks for quality enforcement
hooks-install:
    @echo "🪝 Installing git hooks..."
    @mkdir -p .git/hooks
    @echo '#!/bin/bash' > .git/hooks/pre-commit
    @echo 'echo "🔍 Running pre-commit quality gates..."' >> .git/hooks/pre-commit
    @echo 'just gate1 && just gate3 && just gate4' >> .git/hooks/pre-commit
    @echo 'if [ $$? -ne 0 ]; then' >> .git/hooks/pre-commit
    @echo '    echo "❌ Quality gates failed. Commit blocked."' >> .git/hooks/pre-commit
    @echo '    exit 1' >> .git/hooks/pre-commit
    @echo 'fi' >> .git/hooks/pre-commit
    @echo 'echo "✅ Pre-commit checks passed!"' >> .git/hooks/pre-commit
    @chmod +x .git/hooks/pre-commit
    @echo '#!/bin/bash' > .git/hooks/pre-push
    @echo 'echo "🚀 Running pre-push quality gates..."' >> .git/hooks/pre-push
    @echo 'just qa' >> .git/hooks/pre-push
    @echo 'if [ $$? -ne 0 ]; then' >> .git/hooks/pre-push
    @echo '    echo "❌ Quality gates failed. Push blocked."' >> .git/hooks/pre-push
    @echo '    exit 1' >> .git/hooks/pre-push
    @echo 'fi' >> .git/hooks/pre-push
    @chmod +x .git/hooks/pre-push
    @echo "✅ Git hooks installed!"

# Remove git hooks
hooks-remove:
    @rm -f .git/hooks/pre-commit .git/hooks/pre-push
    @echo "🗑️ Git hooks removed"

# Show hook status
hooks-status:
    @echo "Git hooks status:"
    @ls -la .git/hooks/pre-commit 2>/dev/null || echo "  pre-commit: not installed"
    @ls -la .git/hooks/pre-push 2>/dev/null || echo "  pre-push: not installed"

# ═══════════════════════════════════════════════════════════════════════════════
# OPENROUTER / AI MODEL MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

# Fetch latest models from OpenRouter API
fetch-models:
    @echo "🤖 Fetching OpenRouter models..."
    @curl -s https://openrouter.ai/api/v1/models | jq '.data | length' | xargs -I {} echo "Found {} models"
    @curl -s https://openrouter.ai/api/v1/models > data/models/openrouter_models.json
    @echo "✅ Models saved to data/models/openrouter_models.json"

# Show top models by context length
models-context:
    @cat data/models/openrouter_models.json 2>/dev/null | jq -r '.data | sort_by(.context_length) | reverse | .[0:20] | .[] | "\(.context_length // 0) tokens - \(.id)"' || echo "Run 'just fetch-models' first"

# Show cheapest models
models-cheap:
    @cat data/models/openrouter_models.json 2>/dev/null | jq -r '.data | sort_by(.pricing.prompt | tonumber) | .[0:20] | .[] | "$\(.pricing.prompt)/1M - \(.id)"' || echo "Run 'just fetch-models' first"

# ═══════════════════════════════════════════════════════════════════════════════
# QUICK SHORTCUTS
# ═══════════════════════════════════════════════════════════════════════════════

# Quick check (build + test)
check: build test-quiet
    @echo "✅ Quick check passed"

# Full CI pipeline
ci: qa metrics-ci
    @echo "✅ CI pipeline complete"

# Prepare for PR
pr: fmt qa
    @echo "✅ Ready for PR!"

# Morning standup check
standup:
    @echo "📋 ReasonKit Core Status"
    @echo "========================"
    @git status --short
    @echo ""
    @echo "Recent commits:"
    @git log --oneline -5
    @echo ""
    @echo "TODO count: $(grep -r 'TODO' src --include='*.rs' | wc -l)"
    @echo "FIXME count: $(grep -r 'FIXME' src --include='*.rs' | wc -l)"
