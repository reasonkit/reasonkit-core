#!/bin/bash
# ReasonKit LLM CLI Ecosystem Setup
# Integrates Simon Willison's LLM tools with ReasonKit infrastructure

set -euo pipefail

echo "============================================"
echo "  ReasonKit LLM CLI Ecosystem Setup"
echo "============================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check for Python
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 is required. Install it first."
    exit 1
fi

# Detect package manager
if command -v uv &> /dev/null; then
    PKG_MGR="uv tool install"
    PKG_INSTALL="uv pip install"
    log_info "Using uv as package manager"
elif command -v pipx &> /dev/null; then
    PKG_MGR="pipx install"
    PKG_INSTALL="pipx inject llm"
    log_info "Using pipx as package manager"
else
    PKG_MGR="pip install"
    PKG_INSTALL="pip install"
    log_info "Using pip as package manager (consider uv or pipx for isolation)"
fi

echo ""
echo "=== PHASE 1: Core LLM CLI Installation ==="
echo ""

# Install core llm tool
log_info "Installing llm CLI..."
$PKG_MGR llm || pip install llm

# Verify installation
if command -v llm &> /dev/null; then
    log_info "llm CLI installed successfully: $(llm --version)"
else
    log_error "llm CLI installation failed"
    exit 1
fi

echo ""
echo "=== PHASE 2: Model Provider Plugins ==="
echo ""

# Model providers
PROVIDER_PLUGINS=(
    "llm-anthropic"        # Claude models
    "llm-openrouter"       # Multi-provider routing
    "llm-mistral"          # Mistral AI
)

# Optional: Add llm-ollama if Ollama is installed
if command -v ollama &> /dev/null; then
    PROVIDER_PLUGINS+=("llm-ollama")
    log_info "Ollama detected - adding llm-ollama plugin"
fi

for plugin in "${PROVIDER_PLUGINS[@]}"; do
    log_info "Installing $plugin..."
    llm install "$plugin" || log_warn "Failed to install $plugin (may be optional)"
done

echo ""
echo "=== PHASE 3: Tool Plugins ==="
echo ""

# Essential tool plugins
TOOL_PLUGINS=(
    "llm-tools-simpleeval"  # Math calculations
    "llm-tools-sqlite"      # Database queries
    "llm-cluster"           # Semantic clustering
    "llm-cmd"               # Shell command generation
)

for plugin in "${TOOL_PLUGINS[@]}"; do
    log_info "Installing $plugin..."
    llm install "$plugin" || log_warn "Failed to install $plugin"
done

echo ""
echo "=== PHASE 4: Fragment & Template Plugins ==="
echo ""

# Fragment loaders for context
FRAGMENT_PLUGINS=(
    "llm-fragments-github"  # GitHub repos/issues/PRs
)

# Only try PDF if PyMuPDF available
if python3 -c "import fitz" 2>/dev/null; then
    FRAGMENT_PLUGINS+=("llm-fragments-pdf")
fi

for plugin in "${FRAGMENT_PLUGINS[@]}"; do
    log_info "Installing $plugin..."
    llm install "$plugin" || log_warn "Failed to install $plugin"
done

echo ""
echo "=== PHASE 5: Embedding Models ==="
echo ""

# Local embedding support
log_info "Installing sentence-transformers plugin for local embeddings..."
llm install llm-sentence-transformers || log_warn "Failed to install llm-sentence-transformers"

# Download a lightweight embedding model
log_info "Downloading all-MiniLM-L12-v2 embedding model..."
llm embed -m sentence-transformers/all-MiniLM-L12-v2 -c "test" > /dev/null 2>&1 || true

echo ""
echo "=== PHASE 6: Datasette Integration ==="
echo ""

# Install datasette and sqlite-utils
log_info "Installing Datasette and sqlite-utils..."
pip install datasette sqlite-utils || log_warn "Datasette installation may need attention"

echo ""
echo "=== PHASE 7: ReasonKit Templates ==="
echo ""

TEMPLATE_DIR="${HOME}/.config/io.datasette.llm/templates"
mkdir -p "$TEMPLATE_DIR"

# Create ReasonKit protocol templates
cat > "$TEMPLATE_DIR/rk-quick.yaml" << 'EOF'
name: rk-quick
description: ReasonKit QUICK protocol (5 steps, 30s, 0.7 confidence)
system: |
  Execute ReasonKit QUICK protocol:
  1. UNDERSTAND: Clarify the problem (10s)
  2. ANALYZE: Quick assessment (10s)
  3. DECIDE: Choose approach (5s)
  4. EXECUTE: Provide answer (5s)
  5. CONFIDENCE: Rate 0.00-1.00 (must be >= 0.70)

  Format your response with clear step labels.
  End with: **Confidence: X.XX**
EOF

cat > "$TEMPLATE_DIR/rk-scientific.yaml" << 'EOF'
name: rk-scientific
description: ReasonKit SCIENTIFIC protocol (7 steps, 2min, 0.85 confidence)
system: |
  Execute ReasonKit SCIENTIFIC protocol:
  1. UNDERSTAND: Define problem precisely
  2. DECOMPOSE: Break into sub-problems
  3. RESEARCH: Gather evidence (cite sources)
  4. HYPOTHESIZE: Form potential solutions
  5. ANALYZE: Evaluate each hypothesis
  6. VALIDATE: Self-critique, find flaws
  7. CONCLUDE: Final answer with confidence >= 0.85

  Cite sources where applicable.
  End with: **Confidence: X.XX** (must be >= 0.85)
EOF

cat > "$TEMPLATE_DIR/rk-absolute.yaml" << 'EOF'
name: rk-absolute
description: ReasonKit ABSOLUTE protocol (10 steps, 5min, 0.95 confidence)
system: |
  Execute ReasonKit ABSOLUTE protocol with maximum rigor:
  1. UNDERSTAND: Deep problem comprehension
  2. CONTEXT: Gather all relevant context
  3. DECOMPOSE: Atomic sub-problem breakdown
  4. RESEARCH: Exhaustive evidence gathering
  5. HYPOTHESIZE: Generate multiple solutions
  6. ANALYZE: Rigorous logical analysis
  7. CRITIQUE: Adversarial self-review
  8. REFINE: Improve based on critique
  9. VALIDATE: External validation check
  10. CONCLUDE: Final answer with confidence >= 0.95

  This is the highest rigor protocol.
  All claims must be verifiable.
  End with: **Confidence: X.XX** (must be >= 0.95)
EOF

cat > "$TEMPLATE_DIR/rust-review.yaml" << 'EOF'
name: rust-review
description: Rust code review with ReasonKit standards
system: |
  You are a senior Rust engineer reviewing code for ReasonKit.

  Evaluate these dimensions (1-5 scale):
  - SAFETY: Memory safety, no unsafe without justification
  - PERFORMANCE: Zero-cost abstractions, no unnecessary allocations
  - IDIOM: Follows Rust conventions, proper error handling
  - CLARITY: Readable, well-documented where needed
  - TESTING: Testable design, edge cases considered

  Provide specific line-level feedback.
  End with overall rating and top 3 improvements.
EOF

cat > "$TEMPLATE_DIR/triangulate.yaml" << 'EOF'
name: triangulate
description: Research triangulation (3+ sources required)
system: |
  You are executing ReasonKit's Deep Research Triangulation Protocol.

  REQUIREMENTS:
  1. Find information from AT LEAST 3 independent sources
  2. Categorize sources by tier (Official > Reputable > Community)
  3. Note any conflicts between sources
  4. Provide confidence-weighted synthesis

  FORMAT:
  ## Source 1: [Name] (Tier: X)
  Key findings: ...

  ## Source 2: [Name] (Tier: X)
  Key findings: ...

  ## Source 3: [Name] (Tier: X)
  Key findings: ...

  ## Synthesis
  Consensus: ...
  Conflicts: ...
  **Final Answer** (Confidence: X.XX)
EOF

log_info "Created 5 ReasonKit templates in $TEMPLATE_DIR"

echo ""
echo "=== PHASE 8: Verification ==="
echo ""

log_info "Installed plugins:"
llm plugins

log_info "Available templates:"
llm templates list 2>/dev/null || llm templates

log_info "Logs database location:"
llm logs path

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Set API keys:  llm keys set anthropic"
echo "  2. Test:          llm -m claude-haiku 'Hello, ReasonKit!'"
echo "  3. Use template:  llm -t rk-quick 'Your question'"
echo "  4. View logs:     datasette \$(llm logs path)"
echo ""
echo "Documentation: reasonkit-core/docs/LLM_CLI_INTEGRATION.md"
echo ""
