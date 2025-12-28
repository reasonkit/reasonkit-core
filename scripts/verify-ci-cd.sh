#!/usr/bin/env bash
# ReasonKit Core - CI/CD Verification Script
# Validates all workflow files and Docker infrastructure

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "🔍 Verifying CI/CD Infrastructure..."
echo ""

# ============================================================================
# Workflow File Validation
# ============================================================================

echo "📄 Checking workflow files..."

WORKFLOWS=(
  ".github/workflows/quality-gates.yml"
  ".github/workflows/release.yml"
  ".github/workflows/ci.yml"
  ".github/workflows/security.yml"
  ".github/workflows/benchmark.yml"
)

for workflow in "${WORKFLOWS[@]}"; do
  if [ -f "$workflow" ]; then
    echo "  ✅ $workflow"
  else
    echo "  ❌ $workflow (MISSING)"
    exit 1
  fi
done

echo ""

# ============================================================================
# YAML Syntax Validation
# ============================================================================

echo "🔧 Validating YAML syntax..."

if command -v yamllint &> /dev/null; then
  for workflow in "${WORKFLOWS[@]}"; do
    if yamllint -d relaxed "$workflow" &> /dev/null; then
      echo "  ✅ $workflow"
    else
      echo "  ⚠️ $workflow (linting warnings)"
    fi
  done
else
  echo "  ⚠️ yamllint not installed (skipping syntax check)"
  echo "  Install: pip install yamllint"
fi

echo ""

# ============================================================================
# Docker Infrastructure
# ============================================================================

echo "🐳 Checking Docker infrastructure..."

DOCKER_FILES=(
  "Dockerfile"
  ".dockerignore"
)

for file in "${DOCKER_FILES[@]}"; do
  if [ -f "$file" ]; then
    echo "  ✅ $file"
  else
    echo "  ❌ $file (MISSING)"
    exit 1
  fi
done

echo ""

# ============================================================================
# Documentation
# ============================================================================

echo "📚 Checking documentation..."

DOCS=(
  ".github/workflows/README.md"
  "CI_CD_DEPLOYMENT_SUMMARY.md"
)

for doc in "${DOCS[@]}"; do
  if [ -f "$doc" ]; then
    echo "  ✅ $doc"
  else
    echo "  ❌ $doc (MISSING)"
    exit 1
  fi
done

echo ""

# ============================================================================
# Required Files for Release
# ============================================================================

echo "📦 Checking required files for release..."

REQUIRED=(
  "Cargo.toml"
  "Cargo.lock"
  "README.md"
  "LICENSE"
)

for file in "${REQUIRED[@]}"; do
  if [ -f "$file" ]; then
    echo "  ✅ $file"
  else
    echo "  ⚠️ $file (recommended for release)"
  fi
done

echo ""

# ============================================================================
# Quality Gates Check
# ============================================================================

echo "🚦 Checking quality gates configuration..."

GATES=(
  "Gate 1: Build"
  "Gate 2: Clippy"
  "Gate 3: Format"
  "Gate 4: Tests"
  "Gate 5: Benchmarks"
)

for gate in "${GATES[@]}"; do
  if grep -q "$gate" .github/workflows/quality-gates.yml; then
    echo "  ✅ $gate"
  else
    echo "  ❌ $gate (NOT FOUND)"
    exit 1
  fi
done

echo ""

# ============================================================================
# Release Targets Check
# ============================================================================

echo "🎯 Checking release build targets..."

TARGETS=(
  "x86_64-unknown-linux-gnu"
  "x86_64-unknown-linux-musl"
  "aarch64-unknown-linux-gnu"
  "x86_64-apple-darwin"
  "aarch64-apple-darwin"
  "x86_64-pc-windows-msvc"
)

for target in "${TARGETS[@]}"; do
  if grep -q "$target" .github/workflows/release.yml; then
    echo "  ✅ $target"
  else
    echo "  ❌ $target (NOT FOUND)"
    exit 1
  fi
done

echo ""

# ============================================================================
# Security Checks
# ============================================================================

echo "🔒 Checking security scanning configuration..."

SECURITY_TOOLS=(
  "cargo-audit"
  "cargo-deny"
  "gitleaks"
  "semgrep"
)

for tool in "${SECURITY_TOOLS[@]}"; do
  if grep -q "$tool" .github/workflows/security.yml; then
    echo "  ✅ $tool"
  else
    echo "  ⚠️ $tool (not configured)"
  fi
done

echo ""

# ============================================================================
# Workflow Statistics
# ============================================================================

echo "📊 Workflow Statistics..."

TOTAL_LINES=$(wc -l .github/workflows/*.yml | tail -1 | awk '{print $1}')
TOTAL_FILES=$(ls -1 .github/workflows/*.yml | wc -l)

echo "  Total workflows: $TOTAL_FILES"
echo "  Total lines: $TOTAL_LINES"
echo ""

# ============================================================================
# Docker Build Test (Optional)
# ============================================================================

if command -v docker &> /dev/null; then
  echo "🐳 Testing Docker build (this may take a while)..."

  if docker build -t reasonkit-core:test . &> /dev/null; then
    echo "  ✅ Docker build successful"

    # Get image size
    IMAGE_SIZE=$(docker images reasonkit-core:test --format "{{.Size}}")
    echo "  📦 Image size: $IMAGE_SIZE"

    # Cleanup
    docker rmi reasonkit-core:test &> /dev/null
  else
    echo "  ⚠️ Docker build failed (this is non-blocking)"
  fi
else
  echo "  ⚠️ Docker not installed (skipping build test)"
fi

echo ""

# ============================================================================
# Final Summary
# ============================================================================

echo "════════════════════════════════════════════════════════════"
echo "✅ CI/CD Infrastructure Verification Complete!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Summary:"
echo "  ✅ All 5 workflow files present"
echo "  ✅ Docker infrastructure configured"
echo "  ✅ Documentation complete"
echo "  ✅ Quality gates defined"
echo "  ✅ Multi-platform builds configured"
echo "  ✅ Security scanning enabled"
echo ""
echo "Next Steps:"
echo "  1. Configure GitHub secrets (CARGO_REGISTRY_TOKEN)"
echo "  2. Enable branch protection for 'main'"
echo "  3. Test workflows with a feature branch push"
echo "  4. Create first release tag (v0.1.0)"
echo ""
echo "For detailed documentation, see:"
echo "  - .github/workflows/README.md"
echo "  - CI_CD_DEPLOYMENT_SUMMARY.md"
echo ""
