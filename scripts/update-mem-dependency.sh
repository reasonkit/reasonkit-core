#!/bin/bash
# Update reasonkit-mem dependency from path to version
# Usage: ./scripts/update-mem-dependency.sh [version]
# Default version: 0.1.0

set -e

VERSION="${1:-0.1.0}"
CARGO_TOML="Cargo.toml"

echo "🔄 Updating reasonkit-mem dependency to version ${VERSION}..."

# Check if Cargo.toml exists
if [ ! -f "$CARGO_TOML" ]; then
    echo "❌ Error: $CARGO_TOML not found"
    exit 1
fi

# Check if reasonkit-mem is available on crates.io
echo "🔍 Verifying reasonkit-mem v${VERSION} is available on crates.io..."
if ! cargo search reasonkit-mem --limit 1 | grep -q "reasonkit-mem = \"${VERSION}\""; then
    echo "⚠️  Warning: reasonkit-mem v${VERSION} not found on crates.io"
    echo "   Waiting 30 seconds for index update..."
    sleep 30
    
    if ! cargo search reasonkit-mem --limit 1 | grep -q "reasonkit-mem = \"${VERSION}\""; then
        echo "❌ Error: reasonkit-mem v${VERSION} still not available"
        echo "   Please verify publication and wait for index update (5-10 minutes)"
        exit 1
    fi
fi

echo "✅ reasonkit-mem v${VERSION} found on crates.io"

# Backup Cargo.toml
cp "$CARGO_TOML" "${CARGO_TOML}.bak"
echo "📋 Backup created: ${CARGO_TOML}.bak"

# Update the dependency
if sed -i "s|reasonkit-mem = { path = \"../reasonkit-mem\", optional = true }|reasonkit-mem = { version = \"${VERSION}\", optional = true }|" "$CARGO_TOML"; then
    echo "✅ Updated Cargo.toml"
else
    echo "❌ Error: Failed to update Cargo.toml"
    mv "${CARGO_TOML}.bak" "$CARGO_TOML"
    exit 1
fi

# Verify the change
if grep -q "reasonkit-mem = { version = \"${VERSION}\", optional = true }" "$CARGO_TOML"; then
    echo "✅ Verification: Dependency updated successfully"
else
    echo "❌ Error: Verification failed"
    mv "${CARGO_TOML}.bak" "$CARGO_TOML"
    exit 1
fi

# Update Cargo.lock
echo "🔄 Updating Cargo.lock..."
cargo update -p reasonkit-mem

# Verify build
echo "🔨 Verifying build..."
if cargo build --release; then
    echo "✅ Build successful"
else
    echo "❌ Error: Build failed"
    mv "${CARGO_TOML}.bak" "$CARGO_TOML"
    exit 1
fi

# Verify tests
echo "🧪 Running tests..."
if cargo test --all-features; then
    echo "✅ Tests passed"
else
    echo "❌ Error: Tests failed"
    mv "${CARGO_TOML}.bak" "$CARGO_TOML"
    exit 1
fi

# Dry-run publication
echo "📦 Testing publication (dry-run)..."
if cargo publish --dry-run; then
    echo "✅ Publication dry-run successful"
    echo ""
    echo "🎉 Ready to publish reasonkit-core!"
    echo "   Run: cargo publish"
else
    echo "❌ Error: Publication dry-run failed"
    mv "${CARGO_TOML}.bak" "$CARGO_TOML"
    exit 1
fi

echo ""
echo "✅ All checks passed! Cargo.toml updated successfully."
echo "   Backup saved at: ${CARGO_TOML}.bak"

