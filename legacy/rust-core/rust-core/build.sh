#!/bin/bash
# Build Rust core and install Python bindings

set -e

echo "Building ReasonKit Rust Core..."

# Build release
cargo build --release

# Install with maturin (if available)
if command -v maturin &> /dev/null; then
    echo "Installing Python bindings with maturin..."
    maturin develop --release
else
    echo ""
    echo "To install Python bindings, install maturin:"
    echo "  pip install maturin"
    echo "  maturin develop --release"
    echo ""
    echo "Rust library built at: target/release/libreasonkit_core.so"
fi

echo "Done!"
