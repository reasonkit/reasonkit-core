# ReasonKit Core - Multi-stage Docker Build
# Optimized for production deployment
# Supports multi-arch: linux/amd64, linux/arm64

# ============================================================================
# Stage 1: Build Stage
# ============================================================================
FROM rust:1-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /build

# Copy dependency files first for better caching
COPY Cargo.toml Cargo.lock ./

# Create dummy source to build dependencies
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    echo "pub fn lib() {}" > src/lib.rs

# Build dependencies (this layer will be cached)
RUN cargo build --release --locked && \
    rm -rf src

# Copy actual source code
COPY src ./src
COPY benches ./benches
COPY config ./config
COPY schemas ./schemas

# Build the actual binary
RUN cargo build --release --locked && \
    strip target/release/rk-core

# ============================================================================
# Stage 2: Runtime Stage
# ============================================================================
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash reasonkit

# Create app directory
WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/target/release/rk-core /usr/local/bin/rk-core

# Copy configuration files
COPY --from=builder /build/config ./config
COPY --from=builder /build/schemas ./schemas

# Create data directories
RUN mkdir -p /app/data /app/output && \
    chown -R reasonkit:reasonkit /app

# Switch to non-root user
USER reasonkit

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD rk-core --version || exit 1

# Expose default port (if serving HTTP)
EXPOSE 8080

# Set entrypoint
ENTRYPOINT ["rk-core"]

# Default command (can be overridden)
CMD ["--help"]

# Labels
LABEL org.opencontainers.image.title="ReasonKit Core"
LABEL org.opencontainers.image.description="Rust-first RAG/Knowledge Base Engine"
LABEL org.opencontainers.image.url="https://reasonkit.sh"
LABEL org.opencontainers.image.documentation="https://github.com/reasonkit/reasonkit-core"
LABEL org.opencontainers.image.source="https://github.com/reasonkit/reasonkit-core"
LABEL org.opencontainers.image.vendor="ReasonKit"
LABEL org.opencontainers.image.licenses="Apache-2.0"
