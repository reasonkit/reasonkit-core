# ReasonKit Core - Production Docker Build
# Multi-stage build optimized for minimal image size and security
# Supports: linux/amd64, linux/arm64
#
# Build: docker build -t reasonkit/core:latest .
# Run:   docker run --rm -p 8080:8080 reasonkit/core:latest serve
#
# Targets:
#   - runtime (default): Minimal production image (~50MB)
#   - development: Full toolchain with hot-reload

# =============================================================================
# Stage 1: Chef - Dependency caching base
# =============================================================================
FROM rust:1.83-slim-bookworm AS chef

# Install cargo-chef for dependency caching
RUN cargo install cargo-chef --locked

WORKDIR /build

# =============================================================================
# Stage 2: Planner - Generate dependency recipe
# =============================================================================
FROM chef AS planner

# Copy only files needed for dependency resolution
COPY Cargo.toml Cargo.lock ./
COPY src/lib.rs src/lib.rs

# Generate recipe.json (dependency manifest)
RUN cargo chef prepare --recipe-path recipe.json

# =============================================================================
# Stage 3: Builder - Compile with cached dependencies
# =============================================================================
FROM chef AS builder

# Build arguments for feature selection
ARG FEATURES="cli"
ARG PROFILE="release"

# Install build dependencies (minimal set)
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy recipe and build dependencies first (cached layer)
COPY --from=planner /build/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json

# Copy source code
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY benches ./benches
COPY config ./config
COPY protocols ./protocols
COPY schemas ./schemas

# Build the application
RUN cargo build --release --locked --features "${FEATURES}" \
    && strip target/release/rk-core 2>/dev/null || true \
    && strip target/release/rk-compare 2>/dev/null || true \
    && strip target/release/rk-bench 2>/dev/null || true

# =============================================================================
# Stage 4: Runtime - Minimal production image
# =============================================================================
FROM debian:bookworm-slim AS runtime

# Build-time metadata
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION="0.1.0"

# OCI Image Labels
LABEL org.opencontainers.image.title="ReasonKit Core" \
      org.opencontainers.image.description="Rust-first RAG/Knowledge Base Engine for AI Reasoning" \
      org.opencontainers.image.url="https://reasonkit.sh" \
      org.opencontainers.image.documentation="https://docs.reasonkit.sh" \
      org.opencontainers.image.source="https://github.com/reasonkit/reasonkit-core" \
      org.opencontainers.image.vendor="ReasonKit" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}"

# Install runtime dependencies only (minimal footprint)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /var/cache/apt/archives/*

# Create non-root user with explicit UID/GID for security
RUN groupadd --gid 1000 reasonkit \
    && useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home reasonkit

# Create application directories with proper permissions
RUN mkdir -p /app/data /app/config /app/output /app/logs \
    && chown -R reasonkit:reasonkit /app

WORKDIR /app

# Copy binaries from builder
COPY --from=builder --chown=reasonkit:reasonkit /build/target/release/rk-core /usr/local/bin/rk-core
COPY --from=builder --chown=reasonkit:reasonkit /build/target/release/rk-compare /usr/local/bin/rk-compare
COPY --from=builder --chown=reasonkit:reasonkit /build/target/release/rk-bench /usr/local/bin/rk-bench

# Copy configuration files
COPY --from=builder --chown=reasonkit:reasonkit /build/config ./config
COPY --from=builder --chown=reasonkit:reasonkit /build/protocols ./protocols
COPY --from=builder --chown=reasonkit:reasonkit /build/schemas ./schemas

# Make binaries executable
RUN chmod 755 /usr/local/bin/rk-*

# Switch to non-root user
USER reasonkit

# Environment configuration
ENV REASONKIT_ENV="production" \
    REASONKIT_LOG_LEVEL="info" \
    REASONKIT_DATA_DIR="/app/data" \
    REASONKIT_CONFIG_DIR="/app/config" \
    REASONKIT_HOST="0.0.0.0" \
    REASONKIT_PORT="8080" \
    RUST_BACKTRACE="1" \
    RUST_LOG="reasonkit=info,tower_http=info"

# Health check (verify binary runs and can respond)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health 2>/dev/null || rk-core --version >/dev/null

# Expose API port
EXPOSE 8080

# Volume mount points for persistence
VOLUME ["/app/data", "/app/config", "/app/logs"]

# Default entrypoint
ENTRYPOINT ["/usr/local/bin/rk-core"]

# Default command (can be overridden)
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]

# =============================================================================
# Stage 5: Development - Full toolchain for local development
# =============================================================================
FROM rust:1.83-bookworm AS development

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    git \
    curl \
    jq \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install Rust development tools
RUN rustup component add rustfmt clippy rust-analyzer \
    && cargo install cargo-watch cargo-nextest cargo-audit cargo-deny sccache --locked

# Create non-root user
RUN groupadd --gid 1000 developer \
    && useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home developer

# Create workspace
RUN mkdir -p /workspace && chown developer:developer /workspace

WORKDIR /workspace

# Switch to non-root user
USER developer

# Development environment
ENV RUST_LOG="debug" \
    RUST_BACKTRACE="full" \
    CARGO_INCREMENTAL="1" \
    RUSTC_WRAPPER="sccache"

# Volume for source code
VOLUME ["/workspace"]

# Default development command (hot-reload)
CMD ["cargo", "watch", "-x", "run -- serve"]
