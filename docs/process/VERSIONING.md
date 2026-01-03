# Versioning Policy

ReasonKit follows [Semantic Versioning 2.0.0](https://semver.org/).

## Version Format

`MAJOR.MINOR.PATCH` (e.g., `1.2.3`)

- **MAJOR**: Incompatible API changes.
- **MINOR**: Functionality added in a backward-compatible manner.
- **PATCH**: Backward-compatible bug fixes.

## Public API Definition

The following interfaces constitute the public API. Breaking changes to these will trigger a MAJOR version bump:

1.  **CLI Commands:** All subcommands and flags of `rk`.
2.  **ThinkTool Protocols:** The TOML/YAML schema for protocol definitions.
3.  **Rust Crate API:** Public modules in `reasonkit::thinktool` and `reasonkit::client`.
4.  **HTTP API:** Endpoint signatures and JSON schemas (once stable in v1.0).

**Internal APIs:**
Modules marked as `experimental` or `internal` are NOT covered by semantic versioning guarantees.

## Release Cadence

- **Patch Releases:** As needed for bug fixes (weekly/bi-weekly).
- **Minor Releases:** Quarterly feature drops (Q1, Q2, Q3, Q4).
- **Major Releases:** Annually or when significant architectural shifts occur.

## Deprecation Policy

We strive to avoid breaking changes. When they are necessary:

1.  **Announcement:** Deprecation warning added in version `X.Y.0`.
2.  **Warning Period:** Feature remains available but emits warnings for at least 2 minor versions (e.g., until `X.Y+2.0`).
3.  **Removal:** Feature removed in the next Major version.

## Protocol Versioning

ThinkTool protocols use their own versioning (`thinktools.yaml`).

- The engine supports executing older protocol versions if the schema is compatible.
- Breaking schema changes require a parser update and potentially a MAJOR engine update.
