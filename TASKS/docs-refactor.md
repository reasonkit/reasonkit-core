# Documentation Refactoring & Improvement Plan

## Phase 1: Organization & Structure
- [x] Create subdirectories in `docs/`:
    - `getting-started/`
    - `guides/`
    - `reference/`
    - `architecture/`
    - `concepts/`
    - `meta/` (renamed to `process/`)
- [x] Move existing markdown files into appropriate subdirectories.
- [x] Create `docs/README.md` as the main entry point (Table of Contents).
- [x] Update links in `README.md` (root) to point to new locations.

## Phase 2: Content Quality & Consistency
- [x] Standardize headers and frontmatter (Started with `QUICKSTART.md`).
- [x] Ensure every file has a clear title and introduction (Verified `QUICKSTART.md`).
- [x] Cross-link related documents (Updated links in `QUICKSTART.md`).

## Phase 3: Automation & Tooling
- [x] Add `just docs-check` to verify links (using `lychee` or similar if available, or a script).
- [ ] Add `just docs-serve` if we decide to use `mdbook` or `mkdocs`.
- [ ] Ensure `cargo doc` covers the Rust API documentation effectively.

## Phase 4: Maintenance
- [x] Establish a "Documentation Update" protocol in `CONTRIBUTING.md` (Added `docs/MAINTENANCE.md`).
- [x] Create a template for new documentation files (`docs/templates/DOC_TEMPLATE.md`).
