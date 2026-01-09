# Documentation Maintenance Guide

This guide outlines the process for maintaining and updating the ReasonKit Core documentation.

## Structure

The documentation is organized into the following sections:

- `getting-started/`: Onboarding and installation.
- `guides/`: How-to guides and recipes.
- `reference/`: API and CLI references.
- `architecture/`: Design documents.
- `concepts/`: Explanations of core concepts.
- `thinktools/`: Specific documentation for ThinkTools.
- `process/`: Meta-documentation (release, security, etc.).

## Workflow

1.  **New Features**: When adding a new feature, create a corresponding guide in `guides/` or update the relevant reference in `reference/`.
2.  **Updates**: When changing existing functionality, search for relevant keywords in `docs/` to find pages that need updating.
3.  **Links**: Use relative links for internal navigation. Run `just docs-check` to verify links.

## Style Guide

- **Titles**: Use H1 (`# Title`) for the main title.
- **Code Blocks**: Specify the language for syntax highlighting (e.g., ` ```rust `).
- **Tone**: Professional, concise, and helpful.

## Tools

- `just docs`: Builds and opens the Rust API docs.
- `just docs-check`: Checks for broken links.
- `just docs-tree`: Shows the file structure.
