# Legacy Artifacts (ReasonKit Core)

> **Status:** ✅ ARCHIVED  
> **Policy:** Files preserved in git history only - do not modify or reference in new code

This directory contains archived code from earlier development iterations. These modules are **archived in git history** and should not be actively maintained.

## Archived Modules

- `legacy/arf-platform/` — ARF Rust workspace (archived, replaced by `src/thinktool/`)
- `legacy/rust-core/` — Legacy Rust core with Python bindings (archived, replaced by current `reasonkit-core`)

## Accessing Archived Code

All legacy code is preserved in git history. To access:

```bash
# View commits affecting legacy code
git log --all --full-history -- legacy/

# View specific file at specific commit
git show <commit-hash>:reasonkit-core/legacy/arf-platform/arf-core/src/lib.rs
```

## Maintenance Policy

- ✅ **DO:** Preserve in git history, document replacements
- ❌ **DON'T:** Modify legacy files, reference in new code, delete from git

See [ARCHIVING_PLAN.md](ARCHIVING_PLAN.md) for complete archiving documentation.
