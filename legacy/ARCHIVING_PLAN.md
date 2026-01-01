# Legacy Module Archiving Plan

> **Status:** ARCHIVED  
> **Date:** 2025-12-29  
> **Purpose:** Document archiving of legacy modules to git history only

---

## Overview

The `legacy/` directory contains archived code from earlier development iterations. These modules are **archived in git history** and should not be actively maintained or modified.

**Archiving Strategy:** Files remain in git history for reference but are marked as archived and excluded from active development.

---

## Archived Modules

### 1. ARF Platform (`legacy/arf-platform/`)

**Status:** ✅ ARCHIVED  
**Date Archived:** Pre-2025  
**Reason:** Replaced by current ReasonKit-core architecture

**Contents:**

- `arf-cli/` - CLI implementation
- `arf-core/` - Core runtime engine
- `arf-plugins/` - Plugin system

**Replacement:** Current `src/thinktool/` and `src/protocol/` modules

### 2. Rust Core (`legacy/rust-core/`)

**Status:** ✅ ARCHIVED  
**Date Archived:** Pre-2025  
**Reason:** Replaced by current ReasonKit-core implementation

**Contents:**

- `rust-core/` - Legacy Rust core with Python bindings
- Protocol validation and metrics

**Replacement:** Current `reasonkit-core` crate structure

---

## Archiving Process

### Current Status

✅ **Files are in git history** - All legacy code is committed and preserved  
✅ **Documented in README.md** - Clear indication these are archived  
✅ **Separated from active code** - Located in `legacy/` directory

### Recommendations

1. **Do NOT modify** files in `legacy/` directory
2. **Do NOT reference** legacy code in new implementations
3. **Use git history** to access archived code if needed:

   ```bash
   git log --all --full-history -- legacy/
   git show <commit>:reasonkit-core/legacy/arf-platform/arf-core/src/lib.rs
   ```

### If Removal from Working Tree is Desired

**Option A: Keep in working tree (RECOMMENDED)**

- Files remain visible for reference
- No risk of losing context
- Easy to browse git history

**Option B: Remove from working tree**

```bash
# Remove from working tree but keep in git
git rm -r --cached legacy/
git commit -m "chore: Remove legacy modules from working tree (archived in git history)"

# Files remain in git history, accessible via:
git show HEAD~1:reasonkit-core/legacy/README.md
```

**Current Decision:** **Option A** - Keep in working tree for easy reference

---

## Accessing Archived Code

### View in Git History

```bash
# View all commits affecting legacy code
git log --all --full-history -- legacy/

# View specific file at specific commit
git show <commit-hash>:reasonkit-core/legacy/arf-platform/arf-core/src/lib.rs

# Browse entire legacy directory at specific commit
git ls-tree -r <commit-hash> -- legacy/
```

### Create Archive Branch (Optional)

```bash
# Create dedicated archive branch
git checkout -b archive/legacy-modules
git add legacy/
git commit -m "Archive: Legacy modules preserved in dedicated branch"
git checkout main
```

---

## Maintenance Policy

### What to Do

✅ **Document** when new code replaces legacy functionality  
✅ **Reference** legacy code in git history when needed  
✅ **Preserve** legacy code in git (never force-push to remove)

### What NOT to Do

❌ **Modify** files in `legacy/` directory  
❌ **Reference** legacy code in new implementations  
❌ **Delete** legacy code from git history  
❌ **Use** legacy code as basis for new features

---

## Related Documentation

- [CHANGELOG.md](../CHANGELOG.md) - Version history
- [ARCHITECTURE.md](../ARCHITECTURE.md) - Current architecture
- [README.md](../README.md) - Project overview

---

**Last Updated:** 2025-12-29  
**Status:** ✅ ARCHIVED - Files preserved in git history, documented, and separated from active code
