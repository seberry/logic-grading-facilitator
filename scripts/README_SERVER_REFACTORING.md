# Server Refactoring Scripts

> **Status:** Deprecated as the primary refactoring path on 2026-04-02.
> The active direction is small manual extractions into `app/routes/`, documented in
> `docs/REFACTORING_STATUS.md`.
>
> This file is retained only so the old marker/reordering scripts are documented.
> Do not treat these scripts as the recommended next step.

This directory contains two scripts for improving `app/server.py` readability:

## Scripts

### 1. `organize_server.py` (FULL REORGANIZATION)

**Purpose**: Reorganizes server.py by adding section markers AND reordering code into logical blocks.

**What it does**:
- Adds future work note at the top of the file
- Inserts section markers for:
  - SHARED: Core infrastructure and utilities
  - EXAM WORKFLOW: Setup, grading, exports
  - CLASSWORK WORKFLOW: Setup, review, batch management
  - SHARED: Helper functions
- **Reorders code** into: SHARED_CORE → EXAM_WORKFLOW → CLASSWORK_WORKFLOW → SHARED_HELPERS
- Creates a backup: `app/server.py.backup`

**Usage**:
```bash
python scripts/organize_server.py
```

**⚠️ CAUTION**: This script REORDERS code, which can break dependencies if patterns don't match perfectly.

**Status**: ⚠️ **BROKEN** - Caused syntax errors in server.py. Needs debugging before use.

---

### 2. `add_server_markers.py` (SAFE MARKER-ONLY)

**Purpose**: Adds section markers to server.py WITHOUT reordering the code.

**What it does**:
- Adds future work note at the top of the file
- Inserts section markers at appropriate locations:
  - SHARED: Core infrastructure and utilities
  - EXAM WORKFLOW: Setup, grading, exports
  - CLASSWORK WORKFLOW: Setup, review, batch management
  - SHARED: Helper functions
- **Preserves original code order** (no reordering)
- Creates a backup: `app/server.py.backup`

**Usage**:
```bash
python scripts/add_server_markers.py
```

**Status**: ✅ **SAFER** - Only adds comments, doesn't break code structure.

---

## Section Markers

Both scripts use the same section definitions:

### SHARED: Core infrastructure
- FastAPI app initialization
- Session management (get_session, persist_session)
- Shared helper functions

### EXAM WORKFLOW
- `/setup/*` (setup wizard)
- `/name_review` (name grouping & hand-off)
- `/grade/{unit_id}/page` (per-WorkUnit grading)
- `/problem/{problem_id}/strip` (strip grading with segmentation)
- `/export`, `/finalize` (exam exports)
- `/debug/fitch/*` (Fitch OCR debug)

### CLASSWORK WORKFLOW
- `/classwork/*` (all classwork endpoints)
- `/classwork/canvas_setup` (Canvas API integration)
- `/classwork/batch/*` (batch operations)
- `/classwork/review` (grading UI)
- Canvas post comments

### SHARED: Helper functions
- CSV parsing utilities
- File I/O helpers
- Gradebook merge utilities

---

## How the Scripts Work

### Classification Logic

Both scripts use regex patterns to classify lines into sections:

```python
SECTIONS = {
    "SHARED_CORE": {
        "patterns": [
            r"^app = FastAPI\(",
            r"^def get_session\(",
            r"^def persist_session\(",
        ]
    },
    # ... more sections
}
```

When a line matches a pattern, it's classified into that section.

### Safety Features

1. **Backup before changes**: Both scripts create `app/server.py.backup`
2. **Skip early lines**: Section markers are only added after line 50 (to avoid messing with imports/constants)
3. **Deduplication**: Marks are only added once per section
4. **Python comments**: Uses `#` not `"""` to avoid syntax errors

---

## Debugging the Full Reorganization

If you want to fix `organize_server.py`, here's what to check:

### Issues That Occurred

1. **Syntax errors in server.py after running**
   - Likely cause: Incorrect string handling in section markers
   - Check: Are all string literals properly escaped?
   - Check: Are multi-line strings properly indented?

2. **Dependencies broken**
   - Likely cause: Functions moved before they're defined/used
   - Check: Do all functions appear after their dependencies?

3. **Import order issues**
   - Likely cause: Code blocks moved before imports that need them
   - Check: Are all imports at the top of the file?

### Debugging Approach

1. Run on a TEST file first:
   ```bash
   cp app/server.py app/server_test.py
   python scripts/organize_server.py
   # Then manually change the script to work on app/server_test.py
   ```

2. Check the diff:
   ```bash
   git diff app/server.py
   ```

3. Test incrementally:
   - Try adding just ONE section marker manually
   - Test that it works
   - Then add the next one

4. Focus on the `reorder_sections()` function:
   - This is where reordering happens
   - The `add_section_markers()` function is likely safer
   - Consider using just `add_server_markers.py` approach

---

## Future Refactoring Plan

The full reorganization should eventually split server.py into focused modules:

### Proposed Module Structure

```
app/
├── server_core.py       # FastAPI app init, session management
├── server_exam.py        # Exam workflow routes only
├── server_classwork.py   # Classwork workflow routes only
├── server_utils.py      # Shared utilities and helpers
└── server.py             # Main entry point importing above modules
```

### Benefits

1. **Smaller files** - Easier for LLMs and humans to parse
2. **Clear separation** - Exam vs classwork workflows isolated
3. **Shared utilities** - Helpers in one place
4. **Easier testing** - Can test workflows independently

### Migration Strategy

1. Start with marker-only approach (`add_server_markers.py`)
2. Gradually extract sections to separate files
3. Update imports as functions are moved
4. Test incrementally at each step
5. Keep old code until fully verified

---

## Recovery

If either script breaks server.py:

1. Restore from backup:
   ```bash
   cp app/server.py.backup app/server.py
   ```

2. Or use git to rollback:
   ```bash
   git checkout app/server.py
   ```

---

## Questions to Consider

Before using the full reorganization script:

1. **Do all helper functions come after their usage?**
   - Python requires functions to be defined before they're called
   - Check if reordering breaks this

2. **Are there circular dependencies between sections?**
   - Does exam workflow use classwork functions?
   - Does classwork workflow use exam functions?

3. **Are all imports at the top?**
   - Reordering might move code before imports
   - Ensure imports stay at the file beginning

4. **Is the section classification accurate?**
   - Do regex patterns correctly identify all functions?
   - Are any functions misclassified?
