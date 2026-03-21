#!/usr/bin/env python3
"""
Script to add section markers to server.py for better readability.

This script:
1. Adds clear section comment blocks to delineate workflows
2. Adds a future work note at the top
3. Creates a backup before making changes
4. DOES NOT reorder code (preserves original order and dependencies)
"""

import re
from pathlib import Path
from typing import List
import shutil
import sys


# Define section markers and their associated functions/routes
SECTIONS = {
    "SHARED_CORE": {
        "marker": "# ==============================================================================\n# SHARED: Core infrastructure and utilities\n# ==============================================================================\n# - FastAPI app initialization\n# - Session management (get_session, persist_session)\n# - Shared helper functions\n",
        "patterns": [
            r"^app = FastAPI\(",
            r"^def get_session\(",
            r"^def persist_session\(",
        ]
    },
    "EXAM_WORKFLOW": {
        "marker": "# ==============================================================================\n# EXAM WORKFLOW: Setup, grading, exports\n# ==============================================================================\n# - /setup/* (setup wizard)\n# - /name_review (name grouping & hand-off)\n# - /grade/{unit_id}/page (per-WorkUnit grading)\n# - /problem/{problem_id}/strip (strip grading with segmentation)\n# - /export, /finalize (exam exports)\n# - /debug/fitch/* (Fitch OCR debug)",
        "patterns": [
            r"^@app\.get\(\"/setup",
            r"^@app\.post\(\"/setup",
            r"^@app\.get\(\"/name_review",
            r"^@app\.post\(\"/name_review",
            r"^@app\.get\(\"/grade/\{unit_id\}",
            r"^@app\.post\(\"/grade/\{unit_id\}",
            r"^@app\.get\(\"/problem/\{problem_id\}",
            r"^@app\.post\(\"/problem/\{problem_id\}",
            r"^@app\.get\(\"/export",
            r"^@app\.post\(\"/finalize",
            r"^@app\.get\(\"/debug/fitch",
            r"^@app\.post\(\"/debug/fitch",
            r"^@app\.get\(\"/finish",
        ]
    },
    "CLASSWORK_WORKFLOW": {
        "marker": "# ==============================================================================\n# CLASSWORK WORKFLOW: Setup, review, batch management\n# ==============================================================================\n# - /classwork/* (all classwork endpoints)\n# - /classwork/canvas_setup (Canvas API integration)\n# - /classwork/batch/* (batch operations)\n# - /classwork/review (grading UI)\n# - Canvas post comments",
        "patterns": [
            r"^@app\.get\(\"/classwork",
            r"^@app\.post\(\"/classwork",
            r"^@app\.get\(\"/flashcards",  # Flashcards is related to classwork-like workflow
            r"^@app\.post\(\"/flashcards",
        ]
    },
    "SHARED_HELPERS": {
        "marker": "# ==============================================================================\n# SHARED: Helper functions\n# ==============================================================================\n# - CSV parsing utilities\n# - File I/O helpers\n# - Gradebook merge utilities",
        "patterns": [
            r"^def _validate_roster_csv\(",
            r"^def _normalize_headers\(",
            r"^def _extract_canvas_roster\(",
            r"^def _build_total_scores_by_student\(",
            r"^def _required_total_max\(",
            r"^def _score_summary_for_work_unit\(",
            r"^def _parse_gradebook_csv\(",
            r"^def _candidate_assignment_columns\(",
            r"^def _assignment_column_stats\(",
            r"^def _classwork_",
            r"^def _unique_path\(",
            r"^def _save_image_upload\(",
            r"^def _save_pdf_pages\(",
            r"^def _ingest_zip_bytes\(",
            r"^def _normalize_canvas_key\(",
            r"^def _filename_prefix_key\(",
            r"^def _build_roster_prefix_index\(",
            r"^def slugify_name\(",
            r"^def _render_feedback",
        ]
    }
}


def classify_line(line: str) -> str:
    """Classify a line into one of the sections."""
    for section_name, section_info in SECTIONS.items():
        for pattern in section_info["patterns"]:
            if re.search(pattern, line):
                return section_name
    return "UNKNOWN"


def add_section_markers(lines: List[str]) -> List[str]:
    """Add section markers at appropriate locations (without reordering)."""
    result = []
    current_section = None
    sections_seen = set()
    
    for i, line in enumerate(lines):
        # Check if this line starts a new section
        new_section = classify_line(line)
        
        # Add section marker when we encounter a new section (and haven't seen it yet)
        if new_section != "UNKNOWN" and new_section != current_section:
            if new_section in SECTIONS and new_section not in sections_seen:
                marker = SECTIONS[new_section]["marker"]
                # Add marker before the line (but after imports/constants)
                if i > 50:  # Skip very early lines (imports, constants)
                    result.append("")
                    result.append(marker)
                    sections_seen.add(new_section)
                current_section = new_section
        
        result.append(line)
    
    return result


def add_future_work_note(lines: List[str]) -> List[str]:
    """Add a note about future refactoring at the top of the file."""
    result = []
    added = False
    
    for i, line in enumerate(lines):
        # Find the first import line
        if not added and line.strip() and not line.startswith("#") and not line.startswith('"""'):
            # Insert future work note before first import
            note = """
# TODO: Consider splitting this file into focused modules:
#   - app/server_core.py (session management, FastAPI initialization)
#   - app/server_exam.py (exam workflow routes)
#   - app/server_classwork.py (classwork workflow routes)
#   - app/server_utils.py (shared utilities)
#
# The file is currently organized into clear sections for readability.
# See docs/TODO.md or LLM_README.yaml for context.

"""
            result.append(note)
            result.append("")
            added = True
        result.append(line)
    
    return result


def main():
    """Main function to add markers to server.py"""
    server_path = Path(__file__).parent.parent / "app" / "server.py"
    
    if not server_path.exists():
        print(f"Error: server.py not found at {server_path}")
        sys.exit(1)
    
    print(f"Reading {server_path}...")
    with open(server_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"File has {len(lines)} lines")
    
    # Step 1: Add future work note
    print("Adding future work note...")
    lines = add_future_work_note(lines)
    
    # Step 2: Add section markers (NO reordering)
    print("Adding section markers...")
    lines = add_section_markers(lines)
    
    # Create backup
    backup_path = server_path.with_suffix('.py.backup')
    print(f"Creating backup at {backup_path}...")
    shutil.copy2(server_path, backup_path)
    
    # Write updated file
    print(f"Writing updated file to {server_path}...")
    with open(server_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"\n✅ Successfully added markers to server.py")
    print(f"   - Original file backed up to: {backup_path}")
    print(f"   - Added future work note")
    print(f"   - Added section markers")
    print(f"   - Code order preserved (no reordering)")
    print(f"\nPlease test the server to ensure everything still works.")


if __name__ == "__main__":
    main()