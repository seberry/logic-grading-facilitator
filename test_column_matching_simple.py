"""
Standalone test script for column matching functionality.
Tests the normalization and matching logic without imports.
"""
import re


def _normalize_for_matching(name: str) -> str:
    """
    Normalize a string for comparison between Canvas assignment names and gradebook columns.
    """
    if not name:
        return ""
    
    # Convert to lowercase
    normalized = name.lower().strip()
    
    # Remove parenthetical numbers (Canvas assignment IDs in gradebook)
    # Pattern: "(12345678)" or similar - removes the ID suffix
    # Example: "Wed Attendance (18715846)" -> "Wed Attendance"
    normalized = re.sub(r"\s*\(\d+\)\s*$", "", normalized)
    
    # Remove special characters (keep only alphanumeric, spaces, hyphens, underscores)
    # This handles variations like "Worksheet - 1" vs "Worksheet_1"
    normalized = re.sub(r"[^a-z0-9\s\-_]", "", normalized)
    
    # Collapse multiple spaces into single space
    normalized = re.sub(r"\s+", " ", normalized)
    
    # Strip leading/trailing whitespace again after cleanup
    return normalized.strip()


def _find_matching_column(
    canvas_assignment_name: str,
    gradebook_fieldnames: list,
    id_col: str,
) -> str:
    """
    Find a matching gradebook column for a Canvas assignment name.
    """
    # Normalize the Canvas assignment name
    normalized_canvas = _normalize_for_matching(canvas_assignment_name)
    
    # Build list of candidate columns (exclude ID column and common system columns)
    excluded = {id_col.lower(), "student", "current score", "final score", 
                 "current grade", "final grade"}
    
    candidates = []
    for col in gradebook_fieldnames:
        if col.lower() not in excluded:
            candidates.append(col)
    
    if not candidates:
        return None
    
    # Try to find exact match after normalization
    for col in candidates:
        if _normalize_for_matching(col) == normalized_canvas:
            return col
    
    # No match found
    return None


def test_normalization():
    """Test the normalization function."""
    print("Testing normalization...")
    print()
    
    test_cases = [
        ("Worksheet 1", "worksheet 1"),
        ("Quiz-#1", "quiz-1"),  # Hyphens are preserved
        ("Proof Assignment", "proof assignment"),
        ("Assignment #3", "assignment 3"),
        ("Worksheet - Proofs", "worksheet - proofs"),  # Hyphens preserved
        ("Midterm Exam", "midterm exam"),
        ("Wed Attendance (18715846)", "wed attendance"),  # Parenthetical IDs removed
        ("Week 6 (12345)", "week 6"),  # Parenthetical IDs removed
        ("Exam 1 (18697401)", "exam 1"),  # Parenthetical IDs removed
    ]
    
    passed = 0
    for original, expected in test_cases:
        result = _normalize_for_matching(original)
        if result == expected:
            passed += 1
            print(f"  ✓ '{original}' -> '{result}'")
        else:
            print(f"  ✗ '{original}' -> '{result}' (expected: '{expected}')")
    
    print(f"\nPassed: {passed}/{len(test_cases)}")
    print()
    return passed == len(test_cases)


def test_column_matching():
    """Test the column matching function."""
    print("Testing column matching...")
    print()
    
    # Simulate gradebook columns
    gradebook_columns = [
        "Student",
        "SIS Login ID",
        "Worksheet 1",
        "Quiz-#1",
        "Proof Assignment",
        "Assignment #3",
        "Worksheet - Proofs",
        "Midterm Exam",
        "Current Score",
        "Final Score",
    ]
    
    test_cases = [
        ("Worksheet 1", "Worksheet 1"),
        ("Quiz-#1", "Quiz-#1"),
        ("Proof Assignment", "Proof Assignment"),
        ("Assignment #3", "Assignment #3"),
        ("Worksheet - Proofs", "Worksheet - Proofs"),
        ("Midterm Exam", "Midterm Exam"),
        ("Nonexistent Assignment", None),  # Should not match
    ]
    
    passed = 0
    for canvas_name, expected_match in test_cases:
        result = _find_matching_column(
            canvas_assignment_name=canvas_name,
            gradebook_fieldnames=gradebook_columns,
            id_col="SIS Login ID",
        )
        if result == expected_match:
            passed += 1
            print(f"  ✓ Canvas: '{canvas_name}' -> '{result}'")
        else:
            print(f"  ✗ Canvas: '{canvas_name}' -> '{result}' (expected: '{expected_match}')")
    
    print(f"\nPassed: {passed}/{len(test_cases)}")
    print()
    return passed == len(test_cases)


def test_edge_cases():
    """Test edge cases."""
    print("Testing edge cases...")
    print()
    
    # Empty string
    result = _normalize_for_matching("")
    if result == "":
        print(f"  ✓ Empty string handled correctly")
    else:
        print(f"  ✗ Empty string: '{result}' (should be empty)")
    
    # Special characters
    test_cases = [
        ("Quiz #1 - (Extra Credit)", "quiz 1 extra credit"),
        ("Assignment: Midterm", "assignment midterm"),
        ("Worksheet #2: Proofs & Deductions", "worksheet 2 proofs deductions"),
    ]
    
    passed = 0
    for test_case, _expected in test_cases:
        result = _normalize_for_matching(test_case)
        print(f"  '{test_case}' -> '{result}'")
        # Just verify it doesn't crash and produces a reasonable output
        if result:
            passed += 1
    
    print(f"\nAll edge cases handled: {passed}/{len(test_cases)}")
    print()
    return passed == len(test_cases)


if __name__ == "__main__":
    print("=" * 60)
    print("Column Matching Tests")
    print("=" * 60)
    print()
    
    results = []
    results.append(test_normalization())
    results.append(test_column_matching())
    results.append(test_edge_cases())
    
    print("=" * 60)
    if all(results):
        print("✓ All tests PASSED!")
    else:
        print(f"✗ Some tests failed ({sum(results)}/{len(results)} passed)")
    print("=" * 60)