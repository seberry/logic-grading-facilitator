"""
Test script for column matching functionality.
"""
import sys
import re
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from server import _normalize_for_matching, _find_matching_column


def test_normalization():
    """Test the normalization function."""
    print("Testing normalization...")
    
    test_cases = [
        ("Worksheet 1", "worksheet 1"),
        ("Quiz-#1", "quiz 1"),
        ("Proof Assignment", "proof assignment"),
        ("Assignment #3", "assignment 3"),
        ("Worksheet - Proofs", "worksheet proofs"),
        ("Midterm Exam", "midterm exam"),
    ]
    
    for original, expected in test_cases:
        result = _normalize_for_matching(original)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{original}' -> '{result}' (expected: '{expected}')")
    
    print()


def test_column_matching():
    """Test the column matching function."""
    print("Testing column matching...")
    
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
    
    for canvas_name, expected_match in test_cases:
        result = _find_matching_column(
            canvas_assignment_name=canvas_name,
            gradebook_fieldnames=gradebook_columns,
            id_col="SIS Login ID",
        )
        status = "✓" if result == expected_match else "✗"
        print(f"  {status} Canvas: '{canvas_name}' -> Matched: '{result}' (expected: '{expected_match}')")
    
    print()


def test_edge_cases():
    """Test edge cases."""
    print("Testing edge cases...")
    
    # Empty string
    result = _normalize_for_matching("")
    print(f"  Empty string: '{result}' (should be empty)")
    
    # Special characters
    test_cases = [
        "Quiz #1 - (Extra Credit)",
        "Assignment: Midterm",
        "Worksheet #2: Proofs & Deductions",
    ]
    
    for test in test_cases:
        result = _normalize_for_matching(test)
        print(f"  '{test}' -> '{result}'")
    
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Column Matching Tests")
    print("=" * 60)
    print()
    
    test_normalization()
    test_column_matching()
    test_edge_cases()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)