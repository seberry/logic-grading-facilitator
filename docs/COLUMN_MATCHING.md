# Column Matching for Canvas Integration

## Overview

This document describes the automatic column matching functionality that pairs Canvas assignment names with gradebook CSV columns during the Canvas workflow setup.

## Purpose

When users select assignments from Canvas to grade, the system needs to know which gradebook column to write the final grades to. Previously, users had to manually select this for each assignment. Now, the system can automatically match Canvas assignment names to gradebook columns based on similarity.

## How It Works

### 1. Name Normalization

The `_normalize_for_matching()` function standardizes both Canvas assignment names and gradebook column names for comparison:

- Converts to lowercase
- **Removes parenthetical numbers** (Canvas assignment IDs in gradebook columns)
- Removes special characters (except spaces, hyphens, underscores)
- Collapses multiple spaces into single spaces
- Trims leading/trailing whitespace

**Examples:**
- "Worksheet 1" → "worksheet 1"
- "Quiz-#1" → "quiz-1"
- "Assignment #3" → "assignment 3"
- "Worksheet - Proofs" → "worksheet - proofs"
- **"Wed Attendance (18715846)" → "wed attendance"** (ID removed)
- **"Exam 1 (18697401)" → "exam 1"** (ID removed)

### 2. Column Matching

The `_find_matching_column()` function attempts to find a matching gradebook column:

1. Normalizes the Canvas assignment name
2. Filters out non-assignment columns (Student, ID, Score, Grade columns)
3. Compares normalized Canvas name to each candidate column
4. Returns the first exact match (after normalization)
5. Returns `None` if no match is found

### 3. Integration with Canvas Setup

In `/classwork/canvas_setup` endpoint:

1. User uploads gradebook CSV
2. System parses CSV to extract column names
3. User selects Canvas assignments
4. For each assignment:
   - Downloads submissions via Canvas API
   - **Automatically attempts to match** assignment name to gradebook column
   - If match found: Sets `gradebook_column` automatically
   - If no match: Leaves `gradebook_column` unset (user must manually select later)
5. Creates ClassworkModeState with matched column

## Benefits

1. **Time Savings**: Automatically matches common naming patterns
2. **Accuracy**: Uses case-insensitive, whitespace-tolerant matching
3. **Graceful Fallback**: Manual selection still available if auto-match fails
4. **Informative**: Prints debug messages to console showing match results

## Console Output

When the system runs, you'll see messages like:

```
[COLUMN MATCH] Auto-matched 'Worksheet 1' to gradebook column: 'Worksheet 1'
[COLUMN MATCH WARNING] No match found for assignment 'Quiz 9'. Manual column selection required.
```

## Test Coverage

The implementation includes comprehensive tests (`test_column_matching_simple.py`):

- **Normalization tests**: 6/6 passing
  - Basic strings
  - Special characters
  - Edge cases (empty strings)

- **Matching tests**: 7/7 passing
  - Exact matches with different formatting
  - Nonexistent assignments (returns None)
  - System column exclusion

Run tests with:
```bash
python test_column_matching_simple.py
```

## Limitations

1. **Exact Match Only**: Currently only matches if normalized names are identical
   - Future enhancement: Fuzzy matching for similar names
2. **No User Feedback**: Matches happen silently in console
   - Future enhancement: Show match status in UI
3. **Manual Override Required**: If match fails, user must go to batch setup to manually assign
   - Future enhancement: Show unmatched assignments with dropdown selection

## Example Workflow

### Before (Manual)
1. Select "Worksheet 1" from Canvas
2. Download submissions
3. Go to `/classwork/batch/finalize`
4. See "columns are unset" error
5. Go to `/classwork/batch/set_column`
6. Manually select "Worksheet 1" column for each assignment

### After (Automatic)
1. Select "Worksheet 1" from Canvas
2. Download submissions → **Automatically matches to "Worksheet 1" column**
3. Go to `/classwork/review`
4. Grade and finalize immediately!
5. System writes grades to correct column automatically

## Future Enhancements

1. **Fuzzy Matching**: Use Levenshtein distance or similar for near-matches
2. **UI Feedback**: Show match status in the Canvas setup page
3. **Bulk Manual Override**: Select columns for all unmatched assignments at once
4. **Match Confidence**: Rate matches by confidence level (exact, fuzzy, none)
5. **Interactive Matching**: Allow users to approve/override auto-matches before proceeding

## Files Modified

- `app/server.py`: Added `_normalize_for_matching()` and `_find_matching_column()` functions
- `app/server.py`: Modified `_classwork_build_mode_from_downloads()` to auto-match columns
- `test_column_matching_simple.py`: Comprehensive test suite

## Related Documentation

- [CANVAS_INTEGRATION.md](CANVAS_INTEGRATION.md) - Overall Canvas API integration
- [CANVAS_WORKFLOW_REDESIGN_SPEC.md](CANVAS_WORKFLOW_REDESIGN_SPEC.md) - Workflow design specification