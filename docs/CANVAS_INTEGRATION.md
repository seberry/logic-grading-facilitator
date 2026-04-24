# Canvas API Integration Documentation

> **Consolidated reference for all Canvas-related features.** This document covers
> setup, submission download, comment posting, user ID mapping, column matching,
> and known issues. The forward-looking redesign spec lives separately in
> [CANVAS_WORKFLOW_REDESIGN_SPEC.md](CANVAS_WORKFLOW_REDESIGN_SPEC.md).

---

## 1. Overview

The classwork viewer supports Canvas API integration to streamline the grading workflow. This integration automates fetching gradebook and assignment data from Canvas, downloading submissions, and posting comments back to Canvas.

**Two paths exist:**

| Path | Status | Description |
|------|--------|-------------|
| **Manual** | Stable | Upload gradebook CSV + submission ZIPs by hand |
| **Canvas API** | Live (slow for large batches) | Auto-fetch everything from Canvas; see §5 for known performance issue |

---

## 2. Configuration

Required environment variables:

```bash
CANVAS_BASE_URL=https://your-canvas-instance.com
CANVAS_API_KEY=your_api_key_here
```

**IMPORTANT:** `CANVAS_BASE_URL` must be the **root domain only**, not a course-specific URL.

- ✅ **Correct:** `https://iu.instructure.com`
- ❌ **Incorrect:** `https://iu.instructure.com/courses/123456`

**API Key Requirements:**
- Must have OAuth2 scope for course and assignment access
- Should have grading permissions for target courses
- Rate limiting is handled automatically with exponential backoff

---

## 3. Canvas Setup Flow

**Access:** `/classwork/canvas_setup`

The Canvas setup page provides a streamlined three-step workflow:

1. **Select Course** — Dropdown loads all accessible Canvas courses via API
2. **Select Assignments** — Choose assignments to grade from a sorted list (with ungraded counts)
3. **Fetch from Canvas & Setup** — Click to automatically download submissions and setup grading

### 3.1 API Endpoints

#### GET `/classwork/canvas_courses`
- Fetches list of accessible courses from Canvas
- Returns: `{"courses": [{"id": 123, "name": "Course Name", "course_code": "CODE"}, ...]}`
- Caches results for 1 hour

#### GET `/classwork/canvas_assignments?course_id={id}`
- Fetches assignments for a specific Canvas course
- Filters to published assignments with relevant submission types
- Returns: `{"course_id": 123, "assignments": [{"id": 456, "name": "Quiz 1", "max_points": 10, "due_date": "2026-03-20T..."}, ...]}`
- Caches results for 30 minutes
- Sorts assignments by due date DESC

#### POST `/classwork/canvas_setup`
- Sets up classwork using Canvas API data
- Parameters:
  - `course_id`: Canvas course ID
  - `assignment_ids[]`: List of assignment IDs to grade
- Actions:
  1. Fetches gradebook CSV from Canvas
  2. Parses gradebook and builds roster
  3. Fetches assignment metadata (max points)
  4. Downloads all submission files from Canvas for each selected assignment (automatic)
  5. Creates ClassworkBatchState with Canvas metadata
- Redirects to `/classwork/review`

### 3.2 Auto-Download Feature

**Always enabled for Canvas setup:**
- Downloads all submission files directly from Canvas API for selected assignments
- Supports multiple file attachments per student (PDF, images, etc.)
- Handles text submissions (creates `.txt` files)
- Combines submissions from multiple assignments into one batch

**File types handled:**
- File attachments: PDF, JPG, PNG, DOCX, etc.
- Text submissions: Converted to `.txt` files
- Multiple files per student: All downloaded and included

**Filename handling:**
- The app preserves the exact Canvas filename as `original_filename` for display and comment ZIP exports.
- It stores the local file under a separate safe `stored_filename` for disk paths and `/classwork_files/...` preview URLs.
- This avoids viewer failures on Canvas filenames containing URL-like escapes such as `%28` / `%29`, without breaking manual ZIP-based reupload workflows.

### 3.3 Assignment Sorting
- Assignments with due dates are sorted by date DESC (most recent first)
- Assignments without due dates appear at the bottom
- Shows assignment name, max points, due date, and submission counts

---

## 4. Comment Posting

### 4.1 Endpoint

#### POST `/classwork/canvas_post_comments`
- Posts text comments to Canvas assignments
- Parameters:
  - `export_id`: Export ID from batch finalization
  - `overwrite`: Boolean flag (for future use)
- Actions:
  1. Loads session state from export
  2. Builds comments dict from session
  3. Bulk posts comments to Canvas via API
  4. Returns success/failure summary per assignment
- Renders results in `classwork_batch_done.html`

### 4.2 Canvas API Endpoint Used

```
PUT /api/v1/courses/{course_id}/assignments/{assignment_id}/submissions/{canvas_user_id}
```

Request body:
```json
{
  "comment": {
    "text_comment": "Comment text here"
  }
}
```

### 4.3 User Workflow

1. User completes classwork batch grading
2. User clicks "Finalize Batch" on the review page
3. On the done page, user sees:
   - Option A: "Post Comments to Canvas" button
   - Option B: Download ZIP files for manual upload (fallback)
4. User clicks "Post Comments to Canvas"
5. System posts all comments to Canvas via API
6. Page displays results:
   - Success/failure counts
   - Per-assignment status
   - Error details if any failures

### 4.4 Safety Features
- **No duplicate posting:** Form is hidden after posting (prevents accidental re-posts)
- **Overwrite protection:** Checkbox to allow overwriting is unchecked by default
- **Manual fallback:** ZIP download option remains available
- **Clear feedback:** Detailed error messages if posting fails

### 4.5 Rate Limiting
- 0.5 second delay between student comments
- Respects Canvas API rate limits
- Prevents API throttling

### 4.6 Important Notes
- Comments are **additive** — each call adds a new comment (Canvas has no "replace comment" endpoint)
- The overwrite checkbox is advisory only (cannot actually replace)
- Best practice: Grade each assignment once to avoid comment accumulation

---

## 5. User ID Mapping

### 5.1 The Problem

When posting comments to Canvas via the API, the system needs Canvas user IDs (internal identifiers), not SIS login IDs (the external student identifiers used in gradebook CSVs).

### 5.2 The Solution

The system builds and maintains a mapping between SIS login IDs and Canvas user IDs:

1. **Collection Phase:** When downloading submissions from Canvas, extract Canvas user IDs
2. **Storage Phase:** Store the mapping in `ClassworkBatchState.canvas_user_ids`
3. **Usage Phase:** Look up Canvas user IDs when posting comments
4. **Cleanup Phase:** Clear mapping when batch is closed

### 5.3 Data Model

```python
class ClassworkBatchState(BaseModel):
    # ... existing fields ...
    canvas_user_ids: Dict[str, int] = {}  # SIS login ID → Canvas user ID
```

The mapping is populated during `download_assignment_submissions()`, which returns:
```python
# Returns: (Dict[str, List[Path]], Dict[str, int])
student_files, canvas_user_ids = client.download_assignment_submissions(...)
```

### 5.4 Comment Posting with User IDs

```python
for entry in mode.entries:
    if entry.comment and entry.comment.strip():
        canvas_user_id = batch.canvas_user_ids.get(entry.student_id)
        if canvas_user_id:
            comments_dict[canvas_user_id] = {"text": entry.comment.strip()}
```

---

## 6. Column Matching

### 6.1 Purpose

When users select assignments from Canvas, the system needs to know which gradebook column to write grades to. The auto-matching system pairs Canvas assignment names with gradebook CSV columns.

### 6.2 How It Works

**Name normalization** (`_normalize_for_matching()`) standardizes both Canvas assignment names and gradebook column names:
- Converts to lowercase
- Removes parenthetical numbers (Canvas assignment IDs in gradebook columns)
- Removes special characters (except spaces, hyphens, underscores)
- Collapses multiple spaces into single spaces
- Trims whitespace

**Examples:**
- `"Worksheet 1"` → `"worksheet 1"`
- `"Wed Attendance (18715846)"` → `"wed attendance"` (ID removed)
- `"Exam 1 (18697401)"` → `"exam 1"` (ID removed)

**Matching** (`_find_matching_column()`):
1. Normalizes the Canvas assignment name
2. Filters out non-assignment columns (Student, ID, Score, Grade columns)
3. Returns first exact match after normalization, or `None`

### 6.3 Console Output

```
[COLUMN MATCH] Auto-matched 'Worksheet 1' to gradebook column: 'Worksheet 1'
[COLUMN MATCH WARNING] No match found for assignment 'Quiz 9'. Manual column selection required.
```

### 6.4 Limitations
- Exact match only (no fuzzy matching yet)
- Matches happen silently in console (no UI feedback yet)
- Manual override via `/classwork/batch/set_column` if auto-match fails

---

## 7. Data Model

### ClassworkModeState — Canvas fields

```python
class ClassworkModeState(BaseModel):
    # ... other fields ...
    canvas_course_id: Optional[int]      # Set only if Canvas API path
    canvas_assignment_id: Optional[int]  # Set only if Canvas API path
```

### ClassworkBatchState — Canvas fields

```python
class ClassworkBatchState(BaseModel):
    # ... other fields ...
    canvas_user_ids: Dict[str, int] = {}  # SIS login ID → Canvas user ID
```

---

## 8. FERPA Compliance & Caching

All Canvas API data is cached to `data/canvas_cache/` directory:

```
data/canvas_cache/
├── courses_cache.json                           # Course list (1 hour TTL)
├── {course_id}_assignments_{timestamp}.json     # Assignments (30 min TTL)
└── {course_id}_gradebook_{timestamp}.csv        # Gradebook (no caching)
```

**Key principles:**
- No student data stored in code or long-term memory
- All API responses written to cache files
- Cache files include timestamps for expiration
- Temporary gradebook stored in `data/gradebook_merge/tmp/`

---

## 9. Workflow Comparison

### Manual Workflow
1. Download gradebook CSV from Canvas manually
2. Upload CSV to classwork setup
3. Identify assignment columns manually
4. Enter max points manually
5. Upload submissions ZIP
6. Grade submissions
7. Export merged gradebook CSV
8. Upload CSV back to Canvas manually
9. Export comments ZIP
10. Upload comments ZIP to Canvas manually

### Canvas API Workflow (Streamlined)
1. Select course from dropdown (auto-fetched)
2. Select assignments from list (auto-fetched, sorted by due date, with ungraded counts)
3. Click "Fetch from Canvas & Setup"
4. **Submissions downloaded automatically** (no manual ZIP needed)
5. Grade submissions
6. Export merged gradebook CSV
7. Upload CSV back to Canvas (still manual — **INTENTIONAL DESIGN**)
8. Click "Upload Comments to Canvas" button
9. Comments posted automatically via API

**Time Savings:** Eliminates steps 1, 2, 3, 4, 5, 8, 9, and 10 from manual workflow.

### Why Manual Grade Upload?

**Manual grade upload (step 7) is intentional design, not a missing feature.**

- **Safety First:** Canvas has no grade conflict API. Automatic uploads could overwrite existing grades.
- **Review Opportunity:** Manual upload lets you review the merged CSV before committing.
- **Error Recovery:** If grades are incorrect, you can fix them in the CSV before uploading.
- **Consistency:** Keeps grading workflow consistent with existing Canvas procedures.

---

## 10. Known Issues

### Performance Issue: Synchronous File Downloads

**Current behavior of `POST /classwork/canvas_setup`:**
Downloads ALL files for ALL selected assignments in one synchronous handler. N assignments × M students × K files = potentially hundreds of HTTP requests, all blocking, all in one request.

**Workaround:** Use the manual path (upload ZIPs) for large batches.

**Planned fix:** See [CANVAS_WORKFLOW_REDESIGN_SPEC.md](CANVAS_WORKFLOW_REDESIGN_SPEC.md) for the lazy per-assignment download design.

### Bug Fix: Comments Not Persisting (Fixed 2026-03-22)

**Root cause:** Comments were saved in memory but never written to `session.json`. This caused blank comments in ZIP exports and Canvas posts.

**Fix:** Added `persist_session()` calls after form updates and before finalize.

**How to verify:**
1. Add comments in `/classwork/review`
2. Click "Save" — verify comments persist across page refreshes
3. Click "Finalize" and download ZIP — verify comments are present
4. Click "Post to Canvas" — verify debug output shows comments

### Pagination Bug

`download_assignment_submissions()` uses non-paginated API requests, silently dropping students beyond the first page (~100). Planned fix: switch to `_make_paginated_request`.

---

## 11. Error Handling

### Canvas API Errors
- Falls back to cached data (if available)
- Shows error messages in UI
- Allows retry with "Refresh" options

### Rate Limiting
- Automatically detects HTTP 429 errors
- Implements exponential backoff
- Respects Canvas rate limits

### Comment Upload Errors
- Shows per-assignment success/failure summary
- Lists students with failed uploads
- Allows retry without re-grading

---

## 12. Troubleshooting

| Issue | Solution |
|-------|----------|
| "Canvas API credentials not configured" | Set `CANVAS_BASE_URL` and `CANVAS_API_KEY` environment variables |
| "No assignments found for this course" | Verify course has published assignments, check API key permissions |
| Comment upload fails for some students | Check if student IDs in roster match Canvas SIS Login IDs |
| Rate limit errors during setup | Wait a few minutes and retry; caching will reduce API calls |
| Assignments not sorted correctly | Check assignment due dates in Canvas; assignments with no due date appear at bottom |

---

## 13. Development Notes

- Canvas client uses `httpx` for HTTP requests (synchronous, not async)
- Cache files use ISO 8601 timestamps
- All Canvas data is optional; system falls back to manual workflow if API fails
- Comment upload uses per-student PUT requests with 0.5s delay
- Gradebook CSV format expected: SIS Login ID column for student IDs

---

## 14. Testing Checklist

Before using Canvas integration in production:

- [ ] Set CANVAS_BASE_URL environment variable
- [ ] Set CANVAS_API_KEY environment variable
- [ ] Verify API key has course read permissions
- [ ] Verify API key has comment write permissions
- [ ] Test course list loading
- [ ] Test assignment list loading
- [ ] Test setup with real submissions
- [ ] Test comment upload to Canvas
- [ ] Verify comments appear in Canvas SpeedGrader
- [ ] Verify FERPA compliance (all data in `data/` directory)

Run automated tests:
```bash
python scripts/test_canvas_api.py
python scripts/test_canvas_user_id_mapping.py
python test_column_matching_simple.py
```

---

## 15. Future Enhancements

- **Grade Upload:** Add option to post grades via API (with Canvas conflict handling)
- **Lazy Downloads:** Per-assignment file download instead of all-at-once (see CANVAS_WORKFLOW_REDESIGN_SPEC.md)
- **Real-time Sync:** Refresh roster and assignments periodically
- **Progress Indicators:** Show download progress for large submission batches
- **Fuzzy Column Matching:** Use Levenshtein distance for near-match column names
- **UI Feedback:** Show column match status in the Canvas setup page