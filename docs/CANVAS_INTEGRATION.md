# Canvas API Integration Documentation

## Overview

The classwork viewer now supports Canvas API integration to streamline the grading workflow. This integration automates fetching gradebook and assignment data from Canvas, and allows posting comments back to Canvas.

## Features

### 1. Canvas Setup Flow

**Access:** `/classwork/canvas_setup`

The Canvas setup page provides a streamlined three-step workflow:

1. **Select Course** - Dropdown loads all accessible Canvas courses via API
2. **Select Assignments** - Choose assignments to grade from a sorted list (with ungraded counts)
3. **Fetch from Canvas & Setup** - Click to automatically download submissions and setup grading

**Assignment Sorting:**
- Assignments with due dates are sorted by date DESC (most recent first)
- Assignments without due dates appear at the bottom
- Shows assignment name, max points, due date, and submission counts

### 2. API Endpoints

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
  4. **Downloads all submission files from Canvas for each selected assignment** (automatic)
  5. Creates ClassworkBatchState with Canvas metadata
- Redirects to `/classwork/review`

#### Auto-Download Feature

**Always enabled for Canvas setup:**
- Downloads all submission files directly from Canvas API for selected assignments
- Supports multiple file attachments per student (PDF, images, etc.)
- Handles text submissions (creates `.txt` files)
- Combines submissions from multiple assignments into one batch
- Creates Canvas-style ZIP structure for compatibility

**File Types Handled:**
- File attachments: PDF, JPG, PNG, DOCX, etc.
- Text submissions: Converted to `.txt` files
- Multiple files per student: All downloaded and included

**Benefits:**
- Eliminates manual download step from Canvas UI
- Ensures you have latest submissions (not stale ZIP)
- Works with multiple assignments in one batch
- Automatically handles Canvas naming conventions
- Simple, streamlined workflow

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

### 3. Data Model Updates

#### ClassworkModeState
New fields added:
- `canvas_course_id: Optional[int]` - Canvas course ID
- `canvas_assignment_id: Optional[int]` - Canvas assignment ID

These fields are populated when using Canvas setup and used for posting comments back to Canvas.

### 4. FERPA Compliance

All Canvas API data is cached to `data/canvas_cache/` directory:

```
data/canvas_cache/
├── courses_cache.json          # Course list (1 hour TTL)
├── {course_id}_assignments_{timestamp}.json  # Assignments (30 min TTL)
└── {course_id}_gradebook_{timestamp}.csv    # Gradebook (no caching)
```

**Key principles:**
- No student data stored in code or long-term memory
- All API responses written to cache files
- Cache files include timestamps for expiration
- Temporary gradebook stored in `data/gradebook_merge/tmp/`

### 5. Configuration

Required environment variables:

```bash
CANVAS_BASE_URL=https://your-canvas-instance.com
CANVAS_API_KEY=your_api_key_here
```

**IMPORTANT:** `CANVAS_BASE_URL` must be the **root domain only**, not a course-specific URL.

- ✅ **Correct:** `https://iu.instructure.com`
- ❌ **Incorrect:** `https://iu.instructure.com/courses/123456`

The base URL should point to your Canvas instance's home page.

**API Key Requirements:**
- Must have OAuth2 scope for course and assignment access
- Should have grading permissions for target courses
- Rate limiting is handled automatically with exponential backoff

### 6. Workflow Comparison

#### Manual Workflow (Original)
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

#### Canvas API Workflow (Streamlined)
1. Select course from dropdown (auto-fetched)
2. Select assignments from list (auto-fetched, sorted by due date, with ungraded counts)
3. Click "Fetch from Canvas & Setup"
4. **Submissions downloaded automatically from Canvas** (no manual ZIP needed!)
5. Grade submissions
6. Export merged gradebook CSV
7. Upload CSV back to Canvas (still manual - **INTENTIONAL DESIGN**)
8. Click "Upload Comments to Canvas" button
9. Comments posted automatically via API

**Time Savings:** Eliminates steps 1, 2, 3, 4, 5, 8, 9, and 10 from manual workflow.

**IMPORTANT:** Manual grade upload (step 7) is intentional design, not a missing feature.

**Why Manual Grade Upload?**
- **Safety First:** Canvas has no grade conflict API. Automatic uploads could overwrite existing grades.
- **Review Opportunity:** Manual upload lets you review the merged CSV before committing.
- **Error Recovery:** If grades are incorrect, you can fix them in the CSV before uploading.
- **Consistency:** Keeps grading workflow consistent with existing Canvas procedures.

**The app generates a correctly-labeled CSV** that Canvas will accept. You simply:
1. Download the merged CSV from the export page
2. Review it in a spreadsheet
3. Upload it to Canvas using the standard gradebook import feature

This approach balances automation with safety and review capabilities.

### 7. Error Handling

#### Canvas API Errors
- If Canvas API fails, falls back to cached data (if available)
- Shows error messages in UI
- Allows retry with "Refresh" options

#### Rate Limiting
- Automatically detects HTTP 429 errors
- Implements exponential backoff
- Respects Canvas rate limits

#### Comment Upload Errors
- Shows per-assignment success/failure summary
- Lists students with failed uploads
- Allows retry without re-grading

### 8. Future Enhancements

Potential future improvements:
- **Grade Upload:** Add option to post grades via API (with Canvas conflict handling)
- **Real-time Sync:** Refresh roster and assignments periodically
- **Assignment Groups:** Filter assignments by group (quizzes, homework, etc.)
- **Recent Assignments First:** Highlight assignments due in the last 7 days
- **Progress Indicators:** Show download progress for large submissions batches
- **Caching Downloaded Files:** Cache downloaded submissions to avoid re-downloading

### 9. Testing Checklist

Before using Canvas integration in production:

- [ ] Set CANVAS_BASE_URL environment variable
- [ ] Set CANVAS_API_KEY environment variable
- [ ] Verify API key has course read permissions
- [ ] Verify API key has comment write permissions
- [ ] Test course list loading
- [ ] Test assignment list loading
- [ ] Test setup with real submissions ZIP
- [ ] Test comment upload to Canvas
- [ ] Verify comments appear in Canvas SpeedGrader
- [ ] Verify FERPA compliance (all data in data/ directory)

### 10. Troubleshooting

**Issue:** "Canvas API credentials not configured"
- **Solution:** Set CANVAS_BASE_URL and CANVAS_API_KEY environment variables

**Issue:** "No assignments found for this course"
- **Solution:** Verify course has published assignments, check API key permissions

**Issue:** Comment upload fails for some students
- **Solution:** Check if student IDs in your roster match Canvas SIS Login IDs

**Issue:** Rate limit errors during setup
- **Solution:** Wait a few minutes and retry; caching will reduce API calls

**Issue:** Assignments not sorted correctly
- **Solution:** Check assignment due dates in Canvas; assignments with no due date appear at bottom

## Development Notes

- Canvas client uses `httpx` for HTTP requests
- Cache files use ISO 8601 timestamps
- All Canvas data is optional; system falls back to manual workflow if API fails
- Comment upload uses Canvas bulk submission API endpoint
- Gradebook CSV format expected: SIS Login ID column for student IDs