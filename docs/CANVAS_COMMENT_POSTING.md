# Canvas Comment Posting Feature

## Overview

Added the ability to post text comments directly to Canvas via the Canvas API, eliminating the need to manually download and upload comment files.

## Implementation Details

### Backend (app/server.py)

The endpoint `/classwork/canvas_post_comments` (POST) was already implemented and handles:

1. **Loading the batch session** - Retrieves the active batch from the export ID
2. **Iterating over assignments** - For each assignment in the batch:
   - Checks if Canvas course/assignment mapping exists
   - Builds a dictionary of comments keyed by student_id
   - Calls `canvas_client.bulk_post_comments()` to post to Canvas
3. **Tracking results** - Collects success/failure statistics per assignment
4. **Displaying results** - Renders the done page with posting summary

### Frontend (app/templates/classwork_batch_done.html)

Updated the batch done page with:

1. **Canvas Posting Form** (shown only before posting)
   - Green background to make it visually distinct
   - "Post Comments to Canvas" button
   - Optional "Allow overwriting existing comments" checkbox (disabled by default)
   - Clear explanatory text about the operation

2. **Results Display** (shown after posting)
   - Summary of total successes and failures
   - Per-assignment breakdown with:
     - **Posted** (green): Shows number of comments successfully posted
     - **Skipped** (gray): Shows reason (e.g., "No comments to post")
     - **Failed** (red): Shows error details in expandable dropdown
   - Color-coded status indicators

3. **Fallback Options**
   - Still available: Download merged gradebook CSV
   - Still available: Download comments ZIP files
   - Allows users to manually upload if needed

### Canvas Client (app/canvas_client.py)

The `bulk_post_comments()` method handles:

1. **Rate limiting** - Sleeps 0.5 seconds between requests (120 requests/minute max)
2. **Batch processing** - Iterates over student_ids and posts each comment
3. **Error handling** - Tracks successes and failures separately
4. **Logging** - Logs progress and errors for debugging

## User Workflow

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
7. User can still:
   - Retry posting (if failures occurred)
   - Download ZIP files
   - Close batch session

## Key Features

### Safety
- **No duplicate posting**: Form is hidden after posting (prevents accidental re-posts)
- **Overwrite protection**: Checkbox to allow overwriting is unchecked by default
- **Manual fallback**: ZIP download option remains available
- **Clear feedback**: Detailed error messages if posting fails

### Rate Limiting
- 0.5 second delay between student comments
- Respects Canvas API rate limits
- Prevents API throttling

### Error Handling
- Skips assignments without Canvas mappings
- Skips assignments with no comments
- Collects and displays all errors
- Allows retry on partial failures

## Data Flow

```
Batch Session
    ↓
Finalize
    ↓
Export Created (gradebook + ZIPs)
    ↓
User clicks "Post to Canvas"
    ↓
POST /classwork/canvas_post_comments
    ↓
For each assignment:
  - Build comments dict {student_id: {text: "comment"}}
  - Call canvas_client.bulk_post_comments()
  - Track success/failure
    ↓
Render results page
  - Show summary
  - Show per-assignment status
  - Show errors (if any)
```

## Canvas API Endpoint Used

```
PUT /api/v1/courses/{course_id}/assignments/{assignment_id}/submissions/{student_id}
```

Request body:
```json
{
  "comment": {
    "text_comment": "Comment text here"
  }
}
```

## Notes

- Comments are **additive** - each call adds a new comment
- Canvas does not have a "replace comment" endpoint
- The overwrite checkbox is advisory only (cannot actually replace)
- Best practice: Grade each assignment once to avoid comment accumulation

## Testing

To test the feature:

1. Set up a Canvas batch with comments
2. Finalize the batch
3. Click "Post Comments to Canvas"
4. Verify:
   - Success messages appear
   - Comments are visible in Canvas SpeedGrader
   - Error handling works (try with invalid API key)