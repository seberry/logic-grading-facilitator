# Canvas User ID Mapping Implementation

## Problem Statement

When posting comments to Canvas via the API, the system needs to use Canvas user IDs (internal Canvas identifiers), not SIS login IDs (the external student identifiers used in gradebook CSVs). The Canvas API endpoint for posting comments requires:

```
PUT /api/v1/courses/:course_id/assignments/:assignment_id/submissions/:canvas_user_id
```

Where `canvas_user_id` is the internal Canvas user ID, NOT the SIS login ID.

## Solution Overview

The system now builds and maintains a mapping between SIS login IDs and Canvas user IDs throughout the classwork workflow:

1. **Collection Phase**: When downloading submissions from Canvas, extract Canvas user IDs
2. **Storage Phase**: Store the mapping in the batch state
3. **Usage Phase**: Look up Canvas user IDs when posting comments
4. **Cleanup Phase**: Clear mapping when batch is closed

## Implementation Details

### 1. Canvas Client (`app/canvas_client.py`)

Modified `download_assignment_submissions()` to return a tuple:
```python
# Before: Returns Dict[str, List[Path]]
student_files = client.download_assignment_submissions(...)

# After: Returns (Dict[str, List[Path]], Dict[str, int])
student_files, canvas_user_ids = client.download_assignment_submissions(...)
```

The `canvas_user_ids` dictionary maps:
- **Key**: SIS login ID (from submission.sis_login_id)
- **Value**: Canvas user ID (from submission.user_id)

### 2. Data Model (`app/models.py`)

Added new field to `ClassworkBatchState`:
```python
class ClassworkBatchState(BaseModel):
    # ... existing fields ...
    canvas_user_ids: Dict[str, int] = {}  # NEW: SIS login ID -> Canvas user ID
```

### 3. Server Integration (`app/server.py`)

#### Canvas Setup Endpoint
```python
# Accumulate Canvas user IDs across all assignments
canvas_user_ids: Dict[str, int] = {}

for assignment_data in selected_assignments:
    # Download submissions (returns tuple now)
    student_files, canvas_ids_for_assignment = client.download_assignment_submissions(...)
    
    # Accumulate mappings
    canvas_user_ids.update(canvas_ids_for_assignment)
    
    # ... process files ...

# Store in batch state
batch = ClassworkBatchState(
    # ...
    canvas_user_ids=canvas_user_ids,  # Store the mapping
)
```

#### Comment Posting Endpoint
```python
for mode in batch.assignments:
    # Build comments dict using Canvas user IDs
    comments_dict = {}
    for entry in mode.entries:
        if entry.comment and entry.comment.strip():
            # Look up Canvas user ID from batch mapping
            canvas_user_id = batch.canvas_user_ids.get(entry.student_id)
            if canvas_user_id:
                comments_dict[canvas_user_id] = {
                    "text": entry.comment.strip()
                }
            else:
                print(f"Warning: No Canvas user ID for student {entry.student_id}")
    
    # Post using Canvas user IDs
    client.bulk_post_comments(
        course_id=mode.canvas_course_id,
        assignment_id=mode.canvas_assignment_id,
        comments=comments_dict
    )
```

#### Batch Cleanup
Canvas user IDs are automatically cleared when:
- User clicks "Close Session" (sets `session.classwork_batch = None`)
- User clears the batch (sets `session.classwork_batch = None`)

## Benefits

1. **Correct API Usage**: Uses the right identifier for Canvas API calls
2. **No Manual Upload**: Comments go directly to Canvas via API
3. **Error Handling**: Warns if Canvas user ID lookup fails
4. **Data Privacy**: No sensitive data stored (only ID mappings)
5. **Automatic Cleanup**: Mappings cleared when batch is archived

## Testing

Run the test suite to verify:
```bash
python scripts\test_canvas_user_id_mapping.py
```

Tests cover:
- Canvas user ID collection during download
- Batch state storage of mappings
- Comment posting using Canvas user IDs
- Batch closure clears mappings

## Workflow Example

1. **Setup**: User selects Canvas course and assignments
2. **Download**: System downloads submissions and builds SIS → Canvas user ID map
3. **Grading**: User grades assignments and adds comments
4. **Post Comments**: System looks up Canvas user IDs and posts comments
5. **Completion**: User closes batch, mappings are cleared

## Data Flow

```
Canvas API (submissions)
  ↓
Extract: user_id (Canvas ID) + sis_login_id (SIS ID)
  ↓
Build mapping: {"student123": 98765, ...}
  ↓
Store in: ClassworkBatchState.canvas_user_ids
  ↓
When posting comment for "student123":
  ↓
Lookup: canvas_user_id = batch.canvas_user_ids.get("student123")  # 98765
  ↓
POST to: /api/v1/courses/123/assignments/456/submissions/98765