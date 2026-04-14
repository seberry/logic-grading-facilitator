"""
Canvas API Client with FERPA-compliant caching.

All student data is cached to data/canvas_cache/ directory.
No student data is stored in code or memory across requests.
"""

import os
import time
import json
import csv
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
from pathlib import Path
import io

import httpx


# Configuration from environment variables
CANVAS_BASE_URL = os.getenv("CANVAS_BASE_URL", "")
CANVAS_API_KEY = os.getenv("CANVAS_API_KEY", "")
CANVAS_API_VERSION = "/api/v1"

# Cache directory - FERPA compliant: all student data goes here
CACHE_DIR = Path(__file__).parent.parent / "data" / "canvas_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cache TTL in seconds
COURSES_CACHE_TTL = 3600  # 1 hour
ASSIGNMENTS_CACHE_TTL = 1800  # 30 minutes
SUBMISSION_COUNTS_TTL = 300  # 5 minutes
GRADEBOOK_CACHE_TTL = 300  # 5 minutes

# Maximum number of cache files to keep per course
MAX_CACHE_FILES_PER_COURSE = 5
ATTACHMENT_DOWNLOAD_MAX_ATTEMPTS = 4
ATTACHMENT_DOWNLOAD_BACKOFF_SECONDS = 1.0


def _cleanup_old_cache_files(course_id: str, cache_type: str, keep_count: int = MAX_CACHE_FILES_PER_COURSE) -> None:
    """
    Clean up old cache files for a specific course and cache type.
    
    Keeps only the most recent `keep_count` files based on timestamp in filename.
    This prevents cache file accumulation that causes performance issues.
    
    Args:
        course_id: Canvas course ID
        cache_type: Type of cache (e.g., "assignments", "submission_counts")
        keep_count: Maximum number of files to keep (default: MAX_CACHE_FILES_PER_COURSE)
    """
    try:
        # Find all cache files for this course and type
        pattern = f"{course_id}_{cache_type}_*.json"
        cache_files = sorted(
            CACHE_DIR.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True  # Most recent first
        )
        
        # Remove excess files
        if len(cache_files) > keep_count:
            for old_file in cache_files[keep_count:]:
                try:
                    old_file.unlink()
                    print(f"[CACHE CLEANUP] Removed old cache: {old_file.name}")
                except OSError as e:
                    print(f"[CACHE CLEANUP] Failed to remove {old_file.name}: {e}")
    except Exception as e:
        print(f"[CACHE CLEANUP] Error during cleanup: {e}")


class CanvasClientError(Exception):
    """Base exception for Canvas API errors."""
    pass


class CanvasRateLimitError(CanvasClientError):
    """Raised when rate limit is hit."""
    pass


class CanvasClient:
    """Canvas API client with caching and rate limiting."""
    
    def __init__(self, base_url: str = None, api_key: str = None):
        self.base_url = (base_url or CANVAS_BASE_URL).rstrip("/")
        self.api_key = api_key or CANVAS_API_KEY
        self.api_base = f"{self.base_url}{CANVAS_API_VERSION}"
        
        if not self.base_url or not self.api_key:
            raise CanvasClientError(
                "Canvas API credentials not configured. "
                "Set CANVAS_BASE_URL and CANVAS_API_KEY environment variables."
            )
        
        self._client = self._build_http_client()

    def _build_http_client(self) -> httpx.Client:
        return httpx.Client(
            timeout=30.0,
            follow_redirects=True,  # Follow Canvas 302 redirects for file downloads
            headers={"Authorization": f"Bearer {self.api_key}"}
        )

    def _reset_http_client(self) -> None:
        self._client.close()
        self._client = self._build_http_client()

    def _download_attachment_with_retry(
        self,
        *,
        file_url: str,
        filename: str,
        logger: logging.Logger,
        phase_label: str = "pass 1",
    ) -> bytes:
        """Download a Canvas attachment with limited retries for transient failures."""
        last_error: Optional[Exception] = None

        for attempt in range(1, ATTACHMENT_DOWNLOAD_MAX_ATTEMPTS + 1):
            try:
                response = self._client.get(file_url)
                response.raise_for_status()
                if attempt > 1:
                    logger.info(
                        "  Downloaded %s on %s retry %s/%s",
                        filename,
                        phase_label,
                        attempt,
                        ATTACHMENT_DOWNLOAD_MAX_ATTEMPTS,
                    )
                return response.content
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                should_retry = status_code >= 500
                logger.warning(
                    "  %s attempt %s/%s failed for %s with HTTP %s",
                    phase_label,
                    attempt,
                    ATTACHMENT_DOWNLOAD_MAX_ATTEMPTS,
                    filename,
                    status_code,
                )
                last_error = e
                if not should_retry or attempt == ATTACHMENT_DOWNLOAD_MAX_ATTEMPTS:
                    break
            except httpx.RequestError as e:
                logger.warning(
                    "  %s attempt %s/%s failed for %s: %s",
                    phase_label,
                    attempt,
                    ATTACHMENT_DOWNLOAD_MAX_ATTEMPTS,
                    filename,
                    e,
                )
                last_error = e
                if attempt == ATTACHMENT_DOWNLOAD_MAX_ATTEMPTS:
                    break

            time.sleep(ATTACHMENT_DOWNLOAD_BACKOFF_SECONDS * attempt)

        assert last_error is not None
        raise last_error
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict[str, Any]:
        """Make authenticated request to Canvas API."""
        url = f"{self.api_base}{endpoint}"
        
        try:
            response = self._client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise CanvasRateLimitError("Canvas API rate limit exceeded")
            raise CanvasClientError(f"Canvas API error: {e.response.status_code}")
        except httpx.RequestError as e:
            raise CanvasClientError(f"Request failed: {e}")
    
    def _make_paginated_request(self, endpoint: str, params: Dict = None) -> List[Dict[str, Any]]:
        """
        Make authenticated request to Canvas API with full pagination support.
        
        Canvas API paginates results using the Link header. This method
        fetches all pages and returns combined results.
        
        Args:
            endpoint: API endpoint (e.g., "/courses")
            params: Query parameters
            
        Returns:
            Combined list of all items from all pages
        """
        if params is None:
            params = {}
        
        # Request more items per page to reduce number of requests
        params["per_page"] = 100
        
        url = f"{self.api_base}{endpoint}"
        all_items = []
        
        try:
            while url:
                response = self._client.get(url, params=params if not all_items else None)
                response.raise_for_status()
                
                # Add items from this page
                items = response.json()
                if items:
                    all_items.extend(items)
                
                # Check for next page in Link header
                link_header = response.headers.get("Link", "")
                url = None
                
                # Parse Link header for "next" URL
                # Format: <url>; rel="next", <url>; rel="last"
                for link in link_header.split(","):
                    if 'rel="next"' in link:
                        # Extract URL from <url>
                        url_match = link.strip().split("<")[1].split(">")[0]
                        url = url_match
                        break
            
            return all_items
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise CanvasRateLimitError("Canvas API rate limit exceeded")
            raise CanvasClientError(f"Canvas API error: {e.response.status_code}")
        except httpx.RequestError as e:
            raise CanvasClientError(f"Request failed: {e}")
    
    def _get_cache_path(self, course_id: str, cache_type: str) -> Path:
        """Generate cache file path for a specific course and cache type."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{course_id}_{cache_type}_{timestamp}.json"
        return CACHE_DIR / filename
    
    def _read_cache(self, cache_path: Path, ttl: int) -> Optional[Dict]:
        """Read cache if it exists and is not expired."""
        if not cache_path.exists():
            return None
        
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                cache_data = json.load(f)
            
            cache_time = datetime.fromisoformat(cache_data.get("cached_at", ""))
            if datetime.now() - cache_time > timedelta(seconds=ttl):
                return None
            
            return cache_data.get("data")
        except (json.JSONDecodeError, KeyError, ValueError):
            return None
    
    def _write_cache(self, cache_path: Path, data: Dict) -> None:
        """Write data to cache file."""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "cached_at": datetime.now().isoformat(),
            "data": data
        }
        cache_path.write_text(json.dumps(cache_data, indent=2), encoding="utf-8")
    
    def get_courses(self) -> List[Dict[str, Any]]:
        """
        Fetch list of accessible courses with full pagination.
        
        Uses _make_paginated_request to fetch all pages of courses.
        Returns only courses where user has teacher or TA grading permissions.
        
        Uses static cache filename with timestamp stored in file for TTL checking.
        """
        # Use static cache filename (timestamp stored inside file for TTL)
        cache_path = CACHE_DIR / "courses_cache.json"
        cached = self._read_cache(cache_path, COURSES_CACHE_TTL)
        
        if cached:
            return cached.get("courses", [])
        
        try:
            # Use paginated request to get ALL courses, not just first page
            courses = self._make_paginated_request("/courses", params={"enrollment_state": "active"})
            
            # Filter to only courses where user can grade
            # Canvas API returns lowercase "teacher" and "ta" (not "TeacherEnrollment"/"TaEnrollment")
            grading_courses = [
                {
                    "id": c["id"],
                    "name": c["name"],
                    "course_code": c.get("course_code", "")
                }
                for c in courses
                if c.get("enrollments") and any(
                    e.get("type") in ["teacher", "ta"]
                    for e in c["enrollments"]
                )
            ]
            
            self._write_cache(cache_path, {"courses": grading_courses})
            return grading_courses
        except CanvasClientError as e:
            # Return cached data on error if available
            fallback = self._read_cache(cache_path, COURSES_CACHE_TTL * 2)
            if fallback:
                return fallback.get("courses", [])
            raise
    
    def get_assignments(self, course_id: int) -> List[Dict[str, Any]]:
        """Fetch assignments for a course with metadata."""
        cache_path = self._get_cache_path(str(course_id), "assignments")
        cached = self._read_cache(cache_path, ASSIGNMENTS_CACHE_TTL)
        
        if cached:
            return cached
        
        try:
            # Use paginated request to get ALL assignments, not just first page
            assignments = self._make_paginated_request(f"/courses/{course_id}/assignments")
            
            # Filter and transform assignments
            filtered = []
            for a in assignments:
                # Skip unpublished assignments
                if not a.get("published"):
                    continue
                
                # Skip assignments without submission types we care about
                submission_types = a.get("submission_types", [])
                if not any(st in submission_types for st in ["online_upload", "online_text_entry", "on_paper"]):
                    continue
                
                # Parse due date
                due_at = a.get("due_at")
                due_date = None
                if due_at:
                    try:
                        due_date = datetime.fromisoformat(due_at.replace("Z", "+00:00"))
                    except (ValueError, AttributeError):
                        pass
                
                filtered.append({
                    "id": a["id"],
                    "name": a["name"],
                    "max_points": a.get("points_possible", 0),
                    "due_date": due_date.isoformat() if due_date else None,
                    "published": a.get("published", False)
                })
            
            # Sort assignments with date-aware strategy:
            # 1. Past-due assignments first (most recent past-due at top)
            # 2. Future-due assignments next (soonest upcoming first)
            # 3. No due date assignments last
            
            # Get current date (timezone-aware)
            now = datetime.now(timezone.utc)
            
            # Categorize assignments
            past_due = []
            future_due = []
            no_due = []
            
            for assignment in filtered:
                if assignment["due_date"] is None:
                    no_due.append(assignment)
                else:
                    due_date = datetime.fromisoformat(assignment["due_date"])
                    if due_date < now:
                        past_due.append(assignment)
                    else:
                        future_due.append(assignment)
            
            # Sort each bucket
            # Past-due: most recent past-due first (DESC)
            past_due.sort(key=lambda x: x["due_date"], reverse=True)
            # Future-due: soonest upcoming first (ASC)
            future_due.sort(key=lambda x: x["due_date"])
            # No due date: keep original order
            
            # Concatenate buckets
            filtered = past_due + future_due + no_due
            
            self._write_cache(cache_path, filtered)
            # Clean up old cache files to prevent accumulation
            _cleanup_old_cache_files(str(course_id), "assignments")
            return filtered
        except CanvasClientError as e:
            # Return cached data on error if available
            fallback = self._read_cache(cache_path, ASSIGNMENTS_CACHE_TTL * 2)
            if fallback:
                return fallback
            raise
    
    def get_assignment_submission_counts(self, course_id: int, assignment_ids: List[int]) -> Dict[int, Dict[str, int]]:
        """
        Get submission counts for multiple assignments.
        
        Returns:
            Dict mapping assignment_id to {
                "submitted": int,      # total submissions
                "graded": int,         # graded submissions
                "ungraded": int        # submissions needing grading
            }
        """
        cache_path = CACHE_DIR / f"{course_id}_submission_counts.json"
        cached = self._read_cache(cache_path, SUBMISSION_COUNTS_TTL)
        
        if cached:
            return cached
        
        counts = {}
        
        try:
            # Fetch submission summaries for all requested assignments
            # Canvas API provides efficient batch query via include[] parameter
            for assignment_id in assignment_ids:
                endpoint = f"/courses/{course_id}/assignments/{assignment_id}/submissions"
                submissions = self._make_request(
                    endpoint,
                    params={
                        "include": ["submission_history", "user"],
                        "per_page": 100
                    }
                )
                
                # Count submissions
                submitted = 0
                graded = 0
                
                for submission in submissions:
                    # Count only submitted (not missing/excused)
                    if submission.get("workflow_state") != "unsubmitted":
                        submitted += 1
                        # Count as graded if it has a score
                        if submission.get("score") is not None:
                            graded += 1
                
                counts[assignment_id] = {
                    "submitted": submitted,
                    "graded": graded,
                    "ungraded": submitted - graded
                }
            
            self._write_cache(cache_path, counts)
            return counts
            
        except CanvasClientError as e:
            # Return cached data on error if available
            fallback = self._read_cache(cache_path, SUBMISSION_COUNTS_TTL * 2)
            if fallback:
                return fallback
            raise
    
    def get_gradebook_baseline(self, course_id: int) -> str:
        """
        Fetch gradebook baseline from Canvas API.
        
        Uses the Enrollments API to get current grades for overwrite protection.
        Returns a CSV string with student IDs and current scores.
        
        Note: This is not a full Canvas gradebook export - it's a minimal
        baseline for detecting grade conflicts. Canvas does not provide a clean
        gradebook CSV download endpoint via API.
        """
        try:
            # Fetch enrollments with grades
            # This returns current scores for all students in the course
            enrollments = self._make_request(
                f"/courses/{course_id}/enrollments",
                params={"type": "StudentEnrollment", "per_page": 100}
            )
            
            # Handle pagination if needed
            while "next" in enrollments.get("links", {}):
                next_url = enrollments["links"]["next"]
                response = self._client.get(next_url)
                response.raise_for_status()
                more_enrollments = response.json()
                enrollments.extend(more_enrollments)
            
            # Build CSV structure for overwrite protection
            # Format: SIS Login ID, Student, [Assignment Columns]
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Header row
            header = ["Student", "SIS Login ID"]
            writer.writerow(header)
            
            # Data rows
            for enrollment in enrollments:
                student = enrollment.get("user", {})
                name = student.get("name", "")
                
                # Get SIS Login ID from student integration_id
                # Canvas stores this in various places depending on configuration
                sis_id = (
                    student.get("sis_user_id") or
                    student.get("login_id") or
                    student.get("integration_id") or
                    ""
                )
                
                writer.writerow([name, sis_id])
            
            return output.getvalue()
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise CanvasRateLimitError("Canvas API rate limit exceeded")
            raise CanvasClientError(f"Canvas API error: {e.response.status_code}")
        except httpx.RequestError as e:
            raise CanvasClientError(f"Request failed: {e}")
    
    def get_gradebook_csv(self, course_id: int) -> str:
        """
        Fetch gradebook as CSV string.
        
        DEPRECATED: Canvas does not provide a clean gradebook CSV endpoint.
        Use get_gradebook_baseline() instead for API-based gradebook.
        
        This method is kept for backwards compatibility but should not be used.
        """
        # Fallback to API-based baseline
        return self.get_gradebook_baseline(course_id)
    
    def bulk_post_comments(
        self,
        course_id: int,
        assignment_id: int,
        comments: Dict[str, Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Post comments for multiple students at once.
        
        IMPORTANT: Canvas has no true bulk comment endpoint. This loops over
        individual submission API calls. Rate limiting is critical.
        
        Args:
            course_id: Canvas course ID
            assignment_id: Canvas assignment ID
            comments: Dict mapping student_id to {text: str, file_path: Optional[str]}
        
        Returns:
            Dict with success/failure status per student
        """
        import logging
        logger = logging.getLogger(__name__)
        
        results = {
            "success": [],
            "failed": []
        }
        
        total_students = len(comments)
        logger.info(f"Starting bulk comment post for {total_students} students (course {course_id}, assignment {assignment_id})")
        
        for idx, (student_id, comment_data) in enumerate(comments.items(), 1):
            # Rate limiting: sleep between requests to avoid hitting Canvas limits
            # Canvas typically allows ~100-200 requests per minute
            if idx > 1:
                # Sleep 0.5 seconds between requests (120 per minute max)
                time.sleep(0.5)
            
            try:
                # Use Canvas submissions API to post comment
                endpoint = f"/courses/{course_id}/assignments/{assignment_id}/submissions/{student_id}"
                
                # Post text comment
                comment_text = comment_data.get("text", "").strip()
                if comment_text:
                    payload = {
                        "comment": {
                            "text_comment": comment_text
                        }
                    }
                    
                    response = self._client.put(
                        f"{self.api_base}{endpoint}",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json=payload
                    )
                    response.raise_for_status()
                    
                    results["success"].append({
                        "student_id": student_id,
                        "comment_posted": True
                    })
                    logger.info(f"Comment posted for student {student_id} ({idx}/{total_students})")
                else:
                    results["success"].append({
                        "student_id": student_id,
                        "comment_posted": False,
                        "reason": "No text comment provided"
                    })
                    logger.info(f"Skipped student {student_id} ({idx}/{total_students}): no comment text")
                    
            except httpx.HTTPStatusError as e:
                error_msg = f"HTTP {e.response.status_code}"
                results["failed"].append({
                    "student_id": student_id,
                    "error": error_msg
                })
                logger.error(f"Failed to post comment for student {student_id} ({idx}/{total_students}): {error_msg}")
                
                # Handle rate limit specifically
                if e.response.status_code == 429:
                    logger.warning("Rate limit hit. Sleeping for 10 seconds before retrying...")
                    time.sleep(10)
                    
            except httpx.RequestError as e:
                error_msg = str(e)
                results["failed"].append({
                    "student_id": student_id,
                    "error": error_msg
                })
                logger.error(f"Request failed for student {student_id} ({idx}/{total_students}): {error_msg}")
                
            except Exception as e:
                error_msg = str(e)
                results["failed"].append({
                    "student_id": student_id,
                    "error": error_msg
                })
                logger.error(f"Unexpected error for student {student_id} ({idx}/{total_students}): {error_msg}")
        
        logger.info(f"Bulk comment post complete: {len(results['success'])} successful, {len(results['failed'])} failed")
        return results
    
    def close(self):
        """Close HTTP client."""
        self._client.close()

    def download_assignment_submissions(
        self,
        course_id: int,
        assignment_id: int,
        output_dir: Path,
    ) -> tuple[Dict[str, List[Path]], Dict[str, int], List[Dict[str, str]]]:
        """
        Download all submission files for an assignment from Canvas.
        
        Args:
            course_id: Canvas course ID
            assignment_id: Canvas assignment ID
            output_dir: Directory to save downloaded files
            
        Returns:
            Tuple of:
            - Dict mapping student_id (sis_login_id) to list of downloaded file paths
            - Dict mapping student_id (sis_login_id) to canvas_user_id (integer)
            - List of exhausted download warnings with student/file/error context
            
        Raises:
            CanvasClientError: If download fails
        """
        logger = logging.getLogger(__name__)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        student_files: Dict[str, List[Path]] = {}
        canvas_user_ids: Dict[str, int] = {}  # sis_login_id -> canvas_user_id mapping
        download_warnings: List[Dict[str, str]] = []
        failed_downloads: List[Dict[str, str]] = []
        
        try:
            # Fetch all submissions for this assignment
            endpoint = f"/courses/{course_id}/assignments/{assignment_id}/submissions"
            params = {
                "include": ["submission_history", "user"],
                "per_page": 100
            }
            
            submissions = self._make_request(endpoint, params=params)
            total = len(submissions)
            logger.info(f"Fetching {total} submissions for assignment {assignment_id}")
            
            for idx, submission in enumerate(submissions, 1):
                # Skip unsubmitted
                if submission.get("workflow_state") == "unsubmitted":
                    continue
                
                user = submission.get("user", {})
                canvas_user_id = user.get("id")
                student_id = user.get("sis_user_id") or user.get("login_id") or user.get("id")
                
                if not student_id:
                    logger.warning(f"Submission {idx}/{total} has no student ID, skipping")
                    continue
                
                # Build Canvas user ID mapping for comment posting
                # We map sis_login_id (student_id) to canvas_user_id (integer)
                if canvas_user_id and student_id not in canvas_user_ids:
                    canvas_user_ids[student_id] = canvas_user_id
                
                student_name = user.get("name", "")
                logger.info(f"Processing submission {idx}/{total}: {student_name} ({student_id})")
                
                # Handle different submission types
                attachments = submission.get("attachments", [])
                
                # Case 1: File attachments (online_upload, on_paper)
                if attachments:
                    for attachment in attachments:
                        file_url = attachment.get("url")
                        filename = attachment.get("filename")
                        
                        if not file_url or not filename:
                            continue
                        
                        # Download file
                        try:
                            file_bytes = self._download_attachment_with_retry(
                                file_url=file_url,
                                filename=filename,
                                logger=logger,
                                phase_label="pass 1",
                            )
                            
                            # Save to output directory
                            # Preserve Canvas naming: {last_name}_{first_name}_assignment_{id}_{original_name}
                            file_path = output_dir / filename
                            
                            # Handle duplicate filenames
                            counter = 1
                            while file_path.exists():
                                stem = Path(filename).stem
                                suffix = Path(filename).suffix
                                file_path = output_dir / f"{stem}_{counter}{suffix}"
                                counter += 1
                            
                            file_path.write_bytes(file_bytes)
                            
                            # Add to student's file list
                            if student_id not in student_files:
                                student_files[student_id] = []
                            student_files[student_id].append(file_path)
                            
                            logger.info(f"  Downloaded: {filename}")
                            
                        except httpx.HTTPStatusError as e:
                            logger.error(f"  Failed to download {filename}: {e}")
                            failed_downloads.append(
                                {
                                    "student_id": str(student_id),
                                    "student_name": student_name,
                                    "filename": filename,
                                    "file_url": file_url,
                                    "error": str(e),
                                }
                            )
                            continue
                        except Exception as e:
                            logger.error(f"  Error downloading {filename}: {e}")
                            failed_downloads.append(
                                {
                                    "student_id": str(student_id),
                                    "student_name": student_name,
                                    "filename": filename,
                                    "file_url": file_url,
                                    "error": str(e),
                                }
                            )
                            continue
                
                # Case 2: Text submission (online_text_entry)
                elif submission.get("body") or submission.get("text_comment"):
                    text_content = submission.get("body") or submission.get("text_comment", "")
                    
                    # Clean HTML content to remove Canvas-specific tags (link, script, style)
                    from .utils import clean_html_text
                    cleaned_text = clean_html_text(text_content)
                    
                    # Create text file from submission
                    # Canvas doesn't provide filename for text submissions, so we create one
                    filename = f"{student_name.replace(',', '').replace(' ', '_')}_submission.txt"
                    file_path = output_dir / filename
                    
                    # Handle duplicates
                    counter = 1
                    while file_path.exists():
                        file_path = output_dir / f"{student_name.replace(',', '').replace(' ', '_')}_submission_{counter}.txt"
                        counter += 1
                    
                    file_path.write_text(cleaned_text, encoding="utf-8")
                    
                    if student_id not in student_files:
                        student_files[student_id] = []
                    student_files[student_id].append(file_path)
                    
                    logger.info(f"  Created text file from submission")
            
            if failed_downloads:
                logger.info(
                    "Retrying %s exhausted attachment download(s) in a second pass with a fresh client",
                    len(failed_downloads),
                )
                self._reset_http_client()
                second_pass_failures: List[Dict[str, str]] = []
                for item in failed_downloads:
                    filename = item["filename"]
                    file_url = item["file_url"]
                    student_id = item["student_id"]
                    student_name = item["student_name"]
                    try:
                        file_bytes = self._download_attachment_with_retry(
                            file_url=file_url,
                            filename=filename,
                            logger=logger,
                            phase_label="pass 2",
                        )
                        file_path = output_dir / filename
                        counter = 1
                        while file_path.exists():
                            stem = Path(filename).stem
                            suffix = Path(filename).suffix
                            file_path = output_dir / f"{stem}_{counter}{suffix}"
                            counter += 1
                        file_path.write_bytes(file_bytes)
                        if student_id not in student_files:
                            student_files[student_id] = []
                        student_files[student_id].append(file_path)
                        logger.info("  Recovered on second pass: %s", filename)
                    except Exception as e:
                        second_pass_failures.append(
                            {
                                "student_id": student_id,
                                "student_name": student_name,
                                "filename": filename,
                                "error": str(e),
                            }
                        )
                        logger.error("  Final failure after second pass for %s: %s", filename, e)
                download_warnings = second_pass_failures

            logger.info(f"Download complete: {len(student_files)} students with files")
            logger.info(f"Canvas user ID mapping built: {len(canvas_user_ids)} students")
            if download_warnings:
                logger.warning(
                    "Download completed with %s final attachment failure(s)",
                    len(download_warnings),
                )
            return student_files, canvas_user_ids, download_warnings
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise CanvasRateLimitError("Canvas API rate limit exceeded during download")
            raise CanvasClientError(f"Canvas API error: {e.response.status_code}")
        except Exception as e:
            raise CanvasClientError(f"Failed to download submissions: {e}")


# Convenience function to get a client instance
def get_canvas_client() -> CanvasClient:
    """Get a configured Canvas client instance."""
    return CanvasClient()
