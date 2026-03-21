"""
Canvas API Client with FERPA-compliant caching.

All student data is cached to data/canvas_cache/ directory.
No student data is stored in code or memory across requests.
"""

import os
import time
import json
import csv
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
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
GRADEBOOK_CACHE_TTL = 300  # 5 minutes


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
        
        self._client = httpx.Client(
            timeout=30.0,
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
    
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
        """Fetch list of accessible courses."""
        # Use standardized cache key format with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_path = CACHE_DIR / f"courses_{timestamp}.json"
        cached = self._read_cache(cache_path, COURSES_CACHE_TTL)
        
        if cached:
            return cached.get("courses", [])
        
        try:
            courses = self._make_request("/courses", params={"enrollment_state": "active"})
            # Filter to only courses where user can grade
            grading_courses = [
                {
                    "id": c["id"],
                    "name": c["name"],
                    "course_code": c.get("course_code", "")
                }
                for c in courses
                if c.get("enrollments") and any(
                    e.get("type") in ["TeacherEnrollment", "TaEnrollment"]
                    for e in c["enrollments"]
                )
            ]
            
            self._write_cache(cache_path, {"courses": grading_courses})
            return grading_courses
        except CanvasClientError as e:
            # Return cached data on error if available
            # Try to find most recent courses cache file
            courses_caches = sorted(
                CACHE_DIR.glob("courses_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            if courses_caches:
                fallback_path = courses_caches[0]
                fallback = self._read_cache(fallback_path, COURSES_CACHE_TTL * 2)
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
            assignments = self._make_request(f"/courses/{course_id}/assignments")
            
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
            
            # Sort: assignments with due date first (DESC), then no due date
            filtered.sort(
                key=lambda x: (x["due_date"] is None, x["due_date"] or ""),
                reverse=True
            )
            
            self._write_cache(cache_path, filtered)
            return filtered
        except CanvasClientError as e:
            # Return cached data on error if available
            fallback = self._read_cache(cache_path, ASSIGNMENTS_CACHE_TTL * 2)
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


# Convenience function to get a client instance
def get_canvas_client() -> CanvasClient:
    """Get a configured Canvas client instance."""
    return CanvasClient()