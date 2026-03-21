#!/usr/bin/env python3
"""
Canvas API Validation Test Script

This script tests Canvas API connectivity and inspects response shapes.
Run this to verify CANVAS_BASE_URL and CANVAS_API_KEY are configured correctly.

Usage:
    python scripts/test_canvas_api.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.canvas_client import CanvasClient, CanvasClientError


def test_canvas_api():
    """Test Canvas API connectivity and inspect response shapes."""
    
    print("=" * 60)
    print("Canvas API Validation Test")
    print("=" * 60)
    print()
    
    # Test 1: Check environment variables
    print("Test 1: Checking environment variables...")
    base_url = os.getenv("CANVAS_BASE_URL")
    api_key = os.getenv("CANVAS_API_KEY")
    
    if not base_url:
        print("❌ FAILED: CANVAS_BASE_URL environment variable not set")
        print("   Expected format: https://your-canvas-instance.com")
        print("   Example: https://iu.instructure.com")
        return False
    
    if not api_key:
        print("❌ FAILED: CANVAS_API_KEY environment variable not set")
        print("   Generate an API key from your Canvas account settings")
        return False
    
    print(f"✅ CANVAS_BASE_URL: {base_url}")
    print(f"✅ CANVAS_API_KEY: {'*' * min(20, len(api_key))} (hidden)")
    print()
    
    # Test 2: Initialize client
    print("Test 2: Initializing Canvas client...")
    try:
        client = CanvasClient()
        print(f"✅ Client initialized successfully")
        print(f"   API base URL: {client.api_base}")
        print()
    except CanvasClientError as e:
        print(f"❌ FAILED: {e}")
        return False
    
    # Test 3: Fetch courses (with pagination)
    print("Test 3: Fetching accessible courses (with full pagination)...")
    print("   Note: This may take a few seconds if you have many courses...")
    try:
        courses = client.get_courses()
        print(f"✅ Found {len(courses)} teaching/TA courses")
        
        if courses:
            print()
            print("📚 Teaching/TA Courses:")
            for i, course in enumerate(courses[:10], 1):
                print(f"   {i}. {course['name']}")
                print(f"      ID: {course['id']} | Code: {course.get('course_code', 'N/A')}")
            if len(courses) > 10:
                print(f"   ... and {len(courses) - 10} more teaching/TA courses")
            print()
            
            # Check if user likely has current semester courses
            current_year = 2026
            current_semester = "SP26"  # Spring 2026
            has_current = any(
                current_semester in course['name'] or current_semester in course.get('course_code', '')
                for course in courses
            )
            
            if has_current:
                print(f"✅ Found courses with current semester marker ({current_semester})")
            else:
                print(f"⚠️  No courses found with current semester marker ({current_semester})")
                print(f"   Your current courses might use different naming conventions")
            print()
        else:
            print("⚠️  WARNING: No teaching/TA courses found")
            print("   This might mean:")
            print("   - API key doesn't have course access")
            print("   - You have no active teacher or TA enrollments")
            print("   - All your courses are student enrollments (not teaching)")
            print("   - Rate limiting is in effect")
            print()
            return False
    except CanvasClientError as e:
        print(f"❌ FAILED: {e}")
        return False
    
    # Test 4: Test first course's assignments
    if courses:
        test_course = courses[0]
        print(f"Test 4: Fetching assignments for course '{test_course['name']}'...")
        try:
            assignments = client.get_assignments(test_course['id'])
            print(f"✅ Found {len(assignments)} assignments")
            
            if assignments:
                print()
                print("Sample assignments:")
                for i, assignment in enumerate(assignments[:3], 1):
                    due_str = assignment['due_date'][:10] if assignment['due_date'] else 'No due date'
                    print(f"   {i}. {assignment['name']}")
                    print(f"      ID: {assignment['id']}")
                    print(f"      Max Points: {assignment['max_points']}")
                    print(f"      Due: {due_str}")
                    print(f"      Published: {assignment['published']}")
                if len(assignments) > 3:
                    print(f"   ... and {len(assignments) - 3} more")
                print()
                
                # Test 5: Inspect assignment response shape
                print("Test 5: Inspecting assignment response shape...")
                sample_assignment = assignments[0]
                print(f"✅ Assignment keys: {list(sample_assignment.keys())}")
                print(f"   Sample ID: {sample_assignment['id']} (type: {type(sample_assignment['id']).__name__})")
                print(f"   Sample name: {sample_assignment['name']} (type: {type(sample_assignment['name']).__name__})")
                print(f"   Sample max_points: {sample_assignment['max_points']} (type: {type(sample_assignment['max_points']).__name__})")
                print(f"   Sample due_date: {sample_assignment['due_date']} (type: {type(sample_assignment['due_date']).__name__ if sample_assignment['due_date'] else 'NoneType'})")
                print()
            else:
                print("⚠️  WARNING: No assignments found in course")
                print()
        except CanvasClientError as e:
            print(f"❌ FAILED: {e}")
            return False
    
    # Test 6: Test cache directory
    print("Test 6: Checking cache directory...")
    from app.canvas_client import CACHE_DIR
    print(f"✅ Cache directory: {CACHE_DIR}")
    if CACHE_DIR.exists():
        print(f"   Cache directory exists")
        cache_files = list(CACHE_DIR.glob("*"))
        print(f"   Existing cache files: {len(cache_files)}")
        if cache_files:
            for f in cache_files[:3]:
                size_kb = f.stat().st_size / 1024
                print(f"   - {f.name} ({size_kb:.1f} KB)")
            if len(cache_files) > 3:
                print(f"   ... and {len(cache_files) - 3} more")
    else:
        print(f"   Cache directory does not exist (will be created on first use)")
    print()
    
    # Test 7: Clean up
    print("Test 7: Closing client connection...")
    try:
        client.close()
        print("✅ Client closed successfully")
        print()
    except Exception as e:
        print(f"⚠️  WARNING: Error closing client: {e}")
        print()
    
    # Summary
    print("=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)
    print()
    print("Your Canvas API configuration is working correctly!")
    print("You can now use the Canvas integration in the classwork viewer.")
    print()
    
    return True


if __name__ == "__main__":
    success = test_canvas_api()
    sys.exit(0 if success else 1)