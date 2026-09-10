@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if %errorlevel% equ 0 (
    python scripts\start_instructor_grader.py
) else (
    where py >nul 2>&1
    if %errorlevel% neq 0 (
        echo ERROR: Python was not found.
        echo Install Python, then reopen this shortcut and try again.
        pause
        exit /b 1
    )
    py scripts\start_instructor_grader.py
)

if %errorlevel% neq 0 (
    echo.
    echo The grader did not start. Leave this window open and copy the error
    echo message when asking for help.
    pause
)
