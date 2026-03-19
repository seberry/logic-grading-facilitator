@echo off
REM Logic Grading Facilitator - Classwork Workflow Launcher
REM This script starts the server and opens the classwork grading interface

echo ========================================
echo Logic Grading Facilitator
echo Starting Classwork Workflow...
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and try again.
    pause
    exit /b 1
)

REM Change to the script directory (where this .bat file is located)
cd /d "%~dp0"

echo [1/3] Starting the FastAPI server on port 5000...
echo Server will run in this window. Keep it open while grading.
echo.

REM Start the server in the background using START
start "Logic Grading Server" cmd /k "python -m uvicorn app.server:app --reload --port 5000"

echo Waiting for server to initialize...
timeout /t 5 /nobreak >nul

echo.
echo [2/3] Opening browser to classwork workflow...
echo.

REM Open the default browser to the classwork route
start http://127.0.0.1:5000/classwork

echo [3/3] Done!
echo.
echo ========================================
echo Classwork workflow should now be open in your browser.
echo The server is running in a separate window.
echo Close the server window when you're done grading.
echo ========================================
echo.