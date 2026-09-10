# Desktop Shortcuts for Grading

## Instructor paper-quiz shortcut (recommended)

Create one shortcut whose name answers the weekly question of what to open:

1. In File Explorer, open the project folder.
2. Right-click `START_INSTRUCTOR_GRADER.bat`.
3. Choose **Show more options** → **Send to** → **Desktop (create shortcut)**.
4. Rename the shortcut **Logic Grader - Start Here**.

Double-clicking it opens the Setup Wizard for the real-data workflow. Keep the
server window open while grading and close it when finished.

The separate classwork shortcut below is optional. It opens a different part of
the same app and should not be the main entry point for paper quizzes.

## Quick Setup (30 seconds)

1. **Locate the batch file**
   - Find `start_classwork.bat` in your project folder:
   - `c:\Users\seber\logic-grading-facilitator-MAIN\start_classwork.bat`

2. **Create the desktop shortcut**
   - Right-click on `start_classwork.bat`
   - Select "Send to" → "Desktop (create shortcut)"
   - A shortcut will appear on your desktop

3. **Optional: Rename the shortcut**
   - Right-click the desktop shortcut
   - Select "Rename"
   - Name it something like "Classwork Grading" or "Grade Classwork"

4. **Optional: Change the icon** (Recommended!)
   - Right-click the desktop shortcut
   - Select "Properties"
   - Click "Change Icon..."
   - Browse to: `c:\Users\seber\logic-grading-facilitator-MAIN\icon_classwork.ico`
   - Click "Open", then "OK", then "Apply"
   - Your shortcut will now have a custom classwork icon!

## How to Use

Simply double-click the desktop shortcut to start grading classwork!

The script will:
1. ✅ Check that Python is installed
2. ✅ Start the FastAPI server in a separate window
3. ✅ Open your default browser to `http://127.0.0.1:5000/classwork`

## When You're Done

- Close the browser tab when finished reviewing
- Close the "Logic Grading Server" window to stop the server

## Troubleshooting

**If the browser doesn't open:**
- Manually navigate to `http://127.0.0.1:5000/classwork`
- Make sure port 5000 isn't being used by another application

**If you see "Python is not installed":**
- Install Python from https://python.org
- Make sure to check "Add Python to PATH" during installation

**If you need to use environment variables (PRODUCTION mode):**
Edit the `start_classwork.bat` file and add these lines before the server starts:
```batch
set DATA_DIR=C:\grading-data\live
set DATA_ARCHIVE_DIR=C:\grading-data\archive
set ENV_MODE=PRODUCTION
