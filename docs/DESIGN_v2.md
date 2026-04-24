# Design Document: Local Logic Grading Facilitator (v2)
Status: This is a design doc. The current implementation uses FastAPI with server-rendered Jinja templates and file-based `data/grades/session.json`. The JS-heavy frontend and SQLite references below are aspirational; see `README.md` and `HOWTOUSE.md` for the current workflow.

## 1. Executive Summary
The Local Logic Grading Facilitator is a privacy-first (FERPA compliant), local-only tool designed to expedite the grading of handwritten intermediate logic exams.

It employs a **"Cyborg" / Hybrid Intelligence** approach:
1.  **AI (Local VLM):** Handles perception (segmentation, handwriting recognition) and initial translation.
2.  **Deterministic Code (JavaScript):** Handles verification (graph properties, truth evaluation) and visualization.
3.  **Human:** Acts as the final judge, validating the AI's translation and assigning partial credit.

## 2. Core Philosophy
* **Local-First:** No student data leaves the machine. All AI models run locally (e.g., Qwen via LM Studio).
* **Thick Client Architecture:** The browser (JavaScript) handles the "thinking" (logic checks, drawing, interactivity). The Python backend acts primarily as a persistence layer (file system/database).
* **Just-in-Time Grading:** The system avoids rigid pre-configuration. Answer keys and partial credit rules are defined dynamically ("pinned") by the user *during* the grading session.

## 3. System Architecture

### Backend (Python/FastAPI)
* **Role:** "Dumb" Persistence Layer & Orchestrator.
* **Responsibilities:**
    * Serve static HTML/JS/CSS files.
    * Manage the file-based session store (`data/grades/session.json`).
    * Handle file operations (reading raw scans, saving export CSVs).
    * Proxy requests to the Local VLM (if CORS/browser security requires it).
* **Key Endpoints (current app):**
    * `GET /grade/{unit_id}/page`: WorkUnit grading page.
    * `GET /problem/{problem_id}/strips`: Per-problem strip grader.
    * `GET /name_review`: Name review UI.
    * `GET /setup`: Setup wizard.
    * `GET /export`: Canvas CSV export.

### Frontend (HTML/JavaScript/D3.js)
* **Role:** The "Brain" & Interface.
* **Libraries:** `D3.js` (Graphing), `MPL.js` (Logic Evaluation - adapted from `rkirsling/modallogic`).
* **Responsibilities:**
    * **Visualization:** Renders Kripke models from text/JSON data.
    * **Logic Checking:** Runs deterministic checks (`isReflexive`, `evaluateFormula`) directly in the browser.
    * **AI Interaction:** Sends images/prompts to the Local VLM API and parses the response.

## 4. User Interface: The "Vertical Strip" Grading View

The core interface is designed for high-speed grading of a single question across all students.

### A. The Reference Pane (Sticky Header)
Always visible at the top of the screen. Contains three slots:
1.  **The Context:** Crop of the original blank question (so you know what you are grading).
    * *AI Feature:* Auto-extracted text constraints (e.g., "Must be transitive").
2.  **The "Golden Answer" (Pin):** A dynamic slot. When the user finds a perfect student answer, they "Pin" it here to serve as the visual comparison standard for all subsequent students.
3.  **The "Working Rubric":** An editable text area for ad-hoc notes (e.g., "-1 point for missing arrow heads").

### B. The Student Strip (Repeated Row)
Each student's submission for the current question is displayed as a row containing:

1.  **Column 1: The Human Scan**
    * Crop of the student's handwritten answer.
    * **Action:** "Re-crop" button (opens full page modal if AI missed context).

2.  **Column 2: The Machine Verification (Interactive)**
    * **Visualizer:** A D3.js graph rendered from the machine's interpretation of the handwriting.
    * **Workflow:**
        * User compares Scan (Col 1) vs. Visualizer (Col 2).
        * If mismatch: User edits the text representation until they match.
        * If match: User clicks **"Verify Logic"**.
    * **Feedback Output:**
        * The system lists passed/failed properties (e.g., "❌ Not Transitive: Missing <1,3>").
        * Evaluates truth of required sentences (e.g., "✅ < >P is True at w0").

3.  **Column 3: Assessment**
    * **Score Input:** Number field.
    * **Feedback Generator:** Buttons to insert error logs from Column 2 into the comments.
    * **Comment Box:** Free text.

## 5. Workflow: The "Cold Start"
1.  **Ingest:** User uploads scans. System groups them by student (background task).
2.  **Setup:** User selects "Question 1". The Reference Pane shows the blank Q1.
3.  **Grade Student 1:**
    * Answer is correct.
    * User clicks **"Pin as Reference"**.
    * User clicks **"Verify Logic"** to confirm properties.
    * Score is saved.
4.  **Grade Student 2:**
    * User compares handwriting to the Pinned Reference.
    * If visually identical -> Grade immediately.
    * If different -> Use AI/Manual verification tools in Column 2.


    ## Per-Problem Strip View (Segmentation UI)

Route: /problem/{problem_id}/strips

For each WorkUnit:

Runs segment_page_for_problem on its pages using SINGLE_QUESTION_PROMPT.

Picks the first present=True segment.

Renders the underlying scan with an overlaid bbox.

Shows a score field for that problem_id.

This is the main UI for grading a single problem across all students.


## 6. Data Model (Pydantic/JSON Schema)

### Kripke Model Representation
The standard format for data exchange between JS and Python.

```python
class KripkeState(BaseModel):
    id: int
    valuations: Dict[str, bool]  # Only true atoms, e.g. {"p": True}
    successors: List[int]        # IDs of reachable worlds

class KripkeModel(BaseModel):
    states: List[Optional[KripkeState]]
    # Optional: The compact string format from MPL.js for easy storage
    # e.g., "AqS0,2;;AS;"
    model_string: Optional[str]
```

This is the consolidated Design Document, updated to reflect the "Thick Client" (JavaScript-First) architecture we agreed upon.

You can paste this directly into a DESIGN.md file in your project.

Markdown

