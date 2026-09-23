# Private remote grading on TuringMachine

## Use it from the office

Use a laptop for preparing a full TF (teaching fellow) bundle. A phone is useful
for checking that the home server is reachable and uploading a few scans.
Reviewing names, fixing question crops, and downloading a ZIP are easier on a
laptop.

### First time on an office laptop or phone

1. Install [Tailscale](https://tailscale.com/download) on that device. On a
   phone, approve its request to add a VPN configuration.
2. Sign in to Tailscale as **`seberry@invariant.org`**, the same account allowed
   by the tailnet policy. Make sure Tailscale says **Connected**. You do not
   need to connect to the office network specifically; ordinary Wi-Fi or mobile
   data is fine.
3. In your normal browser (Edge, Chrome, or Safari), open and bookmark:
   **[Open the grading Setup Wizard](https://turingmachine.tail238608.ts.net:8443/setup)**.
   Include `https://` and `:8443`. Port 443 without `:8443` opens Open WebUI.
4. You should see **Logic Grading Facilitator — Setup Wizard**. There is no
   second grader password; access is controlled by your Tailscale login.

The home desktop must be **powered on, awake, connected to Tailscale, and signed
in to Windows**. After a reboot, the grader starts when that Windows user logs
in. A sleeping or powered-off desktop cannot answer the office browser.

### Make a TF grading bundle on the laptop

1. Open the bookmarked Setup Wizard. If it shows **Archive & Reset Required**,
   the previous assignment is still active. Archive it only after confirming
   its exports are safe; this ends the active session and clears its scans.
2. Under **Step 1: Roster Selection**, choose an existing roster and click
   **Save roster**, or upload a current Canvas gradebook CSV and click
   **Build roster from gradebook**.
3. Under **Step 2: Scan Upload**, choose your JPG/PNG images, PDF, or ZIP and
   click **Upload scans**. The file picker sees files on the **office laptop**,
   not arbitrary files on the home desktop. If the scan PDF is only on the home
   desktop, upload it there before leaving or move it to the laptop through a
   permitted private method. The upload sends it to the home grader.
4. Under **Step 3: Rubric Definition**, load the correct saved rubric and click
   **Save rubric** after checking its questions, versions, and points. For a new
   quiz, prepare and review its rubric first; see
   [the quiz preparation workflow](docs/QUIZ_VARIANT_TF_WORKFLOW.md).
   **Generate from a blank assessment** is an optional external API path, so
   skip it when you want this workflow entirely local.
5. Under **Step 4: Run Auto-Grouping**, check the quiz ID and click
   **Run Auto-Grouping & Review Names**. The model runs on the home desktop;
   keep the browser page open while it works.
6. On **Name Review**, check every student's name and printed version (A/B/C).
   Correct mistakes and duplicates. **Save Corrections** is a draft save;
   when ready, click **Save assignments & prepare TF handoff**.
7. On the **Finish / Prepare TF handoff** page, click **Prepare remaining
   question locations** and leave the page open until all are ready. Click
   **Open Strip View** to spot-check the answer crops; use **Fix Crop** where
   needed. You do not have to enter grades first.
8. Return to **Finish** and click **Create TF bundle**. Your browser downloads
   a ZIP to its normal Downloads folder on the **office laptop**. That ZIP
   contains student data. Keep it out of Git and send it to the TF only through
   an approved private channel.

If you already started this quiz and just want to resume preparing its bundle,
open [Finish / Prepare TF handoff](https://turingmachine.tail238608.ts.net:8443/finish?mode=tf-prep)
directly. Do not click **Archive & Reset** for an assignment you are still using.
If you are grading it yourself, choose **Save assignments & grade now** on Name
Review instead; that opens the first question in Strip View.

On a phone, connect Tailscale and open the same bookmarked URL in Safari or
Chrome. You can check that Setup loads and upload JPG/PNG/PDF files from the
phone; the upload form does not accept HEIC photos. Use the laptop for the full
review and bundle download. A ZIP downloaded on the phone would also leave a
protected copy there.

If the page does not open, first check **Connected** in Tailscale and the exact
URL, especially `:8443`. If the home desktop is asleep or at the Windows sign-in
screen after a reboot, wake it and sign in. If Setup opens but model work fails,
use the home-computer checks below. You do not need to install Python, LM Studio,
or this repository on the office laptop or phone.

## What runs at home

The pieces have different jobs:

```text
Office browser -> Tailscale HTTPS :8443 -> grader on home PC :5001
                                         -> LM Studio on home PC :1234
```

Tailscale makes a private connection and forwards browser requests to the
grader. The grader reads and writes the home desktop's grading folders and
sends model requests to LM Studio on that same desktop. Windows Task Scheduler
only launches the grader after Windows login; it does not carry the browser
traffic. Tailscale Serve remembers its port mapping across restarts.

The instructor app runs on `127.0.0.1:5001`. LM Studio runs on
`127.0.0.1:1234`; the app uses its local OpenAI-compatible `/v1` API and the
`qwen/qwen3-vl-4b` model. The **Logic Grading Facilitator** Windows task runs at
this user's logon. It starts the LM Studio server if the local API is down, then
starts the app on its fixed port. Sign in to Windows after a reboot; the task
does not run before user logon. Tailscale starts separately.

The private office URL is:

`https://turingmachine.tail238608.ts.net:8443/setup`

This is Tailscale Serve, never Funnel. Port 8443 is configured in the
background, and the tailnet's general access rule currently allows the
instructor's login only. Keep that access rule restricted: the app has no
built-in login. See **Access control** below before inviting another person.

## Check and restart

Run these commands in PowerShell on TuringMachine:

```powershell
Get-ScheduledTask -TaskName 'Logic Grading Facilitator'
Get-ScheduledTaskInfo -TaskName 'Logic Grading Facilitator'
Invoke-WebRequest http://127.0.0.1:5001/setup
lms server status
Invoke-WebRequest http://127.0.0.1:1234/v1/models
& 'C:\Program Files\Tailscale\tailscale.exe' serve status
```

Task Scheduler result `267009` (`0x41301`) means the long-running task is
active. If it is stopped, run:

```powershell
Start-ScheduledTask -TaskName 'Logic Grading Facilitator'
```

To restart, stop the task, confirm port 5001 is no longer listening, then start
the task again. If the process remains, identify the process listening on port
5001 before stopping it; do not stop an unrelated Python process.

```powershell
Stop-ScheduledTask -TaskName 'Logic Grading Facilitator'
Get-NetTCPConnection -State Listen -LocalPort 5001 -ErrorAction SilentlyContinue
Start-ScheduledTask -TaskName 'Logic Grading Facilitator'
```

If Tailscale reaches the desktop but the page fails, check the task and local
`/setup` URL first. If the page loads but name recognition or question-location
preparation fails, check `lms server status`, the `/v1/models` endpoint, and
that `qwen/qwen3-vl-4b` is available. The startup entry point is
[`scripts/start_instructor_grader.py`](scripts/start_instructor_grader.py). To
run it manually from this repo, use:

```powershell
python scripts/start_instructor_grader.py --port 5001 --fixed-port --no-browser --ensure-lmstudio
```

Real scans, rosters, saved question locations, and generated bundles use the
production `DATA_DIR` and `DATA_ARCHIVE_DIR` specified in the ignored `.env`.
The TF bundle is generated by the app and downloaded through the browser; its
downloaded copy is protected student data on the office laptop. Keep it outside
Git and send it only through an approved private channel.

The grader and LM Studio listen on localhost only. Separately, the existing
Docker-backed Open WebUI port 3000 listens on all IPv6 interfaces on this
desktop. Tailscale grants do not limit direct LAN access to that Docker port;
review its Docker publish and Windows firewall if Open WebUI should be private
to Tailscale too. The Open WebUI setup was not changed here.

## Access control

Tailscale Serve follows the tailnet's access controls. This tailnet initially
had the default `*` to `*` on all ports rule. It would also permit port 8443;
adding a narrow grant does **not** override it. On **Policies → General access
rules** in the Tailscale admin console:

1. Click **Add rule**. Set **Source** to `seberry@invariant.org`, **Destination**
   to **All users and devices**, and **Port and protocol** to **All ports and
   protocols**. Save the rule.
2. Delete the original **All users and devices → All users and devices → All
   ports and protocols** rule, then save the policy. Leave separate SSH settings
   alone.

The resulting general access rule is equivalent to this grant and preserves
the instructor's existing tailnet access:

```json
{"src": ["seberry@invariant.org"], "dst": ["*"], "ip": ["*"]}
```

If the husband joins, add one separate grant using his actual Tailscale login:

```json
{"src": ["<husband's actual Tailscale login>"], "dst": ["100.64.124.82"], "ip": ["tcp:443"]}
```

Keep any other needed policy sections, but remove or narrow every rule that
would also grant him port 8443 on this desktop. Tailscale grants combine
permissions; there is no later deny rule that cancels a broader allow. Validate
the policy with Tailscale's editor preview and, when possible, attempt the 8443
URL from a device signed in as each person. Your device should connect; his
should fail to connect. The existing Open WebUI mapping on HTTPS 443 must keep
working.

To check the configured Serve ports without resetting the existing mapping:

```powershell
& 'C:\Program Files\Tailscale\tailscale.exe' serve status
```

The status must show both 443 to `localhost:3000` (Open WebUI) and 8443
to `127.0.0.1:5001` (grader), each marked tailnet only. The `--bg` setting
keeps Serve configured across restarts. Never use `tailscale funnel` or
`tailscale serve reset` for this setup.

References: [Tailscale Serve](https://tailscale.com/docs/features/tailscale-serve),
[Tailscale grants](https://tailscale.com/docs/reference/syntax/grants), and
[LM Studio server startup](https://lmstudio.ai/docs/developer/core/headless).
