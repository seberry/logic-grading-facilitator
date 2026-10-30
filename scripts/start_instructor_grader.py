"""Start the instructor grading app and open the Setup Wizard.

This is the human-facing launcher for the instructor's ordinary weekly workflow.
The app itself loads DATA_DIR, DATA_ARCHIVE_DIR, and ENV_MODE from the repo's
ignored .env file.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

import uvicorn


HOST = "127.0.0.1"
DEFAULT_PORT = 5001
PORT_SEARCH_LIMIT = 20


def _setup_url(port: int) -> str:
    return f"http://{HOST}:{port}/setup"


def _is_grader_running(port: int) -> bool:
    try:
        with urllib.request.urlopen(_setup_url(port), timeout=0.8) as response:
            body = response.read(4096).decode("utf-8", errors="ignore")
    except (OSError, urllib.error.URLError):
        return False
    return "Logic Grading Facilitator" in body and "Setup Wizard" in body


def _port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((HOST, port))
        except OSError:
            return False
    return True


def _choose_port(preferred: int) -> tuple[int, bool]:
    """Return (port, app_already_running)."""
    # Recognize the old documented port too, so a prior command-line launch is
    # reopened instead of silently creating a second server over the same data.
    app_candidates = [preferred, *range(5000, preferred + PORT_SEARCH_LIMIT)]
    for port in dict.fromkeys(app_candidates):
        if _is_grader_running(port):
            return port, True
    for port in range(preferred, preferred + PORT_SEARCH_LIMIT):
        if _port_is_available(port):
            return port, False
    raise RuntimeError(
        f"Could not find an open port from {preferred} through "
        f"{preferred + PORT_SEARCH_LIMIT - 1}."
    )


def _open_when_ready(url: str) -> None:
    for _ in range(100):
        try:
            with urllib.request.urlopen(url, timeout=0.5):
                webbrowser.open(url)
                return
        except (OSError, urllib.error.URLError):
            time.sleep(0.1)
    print(f"The browser did not open automatically. Open this address: {url}")


def _ensure_lmstudio_server() -> None:
    try:
        with urllib.request.urlopen("http://127.0.0.1:1234/v1/models", timeout=5):
            return
    except (OSError, urllib.error.URLError):
        pass

    lms = shutil.which("lms") or str(Path.home() / ".lmstudio" / "bin" / "lms.exe")
    if not Path(lms).is_file():
        print("LM Studio CLI was not found; start its local server manually.")
        return
    try:
        subprocess.run(
            [lms, "server", "start", "--port", "1234", "--bind", "127.0.0.1"],
            check=True,
            timeout=60,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"LM Studio did not start: {exc}. The grader UI will still start.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Open the instructor grading Setup Wizard.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--fixed-port",
        action="store_true",
        help="Use exactly --port; fail if another program owns it (for Tailscale Serve).",
    )
    parser.add_argument(
        "--ensure-lmstudio",
        action="store_true",
        help="Start the local LM Studio API on 127.0.0.1:1234 if needed.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start the server without opening a browser (useful for testing).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))
    env_path = repo_root / ".env"
    if not env_path.is_file():
        raise RuntimeError(
            f"Missing {env_path}. Create it with DATA_DIR, DATA_ARCHIVE_DIR, "
            "and ENV_MODE=PRODUCTION before grading real submissions."
        )

    from dotenv import load_dotenv
    load_dotenv(env_path)
    if args.ensure_lmstudio and os.getenv("GRADING_VLM_PROVIDER", "reallms").strip().lower() == "lmstudio":
        _ensure_lmstudio_server()

    if args.fixed_port:
        port = args.port
        already_running = _is_grader_running(port)
        if not already_running and not _port_is_available(port):
            raise RuntimeError(f"Port {port} is busy, so the grader cannot start there.")
    else:
        port, already_running = _choose_port(args.port)
    url = _setup_url(port)
    if already_running:
        print(f"The grader is already running. Opening {url}")
        if not args.no_browser:
            webbrowser.open(url)
        return 0

    print("=" * 64)
    print("Logic Grading Facilitator - Instructor")
    print(f"Setup Wizard: {url}")
    if port != args.port:
        print(f"Port {args.port} was busy, so this run is using port {port}.")
    print("Keep this window open while grading. Close it when finished.")
    print("=" * 64)

    if not args.no_browser:
        threading.Thread(target=_open_when_ready, args=(url,), daemon=True).start()

    try:
        uvicorn.run("app.server:app", host=HOST, port=port, reload=False, access_log=False)
    except KeyboardInterrupt:
        # Windows can deliver Ctrl+C through both the console and a wrapping
        # shell. Treat that as an ordinary shutdown, not a startup failure.
        pass
    print("Grader stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
