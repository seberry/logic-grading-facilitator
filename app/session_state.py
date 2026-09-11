from __future__ import annotations

import threading
from typing import Optional

from app.models import GradingKey, Session
from app.storage import DEFAULT_SESSION_PATH, load_session, save_session

_session_lock = threading.Lock()
_session: Optional[Session] = None
_session_mtime_ns: Optional[int] = None
_segmentation_lock = threading.Lock()


def get_session() -> Session:
    global _session, _session_mtime_ns
    try:
        current_mtime_ns = DEFAULT_SESSION_PATH.stat().st_mtime_ns
    except FileNotFoundError:
        _session = Session(
            roster=[],
            grading_key=GradingKey(quiz_id="classwork", problems=[]),
            work_units=[],
            unclaimed_pages=[],
        )
        persist_session()
        return _session

    if _session is None or _session_mtime_ns != current_mtime_ns:
        _session = load_session(DEFAULT_SESSION_PATH)
        _session_mtime_ns = current_mtime_ns
    return _session


def persist_session() -> None:
    global _session, _session_mtime_ns
    if _session is None:
        return
    save_session(_session, DEFAULT_SESSION_PATH)
    _session_mtime_ns = DEFAULT_SESSION_PATH.stat().st_mtime_ns
