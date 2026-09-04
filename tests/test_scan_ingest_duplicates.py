from __future__ import annotations

import io
import shutil
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from app import server


def _zip_bytes(entries: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries:
            zf.writestr(name, data)
    return buf.getvalue()


class ScanIngestDuplicateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("working") / "test_tmp" / f"scan_dedup_{uuid4().hex}"
        self.raw_dir = self.root / "raw_scans"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_zip_skips_byte_identical_entries_after_flattening(self) -> None:
        payload = _zip_bytes(
            [
                ("first/scan_001.jpg", b"same-scan-content"),
                ("second/scan_001.jpg", b"same-scan-content"),
            ]
        )

        saved, duplicates = server._ingest_zip_bytes(payload, self.raw_dir)

        self.assertEqual((saved, duplicates), (1, 1))
        self.assertEqual([p.name for p in self.raw_dir.iterdir()], ["scan_001.jpg"])

    def test_zip_preserves_same_name_when_content_differs(self) -> None:
        payload = _zip_bytes(
            [
                ("first/scan_001.jpg", b"first-real-scan"),
                ("second/scan_001.jpg", b"different-real-scan"),
            ]
        )

        saved, duplicates = server._ingest_zip_bytes(payload, self.raw_dir)

        self.assertEqual((saved, duplicates), (2, 0))
        self.assertEqual(
            sorted(p.name for p in self.raw_dir.iterdir()),
            ["scan_001.jpg", "scan_001_2.jpg"],
        )

    def test_existing_exact_duplicates_are_moved_to_quarantine(self) -> None:
        (self.raw_dir / "scan_001.jpg").write_bytes(b"same-scan-content")
        (self.raw_dir / "scan_001_2.jpg").write_bytes(b"same-scan-content")
        (self.raw_dir / "scan_002.jpg").write_bytes(b"different-scan-content")

        with patch.object(server, "DATA_DIR", self.root):
            moved = server._quarantine_exact_duplicate_scans(self.raw_dir)

        self.assertEqual(len(moved), 1)
        self.assertEqual(moved[0].name, "scan_001_2.jpg")
        self.assertTrue(moved[0].is_file())
        self.assertTrue((self.raw_dir / "scan_001.jpg").is_file())
        self.assertFalse((self.raw_dir / "scan_001_2.jpg").exists())
        self.assertTrue((self.raw_dir / "scan_002.jpg").is_file())


if __name__ == "__main__":
    unittest.main()
