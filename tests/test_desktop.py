"""M3: the desktop entry point used by the packaged program."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import freshstock_desktop as desktop
from freshstock import data, portfolio as pf
from tests.test_gui import HAS_BUNDLE, HAS_TK


class PathTests(unittest.TestCase):
    def test_source_run_uses_the_project_folder(self):
        self.assertEqual(desktop.bundle_root(), data.ROOT)

    def test_packaged_run_uses_its_own_resources_not_the_working_folder(self):
        with mock.patch.object(sys, "frozen", True, create=True), \
                mock.patch.object(sys, "_MEIPASS", "C:/Program Folder/_internal", create=True):
            self.assertEqual(desktop.bundle_root(), Path("C:/Program Folder/_internal"))

    def test_reports_go_to_the_user_profile(self):
        with mock.patch.dict(os.environ, {"LOCALAPPDATA": "C:/Users/x/AppData/Local"}):
            self.assertEqual(desktop.reports_dir(), Path("C:/Users/x/AppData/Local/FreshStock/reports"))


@unittest.skipUnless(HAS_BUNDLE and HAS_TK, "needs the frozen run and a display")
class SelfTestTests(unittest.TestCase):
    def test_self_test_drives_the_real_app_and_writes_a_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "folder with spaces" / "receipt.json"
            code = desktop.main(["--self-test", str(out)])
            receipt = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue((out.parent / "replay-2024-06-18.html").is_file())
        self.assertEqual(code, 0)
        self.assertEqual(receipt["status"], "ready")
        self.assertEqual(receipt["run"], pf.RUN_ID)
        self.assertEqual(receipt["summary"], {"logistic": {"tp": 30, "fn": 15, "fp": 24, "tn": 31},
                                              "forest": {"tp": 36, "fn": 9, "fp": 36, "tn": 19},
                                              "persistence": {"tp": 26, "fn": 19, "fp": 19, "tn": 36}})
        self.assertEqual(receipt["days_replayed"], 15)

    def test_self_test_reports_a_damaged_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "receipt.json"
            with mock.patch.object(desktop, "bundle_root", lambda: Path(tmp)):
                code = desktop.main(["--self-test", str(out)])
            receipt = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual((code, receipt["status"]), (1, "blocked"))
        self.assertIn("missing", receipt["message"])


if __name__ == "__main__":
    unittest.main()
