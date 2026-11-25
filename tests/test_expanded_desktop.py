"""FreshStock 2.0 entry point: self-test of all seven days, fail closed."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import freshstock_desktop_2 as desktop
from freshstock import data as b3
from freshstock.expanded import metrics
from tests.test_expanded_bundle import BundleCase
from tests.test_expanded_gui import HAS_TK


class PathTests(unittest.TestCase):
    def test_bundle_is_found_beside_the_source_or_inside_the_program(self):
        self.assertEqual(desktop.bundle_root(), b3.ROOT / "bundle")
        with mock.patch.object(sys, "frozen", True, create=True), \
                mock.patch.object(sys, "_MEIPASS", "C:/Program Folder/_internal", create=True):
            self.assertEqual(desktop.bundle_root(), Path("C:/Program Folder/_internal/bundle"))

    def test_reports_have_their_own_folder(self):
        with mock.patch.dict(os.environ, {"LOCALAPPDATA": "C:/Users/x/AppData/Local"}):
            self.assertEqual(desktop.reports_dir(), Path("C:/Users/x/AppData/Local/FreshStock2/reports"))


@unittest.skipUnless(HAS_TK, "needs a display")
class SelfTestTests(BundleCase):
    def run_self_test(self, folder, pin):
        out = Path(tempfile.mkdtemp(dir=self.root)) / "folder with spaces" / "receipt.json"
        with mock.patch.object(desktop, "bundle_root", lambda: folder), mock.patch.object(desktop, "PIN", pin):
            code = desktop.main(["--self-test", str(out)])
        return code, json.loads(out.read_text(encoding="utf-8")), out

    def test_all_seven_days_and_export(self):
        code, receipt, out = self.run_self_test(self.folder, self.pin)
        self.assertEqual((code, receipt["status"], receipt["problem"]), (0, "ready", None))
        self.assertEqual((receipt["run"], receipt["days_replayed"]), (self.run_dir.name, 7))
        summary = self.loaded.metrics["summary"]
        for e in metrics.ESTIMATORS:
            for count in ("tp", "fp", "fn", "tn"):
                self.assertEqual(receipt["counts"][e][count], summary["estimators"][e][count])
        self.assertLessEqual(receipt["max_score_difference"], 1e-10)
        self.assertTrue(receipt["exported_rows"] == self.loaded.n and receipt["network_blocked"])
        self.assertTrue(list(out.parent.rglob("ranking.csv")))

    def test_damaged_bundle_exits_1(self):
        folder = self.copy_bundle()
        (folder / "scores.npz").write_bytes(b"x")
        code, receipt, _ = self.run_self_test(folder, self.pin)
        self.assertEqual((code, receipt["status"]), (1, "blocked"))
        self.assertIn("scores.npz", receipt["problem"])

    def test_any_error_exits_1(self):
        with mock.patch.object(desktop.bd, "reveal", side_effect=RuntimeError("boom")):
            code, receipt, _ = self.run_self_test(self.folder, self.pin)
        self.assertEqual(code, 1)
        self.assertIn("boom", receipt["problem"])


if __name__ == "__main__":
    unittest.main()
