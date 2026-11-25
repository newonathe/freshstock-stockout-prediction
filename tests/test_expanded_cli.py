"""FreshStock 2.0 command line and training-only pilots."""
import contextlib
import datetime as dt
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from freshstock import data as b3
from freshstock.expanded import __main__ as cli
from freshstock.expanded import data, experiment, pilot
from tests.test_expanded_experiment import N, SMALL, write_source


class CliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)
        cls.train = write_source(cls.root / "train.parquet", "train", dt.date(2024, 3, 28), 90, 1)
        cls.holdout = write_source(cls.root / "eval.parquet", "holdout", dt.date(2024, 6, 26), 7, 2)
        for target, value in ((data, {"TRAIN": cls.train, "HOLDOUT": cls.holdout}),):
            for name, source in value.items():
                patch = mock.patch.object(target, name, source)
                patch.start()
                cls.addClassCleanup(patch.stop)
        patch = mock.patch.dict(experiment.GRIDS, SMALL, clear=True)
        patch.start()
        cls.addClassCleanup(patch.stop)

    def call(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = cli.main([str(a) for a in args])
            except SystemExit as e:
                code = e.code
        return code, out.getvalue(), err.getvalue()

    def test_usage_errors_exit_2(self):
        self.assertEqual(self.call()[0], 2)
        self.assertEqual(self.call("unknown")[0], 2)
        for stage in ("fit", "evaluate", "verify", "pilot", "lock"):
            self.assertEqual(self.call(stage)[0], 2, stage)  # no implicit input, output or newest run
        self.assertEqual(self.call("--help")[0], 0)

    def test_fit_has_no_holdout_option(self):
        self.assertEqual(self.call("fit", "--protocol", "x", "--runs", "y", "--holdout", "z")[0], 2)
        self.assertNotIn("holdout", self.call("fit", "--help")[1].lower())

    def test_refusals_exit_1(self):
        for args in (("fit", "--protocol", self.root / "none.json", "--runs", self.root / "r"),
                     ("evaluate", "--run", self.root / "none", "--freeze-sha256", "0" * 64),
                     ("verify", "--run", self.root / "none", "--freeze-sha256", "0" * 64)):
            code, _, err = self.call(*args)
            self.assertEqual(code, 1, args[0])
            self.assertTrue(err.strip() and "Traceback" not in err)

    def test_all_stages(self):
        audit = self.root / "audit.json"
        self.assertEqual(self.call("audit", "--output", audit)[0], 0)
        self.assertNotIn("stock", audit.read_text())
        receipt = self.root / "pilot.json"
        self.assertEqual(self.call("pilot", "--series", 20, "--threads", "1,2", "--output", receipt)[0], 0)
        text = receipt.read_text()
        for word in ("average_precision", "f1", "accuracy", "brier", "recall"):
            self.assertNotIn(word, text)
        runs = json.loads(text)["runs"]
        self.assertTrue(all(r["seconds"] > 0 and r["parity_max_abs_diff"] <= 1e-10 and r["flags_equal"] for r in runs))
        protocol = self.root / "lock.json"
        self.assertEqual(self.call("lock", "--series", N, "--grid", "full", "--calibration", "sigmoid",
                                   "--forest-jobs", 2, "--blas-threads", 1, "--decision", receipt,
                                   "--output", protocol)[0], 0)
        code, _, err = self.call("lock", "--series", N, "--grid", "full", "--calibration", "sigmoid", "--forest-jobs",
                                 2, "--blas-threads", 1, "--decision", receipt, "--output", protocol)
        self.assertEqual(code, 1)  # a lock is never overwritten
        with mock.patch.object(cli, "FIT_TIMEOUT", 60):
            code, out, err = self.call("fit", "--protocol", protocol, "--runs", self.root / "runs")
        self.assertEqual((code, err), (0, ""))
        run = next(p for p in (self.root / "runs").iterdir() if p.is_dir())
        sha = b3.sha256_file(run / "freeze.json")
        self.assertIn(sha, out)
        self.assertEqual(self.call("evaluate", "--run", run, "--freeze-sha256", sha)[0], 0)
        self.assertEqual(self.call("verify", "--run", run, "--freeze-sha256", sha)[0], 0)
        self.assertEqual(self.call("evaluate", "--run", run, "--freeze-sha256", sha)[0], 1)

    def test_pilot_opens_only_the_training_source(self):
        opened, original = [], data.read_panel

        def spy(source, *args, **kwargs):
            opened.append(source.name)
            return original(source, *args, **kwargs)

        with mock.patch.object(data, "read_panel", spy):
            pilot.run(20, [1], self.root / "p2.json")
        self.assertEqual(set(opened), {"train"})


if __name__ == "__main__":
    unittest.main()
