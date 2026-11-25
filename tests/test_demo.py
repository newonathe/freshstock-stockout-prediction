"""FS-09 and FS-10: history-only replay, offline report and command line behaviour."""
import contextlib
import io
import re
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from freshstock import data, demo, experiment as ex
from freshstock.__main__ import main

DAY = "2024-05-30"  # a validation date, so these tests never score the final test split


@unittest.skipUnless((data.DERIVED / "cohort.csv").is_file(), "run: python -m freshstock audit")
class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cohort, cls.receipt = data.load_cohort()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.runs = Path(cls.tmp.name) / "runs"
        cls.run_dir = ex.train(cls.cohort, cls.receipt, cls.runs)
        cls.metrics = ex.evaluate(cls.run_dir, cls.cohort, split="validation")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def cli(self, *args, answers=()):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
                mock.patch("builtins.input", side_effect=list(answers)):
            code = main([*args, "--runs", str(self.runs), "--reports", str(Path(self.tmp.name) / "reports")])
        return code, out.getvalue(), err.getvalue()

    def test_forecast_uses_history_only(self):
        manifest, models = ex.load_run(self.run_dir)
        history = self.cohort[self.cohort["dt"] < DAY]
        table = demo.forecast(models, manifest["thresholds"], history, DAY)
        self.assertEqual(len(table), 100)
        self.assertNotIn("y_true", table.columns)
        with self.assertRaisesRegex(data.DataError, "on or after"):
            demo.forecast(models, manifest["thresholds"], self.cohort, DAY)

    def test_replay_reproduces_frozen_predictions_and_outcomes(self):
        r = demo.replay(self.run_dir, self.cohort, DAY)
        frozen = ex.load_evaluation(self.run_dir)[1]
        frozen = frozen[frozen["dt"] == DAY].set_index(data.KEY)["y_true"]
        rows = r["table"].set_index(data.KEY)
        self.assertEqual(rows["y_true"].to_dict(), frozen.to_dict())
        self.assertTrue(r["table"]["logistic_score"].is_monotonic_decreasing)

    def test_replay_can_rank_by_the_other_model(self):
        r = demo.replay(self.run_dir, self.cohort, DAY, sort="forest")
        self.assertTrue(r["table"]["forest_score"].is_monotonic_decreasing)

    def test_replay_stops_if_live_scores_differ_from_frozen_scores(self):
        with mock.patch.object(ex, "positive_scores", lambda m, x: __import__("numpy").full(len(x), 0.5)):
            with self.assertRaisesRegex(data.DataError, "frozen"):
                demo.replay(self.run_dir, self.cohort, DAY)

    def test_date_outside_the_replay_window_is_refused(self):
        for day in ("2024-07-01", "2024-03-28", "soon"):
            with self.subTest(day=day), self.assertRaisesRegex(data.DataError, "choose a date"):
                demo.replay(self.run_dir, self.cohort, day)

    def test_html_is_self_contained_and_separates_forecast_from_outcome(self):
        html = demo.render_html(demo.replay(self.run_dir, self.cohort, DAY), self.metrics)
        self.assertIsNone(re.search(r"<script|<link|src=|@import|url\(", html))
        forecast, outcome = html.split('id="outcome"')
        self.assertEqual(forecast.count('class="series"'), 100)
        self.assertEqual(outcome.count('class="series"'), 100)
        self.assertNotIn("Miss", forecast.split('id="forecast"')[1])
        self.assertIn(self.run_dir.name, html)

    def test_demo_command_prints_forecast_before_outcome_without_network(self):
        with mock.patch.object(socket, "socket", side_effect=OSError("network blocked")):
            code, out, err = self.cli("demo", "--date", DAY)
        self.assertEqual((code, err), (0, ""))
        self.assertLess(out.index("FORECAST"), out.index("OBSERVED"))
        report = Path(re.search(r"Report: (.+\.html)", out).group(1))
        self.assertTrue(report.is_file())

    def test_bad_date_exits_nonzero_without_scores(self):
        code, out, err = self.cli("demo", "--date", "2024-13-40")
        self.assertEqual(code, 1)
        self.assertNotIn("FORECAST", out)
        self.assertIn("choose a date", err)

    def test_interactive_prompt_reasks_then_reveals_after_enter(self):
        code, out, _ = self.cli("demo", answers=["nonsense", DAY, "", "q"])
        self.assertEqual(code, 0)
        self.assertIn("choose a date", out)
        self.assertIn("OBSERVED", out)

    def test_interactive_prompt_can_be_quit(self):
        code, out, _ = self.cli("demo", answers=["q"])
        self.assertEqual(code, 0)
        self.assertNotIn("FORECAST", out)

    def test_evaluate_command_refuses_to_rescore(self):
        code, _, err = self.cli("evaluate")
        self.assertEqual(code, 1)
        self.assertIn("already", err)

    def test_missing_run_gives_recovery_instruction(self):
        out, err = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["demo", "--date", DAY, "--runs", d])
        self.assertEqual(code, 1)
        self.assertIn("python -m freshstock", err.getvalue())


if __name__ == "__main__":
    unittest.main()
