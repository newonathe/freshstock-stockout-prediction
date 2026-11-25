"""FreshStock 2.0 experiment stages on a small synthetic source."""
import copy
import datetime as dt
import hashlib
import io
import json
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from freshstock import data as b3
from freshstock.expanded import data, experiment, metrics

N = 40
THREADS = {"forest_n_jobs": 2, "blas_threads": 1}


def write_source(path, name, first, days, seed):
    rng = np.random.default_rng(seed)
    hours = np.zeros((N, days, 24), dtype=np.int64)
    for k in range(N):  # sticky stockout streaks, so that history is informative
        state = 0
        for d in range(days):
            state = int(rng.random() < (.75 if state else .2))
            if state:
                hours[k, d, rng.integers(6, 22, rng.integers(1, 9))] = 1
    rows = [(k, d) for k in range(N) for d in range(days)]
    columns = {"city_id": [k % 4 for k, _ in rows], "store_id": [k for k, _ in rows], "product_id": [0] * len(rows),
               "dt": [(first + dt.timedelta(d)).isoformat() for _, d in rows],
               "sale_amount": rng.random(len(rows)).tolist(),
               "stock_hour6_22_cnt": [int(hours[k, d, 6:22].sum()) for k, d in rows],
               "hours_stock_status": [hours[k, d].tolist() for k, d in rows]}
    schema = pa.schema([("city_id", pa.int64()), ("store_id", pa.int64()), ("product_id", pa.int64()),
                        ("dt", pa.string()), ("sale_amount", pa.float64()), ("stock_hour6_22_cnt", pa.int32()),
                        ("hours_stock_status", pa.list_(pa.int64()))])
    pq.write_table(pa.table(columns, schema=schema), path)
    raw = path.read_bytes()
    return data.Source(name, path, hashlib.sha256(raw).hexdigest(), len(raw), first, days, N)


SMALL = {"full": {"logistic": [{"C": .1}, {"C": 1.}], "forest": [
    {"n_estimators": 5, "max_depth": 2, "min_samples_leaf": 5}, {"n_estimators": 8, "max_depth": 4,
                                                                  "min_samples_leaf": 5}]},
         "reduced": {"logistic": [{"C": 1.}], "forest": [{"n_estimators": 5, "max_depth": 2, "min_samples_leaf": 5}]}}


class Workspace(unittest.TestCase):
    """One frozen run per class; tests that change it work on a copy."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)
        cls.train = write_source(cls.root / "train.parquet", "train", dt.date(2024, 3, 28), 90, 1)
        cls.holdout = write_source(cls.root / "eval.parquet", "holdout", dt.date(2024, 6, 26), 7, 2)
        patch = mock.patch.dict(experiment.GRIDS, SMALL, clear=True)
        patch.start()
        cls.addClassCleanup(patch.stop)
        cls.protocol = cls.root / "protocol.json"
        experiment.write_json(cls.protocol, experiment.build_protocol(
            cls.train, cls.holdout, N, "full", "sigmoid", THREADS, {"tier": [N, "full"]}))
        cls.opened = []
        original = data.read_panel

        def spy(source, *args, **kwargs):
            cls.opened.append(source.path)
            return original(source, *args, **kwargs)

        with mock.patch.object(data, "read_panel", spy):
            cls.run_dir = experiment.fit(cls.protocol, cls.root / "runs", cls.train)
        cls.freeze_sha = b3.sha256_file(cls.run_dir / "freeze.json")

    def copy_run(self):
        folder = Path(tempfile.mkdtemp(dir=self.root)) / self.run_dir.name
        shutil.copytree(self.run_dir, folder)
        return folder

    def evaluate(self, run, **kwargs):
        return experiment.evaluate(run, kwargs.pop("sha", self.freeze_sha), self.train, self.holdout, **kwargs)

    def refused(self, text, call, *args, **kwargs):
        with self.assertRaises(b3.DataError) as caught:
            call(*args, **kwargs)
        self.assertIn(text, str(caught.exception))


class GridTests(unittest.TestCase):
    def test_full_grid_is_the_locked_grid(self):
        self.assertEqual(experiment.GRIDS["full"], {
            "logistic": [{"C": .1}, {"C": 1.}, {"C": 10.}],
            "forest": [{"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 5},
                       {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 10},
                       {"n_estimators": 150, "max_depth": 16, "min_samples_leaf": 20}]})
        self.assertEqual(experiment.GRIDS["reduced"], {
            "logistic": [{"C": 1.}], "forest": [{"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 5}]})
        self.assertEqual(len(experiment.candidates("full")), 12)
        self.assertEqual(len(experiment.candidates("reduced")), 4)
        self.assertEqual(len({c["id"] for c in experiment.candidates("full")}), 12)

    def test_pipelines(self):
        lr = experiment.make_pipeline("logistic", "B", {"C": 10.}, 1)
        p = lr.named_steps["model"].get_params()
        self.assertEqual([p[k] for k in ("C", "solver", "max_iter", "tol", "fit_intercept", "class_weight",
                                         "random_state", "l1_ratio")],
                         [10., "lbfgs", 1000, 1e-4, True, None, 111, 0.0])
        self.assertEqual(experiment.make_pipeline("forest", "A", {"n_estimators": 3, "max_depth": 2,
                                                                  "min_samples_leaf": 5}, 4).steps[0][0], "model")
        p = experiment.make_pipeline("forest", "A", {"n_estimators": 3, "max_depth": 2, "min_samples_leaf": 5},
                                     4).named_steps["model"].get_params()
        self.assertEqual([p[k] for k in ("criterion", "max_features", "bootstrap", "min_samples_split", "class_weight",
                                         "random_state", "n_jobs")], ["gini", "sqrt", True, 2, None, 111, 4])

    def test_only_numeric_columns_are_scaled_with_fit_rows(self):
        rng = np.random.default_rng(0)
        for s in ("A", "B"):
            k = data.N_NUMERIC[s]
            x = np.hstack([rng.normal(5, 3, (200, k)), np.eye(7)[rng.integers(0, 7, 200)]])
            pipe = experiment.make_pipeline("logistic", s, {"C": 1.}, 1).fit(x, rng.integers(0, 2, 200))
            z = pipe[:-1].transform(x)
            np.testing.assert_allclose(z[:, :k].mean(axis=0), 0, atol=1e-9)
            np.testing.assert_allclose(z[:, :k].std(axis=0), 1, atol=1e-9)
            np.testing.assert_array_equal(z[:, k:], x[:, k:])
            np.testing.assert_allclose(pipe[:-1].transform(x + 1)[:, 0].mean() > .01, True)

    def test_selection_prefers_a_then_simpler_within_the_tie_limit(self):
        def result(s, family, params, ap):
            return {"id": f"{s}{family}{params}", "feature_set": s, "family": family, "params": params,
                    "tuning_average_precision": ap}
        big = {"n_estimators": 150, "max_depth": 16, "min_samples_leaf": 20}
        mid = {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 10}
        small = {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 5}
        leaf = {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 9}
        lr = [result("A", "logistic", {"C": 1.}, .5)]
        for rows, expected in (
                ([result("A", "forest", big, .7), result("B", "forest", small, .7000005)], ("A", big)),
                ([result("A", "forest", big, .7), result("B", "forest", small, .7000011)], ("B", small)),
                ([result("A", "forest", big, .7), result("A", "forest", mid, .7), result("A", "forest", small, .7)],
                 ("A", small)),
                ([result("A", "forest", small, .7), result("A", "forest", leaf, .7)], ("A", leaf))):
            got = experiment.select(rows + lr)
            self.assertEqual((got["forest"]["feature_set"], got["forest"]["params"]), expected)
        got = experiment.select(lr + [result("A", "logistic", {"C": .1}, .5 - 9e-7), result("A", "forest", big, .7)])
        self.assertEqual(got["logistic"]["params"], {"C": .1})


class FitTests(Workspace):
    def test_fit_opened_only_the_training_source(self):
        self.assertEqual(set(self.opened), {self.train.path})

    def test_fit_works_when_the_holdout_file_does_not_exist(self):
        away = self.holdout.path.with_name("moved.parquet")
        self.holdout.path.rename(away)
        try:
            experiment.fit(self.protocol, self.root / "runs_without_holdout", self.train)
        finally:
            away.rename(self.holdout.path)

    def test_frozen_files(self):
        self.assertEqual({p.name for p in self.run_dir.iterdir()}, set(experiment.FROZEN_FILES))
        freeze = experiment.read_json(self.run_dir / "freeze.json")
        self.assertEqual(freeze["status"], "FROZEN")
        for name, digest in freeze["artifacts"].items():
            self.assertEqual(b3.sha256_file(self.run_dir / name), digest)
        self.assertEqual(set(freeze["artifacts"]), set(experiment.FROZEN_FILES) - {"freeze.json"})
        self.assertEqual(freeze["rows"], {"initial_fit": 41 * N, "tuning": 7 * N, "refit": 48 * N,
                                          "calibration": 7 * N, "threshold": 7 * N})
        panel = data.read_panel(self.train)
        self.assertEqual(freeze["positives"], {name: int(data.split(panel, name)[1].sum()) for name in freeze["rows"]})
        text = json.dumps(freeze)
        self.assertNotIn(str(self.root), text)
        self.assertNotIn("Users", text)

    def test_every_candidate_is_reported(self):
        table = experiment.read_json(self.run_dir / "candidate_metrics.json")
        self.assertEqual([c["id"] for c in table["candidates"]], [c["id"] for c in experiment.candidates("full")])
        for c in table["candidates"]:
            self.assertTrue(0 < c["tuning_average_precision"] < 1)
            self.assertGreaterEqual(c["fit_seconds"], 0)
        self.assertEqual(set(table["same_feature_comparison"]), {"A", "B"})
        self.assertEqual(table["selected"], {f: experiment.select(table["candidates"])[f]["id"]
                                             for f in ("logistic", "forest")})

    def test_tuning_scores_come_from_the_initial_fit_dates_only(self):
        table = experiment.read_json(self.run_dir / "candidate_metrics.json")
        panel = data.read_panel(self.train)
        for c in table["candidates"]:
            x, y = data.split(panel, "initial_fit", c["feature_set"])
            xt, yt = data.split(panel, "tuning", c["feature_set"])
            model = experiment.make_pipeline(c["family"], c["feature_set"], c["params"], 1).fit(x, y)
            self.assertAlmostEqual(metrics.average_precision(yt, model.predict_proba(xt)[:, 1]),
                                   c["tuning_average_precision"], delta=1e-12)

    def test_final_base_models_use_the_refit_dates(self):
        freeze = experiment.read_json(self.run_dir / "freeze.json")
        panel = data.read_panel(self.train)
        _, models = experiment.load_run(self.run_dir, self.freeze_sha)
        for family in ("logistic", "forest"):
            chosen = freeze["selected"][family]
            x, y = data.split(panel, "refit", chosen["feature_set"])
            ours = experiment.base_pipeline(models[family])
            again = experiment.make_pipeline(family, chosen["feature_set"], chosen["params"], 1).fit(x, y)
            np.testing.assert_allclose(ours.predict_proba(x[:500]), again.predict_proba(x[:500]), rtol=0, atol=1e-10)
            self.assertEqual(ours.n_features_in_, len(data.FEATURES[chosen["feature_set"]]))

    def test_calibration_choice_and_thresholds_can_be_recomputed(self):
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.frozen import FrozenEstimator
        from freshstock.experiment import choose_threshold
        freeze = experiment.read_json(self.run_dir / "freeze.json")
        panel = data.read_panel(self.train)
        _, models = experiment.load_run(self.run_dir, self.freeze_sha)
        for family in ("logistic", "forest"):
            s = freeze["selected"][family]["feature_set"]
            base = experiment.base_pipeline(models[family])
            xc, yc = data.split(panel, "calibration", s)
            xt, yt = data.split(panel, "threshold", s)
            sigmoid = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid", ensemble=False).fit(xc, yc)
            raw, cal = base.predict_proba(xt)[:, 1], sigmoid.predict_proba(xt)[:, 1]
            record = freeze["calibration"][family]
            self.assertAlmostEqual(record["raw"]["brier"], metrics.brier(yt, raw), delta=1e-12)
            self.assertAlmostEqual(record["sigmoid"]["brier"], metrics.brier(yt, cal), delta=1e-12)
            better = metrics.brier(yt, cal) < metrics.brier(yt, raw)
            self.assertEqual(record["chosen"], "sigmoid" if better else "raw")
            score = cal if better else raw
            np.testing.assert_allclose(models[family].predict_proba(xt)[:, 1], score, rtol=0, atol=1e-12)
            self.assertEqual(freeze["thresholds"][family], choose_threshold(yt, score)[0])
            self.assertTrue(record["base_unchanged"])

    def test_same_protocol_cannot_be_fitted_again(self):
        self.refused("already", experiment.fit, self.protocol, self.root / "runs", self.train)

    def test_failure_is_recorded_and_not_retried(self):
        runs = self.root / "runs_failing"
        with mock.patch.object(experiment, "make_pipeline", side_effect=MemoryError("no memory")):
            with self.assertRaises(MemoryError):
                experiment.fit(self.protocol, runs, self.train)
        failed = list(runs.iterdir())
        self.assertEqual(len(failed), 1)
        self.assertTrue(failed[0].name.endswith(".failed"))
        self.assertIn("MemoryError", (failed[0] / "FAILED.json").read_text())
        self.refused("already", experiment.fit, self.protocol, runs, self.train)

    def test_convergence_warning_aborts(self):
        original = experiment.make_pipeline

        def short(*args):
            pipe = original(*args)
            if args[0] == "logistic":
                pipe.named_steps["model"].set_params(max_iter=1)
            return pipe

        with mock.patch.object(experiment, "make_pipeline", short):
            with self.assertRaises(Warning):
                experiment.fit(self.protocol, self.root / "runs_warn", self.train)

    def test_changed_protocol_is_refused(self):
        good = experiment.read_json(self.protocol)
        for key, value in (("seed", 1), ("n_series", 10), ("cohort_digest", "0"), ("grid", "reduced"),
                           ("splits", {}), ("code_sha256", {}), ("features", {}), ("environment", {}),
                           ("threshold_candidates", [.5])):
            bad = copy.deepcopy(good)
            bad[key] = value
            path = self.root / "bad.json"
            experiment.write_json(path, bad)
            self.refused("protocol", experiment.fit, path, self.root / "runs_bad", self.train)

    def test_labels_and_ids_are_not_features(self):
        for names in data.FEATURES.values():
            for name in ("city_id", "store_id", "product_id", "dt", "stock_hour6_22_cnt", "y"):
                self.assertNotIn(name, names)


class CalibrationFailureTests(Workspace):
    def test_raw_fallback_is_logged(self):
        with mock.patch.object(experiment, "fit_sigmoid", side_effect=ValueError("broken")):
            run = experiment.fit(self.protocol, self.root / "runs_raw", self.train)
        freeze = experiment.read_json(run / "freeze.json")
        for family in ("logistic", "forest"):
            self.assertEqual(freeze["calibration"][family]["chosen"], "raw")
            self.assertIn("broken", freeze["calibration"][family]["failure"])

    def test_calibration_off(self):
        path = self.root / "off.json"
        experiment.write_json(path, experiment.build_protocol(self.train, self.holdout, N, "reduced", "off", THREADS,
                                                              {}))
        run = experiment.fit(path, self.root / "runs_off", self.train)
        freeze = experiment.read_json(run / "freeze.json")
        self.assertEqual({freeze["calibration"][f]["chosen"] for f in ("logistic", "forest")}, {"raw"})
        self.assertIsNone(freeze["calibration"]["forest"]["sigmoid"])


class LoadTests(Workspace):
    def test_models_come_from_verified_bytes(self):
        run = self.copy_run()
        seen, original = [], joblib.load

        def spy(stream):
            seen.append(stream)
            return original(stream)

        with mock.patch.object(experiment.joblib, "load", spy):
            experiment.load_run(run, self.freeze_sha)
        self.assertTrue(all(isinstance(s, io.BytesIO) for s in seen) and len(seen) == 2)

    def test_every_change_is_refused_before_any_pickle(self):
        def edit(name):
            return lambda run: (run / name).write_bytes((run / name).read_bytes() + b" ")
        cases = [("freeze", edit("freeze.json")), ("protocol", edit("protocol.json")),
                 ("logistic.joblib", edit("logistic.joblib")), ("forest.joblib", edit("forest.joblib")),
                 ("candidate_metrics", edit("candidate_metrics.json")), ("resource", edit("resource.json")),
                 ("extra", lambda run: (run / "extra.joblib").write_bytes(b"x")),
                 ("folder", lambda run: (run / "sub").mkdir()),
                 ("missing", lambda run: (run / "forest.joblib").unlink())]
        for name, change in cases:
            run = self.copy_run()
            change(run)
            with mock.patch.object(experiment.joblib, "load", side_effect=AssertionError("unpickled")):
                with self.assertRaises(b3.DataError, msg=name):
                    experiment.load_run(run, self.freeze_sha)

    def test_wrong_pin_or_library_is_refused(self):
        self.refused("freeze", experiment.load_run, self.run_dir, "0" * 64)
        with mock.patch.object(experiment, "version", lambda name: "0.0"):
            self.refused("required", experiment.load_run, self.run_dir, self.freeze_sha)


class EvaluateTests(Workspace):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.done = Path(tempfile.mkdtemp(dir=cls.root)) / cls.run_dir.name
        shutil.copytree(cls.run_dir, cls.done)
        cls.result = experiment.evaluate(cls.done, cls.freeze_sha, cls.train, cls.holdout)

    def test_files_and_marker(self):
        self.assertEqual({p.name for p in self.done.iterdir()},
                         set(experiment.FROZEN_FILES) | set(experiment.EVALUATED_FILES))
        marker = experiment.read_json(self.done / "evaluation.json")
        self.assertEqual(marker["status"], "EVALUATED")
        self.assertEqual(marker["freeze_sha256"], self.freeze_sha)
        for name, digest in marker["artifacts"].items():
            self.assertEqual(b3.sha256_file(self.done / name), digest)
        self.assertEqual(len(list((self.done / "days").iterdir())), 21)

    def test_saved_rows_are_complete_and_ordered(self):
        rows = pd.read_parquet(self.done / "predictions.parquet")
        self.assertEqual(list(rows.columns), experiment.PREDICTION_COLUMNS)
        self.assertEqual(len(rows), 7 * N)
        self.assertFalse(rows.duplicated(["dt"] + b3.KEY).any())
        self.assertTrue(rows[["dt"] + b3.KEY].equals(
            rows[["dt"] + b3.KEY].sort_values(["dt"] + b3.KEY).reset_index(drop=True)))
        self.assertEqual((rows["dt"].min(), rows["dt"].max()), ("2024-06-26", "2024-07-02"))
        self.assertEqual({str(rows[c].dtype) for c in ("logistic_score", "forest_score")}, {"float64"})
        self.assertEqual({str(rows[c].dtype) for c in ("y_true", "logistic_flag", "forest_flag", "persistence_flag",
                                                       "past_stockout")}, {"uint8"})

    def test_scores_flags_and_outcomes_match_an_independent_path(self):
        rows = pd.read_parquet(self.done / "predictions.parquet")
        whole = data.read_panel(self.train).join(data.read_panel(self.holdout))
        freeze, models = experiment.load_run(self.run_dir, self.freeze_sha)
        for i, day in enumerate(data.split_days("holdout")):
            part = rows.iloc[i * N:(i + 1) * N]
            at = whole.index(day)
            np.testing.assert_array_equal(part["y_true"], whole.stock[:, at] > 0)
            np.testing.assert_array_equal(part["past_stockout"], whole.stock[:, at - 1] > 0)
            np.testing.assert_array_equal(part["persistence_flag"], part["past_stockout"])
            for f in ("logistic", "forest"):
                x = data.features_on(whole, day)[:, data.COLUMNS[freeze["selected"][f]["feature_set"]]]
                np.testing.assert_allclose(part[f"{f}_score"], models[f].predict_proba(x)[:, 1], rtol=0, atol=1e-10)
                np.testing.assert_array_equal(part[f"{f}_flag"], part[f"{f}_score"] >= freeze["thresholds"][f])

    def test_metrics_equal_a_recomputation_from_saved_rows(self):
        saved = experiment.read_json(self.done / "metrics.json")
        again = metrics.summarize(experiment.load_predictions(self.done / "predictions.parquet"))
        self.assertEqual(saved["summary"], json.loads(json.dumps(again)))
        self.assertTrue(experiment.verify(self.done, self.freeze_sha)["metrics_recomputed"])

    def test_second_evaluation_is_refused(self):
        self.refused("already evaluated", self.evaluate, self.done)

    def test_verify_detects_a_changed_result(self):
        run = Path(tempfile.mkdtemp(dir=self.root)) / self.done.name
        shutil.copytree(self.done, run)
        (run / "metrics.json").write_bytes((run / "metrics.json").read_bytes() + b" ")
        self.refused("metrics.json", experiment.verify, run, self.freeze_sha)

    def test_scoring_never_sees_the_target_day(self):
        run, seen = self.copy_run(), []
        original = data.features_on

        def spy(panel, day):
            seen.append((int(panel.dates[-1]), day.toordinal()))
            return original(panel, day)

        with mock.patch.object(data, "features_on", spy):
            self.evaluate(run)
        self.assertEqual(len(seen), 7)
        self.assertTrue(all(last == day - 1 for last, day in seen))

    def test_interrupted_day_is_quarantined_and_never_rescored(self):
        run, calls = self.copy_run(), []
        original = experiment.score_day

        def crash(*args):
            calls.append(1)
            if len(calls) == 3:
                raise KeyboardInterrupt
            return original(*args)

        with mock.patch.object(experiment, "score_day", crash), self.assertRaises(KeyboardInterrupt):
            self.evaluate(run)
        with mock.patch.object(experiment, "score_day", side_effect=AssertionError("rescored")):
            self.refused("quarantined", self.evaluate, run)
            self.refused("quarantined", self.evaluate, run)
        self.assertTrue((run / "evaluation_quarantined.json").is_file())
        self.assertFalse((run / "metrics.json").exists())

    def test_crash_between_days_resumes_only_the_days_never_started(self):
        run, calls, scored = self.copy_run(), [], []
        write, score = experiment.write_intent, experiment.score_day

        def crash(*args):
            calls.append(1)
            if len(calls) == 4:
                raise KeyboardInterrupt
            return write(*args)

        with mock.patch.object(experiment, "write_intent", crash), self.assertRaises(KeyboardInterrupt):
            self.evaluate(run)

        def count(*args):
            scored.append(1)
            return score(*args)

        with mock.patch.object(experiment, "score_day", count):
            self.evaluate(run)
        self.assertEqual(len(scored), 4)
        self.assertEqual((run / "metrics.json").read_bytes(), (self.done / "metrics.json").read_bytes())

    def test_changed_chunk_is_quarantined(self):
        run, calls = self.copy_run(), []
        write = experiment.write_intent

        def crash(*args):
            calls.append(1)
            if len(calls) == 3:
                raise KeyboardInterrupt
            return write(*args)

        with mock.patch.object(experiment, "write_intent", crash), self.assertRaises(KeyboardInterrupt):
            self.evaluate(run)
        chunk = run / "days" / "2024-06-26.npz"
        chunk.write_bytes(chunk.read_bytes() + b"x")
        self.refused("quarantined", self.evaluate, run)

    def test_wrong_sources_code_or_pin_are_refused_without_a_marker(self):
        run = self.copy_run()
        self.refused("freeze", self.evaluate, run, sha="0" * 64)
        other = write_source(self.root / "other.parquet", "holdout", dt.date(2024, 6, 26), 7, 3)
        self.refused("sources", experiment.evaluate, run, self.freeze_sha, self.train, other)
        with mock.patch.object(experiment, "code_hashes", lambda: {"x": "y"}):
            self.refused("code", self.evaluate, run)
        self.assertFalse((run / "evaluation_started.json").exists())

    def test_scoring_refuses_history_that_does_not_end_the_day_before(self):
        freeze, models = experiment.load_run(self.run_dir, self.freeze_sha)
        dev = data.read_panel(self.train)
        for day in (dt.date(2024, 6, 20), dt.date(2024, 6, 25)):
            self.refused("day before", experiment.score_day, models, freeze, dev, day)

    def test_holdout_outcomes_are_not_needed_for_scoring(self):
        """Scores for the first day do not change when every holdout value changes."""
        freeze, models = experiment.load_run(self.run_dir, self.freeze_sha)
        dev = data.read_panel(self.train)
        chunk = experiment.score_day(models, freeze, dev, dt.date(2024, 6, 26))
        rows = pd.read_parquet(self.done / "predictions.parquet").iloc[:N]
        np.testing.assert_allclose(chunk["forest_score"], rows["forest_score"], rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
