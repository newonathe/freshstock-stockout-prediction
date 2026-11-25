"""Read-only adapter over the frozen pilot run. Expected numbers are copied from its frozen results, not computed here."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from freshstock import data, demo, experiment as ex, portfolio as pf

BUNDLE_FILES = [data.ROOT / name for name in pf.EXPECTED]
HAS_BUNDLE = all(p.is_file() for p in BUNDLE_FILES)
JUNE18 = {"logistic": (30, 24, 15, 31), "forest": (36, 36, 9, 19), "persistence": (26, 19, 19, 36)}  # TP FP FN TN
TOTALS = {"logistic": (464, 315, 218, 503), "forest": (582, 500, 100, 318), "persistence": (418, 247, 264, 571)}


def copy_bundle(target):
    for name in pf.EXPECTED:
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(data.ROOT / name, target / name)
    return target


@unittest.skipUnless(HAS_BUNDLE, "needs the frozen run and the derived cohort")
class RefusalTests(unittest.TestCase):
    """Everything here must stop before any pickle is opened."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = copy_bundle(Path(tmp.name) / "bundle")
        self.run_dir = self.root / "runs" / pf.RUN_ID
        patch = mock.patch("joblib.load", side_effect=AssertionError("a pickle was opened"))
        patch.start()
        self.addCleanup(patch.stop)

    def refuse(self, pattern):
        with self.assertRaisesRegex(data.DataError, pattern):
            pf.load_verified_bundle(self.root)

    def test_each_missing_file_is_refused(self):
        for name in pf.EXPECTED:
            with self.subTest(name=name):
                kept = (self.root / name).read_bytes()
                (self.root / name).unlink()
                self.refuse("missing")
                (self.root / name).write_bytes(kept)

    def test_each_single_byte_change_is_refused(self):
        for name in pf.EXPECTED:
            with self.subTest(name=name):
                kept = (self.root / name).read_bytes()
                (self.root / name).write_bytes(kept[:-1] + bytes([kept[-1] ^ 1]))
                self.refuse("changed")
                (self.root / name).write_bytes(kept)

    def test_model_swapped_with_a_matching_manifest_is_still_refused(self):
        (self.run_dir / "forest.joblib").write_bytes(b"not the approved model")
        manifest = json.loads((self.run_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest["artifacts"]["forest.joblib"] = data.sha256_file(self.run_dir / "forest.joblib")
        (self.run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.refuse("changed")

    def test_extra_file_in_the_run_folder_is_refused(self):
        (self.run_dir / "extra.joblib").write_bytes(b"x")
        self.refuse("unexpected")

    def test_another_run_folder_is_never_chosen(self):
        shutil.move(self.run_dir, self.root / "runs" / "20991231T000000000000")
        self.refuse("missing")

    def test_junction_inside_the_bundle_is_refused(self):
        import _winapi
        outside = self.root.parent / "elsewhere"
        shutil.move(self.run_dir, outside)
        _winapi.CreateJunction(str(outside), str(self.run_dir))
        self.refuse("link")

    def test_manifest_naming_a_path_outside_the_run_is_refused(self):
        manifest = json.loads((self.run_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest["artifacts"]["../../outside.joblib"] = "0" * 64
        (self.run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        name = f"runs/{pf.RUN_ID}/manifest.json"
        with mock.patch.dict(pf.EXPECTED, {name: data.sha256_file(self.run_dir / "manifest.json")}):
            self.refuse("artifact list")

    def test_manifest_with_other_settings_is_refused(self):
        original = json.loads((self.run_dir / "manifest.json").read_text(encoding="utf-8"))
        name = f"runs/{pf.RUN_ID}/manifest.json"
        for key, value in (("thresholds", {"logistic": 0.5, "forest": 0.3}), ("protocol", "B3 v2"), ("seed", 7),
                           ("code_commit", "0" * 40), ("cohort_digest", "0" * 64), ("run_id", "other"),
                           ("features", ["sale_amount"])):
            with self.subTest(key=key):
                (self.run_dir / "manifest.json").write_text(json.dumps({**original, key: value}), encoding="utf-8")
                with mock.patch.dict(pf.EXPECTED, {name: data.sha256_file(self.run_dir / "manifest.json")}):
                    self.refuse(key)

    def test_library_version_mismatch_is_refused(self):
        real = ex.version
        for lib in ("scikit-learn", "numpy", "scipy", "pandas", "joblib"):
            with self.subTest(lib=lib), \
                    mock.patch.object(ex, "version", lambda n, lib=lib: "0.0" if n == lib else real(n)):
                self.refuse(lib)


@unittest.skipUnless(HAS_BUNDLE, "needs the frozen run and the derived cohort")
class BundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        before = {p: data.sha256_file(p) for p in BUNDLE_FILES}
        cls.before = before
        cls.bundle = pf.load_verified_bundle(data.ROOT)

    def test_bundle_is_the_approved_run(self):
        self.assertEqual(self.bundle.manifest["run_id"], "20261124T184117754173")
        self.assertEqual(self.bundle.manifest["thresholds"], {"logistic": 0.4, "forest": 0.3})
        self.assertEqual((len(self.bundle.test), int(self.bundle.test.y_true.sum())), (1500, 682))
        self.assertEqual((len(self.bundle.validation), int(self.bundle.validation.y_true.sum())), (1500, 535))
        self.assertEqual(self.bundle.days[0], "2024-06-11")
        self.assertEqual(self.bundle.days[-1], "2024-06-25")
        self.assertEqual(len(self.bundle.days), 15)

    def test_bundle_cannot_be_reassigned(self):
        with self.assertRaises(AttributeError):
            self.bundle.manifest = {}

    # ---------- saved prediction schema ----------

    def check_bad(self, change, pattern):
        frame = self.bundle.test.copy()
        frame = change(frame)
        with self.assertRaisesRegex(data.DataError, pattern):
            pf.check_predictions(frame, self.bundle.cohort, "test", self.bundle.manifest["thresholds"])

    def test_good_predictions_pass(self):
        for split, frame in (("test", self.bundle.test), ("validation", self.bundle.validation)):
            pf.check_predictions(frame, self.bundle.cohort, split, self.bundle.manifest["thresholds"])

    def test_bad_predictions_are_refused(self):
        def put(col, value):
            def change(f):
                f[col] = f[col].astype(object)
                f.loc[0, col] = value
                f[col] = pd.to_numeric(f[col])
                return f
            return change

        def flip(col):
            def change(f):
                f.loc[0, col] = 1 - f.loc[0, col]
                return f
            return change

        def duplicate(f):
            f.loc[0, data.KEY] = f.loc[1, data.KEY]
            return f

        cases = [(lambda f: f.drop(columns="forest_score"), "columns"), (lambda f: f.iloc[:-1], "rows"),
                 (duplicate, "duplicate"), (put("logistic_score", 1.2), "score"), (put("forest_score", -0.1), "score"),
                 (put("logistic_score", np.nan), "score"), (put("logistic_flag", 2), "binary"),
                 (flip("logistic_flag"), "threshold"), (flip("forest_flag"), "threshold"),
                 (flip("persistence_flag"), "persistence"), (flip("y_true"), "outcome"),
                 (lambda f: f.assign(dt=f["dt"].replace("2024-06-25", "2024-06-26")), "dates")]
        for i, (change, pattern) in enumerate(cases):
            with self.subTest(i=i, pattern=pattern):
                self.check_bad(change, pattern)

    # ---------- replay ----------

    def test_all_fifteen_days_replay_and_sum_to_the_final_confusion(self):
        totals = {n: np.zeros(4, dtype=int) for n in ex.ESTIMATORS}
        for day in self.bundle.days:
            summary = pf.revealed_view(pf.replay_day(self.bundle, day))["summary"]
            for name in ex.ESTIMATORS:
                s = summary[name]
                totals[name] += [s["Correct alert"], s["False alert"], s["Miss"], s["Correct clear"]]
        self.assertEqual({n: tuple(int(x) for x in v) for n, v in totals.items()}, TOTALS)

    def test_june_18_counts(self):
        view = pf.revealed_view(pf.replay_day(self.bundle, "2024-06-18"))
        self.assertEqual((view["positives"], view["total"]), (45, 100))
        got = {n: (s["Correct alert"], s["False alert"], s["Miss"], s["Correct clear"])
               for n, s in view["summary"].items()}
        self.assertEqual(got, JUNE18)

    def test_scores_come_from_rows_before_the_day_only(self):
        seen = []
        real = demo.forecast

        def spy(models, thresholds, history, day):
            seen.append((history["dt"].max(), day))
            return real(models, thresholds, history, day)
        with mock.patch.object(demo, "forecast", spy):
            pf.replay_day(self.bundle, "2024-06-18")
        self.assertEqual(seen, [("2024-06-17", "2024-06-18")])

    def test_forecast_view_holds_no_outcome(self):
        view = pf.forecast_view(pf.replay_day(self.bundle, "2024-06-18"))
        text = json.dumps(view)
        for word in ("y_true", "outcome", "observed", "Miss", "False alert", "Correct", "positives", "summary"):
            self.assertNotIn(word, text)
        self.assertEqual(len(view["rows"]), 100)
        top = view["rows"][0]
        self.assertEqual((top["rank"], top["city_id"], top["store_id"], top["product_id"]), (1, 3, 855, 188))
        self.assertAlmostEqual(top["logistic_score"], 0.8313, places=4)
        self.assertAlmostEqual(top["forest_score"], 0.8280, places=4)
        self.assertEqual(view["alerts"], {"logistic": 54, "forest": 72, "persistence": 45})

    def test_revealed_view_names_real_errors(self):
        view = pf.revealed_view(pf.replay_day(self.bundle, "2024-06-18"))
        example = view["examples"]["logistic"]
        self.assertEqual([example["False alert"][k] for k in data.KEY], [0, 348, 600])
        self.assertEqual([example["Miss"][k] for k in data.KEY], [12, 316, 834])
        self.assertEqual(sum(r["y_true"] for r in view["rows"]), 45)

    def test_sorting_changes_order_only(self):
        a = pf.revealed_view(pf.replay_day(self.bundle, "2024-06-18", "logistic"))
        b = pf.revealed_view(pf.replay_day(self.bundle, "2024-06-18", "forest"))

        def keyed(view):
            return {tuple(r[k] for k in data.KEY): {k: v for k, v in r.items() if k != "rank"} for r in view["rows"]}
        self.assertEqual(keyed(a), keyed(b))
        self.assertNotEqual([r["store_id"] for r in a["rows"]], [r["store_id"] for r in b["rows"]])
        self.assertEqual(a["summary"], b["summary"])
        scores = [r["forest_score"] for r in b["rows"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_bad_date_or_sort_is_refused(self):
        for day in ("2024-06-10", "2024-06-26", "2024-13-40", "", None, "../x"):
            with self.subTest(day=day), self.assertRaisesRegex(data.DataError, "choose a date"):
                pf.replay_day(self.bundle, day)
        with self.assertRaisesRegex(data.DataError, "sort"):
            pf.replay_day(self.bundle, "2024-06-18", "persistence")

    def test_replay_checks_the_files_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = copy_bundle(Path(tmp))
            moved = pf.Bundle(**{**vars(self.bundle), "root": root, "run": root / "runs" / pf.RUN_ID})
            (moved.run / "forest.joblib").write_bytes(b"swapped after start")
            manifest = ex.read_json(moved.run / "manifest.json")  # a consistent swap passes load_run on its own
            manifest["artifacts"]["forest.joblib"] = data.sha256_file(moved.run / "forest.joblib")
            ex.write_json(moved.run / "manifest.json", manifest)
            with mock.patch("joblib.load", side_effect=AssertionError("a pickle was opened")), \
                    self.assertRaisesRegex(data.DataError, "changed"):
                pf.replay_day(moved, "2024-06-18")

    # ---------- export ----------

    def test_export_writes_the_existing_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = pf.replay_day(self.bundle, "2024-06-18")
            path = pf.export_report(self.bundle, result, Path(tmp) / "new folder")
            self.assertEqual(path.name, "replay-2024-06-18.html")
            self.assertEqual(path.read_text(encoding="utf-8"), demo.render_html(result, self.bundle.metrics))

    def test_export_into_a_file_path_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "file"
            blocker.write_text("x")
            with self.assertRaises(OSError):
                pf.export_report(self.bundle, pf.replay_day(self.bundle, "2024-06-18"), blocker)

    # ---------- health (M2) ----------

    def test_health_is_json_safe(self):
        health = pf.build_health(self.bundle)
        self.assertEqual(json.loads(json.dumps(health, allow_nan=False)), health)

    def test_health_prevalence_matches_the_protocol_counts(self):
        got = {p["split"]: (p["rows"], p["positives"], round(p["rate"], 6))
               for p in pf.build_health(self.bundle)["prevalence"]}
        self.assertEqual(got, {"train": (5300, 2403, 0.453396), "validation": (1500, 535, 0.356667),
                               "test": (1500, 682, 0.454667)})

    def test_health_daily_counts_add_up_to_the_receipt(self):
        health = pf.build_health(self.bundle)
        self.assertEqual([d["dt"] for d in health["daily"]["test"]], self.bundle.days)
        for name, expected in TOTALS.items():
            rows = [d["estimators"][name] for d in health["daily"]["test"]]
            self.assertEqual(tuple(sum(r[k] for r in rows) for k in ("tp", "fp", "fn", "tn")), expected)
            self.assertTrue(all(r["tp"] + r["fp"] == r["alerts"] for r in rows))
        for split, positives in (("test", 682), ("validation", 535)):
            self.assertEqual(len(health["daily"][split]), 15)
            self.assertEqual(sum(d["positives"] for d in health["daily"][split]), positives)
            self.assertTrue(all(d["rows"] == 100 for d in health["daily"][split]))

    def test_health_workload_matches_the_receipt(self):
        work = pf.build_health(self.bundle)["workload"]
        self.assertEqual({n: (round(w["alerts_per_day"], 4), w["alerts_min"], w["alerts_max"],
                              round(w["misses_per_day"], 4), round(w["false_alerts_per_day"], 4))
                          for n, w in work.items()},
                         {"logistic": (51.9333, 32, 72, 14.5333, 21.0), "forest": (72.1333, 56, 95, 6.6667, 33.3333),
                          "persistence": (44.3333, 32, 55, 17.6, 16.4667)})

    def test_health_counts_data_problems(self):
        quality = pf.build_health(self.bundle)["quality"]
        self.assertEqual((quality["rows"], quality["series"], quality["first_day"], quality["last_day"]),
                         (9000, 100, "2024-03-28", "2024-06-25"))
        self.assertEqual((quality["missing"], quality["duplicates"], quality["out_of_range"], quality["gaps"]),
                         (0, 0, 0, 0))

    def test_quality_counts_respond_to_bad_rows(self):
        bad = self.bundle.cohort.copy()
        bad.loc[0, "sale_amount"] = -1.0
        bad.loc[1, "stock_hour6_22_cnt"] = 17
        bad.loc[2, "sale_amount"] = np.nan
        bad = pd.concat([bad.drop(index=50), bad.iloc[[10]]], ignore_index=True)
        quality = pf.quality_counts(bad)
        self.assertEqual((quality["missing"], quality["duplicates"], quality["out_of_range"], quality["gaps"]),
                         (1, 1, 2, 1))

    def test_health_explanations_match_the_receipt(self):
        explain = pf.build_health(self.bundle)["explain"]
        self.assertEqual({e["feature"]: round(e["value"], 4) for e in explain["logistic"][:4]},
                         {"stockout_hours_lag1": 0.4659, "sales_lag1": 0.3759, "stockout_days7": 0.3641,
                          "sales_mean7": -0.2814})
        forest = {e["feature"]: e["value"] for e in explain["forest"]}
        self.assertEqual([round(forest[n], 3) for n in ("stockout_hours_lag1", "stockout_days7", "sales_mean7",
                                                        "sales_lag1")], [0.449, 0.202, 0.154, 0.123])
        self.assertAlmostEqual(sum(v for k, v in forest.items() if k.startswith("weekday")), 0.072, places=3)
        self.assertEqual([len(explain[n]) for n in ex.MODELS], [11, 11])

    def test_health_refuses_rows_that_do_not_reconcile(self):
        changed = self.bundle.test.copy()
        changed.loc[0, "logistic_flag"] = 1 - changed.loc[0, "logistic_flag"]
        with self.assertRaisesRegex(data.DataError, "reconcile"):
            pf.build_health(pf.Bundle(**{**vars(self.bundle), "test": changed}))

    # ---------- experiment facts ----------

    def test_metrics_rows_match_the_receipt(self):
        rows = {r["name"]: r for r in pf.metric_rows(self.bundle)}
        self.assertEqual({n: round(r["average_precision"], 6) for n, r in rows.items()},
                         {"logistic": 0.640380, "forest": 0.643949, "persistence": 0.561253})
        self.assertEqual({n: (r["tp"], r["fp"], r["fn"], r["tn"]) for n, r in rows.items()}, TOTALS)

    def test_nothing_in_the_bundle_was_written(self):
        pf.build_health(self.bundle)
        pf.replay_day(self.bundle, "2024-06-11")
        self.assertEqual({p: data.sha256_file(p) for p in BUNDLE_FILES}, self.before)


class SessionTests(unittest.TestCase):
    """Page state for the replay: what the user can see and do at each step."""

    def setUp(self):
        self.calls = []

        def replay(day, sort):
            self.calls.append((day, sort))
            if day == "bad":
                raise data.DataError("choose a date")
            return {"day": day, "sort": sort}
        self.s = pf.Session(replay)

    def test_starts_with_no_result(self):
        self.assertEqual((self.s.state, self.s.result, self.s.can_reveal, self.s.can_export),
                         ("select", None, False, False))

    def test_reveal_needs_a_forecast(self):
        with self.assertRaises(data.DataError):
            self.s.reveal()
        self.s.generate("a", "logistic")
        self.assertEqual((self.s.state, self.s.can_reveal, self.s.can_export), ("forecast", True, False))
        self.s.reveal()
        self.assertEqual((self.s.state, self.s.can_reveal, self.s.can_export), ("revealed", False, True))

    def test_new_date_clears_result_and_reveal(self):
        self.s.generate("a", "logistic")
        self.s.reveal()
        self.s.change_date()
        self.assertEqual((self.s.state, self.s.result, self.s.can_export), ("select", None, False))

    def test_sorting_keeps_the_reveal_state(self):
        self.s.generate("a", "logistic")
        self.s.change_sort("forest")
        self.assertEqual((self.s.state, self.s.result["sort"]), ("forecast", "forest"))
        self.s.reveal()
        self.s.change_sort("logistic")
        self.assertEqual((self.s.state, self.s.result["sort"]), ("revealed", "logistic"))

    def test_sorting_without_a_result_does_nothing(self):
        self.s.change_sort("forest")
        self.assertEqual((self.s.state, self.calls), ("select", []))

    def test_failed_forecast_leaves_no_stale_result(self):
        self.s.generate("a", "logistic")
        with self.assertRaises(data.DataError):
            self.s.generate("bad", "logistic")
        self.assertEqual((self.s.state, self.s.result, self.s.can_reveal), ("select", None, False))


if __name__ == "__main__":
    unittest.main()
