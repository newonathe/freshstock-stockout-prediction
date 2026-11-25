"""FreshStock 2.0 portable bundle: trust, replay, reveal and export."""
import copy
import csv
import datetime as dt
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from freshstock import data as b3
from freshstock.expanded import bundle, data, experiment, metrics
from tests.test_expanded_experiment import N, Workspace


class BundleCase(Workspace):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        experiment.evaluate(cls.run_dir, cls.freeze_sha, cls.train, cls.holdout)
        cls.folder = cls.root / "bundle"
        cls.pin = bundle.build(cls.run_dir, cls.freeze_sha, cls.folder, cls.train, cls.holdout)
        cls.loaded = bundle.load(cls.folder, cls.pin)

    def copy_bundle(self):
        folder = Path(tempfile.mkdtemp(dir=self.root)) / "bundle"
        shutil.copytree(self.folder, folder)
        return folder


class BuildTests(BundleCase):
    def test_files_and_pin(self):
        self.assertEqual({p.name for p in self.folder.iterdir()}, set(self.pin["files"]))
        self.assertEqual(set(self.pin["files"]), set(bundle.FILES))
        for name, entry in self.pin["files"].items():
            self.assertEqual((entry["sha256"], entry["size"]),
                             (b3.sha256_file(self.folder / name), (self.folder / name).stat().st_size))
        self.assertEqual((self.pin["run_id"], self.pin["freeze_sha256"], self.pin["n_series"]),
                         (self.run_dir.name, self.freeze_sha, N))

    def test_history_is_35_days_of_the_sources(self):
        h = self.loaded.history
        whole = data.read_panel(self.train).join(data.read_panel(self.holdout))
        self.assertEqual([dt.date.fromordinal(int(h.dates[i])).isoformat() for i in (0, -1)],
                         ["2024-05-29", "2024-07-02"])
        self.assertEqual((h.sales.shape, h.stock.dtype, h.sales.dtype, h.keys.dtype, h.dates.dtype),
                         ((N, 35), np.uint8, np.float64, np.int64, np.int32))
        part = whole.since(dt.date(2024, 5, 29))
        for name in ("keys", "dates", "sales", "stock"):
            np.testing.assert_array_equal(getattr(h, name), getattr(part, name))

    def test_saved_scores_equal_the_research_predictions(self):
        saved = experiment.load_predictions(self.run_dir / "predictions.parquet")
        p = self.loaded.predictions
        for name in ("dates", "y_true", "past_stockout", "keys"):
            np.testing.assert_array_equal(p[name], saved[name])
        for e in metrics.ESTIMATORS:
            np.testing.assert_array_equal(p["flags"][e], saved["flags"][e])
        for f in metrics.MODELS:
            np.testing.assert_array_equal(p["scores"][f], saved["scores"][f])
        self.assertEqual(json.loads(json.dumps(metrics.summarize(p))), self.loaded.metrics["summary"])

    def test_existing_output_is_never_overwritten(self):
        self.refused("exists", bundle.build, self.run_dir, self.freeze_sha, self.folder, self.train, self.holdout)

    def test_runtime_works_without_pyarrow(self):
        """pandas imports pyarrow on its own when it is installed, so its absence is simulated, as in the program."""
        code = ("import sys, json, pathlib\n"
                "class Absent:\n"
                "    def find_spec(self, name, *rest):\n"
                "        if name.split('.')[0] == 'pyarrow':\n"
                "            raise ModuleNotFoundError(name, name=name)\n"
                "sys.meta_path.insert(0, Absent())\n"
                "from freshstock.expanded import bundle, gui\n"
                "b = bundle.load(sys.argv[1], json.loads(pathlib.Path(sys.argv[2]).read_text()))\n"
                "[bundle.reveal(b, bundle.forecast(b, d)) for d in b.days]\n"
                "assert 'pyarrow' not in sys.modules\n")
        pin = self.root / "pin.json"
        pin.write_text(json.dumps(self.pin))
        done = subprocess.run([sys.executable, "-W", "error", "-c", code, str(self.folder), str(pin)], cwd=b3.ROOT,
                              capture_output=True, text=True)
        self.assertEqual((done.returncode, done.stderr[-1500:]), (0, ""))


class TrustTests(BundleCase):
    def test_any_changed_missing_or_extra_item_is_refused_before_pickle_or_numpy(self):
        cases = {f"changed {n}": (lambda f, n=n: (f / n).write_bytes((f / n).read_bytes()[:-1] + b"!"))
                 for n in bundle.FILES}
        cases.update({f"missing {n}": (lambda f, n=n: (f / n).unlink()) for n in bundle.FILES})
        cases["extra file"] = lambda f: (f / "x.joblib").write_bytes(b"x")
        cases["extra folder"] = lambda f: (f / "sub").mkdir()
        for name, change in cases.items():
            folder = self.copy_bundle()
            change(folder)
            with mock.patch.object(bundle.joblib, "load", side_effect=AssertionError("unpickled")), \
                    mock.patch.object(bundle.np, "load", side_effect=AssertionError("parsed")):
                with self.assertRaises(b3.DataError, msg=name):
                    bundle.load(folder, self.pin)

    def test_link_is_refused(self):
        folder = self.copy_bundle()
        with mock.patch.object(Path, "is_junction", lambda p: p.name == "bundle"):
            self.refused("link", bundle.load, folder, self.pin)
        with mock.patch.object(Path, "is_symlink", lambda p: p.name == "history.npz"):
            self.refused("link", bundle.load, folder, self.pin)

    def test_models_are_loaded_from_the_verified_bytes(self):
        seen, original = [], bundle.joblib.load

        def spy(stream):
            seen.append(type(stream))
            return original(stream)
        with mock.patch.object(bundle.joblib, "load", spy):
            bundle.load(self.folder, self.pin)
        self.assertEqual(seen, [io.BytesIO, io.BytesIO])

    def repinned(self, name, arrays):
        """A bundle whose array file was replaced and re-pinned: structure checks must still refuse it."""
        folder, pin = self.copy_bundle(), copy.deepcopy(self.pin)
        np.savez_compressed(folder / name, **arrays)
        pin["files"][name] = {"sha256": b3.sha256_file(folder / name), "size": (folder / name).stat().st_size}
        return folder, pin

    def arrays(self, name):
        with np.load(self.folder / name, allow_pickle=False) as stored:
            return {k: stored[k] for k in stored.files}

    def test_malformed_arrays_are_refused_even_when_pinned(self):
        def swap(a, **changes):
            return {**a, **changes}
        h, s = self.arrays("history.npz"), self.arrays("scores.npz")
        bad_key = h["keys"].copy()
        bad_key[[0, 1]] = bad_key[[1, 0]]
        outcome, flag, score = s["y_true"].copy(), s["forest_flag"].copy(), s["forest_score"].copy()
        outcome[0, 0] ^= 1
        flag[0, 0] ^= 1
        score[0, 0] = np.nan
        big = h["stock"].copy()
        big[0, 0] = 17
        cases = [("history.npz", swap(h, keys=bad_key)), ("history.npz", swap(h, dates=h["dates"] + 1)),
                 ("history.npz", swap(h, sales=h["sales"].astype(np.float32))),
                 ("history.npz", swap(h, sales=-h["sales"] - 1)), ("history.npz", swap(h, stock=big)),
                 ("history.npz", swap(h, stock=h["stock"][:, 1:])),
                 ("history.npz", swap(h, note=np.zeros(1))), ("history.npz", {k: h[k] for k in ("keys", "dates")}),
                 ("scores.npz", swap(s, y_true=outcome)), ("scores.npz", swap(s, forest_flag=flag)),
                 ("scores.npz", swap(s, forest_score=score)), ("scores.npz", swap(s, dates=s["dates"][::-1])),
                 ("scores.npz", swap(s, past_stockout=1 - s["past_stockout"]))]
        for name, arrays in cases:
            folder, pin = self.repinned(name, arrays)
            with self.assertRaises(b3.DataError, msg=str([k for k in arrays])):
                bundle.load(folder, pin)

    def test_object_arrays_are_refused(self):
        folder, pin = self.copy_bundle(), copy.deepcopy(self.pin)
        np.savez_compressed(folder / "history.npz", keys=np.array([{"x": 1}], dtype=object))
        pin["files"]["history.npz"] = {"sha256": b3.sha256_file(folder / "history.npz"),
                                       "size": (folder / "history.npz").stat().st_size}
        with self.assertRaises(b3.DataError):
            bundle.load(folder, pin)

    def test_other_library_version_is_refused(self):
        with mock.patch.object(bundle, "version", lambda name: "0.0"):
            self.refused("required", bundle.load, self.folder, self.pin)


class ReplayTests(BundleCase):
    def test_all_seven_days_match_saved_scores_and_flags(self):
        self.assertEqual(self.loaded.days, [f"2024-06-{d}" for d in range(26, 31)] + ["2024-07-01", "2024-07-02"])
        for i, day in enumerate(self.loaded.days):
            f = bundle.forecast(self.loaded, day)
            for m in metrics.MODELS:
                np.testing.assert_allclose(f.scores[m], self.loaded.predictions["scores"][m][i], rtol=0, atol=1e-10)
            for e in metrics.ESTIMATORS:
                np.testing.assert_array_equal(f.flags[e], self.loaded.predictions["flags"][e][i])

    def test_forecast_refuses_other_dates(self):
        for day in ("2024-06-25", "2024-07-03", "x", "", "2024-6-26"):
            self.refused("choose a date", bundle.forecast, self.loaded, day)

    def test_parity_failure_stops_the_forecast(self):
        other = copy.copy(self.loaded)
        scores = {k: v.copy() for k, v in other.predictions["scores"].items()}
        scores["forest"][0, 3] += 1e-6
        object.__setattr__(other, "predictions", {**other.predictions, "scores": scores})
        self.refused("saved", bundle.forecast, other, "2024-06-26")

    def test_forecast_has_no_outcome_and_never_reads_the_target_day(self):
        f = bundle.forecast(self.loaded, "2024-06-27")
        self.assertFalse([n for n in vars(f) if "true" in n or "outcome" in n or "actual" in n])
        seen, original = [], data.features_on

        def spy(panel, day):
            seen.append(int(panel.dates[-1]) - day.toordinal())
            return original(panel, day)
        with mock.patch.object(data, "features_on", spy):
            bundle.forecast(self.loaded, "2024-06-27")
        self.assertEqual(set(seen), {-1})

    def test_rank_order_is_global_and_stable(self):
        f = bundle.forecast(self.loaded, "2024-06-26")
        for m in metrics.MODELS:
            order = f.order(m)
            self.assertEqual(sorted(order.tolist()), list(range(N)))
            expected = sorted(range(N), key=lambda i: (-f.scores[m][i], i))
            self.assertEqual(order.tolist(), expected)
            self.assertIs(order, f.order(m))  # sorted once

    def test_pages_cover_every_rank_once(self):
        f = bundle.forecast(self.loaded, "2024-06-26")
        for size in (7, 10, 40, 100):
            ranks = []
            for page in range(bundle.page_count(N, size)):
                ranks += [r["rank"] for r in bundle.page_rows(self.loaded, f, "forest", page, size)]
            self.assertEqual(ranks, list(range(1, N + 1)))
        self.assertEqual([bundle.page_count(n, 100) for n in (1, 100, 101, 10_000, 50_000)], [1, 1, 2, 100, 500])
        row = bundle.page_rows(self.loaded, f, "forest", 0, 100)[0]
        at = int(f.order("forest")[0])
        self.assertEqual([row[k] for k in b3.KEY], self.loaded.history.keys[at].tolist())
        self.assertEqual(row["forest_score"], f.scores["forest"][at])
        self.assertNotIn("y_true", row)
        self.assertEqual(set(row["features"]), set(data.FEATURES["B"]))
        self.refused("page", bundle.page_rows, self.loaded, f, "forest", 1, 100)

    def test_find_series(self):
        f = bundle.forecast(self.loaded, "2024-06-26")
        key = tuple(self.loaded.history.keys[11].tolist())
        rank = bundle.find_rank(self.loaded, f, "logistic", key)
        self.assertEqual(int(f.order("logistic")[rank - 1]), 11)
        self.assertIsNone(bundle.find_rank(self.loaded, f, "logistic", (17, 897, 864)))

    def test_reveal_counts(self):
        f = bundle.forecast(self.loaded, "2024-06-28")
        r = bundle.reveal(self.loaded, f)
        y = self.loaded.predictions["y_true"][2]
        self.assertEqual((r.positives, r.total), (int(y.sum()), N))
        for e in metrics.ESTIMATORS:
            self.assertEqual(r.counts[e], {k: metrics.flag_metrics(y, f.flags[e])[k] for k in ("tp", "fn", "fp", "tn")})
            self.assertEqual(r.review[e], {k: self.loaded.metrics["summary"]["estimators"][e]["top_k"]["daily"][2][k]
                                           for k in ("k", "hits", "false_alerts", "misses", "recall")})
        rows = bundle.page_rows(self.loaded, f, "forest", 0, 100, r)
        for row in rows:
            at = self.loaded.history.keys.tolist().index([row[k] for k in b3.KEY])
            self.assertEqual(row["y_true"], int(y[at]))
            self.assertEqual(row["result"]["forest"], bundle.RESULTS[(int(y[at]), int(f.flags["forest"][at]))])


class ExportTests(BundleCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp(dir=self.root)) / "reports"
        self.f = bundle.forecast(self.loaded, "2024-06-26")
        self.r = bundle.reveal(self.loaded, self.f)

    def test_full_csv_and_bounded_html(self):
        folder = bundle.export(self.loaded, self.f, self.r, "forest", self.out)
        self.assertEqual({p.name for p in folder.iterdir()}, {"ranking.csv", "summary.html", "manifest.json"})
        with open(folder / "ranking.csv", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual([int(r["rank"]) for r in rows], list(range(1, N + 1)))
        order = self.f.order("forest")
        for i, row in enumerate(rows):
            at = int(order[i])
            self.assertEqual([int(row[k]) for k in b3.KEY], self.loaded.history.keys[at].tolist())
            self.assertEqual(float(row["forest_score"]), self.f.scores["forest"][at])
            self.assertEqual(int(row["y_true"]), int(self.loaded.predictions["y_true"][0, at]))
        html = (folder / "summary.html").read_text(encoding="utf-8")
        for text in ("2024-06-26", f"{N} series", "CC BY 4.0", "historical", self.run_dir.name):
            self.assertIn(text, html)
        self.assertNotIn("http://", html.replace("http://www.w3.org", ""))
        self.assertLess(len(html), 100_000)
        manifest = json.loads((folder / "manifest.json").read_text())
        self.assertEqual(manifest["rows"], N)
        for name, digest in manifest["files"].items():
            self.assertEqual(b3.sha256_file(folder / name), digest)

    def test_second_export_does_not_overwrite(self):
        a = bundle.export(self.loaded, self.f, self.r, "forest", self.out)
        b = bundle.export(self.loaded, self.f, self.r, "forest", self.out)
        self.assertNotEqual(a, b)
        self.assertTrue((a / "manifest.json").is_file() and (b / "manifest.json").is_file())

    def test_failure_leaves_no_report(self):
        with mock.patch.object(bundle, "_html", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                bundle.export(self.loaded, self.f, self.r, "forest", self.out)
        self.assertEqual(list(self.out.iterdir()), [])


class HealthTests(BundleCase):
    def test_health_reconciles_with_frozen_files(self):
        h = bundle.health(self.loaded)
        self.assertEqual([p["split"] for p in h["prevalence"]], ["initial_fit", "tuning", "calibration", "threshold",
                                                                 "holdout"])
        for p in h["prevalence"]:
            self.assertEqual(p["rate"], p["positives"] / p["rows"])
        s = self.loaded.metrics["summary"]
        for e in metrics.ESTIMATORS:
            self.assertEqual(sum(d["estimators"][e]["alerts"] for d in h["daily"]), s["estimators"][e]["alerts"])
            self.assertEqual(h["workload"][e]["alerts_per_day"], s["estimators"][e]["alerts"] / 7)
        self.assertEqual({p["feature"] for p in h["explain"]["forest"]}
                         <= set(data.FEATURES["B"]), True)


@unittest.skipUnless((b3.ROOT / "bundle").is_dir(), "needs the local 2.0 bundle")
class ReleaseTests(unittest.TestCase):
    """The committed pin, the real bundle and the committed receipts describe the same run."""

    @classmethod
    def setUpClass(cls):
        from freshstock.expanded.pin import PIN
        cls.pin, cls.loaded = PIN, bundle.load(b3.ROOT / "bundle", PIN)

    def test_all_days_replay_with_full_parity(self):
        total = dict.fromkeys(metrics.ESTIMATORS, 0)
        for day in self.loaded.days:
            f = bundle.forecast(self.loaded, day)  # refuses any score difference above 1e-10
            r = bundle.reveal(self.loaded, f)
            for e in total:
                total[e] += r.counts[e]["tp"]
        self.assertEqual(total, {e: v["tp"] for e, v in self.loaded.metrics["summary"]["estimators"].items()})
        self.assertEqual(self.loaded.n, 50_000)

    def test_committed_receipts_match_the_pin(self):
        results = b3.ROOT / "docs" / "results"
        for receipt, name in (("E2_PROTOCOL_LOCK.json", "protocol.json"), ("E3_CANDIDATES.json", "candidate_metrics.json"),
                              ("E4_FREEZE.json", "freeze.json"), ("E5_METRICS.json", "metrics.json"),
                              ("E5_EVALUATION.json", "evaluation.json")):
            if (results / receipt).is_file():  # git may store these with other line endings
                text = (results / receipt).read_bytes().replace(bytes([13, 10]), bytes([10]))
                self.assertEqual(hashlib.sha256(text).hexdigest(), self.pin["files"][name]["sha256"], receipt)

    def test_export_holds_all_50000_rows(self):
        f = bundle.forecast(self.loaded, self.loaded.days[0])
        with tempfile.TemporaryDirectory() as tmp:
            folder = bundle.export(self.loaded, f, bundle.reveal(self.loaded, f), "forest", tmp)
            with open(folder / "ranking.csv", encoding="utf-8", newline="") as file:
                self.assertEqual(sum(1 for _ in file), 50_001)


if __name__ == "__main__":
    unittest.main()
