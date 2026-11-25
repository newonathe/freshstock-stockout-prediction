"""FreshStock 2.0 metrics from counts. Small expected values were worked out by hand."""
import json
import unittest

import numpy as np
from sklearn import metrics as skm

from freshstock import data as b3
from freshstock.expanded import metrics


class AveragePrecisionTests(unittest.TestCase):
    def test_tied_scores_form_one_step(self):
        self.assertAlmostEqual(metrics.average_precision([1, 0, 1, 0], [.9, .9, .2, .1]), 7 / 12, delta=1e-15)
        self.assertAlmostEqual(metrics.average_precision([0, 1, 0, 1], [.9, .9, .1, .2]), 7 / 12, delta=1e-15)

    def test_all_tied_equals_prevalence(self):
        self.assertAlmostEqual(metrics.average_precision([1, 0, 0, 0], [.5] * 4), .25, delta=1e-15)

    def test_single_class_is_undefined(self):
        self.assertIsNone(metrics.average_precision([0, 0], [.1, .2]))
        self.assertIsNone(metrics.average_precision([1, 1], [.1, .2]))
        self.assertIsNone(metrics.average_precision([], []))

    def test_matches_sklearn_with_many_ties_and_none(self):
        rng = np.random.default_rng(4)
        for decimals in (1, 2, 12):
            score = rng.random(5000).round(decimals)
            y = (rng.random(5000) < score).astype(np.uint8)
            self.assertAlmostEqual(metrics.average_precision(y, score), skm.average_precision_score(y, score),
                                   delta=1e-10)

    def test_large_input_is_fast_enough_to_finish(self):
        rng = np.random.default_rng(4)
        score = rng.random(350_000)
        y = (rng.random(350_000) < score).astype(np.uint8)
        self.assertAlmostEqual(metrics.average_precision(y, score), skm.average_precision_score(y, score), delta=1e-10)


class FlagTests(unittest.TestCase):
    def test_hand_counts(self):
        m = metrics.flag_metrics([1, 0, 1, 0], [1, 1, 0, 0])
        self.assertEqual((m["tn"], m["fp"], m["fn"], m["tp"], m["n"]), (1, 1, 1, 1, 4))
        self.assertEqual((m["accuracy"], m["precision"], m["recall"], m["f1"], m["balanced_accuracy"]), (.5,) * 5)
        self.assertEqual((m["alerts"], m["alert_rate"], m["prevalence"]), (2, .5, .5))

    def test_matches_sklearn(self):
        rng = np.random.default_rng(9)
        y, f = rng.integers(0, 2, 999), rng.integers(0, 2, 999)
        m = metrics.flag_metrics(y, f)
        for name, theirs in (("accuracy", skm.accuracy_score), ("precision", skm.precision_score),
                             ("recall", skm.recall_score), ("f1", skm.f1_score),
                             ("balanced_accuracy", skm.balanced_accuracy_score)):
            self.assertAlmostEqual(m[name], theirs(y, f), delta=1e-12)
        self.assertEqual([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], skm.confusion_matrix(y, f).tolist())

    def test_undefined_values_are_null_not_zero(self):
        m = metrics.flag_metrics([0, 0], [0, 0])
        self.assertEqual([m[k] for k in ("precision", "recall", "f1", "balanced_accuracy")], [None] * 4)
        self.assertEqual(m["accuracy"], 1)
        m = metrics.flag_metrics([], [])
        self.assertEqual((m["n"], m["accuracy"], m["alert_rate"], m["prevalence"]), (0, None, None, None))
        self.assertIsNone(metrics.flag_metrics([0, 1], [0, 0])["precision"])


class TopKTests(unittest.TestCase):
    def test_k_is_the_exact_ceiling_of_ten_percent(self):
        self.assertEqual([metrics.k_of(n) for n in (1, 9, 10, 11, 100, 101, 10_000, 25_000, 50_000)],
                         [1, 1, 1, 2, 10, 11, 1000, 2500, 5000])

    def test_tie_goes_to_the_smaller_key(self):
        self.assertEqual(metrics.ranking(np.array([.2, .9, .9, .1])).tolist(), [1, 2, 0, 3])

    def test_hand_fixture(self):
        r = metrics.top_k(np.array([[1, 0, 1, 0]]), np.array([[.9, .9, .2, .1]]), k=2)
        self.assertEqual(r["daily"], [{"k": 2, "hits": 1, "false_alerts": 1, "misses": 1, "coverage": .5,
                                       "precision": .5, "recall": .5}])

    def test_pooled_counts_and_day_without_positives(self):
        y = np.array([[0] * 20, [1] * 3 + [0] * 17])
        score = np.array([np.linspace(1, 0, 20), np.linspace(1, 0, 20)])
        r = metrics.top_k(y, score)
        self.assertIsNone(r["daily"][0]["recall"])
        self.assertEqual((r["k_total"], r["hits"], r["false_alerts"], r["misses"]), (4, 2, 2, 1))
        self.assertEqual((r["precision"], r["recall"]), (.5, 2 / 3))


class CalibrationTests(unittest.TestCase):
    def test_bin_edges(self):
        bins = metrics.reliability([0, 1, 0, 1, 1], [0., .1, .9, 1., .0999999])
        self.assertEqual([b["n"] for b in bins], [2, 1, 0, 0, 0, 0, 0, 0, 0, 2])
        self.assertEqual((bins[0]["mean_score"], bins[0]["observed_rate"]), ((0 + .0999999) / 2, .5))
        self.assertEqual((bins[2]["mean_score"], bins[2]["observed_rate"]), (None, None))
        self.assertEqual((bins[1]["low"], bins[1]["high"], bins[9]["high"]), (.1, .2, 1.))

    def test_every_tenth_falls_in_its_own_bin(self):
        for i in range(10):
            bins = metrics.reliability([1], [i / 10])
            self.assertEqual([b["n"] for b in bins].index(1), i)

    def test_brier_and_log_loss_match_sklearn(self):
        rng = np.random.default_rng(2)
        p = np.concatenate([rng.random(500), [0., 1., 0., 1.]])
        y = np.concatenate([rng.integers(0, 2, 500), [0, 1, 1, 0]])
        self.assertAlmostEqual(metrics.brier(y, p), skm.brier_score_loss(y, p), delta=1e-12)
        self.assertAlmostEqual(metrics.log_loss(y, p), skm.log_loss(y, p), delta=1e-10)


class SummaryTests(unittest.TestCase):
    def predictions(self, n=40, days=7):
        rng = np.random.default_rng(6)
        keys = np.array([(i % 3, i, 0) for i in range(n)])
        keys = keys[np.lexsort(keys.T[::-1])]
        score = {m: rng.random((days, n)) for m in ("logistic", "forest")}
        past = rng.integers(0, 2, (days, n)).astype(np.uint8)
        return {"keys": keys, "dates": np.arange(739063, 739063 + days, dtype=np.int32),
                "y_true": rng.integers(0, 2, (days, n)).astype(np.uint8), "past_stockout": past,
                "scores": score, "flags": {**{m: (s >= .5).astype(np.uint8) for m, s in score.items()},
                                           "persistence": past.copy()}}

    def test_totals_daily_slices_and_json(self):
        p = self.predictions()
        s = metrics.summarize(p)
        json.dumps(s, allow_nan=False)
        self.assertEqual((s["rows"], s["positives"], s["days"], s["series"]), (280, int(p["y_true"].sum()), 7, 40))
        self.assertEqual(set(s["estimators"]), {"logistic", "forest", "persistence"})
        for name, e in s["estimators"].items():
            self.assertEqual(sum(d["alerts"] for d in e["daily"]), e["alerts"])
            self.assertEqual(sum(d["misses"] for d in e["daily"]), e["fn"])
            self.assertEqual(sum(d["false_alerts"] for d in e["daily"]), e["fp"])
            self.assertEqual(len(e["top_k"]["daily"]), 7)
            parts = [s["slices"]["previous_day_stockout"][v]["estimators"][name] for v in ("yes", "no")]
            for count in ("tp", "fp", "fn", "tn"):
                self.assertEqual(sum(x[count] for x in parts), e[count])
                self.assertEqual(sum(c["estimators"][name][count] for c in s["slices"]["city"].values()), e[count])
        self.assertEqual(list(s["slices"]["city"]), [str(i) for i in range(18)])
        self.assertEqual(s["slices"]["city"]["17"]["n"], 0)
        self.assertIsNone(s["slices"]["city"]["17"]["estimators"]["forest"]["average_precision"])
        self.assertIn("calibration", s["estimators"]["forest"])
        self.assertNotIn("calibration", s["estimators"]["persistence"])
        metrics.check_against_sklearn(p, s)

    def test_parity_check_catches_a_changed_number(self):
        p = self.predictions()
        s = metrics.summarize(p)
        s["estimators"]["forest"]["average_precision"] += 1e-6
        with self.assertRaises(b3.DataError):
            metrics.check_against_sklearn(p, s)

    def test_invalid_predictions_are_refused(self):
        for change in ("nan", "range", "flag", "keys", "dates", "shape"):
            p = self.predictions()
            if change == "nan":
                p["scores"]["forest"][0, 0] = np.nan
            elif change == "range":
                p["scores"]["forest"][0, 0] = 1.1
            elif change == "flag":
                p["flags"]["forest"][0, 0] = 2
            elif change == "keys":
                p["keys"] = p["keys"][::-1]
            elif change == "dates":
                p["dates"] = p["dates"][::-1]
            else:
                p["y_true"] = p["y_true"][:, :-1]
            with self.assertRaises(b3.DataError, msg=change):
                metrics.summarize(p)


if __name__ == "__main__":
    unittest.main()
