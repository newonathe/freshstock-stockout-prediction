"""FS-05 to FS-08: fixed models, threshold rule, independent metrics and frozen runs."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from freshstock import data, experiment as ex

HAS_COHORT = (data.DERIVED / "cohort.csv").is_file()


class MetricTests(unittest.TestCase):
    def test_hand_computed_confusion_and_ratios(self):
        y = np.array([1, 1, 1, 0, 0, 0, 0, 1])
        flags = np.array([1, 1, 0, 1, 0, 0, 0, 0])
        m = ex.flag_metrics(y, flags)
        self.assertEqual((m["tn"], m["fp"], m["fn"], m["tp"]), (3, 1, 2, 2))
        self.assertEqual(m["confusion"], [[3, 1], [2, 2]])
        self.assertAlmostEqual(m["accuracy"], 5 / 8)
        self.assertAlmostEqual(m["precision"], 2 / 3)
        self.assertAlmostEqual(m["recall"], 1 / 2)
        self.assertAlmostEqual(m["f1"], 4 / 7)
        self.assertAlmostEqual(m["balanced_accuracy"], (1 / 2 + 3 / 4) / 2)
        self.assertEqual((m["alerts"], m["alert_rate"]), (3, 3 / 8))

    def test_average_precision_groups_tied_scores(self):
        # tie at 0.8 holds one positive of two: AP = 0.5*(1/2) + 0.5*(2/3)
        y = np.array([1, 0, 1, 0])
        self.assertAlmostEqual(ex.average_precision(y, np.array([0.8, 0.8, 0.5, 0.1])), 0.25 + 1 / 3)

    def test_average_precision_matches_sklearn_with_many_ties(self):
        rng = np.random.default_rng(111)
        y = rng.integers(0, 2, 400)
        for score in (rng.integers(0, 2, 400).astype(float), rng.integers(0, 9, 400) / 9, rng.random(400)):
            self.assertAlmostEqual(ex.average_precision(y, score), average_precision_score(y, score), delta=1e-10)

    def test_no_positive_predictions_reports_zero_with_note(self):
        m = ex.flag_metrics(np.array([0, 1]), np.array([0, 0]))
        self.assertEqual((m["precision"], m["f1"]), (0.0, 0.0))
        self.assertIn("no positive predictions", m["note"])

    def test_single_class_outcome_is_refused(self):
        with self.assertRaisesRegex(data.DataError, "both classes"):
            ex.flag_metrics(np.array([1, 1]), np.array([0, 1]))


class ThresholdTests(unittest.TestCase):
    def test_best_f1_is_chosen_and_candidates_listed(self):
        t, table = ex.choose_threshold(np.array([1, 1, 0, 0]), np.array([0.95, 0.65, 0.55, 0.05]))
        self.assertEqual(t, 0.6)
        self.assertEqual([r["threshold"] for r in table], [k / 10 for k in range(1, 10)])

    def test_f1_tie_prefers_higher_precision(self):
        # t=.1: tp=2 fp=2 fn=0 -> F1 2/3, P 1/2; t=.4: tp=1 fp=0 fn=1 -> F1 2/3, P 1
        t, _ = ex.choose_threshold(np.array([1, 1, 0, 0]), np.array([0.45, 0.11, 0.12, 0.13]))
        self.assertEqual(t, 0.4)

    def test_full_tie_prefers_higher_threshold(self):
        t, _ = ex.choose_threshold(np.array([1, 0]), np.array([0.75, 0.05]))
        self.assertEqual(t, 0.7)

    def test_score_equal_to_threshold_is_flagged(self):
        self.assertEqual(ex.flag(np.array([0.3, 0.29999]), 0.3).tolist(), [1, 0])


class FakeModel:
    def __init__(self, classes, proba):
        self.classes_, self.proba = np.array(classes), np.array(proba)

    def predict_proba(self, x):
        return self.proba


class ModelTests(unittest.TestCase):
    def test_exact_approved_settings_and_no_knn(self):
        models = ex.make_models()
        self.assertEqual(list(models), ["logistic", "forest"])
        lr, rf = (models[n].named_steps["model"] for n in models)
        self.assertIsInstance(lr, LogisticRegression)
        self.assertIsInstance(rf, RandomForestClassifier)
        want = dict(C=1.0, l1_ratio=0.0, solver="lbfgs", fit_intercept=True, max_iter=1000, tol=1e-4,
                    class_weight=None, random_state=111)
        self.assertEqual({k: lr.get_params()[k] for k in want}, want)
        want = dict(n_estimators=100, criterion="gini", max_depth=8, min_samples_leaf=5, min_samples_split=2,
                    max_features="sqrt", bootstrap=True, class_weight=None, random_state=111, n_jobs=1)
        self.assertEqual({k: rf.get_params()[k] for k in want}, want)
        source = "".join(p.read_text(encoding="utf-8") for p in (data.ROOT / "freshstock").glob("*.py"))
        self.assertNotIn("Neighbors", source)

    def test_positive_score_uses_classes_index(self):
        self.assertEqual(ex.positive_scores(FakeModel([1, 0], [[0.9, 0.1]]), None).tolist(), [0.9])

    def test_nonfinite_score_is_refused(self):
        with self.assertRaisesRegex(data.DataError, "finite"):
            ex.positive_scores(FakeModel([0, 1], [[np.nan, np.nan]]), None)


@unittest.skipUnless(HAS_COHORT, "run: python -m freshstock audit")
class RunTests(unittest.TestCase):
    """Uses train/validation only, so running the tests never scores the final test split."""

    @classmethod
    def setUpClass(cls):
        cls.cohort, cls.receipt = data.load_cohort()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.run_dir = ex.train(cls.cohort, cls.receipt, Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def copy_run(self):
        d = Path(tempfile.mkdtemp(dir=self.tmp.name)) / "run"
        shutil.copytree(self.run_dir, d)
        return d

    def test_training_never_reads_the_test_partition(self):
        before_test = self.cohort[self.cohort["dt"] <= "2024-06-10"].reset_index(drop=True)
        with tempfile.TemporaryDirectory() as d:
            other = ex.train(before_test, self.receipt, Path(d))
            self.assertEqual((other / "validation_predictions.csv").read_bytes(),
                             (self.run_dir / "validation_predictions.csv").read_bytes())

    def test_scaler_statistics_are_train_only(self):
        x, _ = data.partition(self.cohort, "train")
        scaler = ex.load_run(self.run_dir)[1]["logistic"].named_steps["prepare"].named_transformers_["numeric"]
        np.testing.assert_allclose(scaler.mean_, x[data.NUMERIC].mean().to_numpy(), atol=1e-12)

    def test_models_use_eleven_features_without_ids_or_outcome(self):
        for model in ex.load_run(self.run_dir)[1].values():
            self.assertEqual(model.named_steps["model"].n_features_in_, 11)
            self.assertEqual(list(model.feature_names_in_), data.FEATURES)

    def test_manifest_records_frozen_thresholds_before_any_test_output(self):
        manifest = json.loads((self.run_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "TRAINED_FROZEN")
        self.assertEqual(set(manifest["thresholds"]), {"logistic", "forest"})
        self.assertEqual(set(manifest["artifacts"]), {"logistic.joblib", "forest.joblib", "validation.json",
                                                      "validation_predictions.csv"})
        self.assertFalse((self.run_dir / "evaluation.json").exists())

    def test_changed_model_is_refused(self):
        d = self.copy_run()
        with open(d / "forest.joblib", "ab") as f:
            f.write(b"x")
        with self.assertRaisesRegex(data.DataError, "forest.joblib"):
            ex.load_run(d)

    def test_missing_model_is_refused(self):
        d = self.copy_run()
        (d / "logistic.joblib").unlink()
        with self.assertRaisesRegex(data.DataError, "logistic.joblib"):
            ex.load_run(d)

    def test_incompatible_library_version_is_refused(self):
        d = self.copy_run()
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        m["environment"]["scikit-learn"] = "0.0"
        (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
        with self.assertRaisesRegex(data.DataError, "scikit-learn"):
            ex.load_run(d)

    def test_failed_training_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            bad = self.cohort[self.cohort["dt"] != "2024-05-01"]
            with self.assertRaises(data.DataError):
                ex.train(bad, self.receipt, Path(d))
            self.assertEqual(list(Path(d).glob("*/manifest.json")), [])
            note = next(Path(d).glob("*.failed/FAILED.json")).read_text(encoding="utf-8")
            self.assertEqual(json.loads(note)["status"], "FAILED")

    def test_evaluation_matches_validation_receipt_and_refuses_second_scoring(self):
        d = self.copy_run()
        result = ex.evaluate(d, self.cohort, split="validation")
        self.assertEqual(result["split"], "validation")
        self.assertEqual(set(result["estimators"]), {"logistic", "forest", "persistence"})
        predictions = ex.load_evaluation(d)[1]
        self.assertEqual(list(predictions.columns), ex.PREDICTION_COLUMNS)
        self.assertEqual(len(predictions), 1500)
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        candidates = json.loads((d / "validation.json").read_text(encoding="utf-8"))["logistic"]["candidates"]
        chosen = next(r for r in candidates if r["threshold"] == manifest["thresholds"]["logistic"])
        self.assertEqual(result["estimators"]["logistic"]["tp"], chosen["tp"])
        with self.assertRaisesRegex(data.DataError, "already"):
            ex.evaluate(d, self.cohort, split="validation")

    def test_tampered_predictions_are_refused(self):
        d = self.copy_run()
        ex.evaluate(d, self.cohort, split="validation")
        p = d / "predictions.csv"
        p.write_bytes(p.read_bytes().replace(b",1,", b",0,", 1))
        with self.assertRaisesRegex(data.DataError, "predictions.csv"):
            ex.load_evaluation(d)


if __name__ == "__main__":
    unittest.main()
