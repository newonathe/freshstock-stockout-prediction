"""Train the two fixed models, freeze validation thresholds, and score the final test once (B3 v1)."""
import datetime as dt
import json
import os
import platform
import subprocess
import time
import warnings
from fractions import Fraction
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics as skm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import data
from .data import KEY, FEATURES, NUMERIC, require

PROTOCOL = "B3 v1"
SEED = 111
CANDIDATES = [k / 10 for k in range(1, 10)]
MODELS = ["logistic", "forest"]
ESTIMATORS = MODELS + ["persistence"]
TITLES = {"logistic": "Logistic Regression", "forest": "Random Forest",
          "persistence": "Persistence rule (not a trained model)"}
PREDICTION_COLUMNS = KEY + ["dt", "y_true", "logistic_score", "logistic_flag", "forest_score", "forest_flag",
                            "persistence_flag"]
ARTIFACTS = ["logistic.joblib", "forest.joblib", "validation.json", "validation_predictions.csv"]


# ---------- models ----------

def make_models():
    def pipeline(numeric, model):
        weekday = OneHotEncoder(categories=[list(range(7))], handle_unknown="error", sparse_output=False)
        prepare = ColumnTransformer([("numeric", numeric, NUMERIC), ("weekday", weekday, ["weekday"])])
        return Pipeline([("prepare", prepare), ("model", model)])

    return {
        "logistic": pipeline(StandardScaler(), LogisticRegression(
            C=1.0, l1_ratio=0.0, solver="lbfgs", fit_intercept=True, max_iter=1000, tol=1e-4,
            class_weight=None, random_state=SEED)),
        "forest": pipeline("passthrough", RandomForestClassifier(
            n_estimators=100, criterion="gini", max_depth=8, min_samples_leaf=5, min_samples_split=2,
            max_features="sqrt", bootstrap=True, class_weight=None, random_state=SEED, n_jobs=1)),
    }


def positive_scores(model, x):
    """Score of the positive class, located through classes_ (not a calibrated probability)."""
    score = model.predict_proba(x)[:, list(model.classes_).index(1)]
    require(bool(np.isfinite(score).all() and (score >= 0).all() and (score <= 1).all()),
            "model produced a score that is not finite in [0, 1]")
    return score


def flag(score, threshold):
    return (score >= threshold).astype(int)


def persistence_flags(x):
    return (x["stockout_hours_lag1"].to_numpy() > 0).astype(int)


# ---------- metrics computed from counts, independent of sklearn ----------

def flag_metrics(y, flags):
    y, flags = np.asarray(y), np.asarray(flags)
    tp, fp = int(((y == 1) & (flags == 1)).sum()), int(((y == 0) & (flags == 1)).sum())
    fn, tn = int(((y == 1) & (flags == 0)).sum()), int(((y == 0) & (flags == 0)).sum())
    require(tp + fn > 0 and tn + fp > 0, "both classes must be present")
    recall = tp / (tp + fn)
    out = {"n": len(y), "tn": tn, "fp": fp, "fn": fn, "tp": tp, "confusion": [[tn, fp], [fn, tp]],
           "accuracy": (tp + tn) / len(y), "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": recall,
           "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp else 0.0,
           "balanced_accuracy": (recall + tn / (tn + fp)) / 2,
           "alerts": tp + fp, "alert_rate": (tp + fp) / len(y)}
    if tp + fp == 0:
        out["note"] = "no positive predictions; precision and F1 reported as 0"
    return out


def average_precision(y, score):
    """AP = sum over distinct descending scores of (recall step) * precision; tied scores form one step."""
    y, score = np.asarray(y), np.asarray(score)
    total, seen, hits, last_recall, ap = int(y.sum()), 0, 0, 0.0, 0.0
    for s in np.unique(score)[::-1]:
        group = score == s
        seen += int(group.sum())
        hits += int(y[group].sum())
        ap += (hits / total - last_recall) * (hits / seen)
        last_recall = hits / total
    return ap


def choose_threshold(y, score):
    """Max F1, then higher precision, then higher threshold; ties compared as exact fractions."""
    table, best = [], None
    for t in CANDIDATES:
        m = flag_metrics(y, flag(score, t))
        table.append({"threshold": t, **{k: m[k] for k in ("tn", "fp", "fn", "tp", "precision", "recall", "f1")}})
        f1 = Fraction(2 * m["tp"], 2 * m["tp"] + m["fp"] + m["fn"])
        precision = Fraction(m["tp"], m["alerts"]) if m["alerts"] else Fraction(0)
        best = max(best or (f1, precision, t), (f1, precision, t))
    return best[2], table


def summarize(predictions):
    """All reported numbers, recomputed only from the saved prediction rows."""
    y = predictions["y_true"].to_numpy()
    result = {}
    for name in ESTIMATORS:
        flags = predictions[f"{name}_flag"].to_numpy()
        score = predictions[f"{name}_score"].to_numpy() if name in MODELS else flags.astype(float)
        entry = {"average_precision": average_precision(y, score), **flag_metrics(y, flags)}
        entry["daily"] = [{"dt": day, "positives": int(g["y_true"].sum()), "alerts": int(g[f"{name}_flag"].sum()),
                           "misses": int(((g["y_true"] == 1) & (g[f"{name}_flag"] == 0)).sum())}
                          for day, g in predictions.groupby("dt")]
        entry["first_errors"] = {}
        for kind, rows in (("false_alert", (y == 0) & (flags == 1)), ("miss", (y == 1) & (flags == 0))):
            hit = predictions[rows].head(1)
            entry["first_errors"][kind] = json.loads(hit.to_json(orient="records"))[0] if len(hit) else None
        result[name] = entry
    return result


def check_against_sklearn(y, scores, flags, estimators):
    for name in ESTIMATORS:
        ours, f = estimators[name], flags[name]
        require(skm.confusion_matrix(y, f, labels=[0, 1]).tolist() == ours["confusion"], f"{name}: confusion differs")
        for key, theirs in (("average_precision", skm.average_precision_score(y, scores[name])),
                            ("accuracy", skm.accuracy_score(y, f)),
                            ("precision", skm.precision_score(y, f, zero_division=0)),
                            ("recall", skm.recall_score(y, f)), ("f1", skm.f1_score(y, f, zero_division=0)),
                            ("balanced_accuracy", skm.balanced_accuracy_score(y, f))):
            require(abs(ours[key] - theirs) <= 1e-10, f"{name}: {key} differs from scikit-learn")


# ---------- runs ----------

def environment():
    names = ["scikit-learn", "numpy", "scipy", "pandas", "joblib"]
    return {"python": platform.python_version(), "system": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), **{n: version(n) for n in names}}


def code_commit():
    try:
        def git(*args):
            return subprocess.run(["git", *args], cwd=data.ROOT, capture_output=True, text=True, check=True).stdout
        dirty = bool(git("status", "--porcelain", "--", "freshstock", "requirements.txt").strip())
        return git("rev-parse", "HEAD").strip() + (" (uncommitted changes)" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=1), encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_predictions(path, x, y, scores, flags):
    out = x[KEY + ["dt"]].copy()
    out["y_true"] = y
    for name in MODELS:
        out[f"{name}_score"], out[f"{name}_flag"] = scores[name], flags[name]
    out["persistence_flag"] = flags["persistence"]
    out[PREDICTION_COLUMNS].to_csv(path, index=False, lineterminator="\n")


def score_all(models, thresholds, x):
    scores, seconds = {}, {}
    for name in MODELS:
        start = time.perf_counter()
        scores[name] = positive_scores(models[name], x[FEATURES])
        seconds[name] = time.perf_counter() - start
    flags = {n: flag(scores[n], thresholds[n]) for n in MODELS}
    flags["persistence"] = persistence_flags(x)
    scores["persistence"] = flags["persistence"].astype(float)
    return scores, flags, seconds


def train(cohort, receipt, runs_dir):
    """Fit on train, pick thresholds on validation, publish a frozen run folder. Never reads test rows."""
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    stage, final = runs_dir / f"{run_id}.partial", runs_dir / run_id
    stage.mkdir(parents=True)
    try:
        cohort = cohort[cohort["dt"] <= data.SPLITS["validation"][1]].reset_index(drop=True)
        x_train, y_train = data.partition(cohort, "train")
        x_val, y_val = data.partition(cohort, "validation")
        models, thresholds, validation, fit_seconds, params = make_models(), {}, {}, {}, {}
        for name, model in models.items():
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # a convergence or deprecation warning aborts the run
                start = time.perf_counter()
                model.fit(x_train[FEATURES], y_train)
                fit_seconds[name] = time.perf_counter() - start
            score = positive_scores(model, x_val[FEATURES])
            thresholds[name], candidates = choose_threshold(y_val, score)
            validation[name] = {"threshold": thresholds[name], "average_precision": average_precision(y_val, score),
                                "candidates": candidates}
            params[name] = model.named_steps["model"].get_params()
            joblib.dump(model, stage / f"{name}.joblib")
        scores, flags, _ = score_all(models, thresholds, x_val)
        save_predictions(stage / "validation_predictions.csv", x_val, y_val, scores, flags)
        write_json(stage / "validation.json", validation)
        write_json(stage / "manifest.json", {
            "status": "TRAINED_FROZEN", "run_id": run_id, "created_utc": now(), "protocol": PROTOCOL,
            "code_commit": code_commit(), "seed": SEED,
            "source_url": receipt["source_url"], "source_sha256": receipt["source_sha256"],
            "cohort_digest": receipt["cohort_digest"], "cohort_sha256": receipt["cohort_sha256"],
            "splits": data.SPLITS, "features": FEATURES,
            "label": "1 if stock_hour6_22_cnt > 0 on the target date, else 0",
            "threshold_rule": "validation max F1, then higher precision, then higher threshold; score >= threshold",
            "thresholds": thresholds, "params": params, "fit_seconds": fit_seconds,
            "train_rows": len(x_train), "validation_rows": len(x_val), "environment": environment(),
            "artifacts": {a: data.sha256_file(stage / a) for a in ARTIFACTS},
            "limitations": ["fixed 100-series cohort from one 90-day dataset", "scores are not calibrated",
                            "recorded stockout hours are not lost demand", "peak memory not measured"]})
        stage.rename(final)
        return final
    except BaseException as error:
        write_json(stage / "FAILED.json", {"status": "FAILED", "reason": f"{type(error).__name__}: {error}"})
        (stage / "manifest.json").unlink(missing_ok=True)
        stage.rename(runs_dir / f"{run_id}.failed")
        raise


def load_run(run):
    """Return (manifest, models) only if every frozen file is unchanged and library versions match."""
    require((run / "manifest.json").is_file(), f"{run} is not a completed run. Run: python -m freshstock train")
    manifest = read_json(run / "manifest.json")
    require(manifest["status"] == "TRAINED_FROZEN" and manifest["protocol"] == PROTOCOL, "incompatible run manifest")
    for name, digest in manifest["artifacts"].items():
        require((run / name).is_file() and data.sha256_file(run / name) == digest,
                f"{name} is missing or was changed after training; train again")
    for lib in ("scikit-learn", "numpy"):
        require(manifest["environment"][lib] == version(lib),
                f"{lib} {manifest['environment'][lib]} is required but {version(lib)} is installed; "
                "install requirements.txt")
    return manifest, {n: joblib.load(run / f"{n}.joblib") for n in MODELS}


def evaluate(run, cohort, split="test"):
    """Score one split with the frozen bundle exactly once and verify every metric two ways."""
    require(not (run / "evaluation.json").exists(), "this run was already evaluated; results are final")
    manifest, models = load_run(run)
    x, y = data.partition(cohort, split)
    scores, flags, seconds = score_all(models, manifest["thresholds"], x)
    path = run / "predictions.csv"
    save_predictions(path, x, y, scores, flags)
    saved = pd.read_csv(path, float_precision="round_trip")
    require(saved[KEY + ["dt"]].equals(x[KEY + ["dt"]]) and not saved.duplicated(KEY + ["dt"]).any(),
            "saved predictions do not match the split rows")
    estimators = summarize(saved)
    check_against_sklearn(y, scores, flags, estimators)
    result = {"split": split, "rows": len(x), "positives": int(y.sum()), "negatives": int((1 - y).sum()),
              "prevalence": float(y.mean()), "thresholds": manifest["thresholds"], "estimators": estimators,
              "score_seconds": seconds, "verification": "recomputed from predictions.csv; scikit-learn parity <= 1e-10"}
    write_json(run / "metrics.json", result)
    write_json(run / "evaluation.partial", {
        "status": "EVALUATED", "evaluated_utc": now(), "split": split,
        "manifest_sha256": data.sha256_file(run / "manifest.json"),
        "artifacts": {a: data.sha256_file(run / a) for a in ("predictions.csv", "metrics.json")}})
    os.replace(run / "evaluation.partial", run / "evaluation.json")  # written last: marks the run complete
    return result


def load_evaluation(run):
    """Return (metrics, predictions) after checking the evaluation receipt."""
    require((run / "evaluation.json").is_file(), f"{run.name} has no results yet. Run: python -m freshstock evaluate")
    receipt = read_json(run / "evaluation.json")
    require(receipt["manifest_sha256"] == data.sha256_file(run / "manifest.json"), "manifest.json changed")
    for name, digest in receipt["artifacts"].items():
        require((run / name).is_file() and data.sha256_file(run / name) == digest, f"{name} is missing or was changed")
    return read_json(run / "metrics.json"), pd.read_csv(run / "predictions.csv", float_precision="round_trip")
