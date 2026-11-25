"""Bounded tuning, refit, calibration, freeze and the one-time holdout evaluation (EXP-02)."""
import datetime as dt
import hashlib
import io
import json
import os
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ..data import KEY, ROOT, require, sha256_file
from ..experiment import CANDIDATES, choose_threshold, code_commit, environment, now, read_json
from . import data, metrics
from .metrics import ESTIMATORS, MODELS
from .resources import Guard, snapshot

NAME, SEED, TIE = "EXP-02 v1", 111, 1e-6
GRIDS = {
    "full": {"logistic": [{"C": .1}, {"C": 1.}, {"C": 10.}],
             "forest": [{"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 5},
                        {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 10},
                        {"n_estimators": 150, "max_depth": 16, "min_samples_leaf": 20}]},
    "reduced": {"logistic": [{"C": 1.}], "forest": [{"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 5}]}}
CODE = ["freshstock/data.py", "freshstock/experiment.py", "freshstock/expanded/data.py",
        "freshstock/expanded/experiment.py", "freshstock/expanded/metrics.py"]
LIBRARIES = ["scikit-learn", "numpy", "scipy", "pandas", "joblib", "pyarrow", "threadpoolctl"]
FROZEN_FILES = ["protocol.json", "candidate_metrics.json", "logistic.joblib", "forest.joblib", "resource.json",
                "freeze.json"]
EVALUATED_FILES = ["evaluation_started.json", "days", "predictions.parquet", "metrics.json", "evaluation.json"]
QUARANTINE = "evaluation_quarantined.json"
PREDICTION_COLUMNS = ["dt"] + KEY + ["y_true", "logistic_score", "forest_score", "logistic_flag", "forest_flag",
                                     "persistence_flag", "past_stockout"]
CHUNK = {"logistic_score": np.float64, "forest_score": np.float64, "logistic_flag": np.uint8, "forest_flag": np.uint8,
         "persistence_flag": np.uint8, "past_stockout": np.uint8}
LIMITATIONS = [
    "seven consecutive days and correlated rows; no significance claim",
    "same publisher and same series as training; not an independent retailer",
    "sales are multiplied by an undocumented coefficient",
    "recorded stockout hours are not lost demand; no business outcome is measured",
    "different cohort and dates than B3, so the numbers are not comparable with it"]


def write_json(path, value):
    """Atomic; NaN and infinity are refused."""
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(value, indent=1, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


# ---------- protocol ----------

def code_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for name in CODE}


def candidates(grid):
    return [{"id": f"{s}-{family}-" + "-".join(str(v) for v in params.values()), "feature_set": s, "family": family,
             "params": params}
            for s in ("A", "B") for family in MODELS for params in GRIDS[grid][family]]


def _fixed(train, holdout, n, grid, calibration, threads, digest):
    """Everything that a run must not change."""
    return {"name": NAME, "seed": SEED, "n_series": n, "cohort_digest": digest, "grid": grid,
            "calibration": calibration, "threads": threads,
            "sources": {s.name: {"sha256": s.sha256, "size": s.size, "first_day": s.first_day.isoformat(),
                                 "n_days": s.n_days, "n_keys": s.n_keys} for s in (train, holdout)},
            "revision": data.REVISION, "target": "1 if stock_hour6_22_cnt > 0 on the target date, else 0",
            "splits": {k: [a.isoformat(), b.isoformat()] for k, (a, b) in data.SPLITS.items()},
            "features": data.FEATURES, "candidates": candidates(grid),
            "selection_rule": f"per family: highest tuning AP; within {TIE} of it set A, then the simpler setting",
            "calibration_rule": "sigmoid only if its Brier score on the threshold dates is strictly lower than raw",
            "threshold_candidates": CANDIDATES,
            "threshold_rule": "max F1, then higher precision, then higher threshold; score >= threshold",
            "review_rule": "k = ceil(0.10 * series) per day, descending score then ascending key",
            "primary_metric": "pooled average precision on the holdout", "baseline": "stockout yesterday",
            "environment": {name: version(name) for name in LIBRARIES}, "code_sha256": code_hashes()}


def _cohort(panel, n):
    rows, digest = data.cohort_rows(panel.keys, n)
    return panel.take(rows), digest


def build_protocol(train, holdout, n, grid, calibration, threads, decision):
    """`holdout` supplies only its pinned identity; its file is not opened."""
    require(calibration in ("sigmoid", "off"), "calibration is 'sigmoid' or 'off'")
    digest = data.cohort_rows(data.read_panel(train).keys, n)[1]
    return {**_fixed(train, holdout, n, grid, calibration, threads, digest), "resource_decision": decision,
            "created_utc": now()}


def _check_protocol(protocol, train, holdout, digest):
    expected = _fixed(train, holdout, protocol["n_series"], protocol["grid"], protocol["calibration"],
                      protocol["threads"], digest)
    if protocol.get("code_sha256") != expected["code_sha256"]:
        require(False, "protocol: the code changed after the protocol lock")
    for key, value in json.loads(json.dumps(expected)).items():
        require(protocol.get(key) == value, f"protocol: {key} differs from this program or data")


def _holdout_identity(protocol):
    s = protocol["sources"]["holdout"]
    return data.Source("holdout", Path("not-opened"), s["sha256"], s["size"], dt.date.fromisoformat(s["first_day"]),
                       s["n_days"], s["n_keys"])


# ---------- models ----------

def make_pipeline(family, feature_set, params, n_jobs):
    if family == "forest":
        return Pipeline([("model", RandomForestClassifier(
            **params, criterion="gini", min_samples_split=2, max_features="sqrt", bootstrap=True, class_weight=None,
            random_state=SEED, n_jobs=n_jobs))])
    k, total = data.N_NUMERIC[feature_set], len(data.FEATURES[feature_set])
    prepare = ColumnTransformer([("numeric", StandardScaler(), list(range(k))),
                                 ("weekday", "passthrough", list(range(k, total)))])
    return Pipeline([("prepare", prepare), ("model", LogisticRegression(
        **params, l1_ratio=0.0, solver="lbfgs", fit_intercept=True, max_iter=1000, tol=1e-4, class_weight=None,
        random_state=SEED))])


def fit_model(family, feature_set, params, x, y, threads):
    """Fit one pipeline; any warning (such as no convergence) stops the run. Scoring then uses one thread."""
    pipe = make_pipeline(family, feature_set, params, threads["forest_n_jobs"])
    with warnings.catch_warnings(), threadpool_limits(1 if family == "forest" else threads["blas_threads"]):
        warnings.simplefilter("error")
        pipe.fit(x, y)
    if family == "forest":
        pipe.named_steps["model"].set_params(n_jobs=1)
    return pipe


def positive_scores(model, x):
    with threadpool_limits(1):
        score = model.predict_proba(x)[:, list(model.classes_).index(1)]
    require(bool((np.isfinite(score) & (score >= 0) & (score <= 1)).all()), "model score is not finite in [0, 1]")
    return score


def base_pipeline(model):
    """The fitted pipeline inside a calibrated model, or the model itself."""
    return model.estimator.estimator if isinstance(model, CalibratedClassifierCV) else model


def fit_sigmoid(base, x, y):
    require(len(np.unique(y)) == 2, "calibration dates need both classes")
    with threadpool_limits(1):
        return CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid", ensemble=False).fit(x, y)


def _complexity(c):
    p = c["params"]
    size = (p["C"],) if c["family"] == "logistic" else (p["n_estimators"], p["max_depth"], -p["min_samples_leaf"])
    return (c["feature_set"] != "A", *size)


def select(results):
    chosen = {}
    for family in MODELS:
        rows = [r for r in results if r["family"] == family]
        best = max(r["tuning_average_precision"] for r in rows)
        chosen[family] = min((r for r in rows if r["tuning_average_precision"] >= best - TIE), key=_complexity)
    return chosen


def _calibrate(family, base, x_cal, y_cal, x_val, y_val, mode):
    """Return (selected model, its validation scores, record). The base model is never refitted."""
    raw = positive_scores(base, x_val)
    record = {"raw": {"brier": metrics.brier(y_val, raw), "log_loss": metrics.log_loss(y_val, raw),
                      "bins": metrics.reliability(y_val, raw)},
              "sigmoid": None, "chosen": "raw", "base_unchanged": True, "failure": None}
    if mode == "off":
        return base, raw, record
    try:
        model = fit_sigmoid(base, x_cal, y_cal)
        score = positive_scores(model, x_val)
        require(len(np.unique(y_val)) == 2, "threshold dates need both classes")
    except Exception as error:  # the protocol names raw scores as the fallback
        record["failure"] = f"{type(error).__name__}: {error}"
        return base, raw, record
    record["base_unchanged"] = bool(np.array_equal(raw, positive_scores(base, x_val)))
    require(record["base_unchanged"], f"{family}: calibration changed the fitted model")
    fitted = model.calibrated_classifiers_[0].calibrators[0]
    record["sigmoid"] = {"a": float(fitted.a_), "b": float(fitted.b_), "brier": metrics.brier(y_val, score),
                         "log_loss": metrics.log_loss(y_val, score), "bins": metrics.reliability(y_val, score)}
    if record["sigmoid"]["brier"] < record["raw"]["brier"]:
        record["chosen"] = "sigmoid"
        return model, score, record
    return base, raw, record


# ---------- fit ----------

def fit(protocol_path, runs_dir, train=data.TRAIN, stop=None, timeout=None):
    """Tune, refit, calibrate, choose thresholds and freeze. Only the training source is opened."""
    protocol = read_json(protocol_path)
    digest_of_protocol = sha256_file(protocol_path)
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    for other in runs_dir.iterdir():
        used = other / "protocol.json"
        require(not used.is_file() or sha256_file(used) != digest_of_protocol,
                f"protocol already used by {other.name}; results are final or invalid, see its files")
    start = time.perf_counter()
    panel, digest = _cohort(data.read_panel(train), protocol["n_series"])
    _check_protocol(protocol, train, _holdout_identity(protocol), digest)
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    stage, final = runs_dir / f"{run_id}.partial", runs_dir / run_id
    stage.mkdir()
    (stage / "protocol.json").write_bytes(Path(protocol_path).read_bytes())
    stages, rows = {"start": snapshot(runs_dir), "read_seconds": round(time.perf_counter() - start, 3)}, {}
    threads = protocol["threads"]
    try:
        results = []
        for s in ("A", "B"):
            x, y = data.split(panel, "initial_fit", s)
            x_tune, y_tune = data.split(panel, "tuning", s)
            _count(rows, initial_fit=y, tuning=y_tune)
            for c in (c for c in protocol["candidates"] if c["feature_set"] == s):
                with Guard(timeout=timeout, stop=stop) as guard:
                    model = fit_model(c["family"], s, c["params"], x, y, threads)
                begin = time.perf_counter()
                score = positive_scores(model, x_tune)
                report = guard.report()
                results.append({**c, "tuning_average_precision": metrics.average_precision(y_tune, score),
                                "tuning_brier": metrics.brier(y_tune, score), "fit_seconds": report["seconds"],
                                "score_seconds": round(time.perf_counter() - begin, 3), **_size(model),
                                "peak_additional_bytes": report["peak_additional_bytes"],
                                "min_available_ram": report["min_available_ram"]})
                write_json(stage / "candidate_metrics.json", {"status": "PARTIAL", "candidates": results})
                del model
            del x, y, x_tune, y_tune
        chosen = select(results)
        write_json(stage / "candidate_metrics.json", {
            "status": "COMPLETE", "candidates": results, "selected": {f: chosen[f]["id"] for f in MODELS},
            "same_feature_comparison": {s: {f: max(r["tuning_average_precision"] for r in results
                                                   if r["feature_set"] == s and r["family"] == f) for f in MODELS}
                                        for s in ("A", "B")}})
        stages["tuning_seconds"] = round(time.perf_counter() - start, 3)

        calibration, thresholds, tables, refit = {}, {}, {}, {}
        for family in MODELS:
            c = chosen[family]
            x, y = data.split(panel, "refit", c["feature_set"])
            with Guard(timeout=timeout, stop=stop) as guard:
                base = fit_model(family, c["feature_set"], c["params"], x, y, threads)
            _count(rows, refit=y)
            del x, y
            x_cal, y_cal = data.split(panel, "calibration", c["feature_set"])
            x_val, y_val = data.split(panel, "threshold", c["feature_set"])
            _count(rows, calibration=y_cal, threshold=y_val)
            model, score, calibration[family] = _calibrate(family, base, x_cal, y_cal, x_val, y_val,
                                                           protocol["calibration"])
            thresholds[family], tables[family] = choose_threshold(y_val, score)
            joblib.dump(model, stage / f"{family}.joblib", compress=3)
            refit[family] = {**guard.report(), **_size(base), "file_bytes": (stage / f"{family}.joblib").stat().st_size,
                             "threshold_average_precision": metrics.average_precision(y_val, score)}
            del x_cal, y_cal, x_val, y_val, base, model
        write_json(stage / "resource.json", {"stages": stages, "refit": refit, "end": snapshot(runs_dir),
                                             "total_seconds": round(time.perf_counter() - start, 3)})
        write_json(stage / "freeze.json", {
            "status": "FROZEN", "run_id": run_id, "created_utc": now(), "protocol": NAME,
            "protocol_sha256": digest_of_protocol, "code_commit": code_commit(), "cohort_digest": digest,
            "source_sha256": train.sha256, "n_series": len(panel.keys), "rows": {k: n for k, (n, _) in rows.items()},
            "positives": {k: p for k, (_, p) in rows.items()},
            "selected": {f: {k: chosen[f][k] for k in ("id", "feature_set", "family", "params")} for f in MODELS},
            "calibration": calibration, "thresholds": thresholds, "threshold_tables": tables,
            "features": data.FEATURES, "environment": environment(), "limitations": LIMITATIONS,
            "artifacts": {name: sha256_file(stage / name) for name in FROZEN_FILES if name != "freeze.json"}})
        stage.rename(final)
        return final
    except BaseException as error:
        write_json(stage / "FAILED.json", {"status": "FAILED", "reason": f"{type(error).__name__}: {error}"})
        stage.rename(runs_dir / f"{run_id}.failed")
        raise


def _count(rows, **outcomes):
    rows.update({name: (len(y), int(y.sum())) for name, y in outcomes.items()})


def _size(model):
    step = model.named_steps["model"]
    if isinstance(step, RandomForestClassifier):
        return {"nodes": int(sum(t.tree_.node_count for t in step.estimators_)),
                "deepest_tree": int(max(t.tree_.max_depth for t in step.estimators_))}
    return {"iterations": int(step.n_iter_[0])}


# ---------- trusted loading ----------

def verified_bytes(path, digest):
    require(path.is_file() and not path.is_symlink() and not path.is_junction(), f"{path.name} is missing")
    content = path.read_bytes()
    require(hashlib.sha256(content).hexdigest() == digest, f"{path.name} was changed after the freeze")
    return content


def load_run(run, freeze_sha256):
    """(freeze, models) after the pinned freeze hash, the exact file set and every file hash are verified."""
    run = Path(run)
    require(run.is_dir() and not run.is_symlink() and not run.is_junction(), f"{run} is not a run folder")
    content = verified_bytes(run / "freeze.json", freeze_sha256)
    allowed = set(FROZEN_FILES) | set(EVALUATED_FILES) | {QUARANTINE}
    for item in run.iterdir():
        require(item.name in allowed and (item.is_dir() if item.name == "days" else item.is_file())
                and not item.is_symlink() and not item.is_junction(), f"unexpected item {item.name} in the run folder")
    freeze = json.loads(content)
    require(freeze["status"] == "FROZEN" and freeze["protocol"] == NAME, "incompatible freeze.json")
    require(set(freeze["artifacts"]) == set(FROZEN_FILES) - {"freeze.json"}, "freeze.json lists other files")
    verified = {name: verified_bytes(run / name, digest) for name, digest in freeze["artifacts"].items()}
    for lib in ("scikit-learn", "numpy"):
        require(freeze["environment"][lib] == version(lib),
                f"{lib} {freeze['environment'][lib]} is required but {version(lib)} is installed")
    return freeze, {f: joblib.load(io.BytesIO(verified[f"{f}.joblib"])) for f in MODELS}


# ---------- one-time evaluation ----------

def score_day(models, freeze, history, day):
    """Scores and flags from dates before `day` only."""
    require(int(history.dates[-1]) == day.toordinal() - 1, "history must end on the day before the target")
    x = data.features_on(history, day)
    out = {}
    for f in MODELS:
        out[f"{f}_score"] = positive_scores(models[f], x[:, data.COLUMNS[freeze["selected"][f]["feature_set"]]])
        out[f"{f}_flag"] = (out[f"{f}_score"] >= freeze["thresholds"][f]).astype(np.uint8)
    past = (history.stock[:, -1] > 0).astype(np.uint8)
    return {**out, "persistence_flag": past, "past_stockout": past}


def write_intent(path, day, freeze_sha256):
    write_json(path, {"day": day.isoformat(), "freeze_sha256": freeze_sha256, "utc": now()})


def _write_chunk(path, chunk):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **chunk)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as f:
        f.write(buffer.getvalue())
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def _read_chunk(path, digest, n):
    with np.load(io.BytesIO(verified_bytes(path, digest)), allow_pickle=False) as stored:
        require(set(stored.files) == set(CHUNK), f"{path.name} has other columns")
        chunk = {name: stored[name] for name in CHUNK}
    require(all(chunk[k].dtype == t and chunk[k].shape == (n,) for k, t in CHUNK.items()), f"{path.name} is malformed")
    return chunk


def _quarantine(run, reason):
    write_json(run / QUARANTINE, {"status": "QUARANTINED", "reason": reason, "utc": now()})
    require(False, f"evaluation quarantined: {reason}. It is never rescored automatically")


def evaluate(run, freeze_sha256, train=data.TRAIN, holdout=data.HOLDOUT):
    """Score the holdout once, day by day, then join outcomes. Only days never started may resume."""
    run = Path(run)
    require(not (run / "evaluation.json").exists(), "this run was already evaluated; results are final")
    require(not (run / QUARANTINE).exists(), "this evaluation is quarantined; see evaluation_quarantined.json")
    freeze, models = load_run(run, freeze_sha256)
    protocol = read_json(run / "protocol.json")
    require(protocol["code_sha256"] == code_hashes(), "protocol: the code changed after the protocol lock")
    dev, digest = _cohort(data.read_panel(train), protocol["n_series"])
    _check_protocol(protocol, train, holdout, digest)
    fresh, _ = _cohort(data.read_panel(holdout), protocol["n_series"])
    whole = dev.join(fresh)  # later days use earlier observed holdout days as history, as of each date
    identity = {"freeze_sha256": freeze_sha256, "protocol_sha256": freeze["protocol_sha256"],
                "train_sha256": train.sha256, "holdout_sha256": holdout.sha256, "n_series": len(dev.keys)}
    marker, folder = run / "evaluation_started.json", run / "days"
    if marker.exists():
        started = read_json(marker)
        if {k: started.get(k) for k in identity} != identity:
            _quarantine(run, "evaluation_started.json does not match this run")
    else:
        require(not folder.exists(), "day results exist without a start marker")
        write_json(marker, {**identity, "status": "STARTED", "utc": now()})
    folder.mkdir(exist_ok=True)
    chunks, seconds, n = [], {}, len(dev.keys)
    for day in data.split_days("holdout"):
        intent, chunk_path, done = (folder / f"{day.isoformat()}{suffix}"
                                    for suffix in (".intent.json", ".npz", ".done.json"))
        if done.exists():
            receipt = read_json(done)
            try:
                require(receipt["freeze_sha256"] == freeze_sha256, "another freeze")
                chunks.append(_read_chunk(chunk_path, receipt["sha256"], n))
            except ValueError as error:
                _quarantine(run, f"{day.isoformat()}: {error}")
            seconds[day.isoformat()] = receipt["seconds"]
            continue
        if intent.exists() or chunk_path.exists():
            _quarantine(run, f"{day.isoformat()} was started but not completed")
        history = whole.until(day - dt.timedelta(1))
        write_intent(intent, day, freeze_sha256)
        begin = time.perf_counter()
        chunk = score_day(models, freeze, history, day)
        seconds[day.isoformat()] = round(time.perf_counter() - begin, 3)
        _write_chunk(chunk_path, chunk)
        write_json(done, {"day": day.isoformat(), "sha256": sha256_file(chunk_path), "freeze_sha256": freeze_sha256,
                          "seconds": seconds[day.isoformat()], "utc": now()})
        chunks.append(chunk)
    outcomes = [(whole.stock[:, whole.index(day)] > 0).astype(np.uint8) for day in data.split_days("holdout")]
    _save_predictions(run / "predictions.parquet", dev.keys, data.split_days("holdout"), chunks, outcomes)
    saved = load_predictions(run / "predictions.parquet")
    require(np.array_equal(saved["keys"], dev.keys) and all(
        np.array_equal(saved["scores"][f][i], c[f"{f}_score"]) for f in MODELS for i, c in enumerate(chunks)),
        "saved predictions differ from the scored days")
    summary = metrics.summarize(saved)
    metrics.check_against_sklearn(saved, summary)
    write_json(run / "metrics.json", {
        "protocol": NAME, "run_id": freeze["run_id"], "thresholds": freeze["thresholds"],
        "selected": freeze["selected"], "calibration_chosen": {f: freeze["calibration"][f]["chosen"] for f in MODELS},
        "summary": summary, "limitations": LIMITATIONS,
        "verification": "recomputed from predictions.parquet; scikit-learn parity <= 1e-10"})
    write_json(run / "evaluation.json", {  # written last: marks the run complete
        "status": "EVALUATED", "evaluated_utc": now(), **identity, "score_seconds": seconds,
        "artifacts": {a: sha256_file(run / a) for a in ("evaluation_started.json", "predictions.parquet",
                                                        "metrics.json")},
        "days": {p.name: sha256_file(p) for p in sorted(folder.iterdir())}})
    return summary


def _save_predictions(path, keys, days, chunks, outcomes):
    import pandas as pd

    parts = []
    for day, chunk, y in zip(days, chunks, outcomes):
        part = {"dt": day.isoformat(), **{k: keys[:, i] for i, k in enumerate(KEY)}, "y_true": y, **chunk}
        parts.append(pd.DataFrame(part)[PREDICTION_COLUMNS])
    temporary = path.with_name(path.name + ".tmp")
    pd.concat(parts, ignore_index=True).to_parquet(temporary, index=False)
    os.replace(temporary, path)


def load_predictions(path):
    """Saved rows as [days, series] arrays with explicit identity."""
    import pandas as pd

    rows = pd.read_parquet(path)
    require(list(rows.columns) == PREDICTION_COLUMNS, "predictions.parquet has other columns")
    days = sorted(rows["dt"].unique())
    n, d = len(rows) // len(days), len(days)
    require(len(rows) == n * d, "predictions.parquet is incomplete")
    keys = rows[KEY].to_numpy()[:n]
    require(np.array_equal(rows[KEY].to_numpy(), np.tile(keys, (d, 1))) and
            rows["dt"].tolist() == [day for day in days for _ in range(n)], "prediction rows are out of order")

    def grid(name):
        return rows[name].to_numpy().reshape(d, n)

    return {"keys": keys, "dates": np.array([dt.date.fromisoformat(x).toordinal() for x in days], dtype=np.int32),
            "y_true": grid("y_true"), "past_stockout": grid("past_stockout"),
            "scores": {f: grid(f"{f}_score") for f in MODELS}, "flags": {e: grid(f"{e}_flag") for e in ESTIMATORS}}


def verify(run, freeze_sha256):
    """Check every frozen and evaluated file and recompute all metrics from the saved rows."""
    run = Path(run)
    load_run(run, freeze_sha256)
    require((run / "evaluation.json").is_file(), "this run has not been evaluated")
    marker = read_json(run / "evaluation.json")
    require(marker["status"] == "EVALUATED" and marker["freeze_sha256"] == freeze_sha256, "evaluation.json mismatch")
    for name, digest in marker["artifacts"].items():
        verified_bytes(run / name, digest)
    require({p.name for p in (run / "days").iterdir()} == set(marker["days"]), "unexpected files in days")
    for name, digest in marker["days"].items():
        verified_bytes(run / "days" / name, digest)
    predictions = load_predictions(run / "predictions.parquet")
    summary = metrics.summarize(predictions)
    metrics.check_against_sklearn(predictions, summary)
    require(read_json(run / "metrics.json")["summary"] == json.loads(json.dumps(summary)), "metrics.json differs")
    return {"files_verified": True, "metrics_recomputed": True, "rows": summary["rows"]}
