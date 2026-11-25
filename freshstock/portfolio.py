"""Read-only view of the one approved frozen run, used by the desktop app. Never trains, tunes or rescores."""
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, demo, experiment as ex
from .data import KEY, require

RUN_ID = "20261124T184117754173"
CODE_COMMIT = "a909ff62af8a6ec6ecbc39d625dc788c85371a74"
THRESHOLDS = {"logistic": 0.4, "forest": 0.3}
# Digests recorded when the pilot run was frozen. They detect damage and mix-ups;
# they do not prove who made the package.
EXPECTED = {
    "data/derived/cohort.csv": "041f2c8a4f7bad9c349b096977c41b5437d291abb42e1d33d358286bfe77e296",
    "data/derived/cohort_manifest.json": "f8bc5163ae99003ef654fef6231faaaf500b5789964704ef897f445d35a8be35",
    **{f"runs/{RUN_ID}/{name}": digest for name, digest in {
        "manifest.json": "5d1c73645d66e4dd74b839d432ffa6fd2425e0d1b65d6a13a6fe0b8334f5a331",
        "logistic.joblib": "f3793f68e1db6867798f70179769fc1b9f04d1ec8997d62ff00f68acc1506785",
        "forest.joblib": "5b0fce450f43e2c2ead280eaac73dde0e1359f682bef8fc790472af2fc08a541",
        "validation.json": "fd49a121a54de9712b8c92913613a140928bdb3db8a81a0adc5dfd552cba0650",
        "validation_predictions.csv": "f1494fc0df683499d4a1d0cc166b1cc46e03cbc9cda910ad469e8c1abd5309de",
        "evaluation.json": "aa8cb1ed4dd790f2fd8b82fb18adc1df20cb8fd050bc8e9a4cc7e839efbd1408",
        "predictions.csv": "2bc261917d02810df36f24a78aafac2a73b15f0b71478546b9f9c5d233a2b2af",
        "metrics.json": "2075916d5a070f3427e9cba8e92558f64b37269e70aad882c8e6f09f0c57411e"}.items()}}


@dataclass(frozen=True)
class Bundle:
    root: Path
    run: Path
    manifest: dict
    models: dict
    cohort: pd.DataFrame
    receipt: dict
    metrics: dict
    test: pd.DataFrame
    validation: pd.DataFrame
    evaluated_utc: str

    @property
    def days(self):
        return sorted(self.test["dt"].unique())


# ---------- trust checks, all before a pickle is opened ----------

def verify_files(root):
    """Exact allowlist: every file present, not reached through a link, and byte-identical to the receipts."""
    run = root / "runs" / RUN_ID
    for name, digest in EXPECTED.items():
        path = root
        for part in name.split("/"):
            path = path / part
            require(not (path.is_symlink() or path.is_junction()), f"{path} is a link; reinstall FreshStock")
        require(path.is_file(), f"{name} is missing; reinstall FreshStock")
        require(data.sha256_file(path) == digest, f"{name} was changed; reinstall FreshStock")
    extra = {p.name for p in run.iterdir()} - {Path(n).name for n in EXPECTED}
    require(not extra, f"unexpected files in the run folder: {sorted(extra)}")


def check_manifest(manifest, evaluation):
    approved = {"status": "TRAINED_FROZEN", "run_id": RUN_ID, "protocol": ex.PROTOCOL, "code_commit": CODE_COMMIT,
                "seed": ex.SEED, "thresholds": THRESHOLDS, "features": data.FEATURES,
                "cohort_digest": data.COHORT_DIGEST, "cohort_sha256": EXPECTED["data/derived/cohort.csv"],
                "source_sha256": data.SOURCE_SHA256, "splits": {k: list(v) for k, v in data.SPLITS.items()}}
    for key, value in approved.items():
        require(manifest.get(key) == value, f"manifest {key} is not the approved value")
    require(sorted(manifest["artifacts"]) == sorted(ex.ARTIFACTS)
            and sorted(evaluation["artifacts"]) == ["metrics.json", "predictions.csv"],
            "manifest artifact list is not the approved one")
    for lib in ("scikit-learn", "numpy", "scipy", "pandas", "joblib"):
        require(manifest["environment"][lib] == ex.version(lib),
                f"{lib} {manifest['environment'][lib]} is required but {ex.version(lib)} is installed")


def check_predictions(frame, cohort, split, thresholds):
    first, last, rows, positive = data.SPLITS[split]
    require(list(frame.columns) == ex.PREDICTION_COLUMNS, f"{split} predictions have other columns")
    require(len(frame) == rows, f"{split} predictions have {len(frame)} rows, expected {rows}")
    per_day = frame["dt"].value_counts()
    require(sorted(per_day.index) == day_range(first, last) and (per_day == 100).all(),
            f"{split} predictions do not cover the approved dates")
    require(not frame.duplicated(KEY + ["dt"]).any(), f"duplicate series/date in {split} predictions")
    for col in ["y_true"] + [f"{n}_flag" for n in ex.ESTIMATORS]:
        require(frame[col].isin([0, 1]).all(), f"{split} {col} is not binary")
    for name in ex.MODELS:
        score = frame[f"{name}_score"].to_numpy(dtype=float)
        require((np.isfinite(score) & (score >= 0) & (score <= 1)).all(), f"{split} {name} score is outside [0, 1]")
        require((ex.flag(score, thresholds[name]) == frame[f"{name}_flag"]).all(),
                f"{split} {name} flags do not follow the threshold")
    require((data.outcomes(cohort, frame) == frame["y_true"]).all() and int(frame["y_true"].sum()) == positive,
            f"{split} outcome labels differ from the cohort")
    yesterday = frame[KEY].assign(dt=frame["dt"].map(day_before))
    require((data.outcomes(cohort, yesterday) == frame["persistence_flag"]).all(),
            f"{split} persistence flags differ from the cohort")


def load_verified_bundle(root):
    root = Path(root).resolve()
    verify_files(root)
    run = root / "runs" / RUN_ID
    evaluation = ex.read_json(run / "evaluation.json")
    check_manifest(ex.read_json(run / "manifest.json"), evaluation)
    cohort, receipt = data.load_cohort(root / "data" / "derived")
    manifest, models = ex.load_run(run)
    metrics, test = ex.load_evaluation(run)
    validation = pd.read_csv(run / "validation_predictions.csv", float_precision="round_trip")
    for split, frame in (("test", test), ("validation", validation)):
        check_predictions(frame, cohort, split, THRESHOLDS)
    return Bundle(root, run, manifest, models, cohort, receipt, metrics, test, validation, evaluation["evaluated_utc"])


def day_before(day):
    return (dt.date.fromisoformat(day) - dt.timedelta(1)).isoformat()


def day_range(first, last):
    a, b = dt.date.fromisoformat(first), dt.date.fromisoformat(last)
    return [(a + dt.timedelta(i)).isoformat() for i in range((b - a).days + 1)]


# ---------- replay ----------

def replay_day(bundle, day, sort="logistic"):
    require(day in bundle.days, f"choose a date from {bundle.days[0]} to {bundle.days[-1]} (YYYY-MM-DD)")
    require(sort in ex.MODELS, f"sort must be one of {ex.MODELS}")
    verify_files(bundle.root)
    return demo.replay(bundle.run, bundle.cohort, day, sort)


def _rows(table, columns):
    return [{"rank": i + 1, **row} for i, row in
            enumerate({k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
                      for r in table[columns].to_dict("records"))]


FORECAST_COLUMNS = (KEY + ["dt"] + data.FEATURES + [f"{m}_score" for m in ex.MODELS]
                    + [f"{e}_flag" for e in ex.ESTIMATORS])


def forecast_view(result):
    """What may be shown before Reveal. The outcome column is not copied."""
    table = result["table"]
    return {"day": result["day"], "sort": result["sort"], "thresholds": dict(result["manifest"]["thresholds"]),
            "alerts": {n: int(table[f"{n}_flag"].sum()) for n in ex.ESTIMATORS},
            "rows": _rows(table, FORECAST_COLUMNS)}


def revealed_view(result):
    table = result["table"]
    rows = _rows(table, FORECAST_COLUMNS + ["y_true"])
    for row in rows:
        row["result"] = {n: demo.result_label(row, n) for n in ex.ESTIMATORS}
    examples = {n: {kind: next((r for r in rows if r["result"][n] == kind), None) for kind in ("False alert", "Miss")}
                for n in ex.ESTIMATORS}
    return {**forecast_view(result), "rows": rows, "total": len(rows), "positives": int(table["y_true"].sum()),
            "summary": demo.day_summary(table), "examples": examples}


def export_report(bundle, result, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"replay-{result['day']}.html"
    path.write_text(demo.render_html(result, bundle.metrics), encoding="utf-8")
    return path


class Session:
    """Replay page state: select -> forecast -> revealed. A new date always returns to select."""

    def __init__(self, replay):
        self._replay, self.state, self.result = replay, "select", None

    @property
    def can_reveal(self):
        return self.state == "forecast"

    @property
    def can_export(self):
        return self.state == "revealed"

    def change_date(self):
        self.state, self.result = "select", None

    def generate(self, day, sort):
        self.change_date()
        self.result = self._replay(day, sort)
        self.state = "forecast"

    def change_sort(self, sort):
        if self.result is not None:
            self.result = self._replay(self.result["day"], sort)

    def reveal(self):
        require(self.can_reveal, "generate a forecast first")
        self.state = "revealed"


# ---------- experiment and retrospective health ----------

def metric_rows(bundle):
    return [{"name": n, "title": ex.TITLES[n], **{k: v for k, v in bundle.metrics["estimators"][n].items()
                                                  if k not in ("daily", "first_errors")}} for n in ex.ESTIMATORS]


def quality_counts(cohort):
    values = cohort[["sale_amount"] + list(data.ID_MAX)]
    out = int((values["sale_amount"] < 0).sum())
    for col, top in data.ID_MAX.items():
        out += int(((values[col] < 0) | (values[col] > top)).sum())
    days = pd.to_datetime(cohort["dt"]).to_frame("day").join(cohort[KEY]).drop_duplicates().sort_values(KEY + ["day"])
    steps = days.groupby(KEY)["day"].diff().dropna().dt.days
    return {"rows": len(cohort), "series": len(cohort[KEY].drop_duplicates()),
            "first_day": str(cohort["dt"].min()), "last_day": str(cohort["dt"].max()),
            "missing": int(cohort[data.COLUMNS].isna().sum().sum()),
            "duplicates": int(cohort.duplicated(KEY + ["dt"]).sum()),
            "out_of_range": out, "gaps": int((steps > 1).sum())}


def _daily(frame):
    days = []
    for day, g in frame.groupby("dt"):
        y, estimators = g["y_true"].to_numpy(), {}
        for name in ex.ESTIMATORS:
            f = g[f"{name}_flag"].to_numpy()
            tp, fp = int(((y == 1) & (f == 1)).sum()), int(((y == 0) & (f == 1)).sum())
            fn, tn = int(((y == 1) & (f == 0)).sum()), int(((y == 0) & (f == 0)).sum())
            estimators[name] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "alerts": tp + fp}
        days.append({"dt": day, "rows": len(g), "positives": int(y.sum()), "estimators": estimators})
    return days


def build_health(bundle):
    """Retrospective description of saved rows only. Not live monitoring, not a drift test."""
    daily = {"validation": _daily(bundle.validation), "test": _daily(bundle.test)}
    workload = {}
    for name in ex.ESTIMATORS:
        rows = [d["estimators"][name] for d in daily["test"]]
        for key in ("tp", "fp", "fn", "tn"):
            require(sum(r[key] for r in rows) == bundle.metrics["estimators"][name][key],
                    f"daily {name} {key} counts do not reconcile with the final result")
        alerts = [r["alerts"] for r in rows]
        workload[name] = {"alerts_per_day": sum(alerts) / len(rows), "alerts_min": min(alerts),
                          "alerts_max": max(alerts), "misses_per_day": sum(r["fn"] for r in rows) / len(rows),
                          "false_alerts_per_day": sum(r["fp"] for r in rows) / len(rows)}
    prevalence = []
    for split, (first, last, rows, positive) in data.SPLITS.items():
        part = bundle.cohort[bundle.cohort["dt"].between(first, last)]
        hits = int((part["stock_hour6_22_cnt"] > 0).sum())
        require((len(part), hits) == (rows, positive), f"{split} counts do not reconcile with the protocol")
        prevalence.append({"split": split, "first": first, "last": last, "rows": rows, "positives": hits,
                           "rate": hits / rows})
    explain = {}
    for name, attribute in (("logistic", "coef_"), ("forest", "feature_importances_")):
        steps = bundle.models[name].named_steps
        values = np.ravel(getattr(steps["model"], attribute)).tolist()
        names = [n.split("__", 1)[1] for n in steps["prepare"].get_feature_names_out()]
        pairs = [{"feature": f, "value": v} for f, v in zip(names, values, strict=True)]
        numeric = [p for p in pairs if p["feature"] in data.NUMERIC]
        explain[name] = (sorted(numeric, key=lambda p: -abs(p["value"]))
                         + [p for p in pairs if p["feature"] not in data.NUMERIC])
    return {"evaluated_utc": bundle.evaluated_utc, "cutoff": data.SPLITS["test"][1],
            "quality": quality_counts(bundle.cohort), "prevalence": prevalence, "daily": daily, "workload": workload,
            "explain": explain, "digests": dict(EXPECTED), "source_rows": bundle.receipt["rows"],
            "source_series": bundle.receipt["keys"], "source_sha256": bundle.receipt["source_sha256"]}
