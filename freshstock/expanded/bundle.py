"""Portable read-only bundle: pinned files, past-only replay of the holdout days, reveal and export. No PyArrow."""
import csv
import datetime as dt
import hashlib
import html
import io
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np

from ..data import KEY, parse_day, require, sha256_file
from . import data, experiment, metrics
from .metrics import ESTIMATORS, MODELS

FILES = ["history.npz", "scores.npz", "freeze.json", "metrics.json", "candidate_metrics.json", "protocol.json",
         "evaluation.json", "logistic.joblib", "forest.joblib", "NOTICE.txt"]
FROM_RUN = ["freeze.json", "metrics.json", "candidate_metrics.json", "protocol.json", "evaluation.json",
            "logistic.joblib", "forest.joblib"]
HISTORY_DAYS = 35
PARITY = 1e-10
TITLES = {"logistic": "Logistic Regression", "forest": "Random Forest", "persistence": "Persistence rule"}
RESULTS = {(1, 1): "Correct alert", (1, 0): "Miss", (0, 1): "False alert", (0, 0): "Correct clear"}
NOTICE = """Data: FreshRetailNet-50K, Dingdong-Inc and Wang et al. (2025), arXiv:2505.16319, CC BY 4.0
https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K (revision %s)
https://creativecommons.org/licenses/by/4.0/legalcode
Changes: six of the published columns, 35 days (2024-05-29 to 2024-07-02) and the selected series only, stored as arrays;
model scores, alert flags and outcome flags were added. No endorsement by the publisher is implied.
Historical research data. It is not live inventory.
""" % data.REVISION


@dataclass(frozen=True)
class Bundle:
    root: Path
    pin: dict
    freeze: dict
    metrics: dict
    candidates: dict
    protocol: dict
    evaluation: dict
    history: data.Panel
    predictions: dict
    models: dict

    @property
    def days(self):
        return [dt.date.fromordinal(int(d)).isoformat() for d in self.predictions["dates"]]

    @property
    def n(self):
        return len(self.history.keys)


# ---------- build (research machine only) ----------

def build(run, freeze_sha256, folder, train=data.TRAIN, holdout=data.HOLDOUT):
    """Copy the verified run into a new runtime folder and return its pin. Existing output is never replaced."""
    run, folder = Path(run), Path(folder)
    require(not folder.exists(), f"{folder.name} exists; choose a new output folder")
    experiment.verify(run, freeze_sha256)
    n = experiment.read_json(run / "freeze.json")["n_series"]
    whole = experiment._cohort(data.read_panel(train), n)[0].join(experiment._cohort(data.read_panel(holdout), n)[0])
    last = data.SPLITS["holdout"][1]
    history = whole.since(last - dt.timedelta(HISTORY_DAYS - 1))
    saved = experiment.load_predictions(run / "predictions.parquet")
    require(np.array_equal(saved["keys"], history.keys), "predictions and history contain different series")
    stage = folder.with_name(folder.name + ".partial")
    stage.mkdir(parents=True)
    np.savez_compressed(stage / "history.npz", allow_pickle=False, keys=history.keys, dates=history.dates,
                        sales=history.sales, stock=history.stock)
    np.savez_compressed(stage / "scores.npz", allow_pickle=False, dates=saved["dates"], y_true=saved["y_true"],
                        past_stockout=saved["past_stockout"],
                        **{f"{m}_score": saved["scores"][m] for m in MODELS},
                        **{f"{e}_flag": saved["flags"][e] for e in ESTIMATORS})
    for name in FROM_RUN:
        shutil.copyfile(run / name, stage / name)
    (stage / "NOTICE.txt").write_text(NOTICE, encoding="utf-8", newline="\n")
    pin = {"run_id": run.name, "freeze_sha256": freeze_sha256, "n_series": n,
           "files": {name: {"sha256": sha256_file(stage / name), "size": (stage / name).stat().st_size}
                     for name in FILES}}
    load(stage, pin)
    stage.rename(folder)
    return pin


# ---------- trusted load ----------

def _verified(root, pin):
    """Exact allowlist by name, size and SHA-256, read once into memory. Nothing is parsed before this passes."""
    root = Path(root)
    require(not (root.is_symlink() or root.is_junction()), f"{root} is a link; reinstall FreshStock")
    require(root.is_dir(), f"{root} is missing; reinstall FreshStock")
    require(set(pin["files"]) == set(FILES), "the program pins other files; reinstall FreshStock")
    content = {}
    for name in FILES:
        path, entry = root / name, pin["files"][name]
        require(not (path.is_symlink() or path.is_junction()), f"{name} is a link; reinstall FreshStock")
        require(path.is_file(), f"{name} is missing; reinstall FreshStock")
        require(path.stat().st_size == entry["size"], f"{name} was changed; reinstall FreshStock")
        content[name] = path.read_bytes()
        require(hashlib.sha256(content[name]).hexdigest() == entry["sha256"],
                f"{name} was changed; reinstall FreshStock")
    extra = {p.name for p in root.iterdir()} - set(FILES)
    require(not extra, f"unexpected items in the bundle: {sorted(extra)}")
    return content


def _arrays(content, name, spec):
    """Numeric arrays with exact names, types and shapes; sizes are checked before anything is unpacked."""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            for item in archive.infolist():
                dtype, shape = spec.get(item.filename.removesuffix(".npy"), (np.uint8, (0,)))
                require(item.file_size <= np.dtype(dtype).itemsize * int(np.prod(shape)) + 1024,
                        f"{name} is larger than its contract")
        with np.load(io.BytesIO(content), allow_pickle=False) as stored:
            require(sorted(stored.files) == sorted(spec), f"{name} has other arrays")
            arrays = {k: stored[k] for k in spec}
    except (zipfile.BadZipFile, EOFError, OSError) as error:
        require(False, f"{name} cannot be read: {error}")
    for key, (dtype, shape) in spec.items():
        a = arrays[key]
        require(a.dtype == dtype and a.shape == shape and a.flags.c_contiguous and not a.dtype.hasobject,
                f"{name}: {key} has another type or shape")
        a.flags.writeable = False
    return arrays


def load(root, pin):
    content = _verified(root, pin)
    texts = {name: json.loads(content[name]) for name in FILES if name.endswith(".json")}
    freeze, n, d = texts["freeze.json"], pin["n_series"], len(data.split_days("holdout"))
    require(pin["files"]["freeze.json"]["sha256"] == pin["freeze_sha256"] and freeze["run_id"] == pin["run_id"]
            and freeze["status"] == "FROZEN" and freeze["protocol"] == experiment.NAME and freeze["n_series"] == n,
            "freeze.json is not the approved run")
    for name in ("logistic.joblib", "forest.joblib", "candidate_metrics.json", "protocol.json"):
        require(freeze["artifacts"][name] == pin["files"][name]["sha256"], f"{name} is not the frozen file")
    evaluation = texts["evaluation.json"]
    require(evaluation["status"] == "EVALUATED" and evaluation["freeze_sha256"] == pin["freeze_sha256"]
            and evaluation["artifacts"]["metrics.json"] == pin["files"]["metrics.json"]["sha256"],
            "evaluation.json belongs to another run")
    for lib in ("scikit-learn", "numpy", "scipy", "joblib"):
        require(freeze["environment"][lib] == version(lib),
                f"{lib} {freeze['environment'][lib]} is required but {version(lib)} is installed")
    h = _arrays(content["history.npz"], "history.npz", {
        "keys": (np.int64, (n, 3)), "dates": (np.int32, (HISTORY_DAYS,)), "sales": (np.float64, (n, HISTORY_DAYS)),
        "stock": (np.uint8, (n, HISTORY_DAYS))})
    require(int(h["dates"][-1]) == data.SPLITS["holdout"][1].toordinal(), "history ends on another date")
    require(bool((np.isfinite(h["sales"]) & (h["sales"] >= 0)).all()) and bool((h["stock"] <= 16).all())
            and bool((h["keys"] >= 0).all()), "history values are out of range")
    history = data.Panel(h["keys"], h["dates"], h["sales"], h["stock"])
    p = _arrays(content["scores.npz"], "scores.npz", {
        "dates": (np.int32, (d,)), "y_true": (np.uint8, (d, n)), "past_stockout": (np.uint8, (d, n)),
        **{f"{m}_score": (np.float64, (d, n)) for m in MODELS}, **{f"{e}_flag": (np.uint8, (d, n)) for e in ESTIMATORS}})
    require(np.array_equal(p["dates"], history.dates[-d:]), "scores cover other dates")
    out = (history.stock.T > 0).astype(np.uint8)
    require(np.array_equal(p["y_true"], out[-d:]), "saved outcomes differ from the history")
    require(np.array_equal(p["past_stockout"], out[-d - 1:-1]) and np.array_equal(p["persistence_flag"], out[-d - 1:-1]),
            "previous-day values differ from the history")
    for m in MODELS:
        s = p[f"{m}_score"]
        require(bool((np.isfinite(s) & (s >= 0) & (s <= 1)).all()), f"{m} scores are out of range")
        require(np.array_equal(p[f"{m}_flag"], (s >= freeze["thresholds"][m]).astype(np.uint8)),
                f"{m} flags do not follow the threshold")
    predictions = {"keys": history.keys, "dates": p["dates"], "y_true": p["y_true"], "past_stockout": p["past_stockout"],
                   "scores": {m: p[f"{m}_score"] for m in MODELS}, "flags": {e: p[f"{e}_flag"] for e in ESTIMATORS}}
    models = {m: joblib.load(io.BytesIO(content[f"{m}.joblib"])) for m in MODELS}
    return Bundle(Path(root), pin, freeze, texts["metrics.json"], texts["candidate_metrics.json"],
                  texts["protocol.json"], evaluation, history, predictions, models)


# ---------- replay ----------

@dataclass(frozen=True)
class Forecast:
    """Everything known on the evening before `day`."""
    day: str
    scores: dict
    flags: dict
    features: np.ndarray
    _orders: dict = field(default_factory=dict)

    def order(self, model):
        if model not in self._orders:
            self._orders[model] = metrics.ranking(self.scores[model] if model in MODELS
                                                  else self.flags[model].astype(float))
        return self._orders[model]


@dataclass(frozen=True)
class Revealed:
    day: str
    actual: np.ndarray
    counts: dict
    review: dict

    @property
    def positives(self):
        return int(self.actual.sum())

    @property
    def total(self):
        return len(self.actual)


def forecast(bundle, day):
    """Score one holdout day from the 28 days before it and require equality with the saved scores."""
    require(day in bundle.days, f"choose a date from {bundle.days[0]} to {bundle.days[-1]} (YYYY-MM-DD)")
    target = parse_day(day)
    past = bundle.history.until(target - dt.timedelta(1))
    live = experiment.score_day(bundle.models, bundle.freeze, past, target)
    i = bundle.days.index(day)
    for m in MODELS:
        require(float(np.abs(live[f"{m}_score"] - bundle.predictions["scores"][m][i]).max()) <= PARITY,
                f"{m} scores differ from the saved scores; the forecast was stopped")
    for e in ESTIMATORS:
        require(np.array_equal(live[f"{e}_flag"], bundle.predictions["flags"][e][i]),
                f"{e} flags differ from the saved flags; the forecast was stopped")
    return Forecast(day, {m: live[f"{m}_score"] for m in MODELS}, {e: live[f"{e}_flag"] for e in ESTIMATORS},
                    data.features_on(past, target))


def reveal(bundle, f):
    actual = bundle.predictions["y_true"][bundle.days.index(f.day)]
    counts, review = {}, {}
    for e in ESTIMATORS:
        m = metrics.flag_metrics(actual, f.flags[e])
        counts[e] = {k: m[k] for k in ("tp", "fn", "fp", "tn")}
        score = f.scores[e] if e in MODELS else f.flags[e].astype(float)
        top = metrics.top_k(actual[None, :], score[None, :])["daily"][0]
        review[e] = {k: top[k] for k in ("k", "hits", "false_alerts", "misses", "recall")}
    return Revealed(f.day, actual, counts, review)


def page_count(n, size):
    return (n + size - 1) // size


def _row(bundle, f, model, position, revealed):
    at = int(f.order(model)[position])
    row = {"rank": position + 1, "dt": f.day, **dict(zip(KEY, bundle.history.keys[at].tolist())),
           **{f"{m}_score": float(f.scores[m][at]) for m in MODELS},
           **{f"{e}_flag": int(f.flags[e][at]) for e in ESTIMATORS},
           "features": dict(zip(data.FEATURES["B"], f.features[at].tolist()))}
    if revealed is not None:
        y = int(revealed.actual[at])
        row.update(y_true=y, result={e: RESULTS[(y, row[f"{e}_flag"])] for e in ESTIMATORS})
    return row


def page_rows(bundle, f, model, page, size, revealed=None):
    """One page of the global ranking; ranks and totals never depend on the page."""
    require(0 <= page < page_count(bundle.n, size), "page is outside the ranking")
    return [_row(bundle, f, model, i, revealed) for i in range(page * size, min((page + 1) * size, bundle.n))]


def find_rank(bundle, f, model, key):
    at = np.flatnonzero((bundle.history.keys == np.array(key)).all(axis=1))
    return int(np.flatnonzero(f.order(model) == at[0])[0]) + 1 if len(at) else None


# ---------- export ----------

def export(bundle, f, revealed, model, folder):
    """Full CSV and a short HTML summary, published together as one new folder or not at all."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".partial-", dir=folder))
    try:
        with open(stage / "ranking.csv", "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["rank", "dt", *KEY, "logistic_score", "forest_score", "logistic_flag", "forest_flag",
                             "persistence_flag", "y_true"])
            for position, at in enumerate(f.order(model).tolist()):
                writer.writerow([position + 1, f.day, *bundle.history.keys[at].tolist(),
                                 *(repr(float(f.scores[m][at])) for m in MODELS),
                                 *(int(f.flags[e][at]) for e in ESTIMATORS), int(revealed.actual[at])])
        (stage / "summary.html").write_text(_html(bundle, f, revealed, model), encoding="utf-8", newline="\n")
        experiment.write_json(stage / "manifest.json", {
            "day": f.day, "run_id": bundle.freeze["run_id"], "ranked_by": model, "rows": bundle.n,
            "files": {name: sha256_file(stage / name) for name in ("ranking.csv", "summary.html")}})
        base = f"replay-{f.day}-{model}"
        for number in range(1, 1000):
            final = folder / (base if number == 1 else f"{base}-{number}")
            if not final.exists():
                os.rename(stage, final)
                return final
        raise OSError("too many reports with the same name")
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _html(bundle, f, revealed, model):
    e = html.escape
    head = "".join(f"<th>{t}</th>" for t in ("Method", "Alerts", "Correct alerts", "Misses", "False alerts",
                                             "Correct clears", "Stockouts in review list"))
    body = "".join(
        f"<tr><td>{e(TITLES[n])}</td><td>{c['tp'] + c['fp']:,}</td><td>{c['tp']:,}</td><td>{c['fn']:,}</td>"
        f"<td>{c['fp']:,}</td><td>{c['tn']:,}</td><td>{revealed.review[n]['hits']:,} of {revealed.review[n]['k']:,}</td></tr>"
        for n, c in revealed.counts.items())
    top = "".join(
        f"<tr><td>{r['rank']}</td><td>{r['city_id']}</td><td>{r['store_id']}</td><td>{r['product_id']}</td>"
        f"<td>{r['logistic_score']:.4f}</td><td>{r['forest_score']:.4f}</td>"
        f"<td>{'stockout' if r['y_true'] else 'none'}</td></tr>"
        for r in page_rows(bundle, f, model, 0, 20, revealed))
    limits = "".join(f"<li>{e(t)}</li>" for t in bundle.freeze["limitations"])
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>FreshStock replay {f.day}</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#172C26;max-width:900px}}table{{border-collapse:collapse;
margin:12px 0}}td,th{{border:1px solid #D7E2DA;padding:5px 9px;text-align:right}}td:first-child,th:first-child{{text-align:left}}
</style><h1>FreshStock historical replay, {f.day}</h1>
<p>{revealed.total:,} series ranked by {e(TITLES[model])}; {revealed.positives:,} had a recorded stockout between 06:00 and 22:00.
Scores used data up to the day before. This is a historical replay, not live inventory.</p>
<p>Run {e(bundle.freeze['run_id'])}. Alert from {bundle.freeze['thresholds']['logistic']} (Logistic Regression) and
{bundle.freeze['thresholds']['forest']} (Random Forest). The full list of all {revealed.total:,} rows is in ranking.csv.</p>
<table><tr>{head}</tr>{body}</table><h2>First 20 of {bundle.n} series</h2>
<table><tr><th>Rank</th><th>City</th><th>Store</th><th>Product</th><th>LR score</th><th>RF score</th><th>Recorded</th></tr>
{top}</table><h2>Limits</h2><ul>{limits}</ul>
<p>Data: FreshRetailNet-50K, Dingdong-Inc and Wang et al. (2025), CC BY 4.0.</p></html>
"""


# ---------- retrospective page ----------

def explain(bundle):
    """Whole-model coefficients and impurity importances of the fitted base pipelines. Descriptive only."""
    out = {}
    for m, attribute in (("logistic", "coef_"), ("forest", "feature_importances_")):
        step = experiment.base_pipeline(bundle.models[m]).named_steps["model"]
        names = data.FEATURES[bundle.freeze["selected"][m]["feature_set"]]
        pairs = [{"feature": f, "value": v} for f, v in zip(names, np.ravel(getattr(step, attribute)).tolist(),
                                                           strict=True)]
        numeric = [p for p in pairs if not p["feature"].startswith("weekday_")]
        out[m] = sorted(numeric, key=lambda p: -abs(p["value"])) + [p for p in pairs if p not in numeric]
    return out


def prevalence(bundle):
    f, s = bundle.freeze, bundle.metrics["summary"]
    parts = [(name, f["rows"][name], f["positives"][name]) for name in ("initial_fit", "tuning", "calibration",
                                                                         "threshold")]
    parts.append(("holdout", s["rows"], s["positives"]))
    return [{"split": name, "first": data.SPLITS[name][0].isoformat(), "last": data.SPLITS[name][1].isoformat(),
             "rows": rows, "positives": positives, "rate": positives / rows} for name, rows, positives in parts]


def health(bundle):
    """Description of saved rows only. Not live monitoring and not a drift test."""
    s = bundle.metrics["summary"]
    daily = []
    for i, day in enumerate(bundle.days):
        rows = {e: s["estimators"][e]["daily"][i] for e in ESTIMATORS}
        daily.append({"dt": day, "positives": rows["logistic"]["positives"], "estimators": {
            e: {"alerts": r["alerts"], "fn": r["misses"], "fp": r["false_alerts"]} for e, r in rows.items()}})
    workload = {}
    for e in ESTIMATORS:
        rows = [d["estimators"][e] for d in daily]
        workload[e] = {"alerts_per_day": sum(r["alerts"] for r in rows) / len(rows),
                       "alerts_min": min(r["alerts"] for r in rows), "alerts_max": max(r["alerts"] for r in rows),
                       "misses_per_day": sum(r["fn"] for r in rows) / len(rows),
                       "false_alerts_per_day": sum(r["fp"] for r in rows) / len(rows)}
    return {"evaluated_utc": bundle.evaluation["evaluated_utc"], "series": bundle.n,
            "prevalence": prevalence(bundle), "daily": daily,
            "workload": workload, "explain": explain(bundle),
            "digests": {name: entry["sha256"] for name, entry in bundle.pin["files"].items()}}
