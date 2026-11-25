"""Evaluation numbers computed from counts, with undefined values reported as null."""
import datetime as dt

import numpy as np

from ..data import require

MODELS = ["logistic", "forest"]
ESTIMATORS = MODELS + ["persistence"]
TOLERANCE = 1e-10
N_CITIES = 18


def _ratio(a, b):
    return float(a / b) if b else None


def average_precision(y, score):
    """Sum of (recall step) x precision over distinct descending scores; None without both classes."""
    y, score = np.asarray(y, dtype=np.int64), np.asarray(score, dtype=float)
    total = int(y.sum())
    if total == 0 or total == len(y):
        return None
    order = np.argsort(-score, kind="stable")
    y, score = y[order], score[order]
    last = np.append(np.flatnonzero(np.diff(score)), len(y) - 1)  # last row of each tied group
    hits = np.cumsum(y)[last]
    return float((np.diff(hits, prepend=0) / total * (hits / (last + 1))).sum())


def flag_metrics(y, flags):
    y, flags = np.asarray(y), np.asarray(flags)
    tp, fp = int(((y == 1) & (flags == 1)).sum()), int(((y == 0) & (flags == 1)).sum())
    fn, tn = int(((y == 1) & (flags == 0)).sum()), int(((y == 0) & (flags == 0)).sum())
    n, recall, specificity = len(y), _ratio(tp, tp + fn), _ratio(tn, tn + fp)
    return {"n": n, "positives": tp + fn, "prevalence": _ratio(tp + fn, n), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "accuracy": _ratio(tp + tn, n), "precision": _ratio(tp, tp + fp), "recall": recall,
            "f1": _ratio(2 * tp, 2 * tp + fp + fn),
            "balanced_accuracy": None if recall is None or specificity is None else (recall + specificity) / 2,
            "alerts": tp + fp, "alert_rate": _ratio(tp + fp, n)}


# ---------- fixed review workload ----------

def k_of(n):
    """ceil(0.10 * n) in integer arithmetic."""
    return (n + 9) // 10


def ranking(score):
    """Row order by descending score; rows are stored by ascending key, so ties keep key order."""
    return np.argsort(-np.asarray(score), kind="stable")


def top_k(y, score, k=None):
    """y and score are [days, series]; the same k series are reviewed every day."""
    k = k_of(y.shape[1]) if k is None else k
    daily = []
    for day_y, day_score in zip(y, score):
        hits, positives = int(day_y[ranking(day_score)[:k]].sum()), int(day_y.sum())
        daily.append({"k": k, "hits": hits, "false_alerts": k - hits, "misses": positives - hits,
                      "coverage": k / len(day_y), "precision": hits / k, "recall": _ratio(hits, positives)})
    total = {c: sum(d[c] for d in daily) for c in ("hits", "false_alerts", "misses")}
    return {"k_per_day": k, "k_total": k * len(daily), **total, "precision": _ratio(total["hits"], k * len(daily)),
            "recall": _ratio(total["hits"], total["hits"] + total["misses"]), "daily": daily}


# ---------- calibration ----------

def brier(y, p):
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y)) ** 2))


def log_loss(y, p):
    eps = np.finfo(float).eps
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return float(-np.mean(np.where(np.asarray(y) == 1, np.log(p), np.log1p(-p))))


def reliability(y, p):
    """Ten equal-width bins [0, .1), ..., [.9, 1]."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    at = np.searchsorted([i / 10 for i in range(1, 10)], p, side="right")
    return [{"low": i / 10, "high": (i + 1) / 10, "n": int((at == i).sum()),
             "mean_score": float(p[at == i].mean()) if (at == i).any() else None,
             "observed_rate": float(y[at == i].mean()) if (at == i).any() else None} for i in range(10)]


# ---------- full summary of saved predictions ----------

def _check(p):
    n, d = len(p["keys"]), len(p["dates"])
    codes = p["keys"] @ np.array([1_000_000, 1_000, 1])
    require(n > 0 and bool((np.diff(codes) > 0).all()), "prediction keys must be unique and ascending")
    require(d > 0 and bool((np.diff(p["dates"]) == 1).all()), "prediction dates must be consecutive")
    binary = [p["y_true"], p["past_stockout"], *(p["flags"][e] for e in ESTIMATORS)]
    require(all(a.shape == (d, n) for a in [*binary, *(p["scores"][m] for m in MODELS)]), "prediction shapes differ")
    require(all(bool(np.isin(a, (0, 1)).all()) for a in binary), "flags and outcomes must be 0 or 1")
    require(all(bool((np.isfinite(p["scores"][m]) & (p["scores"][m] >= 0) & (p["scores"][m] <= 1)).all())
                for m in MODELS), "scores must be finite and in 0..1")


def scores_of(p, name):
    return p["scores"][name] if name in MODELS else p["flags"][name].astype(float)


def _slice(p, rows):
    y = p["y_true"][rows]
    return {"n": len(y), "positives": int(y.sum()), "estimators": {
        e: {"average_precision": average_precision(y, scores_of(p, e)[rows]), **flag_metrics(y, p["flags"][e][rows])}
        for e in ESTIMATORS}}


def summarize(p):
    """All reported numbers, from saved identities, scores, flags and outcomes only."""
    _check(p)
    y, estimators = p["y_true"], {}
    for name in ESTIMATORS:
        score, flags = scores_of(p, name), p["flags"][name]
        entry = {"average_precision": average_precision(y.ravel(), score.ravel()),
                 **flag_metrics(y.ravel(), flags.ravel()), "top_k": top_k(y, score)}
        entry["daily"] = [{"dt": dt.date.fromordinal(int(day)).isoformat(), "positives": int(a.sum()),
                           "alerts": int(f.sum()), "hits": int((a & f).sum()), "misses": int((a & (1 - f)).sum()),
                           "false_alerts": int(((1 - a) & f).sum())}
                          for day, a, f in zip(p["dates"], y.astype(np.int64), flags.astype(np.int64))]
        if name in MODELS:
            entry["calibration"] = {"brier": brier(y.ravel(), score.ravel()),
                                    "log_loss": log_loss(y.ravel(), score.ravel()),
                                    "bins": reliability(y.ravel(), score.ravel())}
        estimators[name] = entry
    city = np.broadcast_to(p["keys"][:, 0], y.shape)
    return {"rows": int(y.size), "days": len(p["dates"]), "series": len(p["keys"]), "positives": int(y.sum()),
            "prevalence": float(y.mean()), "estimators": estimators,
            "slices": {"previous_day_stockout": {"yes": _slice(p, p["past_stockout"] == 1),
                                                 "no": _slice(p, p["past_stockout"] == 0)},
                       "city": {str(c): _slice(p, city == c) for c in range(N_CITIES)}}}


def check_against_sklearn(p, summary):
    from sklearn import metrics as skm

    y = p["y_true"].ravel()
    for name in ESTIMATORS:
        ours, flags, score = summary["estimators"][name], p["flags"][name].ravel(), scores_of(p, name).ravel()
        pairs = [("average_precision", skm.average_precision_score(y, score)), ("accuracy", skm.accuracy_score(y, flags)),
                 ("precision", skm.precision_score(y, flags, zero_division=0)), ("recall", skm.recall_score(y, flags)),
                 ("f1", skm.f1_score(y, flags, zero_division=0)),
                 ("balanced_accuracy", skm.balanced_accuracy_score(y, flags))]
        if name in MODELS:
            pairs += [("brier", skm.brier_score_loss(y, score)), ("log_loss", skm.log_loss(y, score))]
        for key, theirs in pairs:
            mine = ours["calibration"][key] if key in ("brier", "log_loss") else ours[key] or 0.0
            require(abs(mine - theirs) <= TOLERANCE, f"{name}: {key} differs from scikit-learn")
        require(skm.confusion_matrix(y, flags, labels=[0, 1]).tolist() ==
                [[ours["tn"], ours["fp"]], [ours["fn"], ours["tp"]]], f"{name}: confusion differs from scikit-learn")
