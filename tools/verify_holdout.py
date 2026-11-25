r"""Independent check of a finished FreshStock 2.0 evaluation from its saved rows only.

Uses pandas, NumPy and scikit-learn but none of the project code, and loads no model. Source files are read once to
confirm identities, labels and the persistence rule.
Usage: .venv\Scripts\python.exe tools/verify_holdout.py <run folder> <train.parquet> <eval.parquet> [receipt.json]
"""
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn import metrics as skm

run, train, evaluation = (Path(a) for a in sys.argv[1:4])
rows = pd.read_parquet(run / "predictions.parquet")
saved = json.loads((run / "metrics.json").read_text(encoding="utf-8"))["summary"]
freeze = json.loads((run / "freeze.json").read_text(encoding="utf-8"))
KEY, checks = ["city_id", "store_id", "product_id"], {}


def check(name, ok):
    checks[name] = bool(ok)


def close(name, mine, theirs):
    check(name, mine is None and theirs is None or None not in (mine, theirs) and abs(mine - theirs) <= 1e-10)


# identity and labels against the publisher file
columns = KEY + ["dt", "stock_hour6_22_cnt"]
actual = pq.read_table(evaluation, columns=columns).to_pandas()
joined = rows.merge(actual, on=KEY + ["dt"], how="left", validate="one_to_one")
check("every row exists in the evaluation file", not joined["stock_hour6_22_cnt"].isna().any())
check("labels equal the publisher's stockout hours", ((joined["stock_hour6_22_cnt"] > 0) == joined["y_true"]).all())
check("7 dates x all series, no duplicates", len(rows) == 7 * rows[KEY].drop_duplicates().shape[0]
      and rows["dt"].nunique() == 7 and not rows.duplicated(KEY + ["dt"]).any())
check("dates", sorted(rows["dt"].unique()) == [f"2024-06-{d}" for d in range(26, 31)] + ["2024-07-01", "2024-07-02"])
last = pq.read_table(train, columns=columns, filters=[("dt", "=", "2024-06-25")]).to_pandas()
past = pd.concat([last, actual])
past["dt"] = (pd.to_datetime(past["dt"]) + np.timedelta64(1, "D")).dt.strftime("%Y-%m-%d")
before = rows.merge(past.rename(columns={"stock_hour6_22_cnt": "h"}), on=KEY + ["dt"], how="left", validate="one_to_one")
check("persistence rule equals the previous day's recorded stockout", ((before["h"] > 0) == rows["persistence_flag"]).all())
check("past_stockout column", (rows["past_stockout"] == rows["persistence_flag"]).all())

y = rows["y_true"].to_numpy()
close("rows", saved["rows"], len(rows))
close("positives", saved["positives"], int(y.sum()))
for name in ("logistic", "forest", "persistence"):
    e, f = saved["estimators"][name], rows[f"{name}_flag"].to_numpy()
    s = rows[f"{name}_score"].to_numpy() if name != "persistence" else f.astype(float)
    if name != "persistence":
        check(f"{name} flags follow the frozen threshold", ((s >= freeze["thresholds"][name]) == f).all())
        close(f"{name} brier", e["calibration"]["brier"], skm.brier_score_loss(y, s))
        close(f"{name} log loss", e["calibration"]["log_loss"], skm.log_loss(y, s))
        cut = pd.cut(s, [i / 10 for i in range(10)] + [1.0000001], right=False, labels=False)
        for i, b in enumerate(e["calibration"]["bins"]):
            part = cut == i
            check(f"{name} bin {i}", b["n"] == int(part.sum()))
            if part.any():
                close(f"{name} bin {i} mean", b["mean_score"], float(s[part].mean()))
                close(f"{name} bin {i} rate", b["observed_rate"], float(y[part].mean()))
    tn, fp, fn, tp = (int(v) for v in pd.crosstab(y, f).reindex(index=[0, 1], columns=[0, 1], fill_value=0).to_numpy().ravel())
    check(f"{name} counts", (tn, fp, fn, tp) == (e["tn"], e["fp"], e["fn"], e["tp"]))
    close(f"{name} AP", e["average_precision"], skm.average_precision_score(y, s))
    close(f"{name} accuracy", e["accuracy"], float(Fraction(tp + tn, len(y))))
    close(f"{name} precision", e["precision"], float(Fraction(tp, tp + fp)))
    close(f"{name} recall", e["recall"], float(Fraction(tp, tp + fn)))
    close(f"{name} F1", e["f1"], float(Fraction(2 * tp, 2 * tp + fp + fn)))
    close(f"{name} balanced accuracy", e["balanced_accuracy"], float((Fraction(tp, tp + fn) + Fraction(tn, tn + fp)) / 2))
    table = rows[["dt"] + KEY + ["y_true"]].assign(score=s, flag=f)
    hits = 0
    for d, (day, g) in zip(e["daily"], table.groupby("dt", sort=True)):
        check(f"{name} {day}", (d["dt"], d["positives"], d["alerts"], d["misses"], d["false_alerts"]) == (
            day, int(g.y_true.sum()), int(g.flag.sum()), int(((g.y_true == 1) & (g.flag == 0)).sum()),
            int(((g.y_true == 0) & (g.flag == 1)).sum())))
        k = math.ceil(Fraction(len(g), 10))
        top = g.sort_values(["score"] + KEY, ascending=[False, True, True, True]).head(k)
        hits += int(top.y_true.sum())
    check(f"{name} top 10%", (e["top_k"]["k_total"], e["top_k"]["hits"]) == (7 * k, hits))
    for value, part in (("yes", rows["past_stockout"] == 1), ("no", rows["past_stockout"] == 0)):
        g = saved["slices"]["previous_day_stockout"][value]
        check(f"{name} slice {value} rows", g["n"] == int(part.sum()))
        both = 0 < y[part].sum() < part.sum()
        close(f"{name} slice {value} AP", g["estimators"][name]["average_precision"],
              skm.average_precision_score(y[part], s[part]) if both else None)
    for c in range(18):
        part, g = (rows["city_id"] == c).to_numpy(), saved["slices"]["city"][str(c)]
        both = part.any() and 0 < y[part].sum() < part.sum()
        check(f"{name} city {c}", (g["n"], g["positives"]) == (int(part.sum()), int(y[part].sum())))
        close(f"{name} city {c} AP", g["estimators"][name]["average_precision"],
              skm.average_precision_score(y[part], s[part]) if both else None)
failed = [n for n, ok in checks.items() if not ok]
receipt = {"run": run.name, "checks": len(checks), "failed": failed, "numpy": np.__version__}
print(json.dumps(receipt, indent=1))
if len(sys.argv) > 4:
    Path(sys.argv[4]).write_text(json.dumps({**receipt, "detail": checks}, indent=1), encoding="utf-8")
sys.exit(1 if failed else 0)
