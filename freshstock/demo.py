"""Offline replay of one held-out day: forecast from history first, observed outcome second."""
from html import escape

import numpy as np

from . import data, experiment as ex
from .data import KEY, require

RESULT = {(1, 1): "Correct alert", (0, 1): "False alert", (1, 0): "Miss", (0, 0): "Correct clear"}
SHORT = {"logistic": "Logistic", "forest": "Forest", "persistence": "Persistence"}


def forecast(models, thresholds, history, day):
    """Score every series for `day`. Takes only rows dated before `day`; no outcome is available here."""
    x = data.features_at(history, day)
    scores, flags, _ = ex.score_all(models, thresholds, x)
    for name in ex.MODELS:
        x[f"{name}_score"] = scores[name]
    for name in ex.ESTIMATORS:
        x[f"{name}_flag"] = flags[name]
    return x


def replay(run, cohort, day, sort="logistic"):
    manifest, models = ex.load_run(run)
    _, frozen = ex.load_evaluation(run)
    days = sorted(frozen["dt"].unique())
    require(day in days, f"choose a date from {days[0]} to {days[-1]} (YYYY-MM-DD)")
    table = forecast(models, manifest["thresholds"], cohort[cohort["dt"] < day], day)
    table["y_true"] = data.outcomes(cohort, table)

    check = table.merge(frozen[frozen["dt"] == day], on=KEY + ["dt"], suffixes=("", "_frozen"), validate="one_to_one")
    require(len(check) == len(table) == (frozen["dt"] == day).sum(), "replay rows differ from the frozen predictions")
    for col in [f"{m}_score" for m in ex.MODELS] + [f"{e}_flag" for e in ex.ESTIMATORS] + ["y_true"]:
        require(np.abs(check[col] - check[f"{col}_frozen"]).max() <= 1e-10,
                f"live {col} differs from the frozen evaluation")
    table = table.sort_values([f"{sort}_score"] + KEY, ascending=[False, True, True, True]).reset_index(drop=True)
    return {"day": day, "days": days, "sort": sort, "table": table, "manifest": manifest, "run": run.name}


def result_label(row, name):
    return RESULT[int(row["y_true"]), int(row[f"{name}_flag"])]


def day_summary(table):
    out = {}
    for name in ex.ESTIMATORS:
        labels = [result_label(r, name) for _, r in table.iterrows()]
        out[name] = {k: labels.count(k) for k in RESULT.values()}
    return out


# ---------- console ----------

def forecast_text(r):
    t = r["manifest"]["thresholds"]
    lines = [f"FORECAST for {r['day']} (made from data up to the day before; run {r['run']})",
             f"Alert when score >= threshold: Logistic {t['logistic']}, Forest {t['forest']}. "
             "Scores are rankings, not percent certainty.",
             f"Sorted by {SHORT[r['sort']]} score. Please review flagged series manually.", "",
             f"{'city':>4} {'store':>5} {'product':>7} | {'logistic':>8} {'':5} | {'forest':>8} {'':5} | persistence"]
    for _, row in r["table"].iterrows():
        mark = {n: "ALERT" if row[f"{n}_flag"] else "" for n in ex.ESTIMATORS}
        lines.append(f"{row.city_id:>4} {row.store_id:>5} {row.product_id:>7} | {row.logistic_score:>8.4f} "
                     f"{mark['logistic']:5} | {row.forest_score:>8.4f} {mark['forest']:5} | {mark['persistence']}")
    return "\n".join(lines)


def outcome_text(r):
    table = r["table"]
    lines = [f"OBSERVED on {r['day']}: {int(table.y_true.sum())} of {len(table)} series recorded a stockout hour", "",
             f"{'':12}" + "".join(f"{k:>15}" for k in RESULT.values())]
    for name, counts in day_summary(table).items():
        lines.append(f"{SHORT[name]:12}" + "".join(f"{v:>15}" for v in counts.values()))
    for name in ex.MODELS:
        for kind in ("False alert", "Miss"):
            hit = next((row for _, row in table.iterrows() if result_label(row, name) == kind), None)
            text = "none" if hit is None else (f"city {hit.city_id} store {hit.store_id} product {hit.product_id} "
                                               f"(score {hit[f'{name}_score']:.4f})")
            lines.append(f"Example {kind.lower()} - {SHORT[name]}: {text}")
    return "\n".join(lines)


def metrics_text(m):
    lines = [f"FINAL {m['split'].upper()} RESULTS ({m['rows']} rows, {m['positives']} with a stockout, "
             f"prevalence {m['prevalence']:.4f})",
             f"{'':12}{'AP':>8}{'F1':>8}{'precision':>11}{'recall':>8}{'accuracy':>10}{'bal.acc':>9}"
             f"{'TN':>6}{'FP':>6}{'FN':>6}{'TP':>6}"]
    for name, e in m["estimators"].items():
        lines.append(f"{SHORT[name]:12}{e['average_precision']:>8.4f}{e['f1']:>8.4f}{e['precision']:>11.4f}"
                     f"{e['recall']:>8.4f}{e['accuracy']:>10.4f}{e['balanced_accuracy']:>9.4f}"
                     f"{e['tn']:>6}{e['fp']:>6}{e['fn']:>6}{e['tp']:>6}")
    return "\n".join(lines)


# ---------- offline HTML report ----------

CSS = """
:root{color-scheme:light;--ink:#1c2b2a;--soft:#5b6b69;--line:#d9e0de;--paper:#fff;--wash:#f4f7f6;--deep:#0f5c56;
--alert:#9a3412;--alert-bg:#ffedd5}
*{box-sizing:border-box}body{margin:0;background:var(--wash);color:var(--ink);font:16px/1.55 "Segoe UI",system-ui,sans-serif}
main{max-width:1040px;margin:0 auto;padding:32px 20px 64px}h1{font-size:30px;line-height:1.2;margin:0 0 4px}
h2{font-size:21px;margin:0 0 6px}p{margin:0 0 10px;max-width:72ch}.soft{color:var(--soft)}
section,details{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:22px 24px;margin-top:20px}
summary{cursor:pointer;font-size:21px;font-weight:600;color:var(--deep)}summary:focus-visible{outline:3px solid #f59e0b;outline-offset:4px}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:15px}
th{font-weight:600;border-bottom:2px solid var(--ink)}th,td{padding:7px 12px;text-align:right;white-space:nowrap}
td{border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}
#metrics th,#metrics td{padding:7px 8px;font-size:14px}
.alert{background:var(--alert-bg);color:var(--alert);font-weight:700;border-radius:4px;padding:1px 7px;font-size:13px}
.off{visibility:hidden}.bad{color:var(--alert);font-weight:600}.ok{color:var(--soft)}ul{margin:0;padding-left:20px;max-width:76ch}
@media print{body{background:#fff}section,details{break-inside:avoid-page}}
"""


def _series(row):
    return f"City {row.city_id} / Store {row.store_id} / Product {row.product_id}"


def _table(head, rows):
    head = "".join(f"<th scope='col'>{escape(h)}</th>" for h in head)
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def render_html(r, m):
    t, table = r["manifest"]["thresholds"], r["table"]
    alert = {n: int(table[f"{n}_flag"].sum()) for n in ex.ESTIMATORS}
    mark = {0: " <span class='alert off' aria-hidden='true'>ALERT</span>", 1: " <span class='alert'>ALERT</span>"}
    forecast_rows, outcome_rows = [], []
    for i, row in table.iterrows():
        cells = "".join(f"<td>{row[f'{n}_score']:.4f}{mark[int(row[f'{n}_flag'])]}</td>" for n in ex.MODELS)
        forecast_rows.append(f'<tr class="series"><td>{i + 1}. {escape(_series(row))}</td>{cells}'
                             f"<td>{mark[int(row.persistence_flag)]}</td></tr>")
        labels = [result_label(row, n) for n in ex.ESTIMATORS]
        cells = "".join(f"<td class='{'ok' if x.startswith('Correct') else 'bad'}'>{x}</td>" for x in labels)
        outcome_rows.append(f'<tr class="series"><td>{i + 1}. {escape(_series(row))}</td>'
                            f"<td>{'Stockout' if row.y_true else 'No stockout'}</td>{cells}</tr>")
    summary_rows = [f"<tr><td>{escape(ex.TITLES[n])}</td>" + "".join(f"<td>{v}</td>" for v in c.values()) + "</tr>"
                    for n, c in day_summary(table).items()]
    metric_rows = []
    for n, e in m["estimators"].items():
        ratios = "".join(f"<td>{e[k]:.4f}</td>" for k in ("average_precision", "f1", "precision", "recall", "accuracy",
                                                          "balanced_accuracy"))
        counts = "".join(f"<td>{e[k]}</td>" for k in ("tn", "fp", "fn", "tp"))
        metric_rows.append(f"<tr><td>{escape(ex.TITLES[n])}</td>{ratios}{counts}</tr>")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FreshStock replay {escape(r['day'])}</title><style>{CSS}</style></head><body><main>
<h1>FreshStock: stockout review list for {escape(r['day'])}</h1>
<p class="soft">Historical replay of a held-out day. Run {escape(r['run'])}, code {escape(r['manifest']['code_commit'][:12])}.</p>
<section id="forecast"><h2>Step 1. Forecast from the day before</h2>
<p>Each score uses only the series' own data up to the previous day. Higher means higher estimated risk of at least one
recorded stockout hour between 06:00 and 22:00. Scores rank series; they are not percent certainty. A planner reviews
the list and decides.</p>
<p>Alerts: Logistic Regression {alert['logistic']} (score at least {t['logistic']}), Random Forest {alert['forest']}
(score at least {t['forest']}), persistence rule {alert['persistence']} (stocked out yesterday).
Sorted by {escape(ex.TITLES[r['sort']])} score.</p>
{_table(['Series (encoded IDs)', 'Logistic Regression', 'Random Forest', 'Persistence'], forecast_rows)}</section>
<details id="outcome"><summary>Step 2. Reveal what was observed on {escape(r['day'])}</summary>
<p>{int(table.y_true.sum())} of {len(table)} series recorded a stockout hour that day.</p>
{_table(['Method', *RESULT.values()], summary_rows)}<br>
{_table(['Series (encoded IDs)', 'Observed', 'Logistic Regression', 'Random Forest', 'Persistence'], outcome_rows)}</details>
<section id="metrics"><h2>Results on the whole {escape(m['split'])} period</h2>
<p>{m['rows']} series-days, {m['positives']} with a recorded stockout (prevalence {m['prevalence']:.4f}). Thresholds
were fixed on earlier validation days. Counts use the order TN, FP, FN, TP.</p>
{_table(['Method', 'Average precision', 'F1', 'Precision', 'Recall', 'Accuracy', 'Balanced acc.', 'TN', 'FP', 'FN', 'TP'],
        metric_rows)}</section>
<section id="limits"><h2>Limits</h2><ul>
<li>100 existing series from one 90-day public dataset; no claim about new stores, products or later periods.</li>
<li>A recorded stockout is not lost demand, waste or money. Sales values were normalized by the publisher.</li>
<li>Associations only, no causes. No ordering or automatic action; a person reviews the list.</li>
<li>Data: FreshRetailNet-50K, Dingdong-Inc and Wang et al. (2025), CC BY 4.0. Subset and features derived for this
project; no endorsement implied.</li></ul></section>
</main></body></html>
"""
