"""FreshStock 2.0 desktop views. Every fact on screen comes from the verified bundle; helpers are shared with the pilot."""
import gc
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk

from .. import charts, gui as base
from ..charts import FONT, INK, SERIES
from ..data import KEY, DataError, parse_day
from ..gui import (Detail, METRICS, SHORT, Scroll, WEEKDAYS, card, columns_of, gaps, heading, para, score_bar, table,
                   tile, wrap)
from . import bundle as bd, data
from .metrics import ESTIMATORS, MODELS

PAGE = 100  # rows shown at once; the ranking always covers the whole cohort
SPLITS = {"initial_fit": "Fit", "tuning": "Tune", "calibration": "Calibrate", "threshold": "Threshold",
          "holdout": "Test"}
FEATURE_TEXT = {"sales_lag1": "Sales yesterday", "stockout_hours_lag1": "Stockout hours yesterday",
                "sales_mean7": "Mean sales, 7 days", "stockout_days7": "Stockout days, 7 days",
                "stockout_hours_mean7": "Mean stockout hours, 7", "stockout_hours_mean14": "Mean stockout hours, 14",
                "stockout_days14": "Stockout days, 14 days", "sales_mean14": "Mean sales, 14 days",
                "sales_mean28": "Mean sales, 28 days", "stockout_days28": "Stockout days, 28 days",
                "days_since_stockout28": "Days since a stockout",
                **{f"weekday_{i}": f"Weekday {d}" for i, d in enumerate(WEEKDAYS)}}


def short_date(text):
    day = parse_day(text)
    return f"{day.day} {day:%b}"


def span(first, last):
    a, b = parse_day(first), parse_day(last)
    return f"{a.day}–{b.day} {b:%b}" if a.month == b.month else f"{a.day} {a:%b} – {b.day} {b:%b}"


def number(value, pattern="{:.4f}"):
    return "–" if value is None else pattern.format(value)


def leaders(estimators, key):
    """Methods with the best value of a measure, and that value."""
    best = max(e[key] for e in estimators.values())
    return [n for n in ESTIMATORS if estimators[n][key] == best], best


def settings(bundle, model):
    """Selected settings in words, from the frozen file."""
    chosen = bundle.freeze["selected"][model]
    p = chosen["params"]
    text = (f"L2 penalty, C = {p['C']:g}, lbfgs, standardized numeric inputs" if model == "logistic" else
            f"{p['n_estimators']} trees, maximum depth {p['max_depth']}, at least {p['min_samples_leaf']} rows per leaf, "
            f"Gini, seed {bundle.protocol['seed']}")
    scores = "calibrated with a sigmoid" if bundle.freeze["calibration"][model]["chosen"] == "sigmoid" else "raw"
    return (f"input set {chosen['feature_set']} ({len(data.FEATURES[chosen['feature_set']])} columns), {text}; scores "
            f"{scores}; alert when score ≥ {bundle.freeze['thresholds'][model]}")


class App(base.App):
    TITLE = "FreshStock 2.0 - next-day stockout risk (historical replay)"
    health_of = staticmethod(bd.health)

    def __init__(self, bundle_root, reports_dir, load):
        super().__init__(bundle_root, reports_dir, load)

    def make_pages(self, bundle, health):
        return {"overview": overview(self.area, self, bundle), "predict": PredictPage(self.area, self),
                "experiment": experiment(self.area, self, bundle), "health": health_page(self.area, self, health)}

    def destroy(self):
        if self.predict is not None:
            self.predict.cancel()
        super().destroy()


# ---------- Overview ----------

def overview(parent, app, bundle):
    page, n, days, s = Scroll(parent, app), bundle.n, bundle.days, bundle.metrics["summary"]
    body = page.body
    top = ttk.Frame(body)
    top.pack(fill="x", pady=(app.px(12), app.px(4)))
    page.open_predict = ttk.Button(top, text="Open Predict  →", style="Main.TButton",
                                   command=lambda: app.show("predict"))
    page.open_predict.pack(side="right")
    ttk.Label(top, text="Which products should be checked first?", style="Title.TLabel").pack(anchor="w")
    wrap(app, ttk.Label(top, style="Muted.TLabel", text=f"Historical replay of {span(days[0], days[-1])} 2024. No live "
                        "inventory, and nothing is ordered.")).pack(fill="x")
    grid = columns_of(app, body, 3)
    tile(app, grid, 0, f"{n:,}", "store-product series ranked each day")
    tile(app, grid, 1, str(len(days)), "fresh test days to replay")
    tile(app, grid, 2, "2", "trained models, plus one simple rule")
    c = card(app, body, "How a forecast is made")
    charts.Pipeline(c, app, [("Past 28 days", "sales and stockout hours"),
                             ("Two frozen models", "Logistic Regression, Random Forest"),
                             ("Risk ranking", f"{n:,} series, one day ahead"),
                             ("Human review", "a planner decides")]).pack(fill="x")
    c = card(app, body, "Rows split by date: the test days stay hidden until the end")
    f = bundle.freeze
    parts = [(SPLITS[name], f["rows"][name], span(*(d.isoformat() for d in data.SPLITS[name])))
             for name in ("initial_fit", "tuning", "calibration", "threshold")]
    parts.append((SPLITS["holdout"], s["rows"], span(days[0], days[-1])))
    page.split = charts.SplitBar(c, app, parts)
    page.split.pack(fill="x")

    d1 = Detail(app, body, "The question, data source and license")
    para(app, d1.body, "Using information available by the end of day t−1, will this series record at least one "
         "stockout hour between 06:00 and 22:00 on day t? Yes (1) or no (0): supervised binary classification.")
    para(app, d1.body, "FreshRetailNet-50K, Dingdong-Inc and Wang et al. (2025), CC BY 4.0. Sales are normalized and "
         "IDs are encoded by the publisher. The test days come from the publisher's separate evaluation file.")
    source = bundle.protocol["sources"]
    rows = [("Publisher files", f"{source['train']['n_keys'] * source['train']['n_days']:,} training rows and "
             f"{source['holdout']['n_keys'] * source['holdout']['n_days']:,} evaluation rows"),
            ("Cohort", f"{n:,} of {source['train']['n_keys']:,} series, first by hash, not by outcome"),
            ("Warm-up", "28 earlier days are needed for the inputs, so examples start on 25 Apr 2024"),
            ("Refit", f"{f['rows']['refit']:,} rows, {f['positives']['refit']:,} with a stockout, 25 Apr to 11 Jun")]
    rows += [(SPLITS[p["split"]], f"{p['rows']:,} rows, {p['positives']:,} with a stockout ({p['rate']:.1%}), "
              f"{p['first']} to {p['last']}") for p in bd.prevalence(bundle)]
    table(app, d1.body, {"part": "Step", "value": "Content"}, rows, first=150, width=540, anchor="w").pack(fill="x")
    d2 = Detail(app, body, "Limits")
    for text in bundle.freeze["limitations"]:
        para(app, d2.body, "•  " + text[0].upper() + text[1:] + ".")
    para(app, d2.body, "•  Associations only, not causes. The app never places an order; a person reviews the list.")
    d3 = Detail(app, body, "File checks and versions")
    libraries = ", ".join(f"{k} {v}" for k, v in f["environment"].items() if k in ("scikit-learn", "numpy", "joblib"))
    for text in (f"✓ {len(bundle.pin['files'])} of {len(bundle.pin['files'])} files match the SHA-256 values built "
                 "into this program.",
                 f"✓ Run {f['run_id']}, protocol {f['protocol']}, seed {bundle.protocol['seed']}.",
                 f"✓ Required libraries: {libraries}.",
                 f"✓ Frozen {f['created_utc'][:16]} UTC, scored once {bundle.evaluation['evaluated_utc'][:16]} UTC."):
        para(app, d3.body, text)
    para(app, d3.body, "Fingerprints detect damage or mixed-up files. They do not prove who built the package.",
         "Card.Muted.TLabel")
    page.details = [d1, d2, d3]
    return page


# ---------- Experiment ----------

def experiment(parent, app, bundle):
    page, s = Scroll(parent, app), bundle.metrics["summary"]
    body, by = page.body, s["estimators"]
    negatives = s["rows"] - s["positives"]
    heading(app, body, "Two models against a simple rule", f"Fresh test, {span(bundle.days[0], bundle.days[-1])} 2024. "
            f"n = {s['rows']:,} series-days, {s['positives']:,} with a stockout. Scored once, after everything was "
            "frozen.")

    def bars(key, of=None):
        return [(SHORT[n], SERIES[n], (of or by)[n][key]) for n in ESTIMATORS]
    grid = columns_of(app, body, 2)
    c = card(app, grid, "Ranking and recall", row=0, column=0)
    page.ap = charts.Compare(c, app, [("Average precision (ranking)", bars("average_precision")),
                                      ("Recall (stockouts found)", bars("recall"))], 1.0, "{:.4f}")
    page.ap.pack(fill="x")
    para(app, c, f"Scale 0 to 1. A random ranking would reach {s['prevalence']:.4f} average precision. Seven days, not "
         "significance-tested.", "Card.Muted.TLabel")
    c = card(app, grid, "Workload: misses against false alerts", row=0, column=1, padx=gaps(app, 1))
    charts.Compare(c, app, [(f"Missed stockouts (of {s['positives']:,})", bars("fn")),
                            (f"False alerts (of {negatives:,})", bars("fp"))], max(negatives, s["positives"]),
                   "{:,.0f}").pack(fill="x")
    grid = columns_of(app, body, 3)
    for i, (title, key) in enumerate((("Best ranking", "average_precision"), ("Finds the most", "recall"),
                                      ("Most accurate", "accuracy"))):
        names, value = leaders(by, key)
        para(app, card(app, grid, f"{title}: {' and '.join(SHORT[n] for n in names)}", row=0, column=i,
                       padx=gaps(app, i)), f"{METRICS[key]} {value:.4f}, with {by[names[0]]['fp']:,} false alerts and "
             f"{by[names[0]]['fn']:,} misses.")
    winners = {n for key in METRICS for n in leaders(by, key)[0]}
    verdict = (f"{SHORT[winners.pop()]} leads on all six measures of these seven days; costs were not measured."
               if len(winners) == 1 else "No universal winner: the choice depends on what a miss and a false alert "
               "cost, and that was not measured.")
    wrap(app, ttk.Label(body, style="Good.TLabel", text=verdict)).pack(fill="x", pady=app.px(4))
    k = by["logistic"]["top_k"]
    c = card(app, body, f"If a planner checks the top 10% each day ({k['k_per_day']:,} series)")
    charts.Compare(c, app, [(f"Stockouts among the {k['k_total']:,} checks", [
        (SHORT[n], SERIES[n], by[n]["top_k"]["hits"]) for n in ESTIMATORS])], k["k_total"], "{:,.0f}").pack(fill="x")
    para(app, c, "The rule has only two score values, so its order inside a tie follows the series key.",
         "Card.Muted.TLabel")

    d1 = Detail(app, body, "All measures")
    table(app, d1.body, {"name": "Method", **METRICS, "precision10": "Top-10% prec.", "brier": "Brier"},
          [[SHORT[n]] + [number(by[n][m]) for m in METRICS] + [number(by[n]["top_k"]["precision"]), number(
              by[n].get("calibration", {}).get("brier"))] for n in ESTIMATORS], first=170, width=82).pack(fill="x")
    rate = s["prevalence"]
    para(app, d1.body, f"Without any model: always giving the commoner answer is right {max(rate, 1 - rate):.4f} of the "
         f"time, and flagging every series gives F1 {2 * rate / (1 + rate):.4f}.", "Card.Muted.TLabel")
    d2 = Detail(app, body, "Confusion matrices (rows: actual, columns: predicted)")
    grid = columns_of(app, d2.body, 3, "Plain.TFrame")
    for i, n in enumerate(ESTIMATORS):
        charts.Confusion(grid, app, SHORT[n], SERIES[n], by[n]).grid(row=0, column=i, sticky="ew")
    d3 = Detail(app, body, "How the settings were chosen (tuning days only)")
    chosen = set(bundle.candidates["selected"].values())
    table(app, d3.body, {"id": "Candidate", "set": "Inputs", "ap": "Tuning AP", "brier": "Brier", "used": "Selected"},
          [[describe(c), c["feature_set"], number(c["tuning_average_precision"]), number(c["tuning_brier"]),
            "✓" if c["id"] in chosen else ""] for c in bundle.candidates["candidates"]], first=300,
          width=92).pack(fill="x")
    same = bundle.candidates["same_feature_comparison"]
    para(app, d3.body, "Same inputs, best of each family: " + "; ".join(
        f"set {x}: Logistic Regression {v['logistic']:.4f}, Random Forest {v['forest']:.4f}" for x, v in same.items())
        + ".", "Card.Muted.TLabel")
    d4 = Detail(app, body, "Score reliability on the test days")
    grid = columns_of(app, d4.body, 2, "Plain.TFrame")
    for i, n in enumerate(MODELS):
        frame = ttk.Frame(grid, style="Plain.TFrame")
        frame.grid(row=0, column=i, sticky="nsew", padx=gaps(app, i))
        record = bundle.freeze["calibration"][n]
        para(app, frame, f"{SHORT[n]}: {record['chosen']} scores used")
        table(app, frame, {"bin": "Score", "n": "Rows", "mean": "Mean score", "rate": "Observed"},
              [[f"{b['low']:.1f}–{b['high']:.1f}", f"{b['n']:,}", number(b["mean_score"], "{:.3f}"),
                number(b["observed_rate"], "{:.3f}")] for b in by[n]["calibration"]["bins"]], first=80,
              width=82).pack(fill="x")
    para(app, d4.body, "The variant was chosen by Brier score on the threshold days, before the test. Scores are not "
         "exact probabilities.", "Card.Muted.TLabel")
    d5 = Detail(app, body, "Results by group")
    rows = [[f"Stockout the day before: {name}", f"{g['n']:,}", f"{g['positives']:,}"] + [
        number(g["estimators"][n][m]) for n in MODELS for m in ("average_precision", "recall")]
        for name, g in s["slices"]["previous_day_stockout"].items()]
    rows += [[f"City {c}", f"{g['n']:,}", f"{g['positives']:,}"] + [
        number(g["estimators"][n][m]) for n in MODELS for m in ("average_precision", "recall")]
        for c, g in s["slices"]["city"].items()]
    table(app, d5.body, {"name": "Group", "n": "Rows", "pos": "Stockouts", "lap": "LR AP", "lr": "LR recall",
                         "fap": "RF AP", "fr": "RF recall"}, rows, first=210, width=78).pack(fill="x")
    para(app, d5.body, "– means undefined: the group is empty or has one class.", "Card.Muted.TLabel")
    d6 = Detail(app, body, "Models, inputs and thresholds")
    for text in (
            "Input set A: sales and stockout hours yesterday, mean sales and stockout days of the last 7 days, weekday. "
            "Set B adds seven longer summaries of the last 7, 14 and 28 days. Target-day values and IDs are never inputs.",
            *(f"{SHORT[n]}: {settings(bundle, n)}." for n in MODELS),
            "Persistence rule: alert if the series had a stockout yesterday. It is a baseline, not a trained model.",
            "Thresholds come from the threshold days only (best F1, then precision, then the higher value).",
            "If the two models use different input sets, this compares two optimized pipelines, not the algorithms alone."):
        para(app, d6.body, "•  " + text)
    page.details = [d1, d2, d3, d4, d5, d6]
    return page


def describe(c):
    p = c["params"]
    return (f"Logistic Regression, C = {p['C']:g}" if c["family"] == "logistic" else
            f"Random Forest, {p['n_estimators']} trees, depth {p['max_depth']}, leaf {p['min_samples_leaf']}")


# ---------- Predict ----------

class PredictPage(ttk.Frame):
    KEYS = base.PredictPage.KEYS
    COLUMNS, WIDTH, WORDS = base.PredictPage.COLUMNS, {**base.PredictPage.WIDTH, "rank": 48}, base.PredictPage.WORDS

    def __init__(self, parent, app):
        super().__init__(parent, padding=(app.px(24), app.px(8), app.px(24), app.px(12)))
        self.app, self.bundle, self.rows, self.saved = app, app.bundle, {}, None
        self.forecast = self.revealed = None
        self.busy, self.token, self.page, self._poll = False, 0, 0, None
        heading(app, self, "Forecast first, then check", "Scores use only data up to the day before. Reveal shows what "
                "was recorded.")
        bar = ttk.Frame(self)
        bar.pack(fill="x")
        self.day, self.sort = tk.StringVar(value=self.bundle.days[0]), tk.StringVar(value=SHORT["logistic"])
        self.compare, self.find = tk.BooleanVar(value=False), tk.StringVar()
        gap = (app.px(6), app.px(12))
        ttk.Label(bar, text="Date").pack(side="left")
        self.day_box = ttk.Combobox(bar, textvariable=self.day, values=self.bundle.days, width=12)
        self.day_box.pack(side="left", padx=gap)
        self.day_box.bind("<Return>", lambda e: self.generate())
        ttk.Label(bar, text="Rank by").pack(side="left")
        self.sort_box = ttk.Combobox(bar, textvariable=self.sort, values=[SHORT[n] for n in MODELS], state="readonly",
                                     width=18)
        self.sort_box.pack(side="left", padx=gap)
        self.generate_button = ttk.Button(bar, text="Generate forecast", command=self.generate, style="Main.TButton")
        self.reveal_button = ttk.Button(bar, text="Reveal outcome", command=self.reveal)
        self.export_button = ttk.Button(bar, text="Export report", command=self.export)
        for b in (self.generate_button, self.reveal_button, self.export_button):
            b.pack(side="left", padx=(0, app.px(8)))
        self.message = wrap(app, ttk.Label(self, style="Muted.TLabel"))
        self.message.pack(fill="x", pady=(app.px(10), app.px(6)))
        self.open_button = ttk.Button(self, text="Open report folder", command=lambda: os.startfile(self.saved))
        self.tiles = columns_of(app, self, 3)
        self.workload = {n: tile(app, self.tiles, i, "", f"alerts of {self.bundle.n:,} by {SHORT[n]}", SERIES[n])
                         for i, n in enumerate(ESTIMATORS)}
        self.outcomes = charts.Stacked(self, app)
        self.review = wrap(app, ttk.Label(self, style="Good.TLabel"))
        self.list_bar = ttk.Frame(self)
        self.caption = ttk.Label(self.list_bar, style="Muted.TLabel")
        self.caption.pack(side="left")
        ttk.Checkbutton(self.list_bar, text="Compare all methods", variable=self.compare,
                        command=self.refresh).pack(side="right")
        self.next_button = ttk.Button(self.list_bar, text="Next ›", command=lambda: self.turn(1))
        self.page_label = ttk.Label(self.list_bar, style="Muted.TLabel")
        self.previous_button = ttk.Button(self.list_bar, text="‹ Previous", command=lambda: self.turn(-1))
        self.find_button = ttk.Button(self.list_bar, text="Find", command=self.find_series)
        self.find_entry = ttk.Entry(self.list_bar, textvariable=self.find, width=14)
        self.find_entry.bind("<Return>", lambda e: self.find_series())
        for w in (self.next_button, self.page_label, self.previous_button, self.find_button, self.find_entry):
            w.pack(side="right", padx=(0, app.px(8)))
        ttk.Label(self.list_bar, text="City, store, product", style="Muted.TLabel").pack(side="right",
                                                                                          padx=(0, app.px(6)))
        self.detail = ttk.Frame(self)
        self.detail.pack(side="bottom", fill="x", pady=(app.px(8), 0))
        self.holder = ttk.Frame(self)
        self.holder.pack(side="bottom", fill="both", expand=True)
        self.list_bar.pack(side="bottom", fill="x", pady=(0, app.px(4)))
        self.table = table(app, self.holder, self.COLUMNS, height=4)
        for c in self.COLUMNS:
            side = "w" if c == "bar" else "center" if c in self.WORDS else "e"
            self.table.heading(c, anchor=side)
            self.table.column(c, anchor=side)
        self.table.tag_configure("alert", background="#FBF3E4")
        slider = ttk.Scrollbar(self.holder, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=slider.set)
        slider.pack(side="right", fill="y")
        self.table.pack(side="left", fill="both", expand=True)
        self.table.bind("<<TreeviewSelect>>", lambda e: self.show_detail())
        self.table.bind("<Configure>", lambda e: self.after_idle(self.fit_columns))
        self.day.trace_add("write", lambda *a: self.reset())
        self.sort.trace_add("write", lambda *a: self.turn_to(0))
        self.refresh()

    # ---------- state ----------

    @property
    def state(self):
        return "select" if self.forecast is None else "forecast" if self.revealed is None else "revealed"

    @property
    def sort_name(self):
        return next(n for n in MODELS if SHORT[n] == self.sort.get())

    def say(self, text, style="Muted.TLabel"):
        self.message.configure(text=text, style=style)

    def reset(self, refresh=True):
        """New date, refusal or failure: no earlier row may stay, and any running job becomes stale."""
        self.token += 1
        self.busy = False
        self.forecast = self.revealed = None
        self.page = 0
        if refresh:
            self.refresh()

    def cancel(self):
        self.token += 1
        if self._poll:
            self.after_cancel(self._poll)
            self._poll = None
        gc.enable()

    def run(self, work, done, *args):
        """One background job without Tk calls. Its result is used only if nothing changed meanwhile."""
        self.busy = True
        self.refresh_buttons()
        token, answers = self.token, queue.Queue()
        gc.collect()  # Tcl aborts if another thread's collection frees Tk objects, so collect here and pause it
        gc.disable()

        def worker():
            try:
                answers.put((work(*args), None))
            except Exception as error:  # shown on the page
                answers.put((None, error))
        threading.Thread(target=worker, daemon=True).start()
        self._poll = self.after(20, self._wait, answers, token, done)

    def _wait(self, answers, token, done):
        try:
            result, error = answers.get_nowait()
        except queue.Empty:
            self._poll = self.after(20, self._wait, answers, token, done)
            return
        self._poll = None
        gc.enable()
        if token != self.token:
            return
        self.busy = False
        done(result, error)

    def stopped(self, error):
        self.reset()
        self.say(f"Stopped: {error if isinstance(error, DataError) else f'{type(error).__name__}: {error}'}",
                 "Danger.TLabel")

    # ---------- actions ----------

    def generate(self):
        if self.busy or self._poll:
            return
        self.reset()
        self.say("Scoring every series from its past 28 days …")
        self.run(bd.forecast, self._generated, self.bundle, self.day.get().strip())

    def _generated(self, result, error):
        if error is not None:
            return self.stopped(error)
        self.forecast, self.page = result, 0
        self.refresh()

    def reveal(self):
        if self.busy or self.state != "forecast":
            return
        try:
            self.revealed = bd.reveal(self.bundle, self.forecast)
        except Exception as error:  # never leave a half-revealed page
            return self.stopped(error)
        self.refresh()

    def export(self):
        if self.busy or self.state != "revealed":
            return
        self.saved = None
        self.run(bd.export, self._exported, self.bundle, self.forecast, self.revealed, self.sort_name,
                 self.app.reports_dir)

    def _exported(self, result, error):
        self.refresh_buttons()
        if error is not None:
            self.say(f"The report was not saved: {error}", "Danger.TLabel")
            return
        self.saved = result
        self.say(f"Saved: {result}", "Good.TLabel")
        self.open_button.pack(anchor="w", pady=(0, self.app.px(8)), after=self.message)

    def turn(self, step):
        if self.state != "select" and 0 <= self.page + step < bd.page_count(self.bundle.n, PAGE):
            self.turn_to(self.page + step)

    def turn_to(self, page):
        self.page = page
        self.table.selection_set(())
        self.refresh()

    def find_series(self):
        if self.state == "select":
            return
        try:
            key = tuple(int(v) for v in self.find.get().split(","))
            if len(key) != 3:
                raise ValueError
        except ValueError:
            return self.say("Type city, store, product as three numbers, for example 0,12,34.", "Warn.TLabel")
        rank = bd.find_rank(self.bundle, self.forecast, self.sort_name, key)
        if rank is None:
            return self.say(f"Series {','.join(map(str, key))} is not in this cohort.", "Warn.TLabel")
        self.turn_to((rank - 1) // PAGE)
        item = next(i for i, row in self.rows.items() if row["rank"] == rank)
        self.table.selection_set(item)
        self.table.see(item)

    # ---------- drawing ----------

    columns = base.PredictPage.columns
    fit_columns = base.PredictPage.fit_columns

    def refresh_buttons(self):
        pages = bd.page_count(self.bundle.n, PAGE)
        for button, on in ((self.generate_button, not self.busy),
                           (self.reveal_button, not self.busy and self.state == "forecast"),
                           (self.export_button, not self.busy and self.state == "revealed"),
                           (self.previous_button, self.state != "select" and self.page > 0),
                           (self.next_button, self.state != "select" and self.page + 1 < pages),
                           (self.find_button, self.state != "select")):
            button.state(["!disabled" if on else "disabled"])
        self.page_label.configure(text=f"Page {self.page + 1} of {pages}" if self.state != "select" else "")

    def refresh(self):
        revealed = self.state == "revealed"
        chosen = next((self.rows[i]["rank"] for i in self.table.selection() if i in self.rows), None)
        for widget in (self.open_button, self.tiles, self.outcomes, self.review):
            widget.pack_forget()
        self.table.delete(*self.table.get_children())
        self.outcomes.show({}, {})
        self.table.configure(displaycolumns=self.columns(revealed))
        self.fit_columns()
        self.rows = {}
        self.refresh_buttons()
        if self.state == "select":
            days = self.bundle.days
            self.say(f"Choose a date from {days[0]} to {days[-1]}, then generate the forecast.")
            self.caption.configure(text="")
            self.show_detail()
            return
        model, f = self.sort_name, self.forecast
        for row in bd.page_rows(self.bundle, f, model, self.page, PAGE, self.revealed):
            score = row[f"{model}_score"]
            flags = {f"{e}_flag": "▲ Alert" if row[f"{e}_flag"] else "–  Clear" for e in ESTIMATORS}
            shown = {**row, **{f"{m}_score": f"{row[f'{m}_score']:.4f}" for m in MODELS}, **flags,
                     "rank": f"{row['rank']:,}", "score": f"{score:.4f}", "bar": score_bar(score),
                     "flag": flags[f"{model}_flag"]}
            if revealed:
                shown["observed"] = "Stockout" if row["y_true"] else "None"
                shown.update({f"{e}_result": row["result"][e] for e in ESTIMATORS})
                shown["result"] = row["result"][model]
            alert = row[f"{model}_flag"] and not self.compare.get()
            item = self.table.insert("", "end", values=[shown.get(c, "") for c in self.COLUMNS],
                                     tags=("alert",) if alert else ())
            self.rows[item] = row
            if row["rank"] == chosen:
                self.table.selection_set(item)
                self.table.see(item)
        self.caption.configure(text=f"Alert from {self.bundle.freeze['thresholds'][model]}")
        gap = (0, self.app.px(6))
        if revealed:
            r = self.revealed
            self.review.configure(text="Top 10% check list, stockouts found: " + ", ".join(
                f"{SHORT[n]} {r.review[n]['hits']:,} of {r.review[n]['k']:,}" for n in ESTIMATORS))
            self.review.pack(fill="x", pady=gap, after=self.message)
            self.outcomes.show(SHORT, {n: tuple(r.counts[n][k] for k in ("tp", "fn", "fp", "tn")) for n in ESTIMATORS})
            self.outcomes.pack(fill="x", pady=gap, after=self.message)
            self.say(f"Recorded on {f.day}: {r.positives:,} of {r.total:,} series had a stockout.", "Head.TLabel")
        else:
            for n, value in self.workload.items():
                value.configure(text=f"{int(f.flags[n].sum()):,}")
            self.tiles.pack(fill="x", pady=gap, after=self.message)
            self.say(f"Forecast for {f.day}. The score ranks risk; it is not an exact probability.")
        self.show_detail()

    def show_detail(self):
        for child in self.detail.winfo_children():
            child.destroy()
        chosen = self.table.selection()
        if not chosen or chosen[0] not in self.rows:
            hint = "Select a row to see what the models used." if self.rows else ""
            ttk.Label(self.detail, text=hint, style="Muted.TLabel").pack(anchor="w")
            return
        row, x = self.rows[chosen[0]], self.rows[chosen[0]]["features"]
        before = parse_day(row["dt"]).toordinal() - 1
        ttk.Label(self.detail, style="Muted.TLabel", text=f"City {row['city_id']}, store {row['store_id']}, product "
                  f"{row['product_id']}: inputs up to {type(parse_day(row['dt'])).fromordinal(before).isoformat()}"
                  ).pack(anchor="w")
        facts = [("Sales yesterday", f"{x['sales_lag1']:.2f}"), ("Stockout hours", f"{x['stockout_hours_lag1']:.0f}"),
                 ("Mean sales, 7 / 28", f"{x['sales_mean7']:.2f} / {x['sales_mean28']:.2f}"),
                 ("Stockout days, 7 / 28", f"{x['stockout_days7']:.0f} / {x['stockout_days28']:.0f}"),
                 ("Days since stockout", f"{x['days_since_stockout28']:.0f}"),
                 ("Weekday", next(d for i, d in enumerate(WEEKDAYS) if x[f"weekday_{i}"])),
                 ("LR score", f"{row['logistic_score']:.4f}"), ("RF score", f"{row['forest_score']:.4f}"),
                 ("Rule", "Alert" if row["persistence_flag"] else "Clear")]
        if self.state == "revealed":
            facts.append(("Observed", "Stockout" if row["y_true"] else "None"))
        line = ttk.Frame(self.detail)
        line.pack(fill="x")
        for i, (name, value) in enumerate(facts):
            cell = ttk.Frame(line)
            cell.grid(row=0, column=i, sticky="w")
            line.columnconfigure(i, weight=1)
            ttk.Label(cell, text=value, font=(FONT, 11, "bold")).pack(anchor="w")
            ttk.Label(cell, text=name, style="Muted.TLabel", font=(FONT, 8)).pack(anchor="w")


# ---------- Model health ----------

def health_page(parent, app, h):
    page = Scroll(parent, app)
    body, daily, work = page.body, h["daily"], h["workload"]
    n = h["series"]
    heading(app, body, "Model health, looking back", f"Saved predictions for {span(daily[0]['dt'], daily[-1]['dt'])} "
            "2024. This page does not monitor live data and does not prove drift.")
    grid = columns_of(app, body, 2, weights=[6, 6])
    c = card(app, grid, "Share of series-days with a stockout", row=0, column=0)
    rates = [p["rate"] for p in h["prevalence"]]
    charts.Columns(c, app, [SPLITS[p["split"]] for p in h["prevalence"]],
                   [span(p["first"], p["last"]) for p in h["prevalence"]], rates, max(.5, *rates), "{:.1%}").pack(fill="x")
    change = (rates[-1] - rates[-2]) * 100
    para(app, c, f"{change:+.1f} points from the threshold days to the test days.", "Card.Good.TLabel")
    c = card(app, grid, f"Average test day ({n:,} series)", row=0, column=1, padx=gaps(app, 1))
    top = max(w[key] for w in work.values() for key in ("alerts_per_day", "misses_per_day"))
    charts.Compare(c, app, [(title, [(SHORT[m], SERIES[m], w[key]) for m, w in work.items()])
                            for title, key in (("Alerts", "alerts_per_day"), ("Missed stockouts", "misses_per_day"))],
                   top, "{:,.0f}").pack(fill="x")
    c = card(app, body, None)
    bar = ttk.Frame(c, style="Plain.TFrame")
    bar.pack(fill="x", pady=(0, app.px(8)))
    ttk.Label(bar, text="Each test day (series)", style="Card.Head.TLabel").pack(side="left", padx=(0, app.px(16)))
    page.daily = charts.Lines(c, app, [d["dt"] for d in daily])
    page.daily.pack(fill="x")
    page.choice = tk.StringVar(app, value="alerts")

    def redraw():
        key = page.choice.get()
        series = [{"name": SHORT[m], "color": SERIES[m], "values": [d["estimators"][m][key] for d in daily]}
                  for m in ESTIMATORS]
        if key == "alerts":
            series.insert(0, {"name": "Recorded stockouts", "color": INK, "values": [d["positives"] for d in daily]})
        page.daily.show(series)
    page.measures = {}
    for key, title in base.MEASURES.items():
        page.measures[key] = ttk.Radiobutton(bar, text=title, value=key, variable=page.choice, command=redraw,
                                             style="Choice.Toolbutton")
        page.measures[key].pack(side="left", padx=(0, app.px(6)))
    redraw()

    d1 = Detail(app, body, "Every test day in numbers")
    columns = {"dt": "Date", "positives": "Recorded"}
    for m, short in (("logistic", "LR"), ("forest", "RF"), ("persistence", "Rule")):
        columns.update({f"{m}_alerts": f"{short} alerts", f"{m}_fn": f"{short} miss", f"{m}_fp": f"{short} false"})
    table(app, d1.body, columns, [[d["dt"], f"{d['positives']:,}"] + [f"{d['estimators'][m][k]:,}" for m in ESTIMATORS
                                                                     for k in ("alerts", "fn", "fp")] for d in daily],
          first=90, width=68).pack(fill="x")
    d2 = Detail(app, body, "What the models lean on")
    grid = columns_of(app, d2.body, 2, "Plain.TFrame")
    for i, (m, title) in enumerate((("logistic", "Logistic Regression: coefficient"),
                                    ("forest", "Random Forest: impurity importance"))):
        pairs = h["explain"][m]
        charts.HBars(grid, app, title, [FEATURE_TEXT[p["feature"]] for p in pairs], [p["value"] for p in pairs],
                     SERIES[m]).grid(row=0, column=i, sticky="new")
    para(app, d2.body, "Whole-model associations of the selected base pipelines, not causes, and not an explanation of "
         "one prediction. Inputs are correlated, so single values can mislead.")
    d3 = Detail(app, body, "File fingerprints")
    table(app, d3.body, {"file": "Verified file", "sha": "SHA-256"}, list(h["digests"].items()), first=190, width=540,
          anchor="w").pack(fill="x", pady=(app.px(6), 0))
    page.details = [d1, d2, d3]
    return page


def main(bundle_root, reports_dir, load):
    base.use_screen_scale()
    App(bundle_root, reports_dir, load).mainloop()
