"""Offline desktop app: Overview, Experiment, Predict and Model health. Shows the frozen run; changes nothing in it."""
import ctypes
import gc
import os
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from . import charts, data, experiment as ex, portfolio as pf
from .charts import BRAND, CANVAS, FONT, INK, LINE, MUTED, SERIES, SOFT, SURFACE

WARNING, DANGER = "#8C5100", "#A32B32"
DISABLED, DISABLED_BACK = "#6E7D75", "#E9EEEA"
SHORT = {"logistic": "Logistic Regression", "forest": "Random Forest", "persistence": "Persistence rule"}
VIEWS = {"overview": "Overview", "predict": "Predict", "experiment": "Experiment", "health": "Model health"}
BAR = 20  # characters of a full score bar
SPLIT_TITLE = {"train": "Train", "validation": "Validation", "test": "Final test"}
METRICS = {"average_precision": "Avg. precision", "f1": "F1", "precision": "Precision", "recall": "Recall",
           "accuracy": "Accuracy", "balanced_accuracy": "Bal. accuracy"}
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
FEATURE_TEXT = {"sales_lag1": "Sales yesterday", "stockout_hours_lag1": "Stockout hours yesterday",
                "sales_mean7": "Mean sales, 7 days", "stockout_days7": "Stockout days, 7 days",
                **{f"weekday_{i}": f"Weekday {d}" for i, d in enumerate(WEEKDAYS)}}
LIMITS = ["One public source and 90 days. The same 100 series appear in training and testing, so nothing is claimed "
          "about new stores, products or later periods.",
          "Scores rank series. They are not calibrated probabilities.",
          "A recorded stockout is not lost demand, waste or money. Sales units were normalized by the publisher.",
          "Associations only, not causes. The app never places an order; a person reviews the list."]


class Scroll(ttk.Frame):
    """Vertically scrolling page. Put content in .body."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.view = tk.Canvas(self, bg=CANVAS, highlightthickness=0)
        bar = ttk.Scrollbar(self, orient="vertical", command=self.view.yview)
        self.view.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        self.view.pack(side="left", fill="both", expand=True)
        pad = app.px(24)
        self.body = ttk.Frame(self.view, padding=(pad, app.px(8), pad, pad))
        window = self.view.create_window(0, 0, anchor="nw", window=self.body)
        self.body.bind("<Configure>", lambda e: self.view.configure(scrollregion=self.view.bbox("all")))
        self.view.bind("<Configure>", lambda e: self.view.itemconfigure(window, width=e.width))

    def scroll(self, amount, what):
        if self.view.yview() != (0.0, 1.0):
            self.view.yview_scroll(amount, what)


class App(tk.Tk):
    TITLE = "FreshStock - next-day stockout risk (historical replay)"
    health_of = staticmethod(pf.build_health)

    def __init__(self, bundle_root, reports_dir, load=pf.load_verified_bundle):
        super().__init__()
        self.bundle_root, self.reports_dir, self._load = Path(bundle_root), Path(reports_dir), load
        self.scale, self._poll = self.winfo_fpixels("1i") / 96, None
        self.title(self.TITLE)
        self.configure(bg=CANVAS)
        width = min(self.px(1240), self.winfo_screenwidth() - self.px(20))
        height = min(self.px(780), self.winfo_screenheight() - self.px(90))
        self.geometry(f"{width}x{height}+{max(0, (self.winfo_screenwidth() - width) // 2)}+{self.px(10)}")
        self.minsize(min(self.px(1010), width), min(self.px(600), height))
        self._style()
        self._shell()
        self.report_callback_exception = self._unexpected
        self.start()

    def px(self, n):
        return round(n * self.scale)

    # ---------- look ----------

    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=CANVAS, foreground=INK, font=(FONT, 10), bordercolor=LINE, lightcolor=SURFACE,
                    darkcolor=LINE, troughcolor=CANVAS, focuscolor=BRAND)
        for name, size, weight, color in (("Title", 18, "bold", INK), ("Head", 13, "bold", INK),
                                          ("Muted", 10, "normal", MUTED), ("Warn", 10, "bold", WARNING),
                                          ("Danger", 10, "bold", DANGER), ("Good", 10, "bold", BRAND),
                                          ("Brand", 15, "bold", BRAND), ("Number", 24, "bold", BRAND)):
            s.configure(f"{name}.TLabel", font=(FONT, size, weight), foreground=color)
            s.configure(f"Card.{name}.TLabel", font=(FONT, size, weight), foreground=color, background=SURFACE)
        s.configure("Card.TLabel", background=SURFACE)
        s.configure("Card.TFrame", background=SURFACE, bordercolor=LINE, relief="solid", borderwidth=1)
        s.configure("Plain.TFrame", background=SURFACE)
        s.configure("TButton", padding=(self.px(11), self.px(7)), background=SURFACE, bordercolor=MUTED)
        s.map("TButton", background=[("disabled", DISABLED_BACK), ("pressed", LINE), ("active", "#EDF3EF")],
              foreground=[("disabled", DISABLED)], bordercolor=[("focus", BRAND)])
        s.configure("Main.TButton", background=BRAND, foreground="#FFFFFF", bordercolor=BRAND, focuscolor="#FFFFFF")
        s.map("Main.TButton", background=[("disabled", DISABLED_BACK), ("pressed", "#0F3F31"), ("active", "#124A39")],
              foreground=[("disabled", DISABLED)])
        s.configure("Nav.TButton", anchor="w", background=SURFACE, bordercolor=SURFACE, lightcolor=SURFACE,
                    darkcolor=SURFACE, relief="flat", font=(FONT, 11), padding=(self.px(12), self.px(9)))
        s.map("Nav.TButton", background=[("selected", "#E3EEE7"), ("active", "#EDF3EF"), ("disabled", SURFACE)],
              foreground=[("disabled", DISABLED), ("selected", BRAND)], bordercolor=[("focus", BRAND)])
        s.configure("Detail.TButton", anchor="w", background=CANVAS, bordercolor=CANVAS, lightcolor=CANVAS,
                    darkcolor=CANVAS, relief="flat", foreground=BRAND, font=(FONT, 10, "bold"))
        s.map("Detail.TButton", background=[("active", SOFT)], bordercolor=[("focus", BRAND)])
        s.configure("Choice.Toolbutton", background=SURFACE, bordercolor=LINE, padding=(self.px(12), self.px(5)),
                    relief="solid", borderwidth=1)
        s.map("Choice.Toolbutton", background=[("selected", BRAND), ("active", SOFT)],
              foreground=[("selected", "#FFFFFF")], bordercolor=[("focus", INK)])
        s.configure("TCheckbutton", indicatorsize=self.px(14), focuscolor=BRAND)
        s.map("TCheckbutton", background=[("active", CANVAS)])
        s.configure("Treeview", background=SURFACE, fieldbackground=SURFACE, foreground=INK, borderwidth=1,
                    rowheight=self.px(25))
        s.configure("Treeview.Heading", background=CANVAS, font=(FONT, 9, "bold"), relief="flat",
                    padding=(self.px(4), self.px(5)))
        s.map("Treeview", background=[("selected", BRAND)], foreground=[("selected", "#FFFFFF")])
        s.configure("TCombobox", fieldbackground=SURFACE, padding=self.px(5), arrowsize=self.px(14))
        s.map("TCombobox", fieldbackground=[("readonly", SURFACE)], foreground=[("readonly", INK)],
              selectbackground=[("readonly", SURFACE)], selectforeground=[("readonly", INK)],
              bordercolor=[("focus", BRAND)])
        self.option_add("*TCombobox*Listbox.font", (FONT, 10))
        self.bind_class("TButton", "<Return>", lambda e: e.widget.invoke())

    def _shell(self):
        side = tk.Frame(self, bg=SURFACE, highlightbackground=LINE, highlightthickness=1, width=self.px(190))
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        left = {"anchor": "w", "padx": self.px(16)}
        ttk.Label(side, text="FreshStock", style="Card.Brand.TLabel").pack(**left, pady=(self.px(18), 0))
        ttk.Label(side, text="Historical replay", style="Card.Muted.TLabel", font=(FONT, 9),
                  wraplength=self.px(160)).pack(**left, pady=(0, self.px(16)))
        self.nav = {}
        for i, (name, title) in enumerate(VIEWS.items(), 1):
            b = ttk.Button(side, text=title, style="Nav.TButton", command=lambda n=name: self.show(n))
            b.pack(fill="x", padx=self.px(8), pady=self.px(2))
            self.nav[name] = b
            self.bind(f"<Control-Key-{i}>", lambda e, n=name: self.show(n))
        self.status_label = ttk.Label(side, style="Card.TLabel", wraplength=self.px(160), font=(FONT, 9, "bold"))
        self.status_label.pack(**left, side="bottom", pady=self.px(16))
        self.error_bar = ttk.Label(self, style="Danger.TLabel", padding=self.px(8), wraplength=self.px(900))
        self.area = ttk.Frame(self)
        self.area.pack(side="left", fill="both", expand=True)
        self.bind("<MouseWheel>", self._wheel)
        self.bind("<Prior>", lambda e: self._page(-1))
        self.bind("<Next>", lambda e: self._page(1))
        for key, name in (("g", "generate"), ("r", "reveal"), ("e", "export")):
            self.bind(f"<Control-{key}>", lambda e, n=name: self._predict_key(n))

    def _wheel(self, event):
        page = self.pages.get(self.current)
        if isinstance(page, Scroll):
            page.scroll(-1 if event.delta > 0 else 1, "units")

    def _page(self, step):
        page = self.pages.get(self.current)
        if isinstance(page, Scroll):
            page.scroll(step, "pages")

    def _predict_key(self, name):
        if self.status == "ready" and self.current == "predict":
            getattr(self.predict, f"{name}_button").invoke()

    # ---------- checking, blocked, ready ----------

    def start(self):
        self._clear()
        self.status = "checking"
        self.status_label.configure(text="… Checking files", foreground=MUTED)
        box = ttk.Frame(self.area, padding=self.px(48))
        box.pack(fill="both", expand=True)
        ttk.Label(box, text="Checking the frozen run", style="Title.TLabel").pack(anchor="w")
        ttk.Label(box, style="Muted.TLabel", text="Comparing every data and model file with its recorded "
                  "fingerprint. This takes a few seconds.").pack(anchor="w", pady=self.px(8))
        gc.collect()  # free closed windows here: Tcl aborts if the loader thread's garbage collection does it
        answers, load, root = queue.Queue(), self._load, self.bundle_root  # the thread must not hold the window
        health_of = self.health_of

        def work():
            try:
                bundle = load(root)
                answers.put((bundle, health_of(bundle), None))
            except Exception as error:  # every failure must reach the screen, not a hidden console
                answers.put((None, None, error))
        threading.Thread(target=work, daemon=True).start()
        self._poll = self.after(30, self._wait, answers)

    def _wait(self, answers):
        try:
            bundle, health, error = answers.get_nowait()
        except queue.Empty:
            self._poll = self.after(30, self._wait, answers)
            return
        self._poll = None
        if error is None:
            self._ready(bundle, health)
        else:
            self._blocked(error)

    def _clear(self):
        self.pages, self.current, self.predict = {}, None, None
        self.error_bar.pack_forget()
        for child in self.area.winfo_children():
            child.destroy()
        for b in self.nav.values():
            b.state(["disabled", "!selected"])

    def _blocked(self, error):
        self._clear()
        self.status = "blocked"
        self.status_label.configure(text="✕ Blocked", foreground=DANGER)
        box = ttk.Frame(self.area, padding=self.px(48))
        box.pack(fill="both", expand=True)
        ttk.Label(box, text="FreshStock cannot start", style="Title.TLabel").pack(anchor="w")
        self.problem = str(error) if isinstance(error, data.DataError) else f"{type(error).__name__}: {error}"
        ttk.Label(box, text=self.problem, style="Danger.TLabel", wraplength=self.px(680)).pack(anchor="w", pady=self.px(12))
        ttk.Label(box, wraplength=self.px(680), text="No scores are shown because the files are not the approved ones. "
                  "Nothing was repaired or retrained. Restore the original files, then retry.").pack(anchor="w")
        row = ttk.Frame(box)
        row.pack(anchor="w", pady=self.px(20))
        self.retry_button = ttk.Button(row, text="Retry", style="Main.TButton", command=self.start)
        self.retry_button.pack(side="left")
        ttk.Button(row, text="Quit", command=self.destroy).pack(side="left", padx=self.px(8))
        self.retry_button.focus_set()

    def _ready(self, bundle, health):
        self._clear()
        self.bundle = bundle
        self.pages = self.make_pages(bundle, health)
        self.predict = self.pages["predict"]
        self.status = "ready"
        self.status_label.configure(text="✓ Files verified\nOffline, read-only", foreground=BRAND)
        for b in self.nav.values():
            b.state(["!disabled"])
        self.show("overview")

    def make_pages(self, bundle, health):
        return {"overview": overview(self.area, self, bundle), "predict": PredictPage(self.area, self),
                "experiment": experiment(self.area, self, bundle), "health": health_page(self.area, self, health)}

    def show(self, name):
        if self.status != "ready":
            return
        for page in self.pages.values():
            page.pack_forget()
        self.pages[name].pack(fill="both", expand=True)
        for n, b in self.nav.items():
            b.state(["selected" if n == name else "!selected"])
        self.current = name

    def _unexpected(self, kind, error, trace):
        self.error_bar.configure(text=f"Unexpected problem: {kind.__name__}: {error}. Restart FreshStock; "
                                      "no frozen files were changed.")
        self.error_bar.pack(side="bottom", fill="x", before=self.area)

    def destroy(self):
        if self._poll:
            self.after_cancel(self._poll)
            self._poll = None
        super().destroy()


# ---------- building blocks ----------

def wrap(app, label):
    label.bind("<Configure>", lambda e: label.configure(wraplength=max(app.px(60), e.width - app.px(4))))
    return label


def heading(app, parent, title, text):
    ttk.Label(parent, text=title, style="Title.TLabel").pack(anchor="w", pady=(app.px(12), 0))
    wrap(app, ttk.Label(parent, text=text, style="Muted.TLabel")).pack(fill="x", pady=(app.px(2), app.px(8)))


def card(app, parent, title, **grid):
    frame = ttk.Frame(parent, style="Card.TFrame", padding=app.px(16))
    if grid:
        frame.grid(sticky="nsew", **grid)
    else:
        frame.pack(fill="x", pady=app.px(8))
    if title:
        wrap(app, ttk.Label(frame, text=title, style="Card.Head.TLabel")).pack(fill="x", pady=(0, app.px(8)))
    return frame


def para(app, parent, text, style="Card.TLabel"):
    wrap(app, ttk.Label(parent, text=text, style=style)).pack(fill="x", pady=app.px(2))


def columns_of(app, parent, n, style="TFrame", weights=None):
    grid = ttk.Frame(parent, style=style)
    grid.pack(fill="x", pady=0 if style != "TFrame" else app.px(8))
    for i in range(n):
        grid.columnconfigure(i, weight=(weights or [1] * n)[i], uniform="column")
    return grid


def gaps(app, i):
    return (app.px(0 if i == 0 else 16), 0)


def tile(app, parent, column, number, text, color=None):
    """One fact: a big number and what it counts."""
    frame = ttk.Frame(parent, style="Card.TFrame", padding=(app.px(16), app.px(12)))
    frame.grid(row=0, column=column, sticky="nsew", padx=gaps(app, column))
    value = ttk.Label(frame, text=number, style="Card.Number.TLabel")
    value.pack(side="left")
    if color:
        tk.Frame(frame, bg=color, width=app.px(8), height=app.px(30)).pack(side="left", padx=(app.px(12), 0))
    wrap(app, ttk.Label(frame, text=text, style="Card.TLabel")).pack(side="left", fill="x", expand=True,
                                                                     padx=(app.px(12), 0))
    return value


def table(app, parent, columns, rows=(), first=200, width=88, height=None, anchor="e"):
    tree = ttk.Treeview(parent, columns=list(columns), show="headings", height=height or len(rows),
                        selectmode="browse")
    for i, (key, text) in enumerate(columns.items()):
        side = anchor if i else "w"
        tree.heading(key, text=text, anchor=side)
        tree.column(key, width=app.px(width if i else first), minwidth=app.px(30), anchor=side)
    for row in rows:
        tree.insert("", "end", values=row)
    return tree


class Detail(ttk.Frame):
    """Section that opens and closes with the mouse, Space or Enter. Put content in .body."""

    def __init__(self, app, parent, title):
        super().__init__(parent)
        self.title = title
        self.pack(fill="x", pady=(app.px(4), 0))
        self.button = ttk.Button(self, text=f"▸  {title}", style="Detail.TButton", command=self.toggle)
        self.button.pack(anchor="w")
        self.body = ttk.Frame(self, style="Card.TFrame", padding=app.px(16))

    def toggle(self):
        is_open = bool(self.body.winfo_manager())
        self.button.configure(text=f"{'▸' if is_open else '▾'}  {self.title}")
        if is_open:
            self.body.pack_forget()
        else:
            self.body.pack(fill="x", pady=(0, self.winfo_toplevel().px(8)))


# ---------- Overview ----------

def overview(parent, app, bundle):
    page = Scroll(parent, app)
    body = page.body
    top = ttk.Frame(body)
    top.pack(fill="x", pady=(app.px(12), app.px(4)))
    page.open_predict = ttk.Button(top, text="Open Predict  →", style="Main.TButton",
                                   command=lambda: app.show("predict"))
    page.open_predict.pack(side="right")
    ttk.Label(top, text="Which products should be checked first?", style="Title.TLabel").pack(anchor="w")
    wrap(app, ttk.Label(top, style="Muted.TLabel", text="Historical replay of 11–25 June 2024. No live inventory, and "
                        "nothing is ordered.")).pack(fill="x")
    grid = columns_of(app, body, 3)
    tile(app, grid, 0, "100", "store-product series ranked each day")
    tile(app, grid, 1, "15", "test days to replay")
    tile(app, grid, 2, "2", "trained models, plus one simple rule")
    c = card(app, body, "How a forecast is made")
    charts.Pipeline(c, app).pack(fill="x")
    c = card(app, body, "Split by date: the future stays hidden")
    parts = [(SPLIT_TITLE[n], rows, f"{date(a)} – {date(b)}") for n, (a, b, rows, _) in data.SPLITS.items()]
    page.split = charts.SplitBar(c, app, parts)
    page.split.pack(fill="x")

    m = bundle.manifest
    d1 = Detail(app, body, "The question, data source and license")
    para(app, d1.body, "Using information available by the end of day t−1, will this series record at least one "
         "stockout hour between 06:00 and 22:00 on day t? Yes (1) or no (0): supervised binary classification. Zero "
         "sales alone cannot answer it, because an empty shelf and no demand look the same.")
    para(app, d1.body, "FreshRetailNet-50K, Dingdong-Inc and Wang et al. (2025), CC BY 4.0. Sales are normalized and "
         "IDs are encoded by the publisher.")
    rows = [("Public training file", "4,500,000 rows, 50,000 series, 90 days (2024-03-28 to 2024-06-25)"),
            ("Fixed sample", "100 series chosen by hash, not by outcome: 9,000 rows"),
            ("Labeled examples", "8,300 after removing 700 warm-up rows (inputs need 7 earlier days)")]
    rows += [(SPLIT_TITLE[n], f"{r:,} rows, {p:,} with a stockout ({p / r:.1%}), {a} to {b}")
             for n, (a, b, r, p) in data.SPLITS.items()]
    table(app, d1.body, {"part": "Step", "value": "Content"}, rows, first=170, width=520, anchor="w").pack(fill="x")
    d2 = Detail(app, body, "Limits")
    for text in LIMITS:
        para(app, d2.body, "•  " + text)
    d3 = Detail(app, body, "File checks and versions")
    libraries = ", ".join(f"{k} {v}" for k, v in m["environment"].items()
                          if k in ("scikit-learn", "numpy", "scipy", "pandas", "joblib"))
    for text in (f"✓ {len(pf.EXPECTED)} of {len(pf.EXPECTED)} data and model files match their recorded SHA-256.",
                 f"✓ Run {m['run_id']}, protocol {m['protocol']}, seed {m['seed']}, code {m['code_commit'][:12]}.",
                 "✓ Saved predictions: 1,500 validation and 1,500 test rows with valid scores, flags and outcomes.",
                 f"✓ Required libraries: {libraries}.",
                 f"✓ Models were frozen before the test was scored once ({bundle.evaluated_utc[:10]} UTC)."):
        para(app, d3.body, text)
    para(app, d3.body, "Fingerprints detect damage or mixed-up files. They do not prove who built the package.",
         "Card.Muted.TLabel")
    page.details = [d1, d2, d3]
    return page


def date(text):
    day = data.parse_day(text)
    return f"{day.day} {day:%b}"


# ---------- Experiment ----------

def experiment(parent, app, bundle):
    rows = pf.metric_rows(bundle)
    page = Scroll(parent, app)
    body = page.body
    heading(app, body, "Two models against a simple rule", "Final test, 11–25 June 2024. n = 1,500 series-days, 682 "
            "with a stockout. Scored once, after everything was frozen.")
    by = {r["name"]: r for r in rows}

    def bars(key):
        return [(SHORT[n], SERIES[n], by[n][key]) for n in ex.ESTIMATORS]
    grid = columns_of(app, body, 2)
    c = card(app, grid, "Ranking and recall", row=0, column=0)
    page.ap = charts.Compare(c, app, [("Average precision (ranking)", bars("average_precision")),
                                      ("Recall (stockouts found)", bars("recall"))], 1.0, "{:.4f}")
    page.ap.pack(fill="x")
    para(app, c, "Scale 0 to 1. Average precision differs by 0.003569 between the models, not significance-tested.",
         "Card.Muted.TLabel")
    c = card(app, grid, "Workload: misses against false alerts", row=0, column=1, padx=gaps(app, 1))
    charts.Compare(c, app, [("Missed stockouts (of 682)", bars("fn")), ("False alerts (of 818)", bars("fp"))], 818,
                   "{:.0f}").pack(fill="x")
    grid = columns_of(app, body, 3)
    for i, (title, text) in enumerate((
            ("Finds the most: Random Forest", f"Recall {by['forest']['recall']:.4f}, and the most false alerts."),
            ("In between: Logistic Regression", f"Recall {by['logistic']['recall']:.4f}, F1 {by['logistic']['f1']:.4f}."
             ),
            ("Most accurate: the rule", f"Accuracy {by['persistence']['accuracy']:.4f}. It also has the best precision "
             "and balanced accuracy."))):
        para(app, card(app, grid, title, row=0, column=i, padx=gaps(app, i)), text)
    wrap(app, ttk.Label(body, style="Good.TLabel", text="No universal winner: the choice depends on what a miss and a "
                        "false alert cost, and that was not measured.")).pack(fill="x", pady=app.px(4))
    d1 = Detail(app, body, "All six measures")
    table(app, d1.body, {"name": "Method", **METRICS},
          [[r["title"]] + [f"{r[k]:.4f}" for k in METRICS] for r in rows], first=240).pack(fill="x")
    para(app, d1.body, "Flagging every series would already give F1 0.6251.", "Card.Muted.TLabel")
    d2 = Detail(app, body, "Confusion matrices (rows: actual, columns: predicted)")
    grid = columns_of(app, d2.body, 3, "Plain.TFrame")
    for i, r in enumerate(rows):
        charts.Confusion(grid, app, SHORT[r["name"]], SERIES[r["name"]], r).grid(row=0, column=i, sticky="ew")
    d3 = Detail(app, body, "Models, inputs and thresholds")
    for text in (
            "Inputs (11 encoded columns): sales yesterday, stockout hours yesterday, mean sales of the last 7 days, "
            "number of stockout days in the last 7 days, and the weekday. IDs only group rows; target-day values are "
            "never inputs.",
            "Split by date: train, then validation, then test. The scaler learned from train only.",
            "Logistic Regression: L2 penalty, C = 1, lbfgs, standardized numeric inputs. Alert when score ≥ 0.4.",
            "Random Forest: 100 trees, maximum depth 8, at least 5 rows per leaf, Gini, seed 111. Alert when score "
            "≥ 0.3.",
            "Persistence rule: alert if the series had a stockout yesterday. It is a baseline, not a trained model.",
            "Thresholds come from validation only (best F1, then precision, then the higher value)."):
        para(app, d3.body, "•  " + text)
    page.details = [d1, d2, d3]
    return page


# ---------- Predict ----------

def score_bar(score):
    """Text bar; the exact score stands beside it."""
    return "▄" * round(float(f"{score:.4f}") * BAR)


class PredictPage(ttk.Frame):
    KEYS = {"rank": "#", "city_id": "City", "store_id": "Store", "product_id": "Product"}
    COLUMNS = {**KEYS, "score": "Risk score", "bar": "", "flag": "Flag", "logistic_score": "LR score",
               "logistic_flag": "LR", "forest_score": "RF score", "forest_flag": "RF", "persistence_flag": "Rule",
               "observed": "Observed", "result": "Result", "logistic_result": "LR result",
               "forest_result": "RF result", "persistence_result": "Rule result"}
    WIDTH = {"rank": 32, "bar": 236, "observed": 96, "result": 110, "logistic_result": 112, "forest_result": 112,
             "persistence_result": 112, "score": 80, "flag": 80, "logistic_flag": 80, "forest_flag": 80,
             "persistence_flag": 80, "logistic_score": 72, "forest_score": 72}
    WORDS = ("flag", "logistic_flag", "forest_flag", "persistence_flag", "observed", "result",
             "logistic_result", "forest_result", "persistence_result")

    def __init__(self, parent, app):
        super().__init__(parent, padding=(app.px(24), app.px(8), app.px(24), app.px(12)))
        self.app, self.bundle, self.rows, self.saved = app, app.bundle, {}, None
        self.session = pf.Session(lambda day, sort: pf.replay_day(self.bundle, day, sort))
        heading(app, self, "Forecast first, then check", "Scores use only data up to the day before. Reveal shows what "
                "was recorded.")
        bar = ttk.Frame(self)
        bar.pack(fill="x")
        self.day, self.sort = tk.StringVar(value="2024-06-18"), tk.StringVar(value=SHORT["logistic"])
        self.compare = tk.BooleanVar(value=False)
        gap = (app.px(6), app.px(12))
        ttk.Label(bar, text="Date").pack(side="left")
        self.day_box = ttk.Combobox(bar, textvariable=self.day, values=self.bundle.days, width=12)
        self.day_box.pack(side="left", padx=gap)
        self.day_box.bind("<Return>", lambda e: self.generate())
        ttk.Label(bar, text="Rank by").pack(side="left")
        ttk.Combobox(bar, textvariable=self.sort, values=[SHORT[n] for n in ex.MODELS], state="readonly",
                     width=18).pack(side="left", padx=gap)
        self.generate_button = ttk.Button(bar, text="Generate forecast", command=self.generate, style="Main.TButton")
        self.reveal_button = ttk.Button(bar, text="Reveal outcome", command=self.reveal)
        self.export_button = ttk.Button(bar, text="Export report", command=self.export)
        for b in (self.generate_button, self.reveal_button, self.export_button):
            b.pack(side="left", padx=(0, app.px(8)))
        self.message = wrap(app, ttk.Label(self, style="Muted.TLabel"))
        self.message.pack(fill="x", pady=(app.px(10), app.px(6)))
        self.open_button = ttk.Button(self, text="Open report", command=lambda: os.startfile(self.saved))
        self.tiles = columns_of(app, self, 3)
        self.workload = {n: tile(app, self.tiles, i, "", f"alerts of 100 by {SHORT[n]}", SERIES[n])
                         for i, n in enumerate(ex.ESTIMATORS)}
        self.outcomes = charts.Stacked(self, app)
        self.list_bar = ttk.Frame(self)
        self.caption = ttk.Label(self.list_bar, style="Muted.TLabel")
        self.caption.pack(side="left")
        self.compare_box = ttk.Checkbutton(self.list_bar, text="Compare all methods", variable=self.compare,
                                           command=self.refresh)
        self.compare_box.pack(side="right")
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
        self.day.trace_add("write", lambda *a: self.date_changed())
        self.sort.trace_add("write", lambda *a: self.guarded(lambda: self.session.change_sort(self.sort_name)))
        self.refresh()

    @property
    def sort_name(self):
        return next(n for n in ex.MODELS if SHORT[n] == self.sort.get())

    def say(self, text, style="Muted.TLabel"):
        self.message.configure(text=text, style=style)

    def guarded(self, action):
        try:
            action()
        except Exception as error:  # never leave earlier rows on the page after a failure
            self.session.change_date()
            self.refresh()
            reason = error if isinstance(error, data.DataError) else f"{type(error).__name__}: {error}"
            self.say(f"Stopped: {reason}", "Danger.TLabel")
        else:
            self.refresh()

    def generate(self):
        self.guarded(lambda: self.session.generate(self.day.get().strip(), self.sort_name))

    def reveal(self):
        self.guarded(self.session.reveal)

    def date_changed(self):
        self.session.change_date()
        self.refresh()

    def columns(self, revealed):
        if self.compare.get():
            flags = [] if revealed else ["persistence_flag"]
            scores = ["logistic_score", "forest_score"] if revealed else ["logistic_score", "logistic_flag",
                                                                          "forest_score", "forest_flag"]
            results = ["observed"] + [f"{e}_result" for e in ex.ESTIMATORS] if revealed else []
            return [*self.KEYS, *scores, *flags, *results]
        return [*self.KEYS, "score", "bar", *(["observed", "result"] if revealed else ["flag"])]

    def refresh(self):
        s, revealed = self.session, self.session.state == "revealed"
        chosen = next((self.series(self.rows[i]) for i in self.table.selection() if i in self.rows), None)
        self.reveal_button.state(["!disabled" if s.can_reveal else "disabled"])
        self.export_button.state(["!disabled" if s.can_export else "disabled"])
        for widget in (self.open_button, self.tiles, self.outcomes):
            widget.pack_forget()
        self.table.delete(*self.table.get_children())
        self.outcomes.show({}, {})
        self.table.configure(displaycolumns=self.columns(revealed))
        self.fit_columns()
        self.rows = {}
        if s.state == "select":
            days = self.bundle.days
            self.say(f"Choose a date from {days[0]} to {days[-1]}, then generate the forecast.")
            self.caption.configure(text="")
            self.show_detail()
            return
        view = pf.revealed_view(s.result) if revealed else pf.forecast_view(s.result)
        model = view["sort"]
        for row in view["rows"]:
            score = row[f"{model}_score"]
            flags = {f"{e}_flag": "▲ Alert" if row[f"{e}_flag"] else "–  Clear" for e in ex.ESTIMATORS}
            shown = {**row, **{f"{m}_score": f"{row[f'{m}_score']:.4f}" for m in ex.MODELS}, **flags,
                     "score": f"{score:.4f}", "bar": score_bar(score),
                     "flag": flags[f"{model}_flag"]}
            if revealed:
                shown["observed"] = "Stockout" if row["y_true"] else "None"
                shown.update({f"{e}_result": row["result"][e] for e in ex.ESTIMATORS})
                shown["result"] = row["result"][model]
            alert = row[f"{model}_flag"] and not self.compare.get()
            item = self.table.insert("", "end", values=[shown.get(c, "") for c in self.COLUMNS],
                                     tags=("alert",) if alert else ())
            self.rows[item] = row
            if self.series(row) == chosen:
                self.table.selection_set(item)
                self.table.see(item)
        self.caption.configure(text=f"Ranked by {SHORT[model]}, alert from {view['thresholds'][model]}")
        gap = (0, self.app.px(8))
        if revealed:
            self.outcomes.show(SHORT, {n: tuple(c[k] for k, _, _ in charts.OUTCOMES)
                                       for n, c in view["summary"].items()})
            self.outcomes.pack(fill="x", pady=gap, after=self.message)
            self.say(f"Recorded on {view['day']}: {view['positives']} of {view['total']} series had a stockout.",
                     "Head.TLabel")
        else:
            for n, value in self.workload.items():
                value.configure(text=str(view["alerts"][n]))
            self.tiles.pack(fill="x", pady=gap, after=self.message)
            self.say(f"Forecast for {view['day']}. The score ranks risk; it is not a percent chance.")
        self.show_detail()

    def fit_columns(self):
        """ttk keeps old widths when columns appear, which would push the outcome columns off the page."""
        base = {c: self.WIDTH.get(c, 60) for c in self.table["displaycolumns"]}
        room = self.table.winfo_width() - self.app.px(4)
        for c, width in base.items():
            self.table.column(c, width=int(room * width / sum(base.values())))

    @staticmethod
    def series(row):
        return tuple(row[k] for k in data.KEY)

    def show_detail(self):
        for child in self.detail.winfo_children():
            child.destroy()
        chosen = self.table.selection()
        if not chosen or chosen[0] not in self.rows:
            hint = "Select a row to see what the models used." if self.rows else ""
            ttk.Label(self.detail, text=hint, style="Muted.TLabel").pack(anchor="w")
            return
        row = self.rows[chosen[0]]
        ttk.Label(self.detail, style="Muted.TLabel", text=f"City {row['city_id']}, store {row['store_id']}, product "
                  f"{row['product_id']}: inputs up to {pf.day_before(row['dt'])}").pack(anchor="w")
        facts = [("Sales yesterday", f"{row['sales_lag1']:.2f}"),
                 ("Stockout hours", f"{row['stockout_hours_lag1']:.0f}"),
                 ("Mean sales, 7 days", f"{row['sales_mean7']:.2f}"),
                 ("Stockout days, 7", f"{row['stockout_days7']:.0f}"), ("Weekday", WEEKDAYS[row["weekday"]]),
                 ("LR, alert from 0.4", f"{row['logistic_score']:.4f}"),
                 ("RF, alert from 0.3", f"{row['forest_score']:.4f}"),
                 ("Rule", "Alert" if row["persistence_flag"] else "Clear")]
        if self.session.state == "revealed":
            facts.append(("Observed", "Stockout" if row["y_true"] else "None"))
        line = ttk.Frame(self.detail)
        line.pack(fill="x")
        for i, (name, value) in enumerate(facts):
            cell = ttk.Frame(line)
            cell.grid(row=0, column=i, sticky="w")
            line.columnconfigure(i, weight=1)
            ttk.Label(cell, text=value, font=(FONT, 11, "bold")).pack(anchor="w")
            ttk.Label(cell, text=name, style="Muted.TLabel", font=(FONT, 8)).pack(anchor="w")

    def export(self):
        try:
            self.saved = pf.export_report(self.bundle, self.session.result, self.app.reports_dir)
        except OSError as error:
            self.say(f"The report was not saved: {error}", "Danger.TLabel")
            return
        self.say(f"Saved: {self.saved}", "Good.TLabel")
        self.open_button.pack(anchor="w", pady=(0, self.app.px(8)), after=self.message)


# ---------- Model health ----------

MEASURES = {"alerts": "Alerts", "fn": "Missed stockouts", "fp": "False alerts"}


def health_page(parent, app, h):
    page = Scroll(parent, app)
    body = page.body
    heading(app, body, "Model health, looking back", "Saved predictions for 11–25 June 2024. This page does not "
            "monitor live data and does not prove drift.")
    grid = columns_of(app, body, 2, weights=[5, 7])
    c = card(app, grid, "Share of series-days with a stockout", row=0, column=0)
    charts.Columns(c, app, [SPLIT_TITLE[p["split"]] for p in h["prevalence"]],
                   [f"{p['positives']:,} of {p['rows']:,}" for p in h["prevalence"]],
                   [p["rate"] for p in h["prevalence"]], 0.5, "{:.1%}").pack(fill="x")
    para(app, c, "+9.8 points from validation to test: thresholds came from a calmer period.", "Card.Good.TLabel")
    c = card(app, grid, "Average test day (100 series)", row=0, column=1, padx=gaps(app, 1))
    work = h["workload"]
    charts.Compare(c, app, [(title, [(SHORT[n], SERIES[n], w[key]) for n, w in work.items()])
                            for title, key in (("Alerts", "alerts_per_day"), ("Missed stockouts", "misses_per_day"))],
                   100, "{:.1f}").pack(fill="x")
    c = card(app, body, None)
    bar = ttk.Frame(c, style="Plain.TFrame")
    bar.pack(fill="x", pady=(0, app.px(8)))
    ttk.Label(bar, text="Each test day", style="Card.Head.TLabel").pack(side="left", padx=(0, app.px(16)))
    daily = h["daily"]["test"]
    page.daily = charts.Lines(c, app, [d["dt"] for d in daily])
    page.daily.pack(fill="x")
    page.choice = tk.StringVar(app, value="alerts")

    def redraw():
        key = page.choice.get()
        series = [{"name": SHORT[n], "color": SERIES[n], "values": [d["estimators"][n][key] for d in daily]}
                  for n in ex.ESTIMATORS]
        if key == "alerts":
            series.insert(0, {"name": "Recorded stockouts", "color": INK, "values": [d["positives"] for d in daily]})
        page.daily.show(series)
    page.measures = {}
    for key, title in MEASURES.items():
        page.measures[key] = ttk.Radiobutton(bar, text=title, value=key, variable=page.choice, command=redraw,
                                             style="Choice.Toolbutton")
        page.measures[key].pack(side="left", padx=(0, app.px(6)))
    redraw()

    d1 = Detail(app, body, "Every test day in numbers")
    columns = {"dt": "Date", "positives": "Recorded"}
    for n, short in (("logistic", "LR"), ("forest", "RF"), ("persistence", "Rule")):
        columns.update({f"{n}_alerts": f"{short} alerts", f"{n}_fn": f"{short} miss", f"{n}_fp": f"{short} false"})
    table(app, d1.body, columns, [[d["dt"], d["positives"]] + [d["estimators"][n][k] for n in ex.ESTIMATORS
                                                               for k in ("alerts", "fn", "fp")] for d in daily],
          first=90, width=68).pack(fill="x")
    table(app, d1.body, {"name": "Average day", "alerts": "Alerts", "range": "Min–max", "misses": "Misses",
                         "false": "False alerts"},
          [[SHORT[n], f"{w['alerts_per_day']:.1f}", f"{w['alerts_min']}–{w['alerts_max']}",
            f"{w['misses_per_day']:.1f}", f"{w['false_alerts_per_day']:.1f}"] for n, w in work.items()],
          width=110).pack(fill="x", pady=(app.px(12), 0))
    d2 = Detail(app, body, "What the models lean on")
    grid = columns_of(app, d2.body, 2, "Plain.TFrame")
    for i, (n, title) in enumerate((("logistic", "Logistic Regression: coefficient"),
                                    ("forest", "Random Forest: impurity importance"))):
        pairs = h["explain"][n]
        charts.HBars(grid, app, title, [FEATURE_TEXT[p["feature"]] for p in pairs], [p["value"] for p in pairs],
                     SERIES[n]).grid(row=0, column=i, sticky="ew")
    para(app, d2.body, "Whole-model associations, not causes, and not an explanation of one prediction. Inputs are "
         "correlated, so single values can mislead. Logistic coefficients are for standardized numeric inputs; "
         "weekday inputs are 0 or 1.")
    d3 = Detail(app, body, "Data checks and file fingerprints")
    q = h["quality"]
    para(app, d3.body, f"Cohort: {q['rows']:,} rows, {q['series']} series, {q['first_day']} to {q['last_day']}. "
         f"Missing values {q['missing']}, duplicate rows {q['duplicates']}, out-of-range values {q['out_of_range']}, "
         f"missing days {q['gaps']}. Data cutoff {h['cutoff']}.")
    para(app, d3.body, f"Source: {h['source_rows']:,} rows, {h['source_series']:,} series, SHA-256 "
         f"{h['source_sha256']}.")
    digests = [(Path(name).name, digest) for name, digest in h["digests"].items()]
    table(app, d3.body, {"file": "Verified file", "sha": "SHA-256"}, digests, first=190, width=540,
          anchor="w").pack(fill="x", pady=(app.px(6), 0))
    page.details = [d1, d2, d3]
    return page


def use_screen_scale():
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except OSError:
            pass  # the host already chose a DPI mode


def main(bundle_root, reports_dir):
    use_screen_scale()
    App(bundle_root, reports_dir).mainloop()
