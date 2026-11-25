"""M1: the four-view desktop app, driven through real Tk widgets."""
import gc
import itertools
import shutil
import tempfile
import threading
import time
import tkinter as tk
import unittest
import weakref
from pathlib import Path
from tkinter import font as tkfont
from unittest import mock

from freshstock import charts, data, demo, gui, portfolio as pf

HAS_BUNDLE = all((data.ROOT / name).is_file() for name in pf.EXPECTED)
try:
    tk.Tk().destroy()
    HAS_TK = True
except tk.TclError:
    HAS_TK = False


def texts(widget):
    """All text a user could read inside a widget, including table cells and canvas labels."""
    out = []
    if isinstance(widget, tk.Canvas):
        out += [widget.itemcget(i, "text") for i in widget.find_all() if widget.type(i) == "text"]
    if isinstance(widget, gui.ttk.Treeview):
        shown = widget["displaycolumns"]
        shown = widget["columns"] if shown in ("#all", ("#all",)) else shown
        out += [widget.heading(c, "text") for c in shown]
        out += [str(widget.set(i, c)) for i in widget.get_children() for c in shown]
    elif "text" in widget.keys():
        out.append(str(widget.cget("text")))
    for child in widget.winfo_children():
        if child.winfo_manager():  # skip widgets that are not placed on the page
            out += texts(child)
    return out


def contrast(a, b):
    def luminance(color):
        parts = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        r, g, b = [p / 12.92 if p <= 0.04045 else ((p + 0.055) / 1.055) ** 2.4 for p in parts]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


class ColorTests(unittest.TestCase):
    def test_text_colors_have_enough_contrast(self):
        for fore in (gui.INK, gui.MUTED, gui.BRAND, gui.WARNING, gui.DANGER):
            for back in (gui.CANVAS, gui.SURFACE):
                with self.subTest(fore=fore, back=back):
                    self.assertGreaterEqual(contrast(fore, back), 4.5)
        self.assertGreaterEqual(contrast("#FFFFFF", gui.BRAND), 4.5)
        self.assertGreaterEqual(contrast(gui.DISABLED, gui.DISABLED_BACK), 3.0)

    def test_chart_colors_have_enough_contrast(self):
        for color in gui.SERIES.values():
            self.assertGreaterEqual(contrast(color, gui.SURFACE), 3.0)

    def test_outcome_segment_labels_have_enough_contrast(self):
        for _, fill, ink in charts.OUTCOMES:
            self.assertGreaterEqual(contrast(ink, fill), 4.5)


@unittest.skipUnless(HAS_BUNDLE and HAS_TK, "needs the frozen run and a display")
class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = pf.load_verified_bundle(data.ROOT)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.reports = Path(tmp.name) / "reports"
        self.load = lambda root: self.bundle
        self.app = None

    def open(self, load=None):
        self.app = gui.App(data.ROOT, self.reports, load or self.load)
        self.addCleanup(self.app.destroy)
        self.settle()
        return self.app

    def settle(self):
        end = time.monotonic() + 20
        while self.app.status == "checking" and time.monotonic() < end:
            self.app.update()
            time.sleep(0.01)
        self.app.update()

    def forecast(self, day="2024-06-18"):
        app = self.open()
        app.show("predict")
        app.predict.day.set(day)
        app.predict.generate_button.invoke()
        app.update()
        return app.predict

    # ---------- start ----------

    def test_starts_checking_then_ready_with_four_views(self):
        self.app = gui.App(data.ROOT, self.reports, self.load)
        self.addCleanup(self.app.destroy)
        self.assertEqual(self.app.status, "checking")
        self.settle()
        self.assertEqual(self.app.status, "ready")
        self.assertEqual(list(self.app.nav), ["overview", "predict", "experiment", "health"])

    # Tcl aborts the process when a closed window is freed inside another thread.

    def test_loader_thread_does_not_keep_a_closed_window_alive(self):
        go = threading.Event()
        self.addCleanup(go.set)
        bundle = self.bundle
        app = gui.App(data.ROOT, self.reports, lambda root: go.wait(20) and bundle)
        app.destroy()
        closed = weakref.ref(app)
        del app
        gc.collect()
        self.assertIsNone(closed())  # the loader is still running here

    def test_closed_window_is_freed_before_the_next_loader_starts(self):
        gc.disable()
        self.addCleanup(gc.enable)
        bundle = self.bundle
        app = gui.App(data.ROOT, self.reports, lambda root: bundle)
        app.destroy()
        closed, freed = weakref.ref(app), []
        del app
        self.open(lambda root: freed.append(closed() is None) or bundle)
        self.assertEqual(freed, [True])

    def test_damaged_bundle_blocks_and_retry_recovers(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in pf.EXPECTED:
                (Path(tmp) / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(data.ROOT / name, Path(tmp) / name)
            broken = Path(tmp) / "runs" / pf.RUN_ID / "predictions.csv"
            broken.write_text("x")
            self.app = gui.App(tmp, self.reports)
            self.addCleanup(self.app.destroy)
            self.settle()
            self.assertEqual(self.app.status, "blocked")
            self.assertIn("predictions.csv was changed", " ".join(texts(self.app)))
            self.assertTrue(all(b.instate(["disabled"]) for b in self.app.nav.values()))
            shutil.copyfile(data.ROOT / "runs" / pf.RUN_ID / "predictions.csv", broken)
            self.app.retry_button.invoke()
            self.assertEqual(self.app.status, "checking")
            self.settle()
            self.assertEqual(self.app.status, "ready")

    def test_unexpected_start_error_is_shown_not_hidden(self):
        def load(root):
            raise OSError("disk unreadable")
        self.open(load)
        self.assertEqual(self.app.status, "blocked")
        self.assertIn("disk unreadable", " ".join(texts(self.app)))

    def test_unexpected_page_error_is_shown_not_hidden(self):
        app = self.open()
        app.after(0, lambda: 1 / 0)
        app.update()
        app.update()
        self.assertIn("division by zero", " ".join(texts(app)))

    # ---------- predict ----------

    def rows(self, *columns):
        table = self.app.predict.table
        return [tuple(table.set(i, c) for c in columns) for i in table.get_children()]

    def shown_columns(self):
        return tuple(self.app.predict.table["displaycolumns"])

    def test_only_generate_is_enabled_at_first(self):
        app = self.open()
        self.assertTrue(app.predict.generate_button.instate(["!disabled"]))
        self.assertTrue(app.predict.reveal_button.instate(["disabled"]))
        self.assertTrue(app.predict.export_button.instate(["disabled"]))
        self.assertEqual(app.predict.day.get(), "2024-06-18")
        self.assertEqual(app.predict.table.get_children(), ())

    def test_forecast_shows_workload_and_100_rows_and_no_outcome_text(self):
        page = self.forecast()
        self.assertEqual(len(page.table.get_children()), 100)
        shown = " ".join(texts(page))
        for word in ("Miss", "False alert", "Correct", "Observed", "45 of 100", "TP ", "FN "):
            self.assertNotIn(word, shown)
        self.assertEqual([t.cget("text") for t in page.workload.values()], ["54", "72", "45"])
        self.assertIn("ranks risk", shown)
        self.assertEqual(self.rows("rank", "city_id", "store_id", "product_id", "score", "flag")[0],
                         ("1", "3", "855", "188", "0.8313", "▲ Alert"))
        self.assertTrue(page.reveal_button.instate(["!disabled"]))
        self.assertTrue(page.export_button.instate(["disabled"]))

    def test_default_list_shows_one_flag_column_and_follows_the_ranking_model(self):
        page = self.forecast()
        self.assertEqual(self.shown_columns(), ("rank", "city_id", "store_id", "product_id", "score", "bar", "flag"))
        self.assertEqual([f for _, f in self.rows("rank", "flag")].count("▲ Alert"), 54)
        page.sort.set("Random Forest")
        self.app.update()
        self.assertEqual([f for _, f in self.rows("rank", "flag")].count("▲ Alert"), 72)
        self.assertEqual(self.rows("score"), self.rows("forest_score"))

    def test_score_bar_length_follows_the_score(self):
        self.forecast()
        for score, bar in self.rows("score", "bar"):
            self.assertEqual(bar, "▄" * round(float(score) * gui.BAR))

    def test_compare_shows_every_score_and_flag_and_changes_nothing_else(self):
        page = self.forecast()
        before = self.rows("rank", "city_id", "store_id", "product_id")
        page.compare_box.invoke()
        self.app.update()
        self.assertEqual(self.shown_columns(), ("rank", "city_id", "store_id", "product_id", "logistic_score",
                                                "logistic_flag", "forest_score", "forest_flag", "persistence_flag"))
        for column, alerts in (("logistic_flag", 54), ("forest_flag", 72), ("persistence_flag", 45)):
            self.assertEqual([r[0] for r in self.rows(column)].count("▲ Alert"), alerts)
        self.assertEqual(self.rows("rank", "city_id", "store_id", "product_id"), before)
        self.assertEqual(page.session.state, "forecast")
        page.compare_box.invoke()
        self.app.update()
        self.assertIn("bar", self.shown_columns())

    def test_compare_works_from_the_keyboard(self):
        page = self.forecast()
        page.compare_box.focus_force()
        self.app.update()
        page.compare_box.event_generate("<space>")
        self.app.update()
        self.assertIn("persistence_flag", self.shown_columns())

    def test_outcome_values_are_not_in_the_table_before_reveal(self):
        page = self.forecast()
        for compare in (False, True):
            for row in self.rows("observed", "result", "logistic_result", "forest_result", "persistence_result"):
                self.assertEqual(row, ("", "", "", "", ""))
            page.compare_box.invoke()
            self.app.update()

    def test_reveal_shows_counts_and_enables_export(self):
        page = self.forecast()
        page.reveal_button.invoke()
        self.app.update()
        shown = " ".join(texts(page))
        for part in ("45 of 100", "TP 30 · FN 15 · FP 24 · TN 31", "TP 36 · FN 9 · FP 36 · TN 19",
                     "TP 26 · FN 19 · FP 19 · TN 36", "Correct alert", "Miss", "False alert", "Correct clear"):
            self.assertIn(part, shown)
        self.assertEqual(page.outcomes.counts, {"logistic": (30, 15, 24, 31), "forest": (36, 9, 36, 19),
                                                "persistence": (26, 19, 19, 36)})
        self.assertTrue(page.export_button.instate(["!disabled"]))
        self.assertTrue(page.reveal_button.instate(["disabled"]))

    def test_outcome_bars_are_drawn_to_scale(self):
        page = self.forecast()
        page.reveal_button.invoke()
        self.app.update()
        widths = {}
        for item in page.outcomes.find_withtag("segment"):
            x1, _, x2, _ = page.outcomes.coords(item)
            name, kind = page.outcomes.gettags(item)[1:3]
            widths[name, int(kind)] = x2 - x1
        full = sum(widths["logistic", k] for k in range(4))
        for name, counts in page.outcomes.counts.items():
            for k, n in enumerate(counts):
                self.assertAlmostEqual(widths[name, k] / full, n / 100, delta=0.01)

    def test_after_reveal_the_list_shows_observed_and_the_ranking_models_result(self):
        page = self.forecast()
        page.reveal_button.invoke()
        self.app.update()
        self.assertEqual(self.shown_columns(),
                         ("rank", "city_id", "store_id", "product_id", "score", "bar", "observed", "result"))
        self.assertEqual(self.rows("result"), self.rows("logistic_result"))
        self.assertEqual([r[0] for r in self.rows("observed")].count("Stockout"), 45)
        page.sort.set("Random Forest")
        self.app.update()
        self.assertEqual(self.rows("result"), self.rows("forest_result"))
        page.compare_box.invoke()
        self.app.update()
        self.assertEqual(self.shown_columns()[-4:], ("observed", "logistic_result", "forest_result",
                                                     "persistence_result"))

    def test_every_shown_column_fits_in_the_table(self):
        page = self.forecast()
        for step in (lambda: None, page.compare_box.invoke, page.reveal_button.invoke, page.compare_box.invoke,
                     lambda: page.day.set("2024-06-19"), page.generate_button.invoke):
            step()
            self.app.update()
            widths = sum(page.table.column(c, "width") for c in page.table["displaycolumns"])
            self.assertLessEqual(widths, page.table.winfo_width())
            self.assertGreater(widths, page.table.winfo_width() * 0.95)

    def test_nothing_is_clipped_at_the_smallest_window(self):
        page = self.forecast()
        app = self.app
        app.geometry("x".join(map(str, app.minsize())))
        cell, head = tkfont.Font(family=gui.FONT, size=10), tkfont.Font(family=gui.FONT, size=9, weight="bold")
        for step in (lambda: None, page.compare_box.invoke, page.reveal_button.invoke, page.compare_box.invoke):
            step()
            app.update()
            for b in (page.generate_button, page.reveal_button, page.export_button, page.compare_box):
                self.assertGreaterEqual(b.winfo_width(), b.winfo_reqwidth(), b.cget("text"))
            self.assertGreaterEqual(page.table.winfo_height(), app.px(120))
            for c in page.table["displaycolumns"]:
                need = max(cell.measure(str(page.table.set(i, c))) for i in page.table.get_children())
                need = max(need, head.measure(page.table.heading(c, "text")))
                self.assertGreaterEqual(page.table.column(c, "width"), need + app.px(8), c)

    def test_resizing_refits_the_columns(self):
        page = self.forecast()
        page.reveal_button.invoke()
        for width in (self.app.minsize()[0], self.app.minsize()[0] + 200):
            self.app.geometry(f"{width}x{self.app.minsize()[1]}")
            self.app.update()
            self.assertLessEqual(sum(page.table.column(c, "width") for c in page.table["displaycolumns"]),
                                 page.table.winfo_width())

    def test_changing_the_date_clears_everything(self):
        page = self.forecast()
        page.reveal_button.invoke()
        self.app.update()
        page.day.set("2024-06-19")
        self.app.update()
        self.assertEqual(page.table.get_children(), ())
        shown = " ".join(texts(page))
        for word in ("45 of 100", "TP 30", "54"):
            self.assertNotIn(word, shown)
        self.assertTrue(page.reveal_button.instate(["disabled"]))
        self.assertTrue(page.export_button.instate(["disabled"]))

    def test_sorting_reorders_and_keeps_the_reveal(self):
        page = self.forecast()
        page.reveal_button.invoke()
        facts = ("city_id", "store_id", "product_id", "logistic_score", "forest_score", "observed", "logistic_result",
                 "forest_result", "persistence_result")
        before = set(self.rows(*facts))
        page.sort.set("Random Forest")
        self.app.update()
        scores = [float(s) for s, in self.rows("forest_score")]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(set(self.rows(*facts)), before)
        self.assertIn("45 of 100", " ".join(texts(page)))

    def test_selected_series_shows_its_inputs_and_stays_selected(self):
        page = self.forecast()
        page.table.selection_set(page.table.get_children()[0])
        self.app.update()
        shown = " ".join(texts(page.detail))
        for part in ("2024-06-17", "16", "Tue", "0.8313", "0.8280", "from 0.4", "from 0.3"):
            self.assertIn(part, shown)
        self.assertNotIn("Observed", shown)
        page.sort.set("Random Forest")
        self.app.update()
        chosen = page.table.selection()[0]
        self.assertEqual([page.table.set(chosen, k) for k in data.KEY], ["3", "855", "188"])
        page.reveal_button.invoke()
        self.app.update()
        self.assertIn("Observed", " ".join(texts(page.detail)))

    def test_invalid_date_is_refused_without_rows(self):
        page = self.forecast("2024-07-01")
        self.assertEqual(page.table.get_children(), ())
        self.assertIn("choose a date from 2024-06-11 to 2024-06-25", page.message.cget("text"))
        self.assertNotIn("54", " ".join(texts(page)))

    def test_unexpected_replay_error_leaves_no_stale_rows(self):
        page = self.forecast()
        page.reveal_button.invoke()
        with mock.patch.object(pf, "replay_day", side_effect=OSError("disk unplugged")):
            page.generate_button.invoke()
        self.app.update()
        self.assertEqual(page.table.get_children(), ())
        self.assertIn("disk unplugged", page.message.cget("text"))
        self.assertNotIn("45 of 100", " ".join(texts(page)))
        self.assertTrue(page.export_button.instate(["disabled"]))

    def test_export_saves_the_report(self):
        page = self.forecast()
        page.compare_box.invoke()
        page.reveal_button.invoke()
        page.export_button.invoke()
        saved = self.reports / "replay-2024-06-18.html"
        self.assertEqual(saved.read_text(encoding="utf-8"),
                         demo.render_html(pf.replay_day(self.bundle, "2024-06-18"), self.bundle.metrics))
        self.assertIn("Saved", page.message.cget("text"))

    def test_export_failure_is_reported(self):
        page = self.forecast()
        page.reveal_button.invoke()
        self.reports.write_text("a file is in the way")
        page.export_button.invoke()
        self.assertIn("not saved", page.message.cget("text"))

    def test_replay_works_with_the_network_blocked(self):
        with mock.patch("socket.socket", side_effect=OSError("network blocked")):
            self.assertEqual(len(self.forecast().table.get_children()), 100)

    # ---------- other views ----------

    def all_text(self, name):
        """Visible text, then visible text with every detail section open."""
        self.app.show(name)
        self.app.update()
        page = self.app.pages[name]
        first = " ".join(texts(page))
        for d in getattr(page, "details", []):
            d.button.invoke()
        self.app.update()
        return first, " ".join(texts(page))

    def test_overview_is_short_and_keeps_the_facts_in_details(self):
        self.open()
        first, full = self.all_text("overview")
        for part in ("100", "15", "2", "5,300", "1,500", "Past 7 days", "Human review", "Historical", "No live"):
            self.assertIn(part, first)
        for part in (pf.RUN_ID, "CC BY 4.0", "06:00"):
            self.assertNotIn(part, first)
        for part in ("4,500,000", "50,000", "9,000", "8,300", "CC BY 4.0", pf.RUN_ID, "06:00", "B3 v1", "1.8.0",
                     "not calibrated"):
            self.assertIn(part, full)

    def test_overview_opens_predict(self):
        app = self.open()
        app.pages["overview"].open_predict.invoke()
        self.assertEqual(app.current, "predict")
        self.assertEqual(app.predict.session.state, "select")

    def test_split_bar_is_proportional(self):
        app = self.open()
        app.show("overview")
        app.update()
        split = app.pages["overview"].split
        widths = [split.coords(i)[2] - split.coords(i)[0] for i in split.find_withtag("segment")]
        for w, rows in zip(widths, (5300, 1500, 1500), strict=True):
            self.assertAlmostEqual(w / sum(widths), rows / 8300, delta=0.01)

    def test_experiment_shows_actual_comparison_first(self):
        self.open()
        first, full = self.all_text("experiment")
        for part in ("0.6404", "0.6439", "0.5613", "218", "100", "264", "315", "500", "247", "1,500", "682",
                     "No universal winner", "not significance-tested", "0.6593"):
            self.assertIn(part, first)
        for part in ("0.6352", "0.6206", "0.6476", "503", "464", "582", "571", "not a trained", "0.4", "0.3",
                     "Predicted 0"):
            self.assertIn(part, full)

    def test_comparison_bars_start_at_zero(self):
        app = self.open()
        app.show("experiment")
        app.update()
        chart = app.pages["experiment"].ap
        widths = [chart.coords(i)[2] - chart.coords(i)[0] for i in chart.find_withtag("bar")]
        for i, value in enumerate((0.640380, 0.643949, 0.561253, 0.680352, 0.853372, 0.612903)):  # AP, then recall
            self.assertAlmostEqual(widths[i] / widths[0], value / 0.640380, delta=0.02)

    def test_health_is_labelled_historical(self):
        self.open()
        first, full = self.all_text("health")
        for part in ("does not monitor live data", "45.3%", "35.7%", "45.5%", "+9.8", "51.9", "72.1", "44.3"):
            self.assertIn(part, first)
        for part in ("0.4659", "not causes", "SHA-256", "2024-06-16"):
            self.assertIn(part, full)

    def test_health_selector_switches_the_one_daily_chart(self):
        app = self.open()
        app.show("health")
        page = app.pages["health"]
        seen = {}
        for name, button in page.measures.items():
            button.invoke()
            app.update()
            seen[name] = {s["name"]: sum(s["values"]) for s in page.daily.series}
        self.assertEqual(seen["alerts"], {"Recorded stockouts": 682, "Logistic Regression": 779, "Random Forest": 1082,
                                          "Persistence rule": 665})
        self.assertEqual(seen["fn"], {"Logistic Regression": 218, "Random Forest": 100, "Persistence rule": 264})
        self.assertEqual(seen["fp"], {"Logistic Regression": 315, "Random Forest": 500, "Persistence rule": 247})

    def test_line_end_labels_do_not_overlap(self):
        app = self.open()
        app.show("health")
        page = app.pages["health"]
        for button in page.measures.values():
            button.invoke()
            app.update()
            boxes = sorted(page.daily.bbox(i) for i in page.daily.find_withtag("end"))
            boxes.sort(key=lambda b: b[1])
            self.assertGreaterEqual(len(boxes), 3)
            for a, b in zip(boxes, boxes[1:]):
                self.assertLessEqual(a[3], b[1] + 1)

    def test_details_open_and_close_from_the_keyboard(self):
        app = self.open()
        app.show("experiment")
        detail = app.pages["experiment"].details[0]
        app.update()
        self.assertFalse(detail.body.winfo_ismapped())
        detail.button.focus_force()
        app.update()
        detail.button.event_generate("<Return>")
        app.update()
        self.assertTrue(detail.body.winfo_ismapped())
        self.assertTrue(detail.button.cget("text").startswith("▾"))
        detail.button.event_generate("<Return>")
        app.update()
        self.assertFalse(detail.body.winfo_ismapped())

    def test_keyboard_switches_views(self):
        app = self.open()
        app.deiconify()
        app.focus_force()
        app.update()
        for key, name in zip("1234", app.nav):
            app.event_generate(f"<Control-Key-{key}>")
            app.update()
            self.assertEqual(app.current, name)

    def test_no_widget_is_wider_than_the_smallest_window(self):
        app = self.open()
        app.geometry("x".join(map(str, app.minsize())))
        for name in app.nav:
            app.show(name)
            app.update()
            for w in app.pages[name].winfo_children():
                self.assertLessEqual(w.winfo_rootx() + w.winfo_width(), app.winfo_rootx() + app.winfo_width() + 1)

    def test_no_label_is_cut_off_at_the_smallest_window(self):
        app = self.open()
        app.geometry("x".join(map(str, app.minsize())))
        app.show("predict")
        app.predict.generate_button.invoke()
        app.predict.table.selection_set(app.predict.table.get_children()[0])
        for name in app.nav:
            self.all_text(name)
            stack = [app.pages[name]]
            while stack:
                w = stack.pop()
                stack += w.winfo_children()
                if isinstance(w, gui.ttk.Label) and w.winfo_ismapped():
                    self.assertLessEqual(w.winfo_reqwidth(), w.winfo_width() + 1, (name, w.cget("text")))

    def test_chart_text_stays_inside_the_chart(self):
        app = self.open()
        app.geometry("x".join(map(str, app.minsize())))
        app.show("predict")
        app.predict.generate_button.invoke()
        app.predict.reveal_button.invoke()
        for name in app.nav:
            self.all_text(name)
            stack = [app.pages[name]]
            while stack:
                w = stack.pop()
                stack += w.winfo_children()
                if isinstance(w, charts.Chart) and w.winfo_ismapped():
                    boxes = [w.bbox(i) for i in w.find_all() if w.type(i) == "text" and w.itemcget(i, "text")]
                    for a, b in itertools.combinations(boxes, 2):
                        self.assertFalse(a[0] < b[2] - 2 and b[0] < a[2] - 2 and a[1] < b[3] - 3 and b[1] < a[3] - 3,
                                         (name, type(w).__name__, a, b))
                    for item in w.find_all():
                        if w.type(item) == "text" and w.itemcget(item, "text"):
                            x1, y1, x2, y2 = w.bbox(item)
                            self.assertTrue(x1 >= -1 and y1 >= -1 and x2 <= w.winfo_width() + 1
                                            and y2 <= w.winfo_height() + 1, (name, w.itemcget(item, "text")))


if __name__ == "__main__":
    unittest.main()
