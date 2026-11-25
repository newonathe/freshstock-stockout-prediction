"""FreshStock 2.0 desktop views, driven through real Tk widgets."""
import csv
import re
import tempfile
import threading
import time
import tkinter as tk
import unittest
from pathlib import Path
from unittest import mock

from freshstock import data as b3, gui as old
from freshstock.expanded import bundle, data, gui, metrics
from tests.test_expanded_bundle import BundleCase
from tests.test_expanded_experiment import N
from tests.test_gui import texts

try:
    tk.Tk().destroy()
    HAS_TK = True
except tk.TclError:
    HAS_TK = False


@unittest.skipUnless(HAS_TK, "needs a display")
class GuiCase(BundleCase):
    def setUp(self):
        self.reports = Path(tempfile.mkdtemp(dir=self.root)) / "reports"
        self.page_size = 7
        patch = mock.patch.object(gui, "PAGE", self.page_size)
        patch.start()
        self.addCleanup(patch.stop)

    def open(self, load=None):
        app = gui.App(self.folder, self.reports, load or (lambda root: self.loaded))
        self.addCleanup(app.destroy)
        self.until(app, lambda: app.status != "checking")
        return app

    def until(self, app, done, seconds=20):
        end = time.monotonic() + seconds
        while not done() and time.monotonic() < end:
            app.update()
            time.sleep(.005)
        app.update()
        self.assertTrue(done(), "timed out")

    def ready(self, day=None):
        app = self.open()
        app.show("predict")
        if day:
            app.predict.day.set(day)
        app.predict.generate_button.invoke()
        self.until(app, lambda: not app.predict.busy)
        self.assertEqual(app.predict.state, "forecast", app.predict.message.cget("text"))
        return app, app.predict

    def shown(self, page):
        return [page.table.item(i, "values") for i in page.table.get_children()]

    def ranks(self, page):
        return [int(str(v[0]).replace(",", "")) for v in self.shown(page)]


class StartTests(GuiCase):
    def test_four_views_and_verified_status(self):
        app = self.open()
        self.assertEqual((app.status, list(app.nav)), ("ready", ["overview", "predict", "experiment", "health"]))
        self.assertIn("2.0", app.title())
        for name in app.nav:
            app.show(name)
            app.update()
            self.assertEqual(app.current, name)

    def test_damaged_bundle_is_blocked_with_the_reason(self):
        folder = self.copy_bundle()
        (folder / "forest.joblib").write_bytes(b"x")
        app = self.open(lambda root: bundle.load(folder, self.pin))
        self.assertEqual(app.status, "blocked")
        self.assertIn("forest.joblib", app.problem)
        self.assertTrue(app.retry_button.winfo_exists())

    def test_numbers_come_from_the_bundle(self):
        app = self.open()
        s = self.loaded.metrics["summary"]
        overview = " ".join(texts(app.pages["overview"]))
        self.assertIn(f"{N}", overview)
        self.assertIn("Past 28 days", overview)
        page = " ".join(texts(app.pages["experiment"]))
        for e in metrics.ESTIMATORS:
            self.assertIn(f"{s['estimators'][e]['average_precision']:.4f}", page)
        app.pages["experiment"].details[0].toggle()
        app.update()
        rate = s["prevalence"]
        shown = " ".join(texts(app.pages["experiment"].details[0]))
        self.assertIn(f"{max(rate, 1 - rate):.4f}", shown)
        self.assertIn(f"{2 * rate / (1 + rate):.4f}", shown)
        for stale in ("682", "818", "0.6439", "0.6404", "11–25 June", "0.003569", "100 series", "June 18"):
            for p in app.pages.values():
                self.assertNotRegex(" ".join(texts(p)), rf"(?<![\d.]){re.escape(stale)}(?!\d)")

    def test_details_open_from_the_keyboard(self):
        app = self.open()
        for name in ("overview", "experiment", "health"):
            app.show(name)
            app.update()
            for d in app.pages[name].details:
                self.assertFalse(d.body.winfo_manager())
                d.button.event_generate("<Return>")
                d.button.invoke() if not d.body.winfo_manager() else None
                app.update()
                self.assertTrue(d.body.winfo_manager())


class ForecastTests(GuiCase):
    def test_forecast_runs_outside_the_event_loop_and_once(self):
        app = self.open()
        app.show("predict")
        page, go, calls = app.predict, threading.Event(), []
        self.addCleanup(go.set)
        original = bundle.forecast

        def slow(*args):
            calls.append(threading.current_thread())
            go.wait(10)
            return original(*args)
        with mock.patch.object(bundle, "forecast", slow):
            page.generate_button.invoke()
            page.generate_button.invoke()
            app.predict.day_box.event_generate("<Return>")
            page.generate()  # what the Return key calls
            for _ in range(20):
                app.update()
            self.assertTrue(page.busy)
            for b in (page.generate_button, page.reveal_button, page.export_button):
                self.assertTrue(b.instate(["disabled"]))
            go.set()
            self.until(app, lambda: not page.busy)
        self.assertEqual(len(calls), 1)
        self.assertIsNot(calls[0], threading.main_thread())
        self.assertEqual(page.state, "forecast")

    def test_stale_result_is_ignored_after_a_date_change(self):
        app = self.open()
        app.show("predict")
        page, go = app.predict, threading.Event()
        self.addCleanup(go.set)
        original = bundle.forecast

        def slow(*args):
            go.wait(10)
            return original(*args)
        with mock.patch.object(bundle, "forecast", slow):
            page.generate_button.invoke()
            page.day.set("2024-06-30")
            go.set()
            end = time.monotonic() + 1
            while time.monotonic() < end:
                app.update()
                time.sleep(.01)
        self.assertEqual((page.state, page.busy, self.shown(page)), ("select", False, []))

    def test_closing_during_a_forecast_is_safe(self):
        go = threading.Event()
        self.addCleanup(go.set)
        with mock.patch.object(bundle, "forecast", lambda *a: go.wait(10)):
            app = gui.App(self.folder, self.reports, lambda root: self.loaded)
            self.until(app, lambda: app.status == "ready")
            app.show("predict")
            app.predict.generate_button.invoke()
            app.destroy()
            go.set()
            time.sleep(.1)

    def test_error_clears_rows_and_allows_retry(self):
        app, page = self.ready()
        with mock.patch.object(bundle, "forecast", side_effect=b3.DataError("scores differ")):
            page.generate_button.invoke()
            self.until(app, lambda: not page.busy)
        self.assertEqual((page.state, self.shown(page)), ("select", []))
        self.assertIn("scores differ", page.message.cget("text"))
        page.generate_button.invoke()
        self.until(app, lambda: not page.busy)
        self.assertEqual(page.state, "forecast")

    def test_bad_date_is_refused(self):
        for day in ("2024-06-25", "2024-07-03", "nonsense"):
            app = self.open()
            app.show("predict")
            app.predict.day.set(day)
            app.predict.generate_button.invoke()
            self.until(app, lambda: not app.predict.busy)
            self.assertEqual(app.predict.state, "select")
            self.assertIn("choose a date", app.predict.message.cget("text"))

    def test_models_are_loaded_once(self):
        with mock.patch.object(bundle, "load", side_effect=AssertionError("reloaded")):
            app, page = self.ready()
            for day in self.loaded.days[1:3]:
                page.day.set(day)
                page.generate_button.invoke()
                self.until(app, lambda: not page.busy)
                self.assertEqual(page.state, "forecast")


class RankingTests(GuiCase):
    def test_pages_follow_the_global_rank(self):
        app, page = self.ready()
        seen = self.ranks(page)
        self.assertEqual(seen, list(range(1, 8)))
        self.assertTrue(page.previous_button.instate(["disabled"]))
        while page.next_button.instate(["!disabled"]):
            page.next_button.invoke()
            app.update()
            seen += self.ranks(page)
        self.assertEqual(seen, list(range(1, N + 1)))
        page.next_button.invoke()
        self.assertEqual(self.ranks(page)[-1], N)
        self.assertIn("6 of 6", page.page_label.cget("text"))

    def test_sort_change_uses_cached_scores_and_returns_to_rank_one(self):
        app, page = self.ready()
        page.next_button.invoke()
        with mock.patch.object(bundle, "forecast", side_effect=AssertionError("recomputed")):
            page.sort.set(old.SHORT["forest"])
            app.update()
        self.assertEqual(self.ranks(page), list(range(1, 8)))
        first = bundle.page_rows(self.loaded, page.forecast, "forest", 0, 7)[0]
        self.assertEqual([int(v) for v in self.shown(page)[0][1:4]], [first[k] for k in b3.KEY])

    def test_selection_is_cleared_when_the_page_changes(self):
        app, page = self.ready()
        page.table.selection_set(page.table.get_children()[2])
        app.update()
        self.assertIn("inputs up to 2024-06-25", " ".join(texts(page.detail)))
        page.next_button.invoke()
        app.update()
        self.assertEqual(page.table.selection(), ())
        self.assertNotIn("inputs up to", " ".join(texts(page.detail)))

    def test_compare_keeps_the_same_rows(self):
        app, page = self.ready()
        before = [v[:4] for v in self.shown(page)]
        page.compare.set(True)
        page.refresh()
        app.update()
        self.assertEqual([v[:4] for v in self.shown(page)], before)
        self.assertIn("RF score", texts(page.table))

    def test_find_jumps_to_the_series(self):
        app, page = self.ready()
        at = int(page.forecast.order("logistic")[N - 1])
        key = self.loaded.history.keys[at].tolist()
        page.find.set(",".join(map(str, key)))
        page.find_button.invoke()
        app.update()
        self.assertIn(N, self.ranks(page))
        self.assertEqual([int(v) for v in page.table.item(page.table.selection()[0], "values")[1:4]], key)
        for text, reason in (("99,1,2", "not in this cohort"), ("abc", "city, store, product"), ("", "city, store")):
            page.find.set(text)
            page.find_button.invoke()
            self.assertIn(reason, page.message.cget("text"))
            self.assertEqual(page.state, "forecast")

    def test_counts_are_for_the_full_cohort(self):
        app, page = self.ready()
        for e in metrics.ESTIMATORS:
            self.assertEqual(page.workload[e].cget("text"), f"{int(page.forecast.flags[e].sum()):,}")


class RevealTests(GuiCase):
    def test_no_outcome_on_screen_before_reveal(self):
        app, page = self.ready()
        for compare in (False, True):
            page.compare.set(compare)
            page.refresh()
            app.update()
            words = " ".join(texts(page))
            for outcome in ("Stockout ", "Miss", "Correct alert", "False alert", "Correct clear", "Observed"):
                self.assertNotIn(outcome, words.replace("Stockout hours", "").replace("Stockout days", ""))
            values = " ".join(str(x) for v in self.shown(page) for x in v)
            self.assertNotIn("Correct", values)
        self.assertTrue(page.export_button.instate(["disabled"]))

    def test_reveal_counts_and_review_list(self):
        app, page = self.ready("2024-06-28")
        page.reveal_button.invoke()
        app.update()
        r = page.revealed
        self.assertEqual(page.state, "revealed")
        self.assertIn(f"{r.positives:,} of {N:,} series", page.message.cget("text"))
        for e in metrics.ESTIMATORS:
            self.assertEqual(page.outcomes.counts[e], tuple(r.counts[e][k] for k in ("tp", "fn", "fp", "tn")))
        self.assertIn(f"{r.review['logistic']['hits']:,} of {r.review['logistic']['k']:,}", page.review.cget("text"))
        page.next_button.invoke()
        app.update()
        self.assertEqual(page.state, "revealed")

    def test_date_change_hides_everything_again(self):
        app, page = self.ready()
        page.reveal_button.invoke()
        app.update()
        page.day.set("2024-07-01")
        app.update()
        self.assertEqual((page.state, self.shown(page), page.revealed, page.forecast), ("select", [], None, None))
        self.assertNotIn("had a stockout", page.message.cget("text"))

    def test_reveal_failure_clears_the_page(self):
        app, page = self.ready()
        with mock.patch.object(bundle, "reveal", side_effect=RuntimeError("boom")):
            page.reveal_button.invoke()
        self.assertEqual((page.state, self.shown(page)), ("select", []))
        self.assertIn("boom", page.message.cget("text"))

    def test_reveal_without_a_forecast_does_nothing(self):
        app = self.open()
        app.show("predict")
        app.predict.reveal()
        self.assertEqual(app.predict.state, "select")

    def test_export_is_complete(self):
        app, page = self.ready()
        page.reveal_button.invoke()
        page.export_button.invoke()
        self.until(app, lambda: not page.busy)
        self.assertTrue((page.saved / "manifest.json").is_file())
        with open(page.saved / "ranking.csv", encoding="utf-8", newline="") as f:
            self.assertEqual(len(list(csv.DictReader(f))), N)

    def test_export_failure_is_shown_and_keeps_the_page(self):
        app, page = self.ready()
        page.reveal_button.invoke()
        with mock.patch.object(bundle, "export", side_effect=PermissionError("denied")):
            page.export_button.invoke()
            self.until(app, lambda: not page.busy)
        self.assertEqual((page.state, page.saved), ("revealed", None))
        self.assertIn("not saved", page.message.cget("text"))


class LayoutTests(GuiCase):
    def test_smallest_window_and_scales(self):
        for scale in (1.0, 1.25, 1.5):
            with mock.patch.object(gui.App, "winfo_fpixels", lambda self, s, k=scale: 96 * k, create=True):
                app, page = self.ready()
                app.geometry(f"{app.minsize()[0]}x{app.minsize()[1]}")
                app.update()
                page.reveal_button.invoke()
                app.update()
                for name in app.nav:
                    app.show(name)
                    app.update()
                app.show("predict")
                app.update()
                for w in (page.generate_button, page.reveal_button, page.export_button, page.next_button,
                          page.previous_button, page.find_button):
                    self.assertLessEqual(w.winfo_rootx() + w.winfo_width(), app.winfo_rootx() + app.winfo_width(),
                                         f"{w.cget('text')} at {scale}")
                self.assertGreater(page.table.winfo_height(), app.px(60))
                app.withdraw()

    def test_keyboard_shortcuts(self):
        app = self.open()
        app.focus_force()
        app.update()
        for i, name in enumerate(app.nav, 1):
            app.event_generate(f"<Control-Key-{i}>")
            app.update()
            self.assertEqual(app.current, name)
        app.show("predict")
        app.focus_force()
        app.event_generate("<Control-g>")
        self.until(app, lambda: app.predict.state == "forecast")
        app.event_generate("<Control-r>")
        app.update()
        self.assertEqual(app.predict.state, "revealed")

    def test_plain_text_has_no_false_claims(self):
        app = self.open()
        for p in app.pages.values():
            words = " ".join(texts(p)).lower()
            for claim in ("guarantee", "live monitoring of", "profit", "real-time"):
                self.assertNotIn(claim, words)
            self.assertFalse(re.search(r"\bnan\b", words))


if __name__ == "__main__":
    unittest.main()
