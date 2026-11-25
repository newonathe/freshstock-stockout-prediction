"""Charts must work for any cohort size and keep the pilot's drawing."""
import tkinter as tk
import unittest

from freshstock import charts, gui

try:
    tk.Tk().destroy()
    HAS_TK = True
except tk.TclError:
    HAS_TK = False


class AxisTests(unittest.TestCase):
    def test_counts_up_to_100_keep_the_old_axis(self):
        for value, top in ((1, 20), (20, 20), (21, 40), (54, 60), (72, 80), (81, 100), (100, 100)):
            self.assertEqual(charts.axis_top(value), top)

    def test_any_count_has_a_round_axis_above_it(self):
        self.assertEqual(charts.axis_top(0), 1)
        for value in (101, 999, 10_000, 10_001, 25_000, 49_999, 50_000, 350_000):
            top = charts.axis_top(value)
            self.assertGreaterEqual(top, value)
            self.assertLessEqual(top, value * 2)
            self.assertEqual(top % 4, 0)


class FakeApp(tk.Tk):
    scale = 1.0

    def px(self, n):
        return round(n * self.scale)


@unittest.skipUnless(HAS_TK, "needs a display")
class DrawTests(unittest.TestCase):
    def setUp(self):
        self.app = FakeApp()
        self.addCleanup(self.app.destroy)
        self.app.geometry("800x400")

    def place(self, chart):
        chart.pack(fill="x")
        self.app.update()
        return chart

    def labels(self, chart):
        return [chart.itemcget(i, "text") for i in chart.find_all() if chart.type(i) == "text"]

    def inside(self, chart):
        for i in chart.find_all():
            if chart.type(i) == "text" and chart.itemcget(i, "text"):
                x0, _, x1, _ = chart.bbox(i)
                self.assertGreaterEqual(x0, -2, chart.itemcget(i, "text"))
                self.assertLessEqual(x1, chart.winfo_width() + 2, chart.itemcget(i, "text"))

    def test_lines_for_every_scale(self):
        days = ["2024-06-29", "2024-06-30", "2024-07-01", "2024-07-02"]
        for n in (0, 100, 101, 10_000, 25_000, 50_000):
            chart = self.place(charts.Lines(self.app, self.app, days))
            chart.show([{"name": "Alerts", "color": "#000000", "values": [0, n // 2, n, n]},
                        {"name": "Misses", "color": "#111111", "values": [n, 0, 0, 0]}])
            self.app.update()
            texts = self.labels(chart)
            self.assertIn(f"{n:,}", texts)
            self.assertIn("Jul 01", texts)
            self.assertIn("Jun 29", texts)
            self.inside(chart)
            chart.destroy()

    def test_stacked_for_every_scale(self):
        for n in (0, 100, 101, 10_000, 25_000, 50_000):
            chart = self.place(charts.Stacked(self.app, self.app))
            chart.show({"a": "Logistic Regression"}, {"a": (n // 2, n - n // 2, 0, 0)})
            self.app.update()
            self.assertIn(f"TP {n // 2:,} · FN {n - n // 2:,} · FP 0 · TN 0", self.labels(chart))
            self.inside(chart)
            chart.destroy()

    def test_compare_with_zero_and_large_values(self):
        for top, values in ((0, (0, 0)), (100, (3, 100)), (350_000, (0, 349_999))):
            chart = self.place(charts.Compare(self.app, self.app, [("Counts", [
                ("A", "#000000", values[0]), ("B", "#111111", values[1])])], top, "{:,.0f}"))
            self.assertIn(f"{values[1]:,.0f}", self.labels(chart))
            self.inside(chart)
            chart.destroy()

    def test_compare_names_never_run_into_their_bars(self):
        self.app.scale = .6
        chart = self.place(charts.Compare(self.app, self.app, [("x", [("Logistic Regression", "#000000", 1)])], 1, "{}"))
        name = next(i for i in chart.find_all() if chart.type(i) == "text" and "Logistic" in chart.itemcget(i, "text"))
        self.assertLess(chart.bbox(name)[2], chart.bbox(chart.find_withtag("bar")[0])[0])

    def test_pipeline_text_is_given_by_the_page(self):
        steps = [("Past 28 days", "x"), ("Two frozen models", "y"), ("Risk ranking", "50,000 series"), ("Review", "z")]
        texts = self.labels(self.place(charts.Pipeline(self.app, self.app, steps)))
        self.assertIn("Past 28 days", texts)
        self.assertNotIn("Past 7 days", texts)
        self.assertIn("Past 7 days", self.labels(self.place(charts.Pipeline(self.app, self.app))))

    def test_split_bar_accepts_more_parts_than_three(self):
        parts = [("Fit", 41, "a"), ("Tune", 7, "b"), ("Calibrate", 7, "c"), ("Threshold", 7, "d"), ("Test", 7, "e")]
        chart = self.place(charts.SplitBar(self.app, self.app, parts))
        self.assertEqual(len(chart.find_withtag("segment")), 5)


class OldAppHooksTests(unittest.TestCase):
    def test_old_app_still_uses_the_frozen_run(self):
        self.assertIs(gui.App.health_of, gui.pf.build_health)


if __name__ == "__main__":
    unittest.main()
