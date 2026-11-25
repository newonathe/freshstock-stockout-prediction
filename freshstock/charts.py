"""Small Tk canvas charts. One axis each, hairline grid, direct labels; exact values are always written out."""
import tkinter as tk
from tkinter import font as tkfont

CANVAS, SURFACE, INK, MUTED, BRAND, LINE = "#F6F8F5", "#FFFFFF", "#172C26", "#52645D", "#185B47", "#D7E2DA"
SOFT = "#E3EEE7"
# Checked with a palette validator on a light surface: lightness, chroma, color-blind separation, contrast.
SERIES = {"logistic": "#12805C", "forest": "#D9730D", "persistence": "#4F6BD8"}
# Outcome, fill, text color. Order puts the recorded stockouts (correct alert + miss) first.
OUTCOMES = [("Correct alert", BRAND, "#FFFFFF"), ("Miss", "#E39B2D", INK), ("False alert", "#A99BD6", INK),
            ("Correct clear", "#DCE5E0", INK)]
FONT = "Segoe UI"
MONTHS = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def axis_top(biggest):
    """Round axis end at or above the largest count; 1 when every count is zero."""
    if biggest <= 0:
        return 1
    step = 10
    while biggest > step * 10:
        step *= 10
    return next(m * step for m in (2, 4, 6, 8, 10) if biggest <= m * step)


class Chart(tk.Canvas):
    def __init__(self, parent, app, height, back=SURFACE):
        super().__init__(parent, bg=back, highlightthickness=0, height=app.px(height), width=app.px(200))
        self.app = app
        self.bind("<Configure>", lambda e: self.redraw())
        self.bind("<Motion>", lambda e: self.hover(e.x, e.y))
        self.bind("<Leave>", lambda e: self.read(""))

    def redraw(self):
        self.delete("all")
        if self.winfo_width() > 60:
            self.draw(self.winfo_width(), self.winfo_height())

    def read(self, text):
        self.itemconfigure("readout", text=text)

    def hover(self, x, y):
        pass

    def label(self, x, y, text, anchor="w", color=MUTED, size=9, weight="normal", **kw):
        return self.create_text(x, y, text=text, anchor=anchor, fill=color, font=(FONT, size, weight), **kw)

    def measure(self, text, size=9, weight="normal"):
        return tkfont.Font(root=self, family=FONT, size=size, weight=weight).measure(text)

    def legend(self, entries, x, y):
        px = self.app.px
        for name, color in entries:
            self.create_rectangle(x, y - px(5), x + px(10), y + px(5), fill=color, outline=LINE)
            item = self.label(x + px(15), y, name, color=INK)
            x = self.bbox(item)[2] + px(16)


class Pipeline(Chart):
    """What the program does, left to right, with the reveal as a separate later step."""

    STEPS = [("Past 7 days", "sales and stockout hours"), ("Two frozen models", "Logistic Regression, Random Forest"),
             ("Risk ranking", "100 series, one day ahead"), ("Human review", "a planner decides")]

    def __init__(self, parent, app, steps=STEPS):
        super().__init__(parent, app, 200)
        self.steps = steps

    def draw(self, w, h):
        px = self.app.px
        gap, top, tall = px(34), px(8), px(76)
        wide = (w - 2 - gap * 3) / 4
        for i, (title, note) in enumerate(self.steps):
            x = 1 + i * (wide + gap)
            main = i == 1
            self.create_rectangle(x, top, x + wide, top + tall, fill=BRAND if main else SOFT, width=0)
            ink = "#FFFFFF" if main else INK
            self.create_text(x + wide / 2, top + px(24), text=title, fill=ink, font=(FONT, 11, "bold"),
                             width=wide - px(8), justify="center")
            self.create_text(x + wide / 2, top + px(53), text=note, fill="#FFFFFF" if main else MUTED, font=(FONT, 9),
                             width=wide - px(10), justify="center")
            if i < 3:
                self.create_line(x + wide + px(5), top + tall / 2, x + wide + gap - px(5), top + tall / 2, fill=MUTED,
                                 width=px(2), arrow="last", arrowshape=(px(9), px(10), px(4)))
        x = 1 + 2 * (wide + gap)
        low = top + tall + px(34)
        self.create_line(x + wide / 2, top + tall + px(4), x + wide / 2, low - px(4), fill=MUTED, width=px(2),
                         arrow="last", arrowshape=(px(9), px(10), px(4)))
        self.create_rectangle(x, low, x + wide, low + px(72), outline=BRAND, dash=(4, 3), width=px(2), fill=SURFACE)
        self.create_text(x + wide / 2, low + px(19), text="Reveal", fill=BRAND, font=(FONT, 11, "bold"))
        self.create_text(x + wide / 2, low + px(47), text="recorded outcome, after the forecast", fill=MUTED,
                         font=(FONT, 9), width=wide - px(8), justify="center")


class SplitBar(Chart):
    """Proportional date split."""

    COLORS = [BRAND, "#6E8F80", INK]
    MANY = [BRAND, "#6E8F80", "#8FA89C", "#6E8F80", INK]

    def __init__(self, parent, app, parts):
        super().__init__(parent, app, 96)
        self.parts = parts  # (title, rows, dates)

    def draw(self, w, h):
        px = self.app.px
        total, x = sum(rows for _, rows, _ in self.parts), 1
        for (title, rows, dates), color in zip(self.parts, self.COLORS if len(self.parts) == 3 else self.MANY):
            wide = (w - 2) * rows / total
            self.create_rectangle(x, px(2), x + wide - px(2), px(34), fill=color, width=0, tags="segment")
            self.label(x, px(52), title, color=INK, size=10, weight="bold")
            self.label(x, px(70), f"{rows:,} rows" if len(self.parts) == 3 else f"{rows:,}")
            self.label(x, px(86), dates)
            x += wide


class Stacked(Chart):
    """One 100-series bar per method, split by outcome, with the exact counts written beside it."""

    def __init__(self, parent, app):
        super().__init__(parent, app, 112, CANVAS)
        self.counts, self.names = {}, {}

    def show(self, names, counts):
        self.names, self.counts = names, counts
        self.redraw()

    def draw(self, w, h):
        px = self.app.px
        if not self.counts:
            return
        texts = {name: "TP {:,} · FN {:,} · FP {:,} · TN {:,}".format(*c) for name, c in self.counts.items()}
        left = px(140)
        right = w - max(px(212), px(24) + max(self.measure(t) for t in texts.values()))
        self.legend([(name, fill) for name, fill, _ in OUTCOMES], left, px(10))
        for r, (name, counts) in enumerate(self.counts.items()):
            y, x, total = px(28 + 28 * r), left, sum(counts) or 1
            self.label(0, y + px(11), self.names[name], color=INK, size=10)
            for k, (n, (_, fill, ink)) in enumerate(zip(counts, OUTCOMES, strict=True)):
                wide = (right - left) * n / total
                self.create_rectangle(x, y, x + wide, y + px(22), fill=fill, outline=CANVAS,
                                      tags=("segment", name, str(k)))
                if wide > self.measure(f"{n:,}") + px(6):
                    self.label(x + wide / 2, y + px(11), f"{n:,}", anchor="center", color=ink, weight="bold")
                x += wide
            self.label(right + px(10), y + px(11), texts[name], color=INK)


class Compare(Chart):
    """Horizontal bars from zero with the value at the end. Groups of (title, [(name, color, value), ...])."""

    def __init__(self, parent, app, groups, top, fmt):
        rows = sum(len(g[1]) for g in groups)
        titles = sum(1 for g in groups if g[0])
        super().__init__(parent, app, 30 * rows + 28 * titles + 6)
        self.groups, self.top, self.fmt = groups, top, fmt

    def draw(self, w, h):
        px = self.app.px
        widest = max(self.measure(self.fmt.format(v), 10, "bold") for _, rows in self.groups for _, _, v in rows)
        names = max(self.measure(name, 10) for _, rows in self.groups for name, _, _ in rows)
        left, right, y = max(px(132), names + px(10)), w - max(px(56), widest + px(12)), 0
        for title, rows in self.groups:
            if title:
                self.label(0, y + px(13), title, color=INK, size=10, weight="bold")
                y += px(28)
            for name, color, value in rows:
                self.label(0, y + px(14), name, color=INK, size=10)
                end = left + (right - left) * value / (self.top or 1)
                self.create_rectangle(left, y + px(4), end, y + px(24), fill=color, width=0, tags="bar")
                self.label(end + px(6), y + px(14), self.fmt.format(value), color=INK, size=10, weight="bold")
                y += px(30)


class Columns(Chart):
    """Few vertical bars, one color, value on top."""

    def __init__(self, parent, app, names, notes, values, top, fmt, height=196):
        super().__init__(parent, app, height)
        self.names, self.notes, self.values, self.top, self.fmt = names, notes, values, top, fmt

    def draw(self, w, h):
        px = self.app.px
        base, high, slot = h - px(44), px(24), w / len(self.names)
        self.create_line(0, base, w, base, fill=LINE)
        for i, (name, note, v) in enumerate(zip(self.names, self.notes, self.values, strict=True)):
            x, y = slot * (i + 0.5), base - (base - high) * v / self.top
            half = min(px(30), slot / 3)
            self.create_rectangle(x - half, y, x + half, base, fill=BRAND, width=0)
            self.label(x, y - px(11), self.fmt.format(v), anchor="center", color=INK, size=12, weight="bold")
            self.label(x, base + px(14), name, anchor="center", color=INK, size=10)
            self.label(x, base + px(32), note, anchor="center")


class Lines(Chart):
    """One line per series over days; end values are labeled without overlap."""

    def __init__(self, parent, app, days, height=250):
        super().__init__(parent, app, height)
        self.days, self.series, self.xs = days, [], []

    def show(self, series):
        self.series = series
        self.redraw()

    def draw(self, w, h):
        px = self.app.px
        if not self.series:
            return
        biggest = max(max(s["values"]) for s in self.series)
        peak = axis_top(biggest)
        left = max(px(34), self.measure(f"{peak:,}") + px(12))
        right = w - max(px(44), self.measure(f"{biggest:,}", 10, "bold") + px(10))
        top, bottom = px(50), h - px(26)
        self.legend([(s["name"], s["color"]) for s in self.series], left, px(10))
        self.label(left, px(31), "", tags="readout")
        for i in range(5):
            y = bottom - (bottom - top) * i / 4
            self.create_line(left, y, right, y, fill=LINE)
            self.label(left - px(6), y, f"{peak * i / 4:,.0f}" if peak >= 4 else f"{peak * i / 4:g}", anchor="e")
        step = (right - left - px(20)) / (len(self.days) - 1)
        self.xs = [left + px(10) + step * i for i in range(len(self.days))]
        for x, day in zip(self.xs, self.days):
            self.label(x, bottom + px(14), f"{MONTHS[int(day[5:7])]} {day[-2:]}" if step > px(44) else day[-2:], anchor="center")
        ends = []
        for s in self.series:
            points = [(x, bottom - (bottom - top) * v / peak) for x, v in zip(self.xs, s["values"])]
            self.create_line(*[c for p in points for c in p], fill=s["color"], width=px(2))
            for x, y in points:
                self.create_oval(x - px(4), y - px(4), x + px(4), y + px(4), fill=s["color"], outline=SURFACE)
            ends.append([points[-1][1], s["values"][-1]])
        ends.sort()
        gap = px(17)
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
        spill = max(0, ends[-1][0] - (h - px(10)))
        for y, value in ends:
            self.label(right + px(4), y - spill, f"{value:,}", color=INK, size=10, weight="bold", tags="end")

    def hover(self, x, y):
        if self.xs:
            i = min(range(len(self.xs)), key=lambda k: abs(self.xs[k] - x))
            self.read(f"{self.days[i]}: " + ", ".join(f"{s['name']} {s['values'][i]:,}" for s in self.series))


class HBars(Chart):
    """Labeled horizontal bars; negative values extend to the left of zero."""

    def __init__(self, parent, app, title, names, values, color):
        super().__init__(parent, app, 40 + 24 * len(names))
        self.title, self.names, self.values, self.color = title, names, values, color

    def draw(self, w, h):
        px = self.app.px
        self.label(0, px(12), self.title, color=INK, size=10, weight="bold")
        left, right = px(165), w - px(64)
        low, high = min(0, *self.values), max(0, *self.values)
        if low < 0:
            left += px(52)  # room for the value label left of a negative bar

        def at(v):
            return left + (right - left) * (v - low) / (high - low)
        for i, (name, v) in enumerate(zip(self.names, self.values)):
            y = px(36 + 24 * i)
            self.label(px(155), y, name, anchor="e", color=INK)
            self.create_rectangle(at(min(0, v)), y - px(6), at(max(0, v)), y + px(6), fill=self.color, width=0)
            self.label(at(v) + px(5 if v >= 0 else -5), y, f"{v:.4f}", anchor="w" if v >= 0 else "e")
        self.create_line(at(0), px(24), at(0), px(26 + 24 * len(self.names)), fill=MUTED)


class Confusion(Chart):
    """Actual rows, predicted columns; each cell names its count."""

    def __init__(self, parent, app, title, color, row):
        super().__init__(parent, app, 178)
        self.title, self.color, self.row = title, color, row

    def draw(self, w, h):
        px, r = self.app.px, self.row
        self.create_rectangle(0, px(7), px(10), px(17), fill=self.color, width=0)
        self.label(px(16), px(12), self.title, color=INK, size=10, weight="bold")
        left, right, top, cell = px(64), w - px(14), px(48), px(58)
        half = (right - left) / 2
        for j, text in enumerate(("Predicted 0", "Predicted 1")):
            self.label(left + half * (j + 0.5), px(37), text, anchor="center")
        for i, names in enumerate((("tn", "fp"), ("fn", "tp"))):
            self.label(left - px(8), top + cell * (i + 0.5), f"Actual {i}", anchor="e")
            for j, name in enumerate(names):
                x, y = left + half * j, top + cell * i
                fill = SOFT if name in ("tn", "tp") else "#F9E7E4"
                self.create_rectangle(x + 1, y + 1, x + half - 1, y + cell - 1, fill=fill, width=0)
                self.label(x + half / 2, y + cell / 2 - px(11), name.upper(), anchor="center")
                self.label(x + half / 2, y + cell / 2 + px(9), f"{r[name]:,}", anchor="center", color=INK, size=13,
                           weight="bold")
