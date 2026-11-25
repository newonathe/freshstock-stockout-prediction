"""FreshStock 2.0 data contract: source audit, cohort, features and splits."""
import datetime as dt
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from freshstock import data as b3
from freshstock.expanded import data

FIRST = dt.date(2024, 3, 28)


def table(keys=((0, 1, 2), (0, 1, 3), (1, 0, 0)), days=10, first=FIRST, seed=5):
    """Long-format rows like the publisher file, in shuffled order."""
    rng = np.random.default_rng(seed)
    rows = [(c, s, p, (first + dt.timedelta(d)).isoformat()) for c, s, p in keys for d in range(days)]
    hours = rng.integers(0, 2, size=(len(rows), 24))
    columns = {"city_id": [r[0] for r in rows], "store_id": [r[1] for r in rows], "product_id": [r[2] for r in rows],
               "dt": [r[3] for r in rows], "sale_amount": rng.random(len(rows)).round(3).tolist(),
               "stock_hour6_22_cnt": hours[:, 6:22].sum(axis=1).tolist(), "hours_stock_status": hours.tolist()}
    order = rng.permutation(len(rows))
    return {k: [v[i] for i in order] for k, v in columns.items()}


SCHEMA = pa.schema([("city_id", pa.int64()), ("store_id", pa.int64()), ("product_id", pa.int64()),
                    ("dt", pa.string()), ("sale_amount", pa.float64()), ("stock_hour6_22_cnt", pa.int32()),
                    ("hours_stock_status", pa.list_(pa.int64()))])


def random_panel(n=6, days=97, seed=3):
    rng = np.random.default_rng(seed)
    keys = np.array(sorted({(int(a), int(b), int(c)) for a, b, c in rng.integers(0, 17, size=(n * 3, 3))})[:n])
    dates = np.arange(FIRST.toordinal(), FIRST.toordinal() + days, dtype=np.int32)
    stock = np.where(rng.random((n, days)) < .4, rng.integers(1, 17, (n, days)), 0).astype(np.uint8)
    return data.Panel(keys.astype(np.int64), dates, rng.random((n, days)), stock)


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "source.parquet"

    def source(self, columns=None, schema=SCHEMA, **changes):
        pq.write_table(pa.table(columns or table(), schema=schema), self.path)
        raw = self.path.read_bytes()
        values = {"name": "synthetic", "path": self.path, "sha256": hashlib.sha256(raw).hexdigest(),
                  "size": len(raw), "first_day": FIRST, "n_days": 10, "n_keys": 3}
        return data.Source(**{**values, **changes})

    def refused(self, source, text):
        with self.assertRaises(b3.DataError) as caught:
            data.read_panel(source)
        self.assertIn(text, str(caught.exception))

    def changed(self, column, row, value):
        columns = table()
        columns[column][row] = value
        return self.source(columns)

    def test_panel_matches_every_source_row_in_canonical_order(self):
        columns = table()
        panel = data.read_panel(self.source(columns))
        self.assertEqual(panel.keys.tolist(), [[0, 1, 2], [0, 1, 3], [1, 0, 0]])
        self.assertEqual(panel.dates.tolist(), [FIRST.toordinal() + i for i in range(10)])
        self.assertEqual((panel.keys.dtype, panel.dates.dtype, panel.sales.dtype, panel.stock.dtype),
                         (np.int64, np.int32, np.float64, np.uint8))
        for i in range(30):
            key = [columns[k][i] for k in b3.KEY]
            at = panel.keys.tolist().index(key), dt.date.fromisoformat(columns["dt"][i]).toordinal() - FIRST.toordinal()
            self.assertEqual(panel.sales[at], columns["sale_amount"][i])
            self.assertEqual(panel.stock[at], columns["stock_hour6_22_cnt"][i])

    def test_row_groups_and_batches_do_not_change_the_panel(self):
        whole = data.read_panel(self.source())
        small = data.read_panel(self.source(), batch_size=4)
        for name in ("keys", "dates", "sales", "stock"):
            np.testing.assert_array_equal(getattr(whole, name), getattr(small, name))

    def test_wrong_bytes_are_refused_before_parsing(self):
        good = self.source()
        self.refused(replace(good, sha256="0" * 64), "SHA256")
        self.refused(replace(good, size=good.size + 1), "size")
        self.path.write_bytes(self.path.read_bytes() + b"x")
        self.refused(good, "size")

    def test_missing_source_is_refused(self):
        self.refused(replace(self.source(), path=self.path.with_name("none.parquet")), "missing")

    def test_bad_values_are_refused(self):
        self.refused(self.changed("sale_amount", 3, float("nan")), "sale_amount")
        self.refused(self.changed("sale_amount", 3, float("inf")), "sale_amount")
        self.refused(self.changed("sale_amount", 3, -.1), "sale_amount")
        self.refused(self.changed("city_id", 3, 18), "city_id")
        self.refused(self.changed("store_id", 3, -1), "store_id")
        self.refused(self.changed("stock_hour6_22_cnt", 3, 17), "stock_hour6_22_cnt")
        self.refused(self.changed("dt", 3, "2024-04-07"), "date")
        self.refused(self.changed("dt", 3, "2024/04/01"), "date")
        self.refused(self.changed("sale_amount", 3, None), "null")

    def test_duplicates_and_gaps_are_refused(self):
        columns = table()
        for name in SCHEMA.names:
            columns[name][1] = columns[name][0]
        self.refused(self.source(columns), "duplicate")
        self.refused(self.source({k: v[1:] for k, v in table().items()}), "rows")
        self.refused(self.source(n_keys=5, n_days=6), "keys")

    def test_hourly_status_must_reproduce_the_label_count(self):
        columns = table()
        columns["hours_stock_status"][5] = list(columns["hours_stock_status"][5])
        columns["hours_stock_status"][5][7] ^= 1
        self.refused(self.source(columns), "hourly")
        columns = table()
        columns["hours_stock_status"][5] = columns["hours_stock_status"][5][:23]
        self.refused(self.source(columns), "hourly")

    def test_column_types_are_checked(self):
        wrong = SCHEMA.set(4, pa.field("sale_amount", pa.string()))
        columns = table()
        columns["sale_amount"] = [str(v) for v in columns["sale_amount"]]
        self.refused(self.source(columns, wrong), "type")

    def test_audit_receipt_has_no_holdout_outcomes(self):
        train = self.source()
        panel = data.read_panel(train)
        holdout_path = self.path.with_name("eval.parquet")
        pq.write_table(pa.table(table(days=7, first=FIRST + dt.timedelta(10), seed=8), schema=SCHEMA), holdout_path)
        raw = holdout_path.read_bytes()
        holdout = data.Source("holdout", holdout_path, hashlib.sha256(raw).hexdigest(), len(raw),
                              FIRST + dt.timedelta(10), 7, 3)
        receipt = data.audit(train, holdout)
        self.assertEqual(set(receipt["holdout"]), {"sha256", "size", "rows", "keys", "first_day", "last_day",
                                                   "hourly_labels_match"})
        self.assertTrue(receipt["same_keys"] and receipt["no_date_overlap"] and receipt["consecutive_dates"])
        self.assertIs(type(receipt["sales_scale_compatible"]), bool)
        self.assertEqual(receipt["train"]["rows"], panel.stock.size)
        json.dumps(receipt, allow_nan=False)

    def test_audit_refuses_other_keys_or_dates(self):
        train = self.source()
        path = self.path.with_name("eval.parquet")
        for keys, first in ((((0, 1, 2), (0, 1, 3), (1, 0, 1)), 10), (((0, 1, 2), (0, 1, 3), (1, 0, 0)), 9),
                            (((0, 1, 2), (0, 1, 3), (1, 0, 0)), 11)):
            start = FIRST + dt.timedelta(first)
            pq.write_table(pa.table(table(keys, 7, start), schema=SCHEMA), path)
            raw = path.read_bytes()
            with self.assertRaises(b3.DataError):
                data.audit(train, data.Source("holdout", path, hashlib.sha256(raw).hexdigest(), len(raw), start, 7, 3))

    def test_pinned_sources(self):
        self.assertEqual(data.TRAIN.sha256, b3.SOURCE_SHA256)
        self.assertEqual((data.HOLDOUT.sha256, data.HOLDOUT.size),
                         ("1b118840664280c6b88bffc84c80ee1f54c05d911e354b7599e5da10995e960e", 8440124))
        self.assertEqual((data.HOLDOUT.first_day, data.HOLDOUT.n_days), (dt.date(2024, 6, 26), 7))


class CohortTests(unittest.TestCase):
    def test_rank_is_independent_of_order_and_nested(self):
        rng = np.random.default_rng(1)
        keys = np.array([(c, s, p) for c in range(3) for s in range(10) for p in range(10)], dtype=np.int64)
        def expected(n):
            def rank(k):
                return hashlib.sha256(f"{k[0]},{k[1]},{k[2]}".encode("ascii")).hexdigest(), k
            return sorted(map(tuple, keys.tolist()), key=rank)[:n]
        previous = set()
        for n in (10, 50, 300):
            rows, digest = data.cohort_rows(keys, n)
            self.assertEqual({tuple(k) for k in keys[rows].tolist()}, set(expected(n)))
            self.assertTrue((np.diff(rows) > 0).all())  # canonical storage order is kept
            text = "".join(f"{a},{b},{c}\n" for a, b, c in expected(n))
            self.assertEqual(digest, hashlib.sha256(text.encode("ascii")).hexdigest())
            self.assertTrue(previous <= set(expected(n)))
            previous = set(expected(n))
        shuffled = keys[rng.permutation(len(keys))]
        rows, digest = data.cohort_rows(shuffled, 50)
        self.assertEqual({tuple(k) for k in shuffled[rows].tolist()}, set(expected(50)))

    def test_too_many_keys_refused(self):
        with self.assertRaises(b3.DataError):
            data.cohort_rows(np.zeros((3, 3), dtype=np.int64), 4)

    @unittest.skipUnless(data.TRAIN.path.is_file(), "needs the pinned source file")
    def test_first_100_is_the_b3_cohort(self):
        panel = data.read_panel(data.TRAIN, hourly=False)
        self.assertEqual(data.cohort_rows(panel.keys, 100)[1], b3.COHORT_DIGEST)


class FeatureTests(unittest.TestCase):
    """Expected values were computed independently of this code."""

    def fixture(self, stock_lags, days=40):
        target = dt.date(2024, 4, 25).toordinal() - FIRST.toordinal()  # index 28
        sales, stock = np.zeros((2, days)), np.zeros((2, days), dtype=np.uint8)
        sales[1] += 50  # a second series that must not leak into the first
        stock[1] += 9
        for lag in range(1, 29):
            sales[0, target - lag] = lag
        for lag, hours in stock_lags.items():
            stock[0, target - lag] = hours
        dates = np.arange(FIRST.toordinal(), FIRST.toordinal() + days, dtype=np.int32)
        return data.Panel(np.array([[0, 0, 0], [0, 0, 1]]), dates, sales, stock)

    def row(self, stock_lags):
        return data.features_on(self.fixture(stock_lags), dt.date(2024, 4, 25))[0].tolist()

    def test_names_and_order(self):
        self.assertEqual(data.FEATURES["A"], ["sales_lag1", "stockout_hours_lag1", "sales_mean7", "stockout_days7"]
                         + [f"weekday_{i}" for i in range(7)])
        self.assertEqual(data.FEATURES["B"], data.FEATURES["A"][:4] + [
            "stockout_hours_mean7", "stockout_hours_mean14", "stockout_days14", "sales_mean14", "sales_mean28",
            "stockout_days28", "days_since_stockout28"] + data.FEATURES["A"][4:])
        self.assertEqual([data.FEATURES["B"][i] for i in data.COLUMNS["A"]], data.FEATURES["A"])
        self.assertEqual([data.FEATURES["B"][i] for i in data.COLUMNS["B"]], data.FEATURES["B"])

    def test_hand_computed_row(self):
        expected = [1, 2, 4, 2] + [6 / 7, 12 / 14, 3, 7.5, 14.5, 4, 1] + [0, 0, 0, 1, 0, 0, 0]
        np.testing.assert_allclose(self.row({1: 2, 7: 4, 14: 6, 28: 8}), expected, rtol=0, atol=1e-12)

    def test_days_since_stockout_is_capped(self):
        at = data.FEATURES["B"].index("days_since_stockout28")
        self.assertEqual(self.row({})[at], 28)
        self.assertEqual(self.row({28: 3})[at], 28)
        self.assertEqual(self.row({5: 1, 27: 3})[at], 5)

    def test_warmup_is_required(self):
        panel = self.fixture({})
        data.features_on(panel, dt.date(2024, 4, 25))
        for day in (dt.date(2024, 4, 24), dt.date(2024, 3, 28), dt.date(2024, 3, 1)):
            with self.assertRaises(b3.DataError):
                data.features_on(panel, day)

    def test_next_day_after_the_history_is_allowed_but_not_two(self):
        panel = self.fixture({}).until(dt.date(2024, 4, 24))
        self.assertEqual(len(panel.dates), 28)
        data.features_on(panel, dt.date(2024, 4, 25))
        with self.assertRaises(b3.DataError):
            data.features_on(panel, dt.date(2024, 4, 26))

    def test_gap_in_dates_is_refused(self):
        p = self.fixture({})
        keep = [i for i in range(40) if i != 20]
        with self.assertRaises(b3.DataError):
            data.Panel(p.keys, p.dates[keep], p.sales[:, keep], p.stock[:, keep])

    def test_duplicate_or_unsorted_keys_are_refused(self):
        p = random_panel()
        for order in ([0, 0, 2, 3, 4, 5], [1, 0, 2, 3, 4, 5]):
            with self.assertRaises(b3.DataError):
                data.Panel(p.keys[order], p.dates, p.sales, p.stock)

    def test_target_and_future_values_never_matter(self):
        panel = random_panel()
        for day in (dt.date(2024, 4, 25), dt.date(2024, 6, 4), dt.date(2024, 6, 5), dt.date(2024, 6, 12),
                    dt.date(2024, 6, 19), dt.date(2024, 6, 26), dt.date(2024, 7, 2)):
            before = data.features_on(panel, day)
            at = day.toordinal() - FIRST.toordinal()
            sales, stock = panel.sales.copy(), panel.stock.copy()
            sales[:, at:], stock[:, at:] = 999, 16
            after = data.features_on(data.Panel(panel.keys, panel.dates, sales, stock), day)
            np.testing.assert_array_equal(before, after)
            sales[:, at - 1] += 1
            changed = data.features_on(data.Panel(panel.keys, panel.dates, sales, stock), day)
            self.assertFalse(np.array_equal(before, changed))

    def test_series_are_isolated(self):
        panel, day = random_panel(), dt.date(2024, 5, 3)
        before = data.features_on(panel, day)
        sales, stock = panel.sales.copy(), panel.stock.copy()
        sales[2], stock[2] = 7, 16
        after = data.features_on(data.Panel(panel.keys, panel.dates, sales, stock), day)
        others = [0, 1, 3, 4, 5]
        np.testing.assert_array_equal(before[others], after[others])
        self.assertFalse(np.array_equal(before[2], after[2]))

    def test_matches_an_exact_date_reference(self):
        panel = random_panel(4)
        for day in (dt.date(2024, 4, 25), dt.date(2024, 5, 31), dt.date(2024, 7, 2)):
            got = data.features_on(panel, day)
            for k in range(4):
                lookup = {int(d): (panel.sales[k, i], int(panel.stock[k, i])) for i, d in enumerate(panel.dates)}
                past = [lookup[day.toordinal() - lag] for lag in range(1, 29)]
                s, h = [p[0] for p in past], [p[1] for p in past]
                since = next((lag for lag in range(1, 29) if h[lag - 1] > 0), 28)
                expected = [s[0], h[0], sum(s[:7]) / 7, sum(x > 0 for x in h[:7]), sum(h[:7]) / 7, sum(h[:14]) / 14,
                            sum(x > 0 for x in h[:14]), sum(s[:14]) / 14, sum(s) / 28, sum(x > 0 for x in h), since]
                expected += [float(day.weekday() == i) for i in range(7)]
                np.testing.assert_allclose(got[k], expected, rtol=0, atol=1e-10)

    @unittest.skipUnless((b3.DERIVED / "cohort.csv").is_file() and data.TRAIN.path.is_file(), "needs local data")
    def test_set_a_equals_b3_on_the_shared_series_and_dates(self):
        cohort, _ = b3.load_cohort()
        panel = data.read_panel(data.TRAIN, hourly=False)
        panel = panel.take(data.cohort_rows(panel.keys, 100)[0])
        for day in ("2024-04-25", "2024-05-26", "2024-06-25"):
            old = b3.features_at(cohort[cohort["dt"] < day], day)
            new = data.features_on(panel, dt.date.fromisoformat(day))[:, data.COLUMNS["A"]]
            self.assertEqual(old[b3.KEY].to_numpy().tolist(), panel.keys.tolist())
            np.testing.assert_allclose(new[:, :4], old[b3.NUMERIC].to_numpy(dtype=float), rtol=0, atol=1e-10)
            np.testing.assert_array_equal(new[:, 4:].argmax(axis=1), old["weekday"])


class SplitTests(unittest.TestCase):
    def test_dates(self):
        self.assertEqual({k: (a.isoformat(), b.isoformat(), (b - a).days + 1) for k, (a, b) in data.SPLITS.items()}, {
            "initial_fit": ("2024-04-25", "2024-06-04", 41), "tuning": ("2024-06-05", "2024-06-11", 7),
            "calibration": ("2024-06-12", "2024-06-18", 7), "threshold": ("2024-06-19", "2024-06-25", 7),
            "refit": ("2024-04-25", "2024-06-11", 48), "holdout": ("2024-06-26", "2024-07-02", 7)})

    def test_rows_are_date_then_key_and_labels_align(self):
        panel = random_panel(5)
        for name in ("initial_fit", "tuning", "calibration", "threshold", "refit"):
            x, y = data.split(panel, name)
            first, last = data.SPLITS[name]
            days = (last - first).days + 1
            self.assertEqual((x.shape, y.shape, y.dtype), ((days * 5, 18), (days * 5,), np.uint8))
            self.assertTrue(x.flags.c_contiguous and np.isfinite(x).all())
            for d in (0, days - 1):
                day = first + dt.timedelta(d)
                np.testing.assert_array_equal(x[d * 5:d * 5 + 5], data.features_on(panel, day))
                np.testing.assert_array_equal(
                    y[d * 5:d * 5 + 5], panel.stock[:, day.toordinal() - FIRST.toordinal()] > 0)

    def test_set_a_is_a_column_selection(self):
        panel = random_panel(5)
        a, _ = data.split(panel, "tuning", "A")
        b, _ = data.split(panel, "tuning", "B")
        np.testing.assert_array_equal(a, b[:, data.COLUMNS["A"]])
        self.assertTrue(a.flags.c_contiguous)

    def test_development_panel_stops_before_the_holdout(self):
        panel = random_panel(5).until(dt.date(2024, 6, 25))
        data.split(panel, "threshold")
        with self.assertRaises(b3.DataError):
            data.split(panel, "holdout")

    def test_join_requires_matching_keys_and_next_day(self):
        panel = random_panel(5)
        dev, rest = panel.until(dt.date(2024, 6, 25)), panel.since(dt.date(2024, 6, 26))
        whole = dev.join(rest)
        np.testing.assert_array_equal(whole.stock, panel.stock)
        with self.assertRaises(b3.DataError):
            dev.join(panel.since(dt.date(2024, 6, 27)))
        with self.assertRaises(b3.DataError):
            dev.join(panel.since(dt.date(2024, 6, 25)))
        with self.assertRaises(b3.DataError):
            dev.take([0, 1, 2, 3]).join(rest)


if __name__ == "__main__":
    unittest.main()
