"""FS-01 to FS-04: source/cohort contract and leakage-safe features."""
import datetime as dt
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from freshstock import data

SOURCE = data.find_source()
D0 = dt.date(2024, 4, 1)  # a Monday


def series(key, sales, stock, start=D0):
    return pd.DataFrame({"city_id": key[0], "store_id": key[1], "product_id": key[2],
                         "dt": [(start + dt.timedelta(i)).isoformat() for i in range(len(sales))],
                         "sale_amount": np.asarray(sales, float), "stock_hour6_22_cnt": stock})


def fixture():
    a = series((1, 2, 3), [1, 2, 3, 4, 5, 6, 7, 8], [0, 0, 5, 0, 1, 0, 16, 2])
    b = series((0, 9, 9), [10] * 8, [0] * 8)
    return pd.concat([a, b], ignore_index=True)


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.df = fixture()
        self.target = "2024-04-08"

    def at(self, df, day):
        return data.features_at(df[df["dt"] < day], day)

    def test_known_seven_day_fixture(self):
        x = self.at(self.df, self.target)
        self.assertEqual(x[data.KEY].values.tolist(), [[0, 9, 9], [1, 2, 3]])
        self.assertEqual(list(x.columns), data.KEY + ["dt"] + data.FEATURES)
        self.assertEqual(x[data.FEATURES].values.tolist(), [[10, 0, 10, 0, 0], [7, 16, 4, 3, 0]])

    def test_target_day_row_is_refused_as_history(self):
        with self.assertRaisesRegex(data.DataError, "on or after"):
            data.features_at(self.df, self.target)

    def test_missing_history_day_is_refused(self):
        with self.assertRaisesRegex(data.DataError, "gap"):
            self.at(self.df[self.df["dt"] != "2024-04-03"], self.target)

    def test_stale_history_without_previous_day_is_refused(self):
        with self.assertRaisesRegex(data.DataError, "history"):
            self.at(self.df[self.df["dt"] != "2024-04-07"], self.target)

    def test_single_series_gap_is_refused(self):
        drop = (self.df["dt"] == "2024-04-03") & (self.df["city_id"] == 0)
        with self.assertRaisesRegex(data.DataError, "gap"):
            data.validate(self.df[~drop])

    def test_fewer_than_seven_days_is_refused(self):
        with self.assertRaisesRegex(data.DataError, "history"):
            self.at(self.df, "2024-04-07")

    def test_row_order_does_not_matter(self):
        s = self.df.sample(frac=1, random_state=111)
        self.assertTrue(self.at(self.df, self.target).equals(self.at(s, self.target)))

    def test_future_mutation_leaves_features_unchanged(self):
        m = self.df.copy()
        f = m["dt"] >= self.target
        m.loc[f, "sale_amount"] += 123
        m.loc[f, "stock_hour6_22_cnt"] = 16 - m.loc[f, "stock_hour6_22_cnt"]
        self.assertTrue(self.at(self.df, self.target).equals(self.at(m, self.target)))

    def test_other_series_is_isolated(self):
        m = self.df.copy()
        m.loc[m["city_id"] == 1, "sale_amount"] += 5
        a, b = self.at(self.df, self.target), self.at(m, self.target)
        self.assertTrue(a.iloc[[0]].equals(b.iloc[[0]]))
        self.assertFalse(a.iloc[[1]].equals(b.iloc[[1]]))


class ValidateTests(unittest.TestCase):
    def refuse(self, col, value, message):
        df = fixture()
        df[col] = df[col].astype(object)
        df.loc[2, col] = value
        with self.assertRaisesRegex(data.DataError, message):
            data.validate(df.infer_objects())

    def test_bad_values_are_refused(self):
        for col, value, message in (("sale_amount", None, "null"), ("sale_amount", -1.0, "sale_amount"),
                                    ("sale_amount", np.inf, "sale_amount"), ("stock_hour6_22_cnt", 17, "stock_hour"),
                                    ("stock_hour6_22_cnt", 1.5, "integer"), ("city_id", 18, "city_id"),
                                    ("dt", "2024-13-01", "date"), ("dt", "04/01/2024", "date")):
            with self.subTest(col=col, value=value):
                self.refuse(col, value, message)

    def test_duplicate_is_refused(self):
        df = fixture()
        with self.assertRaisesRegex(data.DataError, "duplicate"):
            data.validate(pd.concat([df, df.iloc[[0]]]))

    def test_missing_column_is_refused(self):
        with self.assertRaisesRegex(data.DataError, "columns"):
            data.validate(fixture().drop(columns="dt"))


class CohortRuleTests(unittest.TestCase):
    def test_rank_is_ascending_sha256_of_comma_joined_key(self):
        keys = [(0, 0, 1), (0, 0, 2), (3, 4, 5)]
        want = sorted(keys, key=lambda k: hashlib.sha256("{},{},{}".format(*k).encode()).hexdigest())
        self.assertEqual(data.rank_keys(keys, 2), want[:2])

    def test_wrong_file_is_refused_by_checksum(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "train.parquet"
            p.write_bytes(b"PAR1 not the dataset PAR1")
            with self.assertRaisesRegex(data.DataError, "SHA256"):
                data.audit_source(p)


@unittest.skipUnless(SOURCE, "original train.parquet not available")
class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cohort, cls.receipt = data.audit_source(SOURCE)

    def test_full_audit_receipt(self):
        self.assertEqual(self.receipt["rows"], 4_500_000)
        self.assertEqual(self.receipt["keys"], 50_000)
        self.assertEqual(self.receipt["cohort_digest"], data.COHORT_DIGEST)
        self.assertEqual(self.receipt["stockout_histogram"], data.HISTOGRAM)

    def test_cohort_and_split_counts(self):
        self.assertEqual(len(self.cohort), 9000)
        for name, expected in {"train": (5300, 2403), "validation": (1500, 535), "test": (1500, 682)}.items():
            x, y = data.partition(self.cohort, name)
            self.assertEqual((len(x), int(y.sum())), expected)

    def test_split_dates_are_disjoint_and_inputs_exclude_outcome(self):
        days = {n: set(data.partition(self.cohort, n)[0]["dt"]) for n in ("train", "validation", "test")}
        self.assertLess(max(days["train"]), min(days["validation"]))
        self.assertLess(max(days["validation"]), min(days["test"]))
        self.assertEqual(data.FEATURES, ["sales_lag1", "stockout_hours_lag1", "sales_mean7", "stockout_days7", "weekday"])

    def test_real_features_match_independent_date_lookup(self):
        look = {tuple(r[:4]): r[4:] for r in self.cohort.itertuples(index=False)}
        x = pd.concat([data.partition(self.cohort, n)[0] for n in ("train", "validation", "test")])
        self.assertEqual(len(x), 8300)
        for r in x.itertuples(index=False):
            d = dt.date.fromisoformat(r[3])
            past = [look[(*r[:3], (d - dt.timedelta(i)).isoformat())] for i in range(1, 8)]
            want = (past[0][0], past[0][1], sum(p[0] for p in past) / 7, sum(p[1] > 0 for p in past), d.weekday())
            for got, exp in zip(r[4:], want):
                self.assertAlmostEqual(got, exp, delta=1e-12)

    def test_csv_round_trip_is_exact_and_tamper_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            data.save_cohort(self.cohort, self.receipt, Path(d))
            self.assertTrue(data.load_cohort(Path(d))[0].equals(self.cohort))
            p = Path(d) / "cohort.csv"
            p.write_bytes(p.read_bytes().replace(b"2024-03-28", b"2024-03-27", 1))
            with self.assertRaisesRegex(data.DataError, "SHA256"):
                data.load_cohort(Path(d))


if __name__ == "__main__":
    unittest.main()
