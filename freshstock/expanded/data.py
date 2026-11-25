"""Pinned sources, deterministic cohort, past-only features and fixed date splits for the expanded experiment."""
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data import ID_MAX, KEY, ROOT, SOURCE_SHA256, keys_digest, parse_day, rank_keys, require, sha256_file

REVISION = "08c1fab7f9257bc73679d415d65d644165d351d4"
SOURCE_URL = "https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K/resolve/" + REVISION + "/data/{}.parquet"
WARMUP = 28
NUMERIC_A = ["sales_lag1", "stockout_hours_lag1", "sales_mean7", "stockout_days7"]
EXTRAS = ["stockout_hours_mean7", "stockout_hours_mean14", "stockout_days14", "sales_mean14", "sales_mean28",
          "stockout_days28", "days_since_stockout28"]
WEEKDAYS = [f"weekday_{i}" for i in range(7)]
FEATURES = {"A": NUMERIC_A + WEEKDAYS, "B": NUMERIC_A + EXTRAS + WEEKDAYS}
COLUMNS = {s: [FEATURES["B"].index(name) for name in FEATURES[s]] for s in FEATURES}
N_NUMERIC = {"A": len(NUMERIC_A), "B": len(NUMERIC_A) + len(EXTRAS)}
TYPES = {"city_id": "int64", "store_id": "int64", "product_id": "int64", "dt": "string", "sale_amount": "double",
         "stock_hour6_22_cnt": "int32"}
HOURLY = "hours_stock_status"


def _range(first, last):
    return dt.date.fromisoformat(first), dt.date.fromisoformat(last)


SPLITS = {"initial_fit": _range("2024-04-25", "2024-06-04"), "tuning": _range("2024-06-05", "2024-06-11"),
          "calibration": _range("2024-06-12", "2024-06-18"), "threshold": _range("2024-06-19", "2024-06-25"),
          "refit": _range("2024-04-25", "2024-06-11"), "holdout": _range("2024-06-26", "2024-07-02")}
LAST_DEV_DAY = SPLITS["threshold"][1]


@dataclass(frozen=True)
class Source:
    name: str
    path: Path
    sha256: str
    size: int
    first_day: dt.date
    n_days: int
    n_keys: int


TRAIN = Source("train", ROOT / "data" / "raw" / "train.parquet", SOURCE_SHA256, 106436287,
               dt.date(2024, 3, 28), 90, 50_000)
HOLDOUT = Source("holdout", ROOT / "data" / "expanded" / "raw" / "eval.parquet",
                 "1b118840664280c6b88bffc84c80ee1f54c05d911e354b7599e5da10995e960e", 8440124,
                 dt.date(2024, 6, 26), 7, 50_000)


@dataclass(frozen=True)
class Panel:
    """Complete daily values: one row per series (ascending key), one column per consecutive date."""
    keys: np.ndarray   # int64 [N, 3]
    dates: np.ndarray  # int32 ordinals [D]
    sales: np.ndarray  # float64 [N, D]
    stock: np.ndarray  # uint8 stockout hours [N, D]

    def __post_init__(self):
        n, d = len(self.keys), len(self.dates)
        require(n > 0 and d > 0 and self.keys.shape == (n, 3) and self.sales.shape == self.stock.shape == (n, d),
                "panel arrays do not have matching shapes")
        require(bool((np.diff(self.dates) == 1).all()), "dates must be consecutive, without gaps or repeats")
        require(bool((np.diff(_codes(self.keys)) > 0).all()), "keys must be unique and ascending")

    def _days(self, part):
        return Panel(self.keys, self.dates[part], self.sales[:, part], self.stock[:, part])

    def until(self, day):
        return self._days(slice(0, self.index(day) + 1))

    def since(self, day):
        return self._days(slice(self.index(day), None))

    def take(self, rows):
        return Panel(self.keys[rows], self.dates, self.sales[rows], self.stock[rows])

    def join(self, later):
        require(np.array_equal(self.keys, later.keys), "the two panels contain different series")
        return Panel(self.keys, np.concatenate([self.dates, later.dates]),
                     np.concatenate([self.sales, later.sales], axis=1),
                     np.concatenate([self.stock, later.stock], axis=1))

    def index(self, day):
        at = day.toordinal() - int(self.dates[0])
        require(0 <= at < len(self.dates), f"{day.isoformat()} is outside the available dates")
        return at


def _codes(keys):
    """One integer per key; ascending codes are ascending (city, store, product)."""
    return keys[:, 0] * 1_000_000 + keys[:, 1] * 1_000 + keys[:, 2]


# ---------- sources ----------

def read_panel(source, hourly=True, batch_size=65536):
    """Verify the pinned bytes, then every row: type, range, null, duplicate, coverage and hourly label."""
    import pyarrow.parquet as pq

    require(source.path.is_file(), f"{source.path} is missing")
    require(source.path.stat().st_size == source.size, f"{source.name}: size differs from the pinned file")
    require(sha256_file(source.path) == source.sha256, f"{source.name}: SHA256 does not match the pinned file")
    with pq.ParquetFile(source.path) as pf:
        types = {f.name: str(f.type) for f in pf.schema_arrow}
        for name, kind in TYPES.items():
            require(types.get(name) == kind, f"{source.name}: column {name} has type {types.get(name)}, not {kind}")
        n, d = source.n_keys, source.n_days
        require(pf.metadata.num_rows == n * d, f"{source.name}: {pf.metadata.num_rows} rows, expected {n * d}")
        codes = []
        for batch in pf.iter_batches(batch_size, columns=KEY):
            ids = []
            for name, column in zip(KEY, batch.columns):
                require(column.null_count == 0, f"null {name}")
                values = column.to_numpy()
                require(values.min() >= 0 and values.max() <= ID_MAX[name], f"{name} is out of range")
                ids.append(values)
            codes.append(_codes(np.stack(ids, axis=1)))
        unique = np.unique(np.concatenate(codes))
        require(len(unique) == n, f"{source.name}: {len(unique)} keys, expected {n}")
        sales, stock = np.empty((n, d)), np.empty((n, d), dtype=np.uint8)
        seen = np.zeros(n * d, dtype=bool)
        columns = ["dt", "sale_amount", "stock_hour6_22_cnt"] + ([HOURLY] if hourly else [])
        for rows, batch in zip(codes, pf.iter_batches(batch_size, columns=columns)):
            require(all(c.null_count == 0 for c in batch.columns), "null values found")
            text = batch.column(0).dictionary_encode()
            offsets = np.array([parse_day(v).toordinal() for v in text.dictionary.to_pylist()])
            offsets -= source.first_day.toordinal()
            require(bool(((offsets >= 0) & (offsets < d)).all()), f"{source.name}: unexpected date")
            amount, hours = batch.column(1).to_numpy(), batch.column(2).to_numpy()
            require(bool((np.isfinite(amount) & (amount >= 0)).all()), "sale_amount must be finite and >= 0")
            require(bool(((hours >= 0) & (hours <= 16)).all()), "stock_hour6_22_cnt must be in 0..16")
            at = np.searchsorted(unique, rows) * d + offsets[text.indices.to_numpy()]
            require(not seen[at].any() and len(np.unique(at)) == len(at), "duplicate key/date rows")
            seen[at] = True
            sales.reshape(-1)[at], stock.reshape(-1)[at] = amount, hours
            if hourly:
                _check_hourly(batch.column(3), hours)
        require(bool(seen.all()), f"{source.name}: a series does not cover every date")
    keys = np.stack([unique // 1_000_000, unique // 1_000 % 1_000, unique % 1_000], axis=1)
    dates = np.arange(d, dtype=np.int32) + source.first_day.toordinal()
    return Panel(keys, dates.astype(np.int32), sales, stock)


def _check_hourly(column, hours):
    lengths = np.diff(column.offsets.to_numpy())
    require(bool((lengths == 24).all()), "hourly status must have 24 values per row")
    values = column.flatten()
    require(values.null_count == 0, "null hourly status")
    status = values.to_numpy().reshape(-1, 24)
    require(bool(((status == 0) | (status == 1)).all()), "hourly status must be 0 or 1")
    require(np.array_equal(status[:, 6:22].sum(axis=1), hours), "hourly status 06:00-22:00 differs from the label count")


def audit(train=TRAIN, holdout=HOLDOUT):
    """Structure and integrity of both files. The receipt never contains a holdout outcome or a feature-label figure."""
    panels = {s.name: read_panel(s) for s in (train, holdout)}
    first, second = panels.values()
    joined = first.join(second)  # refuses other keys, gaps and overlap
    scale = float(np.median(second.sales.sum(axis=1) / np.maximum(first.sales[:, -second.sales.shape[1]:].sum(axis=1),
                                                                  1e-9)))
    receipt = {"revision": REVISION, "same_keys": True, "no_date_overlap": True, "consecutive_dates": True,
               "days": len(joined.dates),
               # fixed before the audit ran: median per-series week-to-week sales ratio between 0.5 and 2
               "sales_scale_compatible": bool(.5 <= scale <= 2)}
    for source, panel, role in ((train, first, "train"), (holdout, second, "holdout")):
        receipt[role] = {"sha256": source.sha256, "size": source.size, "rows": int(panel.stock.size),
                         "keys": len(panel.keys), "first_day": dt.date.fromordinal(int(panel.dates[0])).isoformat(),
                         "last_day": dt.date.fromordinal(int(panel.dates[-1])).isoformat(),
                         "hourly_labels_match": True}
    return receipt


def cohort_rows(keys, n):
    """Row numbers (ascending) of the first n keys by the B3 SHA-256 rule, and the digest of the ranked list."""
    require(0 < n <= len(keys), f"cannot select {n} of {len(keys)} series")
    ranked = rank_keys(map(tuple, keys.tolist()), n)
    position = {tuple(k): i for i, k in enumerate(keys.tolist())}
    return np.sort(np.array([position[k] for k in ranked])), keys_digest(ranked)


# ---------- features ----------

def features_on(panel, day):
    """Set B for every series on `day`, from the 28 days before it. Later columns are never read."""
    at = day.toordinal() - int(panel.dates[0])
    require(WARMUP <= at <= len(panel.dates), f"{day.isoformat()} needs the {WARMUP} days before it")
    sales, stock = panel.sales[:, at - WARMUP:at], panel.stock[:, at - WARMUP:at]
    out = stock > 0
    x = np.zeros((len(panel.keys), len(FEATURES["B"])))
    newest_first = out[:, ::-1]
    values = [sales[:, -1], stock[:, -1], sales[:, -7:].sum(axis=1) / 7, out[:, -7:].sum(axis=1),
              stock[:, -7:].sum(axis=1) / 7, stock[:, -14:].sum(axis=1) / 14, out[:, -14:].sum(axis=1),
              sales[:, -14:].sum(axis=1) / 14, sales.sum(axis=1) / 28, out.sum(axis=1),
              np.where(out.any(axis=1), newest_first.argmax(axis=1) + 1, WARMUP)]
    for i, column in enumerate(values):
        x[:, i] = column
    x[:, len(values) + day.weekday()] = 1
    return x


def split_days(name):
    first, last = SPLITS[name]
    return [first + dt.timedelta(i) for i in range((last - first).days + 1)]


def split(panel, name, feature_set="B"):
    """(features, outcomes) for one split, ordered by date then key."""
    days, n, columns = split_days(name), len(panel.keys), COLUMNS[feature_set]
    panel.index(days[-1])  # outcome dates must exist
    x, y = np.empty((len(days) * n, len(columns))), np.empty(len(days) * n, dtype=np.uint8)
    for i, day in enumerate(days):
        x[i * n:(i + 1) * n] = features_on(panel, day)[:, columns]
        y[i * n:(i + 1) * n] = panel.stock[:, panel.index(day)] > 0
    return x, y
