"""Data contract: source audit, fixed cohort, leakage-safe features and date splits (B3 v1)."""
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
SOURCE_PATHS = (ROOT / "data" / "raw" / "train.parquet", ROOT / "Final Project" / "train.parquet")
SOURCE_URL = ("https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K/resolve/"
              "08c1fab7f9257bc73679d415d65d644165d351d4/data/train.parquet")
SOURCE_SHA256 = "6706832db892bbae4969c19d87e07975d2543d2ba7d7d4756360654785de5a3d"
COHORT_DIGEST = "1595c82439b79a90b5f68f231e3690605f3f6b3d59d0e1c612c867b95d8d1f7d"
HISTOGRAM = [2507994, 152309, 120358, 142575, 199629, 242299, 203376, 142210, 111818, 102507,
             106773, 108853, 84517, 54059, 26778, 12154, 181791]
KEY = ["city_id", "store_id", "product_id"]
COLUMNS = KEY + ["dt", "sale_amount", "stock_hour6_22_cnt"]
ID_MAX = {"city_id": 17, "store_id": 897, "product_id": 864, "stock_hour6_22_cnt": 16}
NUMERIC = ["sales_lag1", "stockout_hours_lag1", "sales_mean7", "stockout_days7"]
FEATURES = NUMERIC + ["weekday"]
LOOKBACK = 7
FIRST_DAY = dt.date(2024, 3, 28)
N_DAYS = 90
SPLITS = {"train": ("2024-04-04", "2024-05-26", 5300, 2403),
          "validation": ("2024-05-27", "2024-06-10", 1500, 535),
          "test": ("2024-06-11", "2024-06-25", 1500, 682)}


class DataError(ValueError):
    """Input breaks the data contract; nothing is cleaned or imputed silently."""


def require(ok, message):
    if not ok:
        raise DataError(message)


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def find_source():
    return next((p for p in SOURCE_PATHS if p.is_file()), None)


def parse_day(text):
    try:
        day = dt.date.fromisoformat(text)
        require(day.isoformat() == text, f"date {text!r} is not YYYY-MM-DD")
        return day
    except (TypeError, ValueError):
        raise DataError(f"date {text!r} is not a valid YYYY-MM-DD date") from None


def validate(df):
    """Check types, domains, uniqueness and daily continuity; return canonically sorted rows."""
    require(set(COLUMNS) <= set(df.columns), f"missing columns: {sorted(set(COLUMNS) - set(df.columns))}")
    df = df[COLUMNS]
    require(not df.isna().any().any(), "null values found")
    for col, top in ID_MAX.items():
        require(pd.api.types.is_integer_dtype(df[col]), f"{col} must be integer")
        require(df[col].between(0, top).all(), f"{col} must be in 0..{top}")
    df = df.astype({col: "int64" for col in ID_MAX})
    sales = df["sale_amount"].to_numpy(dtype=float)
    require((np.isfinite(sales) & (sales >= 0)).all(), "sale_amount must be finite and >= 0")
    days = df["dt"].map({d: parse_day(d).toordinal() for d in df["dt"].unique()})
    require(not df.duplicated(KEY + ["dt"]).any(), "duplicate key/date rows")
    df = df.assign(_day=days).sort_values(KEY + ["_day"]).reset_index(drop=True)
    step = df.groupby(KEY)["_day"].diff().dropna()
    require((step == 1).all(), "calendar gap inside a series")
    return df.drop(columns="_day")


def rank_keys(keys, n=100):
    """Deterministic sample rule: ascending SHA256 of 'city,store,product', tie-break on the key."""
    def rank(k):
        return hashlib.sha256(",".join(map(str, k)).encode("ascii")).hexdigest(), k
    return sorted(keys, key=rank)[:n]


def keys_digest(ranked):
    return hashlib.sha256("".join(",".join(map(str, k)) + "\n" for k in ranked).encode("ascii")).hexdigest()


def audit_source(path):
    """Read-only check of all 4.5M rows; returns (validated 100-series cohort, receipt)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    require(sha256_file(path) == SOURCE_SHA256, f"{path}: SHA256 does not match the pinned source file")
    pf = pq.ParquetFile(path)
    day_bit = {(FIRST_DAY + dt.timedelta(i)).isoformat(): 1 << i for i in range(N_DAYS)}
    masks, hist, rows, keep = {}, np.zeros(17, dtype=np.int64), 0, {}
    for batch in pf.iter_batches(batch_size=65536, columns=COLUMNS):
        require(all(c.null_count == 0 for c in batch.columns), "null values in source")
        arrays = [batch.column(i).to_numpy(zero_copy_only=False) for i in range(6)]
        for col in ID_MAX:
            a = arrays[COLUMNS.index(col)]
            require(a.min() >= 0 and a.max() <= ID_MAX[col], f"{col} out of range in source")
        require((np.isfinite(arrays[4]) & (arrays[4] >= 0)).all(), "invalid sale_amount in source")
        rows += batch.num_rows
        hist += np.bincount(arrays[5], minlength=17)
        for key, day in zip(zip(*(a.tolist() for a in arrays[:3])), arrays[3]):
            require(day in day_bit, f"unexpected date {day}")
            seen = masks.get(key, 0)
            require(not seen & day_bit[day], f"duplicate row for {key} {day}")
            masks[key] = seen | day_bit[day]
    require(rows == 4_500_000 and len(masks) == 50_000, f"unexpected size: {rows} rows, {len(masks)} keys")
    require(set(masks.values()) == {(1 << N_DAYS) - 1}, "a series does not cover all 90 dates")
    require(hist.tolist() == HISTOGRAM, "stockout-hour histogram differs from the audited receipt")
    ranked = rank_keys(masks)
    require(keys_digest(ranked) == COHORT_DIGEST, "cohort digest mismatch")

    chosen = pd.MultiIndex.from_tuples(ranked, names=KEY)
    parts = []
    for batch in pf.iter_batches(batch_size=65536, columns=COLUMNS):
        df = pa.Table.from_batches([batch]).to_pandas()
        df = df[pd.MultiIndex.from_frame(df[KEY]).isin(chosen)]
        if len(df):
            parts.append(df)
    cohort = validate(pd.concat(parts, ignore_index=True))
    receipt = {"source_url": SOURCE_URL, "source_sha256": SOURCE_SHA256, "rows": rows, "keys": len(masks),
               "stockout_histogram": hist.tolist(), "cohort_digest": COHORT_DIGEST, "cohort_rows": len(cohort),
               "sample_rule": "first 100 keys by ascending SHA256 of 'city_id,store_id,product_id'",
               "ranked_keys": [list(k) for k in ranked]}
    check_cohort(cohort)
    return cohort, receipt


def check_cohort(cohort):
    keys = list(cohort[KEY].drop_duplicates().itertuples(index=False, name=None))
    require(len(cohort) == 9000 and keys_digest(rank_keys(keys)) == COHORT_DIGEST, "not the approved cohort")
    for name, (_, _, rows, positive) in SPLITS.items():
        x, y = partition(cohort, name)
        require((len(x), int(y.sum())) == (rows, positive), f"{name} counts differ from the protocol")


def save_cohort(cohort, receipt, folder=DERIVED):
    folder.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(folder / "cohort.csv", index=False, lineterminator="\n")
    receipt = {**receipt, "cohort_sha256": sha256_file(folder / "cohort.csv")}
    (folder / "cohort_manifest.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    return receipt


def load_cohort(folder=DERIVED):
    path, manifest = folder / "cohort.csv", folder / "cohort_manifest.json"
    require(path.is_file() and manifest.is_file(), f"{path} is missing. Run: python -m freshstock audit")
    receipt = json.loads(manifest.read_text(encoding="utf-8"))
    require(sha256_file(path) == receipt["cohort_sha256"], f"{path}: SHA256 changed. Re-run the audit step")
    cohort = validate(pd.read_csv(path, float_precision="round_trip"))
    check_cohort(cohort)
    return cohort, receipt


def features_at(history, target):
    """Features for every series on `target`, using only rows dated before it."""
    day = parse_day(target)
    require(not (history["dt"] >= target).any(), "history contains rows on or after the target date")
    history = validate(history)
    window = [(day - dt.timedelta(i)).isoformat() for i in range(1, LOOKBACK + 1)]
    recent = history[history["dt"].isin(window)]
    sales = recent.pivot(index=KEY, columns="dt", values="sale_amount").reindex(columns=window)
    stock = recent.pivot(index=KEY, columns="dt", values="stock_hour6_22_cnt").reindex(columns=window)
    n_keys = len(history[KEY].drop_duplicates())
    require(n_keys > 0 and len(sales) == n_keys and not sales.isna().any().any(),
            f"history must cover all {LOOKBACK} days before {target} for every series")
    out = sales.index.to_frame(index=False)
    out["dt"] = target
    out["sales_lag1"] = sales[window[0]].to_numpy()
    out["stockout_hours_lag1"] = stock[window[0]].to_numpy().astype(float)
    out["sales_mean7"] = sales.to_numpy().sum(axis=1) / LOOKBACK
    out["stockout_days7"] = (stock.to_numpy() > 0).sum(axis=1).astype(float)
    out["weekday"] = day.weekday()
    return out


def partition(cohort, name):
    """(feature rows, outcomes) for one split, ordered by date then key."""
    first, last = (parse_day(d) for d in SPLITS[name][:2])
    days = [(first + dt.timedelta(i)).isoformat() for i in range((last - first).days + 1)]
    x = pd.concat([features_at(cohort[cohort["dt"] < d], d) for d in days], ignore_index=True)
    return x, outcomes(cohort, x)


def outcomes(cohort, rows):
    """y = 1 if any stockout hour was recorded on the row's date (joined after features exist)."""
    joined = rows[KEY + ["dt"]].merge(cohort, on=KEY + ["dt"], how="left", validate="one_to_one")
    require(not joined["stock_hour6_22_cnt"].isna().any(), "missing outcome row")
    return (joined["stock_hour6_22_cnt"] > 0).astype(int).to_numpy()
