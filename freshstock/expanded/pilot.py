"""Training-only resource measurements. Writes time, memory and size; never a score or metric."""
import statistics
import tempfile
from pathlib import Path

import joblib
import numpy as np

from . import data, experiment
from .resources import Guard, snapshot

PARITY_ROWS = 2000


def _measure(family, s, params, x, y, threads, reference, scratch, repeats, stop):
    seconds, peak, low = [], 0, None
    for _ in range(repeats):
        with Guard(interval=.25, stop=stop) as guard:
            model = experiment.fit_model(family, s, params, x, y, threads)
        report = guard.report()
        seconds.append(report["seconds"])
        peak, low = max(peak, report["peak_additional_bytes"]), min(low or 2 ** 60, report["min_available_ram"])
    score = experiment.positive_scores(model, x[:PARITY_ROWS])
    reference.setdefault("score", score)
    path = Path(scratch) / "model.joblib"
    joblib.dump(model, path, compress=3)
    return {"family": family, "feature_set": s, "params": params, "threads": threads, "rows": len(y),
            "seconds": round(statistics.median(seconds), 3), "repeats": repeats, "peak_additional_bytes": peak,
            "min_available_ram": low, "file_bytes": path.stat().st_size, **experiment._size(model),
            "parity_max_abs_diff": float(np.abs(score - reference["score"]).max()),
            "flags_equal": bool(np.array_equal(score >= .5, reference["score"] >= .5))}


def run(n, thread_counts, output, repeats=1, profiles=("light", "deep"), stop=None):
    panel, digest = experiment._cohort(data.read_panel(data.TRAIN), n)
    runs, start = [], snapshot()
    with tempfile.TemporaryDirectory() as scratch:
        for s in ("A", "B"):
            x, y = data.split(panel, "initial_fit", s)
            for profile in profiles:
                for family in ("logistic", "forest"):
                    grid = experiment.GRIDS["full"][family]
                    params = grid[0 if profile == "light" else -1]
                    reference = {}
                    for t in thread_counts:
                        threads = {"forest_n_jobs": t, "blas_threads": t}
                        runs.append({"profile": profile, **_measure(family, s, params, x, y, threads, reference,
                                                                    scratch, repeats, stop)})
            del x, y
    experiment.write_json(Path(output), {"stage": "training-only resource pilot", "series": n, "cohort_digest": digest,
                                         "start": start, "end": snapshot(), "runs": runs})
    return runs
