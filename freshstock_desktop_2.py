"""FreshStock 2.0 desktop program. `--self-test FILE` checks all test days offline and writes a receipt."""
import csv
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

START = time.perf_counter()

import numpy as np  # noqa: E402

from freshstock import data  # noqa: E402
from freshstock.expanded import bundle as bd, gui  # noqa: E402
from freshstock.expanded.metrics import ESTIMATORS, MODELS  # noqa: E402
from freshstock.expanded.resources import snapshot  # noqa: E402

try:
    from freshstock.expanded.pin import PIN
except ImportError:  # no verified release has been pinned yet
    PIN = None


def bundle_root():
    return Path(sys._MEIPASS if getattr(sys, "frozen", False) else data.ROOT) / "bundle"


def reports_dir():
    return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "FreshStock2" / "reports"


def load(root):
    data.require(PIN is not None, "this source has no pinned release")
    return bd.load(root, PIN)


def no_network(*args, **kwargs):
    raise OSError("network is blocked during the self-test")


def wait(app, done, seconds=300):
    end = time.perf_counter() + seconds
    while not done():
        data.require(time.perf_counter() < end, "the self-test timed out")
        app.update()
        time.sleep(0.01)
    app.update()


def walk(app, receipt):
    page, b = app.predict, app.bundle
    for name in app.nav:
        app.show(name)
        app.update()
    app.show("predict")
    counts, worst, seconds = {e: dict.fromkeys(("tp", "fp", "fn", "tn"), 0) for e in ESTIMATORS}, 0.0, []
    for i, day in enumerate(b.days):
        page.day.set(day)
        begin = time.perf_counter()
        page.generate_button.invoke()
        wait(app, lambda: not page.busy)
        seconds.append(round(time.perf_counter() - begin, 3))
        data.require(page.state == "forecast", page.message.cget("text"))
        worst = max(worst, *(float(np.abs(page.forecast.scores[m] - b.predictions["scores"][m][i]).max())
                             for m in MODELS))
        page.reveal_button.invoke()
        app.update()
        data.require(page.state == "revealed", page.message.cget("text"))
        for e in ESTIMATORS:
            for k in counts[e]:
                counts[e][k] += page.revealed.counts[e][k]
    for e in ESTIMATORS:
        saved = b.metrics["summary"]["estimators"][e]
        data.require(all(counts[e][k] == saved[k] for k in counts[e]), f"{e} counts differ from the saved result")
    page.export_button.invoke()
    wait(app, lambda: not page.busy)
    data.require(page.saved is not None, page.message.cget("text"))
    with open(page.saved / "ranking.csv", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    y = b.predictions["y_true"][-1]
    data.require(len(rows) == b.n and sum(int(r["y_true"]) for r in rows) == int(y.sum()), "the export is incomplete")
    receipt.update(days_replayed=len(b.days), forecast_seconds=seconds, max_score_difference=worst, counts=counts,
                   exported_rows=len(rows), report=page.saved.name)


def self_test(out):
    out.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"run": PIN and PIN["run_id"], "series": PIN and PIN["n_series"], "status": "failed", "problem": None,
               "frozen_program": bool(getattr(sys, "frozen", False)), "network_blocked": True,
               "python": platform.python_version(), "system": platform.platform()}
    real_socket, socket.socket = socket.socket, no_network
    app = None
    try:
        app = gui.App(bundle_root(), out.parent, load)
        wait(app, lambda: app.status != "checking")
        receipt.update(status=app.status, startup_seconds=round(time.perf_counter() - START, 2))
        data.require(app.status == "ready", getattr(app, "problem", "not ready"))
        walk(app, receipt)
        if sys.platform == "win32":
            receipt["peak_working_set"] = snapshot(out.parent)["peak_working_set"]
    except Exception as error:  # a receipt must exist for every failure
        receipt["status"] = "failed" if receipt["status"] == "ready" else receipt["status"]
        receipt["problem"] = str(error)
    finally:
        if app is not None:
            app.destroy()
        socket.socket = real_socket
    out.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    return 0 if receipt["status"] == "ready" and receipt["problem"] is None else 1


def main(argv):
    if argv[:1] == ["--self-test"] and len(argv) == 2:
        return self_test(Path(argv[1]))
    gui.main(bundle_root(), reports_dir(), load)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
