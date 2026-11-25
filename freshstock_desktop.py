"""FreshStock desktop program. `--self-test FILE` runs the whole June 18 walkthrough offline and writes a receipt."""
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

START = time.perf_counter()

from freshstock import data, experiment as ex, gui, portfolio as pf  # noqa: E402


def bundle_root():
    return Path(sys._MEIPASS) if getattr(sys, "frozen", False) else data.ROOT


def reports_dir():
    return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "FreshStock" / "reports"


def no_network(*args, **kwargs):
    raise OSError("network is blocked during the self-test")


def self_test(out):
    out.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"run": pf.RUN_ID, "frozen_program": bool(getattr(sys, "frozen", False)), "program": sys.executable,
               "python": platform.python_version(), "system": platform.platform(),
               "libraries": {n: ex.version(n) for n in ("scikit-learn", "numpy", "scipy", "pandas", "joblib")}}
    real_socket, socket.socket = socket.socket, no_network
    app = gui.App(bundle_root(), out.parent)
    try:
        while app.status == "checking":
            app.update()
            time.sleep(0.01)
        app.update()
        receipt.update(status=app.status, startup_seconds=round(time.perf_counter() - START, 2))
        if app.status == "ready":
            page = app.predict
            for name in app.nav:
                app.show(name)
                app.update()
            app.show("predict")
            for day in app.bundle.days:  # each call fails unless live scores equal the frozen scores
                pf.replay_day(app.bundle, day)
            receipt["days_replayed"] = len(app.bundle.days)
            page.day.set("2024-06-18")
            start = time.perf_counter()
            page.generate_button.invoke()
            receipt["forecast_seconds"] = round(time.perf_counter() - start, 3)
            page.reveal_button.invoke()
            app.update()
            receipt["summary"] = {n: dict(zip(("tp", "fn", "fp", "tn"), c)) for n, c in page.outcomes.counts.items()}
            page.export_button.invoke()
            receipt["report_saved"] = page.saved is not None and page.saved.is_file()
        else:
            receipt["message"] = app.problem
    finally:
        app.destroy()
        socket.socket = real_socket
    out.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    return 0 if receipt["status"] == "ready" and receipt["report_saved"] else 1


def main(argv):
    if argv[:1] == ["--self-test"] and len(argv) == 2:
        return self_test(Path(argv[1]))
    gui.main(bundle_root(), reports_dir())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
