"""Stage commands: python -m freshstock.expanded {audit,pilot,lock,fit,evaluate,verify}. Exit 0 pass, 1 refused, 2 usage."""
import argparse
import json
import sys
from pathlib import Path

from ..data import require, sha256_file
from . import data, experiment, pilot
from .resources import exit_with_diagnostic

FIT_TIMEOUT = 3600  # seconds for one model fit


def parser():
    p = argparse.ArgumentParser(prog="python -m freshstock.expanded", description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("audit", help="check both pinned source files; writes structure only")
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("pilot", help="training-only time and memory measurements")
    s.add_argument("--series", type=int, required=True)
    s.add_argument("--threads", type=lambda t: [int(v) for v in t.split(",")], required=True)
    s.add_argument("--repeats", type=int, default=1)
    s.add_argument("--profiles", type=lambda t: t.split(","), default=["light", "deep"])
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("lock", help="write the protocol before any tuning score exists")
    s.add_argument("--series", type=int, required=True)
    s.add_argument("--grid", choices=["full", "reduced"], required=True)
    s.add_argument("--calibration", choices=["sigmoid", "off"], required=True)
    s.add_argument("--forest-jobs", type=int, required=True)
    s.add_argument("--blas-threads", type=int, required=True)
    s.add_argument("--decision", type=Path, required=True, help="resource receipt behind the choice")
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("fit", help="tune, refit, calibrate and freeze; opens the training source only")
    s.add_argument("--protocol", type=Path, required=True)
    s.add_argument("--runs", type=Path, required=True)
    for name, text in (("evaluate", "score the fresh dates once"), ("verify", "check files and recompute metrics")):
        s = sub.add_parser(name, help=text)
        s.add_argument("--run", type=Path, required=True)
        s.add_argument("--freeze-sha256", required=True)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    try:
        if a.stage == "audit":
            experiment.write_json(a.output, data.audit(data.TRAIN, data.HOLDOUT))
        elif a.stage == "pilot":
            pilot.run(a.series, a.threads, a.output, a.repeats, a.profiles, exit_with_diagnostic(
                a.output.with_name("pilot_stopped.json")))
        elif a.stage == "lock":
            require(not a.output.exists(), f"{a.output.name} exists; a protocol lock is never overwritten")
            decision = {"receipt": a.decision.name, "receipt_sha256": sha256_file(a.decision)}
            experiment.write_json(a.output, experiment.build_protocol(
                data.TRAIN, data.HOLDOUT, a.series, a.grid, a.calibration,
                {"forest_n_jobs": a.forest_jobs, "blas_threads": a.blas_threads}, decision))
            print("protocol sha256", sha256_file(a.output))
        elif a.stage == "fit":
            run = experiment.fit(a.protocol, a.runs, data.TRAIN, exit_with_diagnostic(a.runs / "fit_stopped.json"),
                                 FIT_TIMEOUT)
            print("run", run.name)
            print("freeze sha256", sha256_file(run / "freeze.json"))
        elif a.stage == "evaluate":
            experiment.evaluate(a.run, a.freeze_sha256, data.TRAIN, data.HOLDOUT)
            print("evaluated", a.run.name)
        else:
            print(json.dumps(experiment.verify(a.run, a.freeze_sha256)))
        return 0
    except (ValueError, OSError, MemoryError, Warning) as error:
        print(f"{a.stage} refused or failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
