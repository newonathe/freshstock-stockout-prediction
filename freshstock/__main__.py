"""Command line: python -m freshstock {audit,train,evaluate,demo}"""
import argparse
import sys
from pathlib import Path

from . import data, demo, experiment as ex
from .data import DataError, require


def newest(runs, marker, step):
    found = sorted(p.parent for p in runs.glob(f"*/{marker}") if not p.parent.name.endswith((".failed", ".partial")))
    require(found, f"no usable run in {runs}. Run: python -m freshstock {step}")
    return found[-1]


def cmd_audit(args):
    source = args.source or data.find_source()
    require(source and source.is_file(), "train.parquet not found. Put it in data/raw/train.parquet or use --source. "
                                         f"Download: {data.SOURCE_URL}")
    print(f"Checking {source} (read-only) ...")
    cohort, receipt = data.audit_source(source)
    receipt = data.save_cohort(cohort, receipt)
    print(f"OK: {receipt['rows']} rows, {receipt['keys']} series, source SHA256 {receipt['source_sha256']}")
    print(f"OK: cohort {receipt['cohort_rows']} rows, digest {receipt['cohort_digest']}")
    for name, (a, b, rows, positive) in data.SPLITS.items():
        print(f"    {name:10} {a} to {b}: {rows} rows, {positive} with a stockout")


def cmd_train(args):
    cohort, receipt = data.load_cohort()
    run = ex.train(cohort, receipt, args.runs)
    manifest = ex.read_json(run / "manifest.json")
    validation = ex.read_json(run / "validation.json")
    for name in ex.MODELS:
        v = validation[name]
        print(f"{ex.TITLES[name]}: fitted in {manifest['fit_seconds'][name]:.3f}s, "
              f"validation AP {v['average_precision']:.4f}, frozen threshold {v['threshold']}")
    print(f"Frozen run: {run}\nNext: python -m freshstock evaluate")


def cmd_evaluate(args):
    run = args.run or newest(args.runs, "manifest.json", "train")
    cohort, _ = data.load_cohort()
    print(demo.metrics_text(ex.evaluate(run, cohort)))
    print(f"Verified and saved in {run}")


def ask_date(days):
    while True:
        answer = input(f"Date to replay, {days[0]} to {days[-1]} (q to quit): ").strip()
        if answer.lower() == "q" or answer in days:
            return answer if answer in days else None
        print(f"Please choose a date from {days[0]} to {days[-1]}, written like {days[0]}.")


def show(args, run, cohort, metrics, day, reveal):
    r = demo.replay(run, cohort, day, args.sort)
    print("\n" + demo.forecast_text(r) + "\n")
    reveal()
    print(demo.outcome_text(r) + "\n")
    print(demo.metrics_text(metrics) + "\n")
    args.reports.mkdir(parents=True, exist_ok=True)
    path = args.reports / f"replay-{day}.html"
    path.write_text(demo.render_html(r, metrics), encoding="utf-8")
    print(f"Report: {path.resolve()}")


def cmd_demo(args):
    run = args.run or newest(args.runs, "evaluation.json", "train, then evaluate")
    cohort, _ = data.load_cohort()
    metrics, frozen = ex.load_evaluation(run)
    if args.date:
        return show(args, run, cohort, metrics, args.date, lambda: None)
    days = sorted(frozen["dt"].unique())
    while day := ask_date(days):
        show(args, run, cohort, metrics, day,
             lambda: input("Press Enter to reveal what was actually observed: "))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m freshstock", description="Next-day stockout risk comparison")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, func in (("audit", cmd_audit), ("train", cmd_train), ("evaluate", cmd_evaluate), ("demo", cmd_demo)):
        p = commands.add_parser(name)
        p.set_defaults(func=func)
        p.add_argument("--runs", type=Path, default=data.ROOT / "runs", help="folder holding runs")
        p.add_argument("--reports", type=Path, default=data.ROOT / "reports", help="folder for HTML reports")
        if name == "audit":
            p.add_argument("--source", type=Path, help="path to the original train.parquet")
        if name in ("evaluate", "demo"):
            p.add_argument("--run", type=Path, help="run folder (default: newest)")
        if name == "demo":
            p.add_argument("--date", help="YYYY-MM-DD; omit to be asked")
            p.add_argument("--sort", choices=ex.MODELS, default="logistic", help="model used to order the list")
    args = parser.parse_args(argv)
    try:
        args.func(args)
        return 0
    except DataError as error:
        print(f"STOPPED: {error}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nStopped by user.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
