r"""Build the portable FreshStock 2.0 program in a new release-2.0 folder. Nothing that exists is replaced.

Usage: .venv-build\Scripts\python.exe tools/build_release.py
Needs the committed pin (freshstock/expanded/pin.py) and the matching bundle/ folder.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from importlib.metadata import distribution
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from freshstock.expanded import bundle as bd  # noqa: E402
from freshstock.expanded.pin import PIN  # noqa: E402

VERSION = "2.0.0"
PYINSTALLER = "6.22.3"
RUNTIME = ["cloudpickle", "joblib", "numpy", "pandas", "python-dateutil", "pytz", "scikit-learn", "scipy",
           "threadpoolctl", "tzdata"]  # pyarrow is only for auditing the raw file and is not shipped
INPUTS = ["freshstock", "freshstock_desktop_2.py", "requirements.txt", "DATA_LICENSE.md", "tools"]


def run(*command, **kw):
    print(">", *[str(c) for c in command], flush=True)
    return subprocess.run([str(c) for c in command], check=True, **kw)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def third_party(folder):
    """License texts and a component list taken from the installed packages themselves."""
    folder.mkdir()
    parts = []
    for name in RUNTIME + ["pyinstaller"]:
        dist = distribution(name)
        texts = {}
        for f in dist.files:
            if ".dist-info" in f.parts[0] and any(w in f.name.upper() for w in ("LICENSE", "COPYING", "NOTICE")):
                texts["-".join([name] + [p for p in f.parts[1:] if p != "licenses"])] = f
        if not texts:
            raise SystemExit(f"no license text found for {name}")
        for target, f in texts.items():
            (folder / target).write_bytes(dist.locate_file(f).read_bytes())
        meta = dist.metadata
        license_name = meta.get("License-Expression") or (meta.get("License") or "see text").splitlines()[0]
        license_name = "see text" if license_name.startswith("Copyright") else license_name
        parts.append({"name": name, "version": dist.version, "license": license_name,
                      "texts": sorted(texts),
                      "role": "bootloader and loader code" if name == "pyinstaller" else "runtime"})
    base = Path(sys.base_prefix)
    for name, version, source in (("python", sys.version.split()[0], base / "LICENSE.txt"),
                                  ("tcl-tk", "8.6.15", base / "tcl" / "tk8.6" / "license.terms")):
        shutil.copyfile(source, folder / f"{name}-LICENSE.txt")
        parts.append({"name": name, "version": version, "license": "see text", "texts": [f"{name}-LICENSE.txt"],
                      "role": "runtime"})
    (folder / "components.json").write_text(json.dumps(parts, indent=1), encoding="utf-8")


def self_test(program, work, name):
    """Run the packaged program from a path with spaces, offline, without Python on PATH."""
    place = work / name / "FreshStock"
    shutil.copytree(program, place)
    env = {k: os.environ[k] for k in ("SYSTEMROOT", "TEMP", "TMP", "LOCALAPPDATA", "USERPROFILE")}
    env["PATH"] = os.path.join(os.environ["SYSTEMROOT"], "System32")
    receipt = work / name / "self-test.json"
    run(place / "FreshStock.exe", "--self-test", receipt, env=env, cwd=work, timeout=900)
    result = json.loads(receipt.read_text(encoding="utf-8"))
    assert (result["status"], result["problem"], result["frozen_program"], result["days_replayed"], result["run"]) == (
        "ready", None, True, 7, PIN["run_id"]), result
    return result


def main():
    commit = run("git", "rev-parse", "HEAD", cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = run("git", "status", "--porcelain", "--", *INPUTS, cwd=ROOT, capture_output=True, text=True).stdout
    if dirty.strip():
        raise SystemExit("Commit source changes first:\n" + dirty)
    installed = {n.lower(): distribution(n).version for n in RUNTIME + ["pyinstaller", "pyarrow"]}
    for line in (ROOT / "requirements.txt").read_text().split():
        name, version = line.split("==")
        if installed.get(name) != version:
            raise SystemExit(f"{name} {version} is locked but {installed.get(name)} is installed")
    if installed["pyinstaller"] != PYINSTALLER:
        raise SystemExit(f"PyInstaller {PYINSTALLER} is required")
    bd.load(ROOT / "bundle", PIN)

    out, work = ROOT / "release-2.0" / f"{VERSION}-{commit[:12]}", ROOT / ".build" / "expanded" / f"release-{commit[:12]}"
    for folder in (work, out):
        if folder.exists():
            raise SystemExit(f"{folder} exists; nothing is overwritten")
    work.mkdir(parents=True)
    data = [arg for name in bd.FILES for arg in ("--add-data", f"{ROOT / 'bundle' / name};bundle")]
    meta = [arg for name in ("scikit-learn", "numpy", "scipy", "pandas", "joblib", "threadpoolctl")
            for arg in ("--copy-metadata", name)]
    skip = [arg for name in ("pyarrow", "matplotlib", "PIL", "IPython", "pytest", "PyQt5", "PySide6")
            for arg in ("--exclude-module", name)]
    with open(work / "pyinstaller.log", "w", encoding="utf-8") as log:
        run(sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--windowed", "--name",
            "FreshStock", "--paths", ROOT, "--distpath", work / "dist", "--workpath", work / "work", "--specpath", work,
            *data, *meta, *skip, ROOT / "freshstock_desktop_2.py", stdout=log, stderr=subprocess.STDOUT)
    program = work / "dist" / "FreshStock"
    shutil.copyfile(ROOT / "tools" / "release_readme.txt", program / "README.txt")
    shutil.copyfile(ROOT / "DATA_LICENSE.md", program / "DATA_LICENSE.md")
    third_party(program / "THIRD_PARTY")
    bd.load(program / "_internal" / "bundle", PIN)
    if any("pyarrow" in p.name.lower() or "arrow.dll" in p.name.lower() for p in program.rglob("*")):
        raise SystemExit("pyarrow must not be in the portable program")
    tests = [self_test(program, work, name) for name in ("first run with spaces", "second run with spaces")]

    stage = out.with_name(out.name + ".partial")
    stage.mkdir(parents=True)
    portable = stage / f"FreshStock-{VERSION}-portable.zip"
    with zipfile.ZipFile(portable, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(program.rglob("*")):
            if p.is_file():
                z.write(p, Path("FreshStock") / p.relative_to(program))
    files = [p for p in program.rglob("*") if p.is_file()]
    manifest = {"version": VERSION, "source_commit": commit, "run": PIN["run_id"], "series": PIN["n_series"],
                "freeze_sha256": PIN["freeze_sha256"], "python": sys.version.split()[0], "pyinstaller": PYINSTALLER,
                "installer_built": False, "signed": False, "unpacked_files": len(files),
                "unpacked_bytes": sum(p.stat().st_size for p in files),
                "artifacts": {portable.name: {"bytes": portable.stat().st_size, "sha256": sha(portable)}},
                "program_sha256": sha(program / "FreshStock.exe"), "bundle": PIN["files"], "self_tests": tests}
    (stage / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (stage / "SHA256SUMS.txt").write_text(f"{sha(portable)}  {portable.name}\n", encoding="utf-8", newline="\n")
    (stage / "FILES.sha256").write_text("".join(f"{sha(p)}  {p.relative_to(program).as_posix()}\n" for p in sorted(files)),
                                        encoding="utf-8", newline="\n")
    shutil.copyfile(program / "THIRD_PARTY" / "components.json", stage / "components.json")
    stage.rename(out)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
