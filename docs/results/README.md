# Frozen results

Machine-written records of run `20261125T134353364564`. They are never edited; tests compare them with the pinned model bundle by SHA-256.

| File | Contents |
|---|---|
| `E2_RESOURCE_DECISION.json` | Measured time, memory and disk behind the cohort size, before any score existed |
| `E2_PROTOCOL_LOCK.json` | Splits, features, candidates, rules, versions and source hashes, locked before tuning |
| `E3_CANDIDATES.json` | Tuning scores of the 12 settings |
| `E4_FREEZE.json` | Selected models, calibration, thresholds and file hashes, frozen before the holdout |
| `E5_METRICS.json` | Holdout metrics, scored once |
| `E5_EVALUATION.json` | When and how the holdout was scored, with the hashes of every output |

## Check the published run

Download `frozen_run_2.0.zip` from the project's [Releases](https://github.com/newonathe/freshstock-stockout-prediction/releases), extract it into the repository root, place the two publisher files ([DATA_LICENSE.md](../../DATA_LICENSE.md)), then:

```powershell
# Recompute all metrics from the saved rows and check every file hash
.venv\Scripts\python.exe -m freshstock.expanded verify --run runs-expanded\20261125T134353364564 --freeze-sha256 a151d11db0b73adfd019002e739147cd20e01666c9e54c8896ee364ebde516a2

# The same results without any project code (237 checks)
.venv\Scripts\python.exe tools\verify_holdout.py runs-expanded\20261125T134353364564 data\raw\train.parquet data\expanded\raw\eval.parquet
```

## Notes

- `code_commit` values name commits of the original development history, which is kept privately. The five files listed under `code_sha256` in the lock are the same here, so the link to the code can be checked by hash.
- "B3" in some records and source files is the internal name of the earlier 100-series pilot protocol.
