# ingest_local_run.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
from config import DATA_DIR, log_header
from progress_utils import Timer

MIN_COLS = ["Date","HomeTeam","AwayTeam","FTHG","FTAG"]  # core columns that must exist

def _validate_csv(p: Path) -> dict:
    try:
        df = pd.read_csv(p)
        ok = all(c in df.columns for c in MIN_COLS)
        return {"file": p.name, "rows": int(len(df)), "ok": bool(ok)}
    except Exception as e:
        return {"file": p.name, "rows": 0, "ok": False, "error": str(e)}

def ingest_local_csvs() -> Path:
    raw = DATA_DIR / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    with Timer("Validating raw CSVs"):
        paths = sorted(raw.glob("*.csv"))
        report = [_validate_csv(p) for p in paths]

    # Handle empty case gracefully
    if not report:
        man = pd.DataFrame(columns=["file","rows","ok","error"])
    else:
        man = pd.DataFrame(report)
        for col in ["file","rows","ok","error"]:
            if col not in man.columns:
                man[col] = None
        man = man.sort_values(["ok","file"], ascending=[False, True])

    out = DATA_DIR / "raw_manifest.csv"
    man.to_csv(out, index=False)
    print(f"Wrote manifest -> {out} (files found: {len(report)})")
    return out

if __name__ == "__main__":
    log_header("Ingest local Football-Data CSVs")
    ingest_local_csvs()
