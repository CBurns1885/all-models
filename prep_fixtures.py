# prep_fixtures.py
from pathlib import Path
import pandas as pd
from config import OUTPUT_DIR, log_header
from progress_utils import Timer

REQ = ["Date","League","HomeTeam","AwayTeam"]

def xlsx_to_csv(xlsx_path: Path) -> Path:
    with Timer("Convert fixtures xlsx -> csv"):
        df = pd.read_excel(xlsx_path)
        missing = [c for c in REQ if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in fixtures: {missing}")
        df["Date"] = pd.to_datetime(df["Date"])
        out = OUTPUT_DIR / "upcoming_fixtures.csv"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Saved {out}")
        return out

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=str, required=True)
    args = ap.parse_args()
    log_header("Prepare fixtures")
    xlsx_to_csv(Path(args.xlsx))
