# dc_predict.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
from config import FEATURES_PARQUET, OUTPUT_DIR, log_header
from models_dc import fit_all, price_match

def build_dc_for_fixtures(fixtures_csv: Path) -> Path:
    log_header("DC: fitting per-league parameters")
    base=pd.read_parquet(FEATURES_PARQUET)
    base=base.dropna(subset=["Date","HomeTeam","AwayTeam","FTHG","FTAG"]).copy()
    base["Date"]=pd.to_datetime(base["Date"])
    params=fit_all(base[["League","Date","HomeTeam","AwayTeam","FTHG","FTAG"]])
    fx=pd.read_csv(fixtures_csv); fx["Date"]=pd.to_datetime(fx["Date"])
    rows=[]
    for _,r in fx.iterrows():
        dc={}
        if r["League"] in params:
            try:
                dc=price_match(params[r["League"]],r["HomeTeam"],r["AwayTeam"])
            except Exception as e:
                print(f"  [WARN] DC price_match failed for {r.get('HomeTeam','?')} vs {r.get('AwayTeam','?')}: {e}")
                dc={}
        if not dc:
            print(f"  [INFO] No DC for {r.get('HomeTeam','?')} vs {r.get('AwayTeam','?')} ({r.get('League','?')})")
        rows.append(dc)
    out=pd.concat([fx.reset_index(drop=True),pd.DataFrame(rows)],axis=1)
    out_path=OUTPUT_DIR/"dc_probabilities.csv"
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    out.to_csv(out_path,index=False)
    print(f"Wrote DC probabilities -> {out_path}")
    return out_path

if __name__=="__main__":
    import argparse; ap=argparse.ArgumentParser()
    ap.add_argument("--fixtures_csv",type=str,required=True); args=ap.parse_args()
    build_dc_for_fixtures(Path(args.fixtures_csv))
