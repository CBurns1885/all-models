# dc_predict.py
from __future__ import annotations
from pathlib import Path
import pickle
import pandas as pd
from config import FEATURES_PARQUET, OUTPUT_DIR, log_header
from models_dc import fit_all, price_match

_DC2_CACHE_DIR = Path(__file__).parent / "outputs" / "dc2_params_cache"

def _dc2_cache_key(df: pd.DataFrame) -> str:
    dt = pd.to_datetime(df["Date"])
    return f"{str(dt.min())[:10]}_{str(dt.max())[:10]}_{len(df)}"

def _load_dc2_cache(key: str):
    p = _DC2_CACHE_DIR / f"dc2_{key}.pkl"
    if p.exists():
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return None

def _save_dc2_cache(key: str, params) -> None:
    try:
        _DC2_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_DC2_CACHE_DIR / f"dc2_{key}.pkl", "wb") as f:
            pickle.dump(params, f)
    except Exception:
        pass

def build_dc_for_fixtures(fixtures_csv: Path) -> Path:
    log_header("DC: fitting per-league parameters")
    base=pd.read_parquet(FEATURES_PARQUET)
    base=base.dropna(subset=["Date","HomeTeam","AwayTeam","FTHG","FTAG"]).copy()
    base["Date"]=pd.to_datetime(base["Date"])

    # Anti-leakage cutoff: only fit on matches strictly BEFORE the earliest
    # fixture being priced. For live predictions this filters nothing (all
    # history is in the past). For backtests over historical fixtures it is
    # essential — without it the DC attack/defence parameters (and the
    # recent-form multipliers inside fit_all) are estimated from the very
    # matches being "predicted" and from matches played after them.
    _fx_dates = pd.to_datetime(pd.read_csv(fixtures_csv)["Date"], errors="coerce")
    _cutoff = _fx_dates.min()
    if pd.notna(_cutoff):
        n_before = len(base)
        base = base[base["Date"] < _cutoff]
        if len(base) < n_before:
            print(f"  [CUTOFF] DC training restricted to {len(base):,}/{n_before:,} matches before {str(_cutoff)[:10]}")

    train_df=base[["League","Date","HomeTeam","AwayTeam","FTHG","FTAG"]]

    cache_key = _dc2_cache_key(train_df)
    params = _load_dc2_cache(cache_key)
    if params is not None:
        print(f"  [CACHE] Loaded DC2 params from disk ({cache_key})")
    else:
        params=fit_all(train_df)
        _save_dc2_cache(cache_key, params)
        print(f"  [CACHE] Saved DC2 params to disk ({cache_key})")

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
