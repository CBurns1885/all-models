import pandas as pd

p = r'C:\Users\Chris\OneDrive\My Documents\NewOnedrive\OneDrive\Desktop\Chris Code\all_models\outputs\tuning_preds_cache.parquet'
try:
    df = pd.read_parquet(p)
    print(f"Cache loaded: {df.shape}")
    if 'referee' in df.columns:
        types = df['referee'].dropna().apply(type).value_counts()
        print("referee types:", types.to_dict())
    else:
        print("No referee column in cache")
    print("Columns:", list(df.columns)[:20])
except Exception as e:
    print(f"Error reading cache: {e}")
