import pandas as pd
from pathlib import Path

DATA = Path("data")

for task in ["4way", "6way"]:
    dfs = [pd.read_csv(DATA / f"iemocap_{task}_data" / f"{split}_{task}_with_minus_one.csv")
           for split in ["train", "val", "test"]]
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_csv(DATA / f"iemocap_{task}_data" / f"all_{task}_with_minus_one.csv", index=False)
    print(f"[OK] merged {task} → {len(df_all)} rows")
