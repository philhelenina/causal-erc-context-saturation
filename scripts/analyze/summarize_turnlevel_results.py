#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_turnlevel_results.py
- Summarizes all turn-level K-sweep results (4way, 6way, SenticNet)
- Aggregates metrics (Acc, F1w, MI, ΔMI) and exports unified CSV + plot
"""
import os

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = (Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve() / Path("results/turnlevel_k_sweep"))
OUT_DIR = ROOT / "../visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_all_results(root):
    rows = []
    for csv in root.rglob("k_sweep_comprehensive.csv"):
        parts = csv.relative_to(root).parts
        tag = parts[0] if len(parts) > 0 else "unknown"
        try:
            df = pd.read_csv(csv)
            df["config"] = tag
            rows.append(df)
        except Exception as e:
            print(f"[WARN] Failed to read {csv}: {e}")
    return pd.concat(rows, ignore_index=True)

print("[INFO] Loading results...")
df = load_all_results(ROOT)
print(f"[INFO] Loaded {len(df)} rows from {df['config'].nunique()} configs")

# Compute summary per config
summary = df.groupby("config")[["accuracy","f1_weighted","mi_ctx","delta_mi"]].mean().reset_index()
summary = summary.sort_values("f1_weighted", ascending=False)
summary.to_csv(OUT_DIR / "turnlevel_summary_table.csv", index=False)
print(f"[OK] Summary saved → {OUT_DIR/'turnlevel_summary_table.csv'}")

# Plot top 8 configs
top = summary.head(8)
plt.figure(figsize=(10,5))
plt.barh(top["config"], top["f1_weighted"], color="skyblue")
plt.xlabel("Weighted F1")
plt.title("Top Turn-Level Configurations (mean F1)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(OUT_DIR / "turnlevel_summary_bar.png", dpi=200)
plt.close()
print(f"[OK] Plot saved → {OUT_DIR/'turnlevel_summary_bar.png'}")
