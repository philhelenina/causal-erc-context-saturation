#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_all_turnlevel_results.py (for TMLR)
Compatible with train_turnlevel_k_sweep.py output format:
columns = ['f1w', 'acc', ...]
"""
import os

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = (Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve() / Path("results/turnlevel_k_sweep"))
OUT = ROOT / "../visualizations"
OUT.mkdir(parents=True, exist_ok=True)

rows = []

for folder in ROOT.iterdir():
    if not folder.is_dir():
        continue

    comp = folder / "k_sweep_comprehensive.csv"
    meta = folder / "analysis_summary.json"
    if not comp.exists():
        continue

    try:
        df = pd.read_csv(comp)
        if df.empty or "f1w" not in df.columns or "acc" not in df.columns:
            print(f"[SKIP] Invalid or empty: {comp}")
            continue

        best = df.loc[df["f1w"].idxmax()]
        entry = {
            "config": folder.name,
            "best_k": best.get("k", None),
            "best_f1w": best.get("f1w", None),
            "best_acc": best.get("acc", None),
            "mean_f1w": df["f1w"].mean(),
            "mean_acc": df["acc"].mean(),
            "task": "6way" if "6way" in folder.name else "4way",
        }

        if meta.exists():
            try:
                meta_data = json.load(open(meta))
                entry["opt_k"] = json.dumps(meta_data.get("optimal_k_per_emotion", {}))
                entry["saturation_tau"] = json.dumps(meta_data.get("saturation_params", {}))
                entry["best_per_emotion"] = json.dumps(meta_data.get("best_performance_per_emotion", {}))
            except Exception as e:
                print(f"[WARN] Could not parse {meta}: {e}")

        rows.append(entry)

    except Exception as e:
        print(f"[WARN] Failed to read {folder.name}: {e}")

# Merge all results
df_all = pd.DataFrame(rows)
if df_all.empty:
    print("⚠️ No valid results found — check folder structure or CSV headers.")
    exit()

out_csv = OUT / "turnlevel_full_summary.csv"
df_all.to_csv(out_csv, index=False)
print(f"[OK] turnlevel_full_summary.csv saved → {out_csv}")

# Visualization
for task, subset in df_all.groupby("task"):
    if subset.empty:
        continue
    top10 = subset.sort_values("best_f1w", ascending=False).head(10)
    plt.figure(figsize=(10, 5))
    plt.barh(top10["config"], top10["best_f1w"], color="royalblue")
    plt.title(f"Top Configurations ({task})", fontsize=14, fontweight="bold")
    plt.xlabel("Best Weighted F1")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUT / f"top10_{task}_f1w.png", dpi=200)
    plt.close()
    print(f"[OK] Saved plot: top10_{task}_f1w.png")

# Markdown report
report = OUT / "turnlevel_summary_report.md"
with open(report, "w") as f:
    f.write("# 📊 Turn-Level Experiment Summary (4way & 6way)\n\n")
    for task in ["4way", "6way"]:
        subset = df_all[df_all["task"] == task]
        if subset.empty:
            continue
        f.write(f"## {task.upper()} Results\n\n")
        f.write(
            subset[["config", "best_k", "best_f1w", "best_acc", "mean_f1w", "mean_acc"]]
            .sort_values("best_f1w", ascending=False)
            .head(10)
            .to_markdown(index=False)
        )
        f.write("\n\n")

print(f"[OK] Markdown summary → {report}")
