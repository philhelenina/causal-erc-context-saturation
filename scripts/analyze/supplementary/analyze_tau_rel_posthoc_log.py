#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_tau_rel_posthoc_log.py
- Collect τ_rel values from 4way & 6way normalized experiments
- Apply log normalization to τ_rel
- Perform ANOVA / Kruskal–Wallis / pairwise post-hoc (Tukey + Wilcoxon)
"""
import os

import json, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import f_oneway, kruskal, wilcoxon
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import itertools

HOME = Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()
RES_DIR = HOME / "results" / "turnlevel_k_sweep_norm_per_emotion"

# ==== 실험 폴더 목록 (4way + 6way 모두) ====
COMBOS = [
    "6way_sr-sentic-fused-alpha010_avg_last4_mean",
    "6way_sr-sentic-fused-alpha010_avg_last4_wmean_pos_rev",
    "6way_sr-sentic-fused-alpha010_avg_last4_wmean_idf",
    "6way_sr-sentic-fused-alpha015_avg_last4_mean",
    "4way_sr-sentic-fused-alpha010_avg_last4_mean",
    "4way_sr-sentic-fused-alpha010_avg_last4_wmean_pos_rev",
    "4way_sr-sentic-fused-alpha010_avg_last4_wmean_idf",
]

# ==== 1) 데이터 수집 ====
records = []
for combo in COMBOS:
    f = RES_DIR / combo / "tau_rel_per_emotion.json"
    if not f.exists():
        print(f"[SKIP] {f} missing")
        continue
    data = json.load(open(f))
    for emo, tau in data.items():
        records.append(dict(task=combo.split("_")[0], combo=combo, emotion=int(emo), tau_rel=tau))
df = pd.DataFrame(records)
if df.empty:
    raise SystemExit("❌ No τ_rel data found. Run train_turnlevel_k_sweep_norm_per_emotion.py first.")
print("\n[INFO] τ_rel data collected:")
print(df.head())

# ==== 2) 로그 정규화 ====
df["tau_log"] = np.log(np.abs(df["tau_rel"]) + 1e-6)
print("\n[INFO] Added log-normalized τ column (tau_log).")

# ==== 3) ANOVA / Kruskal–Wallis (log τ 기준) ====
emotions = sorted(df["emotion"].unique())
grouped = [df[df["emotion"]==e]["tau_log"].dropna().values for e in emotions]

anova_F, anova_p = f_oneway(*grouped)
kw_H, kw_p = kruskal(*grouped)

print(f"\n[ANOVA-log] F={anova_F:.4f}, p={anova_p:.4e}")
print(f"[Kruskal–Wallis-log] H={kw_H:.4f}, p={kw_p:.4e}")

# ==== 4) Post-hoc (Tukey HSD + Wilcoxon pairwise) ====
out_dir = RES_DIR / "posthoc_stats_log"
out_dir.mkdir(parents=True, exist_ok=True)

# Tukey HSD
try:
    tukey = pairwise_tukeyhsd(endog=df["tau_log"], groups=df["emotion"].astype(str), alpha=0.05)
    tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
    tukey_df.to_csv(out_dir / "tukey_posthoc_log.csv", index=False)
    print("[OK] Saved Tukey HSD (log τ) → tukey_posthoc_log.csv")
except Exception as e:
    print(f"[WARN] Tukey HSD failed: {e}")

# Wilcoxon pairwise
pairs, stats = [], []
for (a,b) in itertools.combinations(emotions, 2):
    x = df[df["emotion"]==a]["tau_log"].dropna().values
    y = df[df["emotion"]==b]["tau_log"].dropna().values
    if len(x) > 1 and len(y) > 1:
        try:
            stat, p = wilcoxon(x[:min(len(x),len(y))], y[:min(len(x),len(y))])
            pairs.append((a,b)); stats.append((stat,p))
        except Exception:
            continue
pair_df = pd.DataFrame([(a,b,s,p) for (a,b),(s,p) in zip(pairs,stats)],
                       columns=["emoA","emoB","W","p_value"])
pair_df.to_csv(out_dir / "wilcoxon_posthoc_log.csv", index=False)
print("[OK] Saved Wilcoxon pairwise (log τ) → wilcoxon_posthoc_log.csv")

# ==== 5) 요약 저장 ====
summary = df.groupby(["task","emotion"])[["tau_rel","tau_log"]].agg(["mean","std","count"]).reset_index()
summary.to_csv(out_dir / "tau_log_summary.csv", index=False)
print(f"[OK] Saved summary → {out_dir}/tau_log_summary.csv")
