#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_tau_rel_stats.py
- Collects τ_rel values from multiple experiments
- Performs one-way ANOVA and Kruskal-Wallis tests across emotions
"""
import os

import json, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import f_oneway, kruskal

HOME = Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()
RES_DIR = HOME / "results" / "turnlevel_k_sweep_norm_per_emotion"

# ==== 실험 목록 (조합별 결과 폴더) ====
COMBOS = [
    "6way_sr-sentic-fused-alpha010_avg_last4_mean",
    "6way_sr-sentic-fused-alpha010_avg_last4_wmean_pos_rev",
    "6way_sr-sentic-fused-alpha010_avg_last4_wmean_idf",
    "6way_sr-sentic-fused-alpha015_avg_last4_mean",
]

# ==== 1) τ_rel 값 수집 ====
records = []
for combo in COMBOS:
    f = RES_DIR / combo / "tau_rel_per_emotion.json"
    if not f.exists():
        print(f"[SKIP] {f} missing")
        continue
    data = json.load(open(f))
    for emo, tau in data.items():
        records.append(dict(combo=combo, emotion=int(emo), tau_rel=tau))
df = pd.DataFrame(records)
print("\n[INFO] τ_rel data collected:")
print(df.head())

# ==== 2) 통계 검정 ====
# 감정별 리스트
emotions = sorted(df["emotion"].unique())
grouped = [df[df["emotion"]==e]["tau_rel"].dropna().values for e in emotions]

anova_F, anova_p = f_oneway(*grouped)
kw_H, kw_p = kruskal(*grouped)

print("\n[ANOVA] F=%.4f, p=%.4e" % (anova_F, anova_p))
print("[Kruskal–Wallis] H=%.4f, p=%.4e" % (kw_H, kw_p))

# ==== 3) 요약 저장 ====
out = RES_DIR / "tau_rel_stats_summary.csv"
summary = df.groupby("emotion")["tau_rel"].agg(["mean","std","count"]).reset_index()
summary.to_csv(out, index=False)
print(f"\n[OK] Saved summary → {out}")
