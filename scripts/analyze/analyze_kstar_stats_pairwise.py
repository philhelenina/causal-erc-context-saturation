#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_kstar_stats_pairwise.py
- Pairwise tests on K*_norm (hard-only raw per-utterance)
- Within-layer: emotion vs emotion (Holm correction), Cliff's delta
- Within-emotion: avg_last4 vs last (Holm correction), Cliff's delta
"""

import pandas as pd, numpy as np
from pathlib import Path
from itertools import combinations
from scipy.stats import mannwhitneyu
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--task", choices=["4way","6way"], default="4way")
ap.add_argument("--model", default="sr-sentic-fused-alpha010")
ap.add_argument("--layers", nargs="+", default=["avg_last4","last"])
ap.add_argument("--pool", default="mean")
ap.add_argument("--stable", type=int, default=3)
ap.add_argument("--alpha", type=float, default=0.05)
args = ap.parse_args()

TASK, MODEL, LAYERS, POOL, STABLE = args.task, args.model, args.layers, args.pool, args.stable
ROOT = Path("results/turnlevel_k_sweep_norm_savepreds")
OUTDIR = Path("results/figures_kstar_full"); OUTDIR.mkdir(parents=True, exist_ok=True)

EMO_4W = {0:"Anger", 1:"Happy", 2:"Sad", 3:"Neutral"}
EMO_6W = {0:"Frustrated", 1:"Sad", 2:"Excited", 3:"Happy", 4:"Neutral", 5:"Anger"}
EMO_MAP = EMO_4W if TASK=="4way" else EMO_6W

def read_raw(layer):
    d = ROOT / f"{TASK}_{MODEL}_{layer}_{POOL}" / f"kstar_analysis_hard_s{STABLE}_hard"
    f = d / "kstar_hard_raw_by_utterance.csv"
    if not f.exists(): return None
    df = pd.read_csv(f)
    df["layer"] = layer
    df["Emotion"] = df["emotion"].map(EMO_MAP)
    df["K_star_norm"] = df["K_star_norm"].clip(0, 100)
    return df

def holm_correction(pvals):
    """Holm–Bonferroni: returns array of adjusted p-values in original order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    prev = 0.0
    for rank, idx in enumerate(order):
        p = pvals[idx]
        adj_val = (m - rank) * p
        adj_val = max(adj_val, prev)  # monotone
        prev = adj_val
        adj[idx] = min(1.0, adj_val)
    return adj

def cliffs_delta(x, y):
    """Cliff's delta with magnitude label."""
    nx, ny = len(x), len(y)
    # rank-based count
    gt = sum(1 for xi in x for yj in y if xi > yj)
    lt = sum(1 for xi in x for yj in y if xi < yj)
    d = (gt - lt) / (nx*ny)
    ad = abs(d)
    if   ad < 0.147: mag = "negligible"
    elif ad < 0.33: mag = "small"
    elif ad < 0.474: mag = "medium"
    else: mag = "large"
    return d, mag

# load raws
raws = [read_raw(L) for L in LAYERS]
raws = [r for r in raws if r is not None]
if not raws:
    raise SystemExit("[ERR] No raw files found. Run analyze_context_thresholds_hard.py first.")

R = pd.concat(raws, ignore_index=True)

# 1) within-layer pairwise (emotion vs emotion)
rows = []
for L in LAYERS:
    sub = R[R["layer"]==L]
    emos = [e for e in EMO_MAP.values() if e in sub["Emotion"].unique()]
    pairs = list(combinations(emos, 2))
    pvals = []
    stats = []
    for a,b in pairs:
        xa = sub.loc[sub["Emotion"]==a, "K_star_norm"].values
        xb = sub.loc[sub["Emotion"]==b, "K_star_norm"].values
        if len(xa)==0 or len(xb)==0: 
            p = np.nan; u = np.nan; d = np.nan; mag = "NA"
        else:
            u, p = mannwhitneyu(xa, xb, alternative="two-sided")
            d, mag = cliffs_delta(xa, xb)
        pvals.append(p); stats.append((u,d,mag,len(xa),len(xb)))
    padj = holm_correction([p if np.isfinite(p) else 1.0 for p in pvals])
    for (a,b), (p,pH), (u,d,mag,na,nb) in zip(pairs, zip(pvals,padj), stats):
        rows.append(dict(task=TASK, layer=L, pool=POOL, stable=STABLE,
                         emo1=a, emo2=b, n1=na, n2=nb, U=u, p_raw=p, p_holm=pH,
                         cliffs_delta=d, magnitude=mag))
df_within = pd.DataFrame(rows)
df_within.to_csv(OUTDIR/f"{TASK}_kstar_pairwise_within_layer.csv", index=False)
print("[OK] within-layer pairwise:", OUTDIR/f"{TASK}_kstar_pairwise_within_layer.csv")

# 2) within-emotion (avg_last4 vs last)
rows = []
if set(LAYERS)>=set(["avg_last4","last"]):
    for emo in EMO_MAP.values():
        x = R[(R["Emotion"]==emo) & (R["layer"]=="avg_last4")]["K_star_norm"].values
        y = R[(R["Emotion"]==emo) & (R["layer"]=="last")]["K_star_norm"].values
        if len(x)==0 or len(y)==0:
            u=p=np.nan; d=0; mag="NA"; n1=len(x); n2=len(y)
        else:
            u, p = mannwhitneyu(x, y, alternative="two-sided")
            d, mag = cliffs_delta(x, y)
        rows.append(dict(task=TASK, emotion=emo, n_avg4=len(x), n_last=len(y),
                         U=u, p_raw=p, cliffs_delta=d, magnitude=mag))
    # Holm over emotions
    padj = holm_correction([r["p_raw"] if np.isfinite(r["p_raw"]) else 1.0 for r in rows])
    for r, ph in zip(rows, padj): r["p_holm"] = ph
    df_between = pd.DataFrame(rows)
    df_between.to_csv(OUTDIR/f"{TASK}_kstar_within_emotion_layers.csv", index=False)
    print("[OK] within-emotion (layers):", OUTDIR/f"{TASK}_kstar_within_emotion_layers.csv")
else:
    print("[INFO] Only one layer provided; skip within-emotion layer test.")
