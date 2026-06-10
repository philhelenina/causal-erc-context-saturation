#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fig_kstar_full.py (final)
- Emotion × Layer heatmap (p90 K*_norm)
- Emotion-wise violin (per-utterance K*_norm)
- Auto path detection for *_norm_savepreds/
"""

import pandas as pd, numpy as np
import seaborn as sns, matplotlib.pyplot as plt
from pathlib import Path

# ==== 설정 ====
TASK   = "6way"          # or "6way"
MODEL  = "sr-sentic-fused-alpha010"
LAYERS = ["avg_last4","last"]
POOL   = "mean"
STABLE = 3

# Emotion label map
EMO_4W = {0:"Anger", 1:"Happy", 2:"Sad", 3:"Neutral"}
EMO_6W = {0:"Anger", 1:"Happy", 2:"Sad", 3:"Neutral", 4:"Excited", 5:"Frustrated"}
EMO_MAP = EMO_4W if TASK=="4way" else EMO_6W

ROOT = Path("results/turnlevel_k_sweep_norm_savepreds")
OUT  = Path("results/figures_kstar_full")
OUT.mkdir(parents=True, exist_ok=True)

sns.set(style="whitegrid", font_scale=1.05)

def load_summary(task, model, layer, pool, stable):
    d = ROOT / f"{task}_{model}_{layer}_{pool}" / f"kstar_analysis_hard_s{stable}_hard"
    f = d / "kstar_hard_summary_by_emotion.csv"
    if not f.exists():
        print(f"[WARN] Missing summary: {f}")
        return None

    df = pd.read_csv(f)

    # 🔧 Case 1: emotion 열 없음 (즉, 첫 열이 n_samples)
    if "emotion" not in df.columns:
        df.reset_index(inplace=True)
        df.rename(columns={"index": "emotion"}, inplace=True)

    # 🔧 Case 2: emotion 대신 첫 번째 컬럼이 n_samples일 경우
    if "n_samples" in df.columns and "emotion" not in df.columns:
        df.reset_index(inplace=True)
        df.rename(columns={"index": "emotion"}, inplace=True)

    # emotion ID → 정수
    df["emotion"] = pd.to_numeric(df["emotion"], errors="coerce").astype("Int64")

    # 감정명 매핑
    df["emotion_name"] = df["emotion"].map(EMO_MAP)
    df["layer"] = layer
    df["pool"] = pool
    return df.dropna(subset=["emotion_name"])

def load_raw(task, model, layer, pool, stable):
    """utterance별 raw K*"""
    d = ROOT / f"{task}_{model}_{layer}_{pool}" / f"kstar_analysis_hard_s{stable}_hard"
    f = d / "kstar_hard_raw_by_utterance.csv"
    if not f.exists():
        print(f"[WARN] Missing raw: {f}")
        return None
    df = pd.read_csv(f)
    if "emotion" not in df.columns:
        df = pd.read_csv(f, index_col=0).reset_index().rename(columns={"index":"emotion"})
    df["emotion"] = pd.to_numeric(df["emotion"], errors="coerce").astype("Int64")
    df["emotion_name"] = df["emotion"].map(EMO_MAP)
    df["layer"] = layer
    df["pool"] = pool
    return df.dropna(subset=["emotion_name"])

# === 1) Heatmap ===
frames = [load_summary(TASK, MODEL, L, POOL, STABLE) for L in LAYERS]
frames = [x for x in frames if x is not None]
if frames:
    H = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["emotion_name","layer"])
    P = H.pivot(index="emotion_name", columns="layer", values="p90_Kn")
    plt.figure(figsize=(5.8,3.2))
    ax = sns.heatmap(P, annot=True, fmt=".2f", cmap="YlGnBu",
                     annot_kws={"size":9,"weight":"medium"},
                     cbar_kws={"label":"p90(K*_norm, % of dialogue turns)","shrink":0.8})
    ax.set_xlabel("Layer",fontsize=10)
    ax.set_ylabel("Emotion Category",fontsize=10)
    ax.set_title(f"{TASK.upper()} | p90(K*_norm) by Emotion×Layer (hard-only, s={STABLE})",
                 fontsize=11,pad=10,weight="bold")
    plt.xticks(fontsize=9); plt.yticks(fontsize=9)
    plt.tight_layout(pad=1.3)
    plt.savefig(OUT/f"{TASK}_heatmap_p90K_final.png",dpi=400)
    print("[OK] saved:",OUT/f"{TASK}_heatmap_p90K_final.png")

# === 2) Violin ===
raws = [load_raw(TASK, MODEL, L, POOL, STABLE) for L in LAYERS]
raws = [x for x in raws if x is not None]
if raws:
    R = pd.concat(raws, ignore_index=True)
    plt.figure(figsize=(6.2,3.5))
    ax = sns.violinplot(data=R, x="emotion_name", y="K_star_norm",
                        hue="layer", inner="quartile", split=True,
                        palette="muted", linewidth=0.8)
    ax.set_xlabel("Emotion Category",fontsize=10)
    ax.set_ylabel("Normalized Context Length (K*_norm, %)",fontsize=10)
    ax.set_title(f"{TASK.upper()} | K*_norm distributions (hard-only, s={STABLE})",
                 fontsize=11,pad=10,weight="bold")
    ax.legend(title="Layer",fontsize=9,title_fontsize=9,loc="upper right",frameon=False)
    plt.xticks(fontsize=9); plt.yticks(fontsize=9)
    plt.tight_layout(pad=1.2)
    plt.savefig(OUT/f"{TASK}_violin_Knorm_final.png",dpi=400)
    print("[OK] saved:",OUT/f"{TASK}_violin_Knorm_final.png")

# === 3) CSV 병합 ===
if frames:
    M = pd.concat(frames, ignore_index=True)
    M.to_csv(OUT/f"{TASK}_summary_all_layers_final.csv", index=False)
    print("[OK] saved summary CSV:", OUT/f"{TASK}_summary_all_layers_final.csv")

