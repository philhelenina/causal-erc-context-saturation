#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_discourse_markers.py
Analyzes the distribution and position of discourse markers across emotions.
Usage:
    python3 scripts/analyze_discourse_markers.py --task 4way
    python3 scripts/analyze_discourse_markers.py --task 6way
"""

import re, os, argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

# ---------------------------------------------
# 주요 담화표지 목록
# ---------------------------------------------
MARKERS = [
    "but", "so", "well", "because", "oh",
    "if", "then", "however", "actually", "anyway",
    "indeed", "though", "although", "therefore", "after all"
]

# ---------------------------------------------
def detect_markers(text):
    """주어진 발화에서 담화표지와 그 위치를 반환"""
    text = str(text).lower()
    found = []
    for m in MARKERS:
        for match in re.finditer(rf"\b{re.escape(m)}\b", text):
            pos = match.start() / max(len(text), 1)
            found.append((m, pos))
    return found

# ---------------------------------------------
def analyze(df, out_dir):
    """담화표지 빈도, 위치, 감정별 상관관계 분석"""
    results = []
    for _, row in df.iterrows():
        utt, label = str(row["utterance"]), str(row["label"])
        if label == "-1":  # unlabeled 무시
            continue
        found = detect_markers(utt)
        for marker, pos in found:
            region = (
                "start" if pos < 0.33 else
                "middle" if pos < 0.66 else
                "end"
            )
            results.append({
                "marker": marker,
                "position": pos,
                "region": region,
                "emotion": label
            })

    if not results:
        print("[WARN] No discourse markers detected.")
        return

    df_m = pd.DataFrame(results)
    os.makedirs(out_dir, exist_ok=True)

    # ---------- (1) 감정별 빈도표 ----------
    freq = df_m.groupby(["emotion", "marker"]).size().unstack(fill_value=0)
    freq.to_csv(out_dir / "marker_freq.csv")

    # ---------- (2) 위치 통계 ----------
    pos_stats = df_m.groupby(["emotion", "marker"])["position"].mean().unstack(fill_value=np.nan)
    pos_stats.to_csv(out_dir / "marker_position_mean.csv")

    # ---------- (3) 상관관계 ----------
    freq_filled = freq.T.fillna(0)
    if freq_filled.shape[1] > 1:
        corr, _ = spearmanr(freq_filled)
        corr_df = pd.DataFrame(
            corr,
            index=freq_filled.columns,
            columns=freq_filled.columns
        )
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_df, cmap="coolwarm", center=0, square=True)
        plt.title("Discourse Marker Co-occurrence Correlation")
        plt.tight_layout()
        plt.savefig(out_dir / "marker_heatmap.png", dpi=300)
        plt.close()
    else:
        print("[INFO] Not enough markers for correlation heatmap; skipped.")

    # ---------- (4) 감정별 위치 분포 ----------
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df_m, x="marker", y="position", hue="emotion")
    plt.xticks(rotation=45)
    plt.title("Marker Position by Emotion")
    plt.tight_layout()
    plt.savefig(out_dir / "marker_position_boxplot.png", dpi=300)
    plt.close()

    print(f"[OK] Saved analysis results → {out_dir}")

# ---------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["4way", "6way"])
    args = ap.parse_args()

    DATA = Path("data") / f"iemocap_{args.task}_data"
    FILE = DATA / f"all_{args.task}_with_minus_one.csv"
    OUT = Path("results") / "discourse_markers" / args.task
    os.makedirs(OUT, exist_ok=True)

    df = pd.read_csv(FILE)
    analyze(df, OUT)
