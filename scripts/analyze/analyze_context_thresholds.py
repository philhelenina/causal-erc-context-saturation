#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_context_thresholds.py
- preds_perK.npy, labels.npy, Ks.npy, dialog_len.npy를 이용해
  per-sample K* (최초 정답 K) 및 K*_norm 계산
- 감정별 분포 요약 + 비모수 쌍대 검정(Mann–Whitney U, KS, Cliff's δ)
"""
import os

import numpy as np, pandas as pd
from pathlib import Path
from itertools import combinations
from scipy.stats import mannwhitneyu, ks_2samp

HOME = Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()
RES_ROOT = HOME / "results" / "turnlevel_k_sweep_norm_savepreds"

def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    nx, ny = len(x), len(y)
    U, _ = mannwhitneyu(x, y, alternative="two-sided")
    return 2*U/(nx*ny) - 1  # ≈ Cliff's δ

def holm_bonferroni(pvals):
    """Holm 보정: 반환은 p_adj (원래 순서 유지)"""
    m = len(pvals)
    order = np.argsort(pvals)
    p_sorted = np.array(pvals)[order]
    adj = np.zeros_like(p_sorted)
    for i, p in enumerate(p_sorted):
        adj[i] = (m - i) * p
    # 누적 max 로 단조 증가 보장
    for i in range(1, m):
        adj[i] = max(adj[i], adj[i-1])
    p_adj = np.zeros_like(adj)
    p_adj[order] = np.minimum(adj, 1.0)
    return p_adj

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)            # 4way | 6way
    ap.add_argument("--model_tag", required=True)
    ap.add_argument("--layer", required=True)
    ap.add_argument("--pool", required=True)
    args = ap.parse_args()

    combo = f"{args.task}_{args.model_tag}_{args.layer}_{args.pool}"
    base = RES_ROOT / combo
    if not base.exists():
        raise SystemExit(f"❌ Not found: {base}")

    P = np.load(base/"preds_perK.npy")   # (N_valid, M_K)
    y = np.load(base/"labels.npy")       # (N_valid,)
    Ks = np.load(base/"Ks.npy")          # (M_K,)
    dlen = np.load(base/"dialog_len.npy")# (N_valid,)

    N, M = P.shape
    K_star = np.full(N, np.nan)
    K_star_norm = np.full(N, np.nan)
    solved = np.zeros(N, dtype=bool)

    for i in range(N):
        idx = np.where(P[i, :] == y[i])[0]
        if len(idx) > 0:
            j = idx[0]
            K_star[i] = Ks[j]
            K_star_norm[i] = Ks[j] / max(dlen[i], 1)
            solved[i] = True

    df = pd.DataFrame({
        "emotion": y,
        "K_star": K_star,
        "K_star_norm": K_star_norm,
        "solved": solved
    })

    outdir = base / "kstar_analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    # 감정별 요약
    def q(dfcol, q): return dfcol.quantile(q) if dfcol.notna().any() else np.nan
    g = df.groupby("emotion")
    summ = pd.DataFrame({
        "n": g["K_star_norm"].count(),
        "solved_frac": g["solved"].mean(),
        "median_K": g["K_star"].median(),
        "p90_K": g["K_star"].quantile(0.90),
        "median_Kn": g["K_star_norm"].median(),
        "p90_Kn": g["K_star_norm"].quantile(0.90),
        "IQR_Kn": g["K_star_norm"].quantile(0.75) - g["K_star_norm"].quantile(0.25),
    })
    summ.to_csv(outdir/"kstar_summary_by_emotion.csv")
    print("[OK] saved", outdir/"kstar_summary_by_emotion.csv")

    # 쌍대 비모수 검정
    rows=[]
    emos = sorted(df["emotion"].dropna().unique())
    for a,b in combinations(emos,2):
        xa = df.loc[df["emotion"]==a, "K_star_norm"].dropna().values
        xb = df.loc[df["emotion"]==b, "K_star_norm"].dropna().values
        if len(xa)>0 and len(xb)>0:
            U, p_u = mannwhitneyu(xa, xb, alternative="two-sided")
            ks, p_ks = ks_2samp(xa, xb, alternative="two-sided")
            cd = cliffs_delta(xa, xb)
            rows.append(dict(
                emoA=int(a), emoB=int(b),
                nA=len(xa), nB=len(xb),
                U=U, p_mwu=p_u,
                KS=ks, p_ks=p_ks,
                cliffs_delta=cd
            ))
    pair = pd.DataFrame(rows)

    # Holm 보정
    if not pair.empty:
        for col in ["p_mwu","p_ks"]:
            pair[col+"_holm"] = holm_bonferroni(pair[col].values)
            pair[col+"_reject@0.05"] = pair[col+"_holm"] < 0.05
    pair.to_csv(outdir/"kstar_pairwise_tests.csv", index=False)
    print("[OK] saved", outdir/"kstar_pairwise_tests.csv")

    # 콘솔용 간단 요약
    print("\n=== K* (normalized) median by emotion ===")
    print(summ["median_Kn"].round(4))

if __name__ == "__main__":
    main()
