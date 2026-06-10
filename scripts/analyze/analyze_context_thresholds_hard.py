#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_context_thresholds_hard.py
- preds_perK.npy, labels.npy, Ks.npy, dialog_len.npy 기반
- hard-only(+연속 정답 stable_n) per-utterance K* 및 K*_norm 계산
- 감정별 요약 + 쌍대 검정(MWU/KS + Holm 보정, Cliff's δ) + 원시 분포 CSV 저장
"""
import os
import numpy as np, pandas as pd
from pathlib import Path
from itertools import combinations
from scipy.stats import mannwhitneyu, ks_2samp

HOME = Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()
RES  = HOME / "results" / "turnlevel_k_sweep_norm_savepreds"

def holm_bonferroni(p):
    m=len(p); order=np.argsort(p); p_sorted=np.array(p)[order]
    adj=np.zeros_like(p_sorted, dtype=float)
    for i,pi in enumerate(p_sorted): adj[i]=(m-i)*pi
    for i in range(1,m): adj[i]=max(adj[i],adj[i-1])
    out=np.ones_like(adj, dtype=float); out[order]=np.minimum(adj,1.0); return out

def cliffs_delta(x,y):
    x,y=np.asarray(x),np.asarray(y); nx,ny=len(x),len(y)
    U,_=mannwhitneyu(x,y,alternative="two-sided"); return 2*U/(nx*ny)-1

def find_kstar_row(pred_row, y_true, stable_n):
    M=len(pred_row)
    if stable_n<=1:
        idx=np.where(pred_row==y_true)[0]; return int(idx[0]) if len(idx)>0 else None
    run=0
    for j in range(M):
        if pred_row[j]==y_true:
            run+=1
            if run>=stable_n: return j-stable_n+1
        else: run=0
    return None

def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["4way","6way"])
    ap.add_argument("--model_tag", required=True)
    ap.add_argument("--layer", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--stable_n", type=int, default=3)
    ap.add_argument("--hard_only", action="store_true")
    args=ap.parse_args()

    combo=f"{args.task}_{args.model_tag}_{args.layer}_{args.pool}"
    base = RES / combo
    if not base.exists(): raise SystemExit(f"❌ Not found: {base}")

    P  = np.load(base/"preds_perK.npy")    # (N, M)
    y  = np.load(base/"labels.npy")        # (N,)
    Ks = np.load(base/"Ks.npy")            # (M,)
    dlen = np.load(base/"dialog_len.npy")  # (N,)

    if args.hard_only:
        mask = (P[:,0] != y)
        P,y,dlen = P[mask], y[mask], dlen[mask]

    kstar_idx = np.full(len(y), -1, dtype=int)
    for i in range(len(y)):
        j = find_kstar_row(P[i], int(y[i]), args.stable_n)
        kstar_idx[i]= j if j is not None else -1

    K_star      = np.where(kstar_idx>=0, Ks[kstar_idx], np.nan)
    K_star_norm = K_star / np.maximum(dlen,1)

    df = pd.DataFrame({"emotion": y, "K_star": K_star, "K_star_norm": K_star_norm})
    outdir = base / f"kstar_analysis_hard_s{args.stable_n}{'_hard' if args.hard_only else ''}"
    outdir.mkdir(parents=True, exist_ok=True)

    raw = df.dropna().copy()
    raw.to_csv(outdir/"kstar_hard_raw_by_utterance.csv", index=False)

    g=df.groupby("emotion")
    summ=pd.DataFrame({
        "n_samples": g["K_star_norm"].count(),
        "solved_frac": g["K_star_norm"].apply(lambda s: s.notna().mean()),
        "median_Kn": g["K_star_norm"].median(),
        "p75_Kn": g["K_star_norm"].quantile(0.75),
        "p90_Kn": g["K_star_norm"].quantile(0.90),
        "IQR_Kn": g["K_star_norm"].quantile(0.75)-g["K_star_norm"].quantile(0.25),
    })
    summ.to_csv(outdir/"kstar_hard_summary_by_emotion.csv", index=False)

    emos=sorted(df["emotion"].dropna().unique())
    rows=[]
    for a in emos:
        for b in emos:
            if b<=a: continue
            xa=df.loc[df["emotion"]==a,"K_star_norm"].dropna().values
            xb=df.loc[df["emotion"]==b,"K_star_norm"].dropna().values
            if len(xa)==0 or len(xb)==0: continue
            U,p_u = mannwhitneyu(xa,xb,alternative="two-sided")
            ks,p_ks= ks_2samp(xa,xb,alternative="two-sided")
            cd=cliffs_delta(xa,xb)
            rows.append(dict(emoA=int(a),emoB=int(b),nA=len(xa),nB=len(xb),
                             p_mwu=p_u,p_ks=p_ks,cliffs_delta=cd))
    pair=pd.DataFrame(rows)
    if not pair.empty:
        for col in ["p_mwu","p_ks"]:
            pair[col+"_holm"]=holm_bonferroni(pair[col].values)
            pair[col+"_reject@0.05"] = pair[col+"_holm"]<0.05
    pair.to_csv(outdir/"kstar_hard_pairwise_tests.csv", index=False)

    print("\n=== hard-only / stable_n={} ===".format(args.stable_n))
    print(summ[["n_samples","solved_frac","p90_Kn","IQR_Kn"]].round(4))
    print("[OK] saved:", outdir)

if __name__=="__main__":
    main()

