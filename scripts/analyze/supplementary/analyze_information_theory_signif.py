#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU-accelerated information-theoretic analysis (SenticNet axis-level)
- MI(SRoBERTa, SenticNet, Fusion), ΔMI
- SenticNet 4 axes별 MI (pleasant, attention, sensitivity, aptitude)
- Turn-level sweep 지원 (k=0~100)
"""
import os

import argparse, cupy as cp, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler

HOME = Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()
EMB = HOME / "data" / "embeddings"
CSV4 = HOME / "data" / "iemocap_4way_data"
CSV6 = HOME / "data" / "iemocap_6way_data"
OUT = HOME / "results" / "_it_theory_gpu_axes"
OUT.mkdir(parents=True, exist_ok=True)

AXES = ["pleasantness", "attention", "sensitivity", "aptitude"]

# --------------------- 데이터 로딩 ---------------------
def load_npz_any(p):
    arr = np.load(p, allow_pickle=False)
    if isinstance(arr, np.lib.npyio.NpzFile):
        return arr["embeddings"] if "embeddings" in arr else arr[list(arr.keys())[0]]
    return arr

def load_labels(task, split):
    csv = (CSV6 if task=="6way" else CSV4) / f"{split}_{task}_with_minus_one.csv"
    df = pd.read_csv(csv)
    if "label_num" in df.columns:
        y = pd.to_numeric(df["label_num"], errors="coerce").to_numpy()
    else:
        mp4 = {"ang":0,"hap":1,"sad":2,"neu":3}
        mp6 = {"ang":0,"hap":1,"sad":2,"neu":3,"exc":4,"fru":5}
        mp = mp6 if task=="6way" else mp4
        y = df["label"].astype(str).str.lower().map(mp).to_numpy()
    mask = (~pd.isna(y)) & (y >= 0)
    return y[mask].astype(int), mask

# --------------------- GPU MI 계산 ---------------------
def entropy_gpu(p):
    p = cp.asarray(p, dtype=cp.float32)
    p = p[p>0]
    return -cp.sum(p * cp.log2(p))

def mutual_info_gpu(X, y, bins=50):
    """GPU-based MI estimation (binning entropy method)"""
    Xc = cp.asarray(StandardScaler().fit_transform(X))
    yc = cp.asarray(y)
    mi_vals = []
    for j in range(Xc.shape[1]):
        x = Xc[:, j]
        pxy, _, _ = cp.histogram2d(x, yc, bins=[bins, len(cp.unique(yc))], density=True)
        px = cp.sum(pxy, axis=1)
        py = cp.sum(pxy, axis=0)
        mi = entropy_gpu(px) + entropy_gpu(py) - entropy_gpu(pxy.flatten())
        mi_vals.append(float(mi.get()))
    return np.array(mi_vals, dtype=np.float32)

# --------------------- 분석 함수 ---------------------
def analyze(task="6way", split="train", layer="last4_scalar_top2", pool="wmean_pos_rev", k=None):
    y, mask = load_labels(task, split)

    def _emb(base):
        return load_npz_any(base/f"{split}.npz")[:len(mask)][mask]

    # Contextual / Lexical / Fusion
    Xc = _emb(EMB/task/"sentence-roberta"/layer/pool)
    Xl = _emb(EMB/task/"senticnet-axes")
    Xf = np.concatenate([Xc, Xl], axis=1)

    # 전체 MI
    mi_ctx = mutual_info_gpu(Xc, y).mean()
    mi_axes = mutual_info_gpu(Xl, y)       # 축별
    mi_lex = mi_axes.mean()
    mi_fus = mutual_info_gpu(Xf, y).mean()
    dmi = mi_fus - max(mi_ctx, mi_lex)

    tag = f"{task}-{split}-k{k}" if k is not None else f"{task}-{split}"
    print(f"[{tag}] MI_ctx={mi_ctx:.4f}  MI_lex={mi_lex:.4f}  MI_fus={mi_fus:.4f}  ΔMI={dmi:.4f}")

    # 축별 결과 DataFrame
    df_axes = pd.DataFrame({
        "task": task, "split": split, "k": k or 0,
        "axis": AXES, "mi_axis": mi_axes
    })

    return dict(task=task, split=split, k=k or 0,
                mi_ctx=mi_ctx, mi_lex=mi_lex, mi_fus=mi_fus,
                delta_mi=dmi, axes=df_axes)

# --------------------- Turn-level Sweep ---------------------
def run_turn_sweep(task="6way", layer="last4_scalar_top2", pool="wmean_pos_rev", step=5):
    rows, axis_rows = [], []
    for k in range(0, 101, step):
        base = EMB/task/"turn-level"/f"k{k}"
        if not base.exists():
            print(f"[SKIP] turn-level k={k} 없음")
            continue
        for split in ["train","val","test"]:
            try:
                res = analyze(task, split, layer, pool, k)
                rows.append({k:v for k,v in res.items() if k!="axes"})
                axis_rows.append(res["axes"])
            except FileNotFoundError:
                continue
    df = pd.DataFrame(rows)
    df_axes = pd.concat(axis_rows, ignore_index=True)
    df.to_csv(OUT/f"turn_sweep_{task}.csv", index=False)
    df_axes.to_csv(OUT/f"turn_sweep_{task}_axes.csv", index=False)

    # ΔMI vs k plot
    plt.figure(figsize=(6,4))
    plt.plot(df["k"], df["delta_mi"], marker="o")
    plt.title(f"{task} ΔMI vs Turn Context (k)")
    plt.xlabel("Turn window (k)"); plt.ylabel("ΔMI")
    plt.grid(True); plt.tight_layout()
    plt.savefig(OUT/f"delta_mi_vs_k_{task}.png", dpi=150)
    plt.close()

# --------------------- main ---------------------
if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["4way","6way"])
    ap.add_argument("--splits", nargs="+", default=["train","val","test"])
    ap.add_argument("--layer", default="last4_scalar_top2")
    ap.add_argument("--pool", default="wmean_pos_rev")
    ap.add_argument("--turn_sweep", action="store_true")
    args = ap.parse_args()

    all_rows, axis_rows = [], []
    for t in args.tasks:
        if args.turn_sweep:
            run_turn_sweep(t, args.layer, args.pool)
        else:
            for s in args.splits:
                res = analyze(t, s, args.layer, args.pool)
                all_rows.append({k:v for k,v in res.items() if k!="axes"})
                axis_rows.append(res["axes"])

    if not args.turn_sweep:
        pd.DataFrame(all_rows).to_csv(OUT/"utterance_level.csv", index=False)
        pd.concat(axis_rows, ignore_index=True).to_csv(OUT/"utterance_level_axes.csv", index=False)
