#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_information_flow.py
- SenticCrystal 정보 흐름 분석 전체 파이프라인
- 각 모델별 preds_perK.npy / labels.npy 로부터
  * Accuracy / Entropy / Mutual Information / ECE 계산
  * 4way, 6way 각각 자동 처리
  * 결과: CSV + PNG 저장
"""

import os, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score
from scipy.stats import entropy

# === 설정 ===
TASKS = ["4way", "6way"]
BASES = [
    ("sentence-roberta", "avg_last4", "mean"),
    ("sr-sentic-fused-alpha010", "last", "mean"),
]
LABELS = ["SR-only", "SR+Sentic (α=0.1)"]

RES_ROOT = Path("results/turnlevel_k_sweep_norm_savepreds")
OUTDIR = Path("results/information_flow"); OUTDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 12})

# ------------------- 유틸 함수 -------------------
def expected_calibration_error(y_true, y_prob, n_bins=15):
    """Expected Calibration Error (ECE) 계산"""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    confidences = np.max(y_prob, axis=1)
    preds = np.argmax(y_prob, axis=1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if not np.any(mask):
            continue
        acc_bin = np.mean(preds[mask] == y_true[mask])
        conf_bin = np.mean(confidences[mask])
        ece += np.abs(acc_bin - conf_bin) * np.mean(mask)
    return ece

def mutual_information(pY, pYgX):
    """I(Y;X) = H(Y) - H(Y|X)"""
    HY = entropy(pY)
    Hcond = np.mean([entropy(p) for p in pYgX])
    return HY - Hcond

# ------------------- 분석 함수 -------------------
def analyze(task, model_tag, layer, pool):
    d = RES_ROOT / f"{task}_{model_tag}_{layer}_{pool}"
    preds_path = d / "preds_perK.npy"
    labels_path = d / "labels.npy"
    Ks_path = d / "Ks.npy"

    if not preds_path.exists():
        print(f"[WARN] Missing preds_perK.npy for {model_tag}/{layer}/{pool}")
        return None

    pK = np.load(preds_path, allow_pickle=True)
    y = np.load(labels_path, allow_pickle=True)
    Ks = np.load(Ks_path, allow_pickle=True)

    # ---- shape 보정 ----
    if pK.ndim == 2:  # (n_samples, n_K) 형태면 (n_K, n_samples, 1)로 변환
        pK = np.expand_dims(pK.T, -1)
    elif pK.ndim == 3 and pK.shape[0] < pK.shape[1]:
        pass  # 올바름
    else:
        print(f"[WARN] Unexpected shape {pK.shape}")

    nK = pK.shape[0]
    results = []

    for i in range(nK):
        probs = pK[i]
        if probs.ndim == 1:
            # 정수 예측 레이블만 저장된 경우 → one-hot 변환
            preds = probs.astype(int)
            num_classes = int(np.max(preds)) + 1
            probs = np.eye(num_classes)[preds]
        else:
            preds = probs.argmax(1)

        # === 길이 불일치 보정 ===
        if len(preds) != len(y):
            n = min(len(preds), len(y))
            preds = preds[:n]
            y = y[:n]
            probs = probs[:n]

        acc = accuracy_score(y, preds)
        ece = expected_calibration_error(y, probs)
        pY = np.bincount(y, minlength=probs.shape[1]) / len(y)
        IY = mutual_information(pY, probs)
        H = np.mean([entropy(p) for p in probs])
        results.append(dict(K=Ks[i], acc=acc, entropy=H, mi=IY, ece=ece))

    df = pd.DataFrame(results)
    df["model"] = model_tag
    df["layer"] = layer
    df["pool"] = pool
    df["task"] = task
    return df

# ------------------- 실행 루프 -------------------
print("=== [SenticCrystal] Information-Flow Analysis ===")
print("Start time:", time.strftime("%c"))
print("========================================================")

for task in TASKS:
    dfs = []
    for (tag, layer, pool), label in zip(BASES, LABELS):
        df = analyze(task, tag, layer, pool)
        if df is not None:
            df["label"] = label
            dfs.append(df)

    if not dfs:
        print(f"[ERR] No valid model folders found for {task}")
        continue

    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(OUTDIR / f"{task}_info_curves.csv", index=False)

    # === Plot ===
    plt.figure(figsize=(7,5))
    for lbl, g in all_df.groupby("label"):
        plt.plot(g["K"], g["mi"], label=f"{lbl} (MI)", linestyle="-")
        plt.plot(g["K"], g["entropy"], label=f"{lbl} (Entropy)", linestyle="--")
    plt.xlabel("Context Length K")
    plt.ylabel("Information (nats)")
    plt.title(f"{task.upper()} | Mutual Information & Entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{task}_info_MI_entropy.png", dpi=300)

    plt.figure(figsize=(7,5))
    for lbl, g in all_df.groupby("label"):
        plt.plot(g["K"], g["ece"], label=f"{lbl} (ECE)")
    plt.xlabel("Context Length K")
    plt.ylabel("Calibration Error (ECE)")
    plt.title(f"{task.upper()} | Calibration over Context Length")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{task}_info_ECE.png", dpi=300)

    plt.figure(figsize=(7,5))
    for lbl, g in all_df.groupby("label"):
        plt.plot(g["K"], g["acc"], label=f"{lbl} (Accuracy)")
    plt.xlabel("Context Length K")
    plt.ylabel("Accuracy")
    plt.title(f"{task.upper()} | Accuracy over Context Length")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{task}_info_ACC.png", dpi=300)

    print(f"[OK] Saved {task} info-flow curves → {OUTDIR}")

print("Done.")

