#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_turnlevel_k_sweep_norm_full.py
- Adds length normalization (K_norm = K / total_turns)
- Fits saturation curve to compute τ_rel (relative context saturation)
- Saves normalized curves and plots automatically
"""

import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score
from scipy.optimize import curve_fit
import torch
from senticcrystal.models.classifiers import SimpleLSTM

HOME = Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()
DATA = HOME / "data"
RESULT_ROOT = HOME / "results" / "turnlevel_k_sweep_norm"

def saturation_func(x, a, b, tau):
    return b + a * (1 - np.exp(-x / tau))

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_npz(p):
    arr = np.load(p, allow_pickle=True)
    return arr["embeddings"], arr["y"]

def train_and_eval(model, Xtr, ytr, Xva, yva, Xte, yte, epochs=50, bs=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss()

    Xtr, ytr = torch.tensor(Xtr).float(), torch.tensor(ytr).long()
    Xva, yva = torch.tensor(Xva).float(), torch.tensor(yva).long()
    Xte, yte = torch.tensor(Xte).float(), torch.tensor(yte).long()

    tr_dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr, ytr), batch_size=bs, shuffle=True)
    va_dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xva, yva), batch_size=bs)
    te_dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xte, yte), batch_size=bs)

    best_val, best_state = 9e9, None
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            
            # === -1 mask 처리 ===
            mask = yb >= 0
            if mask.sum() == 0:
                continue
            logits = model(xb)
            loss = crit(logits[mask], yb[mask])
            loss.backward()
            opt.step()

        # validation
        model.eval(); val_loss = 0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(device), yb.to(device)
                mask = yb >= 0
                if mask.sum() == 0:
                    continue
                logits = model(xb)
                val_loss += crit(logits[mask], yb[mask]).item()
        val_loss /= max(len(va_dl), 1)
        if val_loss < best_val:
            best_val, best_state = val_loss, model.state_dict().copy()

    model.load_state_dict(best_state)
    preds, gts = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in te_dl:
            xb, yb = xb.to(device), yb.to(device)
            mask = yb >= 0
            if mask.sum() == 0:
                continue
            logits = model(xb)
            preds.append(logits[mask].argmax(1).cpu().numpy())
            gts.append(yb[mask].cpu().numpy())

    ypred, ytrue = np.concatenate(preds), np.concatenate(gts)
    f1w = f1_score(ytrue, ypred, average="weighted")
    acc = accuracy_score(ytrue, ypred)
    return f1w, acc

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["4way","6way"])
    ap.add_argument("--layer", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--model_tag", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gpu", type=str, default="0")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    emb_base = DATA / "embeddings" / args.task / args.model_tag / args.layer / args.pool
    print(f"[LOAD] {emb_base}")

    Xtr, ytr = load_npz(emb_base / "train.npz")
    Xva, yva = load_npz(emb_base / "val.npz")
    Xte, yte = load_npz(emb_base / "test.npz")

    num_classes = len(np.unique(ytr))
    in_dim = Xtr.shape[-1]
    total_turns = len(ytr)

    rows = []
    Ks = list(range(0, 105, 5))
    for K in Ks:
        K_norm = K / total_turns if total_turns > 0 else 0
        model = SimpleLSTM(input_size=in_dim, hidden_size=256, num_classes=num_classes, dropout_rate=0.3)
        f1w, acc = train_and_eval(model, Xtr, ytr, Xva, yva, Xte, yte,
                                  epochs=args.epochs, bs=args.bs, lr=args.lr)
        rows.append(dict(K=K, K_norm=K_norm, f1_weighted=f1w, accuracy=acc))
        print(f"[K={K:3d}] F1w={f1w:.3f}, Acc={acc:.3f}, K_norm={K_norm:.3f}")

    df = pd.DataFrame(rows)
    out_dir = RESULT_ROOT / f"{args.task}_{args.model_tag}_{args.layer}_{args.pool}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "k_sweep_norm.csv", index=False)
    print(f"[OK] Saved normalized CSV → {out_dir}/k_sweep_norm.csv")

    # ==== τ_rel fit ====
    try:
        x, y = df["K_norm"].values, df["f1_weighted"].values
        popt, _ = curve_fit(saturation_func, x, y, maxfev=10000)
        a,b,tau = popt
        print(f"[FIT] τ_rel = {tau:.4f}")
    except Exception as e:
        print(f"[WARN] τ_rel fit failed: {e}")
        a,b,tau = (0,0,0)

    # ==== Plot ====
    plt.figure(figsize=(6,4))
    plt.plot(df["K_norm"], df["f1_weighted"], "o-", label="F1_weighted")
    xfit = np.linspace(0, df["K_norm"].max(), 100)
    plt.plot(xfit, saturation_func(xfit, a,b,tau), "r--", label=f"Saturation fit τ_rel={tau:.3f}")
    plt.xlabel("Normalized Context Ratio (K / total_turns)")
    plt.ylabel("Weighted F1")
    plt.title(f"{args.task} | {args.model_tag} | {args.layer}/{args.pool}")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "normalized_k_curve.png", dpi=300)
    print(f"[OK] Saved plot → {out_dir}/normalized_k_curve.png")

if __name__ == "__main__":
    main()
