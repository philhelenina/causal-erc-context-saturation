#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_turnlevel_k_sweep_norm_per_emotion.py
- Length-normalized (K_norm) turn-level sweep
- Computes per-emotion F1 and fits τ_rel for each class
"""

import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score
from scipy.optimize import curve_fit
import torch
from senticcrystal.models.classifiers import SimpleLSTM

HOME = Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()
DATA = HOME / "data"
RESULT_ROOT = HOME / "results" / "turnlevel_k_sweep_norm_per_emotion"

def saturation_func(x, a, b, tau):
    return b + a * (1 - np.exp(-x / tau))

def load_npz(p):
    arr = np.load(p, allow_pickle=True)
    X = arr["embeddings"]
    y = arr["y"]
    return X, y

def train_and_eval(model, Xtr, ytr, Xva, yva, Xte, yte, epochs=50, bs=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss()

    to_tensor = lambda X, y: (torch.tensor(X).float(), torch.tensor(y).long())
    Xtr, ytr = to_tensor(Xtr, ytr)
    Xva, yva = to_tensor(Xva, yva)
    Xte, yte = to_tensor(Xte, yte)
    tr_dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr, ytr), batch_size=bs, shuffle=True)
    va_dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xva, yva), batch_size=bs)
    te_dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xte, yte), batch_size=bs)

    best_val, best_state = 9e9, None
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            mask = yb >= 0
            if mask.sum() == 0: continue
            opt.zero_grad()
            out = model(xb)
            loss = crit(out[mask], yb[mask])
            loss.backward(); opt.step()

        model.eval(); val_loss = 0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(device), yb.to(device)
                mask = yb >= 0
                if mask.sum() == 0: continue
                val_loss += crit(model(xb)[mask], yb[mask]).item()
        val_loss /= max(len(va_dl), 1)
        if val_loss < best_val:
            best_val, best_state = val_loss, model.state_dict().copy()

    model.load_state_dict(best_state)

    # Test
    model.eval(); preds, gts = [], []
    with torch.no_grad():
        for xb, yb in te_dl:
            xb, yb = xb.to(device), yb.to(device)
            mask = yb >= 0
            if mask.sum() == 0: continue
            logits = model(xb)
            preds.append(logits[mask].argmax(1).cpu().numpy())
            gts.append(yb[mask].cpu().numpy())

    ypred, ytrue = np.concatenate(preds), np.concatenate(gts)
    f1w = f1_score(ytrue, ypred, average="weighted")
    f1_per_class = f1_score(ytrue, ypred, average=None, labels=np.unique(ytrue))
    acc = accuracy_score(ytrue, ypred)
    return f1w, f1_per_class, acc

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--model_tag", required=True)
    ap.add_argument("--layer", required=True)
    ap.add_argument("--pool", required=True)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    emb_base = DATA / "embeddings" / args.task / args.model_tag / args.layer / args.pool
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
        f1w, f1_cls, acc = train_and_eval(model, Xtr, ytr, Xva, yva, Xte, yte)
        row = dict(K=K, K_norm=K_norm, f1_weighted=f1w, accuracy=acc)
        for i, f1 in enumerate(f1_cls):
            row[f"f1_class{i}"] = f1
        rows.append(row)
        print(f"[K={K:3d}] F1w={f1w:.3f}, Acc={acc:.3f}, K_norm={K_norm:.3f}")

    df = pd.DataFrame(rows)
    out_dir = RESULT_ROOT / f"{args.task}_{args.model_tag}_{args.layer}_{args.pool}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "k_sweep_norm_peremo.csv", index=False)
    print(f"[OK] Saved CSV → {out_dir}/k_sweep_norm_peremo.csv")

    # ==== Per-emotion τ_rel fit ====
    taus = {}
    available_cols = [c for c in df.columns if c.startswith("f1_class")]
    for col in available_cols:
        c = int(col.replace("f1_class", ""))
        y = df[col].values
        x = df["K_norm"].values
        try:
            popt, _ = curve_fit(saturation_func, x, y, maxfev=10000)
            a,b,tau = popt
            taus[c] = tau
        except Exception:
            taus[c] = np.nan
    with open(out_dir / "tau_rel_per_emotion.json","w") as f:
        json.dump(taus, f, indent=2)
    print("τ_rel per emotion:", taus)

    # ==== Plot ====
    plt.figure(figsize=(7,4))
    for col in available_cols:
        c = int(col.replace("f1_class", ""))
        plt.plot(df["K_norm"], df[col], label=f"class{c}")
    plt.xlabel("Normalized Context Ratio (K / total_turns)")
    plt.ylabel("F1 per emotion")
    plt.title(f"{args.task} | {args.model_tag} | {args.layer}/{args.pool}")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "normalized_k_curve_peremo.png", dpi=300)
    print(f"[OK] Saved plot → {out_dir}/normalized_k_curve_peremo.png")

if __name__ == "__main__":
    main()

