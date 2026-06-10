#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_npz_classifier_6way_lexical.py
- IEMOCAP 6-way 전용(ang,hap,sad,neu,exc,fru; -1은 제외)
- 임베딩: /data/embeddings/6way/{lexical_idf|lexical_noidf}/<variant>/{train,val,test}.npz
  * <variant>는 'w2v-avg' 또는 'w2v-wna-blend/a0.7_t0.20_b0.2' 같은 하위 경로 포함 가능
- CSV:    /data/iemocap_6way_data/{split}_6way_with_minus_one.csv
- 저장:   <out_dir>/results.json (+ confusion_matrix.png)

사용 예:
python3 train_npz_classifier_6way_lexical.py \
  --lex_set idf \
  --variant w2v-wna-blend/a0.7_t0.20_b0.2 \
  --model mlp --hidden_size 256 --dropout_rate 0.30 \
  --batch_size 64 --learning_rate 1e-3 --num_epochs 200 --early_stopping_patience 60 \
  --seed 42 --label_col label_num \
  --out_dir /.../results/6way/npz_lexical_grid/idf/w2v-wna-blend/a0.7_t0.20_b0.2/mlp/seed42
"""
from __future__ import annotations
import os
import argparse, json, math, random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt

# ---------------- Paths ----------------
HOME = Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()
DATA6 = HOME / "data" / "iemocap_6way_data"
EMB6  = HOME / "data" / "embeddings" / "6way"

# ---------------- Label map (6-way) ----------------
LABEL_MAP_6 = {
    "ang":0,"anger":0,
    "hap":1,"happy":1,
    "sad":2,"sadness":2,
    "neu":3,"neutral":3,
    "exc":4,"excited":4,
    "fru":5,"frustrated":5,
    "-1":-1, -1:-1, "undefined":-1
}
VALID_SET_6 = {0,1,2,3,4,5}
SPLITS = ["train","val","test"]

# ---------------- Utils ----------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_labels(split: str, label_col: str | None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      y_all: np.int64 array (length=N), -1 포함 가능
      mask_valid: bool array (y in {0..5})
    """
    csv_path = DATA6 / f"{split}_6way_with_minus_one.csv"
    df = pd.read_csv(csv_path)

    # 열 선택: label_col > label_num > label
    col = None
    if label_col and (label_col in df.columns):
        col = label_col
    elif "label_num" in df.columns:
        col = "label_num"
    elif "label" in df.columns:
        col = "label"
    else:
        raise ValueError(f"{csv_path}에 'label_num' 또는 'label' 열이 없습니다. cols={df.columns.tolist()[:12]}")

    ser = df[col]
    if ser.dtype == object:
        ser2 = ser.astype(str).str.strip().str.lower().map(LABEL_MAP_6)
    else:
        ser2 = pd.to_numeric(ser, errors="coerce")
        if ser2.isna().all():
            ser2 = ser.astype(str).str.strip().str.lower().map(LABEL_MAP_6)

    y = ser2.to_numpy()
    y = np.where(pd.isna(y), -1, y).astype("int64")
    mask_valid = np.isin(y, list(VALID_SET_6))

    uniq, cnts = np.unique(y, return_counts=True)
    print(f"[Label] {split} counts:", dict(zip(uniq, cnts)))
    return y, mask_valid

def resolve_npz_path(split: str, lex_set: str, variant: str) -> Path:
    """
    lex_set: 'idf' | 'noidf' | 'auto'
    variant: 'w2v-avg' or 'w2v-wna-blend/a0.7_t0.20_b0.2' 등
    """
    cand_dirs = []
    if lex_set == "idf":
        cand_dirs = [EMB6 / "lexical_idf"]
    elif lex_set == "noidf":
        cand_dirs = [EMB6 / "lexical_noidf"]
    else:  # auto
        cand_dirs = [EMB6 / "lexical_idf", EMB6 / "lexical_noidf", EMB6 / "lexical"]  # 마지막은 심볼릭 호환

    for root in cand_dirs:
        p = root / variant / f"{split}.npz"
        if p.exists():
            return p
    raise FileNotFoundError(f"NPZ not found for split={split}, variant={variant}, lex_set={lex_set} in {cand_dirs}")

def load_split_Xy(split: str, lex_set: str, variant: str, label_col: str | None):
    npz_path = resolve_npz_path(split, lex_set, variant)
    data = np.load(npz_path)
    X = data["embeddings"] if "embeddings" in data else data[list(data.keys())[0]]
    y_all, mask_valid = load_labels(split, label_col)

    n = min(len(X), len(y_all))
    X = X[:n]; y_all = y_all[:n]; mask_valid = mask_valid[:n]

    X = X[mask_valid]
    y = y_all[mask_valid].astype("int64")
    print(f"[Data Loaded] {split}: X_kept={X.shape[0]}/{n}, dim={X.shape[1]}  ({npz_path})")
    return X, y

def make_class_weights(y: np.ndarray, device: torch.device) -> torch.Tensor:
    classes = np.array(sorted(set(y.tolist())))
    counts  = np.array([(y == c).sum() for c in classes], dtype=np.float32) + 1e-6
    weights = counts.sum() / counts
    max_c = int(classes.max()) if classes.size else 0
    w_tensor = torch.ones(max_c + 1, dtype=torch.float32)
    for c, w in zip(classes, weights):
        w_tensor[int(c)] = float(w)
    return w_tensor.to(device)

# ---------------- Models ----------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
    def forward(self, x):
        if x.dim() == 3:  # (B,T,D) → (B, T*D)
            B,T,D = x.shape
            x = x.reshape(B, T*D)
        return self.net(x)

class LSTMClassifier(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden, num_layers=1,
                            batch_first=True, bidirectional=False, dropout=0.0)
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_classes))
    def forward(self, x):
        if x.dim() == 2:  # (B,D) → (B,1,D)
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        return self.proj(out[:, -1, :])

def train_one(args, Xtr, ytr, Xva, yva, in_dim: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 6
    model = MLP(in_dim, args.hidden_size, num_classes, args.dropout_rate) if args.model=="mlp" \
            else LSTMClassifier(in_dim, args.hidden_size, num_classes, args.dropout_rate)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=make_class_weights(ytr, device))
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    tr_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    va_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va_dl = torch.utils.data.DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)

    best_state, best_va, wait = None, math.inf, 0
    for ep in range(1, args.num_epochs + 1):
        model.train(); total = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device).float(), yb.to(device).long()
            optim.zero_grad(); logits = model(xb)
            loss = criterion(logits, yb); loss.backward(); optim.step()
            total += float(loss.item()) * xb.size(0)
        tr_loss = total / len(tr_ds)

        model.eval(); total = 0.0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(device).float(), yb.to(device).long()
                logits = model(xb)
                loss = criterion(logits, yb)
                total += float(loss.item()) * xb.size(0)
        va_loss = total / len(va_ds)

        if ep % 20 == 0 or ep == 1:
            print(f"[{args.model.upper()}] epoch {ep:4d} | train {tr_loss:.4f} | val {va_loss:.4f}")

        if va_loss < best_va - 1e-6:
            best_va = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.early_stopping_patience:
                print(f"[{args.model}] early stopping at epoch {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def eval_and_save(args, model, Xte, yte, out_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    te_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
    te_dl = torch.utils.data.DataLoader(te_ds, batch_size=args.batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for xb, _ in te_dl:
            xb = xb.to(device).float()
            logits = model(xb)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    yhat = np.concatenate(preds, axis=0)

    acc = accuracy_score(yte, yhat)
    macro_f1  = f1_score(yte, yhat, average="macro")
    weighted_f1 = f1_score(yte, yhat, average="weighted")
    macro_p, macro_r, _, _ = precision_recall_fscore_support(yte, yhat, average="macro", zero_division=0)

    cm = confusion_matrix(yte, yhat, labels=[0,1,2,3,4,5])
    fig = plt.figure(figsize=(4.8,4.8))
    plt.imshow(cm, interpolation='nearest')
    plt.title(f"CM: {args.variant} | {args.model}")
    plt.colorbar()
    tick = np.arange(6)
    plt.xticks(tick, ['ang','hap','sad','neu','exc','fru'], rotation=45)
    plt.yticks(tick, ['ang','hap','sad','neu','exc','fru'])
    for i in range(6):
        for j in range(6):
            v = cm[i, j]
            plt.text(j, i, v, ha="center", va="center",
                     fontsize=8, color=("white" if v > cm.max()/2 else "black"))
    plt.tight_layout(); plt.xlabel("Pred"); plt.ylabel("True")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "confusion_matrix.png").parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150); plt.close(fig)

    payload = {
        "embedding_type": "lexical",
        "lex_set": args.lex_set,
        "variant": args.variant,
        "model": args.model,
        "seed": args.seed,
        "hidden_size": args.hidden_size,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "dropout_rate": args.dropout_rate,
        "label_col": args.label_col,
        "metrics": {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
        },
        "paths": {"confusion_matrix_png": str((out_dir / "confusion_matrix.png").resolve())}
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[OK] Results saved to {out_dir / 'results.json'}")

# ---------------- Main ----------------
def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lex_set", choices=["idf","noidf","auto"], default="auto",
                    help="lexical_idf / lexical_noidf 선택(기본 auto: 존재하는 쪽 사용)")
    ap.add_argument("--variant", required=True,
                    help="예: 'w2v-avg' 또는 'w2v-wna-blend/a0.7_t0.20_b0.2'")
    ap.add_argument("--model", choices=["mlp","lstm"], default="mlp")
    ap.add_argument("--hidden_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--num_epochs", type=int, default=200)
    ap.add_argument("--early_stopping_patience", type=int, default=60)
    ap.add_argument("--dropout_rate", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label_col", type=str, default="", help="우선 사용할 라벨 열명(예: 'label_num'); 비우면 자동탐지")
    ap.add_argument("--out_dir", required=True, type=str)
    return ap

def main():
    ap = build_parser()
    args = ap.parse_args()
    set_seed(args.seed)

    # load data
    Xtr, ytr = load_split_Xy("train", args.lex_set, args.variant, args.label_col or None)
    Xva, yva = load_split_Xy("val",   args.lex_set, args.variant, args.label_col or None)
    Xte, yte = load_split_Xy("test",  args.lex_set, args.variant, args.label_col or None)

    if len(Xtr) == 0:
        print("Error: No training data loaded after filtering. Please check your data and labels.")
        return

    in_dim = Xtr.shape[-1]
    print(f"[INFO] in_dim={in_dim}, model={args.model}")

    model = train_one(args, Xtr, ytr, Xva, yva, in_dim)
    eval_and_save(args, model, Xte, yte, Path(args.out_dir))

if __name__ == "__main__":
    main()
