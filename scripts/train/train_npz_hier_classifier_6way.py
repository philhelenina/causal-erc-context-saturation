#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_npz_hier_classifier_6way_v2.py
- Hierarchical: sentence → utterance → classification
- Aggregation: mean, sum, expdecay, attn, lstm
- Classifier: MLP, LSTM (NEW!)
"""
import os

import argparse, json, math, random
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

# -------- Paths --------
HOME = Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()
DATA_DIR = HOME / "data" / "iemocap_6way_data"
DEFAULT_EMB_ROOT = HOME / "data" / "embeddings" / "6way" / "sentence-roberta-hier"

# -------- Utils --------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

DEFAULT_STR_MAP = {
    "ang":0, "anger":0,
    "hap":1, "happy":1,
    "sad":2, "sadness":2,
    "neu":3, "neutral":3,
    "exc":4, "excited":4,
    "fru":5, "frustrated":5,
}

def parse_label_map_kv(kvs):
    m = {}
    for kv in kvs or []:
        if "=" in kv:
            k,v = kv.split("=",1)
            m[k.strip().lower()] = int(v)
    return m

def load_labels(split: str, label_col: str, user_map: dict):
    path = DATA_DIR / f"{split}_6way_unified.csv"
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise ValueError(f"{path} 에 '{label_col}' 컬럼이 없습니다.")
    raw = df[label_col]

    lab_num = pd.to_numeric(raw, errors="coerce")
    if lab_num.isna().all():
        lab_str = raw.astype(str).str.strip().str.lower()
        lab_num = lab_str.map({**DEFAULT_STR_MAP, **user_map})
    else:
        if lab_num.isna().any():
            lab_str = raw.astype(str).str.strip().str.lower()
            lab_num = lab_num.fillna(lab_str.map({**DEFAULT_STR_MAP, **user_map}))

    valid_mask = lab_num.notna() & (lab_num.astype("float64") >= 0)
    y = lab_num[valid_mask].astype("int64").to_numpy()
    info = {"csv_path": str(path), "n_csv": len(df), "n_valid": int(valid_mask.sum())}
    return y, valid_mask.to_numpy(), len(df), info

def load_npz_hier(emb_root: Path, layer: str, pool: str, split: str):
    p = emb_root / layer / pool / f"{split}.npz"
    if not p.exists():
        raise FileNotFoundError(f"NPZ not found: {p}")
    z = np.load(p)
    return z["embeddings"].astype(np.float32), z["lengths"].astype(np.int32)

def align_mask(X: np.ndarray, L: np.ndarray, mask: np.ndarray):
    if len(mask) != len(X):
        n = min(len(mask), len(X))
        X, L, mask = X[:n], L[:n], mask[:n]
    return X[mask], L[mask]

def class_weights(y: np.ndarray, device: torch.device):
    classes = np.array(sorted(set(y.tolist())))
    cnt = np.array([(y==c).sum() for c in classes], dtype=np.float32) + 1e-6
    w = cnt.sum() / cnt
    W = torch.ones(int(classes.max())+1, dtype=torch.float32)
    for c, wi in zip(classes, w): W[int(c)] = float(wi)
    return W.to(device)

# -------- Aggregation helpers (B,S,D) → (B,D) --------
def make_mask(B, S, lens, device):
    return (torch.arange(S, device=device).unsqueeze(0) < lens.unsqueeze(1))  # (B,S)

def agg_mean(xs, mask):
    denom = mask.sum(1, keepdim=True).clamp_min(1)
    return (xs * mask.unsqueeze(-1)).sum(1) / denom  # (B,D)

def agg_sum(xs, mask):
    return (xs * mask.unsqueeze(-1)).sum(1)  # (B,D)

def agg_expdecay(xs, mask, lam: float, reverse: bool = True):
    B, S, D = xs.shape
    pos = torch.arange(S, device=xs.device).float()
    if reverse:
        pos = (S - 1) - pos  # 뒤쪽(문말) 강조
    w = torch.exp(-lam * pos)  # (S,)
    w = w.unsqueeze(0).repeat(B,1) * mask.float()
    w = w / w.sum(1, keepdim=True).clamp_min(1e-6)
    return (xs * w.unsqueeze(-1)).sum(1)

# -------- Aggregator Modules --------
class MeanAggregator(nn.Module):
    def forward(self, xs, lens):
        B, S, D = xs.shape
        mask = make_mask(B, S, lens, xs.device)
        return agg_mean(xs, mask)

class SumAggregator(nn.Module):
    def forward(self, xs, lens):
        B, S, D = xs.shape
        mask = make_mask(B, S, lens, xs.device)
        return agg_sum(xs, mask)

class ExpdecayAggregator(nn.Module):
    def __init__(self, decay_lambda=0.5, decay_reverse=True):
        super().__init__()
        self.decay_lambda = decay_lambda
        self.decay_reverse = decay_reverse
    
    def forward(self, xs, lens):
        B, S, D = xs.shape
        mask = make_mask(B, S, lens, xs.device)
        return agg_expdecay(xs, mask, self.decay_lambda, self.decay_reverse)

class AttnAggregator(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn_w = nn.Linear(d_model, 1, bias=False)
    
    def forward(self, xs, lens):
        B, S, D = xs.shape
        mask = make_mask(B, S, lens, xs.device)
        scores = self.attn_w(xs).squeeze(-1)  # (B,S)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B,S,1)
        return (xs * weights).sum(1)  # (B,D)

class LSTMAggregator(nn.Module):
    def __init__(self, d_model, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(d_model, hidden_size, 1, batch_first=True)
        self.out_dim = hidden_size
    
    def forward(self, xs, lens):
        packed = pack_padded_sequence(xs, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return h[-1]  # (B, hidden_size)

# -------- Classifier Modules --------
class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

class LSTMClassifier(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, 1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_classes)
    
    def forward(self, x):
        # x: (B, D) → (B, 1, D) for LSTM
        x = x.unsqueeze(1)
        out, (h, _) = self.lstm(x)
        h = self.dropout(h[-1])
        return self.fc(h)

# -------- Full Model --------
class HierarchicalModel(nn.Module):
    def __init__(self, aggregator, classifier):
        super().__init__()
        self.aggregator = aggregator
        self.classifier = classifier
    
    def forward(self, xs, lens):
        # xs: (B, S, D)
        pooled = self.aggregator(xs, lens)  # (B, D') where D' = D or hidden_size
        logits = self.classifier(pooled)    # (B, num_classes)
        return logits

def build_model(agg_type: str, clf_type: str, d_model: int, hidden: int, 
                num_classes: int, dropout: float, decay_lambda: float, decay_reverse: bool):
    # Build aggregator
    if agg_type == "mean":
        aggregator = MeanAggregator()
        agg_out_dim = d_model
    elif agg_type == "sum":
        aggregator = SumAggregator()
        agg_out_dim = d_model
    elif agg_type == "expdecay":
        aggregator = ExpdecayAggregator(decay_lambda, decay_reverse)
        agg_out_dim = d_model
    elif agg_type == "attn":
        aggregator = AttnAggregator(d_model)
        agg_out_dim = d_model
    elif agg_type == "lstm":
        aggregator = LSTMAggregator(d_model, hidden)
        agg_out_dim = hidden
    else:
        raise ValueError(f"Unknown aggregator: {agg_type}")
    
    # Build classifier
    if clf_type == "mlp":
        classifier = MLPClassifier(agg_out_dim, hidden, num_classes, dropout)
    elif clf_type == "lstm":
        classifier = LSTMClassifier(agg_out_dim, hidden, num_classes, dropout)
    else:
        raise ValueError(f"Unknown classifier: {clf_type}")
    
    return HierarchicalModel(aggregator, classifier)

# -------- Train/Eval --------
def run_epoch(model, dl, device, criterion, train: bool):
    total = 0.0; preds=[]; golds=[]
    if train: model.train()
    else:     model.eval()
    optim = run_epoch.optim  # set outside

    for xb, lb, yb in dl:
        xb, lb, yb = xb.to(device).float(), lb.to(device).long(), yb.to(device).long()
        logits = model(xb, lb)
        loss = criterion(logits, yb)

        if train:
            optim.zero_grad(); loss.backward(); optim.step()

        total += float(loss.item()) * xb.size(0)
        preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
        golds.append(yb.detach().cpu().numpy())

    yhat = np.concatenate(preds, 0) if preds else np.array([])
    ytrue= np.concatenate(golds, 0) if golds else np.array([])
    return total / max(1, len(dl.dataset)), ytrue, yhat

def evaluate_metrics(ytrue: np.ndarray, yhat: np.ndarray):
    acc = accuracy_score(ytrue, yhat)
    mf1 = f1_score(ytrue, yhat, average="macro")
    wf1 = f1_score(ytrue, yhat, average="weighted")
    mp, mr, _, _ = precision_recall_fscore_support(ytrue, yhat, average="macro", zero_division=0)
    return acc, mf1, wf1, mp, mr

def save_confmat(ytrue, yhat, out_dir: Path, title: str, labels: List[str]):
    cm = confusion_matrix(ytrue, yhat, labels=list(range(len(labels))))
    fig = plt.figure(figsize=(5,5))
    plt.imshow(cm, interpolation='nearest'); plt.title(title); plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45); plt.yticks(ticks, labels)
    vmax = cm.max() if cm.size else 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm[i, j]
            plt.text(j, i, str(v), ha="center", va="center",
                     color=("white" if v > vmax/2 else "black"), fontsize=8)
    plt.tight_layout(); plt.xlabel("Pred"); plt.ylabel("True")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", required=True)
    ap.add_argument("--pool",  required=True)
    ap.add_argument("--aggregator", choices=["mean","sum","expdecay","attn","lstm"], required=True)
    ap.add_argument("--classifier", choices=["mlp","lstm"], required=True)
    ap.add_argument("--hidden_size", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--label_column", default="label")
    ap.add_argument("--label_map", nargs="*", default=[])
    ap.add_argument("--expected_labels", nargs="*", type=int, default=[0,1,2,3,4,5])
    #ap.add_argument("--out_dir", required=True)
    ap.add_argument("--l2_normalize", action="store_true")

    ap.add_argument("--decay_lambda", type=float, default=0.5)
    ap.add_argument("--decay_reverse", action="store_true", default=True)
    ap.add_argument("--no-decay_reverse", dest="decay_reverse", action="store_false")
    ap.add_argument("--emb_root", type=str, default="")

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    emb_root = Path(args.emb_root) if args.emb_root else DEFAULT_EMB_ROOT

    # Load data
    Xtr, Ltr = load_npz_hier(emb_root, args.layer, args.pool, "train")
    Xva, Lva = load_npz_hier(emb_root, args.layer, args.pool, "val")
    Xte, Lte = load_npz_hier(emb_root, args.layer, args.pool, "test")

    ytr, mtr, _, _ = load_labels("train", args.label_column, parse_label_map_kv(args.label_map))
    yva, mva, _, _ = load_labels("val",   args.label_column, parse_label_map_kv(args.label_map))
    yte, mte, _, _ = load_labels("test",  args.label_column, parse_label_map_kv(args.label_map))

    Xtr, Ltr = align_mask(Xtr, Ltr, mtr)
    Xva, Lva = align_mask(Xva, Lva, mva)
    Xte, Lte = align_mask(Xte, Lte, mte)

    if args.l2_normalize:
        def l2norm3d(x):
            n = np.linalg.norm(x, axis=2, keepdims=True) + 1e-12
            return x / n
        Xtr = l2norm3d(Xtr); Xva = l2norm3d(Xva); Xte = l2norm3d(Xte)

    num_classes = max(args.expected_labels) + 1 if args.expected_labels else int(max(ytr.max(), yva.max(), yte.max()) + 1)
    d_model = Xtr.shape[2]

    tr_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ltr), torch.from_numpy(ytr))
    va_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Lva), torch.from_numpy(yva))
    te_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xte), torch.from_numpy(Lte), torch.from_numpy(yte))
    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va_dl = torch.utils.data.DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)
    te_dl = torch.utils.data.DataLoader(te_ds, batch_size=args.batch_size, shuffle=False)

    model = build_model(args.aggregator, args.classifier, d_model, args.hidden_size, 
                        num_classes, args.dropout, args.decay_lambda, args.decay_reverse).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(ytr, device))
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    run_epoch.optim = optim  # type: ignore

    best_va, best_state, wait = math.inf, None, 0
    for ep in range(1, args.epochs+1):
        tr_loss, _, _ = run_epoch(model, tr_dl, device, criterion, train=True)
        va_loss, _, _ = run_epoch(model, va_dl, device, criterion, train=False)
        if ep % 20 == 0 or ep == 1:
            print(f"[{args.aggregator.upper()}-{args.classifier.upper()}] epoch {ep:4d} | train {tr_loss:.4f} | val {va_loss:.4f}")
        if va_loss < best_va - 1e-6:
            best_va = va_loss; best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[{args.aggregator}-{args.classifier}] early stopping at epoch {ep}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    _, yt, yht = run_epoch(model, te_dl, device, criterion, train=False)
    acc, mf1, wf1, mp, mr = evaluate_metrics(yt, yht)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    labels_txt = ['ang','hap','sad','neu','exc','fru']
    save_confmat(yt, yht, out_dir, f"{args.aggregator}-{args.classifier}", labels_txt)

    payload = {
        "embedding_type": "sentence-roberta-hier",
        "layer": args.layer, "pool": args.pool, 
        "aggregator": args.aggregator, "classifier": args.classifier,
        "seed": args.seed, "hidden_size": args.hidden_size,
        "dropout": args.dropout, "batch_size": args.batch_size,
        "learning_rate": args.lr, "weight_decay": args.weight_decay,
        "num_epochs": args.epochs, "early_stopping_patience": args.patience,
        "l2_normalize": args.l2_normalize,
        "decay_lambda": args.decay_lambda, "decay_reverse": args.decay_reverse,
        "metrics": {"accuracy": float(acc), "macro_f1": float(mf1),
                    "weighted_f1": float(wf1), "macro_precision": float(mp), "macro_recall": float(mr)}
    }
    with open(out_dir / "results.json", "w") as f: json.dump(payload, f, indent=2)

    print(f"[OK] Test: acc={acc:.4f}  macroF1={mf1:.4f}  weightedF1={wf1:.4f}")
    print(f"[OK] Saved → {out_dir}")

if __name__ == "__main__":
    main()