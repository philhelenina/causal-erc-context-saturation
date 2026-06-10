#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_npz_hier_fused_classifier.py
- 계층 SRoBERTa × Lexical (SenticNet/WNA 등) 퓨전 임베딩 (N,T,D) 학습
- 4way/6way 공용, CE/FocalLoss, 클래스 가중치, W&B(옵션), CM/리포트 저장
"""
import os

import argparse, json, math, random
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

HOME = Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()
SPLITS = ["train","val","test"]

# ---------------- Utils ----------------
def set_seed(s:int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_npz_any(p: Path) -> np.ndarray:
    arr = np.load(p, allow_pickle=False)
    if isinstance(arr, np.lib.npyio.NpzFile):
        return arr["embeddings"] if "embeddings" in arr else arr[list(arr.keys())[0]]
    return arr

def map_labels(df: pd.DataFrame, task: str) -> np.ndarray:
    if "label_num" in df.columns:
        y = pd.to_numeric(df["label_num"], errors="coerce").fillna(-1).astype("int64").to_numpy()
        return y
    if "label" in df.columns:
        s = df["label"].astype(str).str.strip().str.lower()
        mp4 = {"ang":0,"hap":1,"sad":2,"neu":3}
        mp6 = {"ang":0,"hap":1,"sad":2,"neu":3,"exc":4,"fru":5}
        mp = mp6 if task=="6way" else mp4
        return s.map(mp).fillna(-1).astype("int64").to_numpy()
    raise ValueError("CSV missing label/label_num")

def load_labels(task: str, split: str):
    csv_root = HOME/"data"/f"iemocap_{task}_data"
    csv = csv_root/f"{split}_{task}_with_minus_one.csv"
    df = pd.read_csv(csv)
    y = map_labels(df, task)
    m = (y >= 0)
    return y, m, str(csv)

def align_xy(X: np.ndarray, y: np.ndarray, m: np.ndarray, name: str):
    if len(X) != len(m):
        n = min(len(X), len(m))
        print(f"[WARN] {name}: X rows={len(X)} != CSV rows={len(m)} → trim to {n}")
        X, y, m = X[:n], y[:n], m[:n]
    return X[m], y[m]

def pad_truncate_time_to(X: np.ndarray, T_target: int) -> np.ndarray:
    """(N,T,D) → (N,T_target,D)  ;  2D는 그대로 반환"""
    if X.ndim == 2:  # (N,D)
        return X
    N,T,D = X.shape
    if T == T_target:
        return X
    if T > T_target:
        return X[:, :T_target, :]
    # pad with zeros at tail
    out = np.zeros((N, T_target, D), dtype=X.dtype)
    out[:, :T, :] = X
    return out

def unify_time_dim(Xtr: np.ndarray, Xva: np.ndarray, Xte: np.ndarray):
    """세 split의 T를 공통 T_max로 맞춘다."""
    Ts = [x.shape[1] for x in (Xtr, Xva, Xte) if x.ndim == 3]
    if not Ts:
        return Xtr, Xva, Xte, None
    T_max = int(max(Ts))
    Xtr2 = pad_truncate_time_to(Xtr, T_max)
    Xva2 = pad_truncate_time_to(Xva, T_max)
    Xte2 = pad_truncate_time_to(Xte, T_max)
    return Xtr2, Xva2, Xte2, T_max

# ---------------- Models ----------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, drop):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x):
        if x.dim()==3:
            B,T,D = x.shape
            x = x.reshape(B, T*D)
        return self.net(x)

class LSTMClassifier(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, drop):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden, num_layers=1,
                            batch_first=True, bidirectional=False, dropout=0.0)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(hidden, num_classes))
    def forward(self, x):
        if x.dim()==2:   # (B,D) → (B,1,D)
            x = x.unsqueeze(1)
        out,_ = self.lstm(x)        # (B,T,H)
        return self.head(out[:,-1,:])

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.reduction = reduction
    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1-pt)**self.gamma) * ce
        return loss.mean() if self.reduction=='mean' else loss.sum()

def class_weight_tensor(y: np.ndarray, num_classes: int, device):
    cnts = np.bincount(y, minlength=num_classes).astype(np.float32) + 1e-6
    w = cnts.sum() / cnts
    return torch.tensor(w, dtype=torch.float32, device=device)

# ---------------- Train/Eval ----------------
def train_loop(args, model, Xtr, ytr, Xva, yva, device):
    num_classes = 6 if args.task=="6way" else 4
    alpha = class_weight_tensor(ytr, num_classes, device) if args.use_class_weight else None
    criterion = FocalLoss(alpha=alpha, gamma=args.focal_gamma) if args.loss=="focal" \
               else nn.CrossEntropyLoss(weight=alpha)
    tr_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    va_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va_dl = torch.utils.data.DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best = {"val": 1e9, "state": None, "epoch": 0}
    wait = 0
    for ep in range(1, args.num_epochs+1):
        model.train(); total=0.0
        for xb,yb in tr_dl:
            xb,yb = xb.to(device).float(), yb.to(device).long()
            optim.zero_grad(); loss = criterion(model(xb), yb)
            loss.backward(); optim.step()
            total += float(loss.item()) * xb.size(0)
        tr_loss = total / len(tr_ds)

        model.eval(); total=0.0
        with torch.no_grad():
            for xb,yb in va_dl:
                xb,yb = xb.to(device).float(), yb.to(device).long()
                total += float(criterion(model(xb), yb).item()) * xb.size(0)
        va_loss = total / len(va_ds)

        if ep%10==0 or ep==1:
            print(f"[{args.model.upper()}] ep {ep:03d} | tr {tr_loss:.4f} | va {va_loss:.4f}")

        if va_loss + 1e-6 < best["val"]:
            best.update(val=va_loss, state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, epoch=ep)
            wait = 0
        else:
            wait += 1
            if wait >= args.early_stopping_patience:
                print(f"[ES] stop at ep={ep}")
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, best["epoch"]

def eval_and_save(args, model, Xte, yte, out_dir: Path, name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    te_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
    te_dl = torch.utils.data.DataLoader(te_ds, batch_size=args.batch_size, shuffle=False)

    preds=[]
    with torch.no_grad():
        for xb,_ in te_dl:
            xb = xb.to(device).float()
            logits = model(xb)
            preds.append(torch.argmax(logits,1).cpu().numpy())
    yhat = np.concatenate(preds,0)

    acc = accuracy_score(yte, yhat)
    f1m = f1_score(yte, yhat, average="macro")
    f1w = f1_score(yte, yhat, average="weighted")
    mp, mr, _, _ = precision_recall_fscore_support(yte, yhat, average="macro", zero_division=0)
    cm = confusion_matrix(yte, yhat)
    classes = 6 if args.task=="6way" else 4
    labels = ["ang","hap","sad","neu","exc","fru"] if classes==6 else ["ang","hap","sad","neu"]

    # Confusion matrix
    fig = plt.figure(figsize=(4.8,4.8))
    plt.imshow(cm, interpolation="nearest"); plt.title(f"CM: {name}"); plt.colorbar()
    ticks = np.arange(classes)
    plt.xticks(ticks, labels, rotation=45); plt.yticks(ticks, labels)
    vmax = cm.max() if cm.size else 1
    for i in range(classes):
        for j in range(classes):
            v = cm[i,j]
            plt.text(j,i,str(v), ha="center", va="center",
                     fontsize=8, color=("white" if v>vmax/2 else "black"))
    plt.tight_layout(); plt.xlabel("Pred"); plt.ylabel("True")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"confusion_matrix.png").unlink(missing_ok=True)
    fig.savefig(out_dir/"confusion_matrix.png", dpi=150); plt.close(fig)

    # Report
    rpt = classification_report(yte, yhat, target_names=labels, digits=4, zero_division=0)
    with open(out_dir/"cls_report.txt","w") as f: f.write(rpt)

    payload = dict(
        task=args.task, root_tag=args.root_tag, fused_mode=args.fused_mode,
        layer=args.layer, pool=args.pool, model=args.model, seed=args.seed,
        hidden_size=args.hidden_size, dropout_rate=args.dropout_rate,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience, loss=args.loss,
        use_class_weight=args.use_class_weight, focal_gamma=args.focal_gamma,
        metrics=dict(accuracy=float(acc), macro_f1=float(f1m), weighted_f1=float(f1w),
                     macro_precision=float(mp), macro_recall=float(mr)),
        shapes=dict(train=tuple(Xtr_shape_cache), val=tuple(Xva_shape_cache), test=tuple(Xte_shape_cache)),
        paths=dict(cm=str((out_dir/"confusion_matrix.png").resolve()),
                   report=str((out_dir/"cls_report.txt").resolve()))
    )
    with open(out_dir/"results.json","w") as f: json.dump(payload, f, indent=2)
    print(f"[OK] Saved → {out_dir/'results.json'}  acc={acc:.3f} f1w={f1w:.3f} f1m={f1m:.3f}")

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["4way","6way"], required=True)
    ap.add_argument("--root_tag", required=True, help="예: sentence-roberta-hier-senticnet")
    ap.add_argument("--fused_mode", choices=["concat","proj128","zeropad768"], required=True)
    ap.add_argument("--layer", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--model", choices=["mlp","lstm"], default="mlp")
    # Stage-2 HP (Bayes opt 결과)
    ap.add_argument("--hidden_size", type=int, default=192)
    ap.add_argument("--dropout_rate", type=float, default=0.7129)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--learning_rate", type=float, default=0.0002155)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--num_epochs", type=int, default=69)
    ap.add_argument("--early_stopping_patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--loss", choices=["ce","focal"], default="ce")
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--use_class_weight", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="")
    ap.add_argument("--wandb_run_prefix", type=str, default="")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load X
    ROOT = HOME/"data"/"embeddings"/args.task/args.root_tag/f"fused_{args.fused_mode}"/args.layer/args.pool
    Xtr = load_npz_any(ROOT/"train.npz").astype(np.float32)
    Xva = load_npz_any(ROOT/"val.npz").astype(np.float32)
    Xte = load_npz_any(ROOT/"test.npz").astype(np.float32)

    # Labels
    ytr,mtr,_ = load_labels(args.task,"train")
    yva,mva,_ = load_labels(args.task,"val")
    yte,mte,_ = load_labels(args.task,"test")

    # Align/mask
    Xtr, ytr = align_xy(Xtr, ytr, mtr, "train")
    Xva, yva = align_xy(Xva, yva, mva, "val")
    Xte, yte = align_xy(Xte, yte, mte, "test")

    # Cache original shapes (for results.json)
    global Xtr_shape_cache, Xva_shape_cache, Xte_shape_cache
    Xtr_shape_cache, Xva_shape_cache, Xte_shape_cache = Xtr.shape, Xva.shape, Xte.shape

    # ★ 핵심: T 통일 (Train/Val/Test)
    Xtr, Xva, Xte, T_max = unify_time_dim(Xtr, Xva, Xte)
    if T_max is not None:
        print(f"[INFO] unified T to {T_max} across splits")

    # 입력 차원 설정
    if args.model == "mlp":
        in_dim = int(Xtr.shape[-1]) if Xtr.ndim==2 else int(Xtr.shape[1] * Xtr.shape[2])
    else:
        in_dim = int(Xtr.shape[-1])  # LSTM은 특성 차원 D만 필요

    print(f"[INFO] Xtr={tuple(Xtr.shape)} Xva={tuple(Xva.shape)} Xte={tuple(Xte.shape)} | in_dim={in_dim} | model={args.model}")

    # Model
    num_classes = 6 if args.task=="6way" else 4
    model = (MLP(in_dim, args.hidden_size, num_classes, args.dropout_rate)
             if args.model=="mlp" else
             LSTMClassifier(in_dim, args.hidden_size, num_classes, args.dropout_rate))
    model.to(device)

    # W&B (옵션)
    use_wb = bool(args.wandb_project)
    if use_wb:
        try:
            import wandb
            wandb.init(project=args.wandb_project,
                       name=f"{args.wandb_run_prefix}{args.task}/{args.layer}/{args.pool}/{args.fused_mode}/{args.model}/seed{args.seed}",
                       config=vars(args))
        except Exception as e:
            print(f"[WARN] wandb init failed: {e}")
            use_wb=False

    model, best_ep = train_loop(args, model, Xtr, ytr, Xva, yva, device)
    out_dir = Path(args.out_dir)
    name = f"{args.task}:{args.layer}/{args.pool}/{args.fused_mode}"
    eval_and_save(args, model, Xte, yte, out_dir, name)

    if use_wb:
        try:
            import wandb; wandb.finish()
        except Exception:
            pass

if __name__ == "__main__":
    main()
