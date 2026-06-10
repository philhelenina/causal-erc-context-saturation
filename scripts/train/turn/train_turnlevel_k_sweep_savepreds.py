#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_turnlevel_k_sweep_norm_savepreds.py (final stable)
- K_norm sweep + turn-level test prediction (probabilities)
- Saves per-K softmax probs → (num_K, num_samples, num_classes)
- Compatible with analyze_information_flow.py
"""

import os, json, numpy as np, pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score

# ---------- 경로 ----------
HOME = Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()
DATA = HOME / "data"
RES_ROOT = HOME / "results" / "turnlevel_k_sweep_norm_savepreds"

# ---------- 유틸 ----------
def load_npz(p: Path):
    a = np.load(p, allow_pickle=True)
    X = a["embeddings"]; y = a["y"]
    ids = a["ids"] if "ids" in a else None
    return X, y, ids

def read_meta(task: str, split: str) -> pd.DataFrame:
    csv_dir = DATA / f"iemocap_{task}_data"
    csv_path = csv_dir / f"{split}_{task}_with_minus_one.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    id_cols = [c for c in ["id","utt_id","sample_id","orig_id","file_id","wav_id"] if c in df.columns]
    if not id_cols: raise ValueError(f"No id column in {csv_path}")
    id_col = id_cols[0]
    dlg_cols = [c for c in ["dialogue_id","dialog_id","conv_id","file_root","session_dialog_id","dlg_id","file_id_root"] if c in df.columns]
    if not dlg_cols: dlg_cols = [id_col]
    dlg_col = dlg_cols[0]
    turn_cols = [c for c in ["turn_index","turn_id","turn","utt_no","order","idx"] if c in df.columns]
    if not turn_cols:
        df["turn_index"] = df.groupby(dlg_col).cumcount()
        turn_col = "turn_index"
    else:
        turn_col = turn_cols[0]
    meta = df[[id_col, dlg_col, turn_col]].copy()
    meta.columns = ["id","dialogue","turn_idx"]
    meta["id"] = meta["id"].astype(str)
    meta["dialogue"] = meta["dialogue"].astype(str)
    meta["turn_idx"] = meta["turn_idx"].astype(int)
    return meta

def align_order(ids_npz, meta_df):
    if ids_npz is None:
        meta_df["npz_idx"] = np.arange(len(meta_df))
        return meta_df
    ids_npz = pd.Series(ids_npz.astype(str))
    indexer = pd.Series(np.arange(len(ids_npz)), index=ids_npz)
    if not meta_df["id"].isin(indexer.index).all():
        missing = meta_df.loc[~meta_df["id"].isin(indexer.index),"id"].head(3).tolist()
        raise ValueError(f"Missing ids in NPZ: {missing}")
    meta_df["npz_idx"] = meta_df["id"].map(indexer)
    meta_df = meta_df.sort_values(["dialogue","turn_idx"]).reset_index(drop=True)
    return meta_df

def build_sequences(X, y, order_df, K, pad_value=0.0):
    N, D = X.shape
    Xseq = np.zeros((N, K+1, D), dtype=X.dtype)
    yout = y.copy()
    dlg_id = order_df["dialogue"].to_numpy()
    t_idx  = order_df["turn_idx"].to_numpy()
    grp = order_df.groupby("dialogue").indices

    for row, (npz_i, dlg, t) in enumerate(order_df[["npz_idx","dialogue","turn_idx"]].itertuples(index=False)):
        idxs = order_df.loc[grp[dlg],"npz_idx"].to_numpy()
        turns = order_df.loc[grp[dlg],"turn_idx"].to_numpy()
        pos = int(np.where(turns == t)[0][0])
        start = max(0, pos - K)
        seq_idx = idxs[start:pos+1]
        pad_len = (K+1) - len(seq_idx)
        if pad_len > 0:
            Xseq[row, :pad_len, :] = pad_value
            Xseq[row, pad_len:, :] = X[seq_idx,:]
        else:
            Xseq[row, :, :] = X[seq_idx[-(K+1):], :]
    dlg_len = order_df.groupby("dialogue")["turn_idx"].transform("max").to_numpy() + 1
    return Xseq, yout, dlg_len, dlg_id, t_idx

# ---------- 모델 ----------
class SimpleSeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if x.dim()==2: x = x.unsqueeze(1)
        out,_ = self.lstm(x)
        h = self.dropout(out[:,-1,:])
        return self.fc(h)

def train_one(model, Xtr, ytr, Xva, yva, epochs=50, bs=64, lr=1e-3, device="cuda"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    def toT(X,y): return torch.tensor(X).float(), torch.tensor(y).long()
    Xtr,ytr = toT(Xtr,ytr); Xva,yva = toT(Xva,yva)
    tr_dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr,ytr), batch_size=bs, shuffle=True)
    va_dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xva,yva), batch_size=bs)
    best, state = 1e9, None
    for _ in range(epochs):
        model.train()
        for xb,yb in tr_dl:
            xb,yb = xb.to(device), yb.to(device)
            m = (yb >= 0)
            if m.sum()==0: continue
            opt.zero_grad()
            loss = crit(model(xb)[m], yb[m])
            loss.backward(); opt.step()
        model.eval(); v=0.0
        with torch.no_grad():
            for xb,yb in va_dl:
                xb,yb = xb.to(device), yb.to(device)
                m = (yb >= 0)
                if m.sum()==0: continue
                v += crit(model(xb)[m], yb[m]).item()
        v /= max(len(va_dl),1)
        if v < best: best, state = v, model.state_dict().copy()
    model.load_state_dict(state)
    return model

def predict_probs(model, Xte, bs=64, device="cuda"):
    Xte = torch.tensor(Xte).float()
    dl = torch.utils.data.DataLoader(Xte, batch_size=bs, shuffle=False)
    model.eval(); all_probs=[]
    with torch.no_grad():
        for xb in dl:
            xb = xb.to(device)
            out = model(xb)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)

# ---------- main ----------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["4way","6way"])
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--model_tag", required=True)
    ap.add_argument("--layer", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k_min", type=int, default=0)
    ap.add_argument("--k_max", type=int, default=100)
    ap.add_argument("--k_step", type=int, default=5)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 데이터 로드
    Xtr,ytr,idtr = load_npz(DATA/"embeddings"/args.task/args.model_tag/args.layer/args.pool/"train.npz")
    Xva,yva,idva = load_npz(DATA/"embeddings"/args.task/args.model_tag/args.layer/args.pool/"val.npz")
    Xte,yte,idte = load_npz(DATA/"embeddings"/args.task/args.model_tag/args.layer/args.pool/"test.npz")
    mtr = align_order(idtr, read_meta(args.task,"train"))
    mva = align_order(idva, read_meta(args.task,"val"))
    mte = align_order(idte, read_meta(args.task,"test"))

    Ks = list(range(args.k_min, args.k_max+1, args.k_step))
    mte_valid = mte.assign(y=yte[mte["npz_idx"].to_numpy()])
    mte_valid = mte_valid[mte_valid["y"] >= 0].copy().sort_values(["dialogue","turn_idx"])
    valid_npz_idx = mte_valid["npz_idx"].to_numpy()
    y_ref = yte[valid_npz_idx].astype(int)

    dlg_len_map = (mte.groupby("dialogue")["turn_idx"].max()+1).to_dict()
    sample_dlg = mte_valid["dialogue"].astype(str).tolist()
    sample_turn= mte_valid["turn_idx"].to_numpy()
    sample_dlen= np.array([dlg_len_map[d] for d in sample_dlg],dtype=int)

    D = Xtr.shape[1]
    num_classes = int(np.max(ytr[ytr>=0])) + 1
    total_turns_tr = len(ytr)

    all_probs=[]  # (num_K, N, C)

    for j,K in enumerate(Ks):
        Xtr_seq,ytr_out,_,_,_ = build_sequences(Xtr,ytr,mtr,K)
        Xva_seq,yva_out,_,_,_ = build_sequences(Xva,yva,mva,K)
        Xte_seq,yte_out,_,_,_ = build_sequences(Xte,yte,mte,K)
        model = SimpleSeqLSTM(D,256,num_classes,dropout=0.3)
        model = train_one(model,Xtr_seq,ytr_out,Xva_seq,yva_out,
                          epochs=args.epochs,bs=args.bs,lr=args.lr,device=device)
        probs = predict_probs(model, Xte_seq, bs=args.bs, device=device)
        all_probs.append(probs)
        preds = probs.argmax(1)
        acc = accuracy_score(yte_out[yte_out>=0], preds[yte_out>=0])
        f1w = f1_score(yte_out[yte_out>=0], preds[yte_out>=0], average="weighted", zero_division=0)
        print(f"[K={K:3d}] F1w={f1w:.3f}, Acc={acc:.3f}, K_norm={K/total_turns_tr:.3f}")

    out_dir = RES_ROOT / f"{args.task}_{args.model_tag}_{args.layer}_{args.pool}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir/"preds_perK.npy", np.stack(all_probs, axis=0))
    np.save(out_dir/"labels.npy", y_ref)
    np.save(out_dir/"Ks.npy", np.array(Ks))
    np.save(out_dir/"K_norm.npy", np.array([k/total_turns_tr for k in Ks]))
    np.save(out_dir/"dialog_len.npy", sample_dlen)
    np.save(out_dir/"dialogue_id.npy", np.array(sample_dlg))
    np.save(out_dir/"turn_idx.npy", sample_turn)
    print(f"[OK] Saved → {out_dir} (preds_perK.npy shape={np.stack(all_probs,axis=0).shape})")

if __name__ == "__main__":
    main()
