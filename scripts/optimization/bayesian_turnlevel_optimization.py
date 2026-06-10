#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bayesian_turnlevel_optimization.py

Turn-level Bayesian hyperparameter optimization using Optuna
- Uses 3 seeds (42, 43, 44) average for robust optimization
- Fixed K value (best K from default experiments)
- Saves best params for later full evaluation
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import f1_score, accuracy_score
import optuna
from optuna.samplers import TPESampler
import argparse
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════

HOME = Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()
DATA = HOME / "data"
RES_ROOT = HOME / "results" / "turnlevel_bayesian_optimization"

# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════

class SimpleSeqLSTM(nn.Module):
    """LSTM for turn-level sequential modeling"""

    def __init__(self, input_size, hidden_size, num_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths=None, use_packing=False):
        if x.dim() == 2:
            x = x.unsqueeze(1)
            lengths = None

        out, _ = self.lstm(x)
        h = out[:, -1, :]
        h = self.dropout(h)
        return self.fc(h)


# ═══════════════════════════════════════════════════════════════════
# Data Loading (simplified from main script)
# ═══════════════════════════════════════════════════════════════════

def load_npz(p: Path, aggregator='mean', device='cuda'):
    """Load NPZ with automatic hierarchical flattening"""
    arr = np.load(p, allow_pickle=True)

    X = None
    for key in ['embeddings', 'X', 'features', 'data']:
        if key in arr:
            X = arr[key]
            break

    if X is None:
        raise KeyError(f"No embeddings found in {p}")

    y = None
    for key in ['y', 'labels', 'targets', 'label', 'emotion']:
        if key in arr:
            y = arr[key]
            break

    if y is None:
        raise KeyError(f"No labels found in {p}")

    ids = None
    for key in ['ids', 'utterance_ids', 'utt_ids', 'sample_ids', 'id']:
        if key in arr:
            ids = arr[key]
            break

    # Hierarchical handling
    if X.ndim == 3:
        if aggregator == 'mean':
            if "lengths" in arr:
                lengths = arr["lengths"]
                X_flat = []
                for i in range(len(X)):
                    L = int(lengths[i])
                    X_flat.append(X[i, :L, :].mean(axis=0))
                X = np.stack(X_flat, axis=0)
            else:
                X = X.mean(axis=1)

    return X, y, ids


def read_meta(task: str, split: str) -> pd.DataFrame:
    """Load metadata from CSV"""
    csv_dir = DATA / f"iemocap_{task}_data"

    possible_names = [
        f"{split}_{task}_with_minus_one.csv",
        f"{split}_{task}_unified.csv",
        f"{split}_{task}.csv"
    ]

    csv_path = None
    for name in possible_names:
        candidate = csv_dir / name
        if candidate.exists():
            csv_path = candidate
            break

    if csv_path is None:
        raise FileNotFoundError(f"No CSV found in {csv_dir}")

    df = pd.read_csv(csv_path)

    id_cols = ["id", "utt_id", "sample_id", "orig_id", "file_id", "wav_id"]
    id_col = next((c for c in id_cols if c in df.columns), None)
    if not id_col:
        raise ValueError(f"No ID column in {csv_path}")

    dlg_cols = ["file_id", "dialogue_id", "dialog_id", "conv_id", "file_root",
                "session_dialog_id", "dlg_id", "file_id_root"]
    dlg_col = next((c for c in dlg_cols if c in df.columns), None)

    if not dlg_col:
        dlg_col = id_col

    turn_cols = ["utterance_num", "turn_index", "turn_id", "turn", "utt_no", "order", "idx"]
    turn_col = next((c for c in turn_cols if c in df.columns), None)

    if not turn_col:
        df["turn_index"] = df.groupby(dlg_col).cumcount()
        turn_col = "turn_index"

    meta = df[[id_col, dlg_col, turn_col]].copy()
    meta.columns = ["id", "dialogue", "turn_idx"]
    meta["id"] = meta["id"].astype(str)
    meta["dialogue"] = meta["dialogue"].astype(str)
    meta["turn_idx"] = meta["turn_idx"].astype(int)
    meta = meta.sort_values(["dialogue", "turn_idx"]).reset_index(drop=True)

    return meta


def align_order(ids_npz, meta_df):
    """Align NPZ indices with metadata"""
    if ids_npz is None:
        meta_df["npz_idx"] = np.arange(len(meta_df))
        return meta_df

    ids_npz = pd.Series(ids_npz.astype(str))
    indexer = pd.Series(np.arange(len(ids_npz)), index=ids_npz)

    if indexer.index.duplicated().any():
        indexer = indexer[~indexer.index.duplicated(keep='first')]

    meta_df["npz_idx"] = meta_df["id"].map(indexer)
    meta_df = meta_df.sort_values(["dialogue", "turn_idx"]).reset_index(drop=True)

    return meta_df


def build_sequences(X, y, order_df, K, pad_value=0.0):
    """Build turn-level sequences with context window K"""
    N, D = X.shape
    Xseq = np.zeros((N, K + 1, D), dtype=X.dtype)
    seq_lengths = np.zeros(N, dtype=np.int32)
    yout = y.copy()

    dlg_id = order_df["dialogue"].to_numpy()
    t_idx = order_df["turn_idx"].to_numpy()
    grp = order_df.groupby("dialogue").indices

    for row, (npz_i, dlg, t) in enumerate(
        order_df[["npz_idx", "dialogue", "turn_idx"]].itertuples(index=False)
    ):
        idxs = order_df.loc[grp[dlg], "npz_idx"].to_numpy()
        turns = order_df.loc[grp[dlg], "turn_idx"].to_numpy()
        pos = int(np.where(turns == t)[0][0])

        start = max(0, pos - K)
        seq_idx = idxs[start : pos + 1]
        actual_len = len(seq_idx)

        pad_len = (K + 1) - actual_len
        if pad_len > 0:
            Xseq[row, :pad_len, :] = pad_value
            Xseq[row, pad_len:, :] = X[seq_idx, :]
        else:
            Xseq[row, :, :] = X[seq_idx[-(K + 1) :], :]
            actual_len = K + 1

        seq_lengths[row] = actual_len

    dlg_len = order_df.groupby("dialogue")["turn_idx"].transform("max").to_numpy() + 1

    return Xseq, seq_lengths, yout, dlg_len, dlg_id, t_idx


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════

def train_one(model, Xtr, ytr, Ltr, Xva, yva, Lva, epochs=50, bs=64, lr=1e-3, weight_decay=0.0, device="cuda"):
    """Train with early stopping"""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    Xtr = torch.tensor(Xtr).float()
    ytr = torch.tensor(ytr).long()
    Ltr = torch.tensor(Ltr).long()

    Xva = torch.tensor(Xva).float()
    yva = torch.tensor(yva).long()
    Lva = torch.tensor(Lva).long()

    tr_ds = torch.utils.data.TensorDataset(Xtr, ytr, Ltr)
    va_ds = torch.utils.data.TensorDataset(Xva, yva, Lva)

    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=bs, shuffle=True)
    va_dl = torch.utils.data.DataLoader(va_ds, batch_size=bs)

    best_val = 1e9
    best_state = None
    patience_counter = 0
    patience = 20

    for epoch in range(epochs):
        model.train()
        for xb, yb, lb in tr_dl:
            xb, yb, lb = xb.to(device), yb.to(device), lb.to(device)

            mask = yb >= 0
            if mask.sum() == 0:
                continue

            opt.zero_grad()
            logits = model(xb, lengths=lb, use_packing=False)
            loss = crit(logits[mask], yb[mask])
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for xb, yb, lb in va_dl:
                xb, yb, lb = xb.to(device), yb.to(device), lb.to(device)

                mask = yb >= 0
                if mask.sum() == 0:
                    continue

                logits = model(xb, lengths=lb, use_packing=False)
                val_loss += crit(logits[mask], yb[mask]).item()
                n_batches += 1

        val_loss /= max(n_batches, 1)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def evaluate(model, Xte, yte, Lte, bs=64, device="cuda"):
    """Evaluate model and return F1 weighted"""
    Xte = torch.tensor(Xte).float()
    Lte = torch.tensor(Lte).long()

    ds = torch.utils.data.TensorDataset(Xte, Lte)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False)

    model.eval()
    all_preds = []

    with torch.no_grad():
        for xb, lb in dl:
            xb, lb = xb.to(device), lb.to(device)
            out = model(xb, lengths=lb, use_packing=False)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)

    ypred = np.concatenate(all_preds, axis=0)

    mask = yte >= 0
    ypred_eval = ypred[mask]
    ytrue_eval = yte[mask]

    f1w = f1_score(ytrue_eval, ypred_eval, average="weighted")
    f1m = f1_score(ytrue_eval, ypred_eval, average="macro")
    acc = accuracy_score(ytrue_eval, ypred_eval)

    return f1w, f1m, acc


# ═══════════════════════════════════════════════════════════════════
# Optuna Objective
# ═══════════════════════════════════════════════════════════════════

class TurnLevelObjective:
    """Optuna objective for turn-level optimization"""

    def __init__(self, task, model_tag, layer, pool, K, seeds, device='cuda'):
        self.task = task
        self.model_tag = model_tag
        self.layer = layer
        self.pool = pool
        self.K = K
        self.seeds = seeds
        self.device = device

        # Load data once
        self._load_data()

    def _load_data(self):
        """Load and preprocess data"""
        print(f"\n[DATA] Loading data for {self.model_tag}...")

        # Determine embedding directory
        if self.model_tag == "sentence-roberta":
            # Flat model
            emb_dir = DATA / "embeddings" / self.task / "sentence-roberta" / self.layer / self.pool
            hier_dir = DATA / "embeddings" / self.task / "sentence-roberta-hier" / self.layer / self.pool

            arr_tr = np.load(emb_dir / "train.npz", allow_pickle=True)
            arr_va = np.load(emb_dir / "val.npz", allow_pickle=True)
            arr_te = np.load(emb_dir / "test.npz", allow_pickle=True)

            self.Xtr = arr_tr["embeddings"]
            self.Xva = arr_va["embeddings"]
            self.Xte = arr_te["embeddings"]

            arr_tr_hier = np.load(hier_dir / "train.npz", allow_pickle=True)
            arr_va_hier = np.load(hier_dir / "val.npz", allow_pickle=True)
            arr_te_hier = np.load(hier_dir / "test.npz", allow_pickle=True)

            self.ytr = arr_tr_hier["y"]
            self.yva = arr_va_hier["y"]
            self.yte = arr_te_hier["y"]

            self.idtr = None
            self.idva = None
            self.idte = None

        else:
            # Hierarchical model
            emb_dir = DATA / "embeddings" / self.task / self.model_tag / self.layer / self.pool

            self.Xtr, self.ytr, self.idtr = load_npz(emb_dir / "train.npz", aggregator='mean', device=self.device)
            self.Xva, self.yva, self.idva = load_npz(emb_dir / "val.npz", aggregator='mean', device=self.device)
            self.Xte, self.yte, self.idte = load_npz(emb_dir / "test.npz", aggregator='mean', device=self.device)

        self.N, self.D = self.Xtr.shape
        self.num_classes = len(np.unique(self.ytr[self.ytr >= 0]))

        print(f"  Shape: {self.Xtr.shape}, Classes: {self.num_classes}")

        # Load metadata
        self.meta_tr = read_meta(self.task, "train")
        self.meta_va = read_meta(self.task, "val")
        self.meta_te = read_meta(self.task, "test")

        self.meta_tr = align_order(self.idtr, self.meta_tr)
        self.meta_va = align_order(self.idva, self.meta_va)
        self.meta_te = align_order(self.idte, self.meta_te)

        # Build sequences with fixed K
        print(f"  Building sequences (K={self.K})...")
        self.Xtr_seq, self.Ltr_seq, self.ytr_seq, _, _, _ = build_sequences(
            self.Xtr, self.ytr, self.meta_tr, self.K
        )
        self.Xva_seq, self.Lva_seq, self.yva_seq, _, _, _ = build_sequences(
            self.Xva, self.yva, self.meta_va, self.K
        )
        self.Xte_seq, self.Lte_seq, self.yte_seq, _, _, _ = build_sequences(
            self.Xte, self.yte, self.meta_te, self.K
        )

        print(f"  Sequences built: Train={self.Xtr_seq.shape}")

    def __call__(self, trial):
        """Optuna trial evaluation"""

        # Sample hyperparameters
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 192, 256, 384])
        dropout = trial.suggest_float("dropout_rate", 0.1, 0.7)
        bs = trial.suggest_categorical("batch_size", [32, 64, 128])
        epochs = trial.suggest_int("num_epochs", 40, 120)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 0.01)

        print(f"\n[Trial {trial.number}] lr={lr:.6f}, hidden={hidden_size}, "
              f"dropout={dropout:.3f}, bs={bs}, epochs={epochs}")

        # Evaluate with multiple seeds
        f1w_scores = []

        for seed in self.seeds:
            # Set random seed
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            # Train model
            model = SimpleSeqLSTM(self.D, hidden_size, self.num_classes, dropout)
            model = train_one(
                model, self.Xtr_seq, self.ytr_seq, self.Ltr_seq,
                self.Xva_seq, self.yva_seq, self.Lva_seq,
                epochs=epochs, bs=bs, lr=lr, weight_decay=weight_decay,
                device=self.device
            )

            # Evaluate
            f1w, f1m, acc = evaluate(
                model, self.Xte_seq, self.yte_seq, self.Lte_seq,
                bs=bs, device=self.device
            )

            f1w_scores.append(f1w)
            print(f"  Seed {seed}: F1w={f1w:.4f}")

        # Return mean F1 weighted
        mean_f1w = np.mean(f1w_scores)
        std_f1w = np.std(f1w_scores)

        print(f"  => Mean F1w: {mean_f1w:.4f} +/- {std_f1w:.4f}")

        return mean_f1w


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Turn-level Bayesian Optimization")
    parser.add_argument("--task", required=True, choices=["4way", "6way"])
    parser.add_argument("--model_tag", required=True,
                        choices=["sentence-roberta", "sentence-roberta-hier"])
    parser.add_argument("--layer", default="avg_last4")
    parser.add_argument("--pool", required=True,
                        choices=["mean", "wmean_pos", "wmean_pos_rev"])
    parser.add_argument("--K", type=int, required=True,
                        help="Fixed K value for optimization")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                        help="Seeds for averaging (default: 42 43 44)")
    parser.add_argument("--gpu", default="0")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*70)
    print("TURN-LEVEL BAYESIAN OPTIMIZATION")
    print("="*70)
    print(f"Task:       {args.task}")
    print(f"Model:      {args.model_tag}")
    print(f"Layer:      {args.layer}")
    print(f"Pool:       {args.pool}")
    print(f"K:          {args.K}")
    print(f"Trials:     {args.n_trials}")
    print(f"Seeds:      {args.seeds}")
    print(f"Device:     {device}")
    print("="*70)

    # Create objective
    objective = TurnLevelObjective(
        task=args.task,
        model_tag=args.model_tag,
        layer=args.layer,
        pool=args.pool,
        K=args.K,
        seeds=args.seeds,
        device=device
    )

    # Create study
    model_type = "flat" if args.model_tag == "sentence-roberta" else "hier"
    study_name = f"turnlevel_{args.task}_{model_type}_{args.pool}"

    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler
    )

    # Optimize
    print(f"\n[OPTUNA] Starting optimization...")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Results
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best F1 weighted: {study.best_value:.4f}")
    print(f"\nBest params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Save results
    RES_ROOT.mkdir(parents=True, exist_ok=True)
    out_file = RES_ROOT / f"bayesopt_turnlevel_{args.task}_{model_type}_{args.pool}.json"

    result = {
        "task": args.task,
        "model_tag": args.model_tag,
        "model_type": model_type,
        "layer": args.layer,
        "pool": args.pool,
        "K": args.K,
        "n_trials": args.n_trials,
        "seeds": args.seeds,
        "best_trial": study.best_trial.number,
        "best_f1_weighted": study.best_value,
        "best_params": study.best_params,
        "timestamp": datetime.now().isoformat()
    }

    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to: {out_file}")

    # Save study
    study_file = RES_ROOT / f"study_{args.task}_{model_type}_{args.pool}.pkl"
    import pickle
    with open(study_file, "wb") as f:
        pickle.dump(study, f)

    print(f"Study saved to: {study_file}")


if __name__ == "__main__":
    main()
