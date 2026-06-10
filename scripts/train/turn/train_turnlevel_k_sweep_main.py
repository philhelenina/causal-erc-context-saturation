#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_turnlevel_k_sweep_bayesian.py (v2 - with masking)

K-sweep with Bayesian-optimized hyperparameters
- Automatically loads best params from bayesian_optimization results
- Handles hierarchical embeddings (3D → 2D flattening)
- **NEW: Pack_padded_sequence to ignore zero padding**
- Saves predictions, metadata, and K_norm for analysis
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

# ═══════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════

HOME = Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()
DATA = HOME / "data"
RES_ROOT = HOME / "results" / "turnlevel_k_sweep_bayesian"

# ═══════════════════════════════════════════════════════════════════
# Sentence-level Aggregator (for hierarchical embeddings)
# ═══════════════════════════════════════════════════════════════════

class SentenceLSTMAggregator(nn.Module):
    """LSTM to aggregate sentence-level embeddings into utterance embedding"""
    
    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden_dim = hidden_dim
    
    def forward(self, sentences):
        """
        Args:
            sentences: (S, D) variable-length sentence embeddings
        Returns:
            (hidden_dim,) utterance embedding
        """
        # Add batch dimension
        x = sentences.unsqueeze(0)  # (1, S, D)
        
        # LSTM forward
        out, (h, c) = self.lstm(x)
        
        # Return last hidden state
        return h[-1, 0, :]  # (hidden_dim,)


# ═══════════════════════════════════════════════════════════════════
# Model with Masking Support (Turn-level)
# ═══════════════════════════════════════════════════════════════════

class SimpleSeqLSTM(nn.Module):
    """LSTM for turn-level sequential modeling with masking support"""
    
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths=None, use_packing=False):
        """
        Args:
            x: (B, K+1, D) or (B, D)
            lengths: (B,) actual sequence lengths (optional)
            use_packing: if True, use pack_padded_sequence (default: False)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, D) → (B, 1, D)
            lengths = None
        
        batch_size = x.size(0)
        
        if use_packing and lengths is not None and x.size(1) > 1:
            # Use pack_padded_sequence to ignore padding
            lengths_cpu = lengths.cpu()
            
            # Sort by lengths (required for pack_padded_sequence)
            sorted_lengths, perm_idx = lengths_cpu.sort(descending=True)
            sorted_x = x[perm_idx]
            
            # Pack (excludes padding from computation)
            packed = pack_padded_sequence(
                sorted_x, sorted_lengths, batch_first=True, enforce_sorted=True
            )
            
            # LSTM
            packed_out, _ = self.lstm(packed)
            
            # Unpack
            out, _ = pad_packed_sequence(packed_out, batch_first=True)
            
            # Get last valid output for each sequence
            h = torch.zeros(batch_size, out.size(2), device=x.device)
            for i in range(batch_size):
                h[i] = out[i, sorted_lengths[i]-1, :]
            
            # Restore original order
            _, unperm_idx = perm_idx.sort()
            h = h[unperm_idx]
        else:
            # NO PACKING: Include padding in LSTM computation
            # This was the old approach and may perform better!
            out, _ = self.lstm(x)  # (B, K+1, H)
            h = out[:, -1, :]  # Take last timestep: (B, H)
        
        h = self.dropout(h)
        return self.fc(h)


# ═══════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════

def load_npz(p: Path, aggregator='mean', device='cuda'):
    """
    Load NPZ with automatic hierarchical flattening
    
    Args:
        aggregator: 'mean' or 'lstm' for sentence aggregation
    """
    arr = np.load(p, allow_pickle=True)
    
    # Get embeddings - try multiple keys
    X = None
    for key in ['embeddings', 'X', 'features', 'data']:
        if key in arr:
            X = arr[key]
            break
    
    if X is None:
        raise KeyError(f"No embeddings found in {p}. Available keys: {list(arr.keys())}")
    
    # Get labels - try multiple keys
    y = None
    for key in ['y', 'labels', 'targets', 'label', 'emotion']:
        if key in arr:
            y = arr[key]
            break
    
    if y is None:
        raise KeyError(f"No labels found in {p}. Available keys: {list(arr.keys())}")
    
    # Get IDs - optional
    ids = None
    for key in ['ids', 'utterance_ids', 'utt_ids', 'sample_ids', 'id']:
        if key in arr:
            ids = arr[key]
            break
    
    # ═══ HIERARCHICAL HANDLING ═══
    if X.ndim == 3:
        print(f"  [INFO] Hierarchical embeddings detected: {X.shape}")
        print(f"  [INFO] Using aggregator: {aggregator}")
        
        if aggregator == 'mean':
            # Masked mean if lengths available
            if "lengths" in arr:
                lengths = arr["lengths"]
                X_flat = []
                for i in range(len(X)):
                    L = int(lengths[i])
                    X_flat.append(X[i, :L, :].mean(axis=0))
                X = np.stack(X_flat, axis=0)
                print(f"  [INFO] Using masked mean (lengths-aware)")
            else:
                # Simple mean over sentence dimension
                X = X.mean(axis=1)  # (N, S, D) → (N, D)
                print(f"  [WARN] No lengths found, using simple mean")
        
        elif aggregator == 'lstm':
            # LSTM aggregation
            if "lengths" not in arr:
                raise ValueError("LSTM aggregator requires 'lengths' in NPZ file")
            
            lengths = arr["lengths"]
            sent_dim = X.shape[2]
            lstm_agg = SentenceLSTMAggregator(input_dim=sent_dim, hidden_dim=sent_dim)
            lstm_agg.to(device)
            lstm_agg.eval()
            
            X_flat = []
            with torch.no_grad():
                for i in range(len(X)):
                    L = int(lengths[i])
                    sent_embeds = torch.tensor(X[i, :L, :], dtype=torch.float32, device=device)
                    utt_embed = lstm_agg(sent_embeds)
                    X_flat.append(utt_embed.cpu().numpy())
            
            X = np.stack(X_flat, axis=0)
            print(f"  [INFO] LSTM aggregation complete")
        
        else:
            raise ValueError(f"Unknown aggregator: {aggregator}")
        
        print(f"  [INFO] After aggregation: {X.shape}")
    
    return X, y, ids


def read_meta(task: str, split: str) -> pd.DataFrame:
    """Load metadata from CSV"""
    csv_dir = DATA / f"iemocap_{task}_data"
    
    # Try multiple possible filenames
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
            print(f"  [META] Found CSV: {name}")
            break
    
    if csv_path is None:
        raise FileNotFoundError(f"No CSV found in {csv_dir}. Tried: {possible_names}")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Find ID column
    id_cols = ["id", "utt_id", "sample_id", "orig_id", "file_id", "wav_id"]
    id_col = next((c for c in id_cols if c in df.columns), None)
    if not id_col:
        raise ValueError(f"No ID column in {csv_path}")
    
    # Find dialogue column (prefer file_id)
    dlg_cols = ["file_id", "dialogue_id", "dialog_id", "conv_id", "file_root", 
                "session_dialog_id", "dlg_id", "file_id_root"]
    dlg_col = next((c for c in dlg_cols if c in df.columns), None)
    
    if not dlg_col:
        # Fallback: extract from id column
        dlg_col = id_col
        print(f"  [WARN] No dialogue column found, will extract from {id_col}")
    
    # Find turn column (prefer utterance_num)
    turn_cols = ["utterance_num", "turn_index", "turn_id", "turn", "utt_no", "order", "idx"]
    turn_col = next((c for c in turn_cols if c in df.columns), None)
    
    if not turn_col:
        df["turn_index"] = df.groupby(dlg_col).cumcount()
        turn_col = "turn_index"
    
    # Extract relevant columns
    if not turn_col:
        print(f"  [WARN] No turn column found, generating sequential indices")
        df["turn_index_generated"] = df.groupby(dlg_col).cumcount()
        turn_col = "turn_index_generated"
    
    meta = df[[id_col, dlg_col, turn_col]].copy()
    meta.columns = ["id", "dialogue", "turn_idx"]
    meta["id"] = meta["id"].astype(str)
    meta["dialogue"] = meta["dialogue"].astype(str)
    meta["turn_idx"] = meta["turn_idx"].astype(int)
    
    # Sort by dialogue and turn
    meta = meta.sort_values(["dialogue", "turn_idx"]).reset_index(drop=True)
    
    print(f"  [META] Loaded {len(meta)} utterances, {meta['dialogue'].nunique()} dialogues")
    
    return meta


def align_order(ids_npz, meta_df):
    """Align NPZ indices with metadata"""
    if ids_npz is None:
        meta_df["npz_idx"] = np.arange(len(meta_df))
        return meta_df
    
    ids_npz = pd.Series(ids_npz.astype(str))
    indexer = pd.Series(np.arange(len(ids_npz)), index=ids_npz)
    
    # Handle duplicate IDs in NPZ (keep first occurrence)
    if indexer.index.duplicated().any():
        n_dups = indexer.index.duplicated().sum()
        print(f"  [WARNING] Found {n_dups} duplicate IDs in NPZ, keeping first occurrence")
        indexer = indexer[~indexer.index.duplicated(keep='first')]
    
    if not meta_df["id"].isin(indexer.index).all():
        missing = meta_df.loc[~meta_df["id"].isin(indexer.index), "id"].head(3).tolist()
        raise ValueError(f"Missing IDs in NPZ: {missing}")
    
    meta_df["npz_idx"] = meta_df["id"].map(indexer)
    meta_df = meta_df.sort_values(["dialogue", "turn_idx"]).reset_index(drop=True)
    
    return meta_df


def build_sequences(X, y, order_df, K, pad_value=0.0):
    """
    Build turn-level sequences with context window K
    
    Returns:
        Xseq: (N, K+1, D) sequences
        seq_lengths: (N,) actual sequence lengths (no padding)
        yout: (N,) labels
        dlg_len: (N,) dialogue lengths
        dlg_id: (N,) dialogue IDs
        t_idx: (N,) turn indices
    """
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
        
        # Get K previous + current
        start = max(0, pos - K)
        seq_idx = idxs[start : pos + 1]
        actual_len = len(seq_idx)
        
        # Pad if needed
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
    """
    Train with early stopping and masking support
    
    Args:
        Ltr, Lva: Sequence lengths for masking
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    
    # Convert to tensors
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
        # Train
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
        
        # Validation
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
        
        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model


def predict_probs(model, Xte, Lte, bs=64, device="cuda"):
    """Get softmax probabilities with masking support"""
    Xte = torch.tensor(Xte).float()
    Lte = torch.tensor(Lte).long()
    
    ds = torch.utils.data.TensorDataset(Xte, Lte)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False)
    
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for xb, lb in dl:
            xb, lb = xb.to(device), lb.to(device)
            out = model(xb, lengths=lb, use_packing=False)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            all_probs.append(probs)
    
    return np.concatenate(all_probs, axis=0)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["4way", "6way"])
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--model_tag", required=True)
    ap.add_argument("--layer", required=True)
    ap.add_argument("--pool", required=True)
    
    # Sentence aggregator
    ap.add_argument("--aggregator", default="mean", choices=["mean", "lstm"],
                    help="Sentence-level aggregation method for hierarchical embeddings")
    
    # Use flat embeddings (pre-computed mean)
    ap.add_argument("--use_flat", action="store_true",
                    help="Use flat embeddings from embeddings_flat/ (pre-computed mean)")
    
    # Random seed
    ap.add_argument("--seed", type=int, default=42)
    
    # K sweep range
    ap.add_argument("--k_min", type=int, default=0)
    ap.add_argument("--k_max", type=int, default=100)
    ap.add_argument("--k_step", type=int, default=10)
    
    # Hyperparameters (optional - will use Bayesian if not specified)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--hidden_size", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--weight_decay", type=float, default=None)
    
    # -1 label handling
    ap.add_argument("--treat_minus1_as_class", action="store_true",
                    help="Treat -1 labels as a separate class instead of masking")
    
    args = ap.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"\n[SEED] Random seed set to: {args.seed}")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ═══ LOAD BAYESIAN HYPERPARAMETERS ═══
    bayesian_path = HOME / "results" / "bayesian_optimization" / \
                    f"bayesopt_{args.task}_{args.model_tag}_{args.layer}_{args.pool}_lstm.json"
    
    if bayesian_path.exists() and args.lr is None:
        print(f"\n[INFO] Loading Bayesian-optimized hyperparameters")
        print(f"       from: {bayesian_path}")
        
        with open(bayesian_path) as f:
            data = json.load(f)
        
        params = data.get("best_params", {})
        lr = params.get("learning_rate", 1e-3)
        hidden_size = params.get("hidden_size", 256)
        dropout = params.get("dropout_rate", 0.3)
        bs = params.get("batch_size", 64)
        epochs = params.get("num_epochs", 60)
        weight_decay = params.get("weight_decay", 0.0)
        
        print(f"\n[BAYESIAN PARAMS]")
        print(f"  learning_rate:  {lr:.6f}")
        print(f"  hidden_size:    {hidden_size}")
        print(f"  dropout_rate:   {dropout:.3f}")
        print(f"  batch_size:     {bs}")
        print(f"  num_epochs:     {epochs}")
        print(f"  weight_decay:   {weight_decay:.6f}")
    else:
        # Use manual hyperparameters
        lr = args.lr or 1e-3
        hidden_size = args.hidden_size or 256
        dropout = args.dropout or 0.3
        bs = args.bs or 64
        epochs = args.epochs or 60
        weight_decay = args.weight_decay or 0.0
        
        print(f"\n[INFO] Using manual hyperparameters")
    
    # ═══ LOAD DATA ═══
    if args.use_flat:
        emb_dir = DATA / "embeddings_flat" / args.task / args.model_tag / args.layer / args.pool
        print(f"\n[LOAD] FLAT embeddings: {emb_dir}")
    else:
        emb_dir = DATA / "embeddings" / args.task / args.model_tag / args.layer / args.pool
        print(f"\n[LOAD] Hierarchical embeddings: {emb_dir}")
        print(f"[AGGREGATOR] {args.aggregator}")
    
    # ═══ SPECIAL CASE: sentence-roberta (flat) has no labels ═══
    if args.model_tag == "sentence-roberta":
        print(f"\n[INFO] sentence-roberta has no labels in NPZ")
        print(f"[INFO] Loading embeddings from sentence-roberta, labels from sentence-roberta-hier")
        
        # Load embeddings from flat
        flat_dir = DATA / "embeddings" / args.task / "sentence-roberta" / args.layer / args.pool
        # Load labels from hierarchical
        hier_dir = DATA / "embeddings" / args.task / "sentence-roberta-hier" / args.layer / args.pool
        
        # Embeddings from flat
        arr_tr = np.load(flat_dir / "train.npz", allow_pickle=True)
        arr_va = np.load(flat_dir / "val.npz", allow_pickle=True)
        arr_te = np.load(flat_dir / "test.npz", allow_pickle=True)
        
        Xtr = arr_tr["embeddings"]
        Xva = arr_va["embeddings"]
        Xte = arr_te["embeddings"]
        
        print(f"  [INFO] Loaded embeddings from {args.model_tag}: {Xtr.shape}")
        
        # Labels from hierarchical
        arr_tr_hier = np.load(hier_dir / "train.npz", allow_pickle=True)
        arr_va_hier = np.load(hier_dir / "val.npz", allow_pickle=True)
        arr_te_hier = np.load(hier_dir / "test.npz", allow_pickle=True)
        
        ytr = arr_tr_hier["y"]
        yva = arr_va_hier["y"]
        yte = arr_te_hier["y"]
        
        # Use None for IDs - will trigger sequential indexing in align_order
        idtr = None
        idva = None
        idte = None
        
        print(f"  [INFO] Loaded labels from sentence-roberta-hier: {ytr.shape}")
        print(f"  [INFO] Using sequential indexing (no IDs)")
        
    else:
        # Normal loading
        # Load with appropriate aggregator
        if args.use_flat:
            # Flat embeddings already aggregated, ignore aggregator parameter
            Xtr, ytr, idtr = load_npz(emb_dir / "train.npz", aggregator='mean', device=device)
            Xva, yva, idva = load_npz(emb_dir / "val.npz", aggregator='mean', device=device)
            Xte, yte, idte = load_npz(emb_dir / "test.npz", aggregator='mean', device=device)
        else:
            # Hierarchical embeddings, use specified aggregator
            Xtr, ytr, idtr = load_npz(emb_dir / "train.npz", aggregator=args.aggregator, device=device)
            Xva, yva, idva = load_npz(emb_dir / "val.npz", aggregator=args.aggregator, device=device)
            Xte, yte, idte = load_npz(emb_dir / "test.npz", aggregator=args.aggregator, device=device)
    
    N, D = Xtr.shape
    num_classes = len(np.unique(ytr[ytr >= 0]))
    
    print(f"\n[DATA]")
    print(f"  Input dim:      {D}")
    
    # Handle -1 labels
    if args.treat_minus1_as_class:
        print(f"  [INFO] Treating -1 labels as separate class")
        n_minus1 = (ytr == -1).sum()
        print(f"  [INFO] Converting {n_minus1} (-1) labels to class {num_classes}")
        
        # Convert -1 to new class
        ytr[ytr == -1] = num_classes
        yva[yva == -1] = num_classes
        yte[yte == -1] = num_classes
        
        num_classes += 1  # Add one more class
        print(f"  [INFO] Updated num_classes: {num_classes}")
    
    print(f"  Num classes:    {num_classes}")
    print(f"  Train size:     {len(Xtr)}")
    
    # ═══ LOAD METADATA ═══
    print(f"\n[META] Loading turn-level metadata...")
    
    meta_tr = read_meta(args.task, "train")
    meta_va = read_meta(args.task, "val")
    meta_te = read_meta(args.task, "test")
    
    meta_tr = align_order(idtr, meta_tr)
    meta_va = align_order(idva, meta_va)
    meta_te = align_order(idte, meta_te)
    
    total_turns_tr = meta_tr.groupby("dialogue")["turn_idx"].max().sum() + len(meta_tr["dialogue"].unique())
    
    print(f"  Total turns (train): {int(total_turns_tr)}")
    
    # ═══ K SWEEP ═══
    Ks = list(range(args.k_min, args.k_max + 1, args.k_step))
    
    print(f"\n[K SWEEP] K values: {Ks}")
    print(f"          Total experiments: {len(Ks)}")
    
    results = []
    all_preds = []  # (num_K, N_test, num_classes)
    
    for i, K in enumerate(Ks):
        print(f"\n{'='*80}")
        print(f"[K={K:3d}] ({i+1}/{len(Ks)}) K_norm = {K/total_turns_tr:.4f}")
        print(f"{'='*80}")
        
        # Build sequences
        Xtr_seq, Ltr_seq, ytr_seq, dlg_len_tr, dlg_id_tr, tidx_tr = build_sequences(
            Xtr, ytr, meta_tr, K
        )
        Xva_seq, Lva_seq, yva_seq, dlg_len_va, dlg_id_va, tidx_va = build_sequences(
            Xva, yva, meta_va, K
        )
        Xte_seq, Lte_seq, yte_seq, dlg_len_te, dlg_id_te, tidx_te = build_sequences(
            Xte, yte, meta_te, K
        )
        
        print(f"  Building sequences (K={K})...")
        print(f"  Shapes: Train={Xtr_seq.shape}, Val={Xva_seq.shape}, Test={Xte_seq.shape}")
        
        # Train
        print(f"  Training...", end='', flush=True)
        model = SimpleSeqLSTM(D, hidden_size, num_classes, dropout)
        model = train_one(
            model, Xtr_seq, ytr_seq, Ltr_seq, Xva_seq, yva_seq, Lva_seq,
            epochs=epochs, bs=bs, lr=lr, weight_decay=weight_decay, device=device
        )
        print(" Done.")
        
        # Predict
        probs = predict_probs(model, Xte_seq, Lte_seq, bs=bs, device=device)
        ypred = probs.argmax(axis=1)
        
        all_preds.append(probs)
        
        # Evaluate (exclude -1 class if it was converted)
        if args.treat_minus1_as_class:
            # Evaluate on original labeled classes only
            mask = (yte_seq >= 0) & (yte_seq < num_classes - 1)
            # For predictions, keep original prediction even if it's the -1 class
            ypred_eval = ypred[mask]
            ytrue_eval = yte_seq[mask]
        else:
            # Standard evaluation (skip -1)
            mask = yte_seq >= 0
            ypred_eval = ypred[mask]
            ytrue_eval = yte_seq[mask]
        
        f1w = f1_score(ytrue_eval, ypred_eval, average="weighted")
        f1m = f1_score(ytrue_eval, ypred_eval, average="macro")
        acc = accuracy_score(ytrue_eval, ypred_eval)
        
        results.append({
            "K": K,
            "K_norm": K / total_turns_tr,
            "f1_weighted": f1w,
            "f1_macro": f1m,
            "accuracy": acc
        })
        
        print(f"  [RESULT] F1w={f1w:.4f}, F1m={f1m:.4f}, Acc={acc:.4f}")
        
        # Store metadata for first K only
        if i == 0:
            dialog_len_save = dlg_len_te
            dialogue_id_save = dlg_id_te
            turn_idx_save = tidx_te
            labels_save = yte_seq
    
    # ═══ SAVE RESULTS ═══
    emb_type = "flat" if args.use_flat else args.aggregator
    out_dir = RES_ROOT / f"{args.task}_{args.model_tag}_{args.layer}_{args.pool}_{emb_type}" / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Results CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(out_dir / "k_sweep_results.csv", index=False)
    
    # 2. Predictions per K
    preds_perK = np.stack(all_preds, axis=0)  # (num_K, N, C)
    np.save(out_dir / "preds_perK.npy", preds_perK)
    
    # 3. Labels
    np.save(out_dir / "labels.npy", labels_save)
    
    # 4. K values
    np.save(out_dir / "Ks.npy", np.array(Ks))
    
    # 5. K_norm values
    K_norm = np.array([K / total_turns_tr for K in Ks])
    np.save(out_dir / "K_norm.npy", K_norm)
    
    # 6. Dialogue metadata
    np.save(out_dir / "dialog_len.npy", dialog_len_save)
    np.save(out_dir / "dialogue_id.npy", dialogue_id_save)
    np.save(out_dir / "turn_idx.npy", turn_idx_save)
    
    # 7. Metadata JSON
    metadata = {
        "task": args.task,
        "model_tag": args.model_tag,
        "layer": args.layer,
        "pool": args.pool,
        "use_flat": args.use_flat,
        "aggregator": "precomputed_mean" if args.use_flat else args.aggregator,
        "seed": args.seed,
        "hyperparameters": {
            "learning_rate": float(lr),
            "hidden_size": int(hidden_size),
            "dropout_rate": float(dropout),
            "batch_size": int(bs),
            "num_epochs": int(epochs),
            "weight_decay": float(weight_decay)
        },
        "k_values": Ks,
        "total_turns": int(total_turns_tr),
        "num_classes": int(num_classes),
        "input_dim": int(D)
    }
    
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*80}")
    print("FILES SAVED")
    print(f"{'='*80}")
    print(f"  ✓ k_sweep_results.csv       - Performance metrics per K")
    print(f"  ✓ preds_perK.npy            - Shape: {preds_perK.shape}")
    print(f"  ✓ labels.npy                - Ground truth labels")
    print(f"  ✓ Ks.npy                    - K values")
    print(f"  ✓ K_norm.npy                - Normalized K values")
    print(f"  ✓ dialog_len.npy            - Dialogue lengths")
    print(f"  ✓ dialogue_id.npy           - Dialogue IDs")
    print(f"  ✓ turn_idx.npy              - Turn indices")
    print(f"  ✓ metadata.json             - Experiment configuration")
    
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(df_results.to_string(index=False))
    print(f"{'='*80}")


if __name__ == "__main__":
    main()