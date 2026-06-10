#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_turn_classifier.py
Bridge module for Bayesian Optimization and turn-level experiments.
Standalone version with inline model definitions.
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

HOME = Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()
DATA = HOME / "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS_4 = {"ang": 0, "hap": 1, "sad": 2, "neu": 3}
LABELS_6 = {"ang": 0, "hap": 1, "sad": 2, "neu": 3, "exc": 4, "fru": 5}

# ═══════════════════════════════════════════════════════════════════
# Model Definitions
# ═══════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """Simple MLP classifier for utterance-level classification"""
    
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        # x: (B, D)
        return self.net(x)


class SimpleLSTM(nn.Module):
    """LSTM classifier for sequential/turn-level data"""
    
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: (B, D) or (B, 1, D) or (B, S, D)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, D) → (B, 1, D)
        
        out, _ = self.lstm(x)  # (B, S, H)
        h = self.dropout(out[:, -1, :])  # Take last timestep: (B, H)
        return self.fc(h)  # (B, num_classes)


# ═══════════════════════════════════════════════════════════════════
# Path Resolution
# ═══════════════════════════════════════════════════════════════════

def resolve_embedding_path(task, embedding, layer, pool):
    """Resolve embedding path based on naming conventions"""
    base = DATA / "embeddings" / task
    
    # Common mappings
    mapping = {
        "sentence-roberta": base / "sentence-roberta" / layer / pool,
        "sentence-roberta-hier": base / "sentence-roberta-hier" / layer / pool,
        "senticnet-sroberta-fused": base / "fused" / f"senticnet-sroberta-fused-{task}" / layer / pool,
        "w2v-sentence-roberta-fused": base / "w2v-sentence-roberta-fused" / layer / pool,
    }
    
    return mapping.get(embedding, base / embedding / layer / pool)


# ═══════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════

def load_embeddings(task, embedding, layer, pool, split):
    """Load embeddings from NPZ file"""
    p = resolve_embedding_path(task, embedding, layer, pool) / f"{split}.npz"
    
    if not p.exists():
        raise FileNotFoundError(f"[ERROR] Missing embedding file: {p}")
    
    arr = np.load(p, allow_pickle=True)
    
    # Try different key names
    if "embeddings" in arr:
        X = arr["embeddings"]
    elif "X" in arr:
        X = arr["X"]
    else:
        # Take first available key
        X = arr[list(arr.keys())[0]]
    
    X = X.astype("float32")
    
    # ═══ HIERARCHICAL HANDLING ═══
    # If 3D (N, S, D), aggregate to 2D (N, D)
    if X.ndim == 3:
        print(f"  [INFO] Hierarchical embeddings detected: {X.shape}")
        print(f"  [INFO] Applying sentence-level mean pooling...")
        
        # Check if lengths available for proper masking
        if "lengths" in arr:
            lengths = arr["lengths"]
            # Masked mean pooling
            X_agg = []
            for i in range(len(X)):
                L = int(lengths[i])
                X_agg.append(X[i, :L, :].mean(axis=0))
            X = np.stack(X_agg, axis=0)
        else:
            # Simple mean over sentence dimension
            X = X.mean(axis=1)  # (N, S, D) → (N, D)
        
        print(f"  [INFO] After pooling: {X.shape}")
    
    return X


def load_labels(task, split):
    """Load labels from CSV file"""
    csv = DATA / f"iemocap_{task}_data" / f"{split}_{task}_unified.csv"
    
    if not csv.exists():
        raise FileNotFoundError(f"[ERROR] Missing CSV file: {csv}")
    
    df = pd.read_csv(csv)
    label_map = LABELS_6 if task == "6way" else LABELS_4
    
    # Robust label column detection
    col_candidates = ["label_num", "label", "label_num4", "label_num6"]
    col = next((c for c in col_candidates if c in df.columns), None)
    
    if col is None:
        raise ValueError(f"No label column found in {csv}. Available columns: {df.columns.tolist()}")
    
    ser = df[col]
    
    # Handle numeric vs string labels
    if ser.dtype != object:
        # Already numeric
        y = ser.fillna(-1).astype("int64")
    else:
        # String labels - need mapping
        y = ser.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        y = y.str.lower().map(label_map).fillna(-1).astype("int64")
    
    # Create mask for valid labels
    mask = y >= 0
    
    return y.to_numpy(), mask.to_numpy()


def align_xy(X, y, mask):
    """Align embeddings with labels and filter out invalid samples"""
    n = min(len(X), len(mask))
    X, y, mask = X[:n], y[:n], mask[:n]
    return X[mask], y[mask]


def load_data(task, embedding, layer, pool):
    """
    Load train/val/test data for a given configuration
    
    Returns:
        Xtr, ytr, Xva, yva, Xte, yte (all filtered for valid labels)
    """
    # Train
    Xtr = load_embeddings(task, embedding, layer, pool, "train")
    ytr, mtr = load_labels(task, "train")
    Xtr, ytr = align_xy(Xtr, ytr, mtr)
    
    # Validation
    Xva = load_embeddings(task, embedding, layer, pool, "val")
    yva, mva = load_labels(task, "val")
    Xva, yva = align_xy(Xva, yva, mva)
    
    # Test
    Xte = load_embeddings(task, embedding, layer, pool, "test")
    yte, mte = load_labels(task, "test")
    Xte, yte = align_xy(Xte, yte, mte)
    
    print(f"[LOAD] {task}/{embedding}/{layer}/{pool}")
    print(f"  Train: {Xtr.shape}  Val: {Xva.shape}  Test: {Xte.shape}")
    
    return Xtr, ytr, Xva, yva, Xte, yte


# ═══════════════════════════════════════════════════════════════════
# Training & Evaluation
# ═══════════════════════════════════════════════════════════════════

def train_and_evaluate_once(task, embedding, layer, pool, model_type, params):
    """
    Single training run with given hyperparameters
    
    Args:
        task: "4way" or "6way"
        embedding: encoder name (e.g., "sentence-roberta-hier")
        layer: layer config (e.g., "avg_last4")
        pool: pooling method (e.g., "mean")
        model_type: "mlp" or "lstm"
        params: dict with hyperparameters:
            - learning_rate
            - hidden_size
            - dropout_rate
            - batch_size
            - num_epochs
            - weight_decay
            - early_stopping_patience
    
    Returns:
        metrics: dict with f1_weighted, f1_macro, acc
    """
    
    # Load data
    Xtr, ytr, Xva, yva, Xte, yte = load_data(task, embedding, layer, pool)
    
    # Extract hyperparameters
    lr = float(params.get("learning_rate", 1e-3))
    hidden = int(params.get("hidden_size", 256))
    drop = float(params.get("dropout_rate", 0.3))
    bs = int(params.get("batch_size", 64))
    epochs = int(params.get("num_epochs", 100))
    wd = float(params.get("weight_decay", 0.0))
    patience = int(params.get("early_stopping_patience", 20))
    
    # Model setup
    input_dim = Xtr.shape[1]
    num_classes = len(np.unique(ytr))
    
    if model_type == "lstm":
        model = SimpleLSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_classes=num_classes,
            dropout_rate=drop
        ).to(DEVICE)
    elif model_type == "mlp":
        model = MLP(
            input_size=input_dim,
            hidden_size=hidden,
            num_classes=num_classes,
            dropout_rate=drop
        ).to(DEVICE)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    
    # Data loaders
    def make_loader(X, y, shuffle):
        dataset = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).long()
        )
        return DataLoader(dataset, batch_size=bs, shuffle=shuffle)
    
    train_loader = make_loader(Xtr, ytr, shuffle=True)
    val_loader = make_loader(Xva, yva, shuffle=False)
    test_loader = make_loader(Xte, yte, shuffle=False)
    
    # Training loop with early stopping
    best_state = None
    best_val_loss = 9e9
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_losses = []
        
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validate
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Test evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(yb.numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Compute metrics
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    
    print(f"[RESULT] {task}/{embedding}/{layer}/{pool}/{model_type}")
    print(f"  F1w={f1_weighted:.4f} F1m={f1_macro:.4f} Acc={acc:.4f}")
    
    return {
        "f1_weighted": float(f1_weighted),
        "f1_macro": float(f1_macro),
        "acc": float(acc)
    }


# ═══════════════════════════════════════════════════════════════════
# Manual Test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Quick test
    params = {
        "learning_rate": 1e-3,
        "hidden_size": 128,
        "dropout_rate": 0.3,
        "batch_size": 64,
        "num_epochs": 5,  # Short for testing
        "weight_decay": 0.0,
        "early_stopping_patience": 3
    }
    
    print("="*80)
    print("TEST RUN")
    print("="*80)
    
    result = train_and_evaluate_once(
        task="4way",
        embedding="sentence-roberta-hier",
        layer="avg_last4",
        pool="mean",
        model_type="mlp",
        params=params
    )
    
    print("\nTest result:")
    print(json.dumps(result, indent=2))
    print("\n✅ Test complete!")