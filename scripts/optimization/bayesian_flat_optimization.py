#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bayesian_flat_optimization.py

Bayesian hyperparameter optimization for Flat (sentence-roberta) utterance-level.
Uses 3 seeds average for robust optimization.

Usage:
    python bayesian_flat_optimization.py \
        --task 4way \
        --pool mean \
        --n_trials 30 \
        --seeds 42 43 44 \
        --gpu 0
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
import optuna
from optuna.samplers import TPESampler
import argparse
from datetime import datetime
import pickle

# ═══════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════

HOME = Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()
DATA = HOME / "data"
RESULT_DIR = HOME / "results" / "bayesian_optimization"

LABELS_4 = {"ang": 0, "hap": 1, "sad": 2, "neu": 3}
LABELS_6 = {"ang": 0, "hap": 1, "sad": 2, "neu": 3, "exc": 4, "fru": 5}

# ═══════════════════════════════════════════════════════════════════
# Model
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
        return self.net(x)


class SimpleLSTM(nn.Module):
    """LSTM classifier"""

    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        h = self.dropout(out[:, -1, :])
        return self.fc(h)


# ═══════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════

def load_flat_embeddings(task, layer, pool, split):
    """
    Load Flat (sentence-roberta) embeddings.
    Labels come from sentence-roberta-hier since flat doesn't have labels.
    """
    # Embeddings from sentence-roberta (flat)
    flat_dir = DATA / "embeddings" / task / "sentence-roberta" / layer / pool
    flat_path = flat_dir / f"{split}.npz"

    if not flat_path.exists():
        raise FileNotFoundError(f"Missing flat embeddings: {flat_path}")

    arr = np.load(flat_path, allow_pickle=True)
    X = arr["embeddings"].astype("float32")

    # Labels from sentence-roberta-hier (since flat doesn't store labels)
    hier_dir = DATA / "embeddings" / task / "sentence-roberta-hier" / layer / pool
    hier_path = hier_dir / f"{split}.npz"

    if not hier_path.exists():
        raise FileNotFoundError(f"Missing hier embeddings for labels: {hier_path}")

    arr_hier = np.load(hier_path, allow_pickle=True)
    y = arr_hier["y"]

    return X, y


def load_hier_embeddings(task, layer, pool, split):
    """
    Load Hierarchical (sentence-roberta-hier) embeddings.
    Applies mean pooling over sentences.
    """
    hier_dir = DATA / "embeddings" / task / "sentence-roberta-hier" / layer / pool
    hier_path = hier_dir / f"{split}.npz"

    if not hier_path.exists():
        raise FileNotFoundError(f"Missing hier embeddings: {hier_path}")

    arr = np.load(hier_path, allow_pickle=True)
    X = arr["embeddings"]
    y = arr["y"]

    # Hierarchical: 3D -> 2D via mean pooling
    if X.ndim == 3:
        if "lengths" in arr:
            lengths = arr["lengths"]
            X_agg = []
            for i in range(len(X)):
                L = int(lengths[i])
                X_agg.append(X[i, :L, :].mean(axis=0))
            X = np.stack(X_agg, axis=0)
        else:
            X = X.mean(axis=1)

    X = X.astype("float32")
    return X, y


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════

def train_and_evaluate(model, Xtr, ytr, Xva, yva, Xte, yte, params, device):
    """Train model and return test metrics"""

    model.to(device)

    lr = params["learning_rate"]
    bs = params["batch_size"]
    epochs = params["num_epochs"]
    weight_decay = params.get("weight_decay", 0.0)
    patience = params.get("early_stopping_patience", 20)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    # Convert to tensors
    Xtr_t = torch.tensor(Xtr).float()
    ytr_t = torch.tensor(ytr).long()
    Xva_t = torch.tensor(Xva).float()
    yva_t = torch.tensor(yva).long()
    Xte_t = torch.tensor(Xte).float()
    yte_t = torch.tensor(yte).long()

    tr_ds = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
    va_ds = torch.utils.data.TensorDataset(Xva_t, yva_t)
    te_ds = torch.utils.data.TensorDataset(Xte_t, yte_t)

    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=bs, shuffle=True)
    va_dl = torch.utils.data.DataLoader(va_ds, batch_size=bs)
    te_dl = torch.utils.data.DataLoader(te_ds, batch_size=bs)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)

            # Skip invalid labels
            mask = yb >= 0
            if mask.sum() == 0:
                continue

            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits[mask], yb[mask])
            loss.backward()
            opt.step()

        # Validation
        model.eval()
        val_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(device), yb.to(device)
                mask = yb >= 0
                if mask.sum() == 0:
                    continue
                logits = model(xb)
                val_loss += crit(logits[mask], yb[mask]).item()
                n_batches += 1

        val_loss /= max(n_batches, 1)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test evaluation
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in te_dl:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Filter out invalid labels
    mask = all_labels >= 0
    all_preds = all_preds[mask]
    all_labels = all_labels[mask]

    f1w = f1_score(all_labels, all_preds, average="weighted")
    f1m = f1_score(all_labels, all_preds, average="macro")
    acc = accuracy_score(all_labels, all_preds)

    return {"f1_weighted": f1w, "f1_macro": f1m, "acc": acc}


# ═══════════════════════════════════════════════════════════════════
# Optuna Objective
# ═══════════════════════════════════════════════════════════════════

class FlatObjective:
    """Optuna objective for Flat model optimization"""

    def __init__(self, task, layer, pool, model_type, seeds, device='cuda'):
        self.task = task
        self.layer = layer
        self.pool = pool
        self.model_type = model_type
        self.seeds = seeds
        self.device = device

        # Load data once
        self._load_data()

    def _load_data(self):
        """Load embeddings"""
        print(f"\n[DATA] Loading Flat embeddings...")
        print(f"  Task: {self.task}, Layer: {self.layer}, Pool: {self.pool}")

        self.Xtr, self.ytr = load_flat_embeddings(self.task, self.layer, self.pool, "train")
        self.Xva, self.yva = load_flat_embeddings(self.task, self.layer, self.pool, "val")
        self.Xte, self.yte = load_flat_embeddings(self.task, self.layer, self.pool, "test")

        self.input_dim = self.Xtr.shape[1]
        self.num_classes = len(np.unique(self.ytr[self.ytr >= 0]))

        print(f"  Shape: {self.Xtr.shape}, Classes: {self.num_classes}")

    def __call__(self, trial):
        """Optuna trial evaluation"""

        # Sample hyperparameters
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 192, 256, 384]),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.7),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "num_epochs": trial.suggest_int("num_epochs", 40, 120),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.01),
            "early_stopping_patience": 20,
        }

        print(f"\n[Trial {trial.number}] lr={params['learning_rate']:.6f}, "
              f"hidden={params['hidden_size']}, dropout={params['dropout_rate']:.3f}, "
              f"bs={params['batch_size']}, epochs={params['num_epochs']}")

        # Evaluate with multiple seeds
        f1w_scores = []

        for seed in self.seeds:
            # Set random seed
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            # Create model
            if self.model_type == "mlp":
                model = MLP(self.input_dim, params["hidden_size"],
                           self.num_classes, params["dropout_rate"])
            else:
                model = SimpleLSTM(self.input_dim, params["hidden_size"],
                                  self.num_classes, params["dropout_rate"])

            # Train and evaluate
            metrics = train_and_evaluate(
                model, self.Xtr, self.ytr, self.Xva, self.yva,
                self.Xte, self.yte, params, self.device
            )

            f1w_scores.append(metrics["f1_weighted"])
            print(f"  Seed {seed}: F1w={metrics['f1_weighted']:.4f}")

        # Return mean F1 weighted
        mean_f1w = np.mean(f1w_scores)
        std_f1w = np.std(f1w_scores)

        print(f"  => Mean F1w: {mean_f1w:.4f} +/- {std_f1w:.4f}")

        return mean_f1w


class HierObjective:
    """Optuna objective for Hier model optimization"""

    def __init__(self, task, layer, pool, model_type, seeds, device='cuda'):
        self.task = task
        self.layer = layer
        self.pool = pool
        self.model_type = model_type
        self.seeds = seeds
        self.device = device

        self._load_data()

    def _load_data(self):
        """Load embeddings"""
        print(f"\n[DATA] Loading Hier embeddings...")
        print(f"  Task: {self.task}, Layer: {self.layer}, Pool: {self.pool}")

        self.Xtr, self.ytr = load_hier_embeddings(self.task, self.layer, self.pool, "train")
        self.Xva, self.yva = load_hier_embeddings(self.task, self.layer, self.pool, "val")
        self.Xte, self.yte = load_hier_embeddings(self.task, self.layer, self.pool, "test")

        self.input_dim = self.Xtr.shape[1]
        self.num_classes = len(np.unique(self.ytr[self.ytr >= 0]))

        print(f"  Shape: {self.Xtr.shape}, Classes: {self.num_classes}")

    def __call__(self, trial):
        """Optuna trial evaluation"""

        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 192, 256, 384]),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.7),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "num_epochs": trial.suggest_int("num_epochs", 40, 120),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.01),
            "early_stopping_patience": 20,
        }

        print(f"\n[Trial {trial.number}] lr={params['learning_rate']:.6f}, "
              f"hidden={params['hidden_size']}, dropout={params['dropout_rate']:.3f}, "
              f"bs={params['batch_size']}, epochs={params['num_epochs']}")

        f1w_scores = []

        for seed in self.seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            if self.model_type == "mlp":
                model = MLP(self.input_dim, params["hidden_size"],
                           self.num_classes, params["dropout_rate"])
            else:
                model = SimpleLSTM(self.input_dim, params["hidden_size"],
                                  self.num_classes, params["dropout_rate"])

            metrics = train_and_evaluate(
                model, self.Xtr, self.ytr, self.Xva, self.yva,
                self.Xte, self.yte, params, self.device
            )

            f1w_scores.append(metrics["f1_weighted"])
            print(f"  Seed {seed}: F1w={metrics['f1_weighted']:.4f}")

        mean_f1w = np.mean(f1w_scores)
        std_f1w = np.std(f1w_scores)

        print(f"  => Mean F1w: {mean_f1w:.4f} +/- {std_f1w:.4f}")

        return mean_f1w


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Utterance-level Bayesian Optimization")
    parser.add_argument("--task", required=True, choices=["4way", "6way"])
    parser.add_argument("--model_tag", required=True,
                        choices=["sentence-roberta", "sentence-roberta-hier"],
                        help="sentence-roberta for Flat, sentence-roberta-hier for Hier")
    parser.add_argument("--layer", default="avg_last4")
    parser.add_argument("--pool", required=True,
                        choices=["mean", "wmean_pos", "wmean_pos_rev"])
    parser.add_argument("--model_type", default="mlp", choices=["mlp", "lstm"])
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                        help="Seeds for averaging (default: 42 43 44)")
    parser.add_argument("--gpu", default="0")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine model type
    is_flat = args.model_tag == "sentence-roberta"
    model_label = "flat" if is_flat else "hier"

    print("="*70)
    print("UTTERANCE-LEVEL BAYESIAN OPTIMIZATION")
    print("="*70)
    print(f"Task:       {args.task}")
    print(f"Model:      {args.model_tag} ({model_label})")
    print(f"Layer:      {args.layer}")
    print(f"Pool:       {args.pool}")
    print(f"Classifier: {args.model_type}")
    print(f"Trials:     {args.n_trials}")
    print(f"Seeds:      {args.seeds}")
    print(f"Device:     {device}")
    print("="*70)

    # Create objective
    if is_flat:
        objective = FlatObjective(
            task=args.task,
            layer=args.layer,
            pool=args.pool,
            model_type=args.model_type,
            seeds=args.seeds,
            device=device
        )
    else:
        objective = HierObjective(
            task=args.task,
            layer=args.layer,
            pool=args.pool,
            model_type=args.model_type,
            seeds=args.seeds,
            device=device
        )

    # Create study
    study_name = f"utterance_{args.task}_{model_label}_{args.pool}_{args.model_type}"

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
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULT_DIR / f"bayesopt_{args.task}_{model_label}_{args.pool}_{args.model_type}.json"

    result = {
        "task": args.task,
        "model_tag": args.model_tag,
        "model_type": model_label,
        "layer": args.layer,
        "pool": args.pool,
        "classifier": args.model_type,
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
    study_file = RESULT_DIR / f"study_{args.task}_{model_label}_{args.pool}_{args.model_type}.pkl"
    with open(study_file, "wb") as f:
        pickle.dump(study, f)

    print(f"Study saved to: {study_file}")


if __name__ == "__main__":
    main()
