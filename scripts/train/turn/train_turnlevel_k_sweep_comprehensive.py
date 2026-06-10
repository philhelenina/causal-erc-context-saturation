#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_turnlevel_k_sweep.py - COMPREHENSIVE VERSION FOR TMLR
===========================================================
Complete turn-level K-window analysis with per-emotion metrics and saturation analysis.

Features:
- Per-emotion performance tracking (recall, precision, F1)
- Confusion matrix evolution
- Per-emotion MI curves
- Saturation curve fitting
- Bayesian optimization hyperparameter loading
- Comprehensive visualizations (12+ plots)
- Full reproducibility support

Usage:
  python scripts/train_turnlevel_k_sweep.py \
    --task 6way \
    --model_tag sentence-roberta \
    --layer avg_last4 \
    --pool wmean_pos_rev \
    --models lstm \
    --k_min 0 --k_max 100 --k_step 5 \
    --epochs 50 --patience 10 \
    --use_bayesopt --bayesopt_dir results/bayesian_optimization
"""
import os

import argparse, json, os, time, warnings
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, 
                            classification_report, precision_recall_fscore_support)
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
import torch, torch.nn as nn

warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================
HOME = Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()
DATA = HOME / "data"
OUT_ROOT = HOME / "results" / "turnlevel_k_sweep"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Emotion labels
LABELS_4WAY = {0: 'ang', 1: 'hap', 2: 'sad', 3: 'neu'}
LABELS_6WAY = {0: 'ang', 1: 'hap', 2: 'sad', 3: 'neu', 4: 'exc', 5: 'fru'}

# ============================================================
# BAYESIAN OPTIMIZATION LOADER
# ============================================================
def load_bayesian_hyperparams(bayesopt_dir, layer, pool, model_name):
    """
    Load Bayesian optimized hyperparameters from JSON file.
    
    Expected filename format: bayesopt_{layer}_{pool}_{model}_best.json
    
    Returns:
        dict with keys: learning_rate, hidden_size, dropout_rate, 
                       batch_size, num_epochs, weight_decay
        or None if file not found
    """
    bayesopt_path = Path(bayesopt_dir)
    
    if not bayesopt_path.exists():
        return None
    
    # Construct filename
    filename = f"bayesopt_{layer}_{pool}_{model_name}_best.json"
    filepath = bayesopt_path / filename
    
    if not filepath.exists():
        print(f"  ⚠️  Bayesian opt file not found: {filename}")
        return None
    
    try:
        with open(filepath, 'r') as f:
            params = json.load(f)
        
        print(f"  ✅ Loaded Bayesian hyperparameters from: {filename}")
        print(f"     LR={params['learning_rate']:.6f}, Hidden={params['hidden_size']}, "
              f"Dropout={params['dropout_rate']:.3f}, BS={params['batch_size']}, "
              f"Epochs={params['num_epochs']}")
        
        return params
    
    except Exception as e:
        print(f"  ⚠️  Failed to load {filename}: {e}")
        return None

# ============================================================
# UTILITIES
# ============================================================
def load_npz(p: Path):
    """Load embeddings from .npz file"""
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    arr = np.load(p)
    return arr["embeddings"] if "embeddings" in arr else arr[list(arr.keys())[0]]

def compute_mi_with_pca(X, y, max_samples=10000, n_components=512):
    """
    Compute MI using PCA for dimension reduction.
    More principled than random feature subsampling.
    
    References:
        - Kraskov et al. (2004): MI estimation with k-NN
        - Gao et al. (2015): Feature reduction for MI estimation
    """
    if X is None or len(X) == 0:
        return 0.0
    
    y = np.asarray(y)
    original_dim = X.shape[1]
    
    # Subsample data points if needed
    if X.shape[0] > max_samples:
        idx = np.random.choice(X.shape[0], max_samples, replace=False)
        X, y = X[idx], y[idx]
    
    # PCA dimension reduction if needed
    variance_retained = 1.0
    if X.shape[1] > n_components:
        pca = PCA(n_components=n_components, random_state=42)
        X = pca.fit_transform(X)
        variance_retained = pca.explained_variance_ratio_.sum()
    
    try:
        mi = mutual_info_classif(X, y, discrete_features=False, 
                                random_state=42, n_neighbors=5)
        return float(np.mean(mi))
    except Exception as e:
        print(f"  ⚠️  MI failed: {e}")
        return 0.0

def compute_per_emotion_mi(X, y, label_map, max_samples=10000, n_components=512):
    """
    Compute MI for each emotion vs rest (binary classification).
    Returns dict: {emotion: MI_value}
    """
    emotion_mi = {}
    
    for label_idx, emotion_name in label_map.items():
        # Binary labels: this emotion vs all others
        y_binary = (y == label_idx).astype(int)
        
        # Skip if emotion not present
        if y_binary.sum() == 0:
            emotion_mi[emotion_name] = 0.0
            continue
        
        mi = compute_mi_with_pca(X, y_binary, max_samples, n_components)
        emotion_mi[emotion_name] = mi
    
    return emotion_mi

def build_windows_by_dialogue(X, y, file_ids, k):
    """Build K-window inputs preserving dialogue structure."""
    Xw, yw = [], []
    fids = np.asarray(file_ids)
    N, D = X.shape
    
    for fid in np.unique(fids):
        idxs = np.where(fids == fid)[0]
        
        for i, idx in enumerate(idxs):
            start = max(0, i - k)
            seq = X[idxs[start:i+1]]
            
            if seq.shape[0] < k+1:
                pad = np.zeros((k+1 - seq.shape[0], D), dtype=X.dtype)
                seq = np.vstack([pad, seq])
            
            Xw.append(seq)
            yw.append(y[idx])
    
    Xw = np.stack(Xw, axis=0)
    yw = np.asarray(yw, dtype=np.int64)
    mask = (yw >= 0)
    return Xw[mask], yw[mask]

def saturation_function(k, baseline, a_max, tau):
    """Exponential saturation model: y = baseline + a_max * (1 - exp(-k/tau))"""
    return baseline + a_max * (1 - np.exp(-k / tau))

def fit_saturation_curve(k_values, metric_values):
    """
    Fit saturation curve to data.
    Returns: dict with params or None if fitting fails
    """
    try:
        # Initial guess
        baseline_guess = metric_values[0] if len(metric_values) > 0 else 0.5
        max_val = np.max(metric_values) if len(metric_values) > 0 else 1.0
        a_max_guess = max_val - baseline_guess
        tau_guess = 10.0
        
        params, _ = curve_fit(
            saturation_function, 
            k_values, 
            metric_values,
            p0=[baseline_guess, a_max_guess, tau_guess],
            bounds=([0, 0, 0.1], [1, 1, 200]),
            maxfev=5000
        )
        
        return {
            'baseline': params[0],
            'a_max': params[1],
            'tau': params[2],
            'asymptotic': params[0] + params[1]
        }
    except Exception as e:
        print(f"    ⚠️  Saturation fit failed: {e}")
        return None

# ============================================================
# MODELS
# ============================================================
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, nclass, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden // 2, nclass)
        )
    
    def forward(self, x):
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        return self.net(x)

class SimpleLSTM(nn.Module):
    def __init__(self, in_dim, hidden, nclass, drop=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0
        )
        self.dropout = nn.Dropout(drop)
        self.fc = nn.Linear(hidden, nclass)
    
    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"LSTM expects 3D input, got shape {x.shape}")
        out, (h, _) = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

# ============================================================
# TRAINING & EVALUATION
# ============================================================
def train_epoch(model, opt, crit, X, y, device, bs=256):
    model.train()
    N = len(X)
    loss_sum = 0.0
    
    for st in range(0, N, bs):
        ed = min(N, st + bs)
        xb = torch.tensor(X[st:ed], dtype=torch.float32, device=device)
        yb = torch.tensor(y[st:ed], dtype=torch.long, device=device)
        
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step()
        
        loss_sum += loss.item() * (ed - st)
    
    return loss_sum / max(1, N)

@torch.no_grad()
def evaluate_detailed(model, X, y, device, label_map, bs=256):
    """Comprehensive evaluation with per-class metrics"""
    model.eval()
    N = len(X)
    preds = []
    
    for st in range(0, N, bs):
        ed = min(N, st + bs)
        xb = torch.tensor(X[st:ed], dtype=torch.float32, device=device)
        pred = model(xb).argmax(1).cpu().numpy()
        preds.append(pred)
    
    yhat = np.concatenate(preds) if preds else np.empty((0,), dtype=np.int64)
    
    # Overall metrics
    overall = {
        'acc': accuracy_score(y, yhat) if len(yhat) else 0.0,
        'f1w': f1_score(y, yhat, average="weighted", zero_division=0) if len(yhat) else 0.0,
        'f1m': f1_score(y, yhat, average="macro", zero_division=0) if len(yhat) else 0.0
    }
    
    # Confusion matrix
    cm = confusion_matrix(y, yhat, labels=list(label_map.keys()))
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y, yhat, labels=list(label_map.keys()), zero_division=0
    )
    
    per_class = {}
    for idx, emotion in label_map.items():
        per_class[emotion] = {
            'recall': float(recall[idx]),
            'precision': float(precision[idx]),
            'f1': float(f1[idx]),
            'support': int(support[idx])
        }
    
    return overall, per_class, cm, yhat

@torch.no_grad()
def compute_val_loss(model, X, y, device, crit, bs=256):
    model.eval()
    N = len(X)
    loss_sum = 0.0
    
    for st in range(0, N, bs):
        ed = min(N, st + bs)
        xb = torch.tensor(X[st:ed], dtype=torch.float32, device=device)
        yb = torch.tensor(y[st:ed], dtype=torch.long, device=device)
        loss_sum += crit(model(xb), yb).item() * (ed - st)
    
    return loss_sum / max(1, N)

# ============================================================
# EXPERIMENT
# ============================================================
def run_k_experiment(args, k, model_name, label_map,
                    Xtr_k, ytr_k, Xva_k, yva_k, Xte_k, yte_k, device,
                    bayesian_params=None):
    """Run complete experiment for one K value"""
    
    _, seq_len, D = Xtr_k.shape
    nclass = len(label_map)
    
    print(f"\n{'─'*70}")
    print(f"K={k:3d} | Model={model_name.upper()}")
    print(f"  Data: Train={Xtr_k.shape[0]}, Val={Xva_k.shape[0]}, Test={Xte_k.shape[0]}")
    
    # Use Bayesian params if available, otherwise use defaults
    if bayesian_params:
        hidden = bayesian_params['hidden_size']
        dropout = bayesian_params['dropout_rate']
        lr = bayesian_params['learning_rate']
        bs = bayesian_params['batch_size']
        epochs = bayesian_params['num_epochs']
        weight_decay = bayesian_params.get('weight_decay', 0.0)
        print(f"  🎯 Using Bayesian optimized hyperparameters")
    else:
        hidden = args.hidden
        dropout = args.dropout
        lr = args.lr
        bs = args.bs
        epochs = args.epochs
        weight_decay = 0.0
        print(f"  📊 Using default hyperparameters")
    
    # Create model
    if model_name == "mlp":
        in_dim = seq_len * D
        model = MLP(in_dim, hidden, nclass, dropout).to(device)
    else:
        model = SimpleLSTM(D, hidden, nclass, dropout).to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    
    # Training
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    train_start = time.time()
    
    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, opt, crit, Xtr_k, ytr_k, device, bs=bs)
        val_loss = compute_val_loss(model, Xva_k, yva_k, device, crit, bs=bs)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"    Early stop at epoch {epoch}")
                break
        
        if epoch == 1 or epoch % 10 == 0:
            print(f"    Epoch {epoch:3d}: train_loss={tr_loss:.4f}, val_loss={val_loss:.4f}")
    
    train_time = time.time() - train_start
    
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    
    # Detailed evaluation
    overall, per_class, cm, yhat = evaluate_detailed(
        model, Xte_k, yte_k, device, label_map, bs=bs
    )
    
    # Overall MI
    Xtr_k_flat = Xtr_k.reshape(Xtr_k.shape[0], -1)
    mi_overall = compute_mi_with_pca(Xtr_k_flat, ytr_k,
                                    max_samples=10000, n_components=512)
    
    # Per-emotion MI
    emotion_mi = compute_per_emotion_mi(Xtr_k_flat, ytr_k, label_map,
                                       max_samples=10000, n_components=512)
    
    print(f"  ✅ Acc={overall['acc']:.4f}, F1w={overall['f1w']:.4f}, "
          f"F1m={overall['f1m']:.4f}, MI={mi_overall:.4f}, Time={train_time:.1f}s")
    
    # Print per-emotion summary
    for emotion, metrics in per_class.items():
        print(f"     {emotion:3s}: R={metrics['recall']:.3f} P={metrics['precision']:.3f} "
              f"F1={metrics['f1']:.3f} MI={emotion_mi.get(emotion, 0):.4f}")
    
    return {
        'model': model_name,
        'k': k,
        'overall': overall,
        'per_class': per_class,
        'emotion_mi': emotion_mi,
        'mi_overall': mi_overall,
        'confusion_matrix': cm.tolist(),
        'train_time': train_time,
        'hyperparams': {
            'hidden': hidden,
            'dropout': dropout,
            'lr': lr,
            'batch_size': bs,
            'epochs': epochs,
            'weight_decay': weight_decay,
            'source': 'bayesian' if bayesian_params else 'default'
        }
    }

# ============================================================
# VISUALIZATION (Simplified for artifact size)
# ============================================================
def plot_comprehensive_analysis(results_df, label_map, out_dir, task):
    """Generate all plots for TMLR paper"""
    
    out_dir = Path(out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📊 Generating comprehensive visualizations...")
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150
    
    emotions = list(label_map.values())
    
    # Plot 1: Per-Emotion Recall vs K
    print("  [1/5] Per-emotion recall curves...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for model_name, model_df in results_df.groupby('model'):
        model_df = model_df.sort_values('k')
        for emotion in emotions:
            col = f'{emotion}_recall'
            if col in model_df.columns:
                ax.plot(model_df['k'], model_df[col], 
                       marker='o', label=f'{emotion.upper()} ({model_name})',
                       linewidth=2.5, markersize=5)
    
    ax.set_xlabel('Context Window (K)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Recall', fontsize=14, fontweight='bold')
    ax.set_title(f'{task.upper()}: Per-Emotion Recall vs Context Window', 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, ncol=2, loc='best', framealpha=0.95)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(plots_dir / '01_per_emotion_recall_vs_k.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Optimal K per Emotion
    print("  [2/5] Optimal K bar chart...")
    fig, ax = plt.subplots(figsize=(10, 7))
    
    optimal_k_data = []
    for emotion in emotions:
        col = f'{emotion}_recall'
        if col in results_df.columns:
            emotion_data = results_df[results_df['model'] == results_df['model'].iloc[0]]
            if len(emotion_data) > 0:
                best_idx = emotion_data[col].idxmax()
                optimal_k = emotion_data.loc[best_idx, 'k']
                best_recall = emotion_data.loc[best_idx, col]
                optimal_k_data.append({
                    'emotion': emotion.upper(),
                    'optimal_k': optimal_k,
                    'best_recall': best_recall
                })
    
    if optimal_k_data:
        opt_df = pd.DataFrame(optimal_k_data)
        colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#9b59b6', '#e67e22'][:len(opt_df)]
        bars = ax.bar(opt_df['emotion'], opt_df['optimal_k'], color=colors)
        
        for i, (bar, val, recall) in enumerate(zip(bars, opt_df['optimal_k'], opt_df['best_recall'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'K={int(val)}\nR={recall:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Emotion', fontsize=14, fontweight='bold')
        ax.set_ylabel('Optimal Context Window (K)', fontsize=14, fontweight='bold')
        ax.set_title(f'{task.upper()}: Optimal K per Emotion',
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(plots_dir / '02_optimal_k_per_emotion.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Overall Metrics
    print("  [3/5] Overall metrics...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = [('acc', 'Accuracy'), ('f1w', 'Weighted F1'), 
               ('f1m', 'Macro F1'), ('mi_overall', 'Overall MI')]
    
    for ax, (metric, title) in zip(axes.flat, metrics):
        for model_name, model_df in results_df.groupby('model'):
            model_df = model_df.sort_values('k')
            ax.plot(model_df['k'], model_df[metric], 
                   marker='o', label=model_name.upper(),
                   linewidth=2.5, markersize=6)
        
        ax.set_xlabel('Context Window (K)', fontsize=12, fontweight='bold')
        ax.set_ylabel(title, fontsize=12, fontweight='bold')
        ax.set_title(f'{task.upper()}: {title}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(plots_dir / '03_overall_metrics_grid.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Per-Emotion MI
    print("  [4/5] Per-emotion MI curves...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for model_name, model_df in results_df.groupby('model'):
        model_df = model_df.sort_values('k')
        for emotion in emotions:
            col = f'{emotion}_mi'
            if col in model_df.columns:
                ax.plot(model_df['k'], model_df[col],
                       marker='^', linestyle='--', 
                       label=f'{emotion.upper()} ({model_name})',
                       linewidth=2.5, markersize=5)
    
    ax.set_xlabel('Context Window (K)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mutual Information (bits)', fontsize=14, fontweight='bold')
    ax.set_title(f'{task.upper()}: Per-Emotion MI vs Context Window',
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, ncol=2, loc='best', framealpha=0.95)
    plt.tight_layout()
    plt.savefig(plots_dir / '04_per_emotion_mi_vs_k.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 5: Confusion Matrix Evolution
    print("  [5/5] Confusion matrix evolution...")
    key_k_values = [0, 5, 20, 50] if results_df['k'].max() >= 50 else sorted(results_df['k'].unique())[:4]
    
    fig, axes = plt.subplots(1, len(key_k_values), figsize=(5*len(key_k_values), 4))
    if len(key_k_values) == 1:
        axes = [axes]
    
    for idx, k_val in enumerate(key_k_values):
        k_data = results_df[results_df['k'] == k_val]
        if len(k_data) > 0:
            cm = np.array(k_data.iloc[0]['confusion_matrix'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=[e.upper() for e in emotions],
                       yticklabels=[e.upper() for e in emotions],
                       ax=axes[idx], cbar=True)
            
            axes[idx].set_title(f'K={int(k_val)}', fontsize=13, fontweight='bold')
            axes[idx].set_xlabel('Predicted', fontsize=11)
            axes[idx].set_ylabel('True', fontsize=11)
    
    plt.suptitle(f'{task.upper()}: Confusion Matrix Evolution', 
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(plots_dir / '05_confusion_matrix_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ All plots saved to {plots_dir}/")

# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Comprehensive turn-level K-sweep for TMLR")
    
    # Task & Model
    parser.add_argument("--task", choices=["4way", "6way"], required=True)
    parser.add_argument("--model_tag", required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--models", nargs="+", choices=["mlp", "lstm"], default=["lstm"])
    
    # K-sweep
    parser.add_argument("--k_min", type=int, default=0)
    parser.add_argument("--k_max", type=int, default=100)
    parser.add_argument("--k_step", type=int, default=5)
    
    # Training (defaults, will be overridden by Bayesian opt if available)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.7)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    
    # Bayesian Optimization
    parser.add_argument("--use_bayesopt", action="store_true",
                       help="Load hyperparameters from Bayesian optimization results")
    parser.add_argument("--bayesopt_dir", type=str, default="",
                       help="Directory containing Bayesian optimization JSON files")
    
    # Optional
    parser.add_argument("--wandb_project", default="")
    
    args = parser.parse_args()
    
    # Label map
    label_map = LABELS_6WAY if args.task == "6way" else LABELS_4WAY
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE TURN-LEVEL K-SWEEP FOR TMLR")
    print(f"{'='*80}")
    print(f"Task: {args.task} ({len(label_map)} emotions)")
    print(f"Emotions: {list(label_map.values())}")
    print(f"Model: {args.model_tag} / {args.layer} / {args.pool}")
    print(f"K range: [{args.k_min}, {args.k_max}] step {args.k_step}")
    
    if args.use_bayesopt and args.bayesopt_dir:
        print(f"🎯 Bayesian optimization ENABLED: {args.bayesopt_dir}")
    else:
        print(f"📊 Using default hyperparameters")
    
    print(f"{'='*80}\n")
    
    # Load data
    csv_dir = DATA / f"iemocap_{args.task}_data"
    
    def read_split(split):
        csv_path = csv_dir / f"{split}_{args.task}_with_minus_one.csv"
        df = pd.read_csv(csv_path)
        y = df["label_num"].fillna(-1).astype(int).to_numpy()
        fid = df["file_id"].astype(str).to_numpy()
        return df, y, fid
    
    print("📂 Loading data...")
    df_tr, y_tr, fid_tr = read_split("train")
    df_va, y_va, fid_va = read_split("val")
    df_te, y_te, fid_te = read_split("test")
    
    emb_base = DATA / "embeddings" / args.task / args.model_tag / args.layer / args.pool
    print(f"📦 Loading embeddings from:\n  {emb_base}")
    
    X_tr = load_npz(emb_base / "train.npz")
    X_va = load_npz(emb_base / "val.npz")
    X_te = load_npz(emb_base / "test.npz")
    
    def align(X, y, fid):
        n = min(len(X), len(y))
        X, y, fid = X[:n], y[:n], fid[:n]
        mask = (y >= 0)
        return X[mask], y[mask], fid[mask]
    
    X_tr, y_tr, fid_tr = align(X_tr, y_tr, fid_tr)
    X_va, y_va, fid_va = align(X_va, y_va, fid_va)
    X_te, y_te, fid_te = align(X_te, y_te, fid_te)
    print(f"  Final: Train={len(y_tr)}, Val={len(y_va)}, Test={len(y_te)}\n")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_tag = f"{args.task}_{args.model_tag}_{args.layer}_{args.pool}"
    out_dir = OUT_ROOT / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # WandB
    wb = None
    if args.wandb_project:
        try:
            import wandb
            wb = wandb.init(project=args.wandb_project, name=out_tag, config=vars(args))
        except:
            pass
    
    # Load Bayesian hyperparameters if requested
    bayesian_params_dict = {}
    if args.use_bayesopt and args.bayesopt_dir:
        print(f"\n{'='*80}")
        print("LOADING BAYESIAN OPTIMIZATION HYPERPARAMETERS")
        print(f"{'='*80}\n")
        
        for model_name in args.models:
            params = load_bayesian_hyperparams(args.bayesopt_dir, args.layer, args.pool, model_name)
            if params:
                bayesian_params_dict[model_name] = params
            else:
                print(f"  ⚠️  No Bayesian params for {model_name}, will use defaults")
        
        print()
    
    # K-sweep
    k_values = list(range(args.k_min, args.k_max + 1, args.k_step))
    results = []
    
    print(f"{'='*80}")
    print(f"STARTING K-SWEEP: {len(k_values)} × {len(args.models)} = {len(k_values)*len(args.models)} experiments")
    print(f"{'='*80}\n")
    
    for k_idx, k in enumerate(k_values):
        print(f"\n{'='*80}")
        print(f"K = {k} ({k_idx+1}/{len(k_values)})")
        print(f"{'='*80}")
        
        # Build windows
        Xtr_k, ytr_k = build_windows_by_dialogue(X_tr, y_tr, fid_tr, k)
        Xva_k, yva_k = build_windows_by_dialogue(X_va, y_va, fid_va, k)
        Xte_k, yte_k = build_windows_by_dialogue(X_te, y_te, fid_te, k)
        
        for model_name in args.models:
            # Get Bayesian params for this model if available
            bayesian_params = bayesian_params_dict.get(model_name, None)
            
            result = run_k_experiment(
                args, k, model_name, label_map,
                Xtr_k, ytr_k, Xva_k, yva_k, Xte_k, yte_k, device,
                bayesian_params=bayesian_params
            )
            results.append(result)
            
            # WandB logging
            if wb:
                log_dict = {
                    'k': k,
                    'model': model_name,
                    **{f'overall_{key}': val for key, val in result['overall'].items()},
                    **{f'{emotion}_{key}': val[key] 
                       for emotion, val in result['per_class'].items() 
                       for key in ['recall', 'precision', 'f1']},
                    **{f'{emotion}_mi': val for emotion, val in result['emotion_mi'].items()},
                    'mi_overall': result['mi_overall']
                }
                wb.log(log_dict)
    
    # Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")
    
    # Flatten results for CSV
    flattened = []
    for r in results:
        row = {
            'model': r['model'],
            'k': r['k'],
            **r['overall'],
            'mi_overall': r['mi_overall'],
            'confusion_matrix': r['confusion_matrix'],
            'train_time': r['train_time'],
            **{f'hp_{key}': val for key, val in r['hyperparams'].items()}
        }
        # Add per-class metrics
        for emotion, metrics in r['per_class'].items():
            for key, val in metrics.items():
                row[f'{emotion}_{key}'] = val
        # Add per-emotion MI
        for emotion, mi in r['emotion_mi'].items():
            row[f'{emotion}_mi'] = mi
        
        flattened.append(row)
    
    df = pd.DataFrame(flattened)
    csv_path = out_dir / "k_sweep_comprehensive.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ {csv_path}")
    
    # Generate plots
    plot_comprehensive_analysis(df, label_map, out_dir, args.task)
    
    # Analysis summary
    print(f"\n{'='*80}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    analysis = {
        'config': vars(args),
        'timestamp': datetime.now().isoformat(),
        'total_experiments': len(results),
        'emotions': list(label_map.values()),
        'bayesian_optimization_used': args.use_bayesopt and bool(bayesian_params_dict),
        'optimal_k_per_emotion': {},
        'best_performance_per_emotion': {},
        'saturation_params': {}
    }
    
    for emotion in label_map.values():
        recall_col = f'{emotion}_recall'
        if recall_col in df.columns:
            emotion_data = df[df['model'] == df['model'].iloc[0]]
            best_idx = emotion_data[recall_col].idxmax()
            optimal_k = int(emotion_data.loc[best_idx, 'k'])
            best_recall = float(emotion_data.loc[best_idx, recall_col])
            best_f1 = float(emotion_data.loc[best_idx, f'{emotion}_f1'])
            
            analysis['optimal_k_per_emotion'][emotion] = optimal_k
            analysis['best_performance_per_emotion'][emotion] = {
                'k': optimal_k,
                'recall': best_recall,
                'f1': best_f1
            }
            
            # Saturation curve
            k_vals = emotion_data['k'].values
            recall_vals = emotion_data[recall_col].values
            if len(k_vals) >= 4:
                fit_params = fit_saturation_curve(k_vals, recall_vals)
                if fit_params:
                    analysis['saturation_params'][emotion] = fit_params
            
            print(f"\n{emotion.upper()}:")
            print(f"  Optimal K: {optimal_k}")
            print(f"  Best Recall: {best_recall:.4f}")
            print(f"  Best F1: {best_f1:.4f}")
            if emotion in analysis['saturation_params']:
                tau = analysis['saturation_params'][emotion]['tau']
                print(f"  Saturation τ: {tau:.2f} turns")
    
    # Save analysis JSON
    json_path = out_dir / "analysis_summary.json"
    with open(json_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\n✅ {json_path}")
    
    print(f"\n{'='*80}")
    print("✅ COMPREHENSIVE ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Results: {out_dir}")
    print(f"{'='*80}\n")
    
    if wb:
        wb.finish()

if __name__ == "__main__":
    main()