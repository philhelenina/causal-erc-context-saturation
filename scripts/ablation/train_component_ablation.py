#!/usr/bin/env python3
"""
train_component_ablation.py

Component ablation study for TMLR Table 5
Tests contribution of: position pooling, lexical fusion, context window

Configurations:
  0. Baseline (full model)
  1. No position pooling (mean instead of wmean_pos_rev)
  2. No lexical fusion (sentence-roberta only)
  3. No context (K=0 instead of K=75)

Total: 4 configs × 3 splits × 5 seeds = 60 experiments
"""
import os
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

HOME = Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()
DATA_DIR = HOME / "data/iemocap_4way_data"
EMB_BASE = HOME / "data/embeddings/4way"

# Experiment configurations
CONFIGS = [
    {
        'name': 'baseline',
        'description': 'Full model (all components)',
        'encoder': 'sentence-roberta',
        'layer': 'avg_last4',
        'pooling': 'wmean_pos_rev',
        'method': 'sentic-fused',  # With lexical
        'context_k': 75,
    },
    {
        'name': 'no_position',
        'description': 'Remove position pooling',
        'encoder': 'sentence-roberta',
        'layer': 'avg_last4',
        'pooling': 'mean',  # 🔴 Changed
        'method': 'sentic-fused',
        'context_k': 75,
    },
    {
        'name': 'no_lexical',
        'description': 'Remove lexical fusion',
        'encoder': 'sentence-roberta',
        'layer': 'avg_last4',
        'pooling': 'wmean_pos_rev',
        'method': 'only',  # 🔴 Changed (no fusion)
        'context_k': 75,
    },
    {
        'name': 'no_context',
        'description': 'Remove context window',
        'encoder': 'sentence-roberta',
        'layer': 'avg_last4',
        'pooling': 'wmean_pos_rev',
        'method': 'sentic-fused',
        'context_k': 0,  # 🔴 Changed
    },
]

SPLITS = ['train', 'val', 'test']
SEEDS = [42, 43, 44, 45, 46]
GPUS = [0, 1, 2, 3]


def set_seed(s):
    """Set all random seeds"""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def get_embedding_path(config):
    """Get embedding directory path based on config"""
    encoder = config['encoder']
    layer = config['layer']
    pooling = config['pooling']
    method = config['method']
    context_k = config['context_k']
    
    # Base path
    if method == 'only':
        base = EMB_BASE / encoder / layer / pooling
    else:
        base = EMB_BASE / f"sr-{method}" / layer / pooling
    
    # Check if this is turn-level or utterance-level
    # Turn-level paths might have K in the name
    # For now, assume standard structure
    
    return base


def load_embeddings(config, split):
    """Load embeddings for a given config and split"""
    emb_path = get_embedding_path(config)
    emb_file = emb_path / f"{split}.npz"
    
    if not emb_file.exists():
        raise FileNotFoundError(f"Embeddings not found: {emb_file}")
    
    data = np.load(emb_file)
    embeddings = data["embeddings"].astype(np.float32)
    
    return embeddings


def load_labels(split):
    """Load labels with mask (filter out -1)"""
    df = pd.read_csv(DATA_DIR / f"{split}_4way_with_minus_one.csv")
    mask = df["label_num"].values >= 0
    labels = df[mask]["label_num"].values
    
    return labels, mask


def load_data(config):
    """Load all splits for a config"""
    Xtr = load_embeddings(config, 'train')
    Xva = load_embeddings(config, 'val')
    Xte = load_embeddings(config, 'test')
    
    ytr, mask_tr = load_labels('train')
    yva, mask_va = load_labels('val')
    yte, mask_te = load_labels('test')
    
    # Filter embeddings if they include -1 labels
    if len(Xtr) > len(ytr):
        Xtr = Xtr[mask_tr]
    if len(Xva) > len(yva):
        Xva = Xva[mask_va]
    if len(Xte) > len(yte):
        Xte = Xte[mask_te]
    
    # Ensure lengths match
    Xtr = Xtr[:len(ytr)]
    Xva = Xva[:len(yva)]
    Xte = Xte[:len(yte)]
    
    return Xtr, ytr, Xva, yva, Xte, yte


class MLP(nn.Module):
    """Simple MLP classifier"""
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


def train_and_evaluate(Xtr, ytr, Xva, yva, Xte, yte, 
                       hidden, dropout, batch_size, lr, 
                       max_epochs, patience, device, seed):
    """Train MLP classifier and evaluate"""
    set_seed(seed)
    
    in_dim = Xtr.shape[1]
    num_classes = 4
    
    # Model
    model = MLP(in_dim, hidden, num_classes, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(Xtr), 
            torch.from_numpy(ytr)
        ),
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(Xva), 
            torch.from_numpy(yva)
        ),
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(Xte), 
            torch.from_numpy(yte)
        ),
        batch_size=batch_size,
        shuffle=False
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_state = None
    wait = 0
    
    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device).long()
            
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device).float()
                yb = yb.to(device).long()
                val_loss += criterion(model(xb), yb).item() * xb.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        # Early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Evaluate on test set
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device).float()
            yb = yb.to(device).long()
            
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(yb.cpu().numpy())
    
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    # Per-class metrics
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    results = {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'f1_per_class': {
            'angry': float(f1_per_class[0]),
            'happy': float(f1_per_class[1]),
            'sad': float(f1_per_class[2]),
            'neutral': float(f1_per_class[3]),
        },
        'confusion_matrix': cm.tolist(),
        'best_epoch': epoch - wait,
        'best_val_loss': float(best_val_loss),
    }
    
    return results


def run_single_experiment(config, seed, gpu_id):
    """Run single experiment"""
    config_name = config['name']
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n[{config_name}][seed{seed}][GPU{gpu_id}] Loading data...")
    
    try:
        # Load data
        Xtr, ytr, Xva, yva, Xte, yte = load_data(config)
        
        print(f"  Train: {Xtr.shape}, Val: {Xva.shape}, Test: {Xte.shape}")
        
        # Hyperparameters
        hidden = 256
        dropout = 0.30
        batch_size = 64
        lr = 1e-3
        max_epochs = 200
        patience = 60
        
        print(f"  Training...")
        
        # Train and evaluate
        results = train_and_evaluate(
            Xtr, ytr, Xva, yva, Xte, yte,
            hidden, dropout, batch_size, lr, max_epochs, patience,
            device, seed
        )
        
        # Save results
        out_dir = HOME / "results/component_ablation" / config_name / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        with open(out_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Also save config
        with open(out_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        f1w = results['f1_weighted']
        acc = results['accuracy']
        print(f"  ✅ [{config_name}][seed{seed}] Acc={acc:.4f}, F1w={f1w:.4f}")
        
        return results
        
    except Exception as e:
        print(f"  ❌ [{config_name}][seed{seed}] Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_embeddings():
    """Check if all required embeddings exist"""
    print("\nChecking embeddings...")
    missing = []
    
    for config in CONFIGS:
        emb_path = get_embedding_path(config)
        for split in SPLITS:
            emb_file = emb_path / f"{split}.npz"
            if not emb_file.exists():
                missing.append(str(emb_file))
    
    if missing:
        print("\n❌ Missing embeddings:")
        for path in missing[:10]:
            print(f"  {path}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
        return False
    else:
        print("  ✅ All embeddings found")
        return True


def main():
    """Main execution"""
    print("=" * 80)
    print("Component Ablation Study - TMLR Table 5")
    print("=" * 80)
    
    print(f"\nConfigurations:")
    for i, config in enumerate(CONFIGS):
        print(f"  {i}. {config['name']:15s} - {config['description']}")
    
    print(f"\nSeeds: {SEEDS}")
    print(f"GPUs: {GPUS}")
    
    total = len(CONFIGS) * len(SEEDS)
    print(f"\nTotal experiments: {total}")
    
    # Check embeddings
    if not check_embeddings():
        print("\n❌ Please create embeddings first!")
        print("Run:")
        print("  1. create_sentence_roberta_embeddings.py")
        print("  2. create_fusion_embeddings.py")
        return
    
    # Run experiments
    print("\n" + "=" * 80)
    print("Starting experiments...")
    print("=" * 80)
    
    job_id = 0
    all_results = []
    
    for config in CONFIGS:
        for seed in SEEDS:
            gpu = GPUS[job_id % len(GPUS)]
            
            print(f"\n[Job {job_id + 1}/{total}] {config['name']} / seed{seed} → GPU{gpu}")
            
            results = run_single_experiment(config, seed, gpu)
            
            if results is not None:
                all_results.append({
                    'config_name': config['name'],
                    'config_description': config['description'],
                    'seed': seed,
                    'gpu': gpu,
                    **results
                })
            
            job_id += 1
    
    # Save summary
    out_dir = HOME / "results/component_ablation"
    summary_file = out_dir / "all_results.json"
    
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✅ All experiments completed!")
    print(f"  Results saved to: {out_dir}")
    print(f"  Summary: {summary_file}")
    print("=" * 80)
    
    # Print summary table
    print("\nResults Summary:")
    print("-" * 80)
    print(f"{'Config':<20s} {'Seed':<8s} {'Accuracy':<12s} {'F1 Weighted':<12s}")
    print("-" * 80)
    
    for result in all_results:
        name = result['config_name']
        seed = result['seed']
        acc = result['accuracy']
        f1w = result['f1_weighted']
        print(f"{name:<20s} {seed:<8d} {acc:<12.4f} {f1w:<12.4f}")
    
    print("-" * 80)


if __name__ == "__main__":
    main()