#!/usr/bin/env python3
"""
train_npz_hier_classifier.py

Train hierarchical LSTM classifier on pre-computed embeddings

Usage:
    python train_npz_hier_classifier.py \
        --task 4way \
        --encoder bert-base-hier \
        --layer avg_last4 \
        --pool mean \
        --classifier lstm \
        --seed 42 \
        --gpu 0
"""
import os
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd

# ============================================
# Dataset
# ============================================

class NPZDataset(Dataset):
    """Dataset for pre-computed hierarchical embeddings"""
    
    def __init__(self, embeddings, labels, lengths):
        """
        Args:
            embeddings: (N, S_max, D) numpy array
            labels: (N,) numpy array
            lengths: (N,) numpy array - number of valid sentences per sample
        """
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)
        self.lengths = torch.LongTensor(lengths)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'embeddings': self.embeddings[idx],  # (S_max, D)
            'labels': self.labels[idx],
            'lengths': self.lengths[idx]
        }

# ============================================
# Model
# ============================================

# ============================================
# Aggregators (Stage 1: Sentence → Utterance)
# ============================================

def make_mask(B, S, lens, device):
    """Create mask from lengths"""
    return (torch.arange(S, device=device).unsqueeze(0) < lens.unsqueeze(1))  # (B,S)

def agg_mean(xs, mask):
    """Masked mean aggregation"""
    denom = mask.sum(1, keepdim=True).clamp_min(1)
    return (xs * mask.unsqueeze(-1)).sum(1) / denom  # (B,D)

class MeanAggregator(nn.Module):
    """Mean aggregation over sentences"""
    def forward(self, xs, lens):
        """
        Args:
            xs: (B, S_max, D) - sentence embeddings
            lens: (B,) - number of valid sentences
        Returns:
            (B, D) - utterance embedding
        """
        B, S, D = xs.shape
        mask = make_mask(B, S, lens, xs.device)
        return agg_mean(xs, mask)

# ============================================
# Classifiers (Stage 2: Utterance → Label)
# ============================================

class MLPClassifier(nn.Module):
    """MLP classifier for aggregated utterance embeddings"""
    
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, D) - utterance embedding
        Returns:
            logits: (B, num_classes)
        """
        return self.net(x)

class LSTMClassifier(nn.Module):
    """LSTM classifier for aggregated utterance embeddings"""
    
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, 1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: (B, D) - utterance embedding
        Returns:
            logits: (B, num_classes)
        """
        # x: (B, D) → (B, 1, D) for LSTM
        x = x.unsqueeze(1)
        _, (h, _) = self.lstm(x)
        h = h.squeeze(0)  # (B, hidden_dim)
        h = self.dropout(h)
        return self.fc(h)

# ============================================
# Combined Model (Aggregator + Classifier)
# ============================================

class HierarchicalModel(nn.Module):
    """2-stage hierarchical model: Aggregator → Classifier"""
    
    def __init__(self, aggregator, classifier):
        super().__init__()
        self.aggregator = aggregator
        self.classifier = classifier
    
    def forward(self, embeddings, lengths):
        """
        Args:
            embeddings: (B, S_max, D) - sentence embeddings
            lengths: (B,) - number of valid sentences
        Returns:
            logits: (B, num_classes)
        """
        # Stage 1: Aggregate sentences → utterance
        utterance_emb = self.aggregator(embeddings, lengths)  # (B, D)
        
        # Stage 2: Classify utterance
        logits = self.classifier(utterance_emb)  # (B, num_classes)
        
        return logits

# ============================================
# Legacy Models (DEPRECATED - for backward compatibility)
# ============================================

class HierarchicalLSTM(nn.Module):
    """Hierarchical LSTM classifier"""
    
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        
        # Note: dropout in LSTM only applies between layers when num_layers > 1
        # For single layer, we apply dropout separately after LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0  # Set to 0 for single layer
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, embeddings, lengths):
        """
        Args:
            embeddings: (B, S_max, D)
            lengths: (B,) - number of valid sentences
        
        Returns:
            logits: (B, num_classes)
        """
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, 
            lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # LSTM forward
        packed_output, (hidden, cell) = self.lstm(packed)
        
        # Use last hidden state
        # hidden: (1, B, hidden_dim)
        hidden = hidden.squeeze(0)  # (B, hidden_dim)
        
        # Dropout + FC
        hidden = self.dropout(hidden)
        logits = self.fc(hidden)
        
        return logits

class HierarchicalMLP(nn.Module):
    """Hierarchical MLP classifier (mean pooling + MLP)"""
    
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, embeddings, lengths):
        """
        Args:
            embeddings: (B, S_max, D)
            lengths: (B,) - number of valid sentences
        
        Returns:
            logits: (B, num_classes)
        """
        # Masked mean pooling
        B, S_max, D = embeddings.shape
        
        # Create mask
        mask = torch.arange(S_max, device=embeddings.device)[None, :] < lengths[:, None]
        mask = mask.float().unsqueeze(-1)  # (B, S_max, 1)
        
        # Masked mean
        masked_sum = (embeddings * mask).sum(dim=1)  # (B, D)
        masked_count = mask.sum(dim=1).clamp(min=1)  # (B, 1)
        pooled = masked_sum / masked_count  # (B, D)
        
        # MLP
        hidden = self.fc1(pooled)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)
        logits = self.fc2(hidden)
        
        return logits

# ============================================
# Training
# ============================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in loader:
        embeddings = batch['embeddings'].to(device)
        labels = batch['labels'].to(device)
        lengths = batch['lengths'].to(device)
        
        # Forward
        logits = model(embeddings, lengths)
        loss = criterion(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track
        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy

def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            embeddings = batch['embeddings'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch['lengths'].to(device)
            
            # Forward
            logits = model(embeddings, lengths)
            loss = criterion(logits, labels)
            
            # Track
            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1_macro, f1_weighted, all_preds, all_labels

# ============================================
# Main
# ============================================

def load_data(data_dir, split):
    """Load NPZ data and labels from CSV, with utterance ID matching"""
    npz_file = data_dir / f"{split}.npz"
    
    if not npz_file.exists():
        raise FileNotFoundError(f"Data file not found: {npz_file}")
    
    data = np.load(npz_file, allow_pickle=True)
    
    embeddings = data['embeddings']
    lengths = data['lengths'] if 'lengths' in data else np.array([embeddings.shape[1]] * len(embeddings))
    
    # Try to load labels from NPZ first
    if 'labels' in data and 'y' not in str(data_dir):  # Has labels and not using old format
        labels = data['labels']
        return embeddings, labels, lengths
    
    # Load labels from CSV
    print(f"  [INFO] Labels not in NPZ, loading from CSV...")
    
    # Determine task from path
    if '4way' in str(data_dir):
        task = '4way'
        label_map = {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3}
    elif '6way' in str(data_dir):
        task = '6way'
        label_map = {'ang': 0, 'exc': 1, 'fru': 2, 'hap': 3, 'neu': 4, 'sad': 5}
    else:
        raise ValueError(f"Cannot determine task from path: {data_dir}")
    
    # Load CSV
    csv_path = Path(str(data_dir).split('/embeddings/')[0]) / f"iemocap_{task}_data/{split}_{task}_unified.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    import pandas as pd
    df = pd.read_csv(csv_path)
    
    print(f"  [INFO] CSV has {len(df)} rows, NPZ has {len(embeddings)} embeddings")
    
    # Check if we have utterance IDs in both
    if 'utterance_ids' in data:
        # Match by utterance ID
        npz_utt_ids = data['utterance_ids']
        
        # Get utterance ID column from CSV
        if 'utterance_id' in df.columns:
            csv_utt_col = 'utterance_id'
        elif 'utt_id' in df.columns:
            csv_utt_col = 'utt_id'
        elif 'id' in df.columns:
            csv_utt_col = 'id'
        else:
            # Try to construct utterance IDs from dialog_id and utterance_num
            if 'dialog_id' in df.columns and 'utterance_num' in df.columns:
                df['utterance_id'] = df['dialog_id'] + '_' + df['utterance_num'].astype(str)
                csv_utt_col = 'utterance_id'
            else:
                raise ValueError(f"Cannot find utterance ID column in CSV. Columns: {df.columns.tolist()}")
        
        csv_utt_ids = df[csv_utt_col].values
        
        # Find matching indices
        npz_to_csv_idx = {}
        for npz_idx, utt_id in enumerate(npz_utt_ids):
            npz_to_csv_idx[utt_id] = npz_idx
        
        # Filter embeddings to match CSV
        matched_indices = []
        matched_labels = []
        
        for csv_idx, csv_utt_id in enumerate(csv_utt_ids):
            if csv_utt_id in npz_to_csv_idx:
                npz_idx = npz_to_csv_idx[csv_utt_id]
                matched_indices.append(npz_idx)
                
                # Get label
                if 'label_num' in df.columns:
                    label = df.iloc[csv_idx]['label_num']
                elif 'label' in df.columns:
                    label = label_map[df.iloc[csv_idx]['label']]
                else:
                    raise ValueError(f"No label column found in CSV")
                
                matched_labels.append(label)
        
        if len(matched_indices) == 0:
            raise ValueError("No matching utterances found between NPZ and CSV!")
        
        # Filter embeddings and lengths
        matched_indices = np.array(matched_indices)
        embeddings = embeddings[matched_indices]
        lengths = lengths[matched_indices]
        labels = np.array(matched_labels, dtype=np.int64)
        
        print(f"  [INFO] Matched {len(labels)} utterances by ID")
        
    else:
        # No utterance IDs - assume NPZ and CSV are in same order but NPZ has more
        print(f"  [WARNING] No utterance IDs in NPZ, assuming first {len(df)} embeddings match CSV")
        
        # Get labels
        if 'label_num' in df.columns:
            labels = df['label_num'].values
        elif 'label' in df.columns:
            labels = df['label'].map(label_map).values
        else:
            raise ValueError(f"No label column found in {csv_path}")
        
        # Take only the first len(labels) embeddings
        embeddings = embeddings[:len(labels)]
        lengths = lengths[:len(labels)]
        
        print(f"  [INFO] Using first {len(labels)} embeddings from NPZ")
    
    return embeddings, labels, lengths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, choices=['4way', '6way'])
    parser.add_argument('--encoder', required=True)
    parser.add_argument('--layer', default='avg_last4')
    parser.add_argument('--pool', default='mean')
    parser.add_argument('--classifier', default='lstm', choices=['lstm', 'mlp'])
    parser.add_argument('--seed', type=int, default=42)
    # Note: GPU selection is handled by CUDA_VISIBLE_DEVICES environment variable
    
    # Hyperparameters (matching original experiments)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--patience', type=int, default=60,
                       help='Early stopping patience')
    parser.add_argument('--decay_lambda', type=float, default=0.5,
                       help='Learning rate decay factor')
    
    parser.add_argument('--save_dir', default='results/senticnet_experiments')
    parser.add_argument('--root', default=str(Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()))
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device - always use cuda:0 when CUDA_VISIBLE_DEVICES is set
    # The parallel launcher sets CUDA_VISIBLE_DEVICES, so the visible GPU is always device 0
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    # Paths
    ROOT = Path(args.root)
    data_dir = ROOT / f"data/embeddings/{args.task}/{args.encoder}/{args.layer}/{args.pool}"
    save_dir = ROOT / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Training: {args.encoder} | {args.task} | seed={args.seed}")
    print(f"{'='*80}")
    print(f"Data dir: {data_dir}")
    print(f"Device: {device}")
    print(f"Classifier: {args.classifier}")
    print(f"{'='*80}\n")
    
    # Load data
    print("Loading data...")
    train_emb, train_labels, train_lengths = load_data(data_dir, 'train')
    val_emb, val_labels, val_lengths = load_data(data_dir, 'val')
    test_emb, test_labels, test_lengths = load_data(data_dir, 'test')
    
    # Filter out invalid labels (-1)
    def filter_invalid_labels(embeddings, labels, lengths):
        """Remove samples with label=-1"""
        valid_mask = labels != -1
        n_invalid = (~valid_mask).sum()
        if n_invalid > 0:
            print(f"  [INFO] Filtering {n_invalid} samples with label=-1")
        return embeddings[valid_mask], labels[valid_mask], lengths[valid_mask]
    
    train_emb, train_labels, train_lengths = filter_invalid_labels(train_emb, train_labels, train_lengths)
    val_emb, val_labels, val_lengths = filter_invalid_labels(val_emb, val_labels, val_lengths)
    test_emb, test_labels, test_lengths = filter_invalid_labels(test_emb, test_labels, test_lengths)
    
    print(f"  Train: {train_emb.shape}, labels: {len(np.unique(train_labels))}")
    print(f"  Val:   {val_emb.shape}")
    print(f"  Test:  {test_emb.shape}")
    
    # Create datasets
    train_dataset = NPZDataset(train_emb, train_labels, train_lengths)
    val_dataset = NPZDataset(val_emb, val_labels, val_lengths)
    test_dataset = NPZDataset(test_emb, test_labels, test_lengths)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model
    input_dim = train_emb.shape[2]
    
    # Debug: Check label distribution
    unique_labels = np.unique(train_labels)
    print(f"\n[DEBUG] Label distribution:")
    print(f"  Unique labels in train: {unique_labels}")
    print(f"  Min: {train_labels.min()}, Max: {train_labels.max()}")
    if train_labels.min() >= 0:
        print(f"  Label counts: {np.bincount(train_labels.astype(int))}")
    else:
        print(f"  [WARNING] Negative labels present, skipping bincount")
    
    # Determine num_classes from task, not from data
    if args.task == '4way':
        num_classes = 4
        expected_labels = {0, 1, 2, 3}
    else:  # 6way
        num_classes = 6
        expected_labels = {0, 1, 2, 3, 4, 5}
    
    # Verify labels are correct
    actual_labels = set(unique_labels.tolist())
    if not actual_labels.issubset(expected_labels):
        print(f"\n[ERROR] Invalid labels found!")
        print(f"  Expected: {expected_labels}")
        print(f"  Found: {actual_labels}")
        print(f"  Invalid: {actual_labels - expected_labels}")
        raise ValueError(f"Labels out of range for {args.task}")
    
    print(f"  Num classes (from task): {num_classes}")
    
    # Create 2-stage model: Aggregator + Classifier
    # Stage 1: Mean Aggregator (best performing)
    aggregator = MeanAggregator()
    
    # Stage 2: Classifier
    if args.classifier == 'lstm':
        classifier = LSTMClassifier(input_dim, args.hidden_dim, num_classes, args.dropout)
    else:  # mlp
        classifier = MLPClassifier(input_dim, args.hidden_dim, num_classes, args.dropout)
    
    # Combined model
    model = HierarchicalModel(aggregator, classifier)
    model = model.to(device)
    
    print(f"\nModel: 2-stage Hierarchical (Mean Aggregator + {args.classifier.upper()} Classifier)")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Num classes: {num_classes}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler (exponential decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_lambda)
    
    # Training loop with early stopping
    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})...")
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    best_model_state = None  # Initialize
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1_macro, val_f1_weighted, _, _ = evaluate(model, val_loader, criterion, device)
        
        # Learning rate decay
        if epoch > 0 and epoch % 10 == 0:  # Decay every 10 epochs
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{args.epochs}: "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1_macro:.4f} | "
                  f"lr={current_lr:.6f}")
        else:
            print(f"Epoch {epoch+1:3d}/{args.epochs}: "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1_macro:.4f}")
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best val accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
            break
    
    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"\nBest val accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    
    # Final test evaluation
    test_loss, test_acc, test_f1_macro, test_f1_weighted, test_preds, test_labels = \
        evaluate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1 (macro): {test_f1_macro:.4f}")
    print(f"  F1 (weighted): {test_f1_weighted:.4f}")
    
    # Save results
    result = {
        'encoder': args.encoder,
        'task': args.task,
        'layer': args.layer,
        'pool': args.pool,
        'classifier': args.classifier,
        'seed': args.seed,
        
        # Hyperparameters
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'weight_decay': args.weight_decay,
        'decay_lambda': args.decay_lambda,
        
        # Training info
        'max_epochs': args.epochs,
        'patience': args.patience,
        'best_epoch': best_epoch,
        'stopped_early': best_epoch < args.epochs,
        
        # Results
        'best_val_acc': float(best_val_acc),
        'test_acc': float(test_acc),
        'test_f1_macro': float(test_f1_macro),
        'test_f1_weighted': float(test_f1_weighted),
        'timestamp': datetime.now().isoformat()
    }
    
    # Save to CSV (append mode)
    results_csv = save_dir / 'results.csv'
    results_df = pd.DataFrame([result])
    
    if results_csv.exists():
        results_df.to_csv(results_csv, mode='a', header=False, index=False)
    else:
        results_df.to_csv(results_csv, mode='w', header=True, index=False)
    
    print(f"\n✓ Results saved to {results_csv}")
    
    # Save detailed results
    result_file = save_dir / f"{args.encoder}_{args.task}_seed{args.seed}.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✓ Detailed results saved to {result_file}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()