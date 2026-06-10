#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_sentic_fusion_hierarchical_npz.py

Fuse hierarchical BERT embeddings with sentence-level SenticNet features

Input:
  - BERT embeddings: (N, S_max, 768)
  - SenticNet features: (N, S_max, 4)

Output:
  - Concatenated: (N, S_max, 772) or
  - Gated: (N, S_max, 772) with alpha weighting

Usage:
  # Simple concatenation
  python make_sentic_fusion_hierarchical_npz.py \
      --task 4way \
      --encoder bert-base-hier \
      --fusion_mode concat
  
  # Gated fusion with alpha=0.10
  python make_sentic_fusion_hierarchical_npz.py \
      --task 4way \
      --encoder sentence-roberta-hier \
      --fusion_mode gated \
      --alpha 0.10


for encoder in bert-base-hier roberta-base-hier sentence-roberta-hier; do
    for alpha in "${ALPHAS[@]}"; do
        python make_sentic_fusion_hierarchical_npz.py \
            --task 4way \
            --encoder $encoder \
            --fusion_mode gated \
            --alpha $alpha
    done
done
"""
import os
import argparse
import json
from pathlib import Path
import numpy as np
from typing import Tuple, Optional

def load_npz(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load embeddings, labels, and lengths from NPZ file"""
    data = np.load(path, allow_pickle=True)
    
    # Embeddings
    if 'embeddings' in data:
        X = data['embeddings']
    elif 'X' in data:
        X = data['X']
    else:
        X = data[data.files[0]]
    
    # Labels
    y = None
    if 'labels' in data:
        y = data['labels']
    elif 'y' in data:
        y = data['y']
    
    # Lengths
    lengths = None
    if 'lengths' in data:
        lengths = data['lengths']
    
    return X, y, lengths

def save_npz(path: Path, X: np.ndarray, y: Optional[np.ndarray] = None, 
             lengths: Optional[np.ndarray] = None):
    """Save embeddings, labels, and lengths to NPZ file"""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if y is not None and lengths is not None:
        np.savez_compressed(path, embeddings=X, labels=y, lengths=lengths)
    elif y is not None:
        np.savez_compressed(path, embeddings=X, labels=y)
    elif lengths is not None:
        np.savez_compressed(path, embeddings=X, lengths=lengths)
    else:
        np.savez_compressed(path, embeddings=X)

def standardize_block(X: np.ndarray, mean: Optional[np.ndarray] = None, 
                     std: Optional[np.ndarray] = None, eps: float = 1e-6) -> Tuple:
    """
    Standardize embeddings (z-score normalization)
    
    Args:
        X: (N, S_max, D) embeddings
        mean: Pre-computed mean (for val/test)
        std: Pre-computed std (for val/test)
        eps: Small constant to avoid division by zero
    
    Returns:
        Z: Standardized embeddings
        mean: Mean used
        std: Std used
    """
    # Compute along sample dimension (axis=0)
    if mean is None:
        mean = X.mean(axis=0, keepdims=True)
    if std is None:
        std = X.std(axis=0, keepdims=True)
    
    # Avoid division by zero
    std = np.where(std < eps, eps, std)
    
    # Standardize
    Z = (X - mean) / std
    
    return Z, mean, std

def main():
    parser = argparse.ArgumentParser(description='Fuse hierarchical BERT + SenticNet')
    parser.add_argument('--task', choices=['4way', '6way'], required=True,
                       help='Classification task')
    parser.add_argument('--encoder', required=True,
                       choices=['bert-base-hier', 'roberta-base-hier', 'sentence-roberta-hier'],
                       help='BERT encoder type')
    parser.add_argument('--layer', default='avg_last4',
                       help='Layer aggregation method')
    parser.add_argument('--pool', default='mean',
                       help='Pooling method')
    parser.add_argument('--fusion_mode', choices=['concat', 'gated'], default='concat',
                       help='Fusion mode: concat (simple) or gated (alpha-weighted)')
    parser.add_argument('--alpha', type=float, default=0.10,
                       help='Alpha weight for gated fusion (0.0-1.0)')
    parser.add_argument('--standardize', action='store_true',
                       help='Apply z-score standardization before fusion')
    parser.add_argument('--root', default=str(Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()),
                       help='Root directory')
    
    args = parser.parse_args()
    
    ROOT = Path(args.root)
    
    # Input paths
    bert_dir = ROOT / f"data/embeddings/{args.task}/{args.encoder}/{args.layer}/{args.pool}"
    sentic_dir = ROOT / f"data/embeddings/{args.task}/senticnet-sentence-level"
    
    # Output path
    if args.fusion_mode == 'concat':
        out_tag = f"{args.encoder}-sentic-concat"
    else:  # gated
        alpha_tag = f"alpha{int(args.alpha * 100):03d}"
        out_tag = f"{args.encoder}-sentic-{alpha_tag}"
    
    out_dir = ROOT / f"data/embeddings/{args.task}/{out_tag}/{args.layer}/{args.pool}"
    
    print("="*80)
    print(f"HIERARCHICAL FUSION: {args.encoder.upper()} + SENTICNET")
    print("="*80)
    print(f"Task:         {args.task}")
    print(f"Encoder:      {args.encoder}")
    print(f"Layer:        {args.layer}")
    print(f"Pool:         {args.pool}")
    print(f"Fusion mode:  {args.fusion_mode}")
    if args.fusion_mode == 'gated':
        print(f"Alpha:        {args.alpha}")
    print(f"Standardize:  {args.standardize}")
    print()
    print(f"BERT dir:     {bert_dir}")
    print(f"SenticNet dir: {sentic_dir}")
    print(f"Output dir:   {out_dir}")
    print("="*80)
    print()
    
    # Verify input directories exist
    if not bert_dir.exists():
        raise FileNotFoundError(f"BERT directory not found: {bert_dir}")
    if not sentic_dir.exists():
        raise FileNotFoundError(f"SenticNet directory not found: {sentic_dir}")
    
    # Store statistics for standardization (train only)
    stats = {}
    
    for split in ['train', 'val', 'test']:
        print(f"Processing {split.upper()} split...")
        
        # Load BERT embeddings
        bert_file = bert_dir / f"{split}.npz"
        if not bert_file.exists():
            print(f"  ⚠ Skipping {split}: {bert_file} not found")
            continue
        
        X_bert, y, lengths = load_npz(bert_file)
        print(f"  BERT shape: {X_bert.shape}")
        
        # Load SenticNet features
        sentic_file = sentic_dir / f"{split}.npz"
        if not sentic_file.exists():
            raise FileNotFoundError(f"SenticNet file not found: {sentic_file}")
        
        X_sentic, _, sentic_lengths = load_npz(sentic_file)
        print(f"  SenticNet shape: {X_sentic.shape}")
        
        # Validate shapes
        if X_bert.shape[0] != X_sentic.shape[0]:
            raise ValueError(f"N mismatch: BERT {X_bert.shape[0]} vs SenticNet {X_sentic.shape[0]}")
        
        if X_bert.ndim != 3 or X_sentic.ndim != 3:
            raise ValueError(f"Expected 3D tensors! BERT: {X_bert.ndim}D, SenticNet: {X_sentic.ndim}D")
        
        if X_bert.shape[1] != X_sentic.shape[1]:
            raise ValueError(f"S_max mismatch: BERT {X_bert.shape[1]} vs SenticNet {X_sentic.shape[1]}")
        
        # Verify lengths match
        if lengths is not None and sentic_lengths is not None:
            if not np.array_equal(lengths, sentic_lengths):
                print(f"  ⚠ WARNING: Lengths mismatch detected!")
                print(f"    BERT lengths: {lengths[:5]}")
                print(f"    SenticNet lengths: {sentic_lengths[:5]}")
        
        # Optional standardization
        if args.standardize:
            if split == 'train':
                # Compute and save statistics
                X_bert_std, bert_mean, bert_std = standardize_block(X_bert)
                X_sentic_std, sentic_mean, sentic_std = standardize_block(X_sentic)
                stats = {
                    'bert_mean': bert_mean,
                    'bert_std': bert_std,
                    'sentic_mean': sentic_mean,
                    'sentic_std': sentic_std
                }
                print(f"  Standardized (train): computed statistics")
            else:
                # Use train statistics
                X_bert_std, _, _ = standardize_block(X_bert, stats['bert_mean'], stats['bert_std'])
                X_sentic_std, _, _ = standardize_block(X_sentic, stats['sentic_mean'], stats['sentic_std'])
                print(f"  Standardized ({split}): using train statistics")
        else:
            X_bert_std = X_bert
            X_sentic_std = X_sentic
        
        # Fusion
        if args.fusion_mode == 'concat':
            # Simple concatenation
            X_fused = np.concatenate([X_bert_std, X_sentic_std], axis=2)
            print(f"  Fusion (concat): {X_fused.shape}")
        else:  # gated
            # Alpha-weighted fusion
            X_fused = np.concatenate([X_bert_std, args.alpha * X_sentic_std], axis=2)
            print(f"  Fusion (gated α={args.alpha}): {X_fused.shape}")
        
        # Save
        out_file = out_dir / f"{split}.npz"
        save_npz(out_file, X_fused, y, lengths)
        print(f"  ✓ Saved: {out_file}")
        print()
    
    # Save metadata
    meta = {
        'task': args.task,
        'encoder': args.encoder,
        'layer': args.layer,
        'pool': args.pool,
        'fusion_mode': args.fusion_mode,
        'alpha': args.alpha if args.fusion_mode == 'gated' else None,
        'standardize': args.standardize,
        'bert_dir': str(bert_dir),
        'sentic_dir': str(sentic_dir),
        'out_dir': str(out_dir)
    }
    
    meta_file = out_dir / 'fusion_meta.json'
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"✓ Saved metadata: {meta_file}")
    
    print()
    print("="*80)
    print("✅ FUSION COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
