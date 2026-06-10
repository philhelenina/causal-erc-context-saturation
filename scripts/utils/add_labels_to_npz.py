#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_labels_to_npz.py

Add labels from CSV to existing NPZ files
"""
import os

import numpy as np
import pandas as pd
from pathlib import Path

HOME = Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()
DATA = HOME / "data"

# Label mappings
LABELS_4 = {"ang": 0, "hap": 1, "sad": 2, "neu": 3}
LABELS_6 = {"ang": 0, "hap": 1, "sad": 2, "neu": 3, "exc": 4, "fru": 5}


def load_labels_from_csv(task, split):
    """Load labels from CSV"""
    csv_path = DATA / f"iemocap_{task}_data" / f"{split}_{task}_unified.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing: {csv_path}")
    
    df = pd.read_csv(csv_path)
    label_map = LABELS_6 if task == "6way" else LABELS_4
    
    # Find label column
    label_cols = ["label_num", "label", "label_num4", "label_num6"]
    label_col = next((c for c in label_cols if c in df.columns), None)
    
    if not label_col:
        raise ValueError(f"No label column in {csv_path}")
    
    ser = df[label_col]
    
    # Convert to numeric
    if ser.dtype != object:
        y = ser.fillna(-1).astype("int64").to_numpy()
    else:
        y = ser.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        y = y.str.lower().map(label_map).fillna(-1).astype("int64").to_numpy()
    
    return y


def add_labels_to_npz(npz_path, labels):
    """Add labels to existing NPZ file"""
    # Load existing data
    arr = np.load(npz_path, allow_pickle=True)
    
    # Extract all existing arrays
    data = {key: arr[key] for key in arr.keys()}
    
    # Add labels (use 'y' as standard key)
    if "y" in data:
        print(f"  [SKIP] Labels already exist in {npz_path.name}")
        return False
    
    # Verify dimensions match
    embeddings = data.get("embeddings", data.get("X"))
    if len(embeddings) != len(labels):
        raise ValueError(
            f"Dimension mismatch in {npz_path.name}: "
            f"embeddings={len(embeddings)}, labels={len(labels)}"
        )
    
    # Add labels
    data["y"] = labels
    
    # Save updated NPZ
    np.savez(npz_path, **data)
    print(f"  [OK] Added {len(labels)} labels to {npz_path.name}")
    
    return True


def main():
    import argparse
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["4way", "6way", "both"])
    ap.add_argument("--encoder", default="sentence-roberta-hier")
    ap.add_argument("--layer", default="avg_last4")
    ap.add_argument("--pool", default="mean")
    ap.add_argument("--dry_run", action="store_true", help="Don't modify files")
    args = ap.parse_args()
    
    tasks = ["4way", "6way"] if args.task == "both" else [args.task]
    splits = ["train", "val", "test"]
    
    for task in tasks:
        print(f"\n{'='*80}")
        print(f"TASK: {task}")
        print(f"{'='*80}")
        
        emb_dir = DATA / "embeddings" / task / args.encoder / args.layer / args.pool
        
        if not emb_dir.exists():
            print(f"[SKIP] Directory not found: {emb_dir}")
            continue
        
        print(f"Directory: {emb_dir}")
        
        for split in splits:
            print(f"\n[{split.upper()}]")
            
            npz_path = emb_dir / f"{split}.npz"
            
            if not npz_path.exists():
                print(f"  [SKIP] File not found: {npz_path.name}")
                continue
            
            # Load labels from CSV
            labels = load_labels_from_csv(task, split)
            print(f"  Loaded {len(labels)} labels from CSV")
            
            if args.dry_run:
                print(f"  [DRY RUN] Would add labels to {npz_path.name}")
            else:
                # Add to NPZ
                add_labels_to_npz(npz_path, labels)
    
    print(f"\n{'='*80}")
    print("✅ COMPLETE!")
    print(f"{'='*80}")
    
    if args.dry_run:
        print("\nThis was a dry run. Run without --dry_run to apply changes.")


if __name__ == "__main__":
    main()
