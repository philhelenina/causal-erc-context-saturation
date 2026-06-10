#!/usr/bin/env python3
"""Check NPZ file structure"""
import os

import numpy as np
from pathlib import Path

# Check both flat and hierarchical
paths = [
    str(Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve() / "data/embeddings/4way/sentence-roberta/avg_last4/mean/train.npz"),
    str(Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve() / "data/embeddings/4way/sentence-roberta-hier/avg_last4/mean/train.npz")
]

for path_str in paths:
    path = Path(path_str)
    
    if not path.exists():
        print(f"❌ Not found: {path}")
        continue
    
    print(f"\n{'='*80}")
    print(f"File: {path.name}")
    print(f"Path: .../{'/'.join(path.parts[-6:])}")
    print(f"{'='*80}")
    
    arr = np.load(path, allow_pickle=True)
    
    print(f"\nKeys in NPZ:")
    for key in arr.keys():
        data = arr[key]
        if hasattr(data, 'shape'):
            print(f"  {key:<20} {str(data.shape):<20} {data.dtype}")
        else:
            print(f"  {key:<20} {type(data).__name__}")
    
    # Check specific keys
    print(f"\nChecking common key names:")
    for key in ['embeddings', 'X', 'y', 'labels', 'ids', 'utterance_ids', 'lengths']:
        exists = "✓" if key in arr else "✗"
        print(f"  {exists} {key}")
