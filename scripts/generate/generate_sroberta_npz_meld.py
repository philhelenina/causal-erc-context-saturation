#!/usr/bin/env python3
"""
Generate flat Sentence-RoBERTa embeddings for MELD 7-way.
Output: (N, 768) per split
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "meld_7way_data"
OUT_ROOT = PROJECT_ROOT / "data" / "embeddings" / "meld_7way" / "sentence-roberta"


class TextDataset(Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx]


def masked_mean(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling over non-padded tokens."""
    mask = attn_mask.float().unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return (last_hidden * mask).sum(dim=1) / denom


def combine_layers(hidden_states, mode: str = "avg_last4"):
    """Combine transformer layers."""
    if mode == "last":
        return hidden_states[-1]
    elif mode == "avg_last4":
        hs = torch.stack(hidden_states[-4:], dim=0)
        return hs.mean(dim=0)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def generate_embeddings(
    texts: List[str],
    model,
    tokenizer,
    device,
    layer: str = "avg_last4",
    pool: str = "mean",
    batch_size: int = 32,
    max_length: int = 128
) -> np.ndarray:
    """Generate embeddings for a list of texts."""

    dataset = TextDataset(texts)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []

    with torch.no_grad():
        for batch_texts in loader:
            # Tokenize
            inputs = tokenizer(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            # Combine layers
            combined = combine_layers(hidden_states, mode=layer)

            # Pool tokens
            if pool == "mean":
                pooled = masked_mean(combined, inputs["attention_mask"])
            elif pool == "cls":
                pooled = combined[:, 0, :]
            else:
                pooled = masked_mean(combined, inputs["attention_mask"])

            all_embeddings.append(pooled.cpu().numpy())

    return np.vstack(all_embeddings).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="sentence-transformers/nli-roberta-base")
    parser.add_argument("--layer", default="avg_last4", choices=["last", "avg_last4"])
    parser.add_argument("--pool", default="mean", choices=["mean", "cls"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModel.from_pretrained(args.model_name)
    model.eval().to(device)

    # Output directory
    out_dir = OUT_ROOT / args.layer / args.pool
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        print(f"\n=== Processing {split} ===")

        # Load data
        csv_path = DATA_DIR / f"{split}_meld_unified.csv"
        df = pd.read_csv(csv_path)
        texts = df["utterance"].astype(str).tolist()
        labels = df["label_num"].values

        print(f"  Texts: {len(texts)}")

        # Generate embeddings
        embeddings = generate_embeddings(
            texts, model, tokenizer, device,
            layer=args.layer, pool=args.pool,
            batch_size=args.batch_size
        )

        print(f"  Embeddings: {embeddings.shape}")

        # Save
        out_path = out_dir / f"{split}.npz"
        np.savez(out_path, embeddings=embeddings, labels=labels)
        print(f"  Saved: {out_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
