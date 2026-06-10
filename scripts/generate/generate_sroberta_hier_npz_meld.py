#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_sroberta_hier_npz_meld.py
MELD 7-way emotion recognition dataset embedding generator

Usage:
    python generate_sroberta_hier_npz_meld.py \
        --data_dir /path/to/meld_7way_data \
        --layer avg_last4 \
        --poolings mean wmean_pos

Expected CSV format (train_meld_unified.csv, val_meld_unified.csv, test_meld_unified.csv):
    id,utterance,label_num,file_id,utterance_num
    dia0_utt0,"Oh my God!",3,dia0,0

Label mapping (7-way):
    0: anger, 1: disgust, 2: fear, 3: joy, 4: neutral, 5: sadness, 6: surprise
"""
import argparse
import math
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# Default paths - modify as needed
HOME = Path(__file__).parent.parent.parent  # SenticCrystal root
DATA_DIR_DEFAULT = HOME / "data" / "meld_7way_data"
OUT_ROOT_DEFAULT = HOME / "data" / "embeddings" / "meld_7way" / "sentence-roberta-hier"


def simple_sent_split(text: str) -> List[str]:
    """Conservative sentence splitter (., !, ? + space)"""
    parts = re.split(r'(?<=[\.!\?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def to_device(d, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}


def masked_mean(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    m = attn_mask.float()
    s = m.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return (last_hidden * m.unsqueeze(-1)).sum(dim=1) / s


def attn_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    cls = last_hidden[:, 0, :]
    scores = (last_hidden * cls.unsqueeze(1)).sum(dim=-1) / math.sqrt(last_hidden.size(-1))
    scores = scores.masked_fill(attn_mask == 0, -1e9)
    w = torch.softmax(scores / max(tau, 1e-6), dim=1)
    return (last_hidden * w.unsqueeze(-1)).sum(dim=1)


def pos_weights(
    attn_mask: torch.Tensor,
    mode: str = "linear_rev",
    tau: float = 5.0
) -> torch.Tensor:
    """Position-based weights"""
    B, T = attn_mask.shape
    device = attn_mask.device

    if mode == "linear_rev":
        pos = torch.arange(1, T + 1, device=device, dtype=torch.float32)
        w_raw = torch.flip(pos, dims=[0])
    elif mode == "linear":
        w_raw = torch.arange(1, T + 1, device=device, dtype=torch.float32)
    elif mode == "exp_decay":
        t = torch.arange(1, T + 1, device=device, dtype=torch.float32)
        w_raw = torch.exp(-(t - 1) / tau)
    elif mode == "biperiphery":
        t = torch.arange(1, T + 1, device=device, dtype=torch.float32)
        w_raw = torch.maximum(t, T - t + 1)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    w = w_raw.unsqueeze(0).expand(B, T) * attn_mask.float()
    s = w.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return w / s


def combine_layers(hidden_states: Tuple[torch.Tensor, ...], mode: str, scalar_w: Optional[List[float]]):
    if mode == "last":
        return hidden_states[-1]
    if mode == "avg_last4":
        return torch.stack(hidden_states[-4:], dim=0).mean(0)
    if mode.startswith("last4_scalar"):
        if scalar_w is None:
            raise ValueError("scalar_w required for last4_scalar*")
        w = torch.tensor(scalar_w, dtype=torch.float32, device=hidden_states[-1].device)
        w = w / (w.sum() + 1e-8)
        hs = torch.stack(hidden_states[-4:], dim=0)
        return (hs * w.view(4, 1, 1, 1)).sum(0)
    raise ValueError(f"unknown layer mode: {mode}")


PRESETS = {
    "last4_scalar": [1, 1, 1, 1],
    "last4_scalar_up": [1, 2, 3, 4],
    "last4_scalar_down": [4, 3, 2, 1],
    "last4_scalar_top2": [0, 1, 1, 0],
}


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="sentence-transformers/nli-roberta-base-v2")
    ap.add_argument("--data_dir", default=str(DATA_DIR_DEFAULT))
    ap.add_argument("--out_root", default=str(OUT_ROOT_DEFAULT))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--text_col", default="utterance")
    ap.add_argument("--label_col", default="label_num")
    ap.add_argument("--id_col", default="id")

    # Layer options
    ap.add_argument("--layer", choices=["last", "avg_last4", "last4_scalar",
                                         "last4_scalar_up", "last4_scalar_down",
                                         "last4_scalar_top2"],
                    default="avg_last4")
    ap.add_argument("--scalar_weights", default="")

    # Pooling options
    ap.add_argument("--poolings", nargs="+", default=["mean", "wmean_pos"],
                    choices=["cls", "mean", "attn",
                            "wmean_pos", "wmean_pos_rev", "wmean_biperiphery",
                            "wmean_exp_fast", "wmean_exp_med", "wmean_exp_slow"])
    ap.add_argument("--attn_tau", type=float, default=1.0)
    ap.add_argument("--exp_tau_fast", type=float, default=2.0)
    ap.add_argument("--exp_tau_med", type=float, default=5.0)
    ap.add_argument("--exp_tau_slow", type=float, default=10.0)

    # Other
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--cpu", action="store_true")
    return ap


def main():
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[INFO] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModel.from_pretrained(args.model_name).eval().to(device)
    print(f"[INFO] Loaded model: {args.model_name}")

    # Layer scalar weights
    scalar_w = None
    if args.layer.startswith("last4_scalar"):
        if args.scalar_weights:
            scalar_w = [float(x) for x in args.scalar_weights.split(",")]
        else:
            scalar_w = PRESETS[args.layer]

    out_root = Path(args.out_root)
    data_dir = Path(args.data_dir)

    for split in args.splits:
        # Try different CSV naming patterns
        csv_candidates = [
            f"{split}_meld_unified.csv",
            f"{split}_meld.csv",
            f"{split}.csv",
        ]

        csv_path = None
        for cand in csv_candidates:
            p = data_dir / cand
            if p.exists():
                csv_path = p
                break

        if csv_path is None:
            print(f"[ERROR] No CSV found for {split} in {data_dir}")
            print(f"        Tried: {csv_candidates}")
            continue

        print(f"\n[LOAD] {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"       Columns: {list(df.columns)}")
        print(f"       Rows: {len(df)}")

        # Get text column
        text_col = args.text_col
        if text_col not in df.columns:
            for c in ["text", "utterance", "utt", "sentence", "transcript", "Utterance"]:
                if c in df.columns:
                    text_col = c
                    break

        texts = df[text_col].astype(str).fillna("").tolist()

        # Get labels
        label_col = args.label_col
        if label_col not in df.columns:
            for c in ["label_num", "Emotion", "emotion", "label", "Label"]:
                if c in df.columns:
                    label_col = c
                    break

        # Handle string labels (convert to int)
        if df[label_col].dtype == object:
            label_map = {
                'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3,
                'neutral': 4, 'sadness': 5, 'surprise': 6,
                # Alternative names
                'angry': 0, 'happy': 3, 'sad': 5
            }
            labels = df[label_col].str.lower().map(label_map).fillna(-1).astype(int).values
        else:
            labels = df[label_col].astype(int).values

        print(f"       Text column: {text_col}")
        print(f"       Label column: {label_col}")
        print(f"       Label distribution: {np.bincount(labels[labels >= 0])}")

        # Get utterance IDs
        id_col = args.id_col
        if id_col not in df.columns:
            for c in ["id", "utterance_id", "utt_id", "Utterance_ID"]:
                if c in df.columns:
                    id_col = c
                    break

        if id_col in df.columns:
            utterance_ids = df[id_col].astype(str).tolist()
        elif 'Dialogue_ID' in df.columns and 'Utterance_ID' in df.columns:
            utterance_ids = (df['Dialogue_ID'].astype(str) + '_' +
                           df['Utterance_ID'].astype(str)).tolist()
        else:
            utterance_ids = [f"{split}_{i}" for i in range(len(df))]

        per_pool_arrays = {p: [] for p in args.poolings}
        lengths = []

        print(f"[EMBED] Processing {len(texts)} utterances...")
        for idx, txt in enumerate(texts):
            if idx % 500 == 0:
                print(f"        {idx}/{len(texts)}")

            sents = simple_sent_split(txt)
            if not sents:
                sents = [txt.strip() if txt.strip() else "."]
            lengths.append(len(sents))

            enc = tokenizer(sents, padding=True, truncation=True,
                          max_length=args.max_length, return_tensors="pt")
            enc = to_device(enc, device)

            with torch.no_grad():
                out = model(**enc, output_hidden_states=True, return_dict=True)
                last_h = combine_layers(out.hidden_states, args.layer, scalar_w)
                attn_mask = enc["attention_mask"]

                pool_outs = {}
                for p in args.poolings:
                    if p == "cls":
                        pooled = last_h[:, 0, :]
                    elif p == "mean":
                        pooled = masked_mean(last_h, attn_mask)
                    elif p == "attn":
                        pooled = attn_pool(last_h, attn_mask, tau=args.attn_tau)
                    elif p == "wmean_pos":
                        w = pos_weights(attn_mask, mode="linear")
                        pooled = (last_h * w.unsqueeze(-1)).sum(1)
                    elif p == "wmean_pos_rev":
                        w = pos_weights(attn_mask, mode="linear_rev")
                        pooled = (last_h * w.unsqueeze(-1)).sum(1)
                    elif p == "wmean_biperiphery":
                        w = pos_weights(attn_mask, mode="biperiphery")
                        pooled = (last_h * w.unsqueeze(-1)).sum(1)
                    elif p == "wmean_exp_fast":
                        w = pos_weights(attn_mask, mode="exp_decay", tau=args.exp_tau_fast)
                        pooled = (last_h * w.unsqueeze(-1)).sum(1)
                    elif p == "wmean_exp_med":
                        w = pos_weights(attn_mask, mode="exp_decay", tau=args.exp_tau_med)
                        pooled = (last_h * w.unsqueeze(-1)).sum(1)
                    elif p == "wmean_exp_slow":
                        w = pos_weights(attn_mask, mode="exp_decay", tau=args.exp_tau_slow)
                        pooled = (last_h * w.unsqueeze(-1)).sum(1)
                    else:
                        raise ValueError(p)
                    pool_outs[p] = pooled.detach().cpu().numpy().astype(np.float32)

                for p in args.poolings:
                    per_pool_arrays[p].append(pool_outs[p])

        # Pad and save
        N = len(texts)
        S_max = max(lengths) if lengths else 1

        for p in args.poolings:
            H = model.config.hidden_size
            out = np.zeros((N, S_max, H), dtype=np.float32)
            for i, mat in enumerate(per_pool_arrays[p]):
                s = mat.shape[0]
                out[i, :s, :] = mat

            save_dir = out_root / args.layer / p
            save_dir.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(
                save_dir / f"{split}.npz",
                embeddings=out,
                lengths=np.asarray(lengths, dtype=np.int32),
                utterance_ids=np.asarray(utterance_ids, dtype=object),
                y=labels  # Labels included!
            )
            print(f"[SAVE] {split} -> {save_dir}/{split}.npz")
            print(f"       shape={out.shape}, S_max={S_max}, N={N}")


if __name__ == "__main__":
    main()
