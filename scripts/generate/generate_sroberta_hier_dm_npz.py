#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_sroberta_hier_dm_npz.py
- Hierarchical embeddings with discourse marker weights
- Stores dm_weights array: (N, S_max) - weight per sentence based on DM presence
- NEW: Token-level DM weighting via wmean_dm pooling
- Can be used for DM-weighted sentence aggregation in training

Usage:
    python generate_sroberta_hier_dm_npz.py \
        --task 4way \
        --layer avg_last4 \
        --poolings mean wmean_dm wmean_dm_pos \
        --dm_weight 2.0 \
        --dm_mode count
"""
import argparse
import math
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

HOME = Path(__file__).parent.parent.parent
DATA_DIR_4WAY = HOME / "data" / "iemocap_4way_data"
DATA_DIR_6WAY = HOME / "data" / "iemocap_6way_data"
OUT_ROOT_4WAY = HOME / "data" / "embeddings" / "4way" / "sentence-roberta-hier-dm"
OUT_ROOT_6WAY = HOME / "data" / "embeddings" / "6way" / "sentence-roberta-hier-dm"

# ============================================================================
# ACADEMIC DISCOURSE MARKERS FRAMEWORK
# ============================================================================
DISCOURSE_MARKERS_ACADEMIC = {
    # Schiffrin (1987) - Core markers
    'schiffrin_core': ['oh', 'well', 'you know', 'i mean', 'now', 'then', 'so', 'because', 'and', 'but', 'or'],

    # Fraser (1999, 2009) - Pragmatic markers
    'fraser_contrastive': ['but', 'however', 'although', 'nonetheless', 'nevertheless', 'still', 'yet', 'though'],
    'fraser_elaborative': ['and', 'moreover', 'furthermore', 'besides', 'additionally', 'also', 'too'],
    'fraser_inferential': ['so', 'therefore', 'thus', 'consequently', 'hence', 'accordingly', 'then'],
    'fraser_temporal': ['then', 'meanwhile', 'subsequently', 'afterwards', 'finally', 'next'],

    # Traugott (2010) - Subjectivity markers
    'subjective_epistemic': ['i think', 'i guess', 'i believe', 'maybe', 'perhaps', 'probably', 'possibly'],
    'subjective_attitudinal': ['unfortunately', 'happily', 'sadly', 'frankly', 'honestly', 'personally'],

    # Verhagen (2005) - Intersubjective markers
    'intersubjective': ['you know', 'you see', 'right', 'okay', 'i mean', 'lets say', 'you understand'],

    # Beeching & Detges (2014) - Peripheral markers
    'left_peripheral': ['well', 'so', 'but', 'and', 'oh', 'now', 'look', 'listen'],
    'right_peripheral': ['though', 'right', 'you know', 'i think', 'or something', 'or whatever', 'and stuff'],

    # Aijmer (2013) - Pragmatic particles
    'pragmatic_particles': ['like', 'just', 'really', 'quite', 'pretty', 'sort of', 'kind of', 'actually', 'basically'],

    # Stance markers (Biber & Finegan 1989)
    'stance_certainty': ['definitely', 'certainly', 'obviously', 'clearly', 'surely', 'undoubtedly'],
    'stance_doubt': ['maybe', 'perhaps', 'possibly', 'probably', 'allegedly', 'supposedly']
}

# Build lookup structures
ALL_DM_SINGLE: Set[str] = set()  # single-word markers
ALL_DM_MULTI: List[Tuple[str, ...]] = []  # multi-word markers

for category, markers in DISCOURSE_MARKERS_ACADEMIC.items():
    for marker in markers:
        tokens = marker.split()
        if len(tokens) == 1:
            ALL_DM_SINGLE.add(marker)
        else:
            ALL_DM_MULTI.append(tuple(tokens))

# Legacy list for backward compatibility
DISCOURSE_MARKERS = list(ALL_DM_SINGLE) + [' '.join(m) for m in ALL_DM_MULTI]


def simple_sent_split(text: str) -> List[str]:
    """Conservative sentence splitter (., !, ? + space)"""
    parts = re.split(r'(?<=[\.!\?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def has_discourse_marker(sentence: str, markers: List[str] = DISCOURSE_MARKERS) -> bool:
    """Check if sentence contains any discourse marker"""
    sent_lower = sentence.lower()
    for marker in markers:
        # Check for word boundary matches
        pattern = r'\b' + re.escape(marker) + r'\b'
        if re.search(pattern, sent_lower):
            return True
    return False


def count_discourse_markers(sentence: str, markers: List[str] = DISCOURSE_MARKERS) -> int:
    """Count number of discourse markers in sentence"""
    sent_lower = sentence.lower()
    count = 0
    for marker in markers:
        pattern = r'\b' + re.escape(marker) + r'\b'
        count += len(re.findall(pattern, sent_lower))
    return count


def compute_dm_weights(sentences: List[str], dm_weight: float = 2.0, mode: str = "binary") -> np.ndarray:
    """
    Compute discourse marker weights for sentences

    Args:
        sentences: List of sentences
        dm_weight: Weight multiplier for sentences with DM
        mode: "binary" (has DM or not) or "count" (proportional to DM count)

    Returns:
        weights: (S,) array of weights (unnormalized)
    """
    weights = np.ones(len(sentences), dtype=np.float32)

    for i, sent in enumerate(sentences):
        if mode == "binary":
            if has_discourse_marker(sent):
                weights[i] = dm_weight
        elif mode == "count":
            dm_count = count_discourse_markers(sent)
            weights[i] = 1.0 + (dm_weight - 1.0) * min(dm_count, 3) / 3.0  # Cap at 3

    return weights


def to_device(d, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}


def masked_mean(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    m = attn_mask.float()
    s = m.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return (last_hidden * m.unsqueeze(-1)).sum(dim=1) / s


def pos_weights(attn_mask: torch.Tensor, mode: str = "linear_rev", tau: float = 5.0) -> torch.Tensor:
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


def combine_layers(hidden_states: Tuple[torch.Tensor, ...], mode: str) -> torch.Tensor:
    if mode == "last":
        return hidden_states[-1]
    if mode == "avg_last4":
        return torch.stack(hidden_states[-4:], dim=0).mean(0)
    raise ValueError(f"unknown layer mode: {mode}")


# ============================================================================
# TOKEN-LEVEL DM WEIGHTING (NEW)
# ============================================================================
def dm_weights_token(
    token_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    tokenizer,
    dm_boost: float = 2.0
) -> torch.Tensor:
    """
    Discourse marker based token weights.
    Tokens that are part of discourse markers get boosted weight.

    Args:
        token_ids: (B, T) token IDs
        attn_mask: (B, T) attention mask
        tokenizer: tokenizer for decoding
        dm_boost: multiplicative boost for DM tokens (default 2.0)

    Returns:
        weights: (B, T) normalized weights
    """
    B, T = token_ids.shape
    device = token_ids.device
    weights = torch.ones((B, T), dtype=torch.float32, device=device)

    for b in range(B):
        ids = token_ids[b].tolist()
        # Decode tokens
        toks = tokenizer.convert_ids_to_tokens(ids)
        # Clean tokens (remove special prefix like 'Ġ' for RoBERTa, '▁' for SentencePiece)
        toks_clean = [t.replace('Ġ', '').replace('▁', '').lower().strip('.,!?;:"\'-') for t in toks]

        dm_positions = []
        n = len(toks_clean)

        # Single-word markers
        for i, tok in enumerate(toks_clean):
            if tok in ALL_DM_SINGLE:
                dm_positions.append(i)

        # Multi-word markers (check consecutive tokens)
        for marker_tuple in ALL_DM_MULTI:
            marker_len = len(marker_tuple)
            for i in range(n - marker_len + 1):
                if tuple(toks_clean[i:i+marker_len]) == marker_tuple:
                    dm_positions.extend(range(i, i + marker_len))

        dm_positions = list(set(dm_positions))

        # Apply boost
        row_weights = torch.ones(T, dtype=torch.float32, device=device)
        for pos in dm_positions:
            if pos < T:
                row_weights[pos] = dm_boost

        weights[b] = row_weights

    # Mask and normalize
    weights = weights * attn_mask.float()
    s = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return weights / s


def dm_weights_token_count(
    token_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    tokenizer,
    dm_boost: float = 2.0
) -> torch.Tensor:
    """
    DM-based token weights with count-proportional boosting.
    More DM tokens in the sentence = higher boost.
    """
    B, T = token_ids.shape
    device = token_ids.device
    weights = torch.ones((B, T), dtype=torch.float32, device=device)

    for b in range(B):
        ids = token_ids[b].tolist()
        toks = tokenizer.convert_ids_to_tokens(ids)
        toks_clean = [t.replace('Ġ', '').replace('▁', '').lower().strip('.,!?;:"\'-') for t in toks]

        dm_positions = []
        n = len(toks_clean)

        # Single-word markers
        for i, tok in enumerate(toks_clean):
            if tok in ALL_DM_SINGLE:
                dm_positions.append(i)

        # Multi-word markers
        for marker_tuple in ALL_DM_MULTI:
            marker_len = len(marker_tuple)
            for i in range(n - marker_len + 1):
                if tuple(toks_clean[i:i+marker_len]) == marker_tuple:
                    dm_positions.extend(range(i, i + marker_len))

        dm_positions = list(set(dm_positions))
        dm_count = len(dm_positions)

        # Scale boost by count (capped at 5)
        scaled_boost = 1.0 + (dm_boost - 1.0) * min(dm_count, 5) / 5.0

        row_weights = torch.ones(T, dtype=torch.float32, device=device)
        for pos in dm_positions:
            if pos < T:
                row_weights[pos] = scaled_boost

        weights[b] = row_weights

    weights = weights * attn_mask.float()
    s = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return weights / s


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="sentence-transformers/nli-roberta-base-v2")
    ap.add_argument("--task", choices=["4way", "6way"], default="4way")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])

    # Layer options
    ap.add_argument("--layer", choices=["last", "avg_last4"], default="avg_last4")

    # Pooling options (token -> sentence)
    # NEW: Added wmean_dm, wmean_dm_count, wmean_dm_pos
    ap.add_argument("--poolings", nargs="+", default=["mean", "wmean_dm"],
                    choices=["cls", "mean", "wmean_pos", "wmean_pos_rev", "wmean_biperiphery",
                             "wmean_dm", "wmean_dm_count", "wmean_dm_pos"])

    # Discourse marker options
    ap.add_argument("--dm_weight", type=float, default=2.0,
                    help="Weight multiplier for DM tokens/sentences")
    ap.add_argument("--dm_mode", choices=["binary", "count"], default="count",
                    help="Sentence-level DM weighting: binary (has/not) or count (proportional)")

    # Other
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--cpu", action="store_true")
    return ap


def main():
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[INFO] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModel.from_pretrained(args.model_name).eval().to(device)
    print(f"[INFO] Loaded model: {args.model_name}")

    # Set paths based on task
    if args.task == "4way":
        data_dir = DATA_DIR_4WAY
        out_root = OUT_ROOT_4WAY
    else:
        data_dir = DATA_DIR_6WAY
        out_root = OUT_ROOT_6WAY

    print(f"[INFO] Task: {args.task}")
    print(f"[INFO] DM weight: {args.dm_weight}, mode: {args.dm_mode}")
    print(f"[INFO] Discourse markers (Academic Framework):")
    print(f"       - Single-word: {len(ALL_DM_SINGLE)} markers")
    print(f"       - Multi-word: {len(ALL_DM_MULTI)} markers")
    print(f"       - Total unique: {len(DISCOURSE_MARKERS)} markers")
    print(f"[INFO] Poolings: {args.poolings}")

    for split in args.splits:
        # Try multiple CSV naming patterns
        csv_candidates = [
            f"{split}_{args.task}_unified.csv",
            f"{split}_unified.csv",
            f"{split}.csv",
        ]

        csv_path = None
        for cand in csv_candidates:
            p = data_dir / cand
            if p.exists():
                csv_path = p
                break

        if csv_path is None:
            print(f"[ERROR] Not found in {data_dir}, tried: {csv_candidates}")
            continue

        print(f"\n[LOAD] {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"       Rows: {len(df)}")

        # Get text and labels
        text_col = "text" if "text" in df.columns else "utterance"
        texts = df[text_col].astype(str).fillna("").tolist()

        # Try label_num first, then label
        if "label_num" in df.columns:
            labels = df["label_num"].astype(int).values
        elif "label" in df.columns:
            if df["label"].dtype == object:
                # String labels - map to int
                label_map = {
                    'neutral': 3, 'happy': 0, 'sad': 1, 'angry': 2,  # 4way
                    'excited': 4, 'frustrated': 5  # 6way additional
                }
                labels = df["label"].str.lower().map(label_map).fillna(-1).astype(int).values
            else:
                labels = df["label"].astype(int).values
        else:
            labels = np.zeros(len(df), dtype=int)

        per_pool_arrays = {p: [] for p in args.poolings}
        lengths = []
        dm_weights_all = []
        dm_counts_all = []

        print(f"[EMBED] Processing {len(texts)} utterances...")
        for idx, txt in enumerate(texts):
            if idx % 500 == 0:
                print(f"        {idx}/{len(texts)}")

            sents = simple_sent_split(txt)
            if not sents:
                sents = [txt.strip() if txt.strip() else "."]
            lengths.append(len(sents))

            # Compute DM weights for this utterance's sentences
            dm_w = compute_dm_weights(sents, dm_weight=args.dm_weight, mode=args.dm_mode)
            dm_weights_all.append(dm_w)

            # Also store DM counts for analysis
            dm_c = np.array([count_discourse_markers(s) for s in sents], dtype=np.int32)
            dm_counts_all.append(dm_c)

            # Encode sentences
            enc = tokenizer(sents, padding=True, truncation=True,
                          max_length=args.max_length, return_tensors="pt")
            enc = to_device(enc, device)

            with torch.no_grad():
                out = model(**enc, output_hidden_states=True, return_dict=True)
                last_h = combine_layers(out.hidden_states, args.layer)
                attn_mask = enc["attention_mask"]

                for p in args.poolings:
                    if p == "cls":
                        pooled = last_h[:, 0, :]
                    elif p == "mean":
                        pooled = masked_mean(last_h, attn_mask)
                    elif p == "wmean_pos":
                        w = pos_weights(attn_mask, mode="linear")
                        pooled = (last_h * w.unsqueeze(-1)).sum(1)
                    elif p == "wmean_pos_rev":
                        w = pos_weights(attn_mask, mode="linear_rev")
                        pooled = (last_h * w.unsqueeze(-1)).sum(1)
                    elif p == "wmean_biperiphery":
                        w = pos_weights(attn_mask, mode="biperiphery")
                        pooled = (last_h * w.unsqueeze(-1)).sum(1)
                    # NEW: Discourse marker weighted pooling
                    elif p == "wmean_dm":
                        w = dm_weights_token(enc["input_ids"], attn_mask, tokenizer,
                                            dm_boost=args.dm_weight)
                        pooled = (last_h * w.unsqueeze(-1)).sum(1)
                    elif p == "wmean_dm_count":
                        w = dm_weights_token_count(enc["input_ids"], attn_mask, tokenizer,
                                                   dm_boost=args.dm_weight)
                        pooled = (last_h * w.unsqueeze(-1)).sum(1)
                    elif p == "wmean_dm_pos":
                        # Combined: DM weight * position weight (biperiphery)
                        w_dm = dm_weights_token(enc["input_ids"], attn_mask, tokenizer,
                                               dm_boost=args.dm_weight)
                        w_pos = pos_weights(attn_mask, mode="biperiphery")
                        w = w_dm * w_pos
                        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)
                        pooled = (last_h * w.unsqueeze(-1)).sum(1)
                    else:
                        raise ValueError(p)
                    per_pool_arrays[p].append(pooled.detach().cpu().numpy().astype(np.float32))

        # Pad and save
        N = len(texts)
        S_max = max(lengths) if lengths else 1
        D = per_pool_arrays[args.poolings[0]][0].shape[1]

        for pool_name in args.poolings:
            out_dir = out_root / args.layer / pool_name
            out_dir.mkdir(parents=True, exist_ok=True)

            X = np.zeros((N, S_max, D), dtype=np.float32)
            dm_weights_padded = np.zeros((N, S_max), dtype=np.float32)
            dm_counts_padded = np.zeros((N, S_max), dtype=np.int32)

            for i, (arr, dm_w, dm_c, L) in enumerate(zip(
                    per_pool_arrays[pool_name], dm_weights_all, dm_counts_all, lengths)):
                X[i, :L, :] = arr
                dm_weights_padded[i, :L] = dm_w
                dm_counts_padded[i, :L] = dm_c

            out_path = out_dir / f"{split}.npz"
            np.savez_compressed(
                out_path,
                X=X,
                y=labels,
                lengths=np.array(lengths, dtype=np.int32),
                dm_weights=dm_weights_padded,
                dm_counts=dm_counts_padded
            )
            print(f"[SAVE] {split} -> {out_path}")
            print(f"       X.shape={X.shape}, dm_weights.shape={dm_weights_padded.shape}")

            # Print DM statistics
            total_sents = sum(lengths)
            sents_with_dm = sum(1 for i, L in enumerate(lengths)
                               for j in range(L) if dm_counts_padded[i, j] > 0)
            print(f"       DM stats: {sents_with_dm}/{total_sents} sentences have DM ({100*sents_with_dm/total_sents:.1f}%)")


if __name__ == "__main__":
    main()
