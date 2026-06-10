#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_sroberta_npz_6way.py (FIXED - multiple poolings support)
- IEMOCAP 6-way 전용 (단일 발화 → 2D [N, 768])
- 레이어: last / avg_last4 / last4_scalar_up/down/top2
- 토큰풀링: cls / mean / attn(τ) / wmean_pos / wmean_pos_rev / wmean_idf
- 입력 CSV: data/iemocap_6way_data/{train,val,test}_6way_unified.csv
- 출력 NPZ: data/embeddings/6way/<out_root>/<layer>/<pool>/{split}.npz
"""
import os
import argparse, math
from pathlib import Path
from typing import List, Dict
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel

HOME = Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()
DATA6_DEFAULT = HOME / "data" / "iemocap_6way_data"
OUT6_DEFAULT  = HOME / "data" / "embeddings" / "6way" / "bert-base"
IDF_CSV = HOME / "src" / "features" / "idf.csv"

PRESETS = {
    "last4_scalar":      [1,1,1,1],
    "last4_scalar_up":   [1,2,3,4],
    "last4_scalar_down": [4,3,2,1],
    "last4_scalar_top2": [0,1,1,0],
}

def find_text_column(df: pd.DataFrame) -> str:
    cands = ["utterance", "text", "utt", "sentence", "transcript"]
    for c in cands:
        if c in df.columns:
            return c
    for col in df.columns:
        if df[col].dtype == object:
            return col
    raise ValueError(f"텍스트 컬럼을 찾지 못했습니다. cols={df.columns.tolist()[:10]}")

def load_idf_map(p: Path|None):
    if not p or not p.exists(): return None
    df = pd.read_csv(p)
    return {str(t).lower(): float(i) for t,i in zip(df["token"], df["idf"])}

def safe_mean(x, m):
    m = m.float()
    s = m.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return (x * m.unsqueeze(-1)).sum(dim=1) / s

def attn_pool(x, m, tau=1.0):
    cls = x[:,0,:]
    scr = (x * cls.unsqueeze(1)).sum(-1) / math.sqrt(x.size(-1))
    scr = scr.masked_fill(m==0, -1e9)
    w = torch.softmax(scr / max(tau,1e-6), 1)
    return (x * w.unsqueeze(-1)).sum(1)

def pos_weights(
    m: torch.Tensor, 
    mode: str = "linear",  # "linear", "linear_rev", "exp_decay"
    tau: float = 5.0
) -> torch.Tensor:
    """
    위치 기반 가중치 생성
    
    Args:
        m: (B, T) attention mask
        mode: "linear" (1..T), "linear_rev" (T..1), "exp_decay" (e^(-(t-1)/τ))
        tau: exponential decay constant
    """
    B, T = m.shape
    device = m.device
    
    if mode == "linear":
        w_raw = torch.arange(1, T+1, device=device).float()
    elif mode == "linear_rev":
        w_raw = torch.arange(T, 0, -1, device=device).float()
    elif mode == "exp_decay":
        t = torch.arange(1, T+1, device=device).float()
        w_raw = torch.exp(-(t - 1) / tau)
    elif mode == "biperiphery":
        # U-shape: 양쪽 끝이 높고 중간이 낮음
        t = torch.arange(1, T+1, device=device).float()
        w_raw = torch.maximum(t, T - t + 1)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    w = w_raw.unsqueeze(0).repeat(B,1) * m.float()
    s = w.sum(1, keepdim=True).clamp_min(1e-6)
    return w/s

def idf_weights(token_ids, m, tokenizer, idf_map):
    if idf_map is None: 
        return pos_weights(m, False) * 0 + (m.float()/m.sum(1,keepdim=True).clamp_min(1e-6))
    B,T = token_ids.shape
    W = torch.ones((B,T), device=token_ids.device)
    for b in range(B):
        toks = tokenizer.convert_ids_to_tokens(token_ids[b].tolist())
        row=[]
        for t,mask in zip(toks, m[b].tolist()):
            if mask==0: row.append(0.0); continue
            t=t.replace("Ġ","").lower()
            row.append(1.0 + float(idf_map.get(t,0.0)))
        W[b]=torch.tensor(row, device=token_ids.device)
    s=(W*m.float()).sum(1,keepdim=True).clamp_min(1e-6)
    return (W*m.float())/s

def combine_layers(hs, mode, scalar_w=None):
    if mode=="last": return hs[-1]
    if mode=="avg_last4": return torch.stack(hs[-4:],0).mean(0)
    if mode.startswith("last4_scalar"):
        w = torch.tensor(scalar_w, device=hs[-1].device, dtype=torch.float32)
        w = w / (w.sum()+1e-8)
        H = torch.stack(hs[-4:],0)
        return (H * w.view(4,1,1,1)).sum(0)
    raise ValueError(mode)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--data_dir", default=str(DATA6_DEFAULT))
    ap.add_argument("--out_root", default=str(OUT6_DEFAULT))
    ap.add_argument("--splits", nargs="+", default=["train","val","test"])
    ap.add_argument("--text_col", default=None)
    
    # Layer
    ap.add_argument("--layer", 
                    choices=["last","avg_last4","last4_scalar",
                            "last4_scalar_up","last4_scalar_down","last4_scalar_top2"], 
                    default="avg_last4")
    ap.add_argument("--scalar_weights", default="")
    
    # Pooling - MULTIPLE SUPPORT!
    ap.add_argument("--poolings", nargs="+", 
                    default=["mean"],
                    choices=["cls","mean","attn",
                            "wmean_pos","wmean_pos_rev","wmean_biperiphery",
                            "wmean_exp_fast","wmean_exp_med","wmean_exp_slow",
                            "wmean_idf"])
    
    # Other
    ap.add_argument("--attn_tau", type=float, default=1.0)
    ap.add_argument("--idf_csv", type=str, default=str(IDF_CSV))
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    
    # Exponential decay constants
    ap.add_argument("--exp_tau_fast", type=float, default=2.0,
                   help="τ for fast decay (high-arousal emotions)")
    ap.add_argument("--exp_tau_med", type=float, default=5.0,
                   help="τ for medium decay")
    ap.add_argument("--exp_tau_slow", type=float, default=10.0,
                   help="τ for slow decay (low-arousal emotions)")
    
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[INFO] device = {device}")
    
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    mdl = AutoModel.from_pretrained(args.model_name).eval().to(device)

    # Layer mode
    layer_mode = args.layer
    scalar_w = None
    if layer_mode.startswith("last4_scalar"):
        if args.scalar_weights:
            scalar_w = [float(x) for x in args.scalar_weights.split(",")]
        else:
            scalar_w = PRESETS[layer_mode]
            layer_mode = "last4_scalar"

    # IDF map
    idf_map = None
    if any("idf" in p for p in args.poolings):
        idf_map = load_idf_map(Path(args.idf_csv))
        if idf_map:
            print(f"[INFO] IDF map loaded: {len(idf_map)} entries")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        csv_path = Path(args.data_dir) / f"{split}_6way_unified.csv"
        df = pd.read_csv(csv_path)
        text_col = args.text_col or find_text_column(df)
        texts = df[text_col].astype(str).fillna("").tolist()
        print(f"[{split}] rows={len(texts)} | text_col={text_col} | csv={csv_path}")

        # Buckets for each pooling
        buckets: Dict[str, List[np.ndarray]] = {p: [] for p in args.poolings}

        # Process batches
        for i in range(0, len(texts), args.batch_size):
            batch = texts[i:i+args.batch_size]
            enc = tok(batch, padding=True, truncation=True, 
                     max_length=args.max_length, return_tensors="pt").to(device)
            
            with torch.no_grad():
                o = mdl(**enc, output_hidden_states=True, return_dict=True)
                X = combine_layers(o.hidden_states, layer_mode, scalar_w)
                M = enc["attention_mask"]

                # Compute each pooling
                for pool in args.poolings:
                    if pool=="cls": 
                        pooled = X[:,0,:]
                    elif pool=="mean": 
                        pooled = safe_mean(X,M)
                    elif pool=="attn": 
                        pooled = attn_pool(X,M,args.attn_tau)
                    elif pool=="wmean_pos": 
                        pooled = (X * pos_weights(M, mode="linear").unsqueeze(-1)).sum(1)
                    elif pool=="wmean_pos_rev":
                        pooled = (X * pos_weights(M, mode="linear_rev").unsqueeze(-1)).sum(1)
                    elif pool=="wmean_biperiphery":
                        pooled = (X * pos_weights(M, mode="biperiphery").unsqueeze(-1)).sum(1)

                    # NEW: Exponential decay variants
                    elif pool=="wmean_exp_fast":
                        pooled = (X * pos_weights(M, mode="exp_decay", tau=args.exp_tau_fast).unsqueeze(-1)).sum(1)
                    elif pool=="wmean_exp_med":
                        pooled = (X * pos_weights(M, mode="exp_decay", tau=args.exp_tau_med).unsqueeze(-1)).sum(1)
                    elif pool=="wmean_exp_slow":
                        pooled = (X * pos_weights(M, mode="exp_decay", tau=args.exp_tau_slow).unsqueeze(-1)).sum(1)
                    
                    elif pool=="wmean_idf": 
                        pooled = (X * idf_weights(enc["input_ids"], M, tok, idf_map).unsqueeze(-1)).sum(1)
                    else: 
                        raise ValueError(f"Unknown pooling: {pool}")
                    
                    buckets[pool].append(pooled.detach().cpu().numpy().astype(np.float32))

        # Save each pooling
        for pool, chunks in buckets.items():
            arr = np.concatenate(chunks, axis=0) if len(chunks) > 0 \
                  else np.zeros((0, mdl.config.hidden_size), dtype=np.float32)
            
            save_dir = out_root / layer_mode / pool
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / f"{split}.npz"
            
            np.savez_compressed(out_path, embeddings=arr)
            print(f"[SAVE] {layer_mode}/{pool}/{split} -> {out_path}  shape={arr.shape}")

    print("[DONE] all splits saved.")

if __name__ == "__main__":
    main()