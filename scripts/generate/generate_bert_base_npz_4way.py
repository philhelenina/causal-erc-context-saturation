#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last updated on 11/13
generate_sroberta_npz.py
- Sentence-RoBERTa(NLI-RoBERTa-base, 768d) 임베딩을 생성하여 NPZ로 저장.
- 저장 경로: {OUT_ROOT}/sentence-roberta/{layer}/{pool}/{train|val|test}.npz
  (NPZ 키는 'embeddings')

[레이어 결합 옵션 --layer]
  - last           : 마지막 레이어만 사용
  - avg_last4      : 마지막 4개 레이어 평균
  - last4_scalar   : 마지막 4개 레이어 가중합( --scalar_weights 'a,b,c,d' 필요 )
  - last4_scalar_up     : 프리셋(얕은→깊은 점증)   [0.1,0.2,0.3,0.4]
  - last4_scalar_down   : 프리셋(깊은→얕은 점감)   [0.4,0.3,0.2,0.1]
  - last4_scalar_top2   : 프리셋(마지막 2개만 강조) [0.0,0.0,0.5,0.5]

[토큰 풀링 --poolings] (복수 지정 가능)
  - cls            : 첫 토큰(CLS) 벡터
  - mean           : 마스킹 평균
  - attn           : CLS-쿼리 점수 기반 비학습 어텐션(temperature=--attn_tau)
  - wmean_pos      : 앞쪽 토큰 가중↑  (w_t ∝ t)
  - wmean_pos_rev  : 뒤쪽 토큰 가중↑  (w_t ∝ T-t+1)
  - wmean_idf      : 토큰별 (1+idf) 가중 평균  ※ --idf_csv 필요

입력 CSV: {DATA_DIR}/{split}_4way_unified.csv
텍스트 컬럼 탐색 순서: --text_col 지정 없으면 ["text","utterance","utt","sentence","transcript"] → 그 외 object dtype 후보
"""
import os

import argparse, math
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel


# ------------------- 기본 경로 -------------------
HOME = Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()
DATA_DIR_DEFAULT = HOME / "data" / "iemocap_4way_data"
OUT_ROOT_DEFAULT = HOME / "data" / "embeddings" / "4way" / "bert-base"
IDF_DEFAULT = HOME / "src" / "features" / "idf.csv"

# ------------------- 유틸 -------------------
def find_text_column(df: pd.DataFrame, prefer: str = "text") -> str:
    cands = [prefer, "utterance", "utt", "sentence", "transcript"]
    for c in cands:
        if c in df.columns:
            return c
    for col in df.columns:
        if df[col].dtype == object:
            return col
    raise ValueError(f"텍스트 컬럼을 찾지 못했습니다. 후보={cands}, cols={df.columns.tolist()[:10]}")

def to_device(d, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}

def masked_mean(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden: (B,T,H), attn_mask: (B,T)
    mask = attn_mask.float()
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return (last_hidden * mask.unsqueeze(-1)).sum(dim=1) / denom

def attn_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    # 간단한 CLS-쿼리 점수로 softmax 가중합
    cls = last_hidden[:, 0, :]  # (B,H)
    scores = (last_hidden * cls.unsqueeze(1)).sum(dim=-1) / math.sqrt(last_hidden.size(-1))  # (B,T)
    scores = scores.masked_fill(attn_mask == 0, -1e9)
    w = torch.softmax(scores / max(tau, 1e-6), dim=1)  # (B,T)
    return (last_hidden * w.unsqueeze(-1)).sum(dim=1)

def pos_weights(
    attn_mask: torch.Tensor,
    mode: str = "linear",  # "linear", "linear_rev", "exp_decay", "biperiphery"
    tau: float = 5.0
) -> torch.Tensor:
    """
    위치 기반 가중치 생성

    Args:
        attn_mask: (B, T) attention mask
        mode: "linear" (1..T), "linear_rev" (T..1), "exp_decay", "biperiphery" (U-shape)
        tau: exponential decay constant (작을수록 빠른 붕괴)
    """
    B, T = attn_mask.shape
    device = attn_mask.device

    if mode == "linear":
        # 1..T (뒤쪽 가중↑)
        w_raw = torch.arange(1, T + 1, device=device, dtype=torch.float32)
    elif mode == "linear_rev":
        # T..1 (앞쪽 가중↑)
        w_raw = torch.arange(T, 0, -1, device=device, dtype=torch.float32)
    elif mode == "exp_decay":
        # Exponential decay: w_t = e^(-(t-1)/τ)
        t = torch.arange(1, T + 1, device=device, dtype=torch.float32)
        w_raw = torch.exp(-(t - 1) / tau)
    elif mode == "biperiphery":
        # U-shape: 양쪽 끝이 높고 중간이 낮음
        # w_t = max(t, T-t+1) → 시작과 끝 모두 가중
        t = torch.arange(1, T + 1, device=device, dtype=torch.float32)
        w_raw = torch.maximum(t, T - t + 1)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Apply mask and normalize
    w = w_raw.unsqueeze(0).repeat(B, 1) * attn_mask.float()
    denom = w.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return w / denom

def idf_weights(token_ids: torch.Tensor, attn_mask: torch.Tensor, tokenizer, idf_map: Dict[str, float]) -> torch.Tensor:
    """
    token_ids: (B,T), attn_mask: (B,T), idf_map: {"token": idf}
    RoBERTa 토큰의 'Ġ' 제거 후 소문자 매칭. 가중치는 (1 + idf(token)).
    """
    B, T = token_ids.shape
    weights = torch.ones((B, T), dtype=torch.float32, device=token_ids.device)
    for b in range(B):
        ids = token_ids[b].tolist()
        msk = attn_mask[b].tolist()
        toks = tokenizer.convert_ids_to_tokens(ids)
        row = []
        for t, m in zip(toks, msk):
            if m == 0:
                row.append(0.0); continue
            t_clean = t.replace("Ġ", "").lower()
            row.append(1.0 + float(idf_map.get(t_clean, 0.0)))
        weights[b] = torch.tensor(row, device=token_ids.device, dtype=torch.float32)
    denom = (weights * attn_mask.float()).sum(dim=1, keepdim=True).clamp_min(1e-6)
    return (weights * attn_mask.float()) / denom

def combine_layers(hidden_states: Tuple[torch.Tensor, ...],
                   mode: str = "avg_last4",
                   scalar_weights: Optional[List[float]] = None) -> torch.Tensor:
    """
    hidden_states: tuple(len=L+1), hidden_states[-1]가 마지막 레이어 (B,T,H)
    returns: (B,T,H)
    """
    if mode == "last":
        return hidden_states[-1]
    elif mode == "avg_last4":
        hs = torch.stack(hidden_states[-4:], dim=0)  # (4,B,T,H)
        return hs.mean(dim=0)
    elif mode == "last4_scalar":
        if not scalar_weights or len(scalar_weights) != 4:
            raise ValueError("--scalar_weights 'a,b,c,d' 필요")
        w = torch.tensor(scalar_weights, dtype=torch.float32, device=hidden_states[-1].device)
        w = w / (w.sum() + 1e-8)
        hs = torch.stack(hidden_states[-4:], dim=0)
        return (hs * w.view(4, 1, 1, 1)).sum(dim=0)
    else:
        raise ValueError(f"Unknown layer mode: {mode}")


# ------------------- 데이터셋 -------------------
class TextDataset(Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx): return self.texts[idx]


# ------------------- 메인 루틴 -------------------
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[INFO] device = {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModel.from_pretrained(args.model_name)
    model.eval().to(device)

    # IDF 맵 (필요 시)
    idf_map = None
    if any("idf" in p for p in args.poolings):
        if args.idf_csv and Path(args.idf_csv).exists():
            df_idf = pd.read_csv(args.idf_csv)
            idf_map = {str(t).lower(): float(i) for t, i in zip(df_idf["token"], df_idf["idf"])}
            print(f"[INFO] IDF map loaded: {len(idf_map)} entries")
        else:
            raise FileNotFoundError(f"--idf_csv 경로를 찾을 수 없습니다: {args.idf_csv}")

    # 레이어 모드/프리셋 해석
    layer_mode = args.layer
    scalar_w = None
    if layer_mode == "last4_scalar":
        scalar_w = [float(x) for x in args.scalar_weights.split(",")]
    elif layer_mode == "last4_scalar_up":
        scalar_w = [0.1, 0.2, 0.3, 0.4]; layer_mode = "last4_scalar"
    elif layer_mode == "last4_scalar_down":
        scalar_w = [0.4, 0.3, 0.2, 0.1]; layer_mode = "last4_scalar"
    elif layer_mode == "last4_scalar_top2":
        scalar_w = [0.0, 0.0, 0.5, 0.5]; layer_mode = "last4_scalar"

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        csv_path = Path(args.data_dir) / f"{split}_4way_unified.csv"
        df = pd.read_csv(csv_path)
        text_col = args.text_col or find_text_column(df)
        texts = df[text_col].astype(str).fillna("").tolist()
        print(f"[{split}] rows={len(texts)} | text_col={text_col} | csv={csv_path}")

        ds = TextDataset(texts)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

        buckets: Dict[str, List[np.ndarray]] = {p: [] for p in args.poolings}

        with torch.no_grad():
            for batch_texts in dl:
                enc = tokenizer(
                    list(batch_texts),
                    padding=True, truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt"
                )
                enc = to_device(enc, device)
                out = model(**enc, output_hidden_states=True, return_dict=True)

                # (B,T,H) after layer combination
                last_hidden = combine_layers(out.hidden_states, layer_mode, scalar_w)

                # 공통 마스크
                attn_mask = enc["attention_mask"]

                # 풀링별 계산
                for pooling in args.poolings:
                    if pooling == "cls":
                        pooled = last_hidden[:, 0, :]
                    elif pooling == "mean":
                        pooled = masked_mean(last_hidden, attn_mask)
                    elif pooling == "attn":
                        pooled = attn_pool(last_hidden, attn_mask, tau=args.attn_tau)
                    elif pooling == "wmean_pos":
                        w = pos_weights(attn_mask, mode="linear")
                        pooled = (last_hidden * w.unsqueeze(-1)).sum(dim=1)
                    elif pooling == "wmean_pos_rev":
                        w = pos_weights(attn_mask, mode="linear_rev")
                        pooled = (last_hidden * w.unsqueeze(-1)).sum(dim=1)
                    elif pooling == "wmean_biperiphery":
                        w = pos_weights(attn_mask, mode="biperiphery")
                        pooled = (last_hidden * w.unsqueeze(-1)).sum(dim=1)

                    # NEW: Exponential decay variants
                    elif pooling == "wmean_exp_fast":
                        w = pos_weights(attn_mask, mode="exp_decay", tau=args.exp_tau_fast)
                        pooled = (last_hidden * w.unsqueeze(-1)).sum(dim=1)
                    elif pooling == "wmean_exp_med":
                        w = pos_weights(attn_mask, mode="exp_decay", tau=args.exp_tau_med)
                        pooled = (last_hidden * w.unsqueeze(-1)).sum(dim=1)
                    elif pooling == "wmean_exp_slow":
                        w = pos_weights(attn_mask, mode="exp_decay", tau=args.exp_tau_slow)
                        pooled = (last_hidden * w.unsqueeze(-1)).sum(dim=1)
                    
                    elif pooling == "wmean_idf":
                        if idf_map is None:
                            raise RuntimeError("idf_map is None (idf_csv 필요)")
                        w = idf_weights(enc["input_ids"], attn_mask, tokenizer, idf_map)
                        pooled = (last_hidden * w.unsqueeze(-1)).sum(dim=1)
                    else:
                        raise ValueError(f"Unknown pooling: {pooling}")

                    buckets[pooling].append(pooled.detach().cpu().numpy().astype(np.float32))

        # 저장: sentence-roberta/{layer}/{pool}/{split}.npz
        for pooling, chunks in buckets.items():
            arr = np.concatenate(chunks, axis=0) if len(chunks) > 0 \
                  else np.zeros((0, model.config.hidden_size), dtype=np.float32)
            save_dir = out_root / layer_mode / pooling
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / f"{split}.npz"
            np.savez_compressed(out_path, embeddings=arr)
            print(f"[SAVE] {layer_mode}/{pooling}/{split} -> {out_path}  shape={arr.shape}")

    print("[DONE] all splits saved.")


# ------------------- CLI -------------------
def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="bert-base-uncased")
    ap.add_argument("--data_dir", type=str, default=str(DATA_DIR_DEFAULT))
    ap.add_argument("--out_root", type=str, default=str(OUT_ROOT_DEFAULT))
    ap.add_argument("--splits", nargs="+", default=["train","val","test"])
    ap.add_argument("--text_col", type=str, default=None)

    # 레이어 결합
    ap.add_argument("--layer",
        choices=[
            "last","avg_last4","last4_scalar",
            "last4_scalar_up","last4_scalar_down","last4_scalar_top2"
        ],
        default="avg_last4"
    )
    ap.add_argument("--scalar_weights", type=str, default="1,1,1,1",
                    help="--layer last4_scalar 일 때 'a,b,c,d' (합은 자동 정규화)")

    # 풀링(복수 가능)
    ap.add_argument("--poolings", nargs="+",
                    default=["cls","mean"],
                    choices=["cls","mean","attn",
                            "wmean_pos","wmean_pos_rev","wmean_biperiphery",
                            "wmean_exp_fast","wmean_exp_med","wmean_exp_slow",
                            "wmean_idf"])

    # IDF
    ap.add_argument("--idf_csv", type=str, default=str(IDF_DEFAULT))

    # 토크나이즈 / 배치
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--attn_tau", type=float, default=1.0)
    
    # Exponential decay constants
    ap.add_argument("--exp_tau_fast", type=float, default=2.0,
                   help="τ for fast decay (high-arousal emotions)")
    ap.add_argument("--exp_tau_med", type=float, default=5.0,
                   help="τ for medium decay")
    ap.add_argument("--exp_tau_slow", type=float, default=10.0,
                   help="τ for slow decay (low-arousal emotions)")

    # 디바이스
    ap.add_argument("--cpu", action="store_true")
    return ap


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)