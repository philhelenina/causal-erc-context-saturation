#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_senticnet_sentence_level_4way.py
- 발화(여러 문장) → 문장별 SenticNet features → (N, S_max, 4) + lengths
- 각 문장의 단어를 SenticNet dictionary에서 lookup하여 평균
- BERT hierarchical 구조와 동일한 방식으로 문장 분리
- 저장: {OUT_ROOT}/senticnet-sentence-level/{split}.npz
"""
import os
import argparse
import re
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm

HOME = Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()
DATA_DIR_DEFAULT = HOME / "data" / "iemocap_4way_data"
OUT_ROOT_DEFAULT = HOME / "data" / "embeddings" / "4way" / "senticnet-sentence-level"
SENTICNET_FILE = HOME / "src" / "features" / "senticnet" / "senticnet.xlsx"  # Excel 파일 사용

def simple_sent_split(text: str) -> List[str]:
    """BERT와 동일한 문장 분리 방식"""
    parts = re.split(r'(?<=[\.!\?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def simple_tokenize(text: str) -> List[str]:
    """간단한 단어 분리 (소문자 변환 + 구두점 제거)"""
    # 소문자 변환
    text = text.lower()
    # 구두점을 공백으로 대체
    text = re.sub(r'[^\w\s]', ' ', text)
    # 단어 분리
    words = text.split()
    return words

def load_senticnet_dict(senticnet_file: Path = None) -> Dict[str, List[float]]:
    """
    SenticNet dictionary 로드
    
    Returns:
        dict: {word: [introspection, temper, attitude, sensitivity]}
    
    Supports:
        - Excel file (.xlsx) with columns: CONCEPT, INTROSPECTION, TEMPER, ATTITUDE, SENSITIVITY
        - CSV file (.csv) with same columns
        - Python file (.py) with senticnet dict
    """
    processed_dict = {}
    
    if not senticnet_file or not senticnet_file.exists():
        print(f"[ERROR] SenticNet file not found: {senticnet_file}")
        print(f"[ERROR] Please provide correct path to SenticNet file.")
        return processed_dict
    
    # Load as Excel or CSV
    if senticnet_file.suffix in ['.xlsx', '.xls', '.csv']:
        print(f"[INFO] Loading SenticNet from {senticnet_file.suffix.upper()}: {senticnet_file}")
        
        try:
            if senticnet_file.suffix == '.csv':
                df = pd.read_csv(senticnet_file)
            else:
                df = pd.read_excel(senticnet_file)
            
            print(f"[INFO] Loaded {len(df)} rows from file")
            print(f"[INFO] Columns: {df.columns.tolist()}")
            
            # Expected columns (case-insensitive)
            col_map = {}
            for col in df.columns:
                col_lower = col.lower().strip()
                if 'concept' in col_lower or col_lower == 'word':
                    col_map['concept'] = col
                elif 'introspec' in col_lower:
                    col_map['introspection'] = col
                elif 'temper' in col_lower:
                    col_map['temper'] = col
                elif 'attitude' in col_lower:
                    col_map['attitude'] = col
                elif 'sensitiv' in col_lower:
                    col_map['sensitivity'] = col
            
            print(f"[INFO] Mapped columns: {col_map}")
            
            if len(col_map) < 5:
                print(f"[ERROR] Missing required columns. Found: {col_map.keys()}")
                print(f"[ERROR] Required: concept, introspection, temper, attitude, sensitivity")
                return processed_dict
            
            # Load data
            valid_count = 0
            for _, row in df.iterrows():
                concept = str(row[col_map['concept']]).lower().strip()
                
                # Skip empty concepts
                if not concept or concept == 'nan':
                    continue
                
                try:
                    values = [
                        float(row[col_map['introspection']]),
                        float(row[col_map['temper']]),
                        float(row[col_map['attitude']]),
                        float(row[col_map['sensitivity']])
                    ]
                    processed_dict[concept] = values
                    valid_count += 1
                except (ValueError, KeyError, TypeError) as e:
                    # Skip rows with invalid values
                    continue
            
            print(f"[INFO] Successfully loaded {valid_count} valid entries")
            
        except Exception as e:
            print(f"[ERROR] Failed to load file: {e}")
            return processed_dict
        
    else:
        # Try loading as Python file
        print(f"[INFO] Attempting to load SenticNet from Python file: {senticnet_file}")
        import sys
        sys.path.insert(0, str(senticnet_file.parent))
        try:
            # Import the module
            module_name = senticnet_file.stem
            import importlib.util
            spec = importlib.util.spec_from_file_location(module_name, senticnet_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            sentic_dict = module.senticnet
            print(f"[INFO] Loaded SenticNet dictionary from Python file")
            
            # Extract only numeric values (first 4 elements)
            for word, values in sentic_dict.items():
                if isinstance(values, (list, tuple)) and len(values) >= 4:
                    try:
                        # First 4 values: [introspection, temper, attitude, sensitivity]
                        processed_dict[word.lower()] = [float(values[i]) for i in range(4)]
                    except (ValueError, TypeError):
                        # Skip invalid entries
                        continue
            
            print(f"[INFO] Processed {len(processed_dict)} words from Python dict")
            
        except Exception as e:
            print(f"[ERROR] Cannot load from Python file: {e}")
            return processed_dict
    
    return processed_dict

def get_sentence_senticnet(sentence: str, senticnet_dict: Dict[str, List[float]]) -> np.ndarray:
    """
    한 문장의 SenticNet 표현 계산
    
    Args:
        sentence: 입력 문장
        senticnet_dict: SenticNet dictionary
    
    Returns:
        [4] array: [introspection, temper, attitude, sensitivity]
    """
    words = simple_tokenize(sentence)
    
    sentic_values = []
    for word in words:
        if word in senticnet_dict:
            sentic_values.append(senticnet_dict[word])
    
    if len(sentic_values) > 0:
        # 문장 내 모든 단어의 평균
        return np.mean(sentic_values, axis=0).astype(np.float32)
    else:
        # 사전에 없는 단어만 있으면 0
        return np.zeros(4, dtype=np.float32)

def process_utterance(utterance: str, senticnet_dict: Dict[str, List[float]]) -> np.ndarray:
    """
    Utterance → Sentence-level SenticNet features
    
    Returns:
        (S, 4) array: S개 문장, 각각 4-dim
    """
    # 1. 문장 분리 (BERT와 동일!)
    sentences = simple_sent_split(utterance)
    
    if not sentences:
        sentences = [utterance.strip()]
    
    # 2. 각 문장 처리
    sentic_per_sentence = []
    for sent in sentences:
        sent_sentic = get_sentence_senticnet(sent, senticnet_dict)
        sentic_per_sentence.append(sent_sentic)
    
    return np.array(sentic_per_sentence, dtype=np.float32)  # (S, 4)

def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(DATA_DIR_DEFAULT))
    ap.add_argument("--out_root", default=str(OUT_ROOT_DEFAULT))
    ap.add_argument("--senticnet_file", default=str(SENTICNET_FILE),
                   help="Path to senticnet.py or CSV file")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--text_col", default=None)
    return ap

def main():
    args = build_parser().parse_args()
    
    # Load SenticNet dictionary
    print("="*80)
    print("SENTICNET SENTENCE-LEVEL GENERATION (4-way)")
    print("="*80)
    
    senticnet_dict = load_senticnet_dict(Path(args.senticnet_file))
    
    if len(senticnet_dict) == 0:
        print("\n[ERROR] SenticNet dictionary is empty!")
        print("[ERROR] Please provide correct path to SenticNet file.")
        print("[ERROR] Expected format: dict with word -> [introspection, temper, attitude, sensitivity]")
        return
    
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    
    def find_text_column(df: pd.DataFrame) -> str:
        cands = ["text", "utterance", "utt", "sentence", "transcript"]
        for c in cands:
            if c in df.columns:
                return c
        for c in df.columns:
            if df[c].dtype == object:
                return c
        raise ValueError(f"text column not found. cols={df.columns[:10].tolist()}")
    
    for split in args.splits:
        print(f"\n{'='*80}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*80}")
        
        csv_file = Path(args.data_dir) / f"{split}_4way_unified.csv"
        df = pd.read_csv(csv_file)
        col = args.text_col or find_text_column(df)
        texts = df[col].astype(str).fillna("").tolist()
        
        # Extract utterance IDs for matching
        if 'utterance_id' in df.columns:
            utterance_ids = df['utterance_id'].astype(str).tolist()
        elif 'utt_id' in df.columns:
            utterance_ids = df['utt_id'].astype(str).tolist()
        elif 'id' in df.columns:
            utterance_ids = df['id'].astype(str).tolist()
        elif 'dialog_id' in df.columns and 'utterance_num' in df.columns:
            utterance_ids = (df['dialog_id'].astype(str) + '_' + df['utterance_num'].astype(str)).tolist()
        else:
            print(f"[WARNING] No utterance ID column found, using row indices")
            utterance_ids = [f"{split}_{i}" for i in range(len(df))]
        
        print(f"Total utterances: {len(texts)}")
        
        all_sentic = []
        lengths = []
        
        # Process each utterance
        for txt in tqdm(texts, desc=f"Processing {split}"):
            sentic_matrix = process_utterance(txt, senticnet_dict)  # (S_i, 4)
            all_sentic.append(sentic_matrix)
            lengths.append(len(sentic_matrix))
        
        # Padding
        N = len(texts)
        S_max = max(lengths) if lengths else 1
        
        print(f"\nPadding to S_max = {S_max}")
        
        padded_sentic = np.zeros((N, S_max, 4), dtype=np.float32)
        for i, sentic in enumerate(all_sentic):
            S_i = len(sentic)
            padded_sentic[i, :S_i, :] = sentic
        
        # Save
        out_file = out_root / f"{split}.npz"
        np.savez_compressed(
            out_file,
            embeddings=padded_sentic,
            lengths=np.array(lengths, dtype=np.int32),
            utterance_ids=np.asarray(utterance_ids, dtype=object)
        )
        
        print(f"[SAVE] {out_file}")
        print(f"       Shape: {padded_sentic.shape}")
        print(f"       Lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.2f}")
        print(f"       IDs: {len(utterance_ids)}")
        
        # Statistics
        non_zero_ratio = (padded_sentic != 0).any(axis=2).sum() / (N * S_max) * 100
        print(f"       Non-zero sentences: {non_zero_ratio:.1f}%")
    
    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)
    print(f"Output directory: {out_root}")
    print(f"Files created: {', '.join([f'{s}.npz' for s in args.splits])}")

if __name__ == "__main__":
    main()