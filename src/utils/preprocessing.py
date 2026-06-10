"""
Unified preprocessing pipeline for SenticCrystal.

Standardizes IEMOCAP (4way/6way) and MELD datasets into consistent format.
Supports both 4-label and 6-label emotion classification.
"""

import os
import pandas as pd
import glob
import re
import argparse
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Standard emotion label mappings
EMOTION_MAPPINGS = {
    'iemocap': {
        'original': ['ang', 'hap', 'sad', 'neu', 'fru', 'exc', 'sur', 'dis', 'fea', 'oth', 'xxx'],
        '4way': ['ang', 'hap', 'sad', 'neu'],
        '6way': ['ang', 'hap', 'sad', 'neu', 'exc', 'fru'],
        'label_map': {
            'ang': 'angry', 'hap': 'happy', 'sad': 'sad', 'neu': 'neutral',
            'exc': 'excited', 'fru': 'frustrated', 'sur': 'surprised', 
            'dis': 'disgusted', 'fea': 'fearful', 'oth': 'other', 'xxx': 'undefined'
        }
    },
    'meld': {
        'original': ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger'],
        '4way': ['anger', 'joy', 'sadness', 'neutral'],
        '6way': ['anger', 'joy', 'sadness', 'neutral', 'fear', 'surprise'],
        'label_map': {
            'anger': 'angry', 'joy': 'happy', 'sadness': 'sad', 'neutral': 'neutral',
            'fear': 'fearful', 'surprise': 'surprised', 'disgust': 'disgusted'
        }
    }
}

# Unified emotion labels (target format)
UNIFIED_LABELS = {
    '4way': ['angry', 'happy', 'sad', 'neutral'],
    '6way': ['angry', 'happy', 'sad', 'neutral', 'fearful', 'surprised']
}

class DatasetProcessor:
    """Base class for dataset processing."""
    
    def __init__(self, data_path: str, output_path: str):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def create_unified_format(
        self, 
        df: pd.DataFrame, 
        dataset_name: str,
        classification_type: str = '4way'
    ) -> pd.DataFrame:
        """
        Convert dataset to unified SenticCrystal format.
        
        Standard columns:
        - id: Unique utterance identifier
        - utterance: Text content
        - label: Emotion label (unified format)
        - label_num: Numeric label (0-indexed)
        - file_id: Dialogue/conversation identifier
        - utterance_num: Position within dialogue
        - session: Session/split identifier
        - start: Start timestamp (if available)
        - end: End timestamp (if available)
        """
        required_columns = ['id', 'utterance', 'label', 'label_num', 'file_id', 'utterance_num']
        
        # Ensure all required columns exist
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0 if col in ['label_num', 'utterance_num'] else ''
        
        # Standardize label format
        df = self._standardize_labels(df, dataset_name, classification_type)
        
        # Sort by file_id and utterance_num for consistent ordering
        df = df.sort_values(['file_id', 'utterance_num']).reset_index(drop=True)
        
        return df
    
    def _standardize_labels(
        self, 
        df: pd.DataFrame, 
        dataset_name: str, 
        classification_type: str
    ) -> pd.DataFrame:
        """Standardize emotion labels to unified format."""
        mappings = EMOTION_MAPPINGS[dataset_name]
        target_labels = UNIFIED_LABELS[classification_type]
        
        # Create label mapping dictionary
        label_map = {}
        for orig_label, unified_label in mappings['label_map'].items():
            if unified_label in target_labels:
                label_map[orig_label] = unified_label
        
        # Apply label mapping
        df['label'] = df['label'].map(label_map).fillna('undefined')
        
        # Filter to target labels or mark as undefined
        df.loc[~df['label'].isin(target_labels), 'label'] = 'undefined'
        
        # Create numeric mapping
        label_to_num = {label: i for i, label in enumerate(target_labels)}
        label_to_num['undefined'] = -1
        
        df['label_num'] = df['label'].map(label_to_num)
        
        return df

class IEMOCAPProcessor(DatasetProcessor):
    """IEMOCAP dataset processor."""
    
    def __init__(self, data_path: str, output_path: str):
        super().__init__(data_path, output_path)
        
        # Session splits
        self.train_sessions = ['Session1', 'Session2', 'Session3']
        self.val_sessions = ['Session4']
        self.test_sessions = ['Session5']
    
    def process_raw_iemocap(self, iemocap_raw_path: str) -> Dict[str, pd.DataFrame]:
        """
        Process raw IEMOCAP data from original format.
        
        Args:
            iemocap_raw_path: Path to IEMOCAP_full_release directory
            
        Returns:
            Dict with train/val/test DataFrames
        """
        datasets = {}
        
        for split, sessions in [
            ('train', self.train_sessions),
            ('val', self.val_sessions), 
            ('test', self.test_sessions)
        ]:
            all_data = []
            
            for session in sessions:
                session_data = self._process_session(iemocap_raw_path, session)
                all_data.extend(session_data)
            
            if all_data:
                df = pd.DataFrame(all_data)
                datasets[split] = self._postprocess_iemocap_df(df)
                logger.info(f"Processed {split}: {len(df)} utterances")
        
        return datasets
    
    def _process_session(self, iemocap_path: str, session_name: str) -> List[Dict]:
        """Process single IEMOCAP session."""
        session_path = Path(iemocap_path) / session_name / 'dialog'
        emo_files = glob.glob(str(session_path / 'EmoEvaluation' / '*.txt'))
        trans_files = glob.glob(str(session_path / 'transcriptions' / '*.txt'))
        
        all_data = []
        
        for trans_file in trans_files:
            file_name = os.path.basename(trans_file)
            emo_file = session_path / 'EmoEvaluation' / file_name
            
            if not emo_file.exists():
                continue
            
            emo_dict = self._parse_emo_file(str(emo_file))
            
            with open(trans_file, 'r') as f:
                lines = f.readlines()
            
            utterance_num = 0
            for line in lines:
                processed = self._process_line(line)
                if processed:
                    emotion = emo_dict.get(processed['id'], 'xxx')
                    all_data.append({
                        'session': session_name,
                        'utterance_num': utterance_num,
                        'id': processed['id'],
                        'start': processed['start'],
                        'end': processed['end'],
                        'utterance': processed['utterance'],
                        'label': emotion if emotion != 'xxx' else 'undefined'
                    })
                    utterance_num += 1
        
        return all_data
    
    def _parse_emo_file(self, emo_file: str) -> Dict[str, str]:
        """Parse IEMOCAP emotion file."""
        emo_dict = {}
        with open(emo_file, 'r') as f:
            for line in f:
                if line.startswith('['):
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        file_id = parts[1]
                        emotion = parts[2].lower()
                        emo_dict[file_id] = emotion
        return emo_dict
    
    def _process_line(self, line: str) -> Optional[Dict]:
        """Process transcription line."""
        match = re.match(r'(.*?)\\s*\\[(.*?)\\]:\\s*(.*)', line)
        if match:
            return {
                'id': match.group(1).strip(),
                'start': match.group(2).split('-')[0].strip(),
                'end': match.group(2).split('-')[1].strip(),
                'utterance': match.group(3).strip()
            }
        return None
    
    def _postprocess_iemocap_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Postprocess IEMOCAP DataFrame."""
        # Extract file_id
        df['file_id'] = df['id'].str.extract(r'(Ses\\d+[FM]_(?:impro|script)\\d+)')
        
        # Recalculate utterance numbers within each file
        df['utterance_num'] = df.groupby('file_id').cumcount()
        
        return df
    
    def process_existing_csvs(
        self, 
        classification_type: str = '4way',
        include_undefined: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Process existing IEMOCAP CSV files.
        
        Args:
            classification_type: '4way' or '6way'
            include_undefined: Whether to include undefined labels
            
        Returns:
            Dict with train/val/test DataFrames
        """
        datasets = {}
        suffix = '_with_minus_one' if include_undefined else ''
        
        for split in ['train', 'val', 'test']:
            csv_path = self.data_path / f"{classification_type}/{split}_{classification_type}{suffix}.csv"
            
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                
                # Handle undefined labels
                if not include_undefined:
                    df = df[df['label'] != '-1']
                
                datasets[split] = self.create_unified_format(df, 'iemocap', classification_type)
                logger.info(f"Loaded {split}: {len(df)} utterances")
            else:
                logger.warning(f"File not found: {csv_path}")
        
        return datasets

class MELDProcessor(DatasetProcessor):
    """MELD dataset processor."""
    
    def process_meld_csvs(self, classification_type: str = '4way') -> Dict[str, pd.DataFrame]:
        """
        Process MELD CSV files.
        
        Args:
            classification_type: '4way' or '6way'
            
        Returns:
            Dict with train/dev/test DataFrames
        """
        datasets = {}
        
        # MELD uses train/dev/test splits
        for split in ['train', 'dev', 'test']:
            csv_path = self.data_path / f"{split}.csv"
            
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                
                # Rename columns to match standard format
                column_mapping = {
                    'Utterance': 'utterance',
                    'Emotion': 'label',
                    'Dialogue_ID': 'file_id',
                    'Utterance_ID': 'utterance_num'
                }
                
                df = df.rename(columns=column_mapping)
                
                # Create ID column
                df['id'] = df['file_id'].astype(str) + '_' + df['utterance_num'].astype(str)
                
                # Add session information
                df['session'] = f"MELD_{split}"
                
                # Convert to unified format
                df = self.create_unified_format(df, 'meld', classification_type)
                
                # Map dev to val for consistency
                split_name = 'val' if split == 'dev' else split
                datasets[split_name] = df
                
                logger.info(f"Processed MELD {split}: {len(df)} utterances")
            else:
                logger.warning(f"MELD file not found: {csv_path}")
        
        return datasets

def save_datasets(
    datasets: Dict[str, pd.DataFrame],
    output_path: Path,
    dataset_name: str,
    classification_type: str
):
    """Save processed datasets to CSV files."""
    output_dir = output_path / dataset_name / classification_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split, df in datasets.items():
        # Save main dataset
        main_path = output_dir / f"{split}_{classification_type}_unified.csv"
        df.to_csv(main_path, index=False)
        
        # Save metadata (for backwards compatibility)
        meta_df = df[['id', 'utterance_num', 'label', 'label_num']].copy()
        meta_path = output_dir / f"{split}_{classification_type}_metadata.csv"
        meta_df.to_csv(meta_path, index=False)
        
        # Save utterances only
        utt_df = df[['id', 'utterance']].copy()
        utt_path = output_dir / f"{split}_{classification_type}_utterances.csv"
        utt_df.to_csv(utt_path, index=False)
        
        logger.info(f"Saved {split} to {main_path}")

def print_dataset_statistics(datasets: Dict[str, pd.DataFrame], classification_type: str):
    """Print dataset statistics."""
    target_labels = UNIFIED_LABELS[classification_type]
    
    for split, df in datasets.items():
        print(f"\\n{split.upper()} Statistics:")
        print(f"Total utterances: {len(df)}")
        
        label_counts = df['label'].value_counts()
        print("\\nLabel distribution:")
        
        for i, label in enumerate(target_labels):
            count = label_counts.get(label, 0)
            percentage = (count / len(df)) * 100 if len(df) > 0 else 0
            print(f"{label.capitalize()} ({i}): {count} ({percentage:.1f}%)")
        
        if 'undefined' in label_counts:
            count = label_counts['undefined']
            percentage = (count / len(df)) * 100
            print(f"Undefined (-1): {count} ({percentage:.1f}%)")

def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description="Unified preprocessing for SenticCrystal")
    parser.add_argument('--dataset', choices=['iemocap', 'meld', 'both'], default='both',
                       help='Dataset to process')
    parser.add_argument('--classification', choices=['4way', '6way', 'both'], default='both',
                       help='Classification type')
    parser.add_argument('--iemocap_path', type=str, 
                       default='data/preprocessed-iemocap',
                       help='Path to IEMOCAP data')
    parser.add_argument('--meld_path', type=str,
                       default='data/MELD',
                       help='Path to MELD data')
    parser.add_argument('--output_path', type=str,
                       default='data',
                       help='Output directory')
    parser.add_argument('--include_undefined', action='store_true',
                       help='Include undefined labels')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    output_path = Path(args.output_path)
    classification_types = ['4way', '6way'] if args.classification == 'both' else [args.classification]
    
    # Process IEMOCAP
    if args.dataset in ['iemocap', 'both']:
        logger.info("Processing IEMOCAP dataset...")
        iemocap_processor = IEMOCAPProcessor(args.iemocap_path, output_path)
        
        for classification_type in classification_types:
            logger.info(f"Processing IEMOCAP {classification_type} classification...")
            datasets = iemocap_processor.process_existing_csvs(
                classification_type, args.include_undefined
            )
            
            if datasets:
                print_dataset_statistics(datasets, classification_type)
                save_datasets(datasets, output_path, 'iemocap', classification_type)
    
    # Process MELD
    if args.dataset in ['meld', 'both']:
        logger.info("Processing MELD dataset...")
        meld_processor = MELDProcessor(args.meld_path, output_path)
        
        for classification_type in classification_types:
            logger.info(f"Processing MELD {classification_type} classification...")
            datasets = meld_processor.process_meld_csvs(classification_type)
            
            if datasets:
                print_dataset_statistics(datasets, classification_type)
                save_datasets(datasets, output_path, 'meld', classification_type)
    
    logger.info("Preprocessing completed!")

if __name__ == "__main__":
    main()