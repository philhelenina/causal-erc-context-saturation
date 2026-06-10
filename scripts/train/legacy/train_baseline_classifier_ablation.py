"""
Baseline Classifier Training - Embedding Type Ablation Study
============================================================

Train LSTM or MLP classifiers on embeddings for different embedding types.
"""
import os

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (precision_recall_fscore_support, confusion_matrix, 
                           classification_report, accuracy_score)
from sklearn.preprocessing import LabelEncoder
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
from senticcrystal.models.classifiers import SimpleLSTM, MLP, FocalLoss, EarlyStopping, DEFAULT_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HOME_DIR = Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()
DATA_DIR = HOME_DIR / 'data' / 'iemocap_4way_data'
EMBEDDINGS_BASE_DIR = HOME_DIR / 'data' / 'embeddings' / '4way'
RESULTS_DIR = HOME_DIR / 'results' / 'baseline_classifiers_ablation'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Supported embedding types
EMBEDDING_TYPES = ['bert', 'roberta', 'sentence-roberta', 'wordnet']

# Embedding type specific configurations
EMBEDDING_CONFIGS = {
    'bert': {
        'file_suffix': '_embeddings.npy',  # Simple naming convention
        'expected_dim': 768
    },
    'roberta': {
        'file_suffix': '_embeddings.npy', 
        'expected_dim': 768
    },
    'sentence-roberta': {
        'file_suffix': '_embeddings.npy',
        'expected_dim': 768
    },
    'wordnet': {
        'file_suffix': '_embeddings.npy',  # Assuming same format for wordnet
        'expected_dim': 300
    }
}

def load_embeddings_and_labels(embedding_type, pooling_method=None, dataset_type='train'):
    """Load embeddings and corresponding labels for baseline classification."""
    
    logger.info(f"Loading {dataset_type} data for {embedding_type} embeddings")
    
    # Handle pooling directory structure
    if embedding_type in ['bert', 'roberta']:
        if pooling_method is None:
            pooling_method = 'mean'  # default
        embeddings_dir = EMBEDDINGS_BASE_DIR / embedding_type / pooling_method
    elif embedding_type == 'sentence-roberta':
        embeddings_dir = EMBEDDINGS_BASE_DIR / embedding_type / 'mean'
    elif embedding_type == 'wordnet':
        embeddings_dir = EMBEDDINGS_BASE_DIR / embedding_type
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")
    
    # Load embeddings - {dataset_type}_embeddings.npy
    embedding_file = embeddings_dir / f'{dataset_type}_embeddings.npy'
    
    logger.info(f"Looking for embeddings at: {embedding_file}")
    
    if not embedding_file.exists():
        raise FileNotFoundError(f"Embeddings not found: {embedding_file}")
    
    embeddings = np.load(embedding_file)
    logger.info(f"Loaded embeddings shape: {embeddings.shape}")
    
    # Load corresponding CSV for labels
    csv_file = DATA_DIR / f'{dataset_type}_4way_with_minus_one.csv'
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    logger.info(f"Loaded CSV with {len(df)} rows")
    
    # Filter out -1 labels for training/evaluation
    valid_indices = df['label'] != '-1'
    df_filtered = df[valid_indices].reset_index(drop=True)
    embeddings_filtered = embeddings[valid_indices]
    
    # Encode labels to integers
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(df_filtered['label'])
    
    logger.info(f"Dataset: {dataset_type}")
    logger.info(f"Embedding type: {embedding_type}, Pooling: {pooling_method}")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Valid samples (excluding -1): {len(df_filtered)}")
    logger.info(f"Label distribution: {df_filtered['label'].value_counts().to_dict()}")
    logger.info(f"Filtered embeddings shape: {embeddings_filtered.shape}")
    logger.info(f"Label classes: {label_encoder.classes_}")
    
    return embeddings_filtered, labels_encoded, label_encoder, df_filtered

def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
    """Create PyTorch data loaders."""
    
    # Convert to tensors
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Created data loaders with batch size: {batch_size}")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, config, model_name="model"):
    """Train the model with early stopping."""
    
    device = torch.device(config['device'])
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), 
                          lr=config['learning_rate'], 
                          weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=config['early_stopping_patience'])
    
    best_val_acc = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    logger.info(f"Starting training of {model_name} model...")
    logger.info(f"Device: {device}")
    logger.info(f"Learning rate: {config['learning_rate']}")
    logger.info(f"Batch size: {config['batch_size']}")
    
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if (epoch + 1) % 20 == 0:
            logger.info(f'Epoch [{epoch+1}/{config["num_epochs"]}] '
                       f'Train Loss: {train_loss:.4f}, '
                       f'Val Loss: {val_loss:.4f}, '
                       f'Val Acc: {val_acc:.4f}')
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses, 
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc,
        'final_epoch': epoch + 1
    }


def evaluate_model(model, test_loader, label_encoder, device, model_name="model"):
    """Comprehensive evaluation on test set."""
    
    logger.info(f"Evaluating {model_name} on test set...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    # Average metrics
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro', zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Classification report
    class_names = label_encoder.classes_
    report = classification_report(all_labels, all_predictions, 
                                 target_names=class_names, digits=4)
    
    # Log results
    logger.info(f"\n{model_name} Test Results:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Macro F1: {macro_f1:.4f}")
    logger.info(f"Weighted F1: {weighted_f1:.4f}")
    logger.info(f"Macro Precision: {macro_precision:.4f}")
    logger.info(f"Macro Recall: {macro_recall:.4f}")
    
    logger.info(f"\nPer-class Results:")
    for i, class_name in enumerate(class_names):
        logger.info(f"{class_name}: P={precision[i]:.4f}, R={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}")
    
    logger.info(f"\nClassification Report:\n{report}")
    
    logger.info(f"\nConfusion Matrix:")
    for i, row in enumerate(cm):
        logger.info(f"{class_names[i]}: {row}")
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist(),
        'per_class_f1': f1.tolist(),
        'support': support.tolist(),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'class_names': class_names.tolist()
    }


def save_confusion_matrix_plot(cm, class_names, model_name, embedding_type, save_dir):
    """Save confusion matrix plot."""
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - {embedding_type} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plot_file = save_dir / f'{embedding_type}_{model_name.lower()}_confusion_matrix.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved confusion matrix plot: {plot_file}")
    return plot_file


def main():
    parser = argparse.ArgumentParser(description="Train baseline classifier on embeddings - Ablation Study")
    parser.add_argument('--embedding_type', choices=['bert', 'roberta', 'sentence-roberta', 'wordnet'], 
                       required=True, help='Embedding type for ablation study')
    parser.add_argument('--pooling', choices=['cls', 'mean', 'attn'], default='mean',
                       help='Pooling method (for BERT/RoBERTa only)')
    parser.add_argument('--model', choices=['lstm', 'mlp', 'both'], default='both', 
                       help='Model type to train')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for regularization')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Determine pooling method
    if args.embedding_type in ['bert', 'roberta']:
        pooling = args.pooling
        logger.info(f"Starting baseline classification: {args.embedding_type} with {pooling} pooling")
    else:
        pooling = None
        logger.info(f"Starting baseline classification: {args.embedding_type}")
    
    # Load all datasets with pooling parameter
    logger.info("Loading datasets...")
    X_train, y_train, label_encoder, train_df = load_embeddings_and_labels(
        args.embedding_type, pooling, 'train')
    X_val, y_val, _, val_df = load_embeddings_and_labels(
        args.embedding_type, pooling, 'val') 
    X_test, y_test, _, test_df = load_embeddings_and_labels(
        args.embedding_type, pooling, 'test')
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, args.batch_size
    )
    
    # Model parameters
    input_size = X_train.shape[-1]
    num_classes = len(label_encoder.classes_)
    
    logger.info(f"Input size (embedding dimension): {input_size}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Classes: {label_encoder.classes_}")
    
    # Training configuration
    config = DEFAULT_CONFIG.copy()
    config.update({
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'dropout_rate': args.dropout_rate,
        'weight_decay': args.weight_decay,
        'num_epochs': args.num_epochs,
        'early_stopping_patience': args.early_stopping_patience
    })
    
    results = {}
    models_to_train = ['lstm', 'mlp'] if args.model == 'both' else [args.model]
    
    for model_type in models_to_train:
        logger.info(f"\n{'='*60}")
        if pooling:
            logger.info(f"Training {model_type.upper()} classifier")
            logger.info(f"Embedding: {args.embedding_type} | Pooling: {pooling}")
        else:
            logger.info(f"Training {model_type.upper()} classifier")
            logger.info(f"Embedding: {args.embedding_type}")
        logger.info(f"{'='*60}")
        
        # Create model
        if model_type == 'lstm':
            model = SimpleLSTM(input_size=input_size, hidden_size=args.hidden_size, 
                             num_classes=num_classes, dropout_rate=args.dropout_rate)
        else:
            model = MLP(input_size=input_size, hidden_size=args.hidden_size, 
                       num_classes=num_classes, dropout_rate=args.dropout_rate)
        
        logger.info(f"Model architecture:\n{model}")
        
        # Create model name for logging
        if pooling:
            model_name = f"{model_type.upper()}-{args.embedding_type}-{pooling}"
        else:
            model_name = f"{model_type.upper()}-{args.embedding_type}"
        
        # Train model
        trained_model, training_history = train_model(
            model, train_loader, val_loader, config, model_name
        )
        
        # Evaluate on test set
        device = torch.device(config['device'])
        test_results = evaluate_model(
            trained_model, test_loader, label_encoder, device, model_name
        )
        
        # Save confusion matrix plot with pooling in filename
        cm = np.array(test_results['confusion_matrix'])
        if pooling:
            plot_file = RESULTS_DIR / f'{args.embedding_type}_{pooling}_{model_type}_confusion_matrix.png'
        else:
            plot_file = RESULTS_DIR / f'{args.embedding_type}_{model_type}_confusion_matrix.png'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=test_results['class_names'], 
                    yticklabels=test_results['class_names'])
        plt.title(f'{model_name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved confusion matrix plot: {plot_file}")
        
        # Store results with pooling info
        results[model_type] = {
            'embedding_type': args.embedding_type,
            'pooling_method': pooling if pooling else 'N/A',
            'model_type': model_type,
            'embedding_dimension': input_size,
            'model_params': {
                'input_size': input_size,
                'hidden_size': args.hidden_size,
                'num_classes': num_classes
            },
            'training_config': config,
            'training_history': training_history,
            'test_results': test_results
        }
        
        # Save individual model results with pooling in filename
        if pooling:
            model_results_file = RESULTS_DIR / f'{args.embedding_type}_{pooling}_{model_type}_baseline_results.json'
        else:
            model_results_file = RESULTS_DIR / f'{args.embedding_type}_{model_type}_baseline_results.json'
        
        with open(model_results_file, 'w') as f:
            json.dump(results[model_type], f, indent=2)
        
        logger.info(f"Results saved: {model_results_file}")
    
    # Save combined results with pooling in filename
    if pooling:
        combined_results_file = RESULTS_DIR / f'{args.embedding_type}_{pooling}_baseline_comparison.json'
    else:
        combined_results_file = RESULTS_DIR / f'{args.embedding_type}_baseline_comparison.json'
    
    with open(combined_results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Combined results saved: {combined_results_file}")
    
    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info(f"BASELINE CLASSIFICATION SUMMARY")
    if pooling:
        logger.info(f"Embedding: {args.embedding_type} | Pooling: {pooling} | Dim: {input_size}")
    else:
        logger.info(f"Embedding: {args.embedding_type} | Dim: {input_size}")
    logger.info(f"{'='*70}")
    
    for model_type in models_to_train:
        test_res = results[model_type]['test_results']
        logger.info(f"{model_type.upper()}: Acc={test_res['accuracy']:.4f}, "
                   f"Macro-F1={test_res['macro_f1']:.4f}, "
                   f"Weighted-F1={test_res['weighted_f1']:.4f}")
    
    logger.info("Baseline classification ablation completed!")

if __name__ == "__main__":
    main()

# Examples:
# python train_baseline_classifier_ablation.py --embedding_type bert --model both
# python train_baseline_classifier_ablation.py --embedding_type roberta --model lstm --hidden_size 512
# python train_baseline_classifier_ablation.py --embedding_type wordnet --model mlp --batch_size 64
# python train_baseline_classifier_ablation.py --embedding_type sentence-roberta --model both