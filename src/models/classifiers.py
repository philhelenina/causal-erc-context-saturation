"""
Universal Classifiers for SenticCrystal Project
==============================================

Original structures from config146_lstm.py and config144_mlp.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLSTM(nn.Module):
    """
    Simple LSTM classifier for sequence data.

    Args:
        input_size: Input feature dimension.
        hidden_size: LSTM hidden dimension.
        num_classes: Number of output classes.
        num_layers: Number of LSTM layers.
        dropout_rate: Dropout probability applied after the LSTM output.
    """

    def __init__(
        self,
        input_size=768,
        hidden_size=256,
        num_classes=4,
        num_layers=1,
        dropout_rate=0.3,
    ):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        output, (hidden, cell) = self.lstm(x)
        h = self.dropout(output[:, -1, :])
        return self.fc(h)


class MLP(nn.Module):
    """
    3-layer MLP classifier with ReLU activation and dropout.
    
    Args:
        input_size: Input feature dimension (default: 768 for sentence embeddings)
        hidden_size: First hidden layer dimension (default: 256)
        num_classes: Number of output classes (default: 4 for IEMOCAP 4-way)
        dropout_rate: Dropout probability (default: 0.5)
    """
    
    def __init__(self, input_size=768, hidden_size=256, num_classes=4, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Args:
        alpha: Scaling factor for the modulating term (default: 1)
        gamma: Focusing parameter that reduces loss for well-classified examples (default: 2)
        reduction: Reduction method: 'none' | 'mean' | 'sum' (default: 'mean')
    """
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Args:
        patience: Number of epochs to wait before stopping (default: 5)
        min_delta: Minimum change to qualify as improvement (default: 0)
    """
    
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


# Default training configuration matching original settings
DEFAULT_CONFIG = {
    'learning_rate': 0.0001,
    'batch_size': 32,
    'num_epochs': 300,
    'weight_decay': 0.1,
    'early_stopping_patience': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}