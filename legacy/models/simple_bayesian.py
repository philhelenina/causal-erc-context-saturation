"""
Simple Bayesian Implementation for Config146.

간단한 드롭아웃 기반 베이지안 근사로 시작.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class SimpleBayesianLSTM(nn.Module):
    """
    간단한 베이지안 LSTM - 드롭아웃으로 불확실성 근사.
    
    Monte Carlo Dropout을 사용해서 베이지안 효과를 낸다.
    """
    
    def __init__(self, input_size: int = 768, hidden_size: int = 256, 
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 기본 LSTM
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # 추가 드롭아웃 레이어들 (베이지안 근사용)
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(3)  # 여러 레이어에 드롭아웃
        ])
        
        # Context attention
        self.attention = nn.MultiheadAttention(hidden_size, 8, batch_first=True, dropout=dropout)
        
        logger.info(f"✅ SimpleBayesianLSTM: {input_size}→{hidden_size}, dropout={dropout}")
    
    def forward(self, x: torch.Tensor, n_samples: int = 10, return_uncertainty: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.
        
        Args:
            x: (batch_size, seq_len, input_size)
            n_samples: 몇 번 샘플링할지
            return_uncertainty: 불확실성도 리턴할지
            
        Returns:
            mean_output: 평균 예측
            uncertainty: 불확실성 (분산)
        """
        if not return_uncertainty or not self.training:
            # 일반 forward (드롭아웃 없이)
            self.eval()
            lstm_out, _ = self.lstm(x)
            attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            output = attended_out[:, -1, :]  # 마지막 타임스텝
            return output, None
        
        # 베이지안 샘플링
        sample_outputs = []
        
        for _ in range(n_samples):
            # 매번 다른 드롭아웃 패턴으로 forward
            self.train()  # 드롭아웃 활성화
            
            # LSTM forward
            lstm_out, _ = self.lstm(x)
            
            # 여러 단계에서 드롭아웃 적용
            for dropout_layer in self.dropout_layers:
                lstm_out = dropout_layer(lstm_out)
            
            # Attention
            attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            
            # 마지막 타임스텝 출력
            output = attended_out[:, -1, :]
            sample_outputs.append(output)
        
        # 통계 계산
        sample_outputs = torch.stack(sample_outputs)  # (n_samples, batch_size, hidden_size)
        mean_output = torch.mean(sample_outputs, dim=0)
        uncertainty = torch.var(sample_outputs, dim=0)  # 예측 불확실성
        
        return mean_output, uncertainty

class SimpleBayesianClassifier(nn.Module):
    """
    Config146용 간단한 베이지안 분류기.
    """
    
    def __init__(self, input_size: int = 256, num_classes: int = 4, dropout: float = 0.5):
        super().__init__()
        
        # 베이지안 Context LSTM
        self.bayesian_lstm = SimpleBayesianLSTM(768, input_size, dropout=dropout)
        
        # 분류 헤드 (여러 드롭아웃 포함)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_size // 2, num_classes)
        )
        
        logger.info(f"✅ SimpleBayesianClassifier: {input_size}→{num_classes}")
    
    def forward(self, x: torch.Tensor, n_samples: int = 10) -> dict:
        """
        베이지안 예측 with 불확실성.
        
        Returns:
            dict with 'prediction', 'uncertainty', 'confidence'
        """
        # 베이지안 LSTM forward
        lstm_out, lstm_uncertainty = self.bayesian_lstm(x, n_samples=n_samples)
        
        # 분류 예측도 샘플링
        classification_samples = []
        for _ in range(n_samples):
            self.train()  # 드롭아웃 활성화
            logits = self.classifier(lstm_out)
            probs = torch.softmax(logits, dim=-1)
            classification_samples.append(probs)
        
        # 분류 통계
        classification_samples = torch.stack(classification_samples)
        mean_probs = torch.mean(classification_samples, dim=0)
        classification_uncertainty = torch.var(classification_samples, dim=0)
        
        # 신뢰도 계산 (높은 확률 + 낮은 불확실성)
        max_prob, predicted_class = torch.max(mean_probs, dim=-1)
        total_uncertainty = torch.sum(classification_uncertainty, dim=-1)  # 총 불확실성
        confidence = max_prob / (1.0 + total_uncertainty)  # 정규화된 신뢰도
        
        return {
            'prediction': predicted_class,
            'probabilities': mean_probs,
            'uncertainty': total_uncertainty,
            'confidence': confidence,
            'lstm_uncertainty': lstm_uncertainty
        }
    
    def predict_with_confidence(self, x: torch.Tensor, confidence_threshold: float = 0.7) -> dict:
        """
        신뢰도 기반 예측.
        
        낮은 신뢰도면 "uncertain" 플래그 포함.
        """
        result = self.forward(x)
        
        # 신뢰도 체크
        is_confident = result['confidence'] > confidence_threshold
        
        result.update({
            'is_confident': is_confident,
            'needs_human_review': ~is_confident
        })
        
        return result

def demonstrate_bayesian_difference():
    """베이지안과 일반 LSTM 차이 시연."""
    
    # 가짜 데이터 (4-turn context)
    batch_size, seq_len, embed_dim = 32, 4, 768
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    print("🔍 베이지안 vs 일반 LSTM 비교")
    print("="*50)
    
    # 1. 일반 LSTM
    regular_lstm = nn.LSTM(768, 256, batch_first=True)
    regular_out, _ = regular_lstm(x)
    regular_final = regular_out[:, -1, :]  # 마지막 출력
    
    print(f"일반 LSTM:")
    print(f"  출력 형태: {regular_final.shape}")
    print(f"  평균: {regular_final.mean().item():.4f}")
    print(f"  표준편차: {regular_final.std().item():.4f}")
    print(f"  불확실성: 측정 불가 ❌")
    
    # 2. 베이지안 LSTM  
    bayesian_lstm = SimpleBayesianLSTM(768, 256)
    bayesian_out, uncertainty = bayesian_lstm(x, n_samples=10)
    
    print(f"\n베이지안 LSTM:")
    print(f"  출력 형태: {bayesian_out.shape}")
    print(f"  평균: {bayesian_out.mean().item():.4f}")
    print(f"  표준편차: {bayesian_out.std().item():.4f}")
    print(f"  불확실성: {uncertainty.mean().item():.4f} ✅")
    
    # 3. 분류 비교
    bayesian_classifier = SimpleBayesianClassifier(256, 4)
    result = bayesian_classifier.predict_with_confidence(x)
    
    print(f"\n베이지안 분류:")
    print(f"  예측 클래스: {result['prediction'][0].item()}")
    print(f"  신뢰도: {result['confidence'][0].item():.4f}")
    print(f"  불확실성: {result['uncertainty'][0].item():.4f}")
    print(f"  리뷰 필요: {result['needs_human_review'][0].item()}")

if __name__ == "__main__":
    demonstrate_bayesian_difference()