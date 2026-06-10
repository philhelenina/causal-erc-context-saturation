"""
Sequential Bayesian Models for Emotion Recognition.

True Bayesian approach where only previously seen utterances inform current predictions.
No fixed context windows - pure sequential inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

class SequentialBayesianLSTM(nn.Module):
    """
    Sequential Bayesian LSTM for emotion recognition.
    
    True Bayesian premise: Only previously seen utterances inform current prediction.
    No fixed context windows - processes conversation sequentially.
    """
    
    def __init__(self, input_size: int = 768, hidden_size: int = 256, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Sequential LSTM - maintains conversation state
        self.lstm = nn.LSTM(
            input_size, hidden_size, 
            batch_first=True, num_layers=2, dropout=dropout
        )
        
        # Bayesian classification layers with dropout for uncertainty
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Prior belief initialization (can be learned)
        self.prior_logits = nn.Parameter(torch.zeros(num_classes))
        
        logger.info(f"✅ SequentialBayesianLSTM: {input_size}→{hidden_size}→{num_classes}, dropout={dropout}")
    
    def init_conversation_state(self, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state for new conversation."""
        # For LSTM with batch_first=True, hidden state is (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(2, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(2, batch_size, self.hidden_size, device=device)
        return (h0, c0)
    
    def forward_step(self, x: torch.Tensor, hidden_state: Tuple[torch.Tensor, torch.Tensor], 
                     n_samples: int = 10) -> Dict[str, torch.Tensor]:
        """
        Forward step for single utterance with Bayesian inference.
        
        Args:
            x: Current utterance embedding (batch_size, input_size)
            hidden_state: LSTM hidden state from previous utterances
            n_samples: Number of Monte Carlo samples for uncertainty estimation
            
        Returns:
            Dictionary with prediction, uncertainty, and updated hidden state
        """
        batch_size = x.size(0)
        
        # Add sequence dimension for LSTM
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        
        # Collect samples for uncertainty estimation
        prediction_samples = []
        hidden_samples = []
        
        for _ in range(n_samples):
            # Enable dropout for sampling
            self.train()
            
            # LSTM forward step
            lstm_out, new_hidden = self.lstm(x, hidden_state)
            
            # Classification with dropout sampling
            logits = self.classifier(lstm_out.squeeze(1))  # Remove sequence dim
            probs = torch.softmax(logits, dim=-1)
            
            prediction_samples.append(probs)
            hidden_samples.append(new_hidden)
        
        # Calculate Bayesian statistics
        prediction_samples = torch.stack(prediction_samples)  # (n_samples, batch_size, num_classes)
        mean_probs = torch.mean(prediction_samples, dim=0)
        prediction_variance = torch.var(prediction_samples, dim=0)
        
        # Predictions and confidence
        max_prob, predicted_class = torch.max(mean_probs, dim=-1)
        total_uncertainty = torch.sum(prediction_variance, dim=-1)
        confidence = max_prob / (1.0 + total_uncertainty)
        
        # Use mean hidden state for next step (approximate)
        # Alternative: sample one hidden state randomly
        mean_h = torch.mean(torch.stack([h for (h, c) in hidden_samples]), dim=0)
        mean_c = torch.mean(torch.stack([c for (h, c) in hidden_samples]), dim=0)
        mean_hidden = (mean_h, mean_c)
        
        return {
            'predictions': predicted_class,
            'probabilities': mean_probs,
            'uncertainty': total_uncertainty,
            'confidence': confidence,
            'needs_review': confidence < 0.7,
            'hidden_state': mean_hidden,
            'prediction_variance': prediction_variance
        }
    
    def forward_conversation(self, conversation_embeddings: List[torch.Tensor], 
                           n_samples: int = 10) -> List[Dict[str, torch.Tensor]]:
        """
        Process entire conversation sequentially with Bayesian inference.
        
        Args:
            conversation_embeddings: List of utterance embeddings in temporal order
            n_samples: Number of Monte Carlo samples
            
        Returns:
            List of prediction dictionaries for each utterance
        """
        batch_size = conversation_embeddings[0].size(0)
        device = conversation_embeddings[0].device
        
        # Initialize conversation state
        hidden_state = self.init_conversation_state(batch_size, device)
        
        # Process each utterance sequentially
        results = []
        for i, utterance_emb in enumerate(conversation_embeddings):
            # Make prediction based on current state (informed by previous utterances)
            step_result = self.forward_step(utterance_emb, hidden_state, n_samples)
            
            # Update hidden state for next utterance
            hidden_state = step_result['hidden_state']
            
            # Store results
            step_result['utterance_index'] = i
            results.append(step_result)
            
            logger.debug(f"Utterance {i}: Pred={step_result['predictions'][0].item()}, "
                        f"Conf={step_result['confidence'][0].item():.3f}")
        
        return results

class SequentialBayesianConfig146(nn.Module):
    """
    Sequential Bayesian version of Config146.
    
    Processes conversations sequentially, maintaining belief state throughout.
    """
    
    def __init__(self, num_classes: int = 4, hidden_size: int = 256, dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        
        # Sequential Bayesian LSTM
        self.sequential_lstm = SequentialBayesianLSTM(
            input_size=768,  # SentenceTransformer embedding size
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout=dropout
        )
        
        logger.info(f"✅ SequentialBayesianConfig146: {num_classes} classes, hidden={hidden_size}")
    
    def predict_conversation(self, embeddings: List[torch.Tensor], target_indices: List[int], 
                           n_samples: int = 10) -> Dict[str, List]:
        """
        Predict emotions for specific utterances in conversation.
        
        Args:
            embeddings: All utterance embeddings in temporal order
            target_indices: Indices of utterances to make predictions for
            n_samples: Monte Carlo samples for uncertainty
            
        Returns:
            Dictionary with predictions and uncertainties for target utterances
        """
        # Process entire conversation sequentially
        conversation_results = self.sequential_lstm.forward_conversation(embeddings, n_samples)
        
        # Extract results for target utterances only
        target_predictions = []
        target_confidences = []
        target_uncertainties = []
        target_probabilities = []
        target_needs_review = []
        
        for idx in target_indices:
            if idx < len(conversation_results):
                result = conversation_results[idx]
                target_predictions.append(result['predictions'])
                target_confidences.append(result['confidence'])
                target_uncertainties.append(result['uncertainty'])
                target_probabilities.append(result['probabilities'])
                target_needs_review.append(result['needs_review'])
        
        return {
            'predictions': target_predictions,
            'confidences': target_confidences,
            'uncertainties': target_uncertainties,
            'probabilities': target_probabilities,
            'needs_review': target_needs_review,
            'conversation_length': len(conversation_results)
        }

def demonstrate_sequential_vs_windowed():
    """Demonstrate difference between sequential and windowed approaches."""
    
    print("🔍 Sequential Bayesian vs Windowed Context Comparison")
    print("="*60)
    
    # Simulate conversation embeddings (5 utterances)
    device = 'cpu'
    batch_size = 1
    conversation = [torch.randn(batch_size, 768) for _ in range(5)]
    
    # Sequential approach
    sequential_model = SequentialBayesianLSTM(768, 256, 4, dropout=0.3)
    sequential_results = sequential_model.forward_conversation(conversation, n_samples=5)
    
    print("📈 Sequential Bayesian Results:")
    for i, result in enumerate(sequential_results):
        pred = result['predictions'][0].item()
        conf = result['confidence'][0].item()
        unc = result['uncertainty'][0].item()
        print(f"  Utterance {i}: Pred={pred}, Conf={conf:.3f}, Unc={unc:.4f}")
    
    print("\n🔄 Key Differences:")
    print("1. Sequential: Each prediction uses ALL previous context")
    print("2. Windowed: Each prediction uses fixed N-turn window")
    print("3. Sequential: Hidden state evolves throughout conversation")
    print("4. Windowed: Each window processed independently")
    print("5. Sequential: True Bayesian - prior beliefs update with evidence")
    
    return sequential_results

if __name__ == "__main__":
    demonstrate_sequential_vs_windowed()