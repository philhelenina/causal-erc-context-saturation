"""
Bayesian Neural Network Modules for SenticCrystal.

Implements Bayesian LSTM, Transformer, and Context LSTM with uncertainty quantification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class BayesianLinear(nn.Module):
    """Bayesian Linear Layer with weight distributions."""
    
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight mean and log variance parameters
        self.weight_mean = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features) * 0.1 - 2.0)
        
        # Bias mean and log variance parameters  
        self.bias_mean = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_logvar = nn.Parameter(torch.randn(out_features) * 0.1 - 2.0)
        
        # Prior parameters
        self.prior_std = prior_std
        
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Forward pass with optional weight sampling."""
        if sample and self.training:
            # Sample weights from distributions during training
            weight_std = torch.exp(0.5 * self.weight_logvar)
            bias_std = torch.exp(0.5 * self.bias_logvar)
            
            weight_eps = torch.randn_like(self.weight_mean)
            bias_eps = torch.randn_like(self.bias_mean)
            
            weight = self.weight_mean + weight_std * weight_eps
            bias = self.bias_mean + bias_std * bias_eps
            
        else:
            # Use mean weights during inference
            weight = self.weight_mean
            bias = self.bias_mean
            
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """Calculate KL divergence from prior."""
        # KL divergence for weights
        weight_var = torch.exp(self.weight_logvar)
        weight_kl = 0.5 * torch.sum(
            self.weight_mean.pow(2) / (self.prior_std ** 2) + 
            weight_var / (self.prior_std ** 2) - 
            self.weight_logvar - 
            torch.log(torch.tensor(self.prior_std ** 2)) - 1
        )
        
        # KL divergence for bias
        bias_var = torch.exp(self.bias_logvar)
        bias_kl = 0.5 * torch.sum(
            self.bias_mean.pow(2) / (self.prior_std ** 2) + 
            bias_var / (self.prior_std ** 2) - 
            self.bias_logvar - 
            torch.log(torch.tensor(self.prior_std ** 2)) - 1
        )
        
        return weight_kl + bias_kl

class BayesianLSTMCell(nn.Module):
    """Bayesian LSTM Cell with uncertainty in weights."""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Bayesian linear layers for LSTM gates
        self.input_gate = BayesianLinear(input_size + hidden_size, hidden_size)
        self.forget_gate = BayesianLinear(input_size + hidden_size, hidden_size) 
        self.cell_gate = BayesianLinear(input_size + hidden_size, hidden_size)
        self.output_gate = BayesianLinear(input_size + hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor], 
                sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through Bayesian LSTM cell."""
        h_prev, c_prev = hidden
        combined = torch.cat([x, h_prev], dim=1)
        
        # Compute gates with sampling
        i_t = torch.sigmoid(self.input_gate(combined, sample=sample))
        f_t = torch.sigmoid(self.forget_gate(combined, sample=sample))
        g_t = torch.tanh(self.cell_gate(combined, sample=sample))
        o_t = torch.sigmoid(self.output_gate(combined, sample=sample))
        
        # Update cell state and hidden state
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        
        return h_t, c_t
    
    def kl_divergence(self) -> torch.Tensor:
        """Total KL divergence of all gates."""
        return (self.input_gate.kl_divergence() + 
                self.forget_gate.kl_divergence() +
                self.cell_gate.kl_divergence() + 
                self.output_gate.kl_divergence())

class BayesianContextLSTM(nn.Module):
    """
    Bayesian Context LSTM for Config146.
    
    Processes 4-turn conversation context with uncertainty quantification.
    """
    
    def __init__(self, input_size: int = 768, hidden_size: int = 256, 
                 num_layers: int = 2, context_turns: int = 4):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.num_layers = num_layers
        self.context_turns = context_turns
        
        # Stack of Bayesian LSTM cells
        self.lstm_cells = nn.ModuleList([
            BayesianLSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        # Context attention mechanism
        self.context_attention = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        
        logger.info(f"✅ Initialized BayesianContextLSTM: {input_size}→{hidden_size}, {num_layers} layers")
    
    def forward(self, x: torch.Tensor, n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.
        
        Args:
            x: Input embeddings (batch_size, seq_len, input_size)
            n_samples: Number of Monte Carlo samples for uncertainty
            
        Returns:
            mean_output: Mean prediction
            uncertainty: Predictive uncertainty
        """
        batch_size, seq_len, _ = x.size()
        
        # Collect outputs from multiple samples
        sample_outputs = []
        
        for sample_idx in range(n_samples):
            # Initialize hidden states
            h_states = [torch.zeros(batch_size, self.hidden_size, device=x.device) 
                       for _ in range(self.num_layers)]
            c_states = [torch.zeros(batch_size, self.hidden_size, device=x.device)
                       for _ in range(self.num_layers)]
            
            # Process sequence
            all_outputs = []
            for t in range(seq_len):
                layer_input = x[:, t, :]
                
                # Forward through all LSTM layers
                for layer_idx in range(self.num_layers):
                    h_new, c_new = self.lstm_cells[layer_idx](
                        layer_input, (h_states[layer_idx], c_states[layer_idx]), 
                        sample=self.training
                    )
                    h_states[layer_idx] = h_new
                    c_states[layer_idx] = c_new
                    layer_input = h_new
                
                all_outputs.append(h_states[-1])
            
            # Stack outputs: (batch_size, seq_len, hidden_size)
            lstm_outputs = torch.stack(all_outputs, dim=1)
            
            # Apply context attention
            attended_output, _ = self.context_attention(lstm_outputs, lstm_outputs, lstm_outputs)
            
            # Use last timestep output
            final_output = attended_output[:, -1, :]  # (batch_size, hidden_size)
            sample_outputs.append(final_output)
        
        # Calculate mean and uncertainty
        sample_outputs = torch.stack(sample_outputs)  # (n_samples, batch_size, hidden_size)
        mean_output = torch.mean(sample_outputs, dim=0)
        uncertainty = torch.var(sample_outputs, dim=0)  # Epistemic uncertainty
        
        return mean_output, uncertainty
    
    def kl_divergence(self) -> torch.Tensor:
        """Total KL divergence for all LSTM cells."""
        return sum(cell.kl_divergence() for cell in self.lstm_cells)

class BayesianTransformer(nn.Module):
    """Bayesian Transformer with uncertainty quantification."""
    
    def __init__(self, d_model: int = 768, nhead: int = 8, num_layers: int = 6):
        super().__init__()
        self.d_model = d_model
        
        # Bayesian attention layers
        self.attention_layers = nn.ModuleList([
            BayesianMultiheadAttention(d_model, nhead) 
            for _ in range(num_layers)
        ])
        
        # Bayesian feed-forward layers
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                BayesianLinear(d_model, d_model * 4),
                nn.ReLU(),
                BayesianLinear(d_model * 4, d_model)
            ) for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers * 2)
        ])
    
    def forward(self, x: torch.Tensor, n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with uncertainty estimation."""
        sample_outputs = []
        
        for _ in range(n_samples):
            output = x
            norm_idx = 0
            
            for i, (attn_layer, ff_layer) in enumerate(zip(self.attention_layers, self.ff_layers)):
                # Self-attention with residual connection
                attn_out = attn_layer(output, sample=self.training)
                output = self.layer_norms[norm_idx](output + attn_out)
                norm_idx += 1
                
                # Feed-forward with residual connection  
                ff_out = ff_layer(output)
                output = self.layer_norms[norm_idx](output + ff_out)
                norm_idx += 1
                
            sample_outputs.append(output)
        
        # Calculate statistics
        sample_outputs = torch.stack(sample_outputs)
        mean_output = torch.mean(sample_outputs, dim=0)
        uncertainty = torch.var(sample_outputs, dim=0)
        
        return mean_output, uncertainty
    
    def kl_divergence(self) -> torch.Tensor:
        """Total KL divergence."""
        total_kl = 0.0
        
        for attn_layer in self.attention_layers:
            total_kl += attn_layer.kl_divergence()
            
        for ff_layer in self.ff_layers:
            for module in ff_layer:
                if isinstance(module, BayesianLinear):
                    total_kl += module.kl_divergence()
                    
        return total_kl

class BayesianMultiheadAttention(nn.Module):
    """Bayesian Multi-head Attention."""
    
    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # Bayesian projections
        self.q_proj = BayesianLinear(d_model, d_model)
        self.k_proj = BayesianLinear(d_model, d_model) 
        self.v_proj = BayesianLinear(d_model, d_model)
        self.out_proj = BayesianLinear(d_model, d_model)
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Forward pass."""
        batch_size, seq_len, _ = x.size()
        
        # Bayesian projections
        q = self.q_proj(x, sample=sample)
        k = self.k_proj(x, sample=sample)
        v = self.v_proj(x, sample=sample)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.out_proj(attn_output, sample=sample)
    
    def kl_divergence(self) -> torch.Tensor:
        """Total KL divergence."""
        return (self.q_proj.kl_divergence() + 
                self.k_proj.kl_divergence() +
                self.v_proj.kl_divergence() + 
                self.out_proj.kl_divergence())