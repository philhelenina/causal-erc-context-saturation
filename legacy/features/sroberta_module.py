# filename changed from sroberta_module_val.py to sroberta_module.py (Jan 27, 2025)
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from typing import List, Tuple, Union
import logging
import pyro
import pyro.distributions as dist

from wnaffect_module import preprocess_sentence

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='sentence_embedder.log',
    filemode='w'
)

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :x.size(-1)]

class BayesianLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)

    def model(self, x, hidden):
        weight_ih = pyro.sample("weight_ih", dist.Normal(self.weight_ih, 0.1).to_event(2))
        weight_hh = pyro.sample("weight_hh", dist.Normal(self.weight_hh, 0.1).to_event(2))
        bias_ih = pyro.sample("bias_ih", dist.Normal(self.bias_ih, 0.1).to_event(1))
        bias_hh = pyro.sample("bias_hh", dist.Normal(self.bias_hh, 0.1).to_event(1))

        gates = F.linear(x, weight_ih, bias_ih) + F.linear(hidden[0], weight_hh, bias_hh)
        i, f, g, o = gates.chunk(4, 1)
        i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
        c = f * hidden[1] + i * g
        h = o * torch.tanh(c)
        return h, c

class BayesianTransformer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def model(self, src):
        attn_output, _ = self.self_attn(src, src, src)
        src = self.norm1(src + attn_output)
        ff_output = self.linear2(F.relu(self.linear1(src)))
        src = self.norm2(src + ff_output)
        return src

class SentenceEmbedder:
    def __init__(self, model_name: str = "nli-distilroberta-base-v2", device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = SentenceTransformer(model_name)
        self.device = device
        self.word_to_sentence_pe = PositionalEncoding(768).to(device)
        self.sentence_to_utterance_pe = PositionalEncoding(768).to(device)
        self.lstm = nn.LSTM(2 * 768, 768, batch_first=True).to(device)
        self.self_attention = nn.MultiheadAttention(2 * 768, num_heads=8, batch_first=True).to(device)
        self.cross_attention = nn.MultiheadAttention(768, num_heads=8, batch_first=True).to(device)
        
        self.bayesian_lstm = BayesianLSTM(768 * 3, 768).to(device)
        self.bayesian_transformer = BayesianTransformer(768, 8, 2048).to(device)
        self.bayesian_context_lstm = BayesianLSTM(768 * 2, 768).to(device)
        
        self.optimizer = pyro.optim.Adam({"lr": 0.01})
        self.guide = pyro.infer.autoguide.AutoDiagonalNormal(self.model)
        self.svi = pyro.infer.SVI(self.model, self.guide, self.optimizer, loss=pyro.infer.Trace_ELBO())

    def pad_wn_embedding(self, wn_embedding: np.ndarray, target_size: int = 768) -> np.ndarray:
        if len(wn_embedding) >= target_size:
            return wn_embedding[:target_size]
        padding = np.zeros(target_size - len(wn_embedding))
        return np.concatenate([wn_embedding, padding])
    
    def process_sentence(self, sentence: str, wn_embedder, apply_word_pe: bool = False) -> torch.Tensor:
        preprocessed_words = preprocess_sentence(sentence)
        
        wn_embeddings = [torch.tensor(self.pad_wn_embedding(wn_embedder.get_embedding(word)), device=self.device)
                         for word in preprocessed_words]
        
        if wn_embeddings:
            wn_emb = torch.mean(torch.stack(wn_embeddings), dim=0)
        else:
            wn_emb = torch.zeros(768, device=self.device)

        # essentially sentence roberta
        roberta_emb = self.model.encode(sentence, convert_to_tensor=True).to(self.device)

        logger.debug(f"RoBERTa embedding shape: {roberta_emb.shape}")
        logger.debug(f"WordNet-Affect embedding shape: {wn_emb.shape}")

        if apply_word_pe:
            wn_emb = self.word_to_sentence_pe(wn_emb.unsqueeze(0)).squeeze(0)
            logger.debug(f"WordNet-Affect embedding shape after PE: {wn_emb.shape}")

        combined_emb = torch.stack([roberta_emb, wn_emb])
        logger.debug(f"Combined embedding shape: {combined_emb.shape}")
        return combined_emb
    
    def pool_sentences(self, sentence_embeddings: torch.Tensor, method: str, apply_sentence_pe: bool = False) -> torch.Tensor:
        logger.debug(f"Input sentence embeddings shape: {sentence_embeddings.shape}")

        sentence_embeddings = sentence_embeddings.float()

        if apply_sentence_pe:
            pe_embeddings = []
            for i in range(sentence_embeddings.size(1)):  # 2 for RoBERTa and WordNet
                pe_emb = self.sentence_to_utterance_pe(sentence_embeddings[:, i, :])
                pe_embeddings.append(pe_emb)
            sentence_embeddings = torch.stack(pe_embeddings, dim=1)
            logger.debug(f"Sentence embeddings shape after PE: {sentence_embeddings.shape}")

        if method == 'lstm':
            lstm_input = sentence_embeddings.view(1, sentence_embeddings.size(0), -1)
            logger.debug(f"LSTM input shape: {lstm_input.shape}")
            _, (h_n, _) = self.lstm(lstm_input)
            output = h_n.squeeze(0)
            logger.debug(f"LSTM output shape before repeat: {output.shape}")
            output = output.repeat(2, 1)
        elif method == 'self_attention':
            attn_input = sentence_embeddings.view(sentence_embeddings.size(0), 1, -1)
            logger.debug(f"Self-attention input shape: {attn_input.shape}")
            attn_output, _ = self.self_attention(attn_input, attn_input, attn_input)
            output = attn_output.mean(dim=0).view(2, 768)
        elif method == 'weighted_mean':
            weights = torch.arange(1, sentence_embeddings.size(0) + 1, dtype=torch.float, device=self.device)
            weights = weights / weights.sum()
            output = (sentence_embeddings * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=0)
        elif method == 'simple_mean':
            output = sentence_embeddings.mean(dim=0)
        else:
            raise ValueError("Invalid pooling method")

        logger.debug(f"Pooling output shape: {output.shape}")
        return output

    def combine_embeddings(self, pooled_embedding: torch.Tensor, method: str, alpha: float = None) -> torch.Tensor:
        logger.debug(f"Pooled embedding shape: {pooled_embedding.shape}")

        if method.startswith('cross_attention'):
            if pooled_embedding.dim() != 2 or pooled_embedding.size(0) != 2:
                logger.error(f"Pooled embedding does not have 2 elements: {pooled_embedding.shape}")
                raise ValueError("Expected pooled_embedding to have exactly 2 elements for cross_attention")

            roberta_emb = pooled_embedding[0].unsqueeze(0).unsqueeze(0)
            wn_emb = pooled_embedding[1].unsqueeze(0).unsqueeze(0)
            logger.debug(f"Cross-attention input shapes: {roberta_emb.shape}, {wn_emb.shape}")

            attn_output, _ = self.cross_attention(roberta_emb, wn_emb, wn_emb)
            
            sub_method = method.split('_')[-1]
            if sub_method == 'concatenate':
                output = torch.cat([roberta_emb, attn_output], dim=-1).squeeze(0).squeeze(0)
            elif sub_method == 'sum':
                output = (roberta_emb + attn_output).squeeze(0).squeeze(0)
            elif sub_method == 'average':
                output = ((roberta_emb + attn_output) / 2).squeeze(0).squeeze(0)
            elif sub_method == 'weighted':
                if alpha is None:
                    alpha = 0.5
                    logger.warning(f"Alpha not provided for cross_attention_weighted method. Using default value: {alpha}")
                output = (alpha * roberta_emb + (1-alpha) * attn_output).squeeze(0).squeeze(0)
            else:
                raise ValueError(f"Invalid cross-attention sub-method: {sub_method}")

        elif method == 'sum':
            output = pooled_embedding.sum(dim=0)
        elif method == 'concatenate':
            output = pooled_embedding.view(-1)
        else:
            raise ValueError(f"Invalid combination method: {method}")

        logger.debug(f"Combined embedding shape: {output.shape}")
        return output

    def process_utterance(self, utterance: str, wn_embedder, apply_word_pe: bool = False,
                        pooling_method: str = 'lstm', apply_sentence_pe: bool = False,
                        combination_method: str = 'sum', alpha: float = None) -> np.ndarray:
        logger.debug(f"Processing utterance: {utterance}")
        logger.debug(f"Configuration: word_pe={apply_word_pe}, pooling={pooling_method}, "
                    f"sentence_pe={apply_sentence_pe}, combination={combination_method}, alpha={alpha}")

        sentence_embeddings = [self.process_sentence(sent, wn_embedder, apply_word_pe)
                            for sent in sent_tokenize(utterance)]
        logger.debug(f"Individual sentence embedding shapes: {[emb.shape for emb in sentence_embeddings]}")
        
        sentence_embeddings = torch.stack(sentence_embeddings)
        logger.debug(f"Stacked sentence embeddings shape: {sentence_embeddings.shape}")

        pooled_embedding = self.pool_sentences(sentence_embeddings, pooling_method, apply_sentence_pe)
        logger.debug(f"Pooled embedding shape: {pooled_embedding.shape}")

        final_embedding = self.combine_embeddings(pooled_embedding, combination_method, alpha)
        logger.debug(f"Final embedding shape: {final_embedding.shape}")

        return final_embedding.detach().cpu().numpy()
        
    def process_turn_level(self, utterance_embeddings, method='lstm'):
        turn_embeddings = []
        uncertainties = []
        
        for i, current_emb in enumerate(utterance_embeddings):
            if i < 2:
                if method == 'lstm':
                    output, uncertainty = self.apply_bayesian_lstm(current_emb.unsqueeze(0))
                elif method == 'transformer':
                    output, uncertainty = self.apply_bayesian_transformer(current_emb.unsqueeze(0))
                else:  # context_lstm
                    output, uncertainty = self.apply_bayesian_context_lstm(current_emb.unsqueeze(0))
            else:
                prev_emb1 = utterance_embeddings[i-1]
                prev_emb2 = utterance_embeddings[i-2]
                
                if method == 'lstm':
                    concat_emb = torch.cat([prev_emb2, prev_emb1, current_emb]).unsqueeze(0)
                    output, uncertainty = self.apply_bayesian_lstm(concat_emb)
                elif method == 'transformer':
                    context_emb = torch.stack([prev_emb2, prev_emb1, current_emb])
                    output, uncertainty = self.apply_bayesian_transformer(context_emb)
                else:  # context_lstm
                    context_emb = torch.cat([prev_emb2, prev_emb1]).unsqueeze(0)
                    output, uncertainty = self.apply_bayesian_context_lstm(context_emb)
            
            turn_embeddings.append(output.squeeze(0))
            uncertainties.append(uncertainty.squeeze(0))
        
        return turn_embeddings, uncertainties

    def apply_bayesian_lstm(self, x):
        hidden = (torch.zeros(1, self.bayesian_lstm.hidden_size, device=self.device),
                  torch.zeros(1, self.bayesian_lstm.hidden_size, device=self.device))
        
        pyro.clear_param_store()
        for _ in range(100):  # VI iterations
            loss = self.svi.step(x, hidden)
        
        num_samples = 10
        posterior_samples = [self.bayesian_lstm.model(x, hidden) for _ in range(num_samples)]
        
        outputs = torch.stack([sample[0] for sample in posterior_samples])
        mean_output = outputs.mean(0)
        uncertainty_output = outputs.std(0)
        
        return mean_output, uncertainty_output

    def apply_bayesian_transformer(self, x):
        pyro.clear_param_store()
        for _ in range(100):  # VI iterations
            loss = self.svi.step(x)
        
        num_samples = 10
        posterior_samples = [self.bayesian_transformer.model(x) for _ in range(num_samples)]
        
        outputs = torch.stack(posterior_samples)
        mean_output = outputs.mean(0)
        uncertainty_output = outputs.std(0)
        
        return mean_output, uncertainty_output
        
    def apply_bayesian_context_lstm(self, x):
        hidden = (torch.zeros(1, self.bayesian_context_lstm.hidden_size, device=self.device),
                  torch.zeros(1, self.bayesian_context_lstm.hidden_size, device=self.device))
        
        pyro.clear_param_store()
        for _ in range(100):  # VI iterations
            loss = self.svi.step(x, hidden)
        
        num_samples = 10
        posterior_samples = [self.bayesian_context_lstm.model(x, hidden) for _ in range(num_samples)]
        
        outputs = torch.stack([sample[0] for sample in posterior_samples])
        mean_output = outputs.mean(0)
        uncertainty_output = outputs.std(0)
        
        return mean_output, uncertainty_output

def process_dialogue(dialogue, sentence_embedder, wn_embedder):
    utterance_embeddings = []
    for utterance in dialogue:
        embedding = sentence_embedder.process_utterance(utterance['text'], wn_embedder)
        utterance_embeddings.append(embedding)
    
    context_size = 2
    padded_embeddings = [np.zeros_like(utterance_embeddings[0])] * context_size + utterance_embeddings + [np.zeros_like(utterance_embeddings[0])] * context_size
    
    turn_embeddings, uncertainties = sentence_embedder.process_turn_level(padded_embeddings)
    
    results = []
    for i, (emb, unc, utterance) in enumerate(zip(turn_embeddings[context_size:-context_size], uncertainties[context_size:-context_size], dialogue)):
        results.append({
            'id': utterance['id'],
            'embedding': emb.detach().cpu().numpy(),
            'uncertainty': unc.detach().cpu().numpy(),
            'label': utterance['label']
        })
    
    return results
