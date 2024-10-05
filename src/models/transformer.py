import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .heads import SequenceClassificationHead, SequenceRegressionHead

class BioTransformer(nn.Module):
    def __init__(self, config):
        super(BioTransformer, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)
        self.position_encoding = PositionalEncoding(config.embedding_dim, config.max_seq_length)
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_layers)])
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        
        # Task-specific heads
        self.classification_head = SequenceClassificationHead(config)
        self.regression_head = SequenceRegressionHead(config)
        
    def forward(self, x, mask=None, task='embedding'):
        x = self.embedding(x)
        x = self.position_encoding(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.layer_norm(x)
        
        if task == 'classification':
            return self.classification_head(x)
        elif task == 'regression':
            return self.regression_head(x)
        else:
            return x  # Return embeddings

class TransformerLayer(nn.Module):
    def __init__(self, config):
        super(TransformerLayer, self).__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.embedding_dim, config.ff_dim),
            nn.ReLU(),
            nn.Linear(config.ff_dim, config.embedding_dim)
        )
        self.layer_norm1 = nn.LayerNorm(config.embedding_dim)
        self.layer_norm2 = nn.LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.layer_norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]