import torch.nn as nn
import torch

class SeqTransformer(nn.Module):
    def __init__(self, config):
        super(SeqTransformer, self).__init__()
        self.embedding = nn.Embedding(config['vocab_size'], config['embedding_dim'], padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['embedding_dim'], 
            nhead=config['nhead']
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])
        self.fc = nn.Linear(config['embedding_dim'], config['output_dim'])
    
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        x = self.embedding(x)  # Shape: (batch_size, seq_length, embedding_dim)
        x = x.permute(1, 0, 2)  # Shape: (seq_length, batch_size, embedding_dim)
        x = self.transformer_encoder(x)  # Shape: (seq_length, batch_size, embedding_dim)
        x = x.mean(dim=0)  # Shape: (batch_size, embedding_dim)
        x = self.fc(x)  # Shape: (batch_size, output_dim)
        return x
