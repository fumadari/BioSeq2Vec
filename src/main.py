import torch
from src.training.train import train_model
from model import SeqTransformer
import yaml
import os
from data_loader import SequenceDataset

if __name__ == "__main__":
    # Load configuration
    with open(os.path.join('configs', 'config.yaml'), 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    # Prepare data
    dataset = SequenceDataset(config['data']['sequences_file'])
    
    # Initialize model
    model = SeqTransformer(config['model'])
    
    # Train model
    train_model(model, dataset, config)