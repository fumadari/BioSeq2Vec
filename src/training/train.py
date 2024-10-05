import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb
from src.data.augmentation import SequenceAugmenter
from src.evaluation.metrics import compute_metrics
from src.models.transformer import BioTransformer

def train(config, train_dataset, val_dataset, test_dataset):
    # Initialize wandb for experiment tracking
    wandb.init(project="BioSeq2Vec+", config=config)
    
    # Set up model and optimizer
    model = BioTransformer(config)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    # Set up data loaders with augmentation
    augmenter = SequenceAugmenter(config.augmentation)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              collate_fn=lambda x: augmenter(default_collate(x)))
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            optimizer.zero_grad()
            loss = model(batch['input_ids'], batch['attention_mask'], task=config.task)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_predictions = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch['input_ids'], batch['attention_mask'], task=config.task)
                val_loss += outputs.loss.item()
                val_predictions.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
                val_labels.extend(batch['labels'].cpu().numpy())
        
        # Compute metrics
        metrics = compute_metrics(val_predictions, val_labels, task=config.task)
        
        # Log to wandb
        wandb.log({
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            **metrics
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"results/models/best_model_{config.task}.pt")
        
        scheduler.step()
    
    # Final evaluation on test set
    model.load_state_dict(torch.load(f"results/models/best_model_{config.task}.pt"))
    test_metrics = evaluate(model, test_loader, config.task)
    wandb.log({"test_" + k: v for k, v in test_metrics.items()})
    
    wandb.finish()

def evaluate(model, data_loader, task):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            outputs = model(batch['input_ids'], batch['attention_mask'], task=task)
            predictions.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
            labels.extend(batch['labels'].cpu().numpy())
    
    return compute_metrics(predictions, labels, task=task)

if __name__ == "__main__":
    import yaml
    with open("configs/training_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Load datasets (implement these functions based on your data format)
    train_dataset = load_dataset(config.train_data_path)
    val_dataset = load_dataset(config.val_data_path)
    test_dataset = load_dataset(config.test_data_path)
    
    train(config, train_dataset, val_dataset, test_dataset)