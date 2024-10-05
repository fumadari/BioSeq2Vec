import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

def train_model(model, dataset, config):
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = torch.nn.MSELoss()

    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{config["training"]["epochs"]}'):
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            # Dummy target for demonstration purposes
            target = torch.zeros(outputs.shape).to(device)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
    
    # Save the model
    os.makedirs('results', exist_ok=True)
    torch.save(model.state_dict(), os.path.join('results', 'bioseq2vec_model.pt'))
