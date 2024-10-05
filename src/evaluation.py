import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
import os
from model import SeqTransformer
from data_loader import SequenceDataset
import yaml

def visualize_embeddings(model, dataset, config):
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for sequence in dataset:
            sequence = sequence.unsqueeze(0).to(device)
            embedding = model(sequence)
            embeddings.append(embedding.cpu().numpy()[0])
    embeddings = torch.tensor(embeddings).numpy()
    # Reduce dimensions with PCA for visualization
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    plt.figure(figsize=(8,6))
    plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1])
    plt.title('Sequence Embeddings Visualized with PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    os.makedirs('results/visualizations', exist_ok=True)
    plt.savefig('results/visualizations/embeddings_plot.png')
    plt.show()

if __name__ == "__main__":
    # Load configuration
    with open(os.path.join('configs', 'config.yaml'), 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    # Prepare data
    dataset = SequenceDataset(config['data']['sequences_file'])
    
    # Load model
    model = SeqTransformer(config['model'])
    model.load_state_dict(torch.load('results/bioseq2vec_model.pt'))
    
    # Visualize embeddings
    visualize_embeddings(model, dataset, config)
