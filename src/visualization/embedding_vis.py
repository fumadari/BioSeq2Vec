import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

def plot_embedding_2d(embeddings, labels, method='tsne', title='Embedding Visualization'):
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title(f'{title} ({method.upper()})')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig(f'results/figures/{title.lower().replace(" ", "_")}_{method}.png')
    plt.close()

def plot_embedding_heatmap(embeddings, sequences, n_samples=50, title='Embedding Heatmap'):
    sample_indices = np.random.choice(len(embeddings), n_samples, replace=False)
    sample_embeddings = embeddings[sample_indices]
    sample_sequences = [sequences[i] for i in sample_indices]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(sample_embeddings, cmap='viridis', xticklabels=False)
    plt.title(title)
    plt.xlabel('Embedding Dimensions')
    plt.ylabel('Sequences')
    plt.savefig(f'results/figures/{title.lower().replace(" ", "_")}.png')
    plt.close()

def plot_sequence_similarity(embeddings, sequences, n_samples=10, title='Sequence Similarity'):
    sample_indices = np.random.choice(len(embeddings), n_samples, replace=False)
    sample_embeddings = embeddings[sample_indices]
    sample_sequences = [sequences[i] for i in sample_indices]
    
    similarity_matrix = np.dot(sample_embeddings, sample_embeddings.T)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', xticklabels=sample_sequences, yticklabels=sample_sequences)
    plt.title(title)
    plt.xlabel('Sequences')
    plt.ylabel('Sequences')
    plt.savefig(f'results/figures/{title.lower().replace(" ", "_")}.png')
    plt.close()

def visualize_embeddings(embeddings, sequences, labels, output_dir='results/figures'):
    plot_embedding_2d(embeddings, labels, method='tsne', title='BioSeq2Vec+ Embeddings (t-SNE)')
    plot_embedding_2d(embeddings, labels, method='pca', title='BioSeq2Vec+ Embeddings (PCA)')
    plot_embedding_2d(embeddings, labels, method='umap', title='BioSeq2Vec+ Embeddings (UMAP)')
    plot_embedding_heatmap(embeddings, sequences, title='BioSeq2Vec+ Embedding Heatmap')
    plot_sequence_similarity(embeddings, sequences, title='BioSeq2Vec+ Sequence Similarity')

if __name__ == '__main__':
    # Example usage
    embeddings = np.random.rand(100, 64)  # 100 sequences, 64-dimensional embeddings
    sequences = [''.join(np.random.choice(['A', 'T', 'C', 'G'], 10)) for _ in range(100)]
    labels = np.random.randint(0, 5, 100)  # 5 classes
    visualize_embeddings(embeddings, sequences, labels)
