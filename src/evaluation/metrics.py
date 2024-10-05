from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, r2_score
from scipy.stats import spearmanr
from Bio import pairwise2
import numpy as np

def compute_metrics(predictions, labels, task='classification'):
    if task == 'classification':
        return compute_classification_metrics(predictions, labels)
    elif task == 'regression':
        return compute_regression_metrics(predictions, labels)
    elif task == 'similarity':
        return compute_similarity_metrics(predictions, labels)
    else:
        raise ValueError(f"Unknown task: {task}")

def compute_classification_metrics(predictions, labels):
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def compute_regression_metrics(predictions, labels):
    mse = mean_squared_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    spearman_corr, _ = spearmanr(labels, predictions)
    return {
        'mse': mse,
        'r2': r2,
        'spearman_correlation': spearman_corr
    }

def compute_similarity_metrics(embeddings1, embeddings2):
    cosine_sim = np.mean([np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)) 
                          for e1, e2 in zip(embeddings1, embeddings2)])
    euclidean_dist = np.mean([np.linalg.norm(e1 - e2) for e1, e2 in zip(embeddings1, embeddings2)])
    return {
        'cosine_similarity': cosine_sim,
        'euclidean_distance': euclidean_dist
    }

def sequence_alignment_score(seq1, seq2):
    alignments = pairwise2.align.globalxx(seq1, seq2)
    return alignments[0].score if alignments else 0

def compute_biological_metrics(sequences, embeddings, labels=None):
    # Compute pairwise alignment scores
    alignment_scores = np.array([[sequence_alignment_score(s1, s2) for s2 in sequences] for s1 in sequences])
    
    # Compute pairwise embedding similarities
    embedding_sims = np.array([[np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)) 
                                for e2 in embeddings] for e1 in embeddings])
    
    # Compute correlation between alignment scores and embedding similarities
    correlation, _ = spearmanr(alignment_scores.flatten(), embedding_sims.flatten())
    
    metrics = {
        'sequence_embedding_correlation': correlation
    }
    
    if labels is not None:
        # Compute correlation between labels and embedding similarities
        label_sims = np.array([[1 if l1 == l2 else 0 for l2 in labels] for l1 in labels])
        label_correlation, _ = spearmanr(label_sims.flatten(), embedding_sims.flatten())
        metrics['label_embedding_correlation'] = label_correlation
    
    return metrics