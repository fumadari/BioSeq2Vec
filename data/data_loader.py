from Bio import SeqIO
import torch
from torch.utils.data import Dataset
import os

class SequenceDataset(Dataset):
    def __init__(self, fasta_file, max_len=100):
        self.sequences = []
        self.max_len = max_len
        self.vocab = {'A':1, 'C':2, 'G':3, 'T':4, 'N':0}  # N for unknown
        for record in SeqIO.parse(fasta_file, "fasta"):
            seq = str(record.seq.upper())
            seq_encoded = [self.vocab.get(base, 0) for base in seq]
            if len(seq_encoded) < max_len:
                seq_encoded += [0] * (max_len - len(seq_encoded))
            else:
                seq_encoded = seq_encoded[:max_len]
            self.sequences.append(seq_encoded)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        return sequence

if __name__ == "__main__":
    dataset = SequenceDataset("sample_sequences.fasta")
    print(f"Number of sequences: {len(dataset)}")
    print(f"First sequence encoding: {dataset[0]}")
