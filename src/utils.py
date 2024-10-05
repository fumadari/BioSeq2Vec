def sequence_to_indices(sequence, vocab):
    return [vocab.get(base, 0) for base in sequence.upper()]
