import random
from Bio.Seq import Seq
from Bio.Data import CodonTable

class SequenceAugmenter:
    def __init__(self, config):
        self.config = config
        self.augmentation_methods = {
            'reverse_complement': self.reverse_complement,
            'random_mutations': self.random_mutations,
            'codon_swap': self.codon_swap,
            'sequence_crop': self.sequence_crop,
            'sequence_pad': self.sequence_pad
        }

    def __call__(self, batch):
        augmented_batch = []
        for sequence in batch:
            if random.random() < self.config.augment_prob:
                method = random.choice(self.config.enabled_methods)
                augmented_sequence = self.augmentation_methods[method](sequence)
                augmented_batch.append(augmented_sequence)
            else:
                augmented_batch.append(sequence)
        return augmented_batch

    def reverse_complement(self, sequence):
        return str(Seq(sequence).reverse_complement())

    def random_mutations(self, sequence):
        mutated_sequence = list(sequence)
        num_mutations = int(len(sequence) * self.config.mutation_rate)
        for _ in range(num_mutations):
            pos = random.randint(0, len(sequence) - 1)
            mutated_sequence[pos] = random.choice('ATCG')
        return ''.join(mutated_sequence)

    def codon_swap(self, sequence):
        codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
        swapped_codons = []
        for codon in codons:
            if len(codon) == 3:
                amino_acid = CodonTable.standard_dna_table.forward_table.get(codon, 'X')
                synonymous_codons = [c for c, aa in CodonTable.standard_dna_table.forward_table.items() if aa == amino_acid]
                swapped_codons.append(random.choice(synonymous_codons))
            else:
                swapped_codons.append(codon)
        return ''.join(swapped_codons)

    def sequence_crop(self, sequence):
        crop_size = random.randint(int(len(sequence) * 0.8), len(sequence))
        start = random.randint(0, len(sequence) - crop_size)
        return sequence[start:start+crop_size]

    def sequence_pad(self, sequence):
        pad_size = random.randint(0, int(len(sequence) * 0.2))
        pad_seq = ''.join(random.choices('ATCG', k=pad_size))
        if random.random() < 0.5:
            return pad_seq + sequence
        else:
            return sequence + pad_seq