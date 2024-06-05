
import os
from Bio import SeqIO
from collections import defaultdict

def generate_kmers(sequence, k):
    """
    Generate k-mers from a given sequence.
    """
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    return kmers

def count_kmers_in_file(file_path, k, sample_name):
    """
    Count k-mers in a given file.
    """
    kmer_counts = defaultdict(int)
    for record in SeqIO.parse(file_path, 'fasta'):
        kmers = generate_kmers(str(record.seq), k)
        for kmer in kmers:
            kmer_counts[kmer] += 1
    return {kmer: f"{sample_name}:{count}" for kmer, count in kmer_counts.items()}

def aggregate_kmer_counts(kmer_counts_list):
    """
    Aggregate k-mer counts from multiple samples.
    """
    aggregated_counts = defaultdict(list)
    for kmer_counts in kmer_counts_list:
        for kmer, count in kmer_counts.items():
            aggregated_counts[kmer].append(count)
    return aggregated_counts


def write_aggregated_kmers_to_files(aggregated_counts, output_prefix, num_files):
    """
    Write the aggregated k-mer counts to multiple files.
    """
    items = list(aggregated_counts.items())
    chunk_size = len(items) // num_files
    for i in range(num_files):
        chunk = items[i * chunk_size: (i + 1) * chunk_size]
        output_file = f"{output_prefix}_chunk_{i + 1}.txt"
        with open(output_file, 'w') as out_file:
            for kmer, counts in chunk:
                out_file.write(f"{kmer} | {' '.join(counts)}\n")

    # If there are any remaining items, write them to the last file
    remaining_items = items[num_files * chunk_size:]
    if remaining_items:
        output_file = f"{output_prefix}/chunk_{num_files}.txt"
        with open(output_file, 'a') as out_file:
            for kmer, counts in remaining_items:
                out_file.write(f"{kmer} | {' '.join(counts)}\n")


def main():
    base_path = f'data/assemblies/data'
    k = 54
    num_files = 5  # Number of output files
    output_prefix = 'data/test_aggregated_kmers'

    kmer_counts_list = []

    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            sample_path = os.path.join(root, dir_name, 'assembly_unicycler', 'assembly.fasta')
            if os.path.exists(sample_path):
                sample_name = dir_name
                kmer_counts = count_kmers_in_file(sample_path, k, sample_name)
                kmer_counts_list.append(kmer_counts)

    aggregated_counts = aggregate_kmer_counts(kmer_counts_list)
    write_aggregated_kmers_to_files(aggregated_counts, output_prefix, num_files)


if __name__ == "__main__":
    main()
