import sys
import os
import numpy as np
import torch
from Bio import SeqIO


# Define a function to load Kidera factors
def load_kidera_values(file_path, idx_to_amino):
    dict_aa_values = {}
    with open(file_path) as aa_kidera:
        for line in aa_kidera:
            line = line.strip().split(',')
            list_val = [float(v) for v in line[1:]]  # Read Kidera values starting from the second column
            dict_aa_values[line[0]] = list_val  # The first column is the amino acid symbol
    # Create a matrix containing Kidera vectors for all amino acids
    values = np.array([np.array(dict_aa_values[aa]) for _, aa in idx_to_amino.items()])
    # Convert the matrix to PyTorch's FloatTensor type
    return torch.FloatTensor(values)


# Define amino acid mapping
amino_acids = 'ARNDCQEGHILKMFPSTWYV'
idx_to_amino = {index: amino for index, amino in enumerate(amino_acids)}
aa_to_idx = {amino: index for index, amino in idx_to_amino.items()}  # Create reverse mapping

# Load Kidera factors
kidera_embedding = load_kidera_values(r"/media/ubuntu/2t/zhangzhi/Project01/Utils/kidera.csv", idx_to_amino)

# Create embedding layer and initialize it with Kidera factors
embedding_layer = torch.nn.Embedding(num_embeddings=len(idx_to_amino), embedding_dim=10)
embedding_layer.weight.data.copy_(kidera_embedding)


# Define a function to convert amino acid sequences to index lists
def sequence_to_indices(sequence, aa_to_idx):
    indices = []
    for aa in sequence:
        if aa in aa_to_idx:
            indices.append(aa_to_idx[aa])
        else:
            print(f"Warning: Amino acid '{aa}' not found in mapping.")
    return indices


# Define a function to process FASTA files and generate Kidera embeddings
def generate_kidera_from_fasta(protein_folder_path):
    # Get FASTA files
    for file_name in os.listdir(protein_folder_path):
        if file_name.endswith(".fasta"):
            fasta_file = os.path.join(protein_folder_path, file_name)

            # Read FASTA file
            with open(fasta_file, "r") as f:
                lines = f.readlines()
                if len(lines) < 2:
                    print(f"Warning: FASTA file '{fasta_file}' does not contain enough lines.")
                    continue
                protein_name = lines[0].strip()[1:]
                protein_sequence = lines[1].strip().upper()  # Ensure the sequence is uppercase

                # Convert amino acid sequence to indices
                indices = sequence_to_indices(protein_sequence, aa_to_idx)
                indices_tensor = torch.LongTensor(indices)

                # Get Kidera embedding using the embedding layer
                embedded_seq = embedding_layer(indices_tensor)

                # Detach tensor and convert to NumPy array
                embedded_seq_np = embedded_seq.detach().numpy()

                # Save as a NumPy array file
                feature_file = os.path.join(protein_folder_path, f"{protein_name}_Kidera.npy")
                np.save(feature_file, embedded_seq_np)
                print(f"Feature matrix has been written to {feature_file}")


# Main function to process all protein folders in the specified directory
def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py protein_folder")
        sys.exit(1)

    protein_folder = sys.argv[1]

    # Process each protein folder
    for folder in os.listdir(protein_folder):
        protein_folder_path = os.path.join(protein_folder, folder)
        if os.path.isdir(protein_folder_path):
            generate_kidera_from_fasta(protein_folder_path)


if __name__ == '__main__':
    main()
