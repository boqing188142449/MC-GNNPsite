import sys
import os
import numpy as np
import torch
import esm

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results


# Function to generate feature matrix for a protein sequence
def generate_feature_matrix(sequence):
    # Prepare data
    data = [('tmp_protein', sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1: tokens_len - 1])

    return sequence_representations[0]


# Function to process each protein folder
def process_protein_folder(protein_folder_path):
    fasta_file = os.path.join(protein_folder_path, f"{os.path.basename(protein_folder_path)}.fasta")
    protein_name = os.path.basename(protein_folder_path)

    # Check if feature file already exists
    feature_file = os.path.join(protein_folder_path, f"{protein_name}_esm-2.npy")

    # Read protein sequence from fasta file
    with open(fasta_file, "r") as f:
        lines = f.readlines()
        protein_sequence = lines[1].strip()  # Assuming the sequence is in the second line

    # Generate feature matrix for the protein sequence
    feature_matrix = generate_feature_matrix(protein_sequence)

    # Save feature matrix to a numpy file
    np.save(feature_file, feature_matrix)
    print(f"Feature matrix has been written to {feature_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py protein_folder_path")
        sys.exit(1)

    protein_folder_path = sys.argv[1]

    # Process each protein folder
    for folder in os.listdir(protein_folder_path):
        protein_subfolder_path = os.path.join(protein_folder_path, folder)
        if os.path.isdir(protein_subfolder_path):
            process_protein_folder(protein_subfolder_path)


if __name__ == '__main__':
    main()
