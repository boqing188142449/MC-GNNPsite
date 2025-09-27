from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import sys
import os
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Local cache directory
cache_dir = r"/media/2t/zhangzhi/Project01/model_cache"
# Load tokenizer from local directory
tokenizer = T5Tokenizer.from_pretrained(cache_dir, do_lower_case=False)
# Load model from local directory
model = T5EncoderModel.from_pretrained(cache_dir).to(device)
# Use full precision on CPU only
model = model.to(torch.float32) if device == torch.device("cpu") else model


def generate_embeddings(sequence):
    # Replace rare/ambiguous amino acids with X and insert spaces between amino acids
    processed_sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))

    # Tokenize and pad to the longest sequence length in the batch
    ids = tokenizer([processed_sequence], add_special_tokens=True, padding="longest")

    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # Generate embedding representation
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

    length = len(sequence)  # Original sequence length
    embeddings = embedding_repr.last_hidden_state[0,
                 :length].cpu().numpy()  # Extract embeddings of the corresponding length
    print(embeddings.shape)

    return embeddings


# Function to process each protein folder
def process_protein_folder(protein_folder_path):
    fasta_file = os.path.join(protein_folder_path, f"{os.path.basename(protein_folder_path)}.fasta")
    protein_name = os.path.basename(protein_folder_path)

    # Check if feature file already exists
    feature_file = os.path.join(protein_folder_path, f"{protein_name}_prott5.npy")

    # Read protein sequence from FASTA file
    with open(fasta_file, "r") as f:
        lines = f.readlines()
        protein_sequence = lines[1].strip()  # Assuming the sequence is in the second line

    # Generate feature matrix for the protein sequence
    feature_matrix = generate_embeddings(protein_sequence)

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
