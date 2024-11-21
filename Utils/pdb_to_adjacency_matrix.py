import numpy as np
from Bio.PDB import PDBParser
import os
import sys


# Step 1: Read protein structure data from PDB file
def read_pdb(file_path):
    parser = PDBParser()
    structure = parser.get_structure("protein", file_path)
    return structure


# Step 2: Calculate distances between residues
def calculate_distance(structure):
    distances = []
    for model in structure:
        for chain in model:
            for residue1 in chain:
                for residue2 in chain:
                    if residue1 != residue2:
                        try:
                            distance = residue1["CA"] - residue2["CA"]  # Calculate distance between Cα atoms
                            distances.append((residue1.id[1], residue2.id[1], distance))
                        except KeyError:
                            continue  # Skip if a residue is missing Cα atom
    return distances


# Step 3: Build contact matrix and calculate edge features
def build_contact_matrix_and_edge_features(distances, residue_count, cutoff_start, cutoff_end):
    contact_matrix = np.zeros((residue_count, residue_count))
    for residue1_id, residue2_id, distance in distances:
        if cutoff_start < distance <= cutoff_end:
            contact_matrix[residue1_id - 1][residue2_id - 1] = 1
            contact_matrix[residue2_id - 1][residue1_id - 1] = 1  # Symmetric matrix
    np.fill_diagonal(contact_matrix, 1)  # Set elements on diagonal to 1
    return contact_matrix, edge_features


# Step 4: Generate contact matrices and store them in a list
def generate_contact_matrices_and_edge_features(folder_path, thresholds_start=[0.0, 4.0, 8.0, 12.0],
                                                thresholds_end=[4.0, 8.0, 12.0, 16.0]):
    pdb_file = os.path.join(folder_path, f"{os.path.basename(folder_path)}.pdb")
    structure = read_pdb(pdb_file)
    distances = calculate_distance(structure)

    # Calculate total number of residues in the protein
    residue_count = max(max(pair[0], pair[1]) for pair in distances)

    contact_matrices = []
    # Generate contact matrix for each threshold
    for threshold_start, threshold_end in zip(thresholds_start, thresholds_end):
        contact_matrix= build_contact_matrix_features(distances, residue_count,
                                                                               cutoff_start=threshold_start,
                                                                               cutoff_end=threshold_end)
        contact_matrices.append(contact_matrix)
    return contact_matrices


# Step 5: Main function
def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py root_folder_path")
        sys.exit(1)

    root_folder = sys.argv[1]  # Root folder containing protein folders
    for protein_folder in os.listdir(root_folder):
        protein_folder_path = os.path.join(root_folder, protein_folder)
        # Generate contact matrices and edge features, then store them
        contact_matrices= generate_contact_matrices_and_edge_features(protein_folder_path)
        thresholds = [4, 8, 12, 16]  # Threshold list

        # Convert the list of contact matrices to numpy arrays and save them
        for idx, (contact_matrix, edge_features) in enumerate(zip(contact_matrices, all_edge_features)):
            threshold = thresholds[idx]  # Get the actual threshold value
            contact_matrix_file = os.path.join(protein_folder_path, f"{protein_folder}_{threshold}.npy")
            np.save(contact_matrix_file, contact_matrix)
            print(f"Contact matrix (threshold: {threshold}A) has been saved to {contact_matrix_file}")


if __name__ == "__main__":
    main()
