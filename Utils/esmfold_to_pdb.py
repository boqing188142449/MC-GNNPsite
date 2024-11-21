import sys
import os
import torch
import esm


def generate_pdb_from_fasta(protein_folder, device):
    # Find FASTA file in the protein folder
    for file_name in os.listdir(protein_folder):
        if file_name.endswith(".fasta"):
            fasta_file = os.path.join(protein_folder, file_name)
            # Read fasta file
            with open(fasta_file, "r") as f:
                lines = f.readlines()
                protein_name = lines[0].strip()[1:]
                protein_sequence = lines[1].strip()

            # Write pdb to file if not exists
            pdb_file = os.path.join(protein_folder, f"{protein_name}.pdb")
            if not os.path.exists(pdb_file):
                # Generate pdb
                print(protein_name)
                print(len(protein_sequence))
                model = esm.pretrained.esmfold_v1()
                model = model.eval().to(device)  # Load model to selected device
                with torch.no_grad():
                    output = model.infer_pdb(protein_sequence)
                with open(pdb_file, "w") as f:
                    f.write(output)
                print(f"Generated pdb for {protein_name} in {protein_folder}")
            else:
                print(f"PDB file {protein_name}.pdb already exists in {protein_folder}. Skipping...")
            break  # Stop after finding the first FASTA file


def main():
    if len(sys.argv) < 2:
        print("Usage: python EsmfoldToPdb.py protein_folder")
        sys.exit(1)

    protein_folder = sys.argv[1]
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # Process each protein folder
    for folder in os.listdir(protein_folder):
        protein_folder_path = os.path.join(protein_folder, folder)
        if os.path.isdir(protein_folder_path):
            generate_pdb_from_fasta(protein_folder_path, device)


if __name__ == '__main__':
    main()
