import os
import sys


def generate_dssp_files(protein_directory):
    for folder in os.listdir(protein_directory):
        folder_path = os.path.join(protein_directory, folder)
        if os.path.isdir(folder_path):
            pdb_file = None
            for file in os.listdir(folder_path):
                if file.endswith('.pdb'):
                    pdb_file = os.path.join(folder_path, file)
                    break
            if pdb_file:
                dssp_file = os.path.join(folder_path, f"{folder}.dssp")
                # Use mkdssp function to generate DSSP file
                os.system(f"dssp -i {pdb_file} -o {dssp_file}")
                print(f"DSSP file generated for protein {folder}")
            else:
                print(f"No PDB file found in folder {folder}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py protein_folder")
        sys.exit(1)

    protein_directory = sys.argv[1]

    # Call function to generate DSSP files
    generate_dssp_files(protein_directory)
