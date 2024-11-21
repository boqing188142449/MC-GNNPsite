import os
import sys
import numpy as np


def combine_features(protein_folder):
    # Iterate through protein subfolders
    for folder_name in os.listdir(protein_folder):
        protein_subfolder = os.path.join(protein_folder, folder_name)
        if os.path.isdir(protein_subfolder):
            prott5_file = os.path.join(protein_subfolder, f"{folder_name}_prott5.npy")
            esm_file = os.path.join(protein_subfolder, f"{folder_name}_esm-2.npy")
            dssp_file = os.path.join(protein_subfolder, f"{folder_name}_dssp.npy")
            kidera_file = os.path.join(protein_subfolder, f"{folder_name}_Kidera.npy")
            feature_file = os.path.join(protein_subfolder, f"{folder_name}_feature.npy")

            # Check if files exist
            if os.path.isfile(prott5_file) and os.path.isfile(esm_file) and os.path.isfile(
                    dssp_file) and os.path.isfile(kidera_file):
                try:
                    # Load feature files
                    prott5_feature = np.load(prott5_file)
                    esm_feature = np.load(esm_file)
                    dssp_feature = np.load(dssp_file)
                    kidera_feature = np.load(kidera_file)

                    # Combine features
                    combined_feature = np.concatenate((prott5_feature, esm_feature, dssp_feature, kidera_feature),
                                                      axis=1)

                    # Save combined features
                    np.save(feature_file, combined_feature)
                    print(f"Combined feature saved to {feature_file}")
                except Exception as e:
                    print(f"Error loading or combining features for protein {folder_name}: {e}")
            else:
                # Print missing files
                missing_files = [f for f in [prott5_file, esm_file, dssp_file, kidera_file] if not os.path.isfile(f)]
                print(f"Error: Missing files for protein {folder_name}: {missing_files}")


def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python script.py protein_folder")
        sys.exit(1)

    # Get protein folder path
    protein_folder = sys.argv[1]

    # Combine features
    combine_features(protein_folder)


if __name__ == "__main__":
    main()
