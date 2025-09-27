import os, sys
import numpy as np
from Bio import pairwise2


def process_dssp(dssp_file):
    aa_type = "ACDEFGHIKLMNPQRSTVWY"
    SS_type = "HBEGITSC"
    rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                185, 160, 145, 180, 225, 115, 140, 155, 255, 230]

    with open(dssp_file, "r") as f:
        lines = f.readlines()

    seq = ""
    dssp_feature = []

    p = 0
    while lines[p].strip()[0] != "#":
        p += 1
    for i in range(p + 1, len(lines)):
        aa = lines[i][13]
        if aa == "!" or aa == "*":
            continue
        seq += aa
        SS = lines[i][16]
        if SS == " ":
            SS = "C"
        SS_vec = np.zeros(9)  # The last dim represents "Unknown" for missing residues
        SS_vec[SS_type.find(SS)] = 1
        PHI = float(lines[i][103:109].strip())
        PSI = float(lines[i][109:115].strip())
        ACC = float(lines[i][34:38].strip())
        ASA = min(100, round(ACC / rASA_std[aa_type.find(aa)] * 100)) / 100
        dssp_feature.append(np.concatenate((np.array([PHI, PSI, ASA]), SS_vec)))

    return seq, dssp_feature


def match_dssp(seq, dssp, ref_seq):
    alignments = pairwise2.align.globalxx(ref_seq, seq)
    ref_seq = alignments[0].seqA
    seq = alignments[0].seqB

    SS_vec = np.zeros(9)  # The last dim represent "Unknown" for missing residues
    SS_vec[-1] = 1
    padded_item = np.concatenate((np.array([360, 360, 0]), SS_vec))

    new_dssp = []
    for aa in seq:
        if aa == "-":
            new_dssp.append(padded_item)
        else:
            new_dssp.append(dssp.pop(0))

    matched_dssp = []
    for i in range(len(ref_seq)):
        if ref_seq[i] == "-":
            continue
        matched_dssp.append(new_dssp[i])

    return matched_dssp


def transform_dssp(dssp_feature):
    dssp_feature = np.array(dssp_feature)
    angle = dssp_feature[:, 0:2]
    ASA_SS = dssp_feature[:, 2:]

    radian = angle * (np.pi / 180)
    dssp_feature = np.concatenate([np.sin(radian), np.cos(radian), ASA_SS], axis=1)

    return dssp_feature


def get_dssp(ref_seq, dssp_file, save_file):
    dssp_seq, dssp_matrix = process_dssp(dssp_file)
    if dssp_seq != ref_seq:
        dssp_matrix = match_dssp(dssp_seq, dssp_matrix, ref_seq)

    np.save(save_file, transform_dssp(dssp_matrix))


def main():
    # Check if directory argument is provided
    if len(sys.argv) != 2:
        print("Usage: python script.py protein_folder")
        sys.exit(1)

    # Get protein folder from command line argument
    protein_folder = sys.argv[1]

    # Check if protein folder exists
    if not os.path.isdir(protein_folder):
        print(f"Error: {protein_folder} is not a valid directory.")
        sys.exit(1)

    # Generate FASTA file for each protein folder
    for folder_name in os.listdir(protein_folder):
        protein_subfolder = os.path.join(protein_folder, folder_name)
        if os.path.isdir(protein_subfolder):
            fasta_file = os.path.join(protein_subfolder, f"{folder_name}.fasta")
            pdb_file = os.path.join(protein_subfolder, f"{folder_name}.pdb")
            dssp_file = os.path.join(protein_subfolder, f"{folder_name}.dssp")
            save_file = os.path.join(protein_subfolder, f"{folder_name}_dssp.npy")

            if os.path.isfile(fasta_file) and os.path.isfile(pdb_file) and os.path.isfile(dssp_file):
                with open(fasta_file, "rb") as f:
                    lines = f.readlines()
                    ref_seq = lines[1].strip().decode()
                    get_dssp(ref_seq, dssp_file, save_file)
                print(f"DSSP features saved to {save_file}")
            else:
                print(f"Error: Files not found for protein {folder_name}")


if __name__ == "__main__":
    main()
