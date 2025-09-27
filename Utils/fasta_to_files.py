import sys
import os


def fastaBuilder(dataset, output):
    with open(dataset, "r") as fin:
        while True:
            line_PID = fin.readline().strip()
            line_Pseq = fin.readline().strip()
            line_label = fin.readline().strip()

            if not line_Pseq:
                break

            # Check if protein sequence length is valid
            if len(line_Pseq) < 10240 and len(line_Pseq) > 1:
                print("Processing:", line_PID)

                # Modify protein name to remove special characters
                protein_name = line_PID.replace('>', '')  # Remove '>' character

                # Create directory for each protein if not already exist
                protein_dir = os.path.join(output, protein_name)
                os.makedirs(protein_dir, exist_ok=True)

                # Write sequence and label to file if not already exist
                fasta_file = os.path.join(protein_dir, f"{protein_name}.fasta")
                if not os.path.exists(fasta_file):
                    with open(fasta_file, "w") as w:
                        w.write('>' + protein_name + '\n')
                        w.write(line_Pseq + '\n')
                        w.write(line_label + '\n')
                else:
                    print("File already exists for:", protein_name)
            else:
                print("Invalid sequence length for:", line_PID)


def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py input_file output_directory")
        sys.exit(1)

    dataset = sys.argv[1]
    output = sys.argv[2]

    print("Input dataset:", dataset)
    print("Output directory:", output)

    fastaBuilder(dataset, output)


if __name__ == '__main__':
    main()
