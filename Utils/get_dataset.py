import os
import numpy as np
import torch
from torch_geometric.data import Data, Dataset, Batch


class GetDataset(Dataset):
    def __init__(self, root):
        self.root = root
        super(GetDataset, self).__init__(root)

    def len(self):
        return len(os.listdir(self.root))

    def get(self, idx):
        protein_folder = os.listdir(self.root)[idx]
        features_path = os.path.join(self.root, protein_folder, f'{protein_folder}_feature.npy')
        adj_path1 = os.path.join(self.root, protein_folder, f'{protein_folder}_4.npy')
        adj_path2 = os.path.join(self.root, protein_folder, f'{protein_folder}_8.npy')
        adj_path3 = os.path.join(self.root, protein_folder, f'{protein_folder}_12.npy')
        adj_path4 = os.path.join(self.root, protein_folder, f'{protein_folder}_16.npy')

        # Load feature matrix and adjacency matrices
        features = np.load(features_path)
        adj1 = np.load(adj_path1)
        adj2 = np.load(adj_path2)
        adj3 = np.load(adj_path3)
        adj4 = np.load(adj_path4)

        # Load labels from FASTA file
        fasta_path = os.path.join(self.root, protein_folder, f'{protein_folder}.fasta')
        with open(fasta_path, 'r') as fasta_file:
            fasta_lines = fasta_file.readlines()
            label = fasta_lines[2].strip()  # Assume the label is on the third line of the FASTA file

        x = torch.from_numpy(features).float()
        edge_index1 = torch.from_numpy(np.vstack(adj1.nonzero())).long()
        edge_index2 = torch.from_numpy(np.vstack(adj2.nonzero())).long()
        edge_index3 = torch.from_numpy(np.vstack(adj3.nonzero())).long()
        edge_index4 = torch.from_numpy(np.vstack(adj4.nonzero())).long()

        y = torch.tensor([int(char) for char in label], dtype=torch.long)

        return Data(x=x, edge_index1=edge_index1, edge_index2=edge_index2,
                    edge_index3=edge_index3, edge_index4=edge_index4, y=y)

    def get_all(self):
        data_list = []
        for protein_folder in os.listdir(self.root):
            features_path = os.path.join(self.root, protein_folder, f'{protein_folder}_feature.npy')
            adj_path1 = os.path.join(self.root, protein_folder, f'{protein_folder}_4.npy')
            adj_path2 = os.path.join(self.root, protein_folder, f'{protein_folder}_8.npy')
            adj_path3 = os.path.join(self.root, protein_folder, f'{protein_folder}_12.npy')
            adj_path4 = os.path.join(self.root, protein_folder, f'{protein_folder}_16.npy')

            features = np.load(features_path)
            adj1 = np.load(adj_path1)
            adj2 = np.load(adj_path2)
            adj3 = np.load(adj_path3)
            adj4 = np.load(adj_path4)

            fasta_path = os.path.join(self.root, protein_folder, f'{protein_folder}.fasta')
            with open(fasta_path, 'r') as fasta_file:
                fasta_lines = fasta_file.readlines()
                label = fasta_lines[2].strip()

            x = torch.from_numpy(features).float()
            edge_index1 = torch.from_numpy(np.vstack(adj1.nonzero())).long()
            edge_index2 = torch.from_numpy(np.vstack(adj2.nonzero())).long()
            edge_index3 = torch.from_numpy(np.vstack(adj3.nonzero())).long()
            edge_index4 = torch.from_numpy(np.vstack(adj4.nonzero())).long()

            y = torch.tensor([int(char) for char in label], dtype=torch.long)

            data_list.append(Data(x=x, edge_index1=edge_index1, edge_index2=edge_index2,
                                  edge_index3=edge_index3, edge_index4=edge_index4, y=y))

        return data_list

    @staticmethod
    def collate_fn(batch):
        protein_data, labels = zip(*batch)
        protein_batch = Batch.from_data_list(protein_data)
        labels = torch.stack(labels)
        return protein_batch, labels
