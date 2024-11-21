import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm

class GNN(nn.Module):
    def __init__(self, in_features, hidden_features=1024, dropout=0.2):
        super(GNN, self).__init__()

        self.conv1 = SAGEConv(in_features, hidden_features)
        self.bn1 = BatchNorm(hidden_features)
        self.conv2 = SAGEConv(hidden_features, hidden_features // 2)
        self.bn2 = BatchNorm(hidden_features // 2)
        self.conv3 = SAGEConv(hidden_features // 2, hidden_features // 4)
        self.bn3 = BatchNorm(hidden_features // 4)
        self.conv4 = SAGEConv(hidden_features // 4, hidden_features // 8)
        self.bn4 = BatchNorm(hidden_features // 8)
        self.fc_g1 = nn.Linear(hidden_features // 8, hidden_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = F.relu(self.bn4(self.conv4(x, edge_index)))
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        return x

class AttentionModule(nn.Module):
    def __init__(self, hidden_features):
        super(AttentionModule, self).__init__()
        self.attn_weights = nn.Parameter(torch.ones(4, hidden_features))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x1, x2, x3, x4):
        stack = torch.stack([x1, x2, x3, x4], dim=0)  # [4, batch_size, hidden_features]
        attn_weights = self.softmax(self.attn_weights)  # [4, hidden_features]
        weighted_sum = torch.einsum('ijk,ik->jk', stack, attn_weights)  # [batch_size, hidden_features]
        return weighted_sum

class CombinedGNN(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=1024, dropout=0.2):
        super(CombinedGNN, self).__init__()

        # Initialize four GNN models
        self.gnn1 = GNN(in_features, hidden_features, dropout)
        self.gnn2 = GNN(in_features, hidden_features, dropout)
        self.gnn3 = GNN(in_features, hidden_features, dropout)
        self.gnn4 = GNN(in_features, hidden_features, dropout)

        # Attention module
        self.attention = AttentionModule(hidden_features)

        # Fully connected layers
        self.fc_g2 = nn.Linear(hidden_features, out_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index1, edge_index2, edge_index3, edge_index4):
        # Get outputs from all four GNN models
        x1 = self.gnn1(x, edge_index1)
        x2 = self.gnn2(x, edge_index2)
        x3 = self.gnn3(x, edge_index3)
        x4 = self.gnn4(x, edge_index4)

        # Apply attention to the outputs
        x_combined = self.attention(x1, x2, x3, x4)

        # Fully connected layer
        x_mlp = self.fc_g2(x_combined)
        x_mlp = self.sigmoid(x_mlp)

        return x_mlp
