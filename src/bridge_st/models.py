import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TransformerConv

class GraphAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, gnn_type='gcn'):
        super(GraphAutoencoder, self).__init__()
        # Select GNN type
        if gnn_type == 'gcn':
            self.gnn_layer = GCNConv
        elif gnn_type == 'gat':
            self.gnn_layer = GATConv
        elif gnn_type == 'graphsage':
            self.gnn_layer = SAGEConv
        elif gnn_type == 'graphtransformer':
            self.gnn_layer = TransformerConv
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")

        self.encoder_layer1 = self.gnn_layer(input_dim, hidden_dim)
        self.encoder_layer2 = self.gnn_layer(hidden_dim, hidden_dim)

        self.decoder_layer1 = self.gnn_layer(hidden_dim, hidden_dim)
        self.decoder_layer2 = self.gnn_layer(hidden_dim, input_dim)

        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x, edge_indices):
        edge_index_1 = edge_indices[0]
        edge_index_2 = edge_indices[1]
        x = self.dropout(x)
        x = F.elu(self.encoder_layer1(x, edge_index_1))
        z = F.elu(self.encoder_layer2(x, edge_index_2))  # hidden representation
        x = F.elu(self.decoder_layer1(z, edge_index_1))
        x = F.relu(self.decoder_layer2(x, edge_index_2))
        return z, x

class NoisePredictor(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(NoisePredictor, self).__init__()
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class NormalEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NormalEncoder, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim*8)
        self.ln1 = torch.nn.LayerNorm(hidden_dim*8)  # Replace BatchNorm with LayerNorm
        self.fc2 = torch.nn.Linear(hidden_dim*8, hidden_dim)
        self.ln2 = torch.nn.LayerNorm(hidden_dim)  # Replace BatchNorm with LayerNorm
        
        # Better initialization with positive bias
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.constant_(self.fc1.bias, 0.1)
        torch.nn.init.constant_(self.fc2.bias, 0.1)

    def forward(self, x):
        # Add debugging prints
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.leaky_relu(x, negative_slope=0.01)  # Try LeakyReLU instead of ELU
        
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        return x

# class NormalEncoder(torch.nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(NormalEncoder, self).__init__()
#         hidden_dim = 256
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ELU(),

#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ELU(),

#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ELU(),
            
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ELU(),

#             nn.Linear(hidden_dim, output_dim),
#             nn.ELU()
#         )

#     def forward(self, x):
#         return self.model(x)

def custom_readout(node_hidden):
    return torch.mean(node_hidden, dim=0)
