import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, HeteroConv
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
import random

# Define the InterleavedGCNN model using HeteroConv
class InterleavedGCNN(nn.Module):
    def __init__(self, var_in_channels, cons_in_channels, hidden_channels, num_layers, edge_attr_dim):
        super(InterleavedGCNN, self).__init__()
        self.hidden_channels = hidden_channels

        # Embedding layers for variable and constraint nodes
        self.embedding_var = nn.Linear(var_in_channels, hidden_channels)
        self.embedding_cons = nn.Linear(cons_in_channels, hidden_channels)

        # Define the hetero convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('var_nodes', 'in', 'constr_nodes'): MessagePassingLayer(
                    hidden_channels, hidden_channels, edge_attr_dim),
                ('constr_nodes', 'rev_in', 'var_nodes'): MessagePassingLayer(
                    hidden_channels, hidden_channels, edge_attr_dim),
            }, aggr='sum')
            self.convs.append(conv)

        # Final classifier for variable nodes
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        # Get initial node features
        x_dict = {
            'var_nodes': data['var_nodes'].x,
            'constr_nodes': data['constr_nodes'].x
        }

        # Apply embeddings
        x_dict['var_nodes'] = self.embedding_var(x_dict['var_nodes'])
        x_dict['constr_nodes'] = self.embedding_cons(x_dict['constr_nodes'])

        edge_index_dict = data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict

        # Apply hetero convolution layers
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

        # Final classification on variable nodes
        out = self.classifier(x_dict['var_nodes'])
        return out.squeeze(-1)

# Define the message passing layer
class MessagePassingLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_attr_dim):
        super(MessagePassingLayer, self).__init__(aggr='add')
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + edge_attr_dim, out_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # Concatenate node features with edge attributes
        msg_input = torch.cat([x_j, edge_attr], dim=-1)
        msg = self.mlp(msg_input)
        return msg