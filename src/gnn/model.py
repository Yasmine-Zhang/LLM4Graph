import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SimpleGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        # Output layer
        # Ensure at least 1 layer logic if num_layers=1 could be added, 
        # but usually num_layers >= 2 for GCN
        final_in_dim = hidden_channels if num_layers > 1 else in_channels
        self.convs.append(GCNConv(final_in_dim, out_channels))
        
        self.dropout = dropout

    def forward(self, x, edge_index):
        # Iterate over all layers except the last one
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            # BN/LN could be added here
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # Last layer
        x = self.convs[-1](x, edge_index)
        return x
