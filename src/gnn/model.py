import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SimpleMLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        dimensions = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(input_dim, output_dim)
            for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:])
        )
        self.bns = torch.nn.ModuleList(
            torch.nn.BatchNorm1d(hidden_channels) for _ in range(num_layers - 1)
        )
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        for layer, batch_norm in zip(self.layers[:-1], self.bns):
            x = F.dropout(F.relu(batch_norm(layer(x))), p=self.dropout, training=self.training)
        return self.layers[-1](x)


class SimpleGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            
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
            x = self.bns[i](x)  # Apply BatchNorm
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # Last layer (No BatchNorm, No ReLU usually for logits)
        x = self.convs[-1](x, edge_index)
        return x
