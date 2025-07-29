import torch

import torch.nn as nn
# PyTorch Geometric imports
from torch_geometric.nn import GINConv, GCNConv, TransformerConv, global_mean_pool, global_max_pool



class GNNEncoder(nn.Module):
    """Graph Neural Network encoder that processes graph topology with Transformer-style blocks"""
    
    def __init__(self, 
                 node_dim: int = 64,
                 hidden_dim: int = 64,
                 output_dim: int = 64,
                 num_layers: int = 3,
                 gnn_type: str = "GIN",
                 dropout: float = 0.0,
                 num_heads: int = 4,
                 feedforward_dim: int = 256):
        super(GNNEncoder, self).__init__()
        
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.num_heads = num_heads
        
        # Initial node embedding (since our nodes don't have features, we use learnable embeddings)
        self.node_encoder = nn.Sequential(
            nn.Linear(1, node_dim),  # Each node gets a constant feature of 1
            nn.ReLU()
        )
        
        # Build Transformer-style GNN blocks
        self.gnn_blocks = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = node_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            
            # Create a transformer-style block
            block = TransformerGNNBlock(
                in_dim=in_dim,
                out_dim=out_dim,
                gnn_type=gnn_type,
                dropout=dropout,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim
            )
            self.gnn_blocks.append(block)
        
    def forward(self, x, edge_index, batch=None):
        # Initial node features
        x = self.node_encoder(x)
        
        # Apply Transformer-style GNN blocks
        for block in self.gnn_blocks:
            x = block(x, edge_index)
        
        # Global pooling to get graph-level representation
        if batch is not None:
            graph_embedding = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        else:
            # Single graph case
            graph_embedding = torch.cat([
                x.mean(dim=0, keepdim=True),
                x.max(dim=0, keepdim=True)[0]
            ], dim=1)
            
        return graph_embedding, x


class TransformerGNNBlock(nn.Module):
    """Transformer-style block: GNN -> AddNorm -> FeedForward -> AddNorm"""
    
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 gnn_type: str,
                 dropout: float = 0.0,
                 num_heads: int = 4,
                 feedforward_dim: int = 256):
        super(TransformerGNNBlock, self).__init__()
        
        self.gnn_type = gnn_type
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # GNN layer (attention mechanism)
        if gnn_type == "GIN":
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )
            self.gnn_layer = GINConv(mlp)
            self.gnn_out_dim = out_dim
            
        elif gnn_type == "GCN":
            self.gnn_layer = GCNConv(in_dim, out_dim)
            self.gnn_out_dim = out_dim
            
        elif gnn_type == "TCN":
            # With concat=True and num_heads heads, output dimension is out_dim * num_heads
            self.gnn_layer = TransformerConv(
                in_dim, 
                out_dim, 
                heads=num_heads, 
                concat=True  # Changed to True as requested
            )
            self.gnn_out_dim = out_dim * num_heads
            # Project back to expected dimension
            self.projection = nn.Linear(out_dim * num_heads, out_dim)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        # Layer normalization for residual connections
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        
        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(out_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, out_dim)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Linear projection for residual connection if dimensions don't match
        if in_dim != out_dim:
            self.residual_projection = nn.Linear(in_dim, out_dim)
        else:
            self.residual_projection = None
    
    def forward(self, x, edge_index):
        # Store input for residual connection
        residual = x
        
        # Apply GNN layer (attention mechanism)
        x = self.gnn_layer(x, edge_index)
        
        # Handle TransformerConv concatenated output
        if self.gnn_type == "TCN":
            x = self.projection(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # First Add & Norm (residual connection)
        if self.residual_projection is not None:
            residual = self.residual_projection(residual)
        x = self.norm1(x + residual)
        
        # Store for second residual connection
        residual = x
        
        # Feedforward network
        ff_output = self.feedforward(x)
        
        # Second Add & Norm (residual connection)
        x = self.norm2(ff_output + residual)
        
        return x