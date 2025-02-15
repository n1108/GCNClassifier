import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multi-layer perceptron (MLP) with two layers.
    
    The MLP consists of:
        1) A fully connected layer followed by Batch Normalization and ReLU activation.
        2) A second fully connected layer for output.

    Args:
        input_dim: The number of input features.
        hidden_dim: The number of hidden units in the MLP.
        output_dim: The number of output features.

    Input:
        x: Tensor of shape [B, input_dim], where B is the batch size.

    Output:
        The output of the MLP of shape [B, output_dim].
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self, x):
        return self.net(x)


class GINConv(nn.Module):
    """
    Graph Isomorphism Network (GIN) convolution layer.
    
    The GINConv layer computes the following update for each node:
        out = MLP((1 + eps) * X + sum_{neighbors}(X_neighbors))
    
    Args:
        in_channels: The number of input features for each node.
        out_channels: The number of output features for each node.
        eps: A learnable parameter that controls the importance of the node features versus the neighbors' features.

    Input:
        X: Node features of shape [B, N, d_in], where B is the batch size, 
           N is the number of nodes, and d_in is the input feature dimensionality.
        A: The adjacency matrix of shape [B, N, N].

    Output:
        The updated node features of shape [B, N, d_out], where d_out is the output feature dimensionality.
    """
    def __init__(self, in_channels, out_channels, eps=0.0):
        super(GINConv, self).__init__()
        self.mlp = MLP(in_channels, out_channels, out_channels)
        self.eps = nn.Parameter(torch.FloatTensor([eps]))
        
    def forward(self, X, A):
        """
        Forward pass for the GINConv layer.
        
        Args:
            X: Node features of shape [B, N, d_in].
            A: The adjacency matrix of the graph of shape [B, N, N].

        Returns:
            The updated node features of shape [B, N, d_out].
        """
        B, N, d_in = X.size()

        # Aggregate neighbors' features by summing them using the adjacency matrix
        neighbors_sum = torch.matmul(A, X)  # [B, N, d_in]

        # Update node features by combining original features with the neighbors' features
        out = (1 + self.eps) * X + neighbors_sum
        
        # Apply MLP to the updated node features
        out_2d = out.reshape(B*N, d_in)
        out_2d = self.mlp(out_2d)  # [B*N, d_out]
        out = out_2d.reshape(B, N, -1)
        return out


class GINClassifier(nn.Module):
    """
    GIN Classifier for graph-level classification using Graph Isomorphism Network (GIN).
    
    The model applies GINConv layers followed by global pooling and a final classifier.

    Args:
        input_dim: The number of input features for each node.
        hidden_dims: List of hidden dimensions for each GINConv layer.
        num_classes: The number of output classes.
        dropout: Dropout rate for regularization.

    Input:
        X: Node features of shape [B, N, input_dim], where B is the batch size,
           N is the number of nodes, and input_dim is the number of features per node.
        A: The adjacency matrix of the graph of shape [B, N, N].

    Output:
        logits: The output logits of shape [B, num_classes], representing class probabilities.
    """
    def __init__(self, 
                 input_dim=3, 
                 hidden_dims=[64, 128, 256, 512], 
                 num_classes=40, 
                 dropout=0.5):
        super(GINClassifier, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # Build multiple GINConv layers
        self.gin_layers = nn.ModuleList()
        in_dim = input_dim
        for hid_dim in hidden_dims:
            self.gin_layers.append(GINConv(in_dim, hid_dim))
            in_dim = hid_dim
        
        # The readout dimension is the concatenation of all layers' readouts (including input layer)
        self.readout_dim = input_dim + sum(hidden_dims)
        
        # Final classifier layer using MLP
        self.classifier = nn.Sequential(
            nn.Linear(self.readout_dim, self.readout_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.readout_dim, num_classes)
        )

    def forward(self, X, A):
        """
        Forward pass through the GIN classifier.

        Args:
            X: Node features of shape [B, N, input_dim].
            A: The adjacency matrix of the graph of shape [B, N, N].

        Returns:
            logits: The output logits of shape [B, num_classes].
        """
        B, N, _ = X.shape
        
        # Store the graph-level representations for each layer
        layer_readouts = []

        # Global pooling of the input features (max pooling over nodes)
        readout_0 = X.max(dim=1)[0]  # [B, input_dim]
        layer_readouts.append(readout_0)

        out = X
        for gin_layer in self.gin_layers:
            out = gin_layer(out, A)   # [B, N, d_out]
            out = F.relu(out)         # Apply ReLU activation
            out = self.dropout(out)
            
            # Perform global pooling after each GIN layer
            layer_repr = out.max(dim=1)[0]  # [B, d_out]
            layer_readouts.append(layer_repr)
        
        # Concatenate the readouts from all layers (including input layer)
        graph_representation = torch.cat(layer_readouts, dim=-1)  # [B, readout_dim]
        
        # Apply the final classifier to the concatenated readouts
        logits = self.classifier(graph_representation)
        return logits
