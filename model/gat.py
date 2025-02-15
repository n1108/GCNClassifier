import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Single-head Graph Attention (GAT) Layer.

    Args:
        in_features: The number of input features for each node.
        out_features: The number of output features for each node.
        alpha: The negative slope coefficient for LeakyReLU activation.
    
    Input:
        X: Node features of shape [B, N, in_features], where B is the batch size, 
           N is the number of nodes, and in_features is the feature dimensionality.
        A: The adjacency matrix of the graph of shape [B, N, N], where each entry
           represents the connectivity between nodes.

    Output:
        The updated node features of shape [B, N, out_features].
    """
    def __init__(self, in_features, out_features, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        
        # Learnable parameters: weight matrix (W) and attention coefficients (a)
        self.W = nn.Parameter(torch.empty((in_features, out_features)))
        self.a = nn.Parameter(torch.empty((2 * out_features, 1)))
        
        # Initialize weights using Xavier uniform initialization
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
        
        # LeakyReLU activation function for attention mechanism
        self.leaky_relu = nn.LeakyReLU(self.alpha)

    def forward(self, X, A):
        """
        Forward pass for the graph attention layer.
        
        Args:
            X: Node features of shape [B, N, in_features].
            A: The adjacency matrix of the graph of shape [B, N, N].
        
        Returns:
            The updated node features of shape [B, N, out_features].
        """
        B, N, _ = X.shape

        # 1) Linear transformation of input features
        Wh = torch.matmul(X, self.W)  # [B, N, out_features]

        # 2) Compute attention coefficients e_ij between nodes i and j
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, out_features]
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, out_features]
        e = torch.cat([Wh_i, Wh_j], dim=-1)           # [B, N, N, 2*out_features]
        e = self.leaky_relu(torch.matmul(e, self.a).squeeze(-1))  # [B, N, N]

        # 3) Mask non-adjacent nodes (set their attention coefficient to a large negative value)
        zero_mask = -9e15 * (1.0 - A)
        attention = F.softmax(e + zero_mask, dim=-1)  # [B, N, N]

        # 4) Compute the weighted sum of neighbors' features based on attention coefficients
        h_prime = torch.matmul(attention, Wh)         # [B, N, out_features]
        return h_prime


class MultiHeadGATLayer(nn.Module):
    """
    Multi-head Graph Attention (GAT) Layer.

    This layer performs multiple independent graph attention operations and 
    either concatenates or averages the outputs based on the `concat` parameter.

    Args:
        in_features: The number of input features for each node.
        out_features: The number of output features for each node.
        num_heads: The number of attention heads.
        alpha: The negative slope coefficient for LeakyReLU activation.
        concat: If True, the outputs of all attention heads are concatenated.
                If False, the outputs are averaged.
    
    Input:
        X: Node features of shape [B, N, in_features].
        A: The adjacency matrix of the graph of shape [B, N, N].

    Output:
        The updated node features after applying multi-head attention.
        The shape will be [B, N, out_features * num_heads] if `concat=True`,
        or [B, N, out_features] if `concat=False`.
    """
    def __init__(self, in_features, out_features, num_heads=4, alpha=0.2, concat=True):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features, alpha)
            for _ in range(num_heads)
        ])
        self.concat = concat

    def forward(self, X, A):
        """
        Forward pass for the multi-head GAT layer.
        
        Args:
            X: Node features of shape [B, N, in_features].
            A: The adjacency matrix of the graph of shape [B, N, N].
        
        Returns:
            The updated node features after applying multi-head attention.
        """
        head_outs = [head(X, A) for head in self.heads]
        if self.concat:
            # Concatenate the output of all attention heads
            return torch.cat(head_outs, dim=-1)  # [B, N, out_features * num_heads]
        else:
            # Average the output of all attention heads
            return torch.mean(torch.stack(head_outs), dim=0)  # [B, N, out_features]


class GATClassifier(nn.Module):
    """
    Graph Attention Network (GAT) Classifier.

    This model applies multi-head graph attention layers and a final classifier for node classification.

    Args:
        input_dim: The number of input features for each node.
        hidden_dim: The number of hidden features for each node in the attention layers.
        num_heads: A list of the number of attention heads for each layer.
        num_classes: The number of output classes.
        alpha: The negative slope coefficient for LeakyReLU activation.
        dropout: The dropout rate.
    
    Input:
        X: Node features of shape [B, N, input_dim], where B is the batch size,
           N is the number of nodes, and input_dim is the number of features per node.
        A: The adjacency matrix of the graph of shape [B, N, N].

    Output:
        logits: The output logits of shape [B, num_classes], where `num_classes` is the number of output classes.
    """
    def __init__(self, 
                 input_dim=3,
                 hidden_dim=64,
                 num_heads=[4, 4],
                 num_classes=40,
                 alpha=0.2,
                 dropout=0.5):
        super(GATClassifier, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # 1) Multi-head attention layer 1 with concatenation, output size = hidden_dim * num_heads[0]
        self.gat1 = MultiHeadGATLayer(input_dim, hidden_dim, num_heads[0], alpha, concat=True)
        
        # 2) Multi-head attention layer 2 with averaging, output size = hidden_dim
        self.gat2 = MultiHeadGATLayer(hidden_dim * num_heads[0],
                                      hidden_dim,
                                      num_heads[1],
                                      alpha,
                                      concat=False)

        # Final classification layer
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, X, A):
        """
        Forward pass through the GAT classifier.
        
        Args:
            X: Node features of shape [B, N, input_dim].
            A: The adjacency matrix of the graph of shape [B, N, N].
        
        Returns:
            logits: The output logits of shape [B, num_classes].
        """
        # Apply the first multi-head GAT layer
        out = self.gat1(X, A)    # [B, N, hidden_dim * num_heads[0]]
        out = F.elu(out)
        out = self.dropout(out)
        
        # Apply the second multi-head GAT layer
        out = self.gat2(out, A)  # [B, N, hidden_dim]
        
        # Perform graph-level readout by averaging over all nodes
        out = out.mean(dim=1)  # [B, hidden_dim] 

        # Apply the final classification layer
        logits = self.classifier(out)
        return logits
