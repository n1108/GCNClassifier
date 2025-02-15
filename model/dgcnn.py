import torch
import torch.nn as nn
import torch.nn.functional as F

def get_graph_feature(x, k=20):
    """
    Extracts graph features by computing pairwise distances and concatenating node features 
    and differences with their nearest neighbors.

    Args:
        x: Node features of shape [B, C, N], where B is the batch size, 
           C is the number of input channels (features per node), and N is the number of nodes.
        k: The number of nearest neighbors to consider for each node.

    Returns:
        A tensor of shape [B, 2*C, N, k], where C is the number of input features, 
        containing the concatenated differences of each node with its k nearest neighbors.
    """
    B, C, N = x.size()
    # Compute pairwise distances between nodes
    xx = x.pow(2).sum(dim=1, keepdim=True)  # [B, 1, N]
    dist = xx + xx.transpose(1, 2) - 2 * torch.matmul(x.transpose(1, 2), x)  # [B, N, N]

    # Get the indices of the k nearest neighbors for each node
    _, idx = dist.topk(k=k, dim=-1, largest=False)

    # Adjust indices to account for the batch dimension
    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx = idx + idx_base  # Apply offset to indices

    # Flatten the node features and select the neighbors' features
    x_flat = x.transpose(1, 2).contiguous().view(B*N, C)  # [B*N, C]
    feature = x_flat[idx.view(-1), :].view(B, N, k, C)    # [B, N, k, C]

    # Expand x to match the feature tensor for concatenation
    x = x.transpose(1, 2).unsqueeze(2).expand(-1, -1, k, -1)  # [B, N, k, C]

    # Concatenate the differences (x_j - x_i) and the original features (x_i)
    feature = feature - x
    feature = torch.cat((x, feature), dim=-1)  # [B, N, k, 2*C]
    feature = feature.permute(0, 3, 1, 2)      # [B, 2*C, N, k]
    return feature


class DGCNNClassifier(nn.Module):
    """
    DGCNN (Dynamic Graph CNN) Classifier for point cloud classification.
    
    This model uses edge convolution layers to process the input point cloud and extract 
    features using graph-based convolutions, followed by a classifier for final predictions.

    Args:
        k: The number of nearest neighbors to consider for each node during edge convolutions.
        emb_dims: The dimensionality of the final embedding.
        num_classes: The number of output classes for classification.
        dropout: Dropout rate for regularization.

    Input:
        x: Node features (point cloud) of shape [B, C, N], where B is the batch size, 
           C is the number of features, and N is the number of points.
        A: The adjacency matrix of shape [B, N, N] (optional, can be None).

    Output:
        logits: The output class logits of shape [B, num_classes].
    """
    def __init__(self, k=20, emb_dims=1024, num_classes=40, dropout=0.5):
        super(DGCNNClassifier, self).__init__()
        self.k = k
        self.emb_dims = emb_dims
        self.num_classes = num_classes
        self.dropout = dropout
        
        # EdgeConv Layer 1: Input is 6 channels (node features + difference), output is 64 channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # EdgeConv Layer 2: Input is 64 channels, output is 64 channels
        self.conv2 = nn.Sequential(
            nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # EdgeConv Layer 3: Input is 64 channels, output is 128 channels
        self.conv3 = nn.Sequential(
            nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # EdgeConv Layer 4: Input is 128 channels, output is 256 channels
        self.conv4 = nn.Sequential(
            nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Final convolution layer to produce embedding dimension
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.emb_dims),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Fully connected layers for classification
        self.linear1 = nn.Linear(self.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=self.dropout)

        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=self.dropout)

        self.linear3 = nn.Linear(256, self.num_classes)

    def forward(self, x, A=None):
        """
        Forward pass through the DGCNN model.

        Args:
            x: Node features (point cloud) of shape [B, C, N], where B is the batch size,
               C is the number of features per node, and N is the number of nodes (points).
            A: The adjacency matrix (optional, not used in this specific model).

        Returns:
            logits: The class logits of shape [B, num_classes].
        """
        # Ensure x is of shape [B, 3, N] (3D coordinates)
        if x.shape[1] != 3:
            x = x.permute(0, 2, 1).contiguous()

        # EdgeConv Block 1
        x1 = get_graph_feature(x, k=self.k)   # [B, 6, N, k]
        x1 = self.conv1(x1)                   # [B, 64, N, k]
        x1 = x1.max(dim=-1, keepdim=False)[0] # [B, 64, N]

        # EdgeConv Block 2
        x2 = get_graph_feature(x1, k=self.k)  # [B, 128, N, k]
        x2 = self.conv2(x2)                   # [B, 64, N, k]
        x2 = x2.max(dim=-1, keepdim=False)[0] # [B, 64, N]

        # EdgeConv Block 3
        x3 = get_graph_feature(x2, k=self.k)  # [B, 128, N, k]
        x3 = self.conv3(x3)                   # [B, 128, N, k]
        x3 = x3.max(dim=-1, keepdim=False)[0] # [B, 128, N]

        # EdgeConv Block 4
        x4 = get_graph_feature(x3, k=self.k)  # [B, 256, N, k]
        x4 = self.conv4(x4)                   # [B, 256, N, k]
        x4 = x4.max(dim=-1, keepdim=False)[0] # [B, 256, N]

        # Concatenate all layers' features
        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        x_emb = self.conv5(x_cat)  # [B, emb_dims, N]

        # Global pooling: max and mean
        x_max = F.adaptive_max_pool1d(x_emb, 1).view(x.shape[0], -1)  # [B, 1024]
        x_mean = F.adaptive_avg_pool1d(x_emb, 1).view(x.shape[0], -1) # [B, 1024]
        x_final = torch.cat((x_max, x_mean), dim=1)                   # [B, 2048]

        # Fully connected layers for classification
        x = self.linear1(x_final)   # [B, 512]
        x = self.bn6(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dp1(x)

        x = self.linear2(x)         # [B, 256]
        x = self.bn7(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dp2(x)

        x = self.linear3(x)         # [B, num_classes]
        return x
