import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.data_utils import build_graph_knn_batch

class TNet(nn.Module):
    """
    TNet: A network module used for learning a transformation matrix to align input features.
    
    This module applies a learnable transformation to the input features, represented as a 
    transformation matrix of size (K x K) for each node. The input is of shape [B, N, K], 
    and the output is the transformed feature matrix for each node of shape [B, N, K].
    
    Args:
        k: The number of features for each node (dimensionality of each input feature).
    """
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        # MLP layers to map each node to higher dimensions, followed by max-pooling to global features
        self.mlp1 = nn.Linear(k, 64)
        self.bn1  = nn.BatchNorm1d(64)
        
        self.mlp2 = nn.Linear(64, 128)
        self.bn2  = nn.BatchNorm1d(128)
        
        self.mlp3 = nn.Linear(128, 1024)
        self.bn3  = nn.BatchNorm1d(1024)
        
        # Fully connected layers to map global features back to a k x k transformation matrix
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)

        # Final output layer, resulting in a k x k transformation matrix
        self.fc3 = nn.Linear(256, k*k)

        # Initialize the output matrix close to the identity matrix to preserve the original feature distribution at the start of training
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        """
        Apply the learned transformation to the input features.

        Args:
            x: Input features of shape [B, N, k], where B is the batch size, 
               N is the number of nodes, and k is the dimensionality of each feature.

        Returns:
            The transformed features of shape [B, N, k].
        """
        B, N, k = x.shape
        
        # Apply MLP and batch normalization to each node's features
        out = self.mlp1(x)              
        out = out.view(B*N, 64)
        out = self.bn1(out)
        out = F.relu(out)
        out = out.view(B, N, 64)

        out = self.mlp2(out)
        out = out.view(B*N, 128)
        out = self.bn2(out)
        out = F.relu(out)
        out = out.view(B, N, 128)

        out = self.mlp3(out)
        out = out.view(B*N, 1024)
        out = self.bn3(out)
        out = F.relu(out)
        out = out.view(B, N, 1024)

        # Perform global max-pooling to get a global feature vector
        out = out.max(dim=1)[0]  # [B, 1024]

        # Map the global feature back to a k x k transformation matrix
        out = self.fc1(out)     
        out = self.bn4(out)
        out = F.relu(out)

        out = self.fc2(out)     
        out = self.bn5(out)
        out = F.relu(out)

        out = self.fc3(out)     

        # Add identity matrix to the transformation matrix to initialize close to identity
        out_2d = out.view(-1, self.k, self.k)
        I = torch.eye(self.k, device=x.device).expand(B, self.k, self.k)
        out_2d = out_2d + I

        # Apply the transformation to the input features
        x_transformed = torch.bmm(x, out_2d)  # [B, N, k]

        return x_transformed


class SAGPoolLayer(nn.Module):
    """
    A single layer of SAGPool (Self-Attention Graph Pooling).
    
    1) Use a hidden GCN layer to compute score features for nodes.
    2) Optionally apply batch normalization and activation.
    3) Project the features to a final score, which can be used for topK or soft gating.
    
    Args:
        in_channels: The number of input features for each node.
        hidden_channels: The number of hidden features in the GCN layer.
        use_bn: Whether to apply Batch Normalization after the GCN layer.
        activation: The activation function used after GCN.
        hard_topk: Whether to use hard topK pooling (True) or soft gating (False).
    """
    def __init__(self, in_channels, hidden_channels=512, 
                 use_bn=True, 
                 activation=nn.ReLU(), hard_topk=False):
        super().__init__()

        # Graph convolution layer to generate score features
        self.score_gcn = GraphConvolution(in_channels, hidden_channels)
        
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm1d(hidden_channels)
        
        self.activation = activation
        
        # Project to final score for each node
        self.score_proj = nn.Linear(hidden_channels, 1)
        
        self.hard_topk = hard_topk  # If False, use soft gating, otherwise use hard topK pooling

    def forward(self, X, A, pool_ratio):
        """
        Perform the forward pass of SAGPool.
        
        Args:
            X: Node features of shape [B, N, in_channels].
            A: The adjacency matrix of the graph, of shape [B, N, N].
            pool_ratio: The ratio of nodes to retain after pooling.

        Returns:
            X_pooled: The pooled node features.
            A_pooled: The re-normalized adjacency matrix after pooling.
        """
        B, N, _ = X.size()

        # Perform graph convolution to obtain node representations (score features)
        H = self.score_gcn(X, A)  # [B, N, hidden_channels]

        # Apply batch normalization if specified
        if self.use_bn:
            H = H.permute(0, 2, 1)     # [B, hidden_channels, N]
            H = self.bn(H)
            H = H.permute(0, 2, 1)     # [B, N, hidden_channels]

        # Apply activation function
        H = self.activation(H)

        # Project to a 1D score for each node
        score = self.score_proj(H).squeeze(-1)  # [B, N]

        if not self.hard_topk:
            # --- Soft gating method ---
            gate = torch.sigmoid(score)  # [B, N], range [0, 1]
            X_pooled = X * gate.unsqueeze(-1)
            A_pooled = A
            return X_pooled, A_pooled
        else:
            # --- Hard topK method ---
            k = int(pool_ratio * N)
            if k < 2:
                return X, A

            # Take top-k nodes based on scores
            _, topk_idx = torch.topk(score, k, dim=1)
            batch_idx = torch.arange(B, device=X.device).unsqueeze(1).repeat(1, k)
            
            # Collect the node features for the selected top-k nodes
            X_pooled = X[batch_idx, topk_idx, :]  # [B, k, in_channels]

            # Collect the subgraph's adjacency matrix
            idx_row = topk_idx.unsqueeze(-1).expand(-1, -1, k) 
            idx_col = topk_idx.unsqueeze(1).expand(-1, k, -1)
            b_idx_3d = batch_idx.unsqueeze(-1).expand(-1, k, k)
            
            A_pooled = A[b_idx_3d, idx_row, idx_col]  # [B, k, k]

            return X_pooled, A_pooled


class GraphConvolution(nn.Module):
    """
    Basic GCN layer: A_norm * X * W
    
    Args:
        in_features: The number of input features for each node.
        out_features: The number of output features for each node.
        bias: Whether to include a bias term in the layer.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights and biases
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, X, A):
        """
        Perform the forward pass of the GCN layer.
        
        Args:
            X: Node features of shape [B, N, in_features].
            A: The adjacency matrix of the graph, of shape [B, N, N].

        Returns:
            The output features of the nodes, of shape [B, N, out_features].
        """
        # Compute normalized adjacency matrix A_norm = D^(-1/2) * (A + I) * D^(-1/2)
        I = torch.eye(A.size(-1), device=A.device)
        A_tilde = A + I  # A + I
        D_tilde = torch.sum(A_tilde, dim=-1)  # [B, N]
        D_inv_sqrt = torch.pow(D_tilde, -0.5) # [B, N]
        D_inv_sqrt = torch.where(torch.isinf(D_inv_sqrt),
                                 torch.zeros_like(D_inv_sqrt),
                                 D_inv_sqrt)   # Avoid inf

        D_inv_sqrt_mat = torch.diag_embed(D_inv_sqrt)  # [B, N, N]
        A_norm = D_inv_sqrt_mat @ A_tilde @ D_inv_sqrt_mat  # [B, N, N]
        
        # Apply weight to input features
        support = torch.matmul(X, self.weight)  # [B, N, out_features]
        
        # Compute output by multiplying normalized adjacency with weighted features
        output = torch.matmul(A_norm, support)  # [B, N, out_features]
        
        if self.bias is not None:
            output = output + self.bias
        return output
    

class GCNClassifier(nn.Module):
    """
    GCN Classifier: A graph convolutional network for classification tasks.

    Args:
        input_dim: The number of features for each node.
        hidden_dims: A list of integers specifying the number of hidden units per layer.
        num_classes: The number of output classes.
        dropout: Dropout rate.
        use_bn: Whether to use Batch Normalization in the layers.
        use_residual: Whether to use residual connections in the layers.
        use_hierarchical_pooling: Whether to use hierarchical pooling.
        use_TNet: Whether to use TNet for feature alignment.
        use_virtual_node: Whether to add virtual nodes.
        pool_ratios: A list of ratios indicating the fraction of nodes to retain after pooling for each layer.
    """
    def __init__(
        self, 
        input_dim=3, 
        hidden_dims=[128, 256, 512], 
        num_classes=40, 
        dropout=0.5,
        use_bn=False,
        use_residual=False,
        use_hierarchical_pooling=False,
        use_TNet=False,
        use_virtual_node=False,
        pool_ratios = [0.8, 0.8, 0.6],
    ):
        super(GCNClassifier, self).__init__()
        self.use_residual = use_residual
        self.use_hierarchical_pooling = use_hierarchical_pooling
        self.dropout_rate = dropout
        self.use_TNet = use_TNet
        self.use_virtual_node = use_virtual_node

        # Initialize TNet if specified
        if self.use_TNet:
            self.input_transform = TNet(k=input_dim)
        
        # Initialize virtual node embedding if specified
        if self.use_virtual_node:
            self.virtualnode_embedding = nn.Parameter(
                torch.zeros(1, 1, input_dim)
            )
        
        # Initialize hierarchical pooling layers if specified
        if self.use_hierarchical_pooling:
            if len(pool_ratios) < len(hidden_dims):
                while len(pool_ratios) < len(hidden_dims):
                    pool_ratios.append(pool_ratios[-1])
            elif len(pool_ratios) > len(hidden_dims):
                pool_ratios = pool_ratios[:len(hidden_dims)]
            self.pool_ratios = pool_ratios

        # Define the layers for the GCN model
        gc_layers = []
        bn_layers = []
        res_proj = []
        in_dim = input_dim
        
        for out_dim in hidden_dims:
            gc_layers.append(GraphConvolution(in_dim, out_dim))
            if use_bn:
                bn_layers.append(nn.BatchNorm1d(out_dim))
            else:
                bn_layers.append(None)
            
            if use_residual:
                proj = nn.Linear(in_dim, out_dim, bias=False)
                nn.init.xavier_uniform_(proj.weight)
                res_proj.append(proj)
            else:
                res_proj.append(None)

            in_dim = out_dim
        
        self.gc_layers = nn.ModuleList(gc_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        self.res_proj  = nn.ModuleList(res_proj)

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)

        if self.use_hierarchical_pooling:
            sag_pool_layers = []
            for out_dim in hidden_dims:
                sag_pool_layers.append(SAGPoolLayer(out_dim))
            self.sag_pool_layers = nn.ModuleList(sag_pool_layers)

            self.multiscale_dim = sum(hidden_dims)

            transform_dim = 512
            self.pool_transform = nn.Sequential(
                nn.Linear(self.multiscale_dim, transform_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout)
            )
            self.classifier = nn.Linear(transform_dim, num_classes)
        else:
            self.classifier = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, X, A):
        """
        Forward pass through the GCN model.
        
        Args:
            X: Node features of shape [B, N, input_dim].
            A: The adjacency matrix of the graph, of shape [B, N, N].

        Returns:
            logits: The output class logits of shape [B, num_classes].
        """
        B, N, D = X.shape

        # Apply virtual node embedding if specified
        if self.use_virtual_node:
            vnode = self.virtualnode_embedding.repeat(B, 1, 1)
            X = torch.cat([X, vnode], dim=1)
            
            A_new = torch.zeros((B, N+1, N+1), dtype=A.dtype, device=A.device)
            A_new[:, :N, :N] = A
            A_new[:, :N,  N] = 1.0
            A_new[:,  N, :N] = 1.0
            A = A_new
            N = N + 1

        if self.use_TNet:
            X = self.input_transform(X)
            A = build_graph_knn_batch(X, k=20)

        out = X
        out_scales = []

        for i in range(len(self.gc_layers)):
            gc_layer = self.gc_layers[i]
            bn_layer = self.bn_layers[i]
            proj     = self.res_proj[i]
            if self.use_hierarchical_pooling:
                sag_pool = self.sag_pool_layers[i]
                pool_ratio = self.pool_ratios[i]

            out = self._gcn_block(out, A, gc_layer, bn_layer, proj)

            if self.use_hierarchical_pooling:
                out, A = sag_pool(out, A, pool_ratio)
        
        if self.use_virtual_node:
            vnode_feature = out[:, -1, :]
            out_pooled = vnode_feature
        else:
            out_pooled = torch.max(out, dim=1).values
        
        logits = self.classifier(out_pooled)
        return logits

    def _gcn_block(self, X, A, gc_layer, bn_layer, res_proj=None):
        """
        A single GCN block: GCN -> BN -> (Residual) -> ReLU -> Dropout.
        
        Args:
            X: The input features of shape [B, N, in_features].
            A: The adjacency matrix of shape [B, N, N].
            gc_layer: The GCN layer to apply.
            bn_layer: The BatchNorm layer (optional).
            res_proj: The residual projection layer (optional).

        Returns:
            The output of the block after applying the transformations.
        """
        residual = X
        out = gc_layer(X, A)
        
        if bn_layer is not None:
            out = out.permute(0, 2, 1)
            out = bn_layer(out)
            out = out.permute(0, 2, 1)

        if self.use_residual:
            if residual.shape[-1] == out.shape[-1]:
                out = out + residual
            elif res_proj is not None:
                residual = residual.reshape(-1, residual.shape[-1])
                residual = res_proj(residual)
                residual = residual.reshape(X.size(0), X.size(1), -1)
                out = out + residual

        out = self.relu(out)
        out = self.dropout(out)
        return out
