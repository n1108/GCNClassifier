import torch
import torch.nn as nn
import torch.nn.functional as F

class STN3D(nn.Module):
    """
    3D Spatial Transformer Network (STN).
    
    This module applies a spatial transformation to either input features (d=3) or feature representations (d=64).
    When d=3, it functions as an Input Transformer; when d=64, it acts as a Feature Transformer.

    Args:
        d: The dimensionality of the input or feature space. Typically, d=3 for point clouds and d=64 for feature maps.
    
    Input:
        x: A tensor of shape [B, d, N], where B is the batch size, d is the dimensionality of the input, 
           and N is the number of points or features.

    Output:
        A transformation matrix of shape [B, d, d] that can be used to transform the input or features.
    """
    def __init__(self, d=3):
        super(STN3D, self).__init__()
        self.d = d
        # Convolution layers for learning transformations
        self.conv1 = nn.Conv1d(d, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, d * d)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # Initialize fc3 weights to 0 for adding identity matrix later
        nn.init.constant_(self.fc3.weight, 0)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, x):
        """
        Forward pass through the STN3D layer.

        Args:
            x: Input tensor of shape [B, d, N], where B is the batch size, 
               d is the number of features (3 for input points, 64 for features), 
               and N is the number of points or features.

        Returns:
            A transformation matrix of shape [B, d, d] to transform the input tensor.
        """
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))   # [B, 64, N]
        x = F.relu(self.bn2(self.conv2(x)))   # [B, 128, N]
        x = F.relu(self.bn3(self.conv3(x)))   # [B, 1024, N]

        # Max pooling to obtain a global feature vector
        x, _ = torch.max(x, 2)                # [B, 1024]
        x = F.relu(self.bn4(self.fc1(x)))     # [B, 512]
        x = F.relu(self.bn5(self.fc2(x)))     # [B, 256]

        # Predict transformation matrix (d x d) and add identity matrix
        mat = self.fc3(x)                     # [B, d*d]
        eye = torch.eye(self.d, device=mat.device).view(1, self.d*self.d)
        mat = mat + eye.repeat(B, 1)          # [B, d*d]
        mat = mat.view(-1, self.d, self.d)    # [B, d, d]
        return mat


class PointNetEncoder(nn.Module):
    """
    PointNet Encoder for point cloud feature extraction.

    This module applies spatial and feature transformations, followed by MLP layers to extract global point cloud features.
    
    The architecture consists of:
        1) Input transformation (STN3D) for 3D coordinates.
        2) MLP(64, 64) for initial feature extraction.
        3) Feature transformation (STN3D) for 64-dimensional features.
        4) MLP(64, 128, 1024) for deeper feature extraction.
        5) Global max pooling to aggregate features across all points.

    Args:
        global_feat: Whether to return only the global feature (True) or per-point features (False).
        feature_transform: Whether to use feature transformation (True) or not (False).
    
    Input:
        x: Point cloud tensor of shape [B, 3, N], where B is the batch size, 
           3 is the number of input features (typically x, y, z coordinates), and N is the number of points.

    Output:
        Depending on `global_feat`, returns either global features or both global and per-point features.
    """
    def __init__(self, global_feat=True, feature_transform=True):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3D(d=3)
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STN3D(d=64)

        # First stage: MLP(64, 64)
        self.conv1_1 = nn.Conv1d(3, 64, 1)   
        self.bn1_1 = nn.BatchNorm1d(64)
        self.conv1_2 = nn.Conv1d(64, 64, 1)
        self.bn1_2 = nn.BatchNorm1d(64)

        # Second stage: MLP(64, 128, 1024)
        self.conv2_1 = nn.Conv1d(64, 64, 1)
        self.bn2_1 = nn.BatchNorm1d(64)
        self.conv2_2 = nn.Conv1d(64, 128, 1)
        self.bn2_2 = nn.BatchNorm1d(128)
        self.conv2_3 = nn.Conv1d(128, 1024, 1)
        self.bn2_3 = nn.BatchNorm1d(1024)

        self.global_feat = global_feat

    def forward(self, x):
        """
        Forward pass through the PointNet Encoder.

        Args:
            x: Point cloud tensor of shape [B, 3, N], where B is the batch size, 
               3 is the number of features (typically x, y, z coordinates), and N is the number of points.

        Returns:
            Depending on `global_feat`, returns either global features or both global and per-point features.
        """
        # Input transformation (STN3D) for 3D coordinates
        trans = self.stn(x)  # [B, 3, 3]

        # Apply input transformation to the point cloud
        B, D, N = x.size()
        x = x.transpose(2, 1)            # [B, N, 3]
        x = torch.bmm(x, trans)          # [B, N, 3]
        x = x.transpose(2, 1)            # [B, 3, N]

        # First stage: MLP(64, 64)
        x = F.relu(self.bn1_1(self.conv1_1(x)))   # [B, 64, N]
        x = F.relu(self.bn1_2(self.conv1_2(x)))   # [B, 64, N]

        # Feature transformation (STN3D) for 64-dimensional features
        if self.feature_transform:
            trans_feat = self.fstn(x)            # [B, 64, 64]
            x = x.transpose(2, 1)                # [B, N, 64]
            x = torch.bmm(x, trans_feat)         # [B, N, 64]
            x = x.transpose(2, 1)                # [B, 64, N]
        else:
            trans_feat = None

        # Second stage: MLP(64, 128, 1024)
        x = F.relu(self.bn2_1(self.conv2_1(x)))   # [B, 64, N]
        x = F.relu(self.bn2_2(self.conv2_2(x)))   # [B, 128, N]
        x = F.relu(self.bn2_3(self.conv2_3(x)))   # [B, 1024, N]

        # Global max pooling
        x, _ = torch.max(x, 2)                    # [B, 1024]

        if self.global_feat:
            return x, trans, trans_feat
        else:
            return x, trans, trans_feat


class PointNetClassifier(nn.Module):
    """
    PointNet Classifier for point cloud classification.

    This model uses the PointNet encoder to extract global features and performs classification with
    fully connected layers.

    Args:
        num_classes: The number of output classes.
        feature_transform: Whether to use feature transformation for 64-dimensional features.
    
    Input:
        x: Point cloud tensor of shape [B, N, 3], where B is the batch size, 
           N is the number of points, and 3 represents the coordinates (x, y, z).

    Output:
        logits: The output class logits of shape [B, num_classes].
    """
    def __init__(self, num_classes=40, feature_transform=True):
        super(PointNetClassifier, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetEncoder(global_feat=True, 
                                    feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, num_classes)
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

    def forward(self, x, A=None):
        """
        Forward pass through the PointNet Classifier.

        Args:
            x: Point cloud tensor of shape [B, N, 3], where B is the batch size, 
               N is the number of points, and 3 represents the coordinates (x, y, z).
            A: Optional adjacency matrix (not used in this specific classifier).

        Returns:
            logits: The output class logits of shape [B, num_classes].
        """
        # Transpose to [B, 3, N] for input to the PointNet encoder
        x = x.permute(0, 2, 1)

        # Extract features using PointNet Encoder
        x, trans, trans_feat = self.feat(x)    # [B, 1024]
        
        # Fully connected layers for classification
        x = F.relu(self.bn1(self.fc1(x)))      # [B, 512]
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))      # [B, 256]
        x = self.dropout2(x)
        x = self.fc3(x)                        # [B, num_classes]
        return x
