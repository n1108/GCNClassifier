import torch
import torch.nn as nn
import torch.nn.functional as F

def farthest_point_sample(xyz, npoint):
    """
    Farthest Point Sampling (FPS) for selecting points from a point cloud.

    Args:
        xyz: The input point cloud of shape [B, N, 3], where B is the batch size, 
             N is the number of points in the cloud, and 3 represents the x, y, z coordinates.
        npoint: The number of points to sample from the input point cloud.

    Returns:
        A tensor of shape [B, npoint], containing the indices of the farthest sampled points.
    """
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.ones(B, N, device=xyz.device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(B, dtype=torch.long, device=xyz.device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    """
    Select points from the input tensor based on indices.

    Args:
        points: The input tensor of shape [B, N, C], where B is the batch size, 
                N is the number of points, and C is the number of features per point.
        idx: The indices of the points to select, with shape [B, S] or [B, S, nsample].

    Returns:
        A tensor of selected points or features of shape [B, S, C] or [B, S, nsample, C].
    """
    B = points.shape[0]
    batch_indices = torch.arange(B, device=points.device)

    if idx.dim() == 2:
        # idx: [B, S]
        batch_indices = batch_indices.view(-1, 1).repeat(1, idx.size(1))  # [B, S]
        new_points = points[batch_indices, idx, :]  # [B, S, C]
        return new_points

    elif idx.dim() == 3:
        # idx: [B, S, nsample]
        batch_indices = batch_indices.view(-1, 1, 1).repeat(1, idx.size(1), idx.size(2))
        new_points = points[batch_indices, idx, :]  # [B, S, nsample, C]
        return new_points
    else:
        raise ValueError("Unsupported idx shape")

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Query the neighboring points within a given radius for each center point.

    Args:
        radius: The radius within which to find neighboring points.
        nsample: The number of samples (neighbors) to return.
        xyz: The input point cloud of shape [B, N, 3], where B is the batch size and 
             N is the number of points.
        new_xyz: The center points for which to find the neighbors, of shape [B, S, 3], 
                 where S is the number of centers.

    Returns:
        A tensor of shape [B, S, nsample] containing the indices of the neighboring points.
    """
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    group_idx = torch.arange(N, device=xyz.device).view(1, 1, -1).repeat(B, S, 1)  # [B, S, N]
    sqrdists = torch.cdist(new_xyz, xyz, p=2)  # [B, S, N]
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  # Take the top `nsample` neighbors

    # If there are fewer than `nsample` neighbors, replace the excess indices with the first index
    first_idx = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    group_idx[group_idx == N] = first_idx[group_idx == N]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Perform farthest point sampling (FPS) and group the neighboring points within a given radius.

    Args:
        npoint: The number of center points to sample.
        radius: The radius to search for neighbors.
        nsample: The number of neighbors to sample for each center point.
        xyz: The input point cloud of shape [B, N, 3], where B is the batch size and 
             N is the number of points in the cloud.
        points: The features of the points of shape [B, N, C], where C is the feature dimension. 

    Returns:
        new_xyz: The center points sampled by FPS, of shape [B, npoint, 3].
        new_points: The grouped points with additional features, of shape [B, npoint, nsample, 3 + C].
    """
    B, N, C_ = xyz.shape
    # Perform farthest point sampling (FPS)
    idx = farthest_point_sample(xyz, npoint)  # [B, npoint]
    new_xyz = index_points(xyz, idx)          # [B, npoint, 3]
    # Perform ball query to find neighbors
    group_idx = query_ball_point(radius, nsample, xyz, new_xyz)  # [B, npoint, nsample]
    # Index the neighboring points
    grouped_xyz = index_points(xyz, group_idx)  # [B, npoint, nsample, 3]
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)  # Normalize to the center points

    if points is not None:
        grouped_points = index_points(points, group_idx)  # [B, npoint, nsample, C]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # Concatenate features
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    """
    PointNet Set Abstraction (SA) module for hierarchical feature extraction.
    
    This module performs point sampling, neighborhood grouping, and feature extraction 
    using MLPs on each group of points.

    Args:
        npoint: The number of center points to sample.
        radius: The radius for neighborhood grouping.
        nsample: The number of neighbors to sample for each center point.
        in_channel: The number of input channels (features per point).
        mlp: A list specifying the number of output channels for each MLP layer.
        group_all: Whether to group all points together for the entire point cloud (True) or sample subsets (False).

    Input:
        xyz: The point cloud tensor of shape [B, N, 3].
        points: The feature tensor of shape [B, N, C] (optional).

    Output:
        new_xyz: The center points after sampling, of shape [B, npoint, 3].
        new_points: The features of the grouped points, of shape [B, npoint, nsample, C + 3].
    """
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        last_channel = in_channel
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Forward pass through the Set Abstraction module.

        Args:
            xyz: The point cloud tensor of shape [B, N, 3].
            points: The feature tensor of shape [B, N, C] (optional).

        Returns:
            new_xyz: The center points after sampling, of shape [B, npoint, 3].
            new_points: The features of the grouped points, of shape [B, npoint, nsample, C + 3].
        """
        B, N, _ = xyz.shape
        if self.group_all:
            # If group_all is True, use all points in the point cloud as a single group
            new_xyz = torch.zeros(B, 1, 3, device=xyz.device)
            grouped_xyz_norm = xyz.view(B, 1, N, 3)
            if points is not None:
                new_points = torch.cat([grouped_xyz_norm, points.view(B, 1, N, -1)], dim=-1)
            else:
                new_points = grouped_xyz_norm
        else:
            # Otherwise, perform farthest point sampling and ball query
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        # Rearrange new_points tensor for MLP processing
        new_points = new_points.permute(0, 3, 1, 2)

        # Apply MLPs for feature extraction
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # Perform max pooling to get a global feature representation
        new_points = torch.max(new_points, -1)[0]  # [B, mlp[-1], npoint]
        new_points = new_points.transpose(1, 2)   # [B, npoint, mlp[-1]]
        return new_xyz, new_points

class PointNet2Classifier(nn.Module):
    """
    PointNet2 Classifier for point cloud classification.

    This model applies multiple PointNet Set Abstraction (SA) layers followed by MLP layers 
    for final classification.

    Args:
        num_classes: The number of output classes for classification.

    Input:
        xyz: The point cloud tensor of shape [B, N, 3], where B is the batch size, 
             N is the number of points, and 3 represents the x, y, z coordinates.

    Output:
        logits: The output class logits of shape [B, num_classes].
    """
    def __init__(self, num_classes=40):
        super(PointNet2Classifier, self).__init__()

        # First Set Abstraction (SA) layer
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32,
                                          in_channel=3, mlp=[64, 64, 128], group_all=False)

        # Second Set Abstraction (SA) layer
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64,
                                          in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)

        # Third Set Abstraction (SA) layer with global pooling
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None,
                                          in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        # Classification MLP
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xyz, A=None):
        """
        Forward pass through the PointNet2 classifier.

        Args:
            xyz: The point cloud tensor of shape [B, N, 3].
            A: An optional adjacency matrix (not used in this model).

        Returns:
            logits: The output class logits of shape [B, num_classes].
        """
        B, N, _ = xyz.shape
        points = None

        # Apply first Set Abstraction (SA) layer
        l1_xyz, l1_points = self.sa1(xyz, points)  # [B, 512, 3], [B, 512, 128]

        # Apply second Set Abstraction (SA) layer
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # [B, 128, 3], [B, 128, 256]

        # Apply third Set Abstraction (SA) layer with global pooling
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # [B, 1, 3], [B, 1, 1024]

        # Flatten the global feature for classification
        x = l3_points.view(B, 1024)

        # Apply fully connected layers for classification
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)  # [B, num_classes]
        return x
