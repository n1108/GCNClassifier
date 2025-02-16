import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import zipfile
from urllib.request import urlretrieve

def download_modelnet40(data_dir):
    """
    Downloads and extracts the ModelNet40 dataset to the specified directory.

    Args:
        data_dir (str): Directory to download and extract the dataset.

    Returns:
        str: Path to the extracted dataset directory.
    """
    os.makedirs(data_dir, exist_ok=True)
    url = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
    zip_path = os.path.join(data_dir, "modelnet40_ply_hdf5_2048.zip")
    extract_dir = os.path.join(data_dir, "modelnet40_ply_hdf5_2048")

    # Check if dataset is already extracted
    if os.path.exists(os.path.join(extract_dir, "train_files.txt")):
        # print("ModelNet40 dataset already downloaded and extracted.")
        return extract_dir

    # Download the zip file if not present
    if not os.path.exists(zip_path):
        print("Downloading ModelNet40 dataset...")
        urlretrieve(url, zip_path)
        print("Download completed.")

    # Extract the dataset
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print("Extraction completed.")

    return extract_dir


def random_rotate_point_cloud(pc, angle_range=(-np.pi/20, np.pi/20), axis='z'):
    """
    Apply a random rotation to a point cloud, with the default rotation axis being 'z'.
    
    Args:
        pc: The point cloud, represented as a numpy array of shape [N, 3].
        angle_range: A tuple specifying the range of angles (in radians) for random rotation.
        axis: The axis around which to apply the rotation ('x', 'y', or 'z').

    Returns:
        The rotated point cloud as a numpy array of shape [N, 3].
    """
    angle = np.random.uniform(angle_range[0], angle_range[1])
    cosval = np.cos(angle)
    sinval = np.sin(angle)

    if axis.lower() == 'z':
        rotation_matrix = np.array([
            [cosval, -sinval, 0],
            [sinval,  cosval, 0],
            [0,       0,      1]
        ], dtype=np.float32)
    elif axis.lower() == 'y':
        rotation_matrix = np.array([
            [cosval,  0, sinval],
            [0,       1, 0     ],
            [-sinval, 0, cosval]
        ], dtype=np.float32)
    elif axis.lower() == 'x':
        rotation_matrix = np.array([
            [1, 0,       0      ],
            [0, cosval, -sinval ],
            [0, sinval,  cosval ]
        ], dtype=np.float32)
    else:
        rotation_matrix = np.eye(3, dtype=np.float32)
    
    pc_rotated = pc @ rotation_matrix.T
    return pc_rotated


def random_jitter_point_cloud(pc, sigma=0.01, clip=0.02):
    """
    Apply random jitter to the point cloud by adding noise to the points.
    
    Args:
        pc: The point cloud, represented as a numpy array of shape [N, 3].
        sigma: The standard deviation of the noise to add.
        clip: The maximum noise magnitude.

    Returns:
        The jittered point cloud as a numpy array of shape [N, 3].
    """
    jitter = np.clip(sigma * np.random.randn(*pc.shape), -clip, clip)
    pc_jittered = pc + jitter
    return pc_jittered


def random_scale_point_cloud(pc, scale_range=(0.95, 1.05)):
    """
    Apply random scaling to the point cloud by scaling all points uniformly.
    
    Args:
        pc: The point cloud, represented as a numpy array of shape [N, 3].
        scale_range: The range from which to sample the scaling factor.

    Returns:
        The scaled point cloud as a numpy array of shape [N, 3].
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    pc_scaled = pc * scale
    return pc_scaled

def random_translate_point_cloud(pc, translation_range=(-0.05, 0.05)):
    """
    Apply random translation to the point cloud by shifting all points randomly in 3D space.
    
    Args:
        pc: The point cloud, represented as a numpy array of shape [N, 3].
        translation_range: A tuple specifying the translation range along each axis (x, y, z).
    
    Returns:
        The translated point cloud as a numpy array of shape [N, 3].
    """
    translation = np.random.uniform(translation_range[0], translation_range[1], size=(3,))
    pc_translated = pc + translation
    return pc_translated


class ModelNet40Dataset(Dataset):
    """
    A custom Dataset class to load the ModelNet40 dataset for point cloud classification.
    
    Args:
        data_dir: The directory where the dataset is stored.
        split: Whether to load the 'train' or 'test' split.
        use_rotate: Whether to apply random rotation augmentation.
        rotate_axis: The axis around which to apply rotation ('x', 'y', or 'z').
        use_jitter: Whether to apply random jitter augmentation.
        use_scale: Whether to apply random scaling augmentation.
        use_translate: Whether to apply random translation augmentation.
        num_points: The number of points to sample from each point cloud.
    """
    def __init__(self, data_dir,
                 split='train',
                 use_rotate=False,
                 rotate_axis='z',
                 use_jitter=False,
                 use_scale=False,
                 use_translate=False,
                 num_points=1024):
        super(ModelNet40Dataset, self).__init__()
        self.data_dir = data_dir
        self.split = split
        self.use_rotate = use_rotate
        self.rotate_axis = rotate_axis
        self.use_jitter = use_jitter
        self.use_scale = use_scale
        self.use_translate = use_translate
        self.num_points = num_points

        extract_dir = download_modelnet40(data_dir)

        # Load the list of files for training or testing
        if split == 'train':
            file_list = os.path.join(extract_dir, 'train_files.txt')
        else:
            file_list = os.path.join(extract_dir, 'test_files.txt')

        self.file_paths = []
        with open(file_list, 'r') as f:
            for line in f:
                self.file_paths.append(os.path.join(extract_dir,
                                                    os.path.basename(line.strip())))

        self.points = []
        self.labels = []

        # Load point cloud and label data from h5 files
        for h5_name in self.file_paths:
            with h5py.File(h5_name, 'r') as hf:
                data = hf['data'][:]   # [B, N, 3]
                label = hf['label'][:] # [B, 1]
                self.points.append(data)
                self.labels.append(label)

        self.points = np.concatenate(self.points, axis=0)  # [total_samples, N, 3]
        self.labels = np.concatenate(self.labels, axis=0).squeeze()  # [total_samples]

        if self.points.shape[1] < self.num_points:
            print(f"[Warning] The dataset has fewer points ({self.points.shape[1]}) "
                  f"than num_points ({self.num_points}).")


    def __getitem__(self, index):
        """
        Retrieve a point cloud and its label for a given index, applying data augmentation if needed.
        
        Args:
            index: The index of the sample to retrieve.
        
        Returns:
            A tuple (pc, label) where:
                - pc is the sampled point cloud of shape [num_points, 3].
                - label is the corresponding label.
        """
        pc = self.points[index]   # [N, 3]
        label = self.labels[index]

        N = pc.shape[0]
        if N >= self.num_points:
            # If there are more or equal points than required, randomly select points
            idx = np.random.choice(N, self.num_points, replace=False)
        else:
            # If there are fewer points, perform sampling with replacement or take all points
            idx = np.random.choice(N, self.num_points, replace=True)
        pc = pc[idx, :]
        
        if self.split == 'train':
            # Apply data augmentation only during training
            if self.use_rotate:
                pc = random_rotate_point_cloud(pc, axis=self.rotate_axis)
            if self.use_scale:
                pc = random_scale_point_cloud(pc)
            if self.use_jitter:
                pc = random_jitter_point_cloud(pc)
            if self.use_translate:
                pc = random_translate_point_cloud(pc)

        return pc, label

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return self.points.shape[0]

def build_graph_knn_batch(points_batch, k=20, use_weighted_edge=False, sigma=0.1):
    """
    Build the K-nearest neighbors (KNN) graph adjacency matrix for a batch of point clouds.

    Args:
        points_batch: A batch of point clouds, a tensor of shape [B, N, 3].
        k: The number of neighbors to consider for each point.
        use_weighted_edge: Whether to use Gaussian weighting for the edges.
        sigma: The width of the Gaussian kernel for weighting edges.

    Returns:
        The adjacency matrix for the batch, a tensor of shape [B, N, N].
    """
    B, N, _ = points_batch.shape

    with torch.no_grad():
        dist_matrix = torch.cdist(points_batch, points_batch, p=2)  # [B, N, N]

    # Find the k nearest neighbors (excluding the point itself)
    _, idx = dist_matrix.topk(k=k+1, dim=-1, largest=False)

    adjacency_batch = torch.zeros_like(dist_matrix)
    batch_arange = torch.arange(B, device=points_batch.device)[:, None, None]
    i_arange = torch.arange(N, device=points_batch.device)[None, :, None]
    neighbors = idx[:, :, 1:]  # Skip the point itself

    if use_weighted_edge:
        neighbor_distances = torch.gather(dist_matrix, dim=-1, index=neighbors)
        adjacency_value = torch.exp(- (neighbor_distances ** 2) / (2 * (sigma ** 2)))

        adjacency_batch[batch_arange, i_arange, neighbors] = adjacency_value
        adjacency_batch[batch_arange, neighbors, i_arange] = adjacency_value
    else:
        # Unweighted graph: Set neighbors' edge weights to 1
        adjacency_batch[batch_arange, i_arange, neighbors] = 1.0
        adjacency_batch[batch_arange, neighbors, i_arange] = 1.0

    return adjacency_batch

def collate_fn(batch, k=10, use_weighted_edge=False):
    """
    Collate function to process a batch of point clouds and construct the adjacency matrix.
    
    Args:
        batch: A list of tuples, where each tuple is (point_cloud, label).
        k: The number of neighbors to consider for the KNN graph.
        use_weighted_edge: Whether to use weighted edges in the KNN graph.

    Returns:
        A tuple (points_batch, graphs_batch, labels_batch) where:
            - points_batch is the batch of point clouds.
            - graphs_batch is the batch of KNN graphs.
            - labels_batch is the batch of labels.
    """
    point_list = []
    label_list = []
    for pc, label in batch:
        point_list.append(pc)
        label_list.append(label)

    # Stack the point clouds and labels to form batches
    points_batch = np.stack(point_list, axis=0)
    labels_batch = np.array(label_list, dtype=np.int64)

    # Convert to PyTorch tensors
    points_batch = torch.from_numpy(points_batch).float()
    labels_batch = torch.from_numpy(labels_batch).long()

    # Move data to GPU for graph construction
    points_batch = points_batch.cuda(non_blocking=True)
    graphs_batch = build_graph_knn_batch(points_batch, k=k, use_weighted_edge=use_weighted_edge)

    return points_batch, graphs_batch, labels_batch.cuda(non_blocking=True)
