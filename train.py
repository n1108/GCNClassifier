import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random

from utils.data_utils import ModelNet40Dataset, collate_fn as base_collate_fn

from model.gcn import GCNClassifier
from model.gat import GATClassifier
from model.dgcnn import DGCNNClassifier
from model.pointnet import PointNetClassifier
from model.pointnet2 import PointNet2Classifier
from model.gin import GINClassifier


def set_seed(seed):
    """
    Set the seed for random number generators in PyTorch, NumPy, and Python's random module
    to ensure reproducibility across experiments.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, optimizer, criterion, dataloader, device, num_classes=40):
    """
    Perform one epoch of training, computing loss and accuracy for the entire batch.
    
    Args:
        model: The model to be trained.
        optimizer: The optimizer used for parameter updates.
        criterion: The loss function used to compute the loss.
        dataloader: DataLoader for training data.
        device: The device to run computations on.
        num_classes: The number of output classes for classification.
        
    Returns:
        avg_loss: The average loss for the epoch.
        accuracy_overall: The overall accuracy for the epoch.
        avg_class_acc: The average per-class accuracy for the epoch.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # Initialize arrays to track per-class correct predictions and total instances
    correct_per_class = [0] * num_classes
    total_per_class   = [0] * num_classes
    
    for points, graphs, labels in tqdm(dataloader, desc="Training", leave=False):
        points = points.to(device)  # [B, N, 3]
        if graphs is not None:
            graphs = graphs.to(device)  # [B, N, N]
        labels = labels.to(device)  # [B]
        
        optimizer.zero_grad()
        logits = model(points, graphs)  # Forward pass
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        batch_size = points.size(0)
        total_loss += loss.item() * batch_size
        _, preds = torch.max(logits, dim=1)
        
        # Update total correct predictions
        total_correct += torch.sum(preds == labels).item()
        total_samples += batch_size
        
        # Track per-class correct predictions
        for i in range(batch_size):
            label_i = labels[i].item()
            pred_i  = preds[i].item()
            total_per_class[label_i] += 1
            if pred_i == label_i:
                correct_per_class[label_i] += 1
    
    # Calculate overall average loss and accuracy
    avg_loss = total_loss / total_samples
    accuracy_overall = total_correct / total_samples
    
    # Calculate average per-class accuracy
    class_accs = []
    for c in range(num_classes):
        if total_per_class[c] > 0:
            class_accs.append(correct_per_class[c] / total_per_class[c])
    avg_class_acc = sum(class_accs) / len(class_accs) if len(class_accs) > 0 else 0.0
    
    return avg_loss, accuracy_overall, avg_class_acc


def eval_one_epoch(model, criterion, dataloader, device, num_classes=40):
    """
    Perform one epoch of evaluation on the validation/test set, calculating loss and accuracy.
    
    Args:
        model: The trained model to be evaluated.
        criterion: The loss function used to compute the loss.
        dataloader: DataLoader for the evaluation data.
        device: The device to run computations on.
        num_classes: The number of output classes for classification.
        
    Returns:
        avg_loss: The average loss for the epoch.
        accuracy_overall: The overall accuracy for the epoch.
        avg_class_acc: The average per-class accuracy for the epoch.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # Initialize arrays to track per-class correct predictions and total instances
    correct_per_class = [0] * num_classes
    total_per_class   = [0] * num_classes
    
    with torch.no_grad():
        for points, graphs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            points = points.to(device)
            if graphs is not None:
                graphs = graphs.to(device)
            labels = labels.to(device)
            
            logits = model(points, graphs)
            loss = criterion(logits, labels)
            
            batch_size = points.size(0)
            total_loss += loss.item() * batch_size
            _, preds = torch.max(logits, dim=1)
            
            # Update total correct predictions
            total_correct += torch.sum(preds == labels).item()
            total_samples += batch_size
            
            # Track per-class correct predictions
            for i in range(batch_size):
                label_i = labels[i].item()
                pred_i  = preds[i].item()
                total_per_class[label_i] += 1
                if pred_i == label_i:
                    correct_per_class[label_i] += 1
    
    # Calculate overall average loss and accuracy
    avg_loss = total_loss / total_samples
    accuracy_overall = total_correct / total_samples
    
    # Calculate average per-class accuracy
    class_accs = []
    for c in range(num_classes):
        if total_per_class[c] > 0:
            class_accs.append(correct_per_class[c] / total_per_class[c])
    avg_class_acc = sum(class_accs) / len(class_accs) if len(class_accs) > 0 else 0.0
    
    return avg_loss, accuracy_overall, avg_class_acc


def parse_hidden_dims(hidden_dims_str):
    """
    Convert a comma-separated string of integers (e.g., "64,128,256") into a list of integers.
    
    Args:
        hidden_dims_str: A string representing the dimensions of the hidden layers (comma-separated).
        
    Returns:
        dims: A list of integers corresponding to the hidden dimensions.
    """
    dims = [int(x) for x in hidden_dims_str.split(',')]
    return dims


def main():
    parser = argparse.ArgumentParser()
    
    # Model selection argument
    parser.add_argument('--model_type', type=str, default='gcn',
                    choices=['gcn','gat','gin','dgcnn','pointnet','pointnet2'],
                    help='The type of model to train: gcn, gat, gin, dgcnn, pointnet, or pointnet2')

    # Dataset directory argument
    parser.add_argument('--data_dir', type=str, default='data',
                        help='The directory containing the dataset files')

    # Number of points to sample from the point cloud
    parser.add_argument('--num_points', type=int, default=1024,
                        help='The number of points to sample from each point cloud')

    # Hyperparameter arguments
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=0,
                        help='The random seed for reproducibility of results')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='The weight decay factor for L2 regularization')

    # Learning rate scheduler options
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['none', 'cosine', 'step'],
                        help='The type of learning rate scheduler to use: none, cosine, or step')
    parser.add_argument('--step_size', type=int, default=20,
                        help='The step size for the StepLR scheduler (only used if lr_scheduler=step)')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='The gamma factor for StepLR scheduler (only used if lr_scheduler=step)')
    
    # Data augmentation options
    parser.add_argument('--use_rotate', action='store_true', 
                        help='Whether to use random rotation for data augmentation')
    parser.add_argument('--rotate_axis', type=str, default='z',
                        help='The axis around which to apply rotation (z|y|x)')
    parser.add_argument('--use_jitter', action='store_true',
                        help='Whether to apply random jitter to point clouds')
    parser.add_argument('--use_scale', action='store_true',
                        help='Whether to apply random scaling to point clouds')
    parser.add_argument('--use_translate', action='store_true',
                        help='Whether to apply random translation to point clouds')
    
    # GCN/GAT/GIN/DGCNN specific parameters
    parser.add_argument('--k_neighbors', type=int, default=20,
                        help='The number of neighbors to use for constructing KNN graphs')
    
    # GCN/GAT/GIN specific parameter: Whether to use weighted edges
    parser.add_argument('--use_weighted_edge', action='store_true',
                        help='Whether to use the distance as edge weight in the graph adjacency matrix')
    
    # GCN-specific parameters
    parser.add_argument('--num_layers', type=int, default=3, 
                        help='The number of layers in the GCN model')
    parser.add_argument('--hidden_dims', type=str, default='128,256,512',
                        help='Comma-separated list of hidden layer dimensions for the GCN layers')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--use_bn', action='store_true',
                        help='Whether to use Batch Normalization in GCN layers')
    parser.add_argument('--use_residual', action='store_true',
                        help='Whether to use residual (skip) connections in GCN layers')
    parser.add_argument('--use_hierarchical_pooling', action='store_true',
                            help='Whether to use hierarchical graph pooling')
    parser.add_argument('--use_TNet', action='store_true',
                        help='Whether to use TNet for input alignment')
    parser.add_argument('--use_virtual_node', action='store_true',
                    help='Whether to add virtual nodes to the graph')

    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Determine the device for model training/evaluation
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Args:", args)
    
    # Load datasets for training and testing
    train_dataset = ModelNet40Dataset(
        data_dir=args.data_dir, 
        split='train',
        use_rotate=args.use_rotate,
        rotate_axis=args.rotate_axis,
        use_jitter=args.use_jitter,
        use_scale=args.use_scale,
        use_translate=args.use_translate,
        num_points=args.num_points
    )
    test_dataset = ModelNet40Dataset(
        data_dir=args.data_dir, 
        split='test',
        num_points=args.num_points
    )

    def collate_fn_wrapper(batch):
        """
        Custom collate function for batching data, depending on the model type.
        """
        if args.model_type in ['gcn','gat','gin']:
            return base_collate_fn(batch, k=args.k_neighbors, use_weighted_edge=args.use_weighted_edge)
        else:
            point_list = []
            label_list = []
            for pc, label in batch:
                point_list.append(pc)
                label_list.append(label)

            points_batch = np.stack(point_list, axis=0)
            labels_batch = np.array(label_list, dtype=np.int64)

            points_batch = torch.from_numpy(points_batch).float()
            labels_batch = torch.from_numpy(labels_batch).long()

            points_batch = points_batch.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            
            return points_batch, None, labels_batch
    
    # Set up data loaders for training and testing
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True,
                              collate_fn=collate_fn_wrapper
    )
    test_loader  = DataLoader(test_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              drop_last=False,
                              collate_fn=collate_fn_wrapper
    )

    # Model selection based on the argument passed
    if args.model_type == 'gcn':
        hidden_dims_list = parse_hidden_dims(args.hidden_dims)
        if len(hidden_dims_list) > args.num_layers:
            hidden_dims_list = hidden_dims_list[:args.num_layers]
        else:
            while len(hidden_dims_list) < args.num_layers:
                hidden_dims_list.append(hidden_dims_list[-1])

        model = GCNClassifier(
            input_dim=3, 
            hidden_dims=hidden_dims_list, 
            num_classes=40, 
            dropout=args.dropout,
            use_bn=args.use_bn,
            use_residual=args.use_residual,
            use_hierarchical_pooling=args.use_hierarchical_pooling,
            use_TNet=args.use_TNet,
            use_virtual_node=args.use_virtual_node
        )

    elif args.model_type == 'gat':
        model = GATClassifier(
            input_dim=3,
            num_classes=40
        )

    elif args.model_type == 'gin':
        model = GINClassifier(input_dim=3, 
                              num_classes=40
        )

    elif args.model_type == 'dgcnn':
        model = DGCNNClassifier(num_classes=40, k=args.k_neighbors)

    elif args.model_type == 'pointnet':
        model = PointNetClassifier(num_classes=40)

    else:
        model = PointNet2Classifier(num_classes=40)

    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
 
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Set up learning rate scheduler
    if args.lr_scheduler == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    elif args.lr_scheduler == 'step':
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        scheduler = None
    
    best_acc = 0.0
    num_classes = 40 

    for epoch in range(args.epochs):
        print(f"Epoch [{epoch+1}/{args.epochs}]")
        
        # Training phase
        train_loss, train_acc_overall, train_acc_class = train_one_epoch(
            model, optimizer, criterion, train_loader, device, num_classes
        )
        
        # Validation phase
        val_loss, val_acc_overall, val_acc_class = eval_one_epoch(
            model, criterion, test_loader, device, num_classes
        )
        
        if scheduler is not None:
            scheduler.step()
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc(Overall): {train_acc_overall:.4f}, "
              f"Train Acc(Avg.Class): {train_acc_class:.4f}")
        print(f"  Test  Loss: {val_loss:.4f},   Test Acc(Overall): {val_acc_overall:.4f}, "
              f"Test Acc(Avg.Class): {val_acc_class:.4f}")
        
        # Save the best model based on validation accuracy
        if val_acc_overall > best_acc:
            best_acc = val_acc_overall
            torch.save(model.state_dict(), f'best_{args.model_type}_model.pth')
            print("  => Best model updated, accuracy = {:.4f}".format(best_acc))


if __name__ == '__main__':
    main()
