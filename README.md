# GCNClassifier

This project provides a graph convolutional network (GCN)-based method for point cloud classification (e.g., ModelNet40), and offers multiple deep learning model implementations (such as DGCNN, GAT, GIN, PointNet, PointNet++, etc.). The project mainly includes the following features:
1. **T-Net** module for input alignment (optional).
2. **Hierarchical Pooling (SAGPool)** for multi-scale feature extraction (optional).
3. Additional optional enhancements such as distance weighting, Batch Normalization, residual connections, virtual nodes, etc., to study their impact on classification accuracy.

---

## Project Structure

The main files and directories of the project are as follows:

```
GCNClassifier/
│── data/                  # Directory for datasets
│── model/
│   │── dgcnn.py           # DGCNN model
│   │── gat.py             # GAT model
│   │── gcn.py             # GCN and associated modules (T-Net, SAGPool, etc.)
│   │── gin.py             # GIN model
│   │── pointnet.py        # PointNet model
│   │── pointnet2.py       # PointNet++ model
│── utils/
│   │── data_utils.py      # Utilities for data download, preprocessing, KNN graph construction, etc.
│── train.py               # Script for training and testing (entry point)
```

---

## Environment Requirements

- Python >= 3.8
- PyTorch >= 2.1.0
- NumPy
- h5py
- tqdm

Example installation (it is recommended to use a virtual environment):
```bash
pip install torch numpy h5py tqdm
```

---

## Data Description

- By default, this project uses the [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) dataset for training and testing examples.
- The first time you run `train.py`, the code will automatically download and unzip the dataset to the specified `data_dir` path.
- This dataset contains multiple classes (a total of 40 categories), and each sample is a 3D point cloud.

---

## Example Usage

Below are some common usage examples, which can be modified according to your needs.

1. **Train the GCN model**
    ```bash
    python train.py \
      --model_type gcn \
      --data_dir data \
      --num_points 1024 \
      --batch_size 32 \
      --epochs 50 \
      --lr 0.001 \
      --use_rotate \
      --use_jitter \
      --use_scale \
      --use_translate \
      --use_bn \
      --use_residual \
      --use_hierarchical_pooling \
      --use_TNet \
      --use_virtual_node \
      --k_neighbors 20
    ```

2. **Train the DGCNN model**
    ```bash
    python train.py \
      --model_type dgcnn \
      --data_dir data \
      --k_neighbors 20 \
      --epochs 50 \
      --use_rotate \
      --use_jitter \
      --use_scale \
      --use_translate
    ```

3. **Train the PointNet model**
    ```bash
    python train.py \
      --model_type pointnet \
      --data_dir data \
      --epochs 50 \
      --use_rotate \
      --use_jitter \
      --use_scale \
      --use_translate
    ```
    
4. **More optional parameters**: Run `python train.py --help` to see all available command-line options and their descriptions.

---

## Parameter Descriptions

- `--model_type`: Options are `gcn`, `gat`, `gin`, `dgcnn`, `pointnet`, `pointnet2`.
- `--data_dir`: Path to the dataset. If not found, it will be downloaded automatically.
- `--num_points`: Number of points sampled from the point cloud (default `1024`).
- `--batch_size`: Batch size (default `32`).
- `--epochs`: Number of training epochs (default `300`).
- `--lr`: Initial learning rate (default `0.001`).
- `--device`: Computing device (default `cuda:0`).
- `--seed`: Random seed (default `0`).
- `--weight_decay`: Weight decay coefficient (default `0.0001`).
- `--lr_scheduler`: Learning rate scheduling strategy, options are `none`, `cosine`, or `step` (default `cosine`).
- `--step_size` / `--gamma`: When `lr_scheduler` is `step`, specify the StepLR step size and decay factor.
- **Data Augmentation**:
  - `--use_rotate`: Enable random rotation augmentation.
  - `--rotate_axis`: Specify the axis of rotation (`x`, `y`, or `z`, default `z`).
  - `--use_jitter`: Enable random jitter augmentation.
  - `--use_scale`: Enable random scaling.
  - `--use_translate`: Enable random translation.
- **KNN Graph Construction**:
  - `--k_neighbors`: Number of neighbors in KNN (default `20`).
  - `--use_weighted_edge`: Use weighted edges (distance-based Gaussian kernel).
- **GCN Specific**:
  - `--hidden_dims`: List of channel dimensions for GCN layers (default `'128,256,512'`).
  - `--num_layers`: Number of GCN layers (default `3`).
  - `--dropout`: Dropout rate (default `0.5`).
  - `--use_bn`: Use BatchNorm or not.
  - `--use_residual`: Use residual connections or not.
  - `--use_hierarchical_pooling`: Use hierarchical pooling (SAGPool) or not.
  - `--use_TNet`: Use T-Net for input alignment or not.
  - `--use_virtual_node`: Add a virtual node in the graph for information aggregation.

---

## Training Process and Model Evaluation

- **Training**: The `train_one_epoch` function completes one epoch of forward and backward passes, calculating the average loss, overall accuracy, and mean class accuracy on the training set.
- **Testing**: The `eval_one_epoch` function evaluates the model on the test set and outputs the average loss, overall accuracy, and mean class accuracy.

After training, the best model weights (e.g., `best_gcn_model.pth`) corresponding to the highest test accuracy will be saved.

---

## Conclusion and Future Work

In this project, we demonstrate how to use a graph neural network framework to perform point cloud classification, including:
- Using **T-Net** for pose alignment to enhance robustness to input variations;
- Exploring **hard Top-K** and **soft gating (Soft Gate)** approaches in graph pooling;
- Further discussing the impact of distance weighting, BatchNorm, residual connections, and virtual nodes on classification accuracy.

Potential future improvements include:
- Dynamic graph methods to adaptively update adjacency relationships as the network depth increases;
- Incorporating more geometric features in graph construction to improve the network's adaptation and classification performance on point cloud data.
