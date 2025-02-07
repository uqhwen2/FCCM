#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import csv
from torch.nn.utils import spectral_norm
import torch.nn as nn

from bmdal_reg.bmdal.feature_data import TensorFeatureData
from bmdal_reg.bmdal.algorithms import select_batch

from pathlib import Path
from models.nn_model import train_deep_kernel_gp, predict_deep_kernel_gp
from causal_bald.library import acquisitions
from models.nn_model import nnModel_1, nnModel_0
from torch.utils import data

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import networkx as nx
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import os

import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--bmdal', type=str, default='lcmd')
parser.add_argument('--alpha', type=float, default=2.5)
parser.add_argument('--selwithtrain', type=bool, default=True)
args = parser.parse_args()

import json
# Read the configuration from the file
with open('experiments/config_{}.json'.format(args.bmdal), 'r') as file:
    config = json.load(file)

from torch.utils.data import Dataset
from causal_bald.library.datasets import HCMNIST

def edge_generator(distance_matrix, threshold=0.5):
    n = distance_matrix.shape[0]
    for i in range(n):
        row = distance_matrix[i, :]
        cols = np.where(row < threshold)[0]
        for j in cols:
            if i != j:  # Avoid self-loops if needed
                yield (i, j)

def write_edges_to_file(edge_gen, filename='edges.txt'):
    with open(filename, 'w') as f:
        for edge in edge_gen:
            f.write(f"{edge[0]},{edge[1]}\n")

def create_graph_from_file(filename='edges.txt'):
    G = nx.DiGraph()
    with open(filename, 'r') as f:
        for line in f:
            i, j = map(int, line.strip().split(','))
            G.add_edge(i, j)
    return G



def process_matrix(distance_matrix, threshold=0.5, chunk_size=1000):
    n = distance_matrix.shape[0]
    edge_gen = edge_generator(distance_matrix, threshold)
    write_edges_to_file(edge_gen)
    
    print("Edges written to file. Now creating graph...")
    G = create_graph_from_file()
    print("Graph creation complete.")
    
    return G

class toDataLoader(Dataset):
    def __init__(self, x_train, y_train, t_train):
        # Generate random data for input features (x) and target variable (y)
        self.x = x_train
        self.t = t_train
        self.y = y_train

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Return a single sample as a dictionary containing input features and target variable
        inputs = torch.hstack([self.x[idx], self.t[idx]]).float()
        targets = self.y[idx].float()



        return inputs, targets


def generation(mean_1, std_1, func_1, mean_0, std_0, func_0, device):

    def generate_clusters(n_clusters, scale=3, offset=0):
        centers = []
        attempts = 0
        max_attempts = 100  # Maximum attempts to find a suitable center
        min_dist = 1.5  # Minimum distance between centers

        while len(centers) < n_clusters:
            new_center = np.random.uniform(-3 * scale + offset, 3 * scale + offset, size=2)
            # Check distances to all existing centers
            if all(np.linalg.norm(new_center - center) >= min_dist for center in centers):
                centers.append(new_center)
            else:
                attempts += 1
                if attempts > max_attempts:
                    # Reduce the minimum distance slightly if too many attempts fail
                    min_dist *= 0.9
                    attempts = 0

        variances = np.ones((n_clusters, 2))  # uniform variance
        return np.array(centers), variances

    def generate_data(centers, variances, points_per_cluster):
        data = np.vstack([
            np.random.randn(points_per_cluster, 2) * np.sqrt(variance) + mean
            for mean, variance in zip(centers, variances)
        ])
        return data

    # Set random seed for reproducibility
    np.random.seed(0)

    # Define number of clusters and points per cluster
    n_clusters_0 = 30
    n_clusters_1 = 50
    points_per_cluster = 200 #200

    # Define offsets for each class
    offset_0 = -2  # Shift Class 0 to the left
    offset_1 = 2  # Shift Class 1 to the right

    # Generate centers and variances for each class with different offsets
    class_0_centers, class_0_variances = generate_clusters(n_clusters_0, offset=offset_0)
    class_1_centers, class_1_variances = generate_clusters(n_clusters_1, offset=offset_1)

    # Generate data for each class
    class_0 = generate_data(class_0_centers, class_0_variances, points_per_cluster)
    class_1 = generate_data(class_1_centers, class_1_variances, points_per_cluster)

    class_0 = torch.from_numpy(class_0).float()
    class_1 = torch.from_numpy(class_1).float()

    # Combine the classes for x feature and create corresponding labels
    x_data = torch.vstack((class_0, class_1))
    y_labels = torch.hstack((torch.zeros(len(class_0)), torch.ones(len(class_1))))  # 0 for class 1, 1 for class 2

    # Generate a non-linear response curve for y using a non-linear transformation of x_data
    y_non_linear = torch.sin(1.5 * x_data[:, 0]) + torch.cos(1.5 * x_data[:, 1]) + 5.0 * y_labels

    y_non_linear_1 = torch.sin(1.5 * x_data[:, 0]) + torch.cos(1.5 * x_data[:, 1]) + 5.0 * 1
    y_non_linear_0 = torch.sin(1.5 * x_data[:, 0]) + torch.cos(1.5 * x_data[:, 1]) + 5.0 * 0

    combine_x = x_data
    combined_y = y_non_linear
    tau = y_non_linear_1 - y_non_linear_0
    T = y_labels

    return combine_x.to(device), combined_y.to(device), tau.to(device), T.to(device)


def train_test_splitting(combine_x, combined_y, tau, T, test_size, seed, device):
    # Convert tensors to numpy arrays
    combine_x_np = combine_x.cpu().numpy()
    combined_y_np = combined_y.cpu().numpy()
    tau_np = tau.cpu().numpy()
    T_np = T.cpu().numpy()

    combine_x_train_, combine_x_test, \
        combined_y_train_, combined_y_test, \
        tau_train, tau_test, \
        T_train_, T_test = train_test_split(combine_x_np, combined_y_np, tau_np, T_np, test_size=test_size,
                                            random_state=seed)

    # ihdp_train = HCMNIST(root='assets', split='train', mode='mu', seed=seed)
    # ihdp_test = HCMNIST(root='assets', split='valid', mode='mu', seed=seed)

    valid_size = 0.20
    training_idx, valid_idx = train_test_split(list(range(combine_x_train_.shape[0])),
                                               test_size=valid_size,
                                               random_state=seed)

    # Convert back to PyTorch tensors if needed
    combine_x_train = torch.from_numpy(combine_x_train_[training_idx])
    combine_x_valid = torch.from_numpy(combine_x_train_[valid_idx])
    combine_x_test = torch.from_numpy(combine_x_test)

    combined_y_train = torch.from_numpy(combined_y_train_[training_idx])  # No y normalization
    combined_y_valid = torch.from_numpy(combined_y_train_[valid_idx])

    tau_test = torch.from_numpy(tau_test)

    T_train = torch.from_numpy(T_train_[training_idx])
    T_valid = torch.from_numpy(T_train_[valid_idx])
    T_test = torch.from_numpy(T_test)

    return combine_x_train.to(device), combine_x_test.to(device), combined_y_train.to(device), combine_x_valid.to(
        device), combined_y_valid.to(device), tau_test.to(device), T_train.to(device), T_valid.to(device), T_test.to(
        device)


def trt_ctr(treatment):
    list1, list0 = [], []
    for index, i in enumerate(treatment):
        if i == 1:
            list1.append(index)
        elif i == 0:
            list0.append(index)
        else:
            raise TypeError('Invalid treatment value found')

    return list1, list0


def training_nn(x_1, y_1, x_0, y_0, training_iter, combine_x_valid, combined_y_valid, T_valid):
    input_dim, latent_dim, output_dim = x_1.shape[1], 200, 1

    model_1 = nnModel_1(input_dim, latent_dim, output_dim).to(device)
    model_0 = nnModel_0(input_dim, latent_dim, output_dim).to(device)
    criterion = nn.MSELoss()
    # Use the adam optimizer
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=1e-3,
                                   weight_decay=1e-4)  # Includes GaussianLikelihood parameters
    optimizer_0 = torch.optim.Adam(model_0.parameters(), lr=1e-3,
                                   weight_decay=1e-4)  # Includes GaussianLikelihood parameters

    list_1_valid, list_0_valid = trt_ctr(T_valid)
    last_value = float('inf')
    for i in range(training_iter):
        # Find optimal model hyperparameters
        model_1.train()
        model_0.train()

        # Zero gradients from previous iteration
        optimizer_1.zero_grad()
        # Output from model
        output_1 = model_1(x_1)
        # Calc loss and backprop gradients
        loss_1 = criterion(output_1, y_1)
        loss_1.backward()
        optimizer_1.step()

        # Zero gradients from previous iteration
        optimizer_0.zero_grad()
        # Output from model
        output_0 = model_0(x_0)
        # Calc loss and backprop gradients
        loss_0 = criterion(output_0, y_0)
        loss_0.backward()
        optimizer_0.step()

        model_1.eval()
        model_0.eval()

        with torch.no_grad():
            valid_pred_1 = model_1(combine_x_valid[list_1_valid])
            valid_pred_0 = model_0(combine_x_valid[list_0_valid])
        valid_loss = torch.mean((valid_pred_1 - combined_y_valid[list_1_valid]) ** 2) + torch.mean(
            (valid_pred_0 - combined_y_valid[list_0_valid]) ** 2)

        if last_value > valid_loss:
            # Save only the model state_dict (architecture and parameters)
            torch.save({
                'model_state_dict': model_1.state_dict(),
            }, 'model_selections/sim_nnmodel_1.pth')

            torch.save({
                'model_state_dict': model_0.state_dict(),
            }, 'model_selections/sim_nnmodel_0.pth')

            last_value = valid_loss

        if i % 500 == 0:
            print('Iter %d/%d - Loss: %.3f - Valid Loss: %.3f' % (
                i + 1,
                training_iter,
                loss_0.item(),
                valid_loss))

    # Load the model
    if True:
        model_1 = nnModel_1(input_dim, latent_dim, output_dim).to(device)
        model_0 = nnModel_0(input_dim, latent_dim, output_dim).to(device)

        checkpoint_1 = torch.load('model_selections/sim_nnmodel_1.pth')
        model_1.load_state_dict(checkpoint_1['model_state_dict'])

        checkpoint_0 = torch.load('model_selections/sim_nnmodel_0.pth')
        model_0.load_state_dict(checkpoint_0['model_state_dict'])

    return model_1, model_0


def evaluation_nn(pred_1, pred_0, test_tau, query_step):

    esti_tau = torch.from_numpy(pred_1 - pred_0).float()
    pehe_test = torch.sqrt(torch.mean((esti_tau - test_tau) ** 2))

    print('\n', 'PEHE at query step: {} is {}'.format(query_step, pehe_test), '\n')

    return pehe_test


def pool_updating(idx_remaining, idx_sub_training, querying_idx):
    # Update the training and pool set for the next AL stage
    idx_sub_training = np.concatenate((idx_sub_training, querying_idx), axis=0)  # Update the training pool
    # Update the remaining pool by deleting the selected data
    mask = np.isin(idx_remaining, querying_idx, invert=True)  # Create a mask that selects the elements to delete from array1
    idx_remaining = idx_remaining[mask]  # Update the remaining pool by subtracting the selected samples
    
    return idx_sub_training, idx_remaining


def one_side_uncertainty(combine_x_train, index, num_of_samples, model):
    model.eval()
    
    pred = model(combine_x_train[index])
    pred_variance = pred.variance.sqrt()
    
    uncertainty = pred_variance
    draw_dist = uncertainty.cpu().detach().numpy()
    #quantile_threshold = np.quantile(draw_dist, 1 - percentage)  # taking top 5% of the values
    
    top_k = num_of_samples
    threshold_top_k = np.partition(draw_dist, -top_k)[-top_k]  # Calculate the threshold for the top 5 values instead of top 5%
    print('Uncertainty threshold:', threshold_top_k)

    acquired_idx = []
    for idx, i in enumerate(draw_dist):
        # print(round(i.item(),2), round(uncertainty[idx].item(),2), round(uncertainty[idx].item()/uncertainty_mean.item(),2))
        if draw_dist[idx] >= threshold_top_k:
            acquired_idx.append(idx)
            
    #print('Top 5 uncertain:', draw_dist[acquired_idx])
    acquired_idx = index[acquired_idx]
    random_idx = np.random.permutation(len(acquired_idx))
    acquired_idx = acquired_idx[random_idx]
    
    num_elements_to_select = num_of_samples  # Selecting 5 values randomly as the step size
    
    return acquired_idx[:num_elements_to_select], threshold_top_k


# Function to calculate pairwise Euclidean distance in batches
def pairwise_distances_in_batches(data, batch_size=500):
    n = data.size(0)
    distances = torch.zeros(n, n, device=data.device)

    for i in range(0, n, batch_size):
        for j in range(i, n, batch_size):
            end_i = min(i + batch_size, n)
            end_j = min(j + batch_size, n)
            diff = data[i:end_i].unsqueeze(1) - data[j:end_j].unsqueeze(0)
            dist_batch = torch.sqrt(torch.sum(diff ** 2, dim=-1))
            distances[i:end_i, j:end_j] = dist_batch
            if i != j:
                distances[j:end_j, i:end_i] = dist_batch.T

    return distances


def initial_pruning(factual_index, G, counter_index):
    covered = factual_index.tolist()
    # Identify the direct neighbors (nodes with an incoming edge from the selected node)
    # Initialize an empty set to store unique neighbors

    all_neighbors = set()
    # Iterate over each node in the list
    for node in covered:
        neighbors = list(G.successors(node))  # Get the neighbors of the current node
        all_neighbors.update(neighbors)  # Add neighbors to the set (ensures uniqueness)
    # Convert the set back to a list (if needed)
    all_neighbors = list(all_neighbors)

    covered_neighbors = all_neighbors.copy()
    covered_nodes = covered + covered_neighbors  # treated_neighbours can include control samples

    # Initialize an empty list to store all incoming edges
    all_incoming_edges = []
    for node in covered_nodes:
        incoming_edges = list(G.in_edges(node))  # Get all incoming edges to the node
        # Accumulate the incoming edges into the all_incoming_edges list
        all_incoming_edges.extend(incoming_edges)
    print('Gathered all incoming edges','\n')
    # Filter incoming edges based on the condition: no control samples involved, i.e., none of the edge has control
    # Convert counter_index to a set for faster membership checking
    counter_index_set = set(counter_index)
    # Filter the incoming edges using the set for faster checks
    filtered_incoming_edges = [
        edge for edge in all_incoming_edges if edge[0] not in counter_index_set and edge[1] not in counter_index_set
    ]
    G.remove_edges_from(filtered_incoming_edges)  # Remove the incoming edges

    print('Covered nodes initial pruned on covered factual nodes completed','\n')
 
def initial_pruning_chunk(factual_index, G, counter_index, chunk_size=1000):
    covered = factual_index.tolist()

    all_neighbors = set()
    for node in covered:
        neighbors = list(G.successors(node))
        all_neighbors.update(neighbors)

    covered_neighbors = all_neighbors.copy()
    covered_nodes = covered + list(covered_neighbors)

    counter_index_set = set(counter_index)

    def process_chunk(chunk):
        all_incoming_edges = []
        for node in chunk:
            incoming_edges = list(G.in_edges(node))
            filtered_edges = [
                edge for edge in incoming_edges
                if edge[0] not in counter_index_set and edge[1] not in counter_index_set
            ]
            all_incoming_edges.extend(filtered_edges)
        G.remove_edges_from(all_incoming_edges)

    # Process nodes in chunks
    for i in range(0, len(covered_nodes), chunk_size):
        chunk = covered_nodes[i:i + chunk_size]
        process_chunk(chunk)

    print('Covered nodes initial pruned on covered factual nodes completed', '\n')


# Function to determine weight based on node classes
def get_weight(u, v):
    if node_classes[u] != node_classes[v]:
        return args.alpha  # Different classes
    else:
        return 1  # Same class


def compute_percentile_sparse(sparse_matrix, percentile=10):
    # Extract non-zero elements
    nonzero_elements = sparse_matrix.data
    
    # Calculate the percentile of non-zero elements
    percentile_value = np.percentile(nonzero_elements, percentile)
    return percentile_value


def create_large_graph(distance_matrix, threshold):
    # Convert the distance matrix to a sparse matrix if not already
    sparse_matrix = csr_matrix(distance_matrix)
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Iterate over the sparse matrix in chunks
    for i in range(sparse_matrix.shape[0]):
        row_start = sparse_matrix.indptr[i]
        row_end = sparse_matrix.indptr[i + 1]
        
        for j in range(row_start, row_end):
            if sparse_matrix.data[j] < threshold:
                G.add_edge(i, sparse_matrix.indices[j])

    return G                


def chunked_graph_construction(distance_matrix, threshold=0.5, chunk_size=1000):
    n = distance_matrix.shape[0]
    G = nx.DiGraph()
    
    for i in range(0, n, chunk_size):
        chunk = distance_matrix[i:i+chunk_size, :]
        rows, cols = np.where(chunk < threshold)
        edges = zip(rows + i, cols)
        G.add_edges_from(edges)
        print(f"Processed chunk {i//chunk_size + 1}/{n//chunk_size + 1}")
    
    return G


def acquiring(x_train, y_train, x_pool, custom_model, strategy, num_of_samples):

    train_data = TensorFeatureData(x_train)
    pool_data = TensorFeatureData(x_pool)

    new_idxs, _ = select_batch(batch_size=num_of_samples, models=[custom_model],
                               data={'train': train_data, 'pool': pool_data}, y_train=y_train,
                               selection_method=strategy, sel_with_train=args.selwithtrain,
                               base_kernel='grad', kernel_transforms=[('rp', [512])])

    return new_idxs

# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_size = 0.10
num_trial = 1
al_step = 21
warm_up = 1
num_of_samples = 1
seed = args.seed

if num_trial == 1:
# for seed in range(num_trial):
    
    print('Trial:', seed+1)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    combine_x, combined_y, tau, T = generation(mean_1=0, std_1=1, func_1=torch.sin, mean_0=3, std_0=1, func_0=torch.cos, device=device)
    combine_x_train, combine_x_test, combined_y_train, combine_x_valid, combined_y_valid, tau_test, T_train, T_valid, T_test = train_test_splitting(combine_x, combined_y, tau, T, test_size, seed, device=device)


    idx_pool = np.random.permutation(len(combine_x_train))  # Global dataset index
    idx_sub_training = idx_pool[:warm_up]  # Global dataset index
    idx_remaining = idx_pool[warm_up:]  # Global dataset index
    
    sub_training_1, sub_training_0 = trt_ctr(T_train[idx_sub_training])
    remaining_1, remaining_0 = trt_ctr(T_train[idx_remaining])
    
    # Initialize the data-limited starting size as 20% of whole treated training set
    idx_sub_training_1 = idx_sub_training[sub_training_1]  # 20% as initial
    idx_remaining_1 = idx_remaining[remaining_1]  # 20% left for querying
    
    # Initialize the data-limited starting size as 20% of whole control training set
    idx_sub_training_0 = idx_sub_training[sub_training_0]  # 10% as initial
    idx_remaining_0 = idx_remaining[remaining_0]  # 90% left for querying
    
    treated_index = idx_sub_training_1.tolist() + idx_remaining_1.tolist()
    control_index = idx_sub_training_0.tolist() + idx_remaining_0.tolist()

    # Calculate the number of elements to select (20% of idx_remaining_0)
    #num_to_select = int(len(idx_remaining_0) * 0.1)
    num_to_select = len(idx_remaining_0)
    # Randomly select 50% of elements from idx_remaining_0 without replacement
    selected_elements = np.random.choice(idx_remaining_0, num_to_select, replace=False)
    # Extend idx_sub_training_0 with the selected elements
    idx_sub_training_0 = np.concatenate((idx_sub_training_0, selected_elements))
    # Update idx_remaining_0 to remove the selected elements
    idx_remaining_0 = np.setdiff1d(idx_remaining_0, selected_elements, assume_unique=True)

    '''IMPORTANT: Exclude idx_sub_training_0 (inherit all idx_remaining_0) from idx_remaining!!!'''
    # Update idx_remaining to remove idx_remaining_0
    idx_remaining = np.setdiff1d(idx_remaining, idx_sub_training_0, assume_unique=True)

    acquired_treated, acquired_control = None, None
    error_list = []
    num_of_acquire = [len(idx_sub_training_1) + len(idx_sub_training_0)]

    # Newly added for graph construction
#    combine_x_train_features = torch.load('combine_x_train_features.pt', map_location=device)
    # compute the covering radius delta using the entire dataset
    # Compute pairwise distances in batches
    batch_size = 500  # Adjust the batch size according to your memory limits
    distances = pairwise_distances_in_batches(combine_x_train, batch_size)
    # Normalize the distances
    max_distance = torch.max(distances)
    normalized_distances = distances / max_distance
    print('Normalized maximum distance:', normalized_distances.max(), '\n')

    distances_between_treated = normalized_distances[treated_index][:, treated_index]
    # Step 1: Create a mask that excludes the diagonal
    mask = torch.eye(distances_between_treated.size(0), dtype=torch.bool).to(device)
    # Step 2: Apply the mask to exclude diagonal elements
    masked_distances = distances_between_treated.masked_fill(mask, float('inf'))
    # Step 3: Find the minimum value in the masked distance matrix
    min_value = masked_distances.min()
    print("Minimum non-diagonal distance in treated:", min_value.item())
    # Assuming `normalized_distance` is in some format like a dense NumPy array
    # Convert it to a sparse matrix format
    sparse_matrix = sp.csr_matrix(distances_between_treated.cpu().numpy())
    # Example usage
    percentile_value = compute_percentile_sparse(sparse_matrix)
    print(f'10th percentile value among treated: {percentile_value}', '\n')
 
    distances_between_control = normalized_distances[control_index][:, control_index]
    # Step 1: Create a mask that excludes the diagonal
    mask = torch.eye(distances_between_control.size(0), dtype=torch.bool).to(device)
    # Step 2: Apply the mask to exclude diagonal elements
    masked_distances = distances_between_control.masked_fill(mask, float('inf'))
    # Step 3: Find the minimum value in the masked distance matrix
    min_value = masked_distances.min()
    print("Minimum non-diagonal distance in control:", min_value.item())
    # Assuming `normalized_distance` is in some format like a dense NumPy array
    # Convert it to a sparse matrix format
    sparse_matrix = sp.csr_matrix(distances_between_control.cpu().numpy())
    # Example usage
    percentile_value = compute_percentile_sparse(sparse_matrix)
    print(f'10th percentile value among control: {percentile_value}', '\n')

    distances_between_groups = normalized_distances[treated_index][:, control_index]
    min_value = distances_between_groups.min()
    print("Minimum distance inter-groups:", min_value.item())

    # Assuming `normalized_distance` is in some format like a dense NumPy array
    # Convert it to a sparse matrix format
    sparse_matrix = sp.csr_matrix(distances_between_groups.cpu().numpy())
    # Example usage
    percentile_value = compute_percentile_sparse(sparse_matrix)
    print(f'10th percentile value inter-groups: {percentile_value}', '\n')

    # Compute the graph
    # Assume 'distances' is your distance matrix of shape [9540, 9540]
    # Step 1: Create an adjacency list or matrix
#    n = normalized_distances.size(0)
#    adjacency_list = {i: [] for i in range(n)}
    # Step 2: Populate the adjacency list based on the distance threshold
#    threshold = 0.48
    # Step 2: Create a mask where distances are less than the threshold and not on the diagonal
    # This mask identifies the valid edges
#    valid_edges = (normalized_distances < threshold) & (torch.eye(normalized_distances.size(0), device=device) == 0)
    # Step 3: Get the indices of the valid edges
#    rows, cols = torch.where(valid_edges)
    # Step 4: Construct the directed graph using NetworkX
#    G = nx.DiGraph()
    # Add nodes (optional, as NetworkX will add them automatically with edges)
#    G.add_nodes_from(range(normalized_distances.size(0)))
    # Add edges based on the indices
#    edges = list(zip(cols.cpu().numpy(), rows.cpu().numpy()))  # direction from cols to rows
#    G.add_edges_from(edges)
    radius = 0.11
    threshold_1, threshold_0, threshold_10, threshold_01 = radius, radius, radius, radius  # current result based on all 0.15
    normalized_distances = normalized_distances.cpu().numpy()
#    adj_matrix = (normalized_distances < threshold).astype(np.int8)
    # Convert indices to NumPy arrays if they are not already
    treated_index_ = np.array(treated_index)
    control_index_ = np.array(control_index)

    # Initialize an adjacency matrix with zeros
    adj_matrix = np.zeros_like(normalized_distances, dtype=np.int8)
 
    # 1. Edges within treated_index using threshold_1
    treated_mask = np.ix_(treated_index_, treated_index_)
    adj_matrix[treated_mask] = (normalized_distances[treated_mask] < threshold_1).astype(np.int8)
 
    # 2. Edges within control_index using threshold_0
    control_mask = np.ix_(control_index_, control_index_)
    adj_matrix[control_mask] = (normalized_distances[control_mask] < threshold_0).astype(np.int8)
 
    # 3. Edges between treated_index and control_index using threshold_10
    treated_control_mask = np.ix_(treated_index_, control_index_)
    control_treated_mask = np.ix_(control_index_, treated_index_)
    adj_matrix[treated_control_mask] = (normalized_distances[treated_control_mask] < threshold_10).astype(np.int8)
    adj_matrix[control_treated_mask] = (normalized_distances[control_treated_mask] < threshold_01).astype(np.int8)
 
    # Exclude self-loops by setting the diagonal to 0 (optional)
    np.fill_diagonal(adj_matrix, 0)
    # Convert the adjacency matrix to a sparse format
    sparse_adj_matrix = csr_matrix(adj_matrix)
    # Now, use the sparse adjacency matrix to create the graph
    G = nx.from_scipy_sparse_array(sparse_adj_matrix, create_using=nx.DiGraph)
    print('Graph construction completed','\n')

#    torch.cuda.empty_cache()
#    print('Free Up','\n')

    # Identify all existing nodes to remove incoming edges from
    # For each of the treatment group, we need to break the edges only on that group and keep the otherside untouched.

    # Initialize an empty dictionary
    node_classes = {}
    # Assign class 1 to nodes in list_1
    for node in treated_index:
        node_classes[node] = 1
    # Assign class 0 to nodes in list_0
    for node in control_index:
        node_classes[node] = 0
    # Now node_classes is a dictionary mapping nodes to their classes
    # Assign weights to edges based on the condition
    for u, v in G.edges():
        weight = get_weight(u, v)
        G[u][v]['weight'] = weight


    initial_pruning_chunk(idx_sub_training_1, G, control_index)
    initial_pruning_chunk(idx_sub_training_0, G, treated_index)

    # Compute weighted out-degrees
    out_degrees = dict(G.out_degree(weight='weight'))

    # Normalize out-degrees for color mapping
    max_out_degree = max(out_degrees.values())
    min_out_degree = min(out_degrees.values())
    norm = plt.Normalize(vmin=min_out_degree, vmax=max_out_degree)  # Ensure correct range for normalization

    # Extract node positions (assuming they are already extracted or available)
    positions = {node: (x, y) for node, (x, y) in enumerate(zip(
        combine_x_train[:, 0].cpu().numpy(),
        combine_x_train[:, 1].cpu().numpy()
    ))}

    # Create a custom colormap (orange with varying transparency)
    colors = [(1, 0.5, 0, alpha) for alpha in np.linspace(0, 1, 256)]  # RGBA for orange
    orange_cmap = mcolors.LinearSegmentedColormap.from_list("OrangeTransparent", colors)

    # Map normalized out-degrees to colors using the custom colormap
    node_colors = {node: orange_cmap(norm(out_degrees[node])) for node in out_degrees}

    # Scale marker sizes for additional visual distinction
    marker_sizes = {node: 10 + 50 * (out_degrees[node] / max_out_degree) for node in out_degrees}

    # Filter nodes to include only those in idx_remaining_1
    remaining_nodes = idx_remaining_1  # Convert to NumPy array

    # Filter relevant properties
    filtered_out_degrees = {node: out_degrees[node] for node in remaining_nodes if node in out_degrees}
    filtered_positions = {node: positions[node] for node in remaining_nodes if node in positions}
    filtered_marker_sizes = {node: marker_sizes[node] for node in remaining_nodes if node in marker_sizes}

    # Re-normalize out-degrees for the filtered nodes
    max_out_degree_filtered = max(filtered_out_degrees.values())
    min_out_degree_filtered = min(filtered_out_degrees.values())
    filtered_norm = plt.Normalize(vmin=min_out_degree_filtered, vmax=max_out_degree_filtered)

    # Create a scatter plot for the heatmap
    plt.figure(figsize=(10, 8))

    for node, (x, y) in filtered_positions.items():
        color = orange_cmap(filtered_norm(filtered_out_degrees[node]))
        plt.scatter(x, y, color=color, s=filtered_marker_sizes[node], alpha=0.2, edgecolors='none', linewidths=0)

    # Add a colorbar for the heatmap
    sm = plt.cm.ScalarMappable(cmap=orange_cmap, norm=filtered_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label("Out-Degree (Filtered)", rotation=270, labelpad=20)

    # Add labels and title
    #plt.title("Vertices' Out-Degree Heatmap (Filtered by idx_remaining_1)")
    #plt.xlabel("Feature 1")
    #plt.ylabel("Feature 2")
    #plt.xlim([-6, 6])
    #plt.ylim([-6, 6])
    plt.axis('equal')
    plt.grid(True)

    for query_step in range(al_step):
        
        train_x_1, train_y_1 = combine_x_train[idx_sub_training_1], combined_y_train[idx_sub_training_1]
        train_x_0, train_y_0 = combine_x_train[idx_sub_training_0], combined_y_train[idx_sub_training_0]
        print("Number of data used for training in treated and control:", len(idx_sub_training_1), len(idx_sub_training_0))

        # Data preparation
        class_1_data = combine_x_train[idx_remaining_1].cpu().numpy()  # Class 1
        class_0_data = combine_x_train[idx_sub_training_0].cpu().numpy()  # Class 0

        # Combine data for grid preparation
        all_data = np.vstack([class_1_data, class_0_data])

        # Define grid for KDE
        x_min, x_max = all_data[:, 0].min() - 1, all_data[:, 0].max() + 1
        y_min, y_max = all_data[:, 1].min() - 1, all_data[:, 1].max() + 1
        x, y = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        grid_coords = np.vstack([x.ravel(), y.ravel()]).T

        # KDE for each class
        kde_class_1 = gaussian_kde(class_1_data.T)
        kde_class_0 = gaussian_kde(class_0_data.T)

        # Evaluate densities on the grid
        density_class_1 = kde_class_1(grid_coords.T).reshape(x.shape)
        density_class_0 = kde_class_0(grid_coords.T).reshape(x.shape)

        # Compute overlap density
        overlap_density = np.minimum(density_class_1, density_class_0)

        # Automatically generate levels and select two specific levels
        contour = plt.contour(x, y, overlap_density, levels=5, colors='none')
        selected_levels = [contour.levels[0], contour.levels[1]]  # Select the outermost and the next level

        # Plot the two selected contours
        plt.contour(x, y, overlap_density, levels=selected_levels, colors=['green', 'purple'], linestyles=['--', '--'],
                    linewidths=0.5)

        # Add labels, title, and grid
        #plt.title("Two Levels of Overlap Density Contours Between Class 1 and Class 0")
        #plt.xlabel("Feature 1")
        #plt.ylabel("Feature 2")
        plt.grid(True)
        plt.xlim([-10, 12.5])
        plt.ylim([-10, 12.5])
        # Plot remaining data for Class 1 (points not yet queried)
        #plt.scatter(combine_x_train[idx_remaining_1][:, 0].cpu().numpy(),
        #            combine_x_train[idx_remaining_1][:, 1].cpu().numpy(),
        #            color='red', label="Class 1 - Remaining", alpha=0.1, s=10, marker='o')

        ## Plot training data for Class 0 (points not yet queried)
        #plt.scatter(combine_x_train[idx_sub_training_0][:, 0].cpu().numpy(),
        #            combine_x_train[idx_sub_training_0][:, 1].cpu().numpy(),
        #            color='blue', label="Class 0 - training", alpha=0.1, s=10, marker='o')

        #plt.show()

        # Concatenate vertically and shuffle randomly for the sub training
        combine_train_idx = np.concatenate([idx_sub_training_1, idx_sub_training_0], axis=0)
        np.random.shuffle(combine_train_idx)
        train_dataset = toDataLoader(x_train=combine_x_train[combine_train_idx].cpu(),
                                     y_train=combined_y_train[combine_train_idx].cpu(),
                                     t_train=T_train[combine_train_idx].cpu())

        tune_dataset = toDataLoader(x_train=combine_x_valid.cpu(),
                                    y_train=combined_y_valid.cpu(),
                                    t_train=T_valid.cpu())

        job_dir_path = Path(
            'saved_models/selection_visuals/method_{}/seed_{}/step_{}'.format(config.get("acquisition_function"),
                                                                                   seed, 0))

        train_deep_kernel_gp(ds_train=train_dataset,
                             ds_valid=tune_dataset,
                             job_dir=job_dir_path,
                             config=config,
                             dim_input=train_x_1.shape[1],
                             seed=seed)

        test_dataset = toDataLoader(x_train=combine_x_test.cpu(),
                                    y_train=tau_test.cpu(),
                                    t_train=T_test.cpu())

        (mu_0, mu_1), model = predict_deep_kernel_gp(dataset=test_dataset,
                                                    job_dir=job_dir_path,
                                                    config=config,
                                                    dim_input=train_x_1.shape[1],
                                                    seed=seed
                                                    )

        pehe_error = evaluation_nn(pred_1=mu_1.mean(0),
                                   pred_0=mu_0.mean(0),
                                   test_tau=tau_test.cpu(),
                                   query_step=query_step
                                   )

        error_list.append(np.round(pehe_error.cpu().numpy(), 4))

        inputs = torch.hstack([combine_x_train[combine_train_idx], T_train[combine_train_idx].reshape(-1, 1)]).float()
        pool = torch.hstack([combine_x_train[idx_remaining], T_train[idx_remaining].reshape(-1, 1)]).float()
        targets = combined_y_train[combine_train_idx]

        acquired_index = acquiring(x_train=inputs,
                                     y_train=targets,
                                     x_pool=pool,
                                     custom_model=model.network,
                                     strategy='{}'.format(args.bmdal),
                                     num_of_samples=num_of_samples)

        # The acquired index is actually the relative index to the pool index!!!
        acquired_index = idx_remaining[acquired_index.cpu()]
        ''' The following line 428 especially for only one sample query, otherwise deleted'''
        acquired_index = np.array([acquired_index])

        # Random sampling throughout the AL
        # acquired_index = random_sampling(pool=idx_remaining, num_of_samples=num_of_samples)
        print('acquired index:', acquired_index, type(acquired_index))
        treated_idx, control_idx = trt_ctr(T_train[acquired_index])
        acquired_treated, acquired_control = acquired_index[treated_idx], acquired_index[control_idx]

        if len(acquired_treated) == 0 and len(acquired_control) == 0:
            raise TypeError("Nothing acquired by AL")
        else:
            if query_step != 0:
                num_of_acquire.append(num_of_acquire[query_step - 1] + len(acquired_treated) + len(acquired_control))
            # plot(combine_x_train, combined_y_train, model_1, model_0, likelihood_1, likelihood_0, train_x_1, train_x_0, train_y_1, train_y_0, acquired_treated, acquired_control)

        if len(acquired_treated) != 0:
            idx_sub_training_1, idx_remaining = pool_updating(idx_remaining, idx_sub_training_1, acquired_treated)

            best_node_ = acquired_treated
            # Plot training data for Class 1
            plt.scatter(
                combine_x_train[best_node_][:, 0].cpu().numpy(),
                combine_x_train[best_node_][:, 1].cpu().numpy(),
                color='red', label="Selected", alpha=1.0, s=200, marker='*'
            )

            # Annotate each point with its query step number
            for i, point in enumerate(combine_x_train[best_node_]):
                plt.text(
                    point[0].cpu().numpy(),  # X-coordinate
                    point[1].cpu().numpy(),  # Y-coordinate
                    str(query_step + 1),  # Text label (query step number)
                    fontsize=32, color='black', ha='center', va='top'  # Styling
                )

        if len(acquired_control) != 0:
            idx_sub_training_0, idx_remaining = pool_updating(idx_remaining, idx_sub_training_0, acquired_control)

            best_node_ = acquired_treated
            # Plot training data for Class 1
            plt.scatter(
                combine_x_train[best_node_][:, 0].cpu().numpy(),
                combine_x_train[best_node_][:, 1].cpu().numpy(),
                color='red', label="Selected", alpha=1.0, s=200, marker='*'
            )

            # Annotate each point with its query step number
            for i, point in enumerate(combine_x_train[best_node_]):
                plt.text(
                    point[0].cpu().numpy(),  # X-coordinate
                    point[1].cpu().numpy(),  # Y-coordinate
                    str(query_step + 1),  # Text label (query step number)
                    fontsize=32, color='black', ha='center', va='top'  # Styling
                )

        # Plot training data for Class 0
        #plt.scatter(train_x_0[:, 0].cpu().numpy(),
        #            train_x_0[:, 1].cpu().numpy(), color='blue', label="Class 0 - Training", alpha=0.15, s=10, marker='o')

        # Draw circles around each point in train_x_1
        #for point in train_x_1:
        #    circle = plt.Circle((
        #        point[0], point[1]),
        #        max_distance * radius,
        #        color='green', alpha=0.5, fill=False, linestyle='--'
        #    )
        #    plt.gca().add_patch(circle)

        # Add legend and show the final combined plot
        plt.savefig('data_acquisition_plot/true{}/query_step_{}.pdf'.format(args.bmdal, query_step), bbox_inches='tight')

