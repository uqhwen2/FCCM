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
from models.nn_model import nnModel_1, nnModel_0
from pathlib import Path
from models.nn_model import train_deep_kernel_gp, predict_deep_kernel_gp

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import networkx as nx
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--alpha', type=float, default=2.5)
parser.add_argument('--radius', type=float, default=0.40)
parser.add_argument('--path', type=str, default='')
args = parser.parse_args()

import json
# Read the configuration from the file
with open('config_probcover.json', 'r') as file:
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
    # Training data is 100 points in [0,1] inclusive regularly spaced
    train_x_1 = torch.linspace(-15, 10, 200)
    # True function is sin(2*pi*x) with Gaussian noise
    train_y_1 = func_1(train_x_1 * (2 * math.pi) / math.pi)

    # Generate Gaussian noise
    noise_x_1 = torch.normal(mean=mean_1, std=std_1, size=(1000,))  

    noise_y_1 = func_1(noise_x_1 * (2 * math.pi) / math.pi)
    
    train_x_1 = torch.cat([train_x_1, noise_x_1], dim=0)
    train_y_1 = torch.cat([train_y_1, noise_y_1], dim=0)
    random_idx = np.random.permutation(len(train_x_1))

    train_x_1 = train_x_1[random_idx]
    train_y_1 = train_y_1[random_idx]

    # Training data is 100 points in [0,1] inclusive regularly spaced
    train_x_0 = torch.linspace(-10, 15, 200)
    # True function is sin(2*pi*x) with Gaussian noise
    train_y_0 = func_0(train_x_0 * (2 * math.pi) / math.pi)

    # Generate Gaussian noise
    noise_x_0 = torch.normal(mean=mean_0, std=std_0, size=(1000,)) 

    noise_y_0 = func_0(noise_x_0 * (2 * math.pi) / math.pi) 

    train_x_0 = torch.cat([train_x_0, noise_x_0], dim=0)
    train_y_0 = torch.cat([train_y_0, noise_y_0], dim=0)
    random_idx = np.random.permutation(len(train_x_0))

    train_x_0 = train_x_0[random_idx]
    train_y_0 = train_y_0[random_idx]

    combine_x = torch.cat([train_x_1, train_x_0], dim=0)
    combined_y = torch.cat([train_y_1, train_y_0], dim=0)
    combine_y_1 = func_1(combine_x * (2 * math.pi) / math.pi) 
    combine_y_0 = func_0(combine_x * (2 * math.pi) / math.pi) 
    tau = combine_y_1 - combine_y_0
    
    treated_x = torch.ones_like(train_x_1)
    control_x = torch.zeros_like(train_x_0)
    T = torch.cat([treated_x, control_x], dim=0)

    return combine_x.to(device), combined_y.to(device), tau.to(device), T.to(device)


def train_test_splitting(combine_x, combined_y, tau, T, test_size, seed, device):
    # Convert tensors to numpy arrays
    combine_x_np = combine_x.cpu().numpy()
    combined_y_np = combined_y.cpu().numpy()
    tau_np = tau.cpu().numpy()
    T_np = T.cpu().numpy()
    
    combine_x_train, combine_x_test, \
    combined_y_train, combined_y_test, \
    tau_train, tau_test, \
    T_train, T_test = train_test_split(combine_x_np, combined_y_np, tau_np, T_np, test_size=test_size, random_state=seed)
    
    ihdp_train = HCMNIST(root='assets', split='train', mode='mu', seed=seed)
    ihdp_test = HCMNIST(root='assets', split='valid', mode='mu', seed=seed)

    valid_size = 0.25
    training_idx, valid_idx = train_test_split(list(range(ihdp_train.x.shape[0])),
                                               test_size=valid_size,
                                               random_state=seed)

    # Convert back to PyTorch tensors if needed
    combine_x_train = torch.from_numpy(ihdp_train.x[training_idx])
    combine_x_valid = torch.from_numpy(ihdp_train.x[valid_idx])
    combine_x_test = torch.from_numpy(ihdp_test.x)

    combined_y_train = torch.from_numpy(ihdp_train.y[training_idx])  # No y normalization
    combined_y_valid = torch.from_numpy(ihdp_train.y[valid_idx])

    tau_test = torch.from_numpy(ihdp_test.tau)

    T_train = torch.from_numpy(ihdp_train.t[training_idx])
    T_valid = torch.from_numpy(ihdp_train.t[valid_idx])
    T_test = torch.from_numpy(ihdp_test.t)

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


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_size = 0.1
num_trial = 1
al_step = 50
warm_up = 50
num_of_samples = 50
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
#    num_to_select = int(len(idx_remaining_0) * 0.20)
    num_to_select = len(idx_remaining_0)
    # Randomly select 50% of elements from idx_remaining_0 without replacement
    selected_elements = np.random.choice(idx_remaining_0, num_to_select, replace=False)
    # Extend idx_sub_training_0 with the selected elements
    idx_sub_training_0 = np.concatenate((idx_sub_training_0, selected_elements))
    # Update idx_remaining_0 to remove the selected elements
    idx_remaining_0 = np.setdiff1d(idx_remaining_0, selected_elements, assume_unique=True)

    print('val:', len(T_valid), 'test:', len(T_test),"treated remaining:", len(idx_remaining_1)+len(idx_sub_training_1))

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
    '''Turn counterfactual covering radii to 0 for ablation study'''
    threshold_1, threshold_0, threshold_10, threshold_01 = 0.50, 0.20, args.radius, args.radius  # 0.50, 0.20, 0.40, 0.15 used for results
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
    
    for query_step in range(al_step):
        
        train_x_1, train_y_1 = combine_x_train[idx_sub_training_1], combined_y_train[idx_sub_training_1]
        train_x_0, train_y_0 = combine_x_train[idx_sub_training_0], combined_y_train[idx_sub_training_0]
        print("Number of data used for training in treated and control:", len(idx_sub_training_1), len(idx_sub_training_0))

        # Concatenate vertically and shuffle randomly for the sub training
        combine_train_idx = np.concatenate([idx_sub_training_1, idx_sub_training_0], axis=0)
        np.random.shuffle(combine_train_idx)
        train_dataset = toDataLoader(x_train=combine_x_train[combine_train_idx].cpu(),
                                     y_train=combined_y_train[combine_train_idx].cpu(),
                                     t_train=T_train[combine_train_idx].cpu())

        tune_dataset = toDataLoader(x_train=combine_x_valid.cpu(),
                                    y_train=combined_y_valid.cpu(),
                                    t_train=T_valid.cpu())

        job_dir_path = Path('saved_models/method_{}_{}/seed_{}/step_{}'.format(config.get("acquisition_function"), args.alpha,seed, 0))

        train_deep_kernel_gp(ds_train=train_dataset,
                             ds_valid=tune_dataset,
                             job_dir=job_dir_path,
                             config=config,
                             dim_input=[1, 28, 28],
                             seed=seed)

        test_dataset = toDataLoader(x_train=combine_x_test.cpu(),
                                    y_train=tau_test.cpu(),
                                    t_train=T_test.cpu())

        (mu_0, mu_1), _ = predict_deep_kernel_gp(dataset=test_dataset,
                                                 job_dir=job_dir_path,
                                                 config=config,
                                                 dim_input=[1, 28, 28],
                                                 seed=seed
                                                )

        pehe_error = evaluation_nn(pred_1=mu_1.mean(0),
                                   pred_0=mu_0.mean(0),
                                   test_tau=tau_test.cpu(),
                                   query_step=query_step
                                   )

        error_list.append(np.round(pehe_error.cpu().numpy(), 4))

        selected_points = idx_sub_training_1.tolist() + idx_sub_training_0.tolist()
        # for b in range(num_of_samples * 2):
#        for b in range(num_of_samples):
        b = 0
        while b < num_of_samples:
            # Compute out-degree for each vertex
            out_degrees = dict(G.out_degree(weight='weight'))

            # Filter out the nodes from list_1
            filtered_out_degrees = {node: degree for node, degree in out_degrees.items() if node not in selected_points}

            #            # Find the node with the highest out-degree from the filtered dictionary
            #            max_out_degree_node = max(filtered_out_degrees, key=filtered_out_degrees.get)

            # Step 1: Calculate the ratio for each node
            node_ratios = {}
            # Assuming treated_index and control_index are sets for faster lookup
            treated_index_set = set(treated_index)
            control_index_set = set(control_index)

            for node in filtered_out_degrees.keys():

                neighbors = set(G.successors(node))
                # Count neighbors in each class
                treated_neighbors = len(neighbors & treated_index_set)
                control_neighbors = len(neighbors & control_index_set)
                number_of_neighbors = treated_neighbors + control_neighbors

#                # Calculate the ratio
#                if node in control_index:  # We add  +1 to control just count itself into that
#                    ratio = treated_neighbors * (control_neighbors + 0) / number_of_neighbors ** 2 if number_of_neighbors > 0 else 0
#                elif node in treated_index:
#                    ratio = control_neighbors * (treated_neighbors + 0) / number_of_neighbors ** 2 if number_of_neighbors > 0 else 0
#                else:
#                    raise TypeError('Index out of nowhere')
                    # Calculate the ratio
                if node in control_index:
                    if control_neighbors == 0 and treated_neighbors > 0:
                        ratio = 1  # Assign ratio as 1 if there are no control neighbors but some treated neighbors
                    else:
                        ratio = treated_neighbors * (control_neighbors + 0) / number_of_neighbors ** 2 if number_of_neighbors > 0 else 0
                elif node in treated_index:
                    if treated_neighbors == 0 and control_neighbors > 0:
                        ratio = 1  # Assign ratio as 1 if there are no treated neighbors but some control neighbors
                    else:
                        ratio = control_neighbors * (treated_neighbors + 0) / number_of_neighbors ** 2 if number_of_neighbors > 0 else 0
                else:
                    raise TypeError('Index out of nowhere')

                node_ratios[node] = ratio

            # Step 2: Multiply the ratio with the out-degree and find the node with the highest value
            max_product_value_1 = float('-inf')
            best_node_1 = None

            max_product_value_0 = float('-inf')
            best_node_0 = None

            max_product_value_10 = float('-inf')
            best_node_10 = None

            for node, degree in filtered_out_degrees.items():
                ratio = node_ratios.get(node, 0)
                product_value = degree * ratio

                if ratio !=0 and ratio != 1:
                    # Update the best node if this product is higher
                    if product_value > max_product_value_1:
                        max_product_value_1 = product_value
                        best_node_1 = node
                elif ratio == 1:
                    if degree > max_product_value_10:
                        max_product_value_10 = degree
                        best_node_10 = node
                elif ratio == 0:
                    if degree > max_product_value_0:
                        max_product_value_0 = degree
                        best_node_0 = node
                else:
                    raise TypeError('Ratio not real number!')

            if best_node_1 is None:
                if best_node_10 is None:
                    best_node = best_node_0
                    max_product_value = max_product_value_0
                else:
                    best_node = best_node_10
                    max_product_value = max_product_value_10
            else:
                best_node = best_node_1
                max_product_value = max_product_value_1

            if best_node>=0:
                print(
                    f"{b} Node with the highest product value: {best_node} with value {max_product_value} and ratio {node_ratios.get(best_node, 0)}")
                # Continue with your logic using best_node
            else:
                raise TypeError("No suitable node found.")

            if best_node in treated_index:
                counter_index = control_index.copy()
                print('Treated Selected')
                idx_sub_training_1 = np.concatenate((idx_sub_training_1, [best_node]), axis=0)
            elif best_node in control_index:
                counter_index = treated_index.copy()
                print('Control Selected')
                idx_sub_training_0 = np.concatenate((idx_sub_training_0, [best_node]), axis=0)
            else:
                raise TypeError('Index out of nowhere')

            # print(f"Node with the highest out-degree: {max_out_degree_node}",
            #     'with out-degree:', out_degrees[max_out_degree_node])

            # Step 1: Identify the direct neighbors (nodes with an incoming edge from the selected node)
            neighbors = list(G.successors(best_node))

            # Step 2: Identify all nodes to remove incoming edges from (include the picked node)
            nodes_to_modify = [best_node] + neighbors

            # Initialize an empty list to store all incoming edges
            all_incoming_edges = []
            for node in nodes_to_modify:
                incoming_edges = list(G.in_edges(node))  # Get all incoming edges to the node
                # Accumulate the incoming edges into the all_incoming_edges list
                all_incoming_edges.extend(incoming_edges)
            # print('Gathered all incoming edges', '\n')
            # Filter incoming edges based on the condition: no control samples involved, i.e., none of the edge has control
            # Convert counter_index to a set for faster membership checking
            counter_index_set = set(counter_index)
            # Filter the incoming edges using the set for faster checks
            filtered_incoming_edges = [
                edge for edge in all_incoming_edges if
                edge[0] not in counter_index_set and edge[1] not in counter_index_set
            ]
            G.remove_edges_from(filtered_incoming_edges)  # Remove the incoming edges

#            # Further processing for the best_node's neighbors
#            counter_neighbors = [neighbor for neighbor in neighbors if neighbor in counter_index_set]
#            # Check if any counter neighbor is in the selected_points list
#            selected_counter_neighbors = [neighbor for neighbor in counter_neighbors if neighbor in selected_points]
#            if not selected_counter_neighbors:  # If selected counter neighbors exist
#                # Find the neighbor with the highest out-degree
#                highest_out_degree_neighbor = max(counter_neighbors, key=lambda n: G.out_degree[n])
#                # For example, you can add it to selected_points or perform further actions
#                selected_points.append(highest_out_degree_neighbor)  # Example action
#                b+=1
#
#                if best_node in treated_index:
#                    idx_sub_training_0 = np.concatenate((idx_sub_training_0, [highest_out_degree_neighbor]), axis=0)
#                    print('Added one additional counterfactual covered sample to the control')
#                elif best_node in control_index:
#                    idx_sub_training_1 = np.concatenate((idx_sub_training_1, [highest_out_degree_neighbor]), axis=0)
#                    print('Added one additional counterfactual covered sample to the treated')
#                else:
#                    raise TypeError('Index out of nowhere')
#
#            else:
#                print('Counterfactual covered')

            selected_points.extend([best_node])
            b+=1

        if query_step != 0:
            num_of_acquire.append(len(selected_points) - num_of_samples)

average_pehe = np.array(error_list)

# Specify the file pathstep_1
file_path = 'text_results/probcover/pehe_probcover_{}_{}.csv'.format(args.alpha, args.alpha, args.seed)

# Open the CSV file in write mode
with open(file_path, 'w', newline='') as csvfile:
    # Create a CSV writer object
    csv_writer = csv.writer(csvfile)

    # Write the data for list_1
    csv_writer.writerow(['Number of Samples'] + list(map(str, num_of_acquire)))

    # Write the data for list_2
    csv_writer.writerow(['PEHE'] + list(map(str, average_pehe.tolist())))

print(f'The data has been successfully written to {file_path}')
