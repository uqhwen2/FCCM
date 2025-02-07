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

from sklearn.manifold import TSNE
import networkx as nx
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--alpha', type=float, default=2.5)
parser.add_argument('--radius', type=float, default=0.12)
args = parser.parse_args()

import json

# Read the configuration from the file
with open('experiments/config_fccm.json', 'r') as file:
    config = json.load(file)

from torch.utils.data import Dataset

from models.utils import train_test_splitting


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


def evaluation_nn(pred_1, pred_0, test_tau, query_step):
    esti_tau = torch.from_numpy(pred_1 - pred_0).float()
    pehe_test = torch.sqrt(torch.mean((esti_tau - test_tau) ** 2))

    print('\n', 'PEHE at query step: {} is {}'.format(query_step, pehe_test), '\n')

    return pehe_test


def pool_updating(idx_remaining, idx_sub_training, querying_idx):
    # Update the training and pool set for the next AL stage
    idx_sub_training = np.concatenate((idx_sub_training, querying_idx), axis=0)  # Update the training pool
    # Update the remaining pool by deleting the selected data
    mask = np.isin(idx_remaining, querying_idx,
                   invert=True)  # Create a mask that selects the elements to delete from array1
    idx_remaining = idx_remaining[mask]  # Update the remaining pool by subtracting the selected samples

    return idx_sub_training, idx_remaining


def one_side_uncertainty(combine_x_train, index, num_of_samples, model):
    model.eval()

    pred = model(combine_x_train[index])
    pred_variance = pred.variance.sqrt()

    uncertainty = pred_variance
    draw_dist = uncertainty.cpu().detach().numpy()
    # quantile_threshold = np.quantile(draw_dist, 1 - percentage)  # taking top 5% of the values

    top_k = num_of_samples
    threshold_top_k = np.partition(draw_dist, -top_k)[
        -top_k]  # Calculate the threshold for the top 5 values instead of top 5%
    print('Uncertainty threshold:', threshold_top_k)

    acquired_idx = []
    for idx, i in enumerate(draw_dist):
        # print(round(i.item(),2), round(uncertainty[idx].item(),2), round(uncertainty[idx].item()/uncertainty_mean.item(),2))
        if draw_dist[idx] >= threshold_top_k:
            acquired_idx.append(idx)

    # print('Top 5 uncertain:', draw_dist[acquired_idx])
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
al_step = 60
warm_up = 100
num_of_samples = 100
seed = args.seed

if num_trial == 1:
    # for seed in range(num_trial):

    print('Trial:', seed)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    combine_x_train, \
        combine_x_test, \
        combined_y_train, \
        combine_x_valid, \
        combined_y_valid, \
        tau_test, T_train, \
        T_valid, \
        T_test, \
        y_std = train_test_splitting(seed, device=device)

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

    '''
    # Calculate the number of elements to select (20% of idx_remaining_0)
    num_to_select = int(len(idx_remaining_0) * 1)
    # Randomly select 50% of elements from idx_remaining_0 without replacement
    selected_elements = np.random.choice(idx_remaining_0, num_to_select, replace=False)
    # Extend idx_sub_training_0 with the selected elements
    idx_sub_training_0 = np.concatenate((idx_sub_training_0, selected_elements))
    # Update idx_remaining_0 to remove the selected elements
    idx_remaining_0 = np.setdiff1d(idx_remaining_0, selected_elements, assume_unique=True)
    '''

    acquired_treated, acquired_control = None, None
    error_list = []
    num_of_acquire = [len(idx_sub_training_1) + len(idx_sub_training_0)]

    # Newly added for graph construction

    batch_size = 500  # Adjust the batch size according to your memory limits
    distances = pairwise_distances_in_batches(combine_x_train, batch_size)
    # Normalize the distances
    max_distance = torch.max(distances)
    normalized_distances = distances / max_distance
    save_tensor_normalized_distances = normalized_distances.clone()
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

    '''Set covering radii'''
    threshold_1, threshold_0, threshold_10, threshold_01 = args.radius, args.radius, args.radius, args.radius
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

    coverage_list_11, coverage_list_10 = [], []
    coverage_list_00, coverage_list_01 = [], []
    for query_step in range(al_step):
        print('Query Step:', query_step + 1)

        train_x_1, train_y_1 = combine_x_train[idx_sub_training_1], combined_y_train[idx_sub_training_1]
        train_x_0, train_y_0 = combine_x_train[idx_sub_training_0], combined_y_train[idx_sub_training_0]
        print("Number of data used for training in treated and control:", len(idx_sub_training_1),
              len(idx_sub_training_0))

        selected_points = idx_sub_training_1.tolist() + idx_sub_training_0.tolist()
        #for b in range(num_of_samples * 2):
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

            selected_points.extend([best_node])
            b+=1

        if query_step != 0:
            num_of_acquire.append(len(selected_points)-num_of_samples)

            # Index masks for sub_training and treated points
            mask_sub_training_1 = torch.tensor(
                [i in idx_sub_training_1.tolist() for i in range(save_tensor_normalized_distances.shape[0])])
            mask_treated_index = torch.tensor(
                [i in treated_index for i in range(save_tensor_normalized_distances.shape[0])])

            # Extract the sub-matrix of distances from sub_training_1 to all treated indices
            distances_sub_training_to_treated = save_tensor_normalized_distances[mask_sub_training_1][:,
                                                mask_treated_index]

            # Check for coverage within distance 0.5
            coverage_mask = distances_sub_training_to_treated <= args.radius

            # Sum over each column to see if any point in sub_training_1 covers this specific treated index
            covered_points_11 = coverage_mask.any(dim=0).sum().item()
            coverage_11 = covered_points_11 / len(treated_index) * 100
            print(
                f"Number of points in treated_index covered by idx_sub_training_1 within distance {args.radius}: {round(coverage_11, 2)}")

            # Index masks for sub_training and control points
            mask_control_index = torch.tensor(
                [i in control_index for i in range(save_tensor_normalized_distances.shape[0])])

            # Extract the sub-matrix of distances from sub_training_1 to all control indices
            distances_sub_training_to_control = save_tensor_normalized_distances[mask_sub_training_1][:,
                                                mask_control_index]

            # Check for coverage within distance 0.15
            coverage_mask = distances_sub_training_to_control <= args.radius

            # Sum over each column to see if any point in sub_training_1 covers this specific control index
            covered_points_10 = coverage_mask.any(dim=0).sum().item()
            coverage_10 = covered_points_10 / len(control_index) * 100

            print(
                f"Number of points in control_index covered by idx_sub_training_1 within distance {args.radius}: {round(coverage_10, 2)}")

            coverage_list_11.append(coverage_11)
            coverage_list_10.append(coverage_10)

            # Index masks for sub_training and treated points
            mask_sub_training_0 = torch.tensor(
                [i in idx_sub_training_0.tolist() for i in range(save_tensor_normalized_distances.shape[0])])
            mask_control_index = torch.tensor(
                [i in control_index for i in range(save_tensor_normalized_distances.shape[0])])

            # Extract the sub-matrix of distances from sub_training_1 to all treated indices
            distances_sub_training_to_control = save_tensor_normalized_distances[mask_sub_training_0][:,
                                                mask_control_index]

            # Check for coverage within distance 0.5
            coverage_mask = distances_sub_training_to_control <= args.radius

            # Sum over each column to see if any point in sub_training_1 covers this specific treated index
            covered_points_00 = coverage_mask.any(dim=0).sum().item()
            coverage_00 = covered_points_00 / len(control_index) * 100
            print(
                f"Number of points in control_index covered by idx_sub_training_0 within distance {args.radius}: {round(coverage_00, 2)}")

            # Index masks for sub_training and control points
            mask_treated_index = torch.tensor(
                [i in treated_index for i in range(save_tensor_normalized_distances.shape[0])])

            # Extract the sub-matrix of distances from sub_training_1 to all control indices
            distances_sub_training_to_treated = save_tensor_normalized_distances[mask_sub_training_0][:,
                                                mask_treated_index]

            # Check for coverage within distance 0.15
            coverage_mask = distances_sub_training_to_treated <= args.radius

            # Sum over each column to see if any point in sub_training_1 covers this specific control index
            covered_points_01 = coverage_mask.any(dim=0).sum().item()
            coverage_01 = covered_points_01 / len(treated_index) * 100

            print(
                f"Number of points in treated_index covered by idx_sub_training_0 within distance {args.radius}: {round(coverage_01, 2)}")

            coverage_list_00.append(coverage_00)
            coverage_list_01.append(coverage_01)

    # Specify the file pathstep_1
    file_path = 'text_results/coverage_visuals_{}/coverage_{}_{}_r{}.csv'.format(args.alpha,
                                                                                                args.alpha,
                                                                                                args.seed,
                                                                                                args.radius)

    # Open the CSV file in write mode
    with open(file_path, 'w', newline='') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)

        # Write the data for list_1
        csv_writer.writerow(['Number of Samples'] + list(map(str, num_of_acquire)))

        csv_writer.writerow(['Coverage 11'] + list(map(str, coverage_list_11)))

        csv_writer.writerow(['Coverage 10'] + list(map(str, coverage_list_10)))

        csv_writer.writerow(['Coverage 00'] + list(map(str, coverage_list_00)))

        csv_writer.writerow(['Coverage 01'] + list(map(str, coverage_list_01)))

    print(f'The data has been successfully written to {file_path}')

