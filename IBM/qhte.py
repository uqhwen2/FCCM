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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

import json
# Read the configuration from the file
with open('experiments/config_qhte.json', 'r') as file:
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
    
    print('Trial:', seed)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    combine_x_train,\
    combine_x_test,\
    combined_y_train,\
    combine_x_valid,\
    combined_y_valid,\
    tau_test, T_train,\
    T_valid,\
    T_test,\
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

    # Calculate the number of elements to select (20% of idx_remaining_0)
    num_to_select = int(len(idx_remaining_0) * 1)
    # Randomly select 50% of elements from idx_remaining_0 without replacement
    selected_elements = np.random.choice(idx_remaining_0, num_to_select, replace=False)
    # Extend idx_sub_training_0 with the selected elements
    idx_sub_training_0 = np.concatenate((idx_sub_training_0, selected_elements))
    # Update idx_remaining_0 to remove the selected elements
    idx_remaining_0 = np.setdiff1d(idx_remaining_0, selected_elements, assume_unique=True)

    acquired_treated, acquired_control = None, None
    error_list = []
    num_of_acquire = [len(idx_sub_training_1) + len(idx_sub_training_0)]
    
    for query_step in range(al_step):
        
        train_x_1, train_y_1 = combine_x_train[idx_sub_training_1], combined_y_train[idx_sub_training_1]
        train_x_0, train_y_0 = combine_x_train[idx_sub_training_0], combined_y_train[idx_sub_training_0]
        print("Number of data used for training in treated and control:", len(idx_sub_training_1), len(idx_sub_training_0))
        '''
        # Do t-SNE visualization for the updataed dataset
        # Concatenate the data matrices

        X = np.vstack([train_x_1.cpu(), train_x_0.cpu()])

        # Create labels for the data points
        labels_ = np.ones(X.shape[0])
        labels_[train_x_1.shape[0]:] = 0  # Set labels for control x_0 as 0

        print("Ploting the t-SNE")
        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=args.seed)
        print("Initializing the tsne function to call")
        X_embedded_ = tsne.fit_transform(X)
        print("Embedding complete")

        # Save the embedding and labels to a single file
        np.savez('embeddings/qhte/embedding_and_labels_{}.npz'.format(query_step), X_embedded=X_embedded_, labels=labels_)
        print("Embedding and labels saved to embedding_and_labels.npz")

        # To reload the embedding and labels later
        print("Reloading the embedding and labels from file")
        data = np.load('embeddings/qhte/embedding_and_labels_{}.npz'.format(query_step))
        X_embedded = data['X_embedded']
        labels = data['labels']
        print("Embedding and labels reloaded successfully")
        '''
        # Concatenate vertically and shuffle randomly for the sub training
        combine_train_idx = np.concatenate([idx_sub_training_1, idx_sub_training_0], axis=0)
        np.random.shuffle(combine_train_idx)
        train_dataset = toDataLoader(x_train=combine_x_train[combine_train_idx].cpu(),
                                     y_train=combined_y_train[combine_train_idx].cpu(),
                                     t_train=T_train[combine_train_idx].cpu())

        tune_dataset = toDataLoader(x_train=combine_x_valid.cpu(),
                                    y_train=combined_y_valid.cpu(),
                                    t_train=T_valid.cpu())

        job_dir_path = Path('saved_models/method_{}/seed_{}/step_{}'.format(
            config.get("acquisition_function"), seed, 0)
        )

        train_deep_kernel_gp(ds_train=train_dataset,
                             ds_valid=tune_dataset,
                             job_dir=job_dir_path,
                             config=config,
                             dim_input=train_x_1.shape[1],
                             seed=seed)

        test_dataset = toDataLoader(x_train=combine_x_test.cpu(),
                                    y_train=tau_test.cpu(),
                                    t_train=T_test.cpu())

        (mu_0, mu_1), _ = predict_deep_kernel_gp(dataset=test_dataset,
                                                 job_dir=job_dir_path,
                                                 config=config,
                                                 dim_input=train_x_1.shape[1],
                                                 seed=seed
                                                )

        pehe_error = evaluation_nn(pred_1=mu_1.mean(0) * y_std,
                                   pred_0=mu_0.mean(0) * y_std,
                                   test_tau=tau_test.cpu(),
                                   query_step=query_step
                                   )

        error_list.append(np.round(pehe_error.cpu().numpy(), 4))

        acquired_treated, acquired_control = [], []
        while len(acquired_treated) + len(acquired_control) < num_of_samples:

            part_cov_matrix = torch.cdist(combine_x_train[idx_sub_training_1], combine_x_train[idx_remaining_1], p=2).pow(2)

            max_values_remaining_1, _ = torch.min(part_cov_matrix, dim=0)

            part_cov_matrix = torch.cdist(combine_x_train[idx_sub_training_0], combine_x_train[idx_remaining_0], p=2).pow(2)

            max_values_remaining_0, _ = torch.min(part_cov_matrix, dim=0)

            # Find the largest value and its indices
            max_value_1, index_1 = torch.max(max_values_remaining_1, dim=0)
            '''max_value_0 == 0 when all t=0 sample acquired'''
            # max_value_0, index_0 = torch.max(max_values_remaining_0, dim=0) # for non-zero t=0 pool set

            if max_value_1.item() >= 0: # max_value_0.item(): since the other side is all known
                acquired_treated.append(idx_remaining_1[index_1.item()])
                idx_sub_training_1, idx_remaining_1 = pool_updating(idx_remaining_1, idx_sub_training_1,
                                                                    [idx_remaining_1[index_1.item()]])
            else:
                acquired_control.append(idx_remaining_0[index_0.item()])
                idx_sub_training_0, idx_remaining_0 = pool_updating(idx_remaining_0, idx_sub_training_0,
                                                                    [idx_remaining_0[index_0.item()]])


        if len(acquired_treated) == 0 and len(acquired_control) == 0:
            raise  TypeError("Nothing acquired by AL")
        else:
            print('Acquiring the treated and control:', len(acquired_treated), len(acquired_control))
            if query_step != 0:
                num_of_acquire.append(num_of_acquire[query_step - 1] + len(acquired_treated) + len(acquired_control))
        
#        if len(acquired_treated) != 0:
#            idx_sub_training_1, idx_remaining_1 = pool_updating(idx_remaining_1, idx_sub_training_1, acquired_treated)
#            
#        idx_sub_training_0, idx_remaining_0 = pool_updating(idx_remaining_0, idx_sub_training_0, acquired_control)

    average_pehe = np.array(error_list)

# Specify the file path
file_path = 'text_results/qhte/pehe_trueqhte_{}.csv'.format(args.seed)

# Open the CSV file in write mode
with open(file_path, 'w', newline='') as csvfile:
    # Create a CSV writer object
    csv_writer = csv.writer(csvfile)

    # Write the data for list_1
    csv_writer.writerow(['Number of Samples'] + list(map(str, num_of_acquire)))

    # Write the data for list_2
    csv_writer.writerow(['PEHE'] + list(map(str, average_pehe.tolist())))

print(f'The data has been successfully written to {file_path}')
