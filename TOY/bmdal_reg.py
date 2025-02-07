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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--bmdal', type=str, default='lcmd')
parser.add_argument('--selwithtrain', type=bool, default=False)
args = parser.parse_args()

import json
# Read the configuration from the file
with open('experiments/config_{}.json'.format(args.bmdal), 'r') as file:
    config = json.load(file)


from torch.utils.data import Dataset
from causal_bald.library.datasets import HCMNIST

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
    points_per_cluster = 200

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
            }, 'model_selection/{}_nnmodel_1.pth'.format(args.bmdal))

            torch.save({
                'model_state_dict': model_0.state_dict(),
            }, 'model_selection/{}_nnmodel_0.pth'.format(args.bmdal))

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

        checkpoint_1 = torch.load('model_selection/{}_nnmodel_1.pth'.format(args.bmdal))
        model_1.load_state_dict(checkpoint_1['model_state_dict'])

        checkpoint_0 = torch.load('model_selection/{}_nnmodel_0.pth'.format(args.bmdal))
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


def acquiring(x_train, y_train, x_pool, custom_model, strategy, num_of_samples):

    train_data = TensorFeatureData(x_train)
    pool_data = TensorFeatureData(x_pool)

    new_idxs, _ = select_batch(batch_size=num_of_samples, models=[custom_model],
                               data={'train': train_data, 'pool': pool_data}, y_train=y_train,
                               selection_method=strategy, sel_with_train=args.selwithtrain,
                               base_kernel='grad', kernel_transforms=[('rp', [512])])

    return new_idxs


class DeepKernelGP_(gpytorch.Module):
    def __init__(self, encoder, gp):
        super(DeepKernelGP_, self).__init__()

#        self.encoder = encoder
        self.gp = gp

    def forward(self, inputs):
#        phi = self.encoder(inputs[:, :-1])
#        t = inputs[:, -1:]
#        return self.gp(torch.cat([phi, t], dim=-1))
        return self.gp(inputs)


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_size = 0.10
num_trial = 1
al_step = 50
warm_up = 1
num_of_samples = 1
seed = args.seed

if num_trial == 1:
# for seed in range(num_trial):

    print('Trial:', seed + 1)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    combine_x, combined_y, tau, T = generation(mean_1=0, std_1=1, func_1=torch.sin, mean_0=3, std_0=1, func_0=torch.cos,
                                               device=device)
    combine_x_train, combine_x_test, combined_y_train, combine_x_valid, combined_y_valid, tau_test, T_train, T_valid, T_test = train_test_splitting(
        combine_x, combined_y, tau, T, test_size, seed, device=device)

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
    # num_to_select = int(len(idx_remaining_0) * 0.1)
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

        job_dir_path = Path('saved_models/selection_visuals/method_{}/seed_{}/step_{}'.format(config.get("acquisition_function"), seed, 0))

        train_deep_kernel_gp(ds_train=train_dataset,
                             ds_valid=tune_dataset,
                             job_dir=job_dir_path,
                             config=config,
                             dim_input=train_x_1.shape[1],
                             seed=seed)

        test_dataset = toDataLoader(x_train=combine_x_test.cpu(),
                                    y_train=tau_test.cpu(),
                                    t_train=T_test.cpu()
                                    )

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
        print('acquired index:', acquired_index)
        treated_idx, control_idx = trt_ctr(T_train[acquired_index])
        acquired_treated, acquired_control = acquired_index[treated_idx], acquired_index[control_idx]

        if len(acquired_treated) == 0 and len(acquired_control) == 0:
            raise TypeError("Nothing acquired by AL")
        else:
            print('Acquiring the treated and control:', len(acquired_treated), len(acquired_control))
            
            if query_step != 0:
                num_of_acquire.append(num_of_acquire[query_step - 1] + len(acquired_treated) + len(acquired_control))

        if len(acquired_treated) != 0:
            idx_sub_training_1, idx_remaining = pool_updating(idx_remaining, idx_sub_training_1, acquired_treated)
        if len(acquired_control) != 0:
            idx_sub_training_0, idx_remaining = pool_updating(idx_remaining, idx_sub_training_0, acquired_control)

    average_pehe = np.array(error_list)

# Specify the file path
file_path = 'text_results/{}/pehe_{}_{}.csv'.format(args.bmdal, args.bmdal, args.seed)

# Open the CSV file in write mode
with open(file_path, 'w', newline='') as csvfile:
    # Create a CSV writer object
    csv_writer = csv.writer(csvfile)

    # Write the data for list_1
    csv_writer.writerow(['Number of Samples'] + list(map(str, num_of_acquire)))

    # Write the data for list_2
    csv_writer.writerow(['PEHE'] + list(map(str, average_pehe.tolist())))

print(f'The data has been successfully written to {file_path}')
