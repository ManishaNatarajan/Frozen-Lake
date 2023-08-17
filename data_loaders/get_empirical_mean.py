import os
import numpy as np
import torch
import random
from BC.model import BCModel
from torch.utils.data import DataLoader
from data_loaders.BC_loader import BCDataset
from tqdm import tqdm
import copy
from scipy.stats import beta

if __name__=='__main__':
    # Set seeds
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    num_robot_actions = 5
    num_human_actions = 3

    # Load configs
    # TODO: Load from file (store config files for all trained models)
    batch_size = 1
    seq_len = 16
    use_actions = True

    # Load train dataset
    train_path = "data/User_Study_1/RL_data/train/"
    train_dataset = BCDataset(folder_path=train_path, sequence_length=seq_len, use_actions=use_actions)

    _, all_robot_actions = np.where(train_dataset.robot_actions == 1)
    _, all_human_actions = np.where(train_dataset.human_actions == 1)

    # Compute how many times the users chose to proceed, oppose, detect per robot intervention
    train_data_vals = np.zeros((num_robot_actions, num_human_actions))

    for r in range(num_robot_actions):
        human_action_distribution = all_human_actions[all_robot_actions == r]
        vals, counts = np.unique(human_action_distribution, return_counts=True)
        for c, v in enumerate(vals):
            train_data_vals[r, v] = counts[c]


    print(train_data_vals)

    print("Normalized counts:")
    print(np.divide(train_data_vals, np.sum(train_data_vals, axis=1, keepdims=True)))
