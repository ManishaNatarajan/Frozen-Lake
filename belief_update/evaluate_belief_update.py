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
from sklearn.model_selection import KFold


def create_init_belief(num_particles=500):
    particle_set = []
    beta_params = np.array([8, 3])

    for n in range(num_particles):
        particle_set.append(beta_params)

    return particle_set


def belief_update(particle_set, curr_robot_action, curr_human_action, get_counts=True):
    """
    Updates the current set of particles based on current robot and human actions
    :param map_state: Robot position on the map before the user took action
    :param particle_set: list of particles
    :param curr_robot_action: robot action at timestep t
    :param curr_human_action: human action (observation) to update the beta params
    :return:
    """
    new_particle_set = []
    prediction_counts = np.array([0, 0, 0])

    # sample particles without replacement
    num_particles = len(particle_set)
    sample_idx = np.arange(num_particles)
    np.random.shuffle(sample_idx)
    num_trust_states = 2

    for i in sample_idx:
        beta_params = copy.deepcopy(particle_set[i])
        accept = False

        # Sample from the beta distribution
        acceptance_prob = np.random.beta(beta_params[0], beta_params[1])

        e = np.random.uniform()

        if e < acceptance_prob:
            estimated_human_action = 0
            accept = True

        else:
            if curr_robot_action == 0:
                estimated_human_action = 2
            else:
                # For take control or interrupt
                if np.random.uniform() < 0.9:
                    estimated_human_action = 1
                else:
                    estimated_human_action = 2

        # Update beta params
        if accept:
            beta_params[0] += 1
        else:
            beta_params[1] += 1

        # Get counts
        prediction_counts[estimated_human_action] += 1
        if estimated_human_action == curr_human_action:
            new_particle_set.append(beta_params)

    if get_counts:
        return new_particle_set, prediction_counts
    return new_particle_set  # By looking at the len of new_particle_set and the old_particle_set, we can estimate accuracy


def reinvigorate_particles(particle_set, min_particles=100):
    # Random Reinvigoration
    if len(particle_set) > min_particles:
        return particle_set
    else:
        # Create duplicate particles to reach min particles
        if len(particle_set) == 0:
            print("Error: Belief is empty!!!")
            return create_init_belief(min_particles)
        while len(particle_set) < min_particles:
            p = np.random.choice(np.arange(len(particle_set)))
            new_particle = copy.deepcopy(particle_set[p])
            particle_set.append(new_particle)

        return particle_set


def max_entropy_reinvigorate_particles(particle_set, min_particles=100):
    if len(particle_set) > min_particles:
        return particle_set
    else:
        if len(particle_set) == 0:
            print("Error: Belief is empty!!!")
            return []

        # Compute weights of particles based on entropy of their distributions
        wts = []
        for p in particle_set:
            wts.append(beta.entropy(p[0], p[1]))

        # Softmax the weights
        wts = np.exp(wts)/np.sum(np.exp(wts))
        # Resample particles based on entropy weights
        resample_idxs = np.random.choice(np.arange(len(particle_set)), min_particles-len(particle_set), p=wts)
        # I guess sampling only by entropy weights does not mean that it will pick a diverse set of particles,
        # if one particle has the highest entropy (say uniform distribution) then that particle is more likely to be
        # resampled repeatedly...

        for i in resample_idxs:
            new_particle = copy.deepcopy(particle_set[i])
            particle_set.append(new_particle)

        return particle_set


if __name__ == '__main__':
    # Set seeds
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    num_human_actions = 3

    # Load configs
    # TODO: Load from file (store config files for all trained models)
    batch_size = 1
    seq_len = 16
    use_actions = True

    # Load test dataset
    train_val_path = "data/User_Study_1/RL_data/train_val/"

    # Use k-fold cross validation
    kf = KFold(n_splits=5)

    # get list of all files:
    import glob

    all_files = glob.glob(train_val_path + "*.csv")

    for fold, (train_idx, test_idx) in enumerate(kf.split(all_files)):
        print('Fold: {}, Test Idx: {}'.format(fold, test_idx))
        test_files = [all_files[t] for t in test_idx]

        test_dataset = BCDataset(folder_path=test_files, sequence_length=seq_len, use_actions=use_actions,
                                 num_human_actions=num_human_actions)

        _, all_robot_actions = np.where(test_dataset.robot_actions == 1)

        belief = create_init_belief(num_particles=500)

        CE_loss = []
        accuracy = 0
        pred_counts = np.zeros((3,))
        true_counts = np.zeros((3,))

        # Sensitivity analysis: How accurate is the model in predicting each human action type
        sensitivity = np.zeros((3,))

        # Go through the dataset sequentially (i.e., one step at a time)
        # TODO: Need to reset particles for each user
        for i in tqdm(range(len(test_dataset))):
            _, current_human_action = test_dataset[i]
            current_human_action = np.where(current_human_action == 1)[0][0]
            current_robot_action = all_robot_actions[i]

            belief, counts = belief_update(belief, curr_robot_action=current_robot_action,
                                           curr_human_action=current_human_action)

            model_prediction_probs = counts/np.sum(counts)
            true_val = np.eye(num_human_actions)[current_human_action]
            predicted_val = np.random.choice(np.arange(num_human_actions), p=model_prediction_probs)
            belief = reinvigorate_particles(belief, min_particles=400)
            # belief = max_entropy_reinvigorate_particles(belief, min_particles=500)

            # Get stats
            accuracy += current_human_action == predicted_val
            if current_human_action == predicted_val:
                sensitivity[current_human_action] += 1
            # print(predicted_val, current_human_action)
            CE_loss.append(-np.sum(true_val * model_prediction_probs))
            pred_counts[predicted_val] += 1
            true_counts[current_human_action] += 1

        print("accuracy: {}".format(accuracy / len(test_dataset)))
        print("Counts: {}".format(pred_counts))
        print("True Counts: {}".format(true_counts))
        print("Sensitivity: {}".format(sensitivity/true_counts))
        print("Avg Sensitivity: {}".format(np.mean(sensitivity/true_counts)))

        print('-------------------------------------------------------------------------------------------------------')


