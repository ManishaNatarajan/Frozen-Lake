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


def create_init_belief(num_particles=500, init_vals=[12, 2, 4]):
    particle_set = []
    dirichlet_counts = np.array(init_vals)  # TODO: To be tuned...

    for n in range(num_particles):
        particle_set.append(dirichlet_counts)

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
        dirichilet_params = copy.deepcopy(particle_set[i])
        num_user_actions = len(dirichilet_params)

        # Sample from the dirichlet distribution
        user_action_probs = []
        for a in range(num_user_actions):
            prob = np.random.beta(dirichilet_params[a], np.sum(dirichilet_params) - dirichilet_params[a])
            user_action_probs.append(prob)

        # Reweight the user_action_probs as the prob sampled from the beta distributions may not add to 1
        user_action_probs = np.exp(user_action_probs) / np.sum(np.exp(user_action_probs))

        estimated_human_action = np.random.choice(np.arange(num_user_actions), p=user_action_probs)

        while curr_robot_action == 0 and estimated_human_action == 1:
            # Human cannot oppose the robot when it does not interrupt, so resample
            estimated_human_action = np.random.choice(np.arange(num_user_actions), p=user_action_probs)

        dirichilet_params[estimated_human_action] += 1

        # Get counts
        prediction_counts[estimated_human_action] += 1
        if estimated_human_action == curr_human_action:
            new_particle_set.append(dirichilet_params)

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
        while len(particle_set) < min_particles:
            p = np.random.choice(np.arange(len(particle_set)))
            new_particle = copy.deepcopy(particle_set[p])
            particle_set.append(new_particle)

        return particle_set


def max_entropy_reinvigorate_particles(particle_set, min_particles=100):
    # TODO: For dirichlet counts...
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
        wts = np.exp(wts) / np.sum(np.exp(wts))
        # Resample particles based on entropy weights
        resample_idxs = np.random.choice(np.arange(len(particle_set)), min_particles - len(particle_set), p=wts)
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
    test_path = "data/User_Study_1/RL_data/val/"
    test_dataset = BCDataset(folder_path=test_path, sequence_length=seq_len, use_actions=use_actions)

    _, all_robot_actions = np.where(test_dataset.robot_actions == 1)

    prior = [14, 3, 6]
    belief = create_init_belief(num_particles=500, init_vals=prior)

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

        model_prediction_probs = counts / np.sum(counts)
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
    print("Sensitivity: {}".format(sensitivity / true_counts))
    print("Avg Sensitivity: {}".format(np.mean(sensitivity / true_counts)))

    # Check if a file with params already exists...
    file_path = "belief_update/logs/dirichlet/best_params.npz"
    if os.path.isfile(file_path):
        with np.load(file_path) as data:
            prev_avg_sensitivity = data["avg_sensitivity"]

        curr_sensitivity = np.mean(sensitivity / true_counts)
        print(f"Prev sensitivity: {prev_avg_sensitivity}")
        print(f"Current sensitivity: {curr_sensitivity}")
        if curr_sensitivity > prev_avg_sensitivity:
            print("Saving new best params...")
            # Store new params...
            np.savez(file_path,
                     prior=prior,
                     accuracy=accuracy / len(test_dataset),
                     counts=pred_counts,
                     true_counts=true_counts,
                     sensitivity=sensitivity / true_counts,
                     avg_sensitivity=np.mean(sensitivity / true_counts))

    else:
        # File does not exist, so save params
        print("creating new best params...")
        np.savez(file_path,
                 prior=prior,
                 accuracy=accuracy / len(test_dataset),
                 counts=pred_counts,
                 true_counts=true_counts,
                 sensitivity=sensitivity / true_counts,
                 avg_sensitivity=np.mean(sensitivity / true_counts))
