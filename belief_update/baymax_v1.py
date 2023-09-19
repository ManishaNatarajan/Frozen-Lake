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
import pygad


def create_init_belief(num_particles=50):
    particle_set = []
    beta_params = np.array([8, 3])

    for n in range(num_particles):
        particle_set.append(beta_params)

    return np.array(particle_set)  # Particle set should be a numpy array of shape (num_particles, #params)


def get_user_action_from_beta(beta_params, curr_robot_action=0):
    acceptance_prob = np.random.beta(beta_params[0], beta_params[1])
    e = np.random.uniform()
    if e < acceptance_prob:
        human_action = 0  # Accept
        accept = True
    else:
        if curr_robot_action == 0:
            human_action = 2  # Detect
        else:
            # For take control or interrupt
            if np.random.uniform() < 0.9:
                human_action = 1  # Oppose
            else:
                human_action = 2  # Detect

    return human_action


def prediction_counts_after_belief_update(particle_set, curr_robot_action):
    prediction_counts = np.array([0, 0, 0])
    num_particles = particle_set.shape[0]

    predicted_human_actions = np.zeros((num_particles, 1))

    # Random draw...
    random_samples = np.random.uniform(0, 1, num_particles).reshape(-1, 1)
    acceptance_probs = np.random.beta(particle_set[:, 0], particle_set[:, 1]).reshape(-1, 1)
    if curr_robot_action == 0:
        predicted_human_actions[random_samples > acceptance_probs] = 2

    else:
        # Random draw for choosing between oppose and detect
        temp = np.zeros((num_particles, 1))
        temp[random_samples > acceptance_probs] = 1
        random_samples = random_samples * temp

        # Threshold for oppose vs detect
        threshold = 0.9
        predicted_human_actions[np.logical_and(random_samples > 0.01, random_samples < threshold)] = 1
        predicted_human_actions[random_samples > threshold] = 2

    idxs, count_vals = np.unique(predicted_human_actions, return_counts=True)
    prediction_counts[idxs.astype(int)] = count_vals
    return prediction_counts


def distance_from_true_action(user_action, predicted_action):
    return 1 if user_action == predicted_action else 0


def get_entropy(beta_params):
    return beta.entropy(beta_params[0], beta_params[1])


def belief_update(particle_set, curr_robot_action, curr_human_action, get_counts=False):
    # ------------------------------------ Define GA parameters ----------------------------------------------- #
    def fitness_func(ga_instance, solution, solution_idx):
        # Fitness function is a combination of the predicted accuracy, and entropy of the distribution
        predicted_action = get_user_action_from_beta(solution, curr_robot_action)
        prediction_accuracy = distance_from_true_action(curr_human_action, predicted_action)

        fitness_val = prediction_accuracy + 0.5 * get_entropy(solution)
        return fitness_val

    fitness_function = fitness_func

    num_generations = 100
    num_parents_mating = 5

    sol_per_pop = 50
    num_genes = 2

    # To initialize population
    init_range_low = 0.1
    init_range_high = 5

    parent_selection_type = "sss"
    keep_parents = 1

    crossover_type = "single_point"

    mutation_type = "random"
    mutation_percent_genes = 50

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_probability=0,
                           mutation_percent_genes=mutation_percent_genes,
                           random_seed=seed)

    # Use the current set of particles as the initial population for the Genetic Algorithm
    ga_instance.initial_population = particle_set

    # Run the genetic algorithm to get the next gen particles
    ga_instance.run()

    new_particle_set = ga_instance.population
    prediction_counts = prediction_counts_after_belief_update(new_particle_set, curr_robot_action)

    # update new particle set with Bayes...
    if curr_human_action == 0:
        new_particle_set[:, 0] += 1
    else:
        new_particle_set[:, 1] += 1

    if get_counts:
        return new_particle_set, prediction_counts

    return new_particle_set


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

        belief = create_init_belief(num_particles=5)

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
                                           curr_human_action=current_human_action, get_counts=True)

            model_prediction_probs = counts/np.sum(counts)
            true_val = np.eye(num_human_actions)[current_human_action]
            predicted_val = np.random.choice(np.arange(num_human_actions), p=model_prediction_probs)
            # belief = reinvigorate_particles(belief, min_particles=400)
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


