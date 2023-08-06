import os
import numpy as np
import torch
import random
from BC.model import BCModel
from torch.utils.data import DataLoader
from data_loaders.BC_loader import BCDataset
from tqdm import tqdm

if __name__ == '__main__':
    # Set seeds
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    num_human_actions = 5

    # Load configs
    # TODO: Load from file (store config files for all trained models)
    batch_size = 1
    seq_len = 16
    use_actions = True

    # Load test dataset
    test_path = "data/User_Study_1/RL_data/val/"
    test_dataset = BCDataset(folder_path=test_path, sequence_length=seq_len, use_actions=use_actions,
                             num_human_actions=num_human_actions)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model path
    model_folder_path = "BC_logs/20230718-1054/"

    # Configure model
    model = BCModel(obs_shape=(8, 8, 3), robot_action_shape=5, human_action_shape=num_human_actions,
                    conv_hidden=32, action_hidden=32, num_layers=1, use_actions=use_actions)

    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_folder_path, "best.pth")))

    accuracy = 0
    counts = torch.zeros((num_human_actions,)).to(device)
    true_counts = torch.zeros((num_human_actions,)).to(device)

    # Sensitivity analysis: How accurate is the model in predicting each human action type
    true_preds = torch.zeros((num_human_actions,)).to(device)
    false_preds = torch.zeros((num_human_actions,)).to(device)

    # Use the same model to predict: proceed, oppose, detect [Store the true pred, and false pred counts]
    proceed_counts = torch.zeros((2,)).to(device)
    oppose_counts = torch.zeros((2,)).to(device)
    detect_counts = torch.zeros((2,)).to(device)

    # Extract info about last state, curr state and last action to determine whether the user decided to
    #  proceed, oppose or detect
    last_states = test_dataset.last_states
    curr_states = test_dataset.current_states
    last_human_actions = test_dataset.last_actions
    _, all_robot_actions = np.where(test_dataset.robot_actions == 1)  # Get idx from one-hot encoding

    # Go through the dataset sequentially (i.e., one step at a time)
    for i in tqdm(range(len(test_dataset))):
        x_test, y_test = test_dataset[i]
        if use_actions:
            x_test = [torch.from_numpy(x).to(device).float().unsqueeze(0) for x in x_test]
        else:
            x_test = torch.from_numpy(x_test).to(device).float().unsqueeze(0)

        curr_human_action = np.where(y_test == 1)
        y_test = torch.from_numpy(y_test).to(device).float()
        model_predictions = model.get_predictions(x_test)

        # Compute accuracy...
        counts[torch.argmax(model_predictions, axis=-1)] += 1
        true_counts[torch.argmax(y_test, axis=-1)] += 1
        accuracy += torch.sum(torch.argmax(model_predictions, axis=-1) == torch.argmax(y_test, axis=-1))

        if torch.argmax(model_predictions, axis=-1) == torch.argmax(y_test, axis=-1):
            true_preds[torch.argmax(model_predictions, axis=-1)] += 1
            accurate = True
        else:
            false_preds[torch.argmax(y_test, axis=-1)] += 1
            accurate = False

        # Use the same model to predict: proceed, oppose, detect
        if all_robot_actions[i] == 0:
            # Human can only procced or detect
            if curr_human_action[0] == 4:  # detect
                detect_counts[1 - accurate] += 1
            else:
                proceed_counts[
                    1 - accurate] += 1  # TODO: increment proceed as long as they didn't oppose (it doesn't have to be the exact direction)

        elif all_robot_actions[i] == 1 or all_robot_actions[i] == 3:
            # Human can proceed, oppose or detect
            if curr_human_action[0] == 4:  # detect
                detect_counts[1 - accurate] += 1
            elif last_human_actions[i] == curr_human_action:  # oppose
                oppose_counts[1 - accurate] += 1
            else:
                proceed_counts[1 - accurate] += 1

        else:
            # Human can proceed, oppose or detect
            if curr_human_action[0] == 4:  # detect
                detect_counts[1 - accurate] += 1
            else:
                DIRECTION = {8: 3, -8: 1, 1: 0, -1: 2}
                # Choose opposite direction to check whether user opposed the robot
                if curr_states[i] != last_states[i]:
                    robot_direction = DIRECTION[curr_states[i] - last_states[i]]
                    if robot_direction == curr_human_action:
                        oppose_counts[1 - accurate] += 1
                else:
                    proceed_counts[1 - accurate] += 1

    print("accuracy: {}".format(accuracy / len(test_dataset)))
    print("Counts: {}".format(counts))
    print("True Counts: {}".format(true_counts))
    print("Preds: {}, {}".format(true_preds, false_preds))
    print("Sensitivity: {}".format(true_preds / (true_preds + false_preds)))
    print("Avg Sensitivity: {}".format(torch.mean(true_preds / (true_preds + false_preds))))
    print("----------------------------------------------------------------------------------")
    print("Proceed counts: ", proceed_counts)
    print("Oppose counts: ", oppose_counts)
    print("detect counts: ", detect_counts)
    print("Sensitivity: {}".format([proceed_counts[0] / torch.sum(proceed_counts),
                                    oppose_counts[0] / torch.sum(oppose_counts),
                                    detect_counts[0] / torch.sum(detect_counts)]))
