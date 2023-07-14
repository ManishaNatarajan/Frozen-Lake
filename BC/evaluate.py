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

    # Load configs
    # TODO: Load from file (store config files for all trained models)
    batch_size = 1
    seq_len = 16
    use_actions = True

    # Load test dataset
    test_path = "data/User_Study_1/RL_data/val/"
    test_dataset = BCDataset(folder_path=test_path, sequence_length=seq_len, use_actions=use_actions)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model path
    model_folder_path = "BC_logs/20230712-1343/"

    # Configure model
    model = BCModel(obs_shape=(8, 8, 3), robot_action_shape=5, human_action_shape=5,
                    conv_hidden=32, action_hidden=32, num_layers=1, use_actions=use_actions)

    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_folder_path, "best.pth")))

    accuracy = 0

    # Go through the dataset sequentially (i.e., one step at a time)
    # TODO: Go through different episodes for validation...
    for i in tqdm(range(len(test_dataset))):
        x_test, y_test = test_dataset[i]
        if use_actions:
            x_test = [torch.from_numpy(x).to(device).float().unsqueeze(0) for x in x_test]
        else:
            x_test = torch.from_numpy(x_test).to(device).float().unsqueeze(0)

        y_test = torch.from_numpy(y_test).to(device).float()
        model_predictions = model.get_predictions(x_test)

        # Compute accuracy...
        accuracy += torch.sum(torch.argmax(model_predictions, axis=-1) == torch.argmax(y_test, axis=-1))

    print("accuracy: {}".format(accuracy/len(test_dataset)))








