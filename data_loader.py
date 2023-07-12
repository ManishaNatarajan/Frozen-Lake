import numpy as np
import glob
import random
import pandas as pd
from data_loaders.maps import MAPS, FOG
SEED = 0


def train_test_split(filenames, test_split=0.2):
    num_files = len(filenames)
    assert num_files != 0
    # Shuffle the file indexes
    file_idxs = np.random.permutation(np.arange(num_files))
    train, test = [], []

    for i, idx in enumerate(file_idxs):
        if i < test_split * num_files:
            test.append(filenames[idx])
        else:
            train.append(filenames[idx])
    return train, test


def convert_robot_actions(robot_type, actions):
    idx = 0
    if robot_type == "interrupt":
        idx = 1
    elif robot_type == "control":
        idx = 2
    elif robot_type == "interrupt_w_explain":
        idx = 3
    elif robot_type == "control_w_explain":
        idx = 4

    actions[actions > 0] = idx
    return actions


def convert_human_actions(actions):
    actions[actions >= 4] = 4  # Detection
    return actions


def extract_data(filename):
    df = pd.read_csv(filename)

    # For now: Extract only map 5 data
    map_5_data = df.loc[df['map'] == 5]
    states = map_5_data["last_state"].to_numpy()
    next_states = map_5_data["current_state"].to_numpy()
    robot_type = map_5_data["robot_type"].to_numpy()[0]
    robot_actions = map_5_data["robot_action"].to_numpy()
    robot_actions = convert_robot_actions(robot_type, robot_actions)
    human_actions = map_5_data["last_human_action"].to_numpy()
    human_actions = convert_human_actions(human_actions)
    rewards = map_5_data["reward"].to_numpy()
    timesteps = map_5_data["timestep"].to_numpy()

    episode = {
        "state": states,
        "action": robot_actions,
        "obs": human_actions,
        "next_state": next_states,
        "reward": rewards,
        "timestep": timesteps,
        "map_id": 5
    }
    return episode


def encode_full_state(map_id, states):
    num_steps = states.shape[0]
    all_state_info = []
    map_info = np.array([list(x) for x in MAPS["MAP{}".format(map_id)]])
    human_map_info = map_info  # TODO: Account for human error in identifying slippery regions
    fog_info = np.array([list(x) for x in FOG["MAP{}".format(map_id)]])
    map_state = np.zeros((8, 8, 2))
    # Map state idx:
    # 0: Agent Pos
    # 1: Empty(0) / Slip(1) / Hole(2) / Goal (3)
    # 2: Fog

    # Set fog
    row, col = np.where(fog_info == 'F')
    map_state[row, col, 1] = 1

    # Set slip, holes and goal
    row, col = np.where(human_map_info == 'S')
    map_state[row, col, 0] = 1  # Slippery region  # TODO: Incorporate detections...

    row, col = np.where(human_map_info == 'H')
    map_state[row, col, 0] = 2  # Hole

    map_state[7, 7, 0] = 3  # Goal

    for n in range(num_steps):
        agent_pos = states[n]
        row = agent_pos // 8
        col = agent_pos % 8
        temp_state = np.zeros((8, 8, 1))
        temp_state[row, col, 0] = 1
        full_state = np.concatenate([temp_state, map_state], axis=2)
        all_state_info.append(full_state)
    return all_state_info


def stack_observations(states, nstack=4):
    raise NotImplementedError


if __name__ == '__main__':
    random.seed(SEED)
    np.random.seed(SEED)
    DIR_PATH = "data/User Study 1/RL_data/"
    all_files = glob.glob(DIR_PATH + "*.csv")
    train_files, test_files = train_test_split(all_files, test_split=0.2)

    temp = train_files[0]
    eps_data = extract_data(temp)
    map_state = encode_full_state(map_id=5, states=eps_data["state"])  # Ordered according to timesteps in the rollout


