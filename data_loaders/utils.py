import pandas as pd
import numpy as np
from data_loaders.maps import MAPS, FOG


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


def encode_full_state(map_id, states):
    num_steps = states.shape[0]
    all_state_info = []
    map_info = np.array([list(x) for x in MAPS["MAP{}".format(map_id)]])
    human_map_info = map_info  # TODO: Account for human error in identifying slippery regions
    fog_info = np.array([list(x) for x in FOG["MAP{}".format(map_id)]])
    map_state = np.zeros((2, 8, 8))
    # Map state idx:
    # 0: Agent Pos
    # 1: Empty(0) / Slip(1) / Hole(2) / Goal (3)
    # 2: Fog

    # Set fog
    row, col = np.where(fog_info == 'F')
    map_state[1, row, col] = 1

    # Set slip, holes and goal
    row, col = np.where(human_map_info == 'S')
    map_state[0, row, col] = 1  # Slippery region  # TODO: Incorporate detections...

    row, col = np.where(human_map_info == 'H')
    map_state[0, row, col] = 2  # Hole

    map_state[0, 7, 7] = 3  # Goal

    for n in range(num_steps):
        agent_pos = states[n]
        row = agent_pos // 8
        col = agent_pos % 8
        temp_state = np.zeros((1, 8, 8))
        temp_state[0, row, col] = 1
        full_state = np.concatenate([temp_state, map_state], axis=0)
        all_state_info.append(full_state)
    return all_state_info


def extract_episode_data(filename, map_ids=(5, 6, 7, 8)):
    df = pd.read_csv(filename)
    # For now: Extract only select maps --> map_ids
    episode_data = []
    for id in map_ids:
        map_data = df.loc[df['map'] == id]
        states = map_data["last_state"].to_numpy()
        next_states = map_data["current_state"].to_numpy()
        robot_type = map_data["robot_type"].to_numpy()[0]
        robot_actions = map_data["robot_action"].to_numpy()
        robot_actions = convert_robot_actions(robot_type, robot_actions)
        human_actions = map_data["last_human_action"].to_numpy()
        human_actions = convert_human_actions(human_actions)
        rewards = map_data["reward"].to_numpy()
        timesteps = map_data["timestep"].to_numpy()
        dones = np.zeros((timesteps.shape[0], 1))
        dones[-1] = 1

        episode = {
            "state": states,
            "action": robot_actions,
            "obs": human_actions,
            "next_state": next_states,
            "reward": rewards,
            "timestep": timesteps,
            "dones": dones,
            "map_id": id
        }

        episode_data.append(episode)
    return episode_data
