import pandas as pd
import numpy as np
from data_loaders.maps import MAPS, FOG, HUMAN_ERR


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


def check_neighboring_states(row, col, human_map, slippery_states, detected_error):
    """
    Detect if neighboring states are slippery regions based on map
    :param row:
    :param col:
    :param map: 2d array of map states
    :param detected_error: list of (row, col) that have been already detected
    :return: list of slippery states in neighboring states
    """
    # Left: (0, -1); down: (1, 0); right: (0, 1); up: (-1, 0)
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    n_rows, n_cols = human_map.shape

    for d in directions:
        next_row = row + d[0]
        next_col = col + d[1]
        if 0 <= next_row < n_rows and 0 <= next_col < n_cols:
            if human_map[next_row, next_col] == 'S' and (next_row, next_col) not in detected_error:  # Not False Positive
                slippery_states.add((next_row, next_col))
            elif human_map[next_row, next_col] == 'F' and (next_row, next_col) in detected_error:
                # False Negative
                slippery_states.add((next_row, next_col))


def encode_full_state(map_id, states, human_actions=None):
    num_steps = states.shape[0]
    all_state_info = []
    map_info = np.array([list(x) for x in MAPS["MAP{}".format(map_id)]])  # Ground truth

    human_map_info = np.array([list(x) for x in MAPS["MAP{}".format(map_id)]])
    human_err_info = HUMAN_ERR["MAP{}".format(map_id)]
    # Switch the slippery regions based on error
    for err in human_err_info:
        if human_map_info[err[0], err[1]] == 'S':
            human_map_info[err[0], err[1]] = 'F'
        else:
            human_map_info[err[0], err[1]] = 'S'

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
    # row, col = np.where(human_map_info == 'S')
    # map_state[0, row, col] = 1  # Slippery region  # TODO: Incorporate detections...
    # Slippery regions are included as the robot traverses the map (not at the start)

    row, col = np.where(human_map_info == 'H')
    map_state[0, row, col] = 2  # Hole

    map_state[0, 7, 7] = 3  # Goal

    detected_error = set()
    slippery_states = set()

    for n in range(num_steps):
        agent_pos = states[n]
        row = agent_pos // 8
        col = agent_pos % 8
        temp_state = np.zeros((1, 8, 8))
        temp_state[0, row, col] = 1

        if human_actions is not None:
            # Check if human detected
            if human_actions[n] >= 4:
                if human_actions[n] == 4:  # detect left
                    check_row = row
                    check_col = col-1

                elif human_actions[n] == 5:  # detect down
                    check_row = row+1
                    check_col = col

                elif human_actions[n] == 6:  # detect right
                    check_row = row
                    check_col = col+1

                else:  # detect up
                    check_row = row-1
                    check_col = col

                if 0 <= check_row < 8 and 0 <= check_col < 8:
                    if map_info[check_row, check_col] != human_map_info[check_row, check_col]:
                        detected_error.add((check_row, check_col))

        # Check for slippery regions...
        check_neighboring_states(row, col, human_map_info, slippery_states, detected_error)
        if len(slippery_states) == 0:
            full_state = np.concatenate([temp_state, map_state], axis=0)
        else:
            # Remove slippery states that are previously detected...
            remove = []
            for s in slippery_states:
                if s in detected_error and map_info[s[0], s[1]] == 'F':
                    remove.append(s)

            for r in remove:
                slippery_states.remove(r)

            # update map_state
            # Check again if slippery state is empty after removing...
            if len(slippery_states) == 0:
                full_state = np.concatenate([temp_state, map_state], axis=0)
            else:
                s_row, s_col = np.array(list(slippery_states))[:, 0], np.array(list(slippery_states))[:, 1]
                new_slip_state = np.zeros((2, 8, 8))
                new_slip_state[0, s_row, s_col] = 1
                new_slip_state = new_slip_state + map_state
                full_state = np.concatenate([temp_state, new_slip_state], axis=0)

                # Remove slippery state after visiting (if safe)...
                if (row, col) in slippery_states and map_info[row, col] == 'F':
                    slippery_states.remove((row, col))
                    detected_error.add((row, col))
        all_state_info.append(full_state)
    return all_state_info


def extract_episode_data(filename, map_ids=(4, 5, 6, 7, 8, 9, 10, 11, 12, 13)):
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
