"""
Creates a simulator to mimic the human-robot interaction in Frozen Lake.
The robot can have five different modalities - no interruption, interruption, taking control,
interruption with explanation, taking control with explanation.
We are modeling the environment from the robot's perspective.

(Augmented) State space:
  - World state (observable): slippery, optional direction, ice hole, goal, fog X 8 grids (2 X 5 X 8)
  - Hidden states (non-observable): human trust, human capability, human preference

Action space:
    - Robot Action: No-assist, Interruption, Taking control, Interruption with explanation, Taking control with explanation

Observation space:
    - Human action: Move X 4 directions, detect X 4 directions

Reward:
    - TODO

"""
from typing import List, Optional
import random
import numpy as np
import torch
import os
from gym import spaces, utils
from frozenlake_map import MAPS, FOG, HUMAN_ERR, ROBOT_ERR

from BC.model import BCModel

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

CONDITION = {
    'no_interrupt': 0,
    'interrupt': 1,
    'control': 2,
    # 'interrupt_w_explain': 3,
    # 'control_w_explain': 4
}


# DFS to check that it's a valid path.
def is_valid(board: List[List[str]], max_size: int) -> bool:
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new][c_new] == b"G":
                    return True
                if board[r_new][c_new] != b"H":
                    frontier.append((r_new, c_new))
    return False


def find_shortest_path(board, slippery_region, start, max_size):
    path_list = [[[(start // max_size, start % max_size), None]]]
    path_index = 0
    # To keep track of previously visited nodes
    previous_nodes = {(start // max_size, start % max_size)}
    if start == max_size * max_size - 1:
        return path_list[0]

    while path_index < len(path_list):
        current_path = path_list[path_index]
        last_node, _ = current_path[-1]
        r, c = last_node
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        for i in range(4):
            x, y = directions[i]
            r_new = r + x
            c_new = c + y
            if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                continue
            if board[r_new][c_new] == b"G":
                next_node = (r_new, c_new)
                current_path.append([next_node, i])
                return current_path
            # if board[r_new][c_new] != b"H" and board[r_new][c_new] != b"S":
            if (r_new, c_new) not in slippery_region:
                next_node = (r_new, c_new)
                # next_nodes.append((r_new, c_new))
                # # Add new paths
                # for next_node in next_nodes:
                if not next_node in previous_nodes:
                    new_path = current_path[:]
                    new_path.append([next_node, i])
                    path_list.append(new_path)
                    # To avoid backtracking
                    previous_nodes.add(next_node)
        # Continue to next path in list
        path_index += 1
    # No path is found
    return []


# Set up the drawing window
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 900


class FrozenLakeEnv:
    """
    Frozen lake involves crossing a frozen lake from start to goal without falling into any holes
    by walking over the frozen lake.
    The player may not always move in the intended direction due to the slippery nature of the frozen lake.
    ## Description
    The game starts with the player at location [0,0] of the frozen lake grid world with the
    goal located at far extent of the world e.g. [3,3] for the 4x4 environment.
    Holes in the ice are distributed in set locations when using a pre-determined map
    or in random locations when a random map is generated.
    The player makes moves until they reach the goal or fall in a hole.
    The lake is slippery (unless disabled) so the player may move perpendicular
    to the intended direction sometimes (see <a href="#is_slippy">`is_slippery`</a>).
    Randomly generated worlds will always have a path to the goal.
    Elf and stool from [https://franuka.itch.io/rpg-snow-tileset](https://franuka.itch.io/rpg-snow-tileset).
    All other assets by Mel Tillery [http://www.cyaneus.com/](http://www.cyaneus.com/).
    ## Action Space
    The action shape is `(1,)` in the range `{0, 3}` indicating
    which direction to move the player.
    - 0: Move left
    - 1: Move down
    - 2: Move right
    - 3: Move up
    ## Observation Space
    The observation is a value representing the player's current position as
    current_row * nrows + current_col (where both the row and col start at 0).
    For example, the goal position in the 4x4 map can be calculated as follows: 3 * 4 + 3 = 15.
    The number of possible observations is dependent on the size of the map.
    The observation is returned as an `int()`.
    ## Starting State
    The episode starts with the player in state `[0]` (location [0, 0]).
    ## Rewards
    Reward schedule:
    - Reach goal: +1
    - Reach hole: 0
    - Reach frozen: 0
    ## Episode End
    The episode ends if the following happens:
    - Termination:
        1. The player moves into a hole.
        2. The player reaches the goal at `max(nrow) * max(ncol) - 1` (location `[max(nrow)-1, max(ncol)-1]`).
    - Truncation (when using the time_limit wrapper):
        1. The length of the episode is 100 for 4x4 environment, 200 for 8x8 environment.
    ## Information
    `step()` and `reset()` return a dict with the following keys:
    - p - transition probability for the state.
    See <a href="#is_slippy">`is_slippery`</a> for transition probability information.
    ## Arguments
    ```python
    import gymnasium as gym
    gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    ```
    `desc=None`: Used to specify maps non-preloaded maps.
    Specify a custom map.
    ```
        desc=["SFFF", "FHFH", "FFFH", "HFFG"].
    ```
    A random generated map can be specified by calling the function `generate_random_map`.
    ```
    from gymnasium.envs.toy_text.frozen_lake import generate_random_map
    gym.make('FrozenLake-v1', desc=generate_random_map(size=8))
    ```
    `map_name="4x4"`: ID to use any of the preloaded maps.
    ```
        "4x4":[
            "SFFF",
            "FHFH",
            "FFFH",
            "HFFG"
            ]
        "8x8": [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFHFFFF",
            "FFFFFHFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHF",
            "FFFHFFFG",
        ]
    ```
    If `desc=None` then `map_name` will be used. If both `desc` and `map_name` are
    `None` a random 8x8 map with 80% of locations frozen will be generated.
    <a id="is_slippy"></a>`is_slippery=True`: If true the player will move in intended direction with
    probability of 1/3 else will move in either perpendicular direction with
    equal probability of 1/3 in both directions.
    For example, if action is left and is_slippery is True, then:
    - P(move left)=1/3
    - P(move up)=1/3
    - P(move down)=1/3
    ## Version History
    * v1: Bug fixes to rewards
    * v0: Initial version release
    """

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(
            self,
            render_mode: Optional[str] = None,
            desc=None,
            foggy=None,
            human_err=None,
            robot_err=None,
            true_human_trust=None,
            true_human_capability=None,
            true_robot_capability=None,
            map_name="4x4",
            is_slippery=True,
            beta=None,
            c=15,
            gamma=0.99,
            seed=None,
            human_type="random",
            model_folder_path="BC_logs/20230712-1343/"
    ):
        random.seed(seed)
        np.random.seed(seed)

        self.is_error = False
        assert (desc is not None and map_name is not None)
        if desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype="c")
        self.fog = np.asarray(foggy, dtype="c")

        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.human_err = human_err
        self.robot_err = robot_err
        self.reward_range = (0, 1)

        self.initial_state_distrib = np.array(desc == b"B").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.robot_action = None
        self.last_interrupt = [None, None]

        # Pre-load the map as an 8x8x2 grid (only keep changing the agent positions when passing through the model...)
        self.map_state = np.zeros((2, 8, 8))

        # Position of holes (currently fully accessible to human and robot)
        rows, cols = np.where(self.desc == b'H')
        self.hole = [(r, c) for r, c in zip(rows, cols)]
        self.map_state[0, rows, cols] = 2  # set Hole

        rows, cols = np.where(self.desc == b'S')
        self.slippery = [(r, c) for r, c in zip(rows, cols)]
        self.map_state[0, rows, cols] = 1  # set Slippery regions

        rows, cols = np.where(self.desc == b'F')
        self.fog = [(r, c) for r, c in zip(rows, cols)]
        self.map_state[1, rows, cols] = 1  # set fog

        self.score = 0

        self.interrupted = 0
        self.truncated = False
        self.is_error = False
        self.num_error = 0
        self.num_interrupt = 0
        self.interrupt_state = []

        self.running = True

        self.true_human_trust = true_human_trust  # at the start of the study
        self.true_human_capability = true_robot_capability  # fixed - parameter (assume known??) at the start of the study
        self.world_state = []
        # Action space: 'no_interrupt': 0,
        #     'interrupt': 1,
        #     'control': 2
        self.robot_action_space = spaces.MultiDiscrete([3, 4], seed=seed)
        # Human's action space is whether they accepted the robot's suggestion and the option that they choose
        self.human_action_space = spaces.MultiDiscrete([3, 2, 4], seed=seed)  # (no-assist/accept/reject, detect/no-detect, LEFT/DOWN/RIGHT/UP)
        # Robot's observation space (for now is the human's last action)
        self.robot_observation_space = spaces.MultiDiscrete([2, 4], seed=seed)
        # Human's observation space (Not sure what this should be... Idk if I need it)
        self.human_observation_space = None

        # MCTS parameters
        self.c = c  # Exploration Bonus
        self.gamma = gamma
        self.beta = beta
        self.seed = seed
        self.human_type = human_type

        self.render_mode = render_mode

        # pygame utils
        self.window_size = (min(64 * ncol, 512) + 128 * 2, min(64 * nrow, 512))
        self.cell_size = (
            min(64 * ncol, 512) // self.ncol,
            min(64 * nrow, 512) // self.nrow,
        )

        # Load pre-trained BC Model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = BCModel(obs_shape=(8, 8, 3), robot_action_shape=5, human_action_shape=5,
                             conv_hidden=32, action_hidden=32, num_layers=1, use_actions=True)

        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(model_folder_path, "best.pth")))

    def to_s(self, row, col):
        return row * self.ncol + col

    def inc(self, row, col, a):
        if a == LEFT:
            col = max(col - 1, 0)
        elif a == DOWN:
            row = min(row + 1, self.nrow - 1)
        elif a == RIGHT:
            col = min(col + 1, self.ncol - 1)
        elif a == UP:
            row = max(row - 1, 0)
        return row, col

    # Transition to the next state s' from s after action a
    def move(self, s, a):
        row = s // self.ncol
        col = s % self.ncol
        next_row, next_col = self.inc(row, col, a)
        return self.to_s(next_row, next_col)

    def detect_slippery_region(self, position, human_slippery, robot_slippery, human_err, robot_err, detecting=None):
        curr_row = position // self.ncol
        curr_col = position % self.ncol
        actions = [0, 1, 2, 3]
        next_human_slippery = {i for i in human_slippery}
        next_robot_slippery = {i for i in robot_slippery}
        for a in actions:
            row, col = self.inc(curr_row, curr_col, a)
            # Add robot slippery regions
            if (self.desc[row, col] in b"S" and ((row, col) not in self.robot_err or (row, col) in robot_err)) or \
                    (self.desc[row, col] in b"F" and ((row, col) in self.robot_err and (row, col) not in robot_err)):
                next_robot_slippery.add((row, col))

            # Add human slippery regions
            if (self.desc[row, col] in b"S" and ((row, col) not in self.human_err or (row, col) in human_err)) or \
                    (self.desc[row, col] in b"F" and ((row, col) in self.human_err and (row, col) not in human_err)):
                next_human_slippery.add((row, col))

        return next_human_slippery, next_robot_slippery

    def get_next_action(self, s, a):
        actions = [0, 1, 2, 3]
        actions.remove(a)
        action = np.random.choice(actions)
        s_robot = self.move(s, action)

        while (self.desc[s_robot // self.ncol, s_robot % self.ncol] == b"S" or \
               self.desc[s_robot // self.ncol, s_robot % self.ncol] == b"H" or s_robot == s) and len(actions) > 1:
            actions.remove(action)
            action = np.random.choice(actions)
            s_robot = self.move(s, action)
        return action

    def get_last_action(self, curr_position, last_position):
        curr_row = curr_position // self.ncol
        curr_col = curr_position % self.ncol
        last_row = last_position // self.ncol
        last_col = last_position % self.ncol
        if self.desc[last_row, last_col] in b'HS' and curr_position == 0:
            return 0  # The game was restarted, not sure what to return
        if curr_row == last_row:
            if curr_col == last_col + 1:
                return 2  # Right
            else:
                return 0  # Left
        else:
            if curr_row == last_row + 1:
                return 1  # Down
            else:
                return 3

    def augmented_state_transition(self, current_augmented_state, robot_action, human_action):
        # observed states
        current_world_state = current_augmented_state[:5]

        # Latent states
        human_trust = current_augmented_state[5]
        human_capability = current_augmented_state[6]

        # World state (observable) (human's)
        next_world_state = self.world_state_transition(current_world_state, robot_action, human_action)

        # Update human states based on human actions
        # Fixed
        next_human_capability = human_capability

        if not human_action:
            next_human_trust = human_trust
        elif human_action[0] == 0:
            # No assist condition: do not update trust
            next_human_trust = human_trust
        else:
            human_accept = human_action[0]  # 1 indicates accept, 2 indicates reject
            human_trust[human_accept - 1] += 1  # index 0 is acceptance count, index 1 is rejection count
            next_human_trust = human_trust

        next_augmented_state = [next_world_state[0], next_world_state[1], next_world_state[2],
                                next_world_state[3], next_world_state[4], next_human_trust, next_human_capability]

        return next_augmented_state

    def world_state_transition(self, current_world_state, robot_action, human_action):
        if human_action:
            return list(self.step(current_world_state, None, human_action))

        else:
            return list(self.step(current_world_state, robot_action, None))

    def step(self, current_world_state, robot_action, human_action):
        position_history, human_slippery, robot_slippery, human_err, robot_err = current_world_state
        position, last_position = position_history[-1], position_history[-2]
        if human_action != None:
            human_accept = human_action[0]
            human_detect = human_action[1]
            human_direction = human_action[2]
            if human_detect:  # Use detection function
                s = position
                next_human_slippery = {i for i in human_slippery}
                next_robot_slippery = {i for i in robot_slippery}
                next_human_err = {i for i in human_err}
                next_robot_err = {i for i in robot_err}
                detected_s = self.move(position, human_direction)
                row = detected_s // self.ncol
                col = detected_s % self.ncol
                if self.desc[row, col] in b'S':
                    if (row, col) not in next_human_slippery:
                        next_human_slippery.add((row, col))
                        next_human_err.add((row, col))

                elif self.desc[row, col] in b'F':
                    if (row, col) in next_human_slippery:
                        next_human_slippery.add((row, col))
                        next_human_err.add((row, col))

                next_position_history = position_history + [s]
                next_position_history.pop(0)

            else:  # No detection -> Move
                s = self.move(position, human_direction)
                next_human_slippery, next_robot_slippery = self.detect_slippery_region(s, human_slippery,
                                                                                       robot_slippery, human_err,
                                                                                       robot_err, detecting=True)
                next_position_history = position_history + [s]
                next_position_history.pop(0)
            return next_position_history, next_human_slippery, next_robot_slippery, human_err, robot_err

        else:
            robot_type = robot_action[0]
            robot_direction = robot_action[1]
            if robot_type == CONDITION['interrupt']:
                s = last_position
                next_position_history = position_history + [s]
                next_position_history.pop(0)
                return next_position_history, human_slippery, robot_slippery, human_err, robot_err
            elif robot_type == CONDITION['control']:
                s = self.move(last_position, robot_direction)

                if self.desc[s // self.ncol, s % self.ncol] in b'HS':  # if human falls into a hole, restart
                    s = 0
                    self.interrupted = 0
                    self.truncated = True
                    self.last_interrupt = [None, None]
                    # Remove the error an show the ground truth
                    next_human_slippery = {i for i in human_slippery}
                    next_robot_slippery = {i for i in robot_slippery}
                    next_human_err = {i for i in human_err}
                    next_robot_err = {i for i in robot_err}
                    if (s // self.ncol, s % self.ncol) not in human_slippery:
                        next_human_slippery.add((s // self.ncol, s % self.ncol))
                        if (s // self.ncol, s % self.ncol) in self.human_err:
                            next_human_err.add((s // self.ncol, s % self.ncol))
                    if (s // self.ncol, s % self.ncol) not in robot_slippery:
                        next_robot_slippery.add((s // self.ncol, s % self.ncol))
                        if (s // self.ncol, s % self.ncol) in self.robot_err:
                            next_robot_err.add((s // self.ncol, s % self.ncol))
                    next_position_history = position_history + [0]
                    next_position_history.pop(0)
                    return next_position_history, next_human_slippery, next_robot_slippery, next_human_err, next_robot_err
                next_human_slippery, next_robot_slippery = self.detect_slippery_region(s, human_slippery,
                                                                                       robot_slippery, human_err,
                                                                                       robot_err)
                next_position_history = position_history + [s]
                next_position_history.pop(0)
                return next_position_history, next_human_slippery, next_robot_slippery, human_err, robot_err
            else:
                curr_row = position // self.ncol
                curr_col = position % self.ncol
                if robot_type == CONDITION['no_interrupt'] and self.desc[
                    curr_row, curr_col] in b'HS':  # if human falls into a hole, restart
                    s = 0
                    self.interrupted = 0
                    self.truncated = True
                    self.last_interrupt = [None, None]
                    next_human_slippery = {i for i in human_slippery}
                    next_robot_slippery = {i for i in robot_slippery}
                    next_human_err = {i for i in human_err}
                    next_robot_err = {i for i in robot_err}
                    if (curr_row, curr_col) not in human_slippery:
                        next_human_slippery.add((curr_row, curr_col))
                        if (curr_row, curr_col) in self.human_err:
                            next_human_err.add((curr_row, curr_col))
                    if (curr_row, curr_col) not in robot_slippery:
                        next_robot_slippery.add((curr_row, curr_col))
                        if (curr_row, curr_col) in self.robot_err:
                            next_robot_err.add((curr_row, curr_col))
                    next_position_history = position_history + [s]
                    next_position_history.pop(0)
                    return next_position_history, next_human_slippery, next_robot_slippery, next_human_err, next_robot_err
                next_position_history = [p for p in position_history]
                return next_position_history, human_slippery, robot_slippery, human_err, robot_err

    def get_rollout_observation(self, current_augmented_state, robot_action):
        current_world_state = current_augmented_state[:3]
        current_human_trust = current_augmented_state[5]
        current_human_capability = current_augmented_state[6]
        current_position = current_world_state[0][0]
        last_position = current_world_state[0][1]
        # Get human action from heuristic_interrupt model (Needs access to game state info)
        robot_assist_type = robot_action[0]
        robot_direction = robot_action[1]
        human_slippery = current_augmented_state[1]
        robot_slippery = current_augmented_state[2]
        # When the robot is assisting -> human choice depends on trust
        human_acceptance_probability = (np.array(current_human_trust) / np.sum(current_human_trust))[0]

        # If acceptance <= prob < acceptance + 0.5(1-acceptance) then reject + no detection, if prob >= acceptance + 0.5(1-acceptance) then reject + detection
        actions = [0, 1, 2, 3]
        prob = np.random.uniform()
        detect = 0
        detect_new_grid_prob = 0.2
        if self.human_type == "random":
            if robot_assist_type == 0:
                # No assistance
                # human_choice = np.random.choice(4)
                accept = 0
                if human_acceptance_probability <= prob < 0.5 + 0.5 * human_acceptance_probability:
                    detect = 1
            elif robot_assist_type == 1:  # or robot_action_type == 3: #Interrupt
                # print(self.env.lastaction)
                # User either chooses the robot's suggestion or their own based on their trust in the robot and their capablity
                if robot_direction - 2 >= 0:
                    undo_action = robot_direction - 2
                else:
                    undo_action = robot_direction + 2
                if prob < human_acceptance_probability:
                    actions.remove(undo_action)
                    accept = 1  # Accept
                    # human_choice = np.random.choice(actions)
                elif human_acceptance_probability <= prob < 0.5 + 0.5 * human_acceptance_probability:
                    # Currently will choose the last position following epsilon-greedy strategy
                    if np.random.uniform() < detect_new_grid_prob:
                        actions.remove(undo_action)
                    else:
                        actions = [undo_action]
                    detect = 1
                    accept = 2
                else:
                    actions = [undo_action]
                    accept = 2  # Reject
            else:  # taking control
                if prob < human_acceptance_probability:
                    accept = 1
                    if robot_direction is not None:
                        if robot_direction - 2 >= 0:
                            actions.remove(robot_direction - 2)
                        else:
                            actions.remove(robot_direction + 2)
                # Detection when robot took over control: check one surrounding grid
                elif human_acceptance_probability <= prob < 0.5 + 0.5 * human_acceptance_probability:
                    detect = 1
                    accept = 2
                    if robot_direction is not None:
                        if robot_direction - 2 >= 0:
                            actions.remove(robot_direction - 2)
                        else:
                            actions.remove(robot_direction + 2)
                # Return to the last state after refusing
                else:
                    accept = 2
                    if robot_direction is not None:
                        if robot_direction - 2 >= 0:
                            actions = [robot_direction - 2]
                        else:
                            actions = [robot_direction + 2]

            human_choice = np.random.choice(actions)
            s = self.move(current_position, human_choice)
            while s == current_position and len(actions) > 1:
                actions.remove(human_choice)
                human_choice = np.random.choice(actions)
                s = self.move(current_position, human_choice)

        elif self.human_type == "epsilon_greedy":
            epsilon = 0.2
            if robot_assist_type == 0:
                # No assistance
                # human_choice = np.random.choice(4)
                accept = 0
                if human_acceptance_probability <= prob < 0.5 + 0.5 * human_acceptance_probability:
                    detect = 1
            elif robot_assist_type == 1:  # or robot_action_type == 3: #Interrupt
                # print(self.env.lastaction)
                # User either chooses the robot's suggestion or their own based on their trust in the robot and their capablity
                if robot_direction - 2 >= 0:
                    undo_action = robot_direction - 2
                else:
                    undo_action = robot_direction + 2
                if prob >= human_acceptance_probability:
                    actions.remove(undo_action)
                    accept = 1  # Accept
                    # human_choice = np.random.choice(actions)
                elif 0.5 * human_acceptance_probability <= prob < human_acceptance_probability:
                    # Currently will choose the last position following epsilon-greedy strategy
                    if np.random.uniform() < detect_new_grid_prob:
                        actions.remove(undo_action)
                    else:
                        actions = [undo_action]
                    detect = 1
                    accept = 2
                else:
                    actions = [undo_action]
                    accept = 2  # Reject
            else:  # taking control
                if prob >= human_acceptance_probability:
                    accept = 1
                    if robot_direction is not None:
                        if robot_direction - 2 >= 0:
                            actions.remove(robot_direction - 2)
                        else:
                            actions.remove(robot_direction + 2)
                # Detection when robot took over control: check one surrounding grid
                elif human_acceptance_probability <= prob < 0.5 + 0.5 * human_acceptance_probability:
                    detect = 1
                    accept = 2
                    if robot_direction is not None:
                        if robot_direction - 2 >= 0:
                            actions.remove(robot_direction - 2)
                        else:
                            actions.remove(robot_direction + 2)
                # Return to the last state after refusing
                else:
                    accept = 2
                    if robot_direction is not None:
                        if robot_direction - 2 >= 0:
                            actions = [robot_direction - 2]
                        else:
                            actions = [robot_direction + 2]

            shortest_path = find_shortest_path(self.desc, human_slippery, current_position, self.ncol)

            e = np.random.uniform()
            if len(shortest_path) < 2:  # No valid path
                # Temporally use robot slippery region as the ground truth
                true_shortest_path = find_shortest_path(self.desc, robot_slippery, current_position, self.ncol)
                if len(true_shortest_path) > 1:
                    true_best_action = true_shortest_path[1][1]
                else:
                    true_best_action = np.random.choice([0, 1, 2, 3])
                if e < epsilon:  # Choose action randomly
                    if true_best_action in actions and len(actions) > 1:
                        actions.remove(true_best_action)
                    human_choice = np.random.choice(actions)
                    s = self.move(current_position, human_choice)
                    while s == current_position and len(actions) > 1:
                        actions.remove(human_choice)
                        human_choice = np.random.choice(actions)
                        s = self.move(current_position, human_choice)
                else:  # Choose optimal action
                    human_choice = true_best_action
            # Choose best action using the human map
            else:
                best_action = shortest_path[1][1]
                if best_action in actions:
                    human_choice = best_action
                else:
                    human_choice = np.random.choice(actions)
                    s = self.move(current_position, human_choice)
                    while s == current_position and len(actions) > 1:
                        actions.remove(human_choice)
                        human_choice = np.random.choice(actions)
                        s = self.move(current_position, human_choice)

        return accept, detect, human_choice

    def reward(self, augmented_state, robot_action, human_action=None):
        position_history, human_slippery, robot_slippery = augmented_state[:3]
        position, last_position = position_history[-1], position_history[-2]
        # Get reward based on the optimality of the human action and the turn number
        # TODO: add penalty if robot takes control etc.
        curr_col = position // self.ncol
        curr_row = position % self.ncol
        last_col = last_position // self.ncol
        last_row = last_position % self.ncol
        reward = -1
        detect = None
        if human_action:
            human_accept, detect, human_choice = human_action
        if detect == 1:
            reward = -2
        if self.desc[curr_col, curr_row] in b'HS' or (self.desc[last_col, last_row] in b'HS' and robot_action[0] == 0):
            reward = -10
        elif self.desc[curr_col, curr_row] in b'G':
            reward = 30
        return reward

    def final_reward(self, augmented_state):
        # TODO
        return 0

    def get_human_action(self, robot_action=None):
        raise NotImplementedError

    def get_robot_action(self, world_state, robot_assistance_mode=0):
        # Robot's recommended action with or without explanations
        # For the purpose of data collection, the robot will follow static_take_control policies

        position = world_state[0][0]
        last_position = world_state[0][1]
        robot_slippery = world_state[2]

        curr_row = position // self.ncol
        curr_col = position % self.ncol

        # Interrupt
        if robot_assistance_mode == CONDITION['interrupt']:  # or robot_type == CONDITION['interrupt_w_explain']:
            # wait in the same state
            # if self.robot_map[curr_row, curr_col] in b'S':
            self.interrupted = 1
            if self.desc[curr_row, curr_col] == b"H":
                self.interrupted = 2
            # Useless if we store the previous position
            last_human_action = self.get_last_action(position, last_position)
            if last_human_action - 2 >= 0:
                undo_action = last_human_action - 2
            else:
                undo_action = last_human_action + 2
            self.robot_action = undo_action
            return robot_assistance_mode, undo_action
        # Take over control
        elif robot_assistance_mode == CONDITION['control']:  # or robot_type == CONDITION['control_w_explain']:
            # print("first interrupt")
            # shortest_path = find_shortest_path(self.robot_map, self.s, self.ncol)
            last_human_action = self.get_last_action(position, last_position)

            s_previous = last_position
            shortest_path = find_shortest_path(self.desc, robot_slippery, s_previous, self.ncol)
            if len(shortest_path) < 2:
                self.robot_action = self.get_next_action(s_previous, last_human_action)
            else:
                best_action = shortest_path[1][1]
                self.robot_action = best_action
            # if self.robot_map[curr_row, curr_col] == b'S':
            self.interrupted = 1
            if self.desc[curr_row, curr_col] == b"H":
                self.interrupted = 2
            self.last_interrupt = [position, last_human_action]
            return robot_assistance_mode, self.robot_action
        # No interruption
        else:
            self.robot_action = None
            return robot_assistance_mode, None

    def get_action_space(self, agent):
        if agent == "robot":
            return self.robot_action_space
        else:
            return self.human_action_space

    def get_observation_space(self, agent):
        if agent == "robot":
            return self.robot_observation_space
        else:
            raise NotImplementedError

    def reset(self):
        s = 0
        self.visited_slippery_region = []
        self.score = 0
        next_human_slippery, next_robot_slippery = self.detect_slippery_region(0, {i for i in self.hole},
                                                                               {i for i in (self.hole + self.slippery)},
                                                                               (), ())
        self.world_state = [[0, 0, 0, 0], next_human_slippery, next_robot_slippery, set(), set()]

        return [[0, 0, 0, 0], next_human_slippery, next_robot_slippery, set(), set()]

    def isTerminal(self, world_state):
        """
        Checks if the current world_state is a terminal state (i.e., either user found the code, or ran out of max turns)
        :param world_state:
        :return: returns true if world_state is terminal
        """
        position_history, human_slippery, robot_slippery, human_err, robot_err = world_state
        position, last_position = position_history[-1], position_history[-2]
        curr_col = position // self.ncol
        curr_row = position % self.ncol
        if self.desc[curr_col, curr_row] in b'G':
            # print("SUCCESS! You found the code {} in {} turns".format(self.answer, turn_num))
            # print("*******************************************************************")
            self.running = False
            return True
        return False

    # def render(self, map):
    #     desc = map.tolist()
    #     # outfile = StringIO()
    #
    #     row, col = self.s // self.ncol, self.s % self.ncol
    #     desc = [[c.decode("utf-8") for c in line] for line in desc]
    #     desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
    #     # print("\n")
    #     print("\n".join("".join(line) for line in desc))

    # with closing(outfile):
    #     return outfile.getvalue()


if __name__ == '__main__':
    # env = FrozenLakeEnv()
    map_num = 0
    map = MAPS["MAP" + str(map_num)]
    foggy = FOG["MAP" + str(map_num)]
    human_err = HUMAN_ERR["MAP" + str(map_num)]
    robot_err = ROBOT_ERR["MAP" + str(map_num)]
    # slippery_region = SLIPPERY["MAP" + str(round + 1)]
    env = FrozenLakeEnv(desc=map, foggy=foggy, human_err=human_err, robot_err=robot_err,
                        is_slippery=False, render_mode="human", round=0)
    curr_augmented_state = env.reset()
    # world_state = env.view_world_state()
    # print(world_state)
    while env.running:
        # input("press Enter to continue...")
        print("----------------------------------------------------------------------")
        curr_human_action = env.get_rollout_observation(curr_augmented_state, CONDITION["no_interrupt"])
        print("Human action: {}".format(curr_human_action))
        # Get the robot's suggestion for the current turn
        # robot_action = env.get_robot_action(random.choice([0, 1, 2]))
        robot_action = env.get_robot_action(2)
        print("Robot's suggestion: {}".format(robot_action))
        print("----------------------------------------------------------------------")
        curr_augmented_state = env.augmented_state_transition(curr_augmented_state,
                                                              robot_action=robot_action,
                                                              human_action=curr_human_action)
        reward = env.reward(curr_augmented_state, curr_human_action)
        print("State: {}".format((curr_augmented_state[0])))
        print("Reward: {}".format(reward))
        curr_human_action = env.get_rollout_observation(curr_augmented_state, robot_action)
        print("Human responce: {}".format(curr_human_action))
        print("----------------------------------------------------------------------")
        curr_augmented_state = env.augmented_state_transition(curr_augmented_state,
                                                              robot_action=CONDITION["no_interrupt"],
                                                              human_action=curr_human_action)
        reward = env.reward(curr_augmented_state, curr_human_action)
        print("State: {}".format((curr_augmented_state[0])))
        print("Reward: {}".format(reward))
        position = curr_augmented_state[0]
        curr_col = position // env.ncol
        curr_row = position % env.ncol
        if env.desc[curr_col, curr_row] in b'H':
            env.s = 0
            env.interrupted = 0
            env.last_interrupt = [None, None]
            info = env.update_grid_info(env.s)
            curr_augmented_state[:6] = [env.s, info[0], info[1], info[2], info[3], info[4]]

    print("----------------------------------------------------------------------")
    print("Final Reward: {}".format(env.final_reward(curr_augmented_state)))
