# TODO: Redo this since we do not have any capability matrix
import random as rand
import operator
import os
from queue import Queue
from frozen_lake.frozen_lake_env import *


class SimulatedHuman:
    """
    The simulated human.
    """

    def __init__(self, env, true_trust=(1, 0),
                true_capability=(0, 0, 0, 0), alpha=0.5, type="random", seed=0):
        """
        Initializes an instance of simulated human.

        :param env: the environment
        :type: Environment
        :param pedagogy_constant: the chance of human demonstrating incapable action
        :type pedagogy_constant: float
        :param decay: the decay rate of chance of human demonstrating incapable action
        :type decay: float
        """
        self.env = env
        self.alpha = alpha  # mixing policy (how much to rely on the robot vs the user's independent decision) for human's decision.
        self.true_human_trust = true_trust
        self.true_human_capability = true_capability
        self.type = type
        rand.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        self.detected_grids = []  # Keep a list of grids that were previously detected
        self.memory_len = 10  # Length to determine bounded memory of previously visited grids
        self.recent_visited_grids = []

    def simulateHumanAction(self, world_state, robot_action):
        """
        Simulates actual human action given the actual robot action.

        :param world_state: the current world state
        :type world_state: list
        :param actual_robot_action: the current actual robot action
        :type actual_robot_action: list representing one hot vector of actual robot action

        :return: rollout intended human action, rollout actual human action
        :rtype: lists representing one hot vector of intended and actual human actions
        """
        # TODO: Currently human behavior is fixed, i.e., trust in the agent and capability are not updated.
        current_position, last_position, human_slippery, robot_slippery, human_err, robot_err = world_state

        robot_assist_type = robot_action[0]
        robot_direction = robot_action[1]
        true_human_trust = self.true_human_trust
        true_human_capability = self.true_human_capability

        human_acceptance_probability = np.random.beta(true_human_trust[0], true_human_trust[1])
        # human_acceptance_probability = (np.array(true_human_trust) / np.sum(true_human_trust))[0]

        # For actions with explanation, increase the human acceptance probability
        if robot_assist_type in [3, 4]:
            human_acceptance_probability = np.minimum(human_acceptance_probability + 0.1, 1.0)

        # Human's action decision is defined by:
        # - the underlying task difficulty
        # - their capability
        # - their trust in the agent
        # - the robot's action (whether it provided explanations)

        # If acceptance <= prob < acceptance + 0.5(1-acceptance) then reject + no detection, if prob >= acceptance + 0.5(1-acceptance) then reject + detection
        actions = [0, 1, 2, 3]
        prob = np.random.uniform()
        detect = 0
        detect_new_grid_prob = 0.2
        if self.type == "random":
            if robot_assist_type == 0:
                # No assistance
                # human_choice = np.random.choice(4)
                accept = 0
            elif robot_assist_type == 1 or robot_assist_type == 3:  # Interrupt
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
                elif human_acceptance_probability <= prob < 0.5 + 0.5*human_acceptance_probability:
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
            s = self.env.move(current_position, human_choice)
            while s == current_position and len(actions) > 1:
                actions.remove(human_choice)
                human_choice = np.random.choice(actions)
                s = self.env.move(current_position, human_choice)

        # The human will always choose the optimal action based on the human map, and when there's no valid path,
        # if rand() < epsilon, then choose suboptimal action, otherwise choose the ground truth optimal action.
        elif self.type == "epsilon_greedy":
            epsilon = 0.2

            if robot_assist_type == 0:
                # No assistance
                # human_choice = np.random.choice(4)
                accept = 0
                if human_acceptance_probability <= prob < 0.5 + 0.5*human_acceptance_probability:
                    detect = 1

            elif robot_assist_type == 1 or robot_assist_type == 3: #Interrupt
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
                elif human_acceptance_probability <= prob < 0.5 + 0.5*human_acceptance_probability:
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

            # Determine direction of human action (TODO: Fix epsilon-greedy policy to mostly choose from the human knowledge)

            shortest_path = self.env.find_shortest_path(self.env.desc, human_slippery, current_position, self.env.ncol)

            e = np.random.uniform()
            # if len(shortest_path) < 2: # No valid path
            #     # Temporally use robot slippery region as the ground truth
            true_shortest_path = self.env.find_shortest_path(self.env.desc, self.env.hole + self.env.slippery, current_position, self.env.ncol)
            if len(true_shortest_path) > 1:
                true_best_action = true_shortest_path[1][1]
            else:
                true_best_action = np.random.choice([0, 1, 2, 3])
            if e < epsilon: # Choose best action using the human map
                if len(shortest_path) > 1:
                    best_action = shortest_path[1][1]
                else:
                    best_action = np.random.choice([0, 1, 2, 3])
                if best_action in actions:
                    human_choice = best_action
                else:
                    # Cannot choose the best action because of acceptance,
                    # so randomly choose another suboptimal action
                    if true_best_action in actions and len(actions) > 1:
                        actions.remove(true_best_action)
                    human_choice = np.random.choice(actions)
                    s = self.env.move(current_position, human_choice)
                    while s == current_position and len(actions) > 1:
                        actions.remove(human_choice)
                        human_choice = np.random.choice(actions)
                        s = self.env.move(current_position, human_choice)
            else: # Choose optimal action
                human_choice = true_best_action

            # Check if the action is detect whether we've already previously detected the state.
            if detect == 1:
                s = self.env.move(current_position, human_choice)
                if s == 63:
                    # Reached goal
                    accept = 0 if robot_assist_type == 0 else 2
                    detect = 0  # Do not detect goal

                elif s not in self.detected_grids:
                    self.detected_grids.append(s)  # append to the list of detected grids
                else:
                    curr_row = s // self.env.ncol
                    curr_col = s % self.env.ncol
                    if s in self.detected_grids or s in self.recent_visited_grids:
                        # Do not detect (since we recently detected or visited and know it to be safe)
                        # Just move
                        accept = 0 if robot_assist_type == 0 else 2
                        detect = 0
            #     #     while s in self.detected_grids or s in self.recent_visited_grids\
            #     #             or self.env.desc[curr_row][curr_col] == b'H'\
            #     #             or self.env.desc[curr_row][curr_col] == b'G':
            #     #         if human_choice in actions:
            #     #             actions.remove(human_choice)
            #     #         if len(actions) == 0:
            #     #             # Need to check still if this action is not leading to a hole...
            #     #             while self.env.desc[curr_row][curr_col] == b'H':
            #     #                 human_choice = np.random.choice(np.arange(4))
            #     #                 s = self.env.move(current_position, human_choice)
            #     #                 curr_row = s // self.env.ncol
            #     #                 curr_col = s % self.env.ncol
            #     #             accept = 0 if robot_assist_type == 0 else 2
            #     #             detect = 0  # Do not detect as there's no feasible direction
            #     #             break
            #     #         human_choice = np.random.choice(actions)
            #     #         s = self.env.move(current_position, human_choice)
            #     #         curr_row = s // self.env.ncol
            #     #         curr_col = s % self.env.ncol
            # #
                if detect == 1 and s not in self.detected_grids:  # Need to check again after exiting the while loop
                    self.detected_grids.append(s)
            #
            if detect != 1:
                s = self.env.move(current_position, human_choice)
                while len(self.recent_visited_grids) >= self.memory_len:
                    self.recent_visited_grids.pop(0)  # Remove the oldest grid visited
                if s not in self.recent_visited_grids:
                    self.recent_visited_grids.append(s)  # Only append if not in recent visits

        return accept, detect, human_choice
