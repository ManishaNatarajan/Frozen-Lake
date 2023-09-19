"""
Script to evaluate different baselines for comparing human-robot performance in Tiger DSS
Baselines:
- Random (Robot chooses a random policy to assist user)
- Inverse Reward POMCP
- Static Policy (always provides one type of assistance: assist / assist + explanations)
- No assistance
"""

import random
import numpy as np
import os
import time
from frozen_lake.frozen_lake_env import FrozenLakeEnv
from frozen_lake.simulated_human import *
import math


class RandomAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def get_action(self, env, human_action):
        return np.random.choice(self.num_actions)


class StaticAgent:
    def __init__(self, fixed_action=1):
        self.robot_action = fixed_action

    def get_action(self, env, human_action):
        return self.robot_action


class NoAssistAgent:
    def __init__(self):
        self.robot_action = 0

    def get_action(self, env, human_action):
        return self.robot_action


class HeuristicAgent:
    def __init__(self, type):
        self.robot_action = 0
        self.type = type
        self.num_interrupt = 0  # Number of interruption when taking a longer path (<3)

    def get_action(self, env, human_action):
        position = env.world_state[0]
        last_position = env.world_state[1]
        robot_slippery = env.world_state[3]
        robot_err = env.world_state[5]
        last_path = env.find_shortest_path(env.desc, robot_slippery, last_position, env.ncol)
        current_path = env.find_shortest_path(env.desc, robot_slippery, position, env.ncol)
        if env.desc[position // env.ncol, position % env.ncol] in b'HS' and \
                ((position // env.ncol, position % env.ncol) not in env.robot_err or
                 (position // env.ncol, position % env.ncol) in robot_err):
            self.robot_action = self.type
        elif env.desc[position // env.ncol, position % env.ncol] in b'F' and \
                ((position // env.ncol, position % env.ncol) in env.robot_err and
                 (position // env.ncol, position % env.ncol) not in robot_err):
            self.robot_action = self.type
        elif len(current_path) > 1 and len(last_path) > 1 and len(last_path) <= len(current_path) and \
                self.num_interrupt < 3:
            self.robot_action = self.type
            self.num_interrupt += 1
            # print("Longer path!!!!!!!!!!!!!!!!!")
        else:
            self.robot_action = 0

        return self.robot_action


def execute(round_num, num_steps, env, human_agent, robot_agent=None, render_game_states=False):
    robot_actions = []
    human_actions = []
    num_actions = 3

    if robot_agent is None:
        robot_agent = RandomAgent(num_actions)

    final_env_reward = 0

    # print("Execute round {} of search".format(round_num))
    start_time = time.time()

    human_action = human_agent.simulateHumanAction(env.world_state, (0, None))
    last_robot_action = (0, None)

    # update the environment
    env.world_state = env.world_state_transition(env.world_state, None, human_action)
    final_env_reward += env.reward(env.world_state, (0, None), human_action)

    human_actions.append(human_action)

    for step in range(num_steps):
        if last_robot_action[0] or human_action[1]:
            robot_action_type = 0  # Cannot interrupt twice successively
        else:
            robot_action_type = robot_agent.get_action(env, human_action)

        robot_action = env.get_robot_action(env.world_state, robot_action_type)
        last_robot_action = robot_action

        # update the environment
        env.world_state = env.world_state_transition(env.world_state, robot_action, None)
        # print(env.world_state)
        # print("Robot action", robot_action)
        if render_game_states:
            env.render(env.desc)

        human_action = human_agent.simulateHumanAction(env.world_state, robot_action)
        # human_action = [human_action[0], 0, human_action[2]]


        curr_reward = env.reward(env.world_state, robot_action, human_action)
        # print(curr_reward)
        final_env_reward += curr_reward
        if env.isTerminal(env.world_state):
            break

        # update the environment
        env.world_state = env.world_state_transition(env.world_state, None, human_action)
        # print(env.world_state)
        # print("Human action", human_action)
        if render_game_states:
            env.render(env.desc)
        # print("----------------------------------------------------------------------------------------------")
        robot_actions.append(robot_action)
        human_actions.append(human_action)

    # print("===================================================================================================")
    # print("Round {} completed!".format(round_num))
    # print("Time taken:")
    # print("{} seconds".format(time.time() - start_time))
    # print('Robot Actions: {}'.format(robot_actions))
    # print('Human Actions: {}'.format(human_actions))
    # print('Final Reward: {}'.format(final_env_reward))

    # Final env reward is calculated based on the true state of the tiger and what the human finally decided to do
    # The step loop terminates if the env terminates

    return final_env_reward


if __name__ == '__main__':
    # Set appropriate seeds

    # Initialize constants for setting up the environment
    max_steps = 80
    num_choices = 3

    # Human latent parameters (set different values for each test)
    true_trust = [(5, 50), (10, 40), (24, 36), (40, 45), (45, 20), (99, 1)]
    # true_trust = [(5, 50), (5, 50), (5, 50), (5, 50), (5, 50), (5, 50), (5, 50), (5, 50), (5, 50), (5, 50), (5, 50)]
    true_capability = 0.85  # fixed - parameter (assume known??) at the start of the study
    # true_aggressiveness = (25, 25)

    # Executes num_tests of experiments
    num_test = 6


    # factors for POMCP (also used in the environment for get_observations which uses UCT for human policy)
    gamma = 0.99
    c = 20  # Exploration bonus
    beta = 0.9

    e = 0.1  # For epsilon-greedy policy
    epsilon = math.pow(gamma, 30)  # tolerance factor to terminate rollout
    num_iter = 100
    num_steps = max_steps
    initial_belief = []

    for n in range(num_test):
        print("*********************************************************************")
        print(f"Executing test number {n}, Trust:{true_trust[n]}......")
        # print("*********************************************************************")
        mean_rewards = []
        std_rewards = []
        all_rewards = []
        for SEED in [0, 5, 21, 25, 42]:
            random.seed(SEED)
            np.random.seed(SEED)
            os.environ['PYTHONHASHSEED'] = str(SEED)

            # Setup Driver
            map_num = 12
            map = MAPS["MAP" + str(map_num)]
            foggy = FOG["MAP" + str(map_num)]
            human_err = HUMAN_ERR["MAP" + str(map_num)]
            robot_err = ROBOT_ERR["MAP" + str(map_num)]
            env = FrozenLakeEnv(desc=map, foggy=foggy, human_err=human_err, robot_err=robot_err,
                                is_slippery=False, render_mode="human", true_human_trust=true_trust[n],
                                true_human_capability=true_capability,
                                true_robot_capability=0.85, beta=beta,
                                c=c, gamma=gamma, seed=SEED,
                                human_type="epsilon_greedy")

            # Reset the environment to initialize everything correctly
            env.reset()
            # robot_policy = RandomAgent(num_choices)
            # robot_policy = StaticAgent(fixed_action=2)
            # robot_policy = NoAssistAgent()
            robot_policy = HeuristicAgent(type=2)

            simulated_human = SimulatedHuman(env, true_trust=true_trust[n],
                                             true_capability=true_capability,
                                             type="epsilon_greedy")  # TODO: Can test with different humans

            # Executes num_rounds of search (calibration)
            num_rounds = 1
            total_env_reward = 0


            for i in range(num_rounds):
                # We should only change the true state of the tiger for every round (or after every termination)
                env.reset()  # Note tiger_idx can be either 0 or 1 indicating left or right door

                env_reward = execute(round_num=i, num_steps=max_steps, env=env,
                                     human_agent=simulated_human, robot_agent=robot_policy)
                mean_rewards.append(env_reward)
                total_env_reward += env_reward

            # print("===================================================================================================")
            # print("===================================================================================================")
            # print("Average environmental reward after {} rounds:{}".format(num_rounds,
            #                                                                total_env_reward / float(num_rounds)))
            # all_rewards.append(rewards)
            # mean_rewards.append(np.mean(rewards))
            # std_rewards.append(np.std(rewards))
        # print("===================================================================================================")
        # print("===================================================================================================")
        print(mean_rewards)
        print("===================================================================================================")
        # print("===================================================================================================")

        all_rewards = np.array(all_rewards)
        # with open("files/random/{}.npy".format(SEED), 'wb') as f:
        # with open("files/no_assist/{}.npy".format(SEED), 'wb') as f:
        # with open("frozen_lake/files/static_take_control/{}.npy".format(SEED), 'wb') as f:
        # with open("files/heuristic_interrupt/{}.npy".format(SEED), 'wb') as f:
        # with open("files/heuristic_test/map{}_interrupt_{}.npy".format(map_num, SEED), 'wb') as f:
        #     np.save(f, all_rewards)