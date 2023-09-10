### Main function to learn the POMCP policy for Mastermind (Simulation)
# Implemented by Manisha Natarajan
# Last Update: 04/24/2023

import os

from frozen_lake.frozen_lake_env import *
from frozen_lake.solver import *
from frozen_lake.root_node import *
from frozen_lake.robot_action_node import *
from frozen_lake.human_action_node import *
from frozen_lake.simulated_human import *
from utils import *
import time
from visualize_rollouts import view_rollout
from simulated_human_experiments.utils import *
import pygad


class Driver:
    def __init__(self, env, solver, num_steps, simulated_human):
        """
        Initializes a driver : uses particle filter to maintain belief over hidden states,
        and uses POMCP to determine the optimal robot action

        :param env: (type: Environment) Instance of the Mastermind environment
        :param solver: (type: POMCPSolver) Instance of the POMCP Solver for the robot policy
        :param num_steps: (type: int) number of actions allowed -- I think it's the depth of search in the tree
        :param simulated_human: (type: SimulatedHuman) the simulated human model
        """
        self.env = env
        self.solver = solver
        self.num_steps = num_steps
        self.simulated_human = simulated_human
        self.num_world_states = len(self.env.world_state)
        self.num_total_states = self.num_world_states + 2  # Should change this to 1, as we are not looking at capability

    def invigorate_belief(self, current_human_action_node, parent_human_action_node, robot_action, human_action, env):
        """
        Invigorates the belief space when a new human action node is created
        Updates the belief to match the world state, whenever a new human action node is created
        :param current_human_action_node: Current human action node is the hao node.
        :param parent_human_action_node: Parent human action node is the h node (root of the current search tree).
        :param robot_action: Robot action (a) taken after parent node state
        :param human_action: Human action (o) in response to robot action
        :param env: gym env object to determine current world state of the Frozen Lake
        :return:
        """

        for belief_state in parent_human_action_node.belief:
            # Update the belief world state for the current human action node
            # if the belief of the parent human action node is the same as the actual world state

            # Update parent belief state to match world state (i.e., after robot action)
            belief_state = env.augmented_state_transition(belief_state, robot_action, None)
            # if belief_state[:2] == env.world_state[:2]:
            if belief_state[:6] == env.world_state:
                next_augmented_state = env.augmented_state_transition(belief_state, None, human_action)
                current_human_action_node.update_belief(next_augmented_state)
            else:
                print("Node belief is empty!!! Particle Reinvigoration failed!!!")

    def updateBeliefWorldState(self, human_action_node, env):
        """
        Updates the world state in the belief if there are any discrepancies...
        :param human_action_node:
        :param env:
        :return:
        """
        if len(human_action_node.belief) == 0:
            print("Node belief is empty!!!")
            return
        # Update the belief (i.e., all particles) in the current node to match the current world state
        if human_action_node.belief[0][:self.num_world_states] != env.world_state:
            human_action_node.belief = [[env.world_state[0], env.world_state[1], env.world_state[2], env.world_state[3],
                                         env.world_state[4], env.world_state[5], belief[6], belief[7]] for belief in
                                        human_action_node.belief]

    def updateBeliefTrust(self, human_action_node, human_action, robot_action=(0,None)):
        """
        Updates belief of human trust based on true user's action
        TODO: I feel like this is a redundant update (but this was in the code for the user study...)
        :param human_action_node:
        :param human_action:
        :return:
        """
        # BayMax Update
        # Here, I use the same update as in the augmented_state_transition function in the env for updating the belief
        # about user trust

        # ------------------------------------ Define GA parameters ----------------------------------------------- #
        def fitness_func(ga_instance, solution, solution_idx):
            # Fitness function is a combination of the predicted accuracy, and entropy of the distribution
            predicted_action = get_user_action_from_beta(solution, robot_action)
            prediction_accuracy = distance_from_true_action(human_action, predicted_action)

            fitness_val = prediction_accuracy + 0.5 * get_entropy(solution)
            return fitness_val

        fitness_function = fitness_func

        num_generations = 100
        num_parents_mating = 5

        sol_per_pop = 5
        num_genes = 2

        # To initialize population
        init_range_low = -2
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
                               random_seed=SEED)

        # Use the current set of particles as the initial population for the Genetic Algorithm
        particle_set = []
        for belief in human_action_node.belief:
            particle_set.append(belief[self.num_world_states])

        ga_instance.initial_population = np.array(particle_set)  # Must be a numpy array

        # Run the genetic algorithm to get the next gen particles
        ga_instance.run()

        # Update particles in node belief
        new_particle_set = list(ga_instance.population)  # convert back to list
        # prediction_counts = prediction_counts_after_belief_update(new_particle_set, robot_action)

        human_accept, detect, human_choice_idx = human_action  # human accept: 0:no-assist, 1:accept, 2:reject


        #  World state should be the same for all particles
        world_state = human_action_node.belief[0]
        updated_belief = []
        for i, p in enumerate(new_particle_set):
            updated_belief.append(copy.deepcopy(world_state))
            updated_belief[i][self.num_world_states] = list(p)
            if human_accept != 0:  # In case of robot assistance
                # Update trust
                updated_belief[i][self.num_world_states][human_accept - 1] += 1  # index 0 is acceptance count, index 1 is rejection count

        human_action_node.belief = updated_belief

    def updateRootCapabilitiesBelief(self, root_node, current_node):
        """
        Updates the root belief about capabilities to the capabilities of the current node.
        :param root_node:
        :param current_node:
        :return:
        """
        initial_world_state = copy.deepcopy(self.env.world_state)
        root_node.belief = []
        num_samples = 10000
        # Sample belief_trust and belief_capability from a distribution
        sampled_beliefs = random.sample(current_node.belief, num_samples) if len(
            current_node.belief) > num_samples else current_node.belief
        root_node.belief.extend([[initial_world_state[0], initial_world_state[1], initial_world_state[2],
                                  initial_world_state[3], initial_world_state[4], initial_world_state[5],
                                  current_node_belief[6], current_node_belief[7]] for current_node_belief in
                                 sampled_beliefs])

    def execute(self, round_num, render_game_states=False, debug_tree=False):
        """
        Executes one round of search with the POMCP policy
        :param round_num: (type: int) the round number of the current execution
        :param render_game_states: (type: bool) to render the game state in the terminal for debugging
        :param debug_tree: (type: bool) to visualize the POMCP search tree after every iteration for debugging
        :return: (type: float) final reward from the environment
        """
        robot_actions = []
        human_actions = []
        all_states = []

        # create a deep copy of the env and the solver
        env = copy.deepcopy(self.env)
        solver = copy.deepcopy(self.solver)

        print("Execute round {} of search".format(round_num))
        start_time = time.time()
        final_env_reward = 0

        # Initial human action
        robot_action = (0, None)  # No interruption (default assumption since human takes the first action)
        human_action = self.simulated_human.simulateHumanAction(env.world_state, robot_action)

        print("Human Initial Action: ", human_action)

        # Here we are adding to the tree as this will become the root for the search
        human_action_node = HumanActionNode(env)
        # We call the invigorate belief whenever we add a new human action node to the tree
        self.invigorate_belief(human_action_node, solver.root_action_node, robot_action, human_action, env)
        solver.root_action_node = human_action_node
        env.world_state = env.world_state_transition(env.world_state, robot_action, human_action)
        all_states.append(env.world_state[0])
        final_env_reward += env.reward(env.world_state, (0, None), human_action)
        human_actions.append(human_action)

        for step in range(self.num_steps):
            t = time.time()
            # Do not intervene if user is using the detection sensor
            if human_action[1] == 1:
                robot_action_type = 0
            else:
                robot_action_type = solver.search()  # One iteration of the POMCP search  # Here the robot action indicates the type of assistance

            robot_action = env.get_robot_action(env.world_state[:6], robot_action_type)
            print("Robot action: ", robot_action)
            robot_action_node = solver.root_action_node.robot_node_children[robot_action[0]]

            # if debug_tree:
            #     visualize_tree(solver.root_action_node)

            if robot_action_node == "empty":
                # We're not adding to the tree though here
                # It doesn't matter because we are going to update the root from h to hao
                robot_action_node = RobotActionNode(env)

            # Update the environment
            env.world_state = env.world_state_transition(env.world_state, robot_action, None)
            robot_action_node.position = env.world_state[0]

            all_states.append(env.world_state[0])
            if render_game_states:
                env.render(env.desc)

            # We finally use the real observation / human action (i.e., from the simulated human model)

            # Note here that it is not the augmented state
            # (the latent parameters are already defined in the SimulatedHuman model I think)
            human_action = self.simulated_human.simulateHumanAction(env.world_state, robot_action)
            human_action_node = robot_action_node.human_node_children[human_action[1] * 4 + human_action[2]]

            final_env_reward += env.reward(env.world_state, robot_action, human_action)

            # Terminates if goal is reached
            if env.isTerminal(env.world_state):
                print("Final reward: ", final_env_reward)
                break

            if human_action_node == "empty":
                # Here we are adding to the tree as this will become the root for the search in the next turn
                human_action_node = robot_action_node.human_node_children[
                    human_action[1] * 4 + human_action[2]] = HumanActionNode(env)
                # We call the invigorate belief whenever we add a new human action node to the tree
                self.invigorate_belief(human_action_node, solver.root_action_node, robot_action, human_action, env)

            # Update the environment
            solver.root_action_node = human_action_node  # Update the root node from h to hao
            env.world_state = env.world_state_transition(env.world_state, robot_action, human_action)
            # all_states.append(env.world_state[0])
            
            # Updates the world state in the belief to match the actual world state
            # Technically if all the belief updates are performed correctly, then there's no need for this.
            self.updateBeliefWorldState(human_action_node, env)

            # Updates robot's belief of human reliance based on the current human action
            # TODO: We cannot really evaluate the outcome at every turn... but I'm still updating based on choice optimality
            #  So should I only update human capability after the round is over?
            #  Should this come before root node transfer? It dm in their case
            self.updateBeliefTrust(human_action_node, human_action, robot_action)  # For now I'm updating every turn.
            print("Human action: ", human_action)

            if render_game_states:
                env.render(env.desc)

            robot_actions.append(robot_action)
            human_actions.append(human_action)

        # Transfer current capabilities beliefs to the next round
        self.updateRootCapabilitiesBelief(self.solver.root_action_node, solver.root_action_node)

        print("===================================================================================================")
        print("Round {} completed!".format(round_num))
        print("Time taken:")
        print("{} seconds".format(time.time() - start_time))
        print('Robot Actions: {}'.format(robot_actions))
        print('Human Actions: {}'.format(human_actions))
        print("final world state for the round: {}".format(final_env_reward))

        return final_env_reward, human_actions, robot_actions, all_states


if __name__ == '__main__':
    # Set appropriate seeds
    for SEED in [0]:  # [0, 5, 21, 25, 42]
        random.seed(SEED)
        np.random.seed(SEED)
        os.environ['PYTHONHASHSEED'] = str(SEED)

        # Initialize constants for setting up the environment
        max_steps = 80
        num_choices = 3

        # Human latent parameters (set different values for each test)
        true_trust = [(5, 50), (10, 40), (18, 40), (24, 36), (35, 35), (40, 65), (45, 20), (45, 56),
                      (40, 45), (99, 1)]
        # true_trust = [(99, 1)]
        true_capability = 0.85  # fixed - parameter (assume known??) at the start of the study

        # The following two parameters are for human behavior. They are currently not used.
        human_behavior = "rational"  # TODO: they use this in the observation function in the environment
        beta = 0.9  # Boltzmann rationality parameter (for human behavior)

        # factors for POMCP
        gamma = 0.99  # gamma for terminating rollout based on depth in MCTS
        c = 20  # 400  # exploration constant for UCT (taken as R_high - R_low)
        e = 0.1  # For epsilon-greedy policy
        epsilon = math.pow(gamma, 30)  # tolerance factor to terminate rollout
        num_iter = 100
        num_steps = max_steps
        initial_belief = []

        # Executes num_tests of experiments

        num_test = 1
        mean_rewards = []
        std_rewards = []
        all_rewards = []
        for n in range(num_test):
            print("*********************************************************************")
            print("Executing test number {}......".format(n))
            print("*********************************************************************")

            # Robot's belief of human parameters
            all_initial_belief_trust = []
            for _ in range(1000):
                all_initial_belief_trust.append((1, 1))

            # Setup Driver
            map_num = 5
            map = MAPS["MAP" + str(map_num)]
            foggy = FOG["MAP" + str(map_num)]
            human_err = HUMAN_ERR["MAP" + str(map_num)]
            robot_err = ROBOT_ERR["MAP" + str(map_num)]
            # slippery_region = SLIPPERY["MAP" + str(round + 1)]
            env = FrozenLakeEnv(desc=map, foggy=foggy, human_err=human_err, robot_err=robot_err,
                                is_slippery=False, render_mode="human", true_human_trust=true_trust[-1],
                                true_human_capability=true_capability,
                                true_robot_capability=0.85, beta=beta, c=c, gamma=gamma, seed=SEED,
                                human_type="epsilon_greedy")

            # Reset the environment to initialize everything correctly
            env.reset()
            init_world_state = env.world_state

            # TODO: Initialize belief: Currently only using the 4 combinations
            for i in range(len(all_initial_belief_trust)):
                initial_belief.append(
                    [init_world_state[0], init_world_state[1], init_world_state[2], init_world_state[3],
                     init_world_state[4], init_world_state[5],
                     list(all_initial_belief_trust[i]),
                     true_capability])

            root_node = RootNode(env, initial_belief)
            solver = POMCPSolver(epsilon, env, root_node, num_iter, c)
            simulated_human = SimulatedHuman(env, true_trust=true_trust[-1],
                                             true_capability=true_capability,
                                             type="epsilon_greedy")

            driver = Driver(env, solver, num_steps, simulated_human)

            # Executes num_rounds of search (calibration)
            num_rounds = 1
            total_env_reward = 0

            rewards = []
            for i in range(num_rounds):
                # We should only change the true state of the tiger for every round (or after every termination)
                driver.env.reset()  # Note tiger_idx can be either 0 or 1 indicating left or right door
                env_reward, human_actions, robot_actions, all_states = driver.execute(i, debug_tree=False)
                rewards.append(env_reward)
                total_env_reward += env_reward

            print("===================================================================================================")
            print("===================================================================================================")
            print("Average environmental reward after {} rounds:{}".format(num_rounds,
                                                                           total_env_reward / float(num_rounds)))
            print("Num Particles: ", len(driver.solver.root_action_node.belief))

            # To visualize the rollout...

            history = []
            print("Path: ", all_states)
            for t in range(len(human_actions)-1):
                history.append({"human_action": list(human_actions[t]),
                                "robot_action": list(robot_actions[t])})

            user_data = {"mapOrder": [map_num],
                         str(i): {
                             "history": history
                            }
                         }

            view_rollout(user_data, rollout_idx=0)

            all_rewards.append(rewards)
            mean_rewards.append(np.mean(rewards))
            std_rewards.append(np.std(rewards))

        print("===================================================================================================")
        print("===================================================================================================")
        print(mean_rewards, std_rewards)
        print("===================================================================================================")
        print("===================================================================================================")

        all_rewards = np.array(all_rewards)
        # with open("files/pomcp_test/map{}_iter{}_{}.npy".format(map_num, num_iter, SEED), 'wb') as f:
        #     np.save(f, all_rewards)
