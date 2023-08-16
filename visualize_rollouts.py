from frozen_lake.simulated_human import *
import time
import pygame
import json

from frozen_lake.frozen_lake_interface import FrozenLakeEnvInterface

order = [4, 8, 6]
heuristic_order = [0, 1]  # First one is the order of interrupting agent, second is the order of taking control agent.
random.shuffle(heuristic_order)
CONDITION = {
    'practice': [0, 1, 2, 3],
    'pomcp': [order[0], order[0] + 1],
    'pomcp_inverse': [order[1], order[1] + 1],
    'interrupt': [order[2] + heuristic_order[0]],
    'take_control': [order[2] + heuristic_order[1]]
}


if __name__ == '__main__':
    # Set seed for reproducibility
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

    # Initialize pygame
    pygame.init()
    pygame.display.init()
    pygame.display.set_caption("Frozen Lake")
    window_surface = pygame.display.set_mode((min(100 * 8, 1024) + 256 * 2, min(100 * 8, 800) + 300))
    pygame.font.init()
    font = pygame.font.Font(None, 30)

    # Load a trajectory from json
    with open("frozen_lake/files/user_study/cHtG2.json", "r") as f:
        user_data = json.load(f)

    # Ignore the practice rollouts
    map_num = user_data["mapOrder"][4]
    rollout = user_data["4"]["history"]
    map = MAPS["MAP" + str(map_num)]
    foggy = FOG["MAP" + str(map_num)]
    human_err = HUMAN_ERR["MAP" + str(map_num)]
    robot_err = ROBOT_ERR["MAP" + str(map_num)]

    round_num = 1

    # Create Env based on the first rollout after practice rounds...
    env = FrozenLakeEnvInterface(desc=map, foggy=foggy, human_err=human_err, robot_err=robot_err,
                                 is_slippery=False, render_mode="human", true_human_trust=(5, 50),
                                 true_human_capability=0.85,
                                 true_robot_capability=0.85, beta=0.9, c=20, gamma=0.99, seed=SEED,
                                 human_type="epsilon_greedy", round=round_num)

    # Reset the environment to initialize everything correctly
    env.reset(round_num=round_num)
    init_world_state = env.world_state

    detection_num = 0
    step = 0

    for t in range(len(rollout)):
        curr_human_action = rollout[t]['human_action']
        curr_robot_action = rollout[t]['robot_action']

        is_accept, detecting, action = curr_human_action
        if detecting:
            # TODO: Render detection action
            env.render(round_num, None, None, env.world_state)
            time.sleep(0.5)
            detection_num += 1
            step += 1  # one extra step penalty for using detection

        step += 1

        # Account for human action transition (not visualized)...
        env.world_state = env.world_state_transition(env.world_state, None, curr_human_action)
        env.world_state = env.world_state_transition(env.world_state, curr_robot_action, None)
        # Visualize after robot action
        env.render(round_num=round_num, human_action=curr_human_action, robot_action=curr_robot_action, world_state=env.world_state)
        if curr_robot_action[0] > 0:
            time.sleep(1.5)
            # input("Press enter to continue")
        time.sleep(0.5)