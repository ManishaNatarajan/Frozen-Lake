import gym
import d4rl_atari
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DiscreteCQL
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sklearn.model_selection import train_test_split


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env = gym.make('breakout-expert-v0')
    data = env.get_dataset()

    # setup MDPDataset
    dataset = MDPDataset(data['observations'],
                         data['actions'],
                         data['rewards'],
                         data['terminals'],
                         discrete_action=True)  # this flag is necessary!

    # this can be replaced with the builtin helper function as follows but does not work for me!!!:
    # from d3rlpy.datasets import get_atari
    # dataset, env = get_atari('breakout-expert-v0')

    # setup CQL algorithm (discrete version)
    cql = DiscreteCQL(n_frames=1, scaler='pixel', use_gpu=True)

    # split train and test episodes
    train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

    # start training
    cql.fit(train_episodes,
            eval_episodes=test_episodes,
            n_epochs=10,
            scorers={
                'environment': evaluate_on_environment(env),  # use d4rl-atari environment
                'advantage': discounted_sum_of_advantage_scorer,  # smaller is better
                'td_error': td_error_scorer,  # smaller is better
                'value_scale': average_value_estimation_scorer  # smaller is better
            })
