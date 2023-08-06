import torch
import numpy as np
import glob
import pandas as pd
from torch.utils.data import DataLoader
from data_loaders.utils import *


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, sequence_length):
        self.sequence_length = sequence_length  # Number of past frames to stack
        self.folder_path = folder_path
        self.dones = []
        self._load_data(folder_path)

    def _load_data(self, folder_path):
        raise NotImplementedError

    def _produce_input(self, idx):
        raise NotImplementedError

    def _produce_output(self, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        x = self._produce_input(idx)
        y = self._produce_output(idx)
        return x, y

    def __len__(self):
        return len(self.dones)


class BCDataset(BaseDataset):
    def __init__(self, folder_path, sequence_length, num_human_actions=5, use_actions=False):
        self.process_first_episode = True
        self.use_actions = use_actions
        self.num_human_actions = num_human_actions
        self.num_robot_actions = 5

        super().__init__(folder_path, sequence_length)

    def _load_data(self, folder_path):
        all_files = glob.glob(self.folder_path + "*.csv")
        # Load all map 5 rollouts of every user in the folder path
        all_episode_data = []
        for file in all_files:
            print(file)
            all_episode_data.extend(extract_episode_data(file))

        # all_episode_data = [extract_episode_data(all_files[0])]

        for episode in all_episode_data:
            agent_obs = np.array(encode_full_state(map_id=episode["map_id"], states=episode["state"],
                                                   human_actions=episode["obs"]))
            # One hot encoding of the actions of both agents
            human_action = np.eye(self.num_human_actions)[episode["obs"]]
            robot_action = np.eye(self.num_robot_actions)[episode["action"]]
            if self.process_first_episode:
                self.process_first_episode = False
                self.agent_obs = agent_obs
                self.human_actions = human_action
                self.robot_actions = robot_action
                self.dones = episode["dones"]
                self.last_states = episode["last_state"]
                self.current_states = episode["state"]
                self.last_actions = episode["last_action"]

            else:
                self.agent_obs = np.concatenate((self.agent_obs, agent_obs), axis=0)
                self.human_actions = np.concatenate((self.human_actions, human_action), axis=0)
                self.robot_actions = np.concatenate((self.robot_actions, robot_action), axis=0)
                self.dones = np.append(self.dones, episode["dones"])
                self.last_states = np.concatenate((self.last_states, episode["last_state"]), axis=0)
                self.current_states = np.concatenate((self.current_states, episode["state"]), axis=0)
                self.last_actions = np.append(self.last_actions, episode["last_action"])

        self.done_idxs = np.where(self.dones == 1)[0]  # Mark the end of an episode

    def process_start_observations(self, np_array, idx, episode_start_idx):
        # Pad observations with zeros at the start of the episode
        last_obs = np_array[idx]
        shape = (self.sequence_length - (idx - episode_start_idx + 1), ) + last_obs.shape
        empty_sequence = np.zeros(shape)
        sequence = np_array[episode_start_idx:idx+1]
        sequence = np.concatenate((empty_sequence, sequence), axis=0)

        return sequence

    def _produce_input(self, idx):
        # Stack observations based on sequence length
        if idx < self.done_idxs[0] + 1:
            episode_start_idx = 0
        else:
            # Get the index of the start of the current episode
            episode_start_idx = self.dones[np.where(self.done_idxs <= idx - 1)[0][-1]] + 1

        if idx - episode_start_idx >= self.sequence_length:
            agent_obs = self.agent_obs[idx - self.sequence_length:idx]
            robot_actions = self.robot_actions[idx - self.sequence_length:idx]
            human_actions = self.human_actions[idx - self.sequence_length:idx]  # TODO: Fix --> Can't feed in the current human action

        else:
            # Pad the observations and actions at the start of the episode
            agent_obs = self.process_start_observations(self.agent_obs, idx, episode_start_idx)
            robot_actions = self.process_start_observations(self.robot_actions, idx, episode_start_idx)
            human_actions = self.process_start_observations(self.human_actions, idx, episode_start_idx)

        # Concatenate the robot and the human action interaction history
        all_actions = np.concatenate((robot_actions, human_actions), axis=1)

        if self.use_actions:
            sample = [agent_obs, all_actions]
        else:
            sample = agent_obs
        return sample

    def _produce_output(self, idx):
        return self.human_actions[idx]
