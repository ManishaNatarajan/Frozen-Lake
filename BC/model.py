import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.categorical import Categorical


class BCModel(nn.Module):
    def __init__(self, obs_shape, robot_action_shape, human_action_shape,
                 conv_hidden=32, action_hidden=32, num_layers=1, use_actions=True):
        super().__init__()
        self.obs_shape = obs_shape
        self.act_shape = human_action_shape

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n, m = obs_shape[0], obs_shape[1]
        self.conv_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        self.num_layers = num_layers
        self.use_actions = use_actions
        self.conv_hidden = conv_hidden
        self.action_hidden = action_hidden

        self.conv_lstm = nn.LSTM(input_size=self.conv_embedding_size,  hidden_size=conv_hidden,
                                 num_layers=self.num_layers, batch_first=True)

        self.action_lstm = nn.LSTM(input_size=robot_action_shape+human_action_shape, hidden_size=action_hidden,
                                   num_layers=self.num_layers, batch_first=True)

        if use_actions:
            # Whether to use robot-human action history or not as part of the input
            final_embedding_size = conv_hidden + action_hidden
        else:
            final_embedding_size = conv_hidden

        self.fc = nn.Sequential(
            nn.Linear(final_embedding_size, 32),
            nn.Tanh(),
            nn.Linear(32, self.act_shape))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, conv_hidden_state=None, action_hidden_state=None):
        """
        Forward Pass
        :param x: [batch, Image, t], where Image is 3D (8 x 8 x 3)
        :param conv_hidden_state:
        :param action_hidden_state:
        :return:
        """

        if self.use_actions:
            obs, acts = x
        else:
            obs = x
            acts = None

        batch_size = obs.size(0)
        img_embeds = []
        assert len(obs.shape) == 5  # [batch size, seq len, grid_info, grid_x, grid_y]

        history = obs.shape[1]
        if conv_hidden_state is None:
            conv_h0 = torch.zeros(self.num_layers, batch_size, self.conv_hidden).requires_grad_().to(self.device)
            conv_c0 = torch.zeros(self.num_layers, batch_size, self.conv_hidden).requires_grad_().to(self.device)
        else:
            conv_h0, conv_c0 = conv_hidden_state

        # Perform conv + LSTM to get the obs embedding
        for t in range(history):
            img_embeds = self.conv(obs[:, t, :, :, :])
            img_embeds = img_embeds.reshape(img_embeds.shape[0], -1)
            img_embeds = img_embeds.unsqueeze(dim=1)
            _, (conv_h0, conv_c0) = self.conv_lstm(img_embeds, (conv_h0, conv_c0))

        if self.use_actions:
            assert acts is not None

            if action_hidden_state is None:
                act_h0 = torch.zeros(self.num_layers, batch_size, self.action_hidden).requires_grad_().to(self.device)
                act_c0 = torch.zeros(self.num_layers, batch_size, self.action_hidden).requires_grad_().to(self.device)
            else:
                act_h0, act_c0 = action_hidden_state
            _, (hn, cn) = self.action_lstm(acts, (act_h0, act_c0))

            final_embed = torch.cat((conv_h0, hn), dim=-1)

        else:
            final_embed = conv_h0
        final_embed = final_embed.squeeze()  # TODO: Fix error if there's only sample in the batch and we squeeze
        out = self.fc(final_embed)
        dist = F.softmax(out, dim=-1)
        return dist

    def compute_loss(self, x, y):
        predictions = self.forward(x)
        loss = F.cross_entropy(input=predictions, target=y)
        return loss

    def get_predictions(self, x):
        predictions = self.forward(x)
        return predictions
