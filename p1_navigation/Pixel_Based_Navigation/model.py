import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, channel_1=4, channel_2=4):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        # Needed to inherit functionalities from nn.Module
        # super(QNetwork, self).__init__()
        super().__init__()

        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0 ...)
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(state_size[3], channel_1, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(channel_1, channel_2, 3, stride=1, padding=1) 
        self.fc1 = nn.Linear(channel_2 * state_size[1] * state_size[2], action_size)


    def forward(self, state):

        # PERMUTE DIMs: (N, H, W, C) --> (N, C, H, W)
        # NOTE: Some inputs are 4D, some are 3D (I.e. from Learn method)
        state = torch.unsqueeze(state[0].squeeze(), 0)  
        state = torch.permute(state, (0, 3, 1, 2))

        """Build a network that maps state -> action values."""
        conv1_relu_out = F.relu(self.conv1(state))
        conv2_relu_out = F.relu(self.conv2(conv1_relu_out))
        return self.fc1(conv2_relu_out.flatten(1, -1))
