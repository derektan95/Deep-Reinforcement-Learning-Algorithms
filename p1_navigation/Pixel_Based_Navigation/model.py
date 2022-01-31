import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class QNetwork(nn.Module):
    """QNetwork.
    
    Simple Dense neural network
    to serve as funcction approximator.
    """
    def __init__(
        self, 
        state_size,
        action_size, 
        seed, 
        in_channels=3,  # DEFAULT: 3
        conv1_kernel=3,
        conv1_filters=16,
        conv1_strides=1,
        conv2_kernel=3,
        conv2_filters=32,
        conv2_strides=1,
        conv3_kernel=3,
        conv3_filters=64,
        conv3_strides=1,
        fc1_units=512, 
        fc2_units=512, 
        fc3_units=256
    ):
        # super(QNetwork, self).__init__()
        super().__init__()
        self.seed = seed
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, conv1_filters, kernel_size=conv1_kernel, stride=conv1_strides, padding=1),
            nn.BatchNorm2d(conv1_filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(conv1_filters, conv2_filters, kernel_size=conv2_kernel, stride=conv2_strides, padding=1),
            nn.BatchNorm2d(conv2_filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(conv2_filters * 21 * 21, fc1_units),
            nn.BatchNorm1d(fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.BatchNorm1d(fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, action_size)
        )
        
    def forward(self, x):

        x =  x.squeeze()

        # For single inputs (act)
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)

        x = torch.permute(x, (0, 3, 1, 2))
        return self.network(x)


###################################################################

# class QNetwork(nn.Module):
#     """Actor (Policy) Model."""

#     def __init__(self, state_size, action_size, seed, in_channels=3, channel_1=32, channel_2=64, channel_3=64, linear_1=512):
#         """Initialize parameters and build model.
#         Params
#         ======
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             seed (int): Random seed
#             fc1_units (int): Number of nodes in first hidden layer
#             fc2_units (int): Number of nodes in second hidden layer
#         """
#         # Needed to inherit functionalities from nn.Module
#         # super(QNetwork, self).__init__()
#         super().__init__()

#         # NOTE: Following https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
#         # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0 ...)
#         self.seed = torch.manual_seed(seed)

#         self.network = nn.Sequential(
#             nn.Conv2d(in_channels, channel_1, kernel_size=8, stride=4, padding=0),
#             nn.BatchNorm2d(channel_1),
#             nn.ReLU(),
#             # nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(channel_1, channel_2, kernel_size=4, stride=2, padding=0),
#             nn.BatchNorm2d(channel_2),
#             nn.ReLU(),
#             # nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(channel_2, channel_3, kernel_size=3, stride=1, padding=0),
#             nn.BatchNorm2d(channel_3),
#             nn.ReLU(),
#             # nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Flatten(),
#             nn.Linear(3136, linear_1),
#             nn.BatchNorm1d(linear_1),
#             nn.ReLU(),
#             nn.Linear(linear_1, action_size),
#             # nn.BatchNorm1d(action_size),
#             # nn.ReLU(),
#             # nn.Linear(fc2_units, action_size)
#         )


#         # self.conv1 = nn.Conv2d(state_size[3], channel_1, kernel_size=8, stride=4, padding=0)
#         # self.batchnorm1 = nn.BatchNorm2d(channel_1)
#         # self.conv2 = nn.Conv2d(channel_1, channel_2, kernel_size=4, stride=2, padding=0) 
#         # self.batchnorm2 = nn.BatchNorm2d(channel_2)
#         # self.conv3 = nn.Conv2d(channel_2, channel_3, kernel_size=3, stride=1, padding=0) 
#         # self.batchnorm3 = nn.BatchNorm2d(channel_3)
#         # self.fc1 = nn.Linear(5184, 512)
#         # self.bn1 = nn.BatchNorm1d(fc1_units)
#         # self.fc2 = nn.Linear(512, action_size)

#     def forward(self, state):

#         # PERMUTE DIMs: (N, H, W, C) --> (N, C, H, W)
#         # NOTE: Some inputs are 4D, some are 3D (I.e. from Learn method)
#         state = torch.unsqueeze(state[0].squeeze(), 0)
#         state = torch.permute(state, (0, 3, 1, 2))
#         return self.network(state)

#         # """Build a network that maps state -> action values."""
#         # conv1_relu_out = F.relu(self.conv1(state))
#         # conv2_relu_out = F.relu(self.conv2(conv1_relu_out))
#         # conv3_relu_out = F.relu(self.conv3(conv2_relu_out))
#         # linear1_out = self.fc1(conv2_relu_out.flatten(1, -1))
#         # return self.fc2(linear1_out)