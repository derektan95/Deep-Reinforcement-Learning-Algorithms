import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
#         super(QNetwork, self).__init__()
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    

    
    
#     def __init__(self, state_size, action_size, seed):
#         """Initialize parameters and build model.
#         Params
#         ======
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             seed (int): Random seed
#         """
# #         super(QNetwork, self).__init__()
#         super().__init__()
#         self.seed = torch.manual_seed(seed)
        
#         "*** YOUR CODE HERE ***"
#         # Define NN model in OrderedDict data struct
#         self.model = nn.Sequential(OrderedDict([
#                 ('fc1', nn.Linear((state_size), state_size)),
#                 ('relu', nn.ReLU()),
#                 ('fc2', nn.Linear((state_size), action_size)),
#               ]))
        
#         # Initialize Weights & Biases
# #         print(self.model[0])
#         nn.init.kaiming_normal_(self.model[0].weight)
#         nn.init.kaiming_normal_(self.model[2].weight)
#         nn.init.constant_(self.model[0].bias, 0)
#         nn.init.constant_(self.model[2].bias, 0)
# #         self.model[0].bias.zeros_()
# #         self.model[2].bias.zeros_()
#         print(self.model)
        

#     def forward(self, state):
#         """Build a network that maps state -> action values."""
#         return self.model(state)
