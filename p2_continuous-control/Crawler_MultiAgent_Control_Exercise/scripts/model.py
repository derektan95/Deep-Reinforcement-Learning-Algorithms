import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, params):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_sizes (tuple of ints): Number of nodes in the hidden layers
        """
        # Needed to inherit functionalities from nn.Module
        # super(Actor, self).__init__()
        super().__init__()    
        
        self.seed = torch.manual_seed(params.random_seed)
        self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, params.hidden_sizes_actor[0])
        self.bn1 = nn.BatchNorm1d(params.hidden_sizes_actor[0])
        self.fc2 = nn.Linear(params.hidden_sizes_actor[0], params.hidden_sizes_actor[1])
        self.bn2 = nn.BatchNorm1d(params.hidden_sizes_actor[1])
        self.fc3 = nn.Linear(params.hidden_sizes_actor[1], params.hidden_sizes_actor[2])
        self.bn3 = nn.BatchNorm1d(params.hidden_sizes_actor[2])
        self.fc4 = nn.Linear(params.hidden_sizes_actor[2], action_size)
        if params.restart_training:
            self.reset_parameters()

        # torch.autograd.set_detect_anomaly(True)         


    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""

        # Add 1 dimension @dim=0 for batchnorm to work properly
        if state.dim() == 1:
            state = state.unsqueeze(0)

        x = self.bn0(state)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = torch.tanh(self.fc4(x))    # F.tanh is deperecated
        return x.squeeze()             # Remove extra dimensions to output action list


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, params):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_sizes (tuple of ints): Number of nodes in the hidden layers
        """
        # Needed to inherit functionalities from nn.Module
        # super(Critic, self).__init__()
        super().__init__()    
        
        self.seed = torch.manual_seed(params.random_seed)
        self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, params.hidden_sizes_critic[0])
        self.bn1 = nn.BatchNorm1d(params.hidden_sizes_critic[0])
        self.fc2 = nn.Linear(params.hidden_sizes_critic[0]+action_size, params.hidden_sizes_critic[1])
        self.fc3 = nn.Linear(params.hidden_sizes_critic[1], params.hidden_sizes_critic[2])
        self.fc4 = nn.Linear(params.hidden_sizes_critic[2], params.num_atoms)
        if params.restart_training:
            self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action, log=False):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = self.bn0(state)
        x = F.relu(self.bn1(self.fc1(x)))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Only calculate the type of softmax needed by the foward call, to save
        # a modest amount of calculation across 1000s of timesteps.
        if log:
            return F.log_softmax(self.fc4(x), dim=-1)
        else:
            return F.softmax(self.fc4(x), dim=-1)

class ActorCriticWrapper(nn.Module):
    """ A wrapper, purely for visualization of multiple graphs on Tensorboard"""

    def __init__(self, state_size, action_size, params):
        super().__init__()

        # build policy and value functions
        self.actor = Actor(state_size, action_size, params).to(params.device)
        self.critic = Critic(state_size, action_size, params).to(params.device)

    def forward(self, state, act):

        # Perform a forward pass through all the networks and return the result
        q1 = self.actor(state)
        q2 = self.critic(state, act)
        return q1, q2

####################################################################################################

# class Actor(nn.Module):
#     """Actor (Policy) Model."""

#     def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
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
#         # super(Actor, self).__init__()
#         super().__init__()    
        
#         self.seed = torch.manual_seed(seed)
#         self.fc1 = nn.Linear(state_size, fc1_units)
#         self.fc2 = nn.Linear(fc1_units, fc2_units)
#         self.fc3 = nn.Linear(fc2_units, action_size)
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
#         self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
#         self.fc3.weight.data.uniform_(-3e-3, 3e-3)

#     def forward(self, state):
#         """Build an actor (policy) network that maps states -> actions."""

#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         return (self.fc3(x)).tanh()    # F.tanh is deperecated


# class Critic(nn.Module):
#     """Critic (Value) Model."""

#     def __init__(self, state_size, action_size, seed, num_atoms, fc1_units=400, fc2_units=300):
#         """Initialize parameters and build model.
#         Params
#         ======
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             seed (int): Random seed
#             fc1_units (int): Number of nodes in the first hidden layer
#             fc2_units (int): Number of nodes in the second hidden layer
#         """
#         # Needed to inherit functionalities from nn.Module
#         # super(Critic, self).__init__()
#         super().__init__()    
        
#         self.seed = torch.manual_seed(seed)
#         self.fc1 = nn.Linear(state_size, fc1_units)
#         self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
#         self.fc3 = nn.Linear(fc2_units, num_atoms)
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
#         self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
#         self.fc3.weight.data.uniform_(-3e-3, 3e-3)

#     def forward(self, state, action, log=False):
#         """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
#         x = F.relu(self.fc1(state))
#         x = torch.cat((x, action), dim=1)
#         x = F.relu(self.fc2(x))
#         logits = self.fc3(x)

#         if log:
#             return F.log_softmax(logits, dim=-1)
#         else:
#             return F.softmax(logits, dim=-1)




#################################################################

# class Actor(nn.Module):
#     """Actor (Policy) Model."""

#     def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300, fc3_units=200):
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
#         # super(Actor, self).__init__()
#         super().__init__()    
        
#         self.seed = torch.manual_seed(seed)
#         self.bn0 = nn.BatchNorm1d(state_size)
#         self.fc1 = nn.Linear(state_size, fc1_units)
#         self.bn1 = nn.BatchNorm1d(fc1_units)
#         self.fc2 = nn.Linear(fc1_units, fc2_units)
#         self.bn2 = nn.BatchNorm1d(fc2_units)
#         self.fc3 = nn.Linear(fc2_units, fc3_units)
#         self.bn3 = nn.BatchNorm1d(fc3_units)
#         self.fc4 = nn.Linear(fc3_units, action_size)
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
#         self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
#         self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
#         self.fc4.weight.data.uniform_(-3e-3, 3e-3)

#     def forward(self, state):
#         """Build an actor (policy) network that maps states -> actions."""

#         # Add 1 dimension @dim=0 for batchnorm to work properly
#         if state.dim() == 1:
#             state = state.unsqueeze(0)

#         x = self.bn0(state)
#         x = F.relu(self.bn1(self.fc1(x)))
#         x = F.relu(self.bn2(self.fc2(x)))
#         x = F.relu(self.bn3(self.fc3(x)))
#         x = torch.tanh(self.fc4(x))    # F.tanh is deperecated
#         return x.squeeze()             # Remove extra dimensions to output action list


# class Critic(nn.Module):
#     """Critic (Value) Model."""

#     def __init__(self, state_size, action_size, seed, num_atoms, fc1_units=400, fc2_units=300, fc3_units=200):
#         """Initialize parameters and build model.
#         Params
#         ======
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             seed (int): Random seed
#             fc1_units (int): Number of nodes in the first hidden layer
#             fc2_units (int): Number of nodes in the second hidden layer
#         """
#         # Needed to inherit functionalities from nn.Module
#         # super(Critic, self).__init__()
#         super().__init__()    
        
#         self.seed = torch.manual_seed(seed)
#         self.bn0 = nn.BatchNorm1d(state_size)
#         self.fc1 = nn.Linear(state_size, fc1_units)
#         self.bn1 = nn.BatchNorm1d(fc1_units)
#         self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
#         self.fc3 = nn.Linear(fc2_units, fc3_units)
#         self.fc4 = nn.Linear(fc3_units, num_atoms)
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
#         self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
#         self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
#         self.fc4.weight.data.uniform_(-3e-3, 3e-3)

#     def forward(self, state, action, log=False):
#         """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
#         x = self.bn0(state)
#         x = F.relu(self.bn1(self.fc1(x)))
#         x = torch.cat((x, action), dim=1)
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))

#         # Only calculate the type of softmax needed by the foward call, to save
#         # a modest amount of calculation across 1000s of timesteps.
#         if log:
#             return F.log_softmax(self.fc4(x), dim=-1)
#         else:
#             return F.softmax(self.fc4(x), dim=-1)