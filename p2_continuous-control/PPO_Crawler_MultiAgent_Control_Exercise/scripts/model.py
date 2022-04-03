import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class PPO_ActorCritic(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, params):
        """Categorical policy.

        Neural networks cannot directly backpropagate through random samples. PyTorch and Tensorflow offer a distribution function to solve this that makes the action 
        selection differentiable. The actor passes the softmax output through this distribution function to select the action and then backpropagation can occur.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            params: Set of essential parameters
        """
        # Needed to inherit functionalities from nn.Module
        # super(Actor, self).__init__()
        super().__init__()    
        
        self.seed = torch.manual_seed(params.random_seed)
        self.std = nn.Parameter(torch.ones(1, action_size)*0.15)

        self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, params.hidden_sizes_actor[0])
        self.bn1 = nn.BatchNorm1d(params.hidden_sizes_actor[0])
        self.fc2 = nn.Linear(params.hidden_sizes_actor[0], params.hidden_sizes_actor[1])
        self.bn2 = nn.BatchNorm1d(params.hidden_sizes_actor[1])
        self.fc3a = nn.Linear(params.hidden_sizes_actor[1], action_size)       # Action Head
        self.fc3c = nn.Linear(params.hidden_sizes_actor[1], 1)                 # Critic Head
        
        if params.restart_training:
            self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3a.weight.data.uniform_(*hidden_init(self.fc3a))    ## CONSIDER: LAST LAYER = data.uniform_(-3e-3, 3e-3)  ?
        self.fc3c.weight.data.uniform_(*hidden_init(self.fc3c))    ## CONSIDER: LAST LAYER = data.uniform_(-3e-3, 3e-3)  ?
        #self.fc4.weight.data.uniform_(-3e-3, 3e-3)            

    def forward(self, state, action=None, std_scale=1.0):
        """Build an actor (policy) network that maps states -> actions."""

        x = F.relu(self.fc1(self.bn0(state)))
        x = F.relu(self.fc2(self.bn1(x)))
        intermediate = self.bn2(x)
        values = F.relu(self.fc3v(intermediate))
        actions_mean = torch.tanh(self.fc3a(intermediate))

        # Action distribution --> Normal 
        dist = distributions.Normal(actions_mean, F.hardtanh(self.std, min_val=0.05*std_scale, max_val=0.5*std_scale))
        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(actions_mean)      # NEED TO SUM?
        entropy = dist.entropy()                    # NEED TO SUM?

        return action, log_prob, entropy, values
