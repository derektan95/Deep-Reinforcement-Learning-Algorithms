import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, params):
        """Categorical policy.

        It is not possible to have the actor simply output a softmax distribution of action probabilities and then choose an action off a random sampling of those probabilities. 
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
        # torch.autograd.set_detect_anomaly(True)         

        if params.restart_training:
            self.fc1 = layer_init(nn.Linear(state_size, params.hidden_sizes_actor[0]))
            self.fc2 = layer_init(nn.Linear(params.hidden_sizes_actor[0], params.hidden_sizes_actor[1]))
            self.fc3 = layer_init(nn.Linear(params.hidden_sizes_actor[1], action_size))
        else:
            self.fc1 = nn.Linear(state_size, params.hidden_sizes_actor[0])
            self.fc2 = nn.Linear(params.hidden_sizes_actor[0], params.hidden_sizes_actor[1])
            self.fc3 = nn.Linear(params.hidden_sizes_actor[1], action_size)


    def forward(self, state, action=None):
        """Build an actor (policy) network that maps states -> actions."""

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)

        dist = distributions.Categorical(x)

        if action is None:
            action = dist.sample()

        # Squeeze to output (batch_size, 1) instead of (batch_size, batch_size) during training
        log_prob = dist.log_prob(action.squeeze())  

        return action, log_prob, dist.entropy()


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, params):
        """Initialize parameters and build model.
        NOTE: Centralized critic, hence state includes combination of all states from all 4 agents.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            params: Set of essential parameters
        """
        # Needed to inherit functionalities from nn.Module
        # super(Critic, self).__init__()
        super().__init__()    

        self.seed = torch.manual_seed(params.random_seed)

        if params.restart_training:
            self.fc1 = layer_init(nn.Linear(state_size, params.hidden_sizes_critic[0]))
            self.fc2 = layer_init(nn.Linear(params.hidden_sizes_critic[0], params.hidden_sizes_critic[1]))
            self.fc3 = layer_init(nn.Linear(params.hidden_sizes_critic[1], 1))
        else:
            self.fc1 = nn.Linear(state_size, params.hidden_sizes_critic[0])
            self.fc2 = nn.Linear(params.hidden_sizes_critic[0], params.hidden_sizes_critic[1])
            self.fc3 = nn.Linear(params.hidden_sizes_critic[1], 1)            


    def forward(self, state):
        """Build a critic (value) network that predicts V(s)."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ActorCriticWrapper(nn.Module):
    """ A wrapper, purely for visualization of multiple graphs on Tensorboard"""

    def __init__(self, state_size, action_size, params):
        super().__init__()

        # build policy and value functions
        self.actor = Actor(state_size, action_size, params).to(params.device)
        self.critic = Critic(state_size, action_size, params).to(params.device)

    def forward(self, state):

        # Perform a forward pass through all the networks and return the result
        q1 = self.actor(state)
        q2 = self.critic(state)
        return q1, q2