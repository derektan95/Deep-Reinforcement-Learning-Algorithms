from collections import deque, namedtuple
from params import Params
import numpy as np
import random
import torch

class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    NOTE: Training data is collected across multiple episodes to ensure learning stability. 
    Tested with learning within each episode, but training is always unstable (max +1000 rewards before dipping)
    """

    def __init__(self, state_size, action_size, seed, params=Params()):
        """Initialize a ReplayBuffer object that stores only the current trajectory.
        Params
        ======
        """
        self.params = params
        self.device = self.params.device
        self.state_size = state_size
        self.action_size = action_size
        self.temp_memory = []
        self.memory = []  
        self.intermediate_exp = namedtuple("Intermediate_Experience", field_names=["state", "action", "reward", "log_prob", "value", "done"])
        self.exp = namedtuple("Experience", field_names=["state", "action", "reward_future", "log_prob", "advantage_normalized"])
        self.seed = random.seed(seed)


    #### TEMP MEMORY ####        

    def add_temp_memory(self, states, actions, rewards, log_probs, values, dones):
        """Add a new experience to memory."""
        e = self.intermediate_exp(states, actions, rewards, log_probs, values, dones)
        self.temp_memory.append(e)

    def retrieve_temp_memory(self):
        """Randomly sample a batch of experiences from memory."""

        # Stack to get (num_exp, num_robot, exp_shape)
        states = np.stack([e.state for e in self.temp_memory if e is not None])
        actions = np.stack([e.action for e in self.temp_memory if e is not None])
        rewards = np.stack([e.reward for e in self.temp_memory if e is not None])
        log_probs = np.stack([e.log_prob for e in self.temp_memory if e is not None])
        values = np.stack([e.value for e in self.temp_memory if e is not None])
        dones = np.stack([e.done for e in self.temp_memory if e is not None])        

        return (states, actions, rewards, log_probs, values, dones)

    def clear_temp_memory(self):
       """Deletes all temp experiences from the replay buffer."""
       self.temp_memory = []

    def add_last_state(self, last_state):
        """Add a new experience to memory."""
        self.last_state = last_state

    def retrieve_last_state(self):
        return self.last_state


    #### PERMANENT MEMORY ####        

    def add_memory(self, states, actions, rewards_future, log_probs, advantages_normalized):
        """Add a new experience to memory."""
        e = self.exp(states, actions, rewards_future, log_probs, advantages_normalized)
        self.memory.append(e)

    def retrieve_memory(self):
        """Randomly sample a batch of experiences from memory."""

        # Stack to get (num_exp, num_robot, exp_shape)
        states = torch.vstack([e.state for e in self.memory if e is not None])
        actions = torch.vstack([e.action for e in self.memory if e is not None])
        rewards_future = torch.vstack([e.reward_future for e in self.memory if e is not None])
        log_probs = torch.vstack([e.log_prob for e in self.memory if e is not None])
        advantages_normalized = torch.vstack([e.advantage_normalized for e in self.memory if e is not None])

        return (states, actions, rewards_future, log_probs, advantages_normalized)

    def clear_memory(self):
       """Deletes all experiences from the replay buffer."""
       self.memory = []


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)