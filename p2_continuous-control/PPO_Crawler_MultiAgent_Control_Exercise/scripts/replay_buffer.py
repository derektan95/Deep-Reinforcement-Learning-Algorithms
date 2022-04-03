from collections import deque, namedtuple
from params import Params
import numpy as np
import random
import torch

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, state_size, action_size, seed, params=Params()):
        """Initialize a ReplayBuffer object that stores only the current trajectory.
        Params
        ======
        """
        self.params = params
        self.device = self.params.device
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []  
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "log_prob", "value", "done"])
        self.seed = random.seed(seed)

        # torch.autograd.set_detect_anomaly(True)         

    def add(self, states, actions, rewards, log_probs, values, dones):
        """Add a new experience to memory."""
        e = self.experience(states, actions, rewards, log_probs, values, dones)
        self.memory.append(e)

    def add_last_state(self, last_state):
        """Add a new experience to memory."""
        self.last_state = last_state

    def retrieve_last_state(self):
        return self.last_state

    def clear(self):
       """Deletes all experiences from the replay buffer."""
       self.memory.clear()

    def sample(self):
        """Randomly sample a batch of experiences from memory."""

        states = np.vstack([e.state for e in self.memory if e is not None])
        actions = np.vstack([e.action for e in self.memory if e is not None])
        rewards = np.vstack([e.reward for e in self.memory if e is not None])
        log_probs = np.vstack([e.log_prob for e in self.memory if e is not None])
        values = np.vstack([e.value for e in self.memory if e is not None])
        dones = np.vstack([e.done for e in self.memory if e is not None])

        return (states, actions, rewards, log_probs, values, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)