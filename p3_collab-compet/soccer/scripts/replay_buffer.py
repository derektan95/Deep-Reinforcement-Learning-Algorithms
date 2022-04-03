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
        self.experience = namedtuple("Experience", field_names=["state", "all_state", "action", "reward", "log_prob", "done"])
        self.seed = random.seed(seed)

        # torch.autograd.set_detect_anomaly(True)         

    def add(self, states, all_states, actions, rewards, log_probs, dones):
        """Add a new experience to memory."""
        e = self.experience(states, all_states, actions, rewards, log_probs, dones)
        self.memory.append(e)

    def add_last_all_state(self, last_all_state):
        """Add a new experience to memory."""
        self.last_all_state = last_all_state

    def retrieve_last_all_state(self):
        return self.last_all_state

    def clear(self):
       """Deletes all experiences from the replay buffer."""
       self.memory.clear()

    def sample(self):
        """Randomly sample a batch of experiences from memory."""

        states = np.vstack([e.state for e in self.memory if e is not None])
        all_states = np.vstack([e.all_state for e in self.memory if e is not None])
        actions = np.vstack([e.action for e in self.memory if e is not None])
        rewards = np.vstack([e.reward for e in self.memory if e is not None])
        log_probs = np.vstack([e.log_prob for e in self.memory if e is not None])
        dones = np.vstack([e.done for e in self.memory if e is not None])

        return (states, all_states, actions, rewards, log_probs, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)