from params import Params
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

class Logger():
    """Simple Logger class to store stats for printing."""

    def __init__(self, params=Params()):

        self.params = params
        self.scores_list = []
        self.actor_loss_list = []
        self.critic_loss_list = []
        self.scores_deque = deque(maxlen=params.print_every)
        self.actor_loss_deque = deque(maxlen=params.print_every)
        self.critic_loss_deque = deque(maxlen=params.print_every)

    def plot_stats(self):

        _, axs = plt.subplots(1, 3, figsize=(20, 5))

        # Scores
        axs[0].plot(np.arange(1, len(self.scores_list)+1), self.scores_list)
        axs[0].set(xlabel='Episode #', ylabel='Score')
        axs[0].set_title('Rewards')
        
        # Actor Loss
        axs[1].plot(np.arange(1, len(self.actor_loss_list)+1), self.actor_loss_list)
        axs[1].set(xlabel='Episode #', ylabel='Loss')
        axs[1].set_title('Actor Loss')
    
        # Critic Loss
        axs[2].plot(np.arange(1, len(self.critic_loss_list)+1), self.critic_loss_list)
        axs[2].set(xlabel='Episode #', ylabel='Loss')
        axs[2].set_title('Critic Loss')
        plt.show()