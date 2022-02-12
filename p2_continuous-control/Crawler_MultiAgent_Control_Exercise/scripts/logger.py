from params import Params
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Logger():
    """Simple Logger class to store stats for printing."""

    def __init__(self, params=Params(), tb_comment=''):

        self.params = params
        self.scores_list = []
        self.actor_loss_list = []
        self.critic_loss_list = []
        self.scores_deque = deque(maxlen=params.print_every)
        self.actor_loss_deque = deque(maxlen=params.print_every)
        self.critic_loss_deque = deque(maxlen=params.print_every)
        self.tb = SummaryWriter(comment=tb_comment)          # Tensorboard Logging

    def load_agent(self, agent):
        self.agent = agent
        self.tb.add_graph(agent.actor_local)
        self.tb.add_graph(agent.critic_local)

    def log_stats(episode, score, actor_loss, critic_loss):
        self.scores_deque.append(score)
        self.actor_loss_deque.append(actor_loss)
        self.critic_loss_deque.append(critic_loss)
        self.scores_list.append(score)
        self.actor_loss_list.append(actor_loss)
        self.critic_loss_list.append(critic_loss)

        # Tensorboard Logging
        self.tb.add_scalar("Reward", score, episode)
        self.tb.add_scalar("Actor Loss", actor_loss, episode)
        self.tb.add_scalar("Critic Loss", critic_loss, episode)

        # if self.agent.actor_local is not None:
        #     for name, weight in self.agent.actor_local.named_parameters():
        #         self.tb.add_histogram(name, weight, episode)
        #         self.tb.add_histogram(f'Actor/{name}.grad',weight.grad, episode)

        # if self.agent.critic_local is not None:        
        #     for name, weight in self.agent.critic_local.named_parameters():
        #         self.tb.add_histogram(name, weight, episode)
        #         self.tb.add_histogram(f'Critic/{name}.critic.grad',weight.grad, episode)   

    def terminate(self):
        self.tb.close()

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