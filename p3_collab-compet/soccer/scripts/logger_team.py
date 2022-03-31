import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import torch
import shutil
import os
from CustomSummaryWriter import CustomSummaryWriter
from params import Params
from model import ActorCriticWrapper

class Logger_Team():
    """Simple Logger class to store stats for printing & Tensorboard Visualization (Custom Team Stats)"""

    def __init__(self, params=Params(), tb=CustomSummaryWriter(), agent_ns="Team"):

        self.params = params
        self.agent_ns = agent_ns    # Agent's Namespace
        self.scores_cum_list = []
        self.scores_deque = deque(maxlen=100)
        self.scores_cum_deque = deque(maxlen=100)
        self.scores_cum_opp_list = []
        self.scores_opp_deque = deque(maxlen=100)
        self.scores_opp_cum_deque = deque(maxlen=100)
        self.wins_deque = deque(maxlen=100)
        self.draws_deque = deque(maxlen=100)
        self.lose_deque = deque(maxlen=100)
        self.wins_cum_deque = deque(maxlen=100)    # Cumulative
        self.draws_cum_deque = deque(maxlen=100)   # Cumulative
        self.lose_cum_deque = deque(maxlen=100)    # Cumulative
        self.hparam_dict = params.get_hparam_dict()
        self.t = 0
        self.tb = tb

        # torch.autograd.set_detect_anomaly(True)         


    def initialize(self, agent, state_size, action_size):
        """ Initializes agent within logger class."""

        self.agent = agent
        if not self.params.restart_training:
            agent.actor_net.load_state_dict(torch.load("{}/{}".format(self.params.checkpoint_actor_weights_dir, self.params.actor_weights_filename_to_resume)))
            agent.critic_net.load_state_dict(torch.load("{}/{}".format(self.params.checkpoint_critic_weights_dir, self.params.critic_weights_filename_to_resume)))
        else:
            self.clear_weights()

        # Initialize network wrapper for model visualization on TensorBoard
        wrapper_net = ActorCriticWrapper(state_size, action_size, self.params)
        self.tb.add_graph(wrapper_net, 
                          (torch.zeros(state_size).unsqueeze(0).to(self.params.device)))

    def log_stats(self, episode, team_score, opponent_score):
        """ Log stats onto Tensorboard on every interations """

        self.scores_deque.append(team_score)
        self.scores_cum_list.append(sum(self.scores_deque))
        self.scores_cum_deque.append(sum(self.scores_deque))
        self.scores_opp_deque.append(opponent_score)
        self.scores_cum_opp_list.append(sum(self.scores_opp_deque))
        self.scores_opp_cum_deque.append(sum(self.scores_opp_deque))

        self.wins_deque.append(int(1 if team_score > opponent_score else 0))
        self.draws_deque.append(int(1 if team_score == opponent_score else 0))
        self.lose_deque.append(int(1 if team_score < opponent_score else 0))
        self.wins_cum_deque.append(np.count_nonzero(self.wins_deque))
        self.draws_cum_deque.append(np.count_nonzero(self.draws_deque))
        self.lose_cum_deque.append(np.count_nonzero(self.lose_deque))

        # Tensorboard Logging
        self.tb.add_scalar(f"{self.agent_ns}/Score", self.scores_cum_deque[-1], episode)
        self.tb.add_scalar(f"Opponent/Score", self.scores_opp_cum_deque[-1], episode)
        self.tb.add_scalar(f"{self.agent_ns}/Wins", self.wins_cum_deque[-1], episode)
        self.tb.add_scalar(f"{self.agent_ns}/Draws", self.draws_cum_deque[-1], episode)
        self.tb.add_scalar(f"{self.agent_ns}/Loss", self.lose_cum_deque[-1], episode)

    def log_overall_perf_tb(self):
        """ Log overall performance of training cycle """

        self.tb.add_hparams(self.hparam_dict,
            {
                f"{self.agent_ns}/Score": self.scores_cum_deque[-1],
                f"Opponent/Score": self.scores_opp_cum_deque[-1],
                f"{self.agent_ns}/Wins": self.wins_cum_deque[-1],
                f"{self.agent_ns}/Draws": self.draws_cum_deque[-1],
                f"{self.agent_ns}/Loss": self.lose_cum_deque[-1],
            },
        )
        self.tb.close()
   

    def plot_stats(self):
        """ Plots stats recorded """

        print("\n=====", self.agent_ns, "=====")
        _, axs = plt.subplots(2, 3, figsize=(30, 10))

        # Team Scores
        axs[0,0].plot(np.arange(1, len(self.scores_cum_list)+1), self.scores_cum_list)
        axs[0,0].set(xlabel='Episode #', ylabel='Score')
        axs[0,0].set_title(f'{self.agent_ns}/Score')

        # Opponent Scores
        axs[0,1].plot(np.arange(1, len(self.scores_cum_opp_list)+1), self.scores_cum_opp_list)
        axs[0,1].set(xlabel='Episode #', ylabel='Score')
        axs[0,1].set_title(f'Opponent/Score')

        # Wins
        axs[0,2].plot(np.arange(1, len(self.wins_cum_deque)+1), self.wins_cum_deque)
        axs[0,2].set(xlabel='Episode #', ylabel='Wins')
        axs[0,2].set_title(f'{self.agent_ns}/Wins [{len(self.wins_cum_deque)} Matches]')

        # Draws
        axs[1,0].plot(np.arange(1, len(self.draws_cum_deque)+1), self.draws_cum_deque)
        axs[1,0].set(xlabel='Episode #', ylabel='Draws')
        axs[1,0].set_title(f'{self.agent_ns}/Draws [{len(self.draws_cum_deque)} Matches]')

        # Lose
        axs[1,1].plot(np.arange(1, len(self.lose_cum_deque)+1), self.lose_cum_deque)
        axs[1,1].set(xlabel='Episode #', ylabel='Lose')
        axs[1,1].set_title(f'{self.agent_ns}/Lose [{len(self.lose_cum_deque)} Matches]')
    
        plt.show()

    def clear_weights(self):
        if os.path.exists(self.params.checkpoint_actor_weights_dir):
            shutil.rmtree(self.params.checkpoint_actor_weights_dir)
        if os.path.exists(self.params.checkpoint_critic_weights_dir):
            shutil.rmtree(self.params.checkpoint_critic_weights_dir)
        os.makedirs(self.params.checkpoint_actor_weights_dir)
        os.makedirs(self.params.checkpoint_critic_weights_dir)

    def save_weights(self, episode):
        torch.save(self.agent.actor_net.state_dict(), "{}/checkpoint_actor_ep{}.pth".format(self.params.checkpoint_actor_weights_dir, episode))
        torch.save(self.agent.critic_net.state_dict(), "{}/checkpoint_critic_ep{}.pth".format(self.params.checkpoint_critic_weights_dir, episode))


####################################################
# NOTE: Unable to add_graph for multiple graphs.   #
# https://github.com/lanpa/tensorboardX/issues/319 #
####################################################

# self.tb.add_graph(agent.actor_net, torch.zeros(state_size).to(self.params.device))
# self.tb.add_graph(agent.critic_net,
                #   (torch.zeros(state_size).unsqueeze(0).to(self.params.device),
                #   torch.zeros(action_size).unsqueeze(0).to(self.params.device)))