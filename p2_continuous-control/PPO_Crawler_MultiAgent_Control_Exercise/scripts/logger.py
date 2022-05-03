import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import torch
import shutil
import os
from CustomSummaryWriter import CustomSummaryWriter
from params import Params
from model import PPO_ActorCritic


class Logger():
    """Generic Logger class to store stats for printing & Tensorboard Visualization """

    def __init__(self, params=Params(), tb=CustomSummaryWriter(), agent_ns=""):

        self.params = params
        self.agent_ns = agent_ns    # Agent's Namespace
        self.scores_list = []
        self.actor_loss_list = []
        self.critic_loss_list = []
        self.entropy_loss_list = []
        self.scores_deque = deque(maxlen=100)
        self.actor_loss_deque = deque(maxlen=100)
        self.critic_loss_deque = deque(maxlen=100)
        self.hparam_dict = params.get_hparam_dict()
        self.t = 0
        self.tb = tb

        # Output init messages
        if self.params.verbose:
            self.params.print_init_messages(self.agent_ns)

        # torch.autograd.set_detect_anomaly(True)         


    def initialize(self, agent, state_size, action_size):
        """ Initializes agent within logger class."""

        self.agent = agent
        if not self.params.restart_training:
            agent.actor_net.load_state_dict(torch.load("{}/{}".format(self.params.checkpoint_actor_weights_dir, self.params.actor_weights_filename_to_resume)))
            agent.critic_net.load_state_dict(torch.load("{}/{}".format(self.params.checkpoint_critic_weights_dir, self.params.critic_weights_filename_to_resume)))
        else:
            self.clear_weights()

        # # Initialize network wrapper for model visualization on TensorBoard
        # ppo_ac_net = PPO_ActorCritic(state_size, action_size, self.params).to(self.params.device)
        # ppo_ac_net.eval()
        # with torch.no_grad():
        #     self.tb.add_graph(ppo_ac_net, torch.zeros(state_size).unsqueeze(0).to(self.params.device))
        # ppo_ac_net.train()

    def log_score(self, score):
        self.scores_deque.append(score)

    def log_stats(self, episode, actor_loss, critic_loss, entropy_loss):
        """ Log stats onto Tensorboard on every interations """

        #self.scores_deque.append(score)
        self.actor_loss_deque.append(actor_loss)
        self.critic_loss_deque.append(critic_loss)
        self.scores_list.append(np.nanmean(self.scores_deque))
        self.actor_loss_list.append(actor_loss)
        self.critic_loss_list.append(critic_loss)
        self.entropy_loss_list.append(entropy_loss)

        # Tensorboard Logging
        self.tb.add_scalar(f"{self.agent_ns}/Reward", np.nanmean(self.scores_deque), episode)
        self.tb.add_scalar(f"{self.agent_ns}/Actor Loss", actor_loss, episode)
        self.tb.add_scalar(f"{self.agent_ns}/Critic Loss", critic_loss, episode)
        self.tb.add_scalar(f"{self.agent_ns}/Entropy Loss", entropy_loss, episode)


    def log_weights(self, episode):
        """ Log weights on Tensorboard every params.log_weights_every iters   """

        self.t = (self.t + 1) % self.params.log_weights_every
        if self.agent.actor_net is not None and self.t == 0:
            for name, weight in self.agent.actor_net.named_parameters():
                self.tb.add_histogram(f'{self.agent_ns}/Actor/'+name, weight, episode)
                self.tb.add_histogram(f'{self.agent_ns}/Actor/{name}.grad',weight.grad, episode)

        if self.agent.critic_net is not None and self.t == 0:        
            for name, weight in self.agent.critic_net.named_parameters():
                self.tb.add_histogram(f'{self.agent_ns}/Critic/'+name, weight, episode)
                self.tb.add_histogram(f'{self.agent_ns}/Critic/{name}.grad',weight.grad, episode)


    def log_overall_perf_tb(self):
        """ Log overall performance of training cycle """

        self.tb.add_hparams(self.hparam_dict,
            {
                f"{self.agent_ns}/Reward": np.mean(self.scores_deque),
                f"{self.agent_ns}/Actor Loss": np.mean(self.actor_loss_deque),
                f"{self.agent_ns}/Critic Loss": np.mean(self.critic_loss_deque),
            },
        )
        self.tb.close()


    def print_weights(self):
        print("\n====== ACTOR WEIGHTS ===== \n")
        for name, weight in self.agent.actor_net.named_parameters():
            print(f'{self.agent_ns}/Actor/'+name, weight)
        print("\n====== CRITIC WEIGHTS ===== \n")
        for name, weight in self.agent.critic_net.named_parameters():
            print(f'{self.agent_ns}/Critic/'+name, weight)       


    def plot_stats(self):
        """ Plots stats recorded """

        print("=====", self.agent_ns, "=====")
        _, axs = plt.subplots(1, 3, figsize=(20, 5))

        # # Scores
        # axs[0].plot(np.arange(1, len(self.scores_list)+1), self.scores_list)
        # axs[0].set(xlabel='Episode #', ylabel='Score')
        # axs[0].set_title(f'{self.agent_ns}/Rewards')
        
        # Actor Loss
        axs[0].plot(np.arange(1, len(self.actor_loss_list)+1), self.actor_loss_list)
        axs[0].set(xlabel='Episode #', ylabel='Loss')
        axs[0].set_title(f'{self.agent_ns}/Actor Loss')
    
        # Critic Loss
        axs[1].plot(np.arange(1, len(self.critic_loss_list)+1), self.critic_loss_list)
        axs[1].set(xlabel='Episode #', ylabel='Loss')
        axs[1].set_title(f'{self.agent_ns}/Critic Loss')

        # Entropy Loss
        axs[2].plot(np.arange(1, len(self.entropy_loss_list)+1), self.entropy_loss_list)
        axs[2].set(xlabel='Episode #', ylabel='Loss')
        axs[2].set_title(f'{self.agent_ns}/Entropy Loss')
        plt.show()

    def clear_weights(self):
        if os.path.exists(self.params.checkpoint_actor_weights_dir):
            shutil.rmtree(self.params.checkpoint_actor_weights_dir)
        if os.path.exists(self.params.checkpoint_critic_weights_dir):
            shutil.rmtree(self.params.checkpoint_critic_weights_dir)
        os.makedirs(self.params.checkpoint_actor_weights_dir)
        os.makedirs(self.params.checkpoint_critic_weights_dir)

    def save_weights(self, episode):
        #torch.save(self.agent.ppo_ac_net.state_dict(), "{}/checkpoint_ep{}.pth".format(self.params.checkpoint_actor_weights_dir, episode))
        torch.save(self.agent.ppo_ac_net.actor.state_dict(), "{}/checkpoint_actor_ep{}.pth".format(self.params.checkpoint_actor_weights_dir, episode))
        torch.save(self.agent.ppo_ac_net.critic.state_dict(), "{}/checkpoint_critic_ep{}.pth".format(self.params.checkpoint_critic_weights_dir, episode))


####################################################
# NOTE: Unable to add_graph for multiple graphs.   #
# https://github.com/lanpa/tensorboardX/issues/319 #
####################################################

# self.tb.add_graph(agent.actor_net, torch.zeros(state_size).to(self.params.device))
# self.tb.add_graph(agent.critic_net,
                #   (torch.zeros(state_size).unsqueeze(0).to(self.params.device),
                #   torch.zeros(action_size).unsqueeze(0).to(self.params.device)))