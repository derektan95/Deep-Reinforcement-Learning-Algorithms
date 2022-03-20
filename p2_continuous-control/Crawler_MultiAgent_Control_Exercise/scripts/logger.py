import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import torch
import shutil
import os
from CustomSummaryWriter import CustomSummaryWriter
from params import Params
from model import ActorCriticWrapper

class Logger():
    """Simple Logger class to store stats for printing & Tensorboard Visualization """

    def __init__(self, params=Params()):

        self.params = params
        self.scores_list = []
        self.actor_loss_list = []
        self.critic_loss_list = []
        self.scores_deque = deque(maxlen=params.print_every)
        self.actor_loss_deque = deque(maxlen=params.print_every)
        self.critic_loss_deque = deque(maxlen=params.print_every)
        self.hparam_dict = params.get_hparam_dict()
        self.t = 0
        self.tb = CustomSummaryWriter() 

        # torch.autograd.set_detect_anomaly(True)         


    def initialize(self, agent, state_size, action_size):
        """ Initializes agent within logger class."""

        self.agent = agent
        if not self.params.restart_training:
            agent.actor_local.load_state_dict(torch.load("{}/{}".format(self.params.checkpoint_actor_weights_dir, self.params.actor_weights_filename_to_resume)))
            agent.critic_local.load_state_dict(torch.load("{}/{}".format(self.params.checkpoint_critic_weights_dir, self.params.critic_weights_filename_to_resume)))
            agent.actor_target.load_state_dict(torch.load("{}/{}".format(self.params.checkpoint_actor_weights_dir, self.params.actor_weights_filename_to_resume)))
            agent.critic_target.load_state_dict(torch.load("{}/{}".format(self.params.checkpoint_critic_weights_dir, self.params.critic_weights_filename_to_resume)))
        else:
            self.clear_weights()

        # Initialize network wrapper for model visualization on TensorBoard
        wrapper_net = ActorCriticWrapper(state_size, action_size, self.params)
        self.tb.add_graph(wrapper_net, 
                          (torch.zeros(state_size).unsqueeze(0).to(self.params.device), 
                          torch.zeros(action_size).unsqueeze(0).to(self.params.device)))

    def log_stats(self, episode, score, actor_loss, critic_loss):
        """ Log stats onto Tensorboard on every interations """

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

        # Track weights on Tensorboard every params.log_weights_every iters
        self.t = (self.t + 1) % self.params.log_weights_every
        if self.agent.actor_local is not None and self.t == 0:
            for name, weight in self.agent.actor_local.named_parameters():
                self.tb.add_histogram('Actor/'+name, weight, episode)
                self.tb.add_histogram(f'Actor/{name}.grad',weight.grad, episode)

        if self.agent.critic_local is not None and self.t == 0:        
            for name, weight in self.agent.critic_local.named_parameters():
                self.tb.add_histogram('Critic/'+name, weight, episode)
                self.tb.add_histogram(f'Critic/{name}.grad',weight.grad, episode)   

        if self.agent is not None and self.t == 0:   
            self.tb.add_histogram('Predicted Prob Distribution', self.agent.predicted_probs, episode)
            self.tb.add_histogram('Target Prob Distribution', self.agent.projected_target_probs, episode)

    def log_overall_perf_tb(self):
        """ Log overall performance of training cycle """

        self.tb.add_hparams(self.hparam_dict,
            {
                "Reward": np.mean(self.scores_deque),
                "Actor Loss": np.mean(self.actor_loss_deque),
                "Critic Loss": np.mean(self.critic_loss_deque),
            },
        )
        self.tb.close()


    def print_weights(self):
        print("\n====== ACTOR WEIGHTS ===== \n")
        for name, weight in self.agent.actor_local.named_parameters():
            print('Actor/'+name, weight)
        print("\n====== CRITIC WEIGHTS ===== \n")
        for name, weight in self.agent.critic_local.named_parameters():
            print('Critic/'+name, weight)       


    def plot_stats(self):
        """ Plots stats recorded """

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

    def plot_categorical_probs(self):

        _, axs = plt.subplots(1, 2, figsize=(20, 5))

        # Predicted Probs`
        axs[0].plot(np.arange(1, len(self.agent.predicted_probs)+1), self.agent.predicted_probs);
        axs[0].set(xlabel='Atom #', ylabel='Predicted Probs')
        axs[0].set_title('Predicted Q Probs')
        
        # Target Projected Probs
        axs[1].plot(np.arange(1, len(self.agent.projected_target_probs)+1), self.agent.projected_target_probs);
        axs[1].set(xlabel='Atom #', ylabel='Target Probs')
        axs[1].set_title('Projected Target Q Probs')
        plt.show()

    def clear_weights(self):
        if os.path.exists(self.params.checkpoint_actor_weights_dir):
            shutil.rmtree(self.params.checkpoint_actor_weights_dir)
        if os.path.exists(self.params.checkpoint_critic_weights_dir):
            shutil.rmtree(self.params.checkpoint_critic_weights_dir)
        os.makedirs(self.params.checkpoint_actor_weights_dir)
        os.makedirs(self.params.checkpoint_critic_weights_dir)

    def save_weights(self, episode):
        torch.save(self.agent.actor_local.state_dict(), "{}/checkpoint_actor_ep{}.pth".format(self.params.checkpoint_actor_weights_dir, episode))
        torch.save(self.agent.critic_local.state_dict(), "{}/checkpoint_critic_ep{}.pth".format(self.params.checkpoint_critic_weights_dir, episode))


####################################################
# NOTE: Unable to add_graph for multiple graphs.   #
# https://github.com/lanpa/tensorboardX/issues/319 #
####################################################

# self.tb.add_graph(agent.actor_local, torch.zeros(state_size).to(self.params.device))
# self.tb.add_graph(agent.critic_local,
                #   (torch.zeros(state_size).unsqueeze(0).to(self.params.device),
                #   torch.zeros(action_size).unsqueeze(0).to(self.params.device)))