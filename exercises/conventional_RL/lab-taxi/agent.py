import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        
        self.eps_initial = 1.0   # How likely will agent explore vs exploit (For eps-greedy policy)
        self.episode_count = 1   # To update eps used for policy
        self.gamma = 0.95        # Discount Rate
        self.alpha = 0.02        # How much to update Q-table during policy evaluation
        

    # ACTION selected based on SARSA-Max Policy (Q-Learning)
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        
        #  Becomes increasingly exploitative (greedy) than exploratory: eps = 1.0 -> SMALL
        #eps = self.eps_initial - (self.episode_count / self.episode_total) * (self.eps_initial - self.eps_min)
        eps = (self.eps_initial / self.episode_count)
        
        # Take next action from epsilon-greedy policy
        greedy_action = np.argmax(self.Q[state])   # Q initialized to be 0 using defaultDict
        policy = np.ones(self.nA) * (eps / self.nA)
        policy[greedy_action] = 1 - eps + (eps / self.nA)
        action = np.random.choice(np.arange(self.nA), p=policy)
        
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        # POLICY EVALUATION: Update Q table & other impt vars - GREEDY
        if next_state != None:
            self.Q[state][action] += (self.alpha * (reward + (self.gamma * np.max(self.Q[next_state])) - self.Q[state][action]))
        else:
            self.Q[state][action] += ( self.alpha * (reward + 0 - self.Q[state][action]) )
        
        if done: 
            self.episode_count += 1