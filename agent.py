import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        
        """
            task is an enum with the following values:
            - 0 = navigate the environment in the most effective route to find a passenger
                - (a) first set action-values are based on trying to traverse the environment in the quickest / most effective manner
                - (b) second action-value set is pick-up action-values... if passenger is on block, this must have high reward and be                   greedy option
            - 1 = once a passenger has been picked up, take the passenger to a specfic destination (t)
                - (a) first set action-values are based on trying to traverse the environment in the quickest / most effective manner
                - (d) second action-value set is drop-off action-values... if passenger is on block, this must have high reward and be                   greedy option
        
        self.task = 0
        initial_actions = 4
        self.Q_navigate = defaultdict(lambda: np.zeros(initial_actions))
        self.Q_pickup = defaultdict(lambda: np.zeros(1))
        self.Q_dropoff = defaultdict(lambda: np.zeros(1))
        self.next_state = np.zeros(initial_actions)
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = .1
        self.gamma = 0.9
        self.num_episodes = 0
        
    def update_Q(self,Qsa, Qsa_next, reward, alpha, gamma):
        """ updates the action-value function estimate using the most recent time step """
        return ((1-alpha) * Qsa + alpha * (reward + (gamma * Qsa_next)))

    def epsilon_greedy_probs(self, Q_s, i_episode, eps=None):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        epsilon = 1.0 / i_episode
        if eps is not None:
            epsilon = eps
            
        policy_s = np.ones(self.nA) * epsilon / self.nA
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / self.nA)
        return policy_s

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        self.num_episodes += 1
        # get epsilon-greedy action probabilities
        policy_s = self.epsilon_greedy_probs(self.Q[state], self.num_episodes)
        # pick action A
        action = np.random.choice(np.arange(self.nA), p=policy_s)
        # limit number of time steps per episode
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


        self.Q[state][action] = self.update_Q(self.Q[state][action], np.max(self.Q[next_state]), reward, self.alpha, self.gamma)       