import numpy as np
import random
import pickle
from typing import List
from collections import defaultdict
from .base_agent import BaseAgent, AgentList


class DecPOMDPAgent(BaseAgent):
    def __init__(self, agent_id, num_actions, num_histories, lr=0.1, gamma=0.99, epsilon=1.0):
        super().__init__(agent_id, num_actions)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Tabular Q-Table: Map[State_Key -> Array[Actions]]
        self.q_table = defaultdict(lambda: np.zeros((num_histories, num_actions)))

    def _get_key(self, obs):
        return tuple(obs) if not isinstance(obs, tuple) else obs

    # --- ACTING ---
    def act(self, observation):
        if random.random() < self.epsilon:
            return self.act_normally(observation)
        return self.act_greedy(observation)

    def act_greedy(self, observation):
        key = self._get_key(observation)
        q_vals = self.q_table[key]
        max_q = np.max(q_vals)
        ties = np.where(q_vals == max_q)[0]
        return np.random.choice(ties)

    # --- TRAINING DATA STORAGE ---
    def save_transition(self, observation, action, next_observation, reward, done):
        """Buffer data. Does NOT train immediately."""
        self.last_transition = {
            'obs': observation, 'act': action,
            'next_obs': next_observation, 'reward': reward, 'done': done
        }

    # --- DECENTRALIZED TRAINING (IQL) ---
    def train(self):
        """
        Called by standard AgentList.
        Updates Q-table based on LOCAL Q-values only.
        """
        if self.last_transition is None: return

        t = self.last_transition
        key = self._get_key(t['obs'])
        act = t['act']
        current_q = self.q_table[key][act]

        # Independent Target: r + gamma * max Q_local(s')
        if t['done'] or t['next_obs'] is None:
            target = t['reward']
        else:
            next_max = np.max(self.q_table[self._get_key(t['next_obs'])])
            target = t['reward'] + self.gamma * next_max

        # Update
        self.q_table[key][act] += self.lr * (target - current_q)
        self.last_transition = None

    # --- CENTRALIZED TRAINING SUPPORT (VDN) ---
    def update_from_global_error(self, td_error):
        """
        Called by VDN Class.
        Updates Q-table based on GLOBAL TD error passed down.
        """
        if self.last_transition is None: return
        
        t = self.last_transition
        key = self._get_key(t['obs'])
        act = t['act']
        
        # VDN Tabular Update Rule:
        # Q_i(s,a) += alpha * (Target_Global - Q_Total)
        # effectively distributing the global error to the local node.
        self.q_table[key][act] += self.lr * td_error
        
        self.last_transition = None

    # --- HELPERS ---
    def get_q_value(self, observation, action):
        return self.q_table[self._get_key(observation)][action]

    def get_max_q(self, observation):
        return np.max(self.q_table[self._get_key(observation)])

    def decay_epsilon(self, decay, min_e):
        self.epsilon = max(min_e, self.epsilon * decay)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.q_table = defaultdict(lambda: np.zeros(self.num_actions), pickle.load(f))
            
    def reset(self):
        self.last_transition = None

    def act_normally(self, observation):
        key = self._get_key(observation)
        q_vals = self.q_table[key]

        # Softmax Calculation
        # We subtract max(q_vals) for numerical stability (prevents overflow)
        # Temperature (tau) is set to 1.0 here. Lower tau = more greedy.
        tau = 1.0 
        exp_values = np.exp((q_vals - np.max(q_vals)) / tau)
        probabilities = exp_values / np.sum(exp_values)

        # Sample from the distribution
        return np.random.choice(self.num_actions, p=probabilities)
    
    def update_rates(self):
        decay_factor = 0.9995
        min_epsilon = 0.05
        self.epsilon = max(min_epsilon, self.epsilon * decay_factor)



class VDN(AgentList):
    """
    Value Decomposition Network Controller.
    - Execution: Decentralized (Agents act based on local Q).
    - Training: Centralized (VDN calculates Global Q = Sum Q_i).
    """
    def __init__(self, agents: List[BaseAgent]):
        super().__init__(agents)

    def train(self):
        """
        Overrides the default decentralized training.
        Computes global error and distributes it.
        """
        # 1. Verification: Do we have data from all agents?
        for agent in self:
            if agent.last_transition is None:
                return # Can't train if an agent didn't act or buffer missing

        # 2. Compute Q_Total (Current)
        # Q_tot = Sum(Q_i(s_i, a_i))
        q_total_current = 0.0
        
        # Grab shared info from Agent 0
        shared_reward = self[0].last_transition['reward']
        is_done = self[0].last_transition['done']
        gamma = getattr(self[0], 'gamma', 0.99)

        for agent in self:
            t = agent.last_transition
            q_val = agent.get_q_value(t['obs'], t['act'])
            q_total_current += q_val

        # 3. Compute Q_Total (Target)
        # Target = r + gamma * Sum(max Q_i(s_i'))
        target = shared_reward
        
        if not is_done:
            next_q_sum = 0.0
            for agent in self:
                t = agent.last_transition
                if t['next_obs'] is not None:
                    next_q_sum += agent.get_max_q(t['next_obs'])
            
            target += gamma * next_q_sum

        # 4. Global TD Error
        global_error = target - q_total_current

        # 5. Distribute Error (Centralized Update)
        for agent in self:
            agent.update_from_global_error(global_error)

class BeliefMDPAgent(DecPOMDPAgent):
    """
    Belief-based Agent (Finite Horizon).
    State = (Time_Step, Tuple(History)).
    
    In matrix games, the 'Belief' about the state of the game is perfectly 
    encapsulated by the current Step and the History.
    """
    def _get_state_key(self, observation):
        # Observation is the history list.
        # We derive the step number from the length of the history.
        # This allows the agent to learn "At step 0, do X" vs "At step 2, do Y".
        step_idx = len(observation)
        return (step_idx, tuple(observation))