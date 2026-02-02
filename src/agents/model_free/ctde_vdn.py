import numpy as np
import pickle
import random
from collections import defaultdict, deque, namedtuple
from typing import Tuple, List, Dict, Any, Union, Optional

from replaybuffer import EpisodeStep, EpisodicReplayBuffer

from agents.base_agent import ModelFreeAgent, AgentList




class CTDE_VDN_MF_Agent(ModelFreeAgent):
    """
    Decentralized Execution Agent for VDN.
    - Acts using a shared Q-table.
    - Sends experiences to a shared Episodic Buffer.
    """
    def __init__(
        self, 
        num_cards: int, 
        num_actions: int,
        # Shared components passed from the List
        q_table: Dict,
        buffer: EpisodicReplayBuffer,
        epsilon_start: float
    ):
        super().__init__(num_cards, num_actions)
        
        # References to shared objects managed by the List
        self.q_table = q_table 
        self.buffer = buffer   
        
        # Local Epsilon (synced by list during training)
        self.epsilon = epsilon_start

    @property
    def requires_tensor(self) -> bool:
        return False

    def act(self, input_state: Tuple[int], exploit: bool = False) -> int:
        """
        Epsilon-Greedy Action Selection.
        """
        state_key = tuple(input_state)
        
        if exploit or np.random.rand() > self.epsilon:
            # Greedy: Maximize local Q value
            q_values_array = self.q_table[state_key]
            max_val = np.max(q_values_array)
            best_actions = np.flatnonzero(q_values_array == max_val)
            action = np.random.choice(best_actions)
        else:
            # Random exploration
            action = np.random.randint(0, self.num_actions)
        
        return int(action)

    def save_transition(self, observation, action, next_observation, reward, done):
        """
        Records the step. If done, finalizes the episode in the shared buffer.
        """
        # 1. Record the step (We don't need next_obs or intermediate reward for VDN)
        self.buffer.push_step(tuple(observation), action)

        # 2. If episode ended, commit it with the final reward
        if done:
            self.buffer.close_episode(reward)

    def train(self):
        pass # Training is handled centrally by the List class

    def save(self, *args):
        pass # Saving is handled centrally by the List class
    
    def load(self, *args):
        pass

    def reset(self):
        pass


class CTDE_VDN_MF_List(AgentList):
    """
    Centralized Training Controller for VDN.
    - Manages the Shared Q-Table (Parameter Sharing).
    - Manages the Shared Replay Buffer.
    - Performs the Joint Update (Sum of Qs).
    """
    def __init__(
        self, 
        num_cards: int, 
        num_actions: int,
        # Hyperparameters
        lr: float = 0.5,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
        batch_size: int = 32,
        buffer_size: int = 1_000,
        updates_per_train: int = 5,
        *args, **kwargs
    ):
        self.model_based = False
        self.num_actions = num_actions
        
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.updates_per_train = int(updates_per_train)
        
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # 1. Shared Components
        self.q_table = defaultdict(lambda: np.ones(self.num_actions) * 10.0)
        
        self.buffer = EpisodicReplayBuffer(buffer_size)
        
        # 2. Create Agents
        agent_0 = CTDE_VDN_MF_Agent(num_cards, num_actions, self.q_table, self.buffer, epsilon_start)
        agent_1 = CTDE_VDN_MF_Agent(num_cards, num_actions, self.q_table, self.buffer, epsilon_start)
        
        # Initialize List
        super().__init__([agent_0, agent_1])

    def train(self) -> float:
        """
        Centralized Training Loop.
        Samples episodes, sums Q-values, and backpropagates error.
        """
        # 1. Check Buffer
        if len(self.buffer) < self.batch_size:
            return 0.0

        avg_loss = 0.0

        # 2. Training Loop
        for _ in range(self.updates_per_train):
            batch = self.buffer.sample(self.batch_size)
            batch_loss = 0.0
            
            for episode_steps, reward in batch:
                # episode_steps is a list of Step(state, action)
                
                # --- VDN Logic ---
                # Q_tot = Sum(Q_i(s_i, a_i))
                
                current_q_tot = 0.0
                
                # Calculate Sum of Qs
                for step in episode_steps:
                    q_val = self.q_table[step.state][step.action]
                    current_q_tot += q_val
                
                # Target is just the Reward (since gamma=1.0 and it's episodic/terminal)
                target = reward
                
                # TD Error
                td_error = target - current_q_tot
                
                # Update: Distribute error to all steps
                # (Simple Gradient Descent on the sum)
                for step in episode_steps:
                    self.q_table[step.state][step.action] += self.lr * td_error
                
                batch_loss += abs(td_error)
            
            avg_loss += (batch_loss / self.batch_size)
        
        # 3. Update Epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        for agent in self:
            agent.epsilon = self.epsilon
        return avg_loss / self.updates_per_train

    def save(self, filepath: str):
        """
        Saves the Shared Q-Table.
        """
        data = {
            "q_table": dict(self.q_table),
            "epsilon": self.epsilon
        }
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error saving VDN model: {e}")

    def load(self, filepath: str):
        """
        Loads the Shared Q-Table.
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.q_table = defaultdict(lambda: np.zeros(self.num_actions))
            self.q_table.update(data["q_table"])
            self.epsilon = data.get("epsilon", self.epsilon)
            
            # Re-sync agents
            for agent in self:
                agent.q_table = self.q_table
                agent.epsilon = self.epsilon
                
        except Exception as e:
            print(f"Error loading VDN model: {e}")