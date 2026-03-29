# agents/model_free/dtde_qlarning.py
from collections import defaultdict, namedtuple
import numpy as np
import os
import pickle
from typing import Any


from tiny_game import DecPOMDP, Game, MyHanabi
from replaybuffer import ReplayBuffer, Transition # Using standard ReplayBuffer

from ..base_agent import BaseAgent, ModelFreeAgent


class DTDE_QLearning_MF_Agent(ModelFreeAgent):
    """
    Independent Model Free Reinforcement Learning Agent
    """

    def __init__(
            self,
            env : Game,
            num_cards : int,
            num_actions : int,
            # Hyperparameters
            lr: float = 0.1,
            gamma: float = 0.9,
            epsilon_start: float = 1.0,
            epsilon_min: float = 0.05,
            epsilon_decay: float = 0.9995,
            batch_size: int = 32,
            buffer_size: int = 10_000,

            *args, **kwargs):
        super().__init__(env, num_cards, num_actions, *args, **kwargs)

        self.lr = lr
        
        self.gamma = gamma
        
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Q-Table
        self.q_table = {}
        #self.q_table = defaultdict(
        #    lambda: np.ones(self.num_actions) * 10.0
        #)
        return
    
    def get_legal_actions_mask(self, obs: tuple[int]) -> np.ndarray:
        """Helper to get legal actions mask for an observation."""
        if isinstance(self.env, DecPOMDP):
            return np.ones(self.num_actions, dtype=bool)
        else: # MyHanabi
            mask, _ = self.env.num_legal_actions(full_history=obs)
            return np.array(mask, dtype=bool)

    def act(self, input_state: tuple[int], exploit=False) -> int:
        # Get legal actions
        if isinstance(self.env, DecPOMDP):
            legal_actions = np.arange(self.num_actions)
            legal_action_mask = np.ones(self.num_actions, dtype=bool)
        else:
            legal_action_mask, legal_actions = self.env.num_legal_actions()
            legal_action_mask = np.array(legal_action_mask, dtype=bool)

        # Lazy init
        if input_state not in self.q_table:
            q_values = np.zeros(self.num_actions)
            q_values[legal_action_mask] = 10.0

            self.q_table[input_state] = q_values


        # Exploit
        if exploit or np.random.rand() > self.epsilon:
            q_values = self.q_table[input_state].copy()
            q_values[~legal_action_mask] = -np.inf
            max_val = np.max(q_values)
            best_actions = np.flatnonzero(q_values == max_val)

            action = np.random.choice(best_actions)
        else:
            action = np.random.choice(legal_actions)
        return int(action)
    
    def save_transition(self, observation, action, next_observation, reward, done):
        """
        Store experience in the Replay Buffer.
        """
        self.replay_buffer.push(
            tuple(observation), 
            action, 
            reward, 
            tuple(next_observation), 
            done
        )

    def train(self)->float:
        """
        Samples a batch from memory and performs Q-Learning updates.
        Returns the average loss (temporal difference error).
        """
        # 1. Check Buffer Size
        if len(self.replay_buffer) < self.batch_size:
            return 0.0  # Not enough samples to train
        
        total_loss = 0.0
        batch = self.replay_buffer.sample(self.batch_size)


        # 2. Loop over Batch
        for transition in batch:
            state, action, reward, next_state, done = transition

            # Lazy init Q-values for current state if not present
            if state not in self.q_table:
                self.q_table[state] = np.zeros(self.num_actions, dtype=np.float32)
                self.q_table[state][self.get_legal_actions_mask(state)] = 10.0

            current_q = self.q_table[state][action]
            
            # Calculate target Q-value
            target_q = reward
            if not done:
                # Lazy init Q-values for next state if not present
                if next_state not in self.q_table:
                    self.q_table[next_state] = np.zeros(self.num_actions, dtype=np.float32)
                    self.q_table[next_state][self.get_legal_actions_mask(next_state)] = 10.0

                # Max Q-value of next state (for Q-Learning)
                next_q_values = self.q_table[next_state].copy()
                legal_next_actions_mask = self.get_legal_actions_mask(next_state)
                next_q_values[~legal_next_actions_mask] = -np.inf # Mask illegal next actions
                max_next_q = np.max(next_q_values)
                
                target_q += self.gamma * max_next_q

            # Q-Learning update
            td_error = target_q - current_q
            self.q_table[state][action] += self.lr * td_error
            
            total_loss += abs(td_error)

        # Epsilon decay
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return total_loss / self.batch_size if self.batch_size > 0 else 0.0
    
    def save(self, save_path: str):
        """
        Save the Q-Table parameters to a file.
        """
        data = {
            "q_vals" : dict(self.q_table)
        }
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        return

    def load(self, load_path: str):
        """
        Load the Q-Table parameters from a file.
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(load_path)
        if os.path.getsize(load_path) == 0:
            raise ValueError("Q-table file is empty")
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)

        
        # Verify data integrity
        if not isinstance(data, dict):
            raise ValueError("Loaded file does not contain a valid dictionary.")
        if len(data.keys()) == 0:
            raise ValueError("Empty File")
        
        # Reconstruct defaultdict
        self.q_table.update(data['q_vals'])
        return