
from collections import defaultdict
import itertools
import numpy as np
import os
import pickle
from typing import Any, Tuple

from tiny_game import DecPOMP_Rework
from replaybuffer import ReplayBuffer, Transition, EpisodicReplayBuffer, EpisodeStep

from ..base_agent import BaseAgent, ModelFreeAgent


class DTDE_QSarsa_MF_Agent(ModelFreeAgent):
    """
    Independent Model Free Reinforcement Learning Agent
    """

    def __init__(
            self,
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
        super().__init__(num_cards, num_actions, *args, **kwargs)

        self.lr = lr
        self.gamma = gamma
        
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        # Replay Buffer
        self.replay_buffer = EpisodicReplayBuffer(buffer_size)
        
        # Q-Table
        self.q_table = defaultdict(
            lambda: np.ones(self.num_actions) * 10.0
        )
        for card in range(self.num_cards):
            tmp_input = (-1, card)
            self.q_table[tmp_input] = np.ones(self.num_actions) * 10.0
            for action in range(self.num_actions):
                tmp_input = (card, -1, action)
                self.q_table[tmp_input] = np.ones(self.num_actions) * 10.0
    
    def act(self, input_state: Tuple[int], exploit : bool = False) -> int:
        """
        Epsilon-Greedy Action Selection.
        """
        if exploit or np.random.rand() > self.epsilon:
            # Exploit: Choose best action
            q_values  = self.q_table[input_state]
            max_val = np.max(q_values)
            best_actions = np.flatnonzero(q_values == max_val)
            action = np.random.choice(best_actions) # Break Ties Randomly
        else:
            # Explore: Random Action
            action = np.random.randint(0, self.num_actions)

        return int(action)
    
    def save_transition(self, observation, action, next_observation, reward, done):
        """
        Store experience in the Replay Buffer.
        """
        self.replay_buffer.push_step(tuple(observation), action)
        if done:
            self.replay_buffer.close_episode(reward)

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


        # 3. Loop over Batch
        for episode_steps, final_reward in batch:
            G = final_reward
            for step in reversed(episode_steps):
                state = step.state
                action = step.action
                
                current_q = self.q_table[state][action]
                
                # Update Q towards G
                td_error = G - current_q
                self.q_table[state][action] += self.lr * td_error
                
                total_loss += abs(td_error)
                
                # Discount G for the previous step (if any)
                G = G * self.gamma

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return total_loss / self.batch_size
    
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