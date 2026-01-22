
from collections import defaultdict
import itertools
import numpy as np
import pickle
from typing import Any, Tuple

from tiny_game import DecPOMP_Rework
from replaybuffer import ReplayBuffer, Transition

from ..base_agent import BaseAgent, ModelFreeAgent


class Independent_RL_Agent(ModelFreeAgent):
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
            updates_per_train : int = 1,
            buffer_size: int = 10_000,

            *args, **kwargs):
        super().__init__(num_cards, num_actions, *args, **kwargs)

        self.lr = lr
        self.gamma = gamma
        
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Replay Buffer
        self.buffer_size = buffer_size
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.updates_per_train : int = updates_per_train
        
        # Q-Table
        self.q_table = defaultdict(
            lambda: np.ones(self.num_actions) * 10.0
        )
    
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
        self.replay_buffer.push(
            observation,
            action,
            next_observation,
            reward,
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

        for _ in range(self.updates_per_train):
            # 2. Sample Batch
            batch = self.replay_buffer.sample(self.batch_size)
            batch_loss = 0.0

            # 3. Loop over Batch
            for transition in batch:
                # 3.1 Set up Transition Elements
                state = tuple(transition.state)
                action = transition.action
                reward = transition.reward
                done = transition.done
                if transition.next_state is not None:
                    next_state = tuple(transition.next_state)
                else:
                    next_state = None

                # 3.2 Q-Learning Update
                # 3.2.1 Get Current Q
                current_q = self.q_table[state][action]

                # 3.2.2. Calc Target Q
                if done or next_state is None:
                    max_next_q = 0.0
                else:
                    max_next_q = np.max(self.q_table[next_state])
                td_target = reward + self.gamma * max_next_q

                # 3.2.3 Calc Error + update
                td_error = td_target - current_q
                self.q_table[state][action] += self.lr * td_error

                # 3.3 Update loss
                batch_loss += abs(td_error)
            total_loss += (batch_loss / self.batch_size)
        # 4. Update Epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        # Return Loss
        return total_loss / self.updates_per_train
    
    def save(self, save_path: str):
        """
        Save the Q-Table parameters to a file.
        """
        # Convert defaultdict to standard dict for pickling
        model_parameters = dict(self.q_table)
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(model_parameters, f)
        except Exception as e:
            print(f"Error saving model parameters for agent")

    def load(self, load_path: str):
        """
        Load the Q-Table parameters from a file.
        """
        try:
            with open(load_path, 'rb') as f:
                model_parameters = pickle.load(f)
            
            # Verify data integrity
            if not isinstance(model_parameters, dict):
                raise ValueError("Loaded file does not contain a valid dictionary.")
            
            # Reconstruct defaultdict
            self.q_table = defaultdict(lambda: np.zeros(self.num_actions))
            self.q_table.update(model_parameters)
            
        except FileNotFoundError:
            print(f"File not found: {load_path}. keeping existing parameters.")
        except Exception as e:
            print(f"Error loading model parameters for agent: {e}")