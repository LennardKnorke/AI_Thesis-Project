# agents/model_free/ctde_vdn.py
import numpy as np
import pickle
import random
from collections import defaultdict, deque, namedtuple
from typing import Any

from replaybuffer import EpisodeStep, EpisodicReplayBuffer

from agents.base_agent import ModelFreeAgent, AgentList

from tiny_game import DecPOMDP, MyHanabi, Game


class CTDE_VDN_MF_Agent(ModelFreeAgent):
    """
    Decentralized Execution Agent for VDN.
    - Acts using a shared Q-table.
    - Sends experiences to a shared Episodic Buffer.
    """
    def __init__(
        self, 
        env : Game,
        num_cards: int, 
        num_actions: int,
        # Shared components passed from the List
        q_table: dict,
        buffer: EpisodicReplayBuffer,
        epsilon_start: float
    ):
        super().__init__(env, num_cards, num_actions)

        # References to shared objects managed by the List
        self.q_table = q_table
        self.buffer = buffer
        
        # Local Epsilon (synced by list during training)
        self.epsilon = epsilon_start

    def get_legal_actions_mask(self, obs: tuple[int]) -> np.ndarray:
        """Helper to get legal actions mask for an observation."""
        if isinstance(self.env, DecPOMDP):
            return np.ones(self.num_actions, dtype=bool)
        else: # MyHanabi
            mask, _ = self.env.num_legal_actions(history=obs)
            return np.array(mask, dtype=bool)
    
    @property
    def requires_tensor(self) -> bool:
        return False

    def act(self, input_state: tuple[int], exploit: bool = False) -> int:
        """
        Epsilon-Greedy Action Selection.
        """
        state_key = tuple(input_state)

        # Get legal actions
        legal_action_mask = self.get_legal_actions_mask(state_key)
        legal_actions = np.where(legal_action_mask)[0]

        # Lazy init
        if state_key not in self.q_table:
            q_values = np.zeros(self.num_actions, dtype=np.float32)
            q_values[legal_action_mask] = 10.0
            self.q_table[state_key] = q_values
        
        # Exploit 
        if exploit or np.random.rand() > self.epsilon:
            q_values = self.q_table[state_key].copy()
            q_values[~legal_action_mask] = -np.inf
            max_val = np.max(q_values)
            best_actions = np.flatnonzero(q_values == max_val)
            action = np.random.choice(best_actions)
        # or Explore
        else:
            action = np.random.choice(legal_actions)
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
        env : Game,
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
        self.q_table = {}
        
        self.buffer = EpisodicReplayBuffer(buffer_size)
        
        # 2. Create Agents
        agent_0 = CTDE_VDN_MF_Agent(env, num_cards, num_actions, self.q_table, self.buffer, self.epsilon)
        agent_1 = CTDE_VDN_MF_Agent(env, num_cards, num_actions, self.q_table, self.buffer, self.epsilon)
        
        # Initialize List
        super().__init__([agent_0, agent_1])

    def get_legal_actions_mask(self, obs: tuple[int]) -> np.ndarray:
        """Helper to get legal actions mask for an observation, used for Q-table init."""
        if isinstance(self.agents[0].env, DecPOMDP): # Use agent's env to determine type
            return np.ones(self.num_actions, dtype=bool)
        else: # MyHanabi
            mask, _ = self.agents[0].env.num_legal_actions(history=obs)
            return np.array(mask, dtype=bool)
        
    def train(self) -> float:
        """
        Centralized Training Loop.
        Samples episodes, sums Q-values, and backpropagates error.
        """
        # 1. Check Buffer
        if len(self.buffer) < self.batch_size:
            return 0.0

        total_loss = 0.0

        # 2. Training Loop
        for _ in range(self.updates_per_train):
            batch = self.buffer.sample(self.batch_size)
            batch_loss = 0.0
            
            for episode_steps, final_reward in batch:
                G = final_reward # G is the discounted return to be backed up
                
                num_agent_steps = len(episode_steps)
                # Assuming 2 players, each game turn consists of 2 agent steps (P0 then P1)
                num_game_turns = num_agent_steps // 2 

                # Iterate in reverse over game turns for Monte Carlo backup
                for turn_idx_rev in range(num_game_turns): 
                    current_game_turn = num_game_turns - 1 - turn_idx_rev

                    # Get steps for P0 and P1 for this specific game turn
                    p0_step = episode_steps[current_game_turn * 2]
                    p1_step = episode_steps[current_game_turn * 2 + 1]

                    p0_state, p0_action = p0_step.state, p0_step.action
                    p1_state, p1_action = p1_step.state, p1_step.action

                    # Lazy init Q-values if not present
                    if p0_state not in self.q_table:
                        self.q_table[p0_state] = np.zeros(self.num_actions, dtype=np.float32)
                        self.q_table[p0_state][self.get_legal_actions_mask(p0_state)] = 10.0
                    if p1_state not in self.q_table:
                        self.q_table[p1_state] = np.zeros(self.num_actions, dtype=np.float32)
                        self.q_table[p1_state][self.get_legal_actions_mask(p1_state)] = 10.0

                    q_p0 = self.q_table[p0_state][p0_action]
                    q_p1 = self.q_table[p1_state][p1_action]
                    
                    q_tot_current = q_p0 + q_p1 # The VDN sum for the current joint action

                    # The target is G (the discounted final reward from this game turn onwards)
                    td_error = G - q_tot_current

                    # Apply the update to individual Q-values (shared error)
                    self.q_table[p0_state][p0_action] += self.lr * td_error
                    self.q_table[p1_state][p1_action] += self.lr * td_error
                    
                    batch_loss += abs(td_error)

                    # Discount G for the next (earlier) game turn
                    G = G * self.gamma

            total_loss += batch_loss / num_game_turns if num_game_turns > 0 else 0.0

        avg_loss = total_loss / self.updates_per_train

        # Epsilon decay
        self.epsilon = max(
            self.epsilon * self.epsilon_decay,
            self.epsilon_min
        )

        # Sync epsilon to individual agents
        for agent in self:
            agent.epsilon = self.epsilon

        return avg_loss

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
            
            self.q_table = data["q_table"]
            self.epsilon = data.get("epsilon", self.epsilon)
            
            # Re-sync agents
            for agent in self:
                agent.q_table = self.q_table
                agent.epsilon = self.epsilon
                
        except Exception as e:
            print(f"Error loading VDN model: {e}")