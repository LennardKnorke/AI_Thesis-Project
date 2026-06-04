# agents/model_free/ctde_vdn.py
import numpy as np
import pickle
import random
from collections import defaultdict, deque, namedtuple
from typing import Any

from replaybuffer import EpisodeStep, EpisodicReplayBuffer

from agents.base_agent import ModelFreeAgent, AgentList

from tiny_game import Game, PRIV_HISTORIES


class VDN_Agent(ModelFreeAgent):
    """
    CTDE Model Free Reinfocement Learning Agent
    - Acts using a shared Q-table.
    """
    def __init__(
        self, 
        env : Game,
        num_cards: int, 
        num_actions: int,
        # Shared components passed from the List
        q_table: dict,
        policy:dict,
        legal_actions_cache:dict,
        buffer: EpisodicReplayBuffer,
        epsilon_start: float
    ):
        super().__init__(env, num_cards, num_actions)

        # References to shared objects managed by the List
        self.q_values =             q_table
        self.policy =               policy
        self.buffer =               buffer
        self.legal_actions_cache =  legal_actions_cache
        
        # Local Epsilon (synced by list during training)
        self.epsilon =              epsilon_start
    
    @property
    def requires_tensor(self) -> bool:
        return False

    def act(self, input_state: tuple[int], exploit: bool = False) -> int:
        state_key = tuple(input_state)

        # Get legal actions
        legal_actions = self.legal_actions_cache[state_key]

        # Lazy init
        if state_key not in self.q_values:
            q_values = np.zeros(self.num_actions, dtype=np.float32)
            for a in legal_actions:
                q_values[int(a)] = 10.0
            self.q_values[state_key] = q_values
            self._refresh_policy(state_key)

        # Exploit or Expl
        if exploit or np.random.rand() > self.epsilon:
            action = self.policy[state_key]
        else:
            action = np.random.choice(legal_actions)
        return int(action)

    def _refresh_policy(self, state_key: tuple) -> None:
        legal = self.legal_actions_cache[state_key]
        q = self.q_values[state_key]
        masked = np.full_like(q, -np.inf)
        for a in legal:
            masked[int(a)] = q[int(a)]
        self.policy[state_key] = int(np.argmax(masked))
        return

    def save_transition(self, observation, action, next_observation, reward, done):
        self.buffer.push_step(tuple(observation), action)
        if done:
            self.buffer.close_episode(reward)
        return

    def train(self):        pass
    def save(self, *args):  pass
    def load(self, *args):  pass
    def reset(self):        pass


class VDN_CentralPlanner(AgentList):
    """
    Centralized Training Controller for VDN.
    """
    def __init__(
        self,
        env : Game,
        game_name : str,
        num_cards: int, 
        num_actions: int,
        # Hyperparameters
        lr: float                   = 0.5,
        gamma: float                = 0.99,
        epsilon_start: float        =  1.0,
        epsilon_min: float          = 0.05,
        epsilon_decay: float        = 0.9995,
        batch_size: int             = 32,
        updates_per_train : int     = 1,
        buffer_size: int            = 1_000,
        *args, **kwargs
    ):
        self.model_based =  False
        self.num_actions =  num_actions
        self.game_name =    game_name

        self.legal_actions_cache = {
            state : actions for state, actions, done, _, _ in PRIV_HISTORIES[self.game_name] if not done
        }
        
        self.lr =                   lr
        self.gamma =                gamma
        self.batch_size =           batch_size
        self.updates_per_train =    int(updates_per_train)
        
        self.epsilon =              epsilon_start
        self.epsilon_min =          epsilon_min
        self.epsilon_decay =        epsilon_decay
        
        # Shared Components
        self.policy =   {}
        self.q_values = {}
        
        self.buffer =   EpisodicReplayBuffer(buffer_size)
        
        # Create Agents
        agent_0 = VDN_Agent(env, num_cards, num_actions, self.q_values, self.policy, self.legal_actions_cache, self.buffer, self.epsilon)
        agent_1 = VDN_Agent(env, num_cards, num_actions, self.q_values, self.policy, self.legal_actions_cache, self.buffer, self.epsilon)
        
        # Initialize List
        super().__init__([agent_0, agent_1])
        
    @property
    def centralized_planning(self):
        return True
    
    def _lazy_init_state(self, state: tuple) -> None:
        if state in self.q_values:
            return
        legal = self.legal_actions_cache[state]
        q = np.zeros(self.num_actions, dtype=np.float32)
        for a in legal:
            q[int(a)] = 10.0
        self.q_values[state] = q
        self._refresh_policy_state(state)

    def _refresh_policy_state(self, state: tuple) -> None:
        legal = self.legal_actions_cache[state]
        q = self.q_values[state]
        masked = np.full_like(q, -np.inf)
        for a in legal:
            masked[int(a)] = q[int(a)]
        self.policy[state] = int(np.argmax(masked))

    def train(self) -> float:
        # Check Buffer
        if len(self.buffer) < self.batch_size:
            return 0.0

        total_loss = 0.0
        total_turns = 0

        # Training Loop
        for _ in range(self.updates_per_train):
            batch = self.buffer.sample(self.batch_size)
            batch_loss = 0.0
            batch_turns = 0

            for episode_steps, final_reward in batch:
                G = final_reward
                num_agent_steps = len(episode_steps)
                num_game_turns = num_agent_steps // 2

                # Iterate
                for turn_idx_rev in range(num_game_turns):
                    current_game_turn = num_game_turns - 1 - turn_idx_rev

                    # Get steps for P0 and P1 for this specific game turn
                    p0_step = episode_steps[current_game_turn * 2]
                    p1_step = episode_steps[current_game_turn * 2 + 1]

                    p0_state, p0_action = p0_step.state, p0_step.action
                    p1_state, p1_action = p1_step.state, p1_step.action

                    # Lazy init Q-values if not present
                    self._lazy_init_state(p0_state)
                    self._lazy_init_state(p1_state)

                    q_p0 = self.q_values[p0_state][p0_action]
                    q_p1 = self.q_values[p1_state][p1_action]

                    q_tot_current = q_p0 + q_p1 # The VDN sum for the current joint action

                    # The target is G (the discounted final reward from this game turn onwards)
                    td_error = G - q_tot_current

                    # Apply the update to individual Q-values (shared error)
                    self.q_values[p0_state][p0_action] += self.lr * td_error
                    self.q_values[p1_state][p1_action] += self.lr * td_error

                    # Refresh greedy policy (masked to legal actions) after Q update
                    self._refresh_policy_state(p0_state)
                    self._refresh_policy_state(p1_state)

                    batch_loss += abs(td_error)
                    batch_turns += 1

                    # Discount G for the next (earlier) game turn
                    G = G * self.gamma

            total_loss += batch_loss
            total_turns += batch_turns

        avg_loss = total_loss / total_turns if total_turns > 0 else 0.0

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
        data = {
            "q_table": dict(self.q_values),
            "epsilon": self.epsilon,
            "policy" : dict(self.policy)
        }
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error saving VDN model: {e}")
        return

    def load(self, filepath: str):
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.q_values = data["q_table"]
            self.epsilon = data.get("epsilon", self.epsilon)
            # Rebuild policy from loaded Q-table with legal-action masking,
            # in case a stale/unmasked policy was pickled.
            self.policy = {}
            for state in self.q_values:
                if state in self.legal_actions_cache:
                    self._refresh_policy_state(state)
            # Re-sync agents (shared dicts must point at the same objects)
            for agent in self:
                agent.q_values = self.q_values
                agent.policy = self.policy
                agent.legal_actions_cache = self.legal_actions_cache
                agent.epsilon = self.epsilon
        except Exception as e:
            print(f"Error loading VDN model: {e}")
        return