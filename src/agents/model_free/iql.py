# agents/model_free/iql.py

import numpy as np
import os
import pickle

from tiny_game import DecPOMDP, Game
from replaybuffer import ReplayBuffer
from ..base_agent import ModelFreeAgent


class IQ_Learning_Agent(ModelFreeAgent):
    """
    Independent Model Free Reinforcement Learning Agent
    Lower Performance Bound
    """
    def __init__(
            self,
            env: Game,
            game_name: str,
            num_cards: int,
            num_actions: int,
            # Params
            lr: float               = 0.1,
            gamma: float            = 0.9,
            epsilon_start: float    = 1.0,
            epsilon_min: float      = 0.05,
            epsilon_decay: float    = 0.9995,
            batch_size: int         = 32,
            buffer_size: int        = 10_000,
            updates_per_train: int  = 1,
            *args, **kwargs):
        super().__init__(env, num_cards, num_actions, *args, **kwargs)
        self.game_name = game_name

        # Params
        self.lr =                   lr
        self.gamma =                gamma
        self.epsilon =              epsilon_start
        self.epsilon_min =          epsilon_min
        self.epsilon_decay =        epsilon_decay
        self.batch_size =           batch_size
        self.buffer_size =          buffer_size
        self.updates_per_train =    int(updates_per_train)

        self.replay_buffer =        ReplayBuffer(buffer_size)

        # Q-Table and greedy policy
        self.policy =               {}
        self.q_values =             {}
        # Legal actions
        self.legal_actions_cache =  {}
        return


    def get_legal_actions_mask(self, history: tuple[int]) -> np.ndarray:
        if isinstance(self.env, DecPOMDP):
            return np.ones(self.num_actions, dtype=bool)
        else: 
            mask, _ = self.env.num_legal_actions(full_history=history)
            return np.array(mask, dtype=bool)


    def act(self, input_state: tuple[int], exploit=False) -> int:
        # --- Ensure legal actions are cached for this state ---
        if input_state not in self.legal_actions_cache:
            mask = self.get_legal_actions_mask(input_state)
            self.legal_actions_cache[input_state] = np.where(mask)[0].tolist()
        legal_as = self.legal_actions_cache[input_state]

        # --- Lazy init for Q‑values ---
        if input_state not in self.q_values:
            q_values = np.zeros(self.num_actions)
            for a in legal_as:
                q_values[a] = 1.0          # optimistic initial value for legal actions
            self.q_values[input_state] = q_values

        # --- Compute greedy action (only over legal actions) ---
        if input_state not in self.policy:
            # Mask illegal actions to -inf before argmax
            q_vals = self.q_values[input_state].copy()
            illegal_mask = np.ones(self.num_actions, dtype=bool)
            illegal_mask[legal_as] = False
            q_vals[illegal_mask] = -np.inf
            best_action = int(np.argmax(q_vals))
            self.policy[input_state] = best_action

        # --- Epsilon‑greedy selection ---
        if exploit or np.random.rand() > self.epsilon:
            action = self.policy[input_state]
        else:
            action = np.random.choice(legal_as)
        return int(action)

    def save_transition(self, observation, action, next_observation, reward, done):
        self.replay_buffer.push(
            tuple(observation),
            action,
            reward,
            tuple(next_observation),
            done
        )
        return

    def train(self) -> float:
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        total_loss = 0.0
        for _ in range(self.updates_per_train):
            batch = self.replay_buffer.sample(self.batch_size)
            batch_loss = 0.0

            for transition in batch:
                state, action, reward, next_state, done = transition

                # Lazy init for Q‑values
                if state not in self.q_values:
                    self.q_values[state] = np.ones(self.num_actions, dtype=np.float32) * 10.0
                if next_state not in self.q_values and not done:
                    self.q_values[next_state] = np.ones(self.num_actions, dtype=np.float32) * 10.0

                # Current Q
                current_q = self.q_values[state][action]

                # Target Q
                target_q = reward
                if not done:
                    # Cache legal actions for next_state if needed
                    if next_state not in self.legal_actions_cache:
                        mask = self.get_legal_actions_mask(next_state)
                        self.legal_actions_cache[next_state] = np.where(mask)[0].tolist()
                    legal_next = self.legal_actions_cache[next_state]

                    # Compute max over legal next actions
                    next_q_vals = self.q_values[next_state].copy()
                    illegal_mask = np.ones(self.num_actions, dtype=bool)
                    illegal_mask[legal_next] = False
                    next_q_vals[illegal_mask] = -np.inf
                    max_next_q = np.max(next_q_vals)
                    target_q += self.gamma * max_next_q

                # Q‑learning update
                td_error = target_q - current_q
                self.q_values[state][action] += self.lr * td_error
                batch_loss += abs(td_error)

            batch_loss /= self.batch_size
            total_loss += batch_loss

        # --- Epsilon decay ---
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        # --- Update policy for every seen state ---
        for state in self.q_values.keys():
            # Ensure legal actions are cached
            if state not in self.legal_actions_cache:
                mask = self.get_legal_actions_mask(state)
                self.legal_actions_cache[state] = np.where(mask)[0].tolist()
            legal_as = self.legal_actions_cache[state]

            if legal_as:
                q_vals = self.q_values[state].copy()
                illegal_mask = np.ones(self.num_actions, dtype=bool)
                illegal_mask[legal_as] = False
                q_vals[illegal_mask] = -np.inf
                best_action = int(np.argmax(q_vals))
                self.policy[state] = best_action
            else:
                if state in self.policy:
                    del self.policy[state]

        return total_loss / self.updates_per_train

    def save(self, save_path: str):
        data = {
            "q_vals": dict(self.q_values),
            "policy": dict(self.policy)
        }
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, load_path: str):
        if not os.path.exists(load_path):
            raise FileNotFoundError(load_path)
        if os.path.getsize(load_path) == 0:
            raise ValueError("Q-table file is empty")
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, dict) or len(data) == 0:
            raise ValueError("Invalid or empty save file")
        self.q_values.update(data['q_vals'])
        self.policy.update(data['policy'])