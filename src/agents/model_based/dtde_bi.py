# agents/model_based/dtde_bi.py
import random
import numpy as np
import os
import pickle
from collections import defaultdict

from ..base_agent import ModelBasedAgent
from tiny_game import (
    DecPOMDP, MyHanabi, Game,
    get_all_possible_histories, get_all_possible_states
)


class DTDE_BI_MB_Agent(ModelBasedAgent):
    """
    Decentralized Backward Induction Agent.
    Solves the game in reverse order (Turn 2 -> Turn 1).
    Assumes the partner is Rational (maximizes their own expected return).
    """
    def __init__(
        self, 
        env: Game,
        num_cards: int, 
        num_actions: int,
        agent_id : int,
        gamma : float = 0.99,
        *args, **kwargs
    ):
        super().__init__(env, num_cards, num_actions)
        self.agent_id = agent_id
        self.NULL_VALUE = -1

        # Cache observations for planning
        self.legal_actions_cache : dict[tuple, tuple]= {}
        self.worlds_cache: dict[tuple, list[tuple]] = {}
        
        # Tables necessary
        self.policy: dict[tuple, int] = {}
        self.v_values: dict[tuple, float] = defaultdict(float)        

        self._init_tables()
        return

    def _init_tables(self):
        for history, done, turn_id, reward in self.all_private_histories:
            if done:
                self.v_values[history] = reward
                possible_actions = ()
            else:
                self.v_values[history] = 0.0
                if self.is_decpomdp:
                    possible_actions = tuple(range(self.num_actions))
                else:
                    _, possible_actions = self.env.num_legal_actions(history)
                self.legal_actions_cache[history] = possible_actions
                self.policy[history] = random.choice(possible_actions)
        return
    
    def calc_expected_reward(self, obs)->float:
        return 0.0

    def train(self) -> float:
        """
        Executes Backward Induction (Single Pass).
        """
        max_delta = 0.0
        
        # Sort observation longest first
        for i, (obs, done, turn_id, reward) in enumerate(self.all_private_histories):
            if done:
                continue
            if turn_id != self.agent_id:
                continue

            delta = self._optimize_node(obs)
            if delta > max_delta:
                max_delta = delta
        return max_delta

    def _optimize_node(self, hist) -> float:
        old_val = self.v_values[hist]
        legal_actions = self.legal_actions_cache[hist]

        worlds = self._get_consistent_worlds(hist)
        
        # Initialize with -infinity so any valid path overwrites it
        best_value = -float('inf')
        best_action = legal_actions[0]

        for action in legal_actions:
            total = 0.0
            valid_worlds = 0

            for world_state in worlds:                
                # A. Reset Env to this specific world
                # B. Take Step
                try:
                    self.env.reset(list(world_state))
                    self.env.step(action)
                except ValueError:
                    # Action invalid in this specific world 
                    continue
                valid_worlds += 1

                # C. Check Consequence
                if self.env.is_terminal():
                    total += self.env.payoff()
                    continue

                next_history = tuple(self.env.history)
                next_obs = self._mask_state(next_history)
                total += self.v_values[next_obs]

            if valid_worlds == 0:
                continue

            avg = total / valid_worlds
            if avg > best_value:
                best_value = avg
                best_action = action

        self.policy[hist] = best_action
        self.v_values[hist] = best_value
        return abs(old_val - best_value)

    def _get_consistent_worlds(self, obs: tuple) -> list[tuple]:
        """
        Returns all possible ground-truth histories consistent with the observation.
        Uses caching for speed.
        """
        if obs in self.worlds_cache:
            return self.worlds_cache[obs]
        
        consistent = []
        
        all_deals = self.env.start_states()
        
        # Extract the visible part of the deal from obs
        if self.is_decpomdp:
            obs_deal = obs[:2]
            obs_history = obs[2:]
            deal_len = 2
        else:
            obs_deal = obs[:4]
            obs_history = obs[4:]
            deal_len = 4
            
        for deal in all_deals:
            match = True
            for i in range(deal_len):
                # If obs has -1, any card is valid.
                if obs_deal[i] != -1 and obs_deal[i] != deal[i]:
                    match = False
                    break
            
            if match:
                # Construct the full state candidate
                candidate_state = tuple(list(deal) + list(obs_history))
                consistent.append(candidate_state)
        
        self.worlds_cache[obs] = consistent
        return consistent

    def act(self, input_state: tuple, exploit: bool = False, *args, **kwargs) -> int:
        action = self.policy[input_state]
        return action

    def save(self, filepath: str):
        data = {"policy": self.policy, "v_values": dict(self.v_values)}
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        return
    
    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        if os.path.getsize(filepath) == 0:
            raise ValueError("Q-table file is empty")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            raise ValueError("Loaded file does not contain a valid dictionary.")
        
        # Reconstruct defaultdict
        self.policy.update(data['policy'])
        self.v_values.update(data['v_values'])
        return
    
    def save_transition(self, *args): pass
    def reset(self): pass


    def _mask_state(self, state: tuple) -> tuple:
        """
        Converts a full ground-truth state/history into the observation 
        seen by the player whose turn it IS.
        """
        s_list = list(state)
        if self.is_decpomdp:
            num_actions = len(state) - 2
            p0_turn = (num_actions % 2 == 0)
            if p0_turn:
                s_list[0] = -1
            else:
                s_list[1] = -1
        elif self.is_myhanabi:
            # history: [c0a, c0b, c1a, c1b, a1, a2...]
            num_actions = len(state) - 4
            p0_turn = (num_actions % 2 == 0)
            
            if p0_turn:
                s_list[0] = -1; s_list[1] = -1
            else:
                s_list[2] = -1; s_list[3] = -1
        return tuple(s_list)