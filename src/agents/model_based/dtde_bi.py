# dtde_bi.py
import random
import numpy as np
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Optional

from ..base_agent import ModelBasedAgent
from tiny_game import DecPOMDP

class DTDE_BI_MB_Agent(ModelBasedAgent):
    """
    Decentralized Backward Induction Agent.
    Solves the game in reverse order (Turn 2 -> Turn 1).
    Assumes the partner is Rational (maximizes their own expected return).
    """
    def __init__(
        self, 
        num_cards: int, 
        num_actions: int, 
        env: DecPOMDP,
        # BI implies rationality, but we can add noise modeling if desired
        partner_optimality: float = 1.0, # 1.0 = Assume Partner is Perfect, 0.0 = Random
        *args, **kwargs
    ):
        super().__init__(num_cards, num_actions)
        self.env = env
        self.partner_optimality = partner_optimality
        self.NULL_VALUE = -1
        
        self.policy: Dict[tuple, int] = {}
        self.v_values: Dict[tuple, float] = defaultdict(float)
        
        self.all_observations = self._generate_state_space()
        self._init_tables(**kwargs)

    def _generate_state_space(self) -> List[tuple]:
        observations = []
        # P0 (Turn 1): (NULL, c1)
        for c1 in range(self.num_cards):
            observations.append((self.NULL_VALUE, c1))
        # P1 (Turn 2): (c0, NULL, a0)
        for c0 in range(self.num_cards):
            for a0 in range(self.num_actions):
                observations.append((c0, self.NULL_VALUE, a0))
        return observations

    def _init_tables(self, model_path: Optional[str] = None, *args, **kwargs):
        if model_path is not None:
            self.load(model_path)
            return
        for obs in self.all_observations:
            self.v_values[obs] = 0.0
            self.policy[obs] = random.randint(0, self.num_actions - 1)

    def train(self) -> float:
        """
        Executes Backward Induction (Single Pass).
        1. Solve P1 (Last Mover).
        2. Solve P0 (First Mover) using P1's derived values.
        """
        max_delta = 0.0
        
        # Split observations by time-step/role
        p1_obs_list = [o for o in self.all_observations if len(o) == 3] # Last Mover
        p0_obs_list = [o for o in self.all_observations if len(o) == 2] # First Mover

        # --- STEP 1: SOLVE LAST MOVER (P1) ---
        # P1's value depends only on the Payoff Matrix (Leaf Nodes)
        for obs in p1_obs_list:
            delta = self._optimize_node(obs, is_last_mover=True)
            if delta > max_delta: max_delta = delta

        # --- STEP 2: SOLVE FIRST MOVER (P0) ---
        # P0's value depends on P1's expected response
        for obs in p0_obs_list:
            delta = self._optimize_node(obs, is_last_mover=False)
            if delta > max_delta: max_delta = delta
            
        return max_delta

    def _optimize_node(self, obs, is_last_mover) -> float:
        old_v = self.v_values[obs]
        
        q_values = np.zeros(self.num_actions)
        for action in range(self.num_actions):
            q_values[action] = self._calculate_expected_return(obs, action, is_last_mover)
            
        best_val = np.max(q_values)
        best_actions = np.flatnonzero(q_values == best_val)
        best_action = int(np.random.choice(best_actions))
        
        self.v_values[obs] = best_val
        self.policy[obs] = best_action
        
        return abs(best_val - old_v)

    def _calculate_expected_return(self, obs, action, is_last_mover):
        total_payoff = 0.0
        scenarios = 0
        
        # Ranges for Hidden Cards
        if is_last_mover: # P1 sees (c0, NULL, a0)
            c0_fixed = obs[0]
            range_c0 = [c0_fixed]
            range_c1 = range(self.num_cards) # Hidden c1
            a0_prev = obs[2]
        else: # P0 sees (NULL, c1)
            range_c0 = range(self.num_cards) # Hidden c0
            c1_fixed = obs[1]
            range_c1 = [c1_fixed]

        for c0 in range_c0:
            for c1 in range_c1:
                if is_last_mover:
                    # Immediate Payoff
                    total_payoff += self.env.payoffs[c0, c1, a0_prev, action]
                    scenarios += 1
                else:
                    # P0 predicting P1.
                    # We solve the subgame for P1 on the fly.
                    # P1 sees: (c0, NULL, action)
                    
                    # 1. Calculate P1's Q-values for this specific hypothetical state
                    p1_q_values = []
                    for a1 in range(self.num_actions):
                        # P1 calculates their own EV over hidden c1
                        p1_ev_sum = 0
                        for c1_hyp in range(self.num_cards):
                            p1_ev_sum += self.env.payoffs[c0, c1_hyp, action, a1]
                        p1_q_values.append(p1_ev_sum / self.num_cards)
                    
                    # 2. Determine P1's choice based on Rationality
                    best_p1_val = max(p1_q_values)
                    avg_p1_val = sum(p1_q_values) / len(p1_q_values)
                    
                    # E[V] = (Optimality * Best) + ((1-Optimality) * Avg)
                    # Note: We track the VALUE here, but we need the ACTION to get the REAL payoff 
                    # for the actual (c0, c1) pair we are iterating in the outer loop.
                    
                    # Since P1 optimizes for average c1, but we hold a specific c1 in this loop:
                    # We find the set of optimal actions for P1
                    best_p1_actions = [i for i, v in enumerate(p1_q_values) if v == best_p1_val]
                    
                    # Expected payoff for P0 is the average payoff of those actions given REAL c1
                    
                    # Rational Component
                    rational_payoff_sum = 0
                    for a1_opt in best_p1_actions:
                        rational_payoff_sum += self.env.payoffs[c0, c1, action, a1_opt]
                    rational_component = rational_payoff_sum / len(best_p1_actions)
                    
                    # Random Component
                    random_payoff_sum = 0
                    for a1_rnd in range(self.num_actions):
                        random_payoff_sum += self.env.payoffs[c0, c1, action, a1_rnd]
                    random_component = random_payoff_sum / self.num_actions
                    
                    expected_val = (self.partner_optimality * rational_component) + \
                                   ((1 - self.partner_optimality) * random_component)
                                   
                    total_payoff += expected_val
                    scenarios += 1
                    
        return total_payoff / max(1, scenarios)

    def act(self, input_state: tuple, exploit: bool = False) -> int:
        return self.policy.get(input_state, 0)

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