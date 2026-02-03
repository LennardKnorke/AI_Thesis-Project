# ctde_bi.py
import random
import numpy as np
import pickle
from typing import Dict, Tuple, List

from ..base_agent import ModelBasedAgent, AgentList
from tiny_game import DecPOMDP

class CTDE_BI_MB_Agent(ModelBasedAgent):
    """Executor shell for Centralized BI. Just holds the policy."""
    def __init__(self, num_cards, num_actions, policy: Dict):
        super().__init__(num_cards, num_actions)
        self.policy = policy
    def act(self, input_state: tuple, exploit: bool = False) -> int:
        return self.policy.get(input_state, 0)
    def train(self): return 0.0
    def save_transition(self, *args): pass
    def save(self, *args): pass
    def reset(self): pass

class CTDE_BI_MB_List(AgentList):
    """
    Centralized Backward Induction Planner.
    Optimizes the joint policy in exactly one backwards sweep (P1 -> P0).
    """
    def __init__(self, num_cards: int, num_actions: int, env: DecPOMDP, *args, **kwargs):
        self.env = env
        self.num_cards = num_cards
        self.num_actions = num_actions
        self.NULL_VALUE = -1
        
        # Shared Policy
        self.policy: Dict[Tuple, int] = {}
        
        # 1. Generate State Space
        self.all_observations = self._generate_state_space()
        
        # 2. Init Random Policy
        for obs in self.all_observations:
            self.policy[obs] = random.randint(0, num_actions - 1)

        # 3. Create Executors
        agent_0 = CTDE_BI_MB_Agent(num_cards, num_actions, self.policy)
        agent_1 = CTDE_BI_MB_Agent(num_cards, num_actions, self.policy)
        super().__init__([agent_0, agent_1])

    def _generate_state_space(self) -> List[tuple]:
        obs_list = []
        # P0 (Turn 1)
        for c1 in range(self.num_cards): obs_list.append((self.NULL_VALUE, c1))
        # P1 (Turn 2)
        for c0 in range(self.num_cards):
            for a0 in range(self.num_actions):
                obs_list.append((c0, self.NULL_VALUE, a0))
        return obs_list

    def train(self) -> float:
        """
        1. Optimize P1 (Leaf nodes).
        2. Optimize P0 (Root nodes) based on P1's fixed policy.
        Returns max_delta (change in value).
        """
        max_delta = 0.0
        
        # Sort observations to ensure we process Turn 2 (Length 3) BEFORE Turn 1 (Length 2)
        # This guarantees Backward Induction order.
        p1_obs_list = [o for o in self.all_observations if len(o) == 3]
        p0_obs_list = [o for o in self.all_observations if len(o) == 2]

        # --- STEP 1: SOLVE P1 ---
        for obs in p1_obs_list:
            delta = self._optimize_p1(obs)
            if delta > max_delta: max_delta = delta

        # --- STEP 2: SOLVE P0 ---
        for obs in p0_obs_list:
            delta = self._optimize_p0(obs)
            if delta > max_delta: max_delta = delta
            
        return max_delta

    def _optimize_p1(self, obs) -> float:
        c0, a0 = obs[0], obs[2]
        
        # Calculate current value for delta tracking
        curr_act = self.policy[obs]
        curr_ev = self._calc_p1_ev(c0, a0, curr_act)
        
        # Find Best
        best_a = curr_act
        best_val = curr_ev
        
        for a1 in range(self.num_actions):
            if a1 == curr_act: continue
            ev = self._calc_p1_ev(c0, a0, a1)
            if ev > best_val:
                best_val = ev
                best_a = a1
        
        self.policy[obs] = best_a
        return abs(best_val - curr_ev)

    def _optimize_p0(self, obs) -> float:
        c1 = obs[1]
        
        curr_act = self.policy[obs]
        curr_ev = self._calc_p0_ev(c1, curr_act)
        
        best_a = curr_act
        best_val = curr_ev
        
        for a0 in range(self.num_actions):
            if a0 == curr_act: continue
            ev = self._calc_p0_ev(c1, a0)
            if ev > best_val:
                best_val = ev
                best_a = a0
                
        self.policy[obs] = best_a
        return abs(best_val - curr_ev)

    def _calc_p1_ev(self, c0, a0, a1_act):
        # Marginalize over hidden c1
        total = 0.0
        scenarios = 0
        for c1 in range(self.num_cards):
            total += self.env.payoffs[c0, c1, a0, a1_act]
            scenarios += 1
        return total / max(1, scenarios)

    def _calc_p0_ev(self, c1, a0_act):
        # Marginalize over hidden c0 AND use P1's fixed policy
        total = 0.0
        scenarios = 0
        for c0 in range(self.num_cards):
            # Lookup P1's move
            p1_obs = (c0, self.NULL_VALUE, a0_act)
            a1_resp = self.policy[p1_obs]
            
            total += self.env.payoffs[c0, c1, a0_act, a1_resp]
            scenarios += 1
        return total / max(1, scenarios)

    def save(self, filepath: str):
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.policy, f)
        except Exception as e:
            print(f"Error saving BI model: {e}")
            
    def reset(self):
        # For BI, we might want to reset to random before a new training attempt
        # if using the 'attempts' loop in runner.
        for obs in self.all_observations:
            self.policy[obs] = random.randint(0, self.num_actions - 1)

    def load(self, filepath: str):
        """
        Loads the Shared Policy.
        """
        try:
            with open(filepath, 'rb') as f:
                loaded_policy = pickle.load(f)
            # Make sure it's a dict and update our policy
            if isinstance(loaded_policy, dict):
                self.policy = loaded_policy
                # Important: The CTDE_BI_MB_Agent instances also hold references
                # to this self.policy, so they will automatically be updated.
            else:
                raise ValueError("Loaded file does not contain a valid policy dictionary.")
        except FileNotFoundError:
            print(f"File not found: {filepath}. Initializing policy randomly.")
            self.reset()
        except Exception as e:
            print(f"Error loading BI model from {filepath}: {e}. Initializing policy randomly.")
            self.reset()