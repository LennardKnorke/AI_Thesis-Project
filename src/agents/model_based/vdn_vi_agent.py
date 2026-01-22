import numpy as np
import pickle
import random
from typing import Dict, Tuple, List, Optional, Union

from ..base_agent import ModelBasedAgent, AgentList
from tiny_game import DecPOMDP

class VDN_VI_Agent(ModelBasedAgent):
    """
    The Executor for Centralized Model-Based Planning.
    It is a 'dumb' executor that holds a policy provided by the LearnerList.
    It does not train itself.
    """
    def __init__(
        self, 
        num_cards: int, 
        num_actions: int,
        # The shared policy is passed from the List
        policy: Dict[tuple, int],
        *args, **kwargs
    ):
        super().__init__(num_cards, num_actions)
        self.policy = policy
        
    def act(self, input_state: tuple) -> int:
        return self.policy.get(input_state, 0)
    
    def train(self):
        return 0.0 # Handled by List
    
    def save_transition(self, *args):
        pass
        
    def save(self, filepath: str):
        # We save the shared policy
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.policy, f)
        except Exception as e:
            print(f"Error saving agent: {e}")

    def load(self, filepath: str):
        pass # Loading handled by List setup usually, or manual reload


class VDN_VI_AgentList(AgentList):
    """
    Centralized Model-Based Planner (VDN / Coordinate Ascent).
    
    Strategy:
    - Maintains a SHARED Policy (Parameter Sharing) for both roles (P0 and P1).
    - Uses Random Restarts to avoid local optima (common in Coordinate Ascent).
    """
    def __init__(
        self, 
        num_cards: int, 
        num_actions: int, 
        env: DecPOMDP, 
        *args, **kwargs
    ):
        self.env = env
        self.num_cards = num_cards
        self.num_actions = num_actions
        self.NULL_VALUE = -1 # Must match the runner's masking logic
        
        # 1. Generate State Space Internally
        # (Original Env does not provide this)
        self.all_observations = self._generate_state_space()

        # 2. Shared Policy Dictionary
        self.policy: Dict[Tuple, int] = {}
        
        # Initialize Random Policy
        for obs in self.all_observations:
            self.policy[obs] = random.randint(0, num_actions - 1)

        # 3. Create Agents sharing this policy
        agent_0 = VDN_VI_Agent(num_cards, num_actions, self.policy)
        agent_1 = VDN_VI_Agent(num_cards, num_actions, self.policy)

        super().__init__([agent_0, agent_1])

    def _generate_state_space(self) -> List[tuple]:
        """
        Generates all valid observation keys for the game logic.
        """
        observations = []
        
        # 1. Player 0 Contexts (Turn 1) -> (NULL, c1)
        for c1 in range(self.num_cards):
            obs = (self.NULL_VALUE, c1)
            observations.append(obs)

        # 2. Player 1 Contexts (Turn 2) -> (c0, NULL, a0)
        for c0 in range(self.num_cards):
            for a0 in range(self.num_actions):
                obs = (c0, self.NULL_VALUE, a0)
                observations.append(obs)
        
        return observations

    def train(self, restarts: int = 10) -> float:
        """
        Runs Coordinate Ascent multiple times with random initializations 
        and keeps the best resulting policy.
        """
        best_global_policy = None
        best_global_score = -float('inf')
        
        for i in range(restarts):
            # 1. Randomize Policy for this restart
            current_run_policy = {}
            for obs in self.all_observations:
                current_run_policy[obs] = random.randint(0, self.num_actions - 1)
            
            # Temporarily inject this policy into self and agents for calculation
            self.policy = current_run_policy
            self[0].policy = current_run_policy
            self[1].policy = current_run_policy
            
            # 2. Run Convergence Loop (Coordinate Ascent)
            for _ in range(20): 
                max_delta = self._run_one_sweep() 
                if max_delta < 0.0001:
                    break
            
            # 3. Evaluate This Policy
            score = self._evaluate_current_policy()
            
            if score > best_global_score:
                best_global_score = score
                best_global_policy = current_run_policy.copy()

        # 4. Set the best policy found
        self.policy = best_global_policy
        self[0].policy = best_global_policy
        self[1].policy = best_global_policy
        
        return 0.0 # Metric is less relevant with restarts, we return 0 or best_score

    def _run_one_sweep(self):
        max_delta = 0.0
        
        # Split observations by role for ordered optimization
        p1_obs_list = [o for o in self.all_observations if len(o) == 3] # P1: (c0, NULL, a0)
        p0_obs_list = [o for o in self.all_observations if len(o) == 2] # P0: (NULL, c1)

        # --- STEP 1: Optimize Last Mover (Player 1) ---
        for obs in p1_obs_list:
            c0 = obs[0]
            a0 = obs[2]
            
            current_action = self.policy[obs]
            current_ev = self._calc_p1_ev(c0, a0, current_action)
            
            best_a = current_action
            best_val = current_ev
            
            for a1_cand in range(self.num_actions):
                if a1_cand == current_action: continue
                
                ev = self._calc_p1_ev(c0, a0, a1_cand)
                if ev > best_val:
                    best_val = ev
                    best_a = a1_cand
            
            if best_a != self.policy[obs]:
                self.policy[obs] = best_a

            delta = abs(best_val - current_ev)
            if delta > max_delta:
                max_delta = delta

        # --- STEP 2: Optimize First Mover (Player 0) ---
        for obs in p0_obs_list:
            c1 = obs[1]
            
            current_action = self.policy[obs]
            current_ev = self._calc_p0_ev(c1, current_action)
            
            best_a = current_action
            best_val = current_ev
            
            for a0_cand in range(self.num_actions):
                if a0_cand == current_action: continue
                
                ev = self._calc_p0_ev(c1, a0_cand)
                if ev > best_val:
                    best_val = ev
                    best_a = a0_cand
                    
            if best_a != current_action:
                self.policy[obs] = best_a
                
            delta = abs(best_val - current_ev)
            if delta > max_delta:
                max_delta = delta
                
        return max_delta

    def _calc_p1_ev(self, c0, a0, a1_action):
        """Helper to calc Expected Value for P1 (marginalize c1)"""
        total_payoff = 0.0
        scenarios = 0
        for c1 in range(self.num_cards):
            total_payoff += self.env.payoffs[c0, c1, a0, a1_action]
            scenarios += 1
        return total_payoff / max(1, scenarios)

    def _calc_p0_ev(self, c1, a0_action):
        """Helper to calc Expected Value for P0 (marginalize c0 & predict P1)"""
        total_payoff = 0.0
        scenarios = 0
        for c0 in range(self.num_cards):
            # Predict P1's move based on current shared policy
            p1_obs = (c0, self.NULL_VALUE, a0_action)
            a1_response = self.policy[p1_obs]
            
            total_payoff += self.env.payoffs[c0, c1, a0_action, a1_response]
            scenarios += 1
        return total_payoff / max(1, scenarios)

    def _evaluate_current_policy(self):
        # Calculate expected return from start states (P0 view)
        p0_obs_list = [o for o in self.all_observations if len(o) == 2]
        
        if not p0_obs_list: return 0.0
        
        total_ev = 0
        for obs in p0_obs_list:
            c1 = obs[1]
            action = self.policy[obs]
            total_ev += self._calc_p0_ev(c1, action)
        return total_ev / len(p0_obs_list)

    def save(self, filepath: str):
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.policy, f)
        except Exception as e:
            print(f"Error saving VDN_VI List: {e}")