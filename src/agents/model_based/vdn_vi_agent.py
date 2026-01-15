import numpy as np
import pickle
import random
from typing import Dict, Tuple, List, Optional, Union

from ..base_agent import ModelBasedAgent, AgentList
from tiny_game import DecPOMP_Rework

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
    - Iteratively optimizes the policy:
      1. Optimize "Last Mover" (P1) situations based on payoffs.
      2. Optimize "First Mover" (P0) situations based on P1's current policy.
    """
    def __init__(
        self, 
        num_cards: int, 
        num_actions: int, 
        env: DecPOMP_Rework, 
        *args, **kwargs
    ):
        self.env = env
        self.num_cards = num_cards
        self.num_actions = num_actions
        
        # 1. Shared Policy Dictionary
        # Maps observation -> Best Action
        self.policy: Dict[Tuple, int] = {}
        
        # Initialize Random Policy for all possible observations
        all_obs = env.get_all_observations(as_tensor=False)
        for obs in all_obs:
            self.policy[obs] = random.randint(0, num_actions - 1)

        # 2. Create Agents sharing this policy
        agent_0 = VDN_VI_Agent(num_cards, num_actions, self.policy)
        agent_1 = VDN_VI_Agent(num_cards, num_actions, self.policy)

        super().__init__([agent_0, agent_1])
        return
        
    
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
            p1_obs = (c0, self.env.NULL_VALUE, a0_action)
            a1_response = self.policy[p1_obs]
            
            total_payoff += self.env.payoffs[c0, c1, a0_action, a1_response]
            scenarios += 1
        return total_payoff / max(1, scenarios)

    def save(self, filepath: str):
        # Save the dict directly
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.policy, f)
        except Exception as e:
            print(f"Error saving VDN_VI List: {e}")

    def train(self, restarts: int = 10) -> float:
        """
        Runs Coordinate Ascent multiple times with random initializations 
        and keeps the best resulting policy.
        """
        best_global_policy = None
        best_global_score = -float('inf')
        
        # We assume the user config passes 'iterations' which maps to 'restarts' here
        # or we just run the loop below.
        
        original_policy_ref = self.policy
        
        for i in range(restarts):
            # 1. Randomize Policy
            current_run_policy = {}
            all_obs = self.env.get_all_observations(as_tensor=False)
            for obs in all_obs:
                current_run_policy[obs] = random.randint(0, self.num_actions - 1)
            
            # Temporarily inject this policy into self and agents
            self.policy = current_run_policy
            self[0].policy = current_run_policy
            self[1].policy = current_run_policy
            
            # 2. Run Convergence Loop (Coordinate Ascent)
            converged = False
            for _ in range(20): # Fixed inner loop usually enough
                max_delta = self._run_one_sweep() # (Refactor logic below into this)
                if max_delta < 0.0001:
                    break
            
            # 3. Evaluate This Policy
            # We can use the environment's optimal_return check or just our internal V calculation
            # Let's calculate the expected return of P0's start states
            score = self._evaluate_current_policy()
            
            if score > best_global_score:
                best_global_score = score
                best_global_policy = current_run_policy.copy()

        # 4. Set the best policy found
        self.policy = best_global_policy
        self[0].policy = best_global_policy
        self[1].policy = best_global_policy
        
        return 0.0 # Done

    def _run_one_sweep(self):
        max_delta = 0.0
        NULL = self.env.NULL_VALUE
        
        # We can optimize in a specific order to speed up convergence.
        # In Tiny Hanabi: P0 acts -> P1 acts.
        # If we optimize P1 first, P0 has perfect info to optimize against.
        all_obs = self.env.get_all_observations(as_tensor=False)
        
        # Separate observations by role
        p1_obs_list = [o for o in all_obs if o[1] == NULL] # P1 sees [c0, NULL, a0]
        p0_obs_list = [o for o in all_obs if o[0] == NULL] # P0 sees [NULL, c1]

        # --- STEP 1: Optimize Last Mover (Player 1) ---
        # P1 maximizes immediate payoff.
        for obs in p1_obs_list:
            # obs structure: (c0, NULL, a0)
            c0 = obs[0]
            a0 = obs[2]
            
            best_a = self.policy[obs]
            best_val = -float('inf')
            current_action = best_a
            current_ev = self._calc_p1_ev(c0, a0, current_action)
            
            for a1_cand in range(self.num_actions):
                if a1_cand == current_action: continue
                
                ev = self._calc_p1_ev(c0, a0, a1_cand)
                if ev > best_val:
                    best_val = ev
                    best_a = a1_cand
            
            if best_a != self.policy[obs]:
                self.policy[obs] = best_a

            # Delta is how much we improved the value
            delta = abs(best_val - current_ev)
            if delta > max_delta:
                max_delta = delta

        # --- STEP 2: Optimize First Mover (Player 0) ---
        # P0 maximizes the result of P1's reaction.
        for obs in p0_obs_list:
            # obs structure: (NULL, c1)
            c1 = obs[1]
            
            best_a = self.policy[obs]
            best_val = -float('inf')
            current_action = best_a
            current_ev = self._calc_p0_ev(c1, current_action)
            
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

    def _evaluate_current_policy(self):
        # Calculate expected return from start states (P0 view)
        p0_obs_list = [o for o in self.env.get_all_observations(False) if o[0] == self.env.NULL_VALUE]
        total_ev = 0
        for obs in p0_obs_list:
            # obs is (NULL, c1)
            c1 = obs[1]
            action = self.policy[obs]
            total_ev += self._calc_p0_ev(c1, action)
        return total_ev / len(p0_obs_list)