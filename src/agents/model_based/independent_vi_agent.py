# src/agents/model_based/dtde_vi.py
from collections import defaultdict
import itertools
import numpy as np
import pickle
import random
from typing import Dict, Tuple, List, Optional, Union

from ..base_agent import ModelBasedAgent
from tiny_game import DecPOMP_Rework

class Independent_VI_Agent(ModelBasedAgent):
    """
    Decentralized Model-Based Value Iteration Agent.
    Assumes the partner acts UNIFORMLY RANDOMLY (Level-1 Reasoning).
    """
    def __init__(
        self, 
        num_cards: int, 
        num_actions: int,
        env: DecPOMP_Rework,
        # Hyperparameters
        partner_epsilon : float = 1.0,
        *args, **kwargs
    ):
        super().__init__(num_cards, num_actions)
        self.env : DecPOMP_Rework = env
        self.partner_epsilon = partner_epsilon

        # Tables: Observation -> Value/Action
        self.policy: Dict[tuple, int] = {}
        self.v_values: Dict[tuple, float] = defaultdict(float)
        
        self._init_tables(**kwargs)
        return
    
    def _init_tables(self, model_path: Optional[str] = None, *args, **kwargs):
        """
        Initialize tables either from disk or from scratch.
        """
        if model_path is not None:
            self.load(model_path)
            return
        
        all_visible_histories = self.env.get_all_observations(as_tensor=False)

        # Initialize
        for obs in all_visible_histories:
            self.v_values[obs] = 0.0
            self.policy[obs] = random.randint(0, self.num_actions - 1)
        return
    
    def train(self)->float:
        """
         
        """
        max_delta = 0.0
        all_obs = self.env.get_all_observations(as_tensor=False)
        for obs in all_obs:
            old_v = self.v_values[obs]
            q_values = np.zeros(self.num_actions)
            for action in range(self.num_actions):
              q_values[action] = self._calculate_expected_return(obs, action)
            
            # Get Best Action
            best_value = np.max(q_values)
            best_actions = np.flatnonzero(q_values == best_value)
            best_action = int(np.random.choice(best_actions))
            
            # Update Value
            self.v_values[obs] = best_value
            self.policy[obs] = best_action

            delta = abs(old_v - self.v_values[obs])
            if delta > max_delta:
                max_delta = delta    
        return max_delta
    
    def _calculate_expected_return(self, history: tuple, action: int) -> float:
        """
        Calculates expected return.
        Dynamically determines if it acts as First Mover (P0) or Last Mover (P1)
        based on the observation structure.
        """
        total_payoff = 0.0
        scenarios = 0
        NULL = self.env.NULL_VALUE

        is_player_0 = (history[0] == NULL)
        is_player_1 = (history[1] == NULL)
        if is_player_0 and is_player_1:
            raise ValueError("Invalid observation: Both cards hidden")
        
        range_c0 = range(self.env.num_cards) if history[0] == NULL else [history[0]]
        range_c1 = range(self.env.num_cards) if history[1] == NULL else [history[1]]

        for c0 in range_c0:
            for c1 in range_c1:
                
                # CASE A: Acting as Player 1 (Last Mover)
                if is_player_1:
                    a0_prev = history[2] 
                    payoff = self.env.payoffs[c0, c1, a0_prev, action]
                    total_payoff += payoff
                    scenarios += 1
                
                # CASE B: Acting as Player 0 (First Mover)
                elif is_player_0:
                    # Possible Rewards
                    possible_payoffs = []
                    for a1_response in range(self.num_actions):
                        possible_payoffs.append(self.env.payoffs[c0, c1, action, a1_response])
                    max_payoff = max(possible_payoffs)
                    avg_payoff = sum(possible_payoffs) / len(possible_payoffs)

                    # I play 'action'. Partner (P1) plays next.
                    sum_p1_responses = 0.0
                    for a1_response in range(self.num_actions):
                        #
                        sum_p1_responses += self.env.payoffs[c0, c1, action, a1_response]
                    expected_val = (
                        (1.0 - self.partner_epsilon) * max_payoff + 
                        (self.partner_epsilon) * avg_payoff
                    )
                    total_payoff += expected_val
                    scenarios += 1

        return total_payoff / max(1, scenarios)
    


    def act(self, input_state: tuple) -> int:
        return self.policy.get(input_state, 0) # Default 0 if unseen

    def save_transition(self, *args):
        pass # Model-based does not learn from transitions

    def save(self, filepath: str):
        """
        Saves both the Policy and V-Values to a pickle file.
        """
        # Convert defaultdict to standard dict for safe pickling
        data = {
            "policy": self.policy,
            "v_values": dict(self.v_values) 
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error saving model for agent {self.agent_id}: {e}")

    def load(self, filepath: str):
        """
        Loads both the Policy and V-Values from a pickle file.
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Basic validation
            if not isinstance(data, dict) or "policy" not in data or "v_values" not in data:
                raise ValueError("Loaded file does not contain valid policy and v_values.")

            # Restore Policy
            self.policy = data["policy"]
            
            # Restore V-Values (Reconstruct defaultdict)
            self.v_values = defaultdict(float)
            self.v_values.update(data["v_values"])
            
        except FileNotFoundError:
            print(f"File not found: {filepath}. Starting from scratch.")
        except Exception as e:
            print(f"Error loading agent {self.agent_id}: {e}")