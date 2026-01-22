import numpy as np
import pickle
import random
from collections import defaultdict
from typing import Dict, Tuple, List, Optional

from ..base_agent import ModelBasedAgent
from tiny_game import DecPOMDP 

class Independent_VI_Agent(ModelBasedAgent):
    """
    Decentralized Model-Based Value Iteration Agent.
    Role-Agnostic: Learns optimal policies for both First and Last mover roles
    stored in a single policy dictionary.
    """
    
    def __init__(
        self, 
        num_cards: int, 
        num_actions: int, 
        env: DecPOMDP, 
        # agent_id is REMOVED
        partner_epsilon: float = 1.0,
        *args, **kwargs
    ):
        super().__init__(num_cards, num_actions)
        self.env = env 
        self.partner_epsilon = partner_epsilon
        self.NULL_VALUE = -1
        
        self.policy: Dict[tuple, int] = {}
        self.v_values: Dict[tuple, float] = defaultdict(float)
        
        # Generate State Space Internally
        self.all_observations = self._generate_state_space()
        
        self._init_tables(**kwargs)

    def _generate_state_space(self) -> List[tuple]:
        """
        Generates keys for both Player 0 and Player 1 contexts.
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

    def _init_tables(self, model_path: Optional[str] = None, *args, **kwargs):
        if model_path is not None:
            self.load(model_path)
            return
        
        for obs in self.all_observations:
            self.v_values[obs] = 0.0
            self.policy[obs] = random.randint(0, self.num_actions - 1)

    def train(self) -> float:
        """
        Executes one sweep over ALL observations (P0 and P1).
        """
        max_delta = 0.0
        
        for obs in self.all_observations:
            old_v = self.v_values[obs]

            # Calculate Q-Values
            q_values = np.zeros(self.num_actions)
            for action in range(self.num_actions):
                q_values[action] = self._calculate_expected_return(obs, action)

            # Greedy Update
            best_value = np.max(q_values)
            best_actions = np.flatnonzero(q_values == best_value)
            best_action = int(np.random.choice(best_actions))
            
            self.v_values[obs] = best_value
            self.policy[obs] = best_action

            delta = abs(old_v - self.v_values[obs])
            if delta > max_delta:
                max_delta = delta
                
        return max_delta

    def _calculate_expected_return(self, obs: tuple, action: int) -> float:
        total_payoff = 0.0
        scenarios = 0
        
        # Detect Role based on Observation Structure
        # P0: (NULL, c1) -> NULL at index 0
        # P1: (c0, NULL, a0) -> NULL at index 1
        is_player_0 = (obs[0] == self.NULL_VALUE)

        # Identify Hidden State Range
        if is_player_0:
            range_c0 = range(self.num_cards)
            c1_fixed = obs[1]
            range_c1 = [c1_fixed]
        else:
            c0_fixed = obs[0]
            range_c0 = [c0_fixed]
            range_c1 = range(self.num_cards)

        for c0 in range_c0:
            for c1 in range_c1:
                
                # --- CASE A: Last Mover (Player 1) ---
                if not is_player_0:
                    a0_prev = obs[2]
                    payoff = self.env.payoffs[c0, c1, a0_prev, action]
                    total_payoff += payoff
                    scenarios += 1
                
                # --- CASE B: First Mover (Player 0) ---
                else:
                    # Model P1 Response
                    p1_possible_payoffs = []
                    for a1_response in range(self.num_actions):
                        val = self.env.payoffs[c0, c1, action, a1_response]
                        p1_possible_payoffs.append(val)
                    
                    max_payoff = max(p1_possible_payoffs)
                    avg_payoff = sum(p1_possible_payoffs) / len(p1_possible_payoffs)
                    
                    expected_val = (
                        (1.0 - self.partner_epsilon) * max_payoff + 
                        (self.partner_epsilon) * avg_payoff
                    )
                    
                    total_payoff += expected_val
                    scenarios += 1

        return total_payoff / max(1, scenarios)

    def act(self, input_state: tuple) -> int:
        return self.policy.get(input_state, 0)
    
    def reset(self):
        """Resets policy to random for a new attempt."""
        for obs in self.all_observations:
            self.v_values[obs] = 0.0
            self.policy[obs] = random.randint(0, self.num_actions - 1)

    def save_transition(self, *args):
        pass

    def save(self, filepath: str):
        data = {"policy": self.policy, "v_values": dict(self.v_values)}
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error saving model: {e}")

    def load(self, filepath: str):
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.policy = data["policy"]
            self.v_values = defaultdict(float, data["v_values"])
        except Exception as e:
            print(f"Error loading model: {e}")