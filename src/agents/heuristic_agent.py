import numpy as np
import random
from tiny_hanabi.agent.actors import Actor
from .base_agent import BaseAgent

class HeuristicAgent(BaseAgent):
    def __init__(self, agent_id, num_actions, config=None):
        self.id = agent_id
        self.num_actions = num_actions
        self.preferred_action = None
        
        # Calculate greedy strategy from config
        if config and 'payoff_matrix' in config:
            self._calculate_greedy_strategy(config['payoff_matrix'])

    def _calculate_greedy_strategy(self, matrix):
        try:
            best_joint_action = np.unravel_index(np.argmax(matrix), matrix.shape)
            self.preferred_action = best_joint_action[self.id]
        except Exception:
            self.preferred_action = 0

    def act(self, observation):
        # 1. Check for 'suggestion' in observation (if your ToM agent sends hints later)
        if hasattr(observation, 'suggestion') and observation.suggestion is not None:
             return observation.suggestion
        
        # 2. Greedy/Optimistic
        if self.preferred_action is not None:
            return self.preferred_action
            
        # 3. Fallback
        return np.random.randint(0, self.num_actions)
        
    def update(self, batch):
        # Heuristic agent does not learn
        pass