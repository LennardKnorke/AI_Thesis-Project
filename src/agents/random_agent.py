

import numpy as np
import random

from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
    def __init__(self, agent_id, num_actions):
        # PolicyActor usually expects a 'policy' function, 
        # but for random we can override act directly or provide a uniform policy.
        self.id = agent_id
        self.num_actions = num_actions

    def act(self, observation):
        return self.act_random(observation)
    
    def act_normally(self, state):
        return self.act_random(state)

    def act_greedily(self, state):
        return self.act_random(state)
    
    def act_random(self, state, available_actions : tuple = None):
        if available_actions != None:
            return random.choice(available_actions)
        return random.randint(0, self.num_actions-1)

    def update(self, batch):
        # Random agent does not learn
        pass
    def reset(self):
        # Does not reset
        pass
    def train(self):
        return None
    def update_rates(self):
        pass
    def save_transition(self, observation, action, next_observation, reward, done):
        pass