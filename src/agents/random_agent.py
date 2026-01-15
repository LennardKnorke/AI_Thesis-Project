# /agents/random_agent.py

import numpy as np
import random

from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
    def act(self, observation):
        return random.randint(0, self.num_actions-1)

    def train(self):
        pass
    def save_transition(self):
        pass


###################################