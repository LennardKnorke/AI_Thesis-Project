# /agents/heuristic_agent.py

import numpy as np
import random
from tiny_hanabi.agent.actors import Actor
from .base_agent import BaseAgent

class HeuristicAgent(BaseAgent):
    def __init__(
            self,
            num_cards,
            num_actions,
            config=None
        ):
        super().__init__(num_cards, num_actions)
        self.NULL_VALUE = -1  # Padding for no previous action seen

        return
    def train(self):
        pass

    def save_transition(self):
        pass 
    
    def act(self, input_state: tuple) -> int:
        """
        input_state is [c0, c1, a0, a1] (masked).
        """
        partner_card = -1
        
        if input_state[0] == self.NULL_VALUE:
            # I am Player 0, looking at Player 1's card
            partner_card = input_state[1]
        else:
            # I am Player 1, looking at Player 0's card
            partner_card = input_state[0]
            
        # If for some reason both are NULL (shouldn't happen in this game), fallback
        if partner_card == self.NULL_VALUE:
            return 0 # Default fallback

        # 2. Heuristic Logic: Play the card value seen
        if partner_card < self.num_actions:
            return int(partner_card)
        return 0
