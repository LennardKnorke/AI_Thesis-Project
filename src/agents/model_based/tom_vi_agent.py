import numpy as np
import torch
from ..base_agent import ModelBasedAgent
from .tom_net import ToMnet_WorldModel

class ToM_VI_Agent(ModelBasedAgent):
    def __init__(self, 
                 agent_id, num_cards, num_actions, env, 
                 tom_model_path=None, # Path to pre-trained weights
                 *args, **kwargs):
        super().__init__(agent_id, num_cards, num_actions)
        self.env = env
        
        # Initialize the World Model
        self.tom_net = ToMnet_WorldModel(
            num_cards_per_hand=2,
            num_card_types=num_cards, # 2 types (0, 1)
            num_actions=num_actions
        )
        
        if tom_model_path:
            self.load_world_model(tom_model_path)
            
        self.policy = {}
        self.v_values = {}
        
        # Pre-calculate possible observations for planning
        self._init_tables()

    def load_world_model(self, path):
        self.tom_net.load_state_dict(torch.load(path))
        self.tom_net.eval() # Set to inference mode

    def _init_tables(self):
        # Initialize standard VI tables...
        pass 

    def train(self):
        # Standard Value Iteration Loop...
        # Calls _calculate_expected_return
        pass

    def _calculate_expected_return(self, history: tuple, action: int) -> float:
        """
        Calculates EV using ToMnet for Partner Prediction.
        """
        total_payoff = 0.0
        scenarios = 0
        NULL = self.env.NULL_VALUE

        # Identify Role
        is_player_0 = (history[0] == NULL)
        
        # Ranges for Hidden Cards (My cards)
        range_c0 = range(self.env.num_cards) if history[0] == NULL else [history[0]]
        range_c1 = range(self.env.num_cards) if history[1] == NULL else [history[1]]

        for c0 in range_c0:
            for c1 in range_c1:
                
                # CASE A: I am Last Mover (No Prediction needed)
                if not is_player_0:
                    a0_prev = history[2]
                    payoff = self.env.payoffs[c0, c1, a0_prev, action]
                    total_payoff += payoff
                    scenarios += 1
                
                # CASE B: I am First Mover (PREDICTION NEEDED)
                else:
                    # I play 'action'. Partner acts next.
                    # ----------------------------------------------------
                    # MACHINE THEORY OF MIND INTEGRATION
                    # ----------------------------------------------------
                    
                    # 1. Prepare Inputs for ToMnet
                    # Partner sees: [c0 (my card), NULL, action (what I just did)]
                    # Actually, ToMnet takes (PartnerHand, History).
                    
                    # Partner's Hand: [c1_0, c1_1] (Indices 2,3 in matrix dims)
                    # Wait, in Medium Game, Partner ID is 1. They hold cards c1.
                    # I (P0) can SEE c1. So I pass c1 to the Net.
                    
                    # Convert c1 index to tensor (Assuming 2-card hand logic from Medium)
                    # Note: Medium Env simplifies "Hand" to just card values in matrix for now.
                    # Let's assume c1 represents the full hand for this matrix simplified logic.
                    
                    # Construct History: Previous Hist + My Action
                    # History in 'history' tuple is raw.
                    # We need to construct the tensor for the network.
                    
                    # For matrix logic simplification, we need the probability distribution
                    # P(a_response | I_played_action, I_hold_c0, Partner_holds_c1)
                    
                    # Query ToMnet:
                    with torch.no_grad():
                        # Prepare Tensors
                        # Partner Hand (Visible to me): c1 (Value)
                        # We might need to map matrix index back to hand representation if complex.
                        # Assuming simple case:
                        p_hand = torch.tensor([[c1, c1]], dtype=torch.long) # Placeholder logic
                        
                        # History: [action]
                        hist_tensor = torch.tensor([[action]], dtype=torch.long)
                        
                        # Get Probabilities
                        probs = self.tom_net.get_action_probs(p_hand, hist_tensor)
                        probs = probs[0].numpy() # [p_a0, p_a1, p_a2, p_a3]

                    # 2. Calculate Expectation
                    expected_p1_response = 0.0
                    for a1_response, prob in enumerate(probs):
                        val = self.env.payoffs[c0, c1, action, a1_response]
                        expected_p1_response += (val * prob)
                    
                    total_payoff += expected_p1_response
                    scenarios += 1

        return total_payoff / max(1, scenarios)