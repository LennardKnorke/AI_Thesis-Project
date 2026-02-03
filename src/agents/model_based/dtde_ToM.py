#agents/model_based/dtde_ToM.py
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from tiny_game import DecPOMDP

from ..base_agent import ModelBasedAgent, AgentList, BaseAgent


# --- GLOBAL CONSTANTS (for ToMnet input formatting) ---
OBS_DIM = 4             # Max observation vector length for Tiny Hanabi (e.g., [-1, C1, -1, -1])
MAX_SEQ_LEN = 4         # Max history length for LSTMs (e.g., [obs, act, obs, act])
PAST_EPISODES_CONTEXT = 5 # Number of past episodes the CharacterNet expects

class CharacterNet(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_agent_types):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, embedding_dim, batch_first=True)
        
        # --- INTERNAL AUXILIARY HEAD ---
        # Forces the embedding to contain "Identity" information
        self.identity_classifier = nn.Linear(embedding_dim, num_agent_types)
    
    def forward(self, past_episodes):
        # past_episodes: (Batch, Num_Episodes, Seq_Len, Feat_Dim)
        b, n_eps, seq, feat = past_episodes.size()
        
        # Flatten to process all episodes in parallel
        flat_input = past_episodes.view(-1, seq, feat) 
        
        # Run LSTM
        _, (h_n, _) = self.lstm(flat_input)
        episode_embeddings = h_n[-1] # (Batch*Num_Episodes, Emb_Dim)
        
        # Reshape back to separate episodes
        episode_embeddings = episode_embeddings.view(b, n_eps, -1)
        
        # Average Pooling: Create a general profile ($e_{char}$)
        e_char = torch.mean(episode_embeddings, dim=1) # (Batch, Emb_Dim)
        
        # Calculate Auxiliary Logits (for Loss calculation only)
        identity_logits = self.identity_classifier(e_char)
        
        return e_char, identity_logits


class MentalNet(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, embedding_dim, batch_first=True)
    
    def forward(self, current_history):
        # current_history: (Batch, Seq_Len, Feat_Dim)
        # Summarize the current interaction so far
        _, (h_n, _) = self.lstm(current_history)
        e_mental = h_n[-1] # (Batch, Emb_Dim)
        return e_mental


class ToM_WorldModel(nn.Module):
    def __init__(self, 
                 obs_dim, 
                 action_dim, 
                 num_agent_types, 
                 char_embed_dim=32,
                 mental_embed_dim=16):
        super().__init__()
        
        # Input Feature: Observation + OneHot(Action)
        self.input_dim = obs_dim + action_dim 
        
        # 1. Sub-Nets
        self.char_net = CharacterNet(self.input_dim, char_embed_dim, num_agent_types)
        self.mental_net = MentalNet(self.input_dim, mental_embed_dim)
        
        # 2. Prediction Trunk
        # Input: Character + Mental + Current_Observation (Context)
        self.trunk_input = char_embed_dim + mental_embed_dim + obs_dim
        self.trunk = nn.Sequential(
            nn.Linear(self.trunk_input, 64),
            nn.ReLU()
        )
        
        # 3. Heads
        # Head A: Action Prediction -> P(a^{-i}_t)
        self.action_head = nn.Linear(64, action_dim)
        
        # Head B: Observation Prediction -> P(o^{-i}_{t+1})
        # Note: We predict the *next* observation vector
        self.observation_head = nn.Linear(64, obs_dim)

    def forward(self, past_episodes, current_history, current_obs):
        """
        Args:
            past_episodes: (Batch, N_Eps, Seq, Feat)
            current_history: (Batch, Seq, Feat)
            current_obs: (Batch, Obs_Dim)
        """
        # 1. Get Embeddings (Character handles its own classification internally)
        e_char, identity_logits = self.char_net(past_episodes)
        e_mental = self.mental_net(current_history)
        
        # 2. Fusion
        combined = torch.cat([e_char, e_mental, current_obs], dim=1)
        features = self.trunk(combined)
        
        # 3. Predictions
        action_logits = self.action_head(features)
        next_obs_pred = self.observation_head(features)
        
        return action_logits, next_obs_pred, identity_logits


def _one_hot(idx, size):
    vec = np.zeros(size, dtype=np.float32)
    vec[idx] = 1.0
    return vec

# Helper function to pad sequence for LSTM
def _pad_sequence(seq_vectors, max_len, feat_dim):
    arr = np.array(seq_vectors, dtype=np.float32)
    if len(arr) == 0:
        return np.zeros((max_len, feat_dim), dtype=np.float32)
    if len(arr) < max_len:
        padding = np.zeros((max_len - len(arr), feat_dim), dtype=np.float32)
        arr = np.vstack([arr, padding])
    return arr[:max_len]



class DTDE_ToMBI_Agent(ModelBasedAgent):
    """
    Theory of Mind Backward Induction Agent.
    
    Implements a Belief-MDP solver where:
    1. Beliefs are updated using the ToMnet (Inverse Planning).
    2. Transitions are predicted using the ToMnet (Forward Simulation).
    3. Policy is optimized via Backward Induction (P1 -> P0).
    """
    def __init__(
        self, 
        num_cards: int, 
        num_actions: int, 
        env: DecPOMDP,
        world_model: ToM_WorldModel, # Pre-trained ToMnet
        device: str = "cpu",
        *args, **kwargs
    ):
        super().__init__(num_cards, num_actions)
        self.env = env
        self.tom_net = world_model
        self.device = device
        self.NULL_VALUE = -1
        
        self.policy: Dict[tuple, int] = {}
        self.v_values: Dict[tuple, float] = defaultdict(float)
        
        # Dimensions for ToMnet input formatting
        self.obs_dim = OBS_DIM 
        self.num_actions = num_actions # Ensure this is consistent
        self.step_feat_dim = self.obs_dim + self.num_actions # Input to LSTMs

        # 1. Generate State Space
        self.all_observations = self._generate_private_observation_space()
        
        # 2. Init Tables
        self._init_tables(**kwargs)
        return
    def _generate_private_observation_space(self) -> List[tuple]:
        """
        Generates the space of *private observations* for the focal agent.
        """
        observations = []
        # Focal P0 (sees opponent's card, own is NULL)
        # Pattern: (NULL, c1) -> e.g. (-1, 0), (-1, 1)
        for c1 in range(self.num_cards):
            observations.append((self.NULL_VALUE, c1))
        
        # Focal P1 (sees opponent's card and P0's action, own is NULL)
        # Pattern: (c0, NULL, a0) -> e.g. (0, -1, 0), (0, -1, 1)
        for c0 in range(self.num_cards):
            for a0 in range(self.num_actions):
                observations.append((c0, self.NULL_VALUE, a0))
        return observations

    def _init_tables(self, model_path: Optional[str] = None, *args, **kwargs):
        """
        Initializes policy and value tables, optionally loading from a file.
        """
        if model_path is not None:
            self.load(model_path)
            return
        for obs in self.all_observations:
            self.v_values[obs] = 0.0
            self.policy[obs] = random.randint(0, self.num_actions - 1)
        return

    def train(self) -> float:
        """
        Executes Backward Induction using the Belief-ToM framework.
        """
        max_delta = 0.0
        
        # Split observations by turn for BI order
        p1_obs_list = [o for o in self.all_observations if len(o) == 3] # P1 is the last mover
        p0_obs_list = [o for o in self.all_observations if len(o) == 2] # P0 is the first mover

        # --- STEP 1: SOLVE LAST MOVER (P1) ---
        # P1 needs to infer its own hidden card (c1) from P0's observed action (a0)
        # and then choose a1.
        for obs in p1_obs_list: # obs = (c0, -1, a0)
            delta = self._optimize_node(obs, is_focal_p1=True)
            if delta > max_delta: max_delta = delta

        # --- STEP 2: SOLVE FIRST MOVER (P0) ---
        # P0 needs to predict P1's response (a1) given its chosen action (a0)
        # and its own hidden card (c0).
        for obs in p0_obs_list: # obs = (-1, c1)
            delta = self._optimize_node(obs, is_focal_p1=False)
            if delta > max_delta: max_delta = delta
            
        return max_delta

    def _optimize_node(self, obs: tuple, is_focal_p1: bool) -> float:
        """
        Calculates optimal action and updates value for a given observation node.
        """
        old_v = self.v_values[obs]
        q_values = np.zeros(self.num_actions) # Q-values for the focal agent's actions
        
        for action in range(self.num_actions): # Iterate over focal agent's possible actions
            q_values[action] = self._calculate_expected_return_with_tom(obs, action, is_focal_p1)
            
        best_val = np.max(q_values)
        best_actions = np.flatnonzero(q_values == best_val)
        best_action = int(np.random.choice(best_actions)) # Break ties randomly
        
        self.v_values[obs] = best_val
        self.policy[obs] = best_action
        
        return abs(best_val - old_v)

    def _calculate_expected_return_with_tom(self, focal_obs: tuple, focal_action: int, is_focal_p1: bool) -> float:
        """
        Calculates the expected return for the focal agent taking `focal_action`
        from `focal_obs`, using the ToMnet for partner predictions/beliefs.
        """
        total_expected_payoff = 0.0
        
        # --- CASE A: Focal Agent is P1 (Last Mover) ---
        if is_focal_p1: # focal_obs = (c0_actual, -1, a0_actual)
            c0_actual = focal_obs[0]
            a0_actual = focal_obs[2] # P0's action already observed
            
            # P1 needs to infer its own hidden card (c1) based on a0_actual.
            # Assume uniform prior over c1.
            unnormalized_beliefs = []
            
            for c1_hyp in range(self.num_cards): # Iterate over P1's hypothetical hidden cards
                # Construct what P0 (the partner) *would have observed* to take a0_actual
                # P0's view at its turn: (-1, c1_hyp)
                partner_obs_at_turn = [self.NULL_VALUE, c1_hyp] + [self.NULL_VALUE]*(self.obs_dim - 2)
                
                # P0's history at its turn is empty (first mover in episode)
                partner_history_so_far = [] 
                
                # Query ToMnet: "Given P0 saw c1_hyp, how likely is P0 to take a0_actual?"
                # This gives P(a0_actual | P0's view of c1_hyp)
                prob_a0_given_c1_hyp, _ = self._query_tom(
                    partner_current_obs=np.array(partner_obs_at_turn),
                    partner_history_so_far=partner_history_so_far # Empty
                )
                
                # Get probability for the specific action a0_actual
                likelihood_a0 = prob_a0_given_c1_hyp[a0_actual]
                unnormalized_beliefs.append(likelihood_a0)
            
            # Normalize the beliefs over c1
            sum_beliefs = sum(unnormalized_beliefs)
            if sum_beliefs == 0:
                # Fallback to uniform if ToMnet provides no distinguishing info
                beliefs_c1 = [1.0 / self.num_cards] * self.num_cards
            else:
                beliefs_c1 = [b / sum_beliefs for b in unnormalized_beliefs]
            
            # Calculate Expected Payoff for P1's action (focal_action) over inferred c1
            for c1_inferred, belief_prob in enumerate(beliefs_c1):
                # Payoff is for (c0_actual, c1_inferred, a0_actual, P1's focal_action)
                payoff = self.env.payoffs[c0_actual, c1_inferred, a0_actual, focal_action]
                total_expected_payoff += (payoff * belief_prob)

        # --- CASE B: Focal Agent is P0 (First Mover) ---
        else: # focal_obs = (-1, c1_actual)
            c1_actual = focal_obs[1] # P1's card (hidden from P0 initially)
            
            # P0 has no info about c0 yet, so assume uniform prior over c0.
            for c0_hyp in range(self.num_cards): # Iterate over P0's hypothetical hidden cards
                # P0 chooses focal_action (a0). Now P1 (partner) will act.
                # Construct what P1 (the partner) *would observe*
                # P1's view at its turn: (c0_hyp, -1, focal_action)
                partner_obs_at_turn = [c0_hyp, self.NULL_VALUE, focal_action] + [self.NULL_VALUE]*(self.obs_dim - 3)
                
                # P1's history at its turn: only P0's action (focal_action)
                # This needs to be formatted as (obs_vector, one_hot_action)
                p0_obs_for_hist = np.array([self.NULL_VALUE, c1_actual] + [self.NULL_VALUE]*(self.obs_dim - 2))
                p0_act_oh = _one_hot(focal_action, self.num_actions)
                partner_history_so_far = [np.concatenate([p0_obs_for_hist, p0_act_oh])]
                
                # Query ToMnet: "Given P1's view, what is P1's action distribution?"
                # This gives P(a1 | P1's view)
                p1_action_dist, _ = self._query_tom(
                    partner_current_obs=np.array(partner_obs_at_turn),
                    partner_history_so_far=partner_history_so_far # History includes P0's action
                )
                
                # Sum over P1's possible responses
                expected_response_value = 0.0
                for a1_resp in range(self.num_actions):
                    prob_a1 = p1_action_dist[a1_resp]
                    # Payoff is for (c0_hyp, c1_actual, focal_action, a1_resp)
                    payoff = self.env.payoffs[c0_hyp, c1_actual, focal_action, a1_resp]
                    expected_response_value += (payoff * prob_a1)
                
                # Average over c0_hyp (uniform prior for c0)
                total_expected_payoff += (expected_response_value / self.num_cards) # P(c0) is 1/num_cards

        return total_expected_payoff

    # --- ToM Helper Methods ---

    def _query_tom(self, 
                   partner_current_obs: np.ndarray, 
                   partner_history_so_far: List[np.ndarray]
                  ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Formats inputs and queries the Neural World Model.
        
        Args:
            partner_current_obs: What the *partner* observes at its decision point. (Obs_DIM,)
            partner_history_so_far: Sequence of (Obs_Vector + OneHot_Action) for partner's past steps.
        
        Returns:
            Tuple: (action_probabilities (np.ndarray), next_observation_prediction (np.ndarray))
        """
        self.tom_net.eval()
        with torch.no_grad():
            # 1. Past Episodes (Assume zero for BI, relying on general character type)
            # Shape: (Batch=1, N_Eps=PAST_EPISODES_CONTEXT, Seq_Len=MAX_SEQ_LEN, Feat_Dim)
            past_eps_input = torch.zeros((1, PAST_EPISODES_CONTEXT, MAX_SEQ_LEN, self.step_feat_dim), 
                                         dtype=torch.float32).to(self.device)
            
            # 2. Current History
            # Shape: (Batch=1, Seq_Len=MAX_SEQ_LEN, Feat_Dim)
            curr_hist_padded = _pad_sequence(partner_history_so_far, MAX_SEQ_LEN, self.step_feat_dim)
            curr_hist_input = torch.tensor(curr_hist_padded, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # 3. Current Observation (Partner's current view)
            # Shape: (Batch=1, Obs_Dim)
            current_obs_input = torch.tensor(partner_current_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # 4. Inference
            action_logits, next_obs_pred, _ = self.tom_net(
                past_eps_input, curr_hist_input, current_obs_input
            )
            
            action_probs = torch.softmax(action_logits, dim=-1)
            
        return action_probs[0].cpu().numpy(), next_obs_pred[0].cpu().numpy()

    # --- Standard Interfaces ---
    def act(self, input_state: tuple, exploit: bool = False) -> int:
        """
        During execution, the agent acts based on its pre-computed policy.
        """
        # Policy is a mapping from private observation to action
        return self.policy.get(input_state, 0)
    
    def save(self, filepath: str):
        """Saves the learned policy and value function."""
        data = {"policy": self.policy, "v_values": dict(self.v_values)}
        with open(filepath, 'wb') as f: 
            pickle.dump(data, f)
    
    def load(self, filepath: str): 
        """Loads a pre-trained policy and value function."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.policy = data.get("policy", {})
            self.v_values = defaultdict(float)
            self.v_values.update(data.get("v_values", {}))
            print(f"Loaded ToMBI policy from {filepath}")
        except FileNotFoundError:
            print(f"File not found: {filepath}. Initializing random policy.")
            self._init_tables()
        except Exception as e:
            print(f"Error loading ToMBI policy from {filepath}: {e}. Initializing random policy.")
            self._init_tables()

    def save_transition(self, *args): 
        # Model-based agent, doesn't directly store transitions for Q-learning
        pass

    def reset(self): 
        # For new training attempts, reset policy to random for exploration
        self._init_tables()


class DTDE_ToMBI_List(AgentList):
    """
    A specialized AgentList for the ToM Agent's training.
    
    It always contains ONE DTDE_ToMBI_Agent (the focal agent) and ONE partner agent.
    Only the focal ToM agent is trained; the partner's policy is fixed.
    """
    def __init__(self, focal_tom_agent: DTDE_ToMBI_Agent, initial_partner_agent: BaseAgent):
        # Validate types
        if not isinstance(focal_tom_agent, DTDE_ToMBI_Agent):
            raise TypeError(f"Focal agent must be an instance of DTDE_ToMBI_Agent. Got {type(focal_tom_agent).__name__}.")
        if not isinstance(initial_partner_agent, BaseAgent):
            raise TypeError(f"Partner agent must be an instance of BaseAgent. Got {type(initial_partner_agent).__name__}.")
            
        # Store agents internally. The 'list' aspect of AgentList is just for iteration.
        self._focal_agent = focal_tom_agent
        self._partner_agent = initial_partner_agent
        
        # Initialize the base AgentList with the current pair.
        # This is the actual list that `run_episode` will iterate over.
        # Player 0 is typically the focal agent during planning here.
        super().__init__([self._focal_agent, self._partner_agent])

    @property
    def centralized_planning(self) -> bool:
        # ToM BI is decentralized planning from the focal agent's perspective.
        return False

    def act(self, observations: List[Any]) -> List[int]:
        """
        Queries the current two agents for their actions.
        """
        if len(observations) != 2: # Always expect 2 observations for 2 agents
            raise ValueError(f"Expected 2 observations for 2 agents, got {len(observations)}.")

        joint_action = [
            self._focal_agent.act(observations[0]),  # P0 is focal
            self._partner_agent.act(observations[1])  # P1 is partner
        ]
        return joint_action

    def train(self) -> float:
        """
        Only trains (or plans for) the focal ToM agent. The partner is fixed.
        """
        # The 'train' method of the DTDE_ToMBI_Agent will perform Backward Induction
        # for *itself*, considering the world model's predictions of the partner.
        loss = self._focal_agent.train()
        return loss

    def reset(self):
        """
        Resets both the focal ToM agent and the partner agent.
        """
        self._focal_agent.reset()
        self._partner_agent.reset()

    def switch_partner(self, new_partner_agent: BaseAgent):
        """
        Switches the partner agent for the ToM agent.
        The list of agents managed by AgentList is updated.
        """
        if not isinstance(new_partner_agent, BaseAgent):
            raise TypeError(f"New partner must be an instance of BaseAgent. Got {type(new_partner_agent).__name__}.")
        
        self._partner_agent = new_partner_agent
        # Update the internal list of the base AgentList class
        self.clear() # Clear the old agents
        self.append(self._focal_agent)
        self.append(self._partner_agent)
        # Note: If order matters (P0/P1), ensure it's (focal, partner)

    def get_focal_agent(self) -> DTDE_ToMBI_Agent:
        return self._focal_agent

    def get_partner_agent(self) -> BaseAgent:
        return self._partner_agent