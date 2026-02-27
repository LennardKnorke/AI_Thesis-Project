# agents/model_based/dtde_ToM.py
from collections import defaultdict, deque
from copy import deepcopy
import random
import os
from typing import Any

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from tiny_game import DecPOMDP, MyHanabi, Game, get_all_possible_states, get_all_possible_histories

from ..base_agent import ModelBasedAgent, AgentList, BaseAgent


def _encode_joint_observation(obs : int|tuple[int, int], obs_dim : int, env : Game, s0 : list|tuple|None = None, is_p0_turn : bool= False):
    vec = np.zeros(obs_dim, dtype=np.float32)
    num_actions = env.num_actions
    num_cards = env.num_cards
    if isinstance(env, DecPOMDP):
        if isinstance(obs, int):
            vec[obs] = 1.0
        else:
            for i, o in enumerate(obs):
                idx = num_actions + (i * num_cards) + o
                vec[idx] = 1.0
    else:
        if isinstance(obs, int):
            vec[obs] = 1.0
            if obs < 2:
                own_cards = s0[:2] if is_p0_turn else s0[2:]
                discarded_card = own_cards[obs]
                player_offset = 0 if is_p0_turn else num_cards * 2
                idx = num_actions + (num_cards * obs)+ player_offset + discarded_card
                vec[idx] = 1.0
        else:
            for i, o in enumerate(obs):
                idx = num_actions + (i * num_cards) + o
                vec[idx] = 1.0
    return vec



def _encode_observation(obs : int|tuple[int, int], obs_dim : int, env : Game, s0 : list|tuple|None = None, is_p0_turn : bool = False):
    """
    Encode observation from a player's perspective.
    
    For DecPOMDP:
        - If tuple: initial card(s) seen from partner
        - If int: action taken (which is also the observation)    
    For MyHanabi:
        - If tuple/list: initial cards (players see partner's cards)
        - If int: action taken, which may reveal a card if it was a discard
    """
    vec = np.zeros(obs_dim, dtype=np.float32)
    num_actions = env.num_actions
    num_cards = env.num_cards
    if isinstance(env, DecPOMDP):
        if isinstance(obs, int):
            vec[obs] = 1.0
        else:
            o = obs[0]
            vec[num_actions + o] = 1.0
    else:
        if isinstance(obs, int):
            vec[obs] = 1.0
            if obs < 2:
                own_cards = s0[:2] if is_p0_turn else s0[2:]
                discarded_card = own_cards[obs]
                idx = num_actions + (num_cards * obs) + discarded_card
                vec[idx] = 1.0
        else:
            for i, o in enumerate(obs):
                idx = num_actions + (i * num_cards) + o
                vec[idx] = 1.0
    return vec


def _encode_action(action : int, action_dim : int):
    vec = np.zeros(action_dim, dtype=np.float32)
    vec[action] = 1.0
    return vec


class CharacterNet(nn.Module):
    """
    Input:
        -N past joint Histories
    Output:
        - embedding/prediction of identification
    """
    def __init__(self, input_dim, embedding_dim, num_agent_types):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, embedding_dim, batch_first=True)
        
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
    """
    Input:
        - Output CharNet
        - Current step-t joint/public history
    Output:
        - Embedding
    """
    def __init__(self, input_dim, num_agent_types, embedding_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim + num_agent_types, embedding_dim, batch_first=True)
    
    def forward(self, current_history, identification_logits):
        # Expand agent identity across sequence dimension
        id_expanded = identification_logits.unsqueeze(1).expand(-1, current_history.size(1), -1)
        x_mental = torch.cat([current_history, id_expanded], dim=2)

        # LSTM forward
        _, (h_n, _) = self.lstm(x_mental)
        e_mental = h_n[-1]  # (Batch, Emb_Dim)
        return e_mental


class ToM_WorldModel(nn.Module):
    """
    Input:
        - Output CharNet
        - Output MentalNet
        - Private agent_{i} action_{t}
        - Private agent_{i} observation_{t+1}
    Outputs:
        - Private Agent_{not i} Action_{t}
        - Private Agent_{not i} Observation_{t+1}
    """
    def __init__(self, 
                 obs_dim : int,
                 joint_obs_dim : int,
                 action_dim : int, 
                 num_agent_types : int = 4,
                 max_seq_len : int = 8,
                 past_episode_context : int = 5,
                 char_embed_dim : int = 32,
                 mental_embed_dim : int = 16,
                 trunk_dim : int = 64):
        super().__init__()       
        # 1. Sub-Nets
        self.char_net = CharacterNet(joint_obs_dim, char_embed_dim, num_agent_types)
        self.mental_net = MentalNet(joint_obs_dim, num_agent_types, mental_embed_dim)
        
        # 2. Prediction Trunk
        self.trunk_input_dim = char_embed_dim + mental_embed_dim + obs_dim + action_dim
        
        self.trunk = nn.Sequential(
            nn.Linear(self.trunk_input_dim, trunk_dim),
            nn.ReLU(),
            nn.Linear(trunk_dim, trunk_dim),
            nn.ReLU()
        )
        
        # 3. Heads
        # Head A: Action Prediction -> P(a^{-i}_t)
        self.action_head = nn.Linear(trunk_dim, action_dim)
        
        # Head B: Observation Prediction
        self.observation_head = nn.Linear(trunk_dim, obs_dim)

    def forward(self, past_episodes, current_history, current_obs, own_action):
        """
        Args:
            past_episodes: (Batch, N_Eps, Seq, Feat)
            current_history: (Batch, Seq, Feat)
            current_obs: (Batch, Obs_Dim)
        """
        # Character embedding
        e_char, identity_logits = self.char_net(past_episodes)

        # Mental embedding
        e_mental = self.mental_net(current_history, identity_logits)
        
        # Trunk features
        x_trunk = torch.cat([e_char, e_mental, current_obs, own_action], dim=1)
        features = self.trunk(x_trunk)
        
        # Predictions
        action_logits = self.action_head(features)
        next_obs_pred = self.observation_head(features)
        
        return action_logits, next_obs_pred, identity_logits


PAST_EPISODES_CONTEXT = 5
DEFAULT_NUM_AGENT_TYPES = 4 # Based on the number of baseline experiments used for WM training
N_DECIMALS_FOR_BELIEF = 5 # For robust hashing of belief probabilities (e.g., 0.12345)


class DTDE_ToMBI_Agent(ModelBasedAgent):
    """
    Theory of Mind Backward Induction Agent.
    
    A single agent instance capable of planning and acting for both player roles (0 and 1)
    within a unified belief space.
    """
    def __init__(
        self, 
        env: Game,
        num_cards: int, 
        num_actions: int, 
        world_model: ToM_WorldModel, # Pre-trained ToMnet
        world_model_config: dict[str, Any], # Config to extract feat_dim, max_seq_len, etc.
        device: str = "cpu",
        gamma: float = 0.99, # Discount factor for planning
        *args, **kwargs
    ):
        super().__init__(env, num_cards, num_actions)
        self.is_decpomdp = isinstance(self.env, DecPOMDP)
        self.is_myhanabi = isinstance(self.env, MyHanabi)
        self.world_model = world_model.to(device)
        self.world_model.eval() # Set to eval mode for inference
        self.device = device
        self.gamma = gamma

        # Relevant for indexeing
        self.num_cards = self.env.num_cards
        self.num_actions = self.env.num_actions
        self.num_cards_in_hand = 1 if self.is_decpomdp else 2

        # Extract World Model configuration parameters
        self.feat_dim = world_model_config['feat_dim']
        self.max_seq_len = world_model_config['max_seq_len']
        self.action_output_dim = world_model_config['action_output_dim']
        self.num_agent_types = world_model_config.get('num_agent_types', DEFAULT_NUM_AGENT_TYPES)

        # All possible ground-truth joint histories
        self.all_possible_observations = sorted(get_all_possible_histories(self.env), key=len, reverse=True)
        self.all_joint_histories = sorted(get_all_possible_states(self.env), key=len, reverse=True)
        
        self.private_to_public_histories : dict[tuple, dict[tuple, float]] = {}
        self.consistent_joint_histories_cache = {}
        
        # Unified internal states
        self.v_values: dict[tuple, float] = {} 
        self.policy: dict[tuple, int] = {}
        self.legal_actions_cache = {}

        self._init_legal_actions()
        self._init_tables()
        self._map_histories_to_states()
        return

    def _init_legal_actions(self):
        for obs in self.all_possible_observations:
            if self.is_decpomdp:
                self.legal_actions_cache[obs] = tuple(range(self.num_actions))
            elif self.is_myhanabi:
                _, legal_actions_tuple = self.env.num_legal_actions(obs)
                self.legal_actions_cache[obs] = legal_actions_tuple
            else:
                raise ValueError("No valid environment proided")
    
    def _init_tables(self, model_path : str|None = None):
        if model_path:
            self.load()
            return
        
        for obs in self.all_possible_observations:
            legal_actions = self.legal_actions_cache.get(obs, ())
            if legal_actions:
                q_values = np.zeros(self.num_actions, dtype=np.float32)
                for a_idx in legal_actions:
                    q_values[a_idx] = 4.0 # Heuristic initial Q-value
                self.v_values[obs] = q_values
                self.policy[obs] = random.choice(legal_actions) # Random initial policy
            else:
                self.v_values[obs] = np.full(self.num_actions, -np.inf, dtype=np.float32) # No legal actions
                self.policy[obs] = None

    def _mask_state(self, joint_history: tuple, player_id: int) -> tuple:
        """
        """
        s_list = list(joint_history)
        
        if self.is_decpomdp:
            if player_id == 0:
                s_list[0] = -1
            else:
                s_list[1] = -1

        elif self.is_myhanabi:
            if player_id == 0:
                s_list[0] = -1; s_list[1] = -1
            else:
                s_list[2] = -1; s_list[3] = -1
        return tuple(s_list)
    
    def _get_current_player_id(self, obs : tuple)->int:
        return len(obs) % 2
    

    def _get_consistent_joint_histories(self, obs: tuple, focal_player_id: int) -> list[tuple]:
        """
        Returns all possible ground-truth joint histories consistent with the focal agent's observation `obs`.
        Uses caching for speed.
        """
        cache_key = (obs, focal_player_id)
        if cache_key in self.consistent_joint_histories_cache:
            return self.consistent_joint_histories_cache[cache_key]
        
        consistent_histories = []
        
        num_actions_in_obs = len(obs) - self.num_card_slots_in_obs

        possible_jhs_by_len = [jh for jh in self.all_joint_histories 
                               if len(jh) == (self.num_card_slots_in_obs + num_actions_in_obs)]

        for joint_hist in possible_jhs_by_len:
            derived_obs = self._mask_state(joint_hist, focal_player_id)
            if derived_obs == obs:
                consistent_histories.append(joint_hist)
        
        self.consistent_joint_histories_cache[cache_key] = consistent_histories
        return consistent_histories
    

    def _map_histories_to_states(self):
        self.private_to_public_histories = defaultdict(dict) # Reset for fresh computation
        self.world_model.eval() # Ensure World Model is in evaluation mode

        start_length_obs = 2 if self.is_decpomdp else 4
        for focal_obs in reversed(self.all_possible_observations):
            focal_player_id = len(focal_obs) % 2
            partner_player_id = 1 - focal_player_id

            consistent_joint_histories = self._get_consistent_joint_histories(focal_obs, focal_player_id)

            if not consistent_joint_histories:
                continue
            
            likelihood_scores_for_obs = {}
            for jh in consistent_joint_histories:
                avg_nll_across_partner_types = 0.0
                
                # Average likelihood over all possible partner types
                for partner_type_id in range(self.num_agent_types):
                    temp_env = deepcopy(self.env)
                    
                    initial_deal = jh[:self.num_card_slots_in_obs]
                    try:
                        temp_env.reset(list(initial_deal))
                    except (AssertionError, ValueError):
                        # This joint history is invalid at its start. Assign infinite NLL.
                        avg_nll_across_partner_types += float('inf') 
                        continue

                    actions_in_jh = jh[self.num_card_slots_in_obs:]
                    nll_for_type_and_jh = 0.0
                    num_partner_action_predictions = 0
                    
                    # This `current_focal_history_list` tracks the focal agent's observation sequence 
                    # for the MentalNet input, built as we simulate this `jh`.
                    current_focal_history_list = [] 
                    
                    # Store history of (focal_obs, focal_action) pairs to get context for WM.
                    # This list holds the focal observations *before* focal acts.
                    focal_obs_history_for_wm_input = []
                    # This list holds the focal actions *taken*.
                    focal_action_history_for_wm_input = []
                    
                    # Iterate through the sequence of actions that constitute this `jh`
                    for k in range(len(actions_in_jh)):
                        current_player_of_turn = self._get_current_player_id(tuple(temp_env.history))
                        action_taken_in_this_turn = actions_in_jh[k]

                        # Focal agent's actual observation *before* this action in `temp_env`
                        focal_obs_before_this_action = self._mask_state(tuple(temp_env.history), focal_player_id)
                        current_focal_history_list.append(_encode_observation(focal_obs_before_this_action, self.feat_dim, self.env))
                        
                        # If it's the partner's turn in this hypothetical JH, we use WM to predict their action
                        # and compare it to the action actually taken in the JH (`action_taken_in_this_turn`).
                        if current_player_of_turn == partner_player_id:
                            # To predict partner's action, WM needs focal's observation *before focal acted* and *focal's last action*.
                            # This means we look at the state `k-1` where focal last acted.
                            if len(focal_obs_history_for_wm_input) > 0 and len(focal_action_history_for_wm_input) > 0:
                                wm_focal_obs_prev = focal_obs_history_for_wm_input[-1]
                                wm_focal_action_prev = focal_action_history_for_wm_input[-1]
                                
                                wm_past_eps, wm_curr_hist, wm_curr_obs_tensor, wm_own_act_tensor = self._prepare_wm_inputs(
                                    focal_obs_at_step=wm_focal_obs_prev, # Focal's obs *before* focal's last action
                                    focal_action_at_step=wm_focal_action_prev, # Focal's last action
                                    partner_type_id=partner_type_id,
                                    current_focal_history_list=current_focal_history_list[:-1] # Focal mental history up to wm_focal_obs_prev
                                )
                                
                                with torch.no_grad():
                                    action_logits, _, _ = self.world_model(wm_past_eps, wm_curr_hist, wm_curr_obs_tensor, wm_own_act_tensor)
                                
                                # Calculate NLL of the *actual partner action* (from jh) given WM's prediction
                                nll = F.cross_entropy(action_logits, 
                                                      torch.tensor([action_taken_in_this_turn], device=self.device)).item()
                                nll_for_type_and_jh += nll
                                num_partner_action_predictions += 1
                        
                        # After potential prediction, simulate the action taken in this turn to advance `temp_env.history`
                        try:
                            temp_env.step(action_taken_in_this_turn)
                            if current_player_of_turn == focal_player_id:
                                # Record focal's state and action for potential future WM input
                                focal_obs_history_for_wm_input.append(focal_obs_before_this_action)
                                focal_action_history_for_wm_input.append(action_taken_in_this_turn)
                        except ValueError:
                            # If `jh` contains an illegal action at any point, it's an impossible history.
                            nll_for_type_and_jh = float('inf')
                            break # Exit inner loop for this JH

                    # If this joint history was valid and some partner actions were predicted
                    if nll_for_type_and_jh != float('inf'):
                        if num_partner_action_predictions > 0:
                            avg_nll_across_partner_types += (nll_for_type_and_jh / num_partner_action_predictions)
                        else:
                            # No partner actions in this history to predict (e.g., very short history or focal agent always acts first)
                            # Assign a neutral NLL (0.0 implies perfect prediction, which might be an overestimation)
                            avg_nll_across_partner_types += 0.0
                    else:
                        avg_nll_across_partner_types += float('inf') # Invalid JH for this type

                # Store the average NLL across partner types for this JH
                if self.num_agent_types > 0:
                    likelihood_scores_for_obs[jh] = - (avg_nll_across_partner_types / self.num_agent_types)
                else:
                    likelihood_scores_for_obs[jh] = 0.0 # Should not happen if DEFAULT_NUM_AGENT_TYPES > 0

            # 3. Normalize scores to probabilities for this focal_obs
            total_score_exp = 0.0
            for jh_key, score in likelihood_scores_for_obs.items():
                # Avoid overflow/underflow. Max score should be 0.0 (perfect prediction, NLL=0).
                # Clip very low scores to prevent exp(-inf) == 0.0 from losing valid histories.
                exp_score = np.exp(score) if score != float('-inf') else 0.0 # exp(-inf) is 0
                likelihood_scores_for_obs[jh_key] = exp_score
                total_score_exp += exp_score
            
            if total_score_exp > 0:
                for jh_key, score_exp in likelihood_scores_for_obs.items():
                    prob = score_exp / total_score_exp
                    self.private_to_public_histories[focal_obs][jh_key] = round(prob, N_DECIMALS_FOR_BELIEF)
            elif consistent_joint_histories: # All histories were invalid or had infinite NLL, assign uniform if any exist
                uniform_prob = 1.0 / len(consistent_joint_histories)
                for jh_key in consistent_joint_histories:
                    self.private_to_public_histories[focal_obs][jh_key] = round(uniform_prob, N_DECIMALS_FOR_BELIEF)
            # If no consistent_joint_histories, private_to_public_histories[focal_obs] remains an empty dict (not defaultdict).

    
    def train(self):
        """
        Main planning loop for the DTDE ToMBI agent using Backward Induction.
        It computes a policy and value function for the focal agent by considering
        all possible baseline partner types and using the pre-computed belief over
        joint histories.
        """
        max_delta = 0.0
        
        # Iterate through observations from longest to shortest for Backward Induction
        # This ensures that V(next_obs) values are already computed when needed.
        # So we need to reverse the order of `self.all_possible_observations` here.
        for obs in tqdm(self.all_possible_observations, desc="ToMBI Planning"):
            delta = self._optimize_node(obs)
            if delta > max_delta:
                max_delta = delta
        return max_delta
    
    def _optimize_node(self, obs: tuple) -> float:
        """
        Calculates the optimal Q-values and policy for a given observation `obs`.
        This is done by considering all possible actions of the focal agent,
        simulating interaction with all partner types (using the world model),
        and backing up values.
        
        This method now uses the pre-computed belief distribution `self.private_to_public_histories[obs]`
        to weight the contributions of different joint histories.
        """
        old_q_values = self.v_values.get(obs, np.full(self.num_actions, -np.inf, dtype=np.float32)).copy()
        
        focal_player_id = self._get_current_player_id(obs)
        partner_player_id = 1 - focal_player_id

        legal_actions_for_focal_agent = self.legal_actions_cache.get(obs, ())
        
        # If no legal actions, return 0.0 delta and ensure values are -inf
        if not legal_actions_for_focal_agent:
            self.v_values[obs] = np.full(self.num_actions, -np.inf, dtype=np.float32)
            self.policy[obs] = -1
            return np.max(np.abs(self.v_values[obs] - old_q_values)) if old_q_values.size > 0 else 0.0

        # Initialize Q-values for this observation (current focal agent's turn)
        new_q_values_for_obs = np.full(self.num_actions, -np.inf, dtype=np.float32)

        # Retrieve the belief distribution for this observation
        belief_distribution = self.private_to_public_histories.get(obs, {})
        if not belief_distribution:
            # Fallback if no belief distribution (e.g., all consistent JHs were deemed impossible by WM).
            # Treat all consistent joint histories as uniformly probable.
            consistent_jhs = self._get_consistent_joint_histories(obs, focal_player_id)
            if consistent_jhs:
                uniform_prob = 1.0 / len(consistent_jhs)
                belief_distribution = {jh: uniform_prob for jh in consistent_jhs}
            else: # No consistent histories at all
                self.v_values[obs] = np.full(self.num_actions, -np.inf, dtype=np.float32)
                self.policy[obs] = -1
                return np.max(np.abs(self.v_values[obs] - old_q_values)) if old_q_values.size > 0 else 0.0
        
        # Iterate through each possible action the focal agent can take
        for focal_agent_action in legal_actions_for_focal_agent:
            expected_return_for_action = 0.0
            
            # Iterate over the belief distribution (joint histories with their probabilities)
            for ground_truth_joint_history, jh_prob in belief_distribution.items():
                if jh_prob == 0: continue # Skip impossible histories

                avg_return_for_jh_across_partner_types = 0.0
                num_valid_partner_types_for_jh = 0

                # For each consistent joint history, average its expected outcome over all known partner types
                for partner_type_id in range(self.num_agent_types):
                    temp_env = deepcopy(self.env)
                    
                    try:
                        temp_env.reset(list(ground_truth_joint_history))
                    except (AssertionError, ValueError):
                        continue

                    # Simulate focal agent's action
                    # Rebuild `current_focal_history_list` for this `jh` and `focal_agent_action` for MentalNet
                    current_focal_history_list = []
                    
                    # Store focal_obs *before* focal_agent_action for WM input (current_obs) and MentalNet
                    focal_obs_before_focal_action = self._mask_state(tuple(temp_env.history), focal_player_id)
                    current_focal_history_list.append(_encode_observation(focal_obs_before_focal_action, self.feat_dim, self.env))

                    try:
                        # Check if it's the focal player's turn in the temporary environment for this JH
                        current_turn_in_temp_env = self._get_current_player_id(tuple(temp_env.history))
                        
                        if current_turn_in_temp_env != focal_player_id:
                            # This specific joint_history state implies a different player's turn or inconsistency.
                            # Penalize heavily and skip this partner type for this JH.
                            avg_return_for_jh_across_partner_types += -100.0
                            num_valid_partner_types_for_jh += 1
                            continue

                        temp_env.step(focal_agent_action)
                    except ValueError:
                        # Focal agent's action is illegal in this specific underlying world state given the cards
                        avg_return_for_jh_across_partner_types += -100.0 # Penalize
                        num_valid_partner_types_for_jh += 1
                        continue

                    # Predict partner's action using the World Model given focal's observation (pre-action) and focal's action
                    wm_past_eps, wm_curr_hist, wm_curr_obs, wm_own_act = self._prepare_wm_inputs(
                        focal_obs_before_focal_action, focal_agent_action, # Inputs to WM
                        partner_type_id, current_focal_history_list # Partner type, Focal's history for MentalNet
                    )
                    
                    self.world_model.eval() # Ensure eval mode for inference
                    with torch.no_grad():
                        action_logits, _, _ = self.world_model(
                            wm_past_eps, wm_curr_hist, wm_curr_obs, wm_own_act
                        )
                    
                    partner_action_pred = torch.argmax(action_logits, dim=-1).item()

                    # Simulate partner's *predicted* action
                    try:
                        current_turn_in_temp_env = self._get_current_player_id(tuple(temp_env.history))
                        
                        if current_turn_in_temp_env != partner_player_id:
                            # Inconsistency: partner should be acting now.
                            avg_return_for_jh_across_partner_types += -100.0
                            num_valid_partner_types_for_jh += 1
                            continue

                        temp_env.step(partner_action_pred)
                    except ValueError:
                        # Partner's predicted action is illegal in this specific underlying world state
                        avg_return_for_jh_across_partner_types += -100.0 # Penalize
                        num_valid_partner_types_for_jh += 1
                        continue

                    # Calculate value for this trajectory
                    current_return = 0.0
                    if temp_env.is_terminal():
                        current_return = temp_env.payoff()
                    else:
                        # Non-terminal: Look up value of the next observation for the focal agent
                        next_joint_history = tuple(temp_env.history)
                        next_focal_obs = self._mask_state(next_joint_history, focal_player_id)
                        
                        # Use the max Q-value from the next state (already computed by BI)
                        if next_focal_obs in self.v_values:
                            # Filter out -inf for illegal actions before taking max
                            valid_q_values = self.v_values[next_focal_obs][
                                np.isin(np.arange(self.num_actions), self.legal_actions_cache.get(next_focal_obs, ()))
                            ]
                            if valid_q_values.size > 0:
                                max_next_q = np.max(valid_q_values)
                            else: # No legal actions in next_focal_obs
                                max_next_q = 0.0
                            current_return = self.gamma * max_next_q
                        else:
                            current_return = 0.0 # Default if next_focal_obs not yet computed (should not happen in BI)
                    
                    avg_return_for_jh_across_partner_types += current_return
                    num_valid_partner_types_for_jh += 1

                # Average return for this JH across valid partner types
                if num_valid_partner_types_for_jh > 0:
                    avg_return_for_jh = avg_return_for_jh_across_partner_types / num_valid_partner_types_for_jh
                else:
                    avg_return_for_jh = -100.0 # Severe penalty if no valid scenarios for any partner type

                expected_return_for_action += avg_return_for_jh * jh_prob

            new_q_values_for_obs[focal_agent_action] = expected_return_for_action

        # Update V-values (Q-table) and policy for this observation
        self.v_values[obs] = new_q_values_for_obs
        
        # Select best action for the policy
        if np.all(np.isneginf(new_q_values_for_obs)): # If all actions are illegal or heavily penalized
            self.policy[obs] = -1
        else:
            best_action_candidates = np.flatnonzero(new_q_values_for_obs == np.max(new_q_values_for_obs))
            self.policy[obs] = np.random.choice(best_action_candidates) # Break ties randomly
        
        delta = np.max(np.abs(new_q_values_for_obs - old_q_values))
        return delta
    
    def act(self, input_state, *args, **kwargs):
        action = self.policy[input_state]
        return action
    
    def save(self, filepath : str, *args, **kwargs):
        data = {
            "policy" : self.policy,
            "v_values" : dict(self.v_values)
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        return
    
    def load(self, filepath : str, *args, **kwargs):
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        if os.path.getsize(filepath) == 0:
            raise ValueError("Q-table file is empty")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            raise ValueError("Loaded file does not contain a valid dictionary.")
        
        # Reconstruct defaultdict
        self.policy.update(data['policy'])
        self.v_values.update(data['v_values'])
        return
    
    def reset(self):
        return
    
    # EMPTY FUNCTIONS -- LEAVE THEM
    def save_transition(self):pass   


class ToMBI_AgentList(AgentList):
    def __init__(
            self,
            tom_agent : DTDE_ToMBI_Agent,
            baseline_agents : dict[BaseAgent]):
        self.baseline_agents : dict[BaseAgent] = baseline_agents
        self.focal_tom_agent : DTDE_ToMBI_Agent = tom_agent
        self.current_partner : None|BaseAgent = None
        self.partner_role : int|None = None
        return

    def set_current_partner(self, agent_type : str, partner_role : int):
        """Switch around active partner and role"""
        assert agent_type in self.baseline_agents.keys()
        assert partner_role in [0,1]
        self.partner_role = partner_role
        self.current_partner = self.baseline_agents[agent_type][partner_role]

        self.clear()
        # ToM Plays first
        if self.partner_role == 1:
            self.append(self.focal_tom_agent)
            self.append(self.current_partner)
        # Baseline Agents plays first
        else:
            self.append(self.current_partner)
            self.append(self.focal_tom_agent)
        return
    
    def train(self): return self.focal_tom_agent.train()      
    def save(self, filepath: str): return self.focal_tom_agent.save(filepath)
        