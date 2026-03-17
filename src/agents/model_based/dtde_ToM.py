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



def _encode_decPOMDP_jh(obs : int|tuple[int, int], obs_dim : int, env : Game, s0 : list|tuple|None = None, is_p0_turn : bool= False):
    vec = np.zeros(obs_dim, dtype=np.float32)
    num_actions = env.num_actions
    num_cards = env.num_cards

    if isinstance(obs, int):
        # Encode observed action
        vec[obs] = 1.0
    else:
        # encode initial cards
        for i, o in enumerate(obs):
            if o < 0: continue
            idx = num_actions + (i * num_cards) + o
            vec[idx] = 1.0
    return vec


def _encode_MyHanabi_jh(obs : int|tuple[int, int], obs_dim : int, env : Game, s0 : list|tuple|None = None, is_p0_turn : bool= False):
    vec = np.zeros(obs_dim, dtype=np.float32)
    num_actions = env.num_actions
    num_cards = env.num_cards

    if len(obs) == 4:
        # Initial Card Observation
        for i, o in enumerate(obs):
            idx = num_actions + (i * num_cards) + o
            vec[idx] = 1.0
    elif len(obs) == 2:
        # Action observation
        action = obs[0]
        card_revealed = obs[1]
        card_idx = num_actions + card_revealed

        vec[action] = 1.0
        vec[card_idx] = 1.0
    else:
        raise ValueError("Invalid Observation provided")
    return vec


def _encode_joint_observation(obs : int|tuple[int, int], obs_dim : int, env : Game, s0 : list|tuple|None = None, is_p0_turn : bool= False):
    if isinstance(env, DecPOMDP):
        return _encode_decPOMDP_jh(obs, obs_dim, env, s0, is_p0_turn)
    elif isinstance(env, MyHanabi):
        return _encode_MyHanabi_jh(obs, obs_dim, env, s0, is_p0_turn)
    else:
        raise ValueError("Faulty Environment")


def _encode_decPOMDP_o(obs : int|tuple[int, int], obs_dim : int, env : Game, s0 : list|tuple|None = None, is_p0_turn : bool = False):
    vec = np.zeros(obs_dim, dtype=np.float32)
    num_actions = env.num_actions
    num_cards = env.num_cards

    if isinstance(obs, int):
        vec[obs] = 1.0
    else:
        #o = obs[0]
        for o in obs:
            if o == -1: continue
            vec[num_actions + o] = 1.0
    
    return vec


def _encode_MyHanabi_o(obs : tuple[int,...], obs_dim : int, env : Game, s0 : list|tuple|None = None, is_p0_turn : bool = False):
    vec = np.zeros(obs_dim, dtype=np.float32)
    num_actions = env.num_actions
    num_cards = env.num_cards

    num_cards_revealed = num_cards + 1
    
    if len(obs) == 4:
        # Initial Card Observation
        for i, o in enumerate(obs):
            if o == -1: continue
            idx = num_actions + num_cards_revealed + (i * num_cards) + o
            vec[idx] = 1.0
    elif len(obs) == 2:
        # Action observation
        action = obs[0]
        card_revealed = obs[1]
        card_idx = num_actions + card_revealed

        vec[action] = 1.0
        vec[card_idx] = 1.0
    else:
        raise ValueError("Invalid Observation provided")
    return vec


def _encode_observation(obs : int|tuple[int, int], obs_dim : int, env : Game, s0 : list|tuple|None = None, is_p0_turn : bool = False):
    if isinstance(env, DecPOMDP):
        return _encode_decPOMDP_o(obs, obs_dim, env, s0, is_p0_turn)
    elif isinstance(env, MyHanabi):
        return _encode_MyHanabi_o(obs, obs_dim, env, s0, is_p0_turn)
    else:
        raise ValueError("Faulty Environment")


def _encode_action(action : int, action_dim : int):
    vec = np.zeros(action_dim, dtype=np.float32)
    #if isinstance(action, (list, tuple)):
    #    a = action[0]
    #else:
    #    a = action
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
                 trunk_dim : int = 64,
                 *args, **kwargs):
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
        world_model: ToM_WorldModel,
        world_model_config: dict[str, Any],
        ensemble : np.ndarray,
        device: str = "cpu",
        gamma: float = 0.99,
        *args, **kwargs
    ):
        super().__init__(env, num_cards, num_actions)
        self.device = device
        self.is_decpomdp = isinstance(self.env, DecPOMDP)
        self.is_myhanabi = isinstance(self.env, MyHanabi)
        self.num_cards_in_hand = 1 if self.is_decpomdp else 2

        self.world_model = world_model.to(device)
        self.world_model.eval() # Set to eval mode for inference

        self.ensemble : np.ndarray = ensemble
        self.ensemble_tensor = torch.tensor(
            ensemble,
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)

        self.past_episode_context = ensemble.shape[0]

        self.gamma = gamma

        # Extract World Model configuration parameters
        self.world_model_config = world_model_config
        self.max_seq_len = world_model_config['max_seq_len']
        self.obs_dim = world_model_config['obs_dim']
        self.joint_obs_dim = world_model_config['joint_obs_dim']
        self.action_dim = world_model_config['action_dim']

        # All possible ground-truth joint histories
        all_histories, all_jh = get_all_possible_histories(self.env)
        self.all_private_histories = sorted(
            all_histories, 
            key=lambda x: len(x[0]), 
            reverse=True
        )
        self.all_joint_histories = sorted(
            all_jh, 
            key=lambda x: len(x[0]), 
            reverse=True
        )
        
        # Belief mapping: private history -> {joint history: probability}
        self.private_to_joint_histories : dict[tuple, dict[tuple, float]] = {} # Private histories - {joint history - probablity}
        
        # Planning tables
        self.v_values: dict[tuple, float] = {}      # Joint histories - Values
        self.policy: dict[tuple, int|None] = {}     # Private histories - action
        self.legal_actions_cache = {}               # Private histories - actions
        
        self.joint_transition_cache : dict[tuple, list[tuple]] = {}
        self.partner_prediction_cache : dict[tuple, np.ndarray] = {}

        self._init_legal_actions()
        self._init_tables()
        #self._precompute_beliefs()
        #self._precompute_joint_transitions()
        return

    def _init_legal_actions(self):
        for obs, done, turn_id, reward in self.all_private_histories:
            if done:
                self.legal_actions_cache[obs] = ()
                continue

            if self.is_decpomdp:
                self.legal_actions_cache[obs] = tuple(range(self.num_actions))
            elif self.is_myhanabi:
                _, actions = self.env.num_legal_actions(obs)
                self.legal_actions_cache[obs] = actions
            else:
                raise ValueError("No valid environment proided")
    
    def _init_tables(self):
        # Init policy action
        for obs, done, turn_id, reward in self.all_private_histories:
            if obs not in self.v_values.keys():
                self.v_values[obs] = reward
            if done:
                self.policy[obs] = None
                self.v_values[obs] = reward
            else:
                acts = self.legal_actions_cache.get(obs, ())
                if acts:
                    self.policy[obs] = random.choice(acts)
                else:
                    self.policy[obs] = None 
        return
    
    def _mask_state(self, joint_hist, player_id):
        s_list = list(joint_hist)
        if self.is_decpomdp:
            if player_id == 0:
                s_list[0] = -1
            else:
                s_list[1] = -1
        else:
            if player_id == 0: 
                s_list[0] = -1; s_list[1] = -1
            else: 
                s_list[2] = -1; s_list[3] = -1

        return tuple(s_list)
    
    def _get_joint_belief(self, private_hist):
        return self.private_to_joint_histories.get(private_hist, {})


    def train(self):
        max_delta = 0
        pbar_ = tqdm(self.all_private_histories, leave=False, desc="ToM-BI Planning")
        for private_hist, done, _, _ in pbar_:

            if done:
                continue

            delta = self._optimize_private_node(private_hist)

            max_delta = max(max_delta, delta)

        return max_delta
    
    def _optimize_private_node(self, private_hist):
        # Determine whose turn it is from the private_hist length
 
        
        return abs(0.0) # Return delta for convergence check

    
    def act(self, obs, exploit : bool, *args, **kwargs):
        if not isinstance(obs, tuple):
            obs = tuple(obs)
        
        action = self.policy.get(obs)
        if action is None:
            legal_actions = self.legal_actions_cache.get(obs, ())
            if legal_actions:
                return random.choice(legal_actions)
            else:
                raise ValueError(f"No policy or legal actions for observation: {obs}")
        return action
    
    def save_transition(self, *args, **kwargs):
        return
    
    def load(self, filepath: str, *args, **kwargs):
        """Loads policy and V-values from file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            
        if not isinstance(data, dict):
            raise ValueError("Invalid checkpoint format")

        if "policy" not in data or "v_values" not in data:
            raise ValueError("Checkpoint missing required fields: 'policy' and 'v_values'.")
        
        self.policy = data["policy"]
        self.v_values = data["v_values"]
        print(f"Loaded ToM-BI agent from {filepath}")
        return
    
    def save(self, filepath: str, *args, **kwargs):
        """Saves policy and V-values to file."""
        data = {
            "policy": self.policy,
            "v_values": self.v_values
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved ToM-BI agent to {filepath}")
        return

class ToMBI_AgentList(AgentList):
    def __init__(self, tom_agent : DTDE_ToMBI_Agent, agents : dict[str, AgentList], *args, **kwargs):
        self.baseline_agents = agents
        self.tom_agent = tom_agent

        super().__init__([tom_agent, tom_agent])

        self.current_partner_name = None
        self.tom_side = 0

        first_partner = list(self.baseline_agents.keys())[0]
        self.set_current_partner(first_partner, 0)
        return


    def set_current_partner(self, agent_name : str, side : int):
        if agent_name not in self.baseline_agents:
            raise ValueError(f"Unknown baseline agent {agent_name}")
        
        partner_agents = self.baseline_agents[agent_name]

        self.clear()

        if side == 0:
            self.append(self.tom_agent)
            self.append(partner_agents[1])
        else:
            self.append(partner_agents[0])
            self.append(self.tom_agent)

        self.tom_side = side
        self.current_partner_name = agent_name
        return

    def train(self):
        """
        Only train the ToM agent.
        """
        return self.tom_agent.train()
    
    def reset(self):
        """
        Reset both agents before a new run.
        """
        self.tom_agent.reset()

        for agents in self.baseline_agents.values():
            for a in agents:
                a.reset()

    def save(self, path : str, *args, **kwargs):
        self.tom_agent.save(path)
        return