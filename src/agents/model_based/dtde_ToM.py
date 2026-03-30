# agents/model_based/dtde_ToM.py
from collections import defaultdict, deque

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



def _encode_decPOMDP_jh(obs : int|tuple[int, int], obs_dim : int, env : Game):
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


def _encode_MyHanabi_jh(obs : int|tuple[int, int], obs_dim : int, env : Game):
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


def _encode_joint_observation(obs : int|tuple[int, int], obs_dim : int, env : Game):
    if isinstance(env, DecPOMDP):
        return _encode_decPOMDP_jh(obs, obs_dim, env)
    elif isinstance(env, MyHanabi):
        return _encode_MyHanabi_jh(obs, obs_dim, env)
    else:
        raise ValueError("Faulty Environment")


def _encode_decPOMDP_o(obs : int|tuple[int, int], obs_dim : int, env : Game):
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


def _encode_MyHanabi_o(obs : tuple[int,...], obs_dim : int, env : Game):
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


def _encode_observation(obs : int|tuple[int, int], obs_dim : int, env : Game):
    if isinstance(env, DecPOMDP):
        return _encode_decPOMDP_o(obs, obs_dim, env)
    elif isinstance(env, MyHanabi):
        return _encode_MyHanabi_o(obs, obs_dim, env)
    else:
        raise ValueError("Faulty Environment")


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
        - e_char: character embedding from CharacterNet
        - Current step-t joint/public history
    Output:
        - Embedding
    """
    def __init__(self, input_dim, char_embed_dim, embedding_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim + char_embed_dim, embedding_dim, batch_first=True)

    def forward(self, current_history, e_char):
        # Expand character embedding across sequence dimension
        e_char_expanded = e_char.unsqueeze(1).expand(-1, current_history.size(1), -1)
        x_mental = torch.cat([current_history, e_char_expanded], dim=2)

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
        self.mental_net = MentalNet(joint_obs_dim, char_embed_dim, mental_embed_dim)

        # 2a. Action trunk: psi^2_i = Pr(a^{-i}_t | h^i_t, h^{-i}_t, a^i_t)
        #     Does NOT include z^i_{t+1} — it occurs causally after a^{-i}_t is chosen
        self.action_trunk_input_dim = char_embed_dim + mental_embed_dim + action_dim
        self.action_trunk = nn.Sequential(
            nn.Linear(self.action_trunk_input_dim, trunk_dim),
            nn.ReLU(),
            nn.Linear(trunk_dim, trunk_dim),
            nn.ReLU()
        )

        # 2b. Observation trunk: psi^1_i = Pr(z^{-i}_{t+1} | h^i_t, h^{-i}_t, z^i_{t+1}, a^i_t)
        #     Includes z^i_{t+1} (current_obs) as conditioning information
        self.obs_trunk_input_dim = char_embed_dim + mental_embed_dim + obs_dim + action_dim
        self.obs_trunk = nn.Sequential(
            nn.Linear(self.obs_trunk_input_dim, trunk_dim),
            nn.ReLU(),
            nn.Linear(trunk_dim, trunk_dim),
            nn.ReLU()
        )

        # 3. Heads
        # Head A: Action Prediction -> P(a^{-i}_t)
        self.action_head = nn.Linear(trunk_dim, action_dim)

        # Head B: Observation Prediction -> P(z^{-i}_{t+1})
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

        # Mental embedding — conditioned on e_char (not logits)
        e_mental = self.mental_net(current_history, e_char)

        # Action prediction: psi^2_i — no z^i_{t+1} (causal constraint)
        x_action = torch.cat([e_char, e_mental, own_action], dim=1)
        action_features = self.action_trunk(x_action)
        action_logits = self.action_head(action_features)

        # Observation prediction: psi^1_i — conditioned on z^i_{t+1}
        x_obs = torch.cat([e_char, e_mental, current_obs, own_action], dim=1)
        obs_features = self.obs_trunk(x_obs)
        next_obs_pred = self.observation_head(obs_features)

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
        agent_id: int = 0,
        device: str = "cpu",
        gamma: float = 0.99,
        *args, **kwargs
    ):
        super().__init__(env, num_cards, num_actions)
        self.agent_id = agent_id
        self.device = device
        self.gamma = gamma

        #self.num_cards_in_hand = 1 if self.is_decpomdp else 2
        #self.horizon = 4 if self.is_decpomdp else 12

        self.world_model = world_model.to(device)
        self.world_model.eval() # Set to eval mode for inference

        # Extract World Model configuration parameters
        self.world_model_config = world_model_config
        self.max_seq_len = world_model_config['max_seq_len']
        self.obs_dim = world_model_config['obs_dim']
        self.joint_obs_dim = world_model_config['joint_obs_dim']
        self.action_dim = world_model_config['action_dim']

        self.ensemble : np.ndarray = ensemble
        self.ensemble_tensor = torch.tensor(
            ensemble,
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)

        self.past_episode_context = ensemble.shape[0]

        # caches
        self.worlds_cache = {}
        self.legal_actions_cache = {}

        # planning tables
        self.policy = {}
        self.v_values = defaultdict(float)

        self._init_tables()
        return


    def _init_tables(self):
        for history, done, turn_id, reward in self.all_private_histories:

            if done:
                self.v_values[history] = reward
                continue

            if turn_id != self.agent_id:
                continue

            if self.is_decpomdp:
                legal = tuple(range(self.num_actions))
            else:
                _, legal = self.env.num_legal_actions(history)

            self.legal_actions_cache[history] = legal
            self.policy[history] = random.choice(legal)
            self.v_values[history] = 0.0


    def _get_consistent_worlds(self, obs):
        if obs in self.worlds_cache:
            return self.worlds_cache[obs]
        consistent = []

        # Split cards actions
        if self.is_decpomdp:
            deal_obs = obs[:2]
            hist_obs = obs[2:]
            deal_len = 2
        else:
            deal_obs = obs[:4]
            hist_obs = obs[4:]
            deal_len = 4

        # Loop over start states
        for deal in self.env.start_states():
            # Figure out match
            match = True
            for i in range(deal_len):
                if deal_obs[i] != -1 and deal_obs[i] != deal[i]:
                    match = False
                    break

            if not match:
                continue

            # If valid start cards, replay possible scenarios
            self.env.reset(list(deal))
            legal = True

            for event in hist_obs:

                if self.is_decpomdp:
                    if self.env.is_terminal():
                        legal = False
                        break
                    self.env.step(event)

                else:
                    action, obs_card = event
                    mask, _ = self.env.num_legal_actions()

                    if mask[action] == 0:
                        legal = False
                        break

                    self.env.step(action)

                    if self.env.history[-1][1] != obs_card:
                        legal = False
                        break

            if legal:
                consistent.append(tuple(list(deal) + list(hist_obs)))

        self.worlds_cache[obs] = consistent
        return consistent


    def _predict_partner_policy(self, world, action, next_obs):
        
        # Encode joint history
        world_hands = list(world[:self.min_hist_length])
        actions = list(world[self.min_hist_length:])

        joint_h = [world_hands] + actions

        h_enc = np.zeros((self.max_seq_len, self.joint_obs_dim), dtype=np.float32)

        for i, obs in enumerate(joint_h):
            z_enc = _encode_joint_observation(obs, self.joint_obs_dim, self.env)
            h_enc[i] = z_enc

        a_enc = _encode_action(action, self.action_dim)
        priv_z_enc = _encode_observation(next_obs, self.obs_dim, self.env)

        next_obs_tensor = torch.tensor(priv_z_enc, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_tensor = torch.tensor(a_enc, dtype=torch.float32, device=self.device).unsqueeze(0)
        hist_tensor = torch.tensor(h_enc, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits, _, _ = self.world_model(
                self.ensemble_tensor,
                hist_tensor,
                next_obs_tensor,
                action_tensor
            )

        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return probs


    def train(self):

        max_delta = 0.0

        pbar = tqdm(self.all_private_histories, desc="Train Sweep", leave=False)
        for priv_h, done, turn_id, reward in pbar:
            postfix = {
                "max_delta" : max_delta
            }
            pbar.set_postfix(postfix)

            if done:
                continue

            if turn_id != self.agent_id:
                continue

            old_val = self.v_values[priv_h]

            new_val, best_action = self._evaluate_belief(priv_h)

            self.v_values[priv_h] = new_val
            self.policy[priv_h] = best_action

            max_delta = max(max_delta, abs(old_val - new_val))

        return max_delta
    

    def _evaluate_belief(self, private_history):
        legal_actions = self.legal_actions_cache[private_history]

        worlds = self._get_consistent_worlds(private_history)

        best_value = -float("inf")
        best_action = legal_actions[0]

        for action in legal_actions:
            total_value = 0.0
            count = 0

            for world in worlds:
                self.env.reset(list(world))
                try:
                    self.env.step(action)
                except ValueError:
                    continue
                count += 1

                # If the focal agent's action ends the game, collect the reward directly
                if self.env.is_terminal():
                    total_value += self.env.payoff()
                    continue

                # Otherwise predict the partner's response and propagate
                state_after_own_action = list(self.env.history)
                next_obs = self.env.context()[-1]
                partner_probs = self._predict_partner_policy(world, action, next_obs)

                if self.is_decpomdp:
                    legal_partner = list(range(self.num_actions))
                else:
                    _, legal_partner = self.env.num_legal_actions(tuple(state_after_own_action))

                world_value = 0.0
                for partner_action in legal_partner:
                    p_partner = partner_probs[partner_action]

                    self.env.reset(state_after_own_action)
                    try:
                        self.env.step(partner_action)
                    except ValueError:
                        continue

                    if self.env.is_terminal():
                        world_value += p_partner * self.env.payoff()
                    else:
                        next_state = tuple(self.env.history)
                        next_obs_masked = self._mask_state(next_state)
                        world_value += p_partner * self.gamma * self.v_values[next_obs_masked]

                total_value += world_value

            if count > 0:
                avg = total_value / count
                if avg > best_value:
                    best_value = avg
                    best_action = action

        if best_value == -float("inf"):
            best_value = self.v_values.get(private_history, 0.0)
        return best_value, best_action


    def act(self, private_history, exploit=False):
        return self.policy[private_history]
    

    def save(self, path):

        data = {
            "policy": self.policy,
            "values": dict(self.v_values)
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)


    def load(self, path):

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.policy.update(data["policy"])
        self.v_values.update(data["values"])
        return
    

    def _mask_state(self, state):

        s = list(state)

        if self.is_decpomdp:
            num_actions = len(state) - 2
            p0_turn = (num_actions % 2 == 0)
            if p0_turn:
                s[0] = -1
            else:
                s[1] = -1
        else:
            num_actions = len(state) - 4
            p0_turn = (num_actions % 2 == 0)

            if p0_turn:
                s[0] = -1
                s[1] = -1
            else:
                s[2] = -1
                s[3] = -1
        return tuple(s)