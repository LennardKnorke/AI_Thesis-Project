import random
from typing import Any

import numpy as np
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
from tiny_game import DecPOMDP, MyHanabi, Game, get_all_possible_histories

from ..base_agent import ModelBasedAgent


def _encode_action(action: int, action_dim: int):
    """One-hot action encoding. Unused — own action is always null; kept for reference."""
    vec = np.zeros(action_dim, dtype=np.float32)
    vec[action] = 1.0
    return vec


def _encode_decPOMDP_jh(obs: int | tuple[int, int], obs_dim: int, env: Game):
    vec = np.zeros(obs_dim, dtype=np.float32)
    num_actions = env.num_actions
    num_cards   = env.num_cards

    if isinstance(obs, int):
        vec[obs] = 1.0
    else:
        for i, o in enumerate(obs):
            if o < 0: continue
            idx = num_actions + (i * num_cards) + o
            vec[idx] = 1.0
    return vec


def _encode_MyHanabi_jh(obs: int | tuple[int, int], obs_dim: int, env: Game):
    vec = np.zeros(obs_dim, dtype=np.float32)
    num_actions = env.num_actions
    num_cards   = env.num_cards

    if len(obs) == 4:
        # Initial card observation
        for i, o in enumerate(obs):
            idx = num_actions + (i * num_cards) + o
            vec[idx] = 1.0
    elif len(obs) == 2:
        # Action observation: (action, card_revealed)
        action       = obs[0]
        card_revealed = obs[1]
        vec[action]                        = 1.0
        vec[num_actions + card_revealed]   = 1.0
    else:
        raise ValueError("Invalid Observation provided")
    return vec


def _encode_joint_observation(obs: int | tuple[int, int], obs_dim: int, env: Game):
    if isinstance(env, DecPOMDP):
        return _encode_decPOMDP_jh(obs, obs_dim, env)
    elif isinstance(env, MyHanabi):
        return _encode_MyHanabi_jh(obs, obs_dim, env)
    else:
        raise ValueError("Faulty Environment")


def _encode_decPOMDP_o(obs: int | tuple[int, int], obs_dim: int, env: Game):
    vec = np.zeros(obs_dim, dtype=np.float32)
    num_actions = env.num_actions
    num_cards   = env.num_cards

    if isinstance(obs, int):
        vec[obs] = 1.0
    else:
        for o in obs:
            if o == -1: continue
            vec[num_actions + o] = 1.0
    return vec


def _encode_MyHanabi_o(obs: tuple[int, ...], obs_dim: int, env: Game):
    vec = np.zeros(obs_dim, dtype=np.float32)
    num_actions        = env.num_actions
    num_cards          = env.num_cards
    num_cards_revealed = num_cards + 1

    if len(obs) == 4:
        # Initial card observation
        for i, o in enumerate(obs):
            if o == -1: continue
            idx = num_actions + num_cards_revealed + (i * num_cards) + o
            vec[idx] = 1.0
    elif len(obs) == 2:
        # Action observation: (action, card_revealed)
        action        = obs[0]
        card_revealed = obs[1]
        vec[action]                       = 1.0
        vec[num_actions + card_revealed]  = 1.0
    else:
        raise ValueError("Invalid Observation provided")
    return vec


def _encode_observation(obs: int | tuple[int, int], obs_dim: int, env: Game):
    if isinstance(env, DecPOMDP):
        return _encode_decPOMDP_o(obs, obs_dim, env)
    elif isinstance(env, MyHanabi):
        return _encode_MyHanabi_o(obs, obs_dim, env)
    else:
        raise ValueError("Faulty Environment")


class CharacterNet(nn.Module):
    """
    Encodes past episodes via LSTM and sums valid episode embeddings into e_char.
    At k=0 (no past episodes) e_char is all-zeros; downstream biases learn the prior.
    """
    def __init__(self, input_dim, embedding_dim, num_agent_types):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, embedding_dim, batch_first=True)
        self.identity_classifier = nn.Linear(embedding_dim, num_agent_types)

    def forward(self, past_episodes, mask):
        B, N, seq, feat = past_episodes.size()
        _, (h_n, _) = self.lstm(past_episodes.view(B * N, seq, feat))
        emb    = h_n[-1].view(B, N, -1)
        emb    = emb * mask.unsqueeze(-1).float()   # zero out invalid episodes
        e_char = emb.sum(dim=1)                     # (B, embed_dim)
        identity_logits = self.identity_classifier(e_char)
        return e_char, identity_logits


# --- MLP-based CharacterNet (performed worse, kept for reference) ---
# class CharacterNet(nn.Module):
#     def __init__(self, input_dim, embedding_dim, num_agent_types):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(input_dim, embedding_dim), nn.ReLU(),
#             nn.Linear(embedding_dim, embedding_dim), nn.ReLU(),
#         )
#         self.identity_classifier = nn.Linear(embedding_dim, num_agent_types)
#
#     def forward(self, past_episodes, mask):
#         B, N, seq, feat = past_episodes.size()
#         step_emb = self.mlp(past_episodes.view(B * N * seq, feat)).view(B, N, seq, -1)
#         step_mask = (past_episodes.abs().sum(dim=-1, keepdim=True) > 0).float()
#         ep_emb = (step_emb * step_mask).sum(dim=2)
#         e_char = (ep_emb * mask.unsqueeze(-1).float()).sum(dim=1)
#         return e_char, self.identity_classifier(e_char)


class MentalNet(nn.Module):
    """
    Encodes the current joint history conditioned on e_char into a mental state embedding.
    e_char is concatenated to every LSTM input so type identity persists as a conditioning signal.
    """
    def __init__(self, input_dim, char_embed_dim, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.lstm      = nn.LSTM(input_dim + char_embed_dim, embedding_dim, batch_first=True)
        self.char_to_h = nn.Linear(char_embed_dim, embedding_dim)
        self.char_to_c = nn.Linear(char_embed_dim, embedding_dim)

    def forward(self, current_history, e_char):
        h0 = self.char_to_h(e_char).unsqueeze(0).contiguous()
        c0 = self.char_to_c(e_char).unsqueeze(0).contiguous()
        B, seq, _  = current_history.size()
        e_char_exp = e_char.unsqueeze(1).expand(B, seq, -1)
        lstm_input = torch.cat([current_history, e_char_exp], dim=-1)
        _, (h_n, _) = self.lstm(lstm_input, (h0, c0))
        return h_n[-1]


class ToM_WorldModel(nn.Module):
    """
    Theory-of-Mind world model predicting the partner's next action.

    Architecture: CharacterNet (past episodes → e_char) feeds into MentalNet
    (current history → e_mental); a shared MLP trunk produces action logits.
    Identity logits from CharacterNet enable auxiliary agent-type classification.
    """
    def __init__(self,
                 obs_dim:              int,
                 joint_obs_dim:        int,
                 action_dim:           int,
                 num_agent_types:      int = 4,
                 max_seq_len:          int = 8,
                 past_episode_context: int = 5,
                 char_embed_dim:       int = 32,
                 mental_embed_dim:     int = 16,
                 trunk_dim:            int = 64,
                 use_obs:              bool = False,
                 *args, **kwargs):
        super().__init__()
        self.register_buffer('null_action', torch.zeros(1, action_dim))
        self.obs_dim = obs_dim
        self.use_obs = use_obs

        self.char_net   = CharacterNet(joint_obs_dim, char_embed_dim, num_agent_types)
        self.mental_net = MentalNet(joint_obs_dim, char_embed_dim, mental_embed_dim)

        # Action trunk — use_obs=False forces the network to use e_char rather than a shortcut
        # through the current observation.
        self.action_trunk_input_dim = char_embed_dim + mental_embed_dim + (joint_obs_dim if use_obs else 0)
        # LayerNorm balances e_char (summed, small magnitude) and e_mental (LSTM output).
        self.action_input_norm = nn.LayerNorm(self.action_trunk_input_dim)
        self.action_trunk = nn.Sequential(
            nn.Linear(self.action_trunk_input_dim, trunk_dim), nn.ReLU(),
            nn.Linear(trunk_dim, trunk_dim),                   nn.ReLU(),
        )

        # Observation trunk — REMOVED.
        # In both DecPOMDP and MyHanabi, z^i == z^{-i} for all t > 0, so predicting
        # the partner's next observation is a trivial identity with no useful signal.
        # Kept here as a rollback reference if future games introduce asymmetric obs.
        # self.obs_trunk_input_dim = char_embed_dim + mental_embed_dim + obs_dim + action_dim
        # self.obs_trunk = nn.Sequential(
        #     nn.Linear(self.obs_trunk_input_dim, trunk_dim), nn.ReLU(),
        #     nn.Linear(trunk_dim, trunk_dim),                nn.ReLU(),
        # )

        # Action head: raw logits — softmax is applied at the call site; CE loss expects logits.
        self.action_head = nn.Linear(trunk_dim, action_dim)

        # Observation head — REMOVED (see obs_trunk above).
        # self.observation_head = nn.Linear(trunk_dim, obs_dim)

    def forward(self, past_episodes, past_mask, current_history, current_obs):
        """
        Args:
            past_episodes:   (B, N_eps, seq, feat)
            past_mask:       (B, N_eps)  — 1 = real episode, 0 = padded
            current_history: (B, seq, feat)
            current_obs:     (B, obs_dim)
        Returns:
            action_logits, None, identity_logits, None
        """
        e_char, identity_logits = self.char_net(past_episodes, past_mask)

        # Mental net sees history up to t-1 only (causal, per ToMNet).
        history_tm1 = current_history[:, :-1, :]
        if history_tm1.shape[1] == 0:
            history_tm1 = torch.zeros(
                current_history.shape[0], 1, current_history.shape[2],
                dtype=current_history.dtype, device=current_history.device
            )
        e_mental = self.mental_net(history_tm1, e_char)

        z_now  = current_history[:, -1, :]
        parts  = [e_char, e_mental, z_now] if self.use_obs else [e_char, e_mental]
        x      = self.action_input_norm(torch.cat(parts, dim=1))
        action_logits = self.action_head(self.action_trunk(x))

        # Observation prediction removed — rollback reference:
        # act = self.null_action.expand(current_obs.shape[0], -1)
        # x_obs = torch.cat([e_char, e_mental, current_obs, act], dim=1)
        # next_obs_pred = self.observation_head(self.obs_trunk(x_obs))

        return action_logits, None, identity_logits, None
