# agents/model_based/ToM_pbvi.py

import random
from typing import Any

import numpy as np
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
from tiny_game import DecPOMDP, MyHanabi, Game, get_all_possible_histories

from ..base_agent import ModelBasedAgent



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


class ToM_PBVI_Agent(ModelBasedAgent):
    """
    Theory-of-Mind Point-Based Value Iteration Agent.

    Converts the Dec-POMDP into a single-agent POMDP by modelling the
    partner's behaviour with a learned world model, then solves that
    POMDP via PBVI.

    The world model predicts  P(a_partner | history, a_own, z_own),
    making the partner part of a stochastic environment transition.

    Value function:
      alpha_vectors[h] : {world -> float}
      One alpha-vector per private observation (belief point).
      The alpha-vector stores the backed-up value for each specific
      consistent world, preserving per-world information that a scalar
      average would lose.

    Backup (longest-first, in-place):
      For each world s in B(h) and action a_i:
        1. Step env with a_i
        2. If terminal: alpha_a(s) = R
        3. Else: query world model for P(a_j | ...)
           alpha_a(s) = sum_j  p_j * [R_terminal  or
                         gamma * alpha_{h'}(s')]
           where h' = own next private obs, s' = next full state
      Select a* = argmax_a  b_h . alpha_a
    """

    def __init__(
        self,
        env: Game,
        num_cards: int,
        num_actions: int,
        world_model: ToM_WorldModel,
        world_model_config: dict[str, Any],
        ensemble: np.ndarray,
        agent_id: int = 0,
        device: str = "cpu",
        gamma: float = 0.99,
        *args, **kwargs,
    ):
        super().__init__(env, num_cards, num_actions)
        self.agent_id = agent_id
        self.device = device
        self.gamma = gamma

        all_private_histories, all_joint_histories = get_all_possible_histories(self.env)
        self.all_private_histories = sorted(all_private_histories, key=lambda x: len(x[0]), reverse=True)
        self.all_joint_histories = sorted(all_joint_histories, key=lambda x: len(x[0]), reverse=True)
    
        self.world_model = world_model.to(device)
        self.world_model.eval()

        # World-model encoding dimensions
        self.world_model_config = world_model_config
        self.max_seq_len = world_model_config['max_seq_len']
        self.obs_dim = world_model_config['obs_dim']
        self.joint_obs_dim = world_model_config['joint_obs_dim']
        self.action_dim = world_model_config['action_dim']

        self.ensemble: np.ndarray = ensemble
        self.ensemble_tensor = torch.tensor(
            ensemble, dtype=torch.float32, device=device,
        ).unsqueeze(0)
        self.past_episode_context = ensemble.shape[0]

        # Caches
        self.worlds_cache: dict[tuple, list[tuple]] = {}
        self.legal_actions_cache: dict[tuple, tuple] = {}

        # PBVI planning tables
        self.policy: dict[tuple, int] = {}
        # Alpha-vectors: priv_obs -> {consistent_world -> backed-up value}
        self.alpha_vectors: dict[tuple, dict[tuple, float]] = {}

        self._init_tables()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_tables(self):
        for history, done, turn_id, _reward in self.all_private_histories:
            if done or turn_id != self.agent_id:
                continue
            if self.is_decpomdp:
                legal = tuple(range(self.num_actions))
            else:
                _, legal = self.env.num_legal_actions(history)
            self.legal_actions_cache[history] = legal
            self.policy[history] = random.choice(legal)

    # ------------------------------------------------------------------
    # Belief / consistent-world computation  (unchanged)
    # ------------------------------------------------------------------

    def _get_consistent_worlds(self, obs):
        if obs in self.worlds_cache:
            return self.worlds_cache[obs]
        consistent = []

        if self.is_decpomdp:
            deal_obs = obs[:2]; hist_obs = obs[2:]; deal_len = 2
        else:
            deal_obs = obs[:4]; hist_obs = obs[4:]; deal_len = 4

        for deal in self.env.start_states():
            match = all(
                deal_obs[i] == -1 or deal_obs[i] == deal[i]
                for i in range(deal_len)
            )
            if not match:
                continue

            self.env.reset(list(deal))
            legal = True
            for event in hist_obs:
                if self.is_decpomdp:
                    if self.env.is_terminal():
                        legal = False; break
                    self.env.step(event)
                else:
                    action, obs_card = event
                    mask, _ = self.env.num_legal_actions()
                    if mask[action] == 0:
                        legal = False; break
                    self.env.step(action)
                    if self.env.history[-1][1] != obs_card:
                        legal = False; break
            if legal:
                consistent.append(tuple(list(deal) + list(hist_obs)))

        self.worlds_cache[obs] = consistent
        return consistent

    # ------------------------------------------------------------------
    # World-model query  (unchanged)
    # ------------------------------------------------------------------

    def _predict_partner_policy(self, world, action, next_obs):
        """Return P(a_partner) as a numpy array of shape (action_dim,)."""
        world_hands = list(world[:self.min_hist_length])
        actions = list(world[self.min_hist_length:])
        joint_h = [world_hands] + actions

        h_enc = np.zeros((self.max_seq_len, self.joint_obs_dim), dtype=np.float32)
        for i, obs in enumerate(joint_h):
            h_enc[i] = _encode_joint_observation(obs, self.joint_obs_dim, self.env)

        a_enc = _encode_action(action, self.action_dim)
        z_enc = _encode_observation(next_obs, self.obs_dim, self.env)

        hist_t = torch.tensor(h_enc, dtype=torch.float32, device=self.device).unsqueeze(0)
        act_t  = torch.tensor(a_enc, dtype=torch.float32, device=self.device).unsqueeze(0)
        obs_t  = torch.tensor(z_enc, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits, _, _ = self.world_model(
                self.ensemble_tensor, hist_t, obs_t, act_t,
            )
        return torch.softmax(logits, dim=-1).cpu().numpy()[0]

    # ------------------------------------------------------------------
    # PBVI training
    # ------------------------------------------------------------------

    def train(self) -> float:
        """
        One PBVI backward sweep (longest-first, in-place update).

        Because observations are processed longest-first and alpha-vectors
        are written immediately, deeper values are available when backing
        up shallower observations.  One sweep is therefore equivalent to
        exact backward induction with alpha-vectors.

        Returns max |delta V(b)| across all belief points.
        """
        max_delta = 0.0

        pbar = tqdm(self.all_private_histories, desc="ToM-PBVI sweep", leave=False)
        for priv_h, done, turn_id, _ in pbar:
            if done or turn_id != self.agent_id:
                continue

            worlds = self._get_consistent_worlds(priv_h)
            if not worlds:
                continue

            best_alpha, best_action = self._pbvi_backup(priv_h, worlds)

            # Convergence delta (uniform belief)
            prob = 1.0 / len(worlds)
            new_val = sum(prob * best_alpha.get(w, 0.0) for w in worlds)
            old_alpha = self.alpha_vectors.get(priv_h, {})
            old_val = sum(prob * old_alpha.get(w, 0.0) for w in worlds)
            max_delta = max(max_delta, abs(new_val - old_val))

            # In-place update so shallower backups see fresh values
            self.alpha_vectors[priv_h] = best_alpha
            self.policy[priv_h] = best_action

            pbar.set_postfix({"max_delta": f"{max_delta:.6f}"})

        return max_delta

    def _pbvi_backup(
        self,
        priv_h: tuple,
        worlds: list[tuple],
    ) -> tuple[dict[tuple, float], int]:
        """
        Point-based backup for a single belief point.

        For each legal action a_i, builds an alpha-vector  alpha_a : world -> float
        by simulating the action, querying the world model for the partner's
        stochastic response, and looking up the per-world continuation value
        from the already-computed alpha-vector at the next private observation.

        Returns (best_alpha_vector, best_action).
        """
        legal_actions = self.legal_actions_cache[priv_h]
        prob = 1.0 / len(worlds)

        best_value  = -float('inf')
        best_alpha:  dict[tuple, float] = {}
        best_action = legal_actions[0]

        for a_i in legal_actions:
            alpha_a: dict[tuple, float] = {}

            for world in worlds:
                self.env.reset(list(world))
                try:
                    self.env.step(a_i)
                except ValueError:
                    alpha_a[world] = 0.0
                    continue

                # Own action ends the game
                if self.env.is_terminal():
                    alpha_a[world] = self.env.payoff()
                    continue

                # --- partner's stochastic response via world model ---
                state_after_own = list(self.env.history)
                own_next_obs = self.env.context()[-1]
                partner_probs = self._predict_partner_policy(world, a_i, own_next_obs)

                # Legal partner actions
                if self.is_decpomdp:
                    legal_partner = list(range(self.num_actions))
                else:
                    _, legal_partner = self.env.num_legal_actions(tuple(state_after_own))

                # Renormalise predicted probs to legal actions
                legal_arr = np.array(legal_partner, dtype=np.int64)
                lp = np.atleast_1d(partner_probs[legal_arr])
                psum = lp.sum()
                lp = lp / psum if psum > 1e-8 else np.ones(len(legal_partner)) / len(legal_partner)

                # Expected value over partner actions
                world_val = 0.0
                for idx, a_j in enumerate(legal_partner):
                    p_j = float(lp[idx])

                    self.env.reset(state_after_own)
                    try:
                        self.env.step(a_j)
                    except ValueError:
                        continue

                    if self.env.is_terminal():
                        world_val += p_j * self.env.payoff()
                    else:
                        next_state = tuple(self.env.history)
                        next_priv  = self._mask_state(next_state)
                        # PBVI: per-world value from the successor alpha-vector
                        next_av = self.alpha_vectors.get(next_priv, {})
                        world_val += p_j * self.gamma * next_av.get(next_state, 0.0)

                alpha_a[world] = world_val

            # b_h . alpha_a  (uniform belief)
            value = sum(prob * alpha_a.get(w, 0.0) for w in worlds)
            if value > best_value:
                best_value  = value
                best_alpha  = alpha_a
                best_action = a_i

        return best_alpha, best_action

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def act(self, private_history, exploit=False):
        return self.policy[private_history]

    def save_transition(self, *args): pass

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path):
        data = {
            "policy": self.policy,
            "alpha_vectors": {k: dict(v) for k, v in self.alpha_vectors.items()},
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.policy.update(data["policy"])
        self.alpha_vectors.update(data.get("alpha_vectors", {}))

    def set_ensemble(self, ensemble: np.ndarray) -> None:
        """Swap the past-episode context fed to CharacterNet."""
        self.ensemble = ensemble
        self.ensemble_tensor = torch.tensor(
            ensemble, dtype=torch.float32, device=self.device,
        ).unsqueeze(0)

    # ------------------------------------------------------------------
    # Observation masking
    # ------------------------------------------------------------------

    def _mask_state(self, state):
        """Mask the acting player's own cards to produce their private observation."""
        s = list(state)
        if self.is_decpomdp:
            p0_turn = ((len(state) - 2) % 2 == 0)
            s[0 if p0_turn else 1] = -1
        else:
            p0_turn = ((len(state) - 4) % 2 == 0)
            if p0_turn:
                s[0] = s[1] = -1
            else:
                s[2] = s[3] = -1
        return tuple(s)

    def reset(self):
        self.alpha_vectors.clear()
        self.policy.clear()
        self._init_tables()