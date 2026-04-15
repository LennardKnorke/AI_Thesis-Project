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



def _encode_action(action : int, action_dim : int):
    """
    NOT NEEDED. Own Action is always Null-action. Left for documentation purposes.
    """
    vec = np.zeros(action_dim, dtype=np.float32)
    vec[action] = 1.0
    return vec


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
        - Current step-t joint history (state for hanabi)
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
        # Null action buffer: always zeros, expands to (batch, action_dim) in forward().
        # Registered so it follows the model to whatever device it lives on.
        self.register_buffer('null_action', torch.zeros(1, action_dim))
        self.obs_dim = obs_dim  # kept for API compatibility / obs_trunk rollback

        # 1. Sub-Nets
        self.char_net = CharacterNet(joint_obs_dim, char_embed_dim, num_agent_types)
        self.mental_net = MentalNet(joint_obs_dim, char_embed_dim, mental_embed_dim)

        # 2a. Action trunk: psi^2_i = Pr(a^{-i}_t | h^i_t, h^{-i}_t, a^i_t)
        #     Does NOT include z^i_{t+1} — it occurs causally after a^{-i}_t is chosen
        self.action_trunk_input_dim = char_embed_dim + mental_embed_dim #+ action_dim
        self.action_trunk = nn.Sequential(
            nn.Linear(self.action_trunk_input_dim, trunk_dim),
            nn.ReLU(),
            nn.Linear(trunk_dim, trunk_dim),
            nn.ReLU()
        )

        # 2b. Observation trunk — REMOVED.
        #     In both TinyHanabi (DecPOMDP) and MyHanabi, observations are fully joint:
        #     z^i_{t+1} == z^{-i}_{t+1} for all t > 0 (both players see the same event).
        #     Predicting z^{-i} from z^i is therefore a trivial identity mapping that
        #     carries no useful signal. Kept here for documentation and rollback if future
        #     game variants introduce asymmetric observations.
        # self.obs_trunk_input_dim = char_embed_dim + mental_embed_dim + obs_dim + action_dim
        # self.obs_trunk = nn.Sequential(
        #     nn.Linear(self.obs_trunk_input_dim, trunk_dim),
        #     nn.ReLU(),
        #     nn.Linear(trunk_dim, trunk_dim),
        #     nn.ReLU()
        # )

        # 3. Heads
        # Head A: Action Prediction -> P(a^{-i}_t)
        self.action_head = nn.Linear(trunk_dim, action_dim)

        # Head B: Observation Prediction -> P(z^{-i}_{t+1})  — REMOVED (see obs_trunk above)
        # self.observation_head = nn.Linear(trunk_dim, obs_dim)

    def forward(self, past_episodes, current_history, current_obs):
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
        x_action = torch.cat([e_char, e_mental], dim=1)
        action_features = self.action_trunk(x_action)
        action_logits = self.action_head(action_features)

        # Observation prediction — REMOVED (obs_trunk disabled, z^i == z^{-i} in current games).
        # Rollback reference:
        # act = self.null_action.expand(current_obs.shape[0], -1)
        # x_obs = torch.cat([e_char, e_mental, current_obs, act], dim=1)
        # obs_features = self.obs_trunk(x_obs)
        # next_obs_pred = self.observation_head(obs_features)

        return action_logits, None, identity_logits  # obs_pred=None (head removed)


PAST_EPISODES_CONTEXT = 5
DEFAULT_NUM_AGENT_TYPES = 4 # Based on the number of baseline experiments used for WM training
N_DECIMALS_FOR_BELIEF = 5 # For robust hashing of belief probabilities (e.g., 0.12345)


class ToM_PBVI_Agent(ModelBasedAgent):
    def __init__(
        self,
        env: Game,
        num_cards: int,
        num_actions: int,
        world_model: ToM_WorldModel,
        world_model_config: dict[str, Any],
        ensemble: np.ndarray,
        spec_ensembles: dict[str, np.ndarray] | None = None,
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

        # Joint ensemble: "mixed" + one per specialized agent type.
        # The agent maintains one CharNet embedding per ensemble and optimises
        # over the full (type × world) belief space simultaneously.
        self.all_ensembles: dict[str, np.ndarray] = {"mixed": ensemble}
        if spec_ensembles:
            self.all_ensembles.update(spec_ensembles)
        self.ensemble_tensors: dict[str, torch.Tensor] = {
            name: torch.tensor(ens, dtype=torch.float32, device=device).unsqueeze(0)
            for name, ens in self.all_ensembles.items()
        }
        self.past_episode_context = ensemble.shape[0]

        # Caches
        self.worlds_cache: dict[tuple, list[tuple]] = {}
        self.legal_actions_cache: dict[tuple, tuple] = {}

        # PBVI planning tables
        self.policy: dict[tuple, int] = {}
        # Alpha-vectors: priv_h -> type_name -> {world -> backed-up value}
        self.alpha_vectors: dict[tuple, dict[str, dict[tuple, float]]] = {}
        # Per-type Bayesian world beliefs: type_name -> priv_h -> {world -> prob}
        self.beliefs_per_type: dict[str, dict[tuple, dict[tuple, float]]] = {}
        # Marginal type posterior: priv_h -> {type_name -> prob}
        self.type_posteriors: dict[tuple, dict[str, float]] = {}

        self._init_tables()
        return

    # Initialisation
    def _init_tables(self):
        pbar = tqdm(self.all_private_histories, desc="Init Tables", leave=False)
        
        for history, done, _, _ in pbar:
            worlds = self._get_consistent_worlds(history)

            if done or not worlds:
                continue

            if self.is_decpomdp:
                legal = tuple(range(self.num_actions))
            else:
                _, legal = self.env.num_legal_actions(history)
            self.legal_actions_cache[history] = legal
            self.policy[history] = random.choice(legal)
        return


    # Consistent-world computation  (unchanged)
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

    def _predict_partner_policy(
        self,
        world,
        next_obs,
        ens_tensor: torch.Tensor,
    ) -> np.ndarray:
        """Return P(a_partner) as a numpy array of shape (action_dim,).

        We always predict the partner's action (never our own), so own_action
        is always irrelevant — a zero vector is passed to the obs_trunk input.

        Parameters
        ----------
        world      : consistent joint-history tuple (deal + past actions)
        next_obs   : own follow-up observation at this step
        ens_tensor : CharNet context for a specific partner type, shape (1, N, seq, feat)
        """
        deal_len = 2 if self.is_decpomdp else 4
        world_hands = list(world[:deal_len])
        actions = list(world[deal_len:])
        joint_h = [world_hands] + actions

        h_enc = np.zeros((self.max_seq_len, self.joint_obs_dim), dtype=np.float32)
        for i, obs in enumerate(joint_h):
            h_enc[i] = _encode_joint_observation(obs, self.joint_obs_dim, self.env)

        # Include the focal agent's own observation so the history matches
        # what the world model saw during training at this timestep.
        own_step_idx = len(joint_h)
        if own_step_idx < self.max_seq_len:
            h_enc[own_step_idx] = _encode_joint_observation(next_obs, self.joint_obs_dim, self.env)

        z_enc = _encode_observation(next_obs, self.obs_dim, self.env)

        hist_t = torch.tensor(h_enc, dtype=torch.float32, device=self.device).unsqueeze(0)
        obs_t  = torch.tensor(z_enc, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits, _, _ = self.world_model(ens_tensor, hist_t, obs_t)
        return torch.softmax(logits, dim=-1).cpu().numpy()[0]

    def _query_action_probs(
        self,
        world_prefix: tuple,
        next_obs,
        ens_tensor: torch.Tensor,
    ) -> np.ndarray:
        """Return P(next_action | history_prefix) for Bayesian belief updates.

        Encodes *only* the world prefix (deal + past actions) without appending
        the focal agent's next observation — matching training at that timestep.

        Parameters
        ----------
        world_prefix : tuple  — deal + actions up to (not including) the one being predicted
        next_obs              — own follow-up observation (matches training's priv_z_enc)
        ens_tensor            — CharNet context for a specific partner type
        """
        deal_len = 2 if self.is_decpomdp else 4
        world_hands = list(world_prefix[:deal_len])
        actions = list(world_prefix[deal_len:])
        joint_h = [world_hands] + actions

        h_enc = np.zeros((self.max_seq_len, self.joint_obs_dim), dtype=np.float32)
        for i, obs in enumerate(joint_h):
            h_enc[i] = _encode_joint_observation(obs, self.joint_obs_dim, self.env)

        z_enc = _encode_observation(next_obs, self.obs_dim, self.env)

        hist_t = torch.tensor(h_enc, dtype=torch.float32, device=self.device).unsqueeze(0)
        obs_t  = torch.tensor(z_enc, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits, _, _ = self.world_model(ens_tensor, hist_t, obs_t)
        return torch.softmax(logits, dim=-1).cpu().numpy()[0]

    # ------------------------------------------------------------------
    # Bayesian belief computation
    # ------------------------------------------------------------------

    def _compute_beliefs(self):
        """Compute the joint Bayesian posterior b(w, k | h^i) for every
        non-terminal private history h^i, over all consistent worlds w and
        all ensemble types k.

        The joint posterior factorises as:

            b(w, k | h^i) ∝ P(k) · ∏_{partner steps t} P_WM^k(a^{-i}_t | prefix_t)

        with a uniform prior P(k) = 1/K.  After normalisation over all
        (w, k) pairs we recover:

            beliefs_per_type[k][h^i][w]  — world belief conditioned on type k
            type_posteriors[h^i][k]      — marginal type posterior P(k | h^i)

        Root histories receive a uniform prior over all (w, k) pairs.
        """
        deal_len = 2 if self.is_decpomdp else 4
        type_names = list(self.all_ensembles.keys())
        n_types = len(type_names)

        # Re-initialise per-type belief dicts
        for k in type_names:
            self.beliefs_per_type[k] = {}
        self.type_posteriors.clear()

        pbar = tqdm(self.all_private_histories, desc="Belief computation", leave=False)
        for priv_h, done, _turn_id, _ in pbar:
            if done:
                continue
            worlds = self._get_consistent_worlds(priv_h)
            if not worlds:
                continue

            actions_in_h = priv_h[deal_len:]

            # Determine which player we are from the masked cards
            if self.is_decpomdp:
                we_are_p0 = (priv_h[0] == -1)
            else:
                we_are_p0 = (priv_h[0] == -1 and priv_h[1] == -1)

            if not actions_in_h:
                # Root: uniform joint prior over all (type, world) pairs
                u_world = 1.0 / len(worlds)
                u_type  = 1.0 / n_types
                for k in type_names:
                    self.beliefs_per_type[k][priv_h] = {w: u_world for w in worlds}
                self.type_posteriors[priv_h] = {k: u_type for k in type_names}
                continue

            # ----------------------------------------------------------------
            # log P(observed partner actions | world w, type k)
            # for every (k, w) pair.
            # ----------------------------------------------------------------
            log_joint: dict[tuple[str, tuple], float] = {}

            pbar_k = tqdm(type_names, desc="  Types", leave=False)
            for k in pbar_k:
                pbar_k.set_postfix({"type": k})
                ens_t = self.ensemble_tensors[k]
                for w in worlds:
                    w_actions = w[deal_len:]
                    lb = 0.0
                    for step_idx in range(len(w_actions)):
                        is_p0_action = (step_idx % 2 == 0)
                        is_partner   = (is_p0_action != we_are_p0)
                        if not is_partner:
                            continue
                        prefix       = w[:deal_len + step_idx]
                        own_next_obs = w_actions[step_idx]  # both players see same event
                        probs = self._query_action_probs(prefix, own_next_obs, ens_t)
                        partner_action = own_next_obs if self.is_decpomdp else own_next_obs[0]
                        lb += np.log(float(probs[partner_action]) + 1e-10)
                    log_joint[(k, w)] = lb

            # Log-sum-exp normalisation over all (k, w) pairs
            max_lb = max(log_joint.values())
            joint = {kw: np.exp(lb - max_lb) for kw, lb in log_joint.items()}
            total = sum(joint.values())
            if total < 1e-30:
                # Numerical underflow — fall back to uniform
                u_world = 1.0 / len(worlds)
                u_type  = 1.0 / n_types
                for k in type_names:
                    self.beliefs_per_type[k][priv_h] = {w: u_world for w in worlds}
                self.type_posteriors[priv_h] = {k: u_type for k in type_names}
                continue

            joint = {kw: v / total for kw, v in joint.items()}

            # Marginalise → type posterior P(k | h^i) and world belief b_k(w | h^i)
            type_post: dict[str, float] = {}
            for k in type_names:
                pk = sum(joint[(k, w)] for w in worlds)
                type_post[k] = pk
                if pk > 1e-30:
                    self.beliefs_per_type[k][priv_h] = {
                        w: joint[(k, w)] / pk for w in worlds
                    }
                else:
                    self.beliefs_per_type[k][priv_h] = {w: 1.0 / len(worlds) for w in worlds}
            self.type_posteriors[priv_h] = type_post

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
        # Compute Bayesian beliefs before the backward sweep
        self._compute_beliefs()

        max_delta = 0.0

        pbar = tqdm(self.all_private_histories, desc="ToM-PBVI sweep", leave=False)
        for priv_h, done, _, _ in pbar:
            if done:
                continue

            worlds = self._get_consistent_worlds(priv_h)
            if not worlds:
                continue

            best_alphas, best_action = self._pbvi_backup(priv_h, worlds)

            # Convergence delta: joint expected value under the full (type × world) belief
            type_names = list(self.all_ensembles.keys())
            n_types    = len(type_names)
            type_post  = self.type_posteriors.get(
                priv_h, {k: 1.0 / n_types for k in type_names}
            )

            def _joint_val(alpha_pt):
                return sum(
                    type_post.get(k, 0.0) * sum(
                        self.beliefs_per_type.get(k, {}).get(priv_h, {}).get(w, 0.0)
                        * alpha_pt.get(k, {}).get(w, 0.0)
                        for w in worlds
                    )
                    for k in type_names
                )

            old_alphas = self.alpha_vectors.get(priv_h, {})
            new_val = _joint_val(best_alphas)
            old_val = _joint_val(old_alphas)
            max_delta = max(max_delta, abs(new_val - old_val))

            # In-place update so shallower backups see fresh values
            self.alpha_vectors[priv_h] = best_alphas
            self.policy[priv_h] = best_action

            pbar.set_postfix({"max_delta": f"{max_delta:.6f}"})

        return max_delta
    
    def _pbvi_backup(
        self,
        priv_h: tuple,
        worlds: list[tuple],
    ) -> tuple[dict[str, dict[tuple, float]], int]:
        """Point-based backup for a single belief point over the joint
        (type × world) space.

        For each legal own-action a_i and each ensemble type k, builds an
        alpha-vector  alpha_a^k : world -> float  using the type-k world model
        to predict the partner's response.  Action selection maximises the
        joint expected value:

            Q(h^i, a_i) = Σ_k P(k|h^i) · Σ_w b_k(w|h^i) · alpha_a^k(w)

        Returns (best_alpha_per_type, best_action)
            where best_alpha_per_type : {type_name -> {world -> float}}
        """
        type_names    = list(self.all_ensembles.keys())
        n_types       = len(type_names)
        legal_actions = self.legal_actions_cache[priv_h]

        # Retrieve joint beliefs; fall back to uniform if not yet computed
        type_post  = self.type_posteriors.get(
            priv_h, {k: 1.0 / n_types for k in type_names}
        )
        beliefs_pt = {
            k: self.beliefs_per_type.get(k, {}).get(
                priv_h, {w: 1.0 / len(worlds) for w in worlds}
            )
            for k in type_names
        }

        best_value   = -float('inf')
        best_alphas: dict[str, dict[tuple, float]] = {}
        best_action  = legal_actions[0]

        pbar_a = tqdm(legal_actions, desc="    Actions", leave=False)
        for a_i in pbar_a:
            alpha_per_type: dict[str, dict[tuple, float]] = {}

            pbar_k = tqdm(type_names, desc="      Types", leave=False)
            for k in pbar_k:
                ens_t   = self.ensemble_tensors[k]
                alpha_k: dict[tuple, float] = {}

                for world in worlds:
                    self.env.reset(list(world))
                    try:
                        self.env.step(a_i)
                    except ValueError:
                        alpha_k[world] = 0.0
                        continue

                    if self.env.is_terminal():
                        alpha_k[world] = self.env.payoff()
                        continue

                    # Partner's stochastic response under type-k world model
                    state_after_own = list(self.env.history)
                    own_next_obs    = self.env.context()[-1]
                    partner_probs   = self._predict_partner_policy(
                        world, own_next_obs, ens_t
                    )

                    if self.is_decpomdp:
                        legal_partner = list(range(self.num_actions))
                    else:
                        _, legal_partner = self.env.num_legal_actions(
                            tuple(state_after_own)
                        )

                    legal_arr = np.array(legal_partner, dtype=np.int64)
                    lp = np.atleast_1d(partner_probs[legal_arr])
                    psum = lp.sum()
                    lp = (
                        lp / psum if psum > 1e-8
                        else np.ones(len(legal_partner)) / len(legal_partner)
                    )

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
                            # Look up successor alpha-vector for this type
                            next_av_k  = self.alpha_vectors.get(next_priv, {}).get(k, {})
                            world_val  += p_j * self.gamma * next_av_k.get(next_state, 0.0)

                    alpha_k[world] = world_val

                alpha_per_type[k] = alpha_k

            # Joint expected value: Σ_k P(k|h^i) · Σ_w b_k(w|h^i) · alpha_a^k(w)
            value = sum(
                type_post.get(k, 0.0) * sum(
                    beliefs_pt[k].get(w, 0.0) * alpha_per_type[k].get(w, 0.0)
                    for w in worlds
                )
                for k in type_names
            )
            if value > best_value:
                best_value  = value
                best_alphas = alpha_per_type
                best_action = a_i

        return best_alphas, best_action

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def act(self, private_history, exploit=False):
        """Select an action via a one-step joint (type × world) belief backup.

        For each candidate action a_i the expected value is:

            Q(h^i, a_i) = Σ_k P(k|h^i) · Σ_w b_k(w|h^i)
                            · [R_k(w, a_i) + γ Σ_{a_j} P_WM^k(a_j|…) · V^k(next)]

        where P(k|h^i) and b_k(w|h^i) come from the pre-computed joint belief
        and V^k(next) from the per-type alpha-vectors stored at training time.

        The env state is saved and restored so that act() is side-effect free.
        """
        worlds        = self._get_consistent_worlds(private_history)
        legal_actions = self.legal_actions_cache.get(private_history)
        type_names    = list(self.all_ensembles.keys())
        n_types       = len(type_names)

        saved_history = list(self.env.history)

        # Retrieve pre-computed joint beliefs; fall back to uniform
        type_post  = self.type_posteriors.get(
            private_history, {k: 1.0 / n_types for k in type_names}
        )
        beliefs_pt = {
            k: self.beliefs_per_type.get(k, {}).get(
                private_history, {w: 1.0 / len(worlds) for w in worlds}
            )
            for k in type_names
        }

        best_a   = legal_actions[0]
        best_val = -float('inf')

        for a_i in legal_actions:
            total_val = 0.0

            for k in type_names:
                pk    = type_post.get(k, 0.0)
                bk    = beliefs_pt[k]
                ens_t = self.ensemble_tensors[k]

                for w in worlds:
                    bwk = bk.get(w, 0.0)
                    # Skip negligible (type, world) pairs for speed
                    if pk * bwk < 1e-12:
                        continue

                    self.env.reset(list(w))
                    try:
                        self.env.step(a_i)
                    except ValueError:
                        continue

                    if self.env.is_terminal():
                        total_val += pk * bwk * self.env.payoff()
                        continue

                    state_after_own = list(self.env.history)
                    own_next_obs    = self.env.context()[-1]

                    partner_probs = self._predict_partner_policy(
                        w, own_next_obs, ens_t
                    )

                    if self.is_decpomdp:
                        legal_partner = list(range(self.num_actions))
                    else:
                        _, legal_partner = self.env.num_legal_actions(
                            tuple(state_after_own)
                        )

                    legal_arr = np.array(legal_partner, dtype=np.int64)
                    lp = np.atleast_1d(partner_probs[legal_arr])
                    psum = lp.sum()
                    lp = (
                        lp / psum if psum > 1e-8
                        else np.ones(len(legal_partner)) / len(legal_partner)
                    )

                    for idx, a_j in enumerate(legal_partner):
                        p_j = float(lp[idx])
                        self.env.reset(state_after_own)
                        try:
                            self.env.step(a_j)
                        except ValueError:
                            continue

                        if self.env.is_terminal():
                            total_val += pk * bwk * p_j * self.env.payoff()
                        else:
                            next_state = tuple(self.env.history)
                            next_priv  = self._mask_state(next_state)
                            next_av_k  = self.alpha_vectors.get(next_priv, {}).get(k, {})
                            total_val  += (
                                pk * bwk * p_j
                                * self.gamma * next_av_k.get(next_state, 0.0)
                            )

            if total_val > best_val:
                best_val = total_val
                best_a   = a_i

        self.env.reset(saved_history)
        return best_a

    def save_transition(self, *args): pass

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path):
        data = {
            "policy": self.policy,
            # alpha_vectors: priv_h -> {type_name -> {world -> value}}
            "alpha_vectors": {
                priv_h: {k: dict(wv) for k, wv in type_av.items()}
                for priv_h, type_av in self.alpha_vectors.items()
            },
            "beliefs_per_type": {
                type_name: {priv_h: dict(wv) for priv_h, wv in bpt.items()}
                for type_name, bpt in self.beliefs_per_type.items()
            },
            "type_posteriors": dict(self.type_posteriors),
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.policy.update(data["policy"])
        self.alpha_vectors.update(data.get("alpha_vectors", {}))
        self.beliefs_per_type.update(data.get("beliefs_per_type", {}))
        self.type_posteriors.update(data.get("type_posteriors", {}))

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
        self.beliefs_per_type.clear()
        self.type_posteriors.clear()
        self._init_tables()