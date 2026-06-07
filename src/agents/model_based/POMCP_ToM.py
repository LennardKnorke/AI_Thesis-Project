from collections import defaultdict
import math
import numpy as np
import pickle
import random
import torch
from tqdm import tqdm

from tiny_game import *

from ..base_agent import ModelBasedAgent, AgentList, BaseAgent
from .worldmodel import ToM_WorldModel, _encode_joint_observation, _encode_observation


class _Node:
    """MCTS tree node — visit counts, Q-values, and particle belief B."""
    __slots__ = ('N', 'N_ha', 'Q_ha', 'Q2_ha', 'P_ha', 'children', 'expanded', 'B')

    def __init__(self):
        self.N        = 0
        self.N_ha     = defaultdict(int)
        self.Q_ha     = defaultdict(float)
        self.Q2_ha    = defaultdict(float)   # running mean of squared returns (UCB1-Tuned)
        self.P_ha     = {}                   # PUCT prior P(a|h), populated lazily
        self.children = defaultdict(dict)
        self.expanded = False
        self.B        = defaultdict(int)


class POMCP_ToM_Agent(ModelBasedAgent):
    """
    CTDE – Model-Based: Partially Observable Monte Carlo Planning with Theory of Mind.

    Runs POMCP from the focal agent's perspective while predicting the partner's
    action via a learned ToM world model (ToM_WorldModel). The world model is
    conditioned on a rolling ensemble of past episodes to infer partner type.
    """

    def __init__(
            self,
            env: Game, game_name: str, num_cards: int, num_actions: int,
            world_model: ToM_WorldModel, world_model_config: dict,
            base_ensemble: np.ndarray,
            agent_id: int = 0,
            n_simulations: int = 1_000,
            exploration_constant: float = 1.41,
            selection_rule: str = 'ucb1',
            update_rule: str = 'uniform',
            cheat_partner: BaseAgent | None = None,
            device: str = "cpu",
            gamma: float = 0.99,
    ):
        super().__init__(env, num_cards, num_actions)

        assert update_rule in ['uniform', 'update']
        assert selection_rule in ["ucb1", "ucb1_tuned", "puct"]

        self.game_name      = game_name
        self.agent_id       = agent_id
        self.gamma          = gamma

        self.policy:               dict[tuple, int]            = {}
        self.legal_actions_cache:  dict[tuple, tuple[int, ...]] = {}

        # MCTS state
        self.root_node      = None
        self._last_action   = None
        self.n_simulations  = n_simulations
        self.c              = exploration_constant
        self.selection_rule = selection_rule
        self._particles     = None
        self._belief_update_rule = update_rule

        # ToM world model
        self.device           = device
        self.base_ensemble    = base_ensemble.copy()
        self.current_ensemble = base_ensemble.copy()
        self.current_mask     = np.zeros(base_ensemble.shape[0], dtype=np.float32)
        self.past_episode_context = base_ensemble.shape[0]

        self.max_seq_len   = world_model_config['max_seq_len']
        self.obs_dim       = world_model_config['obs_dim']
        self.joint_obs_dim = world_model_config['joint_obs_dim']
        self.action_dim    = world_model_config['action_dim']
        self.wm_config     = world_model_config

        self.cheat_partner = cheat_partner
        self.world_model   = world_model.to(device)
        self.world_model.eval()

        # Torch mirrors of the ensemble/mask (fed to world_model.forward)
        self.ensemble_tens = torch.tensor(self.current_ensemble, dtype=torch.float32, device=self.device)
        self.mask_tens     = torch.tensor(self.current_mask,     dtype=torch.float32, device=self.device)

        self._init_tables()

    def _init_tables(self):
        for priv_h, legal_as, done, _, _ in PRIV_HISTORIES[self.game_name]:
            if not done:
                self.policy[priv_h] = random.choice(legal_as)
                self.legal_actions_cache[priv_h] = legal_as


    # ------------------------------------------------------------------
    # Core act / plan / reset
    # ------------------------------------------------------------------

    def act(self, priv_history, *args, **kwargs):
        saved = tuple(self.env.history)
        self.reuse_tree(priv_history)
        action = self.plan(priv_history)
        self._last_action = action
        self.env.reset(list(saved))
        return action

    def reuse_tree(self, priv_h=None):
        """Navigate to the subtree for (last_action, priv_h); reset tree if unavailable."""
        if priv_h is None:
            self.root_node = None
            self._last_action = None
            return

        if (self.root_node is not None
                and self._last_action is not None
                and self._last_action in self.root_node.children
                and priv_h in self.root_node.children[self._last_action]):
            self.root_node = self.root_node.children[self._last_action][priv_h]
        else:
            self.root_node = _Node()
        self._last_action = None

    def plan(self, priv_history):
        legal_as = self.legal_actions_cache[priv_history]
        self._update_belief(priv_history)
        states, weights = self._sample_particles(priv_h=priv_history)

        assert states,     f"empty belief for {priv_history}"
        assert legal_as,   f"empty actions for {priv_history}"

        if self.root_node is None:
            self.root_node = _Node()

        for _ in range(self.n_simulations):
            particle = random.choices(states, weights=weights, k=1)[0]
            node, leaf_state, leaf_ph, path, done = self._selection(self.root_node, particle, priv_history)
            if not done:
                self._expansion(node, leaf_ph)
                G = self._simulate_rollout(leaf_state)
            else:
                G = 0.0
            self._backup(G, path)

        return max(legal_as, key=lambda a: self.root_node.Q_ha[a])

    def reset(self):
        self.current_ensemble = self.base_ensemble.copy()
        self.current_mask     = np.zeros(self.past_episode_context, dtype=np.float32)
        self.ensemble_tens = torch.tensor(self.current_ensemble, dtype=torch.float32, device=self.device)
        self.mask_tens     = torch.tensor(self.current_mask,     dtype=torch.float32, device=self.device)
        self.root_node    = None
        self._last_action = None
        self._init_tables()

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'policy':     self.policy,
                'c_ens':      self.current_ensemble,
                'c_mask_ens': self.current_mask,
            }, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.current_mask     = data['c_mask_ens']
        self.current_ensemble = data['c_ens']
        self.policy           = data['policy']

    def train(self):            return 0.0
    def save_transition(self, *_): pass


    # ------------------------------------------------------------------
    # Environment helpers
    # ------------------------------------------------------------------

    def _mask_state(self, jh: tuple, turn_id: int):
        """Return private observation by masking the focal agent's own cards."""
        priv_h = list(jh)
        if self.is_decpomdp:
            priv_h[0 if turn_id == 0 else 1] = -1
        else:
            if turn_id == 0:
                priv_h[0] = priv_h[1] = -1
            else:
                priv_h[2] = priv_h[3] = -1
        return tuple(priv_h)


    # ------------------------------------------------------------------
    # MCTS core: selection, expansion, rollout, backup
    # ------------------------------------------------------------------

    def _selection(self, root_node: _Node, particle: tuple, priv_history: tuple):
        path = []
        node, s_t, h_t = root_node, particle, priv_history

        while node.expanded:
            legal_as = self.legal_actions_cache[h_t]
            if self.selection_rule == "ucb1":
                a_t = self._ucb1_action(node, legal_as)
            elif self.selection_rule == "ucb1_tuned":
                a_t = self._ucb1_tuned_policy(node, legal_as)
            else:
                a_t = self._puct_policy(node, legal_as, s_t)

            s_t1, h_t1, r_t, done = self._env_step(s_t, h_t, a_t)
            path.append((node, a_t, r_t))

            if done:
                return node, s_t1, h_t1, path, True

            node = node.children[a_t].setdefault(h_t1, _Node())
            node.B[tuple(s_t1)] += 1
            s_t, h_t = s_t1, h_t1

        return node, s_t, h_t, path, False

    def _ucb1_action(self, node: _Node, legal_actions: tuple) -> int:
        for a in legal_actions:
            if node.N_ha[a] == 0:
                return a
        log_n = math.log(max(node.N, 1))
        return max(legal_actions, key=lambda a: node.Q_ha[a] + self.c * math.sqrt(log_n / node.N_ha[a]))

    def _ucb1_tuned_policy(self, node: _Node, legal_actions: tuple) -> int:
        for a in legal_actions:
            if node.N_ha[a] == 0:
                return a
        log_n = math.log(max(node.N, 1))

        def score(a):
            n_a    = node.N_ha[a]
            emp_var = max(0.0, node.Q2_ha[a] - node.Q_ha[a] ** 2)
            V_a    = min(0.25, emp_var + math.sqrt(2.0 * log_n / n_a))
            return node.Q_ha[a] + self.c * math.sqrt(log_n / n_a * V_a)
        return max(legal_actions, key=score)

    def _puct_policy(self, node: _Node, legal_actions: tuple, jh: tuple) -> int:
        for a in legal_actions:
            if node.N_ha[a] == 0:
                return a
        if not node.P_ha:
            node.P_ha = self._compute_puct_prior(jh, legal_actions)
        sqrt_N = math.sqrt(max(node.N, 1))
        return max(legal_actions, key=lambda a: node.Q_ha[a] + self.c * node.P_ha.get(a, 0.0) * sqrt_N / (1 + node.N_ha[a]))

    def _compute_puct_prior(self, jh: tuple, legal_actions: tuple) -> dict:
        """Query the world model for action priors; normalise over legal actions."""
        h_enc = self._encode_history(jh)
        with torch.no_grad():
            action_logits, _, _, _ = self.world_model(
                self.ensemble_tens.unsqueeze(0),
                self.mask_tens.unsqueeze(0),
                torch.tensor(h_enc, dtype=torch.float32, device=self.device).unsqueeze(0),
                torch.zeros(1, self.obs_dim, dtype=torch.float32, device=self.device),
            )
            probs = torch.softmax(action_logits, dim=1)[0].cpu().numpy()
        masked = np.zeros(self.action_dim, dtype=np.float32)
        masked[list(legal_actions)] = probs[list(legal_actions)]
        total = masked.sum()
        masked = masked / total if total > 1e-12 else np.full(self.action_dim, 1.0 / len(legal_actions))
        return {a: float(masked[a]) for a in legal_actions}

    def _env_step(self, state, priv_h, action):
        """Step env with focal action then partner prediction; return (next_state, next_priv_h, reward, done)."""
        prev_state = tuple(self.env.history)
        turn_id    = 0 if priv_h[0] == -1 else 1

        self.env.reset(list(state))
        self.env.step(action)

        next_state = self.env.history
        done = self.env.is_terminal()
        if done:
            next_h = self._mask_state(next_state, turn_id)
            reward = self.env.payoff()
            self.env.reset(list(prev_state))
            return next_state, next_h, reward, done

        partner_action = (self._cheat_partner_prediction(next_state, 1 - turn_id)
                          if self.cheat_partner is not None
                          else self._predict_partner_action(next_state, 1 - turn_id))
        self.env.step(partner_action)

        next_state = self.env.history
        next_h     = self._mask_state(next_state, turn_id)
        done       = self.env.is_terminal()
        reward     = self.env.payoff() if done else 0.0

        self.env.reset(list(prev_state))
        return next_state, next_h, reward, done

    def _expansion(self, node: _Node, priv_h: tuple):
        legal_as = self.legal_actions_cache[priv_h]
        for a in legal_as:
            node.N_ha[a]
            node.Q_ha[a]
        node.expanded = True

    def _simulate_rollout(self, state: tuple) -> float:
        """Random-policy rollout from state until terminal; returns discounted return."""
        G, discount = 0.0, 1.0
        prev_state  = tuple(self.env.history)
        self.env.reset(list(state))
        s_t  = state
        done = self.env.is_terminal()

        while not done:
            h_i     = self._mask_state(s_t, self.agent_id)
            focal_a = random.choice(self.legal_actions_cache[h_i])
            self.env.step(focal_a)
            s_t1 = tuple(self.env.history)
            done = self.env.is_terminal()

            if not done:
                partner_a = (self._cheat_partner_prediction(s_t1, 1 - self.agent_id)
                             if self.cheat_partner is not None
                             else self._predict_partner_action(s_t1, 1 - self.agent_id))
                self.env.step(partner_a)
                s_t1 = tuple(self.env.history)
                done = self.env.is_terminal()

            G  += discount * self.env.payoff()
            s_t = s_t1
            discount *= self.gamma

        self.env.reset(list(prev_state))
        return G

    def _backup(self, expected_reward: float, path: list[tuple[_Node, int, float]]):
        total = expected_reward
        for node, action, reward in reversed(path):
            total           = reward + self.gamma * total
            node.N          += 1
            node.N_ha[action] += 1
            n_a = node.N_ha[action]
            # Incremental mean; Q2 stores E[X^2] for UCB1-Tuned variance.
            node.Q_ha[action]  += (total       - node.Q_ha[action])  / n_a
            node.Q2_ha[action] += (total*total - node.Q2_ha[action]) / n_a


    # ------------------------------------------------------------------
    # Belief management
    # ------------------------------------------------------------------

    def _sample_particles(self, priv_h):
        if self._particles is None:
            self._particles = self.__calc_uniform_belief(priv_h)
        return list(self._particles.keys()), list(self._particles.values())

    def __calc_uniform_belief(self, priv_h):
        cons = CONSISTENT_WORLDS[self.game_name][priv_h]
        val  = 1 / len(cons)
        return {s: val for s in cons}

    def _update_belief(self, priv_h):
        """Update particle belief; falls back to uniform when insufficient MCTS data."""
        if self._belief_update_rule != 'update':
            self._particles = None
            return

        if self.root_node is None or not self.root_node.B:
            if self.agent_id == 0:
                self._particles = None
            else:
                # P1 weights start states by the probability of P0's observed first action.
                cons     = CONSISTENT_WORLDS[self.game_name][priv_h]
                deal_len = 2 if self.is_decpomdp else 4
                _particles = {}
                for state in cons:
                    partner_priv_h = self._mask_state(state[:deal_len], 1 - self.agent_id)
                    action = state[2] if self.is_decpomdp else state[4][0]
                    _particles[state] = self._inverse_prop(state, partner_priv_h, action)
                total = sum(_particles.values())
                if total > 1e-12:
                    self._particles = {s: v / total for s, v in _particles.items()}
                else:
                    self._particles = None  # world model gives zero prob to all observed actions; fall back to uniform
            return

        cons     = CONSISTENT_WORLDS[self.game_name][priv_h]
        filtered = {s: c for s, c in self.root_node.B.items() if s in cons}
        total    = sum(filtered.values())
        if total < 1e-12:
            tqdm.write("POMCP BELIEF WARNING C")
            self._particles = None
            return
        self._particles = {s: c / total for s, c in filtered.items()}


    # ------------------------------------------------------------------
    # ToM: partner prediction and ensemble update
    # ------------------------------------------------------------------

    def _encode_history(self, jh: tuple) -> np.ndarray:
        """Encode joint history into a right-aligned (max_seq_len, joint_obs_dim) array."""
        start_len  = 2 if self.is_decpomdp else 4
        full_obs   = [jh[:start_len]] + list(jh[start_len:])
        seq_length = min(len(full_obs), self.max_seq_len)
        h_enc = np.zeros((self.max_seq_len, self.joint_obs_dim), dtype=np.float32)
        for i, obs in enumerate(full_obs[-seq_length:]):
            h_enc[self.max_seq_len - seq_length + i] = _encode_joint_observation(obs, self.joint_obs_dim, self.env)
        return h_enc

    def _predict_partner_action(self, step_t_joint_h, partner_turn_id) -> int:
        """Sample partner action from the world model's predicted action distribution."""
        partner_priv_h = self._mask_state(step_t_joint_h, partner_turn_id)
        legal_as       = self.legal_actions_cache[partner_priv_h]
        h_enc          = self._encode_history(step_t_joint_h)

        with torch.no_grad():
            action_logits, _, _, _ = self.world_model(
                self.ensemble_tens.unsqueeze(0),
                self.mask_tens.unsqueeze(0),
                torch.tensor(h_enc, dtype=torch.float32, device=self.device).unsqueeze(0),
                torch.zeros(1, self.obs_dim, dtype=torch.float32, device=self.device),
            )
            probs = torch.softmax(action_logits, dim=1)[0].cpu().numpy()

        # Mask illegal actions and renormalise.
        legal_mask = np.zeros(self.action_dim, dtype=np.float32)
        legal_mask[list(legal_as)] = 1.0
        probs = probs * legal_mask
        total = probs.sum()
        probs = probs / total if total > 1e-12 else legal_mask / legal_mask.sum()
        # return int(np.random.choice(self.action_dim, p=probs))  # stochastic sampling
        return int(np.argmax(probs))

    def _inverse_prop(self, state, partner_priv_h, partner_a) -> float:
        """Return P(partner_a | history) for belief weighting when agent_id == 1."""
        assert self.agent_id == 1

        if self.cheat_partner is not None:
            a_for_h = self.cheat_partner.policy[partner_priv_h]
            if not isinstance(a_for_h, int):
                a_for_h = int(np.argmax(a_for_h))
            return 1.0 if partner_a == a_for_h else 0.0

        h_enc = self._encode_history(state)
        with torch.no_grad():
            action_logits, _, _, _ = self.world_model(
                self.ensemble_tens.unsqueeze(0),
                self.mask_tens.unsqueeze(0),
                torch.tensor(h_enc, dtype=torch.float32, device=self.device).unsqueeze(0),
                torch.zeros(1, self.obs_dim, dtype=torch.float32, device=self.device),
            )
            action_probs = torch.softmax(action_logits, dim=1)
        return float(action_probs[0, partner_a].item())

    def update_ensemble(self, episode_log):
        """Roll the episode ensemble with the latest episode and update torch mirrors."""
        h_enc = self._encode_history(episode_log)

        # Roll numpy buffers (persisted via save/load).
        self.current_ensemble = np.roll(self.current_ensemble, -1, axis=0)
        self.current_ensemble[-1] = h_enc
        self.current_mask = np.roll(self.current_mask, -1)
        self.current_mask[-1] = 1.0

        # Roll torch mirrors in-place.
        self.ensemble_tens = torch.roll(self.ensemble_tens, shifts=-1, dims=0)
        self.ensemble_tens[-1] = torch.tensor(h_enc, dtype=torch.float32, device=self.device)
        self.mask_tens = torch.roll(self.mask_tens, shifts=-1, dims=0)
        self.mask_tens[-1] = 1.0

    def _cheat_partner_prediction(self, step_t_joint_h, partner_turn_id) -> int:
        """Oracle partner prediction using ground-truth policy (debug only)."""
        partner_history = self._mask_state(step_t_joint_h, partner_turn_id)
        return self.cheat_partner.act(partner_history, exploit=True)
