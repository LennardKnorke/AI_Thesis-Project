# agents/model_based/pbvi.py
import os
import pickle
from collections import defaultdict
import random
from tqdm import tqdm

from ..base_agent import ModelBasedAgent, AgentList
from tiny_game import DecPOMDP, MyHanabi, Game, get_all_possible_histories


class PBVI_Agent(ModelBasedAgent):
    """Thin executor whose policy is maintained by PBVI_List."""

    def __init__(self, env, num_cards, num_actions, agent_id, policy):
        super().__init__(env, num_cards, num_actions)
        self.agent_id = agent_id
        self.policy = policy

    def act(self, input_state: tuple, exploit: bool = False) -> int:
        return self.policy[input_state]

    def train(self): return 0.0
    def save_transition(self, *args): pass
    def save(self, *args): pass
    def reset(self): pass


class PBVI_List(AgentList):
    """
    Centralized PBVI planner for a two-player sequential Dec-POMDP.

    Data structures
    ---------------
    beliefs[priv]         : {joint_hist: prob}  — uniform over consistent worlds
    alpha_vectors[priv]   : {joint_hist: float} — backed-up value per world
    transitions[jh][a]    : (next_jh | None, reward, is_terminal)
    joint_turn[jh]        : int — whose turn it is (0 or 1)
    """

    def __init__(self, env: Game, num_cards: int, num_actions: int, *args, **kwargs):
        self.env = env
        self.gamma = 0.99
        self.num_cards = num_cards
        self.num_actions = num_actions
        self.is_decpomdp = isinstance(env, DecPOMDP)
        self.is_myhanabi = isinstance(env, MyHanabi)

        all_private_histories, all_joint_histories = get_all_possible_histories(self.env)
        self.all_private_histories = sorted(all_private_histories, key=lambda x: len(x[0]), reverse=True)
        self.all_joint_histories = sorted(all_joint_histories, key=lambda x: len(x[0]), reverse=True)
    

        self.policy: dict[tuple, int] = {}
        self.legal_actions_cache: dict[tuple, tuple] = {}
        self.alpha_vectors: dict[tuple, dict[tuple, float]] = {}
        self.beliefs: dict[tuple, dict[tuple, float]] = {}
        # Precomputed deterministic transitions
        self.transitions: dict[tuple, dict[int, tuple]] = defaultdict(dict)
        self.joint_turn: dict[tuple, int] = {}

        self._init_structures()

        agent_0 = PBVI_Agent(env, num_cards, num_actions, 0, self.policy)
        agent_1 = PBVI_Agent(env, num_cards, num_actions, 1, self.policy)
        super().__init__([agent_0, agent_1])

    @property
    def centralized_planning(self):
        return True

    # ------------------------------------------------------------------
    def _mask_state(self, state: tuple, turn_id: int) -> tuple:
        """Private observation for player `turn_id`: mask their own cards."""
        s = list(state)
        if self.is_decpomdp:
            s[0 if turn_id == 0 else 1] = -1
        else:
            if turn_id == 0:
                s[0] = s[1] = -1
            else:
                s[2] = s[3] = -1
        return tuple(s)

    def _init_structures(self):
        # 1. Precompute transitions and turn ownership for every joint history
        pbar = tqdm(self.all_joint_histories, desc="Init JH", leave=False)
        for jh, done, turn_id, _reward in pbar:
            self.joint_turn[jh] = turn_id
            if done:
                continue

            if self.is_decpomdp:
                legal = tuple(range(self.num_actions))
            else:
                _, legal = self.env.num_legal_actions(jh)

            for a in legal:
                self.env.reset(list(jh))
                try:
                    self.env.step(a)
                except ValueError:
                    continue
                if self.env.is_terminal():
                    self.transitions[jh][a] = (None, self.env.payoff(), True)
                else:
                    self.transitions[jh][a] = (tuple(self.env.history), 0.0, False)

        # 2. Build belief points for every non-terminal private observation
        pbar = tqdm(self.all_private_histories, desc="Init PH", leave=False)
        for priv, done, turn_id, _ in pbar:
            if done:
                continue
            consistent = [
                jh for jh, jdone, jt, _ in self.all_joint_histories
                if not jdone and self._mask_state(jh, jt) == priv
            ]
            if not consistent:
                continue
            prob = 1.0 / len(consistent)
            self.beliefs[priv] = {jh: prob for jh in consistent}

            if self.is_decpomdp:
                legal = tuple(range(self.num_actions))
            else:
                _, legal = self.env.num_legal_actions(consistent[0])
            self.legal_actions_cache[priv] = legal
            self.policy[priv] = random.choice(legal)

    # ------------------------------------------------------------------
    def train(self) -> float:
        """
        One PBVI backward sweep.

        For each non-terminal private observation h (longest first):
          For each legal action a:
            For each world s in B(h):
              alpha_a(s) = R(s,a) + gamma * alpha_{h'}(s')
              where s' = successor joint history,
                    h' = partner's private obs at s'
          Select a* = argmax_a  b_h . alpha_a
          Store alpha_{a*} and pi(h) = a*

        Returns max |delta V| across all belief points (0.0 = converged).
        """
        max_delta = 0.0
        new_alpha: dict[tuple, dict[tuple, float]] = {}

        pbar = tqdm(self.all_private_histories, desc="PBVI sweep", leave=False)
        for priv, done, turn_id, _ in pbar:
            if done or priv not in self.beliefs:
                continue

            belief = self.beliefs[priv]
            legal = self.legal_actions_cache[priv]
            partner_turn = 1 - turn_id

            best_value = -float('inf')
            best_alpha: dict[tuple, float] = {}
            best_action = legal[0]

            for a in legal:
                # Build alpha-vector for action a over the belief support
                alpha_a: dict[tuple, float] = {}
                for s, prob in belief.items():
                    if a not in self.transitions.get(s, {}):
                        alpha_a[s] = 0.0
                        continue

                    next_s, r, terminal = self.transitions[s][a]
                    if terminal:
                        alpha_a[s] = r
                    else:
                        # Value from the PARTNER's alpha-vector at the successor
                        next_priv = self._mask_state(next_s, partner_turn)
                        next_av = self.alpha_vectors.get(next_priv, {})
                        alpha_a[s] = r + self.gamma * next_av.get(next_s, 0.0)

                # b_h . alpha_a
                value = sum(belief[s] * alpha_a[s] for s in belief)
                if value > best_value:
                    best_value = value
                    best_alpha = alpha_a
                    best_action = a

            # Track convergence
            old_value = 0.0
            if priv in self.alpha_vectors:
                old_value = sum(
                    belief[s] * self.alpha_vectors[priv].get(s, 0.0)
                    for s in belief
                )
            max_delta = max(max_delta, abs(best_value - old_value))

            new_alpha[priv] = best_alpha
            self.policy[priv] = best_action

        self.alpha_vectors = new_alpha
        return max_delta

    # ------------------------------------------------------------------
    def save(self, filepath: str):
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        data = {'policy': self.policy, 'alpha_vectors': self.alpha_vectors}
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            return
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.policy = data['policy']
        self.alpha_vectors = data['alpha_vectors']

    def reset(self):
        self.policy.clear()
        self.alpha_vectors.clear()
