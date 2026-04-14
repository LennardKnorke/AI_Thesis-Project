# agents/model_based/maa.py
import heapq
import os
import pickle
import random

from tqdm import tqdm

from ..base_agent import ModelBasedAgent, AgentList
from tiny_game import DecPOMDP, MyHanabi, Game, get_all_possible_histories


class MAA_Agent(ModelBasedAgent):
    """Thin executor whose policy is maintained by MAA_Central_Planner."""

    def __init__(self, env: Game, num_cards: int, num_actions: int, agent_id: int, policy: dict):
        super().__init__(env, num_cards, num_actions)
        self.agent_id = agent_id
        self.policy = policy  # shared reference into MAA_Central_Planner.policy

    def act(self, input_state: tuple, exploit: bool = False) -> int:
        return self.policy.get(input_state, 0)

    def train(self):                    return 0.0
    def save_transition(self, *_):      pass
    def save(self, *_):                 pass
    def load(self, *_):                 pass
    def reset(self):                    pass


class MAA_Central_Planner(AgentList):
    """
    Multi-Agent A* (MAA*) planner for Dec-POMDPs.

    Based on Szer, Charpillet & Zilberstein (UAI 2005).

    Lifecycle
    ---------
    __init__ / _init_tables:
        Enumerate all histories, cache turn ownership and legal actions,
        build expansion order, install a random default policy.
        No value computation.

    train():
        1. Lazily compute the MDP upper-bound heuristic (once, backward DP).
        2. Run a complete A* search over partial joint policies.
        3. Install the best policy found into self.policy.
        Returns 0.0 so the engine treats one call as converged.

    F-value decomposition (Szer et al. 2005):
        F(N) = (1/|S0|) Σ_{s0} [ g(s0, N) + h(s0, N) ]
        g = actual payoff when N fully specifies the trajectory from s0
        h = MDP upper bound at the first unspecified joint state

    Heuristics
    ----------
    'mdp'    admissible: fully-observable backward DP over joint states
    'greedy' loose: max possible payoff (constant)
    """

    def __init__(
        self,
        env: Game,
        num_cards: int,
        num_actions: int,
        heuristic: str = 'mdp',
        gamma: float = 0.99,
        max_nodes: int = None,
        *_args, **_kwargs,
    ):
        self.env          = env
        self.gamma        = gamma
        self.num_cards    = num_cards
        self.num_actions  = num_actions
        self.heuristic    = heuristic
        self.max_nodes    = max_nodes
        self.is_decpomdp  = isinstance(env, DecPOMDP)
        self.is_myhanabi  = isinstance(env, MyHanabi)

        priv, joint = get_all_possible_histories(env)
        self.all_private_histories = sorted(priv,  key=lambda x: len(x[0]), reverse=True)
        self.all_joint_histories   = sorted(joint, key=lambda x: len(x[0]), reverse=True)

        self.start_states_list: list = list(env.start_states())
        self.max_payoff: float = 0.0

        # Value / policy tables
        self.mdp_values:          dict[tuple, float] = {}  # joint_h  -> MDP upper bound
        self.policy:              dict[tuple, int]   = {}  # priv_obs -> action
        self.legal_actions_cache: dict[tuple, tuple] = {}  # priv_obs -> legal actions
        self.joint_turn:          dict[tuple, int]   = {}  # joint_h  -> acting player

        self._mdp_precomputed: bool  = False
        self.best_reward:      float = -float('inf')

        self._init_tables()

        agent_0 = MAA_Agent(env, num_cards, num_actions, 0, self.policy)
        agent_1 = MAA_Agent(env, num_cards, num_actions, 1, self.policy)
        super().__init__([agent_0, agent_1])

    # ------------------------------------------------------------------
    @property
    def centralized_planning(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _mask_state(self, state: tuple, turn_id: int) -> tuple:
        s = list(state)
        if self.is_decpomdp:
            s[0 if turn_id == 0 else 1] = -1
        else:
            if turn_id == 0:
                s[0] = -1; s[1] = -1
            else:
                s[2] = -1; s[3] = -1
        return tuple(s)

    # ------------------------------------------------------------------
    # Initialization — no value computation
    # ------------------------------------------------------------------

    def _init_tables(self):
        """
        Populate joint_turn, legal_actions_cache, expansion_order.
        Install a random default policy so act() is safe before train().
        """
        # Turn ownership + max-payoff scan
        pbar = tqdm(self.all_joint_histories, desc="MAA* Init JH", leave=False)
        for jh, done, turn_id, reward in pbar:
            self.joint_turn[jh] = turn_id
            if done:
                self.max_payoff = max(self.max_payoff, reward)

        # Legal actions per private observation + random default policy
        pbar = tqdm(self.all_private_histories, desc="MAA* Init JH", leave=False)
        for priv_h, done, _, _ in pbar:
            if done:
                continue
            if self.is_decpomdp:
                legal = tuple(range(self.num_actions))
            else:
                _, legal = self.env.num_legal_actions(priv_h)
            self.legal_actions_cache[priv_h] = legal
            self.policy[priv_h] = random.choice(legal)

        # Expansion order: ascending history length (root first), then fewest legal
        # actions first at the same depth — smaller branching factor enables earlier
        # pruning without changing the algorithm's correctness.
        self.expansion_order: list[tuple] = [
            h for h, done, _, _ in sorted(
                self.all_private_histories,
                key=lambda x: (len(x[0]), len(self.legal_actions_cache.get(x[0], ())))
            )
            if not done and h in self.legal_actions_cache
        ]

    # ------------------------------------------------------------------
    # MDP upper-bound heuristic — lazy, computed once per search
    # ------------------------------------------------------------------

    def _compute_mdp_heuristic(self):
        """
        Backward DP over joint histories assuming full observability.
        V_mdp(s) >= V_decpomdp(s), so it is an admissible upper bound.
        Processed longest-first so successor values are already available.
        """
        for jh, done, turn_id, reward in self.all_joint_histories:
            if done:
                self.mdp_values[jh] = reward
                continue
            if self.is_decpomdp:
                legal = list(range(self.num_actions))
            else:
                _, legal = self.env.num_legal_actions(jh)
            best = -float('inf')
            for a in legal:
                self.env.reset(list(jh))
                try:
                    self.env.step(a)
                except ValueError:
                    continue
                if self.env.is_terminal():
                    v = self.env.payoff()
                else:
                    v = self.gamma * self.mdp_values.get(tuple(self.env.history), 0.0)
                if v > best:
                    best = v
            self.mdp_values[jh] = best if best > -float('inf') else 0.0
        self._mdp_precomputed = True

    # ------------------------------------------------------------------
    # F-value computation
    # ------------------------------------------------------------------

    def _get_heuristic(self, joint_state: tuple) -> float:
        if self.heuristic == 'mdp':
            return self.mdp_values.get(joint_state, self.max_payoff)
        return self.max_payoff  # 'greedy': loose constant bound

    def _simulate_one_deal(self, start_state: tuple, partial: dict) -> tuple[float, float]:
        """
        Follow `partial` from `start_state` until it ends or hits an unassigned obs.
        Returns (actual_reward, heuristic) where exactly one is non-zero.
        """
        self.env.reset(list(start_state))
        current = tuple(start_state)

        while not self.env.is_terminal():
            turn_id  = self.joint_turn.get(current, 0 if len(current) % 2 == 0 else 1)
            priv_obs = self._mask_state(current, turn_id)

            if priv_obs not in partial:
                return 0.0, self._get_heuristic(current)

            self.env.step(partial[priv_obs])
            current = tuple(self.env.history)

        return self.env.payoff(), 0.0

    def _compute_fvalue(self, policy_items: tuple) -> float:
        """F(N) = (1/|S0|) Σ_{s0} [ g(s0, N) + h(s0, N) ]"""
        partial = dict(policy_items)
        total = 0.0
        for s0 in self.start_states_list:
            g, h = self._simulate_one_deal(s0, partial)
            total += g + h
        return total / len(self.start_states_list)

    # ------------------------------------------------------------------
    # Training — one complete MAA* search
    # ------------------------------------------------------------------

    def _greedy_rollout_value(self) -> float:
        """
        Evaluate the current (random default) policy across all start states.
        Used to seed best_reward before A* begins so pruning kicks in immediately.
        """
        total = 0.0
        for s0 in self.start_states_list:
            g, _ = self._simulate_one_deal(s0, self.policy)
            total += g
        return total / len(self.start_states_list)

    def train(self) -> float:
        """
        Run one MAA* search over all partial joint policies (anytime variant).

        Optimizations vs. baseline A*:
        - Greedy lower bound seeds best_reward before search, enabling immediate pruning.
        - Expansion order sorts by fewest legal actions at equal depth (set in _init_tables).
        - max_nodes cutoff: stops after N expansions and returns best complete policy
          found so far. MAA* is an anytime algorithm — early termination is valid.

        Returns 0.0 to signal convergence (one call is sufficient).
        """
        if not self._mdp_precomputed and self.heuristic == 'mdp':
            self._compute_mdp_heuristic()

        # Seed best_reward with a greedy lower bound so pruning works from the start
        best_reward:  float = self._greedy_rollout_value()
        best_policy:  dict  = dict(self.policy)

        open_list:    list  = []
        node_counter: int   = 0
        nodes_expanded: int = 0

        empty = tuple()
        f0 = self._compute_fvalue(empty)
        heapq.heappush(open_list, (-f0, node_counter, empty))
        node_counter += 1

        pbar = tqdm(desc="MAA* Search", unit="node", leave=False)
        while open_list:
            neg_f, _, policy_items = heapq.heappop(open_list)
            f = -neg_f

            nodes_expanded += 1
            pbar.update(1)
            pbar.set_postfix({
                "open":      len(open_list),
                "best":      f"{best_reward:.4f}",
                "f":         f"{f:.4f}",
            })

            # Anytime cutoff
            if self.max_nodes is not None and nodes_expanded >= self.max_nodes:
                break

            if f <= best_reward:
                continue

            partial = dict(policy_items)

            # Find the next unassigned private observation (root-first order)
            next_obs = next((o for o in self.expansion_order if o not in partial), None)

            if next_obs is None:
                # Complete policy — check if best so far
                if f > best_reward:
                    best_reward = f
                    best_policy = partial
                continue

            # Expand: one child per legal action at next_obs
            legal = self.legal_actions_cache.get(next_obs, tuple(range(self.num_actions)))
            for a in legal:
                child_items = tuple(sorted({**partial, next_obs: a}.items()))
                child_f = self._compute_fvalue(child_items)
                if child_f > best_reward:
                    heapq.heappush(open_list, (-child_f, node_counter, child_items))
                    node_counter += 1

        pbar.close()
        self.policy.clear()
        self.policy.update(best_policy)
        self.best_reward = best_reward
        return 0.0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str):
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(
                {'policy': dict(self.policy), 'best_reward': self.best_reward},
                f, protocol=pickle.HIGHEST_PROTOCOL
            )

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            return
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.policy.clear()
        self.policy.update(data.get('policy', {}))
        self.best_reward = data.get('best_reward', -float('inf'))

    # ------------------------------------------------------------------
    # Reset — reinitialize for a new attempt
    # ------------------------------------------------------------------

    def reset(self):
        self.policy.clear()
        self.mdp_values.clear()
        self._mdp_precomputed = False
        self.best_reward = -float('inf')
        # Re-install random default policy (game structure caches unchanged)
        for priv_h, legal in self.legal_actions_cache.items():
            self.policy[priv_h] = random.choice(legal)
