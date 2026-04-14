# agents/model_based/OSarsa.py
import os
import pickle
import random
from collections import defaultdict
from tqdm import tqdm

from ..base_agent import ModelBasedAgent, AgentList
from tiny_game import DecPOMDP, MyHanabi, Game, get_all_possible_histories


class OSarsa_Agent(ModelBasedAgent):
    """Thin executor whose policy is maintained by OSarsa_Planner."""

    def __init__(self, env: Game, num_cards: int, num_actions: int, agent_id: int, policy: dict):
        super().__init__(env, num_cards, num_actions)
        self.agent_id = agent_id
        self.policy   = policy  # shared reference into OSarsa_Planner.policy

    def act(self, input_state: tuple, exploit: bool = False) -> int:  # noqa: ARG002
        return self.policy.get(input_state, 0)

    def train(self):                return 0.0
    def save_transition(self, *_):  pass
    def save(self, *_):             pass
    def load(self, *_):             pass
    def reset(self):                pass


class OSarsa_Planner(AgentList):
    """
    OSarsa: Occupancy-State Sarsa for Dec-POMDPs (sequential, 2-agent).

    Based on Dibangoye et al. (JAIR 2016) and Peralez et al. (AAAI 2025).

    Reformulates the Dec-POMDP as an occupancy-state MDP (oMDP) and applies
    a Sarsa-style Q-learning update. The occupancy state is a distribution
    over joint histories reachable from the initial belief.

    Algorithm per train() call:
        1. Policy improvement (epsilon-greedy):
           For each private history priv_h at time τ, compute:
             Q̄(priv_h, a) = (1/|worlds|) Σ_{jh ∈ worlds(priv_h)} Q[τ][(jh, a)]
           and select action argmax_a Q̄ (or random with prob ε).

        2. Backward Q-update (DP, deterministic oMDP → learning_rate = 1.0):
           Iterate τ from tau_max down to 0:
           For each reachable joint history jh and each individual action a:
             - Agent 0 (not last in round), game not terminal:
               Q[τ][(jh, a0)] = Q[τ+1][(jh_ext, π_1(priv_h_1))]
               where jh_ext = history after a0, no discount within a round.
             - Agent 1 (last in round), game not terminal:
               Q[τ][(jh, a1)] = γ · Q[τ+1][(jh_next, π_0(priv_h_0_next))]
               with environment transition and discount γ.
             - Terminal after any action:
               Q[τ][(jh, a)] = payoff()

    Convergence is detected when policy stops changing (returns 0.0).
    Multiple restarts ('attempts' engine param) escape local optima.

    Reference:
        Dibangoye et al., "Optimally Solving Dec-POMDPs as Continuous-State MDPs",
        JAIR 2016.
        Peralez et al., "OSarsa: Occupancy-State Sarsa for Dec-POMDPs",
        AAAI 2025.
    """

    def __init__(
        self,
        env:           Game,
        num_cards:     int,
        num_actions:   int,
        gamma:         float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min:   float = 0.05,
        epsilon_decay: float = 0.99,
        *_args, **_kwargs,
    ):
        self.env           = env
        self.gamma         = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.num_cards     = num_cards
        self.num_actions   = num_actions
        self.is_decpomdp   = isinstance(env, DecPOMDP)
        self.is_myhanabi   = isinstance(env, MyHanabi)

        priv, joint = get_all_possible_histories(env)
        self.all_private_histories = priv
        self.all_joint_histories   = joint

        self.policy:              dict[tuple, int]           = {}
        self.legal_actions_cache: dict[tuple, tuple]         = {}
        self.consistent_worlds:   dict[tuple, list[tuple]]   = {}
        self.joint_turn:          dict[tuple, int]           = {}
        self.reachable_by_tau:    list[list[tuple]]          = []  # [(jh, turn_id), ...]
        self.Q:                   list[dict[tuple, float]]   = []  # Q[τ][(jh, a)]
        self.tau_max:             int                        = 0
        self.current_iter:        int                        = 0

        self._init_game_structure()
        self._init_policy()

        agent_0 = OSarsa_Agent(env, num_cards, num_actions, 0, self.policy)
        agent_1 = OSarsa_Agent(env, num_cards, num_actions, 1, self.policy)
        super().__init__([agent_0, agent_1])

    # ------------------------------------------------------------------
    @property
    def centralized_planning(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _deal_length(self) -> int:
        return 2 if self.is_decpomdp else 4

    def _mask_state(self, state: tuple, turn_id: int) -> tuple:
        s = list(state)
        if self.is_decpomdp:
            s[0 if turn_id == 0 else 1] = -1
        else:
            if turn_id == 0:
                s[0] = s[1] = -1
            else:
                s[2] = s[3] = -1
        return tuple(s)

    def _get_turn(self, jh: tuple) -> int:
        turn = self.joint_turn.get(jh)
        if turn is not None:
            return turn
        # Fallback: infer from history length (alternating turns starting with 0)
        deal_len = self._deal_length()
        return (len(jh) - deal_len) % 2

    # ------------------------------------------------------------------
    # Initialization — called once; game structure does not change
    # ------------------------------------------------------------------

    def _init_game_structure(self):
        """
        Build all caches from the precomputed history sets.
        Uses inverse index for O(|joint|) consistent-worlds lookup.
        Groups joint histories by τ (number of actions taken) for the DP.
        """
        deal_len = self._deal_length()

        # 1. Turn ownership cache
        for jh, _, turn_id, _ in self.all_joint_histories:
            self.joint_turn[jh] = turn_id

        # 2. Group non-terminal joint histories by tau = len(jh) - deal_len
        tau_to_jhs: dict[int, list] = defaultdict(list)
        for jh, done, turn_id, _ in self.all_joint_histories:
            if done:
                continue
            tau = len(jh) - deal_len
            tau_to_jhs[tau].append((jh, turn_id))

        self.tau_max = max(tau_to_jhs.keys()) if tau_to_jhs else 0
        self.reachable_by_tau = [
            tau_to_jhs.get(t, []) for t in range(self.tau_max + 1)
        ]

        # Q tables: one dict per tau (+1 sentinel for terminal lookups)
        self.Q = [{} for _ in range(self.tau_max + 2)]

        # 3. Inverse index: masked_jh -> [jh, ...] for consistent-worlds lookup
        masked_to_joints: dict[tuple, list[tuple]] = defaultdict(list)
        for jh, done, turn_id, _ in self.all_joint_histories:
            if done:
                continue
            masked_to_joints[self._mask_state(jh, turn_id)].append(jh)

        # 4. Populate consistent worlds and legal actions per private history
        for priv_h, done, turn_id, _ in self.all_private_histories:
            if done:
                continue
            consistent = masked_to_joints.get(priv_h, [])
            if not consistent:
                continue
            self.consistent_worlds[priv_h] = consistent

            if self.is_decpomdp:
                legal = tuple(range(self.num_actions))
            else:
                _, legal = self.env.num_legal_actions(consistent[0])
            self.legal_actions_cache[priv_h] = legal

    # ------------------------------------------------------------------

    def _init_policy(self):
        """Initialise with a random policy."""
        self.policy.clear()
        for priv_h, done, _, _ in self.all_private_histories:
            if done or priv_h not in self.legal_actions_cache:
                continue
            self.policy[priv_h] = random.choice(self.legal_actions_cache[priv_h])

    # ------------------------------------------------------------------
    # Core algorithm components
    # ------------------------------------------------------------------

    def _greedy_action(self, priv_h: tuple, tau: int) -> int:
        """
        Greedy action for priv_h at τ: argmax_a  (1/|worlds|) Σ_{jh} Q[τ][(jh, a)]
        """
        consistent = self.consistent_worlds.get(priv_h, [])
        legal      = self.legal_actions_cache.get(priv_h, ())
        if not legal:
            return 0
        if not consistent:
            return legal[0]

        q_tab    = self.Q[tau]
        n        = len(consistent)
        best_a   = legal[0]
        best_val = -float('inf')

        for a in legal:
            val = sum(q_tab.get((jh, a), 0.0) for jh in consistent) / n
            if val > best_val:
                best_val = val
                best_a   = a
        return best_a

    def _update_policy(self, epsilon: float) -> int:
        """
        Epsilon-greedy policy improvement over all private histories.
        Returns the number of policy entries that changed.
        """
        changes  = 0
        seen     = set()

        for tau, jhs in enumerate(self.reachable_by_tau):
            for jh, turn_id in jhs:
                priv_h = self._mask_state(jh, turn_id)
                if priv_h in seen or priv_h not in self.legal_actions_cache:
                    continue
                seen.add(priv_h)

                old_a = self.policy.get(priv_h)
                if random.random() < epsilon:
                    new_a = random.choice(self.legal_actions_cache[priv_h])
                else:
                    new_a = self._greedy_action(priv_h, tau)

                self.policy[priv_h] = new_a
                if old_a != new_a:
                    changes += 1

        return changes

    def _update_Q(self):
        """
        Backward Q-update over all reachable (jh, action) pairs.

        For deterministic oMDPs the learning rate is 1.0:
            Q[τ][(jh, a)] ← q_estimate  (overwrite, not interpolate)

        Two cases (sequential 2-agent framework):
          Agent 0 (not last in round):
            q = Q[τ+1][(jh_ext, π_1(priv_h_1))]          — no discount
          Agent 1 (last in round):
            q = γ · Q[τ+1][(jh_next, π_0(priv_h_0_next))] — with discount
          Terminal after any action:
            q = payoff()
        """
        pbar = tqdm(
            range(self.tau_max, -1, -1),
            desc="OSarsa Q-backup",
            leave=False,
            total=self.tau_max + 1,
        )

        for tau in pbar:
            q_tab      = self.Q[tau]
            q_tab_next = self.Q[tau + 1]  # sentinel empty dict when tau == tau_max

            for jh, turn_id in self.reachable_by_tau[tau]:
                priv_h = self._mask_state(jh, turn_id)
                legal  = self.legal_actions_cache.get(priv_h, ())

                for a in legal:
                    self.env.reset(list(jh))
                    try:
                        self.env.step(a)
                    except ValueError:
                        q_tab[(jh, a)] = 0.0
                        continue

                    if self.env.is_terminal():
                        q_tab[(jh, a)] = self.env.payoff()
                    else:
                        jh_next    = tuple(self.env.history)
                        turn_next  = self._get_turn(jh_next)
                        priv_next  = self._mask_state(jh_next, turn_next)
                        a_next     = self.policy.get(priv_next, 0)
                        q_next     = q_tab_next.get((jh_next, a_next), 0.0)

                        # Agent 1 is the "last" agent — apply discount after env transition
                        if turn_id == 1:
                            q_tab[(jh, a)] = self.gamma * q_next
                        else:
                            q_tab[(jh, a)] = q_next

            pbar.set_postfix({"tau": tau, "entries": len(q_tab)})

    # ------------------------------------------------------------------
    # Training — one forward + one backward pass
    # ------------------------------------------------------------------

    def train(self) -> float:
        """
        One OSarsa iteration:
          1. Epsilon-greedy policy improvement (forward)
          2. Full backward Q-update (DP over all reachable states)

        Returns the number of policy changes (0.0 = converged).
        """
        self.current_iter += 1
        epsilon = max(
            self.epsilon_min,
            self.epsilon_start * (self.epsilon_decay ** self.current_iter),
        )

        changes = self._update_policy(epsilon)
        self._update_Q()

        return float(changes)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str):
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        data = {
            'policy':       dict(self.policy),
            'Q':            [dict(q) for q in self.Q],
            'current_iter': self.current_iter,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            return
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.policy.clear()
        self.policy.update(data.get('policy', {}))
        for tau, q in enumerate(data.get('Q', [])):
            if tau < len(self.Q):
                self.Q[tau] = q
        self.current_iter = data.get('current_iter', 0)

    # ------------------------------------------------------------------
    # Reset — reinitialise for a new random restart
    # ------------------------------------------------------------------

    def reset(self):
        """Reinitialise policy and Q tables for a fresh random restart."""
        self.Q          = [{} for _ in range(self.tau_max + 2)]
        self.current_iter = 0
        self._init_policy()
