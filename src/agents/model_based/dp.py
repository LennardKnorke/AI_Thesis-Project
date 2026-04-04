# agents/model_based/dp.py
import os
import pickle
from collections import defaultdict
from tqdm import tqdm

from ..base_agent import ModelBasedAgent, AgentList
from tiny_game import DecPOMDP, MyHanabi, Game, get_all_possible_histories


class DP_Agent(ModelBasedAgent):
    """Thin executor whose policy is maintained by DP_List."""

    def __init__(
        self,
        env: Game,
        num_cards: int,
        num_actions: int,
        agent_id: int,
        policy: dict,
    ):
        super().__init__(env, num_cards, num_actions)
        self.agent_id = agent_id
        self.policy   = policy  # shared reference into DP_List.policy

    def act(self, input_state: tuple, exploit: bool = False) -> int:
        return self.policy[input_state]

    def train(self):                  return 0.0
    def save_transition(self, *args): pass
    def save(self, *args):            pass
    def reset(self):                  pass


class DP_List(AgentList):
    """
    Centralized MA-Belief-DP planner.

    Computes V(h) for every private observation h via backward Bellman backups
    in private-observation space.  The partner handoff is encoded directly:
    when agent i takes action a in world s, the continuation value is
    V(h'_j(s, a)) — agent j's private observation in the resulting world.
    """

    def __init__(
        self,
        env: Game,
        num_cards: int,
        num_actions: int,
        *args,
        **kwargs,
    ):
        self.env         = env
        self.gamma       = 0.99
        self.num_cards   = num_cards
        self.num_actions = num_actions
        self.is_decpomdp = isinstance(env, DecPOMDP)
        self.is_myhanabi = isinstance(env, MyHanabi)

        priv, joint = get_all_possible_histories(env)
        # Longest first so successor values are ready when we back up a node
        self.all_private_histories = sorted(priv,  key=lambda x: len(x[0]), reverse=True)
        self.all_joint_histories   = sorted(joint, key=lambda x: len(x[0]), reverse=True)

        self.policy:              dict[tuple, int]   = {}
        self.legal_actions_cache: dict[tuple, tuple] = {}
        # Scalar value function over private observations
        self.V:                   dict[tuple, float] = defaultdict(float)
        # Consistent joint histories per private observation
        self.consistent_worlds:   dict[tuple, list[tuple]] = {}
        self.joint_turn:          dict[tuple, int]   = {}

        self._init_structures()

        agent_0 = DP_Agent(env, num_cards, num_actions, 0, self.policy)
        agent_1 = DP_Agent(env, num_cards, num_actions, 1, self.policy)
        super().__init__([agent_0, agent_1])

    # ------------------------------------------------------------------
    @property
    def centralized_planning(self) -> bool:
        return True

    # ------------------------------------------------------------------
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

    def _init_structures(self):
        # Store turn ownership for every joint history
        pbar = tqdm(self.all_joint_histories, leave=False, desc="Init JH")
        for jh, _done, turn_id, _reward in pbar:
            self.joint_turn[jh] = turn_id

        # Build consistent-worlds lists and legal-action cache
        pbar = tqdm(self.all_private_histories, leave=False, desc="Init PH")
        for priv, done, turn_id, reward in pbar:
            if done:
                # Terminal obs: value is the expected payoff over consistent worlds.
                # (Used only as initial value; terminal transitions are handled
                #  inline via env.payoff() in the backup.)
                self.V[priv] = reward
                continue

            consistent = [
                jh
                for jh, jdone, jt, _r in self.all_joint_histories
                if not jdone and self._mask_state(jh, jt) == priv
            ]
            if not consistent:
                continue

            self.consistent_worlds[priv] = consistent

            if self.is_decpomdp:
                legal = tuple(range(self.num_actions))
            else:
                _, legal = self.env.num_legal_actions(consistent[0])
            self.legal_actions_cache[priv] = legal

    # ------------------------------------------------------------------
    def train(self) -> float:
        """
        One backward DP sweep over private observations (belief states).

        For each non-terminal private observation h (longest first):
          Q(h, a) = mean over consistent worlds of [R(s,a) or gamma*V(h'_partner)]
          V(h)    = max_a Q(h, a)
          pi(h)   = argmax_a Q(h, a)

        Returns the maximum absolute value change (convergence signal).
        """
        max_delta = 0.0

        pbar = tqdm(self.all_private_histories, leave=False, desc="Train Sweep")
        for priv, done, turn_id, _ in pbar:
            if done or priv not in self.consistent_worlds:
                continue

            worlds = self.consistent_worlds[priv]
            legal  = self.legal_actions_cache.get(priv, ())
            if not legal:
                continue

            partner_turn = 1 - turn_id   # agent who acts after the current one

            best_q      = -float('inf')
            best_action = legal[0]

            for a in legal:
                total = 0.0
                count = 0

                for jh in worlds:
                    self.env.reset(list(jh))
                    try:
                        self.env.step(a)
                    except ValueError:
                        continue
                    count += 1

                    if self.env.is_terminal():
                        total += self.env.payoff()
                    else:
                        next_jh        = tuple(self.env.history)
                        # Partner's private observation in the resulting world
                        next_priv      = self._mask_state(next_jh, partner_turn)
                        total         += self.gamma * self.V.get(next_priv, 0.0)

                if count > 0:
                    q = total / count
                    if q > best_q:
                        best_q      = q
                        best_action = a

            old_val         = self.V.get(priv, 0.0)
            new_val         = best_q if best_q > -float('inf') else old_val
            self.V[priv]    = new_val
            self.policy[priv] = best_action
            max_delta       = max(max_delta, abs(new_val - old_val))

        return max_delta

    # ------------------------------------------------------------------
    def save(self, filepath: str):
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        data = {'policy': dict(self.policy), 'V_values': dict(self.V)}
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            return
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.policy.clear()
        self.policy.update(data.get('policy', {}))
        self.V.clear()
        self.V.update(data.get('V_values', {}))

    def reset(self):
        self.V.clear()
        self.policy.clear()
        for priv, done, _turn_id, reward in self.all_private_histories:
            if done:
                self.V[priv] = reward
