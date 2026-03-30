# agents/model_based/ctde_cibi.py
"""
Centralized Training with Decentralized Execution via Complete-Information-State
Backward Induction  (ι-MDP).

Unlike CTDE-BI, which optimises action-by-action over individual joint histories
and then extracts a private policy by averaging, this agent directly optimises
*joint decision rules*  d_t : H^i_t → A.

A decision rule is a complete mapping from every possible private observation at
step t to an action.  Optimising over decision rules rather than individual
actions explicitly encodes the decentralisation constraint: all joint histories
that share the same private observation MUST receive the same action.

At each step t (processed from last to first), we select the joint decision rule
that maximises the expected cumulative return summed over all joint histories at
that step, weighted uniformly (reflecting the uniform start-state distribution).

Tractability
------------
For small games (A–F):  the observation space at each step is tiny, so the full
Cartesian product of actions is enumerated.  The threshold is _MAX_DR_SPACE.

For large games (G, MyHanabi):  the obs space grows quickly and full enumeration
becomes infeasible.  The agent falls back to a greedy, observation-by-observation
optimisation.  This is equivalent to the private-policy extraction in CTDE-BI,
but it still uses the correct V-values propagated from subsequent steps.
"""

import random
import itertools
import os
import pickle

from ..base_agent import ModelBasedAgent, AgentList
from tiny_game import DecPOMDP, MyHanabi, Game, get_all_possible_histories


# Maximum joint-decision-rule space size before switching to greedy fallback.
_MAX_DR_SPACE = 100_000


class CTDE_CIBI_MB_Agent(ModelBasedAgent):
    """
    Thin wrapper around a shared policy dict.  Identical in role to
    CTDE_BI_MB_Agent — the planning is done entirely by CTDE_CIBI_MB_List.
    """

    def __init__(
        self,
        env: Game,
        num_cards: int,
        num_actions: int,
        agent_id: int,
        policy: dict,
    ):
        super().__init__(env, num_cards, num_actions)
        self.policy = policy
        self.agent_id = agent_id

    def act(self, input_state: tuple, exploit: bool = False) -> int:
        return self.policy[input_state]

    def train(self): return 0.0
    def save_transition(self, *args): pass
    def save(self, *args): pass
    def reset(self): pass


class CTDE_CIBI_MB_List(AgentList):
    """
    ι-MDP planner.

    Planning is a single backward-induction sweep over *joint decision rules*.
    At each timestep (identified by joint-history length), the planner:

      1. Groups all non-terminal joint histories at that step.
      2. Determines the unique private observations seen by the acting player.
      3. Enumerates all decision rules  d : {priv_obs} → A  (full enumeration
         when feasible; greedy fallback otherwise).
      4. Selects the decision rule maximising expected return averaged over all
         joint histories at that step (using V-values already computed for
         successor states).
      5. Updates V[joint_hist] and writes d into the shared policy dict.
    """

    def __init__(
        self,
        env: Game,
        num_cards: int,
        num_actions: int,
        *args,
        **kwargs,
    ):
        self.env = env
        self.gamma = 0.99
        self.num_cards = num_cards
        self.num_actions = num_actions

        self.is_decpomdp = isinstance(env, DecPOMDP)
        self.is_myhanabi = isinstance(env, MyHanabi)

        priv, joint = get_all_possible_histories(env)
        self.all_private_histories = sorted(priv, key=lambda x: len(x[0]), reverse=True)
        self.all_joint_histories = sorted(joint, key=lambda x: len(x[0]), reverse=True)

        # V-values keyed by joint-history tuple (same role as in CTDE_BI_MB_List)
        self.v_values: dict[tuple, float] = {}

        # Shared policy: private_obs → action
        self.policy: dict[tuple, int] = {}

        self._build_step_index()
        self._init_v_values()

        agent_0 = CTDE_CIBI_MB_Agent(env, num_cards, num_actions, 0, self.policy)
        agent_1 = CTDE_CIBI_MB_Agent(env, num_cards, num_actions, 1, self.policy)
        super().__init__([agent_0, agent_1])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def centralized_planning(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _mask_state(self, state: tuple, turn_id: int) -> tuple:
        """Convert a joint history to the private observation of the acting player."""
        s = list(state)
        if self.is_decpomdp:
            if turn_id == 0:
                s[0] = -1
            else:
                s[1] = -1
        else:  # MyHanabi
            if turn_id == 0:
                s[0] = -1
                s[1] = -1
            else:
                s[2] = -1
                s[3] = -1
        return tuple(s)

    def _build_step_index(self):
        """
        Group non-terminal joint histories by their tuple length (proxy for step).
        All entries at a given length share the same turn_id.
        """
        self._step_index: dict[int, list[tuple]] = {}
        for joint_hist, done, turn_id, _ in self.all_joint_histories:
            if done:
                continue
            n = len(joint_hist)
            if n not in self._step_index:
                self._step_index[n] = []
            self._step_index[n].append((joint_hist, turn_id))

    def _init_v_values(self):
        """Terminal states get their reward; non-terminal states start at 0."""
        for joint_hist, done, _, reward in self.all_joint_histories:
            self.v_values[joint_hist] = reward if done else 0.0

    # ------------------------------------------------------------------
    # Legal actions
    # ------------------------------------------------------------------

    def _get_legal_actions(self, priv_obs: tuple) -> tuple:
        if self.is_decpomdp:
            return tuple(range(self.num_actions))
        _, legal = self.env.num_legal_actions(priv_obs)
        return tuple(legal)

    # ------------------------------------------------------------------
    # Decision-rule evaluation
    # ------------------------------------------------------------------

    def _evaluate_dr(
        self,
        joint_hists: list,
        dr: dict,
        turn_id: int,
    ) -> float:
        """
        Average Q-value of decision rule *dr* over all joint histories at this step.
        """
        total = 0.0
        count = 0
        for joint_hist in joint_hists:
            priv_obs = self._mask_state(joint_hist, turn_id)
            action = dr[priv_obs]

            self.env.reset(list(joint_hist))
            try:
                self.env.step(action)
            except ValueError:
                continue
            count += 1

            if self.env.is_terminal():
                total += self.env.payoff()
            else:
                next_hist = tuple(self.env.history)
                total += self.gamma * self.v_values.get(next_hist, 0.0)

        return total / count if count > 0 else 0.0

    def _apply_dr(self, joint_hists: list, dr: dict, turn_id: int):
        """
        Update V[joint_hist] for all histories at this step using the chosen *dr*.
        """
        for joint_hist in joint_hists:
            priv_obs = self._mask_state(joint_hist, turn_id)
            action = dr[priv_obs]

            self.env.reset(list(joint_hist))
            try:
                self.env.step(action)
            except ValueError:
                # Keep previous value unchanged
                continue

            if self.env.is_terminal():
                self.v_values[joint_hist] = self.env.payoff()
            else:
                next_hist = tuple(self.env.history)
                self.v_values[joint_hist] = self.gamma * self.v_values.get(next_hist, 0.0)

    # ------------------------------------------------------------------
    # Optimisation strategies
    # ------------------------------------------------------------------

    def _optimise_full(
        self,
        joint_hists: list,
        obs_to_legal: dict,
        turn_id: int,
    ) -> dict:
        """
        Full enumeration of the joint decision-rule space.

        Considers every combination of (action per private obs) and returns the
        decision rule with the highest average Q-value across all joint histories.
        This is the exact ι-MDP solution for this timestep.
        """
        obs_list = list(obs_to_legal.keys())
        action_choices = [obs_to_legal[o] for o in obs_list]

        best_dr = {o: obs_to_legal[o][0] for o in obs_list}
        best_val = -float('inf')

        for combo in itertools.product(*action_choices):
            dr = dict(zip(obs_list, combo))
            val = self._evaluate_dr(joint_hists, dr, turn_id)
            if val > best_val:
                best_val = val
                best_dr = dr

        return best_dr

    def _optimise_greedy(
        self,
        joint_hists: list,
        obs_to_legal: dict,
        turn_id: int,
    ) -> dict:
        """
        Greedy, observation-by-observation fallback for large obs spaces.

        For each private observation independently, selects the action that
        maximises expected Q-value over that observation's consistent joint
        histories.  Random tie-breaking is applied.

        Note: This does not jointly optimise across observations, so it may
        miss improvements that require coordinated action choices.  It is
        equivalent to the private-policy extraction used in CTDE-BI, but
        applied directly to the already-propagated V-values.
        """
        # Group joint histories by private observation
        obs_to_hists: dict[tuple, list] = {}
        for jh in joint_hists:
            priv = self._mask_state(jh, turn_id)
            obs_to_hists.setdefault(priv, []).append(jh)

        dr = {}
        for priv_obs, legal in obs_to_legal.items():
            hists = obs_to_hists.get(priv_obs, [])
            best_val = -float('inf')
            best_acts = [legal[0]]

            for a in legal:
                total = 0.0
                count = 0
                for jh in hists:
                    self.env.reset(list(jh))
                    try:
                        self.env.step(a)
                    except ValueError:
                        continue
                    count += 1
                    if self.env.is_terminal():
                        total += self.env.payoff()
                    else:
                        nh = tuple(self.env.history)
                        total += self.gamma * self.v_values.get(nh, 0.0)

                avg = total / count if count > 0 else 0.0
                if avg > best_val:
                    best_val = avg
                    best_acts = [a]
                elif avg == best_val:
                    best_acts.append(a)

            dr[priv_obs] = random.choice(best_acts)

        return dr

    # ------------------------------------------------------------------
    # Main training / planning entry-point
    # ------------------------------------------------------------------

    def train(self) -> float:
        """
        Perform one complete backward-induction sweep over joint decision rules.

        Steps are processed from longest joint-history length to shortest
        (i.e. from the last decision point to the first).  At each step the
        planner selects the optimal joint decision rule and propagates the
        resulting values backward.

        Returns 0.0 — a single sweep is exact for finite-horizon games with
        no stochastic transitions, so the loss is immediately zero.
        """
        for length in sorted(self._step_index.keys(), reverse=True):
            entries = self._step_index[length]
            joint_hists = [jh for jh, _ in entries]
            turn_id = entries[0][1]  # consistent within a length

            # Build obs → legal-actions map for this step
            obs_to_legal: dict[tuple, tuple] = {}
            for jh in joint_hists:
                priv = self._mask_state(jh, turn_id)
                if priv not in obs_to_legal:
                    obs_to_legal[priv] = self._get_legal_actions(priv)

            # Compute decision-rule space size
            dr_space = 1
            for legal in obs_to_legal.values():
                dr_space *= len(legal)

            # Choose optimisation strategy
            if dr_space <= _MAX_DR_SPACE:
                best_dr = self._optimise_full(joint_hists, obs_to_legal, turn_id)
            else:
                best_dr = self._optimise_greedy(joint_hists, obs_to_legal, turn_id)

            # Propagate values and write into shared policy
            self._apply_dr(joint_hists, best_dr, turn_id)
            for priv_obs, action in best_dr.items():
                self.policy[priv_obs] = action

        return 0.0  # single-pass backward induction; converged

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "policy": dict(self.policy),
            "v_values": dict(self.v_values),
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            return
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.policy.clear()
        self.policy.update(data["policy"])
        self.v_values.clear()
        self.v_values.update(data["v_values"])

    def reset(self):
        # Re-initialise V-values for a fresh planning attempt.
        # Policy is not cleared — the engine will call train() immediately after.
        self._init_v_values()
        self.policy.clear()
