# agents/model_based/jesp.py
import os
import pickle
import random
from collections import defaultdict
from tqdm import tqdm

from ..base_agent import ModelBasedAgent, AgentList
from tiny_game import DecPOMDP, MyHanabi, Game, get_all_possible_histories


class JESP_Agent(ModelBasedAgent):
    """Thin executor whose policy is maintained by JESP_List."""

    def __init__(self, env: Game, num_cards: int, num_actions: int, agent_id: int, policy: dict):
        super().__init__(env, num_cards, num_actions)
        self.agent_id = agent_id
        self.policy = policy  # shared reference into JESP_List.policy

    def act(self, input_state: tuple, exploit: bool = False) -> int:  # noqa: ARG002
        return self.policy.get(input_state, 0)

    def train(self):                return 0.0
    def save_transition(self, *_):  pass
    def save(self, *_):             pass
    def load(self, *_):             pass
    def reset(self):                pass


class JESP_List(AgentList):
    """
    JESP: Joint Equilibrium-based Search for Policies.

    Based on Nair, Tambe, Yokoo, Pynadath & Marsella (IJCAI 2003).

    Algorithm (Iterative Best Response):
        1. Initialise both agents with random policies.
        2. Fix agent 1's policy → compute agent 0's best response via
           backward induction over agent 0's private observations.
        3. Fix agent 0's new policy → compute agent 1's best response.
        4. Repeat until neither policy changes (Nash equilibrium).

    Each agent's subproblem given a fixed partner is a single-agent POMDP
    solved exactly via backward induction (longest private observation first).
    The partner's actions are read deterministically from the fixed policy.

    Convergence is to a Nash equilibrium — not necessarily the global optimum.
    Multiple random restarts (engine 'attempts' parameter) help escape local optima.

    train():
        One full IBR round. Returns total number of policy entries that changed.
        Returns 0.0 when converged (engine stops).
    """

    def __init__(
        self,
        env: Game,
        num_cards: int,
        num_actions: int,
        gamma: float = 0.99,
        *_args, **_kwargs,
    ):
        self.env          = env
        self.gamma        = gamma
        self.num_cards    = num_cards
        self.num_actions  = num_actions
        self.is_decpomdp  = isinstance(env, DecPOMDP)
        self.is_myhanabi  = isinstance(env, MyHanabi)

        priv, joint = get_all_possible_histories(env)
        self.all_private_histories = sorted(priv,  key=lambda x: len(x[0]), reverse=True)
        self.all_joint_histories   = sorted(joint, key=lambda x: len(x[0]), reverse=True)

        # Combined policy dict: keys for both agents are non-overlapping
        # (agent 0's private obs have -1 in their own card position, agent 1's in theirs)
        self.policy:              dict[tuple, int]         = {}
        self.legal_actions_cache: dict[tuple, tuple]       = {}
        self.consistent_worlds:   dict[tuple, list[tuple]] = {}
        self.joint_turn:          dict[tuple, int]         = {}

        self._init_game_structure()

        agent_0 = JESP_Agent(env, num_cards, num_actions, 0, self.policy)
        agent_1 = JESP_Agent(env, num_cards, num_actions, 1, self.policy)
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
                s[0] = s[1] = -1
            else:
                s[2] = s[3] = -1
        return tuple(s)

    def _deal_length(self) -> int:
        return 2 if self.is_decpomdp else 4

    # ------------------------------------------------------------------
    # Initialization — called once; game structure does not change
    # ------------------------------------------------------------------

    def _init_game_structure(self):
        """
        Build game-structure caches (turn ownership, consistent worlds, legal actions).
        Uses an inverse index to avoid the O(|priv| × |joint|) quadratic scan.
        Called once in __init__; not repeated on reset.
        """
        # 1. Turn ownership
        for jh, done, turn_id, _ in self.all_joint_histories:
            self.joint_turn[jh] = turn_id

        # 2. Inverse index: masked joint history -> list of consistent joint histories
        #    This reduces consistent-world lookup from O(|priv| × |joint|) to O(|joint|).
        masked_to_joints: dict[tuple, list[tuple]] = defaultdict(list)
        for jh, done, turn_id, _ in self.all_joint_histories:
            if done:
                continue
            masked_to_joints[self._mask_state(jh, turn_id)].append(jh)

        # 3. Consistent worlds + legal actions per private observation
        for priv_h, done, _, _ in self.all_private_histories:
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
            self.policy[priv_h] = random.choice(legal)
            

    # ------------------------------------------------------------------
    # Best-response computation
    # ------------------------------------------------------------------

    def _best_response(self, agent_id: int) -> int:
        """
        Backward induction for agent_id given the other agent's current policy.

        For each of agent_id's private observations h (longest first):
          For each legal action a:
            For each consistent world s:
              - Step agent_id's action a
              - While it is the partner's turn: apply partner's policy
              - If terminal: add payoff
              - Else: add gamma * V[next private obs of agent_id]
          policy[h] = argmax_a avg_value(a)

        Returns the count of policy entries that changed.
        """
        v_values: dict[tuple, float] = {}
        policy_changes = 0

        pbar = tqdm(self.all_private_histories, desc=f"JESP BR agent {agent_id}", leave=False)
        for priv_h, done, turn_id, _ in pbar:
            if done or turn_id != agent_id:
                continue
            if priv_h not in self.consistent_worlds:
                continue

            worlds = self.consistent_worlds[priv_h]
            legal  = self.legal_actions_cache.get(priv_h, ())
            if not legal:
                continue

            belief_weight = 1.0 / len(worlds)
            best_val = -float('inf')
            best_a   = legal[0]

            for a_i in legal:
                total = 0.0

                for s in worlds:
                    self.env.reset(list(s))
                    try:
                        self.env.step(a_i)
                    except ValueError:
                        continue

                    if self.env.is_terminal():
                        total += self.env.payoff()
                        continue

                    # Apply partner's policy for all consecutive partner turns
                    while not self.env.is_terminal():
                        curr_state = tuple(self.env.history)
                        curr_turn  = self.joint_turn.get(curr_state)
                        if curr_turn is None:
                            dl = self._deal_length()
                            curr_turn = (len(curr_state) - dl) % 2
                        if curr_turn == agent_id:
                            break  # back to our turn

                        h_partner = self._mask_state(curr_state, curr_turn)
                        a_partner = self.policy.get(h_partner)
                        if a_partner is None:
                            if self.is_decpomdp:
                                a_partner = random.randrange(self.num_actions)
                            else:
                                _, lp = self.env.num_legal_actions()
                                a_partner = random.choice(lp)
                        try:
                            self.env.step(a_partner)
                        except ValueError:
                            break

                    if self.env.is_terminal():
                        total += self.env.payoff()
                    else:
                        h_next = self._mask_state(tuple(self.env.history), agent_id)
                        total += self.gamma * v_values.get(h_next, 0.0)

                avg = total * belief_weight
                if avg > best_val:
                    best_val = avg
                    best_a   = a_i

            v_values[priv_h] = best_val if best_val > -float('inf') else 0.0

            old_a = self.policy.get(priv_h)
            self.policy[priv_h] = best_a
            if old_a != best_a:
                policy_changes += 1

            pbar.set_postfix({"changes": policy_changes, "val": f"{best_val:.4f}"})

        return policy_changes

    # ------------------------------------------------------------------
    # Training — one full IBR round
    # ------------------------------------------------------------------

    def train(self) -> float:
        """
        One full IBR round: agent 0 best response, then agent 1 best response.
        Returns 0.0 when neither policy changed (Nash equilibrium reached).
        The engine calls train() repeatedly until convergence or max_iterations.
        """
        changes_0 = self._best_response(agent_id=0)
        changes_1 = self._best_response(agent_id=1)
        return float(changes_0 + changes_1)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str):
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({'policy': dict(self.policy)}, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            return
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.policy.clear()
        self.policy.update(data.get('policy', {}))

    # ------------------------------------------------------------------
    # Reset — reinitialize policy for a new random restart
    # ------------------------------------------------------------------

    def reset(self):
        """Re-randomise policy only; game-structure caches are unchanged."""
        self.policy.clear()
        for priv_h, done, _, _ in self.all_private_histories:
            if done or priv_h not in self.legal_actions_cache:
                continue
            self.policy[priv_h] = random.choice(self.legal_actions_cache[priv_h])
