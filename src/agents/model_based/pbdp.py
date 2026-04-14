# agents/model_based/pbdp.py
from collections import defaultdict
import os
import pickle
import random
from tqdm import tqdm

from tiny_game import DecPOMDP, MyHanabi, Game, get_all_possible_histories

from ..base_agent import ModelBasedAgent, AgentList


class PBDP_Agent(ModelBasedAgent):
    """Thin executor whose policy is maintained by PBDP_Central_Planner."""

    def __init__(self, env: Game, num_cards: int, num_actions: int, agent_id: int, policy: dict):
        super().__init__(env, num_cards, num_actions)
        self.agent_id = agent_id
        self.policy = policy  # shared reference into PBDP_Central_Planner.policy

    def act(self, input_state: tuple, exploit: bool = False) -> int:
        return self.policy[input_state]

    def train(self):                return 0.0
    def save_transition(self, *_):  pass
    def save(self, *_):             pass
    def load(self, *_):             pass
    def reset(self):                pass



class PBDP_Central_Planner(AgentList):
    """
    Approximate Point-Based Dynamic Programming for Dec-POMDPs.
    Szer & Charpillet (AAAI 2006), Figure 2.
    """
    def __init__(
        self,
        env: Game,
        num_cards: int,
        num_actions: int,
        *args, **kwargs,
    ):
        self.env         = env
        self.gamma       = 0.99
        self.num_cards   = num_cards
        self.num_actions = num_actions
        self.is_decpomdp = isinstance(env, DecPOMDP)
        self.is_myhanabi = isinstance(env, MyHanabi)

        priv, joint = get_all_possible_histories(env)
        self.all_private_histories = sorted(priv,  key=lambda x: len(x[0]), reverse=True)
        self.all_joint_histories   = sorted(joint, key=lambda x: len(x[0]), reverse=True)

        self.policy:              dict[tuple, int]         = {}
        self.legal_actions_cache: dict[tuple, list[int]]   = {}
        self.consistent_worlds:   dict[tuple, list[tuple]] = {}

        self._init_tables()

        agent_0 = PBDP_Agent(env, num_cards, num_actions, 0, self.policy)
        agent_1 = PBDP_Agent(env, num_cards, num_actions, 1, self.policy)
        super().__init__([agent_0, agent_1])

    # ------------------------------------------------------------------
    @property
    def centralized_planning(self) -> bool:
        return True

    @property
    def _deal_len(self) -> int:
        """Number of elements in the initial deal portion of the history."""
        return 2 if self.is_decpomdp else 4

    @property
    def _max_actions(self) -> int:
        """Maximum total actions in a game (both agents combined)."""
        return 2 if self.is_decpomdp else 8


    def _init_tables(self):
        """
        Build three caches:
          consistent_worlds[h_i]    = joint histories compatible with private history h_i
          legal_actions_cache[h_i]  = legal actions at h_i
          policy[h_i]               = initial random action (re-randomised on reset)
        """
        # Inverse index: masked joint history → matching joint histories
        masked_to_joints: dict[tuple, list[tuple]] = defaultdict(list)
        for jh, done, turn_id, _ in self.all_joint_histories:
            if not done:
                masked_to_joints[self._mask_state(jh, turn_id)].append(jh)


        pbar = tqdm(self.all_private_histories, desc="PBDP Init", leave=False)
        for priv_h, done, turn_id, _ in pbar:
            consistent = masked_to_joints.get(priv_h, [])
            if not consistent:
                continue
            # Remember consisten world
            self.consistent_worlds[priv_h] = consistent

            # Get available actions
            if done:
                actions = ()
            else:
                if self.is_decpomdp:
                    actions = list(range(self.num_actions))
                else:
                    _, actions = self.env.num_legal_actions(consistent[0])
                # Random Default action
                self.policy[priv_h] = random.choice(list(actions))      
            self.legal_actions_cache[priv_h] = list(actions)        # Actions available in priv_h
        return


    def train(self) -> float:
        """
        One full backward PBDP pass (Szer & Charpillet 2006, Figure 2).

        Iterates from the deepest non-terminal step (t = max_actions-1)
        back to the initial step (t = 0).  At each step t, processes all
        private histories of length deal_len + t.

        Returns 0.0 (one-shot algorithm; engine checks reward for termination).
        """
        pbar = tqdm(range(self._max_actions - 1, -1, -1),
                    desc="PBDP Backward", leave=False)

        for t in pbar:
            pbar.set_postfix({"t": t})

            # [Fig. 2, Step 2.a] — get all private histories at depth t
            H = self.get_stept_priv_histories(t)
            if not H:
                continue

            # [Fig. 2, Step 2.a.i] — Exhaustive backup: candidate actions per history
            candidate_policies = self.generate_step_t_policies(H)

            # [Fig. 3] — Belief set: consistent joint worlds per private history
            beliefs = self.generate_belief_states(H)

            # [Fig. 2, Step 2.a.iii] — Point-based argmax over each belief set
            best_policy = self.evaluate_belief(beliefs, candidate_policies)

            # [Fig. 2, Step 2.b] — Commit best actions to policy
            self.update_policy(best_policy)

        return 0.0


    # Sub-routines (algorithm steps)
    def get_stept_priv_histories(self, t: int) -> list[tuple]:
        """
        Returns all non-terminal private histories at total action depth t.

        History length at depth t = deal_len + t:
          DecPOMDP  t=0 → len 2 (initial deal, agent 0 to act)
                    t=1 → len 3 (after agent 0 acted, agent 1 to act)
          MyHanabi  t=0 → len 4
                    t=7 → len 11 (deepest non-terminal)
        """
        h_length = self._deal_len + t
        H = []
        for priv_h, done, _, _ in self.all_private_histories:
            if not done and len(priv_h) == h_length and priv_h in self.consistent_worlds:
                H.append(priv_h)
        return H


    def generate_step_t_policies(self, H: list[tuple]) -> dict[tuple, list[int]]:
        step_t_policies = {}
        for priv_h in H:
            if priv_h in self.legal_actions_cache:
                step_t_policies[priv_h] = self.legal_actions_cache[priv_h]
        return step_t_policies


    def generate_belief_states(self, H: list[tuple]) -> dict[tuple, list[tuple]]:
        """
        [Fig. 3] Belief set generation.

        For each private history h_i the multi-agent belief b_i ∈ Δ(S × Q_{-i})
        is uniform over the consistent joint histories (worlds).

        For each consistent joint history jh:
          - State s = jh  (full joint history = world state in our deterministic game)
          - Partner's sub-tree q_{-i} is fixed by mask(jh, partner_turn),
            which indexes into self.policy (already set for deeper histories).

        In the deterministic sequential game no stochastic Bayes update is needed;
        the belief is simply the uniform distribution over consistent_worlds(h_i).

        Returns: {priv_h → [(joint_history, partner_priv_history), ...]}
        """
        beliefs: dict[tuple, list[tuple]] = {}
        for priv_h in H:
            if priv_h not in self.consistent_worlds:
                continue
            turn_id      = len(priv_h) % 2   # 0 → agent 0's turn, 1 → agent 1's turn
            partner_turn = 1 - turn_id

            beliefs[priv_h] = [
                (jh, self._mask_state(jh, partner_turn))
                for jh in self.consistent_worlds[priv_h]
            ]
        return beliefs


    def evaluate_belief(
        self,
        beliefs: dict[tuple, list[tuple]],
        candidate_policies: dict[tuple, list[int]],
    ) -> dict[tuple, int]:
        """
        [Fig. 2, Step 2.a.iii] Point-based argmax.

        For each private history h_i with belief set B(h_i):
          V(b_i, a) = (1/|B|) Σ_{jh ∈ B} V_sim(jh, a)
          best_a    = argmax_{a ∈ Q̄_i^t} V(b_i, a)

        V_sim(jh, a): simulate the complete remaining game from joint world jh,
        where the current agent plays a and all subsequent decisions follow
        self.policy (set for deeper histories in earlier backward steps).

        Returns: {priv_h → best_action}
        """
        best_policy: dict[tuple, int] = {}

        for priv_h, world_pairs in beliefs.items():
            legal = candidate_policies.get(priv_h, [])
            if not legal:
                continue

            n           = len(world_pairs)
            best_action = legal[0]
            best_value  = -float('inf')

            for a in legal:
                # V(b_i, a) — average simulated value over consistent worlds
                total = sum(
                    self._simulate_remaining(jh, a)
                    for jh, _ in world_pairs
                )
                avg_value = total / n

                if avg_value > best_value:
                    best_value  = avg_value
                    best_action = a

            best_policy[priv_h] = best_action

        return best_policy


    def _simulate_remaining(self, jh: tuple, first_action: int) -> float:
        """
        Simulate the full game from joint history jh:
          1. Current agent takes first_action.
          2. All subsequent decisions follow self.policy (backward-induction
             values for deeper histories are already set in self.policy).
        Returns the terminal payoff.
        """
        self.env.reset(list(jh))

        try:
            self.env.step(first_action)
        except ValueError:
            return 0.0

        if self.env.is_terminal():
            return self.env.payoff()

        while not self.env.is_terminal():
            curr_state = tuple(self.env.history)
            curr_turn  = len(curr_state) % 2
            curr_priv  = self._mask_state(curr_state, curr_turn)

            action = self.policy.get(curr_priv)
            if action is None:
                break
            try:
                self.env.step(action)
            except ValueError:
                break

        return self.env.payoff() if self.env.is_terminal() else 0.0
    
    
    def _mask_state(self, state: tuple, turn_id: int) -> tuple:
        """Return private observation: mask the acting agent's own cards."""
        s = list(state)
        if self.is_decpomdp:
            s[0 if turn_id == 0 else 1] = -1
        else:
            if turn_id == 0:
                s[0] = s[1] = -1
            else:
                s[2] = s[3] = -1
        return tuple(s)


    def update_policy(self, best_policy: dict[tuple, int]) -> None:
        self.policy.update(best_policy)
        return
    

    def reset(self):
        """Re-randomise policy for a new attempt; game-structure caches are unchanged."""
        self.policy.clear()
        for priv_h, done, _, _ in self.all_private_histories:
            if done or priv_h not in self.legal_actions_cache:
                continue
            self.policy[priv_h] = random.choice(self.legal_actions_cache[priv_h])
        return


    def save(self, filepath: str):
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({'policy': dict(self.policy)}, f, protocol=pickle.HIGHEST_PROTOCOL)
        return


    def load(self, filepath: str):
        if not os.path.exists(filepath):
            return
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.policy.clear()
        self.policy.update(data.get('policy', {}))
        return