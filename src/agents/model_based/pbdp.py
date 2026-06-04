from collections import defaultdict
import os
import pickle
import random
from tqdm import tqdm

from tiny_game import *

from ..base_agent import ModelBasedAgent, AgentList


class PBDP_Agent(ModelBasedAgent):
    """Thin executor whose policy is maintained by PBDP_Central_Planner."""

    def __init__(self, env: Game, num_cards: int, num_actions: int, agent_id: int, policy: dict):
        super().__init__(env, num_cards, num_actions)
        self.agent_id = agent_id
        self.policy = policy

    def act(self, input_state: tuple, exploit: bool = False) -> int:
        return self.policy[input_state]

    def train(self):                return 0.0
    def save_transition(self, *_):  pass
    def save(self, *_):             pass
    def load(self, *_):             pass
    def reset(self):                pass



class PBDP_Central_Planner(AgentList):
    """
    CTDE - Model based: 
    Point-Based Dynamic Programming for Dec-POMDPs (Szer & Charpillet, AAAI 2006, Fig. 2).
    """

    def __init__(
        self,
        env: Game,
        num_cards: int,
        num_actions: int,
        game_name : str,
        *args, **kwargs,
    ):
        self.env         = env
        self.game_name = game_name
        self.gamma       = 0.99
        self.num_cards   = num_cards
        self.num_actions = num_actions

        self.is_decpomdp = isinstance(env, DecPOMDP)
        self.is_myhanabi = isinstance(env, MyHanabi)

        self.policy : dict[tuple, int] = {}

        self._init_tables()

        agent_0 = PBDP_Agent(env, num_cards, num_actions, 0, self.policy)
        agent_1 = PBDP_Agent(env, num_cards, num_actions, 1, self.policy)
        super().__init__([agent_0, agent_1])

    @property
    def centralized_planning(self) -> bool:
        return True

    @property
    def _deal_len(self) -> int:
        return 2 if self.is_decpomdp else 4

    @property
    def _max_actions(self) -> int:
        return 2 if self.is_decpomdp else 8


    def _init_tables(self):
        pbar = tqdm(PRIV_HISTORIES[self.game_name], desc="PBDP Init Random Policy", leave=False)
        for priv_h, actions, done, turn_id, _ in pbar:
            if not done:
                self.policy[priv_h] = random.choice(list(actions))


    def train(self) -> float:
        # Backward pass over decision depths
        pbar = tqdm(range(self._max_actions - 1, -1, -1), desc="PBDP Backward", leave=False)
        for t in pbar:
            pbar.set_postfix({"t": t})
            H = self.get_stept_priv_histories(t)
            candidate_policies = {priv_h : a for priv_h, a, _, _, _ in H}
            beliefs            = self.generate_belief_states(H)
            best_policy        = self.evaluate_belief(beliefs, candidate_policies)
            self.update_policy(best_policy)
        return 0.0


    def get_stept_priv_histories(self, t: int) -> list[tuple]:
        h_length = self._deal_len + t

        step_ts = []
        for history_summary in PRIV_HISTORIES[self.game_name]:
            priv_h, _, done, _, _ = history_summary
            if not done and len(priv_h) == h_length and priv_h in CONSISTENT_WORLDS[self.game_name]:
                step_ts.append(history_summary)
        return step_ts


    def generate_belief_states(self, H: list[tuple]) -> dict[tuple, list[tuple]]:
        # Uniform belief over consistent joint worlds
        beliefs: dict[tuple, list[tuple]] = {}
        for priv_h, action, done, turn_id, reward in H:
            consistent_worlds = CONSISTENT_WORLDS[self.game_name][priv_h]
            beliefs[priv_h] = [jh for jh in consistent_worlds]
        return beliefs


    def evaluate_belief(
        self,
        beliefs: dict[tuple, list[tuple]],
        candidate_policies: dict[tuple, list[int]],
    ) -> dict[tuple, int]:
        best_policy: dict[tuple, int] = {}

        for priv_h, consistent_worlds in beliefs.items():
            legal = candidate_policies[priv_h]

            if not legal:
                continue
            n           = len(consistent_worlds)
            best_action = legal[0]  # Random start best_actions
            best_value  = -float('inf')

            for a in legal:
                total = sum(self._simulate_remaining(jh, a) for jh in consistent_worlds)
                avg_value = total / n
                if avg_value > best_value:
                    best_value  = avg_value
                    best_action = a

            best_policy[priv_h] = best_action
        return best_policy


    def _simulate_remaining(self, jh: tuple, first_action: int) -> float:
        self.env.reset(list(jh))
        self.env.step(first_action)

        while not self.env.is_terminal():
            curr_state = tuple(self.env.history)
            curr_priv  = mask_state(curr_state, len(curr_state) % 2, self.env)
            action = self.policy.get(curr_priv)
            self.env.step(action)
        reward = self.env.payoff()
        return reward


    def update_policy(self, best_policy: dict[tuple, int]) -> None:
        self.policy.update(best_policy)


    def reset(self):
        self.policy.clear()
        self._init_tables()


    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({'policy': dict(self.policy)}, f, protocol=pickle.HIGHEST_PROTOCOL)


    def load(self, filepath: str):
        if not os.path.exists(filepath):
            return
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.policy.clear()
        self.policy.update(data.get('policy', {}))
