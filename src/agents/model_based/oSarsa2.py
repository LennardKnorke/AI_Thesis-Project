import os
import pickle
import sys
from pathlib import Path
import random
import pandas as pd


from tiny_game import *
from ..base_agent import ModelBasedAgent, AgentList

class OSarsa_Agent(ModelBasedAgent):
    """
    CTDE - ModelBased Occupancy MDP
    Pre-trained agent through c++ program
    """
    MODEL_BASED = True

    def __init__(
        self,
        env:         Game,
        num_cards:   int,
        num_actions: int,
        agent_id:    int,
        policy:      dict | None = None,
    ):
        super().__init__(env, num_cards, num_actions)
        self.agent_id:              int = agent_id

        self.policy:                dict = policy if policy is not None else {}
        self.legal_actions_cache:   dict = {}
        self.best_value:            float | None = None
        return

    
    def update_policy(self, new_policy : dict[tuple, int]):
        self.policy.update(new_policy)
        return


    def act(self, input_state: tuple, exploit: bool = False) -> int:
        action = self.policy.get(input_state)
        legal = self.legal_actions_cache.get(input_state)
        if legal is None or len(legal) == 0:
            return action if action is not None else 0
        if action is None or action not in legal:
            return random.choice(legal)
        return action


    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Policy file not found: {filepath}")
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.policy.clear()
        self.policy.update(data.get("policy", {}))
        self.best_value = data.get("best_value")

    def save(self, filepath: str):
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(
                {"policy": dict(self.policy), "best_value": self.best_value},
                f, protocol=pickle.HIGHEST_PROTOCOL,
            )

    def train(self) -> float:       return 0.0
    def save_transition(self, *_):  pass
    def reset(self):                pass



class OSarsa_Planner(AgentList):
    def __init__(self, env: Game, game_name:str, num_cards: int, num_actions: int):
        self.env         = env
        self.game_name   = game_name
        self.num_cards   = num_cards
        self.num_actions = num_actions
        self.best_value: float | None = None

        # Shared policy dict — both agents read from the same object
        self.policy: dict[tuple, int] = {}
        self._legal_cache = {}
        for priv_h, legal_actions, done ,_, _ in PRIV_HISTORIES[self.game_name]:
            if not done:
                self._legal_cache[priv_h] = legal_actions
                self.policy[priv_h] = random.choice(legal_actions)
        

        agent_0 = OSarsa_Agent(env, num_cards, num_actions, agent_id=0,
                                  policy=self.policy)
        agent_1 = OSarsa_Agent(env, num_cards, num_actions, agent_id=1,
                                  policy=self.policy)
        agent_0.legal_actions_cache = self._legal_cache
        agent_1.legal_actions_cache = self._legal_cache
        super().__init__([agent_0, agent_1])

    @property
    def centralized_planning(self) -> bool:
        return True


    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Policy file not found: {filepath}")
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.policy.clear()
        self.policy.update(data.get("policy", {}))
        self.best_value = data.get("best_value")
        for agent in self:
            agent.best_value = self.best_value


    @classmethod
    def from_pickle(cls, env: Game, num_cards: int, num_actions: int,
                    filepath: str) -> "OSarsa_Planner":
        # Create a planner and immediately load a policy from a pickle file
        planner = cls(env, num_cards, num_actions)
        planner.load(filepath)
        return planner


    def save(self, filepath: str):
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(
                {"policy": dict(self.policy), "best_value": self.best_value},
                f, protocol=pickle.HIGHEST_PROTOCOL,
            )

    def train(self) -> float:  return 0.0
    def reset(self):           pass