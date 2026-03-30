# /agents/base_agent.py

from abc import ABC, abstractmethod
from typing import Any, Iterator

import numpy as np

from tiny_game import Game, DecPOMDP, MyHanabi, get_all_possible_histories


class BaseAgent:
    MODEL_BASED : bool
    def __init__(
            self,
            env : Game,
            num_cards : int,
            num_actions : int,
            *args,
            **kwargs
    ):
        """
        Base class for all agents.
        """
        self.num_actions: int = num_actions
        self.num_cards: int = num_cards
        self.env : Game = env
        return

    @abstractmethod
    def act(self, input_state: Any, *args, **kwargs) -> int:
        pass
    
    @abstractmethod
    def save(self, filepath: str, *args, **kwargs):
        """
        Save the agent's model to disk.
        """
        pass

    @abstractmethod
    def load(self, filepath : str, *args, **kwargs):
        """
        Load Agent model
        """
        pass

    def reset(self):
        pass


    @abstractmethod
    def save_transition(self):
        """
        Save any transitions for learning.
        """
        return
    @abstractmethod
    def train(self):
        """
        Will Train or Plan for the agent based on its internal model or data.
        """
        pass
    

class ModelBasedAgent(BaseAgent, ABC):
    MODEL_BASED = True
    def __init__(self, env : Game, num_card : int, num_actions:int):
        super().__init__(env, num_card, num_actions)
        self.is_decpomdp = isinstance(self.env, DecPOMDP)
        self.is_myhanabi = isinstance(self.env, MyHanabi)

        self.max_hist_length : int = self.env.horizon
        self.min_hist_length : int = 2 if self.is_decpomdp else 4

        all_private_histories, all_joint_histories = get_all_possible_histories(self.env)
        self.all_private_histories = sorted(all_private_histories, key=lambda x: len(x[0]), reverse=True)
        self.all_joint_histories = sorted(all_joint_histories, key=lambda x: len(x[0]), reverse=True)
    


    @abstractmethod
    def train(self):
        """
        Model-based agents may implement planning here.
        Value Iteration over the learned models.
        """
        pass


class ModelFreeAgent(BaseAgent, ABC):
    MODEL_BASED = False
    @abstractmethod
    def train(self):
        """
        Model-free agents may implement learning here.
        E.g., updating Q-values or policy networks.
        """
        pass


class AgentList(list[BaseAgent]):
    """
    A list wrapper that holds multiple Actor objects.
    Inheriting from List[BaseAgent] tells the IDE that 'self' contains BaseAgents.
    """
    def __init__(self, agents: None|list[BaseAgent], *args, **kwargs):

        # 1. Runtime Enforcement: Validate types immediately
        for i, agent in enumerate(agents):
            if not isinstance(agent, BaseAgent):
                raise TypeError(
                    f"Agent at index {i} is not a BaseAgent instance. "
                    f"Got {type(agent).__name__}."
                )
        
        # Initialize the standard list
        super().__init__(agents)
    
    @property
    def centralized_planning(self) -> bool:
        return False


    def act(self, observations: list[Any]) -> list[int]:
        """
        Queries all agents for their actions.
        """
        if len(observations) != len(self):
            raise ValueError(
                f"Mismatch: {len(self)} agents but {len(observations)} observations."
            )

        joint_action = []
        for i, agent in enumerate(self):
            action = agent.act(observations[i])
            joint_action.append(action)
        return joint_action

    def train(self):
        # IBR: if agents expose set_partner_policy, exchange policies before
        # each sweep so each agent plans against the other's current policy.
        if len(self) == 2 and hasattr(self[0], 'set_partner_policy'):
            self[0].set_partner_policy(dict(self[1].policy))
            self[1].set_partner_policy(dict(self[0].policy))
        losses = []
        for agent in self:
            losses.append(agent.train())
        return np.mean(losses)

    # Optional: If you want to strictly prevent adding non-agents later
    def append(self, object: BaseAgent):
        if not isinstance(object, BaseAgent):
            raise TypeError(f"Cannot append {type(object).__name__}; must be BaseAgent")
        super().append(object)

    def reset(self):
        for agent in self:
            agent.reset()