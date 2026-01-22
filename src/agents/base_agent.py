# /agents/base_agent.py

from abc import ABC, abstractmethod
from typing import List, Any, Tuple, Iterator, Optional

import numpy as np


class BaseAgent:
    MODEL_BASED : bool
    def __init__(
            self,
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
        return

    @abstractmethod
    def train(self):
        """
        Will Train or Plan for the agent based on its internal model or data.
        """
        pass

    @abstractmethod
    def act(self, input_state: Any) -> int:
        pass

    @abstractmethod
    def save_transition(self):
        """
        Save any transitions for learning.
        """
        return
    
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
    

class ModelBasedAgent(BaseAgent, ABC):
    MODEL_BASED = True
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

    


class AgentList(List[BaseAgent]):
    """
    A list wrapper that holds multiple Actor objects.
    Inheriting from List[BaseAgent] tells the IDE that 'self' contains BaseAgents.
    """
    def __init__(self, agents: Optional[List[BaseAgent]], *args, **kwargs):

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


    def act(self, observations: List[Any]) -> List[int]:
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