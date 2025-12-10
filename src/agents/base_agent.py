from abc import ABC, abstractmethod
from typing import List, Any, Tuple, Iterator

from tiny_hanabi.agent.actors import Actor

class BaseAgent(Actor, ABC):
    def __init__(
            self,
            agent_id : int,
            num_actions : int,
            *args,
            **kwargs
    ):
        # Initialize Actor (if it has an __init__)
        super().__init__(*args, **kwargs) 
        self.agent_id: int = agent_id
        self.num_actions = num_actions
        # Buffer for the last transition (needed for both IQL and VDN)
        self.last_transition = None
        
    @abstractmethod
    def train(self):
        """
        Will potentially train the agent (if trainable and single agent training)
        """
        pass

    # We re-declare act here (or ensure Actor has it) so the IDE knows 
    # BaseAgent definitely has this method.
    @abstractmethod
    def act(self, observation: Any) -> int:
        pass
    @abstractmethod
    def reset():
        """
        """
        pass
    @abstractmethod
    def save_transition(
        self,
        observation,
        action, 
        next_observation,
        reward,
        done
    ):
        """
        """


class AgentList(List[BaseAgent]):
    """
    A list wrapper that holds multiple Actor objects.
    Inheriting from List[BaseAgent] tells the IDE that 'self' contains BaseAgents.
    """
    def __init__(self, agents: List[BaseAgent]):
        # 1. Runtime Enforcement: Validate types immediately
        for i, agent in enumerate(agents):
            if not isinstance(agent, BaseAgent):
                raise TypeError(
                    f"Agent at index {i} is not a BaseAgent instance. "
                    f"Got {type(agent).__name__}."
                )
        
        # Initialize the standard list
        super().__init__(agents)

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
        for agent in self:
            agent.train()

    # Optional: If you want to strictly prevent adding non-agents later
    def append(self, object: BaseAgent):
        if not isinstance(object, BaseAgent):
            raise TypeError(f"Cannot append {type(object).__name__}; must be BaseAgent")
        super().append(object)

    def reset(self):
        for agent in self:
            agent.reset()