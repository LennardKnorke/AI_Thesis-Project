# tiny_game/__init__.py

#A Supper Simple Wrapper for the tiny-hanabi github repo.
#Easier to import and to get an overview of relevant code
from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Optional, Tuple, List, Union

from tiny_hanabi.game.payoff_matrices import (
    GameNames,PAYOFFS,OPTIMAL_RETURNS
)

from tiny_hanabi.game.settings import (
    statetype, Settings, Game,
    DecPOMDP
)



from tiny_hanabi.game.assemblers import (
    get_game, normalize_payoffs
)


class DecPOMP_Rework(DecPOMDP):
    def __init__(self, payoffs: np.ndarray, optimal_return: float):
        super().__init__(payoffs, optimal_return)
        
        self.history_length : int = 4  # 2 cards + 2 actions
        
        self.NULL_VALUE : int = -1  # Padding for no previous action seen
        return


    def get_playerId(self):
        """
        Determines current player based on history length.
        (History contains 2 cards + N actions)
        """
        return 0 if len(self.history) == 2 or len(self.history) == 4 else 1
    
    def get_observation(
            self, 
            agent_id : Optional[int],
            as_tensor : bool = False
        ) -> Union[tuple, torch.Tensor]:
        """
        Format: [Card_P0, Card_P1, Action_P0, Action_P1]
        
        - Unknown/Future values are masked with NULL_VALUE (-1).
        - Private info (my own card) is masked with NULL_VALUE (-1).

        Args:
            agent_id (int): The ID of the agent observing.
            as_tensor (bool): If True, returns a torch.FloatTensor suitable for DRL.
                              If False, returns a tuple (suitable for tabular/hashing).
        """
        # Get full state/history
        current_history = [e for e in self.context()]

        # Mask agents card
        if agent_id == 0 or (len(current_history) == 2 or len(current_history) == 4):
            current_history[0] = self.NULL_VALUE
        elif agent_id == 1:
            current_history[1] = self.NULL_VALUE
        else:
            raise ValueError(f"Invalid Agent ID: {agent_id}")
        
        # (Optional return as a tensor)
        if as_tensor:
            padded = current_history + [self.NULL_VALUE] * (self.history_length - len(current_history))
            return torch.tensor(padded, dtype=torch.float32)
        
        return tuple(current_history)

        

    
    def get_all_observations(self, as_tensor : bool = False) -> List[Union[tuple, torch.Tensor]]:
        """
        Generates a comprehensive list of all valid observations that can occur 
        in the game for any agent at any decision point.
        
        Used by Model-Based Planners to initialize the policy table.
        """
        observations = []

        # All Player 1 observations (no actions seen)
        for c1 in range(self.num_cards):
            obs = (self.NULL_VALUE, c1)

        # All Player 2 observations (one action seen)
        for c0 in range(self.num_cards):
            for a0 in range(self.num_actions):
                obs = (c0, self.NULL_VALUE, a0)
                observations.append(obs)
            
        # (Optional) Return list of tensors
        if as_tensor:
            tensor_obs = []
            for obs in observations:
                padded = list(obs) + [self.NULL_VALUE] * (self.history_length - len(obs))
                tensor_obs.append(torch.tensor(padded, dtype=torch.float32))
            return tensor_obs
        
        return observations
    
    def is_terminal_observation(self, observation: Union[tuple, torch.Tensor]) -> bool:
        """
        Determines if the given observation corresponds to a terminal state
        for the specified agent.
        """
        if isinstance(observation, torch.Tensor):
            observation = tuple(observation.tolist())
        
        # Terminal if both actions have been taken
        return observation[2] != self.NULL_VALUE and observation[3] != self.NULL_VALUE

    def get_state(self, as_tensor : bool = False):
        if as_tensor:
             # State is usually [c0, c1, a0, a1] full info
             return torch.tensor(self.history, dtype=torch.float32)
        return tuple(self.history)
    
    def transition_fn(
            self,
            agent_id : int,
            history: tuple[int],
            action : int,
        ) -> List[int]:
        """
        Given the current history and an action by the specified agent,
        returns the new history after applying the action.
        """
        new_history = [*history]
        if agent_id == 0:
            new_history[2] = action  # Player 0's action
        else:
            new_history[3] = action  # Player 1's action
        return new_history
    
    def reward_fn(
            self,
            agent_id : int,
            history: tuple[int],
            action : int
        ) -> float:
        """
        Returns the reward obtained after the specified agent takes the action
        given the current history.
        """
        return self.payoffs[tuple(history)] if history[2] != self.NULL_VALUE and history[3] != self.NULL_VALUE else 0.0

    
    
def get_game_Rework(gamename: GameNames, setting: Settings, normalize: bool = True):
    base_game = get_game(gamename=gamename, setting=setting, normalize=normalize)
    matrix = np.copy(base_game.payoffs)
    optimal_returns = base_game.optimal_return
    game = DecPOMP_Rework(matrix, optimal_returns)
    return game

GAMES = ["A", "B", "C", "D", "E", "F"]

__all__ = [
    "GameNames", "PAYOFFS","OPTIMAL_RETURNS",
    "statetype", "Settings", "Game",
    #"DecPOMDP","get_game", 
    "normalize_payoffs",

    "GAMES", "DecPOMP_Rework", "get_game_Rework"
]