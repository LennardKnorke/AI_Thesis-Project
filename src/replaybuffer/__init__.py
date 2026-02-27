# /replaybuffer/__init__.py

import random
from collections import deque, namedtuple
from typing import Any


# Define a simple Transition structure
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque(maxlen=int(capacity))

    def push(self, state, action, reward, next_state, done):
        """Save a transition"""
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[Transition]:
        if len(self.memory) < batch_size:
            return list(self.memory)
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# --- 2. Episodic Structures (For VDN / Trajectory-based RL) ---
EpisodeStep = namedtuple('EpisodeStep', ('state', 'action'))

class EpisodicReplayBuffer:
    """
    Episodic Buffer: Stores full trajectories [step1, step2, ...] + Final Reward.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque(maxlen=int(capacity))
        self.current_episode: list[EpisodeStep] = []

    def push_step(self, state: Any, action: Any):
        """Records a single step in the temporary current episode buffer"""
        self.current_episode.append(EpisodeStep(state, action))

    def close_episode(self, final_reward: float):
        """
        Commits the current episode to memory associated with the final reward.
        """
        if self.current_episode:
            self.memory.append((list(self.current_episode), final_reward))
            self.current_episode.clear()

    def sample(self, batch_size: int) -> list[tuple[list[EpisodeStep], float]]:
        """
        Returns a list of tuples: (Trajectory_List, Final_Reward)
        """
        if len(self.memory) < batch_size:
            return list(self.memory)
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def clear_current_episode(self):
        """Clears the steps of the current episode without adding it to the buffer."""
        self.current_episode.clear()

__all__ = ['ReplayBuffer', 'Transition', 'EpisodicReplayBuffer', 'EpisodeStep']