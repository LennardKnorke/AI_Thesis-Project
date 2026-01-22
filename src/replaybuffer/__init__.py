# /replaybuffer/__init__.py

import random
from collections import deque, namedtuple
from typing import List, Tuple


# Define a simple Transition structure
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
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
        self.memory = deque(maxlen=int(capacity))
        self.current_episode: List[EpisodeStep] = []

    def push_step(self, state, action):
        """Records a single step in the temporary current episode buffer"""
        self.current_episode.append(EpisodeStep(state, action))

    def close_episode(self, final_reward: float):
        """
        Commits the current episode to memory associated with the final reward.
        """
        if len(self.current_episode) > 0:
            # Store tuple: (List[Steps], Final_Reward)
            self.memory.append((list(self.current_episode), final_reward))
            self.current_episode = []

    def sample(self, batch_size: int) -> List[Tuple[List[EpisodeStep], float]]:
        """
        Returns a list of tuples: (Trajectory_List, Final_Reward)
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

__all__ = ['ReplayBuffer', 'Transition', 'EpisodicReplayBuffer', 'EpisodeStep']