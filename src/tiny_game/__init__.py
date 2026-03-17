# tiny_game/__init__.py

#A Supper Simple Wrapper for the tiny-hanabi github repo.
#Easier to import and to get an overview of relevant code

from abc import ABC, abstractmethod
from collections import deque
import numpy as np
import torch

# Tiny Hanabi
from tiny_hanabi.game.payoff_matrices import GameNames, PAYOFFS, OPTIMAL_RETURNS
from tiny_hanabi.game.settings import statetype, Settings, Game, DecPOMDP
from tiny_hanabi.game.assemblers import get_game, normalize_payoffs

from .my_hanabi import MyHanabi
    
    
def get_game_Rework(gamename: GameNames, normalize: bool = True):
    if gamename.value == "G":
        game = MyHanabi(normalize=normalize)
    else:
        game = get_game(gamename=gamename, setting=Settings.decpomdp, normalize=normalize)
    return game


def get_all_possible_states(env : Game)->tuple:
    all_states = []
    
    # 1. Setup Initial Queue with Start States
    # Queue stores: tuple(state)
    processing_queue = deque(env.start_states())

    max_len = env.horizon

    while processing_queue:
        current_state = processing_queue.popleft()

        env.reset(list(current_state))
        reward = env.payoff()
        turn_id = 0 if len(current_state) % 2 == 0 else 1
        total_ = (tuple(current_state), env.is_terminal(), turn_id, reward)

        all_states.append(total_)

        # Stop expanding if we reached max depth
        if len(current_state) >= max_len:
            continue

        valid_next_actions = []
        if isinstance(env, DecPOMDP):
            # TinyHanabi: All actions (0, 1) are always valid
            valid_next_actions = range(env.num_actions)
        elif isinstance(env, MyHanabi):
            _, valid_next_actions = env.num_legal_actions()
        else:
            raise ValueError("Unknown Environment Type")

        for action in valid_next_actions:
            env.reset(list(current_state))
            env.step(action)
            new_state = env.context()
            processing_queue.append(tuple(new_state))

    return tuple(all_states)


def get_all_possible_histories(env : Game):
    all_states = get_all_possible_states(env=env)
    
    # Use a set immediately to handle deduplication automatically
    observations = []
    unique_observations = set()

    for state, done, turn_id, reward in all_states:
        # 1. Determine whose turn it is
        if isinstance(env, DecPOMDP):
            turn_idx = len(state) - 2
        elif isinstance(env, MyHanabi):
            turn_idx = len(state) - 4
        else:
            raise ValueError("Unknown Env")

        # If turn_idx is even -> Player 0 acts.
        # If turn_idx is odd  -> Player 1 acts.
        is_p0_turn = bool(turn_id == 0)

        # 2. Create ONLY the relevant observation for the acting player
        obs_list = list(state)
        
        if is_p0_turn:
            if isinstance(env, DecPOMDP):
                obs_list[0] = -1
            else:
                obs_list[0] = -1
                obs_list[1] = -1
        else:
            if isinstance(env, DecPOMDP):
                obs_list[1] = -1
            else:
                obs_list[2] = -1
                obs_list[3] = -1

        # Add to the set
        obs_list = tuple(obs_list)
        total_ = tuple([obs_list, done, turn_id, reward])
        if obs_list not in unique_observations:
            unique_observations.add(obs_list)
            observations.append(total_)
    return tuple(observations), all_states


GAMES = ["A", "B", "C", "D", "E", "F", "G"]


__all__ = [
    "GameNames", "PAYOFFS","OPTIMAL_RETURNS",
    "statetype", "Settings", "Game",
    "DecPOMDP","get_game", 
    "normalize_payoffs",

    "GAMES", "get_game_Rework", "MyHanabi",
]