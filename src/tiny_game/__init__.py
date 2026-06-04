# tiny_game/__init__.py
# Wrapper around tiny-hanabi providing shared game tables loaded at import time.

from abc import ABC, abstractmethod
from collections import deque
import numpy as np
from tqdm import tqdm
import torch

from tiny_hanabi.game.payoff_matrices import GameNames, PAYOFFS, OPTIMAL_RETURNS
from tiny_hanabi.game.settings import statetype, Settings, Game, DecPOMDP
from tiny_hanabi.game.assemblers import get_game, normalize_payoffs

from .my_hanabi import MyHanabi

_reward        = float
_terminal      = bool
_turn_id       = int
_actions       = tuple[int, ...]
_state         = tuple[int, ...]
_state_summary = tuple[_state, _actions, _terminal, _turn_id, _reward]

GAMES = ["A", "B", "C", "D", "E", "F", "G"]
ENVIRONMENTS      : dict[str, Game]                              = {}
STATES            : dict[str, _state_summary]                    = {}
PRIV_HISTORIES    : dict[str, _state_summary]                    = {}
STATE_TRANSITIONS : dict[tuple[_state, _actions], tuple[_state, ...] | None] = {g: {} for g in GAMES}
CONSISTENT_WORLDS : dict[str, dict[_state, tuple[_state]]]       = {g: {} for g in GAMES}


def mask_state(state: tuple, turn_id: int, env: Game):
    state_copy = list(state)
    if isinstance(env, DecPOMDP):
        state_copy[0 if turn_id == 0 else 1] = -1
    else:
        if turn_id == 0:
            state_copy[0] = state_copy[1] = -1
        else:
            state_copy[2] = state_copy[3] = -1
    return tuple(state_copy)


def get_game_Rework(gamename: GameNames, normalize: bool = False):
    if gamename.value == "G":
        return MyHanabi(normalize=normalize)
    return get_game(gamename=gamename, setting=Settings.decpomdp, normalize=normalize)


def get_all_possible_states(env: Game) -> tuple:
    all_states       = []
    processing_queue = deque(env.start_states())

    while processing_queue:
        current_state = processing_queue.popleft()
        env.reset(list(current_state))

        terminal = env.is_terminal()
        turn_id  = 0 if len(current_state) % 2 == 0 else 1

        if not terminal:
            if isinstance(env, DecPOMDP):
                valid_next_actions = range(env.num_actions)
            elif isinstance(env, MyHanabi):
                _, valid_next_actions = env.num_legal_actions()
            else:
                raise ValueError("Unknown Environment Type")
            valid_next_actions = tuple(valid_next_actions)
        else:
            valid_next_actions = ()

        all_states.append(_state_summary((
            _state(current_state),
            _actions(valid_next_actions),
            _terminal(terminal),
            _turn_id(turn_id),
            _reward(env.payoff()),
        )))

        if len(current_state) >= env.horizon:
            continue

        for action in valid_next_actions:
            env.reset(list(current_state))
            env.step(action)
            processing_queue.append(tuple(env.context()))

    return tuple(all_states)


def get_all_possible_histories(env: Game):
    all_states          = get_all_possible_states(env=env)
    observations        = []
    unique_observations = set()

    for state, actions, done, turn_id, reward in all_states:
        obs_list = list(state)
        if turn_id == 0:
            if isinstance(env, DecPOMDP):
                obs_list[0] = -1
            else:
                obs_list[0] = obs_list[1] = -1
        else:
            if isinstance(env, DecPOMDP):
                obs_list[1] = -1
            else:
                obs_list[2] = obs_list[3] = -1

        obs_list = tuple(obs_list)
        if obs_list not in unique_observations:
            unique_observations.add(obs_list)
            observations.append(tuple([obs_list, actions, done, turn_id, reward]))

    return tuple(observations), all_states


# --- Module-level table initialisation ---

pbar = tqdm(GAMES, desc="Setting up Games", leave=False)
for _g in pbar:
    pbar.set_postfix({"G": _g})

    _env: Game = get_game_Rework(GameNames(_g))
    ENVIRONMENTS[_g] = _env

    _priv_histories, _states = get_all_possible_histories(_env)
    STATES[_g]         = _states
    PRIV_HISTORIES[_g] = _priv_histories

    # State-action transition table
    for state_sum in STATES[_g]:
        state, actions, terminal, turn_id, reward = state_sum
        if terminal:
            STATE_TRANSITIONS[state] = None
            continue
        for a in actions:
            _env.reset(list(state))
            _env.step(a)
            STATE_TRANSITIONS[_g][(state, a)] = tuple(_env.history)

    # Consistent-worlds lookup: private observation → tuple of joint states
    # Built via a mask→[full_state] dict for O(1) lookup per history.
    mask_lookup: dict = {}

    def add_mappings(full_state):
        if isinstance(_env, DecPOMDP):
            masked0 = (-1, full_state[1]) + full_state[2:]
            masked1 = (full_state[0], -1) + full_state[2:]
        else:
            masked0 = (-1, -1, full_state[2], full_state[3]) + full_state[4:]
            masked1 = (full_state[0], full_state[1], -1, -1)  + full_state[4:]
        mask_lookup.setdefault(masked0, []).append(full_state)
        mask_lookup.setdefault(masked1, []).append(full_state)

    for state_sum in STATES[_g]:
        add_mappings(state_sum[0])

    for priv_h_sum in PRIV_HISTORIES[_g]:
        priv_h = priv_h_sum[0]
        CONSISTENT_WORLDS[_g][priv_h] = tuple(mask_lookup.get(priv_h, []))

    del mask_lookup

    # --- Implementation kept for reference ---
    # pbar2 = tqdm(PRIV_HISTORIES[_g], leave=False)
    # for priv_h_sum in pbar2:
    #     priv_h = priv_h_sum[0]
    #     partner_hand_cutoff_idx = 1 if isinstance(_env, DecPOMDP) else 2
    #     actions_cutoff_idx = 2 if isinstance(_env, DecPOMDP) else 4
    #     consistent_list = []
    #     pbar3 = tqdm(STATES[_g])
    #     for state_summary in pbar3:
    #         state = state_summary[0]
    #         if state not in consistent_list and len(state) == len(priv_h) and state[actions_cutoff_idx:] == priv_h[actions_cutoff_idx:]:
    #             if isinstance(_env, DecPOMDP):
    #                 if priv_h[0] == -1 and priv_h[1] == state[1]:
    #                     consistent_list.append(state)
    #                 elif priv_h[1] == -1 and priv_h[0] == state[0]:
    #                     consistent_list.append(state)
    #             else:
    #                 if priv_h[0] == -1 and priv_h[2] == state[2] and priv_h[3] == state[3]:
    #                     consistent_list.append(state)
    #                 elif priv_h[2] == -1 and priv_h[0] == state[0] and priv_h[1] == state[1]:
    #                     consistent_list.append(state)
    #     CONSISTENT_WORLDS[_g][priv_h] = tuple(consistent_list)


__all__ = [
    # tiny-hanabi forwarding
    "GameNames", "statetype", "Settings",
    "Game", "DecPOMDP",
    "PAYOFFS", "OPTIMAL_RETURNS",
    "get_game", "normalize_payoffs",
    # own additions
    "get_all_possible_histories", "get_all_possible_states",
    "mask_state",
    "GAMES", "ENVIRONMENTS", "STATES", "PRIV_HISTORIES", "STATE_TRANSITIONS", "CONSISTENT_WORLDS",
    "get_game_Rework", "MyHanabi",
]