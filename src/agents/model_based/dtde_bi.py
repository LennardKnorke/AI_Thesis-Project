# agents/model_based/dtde_bi.py
import random
import numpy as np
import os
import pickle
from collections import defaultdict

from ..base_agent import ModelBasedAgent
from tiny_game import (
    DecPOMDP, MyHanabi, Game,
    get_all_possible_histories, get_all_possible_states
)


class DTDE_BI_MB_Agent(ModelBasedAgent):
    """
    Decentralized Backward Induction Agent.
    Solves the game in reverse order (Turn 2 -> Turn 1).
    Assumes the partner is Rational (maximizes their own expected return).
    """
    def __init__(
        self, 
        env: Game,
        num_cards: int, 
        num_actions: int,
        agent_id : int,
        gamma : float = 0.99,
        *args, **kwargs
    ):
        super().__init__(env, num_cards, num_actions)
        self.agent_id = agent_id
        self.gamma = 0.99

        # Cache observations for planning
        self.legal_actions_cache : dict[tuple, tuple]= {}
        self.worlds_cache: dict[tuple, list[tuple]] = {}
        
        # Tables necessary
        self.policy: dict[tuple, int] = {}
        self.v_values: dict[tuple, float] = defaultdict(float)        

        self.partner_policy: dict = {}   # set by AgentList before each planning sweep

        self._init_tables()
        return

    def _init_tables(self):
        for history, done, turn_id, reward in self.all_private_histories:
            if done:
                self.v_values[history] = reward
            elif turn_id == self.agent_id:
                self.v_values[history] = 0.0
                if self.is_decpomdp:
                    possible_actions = tuple(range(self.num_actions))
                else:
                    _, possible_actions = self.env.num_legal_actions(history)
                self.legal_actions_cache[history] = possible_actions
                self.policy[history] = random.choice(possible_actions)
            # Partner non-terminal turns: no policy or V-value needed
        return

    def _get_consistent_worlds(self, obs: tuple) -> list[tuple]:
        if obs in self.worlds_cache:
            return self.worlds_cache[obs]

        consistent = []
        # Determine which part of the history is masked
        if self.is_decpomdp:
            # obs format: (p0_card? or -1, p1_card? or -1, actions...)
            deal_obs = obs[:2]
            hist_obs = obs[2:]
            deal_len = 2
        else:  # MyHanabi
            deal_obs = obs[:4]
            hist_obs = obs[4:]
            deal_len = 4

        for deal in self.env.start_states():
            # Check deal compatibility
            match = True
            for i in range(deal_len):
                if deal_obs[i] != -1 and deal_obs[i] != deal[i]:
                    match = False
                    break
            if not match:
                continue

            # Replay the public actions to see if they are legal and produce the observed private info
            self.env.reset(list(deal))
            legal_so_far = True
            for t, event in enumerate(hist_obs):
                if self.is_decpomdp:
                    # event is an action
                    if self.env.is_terminal():
                        legal_so_far = False
                        break
                    # Check legality (in DecPOMDP all actions are always legal)
                    self.env.step(event)
                else:  # MyHanabi: event is (action, observed_card)
                    action, obs_card = event
                    legal_mask, _ = self.env.num_legal_actions()
                    if legal_mask[action] == 0:
                        legal_so_far = False
                        break
                    self.env.step(action)
                    # After step, check that the observed card matches what was actually revealed
                    if self.env.history[-1][1] != obs_card:
                        legal_so_far = False
                        break
            if legal_so_far:
                consistent.append(tuple(list(deal) + list(hist_obs)))

        self.worlds_cache[obs] = consistent
        return consistent

    def train(self) -> float:
        """
        Executes Backward Induction (Single Pass).
        """
        max_delta = 0.0
        
        # Sort observation longest first
        for obs, done, turn_id, _ in self.all_private_histories:
            if done or turn_id != self.agent_id:
                continue

            old_val = self.v_values[obs]
            new_val, best_act = self._evaluate_observation(obs)
            self.v_values[obs] = new_val
            self.policy[obs] = best_act
            max_delta = max(max_delta, abs(old_val - new_val))
        return max_delta

    def _evaluate_observation(self, obs: tuple) -> tuple[float, int]:
        """Compute the value and best action for a given private observation."""
        legal_actions = self.legal_actions_cache[obs]
        worlds = self._get_consistent_worlds(obs)

        best_value = -float('inf')
        best_action = legal_actions[0]  # fallback

        for action in legal_actions:
            total = 0.0
            count = 0
            for full_state in worlds:
                # Reset to this world and advance to the current point
                self.env.reset(list(full_state))
                try:
                    self.env.step(action)
                except ValueError:
                    continue  # action illegal in this world
                count += 1

                if self.env.is_terminal():
                    total += self.env.payoff()
                else:
                    # It's now the partner's turn — propagate through it
                    next_full = tuple(self.env.history)
                    total += self.gamma * self._get_partner_value(next_full)

            if count > 0:
                avg = total / count
                if avg > best_value:
                    best_value = avg
                    best_action = action

        # If no world allowed the action (should not happen for a reachable obs), keep previous value
        if best_value == -float('inf'):
            best_value = self.v_values[obs]
        return best_value, best_action

    def _get_partner_value(self, full_state: tuple) -> float:
        """
        Expected value through the partner's turn.

        IBR mode  (partner_policy is set): use the partner's current planned
        action for their private observation — one deterministic rollout.
        Uniform mode (first iteration / fallback): average over all legal
        partner actions with equal weight.
        """
        if self.is_decpomdp:
            partner_legal = list(range(self.num_actions))
        else:
            _, partner_legal = self.env.num_legal_actions(full_state)

        # --- IBR branch ---
        if self.partner_policy:
            partner_obs = self._mask_state(full_state)   # partner's private observation
            p_action = self.partner_policy.get(partner_obs)
            if p_action is not None and p_action in partner_legal:
                self.env.reset(list(full_state))
                try:
                    self.env.step(p_action)
                except ValueError:
                    pass   # illegal in this world — fall through to uniform
                else:
                    if self.env.is_terminal():
                        return self.env.payoff()
                    next_full = tuple(self.env.history)
                    next_obs  = self._mask_state(next_full)
                    return self.gamma * self.v_values.get(next_obs, 0.0)

        # --- Uniform fallback (first iteration or missing obs) ---
        total = 0.0
        count = 0
        for p_action in partner_legal:
            self.env.reset(list(full_state))
            try:
                self.env.step(p_action)
            except ValueError:
                continue
            count += 1
            if self.env.is_terminal():
                total += self.env.payoff()
            else:
                next_full = tuple(self.env.history)
                next_obs  = self._mask_state(next_full)
                total += self.gamma * self.v_values.get(next_obs, 0.0)
        return total / count if count > 0 else 0.0

    def act(self, input_state: tuple, exploit: bool = False, *args, **kwargs) -> int:
        action = self.policy[input_state]
        return action

    def save(self, filepath: str):
        data = {"policy": self.policy, "v_values": dict(self.v_values)}
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        return
    
    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        if os.path.getsize(filepath) == 0:
            raise ValueError("Q-table file is empty")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            raise ValueError("Loaded file does not contain a valid dictionary.")
        
        # Reconstruct defaultdict
        self.policy.update(data['policy'])
        self.v_values.update(data['v_values'])
        return
    
    def set_partner_policy(self, policy: dict) -> None:
        """Called by AgentList before each planning sweep to enable IBR."""
        self.partner_policy = policy

    def save_transition(self, *args): pass
    def reset(self): pass


    def _mask_state(self, state: tuple) -> tuple:
        """
        Converts a full ground-truth state/history into the observation 
        seen by the player whose turn it IS.
        """
        s_list = list(state)
        if self.is_decpomdp:
            num_actions = len(state) - 2
            p0_turn = (num_actions % 2 == 0)
            if p0_turn:
                s_list[0] = -1
            else:
                s_list[1] = -1
        else:
            # history: [c0a, c0b, c1a, c1b, a1, a2...]
            num_actions = len(state) - 4
            p0_turn = (num_actions % 2 == 0)
            
            if p0_turn:
                s_list[0] = -1; s_list[1] = -1
            else:
                s_list[2] = -1; s_list[3] = -1
        return tuple(s_list)