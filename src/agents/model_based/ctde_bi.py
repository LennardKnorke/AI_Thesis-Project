# agents/model_based/ctde_bi.py (corrected version)

import random
import numpy as np
import pickle
import os
from collections import defaultdict

from ..base_agent import ModelBasedAgent, AgentList
from tiny_game import DecPOMDP, MyHanabi, Game, get_all_possible_histories


class CTDE_BI_MB_Agent(ModelBasedAgent):
    def __init__(self, env: Game, num_cards: int, num_actions: int, agent_id: int, policy: dict):
        super().__init__(env, num_cards, num_actions)
        self.policy = policy          # maps private observation -> action
        self.agent_id = agent_id
        return

    def act(self, input_state: tuple, exploit: bool = False) -> int:
        return self.policy[input_state]

    def train(self): return 0.0
    def save_transition(self, *args): pass
    def save(self, *args): pass
    def reset(self): pass


class CTDE_BI_MB_List(AgentList):
    """
    Centralized Backward Induction Planner.
    Performs value iteration on joint histories (fully observable states)
    and then extracts a decentralized policy for each private observation.
    """
    def __init__(self, env: Game, num_cards: int, num_actions: int, *args, **kwargs):
        self.env = env
        self.gamma = 0.99
        self.num_cards = num_cards
        self.num_actions = num_actions

        self.is_decpomdp = isinstance(self.env, DecPOMDP)
        self.is_myhanabi = isinstance(self.env, MyHanabi)

        # Get all possible histories (joint and private)
        priv, joint = get_all_possible_histories(env)
        # Each entry: (history_tuple, done, turn_id, reward)
        self.all_private_histories = sorted(priv, key=lambda x: len(x[0]), reverse=True)
        self.all_joint_histories = sorted(joint, key=lambda x: len(x[0]), reverse=True)
        self.legal_actions_cache = {}
        self.policy = {}   # private_hist -> action

        # Data structures for joint histories
        self.joint_turn = {}          # joint_hist -> whose turn it is (0 or 1)
        self.joint_reward = {}         # joint_hist -> terminal reward (if terminal)
        self.joint_legal_actions = {}  # joint_hist -> tuple of legal actions
        self.v_values = defaultdict(float)          # joint_hist -> value
        self.joint_optimal_action = {} # joint_hist -> optimal action for current player

        # For private observations: mapping to consistent joint histories
        self.private_to_joint = defaultdict(list)

        # Initialize structures
        self._init_structures()

        # Create the two agent instances (they share the same policy dict)
        agent_0 = CTDE_BI_MB_Agent(self.env, num_cards, num_actions, 0, self.policy)
        agent_1 = CTDE_BI_MB_Agent(self.env, num_cards, num_actions, 1, self.policy)
        super().__init__([agent_0, agent_1])

    @property
    def centralized_planning(self):
        return True

    def _init_structures(self):
        """Build dictionaries for joint histories and private-to-joint mapping."""
        for joint_hist, done, turn_id, reward in self.all_joint_histories:
            self.joint_turn[joint_hist] = turn_id
            if done:
                self.joint_reward[joint_hist] = reward
                self.v_values[joint_hist] = reward
                legal = ()
            else:
                # Compute legal actions for this joint history
                if self.is_decpomdp:
                    legal = tuple(range(self.num_actions))
                else:
                    _, legal = self.env.num_legal_actions(joint_hist)
                self.joint_legal_actions[joint_hist] = legal
                self.v_values[joint_hist] = 0.0   # initial value
                self.joint_optimal_action[joint_hist] = random.choice(legal)  # placeholder

            # Map to private observation of the player whose turn it is
            priv = self._mask_state(joint_hist, turn_id)
            self.legal_actions_cache[priv] = legal
            self.private_to_joint[priv].append(joint_hist)

    def _mask_state(self, state: tuple, turn_id: int) -> tuple:
        """
        Convert a full ground-truth state into the observation seen by the player whose turn it is.
        (Identical to the original method but now takes turn_id explicitly.)
        """
        s_list = list(state)
        if self.is_decpomdp:
            # history: [c1, c2, a1, a2, ...]
            if turn_id == 0:
                s_list[0] = -1
            else:
                s_list[1] = -1
        else:  # MyHanabi
            if turn_id == 0:
                s_list[0] = -1
                s_list[1] = -1
            else:
                s_list[2] = -1
                s_list[3] = -1
        return tuple(s_list)

    def train(self) -> float:
        """
        Perform one complete backward induction sweep over joint histories.
        Returns the maximum change in value (should be 0 after one correct pass).
        """
        max_delta = 0.0
        # Process joint histories from longest to shortest (excluding terminal)
        for joint_hist, done, turn_id, _ in self.all_joint_histories:
            if done:
                continue
            delta = self._optimize_joint_node(joint_hist, turn_id)
            max_delta = max(max_delta, delta)
        # After joint values are computed, derive the decentralized policy
        self._compute_private_policy()
        return max_delta

    def _optimize_joint_node(self, joint_hist: tuple, turn_id: int) -> float:
        """
        Compute the optimal value and action for a single joint history.
        Returns the absolute change in value.
        """
        legal = self.joint_legal_actions[joint_hist]
        best_val = -float('inf')
        best_act = None

        for a in legal:
            # Simulate taking this action from the joint history
            self.env.reset(list(joint_hist))
            try:
                self.env.step(a)
            except ValueError:
                continue   # action illegal (should not happen)

            if self.env.is_terminal():
                q = self.env.payoff()          # immediate reward, no discount
            else:
                next_hist = tuple(self.env.history)   # joint history after our action
                q = self.gamma * self.v_values[next_hist]

            if q > best_val:
                best_val = q
                best_act = a

        # Fallback (should not happen for a reachable history)
        if best_act is None:
            best_val = self.v_values[joint_hist]
            best_act = legal[0]

        old_val = self.v_values[joint_hist]
        self.v_values[joint_hist] = best_val
        self.joint_optimal_action[joint_hist] = best_act
        return abs(old_val - best_val)

    def _compute_private_policy(self):
        """
        Derive a decentralized policy for each private observation by averaging
        over consistent joint histories and using the computed joint values.
        """
        
        for priv, done, _, _ in self.all_private_histories:
            if done:
                # No need to store policy for terminal observations
                continue
            worlds = self.private_to_joint.get(priv, [])
            if not worlds:
                # Should not happen for reachable observations
                continue

            legal = self.legal_actions_cache.get(priv, ())  # from base class (precomputed)
            if not legal:
                continue

            best_val = -float('inf')
            best_act = legal[0]

            for a in legal:
                total = 0.0
                for joint_hist in worlds:
                    # Simulate taking action a from this world
                    self.env.reset(list(joint_hist))
                    try:
                        self.env.step(a)
                    except ValueError:
                        continue   # action illegal in this world (should not happen)
                    if self.env.is_terminal():
                        total += self.env.payoff()
                    else:
                        next_hist = tuple(self.env.history)
                        total += self.gamma * self.v_values[next_hist]
                if worlds:
                    avg = total / len(worlds)
                    if avg > best_val:
                        best_val = avg
                        best_act = a
            self.policy[priv] = best_act

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "policy": dict(self.policy),
            "v_values": dict(self.v_values),
            "joint_optimal_action": self.joint_optimal_action,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            return
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.policy.clear()
        self.policy.update(data['policy'])
        self.v_values.clear()
        self.v_values.update(data['v_values'])
        self.joint_optimal_action = data.get('joint_optimal_action', {})

    def reset(self):
        pass