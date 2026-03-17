# agents/model_based/ctde_bi.py
import random
import numpy as np
import pickle
import os
from collections import defaultdict

from ..base_agent import ModelBasedAgent, AgentList, BaseAgent
from tiny_game import DecPOMDP, MyHanabi, Game, get_all_possible_histories, get_all_possible_states

class CTDE_BI_MB_Agent(ModelBasedAgent):
    """Executor shell for Centralized BI. Just holds the policy."""
    def __init__(self, env : Game, num_cards : int, num_actions : int, agent_id : int, policy: dict):
        super().__init__(env, num_cards, num_actions)
        self.policy = policy # Shared Dict
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
    Optimizes the joint policy in exactly one backwards sweep (P1 -> P0).
    """
    def __init__(self, env: Game, num_cards: int, num_actions: int, *args, **kwargs):
        self.env = env
        self.num_cards = num_cards
        self.num_actions = num_actions
        self.NULL_VALUE = -1
        
        self.is_decpomdp = isinstance(self.env, DecPOMDP)
        self.is_myhanabi = isinstance(self.env, MyHanabi)

        # Shared Policy
        self.policy: dict[tuple, int] = {}
        self.joint_policy : dict[tuple, int] = {}
        self.joint_values: dict[tuple, float] = defaultdict(float)
        
        # Caches
        self.legal_actions_cache: dict[tuple, tuple] = {}
        self.joint_legal_actions_cache : dict[tuple, tuple] = {}

        # 1. Generate State Space
        all_private_histories, all_possible_joint_histories = get_all_possible_histories(self.env)
        self.all_private_histories = sorted(all_private_histories, key = lambda x:len(x[0]), reverse=True)
        self.all_joint_histories = sorted(all_possible_joint_histories, key = lambda x:len(x[0]), reverse=True)

        self._init_tables()

        agent_0 = CTDE_BI_MB_Agent(self.env, num_cards, num_actions, 0, self.policy)
        agent_1 = CTDE_BI_MB_Agent(self.env, num_cards, num_actions, 1, self.policy)
        super().__init__([agent_0, agent_1])
        return

    def _init_tables(self):
        # Init joint tables
        for joint_history, done, _, reward in self.all_joint_histories:
            if done:
                self.joint_legal_actions_cache[joint_history] = ()
                self.joint_values[joint_history] = reward
                continue

            # Else
            if self.is_decpomdp:
                legal = tuple(range(self.num_actions))
            else:
                _, legal = self.env.num_legal_actions(joint_history)
            self.joint_legal_actions_cache[joint_history] = legal
            self.joint_policy[joint_history] = random.choice(legal)
        
        # Init private tables
        for history, done, _, reward in self.all_private_histories:
            if done:
                possible_actions = ()
            else:
                if self.is_decpomdp:
                    possible_actions = tuple(range(self.num_actions))
                else:
                    _, possible_actions = self.env.num_legal_actions(history)
                self.legal_actions_cache[history] = possible_actions
                self.policy[history] = random.choice(possible_actions)
        return
    
    def train(self) -> float:
        max_delta = 0.0

        for i, (joint_history, done, turn_id, reward) in enumerate(self.all_joint_histories):
            if done:
                continue
            delta = self._optimize_joint_node(joint_history, turn_id)
            if delta > max_delta:
                max_delta = delta

        self._extract_private_policy()
        return max_delta
    
    def _optimize_joint_node(self, joint_history : tuple, turn_id : int):
        old_val = self.joint_values[joint_history]
        legal_actions = self.joint_legal_actions_cache[joint_history]

        if not legal_actions:
            return 0
        
        best_value = -float('inf')
        best_action = legal_actions[0]

        for action in legal_actions:
            self.env.reset(list(joint_history))

            self.env.step(action)

            if self.env.is_terminal():
                value = self.env.payoff()
            else:
                next_joint = tuple(self.env.history)
                value = self.joint_values[next_joint]

            if value > best_value:
                best_value = value
                best_action = action

        self.joint_values[joint_history] = best_value
        self.joint_policy[joint_history] = best_action
        return abs(old_val - best_value)
    
    def _extract_private_policy(self):
        for joint_history, done, turn_id, reward  in self.all_joint_histories:
            if done:
                continue
            if joint_history not in self.joint_policy:
                continue
            private_history = self._mask_state(joint_history)
            self.policy[private_history] = self.joint_policy[joint_history]
        return

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "policy": dict(self.policy),
            "joint_policy": dict(self.joint_policy),
            "joint_values": dict(self.joint_values),
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _mask_state(self, state: tuple) -> tuple:
        """
        Converts a full ground-truth state into the observation 
        for the player whose turn it is.
        """
        s_list = list(state)
        
        if self.is_decpomdp:
            # history: [c1, c2, a1, a2...]
            num_actions = len(state) - 2
            p0_turn = (num_actions % 2 == 0)
            if p0_turn: s_list[0] = -1
            else:       s_list[1] = -1

        elif self.is_myhanabi:
            # history: [c0a, c0b, c1a, c1b, a1, a2...]
            num_actions = len(state) - 4
            p0_turn = (num_actions % 2 == 0)
            if p0_turn: 
                s_list[0] = -1; s_list[1] = -1
            else:       
                s_list[2] = -1; s_list[3] = -1
        
        return tuple(s_list)
            
    def reset(self):
        return

    def load(self, filepath: str):
        """
        Loads the Shared Policy.
        """
        if not os.path.exists(filepath): return
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        # overwrite, don't merge
        self.policy.clear()
        self.policy.update(data.get("policy", {}))

        self.joint_policy.clear()
        self.joint_policy.update(data.get("joint_policy", {}))

        self.joint_values.clear()
        self.joint_values.update(data.get("joint_values", {}))