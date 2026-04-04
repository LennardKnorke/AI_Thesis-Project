# config.py
import itertools
import os
import json
import pandas as pd
from typing import Any
from tiny_game import GameNames, DecPOMDP

from agents import (
    BaseAgent, ModelBasedAgent, AgentList,
    IQ_Learning_Agent,
    VDN_Agent, VDN_CentralPlanner,
    PBVI_Agent, PBVI_List,
    DP_Agent, DP_List,
)

# Episode Macros
TRAINING_EPISODES_HYPERSEARCH = 1_000
TRAINING_EPISODES_FINAL = 10_000

# Directory Macros
HYPERSEARCH_RESULTS_DIR = "HyperSearchResults"
WORLD_MODELS_DIR = "Results\\WorldModels"
RESULTS_DIR = "Results"

class Experiment:
    """
    Encapsulates the configuration and instantiation logic for a specific Agent type.
    """
    def __init__(self, 
                 name: str,
                 agent_class: type[BaseAgent],
                 param_list: list[dict[str, Any]],
                 list_class: type[AgentList] = AgentList):
        """
        Args:
            name: Name of the experiment (used for folder naming).
            agent_class: The class of the individual agent (e.g., Independent_RL_Agent).
            param_list: List of dictionaries containing hyperparameters to search.
            list_class: The container class. Use AgentList for decentralized, 
                        or specific classes (VDN_AgentList) for centralized.
        """
        self.name = name
        self.agent_class = agent_class
        self.param_list = param_list
        self.list_class = list_class

    def make_agents(self,
                    env : DecPOMDP,
                    params: dict[str, Any]) -> AgentList:
        """
        Factory method to instantiate the agents for a specific environment and parameter set.
        """
        run_args = params.copy()
        run_args['num_cards'] = env.num_cards
        run_args['num_actions'] = env.num_actions
        run_args['env'] = env

        if self.list_class is AgentList:
            # Decentralized: two independent agent instances
            agents = [
                self.agent_class(agent_id=0, **run_args),
                self.agent_class(agent_id=1, **run_args),
            ]
            return AgentList(agents)
        else:
            # Centralized list class (VDN_CentralPlanner, PBVI_List, DP_List)
            # — creates its own child agents internally
            return self.list_class(**run_args)

    @property
    def is_model_based(self) -> bool:
        """Helper to determine if this experiment requires Planning loops."""
        return issubclass(self.agent_class, ModelBasedAgent)


def generate_param_grid(grid_dict):
    """Helper to create list of all combinations from a dictionary of lists."""
    keys, values = zip(*grid_dict.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

# DTDE_QSarsa_MF
iql_rid = {
    'lr': [0.01, 0.001],
    'gamma': [0.99],
    'batch_size': [32, 64, 128],
    'updates_per_train' : [1, 3],
    'buffer_size': [250, 500, 1_000],
    'epsilon_start': [1.0],
    'epsilon_min': [0.05, 0.1],
    'epsilon_decay': [0.999, 0.9995, 0.9999]
}
iql_params = generate_param_grid(iql_rid)

# CTDE_VDN_MF
vdn_grid = {
    "lr": [0.01, 0.001],
    'gamma': [0.99],
    "batch_size": [32, 64, 128],
    'updates_per_train' : [1, 3],
    'buffer_size': [250, 500, 1_000],
    "epsilon_start": [1.0],
    'epsilon_min': [0.05, 0.1],
    "epsilon_decay": [0.999, 0.9995, 0.9999]
}
vdn_params = generate_param_grid(vdn_grid)

# PBVI
PBVI_params = [
    {"max_iterations": 1, "attempts": 3},
    {"max_iterations": 5, "attempts": 3},
    {"max_iterations" : 10, "attempts" : 3},
    {"max_iterations" : 50, "attempts" : 3}
]

# MA-Belief-DP
DP_params = [
    {"max_iterations": 1, "attempts": 3},
    {"max_iterations": 5, "attempts": 3},
    {"max_iterations" : 10, "attempts" : 3},
    {"max_iterations" : 50, "attempts" : 3}
]


# Define Experiments List
BASELINE_EXPERIMENTS = [
    Experiment(
        name="IQ Learning",
        agent_class=IQ_Learning_Agent,
        param_list=iql_params,
        list_class=AgentList
    ),
    Experiment(
        name="VDN",
        agent_class=VDN_Agent,
        param_list=vdn_params,
        list_class=VDN_CentralPlanner
    ),
    Experiment(
        name="PBVI",
        agent_class=PBVI_Agent,
        param_list=PBVI_params,
        list_class=PBVI_List
    ),
    Experiment(
        name="MA Belief DP",
        agent_class=DP_Agent,
        param_list=DP_params,
        list_class=DP_List
    ),
]

NUM_AGENT_TYPES = len(BASELINE_EXPERIMENTS)

def load_best_params(agent_name : str):
    agent_name_str = agent_name.replace(" ", "_")
    best_params_path = os.path.join(RESULTS_DIR, agent_name_str, "best_params.json")

    params_list = []

    with open(best_params_path) as f:
        data = json.load(f)
        params_list.append(data)

    return params_list


def load_best_baselinesagents():
    experiments = [
        Experiment(
            name="IQ Learning",
            agent_class=IQ_Learning_Agent,
            param_list=load_best_params("IQ Learning"),
            list_class=AgentList
        ),
        Experiment(
            name="VDN",
            agent_class=VDN_Agent,
            param_list=load_best_params("VDN"),
            list_class=VDN_CentralPlanner
        ),
        Experiment(
            name="PBVI",
            agent_class=PBVI_Agent,
            param_list=load_best_params("PBVI"),
            list_class=PBVI_List
        ),
        Experiment(
            name="MA Belief DP",
            agent_class=DP_Agent,
            param_list=load_best_params("MA Belief DP"),
            list_class=DP_List
        ),
    ]
    return experiments


# --- WORLD MODEL PARAMETERS ---
tom_worldmodel_grid = {
    "char_dim": [16, 32],
    "mental_dim": [16, 32],
    "batch_size": [16, 32],
    "epochs": [20, 50, 100],
    "optimizer": ["Adam", "RMSprop"],
    "lr": [0.001, 0.01]
}
tom_worldmodel_params : list[dict[str, float | str | int]]= generate_param_grid(tom_worldmodel_grid)


# --- THEORY OF MIND AGENT HYPERPARAMETERS ---
ToM_PBVI_params = [
    {"max_iterations": 1},
    {"max_iterations": 5},
]