# config.py
import itertools
import os
import json
import pandas as pd
from typing import Any
from tiny_game import GameNames, DecPOMDP

from agents import *

# Episode Macros
TRAINING_EPISODES_HYPERSEARCH = 1_000
TRAINING_EPISODES_FINAL = 100_000

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
        Automatically handles Model-Based vs Model-Free requirements (passing 'env').
        """
        # 1. Prepare base arguments
        run_args = params.copy()
        run_args['num_cards'] = env.num_cards
        run_args['num_actions'] = env.num_actions

        # 2. Determine if we need to inject 'env'
        is_model_based_agent = issubclass(self.agent_class, ModelBasedAgent)
        
        # Heuristic check for List (since VDN lists might track the env)
        # Assuming VDN VI/BI lists act as the primary model holder
        list_name = self.list_class.__name__
        is_model_based_list = "VI" in list_name or "BI" in list_name

        run_args['env'] = env

        # Finally Set Up Agents
        if self.list_class is AgentList:
            # --- DECENTRALIZED (Independent) ---
            agents = [
                self.agent_class(agent_id = 0,**run_args),
                self.agent_class(agent_id = 1, **run_args)
            ]
            return AgentList(agents)
        else:
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
DTDE_QLearning_MF_grid = {
    'lr': [0.1, 0.01, 0.001],
    'gamma': [0.99],
    'batch_size': [32, 64],
    'buffer_size': [100, 500, 1_000],
    'epsilon_start': [1.0],
    'epsilon_min': [0.05, 0.1],
    'epsilon_decay': [0.999, 0.9995, 0.9999]
}
DTDE_QLearning_MF_params = generate_param_grid(DTDE_QLearning_MF_grid)

# CTDE_VDN_MF
CTDE_VDN_MF_grid = {
    "lr": [0.1, 0.01, 0.001],
    'gamma': [0.99],
    "batch_size": [32, 64],
    'buffer_size': [100, 500, 1_000],
    "epsilon_start": [1.0],
    'epsilon_min': [0.05, 0.1],
    "epsilon_decay": [0.999, 0.9995, 0.9999]
}
CTDE_VDN_MF_params = generate_param_grid(CTDE_VDN_MF_grid)

# DTDE_BI_MB
DTDE_BI_MB_params = [
    {"max_iterations": 1}, 
    {"max_iterations": 5},
] 

# CTDE_BI_MB
CTDE_BI_MB_params = [
    {"max_iterations": 1}, 
    {"max_iterations": 5},
] 


# Define Experiments List
BASELINE_EXPERIMENTS = [
    Experiment(
        name="DTDE QLearning",
        agent_class=DTDE_QLearning_MF_Agent,
        param_list=DTDE_QLearning_MF_params,
        list_class=AgentList
    ),
    Experiment(
        name="CTDE VDN",
        agent_class=CTDE_VDN_MF_Agent,
        param_list=CTDE_VDN_MF_params,
        list_class=CTDE_VDN_MF_List
    ),
    Experiment(
        name="DTDE BI",
        agent_class=DTDE_BI_MB_Agent,
        param_list=DTDE_BI_MB_params,
        list_class=AgentList
    ),
    Experiment(
        name="CTDE BI",
        agent_class=CTDE_BI_MB_Agent, # Or appropriate executor
        param_list=CTDE_BI_MB_params,
        list_class=CTDE_BI_MB_List
    )
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
            name="DTDE QLearning",
            agent_class=DTDE_QLearning_MF_Agent,
            param_list=load_best_params("DTDE QLearning"),
            list_class=AgentList
        ),
        Experiment(
            name="CTDE VDN",
            agent_class=CTDE_VDN_MF_Agent,
            param_list=load_best_params("CTDE VDN"),
            list_class=CTDE_VDN_MF_List
        ),
        Experiment(
            name="DTDE BI",
            agent_class=DTDE_BI_MB_Agent,
            param_list=load_best_params("DTDE BI"),
            list_class=AgentList
        ),
        Experiment(
            name="CTDE BI",
            agent_class=CTDE_BI_MB_Agent,
            param_list=load_best_params("CTDE BI"),
            list_class=CTDE_BI_MB_List
        )
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
DTDE_ToMBI_params = [
    {"max_iterations": 1}, 
    {"max_iterations": 5},
] 