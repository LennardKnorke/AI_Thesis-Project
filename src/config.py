# config.py
import itertools
import pandas as pd
from typing import List, Dict, Type, Union, Any

from tiny_game import GameNames, DecPOMDP

from agents import *

# Episode Macros
TRAINING_EPISODES_HYPERSEARCH = 10_000
TRAINING_EPISODES_FINAL = 100_000


class Experiment:
    """
    Encapsulates the configuration and instantiation logic for a specific Agent type.
    """
    def __init__(self, 
                 name: str,
                 agent_class: Type[BaseAgent],
                 param_list: List[Dict[str, Any]],
                 list_class: Type[AgentList] = AgentList):
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
                    params: Dict[str, Any]) -> AgentList:
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

        if is_model_based_agent or is_model_based_list:
            run_args['env'] = env

        # Finally Set Up Agents
        if self.list_class is AgentList:
            # --- DECENTRALIZED (Independent) ---
            agents = [
                self.agent_class(**run_args),
                self.agent_class(**run_args)
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
DTDE_QSarsa_MF_grid = {
    'lr': [0.1, 0.01, 0.001],
    'gamma': [0.99],
    'batch_size': [32, 64],
    'buffer_size': [100, 500, 1_000],
    'epsilon_start': [1.0],
    'epsilon_min': [0.05, 0.1],
    'epsilon_decay': [0.999, 0.9995, 0.9999]
}
DTDE_QSarsa_MF_params = generate_param_grid(DTDE_QSarsa_MF_grid)

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
    {"max_iterations": 1, "partner_optimality": 1.0},
    {"max_iterations": 10, "partner_optimality": 1.0},
    {"max_iterations": 100, "partner_optimality": 1.0},
    {"max_iterations": 500, "partner_optimality": 1.0},
    {"max_iterations": 1000, "partner_optimality": 1.0},
]

# CTDE_BI_MB
CTDE_BI_MB_params = [
    {"max_iterations": 1}, 
    {"max_iterations": 10}, 
    {"max_iterations": 100},
    {"max_iterations": 500},
    {"max_iterations": 1000}

] 




# Define Experiments List
BASELINE_EXPERIMENTS = [
    Experiment(
        name="DTDE QSarsa",
        agent_class=DTDE_QSarsa_MF_Agent,
        param_list=DTDE_QSarsa_MF_params,
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

# --- WORLD MODEL PARAMETERS ---
tom_worldmodel_grid = {
    "char_dim": [16, 32],
    "mental_dim": [16, 32],
    "batch_size": [16, 32],
    "epochs": [20, 50, 100],
    "optimizer": ["Adam", "RMSprop"],
    "lr": [0.001, 0.01]
}
tom_worldmodel_params : List[Dict[str, Union[float, str, int]]]= generate_param_grid(tom_worldmodel_grid)


# --- THEORY OF MIND AGENT HYPERPARAMETERS ---
# Parameters for DTDE_ToMBI_Agent itself
DTDE_ToMBI_grid = {
    'max_iterations': [1], # Backward Induction is a single pass. Max_iterations here refers to the planning loop iterations.
    'convergence_threshold': [1e-5], # How much the Q-values need to change to consider converged
    'attempts': [1, 5] # Multiple random restarts for BI (optional, but can help against local optima if BI were iterative)
}
DTDE_ToMBI_params = generate_param_grid(DTDE_ToMBI_grid)


# --- THEORY OF MIND EXPERIMENT ---
TOM_AGENT_EXPERIMENT = Experiment(
    name="DTDE ToMBI", # Name for results folder
    agent_class=DTDE_ToMBI_Agent,
    param_list=DTDE_ToMBI_params,
    list_class=AgentList # ToMBI agents are decentralized
)