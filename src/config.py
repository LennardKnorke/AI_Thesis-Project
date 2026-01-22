# config.py
import itertools
import pandas as pd
from tiny_game import GameNames


# Episode Macros
TRAINING_EPISODES_HYPERSEARCH = 25_000
TRAINING_EPISODES_FINAL = 100_000


def generate_param_grid(grid_dict):
    """Helper to create list of all combinations from a dictionary of lists."""
    keys, values = zip(*grid_dict.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

# IL - RL
iq_grid = {
    'lr': [0.1, 0.01, 0.001],
    'gamma': [0.99],
    'batch_size': [32, 64],
    'buffer_size': [500, 1_000],
    'updates_per_train': [1, 5],
    'epsilon_start': [1.0],
    'epsilon_min': [0.05, 0.1],
    'epsilon_decay': [0.999, 0.9995, 0.9999]
}
modelfree_IQ_Learner_paramList = generate_param_grid(iq_grid)

# IL - VI
dtde_vi_grid = {
    'convergence_threshold': [1e-5],
    'max_iterations': [10, 20, 50],
    'partner_epsilon' : [1.0]
}
modelbased_IQ_Learner_paramList = generate_param_grid(dtde_vi_grid)

# VDN - RL
vdn_rl_gri = {
    "lr": [0.1, 0.01, 0.001],
    'gamma': [0.99],
    "batch_size": [32, 64],
    'buffer_size': [500, 1_000],
    "updates_per_train": [1, 5],
    "epsilon_start": [1.0],
    'epsilon_min': [0.05, 0.1],
    "epsilon_decay": [0.999, 0.9995, 0.9999]
}
vdn_rl_paramList = generate_param_grid(vdn_rl_gri)

# VDN - VI
vdn_vi_grid = {
    'convergence_threshold': [1e-5],
    'max_iterations': [20, 10, 50],
}
vdn_vi_paramList = generate_param_grid(vdn_vi_grid)