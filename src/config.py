# config.py
import itertools
import os
import json
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from typing import Any

from tiny_game import GameNames, DecPOMDP

from agents import *


PLOT_DIR = "Plots"
os.makedirs(PLOT_DIR, exist_ok=True)

LAST_N = 50

AGENT_COLORS = {
    "IQL":    "#d62728",
    "VDN":    "#ff7f0e",
    "PBDP":   "#5c2d91",
    "OSarsa": "#2ca02c",
    "ToM":    "#000000",
}
GAME_COLORS = {
    "A": "#1f77b4",
    "B": "#ff7f0e",
    "C": "#2ca02c",
    "D": "#d62728",
    "E": "#9467bd",
    "F": "#8c564b",
    "G": "#e377c2",
}

plt.rcParams.update({
    "font.size":        16,
    "axes.titlesize":   18,
    "axes.labelsize":   16,
    "xtick.labelsize":  15,
    "ytick.labelsize":  15,
    "legend.fontsize":  14,
    "lines.linewidth":  3.5,
    "figure.dpi":       100,
})



# Directory Macros
HYPERSEARCH_RESULTS_DIR = "HyperSearchResults"
WORLD_MODELS_DIR        = "Results\\WorldModels"
RESULTS_DIR             = "Results"


# Episode Macros
TRAINING_EPISODES_HYPERSEARCH = 1_000
TRAINING_EPISODES_FINAL = 10_000
NUM_RUNS                = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAIN_SEED = 42
random.seed(MAIN_SEED)
NUM_AGENT_TYPES = len([IQ_Learning_Agent, VDN_Agent, PBDP_Agent, OSarsa_Agent])
BASELINES = ["IQL", "VDN", "PBDP", "OSarsa"]

def load_best_params(path : str):
    with open(path) as f:
        data = json.load(f)
    return data


def generate_param_grid(grid_dict : dict):
    """Helper to create list of all combinations from a dictionary of lists."""
    keys, values = zip(*grid_dict.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]