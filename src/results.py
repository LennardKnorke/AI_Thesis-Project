import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys
import torch

#sys.path.append('src')

from agents import *
from tiny_game import *
from config import *
from runner import *

from train_worldmodel import setup_baseline_agents

from results_wm import run_results_wm
from results_tom import run_results_tom


def main():
    bagents = setup_baseline_agents()
    run_results_wm(bagents)
    run_results_tom(bagents)


if __name__ == "__main__":
    main()