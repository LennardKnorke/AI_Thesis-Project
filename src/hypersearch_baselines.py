#hypersearch_baselines.py
import json
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from typing import Tuple, Dict, Union, Any, List, Optional

# Own Imports
from tiny_game import GAMES, Settings, GameNames, DecPOMDP, get_game
from runner import run_training
from config import (
    TRAINING_EPISODES_HYPERSEARCH,
    BASELINE_EXPERIMENTS, Experiment
)

RESULTS_DIR = "HyperSearchResults/"

def hypersearch_baselines() -> None:
    """
    Loop over all baseline algorithms and perform hyperparameter search training.
    """
    for exp in BASELINE_EXPERIMENTS:
        hypersearch_algorithm(exp)
    return


def hypersearch_algorithm(exp: Experiment,*args, **kwargs) -> None:
    """
    Train a specific baseline algorithm and do hyperparameter search.
    """
    # 1. Create Directory Structure
    # Structure: HyperSearchResults / {Agent_Name}
    results_dir = os.path.join(RESULTS_DIR, exp.name.replace(" ", "_"))
    os.makedirs(results_dir, exist_ok=True)

    # START - LOOP OVER PARAM SETS
    print(f"\nHyperparameter Search for {exp.name}")
    pbar = tqdm(range(len(exp.param_list)))
    
    for idx in pbar:
        params : Dict[str, Any] = exp.param_list[idx]

        results_file = os.path.join(results_dir, f"{idx}_results.csv")
        
        # Check if completed
        if results_do_exists(results_path=results_file, **params):
            continue
        results_cache = {}

        # START - TRAIN PARAMS ON ALL GAMES
        for game_name in GAMES:
            # Set up Game Instance
            ENV = get_game(GameNames(game_name), Settings.decpomdp, normalize=False)

            # Set up Agents
            AGENTS = exp.make_agents(ENV, params)

            # Update kwargs for search
            run_kwargs = params.copy()
            
            # Handle Model-Based vs Model-Free arguments
            if exp.is_model_based:
                if 'iterations' in run_kwargs:
                    run_kwargs['max_iterations'] = run_kwargs.pop('iterations')
            else:
                run_kwargs['train_episodes'] = TRAINING_EPISODES_HYPERSEARCH
            
            # Pass metadata to runner
            run_kwargs['pbar'] = pbar
            run_kwargs['game_name'] = game_name
            
            # Run Training
            game_rewards, game_losses, _ = run_training(
                env=ENV,
                agents=AGENTS,
                **run_kwargs
            )

            # Update Results in cache
            results_cache[f"reward_{game_name}"] = game_rewards
            results_cache[f"loss_{game_name}"] = game_losses
        
        # END - TRAIN PARAMS ON ALL GAMES
        
        # Save Combined CSV
        results_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results_cache.items()]))
        results_df.index.name = "episode"
        results_df.reset_index(inplace=True)
        results_df.to_csv(results_file, index=False)

        # Save params used
        params_filepath = os.path.join(results_dir, f"{idx}_params.json")
        with open(params_filepath, 'w') as f:
            json.dump(params, f, indent=4)
            
    return


def results_do_exists(results_path: str, **params) -> bool:
    """
    Checks if results exist and if the parameters match.
    """
    json_path = results_path.replace("_results.csv", "_params.json")
    if not os.path.exists(results_path) or not os.path.exists(json_path):
        return False

    try:
        with open(json_path, 'r') as f:
            saved_params = json.load(f)
    except json.JSONDecodeError:
        return False

    # Check for mismatches
    for key, value in saved_params.items():
        if key not in params or saved_params[key] != value:
            return False
    return True